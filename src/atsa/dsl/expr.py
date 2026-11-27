"""Safe, vectorised strategy expression language.

LLMs write rules as plain readable expressions, e.g.

    rsi(close, 14) < 30 and close > sma(close, 200)
    cross_above(ema(close, 12), ema(close, 26)) and adx(14) > 20
    fv_gap(close, 60) < -2

Expressions are parsed with Python's `ast`, checked against a strict whitelist (no attribute
access, no imports, no subscripts, no lambdas) and evaluated on whole NumPy arrays. LLM output is
never executed as code. All functions are causal, so there is no look-ahead bias.

Indicator results are memoised per dataset, so thousands of strategies sharing `sma(close, 50)`
compute it once.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from ..data.base import Bars
from . import indicators as ind

MAX_EXPR_LEN = 600
MAX_CALLS = 40
MAX_PERIOD = 1000


class ExprError(ValueError):
    """Raised for invalid or unsafe strategy expressions."""


@dataclass(frozen=True)
class FnSpec:
    fn: Callable
    args: str  # one char per arg: s=series expr, i=int period, f=float constant
    doc: str
    needs_bars: bool = False


def _cross_above(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    la, lb = ind.lag(a, 1), ind.lag(b, 1)
    return (a > b) & (la <= lb)


def _cross_below(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    la, lb = ind.lag(a, 1), ind.lag(b, 1)
    return (a < b) & (la >= lb)


def _safe_log(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(x)
    return np.where(np.isfinite(out), out, np.nan)


def _bb(x, n, k, sign):
    return ind.rolling_mean(x, n) + sign * k * ind.rolling_std(x, n)


def _macd(x, f, s):
    return ind.ema(x, f) - ind.ema(x, s)


def _macd_signal(x, f, s, k):
    return ind.ema(_macd(x, f, s), k)


def _rising(x, n):
    return x > ind.lag(x, n)


def _falling(x, n):
    return x < ind.lag(x, n)


def _streak_true(cond, n):
    """True when cond has been true for the last n bars."""
    c = _as_bool(cond).astype(np.float64)
    return ind.rolling_mean(c, n) >= 1.0 - 1e-12


FUNCTIONS: dict[str, FnSpec] = {
    # moving averages / smoothing
    "sma": FnSpec(lambda x, n: ind.sma(x, n), "si", "simple moving average"),
    "ema": FnSpec(lambda x, n: ind.ema(x, n), "si", "exponential moving average"),
    "wilder": FnSpec(lambda x, n: ind.wilder_smooth(x, n), "si", "Wilder smoothing"),
    # oscillators
    "rsi": FnSpec(lambda x, n: ind.wilder_rsi(x, n), "si", "relative strength index 0-100"),
    "zscore": FnSpec(lambda x, n: ind.zscore(x, n), "si", "rolling z-score"),
    "roc": FnSpec(lambda x, n: ind.roc(x, n), "si", "rate of change (fraction)"),
    "pct_rank": FnSpec(lambda x, n: ind.rolling_pct_rank(x, n), "si", "percentile rank in window 0-1"),
    "macd": FnSpec(_macd, "sii", "MACD line ema(fast)-ema(slow)"),
    "macd_signal": FnSpec(_macd_signal, "siii", "MACD signal line"),
    "stoch_k": FnSpec(lambda b, n: ind.stoch_k(b.high, b.low, b.close, n), "i", "stochastic %K 0-100", True),
    "adx": FnSpec(lambda b, n: ind.adx(b.high, b.low, b.close, n), "i", "trend strength 0-100", True),
    # volatility / ranges
    "stdev": FnSpec(lambda x, n: ind.rolling_std(x, n), "si", "rolling standard deviation"),
    "atr": FnSpec(lambda b, n: ind.atr(b.high, b.low, b.close, n), "i", "average true range", True),
    "highest": FnSpec(lambda x, n: ind.rolling_max(x, n), "si", "rolling max (includes current bar)"),
    "lowest": FnSpec(lambda x, n: ind.rolling_min(x, n), "si", "rolling min (includes current bar)"),
    "bb_upper": FnSpec(lambda x, n, k: _bb(x, n, k, 1.0), "sif", "Bollinger upper band"),
    "bb_lower": FnSpec(lambda x, n, k: _bb(x, n, k, -1.0), "sif", "Bollinger lower band"),
    # fair value / mispricing
    "fair_value": FnSpec(lambda x, n: ind.fair_value(x, n), "si", "rolling regression fair value"),
    "fv_gap": FnSpec(lambda x, n: ind.fv_gap(x, n), "si", "mispricing z-score vs fair value"),
    "linreg_slope": FnSpec(lambda x, n: ind.rolling_linreg(x, n, True), "si", "rolling regression slope"),
    "vwap": FnSpec(lambda b, n: ind.rolling_vwap((b.high + b.low + b.close) / 3.0, b.volume, n), "i",
                   "rolling volume-weighted average price", True),
    # series helpers
    "lag": FnSpec(lambda x, n: ind.lag(np.asarray(x, float), n), "si", "value n bars ago"),
    "diff": FnSpec(lambda x, n: np.asarray(x, float) - ind.lag(np.asarray(x, float), n), "si", "x - lag(x,n)"),
    "rising": FnSpec(_rising, "si", "x > lag(x,n)"),
    "falling": FnSpec(_falling, "si", "x < lag(x,n)"),
    "streak": FnSpec(_streak_true, "si", "condition true for each of the last n bars"),
    "cross_above": FnSpec(_cross_above, "ss", "a crosses above b on this bar"),
    "cross_below": FnSpec(_cross_below, "ss", "a crosses below b on this bar"),
    "abs": FnSpec(lambda x: np.abs(x), "s", "absolute value"),
    "log": FnSpec(_safe_log, "s", "natural log"),
    "max": FnSpec(lambda a, b: np.fmax(a, b), "ss", "element-wise max"),
    "min": FnSpec(lambda a, b: np.fmin(a, b), "ss", "element-wise min"),
}

VARIABLES = {
    "open": "bar open", "high": "bar high", "low": "bar low", "close": "bar close", "volume": "bar volume",
    "hl2": "(high+low)/2", "hlc3": "(high+low+close)/3", "ret": "1-bar close-to-close return",
    "dow": "day of week 0=Mon..6=Sun", "hour": "hour of day UTC 0-23", "month": "month 1-12",
}

_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.BinOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.Call, ast.Name, ast.Load, ast.Constant,
)


def _as_bool(v) -> np.ndarray:
    a = np.asarray(v)
    if a.dtype == np.bool_:
        return a
    a = a.astype(np.float64)
    return (a != 0.0) & ~np.isnan(a)


def parse(expr: str) -> ast.Expression:
    """Parse + validate an expression (memoised). Raises ExprError with a helpful message."""
    if not isinstance(expr, str) or not expr.strip():
        raise ExprError("empty expression")
    return _parse_cached(expr)


@lru_cache(maxsize=16384)
def _parse_cached(expr: str) -> ast.Expression:
    src = expr.strip().replace("&&", " and ").replace("||", " or ")
    if len(src) > MAX_EXPR_LEN:
        raise ExprError(f"expression longer than {MAX_EXPR_LEN} chars")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"syntax error: {e.msg}") from None
    calls = 0
    func_nodes = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if id(node) in func_nodes:
            continue
        if not isinstance(node, _ALLOWED_NODES):
            raise ExprError(f"'{type(node).__name__}' is not allowed")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            raise ExprError("only numeric constants are allowed")
        if isinstance(node, ast.Name) and node.id not in VARIABLES and node.id not in ("True", "False"):
            raise ExprError(f"unknown variable '{node.id}'. Allowed: {', '.join(VARIABLES)}")
        if isinstance(node, ast.Call):
            calls += 1
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                name = getattr(node.func, "id", "?")
                raise ExprError(f"unknown function '{name}'")
            if node.keywords:
                raise ExprError("keyword arguments are not supported")
            spec = FUNCTIONS[node.func.id]
            if len(node.args) != len(spec.args):
                raise ExprError(f"{node.func.id}() takes {len(spec.args)} args ({spec.args}), got {len(node.args)}")
            for a, kind in zip(node.args, spec.args):
                if kind == "i":
                    if not (isinstance(a, ast.Constant) and isinstance(a.value, (int, float))
                            and float(a.value).is_integer()):
                        raise ExprError(f"{node.func.id}(): period must be an integer constant")
                    if not 1 <= int(a.value) <= MAX_PERIOD:
                        raise ExprError(f"{node.func.id}(): period must be in 1..{MAX_PERIOD}")
                elif kind == "f":
                    if not (isinstance(a, ast.Constant) and isinstance(a.value, (int, float))):
                        raise ExprError(f"{node.func.id}(): expected a numeric constant")
    if calls > MAX_CALLS:
        raise ExprError(f"too many function calls (>{MAX_CALLS})")
    return tree


@lru_cache(maxsize=16384)
def max_lookback(expr: str) -> int:
    """Rough warm-up estimate: sum of the periods along the deepest nesting."""
    tree = parse(expr)

    def walk(node) -> int:
        if isinstance(node, ast.Call):
            spec = FUNCTIONS[node.func.id]
            own = sum(int(a.value) for a, k in zip(node.args, spec.args) if k == "i")
            inner = max([walk(a) for a, k in zip(node.args, spec.args) if k == "s"] or [0])
            return own + inner
        return max([walk(c) for c in ast.iter_child_nodes(node)] or [0])

    return walk(tree)


class Evaluator:
    """Evaluates expressions on one dataset with an indicator cache shared by all strategies."""

    def __init__(self, bars: Bars):
        self.bars = bars
        self.n = len(bars)
        self._cache: dict[str, np.ndarray] = {}
        self._vars: dict[str, np.ndarray] = {}
        self._conditions: dict[str, np.ndarray] = {}

    def var(self, name: str) -> np.ndarray:
        if name in self._vars:
            return self._vars[name]
        b = self.bars
        if name in ("open", "high", "low", "close", "volume"):
            v = getattr(b, name)
        elif name == "hl2":
            v = (b.high + b.low) / 2.0
        elif name == "hlc3":
            v = (b.high + b.low + b.close) / 3.0
        elif name == "ret":
            v = np.full(self.n, np.nan)
            v[1:] = b.close[1:] / b.close[:-1] - 1.0
        elif name in ("dow", "hour", "month"):
            dt = b.ts.astype("datetime64[ms]")
            if name == "dow":
                v = ((dt.astype("datetime64[D]").astype(np.int64) + 3) % 7).astype(np.float64)
            elif name == "hour":
                v = ((dt.astype("datetime64[h]").astype(np.int64)) % 24).astype(np.float64)
            else:
                v = (dt.astype("datetime64[M]").astype(np.int64) % 12 + 1).astype(np.float64)
        else:
            raise ExprError(f"unknown variable '{name}'")
        self._vars[name] = v
        return v

    def eval(self, expr: str) -> np.ndarray:
        tree = parse(expr)
        out = self._eval(tree.body)
        return np.broadcast_to(np.asarray(out), (self.n,)).copy() if np.ndim(out) == 0 else out

    def condition(self, expr: str | None) -> np.ndarray | None:
        if expr is None or (isinstance(expr, str) and not expr.strip()):
            return None
        hit = self._conditions.get(expr)
        if hit is None:
            hit = np.ascontiguousarray(_as_bool(self.eval(expr)))
            if len(self._conditions) > 20000:
                self._conditions.clear()
            self._conditions[expr] = hit
        return hit

    def _eval(self, node):
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id == "True":
                return 1.0
            if node.id == "False":
                return 0.0
            return self.var(node.id)
        if isinstance(node, ast.BoolOp):
            vals = [_as_bool(self._eval(v)) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = np.logical_and(out, v) if isinstance(node.op, ast.And) else np.logical_or(out, v)
            return out
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand)
            if isinstance(node.op, ast.Not):
                return np.logical_not(_as_bool(v))
            if isinstance(node.op, ast.USub):
                return -np.asarray(v, dtype=np.float64)
            return v
        if isinstance(node, ast.BinOp):
            a = np.asarray(self._eval(node.left), dtype=np.float64)
            b = np.asarray(self._eval(node.right), dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if isinstance(node.op, ast.Add):
                    r = a + b
                elif isinstance(node.op, ast.Sub):
                    r = a - b
                elif isinstance(node.op, ast.Mult):
                    r = a * b
                else:
                    r = a / b
            if np.ndim(r):
                r = np.where(np.isfinite(r), r, np.nan)
            return r
        if isinstance(node, ast.Compare):
            left = np.asarray(self._eval(node.left), dtype=np.float64)
            result = None
            for op, comp in zip(node.ops, node.comparators):
                right = np.asarray(self._eval(comp), dtype=np.float64)
                if isinstance(op, ast.Lt):
                    r = left < right
                elif isinstance(op, ast.LtE):
                    r = left <= right
                elif isinstance(op, ast.Gt):
                    r = left > right
                elif isinstance(op, ast.GtE):
                    r = left >= right
                elif isinstance(op, ast.Eq):
                    r = np.isclose(left, right)
                else:
                    r = ~np.isclose(left, right) & ~np.isnan(left) & ~np.isnan(right)
                result = r if result is None else np.logical_and(result, r)
                left = right
            return result
        if isinstance(node, ast.Call):
            key = ast.dump(node)
            hit = self._cache.get(key)
            if hit is not None:
                return hit
            spec = FUNCTIONS[node.func.id]
            args = []
            for a, kind in zip(node.args, spec.args):
                if kind == "i":
                    args.append(int(a.value))
                elif kind == "f":
                    args.append(float(a.value))
                else:
                    v = self._eval(a)
                    if np.ndim(v) == 0:
                        v = np.full(self.n, float(v))
                    args.append(np.ascontiguousarray(v, dtype=np.float64))
            res = spec.fn(self.bars, *args) if spec.needs_bars else spec.fn(*args)
            res = np.asarray(res)
            self._cache[key] = res
            return res
        raise ExprError(f"unsupported node {type(node).__name__}")


def reference() -> str:
    """Human/LLM-readable language reference (injected into agent prompts)."""
    lines = ["VARIABLES: " + ", ".join(f"{k} ({v})" for k, v in VARIABLES.items()), "FUNCTIONS:"]
    names = {"s": "x", "i": "n", "f": "k"}
    for name, spec in FUNCTIONS.items():
        sig = ", ".join(names[c] for c in spec.args)
        lines.append(f"  {name}({sig}) - {spec.doc}")
    lines.append("OPERATORS: + - * /  < <= > >= == !=  and or not  (parentheses allowed)")
    lines.append("RULES: periods (n) must be integer literals 1..1000; no other syntax is allowed.")
    return "\n".join(lines)
