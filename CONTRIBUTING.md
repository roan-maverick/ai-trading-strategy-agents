# Contributing

Pull requests are welcome. The codebase is built to be extended one file at a time.

| Want to add | Where |
|---|---|
| A research agent | `src/atsa/agents/personas.py` (or just add a name in `atsa.yaml`) |
| An indicator | `src/atsa/dsl/indicators.py` + register it in `src/atsa/dsl/expr.py` |
| An LLM provider | `src/atsa/llm/providers.py` (see `docs/PROVIDERS.md`) |
| A data source | `src/atsa/data/sources.py` + `src/atsa/data/__init__.py` |
| A robustness test | `src/atsa/backtest/gate.py` + a field in `GateConfig` |

Before you open a PR:

```bash
pip install -e ".[all,dev]"
ruff check src tests
pytest
atsa demo
```

Rules of the road: indicators must be causal (no look-ahead), and LLM output must never be executed as code.
