import asyncio
import json

import httpx
import pytest

from atsa.agents import Swarm, parse_strategies
from atsa.config import BacktestConfig, LLMConfig, Settings, SwarmConfig
from atsa.llm import extract_json, get_provider
from atsa.llm.base import LLMError
from atsa.pipeline import Pipeline
from atsa.store import Store

STRATS = {"strategies": [
    {"name": "Dip buyer", "thesis": "oversold bounce", "entry_long": "rsi(close,2) < 10", "exit_long": "rsi(close,2) > 60"},
    {"name": "Broken", "entry_long": "import os"},
    {"name": "Trend", "entry_long": "ema(close,20) > ema(close,100)", "stop_loss": 0.08},
]}


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"a": [1,2]} hope it helps') == {"a": [1, 2]}
    assert extract_json("[1, 2]") == [1, 2]
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_parse_strategies_keeps_valid_drops_invalid():
    valid, invalid, n = parse_strategies(json.dumps(STRATS), "tester", 2)
    assert n == 3 and len(valid) == 2 and len(invalid) == 1
    assert all(s.agent == "tester" and s.generation == 2 for s in valid)


def _mock(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_gemini_request_and_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["key"] = req.headers.get("x-goog-api-key")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True},
                                                  {"text": json.dumps(STRATS)}]}}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50}})

    cfg = LLMConfig(provider="gemini", thinking_level="low")
    p = get_provider(cfg)
    p._client = _mock(handler)
    text = asyncio.run(p.complete("sys", "user"))
    assert "models/gemini-4-neon:generateContent" in seen["url"]
    assert seen["key"] == "test-key"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["body"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert json.loads(text) == STRATS  # thought parts are dropped
    assert p.usage["input_tokens"] == 100


def test_openai_compatible_and_anthropic(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k2")

    def oa(req):
        assert req.headers["authorization"] == "Bearer k1"
        assert req.url.path.endswith("/chat/completions")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(STRATS)}}]})

    p = get_provider(LLMConfig(provider="openrouter", model="google/gemini-4-neon"))
    p._client = _mock(oa)
    assert json.loads(asyncio.run(p.complete("s", "u"))) == STRATS

    def an(req):
        assert req.headers["x-api-key"] == "k2"
        body = json.loads(req.content)
        assert body["system"] == "s" and body["max_tokens"] > 0
        return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps(STRATS)}]})

    p = get_provider(LLMConfig(provider="anthropic"))
    p._client = _mock(an)
    assert json.loads(asyncio.run(p.complete("s", "u"))) == STRATS


def test_retries_then_fails(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503, text="overloaded")

    p = get_provider(LLMConfig(provider="gemini", max_retries=1))
    p._client = _mock(handler)
    monkeypatch.setattr(asyncio, "sleep", _noop)
    with pytest.raises(LLMError):
        asyncio.run(p.complete("s", "u"))
    assert calls["n"] == 2


async def _noop(*_args):
    return None


def test_missing_key_and_unknown_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        get_provider(LLMConfig(provider="openai"))  # no model
    with pytest.raises(LLMError):
        get_provider(LLMConfig(provider="nope"))
    p = get_provider(LLMConfig(provider="openai", model="m"))
    with pytest.raises(LLMError):
        p.api_key()


def test_swarm_with_mocked_llm(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    p = get_provider(LLMConfig(provider="gemini"))
    p._client = _mock(lambda req: httpx.Response(200, json={
        "candidates": [{"content": {"parts": [{"text": json.dumps(STRATS)}]}}]}))
    swarm = Swarm(p, SwarmConfig(agents=["momentum_hunter", "my_custom_agent"], strategies_per_agent=3),
                  BacktestConfig())
    reports = asyncio.run(swarm.research({"symbol": "X", "interval": "1d"}))
    assert len(reports) == 2 and all(len(r.valid) == 2 for r in reports)


def test_full_offline_pipeline(tmp_path):
    s = Settings()
    s.llm.provider = "offline"
    s.swarm.seed = 7
    s.swarm.generations = 2
    s.data.symbols = ["DEMO-A", "DEMO-C"]
    s.runtime.db_path = str(tmp_path / "atsa.db")
    s.runtime.reports_dir = str(tmp_path / "reports")
    s.runtime.workers = 1
    pipe = Pipeline(s, quiet=True)
    cyc = asyncio.run(pipe.run_cycle())
    pipe.close()
    assert cyc.tested > 100
    assert (tmp_path / "reports" / "winners_latest.json").exists()
    payload = json.loads((tmp_path / "reports" / "winners_latest.json").read_text())
    assert payload["engine"] == "ai-trading-strategy-agents"
    st = Store(s.runtime.db_path)
    assert st.counts()["tested"] == cyc.tested
    assert st.runs()[0]["status"] == "ok"
    st.close()
    # second cycle re-validates champions and still works
    pipe = Pipeline(s, quiet=True)
    cyc2 = asyncio.run(pipe.run_cycle())
    pipe.close()
    assert cyc2.run_id == cyc.run_id + 1
