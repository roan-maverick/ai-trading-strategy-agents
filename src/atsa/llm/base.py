"""Provider-agnostic LLM interface. Every model is just `complete(system, user) -> text`."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..config import LLMConfig


class LLMError(RuntimeError):
    pass


class LLMProvider(ABC):
    name: str = "base"
    default_model: str | None = None
    key_envs: tuple[str, ...] = ()
    needs_key: bool = True

    def __init__(self, cfg: LLMConfig, client: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.model = cfg.model or self.default_model
        if not self.model:
            raise LLMError(f"Provider '{self.name}' needs a model id: set llm.model in atsa.yaml or ATSA_MODEL")
        self._client = client
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    # -- keys -------------------------------------------------------------------------------
    def api_key(self) -> str | None:
        envs = (self.cfg.api_key_env,) if self.cfg.api_key_env else self.key_envs
        for env in envs:
            if env and os.environ.get(env):
                return os.environ[env]
        if self.needs_key:
            raise LLMError(f"No API key for '{self.name}'. Set {' or '.join(e for e in envs if e)} in your env/.env")
        return None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.cfg.timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- core -------------------------------------------------------------------------------
    @abstractmethod
    async def complete(self, system: str, user: str, meta: dict | None = None) -> str:
        """Return the model's text answer."""

    async def post(self, url: str, headers: dict, body: dict) -> dict:
        """POST with exponential back-off on rate limits / transient server errors."""
        last = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = await self.client.post(url, headers=headers, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = e
            else:
                if resp.status_code < 400:
                    self.usage["calls"] += 1
                    return resp.json()
                if resp.status_code not in (408, 409, 429, 500, 502, 503, 504, 529):
                    raise LLMError(f"{self.name} HTTP {resp.status_code}: {resp.text[:500]}")
                last = LLMError(f"{self.name} HTTP {resp.status_code}: {resp.text[:200]}")
            if attempt < self.cfg.max_retries:
                await asyncio.sleep(min(30.0, (2 ** attempt) + random.random()))
        raise LLMError(f"{self.name} failed after {self.cfg.max_retries + 1} attempts: {last}")

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model reply (handles code fences and chatter)."""
    if not text:
        raise ValueError("empty model reply")
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for open_c, close_c in (("{", "}"), ("[", "]")):
            i, j = cand.find(open_c), cand.rfind(close_c)
            if i != -1 and j > i:
                try:
                    return json.loads(cand[i:j + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError("no valid JSON found in model reply")
