"""LLM layer: one interface, any model. Add a provider by subclassing LLMProvider (see docs/PROVIDERS.md)."""

from __future__ import annotations

from ..config import LLMConfig
from .base import LLMError, LLMProvider, extract_json
from .offline import OfflineProvider
from .providers import PROVIDERS

ALL_PROVIDERS = {**PROVIDERS, "offline": OfflineProvider}


def get_provider(cfg: LLMConfig, seed: int | None = None) -> LLMProvider:
    name = (cfg.provider or "").lower().strip()
    if name not in ALL_PROVIDERS:
        raise LLMError(f"Unknown provider '{cfg.provider}'. Available: {', '.join(sorted(ALL_PROVIDERS))}")
    if name == "offline":
        return OfflineProvider(cfg, seed=seed)
    return ALL_PROVIDERS[name](cfg)


__all__ = ["ALL_PROVIDERS", "LLMError", "LLMProvider", "OfflineProvider", "extract_json", "get_provider"]
