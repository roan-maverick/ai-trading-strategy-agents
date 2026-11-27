"""Concrete providers. Raw HTTP (httpx) - no vendor SDKs, so swapping models never breaks the install."""

from __future__ import annotations

from .base import LLMError, LLMProvider


class GeminiProvider(LLMProvider):
    """Google Gemini API (default model: gemini-4-neon)."""

    name = "gemini"
    default_model = "gemini-4-neon"
    key_envs = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

    async def complete(self, system: str, user: str, meta: dict | None = None) -> str:
        base = (self.cfg.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        gen: dict = {"maxOutputTokens": self.cfg.max_output_tokens}
        if self.cfg.temperature is not None:
            gen["temperature"] = self.cfg.temperature
        if self.cfg.json_mode is not False:
            gen["responseMimeType"] = "application/json"
        if self.cfg.thinking_level:
            gen["thinkingConfig"] = {"thinkingLevel": self.cfg.thinking_level}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }
        data = await self.post(f"{base}/models/{self.model}:generateContent",
                               {"x-goog-api-key": self.api_key() or "", "Content-Type": "application/json"}, body)
        um = data.get("usageMetadata", {})
        self.usage["input_tokens"] += int(um.get("promptTokenCount", 0) or 0)
        self.usage["output_tokens"] += int(um.get("candidatesTokenCount", 0) or 0)
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            reason = data.get("promptFeedback") or data.get("candidates", [{}])[0].get("finishReason")
            raise LLMError(f"gemini returned no content ({reason})") from None
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))


class OpenAICompatibleProvider(LLMProvider):
    """Any /chat/completions endpoint: OpenAI, OpenRouter, Groq, DeepSeek, xAI, Together, Vercel AI
    Gateway, Ollama, LM Studio, vLLM..."""

    name = "openai"
    default_base = "https://api.openai.com/v1"
    key_envs = ("OPENAI_API_KEY",)
    json_default = True

    async def complete(self, system: str, user: str, meta: dict | None = None) -> str:
        base = (self.cfg.base_url or self.default_base).rstrip("/")
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        if self.cfg.json_mode if self.cfg.json_mode is not None else self.json_default:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        key = self.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = await self.post(f"{base}/chat/completions", headers, body)
        u = data.get("usage") or {}
        self.usage["input_tokens"] += int(u.get("prompt_tokens", 0) or 0)
        self.usage["output_tokens"] += int(u.get("completion_tokens", 0) or 0)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise LLMError(f"{self.name}: unexpected response {str(data)[:300]}") from None
        if isinstance(content, list):  # some gateways return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""


def _compat(name: str, base: str, keys: tuple[str, ...], needs_key: bool = True, json_default: bool = False,
            default_model: str | None = None) -> type[OpenAICompatibleProvider]:
    return type(f"{name.title()}Provider", (OpenAICompatibleProvider,), {
        "name": name, "default_base": base, "key_envs": keys, "needs_key": needs_key,
        "json_default": json_default, "default_model": default_model,
    })


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    default_model = "claude-opus-5-5"
    key_envs = ("ANTHROPIC_API_KEY",)

    async def complete(self, system: str, user: str, meta: dict | None = None) -> str:
        base = (self.cfg.base_url or "https://api.anthropic.com").rstrip("/")
        body: dict = {
            "model": self.model, "max_tokens": self.cfg.max_output_tokens, "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.cfg.temperature is not None:
            body["temperature"] = min(self.cfg.temperature, 1.0)
        headers = {"x-api-key": self.api_key() or "", "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}
        data = await self.post(f"{base}/v1/messages", headers, body)
        u = data.get("usage") or {}
        self.usage["input_tokens"] += int(u.get("input_tokens", 0) or 0)
        self.usage["output_tokens"] += int(u.get("output_tokens", 0) or 0)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


PROVIDERS: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "openrouter": _compat("openrouter", "https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",)),
    "vercel": _compat("vercel", "https://ai-gateway.vercel.sh/v1", ("AI_GATEWAY_API_KEY",),
                      default_model="google/gemini-4-neon"),
    "groq": _compat("groq", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
    "deepseek": _compat("deepseek", "https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",)),
    "xai": _compat("xai", "https://api.x.ai/v1", ("XAI_API_KEY",)),
    "together": _compat("together", "https://api.together.xyz/v1", ("TOGETHER_API_KEY",)),
    "mistral": _compat("mistral", "https://api.mistral.ai/v1", ("MISTRAL_API_KEY",)),
    "ollama": _compat("ollama", "http://localhost:11434/v1", ("OLLAMA_API_KEY",), needs_key=False),
    "lmstudio": _compat("lmstudio", "http://localhost:1234/v1", ("LMSTUDIO_API_KEY",), needs_key=False),
    "custom": _compat("custom", "http://localhost:8000/v1", ("CUSTOM_LLM_API_KEY",), needs_key=False),
}
