# LLM providers

Every provider implements one method:

```python
async def complete(self, system: str, user: str, meta: dict | None = None) -> str
```

## Configure

```yaml
llm:
  provider: gemini          # see the table in the README
  model: gemini-4-neon
  temperature: 0.8
  thinking_level: medium    # Gemini 4 only (low | medium | high | xhigh)
  base_url: null            # override the endpoint (proxies, gateways, local servers)
  api_key_env: null         # read the key from a different env var
  json_mode: null           # force JSON mode on/off (null = provider default)
```

Or use flags (`atsa run --provider openrouter --model <vendor/model>`) or env vars (`ATSA_PROVIDER`, `ATSA_MODEL`).

### Local models

```bash
ollama pull <model>
atsa run --provider ollama --model <model>
```

Smaller local models work. Invalid strategies are dropped and counted, never crash the run. Bigger models produce more
valid and more creative ideas per call.

## Add a provider

```python
# src/atsa/llm/providers.py
class MyProvider(LLMProvider):
    name = "myllm"
    default_model = "my-model-1"
    key_envs = ("MYLLM_API_KEY",)

    async def complete(self, system, user, meta=None):
        data = await self.post("https://api.myllm.com/v1/generate",
                               {"Authorization": f"Bearer {self.api_key()}"},
                               {"model": self.model, "system": system, "prompt": user})
        return data["text"]

PROVIDERS["myllm"] = MyProvider
```

`self.post` handles retries with exponential backoff on 429/5xx and timeouts.
If your endpoint is OpenAI-compatible, you don't need code: use `provider: custom` with `base_url`.

## Cost control

* `swarm.agents × swarm.strategies_per_agent × swarm.generations × symbols` = strategies per cycle.
* `(agents + 1 evolver) × generations × symbols` = LLM calls per cycle (e.g. 7 × 3 × 3 = 63).
* `runtime.interval_minutes` sets how often the 24/7 loop hunts.
* Gemini 4 Neon: lower `thinking_level` to spend fewer thinking tokens per call.
