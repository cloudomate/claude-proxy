# tf-anthropic-proxy

A tiny Go service that exposes an **Anthropic Messages API** (`/v1/messages`) and
translates to the tokenfactory **OpenAI** endpoint (`/v1/chat/completions`).

Why it exists:
- tokenfactory's native `/v1/messages` **streaming is non-compliant** — it emits
  OpenAI `chat.completion.chunk` objects instead of Anthropic SSE events, so
  Claude Code and the Anthropic SDK (`stream=true`) break.
- tokenfactory's WAF **403s** the default Go/Python `User-Agent`. This proxy
  spoofs `curl/8.4.0` upstream.

This proxy emits correct Anthropic SSE (`message_start`, `content_block_start`,
`content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`).

## Run with Docker Compose (recommended)

```bash
cp .env.example .env            # then put your UPSTREAM_API_KEY in it
docker compose up -d --build    # listens on :4000
curl localhost:4000/v1/models   # sanity check
```

The proxy takes the `model` id from each request and passes it straight through,
so one running container serves **any** model the backend lists in `/v1/models`
(e.g. `Qwen/Qwen3.6-27B`, `openai/gpt-oss-120b`, `z-ai/glm-5.2`). No per-model
launch, no default model.

## Or run the binary directly

```bash
go build -o tf-anthropic-proxy .
export UPSTREAM_API_KEY="tf-...your key..."
./tf-anthropic-proxy            # listens on :4000
```

## Config (env)

| Var | Default | Notes |
|-----|---------|-------|
| `UPSTREAM_API_KEY` | — (required) | tokenfactory key; falls back to `AIGATEWAY_API_KEY` |
| `UPSTREAM_BASE_URL` | `https://api.tokenfactory.iamsaif.ai/v1` | upstream OpenAI base |
| `LISTEN_ADDR` | `:4000` | bind address |
| `UPSTREAM_UA` | `curl/8.4.0` | User-Agent sent upstream (WAF workaround) |

## Endpoints

- `POST /v1/messages` — streaming + non-streaming
- `POST /v1/messages/count_tokens` — rough estimate (~4 chars/token)
- `GET  /v1/models` — proxied from upstream `/models`
- `GET  /healthz`

## Use with Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_AUTH_TOKEN=anything     # proxy uses its own UPSTREAM_API_KEY
claude --model Qwen/Qwen3.6-27B
```

### Listing backend models in the `/model` picker

Claude Code (v2.1.129+) can populate `/model` from the proxy if you enable
gateway discovery:

```bash
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
export ANTHROPIC_BASE_URL=http://localhost:4000
claude          # /model now shows a "From gateway" group with backend models
```

Discovery only accepts model ids beginning with `claude`/`anthropic`, so the
proxy's `/v1/models` aliases every backend id as `claude-proxy--<id>` (the picker
shows the real name via `display_name`). On incoming requests the proxy strips
that prefix back off, so the real backend id is what gets sent upstream. You can
still pass a raw id (e.g. `Qwen/Qwen3.6-27B`) directly — the prefix is optional.

## Use with the Anthropic SDK

```python
from anthropic import Anthropic
c = Anthropic(base_url="http://localhost:4000", api_key="anything")
c.messages.create(model="Qwen/Qwen3.6-27B", max_tokens=512,
                  messages=[{"role": "user", "content": "hi"}])   # stream=True works too
```

## What's translated

- Text chat + system prompts (system is merged and hoisted to the front, as the
  upstream requires the system message first).
- **Tool use**, streaming and non-streaming: Anthropic `tools` ⇆ OpenAI `tools`,
  assistant `tool_use` ⇆ `tool_calls`, user `tool_result` ⇆ `role:"tool"`
  messages, and streaming `input_json_delta` for tool arguments. Enough to drive
  Claude Code as a coding agent (requires the backend model to support function
  calling — `Qwen/Qwen3.6-27B` does).

## Limitations

- **Image / document content blocks are not translated** (dropped).
- `Qwen3.6` is a reasoning model: budget `max_tokens` generously or the answer
  gets truncated by hidden reasoning tokens. Reasoning text is not surfaced.
- The model name is passed through unchanged, so the client must send a real
  tokenfactory model id (e.g. `Qwen/Qwen3.6-27B`).
```
