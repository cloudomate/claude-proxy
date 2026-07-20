# tf-anthropic-proxy

A tiny Go service that exposes an **Anthropic Messages API** (`/v1/messages`) and
translates to an OpenAI-compatible backend (`/v1/chat/completions`) — with proper
Anthropic SSE streaming and full tool use.

## Run as a container

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
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_AUTH_TOKEN=anything     # proxy uses its own UPSTREAM_API_KEY
claude
```

`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` makes Claude Code (v2.1.129+)
populate the `/model` picker with a "From gateway" group listing the backend
models.

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
- **Images**: Anthropic `image` blocks (base64 or URL) ⇆ OpenAI `image_url`
  content parts. Forwarded only when you route to a vision-capable model (e.g.
  `qwen/qwen3-omni-30b-a3b-instruct`); text-only models will ignore them.

## Limitations

- **`document` (PDF) content blocks are not translated** (dropped).
- Reasoning models (e.g. `Qwen3.6`) burn hidden tokens before output — budget
  `max_tokens` generously or turns get truncated. Reasoning text is not surfaced.
- The model name is passed through unchanged, so the client must send a real
  backend model id (e.g. `Qwen/Qwen3.6-27B`).
```
