# Kalygo3 Agent API

Streaming LLM completion microservice. Handles agent completion (streaming tokens, tools, RAG) independently from the main AI API for better scalability.

## Endpoints

All `/api` endpoints accept unified auth — a JWT cookie or a `kalygo_` API key
(via `Authorization: Bearer` or `X-API-Key`).

- `GET /` – Health check
- `POST /api/agents/{agent_id}/stream` – Stream an agent completion as Server-Sent
  Events (tokens, tool calls, RAG, file uploads), ending on `on_chain_end`.
- `GET /api/agents/{agent_id}` – Read an agent's configuration (owner or granted access).
- `POST /api/contact-chat/{session_id}/stream` – Stream the code-defined,
  contact-scoped CRM agent for a session (no `agent_id`; the path `session_id` is authoritative).

Interactive API docs (Swagger UI) are served at `/api/docs`; the raw OpenAPI
schema is at `/openapi.json`.

## Environment

Use the same `.env` as the main AI API (or subset): `POSTGRES_URL`, `AUTH_SECRET_KEY`, `AUTH_ALGORITHM`, `CREDENTIALS_ENCRYPTION_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `EMBEDDINGS_API_URL`, `RERANKER_API_URL`, `PINECONE_*`, etc.

Optional: `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT`, `DB_POOL_RECYCLE` to tune connection pool for streaming load.

## Run locally

```bash
uvicorn src.main:app --host 0.0.0.0 --port 4100 --reload
```

Or with Docker Compose from the repo root:

```bash
docker compose -f docker-compose.dev.yml up
```

Agent API runs on port **4100** by default.

## Testing

````sh
python scripts/load_test_agents.py \
  --base-url http://127.0.0.1:4100 \
  --agent-id 11 \
  --jwt "<JWT_HERE>" \
  --users 24 \
  --rounds 1 \
  --prompt "Give me a short summary of what you can do." \
  --timeout 0 \
  --max-request-seconds 180
``

```sh
python scripts/load_test_sweep.py \
  --base-url http://127.0.0.1:4100 \
  --agent-id 11 \
  --jwt "<JWT_HERE>" \
  --steps 1,2,4,8,12,16,20,24,32,48,64 \
  --rounds-per-step 1 \
  --prompt "Give me a short summary of what you can do." \
  --max-request-seconds 180
  --fail-threshold 0.5
````

```sh
python scripts/load_test_sweep.py \
  --base-url https://agent.kalygo.io \
  --agent-id 11 \
  --jwt "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0YWRAY21kbGFicy5pbyIsImlkIjoxLCJleHAiOjE3NzEyMzgyNDF9.XdxdC1JcX2-90WU3LpbX8Z4XpL7RPDd4oS-HRrvGot8" \
  --steps 1,2,4,8,12,16,20,24,32,48,64 \
  --rounds-per-step 1 \
  --prompt "Give me a short summary of what you can do." \
  --max-request-seconds 180
  --fail-threshold 0.5
```

## TRIGGER CICD

7
