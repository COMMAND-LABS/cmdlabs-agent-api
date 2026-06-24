# Kalygo3 Agent API — Client Contract

This document is the client-facing interface for the Agent API. It defines the
HTTP endpoints, authentication, request bodies, and the **exact shape of every
event** emitted by the streaming endpoints.

The machine-readable companion is [`agent-api.d.ts`](./agent-api.d.ts) —
TypeScript types for every request body and stream event. Use that file for
compile-time checking; use this file for prose, examples, and gotchas.

> Keep both files in sync with `src/` when the server changes.

---

## Base URL

All paths below are relative to the deployment origin (e.g.
`https://<host>`). Interactive OpenAPI docs are served at `/api/docs`.

---

## Authentication

Every endpoint accepts **either** of:

1. A `jwt` cookie (browser session), **or**
2. An API key (prefixed `kalygo_`), sent as **either**:
   - `Authorization: Bearer <key>`, or
   - `X-API-Key: <key>`

On failure: `401` with `{ "detail": "..." }`.

---

## Endpoints

| Method | Path | Purpose | Response |
| ------ | ---- | ------- | -------- |
| `GET`  | `/` | Health check | `{ "status": "OK!" }` |
| `GET`  | `/api/agents/{agentId}` | Read agent config | `AgentResponse` (404 if no access) |
| `POST` | `/api/agents/{agentId}/stream` | Stream a completion from a configured agent | event stream |
| `POST` | `/api/contact-chat/{sessionId}/stream` | Stream the fixed, contact-scoped CRM agent | event stream |

**Rate limits:** stream endpoints `200/min`; `GET /api/agents/{agentId}` `60/min`.
Exceeding a limit returns `429`.

### `GET /api/agents/{agentId}`

Returns the agent if the caller owns it or has been granted access; otherwise
`404` (existence is deliberately not leaked).

```jsonc
// 200 OK — AgentResponse
{
  "id": 42,
  "name": "Support Bot",
  "config": { /* model, systemPrompt, tools, ... */ },
  "is_owner": true
}
```

### `POST /api/agents/{agentId}/stream`

Body: [`ChatSessionPrompt`](#request-body-chatsessionprompt). Returns a stream
of [events](#stream-events).

### `POST /api/contact-chat/{sessionId}/stream`

Same request body and same event stream, but runs a fixed server-defined
contact-scoped CRM agent — there is no `agentId`. The **path `sessionId` is
authoritative**; the `sessionId` field in the body is ignored on this route.

---

## Request body (`ChatSessionPrompt`)

Used by both streaming endpoints.

| Field | Type | Required | Notes |
| ----- | ---- | :------: | ----- |
| `prompt` | string | ✅ | The user's message. |
| `sessionId` | string (UUID) | ✅ | Chat session UUID. Ignored on the contact-chat route (path wins). |
| `pdf` | string \| null | | Base64-encoded PDF. |
| `pdfFilename` | string \| null | | |
| `pdfUseVision` | boolean \| null | | `true` = render pages as images (vision); `false` = text extraction (default). |
| `image` | string \| null | | Base64-encoded image for vision models. |
| `documentText` | string \| null | | Inline text for txt/csv/md attachments. |
| `gcsBucket` | string \| null | | Durable GCS reference persisted on the stored message. |
| `gcsFilePath` | string \| null | | |
| `attachmentFilename` | string \| null | | |
| `attachmentContentType` | string \| null | | |

At most one inline attachment (`pdf` | `image` | `documentText`) is meaningful
per request.

```jsonc
// Minimal request
{ "prompt": "Summarize my last support thread", "sessionId": "0c8b...-uuid" }
```

---

## Stream framing

The response `Content-Type` is `text/event-stream` and the server emits
**standard SSE frames**: each event is a single `data:` line holding the
compact JSON payload, terminated by a blank line.

```
data: {"event":"on_chat_model_stream","data":"Hello"}

data: {"event":"on_chain_end","data":"Hello there"}

```

Parse with any standard SSE parser ([`eventsource-parser`](https://www.npmjs.com/package/eventsource-parser),
[`@microsoft/fetch-event-source`](https://www.npmjs.com/package/@microsoft/fetch-event-source)),
then `JSON.parse` each message's `data` to get an event object. The event
**type lives inside the JSON payload** (the `event` field), **not** the SSE
`event:` line — discriminate on the parsed object's `event` field.

> Native `EventSource` is GET-only and cannot send the `Authorization` /
> `X-API-Key` header or a POST body, so use `fetch` + a parser (or
> `@microsoft/fetch-event-source`), not `EventSource`.

---

## Stream events

All events have an `event` discriminant. Two execution paths exist:

- **Agent path** (agent has tools): emits `on_chain_start`, `on_chat_model_start`,
  `on_chat_model_stream`, `on_tool_start`, `on_tool_end`, optionally
  `tool_approval_required`, then `on_chain_end`.
- **Simple chat path** (no tools): emits `on_chat_model_start`,
  `on_chat_model_stream`, then `on_chain_end`. No tool events, no `toolCalls`.

Either path may end early with an `error` event.

### `on_chain_start`
Run started (agent path only).
```json
{ "event": "on_chain_start" }
```

### `on_chat_model_start`
A model invocation is starting. On the agent path may include accumulated
`toolCalls` (omitted when empty). On the simple path, payload-less.
```json
{ "event": "on_chat_model_start", "toolCalls": [] }
```

### `on_chat_model_stream`
A streamed text token. **Concatenate `data` across all of these to build the
reply.**
```json
{ "event": "on_chat_model_stream", "data": "Hello" }
```

### `on_tool_start`
A tool is about to run. `run_id` pairs with the matching `on_tool_end`.
```json
{ "event": "on_tool_start", "data": { "name": "vector_search", "input": { "query": "refunds" } }, "run_id": "abc-123" }
```

### `on_tool_end`
A tool finished. `data` is a formatted [`ToolCall`](#tool-call-payloads). `data`
is **absent** when the formatter rejects the output, or when the tool was a
HITL email (a `tool_approval_required` event precedes it).
```json
{ "event": "on_tool_end", "data": { "toolType": "vectorSearch", "toolName": "vector_search", "input": { "query": "refunds" }, "output": { "results": [], "namespace": "kb", "index": "main" } }, "run_id": "abc-123" }
```

### `tool_approval_required`
A human-in-the-loop tool queued an action for approval instead of executing.
Surface the preview; collect the decision via a separate approval API (not this
stream).
```json
{ "event": "tool_approval_required", "data": { "approval_id": "appr_1", "tool_type": "sendHtmlEmailWithSes", "preview": { "to_email": "x@y.com", "subject": "Hi", "body": "..." } } }
```

### `on_chain_end`
Terminal success. `data` is the full final assistant text. On the agent path,
also carries the complete `toolCalls` made during the run (omitted when none).
```json
{ "event": "on_chain_end", "data": "Here is your summary...", "toolCalls": [] }
```

### `error`
Terminal error (setup or mid-stream).
```json
{ "event": "error", "data": { "error": "Agent not found", "message": "The specified agent was not found or you do not have access." } }
```

---

## Tool-call payloads

The object carried by `on_tool_end.data` and inside the `toolCalls` arrays.
Discriminate on `toolType`. Full types in [`agent-api.d.ts`](./agent-api.d.ts).

| `toolType` | `toolName` | `input` keys | `output` keys |
| ---------- | ---------- | ------------ | ------------- |
| `vectorSearch` / `vectorSearchWithReranking` | tool name | `query`, `topK?` | `results[]` (`id`, `score`, `metadata`), `namespace`, `index` |
| `dbTableRead` | starts `db_table_read` | `filters?`, `limit?`, `offset?` | `results[]`, `table`, `count` |
| `dbTableWrite` | starts `db_table_write` | `data` | `success`, `table`, `inserted`, `message`, `error?` |
| `sendTxtEmailWithSes` | tool name | `to`, `subject`, `body` | `success`, `messageId?`, `error?` |
| `sendHtmlEmailWithSes` | tool name | `to`, `subject`, `html_body?`, `template_id?`, `variables?` | `success`, `messageId?`, `error?` |
| `custom` | tool name | passthrough input | passthrough output |

---

## Minimal client sketch

```ts
import type { AgentStreamEvent } from "./agent-api.d.ts";
import { createParser } from "eventsource-parser";

const res = await fetch(`/api/agents/${agentId}/stream`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
  body: JSON.stringify({ prompt, sessionId }),
});

let reply = "";
const parser = createParser((event) => {
  if (event.type !== "event") return;
  const evt = JSON.parse(event.data) as AgentStreamEvent;
  switch (evt.event) {
    case "on_chat_model_stream": reply += evt.data; break;
    case "on_tool_start": /* show "running <name>..." */ break;
    case "tool_approval_required": /* prompt user to approve */ break;
    case "on_chain_end": /* done; evt.data is the final text */ break;
    case "error": /* show evt.data.message */ break;
  }
});

const reader = res.body!.getReader();
const decoder = new TextDecoder();
for (;;) {
  const { done, value } = await reader.read();
  if (done) break;
  parser.feed(decoder.decode(value, { stream: true }));
}
```
