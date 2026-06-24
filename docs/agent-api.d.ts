/**
 * Kalygo3 Agent API — client contract.
 *
 * This file is the single source of truth for a client integrating with the
 * Agent API: the HTTP request/response bodies and the exact shape of every
 * event emitted over the streaming endpoints.
 *
 * Companion prose + examples: ./AGENT_API.md
 *
 * Generated to mirror the server in src/. If the server changes, update both
 * this file and AGENT_API.md.
 */

// ===========================================================================
// Authentication
// ===========================================================================
//
// Every endpoint accepts EITHER of:
//   1. A `jwt` cookie (browser session), OR
//   2. An API key (prefixed `kalygo_`) sent as:
//        Authorization: Bearer <key>
//      or
//        X-API-Key: <key>
//
// On failure the server responds 401 with `{ detail: string }`.

// ===========================================================================
// HTTP endpoints
// ===========================================================================
//
//   GET  /                                  -> HealthResponse
//   GET  /api/agents/{agentId}              -> AgentResponse        (404 if no access)
//   POST /api/agents/{agentId}/stream       -> AgentStreamEvent[]   (streaming, see below)
//   POST /api/contact-chat/{sessionId}/stream -> AgentStreamEvent[] (streaming, fixed CRM agent)
//
// Rate limits: stream endpoints 200/min, GET agent 60/min.

export interface HealthResponse {
  status: string; // "OK!"
}

export interface AgentResponse {
  id: number;
  name: string;
  /** Arbitrary agent configuration blob (model, systemPrompt, tools, ...). */
  config: Record<string, unknown> | null;
  /** True if the authenticated caller owns this agent (vs. shared access). */
  is_owner: boolean | null;
}

/**
 * Request body for BOTH streaming endpoints.
 *
 * `prompt` and `sessionId` are required; everything else is an optional
 * attachment. At most one inline attachment (pdf | image | documentText) is
 * meaningful per request. The gcs* / attachment* fields are a durable
 * reference persisted onto the stored chat message.
 *
 * NOTE: for POST /api/contact-chat/{sessionId}/stream the PATH sessionId is
 * authoritative; the body `sessionId` is ignored on that route.
 */
export interface ChatSessionPrompt {
  /** The user's message. */
  prompt: string;
  /** Chat session UUID (string form). Must be a valid UUID. */
  sessionId: string;

  /** Base64-encoded PDF attachment. */
  pdf?: string | null;
  pdfFilename?: string | null;
  /**
   * PDF processing mode.
   *  - true:  render pages as images (vision) — scanned PDFs, charts, layout.
   *  - false: text extraction — data extraction, cheaper. Default false.
   */
  pdfUseVision?: boolean | null;

  /** Base64-encoded image attachment for vision models. */
  image?: string | null;

  /** Inline text content for txt/csv/md attachments. */
  documentText?: string | null;

  // Durable GCS reference for the persisted attachment (model-facing content
  // still rides inline via the fields above).
  gcsBucket?: string | null;
  gcsFilePath?: string | null;
  attachmentFilename?: string | null;
  attachmentContentType?: string | null;
}

// ===========================================================================
// Stream framing
// ===========================================================================
//
// The response Content-Type is `text/event-stream` and the server emits
// standard SSE frames: each event is a single `data:` line holding the compact
// JSON payload, terminated by a blank line:
//
//     data: {"event":"on_chat_model_stream","data":"Hello"}\n\n
//
// You can parse the stream with any standard SSE parser (e.g.
// `eventsource-parser`, `@microsoft/fetch-event-source`), then `JSON.parse`
// the `data` field of each message to get an `AgentStreamEvent`. The event
// TYPE lives inside the JSON payload (the `event` field), NOT the SSE `event:`
// line — discriminate on the parsed object's `event` field.
//
// Note: native `EventSource` is GET-only and cannot send the Authorization /
// X-API-Key header or a POST body, so use `fetch` + a parser (or
// `@microsoft/fetch-event-source`) rather than `EventSource`.

// ===========================================================================
// Stream events
// ===========================================================================

/** Discriminated union of every event a stream can emit. Switch on `event`. */
export type AgentStreamEvent =
  | OnChainStartEvent
  | OnChatModelStartEvent
  | OnChatModelStreamEvent
  | OnToolStartEvent
  | OnToolEndEvent
  | ToolApprovalRequiredEvent
  | OnChainEndEvent
  | ErrorEvent;

/** The run has started. Emitted once at the top of the agent-executor path. */
export interface OnChainStartEvent {
  event: "on_chain_start";
}

/**
 * A model invocation is starting. On the tool-enabled agent path this MAY
 * carry the tool calls accumulated so far (omitted when empty). On the simple
 * (no-tool) chat path it carries no payload.
 */
export interface OnChatModelStartEvent {
  event: "on_chat_model_start";
  toolCalls?: ToolCall[];
}

/**
 * A streamed text token chunk. This is the primary content stream — concatenate
 * `data` across all on_chat_model_stream events to render the reply.
 */
export interface OnChatModelStreamEvent {
  event: "on_chat_model_stream";
  data: string;
}

/** A tool is about to execute. `run_id` correlates with the matching on_tool_end. */
export interface OnToolStartEvent {
  event: "on_tool_start";
  data: {
    name: string;
    /** Parsed tool input; `{}` if the server could not parse it. */
    input: Record<string, unknown>;
  };
  run_id: string;
}

/**
 * A tool finished.
 *  - Normal case: `data` is the formatted ToolCall.
 *  - When the formatter rejects the output (e.g. error results), `data` is absent.
 *  - When the tool is a HITL-gated email, `data` is absent and a separate
 *    tool_approval_required event is emitted immediately BEFORE this one.
 * `run_id` correlates with the matching on_tool_start.
 */
export interface OnToolEndEvent {
  event: "on_tool_end";
  data?: ToolCall | null;
  run_id: string;
}

/**
 * A human-in-the-loop tool (e.g. send-email) has queued an action for approval
 * instead of executing it. The client should surface the preview and collect a
 * decision out-of-band (handled by a separate approval API, not this stream).
 */
export interface ToolApprovalRequiredEvent {
  event: "tool_approval_required";
  data: {
    approval_id: string;
    tool_type: string;
    preview: {
      to_email?: string;
      subject?: string;
      body?: string;
      [k: string]: unknown;
    };
  };
}

/**
 * Terminal success event. `data` is the full final assistant text. On the
 * tool-enabled path it also carries the complete list of tool calls made
 * during the run (omitted when none).
 */
export interface OnChainEndEvent {
  event: "on_chain_end";
  data: string;
  toolCalls?: ToolCall[];
}

/** Terminal error event. May occur during setup or mid-stream. */
export interface ErrorEvent {
  event: "error";
  data: {
    /** Short error type/code, e.g. "Agent not found", "Streaming error". */
    error: string;
    /** Human-readable detail. */
    message: string;
  };
}

// ===========================================================================
// Tool-call payloads
// ===========================================================================
//
// The shape carried by on_tool_end.data and within on_chain_end.toolCalls /
// on_chat_model_start.toolCalls. Discriminate on `toolType`.

export type ToolCall =
  | VectorSearchToolCall
  | DbTableReadToolCall
  | DbTableWriteToolCall
  | SendTxtEmailToolCall
  | SendHtmlEmailToolCall
  | CustomToolCall;

export interface VectorSearchToolCall {
  toolType: "vectorSearch" | "vectorSearchWithReranking";
  toolName: string;
  input: {
    query: string;
    topK?: number;
  };
  output: {
    results: VectorSearchResult[];
    namespace: string;
    index: string;
  };
}

export interface VectorSearchResult {
  id: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface DbTableReadToolCall {
  toolType: "dbTableRead";
  /** Tool name starts with "db_table_read". */
  toolName: string;
  input: {
    filters?: unknown;
    limit?: number | null;
    offset?: number | null;
  };
  output: {
    results: Array<Record<string, unknown>>;
    table: string;
    count: number;
  };
}

export interface DbTableWriteToolCall {
  toolType: "dbTableWrite";
  /** Tool name starts with "db_table_write". */
  toolName: string;
  input: {
    /** The flat tool input IS the row data. */
    data: Record<string, unknown>;
  };
  output: {
    success: boolean;
    table: string;
    inserted: Record<string, unknown>;
    message: string;
    error?: string | null;
  };
}

export interface SendTxtEmailToolCall {
  toolType: "sendTxtEmailWithSes";
  toolName: string;
  input: {
    to: string;
    subject: string;
    body: string;
  };
  output: {
    success: boolean;
    messageId?: string | null;
    error?: string | null;
  };
}

export interface SendHtmlEmailToolCall {
  toolType: "sendHtmlEmailWithSes";
  toolName: string;
  input: {
    to: string;
    subject: string;
    html_body?: string | null;
    /** Present in template mode. */
    template_id?: string | number;
    /** Present in template mode. */
    variables?: Record<string, unknown>;
  };
  output: {
    success: boolean;
    messageId?: string | null;
    error?: string | null;
  };
}

/** Fallback shape for any tool without a dedicated formatter. */
export interface CustomToolCall {
  toolType: "custom";
  toolName: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
}
