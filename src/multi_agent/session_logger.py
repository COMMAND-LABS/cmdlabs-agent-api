"""Per-session log files for multi-agent debugging.

Each session gets a single ``{session_uuid}.log`` file inside the
configured log directory.  Entries are appended across requests so
the full conversation is captured in one place.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List


class SessionLogger:
    """Append-only, plain-text logger keyed by session UUID."""

    def __init__(self, session_id: str, log_dir: str = "logs/multi-agent"):
        self._session_id = session_id
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{session_id}.log"

    def _write(self, line: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        with open(self._path, "a") as f:
            f.write(f"[{ts}] {line}\n")

    def _write_block(self, header: str, body: str) -> None:
        """Write a multi-line block with a header and indented body."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        with open(self._path, "a") as f:
            f.write(f"[{ts}] {header}\n")
            for line in body.splitlines():
                f.write(f"  {line}\n")

    def log_route(self, prompt: str, speakers: List[str]) -> None:
        self._write(f"ROUTE  prompt={prompt!r}  speakers={speakers}")

    def log_llm_messages(self, label: str, messages: list) -> None:
        """Log the full message list sent to an LLM call.

        *messages* can be LangChain message objects or plain dicts.
        """
        lines: list[str] = []
        for m in messages:
            if hasattr(m, "type"):
                role = m.type
                name = getattr(m, "name", None)
                content = m.content
            else:
                role = m.get("role", "?")
                name = m.get("name")
                content = m.get("content", "")
            tag = f"{role}({name})" if name else role
            lines.append(f"[{tag}] {content}")
        self._write_block(f"LLM_MESSAGES  step={label!r}  count={len(messages)}", "\n".join(lines))

    def log_agent_start(self, agent_name: str, history_len: int, prompt: str) -> None:
        self._write(f"AGENT_START  agent={agent_name!r}  history_msgs={history_len}  prompt={prompt!r}")

    def log_agent_end(self, agent_name: str, response: str, elapsed_s: float) -> None:
        preview = response[:200] + ("\u2026" if len(response) > 200 else "")
        self._write(f"AGENT_END  agent={agent_name!r}  elapsed={elapsed_s:.2f}s  response={preview!r}")

    def log_error(self, context: str, error: Exception) -> None:
        self._write(f"ERROR  context={context!r}  error={error!r}")

    @staticmethod
    def timer() -> float:
        """Return a monotonic timestamp; subtract two to get elapsed seconds."""
        return time.monotonic()
