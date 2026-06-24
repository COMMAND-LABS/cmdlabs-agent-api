#!/usr/bin/env python3
"""
Load-test the Agent Completion SSE endpoint.

Simulates concurrent users sending prompts to
POST /api/agents/{agent_id}/stream and consuming the
Server-Sent Events stream until the final on_chain_end event.

Usage
-----
python scripts/load_test_agents.py \
  --base-url http://127.0.0.1:4000 \
  --agent-id 11 \
  --jwt "<JWT>" \
  --users 12 \
  --rounds 1 \
  --prompt "Give me a short summary of what you can do." \
  --timeout 0 \
  --max-request-seconds 180

Dependencies (stdlib + one extra):
    pip install httpx
    (httpx is already in the project's pyproject.toml)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Result data
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    """Outcome of a single SSE completion request."""
    user_id: int
    round_id: int
    session_id: str
    status: str  # "ok", "error", "timeout"
    http_status: Optional[int] = None
    duration_s: float = 0.0
    first_token_s: Optional[float] = None  # time-to-first-token
    chunks: int = 0
    chars: int = 0
    error: str = ""
    events: dict = field(default_factory=dict)  # event_name -> count


# ---------------------------------------------------------------------------
# SSE stream consumer
# ---------------------------------------------------------------------------

async def consume_sse(
    base_url: str,
    agent_id: int,
    jwt_token: str,
    prompt: str,
    session_id: str,
    user_id: int,
    round_id: int,
    max_request_seconds: float,
    api_key: Optional[str] = None,
) -> RequestResult:
    """Fire one completion request and consume the SSE stream."""
    import httpx

    url = f"{base_url.rstrip('/')}/api/agents/{agent_id}/stream"
    payload = {"prompt": prompt, "sessionId": session_id}

    headers: dict[str, str] = {"Content-Type": "application/json"}
    cookies: dict[str, str] = {}

    if api_key:
        headers["X-API-Key"] = api_key
    elif jwt_token:
        # Send JWT both as a cookie (primary) and Authorization header (fallback)
        cookies["jwt"] = jwt_token
        headers["Authorization"] = f"Bearer {jwt_token}"

    timeout = httpx.Timeout(
        connect=10.0,
        read=max_request_seconds if max_request_seconds > 0 else None,
        write=10.0,
        pool=10.0,
    )

    result = RequestResult(
        user_id=user_id,
        round_id=round_id,
        session_id=session_id,
        status="ok",
    )

    t_start = time.monotonic()
    first_token_recorded = False

    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True
        ) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers, cookies=cookies
            ) as resp:
                result.http_status = resp.status_code

                if resp.status_code != 200:
                    body = await resp.aread()
                    result.status = "error"
                    result.error = f"HTTP {resp.status_code}: {body.decode(errors='replace')[:500]}"
                    result.duration_s = time.monotonic() - t_start
                    return result

                buf = ""
                async for raw_chunk in resp.aiter_text():
                    buf += raw_chunk
                    # SSE frames are separated by double newlines
                    while "\n\n" in buf:
                        frame, buf = buf.split("\n\n", 1)
                        frame = frame.strip()
                        if not frame:
                            continue

                        # Parse SSE "data: ..." lines
                        data_lines = []
                        for line in frame.splitlines():
                            if line.startswith("data:"):
                                data_lines.append(line[len("data:"):].strip())
                            elif line.startswith("data :"):
                                data_lines.append(line[len("data :"):].strip())

                        if not data_lines:
                            continue

                        data_str = "\n".join(data_lines)
                        try:
                            event_obj = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_name = event_obj.get("event", "unknown")
                        result.events[event_name] = result.events.get(event_name, 0) + 1

                        if event_name == "on_chat_model_stream":
                            content = event_obj.get("data", "")
                            if content:
                                result.chunks += 1
                                result.chars += len(content)
                                if not first_token_recorded:
                                    result.first_token_s = time.monotonic() - t_start
                                    first_token_recorded = True

                        elif event_name == "error":
                            err_data = event_obj.get("data", {})
                            result.status = "error"
                            result.error = (
                                err_data.get("message", "")
                                or err_data.get("error", "")
                                or str(err_data)
                            )

    except httpx.ReadTimeout:
        result.status = "timeout"
        result.error = f"Read timed out after {max_request_seconds}s"
    except httpx.ConnectTimeout:
        result.status = "timeout"
        result.error = "Connection timed out"
    except httpx.ConnectError as exc:
        result.status = "error"
        result.error = f"Connection refused: {exc}"
    except Exception as exc:
        result.status = "error"
        result.error = str(exc)

    result.duration_s = time.monotonic() - t_start
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_round(
    round_id: int,
    users: int,
    base_url: str,
    agent_id: int,
    jwt_token: str,
    prompt: str,
    max_request_seconds: float,
    api_key: Optional[str] = None,
) -> list[RequestResult]:
    """Launch *users* concurrent requests for a single round."""
    tasks = []
    for uid in range(1, users + 1):
        session_id = str(uuid.uuid4())
        tasks.append(
            consume_sse(
                base_url=base_url,
                agent_id=agent_id,
                jwt_token=jwt_token,
                prompt=prompt,
                session_id=session_id,
                user_id=uid,
                round_id=round_id,
                max_request_seconds=max_request_seconds,
                api_key=api_key,
            )
        )
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _color(status: str) -> str:
    return {"ok": _GREEN, "error": _RED, "timeout": _YELLOW}.get(status, "")


def print_header(args: argparse.Namespace) -> None:
    print()
    print(f"{_BOLD}{'=' * 64}")
    print(f"  LOAD TEST — Agent Completion SSE Endpoint")
    print(f"{'=' * 64}{_RESET}")
    print(f"  Base URL           : {args.base_url}")
    print(f"  Agent ID           : {args.agent_id}")
    print(f"  Concurrent users   : {args.users}")
    print(f"  Rounds             : {args.rounds}")
    print(f"  Max request (s)    : {args.max_request_seconds}")
    print(f"  Prompt             : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    auth = "API Key" if args.api_key else ("JWT" if args.jwt else "None")
    print(f"  Auth               : {auth}")
    print(f"{'=' * 64}")
    print()


def print_result_line(r: RequestResult) -> None:
    color = _color(r.status)
    ttft = f"{r.first_token_s:.2f}s" if r.first_token_s is not None else "  n/a"
    print(
        f"  {_DIM}R{r.round_id:>2d}/U{r.user_id:>3d}{_RESET}  "
        f"[{color}{r.status:>7s}{_RESET}]  "
        f"HTTP {r.http_status or '---':>3}  "
        f"dur {r.duration_s:>7.2f}s  "
        f"TTFT {ttft}  "
        f"chunks {r.chunks:>4d}  "
        f"chars {r.chars:>6d}"
        f"{f'  {_RED}{r.error[:60]}{_RESET}' if r.error else ''}"
    )


def print_summary(all_results: list[RequestResult]) -> None:
    ok = [r for r in all_results if r.status == "ok"]
    errors = [r for r in all_results if r.status == "error"]
    timeouts = [r for r in all_results if r.status == "timeout"]
    total = len(all_results)

    durations = [r.duration_s for r in all_results]
    ok_durations = [r.duration_s for r in ok]
    ttfts = [r.first_token_s for r in ok if r.first_token_s is not None]
    total_chars = sum(r.chars for r in ok)
    total_chunks = sum(r.chunks for r in ok)

    print()
    print(f"{_BOLD}{'=' * 64}")
    print(f"  SUMMARY")
    print(f"{'=' * 64}{_RESET}")

    print(f"\n  {_BOLD}Requests{_RESET}")
    print(f"    Total            : {total}")
    print(f"    {_GREEN}OK{_RESET}               : {len(ok)}")
    print(f"    {_RED}Errors{_RESET}           : {len(errors)}")
    print(f"    {_YELLOW}Timeouts{_RESET}         : {len(timeouts)}")
    if total:
        print(f"    Success rate     : {len(ok) / total * 100:.1f}%")

    if durations:
        print(f"\n  {_BOLD}Latency (all requests){_RESET}")
        print(f"    Min              : {min(durations):.2f}s")
        print(f"    Max              : {max(durations):.2f}s")
        print(f"    Mean             : {statistics.mean(durations):.2f}s")
        if len(durations) >= 2:
            print(f"    Median           : {statistics.median(durations):.2f}s")
            print(f"    Stdev            : {statistics.stdev(durations):.2f}s")
        sorted_dur = sorted(durations)
        p95_idx = max(0, int(len(sorted_dur) * 0.95) - 1)
        p99_idx = max(0, int(len(sorted_dur) * 0.99) - 1)
        print(f"    P95              : {sorted_dur[p95_idx]:.2f}s")
        print(f"    P99              : {sorted_dur[p99_idx]:.2f}s")

    if ok_durations:
        print(f"\n  {_BOLD}Latency (successful only){_RESET}")
        print(f"    Min              : {min(ok_durations):.2f}s")
        print(f"    Max              : {max(ok_durations):.2f}s")
        print(f"    Mean             : {statistics.mean(ok_durations):.2f}s")
        if len(ok_durations) >= 2:
            print(f"    Median           : {statistics.median(ok_durations):.2f}s")

    if ttfts:
        print(f"\n  {_BOLD}Time-to-First-Token (TTFT){_RESET}")
        print(f"    Min              : {min(ttfts):.2f}s")
        print(f"    Max              : {max(ttfts):.2f}s")
        print(f"    Mean             : {statistics.mean(ttfts):.2f}s")
        if len(ttfts) >= 2:
            print(f"    Median           : {statistics.median(ttfts):.2f}s")

    if ok:
        print(f"\n  {_BOLD}Throughput{_RESET}")
        wall_time = max(durations) if durations else 1
        print(f"    Wall-clock time  : {wall_time:.2f}s")
        print(f"    Requests/sec     : {total / wall_time:.2f}")
        print(f"    Total chunks     : {total_chunks}")
        print(f"    Total chars      : {total_chars}")
        if wall_time > 0:
            print(f"    Chars/sec        : {total_chars / wall_time:.1f}")

    # SSE event breakdown
    all_events: dict[str, int] = {}
    for r in all_results:
        for evt, cnt in r.events.items():
            all_events[evt] = all_events.get(evt, 0) + cnt
    if all_events:
        print(f"\n  {_BOLD}SSE Events (aggregate){_RESET}")
        for evt in sorted(all_events):
            print(f"    {evt:<28s}: {all_events[evt]:>6d}")

    # Error breakdown
    if errors or timeouts:
        print(f"\n  {_BOLD}Error Details{_RESET}")
        seen: dict[str, int] = {}
        for r in errors + timeouts:
            key = r.error[:120] if r.error else "(no message)"
            seen[key] = seen.get(key, 0) + 1
        for msg, cnt in sorted(seen.items(), key=lambda x: -x[1]):
            print(f"    [{cnt:>3d}x] {msg}")

    print(f"\n{'=' * 64}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load-test the Agent Completion SSE endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--base-url",
        required=True,
        help="Base URL of the API (e.g. http://127.0.0.1:4000)",
    )
    p.add_argument(
        "--agent-id",
        type=int,
        required=True,
        help="Agent ID to target",
    )
    p.add_argument(
        "--jwt",
        default="",
        help="JWT token for authentication",
    )
    p.add_argument(
        "--api-key",
        default="",
        help="API key for authentication (alternative to --jwt)",
    )
    p.add_argument(
        "--users",
        type=int,
        default=1,
        help="Number of concurrent simulated users (default: 1)",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of sequential rounds to run (default: 1)",
    )
    p.add_argument(
        "--prompt",
        default="Hello, what can you help me with?",
        help="Prompt to send to the agent",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Alias for --max-request-seconds (0 = no timeout). "
             "Kept for backwards compatibility.",
    )
    p.add_argument(
        "--max-request-seconds",
        type=float,
        default=0,
        help="Max seconds to wait for each request to complete (0 = no limit, default: 0)",
    )
    p.add_argument(
        "--delay-between-rounds",
        type=float,
        default=1.0,
        help="Seconds to pause between rounds (default: 1.0)",
    )
    p.add_argument(
        "--json-output",
        default="",
        help="Optional path to write raw results as a JSON file",
    )
    return p


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve effective timeout: --max-request-seconds takes precedence,
    # fall back to --timeout for the alias
    if args.max_request_seconds <= 0 and args.timeout > 0:
        args.max_request_seconds = args.timeout

    if not args.jwt and not args.api_key:
        print(
            f"{_YELLOW}Warning: No --jwt or --api-key provided. "
            f"Requests will likely fail with 401.{_RESET}"
        )

    print_header(args)

    all_results: list[RequestResult] = []

    for round_id in range(1, args.rounds + 1):
        print(f"{_CYAN}--- Round {round_id}/{args.rounds} "
              f"({args.users} users) ---{_RESET}")

        round_results = await run_round(
            round_id=round_id,
            users=args.users,
            base_url=args.base_url,
            agent_id=args.agent_id,
            jwt_token=args.jwt,
            prompt=args.prompt,
            max_request_seconds=args.max_request_seconds,
            api_key=args.api_key,
        )
        for r in sorted(round_results, key=lambda x: x.user_id):
            print_result_line(r)
            all_results.append(r)

        if round_id < args.rounds and args.delay_between_rounds > 0:
            print(f"  {_DIM}(pausing {args.delay_between_rounds:.1f}s){_RESET}")
            await asyncio.sleep(args.delay_between_rounds)

    print_summary(all_results)

    # Optional JSON dump
    if args.json_output:
        import dataclasses
        with open(args.json_output, "w") as f:
            json.dump(
                [dataclasses.asdict(r) for r in all_results],
                f,
                indent=2,
            )
        print(f"Raw results written to {args.json_output}")

    # Exit code: non-zero if any requests failed
    failures = sum(1 for r in all_results if r.status != "ok")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
