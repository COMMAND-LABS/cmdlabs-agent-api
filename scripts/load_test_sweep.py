#!/usr/bin/env python3
"""
Automated load-test sweep for the Agent Completion SSE endpoint.

Runs a staircase of increasing concurrency levels, collects metrics at
each step, and prints a final comparative report.  Optionally stops
early when a failure-rate threshold is breached.

Imports the core SSE consumer from load_test_agents.py (same directory).

Usage
-----
python scripts/load_test_sweep.py \
  --base-url http://127.0.0.1:4100 \
  --agent-id 11 \
  --jwt "<JWT>" \
  --steps 1,2,4,8,12,16,20,24 \
  --rounds-per-step 1 \
  --prompt "Give me a short summary of what you can do." \
  --max-request-seconds 180

Dependencies: httpx  (already in pyproject.toml)
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Import core helpers from the sibling script
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_LOAD_TEST_PATH = _SCRIPT_DIR / "load_test_agents.py"

_spec = importlib.util.spec_from_file_location("load_test_agents", _LOAD_TEST_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["load_test_agents"] = _mod  # register before exec so dataclasses resolve
_spec.loader.exec_module(_mod)

run_round = _mod.run_round
RequestResult = _mod.RequestResult
print_result_line = _mod.print_result_line


# ---------------------------------------------------------------------------
# Preflight health check
# ---------------------------------------------------------------------------

async def preflight_check(base_url: str, timeout_s: float = 10.0) -> bool:
    """
    Hit GET / (healthcheck) with a short timeout to verify the server is
    reachable *before* starting the expensive sweep.
    Returns True if the server responded with 2xx.
    """
    import httpx

    url = f"{base_url.rstrip('/')}/"
    print(f"  Preflight: GET {url} … ", end="", flush=True)
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url)
            if 200 <= resp.status_code < 300:
                print(f"{_GREEN}OK{_RESET} (HTTP {resp.status_code})")
                return True
            else:
                print(f"{_RED}FAIL{_RESET} (HTTP {resp.status_code}: {resp.text[:120]})")
                return False
    except httpx.ConnectError as exc:
        print(f"{_RED}FAIL{_RESET} — connection refused ({exc})")
        return False
    except httpx.ConnectTimeout:
        print(f"{_RED}FAIL{_RESET} — connection timed out after {timeout_s}s")
        return False
    except Exception as exc:
        print(f"{_RED}FAIL{_RESET} — {exc}")
        return False


# ---------------------------------------------------------------------------
# Progress ticker — prints elapsed time while requests are in flight
# ---------------------------------------------------------------------------

async def _ticker(concurrency: int, interval: float = 5.0) -> None:
    """Print a heartbeat every *interval* seconds so the terminal isn't silent."""
    t0 = time.monotonic()
    while True:
        await asyncio.sleep(interval)
        elapsed = time.monotonic() - t0
        print(f"  {_DIM}⏳ waiting … {elapsed:.0f}s elapsed "
              f"({concurrency} request{'s' if concurrency != 1 else ''} in flight){_RESET}",
              flush=True)


async def run_round_with_ticker(
    round_id: int,
    users: int,
    base_url: str,
    agent_id: int,
    jwt_token: str,
    prompt: str,
    max_request_seconds: float,
    api_key: str | None = None,
    ticker_interval: float = 10.0,
) -> list:
    """Wrap run_round with a background ticker that prints elapsed time."""
    ticker_task = asyncio.create_task(_ticker(users, ticker_interval))
    try:
        results = await run_round(
            round_id=round_id,
            users=users,
            base_url=base_url,
            agent_id=agent_id,
            jwt_token=jwt_token,
            prompt=prompt,
            max_request_seconds=max_request_seconds,
            api_key=api_key,
        )
    finally:
        ticker_task.cancel()
        try:
            await ticker_task
        except asyncio.CancelledError:
            pass
    return results


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_WHITE_BG = "\033[47m\033[30m"


def _bar(fraction: float, width: int = 30) -> str:
    """Render a tiny inline bar chart."""
    filled = int(round(fraction * width))
    return f"{'█' * filled}{'░' * (width - filled)}"


def _rate_color(rate: float) -> str:
    if rate >= 0.95:
        return _GREEN
    if rate >= 0.50:
        return _YELLOW
    return _RED


# ---------------------------------------------------------------------------
# Per-step aggregate
# ---------------------------------------------------------------------------

@dataclass
class StepSummary:
    """Aggregated metrics for one concurrency level."""
    concurrency: int
    rounds: int
    total_requests: int = 0
    ok: int = 0
    errors: int = 0
    timeouts: int = 0
    # latency (successful only)
    durations_ok: list = field(default_factory=list)
    # time-to-first-token (successful only)
    ttfts: list = field(default_factory=list)
    # total output volume
    total_chars: int = 0
    total_chunks: int = 0
    # wall clock for the step (all rounds)
    wall_s: float = 0.0
    # error detail
    error_messages: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.ok / self.total_requests if self.total_requests else 0.0

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.success_rate

    def _safe_stat(self, values: list, fn):
        return fn(values) if values else None

    @property
    def latency_min(self) -> Optional[float]:
        return self._safe_stat(self.durations_ok, min)

    @property
    def latency_max(self) -> Optional[float]:
        return self._safe_stat(self.durations_ok, max)

    @property
    def latency_mean(self) -> Optional[float]:
        return self._safe_stat(self.durations_ok, statistics.mean)

    @property
    def latency_median(self) -> Optional[float]:
        return self._safe_stat(self.durations_ok, statistics.median) if len(self.durations_ok) >= 2 else self.latency_mean

    @property
    def latency_p95(self) -> Optional[float]:
        if not self.durations_ok:
            return None
        s = sorted(self.durations_ok)
        return s[max(0, int(len(s) * 0.95) - 1)]

    @property
    def ttft_mean(self) -> Optional[float]:
        return self._safe_stat(self.ttfts, statistics.mean)

    @property
    def ttft_median(self) -> Optional[float]:
        return self._safe_stat(self.ttfts, statistics.median) if len(self.ttfts) >= 2 else self.ttft_mean

    @property
    def throughput_rps(self) -> float:
        return self.total_requests / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def throughput_cps(self) -> float:
        return self.total_chars / self.wall_s if self.wall_s > 0 else 0.0


def summarise_step(
    concurrency: int,
    rounds: int,
    results: list,
    wall_s: float,
) -> StepSummary:
    s = StepSummary(concurrency=concurrency, rounds=rounds, wall_s=wall_s)
    for r in results:
        s.total_requests += 1
        if r.status == "ok":
            s.ok += 1
            s.durations_ok.append(r.duration_s)
            if r.first_token_s is not None:
                s.ttfts.append(r.first_token_s)
            s.total_chars += r.chars
            s.total_chunks += r.chunks
        elif r.status == "timeout":
            s.timeouts += 1
        else:
            s.errors += 1

        if r.error:
            key = r.error[:120]
            s.error_messages[key] = s.error_messages.get(key, 0) + 1
    return s


# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------

def print_sweep_header(args: argparse.Namespace, steps: list[int]) -> None:
    print()
    print(f"{_BOLD}{'=' * 72}")
    print(f"  LOAD TEST SWEEP — Agent Completion SSE Endpoint")
    print(f"{'=' * 72}{_RESET}")
    print(f"  Base URL             : {args.base_url}")
    print(f"  Agent ID             : {args.agent_id}")
    print(f"  Concurrency steps    : {', '.join(str(s) for s in steps)}")
    print(f"  Rounds per step      : {args.rounds_per_step}")
    print(f"  Max request (s)      : {args.max_request_seconds}")
    print(f"  Prompt               : {args.prompt[:70]}{'...' if len(args.prompt) > 70 else ''}")
    auth = "API Key" if args.api_key else ("JWT" if args.jwt else "None")
    print(f"  Auth                 : {auth}")
    if args.fail_threshold < 1.0:
        print(f"  Early-stop threshold : {args.fail_threshold * 100:.0f}% failure rate")
    print(f"{'=' * 72}")
    print()


def print_step_banner(step_idx: int, total_steps: int, concurrency: int) -> None:
    print(f"{_CYAN}{_BOLD}┌──────────────────────────────────────────────┐")
    print(f"│  Step {step_idx}/{total_steps}  —  {concurrency} concurrent user{'s' if concurrency != 1 else ''}{'  ' * 5}│")
    print(f"└──────────────────────────────────────────────┘{_RESET}")


def print_step_result(s: StepSummary) -> None:
    rc = _rate_color(s.success_rate)
    print(
        f"  {_BOLD}Result:{_RESET}  "
        f"{rc}{s.ok}/{s.total_requests} OK{_RESET}  "
        f"({rc}{s.success_rate * 100:5.1f}%{_RESET})  "
        f"{_bar(s.success_rate, 20)}  "
        f"wall {s.wall_s:.1f}s"
    )
    if s.durations_ok:
        print(
            f"  {_DIM}Latency:{_RESET}  "
            f"mean {s.latency_mean:.2f}s  "
            f"p50 {s.latency_median:.2f}s  "
            f"p95 {s.latency_p95:.2f}s  "
            f"min {s.latency_min:.2f}s  "
            f"max {s.latency_max:.2f}s"
        )
    if s.ttfts:
        print(
            f"  {_DIM}TTFT:   {_RESET}  "
            f"mean {s.ttft_mean:.2f}s  "
            f"median {s.ttft_median:.2f}s"
        )
    if s.error_messages:
        for msg, cnt in sorted(s.error_messages.items(), key=lambda x: -x[1])[:3]:
            print(f"  {_RED}[{cnt}x]{_RESET} {msg[:80]}")
    print()


def _fmt(val, fmt=".2f", suffix="s"):
    return f"{val:{fmt}}{suffix}" if val is not None else "—"


def print_final_report(summaries: list[StepSummary], stopped_early: bool) -> None:
    print()
    print(f"{_BOLD}{'=' * 72}")
    print(f"  FINAL REPORT")
    print(f"{'=' * 72}{_RESET}")

    if stopped_early:
        print(f"\n  {_YELLOW}(Sweep stopped early due to failure-rate threshold){_RESET}")

    # ---- Comparison table ----
    hdr = (
        f"  {'Users':>5s}  "
        f"{'OK':>4s}  "
        f"{'Err':>4s}  "
        f"{'T/O':>4s}  "
        f"{'Pass%':>6s}  "
        f"{'Lat mean':>9s}  "
        f"{'Lat p95':>9s}  "
        f"{'TTFT':>9s}  "
        f"{'Chars':>7s}  "
        f"{'RPS':>6s}  "
        f"{'Bar'}"
    )
    print(f"\n{_BOLD}{hdr}{_RESET}")
    print(f"  {'─' * 68}")

    for s in summaries:
        rc = _rate_color(s.success_rate)
        lat_mean = _fmt(s.latency_mean)
        lat_p95 = _fmt(s.latency_p95)
        ttft = _fmt(s.ttft_mean)
        print(
            f"  {s.concurrency:>5d}  "
            f"{s.ok:>4d}  "
            f"{s.errors:>4d}  "
            f"{s.timeouts:>4d}  "
            f"{rc}{s.success_rate * 100:>5.1f}%{_RESET}  "
            f"{lat_mean:>9s}  "
            f"{lat_p95:>9s}  "
            f"{ttft:>9s}  "
            f"{s.total_chars:>7d}  "
            f"{s.throughput_rps:>5.2f}  "
            f"{_bar(s.success_rate, 15)}"
        )

    # ---- Findings ----
    print(f"\n{_BOLD}  Findings{_RESET}")

    passing = [s for s in summaries if s.success_rate >= 0.95]
    degraded = [s for s in summaries if 0.0 < s.success_rate < 0.95]
    failing = [s for s in summaries if s.success_rate == 0.0 and s.total_requests > 0]

    if passing:
        best = max(passing, key=lambda s: s.concurrency)
        print(f"    {_GREEN}Max stable concurrency  : {best.concurrency} users "
              f"({best.success_rate * 100:.0f}% pass, mean {_fmt(best.latency_mean)}){_RESET}")

    if degraded:
        first_deg = min(degraded, key=lambda s: s.concurrency)
        print(f"    {_YELLOW}Degradation starts at   : {first_deg.concurrency} users "
              f"({first_deg.success_rate * 100:.0f}% pass){_RESET}")

    if failing:
        first_fail = min(failing, key=lambda s: s.concurrency)
        print(f"    {_RED}Total failure at        : {first_fail.concurrency} users{_RESET}")

    # Latency trend
    if len([s for s in summaries if s.latency_mean is not None]) >= 2:
        lats = [(s.concurrency, s.latency_mean) for s in summaries if s.latency_mean is not None]
        if len(lats) >= 2 and lats[-1][1] > lats[0][1] * 1.5:
            print(f"    {_YELLOW}Latency growth          : "
                  f"{lats[0][1]:.2f}s @ {lats[0][0]} users -> "
                  f"{lats[-1][1]:.2f}s @ {lats[-1][0]} users "
                  f"({lats[-1][1] / lats[0][1]:.1f}x){_RESET}")

    if not passing and not degraded:
        print(f"    {_RED}No successful requests at any concurrency level.{_RESET}")
        print(f"    {_DIM}Tip: verify the server is reachable with --steps 1 first.{_RESET}")

    # TTFT trend
    ttft_points = [(s.concurrency, s.ttft_mean) for s in summaries if s.ttft_mean is not None]
    if len(ttft_points) >= 2:
        print(f"\n{_BOLD}  TTFT Trend{_RESET}")
        for conc, t in ttft_points:
            bar = _bar(min(t / max(tp[1] for tp in ttft_points), 1.0), 20)
            print(f"    {conc:>3d} users : {t:.2f}s  {bar}")

    # Error summary
    all_errors: dict[str, int] = {}
    for s in summaries:
        for msg, cnt in s.error_messages.items():
            all_errors[msg] = all_errors.get(msg, 0) + cnt
    if all_errors:
        print(f"\n{_BOLD}  Error Summary{_RESET}")
        for msg, cnt in sorted(all_errors.items(), key=lambda x: -x[1])[:5]:
            print(f"    [{cnt:>4d}x] {msg[:90]}")

    print(f"\n{'=' * 72}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_STEPS = "1,2,4,8,12,16,20,24"


def parse_steps(raw: str) -> list[int]:
    """Parse a comma-separated list of concurrency levels."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    steps = []
    for p in parts:
        if "-" in p and not p.startswith("-"):
            lo, hi = p.split("-", 1)
            steps.extend(range(int(lo), int(hi) + 1))
        else:
            steps.append(int(p))
    return sorted(set(steps))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automated load-test sweep with comparative report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--base-url", required=True,
                    help="Base URL of the API")
    p.add_argument("--agent-id", type=int, required=True,
                    help="Agent ID to target")
    p.add_argument("--jwt", default="",
                    help="JWT token for authentication")
    p.add_argument("--api-key", default="",
                    help="API key (alternative to --jwt)")
    p.add_argument("--steps", default=DEFAULT_STEPS,
                    help=f"Comma-separated concurrency levels to test (default: {DEFAULT_STEPS}). "
                         "Supports ranges like 1-8.")
    p.add_argument("--rounds-per-step", type=int, default=1,
                    help="Rounds to run at each concurrency level (default: 1)")
    p.add_argument("--prompt", default="Hello, what can you help me with?",
                    help="Prompt to send to the agent")
    p.add_argument("--max-request-seconds", type=float, default=120,
                    help="Max seconds per request (default: 120)")
    p.add_argument("--delay-between-steps", type=float, default=3.0,
                    help="Seconds to pause between concurrency steps (default: 3.0)")
    p.add_argument("--delay-between-rounds", type=float, default=1.0,
                    help="Seconds to pause between rounds within a step (default: 1.0)")
    p.add_argument("--fail-threshold", type=float, default=1.0,
                    help="Stop sweep when failure rate exceeds this (0.0–1.0). "
                         "E.g. 0.5 = stop if >50%% fail. Default 1.0 (never stop).")
    p.add_argument("--skip-preflight", action="store_true",
                    help="Skip the initial health-check probe")
    p.add_argument("--verbose", action="store_true",
                    help="Print per-request detail lines")
    p.add_argument("--json-output", default="",
                    help="Optional path to write full results as JSON")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    steps = parse_steps(args.steps)

    if not steps:
        print(f"{_RED}Error: no valid concurrency steps.{_RESET}")
        sys.exit(2)

    if not args.jwt and not args.api_key:
        print(f"{_YELLOW}Warning: No --jwt or --api-key provided. "
              f"Requests will likely fail with 401.{_RESET}")

    print_sweep_header(args, steps)

    # ── Preflight ─────────────────────────────────────────────────────
    if not args.skip_preflight:
        reachable = await preflight_check(args.base_url)
        if not reachable:
            print(f"\n  {_RED}Server is not reachable — aborting sweep.{_RESET}")
            print(f"  {_DIM}Use --skip-preflight to bypass this check.{_RESET}\n")
            sys.exit(2)
        print()

    summaries: list[StepSummary] = []
    all_raw: list[dict] = []
    stopped_early = False
    sweep_t0 = time.monotonic()

    for step_idx, concurrency in enumerate(steps, 1):
        print_step_banner(step_idx, len(steps), concurrency)

        step_results = []
        step_t0 = time.monotonic()

        for rnd in range(1, args.rounds_per_step + 1):
            if args.rounds_per_step > 1:
                print(f"  {_DIM}Round {rnd}/{args.rounds_per_step}{_RESET}")

            round_results = await run_round_with_ticker(
                round_id=rnd,
                users=concurrency,
                base_url=args.base_url,
                agent_id=args.agent_id,
                jwt_token=args.jwt,
                prompt=args.prompt,
                max_request_seconds=args.max_request_seconds,
                api_key=args.api_key,
                ticker_interval=10.0,
            )

            if args.verbose:
                for r in sorted(round_results, key=lambda x: x.user_id):
                    print_result_line(r)

            step_results.extend(round_results)

            if rnd < args.rounds_per_step and args.delay_between_rounds > 0:
                await asyncio.sleep(args.delay_between_rounds)

        step_wall = time.monotonic() - step_t0
        summary = summarise_step(concurrency, args.rounds_per_step, step_results, step_wall)
        summaries.append(summary)
        print_step_result(summary)

        # Collect raw results for JSON export
        for r in step_results:
            all_raw.append(dataclasses.asdict(r))

        # Early-stop check
        if args.fail_threshold < 1.0 and summary.failure_rate > args.fail_threshold:
            print(f"  {_RED}Failure rate {summary.failure_rate * 100:.0f}% exceeds "
                  f"threshold {args.fail_threshold * 100:.0f}% — stopping sweep.{_RESET}\n")
            stopped_early = True
            break

        # Pause between steps
        if step_idx < len(steps) and args.delay_between_steps > 0:
            print(f"  {_DIM}(pausing {args.delay_between_steps:.1f}s before next step){_RESET}\n")
            await asyncio.sleep(args.delay_between_steps)

    sweep_wall = time.monotonic() - sweep_t0
    print(f"{_DIM}  Total sweep time: {sweep_wall:.1f}s{_RESET}")

    print_final_report(summaries, stopped_early)

    # JSON output
    if args.json_output:
        output = {
            "sweep": {
                "base_url": args.base_url,
                "agent_id": args.agent_id,
                "steps": [s.concurrency for s in summaries],
                "rounds_per_step": args.rounds_per_step,
                "max_request_seconds": args.max_request_seconds,
                "prompt": args.prompt,
                "total_wall_s": sweep_wall,
                "stopped_early": stopped_early,
            },
            "summaries": [
                {
                    "concurrency": s.concurrency,
                    "total_requests": s.total_requests,
                    "ok": s.ok,
                    "errors": s.errors,
                    "timeouts": s.timeouts,
                    "success_rate": round(s.success_rate, 4),
                    "latency_mean_s": round(s.latency_mean, 4) if s.latency_mean is not None else None,
                    "latency_median_s": round(s.latency_median, 4) if s.latency_median is not None else None,
                    "latency_p95_s": round(s.latency_p95, 4) if s.latency_p95 is not None else None,
                    "ttft_mean_s": round(s.ttft_mean, 4) if s.ttft_mean is not None else None,
                    "total_chars": s.total_chars,
                    "throughput_rps": round(s.throughput_rps, 4),
                    "wall_s": round(s.wall_s, 2),
                }
                for s in summaries
            ],
            "raw_results": all_raw,
        }
        with open(args.json_output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Full results written to {args.json_output}")

    # Exit code
    any_failures = any(s.failure_rate > 0 for s in summaries)
    if any_failures:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
