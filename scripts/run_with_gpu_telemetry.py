#!/usr/bin/env python3
"""Run an arbitrary command while sampling GPU telemetry with nvidia-smi."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpu_telemetry import NvidiaSmiSampler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command with GPU telemetry capture")
    parser.add_argument("--interval-ms", type=int, default=500, help="Sampling interval for nvidia-smi")
    parser.add_argument("--json", default=None, help="Optional path to write JSON output")
    parser.add_argument("--include-samples", action="store_true", help="Include raw per-sample telemetry in JSON")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run, prefixed with --")
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command to run; use -- <command> [args...]")

    sampler = NvidiaSmiSampler(interval_s=max(args.interval_ms, 100) / 1000.0)
    sampler.start()
    start_time = time.time()
    start_perf = time.perf_counter()
    proc = subprocess.run(command)
    elapsed_s = time.perf_counter() - start_perf
    end_time = time.time()
    sampler.stop()

    payload = {
        "command": command,
        "returncode": proc.returncode,
        "started_at_unix_s": start_time,
        "ended_at_unix_s": end_time,
        "wall_time_s": elapsed_s,
        "gpu_telemetry": sampler.report(include_samples=args.include_samples),
    }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    else:
        print(json.dumps(payload, indent=2))

    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
