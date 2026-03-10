#!/usr/bin/env python3
"""Derive real-time metrics from demo_dishbrain_pong.py JSON output."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

DEFAULT_DT_MS = 0.1
DEFAULT_FRAMES_PER_RALLY = 12
DEFAULT_STIM_STEPS = 30
DEFAULT_SETTLE_STEPS = 5
DEFAULT_HIT_FEEDBACK_STEPS = 50
DEFAULT_MISS_FEEDBACK_STEPS = 100


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_div(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def analyze_exp1(payload: Dict[str, Any], run_index: int, args: argparse.Namespace) -> Dict[str, Any]:
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Input JSON does not contain any runs")
    if run_index < 0 or run_index >= len(runs):
        raise ValueError(f"run_index {run_index} is out of range for {len(runs)} runs")

    run = runs[run_index]
    experiments = run.get("experiments", {})
    exp1 = experiments.get("1")
    if not isinstance(exp1, dict):
        raise ValueError("Input JSON does not contain experiment 1")

    system = payload.get("system", {})
    result: Dict[str, Any] = {
        "source_json": str(args.input_json),
        "run_index": run_index,
        "seed": run.get("seed"),
        "experiment": "1",
        "experiment_name": exp1.get("name", "DishBrain Pong Replication"),
        "status": "ok" if "outcomes" in exp1 else "error",
        "scale": payload.get("scale"),
        "benchmark_mode": exp1.get("benchmark_mode", payload.get("benchmark_mode", False)),
        "no_interval_biology": exp1.get(
            "no_interval_biology",
            payload.get("no_interval_biology", False),
        ),
        "requested_device": payload.get("device"),
        "platform": system.get("platform"),
        "python": system.get("python"),
        "torch": system.get("torch"),
        "backend": system.get("backend"),
        "gpu": system.get("gpu"),
        "gpu_memory_gb": system.get("gpu_memory_gb"),
        "passed": exp1.get("passed"),
        "error": exp1.get("error"),
        "assumptions": {
            "dt_ms": args.dt_ms,
            "frames_per_rally": args.frames_per_rally,
            "stim_steps": args.stim_steps,
            "settle_steps": args.settle_steps,
            "hit_feedback_steps": args.hit_feedback_steps,
            "miss_feedback_steps": args.miss_feedback_steps,
        },
    }

    outcomes = exp1.get("outcomes")
    if not isinstance(outcomes, list):
        result["wall_time_s"] = payload.get("total_time_s")
        result["note"] = (
            "Experiment 1 did not complete, so derived real-time metrics are unavailable."
        )
        return result

    rallies = len(outcomes)
    hits = sum(int(x) for x in outcomes)
    misses = rallies - hits
    rally_frame_steps = args.frames_per_rally * (args.stim_steps + args.settle_steps)
    neural_steps = (
        rallies * rally_frame_steps
        + hits * args.hit_feedback_steps
        + misses * args.miss_feedback_steps
    )
    bio_time_ms = neural_steps * args.dt_ms
    wall_time_s = exp1.get("time")
    realtime_factor = _safe_div(bio_time_ms / 1000.0, wall_time_s)
    slower_than_real_time = _safe_div(wall_time_s, bio_time_ms / 1000.0)

    result.update(
        {
            "wall_time_s": wall_time_s,
            "rallies": rallies,
            "hits": hits,
            "misses": misses,
            "total_hitrate": exp1.get("total_hitrate"),
            "first_10": exp1.get("first_10"),
            "last_10": exp1.get("last_10"),
            "neural_steps": neural_steps,
            "bio_time_ms": bio_time_ms,
            "realtime_factor": realtime_factor,
            "slower_than_real_time": slower_than_real_time,
            "rallies_per_s": _safe_div(rallies, wall_time_s),
            "frames_per_s": _safe_div(rallies * args.frames_per_rally, wall_time_s),
            "neural_steps_per_s": _safe_div(neural_steps, wall_time_s),
            "note": (
                "Derived from the fixed Exp 1 Pong loop in demos/demo_dishbrain_pong.py."
            ),
        }
    )
    if result["benchmark_mode"]:
        result["note"] += " Benchmark mode disables interval biology and uses reduced-overhead CUDA stepping."
    return result


def analyze_rust_result(payload: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    wall_time_s = payload.get("wall_time_s")
    neural_steps = payload.get("neural_steps")
    bio_time_ms = payload.get("bio_time_ms")
    realtime_factor = payload.get("realtime_factor")
    slower_than_real_time = payload.get("slowdown_vs_realtime")
    if slower_than_real_time is None:
        slower_than_real_time = _safe_div(wall_time_s, (bio_time_ms or 0) / 1000.0)

    gpu_label = "Apple Metal" if payload.get("gpu_available") else None
    backend = "rust-metal" if payload.get("gpu_dispatch_active") else "rust-cpu"

    result: Dict[str, Any] = {
        "source_json": str(args.input_json),
        "run_index": 0,
        "seed": payload.get("seed"),
        "experiment": "1",
        "experiment_name": "DishBrain Pong Replication",
        "status": "ok",
        "scale": payload.get("scale"),
        "benchmark_mode": payload.get("benchmark_mode", False),
        "no_interval_biology": payload.get("no_interval_biology", payload.get("benchmark_mode", False)),
        "requested_device": "metal" if payload.get("gpu_available") else "cpu",
        "platform": None,
        "python": None,
        "torch": None,
        "backend": backend,
        "gpu": gpu_label,
        "gpu_memory_gb": None,
        "passed": payload.get("passed"),
        "error": payload.get("gpu_init_error"),
        "assumptions": {
            "dt_ms": args.dt_ms,
            "frames_per_rally": args.frames_per_rally,
            "stim_steps": args.stim_steps,
            "settle_steps": args.settle_steps,
            "hit_feedback_steps": args.hit_feedback_steps,
            "miss_feedback_steps": args.miss_feedback_steps,
        },
        "wall_time_s": wall_time_s,
        "rallies": payload.get("rallies"),
        "hits": payload.get("total_hits"),
        "misses": None
        if payload.get("rallies") is None or payload.get("total_hits") is None
        else int(payload["rallies"]) - int(payload["total_hits"]),
        "total_hitrate": payload.get("total_hitrate"),
        "first_10": payload.get("first_10"),
        "last_10": payload.get("last_10"),
        "neural_steps": neural_steps,
        "bio_time_ms": bio_time_ms,
        "realtime_factor": realtime_factor,
        "slower_than_real_time": slower_than_real_time,
        "rallies_per_s": _safe_div(payload.get("rallies"), wall_time_s),
        "frames_per_s": _safe_div(
            (payload.get("rallies") or 0) * args.frames_per_rally,
            wall_time_s,
        ),
        "neural_steps_per_s": _safe_div(neural_steps, wall_time_s),
        "note": (
            "Measured from the Rust/Metal DishBrain Pong runner in oneuro-metal/src/bin/"
            "dishbrain_pong.rs using direct neural step counts."
        ),
    }
    if payload.get("benchmark_mode"):
        result["note"] += " Latency benchmark mode disables nonessential interval biology."
    return result


def _print_summary(result: Dict[str, Any]) -> None:
    print(f"Source:   {result['source_json']}")
    print(
        f"System:   scale={result.get('scale')} device={result.get('requested_device')} "
        f"backend={result.get('backend')} gpu={result.get('gpu')}"
    )
    print(f"Status:   {result['status']}")

    error = result.get("error")
    if error:
        short_error = str(error).splitlines()[0]
        print(f"Error:    {short_error}")

    wall_time_s = result.get("wall_time_s")
    if wall_time_s is not None:
        print(f"Wall:     {wall_time_s:.3f}s")

    if result["status"] != "ok":
        note = result.get("note")
        if note:
            print(f"Note:     {note}")
        return

    print(
        f"Rallies:  {result['rallies']} total, {result['hits']} hits, {result['misses']} misses"
    )
    print(
        f"Bio:      {result['bio_time_ms']:.1f} ms "
        f"({result['neural_steps']} neural steps at dt={result['assumptions']['dt_ms']} ms)"
    )
    print(f"Speed:    {result['realtime_factor']:.6f}x real time")
    print(f"Slowdown: {result['slower_than_real_time']:.2f}x slower than real time")
    print(
        f"Throughput: {result['rallies_per_s']:.3f} rallies/s, "
        f"{result['frames_per_s']:.3f} frames/s, "
        f"{result['neural_steps_per_s']:.1f} neural steps/s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path, help="Path to demo_dishbrain_pong.py JSON output")
    parser.add_argument("--run-index", type=int, default=0, help="Run index inside the JSON payload")
    parser.add_argument("--dt-ms", type=float, default=DEFAULT_DT_MS)
    parser.add_argument("--frames-per-rally", type=int, default=DEFAULT_FRAMES_PER_RALLY)
    parser.add_argument("--stim-steps", type=int, default=DEFAULT_STIM_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--hit-feedback-steps", type=int, default=DEFAULT_HIT_FEEDBACK_STEPS)
    parser.add_argument("--miss-feedback-steps", type=int, default=DEFAULT_MISS_FEEDBACK_STEPS)
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write analysis JSON")
    args = parser.parse_args()

    payload = _load_json(args.input_json)
    if "runs" in payload:
        result = analyze_exp1(payload, args.run_index, args)
    elif "neural_steps" in payload and "realtime_factor" in payload:
        result = analyze_rust_result(payload, args)
    else:
        raise ValueError(
            "Unsupported input JSON shape: expected demo_dishbrain_pong.py output or "
            "oneuro-metal DishBrainPongResult output"
        )
    _print_summary(result)

    if args.json_out:
        serializable = {
            key: _round(value) if isinstance(value, float) and not math.isnan(value) else value
            for key, value in result.items()
        }
        with args.json_out.open("w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)
            fh.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
