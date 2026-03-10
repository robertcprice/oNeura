#!/usr/bin/env python3
"""Normalize checked-in 25K A100 Spatial Arena artifacts for comparison/reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_RESULTS = Path("papers/data/results_doom_full_large.json")
DEFAULT_NEURONS = 25_250
DEFAULT_SYNAPSES = 6_975_250


def _normalize_task_profile(
    label: str,
    experiment_id: str,
    experiment: Dict[str, Any],
    *,
    dt_ms: float,
    neural_steps_per_action_lower: int,
    neurons: int,
    synapses: int,
    source_path: Path,
) -> Dict[str, Any]:
    total_steps = sum(item["steps"] for item in experiment.get("episode_results", []))
    wall_time_s = experiment["time"]
    actions_per_s = total_steps / wall_time_s if wall_time_s else None
    bio_time_ms_lower = total_steps * neural_steps_per_action_lower * dt_ms if total_steps else None
    realtime_factor_lower = (
        (bio_time_ms_lower / 1000.0) / wall_time_s if bio_time_ms_lower is not None and wall_time_s else None
    )
    note = (
        "Historical A100 task-loop artifact. realtime_factor is a lower bound computed from "
        f"{neural_steps_per_action_lower} neural steps/action at dt={dt_ms} ms and excludes extra feedback steps."
    )
    return {
        "meta": {
            "backend": "cuda",
            "platform": "historical_a100_artifact",
            "source_file": str(source_path),
            "gpu_label": "NVIDIA A100 SXM4",
            "experiment_id": experiment_id,
            "label": label,
        },
        "result": {
            "target": "doom_arena_large",
            "measurement_kind": "historical_task_loop_lower_bound",
            "device_type": "cuda",
            "neurons": neurons,
            "synapses": synapses,
            "steps": total_steps or None,
            "bio_time_ms": bio_time_ms_lower,
            "wall_time_s": wall_time_s,
            "steps_per_s": actions_per_s,
            "realtime_factor": realtime_factor_lower,
            "note": note,
        },
    }


def _normalize_suite_total(
    payload: Dict[str, Any],
    *,
    neurons: int,
    synapses: int,
    source_path: Path,
) -> Dict[str, Any]:
    note = (
        "Historical A100 task-suite wall time only. Experiment 3 does not include per-episode step counts, "
        "so no biological-time or realtime-factor estimate is emitted for the full suite."
    )
    return {
        "meta": {
            "backend": "cuda",
            "platform": "historical_a100_artifact",
            "source_file": str(source_path),
            "gpu_label": "NVIDIA A100 SXM4",
            "label": "suite_total",
        },
        "result": {
            "target": "doom_arena_large_suite",
            "measurement_kind": "historical_task_suite_wall_time",
            "device_type": "cuda",
            "neurons": neurons,
            "synapses": synapses,
            "steps": None,
            "bio_time_ms": None,
            "wall_time_s": payload["total_time"],
            "steps_per_s": None,
            "realtime_factor": None,
            "note": note,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the checked-in 25K A100 Spatial Arena artifact")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="Path to results_doom_full_large.json")
    parser.add_argument("--dt-ms", type=float, default=0.1, help="Biological dt used for lower-bound estimates")
    parser.add_argument(
        "--neural-steps-per-action-lower",
        type=int,
        default=20,
        help="Lower-bound neural steps per arena action used in the realtime estimate",
    )
    parser.add_argument("--neurons", type=int, default=DEFAULT_NEURONS, help="Neuron count for the historical run")
    parser.add_argument("--synapses", type=int, default=DEFAULT_SYNAPSES, help="Synapse count for the historical run")
    parser.add_argument(
        "--experiment",
        choices=["1", "2", "suite"],
        default=None,
        help="Emit a single normalized profile instead of the whole bundle",
    )
    parser.add_argument("--json", default=None, help="Optional output path")
    args = parser.parse_args()

    source_path = Path(args.results)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    bundle = {
        "navigation": _normalize_task_profile(
            "navigation",
            "1",
            payload["experiments"]["1"],
            dt_ms=args.dt_ms,
            neural_steps_per_action_lower=args.neural_steps_per_action_lower,
            neurons=args.neurons,
            synapses=args.synapses,
            source_path=source_path,
        ),
        "threat_avoidance": _normalize_task_profile(
            "threat_avoidance",
            "2",
            payload["experiments"]["2"],
            dt_ms=args.dt_ms,
            neural_steps_per_action_lower=args.neural_steps_per_action_lower,
            neurons=args.neurons,
            synapses=args.synapses,
            source_path=source_path,
        ),
        "suite_total": _normalize_suite_total(
            payload,
            neurons=args.neurons,
            synapses=args.synapses,
            source_path=source_path,
        ),
    }
    if args.experiment == "1":
        output: Dict[str, Any] = bundle["navigation"]
    elif args.experiment == "2":
        output = bundle["threat_avoidance"]
    elif args.experiment == "suite":
        output = bundle["suite_total"]
    else:
        output = bundle

    print(json.dumps(output, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
