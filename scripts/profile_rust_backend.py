#!/usr/bin/env python3
"""Profile the Rust/Metal backend with consistent timing output."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from typing import Any, Dict

try:
    from oneuro_metal import (
        DrosophilaSim,
        DoomBrainSim,
        MolecularBrain,
        RegionalBrain,
        has_gpu,
        version,
    )
except ImportError:
    print("ERROR: oneuro_metal is not installed.", file=sys.stderr)
    print("Build it first with: cd oneuro-metal && maturin develop --release", file=sys.stderr)
    sys.exit(1)


DEFAULT_DT_MS = 0.1


def _backend_name() -> str:
    return "metal" if has_gpu() else "cpu"


def _gpu_note(instance_gpu_flag: bool) -> str | None:
    if has_gpu() and not instance_gpu_flag:
        return (
            "Metal is available on this machine, but this brain instance did not "
            "report an initialized GPU context. Treat the timing as possible CPU fallback."
        )
    return None


def _gpu_context_initialized(brain: Any) -> bool:
    return bool(getattr(brain, "gpu_active", lambda: False)())


def _gpu_dispatch_active(brain: Any) -> bool | None:
    getter = getattr(brain, "gpu_dispatch_active", None)
    if getter is None:
        return None
    return bool(getter())


def _gpu_init_error(brain: Any) -> str | None:
    getter = getattr(brain, "gpu_init_error", None)
    if getter is None:
        return None
    return getter()


def _peak_rss_gb() -> float:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return peak_rss / 1e9
    return (peak_rss * 1024) / 1e9


def _profile_molecular(args: argparse.Namespace) -> Dict[str, Any]:
    brain = MolecularBrain(args.neurons, dt=args.dt_ms)
    if args.warmup_steps:
        brain.run(args.warmup_steps)

    start_steps = brain.step_count
    start_time = time.perf_counter()
    brain.run(args.steps)
    elapsed_s = time.perf_counter() - start_time
    steps_run = brain.step_count - start_steps
    bio_time_ms = steps_run * brain.dt
    return {
        "target": "molecular",
        "measurement_kind": "raw_neural_step_profile",
        "neurons": brain.n_neurons,
        "synapses": brain.n_synapses,
        "steps": steps_run,
        "dt_ms": brain.dt,
        "bio_time_ms": bio_time_ms,
        "wall_time_s": elapsed_s,
        "steps_per_s": steps_run / elapsed_s if elapsed_s > 0 else 0.0,
        "realtime_factor": (bio_time_ms / 1000.0) / elapsed_s if elapsed_s > 0 else 0.0,
        "process_peak_rss_gb": _peak_rss_gb(),
        "gpu_context_initialized": _gpu_context_initialized(brain),
        "gpu_dispatch_active": _gpu_dispatch_active(brain),
        "gpu_init_error": _gpu_init_error(brain),
        "note": _gpu_note(_gpu_context_initialized(brain)),
    }


def _profile_regional(args: argparse.Namespace) -> Dict[str, Any]:
    brain = RegionalBrain.with_columns(args.columns, seed=args.seed)
    if args.warmup_steps:
        brain.run(args.warmup_steps)

    start_bio_time_ms = brain.time
    start_time = time.perf_counter()
    brain.run(args.steps)
    elapsed_s = time.perf_counter() - start_time
    bio_time_ms = brain.time - start_bio_time_ms
    steps_run = int(round(bio_time_ms / brain.dt)) if brain.dt else 0
    return {
        "target": "regional",
        "measurement_kind": "raw_neural_step_profile",
        "columns": args.columns,
        "neurons": brain.n_neurons,
        "synapses": brain.n_synapses,
        "steps": steps_run,
        "dt_ms": brain.dt,
        "bio_time_ms": bio_time_ms,
        "wall_time_s": elapsed_s,
        "steps_per_s": steps_run / elapsed_s if elapsed_s > 0 else 0.0,
        "realtime_factor": (bio_time_ms / 1000.0) / elapsed_s if elapsed_s > 0 else 0.0,
        "process_peak_rss_gb": _peak_rss_gb(),
        "gpu_context_initialized": _gpu_context_initialized(brain),
        "gpu_dispatch_active": _gpu_dispatch_active(brain),
        "gpu_init_error": _gpu_init_error(brain),
        "note": _gpu_note(_gpu_context_initialized(brain)),
    }


def _profile_doom(args: argparse.Namespace) -> Dict[str, Any]:
    sim = DoomBrainSim(args.neurons, mode=args.mode, seed=args.seed)
    start_steps = sim.step_count
    start_time = time.perf_counter()
    result = sim.run_navigation(args.episodes)
    elapsed_s = time.perf_counter() - start_time
    steps_run = sim.step_count - start_steps
    bio_time_ms = steps_run * args.dt_ms
    return {
        "target": "doom_navigation",
        "measurement_kind": "task_loop_profile",
        "mode": args.mode,
        "neurons": sim.n_neurons,
        "episodes": args.episodes,
        "steps": steps_run,
        "dt_ms_assumed": args.dt_ms,
        "bio_time_ms": bio_time_ms,
        "wall_time_s": elapsed_s,
        "steps_per_s": steps_run / elapsed_s if elapsed_s > 0 else 0.0,
        "realtime_factor": (bio_time_ms / 1000.0) / elapsed_s if elapsed_s > 0 else 0.0,
        "process_peak_rss_gb": _peak_rss_gb(),
        "metal_available": has_gpu(),
        "result": {
            "name": result.name,
            "passed": result.passed,
            "metric_name": result.metric_name,
            "metric_value": result.metric_value,
            "threshold": result.threshold,
            "details": result.details,
        },
    }


def _profile_drosophila(args: argparse.Namespace) -> Dict[str, Any]:
    sim = DrosophilaSim(args.neurons, seed=args.seed)
    runners = {
        "olfactory": sim.run_olfactory,
        "phototaxis": sim.run_phototaxis,
        "thermotaxis": sim.run_thermotaxis,
        "foraging": sim.run_foraging,
        "drug": sim.run_drug_response,
        "circadian": sim.run_circadian,
    }

    start_steps = sim.step_count
    start_time = time.perf_counter()
    if args.experiment == "all":
        results = sim.run_all()
        payload: Any = [
            {
                "name": r.name,
                "passed": r.passed,
                "metric_name": r.metric_name,
                "metric_value": r.metric_value,
                "threshold": r.threshold,
                "details": r.details,
            }
            for r in results
        ]
    else:
        result = runners[args.experiment](args.episodes)
        payload = {
            "name": result.name,
            "passed": result.passed,
            "metric_name": result.metric_name,
            "metric_value": result.metric_value,
            "threshold": result.threshold,
            "details": result.details,
        }
    elapsed_s = time.perf_counter() - start_time
    steps_run = sim.step_count - start_steps
    bio_time_ms = steps_run * args.dt_ms
    return {
        "target": "drosophila",
        "measurement_kind": "task_loop_profile",
        "experiment": args.experiment,
        "neurons": sim.n_neurons,
        "episodes": args.episodes if args.experiment != "all" else None,
        "steps": steps_run,
        "dt_ms_assumed": args.dt_ms,
        "bio_time_ms": bio_time_ms,
        "wall_time_s": elapsed_s,
        "steps_per_s": steps_run / elapsed_s if elapsed_s > 0 else 0.0,
        "realtime_factor": (bio_time_ms / 1000.0) / elapsed_s if elapsed_s > 0 else 0.0,
        "process_peak_rss_gb": _peak_rss_gb(),
        "metal_available": has_gpu(),
        "result": payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the Rust/Metal oNeuro backend")
    parser.add_argument(
        "--target",
        default="molecular",
        choices=["molecular", "regional", "doom", "drosophila"],
        help="Which Rust-backed workload to profile",
    )
    parser.add_argument("--neurons", type=int, default=25_000, help="Neuron count where applicable")
    parser.add_argument("--columns", type=int, default=250, help="Cortical columns for RegionalBrain")
    parser.add_argument("--steps", type=int, default=1_000, help="Raw simulation steps for molecular/regional runs")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps before timing")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes for Doom/Drosophila experiments")
    parser.add_argument("--mode", default="disembodied", choices=["embodied", "disembodied"], help="DoomBrainSim mode")
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["all", "olfactory", "phototaxis", "thermotaxis", "foraging", "drug", "circadian"],
        help="Drosophila experiment to run",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--dt-ms", type=float, default=DEFAULT_DT_MS, help="Biological dt assumption for experiment runners")
    parser.add_argument("--json", default=None, help="Optional path to write JSON output")
    args = parser.parse_args()

    meta = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "backend": _backend_name(),
        "metal_available": has_gpu(),
        "oneuro_metal_version": version(),
    }

    if args.target == "molecular":
        result = _profile_molecular(args)
    elif args.target == "regional":
        result = _profile_regional(args)
    elif args.target == "doom":
        result = _profile_doom(args)
    else:
        result = _profile_drosophila(args)

    payload = {"meta": meta, "result": result}

    print(json.dumps(payload, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()
