#!/usr/bin/env python3
"""Profile the PyTorch CUDA/MPS backend with optional NVIDIA telemetry."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (str(SRC_ROOT), str(SCRIPT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from gpu_telemetry import NvidiaSmiSampler

try:
    import torch
    from oneuro.molecular.cuda_backend import CUDAMolecularBrain, CUDARegionalBrain, detect_backend
except ImportError as exc:
    print(f"ERROR: failed to import CUDA backend: {exc}", file=sys.stderr)
    print("Run from the repo root or set PYTHONPATH=src.", file=sys.stderr)
    sys.exit(1)


def _device_type(device: Any) -> str:
    if hasattr(device, "type"):
        return str(device.type)
    return str(device).split(":", 1)[0]


def _synchronize(device_type: str) -> None:
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _peak_torch_memory(device_type: str) -> Dict[str, Any]:
    if device_type != "cuda" or not torch.cuda.is_available():
        return {
            "device_name": None,
            "peak_allocated_gb": None,
            "peak_reserved_gb": None,
        }

    props = torch.cuda.get_device_properties(0)
    return {
        "device_name": torch.cuda.get_device_name(0),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "total_memory_gb": props.total_memory / 1e9,
    }


def _peak_rss_gb() -> float:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return peak_rss / 1e9
    return (peak_rss * 1024) / 1e9


def _profile_molecular(args: argparse.Namespace) -> Dict[str, Any]:
    brain = CUDAMolecularBrain(args.neurons, device=args.device, dt=args.dt_ms)
    if args.random_synapses:
        brain.add_random_synapses(args.random_synapses)

    device_type = _device_type(brain.device)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.warmup_steps:
        brain.run(args.warmup_steps)
        _synchronize(device_type)

    sampler = NvidiaSmiSampler(interval_s=max(args.interval_ms, 100) / 1000.0)
    if device_type == "cuda":
        sampler.start()

    start_steps = brain.step_count
    _synchronize(device_type)
    start_time = time.perf_counter()
    brain.run(args.steps)
    _synchronize(device_type)
    elapsed_s = time.perf_counter() - start_time
    sampler.stop()

    steps_run = brain.step_count - start_steps
    bio_time_ms = steps_run * brain.dt
    return {
        "target": "molecular",
        "measurement_kind": "raw_neural_step_profile",
        "device_type": device_type,
        "neurons": brain.n,
        "synapses": brain.n_synapses,
        "steps": steps_run,
        "dt_ms": brain.dt,
        "bio_time_ms": bio_time_ms,
        "wall_time_s": elapsed_s,
        "steps_per_s": steps_run / elapsed_s if elapsed_s > 0 else 0.0,
        "realtime_factor": (bio_time_ms / 1000.0) / elapsed_s if elapsed_s > 0 else 0.0,
        "process_peak_rss_gb": _peak_rss_gb(),
        "torch_memory": _peak_torch_memory(device_type),
        "gpu_telemetry": sampler.report(include_samples=args.include_samples),
    }


def _profile_regional(args: argparse.Namespace) -> Dict[str, Any]:
    rb = CUDARegionalBrain._build(
        n_columns=args.columns,
        n_per_layer=args.n_per_layer,
        device=args.device,
        seed=args.seed,
    )
    device_type = _device_type(rb.brain.device)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.warmup_steps:
        rb.run(args.warmup_steps)
        _synchronize(device_type)

    sampler = NvidiaSmiSampler(interval_s=max(args.interval_ms, 100) / 1000.0)
    if device_type == "cuda":
        sampler.start()

    start_steps = rb.brain.step_count
    _synchronize(device_type)
    start_time = time.perf_counter()
    rb.run(args.steps)
    _synchronize(device_type)
    elapsed_s = time.perf_counter() - start_time
    sampler.stop()

    steps_run = rb.brain.step_count - start_steps
    bio_time_ms = steps_run * rb.brain.dt
    return {
        "target": "regional",
        "measurement_kind": "raw_neural_step_profile",
        "device_type": device_type,
        "columns": args.columns,
        "n_per_layer": args.n_per_layer,
        "neurons": rb.brain.n,
        "synapses": rb.brain.n_synapses,
        "steps": steps_run,
        "dt_ms": rb.brain.dt,
        "bio_time_ms": bio_time_ms,
        "wall_time_s": elapsed_s,
        "steps_per_s": steps_run / elapsed_s if elapsed_s > 0 else 0.0,
        "realtime_factor": (bio_time_ms / 1000.0) / elapsed_s if elapsed_s > 0 else 0.0,
        "process_peak_rss_gb": _peak_rss_gb(),
        "torch_memory": _peak_torch_memory(device_type),
        "gpu_telemetry": sampler.report(include_samples=args.include_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the PyTorch oNeuro backend")
    parser.add_argument("--target", default="regional", choices=["molecular", "regional"])
    parser.add_argument("--device", default="auto", help="Requested torch device: auto/cuda/mps/cpu")
    parser.add_argument("--neurons", type=int, default=25_000, help="Neuron count for molecular runs")
    parser.add_argument("--random-synapses", type=int, default=0, help="Random synapse count for molecular runs")
    parser.add_argument("--columns", type=int, default=250, help="Cortical columns for regional runs")
    parser.add_argument("--n-per-layer", type=int, default=20, help="Neurons per cortical layer")
    parser.add_argument("--steps", type=int, default=100, help="Timed steps")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps before timing")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for regional builds")
    parser.add_argument("--dt-ms", type=float, default=0.1, help="Biological timestep in milliseconds")
    parser.add_argument("--interval-ms", type=int, default=500, help="nvidia-smi sample interval")
    parser.add_argument("--include-samples", action="store_true", help="Include raw telemetry samples in JSON")
    parser.add_argument("--json", default=None, help="Optional path to write JSON output")
    args = parser.parse_args()

    meta = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "global_backend_hint": detect_backend(),
        "requested_device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }

    if args.target == "molecular":
        result = _profile_molecular(args)
    else:
        result = _profile_regional(args)

    payload = {"meta": meta, "result": result}
    print(json.dumps(payload, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()
