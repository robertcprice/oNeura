#!/usr/bin/env python3
"""
Doom Brain — GPU Simulation Demo

Runs embodied and/or disembodied Doom FPS experiments with a biophysical
brain. Uses Metal GPU on macOS Apple Silicon, CPU fallback elsewhere.

Usage:
    python3 demos/demo_doom_brain_gpu.py [--scale N] [--mode MODE]

Arguments:
    --scale N     Number of neurons (default: 5000)
    --mode MODE   "embodied", "disembodied", or "both" (default: both)
"""

import argparse
import time
import sys

try:
    from oneuro_metal import DoomBrainSim, has_gpu
except ImportError:
    print("ERROR: oneuro_metal not installed.")
    print("Build with: maturin develop --release")
    sys.exit(1)


def run_mode(mode_name: str, n_neurons: int, n_episodes: int = 10, seed: int = 42):
    """Run all Doom experiments in a given mode."""
    print(f"\n{'─'*50}")
    print(f"  Mode: {mode_name}")
    print(f"{'─'*50}")

    t0 = time.time()
    sim = DoomBrainSim(n_neurons, mode=mode_name, seed=seed)
    print(f"  [INIT] {sim}")
    print(f"  [INIT] Allocation: {time.time() - t0:.2f}s")

    t0 = time.time()
    results = sim.run_all(episodes_per_experiment=n_episodes)
    elapsed = time.time() - t0

    passed = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}")
        print(f"         {r.metric_name} = {r.metric_value:.4f} (threshold: {r.threshold:.4f})")
        print(f"         {r.details}")
        if r.passed:
            passed += 1

    print(f"\n  Score: {passed}/{len(results)} | Time: {elapsed:.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser(description="Doom Brain Demo")
    parser.add_argument("--scale", type=int, default=5000, help="Number of neurons")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["embodied", "disembodied", "both"],
                        help="Operating mode")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per experiment")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    gpu_status = "Metal GPU" if has_gpu() else "CPU"
    print(f"{'='*60}")
    print(f"  Doom Brain — {gpu_status}")
    print(f"  Scale: {args.scale:,} neurons")
    print(f"  Mode:  {args.mode}")
    print(f"{'='*60}")

    all_results = []

    if args.mode in ("disembodied", "both"):
        results = run_mode("disembodied", args.scale, args.episodes, args.seed)
        all_results.extend(results)

    if args.mode in ("embodied", "both"):
        results = run_mode("embodied", args.scale, args.episodes, args.seed)
        all_results.extend(results)

    # Summary
    total_pass = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    print(f"\n{'='*60}")
    print(f"  Overall: {total_pass}/{total} PASS")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
