#!/usr/bin/env python3
"""
Drosophila Brain — GPU Simulation Demo

Runs all 6 Drosophila experiments with a biophysical HH brain.
Uses Metal GPU on macOS Apple Silicon, CPU fallback elsewhere.

Usage:
    python3 demos/demo_drosophila_gpu.py [--scale N] [--runs R]

Arguments:
    --scale N   Number of neurons (default: 5000)
    --runs  R   Number of episodes per experiment (default: 10)
"""

import argparse
import time
import sys

try:
    from oneuro_metal import DrosophilaSim, has_gpu
except ImportError:
    print("ERROR: oneuro_metal not installed.")
    print("Build with: maturin develop --release")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Drosophila Brain Demo")
    parser.add_argument("--scale", type=int, default=5000, help="Number of neurons")
    parser.add_argument("--runs", type=int, default=10, help="Episodes per experiment")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "olfactory", "phototaxis", "thermotaxis",
                                 "foraging", "drug", "circadian"],
                        help="Which experiment to run")
    args = parser.parse_args()

    gpu_status = "Metal GPU" if has_gpu() else "CPU"
    print(f"{'='*60}")
    print(f"  Drosophila Brain — {gpu_status}")
    print(f"  Scale: {args.scale:,} neurons")
    print(f"  Runs:  {args.runs} episodes per experiment")
    print(f"{'='*60}")
    print()

    # Create simulator
    t0 = time.time()
    sim = DrosophilaSim(args.scale, seed=args.seed)
    print(f"[INIT] {sim}")
    print(f"[INIT] Allocation: {time.time() - t0:.2f}s")
    print()

    if args.experiment == "all":
        t0 = time.time()
        results = sim.run_all()
        elapsed = time.time() - t0

        print(f"{'='*60}")
        print(f"  Results ({elapsed:.1f}s total)")
        print(f"{'='*60}")

        passed = 0
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}")
            print(f"         {r.metric_name} = {r.metric_value:.4f} (threshold: {r.threshold:.4f})")
            print(f"         {r.details}")
            print()
            if r.passed:
                passed += 1

        print(f"{'='*60}")
        print(f"  Score: {passed}/{len(results)} PASS")
        print(f"  Time:  {elapsed:.1f}s ({elapsed/len(results):.1f}s/experiment)")
        print(f"{'='*60}")

    else:
        exp_map = {
            "olfactory": sim.run_olfactory,
            "phototaxis": sim.run_phototaxis,
            "thermotaxis": sim.run_thermotaxis,
            "foraging": sim.run_foraging,
            "drug": sim.run_drug_response,
            "circadian": sim.run_circadian,
        }
        func = exp_map[args.experiment]
        t0 = time.time()
        result = func(args.runs)
        elapsed = time.time() - t0

        print(f"  {result}")
        print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
