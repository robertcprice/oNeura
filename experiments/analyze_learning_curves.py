#!/usr/bin/env python3
"""
Learning Curve Analysis for Go/No-Go Benchmark

Generates publication-ready visualizations of learning dynamics.
"""

import json
import sys
from pathlib import Path
import numpy as np

def load_results(filepath):
    """Load benchmark results JSON."""
    with open(filepath) as f:
        return json.load(f)

def extract_trial_data(results, condition):
    """Extract trial-by-trial accuracy for a condition."""
    trials_list = []
    for r in results['results']:
        if r['condition'] == condition and r['success']:
            trials = [t['correct'] for t in r['trials']]
            trials_list.append(trials)
    return np.array(trials_list)

def compute_rolling_accuracy(trials_array, window=10):
    """Compute rolling accuracy with confidence bands."""
    mean_curve = np.mean(trials_array, axis=0)

    # Rolling average
    kernel = np.ones(window) / window
    rolling = np.convolve(mean_curve, kernel, mode='valid')

    # Standard error per window
    se_curve = []
    for i in range(len(rolling)):
        window_data = trials_array[:, i:i+window]
        se = np.std(window_data) / np.sqrt(len(window_data.flatten()))
        se_curve.append(se)
    se_curve = np.array(se_curve)

    return rolling, se_curve

def print_ascii_learning_curve(rolling, se, condition, width=60, height=15):
    """Print ASCII art learning curve."""
    n_points = len(rolling)

    # Normalize to 0-1 range
    min_val, max_val = 0.3, 1.0  # Expected accuracy range

    print(f"\n{condition} Learning Curve:")
    print("─" * (width + 10))

    for row in range(height, 0, -1):
        threshold = min_val + (max_val - min_val) * row / height
        line = f"{threshold:.0%} │"

        for i in range(0, n_points, max(1, n_points // width)):
            val = rolling[i] if i < len(rolling) else rolling[-1]
            if val >= threshold:
                line += "█"
            elif val >= threshold - 0.03:
                line += "▄"
            else:
                line += " "

        line += " │"
        print(line)

    print("0%  └" + "─" * width + "┘")
    print("    Trial →")
    print(f"    (Window size = 10 trials)")

def generate_summary_table(results):
    """Generate publication-ready summary table."""
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Condition':<20} {'Pre':>8} {'Post':>8} {'Δ':>8} {'95% CI':>20}")
    print("-"*70)

    for cond, s in results['summaries'].items():
        if not cond.startswith('_'):
            ci = f"[{s['ci_95_lower']:+.1%}, {s['ci_95_upper']:+.1%}]"
            print(f"{cond:<20} {s['mean_pre_accuracy']:>7.1%} {s['mean_post_accuracy']:>7.1%} {s['mean_improvement']:>+7.1%} {ci:>20}")

    print("="*70)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_learning_curves.py <results.json>")
        print("\nExample:")
        print("  python analyze_learning_curves.py results/go_no_go_standard_v2.json")
        sys.exit(1)

    filepath = sys.argv[1]
    results = load_results(filepath)

    print("="*70)
    print("LEARNING CURVE ANALYSIS")
    print("="*70)
    print(f"File: {filepath}")
    print(f"Conditions: {list(set(r['condition'] for r in results['results']))}")

    # Generate summary table
    generate_summary_table(results)

    # Generate learning curves for each condition
    conditions = list(set(r['condition'] for r in results['results']))

    for condition in conditions:
        trials = extract_trial_data(results, condition)
        if len(trials) > 0:
            rolling, se = compute_rolling_accuracy(trials)
            print_ascii_learning_curve(rolling, se, condition)

    # Print per-seed improvement pattern
    print("\n" + "="*70)
    print("PER-SEED IMPROVEMENT PATTERNS")
    print("="*70)

    for condition in conditions:
        improvements = []
        for r in results['results']:
            if r['condition'] == condition and r['success']:
                improvements.append(r['improvement'])

        if improvements:
            print(f"\n{condition}:")
            print(f"  Seeds: {[f'{x:+.2f}' for x in improvements]}")
            print(f"  Mean: {np.mean(improvements):+.2f}")
            print(f"  SD: {np.std(improvements):.2f}")
            print(f"  Min: {min(improvements):+.2f}")
            print(f"  Max: {max(improvements):+.2f}")

            # Count positive/negative
            n_pos = sum(1 for x in improvements if x > 0)
            n_zero = sum(1 for x in improvements if abs(x) < 0.01)
            n_neg = sum(1 for x in improvements if x < 0)
            print(f"  Positive/Zero/Negative: {n_pos}/{n_zero}/{n_neg}")

if __name__ == "__main__":
    main()
