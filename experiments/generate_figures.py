#!/usr/bin/env python3
"""
Publication-Quality Figure Generator for Go/No-Go Benchmark

Creates:
1. Learning curves with 95% CI bands
2. Per-seed improvement scatter plots
3. Condition comparison bar charts
4. LaTeX-ready summary tables
"""

import json
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, ASCII-only mode")


@dataclass
class ConditionSummary:
    name: str
    n_seeds: int
    pre_accuracy: float
    post_accuracy: float
    improvement: float
    sd: float
    ci_lower: float
    ci_upper: float
    improvements: List[float]


def load_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results JSON."""
    with open(filepath) as f:
        return json.load(f)


def extract_summaries(results: Dict) -> Dict[str, ConditionSummary]:
    """Extract condition summaries from results."""
    summaries = {}
    for name, data in results.get('summaries', {}).items():
        if name.startswith('_'):
            continue
        summaries[name] = ConditionSummary(
            name=name,
            n_seeds=data.get('n_seeds', 0),
            pre_accuracy=data.get('mean_pre_accuracy', 0),
            post_accuracy=data.get('mean_post_accuracy', 0),
            improvement=data.get('mean_improvement', 0),
            sd=data.get('sd_improvement', 0),
            ci_lower=data.get('ci_95_lower', 0),
            ci_upper=data.get('ci_95_upper', 0),
            improvements=data.get('improvements', [])
        )
    return summaries


def extract_trial_data(results: Dict, condition: str) -> np.ndarray:
    """Extract trial-by-trial accuracy for a condition."""
    trials_list = []
    for r in results.get('results', []):
        if r.get('condition') == condition and r.get('success'):
            trials = [t['correct'] for t in r.get('trials', [])]
            if trials:
                trials_list.append(trials)
    return np.array(trials_list)


def compute_learning_curve(trials: np.ndarray, window: int = 10) -> tuple:
    """Compute rolling accuracy with confidence bands."""
    if len(trials) == 0:
        return np.array([]), np.array([]), np.array([])

    # Mean accuracy per trial
    mean_per_trial = np.mean(trials, axis=0)

    # Rolling average
    kernel = np.ones(window) / window
    rolling = np.convolve(mean_per_trial, kernel, mode='valid')

    # Standard error
    se = np.std(trials[:, :len(rolling)], axis=0) / np.sqrt(len(trials))

    return np.arange(len(rolling)), rolling, se


def plot_learning_curves(results: Dict, output_path: str = None):
    """Create learning curve figure."""
    if not HAS_MATPLOTLIB:
        return

    conditions = list(set(r['condition'] for r in results.get('results', [])))
    colors = {'full_learning': '#2E86AB', 'no_dopamine': '#E94F37',
              'nmda_block': '#8B5CF6', 'reward_shuffle': '#F5B041',
              'anti_correlated': '#E74C3C'}

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in conditions:
        trials = extract_trial_data(results, cond)
        if len(trials) > 0:
            x, y, se = compute_learning_curve(trials)
            color = colors.get(cond, '#333333')
            ax.plot(x, y, label=cond.replace('_', ' ').title(), color=color, linewidth=2)
            ax.fill_between(x, y - 1.96*se, y + 1.96*se, alpha=0.2, color=color)

    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel('Accuracy (rolling 10-trial window)', fontsize=12)
    ax.set_title('Go/No-Go Learning Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_condition_comparison(summaries: Dict[str, ConditionSummary], output_path: str = None):
    """Create bar chart comparing conditions."""
    if not HAS_MATPLOTLIB:
        return

    names = [s.name.replace('_', ' ').title() for s in summaries.values()]
    improvements = [s.improvement for s in summaries.values()]
    ci_lower = [s.ci_lower - s.improvement for s in summaries.values()]
    ci_upper = [s.ci_upper - s.improvement for s in summaries.values()]

    colors = ['#2E86AB', '#E94F37', '#8B5CF6', '#F5B041', '#E74C3C']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, improvements, color=colors[:len(names)], alpha=0.8)

    # Error bars (use symmetric error for negative improvements)
    yerr_lower = [min(abs(ci), abs(im)) for ci, im in zip(ci_lower, improvements)]
    yerr_upper = [min(abs(ci), abs(im)) for ci, im in zip(ci_upper, improvements)]
    ax.errorbar(range(len(names)), improvements,
                yerr=[yerr_lower, yerr_upper],
                fmt='none', color='black', capsize=5)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Mean Improvement (Δ accuracy)', fontsize=12)
    ax.set_title('Condition Comparison: Pre to Post Training', fontsize=14, fontweight='bold')
    ax.set_ylim(min(ci_lower) - 0.1, max(ci_upper) + 0.15)

    # Value labels
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:+.1%}', ha='center', fontsize=10)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_per_seed_scatter(summaries: Dict[str, ConditionSummary], output_path: str = None):
    """Create per-seed improvement scatter plot."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2E86AB', '#E94F37', '#8B5CF6', '#F5B041', '#E74C3C']
    offsets = np.linspace(-0.15, 0.15, len(summaries))

    for i, (name, summary) in enumerate(summaries.items()):
        x = np.arange(len(summary.improvements)) + offsets[i]
        ax.scatter(x, summary.improvements, label=name.replace('_', ' ').title(),
                   color=colors[i], s=80, alpha=0.7)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Seed Index', fontsize=12)
    ax.set_ylabel('Improvement (Δ accuracy)', fontsize=12)
    ax.set_title('Per-Seed Improvement by Condition', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def generate_latex_table(summaries: Dict[str, ConditionSummary]) -> str:
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Go/No-Go Learning Benchmark Results}",
        r"\label{tab:gonogo}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Condition & Pre & Post & $\Delta$ & 95\% CI \\",
        r"\midrule"
    ]

    for s in summaries.values():
        pre = f"{s.pre_accuracy:.2f}"
        post = f"{s.post_accuracy:.2f}"
        delta = f"{s.improvement:+.2f}"
        ci = f"[{s.ci_lower:.2f}, {s.ci_upper:.2f}]"
        name = s.name.replace('_', ' ').title()
        lines.append(f"{name} & {pre} & {post} & {delta} & {ci} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    return '\n'.join(lines)


def generate_ascii_summary(summaries: Dict[str, ConditionSummary]) -> str:
    """Generate ASCII summary table."""
    lines = [
        "=" * 75,
        "GO/NO-GO BENCHMARK SUMMARY",
        "=" * 75,
        f"{'Condition':<20} {'Pre':>8} {'Post':>8} {'Δ':>10} {'95% CI':>20} {'n':>4}",
        "-" * 75
    ]

    for s in summaries.values():
        name = s.name.replace('_', ' ').title()[:19]
        ci = f"[{s.ci_lower:+.2f}, {s.ci_upper:+.2f}]"
        lines.append(f"{name:<20} {s.pre_accuracy:>7.1%} {s.post_accuracy:>7.1%} {s.improvement:>+9.1%} {ci:>20} {s.n_seeds:>4}")

    lines.append("=" * 75)
    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_figures.py <results.json> [output_dir]")
        print("\nExample:")
        print("  python generate_figures.py results/go_no_go_confirmatory_50seeds.json figures/")
        sys.exit(1)

    filepath = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('figures')

    output_dir.mkdir(exist_ok=True)

    print(f"Loading: {filepath}")
    results = load_results(filepath)

    summaries = extract_summaries(results)

    # Generate ASCII summary
    print("\n" + generate_ascii_summary(summaries))

    # Generate LaTeX table
    print("\n--- LaTeX Table ---\n")
    print(generate_latex_table(summaries))

    # Generate figures
    base = Path(filepath).stem

    if HAS_MATPLOTLIB:
        plot_learning_curves(results, str(output_dir / f"{base}_learning_curves.png"))
        plot_condition_comparison(summaries, str(output_dir / f"{base}_comparison.png"))
        plot_per_seed_scatter(summaries, str(output_dir / f"{base}_scatter.png"))
    else:
        print("\nSkipping figures (matplotlib not available)")

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
