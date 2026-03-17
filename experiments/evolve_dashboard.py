#!/usr/bin/env python3
"""
Convergence Visualization Dashboard for Terrarium Evolution Engine.

Reads telemetry JSON, landscape JSON, and Pareto JSON to produce
matplotlib convergence analysis plots.

Usage:
    python evolve_dashboard.py --telemetry telemetry.json --output figures/
    python evolve_dashboard.py --telemetry telemetry.json --landscape landscape.json
    python evolve_dashboard.py --pareto pareto.json --output figures/
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib numpy")

# Parameter names in order (must match WorldGenome.normalized_params())
PARAM_NAMES = [
    'proton_scale', 'temperature', 'water_count', 'water_volume',
    'moisture_scale', 'plant_count', 'fruit_count', 'fly_count',
    'microbe_count', 'resp_vmax', 'nitr_vmax', 'photo_vmax',
    'miner_vmax', 'seed', 'time_warp'
]

# Objective names for Pareto plots
OBJ_NAMES = ['biomass', 'biodiversity', 'stability', 'carbon', 'fruit', 'microbial']


def load_json(path: str) -> Any:
    """Load JSON from file."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_convergence(telemetry: list, output_dir: Path) -> None:
    """Plot fitness convergence curve (best/mean/worst over generations)."""
    if not telemetry:
        print("No telemetry data for convergence plot")
        return

    generations = [r['generation'] for r in telemetry]
    best = [r['best_fitness'] for r in telemetry]
    mean = [r['mean_fitness'] for r in telemetry]
    worst = [r['worst_fitness'] for r in telemetry]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines
    ax.plot(generations, best, 'g-', linewidth=2, label='Best', marker='o', markersize=4)
    ax.plot(generations, mean, 'b--', linewidth=1.5, label='Mean', marker='s', markersize=3)
    ax.plot(generations, worst, 'r:', linewidth=1.5, label='Worst', marker='^', markersize=3)

    # Fill area between best and worst
    ax.fill_between(generations, worst, best, alpha=0.2, color='green')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title('Evolution Convergence', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add mode info if available
    mode = telemetry[0].get('mode', 'unknown')
    ax.text(0.02, 0.98, f'Mode: {mode}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / 'convergence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_diversity(telemetry: list, output_dir: Path) -> None:
    """Plot population diversity over generations."""
    if not telemetry:
        print("No telemetry data for diversity plot")
        return

    generations = [r['generation'] for r in telemetry]
    diversity = [r['population_diversity'] for r in telemetry]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(generations, diversity, 'purple', linewidth=2, marker='o', markersize=4)
    ax.fill_between(generations, diversity, alpha=0.3, color='purple')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Population Diversity (best - worst)', fontsize=12)
    ax.set_title('Population Diversity Over Generations', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Annotate min/max
    min_idx = np.argmin(diversity)
    max_idx = np.argmax(diversity)
    ax.annotate(f'Min: {diversity[min_idx]:.2f}',
                xy=(generations[min_idx], diversity[min_idx]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.annotate(f'Max: {diversity[max_idx]:.2f}',
                xy=(generations[max_idx], diversity[max_idx]),
                xytext=(5, -15), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'diversity.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_param_heatmap(telemetry: list, output_dir: Path) -> None:
    """Plot parameter evolution heatmap (params x generations)."""
    if not telemetry:
        print("No telemetry data for parameter heatmap")
        return

    # Extract parameter matrix
    n_gens = len(telemetry)
    n_params = len(PARAM_NAMES)

    # Check if params exist
    if 'best_genome_params' not in telemetry[0]:
        print("No genome params in telemetry for heatmap")
        return

    data = np.zeros((n_params, n_gens))
    for i, record in enumerate(telemetry):
        params = record.get('best_genome_params', [])
        if len(params) == n_params:
            data[:, i] = params

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_yticks(range(n_params))
    ax.set_yticklabels(PARAM_NAMES)
    ax.set_xticks(range(n_gens))
    ax.set_xticklabels([f'G{r["generation"]}' for r in telemetry], rotation=45, ha='right')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Parameter', fontsize=12)
    ax.set_title('Parameter Evolution Heatmap (Normalized 0-1)', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Value', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'param_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_stress_timeline(telemetry: list, output_dir: Path) -> None:
    """Plot stress recovery timeline if stress metrics present."""
    stress_records = [r for r in telemetry if r.get('stress_metrics')]
    if not stress_records:
        print("No stress metrics in telemetry")
        return

    generations = [r['generation'] for r in stress_records]
    pre_stress = [r['stress_metrics']['pre_stress_biomass'] for r in stress_records]
    min_stress = [r['stress_metrics']['min_stress_biomass'] for r in stress_records]
    post_recovery = [r['stress_metrics']['post_recovery_biomass'] for r in stress_records]

    x = np.arange(len(generations))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, pre_stress, width, label='Pre-stress', color='green', alpha=0.8)
    bars2 = ax.bar(x, min_stress, width, label='Min during stress', color='red', alpha=0.8)
    bars3 = ax.bar(x + width, post_recovery, width, label='Post-recovery', color='blue', alpha=0.8)

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Biomass', fontsize=12)
    ax.set_title('Stress Recovery Timeline (Resilience Tracking)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{g}' for g in generations])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'stress_timeline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_landscape_fdc(landscape: dict, output_dir: Path) -> None:
    """Plot fitness landscape FDC scatter (distance vs fitness)."""
    points = landscape.get('points', [])
    if not points:
        print("No landscape points for FDC plot")
        return

    # Compute distances from best
    best_fitness = max(p['fitness'] for p in points)
    best_params = next(p['best_genome_params'] for p in points if p['fitness'] == best_fitness)

    distances = []
    fitnesses = []
    for p in points:
        params = p.get('best_genome_params', p.get('genome_params', []))
        if params:
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(params, best_params)))
            distances.append(dist)
            fitnesses.append(p['fitness'])

    if not distances:
        print("No valid points for FDC plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(distances, fitnesses, c=fitnesses, cmap='viridis', alpha=0.6, s=50)

    ax.set_xlabel('Distance from Best Genome', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title('Fitness Landscape (FDC Analysis)', fontsize=14)

    # Add correlation annotation
    fdc = landscape.get('fitness_distance_correlation', 0)
    ruggedness = landscape.get('ruggedness', 0)
    ax.text(0.98, 0.02, f'FDC: {fdc:.3f}\nRuggedness: {ruggedness:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.colorbar(sc, ax=ax, label='Fitness')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'landscape_fdc.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pareto_projections(pareto: dict, output_dir: Path) -> None:
    """Plot Pareto front 2D projections."""
    front = pareto.get('pareto_front', [])
    if not front:
        print("No Pareto front solutions")
        return

    # Extract objective values
    objectives = []
    for sol in front:
        fitness = sol.get('fitness', {})
        obj = [fitness.get(name, 0) for name in OBJ_NAMES]
        objectives.append(obj)

    objectives = np.array(objectives)

    # Find which objectives have variance
    active = []
    for i, name in enumerate(OBJ_NAMES):
        if objectives[:, i].std() > 1e-6:
            active.append(i)

    if len(active) < 2:
        print("Not enough active objectives for Pareto plot")
        return

    # Create pairwise scatter plots
    n_active = len(active)
    n_plots = n_active * (n_active - 1) // 2

    if n_plots == 0:
        return

    # Determine grid size
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    plot_idx = 0
    for i, idx_a in enumerate(active):
        for idx_b in active[i + 1:]:
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            ax.scatter(objectives[:, idx_a], objectives[:, idx_b],
                      c='gold', edgecolors='black', s=80, alpha=0.8)

            ax.set_xlabel(OBJ_NAMES[idx_a], fontsize=10)
            ax.set_ylabel(OBJ_NAMES[idx_b], fontsize=10)
            ax.set_title(f'{OBJ_NAMES[idx_a]} vs {OBJ_NAMES[idx_b]}', fontsize=11)
            ax.grid(True, alpha=0.3)

            plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f'Pareto Front Projections (n={len(front)} solutions)', fontsize=14)
    plt.tight_layout()
    output_path = output_dir / 'pareto_projections.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_multi_objective_radar(telemetry: list, output_dir: Path) -> None:
    """Plot radar chart for multi-objective fitness if available."""
    records_with_mof = [r for r in telemetry if r.get('multi_objective_fitness')]
    if not records_with_mof:
        print("No multi-objective fitness in telemetry")
        return

    # Take last 5 generations or all if fewer
    records = records_with_mof[-5:] if len(records_with_mof) > 5 else records_with_mof

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(OBJ_NAMES), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    colors = plt.cm.viridis(np.linspace(0, 1, len(records)))

    for i, record in enumerate(records):
        mof = record['multi_objective_fitness']
        values = [mof.get(name, 0) for name in OBJ_NAMES]
        values += values[:1]  # Close the loop

        ax.plot(angles, values, 'o-', linewidth=1.5, color=colors[i],
                label=f"Gen {record['generation']}")
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(OBJ_NAMES)
    ax.set_title('Multi-Objective Fitness (Pareto)', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    output_path = output_dir / 'multi_objective_radar.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_report(telemetry: list, landscape: dict, pareto: dict, output_dir: Path) -> None:
    """Generate a text summary report."""
    report_lines = ["# Evolution Summary Report\n"]

    if telemetry:
        report_lines.append("## Telemetry Summary\n")
        report_lines.append(f"- Total generations: {len(telemetry)}")
        report_lines.append(f"- Best fitness: {max(r['best_fitness'] for r in telemetry):.4f}")
        report_lines.append(f"- Final best: {telemetry[-1]['best_fitness']:.4f}")
        report_lines.append(f"- Final mean: {telemetry[-1]['mean_fitness']:.4f}")
        report_lines.append(f"- Final diversity: {telemetry[-1]['population_diversity']:.4f}")
        if telemetry[0].get('mode'):
            report_lines.append(f"- Mode: {telemetry[0]['mode']}")
        report_lines.append("")

    if landscape:
        report_lines.append("## Landscape Analysis\n")
        report_lines.append(f"- Points evaluated: {landscape.get('total_worlds_evaluated', 'N/A')}")
        report_lines.append(f"- FDC correlation: {landscape.get('fitness_distance_correlation', 0):.4f}")
        report_lines.append(f"- Ruggedness: {landscape.get('ruggedness', 0):.4f}")
        report_lines.append("")

    if pareto:
        report_lines.append("## Pareto Front\n")
        front = pareto.get('pareto_front', [])
        report_lines.append(f"- Front size: {len(front)}")
        report_lines.append(f"- Generations run: {pareto.get('generations_run', 'N/A')}")
        report_lines.append(f"- Total worlds evaluated: {pareto.get('total_worlds_evaluated', 'N/A')}")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    output_path = output_dir / 'summary_report.md'
    with open(output_path, 'w') as f:
        f.write(report_text)
    print(f"Saved: {output_path}")


def main():
    if not HAS_MPL:
        print("Error: matplotlib is required. Install with: pip install matplotlib numpy")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='Convergence Visualization Dashboard for Terrarium Evolution Engine'
    )
    parser.add_argument('--telemetry', help='Telemetry JSON path')
    parser.add_argument('--landscape', help='Landscape JSON path')
    parser.add_argument('--pareto', help='Pareto result JSON path')
    parser.add_argument('--output', default='figures', help='Output directory (default: figures)')

    args = parser.parse_args()

    if not args.telemetry and not args.landscape and not args.pareto:
        parser.print_help()
        print("\nError: At least one input file is required")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry = None
    landscape = None
    pareto = None

    if args.telemetry:
        print(f"Loading telemetry from {args.telemetry}...")
        telemetry = load_json(args.telemetry)
        plot_convergence(telemetry, output_dir)
        plot_diversity(telemetry, output_dir)
        plot_param_heatmap(telemetry, output_dir)
        plot_stress_timeline(telemetry, output_dir)
        plot_multi_objective_radar(telemetry, output_dir)

    if args.landscape:
        print(f"Loading landscape from {args.landscape}...")
        landscape = load_json(args.landscape)
        plot_landscape_fdc(landscape, output_dir)

    if args.pareto:
        print(f"Loading Pareto from {args.pareto}...")
        pareto = load_json(args.pareto)
        plot_pareto_projections(pareto, output_dir)

    generate_summary_report(telemetry or [], landscape or {}, pareto or {}, output_dir)

    print(f"\nDashboard complete! Output saved to {output_dir}/")


if __name__ == '__main__':
    main()
