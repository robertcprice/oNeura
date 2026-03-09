#!/usr/bin/env python3
"""Go/No-Go Benchmark for oNeuro Basal Ganglia Learning.

This benchmark tests whether the basal ganglia D1/D2 pathways can learn
a simple Go/No-Go discrimination task through dopamine-modulated STDP.

## Calibration Status (2024-03)

After the first pilot showed no learning signal, this version includes
calibrated parameters:

1. **Stimulation intensity**: 50-60 uA/cm² (up from 35-45)
2. **Pulse frequency**: Every step (up from every other step)
3. **Trial count**: 120 (up from 80)
4. **Trial duration**: 80 steps (up from 60)
5. **Decision margin**: Configurable threshold (not just D1 > D2)

## Conditions

- `full_learning`: Normal dopamine signaling
- `no_dopamine`: Zero dopamine (tests reward-gated learning)
- `nmda_block`: NMDA receptors blocked (tests STDP mechanism)
- `reward_shuffle`: Random reward timing (tests contingency)
- `anti_correlated`: Punish correct, reward incorrect (tests if learning is active)

## Usage

    PYTHONPATH=src python3 experiments/go_no_go_benchmark.py \
        --conditions full_learning no_dopamine \
        --n-seeds 10 \
        --scale minimal \
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, "src")

from oneuro.molecular.brain_regions import RegionalBrain, _connect_layers
from oneuro.molecular.ion_channels import IonChannelType


# =============================================================================
# CALIBRATION PARAMETERS
# =============================================================================

# Stimulation intensity (uA/cm²) - INCREASED from 35-45
STIM_INTENSITY_SMALL = 55.0  # For <500 neurons
STIM_INTENSITY_LARGE = 65.0  # For >=500 neurons

# Trial structure - INCREASED
N_TRIALS = 120  # Up from 80
TRIAL_STEPS = 80  # Up from 60
WARMUP_STEPS = 300  # Up from 200

# Decision parameters
DECISION_MARGIN = 0.0  # D1 must exceed D2 by this ratio (0.0 = any difference)

# Dopamine levels
DA_REWARD = 2.0  # Up from 1.5
DA_PUNISHMENT = -0.8  # Up from -0.5

# Connection probability
CONN_P_SMALL = 0.5  # Up from 0.4
CONN_P_LARGE = 0.6  # Up from 0.5


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrialResult:
    """Result from a single trial."""
    trial: int
    stimulus: str  # "GREEN" or "RED"
    expected: str  # "Go" or "NoGo"
    decision: str  # "Go" or "NoGo"
    correct: bool
    d1_spikes: int
    d2_spikes: int
    d1_weight_delta: float
    d2_weight_delta: float


@dataclass
class ConditionResult:
    """Aggregated results for one condition."""
    condition: str
    seed: int
    trials: List[TrialResult]
    pre_accuracy: float  # First 20 trials
    post_accuracy: float  # Last 20 trials
    improvement: float  # post - pre
    aligned_weight_delta: float  # D1 - D2 weight change
    d1_total_weight_delta: float
    d2_total_weight_delta: float


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    conditions: List[str]
    n_seeds: int
    scale: str
    workers: int
    stim_intensity: float
    n_trials: int
    trial_steps: int
    decision_margin: float


# =============================================================================
# CORE BENCHMARK LOGIC
# =============================================================================

def build_brain(scale: str, seed: int) -> RegionalBrain:
    """Build a RegionalBrain at the requested scale."""
    if scale == "xlarge":
        return RegionalBrain.xlarge(seed=seed)
    elif scale == "large":
        return RegionalBrain.large(seed=seed)
    elif scale == "standard":
        return RegionalBrain.standard(seed=seed)
    else:
        return RegionalBrain.minimal(seed=seed)


def stimulate_subset(
    network, neuron_ids: List[int], pattern: List[float], intensity: float,
) -> None:
    """Inject current into a subset of neurons based on pattern values."""
    for i, nid in enumerate(neuron_ids):
        val = pattern[i % len(pattern)] if pattern else 0.0
        if val > 0.3:
            network._external_currents[nid] = (
                network._external_currents.get(nid, 0.0) + val * intensity
            )


def apply_ablation(brain: RegionalBrain, condition: str) -> None:
    """Apply condition-specific ablations to the brain."""
    net = brain.network

    if condition == "no_dopamine":
        # Override dopamine release to do nothing
        original_release = net.release_dopamine
        def noop_release(*args, **kwargs):
            pass
        net.release_dopamine = noop_release
        net._original_release_dopamine = original_release

    elif condition == "nmda_block":
        # Block NMDA receptors
        for neuron in net._molecular_neurons.values():
            nmda = neuron.membrane.channels.get_channel(IonChannelType.NMDA)
            if nmda is not None:
                nmda.conductance_scale *= 0.05  # 95% block

    elif condition == "reward_shuffle":
        # Will be handled in trial loop - random reward timing
        pass

    elif condition == "anti_correlated":
        # Will be handled in trial loop - inverted rewards
        pass


def remove_ablation(brain: RegionalBrain, condition: str) -> None:
    """Remove condition-specific ablations."""
    net = brain.network

    if condition == "no_dopamine":
        if hasattr(net, '_original_release_dopamine'):
            net.release_dopamine = net._original_release_dopamine

    elif condition == "nmda_block":
        # Restore NMDA (note: this is approximate)
        for neuron in net._molecular_neurons.values():
            nmda = neuron.membrane.channels.get_channel(IonChannelType.NMDA)
            if nmda is not None:
                nmda.conductance_scale /= 0.05


def run_single_condition(
    condition: str,
    seed: int,
    scale: str,
    config: BenchmarkConfig,
) -> ConditionResult:
    """Run one condition with one seed."""

    # Build brain
    brain = build_brain(scale, seed)
    net = brain.network

    # Get neuron groups
    relay = brain.thalamus.get_ids("relay")
    d1_ids = brain.basal_ganglia.get_ids("D1")
    d2_ids = brain.basal_ganglia.get_ids("D2")
    d1_set = set(d1_ids)
    d2_set = set(d2_ids)

    n_neurons = len(net._molecular_neurons)
    stim_intensity = config.stim_intensity

    # Add thalamostriatal projections
    half = len(relay) // 2
    conn_p = CONN_P_SMALL if n_neurons < 500 else CONN_P_LARGE
    _connect_layers(net, relay[:half], d1_ids, p=conn_p, nt="glutamate")
    _connect_layers(net, relay[half:], d2_ids, p=conn_p, nt="glutamate")

    # Record initial weights for aligned pathway
    relay_first_half = set(relay[:half])
    relay_second_half = set(relay[half:])
    d1_set_weights = set(d1_ids)
    d2_set_weights = set(d2_ids)

    def get_pathway_weights() -> Tuple[float, float]:
        """Get total weight for thalamus->D1 and thalamus->D2 pathways."""
        d1_total = 0.0
        d2_total = 0.0
        for syn in net._molecular_synapses.values():
            if syn.pre_neuron in relay_first_half and syn.post_neuron in d1_set_weights:
                d1_total += syn.weight
            elif syn.pre_neuron in relay_second_half and syn.post_neuron in d2_set_weights:
                d2_total += syn.weight
        return d1_total, d2_total

    d1_weight_initial, d2_weight_initial = get_pathway_weights()

    # Warmup
    for s in range(WARMUP_STEPS):
        if s % 4 == 0:
            brain.stimulate_thalamus(intensity=15.0)
        net.step(0.1)

    # Define stimulus patterns
    green_pattern = [1.0] * half + [0.0] * (len(relay) - half)  # Left-half -> D1
    red_pattern = [0.0] * half + [1.0] * (len(relay) - half)    # Right-half -> D2

    # Apply ablation
    apply_ablation(brain, condition)

    # Run trials
    trials: List[TrialResult] = []
    rng = np.random.RandomState(seed + 1000)

    for trial in range(config.n_trials):
        is_green = trial % 2 == 0
        pattern = green_pattern if is_green else red_pattern
        expected = "Go" if is_green else "NoGo"

        # Record pre-trial weights
        d1_w_pre, d2_w_pre = get_pathway_weights()

        # Present stimulus - CALIBRATED: every step, not every other
        d1_spikes = 0
        d2_spikes = 0
        for s in range(config.trial_steps):
            # PULSED EVERY STEP for stronger stimulation
            stimulate_subset(net, relay, pattern, intensity=stim_intensity)
            net.step(0.1)
            d1_spikes += len(net.last_fired & d1_set)
            d2_spikes += len(net.last_fired & d2_set)

        # Decision with configurable margin
        spike_margin = config.decision_margin * (d1_spikes + d2_spikes)
        decision = "Go" if d1_spikes > d2_spikes + spike_margin else "NoGo"
        is_correct = decision == expected

        # Reward/punishment
        if condition == "reward_shuffle":
            # Random reward timing - breaks contingency
            if rng.random() < 0.5:
                net.release_dopamine(DA_REWARD if is_green else DA_PUNISHMENT)
            else:
                net.release_dopamine(rng.choice([DA_REWARD, DA_PUNISHMENT, 0.0]))
        elif condition == "anti_correlated":
            # Inverted rewards - punish correct, reward incorrect
            # This should cause active learning of the WRONG association
            if is_correct:
                net.release_dopamine(DA_PUNISHMENT)
            else:
                net.release_dopamine(DA_REWARD)
        else:
            if is_green:
                net.release_dopamine(DA_REWARD)
            else:
                net.release_dopamine(DA_PUNISHMENT)

        net.apply_reward_modulated_plasticity()
        net.update_eligibility_traces(dt=1.0)

        # Record post-trial weights
        d1_w_post, d2_w_post = get_pathway_weights()

        trials.append(TrialResult(
            trial=trial,
            stimulus="GREEN" if is_green else "RED",
            expected=expected,
            decision=decision,
            correct=is_correct,
            d1_spikes=d1_spikes,
            d2_spikes=d2_spikes,
            d1_weight_delta=d1_w_post - d1_w_pre,
            d2_weight_delta=d2_w_post - d2_w_pre,
        ))

    # Remove ablation
    remove_ablation(brain, condition)

    # Compute final weights
    d1_weight_final, d2_weight_final = get_pathway_weights()

    # Compute metrics
    pre_acc = sum(t.correct for t in trials[:20]) / 20
    post_acc = sum(t.correct for t in trials[-20:]) / 20
    improvement = post_acc - pre_acc
    d1_weight_delta = d1_weight_final - d1_weight_initial
    d2_weight_delta = d2_weight_final - d2_weight_initial
    aligned_delta = d1_weight_delta - d2_weight_delta  # Positive = Go pathway strengthened

    return ConditionResult(
        condition=condition,
        seed=seed,
        trials=trials,
        pre_accuracy=pre_acc,
        post_accuracy=post_acc,
        improvement=improvement,
        aligned_weight_delta=aligned_delta,
        d1_total_weight_delta=d1_weight_delta,
        d2_total_weight_delta=d2_weight_delta,
    )


def run_condition_worker(args: Tuple) -> Dict:
    """Worker function for multiprocessing."""
    condition, seed, scale, config_dict = args
    config = BenchmarkConfig(**config_dict)

    try:
        result = run_single_condition(condition, seed, scale, config)
        return {
            "condition": result.condition,
            "seed": result.seed,
            "pre_accuracy": result.pre_accuracy,
            "post_accuracy": result.post_accuracy,
            "improvement": result.improvement,
            "aligned_weight_delta": result.aligned_weight_delta,
            "d1_total_weight_delta": result.d1_total_weight_delta,
            "d2_total_weight_delta": result.d2_total_weight_delta,
            "trials": [
                {
                    "trial": t.trial,
                    "stimulus": t.stimulus,
                    "expected": t.expected,
                    "decision": t.decision,
                    "correct": t.correct,
                    "d1_spikes": t.d1_spikes,
                    "d2_spikes": t.d2_spikes,
                }
                for t in result.trials
            ],
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "condition": condition,
            "seed": seed,
            "success": False,
            "error": str(e),
        }


def bootstrap_ci(data: List[float], n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Returns (lower, upper) bounds of the CI.
    """
    if len(data) < 2:
        return (0.0, 0.0)

    rng = np.random.RandomState(42)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(bootstrap_means, alpha * 100))
    upper = float(np.percentile(bootstrap_means, (1.0 - alpha) * 100))
    return (lower, upper)


def summarize_results(results: List[Dict]) -> Dict:
    """Summarize benchmark results across conditions."""

    # Group by condition
    by_condition: Dict[str, List[Dict]] = {}
    for r in results:
        cond = r["condition"]
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(r)

    summaries = {}
    for cond, cond_results in by_condition.items():
        improvements = [r["improvement"] for r in cond_results if r.get("success")]
        aligned_deltas = [r["aligned_weight_delta"] for r in cond_results if r.get("success")]
        pre_accs = [r["pre_accuracy"] for r in cond_results if r.get("success")]
        post_accs = [r["post_accuracy"] for r in cond_results if r.get("success")]

        # Compute bootstrap 95% CI for improvement
        ci_lower, ci_upper = bootstrap_ci(improvements)

        # Compute standard error
        se = np.std(improvements, ddof=1) / np.sqrt(len(improvements)) if len(improvements) > 1 else 0.0

        summaries[cond] = {
            "n_seeds": len(improvements),
            "mean_pre_accuracy": np.mean(pre_accs) if pre_accs else 0.0,
            "mean_post_accuracy": np.mean(post_accs) if post_accs else 0.0,
            "mean_improvement": np.mean(improvements) if improvements else 0.0,
            "sd_improvement": np.std(improvements, ddof=1) if len(improvements) > 1 else 0.0,
            "se_improvement": se,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "mean_aligned_weight_delta": np.mean(aligned_deltas) if aligned_deltas else 0.0,
            "improvements": improvements,
        }

    # Paired contrasts (full_learning vs others)
    if "full_learning" in summaries:
        full_imps = summaries["full_learning"]["improvements"]
        contrasts = {}
        for cond, summary in summaries.items():
            if cond == "full_learning":
                continue
            other_imps = summary["improvements"]
            if len(full_imps) == len(other_imps):
                paired_diffs = [f - o for f, o in zip(full_imps, other_imps)]
                mean_diff = np.mean(paired_diffs)

                # Bootstrap CI for paired difference
                ci_lower, ci_upper = bootstrap_ci(paired_diffs)

                # Determine if CI excludes 0 (statistically significant)
                significant = ci_lower > 0 or ci_upper < 0

                contrasts[f"full_learning - {cond}"] = {
                    "mean_improvement_diff": float(mean_diff),
                    "ci_95_lower": float(ci_lower),
                    "ci_95_upper": float(ci_upper),
                    "significant": bool(significant),
                    "interpretation": "full_learning better" if mean_diff > 0.05 else "negligible",
                }
        summaries["_contrasts"] = contrasts

    return summaries


def print_summary(summaries: Dict) -> None:
    """Print human-readable summary."""

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Condition table with confidence intervals
    print("\n### Condition Summaries\n")
    print(f"| {'Condition':<20} | {'Pre Acc':>10} | {'Post Acc':>10} | {'Improvement':>12} | {'95% CI':>20} | {'Aligned ΔW':>12} |")
    print("|" + "-" * 22 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 22 + "|" + "-" * 14 + "|")

    for cond in sorted(summaries.keys()):
        if cond.startswith("_"):
            continue
        s = summaries[cond]
        ci_str = f"[{s['ci_95_lower']:+.3f}, {s['ci_95_upper']:+.3f}]"
        print(f"| {cond:<20} | {s['mean_pre_accuracy']:>10.3f} | {s['mean_post_accuracy']:>10.3f} | "
              f"{s['mean_improvement']:>+12.3f} | {ci_str:>20} | "
              f"{s['mean_aligned_weight_delta']:>+12.4f} |")

    # Contrasts with CIs and significance
    if "_contrasts" in summaries:
        print("\n### Paired Contrasts Against full_learning\n")
        print(f"| {'Contrast':<35} | {'Mean Δ':>10} | {'95% CI':>20} | {'Sig':>4} |")
        print("|" + "-" * 37 + "|" + "-" * 12 + "|" + "-" * 22 + "|" + "-" * 6 + "|")
        for contrast, data in summaries["_contrasts"].items():
            ci_str = f"[{data['ci_95_lower']:+.3f}, {data['ci_95_upper']:+.3f}]"
            sig = "✓" if data["significant"] else " "
            print(f"| {contrast:<35} | {data['mean_improvement_diff']:>+10.3f} | {ci_str:>20} | {sig:>4} |")

    # Per-seed improvements for full_learning
    if "full_learning" in summaries:
        print("\n### Per-Seed Improvement Pattern (full_learning)\n")
        imps = summaries["full_learning"]["improvements"]
        print(f"```text\n{[round(x, 2) for x in imps]}\n```")

    # Verdict
    print("\n### Interpretation\n")
    if "full_learning" in summaries:
        s = summaries["full_learning"]
        full_imp = s["mean_improvement"]
        ci_excludes_zero = s["ci_95_lower"] > 0

        if full_imp > 0.05:
            sig_marker = " (significant)" if ci_excludes_zero else ""
            print(f"✅ **POSITIVE SIGNAL**: full_learning shows meaningful improvement (>5%){sig_marker}")
            if "_contrasts" in summaries:
                all_significant = all(
                    c["significant"] and c["mean_improvement_diff"] > 0
                    for c in summaries["_contrasts"].values()
                )
                if all_significant:
                    print("✅ **SIGNIFICANT SEPARATION**: full_learning significantly exceeds all ablations (CI excludes 0)")
                else:
                    some_sig = sum(1 for c in summaries["_contrasts"].values() if c["significant"] and c["mean_improvement_diff"] > 0)
                    total = len(summaries["_contrasts"])
                    print(f"⚠️ **PARTIAL SEPARATION**: {some_sig}/{total} contrasts are significant")
        elif full_imp > 0.0:
            print("⚠️ **WEAK SIGNAL**: full_learning shows small improvement (<5%)")
            print("   Consider further calibration: longer training, higher intensity")
        else:
            print("❌ **NO SIGNAL**: full_learning did not improve over baseline")
            print("   Calibration needed: stimulation regime, decision rule, or task design")


def main():
    parser = argparse.ArgumentParser(description="Go/No-Go Benchmark for oNeuro")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["full_learning", "no_dopamine", "nmda_block", "reward_shuffle", "anti_correlated"],
        choices=["full_learning", "no_dopamine", "nmda_block", "reward_shuffle", "anti_correlated"],
        help="Conditions to test",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of paired seeds per condition",
    )
    parser.add_argument(
        "--scale",
        choices=["minimal", "standard", "large", "xlarge"],
        default="minimal",
        help="Brain scale",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--stim-intensity",
        type=float,
        default=None,
        help="Override stimulation intensity",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override number of trials",
    )
    args = parser.parse_args()

    # Determine stimulation intensity based on scale
    if args.stim_intensity is not None:
        stim_intensity = args.stim_intensity
    else:
        test_brain = build_brain(args.scale, 42)
        n_neurons = len(test_brain.network._molecular_neurons)
        stim_intensity = STIM_INTENSITY_SMALL if n_neurons < 500 else STIM_INTENSITY_LARGE
        del test_brain

    n_trials = args.n_trials if args.n_trials is not None else N_TRIALS

    config = BenchmarkConfig(
        conditions=args.conditions,
        n_seeds=args.n_seeds,
        scale=args.scale,
        workers=args.workers,
        stim_intensity=stim_intensity,
        n_trials=n_trials,
        trial_steps=TRIAL_STEPS,
        decision_margin=DECISION_MARGIN,
    )

    print("=" * 80)
    print("oNeuro Go/No-Go Benchmark (CALIBRATED)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Conditions: {args.conditions}")
    print(f"  Seeds per condition: {args.n_seeds}")
    print(f"  Scale: {args.scale}")
    print(f"  Workers: {args.workers}")
    print(f"  Stimulation intensity: {stim_intensity} uA/cm²")
    print(f"  Trials per run: {n_trials}")
    print(f"  Trial steps: {TRIAL_STEPS}")
    print(f"  Decision margin: {DECISION_MARGIN}")

    # Create work items (paired seeds)
    seeds = list(range(args.n_seeds))
    work_items = []
    for seed in seeds:
        for cond in args.conditions:
            work_items.append((cond, seed, args.scale, asdict(config)))

    print(f"\nRunning {len(work_items)} benchmark configurations...", flush=True)

    # Run with multiprocessing - use imap_unordered for progress tracking
    t0 = time.time()
    results = []
    with mp.Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_condition_worker, work_items)):
            results.append(result)
            if (i + 1) % 5 == 0 or i + 1 == len(work_items):
                print(f"  Progress: {i+1}/{len(work_items)} ({100*(i+1)//len(work_items)}%)", flush=True)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")

    # Check for failures
    failures = [r for r in results if not r.get("success")]
    if failures:
        print(f"\n⚠️ {len(failures)} runs failed:")
        for f in failures[:5]:
            print(f"  {f['condition']} seed={f['seed']}: {f['error']}")

    # Summarize
    summaries = summarize_results(results)
    print_summary(summaries)

    # Save results
    output = {
        "config": asdict(config),
        "elapsed_seconds": elapsed,
        "results": results,
        "summaries": summaries,
    }

    if args.output:
        output_path = args.output
    else:
        os.makedirs("experiments/results", exist_ok=True)
        output_path = f"experiments/results/go_no_go_benchmark_{int(time.time())}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")

    # Return code based on signal detection
    if "full_learning" in summaries:
        full_imp = summaries["full_learning"]["mean_improvement"]
        if full_imp > 0.05:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Calibration needed
    else:
        sys.exit(2)  # Missing full_learning condition


if __name__ == "__main__":
    # Fix for macOS multiprocessing issues
    mp.freeze_support()
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    main()
