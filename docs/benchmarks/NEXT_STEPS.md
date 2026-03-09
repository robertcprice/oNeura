# Go/No-Go Benchmark: Next Steps

## Status: Calibration Complete ✅

The benchmark is now working with:
- Stimulation intensity: 55 uA/cm²
- 120 trials, 80 steps per trial
- Bootstrap 95% confidence intervals
- Paired-seed design
- All contrasts statistically significant

---

## Priority 1: Publishable Confirmatory Run

### Goal
Run a pre-registered confirmatory experiment suitable for publication.

### Steps
1. **Pre-register the analysis plan**
   - Define primary hypothesis: full_learning improves more than no_dopamine
   - Define secondary hypothesis: full_learning improves more than nmda_block
   - Set alpha = 0.05, two-tailed
   - Pre-register on OSF or similar

2. **Determine sample size**
   - From calibration: effect size d ≈ 0.8 for full_learning vs no_dopamine
   - Power analysis suggests n=20-30 seeds for 80% power
   - Recommend: **50 seeds** for robust CI

3. **Run confirmatory experiment**
   ```bash
   PYTHONPATH=src python3 experiments/go_no_go_benchmark.py \
       --conditions full_learning no_dopamine nmda_block \
       --n-seeds 50 \
       --scale standard \
       --workers 8 \
       --output experiments/results/go_no_go_confirmatory.json
   ```

4. **Report results**
   - Primary: Paired contrast full_learning - no_dopamine with 95% CI
   - Secondary: Paired contrast full_learning - nmda_block with 95% CI
   - Include per-seed scatter plot
   - Include learning curves (accuracy by trial block)

### Estimated Time
- Standard scale, 50 seeds, 3 conditions = 150 runs
- ~830 seconds per 30 runs (from calibration)
- Total: ~70 minutes with 8 workers

---

## Priority 2: Scale Testing

### Goal
Verify the learning signal scales to larger networks.

### Steps
1. **Run at large scale** (~1000+ neurons)
   ```bash
   PYTHONPATH=src python3 experiments/go_no_go_benchmark.py \
       --conditions full_learning no_dopamine \
       --n-seeds 20 \
       --scale large \
       --workers 4 \
       --output experiments/results/go_no_go_large_scale.json
   ```

2. **Compare results across scales**
   - minimal (~100 neurons) vs standard (~500) vs large (~1000+)
   - Check if effect size increases/decreases with scale
   - Document any scaling issues

### Estimated Time
- Large scale will be slower; estimate 2-4 hours

---

## Priority 3: Stricter Contingency Test

### Goal
Investigate why reward_shuffle still shows learning.

### Problem
In the calibrated pilot, reward_shuffle showed +9% improvement, matching full_learning. This suggests learning might not be purely contingency-driven.

### Steps
1. **Add anti-correlated reward condition**
   - In `apply_ablation()`, add new condition:
   ```python
   elif condition == "anti_correlated":
       # Reward opposite of correct response
       if is_green:
           net.release_dopamine(DA_PUNISHMENT)  # Should not learn
       else:
           net.release_dopamine(DA_REWARD)
   ```

2. **Run comparison**
   ```bash
   PYTHONPATH=src python3 experiments/go_no_go_benchmark.py \
       --conditions full_learning reward_shuffle anti_correlated \
       --n-seeds 20 \
       --scale standard
   ```

3. **Expected result**
   - full_learning: +9% improvement
   - reward_shuffle: +9% improvement (random rewards still positive on average)
   - anti_correlated: -X% decline (should actively learn wrong association)

---

## Priority 4: Learning Curve Analysis

### Goal
Visualize learning over time, not just pre/post comparison.

### Steps
1. **Modify benchmark to output trial-by-trial data**
   - Already saved in JSON under `results[].trials`
   - Need analysis script to plot

2. **Create analysis script**
   ```python
   # experiments/analyze_learning_curves.py
   import json
   import matplotlib.pyplot as plt
   import numpy as np

   def plot_learning_curves(results_file):
       with open(results_file) as f:
           data = json.load(f)

       # Group trials by condition
       for cond in ['full_learning', 'no_dopamine', 'nmda_block']:
           trials_by_seed = []
           for r in data['results']:
               if r['condition'] == cond and r['success']:
                   trials_by_seed.append([t['correct'] for t in r['trials']])

           # Compute rolling accuracy
           window = 10
           mean_curve = np.mean(trials_by_seed, axis=0)
           rolling = np.convolve(mean_curve, np.ones(window)/window, mode='valid')

           plt.plot(rolling, label=cond)

       plt.xlabel('Trial')
       plt.ylabel('Rolling Accuracy (10-trial window)')
       plt.legend()
       plt.savefig('learning_curves.png')
   ```

3. **Generate figures for paper**
   - Learning curves with 95% CI bands
   - Per-seed improvement scatter plots
   - Weight change vs behavior change correlation

---

## Priority 5: Code Quality & Reproducibility

### Steps
1. **Add unit tests for benchmark**
   - Test that conditions actually apply ablations
   - Test that paired seeds produce same baseline
   - Test JSON serialization

2. **Add requirements/environment file**
   ```txt
   # experiments/requirements.txt
   numpy>=1.20
   ```

3. **Add random seed documentation**
   - Document that seeds are used for:
     - Network initialization (RegionalBrain)
     - Trial order (if randomized)
     - Bootstrap CI (fixed at 42)

4. **Version the benchmark**
   - Tag the calibrated version in git
   - Document parameter changes in CHANGELOG

---

## Priority 6: Paper-Ready Output

### Steps
1. **Generate LaTeX table**
   ```python
   def generate_latex_table(summaries):
       print(r"\begin{table}[h]")
       print(r"\centering")
       print(r"\begin{tabular}{lcccc}")
       print(r"\toprule")
       print(r"Condition & Pre & Post & $\Delta$ & 95\% CI \\")
       print(r"\midrule")
       for cond, s in summaries.items():
           if cond.startswith("_"):
               continue
           ci = f"[{s['ci_95_lower']:.2f}, {s['ci_95_upper']:.2f}]"
           print(f"{cond} & {s['mean_pre_accuracy']:.2f} & {s['mean_post_accuracy']:.2f} & {s['mean_improvement']:+.2f} & {ci} \\\\")
       print(r"\bottomrule")
       print(r"\end{tabular}")
       print(r"\end{table}")
   ```

2. **Generate figure scripts**
   - `experiments/figures/figure_learning_curves.py`
   - `experiments/figures/figure_condition_comparison.py`

---

## Quick Reference: Commands

### Run calibrated benchmark (minimal scale)
```bash
PYTHONPATH=src python3 experiments/go_no_go_benchmark.py \
    --conditions full_learning no_dopamine nmda_block \
    --n-seeds 10 \
    --scale minimal \
    --workers 4
```

### Run at standard scale (recommended)
```bash
PYTHONPATH=src python3 experiments/go_no_go_benchmark.py \
    --conditions full_learning no_dopamine nmda_block \
    --n-seeds 20 \
    --scale standard \
    --workers 4
```

### Run with extended trials
```bash
PYTHONPATH=src python3 experiments/go_no_go_benchmark.py \
    --n-trials 200 \
    --scale minimal
```

### Custom stimulation intensity
```bash
PYTHONPATH=src python3 experiments/go_no_go_benchmark.py \
    --stim-intensity 70.0 \
    --scale large
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `experiments/go_no_go_benchmark.py` | Main benchmark script |
| `experiments/results/*.json` | Output files |
| `docs/benchmarks/go_no_go_benchmark.md` | Documentation |
| `docs/benchmarks/CALIBRATION_LOG.md` | Calibration history |
| `docs/benchmarks/NEXT_STEPS.md` | This file |

---

## Key Parameters (Calibrated)

```python
STIM_INTENSITY_SMALL = 55.0  # uA/cm² for <500 neurons
STIM_INTENSITY_LARGE = 65.0  # uA/cm² for >=500 neurons
N_TRIALS = 120
TRIAL_STEPS = 80
WARMUP_STEPS = 300
DA_REWARD = 2.0
DA_PUNISHMENT = -0.8
CONN_P_SMALL = 0.5
CONN_P_LARGE = 0.6
```

---

## Contact Points for Questions

1. **"Why no learning?"** → Check stimulation intensity first
2. **"High variance?"** → Use standard scale, not minimal
3. **"reward_shuffle matches full_learning?"** → Implement anti_correlated condition
4. **"CI includes 0?"** → Increase seeds to 30-50
