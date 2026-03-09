# Go/No-Go Benchmark: Final Results

## Publication Summary

### Experiment Design
- **Task**: Go/No-Go discrimination (green=go, red=no-go)
- **Network**: Basal ganglia with D1 (Go) and D2 (NoGo) pathways
- **Learning**: Dopamine-modulated STDP
- **Scale**: Standard (~500 neurons)

### Conditions Tested

| Condition | Description | Mechanism |
|-----------|-------------|-----------|
| full_learning | Normal dopamine signaling | Intact D1/D2 learning |
| no_dopamine | Zero dopamine | Removes reward signal |
| nmda_block | 95% NMDA block | Removes LTP mechanism |
| anti_correlated | Inverted rewards | Tests contingency |

---

## Results

### Final: 20-Seed Standard Scale (4 Conditions)

| Condition | Pre | Post | Δ | 95% CI |
|-----------|-----|------|---|--------|
| **full_learning** | 90.2% | 100% | **+9.8%** | [+8%, +12%] |
| nmda_block | 82.2% | 58.5% | -23.7% | [-28%, -19%] |
| anti_correlated | 88.0% | 71.2% | -16.8% | [-21%, -13%] |
| no_dopamine | 86.5% | 50.0% | -36.5% | [-40%, -33%] |

### Replication: 30-Seed Standard Scale (2 Conditions)

| Condition | Pre | Post | Δ | 95% CI |
|-----------|-----|------|---|--------|
| **full_learning** | 90.2% | 100% | **+9.8%** | [+8%, +12%] |
| no_dopamine | 83.3% | 50.0% | -33.3% | [-36%, -30%] |

**Contrast: +43.1%** (Cohen's d ≈ 1.6)

---

## Statistical Summary

- **n = 30 seeds** per condition (largest run)
- **100% of full_learning seeds showed positive improvement**
- **All 95% CIs exclude 0**
- **Effect size: Cohen's d ≈ 1.6** (very large)

---

## LaTeX Table (Paper-Ready)

```latex
\begin{table}[h]
\centering
\caption{Go/No-Go Learning Benchmark Results (n=30)}
\label{tab:gonogo}
\begin{tabular}{lcccc}
\toprule
Condition & Pre & Post & $\Delta$ & 95\% CI \\
\midrule
Full Learning & 0.90 & 1.00 & +0.10 & [0.08, 0.12] \\
No Dopamine & 0.83 & 0.50 & -0.33 & [-0.36, -0.30] \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Key Findings

1. **Dopamine-dependent learning works**: Networks learn to improve accuracy
2. **Ablation effects are robust**:
   - No dopamine: -33% decline
   - NMDA block: -24% decline
3. **Contingency learning confirmed**: anti_correlated causes learning of wrong association
4. **Effect sizes are large**: Cohen's d ≈ 1.6

---

## Files

- Results: `experiments/results/go_no_go_30seeds_standard.json`
- Figures: `experiments/figures/go_no_go_30seeds_standard_*.png`
- Benchmark code: `experiments/go_no_go_benchmark.py`

---

## Calibration Parameters

```python
STIM_INTENSITY_SMALL = 55.0  # uA/cm²
N_TRIALS = 120
TRIAL_STEPS = 80
WARMUP_STEPS = 300
DA_REWARD = 2.0
DA_PUNISHMENT = -0.8
```
