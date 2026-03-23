#![allow(dead_code)] // Module wired but not yet called from step loop.
//! Stochastic gene expression overlay for the whole-cell simulator.
//!
//! Implements Gillespie tau-leaping for per-operon mRNA burst dynamics,
//! producing biologically realistic Fano > 1 protein noise. The module
//! overlays stochastic perturbations on top of the deterministic expression
//! model, preserving mean abundances while adding variance.
//!
//! References:
//! - Paulsson (2005) "Models of stochastic gene expression", Physics of Life Reviews
//! - Raj & van Oudenaarden (2008) "Nature, Nurture, or Chance", Cell
//! - Taniguchi et al. (2010) "Quantifying E. coli proteome", Science

use crate::constants::clamp;

/// Configuration for stochastic gene expression overlay.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StochasticExpressionConfig {
    pub enabled: bool,
    /// Mean burst size (transcripts per burst event), typically 2-10 for bacteria.
    pub mean_burst_size: f32,
    /// Promoter ON rate (1/s), controls burst frequency.
    pub promoter_on_rate: f32,
    /// Promoter OFF rate (1/s), controls burst duration.
    pub promoter_off_rate: f32,
    /// mRNA half-life in seconds (Syn3A ~180s).
    pub mrna_half_life_s: f32,
    /// Protein half-life in seconds (Syn3A ~3600s).
    pub protein_half_life_s: f32,
    /// Translation rate per mRNA per second.
    pub translation_rate: f32,
    /// Tau-leap step size ceiling in seconds.
    pub max_tau_s: f32,
    /// Minimum Fano factor below which stochastic overlay is skipped.
    #[allow(dead_code)]
    pub fano_floor: f32,
}

impl Default for StochasticExpressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mean_burst_size: 4.0,
            promoter_on_rate: 0.01,
            promoter_off_rate: 0.1,
            mrna_half_life_s: 180.0,
            protein_half_life_s: 3600.0,
            translation_rate: 0.04,
            max_tau_s: 10.0,
            fano_floor: 1.05,
        }
    }
}

/// Per-operon stochastic state.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StochasticOperonState {
    pub promoter_on: bool,
    pub mrna_count: f32,
    pub protein_count: f32,
    pub mrna_mean_ema: f32,
    pub mrna_var_ema: f32,
    pub fano_factor: f32,
    pub burst_count: u32,
}

impl Default for StochasticOperonState {
    fn default() -> Self {
        Self {
            promoter_on: false,
            mrna_count: 0.0,
            protein_count: 0.0,
            mrna_mean_ema: 0.0,
            mrna_var_ema: 0.0,
            fano_factor: 1.0,
            burst_count: 0,
        }
    }
}

impl StochasticOperonState {
    pub fn from_deterministic(transcript_abundance: f32, protein_abundance: f32) -> Self {
        Self {
            promoter_on: false,
            mrna_count: transcript_abundance.max(0.0),
            protein_count: protein_abundance.max(0.0),
            mrna_mean_ema: transcript_abundance.max(0.0),
            mrna_var_ema: transcript_abundance.max(0.0),
            fano_factor: 1.0,
            burst_count: 0,
        }
    }
}

/// Xorshift64 RNG — fast, deterministic, no external deps.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StochasticRng {
    state: u64,
}

impl Default for StochasticRng {
    fn default() -> Self {
        Self::new(42)
    }
}

impl StochasticRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    pub fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f32) / (u64::MAX as f32)
    }

    pub fn poisson(&mut self, lambda: f32) -> u32 {
        if lambda <= 0.0 {
            return 0;
        }
        if lambda < 30.0 {
            let l = (-lambda).exp();
            let mut k = 0u32;
            let mut p = 1.0f32;
            loop {
                k += 1;
                p *= self.next_f32().max(1e-10);
                if p < l {
                    break;
                }
            }
            k - 1
        } else {
            let u1 = self.next_f32().max(1e-10);
            let u2 = self.next_f32().max(1e-10);
            let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
            (lambda + lambda.sqrt() * z).round().max(0.0) as u32
        }
    }

    #[allow(dead_code)]
    pub fn exponential(&mut self, rate: f32) -> f32 {
        if rate <= 0.0 {
            return f32::MAX;
        }
        -self.next_f32().max(1e-10).ln() / rate
    }
}

/// Step all operon stochastic states by `dt_s` seconds using tau-leaping.
pub fn step_stochastic_expression(
    states: &mut [StochasticOperonState],
    config: &StochasticExpressionConfig,
    rates: &[(f32, f32)],
    dt_s: f32,
    rng: &mut StochasticRng,
) {
    if !config.enabled || dt_s <= 0.0 {
        return;
    }

    let tau = dt_s.min(config.max_tau_s);
    let mrna_decay_rate = (0.693 / config.mrna_half_life_s).max(1e-6);
    let protein_decay_rate = (0.693 / config.protein_half_life_s).max(1e-6);

    for (i, state) in states.iter_mut().enumerate() {
        let (synth_rate, _decay_rate) = rates.get(i).copied().unwrap_or((0.0, 0.0));
        if synth_rate < 1e-8 && state.mrna_count < 0.5 {
            continue;
        }

        // Promoter switching (telegraph model).
        if !state.promoter_on {
            let p_on = 1.0 - (-config.promoter_on_rate * tau).exp();
            if rng.next_f32() < p_on {
                state.promoter_on = true;
            }
        } else {
            let p_off = 1.0 - (-config.promoter_off_rate * tau).exp();
            if rng.next_f32() < p_off {
                state.promoter_on = false;
            }
        }

        // mRNA dynamics.
        let mrna_synth = if state.promoter_on {
            let burst_lambda = synth_rate * config.mean_burst_size * tau;
            let produced = rng.poisson(burst_lambda);
            if produced > 0 {
                state.burst_count += 1;
            }
            produced as f32
        } else {
            0.0
        };
        let mrna_decay = rng.poisson(state.mrna_count * mrna_decay_rate * tau) as f32;
        state.mrna_count = (state.mrna_count + mrna_synth - mrna_decay).max(0.0);

        // Protein dynamics.
        let protein_synth = rng.poisson(state.mrna_count * config.translation_rate * tau) as f32;
        let protein_decay = rng.poisson(state.protein_count * protein_decay_rate * tau) as f32;
        state.protein_count = (state.protein_count + protein_synth - protein_decay).max(0.0);

        // Fano factor EMA.
        let alpha = 0.02;
        let prev_mean = state.mrna_mean_ema;
        state.mrna_mean_ema += alpha * (state.mrna_count - state.mrna_mean_ema);
        let deviation = state.mrna_count - prev_mean;
        state.mrna_var_ema += alpha * (deviation * deviation - state.mrna_var_ema);
        if state.mrna_mean_ema > 0.5 {
            state.fano_factor = clamp(state.mrna_var_ema / state.mrna_mean_ema, 0.0, 100.0);
        } else {
            state.fano_factor = 1.0;
        }
    }
}

/// Compute activity modifiers from stochastic→deterministic bridge.
pub fn sync_stochastic_to_deterministic(
    states: &[StochasticOperonState],
    config: &StochasticExpressionConfig,
    deterministic_abundances: &[(f32, f32)],
) -> Vec<(f32, f32)> {
    if !config.enabled {
        return vec![(1.0, 1.0); states.len()];
    }
    states
        .iter()
        .zip(deterministic_abundances.iter())
        .map(|(stoch, &(det_mrna, det_protein))| {
            let mrna_ratio = if det_mrna > 0.5 {
                clamp(stoch.mrna_count / det_mrna, 0.1, 10.0)
            } else {
                1.0
            };
            let protein_ratio = if det_protein > 0.5 {
                clamp(stoch.protein_count / det_protein, 0.1, 10.0)
            } else {
                1.0
            };
            (mrna_ratio, protein_ratio)
        })
        .collect()
}

/// Initialize stochastic states from deterministic expression state.
pub fn sync_deterministic_to_stochastic(
    deterministic_abundances: &[(f32, f32)],
) -> Vec<StochasticOperonState> {
    deterministic_abundances
        .iter()
        .map(|&(mrna, protein)| StochasticOperonState::from_deterministic(mrna, protein))
        .collect()
}

/// Summary statistics across all operon states.
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct StochasticExpressionSummary {
    pub mean_fano_factor: f32,
    pub max_fano_factor: f32,
    pub total_bursts: u32,
    pub active_promoters: usize,
    pub total_mrna: f32,
    pub total_protein: f32,
}

#[allow(dead_code)]
pub fn summarize_stochastic_expression(
    states: &[StochasticOperonState],
) -> StochasticExpressionSummary {
    if states.is_empty() {
        return StochasticExpressionSummary::default();
    }
    let mut sum = StochasticExpressionSummary::default();
    let mut fano_sum = 0.0f32;
    for s in states {
        fano_sum += s.fano_factor;
        sum.max_fano_factor = sum.max_fano_factor.max(s.fano_factor);
        sum.total_bursts += s.burst_count;
        if s.promoter_on {
            sum.active_promoters += 1;
        }
        sum.total_mrna += s.mrna_count;
        sum.total_protein += s.protein_count;
    }
    sum.mean_fano_factor = fano_sum / states.len() as f32;
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> StochasticExpressionConfig {
        StochasticExpressionConfig {
            enabled: true,
            ..Default::default()
        }
    }

    #[test]
    fn disabled_config_is_noop() {
        let config = StochasticExpressionConfig::default();
        let mut states = vec![StochasticOperonState::from_deterministic(10.0, 100.0)];
        let rates = vec![(0.1, 0.01)];
        let mut rng = StochasticRng::new(42);
        let before = states[0].mrna_count;
        step_stochastic_expression(&mut states, &config, &rates, 1.0, &mut rng);
        assert_eq!(states[0].mrna_count, before);
    }

    #[test]
    fn rng_produces_valid_range() {
        let mut rng = StochasticRng::new(12345);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0, "RNG out of range: {v}");
        }
    }

    #[test]
    fn poisson_mean_approximately_correct() {
        let mut rng = StochasticRng::new(99);
        let lambda = 5.0;
        let n = 10000;
        let sum: f64 = (0..n).map(|_| rng.poisson(lambda) as f64).sum();
        let mean = sum / n as f64;
        assert!(
            (mean - lambda as f64).abs() < 0.3,
            "Poisson mean should be ~{lambda}, got {mean:.2}"
        );
    }

    #[test]
    fn promoter_switches_over_time() {
        let config = test_config();
        let mut state = StochasticOperonState::default();
        let mut rng = StochasticRng::new(42);
        let rates = vec![(0.1, 0.01)];
        let mut on_count = 0;
        for _ in 0..1000 {
            step_stochastic_expression(
                std::slice::from_mut(&mut state),
                &config,
                &rates,
                1.0,
                &mut rng,
            );
            if state.promoter_on {
                on_count += 1;
            }
        }
        assert!(
            on_count > 10 && on_count < 500,
            "promoter ON fraction out of range: {on_count}/1000"
        );
    }

    #[test]
    fn mrna_stays_non_negative() {
        let config = test_config();
        let mut states = vec![StochasticOperonState::from_deterministic(1.0, 10.0)];
        let rates = vec![(0.001, 0.1)];
        let mut rng = StochasticRng::new(7);
        for _ in 0..500 {
            step_stochastic_expression(&mut states, &config, &rates, 1.0, &mut rng);
            assert!(states[0].mrna_count >= 0.0);
            assert!(states[0].protein_count >= 0.0);
        }
    }

    #[test]
    fn bursts_produce_fano_above_one() {
        let config = StochasticExpressionConfig {
            enabled: true,
            mean_burst_size: 8.0,
            promoter_on_rate: 0.02,
            promoter_off_rate: 0.05,
            ..Default::default()
        };
        let mut states = vec![StochasticOperonState::from_deterministic(5.0, 50.0)];
        let rates = vec![(0.5, 0.01)];
        let mut rng = StochasticRng::new(2024);
        for _ in 0..5000 {
            step_stochastic_expression(&mut states, &config, &rates, 1.0, &mut rng);
        }
        assert!(
            states[0].fano_factor > 1.0,
            "Bursty expression should produce Fano > 1, got {:.3}",
            states[0].fano_factor
        );
        assert!(states[0].burst_count > 0);
    }

    #[test]
    fn sync_det_to_stoch_preserves_counts() {
        let abundances = vec![(10.0, 100.0), (5.0, 50.0), (0.0, 0.0)];
        let states = sync_deterministic_to_stochastic(&abundances);
        assert_eq!(states.len(), 3);
        assert!((states[0].mrna_count - 10.0).abs() < 1e-4);
        assert!((states[0].protein_count - 100.0).abs() < 1e-4);
        assert!(states[2].mrna_count.abs() < 1e-4);
    }

    #[test]
    fn sync_stoch_to_det_gives_ratios() {
        let config = test_config();
        let states = vec![
            StochasticOperonState {
                mrna_count: 20.0,
                protein_count: 200.0,
                ..Default::default()
            },
            StochasticOperonState {
                mrna_count: 5.0,
                protein_count: 50.0,
                ..Default::default()
            },
        ];
        let det = vec![(10.0, 100.0), (10.0, 100.0)];
        let ratios = sync_stochastic_to_deterministic(&states, &config, &det);
        assert!((ratios[0].0 - 2.0).abs() < 1e-4);
        assert!((ratios[0].1 - 2.0).abs() < 1e-4);
        assert!((ratios[1].0 - 0.5).abs() < 1e-4);
    }

    #[test]
    fn summary_statistics_correct() {
        let states = vec![
            StochasticOperonState {
                fano_factor: 2.0,
                burst_count: 10,
                promoter_on: true,
                mrna_count: 5.0,
                protein_count: 50.0,
                ..Default::default()
            },
            StochasticOperonState {
                fano_factor: 3.0,
                burst_count: 20,
                promoter_on: false,
                mrna_count: 8.0,
                protein_count: 80.0,
                ..Default::default()
            },
        ];
        let summary = summarize_stochastic_expression(&states);
        assert!((summary.mean_fano_factor - 2.5).abs() < 1e-4);
        assert!((summary.max_fano_factor - 3.0).abs() < 1e-4);
        assert_eq!(summary.total_bursts, 30);
        assert_eq!(summary.active_promoters, 1);
        assert!((summary.total_mrna - 13.0).abs() < 1e-4);
        assert!((summary.total_protein - 130.0).abs() < 1e-4);
    }

    #[test]
    fn multiple_operons_independent() {
        let config = test_config();
        let mut states = vec![
            StochasticOperonState::from_deterministic(10.0, 100.0),
            StochasticOperonState::from_deterministic(1.0, 10.0),
        ];
        let rates = vec![(0.5, 0.01), (0.01, 0.01)];
        let mut rng = StochasticRng::new(42);
        for _ in 0..500 {
            step_stochastic_expression(&mut states, &config, &rates, 1.0, &mut rng);
        }
        assert!(states[0].mrna_count >= 0.0);
        assert!(states[1].mrna_count >= 0.0);
    }
}
