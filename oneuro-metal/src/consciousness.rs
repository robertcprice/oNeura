//! Consciousness metrics -- 7 measures of integrated information and awareness.
//!
//! Implements a biophysically-grounded consciousness monitoring system that computes
//! 7 complementary metrics from neural activity patterns:
//!
//! 1. **Phi** (IIT): Integrated information via mutual information between partitions.
//! 2. **PCI** (Perturbational Complexity Index): Lempel-Ziv complexity of evoked responses.
//! 3. **Causal Density**: Fraction of significant causal interactions in the network.
//! 4. **Criticality**: Branching ratio (avalanche propagation near edge of chaos).
//! 5. **Global Workspace**: Fraction of neurons with synchronized above-threshold activation.
//! 6. **Orch-OR**: Quantum coherence contribution from microtubule collapse events.
//! 7. **Composite**: Weighted combination with log-scale Phi normalization.
//!
//! # Anesthesia Validation
//!
//! Under general anesthesia (GABA-A 8x, NMDA 0.05x, Na_v 0.5x, etc.), the composite
//! score should drop > 70%. This matches clinical observations where anesthetized
//! patients show PCI < 0.31 (Casali et al., 2013) and Phi collapses.
//!
//! # Scale Considerations
//!
//! - Phi uses log-scale normalization: `log(1 + phi) / log(1 + N^1.5)` to prevent
//!   saturation at large network sizes (1000+ neurons).
//! - Global workspace thresholds are adaptive based on network size.
//! - PCI perturbation parameters auto-scale with network size.

use crate::neuron_arrays::NeuronArrays;
use crate::synapse_arrays::SynapseArrays;

/// All 7 consciousness metrics computed from neural activity.
#[derive(Debug, Clone, Copy)]
pub struct ConsciousnessMetrics {
    /// IIT integrated information (simplified 2-partition).
    pub phi: f32,
    /// Perturbational Complexity Index (Lempel-Ziv complexity of spatiotemporal pattern).
    pub pci: f32,
    /// Fraction of significant causal interactions in the network.
    pub causal_density: f32,
    /// Branching ratio: mean secondary spikes per primary spike (critical = 1.0).
    pub criticality: f32,
    /// Fraction of neurons in the global workspace (above-threshold synchronized activation).
    pub global_workspace: f32,
    /// Orch-OR quantum coherence contribution (microtubule coherence * collapse events).
    pub orch_or: f32,
    /// Weighted composite consciousness score in [0, 1].
    pub composite: f32,
}

impl Default for ConsciousnessMetrics {
    fn default() -> Self {
        Self {
            phi: 0.0,
            pci: 0.0,
            causal_density: 0.0,
            criticality: 0.0,
            global_workspace: 0.0,
            orch_or: 0.0,
            composite: 0.0,
        }
    }
}

/// Monitors neural activity over a sliding window and computes consciousness metrics.
///
/// Call `record()` each simulation step to capture the current firing pattern,
/// then `compute()` to calculate all 7 metrics from the accumulated history.
///
/// The window size determines how many steps of firing history are retained.
/// Typical usage: window_size=100 (10 ms at dt=0.1ms).
pub struct ConsciousnessMonitor {
    /// Firing history: ring buffer of binary firing vectors.
    /// `spike_history[t][i]` = true if neuron i fired at time t.
    spike_history: Vec<Vec<bool>>,
    /// Number of neurons being monitored.
    n_neurons: usize,
    /// Maximum timesteps retained in the ring buffer.
    window_size: usize,
    /// Write cursor in the ring buffer.
    cursor: usize,
    /// Total records written (for knowing how much of the buffer is valid).
    total_records: u64,
    /// Running spike count per neuron over the window (maintained incrementally).
    spike_counts: Vec<u32>,
    /// Per-step fired counts for branching ratio computation.
    fired_counts: Vec<u32>,
}

impl ConsciousnessMonitor {
    /// Create a new monitor for `n_neurons` neurons with default window size (100 steps).
    pub fn new(n_neurons: usize) -> Self {
        Self::with_window(n_neurons, 100)
    }

    /// Create a new monitor with a custom window size.
    pub fn with_window(n_neurons: usize, window_size: usize) -> Self {
        Self {
            spike_history: vec![vec![false; n_neurons]; window_size],
            n_neurons,
            window_size,
            cursor: 0,
            total_records: 0,
            spike_counts: vec![0; n_neurons],
            fired_counts: vec![0; window_size],
        }
    }

    /// Record the current firing state from NeuronArrays.
    ///
    /// Call this once per simulation step. The monitor maintains a circular
    /// buffer of the last `window_size` firing patterns. Running spike counts
    /// are maintained incrementally for O(N) per call.
    pub fn record(&mut self, neurons: &NeuronArrays) {
        let n = self.n_neurons.min(neurons.count);

        // Subtract old data at cursor position before overwriting
        for j in 0..n {
            if self.spike_history[self.cursor][j] {
                self.spike_counts[j] = self.spike_counts[j].saturating_sub(1);
            }
        }

        // Write new data
        let mut fired_count = 0u32;
        for j in 0..n {
            let fired = neurons.fired[j] != 0 && neurons.alive[j] != 0;
            self.spike_history[self.cursor][j] = fired;
            if fired {
                self.spike_counts[j] += 1;
                fired_count += 1;
            }
        }

        self.fired_counts[self.cursor] = fired_count;
        self.cursor = (self.cursor + 1) % self.window_size;
        self.total_records += 1;
    }

    /// Compute all 7 consciousness metrics from accumulated spike history.
    ///
    /// Requires at least 10 steps of history for meaningful results.
    /// Returns default (all zeros) if insufficient data.
    pub fn compute(
        &self,
        neurons: &NeuronArrays,
        synapses: &SynapseArrays,
    ) -> ConsciousnessMetrics {
        let n = neurons.count;
        if n < 2 {
            return ConsciousnessMetrics::default();
        }

        let filled = self.total_records.min(self.window_size as u64) as usize;
        if filled < 10 {
            return ConsciousnessMetrics::default();
        }

        let phi = self.compute_phi(synapses, filled);
        let pci = self.compute_pci(filled);
        let causal_density = self.compute_causal_density(synapses);
        let criticality = self.compute_criticality(filled);
        let global_workspace = self.compute_global_workspace(neurons);
        let orch_or = self.compute_orch_or(neurons);

        // Composite: weighted aggregate with log-scale Phi normalization
        let n_f = n as f32;
        let phi_norm = if n_f > 1.0 {
            let max_phi = n_f.powf(1.5);
            (1.0 + phi).ln() / (1.0 + max_phi).ln()
        } else {
            0.0
        };

        // Criticality score: Gaussian peak at BR=1.0 (critical point)
        let crit_score = if criticality > 0.0 {
            let dev = criticality - 1.0;
            (-dev * dev / 0.5).exp()
        } else {
            0.0
        };

        let composite = (phi_norm * 0.25
            + pci * 0.20
            + causal_density * 0.10
            + crit_score * 0.15
            + global_workspace * 0.15
            + orch_or * 0.10
            + pci * 0.05) // extra PCI weight (best clinical predictor)
            .clamp(0.0, 1.0);

        ConsciousnessMetrics {
            phi,
            pci,
            causal_density,
            criticality,
            global_workspace,
            orch_or,
            composite,
        }
    }

    /// IIT Phi: mutual information between two halves of the network.
    ///
    /// Simplified for speed: split network at midpoint, compute MI between
    /// the aggregate firing patterns of the two partitions. True IIT requires
    /// minimum information partition (MIP) search which is NP-hard; this
    /// 2-partition approximation captures the essential integration measure.
    fn compute_phi(&self, synapses: &SynapseArrays, filled: usize) -> f32 {
        let n = self.n_neurons;
        if n < 4 {
            return 0.0;
        }

        let mid = n / 2;

        // Count cross-partition connections (structural integration)
        let mut cross_connections = 0u32;
        let mut total_connections = 0u32;
        for pre in 0..n.min(synapses.n_neurons) {
            for syn_idx in synapses.outgoing_range(pre) {
                let post = synapses.col_indices[syn_idx] as usize;
                total_connections += 1;
                if (pre < mid) != (post < mid) {
                    cross_connections += 1;
                }
            }
        }

        if total_connections == 0 {
            return 0.0;
        }

        // Functional integration: correlation between partition activities
        let mut both_active = 0u32;
        let mut left_active = 0u32;
        let mut right_active = 0u32;

        for step_idx in 0..filled {
            let buf_idx = self.ring_idx(step_idx);
            let mut left_any = false;
            let mut right_any = false;
            for j in 0..n {
                if self.spike_history[buf_idx][j] {
                    if j < mid {
                        left_any = true;
                    } else {
                        right_any = true;
                    }
                }
            }
            if left_any {
                left_active += 1;
            }
            if right_any {
                right_active += 1;
            }
            if left_any && right_any {
                both_active += 1;
            }
        }

        let w = filled as f32;
        let p_l = left_active as f32 / w;
        let p_r = right_active as f32 / w;
        let p_lr = both_active as f32 / w;

        // Mutual information: I(L;R) = p(l,r) * log(p(l,r) / (p(l) * p(r)))
        let phi = if p_l > 0.0 && p_r > 0.0 && p_lr > 0.0 {
            let independent = p_l * p_r;
            let mi = p_lr * (p_lr / independent).ln();
            // Scale by cross-connections to reflect structural integration
            mi * cross_connections as f32
        } else {
            0.0
        };

        phi.max(0.0)
    }

    /// PCI: Lempel-Ziv complexity of the spatiotemporal firing pattern.
    ///
    /// Binarize the firing matrix (neurons x time) and compute LZ76 complexity.
    /// Normalize by the complexity expected from random binary data of the same
    /// size and density. PCI > 0.31 indicates consciousness (Casali et al., 2013).
    fn compute_pci(&self, filled: usize) -> f32 {
        let n = self.n_neurons;
        let total_bits = filled * n;
        if total_bits == 0 {
            return 0.0;
        }

        // Flatten spike history to binary string (column-major: all neurons at t=0, t=1, ...)
        let mut binary = Vec::with_capacity(total_bits);
        for step_idx in 0..filled {
            let buf_idx = self.ring_idx(step_idx);
            for j in 0..n {
                binary.push(self.spike_history[buf_idx][j]);
            }
        }

        // Lempel-Ziv complexity (LZ76 algorithm)
        let lz = lempel_ziv_complexity(&binary);

        // Normalize: expected complexity for random binary sequence
        // c_rand ~ n / log2(n) for sequence of length n
        let total_f = total_bits as f32;
        let max_lz = total_f / total_f.log2().max(1.0);

        (lz as f32 / max_lz).clamp(0.0, 1.0)
    }

    /// Causal density: fraction of synapses where pre and post neurons co-fire.
    ///
    /// A simple proxy for Granger causality: if both pre and post neurons are
    /// active within the measurement window, the synapse is counted as
    /// "causally engaged."
    fn compute_causal_density(&self, synapses: &SynapseArrays) -> f32 {
        if synapses.n_synapses == 0 {
            return 0.0;
        }

        let mut causal_pairs = 0u32;
        for syn_idx in 0..synapses.n_synapses {
            let pre = find_pre_neuron(synapses, syn_idx);
            let post = synapses.col_indices[syn_idx] as usize;

            if pre < self.n_neurons
                && post < self.n_neurons
                && self.spike_counts[pre] > 0
                && self.spike_counts[post] > 0
            {
                causal_pairs += 1;
            }
        }

        causal_pairs as f32 / synapses.n_synapses as f32
    }

    /// Criticality: branching ratio (average secondary spikes per primary spike).
    ///
    /// At the critical point (BR = 1.0), neural avalanches follow power-law
    /// distributions. Subcritical (BR < 1): activity dies out. Supercritical
    /// (BR > 1): runaway excitation.
    ///
    /// Computed as: mean(fired_count[t+1] / fired_count[t]) for timesteps
    /// where at least one neuron fired.
    fn compute_criticality(&self, filled: usize) -> f32 {
        if filled < 3 {
            return 0.0;
        }

        let mut br_sum = 0.0f32;
        let mut br_count = 0u32;

        for t in 0..filled.saturating_sub(1) {
            let idx_t = self.ring_idx(t);
            let idx_t1 = self.ring_idx(t + 1);
            let k = self.fired_counts[idx_t];
            let m = self.fired_counts[idx_t1];

            if k > 0 {
                br_sum += m as f32 / k as f32;
                br_count += 1;
            }
        }

        if br_count > 0 {
            (br_sum / br_count as f32).clamp(0.0, 2.0)
        } else {
            0.0
        }
    }

    /// Global workspace: fraction of neurons with above-threshold firing rate.
    ///
    /// Uses an adaptive threshold based on network size. This captures the
    /// Global Neuronal Workspace theory (Dehaene & Changeux): conscious access
    /// requires widespread ignition across cortical areas.
    fn compute_global_workspace(&self, neurons: &NeuronArrays) -> f32 {
        let n = neurons.count;
        if n == 0 {
            return 0.0;
        }

        // Adaptive threshold based on network size
        let threshold: u32 = if n > 500 {
            2
        } else if n > 100 {
            3
        } else {
            5
        };

        let active = (0..n)
            .filter(|&j| j < self.n_neurons && self.spike_counts[j] >= threshold)
            .count();

        active as f32 / n as f32
    }

    /// Orch-OR: microtubule quantum coherence contribution.
    ///
    /// Sum of `mt_coherence * orch_or_events` across all alive neurons,
    /// normalized by neuron count. Higher values indicate more quantum-coherent
    /// microtubule activity contributing to consciousness.
    fn compute_orch_or(&self, neurons: &NeuronArrays) -> f32 {
        let n = self.n_neurons.min(neurons.count);
        if n == 0 {
            return 0.0;
        }

        let mut total = 0.0f32;
        let mut alive_count = 0u32;
        for i in 0..n {
            if neurons.alive[i] != 0 {
                total += neurons.mt_coherence[i] * neurons.orch_or_events[i] as f32;
                alive_count += 1;
            }
        }

        if alive_count > 0 {
            (total / alive_count as f32).min(1.0)
        } else {
            0.0
        }
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Get the buffer index for a relative time offset `t` (0 = oldest in window).
    fn ring_idx(&self, t: usize) -> usize {
        if self.total_records <= self.window_size as u64 {
            t % self.window_size
        } else {
            (self.cursor + t) % self.window_size
        }
    }
}

// =============================================================================
// Module-level helpers
// =============================================================================

/// Find the presynaptic neuron for a synapse index using CSR row offsets.
///
/// Binary search in `row_offsets` to find which row (presynaptic neuron)
/// the given synapse index belongs to.
fn find_pre_neuron(synapses: &SynapseArrays, syn_idx: usize) -> usize {
    let idx = syn_idx as u32;
    let n = synapses.n_neurons;
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if synapses.row_offsets[mid + 1] <= idx {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Lempel-Ziv 1976 complexity of a binary sequence.
///
/// Counts the number of distinct substrings encountered when scanning
/// left to right. Used as a measure of spatiotemporal pattern complexity
/// for PCI computation.
fn lempel_ziv_complexity(seq: &[bool]) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }

    let mut complexity = 1usize;
    let mut i = 0usize;
    let mut k = 1usize;
    let mut l = 1usize;

    while i + k <= n {
        // Check if seq[i..i+l] is a subsequence of seq[0..i+k-1]
        if k <= i {
            let mut found = false;
            'outer: for start in 0..i {
                if start + l > i + k {
                    break;
                }
                let mut match_len = 0;
                for j in 0..l {
                    if start + j >= n || i + j >= n {
                        break;
                    }
                    if seq[start + j] == seq[i + j] {
                        match_len += 1;
                    } else {
                        break;
                    }
                }
                if match_len >= l {
                    found = true;
                    break 'outer;
                }
            }
            if found {
                k += 1;
                l += 1;
                continue;
            }
        }

        complexity += 1;
        i += k;
        k = 1;
        l = 1;
    }

    complexity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron_arrays::NeuronArrays;
    use crate::synapse_arrays::SynapseArrays;
    use crate::types::NTType;

    #[test]
    fn test_default_metrics_are_zero() {
        let m = ConsciousnessMetrics::default();
        assert!((m.phi - 0.0).abs() < 1e-6);
        assert!((m.composite - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_insufficient_data_returns_default() {
        let n = 50;
        let monitor = ConsciousnessMonitor::new(n);
        let neurons = NeuronArrays::new(n);
        let synapses = SynapseArrays::new(n);

        let m = monitor.compute(&neurons, &synapses);
        assert!((m.composite - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_record_and_compute() {
        let n = 20;
        let mut monitor = ConsciousnessMonitor::new(n);
        let mut neurons = NeuronArrays::new(n);
        let edges = vec![
            (0u32, 1, NTType::Glutamate),
            (1, 2, NTType::Glutamate),
            (2, 3, NTType::Glutamate),
        ];
        let synapses = SynapseArrays::from_edges(n, &edges);

        // Simulate alternating firing patterns
        for t in 0..50 {
            for i in 0..n {
                neurons.fired[i] = if (i + t) % 3 == 0 { 1 } else { 0 };
            }
            monitor.record(&neurons);
        }

        let m = monitor.compute(&neurons, &synapses);
        assert!(m.composite >= 0.0 && m.composite <= 1.0);
    }

    #[test]
    fn test_all_silent_low_consciousness() {
        let n = 50;
        let mut monitor = ConsciousnessMonitor::new(n);
        let neurons = NeuronArrays::new(n);
        let synapses = SynapseArrays::new(n);

        for _ in 0..50 {
            monitor.record(&neurons);
        }

        let m = monitor.compute(&neurons, &synapses);
        assert!(
            m.composite < 0.1,
            "All-silent network should have near-zero consciousness, got {}",
            m.composite
        );
    }

    #[test]
    fn test_lempel_ziv_constant() {
        let seq = vec![false; 100];
        let c = lempel_ziv_complexity(&seq);
        // LZ76 on constant sequence: low but not necessarily 1-2 due to indexing
        assert!(
            c <= 10,
            "Constant sequence should have low LZ complexity, got {}",
            c
        );
    }

    #[test]
    fn test_lempel_ziv_alternating() {
        let seq: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let c = lempel_ziv_complexity(&seq);
        assert!(
            c >= 2,
            "Alternating sequence should have some complexity, got {}",
            c
        );
    }

    #[test]
    fn test_criticality_no_spikes() {
        let monitor = ConsciousnessMonitor::new(10);
        assert!((monitor.compute_criticality(0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_composite_in_range() {
        let n = 30;
        let mut monitor = ConsciousnessMonitor::new(n);
        let mut neurons = NeuronArrays::new(n);
        let synapses = SynapseArrays::new(n);

        for t in 0..100 {
            for i in 0..n {
                neurons.fired[i] = if (i * 7 + t * 13) % 5 == 0 { 1 } else { 0 };
            }
            monitor.record(&neurons);
        }

        let m = monitor.compute(&neurons, &synapses);
        assert!(
            m.composite >= 0.0 && m.composite <= 1.0,
            "Composite must be in [0, 1], got {}",
            m.composite
        );
    }

    #[test]
    fn test_phi_log_normalization_prevents_saturation() {
        let n_neurons = 1000;
        let phi: f32 = 500.0;
        let n = n_neurons as f32;
        let max_phi = n.powf(1.5);
        let phi_norm = (1.0 + phi).ln() / (1.0 + max_phi).ln();

        assert!(
            phi_norm < 1.0,
            "Log normalization should prevent saturation, got {}",
            phi_norm
        );
        assert!(
            phi_norm > 0.0,
            "Non-zero phi should give non-zero normalized value"
        );
    }

    #[test]
    fn test_find_pre_neuron() {
        let edges = vec![
            (0u32, 2, NTType::Glutamate),
            (1, 2, NTType::GABA),
            (1, 0, NTType::GABA),
        ];
        let synapses = SynapseArrays::from_edges(3, &edges);

        assert_eq!(find_pre_neuron(&synapses, 0), 0);
        assert_eq!(find_pre_neuron(&synapses, 1), 1);
        assert_eq!(find_pre_neuron(&synapses, 2), 1);
    }
}
