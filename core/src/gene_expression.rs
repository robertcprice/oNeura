//! Gene expression pipeline: immediate-early genes, neurotrophins, and
//! activity-dependent transcription.
//!
//! This module implements the slow (minutes-to-hours) molecular processes
//! that translate neural activity into long-term structural changes. It is
//! **interval-gated**: called every `GENE_EXPRESSION_INTERVAL` steps (10),
//! not every step, because gene expression operates on timescales 3-4 orders
//! of magnitude slower than membrane dynamics.
//!
//! Three gene products are tracked per neuron:
//!
//! - **c-Fos** (IEG): Rapid induction by calcium influx. Peaks within
//!   minutes of strong activity, then decays. A marker of recent neural
//!   activation used in experimental neuroscience.
//!
//! - **Arc** (activity-regulated cytoskeletal): Induced by CREB
//!   phosphorylation. Critical for AMPA receptor endocytosis and long-term
//!   depression. Somewhat slower than c-Fos.
//!
//! - **BDNF** (brain-derived neurotrophic factor): Induced by sustained
//!   CREB phosphorylation. The primary neurotrophin supporting synapse
//!   survival and growth. Very slow dynamics (hours-scale).
//!
//! All concentrations are normalized to [0, 1] for simplicity.

use crate::constants::*;
use crate::neuron_arrays::NeuronArrays;

// ===== Gene Expression Time Constants =====

/// c-Fos induction rate from calcium (nM^-1 ms^-1, scaled for interval).
/// In reality, c-Fos mRNA appears within 5-15 minutes of stimulation.
const CFOS_INDUCTION_RATE: f32 = 0.0001;

/// c-Fos calcium threshold for induction (nM).
/// Requires significant calcium influx above resting (~50 nM).
const CFOS_CA_THRESHOLD_NM: f32 = 200.0;

/// c-Fos decay time constant (ms). Half-life ~30 minutes.
const CFOS_DECAY_TAU_MS: f32 = 1_800_000.0;

/// Arc induction rate from CREB phosphorylation.
const ARC_INDUCTION_RATE: f32 = 0.00005;

/// Arc CREB threshold for induction.
const ARC_CREB_THRESHOLD: f32 = 0.1;

/// Arc decay time constant (ms). Half-life ~1 hour.
const ARC_DECAY_TAU_MS: f32 = 3_600_000.0;

/// BDNF induction rate from CREB phosphorylation.
/// Slower than Arc because BDNF requires sustained CREB activation.
const BDNF_INDUCTION_RATE: f32 = 0.00001;

/// BDNF CREB threshold (higher than Arc; needs sustained phosphorylation).
const BDNF_CREB_THRESHOLD: f32 = 0.2;

/// BDNF decay time constant (ms). Half-life ~12 hours.
const BDNF_DECAY_TAU_MS: f32 = 43_200_000.0;

/// Update gene expression for all alive neurons.
///
/// This function advances the c-Fos, Arc, and BDNF levels for each neuron
/// using simple first-order ODEs:
///
/// ```text
/// d[c-Fos]/dt = induction_rate * max(0, Ca - threshold) * (1 - [c-Fos]) - decay
/// d[Arc]/dt   = induction_rate * max(0, CREB_p - threshold) * (1 - [Arc]) - decay
/// d[BDNF]/dt  = induction_rate * max(0, CREB_p - threshold) * (1 - [BDNF]) - decay
/// ```
///
/// The `(1 - level)` term provides natural saturation at 1.0.
///
/// # Arguments
/// * `neurons` - Mutable neuron arrays. Reads `ca_cytoplasmic`, `creb_p`, `alive`.
///   Writes `cfos_level`, `arc_level`, `bdnf_level`.
/// * `dt` - **Effective** time step in ms. Since this function is called every
///   `GENE_EXPRESSION_INTERVAL` steps, the caller should pass `dt * interval`
///   to account for the skipped steps.
pub fn update_gene_expression(neurons: &mut NeuronArrays, dt: f32) {
    for i in 0..neurons.count {
        if neurons.alive[i] == 0 {
            continue;
        }

        // ----- c-Fos: calcium-driven IEG -----
        let ca = neurons.ca_cytoplasmic[i];
        let ca_drive = (ca - CFOS_CA_THRESHOLD_NM).max(0.0);
        let cfos_induction = CFOS_INDUCTION_RATE * ca_drive * (1.0 - neurons.cfos_level[i]);
        let cfos_decay = neurons.cfos_level[i] * dt / CFOS_DECAY_TAU_MS;
        neurons.cfos_level[i] += (cfos_induction * dt - cfos_decay).max(-neurons.cfos_level[i]);
        neurons.cfos_level[i] = clamp(neurons.cfos_level[i], 0.0, 1.0);

        // ----- Arc: CREB-driven activity-regulated gene -----
        let creb = neurons.creb_p[i];
        let creb_drive_arc = (creb - ARC_CREB_THRESHOLD).max(0.0);
        let arc_induction = ARC_INDUCTION_RATE * creb_drive_arc * (1.0 - neurons.arc_level[i]);
        let arc_decay = neurons.arc_level[i] * dt / ARC_DECAY_TAU_MS;
        neurons.arc_level[i] += (arc_induction * dt - arc_decay).max(-neurons.arc_level[i]);
        neurons.arc_level[i] = clamp(neurons.arc_level[i], 0.0, 1.0);

        // ----- BDNF: CREB-driven neurotrophin (slowest) -----
        let creb_drive_bdnf = (creb - BDNF_CREB_THRESHOLD).max(0.0);
        let bdnf_induction = BDNF_INDUCTION_RATE * creb_drive_bdnf * (1.0 - neurons.bdnf_level[i]);
        let bdnf_decay = neurons.bdnf_level[i] * dt / BDNF_DECAY_TAU_MS;
        neurons.bdnf_level[i] += (bdnf_induction * dt - bdnf_decay).max(-neurons.bdnf_level[i]);
        neurons.bdnf_level[i] = clamp(neurons.bdnf_level[i], 0.0, 1.0);
    }
}

/// Apply BDNF-mediated synapse strengthening.
///
/// BDNF promotes synapse survival and growth. Synapses on neurons with high
/// BDNF levels get a modest strength increase. Call this at the same interval
/// as `update_gene_expression`.
///
/// # Arguments
/// * `neurons` - Neuron arrays (read `bdnf_level`).
/// * `synapses` - Mutable synapse arrays (write `strength`).
/// * `dt` - Effective time step in ms.
pub fn apply_bdnf_effects(
    neurons: &NeuronArrays,
    synapses: &mut crate::synapse_arrays::SynapseArrays,
    dt: f32,
) {
    let bdnf_strengthen_rate = 0.000001; // very slow strengthening

    for pre in 0..synapses.n_neurons {
        for syn_idx in synapses.outgoing_range(pre) {
            let post = synapses.col_indices[syn_idx] as usize;
            if post >= neurons.count {
                continue;
            }

            let bdnf = neurons.bdnf_level[post];
            if bdnf > 0.1 {
                // BDNF increases synaptic strength (health) slowly
                let strengthen = bdnf_strengthen_rate * bdnf * dt;
                synapses.strength[syn_idx] = (synapses.strength[syn_idx] + strengthen).min(1.0);
                synapses.recompute_weight(syn_idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gene_expression_at_rest() {
        let mut neurons = NeuronArrays::new(5);
        // At rest: ca ~ 50 nM (below threshold), creb_p ~ 0
        let dt = 1.0; // 1 ms effective
        update_gene_expression(&mut neurons, dt);

        // No genes should be induced at rest
        for i in 0..5 {
            assert!(
                neurons.cfos_level[i] < 1e-6,
                "c-Fos should not be induced at resting Ca"
            );
            assert!(
                neurons.arc_level[i] < 1e-6,
                "Arc should not be induced without CREB"
            );
            assert!(
                neurons.bdnf_level[i] < 1e-6,
                "BDNF should not be induced without CREB"
            );
        }
    }

    #[test]
    fn test_cfos_induction_by_calcium() {
        let mut neurons = NeuronArrays::new(2);
        // Elevate calcium well above threshold
        neurons.ca_cytoplasmic[0] = 5000.0;
        neurons.ca_cytoplasmic[1] = 50.0; // resting, control

        // Run for many effective milliseconds
        for _ in 0..100 {
            update_gene_expression(&mut neurons, 100.0);
        }

        assert!(
            neurons.cfos_level[0] > neurons.cfos_level[1],
            "High-Ca neuron should have more c-Fos: {} vs {}",
            neurons.cfos_level[0],
            neurons.cfos_level[1]
        );
    }

    #[test]
    fn test_arc_induction_by_creb() {
        let mut neurons = NeuronArrays::new(2);
        neurons.creb_p[0] = 0.5; // strong CREB phosphorylation
        neurons.creb_p[1] = 0.0; // control

        for _ in 0..100 {
            update_gene_expression(&mut neurons, 100.0);
        }

        assert!(
            neurons.arc_level[0] > neurons.arc_level[1],
            "CREB-active neuron should have more Arc: {} vs {}",
            neurons.arc_level[0],
            neurons.arc_level[1]
        );
    }

    #[test]
    fn test_gene_levels_bounded() {
        let mut neurons = NeuronArrays::new(1);
        neurons.ca_cytoplasmic[0] = 100_000.0;
        neurons.creb_p[0] = 1.0;

        // Run for a very long effective time
        for _ in 0..10_000 {
            update_gene_expression(&mut neurons, 1000.0);
        }

        assert!(neurons.cfos_level[0] <= 1.0, "c-Fos must not exceed 1.0");
        assert!(neurons.arc_level[0] <= 1.0, "Arc must not exceed 1.0");
        assert!(neurons.bdnf_level[0] <= 1.0, "BDNF must not exceed 1.0");
    }

    #[test]
    fn test_dead_neurons_skipped() {
        let mut neurons = NeuronArrays::new(2);
        neurons.alive[0] = 0;
        neurons.ca_cytoplasmic[0] = 100_000.0;
        neurons.ca_cytoplasmic[1] = 100_000.0;

        for _ in 0..100 {
            update_gene_expression(&mut neurons, 100.0);
        }

        assert!(
            neurons.cfos_level[0] < 1e-6,
            "Dead neuron should not have gene expression"
        );
        assert!(
            neurons.cfos_level[1] > 0.0,
            "Alive neuron with high Ca should have c-Fos induction"
        );
    }
}
