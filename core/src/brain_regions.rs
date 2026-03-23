//! Brain region architectures -- CorticalColumn, Thalamus, Hippocampus, BasalGanglia.
//!
//! Provides `RegionalBrain`, a wrapper around `MolecularBrain` that pre-wires
//! biologically plausible connectivity patterns for brain regions. Each region
//! contains neurons of appropriate archetypes with intra- and inter-region
//! synaptic connections.
//!
//! # Region Architectures
//!
//! - **CorticalColumn**: 80% pyramidal (excitatory, glutamatergic), 20% interneuron
//!   (inhibitory, GABAergic). Recurrent excitation + lateral inhibition.
//! - **Thalamus**: Relay neurons with bidirectional connections to cortex.
//!   Thalamic input drives cortical activation.
//! - **Hippocampus**: DG -> CA3 -> CA1 pathway. Dentate gyrus provides pattern
//!   separation (sparse encoding), CA3 provides pattern completion (recurrent
//!   excitation), CA1 is the output layer.
//! - **Basal Ganglia**: Medium spiny neurons (MSN), DA-modulated.
//!   Receives cortical glutamatergic input, produces GABAergic output.
//!
//! # Scale Presets
//!
//! - `minimal()`: 75 neurons for fast unit testing.
//! - `xlarge()`: 1018 neurons (6 cortical columns + full subcortical) for
//!   behavioral demos and consciousness measurement.

use crate::network::MolecularBrain;
use crate::types::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A brain organized into named regions with pre-wired connectivity.
pub struct RegionalBrain {
    /// The underlying `MolecularBrain` that owns all neural state.
    pub brain: MolecularBrain,
    /// Named regions mapping to neuron index ranges.
    pub regions: Vec<BrainRegion>,
}

/// A named group of neurons within a `RegionalBrain`.
pub struct BrainRegion {
    /// What type of brain structure this region represents.
    pub region_type: BrainRegionType,
    /// Human-readable name (e.g., "cortex_0", "hippocampus_DG").
    pub name: String,
    /// Indices into the `MolecularBrain.neurons` arrays.
    pub neuron_indices: Vec<usize>,
}

impl RegionalBrain {
    /// Create a minimal brain (75 neurons) for testing.
    ///
    /// Contains 1 cortical column (40 neurons), thalamus (15), hippocampus (12),
    /// and basal ganglia (8). Good for unit tests and smoke tests where speed
    /// matters more than biological fidelity.
    pub fn minimal(seed: u64) -> Self {
        Self::build(
            1,  // 1 cortical column
            40, // neurons per column
            15, // thalamic neurons
            6, 4, 4,
            2, // hippocampus: DG, CA3, CA1, interneurons (= 16 total, but kept small)
            8, // basal ganglia MSNs
            seed,
        )
    }

    /// Create an xlarge brain (1018 neurons) for behavioral demos.
    ///
    /// 6 cortical columns (100 neurons each = 600), thalamus (150),
    /// hippocampus (168: DG=60, CA3=50, CA1=50, interneurons=8),
    /// basal ganglia (100). Total = 1018.
    ///
    /// This is the standard scale for consciousness measurement, sleep
    /// consolidation, valence learning, and Pavlovian conditioning demos.
    pub fn xlarge(seed: u64) -> Self {
        Self::build(
            6,   // 6 cortical columns
            100, // neurons per column
            150, // thalamic neurons
            60, 50, 50, 8,   // hippocampus: DG, CA3, CA1, interneurons
            100, // basal ganglia MSNs
            seed,
        )
    }

    /// Create a brain with the specified number of cortical columns.
    ///
    /// Each column has 100 neurons. Subcortical regions scale proportionally.
    pub fn with_columns(n_columns: usize, seed: u64) -> Self {
        let per_col = 100;
        let thalamic = 25 * n_columns;
        let dg = 10 * n_columns;
        let ca3 = 8 * n_columns;
        let ca1 = 8 * n_columns;
        let hipp_int = n_columns.max(2);
        let bg = 15 * n_columns;
        Self::build(
            n_columns, per_col, thalamic, dg, ca3, ca1, hipp_int, bg, seed,
        )
    }

    /// Step the simulation by one dt tick.
    pub fn step(&mut self) {
        self.brain.step();
        self.brain.sync_shadow_from_gpu();
    }

    /// Run multiple simulation steps.
    pub fn run(&mut self, steps: u64) {
        self.brain.run(steps);
    }

    /// Stimulate thalamic relay neurons to drive cortical activation.
    ///
    /// Thalamic input is the primary pathway for sensory information into cortex.
    /// `current` is in uA/cm^2; typical values 10-40.
    pub fn stimulate_thalamus(&mut self, current: f32) {
        for region in &self.regions {
            if region.region_type == BrainRegionType::ThalamicNucleus {
                for &idx in &region.neuron_indices {
                    self.brain.stimulate(idx, current);
                }
            }
        }
    }

    /// Stimulate a specific brain region by type.
    pub fn stimulate_region(&mut self, region_type: BrainRegionType, current: f32) {
        for region in &self.regions {
            if region.region_type == region_type {
                for &idx in &region.neuron_indices {
                    self.brain.stimulate(idx, current);
                }
            }
        }
    }

    /// Get neuron indices for a specific region type.
    pub fn region_indices(&self, region_type: BrainRegionType) -> Vec<usize> {
        self.regions
            .iter()
            .filter(|r| r.region_type == region_type)
            .flat_map(|r| r.neuron_indices.iter().copied())
            .collect()
    }

    /// Get mean voltage for a specific brain region.
    pub fn region_mean_voltage(&self, region_type: BrainRegionType) -> f32 {
        let indices = self.region_indices(region_type);
        if indices.is_empty() {
            return 0.0;
        }
        let sum: f32 = indices.iter().map(|&i| self.brain.neurons.voltage[i]).sum();
        sum / indices.len() as f32
    }

    /// Count fired neurons in a specific region this step.
    pub fn region_fired_count(&self, region_type: BrainRegionType) -> usize {
        let indices = self.region_indices(region_type);
        indices
            .iter()
            .filter(|&&i| self.brain.neurons.fired[i] != 0)
            .count()
    }

    // =========================================================================
    // Internal builder
    // =========================================================================

    fn build(
        n_columns: usize,
        per_column: usize,
        n_thalamic: usize,
        n_dg: usize,
        n_ca3: usize,
        n_ca1: usize,
        n_hipp_int: usize,
        n_bg: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let n_cortex = n_columns * per_column;
        let n_hipp = n_dg + n_ca3 + n_ca1 + n_hipp_int;
        let n_total = n_cortex + n_thalamic + n_hipp + n_bg;

        let mut edges: Vec<(u32, u32, NTType)> = Vec::new();
        let mut regions: Vec<BrainRegion> = Vec::new();

        // Track global neuron index offset
        let mut offset: usize = 0;

        // =================================================================
        // 1. Cortical Columns
        // =================================================================
        let cortex_start = offset;
        let mut all_cortex_indices = Vec::new();

        for col in 0..n_columns {
            let col_start = offset;
            let n_pyr = (per_column as f32 * 0.8) as usize;
            let _n_int = per_column - n_pyr;

            let col_end = col_start + per_column;
            let pyr_range = col_start..col_start + n_pyr;
            let int_range = col_start + n_pyr..col_end;

            let col_indices: Vec<usize> = (col_start..col_end).collect();
            all_cortex_indices.extend_from_slice(&col_indices);

            // Recurrent excitation: pyramidal -> pyramidal (sparse, ~10%)
            for &pre in pyr_range.clone().collect::<Vec<_>>().iter() {
                for &post in pyr_range.clone().collect::<Vec<_>>().iter() {
                    if pre != post && rng.gen::<f32>() < 0.10 {
                        edges.push((pre as u32, post as u32, NTType::Glutamate));
                    }
                }
            }

            // Pyramidal -> interneuron (feedforward inhibition, ~20%)
            for &pre in pyr_range.clone().collect::<Vec<_>>().iter() {
                for &post in int_range.clone().collect::<Vec<_>>().iter() {
                    if rng.gen::<f32>() < 0.20 {
                        edges.push((pre as u32, post as u32, NTType::Glutamate));
                    }
                }
            }

            // Interneuron -> pyramidal (lateral inhibition, ~25%)
            for &pre in int_range.clone().collect::<Vec<_>>().iter() {
                for &post in pyr_range.clone().collect::<Vec<_>>().iter() {
                    if rng.gen::<f32>() < 0.25 {
                        edges.push((pre as u32, post as u32, NTType::GABA));
                    }
                }
            }

            // Interneuron -> interneuron (mutual inhibition, ~15%)
            for &pre in int_range.clone().collect::<Vec<_>>().iter() {
                for &post in int_range.clone().collect::<Vec<_>>().iter() {
                    if pre != post && rng.gen::<f32>() < 0.15 {
                        edges.push((pre as u32, post as u32, NTType::GABA));
                    }
                }
            }

            regions.push(BrainRegion {
                region_type: BrainRegionType::CorticalColumn,
                name: format!("cortex_{}", col),
                neuron_indices: col_indices,
            });

            offset = col_end;
        }

        // =================================================================
        // 2. Thalamus
        // =================================================================
        let thal_start = offset;
        let thal_end = thal_start + n_thalamic;
        let thal_indices: Vec<usize> = (thal_start..thal_end).collect();

        // Thalamic relay -> cortical pyramidal (feedforward, ~8%)
        for &thal in &thal_indices {
            for &ctx in &all_cortex_indices {
                if rng.gen::<f32>() < 0.08 {
                    edges.push((thal as u32, ctx as u32, NTType::Glutamate));
                }
            }
        }

        // Cortical pyramidal -> thalamic (feedback, ~5%)
        // Only pyramidal neurons (first 80% of each column) project back
        for col in 0..n_columns {
            let col_start = cortex_start + col * per_column;
            let n_pyr = (per_column as f32 * 0.8) as usize;
            for pyr_idx in col_start..col_start + n_pyr {
                for &thal in &thal_indices {
                    if rng.gen::<f32>() < 0.05 {
                        edges.push((pyr_idx as u32, thal as u32, NTType::Glutamate));
                    }
                }
            }
        }

        // Intra-thalamic recurrent (sparse, ~5%)
        for &pre in &thal_indices {
            for &post in &thal_indices {
                if pre != post && rng.gen::<f32>() < 0.05 {
                    edges.push((pre as u32, post as u32, NTType::Glutamate));
                }
            }
        }

        regions.push(BrainRegion {
            region_type: BrainRegionType::ThalamicNucleus,
            name: "thalamus".to_string(),
            neuron_indices: thal_indices.clone(),
        });
        offset = thal_end;

        // =================================================================
        // 3. Hippocampus (DG -> CA3 -> CA1)
        // =================================================================
        let dg_start = offset;
        let dg_end = dg_start + n_dg;
        let ca3_start = dg_end;
        let ca3_end = ca3_start + n_ca3;
        let ca1_start = ca3_end;
        let ca1_end = ca1_start + n_ca1;
        let hi_start = ca1_end;
        let hi_end = hi_start + n_hipp_int;

        let dg_indices: Vec<usize> = (dg_start..dg_end).collect();
        let ca3_indices: Vec<usize> = (ca3_start..ca3_end).collect();
        let ca1_indices: Vec<usize> = (ca1_start..ca1_end).collect();
        let hi_indices: Vec<usize> = (hi_start..hi_end).collect();

        // DG -> CA3 (mossy fibers, sparse pattern separation, ~8%)
        for &dg in &dg_indices {
            for &ca3 in &ca3_indices {
                if rng.gen::<f32>() < 0.08 {
                    edges.push((dg as u32, ca3 as u32, NTType::Glutamate));
                }
            }
        }

        // CA3 -> CA3 (recurrent excitation for pattern completion, ~12%)
        for &pre in &ca3_indices {
            for &post in &ca3_indices {
                if pre != post && rng.gen::<f32>() < 0.12 {
                    edges.push((pre as u32, post as u32, NTType::Glutamate));
                }
            }
        }

        // CA3 -> CA1 (Schaffer collaterals, ~15%)
        for &ca3 in &ca3_indices {
            for &ca1 in &ca1_indices {
                if rng.gen::<f32>() < 0.15 {
                    edges.push((ca3 as u32, ca1 as u32, NTType::Glutamate));
                }
            }
        }

        // Hippocampal interneurons: receive from CA3, inhibit CA1 (feedforward)
        for &ca3 in &ca3_indices {
            for &hi in &hi_indices {
                if rng.gen::<f32>() < 0.20 {
                    edges.push((ca3 as u32, hi as u32, NTType::Glutamate));
                }
            }
        }
        for &hi in &hi_indices {
            for &ca1 in &ca1_indices {
                if rng.gen::<f32>() < 0.30 {
                    edges.push((hi as u32, ca1 as u32, NTType::GABA));
                }
            }
        }

        // Cortex -> DG (perforant path, ~3%)
        for &ctx in &all_cortex_indices {
            for &dg in &dg_indices {
                if rng.gen::<f32>() < 0.03 {
                    edges.push((ctx as u32, dg as u32, NTType::Glutamate));
                }
            }
        }

        // CA1 -> cortex (hippocampal output, ~3%)
        for &ca1 in &ca1_indices {
            for &ctx in &all_cortex_indices {
                if rng.gen::<f32>() < 0.03 {
                    edges.push((ca1 as u32, ctx as u32, NTType::Glutamate));
                }
            }
        }

        let mut hipp_all = Vec::new();
        hipp_all.extend_from_slice(&dg_indices);
        hipp_all.extend_from_slice(&ca3_indices);
        hipp_all.extend_from_slice(&ca1_indices);
        hipp_all.extend_from_slice(&hi_indices);

        regions.push(BrainRegion {
            region_type: BrainRegionType::Hippocampus,
            name: "hippocampus".to_string(),
            neuron_indices: hipp_all,
        });
        offset = hi_end;

        // =================================================================
        // 4. Basal Ganglia
        // =================================================================
        let bg_start = offset;
        let bg_end = bg_start + n_bg;
        let bg_indices: Vec<usize> = (bg_start..bg_end).collect();

        // Intra-BG: MSN lateral inhibition (~10%)
        for &pre in &bg_indices {
            for &post in &bg_indices {
                if pre != post && rng.gen::<f32>() < 0.10 {
                    edges.push((pre as u32, post as u32, NTType::GABA));
                }
            }
        }

        // Cortex -> BG (corticostriatal glutamatergic, ~5%)
        for &ctx in &all_cortex_indices {
            for &bg in &bg_indices {
                if rng.gen::<f32>() < 0.05 {
                    edges.push((ctx as u32, bg as u32, NTType::Glutamate));
                }
            }
        }

        // BG -> thalamus (GABAergic output, ~8%)
        for &bg in &bg_indices {
            for &thal in &thal_indices {
                if rng.gen::<f32>() < 0.08 {
                    edges.push((bg as u32, thal as u32, NTType::GABA));
                }
            }
        }

        regions.push(BrainRegion {
            region_type: BrainRegionType::BasalGanglia,
            name: "basal_ganglia".to_string(),
            neuron_indices: bg_indices,
        });
        offset = bg_end;

        assert_eq!(offset, n_total);

        // =================================================================
        // Build brain from edges and assign archetypes
        // =================================================================
        let mut brain = MolecularBrain::from_edges(n_total, &edges);

        // Assign archetypes by region
        for region in &regions {
            match region.region_type {
                BrainRegionType::CorticalColumn => {
                    // First 80% pyramidal, last 20% interneuron
                    let n_pyr = (region.neuron_indices.len() as f32 * 0.8) as usize;
                    for (j, &idx) in region.neuron_indices.iter().enumerate() {
                        if j < n_pyr {
                            brain.neurons.archetype[idx] = NeuronArchetype::Pyramidal as u8;
                        } else {
                            brain.neurons.archetype[idx] = NeuronArchetype::Interneuron as u8;
                        }
                    }
                }
                BrainRegionType::ThalamicNucleus => {
                    for &idx in &region.neuron_indices {
                        brain.neurons.archetype[idx] = NeuronArchetype::Stellate as u8;
                    }
                }
                BrainRegionType::Hippocampus => {
                    // DG = Granule, CA3/CA1 = Pyramidal, interneurons = Interneuron
                    for &idx in &region.neuron_indices {
                        let local = idx - region.neuron_indices[0];
                        let total_hipp = n_dg + n_ca3 + n_ca1 + n_hipp_int;
                        if local < n_dg {
                            brain.neurons.archetype[idx] = NeuronArchetype::Granule as u8;
                        } else if local < n_dg + n_ca3 + n_ca1 {
                            brain.neurons.archetype[idx] = NeuronArchetype::Pyramidal as u8;
                        } else if local < total_hipp {
                            brain.neurons.archetype[idx] = NeuronArchetype::Interneuron as u8;
                        }
                    }
                }
                BrainRegionType::BasalGanglia => {
                    for &idx in &region.neuron_indices {
                        brain.neurons.archetype[idx] = NeuronArchetype::MediumSpiny as u8;
                    }
                }
            }
        }

        // Assign random 3D positions within each region for spatial structure
        for region in &regions {
            let (cx, cy, cz) = region_center(region.region_type);
            let spread = region_spread(region.region_type);
            for &idx in &region.neuron_indices {
                brain.neurons.x[idx] = cx + (rng.gen::<f32>() - 0.5) * spread;
                brain.neurons.y[idx] = cy + (rng.gen::<f32>() - 0.5) * spread;
                brain.neurons.z[idx] = cz + (rng.gen::<f32>() - 0.5) * spread;
            }
        }

        // Initialize glia state with correct neuron count
        brain.glia = crate::glia::GliaState::new(n_total);

        RegionalBrain { brain, regions }
    }
}

/// Get the 3D center point for a brain region (arbitrary spatial layout).
fn region_center(region_type: BrainRegionType) -> (f32, f32, f32) {
    match region_type {
        BrainRegionType::CorticalColumn => (0.0, 5.0, 0.0), // dorsal
        BrainRegionType::ThalamicNucleus => (0.0, 0.0, 0.0), // central
        BrainRegionType::Hippocampus => (3.0, -2.0, 0.0),   // medial temporal
        BrainRegionType::BasalGanglia => (-2.0, -1.0, 0.0), // subcortical
    }
}

/// Get the spatial spread (diameter) for a brain region.
fn region_spread(region_type: BrainRegionType) -> f32 {
    match region_type {
        BrainRegionType::CorticalColumn => 2.0,
        BrainRegionType::ThalamicNucleus => 3.0,
        BrainRegionType::Hippocampus => 4.0,
        BrainRegionType::BasalGanglia => 2.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_brain_structure() {
        let rb = RegionalBrain::minimal(42);
        let n = rb.brain.neuron_count();
        assert!(
            n >= 50,
            "Minimal brain should have at least 50 neurons, got {}",
            n
        );
        assert!(rb.brain.synapse_count() > 0, "Should have synapses");

        // Should have all 4 region types
        let types: Vec<BrainRegionType> = rb.regions.iter().map(|r| r.region_type).collect();
        assert!(types.contains(&BrainRegionType::CorticalColumn));
        assert!(types.contains(&BrainRegionType::ThalamicNucleus));
        assert!(types.contains(&BrainRegionType::Hippocampus));
        assert!(types.contains(&BrainRegionType::BasalGanglia));
    }

    #[test]
    fn test_xlarge_brain_structure() {
        let rb = RegionalBrain::xlarge(42);
        assert_eq!(rb.brain.neuron_count(), 1018);

        // Count cortical columns
        let n_columns = rb
            .regions
            .iter()
            .filter(|r| r.region_type == BrainRegionType::CorticalColumn)
            .count();
        assert_eq!(n_columns, 6);

        // Count cortical neurons
        let n_cortical: usize = rb
            .regions
            .iter()
            .filter(|r| r.region_type == BrainRegionType::CorticalColumn)
            .map(|r| r.neuron_indices.len())
            .sum();
        assert_eq!(n_cortical, 600);

        // Thalamus
        let n_thal: usize = rb
            .regions
            .iter()
            .filter(|r| r.region_type == BrainRegionType::ThalamicNucleus)
            .map(|r| r.neuron_indices.len())
            .sum();
        assert_eq!(n_thal, 150);

        // Hippocampus
        let n_hipp: usize = rb
            .regions
            .iter()
            .filter(|r| r.region_type == BrainRegionType::Hippocampus)
            .map(|r| r.neuron_indices.len())
            .sum();
        assert_eq!(n_hipp, 168);

        // Basal ganglia
        let n_bg: usize = rb
            .regions
            .iter()
            .filter(|r| r.region_type == BrainRegionType::BasalGanglia)
            .map(|r| r.neuron_indices.len())
            .sum();
        assert_eq!(n_bg, 100);

        // Total adds up
        assert_eq!(n_cortical + n_thal + n_hipp + n_bg, 1018);
    }

    #[test]
    fn test_reproducible_seed() {
        let rb1 = RegionalBrain::minimal(42);
        let rb2 = RegionalBrain::minimal(42);

        // Same seed should produce identical synapse counts
        assert_eq!(rb1.brain.synapse_count(), rb2.brain.synapse_count());

        // Same neuron positions
        for i in 0..rb1.brain.neuron_count() {
            assert!((rb1.brain.neurons.x[i] - rb2.brain.neurons.x[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let rb1 = RegionalBrain::minimal(42);
        let rb2 = RegionalBrain::minimal(99);

        // Different seeds should (almost certainly) produce different synapse counts
        // This is probabilistic but extremely unlikely to fail
        let same_count = rb1.brain.synapse_count() == rb2.brain.synapse_count();
        let same_pos = (0..rb1.brain.neuron_count())
            .all(|i| (rb1.brain.neurons.x[i] - rb2.brain.neurons.x[i]).abs() < 1e-6);
        assert!(
            !same_count || !same_pos,
            "Different seeds should produce different networks"
        );
    }

    #[test]
    fn test_stimulate_thalamus() {
        let mut rb = RegionalBrain::minimal(42);
        rb.stimulate_thalamus(20.0);

        let thal_indices = rb.region_indices(BrainRegionType::ThalamicNucleus);
        for &idx in &thal_indices {
            assert!(
                (rb.brain.neurons.external_current[idx] - 20.0).abs() < 1e-6,
                "Thalamic neuron {} should have 20.0 uA/cm^2 current",
                idx,
            );
        }
    }

    #[test]
    fn test_archetypes_assigned() {
        let rb = RegionalBrain::xlarge(42);

        // Cortical: first 80% pyramidal, last 20% interneuron
        for region in &rb.regions {
            if region.region_type == BrainRegionType::CorticalColumn {
                let n = region.neuron_indices.len();
                let n_pyr = (n as f32 * 0.8) as usize;
                for j in 0..n_pyr {
                    assert_eq!(
                        rb.brain.neurons.archetype[region.neuron_indices[j]],
                        NeuronArchetype::Pyramidal as u8
                    );
                }
                for j in n_pyr..n {
                    assert_eq!(
                        rb.brain.neurons.archetype[region.neuron_indices[j]],
                        NeuronArchetype::Interneuron as u8
                    );
                }
            }
        }

        // Basal ganglia: all MediumSpiny
        for region in &rb.regions {
            if region.region_type == BrainRegionType::BasalGanglia {
                for &idx in &region.neuron_indices {
                    assert_eq!(
                        rb.brain.neurons.archetype[idx],
                        NeuronArchetype::MediumSpiny as u8
                    );
                }
            }
        }

        // Thalamus: all Stellate
        for region in &rb.regions {
            if region.region_type == BrainRegionType::ThalamicNucleus {
                for &idx in &region.neuron_indices {
                    assert_eq!(
                        rb.brain.neurons.archetype[idx],
                        NeuronArchetype::Stellate as u8
                    );
                }
            }
        }
    }

    #[test]
    fn test_with_columns() {
        let rb = RegionalBrain::with_columns(3, 42);
        let n_cortical: usize = rb
            .regions
            .iter()
            .filter(|r| r.region_type == BrainRegionType::CorticalColumn)
            .map(|r| r.neuron_indices.len())
            .sum();
        assert_eq!(n_cortical, 300); // 3 columns * 100
    }

    #[test]
    fn test_region_fired_count_initially_zero() {
        let rb = RegionalBrain::minimal(42);
        assert_eq!(rb.region_fired_count(BrainRegionType::CorticalColumn), 0);
        assert_eq!(rb.region_fired_count(BrainRegionType::ThalamicNucleus), 0);
    }
}
