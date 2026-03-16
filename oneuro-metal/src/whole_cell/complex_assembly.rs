//! Named complex assembly logic for whole-cell simulation.
//!
//! Extracted from the monolithic `whole_cell.rs` — contains all methods on
//! `WholeCellSimulator` that deal with named-complex stoichiometry, capacity,
//! supply/demand signals, inventory shares, and the assembly/degradation
//! state machine.

use super::*;

impl WholeCellSimulator {
    fn expression_state_for_operon(
        &self,
        operon: &str,
    ) -> Option<&WholeCellTranscriptionUnitState> {
        self.organism_expression
            .transcription_units
            .iter()
            .find(|unit| unit.name == operon)
    }

    fn operon_gene_count(assets: &WholeCellGenomeAssetPackage, operon: &str) -> usize {
        assets
            .operons
            .iter()
            .find(|candidate| candidate.name == operon)
            .map(|candidate| candidate.genes.len().max(1))
            .unwrap_or(1)
    }

    fn named_complex_total_stoichiometry(complex: &WholeCellComplexSpec) -> f32 {
        complex
            .components
            .iter()
            .map(|component| component.stoichiometry.max(1) as f32)
            .sum::<f32>()
            .max(1.0)
    }

    fn named_complex_component_capacity(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
    ) -> f32 {
        if complex.components.is_empty() {
            return complex.basal_abundance.max(0.0);
        }
        let mut limiting_capacity = f32::INFINITY;
        let mut mean_capacity = 0.0;
        let mut counted = 0usize;
        for component in &complex.components {
            let Some(protein) = assets
                .proteins
                .iter()
                .find(|protein| protein.id == component.protein_id)
            else {
                continue;
            };
            let available = self
                .species_runtime_count(&protein.id)
                .or_else(|| {
                    self.expression_state_for_operon(&protein.operon)
                        .map(|state| {
                            state.mature_protein_abundance
                                / Self::operon_gene_count(assets, &protein.operon) as f32
                        })
                })
                .unwrap_or(protein.basal_abundance.max(0.0))
                .max(0.0);
            let capacity = available / component.stoichiometry.max(1) as f32;
            limiting_capacity = limiting_capacity.min(capacity);
            mean_capacity += capacity;
            counted += 1;
        }
        if counted == 0 {
            complex.basal_abundance.max(0.0)
        } else {
            (0.72 * limiting_capacity + 0.28 * (mean_capacity / counted as f32)).clamp(0.0, 1024.0)
        }
    }

    fn named_complex_component_supply_signal(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
    ) -> f32 {
        if complex.components.is_empty() {
            return 1.0;
        }
        let mut mean_signal = 0.0;
        let mut counted = 0usize;
        for component in &complex.components {
            let Some(protein) = assets
                .proteins
                .iter()
                .find(|protein| protein.id == component.protein_id)
            else {
                continue;
            };
            let unit_abundance = self
                .species_runtime_count(&protein.id)
                .or_else(|| {
                    self.expression_state_for_operon(&protein.operon)
                        .map(|state| {
                            state.protein_abundance
                                / Self::operon_gene_count(assets, &protein.operon) as f32
                        })
                })
                .unwrap_or(protein.basal_abundance.max(0.0));
            let per_subunit = unit_abundance / component.stoichiometry.max(1) as f32;
            let half_saturation = 2.0
                + 0.75 * component.stoichiometry.max(1) as f32
                + 0.01 * protein.aa_length as f32;
            mean_signal += Self::saturating_signal(per_subunit, half_saturation);
            counted += 1;
        }

        if counted == 0 {
            1.0
        } else {
            (mean_signal / counted as f32).clamp(0.0, 1.0)
        }
    }

    fn named_complex_component_satisfaction(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
    ) -> f32 {
        if complex.components.is_empty() {
            return 1.0;
        }
        let mut min_signal: f32 = 1.0;
        let mut mean_signal = 0.0;
        let mut counted = 0usize;
        for component in &complex.components {
            let Some(protein) = assets
                .proteins
                .iter()
                .find(|protein| protein.id == component.protein_id)
            else {
                continue;
            };
            let unit_abundance = self
                .species_runtime_count(&protein.id)
                .or_else(|| {
                    self.expression_state_for_operon(&protein.operon)
                        .map(|state| {
                            state.protein_abundance
                                / Self::operon_gene_count(assets, &protein.operon) as f32
                        })
                })
                .unwrap_or(protein.basal_abundance.max(0.0));
            let half_saturation = 2.0
                + 1.25 * component.stoichiometry.max(1) as f32
                + 0.015 * protein.aa_length as f32;
            let signal = Self::saturating_signal(unit_abundance, half_saturation);
            min_signal = min_signal.min(signal);
            mean_signal += signal;
            counted += 1;
        }

        if counted == 0 {
            1.0
        } else {
            (0.65 * min_signal + 0.35 * (mean_signal / counted as f32)).clamp(0.0, 1.0)
        }
    }

    fn named_complex_limiting_component_signal(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
    ) -> f32 {
        if complex.components.is_empty() {
            return 1.0;
        }
        let mut limiting_signal: f32 = 1.0;
        let mut counted = 0usize;
        for component in &complex.components {
            let Some(protein) = assets
                .proteins
                .iter()
                .find(|protein| protein.id == component.protein_id)
            else {
                continue;
            };
            let unit_abundance = self
                .species_runtime_count(&protein.id)
                .or_else(|| {
                    self.expression_state_for_operon(&protein.operon)
                        .map(|state| {
                            state.mature_protein_abundance
                                / Self::operon_gene_count(assets, &protein.operon) as f32
                        })
                })
                .unwrap_or(protein.basal_abundance.max(0.0));
            let per_subunit = unit_abundance / component.stoichiometry.max(1) as f32;
            let half_saturation = 2.0 + 1.5 * component.stoichiometry.max(1) as f32;
            limiting_signal =
                limiting_signal.min(Self::saturating_signal(per_subunit, half_saturation));
            counted += 1;
        }
        if counted == 0 {
            1.0
        } else {
            limiting_signal.clamp(0.0, 1.0)
        }
    }

    fn named_complex_component_demand_map(
        &self,
        assets: &WholeCellGenomeAssetPackage,
    ) -> HashMap<String, f32> {
        let mut demand = HashMap::new();
        for (state, complex) in self.named_complexes.iter().zip(assets.complexes.iter()) {
            let assembly_drive =
                (state.target_abundance.max(state.abundance) + state.stalled_intermediate).max(0.0);
            for component in &complex.components {
                *demand.entry(component.protein_id.clone()).or_insert(0.0) +=
                    assembly_drive * component.stoichiometry.max(1) as f32;
            }
        }
        demand
    }

    fn named_complex_shared_component_pressure(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
        component_demand: &HashMap<String, f32>,
    ) -> f32 {
        if complex.components.is_empty() {
            return 0.0;
        }
        let mut mean_pressure = 0.0;
        let mut counted = 0usize;
        for component in &complex.components {
            let Some(protein) = assets
                .proteins
                .iter()
                .find(|protein| protein.id == component.protein_id)
            else {
                continue;
            };
            let available = self
                .species_runtime_count(&protein.id)
                .or_else(|| {
                    self.expression_state_for_operon(&protein.operon)
                        .map(|state| state.mature_protein_abundance.max(0.0))
                })
                .unwrap_or(protein.basal_abundance.max(0.0))
                .max(1.0);
            let demand = component_demand
                .get(&component.protein_id)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let pressure = ((demand / available) - 1.0).max(0.0);
            mean_pressure += pressure;
            counted += 1;
        }
        if counted == 0 {
            0.0
        } else {
            (mean_pressure / counted as f32).clamp(0.0, 4.0)
        }
    }

    fn named_complex_family_gate(&self, complex: &WholeCellComplexSpec) -> f32 {
        let replicated_fraction = self.current_replicated_fraction();
        let division_progress = self.current_division_progress();
        match complex.family {
            WholeCellAssemblyFamily::Ribosome => 1.0,
            WholeCellAssemblyFamily::RnaPolymerase => {
                (0.82 + 0.18 * replicated_fraction).clamp(0.75, 1.15)
            }
            WholeCellAssemblyFamily::Replisome => {
                (0.70 + 0.50 * (1.0 - replicated_fraction)).clamp(0.55, 1.25)
            }
            WholeCellAssemblyFamily::ReplicationInitiator => {
                (0.86 + 0.28 * (1.0 - replicated_fraction) + 0.12 * division_progress)
                    .clamp(0.70, 1.30)
            }
            WholeCellAssemblyFamily::AtpSynthase => Self::finite_scale(
                0.55 * self.chemistry_report.atp_support
                    + 0.25 * self.organism_expression.membrane_support
                    + 0.20 * self.localized_supply_scale(),
                1.0,
                0.55,
                1.25,
            ),
            WholeCellAssemblyFamily::Transporter => Self::finite_scale(
                0.60 * self.localized_supply_scale()
                    + 0.25 * self.organism_expression.membrane_support
                    + 0.15 * self.organism_expression.energy_support,
                1.0,
                0.55,
                1.25,
            ),
            WholeCellAssemblyFamily::MembraneEnzyme => Self::finite_scale(
                0.55 * self.organism_expression.membrane_support
                    + 0.20 * self.localized_supply_scale()
                    + 0.25 * (1.0 - division_progress).clamp(0.0, 1.0),
                1.0,
                0.55,
                1.20,
            ),
            WholeCellAssemblyFamily::ChaperoneClient => Self::finite_scale(
                0.80 + 0.25 * (self.effective_metabolic_load() - 1.0).max(0.0),
                1.0,
                0.75,
                1.40,
            ),
            WholeCellAssemblyFamily::Divisome => {
                (0.55 + 0.80 * replicated_fraction + 0.25 * division_progress).clamp(0.45, 1.45)
            }
            WholeCellAssemblyFamily::Generic => 1.0,
        }
    }

    fn named_complex_structural_support(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
    ) -> f32 {
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        let localized_supply = self.localized_supply_scale();
        let process_scale = self.asset_class_process_scale(complex.asset_class);
        let operon_support = self
            .expression_state_for_operon(&complex.operon)
            .map(|state| {
                Self::finite_scale(
                    0.42 * state.support_level
                        + 0.28 * state.effective_activity
                        + 0.18 * (1.0 / state.stress_penalty.clamp(0.80, 1.60))
                        + 0.12 * localized_supply,
                    1.0,
                    0.55,
                    1.65,
                )
            })
            .unwrap_or(1.0);
        let component_operon_mean = if complex.components.is_empty() {
            1.0
        } else {
            let sum = complex
                .components
                .iter()
                .filter_map(|component| {
                    assets
                        .proteins
                        .iter()
                        .find(|protein| protein.id == component.protein_id)
                        .and_then(|protein| self.expression_state_for_operon(&protein.operon))
                        .map(|state| state.effective_activity)
                })
                .sum::<f32>();
            if sum <= 1.0e-6 {
                1.0
            } else {
                sum / complex.components.len() as f32
            }
        };
        Self::finite_scale(
            0.38 * process_scale
                + 0.32 * operon_support
                + 0.18 * component_operon_mean
                + 0.12 * crowding,
            1.0,
            0.55,
            1.80,
        )
    }

    fn named_complex_subunit_pool_target(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
    ) -> f32 {
        let component_capacity = self
            .named_complex_component_capacity(assets, complex)
            .max(0.1);
        let supply_signal = self.named_complex_component_supply_signal(assets, complex);
        let structural_support = self.named_complex_structural_support(assets, complex);
        let total_stoichiometry = Self::named_complex_total_stoichiometry(complex);
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        (component_capacity
            * (0.58 + 0.92 * supply_signal)
            * (0.82 + 0.18 * structural_support)
            * total_stoichiometry.sqrt()
            * crowding)
            .clamp(0.0, 1024.0)
    }

    pub(super) fn initialize_named_complexes_state(&mut self) -> bool {
        let Some(assets) = self.organism_assets.clone() else {
            self.named_complexes.clear();
            return false;
        };
        if assets.complexes.is_empty() {
            self.named_complexes.clear();
            return false;
        }
        self.named_complexes = assets
            .complexes
            .iter()
            .map(|complex| {
                let component_satisfaction =
                    self.named_complex_component_satisfaction(&assets, complex);
                let component_capacity = self.named_complex_component_capacity(&assets, complex);
                let structural_support = self.named_complex_structural_support(&assets, complex);
                let family_gate = self.named_complex_family_gate(complex);
                let subunit_pool = self.named_complex_subunit_pool_target(&assets, complex);
                let total_stoichiometry = Self::named_complex_total_stoichiometry(complex);
                let target_abundance = (component_capacity
                    * (0.58 + 0.42 * component_satisfaction)
                    * structural_support
                    * self.organism_expression.crowding_penalty.clamp(0.65, 1.10)
                    * family_gate)
                    .clamp(0.0, 512.0);
                let nucleation_intermediate =
                    (0.10 * target_abundance * component_satisfaction * total_stoichiometry.sqrt())
                        .clamp(0.0, 256.0);
                let elongation_intermediate =
                    (0.08 * target_abundance * structural_support * total_stoichiometry.sqrt())
                        .clamp(0.0, 256.0);
                let assembly_progress = (0.42 * component_satisfaction
                    + 0.30 * structural_support
                    + 0.18
                        * Self::saturating_signal(
                            subunit_pool,
                            6.0 + 0.5 * total_stoichiometry.max(1.0),
                        )
                    + 0.10
                        * Self::saturating_signal(
                            nucleation_intermediate + elongation_intermediate,
                            2.0 + 0.4 * total_stoichiometry.max(1.0),
                        ))
                .clamp(0.0, 1.0);
                let limiting_component_signal =
                    self.named_complex_limiting_component_signal(&assets, complex);
                let insertion_progress = if complex.membrane_inserted {
                    (0.58 * structural_support + 0.42 * component_satisfaction).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                WholeCellNamedComplexState {
                    id: complex.id.clone(),
                    operon: complex.operon.clone(),
                    asset_class: complex.asset_class,
                    family: complex.family,
                    subsystem_targets: complex.subsystem_targets.clone(),
                    subunit_pool,
                    nucleation_intermediate,
                    elongation_intermediate,
                    abundance: target_abundance,
                    target_abundance,
                    assembly_rate: 0.0,
                    degradation_rate: 0.0,
                    nucleation_rate: 0.0,
                    elongation_rate: 0.0,
                    maturation_rate: 0.0,
                    component_satisfaction,
                    structural_support,
                    assembly_progress,
                    stalled_intermediate: 0.0,
                    damaged_abundance: 0.0,
                    limiting_component_signal,
                    shared_component_pressure: 0.0,
                    insertion_progress,
                    failure_count: 0.0,
                }
            })
            .collect();
        true
    }

    pub(super) fn aggregate_named_complex_assembly_state(
        &self,
        assets: &WholeCellGenomeAssetPackage,
    ) -> WholeCellComplexAssemblyState {
        let mut aggregate = WholeCellComplexAssemblyState::default();
        for (state, complex) in self.named_complexes.iter().zip(assets.complexes.iter()) {
            let insertion_gate = if complex.membrane_inserted {
                (0.65 + 0.35 * state.insertion_progress).clamp(0.40, 1.0)
            } else {
                1.0
            };
            let damage_penalty = (1.0
                - 0.45
                    * Self::saturating_signal(
                        state.damaged_abundance,
                        1.0 + 0.2 * state.target_abundance.max(1.0),
                    ))
            .clamp(0.35, 1.0);
            let effective_abundance = (state.abundance * insertion_gate * damage_penalty)
                + 0.22 * state.elongation_intermediate
                + 0.08 * state.nucleation_intermediate;
            let effective_target = (state.target_abundance * insertion_gate)
                + 0.18 * state.elongation_intermediate
                + 0.06 * state.nucleation_intermediate
                + 0.04 * state.stalled_intermediate;
            let effective_assembly_rate = state.assembly_rate
                + 0.55 * state.maturation_rate
                + 0.30 * state.elongation_rate
                + 0.15 * state.nucleation_rate;
            let shares = self.named_complex_inventory_shares(complex);
            let atp_share = shares.atp_band;
            let ribosome_share = shares.ribosome;
            let rnap_share = shares.rnap;
            let replisome_share = shares.replisome;
            let membrane_share = shares.membrane;
            let constriction_share = shares.ftsz;
            let dnaa_share = shares.dnaa;

            aggregate.atp_band_complexes += effective_abundance * atp_share;
            aggregate.ribosome_complexes += effective_abundance * ribosome_share;
            aggregate.rnap_complexes += effective_abundance * rnap_share;
            aggregate.replisome_complexes += effective_abundance * replisome_share;
            aggregate.membrane_complexes += effective_abundance * membrane_share;
            aggregate.ftsz_polymer += effective_abundance * constriction_share;
            aggregate.dnaa_activity += effective_abundance * dnaa_share;

            aggregate.atp_band_target += effective_target * atp_share;
            aggregate.ribosome_target += effective_target * ribosome_share;
            aggregate.rnap_target += effective_target * rnap_share;
            aggregate.replisome_target += effective_target * replisome_share;
            aggregate.membrane_target += effective_target * membrane_share;
            aggregate.ftsz_target += effective_target * constriction_share;
            aggregate.dnaa_target += effective_target * dnaa_share;

            aggregate.atp_band_assembly_rate += effective_assembly_rate * atp_share;
            aggregate.ribosome_assembly_rate += effective_assembly_rate * ribosome_share;
            aggregate.rnap_assembly_rate += effective_assembly_rate * rnap_share;
            aggregate.replisome_assembly_rate += effective_assembly_rate * replisome_share;
            aggregate.membrane_assembly_rate += effective_assembly_rate * membrane_share;
            aggregate.ftsz_assembly_rate += effective_assembly_rate * constriction_share;
            aggregate.dnaa_assembly_rate += effective_assembly_rate * dnaa_share;

            aggregate.atp_band_degradation_rate += state.degradation_rate * atp_share;
            aggregate.ribosome_degradation_rate += state.degradation_rate * ribosome_share;
            aggregate.rnap_degradation_rate += state.degradation_rate * rnap_share;
            aggregate.replisome_degradation_rate += state.degradation_rate * replisome_share;
            aggregate.membrane_degradation_rate += state.degradation_rate * membrane_share;
            aggregate.ftsz_degradation_rate += state.degradation_rate * constriction_share;
            aggregate.dnaa_degradation_rate += state.degradation_rate * dnaa_share;
        }
        aggregate
    }

    pub(super) fn aggregate_named_complex_assembly_state_without_assets(
        &self,
    ) -> WholeCellComplexAssemblyState {
        let mut aggregate = WholeCellComplexAssemblyState::default();
        for state in &self.named_complexes {
            let shares = self.named_complex_inventory_shares_from_state(state);
            if shares.total() <= 1.0e-6 {
                continue;
            }
            // Bundle-less explicit state still carries family, subsystem, and
            // asset-class metadata, so keep using that compiled semantic
            // ownership instead of dropping back to stale scalar summaries.
            let insertion_gate = if self.named_complex_state_requires_insertion(state) {
                (0.65 + 0.35 * state.insertion_progress).clamp(0.40, 1.0)
            } else {
                1.0
            };
            let damage_penalty = (1.0
                - 0.45
                    * Self::saturating_signal(
                        state.damaged_abundance,
                        1.0 + 0.2 * state.target_abundance.max(1.0),
                    ))
            .clamp(0.35, 1.0);
            let effective_abundance = (state.abundance * insertion_gate * damage_penalty)
                + 0.22 * state.elongation_intermediate
                + 0.08 * state.nucleation_intermediate;
            let effective_target = (state.target_abundance * insertion_gate)
                + 0.18 * state.elongation_intermediate
                + 0.06 * state.nucleation_intermediate
                + 0.04 * state.stalled_intermediate;
            let effective_assembly_rate = state.assembly_rate
                + 0.55 * state.maturation_rate
                + 0.30 * state.elongation_rate
                + 0.15 * state.nucleation_rate;

            aggregate.atp_band_complexes += effective_abundance * shares.atp_band;
            aggregate.ribosome_complexes += effective_abundance * shares.ribosome;
            aggregate.rnap_complexes += effective_abundance * shares.rnap;
            aggregate.replisome_complexes += effective_abundance * shares.replisome;
            aggregate.membrane_complexes += effective_abundance * shares.membrane;
            aggregate.ftsz_polymer += effective_abundance * shares.ftsz;
            aggregate.dnaa_activity += effective_abundance * shares.dnaa;

            aggregate.atp_band_target += effective_target * shares.atp_band;
            aggregate.ribosome_target += effective_target * shares.ribosome;
            aggregate.rnap_target += effective_target * shares.rnap;
            aggregate.replisome_target += effective_target * shares.replisome;
            aggregate.membrane_target += effective_target * shares.membrane;
            aggregate.ftsz_target += effective_target * shares.ftsz;
            aggregate.dnaa_target += effective_target * shares.dnaa;

            aggregate.atp_band_assembly_rate += effective_assembly_rate * shares.atp_band;
            aggregate.ribosome_assembly_rate += effective_assembly_rate * shares.ribosome;
            aggregate.rnap_assembly_rate += effective_assembly_rate * shares.rnap;
            aggregate.replisome_assembly_rate += effective_assembly_rate * shares.replisome;
            aggregate.membrane_assembly_rate += effective_assembly_rate * shares.membrane;
            aggregate.ftsz_assembly_rate += effective_assembly_rate * shares.ftsz;
            aggregate.dnaa_assembly_rate += effective_assembly_rate * shares.dnaa;

            aggregate.atp_band_degradation_rate += state.degradation_rate * shares.atp_band;
            aggregate.ribosome_degradation_rate += state.degradation_rate * shares.ribosome;
            aggregate.rnap_degradation_rate += state.degradation_rate * shares.rnap;
            aggregate.replisome_degradation_rate += state.degradation_rate * shares.replisome;
            aggregate.membrane_degradation_rate += state.degradation_rate * shares.membrane;
            aggregate.ftsz_degradation_rate += state.degradation_rate * shares.ftsz;
            aggregate.dnaa_degradation_rate += state.degradation_rate * shares.dnaa;
        }
        aggregate
    }

    fn named_complex_state_requires_insertion(&self, state: &WholeCellNamedComplexState) -> bool {
        state
            .subsystem_targets
            .contains(&Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            || matches!(
                state.family,
                WholeCellAssemblyFamily::AtpSynthase
                    | WholeCellAssemblyFamily::Transporter
                    | WholeCellAssemblyFamily::MembraneEnzyme
                    | WholeCellAssemblyFamily::Divisome
            )
            || matches!(
                state.asset_class,
                WholeCellAssetClass::Energy | WholeCellAssetClass::Membrane
            )
    }

    fn heuristic_complex_inventory_shares(
        &self,
        complex: &WholeCellComplexSpec,
    ) -> WholeCellAssemblyChannelShares {
        let mut weights = complex.process_weights.clamped();
        for target in &complex.subsystem_targets {
            match target {
                Syn3ASubsystemPreset::AtpSynthaseMembraneBand => {
                    weights.energy += 1.2;
                    weights.membrane += 0.35;
                }
                Syn3ASubsystemPreset::RibosomePolysomeCluster => {
                    weights.translation += 1.25;
                    weights.transcription += 0.20;
                }
                Syn3ASubsystemPreset::ReplisomeTrack => {
                    weights.replication += 1.15;
                    weights.segregation += 0.55;
                }
                Syn3ASubsystemPreset::FtsZSeptumRing => {
                    weights.constriction += 1.25;
                    weights.membrane += 0.30;
                }
            }
        }
        if weights.transcription <= 1.0e-6 {
            weights.transcription += 0.18
                * (weights.translation + weights.replication + weights.membrane + weights.energy)
                    .max(0.1);
        }
        let total = weights.total().max(1.0e-6);
        WholeCellAssemblyChannelShares {
            atp_band: (weights.energy + 0.20 * weights.membrane) / total,
            ribosome: (weights.translation + 0.10 * weights.transcription) / total,
            rnap: (weights.transcription + 0.10 * weights.translation) / total,
            replisome: (weights.replication + 0.70 * weights.segregation) / total,
            membrane: (weights.membrane + 0.15 * weights.energy) / total,
            ftsz: (weights.constriction + 0.20 * weights.membrane) / total,
            dnaa: (0.65 * weights.replication + 0.35 * weights.transcription) / total,
        }
    }

    fn subsystem_target_inventory_shares(
        subsystem_targets: &[Syn3ASubsystemPreset],
    ) -> WholeCellAssemblyChannelShares {
        let mut shares = WholeCellAssemblyChannelShares::default();
        for target in subsystem_targets {
            match target {
                Syn3ASubsystemPreset::AtpSynthaseMembraneBand => {
                    shares.atp_band += 1.0;
                }
                Syn3ASubsystemPreset::RibosomePolysomeCluster => {
                    shares.ribosome += 1.0;
                }
                Syn3ASubsystemPreset::ReplisomeTrack => {
                    shares.replisome += 1.0;
                }
                Syn3ASubsystemPreset::FtsZSeptumRing => {
                    shares.ftsz += 1.0;
                }
            }
        }
        shares
    }

    fn family_inventory_shares(family: WholeCellAssemblyFamily) -> WholeCellAssemblyChannelShares {
        match family {
            WholeCellAssemblyFamily::Ribosome => WholeCellAssemblyChannelShares {
                ribosome: 1.0,
                ..WholeCellAssemblyChannelShares::default()
            },
            WholeCellAssemblyFamily::RnaPolymerase => WholeCellAssemblyChannelShares {
                rnap: 1.0,
                ..WholeCellAssemblyChannelShares::default()
            },
            WholeCellAssemblyFamily::Replisome => WholeCellAssemblyChannelShares {
                replisome: 1.0,
                ..WholeCellAssemblyChannelShares::default()
            },
            WholeCellAssemblyFamily::ReplicationInitiator => WholeCellAssemblyChannelShares {
                dnaa: 1.0,
                ..WholeCellAssemblyChannelShares::default()
            },
            WholeCellAssemblyFamily::AtpSynthase => WholeCellAssemblyChannelShares {
                atp_band: 1.0,
                ..WholeCellAssemblyChannelShares::default()
            },
            WholeCellAssemblyFamily::Transporter | WholeCellAssemblyFamily::MembraneEnzyme => {
                WholeCellAssemblyChannelShares {
                    membrane: 1.0,
                    ..WholeCellAssemblyChannelShares::default()
                }
            }
            WholeCellAssemblyFamily::Divisome => WholeCellAssemblyChannelShares {
                ftsz: 1.0,
                ..WholeCellAssemblyChannelShares::default()
            },
            WholeCellAssemblyFamily::ChaperoneClient | WholeCellAssemblyFamily::Generic => {
                WholeCellAssemblyChannelShares::default()
            }
        }
    }

    fn asset_class_inventory_shares(
        asset_class: WholeCellAssetClass,
    ) -> WholeCellAssemblyChannelShares {
        match asset_class {
            WholeCellAssetClass::Energy => WholeCellAssemblyChannelShares {
                atp_band: 1.0,
                ..WholeCellAssemblyChannelShares::default()
            },
            WholeCellAssetClass::Translation => WholeCellAssemblyChannelShares {
                ribosome: 1.0,
                ..WholeCellAssemblyChannelShares::default()
            },
            WholeCellAssetClass::Replication | WholeCellAssetClass::Segregation => {
                WholeCellAssemblyChannelShares {
                    replisome: 1.0,
                    ..WholeCellAssemblyChannelShares::default()
                }
            }
            WholeCellAssetClass::Membrane => WholeCellAssemblyChannelShares {
                membrane: 1.0,
                ..WholeCellAssemblyChannelShares::default()
            },
            WholeCellAssetClass::Constriction => WholeCellAssemblyChannelShares {
                ftsz: 1.0,
                ..WholeCellAssemblyChannelShares::default()
            },
            WholeCellAssetClass::QualityControl
            | WholeCellAssetClass::Homeostasis
            | WholeCellAssetClass::Generic => WholeCellAssemblyChannelShares::default(),
        }
    }

    fn named_complex_inventory_shares(
        &self,
        complex: &WholeCellComplexSpec,
    ) -> WholeCellAssemblyChannelShares {
        let mut shares = Self::subsystem_target_inventory_shares(&complex.subsystem_targets);
        if shares.total() > 1.0e-6 {
            return shares.normalized();
        }

        shares = Self::family_inventory_shares(complex.family);
        if shares.total() > 1.0e-6 {
            return shares.normalized();
        }

        shares = Self::asset_class_inventory_shares(complex.asset_class);
        if shares.total() > 1.0e-6 {
            return shares.normalized();
        }

        self.heuristic_complex_inventory_shares(complex)
    }

    pub(super) fn named_complex_inventory_shares_from_state(
        &self,
        state: &WholeCellNamedComplexState,
    ) -> WholeCellAssemblyChannelShares {
        let mut shares = Self::subsystem_target_inventory_shares(&state.subsystem_targets);
        if shares.total() > 1.0e-6 {
            return shares.normalized();
        }

        shares = Self::family_inventory_shares(state.family);
        if shares.total() > 1.0e-6 {
            return shares.normalized();
        }

        shares = Self::asset_class_inventory_shares(state.asset_class);
        if shares.total() > 1.0e-6 {
            return shares.normalized();
        }

        WholeCellAssemblyChannelShares::default()
    }

    fn update_named_complexes_state(&mut self, dt: f32) -> bool {
        let Some(assets) = self.organism_assets.clone() else {
            if !self.named_complexes.is_empty() {
                self.complex_assembly =
                    self.aggregate_named_complex_assembly_state_without_assets();
                return true;
            }
            self.named_complexes.clear();
            return false;
        };
        if assets.complexes.is_empty() {
            if !self.named_complexes.is_empty() {
                self.complex_assembly =
                    self.aggregate_named_complex_assembly_state_without_assets();
                return true;
            }
            self.named_complexes.clear();
            return false;
        }
        if self.named_complexes.len() != assets.complexes.len() {
            self.initialize_named_complexes_state();
        }

        let dt_scale = (dt / self.config.dt_ms.max(0.05)).clamp(0.5, 6.0);
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        let effective_load = self.effective_metabolic_load();
        let degradation_pressure =
            (0.68 + 0.22 * (effective_load - 1.0).max(0.0) + 0.16 * (1.0 - crowding).max(0.0))
                .clamp(0.60, 1.80);
        let component_demand = self.named_complex_component_demand_map(&assets);

        let updated_states = self
            .named_complexes
            .iter()
            .zip(assets.complexes.iter())
            .map(|(state, complex)| {
                let component_satisfaction =
                    self.named_complex_component_satisfaction(&assets, complex);
                let limiting_component_signal =
                    self.named_complex_limiting_component_signal(&assets, complex);
                let component_capacity = self.named_complex_component_capacity(&assets, complex);
                let component_supply_signal =
                    self.named_complex_component_supply_signal(&assets, complex);
                let structural_support = self.named_complex_structural_support(&assets, complex);
                let shared_component_pressure = self.named_complex_shared_component_pressure(
                    &assets,
                    complex,
                    &component_demand,
                );
                let subunit_pool_target = self.named_complex_subunit_pool_target(&assets, complex);
                let total_stoichiometry = Self::named_complex_total_stoichiometry(complex);
                let complexity_penalty = 1.0 / total_stoichiometry.sqrt().max(1.0);
                let family_gate = self.named_complex_family_gate(complex);
                let assembly_support = Self::finite_scale(
                    0.36 * component_satisfaction
                        + 0.20 * limiting_component_signal
                        + 0.24 * structural_support
                        + 0.12 * crowding
                        + 0.08 * family_gate
                        - 0.12 * shared_component_pressure,
                    1.0,
                    0.45,
                    1.75,
                );
                let target_abundance = (component_capacity
                    * (0.48 + 0.28 * component_satisfaction + 0.24 * limiting_component_signal)
                    * structural_support
                    * crowding
                    * family_gate)
                    * (1.0 - 0.12 * shared_component_pressure.clamp(0.0, 1.5))
                        .clamp(0.35, 1.0)
                        .clamp(0.0, 512.0);
                let subunit_supply_rate = (subunit_pool_target - state.subunit_pool).max(0.0)
                    * (0.10 + 0.16 * component_supply_signal)
                    * (1.0 - 0.18 * shared_component_pressure.clamp(0.0, 1.5));
                let subunit_turnover_rate = state.subunit_pool
                    * (0.010
                        + 0.012 * (1.0 - component_satisfaction).max(0.0)
                        + 0.008 * (1.0 - crowding).max(0.0));
                let stall_pressure = (0.30 * (1.0 - limiting_component_signal).max(0.0)
                    + 0.24 * (1.0 - structural_support).max(0.0)
                    + 0.22 * shared_component_pressure
                    + 0.18 * (1.0 - family_gate).max(0.0)
                    + 0.12 * (degradation_pressure - 1.0).max(0.0))
                .clamp(0.0, 2.0);
                let nucleation_rate = state.subunit_pool
                    * (0.012 + 0.020 * component_satisfaction)
                    * complexity_penalty
                    * assembly_support
                    * (1.0 - 0.35 * stall_pressure.min(1.0));
                let nucleation_turnover_rate = state.nucleation_intermediate
                    * (0.012
                        + 0.012 * (1.0 - structural_support).max(0.0)
                        + 0.008 * (1.0 - component_satisfaction).max(0.0)
                        + 0.010 * stall_pressure);
                let elongation_rate = state.nucleation_intermediate
                    * (0.016 + 0.024 * structural_support)
                    * (0.82 + 0.18 * component_supply_signal)
                    * (1.0 - 0.28 * stall_pressure.min(1.0));
                let elongation_turnover_rate = state.elongation_intermediate
                    * (0.010
                        + 0.010 * (1.0 - structural_support).max(0.0)
                        + 0.006 * (1.0 - crowding).max(0.0)
                        + 0.012 * stall_pressure);
                let insertion_target = if complex.membrane_inserted {
                    (0.48 * structural_support
                        + 0.26 * component_satisfaction
                        + 0.16 * family_gate
                        + 0.10 * crowding)
                        .clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let insertion_progress = if complex.membrane_inserted {
                    (state.insertion_progress
                        + dt_scale
                            * ((0.020 + 0.028 * structural_support) * insertion_target
                                - state.insertion_progress
                                    * (0.010
                                        + 0.012 * stall_pressure
                                        + 0.008 * (1.0 - crowding).max(0.0))))
                    .clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let maturation_rate = state.elongation_intermediate
                    * (0.020 + 0.032 * assembly_support)
                    * (0.80 + 0.20 * component_satisfaction)
                    * (0.75 + 0.25 * insertion_progress)
                    * (1.0 - 0.30 * stall_pressure.min(1.0));
                let stalled_resolution_rate =
                    state.stalled_intermediate * (0.010 + 0.014 * structural_support);
                let stalled_intermediate = (state.stalled_intermediate
                    + dt_scale
                        * (0.42 * nucleation_turnover_rate
                            + 0.38 * elongation_turnover_rate
                            + 0.30
                                * stall_pressure
                                * (state.nucleation_intermediate + state.elongation_intermediate)
                            - stalled_resolution_rate))
                    .clamp(0.0, 512.0);
                let channel_degradation_pressure = (degradation_pressure
                    + 0.45 * (1.0 - component_satisfaction).max(0.0)
                    + 0.20 * (1.0 - structural_support).max(0.0)
                    + 0.24 * stall_pressure)
                    .clamp(0.60, 2.10);
                let (abundance, assembly_rate, degradation_rate) = Self::complex_channel_step(
                    state.abundance,
                    target_abundance,
                    assembly_support,
                    channel_degradation_pressure,
                    dt_scale,
                    512.0,
                );
                let subunit_pool = (state.subunit_pool
                    + dt_scale
                        * (subunit_supply_rate - 0.70 * nucleation_rate - subunit_turnover_rate))
                    .clamp(0.0, 2048.0);
                let nucleation_intermediate = (state.nucleation_intermediate
                    + dt_scale
                        * (nucleation_rate - 0.65 * elongation_rate - nucleation_turnover_rate))
                    .clamp(0.0, 512.0);
                let elongation_intermediate = (state.elongation_intermediate
                    + dt_scale
                        * (elongation_rate - 0.75 * maturation_rate - elongation_turnover_rate))
                    .clamp(0.0, 512.0);
                let damaged_abundance = (state.damaged_abundance
                    + dt_scale
                        * (abundance
                            * (0.004
                                + 0.006 * stall_pressure
                                + 0.004 * shared_component_pressure)
                            - state.damaged_abundance
                                * (0.008 + 0.010 * structural_support + 0.006 * family_gate)))
                    .clamp(0.0, 512.0);
                let assembly_progress = (0.36 * component_satisfaction
                    + 0.26 * structural_support
                    + 0.18
                        * Self::saturating_signal(
                            subunit_pool,
                            6.0 + 0.5 * total_stoichiometry.max(1.0),
                        )
                    + 0.20
                        * Self::saturating_signal(
                            nucleation_intermediate + elongation_intermediate,
                            2.0 + 0.35 * total_stoichiometry.max(1.0),
                        ))
                .clamp(0.0, 1.0);
                let failure_count = (state.failure_count
                    + dt_scale * (0.08 * stall_pressure + 0.04 * shared_component_pressure))
                    .clamp(0.0, 1.0e6);
                WholeCellNamedComplexState {
                    id: state.id.clone(),
                    operon: state.operon.clone(),
                    asset_class: state.asset_class,
                    family: state.family,
                    subsystem_targets: state.subsystem_targets.clone(),
                    subunit_pool,
                    nucleation_intermediate,
                    elongation_intermediate,
                    abundance,
                    target_abundance,
                    assembly_rate,
                    degradation_rate,
                    nucleation_rate,
                    elongation_rate,
                    maturation_rate,
                    component_satisfaction,
                    structural_support,
                    assembly_progress,
                    stalled_intermediate,
                    damaged_abundance,
                    limiting_component_signal,
                    shared_component_pressure,
                    insertion_progress,
                    failure_count,
                }
            })
            .collect();

        self.named_complexes = updated_states;
        self.complex_assembly = self.aggregate_named_complex_assembly_state(&assets);
        true
    }

    pub(super) fn derived_complex_assembly_target(&self) -> WholeCellComplexAssemblyState {
        let prior = self.legacy_fallback_assembly_inventory();
        let expression = &self.organism_expression;
        if !self.has_explicit_expression_state() {
            let replicated_fraction = self.current_replicated_fraction();
            let energy_signal = Self::saturating_signal(
                0.70 * self.atp_mm.max(0.0) + 0.30 * self.chemistry_report.atp_support.max(0.0),
                1.2,
            );
            let transcription_signal = Self::saturating_signal(
                0.65 * self.nucleotides_mm.max(0.0) + 0.35 * self.glucose_mm.max(0.0),
                1.0,
            );
            let translation_signal = Self::saturating_signal(
                0.70 * self.amino_acids_mm.max(0.0) + 0.30 * self.md_translation_scale.max(0.0),
                1.1,
            );
            let membrane_signal = Self::saturating_signal(
                0.75 * self.membrane_precursors_mm.max(0.0)
                    + 0.25 * self.md_membrane_scale.max(0.0),
                0.8,
            );
            return WholeCellComplexAssemblyState {
                atp_band_target: (prior.atp_band_complexes * (0.76 + 0.44 * energy_signal))
                    .clamp(4.0, 512.0),
                ribosome_target: (prior.ribosome_complexes * (0.74 + 0.42 * translation_signal))
                    .clamp(8.0, 640.0),
                rnap_target: (prior.rnap_complexes * (0.76 + 0.38 * transcription_signal))
                    .clamp(6.0, 384.0),
                replisome_target: (prior.replisome_complexes
                    * (0.74 + 0.42 * transcription_signal)
                    * (0.82 + 0.24 * (1.0 - replicated_fraction)))
                    .clamp(2.0, 192.0),
                membrane_target: (prior.membrane_complexes * (0.74 + 0.40 * membrane_signal))
                    .clamp(4.0, 384.0),
                ftsz_target: (prior.ftsz_polymer
                    * (0.72 + 0.36 * membrane_signal + 0.36 * replicated_fraction))
                    .clamp(8.0, 768.0),
                dnaa_target: (prior.dnaa_activity
                    * (0.74 + 0.34 * transcription_signal)
                    * (0.84 + 0.20 * (1.0 - replicated_fraction)))
                    .clamp(4.0, 256.0),
                ..prior
            };
        }

        let mut protein_drive = WholeCellProcessWeights::default();
        let mut total_signal = 0.0;
        for unit in &expression.transcription_units {
            let assembly_mass = (0.76 * unit.protein_abundance.max(0.0)
                + 0.24 * unit.transcript_abundance.max(0.0))
                * unit.support_level.clamp(0.55, 1.55)
                / unit.stress_penalty.clamp(0.80, 1.60);
            protein_drive.add_weighted(unit.process_drive, assembly_mass);
            total_signal += unit.process_drive.total() * assembly_mass;
        }

        let mean_signal = if total_signal > 1.0e-6 {
            total_signal / 7.0
        } else {
            1.0
        };
        let protein_signal = Self::saturating_signal(
            expression.total_protein_abundance,
            36.0 + 3.0 * expression.transcription_units.len() as f32,
        );
        let transcript_signal = Self::saturating_signal(
            expression.total_transcript_abundance,
            24.0 + 2.0 * expression.transcription_units.len() as f32,
        );
        let localized_supply = self.localized_supply_scale();
        let crowding = expression.crowding_penalty.clamp(0.65, 1.10);
        let replicated_fraction = self.current_replicated_fraction();

        let energy_scale = Self::finite_scale(
            0.45 * expression.process_scales.energy
                + 0.30
                    * Self::complex_assembly_signal_scale(protein_drive.energy, mean_signal, 1.0)
                + 0.15 * expression.energy_support
                + 0.10 * localized_supply,
            1.0,
            0.55,
            1.85,
        );
        let transcription_scale = Self::finite_scale(
            0.42 * expression.process_scales.transcription
                + 0.30
                    * Self::complex_assembly_signal_scale(
                        protein_drive.transcription,
                        mean_signal,
                        1.0,
                    )
                + 0.18 * expression.translation_support
                + 0.10 * transcript_signal,
            1.0,
            0.55,
            1.85,
        );
        let translation_scale = Self::finite_scale(
            0.40 * expression.process_scales.translation
                + 0.34
                    * Self::complex_assembly_signal_scale(
                        protein_drive.translation,
                        mean_signal,
                        1.0,
                    )
                + 0.16 * expression.translation_support
                + 0.10 * protein_signal,
            1.0,
            0.55,
            1.95,
        );
        let replication_signal = 0.5 * (protein_drive.replication + protein_drive.segregation);
        let replication_scale = Self::finite_scale(
            0.38 * expression.process_scales.replication
                + 0.26 * Self::complex_assembly_signal_scale(replication_signal, mean_signal, 1.0)
                + 0.22 * expression.nucleotide_support
                + 0.14 * (1.0 - 0.35 * replicated_fraction).clamp(0.65, 1.15),
            1.0,
            0.55,
            1.95,
        );
        let membrane_scale = Self::finite_scale(
            0.44 * expression.process_scales.membrane
                + 0.30
                    * Self::complex_assembly_signal_scale(protein_drive.membrane, mean_signal, 1.0)
                + 0.16 * expression.membrane_support
                + 0.10 * protein_signal,
            1.0,
            0.55,
            1.95,
        );
        let constriction_scale = Self::finite_scale(
            0.36 * expression.process_scales.constriction
                + 0.26
                    * Self::complex_assembly_signal_scale(
                        protein_drive.constriction,
                        mean_signal,
                        1.0,
                    )
                + 0.18 * expression.membrane_support
                + 0.20 * (0.70 + 0.60 * replicated_fraction).clamp(0.70, 1.30),
            1.0,
            0.55,
            2.10,
        );
        let replication_window = (0.78 + 0.30 * (1.0 - replicated_fraction)).clamp(0.60, 1.20);
        let constriction_window = (0.65 + 0.60 * replicated_fraction).clamp(0.65, 1.35);

        WholeCellComplexAssemblyState {
            atp_band_complexes: prior.atp_band_complexes,
            ribosome_complexes: prior.ribosome_complexes,
            rnap_complexes: prior.rnap_complexes,
            replisome_complexes: prior.replisome_complexes,
            membrane_complexes: prior.membrane_complexes,
            ftsz_polymer: prior.ftsz_polymer,
            dnaa_activity: prior.dnaa_activity,
            atp_band_target: (prior.atp_band_complexes
                * energy_scale
                * crowding
                * (0.72 + 0.28 * protein_signal))
                .clamp(4.0, 512.0),
            ribosome_target: (prior.ribosome_complexes
                * translation_scale
                * crowding
                * (0.68 + 0.32 * protein_signal))
                .clamp(8.0, 640.0),
            rnap_target: (prior.rnap_complexes
                * transcription_scale
                * crowding
                * (0.72 + 0.28 * transcript_signal))
                .clamp(6.0, 384.0),
            replisome_target: (prior.replisome_complexes
                * replication_scale
                * replication_window
                * (0.72 + 0.28 * transcript_signal))
                .clamp(2.0, 192.0),
            membrane_target: (prior.membrane_complexes
                * membrane_scale
                * crowding
                * (0.72 + 0.28 * protein_signal))
                .clamp(4.0, 384.0),
            ftsz_target: (prior.ftsz_polymer
                * constriction_scale
                * constriction_window
                * (0.66 + 0.34 * protein_signal))
                .clamp(8.0, 768.0),
            dnaa_target: (prior.dnaa_activity
                * replication_scale
                * replication_window
                * (0.70 + 0.30 * transcript_signal))
                .clamp(4.0, 256.0),
            ..WholeCellComplexAssemblyState::default()
        }
    }

    pub(super) fn initialize_complex_assembly_state(&mut self) {
        if self.initialize_named_complexes_state() {
            if let Some(assets) = self.organism_assets.as_ref() {
                self.complex_assembly = self.aggregate_named_complex_assembly_state(assets);
                return;
            }
        }
        if self.organism_assets.is_some() {
            self.complex_assembly = WholeCellComplexAssemblyState::default();
            return;
        }
        let target = self.derived_complex_assembly_target();
        self.complex_assembly = WholeCellComplexAssemblyState {
            atp_band_complexes: target.atp_band_target,
            ribosome_complexes: target.ribosome_target,
            rnap_complexes: target.rnap_target,
            replisome_complexes: target.replisome_target,
            membrane_complexes: target.membrane_target,
            ftsz_polymer: target.ftsz_target,
            dnaa_activity: target.dnaa_target,
            ..target
        };
    }

    pub(super) fn update_complex_assembly_state(&mut self, dt: f32) {
        if self.update_named_complexes_state(dt) {
            self.sync_runtime_process_species(dt);
            return;
        }
        if self.organism_assets.is_some() {
            if self.complex_assembly.total_complexes() <= 1.0e-6 {
                self.complex_assembly = WholeCellComplexAssemblyState::default();
            }
            self.sync_runtime_process_species(dt);
            return;
        }
        let target = self.derived_complex_assembly_target();
        let current = if self.complex_assembly.total_complexes() > 1.0e-6 {
            self.complex_assembly
        } else {
            WholeCellComplexAssemblyState {
                atp_band_complexes: target.atp_band_target,
                ribosome_complexes: target.ribosome_target,
                rnap_complexes: target.rnap_target,
                replisome_complexes: target.replisome_target,
                membrane_complexes: target.membrane_target,
                ftsz_polymer: target.ftsz_target,
                dnaa_activity: target.dnaa_target,
                ..target
            }
        };

        let dt_scale = (dt / self.config.dt_ms.max(0.05)).clamp(0.5, 6.0);
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        let effective_load = self.effective_metabolic_load();
        let degradation_pressure =
            (0.70 + 0.24 * (effective_load - 1.0).max(0.0) + 0.18 * (1.0 - crowding).max(0.0))
                .clamp(0.65, 1.80);

        let energy_support = Self::finite_scale(
            0.55 * self.organism_expression.energy_support
                + 0.45 * self.chemistry_report.atp_support,
            1.0,
            0.55,
            1.55,
        );
        let transcription_support = Self::finite_scale(
            0.55 * self.organism_expression.translation_support
                + 0.45 * self.organism_expression.process_scales.transcription,
            1.0,
            0.55,
            1.60,
        );
        let translation_support = Self::finite_scale(
            0.60 * self.organism_expression.translation_support
                + 0.40 * self.organism_expression.process_scales.translation,
            1.0,
            0.55,
            1.60,
        );
        let replication_support = Self::finite_scale(
            0.55 * self.organism_expression.nucleotide_support
                + 0.45
                    * (0.5
                        * (self.organism_expression.process_scales.replication
                            + self.organism_expression.process_scales.segregation)),
            1.0,
            0.55,
            1.60,
        );
        let membrane_support = Self::finite_scale(
            0.60 * self.organism_expression.membrane_support
                + 0.40 * self.organism_expression.process_scales.membrane,
            1.0,
            0.55,
            1.60,
        );
        let constriction_support = Self::finite_scale(
            0.50 * self.organism_expression.membrane_support
                + 0.30 * self.organism_expression.process_scales.constriction
                + 0.20 * (0.70 + 0.60 * self.current_replicated_fraction()),
            1.0,
            0.55,
            1.70,
        );

        let (atp_band_complexes, atp_band_assembly_rate, atp_band_degradation_rate) =
            Self::complex_channel_step(
                current.atp_band_complexes,
                target.atp_band_target,
                energy_support,
                degradation_pressure,
                dt_scale,
                512.0,
            );
        let (ribosome_complexes, ribosome_assembly_rate, ribosome_degradation_rate) =
            Self::complex_channel_step(
                current.ribosome_complexes,
                target.ribosome_target,
                translation_support,
                degradation_pressure,
                dt_scale,
                640.0,
            );
        let (rnap_complexes, rnap_assembly_rate, rnap_degradation_rate) =
            Self::complex_channel_step(
                current.rnap_complexes,
                target.rnap_target,
                transcription_support,
                degradation_pressure,
                dt_scale,
                384.0,
            );
        let (replisome_complexes, replisome_assembly_rate, replisome_degradation_rate) =
            Self::complex_channel_step(
                current.replisome_complexes,
                target.replisome_target,
                replication_support,
                degradation_pressure,
                dt_scale,
                192.0,
            );
        let (membrane_complexes, membrane_assembly_rate, membrane_degradation_rate) =
            Self::complex_channel_step(
                current.membrane_complexes,
                target.membrane_target,
                membrane_support,
                degradation_pressure,
                dt_scale,
                384.0,
            );
        let (ftsz_polymer, ftsz_assembly_rate, ftsz_degradation_rate) = Self::complex_channel_step(
            current.ftsz_polymer,
            target.ftsz_target,
            constriction_support,
            degradation_pressure,
            dt_scale,
            768.0,
        );
        let (dnaa_activity, dnaa_assembly_rate, dnaa_degradation_rate) = Self::complex_channel_step(
            current.dnaa_activity,
            target.dnaa_target,
            replication_support,
            degradation_pressure,
            dt_scale,
            256.0,
        );

        self.complex_assembly = WholeCellComplexAssemblyState {
            atp_band_complexes,
            ribosome_complexes,
            rnap_complexes,
            replisome_complexes,
            membrane_complexes,
            ftsz_polymer,
            dnaa_activity,
            atp_band_target: target.atp_band_target,
            ribosome_target: target.ribosome_target,
            rnap_target: target.rnap_target,
            replisome_target: target.replisome_target,
            membrane_target: target.membrane_target,
            ftsz_target: target.ftsz_target,
            dnaa_target: target.dnaa_target,
            atp_band_assembly_rate,
            ribosome_assembly_rate,
            rnap_assembly_rate,
            replisome_assembly_rate,
            membrane_assembly_rate,
            ftsz_assembly_rate,
            dnaa_assembly_rate,
            atp_band_degradation_rate,
            ribosome_degradation_rate,
            rnap_degradation_rate,
            replisome_degradation_rate,
            membrane_degradation_rate,
            ftsz_degradation_rate,
            dnaa_degradation_rate,
        };
        self.sync_runtime_process_species(dt);
    }
}
