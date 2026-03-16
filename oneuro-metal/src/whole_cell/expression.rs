//! Expression and runtime-process dynamics for the whole-cell simulator.
//!
//! Extracted methods: species synchronisation, reaction-rate updates,
//! organism expression-state refresh, inventory dynamics, and process
//! scale derivation.

use super::*;

impl WholeCellSimulator {
    pub(super) fn sync_runtime_process_species(&mut self, dt: f32) {
        if self.organism_species.is_empty() {
            if self.ensure_process_registry().is_some() {
                self.initialize_runtime_process_state();
            }
            return;
        }
        self.normalize_runtime_species_bulk_fields();

        let dt_scale = if dt > 0.0 {
            (dt / self.config.dt_ms.max(0.05)).clamp(0.25, 6.0)
        } else {
            0.0
        };
        let expression = self.organism_expression.clone();
        let (rna_totals, protein_totals) = if let Some(assets) = self.organism_assets.as_ref() {
            (
                assets.rnas.iter().fold(HashMap::new(), |mut acc, rna| {
                    *acc.entry(rna.operon.clone()).or_insert(0.0) += rna.basal_abundance.max(0.0);
                    acc
                }),
                assets
                    .proteins
                    .iter()
                    .fold(HashMap::new(), |mut acc, protein| {
                        *acc.entry(protein.operon.clone()).or_insert(0.0) +=
                            protein.basal_abundance.max(0.0);
                        acc
                    }),
            )
        } else {
            // Registry-backed or explicit runtime species can still define operon
            // totals even when the caller stripped bundle assets. Use the richer
            // runtime payload instead of dropping the entire chemistry layer.
            let rna_totals = self.organism_species.iter().fold(
                HashMap::new(),
                |mut acc: HashMap<String, f32>, species| {
                    if species.species_class == WholeCellSpeciesClass::Rna {
                        if let Some(operon) = species.operon.as_ref() {
                            *acc.entry(operon.clone()).or_insert(0.0) +=
                                species.basal_abundance.max(0.0);
                        }
                    }
                    acc
                },
            );
            let protein_totals = self.organism_species.iter().fold(
                HashMap::new(),
                |mut acc: HashMap<String, f32>, species| {
                    if species.species_class == WholeCellSpeciesClass::Protein {
                        if let Some(operon) = species.operon.as_ref() {
                            *acc.entry(operon.clone()).or_insert(0.0) +=
                                species.basal_abundance.max(0.0);
                        }
                    }
                    acc
                },
            );
            (rna_totals, protein_totals)
        };
        let unit_transcripts = expression
            .transcription_units
            .iter()
            .map(|unit| (unit.name.clone(), unit.transcript_abundance))
            .collect::<HashMap<_, _>>();
        let unit_proteins = expression
            .transcription_units
            .iter()
            .map(|unit| (unit.name.clone(), unit.protein_abundance))
            .collect::<HashMap<_, _>>();
        let named_complexes = self
            .named_complexes
            .iter()
            .map(|state| (state.id.clone(), state.clone()))
            .collect::<HashMap<_, _>>();
        let spatial_cache = self.compute_spatial_coupling_cache();

        for species in &mut self.organism_species {
            let anchor = match species.species_class {
                WholeCellSpeciesClass::Pool => Self::pool_species_anchor(
                    species.bulk_field,
                    species.spatial_scope,
                    species.patch_domain,
                    species.basal_abundance,
                    &spatial_cache,
                ),
                WholeCellSpeciesClass::Rna => {
                    if let Some(operon) = species.operon.as_ref() {
                        let total = rna_totals
                            .get(operon)
                            .copied()
                            .unwrap_or(species.basal_abundance.max(0.01))
                            .max(0.01);
                        let unit_total = unit_transcripts
                            .get(operon)
                            .copied()
                            .unwrap_or(species.basal_abundance.max(0.0));
                        (unit_total * species.basal_abundance.max(0.01) / total).clamp(0.0, 2048.0)
                    } else {
                        species.basal_abundance.max(0.0)
                    }
                }
                WholeCellSpeciesClass::Protein => {
                    if let Some(operon) = species.operon.as_ref() {
                        let total = protein_totals
                            .get(operon)
                            .copied()
                            .unwrap_or(species.basal_abundance.max(0.01))
                            .max(0.01);
                        let unit_total = unit_proteins
                            .get(operon)
                            .copied()
                            .unwrap_or(species.basal_abundance.max(0.0));
                        (unit_total * species.basal_abundance.max(0.01) / total).clamp(0.0, 4096.0)
                    } else {
                        species.basal_abundance.max(0.0)
                    }
                }
                WholeCellSpeciesClass::ComplexSubunitPool => species
                    .parent_complex
                    .as_ref()
                    .and_then(|id| named_complexes.get(id))
                    .map(|state| state.subunit_pool)
                    .unwrap_or(species.basal_abundance.max(0.0)),
                WholeCellSpeciesClass::ComplexNucleationIntermediate => species
                    .parent_complex
                    .as_ref()
                    .and_then(|id| named_complexes.get(id))
                    .map(|state| state.nucleation_intermediate)
                    .unwrap_or(species.basal_abundance.max(0.0)),
                WholeCellSpeciesClass::ComplexElongationIntermediate => species
                    .parent_complex
                    .as_ref()
                    .and_then(|id| named_complexes.get(id))
                    .map(|state| state.elongation_intermediate)
                    .unwrap_or(species.basal_abundance.max(0.0)),
                WholeCellSpeciesClass::ComplexMature => species
                    .parent_complex
                    .as_ref()
                    .and_then(|id| named_complexes.get(id))
                    .map(|state| state.abundance)
                    .unwrap_or(species.basal_abundance.max(0.0)),
            };
            let previous = species.count.max(0.0);
            let support = Self::asset_class_expression_support(&expression, species.asset_class);
            let next = if dt_scale <= 0.0 {
                anchor
            } else {
                let coupling = ((0.16 + 0.08 * support) * dt_scale).clamp(0.08, 0.95);
                previous + coupling * (anchor - previous)
            }
            .clamp(
                0.0,
                Self::runtime_species_upper_bound(species.species_class),
            );
            species.anchor_count = anchor;
            if dt_scale > 0.0 {
                let delta = next - previous;
                species.synthesis_rate = (delta.max(0.0) / dt_scale).max(0.0);
                species.turnover_rate = ((-delta).max(0.0) / dt_scale).max(0.0);
            } else {
                species.synthesis_rate = 0.0;
                species.turnover_rate = 0.0;
            }
            species.count = next;
        }
    }

    pub(super) fn update_runtime_process_reactions(
        &mut self,
        dt: f32,
        transcription_flux: f32,
        translation_flux: f32,
    ) {
        if self.organism_reactions.is_empty() {
            if self.ensure_process_registry().is_some() {
                self.initialize_runtime_process_state();
            }
            return;
        }
        let dt_scale = if dt > 0.0 {
            (dt / self.config.dt_ms.max(0.05)).clamp(0.25, 6.0)
        } else {
            0.0
        };
        let expression = self.organism_expression.clone();
        let effective_load = self.effective_metabolic_load();
        let species_counts = self
            .organism_species
            .iter()
            .map(|species| (species.id.clone(), species.count))
            .collect::<HashMap<_, _>>();
        let species_state = self
            .organism_species
            .iter()
            .map(|species| (species.id.clone(), species.clone()))
            .collect::<HashMap<_, _>>();
        let species_anchors = self
            .organism_species
            .iter()
            .map(|species| (species.id.clone(), species.anchor_count))
            .collect::<HashMap<_, _>>();
        let species_basal = self
            .organism_species
            .iter()
            .map(|species| (species.id.clone(), species.basal_abundance))
            .collect::<HashMap<_, _>>();
        let rna_operon_basal_totals = self.organism_species.iter().fold(
            HashMap::new(),
            |mut acc: HashMap<String, f32>, species| {
                if species.species_class == WholeCellSpeciesClass::Rna {
                    if let Some(operon) = species.operon.as_ref() {
                        *acc.entry(operon.clone()).or_insert(0.0) +=
                            species.basal_abundance.max(0.0);
                    }
                }
                acc
            },
        );
        let protein_operon_basal_totals = self.organism_species.iter().fold(
            HashMap::new(),
            |mut acc: HashMap<String, f32>, species| {
                if species.species_class == WholeCellSpeciesClass::Protein {
                    if let Some(operon) = species.operon.as_ref() {
                        *acc.entry(operon.clone()).or_insert(0.0) +=
                            species.basal_abundance.max(0.0);
                    }
                }
                acc
            },
        );
        let unit_state_by_operon = expression
            .transcription_units
            .iter()
            .map(|unit| (unit.name.clone(), (unit.support_level, unit.stress_penalty)))
            .collect::<HashMap<_, _>>();
        let spatial_cache = self.compute_spatial_coupling_cache();
        let mut species_deltas: HashMap<String, f32> = HashMap::new();
        let mut bulk_field_deltas: HashMap<
            (
                WholeCellBulkField,
                WholeCellSpatialScope,
                WholeCellPatchDomain,
                Option<String>,
            ),
            f32,
        > = HashMap::new();
        let mut stress_relief: HashMap<String, f32> = HashMap::new();
        let mut metabolic_load_relief = 0.0f32;

        for reaction_index in 0..self.organism_reactions.len() {
            let reaction = &self.organism_reactions[reaction_index];
            let reaction_scope = reaction.spatial_scope;
            let reaction_patch_domain = reaction.patch_domain;
            let reactant_satisfaction = if reaction.reactants.is_empty() {
                1.0
            } else {
                reaction
                    .reactants
                    .iter()
                    .map(|participant| {
                        let count = species_state
                            .get(&participant.species_id)
                            .map(|species| {
                                self.effective_runtime_species_count_for_locality(
                                    species,
                                    reaction_scope,
                                    reaction_patch_domain,
                                    reaction.chromosome_domain.as_deref(),
                                    &spatial_cache,
                                )
                            })
                            .or_else(|| species_counts.get(&participant.species_id).copied())
                            .unwrap_or(0.0);
                        Self::saturating_signal(
                            count,
                            1.0 + 2.0 * participant.stoichiometry.max(0.25),
                        )
                    })
                    .fold(1.0, f32::min)
            };
            let catalyst_support = reaction
                .catalyst
                .as_ref()
                .map(|species_id| {
                    let count = species_state
                        .get(species_id)
                        .map(|species| {
                            self.effective_runtime_species_count_for_locality(
                                species,
                                reaction_scope,
                                reaction_patch_domain,
                                reaction.chromosome_domain.as_deref(),
                                &spatial_cache,
                            )
                        })
                        .or_else(|| species_counts.get(species_id).copied())
                        .unwrap_or(0.0);
                    Self::saturating_signal(count, 4.0)
                })
                .unwrap_or(1.0);
            let asset_support =
                Self::asset_class_expression_support(&expression, reaction.asset_class);
            let external_hint = match reaction.reaction_class {
                WholeCellReactionClass::PoolTransport => {
                    let pool_deficit = reaction
                        .products
                        .iter()
                        .map(|participant| {
                            let count = species_counts
                                .get(&participant.species_id)
                                .copied()
                                .unwrap_or(0.0);
                            let target = species_anchors
                                .get(&participant.species_id)
                                .copied()
                                .or_else(|| species_basal.get(&participant.species_id).copied())
                                .unwrap_or(1.0)
                                .max(1.0);
                            ((target - count).max(0.0) / target).clamp(0.0, 1.0)
                        })
                        .fold(0.0, f32::max);
                    Self::finite_scale(
                        0.12 + 1.65 * pool_deficit + 0.14 * expression.energy_support,
                        1.0,
                        0.05,
                        2.5,
                    )
                }
                WholeCellReactionClass::LocalizedPoolTransfer => {
                    self.localized_pool_transfer_hint(reaction, &species_state, &spatial_cache)
                }
                WholeCellReactionClass::LocalizedPoolTurnover => {
                    self.localized_pool_turnover_hint(reaction, &species_state, &spatial_cache)
                }
                WholeCellReactionClass::MembranePatchTransfer => {
                    self.localized_pool_transfer_hint(reaction, &species_state, &spatial_cache)
                }
                WholeCellReactionClass::MembranePatchTurnover => {
                    self.localized_pool_turnover_hint(reaction, &species_state, &spatial_cache)
                }
                WholeCellReactionClass::Transcription => {
                    Self::finite_scale(1.0 + 0.12 * transcription_flux.max(0.0), 1.0, 0.75, 2.5)
                }
                WholeCellReactionClass::Translation => {
                    Self::finite_scale(1.0 + 0.12 * translation_flux.max(0.0), 1.0, 0.75, 2.5)
                }
                WholeCellReactionClass::RnaDegradation => Self::finite_scale(
                    0.72 + 0.24 * expression.crowding_penalty.max(0.0)
                        + 0.12 * (1.0 - expression.process_scales.transcription).max(0.0),
                    1.0,
                    0.40,
                    2.0,
                ),
                WholeCellReactionClass::ProteinDegradation => Self::finite_scale(
                    0.68 + 0.28 * expression.crowding_penalty.max(0.0)
                        + 0.14 * (1.0 - expression.translation_support).max(0.0),
                    1.0,
                    0.35,
                    2.2,
                ),
                WholeCellReactionClass::StressResponse => {
                    let (support_level, stress_penalty) = reaction
                        .operon
                        .as_ref()
                        .and_then(|operon| unit_state_by_operon.get(operon).copied())
                        .unwrap_or((1.0, 1.0));
                    let stress_signal = (stress_penalty - 1.0).max(0.0);
                    Self::finite_scale(
                        0.08 + 1.05 * stress_signal
                            + 0.18 * (1.0 - support_level).max(0.0)
                            + 0.12 * (effective_load - 1.0).max(0.0),
                        1.0,
                        0.0,
                        2.4,
                    )
                }
                WholeCellReactionClass::SubunitPoolFormation => {
                    Self::finite_scale(0.9 + 0.08 * translation_flux.max(0.0), 1.0, 0.70, 2.0)
                }
                WholeCellReactionClass::ComplexNucleation
                | WholeCellReactionClass::ComplexElongation
                | WholeCellReactionClass::ComplexMaturation => Self::finite_scale(
                    0.78 + 0.22 * expression.translation_support
                        + 0.12 * expression.membrane_support,
                    1.0,
                    0.70,
                    2.0,
                ),
                WholeCellReactionClass::ComplexRepair => {
                    let repair_deficit = reaction
                        .products
                        .iter()
                        .map(|participant| {
                            let count = species_counts
                                .get(&participant.species_id)
                                .copied()
                                .unwrap_or(0.0);
                            let target = species_anchors
                                .get(&participant.species_id)
                                .copied()
                                .or_else(|| species_basal.get(&participant.species_id).copied())
                                .unwrap_or(1.0)
                                .max(1.0);
                            ((target - count).max(0.0) / target).clamp(0.0, 1.0)
                        })
                        .fold(0.0, f32::max);
                    Self::finite_scale(
                        0.14 + 1.10 * repair_deficit
                            + 0.16 * expression.translation_support
                            + 0.08 * expression.membrane_support,
                        1.0,
                        0.05,
                        2.6,
                    )
                }
                WholeCellReactionClass::ComplexTurnover => Self::finite_scale(
                    0.72 + 0.30 * expression.crowding_penalty.max(0.0),
                    1.0,
                    0.60,
                    1.8,
                ),
            };
            let current_flux = (reaction.nominal_rate.max(0.0)
                * reactant_satisfaction
                * catalyst_support
                * asset_support
                * external_hint)
                .max(0.0);
            let reaction_class = reaction.reaction_class;
            let reaction_operon = reaction.operon.clone();
            let reaction_chromosome_domain = reaction.chromosome_domain.clone();
            let reaction_reactants = if dt_scale > 0.0 {
                reaction.reactants.clone()
            } else {
                Vec::new()
            };
            let reaction_products = if dt_scale > 0.0 {
                reaction.products.clone()
            } else {
                Vec::new()
            };
            let reaction = &mut self.organism_reactions[reaction_index];
            reaction.current_flux = current_flux;
            reaction.reactant_satisfaction = reactant_satisfaction;
            reaction.catalyst_support = catalyst_support;
            if dt_scale > 0.0 {
                reaction.cumulative_extent += current_flux * dt_scale;
                let extent = 0.05 * current_flux * dt_scale;
                if reaction_class == WholeCellReactionClass::StressResponse {
                    metabolic_load_relief += 0.08 * extent;
                    if let Some(operon) = reaction_operon.as_ref() {
                        *stress_relief.entry(operon.clone()).or_insert(0.0) += 0.18 * extent;
                    }
                }
                for participant in &reaction_reactants {
                    *species_deltas
                        .entry(participant.species_id.clone())
                        .or_insert(0.0) -= extent * participant.stoichiometry.max(0.0);
                    if let Some(species) = species_state.get(&participant.species_id) {
                        if let Some(field) = species.bulk_field {
                            *bulk_field_deltas
                                .entry((
                                    field,
                                    species.spatial_scope,
                                    species.patch_domain,
                                    species
                                        .chromosome_domain
                                        .clone()
                                        .or_else(|| reaction_chromosome_domain.clone()),
                                ))
                                .or_insert(0.0) -= extent * participant.stoichiometry.max(0.0);
                        }
                    }
                }
                for participant in &reaction_products {
                    *species_deltas
                        .entry(participant.species_id.clone())
                        .or_insert(0.0) += extent * participant.stoichiometry.max(0.0);
                    if let Some(species) = species_state.get(&participant.species_id) {
                        if let Some(field) = species.bulk_field {
                            *bulk_field_deltas
                                .entry((
                                    field,
                                    species.spatial_scope,
                                    species.patch_domain,
                                    species
                                        .chromosome_domain
                                        .clone()
                                        .or_else(|| reaction_chromosome_domain.clone()),
                                ))
                                .or_insert(0.0) += extent * participant.stoichiometry.max(0.0);
                        }
                    }
                }
            }
        }

        if dt_scale > 0.0 {
            let mut transcript_delta_sum: HashMap<String, f32> = HashMap::new();
            let mut transcript_delta_count: HashMap<String, f32> = HashMap::new();
            let mut protein_delta_sum: HashMap<String, f32> = HashMap::new();
            let mut protein_delta_count: HashMap<String, f32> = HashMap::new();
            let mut complex_subunit_deltas: HashMap<String, f32> = HashMap::new();
            let mut complex_nucleation_deltas: HashMap<String, f32> = HashMap::new();
            let mut complex_elongation_deltas: HashMap<String, f32> = HashMap::new();
            let mut complex_mature_deltas: HashMap<String, f32> = HashMap::new();
            for species in &mut self.organism_species {
                if let Some(delta) = species_deltas.get(&species.id).copied() {
                    species.count = (species.count + delta).clamp(
                        0.0,
                        Self::runtime_species_upper_bound(species.species_class),
                    );
                    if delta >= 0.0 {
                        species.synthesis_rate += delta / dt_scale;
                    } else {
                        species.turnover_rate += (-delta) / dt_scale;
                    }
                    if species.bulk_field.is_none() {
                        match species.species_class {
                            WholeCellSpeciesClass::Rna => {
                                if let Some(operon) = species.operon.as_ref() {
                                    let total = rna_operon_basal_totals
                                        .get(operon)
                                        .copied()
                                        .unwrap_or(species.basal_abundance.max(0.01))
                                        .max(0.01);
                                    let candidate =
                                        delta * total / species.basal_abundance.max(0.01);
                                    *transcript_delta_sum.entry(operon.clone()).or_insert(0.0) +=
                                        candidate;
                                    *transcript_delta_count.entry(operon.clone()).or_insert(0.0) +=
                                        1.0;
                                }
                            }
                            WholeCellSpeciesClass::Protein => {
                                if let Some(operon) = species.operon.as_ref() {
                                    let total = protein_operon_basal_totals
                                        .get(operon)
                                        .copied()
                                        .unwrap_or(species.basal_abundance.max(0.01))
                                        .max(0.01);
                                    let candidate =
                                        delta * total / species.basal_abundance.max(0.01);
                                    *protein_delta_sum.entry(operon.clone()).or_insert(0.0) +=
                                        candidate;
                                    *protein_delta_count.entry(operon.clone()).or_insert(0.0) +=
                                        1.0;
                                }
                            }
                            WholeCellSpeciesClass::ComplexSubunitPool => {
                                if let Some(parent) = species.parent_complex.as_ref() {
                                    *complex_subunit_deltas.entry(parent.clone()).or_insert(0.0) +=
                                        delta;
                                }
                            }
                            WholeCellSpeciesClass::ComplexNucleationIntermediate => {
                                if let Some(parent) = species.parent_complex.as_ref() {
                                    *complex_nucleation_deltas
                                        .entry(parent.clone())
                                        .or_insert(0.0) += delta;
                                }
                            }
                            WholeCellSpeciesClass::ComplexElongationIntermediate => {
                                if let Some(parent) = species.parent_complex.as_ref() {
                                    *complex_elongation_deltas
                                        .entry(parent.clone())
                                        .or_insert(0.0) += delta;
                                }
                            }
                            WholeCellSpeciesClass::ComplexMature => {
                                if let Some(parent) = species.parent_complex.as_ref() {
                                    *complex_mature_deltas.entry(parent.clone()).or_insert(0.0) +=
                                        delta;
                                }
                            }
                            WholeCellSpeciesClass::Pool => {}
                        }
                    }
                }
            }
            let mut lattice_bulk_changed = false;
            for ((field, scope, patch_domain, chromosome_domain), delta) in bulk_field_deltas {
                lattice_bulk_changed |= self.apply_bulk_field_delta(
                    field,
                    scope,
                    patch_domain,
                    chromosome_domain.as_deref(),
                    delta,
                );
            }
            let mut expression_changed = false;
            for (operon, sum) in transcript_delta_sum {
                let average = sum
                    / transcript_delta_count
                        .get(&operon)
                        .copied()
                        .unwrap_or(1.0)
                        .max(1.0);
                expression_changed |=
                    self.apply_operon_inventory_delta(&operon, average, 0.0, dt_scale);
            }
            for (operon, sum) in protein_delta_sum {
                let average = sum
                    / protein_delta_count
                        .get(&operon)
                        .copied()
                        .unwrap_or(1.0)
                        .max(1.0);
                expression_changed |=
                    self.apply_operon_inventory_delta(&operon, 0.0, average, dt_scale);
            }
            for (operon, relief) in stress_relief {
                if let Some(unit) = self
                    .organism_expression
                    .transcription_units
                    .iter_mut()
                    .find(|unit| unit.name == operon)
                {
                    unit.stress_penalty = (unit.stress_penalty - relief).clamp(0.80, 1.60);
                    unit.support_level = (unit.support_level + 0.10 * relief).clamp(0.55, 1.55);
                    expression_changed = true;
                }
            }
            if expression_changed {
                self.refresh_expression_inventory_totals();
            }
            let mut complex_changed = false;
            for (complex_id, delta) in complex_subunit_deltas {
                complex_changed |= self.apply_named_complex_species_delta(
                    &complex_id,
                    WholeCellSpeciesClass::ComplexSubunitPool,
                    delta,
                );
            }
            for (complex_id, delta) in complex_nucleation_deltas {
                complex_changed |= self.apply_named_complex_species_delta(
                    &complex_id,
                    WholeCellSpeciesClass::ComplexNucleationIntermediate,
                    delta,
                );
            }
            for (complex_id, delta) in complex_elongation_deltas {
                complex_changed |= self.apply_named_complex_species_delta(
                    &complex_id,
                    WholeCellSpeciesClass::ComplexElongationIntermediate,
                    delta,
                );
            }
            for (complex_id, delta) in complex_mature_deltas {
                complex_changed |= self.apply_named_complex_species_delta(
                    &complex_id,
                    WholeCellSpeciesClass::ComplexMature,
                    delta,
                );
            }
            if complex_changed {
                if let Some(assets) = self.organism_assets.clone() {
                    self.complex_assembly = self.aggregate_named_complex_assembly_state(&assets);
                }
            }
            if metabolic_load_relief > 0.0 {
                self.metabolic_load =
                    (self.metabolic_load - metabolic_load_relief).clamp(0.25, 8.0);
            }
            if lattice_bulk_changed {
                self.sync_from_lattice();
            }
        }
    }


    pub(super) fn refresh_organism_expression_state(&mut self) {
        let Some(organism) = self.organism_data.clone() else {
            if !self.has_explicit_expression_state() {
                self.organism_expression = self
                    .synthesize_expression_state_from_runtime_process()
                    .unwrap_or_default();
            }
            return;
        };
        let previous_units = self.organism_expression.transcription_units.clone();
        let profile = derive_organism_profile(&organism);
        let localized_supply = self.localized_supply_scale();
        let crowding_penalty =
            Self::finite_scale(self.chemistry_report.crowding_penalty, 1.0, 0.65, 1.10);
        let adenylate_ratio = self.atp_mm.max(0.0) / (self.adp_mm + 0.25).max(0.25);
        let energy_support = Self::finite_scale(
            0.44 * self.chemistry_report.atp_support
                + 0.24 * Self::saturating_signal(adenylate_ratio, 1.0)
                + 0.16 * Self::saturating_signal(self.glucose_mm, 0.8)
                + 0.16 * Self::saturating_signal(self.oxygen_mm, 0.7),
            1.0,
            0.55,
            1.55,
        );
        let translation_support = Self::finite_scale(
            0.72 * self.chemistry_report.translation_support + 0.28 * localized_supply,
            1.0,
            0.55,
            1.55,
        );
        let nucleotide_support = Self::finite_scale(
            0.72 * self.chemistry_report.nucleotide_support + 0.28 * localized_supply,
            1.0,
            0.55,
            1.55,
        );
        let chromosome_domain_count = self.chromosome_domain_count();
        let chromosome_domain_energy_support = (0..chromosome_domain_count)
            .map(|index| self.chromosome_domain_energy_support(index as u32))
            .collect::<Vec<_>>();
        let chromosome_domain_nucleotide_support = (0..chromosome_domain_count)
            .map(|index| self.chromosome_domain_nucleotide_support(index as u32))
            .collect::<Vec<_>>();
        let chromosome_domain_localized_supply = (0..chromosome_domain_count)
            .map(|index| self.chromosome_domain_localized_supply(index as u32))
            .collect::<Vec<_>>();
        let membrane_support = Self::finite_scale(
            0.72 * self.chemistry_report.membrane_support + 0.28 * localized_supply,
            1.0,
            0.55,
            1.55,
        );
        let load_penalty = (1.0 + 0.22 * (self.metabolic_load - 1.0).max(0.0)).clamp(1.0, 1.45);

        let mut process_signal = WholeCellProcessWeights::default();
        let mut transcription_units = Vec::new();
        let mut activity_total = 0.0;
        let mut amino_cost_signal = 0.0;
        let mut nucleotide_cost_signal = 0.0;
        let mut total_transcript_abundance = 0.0;
        let mut total_protein_abundance = 0.0;

        for unit in &organism.transcription_units {
            let mut unit_weights = unit.process_weights.clamped();
            let mut midpoint_sum = 0.0;
            let mut midpoint_count = 0usize;
            let mut mean_translation_cost = 1.0;
            let mut mean_nucleotide_cost = 1.0;
            let mut cost_count = 0.0;

            for gene_name in &unit.genes {
                if let Some(feature) = organism
                    .genes
                    .iter()
                    .find(|feature| feature.gene == *gene_name)
                {
                    let midpoint_bp = 0.5 * (feature.start_bp as f32 + feature.end_bp as f32);
                    midpoint_sum += midpoint_bp;
                    midpoint_count += 1;
                    unit_weights.add_weighted(
                        feature.process_weights,
                        0.35 * feature.basal_expression.max(0.1),
                    );
                    mean_translation_cost += feature.translation_cost.max(0.0);
                    mean_nucleotide_cost += feature.nucleotide_cost.max(0.0);
                    cost_count += 1.0;
                }
            }

            if cost_count > 0.0 {
                mean_translation_cost /= 1.0 + cost_count;
                mean_nucleotide_cost /= 1.0 + cost_count;
            }

            let midpoint_bp = if midpoint_count > 0 {
                (midpoint_sum / midpoint_count as f32).round() as u32
            } else {
                organism.origin_bp.min(organism.chromosome_length_bp.max(1))
            };
            let domain_index = self
                .chromosome_domain_index(midpoint_bp, organism.chromosome_length_bp.max(1))
                as usize;
            let copy_gain = self.chromosome_copy_number_at(midpoint_bp);
            let chromosome_accessibility = self.chromosome_locus_accessibility_at(midpoint_bp);
            let chromosome_torsion = self.chromosome_locus_torsional_stress_at(midpoint_bp);
            let (transcription_length_nt, translation_length_aa) =
                Self::expression_lengths_for_operon(&organism, &unit.genes);
            let previous_state = previous_units
                .iter()
                .find(|previous| previous.name == unit.name);
            let transcript_abundance = previous_state
                .map(|previous| previous.transcript_abundance)
                .unwrap_or_else(|| {
                    (8.0 + 5.0 * unit.basal_activity.max(0.05) * copy_gain)
                        * (unit.genes.len().max(1) as f32).sqrt()
                });
            let protein_abundance = previous_state
                .map(|previous| previous.protein_abundance)
                .unwrap_or_else(|| {
                    (14.0 + 9.0 * unit.basal_activity.max(0.05) * copy_gain)
                        * (unit.genes.len().max(1) as f32).sqrt()
                });
            let support_level = Self::unit_support_level(
                unit_weights,
                Self::chromosome_domain_value(&chromosome_domain_energy_support, domain_index),
                translation_support,
                Self::chromosome_domain_value(&chromosome_domain_nucleotide_support, domain_index),
                membrane_support,
                Self::chromosome_domain_value(&chromosome_domain_localized_supply, domain_index),
            );
            let stress_penalty = (load_penalty
                + 0.22 * (1.0 - crowding_penalty).max(0.0)
                + 0.18 * (1.0 - support_level).max(0.0))
            .clamp(0.80, 1.60);
            let effective_activity = (unit.basal_activity.max(0.05)
                * copy_gain
                * support_level
                * chromosome_accessibility
                / (stress_penalty * (1.0 + 0.18 * chromosome_torsion)))
                .clamp(0.05, 2.50);
            let inventory_scale = Self::unit_inventory_scale(
                transcript_abundance,
                protein_abundance,
                unit.genes.len(),
            );

            process_signal.add_weighted(unit_weights, effective_activity * inventory_scale);
            activity_total += effective_activity;
            amino_cost_signal += mean_translation_cost * effective_activity * inventory_scale;
            nucleotide_cost_signal += mean_nucleotide_cost * effective_activity * inventory_scale;
            total_transcript_abundance += transcript_abundance;
            total_protein_abundance += protein_abundance;
            transcription_units.push(WholeCellTranscriptionUnitState {
                name: unit.name.clone(),
                gene_count: unit.genes.len(),
                copy_gain,
                basal_activity: unit.basal_activity.max(0.0),
                effective_activity,
                support_level,
                stress_penalty,
                transcript_abundance,
                protein_abundance,
                transcript_synthesis_rate: previous_state
                    .map(|previous| previous.transcript_synthesis_rate)
                    .unwrap_or(0.0),
                protein_synthesis_rate: previous_state
                    .map(|previous| previous.protein_synthesis_rate)
                    .unwrap_or(0.0),
                transcript_turnover_rate: previous_state
                    .map(|previous| previous.transcript_turnover_rate)
                    .unwrap_or(0.0),
                protein_turnover_rate: previous_state
                    .map(|previous| previous.protein_turnover_rate)
                    .unwrap_or(0.0),
                promoter_open_fraction: previous_state
                    .map(|previous| previous.promoter_open_fraction)
                    .unwrap_or(0.0),
                active_rnap_occupancy: previous_state
                    .map(|previous| previous.active_rnap_occupancy)
                    .unwrap_or(0.0),
                transcription_length_nt: previous_state
                    .map(|previous| previous.transcription_length_nt.max(90.0))
                    .unwrap_or(transcription_length_nt),
                transcription_progress_nt: previous_state
                    .map(|previous| previous.transcription_progress_nt)
                    .unwrap_or(0.0),
                nascent_transcript_abundance: previous_state
                    .map(|previous| previous.nascent_transcript_abundance)
                    .unwrap_or(0.0),
                mature_transcript_abundance: previous_state
                    .map(|previous| previous.mature_transcript_abundance)
                    .unwrap_or(transcript_abundance),
                damaged_transcript_abundance: previous_state
                    .map(|previous| previous.damaged_transcript_abundance)
                    .unwrap_or(0.0),
                active_ribosome_occupancy: previous_state
                    .map(|previous| previous.active_ribosome_occupancy)
                    .unwrap_or(0.0),
                translation_length_aa: previous_state
                    .map(|previous| previous.translation_length_aa.max(30.0))
                    .unwrap_or(translation_length_aa),
                translation_progress_aa: previous_state
                    .map(|previous| previous.translation_progress_aa)
                    .unwrap_or(0.0),
                nascent_protein_abundance: previous_state
                    .map(|previous| previous.nascent_protein_abundance)
                    .unwrap_or(0.0),
                mature_protein_abundance: previous_state
                    .map(|previous| previous.mature_protein_abundance)
                    .unwrap_or(protein_abundance),
                damaged_protein_abundance: previous_state
                    .map(|previous| previous.damaged_protein_abundance)
                    .unwrap_or(0.0),
                process_drive: unit_weights.clamped(),
            });
        }

        if transcription_units.is_empty() && !organism.genes.is_empty() {
            for feature in &organism.genes {
                let midpoint_bp = 0.5 * (feature.start_bp as f32 + feature.end_bp as f32);
                let midpoint_bp = midpoint_bp.round() as u32;
                let domain_index = self
                    .chromosome_domain_index(midpoint_bp, organism.chromosome_length_bp.max(1))
                    as usize;
                let copy_gain = self.chromosome_copy_number_at(midpoint_bp);
                let chromosome_accessibility = self.chromosome_locus_accessibility_at(midpoint_bp);
                let chromosome_torsion = self.chromosome_locus_torsional_stress_at(midpoint_bp);
                let (transcription_length_nt, translation_length_aa) =
                    Self::expression_lengths_for_gene(feature);
                let previous_state = previous_units
                    .iter()
                    .find(|previous| previous.name == feature.gene);
                let transcript_abundance = previous_state
                    .map(|previous| previous.transcript_abundance)
                    .unwrap_or_else(|| 6.0 + 4.0 * feature.basal_expression.max(0.05) * copy_gain);
                let protein_abundance = previous_state
                    .map(|previous| previous.protein_abundance)
                    .unwrap_or_else(|| 10.0 + 8.0 * feature.basal_expression.max(0.05) * copy_gain);
                let support_level = Self::unit_support_level(
                    feature.process_weights,
                    Self::chromosome_domain_value(&chromosome_domain_energy_support, domain_index),
                    translation_support,
                    Self::chromosome_domain_value(
                        &chromosome_domain_nucleotide_support,
                        domain_index,
                    ),
                    membrane_support,
                    Self::chromosome_domain_value(
                        &chromosome_domain_localized_supply,
                        domain_index,
                    ),
                );
                let stress_penalty = (load_penalty
                    + 0.22 * (1.0 - crowding_penalty).max(0.0)
                    + 0.18 * (1.0 - support_level).max(0.0))
                .clamp(0.80, 1.60);
                let effective_activity =
                    (feature.basal_expression.max(0.05) * copy_gain * chromosome_accessibility
                        / (stress_penalty * (1.0 + 0.18 * chromosome_torsion)))
                        .clamp(0.05, 2.50);
                let inventory_scale =
                    Self::unit_inventory_scale(transcript_abundance, protein_abundance, 1);
                process_signal.add_weighted(
                    feature.process_weights,
                    effective_activity * inventory_scale,
                );
                activity_total += effective_activity;
                amino_cost_signal +=
                    feature.translation_cost.max(0.0) * effective_activity * inventory_scale;
                nucleotide_cost_signal +=
                    feature.nucleotide_cost.max(0.0) * effective_activity * inventory_scale;
                total_transcript_abundance += transcript_abundance;
                total_protein_abundance += protein_abundance;
                transcription_units.push(WholeCellTranscriptionUnitState {
                    name: feature.gene.clone(),
                    gene_count: 1,
                    copy_gain,
                    basal_activity: feature.basal_expression.max(0.0),
                    effective_activity,
                    support_level,
                    stress_penalty,
                    transcript_abundance,
                    protein_abundance,
                    transcript_synthesis_rate: previous_state
                        .map(|previous| previous.transcript_synthesis_rate)
                        .unwrap_or(0.0),
                    protein_synthesis_rate: previous_state
                        .map(|previous| previous.protein_synthesis_rate)
                        .unwrap_or(0.0),
                    transcript_turnover_rate: previous_state
                        .map(|previous| previous.transcript_turnover_rate)
                        .unwrap_or(0.0),
                    protein_turnover_rate: previous_state
                        .map(|previous| previous.protein_turnover_rate)
                        .unwrap_or(0.0),
                    promoter_open_fraction: previous_state
                        .map(|previous| previous.promoter_open_fraction)
                        .unwrap_or(0.0),
                    active_rnap_occupancy: previous_state
                        .map(|previous| previous.active_rnap_occupancy)
                        .unwrap_or(0.0),
                    transcription_length_nt: previous_state
                        .map(|previous| previous.transcription_length_nt.max(90.0))
                        .unwrap_or(transcription_length_nt),
                    transcription_progress_nt: previous_state
                        .map(|previous| previous.transcription_progress_nt)
                        .unwrap_or(0.0),
                    nascent_transcript_abundance: previous_state
                        .map(|previous| previous.nascent_transcript_abundance)
                        .unwrap_or(0.0),
                    mature_transcript_abundance: previous_state
                        .map(|previous| previous.mature_transcript_abundance)
                        .unwrap_or(transcript_abundance),
                    damaged_transcript_abundance: previous_state
                        .map(|previous| previous.damaged_transcript_abundance)
                        .unwrap_or(0.0),
                    active_ribosome_occupancy: previous_state
                        .map(|previous| previous.active_ribosome_occupancy)
                        .unwrap_or(0.0),
                    translation_length_aa: previous_state
                        .map(|previous| previous.translation_length_aa.max(30.0))
                        .unwrap_or(translation_length_aa),
                    translation_progress_aa: previous_state
                        .map(|previous| previous.translation_progress_aa)
                        .unwrap_or(0.0),
                    nascent_protein_abundance: previous_state
                        .map(|previous| previous.nascent_protein_abundance)
                        .unwrap_or(0.0),
                    mature_protein_abundance: previous_state
                        .map(|previous| previous.mature_protein_abundance)
                        .unwrap_or(protein_abundance),
                    damaged_protein_abundance: previous_state
                        .map(|previous| previous.damaged_protein_abundance)
                        .unwrap_or(0.0),
                    process_drive: feature.process_weights.clamped(),
                });
            }
        }

        let registry_drive = self.registry_process_drive();
        process_signal.add_weighted(registry_drive, 0.14);
        activity_total += 0.02 * registry_drive.total().max(0.0);
        amino_cost_signal +=
            0.06 * (registry_drive.translation + 0.45 * registry_drive.membrane).max(0.0);
        nucleotide_cost_signal +=
            0.05 * (registry_drive.transcription + 0.70 * registry_drive.replication).max(0.0);

        let process_mean = (process_signal.energy
            + process_signal.transcription
            + process_signal.translation
            + process_signal.replication
            + process_signal.segregation
            + process_signal.membrane
            + process_signal.constriction)
            / 7.0;
        let cost_mean = if activity_total > 1.0e-6 {
            (amino_cost_signal + nucleotide_cost_signal) / (2.0 * activity_total.max(1.0e-6))
        } else {
            1.0
        };
        let global_activity = if transcription_units.is_empty() {
            1.0
        } else {
            activity_total / transcription_units.len() as f32
        };
        let process_scales = WholeCellProcessWeights {
            energy: Self::finite_scale(
                profile.process_scales.energy
                    * Self::process_scale(process_signal.energy, process_mean),
                1.0,
                0.70,
                1.45,
            ),
            transcription: Self::finite_scale(
                profile.process_scales.transcription
                    * Self::process_scale(process_signal.transcription, process_mean),
                1.0,
                0.70,
                1.45,
            ),
            translation: Self::finite_scale(
                profile.process_scales.translation
                    * Self::process_scale(process_signal.translation, process_mean),
                1.0,
                0.70,
                1.45,
            ),
            replication: Self::finite_scale(
                profile.process_scales.replication
                    * Self::process_scale(process_signal.replication, process_mean),
                1.0,
                0.70,
                1.45,
            ),
            segregation: Self::finite_scale(
                profile.process_scales.segregation
                    * Self::process_scale(process_signal.segregation, process_mean),
                1.0,
                0.70,
                1.45,
            ),
            membrane: Self::finite_scale(
                profile.process_scales.membrane
                    * Self::process_scale(process_signal.membrane, process_mean),
                1.0,
                0.70,
                1.45,
            ),
            constriction: Self::finite_scale(
                profile.process_scales.constriction
                    * Self::process_scale(process_signal.constriction, process_mean),
                1.0,
                0.70,
                1.45,
            ),
        };

        self.organism_expression = WholeCellOrganismExpressionState {
            global_activity: Self::finite_scale(global_activity, 1.0, 0.50, 1.80),
            energy_support,
            translation_support,
            nucleotide_support,
            membrane_support,
            crowding_penalty,
            metabolic_burden_scale: Self::finite_scale(
                profile.metabolic_burden_scale * (0.92 + 0.08 * global_activity),
                profile.metabolic_burden_scale,
                0.85,
                1.65,
            ),
            process_scales,
            amino_cost_scale: Self::process_scale(amino_cost_signal, cost_mean),
            nucleotide_cost_scale: Self::process_scale(nucleotide_cost_signal, cost_mean),
            total_transcript_abundance,
            total_protein_abundance,
            transcription_units,
        };
    }

    pub(super) fn update_organism_inventory_dynamics(
        &mut self,
        dt: f32,
        transcription_flux: f32,
        translation_flux: f32,
    ) {
        if !self.has_explicit_expression_state() {
            return;
        }
        let dt_scale = (dt / self.config.dt_ms.max(0.05)).clamp(0.5, 4.0);
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        let transcription_flux = transcription_flux.max(0.0);
        let translation_flux = translation_flux.max(0.0);
        let engaged_rnap_capacity = self.complex_assembly.rnap_complexes.clamp(8.0, 256.0)
            * (0.08 + 0.04 * transcription_flux.clamp(0.0, 2.5));
        let engaged_ribosome_capacity = self.complex_assembly.ribosome_complexes.clamp(12.0, 320.0)
            * (0.10 + 0.05 * translation_flux.clamp(0.0, 2.5));

        let mut transcription_demands =
            Vec::with_capacity(self.organism_expression.transcription_units.len());
        let mut translation_demands =
            Vec::with_capacity(self.organism_expression.transcription_units.len());

        for unit in &mut self.organism_expression.transcription_units {
            Self::normalize_expression_execution_state(unit);
            let gene_scale = (unit.gene_count.max(1) as f32).sqrt();
            let support = unit.support_level.clamp(0.55, 1.55);
            let stress = unit.stress_penalty.clamp(0.80, 1.60);
            unit.promoter_open_fraction = Self::finite_scale(
                0.34 + 0.30 * support + 0.16 * unit.copy_gain.clamp(0.8, 1.5) + 0.12 * crowding
                    - 0.18 * (stress - 1.0).max(0.0),
                0.5,
                0.05,
                1.0,
            );
            let transcription_demand = unit.promoter_open_fraction
                * unit.effective_activity
                * (0.55 + 0.25 * gene_scale)
                * (0.75 + 0.25 * crowding)
                * (0.85 + 0.25 * transcription_flux);
            let transcript_signal = Self::saturating_signal(
                unit.mature_transcript_abundance + 0.5 * unit.nascent_transcript_abundance,
                8.0 + 2.0 * gene_scale,
            );
            let translation_demand = transcript_signal
                * unit.effective_activity
                * support
                * (0.45 + 0.20 * gene_scale)
                * (0.80 + 0.20 * translation_flux);
            transcription_demands.push(transcription_demand.max(0.0));
            translation_demands.push(translation_demand.max(0.0));
        }

        let total_transcription_demand = transcription_demands
            .iter()
            .copied()
            .sum::<f32>()
            .max(1.0e-6);
        let total_translation_demand = translation_demands.iter().copied().sum::<f32>().max(1.0e-6);

        for ((unit, transcription_demand), translation_demand) in self
            .organism_expression
            .transcription_units
            .iter_mut()
            .zip(transcription_demands.into_iter())
            .zip(translation_demands.into_iter())
        {
            let gene_scale = (unit.gene_count.max(1) as f32).sqrt();
            let support = unit.support_level.clamp(0.55, 1.55);
            let stress = unit.stress_penalty.clamp(0.80, 1.60);

            let rnap_target = engaged_rnap_capacity
                * (transcription_demand / total_transcription_demand)
                * (0.60 + 0.40 * unit.promoter_open_fraction);
            let ribosome_target = engaged_ribosome_capacity
                * (translation_demand / total_translation_demand)
                * (0.55 + 0.45 * Self::saturating_signal(unit.mature_transcript_abundance, 6.0));
            let rnap_ceiling = (3.0 + 2.5 * gene_scale * unit.copy_gain.clamp(0.8, 1.8)).max(1.0);
            let ribosome_ceiling =
                (4.0 + 3.5 * gene_scale * unit.copy_gain.clamp(0.8, 1.8)).max(1.0);
            unit.active_rnap_occupancy =
                (0.58 * unit.active_rnap_occupancy + 0.42 * rnap_target).clamp(0.0, rnap_ceiling);
            unit.active_ribosome_occupancy = (0.56 * unit.active_ribosome_occupancy
                + 0.44 * ribosome_target)
                .clamp(0.0, ribosome_ceiling);

            let transcription_elongation_rate = (18.0 + 14.0 * support + 5.0 * transcription_flux
                - 3.0 * (stress - 1.0).max(0.0))
            .max(6.0);
            unit.transcription_progress_nt +=
                unit.active_rnap_occupancy * transcription_elongation_rate * dt_scale * crowding;
            let completed_transcripts =
                (unit.transcription_progress_nt / unit.transcription_length_nt.max(1.0)).max(0.0);
            unit.transcription_progress_nt %= unit.transcription_length_nt.max(1.0);

            let nascent_transcript_target =
                unit.active_rnap_occupancy * (0.35 + 0.10 * support + 0.08 * gene_scale);
            unit.nascent_transcript_abundance = (0.62 * unit.nascent_transcript_abundance
                + 0.38 * nascent_transcript_target)
                .clamp(0.0, 512.0);
            let transcript_damage = unit.mature_transcript_abundance
                * (0.0015 + 0.008 * (stress - 1.0).max(0.0) + 0.003 * (1.0 - support).max(0.0))
                * dt_scale;
            let transcript_turnover = unit.mature_transcript_abundance
                * (0.008 + 0.010 * (stress - 1.0).max(0.0) + 0.004 * (1.0 - support).max(0.0))
                * dt_scale;
            let transcript_damage_clearance =
                unit.damaged_transcript_abundance * (0.012 + 0.010 * support) * dt_scale;
            let matured_transcripts = completed_transcripts
                * (0.68 + 0.22 * support + 0.10 * crowding)
                * transcription_flux.max(0.25);
            unit.mature_transcript_abundance = (unit.mature_transcript_abundance
                + matured_transcripts
                - transcript_turnover
                - transcript_damage)
                .clamp(0.0, 1024.0);
            unit.damaged_transcript_abundance = (unit.damaged_transcript_abundance
                + transcript_damage
                - transcript_damage_clearance)
                .clamp(0.0, 512.0);

            let translation_elongation_rate = (9.0 + 8.0 * support + 4.0 * translation_flux
                - 2.0 * (stress - 1.0).max(0.0))
            .max(3.0);
            unit.translation_progress_aa +=
                unit.active_ribosome_occupancy * translation_elongation_rate * dt_scale * crowding;
            let completed_proteins =
                (unit.translation_progress_aa / unit.translation_length_aa.max(1.0)).max(0.0);
            unit.translation_progress_aa %= unit.translation_length_aa.max(1.0);

            let nascent_protein_target =
                unit.active_ribosome_occupancy * (0.28 + 0.08 * support + 0.06 * gene_scale);
            unit.nascent_protein_abundance = (0.64 * unit.nascent_protein_abundance
                + 0.36 * nascent_protein_target)
                .clamp(0.0, 768.0);
            let protein_damage = unit.mature_protein_abundance
                * (0.0008 + 0.005 * (stress - 1.0).max(0.0) + 0.002 * (1.0 - support).max(0.0))
                * dt_scale;
            let protein_turnover = unit.mature_protein_abundance
                * (0.003 + 0.005 * (stress - 1.0).max(0.0) + 0.002 * (1.0 - support).max(0.0))
                * dt_scale;
            let protein_damage_clearance =
                unit.damaged_protein_abundance * (0.007 + 0.008 * support) * dt_scale;
            let matured_proteins = completed_proteins
                * (0.70 + 0.20 * support + 0.10 * crowding)
                * translation_flux.max(0.25);
            unit.mature_protein_abundance = (unit.mature_protein_abundance + matured_proteins
                - protein_turnover
                - protein_damage)
                .clamp(0.0, 2048.0);
            unit.damaged_protein_abundance = (unit.damaged_protein_abundance + protein_damage
                - protein_damage_clearance)
                .clamp(0.0, 1024.0);

            unit.transcript_synthesis_rate = (matured_transcripts / dt_scale.max(1.0e-6)).max(0.0);
            unit.protein_synthesis_rate = (matured_proteins / dt_scale.max(1.0e-6)).max(0.0);
            unit.transcript_turnover_rate = ((transcript_turnover + transcript_damage_clearance)
                / dt_scale.max(1.0e-6))
            .max(0.0);
            unit.protein_turnover_rate =
                ((protein_turnover + protein_damage_clearance) / dt_scale.max(1.0e-6)).max(0.0);
            Self::normalize_expression_execution_state(unit);
        }

        self.refresh_expression_inventory_totals();
    }

    pub(super) fn organism_process_scales(&self) -> WholeCellOrganismProcessScales {
        let expression = &self.organism_expression;
        if !self.has_explicit_expression_state() {
            return WholeCellOrganismProcessScales::default();
        }
        WholeCellOrganismProcessScales {
            energy_scale: expression.process_scales.energy,
            transcription_scale: expression.process_scales.transcription,
            translation_scale: expression.process_scales.translation,
            replication_scale: expression.process_scales.replication,
            segregation_scale: expression.process_scales.segregation,
            membrane_scale: expression.process_scales.membrane,
            constriction_scale: expression.process_scales.constriction,
            amino_cost_scale: expression.amino_cost_scale,
            nucleotide_cost_scale: expression.nucleotide_cost_scale,
        }
    }
}
