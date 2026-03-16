//! RDME drive fields, spatial coupling, and bulk-field concentration helpers.
//!
//! These methods on `WholeCellSimulator` manage the RDME spatial drive field
//! computation, bulk-field concentration lookups, species-count adjustments,
//! and the per-domain coupling cache that feeds the stochastic lattice solver.

use super::*;

impl WholeCellSimulator {
    pub(super) fn species_runtime_count(&self, species_id: &str) -> Option<f32> {
        self.organism_species
            .iter()
            .find(|species| species.id == species_id)
            .map(|species| species.count)
    }

    pub(super) fn asset_class_expression_support(
        expression: &WholeCellOrganismExpressionState,
        asset_class: crate::whole_cell_data::WholeCellAssetClass,
    ) -> f32 {
        match asset_class {
            crate::whole_cell_data::WholeCellAssetClass::Energy => expression.process_scales.energy,
            crate::whole_cell_data::WholeCellAssetClass::Translation => {
                expression.process_scales.translation
            }
            crate::whole_cell_data::WholeCellAssetClass::Replication => {
                expression.process_scales.replication
            }
            crate::whole_cell_data::WholeCellAssetClass::Segregation => {
                expression.process_scales.segregation
            }
            crate::whole_cell_data::WholeCellAssetClass::Membrane => {
                expression.process_scales.membrane
            }
            crate::whole_cell_data::WholeCellAssetClass::Constriction => {
                expression.process_scales.constriction
            }
            crate::whole_cell_data::WholeCellAssetClass::QualityControl => Self::finite_scale(
                0.55 * expression.process_scales.translation
                    + 0.45 * expression.process_scales.energy,
                1.0,
                0.70,
                1.55,
            ),
            crate::whole_cell_data::WholeCellAssetClass::Homeostasis => Self::finite_scale(
                0.55 * expression.process_scales.transcription
                    + 0.45 * expression.process_scales.membrane,
                1.0,
                0.70,
                1.55,
            ),
            crate::whole_cell_data::WholeCellAssetClass::Generic => Self::finite_scale(
                0.18 * expression.process_scales.energy
                    + 0.16 * expression.process_scales.transcription
                    + 0.16 * expression.process_scales.translation
                    + 0.16 * expression.process_scales.replication
                    + 0.12 * expression.process_scales.segregation
                    + 0.12 * expression.process_scales.membrane
                    + 0.10 * expression.process_scales.constriction,
                1.0,
                0.70,
                1.45,
            ),
        }
    }

    pub(super) fn runtime_species_upper_bound(species_class: WholeCellSpeciesClass) -> f32 {
        match species_class {
            WholeCellSpeciesClass::Pool => 4096.0,
            WholeCellSpeciesClass::Rna => 2048.0,
            WholeCellSpeciesClass::Protein => 4096.0,
            WholeCellSpeciesClass::ComplexSubunitPool => 2048.0,
            WholeCellSpeciesClass::ComplexNucleationIntermediate => 1024.0,
            WholeCellSpeciesClass::ComplexElongationIntermediate => 1024.0,
            WholeCellSpeciesClass::ComplexMature => 1024.0,
        }
    }

    fn spatial_scope_field(scope: WholeCellSpatialScope) -> Option<IntracellularSpatialField> {
        match scope {
            WholeCellSpatialScope::WellMixed => None,
            WholeCellSpatialScope::MembraneAdjacent => {
                Some(IntracellularSpatialField::MembraneAdjacency)
            }
            WholeCellSpatialScope::SeptumLocal => Some(IntracellularSpatialField::SeptumZone),
            WholeCellSpatialScope::NucleoidLocal => {
                Some(IntracellularSpatialField::NucleoidOccupancy)
            }
        }
    }

    fn patch_domain_field(domain: WholeCellPatchDomain) -> Option<IntracellularSpatialField> {
        match domain {
            WholeCellPatchDomain::Distributed => None,
            WholeCellPatchDomain::MembraneBand => Some(IntracellularSpatialField::MembraneBandZone),
            WholeCellPatchDomain::SeptumPatch => Some(IntracellularSpatialField::SeptumZone),
            WholeCellPatchDomain::PolarPatch => Some(IntracellularSpatialField::PoleZone),
            WholeCellPatchDomain::NucleoidTrack => {
                Some(IntracellularSpatialField::NucleoidOccupancy)
            }
        }
    }

    fn spatial_scope_index(scope: WholeCellSpatialScope) -> usize {
        match scope {
            WholeCellSpatialScope::WellMixed => 0,
            WholeCellSpatialScope::MembraneAdjacent => 1,
            WholeCellSpatialScope::SeptumLocal => 2,
            WholeCellSpatialScope::NucleoidLocal => 3,
        }
    }

    fn patch_domain_index(domain: WholeCellPatchDomain) -> usize {
        match domain {
            WholeCellPatchDomain::Distributed => 0,
            WholeCellPatchDomain::MembraneBand => 1,
            WholeCellPatchDomain::SeptumPatch => 2,
            WholeCellPatchDomain::PolarPatch => 3,
            WholeCellPatchDomain::NucleoidTrack => 4,
        }
    }

    fn lattice_species_for_bulk_field(field: WholeCellBulkField) -> Option<IntracellularSpecies> {
        match field {
            WholeCellBulkField::ATP => Some(IntracellularSpecies::ATP),
            WholeCellBulkField::AminoAcids => Some(IntracellularSpecies::AminoAcids),
            WholeCellBulkField::Nucleotides => Some(IntracellularSpecies::Nucleotides),
            WholeCellBulkField::MembranePrecursors => {
                Some(IntracellularSpecies::MembranePrecursors)
            }
            WholeCellBulkField::ADP | WholeCellBulkField::Glucose | WholeCellBulkField::Oxygen => {
                None
            }
        }
    }

    fn bulk_field_species_scale(field: WholeCellBulkField) -> f32 {
        match field {
            WholeCellBulkField::ATP => 56.0,
            WholeCellBulkField::ADP => 48.0,
            WholeCellBulkField::Glucose => 40.0,
            WholeCellBulkField::Oxygen => 36.0,
            WholeCellBulkField::AminoAcids => 64.0,
            WholeCellBulkField::Nucleotides => 64.0,
            WholeCellBulkField::MembranePrecursors => 52.0,
        }
    }

    fn bulk_field_index(field: WholeCellBulkField) -> usize {
        match field {
            WholeCellBulkField::ATP => 0,
            WholeCellBulkField::ADP => 1,
            WholeCellBulkField::Glucose => 2,
            WholeCellBulkField::Oxygen => 3,
            WholeCellBulkField::AminoAcids => 4,
            WholeCellBulkField::Nucleotides => 5,
            WholeCellBulkField::MembranePrecursors => 6,
        }
    }

    #[allow(dead_code)]
    fn bulk_field_concentration(
        field: WholeCellBulkField,
        atp_mm: f32,
        adp_mm: f32,
        glucose_mm: f32,
        oxygen_mm: f32,
        amino_acids_mm: f32,
        nucleotides_mm: f32,
        membrane_precursors_mm: f32,
    ) -> f32 {
        match field {
            WholeCellBulkField::ATP => atp_mm,
            WholeCellBulkField::ADP => adp_mm,
            WholeCellBulkField::Glucose => glucose_mm,
            WholeCellBulkField::Oxygen => oxygen_mm,
            WholeCellBulkField::AminoAcids => amino_acids_mm,
            WholeCellBulkField::Nucleotides => nucleotides_mm,
            WholeCellBulkField::MembranePrecursors => membrane_precursors_mm,
        }
    }

    pub(super) fn compute_spatial_coupling_cache(&self) -> WholeCellSpatialCouplingCache {
        let mut bulk_concentrations = [[0.0; 7]; 4];
        let mut patch_bulk_concentrations = [[0.0; 7]; 5];
        let global_fields = [
            self.atp_mm,
            self.adp_mm,
            self.glucose_mm,
            self.oxygen_mm,
            self.amino_acids_mm,
            self.nucleotides_mm,
            self.membrane_precursors_mm,
        ];
        bulk_concentrations[Self::spatial_scope_index(WholeCellSpatialScope::WellMixed)] =
            global_fields;
        patch_bulk_concentrations[Self::patch_domain_index(WholeCellPatchDomain::Distributed)] =
            global_fields;
        for scope in [
            WholeCellSpatialScope::MembraneAdjacent,
            WholeCellSpatialScope::SeptumLocal,
            WholeCellSpatialScope::NucleoidLocal,
        ] {
            let scope_index = Self::spatial_scope_index(scope);
            bulk_concentrations[scope_index] = global_fields;
            let Some(field) = Self::spatial_scope_field(scope) else {
                continue;
            };
            for bulk_field in [
                WholeCellBulkField::ATP,
                WholeCellBulkField::AminoAcids,
                WholeCellBulkField::Nucleotides,
                WholeCellBulkField::MembranePrecursors,
            ] {
                if let Some(lattice_species) = Self::lattice_species_for_bulk_field(bulk_field) {
                    bulk_concentrations[scope_index][Self::bulk_field_index(bulk_field)] =
                        self.spatial_species_mean(lattice_species, field);
                }
            }
        }
        for domain in [
            WholeCellPatchDomain::MembraneBand,
            WholeCellPatchDomain::SeptumPatch,
            WholeCellPatchDomain::PolarPatch,
            WholeCellPatchDomain::NucleoidTrack,
        ] {
            let domain_index = Self::patch_domain_index(domain);
            patch_bulk_concentrations[domain_index] = global_fields;
            let Some(field) = Self::patch_domain_field(domain) else {
                continue;
            };
            for bulk_field in [
                WholeCellBulkField::ATP,
                WholeCellBulkField::AminoAcids,
                WholeCellBulkField::Nucleotides,
                WholeCellBulkField::MembranePrecursors,
            ] {
                if let Some(lattice_species) = Self::lattice_species_for_bulk_field(bulk_field) {
                    patch_bulk_concentrations[domain_index][Self::bulk_field_index(bulk_field)] =
                        self.spatial_species_mean(lattice_species, field);
                }
            }
        }

        let mut overlap = [[1.0; 4]; 4];
        for source_scope in [
            WholeCellSpatialScope::WellMixed,
            WholeCellSpatialScope::MembraneAdjacent,
            WholeCellSpatialScope::SeptumLocal,
            WholeCellSpatialScope::NucleoidLocal,
        ] {
            for target_scope in [
                WholeCellSpatialScope::WellMixed,
                WholeCellSpatialScope::MembraneAdjacent,
                WholeCellSpatialScope::SeptumLocal,
                WholeCellSpatialScope::NucleoidLocal,
            ] {
                let source_index = Self::spatial_scope_index(source_scope);
                let target_index = Self::spatial_scope_index(target_scope);
                if source_scope == WholeCellSpatialScope::WellMixed
                    || target_scope == WholeCellSpatialScope::WellMixed
                    || source_scope == target_scope
                {
                    overlap[source_index][target_index] = 1.0;
                    continue;
                }
                let Some(source_field) = Self::spatial_scope_field(source_scope) else {
                    continue;
                };
                let Some(target_field) = Self::spatial_scope_field(target_scope) else {
                    continue;
                };
                let source = self.spatial_fields.field_slice(source_field);
                let target = self.spatial_fields.field_slice(target_field);
                let source_total = source.iter().sum::<f32>().max(1.0e-6);
                let shared = source
                    .iter()
                    .zip(target.iter())
                    .map(|(lhs, rhs)| lhs.min(*rhs))
                    .sum::<f32>();
                overlap[source_index][target_index] = (shared / source_total).clamp(0.02, 1.0);
            }
        }
        let mut patch_overlap = [[1.0; 5]; 5];
        for source_domain in [
            WholeCellPatchDomain::Distributed,
            WholeCellPatchDomain::MembraneBand,
            WholeCellPatchDomain::SeptumPatch,
            WholeCellPatchDomain::PolarPatch,
            WholeCellPatchDomain::NucleoidTrack,
        ] {
            for target_domain in [
                WholeCellPatchDomain::Distributed,
                WholeCellPatchDomain::MembraneBand,
                WholeCellPatchDomain::SeptumPatch,
                WholeCellPatchDomain::PolarPatch,
                WholeCellPatchDomain::NucleoidTrack,
            ] {
                let source_index = Self::patch_domain_index(source_domain);
                let target_index = Self::patch_domain_index(target_domain);
                if source_domain == WholeCellPatchDomain::Distributed
                    || target_domain == WholeCellPatchDomain::Distributed
                    || source_domain == target_domain
                {
                    patch_overlap[source_index][target_index] = 1.0;
                    continue;
                }
                let Some(source_field) = Self::patch_domain_field(source_domain) else {
                    continue;
                };
                let Some(target_field) = Self::patch_domain_field(target_domain) else {
                    continue;
                };
                let source = self.spatial_fields.field_slice(source_field);
                let target = self.spatial_fields.field_slice(target_field);
                let source_total = source.iter().sum::<f32>().max(1.0e-6);
                let shared = source
                    .iter()
                    .zip(target.iter())
                    .map(|(lhs, rhs)| lhs.min(*rhs))
                    .sum::<f32>();
                patch_overlap[source_index][target_index] =
                    (shared / source_total).clamp(0.02, 1.0);
            }
        }

        WholeCellSpatialCouplingCache {
            bulk_concentrations,
            overlap,
            patch_bulk_concentrations,
            patch_overlap,
        }
    }

    fn bulk_field_concentration_for_scope(
        cache: &WholeCellSpatialCouplingCache,
        field: WholeCellBulkField,
        spatial_scope: WholeCellSpatialScope,
    ) -> f32 {
        cache.bulk_concentrations[Self::spatial_scope_index(spatial_scope)]
            [Self::bulk_field_index(field)]
    }

    fn bulk_field_concentration_for_patch_domain(
        cache: &WholeCellSpatialCouplingCache,
        field: WholeCellBulkField,
        patch_domain: WholeCellPatchDomain,
    ) -> f32 {
        cache.patch_bulk_concentrations[Self::patch_domain_index(patch_domain)]
            [Self::bulk_field_index(field)]
    }

    fn bulk_field_concentration_for_locality(
        cache: &WholeCellSpatialCouplingCache,
        field: WholeCellBulkField,
        spatial_scope: WholeCellSpatialScope,
        patch_domain: WholeCellPatchDomain,
    ) -> f32 {
        let scope_value = Self::bulk_field_concentration_for_scope(cache, field, spatial_scope);
        if patch_domain == WholeCellPatchDomain::Distributed {
            scope_value
        } else {
            let patch_value =
                Self::bulk_field_concentration_for_patch_domain(cache, field, patch_domain);
            if spatial_scope == WholeCellSpatialScope::WellMixed {
                patch_value
            } else {
                (scope_value.max(0.0) * patch_value.max(0.0)).sqrt()
            }
        }
    }

    fn bulk_field_concentration_for_locality_with_domain(
        &self,
        cache: &WholeCellSpatialCouplingCache,
        field: WholeCellBulkField,
        spatial_scope: WholeCellSpatialScope,
        patch_domain: WholeCellPatchDomain,
        chromosome_domain: Option<&str>,
    ) -> f32 {
        if chromosome_domain.is_none() {
            return Self::bulk_field_concentration_for_locality(
                cache,
                field,
                spatial_scope,
                patch_domain,
            );
        }
        let Some(lattice_species) = Self::lattice_species_for_bulk_field(field) else {
            return Self::bulk_field_concentration_for_locality(
                cache,
                field,
                spatial_scope,
                patch_domain,
            );
        };
        let uniform_weights = vec![1.0; self.lattice.total_voxels()];
        let weights = self.locality_weights(
            spatial_scope,
            patch_domain,
            chromosome_domain,
            &uniform_weights,
        );
        self.weighted_species_mean(lattice_species, &weights)
    }

    fn localized_pool_signal_target(field: WholeCellBulkField) -> f32 {
        match field {
            WholeCellBulkField::ATP => 0.52,
            WholeCellBulkField::AminoAcids => 0.44,
            WholeCellBulkField::Nucleotides => 0.40,
            WholeCellBulkField::MembranePrecursors => 0.36,
            WholeCellBulkField::ADP | WholeCellBulkField::Glucose | WholeCellBulkField::Oxygen => {
                0.40
            }
        }
    }

    fn drive_field_mean(&self, field: WholeCellRdmeDriveField) -> f32 {
        let values = self.rdme_drive_fields.field_slice(field);
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    }

    fn drive_field_mean_for_locality(
        &self,
        field: WholeCellRdmeDriveField,
        spatial_scope: WholeCellSpatialScope,
        patch_domain: WholeCellPatchDomain,
        chromosome_domain: Option<&str>,
    ) -> f32 {
        if chromosome_domain.is_some() {
            let uniform_weights = vec![1.0; self.lattice.total_voxels()];
            let weights = self.locality_weights(
                spatial_scope,
                patch_domain,
                chromosome_domain,
                &uniform_weights,
            );
            return self.rdme_drive_fields.weighted_mean(field, &weights);
        }
        let scope_value = Self::spatial_scope_field(spatial_scope)
            .map(|scope_field| self.localized_drive_mean(field, scope_field))
            .unwrap_or_else(|| self.drive_field_mean(field));
        if patch_domain == WholeCellPatchDomain::Distributed {
            scope_value
        } else {
            let patch_value = Self::patch_domain_field(patch_domain)
                .map(|patch_field| self.localized_drive_mean(field, patch_field))
                .unwrap_or(scope_value);
            if spatial_scope == WholeCellSpatialScope::WellMixed {
                patch_value
            } else {
                (scope_value.max(0.0) * patch_value.max(0.0)).sqrt()
            }
        }
    }

    fn localized_pool_reaction_field(
        reaction: &WholeCellReactionRuntimeState,
        species_state: &HashMap<String, WholeCellSpeciesRuntimeState>,
        use_products: bool,
    ) -> Option<(
        WholeCellBulkField,
        WholeCellSpatialScope,
        WholeCellPatchDomain,
        Option<String>,
    )> {
        let participants = if use_products {
            &reaction.products
        } else {
            &reaction.reactants
        };
        participants.iter().find_map(|participant| {
            let species = species_state.get(&participant.species_id)?;
            let field = species.bulk_field?;
            if species.spatial_scope == WholeCellSpatialScope::WellMixed
                && species.patch_domain == WholeCellPatchDomain::Distributed
            {
                return None;
            }
            Some((
                field,
                species.spatial_scope,
                species.patch_domain,
                species
                    .chromosome_domain
                    .clone()
                    .or_else(|| reaction.chromosome_domain.clone()),
            ))
        })
    }

    pub(super) fn localized_pool_transfer_hint(
        &self,
        reaction: &WholeCellReactionRuntimeState,
        species_state: &HashMap<String, WholeCellSpeciesRuntimeState>,
        cache: &WholeCellSpatialCouplingCache,
    ) -> f32 {
        let Some((field, spatial_scope, patch_domain, chromosome_domain)) =
            Self::localized_pool_reaction_field(reaction, species_state, true)
        else {
            return 1.0;
        };
        let local_level = Self::saturating_signal(
            self.bulk_field_concentration_for_locality_with_domain(
                cache,
                field,
                spatial_scope,
                patch_domain,
                chromosome_domain.as_deref(),
            ),
            Self::localized_pool_signal_target(field),
        );
        let crowding = self.drive_field_mean_for_locality(
            WholeCellRdmeDriveField::Crowding,
            spatial_scope,
            patch_domain,
            chromosome_domain.as_deref(),
        );
        let (demand, support, base, max_scale) = match field {
            WholeCellBulkField::ATP => (
                self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::AtpDemand,
                    spatial_scope,
                    patch_domain,
                    chromosome_domain.as_deref(),
                ),
                self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::EnergySource,
                    spatial_scope,
                    patch_domain,
                    chromosome_domain.as_deref(),
                ),
                0.16,
                2.6,
            ),
            WholeCellBulkField::AminoAcids => (
                self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::AminoDemand,
                    spatial_scope,
                    patch_domain,
                    chromosome_domain.as_deref(),
                ),
                0.42 * self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::EnergySource,
                    spatial_scope,
                    patch_domain,
                    chromosome_domain.as_deref(),
                ),
                0.14,
                2.4,
            ),
            WholeCellBulkField::Nucleotides => (
                self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::NucleotideDemand,
                    spatial_scope,
                    patch_domain,
                    chromosome_domain.as_deref(),
                ),
                0.36 * self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::EnergySource,
                    spatial_scope,
                    patch_domain,
                    chromosome_domain.as_deref(),
                ),
                0.15,
                2.5,
            ),
            WholeCellBulkField::MembranePrecursors => (
                self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::MembraneDemand,
                    spatial_scope,
                    patch_domain,
                    chromosome_domain.as_deref(),
                ),
                self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::MembraneSource,
                    spatial_scope,
                    patch_domain,
                    chromosome_domain.as_deref(),
                ),
                0.16,
                2.6,
            ),
            WholeCellBulkField::ADP | WholeCellBulkField::Glucose | WholeCellBulkField::Oxygen => {
                (0.0, 0.0, 0.10, 1.8)
            }
        };
        Self::finite_scale(
            base + 0.56 * demand + 0.20 * support - 0.18 * crowding - 0.28 * local_level,
            1.0,
            0.04,
            max_scale,
        )
    }

    pub(super) fn localized_pool_turnover_hint(
        &self,
        reaction: &WholeCellReactionRuntimeState,
        species_state: &HashMap<String, WholeCellSpeciesRuntimeState>,
        cache: &WholeCellSpatialCouplingCache,
    ) -> f32 {
        let Some((field, spatial_scope, patch_domain, chromosome_domain)) =
            Self::localized_pool_reaction_field(reaction, species_state, false)
        else {
            return 1.0;
        };
        let local_level = Self::saturating_signal(
            self.bulk_field_concentration_for_locality_with_domain(
                cache,
                field,
                spatial_scope,
                patch_domain,
                chromosome_domain.as_deref(),
            ),
            Self::localized_pool_signal_target(field),
        );
        let crowding = self.drive_field_mean_for_locality(
            WholeCellRdmeDriveField::Crowding,
            spatial_scope,
            patch_domain,
            chromosome_domain.as_deref(),
        );
        let demand = match field {
            WholeCellBulkField::ATP => self.drive_field_mean_for_locality(
                WholeCellRdmeDriveField::AtpDemand,
                spatial_scope,
                patch_domain,
                chromosome_domain.as_deref(),
            ),
            WholeCellBulkField::AminoAcids => self.drive_field_mean_for_locality(
                WholeCellRdmeDriveField::AminoDemand,
                spatial_scope,
                patch_domain,
                chromosome_domain.as_deref(),
            ),
            WholeCellBulkField::Nucleotides => self.drive_field_mean_for_locality(
                WholeCellRdmeDriveField::NucleotideDemand,
                spatial_scope,
                patch_domain,
                chromosome_domain.as_deref(),
            ),
            WholeCellBulkField::MembranePrecursors => self.drive_field_mean_for_locality(
                WholeCellRdmeDriveField::MembraneDemand,
                spatial_scope,
                patch_domain,
                chromosome_domain.as_deref(),
            ),
            WholeCellBulkField::ADP | WholeCellBulkField::Glucose | WholeCellBulkField::Oxygen => {
                0.0
            }
        };
        Self::finite_scale(
            0.12 + 0.34 * local_level + 0.18 * crowding - 0.30 * demand,
            1.0,
            0.03,
            2.2,
        )
    }

    pub(super) fn pool_species_anchor(
        bulk_field: Option<WholeCellBulkField>,
        spatial_scope: WholeCellSpatialScope,
        patch_domain: WholeCellPatchDomain,
        basal_abundance: f32,
        cache: &WholeCellSpatialCouplingCache,
    ) -> f32 {
        let anchor = if let Some(field) = bulk_field {
            Self::bulk_field_species_scale(field)
                * Self::bulk_field_concentration_for_locality(
                    cache,
                    field,
                    spatial_scope,
                    patch_domain,
                )
                .max(0.0)
        } else {
            basal_abundance.max(0.0)
        };
        anchor.clamp(0.0, 4096.0)
    }

    fn spatial_scope_overlap(
        cache: &WholeCellSpatialCouplingCache,
        source_scope: WholeCellSpatialScope,
        target_scope: WholeCellSpatialScope,
    ) -> f32 {
        cache.overlap[Self::spatial_scope_index(source_scope)]
            [Self::spatial_scope_index(target_scope)]
    }

    fn patch_domain_overlap(
        cache: &WholeCellSpatialCouplingCache,
        source_patch_domain: WholeCellPatchDomain,
        target_patch_domain: WholeCellPatchDomain,
    ) -> f32 {
        cache.patch_overlap[Self::patch_domain_index(source_patch_domain)]
            [Self::patch_domain_index(target_patch_domain)]
    }

    fn chromosome_domain_overlap(
        &self,
        source_domain: Option<&str>,
        target_domain: Option<&str>,
    ) -> f32 {
        match (source_domain, target_domain) {
            (Some(source), Some(target)) if source == target => 1.0,
            (Some(source), Some(target)) => {
                let Some(source_index) = self.chromosome_domain_index_by_id(source) else {
                    return 1.0;
                };
                let Some(target_index) = self.chromosome_domain_index_by_id(target) else {
                    return 1.0;
                };
                let source_weights = self.chromosome_domain_weights(source_index);
                let target_weights = self.chromosome_domain_weights(target_index);
                let source_total = source_weights
                    .iter()
                    .map(|value| value.max(0.0))
                    .sum::<f32>();
                if source_total <= 1.0e-6 {
                    1.0
                } else {
                    let shared = source_weights
                        .iter()
                        .zip(target_weights.iter())
                        .map(|(lhs, rhs)| lhs.max(0.0).min(rhs.max(0.0)))
                        .sum::<f32>();
                    (shared / source_total).clamp(0.02, 1.0)
                }
            }
            _ => 1.0,
        }
    }

    pub(super) fn effective_runtime_species_count_for_locality(
        &self,
        species: &WholeCellSpeciesRuntimeState,
        target_scope: WholeCellSpatialScope,
        target_patch_domain: WholeCellPatchDomain,
        target_chromosome_domain: Option<&str>,
        cache: &WholeCellSpatialCouplingCache,
    ) -> f32 {
        let preferred_scope = if target_scope == WholeCellSpatialScope::WellMixed {
            species.spatial_scope
        } else {
            target_scope
        };
        let preferred_patch_domain = if target_patch_domain == WholeCellPatchDomain::Distributed {
            species.patch_domain
        } else {
            target_patch_domain
        };
        if let Some(field) = species.bulk_field {
            return Self::bulk_field_species_scale(field)
                * self
                    .bulk_field_concentration_for_locality_with_domain(
                        cache,
                        field,
                        preferred_scope,
                        preferred_patch_domain,
                        species
                            .chromosome_domain
                            .as_deref()
                            .or(target_chromosome_domain),
                    )
                    .max(0.0);
        }
        species.count.max(0.0)
            * Self::spatial_scope_overlap(cache, species.spatial_scope, preferred_scope)
            * Self::patch_domain_overlap(cache, species.patch_domain, preferred_patch_domain)
            * self.chromosome_domain_overlap(
                species.chromosome_domain.as_deref(),
                target_chromosome_domain,
            )
    }

    fn locality_weights(
        &self,
        scope: WholeCellSpatialScope,
        patch_domain: WholeCellPatchDomain,
        chromosome_domain: Option<&str>,
        uniform_weights: &[f32],
    ) -> Vec<f32> {
        let scope_field = Self::spatial_scope_field(scope);
        let patch_field = Self::patch_domain_field(patch_domain);
        let mut weights = match (scope_field, patch_field) {
            (None, None) => uniform_weights.to_vec(),
            (Some(field), None) | (None, Some(field)) => {
                self.spatial_fields.field_slice(field).to_vec()
            }
            (Some(lhs), Some(rhs)) if lhs == rhs => self.spatial_fields.field_slice(lhs).to_vec(),
            (Some(lhs), Some(rhs)) => self
                .spatial_fields
                .field_slice(lhs)
                .iter()
                .zip(self.spatial_fields.field_slice(rhs).iter())
                .map(|(left, right)| (left.max(0.0) * right.max(0.0)).sqrt())
                .collect(),
        };
        if let Some(domain_id) = chromosome_domain {
            if let Some(domain_index) = self.chromosome_domain_index_by_id(domain_id) {
                let domain_weights = self.chromosome_domain_weights(domain_index);
                if domain_weights.len() == weights.len() {
                    if weights.iter().all(|weight| (*weight - 1.0).abs() < 1.0e-6) {
                        weights = domain_weights;
                    } else {
                        weights = weights
                            .iter()
                            .zip(domain_weights.iter())
                            .map(|(locality, domain)| (locality.max(0.0) * domain.max(0.0)).sqrt())
                            .collect();
                    }
                }
            }
        }
        weights
    }

    pub(super) fn localized_drive_mean(
        &self,
        drive_field: WholeCellRdmeDriveField,
        spatial_field: IntracellularSpatialField,
    ) -> f32 {
        self.rdme_drive_fields
            .weighted_mean(drive_field, self.spatial_fields.field_slice(spatial_field))
    }

    fn accumulate_normalized_signal(target: &mut [f32], weights: &[f32], amplitude: f32) {
        if !amplitude.is_finite() || amplitude.abs() <= 1.0e-6 || target.len() != weights.len() {
            return;
        }
        let weight_sum = weights
            .iter()
            .map(|weight| weight.max(0.0))
            .sum::<f32>()
            .max(1.0e-6);
        let mean_scale = target.len() as f32 / weight_sum;
        for (value, weight) in target.iter_mut().zip(weights.iter()) {
            *value += amplitude * weight.max(0.0) * mean_scale;
        }
    }

    fn accumulate_patch_signal(
        target: &mut [f32],
        x_dim: usize,
        y_dim: usize,
        z_dim: usize,
        center_x: usize,
        center_y: usize,
        center_z: usize,
        radius: usize,
        amplitude: f32,
    ) {
        if !amplitude.is_finite() || amplitude.abs() <= 1.0e-6 || target.is_empty() {
            return;
        }
        let radius = radius.max(1) as isize;
        let sigma = (radius as f32 * 0.65).max(0.75);
        let x_start = center_x.saturating_sub(radius as usize);
        let y_start = center_y.saturating_sub(radius as usize);
        let z_start = center_z.saturating_sub(radius as usize);
        let x_end = (center_x + radius as usize + 1).min(x_dim);
        let y_end = (center_y + radius as usize + 1).min(y_dim);
        let z_end = (center_z + radius as usize + 1).min(z_dim);
        let mut weighted_indices = Vec::new();
        let mut weight_total = 0.0f32;
        for z in z_start..z_end {
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let dx = x as isize - center_x as isize;
                    let dy = y as isize - center_y as isize;
                    let dz = z as isize - center_z as isize;
                    let dist2 = (dx * dx + dy * dy + dz * dz) as f32;
                    if dist2 > (radius * radius) as f32 {
                        continue;
                    }
                    let weight = (-0.5 * dist2 / (sigma * sigma)).exp().max(1.0e-4);
                    let gid = z * y_dim * x_dim + y * x_dim + x;
                    weighted_indices.push((gid, weight));
                    weight_total += weight;
                }
            }
        }
        if weight_total <= 1.0e-6 {
            return;
        }
        let mean_scale = weighted_indices.len() as f32 / weight_total;
        for (gid, weight) in weighted_indices {
            target[gid] += amplitude * weight * mean_scale;
        }
    }

    fn species_crowding_weight(species_class: WholeCellSpeciesClass) -> f32 {
        match species_class {
            WholeCellSpeciesClass::Pool => 0.0,
            WholeCellSpeciesClass::Rna => 0.10,
            WholeCellSpeciesClass::Protein => 0.14,
            WholeCellSpeciesClass::ComplexSubunitPool => 0.18,
            WholeCellSpeciesClass::ComplexNucleationIntermediate => 0.22,
            WholeCellSpeciesClass::ComplexElongationIntermediate => 0.24,
            WholeCellSpeciesClass::ComplexMature => 0.28,
        }
    }

    fn reaction_flux_signal(reaction: &WholeCellReactionRuntimeState) -> f32 {
        let fallback = reaction.nominal_rate.max(0.0)
            * reaction.reactant_satisfaction.max(0.0)
            * reaction.catalyst_support.max(0.0)
            * 0.08;
        reaction
            .current_flux
            .max(fallback)
            .clamp(0.0, 12.0)
            .sqrt()
            .clamp(0.0, 2.5)
    }

    fn clamp_signal_field(values: &mut [f32], max_value: f32) {
        for value in values {
            *value = value.clamp(0.0, max_value);
        }
    }

    pub(super) fn refresh_rdme_drive_fields(&mut self) {
        let total_voxels = self.lattice.total_voxels();
        if total_voxels == 0 {
            return;
        }

        let uniform_weights = vec![1.0; total_voxels];
        let mut energy_source = vec![0.0; total_voxels];
        let mut atp_demand = vec![0.0; total_voxels];
        let mut amino_demand = vec![0.0; total_voxels];
        let mut nucleotide_demand = vec![0.0; total_voxels];
        let mut membrane_source = vec![0.0; total_voxels];
        let mut membrane_demand = vec![0.0; total_voxels];
        let mut crowding = vec![0.0; total_voxels];

        let species_by_id = self
            .organism_species
            .iter()
            .map(|species| (species.id.clone(), species))
            .collect::<HashMap<_, _>>();

        for species in &self.organism_species {
            let weights = self.locality_weights(
                species.spatial_scope,
                species.patch_domain,
                species.chromosome_domain.as_deref(),
                &uniform_weights,
            );
            let anchor = species.anchor_count.max(species.basal_abundance).max(1.0);
            let abundance_signal = (species.count.max(0.0) / anchor).sqrt().clamp(0.0, 2.5);
            let crowding_signal =
                abundance_signal * Self::species_crowding_weight(species.species_class);
            if crowding_signal > 0.0 {
                Self::accumulate_normalized_signal(&mut crowding, &weights, crowding_signal);
            }
            if species.spatial_scope != WholeCellSpatialScope::WellMixed
                && species.species_class != WholeCellSpeciesClass::Pool
            {
                match species.asset_class {
                    WholeCellAssetClass::Energy => {
                        Self::accumulate_normalized_signal(
                            &mut energy_source,
                            &weights,
                            0.26 * abundance_signal,
                        );
                    }
                    WholeCellAssetClass::Membrane | WholeCellAssetClass::Constriction => {
                        Self::accumulate_normalized_signal(
                            &mut membrane_source,
                            &weights,
                            0.22 * abundance_signal,
                        );
                    }
                    WholeCellAssetClass::Replication => {
                        Self::accumulate_normalized_signal(
                            &mut nucleotide_demand,
                            &weights,
                            0.16 * abundance_signal,
                        );
                    }
                    _ => {}
                }
            }
        }

        for reaction in &self.organism_reactions {
            let weights = self.locality_weights(
                reaction.spatial_scope,
                reaction.patch_domain,
                reaction.chromosome_domain.as_deref(),
                &uniform_weights,
            );
            let flux_signal = Self::reaction_flux_signal(reaction);
            if flux_signal <= 0.0 {
                continue;
            }
            if reaction.spatial_scope != WholeCellSpatialScope::WellMixed {
                Self::accumulate_normalized_signal(&mut crowding, &weights, 0.04 * flux_signal);
            }
            for participant in &reaction.reactants {
                let Some(species) = species_by_id.get(&participant.species_id) else {
                    continue;
                };
                let amplitude = flux_signal * participant.stoichiometry.max(0.0);
                match species.bulk_field {
                    Some(WholeCellBulkField::ATP) => {
                        Self::accumulate_normalized_signal(&mut atp_demand, &weights, amplitude);
                    }
                    Some(WholeCellBulkField::AminoAcids) => {
                        Self::accumulate_normalized_signal(&mut amino_demand, &weights, amplitude);
                    }
                    Some(WholeCellBulkField::Nucleotides) => Self::accumulate_normalized_signal(
                        &mut nucleotide_demand,
                        &weights,
                        amplitude,
                    ),
                    Some(WholeCellBulkField::MembranePrecursors) => {
                        Self::accumulate_normalized_signal(
                            &mut membrane_demand,
                            &weights,
                            amplitude,
                        )
                    }
                    _ => {}
                }
            }
            for participant in &reaction.products {
                let Some(species) = species_by_id.get(&participant.species_id) else {
                    continue;
                };
                let amplitude = flux_signal * participant.stoichiometry.max(0.0);
                match species.bulk_field {
                    Some(WholeCellBulkField::ATP) => Self::accumulate_normalized_signal(
                        &mut energy_source,
                        &weights,
                        0.85 * amplitude,
                    ),
                    Some(WholeCellBulkField::MembranePrecursors) => {
                        Self::accumulate_normalized_signal(
                            &mut membrane_source,
                            &weights,
                            amplitude,
                        )
                    }
                    _ => {}
                }
            }
        }

        for report in &self.chemistry_site_reports {
            let patch_scale = report.localization_score.clamp(0.0, 1.5);
            if patch_scale <= 0.0 {
                continue;
            }
            let radius = report.patch_radius.max(1);
            let energy_source_signal = patch_scale
                * (0.22 * report.atp_support.max(0.0) + 0.28 * report.mean_atp_flux.max(0.0));
            let atp_demand_signal = patch_scale * report.energy_draw.max(0.0);
            let amino_demand_signal = patch_scale
                * (0.45 * report.substrate_draw.max(0.0)
                    + 0.55 * report.biosynthetic_draw.max(0.0));
            let nucleotide_signal = patch_scale
                * (0.60 * report.biosynthetic_draw.max(0.0)
                    + 0.35 * (1.0 - report.nucleotide_support).max(0.0));
            let membrane_source_signal = patch_scale
                * (0.40 * report.membrane_support.max(0.0)
                    + 0.24 * report.assembly_stability.max(0.0));
            let membrane_demand_signal = patch_scale
                * (0.38 * report.assembly_turnover.max(0.0)
                    + 0.22 * report.byproduct_load.max(0.0));
            let crowding_signal = patch_scale
                * (0.25 * report.assembly_occupancy.max(0.0)
                    + 0.20 * report.byproduct_load.max(0.0)
                    + 0.30 * (1.0 - report.demand_satisfaction).max(0.0));

            Self::accumulate_patch_signal(
                &mut energy_source,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                energy_source_signal,
            );
            Self::accumulate_patch_signal(
                &mut atp_demand,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                atp_demand_signal,
            );
            Self::accumulate_patch_signal(
                &mut amino_demand,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                amino_demand_signal,
            );
            Self::accumulate_patch_signal(
                &mut nucleotide_demand,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                nucleotide_signal,
            );
            Self::accumulate_patch_signal(
                &mut membrane_source,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                membrane_source_signal,
            );
            Self::accumulate_patch_signal(
                &mut membrane_demand,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                membrane_demand_signal,
            );
            Self::accumulate_patch_signal(
                &mut crowding,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                crowding_signal,
            );
        }

        Self::clamp_signal_field(&mut energy_source, 2.5);
        Self::clamp_signal_field(&mut atp_demand, 2.5);
        Self::clamp_signal_field(&mut amino_demand, 2.5);
        Self::clamp_signal_field(&mut nucleotide_demand, 2.5);
        Self::clamp_signal_field(&mut membrane_source, 2.5);
        Self::clamp_signal_field(&mut membrane_demand, 2.5);
        Self::clamp_signal_field(&mut crowding, 1.6);

        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::EnergySource, &energy_source);
        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::AtpDemand, &atp_demand);
        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::AminoDemand, &amino_demand);
        let _ = self.rdme_drive_fields.set_field(
            WholeCellRdmeDriveField::NucleotideDemand,
            &nucleotide_demand,
        );
        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::MembraneSource, &membrane_source);
        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::MembraneDemand, &membrane_demand);
        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::Crowding, &crowding);
    }

    pub(super) fn apply_bulk_field_delta(
        &mut self,
        field: WholeCellBulkField,
        spatial_scope: WholeCellSpatialScope,
        patch_domain: WholeCellPatchDomain,
        chromosome_domain: Option<&str>,
        delta_count: f32,
    ) -> bool {
        if !delta_count.is_finite() || delta_count.abs() <= 1.0e-6 {
            return false;
        }
        let delta_mm = delta_count / Self::bulk_field_species_scale(field).max(1.0);
        if !delta_mm.is_finite() || delta_mm.abs() <= 1.0e-8 {
            return false;
        }
        let uniform_weights = vec![1.0; self.lattice.total_voxels()];
        let weights = self.locality_weights(
            spatial_scope,
            patch_domain,
            chromosome_domain,
            &uniform_weights,
        );
        match field {
            WholeCellBulkField::ATP => {
                if spatial_scope == WholeCellSpatialScope::WellMixed
                    && patch_domain == WholeCellPatchDomain::Distributed
                {
                    self.lattice
                        .apply_uniform_delta(IntracellularSpecies::ATP, delta_mm);
                } else {
                    self.lattice.apply_weighted_delta(
                        IntracellularSpecies::ATP,
                        delta_mm,
                        &weights,
                    );
                }
                self.atp_mm = (self.atp_mm + delta_mm).max(0.0);
                true
            }
            WholeCellBulkField::AminoAcids => {
                if spatial_scope == WholeCellSpatialScope::WellMixed
                    && patch_domain == WholeCellPatchDomain::Distributed
                {
                    self.lattice
                        .apply_uniform_delta(IntracellularSpecies::AminoAcids, delta_mm);
                } else {
                    self.lattice.apply_weighted_delta(
                        IntracellularSpecies::AminoAcids,
                        delta_mm,
                        &weights,
                    );
                }
                self.amino_acids_mm = (self.amino_acids_mm + delta_mm).max(0.0);
                true
            }
            WholeCellBulkField::Nucleotides => {
                if spatial_scope == WholeCellSpatialScope::WellMixed
                    && patch_domain == WholeCellPatchDomain::Distributed
                {
                    self.lattice
                        .apply_uniform_delta(IntracellularSpecies::Nucleotides, delta_mm);
                } else {
                    self.lattice.apply_weighted_delta(
                        IntracellularSpecies::Nucleotides,
                        delta_mm,
                        &weights,
                    );
                }
                self.nucleotides_mm = (self.nucleotides_mm + delta_mm).max(0.0);
                true
            }
            WholeCellBulkField::MembranePrecursors => {
                if spatial_scope == WholeCellSpatialScope::WellMixed
                    && patch_domain == WholeCellPatchDomain::Distributed
                {
                    self.lattice
                        .apply_uniform_delta(IntracellularSpecies::MembranePrecursors, delta_mm);
                } else {
                    self.lattice.apply_weighted_delta(
                        IntracellularSpecies::MembranePrecursors,
                        delta_mm,
                        &weights,
                    );
                }
                self.membrane_precursors_mm = (self.membrane_precursors_mm + delta_mm).max(0.0);
                true
            }
            WholeCellBulkField::ADP => {
                self.adp_mm = (self.adp_mm + delta_mm).max(0.0);
                false
            }
            WholeCellBulkField::Glucose => {
                self.glucose_mm = (self.glucose_mm + delta_mm).max(0.0);
                false
            }
            WholeCellBulkField::Oxygen => {
                self.oxygen_mm = (self.oxygen_mm + delta_mm).max(0.0);
                false
            }
        }
    }

    pub(super) fn apply_operon_inventory_delta(
        &mut self,
        operon: &str,
        transcript_delta: f32,
        protein_delta: f32,
        dt_scale: f32,
    ) -> bool {
        let Some(unit) = self
            .organism_expression
            .transcription_units
            .iter_mut()
            .find(|unit| unit.name == operon)
        else {
            return false;
        };
        Self::normalize_expression_execution_state(unit);
        if transcript_delta.is_finite() && transcript_delta.abs() > 1.0e-6 {
            if transcript_delta >= 0.0 {
                unit.mature_transcript_abundance =
                    (unit.mature_transcript_abundance + transcript_delta).clamp(0.0, 1024.0);
            } else {
                let mut remaining = -transcript_delta;
                remaining -= Self::subtract_expression_pool(
                    &mut unit.damaged_transcript_abundance,
                    remaining,
                );
                remaining -= Self::subtract_expression_pool(
                    &mut unit.nascent_transcript_abundance,
                    remaining,
                );
                let _ = Self::subtract_expression_pool(
                    &mut unit.mature_transcript_abundance,
                    remaining,
                );
            }
            if dt_scale > 0.0 {
                if transcript_delta >= 0.0 {
                    unit.transcript_synthesis_rate += transcript_delta / dt_scale;
                } else {
                    unit.transcript_turnover_rate += (-transcript_delta) / dt_scale;
                }
            }
        }
        if protein_delta.is_finite() && protein_delta.abs() > 1.0e-6 {
            if protein_delta >= 0.0 {
                unit.mature_protein_abundance =
                    (unit.mature_protein_abundance + protein_delta).clamp(0.0, 2048.0);
            } else {
                let mut remaining = -protein_delta;
                remaining -=
                    Self::subtract_expression_pool(&mut unit.damaged_protein_abundance, remaining);
                remaining -=
                    Self::subtract_expression_pool(&mut unit.nascent_protein_abundance, remaining);
                let _ =
                    Self::subtract_expression_pool(&mut unit.mature_protein_abundance, remaining);
            }
            if dt_scale > 0.0 {
                if protein_delta >= 0.0 {
                    unit.protein_synthesis_rate += protein_delta / dt_scale;
                } else {
                    unit.protein_turnover_rate += (-protein_delta) / dt_scale;
                }
            }
        }
        Self::normalize_expression_execution_state(unit);
        true
    }

    pub(super) fn refresh_expression_inventory_totals(&mut self) {
        for unit in &mut self.organism_expression.transcription_units {
            Self::normalize_expression_execution_state(unit);
        }
        self.organism_expression.total_transcript_abundance = self
            .organism_expression
            .transcription_units
            .iter()
            .map(|unit| unit.transcript_abundance.max(0.0))
            .sum::<f32>();
        self.organism_expression.total_protein_abundance = self
            .organism_expression
            .transcription_units
            .iter()
            .map(|unit| unit.protein_abundance.max(0.0))
            .sum::<f32>();
    }

    pub(super) fn apply_named_complex_species_delta(
        &mut self,
        complex_id: &str,
        species_class: WholeCellSpeciesClass,
        delta: f32,
    ) -> bool {
        let Some(state) = self
            .named_complexes
            .iter_mut()
            .find(|state| state.id == complex_id)
        else {
            return false;
        };
        match species_class {
            WholeCellSpeciesClass::ComplexSubunitPool => {
                state.subunit_pool = (state.subunit_pool + delta).clamp(0.0, 2048.0);
            }
            WholeCellSpeciesClass::ComplexNucleationIntermediate => {
                state.nucleation_intermediate =
                    (state.nucleation_intermediate + delta).clamp(0.0, 512.0);
            }
            WholeCellSpeciesClass::ComplexElongationIntermediate => {
                state.elongation_intermediate =
                    (state.elongation_intermediate + delta).clamp(0.0, 512.0);
            }
            WholeCellSpeciesClass::ComplexMature => {
                state.abundance = (state.abundance + delta).clamp(0.0, 512.0);
            }
            _ => return false,
        }
        true
    }
}
