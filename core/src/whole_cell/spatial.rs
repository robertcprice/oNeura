//! Spatial-field and localized-pool helpers for the native whole-cell runtime.
//!
//! These methods own the voxel-weighted geometry fields and the localized pool
//! summaries derived from them, so RDME/locality logic does not stay embedded
//! in the main simulator file.

use super::*;

impl WholeCellSimulator {
    pub(super) fn refresh_spatial_fields(&mut self) {
        let total_voxels = self.lattice.total_voxels();
        if total_voxels == 0 {
            return;
        }
        let x_dim = self.lattice.x_dim.max(1);
        let y_dim = self.lattice.y_dim.max(1);
        let z_dim = self.lattice.z_dim.max(1);
        let center_x = (x_dim as f32 - 1.0) * 0.5;
        let center_y = (y_dim as f32 - 1.0) * 0.5;
        let center_z = (z_dim as f32 - 1.0) * 0.5;
        let axial_scale = center_x.max(1.0);
        let radial_y_scale = center_y.max(1.0);
        let radial_z_scale = center_z.max(1.0);
        let membrane_depth = (0.18
            + self
                .organism_data
                .as_ref()
                .map(|organism| organism.geometry.membrane_fraction.clamp(0.08, 0.32))
                .unwrap_or(0.16))
        .clamp(0.10, 0.35);
        let septum_width =
            (0.10 + 0.18 * self.membrane_division_state.septum_radius_fraction).clamp(0.06, 0.32);
        let septum_gain =
            (0.35 + 0.65 * self.membrane_division_state.septum_localization).clamp(0.20, 1.25);
        let division_progress = self.current_division_progress();
        let cardiolipin_share = self.membrane_cardiolipin_share();
        let pole_width =
            (0.10 + 0.18 * cardiolipin_share + 0.08 * division_progress).clamp(0.08, 0.26);
        let chromosome_radius_fraction = self
            .organism_data
            .as_ref()
            .map(|organism| {
                organism
                    .geometry
                    .chromosome_radius_fraction
                    .clamp(0.18, 0.70)
            })
            .unwrap_or(0.40);
        let axial_spread = (0.12
            + 0.22 * (1.0 - self.chromosome_state.compaction_fraction.clamp(0.0, 1.0))
            + 0.08 * self.chromosome_state.replicated_fraction.clamp(0.0, 1.0))
        .clamp(0.08, 0.45);
        let radial_spread = (0.10
            + 0.30 * chromosome_radius_fraction
            + 0.12 * (1.0 - self.chromosome_state.compaction_fraction.clamp(0.0, 1.0)))
        .clamp(0.10, 0.42);
        let separation_fraction = (self.current_chromosome_separation_nm()
            / (self.current_radius_nm() * 2.2).max(self.lattice.voxel_size_nm))
        .clamp(0.0, 0.45);
        let left_center = 0.5 - 0.5 * separation_fraction;
        let right_center = 0.5 + 0.5 * separation_fraction;
        let occupancy_gain = (0.45
            + 0.55 * self.chromosome_state.compaction_fraction.clamp(0.0, 1.0)
            + 0.20 * self.chromosome_state.replicated_fraction.clamp(0.0, 1.0))
        .clamp(0.20, 1.35);

        let mut membrane = vec![0.0; total_voxels];
        let mut septum = vec![0.0; total_voxels];
        let mut nucleoid = vec![0.0; total_voxels];
        let mut membrane_band = vec![0.0; total_voxels];
        let mut poles = vec![0.0; total_voxels];

        for gid in 0..total_voxels {
            let z = gid / (y_dim * x_dim);
            let rem = gid - z * y_dim * x_dim;
            let y = rem / x_dim;
            let x = rem - y * x_dim;

            let x_norm = (x as f32 - center_x) / axial_scale;
            let y_norm = (y as f32 - center_y) / radial_y_scale;
            let z_norm = (z as f32 - center_z) / radial_z_scale;

            let boundary_distance = x_norm
                .abs()
                .max(y_norm.abs())
                .max(z_norm.abs())
                .clamp(0.0, 1.0);
            let membrane_adjacency =
                ((boundary_distance - (1.0 - membrane_depth)) / membrane_depth).clamp(0.0, 1.0);
            let septum_axis = (-0.5 * (x_norm / septum_width).powi(2)).exp();
            let radial_centering = (1.0 - 0.45 * (y_norm.abs().max(z_norm.abs()))).clamp(0.15, 1.0);
            let septum_zone =
                (septum_gain * septum_axis * membrane_adjacency * radial_centering).clamp(0.0, 1.0);
            let pole_axis =
                (-0.5 * ((1.0 - x_norm.abs()).max(0.0) / pole_width.max(1.0e-3)).powi(2)).exp();
            let pole_zone = (membrane_adjacency
                * pole_axis
                * (0.65 + 0.35 * radial_centering)
                * (0.65 + 0.35 * cardiolipin_share))
                .clamp(0.0, 1.0);
            let membrane_band_zone = (membrane_adjacency
                * (1.0 - pole_axis).clamp(0.0, 1.0)
                * (1.0 - 0.70 * septum_axis).clamp(0.0, 1.0)
                * radial_centering)
                .clamp(0.0, 1.0);

            let y_frac = (y as f32 + 0.5) / y_dim as f32 - 0.5;
            let z_frac = (z as f32 + 0.5) / z_dim as f32 - 0.5;
            let radial_distance = (y_frac.powi(2) + z_frac.powi(2)).sqrt();
            let radial_term = (radial_distance / radial_spread.max(1.0e-3)).powi(2);
            let x_frac = (x as f32 + 0.5) / x_dim as f32;
            let left_term = ((x_frac - left_center) / axial_spread.max(1.0e-3)).powi(2);
            let right_term = ((x_frac - right_center) / axial_spread.max(1.0e-3)).powi(2);
            let nucleoid_occupancy = (occupancy_gain
                * ((-0.5 * (left_term + radial_term)).exp()
                    + (-0.5 * (right_term + radial_term)).exp()))
            .clamp(0.0, 1.0);

            membrane[gid] = membrane_adjacency;
            septum[gid] = septum_zone;
            nucleoid[gid] = nucleoid_occupancy;
            membrane_band[gid] = membrane_band_zone;
            poles[gid] = pole_zone;
        }

        let _ = self
            .spatial_fields
            .set_field(IntracellularSpatialField::MembraneAdjacency, &membrane);
        let _ = self
            .spatial_fields
            .set_field(IntracellularSpatialField::SeptumZone, &septum);
        let _ = self
            .spatial_fields
            .set_field(IntracellularSpatialField::NucleoidOccupancy, &nucleoid);
        let _ = self
            .spatial_fields
            .set_field(IntracellularSpatialField::MembraneBandZone, &membrane_band);
        let _ = self
            .spatial_fields
            .set_field(IntracellularSpatialField::PoleZone, &poles);
    }

    pub(super) fn apply_spatial_field_state(&mut self, spatial: &WholeCellSpatialFieldState) {
        let _ = self.spatial_fields.set_field(
            IntracellularSpatialField::MembraneAdjacency,
            &spatial.membrane_adjacency,
        );
        let _ = self
            .spatial_fields
            .set_field(IntracellularSpatialField::SeptumZone, &spatial.septum_zone);
        let _ = self.spatial_fields.set_field(
            IntracellularSpatialField::NucleoidOccupancy,
            &spatial.nucleoid_occupancy,
        );
        let _ = self.spatial_fields.set_field(
            IntracellularSpatialField::MembraneBandZone,
            &spatial.membrane_band_zone,
        );
        let _ = self
            .spatial_fields
            .set_field(IntracellularSpatialField::PoleZone, &spatial.pole_zone);
    }

    pub(super) fn spatial_species_mean(
        &self,
        species: IntracellularSpecies,
        field: IntracellularSpatialField,
    ) -> f32 {
        self.spatial_fields
            .weighted_mean_species(&self.lattice, species, field)
    }

    pub(super) fn weighted_species_mean(
        &self,
        species: IntracellularSpecies,
        weights: &[f32],
    ) -> f32 {
        let total_voxels = self.lattice.total_voxels();
        if weights.len() != total_voxels {
            return self.lattice.mean_species(species);
        }
        let start = species.index() * total_voxels;
        let species_values = &self.lattice.current[start..start + total_voxels];
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;
        for (&value, &weight) in species_values.iter().zip(weights.iter()) {
            let clamped = weight.max(0.0);
            weighted_sum += value * clamped;
            weight_total += clamped;
        }
        if weight_total <= 1.0e-6 {
            self.lattice.mean_species(species)
        } else {
            weighted_sum / weight_total
        }
    }

    pub(super) fn chromosome_domain_weights(&self, domain_index: u32) -> Vec<f32> {
        let total_voxels = self.lattice.total_voxels();
        let x_dim = self.lattice.x_dim.max(1);
        let selected_domain = self.compiled_chromosome_domains().and_then(|domains| {
            if domains.is_empty() {
                None
            } else {
                let clamped_index = (domain_index as usize).min(domains.len().saturating_sub(1));
                domains.get(clamped_index)
            }
        });
        let nucleoid = self
            .spatial_fields
            .field_slice(IntracellularSpatialField::NucleoidOccupancy);
        let mut weights = vec![0.0; total_voxels];
        if let Some(domain) = selected_domain {
            let axial_center = domain.axial_center_fraction.clamp(0.02, 0.98);
            let base_spread = domain.axial_spread_fraction.clamp(0.05, 0.30);
            let axial_spread = (base_spread
                * (1.0
                    + 0.30 * (1.0 - self.chromosome_state.compaction_fraction.clamp(0.0, 1.0))
                    + 0.18 * (1.0 - self.chromosome_state.segregation_progress.clamp(0.0, 1.0))))
            .clamp(0.06, 0.28);
            for gid in 0..total_voxels {
                let x = gid % x_dim;
                let x_frac = (x as f32 + 0.5) / x_dim as f32;
                let axial_term =
                    (-0.5 * ((x_frac - axial_center) / axial_spread.max(1.0e-3)).powi(2)).exp();
                weights[gid] = nucleoid[gid].max(0.0) * axial_term;
            }
        } else {
            for gid in 0..total_voxels {
                weights[gid] = nucleoid[gid].max(0.0);
            }
        }
        weights
    }

    pub(super) fn chromosome_domain_species_mean(
        &self,
        species: IntracellularSpecies,
        domain_index: u32,
    ) -> f32 {
        let weights = self.chromosome_domain_weights(domain_index);
        self.weighted_species_mean(species, &weights)
    }

    pub(super) fn localized_nucleotide_pool_mm(&self) -> f32 {
        self.spatial_species_mean(
            IntracellularSpecies::Nucleotides,
            IntracellularSpatialField::NucleoidOccupancy,
        )
    }

    pub(super) fn localized_nucleoid_atp_pool_mm(&self) -> f32 {
        self.spatial_species_mean(
            IntracellularSpecies::ATP,
            IntracellularSpatialField::NucleoidOccupancy,
        )
    }

    pub(super) fn chromosome_domain_localized_supply(&self, domain_index: u32) -> f32 {
        Self::finite_scale(
            0.42 * self.localized_supply_scale()
                + 0.34
                    * Self::saturating_signal(
                        self.chromosome_domain_species_mean(
                            IntracellularSpecies::ATP,
                            domain_index,
                        ),
                        0.90,
                    )
                + 0.24
                    * Self::saturating_signal(
                        self.chromosome_domain_species_mean(
                            IntracellularSpecies::Nucleotides,
                            domain_index,
                        ),
                        0.55,
                    ),
            1.0,
            0.55,
            1.25,
        )
    }

    pub(super) fn chromosome_domain_energy_support(&self, domain_index: u32) -> f32 {
        Self::finite_scale(
            0.54 * Self::saturating_signal(
                self.chromosome_domain_species_mean(IntracellularSpecies::ATP, domain_index),
                0.90,
            ) + 0.26 * self.organism_expression.energy_support
                + 0.20 * self.chemistry_report.atp_support,
            1.0,
            0.50,
            1.75,
        )
    }

    pub(super) fn chromosome_domain_nucleotide_support(&self, domain_index: u32) -> f32 {
        Self::finite_scale(
            0.56 * Self::saturating_signal(
                self.chromosome_domain_species_mean(
                    IntracellularSpecies::Nucleotides,
                    domain_index,
                ),
                0.55,
            ) + 0.24 * self.organism_expression.nucleotide_support
                + 0.20 * self.chemistry_report.nucleotide_support,
            1.0,
            0.50,
            1.75,
        )
    }

    pub(super) fn localized_membrane_precursor_pool_mm(&self) -> f32 {
        0.55 * self.spatial_species_mean(
            IntracellularSpecies::MembranePrecursors,
            IntracellularSpatialField::MembraneAdjacency,
        ) + 0.45
            * self.spatial_species_mean(
                IntracellularSpecies::MembranePrecursors,
                IntracellularSpatialField::SeptumZone,
            )
    }

    pub(super) fn localized_membrane_atp_pool_mm(&self) -> f32 {
        0.65 * self.spatial_species_mean(
            IntracellularSpecies::ATP,
            IntracellularSpatialField::MembraneAdjacency,
        ) + 0.35
            * self.spatial_species_mean(
                IntracellularSpecies::ATP,
                IntracellularSpatialField::SeptumZone,
            )
    }

    pub(super) fn localized_membrane_band_precursor_pool_mm(&self) -> f32 {
        self.spatial_species_mean(
            IntracellularSpecies::MembranePrecursors,
            IntracellularSpatialField::MembraneBandZone,
        )
    }

    pub(super) fn localized_polar_precursor_pool_mm(&self) -> f32 {
        self.spatial_species_mean(
            IntracellularSpecies::MembranePrecursors,
            IntracellularSpatialField::PoleZone,
        )
    }
}
