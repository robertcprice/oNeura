// Explicit microbe management methods for TerrariumWorld.
//
// Extracted from terrarium_world.rs for modularity.

use super::*;

impl super::TerrariumWorld {
    // ── Block 1: Infrastructure methods ──────────────────────────────────

    pub(super) fn rebuild_explicit_microbe_fields(&mut self) {
        self.explicit_microbe_authority.fill(0.0);
        self.explicit_microbe_activity.fill(0.0);
        for cohort in &self.explicit_microbes {
            let chemistry = cohort.last_snapshot.local_chemistry.unwrap_or_default();
            let genotype_boost = 0.88
                + cohort.identity.gene_catabolic * 0.10
                + cohort.identity.gene_extracellular_scavenging * 0.08
                - cohort.identity.gene_stress_response * 0.05;
            let authority = clamp(
                (cohort.represented_cells / EXPLICIT_MICROBE_COHORT_CELLS).sqrt()
                    * (0.85 + cohort.represented_packets * 0.08).clamp(0.85, 1.25)
                    * (0.40
                        + chemistry.translation_support * 0.35
                        + cohort.last_snapshot.atp_mm * 0.08)
                    * genotype_boost
                    * (0.92 + cohort.identity.catalog.local_bank_share * 0.10),
                0.0,
                0.95,
            );
            let activity = clamp(
                cohort.last_snapshot.atp_mm * 0.22
                    + cohort.last_snapshot.glucose_mm * 0.08
                    + cohort.last_snapshot.oxygen_mm * 0.08
                    + chemistry.translation_support * 0.35
                    + chemistry.atp_support * 0.18
                    + cohort.smoothed_energy * 0.16
                    - cohort.smoothed_stress * 0.10,
                0.0,
                1.25,
            ) * (cohort.represented_cells / EXPLICIT_MICROBE_COHORT_CELLS).max(0.25);
            deposit_2d(
                &mut self.explicit_microbe_authority,
                self.config.width,
                self.config.height,
                cohort.x,
                cohort.y,
                cohort.radius.max(1) + 1,
                authority,
            );
            deposit_2d(
                &mut self.explicit_microbe_activity,
                self.config.width,
                self.config.height,
                cohort.x,
                cohort.y,
                cohort.radius.max(1) + 1,
                activity,
            );
        }

        // Packet populations also contribute to ownership authority.
        // Packet authority is weaker than cohort authority (packets are
        // lightweight) and capped below the cohort maximum of 0.95.
        let width = self.config.width;
        for pop in &self.packet_populations {
            if pop.total_cells < 1.0 {
                continue;
            }
            let flat = idx2(width, pop.x, pop.y);
            let packet_authority = clamp(
                (pop.total_cells / 100.0).sqrt() * pop.mean_activity() * 0.5,
                0.0,
                0.60,
            );
            let existing = self.explicit_microbe_authority[flat];
            self.explicit_microbe_authority[flat] =
                clamp(existing + packet_authority, 0.0, 0.95);
            // Also contribute to activity field
            let existing_activity = self.explicit_microbe_activity[flat];
            self.explicit_microbe_activity[flat] = clamp(
                existing_activity
                    + pop.mean_activity() * (pop.total_cells / 200.0).min(1.0),
                0.0,
                2.0,
            );
        }

        for value in &mut self.explicit_microbe_authority {
            *value = value.clamp(0.0, 0.95);
        }
    }

    pub(crate) fn explicit_microbe_step_count(simulator: &WholeCellSimulator, eco_dt: f32) -> usize {
        ((((eco_dt * 1000.0) * EXPLICIT_MICROBE_TIME_COMPRESSION)
            / simulator.environment_dt_ms().max(1.0e-6))
        .ceil() as usize)
            .clamp(EXPLICIT_MICROBE_MIN_STEPS, EXPLICIT_MICROBE_MAX_STEPS)
    }

    pub(super) fn explicit_microbe_identity_at(&self, flat: usize) -> TerrariumExplicitMicrobeIdentity {
        let Some((bank_idx, represented_packets, entry)) =
            self.microbial_secondary.dominant_catalog_entry_at(flat)
        else {
            return TerrariumExplicitMicrobeIdentity::default();
        };
        TerrariumExplicitMicrobeIdentity {
            bank_idx,
            represented_packets,
            genes: entry.genes,
            record: entry.record,
            catalog: entry.catalog,
            gene_catabolic: decode_secondary_gene_module(
                &entry.genes,
                &MICROBIAL_GENE_CATABOLIC_WEIGHTS,
            ),
            gene_stress_response: decode_secondary_gene_module(
                &entry.genes,
                &MICROBIAL_GENE_STRESS_RESPONSE_WEIGHTS,
            ),
            gene_dormancy_maintenance: decode_secondary_gene_module(
                &entry.genes,
                &MICROBIAL_GENE_DORMANCY_MAINTENANCE_WEIGHTS,
            ),
            gene_extracellular_scavenging: decode_secondary_gene_module(
                &entry.genes,
                &MICROBIAL_GENE_EXTRACELLULAR_SCAVENGING_WEIGHTS,
            ),
        }
    }

    pub(super) fn explicit_microbe_growth_signal(
        cohort: &TerrariumExplicitMicrobe,
        after: &WholeCellSnapshot,
        chemistry: LocalChemistryReport,
        stress_load: f32,
    ) -> (f32, f32, f32) {
        let energy_state = clamp(
            after.atp_mm * 0.34
                + chemistry.atp_support * 0.24
                + chemistry.translation_support * 0.22
                + after.glucose_mm * 0.10
                + after.oxygen_mm * 0.10,
            0.0,
            2.0,
        );
        let stress_state = clamp(
            stress_load * 0.24
                + (1.0 - chemistry.crowding_penalty).max(0.0) * 1.35
                + (CRITICAL_OXYGEN_FOR_STRESS - after.oxygen_mm).max(0.0) * 0.80
                + (CRITICAL_GLUCOSE_FOR_STRESS - after.glucose_mm).max(0.0) * 0.60,
            0.0,
            2.0,
        );
        let growth_signal = clamp(
            energy_state * (0.42 + cohort.identity.gene_catabolic * 0.15)
                + chemistry.translation_support * 0.28
                + after.division_progress * 0.26
                + cohort.identity.gene_extracellular_scavenging * 0.12
                - stress_state * (0.42 + cohort.identity.gene_stress_response * 0.12)
                - cohort.identity.gene_dormancy_maintenance * 0.08,
            -0.9,
            1.4,
        );
        (energy_state, stress_state, growth_signal)
    }

    pub(super) fn recruit_explicit_microbes_from_soil(&mut self) -> Result<(), String> {
        if self.explicit_microbes.len() >= self.config.max_explicit_microbes {
            return Ok(());
        }
        let width = self.config.width;
        let height = self.config.height;
        let depth = self.config.depth.max(1);
        let mut candidates = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let flat = idx2(width, x, y);
                if self.explicit_microbe_authority[flat] >= EXPLICIT_OWNERSHIP_THRESHOLD {
                    continue;
                }
                let active_fraction = (1.0 - self.microbial_dormancy[flat]).clamp(0.05, 1.0);
                let vitality = self.microbial_vitality[flat].clamp(0.0, 1.25);
                let packet_density = self.microbial_packets[flat].max(0.0);
                let carbon_signal = self.root_exudates[flat] * 8.0
                    + self.litter_carbon[flat] * 5.0
                    + self.organic_matter[flat] * 2.5;
                let genotype_signal = self
                    .microbial_secondary
                    .dominant_catalog_entry_at(flat)
                    .map(|(_, packets, entry)| {
                        packets * 0.55
                            + decode_secondary_gene_module(
                                &entry.genes,
                                &MICROBIAL_GENE_CATABOLIC_WEIGHTS,
                            ) * 0.80
                            + decode_secondary_gene_module(
                                &entry.genes,
                                &MICROBIAL_GENE_EXTRACELLULAR_SCAVENGING_WEIGHTS,
                            ) * 0.55
                            + entry.catalog.local_bank_share * 0.65
                    })
                    .unwrap_or(0.0);
                let score = self.microbial_cells[flat].max(0.0)
                    * active_fraction
                    * (0.35 + vitality)
                    * (0.40 + packet_density * 0.22)
                    + carbon_signal
                    + genotype_signal;
                if score >= EXPLICIT_MICROBE_RECRUITMENT_MIN_SCORE {
                    candidates.push((score, x, y));
                }
            }
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut recruited = 0usize;
        for (score, x, y) in candidates {
            if self.explicit_microbes.len() >= self.config.max_explicit_microbes {
                break;
            }
            if self.explicit_microbes.iter().any(|cohort| {
                cohort.x.abs_diff(x) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
                    && cohort.y.abs_diff(y) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
            }) {
                continue;
            }
            let flat = idx2(width, x, y);
            let represented_cells = clamp(
                self.microbial_cells[flat] * (0.85 + self.microbial_packets[flat] * 0.08)
                    + score * 1.5,
                EXPLICIT_MICROBE_COHORT_CELLS * 0.50,
                EXPLICIT_MICROBE_COHORT_CELLS * 3.0,
            );
            self.add_explicit_microbe(x, y, 1.min(depth - 1), represented_cells)?;
            recruited += 1;
        }
        if recruited > 0 {
            self.rebuild_explicit_microbe_fields();
        }
        Ok(())
    }

    pub(super) fn explicit_microbe_environment_inputs(
        &self,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
    ) -> WholeCellEnvironmentInputs {
        let material_inputs = self
            .explicit_microbe_material_inventory(x, y, z, radius)
            .estimate_whole_cell_environment_inputs(&[
                MaterialRegionKind::PoreWater,
                MaterialRegionKind::GasPhase,
                MaterialRegionKind::MineralSurface,
                MaterialRegionKind::BiofilmMatrix,
            ]);
        let depth = self.config.depth.max(1);
        let z = z.min(depth - 1);
        let flat = idx2(self.config.width, x, y);
        let soil_oxygen =
            self.substrate
                .patch_mean_species(TerrariumSpecies::OxygenGas, x, y, z, radius);
        let soil_co2 =
            self.substrate
                .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, radius);
        let soil_proton =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Proton, x, y, z, radius);
        let explicit_authority = self.explicit_microbe_authority[flat].clamp(0.0, 0.95);
        let coarse_pool_factor = if explicit_authority >= EXPLICIT_OWNERSHIP_THRESHOLD {
            0.0
        } else {
            1.0 - explicit_authority
        };
        let water_signal = (self.moisture[flat] * (1.0 - z as f32 * 0.18)
            + self.deep_moisture[flat] * 0.45)
            .clamp(0.0, 1.0);
        let stress_load = clamp(
            0.82 + (0.34 - water_signal).max(0.0) * 1.10
                + soil_proton * 24.0
                + soil_co2 * 9.0
                + (0.22 - soil_oxygen).max(0.0) * 2.0,
            0.25,
            3.8,
        );

        WholeCellEnvironmentInputs {
            glucose_mm: material_inputs.glucose_mm,
            oxygen_mm: material_inputs.oxygen_mm,
            amino_acids_mm: material_inputs.amino_acids_mm,
            nucleotides_mm: material_inputs.nucleotides_mm,
            membrane_precursors_mm: material_inputs.membrane_precursors_mm,
            metabolic_load: clamp(
                material_inputs.metabolic_load * 0.35
                    + stress_load * 0.65
                    + soil_proton * 6.0
                    + soil_co2 * 1.5
                    + (1.0 - coarse_pool_factor) * 0.08,
                0.25,
                3.8,
            ),
        }
    }

    pub(super) fn explicit_microbe_material_inventory(
        &self,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
    ) -> RegionalMaterialInventory {
        if let Some(cohort) = self.explicit_microbes.iter().find(|cohort| {
            cohort.x == x && cohort.y == y && cohort.z == z && cohort.radius == radius
        }) {
            return cohort.material_inventory.clone();
        }
        self.explicit_microbe_material_inventory_target_from_patch(x, y, z, radius)
    }

    pub(super) fn explicit_microbe_material_inventory_target_from_patch(
        &self,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
    ) -> RegionalMaterialInventory {
        let depth = self.config.depth.max(1);
        let z = z.min(depth - 1);
        let flat = idx2(self.config.width, x, y);
        let soil_glucose =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, radius);
        let soil_oxygen =
            self.substrate
                .patch_mean_species(TerrariumSpecies::OxygenGas, x, y, z, radius);
        let soil_ammonium =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Ammonium, x, y, z, radius);
        let soil_nitrate =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Nitrate, x, y, z, radius);
        let soil_phosphorus =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Phosphorus, x, y, z, radius);
        let soil_atp_flux =
            self.substrate
                .patch_mean_species(TerrariumSpecies::AtpFlux, x, y, z, radius);
        let atmos_o2 = self.sample_odorant_patch(ATMOS_O2_IDX, x, y, z, radius);
        let explicit_authority = self.explicit_microbe_authority[flat].clamp(0.0, 0.95);
        let coarse_pool_factor = if explicit_authority >= EXPLICIT_OWNERSHIP_THRESHOLD {
            0.0
        } else {
            1.0 - explicit_authority
        };

        let mut inventory =
            RegionalMaterialInventory::new(format!("explicit_microbe_patch::{x}::{y}::{z}"));
        let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
        let gas = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas);
        let interfacial = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Interfacial);
        let amorphous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Amorphous);

        let glucose_moles = f64::from(
            soil_glucose * 18.0
                + coarse_pool_factor * self.root_exudates[flat] * 9.0
                + coarse_pool_factor * self.litter_carbon[flat] * 4.0,
        );
        if glucose_moles > 1.0e-9 {
            inventory.add_component(
                MaterialRegionKind::PoreWater,
                MoleculeGraph::representative_glucose(),
                glucose_moles,
                aqueous.clone(),
            );
        }

        let oxygen_moles = f64::from(soil_oxygen * 8.0 + atmos_o2 * 1.8);
        if oxygen_moles > 1.0e-9 {
            inventory.add_component(
                MaterialRegionKind::GasPhase,
                MoleculeGraph::representative_oxygen_gas(),
                oxygen_moles,
                gas,
            );
        }

        let amino_moles = f64::from(
            soil_ammonium * 10.0
                + soil_nitrate * 3.5
                + coarse_pool_factor * self.dissolved_nutrients[flat] * 5.0,
        );
        if amino_moles > 1.0e-9 {
            inventory.add_component(
                MaterialRegionKind::PoreWater,
                MoleculeGraph::representative_amino_acid_pool(),
                amino_moles,
                aqueous.clone(),
            );
        }

        let nucleotide_moles = f64::from(
            soil_nitrate * 7.0
                + soil_phosphorus * 8.0
                + coarse_pool_factor * self.shallow_nutrients[flat] * 3.5,
        );
        if nucleotide_moles > 1.0e-9 {
            inventory.add_component(
                MaterialRegionKind::PoreWater,
                MoleculeGraph::representative_nucleotide_pool(),
                nucleotide_moles,
                aqueous,
            );
        }

        let atp_moles = f64::from(soil_atp_flux * 160.0);
        if atp_moles > 1.0e-9 {
            inventory.add_component(
                MaterialRegionKind::BiofilmMatrix,
                MoleculeGraph::representative_atp(),
                atp_moles,
                MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous),
            );
        }

        let membrane_moles = f64::from(
            coarse_pool_factor * self.root_exudates[flat] * 6.0
                + coarse_pool_factor * self.organic_matter[flat] * 3.0
                + soil_atp_flux * 180.0,
        );
        if membrane_moles > 1.0e-9 {
            inventory.add_component(
                MaterialRegionKind::BiofilmMatrix,
                MoleculeGraph::representative_membrane_precursor_pool(),
                membrane_moles * 0.55,
                amorphous,
            );
            inventory.add_component(
                MaterialRegionKind::MineralSurface,
                MoleculeGraph::representative_membrane_precursor_pool(),
                membrane_moles * 0.45,
                interfacial,
            );
        }

        inventory
    }

    pub(super) fn explicit_microbe_material_exchange_fraction(authority: f32) -> f64 {
        if authority >= EXPLICIT_OWNERSHIP_THRESHOLD {
            0.12
        } else {
            f64::from((0.42 - authority * 0.18).clamp(0.18, 0.42))
        }
    }

    pub(super) fn normalize_component_shares(potentials: &[f64], fallback: &[f64]) -> Vec<f64> {
        let total = potentials
            .iter()
            .copied()
            .filter(|value| *value > f64::EPSILON)
            .sum::<f64>();
        if total > f64::EPSILON {
            return potentials
                .iter()
                .map(|value| {
                    if *value > f64::EPSILON {
                        *value / total
                    } else {
                        0.0
                    }
                })
                .collect();
        }

        let fallback_total = fallback
            .iter()
            .copied()
            .filter(|value| *value > f64::EPSILON)
            .sum::<f64>();
        if fallback_total > f64::EPSILON {
            fallback
                .iter()
                .map(|value| {
                    if *value > f64::EPSILON {
                        *value / fallback_total
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            vec![0.0; potentials.len().max(fallback.len())]
        }
    }

    pub(super) fn reserve_band(
        target_amount: f64,
        lower_fraction: f64,
        upper_fraction: f64,
        absolute_reserve: f64,
        absolute_band_width: f64,
    ) -> (f64, f64) {
        let lower = (target_amount * lower_fraction).max(absolute_reserve);
        let upper = (target_amount * upper_fraction).max(lower + absolute_band_width);
        (lower, upper)
    }

    pub(super) fn sync_owned_component_single_species(
        &mut self,
        idx: usize,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        region: MaterialRegionKind,
        molecule: MoleculeGraph,
        phase: MaterialPhaseDescriptor,
        target_amount: f64,
        relaxation: f64,
        species: TerrariumSpecies,
        factor: f32,
    ) -> Result<(), String> {
        let selector = MaterialPhaseSelector::Exact(phase.clone());
        let current_amount = self.explicit_microbes[idx]
            .material_inventory
            .total_amount_for_component(region, &molecule, &selector);
        let (lower, upper) = Self::reserve_band(target_amount, 0.55, 1.35, 0.02, 0.03);
        let desired_delta = if current_amount < lower {
            (lower - current_amount) * relaxation
        } else if current_amount > upper {
            (upper - current_amount) * relaxation
        } else {
            0.0
        };
        if desired_delta.abs() <= 1.0e-12 {
            return Ok(());
        }

        let actual_component_delta = if desired_delta > 0.0 {
            let extracted = self.substrate.extract_patch_species(
                species,
                x,
                y,
                z,
                radius,
                (desired_delta / f64::from(factor)) as f32,
            );
            f64::from(extracted) * f64::from(factor)
        } else {
            let deposited = self.substrate.deposit_patch_species(
                species,
                x,
                y,
                z,
                radius,
                (-desired_delta / f64::from(factor)) as f32,
            );
            -(f64::from(deposited) * f64::from(factor))
        };

        self.explicit_microbes[idx]
            .material_inventory
            .set_component_amount(
                region,
                molecule,
                phase,
                (current_amount + actual_component_delta).max(0.0),
            )
            .map_err(|err| err.to_string())
    }

    pub(super) fn sync_owned_component_amino_pool(
        &mut self,
        idx: usize,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        target_amount: f64,
        relaxation: f64,
        soil_ammonium: f32,
        soil_nitrate: f32,
    ) -> Result<(), String> {
        let molecule = MoleculeGraph::representative_amino_acid_pool();
        let phase = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
        let selector = MaterialPhaseSelector::Exact(phase.clone());
        let current_amount = self.explicit_microbes[idx]
            .material_inventory
            .total_amount_for_component(MaterialRegionKind::PoreWater, &molecule, &selector);
        let (lower, upper) = Self::reserve_band(target_amount, 0.55, 1.35, 0.015, 0.025);
        let desired_delta = if current_amount < lower {
            (lower - current_amount) * relaxation
        } else if current_amount > upper {
            (upper - current_amount) * relaxation
        } else {
            0.0
        };
        if desired_delta.abs() <= 1.0e-12 {
            return Ok(());
        }

        let shares = Self::normalize_component_shares(
            &[
                f64::from(soil_ammonium) * 10.0,
                f64::from(soil_nitrate) * 3.5,
            ],
            &[10.0, 3.5],
        );
        let actual_component_delta = if desired_delta > 0.0 {
            let ammonium = self.substrate.extract_patch_species(
                TerrariumSpecies::Ammonium,
                x,
                y,
                z,
                radius,
                (desired_delta * shares[0] / 10.0) as f32,
            );
            let nitrate = self.substrate.extract_patch_species(
                TerrariumSpecies::Nitrate,
                x,
                y,
                z,
                radius,
                (desired_delta * shares[1] / 3.5) as f32,
            );
            f64::from(ammonium) * 10.0 + f64::from(nitrate) * 3.5
        } else {
            let ammonium = self.substrate.deposit_patch_species(
                TerrariumSpecies::Ammonium,
                x,
                y,
                z,
                radius,
                ((-desired_delta) * shares[0] / 10.0) as f32,
            );
            let nitrate = self.substrate.deposit_patch_species(
                TerrariumSpecies::Nitrate,
                x,
                y,
                z,
                radius,
                ((-desired_delta) * shares[1] / 3.5) as f32,
            );
            -(f64::from(ammonium) * 10.0 + f64::from(nitrate) * 3.5)
        };

        self.explicit_microbes[idx]
            .material_inventory
            .set_component_amount(
                MaterialRegionKind::PoreWater,
                molecule,
                phase,
                (current_amount + actual_component_delta).max(0.0),
            )
            .map_err(|err| err.to_string())
    }

    pub(super) fn sync_owned_component_nucleotide_pool(
        &mut self,
        idx: usize,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        target_amount: f64,
        relaxation: f64,
        soil_nitrate: f32,
        soil_phosphorus: f32,
    ) -> Result<(), String> {
        let molecule = MoleculeGraph::representative_nucleotide_pool();
        let phase = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
        let selector = MaterialPhaseSelector::Exact(phase.clone());
        let current_amount = self.explicit_microbes[idx]
            .material_inventory
            .total_amount_for_component(MaterialRegionKind::PoreWater, &molecule, &selector);
        let (lower, upper) = Self::reserve_band(target_amount, 0.55, 1.35, 0.012, 0.02);
        let desired_delta = if current_amount < lower {
            (lower - current_amount) * relaxation
        } else if current_amount > upper {
            (upper - current_amount) * relaxation
        } else {
            0.0
        };
        if desired_delta.abs() <= 1.0e-12 {
            return Ok(());
        }

        let shares = Self::normalize_component_shares(
            &[
                f64::from(soil_nitrate) * 7.0,
                f64::from(soil_phosphorus) * 8.0,
            ],
            &[7.0, 8.0],
        );
        let actual_component_delta = if desired_delta > 0.0 {
            let nitrate = self.substrate.extract_patch_species(
                TerrariumSpecies::Nitrate,
                x,
                y,
                z,
                radius,
                (desired_delta * shares[0] / 7.0) as f32,
            );
            let phosphorus = self.substrate.extract_patch_species(
                TerrariumSpecies::Phosphorus,
                x,
                y,
                z,
                radius,
                (desired_delta * shares[1] / 8.0) as f32,
            );
            f64::from(nitrate) * 7.0 + f64::from(phosphorus) * 8.0
        } else {
            let nitrate = self.substrate.deposit_patch_species(
                TerrariumSpecies::Nitrate,
                x,
                y,
                z,
                radius,
                ((-desired_delta) * shares[0] / 7.0) as f32,
            );
            let phosphorus = self.substrate.deposit_patch_species(
                TerrariumSpecies::Phosphorus,
                x,
                y,
                z,
                radius,
                ((-desired_delta) * shares[1] / 8.0) as f32,
            );
            -(f64::from(nitrate) * 7.0 + f64::from(phosphorus) * 8.0)
        };

        self.explicit_microbes[idx]
            .material_inventory
            .set_component_amount(
                MaterialRegionKind::PoreWater,
                molecule,
                phase,
                (current_amount + actual_component_delta).max(0.0),
            )
            .map_err(|err| err.to_string())
    }

    pub(super) fn sync_owned_component_oxygen_pool(
        &mut self,
        idx: usize,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        target_amount: f64,
        relaxation: f64,
        soil_oxygen: f32,
        atmos_o2: f32,
    ) -> Result<(), String> {
        let molecule = MoleculeGraph::representative_oxygen_gas();
        let phase = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas);
        let selector = MaterialPhaseSelector::Exact(phase.clone());
        let current_amount = self.explicit_microbes[idx]
            .material_inventory
            .total_amount_for_component(MaterialRegionKind::GasPhase, &molecule, &selector);
        let (lower, upper) = Self::reserve_band(target_amount, 0.45, 1.30, 0.01, 0.02);
        let desired_delta = if current_amount < lower {
            (lower - current_amount) * relaxation
        } else if current_amount > upper {
            (upper - current_amount) * relaxation
        } else {
            0.0
        };
        if desired_delta.abs() <= 1.0e-12 {
            return Ok(());
        }

        let shares = Self::normalize_component_shares(
            &[f64::from(soil_oxygen) * 8.0, f64::from(atmos_o2) * 1.8],
            &[8.0, 1.8],
        );
        let actual_component_delta = if desired_delta > 0.0 {
            let soil = self.substrate.extract_patch_species(
                TerrariumSpecies::OxygenGas,
                x,
                y,
                z,
                radius,
                (desired_delta * shares[0] / 8.0) as f32,
            );
            let atmos = -self.exchange_atmosphere_odorant_actual(
                ATMOS_O2_IDX,
                x,
                y,
                z,
                radius,
                -((desired_delta * shares[1] / 1.8) as f32),
            );
            f64::from(soil) * 8.0 + f64::from(atmos.max(0.0)) * 1.8
        } else {
            let soil = self.substrate.deposit_patch_species(
                TerrariumSpecies::OxygenGas,
                x,
                y,
                z,
                radius,
                ((-desired_delta) * shares[0] / 8.0) as f32,
            );
            let atmos = self.exchange_atmosphere_odorant_actual(
                ATMOS_O2_IDX,
                x,
                y,
                z,
                radius,
                ((-desired_delta) * shares[1] / 1.8) as f32,
            );
            -(f64::from(soil) * 8.0 + f64::from(atmos.max(0.0)) * 1.8)
        };

        self.explicit_microbes[idx]
            .material_inventory
            .set_component_amount(
                MaterialRegionKind::GasPhase,
                molecule,
                phase,
                (current_amount + actual_component_delta).max(0.0),
            )
            .map_err(|err| err.to_string())
    }

    pub(super) fn sync_owned_component_membrane_pool(
        &mut self,
        idx: usize,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        target_biofilm: f64,
        target_surface: f64,
        relaxation: f64,
    ) -> Result<(), String> {
        let molecule = MoleculeGraph::representative_membrane_precursor_pool();
        let amorphous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Amorphous);
        let interfacial = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Interfacial);
        let current_biofilm = self.explicit_microbes[idx]
            .material_inventory
            .total_amount_for_component(
                MaterialRegionKind::BiofilmMatrix,
                &molecule,
                &MaterialPhaseSelector::Exact(amorphous.clone()),
            );
        let current_surface = self.explicit_microbes[idx]
            .material_inventory
            .total_amount_for_component(
                MaterialRegionKind::MineralSurface,
                &molecule,
                &MaterialPhaseSelector::Exact(interfacial.clone()),
            );
        let current_total = current_biofilm + current_surface;
        let target_total = target_biofilm + target_surface;
        let (lower, upper) = Self::reserve_band(target_total, 0.60, 1.40, 0.02, 0.03);
        let desired_delta = if current_total < lower {
            (lower - current_total) * relaxation
        } else if current_total > upper {
            (upper - current_total) * relaxation
        } else {
            0.0
        };
        if desired_delta.abs() <= 1.0e-12 {
            return Ok(());
        }

        let actual_component_delta = if desired_delta > 0.0 {
            let atp_molecule = MoleculeGraph::representative_atp();
            let atp_selector = MaterialPhaseSelector::Kind(MaterialPhaseKind::Aqueous);
            let local_atp = self.explicit_microbes[idx]
                .material_inventory
                .total_amount_for_component(
                    MaterialRegionKind::BiofilmMatrix,
                    &atp_molecule,
                    &atp_selector,
                );
            let internal_support = local_atp.min(desired_delta);
            if internal_support > f64::EPSILON {
                let _ = self.explicit_microbes[idx]
                    .material_inventory
                    .remove_component_amount(
                        MaterialRegionKind::BiofilmMatrix,
                        &atp_molecule,
                        &atp_selector,
                        internal_support,
                    );
            }
            let remaining = (desired_delta - internal_support).max(0.0);
            let extracted = if remaining > f64::EPSILON {
                self.substrate.extract_patch_species(
                    TerrariumSpecies::AtpFlux,
                    x,
                    y,
                    z,
                    radius,
                    (remaining / 180.0) as f32,
                )
            } else {
                0.0
            };
            internal_support + f64::from(extracted) * 180.0
        } else {
            let deposited = self.substrate.deposit_patch_species(
                TerrariumSpecies::AtpFlux,
                x,
                y,
                z,
                radius,
                ((-desired_delta) / 180.0) as f32,
            );
            -(f64::from(deposited) * 180.0)
        };

        let new_total = (current_total + actual_component_delta).max(0.0);
        let biofilm_ratio = if target_total > f64::EPSILON {
            (target_biofilm / target_total).clamp(0.0, 1.0)
        } else {
            0.55
        };
        self.explicit_microbes[idx]
            .material_inventory
            .set_component_amount(
                MaterialRegionKind::BiofilmMatrix,
                molecule.clone(),
                amorphous,
                new_total * biofilm_ratio,
            )
            .map_err(|err| err.to_string())?;
        self.explicit_microbes[idx]
            .material_inventory
            .set_component_amount(
                MaterialRegionKind::MineralSurface,
                molecule,
                interfacial,
                new_total * (1.0 - biofilm_ratio),
            )
            .map_err(|err| err.to_string())
    }

    pub(super) fn sync_owned_component_atp_pool(
        &mut self,
        idx: usize,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        target_amount: f64,
        relaxation: f64,
    ) -> Result<(), String> {
        self.sync_owned_component_single_species(
            idx,
            x,
            y,
            z,
            radius,
            MaterialRegionKind::BiofilmMatrix,
            MoleculeGraph::representative_atp(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous),
            target_amount,
            relaxation,
            TerrariumSpecies::AtpFlux,
            160.0,
        )
    }

    pub(super) fn spill_owned_component_carbon_dioxide_pool(
        &mut self,
        idx: usize,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        relaxation: f64,
        soil_carbon_dioxide: f32,
        atmos_carbon_dioxide: f32,
    ) -> Result<(), String> {
        let molecule = MoleculeGraph::representative_carbon_dioxide();
        let phase = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas);
        let selector = MaterialPhaseSelector::Exact(phase.clone());
        let current_amount = self.explicit_microbes[idx]
            .material_inventory
            .total_amount_for_component(MaterialRegionKind::GasPhase, &molecule, &selector);
        let retention_target = (f64::from(soil_carbon_dioxide) * 0.18
            + f64::from(atmos_carbon_dioxide) * 0.10)
            .max(0.01);
        if current_amount <= retention_target + 1.0e-12 {
            return Ok(());
        }

        let desired_release = (current_amount - retention_target) * relaxation;
        if desired_release <= 1.0e-12 {
            return Ok(());
        }

        let shares = Self::normalize_component_shares(
            &[
                f64::from(soil_carbon_dioxide) * 1.0,
                f64::from(atmos_carbon_dioxide) * 0.65,
            ],
            &[0.65, 0.35],
        );
        let to_soil = desired_release * shares[0];
        let to_atmos = desired_release * shares[1];
        let soil_deposited = self.substrate.deposit_patch_species(
            TerrariumSpecies::CarbonDioxide,
            x,
            y,
            z,
            radius,
            to_soil as f32,
        );
        let atmos_deposited = self.exchange_atmosphere_odorant_actual(
            ATMOS_CO2_IDX,
            x,
            y,
            z,
            radius,
            to_atmos as f32,
        );
        let actual_release =
            f64::from(soil_deposited.max(0.0)) + f64::from(atmos_deposited.max(0.0));

        self.explicit_microbes[idx]
            .material_inventory
            .set_component_amount(
                MaterialRegionKind::GasPhase,
                molecule,
                phase,
                (current_amount - actual_release).max(0.0),
            )
            .map_err(|err| err.to_string())
    }

    pub(super) fn spill_owned_component_proton_pool(
        &mut self,
        idx: usize,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        relaxation: f64,
        soil_proton: f32,
    ) -> Result<(), String> {
        let molecule = MoleculeGraph::representative_proton_pool();
        let phase = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
        let selector = MaterialPhaseSelector::Exact(phase.clone());
        let current_amount = self.explicit_microbes[idx]
            .material_inventory
            .total_amount_for_component(MaterialRegionKind::PoreWater, &molecule, &selector);
        let retention_target = (f64::from(soil_proton) * 0.18).max(0.003);
        if current_amount <= retention_target + 1.0e-12 {
            return Ok(());
        }

        let desired_release = (current_amount - retention_target) * relaxation;
        if desired_release <= 1.0e-12 {
            return Ok(());
        }

        let deposited = self.substrate.deposit_patch_species(
            TerrariumSpecies::Proton,
            x,
            y,
            z,
            radius,
            desired_release as f32,
        );
        let actual_release = f64::from(deposited.max(0.0));
        self.explicit_microbes[idx]
            .material_inventory
            .set_component_amount(
                MaterialRegionKind::PoreWater,
                molecule,
                phase,
                (current_amount - actual_release).max(0.0),
            )
            .map_err(|err| err.to_string())
    }

    pub(super) fn sync_owned_explicit_microbe_material_inventory(
        &mut self,
        idx: usize,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        relaxation: f64,
    ) -> Result<(), String> {
        let depth = self.config.depth.max(1);
        let z = z.min(depth - 1);
        let target = self.explicit_microbe_material_inventory_target_from_patch(x, y, z, radius);
        let soil_oxygen =
            self.substrate
                .patch_mean_species(TerrariumSpecies::OxygenGas, x, y, z, radius);
        let soil_ammonium =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Ammonium, x, y, z, radius);
        let soil_nitrate =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Nitrate, x, y, z, radius);
        let soil_phosphorus =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Phosphorus, x, y, z, radius);
        let soil_carbon_dioxide =
            self.substrate
                .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, radius);
        let soil_proton =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Proton, x, y, z, radius);
        let atmos_o2 = self.sample_odorant_patch(ATMOS_O2_IDX, x, y, z, radius);
        let atmos_co2 = self.sample_odorant_patch(ATMOS_CO2_IDX, x, y, z, radius);

        let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
        let gas = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas);
        self.sync_owned_component_single_species(
            idx,
            x,
            y,
            z,
            radius,
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_glucose(),
            aqueous.clone(),
            target.total_amount_for_component(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_glucose(),
                &MaterialPhaseSelector::Exact(aqueous.clone()),
            ),
            relaxation,
            TerrariumSpecies::Glucose,
            18.0,
        )?;
        self.sync_owned_component_oxygen_pool(
            idx,
            x,
            y,
            z,
            radius,
            target.total_amount_for_component(
                MaterialRegionKind::GasPhase,
                &MoleculeGraph::representative_oxygen_gas(),
                &MaterialPhaseSelector::Exact(gas),
            ),
            relaxation,
            soil_oxygen,
            atmos_o2,
        )?;
        self.sync_owned_component_amino_pool(
            idx,
            x,
            y,
            z,
            radius,
            target.total_amount_for_component(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_amino_acid_pool(),
                &MaterialPhaseSelector::Exact(aqueous.clone()),
            ),
            relaxation,
            soil_ammonium,
            soil_nitrate,
        )?;
        self.sync_owned_component_nucleotide_pool(
            idx,
            x,
            y,
            z,
            radius,
            target.total_amount_for_component(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_nucleotide_pool(),
                &MaterialPhaseSelector::Exact(aqueous),
            ),
            relaxation,
            soil_nitrate,
            soil_phosphorus,
        )?;
        self.sync_owned_component_atp_pool(
            idx,
            x,
            y,
            z,
            radius,
            target.total_amount_for_component(
                MaterialRegionKind::BiofilmMatrix,
                &MoleculeGraph::representative_atp(),
                &MaterialPhaseSelector::Kind(MaterialPhaseKind::Aqueous),
            ),
            relaxation,
        )?;
        self.sync_owned_component_membrane_pool(
            idx,
            x,
            y,
            z,
            radius,
            target.total_amount_for_component(
                MaterialRegionKind::BiofilmMatrix,
                &MoleculeGraph::representative_membrane_precursor_pool(),
                &MaterialPhaseSelector::Kind(MaterialPhaseKind::Amorphous),
            ),
            target.total_amount_for_component(
                MaterialRegionKind::MineralSurface,
                &MoleculeGraph::representative_membrane_precursor_pool(),
                &MaterialPhaseSelector::Kind(MaterialPhaseKind::Interfacial),
            ),
            relaxation,
        )?;
        self.spill_owned_component_carbon_dioxide_pool(
            idx,
            x,
            y,
            z,
            radius,
            relaxation,
            soil_carbon_dioxide,
            atmos_co2,
        )?;
        self.spill_owned_component_proton_pool(idx, x, y, z, radius, relaxation, soil_proton)
    }

    pub(super) fn sync_explicit_microbe_material_inventory(&mut self, idx: usize) -> Result<(), String> {
        let (x, y, z, radius) = {
            let cohort = &self.explicit_microbes[idx];
            (cohort.x, cohort.y, cohort.z, cohort.radius)
        };
        let flat = idx2(self.config.width, x, y);
        let authority = self.explicit_microbe_authority[flat].clamp(0.0, 1.0);
        let target = self.explicit_microbe_material_inventory_target_from_patch(x, y, z, radius);
        let relaxation = Self::explicit_microbe_material_exchange_fraction(authority);
        if authority >= EXPLICIT_OWNERSHIP_THRESHOLD {
            if self.explicit_microbes[idx]
                .material_inventory
                .total_amount_moles()
                <= f64::EPSILON
            {
                self.explicit_microbes[idx].material_inventory = target;
            } else {
                self.sync_owned_explicit_microbe_material_inventory(
                    idx, x, y, z, radius, relaxation,
                )?;
            }
            return Ok(());
        }
        let cohort = &mut self.explicit_microbes[idx];
        if cohort.material_inventory.total_amount_moles() <= f64::EPSILON {
            cohort.material_inventory = target;
        } else {
            cohort
                .material_inventory
                .relax_toward(&target, relaxation)
                .map_err(|err| err.to_string())?;
        }
        Ok(())
    }

    pub(super) fn owned_inventory_available(
        &self,
        idx: usize,
    ) -> (f32, f32, f32, f32) {
        let cohort = &self.explicit_microbes[idx];
        let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
        let gas = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas);
        let glucose = cohort.material_inventory.total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_glucose(),
            &MaterialPhaseSelector::Exact(aqueous.clone()),
        ) as f32;
        let oxygen = cohort.material_inventory.total_amount_for_component(
            MaterialRegionKind::GasPhase,
            &MoleculeGraph::representative_oxygen_gas(),
            &MaterialPhaseSelector::Exact(gas),
        ) as f32;
        let amino = cohort.material_inventory.total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_amino_acid_pool(),
            &MaterialPhaseSelector::Exact(aqueous.clone()),
        ) as f32;
        let nucleo = cohort.material_inventory.total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_nucleotide_pool(),
            &MaterialPhaseSelector::Exact(aqueous),
        ) as f32;
        (glucose, oxygen, amino, nucleo)
    }

    pub(super) fn apply_explicit_microbe_material_fluxes(
        &mut self,
        idx: usize,
        glucose_draw: f32,
        oxygen_draw: f32,
        ammonium_draw: f32,
        nitrate_draw: f32,
        atp_flux_signal: f32,
        carbon_dioxide_release: f32,
        proton_release: f32,
    ) {
        let cohort = &mut self.explicit_microbes[idx];
        let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
        let gas = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas);
        let interfacial = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Interfacial);
        let amorphous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Amorphous);

        if glucose_draw > 0.0 {
            let _ = cohort.material_inventory.remove_component_amount(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_glucose(),
                &MaterialPhaseSelector::Exact(aqueous.clone()),
                f64::from(glucose_draw),
            );
        }
        if oxygen_draw > 0.0 {
            let _ = cohort.material_inventory.remove_component_amount(
                MaterialRegionKind::GasPhase,
                &MoleculeGraph::representative_oxygen_gas(),
                &MaterialPhaseSelector::Exact(gas.clone()),
                f64::from(oxygen_draw),
            );
        }
        if ammonium_draw > 0.0 {
            let _ = cohort.material_inventory.remove_component_amount(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_amino_acid_pool(),
                &MaterialPhaseSelector::Exact(aqueous.clone()),
                f64::from(ammonium_draw),
            );
        }
        if nitrate_draw > 0.0 {
            let _ = cohort.material_inventory.remove_component_amount(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_nucleotide_pool(),
                &MaterialPhaseSelector::Exact(aqueous.clone()),
                f64::from(nitrate_draw),
            );
        }
        let membrane_region_flux = f64::from((ammonium_draw + nitrate_draw) * 0.12);
        if membrane_region_flux > f64::EPSILON {
            let _ = cohort.material_inventory.set_component_amount(
                MaterialRegionKind::BiofilmMatrix,
                MoleculeGraph::representative_membrane_precursor_pool(),
                amorphous,
                cohort.material_inventory.total_amount_for_component(
                    MaterialRegionKind::BiofilmMatrix,
                    &MoleculeGraph::representative_membrane_precursor_pool(),
                    &MaterialPhaseSelector::Kind(MaterialPhaseKind::Amorphous),
                ) + membrane_region_flux * 0.55,
            );
            let _ = cohort.material_inventory.set_component_amount(
                MaterialRegionKind::MineralSurface,
                MoleculeGraph::representative_membrane_precursor_pool(),
                interfacial,
                cohort.material_inventory.total_amount_for_component(
                    MaterialRegionKind::MineralSurface,
                    &MoleculeGraph::representative_membrane_precursor_pool(),
                    &MaterialPhaseSelector::Kind(MaterialPhaseKind::Interfacial),
                ) + membrane_region_flux * 0.45,
            );
        }
        if atp_flux_signal > 0.0 {
            let _ = cohort.material_inventory.set_component_amount(
                MaterialRegionKind::BiofilmMatrix,
                MoleculeGraph::representative_atp(),
                aqueous.clone(),
                cohort.material_inventory.total_amount_for_component(
                    MaterialRegionKind::BiofilmMatrix,
                    &MoleculeGraph::representative_atp(),
                    &MaterialPhaseSelector::Kind(MaterialPhaseKind::Aqueous),
                ) + f64::from(atp_flux_signal),
            );
        }
        if carbon_dioxide_release > 0.0 {
            let _ = cohort.material_inventory.set_component_amount(
                MaterialRegionKind::GasPhase,
                MoleculeGraph::representative_carbon_dioxide(),
                gas.clone(),
                cohort.material_inventory.total_amount_for_component(
                    MaterialRegionKind::GasPhase,
                    &MoleculeGraph::representative_carbon_dioxide(),
                    &MaterialPhaseSelector::Kind(MaterialPhaseKind::Gas),
                ) + f64::from(carbon_dioxide_release),
            );
        }
        if proton_release > 0.0 {
            let _ = cohort.material_inventory.set_component_amount(
                MaterialRegionKind::PoreWater,
                MoleculeGraph::representative_proton_pool(),
                aqueous.clone(),
                cohort.material_inventory.total_amount_for_component(
                    MaterialRegionKind::PoreWater,
                    &MoleculeGraph::representative_proton_pool(),
                    &MaterialPhaseSelector::Kind(MaterialPhaseKind::Aqueous),
                ) + f64::from(proton_release),
            );
        }
    }

    // ── Block 2: Stepping methods ────────────────────────────────────────

    pub(super) fn compute_explicit_microbe_fluxes(
        k: &SubstrateKinetics,
        eco_dt: f32,
        uptake_scale: f32,
        soil_glucose: f32,
        soil_oxygen: f32,
        soil_ammonium: f32,
        soil_nitrate: f32,
        wcs_glucose_delta: f32,
        wcs_oxygen_delta: f32,
        wcs_ammonium_delta: f32,
        wcs_nitrate_delta: f32,
        wcs_atp_delta: f32,
        local_atp_flux: f32,
    ) -> (f32, f32, f32, f32, f32, f32, f32) {
        // Bridge ecological dt (seconds) to substrate kinetics timescale (ms).
        let effective_dt = eco_dt * 1000.0 * EXPLICIT_MICROBE_TIME_COMPRESSION;

        // WCS deltas as activity gates: how actively is the cell metabolizing?
        // Clamped to [0, 1] so the gate only modulates, never amplifies.
        let glucose_activity = wcs_glucose_delta.max(0.0).min(1.0);
        let _oxygen_activity = wcs_oxygen_delta.max(0.0).min(1.0);
        let ammonium_activity = wcs_ammonium_delta.max(0.0).min(1.0);
        let nitrate_activity = wcs_nitrate_delta.max(0.0).min(1.0);

        // ── Michaelis-Menten saturation terms (same as terrarium.rs:1013-1064) ──
        let glucose_sat = soil_glucose / (k.respiration_km_glucose + soil_glucose);
        let oxygen_sat = soil_oxygen / (k.respiration_km_oxygen + soil_oxygen);
        let ferm_glucose_sat = soil_glucose / (k.fermentation_km_glucose + soil_glucose);
        let nh4_sat = soil_ammonium / (k.nitrification_km_ammonium + soil_ammonium);
        let nit_o2_sat = soil_oxygen / (k.nitrification_km_oxygen + soil_oxygen);
        let no3_sat = soil_nitrate / (k.denitrification_km_nitrate + soil_nitrate);
        let anoxic_gate = (1.0 - oxygen_sat).max(0.0);

        // ── Aerobic Respiration (dual M-M on glucose + O2) ──
        // C6H12O6 + 6O2 → 6CO2 + 6H2O + ~30 ATP
        let respiration_flux = (k.respiration_vmax
            * glucose_sat
            * oxygen_sat
            * effective_dt
            * uptake_scale
            * glucose_activity)
            .min(soil_glucose)
            .min(soil_oxygen * 1.2);

        // ── Fermentation (M-M on glucose, anoxic gate) ──
        // C6H12O6 → 2 Ethanol + 2 CO2 + 2 ATP
        let fermentation_flux = (k.fermentation_vmax
            * ferm_glucose_sat
            * anoxic_gate
            * effective_dt
            * uptake_scale
            * glucose_activity)
            .min(soil_glucose - respiration_flux)
            .max(0.0);

        // ── Nitrification (dual M-M on NH4 + O2) ──
        // NH4+ + 2O2 → NO3- + 2H+ + H2O + ~0.5 ATP
        let nitrification_flux = (k.nitrification_vmax
            * nh4_sat
            * nit_o2_sat
            * effective_dt
            * uptake_scale
            * ammonium_activity)
            .min(soil_ammonium)
            .min(soil_oxygen * 1.5);

        // ── Denitrification (M-M on NO3, anoxic gate) ──
        // NO3- → N2 + ... (under anoxic conditions)
        let denitrification_flux = (k.denitrification_vmax
            * no3_sat
            * anoxic_gate
            * effective_dt
            * uptake_scale
            * nitrate_activity)
            .min(soil_nitrate);

        // ── Aggregate substrate draws ──
        let glucose_draw = clamp(respiration_flux + fermentation_flux, 0.0, 0.0035);
        let oxygen_draw = clamp(
            respiration_flux + nitrification_flux,
            0.0,
            0.0032,
        );
        let ammonium_draw = clamp(nitrification_flux, 0.0, 0.0020);
        let nitrate_draw = clamp(denitrification_flux, 0.0, 0.0018);

        // ── ATP flux signal ──
        let atp_flux = clamp(
            respiration_flux * k.respiration_atp_yield
                + fermentation_flux * k.fermentation_atp_yield
                + nitrification_flux * k.nitrification_atp_yield
                + wcs_atp_delta.abs() * 0.00004 * uptake_scale
                + local_atp_flux * 0.22 * 0.00004 * uptake_scale,
            0.0,
            0.0018,
        );

        // ── CO2 release (stoichiometric, matching substrate PDE) ──
        let co2_release = clamp(
            respiration_flux * STOICH_RESPIRATION_CO2_PER_GLUCOSE
                + fermentation_flux * STOICH_FERMENTATION_CO2_PER_GLUCOSE,
            0.0,
            0.0040,
        );

        // ── Proton release (stoichiometric, matching substrate PDE) ──
        let proton_release = clamp(
            co2_release * CO2_PROTON_FRACTION_AT_SOIL_PH
                + nitrification_flux * STOICH_NITRIFICATION_PROTON_YIELD,
            0.0,
            0.0012,
        );

        (
            glucose_draw,
            oxygen_draw,
            ammonium_draw,
            nitrate_draw,
            atp_flux,
            co2_release,
            proton_release,
        )
    }

    /// Shared flux computation, inventory capping, cohort state update,
    /// material inventory mutation, and substrate extraction for a single
    /// explicit microbe cohort step.  Returns (co2_flux, o2_flux, humidity_flux)
    /// for optional atmospheric exchange.
    pub(super) fn step_single_explicit_microbe(
        &mut self,
        idx: usize,
        eco_dt: f32,
        step_count: u64,
    ) -> Result<(usize, usize, usize, usize, f32, f32, f32), String> {
        self.sync_explicit_microbe_material_inventory(idx)?;
        let (x, y, z, radius, represented_cells, inputs) = {
            let cohort = &self.explicit_microbes[idx];
            (
                cohort.x,
                cohort.y,
                cohort.z,
                cohort.radius,
                cohort.represented_cells,
                self.explicit_microbe_environment_inputs(
                    cohort.x,
                    cohort.y,
                    cohort.z,
                    cohort.radius,
                ),
            )
        };
        let owned_authority = self.explicit_microbe_authority[idx2(self.config.width, x, y)]
            >= EXPLICIT_OWNERSHIP_THRESHOLD;

        let (before, after) = {
            let cohort = &mut self.explicit_microbes[idx];
            cohort.simulator.apply_environment_inputs(&inputs);
            let before = cohort.simulator.snapshot();
            cohort.simulator.run(step_count);
            let after = cohort.simulator.snapshot();
            cohort.last_snapshot = after.clone();
            (before, after)
        };

        // Sample local substrate concentrations for M-M saturation terms.
        let soil_glucose =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, radius);
        let soil_oxygen =
            self.substrate
                .patch_mean_species(TerrariumSpecies::OxygenGas, x, y, z, radius);
        let soil_ammonium =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Ammonium, x, y, z, radius);
        let soil_nitrate =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Nitrate, x, y, z, radius);

        let before_local = before.local_chemistry.unwrap_or_default();
        let after_local = after.local_chemistry.unwrap_or_default();
        let uptake_scale = represented_cells.max(1.0);

        // WCS deltas: how much did the whole-cell simulator consume each pool?
        let wcs_glucose_delta = (before.glucose_mm - after.glucose_mm).max(0.0)
            + (before_local.mean_glucose - after_local.mean_glucose).max(0.0) * 0.35;
        let wcs_oxygen_delta = (before.oxygen_mm - after.oxygen_mm).max(0.0)
            + (before_local.mean_oxygen - after_local.mean_oxygen).max(0.0) * 0.30;
        let wcs_ammonium_delta = (before.amino_acids_mm - after.amino_acids_mm).max(0.0);
        let wcs_nitrate_delta = (before.nucleotides_mm - after.nucleotides_mm).max(0.0);
        let wcs_atp_delta = after.atp_mm - before.atp_mm;

        let (mut glucose_draw, mut oxygen_draw, mut ammonium_draw, mut nitrate_draw,
             atp_flux_signal, carbon_dioxide_release, proton_release) =
            Self::compute_explicit_microbe_fluxes(
                &self.md_calibrator.kinetics,
                eco_dt,
                uptake_scale,
                soil_glucose,
                soil_oxygen,
                soil_ammonium,
                soil_nitrate,
                wcs_glucose_delta,
                wcs_oxygen_delta,
                wcs_ammonium_delta,
                wcs_nitrate_delta,
                wcs_atp_delta,
                after_local.mean_atp_flux,
            );

        // For owned cells, cap draws to what the inventory actually holds
        // so the cell can't consume more than its local material state contains.
        if owned_authority {
            let (avail_glucose, avail_oxygen, avail_amino, avail_nucleo) =
                self.owned_inventory_available(idx);
            glucose_draw = glucose_draw.min(avail_glucose);
            oxygen_draw = oxygen_draw.min(avail_oxygen);
            ammonium_draw = ammonium_draw.min(avail_amino);
            nitrate_draw = nitrate_draw.min(avail_nucleo);
        }

        let (energy_state, stress_state, growth_signal) = Self::explicit_microbe_growth_signal(
            &self.explicit_microbes[idx],
            &after,
            after_local,
            inputs.metabolic_load,
        );

        {
            let cohort = &mut self.explicit_microbes[idx];
            cohort.age_s += eco_dt.max(0.0);
            cohort.cumulative_glucose_draw += glucose_draw;
            cohort.cumulative_oxygen_draw += oxygen_draw;
            cohort.cumulative_ammonium_draw += ammonium_draw;
            cohort.cumulative_nitrate_draw += nitrate_draw;
            cohort.cumulative_co2_release += carbon_dioxide_release;
            cohort.cumulative_proton_release += proton_release;
            cohort.smoothed_energy = clamp(
                cohort.smoothed_energy * 0.84 + energy_state * 0.16,
                0.0,
                2.0,
            );
            cohort.smoothed_stress = clamp(
                cohort.smoothed_stress * 0.84 + stress_state * 0.16,
                0.0,
                2.0,
            );

            let growth_factor = if growth_signal >= 0.0 {
                1.0 + eco_dt * EXPLICIT_MICROBE_GROWTH_RATE * growth_signal
            } else {
                1.0 + eco_dt * EXPLICIT_MICROBE_DECAY_RATE * growth_signal
            };
            cohort.represented_cells = clamp(
                cohort.represented_cells * growth_factor,
                EXPLICIT_MICROBE_MIN_REPRESENTED_CELLS,
                EXPLICIT_MICROBE_MAX_REPRESENTED_CELLS,
            );
            cohort.represented_packets = clamp(
                cohort.represented_cells / MICROBIAL_PACKET_TARGET_CELLS,
                0.05,
                EXPLICIT_MICROBE_MAX_REPRESENTED_CELLS / MICROBIAL_PACKET_TARGET_CELLS,
            );
            cohort.radius =
                if cohort.represented_cells
                    >= EXPLICIT_MICROBE_COHORT_CELLS * EXPLICIT_MICROBE_RADIUS_EXPAND_2_CELLS
                    && cohort.smoothed_energy >= EXPLICIT_MICROBE_RADIUS_EXPAND_2_ENERGY
                {
                    EXPLICIT_MICROBE_PATCH_RADIUS + 2
                } else if cohort.represented_cells
                    >= EXPLICIT_MICROBE_COHORT_CELLS * EXPLICIT_MICROBE_RADIUS_EXPAND_1_CELLS
                    && cohort.smoothed_energy >= EXPLICIT_MICROBE_RADIUS_EXPAND_1_ENERGY
                {
                    EXPLICIT_MICROBE_PATCH_RADIUS + 1
                } else {
                    EXPLICIT_MICROBE_PATCH_RADIUS
                };
        }

        self.apply_explicit_microbe_material_fluxes(
            idx,
            glucose_draw,
            oxygen_draw,
            ammonium_draw,
            nitrate_draw,
            atp_flux_signal,
            carbon_dioxide_release,
            proton_release,
        );

        if !owned_authority && glucose_draw > 0.0 {
            let _ = self.substrate.extract_patch_species(
                TerrariumSpecies::Glucose, x, y, z, radius, glucose_draw,
            );
        }
        if !owned_authority && oxygen_draw > 0.0 {
            let _ = self.substrate.extract_patch_species(
                TerrariumSpecies::OxygenGas, x, y, z, radius, oxygen_draw,
            );
        }
        if !owned_authority && ammonium_draw > 0.0 {
            let _ = self.substrate.extract_patch_species(
                TerrariumSpecies::Ammonium, x, y, z, radius, ammonium_draw,
            );
        }
        if !owned_authority && nitrate_draw > 0.0 {
            let _ = self.substrate.extract_patch_species(
                TerrariumSpecies::Nitrate, x, y, z, radius, nitrate_draw,
            );
        }
        if !owned_authority && atp_flux_signal > 0.0 {
            let _ = self.substrate.deposit_patch_species(
                TerrariumSpecies::AtpFlux, x, y, z, radius, atp_flux_signal,
            );
        }
        if !owned_authority && carbon_dioxide_release > 0.0 {
            let _ = self.substrate.deposit_patch_species(
                TerrariumSpecies::CarbonDioxide, x, y, z, radius, carbon_dioxide_release,
            );
        }
        if !owned_authority && proton_release > 0.0 {
            let _ = self.substrate.deposit_patch_species(
                TerrariumSpecies::Proton, x, y, z, radius, proton_release,
            );
        }

        // Atmospheric exchange: Henry's Law gas-phase fractions with Fick's
        // conductance damping to avoid double-counting with couple_soil_atmosphere_gases.
        let fick_damping = (FICK_SURFACE_CONDUCTANCE * eco_dt).min(1.0);
        let co2_flux = if owned_authority {
            0.0
        } else {
            carbon_dioxide_release * CO2_GAS_PHASE_FRACTION * fick_damping
        };
        let o2_flux = if owned_authority {
            0.0
        } else {
            -oxygen_draw * O2_GAS_PHASE_FRACTION * fick_damping
        };
        let humidity_flux = carbon_dioxide_release * 0.05;
        Ok((x, y, z, radius.max(1), co2_flux, o2_flux, humidity_flux))
    }


    /// Bridge an explicit cell's phenotype into a coarse genotype packet.
    /// Called when a daughter cell from division can't be promoted to a full
    /// explicit cohort, or when an explicit cell dies.
    pub(super) fn bridge_explicit_to_coarse_packet(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        snapshot: &WholeCellSnapshot,
        identity: &TerrariumExplicitMicrobeIdentity,
        represented_cells: f32,
    ) {
        // Find or create the packet population at this (x,y) position.
        let pop_idx = if let Some(idx) = self
            .packet_populations
            .iter()
            .position(|pop| pop.x == x && pop.y == y)
        {
            idx
        } else if self.packet_populations.len() < GENOTYPE_PACKET_POPULATION_MAX_CELLS {
            let depth = self.config.depth.max(1);
            self.packet_populations
                .push(GenotypePacketPopulation::new(x, y, z.min(depth - 1)));
            self.packet_populations.len() - 1
        } else {
            return; // No room for a new population; silently discard.
        };

        // Extract phenotype axes from whole-cell snapshot.
        // Reserve: ATP production efficiency (axis 0, catabolic).
        let atp_efficiency = if snapshot.glucose_mm > 0.001 {
            (snapshot.atp_mm / snapshot.glucose_mm).clamp(0.0, 10.0) / 10.0
        } else {
            0.5
        };
        // Activity: healthy ATP level is ~2 mM.
        let activity = (snapshot.atp_mm / 2.0).clamp(0.0, 1.0);

        let packet = GenotypePacket {
            catalog_slot: identity.bank_idx as u32,
            genotype_id: identity.bank_idx as u32,
            lineage_id: 0,
            represented_cells,
            activity,
            dormancy: 0.0,
            reserve: atp_efficiency,
            damage: 0.0,
            cumulative_glucose_draw: 0.0,
            cumulative_oxygen_draw: 0.0,
            cumulative_co2_release: 0.0,
            cumulative_ammonium_draw: 0.0,
            cumulative_proton_release: 0.0,
        };
        let pop = &mut self.packet_populations[pop_idx];
        if pop.packets.len() < GENOTYPE_PACKET_MAX_PER_CELL {
            pop.total_cells += represented_cells;
            pop.packets.push(packet);
        }
    }

    pub(super) fn step_explicit_microbes(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.explicit_microbes.is_empty() {
            return Ok(());
        }

        let mut atmospheric_fluxes = Vec::with_capacity(self.explicit_microbes.len());
        for idx in 0..self.explicit_microbes.len() {
            let step_count = {
                let cohort = &self.explicit_microbes[idx];
                Self::explicit_microbe_step_count(&cohort.simulator, eco_dt)
            };
            let flux = self.step_single_explicit_microbe(idx, eco_dt, step_count as u64)?;
            atmospheric_fluxes.push(flux);
        }

        for (x, y, z, radius, co2_flux, o2_flux, humidity_flux) in atmospheric_fluxes {
            self.exchange_atmosphere_flux_bundle(x, y, z, radius, co2_flux, o2_flux, humidity_flux);
        }
        // Handle cell division: check which cohorts are ready to split.
        let mut division_indices: Vec<usize> = Vec::new();
        for (idx, cohort) in self.explicit_microbes.iter().enumerate() {
            if cohort.simulator.pending_division() {
                division_indices.push(idx);
            }
        }
        // Process divisions in reverse order to preserve indices.
        for &idx in division_indices.iter().rev() {
            let parent = &self.explicit_microbes[idx];
            let (daughter_a, daughter_b) = parent.simulator.split_into_daughters();
            let parent_x = parent.x;
            let parent_y = parent.y;
            let parent_z = parent.z;
            let parent_radius = parent.radius;
            let parent_represented_cells = parent.represented_cells;
            let parent_represented_packets = parent.represented_packets;
            let parent_identity = parent.identity;
            let parent_material_inventory = parent.material_inventory.clone();

            // Each daughter inherits half the parent's represented population.
            let daughter_represented_cells = parent_represented_cells * 0.5;
            let daughter_represented_packets = parent_represented_packets * 0.5;

            // Replace parent with daughter A.
            let snap_a = daughter_a.snapshot();
            self.explicit_microbes[idx] = TerrariumExplicitMicrobe {
                x: parent_x,
                y: parent_y,
                z: parent_z,
                radius: parent_radius,
                represented_cells: daughter_represented_cells,
                represented_packets: daughter_represented_packets,
                identity: parent_identity,
                age_s: 0.0,
                smoothed_energy: (snap_a.atp_mm * 0.40).clamp(0.0, 1.5),
                smoothed_stress: 0.12,
                cumulative_glucose_draw: 0.0,
                cumulative_oxygen_draw: 0.0,
                cumulative_ammonium_draw: 0.0,
                cumulative_nitrate_draw: 0.0,
                cumulative_co2_release: 0.0,
                cumulative_proton_release: 0.0,
                material_inventory: parent_material_inventory.clone(),
                simulator: daughter_a,
                last_snapshot: snap_a,
            };

            // Emit division telemetry event.
            self.ecology_events.push(EcologyTelemetryEvent::CellDivision {
                x: parent_x,
                y: parent_y,
                z: parent_z,
                parent_represented_cells,
                daughter_represented_cells,
            });

            // Add daughter B if we have room.
            if self.explicit_microbes.len() < self.config.max_explicit_microbes {
                let snap_b = daughter_b.snapshot();
                // Slight spatial offset so daughters don't overlap.
                let offset_x = if parent_x + 1 < self.config.width { parent_x + 1 } else { parent_x.saturating_sub(1) };
                self.explicit_microbes.push(TerrariumExplicitMicrobe {
                    x: offset_x,
                    y: parent_y,
                    z: parent_z,
                    radius: parent_radius,
                    represented_cells: daughter_represented_cells,
                    represented_packets: daughter_represented_packets,
                    identity: parent_identity,
                    age_s: 0.0,
                    smoothed_energy: (snap_b.atp_mm * 0.40).clamp(0.0, 1.5),
                    smoothed_stress: 0.12,
                    cumulative_glucose_draw: 0.0,
                    cumulative_oxygen_draw: 0.0,
                    cumulative_ammonium_draw: 0.0,
                    cumulative_nitrate_draw: 0.0,
                    cumulative_co2_release: 0.0,
                    cumulative_proton_release: 0.0,
                    material_inventory: parent_material_inventory,
                    simulator: daughter_b,
                    last_snapshot: snap_b,
                });
            } else {
                // At max cohorts: demote daughter B to coarse packet.
                let snap_b = daughter_b.snapshot();
                self.ecology_events.push(EcologyTelemetryEvent::ExplicitDemotion {
                    x: parent_x,
                    y: parent_y,
                    z: parent_z,
                    represented_cells: daughter_represented_cells,
                    atp_mm: snap_b.atp_mm,
                });
                self.bridge_explicit_to_coarse_packet(
                    parent_x, parent_y, parent_z, &snap_b, &parent_identity, daughter_represented_cells,
                );
            }
        }
        // Bridge dying cells to coarse packets before removing them.
        let mut dying_bridges: Vec<(usize, usize, usize, WholeCellSnapshot, TerrariumExplicitMicrobeIdentity, f32)> = Vec::new();
        for cohort in &self.explicit_microbes {
            let should_die = cohort.represented_cells < EXPLICIT_MICROBE_MIN_REPRESENTED_CELLS
                || (cohort.age_s > eco_dt * 3.0
                    && cohort.smoothed_energy < 0.08
                    && cohort.smoothed_stress > 1.15);
            if should_die {
                self.ecology_events.push(EcologyTelemetryEvent::ExplicitDeath {
                    x: cohort.x,
                    y: cohort.y,
                    z: cohort.z,
                    represented_cells: cohort.represented_cells,
                    age_s: cohort.age_s,
                });
                dying_bridges.push((
                    cohort.x,
                    cohort.y,
                    cohort.z,
                    cohort.last_snapshot.clone(),
                    cohort.identity,
                    cohort.represented_cells,
                ));
            }
        }
        self.explicit_microbes.retain(|cohort| {
            cohort.represented_cells >= EXPLICIT_MICROBE_MIN_REPRESENTED_CELLS
                && !(cohort.age_s > eco_dt * 3.0
                    && cohort.smoothed_energy < 0.08
                    && cohort.smoothed_stress > 1.15)
        });
        for (x, y, z, snapshot, identity, cells) in dying_bridges {
            self.bridge_explicit_to_coarse_packet(x, y, z, &snapshot, &identity, cells);
        }
        self.rebuild_explicit_microbe_fields();
        Ok(())
    }

    pub(super) fn step_explicit_microbes_incremental(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.explicit_microbes.is_empty() {
            return Ok(());
        }

        let n = self.explicit_microbes.len();
        let per_frame = INTERACTIVE_MICROBES_PER_FRAME.min(n);
        let start_idx = self.next_microbe_idx % n;

        for offset in 0..per_frame {
            let idx = (start_idx + offset) % n;
            // Interactive mode: only 1 ODE step per microbe.
            let _ = self.step_single_explicit_microbe(idx, eco_dt, 1)?;
        }

        self.next_microbe_idx = (start_idx + per_frame) % n.max(1);
        // Handle cell division: check which cohorts are ready to split.
        let mut division_indices: Vec<usize> = Vec::new();
        for (idx, cohort) in self.explicit_microbes.iter().enumerate() {
            if cohort.simulator.pending_division() {
                division_indices.push(idx);
            }
        }
        // Process divisions in reverse order to preserve indices.
        for &idx in division_indices.iter().rev() {
            let parent = &self.explicit_microbes[idx];
            let (daughter_a, daughter_b) = parent.simulator.split_into_daughters();
            let parent_x = parent.x;
            let parent_y = parent.y;
            let parent_z = parent.z;
            let parent_radius = parent.radius;
            let parent_represented_cells = parent.represented_cells;
            let parent_represented_packets = parent.represented_packets;
            let parent_identity = parent.identity;
            let parent_material_inventory = parent.material_inventory.clone();

            // Each daughter inherits half the parent's represented population.
            let daughter_represented_cells = parent_represented_cells * 0.5;
            let daughter_represented_packets = parent_represented_packets * 0.5;

            // Replace parent with daughter A.
            let snap_a = daughter_a.snapshot();
            self.explicit_microbes[idx] = TerrariumExplicitMicrobe {
                x: parent_x,
                y: parent_y,
                z: parent_z,
                radius: parent_radius,
                represented_cells: daughter_represented_cells,
                represented_packets: daughter_represented_packets,
                identity: parent_identity,
                age_s: 0.0,
                smoothed_energy: (snap_a.atp_mm * 0.40).clamp(0.0, 1.5),
                smoothed_stress: 0.12,
                cumulative_glucose_draw: 0.0,
                cumulative_oxygen_draw: 0.0,
                cumulative_ammonium_draw: 0.0,
                cumulative_nitrate_draw: 0.0,
                cumulative_co2_release: 0.0,
                cumulative_proton_release: 0.0,
                material_inventory: parent_material_inventory.clone(),
                simulator: daughter_a,
                last_snapshot: snap_a,
            };

            // Emit division telemetry event.
            self.ecology_events.push(EcologyTelemetryEvent::CellDivision {
                x: parent_x,
                y: parent_y,
                z: parent_z,
                parent_represented_cells,
                daughter_represented_cells,
            });

            // Add daughter B if we have room.
            if self.explicit_microbes.len() < self.config.max_explicit_microbes {
                let snap_b = daughter_b.snapshot();
                // Slight spatial offset so daughters don't overlap.
                let offset_x = if parent_x + 1 < self.config.width { parent_x + 1 } else { parent_x.saturating_sub(1) };
                self.explicit_microbes.push(TerrariumExplicitMicrobe {
                    x: offset_x,
                    y: parent_y,
                    z: parent_z,
                    radius: parent_radius,
                    represented_cells: daughter_represented_cells,
                    represented_packets: daughter_represented_packets,
                    identity: parent_identity,
                    age_s: 0.0,
                    smoothed_energy: (snap_b.atp_mm * 0.40).clamp(0.0, 1.5),
                    smoothed_stress: 0.12,
                    cumulative_glucose_draw: 0.0,
                    cumulative_oxygen_draw: 0.0,
                    cumulative_ammonium_draw: 0.0,
                    cumulative_nitrate_draw: 0.0,
                    cumulative_co2_release: 0.0,
                    cumulative_proton_release: 0.0,
                    material_inventory: parent_material_inventory,
                    simulator: daughter_b,
                    last_snapshot: snap_b,
                });
            } else {
                // At max cohorts: demote daughter B to coarse packet.
                let snap_b = daughter_b.snapshot();
                self.ecology_events.push(EcologyTelemetryEvent::ExplicitDemotion {
                    x: parent_x,
                    y: parent_y,
                    z: parent_z,
                    represented_cells: daughter_represented_cells,
                    atp_mm: snap_b.atp_mm,
                });
                self.bridge_explicit_to_coarse_packet(
                    parent_x, parent_y, parent_z, &snap_b, &parent_identity, daughter_represented_cells,
                );
            }
        }
        // Bridge dying cells to coarse packets before removing them.
        let mut dying_bridges: Vec<(usize, usize, usize, WholeCellSnapshot, TerrariumExplicitMicrobeIdentity, f32)> = Vec::new();
        for cohort in &self.explicit_microbes {
            let should_die = cohort.represented_cells < EXPLICIT_MICROBE_MIN_REPRESENTED_CELLS
                || (cohort.age_s > eco_dt * 3.0
                    && cohort.smoothed_energy < 0.08
                    && cohort.smoothed_stress > 1.15);
            if should_die {
                dying_bridges.push((
                    cohort.x,
                    cohort.y,
                    cohort.z,
                    cohort.last_snapshot.clone(),
                    cohort.identity,
                    cohort.represented_cells,
                ));
            }
        }
        self.explicit_microbes.retain(|cohort| {
            cohort.represented_cells >= EXPLICIT_MICROBE_MIN_REPRESENTED_CELLS
                && !(cohort.age_s > eco_dt * 3.0
                    && cohort.smoothed_energy < 0.08
                    && cohort.smoothed_stress > 1.15)
        });
        for (x, y, z, snapshot, identity, cells) in dying_bridges {
            self.bridge_explicit_to_coarse_packet(x, y, z, &snapshot, &identity, cells);
        }
        if self.explicit_microbes.is_empty() {
            self.next_microbe_idx = 0;
        }
        self.rebuild_explicit_microbe_fields();
        Ok(())
    }
}
