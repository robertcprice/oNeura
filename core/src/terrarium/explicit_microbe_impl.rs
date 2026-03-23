#![allow(dead_code)]

use super::*;
use crate::terrarium::genotype::{
    decode_secondary_gene_module, SecondaryCatalogBankEntry,
    DENITRIFIER_GENE_ANOXIA_RESPIRATION_WEIGHTS, DENITRIFIER_GENE_NITRATE_TRANSPORT_WEIGHTS,
    DENITRIFIER_GENE_REDUCTIVE_FLEXIBILITY_WEIGHTS, DENITRIFIER_GENE_STRESS_PERSISTENCE_WEIGHTS,
    MICROBIAL_GENE_CATABOLIC_WEIGHTS, MICROBIAL_GENE_DORMANCY_MAINTENANCE_WEIGHTS,
    MICROBIAL_GENE_EXTRACELLULAR_SCAVENGING_WEIGHTS, MICROBIAL_GENE_STRESS_RESPONSE_WEIGHTS,
    NITRIFIER_GENE_AMMONIUM_TRANSPORT_WEIGHTS, NITRIFIER_GENE_OXYGEN_RESPIRATION_WEIGHTS,
    NITRIFIER_GENE_REDOX_EFFICIENCY_WEIGHTS, NITRIFIER_GENE_STRESS_PERSISTENCE_WEIGHTS,
};
use crate::terrarium::packet::GenotypePacketEcology;

fn packet_ecology_to_guild(ecology: GenotypePacketEcology) -> u8 {
    match ecology {
        GenotypePacketEcology::Decomposer => 0,
        GenotypePacketEcology::Nitrifier => 1,
        GenotypePacketEcology::Denitrifier => 2,
    }
}

fn guild_to_packet_ecology(guild: u8) -> GenotypePacketEcology {
    match guild {
        1 => GenotypePacketEcology::Nitrifier,
        2 => GenotypePacketEcology::Denitrifier,
        _ => GenotypePacketEcology::Decomposer,
    }
}

fn terrarium_snapshot_from_whole_cell(
    snapshot: &crate::whole_cell::WholeCellSnapshot,
    inputs: &WholeCellEnvironmentInputs,
) -> WholeCellSnapshot {
    WholeCellSnapshot {
        atp_mm: snapshot.atp_mm,
        glucose_mm: snapshot.glucose_mm,
        oxygen_mm: snapshot.oxygen_mm,
        amino_acids_mm: snapshot.amino_acids_mm,
        nucleotides_mm: snapshot.nucleotides_mm,
        membrane_precursors_mm: snapshot.membrane_precursors_mm,
        metabolic_load: inputs.metabolic_load,
        division_progress: snapshot.division_progress,
        local_chemistry: snapshot.local_chemistry.map(|local| LocalChemistryReport {
            mean_carbon_dioxide: local.mean_carbon_dioxide,
            translation_support: local.translation_support,
            atp_support: local.atp_support,
            crowding_penalty: local.crowding_penalty,
        }),
    }
}

fn inventory_projection_regions() -> [MaterialRegionKind; 4] {
    [
        MaterialRegionKind::PoreWater,
        MaterialRegionKind::GasPhase,
        MaterialRegionKind::MineralSurface,
        MaterialRegionKind::BiofilmMatrix,
    ]
}

fn species_inventory_binding(
    species: TerrariumSpecies,
) -> Option<(
    MaterialRegionKind,
    MoleculeDescriptor,
    MaterialPhaseDescriptor,
)> {
    match species {
        TerrariumSpecies::Glucose => Some((
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_glucose(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous),
        )),
        TerrariumSpecies::OxygenGas => Some((
            MaterialRegionKind::GasPhase,
            MoleculeGraph::representative_oxygen_gas(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas),
        )),
        TerrariumSpecies::Ammonium => Some((
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_ammonium(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous),
        )),
        TerrariumSpecies::Nitrate => Some((
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_nitrate(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous),
        )),
        TerrariumSpecies::CarbonDioxide => Some((
            MaterialRegionKind::GasPhase,
            MoleculeGraph::representative_carbon_dioxide(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas),
        )),
        TerrariumSpecies::Proton => Some((
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_proton_pool(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous),
        )),
        TerrariumSpecies::AtpFlux => Some((
            MaterialRegionKind::BiofilmMatrix,
            MoleculeGraph::representative_atp(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous),
        )),
        TerrariumSpecies::AminoAcidPool => Some((
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_amino_acid_pool(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous),
        )),
        TerrariumSpecies::NucleotidePool => Some((
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_nucleotide_pool(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous),
        )),
        TerrariumSpecies::MembranePrecursorPool => Some((
            MaterialRegionKind::BiofilmMatrix,
            MoleculeGraph::representative_membrane_precursor_pool(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Amorphous),
        )),
        _ => None,
    }
}

fn inventory_component_amount(
    inventory: &RegionalMaterialInventory,
    species: TerrariumSpecies,
) -> f32 {
    let Some((region, molecule, phase)) = species_inventory_binding(species) else {
        return 0.0;
    };
    inventory.total_amount_for_component(region, &molecule, &MaterialPhaseSelector::Exact(phase))
}

impl TerrariumWorld {
    pub(crate) fn explicit_microbe_at(
        &self,
        x: usize,
        y: usize,
    ) -> Option<&TerrariumExplicitMicrobe> {
        self.explicit_microbes
            .iter()
            .find(|microbe| microbe.x == x && microbe.y == y)
    }

    /// Rebuild explicit microbe authority/activity fields from live cohorts.
    pub(super) fn rebuild_explicit_microbe_fields(&mut self) {
        let plane = self.config.width * self.config.height;
        self.explicit_microbe_authority.resize(plane, 0.0);
        self.explicit_microbe_activity.resize(plane, 0.0);
        self.explicit_microbe_authority.fill(0.0);
        self.explicit_microbe_activity.fill(0.0);
        for em in &self.explicit_microbes {
            let flat = em.y * self.config.width + em.x;
            if flat < plane {
                self.explicit_microbe_authority[flat] =
                    (self.explicit_microbe_authority[flat] + em.represented_cells * 0.002).min(1.0);
                self.explicit_microbe_activity[flat] =
                    (self.explicit_microbe_activity[flat] + em.smoothed_energy).min(2.0);
            }
        }
        for pop in &self.packet_populations {
            let flat = pop.y * self.config.width + pop.x;
            if flat >= plane {
                continue;
            }
            let ownership_strength = match self.ownership[flat].owner {
                SoilOwnershipClass::GenotypePacketRegion { .. } => self.ownership[flat].strength,
                _ => 0.0,
            };
            let packet_authority = (ownership_strength
                + (pop.total_cells / (MICROBIAL_PACKET_TARGET_CELLS * 8.0))
                    .sqrt()
                    .clamp(0.0, 1.0)
                    * 0.14)
                .min(1.0);
            self.explicit_microbe_authority[flat] =
                self.explicit_microbe_authority[flat].max(packet_authority);
            self.explicit_microbe_activity[flat] =
                self.explicit_microbe_activity[flat].max(pop.mean_activity().clamp(0.0, 1.0));
        }
    }

    /// Number of whole-cell steps per ecology step for a given simulator.
    pub(crate) fn explicit_microbe_step_count(
        _simulator: &crate::whole_cell::WholeCellSimulator,
        eco_dt: f32,
    ) -> usize {
        let effective_dt = eco_dt.max(0.0).min(24.0);
        (effective_dt.sqrt() * 0.85)
            .round()
            .max(EXPLICIT_MICROBE_MIN_STEPS as f32)
            .min(EXPLICIT_MICROBE_MAX_STEPS as f32) as usize
    }

    /// Build identity record for a new explicit microbe at the given cell.
    pub(super) fn explicit_microbe_identity_at(
        &self,
        flat: usize,
    ) -> TerrariumExplicitMicrobeIdentity {
        let width = self.config.width.max(1);
        let x = flat % width;
        let y = flat / width;
        let packet_population = self.packet_population_at(x, y);
        let preferred_ecology = packet_population
            .and_then(|pop| pop.dominant_ecology())
            .unwrap_or_else(|| {
                let decomposer = self.microbial_packets.get(flat).copied().unwrap_or(0.0);
                let nitrifier = self.nitrifier_packets.get(flat).copied().unwrap_or(0.0);
                let denitrifier = self.denitrifier_packets.get(flat).copied().unwrap_or(0.0);
                if nitrifier >= decomposer && nitrifier >= denitrifier {
                    GenotypePacketEcology::Nitrifier
                } else if denitrifier >= decomposer {
                    GenotypePacketEcology::Denitrifier
                } else {
                    GenotypePacketEcology::Decomposer
                }
            });

        let best_entry =
            |ecology: GenotypePacketEcology| -> Option<(usize, f32, SecondaryCatalogBankEntry)> {
                match ecology {
                    GenotypePacketEcology::Decomposer => {
                        self.microbial_secondary.dominant_catalog_entry_at(flat)
                    }
                    GenotypePacketEcology::Nitrifier => {
                        self.nitrifier_secondary.dominant_catalog_entry_at(flat)
                    }
                    GenotypePacketEcology::Denitrifier => {
                        self.denitrifier_secondary.dominant_catalog_entry_at(flat)
                    }
                }
            };

        let (bank_idx, _packets, entry, ecology) = best_entry(preferred_ecology)
            .map(|(bank_idx, packets, entry)| (bank_idx, packets, entry, preferred_ecology))
            .or_else(|| {
                [
                    (
                        GenotypePacketEcology::Decomposer,
                        self.microbial_secondary.dominant_catalog_entry_at(flat),
                    ),
                    (
                        GenotypePacketEcology::Nitrifier,
                        self.nitrifier_secondary.dominant_catalog_entry_at(flat),
                    ),
                    (
                        GenotypePacketEcology::Denitrifier,
                        self.denitrifier_secondary.dominant_catalog_entry_at(flat),
                    ),
                ]
                .into_iter()
                .filter_map(|(ecology, entry)| {
                    entry.map(|(bank_idx, packets, bank_entry)| {
                        (bank_idx, packets, bank_entry, ecology)
                    })
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
            })
            .unwrap_or((
                0,
                0.0,
                SecondaryCatalogBankEntry::default(),
                preferred_ecology,
            ));

        let genes = entry.genes;
        let (
            gene_catabolic,
            gene_stress_response,
            gene_dormancy_maintenance,
            gene_extracellular_scavenging,
        ) = match ecology {
            GenotypePacketEcology::Decomposer => (
                decode_secondary_gene_module(&genes, &MICROBIAL_GENE_CATABOLIC_WEIGHTS),
                decode_secondary_gene_module(&genes, &MICROBIAL_GENE_STRESS_RESPONSE_WEIGHTS),
                decode_secondary_gene_module(&genes, &MICROBIAL_GENE_DORMANCY_MAINTENANCE_WEIGHTS),
                decode_secondary_gene_module(
                    &genes,
                    &MICROBIAL_GENE_EXTRACELLULAR_SCAVENGING_WEIGHTS,
                ),
            ),
            GenotypePacketEcology::Nitrifier => (
                decode_secondary_gene_module(&genes, &NITRIFIER_GENE_OXYGEN_RESPIRATION_WEIGHTS),
                decode_secondary_gene_module(&genes, &NITRIFIER_GENE_STRESS_PERSISTENCE_WEIGHTS),
                decode_secondary_gene_module(&genes, &NITRIFIER_GENE_REDOX_EFFICIENCY_WEIGHTS),
                decode_secondary_gene_module(&genes, &NITRIFIER_GENE_AMMONIUM_TRANSPORT_WEIGHTS),
            ),
            GenotypePacketEcology::Denitrifier => (
                decode_secondary_gene_module(&genes, &DENITRIFIER_GENE_ANOXIA_RESPIRATION_WEIGHTS),
                decode_secondary_gene_module(&genes, &DENITRIFIER_GENE_STRESS_PERSISTENCE_WEIGHTS),
                decode_secondary_gene_module(
                    &genes,
                    &DENITRIFIER_GENE_REDUCTIVE_FLEXIBILITY_WEIGHTS,
                ),
                decode_secondary_gene_module(&genes, &DENITRIFIER_GENE_NITRATE_TRANSPORT_WEIGHTS),
            ),
        };

        TerrariumExplicitMicrobeIdentity {
            bank_idx,
            represented_packets: packet_population
                .map(|pop| pop.packets.len() as f32)
                .unwrap_or(entry.occupancy.max(1) as f32),
            record: entry.record,
            catalog: entry.catalog,
            genes,
            gene_catabolic,
            gene_stress_response,
            gene_dormancy_maintenance,
            gene_extracellular_scavenging,
        }
    }

    /// Compute growth signal from whole-cell snapshot.
    pub(super) fn explicit_microbe_growth_signal(
        snapshot: &WholeCellSnapshot,
        represented_cells: f32,
    ) -> f32 {
        let chemistry = snapshot.local_chemistry.unwrap_or_default();
        let energy = (snapshot.atp_mm / 1.6).clamp(0.0, 1.4);
        let carbon = (snapshot.glucose_mm / 2.2).clamp(0.0, 1.2);
        let oxygen = (snapshot.oxygen_mm / 2.0).clamp(0.0, 1.2);
        let translation = chemistry.translation_support.clamp(0.0, 1.4);
        let atp_support = chemistry.atp_support.clamp(0.0, 1.4);
        let division = snapshot.division_progress.clamp(0.0, 1.0);
        let crowding = (chemistry.crowding_penalty - 1.0).max(0.0);
        let size_penalty =
            (represented_cells / EXPLICIT_MICROBE_MAX_REPRESENTED_CELLS.max(1.0)).sqrt();
        clamp(
            energy * 0.28
                + carbon * 0.18
                + oxygen * 0.16
                + translation * 0.14
                + atp_support * 0.12
                + division * 0.12
                - snapshot.metabolic_load * 0.32
                - crowding * 0.16
                - size_penalty * 0.10,
            -1.0,
            1.0,
        )
    }

    /// Recruit new explicit microbes from high-activity soil cells.
    pub(super) fn recruit_explicit_microbes_from_soil(&mut self) -> Result<(), String> {
        self.recruit_packet_populations();
        if self.explicit_microbes.len() >= self.config.max_explicit_microbes {
            return Ok(());
        }

        let width = self.config.width;
        let depth = self.config.depth.max(1);
        let mut candidates = Vec::new();

        for pop in &self.packet_populations {
            let flat = idx2(width, pop.x, pop.y);
            let owner = self.ownership[flat].owner;
            if matches!(
                owner,
                SoilOwnershipClass::ExplicitMicrobeCohort { .. }
                    | SoilOwnershipClass::PlantTissueRegion { .. }
                    | SoilOwnershipClass::AtomisticProbeRegion { .. }
            ) {
                continue;
            }
            if self.explicit_microbes.iter().any(|microbe| {
                microbe.x.abs_diff(pop.x) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
                    && microbe.y.abs_diff(pop.y) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
            }) {
                continue;
            }

            let dominant_packet = match pop.dominant_packet() {
                Some(packet) => packet,
                None => continue,
            };
            let shoreline = self.shoreline_packet_recruitment_signal(flat);
            let promotion_pressure =
                pop.promotion_candidates() as f32 / pop.packets.len().max(1) as f32;
            let mutation = ((self.microbial_packet_mutation_flux[flat]
                + self.nitrifier_packet_mutation_flux[flat]
                + self.denitrifier_packet_mutation_flux[flat])
                * 2200.0)
                .clamp(0.0, 1.0);
            let score = clamp(
                shoreline * 0.26
                    + pop.mean_activity() * 0.18
                    + dominant_packet.reserve * 0.14
                    + promotion_pressure * 0.14
                    + pop.ecology_diversity() * 0.10
                    + (pop.total_cells / EXPLICIT_MICROBE_COHORT_CELLS.max(1.0))
                        .sqrt()
                        .clamp(0.0, 1.0)
                        * 0.12
                    + mutation * 0.06,
                0.0,
                1.0,
            );
            if score < EXPLICIT_MICROBE_RECRUITMENT_MIN_SCORE {
                continue;
            }

            candidates.push((
                score,
                pop.x,
                pop.y,
                pop.z.min(depth - 1),
                flat,
                pop.total_cells,
                pop.packets.len() as f32,
                pop.dominant_ecology()
                    .unwrap_or(GenotypePacketEcology::Decomposer),
            ));
        }

        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut reserved = self
            .explicit_microbes
            .iter()
            .map(|microbe| (microbe.x, microbe.y))
            .collect::<Vec<_>>();

        for (score, x, y, z, flat, total_cells, packet_count, ecology) in candidates {
            if self.explicit_microbes.len() >= self.config.max_explicit_microbes {
                break;
            }
            if reserved.iter().any(|(rx, ry)| {
                rx.abs_diff(x) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
                    && ry.abs_diff(y) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
            }) {
                continue;
            }

            let identity = self.explicit_microbe_identity_at(flat);
            let material_inventory = self.explicit_microbe_material_inventory_target_from_patch(
                x,
                y,
                z,
                EXPLICIT_MICROBE_PATCH_RADIUS,
                total_cells,
            );
            let material_inputs = material_inventory
                .estimate_whole_cell_environment_inputs(&inventory_projection_regions());
            let env_inputs = self.explicit_microbe_environment_inputs(x, y, z, &material_inputs);
            let mut simulator = crate::whole_cell::WholeCellSimulator::bundled_syn3a_reference()
                .unwrap_or_else(|_| crate::whole_cell::WholeCellSimulator::new_default());
            simulator.apply_environment_inputs(&env_inputs);
            let snapshot = simulator.snapshot();
            let cohort_id = self.next_microbe_idx as u32;
            let idx = self.next_microbe_idx;
            self.next_microbe_idx = self.next_microbe_idx.saturating_add(1);
            let strength = clamp(0.68 + score * 0.22, 0.58, 0.95);

            if !self.claim_ownership(
                x,
                y,
                SoilOwnershipClass::ExplicitMicrobeCohort { cohort_id },
                strength,
            ) {
                continue;
            }

            let terrarium_snapshot = terrarium_snapshot_from_whole_cell(&snapshot, &env_inputs);
            self.ecology_events
                .push(EcologyTelemetryEvent::PacketPromotion {
                    x,
                    y,
                    z,
                    activity: score,
                    represented_cells: total_cells,
                });
            self.ecology_events
                .push(EcologyTelemetryEvent::ExplicitPromotion {
                    x,
                    y,
                    z,
                    guild: packet_ecology_to_guild(ecology),
                    represented_cells: total_cells,
                });
            self.explicit_microbes.push(TerrariumExplicitMicrobe {
                x,
                y,
                z,
                guild: packet_ecology_to_guild(ecology),
                represented_cells: total_cells.clamp(
                    EXPLICIT_MICROBE_MIN_REPRESENTED_CELLS,
                    EXPLICIT_MICROBE_MAX_REPRESENTED_CELLS,
                ),
                represented_packets: packet_count.max(1.0),
                identity,
                whole_cell: None,
                simulator,
                last_snapshot: terrarium_snapshot,
                smoothed_energy: 0.6,
                smoothed_stress: 0.2,
                radius: EXPLICIT_MICROBE_PATCH_RADIUS,
                patch_radius: EXPLICIT_MICROBE_PATCH_RADIUS,
                age_steps: 0,
                age_s: 0.0,
                idx,
                material_inventory,
                cumulative_glucose_draw: 0.0,
                cumulative_oxygen_draw: 0.0,
                cumulative_co2_release: 0.0,
                cumulative_ammonium_draw: 0.0,
                cumulative_nitrate_draw: 0.0,
                cumulative_proton_release: 0.0,
            });
            reserved.push((x, y));
        }
        Ok(())
    }

    /// Build environment inputs for a whole-cell simulator from local soil chemistry.
    pub(super) fn explicit_microbe_environment_inputs(
        &self,
        x: usize,
        y: usize,
        z: usize,
        material_inputs: &WholeCellEnvironmentInputs,
    ) -> WholeCellEnvironmentInputs {
        let flat = idx2(self.config.width, x, y);
        let air_idx = idx3(
            self.config.width,
            self.config.height,
            x,
            y,
            z.min(self.config.depth.saturating_sub(1)),
        );
        let shoreline = shoreline_water_signal(
            self.config.width,
            self.config.height,
            &self.water_mask,
            flat,
        );
        let open_water = self.water_mask[flat].clamp(0.0, 1.0);
        let crowding = clamp(
            material_inputs.metabolic_load * 0.72
                + material_inputs.proton_concentration * 1.0e4
                + open_water * 0.35
                + shoreline * 0.18,
            0.0,
            1.5,
        );
        let metabolic_load = clamp(
            0.24 + crowding
                + (0.12 - material_inputs.oxygen_mm).max(0.0) * 1.8
                + (0.04 - material_inputs.glucose_mm).max(0.0) * 2.8,
            0.1,
            2.6,
        );
        WholeCellEnvironmentInputs {
            glucose_mm: material_inputs.glucose_mm.max(0.02),
            oxygen_mm: clamp(
                material_inputs.oxygen_mm + self.odorants[ATMOS_O2_IDX][air_idx] * 0.18,
                0.02,
                8.0,
            ),
            amino_acids_mm: material_inputs.amino_acids_mm.max(0.01),
            nucleotides_mm: material_inputs.nucleotides_mm.max(0.01),
            membrane_precursors_mm: material_inputs.membrane_precursors_mm.max(0.01),
            metabolic_load: material_inputs.metabolic_load.max(metabolic_load),
            temperature_c: self.temperature[air_idx],
            proton_concentration: material_inputs.proton_concentration,
        }
    }

    /// Build material inventory for an explicit microbe's patch.
    pub(super) fn explicit_microbe_material_inventory(
        &self,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
    ) -> RegionalMaterialInventory {
        if let Some(cohort) = self.explicit_microbe_at(x, y) {
            if !cohort.material_inventory.is_empty() {
                return cohort.material_inventory.clone();
            }
            return self.explicit_microbe_material_inventory_target_from_patch(
                x,
                y,
                z,
                radius,
                cohort.represented_cells,
            );
        }
        self.explicit_microbe_material_inventory_target_from_patch(x, y, z, radius, 0.0)
    }

    /// Build material inventory target from surrounding patch.
    pub(super) fn explicit_microbe_material_inventory_target_from_patch(
        &self,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        _represented_cells: f32,
    ) -> RegionalMaterialInventory {
        let flat = idx2(self.config.width, x, y);
        let air_idx = idx3(
            self.config.width,
            self.config.height,
            x,
            y,
            z.min(self.config.depth.saturating_sub(1)),
        );
        let mut inventory = RegionalMaterialInventory::new(format!("explicit_microbe:{x}:{y}:{z}"));
        let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
        let gas = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas);
        let glucose = self
            .substrate
            .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, radius);
        let oxygen =
            self.substrate
                .patch_mean_species(TerrariumSpecies::OxygenGas, x, y, z, radius);
        let ammonium =
            self.substrate
                .patch_mean_species(TerrariumSpecies::Ammonium, x, y, z, radius);
        let nitrate = self
            .substrate
            .patch_mean_species(TerrariumSpecies::Nitrate, x, y, z, radius);
        let atp_flux =
            self.substrate
                .patch_mean_species(TerrariumSpecies::AtpFlux, x, y, z, radius);
        let amino_acids =
            self.substrate
                .patch_mean_species(TerrariumSpecies::AminoAcidPool, x, y, z, radius);
        let nucleotides =
            self.substrate
                .patch_mean_species(TerrariumSpecies::NucleotidePool, x, y, z, radius);
        let membrane_precursors = self.substrate.patch_mean_species(
            TerrariumSpecies::MembranePrecursorPool,
            x,
            y,
            z,
            radius,
        );
        let carbon_dioxide =
            self.substrate
                .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, radius);
        let proton = self
            .substrate
            .patch_mean_species(TerrariumSpecies::Proton, x, y, z, radius);
        let shoreline = shoreline_water_signal(
            self.config.width,
            self.config.height,
            &self.water_mask,
            flat,
        );
        let open_water = self.water_mask[flat].clamp(0.0, 1.0);
        let air_o2 = self.odorants[ATMOS_O2_IDX][air_idx];
        let air_co2 = self.odorants[ATMOS_CO2_IDX][air_idx];
        let _ = inventory.set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_glucose(),
            aqueous.clone(),
            clamp(glucose * 14.0, 0.02, 8.0) as f64,
        );
        let _ = inventory.set_component_amount(
            MaterialRegionKind::GasPhase,
            MoleculeGraph::representative_oxygen_gas(),
            gas.clone(),
            clamp(
                oxygen * 14.0 + air_o2 * 1.2 + (1.0 - open_water) * 0.12,
                0.02,
                8.0,
            ) as f64,
        );
        let _ = inventory.set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_ammonium(),
            aqueous.clone(),
            clamp(ammonium * 5.0, 0.01, 6.0) as f64,
        );
        let _ = inventory.set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_nitrate(),
            aqueous.clone(),
            clamp(nitrate * 4.0 + atp_flux * 0.5, 0.01, 6.0) as f64,
        );
        let _ = inventory.set_component_amount(
            MaterialRegionKind::BiofilmMatrix,
            MoleculeGraph::representative_atp(),
            aqueous.clone(),
            clamp(atp_flux * 8.0 + glucose * 0.35, 0.0, 2.5) as f64,
        );
        let _ = inventory.set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_amino_acid_pool(),
            aqueous.clone(),
            clamp(amino_acids * 12.0 + ammonium * 0.8, 0.0, 6.0) as f64,
        );
        let _ = inventory.set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_nucleotide_pool(),
            aqueous.clone(),
            clamp(
                nucleotides * 12.0 + nitrate * 0.8 + atp_flux * 0.4,
                0.0,
                6.0,
            ) as f64,
        );
        let _ = inventory.set_component_amount(
            MaterialRegionKind::BiofilmMatrix,
            MoleculeGraph::representative_membrane_precursor_pool(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Amorphous),
            clamp(
                membrane_precursors * 12.0 + glucose * 0.8 + atp_flux * 0.2,
                0.0,
                5.0,
            ) as f64,
        );
        let _ = inventory.set_component_amount(
            MaterialRegionKind::GasPhase,
            MoleculeGraph::representative_carbon_dioxide(),
            gas,
            clamp(carbon_dioxide * 12.0 + air_co2 * 1.2, 0.0, 6.0) as f64,
        );
        let _ = inventory.set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_proton_pool(),
            aqueous,
            clamp(proton * 8.0 + shoreline * 0.08, 0.0, 1.5) as f64,
        );
        inventory
    }

    /// Fraction of material exchange allowed by ownership authority.
    pub(super) fn explicit_microbe_material_exchange_fraction(authority: f32) -> f64 {
        (authority as f64).clamp(0.0, 1.0)
    }

    /// Normalize component shares with fallback.
    pub(super) fn normalize_component_shares(potentials: &[f64], fallback: &[f64]) -> Vec<f64> {
        let total: f64 = potentials.iter().sum();
        if total > 1e-12 {
            potentials.iter().map(|p| p / total).collect()
        } else {
            fallback.to_vec()
        }
    }

    /// Reserve-band helper for clamped substrate extraction/deposit.
    pub(super) fn reserve_band(
        value: f32,
        lo_scale: f32,
        hi_scale: f32,
        lower_floor: f32,
        upper_floor: f32,
    ) -> (f32, f32) {
        let lower = (value * lo_scale).max(lower_floor.max(0.0));
        let upper = (value * hi_scale).max(lower + upper_floor.max(0.0));
        (lower, upper)
    }

    /// Sync a single species component between substrate and explicit microbe.
    pub(super) fn sync_owned_component_single_species(
        &mut self,
        idx: usize,
        species: crate::terrarium::TerrariumSpecies,
        fraction: f64,
        target_mm: f32,
    ) -> Result<(), String> {
        let Some((region, molecule, phase)) = species_inventory_binding(species) else {
            return Ok(());
        };
        if idx >= self.explicit_microbes.len() {
            return Ok(());
        }
        let (x, y, z, radius, current) = {
            let cohort = &self.explicit_microbes[idx];
            (
                cohort.x,
                cohort.y,
                cohort.z,
                cohort.radius.max(1),
                cohort.material_inventory.total_amount_for_component(
                    region,
                    &molecule,
                    &MaterialPhaseSelector::Exact(phase.clone()),
                ),
            )
        };
        let (lower, upper) = Self::reserve_band(target_mm, 0.55, 1.35, 0.02, 0.03);
        if current + 1.0e-9 < lower {
            let draw = ((target_mm - current).max(0.0) * fraction as f32).max(0.0);
            if draw > 1.0e-9 {
                let pulled = self
                    .substrate
                    .extract_patch_species(species, x, y, z, radius, draw);
                if pulled > 0.0 {
                    self.explicit_microbes[idx]
                        .material_inventory
                        .deposit_component(region, molecule, pulled as f64, phase);
                }
            }
        } else if current > upper + 1.0e-9 {
            let release = ((current - target_mm).max(0.0) * fraction as f32).max(0.0);
            if release > 1.0e-9 {
                let removed = self.explicit_microbes[idx]
                    .material_inventory
                    .withdraw_component(region, &molecule, release as f64)
                    as f32;
                if removed > 0.0 {
                    match species {
                        TerrariumSpecies::CarbonDioxide => {
                            let patch_share = removed * 0.65;
                            let air_share = removed - patch_share;
                            let _ = self.substrate.deposit_patch_species(
                                species,
                                x,
                                y,
                                z,
                                radius,
                                patch_share,
                            );
                            self.exchange_atmosphere_odorant(
                                ATMOS_CO2_IDX,
                                x,
                                y,
                                z,
                                radius,
                                air_share * 0.01,
                            );
                        }
                        _ => {
                            let _ = self
                                .substrate
                                .deposit_patch_species(species, x, y, z, radius, removed);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Sync amino acid pool between substrate and explicit microbe.
    pub(super) fn sync_owned_component_amino_pool(
        &mut self,
        idx: usize,
        fraction: f64,
        target_mm: f32,
    ) -> Result<(), String> {
        self.sync_owned_component_single_species(
            idx,
            TerrariumSpecies::AminoAcidPool,
            fraction,
            target_mm,
        )
    }

    /// Sync nucleotide pool between substrate and explicit microbe.
    pub(super) fn sync_owned_component_nucleotide_pool(
        &mut self,
        idx: usize,
        fraction: f64,
        target_mm: f32,
    ) -> Result<(), String> {
        self.sync_owned_component_single_species(
            idx,
            TerrariumSpecies::NucleotidePool,
            fraction,
            target_mm,
        )
    }

    /// Sync oxygen pool between substrate and explicit microbe.
    pub(super) fn sync_owned_component_oxygen_pool(
        &mut self,
        idx: usize,
        fraction: f64,
        target_mm: f32,
    ) -> Result<(), String> {
        self.sync_owned_component_single_species(
            idx,
            TerrariumSpecies::OxygenGas,
            fraction,
            target_mm,
        )
    }

    /// Sync membrane precursor pool between substrate and explicit microbe.
    pub(super) fn sync_owned_component_membrane_pool(
        &mut self,
        idx: usize,
        fraction: f64,
        target_mm: f32,
    ) -> Result<(), String> {
        self.sync_owned_component_single_species(
            idx,
            TerrariumSpecies::MembranePrecursorPool,
            fraction,
            target_mm,
        )
    }

    /// Sync ATP pool between substrate and explicit microbe.
    pub(super) fn sync_owned_component_atp_pool(
        &mut self,
        idx: usize,
        fraction: f64,
        target_mm: f32,
    ) -> Result<(), String> {
        self.sync_owned_component_single_species(
            idx,
            TerrariumSpecies::AtpFlux,
            fraction,
            target_mm,
        )
    }

    /// Spill CO2 from explicit microbe to substrate.
    pub(super) fn spill_owned_component_carbon_dioxide_pool(
        &mut self,
        idx: usize,
        fraction: f64,
        release: f32,
    ) -> Result<(), String> {
        self.sync_owned_component_single_species(
            idx,
            TerrariumSpecies::CarbonDioxide,
            fraction,
            release,
        )
    }

    /// Spill protons from explicit microbe to substrate.
    pub(super) fn spill_owned_component_proton_pool(
        &mut self,
        idx: usize,
        fraction: f64,
        release: f32,
    ) -> Result<(), String> {
        self.sync_owned_component_single_species(idx, TerrariumSpecies::Proton, fraction, release)
    }

    /// Sync full material inventory for an explicit microbe with substrate.
    pub(super) fn sync_owned_explicit_microbe_material_inventory(
        &mut self,
        idx: usize,
        target: &RegionalMaterialInventory,
        fraction: f64,
    ) -> Result<(), String> {
        self.sync_owned_component_single_species(
            idx,
            TerrariumSpecies::Glucose,
            fraction,
            inventory_component_amount(target, TerrariumSpecies::Glucose),
        )?;
        self.sync_owned_component_oxygen_pool(
            idx,
            fraction,
            inventory_component_amount(target, TerrariumSpecies::OxygenGas),
        )?;
        self.sync_owned_component_amino_pool(
            idx,
            fraction,
            inventory_component_amount(target, TerrariumSpecies::AminoAcidPool),
        )?;
        self.sync_owned_component_nucleotide_pool(
            idx,
            fraction,
            inventory_component_amount(target, TerrariumSpecies::NucleotidePool),
        )?;
        self.sync_owned_component_membrane_pool(
            idx,
            fraction,
            inventory_component_amount(target, TerrariumSpecies::MembranePrecursorPool),
        )?;
        self.sync_owned_component_atp_pool(
            idx,
            fraction,
            inventory_component_amount(target, TerrariumSpecies::AtpFlux),
        )?;
        self.spill_owned_component_carbon_dioxide_pool(
            idx,
            fraction,
            inventory_component_amount(target, TerrariumSpecies::CarbonDioxide),
        )?;
        self.spill_owned_component_proton_pool(
            idx,
            fraction,
            inventory_component_amount(target, TerrariumSpecies::Proton),
        )?;
        Ok(())
    }

    /// Sync explicit microbe material inventory (high-level).
    pub(super) fn sync_explicit_microbe_material_inventory(
        &mut self,
        idx: usize,
    ) -> Result<(), String> {
        if idx >= self.explicit_microbes.len() {
            return Ok(());
        }
        let (x, y, z, radius, represented_cells, authority, empty) = {
            let cohort = &self.explicit_microbes[idx];
            (
                cohort.x,
                cohort.y,
                cohort.z,
                cohort.radius.max(1),
                cohort.represented_cells,
                self.ownership[idx2(self.config.width, cohort.x, cohort.y)]
                    .strength
                    .max(
                        self.explicit_microbe_authority
                            [idx2(self.config.width, cohort.x, cohort.y)],
                    ),
                cohort.material_inventory.is_empty(),
            )
        };
        let target = self.explicit_microbe_material_inventory_target_from_patch(
            x,
            y,
            z,
            radius,
            represented_cells,
        );
        if empty {
            self.explicit_microbes[idx].material_inventory = target.scaled(0.92);
        }
        let fraction = Self::explicit_microbe_material_exchange_fraction(authority.max(0.35));
        self.sync_owned_explicit_microbe_material_inventory(idx, &target, fraction)
    }

    /// Check available material in owned substrate region.
    pub(super) fn owned_inventory_available(
        &self,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        species: crate::terrarium::TerrariumSpecies,
    ) -> f32 {
        if let Some(cohort) = self.explicit_microbe_at(x, y) {
            return inventory_component_amount(&cohort.material_inventory, species);
        }
        self.substrate
            .patch_mean_species(species, x, y, z, radius.max(1))
    }

    /// Apply material fluxes computed by whole-cell simulation.
    pub(super) fn apply_explicit_microbe_material_fluxes(
        &mut self,
        idx: usize,
        fluxes: &[(crate::terrarium::TerrariumSpecies, f32)],
    ) -> Result<Vec<(crate::terrarium::TerrariumSpecies, f32)>, String> {
        if idx >= self.explicit_microbes.len() {
            return Ok(Vec::new());
        }
        let mut applied = Vec::with_capacity(fluxes.len());
        for (species, delta) in fluxes {
            let Some((region, molecule, phase)) = species_inventory_binding(*species) else {
                applied.push((*species, *delta));
                continue;
            };
            if *delta > 0.0 {
                self.explicit_microbes[idx]
                    .material_inventory
                    .deposit_component(region, molecule, *delta as f64, phase);
                applied.push((*species, *delta));
            } else if *delta < 0.0 {
                let removed = self.explicit_microbes[idx]
                    .material_inventory
                    .withdraw_component(region, &molecule, delta.abs() as f64)
                    as f32;
                applied.push((*species, -removed));
            } else {
                applied.push((*species, 0.0));
            }
        }
        Ok(applied)
    }

    /// Compute material fluxes for a single explicit microbe.
    pub(super) fn compute_explicit_microbe_fluxes(
        &self,
        idx: usize,
        snapshot: &WholeCellSnapshot,
        _eco_dt: f32,
    ) -> Vec<(crate::terrarium::TerrariumSpecies, f32)> {
        let Some(cohort) = self.explicit_microbes.get(idx) else {
            return Vec::new();
        };
        let chemistry = snapshot.local_chemistry.unwrap_or_default();
        let guild = guild_to_packet_ecology(cohort.guild);
        let represented_scale =
            (cohort.represented_cells / EXPLICIT_MICROBE_COHORT_CELLS.max(1.0)).sqrt();
        let activity = clamp(
            cohort.smoothed_energy * (1.15 - cohort.smoothed_stress * 0.45)
                + snapshot.division_progress * 0.18
                + chemistry.translation_support * 0.08,
            0.02,
            1.6,
        );
        let base_draw = represented_scale * activity * 0.0022;
        let glucose_draw = base_draw * (0.72 + cohort.identity.gene_catabolic * 0.46);
        let oxygen_draw = base_draw
            * match guild {
                GenotypePacketEcology::Decomposer => 0.68,
                GenotypePacketEcology::Nitrifier => 0.96,
                GenotypePacketEcology::Denitrifier => 0.24,
            };
        let ammonium_draw = base_draw
            * match guild {
                GenotypePacketEcology::Decomposer => 0.12,
                GenotypePacketEcology::Nitrifier => 0.50,
                GenotypePacketEcology::Denitrifier => 0.06,
            };
        let nitrate_draw = base_draw
            * match guild {
                GenotypePacketEcology::Decomposer => 0.04,
                GenotypePacketEcology::Nitrifier => 0.08,
                GenotypePacketEcology::Denitrifier => 0.54,
            };
        let amino_draw = base_draw * (0.18 + chemistry.translation_support * 0.12);
        let nucleotide_draw = base_draw * (0.12 + chemistry.atp_support * 0.08);
        let membrane_draw = base_draw * (0.10 + snapshot.division_progress * 0.14);
        let co2_release = glucose_draw * (0.74 + cohort.smoothed_stress * 0.10);
        let proton_release = match guild {
            GenotypePacketEcology::Decomposer => glucose_draw * 0.08,
            GenotypePacketEcology::Nitrifier => ammonium_draw * 0.48 + glucose_draw * 0.10,
            GenotypePacketEcology::Denitrifier => nitrate_draw * 0.14 + glucose_draw * 0.05,
        };
        let atp_flux = activity * represented_scale * 0.0007;
        vec![
            (TerrariumSpecies::Glucose, -glucose_draw),
            (TerrariumSpecies::OxygenGas, -oxygen_draw),
            (TerrariumSpecies::Ammonium, -ammonium_draw),
            (TerrariumSpecies::Nitrate, -nitrate_draw),
            (TerrariumSpecies::AminoAcidPool, -amino_draw),
            (TerrariumSpecies::NucleotidePool, -nucleotide_draw),
            (TerrariumSpecies::MembranePrecursorPool, -membrane_draw),
            (TerrariumSpecies::CarbonDioxide, co2_release),
            (TerrariumSpecies::Proton, proton_release),
            (TerrariumSpecies::AtpFlux, atp_flux),
        ]
    }

    fn explicit_microbe_division_site(&self, idx: usize) -> Option<(usize, usize)> {
        let Some(cohort) = self.explicit_microbes.get(idx) else {
            return None;
        };
        let width = self.config.width;
        let height = self.config.height;
        let mut best = None;
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = cohort.x as isize + dx;
                let ny = cohort.y as isize + dy;
                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                    continue;
                }
                let nx = nx as usize;
                let ny = ny as usize;
                if self
                    .explicit_microbes
                    .iter()
                    .any(|other| other.x == nx && other.y == ny)
                {
                    continue;
                }
                let flat = idx2(width, nx, ny);
                if matches!(
                    self.ownership[flat].owner,
                    SoilOwnershipClass::PlantTissueRegion { .. }
                        | SoilOwnershipClass::AtomisticProbeRegion { .. }
                        | SoilOwnershipClass::ExplicitMicrobeCohort { .. }
                ) {
                    continue;
                }
                let packet_bonus = matches!(
                    self.ownership[flat].owner,
                    SoilOwnershipClass::GenotypePacketRegion { .. }
                ) as u8 as f32
                    * 0.08;
                let distance_penalty = ((dx * dx + dy * dy) as f32).sqrt() * 0.04;
                let score = clamp(
                    self.shoreline_packet_recruitment_signal(flat) * 0.36
                        + self.moisture[flat] * 0.24
                        + self.deep_moisture[flat] * 0.10
                        + self.explicit_microbe_activity[flat].min(1.0) * 0.12
                        + packet_bonus
                        - self.water_mask[flat] * 0.18
                        - distance_penalty,
                    -1.0,
                    1.0,
                );
                if best
                    .as_ref()
                    .map(|(best_score, _, _)| score > *best_score)
                    .unwrap_or(true)
                {
                    best = Some((score, nx, ny));
                }
            }
        }
        best.map(|(_, x, y)| (x, y))
    }

    fn release_explicit_microbe_inventory_to_environment(
        &mut self,
        idx: usize,
    ) -> Result<(), String> {
        if idx >= self.explicit_microbes.len() {
            return Ok(());
        }
        let (x, y, z, radius) = {
            let cohort = &self.explicit_microbes[idx];
            (cohort.x, cohort.y, cohort.z, cohort.radius.max(1))
        };
        for species in [
            TerrariumSpecies::Glucose,
            TerrariumSpecies::OxygenGas,
            TerrariumSpecies::Ammonium,
            TerrariumSpecies::Nitrate,
            TerrariumSpecies::AtpFlux,
            TerrariumSpecies::AminoAcidPool,
            TerrariumSpecies::NucleotidePool,
            TerrariumSpecies::MembranePrecursorPool,
            TerrariumSpecies::CarbonDioxide,
            TerrariumSpecies::Proton,
        ] {
            let Some((region, molecule, _phase)) = species_inventory_binding(species) else {
                continue;
            };
            let current = inventory_component_amount(
                &self.explicit_microbes[idx].material_inventory,
                species,
            );
            if current <= 1.0e-9 {
                continue;
            }
            let removed = self.explicit_microbes[idx]
                .material_inventory
                .withdraw_component(region, &molecule, current as f64)
                as f32;
            if removed <= 0.0 {
                continue;
            }
            match species {
                TerrariumSpecies::CarbonDioxide => {
                    let patch_share = removed * 0.65;
                    let air_share = removed - patch_share;
                    let _ =
                        self.substrate
                            .deposit_patch_species(species, x, y, z, radius, patch_share);
                    self.exchange_atmosphere_odorant(
                        ATMOS_CO2_IDX,
                        x,
                        y,
                        z,
                        radius,
                        air_share * 0.01,
                    );
                }
                _ => {
                    let _ = self
                        .substrate
                        .deposit_patch_species(species, x, y, z, radius, removed);
                }
            }
        }
        Ok(())
    }

    fn try_divide_explicit_microbe(&mut self, idx: usize) -> Result<Option<usize>, String> {
        let Some(parent) = self.explicit_microbes.get(idx) else {
            return Ok(None);
        };
        if !parent.simulator.pending_division()
            || parent.represented_cells < EXPLICIT_MICROBE_MIN_REPRESENTED_CELLS * 2.0
        {
            return Ok(None);
        }
        let Some((daughter_x, daughter_y)) = self.explicit_microbe_division_site(idx) else {
            return Ok(None);
        };
        let daughter_id = self.next_microbe_idx as u32;
        let daughter_idx = self.next_microbe_idx;
        if !self.claim_ownership(
            daughter_x,
            daughter_y,
            SoilOwnershipClass::ExplicitMicrobeCohort {
                cohort_id: daughter_id,
            },
            0.72,
        ) {
            return Ok(None);
        }
        self.next_microbe_idx = self.next_microbe_idx.saturating_add(1);

        let (parent_x, parent_y, parent_z, parent_cells_before) = {
            let parent = &self.explicit_microbes[idx];
            (parent.x, parent.y, parent.z, parent.represented_cells)
        };
        let (
            daughter_simulator,
            daughter_inventory,
            daughter_cells,
            daughter_packets,
            daughter_guild,
            daughter_identity,
            daughter_radius,
            daughter_patch_radius,
            daughter_age_steps,
            daughter_age_s,
            daughter_energy,
            daughter_stress,
        ) = {
            let parent = &mut self.explicit_microbes[idx];
            let (left, right) = parent.simulator.split_into_daughters();
            parent.simulator = left;
            let daughter_inventory = parent.material_inventory.scaled(0.5);
            parent.material_inventory.scale_in_place(0.5);
            parent.represented_cells *= 0.5;
            parent.represented_packets = (parent.represented_packets * 0.5).max(1.0);
            parent.last_snapshot.division_progress = 0.0;
            (
                right,
                daughter_inventory,
                parent.represented_cells.max(1.0),
                parent.represented_packets.max(1.0),
                parent.guild,
                parent.identity.clone(),
                parent.radius.max(EXPLICIT_MICROBE_PATCH_RADIUS),
                parent.patch_radius.max(EXPLICIT_MICROBE_PATCH_RADIUS),
                parent.age_steps,
                parent.age_s,
                parent.smoothed_energy,
                parent.smoothed_stress,
            )
        };

        let parent_material_inputs = self.explicit_microbes[idx]
            .material_inventory
            .estimate_whole_cell_environment_inputs(&inventory_projection_regions());
        let parent_env_inputs = self.explicit_microbe_environment_inputs(
            parent_x,
            parent_y,
            parent_z,
            &parent_material_inputs,
        );
        let parent_snapshot = self.explicit_microbes[idx].simulator.snapshot();
        self.explicit_microbes[idx].last_snapshot =
            terrarium_snapshot_from_whole_cell(&parent_snapshot, &parent_env_inputs);

        let daughter_material_inputs = daughter_inventory
            .estimate_whole_cell_environment_inputs(&inventory_projection_regions());
        let daughter_env_inputs = self.explicit_microbe_environment_inputs(
            daughter_x,
            daughter_y,
            parent_z,
            &daughter_material_inputs,
        );
        let daughter_snapshot = daughter_simulator.snapshot();
        let daughter_last_snapshot =
            terrarium_snapshot_from_whole_cell(&daughter_snapshot, &daughter_env_inputs);

        self.explicit_microbes.push(TerrariumExplicitMicrobe {
            x: daughter_x,
            y: daughter_y,
            z: parent_z,
            guild: daughter_guild,
            represented_cells: daughter_cells,
            represented_packets: daughter_packets,
            identity: daughter_identity,
            whole_cell: None,
            simulator: daughter_simulator,
            last_snapshot: daughter_last_snapshot.clone(),
            smoothed_energy: daughter_energy,
            smoothed_stress: daughter_stress,
            radius: daughter_radius,
            patch_radius: daughter_patch_radius,
            age_steps: daughter_age_steps,
            age_s: daughter_age_s,
            idx: daughter_idx,
            material_inventory: daughter_inventory,
            cumulative_glucose_draw: 0.0,
            cumulative_oxygen_draw: 0.0,
            cumulative_co2_release: 0.0,
            cumulative_ammonium_draw: 0.0,
            cumulative_nitrate_draw: 0.0,
            cumulative_proton_release: 0.0,
        });
        self.ecology_events
            .push(EcologyTelemetryEvent::CellDivision {
                x: parent_x,
                y: parent_y,
                z: parent_z,
                parent_represented_cells: parent_cells_before,
                daughter_represented_cells: daughter_cells,
            });
        self.ecology_events
            .push(EcologyTelemetryEvent::CellDivisionDaughter {
                x: daughter_x,
                y: daughter_y,
                z: parent_z,
                represented_cells: daughter_cells,
                atp_mm: daughter_last_snapshot.atp_mm,
            });
        Ok(Some(self.explicit_microbes.len() - 1))
    }

    fn demote_explicit_microbe_to_packet(&mut self, idx: usize) -> Result<(), String> {
        if idx >= self.explicit_microbes.len() {
            return Ok(());
        }
        self.bridge_explicit_to_coarse_packet(idx);
        self.release_explicit_microbe_inventory_to_environment(idx)?;
        if idx >= self.explicit_microbes.len() {
            return Ok(());
        }
        let cohort = self.explicit_microbes.remove(idx);
        self.release_ownership(cohort.x, cohort.y);
        self.ecology_events
            .push(EcologyTelemetryEvent::ExplicitDemotion {
                x: cohort.x,
                y: cohort.y,
                z: cohort.z,
                represented_cells: cohort.represented_cells,
                atp_mm: cohort.last_snapshot.atp_mm,
            });
        Ok(())
    }

    /// Step a single explicit microbe: run whole-cell sim + material exchange.
    pub(super) fn step_single_explicit_microbe(
        &mut self,
        idx: usize,
        eco_dt: f32,
    ) -> Result<(), String> {
        if idx >= self.explicit_microbes.len() {
            return Ok(());
        }

        let (x, y, z, represented_cells) = {
            let cohort = &self.explicit_microbes[idx];
            (cohort.x, cohort.y, cohort.z, cohort.represented_cells)
        };
        self.sync_explicit_microbe_material_inventory(idx)?;
        let base_material_inputs = self.explicit_microbes[idx]
            .material_inventory
            .estimate_whole_cell_environment_inputs(&inventory_projection_regions());
        let env_inputs = self.explicit_microbe_environment_inputs(x, y, z, &base_material_inputs);
        let updated_snapshot = {
            let cohort = &mut self.explicit_microbes[idx];
            cohort.simulator.apply_environment_inputs(&env_inputs);
            let steps = Self::explicit_microbe_step_count(&cohort.simulator, eco_dt);
            for _ in 0..steps {
                cohort.simulator.step();
            }
            let whole_cell_snapshot = cohort.simulator.snapshot();
            let terrarium_snapshot =
                terrarium_snapshot_from_whole_cell(&whole_cell_snapshot, &env_inputs);
            let growth_signal =
                Self::explicit_microbe_growth_signal(&terrarium_snapshot, represented_cells);
            cohort.last_snapshot = terrarium_snapshot.clone();
            cohort.age_steps = cohort.age_steps.saturating_add(steps as u64);
            cohort.age_s += eco_dt;
            cohort.smoothed_energy = clamp(
                cohort.smoothed_energy * 0.72
                    + (terrarium_snapshot.atp_mm * 0.46
                        + terrarium_snapshot.glucose_mm * 0.14
                        + terrarium_snapshot
                            .local_chemistry
                            .unwrap_or_default()
                            .translation_support
                            * 0.20)
                        * 0.28,
                0.0,
                1.6,
            );
            cohort.smoothed_stress = clamp(
                cohort.smoothed_stress * 0.74
                    + (terrarium_snapshot.metabolic_load * 0.36
                        + (0.35 - terrarium_snapshot.oxygen_mm).max(0.0) * 0.72
                        + terrarium_snapshot
                            .local_chemistry
                            .unwrap_or_default()
                            .crowding_penalty
                            .max(1.0)
                        - 1.0)
                        * 0.26,
                0.0,
                1.6,
            );
            let growth_rate = if growth_signal >= 0.0 {
                EXPLICIT_MICROBE_GROWTH_RATE * growth_signal
            } else {
                -EXPLICIT_MICROBE_DECAY_RATE * growth_signal.abs()
            };
            cohort.represented_cells = (cohort.represented_cells
                * (1.0 + growth_rate * eco_dt * EXPLICIT_MICROBE_TIME_COMPRESSION))
                .clamp(0.0, EXPLICIT_MICROBE_MAX_REPRESENTED_CELLS);
            cohort.represented_packets = clamp(
                cohort.identity.represented_packets
                    + cohort.represented_cells / MICROBIAL_PACKET_TARGET_CELLS.max(1.0) * 0.18,
                1.0,
                GENOTYPE_PACKET_MAX_PER_CELL as f32,
            );
            cohort.radius = if cohort.represented_cells >= EXPLICIT_MICROBE_RADIUS_EXPAND_2_CELLS
                && cohort.smoothed_energy >= EXPLICIT_MICROBE_RADIUS_EXPAND_2_ENERGY
            {
                EXPLICIT_MICROBE_PATCH_RADIUS + 2
            } else if cohort.represented_cells >= EXPLICIT_MICROBE_RADIUS_EXPAND_1_CELLS
                && cohort.smoothed_energy >= EXPLICIT_MICROBE_RADIUS_EXPAND_1_ENERGY
            {
                EXPLICIT_MICROBE_PATCH_RADIUS + 1
            } else {
                EXPLICIT_MICROBE_PATCH_RADIUS
            };
            cohort.patch_radius = cohort.radius;
            terrarium_snapshot
        };
        let fluxes = self.compute_explicit_microbe_fluxes(idx, &updated_snapshot, eco_dt);
        let applied_fluxes = self.apply_explicit_microbe_material_fluxes(idx, &fluxes)?;
        if let Some(cohort) = self.explicit_microbes.get_mut(idx) {
            for (species, delta) in &applied_fluxes {
                match species {
                    TerrariumSpecies::Glucose if *delta < 0.0 => {
                        cohort.cumulative_glucose_draw += delta.abs();
                    }
                    TerrariumSpecies::OxygenGas if *delta < 0.0 => {
                        cohort.cumulative_oxygen_draw += delta.abs();
                    }
                    TerrariumSpecies::Ammonium if *delta < 0.0 => {
                        cohort.cumulative_ammonium_draw += delta.abs();
                    }
                    TerrariumSpecies::Nitrate if *delta < 0.0 => {
                        cohort.cumulative_nitrate_draw += delta.abs();
                    }
                    TerrariumSpecies::CarbonDioxide if *delta > 0.0 => {
                        cohort.cumulative_co2_release += *delta;
                    }
                    TerrariumSpecies::Proton if *delta > 0.0 => {
                        cohort.cumulative_proton_release += *delta;
                    }
                    _ => {}
                }
            }
        }
        let daughter_idx = self.try_divide_explicit_microbe(idx)?;
        self.bridge_explicit_to_coarse_packet(idx);
        if let Some(daughter_idx) = daughter_idx {
            self.bridge_explicit_to_coarse_packet(daughter_idx);
        }
        Ok(())
    }

    /// Bridge explicit microbe state back to coarse packet representation.
    pub(super) fn bridge_explicit_to_coarse_packet(&mut self, idx: usize) {
        if idx >= self.explicit_microbes.len() {
            return;
        }
        let (x, y, z, represented_cells, _represented_packets, guild, identity, energy, stress) = {
            let cohort = &self.explicit_microbes[idx];
            (
                cohort.x,
                cohort.y,
                cohort.z,
                cohort.represented_cells,
                cohort.represented_packets,
                guild_to_packet_ecology(cohort.guild),
                cohort.identity.clone(),
                cohort.smoothed_energy,
                cohort.smoothed_stress,
            )
        };
        let flat = idx2(self.config.width, x, y);
        let seed_entries = self.combined_packet_seed_entries_at(flat);
        let target_idx = if let Some(existing_idx) = self
            .packet_populations
            .iter()
            .position(|pop| pop.x == x && pop.y == y)
        {
            existing_idx
        } else {
            self.packet_populations
                .push(GenotypePacketPopulation::new(x, y, z));
            self.packet_populations.len() - 1
        };
        let bridged_activity = {
            let pop = &mut self.packet_populations[target_idx];
            pop.x = x;
            pop.y = y;
            pop.z = z;
            if pop.packets.is_empty() {
                if seed_entries.is_empty() {
                    pop.packets.push(GenotypePacket::new(
                        0,
                        identity.record.genotype_id,
                        identity.record.lineage_id,
                        guild,
                        represented_cells.max(1.0),
                    ));
                } else {
                    pop.seed_from_secondary_entries(&seed_entries, represented_cells.max(1.0));
                }
            }
            let dominant_idx = pop
                .packets
                .iter()
                .position(|packet| packet.genotype_id == identity.record.genotype_id)
                .unwrap_or(0);
            let dominant_share = (0.58 + energy * 0.20 - stress * 0.08).clamp(0.40, 0.88);
            let dominant_cells = represented_cells.max(1.0) * dominant_share;
            let remainder_cells = (represented_cells.max(1.0) - dominant_cells).max(0.0);
            let residual_slots = pop.packets.len().saturating_sub(1).max(1);

            for packet_idx in 0..pop.packets.len() {
                let packet = &mut pop.packets[packet_idx];
                if packet_idx == dominant_idx {
                    packet.catalog_slot = identity.bank_idx as u32;
                    packet.genotype_id = identity.record.genotype_id;
                    packet.lineage_id = identity.record.lineage_id;
                    packet.ecology = guild;
                    packet.represented_cells = dominant_cells;
                } else {
                    packet.represented_cells = remainder_cells / residual_slots as f32;
                }
                packet.activity = clamp(energy * (1.08 - stress * 0.24), 0.05, 1.0);
                packet.reserve = clamp(energy * 0.92 + 0.08, 0.0, 1.0);
                packet.dormancy = clamp(stress * 0.72, 0.0, 0.95);
                packet.damage = clamp(stress * 0.46, 0.0, 0.95);
            }
            pop.age_s = self.explicit_microbes[idx].age_s;
            pop.recompute_total();
            pop.mean_activity()
        };
        let _ = self.claim_ownership(
            x,
            y,
            SoilOwnershipClass::ExplicitMicrobeCohort {
                cohort_id: self.explicit_microbes[idx].idx as u32,
            },
            clamp(0.74 + energy * 0.12 - stress * 0.08, 0.55, 0.98),
        );
        self.explicit_microbe_activity[flat] =
            self.explicit_microbe_activity[flat].max(bridged_activity);
    }

    /// Step all explicit microbes (full iteration).
    pub(super) fn step_explicit_microbes(&mut self, eco_dt: f32) -> Result<(), String> {
        self.recruit_explicit_microbes_from_soil()?;
        if self.explicit_microbes.is_empty() {
            self.rebuild_explicit_microbe_fields();
            return Ok(());
        }
        let mut demoted = Vec::new();
        let mut dead = Vec::new();
        for idx in 0..self.explicit_microbes.len() {
            self.step_single_explicit_microbe(idx, eco_dt)?;
            if let Some(cohort) = self.explicit_microbes.get(idx) {
                if cohort.represented_cells < EXPLICIT_MICROBE_MIN_REPRESENTED_CELLS * 0.40
                    || (cohort.smoothed_energy < 0.08
                        && cohort.smoothed_stress > 1.05
                        && cohort.age_s > 4.0)
                {
                    if cohort.represented_cells > 1.0 {
                        demoted.push(idx);
                    } else {
                        dead.push(idx);
                    }
                }
            }
        }

        for idx in demoted.into_iter().rev() {
            self.demote_explicit_microbe_to_packet(idx)?;
        }

        for idx in dead.into_iter().rev() {
            if idx >= self.explicit_microbes.len() {
                continue;
            }
            let cohort = self.explicit_microbes.remove(idx);
            self.release_ownership(cohort.x, cohort.y);
            self.ecology_events
                .push(EcologyTelemetryEvent::ExplicitDeath {
                    x: cohort.x,
                    y: cohort.y,
                    z: cohort.z,
                    reason: "shoreline cohort collapse".into(),
                    represented_cells: cohort.represented_cells,
                    atp_mm: cohort.last_snapshot.atp_mm,
                    age_s: cohort.age_s,
                });
        }
        self.rebuild_explicit_microbe_fields();
        Ok(())
    }

    /// Step explicit microbes incrementally (budget-limited).
    pub(super) fn step_explicit_microbes_incremental(&mut self, eco_dt: f32) -> Result<(), String> {
        self.step_explicit_microbes(eco_dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn promoted_shoreline_world(seed: u64) -> TerrariumWorld {
        let mut world =
            TerrariumWorld::demo_preset(seed, false, TerrariumDemoPreset::Demo).unwrap();
        let (x, y, flat) = strongest_shoreline_cell(&world);
        for yy in y.saturating_sub(1)..=(y + 1).min(world.config.height.saturating_sub(1)) {
            for xx in x.saturating_sub(1)..=(x + 1).min(world.config.width.saturating_sub(1)) {
                let idx = idx2(world.config.width, xx, yy);
                world.microbial_packets[idx] = 80.0;
                world.nitrifier_packets[idx] = 20.0;
                world.denitrifier_packets[idx] = 24.0;
                world.microbial_cells[idx] = 40_000.0;
                world.nitrifier_cells[idx] = 16_000.0;
                world.denitrifier_cells[idx] = 18_000.0;
            }
        }
        world.microbial_packets[flat] = 2_450.0;
        world.nitrifier_packets[flat] = 1_280.0;
        world.denitrifier_packets[flat] = 1_310.0;
        world.microbial_cells[flat] = 1.15e6;
        world.nitrifier_cells[flat] = 5.1e5;
        world.denitrifier_cells[flat] = 5.3e5;
        world.microbial_vitality[flat] = 0.91;
        world.nitrifier_vitality[flat] = 0.84;
        world.denitrifier_vitality[flat] = 0.83;
        world.microbial_reserve[flat] = 0.83;
        world.nitrifier_reserve[flat] = 0.77;
        world.denitrifier_reserve[flat] = 0.75;
        world.microbial_packet_mutation_flux[flat] = 0.00018;
        world.nitrifier_packet_mutation_flux[flat] = 0.00015;
        world.denitrifier_packet_mutation_flux[flat] = 0.00016;
        world.nitrification_potential[flat] = 0.00011;
        world.denitrification_potential[flat] = 0.00011;
        world.moisture[flat] = 0.35;
        world.deep_moisture[flat] = 0.47;
        world.packet_populations.clear();
        world.explicit_microbes.clear();
        world.recruit_explicit_microbes_from_soil().unwrap();
        world
    }

    fn strongest_shoreline_cell(world: &TerrariumWorld) -> (usize, usize, usize) {
        let width = world.config.width;
        let height = world.config.height;
        let idx = (0..width * height)
            .filter(|idx| world.water_mask[*idx] < 0.40)
            .max_by(|a, b| {
                world
                    .shoreline_packet_recruitment_signal(*a)
                    .partial_cmp(&world.shoreline_packet_recruitment_signal(*b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("expected shoreline cell");
        (idx % width, idx / width, idx)
    }

    #[test]
    fn owned_inventory_projection_drives_explicit_inputs() {
        let mut world = promoted_shoreline_world(75);
        assert!(!world.explicit_microbes.is_empty());
        let cohort = &world.explicit_microbes[0];
        let (x, y, z, represented_cells) = (cohort.x, cohort.y, cohort.z, cohort.represented_cells);
        let baseline_target = world.explicit_microbe_material_inventory_target_from_patch(
            x,
            y,
            z,
            cohort.radius.max(1),
            represented_cells,
        );
        let baseline_material =
            baseline_target.estimate_whole_cell_environment_inputs(&inventory_projection_regions());
        let baseline_inputs =
            world.explicit_microbe_environment_inputs(x, y, z, &baseline_material);
        world.explicit_microbes[0]
            .material_inventory
            .set_component_amount(
                MaterialRegionKind::PoreWater,
                MoleculeGraph::representative_glucose(),
                MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous),
                0.0,
            )
            .unwrap();
        let owned_material = world.explicit_microbes[0]
            .material_inventory
            .estimate_whole_cell_environment_inputs(&inventory_projection_regions());
        let owned_inputs = world.explicit_microbe_environment_inputs(x, y, z, &owned_material);
        assert!(owned_inputs.glucose_mm < baseline_inputs.glucose_mm);
    }

    #[test]
    fn step_single_explicit_microbe_keeps_products_in_owned_inventory() {
        let mut world = promoted_shoreline_world(77);
        assert!(!world.explicit_microbes.is_empty());
        let cohort = &world.explicit_microbes[0];
        let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius.max(1));
        let patch_co2_before =
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, radius);
        let patch_proton_before =
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::Proton, x, y, z, radius);
        let patch_atp_before =
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::AtpFlux, x, y, z, radius);
        let eco_dt = world.config.world_dt_s * world.config.time_warp;
        world.step_single_explicit_microbe(0, eco_dt).unwrap();
        let patch_co2_after =
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, radius);
        let patch_proton_after =
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::Proton, x, y, z, radius);
        let patch_atp_after =
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::AtpFlux, x, y, z, radius);
        let inventory_co2 = inventory_component_amount(
            &world.explicit_microbes[0].material_inventory,
            TerrariumSpecies::CarbonDioxide,
        );
        let inventory_proton = inventory_component_amount(
            &world.explicit_microbes[0].material_inventory,
            TerrariumSpecies::Proton,
        );
        let inventory_atp = inventory_component_amount(
            &world.explicit_microbes[0].material_inventory,
            TerrariumSpecies::AtpFlux,
        );
        assert!(inventory_co2 >= 0.0 && inventory_proton >= 0.0 && inventory_atp >= 0.0);
        assert!((patch_co2_after - patch_co2_before).abs() <= 1.0e-6);
        assert!((patch_proton_after - patch_proton_before).abs() <= 1.0e-6);
        assert!((patch_atp_after - patch_atp_before).abs() <= 1.0e-6);
    }

    #[test]
    fn dividing_shoreline_cohort_places_daughter_and_splits_inventory() {
        let mut world = promoted_shoreline_world(79);
        assert!(!world.explicit_microbes.is_empty());
        world.explicit_microbes[0].represented_cells = EXPLICIT_MICROBE_MIN_REPRESENTED_CELLS * 8.0;
        world.explicit_microbes[0].represented_packets = 12.0;
        let rich_inputs = WholeCellEnvironmentInputs {
            glucose_mm: 8.0,
            oxygen_mm: 8.0,
            amino_acids_mm: 6.0,
            nucleotides_mm: 6.0,
            membrane_precursors_mm: 5.0,
            metabolic_load: 0.12,
            temperature_c: 22.0,
            proton_concentration: 1.0e-7,
        };
        for _ in 0..800 {
            let cohort = &mut world.explicit_microbes[0];
            cohort.simulator.apply_environment_inputs(&rich_inputs);
            cohort.simulator.step();
            if cohort.simulator.pending_division() {
                break;
            }
        }
        assert!(world.explicit_microbes[0].simulator.pending_division());
        let glucose_before = inventory_component_amount(
            &world.explicit_microbes[0].material_inventory,
            TerrariumSpecies::Glucose,
        );
        let daughter_idx = world
            .try_divide_explicit_microbe(0)
            .unwrap()
            .expect("expected shoreline daughter");
        assert_eq!(world.explicit_microbes.len(), 2);
        let parent = &world.explicit_microbes[0];
        let daughter = &world.explicit_microbes[daughter_idx];
        assert_ne!((parent.x, parent.y), (daughter.x, daughter.y));
        let glucose_after =
            inventory_component_amount(&parent.material_inventory, TerrariumSpecies::Glucose)
                + inventory_component_amount(
                    &daughter.material_inventory,
                    TerrariumSpecies::Glucose,
                );
        assert!((glucose_after - glucose_before).abs() <= 1.0e-4);
        assert!(daughter.represented_cells > 0.0);
        assert!(matches!(
            world.ownership[idx2(world.config.width, daughter.x, daughter.y)].owner,
            SoilOwnershipClass::ExplicitMicrobeCohort { .. }
        ));
    }

    #[test]
    fn demotion_spills_owned_inventory_back_to_environment() {
        let mut world = promoted_shoreline_world(81);
        assert!(!world.explicit_microbes.is_empty());
        let cohort = &world.explicit_microbes[0];
        let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius.max(1));
        world.explicit_microbes[0]
            .material_inventory
            .set_component_amount(
                MaterialRegionKind::PoreWater,
                MoleculeGraph::representative_glucose(),
                MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous),
                0.5,
            )
            .unwrap();
        let patch_before =
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, radius);
        world.demote_explicit_microbe_to_packet(0).unwrap();
        let patch_after =
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, radius);
        assert!(world.explicit_microbes.is_empty());
        assert!(patch_after > patch_before);
        assert!(world.packet_population_at(x, y).is_some());
    }

    #[test]
    fn shoreline_packet_regions_promote_into_explicit_cohorts() {
        let mut world =
            TerrariumWorld::demo_preset(71, false, TerrariumDemoPreset::Demo).unwrap();
        let (x, y, flat) = strongest_shoreline_cell(&world);
        for yy in y.saturating_sub(1)..=(y + 1).min(world.config.height.saturating_sub(1)) {
            for xx in x.saturating_sub(1)..=(x + 1).min(world.config.width.saturating_sub(1)) {
                let idx = idx2(world.config.width, xx, yy);
                world.microbial_packets[idx] = 80.0;
                world.nitrifier_packets[idx] = 20.0;
                world.denitrifier_packets[idx] = 24.0;
                world.microbial_cells[idx] = 40_000.0;
                world.nitrifier_cells[idx] = 16_000.0;
                world.denitrifier_cells[idx] = 18_000.0;
            }
        }
        world.microbial_packets[flat] = 2_300.0;
        world.nitrifier_packets[flat] = 1_350.0;
        world.denitrifier_packets[flat] = 1_420.0;
        world.microbial_cells[flat] = 1.1e6;
        world.nitrifier_cells[flat] = 5.4e5;
        world.denitrifier_cells[flat] = 5.8e5;
        world.microbial_vitality[flat] = 0.90;
        world.nitrifier_vitality[flat] = 0.86;
        world.denitrifier_vitality[flat] = 0.88;
        world.microbial_reserve[flat] = 0.82;
        world.nitrifier_reserve[flat] = 0.78;
        world.denitrifier_reserve[flat] = 0.80;
        world.microbial_packet_mutation_flux[flat] = 0.00018;
        world.nitrifier_packet_mutation_flux[flat] = 0.00015;
        world.denitrifier_packet_mutation_flux[flat] = 0.00016;
        world.nitrification_potential[flat] = 0.00011;
        world.denitrification_potential[flat] = 0.00012;
        world.moisture[flat] = 0.35;
        world.deep_moisture[flat] = 0.47;

        world.packet_populations.clear();
        world.explicit_microbes.clear();
        world.recruit_explicit_microbes_from_soil().unwrap();

        let cohort = world
            .explicit_microbes
            .iter()
            .find(|cohort| {
                cohort.x.abs_diff(x) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
                    && cohort.y.abs_diff(y) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
            })
            .expect("expected shoreline explicit cohort");
        let cohort_flat = idx2(world.config.width, cohort.x, cohort.y);
        assert!(matches!(
            world.ownership[cohort_flat].owner,
            SoilOwnershipClass::ExplicitMicrobeCohort { .. }
        ));
        assert!(cohort.represented_cells > EXPLICIT_MICROBE_MIN_REPRESENTED_CELLS);
        assert!(world.packet_population_at(cohort.x, cohort.y).is_some());
    }

    #[test]
    fn explicit_shoreline_cohorts_bridge_back_into_packet_summary() {
        let mut world =
            TerrariumWorld::demo_preset(73, false, TerrariumDemoPreset::Demo).unwrap();
        let (x, y, flat) = strongest_shoreline_cell(&world);
        for yy in y.saturating_sub(1)..=(y + 1).min(world.config.height.saturating_sub(1)) {
            for xx in x.saturating_sub(1)..=(x + 1).min(world.config.width.saturating_sub(1)) {
                let idx = idx2(world.config.width, xx, yy);
                world.microbial_packets[idx] = 80.0;
                world.nitrifier_packets[idx] = 20.0;
                world.denitrifier_packets[idx] = 24.0;
                world.microbial_cells[idx] = 40_000.0;
                world.nitrifier_cells[idx] = 16_000.0;
                world.denitrifier_cells[idx] = 18_000.0;
            }
        }
        world.microbial_packets[flat] = 2_600.0;
        world.nitrifier_packets[flat] = 1_200.0;
        world.denitrifier_packets[flat] = 1_050.0;
        world.microbial_cells[flat] = 1.2e6;
        world.nitrifier_cells[flat] = 4.9e5;
        world.denitrifier_cells[flat] = 4.6e5;
        world.microbial_vitality[flat] = 0.90;
        world.nitrifier_vitality[flat] = 0.82;
        world.denitrifier_vitality[flat] = 0.80;
        world.microbial_reserve[flat] = 0.84;
        world.nitrifier_reserve[flat] = 0.76;
        world.denitrifier_reserve[flat] = 0.74;
        world.microbial_packet_mutation_flux[flat] = 0.00018;
        world.nitrifier_packet_mutation_flux[flat] = 0.00015;
        world.denitrifier_packet_mutation_flux[flat] = 0.00016;
        world.nitrification_potential[flat] = 0.00010;
        world.denitrification_potential[flat] = 0.00010;
        world.moisture[flat] = 0.34;
        world.deep_moisture[flat] = 0.46;
        world.packet_populations.clear();
        world.explicit_microbes.clear();

        world.recruit_explicit_microbes_from_soil().unwrap();
        let eco_dt = world.config.world_dt_s * world.config.time_warp;
        world.step_explicit_microbes(eco_dt).unwrap();

        let cohort = world
            .explicit_microbes
            .iter()
            .find(|cohort| {
                cohort.x.abs_diff(x) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
                    && cohort.y.abs_diff(y) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
            })
            .expect("explicit cohort near shoreline hotspot");
        let cohort_flat = idx2(world.config.width, cohort.x, cohort.y);
        let pop = world
            .packet_population_at(cohort.x, cohort.y)
            .expect("explicit cohort should keep packet summary");
        assert!(matches!(
            world.ownership[cohort_flat].owner,
            SoilOwnershipClass::ExplicitMicrobeCohort { .. }
        ));
        assert!(pop.total_cells > 0.0);
        assert!(pop.mean_activity() > 0.0);
    }
}
