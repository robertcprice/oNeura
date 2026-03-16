// Snapshot serialization for TerrariumWorld.
use super::*;

impl super::TerrariumWorld {
    pub fn snapshot(&self) -> TerrariumWorldSnapshot {
        let live_fruits = self
            .fruits
            .iter()
            .filter(|fruit| fruit.source.alive && fruit.source.sugar_content > 0.01)
            .count();
        let food_remaining = self
            .fruits
            .iter()
            .filter(|fruit| fruit.source.alive)
            .map(|fruit| fruit.source.sugar_content.max(0.0))
            .sum::<f32>();
        let avg_fly_energy = if self.flies.is_empty() {
            0.0
        } else {
            self.flies
                .iter()
                .map(|fly| fly.body_state().energy)
                .sum::<f32>()
                / self.flies.len() as f32
        };
        let n_flies = self.flies.len() as f32;
        let avg_fly_hunger = if self.flies.is_empty() {
            0.0
        } else {
            self.flies.iter().map(|f| f.metabolism.hunger()).sum::<f32>() / n_flies
        };
        let avg_fly_energy_charge = if self.flies.is_empty() {
            0.0
        } else {
            use crate::organism_metabolism::OrganismMetabolism;
            self.flies.iter().map(|f| f.metabolism.energy_charge()).sum::<f32>() / n_flies
        };
        let avg_fly_trehalose_mm = if self.flies.is_empty() {
            0.0
        } else {
            self.flies.iter().map(|f| f.metabolism.hemolymph_trehalose_mm).sum::<f32>() / n_flies
        };
        let avg_fly_atp_mm = if self.flies.is_empty() {
            0.0
        } else {
            self.flies.iter().map(|f| f.metabolism.muscle_atp_mm).sum::<f32>() / n_flies
        };
        let avg_altitude = if self.flies.is_empty() {
            0.0
        } else {
            self.flies.iter().map(|fly| fly.body_state().z).sum::<f32>() / self.flies.len() as f32
        };
        let fly_census = self.fly_population.stage_census();
        let total_plant_cells = self
            .plants
            .iter()
            .map(|plant| plant.cellular.total_cells())
            .sum::<f32>();
        let mean_cell_vitality = if self.plants.is_empty() {
            0.0
        } else {
            self.plants
                .iter()
                .map(|plant| plant.cellular.vitality())
                .sum::<f32>()
                / self.plants.len() as f32
        };
        let mean_cell_energy = if self.plants.is_empty() {
            0.0
        } else {
            self.plants
                .iter()
                .map(|plant| plant.cellular.energy_charge())
                .sum::<f32>()
                / self.plants.len() as f32
        };
        let mean_division_pressure = if self.plants.is_empty() {
            0.0
        } else {
            self.plants
                .iter()
                .map(|plant| plant.cellular.division_signal())
                .sum::<f32>()
                / self.plants.len() as f32
        };

        let mean_soil_glucose = self.substrate.mean_species(TerrariumSpecies::Glucose);
        let mean_soil_oxygen = self.substrate.mean_species(TerrariumSpecies::OxygenGas);
        let mean_soil_ammonium = self.substrate.mean_species(TerrariumSpecies::Ammonium);
        let mean_soil_nitrate = self.substrate.mean_species(TerrariumSpecies::Nitrate);
        let mean_soil_redox = {
            let oxygen = mean_soil_oxygen;
            let nitrate = mean_soil_nitrate;
            let carbon_dioxide = self.substrate.mean_species(TerrariumSpecies::CarbonDioxide);
            let proton = self.substrate.mean_species(TerrariumSpecies::Proton);
            clamp(
                (oxygen * 1.3 + nitrate * 0.7)
                    / (oxygen * 1.3 + nitrate * 0.7 + carbon_dioxide * 0.35 + proton * 0.45 + 1e-9),
                0.0,
                1.0,
            )
        };
        let packet_fraction_mean = |typed: &[f32], total: &[f32]| -> f32 {
            if total.is_empty() {
                0.0
            } else {
                typed
                    .iter()
                    .zip(total.iter())
                    .map(|(typed, total)| clamp(*typed / total.max(0.05), 0.0, 1.0))
                    .sum::<f32>()
                    / total.len() as f32
            }
        };
        let latent_bank_mean =
            |banks: &[Vec<f32>]| -> f32 { banks.iter().map(|bank| mean(bank)).sum::<f32>() };
        let extended_bank_diversity =
            |total: &[f32], shadow: &[f32], variant: &[f32], novel: &[f32], latent: &[Vec<f32>]| {
                if total.is_empty() {
                    0.0
                } else {
                    total
                        .iter()
                        .enumerate()
                        .map(|(idx, total_packets)| {
                            let mut secondary = Vec::with_capacity(3 + latent.len());
                            secondary.extend([shadow[idx], variant[idx], novel[idx]]);
                            for bank in latent {
                                secondary.push(bank[idx]);
                            }
                            bank_simpson_diversity(*total_packets, &secondary)
                        })
                        .sum::<f32>()
                        / total.len() as f32
                }
            };
        let extended_weighted_bank_trait_mean =
            |total: &[f32],
             shadow: &[f32],
             variant: &[f32],
             novel: &[f32],
             primary: &[f32],
             shadow_trait: &[f32],
             variant_trait: &[f32],
             novel_trait: &[f32],
             latent_packets: &[Vec<f32>],
             latent_traits: &[Vec<f32>]| {
                if total.is_empty() {
                    0.0
                } else {
                    total
                        .iter()
                        .enumerate()
                        .map(|(idx, total_packets)| {
                            let mut secondary_packets =
                                Vec::with_capacity(3 + latent_packets.len());
                            let mut secondary_traits = Vec::with_capacity(3 + latent_traits.len());
                            secondary_packets.extend([shadow[idx], variant[idx], novel[idx]]);
                            secondary_traits.extend([
                                shadow_trait[idx],
                                variant_trait[idx],
                                novel_trait[idx],
                            ]);
                            for bank in latent_packets {
                                secondary_packets.push(bank[idx]);
                            }
                            for trait_bank in latent_traits {
                                secondary_traits.push(trait_bank[idx]);
                            }
                            bank_weighted_trait_mean(
                                *total_packets,
                                primary[idx],
                                &secondary_packets,
                                &secondary_traits,
                            )
                        })
                        .sum::<f32>()
                        / total.len() as f32
                }
            };
        let public_weighted_catalog_gene_module_mean =
            |shadow: &[f32],
             variant: &[f32],
             novel: &[f32],
             shadow_slots: &[u32],
             variant_slots: &[u32],
             novel_slots: &[u32],
             shadow_catalog: &[SecondaryCatalogBankEntry],
             variant_catalog: &[SecondaryCatalogBankEntry],
             novel_catalog: &[SecondaryCatalogBankEntry],
             weights: &[f32; INTERNAL_SECONDARY_GENOTYPE_AXES]| {
                if shadow.is_empty() {
                    0.0
                } else {
                    shadow
                        .iter()
                        .enumerate()
                        .map(|(idx, shadow_packets)| {
                            let secondary_total =
                                (shadow_packets + variant[idx] + novel[idx]).max(1.0e-6);
                            let shadow_module = shadow_catalog
                                .get(shadow_slots[idx] as usize)
                                .map(|entry| decode_secondary_gene_module(&entry.genes, weights))
                                .unwrap_or(0.0);
                            let variant_module = variant_catalog
                                .get(variant_slots[idx] as usize)
                                .map(|entry| decode_secondary_gene_module(&entry.genes, weights))
                                .unwrap_or(0.0);
                            let novel_module = novel_catalog
                                .get(novel_slots[idx] as usize)
                                .map(|entry| decode_secondary_gene_module(&entry.genes, weights))
                                .unwrap_or(0.0);
                            (shadow_module * shadow_packets.max(0.0)
                                + variant_module * variant[idx].max(0.0)
                                + novel_module * novel[idx].max(0.0))
                                / secondary_total
                        })
                        .sum::<f32>()
                        / shadow.len() as f32
                }
            };
        let public_weighted_catalog_record_mean =
            |shadow: &[f32],
             variant: &[f32],
             novel: &[f32],
             shadow_slots: &[u32],
             variant_slots: &[u32],
             novel_slots: &[u32],
             shadow_catalog: &[SecondaryCatalogBankEntry],
             variant_catalog: &[SecondaryCatalogBankEntry],
             novel_catalog: &[SecondaryCatalogBankEntry],
             accessor: fn(&SecondaryGenotypeRecord) -> f32| {
                if shadow.is_empty() {
                    0.0
                } else {
                    shadow
                        .iter()
                        .enumerate()
                        .map(|(idx, shadow_packets)| {
                            let secondary_total =
                                (shadow_packets + variant[idx] + novel[idx]).max(1.0e-6);
                            let shadow_record = shadow_catalog
                                .get(shadow_slots[idx] as usize)
                                .map(|entry| entry.record)
                                .unwrap_or_default();
                            let variant_record = variant_catalog
                                .get(variant_slots[idx] as usize)
                                .map(|entry| entry.record)
                                .unwrap_or_default();
                            let novel_record = novel_catalog
                                .get(novel_slots[idx] as usize)
                                .map(|entry| entry.record)
                                .unwrap_or_default();
                            (accessor(&shadow_record) * shadow_packets.max(0.0)
                                + accessor(&variant_record) * variant[idx].max(0.0)
                                + accessor(&novel_record) * novel[idx].max(0.0))
                                / secondary_total
                        })
                        .sum::<f32>()
                        / shadow.len() as f32
                }
            };
        let public_weighted_catalog_metadata_mean =
            |shadow: &[f32],
             variant: &[f32],
             novel: &[f32],
             shadow_slots: &[u32],
             variant_slots: &[u32],
             novel_slots: &[u32],
             shadow_catalog: &[SecondaryCatalogBankEntry],
             variant_catalog: &[SecondaryCatalogBankEntry],
             novel_catalog: &[SecondaryCatalogBankEntry],
             accessor: fn(&SecondaryGenotypeCatalogRecord) -> f32| {
                if shadow.is_empty() {
                    0.0
                } else {
                    shadow
                        .iter()
                        .enumerate()
                        .map(|(idx, shadow_packets)| {
                            let secondary_total =
                                (shadow_packets + variant[idx] + novel[idx]).max(1.0e-6);
                            let shadow_record = shadow_catalog
                                .get(shadow_slots[idx] as usize)
                                .map(|entry| entry.catalog)
                                .unwrap_or_default();
                            let variant_record = variant_catalog
                                .get(variant_slots[idx] as usize)
                                .map(|entry| entry.catalog)
                                .unwrap_or_default();
                            let novel_record = novel_catalog
                                .get(novel_slots[idx] as usize)
                                .map(|entry| entry.catalog)
                                .unwrap_or_default();
                            (accessor(&shadow_record) * shadow_packets.max(0.0)
                                + accessor(&variant_record) * variant[idx].max(0.0)
                                + accessor(&novel_record) * novel[idx].max(0.0))
                                / secondary_total
                        })
                        .sum::<f32>()
                        / shadow.len() as f32
                }
            };
        let public_catalog_bank_dominance_mean =
            |shadow_catalog: &[SecondaryCatalogBankEntry],
             variant_catalog: &[SecondaryCatalogBankEntry],
             novel_catalog: &[SecondaryCatalogBankEntry]| {
                let bank_dominance = |entries: &[SecondaryCatalogBankEntry]| {
                    entries
                        .iter()
                        .map(|entry| entry.catalog.local_bank_share)
                        .fold(0.0, f32::max)
                };
                (bank_dominance(shadow_catalog)
                    + bank_dominance(variant_catalog)
                    + bank_dominance(novel_catalog))
                    / PUBLIC_STRAIN_BANKS as f32
            };
        let public_catalog_bank_richness_mean =
            |shadow_catalog: &[SecondaryCatalogBankEntry],
             variant_catalog: &[SecondaryCatalogBankEntry],
             novel_catalog: &[SecondaryCatalogBankEntry]| {
                let bank_richness = |entries: &[SecondaryCatalogBankEntry]| {
                    entries
                        .iter()
                        .filter(|entry| entry.catalog.local_bank_share >= 1.0e-3)
                        .count() as f32
                };
                (bank_richness(shadow_catalog)
                    + bank_richness(variant_catalog)
                    + bank_richness(novel_catalog))
                    / PUBLIC_STRAIN_BANKS as f32
            };
        let mean_microbial_packet_load = if self.microbial_cells.is_empty() {
            0.0
        } else {
            self.microbial_cells
                .iter()
                .zip(self.microbial_packets.iter())
                .map(|(cells, packets)| packet_load(*cells, *packets))
                .sum::<f32>()
                / self.microbial_cells.len() as f32
        };
        let mean_microbial_oligotroph_packets = if self.microbial_packets.is_empty() {
            0.0
        } else {
            self.microbial_packets
                .iter()
                .zip(self.microbial_copiotroph_packets.iter())
                .map(|(total, typed)| (total - typed).max(0.0))
                .sum::<f32>()
                / self.microbial_packets.len() as f32
        };
        let microbial_secondary_packets = self.microbial_secondary.packets_banks();
        let microbial_secondary_trait_a = self.microbial_secondary.trait_a_banks();
        let microbial_secondary_trait_b = self.microbial_secondary.trait_b_banks();
        let mean_microbial_shadow_fraction = packet_fraction_mean(
            microbial_secondary_packets[SHADOW_BANK_IDX],
            &self.microbial_packets,
        );
        let mean_microbial_variant_fraction = packet_fraction_mean(
            microbial_secondary_packets[VARIANT_BANK_IDX],
            &self.microbial_packets,
        );
        let mean_microbial_novel_fraction = packet_fraction_mean(
            microbial_secondary_packets[NOVEL_BANK_IDX],
            &self.microbial_packets,
        );
        let mean_microbial_latent_packets = latent_bank_mean(&self.microbial_latent_packets);
        let mean_microbial_bank_simpson_diversity = extended_bank_diversity(
            &self.microbial_packets,
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            &self.microbial_latent_packets,
        );
        let mean_microbial_strain_yield = extended_weighted_bank_trait_mean(
            &self.microbial_packets,
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            &self.microbial_strain_yield,
            microbial_secondary_trait_a[SHADOW_BANK_IDX],
            microbial_secondary_trait_a[VARIANT_BANK_IDX],
            microbial_secondary_trait_a[NOVEL_BANK_IDX],
            &self.microbial_latent_packets,
            &self.microbial_latent_strain_yield,
        );
        let mean_microbial_strain_stress_tolerance = extended_weighted_bank_trait_mean(
            &self.microbial_packets,
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            &self.microbial_strain_stress_tolerance,
            microbial_secondary_trait_b[SHADOW_BANK_IDX],
            microbial_secondary_trait_b[VARIANT_BANK_IDX],
            microbial_secondary_trait_b[NOVEL_BANK_IDX],
            &self.microbial_latent_packets,
            &self.microbial_latent_strain_stress_tolerance,
        );
        let microbial_catalog_slots = self.microbial_secondary.catalog_slot_banks();
        let microbial_catalog_bank_entries = self.microbial_secondary.catalog_bank_entries();
        let mean_microbial_gene_catabolic = public_weighted_catalog_gene_module_mean(
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            microbial_catalog_slots[SHADOW_BANK_IDX],
            microbial_catalog_slots[VARIANT_BANK_IDX],
            microbial_catalog_slots[NOVEL_BANK_IDX],
            microbial_catalog_bank_entries[SHADOW_BANK_IDX],
            microbial_catalog_bank_entries[VARIANT_BANK_IDX],
            microbial_catalog_bank_entries[NOVEL_BANK_IDX],
            &MICROBIAL_GENE_CATABOLIC_WEIGHTS,
        );
        let mean_microbial_gene_stress_response = public_weighted_catalog_gene_module_mean(
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            microbial_catalog_slots[SHADOW_BANK_IDX],
            microbial_catalog_slots[VARIANT_BANK_IDX],
            microbial_catalog_slots[NOVEL_BANK_IDX],
            microbial_catalog_bank_entries[SHADOW_BANK_IDX],
            microbial_catalog_bank_entries[VARIANT_BANK_IDX],
            microbial_catalog_bank_entries[NOVEL_BANK_IDX],
            &MICROBIAL_GENE_STRESS_RESPONSE_WEIGHTS,
        );
        let mean_microbial_gene_dormancy_maintenance = public_weighted_catalog_gene_module_mean(
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            microbial_catalog_slots[SHADOW_BANK_IDX],
            microbial_catalog_slots[VARIANT_BANK_IDX],
            microbial_catalog_slots[NOVEL_BANK_IDX],
            microbial_catalog_bank_entries[SHADOW_BANK_IDX],
            microbial_catalog_bank_entries[VARIANT_BANK_IDX],
            microbial_catalog_bank_entries[NOVEL_BANK_IDX],
            &MICROBIAL_GENE_DORMANCY_MAINTENANCE_WEIGHTS,
        );
        let mean_microbial_gene_extracellular_scavenging = public_weighted_catalog_gene_module_mean(
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            microbial_catalog_slots[SHADOW_BANK_IDX],
            microbial_catalog_slots[VARIANT_BANK_IDX],
            microbial_catalog_slots[NOVEL_BANK_IDX],
            microbial_catalog_bank_entries[SHADOW_BANK_IDX],
            microbial_catalog_bank_entries[VARIANT_BANK_IDX],
            microbial_catalog_bank_entries[NOVEL_BANK_IDX],
            &MICROBIAL_GENE_EXTRACELLULAR_SCAVENGING_WEIGHTS,
        );
        let mean_microbial_genotype_divergence = public_weighted_catalog_record_mean(
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            microbial_catalog_slots[SHADOW_BANK_IDX],
            microbial_catalog_slots[VARIANT_BANK_IDX],
            microbial_catalog_slots[NOVEL_BANK_IDX],
            microbial_catalog_bank_entries[SHADOW_BANK_IDX],
            microbial_catalog_bank_entries[VARIANT_BANK_IDX],
            microbial_catalog_bank_entries[NOVEL_BANK_IDX],
            |record| record.genotype_divergence,
        );
        let mean_microbial_catalog_generation = public_weighted_catalog_metadata_mean(
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            microbial_catalog_slots[SHADOW_BANK_IDX],
            microbial_catalog_slots[VARIANT_BANK_IDX],
            microbial_catalog_slots[NOVEL_BANK_IDX],
            microbial_catalog_bank_entries[SHADOW_BANK_IDX],
            microbial_catalog_bank_entries[VARIANT_BANK_IDX],
            microbial_catalog_bank_entries[NOVEL_BANK_IDX],
            |catalog| catalog.generation,
        );
        let mean_microbial_catalog_novelty = public_weighted_catalog_metadata_mean(
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            microbial_catalog_slots[SHADOW_BANK_IDX],
            microbial_catalog_slots[VARIANT_BANK_IDX],
            microbial_catalog_slots[NOVEL_BANK_IDX],
            microbial_catalog_bank_entries[SHADOW_BANK_IDX],
            microbial_catalog_bank_entries[VARIANT_BANK_IDX],
            microbial_catalog_bank_entries[NOVEL_BANK_IDX],
            |catalog| catalog.novelty,
        );
        let mean_microbial_local_catalog_share = public_weighted_catalog_metadata_mean(
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            microbial_catalog_slots[SHADOW_BANK_IDX],
            microbial_catalog_slots[VARIANT_BANK_IDX],
            microbial_catalog_slots[NOVEL_BANK_IDX],
            microbial_catalog_bank_entries[SHADOW_BANK_IDX],
            microbial_catalog_bank_entries[VARIANT_BANK_IDX],
            microbial_catalog_bank_entries[NOVEL_BANK_IDX],
            |catalog| catalog.local_bank_share,
        );
        let mean_microbial_catalog_bank_dominance = public_catalog_bank_dominance_mean(
            microbial_catalog_bank_entries[SHADOW_BANK_IDX],
            microbial_catalog_bank_entries[VARIANT_BANK_IDX],
            microbial_catalog_bank_entries[NOVEL_BANK_IDX],
        );
        let mean_microbial_catalog_bank_richness = public_catalog_bank_richness_mean(
            microbial_catalog_bank_entries[SHADOW_BANK_IDX],
            microbial_catalog_bank_entries[VARIANT_BANK_IDX],
            microbial_catalog_bank_entries[NOVEL_BANK_IDX],
        );
        let mean_microbial_lineage_generation = public_weighted_catalog_record_mean(
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            microbial_catalog_slots[SHADOW_BANK_IDX],
            microbial_catalog_slots[VARIANT_BANK_IDX],
            microbial_catalog_slots[NOVEL_BANK_IDX],
            microbial_catalog_bank_entries[SHADOW_BANK_IDX],
            microbial_catalog_bank_entries[VARIANT_BANK_IDX],
            microbial_catalog_bank_entries[NOVEL_BANK_IDX],
            |record| record.generation,
        );
        let mean_microbial_lineage_novelty = public_weighted_catalog_record_mean(
            microbial_secondary_packets[SHADOW_BANK_IDX],
            microbial_secondary_packets[VARIANT_BANK_IDX],
            microbial_secondary_packets[NOVEL_BANK_IDX],
            microbial_catalog_slots[SHADOW_BANK_IDX],
            microbial_catalog_slots[VARIANT_BANK_IDX],
            microbial_catalog_slots[NOVEL_BANK_IDX],
            microbial_catalog_bank_entries[SHADOW_BANK_IDX],
            microbial_catalog_bank_entries[VARIANT_BANK_IDX],
            microbial_catalog_bank_entries[NOVEL_BANK_IDX],
            |record| record.novelty,
        );
        let mean_nitrifier_packet_load = if self.nitrifier_cells.is_empty() {
            0.0
        } else {
            self.nitrifier_cells
                .iter()
                .zip(self.nitrifier_packets.iter())
                .map(|(cells, packets)| packet_load(*cells, *packets))
                .sum::<f32>()
                / self.nitrifier_cells.len() as f32
        };
        let nitrifier_secondary_packets = self.nitrifier_secondary.packets_banks();
        let nitrifier_secondary_trait_a = self.nitrifier_secondary.trait_a_banks();
        let nitrifier_secondary_trait_b = self.nitrifier_secondary.trait_b_banks();
        let mean_nitrifier_shadow_fraction = packet_fraction_mean(
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            &self.nitrifier_packets,
        );
        let mean_nitrifier_variant_fraction = packet_fraction_mean(
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            &self.nitrifier_packets,
        );
        let mean_nitrifier_novel_fraction = packet_fraction_mean(
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            &self.nitrifier_packets,
        );
        let mean_nitrifier_latent_packets = latent_bank_mean(&self.nitrifier_latent_packets);
        let mean_nitrifier_bank_simpson_diversity = extended_bank_diversity(
            &self.nitrifier_packets,
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            &self.nitrifier_latent_packets,
        );
        let mean_nitrifier_strain_oxygen_affinity = extended_weighted_bank_trait_mean(
            &self.nitrifier_packets,
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            &self.nitrifier_strain_oxygen_affinity,
            nitrifier_secondary_trait_a[SHADOW_BANK_IDX],
            nitrifier_secondary_trait_a[VARIANT_BANK_IDX],
            nitrifier_secondary_trait_a[NOVEL_BANK_IDX],
            &self.nitrifier_latent_packets,
            &self.nitrifier_latent_strain_oxygen_affinity,
        );
        let mean_nitrifier_strain_ammonium_affinity = extended_weighted_bank_trait_mean(
            &self.nitrifier_packets,
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            &self.nitrifier_strain_ammonium_affinity,
            nitrifier_secondary_trait_b[SHADOW_BANK_IDX],
            nitrifier_secondary_trait_b[VARIANT_BANK_IDX],
            nitrifier_secondary_trait_b[NOVEL_BANK_IDX],
            &self.nitrifier_latent_packets,
            &self.nitrifier_latent_strain_ammonium_affinity,
        );
        let nitrifier_catalog_slots = self.nitrifier_secondary.catalog_slot_banks();
        let nitrifier_catalog_bank_entries = self.nitrifier_secondary.catalog_bank_entries();
        let mean_nitrifier_gene_oxygen_respiration = public_weighted_catalog_gene_module_mean(
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            nitrifier_catalog_slots[SHADOW_BANK_IDX],
            nitrifier_catalog_slots[VARIANT_BANK_IDX],
            nitrifier_catalog_slots[NOVEL_BANK_IDX],
            nitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            nitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            nitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            &NITRIFIER_GENE_OXYGEN_RESPIRATION_WEIGHTS,
        );
        let mean_nitrifier_gene_ammonium_transport = public_weighted_catalog_gene_module_mean(
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            nitrifier_catalog_slots[SHADOW_BANK_IDX],
            nitrifier_catalog_slots[VARIANT_BANK_IDX],
            nitrifier_catalog_slots[NOVEL_BANK_IDX],
            nitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            nitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            nitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            &NITRIFIER_GENE_AMMONIUM_TRANSPORT_WEIGHTS,
        );
        let mean_nitrifier_gene_stress_persistence = public_weighted_catalog_gene_module_mean(
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            nitrifier_catalog_slots[SHADOW_BANK_IDX],
            nitrifier_catalog_slots[VARIANT_BANK_IDX],
            nitrifier_catalog_slots[NOVEL_BANK_IDX],
            nitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            nitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            nitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            &NITRIFIER_GENE_STRESS_PERSISTENCE_WEIGHTS,
        );
        let mean_nitrifier_gene_redox_efficiency = public_weighted_catalog_gene_module_mean(
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            nitrifier_catalog_slots[SHADOW_BANK_IDX],
            nitrifier_catalog_slots[VARIANT_BANK_IDX],
            nitrifier_catalog_slots[NOVEL_BANK_IDX],
            nitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            nitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            nitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            &NITRIFIER_GENE_REDOX_EFFICIENCY_WEIGHTS,
        );
        let mean_nitrifier_genotype_divergence = public_weighted_catalog_record_mean(
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            nitrifier_catalog_slots[SHADOW_BANK_IDX],
            nitrifier_catalog_slots[VARIANT_BANK_IDX],
            nitrifier_catalog_slots[NOVEL_BANK_IDX],
            nitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            nitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            nitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            |record| record.genotype_divergence,
        );
        let mean_nitrifier_catalog_generation = public_weighted_catalog_metadata_mean(
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            nitrifier_catalog_slots[SHADOW_BANK_IDX],
            nitrifier_catalog_slots[VARIANT_BANK_IDX],
            nitrifier_catalog_slots[NOVEL_BANK_IDX],
            nitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            nitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            nitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            |catalog| catalog.generation,
        );
        let mean_nitrifier_catalog_novelty = public_weighted_catalog_metadata_mean(
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            nitrifier_catalog_slots[SHADOW_BANK_IDX],
            nitrifier_catalog_slots[VARIANT_BANK_IDX],
            nitrifier_catalog_slots[NOVEL_BANK_IDX],
            nitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            nitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            nitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            |catalog| catalog.novelty,
        );
        let mean_nitrifier_local_catalog_share = public_weighted_catalog_metadata_mean(
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            nitrifier_catalog_slots[SHADOW_BANK_IDX],
            nitrifier_catalog_slots[VARIANT_BANK_IDX],
            nitrifier_catalog_slots[NOVEL_BANK_IDX],
            nitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            nitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            nitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            |catalog| catalog.local_bank_share,
        );
        let mean_nitrifier_catalog_bank_dominance = public_catalog_bank_dominance_mean(
            nitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            nitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            nitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
        );
        let mean_nitrifier_catalog_bank_richness = public_catalog_bank_richness_mean(
            nitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            nitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            nitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
        );
        let mean_nitrifier_lineage_generation = public_weighted_catalog_record_mean(
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            nitrifier_catalog_slots[SHADOW_BANK_IDX],
            nitrifier_catalog_slots[VARIANT_BANK_IDX],
            nitrifier_catalog_slots[NOVEL_BANK_IDX],
            nitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            nitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            nitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            |record| record.generation,
        );
        let mean_nitrifier_lineage_novelty = public_weighted_catalog_record_mean(
            nitrifier_secondary_packets[SHADOW_BANK_IDX],
            nitrifier_secondary_packets[VARIANT_BANK_IDX],
            nitrifier_secondary_packets[NOVEL_BANK_IDX],
            nitrifier_catalog_slots[SHADOW_BANK_IDX],
            nitrifier_catalog_slots[VARIANT_BANK_IDX],
            nitrifier_catalog_slots[NOVEL_BANK_IDX],
            nitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            nitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            nitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            |record| record.novelty,
        );
        let mean_nitrifier_facultative_packets = if self.nitrifier_packets.is_empty() {
            0.0
        } else {
            self.nitrifier_packets
                .iter()
                .zip(self.nitrifier_aerobic_packets.iter())
                .map(|(total, typed)| (total - typed).max(0.0))
                .sum::<f32>()
                / self.nitrifier_packets.len() as f32
        };
        let mean_denitrifier_packet_load = if self.denitrifier_cells.is_empty() {
            0.0
        } else {
            self.denitrifier_cells
                .iter()
                .zip(self.denitrifier_packets.iter())
                .map(|(cells, packets)| packet_load(*cells, *packets))
                .sum::<f32>()
                / self.denitrifier_cells.len() as f32
        };
        let mean_denitrifier_facultative_packets = if self.denitrifier_packets.is_empty() {
            0.0
        } else {
            self.denitrifier_packets
                .iter()
                .zip(self.denitrifier_anoxic_packets.iter())
                .map(|(total, typed)| (total - typed).max(0.0))
                .sum::<f32>()
                / self.denitrifier_packets.len() as f32
        };
        let denitrifier_secondary_packets = self.denitrifier_secondary.packets_banks();
        let denitrifier_secondary_trait_a = self.denitrifier_secondary.trait_a_banks();
        let denitrifier_secondary_trait_b = self.denitrifier_secondary.trait_b_banks();
        let mean_denitrifier_shadow_fraction = packet_fraction_mean(
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            &self.denitrifier_packets,
        );
        let mean_denitrifier_variant_fraction = packet_fraction_mean(
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            &self.denitrifier_packets,
        );
        let mean_denitrifier_novel_fraction = packet_fraction_mean(
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            &self.denitrifier_packets,
        );
        let mean_denitrifier_latent_packets = latent_bank_mean(&self.denitrifier_latent_packets);
        let mean_denitrifier_bank_simpson_diversity = extended_bank_diversity(
            &self.denitrifier_packets,
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            &self.denitrifier_latent_packets,
        );
        let mean_denitrifier_strain_anoxia_affinity = extended_weighted_bank_trait_mean(
            &self.denitrifier_packets,
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            &self.denitrifier_strain_anoxia_affinity,
            denitrifier_secondary_trait_a[SHADOW_BANK_IDX],
            denitrifier_secondary_trait_a[VARIANT_BANK_IDX],
            denitrifier_secondary_trait_a[NOVEL_BANK_IDX],
            &self.denitrifier_latent_packets,
            &self.denitrifier_latent_strain_anoxia_affinity,
        );
        let mean_denitrifier_strain_nitrate_affinity = extended_weighted_bank_trait_mean(
            &self.denitrifier_packets,
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            &self.denitrifier_strain_nitrate_affinity,
            denitrifier_secondary_trait_b[SHADOW_BANK_IDX],
            denitrifier_secondary_trait_b[VARIANT_BANK_IDX],
            denitrifier_secondary_trait_b[NOVEL_BANK_IDX],
            &self.denitrifier_latent_packets,
            &self.denitrifier_latent_strain_nitrate_affinity,
        );
        let denitrifier_catalog_slots = self.denitrifier_secondary.catalog_slot_banks();
        let denitrifier_catalog_bank_entries = self.denitrifier_secondary.catalog_bank_entries();
        let mean_denitrifier_gene_anoxia_respiration = public_weighted_catalog_gene_module_mean(
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            denitrifier_catalog_slots[SHADOW_BANK_IDX],
            denitrifier_catalog_slots[VARIANT_BANK_IDX],
            denitrifier_catalog_slots[NOVEL_BANK_IDX],
            denitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            denitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            denitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            &DENITRIFIER_GENE_ANOXIA_RESPIRATION_WEIGHTS,
        );
        let mean_denitrifier_gene_nitrate_transport = public_weighted_catalog_gene_module_mean(
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            denitrifier_catalog_slots[SHADOW_BANK_IDX],
            denitrifier_catalog_slots[VARIANT_BANK_IDX],
            denitrifier_catalog_slots[NOVEL_BANK_IDX],
            denitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            denitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            denitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            &DENITRIFIER_GENE_NITRATE_TRANSPORT_WEIGHTS,
        );
        let mean_denitrifier_gene_stress_persistence = public_weighted_catalog_gene_module_mean(
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            denitrifier_catalog_slots[SHADOW_BANK_IDX],
            denitrifier_catalog_slots[VARIANT_BANK_IDX],
            denitrifier_catalog_slots[NOVEL_BANK_IDX],
            denitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            denitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            denitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            &DENITRIFIER_GENE_STRESS_PERSISTENCE_WEIGHTS,
        );
        let mean_denitrifier_gene_reductive_flexibility = public_weighted_catalog_gene_module_mean(
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            denitrifier_catalog_slots[SHADOW_BANK_IDX],
            denitrifier_catalog_slots[VARIANT_BANK_IDX],
            denitrifier_catalog_slots[NOVEL_BANK_IDX],
            denitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            denitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            denitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            &DENITRIFIER_GENE_REDUCTIVE_FLEXIBILITY_WEIGHTS,
        );
        let mean_denitrifier_genotype_divergence = public_weighted_catalog_record_mean(
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            denitrifier_catalog_slots[SHADOW_BANK_IDX],
            denitrifier_catalog_slots[VARIANT_BANK_IDX],
            denitrifier_catalog_slots[NOVEL_BANK_IDX],
            denitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            denitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            denitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            |record| record.genotype_divergence,
        );
        let mean_denitrifier_catalog_generation = public_weighted_catalog_metadata_mean(
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            denitrifier_catalog_slots[SHADOW_BANK_IDX],
            denitrifier_catalog_slots[VARIANT_BANK_IDX],
            denitrifier_catalog_slots[NOVEL_BANK_IDX],
            denitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            denitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            denitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            |catalog| catalog.generation,
        );
        let mean_denitrifier_catalog_novelty = public_weighted_catalog_metadata_mean(
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            denitrifier_catalog_slots[SHADOW_BANK_IDX],
            denitrifier_catalog_slots[VARIANT_BANK_IDX],
            denitrifier_catalog_slots[NOVEL_BANK_IDX],
            denitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            denitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            denitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            |catalog| catalog.novelty,
        );
        let mean_denitrifier_local_catalog_share = public_weighted_catalog_metadata_mean(
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            denitrifier_catalog_slots[SHADOW_BANK_IDX],
            denitrifier_catalog_slots[VARIANT_BANK_IDX],
            denitrifier_catalog_slots[NOVEL_BANK_IDX],
            denitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            denitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            denitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            |catalog| catalog.local_bank_share,
        );
        let mean_denitrifier_catalog_bank_dominance = public_catalog_bank_dominance_mean(
            denitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            denitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            denitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
        );
        let mean_denitrifier_catalog_bank_richness = public_catalog_bank_richness_mean(
            denitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            denitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            denitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
        );
        let mean_denitrifier_lineage_generation = public_weighted_catalog_record_mean(
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            denitrifier_catalog_slots[SHADOW_BANK_IDX],
            denitrifier_catalog_slots[VARIANT_BANK_IDX],
            denitrifier_catalog_slots[NOVEL_BANK_IDX],
            denitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            denitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            denitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            |record| record.generation,
        );
        let mean_denitrifier_lineage_novelty = public_weighted_catalog_record_mean(
            denitrifier_secondary_packets[SHADOW_BANK_IDX],
            denitrifier_secondary_packets[VARIANT_BANK_IDX],
            denitrifier_secondary_packets[NOVEL_BANK_IDX],
            denitrifier_catalog_slots[SHADOW_BANK_IDX],
            denitrifier_catalog_slots[VARIANT_BANK_IDX],
            denitrifier_catalog_slots[NOVEL_BANK_IDX],
            denitrifier_catalog_bank_entries[SHADOW_BANK_IDX],
            denitrifier_catalog_bank_entries[VARIANT_BANK_IDX],
            denitrifier_catalog_bank_entries[NOVEL_BANK_IDX],
            |record| record.novelty,
        );
        let explicit_microbes = self.explicit_microbes.len();
        let explicit_microbe_represented_cells = self
            .explicit_microbes
            .iter()
            .map(|microbe| microbe.represented_cells.max(0.0))
            .sum::<f32>();
        let explicit_microbe_represented_packets = self
            .explicit_microbes
            .iter()
            .map(|microbe| microbe.represented_packets.max(0.0))
            .sum::<f32>();
        let explicit_microbe_owned_fraction = if self.explicit_microbe_authority.is_empty() {
            0.0
        } else {
            self.explicit_microbe_authority
                .iter()
                .filter(|value| **value > 1.0e-4)
                .count() as f32
                / self.explicit_microbe_authority.len() as f32
        };
        let explicit_microbe_max_authority = self
            .explicit_microbe_authority
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let mean_explicit_microbe_activity = mean(&self.explicit_microbe_activity);
        let explicit_microbe_count = explicit_microbes.max(1) as f32;
        let mean_explicit_microbe_atp_mm = if explicit_microbes > 0 {
            self.explicit_microbes
                .iter()
                .map(|microbe| microbe.last_snapshot.atp_mm.max(0.0))
                .sum::<f32>()
                / explicit_microbe_count
        } else {
            0.0
        };
        let mean_explicit_microbe_glucose_mm = if explicit_microbes > 0 {
            self.explicit_microbes
                .iter()
                .map(|microbe| microbe.last_snapshot.glucose_mm.max(0.0))
                .sum::<f32>()
                / explicit_microbe_count
        } else {
            0.0
        };
        let mean_explicit_microbe_oxygen_mm = if explicit_microbes > 0 {
            self.explicit_microbes
                .iter()
                .map(|microbe| microbe.last_snapshot.oxygen_mm.max(0.0))
                .sum::<f32>()
                / explicit_microbe_count
        } else {
            0.0
        };
        let mean_explicit_microbe_division_progress = if explicit_microbes > 0 {
            self.explicit_microbes
                .iter()
                .map(|microbe| microbe.last_snapshot.division_progress.max(0.0))
                .sum::<f32>()
                / explicit_microbe_count
        } else {
            0.0
        };
        let mean_explicit_microbe_local_co2 = if explicit_microbes > 0 {
            self.explicit_microbes
                .iter()
                .map(|microbe| {
                    microbe
                        .last_snapshot
                        .local_chemistry
                        .as_ref()
                        .map(|chemistry| chemistry.mean_carbon_dioxide.max(0.0))
                        .unwrap_or(0.0)
                })
                .sum::<f32>()
                / explicit_microbe_count
        } else {
            0.0
        };
        let mean_explicit_microbe_translation_support = if explicit_microbes > 0 {
            self.explicit_microbes
                .iter()
                .map(|microbe| {
                    microbe
                        .last_snapshot
                        .local_chemistry
                        .as_ref()
                        .map(|chemistry| chemistry.translation_support.max(0.0))
                        .unwrap_or(0.0)
                })
                .sum::<f32>()
                / explicit_microbe_count
        } else {
            0.0
        };
        let mean_explicit_microbe_energy_state = if explicit_microbes > 0 {
            self.explicit_microbes
                .iter()
                .map(|microbe| microbe.smoothed_energy.max(0.0))
                .sum::<f32>()
                / explicit_microbe_count
        } else {
            0.0
        };
        let mean_explicit_microbe_stress_state = if explicit_microbes > 0 {
            self.explicit_microbes
                .iter()
                .map(|microbe| microbe.smoothed_stress.max(0.0))
                .sum::<f32>()
                / explicit_microbe_count
        } else {
            0.0
        };
        let mean_explicit_microbe_genotype_divergence = if explicit_microbes > 0 {
            self.explicit_microbes
                .iter()
                .map(|microbe| microbe.identity.record.genotype_divergence.clamp(0.0, 1.0))
                .sum::<f32>()
                / explicit_microbe_count
        } else {
            0.0
        };
        let mean_explicit_microbe_catalog_novelty = if explicit_microbes > 0 {
            self.explicit_microbes
                .iter()
                .map(|microbe| microbe.identity.catalog.novelty.clamp(0.0, 1.0))
                .sum::<f32>()
                / explicit_microbe_count
        } else {
            0.0
        };
        let mean_explicit_microbe_local_catalog_share = if explicit_microbes > 0 {
            self.explicit_microbes
                .iter()
                .map(|microbe| microbe.identity.catalog.local_bank_share.clamp(0.0, 1.0))
                .sum::<f32>()
                / explicit_microbe_count
        } else {
            0.0
        };

        TerrariumWorldSnapshot {
            plants: self.plants.len(),
            fruits: live_fruits,
            seeds: self.seeds.len(),
            flies: self.flies.len(),
            food_remaining,
            fly_food_total: self.fly_food_total,
            avg_fly_energy,
            avg_fly_hunger,
            avg_fly_energy_charge,
            avg_fly_trehalose_mm,
            avg_fly_atp_mm,
            avg_altitude,
            fly_population_eggs: fly_census.total_eggs,
            fly_population_embryos: fly_census.embryos,
            fly_population_larvae: fly_census.larvae,
            fly_population_pupae: fly_census.pupae,
            fly_population_adults: fly_census.adults,
            fly_population_total: fly_census.total_individuals(),
            light: self.daylight(),
            temperature: mean(&self.temperature),
            humidity: mean(&self.humidity),
            mean_air_pressure_kpa: mean(&self.air_pressure_kpa),
            mean_soil_moisture: mean(&self.moisture),
            mean_deep_moisture: mean(&self.deep_moisture),
            mean_microbes: mean(&self.microbial_biomass),
            mean_microbial_cells: mean(&self.microbial_cells),
            mean_microbial_packets: mean(&self.microbial_packets),
            mean_microbial_copiotroph_packets: mean(&self.microbial_copiotroph_packets),
            mean_microbial_shadow_packets: mean(microbial_secondary_packets[SHADOW_BANK_IDX]),
            mean_microbial_variant_packets: mean(microbial_secondary_packets[VARIANT_BANK_IDX]),
            mean_microbial_novel_packets: mean(microbial_secondary_packets[NOVEL_BANK_IDX]),
            mean_microbial_latent_packets,
            mean_microbial_oligotroph_packets,
            mean_microbial_packet_load,
            mean_microbial_copiotroph_fraction: mean(&self.microbial_copiotroph_fraction),
            mean_microbial_shadow_fraction,
            mean_microbial_variant_fraction,
            mean_microbial_novel_fraction,
            mean_microbial_bank_simpson_diversity,
            mean_microbial_strain_yield,
            mean_microbial_strain_stress_tolerance,
            mean_microbial_gene_catabolic,
            mean_microbial_gene_stress_response,
            mean_microbial_gene_dormancy_maintenance,
            mean_microbial_gene_extracellular_scavenging,
            mean_microbial_genotype_divergence,
            mean_microbial_catalog_generation,
            mean_microbial_catalog_novelty,
            mean_microbial_local_catalog_share,
            mean_microbial_catalog_bank_dominance,
            mean_microbial_catalog_bank_richness,
            mean_microbial_lineage_generation,
            mean_microbial_lineage_novelty,
            mean_microbial_packet_mutation_flux: mean(&self.microbial_packet_mutation_flux),
            mean_microbial_vitality: mean(&self.microbial_vitality),
            mean_microbial_dormancy: mean(&self.microbial_dormancy),
            mean_microbial_reserve: mean(&self.microbial_reserve),
            mean_symbionts: mean(&self.symbiont_biomass),
            mean_nitrifiers: mean(&self.nitrifier_biomass),
            mean_nitrifier_cells: mean(&self.nitrifier_cells),
            mean_nitrifier_packets: mean(&self.nitrifier_packets),
            mean_nitrifier_aerobic_packets: mean(&self.nitrifier_aerobic_packets),
            mean_nitrifier_shadow_packets: mean(nitrifier_secondary_packets[SHADOW_BANK_IDX]),
            mean_nitrifier_variant_packets: mean(nitrifier_secondary_packets[VARIANT_BANK_IDX]),
            mean_nitrifier_novel_packets: mean(nitrifier_secondary_packets[NOVEL_BANK_IDX]),
            mean_nitrifier_latent_packets,
            mean_nitrifier_facultative_packets,
            mean_nitrifier_packet_load,
            mean_nitrifier_aerobic_fraction: mean(&self.nitrifier_aerobic_fraction),
            mean_nitrifier_shadow_fraction,
            mean_nitrifier_variant_fraction,
            mean_nitrifier_novel_fraction,
            mean_nitrifier_bank_simpson_diversity,
            mean_nitrifier_strain_oxygen_affinity,
            mean_nitrifier_strain_ammonium_affinity,
            mean_nitrifier_gene_oxygen_respiration,
            mean_nitrifier_gene_ammonium_transport,
            mean_nitrifier_gene_stress_persistence,
            mean_nitrifier_gene_redox_efficiency,
            mean_nitrifier_genotype_divergence,
            mean_nitrifier_catalog_generation,
            mean_nitrifier_catalog_novelty,
            mean_nitrifier_local_catalog_share,
            mean_nitrifier_catalog_bank_dominance,
            mean_nitrifier_catalog_bank_richness,
            mean_nitrifier_lineage_generation,
            mean_nitrifier_lineage_novelty,
            mean_nitrifier_packet_mutation_flux: mean(&self.nitrifier_packet_mutation_flux),
            mean_nitrifier_vitality: mean(&self.nitrifier_vitality),
            mean_nitrifier_dormancy: mean(&self.nitrifier_dormancy),
            mean_nitrifier_reserve: mean(&self.nitrifier_reserve),
            mean_denitrifiers: mean(&self.denitrifier_biomass),
            mean_denitrifier_cells: mean(&self.denitrifier_cells),
            mean_denitrifier_packets: mean(&self.denitrifier_packets),
            mean_denitrifier_anoxic_packets: mean(&self.denitrifier_anoxic_packets),
            mean_denitrifier_shadow_packets: mean(denitrifier_secondary_packets[SHADOW_BANK_IDX]),
            mean_denitrifier_variant_packets: mean(denitrifier_secondary_packets[VARIANT_BANK_IDX]),
            mean_denitrifier_novel_packets: mean(denitrifier_secondary_packets[NOVEL_BANK_IDX]),
            mean_denitrifier_latent_packets,
            mean_denitrifier_facultative_packets,
            mean_denitrifier_packet_load,
            mean_denitrifier_anoxic_fraction: mean(&self.denitrifier_anoxic_fraction),
            mean_denitrifier_shadow_fraction,
            mean_denitrifier_variant_fraction,
            mean_denitrifier_novel_fraction,
            mean_denitrifier_bank_simpson_diversity,
            mean_denitrifier_strain_anoxia_affinity,
            mean_denitrifier_strain_nitrate_affinity,
            mean_denitrifier_gene_anoxia_respiration,
            mean_denitrifier_gene_nitrate_transport,
            mean_denitrifier_gene_stress_persistence,
            mean_denitrifier_gene_reductive_flexibility,
            mean_denitrifier_genotype_divergence,
            mean_denitrifier_catalog_generation,
            mean_denitrifier_catalog_novelty,
            mean_denitrifier_local_catalog_share,
            mean_denitrifier_catalog_bank_dominance,
            mean_denitrifier_catalog_bank_richness,
            mean_denitrifier_lineage_generation,
            mean_denitrifier_lineage_novelty,
            mean_denitrifier_packet_mutation_flux: mean(&self.denitrifier_packet_mutation_flux),
            mean_denitrifier_vitality: mean(&self.denitrifier_vitality),
            mean_denitrifier_dormancy: mean(&self.denitrifier_dormancy),
            mean_denitrifier_reserve: mean(&self.denitrifier_reserve),
            mean_nitrification_potential: mean(&self.nitrification_potential),
            mean_denitrification_potential: mean(&self.denitrification_potential),
            mean_canopy: mean(&self.canopy_cover),
            mean_root_density: mean(&self.root_density),
            total_plant_cells,
            mean_cell_vitality,
            mean_cell_energy,
            mean_division_pressure,
            mean_soil_glucose,
            mean_soil_oxygen,
            mean_soil_ammonium,
            mean_soil_nitrate,
            mean_soil_redox,
            mean_soil_atp_flux: self.substrate.mean_species(TerrariumSpecies::AtpFlux),
            mean_atmospheric_co2: mean(&self.odorants[ATMOS_CO2_IDX]),
            mean_atmospheric_o2: mean(&self.odorants[ATMOS_O2_IDX]),
            explicit_microbes,
            explicit_microbe_represented_cells,
            explicit_microbe_represented_packets,
            explicit_microbe_owned_fraction,
            explicit_microbe_max_authority,
            mean_explicit_microbe_activity,
            mean_explicit_microbe_atp_mm,
            mean_explicit_microbe_glucose_mm,
            mean_explicit_microbe_oxygen_mm,
            mean_explicit_microbe_division_progress,
            mean_explicit_microbe_local_co2,
            mean_explicit_microbe_translation_support,
            mean_explicit_microbe_energy_state,
            mean_explicit_microbe_stress_state,
            mean_explicit_microbe_genotype_divergence,
            mean_explicit_microbe_catalog_novelty,
            mean_explicit_microbe_local_catalog_share,
            // Phase 4: packet population metrics
            packet_population_count: self.packet_populations.len(),
            packet_population_total_cells: self
                .packet_populations
                .iter()
                .map(|pop| pop.total_cells)
                .sum(),
            packet_population_mean_activity: if self.packet_populations.is_empty() {
                0.0
            } else {
                self.packet_populations
                    .iter()
                    .map(|pop| pop.mean_activity())
                    .sum::<f32>()
                    / self.packet_populations.len() as f32
            },
            packet_population_mean_dormancy: if self.packet_populations.is_empty() {
                0.0
            } else {
                self.packet_populations
                    .iter()
                    .map(|pop| pop.mean_dormancy())
                    .sum::<f32>()
                    / self.packet_populations.len() as f32
            },
            packet_population_total_packets: self
                .packet_populations
                .iter()
                .map(|pop| pop.packets.len())
                .sum(),
            packet_population_promotion_candidates: self
                .packet_populations
                .iter()
                .map(|pop| pop.promotion_candidates())
                .sum(),
            substrate_backend: self.substrate.backend().as_str().to_string(),
            substrate_steps: self.substrate.step_count(),
            substrate_time_ms: self.substrate.time_ms(),
            time_s: self.time_s,
            plant_species_count: {
                let mut ids: Vec<u32> = self.plants.iter().map(|p| p.genome.species_id).collect();
                ids.sort_unstable();
                ids.dedup();
                ids.len() as u32
            },
            plant_species_ids: {
                let mut ids: Vec<u32> = self.plants.iter().map(|p| p.genome.species_id).collect();
                ids.sort_unstable();
                ids.dedup();
                ids
            },
            ecology_events: self.ecology_events.clone(),
        }
    }
}
