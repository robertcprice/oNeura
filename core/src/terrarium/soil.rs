use super::genotype::{
    secondary_bank_catalog_divergence, secondary_bank_catalog_signature, PublicSecondaryBanks,
    SecondaryCatalogBankEntry, SecondaryGenotypeCatalogRecord, SecondaryGenotypeRecord,
    PUBLIC_STRAIN_BANKS,
};
use super::packet::GenotypePacketEcology;
use super::*;

const SHORELINE_PACKET_RECRUITMENT_THRESHOLD: f32 = 0.62;
const SHORELINE_PACKET_SUSTAIN_THRESHOLD: f32 = 0.36;

fn normalize_bank_shares(mut shares: [f32; PUBLIC_STRAIN_BANKS]) -> [f32; PUBLIC_STRAIN_BANKS] {
    let total = shares.iter().sum::<f32>().max(1.0e-6);
    for share in &mut shares {
        *share = (*share / total).clamp(0.0, 1.0);
    }
    shares
}

fn emergent_bank_shares(
    mutation_flux: f32,
    niche_shift: f32,
    vitality: f32,
) -> [f32; PUBLIC_STRAIN_BANKS] {
    let mutation_drive = (mutation_flux * 4200.0).clamp(0.0, 1.0);
    let variant = (0.16 + mutation_drive * 0.24 + niche_shift * 0.16).clamp(0.08, 0.36);
    let novel = (0.05 + mutation_drive * 0.18 + niche_shift * 0.14 + (1.0 - vitality) * 0.08)
        .clamp(0.03, 0.24);
    let shadow = (1.0 - variant - novel).clamp(0.22, 0.86);
    normalize_bank_shares([shadow, variant, novel])
}

fn ecology_catalog_seed(ecology: GenotypePacketEcology) -> usize {
    match ecology {
        GenotypePacketEcology::Decomposer => 11,
        GenotypePacketEcology::Nitrifier => 23,
        GenotypePacketEcology::Denitrifier => 37,
    }
}

fn ecology_gene_profile(
    ecology: GenotypePacketEcology,
    trait_a: f32,
    trait_b: f32,
    resource: f32,
    moisture: f32,
    oxygen_bias: f32,
    anoxia_bias: f32,
    shoreline: f32,
    mutation_flux: f32,
) -> [f32; 6] {
    let mutation_drive = (mutation_flux * 4200.0).clamp(0.0, 1.0);
    match ecology {
        GenotypePacketEcology::Decomposer => [
            clamp(
                0.18 + resource * 0.44 + trait_a * 0.22 + shoreline * 0.08,
                0.0,
                1.0,
            ),
            clamp(
                0.12 + mutation_drive * 0.32 + (1.0 - oxygen_bias) * 0.20 + shoreline * 0.08,
                0.0,
                1.0,
            ),
            clamp(
                0.18 + (1.0 - resource) * 0.16 + moisture * 0.14 + trait_b * 0.26,
                0.0,
                1.0,
            ),
            clamp(
                0.16 + resource * 0.26 + shoreline * 0.20 + trait_b * 0.20,
                0.0,
                1.0,
            ),
            clamp(
                0.14 + moisture * 0.30 + shoreline * 0.12 + mutation_drive * 0.10,
                0.0,
                1.0,
            ),
            clamp(
                0.14 + trait_b * 0.30 + mutation_drive * 0.16 + anoxia_bias * 0.10,
                0.0,
                1.0,
            ),
        ],
        GenotypePacketEcology::Nitrifier => [
            clamp(
                0.18 + oxygen_bias * 0.52 + trait_a * 0.18 + resource * 0.08,
                0.0,
                1.0,
            ),
            clamp(
                0.18 + resource * 0.46 + trait_b * 0.18 + shoreline * 0.06,
                0.0,
                1.0,
            ),
            clamp(
                0.12 + mutation_drive * 0.26 + (1.0 - moisture) * 0.12 + trait_b * 0.18,
                0.0,
                1.0,
            ),
            clamp(
                0.16 + oxygen_bias * 0.28 + trait_a * 0.22 + shoreline * 0.10,
                0.0,
                1.0,
            ),
            clamp(
                0.10 + moisture * 0.14 + shoreline * 0.16 + resource * 0.18,
                0.0,
                1.0,
            ),
            clamp(
                0.12 + trait_b * 0.24 + mutation_drive * 0.22 + resource * 0.10,
                0.0,
                1.0,
            ),
        ],
        GenotypePacketEcology::Denitrifier => [
            clamp(
                0.18 + anoxia_bias * 0.52 + shoreline * 0.14 + resource * 0.08,
                0.0,
                1.0,
            ),
            clamp(
                0.18 + resource * 0.46 + trait_a * 0.16 + shoreline * 0.08,
                0.0,
                1.0,
            ),
            clamp(
                0.12 + mutation_drive * 0.28 + moisture * 0.14 + trait_b * 0.18,
                0.0,
                1.0,
            ),
            clamp(
                0.18 + anoxia_bias * 0.28 + shoreline * 0.22 + trait_a * 0.14,
                0.0,
                1.0,
            ),
            clamp(
                0.12 + moisture * 0.20 + shoreline * 0.18 + resource * 0.14,
                0.0,
                1.0,
            ),
            clamp(
                0.14 + trait_b * 0.26 + mutation_drive * 0.18 + anoxia_bias * 0.12,
                0.0,
                1.0,
            ),
        ],
    }
}

fn perturb_ecology_genes(
    base_genes: [f32; 6],
    bank_idx: usize,
    mutation_flux: f32,
    niche_shift: f32,
) -> [f32; 6] {
    if bank_idx == 0 {
        return base_genes;
    }
    let spread = (mutation_flux * 4800.0).clamp(0.0, 1.0) * (0.05 + niche_shift * 0.10);
    std::array::from_fn(|axis_idx| {
        let axis_dir = match (bank_idx + axis_idx) % 3 {
            0 => -1.0,
            1 => 0.45,
            _ => 1.0,
        };
        let axis_scale = match bank_idx {
            1 => 0.45 + axis_idx as f32 * 0.06,
            _ => 0.78 + axis_idx as f32 * 0.08,
        };
        clamp(
            base_genes[axis_idx] + spread * axis_scale * axis_dir,
            0.0,
            1.0,
        )
    })
}

fn write_emergent_secondary_bank_cell(
    secondary: &mut PublicSecondaryBanks,
    flat: usize,
    ecology: GenotypePacketEcology,
    total_packets: f32,
    trait_a: f32,
    trait_b: f32,
    vitality: f32,
    mutation_flux: f32,
    niche_shift: f32,
    base_genes: [f32; 6],
) {
    let bank_count = secondary.banks.len().min(PUBLIC_STRAIN_BANKS);
    if bank_count == 0 {
        return;
    }
    if total_packets <= 0.02 {
        for bank in secondary.banks.iter_mut().take(bank_count) {
            if let Some(cell_packets) = bank.packets.get_mut(flat) {
                *cell_packets = 0.0;
            }
        }
        return;
    }

    let shares = emergent_bank_shares(mutation_flux, niche_shift, vitality);
    let ecology_seed = ecology_catalog_seed(ecology);
    for bank_idx in 0..bank_count {
        let bank = &mut secondary.banks[bank_idx];
        let packet_share = shares[bank_idx];
        let packet_mass = total_packets * packet_share;
        if flat >= bank.packets.len()
            || flat >= bank.catalog_slots.len()
            || bank.catalog_bank.is_empty()
        {
            continue;
        }

        bank.packets[flat] = packet_mass;
        bank.trait_a[flat] = clamp(
            trait_a + (bank_idx as f32 - 1.0) * mutation_flux * 1800.0 * 0.02,
            0.0,
            1.0,
        );
        bank.trait_b[flat] = clamp(
            trait_b + (bank_idx as f32 - 1.0) * niche_shift * 0.06,
            0.0,
            1.0,
        );

        let genes = perturb_ecology_genes(base_genes, bank_idx, mutation_flux, niche_shift);
        let signature_bank = ecology_seed + bank_idx;
        let catalog_id = secondary_bank_catalog_signature(signature_bank, &genes);
        let divergence = secondary_bank_catalog_divergence(&genes);
        let slot = (catalog_id as usize) % bank.catalog_bank.len();
        let previous = bank.catalog_bank[slot];
        let lineage_id =
            if previous.catalog.catalog_id == catalog_id && previous.record.lineage_id != 0 {
                previous.record.lineage_id
            } else {
                secondary_bank_catalog_signature(signature_bank + 97, &genes)
            };
        let generation = if previous.catalog.catalog_id == catalog_id {
            (previous.record.generation + 0.5 + mutation_flux * 4000.0).min(10_000.0)
        } else {
            (0.5 + mutation_flux * 2200.0 + niche_shift * 8.0).min(10_000.0)
        };
        let novelty = if previous.catalog.catalog_id == catalog_id {
            clamp(
                previous.record.novelty * 0.84 + mutation_flux * 900.0 * 0.16,
                0.0,
                1.0,
            )
        } else {
            clamp(0.12 + mutation_flux * 1500.0 + niche_shift * 0.18, 0.0, 1.0)
        };
        bank.catalog_slots[flat] = slot as u32;
        bank.catalog_bank[slot] = SecondaryCatalogBankEntry {
            occupancy: (packet_mass / 12.0).round().clamp(1.0, 4096.0) as u32,
            packet_mass: packet_share,
            record: SecondaryGenotypeRecord {
                genes,
                genotype_divergence: divergence,
                generation,
                novelty,
                genotype_id: catalog_id
                    ^ (((ecology_seed as u32) + 1) << 24)
                    ^ (((bank_idx as u32) + 1) << 20),
                lineage_id,
            },
            catalog: SecondaryGenotypeCatalogRecord {
                catalog_id,
                parent_catalog_id: if previous.catalog.catalog_id != 0 {
                    previous.catalog.catalog_id
                } else {
                    catalog_id
                },
                catalog_divergence: divergence,
                generation,
                novelty,
                local_bank_id: bank_idx as u32,
                local_bank_share: packet_share,
            },
            genes,
        };
    }
}

impl TerrariumWorld {
    pub fn broad_biology_factor(&self, x: usize, y: usize) -> f32 {
        let w = self.config.width;
        let h = self.config.height;
        if x >= w || y >= h {
            return 1.0;
        }
        let cell = self.ownership[y * w + x];
        1.0 - cell.strength.clamp(0.0, 1.0)
    }

    pub fn claim_ownership(
        &mut self,
        x: usize,
        y: usize,
        owner: SoilOwnershipClass,
        strength: f32,
    ) -> bool {
        let w = self.config.width;
        let h = self.config.height;
        if x >= w || y >= h {
            return false;
        }
        let cell = &mut self.ownership[y * w + x];
        let allow_lower_scale_upgrade = matches!(
            (cell.owner, owner),
            (
                SoilOwnershipClass::GenotypePacketRegion { .. },
                SoilOwnershipClass::ExplicitMicrobeCohort { .. }
            )
        );
        if cell.owner.is_explicit() && cell.owner != owner && !allow_lower_scale_upgrade {
            return false;
        }
        cell.owner = owner;
        cell.strength = strength.clamp(0.0, 1.0);
        true
    }

    pub fn release_ownership(&mut self, x: usize, y: usize) {
        let w = self.config.width;
        let h = self.config.height;
        if x < w && y < h {
            self.ownership[y * w + x] = SoilOwnershipCell::default();
        }
    }

    pub fn rebuild_ownership_diagnostics(&mut self) {
        let mut diag = OwnershipDiagnostics::default();
        let total = self.ownership.len() as f32;
        if total <= 0.0 {
            return;
        }

        for cell in &self.ownership {
            if cell.owner.is_explicit() {
                diag.owned_fraction += 1.0 / total;
                diag.max_strength = diag.max_strength.max(cell.strength);
                match cell.owner {
                    SoilOwnershipClass::ExplicitMicrobeCohort { .. } => {
                        diag.microbe_owned_fraction += 1.0 / total
                    }
                    SoilOwnershipClass::GenotypePacketRegion { .. } => {
                        diag.genotype_owned_fraction += 1.0 / total
                    }
                    SoilOwnershipClass::PlantTissueRegion { .. } => {
                        diag.plant_owned_fraction += 1.0 / total
                    }
                    SoilOwnershipClass::AtomisticProbeRegion { .. } => {
                        diag.probe_owned_fraction += 1.0 / total
                    }
                    _ => {}
                }
            }
        }
        self.ownership_diagnostics = diag;
    }

    pub fn ownership_diagnostics(&self) -> OwnershipDiagnostics {
        self.ownership_diagnostics
    }

    pub fn clear_ownership(&mut self) {
        for cell in &mut self.ownership {
            *cell = SoilOwnershipCell::default();
        }
        self.ownership_diagnostics = OwnershipDiagnostics::default();
    }

    pub fn moisture_field(&self) -> &[f32] {
        &self.moisture
    }

    pub fn moisture_field_mut(&mut self) -> &mut [f32] {
        &mut self.moisture
    }

    pub fn temperature_field(&self) -> &[f32] {
        &self.temperature
    }

    pub fn temperature_field_mut(&mut self) -> &mut [f32] {
        &mut self.temperature
    }

    pub(crate) fn packet_population_at(
        &self,
        x: usize,
        y: usize,
    ) -> Option<&crate::terrarium::packet::GenotypePacketPopulation> {
        self.packet_populations
            .iter()
            .find(|pop| pop.x == x && pop.y == y)
    }

    fn materialize_emergent_packet_banks(&mut self) {
        let width = self.config.width;
        let height = self.config.height;
        if width == 0 || height == 0 {
            return;
        }

        for y in 0..height {
            for x in 0..width {
                let flat = idx2(width, x, y);
                let shoreline = shoreline_water_signal(width, height, &self.water_mask, flat);
                let open_water = self.water_mask[flat].clamp(0.0, 1.0);
                let oxygen_bias =
                    (1.0 - self.deep_moisture[flat] * 0.62 - open_water * 0.24).clamp(0.05, 1.0);
                let anoxia_bias = clamp(
                    self.deep_moisture[flat] * 0.86 + shoreline * 0.30 + open_water * 0.28
                        - oxygen_bias * 0.42,
                    0.0,
                    1.0,
                );

                let decomposer_resource = clamp(
                    (self.litter_carbon[flat] * 1.10
                        + self.root_exudates[flat] * 1.35
                        + self.organic_matter[flat] * 0.90
                        + self.microbial_biomass[flat] * 0.18)
                        / 0.18,
                    0.0,
                    1.6,
                );
                let nitrifier_resource = clamp(
                    (self.nitrification_potential[flat] / 0.00025) * 0.76
                        + self.nitrifier_biomass[flat] * 0.18
                        + oxygen_bias * 0.12,
                    0.0,
                    1.4,
                );
                let denitrifier_resource = clamp(
                    (self.denitrification_potential[flat] / 0.00025) * 0.78
                        + self.denitrifier_biomass[flat] * 0.18
                        + anoxia_bias * 0.12,
                    0.0,
                    1.4,
                );

                let decomposer_trait_a = self.microbial_copiotroph_fraction[flat];
                let decomposer_trait_b = clamp(
                    self.microbial_reserve[flat] * 0.58
                        + (1.0 - self.microbial_dormancy[flat]).clamp(0.0, 1.0) * 0.42,
                    0.0,
                    1.0,
                );
                let nitrifier_trait_a = self.nitrifier_aerobic_fraction[flat];
                let nitrifier_trait_b = clamp(
                    self.nitrifier_reserve[flat] * 0.54 + self.nitrifier_vitality[flat] * 0.46,
                    0.0,
                    1.0,
                );
                let denitrifier_trait_a = self.denitrifier_anoxic_fraction[flat];
                let denitrifier_trait_b = clamp(
                    self.denitrifier_reserve[flat] * 0.54 + self.denitrifier_vitality[flat] * 0.46,
                    0.0,
                    1.0,
                );

                let decomposer_shift = clamp(
                    shoreline * 0.32
                        + open_water * 0.12
                        + self.microbial_packet_mutation_flux[flat] * 2200.0
                        + (self.moisture[flat] - self.deep_moisture[flat]).abs() * 0.18,
                    0.0,
                    1.0,
                );
                let nitrifier_shift = clamp(
                    shoreline * 0.18
                        + self.nitrification_potential[flat] / 0.00025 * 0.34
                        + self.nitrifier_packet_mutation_flux[flat] * 2400.0
                        + oxygen_bias * 0.14,
                    0.0,
                    1.0,
                );
                let denitrifier_shift = clamp(
                    shoreline * 0.30
                        + self.denitrification_potential[flat] / 0.00025 * 0.36
                        + self.denitrifier_packet_mutation_flux[flat] * 2400.0
                        + anoxia_bias * 0.14,
                    0.0,
                    1.0,
                );

                write_emergent_secondary_bank_cell(
                    &mut self.microbial_secondary,
                    flat,
                    GenotypePacketEcology::Decomposer,
                    self.microbial_packets[flat],
                    decomposer_trait_a,
                    decomposer_trait_b,
                    self.microbial_vitality[flat].clamp(0.0, 1.0),
                    self.microbial_packet_mutation_flux[flat],
                    decomposer_shift,
                    ecology_gene_profile(
                        GenotypePacketEcology::Decomposer,
                        decomposer_trait_a,
                        decomposer_trait_b,
                        decomposer_resource,
                        self.moisture[flat],
                        oxygen_bias,
                        anoxia_bias,
                        shoreline,
                        self.microbial_packet_mutation_flux[flat],
                    ),
                );
                write_emergent_secondary_bank_cell(
                    &mut self.nitrifier_secondary,
                    flat,
                    GenotypePacketEcology::Nitrifier,
                    self.nitrifier_packets[flat],
                    nitrifier_trait_a,
                    nitrifier_trait_b,
                    self.nitrifier_vitality[flat].clamp(0.0, 1.0),
                    self.nitrifier_packet_mutation_flux[flat],
                    nitrifier_shift,
                    ecology_gene_profile(
                        GenotypePacketEcology::Nitrifier,
                        nitrifier_trait_a,
                        nitrifier_trait_b,
                        nitrifier_resource,
                        self.moisture[flat],
                        oxygen_bias,
                        anoxia_bias,
                        shoreline,
                        self.nitrifier_packet_mutation_flux[flat],
                    ),
                );
                write_emergent_secondary_bank_cell(
                    &mut self.denitrifier_secondary,
                    flat,
                    GenotypePacketEcology::Denitrifier,
                    self.denitrifier_packets[flat],
                    denitrifier_trait_a,
                    denitrifier_trait_b,
                    self.denitrifier_vitality[flat].clamp(0.0, 1.0),
                    self.denitrifier_packet_mutation_flux[flat],
                    denitrifier_shift,
                    ecology_gene_profile(
                        GenotypePacketEcology::Denitrifier,
                        denitrifier_trait_a,
                        denitrifier_trait_b,
                        denitrifier_resource,
                        self.moisture[flat],
                        oxygen_bias,
                        anoxia_bias,
                        shoreline,
                        self.denitrifier_packet_mutation_flux[flat],
                    ),
                );
            }
        }
    }

    pub(super) fn combined_packet_seed_entries_at(
        &self,
        flat: usize,
    ) -> Vec<(SecondaryCatalogBankEntry, GenotypePacketEcology)> {
        let mut weighted = Vec::with_capacity(PUBLIC_STRAIN_BANKS * 3);
        for (secondary, ecology) in [
            (&self.microbial_secondary, GenotypePacketEcology::Decomposer),
            (&self.nitrifier_secondary, GenotypePacketEcology::Nitrifier),
            (
                &self.denitrifier_secondary,
                GenotypePacketEcology::Denitrifier,
            ),
        ] {
            for bank_idx in 0..secondary.banks.len().min(PUBLIC_STRAIN_BANKS) {
                let bank = &secondary.banks[bank_idx];
                let packets = bank.packets.get(flat).copied().unwrap_or(0.0).max(0.0);
                if packets <= 0.01 {
                    continue;
                }
                let slot = bank.catalog_slots.get(flat).copied().unwrap_or(0) as usize;
                let mut entry = bank.catalog_bank.get(slot).copied().unwrap_or_default();
                entry.occupancy = entry.occupancy.max(1);
                weighted.push((packets, entry, ecology));
            }
        }
        weighted.sort_by(|a, b| b.0.total_cmp(&a.0));
        let total_packets = weighted
            .iter()
            .map(|(packets, _, _)| *packets)
            .sum::<f32>()
            .max(1.0e-6);
        weighted
            .into_iter()
            .take(GENOTYPE_PACKET_MAX_PER_CELL)
            .map(|(packets, mut entry, ecology)| {
                entry.packet_mass = (packets / total_packets).clamp(0.0, 1.0);
                (entry, ecology)
            })
            .collect()
    }

    fn packet_community_cells_at(&self, flat: usize) -> f32 {
        self.microbial_cells[flat].max(0.0)
            + self.nitrifier_cells[flat].max(0.0)
            + self.denitrifier_cells[flat].max(0.0)
    }

    pub(super) fn shoreline_packet_recruitment_signal(&self, flat: usize) -> f32 {
        let width = self.config.width;
        let height = self.config.height;
        if width == 0 || height == 0 || flat >= width * height {
            return 0.0;
        }
        let shoreline = shoreline_water_signal(width, height, &self.water_mask, flat);
        let open_water = self.water_mask[flat].clamp(0.0, 1.0);
        if shoreline <= 0.0 || open_water > 0.58 {
            return 0.0;
        }

        let total_packets = self.microbial_packets[flat].max(0.0)
            + self.nitrifier_packets[flat].max(0.0)
            + self.denitrifier_packets[flat].max(0.0);
        let total_cells = self.packet_community_cells_at(flat);
        let guild_total = total_packets.max(1.0e-6);
        let guild_shares = [
            self.microbial_packets[flat].max(0.0) / guild_total,
            self.nitrifier_packets[flat].max(0.0) / guild_total,
            self.denitrifier_packets[flat].max(0.0) / guild_total,
        ];
        let guild_diversity = ((1.0 - guild_shares.iter().map(|share| share * share).sum::<f32>())
            / (1.0 - 1.0 / 3.0))
            .clamp(0.0, 1.0);
        let transition = (self.nitrification_potential[flat]
            .min(self.denitrification_potential[flat])
            * 24_000.0)
            .clamp(0.0, 1.0);
        let mutation = ((self.microbial_packet_mutation_flux[flat]
            + self.nitrifier_packet_mutation_flux[flat]
            + self.denitrifier_packet_mutation_flux[flat])
            * 2200.0)
            .clamp(0.0, 1.0);
        let packet_load = (total_packets / 2600.0).sqrt().clamp(0.0, 1.0);
        let cell_load = (total_cells / 1.6e6).sqrt().clamp(0.0, 1.0);
        clamp(
            shoreline * 0.34
                + guild_diversity * 0.18
                + transition * 0.22
                + mutation * 0.14
                + packet_load * 0.07
                + cell_load * 0.05
                - open_water * 0.12,
            0.0,
            1.0,
        )
    }

    fn shoreline_packet_is_local_peak(&self, flat: usize, signal: f32) -> bool {
        let width = self.config.width;
        let height = self.config.height;
        if width == 0 || height == 0 {
            return false;
        }
        let x = flat % width;
        let y = flat / width;
        for yy in y.saturating_sub(1)..=(y + 1).min(height.saturating_sub(1)) {
            for xx in x.saturating_sub(1)..=(x + 1).min(width.saturating_sub(1)) {
                if xx == x && yy == y {
                    continue;
                }
                let neighbor = idx2(width, xx, yy);
                if self.shoreline_packet_recruitment_signal(neighbor) > signal {
                    return false;
                }
            }
        }
        true
    }

    fn sync_packet_population_ownership(&mut self) {
        let width = self.config.width;
        let height = self.config.height;
        if width == 0 || height == 0 {
            return;
        }
        let mut active = vec![false; width * height];
        let mut claims = Vec::with_capacity(self.packet_populations.len());

        for pop in &self.packet_populations {
            let flat = idx2(width, pop.x, pop.y);
            let owner = self.ownership[flat].owner;
            let dominant = pop
                .dominant_packet()
                .map(|packet| packet.genotype_id)
                .unwrap_or(0);
            let activity = pop.mean_activity();
            let diversity = pop.ecology_diversity();
            let recruitment_signal = self.shoreline_packet_recruitment_signal(flat);
            let strength = clamp(
                0.46 + recruitment_signal * 0.28
                    + diversity * 0.14
                    + (pop.total_cells / 1.2e6).sqrt().clamp(0.0, 1.0) * 0.08
                    + activity * 0.10,
                0.52,
                0.96,
            );
            let keep = match owner {
                SoilOwnershipClass::Background
                | SoilOwnershipClass::GenotypePacketRegion { .. } => {
                    recruitment_signal >= SHORELINE_PACKET_RECRUITMENT_THRESHOLD
                        || (matches!(owner, SoilOwnershipClass::GenotypePacketRegion { .. })
                            && recruitment_signal >= SHORELINE_PACKET_SUSTAIN_THRESHOLD)
                }
                SoilOwnershipClass::ExplicitMicrobeCohort { .. } => true,
                SoilOwnershipClass::PlantTissueRegion { .. }
                | SoilOwnershipClass::AtomisticProbeRegion { .. } => false,
            };
            if !keep {
                continue;
            }
            active[flat] = true;
            claims.push((pop.x, pop.y, dominant, strength, activity));
        }

        for (x, y, genotype_id, strength, activity) in claims {
            let flat = idx2(width, x, y);
            match self.ownership[flat].owner {
                SoilOwnershipClass::Background
                | SoilOwnershipClass::GenotypePacketRegion { .. } => {
                    let _ = self.claim_ownership(
                        x,
                        y,
                        SoilOwnershipClass::GenotypePacketRegion { genotype_id },
                        strength,
                    );
                }
                _ => {}
            }
            self.explicit_microbe_authority[flat] = self.explicit_microbe_authority[flat]
                .max(self.ownership[flat].strength.max(strength));
            self.explicit_microbe_activity[flat] =
                self.explicit_microbe_activity[flat].max(activity.clamp(0.0, 1.0));
        }

        for flat in 0..width * height {
            if !active[flat]
                && matches!(
                    self.ownership[flat].owner,
                    SoilOwnershipClass::GenotypePacketRegion { .. }
                )
            {
                self.release_ownership(flat % width, flat / width);
                self.explicit_microbe_authority[flat] = 0.0;
                self.explicit_microbe_activity[flat] = 0.0;
            }
        }
    }

    fn refresh_background_microbiome_fields(&mut self, eco_dt: f32) {
        let width = self.config.width;
        let height = self.config.height;
        if width == 0 || height == 0 {
            return;
        }
        let depth = self.config.depth.max(1);
        let probe_z = (depth / 3).min(depth.saturating_sub(1));
        let adapt = clamp(eco_dt * 0.012, 0.04, 0.18);

        for y in 0..height {
            for x in 0..width {
                let flat = idx2(width, x, y);
                if self.explicit_microbe_authority[flat] >= EXPLICIT_OWNERSHIP_THRESHOLD {
                    self.microbial_packet_mutation_flux[flat] = 0.0;
                    self.nitrifier_packet_mutation_flux[flat] = 0.0;
                    self.denitrifier_packet_mutation_flux[flat] = 0.0;
                    self.nitrification_potential[flat] = 0.0;
                    self.denitrification_potential[flat] = 0.0;
                    continue;
                }

                let shoreline = shoreline_water_signal(width, height, &self.water_mask, flat);
                let open_water = self.water_mask[flat].clamp(0.0, 1.0);
                let absorbency =
                    soil_texture_absorbency(self.soil_structure[flat], self.organic_matter[flat]);
                let retention =
                    soil_texture_retention(self.soil_structure[flat], self.organic_matter[flat]);
                let capillarity =
                    soil_texture_capillarity(self.soil_structure[flat], self.organic_matter[flat]);
                let patch_water =
                    self.substrate
                        .patch_mean_species(TerrariumSpecies::Water, x, y, probe_z, 1);
                let patch_oxygen = self.substrate.patch_mean_species(
                    TerrariumSpecies::OxygenGas,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_co2 = self.substrate.patch_mean_species(
                    TerrariumSpecies::CarbonDioxide,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_glucose =
                    self.substrate
                        .patch_mean_species(TerrariumSpecies::Glucose, x, y, probe_z, 1);
                let patch_amino_acids = self.substrate.patch_mean_species(
                    TerrariumSpecies::AminoAcidPool,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_nucleotides = self.substrate.patch_mean_species(
                    TerrariumSpecies::NucleotidePool,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_membrane_precursors = self.substrate.patch_mean_species(
                    TerrariumSpecies::MembranePrecursorPool,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_ammonium =
                    self.substrate
                        .patch_mean_species(TerrariumSpecies::Ammonium, x, y, probe_z, 1);
                let patch_nitrate =
                    self.substrate
                        .patch_mean_species(TerrariumSpecies::Nitrate, x, y, probe_z, 1);
                let patch_proton =
                    self.substrate
                        .patch_mean_species(TerrariumSpecies::Proton, x, y, probe_z, 1);
                let patch_dissolved_silicate = self.substrate.patch_mean_species(
                    TerrariumSpecies::DissolvedSilicate,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_bicarbonate = self.substrate.patch_mean_species(
                    TerrariumSpecies::BicarbonatePool,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_surface_proton_load = self.substrate.patch_mean_species(
                    TerrariumSpecies::SurfaceProtonLoad,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_calcium_bicarbonate_complex = self.substrate.patch_mean_species(
                    TerrariumSpecies::CalciumBicarbonateComplex,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_exchangeable_calcium = self.substrate.patch_mean_species(
                    TerrariumSpecies::ExchangeableCalcium,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_exchangeable_magnesium = self.substrate.patch_mean_species(
                    TerrariumSpecies::ExchangeableMagnesium,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_exchangeable_potassium = self.substrate.patch_mean_species(
                    TerrariumSpecies::ExchangeablePotassium,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_exchangeable_sodium = self.substrate.patch_mean_species(
                    TerrariumSpecies::ExchangeableSodium,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_exchangeable_aluminum = self.substrate.patch_mean_species(
                    TerrariumSpecies::ExchangeableAluminum,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_aqueous_iron = self.substrate.patch_mean_species(
                    TerrariumSpecies::AqueousIronPool,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_clay = self.substrate.patch_mean_species(
                    TerrariumSpecies::ClayMineral,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_carbonate = self.substrate.patch_mean_species(
                    TerrariumSpecies::CarbonateMineral,
                    x,
                    y,
                    probe_z,
                    1,
                );
                let patch_atp =
                    self.substrate
                        .patch_mean_species(TerrariumSpecies::AtpFlux, x, y, probe_z, 1);
                let base_saturation = soil_base_saturation(
                    patch_exchangeable_calcium,
                    patch_exchangeable_magnesium,
                    patch_exchangeable_potassium,
                    patch_exchangeable_sodium,
                    patch_exchangeable_aluminum,
                    patch_proton,
                    patch_surface_proton_load,
                );
                let aluminum_toxicity = soil_aluminum_toxicity(
                    patch_exchangeable_aluminum,
                    patch_proton,
                    patch_surface_proton_load,
                    base_saturation,
                );
                let weathering_support = soil_weathering_support(
                    patch_dissolved_silicate,
                    patch_bicarbonate,
                    patch_calcium_bicarbonate_complex,
                    patch_exchangeable_calcium,
                    patch_exchangeable_magnesium,
                    patch_exchangeable_potassium,
                    patch_aqueous_iron,
                    patch_carbonate,
                    patch_clay,
                );
                let mineral_buffer = soil_mineral_buffer(
                    patch_carbonate,
                    patch_bicarbonate,
                    patch_calcium_bicarbonate_complex,
                    patch_exchangeable_calcium,
                    patch_exchangeable_magnesium,
                    patch_proton,
                    patch_surface_proton_load,
                    base_saturation,
                );
                let moisture_factor = clamp(
                    (self.moisture[flat] + self.deep_moisture[flat] * 0.55 + shoreline * 0.16)
                        / (0.34 + retention * 0.22),
                    0.0,
                    1.8,
                );
                let oxygen_factor = clamp(
                    (patch_oxygen / 0.10) * (0.72 + absorbency * 0.32),
                    0.0,
                    1.25,
                );
                let aeration_factor = clamp(
                    oxygen_factor * (0.72 + absorbency * 0.24) + capillarity * 0.10
                        - self.deep_moisture[flat] * 0.22,
                    0.05,
                    1.2,
                );
                let anoxia_factor = clamp(
                    (self.deep_moisture[flat] * 0.82
                        + shoreline * 0.34
                        + patch_water * 0.16
                        + patch_co2 * 1.45)
                        - patch_oxygen * 1.75,
                    0.02,
                    1.35,
                );
                let substrate_gate = clamp(
                    (self.litter_carbon[flat] * 1.10
                        + self.root_exudates[flat] * 1.35
                        + self.organic_matter[flat] * 0.90
                        + patch_glucose * 1.45
                        + patch_amino_acids * 4.40
                        + patch_nucleotides * 5.20
                        + patch_membrane_precursors * 5.80)
                        / 0.12,
                    0.0,
                    1.8,
                );
                let biosynthetic_richness = clamp(
                    patch_amino_acids / 0.028
                        + patch_nucleotides / 0.020
                        + patch_membrane_precursors / 0.018
                        + patch_atp / 0.12
                        + weathering_support * 0.10,
                    0.0,
                    2.2,
                );
                let root_factor = 1.0
                    + self.root_density[flat] * 0.12
                    + self.symbiont_biomass[flat] * 0.24
                    + base_saturation * 0.10
                    + weathering_support * 0.08
                    - aluminum_toxicity * 0.08;

                let copiotroph_target = microbial_copiotroph_target(
                    substrate_gate,
                    moisture_factor,
                    oxygen_factor,
                    root_factor,
                );
                let nitrifier_target =
                    nitrifier_aerobic_target(oxygen_factor, aeration_factor, anoxia_factor);
                let denitrifier_target = denitrifier_anoxic_target(
                    anoxia_factor,
                    self.deep_moisture[flat],
                    oxygen_factor,
                );

                self.microbial_copiotroph_fraction[flat] = clamp(
                    self.microbial_copiotroph_fraction[flat]
                        + (copiotroph_target - self.microbial_copiotroph_fraction[flat]) * adapt,
                    0.05,
                    0.98,
                );
                self.nitrifier_aerobic_fraction[flat] = clamp(
                    self.nitrifier_aerobic_fraction[flat]
                        + (nitrifier_target - self.nitrifier_aerobic_fraction[flat]) * adapt,
                    0.05,
                    0.98,
                );
                self.denitrifier_anoxic_fraction[flat] = clamp(
                    self.denitrifier_anoxic_fraction[flat]
                        + (denitrifier_target - self.denitrifier_anoxic_fraction[flat]) * adapt,
                    0.05,
                    0.98,
                );

                let decomposer_resource = clamp(
                    substrate_gate * moisture_factor * (0.40 + oxygen_factor * 0.34)
                        + biosynthetic_richness * (0.16 + moisture_factor * 0.10)
                        + patch_atp * 1.8
                        + shoreline * 0.14
                        + weathering_support * 0.12
                        + mineral_buffer * 0.08,
                    0.0,
                    2.5,
                );
                let nitrifier_resource = clamp(
                    (patch_ammonium / 0.12)
                        * (0.24 + nitrifier_target * 0.76)
                        * (0.34 + oxygen_factor * 0.66)
                        * (1.08 - self.deep_moisture[flat] * 0.42)
                        * (1.04 - biosynthetic_richness * 0.08).clamp(0.72, 1.04)
                        * (1.0 - shoreline * 0.45)
                        * (1.0 - open_water * 0.72)
                        * (0.72 + base_saturation * 0.30 + mineral_buffer * 0.10)
                        * (1.02 - aluminum_toxicity * 0.14).clamp(0.52, 1.02),
                    0.0,
                    1.4,
                );
                let denitrifier_resource = clamp(
                    (patch_nitrate / 0.12)
                        * (0.26 + denitrifier_target * 0.74)
                        * (0.18 + anoxia_factor * 0.82)
                        * (0.92 + biosynthetic_richness * 0.10)
                        * (0.24
                            + self.deep_moisture[flat] * 0.44
                            + shoreline * 0.32
                            + open_water * 0.58)
                        * (1.04 - oxygen_factor * 0.34).clamp(0.12, 1.0)
                        * (0.86 + patch_aqueous_iron / 0.10 * 0.12 + weathering_support * 0.06)
                        * (1.02 - aluminum_toxicity * 0.08).clamp(0.72, 1.02),
                    0.0,
                    1.4,
                );

                let microbial_vitality_target =
                    clamp(0.32 + decomposer_resource * 0.38, 0.08, 1.22);
                let nitrifier_vitality_target = clamp(0.26 + nitrifier_resource * 0.42, 0.08, 1.22);
                let denitrifier_vitality_target =
                    clamp(0.24 + denitrifier_resource * 0.44, 0.08, 1.22);
                self.microbial_vitality[flat] = clamp(
                    self.microbial_vitality[flat]
                        + (microbial_vitality_target - self.microbial_vitality[flat]) * adapt,
                    0.0,
                    1.25,
                );
                self.nitrifier_vitality[flat] = clamp(
                    self.nitrifier_vitality[flat]
                        + (nitrifier_vitality_target - self.nitrifier_vitality[flat]) * adapt,
                    0.0,
                    1.25,
                );
                self.denitrifier_vitality[flat] = clamp(
                    self.denitrifier_vitality[flat]
                        + (denitrifier_vitality_target - self.denitrifier_vitality[flat]) * adapt,
                    0.0,
                    1.25,
                );

                let microbial_reserve_target = clamp(
                    0.24 + decomposer_resource * 0.20 + retention * 0.18,
                    0.06,
                    1.0,
                );
                let nitrifier_reserve_target = clamp(
                    0.22 + nitrifier_resource * 0.18 + absorbency * 0.12,
                    0.06,
                    1.0,
                );
                let denitrifier_reserve_target = clamp(
                    0.22 + denitrifier_resource * 0.20 + retention * 0.14,
                    0.06,
                    1.0,
                );
                self.microbial_reserve[flat] = clamp(
                    self.microbial_reserve[flat]
                        + (microbial_reserve_target - self.microbial_reserve[flat]) * adapt,
                    0.0,
                    1.0,
                );
                self.nitrifier_reserve[flat] = clamp(
                    self.nitrifier_reserve[flat]
                        + (nitrifier_reserve_target - self.nitrifier_reserve[flat]) * adapt,
                    0.0,
                    1.0,
                );
                self.denitrifier_reserve[flat] = clamp(
                    self.denitrifier_reserve[flat]
                        + (denitrifier_reserve_target - self.denitrifier_reserve[flat]) * adapt,
                    0.0,
                    1.0,
                );

                let nitrifier_biomass_target = clamp(
                    self.microbial_biomass[flat] * (0.04 + nitrifier_target * 0.24)
                        + patch_ammonium * 0.12
                        + absorbency * 0.01,
                    0.001,
                    2.0,
                );
                let denitrifier_biomass_target = clamp(
                    self.microbial_biomass[flat] * (0.03 + denitrifier_target * 0.26)
                        + patch_nitrate * 0.10
                        + shoreline * 0.02,
                    0.001,
                    2.0,
                );
                self.nitrifier_biomass[flat] = clamp(
                    self.nitrifier_biomass[flat]
                        + (nitrifier_biomass_target - self.nitrifier_biomass[flat]) * adapt,
                    0.0005,
                    2.0,
                );
                self.denitrifier_biomass[flat] = clamp(
                    self.denitrifier_biomass[flat]
                        + (denitrifier_biomass_target - self.denitrifier_biomass[flat]) * adapt,
                    0.0005,
                    2.0,
                );

                let microbial_cell_target = self.microbial_biomass[flat]
                    * (850_000.0 + self.microbial_vitality[flat] * 240_000.0);
                let nitrifier_cell_target = self.nitrifier_biomass[flat]
                    * (180_000.0 + self.nitrifier_aerobic_fraction[flat] * 120_000.0);
                let denitrifier_cell_target = self.denitrifier_biomass[flat]
                    * (180_000.0 + self.denitrifier_anoxic_fraction[flat] * 120_000.0);
                self.microbial_cells[flat] = clamp(
                    self.microbial_cells[flat]
                        + (microbial_cell_target - self.microbial_cells[flat]) * adapt,
                    1.0,
                    8.0e6,
                );
                self.nitrifier_cells[flat] = clamp(
                    self.nitrifier_cells[flat]
                        + (nitrifier_cell_target - self.nitrifier_cells[flat]) * adapt,
                    1.0,
                    4.0e6,
                );
                self.denitrifier_cells[flat] = clamp(
                    self.denitrifier_cells[flat]
                        + (denitrifier_cell_target - self.denitrifier_cells[flat]) * adapt,
                    1.0,
                    4.0e6,
                );

                let microbial_packet_target =
                    self.microbial_cells[flat] / MICROBIAL_PACKET_TARGET_CELLS.max(1.0);
                let nitrifier_packet_target =
                    self.nitrifier_cells[flat] / NITRIFIER_PACKET_TARGET_CELLS.max(1.0);
                let denitrifier_packet_target =
                    self.denitrifier_cells[flat] / DENITRIFIER_PACKET_TARGET_CELLS.max(1.0);
                self.microbial_packets[flat] = clamp(
                    self.microbial_packets[flat]
                        + (microbial_packet_target - self.microbial_packets[flat]) * adapt,
                    0.0,
                    20_000.0,
                );
                self.nitrifier_packets[flat] = clamp(
                    self.nitrifier_packets[flat]
                        + (nitrifier_packet_target - self.nitrifier_packets[flat]) * adapt,
                    0.0,
                    12_000.0,
                );
                self.denitrifier_packets[flat] = clamp(
                    self.denitrifier_packets[flat]
                        + (denitrifier_packet_target - self.denitrifier_packets[flat]) * adapt,
                    0.0,
                    12_000.0,
                );
                self.microbial_copiotroph_packets[flat] =
                    self.microbial_packets[flat] * self.microbial_copiotroph_fraction[flat];
                self.nitrifier_aerobic_packets[flat] =
                    self.nitrifier_packets[flat] * self.nitrifier_aerobic_fraction[flat];
                self.denitrifier_anoxic_packets[flat] =
                    self.denitrifier_packets[flat] * self.denitrifier_anoxic_fraction[flat];

                self.nitrification_potential[flat] = clamp(
                    self.nitrifier_biomass[flat] * nitrifier_resource * 0.00085,
                    0.0,
                    0.00025,
                );
                self.denitrification_potential[flat] = clamp(
                    self.denitrifier_biomass[flat] * denitrifier_resource * 0.00105,
                    0.0,
                    0.00025,
                );
                self.microbial_packet_mutation_flux[flat] = clamp(
                    (shoreline * 0.40
                        + (moisture_factor - 0.85).abs() * 0.22
                        + (oxygen_factor - 0.70).abs() * 0.18)
                        * 0.00010,
                    0.0,
                    0.00025,
                );
                self.nitrifier_packet_mutation_flux[flat] = clamp(
                    ((nitrifier_target - self.nitrifier_aerobic_fraction[flat]).abs() * 0.46
                        + patch_ammonium / 0.12 * 0.24
                        + absorbency * 0.10)
                        * 0.00010,
                    0.0,
                    0.00025,
                );
                self.denitrifier_packet_mutation_flux[flat] = clamp(
                    ((denitrifier_target - self.denitrifier_anoxic_fraction[flat]).abs() * 0.46
                        + shoreline * 0.24
                        + patch_nitrate / 0.12 * 0.20)
                        * 0.00011,
                    0.0,
                    0.00025,
                );
            }
        }
        self.materialize_emergent_packet_banks();
    }

    pub(super) fn rebuild_water_mask(&mut self) {
        self.water_mask.fill(0.0);
        for water in &self.waters {
            if !water.alive {
                continue;
            }
            let amplitude = clamp(water.volume / 140.0, 0.06, 1.0);
            let radius: usize = if water.volume >= 120.0 { 3 } else { 2 };
            deposit_2d(
                &mut self.water_mask,
                self.config.width,
                self.config.height,
                water.x,
                water.y,
                radius,
                amplitude,
            );
        }

        let (sub_width, sub_height, sub_depth) = self.substrate.shape();
        let width = self.config.width.min(sub_width);
        let height = self.config.height.min(sub_height);
        let depth = sub_depth.max(1).min(3);
        let plane = sub_width * sub_height;
        let water_field = self.substrate.species_field(TerrariumSpecies::Water);
        // Substrate water → water_mask. The water_mask serves TWO purposes:
        // 1. Ecology: shoreline detection for microbial recruitment (needs the full signal)
        // 2. Rendering: visible standing water on terrain surface
        // We keep the original substrate integration to preserve ecology,
        // but the RENDERING layer (shaders) filters for actual standing water
        // using the van Genuchten porosity model.
        for y in 0..height {
            for x in 0..width {
                let mut surface_open = 0.0f32;
                let mut supported_fill = 0.0f32;
                let mut weight_total = 0.0f32;
                for z in 0..depth {
                    let idx = z * plane + y * sub_width + x;
                    let raw = water_field[idx].max(0.0);
                    let threshold = 0.92 - z as f32 * 0.04;
                    let scale = 1.08 - z as f32 * 0.12;
                    let occupancy = ((raw - threshold) / scale.max(0.28)).clamp(0.0, 1.0);
                    let weight = 1.0 - z as f32 * 0.22;
                    if z == 0 {
                        surface_open = occupancy;
                    }
                    supported_fill += occupancy * weight;
                    weight_total += weight;
                }
                let column_fill = if weight_total > 0.0 {
                    supported_fill / weight_total
                } else {
                    0.0
                };
                let open_water =
                    (surface_open * 0.60 + column_fill * 0.40) * (0.30 + column_fill * 0.70);
                let flat = idx2(self.config.width, x, y);
                self.water_mask[flat] = self.water_mask[flat].max(open_water.clamp(0.0, 1.0));
            }
        }
    }

    pub fn step_soil_fauna_phase(&mut self, eco_dt: f32) {
        let eco_dt_hours = eco_dt / 3600.0;
        let dims = (
            self.config.width,
            self.config.height,
            self.config.depth.max(1),
        );
        // Clone read-only inputs to avoid borrow conflict with &mut substrate.
        let hydration = self.substrate.hydration.clone();
        // soil_temperature not on substrate; use world temperature field sampled at surface.
        let temperature: Vec<f32> = (0..dims.0 * dims.1)
            .map(|i| {
                let x = i % dims.0;
                let y = i / dims.0;
                self.sample_temperature_at(x, y, 0)
            })
            .collect();
        let _result = step_soil_fauna(
            &mut self.earthworm_population,
            &mut self.nematode_guilds,
            &mut self.microbial_biomass,
            &mut self.nitrifier_biomass,
            &mut self.organic_matter,
            &mut self.substrate,
            &hydration,
            &temperature,
            eco_dt_hours,
            dims,
        );
    }

    pub(super) fn sync_substrate_controls(&mut self) -> Result<(), String> {
        let plane = self.config.width * self.config.height;
        let total = plane * self.config.depth.max(1);
        let mut hydration = vec![0.0f32; total];
        let mut microbes = vec![0.0f32; total];
        let mut plant_drive = vec![0.0f32; total];
        let depth = self.config.depth.max(1);
        for z in 0..depth {
            let z_frac = if depth > 1 {
                z as f32 / (depth - 1) as f32
            } else {
                0.0
            };
            for i in 0..plane {
                let gid = z * plane + i;
                hydration[gid] = clamp(
                    self.moisture[i] * (1.0 - z_frac * 0.55) + self.deep_moisture[i] * z_frac,
                    0.02,
                    1.0,
                );
                microbes[gid] = clamp(
                    self.microbial_biomass[i] * (0.65 + self.moisture[i] * 0.55)
                        + self.symbiont_biomass[i] * (0.55 + z_frac * 0.30),
                    0.02,
                    1.2,
                );
                plant_drive[gid] = clamp(
                    self.root_density[i] * (1.0 - z_frac * 0.35) * (0.35 + self.daylight() * 0.65),
                    0.0,
                    1.5,
                );
            }
        }
        self.substrate.set_hydration_field(&hydration)?;
        self.substrate.set_microbial_activity_field(&microbes)?;
        self.substrate.set_plant_drive_field(&plant_drive)?;
        Ok(())
    }

    pub(super) fn step_surface_respiration(&mut self, eco_dt: f32) {
        let depth = self.config.depth.max(1);
        for y in 0..self.config.height {
            for x in 0..self.config.width {
                let flat = idx2(self.config.width, x, y);
                let local_substrate = self.litter_carbon[flat] * 1.10
                    + self.root_exudates[flat] * 1.35
                    + self.organic_matter[flat] * 0.90;
                let moisture_factor = clamp(
                    (self.moisture[flat] + self.deep_moisture[flat] * 0.35) / 0.48,
                    0.0,
                    1.6,
                );
                let oxygen_factor = clamp(1.15 - self.deep_moisture[flat] * 0.55, 0.35, 1.1);
                let aeration_factor = clamp(
                    1.10 - self.moisture[flat] * 0.55 - self.deep_moisture[flat] * 0.45,
                    0.05,
                    1.15,
                );
                let anoxia_factor = clamp(
                    (self.deep_moisture[flat] * 0.95 + self.moisture[flat] * 0.18)
                        - oxygen_factor * 0.28,
                    0.02,
                    1.3,
                );
                let root_factor = 1.0 + self.root_density[flat] * 0.08;
                let substrate_gate = clamp(local_substrate / 0.08, 0.0, 1.35);
                let explicit_authority = self.explicit_microbe_authority[flat].clamp(0.0, 0.95);
                let explicit_activity = self.explicit_microbe_activity[flat].max(0.0);
                if explicit_authority >= EXPLICIT_OWNERSHIP_THRESHOLD {
                    continue;
                }
                let coarse_biology_factor = if explicit_authority >= EXPLICIT_OWNERSHIP_THRESHOLD {
                    0.0
                } else {
                    1.0 - explicit_authority
                };
                let decomposer_trait_factor = trait_match(
                    self.microbial_copiotroph_fraction[flat],
                    microbial_copiotroph_target(
                        substrate_gate,
                        moisture_factor,
                        oxygen_factor,
                        root_factor,
                    ),
                );
                let decomposer_packet_factor = packet_surface_factor(
                    self.microbial_cells[flat],
                    self.microbial_packets[flat],
                    MICROBIAL_PACKET_TARGET_CELLS,
                );
                let decomposer_active = self.microbial_cells[flat]
                    * (1.0 - self.microbial_dormancy[flat]).clamp(0.02, 1.0)
                    * (0.25 + 0.75 * self.microbial_vitality[flat]).clamp(0.0, 1.25)
                    * (0.55 + 0.45 * self.microbial_reserve[flat]).clamp(0.20, 1.25)
                    * decomposer_packet_factor
                    * decomposer_trait_factor
                    * 0.035
                    * coarse_biology_factor;
                let nitrifier_trait_factor = trait_match(
                    self.nitrifier_aerobic_fraction[flat],
                    nitrifier_aerobic_target(oxygen_factor, aeration_factor, anoxia_factor),
                );
                let nitrifier_packet_factor = packet_surface_factor(
                    self.nitrifier_cells[flat],
                    self.nitrifier_packets[flat],
                    NITRIFIER_PACKET_TARGET_CELLS,
                );
                let nitrifier_active = self.nitrifier_cells[flat]
                    * (1.0 - self.nitrifier_dormancy[flat]).clamp(0.02, 1.0)
                    * (0.25 + 0.75 * self.nitrifier_vitality[flat]).clamp(0.0, 1.25)
                    * (0.55 + 0.45 * self.nitrifier_reserve[flat]).clamp(0.20, 1.25)
                    * nitrifier_packet_factor
                    * nitrifier_trait_factor
                    * 0.030
                    * coarse_biology_factor;
                let denitrifier_trait_factor = trait_match(
                    self.denitrifier_anoxic_fraction[flat],
                    denitrifier_anoxic_target(
                        anoxia_factor,
                        self.deep_moisture[flat],
                        oxygen_factor,
                    ),
                );
                let denitrifier_packet_factor = packet_surface_factor(
                    self.denitrifier_cells[flat],
                    self.denitrifier_packets[flat],
                    DENITRIFIER_PACKET_TARGET_CELLS,
                );
                let denitrifier_active = self.denitrifier_cells[flat]
                    * (1.0 - self.denitrifier_dormancy[flat]).clamp(0.02, 1.0)
                    * (0.25 + 0.75 * self.denitrifier_vitality[flat]).clamp(0.0, 1.25)
                    * (0.55 + 0.45 * self.denitrifier_reserve[flat]).clamp(0.20, 1.25)
                    * denitrifier_packet_factor
                    * denitrifier_trait_factor
                    * 0.030
                    * coarse_biology_factor;
                let microbial_drive = (decomposer_active
                    + self.microbial_biomass[flat] * 0.35 * coarse_biology_factor)
                    * (0.55 + self.moisture[flat] * 0.75)
                    + explicit_activity * (0.62 + self.moisture[flat] * 0.44)
                    + self.symbiont_biomass[flat] * 0.22
                    + (nitrifier_active
                        + self.nitrifier_biomass[flat] * 0.22 * coarse_biology_factor)
                        * (0.12 + self.nitrification_potential[flat] * 0.20)
                    + (denitrifier_active
                        + self.denitrifier_biomass[flat] * 0.22 * coarse_biology_factor)
                        * (0.20 + self.denitrification_potential[flat] * 0.26)
                    + self.root_exudates[flat] * 1.35
                    + self.litter_carbon[flat] * 0.42
                    + self.organic_matter[flat] * 0.18;
                let soil_co2_flux = clamp(microbial_drive * eco_dt * 0.000035, 0.0, 0.0025);
                if soil_co2_flux <= 2.0e-6 {
                    continue;
                }
                let soil_o2_flux = -soil_co2_flux * (0.80 + self.moisture[flat] * 0.10);
                let soil_humidity_flux = clamp(
                    self.moisture[flat] * microbial_drive * eco_dt * 0.000008,
                    0.0,
                    0.0015,
                );
                self.exchange_atmosphere_flux_bundle(
                    x,
                    y,
                    0,
                    1,
                    soil_co2_flux,
                    soil_o2_flux,
                    soil_humidity_flux,
                );
            }
        }

        let fruit_fluxes = self
            .fruits
            .iter()
            .filter(|fruit| fruit.source.alive && fruit.source.sugar_content > 0.01)
            .map(|fruit| {
                let flat = idx2(
                    self.config.width,
                    fruit.source.x.min(self.config.width - 1),
                    fruit.source.y.min(self.config.height - 1),
                );
                let microbial = self.microbial_biomass[flat];
                let respiration_driver = fruit.source.sugar_content.max(0.0)
                    * (0.35 + fruit.source.ripeness.max(0.0) * 0.65)
                    * (1.0 + microbial * 6.0);
                let fruit_co2_flux = clamp(respiration_driver * eco_dt * 0.00016, 0.0, 0.004);
                let fruit_o2_flux = -fruit_co2_flux * (0.86 + microbial * 0.08);
                let fruit_humidity_flux = clamp(respiration_driver * eco_dt * 0.00005, 0.0, 0.0018);
                (
                    fruit.source.x,
                    fruit.source.y,
                    fruit.source.z.min(depth - 1),
                    fruit.radius.round().max(1.0) as usize,
                    fruit_co2_flux,
                    fruit_o2_flux,
                    fruit_humidity_flux,
                )
            })
            .collect::<Vec<_>>();
        for (x, y, z, radius, co2_flux, o2_flux, humidity_flux) in fruit_fluxes {
            self.exchange_atmosphere_flux_bundle(x, y, z, radius, co2_flux, o2_flux, humidity_flux);
        }
    }

    #[allow(dead_code)]
    pub(super) fn couple_soil_atmosphere_gases(&mut self, eco_dt: f32) {
        let mut deposits = Vec::new();
        let mut extractions = Vec::new();
        let patch_radius = 1usize;

        for y in 0..self.config.height {
            for x in 0..self.config.width {
                let flat = idx2(self.config.width, x, y);
                let porosity = clamp(
                    self.soil_structure[flat] * (1.08 - self.moisture[flat] * 0.58)
                        + self.deep_moisture[flat] * 0.08,
                    0.05,
                    1.1,
                );
                let air_o2 = self.sample_odorant_patch(ATMOS_O2_IDX, x, y, 0, patch_radius);
                let air_co2 = self.sample_odorant_patch(ATMOS_CO2_IDX, x, y, 0, patch_radius);
                let soil_o2 = self.substrate.patch_mean_species(
                    TerrariumSpecies::OxygenGas,
                    x,
                    y,
                    0,
                    patch_radius,
                );
                let soil_co2 = self.substrate.patch_mean_species(
                    TerrariumSpecies::CarbonDioxide,
                    x,
                    y,
                    0,
                    patch_radius,
                );

                let eq_o2 = HENRY_O2 * air_o2;
                let eq_co2 = HENRY_CO2 * air_co2;
                let o2_flux = FICK_SURFACE_CONDUCTANCE * porosity * eco_dt * (eq_o2 - soil_o2);
                let co2_flux = FICK_SURFACE_CONDUCTANCE * porosity * eco_dt * (eq_co2 - soil_co2);

                let o2_mag = o2_flux.abs().min(0.001);
                if o2_mag > 1.0e-7 {
                    if o2_flux > 0.0 {
                        deposits.push((
                            TerrariumSpecies::OxygenGas,
                            x,
                            y,
                            0usize,
                            patch_radius,
                            o2_mag,
                        ));
                    } else {
                        extractions.push((
                            TerrariumSpecies::OxygenGas,
                            x,
                            y,
                            0usize,
                            patch_radius,
                            o2_mag,
                        ));
                    }
                }

                let co2_mag = co2_flux.abs().min(0.001);
                if co2_mag > 1.0e-7 {
                    if co2_flux > 0.0 {
                        deposits.push((
                            TerrariumSpecies::CarbonDioxide,
                            x,
                            y,
                            0usize,
                            patch_radius,
                            co2_mag,
                        ));
                    } else {
                        extractions.push((
                            TerrariumSpecies::CarbonDioxide,
                            x,
                            y,
                            0usize,
                            patch_radius,
                            co2_mag,
                        ));
                    }
                }
            }
        }

        for (species, x, y, z, radius, amount) in deposits {
            let deposited = self
                .substrate
                .deposit_patch_species(species, x, y, z, radius, amount);
            match species {
                TerrariumSpecies::OxygenGas => {
                    self.exchange_atmosphere_odorant(ATMOS_O2_IDX, x, y, z, radius, -deposited)
                }
                TerrariumSpecies::CarbonDioxide => {
                    self.exchange_atmosphere_odorant(ATMOS_CO2_IDX, x, y, z, radius, -deposited)
                }
                _ => {}
            }
        }

        for (species, x, y, z, radius, amount) in extractions {
            let removed = self
                .substrate
                .extract_patch_species(species, x, y, z, radius, amount);
            match species {
                TerrariumSpecies::OxygenGas => {
                    self.exchange_atmosphere_odorant(ATMOS_O2_IDX, x, y, z, radius, removed)
                }
                TerrariumSpecies::CarbonDioxide => {
                    self.exchange_atmosphere_odorant(ATMOS_CO2_IDX, x, y, z, radius, removed)
                }
                _ => {}
            }
        }
    }

    pub(super) fn step_broad_soil(&mut self, eco_dt: f32) -> Result<(), String> {
        self.rebuild_water_mask();
        let has_ownership = self.ownership_diagnostics.owned_fraction > 0.0;
        let pre_moisture = if has_ownership {
            self.moisture.clone()
        } else {
            Vec::new()
        };
        let pre_deep_moisture = if has_ownership {
            self.deep_moisture.clone()
        } else {
            Vec::new()
        };
        let pre_nutrients = if has_ownership {
            self.dissolved_nutrients.clone()
        } else {
            Vec::new()
        };
        let pre_nitrogen = if has_ownership {
            self.mineral_nitrogen.clone()
        } else {
            Vec::new()
        };
        let pre_shallow = if has_ownership {
            self.shallow_nutrients.clone()
        } else {
            Vec::new()
        };
        let pre_deep_minerals = if has_ownership {
            self.deep_minerals.clone()
        } else {
            Vec::new()
        };
        let pre_organic = if has_ownership {
            self.organic_matter.clone()
        } else {
            Vec::new()
        };
        let pre_litter = if has_ownership {
            self.litter_carbon.clone()
        } else {
            Vec::new()
        };
        let pre_microbes = if has_ownership {
            self.microbial_biomass.clone()
        } else {
            Vec::new()
        };
        let pre_symbionts = if has_ownership {
            self.symbiont_biomass.clone()
        } else {
            Vec::new()
        };
        let pre_exudates = if has_ownership {
            self.root_exudates.clone()
        } else {
            Vec::new()
        };

        let result = step_soil_broad_pools(
            self.config.width,
            self.config.height,
            eco_dt,
            self.daylight(),
            temp_response(
                self.sample_temperature_at(self.config.width / 2, self.config.height / 2, 0),
                24.0,
                10.0,
            ),
            &self.water_mask,
            &self.canopy_cover,
            &self.root_density,
            &self.moisture,
            &self.deep_moisture,
            &self.dissolved_nutrients,
            &self.mineral_nitrogen,
            &self.shallow_nutrients,
            &self.deep_minerals,
            &self.organic_matter,
            &self.litter_carbon,
            &self.microbial_biomass,
            &self.symbiont_biomass,
            &self.root_exudates,
            &self.soil_structure,
        )?;
        self.moisture = result.moisture;
        self.deep_moisture = result.deep_moisture;
        self.dissolved_nutrients = result.dissolved_nutrients;
        self.mineral_nitrogen = result.mineral_nitrogen;
        self.shallow_nutrients = result.shallow_nutrients;
        self.deep_minerals = result.deep_minerals;
        self.organic_matter = result.organic_matter;
        self.litter_carbon = result.litter_carbon;
        self.microbial_biomass = result.microbial_biomass;
        self.symbiont_biomass = result.symbiont_biomass;
        self.root_exudates = result.root_exudates;

        // Authority suppression: for explicitly-owned cells, blend the broad-soil
        // result back toward the pre-step value.
        if has_ownership {
            for (i, cell) in self.ownership.iter().enumerate() {
                if cell.owner.is_background() {
                    continue;
                }
                let s = cell.strength;
                macro_rules! suppress {
                    ($field:expr, $pre:expr) => {
                        $field[i] = $field[i] * (1.0 - s) + $pre[i] * s;
                    };
                }
                suppress!(self.moisture, pre_moisture);
                suppress!(self.deep_moisture, pre_deep_moisture);
                suppress!(self.dissolved_nutrients, pre_nutrients);
                suppress!(self.mineral_nitrogen, pre_nitrogen);
                suppress!(self.shallow_nutrients, pre_shallow);
                suppress!(self.deep_minerals, pre_deep_minerals);
                suppress!(self.organic_matter, pre_organic);
                suppress!(self.litter_carbon, pre_litter);
                suppress!(self.microbial_biomass, pre_microbes);
                suppress!(self.symbiont_biomass, pre_symbionts);
                suppress!(self.root_exudates, pre_exudates);
            }
        }
        self.refresh_background_microbiome_fields(eco_dt);
        Ok(())
    }

    pub(super) fn recruit_packet_populations(&mut self) {
        self.materialize_emergent_packet_banks();
        if self.packet_populations.len() >= GENOTYPE_PACKET_POPULATION_MAX_CELLS {
            return;
        }
        let width = self.config.width;
        let height = self.config.height;
        let depth = self.config.depth.max(1);

        for idx in 0..self.explicit_microbes.len() {
            let cohort = &self.explicit_microbes[idx];
            let (x, y, z) = (cohort.x, cohort.y, cohort.z);

            if self
                .packet_populations
                .iter()
                .any(|pop| pop.x == x && pop.y == y)
            {
                continue;
            }
            if self.packet_populations.len() >= GENOTYPE_PACKET_POPULATION_MAX_CELLS {
                break;
            }

            let flat = idx2(width, x, y);
            let authority = self.explicit_microbe_authority[flat];
            if authority < EXPLICIT_OWNERSHIP_THRESHOLD {
                continue;
            }

            let mut pop = GenotypePacketPopulation::new(x, y, z.min(depth - 1));
            let seed_entries = self.combined_packet_seed_entries_at(flat);
            let total_cells = self.packet_community_cells_at(flat).max(1.0);
            pop.seed_from_secondary_entries(&seed_entries, total_cells);

            if pop.is_alive() {
                self.packet_populations.push(pop);
            }
        }

        for flat in 0..width * height {
            if self.packet_populations.len() >= GENOTYPE_PACKET_POPULATION_MAX_CELLS {
                break;
            }
            let x = flat % width;
            let y = flat / width;
            let owner = self.ownership[flat].owner;
            if !matches!(
                owner,
                SoilOwnershipClass::Background | SoilOwnershipClass::GenotypePacketRegion { .. }
            ) {
                continue;
            }
            if self
                .packet_populations
                .iter()
                .any(|pop| pop.x == x && pop.y == y)
            {
                continue;
            }

            let signal = self.shoreline_packet_recruitment_signal(flat);
            if signal < SHORELINE_PACKET_RECRUITMENT_THRESHOLD
                || !self.shoreline_packet_is_local_peak(flat, signal)
            {
                continue;
            }

            let seed_entries = self.combined_packet_seed_entries_at(flat);
            let total_cells = self.packet_community_cells_at(flat);
            if seed_entries.is_empty() || total_cells < MICROBIAL_PACKET_TARGET_CELLS * 2.0 {
                continue;
            }

            let mut pop = GenotypePacketPopulation::new(x, y, (depth / 3).min(depth - 1));
            pop.seed_from_secondary_entries(&seed_entries, total_cells.max(1.0));
            if pop.is_alive() {
                self.packet_populations.push(pop);
            }
        }
        self.sync_packet_population_ownership();
    }

    pub(super) fn step_packet_populations(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.packet_populations.is_empty() {
            return Ok(());
        }

        let depth = self.config.depth.max(1);
        let width = self.config.width;
        let height = self.config.height;

        for pop_idx in 0..self.packet_populations.len() {
            let (x, y, z) = {
                let pop = &self.packet_populations[pop_idx];
                (pop.x, pop.y, pop.z.min(depth - 1))
            };
            let flat = idx2(width, x, y);
            if matches!(
                self.ownership[flat].owner,
                SoilOwnershipClass::ExplicitMicrobeCohort { .. }
            ) {
                continue;
            }

            let local_glucose =
                self.substrate
                    .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, 1);
            let local_oxygen =
                self.substrate
                    .patch_mean_species(TerrariumSpecies::OxygenGas, x, y, z, 1);
            let local_co2 =
                self.substrate
                    .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, 1);
            let local_stress: f32 = clamp(
                (0.3f32 - local_oxygen).max(0.0) * 2.0
                    + local_co2 * 0.5
                    + (0.05f32 - local_glucose).max(0.0) * 4.0,
                0.0,
                1.0,
            );

            // Snapshot cumulative counters before stepping
            let pre_glucose = self.packet_populations[pop_idx].total_glucose_draw();
            let pre_co2 = self.packet_populations[pop_idx].total_co2_release();
            let pre_oxygen = self.packet_populations[pop_idx].total_oxygen_draw();
            let pre_nh4 = self.packet_populations[pop_idx].total_ammonium_draw();
            let pre_proton = self.packet_populations[pop_idx].total_proton_release();

            self.packet_populations[pop_idx].step(
                eco_dt,
                local_glucose,
                local_oxygen,
                local_stress,
            );

            // Compute deltas from cumulative tracking
            let glucose_delta = self.packet_populations[pop_idx].total_glucose_draw() - pre_glucose;
            let co2_delta = self.packet_populations[pop_idx].total_co2_release() - pre_co2;
            let oxygen_delta = self.packet_populations[pop_idx].total_oxygen_draw() - pre_oxygen;
            let nh4_delta = self.packet_populations[pop_idx].total_ammonium_draw() - pre_nh4;
            let proton_delta = self.packet_populations[pop_idx].total_proton_release() - pre_proton;

            // --- Glucose extraction (existing) ---
            if glucose_delta > 1.0e-8 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Glucose,
                    x,
                    y,
                    z,
                    1,
                    glucose_delta.min(0.003),
                );
            }
            // --- CO2 deposition (existing) ---
            if co2_delta > 1.0e-8 {
                let _ = self.substrate.deposit_patch_species(
                    TerrariumSpecies::CarbonDioxide,
                    x,
                    y,
                    z,
                    1,
                    co2_delta.min(0.003),
                );
            }
            // --- Oxygen consumption (was tracked but never wired) ---
            if oxygen_delta > 1.0e-8 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::OxygenGas,
                    x,
                    y,
                    z,
                    1,
                    oxygen_delta.min(0.003),
                );
            }
            // --- Ammonium draw for nitrogen metabolism (Redfield-like) ---
            if nh4_delta > 1.0e-9 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Ammonium,
                    x,
                    y,
                    z,
                    1,
                    nh4_delta.min(0.001),
                );
            }
            // --- Proton release from metabolic acidification ---
            if proton_delta > 1.0e-9 {
                let _ = self.substrate.deposit_patch_species(
                    TerrariumSpecies::Proton,
                    x,
                    y,
                    z,
                    1,
                    proton_delta.min(0.001),
                );
            }
        }

        // --- Budding: high-activity populations spread to neighbors ---
        let mut buds: Vec<(usize, usize, usize, GenotypePacket)> = Vec::new();
        for pop in &mut self.packet_populations {
            let flat = idx2(width, pop.x, pop.y);
            if matches!(
                self.ownership[flat].owner,
                SoilOwnershipClass::ExplicitMicrobeCohort { .. }
            ) {
                continue;
            }
            if pop.total_cells > 10.0 && pop.age_s > 5.0 {
                if let Some((nx, ny, daughter)) = pop.try_bud(width, height) {
                    buds.push((nx, ny, pop.z, daughter));
                }
            }
        }
        // Insert buds into existing or new populations at target coordinates
        for (nx, ny, nz, packet) in buds {
            let flat = idx2(width, nx, ny);
            if matches!(
                self.ownership[flat].owner,
                SoilOwnershipClass::ExplicitMicrobeCohort { .. }
                    | SoilOwnershipClass::PlantTissueRegion { .. }
                    | SoilOwnershipClass::AtomisticProbeRegion { .. }
            ) {
                continue;
            }
            if let Some(target) = self
                .packet_populations
                .iter_mut()
                .find(|p| p.x == nx && p.y == ny)
            {
                if target.packets.len() < GENOTYPE_PACKET_MAX_PER_CELL {
                    target.packets.push(packet);
                    target.recompute_total();
                }
            } else if self.packet_populations.len() < GENOTYPE_PACKET_POPULATION_MAX_CELLS {
                let mut new_pop = GenotypePacketPopulation::new(nx, ny, nz);
                new_pop.packets.push(packet);
                new_pop.recompute_total();
                self.packet_populations.push(new_pop);
            }
        }

        self.packet_populations.retain(|pop| {
            let flat = idx2(width, pop.x, pop.y);
            !matches!(
                self.ownership[flat].owner,
                SoilOwnershipClass::PlantTissueRegion { .. }
                    | SoilOwnershipClass::AtomisticProbeRegion { .. }
            )
        });
        self.packet_populations.retain(|pop| pop.is_alive());
        self.sync_packet_population_ownership();
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn reconcile_owned_summary_pools_from_substrate(&mut self) {
        project_owned_summary_pools(
            OwnedSummaryProjectionConfig {
                width: self.config.width,
                height: self.config.height,
                depth: self.config.depth,
                ownership_threshold: EXPLICIT_OWNERSHIP_THRESHOLD,
            },
            OwnedSummaryProjectionInputs {
                explicit_microbe_authority: &self.explicit_microbe_authority,
                ammonium: self.substrate.species_field(TerrariumSpecies::Ammonium),
                nitrate: self.substrate.species_field(TerrariumSpecies::Nitrate),
                phosphorus: self.substrate.species_field(TerrariumSpecies::Phosphorus),
                glucose: self.substrate.species_field(TerrariumSpecies::Glucose),
                carbon_dioxide: self
                    .substrate
                    .species_field(TerrariumSpecies::CarbonDioxide),
                atp_flux: self.substrate.species_field(TerrariumSpecies::AtpFlux),
            },
            OwnedSummaryProjectionOutputs {
                root_exudates: &mut self.root_exudates,
                litter_carbon: &mut self.litter_carbon,
                dissolved_nutrients: &mut self.dissolved_nutrients,
                shallow_nutrients: &mut self.shallow_nutrients,
                mineral_nitrogen: &mut self.mineral_nitrogen,
                organic_matter: &mut self.organic_matter,
            },
        )
        .expect("terrarium owned summary pools should match substrate dimensions");
    }

    /// Promote high-fitness coarse packets to explicit WholeCellSimulator cohorts.
    #[allow(dead_code)]
    pub(super) fn promote_qualified_packets(&mut self) -> Result<(), String> {
        // Requires add_explicit_microbe from explicit_microbe_impl.rs (still gated).
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shoreline_absorption_drives_distinct_background_microbiomes() {
        let mut world =
            TerrariumWorld::demo_preset(42, false, TerrariumDemoPreset::Demo).unwrap();
        let width = world.config.width;
        let height = world.config.height;
        let shore_idx = (0..width * height)
            .filter(|idx| world.water_mask[*idx] < 0.30)
            .max_by(|a, b| {
                shoreline_water_signal(width, height, &world.water_mask, *a)
                    .partial_cmp(&shoreline_water_signal(
                        width,
                        height,
                        &world.water_mask,
                        *b,
                    ))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("micro terrarium should expose a shoreline cell");
        let dry_idx = (0..width * height)
            .filter(|idx| world.water_mask[*idx] < 0.02)
            .min_by(|a, b| {
                shoreline_water_signal(width, height, &world.water_mask, *a)
                    .partial_cmp(&shoreline_water_signal(
                        width,
                        height,
                        &world.water_mask,
                        *b,
                    ))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("micro terrarium should expose a dry background cell");

        world
            .step_broad_soil(world.config.world_dt_s * world.config.time_warp)
            .unwrap();

        let shore_moisture = world.moisture[shore_idx] + world.deep_moisture[shore_idx];
        let dry_moisture = world.moisture[dry_idx] + world.deep_moisture[dry_idx];
        let trait_divergence = (world.denitrifier_anoxic_fraction[shore_idx]
            - world.denitrifier_anoxic_fraction[dry_idx])
            .abs()
            + (world.nitrifier_aerobic_fraction[shore_idx]
                - world.nitrifier_aerobic_fraction[dry_idx])
                .abs();
        let shore_process_signal =
            world.denitrification_potential[shore_idx] + world.nitrification_potential[shore_idx];
        let dry_process_signal =
            world.denitrification_potential[dry_idx] + world.nitrification_potential[dry_idx];

        assert!(
            shore_moisture > dry_moisture,
            "shoreline cell should hold more water than dry background"
        );
        assert!(
            trait_divergence > 0.01 || shore_process_signal > dry_process_signal,
            "shoreline chemistry should bias a distinct background microbiome"
        );
    }

    #[test]
    fn shoreline_ecotones_recruit_owned_packet_regions() {
        let mut world =
            TerrariumWorld::demo_preset(11, false, TerrariumDemoPreset::Demo).unwrap();
        let width = world.config.width;
        let height = world.config.height;
        let shore_idx = (0..width * height)
            .filter(|idx| world.water_mask[*idx] < 0.30)
            .max_by(|a, b| {
                shoreline_water_signal(width, height, &world.water_mask, *a)
                    .partial_cmp(&shoreline_water_signal(
                        width,
                        height,
                        &world.water_mask,
                        *b,
                    ))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("micro terrarium should expose a shoreline cell");
        let sx = shore_idx % width;
        let sy = shore_idx / width;

        for yy in sy.saturating_sub(1)..=(sy + 1).min(height.saturating_sub(1)) {
            for xx in sx.saturating_sub(1)..=(sx + 1).min(width.saturating_sub(1)) {
                let idx = idx2(width, xx, yy);
                world.microbial_packets[idx] = 80.0;
                world.nitrifier_packets[idx] = 20.0;
                world.denitrifier_packets[idx] = 24.0;
                world.microbial_cells[idx] = 40_000.0;
                world.nitrifier_cells[idx] = 16_000.0;
                world.denitrifier_cells[idx] = 18_000.0;
            }
        }

        world.microbial_packets[shore_idx] = 2_400.0;
        world.nitrifier_packets[shore_idx] = 1_250.0;
        world.denitrifier_packets[shore_idx] = 1_450.0;
        world.microbial_cells[shore_idx] = 1.1e6;
        world.nitrifier_cells[shore_idx] = 5.8e5;
        world.denitrifier_cells[shore_idx] = 6.6e5;
        world.microbial_vitality[shore_idx] = 0.88;
        world.nitrifier_vitality[shore_idx] = 0.82;
        world.denitrifier_vitality[shore_idx] = 0.86;
        world.microbial_reserve[shore_idx] = 0.74;
        world.nitrifier_reserve[shore_idx] = 0.70;
        world.denitrifier_reserve[shore_idx] = 0.78;
        world.microbial_packet_mutation_flux[shore_idx] = 0.00018;
        world.nitrifier_packet_mutation_flux[shore_idx] = 0.00015;
        world.denitrifier_packet_mutation_flux[shore_idx] = 0.00016;
        world.nitrification_potential[shore_idx] = 0.00010;
        world.denitrification_potential[shore_idx] = 0.00011;
        world.moisture[shore_idx] = 0.34;
        world.deep_moisture[shore_idx] = 0.46;

        world.packet_populations.clear();
        world.explicit_microbe_authority.fill(0.0);
        world.explicit_microbe_activity.fill(0.0);
        world.recruit_packet_populations();

        let pop = world
            .packet_population_at(sx, sy)
            .expect("shoreline hotspot should recruit a packet population");
        assert!(matches!(
            world.ownership[shore_idx].owner,
            SoilOwnershipClass::GenotypePacketRegion { .. }
        ));
        assert!(
            world.explicit_microbe_authority[shore_idx] >= EXPLICIT_OWNERSHIP_THRESHOLD,
            "shoreline packet region did not gain explicit authority"
        );
        assert!(
            pop.ecology_diversity() > 0.20,
            "shoreline packet region should carry a mixed ecotone community"
        );
    }

    #[test]
    fn packet_owned_shoreline_cells_survive_authority_rebuild() {
        let mut world =
            TerrariumWorld::demo_preset(13, false, TerrariumDemoPreset::Demo).unwrap();
        let width = world.config.width;
        let height = world.config.height;
        let shore_idx = (0..width * height)
            .filter(|idx| world.water_mask[*idx] < 0.30)
            .max_by(|a, b| {
                shoreline_water_signal(width, height, &world.water_mask, *a)
                    .partial_cmp(&shoreline_water_signal(
                        width,
                        height,
                        &world.water_mask,
                        *b,
                    ))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("micro terrarium should expose a shoreline cell");
        let sx = shore_idx % width;
        let sy = shore_idx / width;

        world.microbial_packets[shore_idx] = 2_100.0;
        world.nitrifier_packets[shore_idx] = 1_000.0;
        world.denitrifier_packets[shore_idx] = 1_100.0;
        world.microbial_cells[shore_idx] = 1.0e6;
        world.nitrifier_cells[shore_idx] = 5.0e5;
        world.denitrifier_cells[shore_idx] = 5.4e5;
        world.microbial_vitality[shore_idx] = 0.86;
        world.nitrifier_vitality[shore_idx] = 0.80;
        world.denitrifier_vitality[shore_idx] = 0.84;
        world.microbial_reserve[shore_idx] = 0.70;
        world.nitrifier_reserve[shore_idx] = 0.68;
        world.denitrifier_reserve[shore_idx] = 0.74;
        world.microbial_packet_mutation_flux[shore_idx] = 0.00017;
        world.nitrifier_packet_mutation_flux[shore_idx] = 0.00014;
        world.denitrifier_packet_mutation_flux[shore_idx] = 0.00015;
        world.nitrification_potential[shore_idx] = 0.00009;
        world.denitrification_potential[shore_idx] = 0.00010;
        world.moisture[shore_idx] = 0.33;
        world.deep_moisture[shore_idx] = 0.45;

        world.recruit_packet_populations();
        assert!(world.packet_population_at(sx, sy).is_some());

        world.explicit_microbe_authority[shore_idx] = 0.0;
        world.explicit_microbe_activity[shore_idx] = 0.0;
        world.rebuild_explicit_microbe_fields();

        assert!(
            world.explicit_microbe_authority[shore_idx] >= EXPLICIT_OWNERSHIP_THRESHOLD,
            "packet-owned shoreline authority collapsed after rebuild"
        );
        assert!(
            world.explicit_microbe_activity[shore_idx] > 0.0,
            "packet-owned shoreline activity was not restored"
        );
    }
}
