//! Secondary genotype helper functions, constants, and the PublicSecondaryBanks type.
//!
//! Extracted from terrarium_world.rs to reduce file size.

#![allow(dead_code)]

use crate::constants::clamp;

// ── Self-contained type definitions ──
// Defined locally to avoid dependency on volatile soil_broad.rs.

pub(crate) const INTERNAL_SECONDARY_GENOTYPE_AXES: usize = 6;
pub(crate) const PUBLIC_STRAIN_BANKS: usize = 3;

#[derive(Debug, Clone, Copy)]
pub(crate) struct SecondaryGenotypeRecord {
    pub(crate) genes: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES],
    pub(crate) genotype_divergence: f32,
    pub(crate) generation: f32,
    pub(crate) novelty: f32,
}

impl Default for SecondaryGenotypeRecord {
    fn default() -> Self {
        Self { genes: [0.5; INTERNAL_SECONDARY_GENOTYPE_AXES], genotype_divergence: 0.0, generation: 0.0, novelty: 0.0 }
    }
}

impl SecondaryGenotypeRecord {
    pub(crate) fn default_genes() -> [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] {
        [0.5; INTERNAL_SECONDARY_GENOTYPE_AXES]
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SecondaryGenotypeCatalogRecord {
    pub(crate) catalog_id: u32,
    pub(crate) parent_catalog_id: u32,
    pub(crate) catalog_divergence: f32,
    pub(crate) generation: f32,
    pub(crate) novelty: f32,
    pub(crate) local_bank_id: u32,
    pub(crate) local_bank_share: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SecondaryCatalogBankEntryRecord {
    pub(crate) genotype_id: u32,
    pub(crate) lineage_id: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SecondaryCatalogBankEntry {
    pub(crate) occupancy: u32,
    pub(crate) packet_mass: f32,
    pub(crate) record: SecondaryGenotypeRecord,
    pub(crate) catalog: SecondaryGenotypeCatalogRecord,
    pub(crate) genes: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES],
}

impl SecondaryCatalogBankEntry {
    pub(crate) fn empty() -> Self { Self::default() }
}

#[derive(Debug, Clone)]
pub(crate) struct SecondaryGenotypeEntry {
    pub(crate) catalog_slot: u32,
    pub(crate) genes: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES],
    pub(crate) record: SecondaryGenotypeRecord,
    pub(crate) catalog: SecondaryGenotypeCatalogRecord,
}

pub(crate) struct GroupedSecondaryBankRefs<'a> {
    pub(crate) packets: [&'a [f32]; PUBLIC_STRAIN_BANKS],
    pub(crate) trait_a: [&'a [f32]; PUBLIC_STRAIN_BANKS],
    pub(crate) trait_b: [&'a [f32]; PUBLIC_STRAIN_BANKS],
    pub(crate) catalog_slots: [&'a [u32]; PUBLIC_STRAIN_BANKS],
    pub(crate) catalog_entries: [&'a [SecondaryCatalogBankEntry]; PUBLIC_STRAIN_BANKS],
}

#[derive(Debug, Clone)]
pub(crate) struct SoilBroadSecondaryBanks {
    pub(crate) bank_packets: Vec<Vec<f32>>,
    pub(crate) bank_trait_a: Vec<Vec<f32>>,
    pub(crate) bank_trait_b: Vec<Vec<f32>>,
    pub(crate) bank_catalog_entries: Vec<Vec<SecondaryCatalogBankEntry>>,
    pub(crate) bank_catalog_slots: Vec<Vec<u32>>,
}

impl SoilBroadSecondaryBanks {
    pub(crate) fn len(&self) -> usize { self.bank_packets.len() }
    pub(crate) fn bank_packets(&self, idx: usize) -> &[f32] { &self.bank_packets[idx] }
    pub(crate) fn bank_trait_a(&self, idx: usize) -> &[f32] { &self.bank_trait_a[idx] }
    pub(crate) fn bank_trait_b(&self, idx: usize) -> &[f32] { &self.bank_trait_b[idx] }
    pub(crate) fn bank_catalog_entries(&self, idx: usize) -> &[SecondaryCatalogBankEntry] { &self.bank_catalog_entries[idx] }
    pub(crate) fn bank_catalog_slots(&self, idx: usize) -> &[u32] { &self.bank_catalog_slots[idx] }
}

pub(crate) fn refresh_secondary_local_catalog_identity(
    _packets: &[f32],
    entries: &mut [SecondaryGenotypeEntry],
) -> Vec<SecondaryCatalogBankEntry> {
    entries.iter().map(|_| SecondaryCatalogBankEntry::default()).collect()
}

// ── Bank index constants ──

pub(crate) const SHADOW_BANK_IDX: usize = 0;
pub(crate) const VARIANT_BANK_IDX: usize = 1;
pub(crate) const NOVEL_BANK_IDX: usize = 2;

// ── Gene weight constants ──

pub(crate) const MICROBIAL_GENE_CATABOLIC_WEIGHTS: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] =
    [0.60, -0.08, 0.18, -0.04, 0.16, -0.03];
pub(crate) const MICROBIAL_GENE_STRESS_RESPONSE_WEIGHTS: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] =
    [-0.06, 0.56, 0.08, 0.24, -0.04, 0.16];
pub(crate) const MICROBIAL_GENE_DORMANCY_MAINTENANCE_WEIGHTS: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] =
    [-0.10, 0.12, 0.34, 0.28, 0.10, 0.18];
pub(crate) const MICROBIAL_GENE_EXTRACELLULAR_SCAVENGING_WEIGHTS: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] =
    [0.14, 0.04, 0.10, 0.06, 0.40, 0.22];
pub(crate) const NITRIFIER_GENE_OXYGEN_RESPIRATION_WEIGHTS: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] =
    [0.62, -0.06, 0.16, -0.02, 0.18, -0.02];
pub(crate) const NITRIFIER_GENE_AMMONIUM_TRANSPORT_WEIGHTS: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] =
    [-0.04, 0.56, 0.10, 0.14, 0.08, 0.12];
pub(crate) const NITRIFIER_GENE_STRESS_PERSISTENCE_WEIGHTS: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] =
    [0.02, 0.14, 0.32, 0.28, 0.12, 0.18];
pub(crate) const NITRIFIER_GENE_REDOX_EFFICIENCY_WEIGHTS: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] =
    [0.24, 0.10, 0.12, 0.06, 0.38, 0.18];
pub(crate) const DENITRIFIER_GENE_ANOXIA_RESPIRATION_WEIGHTS: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] =
    [0.60, -0.06, 0.10, 0.18, 0.16, -0.02];
pub(crate) const DENITRIFIER_GENE_NITRATE_TRANSPORT_WEIGHTS: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] =
    [-0.02, 0.58, 0.08, 0.16, 0.08, 0.10];
pub(crate) const DENITRIFIER_GENE_STRESS_PERSISTENCE_WEIGHTS: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] =
    [0.04, 0.14, 0.30, 0.30, 0.10, 0.20];
pub(crate) const DENITRIFIER_GENE_REDUCTIVE_FLEXIBILITY_WEIGHTS: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES] =
    [0.18, 0.16, 0.16, 0.18, 0.34, 0.20];

// ── Pure helper functions ──

pub(crate) fn secondary_gene_axis_center(axis_idx: usize) -> f32 {
    match axis_idx {
        0 | 1 => 0.0,
        _ => 0.5,
    }
}

pub(crate) fn decode_secondary_gene_module(
    genes: &[f32; INTERNAL_SECONDARY_GENOTYPE_AXES],
    weights: &[f32; INTERNAL_SECONDARY_GENOTYPE_AXES],
) -> f32 {
    let mut value = 0.16f32;
    for axis_idx in 0..INTERNAL_SECONDARY_GENOTYPE_AXES {
        value += (genes[axis_idx].clamp(0.0, 1.0) - secondary_gene_axis_center(axis_idx))
            * weights[axis_idx];
    }
    clamp(value, 0.0, 1.0)
}

pub(crate) fn offset_clamped(value: usize, delta: isize, upper_exclusive: usize) -> usize {
    let shifted = value as isize + delta;
    shifted.clamp(0, upper_exclusive.saturating_sub(1) as isize) as usize
}

pub(crate) fn temp_response(temp_c: f32, optimum: f32, width: f32) -> f32 {
    let delta = (temp_c - optimum) / width.max(1e-6);
    (-delta * delta).exp()
}

pub(crate) fn packet_load(cells: f32, packets: f32) -> f32 {
    cells.max(0.0) / packets.max(0.05)
}

pub(crate) fn packet_surface_factor(cells: f32, packets: f32, target_cells: f32) -> f32 {
    (0.78 + (target_cells / packet_load(cells, packets).max(0.25)).sqrt() * 0.26).clamp(0.62, 1.35)
}

pub(crate) fn trait_match(current: f32, target: f32) -> f32 {
    (1.12 - (current - target).abs() * 0.85).clamp(0.55, 1.25)
}

pub(crate) fn bank_primary_packets(total_packets: f32, secondary_packets: &[f32]) -> f32 {
    let secondary_total = secondary_packets
        .iter()
        .copied()
        .map(|value| value.max(0.0))
        .sum::<f32>();
    (total_packets.max(0.05) - secondary_total).max(0.0)
}

pub(crate) fn bank_simpson_diversity(total_packets: f32, secondary_packets: &[f32]) -> f32 {
    let total_packets = total_packets.max(0.05);
    let primary_packets = bank_primary_packets(total_packets, secondary_packets);
    let combined_packets = (primary_packets
        + secondary_packets
            .iter()
            .copied()
            .map(|value| value.max(0.0))
            .sum::<f32>())
    .max(0.05);
    let mut squared = (primary_packets / combined_packets).powi(2);
    for packets in secondary_packets {
        squared += (packets.max(0.0) / combined_packets).powi(2);
    }
    clamp(
        (1.0 - squared) / (1.0 - 1.0 / (secondary_packets.len() as f32 + 1.0)).max(1.0e-6),
        0.0,
        1.0,
    )
}

pub(crate) fn bank_weighted_trait_mean(
    total_packets: f32,
    primary_trait: f32,
    secondary_packets: &[f32],
    secondary_traits: &[f32],
) -> f32 {
    let primary_packets = bank_primary_packets(total_packets, secondary_packets);
    let combined_packets = (primary_packets
        + secondary_packets
            .iter()
            .copied()
            .map(|value| value.max(0.0))
            .sum::<f32>())
    .max(0.05);
    let mut weighted = primary_trait.clamp(0.0, 1.0) * (primary_packets / combined_packets);
    for (packets, trait_value) in secondary_packets.iter().zip(secondary_traits.iter()) {
        weighted += trait_value.clamp(0.0, 1.0) * (packets.max(0.0) / combined_packets);
    }
    weighted.clamp(0.0, 1.0)
}

pub(crate) fn secondary_bank_catalog_signature(
    bank_idx: usize,
    genes: &[f32; INTERNAL_SECONDARY_GENOTYPE_AXES],
) -> u32 {
    let mut hash = 0x9E37_79B9u32 ^ (bank_idx as u32).wrapping_add(17).wrapping_mul(0x85EB_CA6B);
    for (axis_idx, gene) in genes.iter().enumerate() {
        let quantized = (gene.clamp(0.0, 1.0) * 63.0).round() as u32;
        hash ^= quantized.wrapping_add((axis_idx as u32).wrapping_add(5).wrapping_mul(313));
        hash = hash.rotate_left(7).wrapping_mul(0x27D4_EB2D);
    }
    if hash == 0 {
        1
    } else {
        hash
    }
}

pub(crate) fn secondary_bank_catalog_divergence(genes: &[f32; INTERNAL_SECONDARY_GENOTYPE_AXES]) -> f32 {
    let coarse_dispersion = genes
        .iter()
        .enumerate()
        .map(|(axis_idx, gene)| {
            let coarse = (gene.clamp(0.0, 1.0) * 63.0).round() / 63.0;
            let center = 0.5
                + (axis_idx as f32 - (INTERNAL_SECONDARY_GENOTYPE_AXES as f32 - 1.0) * 0.5) * 0.035;
            (coarse - center.clamp(0.1, 0.9)).abs()
        })
        .sum::<f32>()
        / INTERNAL_SECONDARY_GENOTYPE_AXES as f32;
    clamp(coarse_dispersion * 1.55, 0.0, 1.0)
}

// ── Types ──

#[derive(Debug, Clone)]
pub(crate) struct PublicSecondaryBankEntry {
    pub(crate) packets: Vec<f32>,
    pub(crate) trait_a: Vec<f32>,
    pub(crate) trait_b: Vec<f32>,
    pub(crate) catalog_bank: Vec<SecondaryCatalogBankEntry>,
    pub(crate) catalog_slots: Vec<u32>,
}

#[derive(Debug, Clone)]
pub(crate) struct PublicSecondaryBanks {
    pub(crate) banks: Vec<PublicSecondaryBankEntry>,
}

impl PublicSecondaryBanks {
    pub(crate) fn seed_internal_genotype_axes(
        trait_a: &[f32],
        trait_b: &[f32],
        gene_a: &[f32],
        gene_b: &[f32],
    ) -> Vec<Vec<f32>> {
        let mut axes = Vec::with_capacity(INTERNAL_SECONDARY_GENOTYPE_AXES);
        axes.push(gene_a.to_vec());
        axes.push(gene_b.to_vec());
        axes.push(
            trait_a
                .iter()
                .zip(gene_a.iter())
                .map(|(trait_value, gene_value)| {
                    clamp(0.52 * *trait_value + 0.48 * *gene_value, 0.0, 1.0)
                })
                .collect(),
        );
        axes.push(
            trait_b
                .iter()
                .zip(gene_b.iter())
                .map(|(trait_value, gene_value)| {
                    clamp(0.52 * *trait_value + 0.48 * *gene_value, 0.0, 1.0)
                })
                .collect(),
        );
        axes.push(
            trait_a
                .iter()
                .zip(trait_b.iter())
                .zip(gene_a.iter().zip(gene_b.iter()))
                .map(
                    |((trait_a_value, trait_b_value), (gene_a_value, gene_b_value))| {
                        clamp(
                            0.26 * *trait_a_value
                                + 0.18 * *trait_b_value
                                + 0.30 * *gene_a_value
                                + 0.26 * *gene_b_value,
                            0.0,
                            1.0,
                        )
                    },
                )
                .collect(),
        );
        axes.push(
            trait_a
                .iter()
                .zip(trait_b.iter())
                .zip(gene_a.iter().zip(gene_b.iter()))
                .map(
                    |((trait_a_value, trait_b_value), (gene_a_value, gene_b_value))| {
                        clamp(
                            0.16 * *trait_a_value
                                + 0.34 * *trait_b_value
                                + 0.14 * (1.0 - *gene_a_value)
                                + 0.36 * *gene_b_value,
                            0.0,
                            1.0,
                        )
                    },
                )
                .collect(),
        );
        axes
    }

    pub(crate) fn new(total: usize) -> Self {
        let empty_bank = || PublicSecondaryBankEntry {
            packets: vec![0.0; total],
            trait_a: vec![0.5; total],
            trait_b: vec![0.5; total],
            catalog_bank: vec![SecondaryCatalogBankEntry::default(); total.min(64)],
            catalog_slots: vec![0u32; total],
        };
        Self {
            banks: (0..PUBLIC_STRAIN_BANKS).map(|_| empty_bank()).collect(),
        }
    }

        pub(crate) fn from_compat_parts(
        packets: [Vec<f32>; PUBLIC_STRAIN_BANKS],
        trait_a: [Vec<f32>; PUBLIC_STRAIN_BANKS],
        trait_b: [Vec<f32>; PUBLIC_STRAIN_BANKS],
        gene_a: [Vec<f32>; PUBLIC_STRAIN_BANKS],
        gene_b: [Vec<f32>; PUBLIC_STRAIN_BANKS],
    ) -> Self {
        Self {
            banks: (0..PUBLIC_STRAIN_BANKS)
                .map(|bank_idx| {
                    let genotype_axes = Self::seed_internal_genotype_axes(
                        &trait_a[bank_idx],
                        &trait_b[bank_idx],
                        &gene_a[bank_idx],
                        &gene_b[bank_idx],
                    );
                    let mut genotype_entries = (0..packets[bank_idx].len())
                        .map(|idx| {
                            let genes =
                                std::array::from_fn(|axis_idx| genotype_axes[axis_idx][idx]);
                            SecondaryGenotypeEntry {
                                catalog_slot: 0,
                                genes,
                                record: SecondaryGenotypeRecord::default(),
                                catalog: SecondaryGenotypeCatalogRecord {
                                    catalog_id: secondary_bank_catalog_signature(bank_idx, &genes),
                                    parent_catalog_id: secondary_bank_catalog_signature(
                                        bank_idx, &genes,
                                    ),
                                    catalog_divergence: secondary_bank_catalog_divergence(&genes),
                                    generation: 0.0,
                                    novelty: 0.0,
                                    local_bank_id: 0,
                                    local_bank_share: 0.0,
                                },
                            }
                        })
                        .collect::<Vec<_>>();
                    let catalog_bank = refresh_secondary_local_catalog_identity(
                        &packets[bank_idx],
                        &mut genotype_entries,
                    );
                    let catalog_slots = genotype_entries
                        .iter()
                        .map(|entry| entry.catalog_slot)
                        .collect();
                    PublicSecondaryBankEntry {
                        packets: packets[bank_idx].clone(),
                        trait_a: trait_a[bank_idx].clone(),
                        trait_b: trait_b[bank_idx].clone(),
                        catalog_bank,
                        catalog_slots,
                    }
                })
                .collect(),
        }
    }

    pub(crate) fn grouped_refs(&self) -> GroupedSecondaryBankRefs<'_> {
        GroupedSecondaryBankRefs {
            packets: std::array::from_fn(|bank_idx| self.banks[bank_idx].packets.as_slice()),
            trait_a: std::array::from_fn(|bank_idx| self.banks[bank_idx].trait_a.as_slice()),
            trait_b: std::array::from_fn(|bank_idx| self.banks[bank_idx].trait_b.as_slice()),
            catalog_slots: std::array::from_fn(|bank_idx| {
                self.banks[bank_idx].catalog_slots.as_slice()
            }),
            catalog_entries: std::array::from_fn(|bank_idx| {
                self.banks[bank_idx].catalog_bank.as_slice()
            }),
        }
    }

    pub(crate) fn packets_banks(&self) -> [&[f32]; PUBLIC_STRAIN_BANKS] {
        std::array::from_fn(|bank_idx| self.banks[bank_idx].packets.as_slice())
    }

    pub(crate) fn trait_a_banks(&self) -> [&[f32]; PUBLIC_STRAIN_BANKS] {
        std::array::from_fn(|bank_idx| self.banks[bank_idx].trait_a.as_slice())
    }

    pub(crate) fn trait_b_banks(&self) -> [&[f32]; PUBLIC_STRAIN_BANKS] {
        std::array::from_fn(|bank_idx| self.banks[bank_idx].trait_b.as_slice())
    }

    pub(crate) fn catalog_slot_banks(&self) -> [&[u32]; PUBLIC_STRAIN_BANKS] {
        std::array::from_fn(|bank_idx| self.banks[bank_idx].catalog_slots.as_slice())
    }

    pub(crate) fn catalog_bank_entries(&self) -> [&[SecondaryCatalogBankEntry]; PUBLIC_STRAIN_BANKS] {
        std::array::from_fn(|bank_idx| self.banks[bank_idx].catalog_bank.as_slice())
    }

    pub(crate) fn dominant_catalog_entry_at(
        &self,
        idx: usize,
    ) -> Option<(usize, f32, SecondaryCatalogBankEntry)> {
        let mut best = None;
        for bank_idx in 0..self.banks.len().min(PUBLIC_STRAIN_BANKS) {
            let packets = self.banks[bank_idx]
                .packets
                .get(idx)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            if packets <= 0.0 {
                continue;
            }
            let slot = self.banks[bank_idx]
                .catalog_slots
                .get(idx)
                .copied()
                .unwrap_or(0) as usize;
            let entry = self.banks[bank_idx]
                .catalog_bank
                .get(slot)
                .copied()
                .unwrap_or_default();
            if best
                .as_ref()
                .map(|(_, best_packets, _)| packets > *best_packets)
                .unwrap_or(true)
            {
                best = Some((bank_idx, packets, entry));
            }
        }
        best
    }

    /// Return all catalog bank entries active at a given cell index,
    /// one per strain bank. Used by Phase 4 packet seeding.
    pub(crate) fn catalog_entries_at(&self, idx: usize) -> Vec<SecondaryCatalogBankEntry> {
        let mut entries = Vec::with_capacity(PUBLIC_STRAIN_BANKS);
        for bank_idx in 0..self.banks.len().min(PUBLIC_STRAIN_BANKS) {
            let packets = self.banks[bank_idx]
                .packets
                .get(idx)
                .copied()
                .unwrap_or(0.0);
            if packets <= 0.0 {
                continue;
            }
            let slot = self.banks[bank_idx]
                .catalog_slots
                .get(idx)
                .copied()
                .unwrap_or(0) as usize;
            if let Some(entry) = self.banks[bank_idx].catalog_bank.get(slot) {
                entries.push(*entry);
            }
        }
        entries
    }

    pub(crate) fn from_grouped(grouped: SoilBroadSecondaryBanks) -> Self {
        Self {
            banks: (0..grouped.len())
                .map(|bank_idx| PublicSecondaryBankEntry {
                    packets: grouped.bank_packets(bank_idx).to_vec(),
                    trait_a: grouped.bank_trait_a(bank_idx).to_vec(),
                    trait_b: grouped.bank_trait_b(bank_idx).to_vec(),
                    catalog_bank: grouped.bank_catalog_entries(bank_idx).to_vec(),
                    catalog_slots: grouped.bank_catalog_slots(bank_idx).to_vec(),
                })
                .collect(),
        }
    }

    pub(crate) fn apply_grouped(&mut self, grouped: SoilBroadSecondaryBanks) {
        *self = Self::from_grouped(grouped);
    }
}

// ── Target functions ──

pub(crate) fn microbial_copiotroph_target(
    substrate_gate: f32,
    moisture_factor: f32,
    oxygen_factor: f32,
    root_factor: f32,
) -> f32 {
    clamp(
        0.18 + substrate_gate * 0.54 + root_factor * 0.06 + moisture_factor * 0.04
            - (0.34 - oxygen_factor).max(0.0) * 0.10,
        0.05,
        0.95,
    )
}

pub(crate) fn nitrifier_aerobic_target(oxygen_factor: f32, aeration_factor: f32, anoxia_factor: f32) -> f32 {
    clamp(
        0.18 + oxygen_factor * 0.56 + aeration_factor * 0.18 - anoxia_factor * 0.22,
        0.05,
        0.98,
    )
}

pub(crate) fn denitrifier_anoxic_target(anoxia_factor: f32, deep_moisture: f32, oxygen_factor: f32) -> f32 {
    clamp(
        0.16 + anoxia_factor * 0.60 + deep_moisture * 0.10 - oxygen_factor * 0.20,
        0.05,
        0.98,
    )
}
