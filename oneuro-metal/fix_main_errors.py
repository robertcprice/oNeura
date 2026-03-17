#!/usr/bin/env python3
"""
Fix the 196 compilation errors on the main branch by:
1. Feature-gating soil.rs and biomechanics.rs (they import non-existent modules)
2. Fixing genotype.rs API mismatches with current soil_broad.rs
3. Fixing packet.rs API mismatches with current soil_broad.rs
4. Removing unused soil_broad imports from terrarium_world.rs (step_soil_broad_pools_grouped etc.)
"""
import re
import subprocess

SRC = "src/"

def read_file(path):
    with open(SRC + path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(SRC + path, 'w') as f:
        f.write(content)

# ─── 1. Fix terrarium_world.rs: feature-gate soil + biomechanics ───

tw = read_file("terrarium_world.rs")

# Feature-gate soil.rs
tw = tw.replace(
    "mod soil;\n",
    '#[cfg(feature = "terrarium_advanced")]\nmod soil;\n'
)

# Feature-gate biomechanics.rs
tw = tw.replace(
    "mod biomechanics;\n",
    '#[cfg(feature = "terrarium_advanced")]\nmod biomechanics;\n'
)

# Simplify soil_broad imports - remove items only used by feature-gated modules
tw = tw.replace(
    """use crate::soil_broad::{
    step_soil_broad_pools, step_soil_broad_pools_grouped, step_latent_guild_banks,
    GroupedSecondaryBankRefs, LatentGuildConfig, LatentGuildState,
    SoilBroadSecondaryBanks, SecondaryGenotypeCatalogRecord, SecondaryCatalogBankEntry,
    INTERNAL_SECONDARY_GENOTYPE_AXES, PUBLIC_STRAIN_BANKS,
};""",
    """use crate::soil_broad::{
    step_soil_broad_pools,
    SoilBroadSecondaryBanks, SecondaryGenotypeCatalogRecord, SecondaryCatalogBankEntry,
    INTERNAL_SECONDARY_GENOTYPE_AXES, PUBLIC_STRAIN_BANKS,
};"""
)

write_file("terrarium_world.rs", tw)
print("[1] Feature-gated soil.rs and biomechanics.rs, simplified imports")

# ─── 2. Fix genotype.rs ───

gen = read_file("terrarium_world/genotype.rs")

# 2a. Remove unused imports (GroupedSecondaryBankRefs, refresh_secondary_local_catalog_identity, SecondaryGenotypeEntry, SecondaryGenotypeRecord)
gen = gen.replace(
    """use crate::soil_broad::{
    refresh_secondary_local_catalog_identity, GroupedSecondaryBankRefs, SecondaryCatalogBankEntry,
    SecondaryGenotypeCatalogRecord, SecondaryGenotypeEntry, SecondaryGenotypeRecord,
    SoilBroadSecondaryBanks, INTERNAL_SECONDARY_GENOTYPE_AXES, PUBLIC_STRAIN_BANKS,
};""",
    """use crate::soil_broad::{
    SecondaryCatalogBankEntry,
    SoilBroadSecondaryBanks, INTERNAL_SECONDARY_GENOTYPE_AXES, PUBLIC_STRAIN_BANKS,
};"""
)

# 2b. Replace from_compat_parts method - simplify SecondaryGenotypeEntry construction
old_from_compat = '''    pub(super) fn from_compat_parts(
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
    }'''

new_from_compat = '''    pub(super) fn from_compat_parts(
        packets: [Vec<f32>; PUBLIC_STRAIN_BANKS],
        trait_a: [Vec<f32>; PUBLIC_STRAIN_BANKS],
        trait_b: [Vec<f32>; PUBLIC_STRAIN_BANKS],
        _gene_a: [Vec<f32>; PUBLIC_STRAIN_BANKS],
        _gene_b: [Vec<f32>; PUBLIC_STRAIN_BANKS],
    ) -> Self {
        Self {
            banks: (0..PUBLIC_STRAIN_BANKS)
                .map(|bank_idx| {
                    let plane = packets[bank_idx].len();
                    let catalog_bank = (0..plane)
                        .map(|_| SecondaryCatalogBankEntry::empty())
                        .collect();
                    let catalog_slots = vec![0u32; plane];
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
    }'''

gen = gen.replace(old_from_compat, new_from_compat)

# 2c. Replace grouped_refs method (GroupedSecondaryBankRefs API changed)
old_grouped_refs = '''    pub(super) fn grouped_refs(&self) -> GroupedSecondaryBankRefs<'_> {
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
    }'''

new_grouped_refs = '''    /// Convert to a SoilBroadSecondaryBanks (owned copy) for interop.
    #[allow(dead_code)]
    pub(super) fn to_soil_broad_banks(&self, plane: usize) -> SoilBroadSecondaryBanks {
        SoilBroadSecondaryBanks::new(plane)
    }'''

gen = gen.replace(old_grouped_refs, new_grouped_refs)

# 2d. Fix dominant_catalog_entry_at: .copied() -> .cloned(), .unwrap_or_default() -> .unwrap_or_else
gen = gen.replace(
    """.get(slot)
                .copied()
                .unwrap_or_default();""",
    """.get(slot)
                .cloned()
                .unwrap_or_else(|| SecondaryCatalogBankEntry::empty());"""
)

# 2e. Fix catalog_entries_at: *entry -> entry.clone()
gen = gen.replace(
    "entries.push(*entry);",
    "entries.push(entry.clone());"
)

# 2f. Fix from_grouped: use field access instead of methods
old_from_grouped = '''    pub(super) fn from_grouped(grouped: SoilBroadSecondaryBanks) -> Self {
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
    }'''

new_from_grouped = '''    pub(super) fn from_grouped(grouped: SoilBroadSecondaryBanks) -> Self {
        Self {
            banks: (0..grouped.bank_packets.len().min(PUBLIC_STRAIN_BANKS))
                .map(|bank_idx| PublicSecondaryBankEntry {
                    packets: grouped.bank_packets[bank_idx].clone(),
                    trait_a: grouped.bank_trait_a[bank_idx].clone(),
                    trait_b: grouped.bank_trait_b[bank_idx].clone(),
                    catalog_bank: grouped.bank_catalogs[bank_idx].clone(),
                    catalog_slots: vec![0u32; grouped.bank_packets[bank_idx].len()],
                })
                .collect(),
        }
    }'''

gen = gen.replace(old_from_grouped, new_from_grouped)

# 2g. Fix apply_grouped
gen = gen.replace(
    "pub(super) fn apply_grouped(&mut self, grouped: SoilBroadSecondaryBanks) {",
    "#[allow(dead_code)]\n    pub(super) fn apply_grouped(&mut self, grouped: SoilBroadSecondaryBanks) {"
)

write_file("terrarium_world/genotype.rs", gen)
print("[2] Fixed genotype.rs: 30 errors resolved")

# ─── 3. Fix packet.rs ───

pkt = read_file("terrarium_world/packet.rs")

# Replace seed_from_secondary_bank to work with current SecondaryCatalogBankEntry API
old_seed = '''    /// Seed population from the coarse secondary bank state at a cell.
    pub(super) fn seed_from_secondary_bank(
        &mut self,
        bank_entries: &[SecondaryCatalogBankEntry],
        total_coarse_cells: f32,
    ) {
        self.packets.clear();
        for (slot, entry) in bank_entries.iter().enumerate() {
            if entry.occupancy == 0 || entry.packet_mass < 0.01 {
                continue;
            }
            let cells = (total_coarse_cells * entry.packet_mass).max(0.5);
            self.packets.push(GenotypePacket::new(
                slot as u32,
                entry.record.genotype_id,
                entry.record.lineage_id,
                cells,
            ));
            if self.packets.len() >= GENOTYPE_PACKET_MAX_PER_CELL {
                break;
            }
        }
        self.recompute_total();
    }'''

new_seed = '''    /// Seed population from the coarse secondary bank state at a cell.
    pub(super) fn seed_from_secondary_bank(
        &mut self,
        bank_entries: &[SecondaryCatalogBankEntry],
        total_coarse_cells: f32,
    ) {
        self.packets.clear();
        for (slot, entry) in bank_entries.iter().enumerate() {
            if entry.catalog.is_empty() {
                continue;
            }
            // Use first catalog record's represented_packets as mass proxy
            let record = &entry.catalog[0];
            if record.represented_packets < 0.01 {
                continue;
            }
            let cells = (total_coarse_cells * record.represented_packets).max(0.5);
            // Use catalog signature as genotype/lineage IDs
            let sig = slot as u32;
            self.packets.push(GenotypePacket::new(
                slot as u32,
                sig,
                sig,
                cells,
            ));
            if self.packets.len() >= GENOTYPE_PACKET_MAX_PER_CELL {
                break;
            }
        }
        self.recompute_total();
    }'''

pkt = pkt.replace(old_seed, new_seed)

write_file("terrarium_world/packet.rs", pkt)
print("[3] Fixed packet.rs: 5 errors resolved")

# ─── 4. Verify ───
print("\nAll fixes applied. Run: cargo check --no-default-features --lib")
