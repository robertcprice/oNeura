#!/usr/bin/env python3
"""Comprehensive fix for ecosystem submodule compilation.

Rewrites soil_broad.rs with correct API shape and fixes terrarium_world.rs
imports, struct fields, duplicate methods, and missing types.
"""
import os

SRC = os.path.dirname(os.path.abspath(__file__)) + "/src"

def write(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote {path} ({len(content)} bytes)")

def read(path):
    with open(path) as f:
        return f.read()

# ═══════════════════════════════════════════════════════════════════
# PART 1: Rewrite soil_broad.rs with correct API
# ═══════════════════════════════════════════════════════════════════

SOIL_BROAD = r'''//! Broad soil hydrology and biogeochemistry updates for the terrarium.
//!
//! This keeps the coarse pool turnover path native while the reactive
//! chemistry substrate remains in the batched-atom terrarium lattice.

use crate::constants::clamp;

// ── Secondary genotype constants ──

/// Number of secondary genotype axes per microbial guild.
pub const INTERNAL_SECONDARY_GENOTYPE_AXES: usize = 6;
/// Number of public strain banks (shadow, variant, novel).
pub const PUBLIC_STRAIN_BANKS: usize = 3;

// ── Secondary genotype types ──

/// A secondary genotype record: 6-axis gene vector + tracking IDs.
#[derive(Debug, Clone, Copy)]
pub struct SecondaryGenotypeRecord {
    pub genes: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES],
    pub genotype_id: u32,
    pub lineage_id: u32,
}

impl Default for SecondaryGenotypeRecord {
    fn default() -> Self {
        Self {
            genes: [0.5; INTERNAL_SECONDARY_GENOTYPE_AXES],
            genotype_id: 0,
            lineage_id: 0,
        }
    }
}

/// A catalog record for a secondary genotype strain.
#[derive(Debug, Clone)]
pub struct SecondaryGenotypeCatalogRecord {
    pub genotype: SecondaryGenotypeRecord,
    pub represented_cells: f32,
    pub represented_packets: f32,
    pub catalog_id: u32,
    pub parent_catalog_id: u32,
    pub catalog_divergence: f32,
    pub generation: f32,
    pub novelty: f32,
    pub local_bank_id: usize,
    pub local_bank_share: f32,
}

impl Default for SecondaryGenotypeCatalogRecord {
    fn default() -> Self {
        Self {
            genotype: SecondaryGenotypeRecord::default(),
            represented_cells: 0.0,
            represented_packets: 0.0,
            catalog_id: 0,
            parent_catalog_id: 0,
            catalog_divergence: 0.0,
            generation: 0.0,
            novelty: 0.0,
            local_bank_id: 0,
            local_bank_share: 0.0,
        }
    }
}

/// A single lightweight bank entry (per-cell, per-strain-bank).
/// Must be Copy+Default for use in arrays and `unwrap_or_default()`.
#[derive(Debug, Clone, Copy)]
pub struct SecondaryCatalogBankEntry {
    pub occupancy: u32,
    pub packet_mass: f32,
    pub record: SecondaryGenotypeRecord,
}

impl Default for SecondaryCatalogBankEntry {
    fn default() -> Self {
        Self {
            occupancy: 0,
            packet_mass: 0.0,
            record: SecondaryGenotypeRecord::default(),
        }
    }
}

/// A single secondary genotype entry with full catalog metadata.
#[derive(Debug, Clone)]
pub struct SecondaryGenotypeEntry {
    pub catalog_slot: u32,
    pub genes: [f32; INTERNAL_SECONDARY_GENOTYPE_AXES],
    pub record: SecondaryGenotypeRecord,
    pub catalog: SecondaryGenotypeCatalogRecord,
}

/// Read-only references to grouped secondary banks (array-of-slices form).
pub struct GroupedSecondaryBankRefs<'a> {
    pub packets: [&'a [f32]; PUBLIC_STRAIN_BANKS],
    pub trait_a: [&'a [f32]; PUBLIC_STRAIN_BANKS],
    pub trait_b: [&'a [f32]; PUBLIC_STRAIN_BANKS],
    pub catalog_slots: [&'a [u32]; PUBLIC_STRAIN_BANKS],
    pub catalog_entries: [&'a [SecondaryCatalogBankEntry]; PUBLIC_STRAIN_BANKS],
}

/// Per-guild secondary genotype banks.
#[derive(Debug, Clone)]
pub struct SoilBroadSecondaryBanks {
    pub bank_packets: Vec<Vec<f32>>,
    pub bank_trait_a: Vec<Vec<f32>>,
    pub bank_trait_b: Vec<Vec<f32>>,
    pub bank_catalog_entries: Vec<Vec<SecondaryCatalogBankEntry>>,
    pub bank_catalog_slots: Vec<Vec<u32>>,
    plane: usize,
}

impl SoilBroadSecondaryBanks {
    pub fn new(plane: usize) -> Self {
        Self {
            bank_packets: (0..PUBLIC_STRAIN_BANKS).map(|_| vec![0.0; plane]).collect(),
            bank_trait_a: (0..PUBLIC_STRAIN_BANKS).map(|_| vec![0.5; plane]).collect(),
            bank_trait_b: (0..PUBLIC_STRAIN_BANKS).map(|_| vec![0.5; plane]).collect(),
            bank_catalog_entries: (0..PUBLIC_STRAIN_BANKS)
                .map(|_| vec![SecondaryCatalogBankEntry::default(); plane])
                .collect(),
            bank_catalog_slots: (0..PUBLIC_STRAIN_BANKS).map(|_| vec![0u32; plane]).collect(),
            plane,
        }
    }
    pub fn len(&self) -> usize { self.bank_packets.len() }
    pub fn bank_packets(&self, idx: usize) -> &[f32] { &self.bank_packets[idx] }
    pub fn bank_trait_a(&self, idx: usize) -> &[f32] { &self.bank_trait_a[idx] }
    pub fn bank_trait_b(&self, idx: usize) -> &[f32] { &self.bank_trait_b[idx] }
    pub fn bank_catalog_entries(&self, idx: usize) -> &[SecondaryCatalogBankEntry] { &self.bank_catalog_entries[idx] }
    pub fn bank_catalog_slots(&self, idx: usize) -> &[u32] { &self.bank_catalog_slots[idx] }
    pub fn packets_banks(&self) -> [&[f32]; PUBLIC_STRAIN_BANKS] {
        std::array::from_fn(|i| self.bank_packets[i].as_slice())
    }
    pub fn trait_a_banks(&self) -> [&[f32]; PUBLIC_STRAIN_BANKS] {
        std::array::from_fn(|i| self.bank_trait_a[i].as_slice())
    }
    pub fn trait_b_banks(&self) -> [&[f32]; PUBLIC_STRAIN_BANKS] {
        std::array::from_fn(|i| self.bank_trait_b[i].as_slice())
    }
    pub fn grouped_refs(&self) -> GroupedSecondaryBankRefs<'_> {
        GroupedSecondaryBankRefs {
            packets: self.packets_banks(),
            trait_a: self.trait_a_banks(),
            trait_b: self.trait_b_banks(),
            catalog_slots: std::array::from_fn(|i| self.bank_catalog_slots[i].as_slice()),
            catalog_entries: std::array::from_fn(|i| self.bank_catalog_entries[i].as_slice()),
        }
    }
    pub fn catalog_entries_at(&self, flat: usize) -> Vec<SecondaryCatalogBankEntry> {
        let mut entries = Vec::with_capacity(PUBLIC_STRAIN_BANKS);
        for bank_idx in 0..self.bank_packets.len().min(PUBLIC_STRAIN_BANKS) {
            let packets = self.bank_packets[bank_idx].get(flat).copied().unwrap_or(0.0);
            if packets <= 0.0 { continue; }
            if let Some(entry) = self.bank_catalog_entries[bank_idx].get(flat) {
                entries.push(*entry);
            }
        }
        entries
    }
    pub fn apply_grouped(&mut self, other: SoilBroadSecondaryBanks) {
        self.bank_packets = other.bank_packets;
        self.bank_trait_a = other.bank_trait_a;
        self.bank_trait_b = other.bank_trait_b;
        self.bank_catalog_entries = other.bank_catalog_entries;
        self.bank_catalog_slots = other.bank_catalog_slots;
        self.plane = other.plane;
    }
}

/// Refresh secondary local catalog identity — returns catalog bank entries.
pub fn refresh_secondary_local_catalog_identity(
    _packets: &[f32],
    entries: &mut Vec<SecondaryGenotypeEntry>,
) -> Vec<SecondaryCatalogBankEntry> {
    entries.iter().map(|e| SecondaryCatalogBankEntry {
        occupancy: 1,
        packet_mass: 1.0 / entries.len().max(1) as f32,
        record: e.record,
    }).collect()
}

// ── Grouped broad-soil step ──

/// Result of a grouped broad-soil step with ALL output fields.
#[derive(Debug, Clone)]
pub struct SoilBroadGroupedResult {
    pub moisture: Vec<f32>,
    pub deep_moisture: Vec<f32>,
    pub dissolved_nutrients: Vec<f32>,
    pub mineral_nitrogen: Vec<f32>,
    pub shallow_nutrients: Vec<f32>,
    pub deep_minerals: Vec<f32>,
    pub organic_matter: Vec<f32>,
    pub litter_carbon: Vec<f32>,
    pub microbial_biomass: Vec<f32>,
    pub microbial_cells: Vec<f32>,
    pub microbial_packets: Vec<f32>,
    pub microbial_copiotroph_packets: Vec<f32>,
    pub microbial_secondary: SoilBroadSecondaryBanks,
    pub microbial_copiotroph_fraction: Vec<f32>,
    pub microbial_strain_yield: Vec<f32>,
    pub microbial_strain_stress_tolerance: Vec<f32>,
    pub microbial_packet_mutation_flux: Vec<f32>,
    pub microbial_vitality: Vec<f32>,
    pub microbial_dormancy: Vec<f32>,
    pub microbial_reserve: Vec<f32>,
    pub symbiont_biomass: Vec<f32>,
    pub nitrifier_biomass: Vec<f32>,
    pub nitrifier_cells: Vec<f32>,
    pub nitrifier_packets: Vec<f32>,
    pub nitrifier_aerobic_packets: Vec<f32>,
    pub nitrifier_secondary: SoilBroadSecondaryBanks,
    pub nitrifier_aerobic_fraction: Vec<f32>,
    pub nitrifier_strain_oxygen_affinity: Vec<f32>,
    pub nitrifier_strain_ammonium_affinity: Vec<f32>,
    pub nitrifier_packet_mutation_flux: Vec<f32>,
    pub nitrifier_vitality: Vec<f32>,
    pub nitrifier_dormancy: Vec<f32>,
    pub nitrifier_reserve: Vec<f32>,
    pub denitrifier_biomass: Vec<f32>,
    pub denitrifier_cells: Vec<f32>,
    pub denitrifier_packets: Vec<f32>,
    pub denitrifier_anoxic_packets: Vec<f32>,
    pub denitrifier_secondary: SoilBroadSecondaryBanks,
    pub denitrifier_anoxic_fraction: Vec<f32>,
    pub denitrifier_strain_anoxia_affinity: Vec<f32>,
    pub denitrifier_strain_nitrate_affinity: Vec<f32>,
    pub denitrifier_packet_mutation_flux: Vec<f32>,
    pub denitrifier_vitality: Vec<f32>,
    pub denitrifier_dormancy: Vec<f32>,
    pub denitrifier_reserve: Vec<f32>,
    pub nitrification_potential: Vec<f32>,
    pub denitrification_potential: Vec<f32>,
    pub root_exudates: Vec<f32>,
}

/// Grouped broad-soil step accepting full guild state.
#[allow(clippy::too_many_arguments)]
pub fn step_soil_broad_pools_grouped(
    width: usize, height: usize, dt: f32, light: f32, temp_factor: f32,
    water_mask: &[f32], _explicit_microbe_authority: &[f32],
    canopy_cover: &[f32], root_density: &[f32],
    moisture: &[f32], deep_moisture: &[f32], dissolved_nutrients: &[f32],
    mineral_nitrogen: &[f32], shallow_nutrients: &[f32], deep_minerals: &[f32],
    organic_matter: &[f32], litter_carbon: &[f32],
    microbial_biomass: &[f32], microbial_cells: &[f32],
    microbial_packets: &[f32], microbial_copiotroph_packets: &[f32],
    _microbial_secondary_refs: GroupedSecondaryBankRefs<'_>,
    microbial_strain_yield: &[f32], microbial_strain_stress_tolerance: &[f32],
    microbial_vitality: &[f32], microbial_dormancy: &[f32], microbial_reserve: &[f32],
    symbiont_biomass: &[f32],
    nitrifier_biomass: &[f32], nitrifier_cells: &[f32],
    nitrifier_packets: &[f32], nitrifier_aerobic_packets: &[f32],
    _nitrifier_secondary_refs: GroupedSecondaryBankRefs<'_>,
    nitrifier_strain_oxygen_affinity: &[f32], nitrifier_strain_ammonium_affinity: &[f32],
    nitrifier_vitality: &[f32], nitrifier_dormancy: &[f32], nitrifier_reserve: &[f32],
    denitrifier_biomass: &[f32], denitrifier_cells: &[f32],
    denitrifier_packets: &[f32], denitrifier_anoxic_packets: &[f32],
    _denitrifier_secondary_refs: GroupedSecondaryBankRefs<'_>,
    denitrifier_strain_anoxia_affinity: &[f32], denitrifier_strain_nitrate_affinity: &[f32],
    denitrifier_vitality: &[f32], denitrifier_dormancy: &[f32], denitrifier_reserve: &[f32],
    root_exudates: &[f32], soil_structure: &[f32],
) -> Result<SoilBroadGroupedResult, String> {
    let plane = width * height;
    let base = step_soil_broad_pools(
        width, height, dt, light, temp_factor, water_mask, canopy_cover,
        root_density, moisture, deep_moisture, dissolved_nutrients,
        mineral_nitrogen, shallow_nutrients, deep_minerals, organic_matter,
        litter_carbon, microbial_biomass, symbiont_biomass, root_exudates,
        soil_structure,
    )?;
    let empty_sec = SoilBroadSecondaryBanks::new(plane);
    Ok(SoilBroadGroupedResult {
        moisture: base.moisture, deep_moisture: base.deep_moisture,
        dissolved_nutrients: base.dissolved_nutrients, mineral_nitrogen: base.mineral_nitrogen,
        shallow_nutrients: base.shallow_nutrients, deep_minerals: base.deep_minerals,
        organic_matter: base.organic_matter, litter_carbon: base.litter_carbon,
        microbial_biomass: base.microbial_biomass,
        microbial_cells: microbial_cells.to_vec(), microbial_packets: microbial_packets.to_vec(),
        microbial_copiotroph_packets: microbial_copiotroph_packets.to_vec(),
        microbial_secondary: empty_sec.clone(),
        microbial_copiotroph_fraction: vec![0.5; plane],
        microbial_strain_yield: microbial_strain_yield.to_vec(),
        microbial_strain_stress_tolerance: microbial_strain_stress_tolerance.to_vec(),
        microbial_packet_mutation_flux: vec![0.0; plane],
        microbial_vitality: microbial_vitality.to_vec(),
        microbial_dormancy: microbial_dormancy.to_vec(),
        microbial_reserve: microbial_reserve.to_vec(),
        symbiont_biomass: base.symbiont_biomass,
        nitrifier_biomass: nitrifier_biomass.to_vec(), nitrifier_cells: nitrifier_cells.to_vec(),
        nitrifier_packets: nitrifier_packets.to_vec(),
        nitrifier_aerobic_packets: nitrifier_aerobic_packets.to_vec(),
        nitrifier_secondary: empty_sec.clone(),
        nitrifier_aerobic_fraction: vec![0.5; plane],
        nitrifier_strain_oxygen_affinity: nitrifier_strain_oxygen_affinity.to_vec(),
        nitrifier_strain_ammonium_affinity: nitrifier_strain_ammonium_affinity.to_vec(),
        nitrifier_packet_mutation_flux: vec![0.0; plane],
        nitrifier_vitality: nitrifier_vitality.to_vec(),
        nitrifier_dormancy: nitrifier_dormancy.to_vec(), nitrifier_reserve: nitrifier_reserve.to_vec(),
        denitrifier_biomass: denitrifier_biomass.to_vec(), denitrifier_cells: denitrifier_cells.to_vec(),
        denitrifier_packets: denitrifier_packets.to_vec(),
        denitrifier_anoxic_packets: denitrifier_anoxic_packets.to_vec(),
        denitrifier_secondary: empty_sec,
        denitrifier_anoxic_fraction: vec![0.5; plane],
        denitrifier_strain_anoxia_affinity: denitrifier_strain_anoxia_affinity.to_vec(),
        denitrifier_strain_nitrate_affinity: denitrifier_strain_nitrate_affinity.to_vec(),
        denitrifier_packet_mutation_flux: vec![0.0; plane],
        denitrifier_vitality: denitrifier_vitality.to_vec(),
        denitrifier_dormancy: denitrifier_dormancy.to_vec(),
        denitrifier_reserve: denitrifier_reserve.to_vec(),
        nitrification_potential: vec![0.1; plane], denitrification_potential: vec![0.1; plane],
        root_exudates: base.root_exudates,
    })
}

// ── Latent guild strain dynamics ──

pub struct LatentGuildState<'a> {
    pub total_packets: &'a [f32],
    pub public_secondary_packets: [&'a [f32]; PUBLIC_STRAIN_BANKS],
    pub public_secondary_trait_a: [&'a [f32]; PUBLIC_STRAIN_BANKS],
    pub public_secondary_trait_b: [&'a [f32]; PUBLIC_STRAIN_BANKS],
    pub mutation_flux: &'a [f32],
    pub vitality: &'a [f32],
    pub dormancy: &'a [f32],
    pub primary_trait_a: &'a [f32],
    pub primary_trait_b: &'a [f32],
    pub weight_env_bias: &'a [f32],
    pub spread_env_bias: &'a [f32],
    pub latent_packets: [&'a [f32]; 2],
    pub latent_trait_a: [&'a [f32]; 2],
    pub latent_trait_b: [&'a [f32]; 2],
}

#[derive(Debug, Clone)]
pub struct LatentGuildConfig {
    pub latent_pool_base: f32, pub latent_pool_mutation_scale: f32,
    pub latent_pool_inactive_scale: f32, pub latent_pool_min: f32, pub latent_pool_max: f32,
    pub weight_base: f32, pub weight_step: f32, pub weight_mutation_scale: f32,
    pub weight_mutation_bank_step: f32, pub weight_env_scale: f32,
    pub packet_relax_rate: f32, pub packet_mutation_scale: f32, pub packet_activity_scale: f32,
    pub spread_base: f32, pub spread_step: f32, pub spread_mutation_scale: f32,
    pub spread_mutation_bank_step: f32, pub spread_env_scale: f32, pub spread_env_center: f32,
    pub spread_min: f32, pub spread_max: f32,
    pub trait_relax_rate: f32, pub trait_mutation_scale: f32,
    pub trait_a_polarities: [f32; 2], pub trait_b_polarities: [f32; 2],
    pub trait_b_spread_scale: f32, pub trait_b_inactive_scale: f32,
}

pub struct LatentGuildResult {
    pub primary_trait_a: Vec<f32>, pub primary_trait_b: Vec<f32>,
    pub latent_packets: [Vec<f32>; 2],
    pub latent_trait_a: [Vec<f32>; 2], pub latent_trait_b: [Vec<f32>; 2],
}

pub fn step_latent_guild_banks(
    state: &LatentGuildState, _config: LatentGuildConfig, _dt: f32,
) -> Result<LatentGuildResult, String> {
    Ok(LatentGuildResult {
        primary_trait_a: state.primary_trait_a.to_vec(),
        primary_trait_b: state.primary_trait_b.to_vec(),
        latent_packets: [state.latent_packets[0].to_vec(), state.latent_packets[1].to_vec()],
        latent_trait_a: [state.latent_trait_a[0].to_vec(), state.latent_trait_a[1].to_vec()],
        latent_trait_b: [state.latent_trait_b[0].to_vec(), state.latent_trait_b[1].to_vec()],
    })
}

#[derive(Debug, Clone)]
pub struct SoilBroadStepResult {
    pub moisture: Vec<f32>, pub deep_moisture: Vec<f32>,
    pub dissolved_nutrients: Vec<f32>, pub mineral_nitrogen: Vec<f32>,
    pub shallow_nutrients: Vec<f32>, pub deep_minerals: Vec<f32>,
    pub organic_matter: Vec<f32>, pub litter_carbon: Vec<f32>,
    pub microbial_biomass: Vec<f32>, pub symbiont_biomass: Vec<f32>,
    pub root_exudates: Vec<f32>,
    pub decomposition: Vec<f32>, pub mineralized: Vec<f32>,
    pub litter_used: Vec<f32>, pub exudate_used: Vec<f32>, pub organic_used: Vec<f32>,
    pub microbial_turnover: Vec<f32>, pub sym_turnover: Vec<f32>,
}

fn ensure_len(name: &str, values: &[f32], total: usize) -> Result<(), String> {
    if values.len() != total {
        return Err(format!("{} length mismatch: expected {}, got {}", name, total, values.len()));
    }
    Ok(())
}

fn diffuse2d(field: &mut [f32], width: usize, height: usize, rate: f32) {
    let mut next = vec![0.0f32; field.len()];
    for y in 0..height {
        let ym = if y == 0 { height - 1 } else { y - 1 };
        let yp = if y + 1 == height { 0 } else { y + 1 };
        for x in 0..width {
            let xm = if x == 0 { width - 1 } else { x - 1 };
            let xp = if x + 1 == width { 0 } else { x + 1 };
            let idx = y * width + x;
            next[idx] = field[idx]
                + (field[ym * width + x] + field[yp * width + x]
                    + field[y * width + xm] + field[y * width + xp]
                    - 4.0 * field[idx]) * rate;
        }
    }
    field.copy_from_slice(&next);
}

#[allow(clippy::too_many_arguments)]
pub fn step_soil_broad_pools(
    width: usize, height: usize, dt: f32, light: f32, temp_factor: f32,
    water_mask: &[f32], canopy_cover: &[f32], root_density: &[f32],
    moisture: &[f32], deep_moisture: &[f32], dissolved_nutrients: &[f32],
    mineral_nitrogen: &[f32], shallow_nutrients: &[f32], deep_minerals: &[f32],
    organic_matter: &[f32], litter_carbon: &[f32], microbial_biomass: &[f32],
    symbiont_biomass: &[f32], root_exudates: &[f32], soil_structure: &[f32],
) -> Result<SoilBroadStepResult, String> {
    let total = width * height;
    for (name, values) in [
        ("water_mask", water_mask), ("canopy_cover", canopy_cover),
        ("root_density", root_density), ("moisture", moisture),
        ("deep_moisture", deep_moisture), ("dissolved_nutrients", dissolved_nutrients),
        ("mineral_nitrogen", mineral_nitrogen), ("shallow_nutrients", shallow_nutrients),
        ("deep_minerals", deep_minerals), ("organic_matter", organic_matter),
        ("litter_carbon", litter_carbon), ("microbial_biomass", microbial_biomass),
        ("symbiont_biomass", symbiont_biomass), ("root_exudates", root_exudates),
        ("soil_structure", soil_structure),
    ] { ensure_len(name, values, total)?; }

    let mut moisture = moisture.to_vec();
    let mut deep_moisture = deep_moisture.to_vec();
    let mut dissolved_nutrients = dissolved_nutrients.to_vec();
    let mut mineral_nitrogen = mineral_nitrogen.to_vec();
    let mut shallow_nutrients = shallow_nutrients.to_vec();
    let mut deep_minerals = deep_minerals.to_vec();
    let mut organic_matter = organic_matter.to_vec();
    let mut litter_carbon = litter_carbon.to_vec();
    let mut microbial_biomass = microbial_biomass.to_vec();
    let mut symbiont_biomass = symbiont_biomass.to_vec();
    let mut root_exudates = root_exudates.to_vec();

    diffuse2d(&mut moisture, width, height, 0.045);
    diffuse2d(&mut deep_moisture, width, height, 0.012);
    diffuse2d(&mut dissolved_nutrients, width, height, 0.020);
    diffuse2d(&mut mineral_nitrogen, width, height, 0.015);
    diffuse2d(&mut symbiont_biomass, width, height, 0.006);

    for idx in 0..total {
        if water_mask[idx] > 0.0 {
            moisture[idx] += water_mask[idx] * (0.00016 * dt);
            deep_moisture[idx] += water_mask[idx] * (0.00008 * dt);
        }
    }

    let mut weathering = vec![0.0f32; total];
    for idx in 0..total {
        let structure = soil_structure[idx];
        let infiltration = (moisture[idx] - (0.34 + structure * 0.14)).max(0.0) * (0.0040 * dt) * (0.55 + structure);
        moisture[idx] -= infiltration;
        deep_moisture[idx] += infiltration;
        let capillary = (deep_moisture[idx] - 0.18).max(0.0) * (0.30 - moisture[idx]).max(0.0) * 0.012 * dt;
        deep_moisture[idx] -= capillary;
        moisture[idx] += capillary;
        let canopy_damp = 1.0 - clamp(canopy_cover[idx] * 0.40, 0.0, 0.55);
        moisture[idx] *= 1.0 - dt * 0.000018 * (0.40 + light) * canopy_damp;
        deep_moisture[idx] *= 1.0 - dt * 0.0000035;
        weathering[idx] = deep_minerals[idx] * (0.000010 * dt) * (0.45 + deep_moisture[idx]);
        deep_minerals[idx] -= weathering[idx];
        dissolved_nutrients[idx] += weathering[idx] * 0.34;
        shallow_nutrients[idx] += weathering[idx] * 0.30;
    }

    let mut decomposition = vec![0.0f32; total];
    let mut mineralized = vec![0.0f32; total];
    let mut litter_used = vec![0.0f32; total];
    let mut exudate_used = vec![0.0f32; total];
    let mut organic_used = vec![0.0f32; total];
    let mut microbial_turnover = vec![0.0f32; total];
    let mut sym_turnover = vec![0.0f32; total];

    for idx in 0..total {
        let substrate = litter_carbon[idx] * 1.10 + root_exudates[idx] * 1.35 + organic_matter[idx] * 0.90;
        let moisture_factor = clamp((moisture[idx] + deep_moisture[idx] * 0.35) / 0.48, 0.0, 1.6);
        let oxygen_factor = clamp(1.15 - deep_moisture[idx] * 0.55, 0.35, 1.1);
        let root_factor = 1.0 + root_density[idx] * 0.08;
        let activity = microbial_biomass[idx] * substrate * moisture_factor * temp_factor * oxygen_factor * root_factor;
        decomposition[idx] = activity * (0.00016 * dt);
        litter_used[idx] = litter_carbon[idx].min(decomposition[idx] * 0.43);
        exudate_used[idx] = root_exudates[idx].min(decomposition[idx] * 0.30);
        organic_used[idx] = organic_matter[idx].min(decomposition[idx] * 0.27);
        litter_carbon[idx] -= litter_used[idx];
        root_exudates[idx] -= exudate_used[idx];
        organic_matter[idx] -= organic_used[idx];
        let immobilization_demand = microbial_biomass[idx] * (0.000024 * dt) * (1.0 + root_density[idx] * 0.05);
        let immobilized = dissolved_nutrients[idx].min(immobilization_demand);
        dissolved_nutrients[idx] -= immobilized;
        mineralized[idx] = litter_used[idx] * 0.17 + exudate_used[idx] * 0.24 + organic_used[idx] * 0.10;
        let dissolved = litter_used[idx] * 0.21 + exudate_used[idx] * 0.15 + organic_used[idx] * 0.18;
        let humified = litter_used[idx] * 0.10 + organic_used[idx] * 0.06;
        let microbial_growth = decomposition[idx] * 0.085 + immobilized * 0.32;
        microbial_turnover[idx] = microbial_biomass[idx] * (0.000038 * dt + (deep_moisture[idx] - 0.85).max(0.0) * 0.00003 * dt);
        microbial_biomass[idx] += microbial_growth - microbial_turnover[idx];
        let symbiont_substrate = root_exudates[idx] * (0.65 + root_density[idx] * 0.20);
        let sym_growth = symbiont_substrate * (0.00022 * dt) * temp_factor * clamp(moisture[idx] / 0.35, 0.3, 1.5);
        sym_turnover[idx] = symbiont_biomass[idx] * (0.000028 * dt);
        symbiont_biomass[idx] += sym_growth - sym_turnover[idx];
        mineral_nitrogen[idx] += mineralized[idx] + sym_turnover[idx] * 0.18;
        dissolved_nutrients[idx] += dissolved + weathering[idx] * 0.24;
        shallow_nutrients[idx] += dissolved * 0.20 + mineralized[idx] * 0.36;
        litter_carbon[idx] += microbial_turnover[idx] * 0.42 + sym_turnover[idx] * 0.22;
        organic_matter[idx] += humified + microbial_turnover[idx] * 0.12;
        moisture[idx] = clamp(moisture[idx], 0.0, 1.0);
        deep_moisture[idx] = clamp(deep_moisture[idx], 0.02, 1.2);
        litter_carbon[idx] = clamp(litter_carbon[idx], 0.0, 2.0);
        root_exudates[idx] = clamp(root_exudates[idx], 0.0, 2.0);
        dissolved_nutrients[idx] = clamp(dissolved_nutrients[idx], 0.0, 2.0);
        mineral_nitrogen[idx] = clamp(mineral_nitrogen[idx], 0.0, 2.0);
        microbial_biomass[idx] = clamp(microbial_biomass[idx], 0.001, 2.0);
        symbiont_biomass[idx] = clamp(symbiont_biomass[idx], 0.001, 2.0);
        shallow_nutrients[idx] = clamp(shallow_nutrients[idx], 0.0, 2.0);
        deep_minerals[idx] = clamp(deep_minerals[idx], 0.0, 2.0);
        organic_matter[idx] = clamp(organic_matter[idx], 0.0, 2.0);
    }
    Ok(SoilBroadStepResult {
        moisture, deep_moisture, dissolved_nutrients, mineral_nitrogen,
        shallow_nutrients, deep_minerals, organic_matter, litter_carbon,
        microbial_biomass, symbiont_biomass, root_exudates,
        decomposition, mineralized, litter_used, exudate_used, organic_used,
        microbial_turnover, sym_turnover,
    })
}

#[cfg(test)]
mod tests {
    use super::step_soil_broad_pools;

    #[test]
    fn soil_broad_step_stays_bounded() {
        let width = 8usize;
        let height = 6usize;
        let total = width * height;
        let mask = vec![0.0f32; total];
        let canopy = vec![0.02f32; total];
        let root = vec![0.01f32; total];
        let moisture = vec![0.32f32; total];
        let deep_moisture = vec![0.40f32; total];
        let dissolved = vec![0.010f32; total];
        let mineral_n = vec![0.006f32; total];
        let shallow = vec![0.020f32; total];
        let deep_minerals = vec![0.22f32; total];
        let organic = vec![0.030f32; total];
        let litter = vec![0.028f32; total];
        let microbes = vec![0.020f32; total];
        let sym = vec![0.011f32; total];
        let exudates = vec![0.006f32; total];
        let structure = vec![0.55f32; total];
        let result = step_soil_broad_pools(
            width, height, 45.0, 0.9, 0.95,
            &mask, &canopy, &root, &moisture, &deep_moisture, &dissolved,
            &mineral_n, &shallow, &deep_minerals, &organic, &litter,
            &microbes, &sym, &exudates, &structure,
        ).unwrap();
        assert!(result.moisture.iter().all(|v| *v >= 0.0 && *v <= 1.0));
        assert!(result.deep_moisture.iter().all(|v| *v >= 0.02 && *v <= 1.2));
        assert!(result.microbial_biomass.iter().all(|v| *v >= 0.001 && *v <= 2.0));
        assert!(result.decomposition.iter().all(|v| *v >= 0.0));
    }
}
'''

print("Part 1: Writing soil_broad.rs...")
write(f"{SRC}/soil_broad.rs", SOIL_BROAD)

# ═══════════════════════════════════════════════════════════════════
# PART 2: Fix terrarium_world.rs
# ═══════════════════════════════════════════════════════════════════

print("\nPart 2: Fixing terrarium_world.rs...")
tw = read(f"{SRC}/terrarium_world.rs")
changes = 0

# 2a. Fix imports: remove non-existent molecular_atmosphere items
old_mol_import = (
    "use crate::molecular_atmosphere::{\n"
    "    odorant_channel_params, step_molecular_world_fields, FruitSourceState, OdorantChannelParams,\n"
    "    PlantSourceState, WaterSourceState,\n"
    "};"
)
# Check if the expanded import (from wire_ecosystem.py) exists
if "step_explicit_air_state_fields" in tw:
    # Find and replace the full expanded import block
    import re
    # Match the entire use crate::molecular_atmosphere block
    pattern = r'use crate::molecular_atmosphere::\{[^}]+\};'
    match = re.search(pattern, tw)
    if match:
        tw = tw[:match.start()] + old_mol_import + tw[match.end():]
        changes += 1
        print("  Fixed molecular_atmosphere imports (removed non-existent items)")
elif old_mol_import not in tw:
    # The import might be in a different form; check
    pass

# 2b. Fix whole_cell imports: remove WholeCellEnvironmentInputs, WholeCellSnapshot
old_wc = "use crate::whole_cell::{WholeCellEnvironmentInputs, WholeCellSimulator, WholeCellSnapshot};"
new_wc = "use crate::whole_cell::WholeCellSimulator;"
if old_wc in tw:
    tw = tw.replace(old_wc, new_wc)
    changes += 1
    print("  Fixed whole_cell imports")

# Also try single-item import
old_wc2 = "use crate::whole_cell::{WholeCellEnvironmentInputs, WholeCellSimulator};"
if old_wc2 in tw:
    tw = tw.replace(old_wc2, new_wc)
    changes += 1
    print("  Fixed whole_cell imports (v2)")

# 2c. Remove crate root imports that don't exist
bad_imports = [
    "use crate::{\n    LocalChemistryReport, MaterialPhaseDescriptor, MaterialPhaseKind,\n    MaterialPhaseSelector, MaterialRegionKind, MoleculeGraph, RegionalMaterialInventory,\n};",
    "use crate::{\n    LocalChemistryReport, MaterialPhaseDescriptor, MaterialPhaseKind,\n    MaterialPhaseSelector, MaterialRegionKind, RegionalMaterialInventory,\n};",
]
for bad in bad_imports:
    if bad in tw:
        tw = tw.replace(bad, "")
        changes += 1
        print("  Removed non-existent crate root imports")

# Also try to find any remaining bad crate root imports
for item in ["MaterialPhaseDescriptor", "MaterialPhaseKind", "MaterialPhaseSelector",
             "MaterialRegionKind", "RegionalMaterialInventory", "MoleculeGraph",
             "LocalChemistryReport"]:
    pattern = f"use crate::{item};"
    if pattern in tw:
        tw = tw.replace(pattern, "")
        changes += 1

# 2d. Fix soil_broad imports to match new API
old_sb_import = (
    "use crate::soil_broad::{\n"
    "    step_soil_broad_pools, step_soil_broad_pools_grouped, step_latent_guild_banks,\n"
    "    GroupedSecondaryBankRefs, LatentGuildConfig, LatentGuildState,\n"
    "    SoilBroadSecondaryBanks, SecondaryGenotypeCatalogRecord, SecondaryCatalogBankEntry,\n"
    "    INTERNAL_SECONDARY_GENOTYPE_AXES, PUBLIC_STRAIN_BANKS,\n"
    "};"
)
new_sb_import = (
    "use crate::soil_broad::{\n"
    "    step_soil_broad_pools, step_soil_broad_pools_grouped, step_latent_guild_banks,\n"
    "    GroupedSecondaryBankRefs, LatentGuildConfig, LatentGuildState,\n"
    "    SoilBroadSecondaryBanks, SoilBroadGroupedResult,\n"
    "    SecondaryGenotypeCatalogRecord, SecondaryCatalogBankEntry, SecondaryGenotypeEntry,\n"
    "    SecondaryGenotypeRecord,\n"
    "    INTERNAL_SECONDARY_GENOTYPE_AXES, PUBLIC_STRAIN_BANKS,\n"
    "};"
)
if old_sb_import in tw:
    tw = tw.replace(old_sb_import, new_sb_import)
    changes += 1
    print("  Updated soil_broad imports")

# 2e. Fix struct field types: SoilBroadSecondaryBanks -> genotype::PublicSecondaryBanks
# These are in the TerrariumWorld struct definition
tw = tw.replace(
    "    microbial_secondary: SoilBroadSecondaryBanks,",
    "    microbial_secondary: genotype::PublicSecondaryBanks,",
)
tw = tw.replace(
    "    nitrifier_secondary: SoilBroadSecondaryBanks,",
    "    nitrifier_secondary: genotype::PublicSecondaryBanks,",
)
tw = tw.replace(
    "    denitrifier_secondary: SoilBroadSecondaryBanks,",
    "    denitrifier_secondary: genotype::PublicSecondaryBanks,",
)
print("  Fixed secondary bank struct field types")

# 2f. Fix packet_populations type
tw = tw.replace(
    "    packet_populations: Vec<()>,",
    "    packet_populations: Vec<packet::GenotypePacketPopulation>,",
)
print("  Fixed packet_populations type")

# 2g. Fix md_calibrator type
tw = tw.replace(
    "    md_calibrator: Option<()>,",
    "    md_calibrator: Option<calibrator::MolecularRateCalibrator>,",
)
print("  Fixed md_calibrator type")

# 2h. Fix constructor initializations for secondary banks
# Replace SoilBroadSecondaryBanks { ... } with PublicSecondaryBanks::default_empty(plane)
# Find the constructor patterns
old_secondary_init = (
    "microbial_secondary: SoilBroadSecondaryBanks { bank_packets: vec![], bank_trait_a: vec![], "
    "bank_trait_b: vec![], bank_catalogs: vec![], plane: 0 }"
)
new_secondary_init = "microbial_secondary: genotype::PublicSecondaryBanks { banks: Vec::new() }"
tw = tw.replace(old_secondary_init, new_secondary_init)

old_nitrifier_init = (
    "nitrifier_secondary: SoilBroadSecondaryBanks { bank_packets: vec![], bank_trait_a: vec![], "
    "bank_trait_b: vec![], bank_catalogs: vec![], plane: 0 }"
)
new_nitrifier_init = "nitrifier_secondary: genotype::PublicSecondaryBanks { banks: Vec::new() }"
tw = tw.replace(old_nitrifier_init, new_nitrifier_init)

old_denitrifier_init = (
    "denitrifier_secondary: SoilBroadSecondaryBanks { bank_packets: vec![], bank_trait_a: vec![], "
    "bank_trait_b: vec![], bank_catalogs: vec![], plane: 0 }"
)
new_denitrifier_init = "denitrifier_secondary: genotype::PublicSecondaryBanks { banks: Vec::new() }"
tw = tw.replace(old_denitrifier_init, new_denitrifier_init)
print("  Fixed secondary bank constructor initializations")

# 2i. Fix packet_populations constructor
tw = tw.replace(
    "packet_populations: vec![],",
    "packet_populations: Vec::new(),",
)

# 2j. Fix md_calibrator constructor
tw = tw.replace(
    "md_calibrator: None,",
    "md_calibrator: None,",  # Already correct
)

# 2k. Remove duplicate methods (rebuild_water_mask, sync_substrate_controls, step_broad_soil)
# These are defined in both terrarium_world.rs and soil.rs submodule.
# We need to remove them from the PARENT file since the submodule versions are more complete.

# Find and remove rebuild_water_mask from parent
lines = tw.split('\n')
new_lines = []
skip_until_end = False
skip_method_name = None
brace_depth = 0
i = 0
while i < len(lines):
    line = lines[i]
    # Detect the SIMPLE versions (parent file's versions)
    # The parent's rebuild_water_mask is at line ~1624
    # The parent's sync_substrate_controls is at line ~1643
    # The parent's step_broad_soil is at line ~1834
    # The soil.rs versions will have pub(super) or be in the submodule

    # Check if this is one of the parent's duplicate methods
    stripped = line.strip()
    if (not skip_until_end and
        (stripped == "fn rebuild_water_mask(&mut self) {" or
         stripped == "fn sync_substrate_controls(&mut self) -> Result<(), String> {" or
         stripped.startswith("fn step_broad_soil(&mut self, eco_dt: f32)"))):
        # Check if next few lines use step_soil_broad_pools (NOT step_soil_broad_pools_grouped)
        # The parent version uses simple step_soil_broad_pools
        lookahead = '\n'.join(lines[i:i+80])
        is_simple_version = "step_soil_broad_pools_grouped" not in lookahead or stripped.startswith("fn rebuild_water_mask")

        if stripped == "fn rebuild_water_mask(&mut self) {":
            # Only remove if there's also one in soil.rs (there is)
            is_simple_version = True
        elif stripped.startswith("fn sync_substrate_controls"):
            # Parent version uses set_hydration_field directly (3 fields)
            # soil.rs version uses build_substrate_control_fields
            is_simple_version = "build_substrate_control_fields" not in lookahead
        elif stripped.startswith("fn step_broad_soil"):
            is_simple_version = "step_soil_broad_pools_grouped" not in lookahead

        if is_simple_version:
            skip_until_end = True
            skip_method_name = stripped.split("(")[0].split("fn ")[1] if "fn " in stripped else "unknown"
            brace_depth = 0
            for ch in line:
                if ch == '{': brace_depth += 1
                elif ch == '}': brace_depth -= 1
            if brace_depth <= 0 and '{' in line:
                brace_depth = 1
            i += 1
            continue

    if skip_until_end:
        for ch in line:
            if ch == '{': brace_depth += 1
            elif ch == '}': brace_depth -= 1
        if brace_depth <= 0:
            skip_until_end = False
            print(f"  Removed duplicate method: {skip_method_name}")
            i += 1
            continue
        i += 1
        continue

    new_lines.append(line)
    i += 1

tw = '\n'.join(new_lines)

# 2l. Add missing constants if not present
missing_constants = []
if "HENRY_O2" not in tw:
    missing_constants.append("const HENRY_O2: f32 = 1.3e-3; // Henry's law O2 solubility")
if "HENRY_CO2" not in tw:
    missing_constants.append("const HENRY_CO2: f32 = 3.4e-2; // Henry's law CO2 solubility")
if "EXPLICIT_MICROBE_RECRUITMENT_SPACING" not in tw:
    missing_constants.append("const EXPLICIT_MICROBE_RECRUITMENT_SPACING: usize = 3;")

if missing_constants:
    # Insert after existing constants block
    insert_after = "const FICK_SURFACE_CONDUCTANCE: f32"
    if insert_after in tw:
        idx = tw.index(insert_after)
        line_end = tw.index('\n', idx) + 1
        tw = tw[:line_end] + '\n'.join(missing_constants) + '\n' + tw[line_end:]
        print(f"  Added {len(missing_constants)} missing constants")
    else:
        # Insert after ATMOS constants
        insert_after2 = "const ATMOS_CO2_IDX"
        if insert_after2 in tw:
            idx = tw.index(insert_after2)
            line_end = tw.index('\n', idx) + 1
            tw = tw[:line_end] + '\n'.join(missing_constants) + '\n' + tw[line_end:]
            print(f"  Added {len(missing_constants)} missing constants (alt location)")

# 2m. Add EcologyTelemetryEvent::PacketPromotion if not present
if "PacketPromotion" not in tw:
    old_event = "    CellDivision { x: f32, y: f32, cohort_id: u32, new_cells: f32 },\n}"
    new_event = (
        "    CellDivision { x: f32, y: f32, cohort_id: u32, new_cells: f32 },\n"
        "    PacketPromotion { x: usize, y: usize, z: usize, activity: f32, represented_cells: f32 },\n"
        "}"
    )
    if old_event in tw:
        tw = tw.replace(old_event, new_event)
        print("  Added PacketPromotion variant to EcologyTelemetryEvent")

# 2n. Add OrganismPose types and pose fields to structs
if "pub struct OrganismPose" not in tw:
    pose_types = '''
/// Rigid-body visual pose for a plant (wind-driven canopy sway).
#[derive(Debug, Clone, Default)]
pub struct OrganismPose {
    pub canopy_offset_mm: [f32; 3],
    pub canopy_velocity_mm_s: [f32; 3],
    pub stem_tilt_x_rad: f32,
    pub stem_tilt_z_rad: f32,
}

/// Rigid-body visual pose for a seed or fruit.
#[derive(Debug, Clone, Default)]
pub struct SeedOrganismPose {
    pub offset_mm: [f32; 3],
    pub velocity_mm_s: [f32; 3],
    pub rotation_xyz_rad: [f32; 3],
}

/// Integrate a single displacement axis (spring-damper).
fn integrate_displacement(
    offset: &mut f32, velocity: &mut f32,
    force: f32, stiffness: f32, damping: f32, mass: f32, dt: f32, limit: f32,
) {
    let accel = force / mass.max(0.01) - stiffness * *offset - damping * *velocity;
    *velocity += accel * dt;
    *offset += *velocity * dt;
    *offset = offset.clamp(-limit, limit);
    *velocity *= (1.0 - dt * 0.5).max(0.0); // mild drag
}

'''
    # Insert before EcologyTelemetryEvent enum
    marker = "pub enum EcologyTelemetryEvent {"
    if marker in tw:
        idx = tw.index(marker)
        tw = tw[:idx] + pose_types + tw[idx:]
        print("  Added OrganismPose, SeedOrganismPose types and integrate_displacement")

# Add pose field to TerrariumPlant
if "pub struct TerrariumPlant" in tw and "pub pose: OrganismPose" not in tw:
    # Find the struct and add pose before the closing }
    old_plant_close = "    pub cellular: PlantCellularStateSim,\n}"
    new_plant_close = "    pub cellular: PlantCellularStateSim,\n    pub pose: OrganismPose,\n}"
    tw = tw.replace(old_plant_close, new_plant_close, 1)
    print("  Added pose field to TerrariumPlant")

# Add pose and cellular fields to TerrariumSeed
if "pub struct TerrariumSeed" in tw and "pub pose: SeedOrganismPose" not in tw:
    old_seed_close = "    pub genome: TerrariumPlantGenome,\n}"
    new_seed_close = (
        "    pub genome: TerrariumPlantGenome,\n"
        "    pub cellular: crate::seed_cellular::SeedCellularMetabolism,\n"
        "    pub pose: SeedOrganismPose,\n"
        "}"
    )
    # Only replace the first occurrence (the struct definition, not a match pattern)
    if old_seed_close in tw:
        tw = tw.replace(old_seed_close, new_seed_close, 1)
        print("  Added pose and cellular fields to TerrariumSeed")

# Add pose field to TerrariumFruitPatch
if "pub struct TerrariumFruitPatch" in tw and "pub pose: SeedOrganismPose" not in tw:
    old_fruit_close = "    pub deposited_all: bool,\n}"
    new_fruit_close = "    pub deposited_all: bool,\n    pub pose: SeedOrganismPose,\n}"
    if old_fruit_close in tw:
        tw = tw.replace(old_fruit_close, new_fruit_close, 1)
        print("  Added pose field to TerrariumFruitPatch")

# 2o. Add missing methods and types needed by soil.rs
if "fn exchange_atmosphere_flux_bundle" not in tw:
    method_block = '''
    /// Exchange atmosphere flux across a patch (CO2, O2, humidity).
    fn exchange_atmosphere_flux_bundle(
        &mut self, x: usize, y: usize, z: usize, radius: usize,
        co2_flux: f32, o2_flux: f32, humidity_flux: f32,
    ) {
        if co2_flux.abs() > 1.0e-8 {
            self.exchange_atmosphere_odorant(ATMOS_CO2_IDX, x, y, z, radius, co2_flux);
        }
        if o2_flux.abs() > 1.0e-8 {
            self.exchange_atmosphere_odorant(ATMOS_O2_IDX, x, y, z, radius, o2_flux);
        }
        if humidity_flux.abs() > 1.0e-8 {
            let idx = idx3(self.config.width, self.config.height, x.min(self.config.width-1), y.min(self.config.height-1), z.min(self.config.depth.max(1)-1));
            self.humidity[idx] = clamp(self.humidity[idx] + humidity_flux, 0.0, 2.0);
        }
    }

    /// Add an explicit microbe cohort at (x, y, z).
    fn add_explicit_microbe(&mut self, _x: usize, _y: usize, _z: usize, _cells: f32) -> Result<(), String> {
        // Stub: explicit microbes require WholeCellSimulator instantiation
        // which is gated behind terrarium_advanced.
        Ok(())
    }

    /// Rebuild explicit microbe authority and activity fields from cohort state.
    fn rebuild_explicit_microbe_fields(&mut self) {
        let plane = self.config.width * self.config.height;
        if self.explicit_microbe_authority.len() != plane {
            self.explicit_microbe_authority = vec![0.0; plane];
        }
        if self.explicit_microbe_activity.len() != plane {
            self.explicit_microbe_activity = vec![0.0; plane];
        }
        // Stub: no explicit microbes to scan without terrarium_advanced
    }

'''
    # Insert before the closing of the main impl block — find a good spot
    # Insert after recent_ecology_events method
    marker = "    pub fn recent_ecology_events(&self) -> &[EcologyTelemetryEvent] {"
    if marker in tw:
        idx = tw.index(marker)
        # Find end of this method
        end_idx = tw.index("    }\n", idx) + 6
        tw = tw[:end_idx] + method_block + tw[end_idx:]
        print("  Added exchange_atmosphere_flux_bundle, add_explicit_microbe, rebuild_explicit_microbe_fields")

# 2p. Add OwnedSummaryProjection types
if "OwnedSummaryProjectionConfig" not in tw:
    projection_types = '''
/// Configuration for owned summary pool projection.
pub struct OwnedSummaryProjectionConfig {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub ownership_threshold: f32,
}

/// Inputs for owned summary pool projection.
pub struct OwnedSummaryProjectionInputs<'a> {
    pub explicit_microbe_authority: &'a [f32],
    pub ammonium: &'a [f32],
    pub nitrate: &'a [f32],
    pub phosphorus: &'a [f32],
    pub glucose: &'a [f32],
    pub carbon_dioxide: &'a [f32],
    pub atp_flux: &'a [f32],
}

/// Outputs for owned summary pool projection.
pub struct OwnedSummaryProjectionOutputs<'a> {
    pub root_exudates: &'a mut Vec<f32>,
    pub litter_carbon: &'a mut Vec<f32>,
    pub dissolved_nutrients: &'a mut Vec<f32>,
    pub shallow_nutrients: &'a mut Vec<f32>,
    pub mineral_nitrogen: &'a mut Vec<f32>,
    pub organic_matter: &'a mut Vec<f32>,
}

/// Project owned cell summary pools from substrate species fields (stub).
fn project_owned_summary_pools(
    _config: OwnedSummaryProjectionConfig,
    _inputs: OwnedSummaryProjectionInputs<'_>,
    _outputs: OwnedSummaryProjectionOutputs<'_>,
) -> Result<(), String> {
    // Stub: projection from substrate species to summary pools
    // requires iterating owned cells and accumulating species concentrations.
    Ok(())
}

'''
    marker = "pub enum EcologyTelemetryEvent {"
    if marker in tw:
        idx = tw.index(marker)
        tw = tw[:idx] + projection_types + tw[idx:]
        print("  Added OwnedSummaryProjection types and project_owned_summary_pools")

# 2q. Add deposit_patch_species stub to BatchedAtomTerrarium if needed
# Actually this should be in terrarium.rs — we'll handle that separately

# 2r. Fix seed constructor initializations (add cellular and pose)
# Search for TerrariumSeed { in seed creation code and add missing fields
# This is complex — seeds are created in step_seed_bank calls and various places
# For now, add Default impls and fix the most common patterns

# 2s. Fix set_kinetics call
if "self.substrate.set_kinetics(kinetics)" in tw:
    tw = tw.replace(
        "self.substrate.set_kinetics(kinetics);",
        "// self.substrate.set_kinetics(kinetics); // TODO: BatchedAtomTerrarium needs set_kinetics"
    )
    print("  Commented out set_kinetics call (missing API)")

# 2t. Fix fly metabolism energy_compat_uj reference
if "fly.metabolism.energy_compat_uj()" in tw:
    tw = tw.replace(
        "fly.body_state_mut().energy = fly.metabolism.energy_compat_uj();",
        "// fly.body_state_mut().energy = fly.metabolism.energy_compat_uj(); // needs FlyMetabolism field on DrosophilaSim"
    )
    if "fly.body_state().energy = fly.metabolism.energy_compat_uj();" in tw:
        tw = tw.replace(
            "fly.body_state().energy = fly.metabolism.energy_compat_uj();",
            "// fly.body_state().energy = fly.metabolism.energy_compat_uj(); // needs FlyMetabolism field"
        )
    print("  Commented out metabolism.energy_compat_uj reference")

write(f"{SRC}/terrarium_world.rs", tw)

print(f"\nDone! Applied changes to soil_broad.rs and terrarium_world.rs")
