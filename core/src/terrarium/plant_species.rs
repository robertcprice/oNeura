use rand::Rng;

use crate::botany::{
    sample_species_profile, species_profile_by_taxonomy, BotanicalGrowthForm,
    BotanicalSpeciesProfile,
};
use crate::constants::clamp;

use super::{TerrariumPlant, TerrariumPlantGenome};

pub const DEMO_PLANT_TAXONOMIES: [u32; 8] = [3750, 23211, 3760, 42229, 2711, 2708, 15368, 3702];

pub fn sample_named_plant_genome<R: Rng + ?Sized>(rng: &mut R) -> TerrariumPlantGenome {
    genome_from_profile(sample_species_profile(rng), rng)
}

pub fn genome_for_taxonomy<R: Rng + ?Sized>(taxonomy_id: u32, rng: &mut R) -> TerrariumPlantGenome {
    species_profile_by_taxonomy(taxonomy_id)
        .map(|profile| genome_from_profile(profile, rng))
        .unwrap_or_else(|| TerrariumPlantGenome::sample(rng))
}

pub fn plant_common_name(taxonomy_id: u32) -> &'static str {
    species_profile_by_taxonomy(taxonomy_id)
        .map(|profile| profile.common_name)
        .unwrap_or("Unknown plant")
}

pub fn plant_scientific_name(taxonomy_id: u32) -> &'static str {
    species_profile_by_taxonomy(taxonomy_id)
        .map(|profile| profile.scientific_name)
        .unwrap_or("Unknown species")
}

pub fn plant_growth_form(taxonomy_id: u32) -> BotanicalGrowthForm {
    species_profile_by_taxonomy(taxonomy_id)
        .map(|profile| profile.growth_form)
        .unwrap_or(BotanicalGrowthForm::RosetteHerb)
}

pub fn nearest_plant_taxonomy(plants: &[TerrariumPlant], x: f32, y: f32) -> u32 {
    plants
        .iter()
        .min_by(|a, b| {
            let da = ((a.x as f32 - x).powi(2) + (a.y as f32 - y).powi(2))
                .partial_cmp(&((b.x as f32 - x).powi(2) + (b.y as f32 - y).powi(2)))
                .unwrap_or(std::cmp::Ordering::Equal);
            da
        })
        .map(|plant| plant.genome.taxonomy_id)
        .unwrap_or(0)
}

fn jitter<R: Rng + ?Sized>(value: f32, rng: &mut R, span: f32, lo: f32, hi: f32) -> f32 {
    clamp(value + rng.gen_range(-span..=span), lo, hi)
}

fn genome_from_profile<R: Rng + ?Sized>(
    profile: BotanicalSpeciesProfile,
    rng: &mut R,
) -> TerrariumPlantGenome {
    TerrariumPlantGenome {
        species_id: profile.taxonomy_id,
        taxonomy_id: profile.taxonomy_id,
        max_height_mm: jitter(profile.max_height_mm, rng, 1.2, 4.0, 22.0),
        canopy_radius_mm: jitter(profile.canopy_radius_mm, rng, 0.55, 1.5, 10.0),
        root_radius_mm: jitter(profile.root_radius_mm, rng, 0.45, 1.2, 8.5),
        leaf_efficiency: jitter(profile.leaf_efficiency, rng, 0.06, 0.55, 1.65),
        root_uptake_efficiency: jitter(profile.root_uptake_efficiency, rng, 0.05, 0.45, 1.65),
        water_use_efficiency: jitter(profile.water_use_efficiency, rng, 0.05, 0.45, 1.50),
        volatile_scale: jitter(profile.volatile_scale, rng, 0.06, 0.45, 1.85),
        fruiting_threshold: jitter(profile.fruiting_threshold, rng, 0.05, 0.30, 1.50),
        litter_turnover: jitter(profile.litter_turnover, rng, 0.05, 0.45, 1.80),
        shade_tolerance: jitter(profile.shade_tolerance, rng, 0.05, 0.40, 1.70),
        root_depth_bias: jitter(profile.root_depth_bias, rng, 0.03, 0.05, 1.10),
        symbiosis_affinity: jitter(profile.symbiosis_affinity, rng, 0.05, 0.35, 1.80),
        seed_mass: jitter(profile.seed_mass, rng, 0.008, 0.03, 0.20),
    }
}
