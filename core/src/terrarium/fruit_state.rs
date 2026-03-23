use super::TerrariumPlantGenome;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TerrariumFruitComposition {
    pub water_fraction: f32,
    pub sugar_fraction: f32,
    pub organic_acid_fraction: f32,
    pub amino_fraction: f32,
    pub bitter_fraction: f32,
    pub aroma_intensity: f32,
}

impl Default for TerrariumFruitComposition {
    fn default() -> Self {
        Self {
            water_fraction: 0.72,
            sugar_fraction: 0.18,
            organic_acid_fraction: 0.08,
            amino_fraction: 0.04,
            bitter_fraction: 0.02,
            aroma_intensity: 0.12,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TerrariumFruitDevelopmentState {
    pub parent_x: usize,
    pub parent_y: usize,
    pub mature_radius: f32,
    pub sugar_capacity: f32,
    pub growth_progress: f32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TerrariumFruitOrganState {
    pub peel_integrity: f32,
    pub flesh_integrity: f32,
    pub vascular_supply: f32,
    pub attachment_strength: f32,
    pub seed_exposure: f32,
    pub rot_progress: f32,
}

impl Default for TerrariumFruitOrganState {
    fn default() -> Self {
        Self {
            peel_integrity: 0.92,
            flesh_integrity: 0.94,
            vascular_supply: 0.0,
            attachment_strength: 0.0,
            seed_exposure: 0.0,
            rot_progress: 0.0,
        }
    }
}

pub fn attached_fruit_organ_state() -> TerrariumFruitOrganState {
    TerrariumFruitOrganState {
        vascular_supply: 0.88,
        attachment_strength: 0.96,
        ..TerrariumFruitOrganState::default()
    }
}

pub fn detached_fruit_organ_state(ripeness: f32) -> TerrariumFruitOrganState {
    let ripeness_t = ripeness.clamp(0.0, 1.0);
    TerrariumFruitOrganState {
        peel_integrity: crate::constants::clamp(0.82 - ripeness_t * 0.08, 0.48, 0.88),
        flesh_integrity: crate::constants::clamp(0.86 - ripeness_t * 0.06, 0.44, 0.92),
        seed_exposure: crate::constants::clamp((ripeness_t - 0.82) * 0.25, 0.0, 0.20),
        ..TerrariumFruitOrganState::default()
    }
}

fn attached_fruit_growth_drive(
    daylight: f32,
    parent_health: f32,
    parent_storage_carbon: f32,
    moisture: f32,
    deep_moisture: f32,
    nutrients: f32,
) -> f32 {
    let storage_t = crate::constants::clamp(parent_storage_carbon / 1.6, 0.0, 1.0);
    crate::constants::clamp(
        daylight * 0.24
            + parent_health.max(0.0) * 0.24
            + moisture.max(0.0) * 0.18
            + deep_moisture.max(0.0) * 0.14
            + nutrients.max(0.0) * 0.10
            + storage_t * 0.10,
        0.0,
        1.0,
    )
}

pub fn advance_attached_fruit(
    source: &mut crate::molecular_atmosphere::FruitSourceState,
    radius: &mut f32,
    development: &mut TerrariumFruitDevelopmentState,
    organ: &mut TerrariumFruitOrganState,
    daylight: f32,
    parent_health: f32,
    parent_storage_carbon: f32,
    moisture: f32,
    deep_moisture: f32,
    nutrients: f32,
    dt: f32,
) -> f32 {
    let growth_drive = attached_fruit_growth_drive(
        daylight,
        parent_health,
        parent_storage_carbon,
        moisture,
        deep_moisture,
        nutrients,
    );
    let support_t = crate::constants::clamp(
        parent_health.max(0.0) * 0.45 + growth_drive * 0.55,
        0.0,
        1.2,
    );
    let hydration_t = crate::constants::clamp(
        moisture.max(0.0) * 0.48 + deep_moisture.max(0.0) * 0.24 + nutrients.max(0.0) * 0.18,
        0.0,
        1.2,
    );
    let ripeness_t = source.ripeness.clamp(0.0, 1.0);
    let vascular_target =
        crate::constants::clamp(0.24 + support_t * 0.82 - ripeness_t * 0.12, 0.0, 1.0);
    let vascular_approach = crate::constants::clamp(dt / 2400.0, 0.02, 0.18);
    organ.vascular_supply = crate::constants::clamp(
        organ.vascular_supply + (vascular_target - organ.vascular_supply) * vascular_approach,
        0.0,
        1.0,
    );
    let progress_gain = dt.max(0.0) * (0.00002 + growth_drive * 0.00016);
    development.growth_progress =
        crate::constants::clamp(development.growth_progress + progress_gain, 0.0, 1.0);
    let growth_t = development.growth_progress;
    let mature_radius = development.mature_radius.max(0.18);
    *radius = crate::constants::clamp(
        0.18 + (mature_radius - 0.18) * growth_t.powf(0.72),
        0.18,
        mature_radius,
    );
    source.ripeness = crate::constants::clamp(0.06 + growth_t * 0.90, 0.06, 0.98);
    let sugar_target = development.sugar_capacity.max(0.02)
        * crate::constants::clamp(
            0.04 + growth_t.powf(1.18) * (0.62 + growth_drive * 0.38),
            0.04,
            1.0,
        );
    let approach = crate::constants::clamp(dt / 1800.0, 0.02, 0.25);
    source.sugar_content = crate::constants::clamp(
        source.sugar_content + (sugar_target - source.sugar_content).max(0.0) * approach,
        0.0,
        development.sugar_capacity.max(0.02),
    );
    let attached_senescence = crate::constants::clamp(
        (ripeness_t - 0.78).max(0.0) * 0.42
            + (1.0 - support_t.min(1.0)) * 0.34
            + (1.0 - hydration_t.min(1.0)) * 0.12,
        0.0,
        1.4,
    );
    organ.rot_progress = crate::constants::clamp(
        organ.rot_progress + dt * (0.000002 + attached_senescence * 0.000012)
            - dt * (support_t.min(1.0) * hydration_t.min(1.0) * 0.000004),
        0.0,
        1.0,
    );
    organ.peel_integrity = crate::constants::clamp(
        organ.peel_integrity + dt * (0.00002 + support_t * hydration_t * 0.00003)
            - dt * (0.00001 + ripeness_t * 0.000015 + attached_senescence * 0.000010),
        0.22,
        1.0,
    );
    organ.flesh_integrity = crate::constants::clamp(
        organ.flesh_integrity + dt * (0.00003 + support_t * hydration_t * 0.00004)
            - dt * (0.000012 + ripeness_t * 0.000014 + attached_senescence * 0.000012),
        0.22,
        1.0,
    );
    let abscission_drive = crate::constants::clamp(
        ripeness_t * 0.30
            + (1.0 - support_t.min(1.0)) * 0.32
            + (1.0 - organ.vascular_supply) * 0.24
            + (1.0 - organ.peel_integrity) * 0.12
            + (1.0 - organ.flesh_integrity) * 0.14
            + organ.rot_progress * 0.16,
        0.0,
        1.4,
    );
    organ.attachment_strength = crate::constants::clamp(
        organ.attachment_strength + dt * (support_t * 0.000015 - abscission_drive * 0.00005),
        0.0,
        1.0,
    );
    organ.seed_exposure = crate::constants::clamp(
        organ.seed_exposure
            + dt * (1.0 - organ.flesh_integrity) * 0.000015
            + dt * organ.rot_progress * 0.000006,
        0.0,
        0.45,
    );
    if organ.attachment_strength <= 0.16 {
        source.attached = false;
        source.z = 0;
        organ.vascular_supply = 0.0;
        organ.attachment_strength = 0.0;
    }
    growth_drive
}

#[allow(clippy::too_many_arguments)]
pub fn advance_detached_fruit(
    source: &mut crate::molecular_atmosphere::FruitSourceState,
    radius: &mut f32,
    organ: &mut TerrariumFruitOrganState,
    moisture: f32,
    deep_moisture: f32,
    humidity: f32,
    microbial_biomass: f32,
    decay_detritus: f32,
    lost_detritus: f32,
    final_detritus: f32,
    dt: f32,
) {
    if source.attached {
        return;
    }

    organ.vascular_supply = 0.0;
    organ.attachment_strength = 0.0;
    let wetness = crate::constants::clamp(
        moisture.max(0.0) * 0.36 + deep_moisture.max(0.0) * 0.22 + humidity.max(0.0) * 0.34,
        0.0,
        1.2,
    );
    let detritus_drive = decay_detritus + lost_detritus + final_detritus;
    let rot_drive = crate::constants::clamp(
        source.ripeness.clamp(0.0, 1.0) * 0.28
            + wetness * 0.22
            + microbial_biomass.max(0.0) * 0.18
            + detritus_drive * 14.0,
        0.0,
        2.0,
    );
    organ.rot_progress = crate::constants::clamp(
        organ.rot_progress + dt * (0.00002 + rot_drive * 0.00006),
        0.0,
        1.5,
    );
    organ.peel_integrity = crate::constants::clamp(
        organ.peel_integrity
            - dt * (0.000015 + rot_drive * 0.000045)
            - lost_detritus * 0.55
            - final_detritus * 0.70,
        0.0,
        1.0,
    );
    organ.flesh_integrity = crate::constants::clamp(
        organ.flesh_integrity
            - dt * (0.00002 + rot_drive * 0.00006)
            - decay_detritus * 0.45
            - lost_detritus * 0.30
            - final_detritus * 0.50,
        0.0,
        1.0,
    );
    organ.seed_exposure = crate::constants::clamp(
        organ.seed_exposure
            + dt * (0.00001 + rot_drive * 0.00004)
            + (1.0 - organ.peel_integrity) * 0.04
            + (1.0 - organ.flesh_integrity) * 0.05
            + lost_detritus * 0.25
            + final_detritus * 0.45,
        0.0,
        1.0,
    );
    *radius = (*radius - dt * rot_drive * 0.00003).max(0.18);
}

pub fn fruit_ready_to_drop_seed(
    source: &crate::molecular_atmosphere::FruitSourceState,
    organ: &TerrariumFruitOrganState,
    detritus_total: f32,
) -> bool {
    if source.attached {
        return false;
    }
    let release_drive = organ.seed_exposure
        + organ.rot_progress * 0.45
        + (1.0 - organ.flesh_integrity) * 0.30
        + detritus_total * 8.0;
    release_drive >= 0.70
}

pub fn fruit_composition_from_parent(
    genome: &TerrariumPlantGenome,
    taxonomy_id: u32,
    sugar_content: f32,
    ripeness: f32,
) -> TerrariumFruitComposition {
    let ripeness_t = ripeness.clamp(0.0, 1.0);
    let sugar_t = sugar_content.clamp(0.0, 1.0);
    let volatile_t = crate::constants::clamp((genome.volatile_scale - 0.45) / 1.35, 0.0, 1.0);
    let seed_t = crate::constants::clamp(genome.seed_mass / 0.20, 0.0, 1.0);
    let uptake_t = crate::constants::clamp((genome.root_uptake_efficiency - 0.45) / 1.15, 0.0, 1.0);
    let water_use_t =
        crate::constants::clamp((genome.water_use_efficiency - 0.45) / 1.05, 0.0, 1.0);
    let symbiosis_t = crate::constants::clamp((genome.symbiosis_affinity - 0.35) / 1.45, 0.0, 1.0);

    let (sugar_bias, acid_bias, bitter_bias, amino_bias, water_bias) =
        match crate::botany::species_profile_by_taxonomy(taxonomy_id).map(|p| p.growth_form) {
            Some(crate::botany::BotanicalGrowthForm::CitrusTree) => (0.92, 0.30, 0.22, 0.05, 0.78),
            Some(crate::botany::BotanicalGrowthForm::StoneFruitTree) => {
                (1.02, 0.18, 0.08, 0.12, 0.72)
            }
            Some(crate::botany::BotanicalGrowthForm::OrchardTree) => (0.98, 0.16, 0.06, 0.14, 0.76),
            Some(crate::botany::BotanicalGrowthForm::RosetteHerb) => (0.66, 0.12, 0.10, 0.16, 0.70),
            Some(crate::botany::BotanicalGrowthForm::GrassClump) => (0.54, 0.08, 0.06, 0.14, 0.66),
            Some(crate::botany::BotanicalGrowthForm::FloatingAquatic) => {
                (0.42, 0.06, 0.03, 0.10, 0.88)
            }
            Some(crate::botany::BotanicalGrowthForm::SubmergedAquatic) => {
                (0.40, 0.06, 0.03, 0.11, 0.90)
            }
            None => (0.82, 0.14, 0.08, 0.12, 0.74),
        };

    TerrariumFruitComposition {
        water_fraction: crate::constants::clamp(
            water_bias + (1.0 - ripeness_t) * 0.06 + water_use_t * 0.04,
            0.45,
            0.94,
        ),
        sugar_fraction: crate::constants::clamp(
            sugar_t * (0.28 + ripeness_t * 0.62) * sugar_bias,
            0.02,
            1.0,
        ),
        organic_acid_fraction: crate::constants::clamp(
            (1.0 - ripeness_t) * 0.26 + volatile_t * 0.08 + acid_bias,
            0.01,
            1.0,
        ),
        amino_fraction: crate::constants::clamp(
            seed_t * 0.24 + uptake_t * 0.14 + symbiosis_t * 0.10 + amino_bias,
            0.01,
            1.0,
        ),
        bitter_fraction: crate::constants::clamp(
            (1.0 - ripeness_t) * 0.22 + volatile_t * 0.12 + bitter_bias,
            0.0,
            1.0,
        ),
        aroma_intensity: crate::constants::clamp(
            volatile_t * (0.25 + ripeness_t * 0.55) + sugar_t * 0.10,
            0.0,
            1.0,
        ),
    }
}

/// Compute fruit composition from the plant's metabolome — the emergent version.
///
/// Instead of hardcoding acid/sugar biases per growth form, this reads actual
/// molecular concentrations from the metabolome. A citrus fruit has high citrate
/// because citrate synthase is highly expressed (from the genome), not because
/// we write `acid_bias = 0.30`.
///
/// The composition differs between species because their enzyme expression patterns
/// differ, and even varies within a species based on growing conditions (light,
/// temperature, water) because those conditions modulate gene expression which
/// modulates metabolite pools.
pub fn fruit_composition_from_metabolome(
    metabolome: &crate::botany::PlantMetabolome,
    ripeness: f32,
) -> TerrariumFruitComposition {
    let ripeness_t = ripeness.clamp(0.0, 1.0);

    // Total sugars: glucose + fructose + sucrose + sorbitol
    let total_sugar = metabolome.glucose_count
        + metabolome.fructose_count
        + metabolome.sucrose_count
        + metabolome.sorbitol_count;

    // Total organic acids: malate + citrate
    let total_acid = metabolome.malate_count + metabolome.citrate_count;

    // Aroma compounds: limonene + benzaldehyde + VOC
    let total_aroma = metabolome.limonene_count
        + metabolome.benzaldehyde_count
        + metabolome.voc_emission_rate;

    // Normalize by total metabolome mass to get fractions
    let total_pool = (total_sugar + total_acid + metabolome.water_count + metabolome.amino_acid_pool + total_aroma).max(1.0);

    // Water fraction: from actual water content, increases with hydration
    let water_frac = (metabolome.water_count / total_pool)
        .clamp(0.45, 0.94) as f32;

    // Sugar fraction: ripeness enhances sugar availability (starch -> sugar conversion)
    let sugar_raw = total_sugar / total_pool;
    let sugar_frac = (sugar_raw * (0.4 + ripeness_t as f64 * 0.6))
        .clamp(0.02, 0.35) as f32;

    // Organic acid fraction: decreases with ripeness (acid degradation during ripening)
    let acid_raw = total_acid / total_pool;
    let acid_frac = (acid_raw * (1.2 - ripeness_t as f64 * 0.5))
        .clamp(0.01, 0.30) as f32;

    // Amino fraction: from amino acid pool
    let amino_frac = (metabolome.amino_acid_pool / total_pool)
        .clamp(0.01, 0.12) as f32;

    // Bitter fraction: from anthocyanin precursors + unripe phenolics
    let bitter_raw = metabolome.anthocyanin_count * 0.3 + (1.0 - ripeness_t as f64) * 0.1;
    let bitter_frac = (bitter_raw / total_pool.max(1.0))
        .clamp(0.0, 0.15) as f32;

    // Aroma intensity: from volatile metabolites
    let aroma_intensity = ((total_aroma / total_pool.max(1.0)) * (0.3 + ripeness_t as f64 * 0.7))
        .clamp(0.0, 1.0) as f32;

    TerrariumFruitComposition {
        water_fraction: water_frac,
        sugar_fraction: sugar_frac,
        organic_acid_fraction: acid_frac,
        amino_fraction: amino_frac,
        bitter_fraction: bitter_frac,
        aroma_intensity,
    }
}

pub fn fruit_surface_taste_profile(
    composition: &TerrariumFruitComposition,
    surface_contact: f32,
) -> (f32, f32, f32) {
    let surface_contact = surface_contact.clamp(0.0, 1.0);
    let sugar_taste = (surface_contact
        * composition.sugar_fraction
        * (0.35 + composition.aroma_intensity * 0.18))
        .clamp(0.0, 1.0);
    let bitter_taste = (surface_contact
        * (composition.bitter_fraction * 0.82 + composition.organic_acid_fraction * 0.24))
        .clamp(0.0, 1.0);
    let amino_taste = (surface_contact
        * (composition.amino_fraction * 0.78 + composition.water_fraction * 0.04))
        .clamp(0.0, 1.0);
    (sugar_taste, bitter_taste, amino_taste)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn fruit_composition_depends_on_parent_identity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let apple = crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut rng);
        let lemon = crate::terrarium::plant_species::genome_for_taxonomy(2708, &mut rng);

        let apple_comp = fruit_composition_from_parent(&apple, apple.taxonomy_id, 0.9, 0.82);
        let lemon_comp = fruit_composition_from_parent(&lemon, lemon.taxonomy_id, 0.9, 0.82);

        assert!(
            (apple_comp.organic_acid_fraction - lemon_comp.organic_acid_fraction).abs() > 1.0e-4
                || (apple_comp.bitter_fraction - lemon_comp.bitter_fraction).abs() > 1.0e-4
                || (apple_comp.sugar_fraction - lemon_comp.sugar_fraction).abs() > 1.0e-4
        );
        assert!(lemon_comp.organic_acid_fraction > apple_comp.organic_acid_fraction);
        assert!(lemon_comp.bitter_fraction > apple_comp.bitter_fraction);
    }

    #[test]
    fn attached_fruit_growth_advances_radius_and_sugar() {
        let mut source = crate::molecular_atmosphere::FruitSourceState {
            x: 8,
            y: 8,
            z: 1,
            attached: true,
            ripeness: 0.08,
            sugar_content: 0.04,
            odorant_emission_rate: 0.02,
            decay_rate: 0.001,
            alive: true,
            odorant_profile: Vec::new(),
        };
        let mut radius = 0.24;
        let mut development = TerrariumFruitDevelopmentState {
            parent_x: 8,
            parent_y: 8,
            mature_radius: 1.3,
            sugar_capacity: 0.95,
            growth_progress: 0.08,
        };
        let mut organ = attached_fruit_organ_state();

        let initial_radius = radius;
        let initial_sugar = source.sugar_content;
        let initial_ripeness = source.ripeness;
        for _ in 0..12 {
            let _ = advance_attached_fruit(
                &mut source,
                &mut radius,
                &mut development,
                &mut organ,
                0.95,
                0.92,
                1.4,
                0.94,
                0.88,
                0.76,
                120.0,
            );
        }

        assert!(radius > initial_radius);
        assert!(source.sugar_content > initial_sugar);
        assert!(source.ripeness > initial_ripeness);
    }

    #[test]
    fn detached_fruit_rot_exposes_seed() {
        let mut source = crate::molecular_atmosphere::FruitSourceState {
            x: 8,
            y: 8,
            z: 0,
            attached: false,
            ripeness: 0.96,
            sugar_content: 0.10,
            odorant_emission_rate: 0.02,
            decay_rate: 0.001,
            alive: true,
            odorant_profile: Vec::new(),
        };
        let mut radius = 1.1;
        let mut organ = detached_fruit_organ_state(source.ripeness);

        for _ in 0..18 {
            advance_detached_fruit(
                &mut source,
                &mut radius,
                &mut organ,
                0.92,
                0.84,
                0.78,
                0.16,
                0.010,
                0.012,
                0.0,
                240.0,
            );
        }

        assert!(organ.seed_exposure > 0.20);
        assert!(organ.rot_progress > 0.05);
        assert!(fruit_ready_to_drop_seed(&source, &organ, 0.01));
    }

    #[test]
    fn low_support_attached_fruit_eventually_drops() {
        let mut source = crate::molecular_atmosphere::FruitSourceState {
            x: 8,
            y: 8,
            z: 1,
            attached: true,
            ripeness: 0.72,
            sugar_content: 0.30,
            odorant_emission_rate: 0.02,
            decay_rate: 0.001,
            alive: true,
            odorant_profile: Vec::new(),
        };
        let mut radius = 0.74;
        let mut development = TerrariumFruitDevelopmentState {
            parent_x: 8,
            parent_y: 8,
            mature_radius: 1.3,
            sugar_capacity: 0.95,
            growth_progress: 0.78,
        };
        let mut organ = attached_fruit_organ_state();

        for _ in 0..96 {
            let _ = advance_attached_fruit(
                &mut source,
                &mut radius,
                &mut development,
                &mut organ,
                0.05,
                0.12,
                0.02,
                0.04,
                0.03,
                0.02,
                300.0,
            );
            if !source.attached {
                break;
            }
        }

        assert!(!source.attached);
        assert_eq!(source.z, 0);
        assert!(organ.attachment_strength <= 0.16);
    }

    #[test]
    fn well_supported_attached_fruit_can_remain_on_plant() {
        let mut source = crate::molecular_atmosphere::FruitSourceState {
            x: 8,
            y: 8,
            z: 1,
            attached: true,
            ripeness: 0.70,
            sugar_content: 0.28,
            odorant_emission_rate: 0.02,
            decay_rate: 0.001,
            alive: true,
            odorant_profile: Vec::new(),
        };
        let mut radius = 0.82;
        let mut development = TerrariumFruitDevelopmentState {
            parent_x: 8,
            parent_y: 8,
            mature_radius: 1.3,
            sugar_capacity: 0.95,
            growth_progress: 0.82,
        };
        let mut organ = attached_fruit_organ_state();

        for _ in 0..96 {
            let _ = advance_attached_fruit(
                &mut source,
                &mut radius,
                &mut development,
                &mut organ,
                0.96,
                0.92,
                1.32,
                0.94,
                0.88,
                0.80,
                300.0,
            );
        }

        assert!(source.attached);
        assert_eq!(source.z, 1);
        assert!(organ.attachment_strength > 0.16);
        assert!(organ.rot_progress < 0.20);
    }
}
