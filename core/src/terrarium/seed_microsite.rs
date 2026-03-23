use super::*;
use rand::Rng;
use rand_distr::StandardNormal;

const SEED_MAX_AGE_S: f32 = 300_000.0;
const SEED_MIN_PERSIST_RESERVE: f32 = 0.006;
const SEED_MIN_PERSIST_VITALITY: f32 = 0.06;
const SEED_MIN_PERSIST_COAT_INTEGRITY: f32 = 0.04;

#[derive(Debug, Clone, Copy)]
pub struct SeedTransportInputs {
    pub cell_size_mm: f32,
    pub bioturbation_mm2_day: f32,
    pub surface_moisture: f32,
    pub deep_moisture: f32,
    pub cover_fraction: f32,
    pub support_depth_mm: f32,
    pub pore_exposure: f32,
    pub roughness_mm: f32,
    pub collapse_rate: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SeedMicrositeInputs {
    pub cell_size_mm: f32,
    pub daylight: f32,
    pub surface_moisture: f32,
    pub deep_moisture: f32,
    pub nutrients: f32,
    pub symbionts: f32,
    pub canopy: f32,
    pub litter_carbon: f32,
    pub organic_matter: f32,
    pub microbial_biomass: f32,
    pub nitrifier_biomass: f32,
    pub denitrifier_biomass: f32,
    pub substrate_microbial_activity: f32,
    pub soil_glucose: f32,
    pub soil_oxygen_gas: f32,
    pub soil_carbon_dioxide: f32,
    pub soil_ammonium: f32,
    pub soil_nitrate: f32,
    pub soil_proton: f32,
    pub soil_atp_flux: f32,
    pub soil_amino_acids: f32,
    pub soil_nucleotides: f32,
    pub soil_membrane_precursors: f32,
    pub soil_dissolved_silicate: f32,
    pub soil_bicarbonate: f32,
    pub soil_surface_proton_load: f32,
    pub soil_calcium_bicarbonate_complex: f32,
    pub soil_exchangeable_calcium: f32,
    pub soil_exchangeable_magnesium: f32,
    pub soil_exchangeable_potassium: f32,
    pub soil_exchangeable_sodium: f32,
    pub soil_exchangeable_aluminum: f32,
    pub soil_aqueous_iron: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SeedMicrositeFeedback {
    pub burial_depth_mm: f32,
    pub surface_exposure: f32,
    pub local_light: f32,
    pub moisture: f32,
    pub deep_moisture: f32,
    pub nutrients: f32,
    pub symbionts: f32,
    pub canopy: f32,
    pub litter: f32,
    pub microbial_pressure: f32,
    pub oxygen_gas: f32,
    pub carbon_dioxide: f32,
    pub ammonium: f32,
    pub nitrate: f32,
    pub acidity: f32,
    pub atp_flux: f32,
    pub amino_acids: f32,
    pub nucleotides: f32,
    pub membrane_precursors: f32,
    pub base_saturation: f32,
    pub aluminum_toxicity: f32,
    pub weathering_support: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SeedMicrositeState {
    pub burial_depth_mm: f32,
    pub surface_exposure: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SeedDetritusReturn {
    pub litter_carbon: f32,
    pub organic_matter: f32,
    pub glucose: f32,
    pub ammonium: f32,
    pub carbon_dioxide: f32,
    pub amino_acids: f32,
    pub nucleotides: f32,
    pub membrane_precursors: f32,
}

impl Default for SeedMicrositeState {
    fn default() -> Self {
        Self {
            burial_depth_mm: 0.0,
            surface_exposure: 1.0,
        }
    }
}

fn seed_mobility(seed_mass: f32) -> f32 {
    let seed_mass_t = clamp(seed_mass / 0.20, 0.0, 1.0);
    clamp(1.08 - seed_mass_t * 0.66, 0.18, 1.0)
}

fn microsite_max_burial_mm(cell_size_mm: f32, support_depth_mm: f32, roughness_mm: f32) -> f32 {
    clamp(
        cell_size_mm * 0.18 + support_depth_mm * 0.78 + roughness_mm * 0.35,
        cell_size_mm * 0.10,
        cell_size_mm * 0.96,
    )
}

pub fn advance_seed_transport<R: Rng + ?Sized>(
    seed: &mut TerrariumSeed,
    eco_dt_s: f32,
    width: usize,
    height: usize,
    inputs: SeedTransportInputs,
    rng: &mut R,
) {
    let cell_size_mm = inputs.cell_size_mm.max(1.0e-3);
    let dt_days = eco_dt_s / 86_400.0;
    let mobility = seed_mobility(seed.genome.seed_mass);
    let mixing_sigma_mm = (2.0 * inputs.bioturbation_mm2_day.max(0.0) * dt_days).sqrt() * mobility;
    let cover_t = clamp(inputs.cover_fraction, 0.0, 1.0);
    let support_t = clamp(inputs.support_depth_mm / cell_size_mm, 0.0, 1.4);
    let pore_t = clamp(inputs.pore_exposure, 0.0, 1.0);
    let roughness_t = clamp(inputs.roughness_mm / cell_size_mm, 0.0, 1.0);
    let collapse_t = clamp(inputs.collapse_rate, 0.0, 1.0);
    let moisture_t = clamp(
        (inputs.surface_moisture * 0.55 + inputs.deep_moisture * 0.45) / 0.60,
        0.0,
        1.4,
    );
    let bioturb_t = clamp(inputs.bioturbation_mm2_day / 5.0, 0.0, 1.4);
    let max_burial_mm =
        microsite_max_burial_mm(cell_size_mm, inputs.support_depth_mm, inputs.roughness_mm);
    let burial_fill_t = clamp(
        seed.microsite.burial_depth_mm / max_burial_mm.max(1.0e-3),
        0.0,
        1.0,
    );
    let burial_bias_mm = mixing_sigma_mm
        * (0.03 + cover_t * 0.12 + support_t * 0.16 + roughness_t * 0.05)
        * (0.35 + moisture_t * 0.65)
        * (0.45 + support_t.min(1.0) * 0.55)
        * (1.0 - burial_fill_t * 0.70);
    let exposure_bias_mm = (cell_size_mm * (0.002 + dt_days * 0.26) + mixing_sigma_mm * 0.10)
        * (0.04
            + pore_t * 0.24
            + collapse_t * 0.20
            + (1.0 - cover_t) * 0.10
            + roughness_t * 0.08
            + burial_fill_t * 0.10
            + bioturb_t * 0.06);
    let burial_noise: f32 = rng.sample(StandardNormal);
    let burial_delta_mm = burial_noise * mixing_sigma_mm * 0.42 + burial_bias_mm - exposure_bias_mm;
    seed.microsite.burial_depth_mm = clamp(
        seed.microsite.burial_depth_mm + burial_delta_mm,
        0.0,
        max_burial_mm,
    );

    let exposure_depth_mm = (cell_size_mm * 0.70).max(1.0e-3);
    seed.microsite.surface_exposure = clamp(
        1.0 - seed.microsite.burial_depth_mm / exposure_depth_mm,
        0.04,
        1.0,
    );

    let lateral_sigma_mm =
        mixing_sigma_mm * (0.18 + seed.microsite.surface_exposure * 0.32).clamp(0.12, 0.50);
    let dx_mm: f32 = rng.sample::<f32, _>(StandardNormal) * lateral_sigma_mm;
    let dy_mm: f32 = rng.sample::<f32, _>(StandardNormal) * lateral_sigma_mm;
    seed.x = clamp(seed.x + dx_mm / cell_size_mm, 0.0, width as f32 - 0.01);
    seed.y = clamp(seed.y + dy_mm / cell_size_mm, 0.0, height as f32 - 0.01);

    seed.pose.offset_mm[0] = 0.0;
    seed.pose.offset_mm[1] = -seed.microsite.burial_depth_mm;
    seed.pose.offset_mm[2] = 0.0;

    let tumble_x: f32 = rng.sample(StandardNormal);
    let tumble_z: f32 = rng.sample(StandardNormal);
    let burial_t = 1.0 - seed.microsite.surface_exposure;
    seed.pose.rotation_xyz_rad[0] = clamp(
        seed.pose.rotation_xyz_rad[0] * 0.45 + tumble_x * 0.08 + burial_t * 0.18,
        -0.85,
        0.85,
    );
    seed.pose.rotation_xyz_rad[1] = clamp(
        seed.pose.rotation_xyz_rad[1] * 0.35 + burial_noise * 0.05,
        -0.45,
        0.45,
    );
    seed.pose.rotation_xyz_rad[2] = clamp(
        seed.pose.rotation_xyz_rad[2] * 0.45 + tumble_z * 0.08 - burial_t * 0.12,
        -0.85,
        0.85,
    );
}

pub fn sample_seed_microsite(
    seed: &TerrariumSeed,
    inputs: SeedMicrositeInputs,
) -> SeedMicrositeFeedback {
    let buried_t = 1.0 - seed.microsite.surface_exposure;
    let light = clamp(inputs.daylight * seed.microsite.surface_exposure, 0.0, 1.0);
    let moisture = clamp(
        inputs.surface_moisture * seed.microsite.surface_exposure + inputs.deep_moisture * buried_t,
        0.0,
        1.2,
    );
    let deep_moisture = clamp(inputs.deep_moisture * (0.92 + buried_t * 0.12), 0.0, 1.2);
    let ammonium = clamp(inputs.soil_ammonium * (0.78 + buried_t * 0.28), 0.0, 1.8);
    let nitrate = clamp(inputs.soil_nitrate * (0.76 + buried_t * 0.24), 0.0, 1.8);
    let amino_acids = clamp(
        inputs.soil_amino_acids * (0.76 + buried_t * 0.30) + inputs.soil_glucose * 0.08,
        0.0,
        1.8,
    );
    let nucleotides = clamp(
        inputs.soil_nucleotides * (0.74 + buried_t * 0.28) + nitrate * 0.10,
        0.0,
        1.8,
    );
    let membrane_precursors = clamp(
        inputs.soil_membrane_precursors * (0.68 + buried_t * 0.34)
            + amino_acids * 0.08
            + inputs.organic_matter * 0.06,
        0.0,
        1.8,
    );
    let oxygen_gas = clamp(
        inputs.soil_oxygen_gas * (0.22 + seed.microsite.surface_exposure * 0.78),
        0.0,
        1.6,
    );
    let carbon_dioxide = clamp(
        inputs.soil_carbon_dioxide * (0.84 + buried_t * 0.24),
        0.0,
        2.0,
    );
    let acidity = clamp(inputs.soil_proton * (0.82 + buried_t * 0.24), 0.0, 2.0);
    let atp_flux = clamp(inputs.soil_atp_flux, 0.0, 1.8);
    let base_saturation = soil_base_saturation(
        inputs.soil_exchangeable_calcium,
        inputs.soil_exchangeable_magnesium,
        inputs.soil_exchangeable_potassium,
        inputs.soil_exchangeable_sodium,
        inputs.soil_exchangeable_aluminum,
        acidity,
        inputs.soil_surface_proton_load,
    );
    let aluminum_toxicity = soil_aluminum_toxicity(
        inputs.soil_exchangeable_aluminum,
        acidity,
        inputs.soil_surface_proton_load,
        base_saturation,
    );
    let weathering_support = soil_weathering_support(
        inputs.soil_dissolved_silicate,
        inputs.soil_bicarbonate,
        inputs.soil_calcium_bicarbonate_complex,
        inputs.soil_exchangeable_calcium,
        inputs.soil_exchangeable_magnesium,
        inputs.soil_exchangeable_potassium,
        inputs.soil_aqueous_iron,
        0.0,
        0.0,
    );
    let nutrient_access = clamp(
        0.18 + oxygen_gas * 0.72
            + seed.microsite.surface_exposure * 0.14
            + base_saturation * 0.18
            + weathering_support * 0.12
            - acidity * 0.16
            - carbon_dioxide * 0.08
            - aluminum_toxicity * 0.20,
        0.04,
        1.0,
    );
    let nutrients = clamp(
        inputs.nutrients * (0.82 + buried_t * 0.08) * nutrient_access
            + inputs.litter_carbon * buried_t * 0.18 * (0.46 + nutrient_access * 0.54)
            + inputs.organic_matter * buried_t * 0.10 * (0.40 + nutrient_access * 0.60)
            + amino_acids * 0.24 * nutrient_access
            + nucleotides * 0.18 * nutrient_access
            + membrane_precursors * 0.10 * nutrient_access
            + ammonium * 0.10 * nutrient_access
            + nitrate * 0.08 * nutrient_access,
        0.0,
        1.6,
    );
    let microbial_pressure = clamp(
        (inputs.microbial_biomass * 0.70
            + inputs.nitrifier_biomass * 0.22
            + inputs.denitrifier_biomass * 0.22
            + inputs.substrate_microbial_activity * 0.55)
            * (0.28 + moisture * 0.46)
            * (0.22
                + inputs.soil_glucose * 0.22
                + amino_acids * 0.12
                + nucleotides * 0.10
                + membrane_precursors * 0.08
                + carbon_dioxide * 0.08)
            * (0.30 + buried_t * 0.70)
            * (0.92 + weathering_support * 0.06)
            * (1.04 - aluminum_toxicity * 0.08).clamp(0.72, 1.04),
        0.0,
        1.8,
    );
    let symbionts = clamp(inputs.symbionts * (0.78 + buried_t * 0.28), 0.0, 1.4);
    let litter = clamp(inputs.litter_carbon * (0.58 + buried_t * 0.72), 0.0, 1.4);
    let canopy = clamp(inputs.canopy, 0.0, 1.6);

    SeedMicrositeFeedback {
        burial_depth_mm: seed
            .microsite
            .burial_depth_mm
            .min(inputs.cell_size_mm.max(1.0e-3)),
        surface_exposure: seed.microsite.surface_exposure,
        local_light: light,
        moisture,
        deep_moisture,
        nutrients,
        symbionts,
        canopy,
        litter,
        microbial_pressure,
        oxygen_gas,
        carbon_dioxide,
        ammonium,
        nitrate,
        acidity,
        atp_flux,
        amino_acids,
        nucleotides,
        membrane_precursors,
        base_saturation,
        aluminum_toxicity,
        weathering_support,
    }
}

pub fn seed_probe_z(seed: &TerrariumSeed, cell_size_mm: f32, depth: usize) -> usize {
    let depth = depth.max(1);
    if depth <= 1 {
        return 0;
    }
    (seed.microsite.burial_depth_mm / cell_size_mm.max(1.0e-3))
        .round()
        .clamp(0.0, (depth - 1) as f32) as usize
}

/// Advance seed dormancy using Eyring TST temperature scaling.
///
/// Temperature sensitivity emerges from CYP707A (ABA catabolism) CO bond
/// physics — no hardcoded T_base cutoff. Rate naturally approaches zero at
/// low temperatures via Eyring transition state theory.
///
/// Moisture gating: Bradford 2002 hydrotime model, ψ_b(50) ≈ −1.0 MPa
/// maps to moisture half-max ~0.3 in [0,1] framework. Hill coefficient 2
/// represents cooperative imbibition kinetics (Bewley+ 2013 Ch.4).
///
/// Light factor: Casal & Sánchez 1998 phytochrome Pfr promotes germination
/// ~10-20% in photoblastic seeds.
pub fn advance_seed_dormancy(
    dormancy_s: f32,
    eco_dt_s: f32,
    moisture: f32,
    local_light: f32,
    temperature_c: f32,
) -> f32 {
    use super::emergent_rates::metabolome_rate;

    // Eyring TST: temperature scaling from CO bond + enzyme eff=0.88
    // At 25°C returns Vmax=1.4; at 0°C near-zero (no hardcoded T_base)
    let temp_rate = metabolome_rate("seed_germination", temperature_c) as f32;

    // Bradford 2002: ψ_b(50) ≈ −1.0 MPa → Km=0.3 moisture fraction
    // Hill=2: cooperative water imbibition (Bewley+ 2013 Ch.4)
    let moisture_gate = moisture * moisture / (0.09 + moisture * moisture);

    // Casal & Sánchez 1998: phytochrome Pfr germination promotion
    let light_factor = 1.0 + local_light * 0.15;

    dormancy_s - temp_rate * moisture_gate * light_factor * eco_dt_s
}

pub fn seed_should_persist(
    age_s: f32,
    reserve_carbon: f32,
    feedback: &crate::seed_cellular::SeedCellularFeedback,
    microsite: &SeedMicrositeFeedback,
) -> bool {
    let decay_pressure = clamp(
        microsite.microbial_pressure * 0.45
            + microsite.acidity * 0.18
            + (0.18 - microsite.oxygen_gas).max(0.0) * 1.6
            + microsite.aluminum_toxicity * 0.18
            + (0.40 - microsite.base_saturation).max(0.0) * 0.14,
        0.0,
        1.8,
    );
    let reserve_floor = clamp(
        SEED_MIN_PERSIST_RESERVE + decay_pressure * 0.03,
        0.006,
        0.08,
    );
    let vitality_floor = clamp(0.06 + decay_pressure * 0.14, 0.06, 0.32);
    let coat_floor = clamp(0.04 + decay_pressure * 0.24, 0.04, 0.55);
    age_s < SEED_MAX_AGE_S
        && reserve_carbon >= reserve_floor
        && feedback.vitality >= vitality_floor.max(SEED_MIN_PERSIST_VITALITY)
        && feedback.coat_integrity >= coat_floor.max(SEED_MIN_PERSIST_COAT_INTEGRITY)
}

pub fn seedling_scale(
    reserve_carbon: f32,
    feedback: &crate::seed_cellular::SeedCellularFeedback,
) -> f32 {
    clamp(
        0.28 + reserve_carbon.max(0.0).powf(0.75) * 2.0
            + feedback.radicle_extension * 0.10
            + feedback.cotyledon_opening * 0.08,
        0.45,
        1.10,
    )
}

pub fn seedling_emergence_capacity_mm(
    seed: &TerrariumSeed,
    feedback: &crate::seed_cellular::SeedCellularFeedback,
    cell_size_mm: f32,
) -> f32 {
    let seed_mass_t = clamp((seed.genome.seed_mass - 0.018) / 0.182, 0.0, 1.0);
    let root_depth_t = clamp(seed.genome.root_depth_bias / 1.1, 0.0, 1.0);
    let vigor_t = clamp(
        feedback.radicle_extension * 0.34
            + feedback.cotyledon_opening * 0.16
            + feedback.energy_charge * 0.16
            + feedback.hydration.min(1.0) * 0.14
            + feedback.vitality * 0.14
            + (1.0 - feedback.coat_integrity) * 0.06,
        0.0,
        1.4,
    );
    cell_size_mm.max(1.0e-3)
        * clamp(
            0.12 + seed_mass_t * 0.22 + root_depth_t * 0.12 + vigor_t * 0.30,
            0.10,
            1.10,
        )
}

pub fn seed_can_emerge(
    seed: &TerrariumSeed,
    feedback: &crate::seed_cellular::SeedCellularFeedback,
    microsite: &SeedMicrositeFeedback,
    cell_size_mm: f32,
) -> bool {
    microsite.burial_depth_mm <= seedling_emergence_capacity_mm(seed, feedback, cell_size_mm)
}

pub fn seed_detritus_return(
    reserve_carbon: f32,
    feedback: &crate::seed_cellular::SeedCellularFeedback,
    microsite: &SeedMicrositeFeedback,
) -> SeedDetritusReturn {
    let necrosis_t = clamp(
        (1.0 - feedback.vitality) * 0.42
            + (1.0 - feedback.coat_integrity) * 0.34
            + microsite.microbial_pressure * 0.22
            + microsite.acidity * 0.10
            + (0.22 - microsite.oxygen_gas).max(0.0) * 0.55
            + microsite.aluminum_toxicity * 0.12,
        0.0,
        1.8,
    );
    let return_mass = reserve_carbon.max(0.0) * (0.35 + necrosis_t * 0.65);
    SeedDetritusReturn {
        litter_carbon: return_mass * 0.42,
        organic_matter: return_mass * 0.18,
        glucose: return_mass * (5.0 + microsite.microbial_pressure * 2.2),
        ammonium: return_mass * (1.6 + microsite.ammonium * 0.35 + microsite.nitrate * 0.12),
        carbon_dioxide: return_mass * (1.1 + microsite.carbon_dioxide * 0.28),
        amino_acids: return_mass * (0.92 + microsite.amino_acids * 0.44),
        nucleotides: return_mass * (0.54 + microsite.nucleotides * 0.36),
        membrane_precursors: return_mass * (0.48 + microsite.membrane_precursors * 0.42),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        advance_seed_dormancy, advance_seed_transport, sample_seed_microsite, seed_can_emerge,
        seed_probe_z, seed_should_persist, seedling_emergence_capacity_mm, seedling_scale,
        SeedMicrositeInputs, SeedTransportInputs,
    };
    use crate::seed_cellular::SeedCellularFeedback;
    use crate::terrarium::seed_microsite::SeedMicrositeState;
    use crate::terrarium::{
        plant_species, OrganismIdentity, RegionalMaterialInventory, SeedPose,
        TerrariumOrganismKind, TerrariumSeed,
    };
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn bioturbation_can_bury_without_forcing_full_seed_loss() {
        let mut rng = StdRng::seed_from_u64(9);
        let genome = plant_species::genome_for_taxonomy(3750, &mut rng);
        let mut seed = TerrariumSeed {
            identity: OrganismIdentity::synthetic(TerrariumOrganismKind::Seed, 1),
            x: 8.0,
            y: 8.0,
            dormancy_s: 10_000.0,
            reserve_carbon: 0.16,
            age_s: 1_200.0,
            genome: genome.clone(),
            cellular: crate::seed_cellular::SeedCellularStateSim::new(
                genome.seed_mass,
                0.16,
                10_000.0,
            ),
            pose: SeedPose::default(),
            microsite: SeedMicrositeState::default(),
            material_inventory: RegionalMaterialInventory::new("seed:test:bioturb".into()),
        };

        for _ in 0..24 {
            advance_seed_transport(
                &mut seed,
                120.0,
                16,
                16,
                SeedTransportInputs {
                    cell_size_mm: 0.24,
                    bioturbation_mm2_day: 5.0,
                    surface_moisture: 0.86,
                    deep_moisture: 0.74,
                    cover_fraction: 0.58,
                    support_depth_mm: 0.14,
                    pore_exposure: 0.10,
                    roughness_mm: 0.03,
                    collapse_rate: 0.08,
                },
                &mut rng,
            );
        }

        assert!(seed.microsite.burial_depth_mm > 0.0);
        assert!(seed.microsite.surface_exposure < 1.0);
        assert!(seed.pose.offset_mm[1] < 0.0);
    }

    #[test]
    fn drying_sparse_surface_can_reexpose_buried_seed() {
        let mut rng = StdRng::seed_from_u64(19);
        let genome = plant_species::genome_for_taxonomy(3750, &mut rng);
        let mut seed = TerrariumSeed {
            identity: OrganismIdentity::synthetic(TerrariumOrganismKind::Seed, 2),
            x: 8.0,
            y: 8.0,
            dormancy_s: 6_000.0,
            reserve_carbon: 0.12,
            age_s: 2_400.0,
            genome: genome.clone(),
            cellular: crate::seed_cellular::SeedCellularStateSim::new(
                genome.seed_mass,
                0.12,
                6_000.0,
            ),
            pose: SeedPose::default(),
            microsite: SeedMicrositeState {
                burial_depth_mm: 0.12,
                surface_exposure: 0.18,
            },
            material_inventory: RegionalMaterialInventory::new("seed:test:reexpose".into()),
        };
        let start_depth = seed.microsite.burial_depth_mm;
        let start_exposure = seed.microsite.surface_exposure;

        for _ in 0..32 {
            advance_seed_transport(
                &mut seed,
                120.0,
                16,
                16,
                SeedTransportInputs {
                    cell_size_mm: 0.24,
                    bioturbation_mm2_day: 2.6,
                    surface_moisture: 0.08,
                    deep_moisture: 0.78,
                    cover_fraction: 0.06,
                    support_depth_mm: 0.02,
                    pore_exposure: 0.72,
                    roughness_mm: 0.05,
                    collapse_rate: 0.64,
                },
                &mut rng,
            );
        }

        assert!(seed.microsite.burial_depth_mm < start_depth);
        assert!(seed.microsite.surface_exposure > start_exposure);
        assert!(seed.pose.offset_mm[1] > -start_depth);
    }

    #[test]
    fn sampled_microsite_blends_surface_and_deep_conditions() {
        let mut rng = StdRng::seed_from_u64(11);
        let genome = plant_species::genome_for_taxonomy(3750, &mut rng);
        let seed = TerrariumSeed {
            identity: OrganismIdentity::synthetic(TerrariumOrganismKind::Seed, 3),
            x: 4.0,
            y: 4.0,
            dormancy_s: 0.0,
            reserve_carbon: 0.18,
            age_s: 0.0,
            genome: genome.clone(),
            cellular: crate::seed_cellular::SeedCellularStateSim::new(genome.seed_mass, 0.18, 0.0),
            pose: SeedPose::default(),
            microsite: SeedMicrositeState {
                burial_depth_mm: 0.12,
                surface_exposure: 0.38,
            },
            material_inventory: RegionalMaterialInventory::new("seed:test:sampled".into()),
        };
        let sampled = sample_seed_microsite(
            &seed,
            SeedMicrositeInputs {
                cell_size_mm: 0.24,
                daylight: 0.9,
                surface_moisture: 0.22,
                deep_moisture: 0.84,
                nutrients: 0.20,
                symbionts: 0.08,
                canopy: 0.18,
                litter_carbon: 0.12,
                organic_matter: 0.66,
                microbial_biomass: 0.24,
                nitrifier_biomass: 0.06,
                denitrifier_biomass: 0.04,
                substrate_microbial_activity: 0.32,
                soil_glucose: 0.20,
                soil_oxygen_gas: 0.14,
                soil_carbon_dioxide: 0.48,
                soil_ammonium: 0.16,
                soil_nitrate: 0.08,
                soil_proton: 0.22,
                soil_atp_flux: 0.10,
                soil_amino_acids: 0.26,
                soil_nucleotides: 0.14,
                soil_membrane_precursors: 0.18,
                soil_dissolved_silicate: 0.08,
                soil_bicarbonate: 0.12,
                soil_surface_proton_load: 0.10,
                soil_calcium_bicarbonate_complex: 0.09,
                soil_exchangeable_calcium: 0.12,
                soil_exchangeable_magnesium: 0.08,
                soil_exchangeable_potassium: 0.05,
                soil_exchangeable_sodium: 0.03,
                soil_exchangeable_aluminum: 0.04,
                soil_aqueous_iron: 0.06,
            },
        );

        assert!(sampled.moisture > 0.22);
        assert!(sampled.deep_moisture > 0.80);
        assert!(sampled.local_light < 0.9);
        assert!(sampled.litter > 0.12);
        assert!(sampled.microbial_pressure > 0.0);
        assert!(sampled.oxygen_gas < 0.14);
        assert!(sampled.carbon_dioxide > 0.45);
        assert!(sampled.base_saturation > 0.0);
        assert!(sampled.weathering_support > 0.0);
    }

    #[test]
    fn seed_persistence_and_scale_stay_bounded() {
        let feedback = SeedCellularFeedback {
            reserve_carbon: 0.12,
            vitality: 0.44,
            energy_charge: 0.35,
            hydration: 0.52,
            germination_drive: 0.66,
            germination_readiness: 0.58,
            ready_to_germinate: true,
            radicle_extension: 0.48,
            cotyledon_opening: 0.42,
            coat_integrity: 0.38,
        };
        let microsite = super::SeedMicrositeFeedback {
            burial_depth_mm: 0.02,
            surface_exposure: 0.84,
            local_light: 0.62,
            moisture: 0.68,
            deep_moisture: 0.54,
            nutrients: 0.22,
            symbionts: 0.12,
            canopy: 0.18,
            litter: 0.10,
            microbial_pressure: 0.08,
            oxygen_gas: 0.32,
            carbon_dioxide: 0.10,
            ammonium: 0.12,
            nitrate: 0.08,
            acidity: 0.08,
            atp_flux: 0.06,
            amino_acids: 0.10,
            nucleotides: 0.06,
            membrane_precursors: 0.04,
            base_saturation: 0.72,
            aluminum_toxicity: 0.04,
            weathering_support: 0.22,
        };

        assert!(seed_should_persist(3_000.0, 0.12, &feedback, &microsite));
        assert!((0.45..=1.10).contains(&seedling_scale(0.12, &feedback)));
        assert!(advance_seed_dormancy(10_000.0, 60.0, 0.7, 0.4, 20.0) < 10_000.0);
    }

    #[test]
    fn seed_probe_z_tracks_burial_depth() {
        let mut rng = StdRng::seed_from_u64(13);
        let genome = plant_species::genome_for_taxonomy(3750, &mut rng);
        let seed = TerrariumSeed {
            identity: OrganismIdentity::synthetic(TerrariumOrganismKind::Seed, 4),
            x: 0.0,
            y: 0.0,
            dormancy_s: 0.0,
            reserve_carbon: 0.10,
            age_s: 0.0,
            genome: genome.clone(),
            cellular: crate::seed_cellular::SeedCellularStateSim::new(genome.seed_mass, 0.10, 0.0),
            pose: SeedPose::default(),
            microsite: SeedMicrositeState {
                burial_depth_mm: 0.18,
                surface_exposure: 0.22,
            },
            material_inventory: RegionalMaterialInventory::new("seed:test:probe".into()),
        };

        assert_eq!(seed_probe_z(&seed, 0.24, 4), 1);
        assert_eq!(seed_probe_z(&seed, 0.24, 1), 0);
    }

    #[test]
    fn emergence_capacity_scales_with_seed_traits_and_vigor() {
        let mut rng = StdRng::seed_from_u64(17);
        let mut shallow_seed_genome = plant_species::genome_for_taxonomy(3750, &mut rng);
        shallow_seed_genome.seed_mass = 0.022;
        shallow_seed_genome.root_depth_bias = 0.08;
        let shallow_seed = TerrariumSeed {
            identity: OrganismIdentity::synthetic(TerrariumOrganismKind::Seed, 5),
            x: 0.0,
            y: 0.0,
            dormancy_s: 0.0,
            reserve_carbon: 0.10,
            age_s: 0.0,
            genome: shallow_seed_genome.clone(),
            cellular: crate::seed_cellular::SeedCellularStateSim::new(
                shallow_seed_genome.seed_mass,
                0.10,
                0.0,
            ),
            pose: SeedPose::default(),
            microsite: SeedMicrositeState {
                burial_depth_mm: 0.09,
                surface_exposure: 0.46,
            },
            material_inventory: RegionalMaterialInventory::new("seed:test:shallow".into()),
        };

        let mut deep_seed_genome = shallow_seed_genome.clone();
        deep_seed_genome.seed_mass = 0.14;
        deep_seed_genome.root_depth_bias = 0.82;
        let deep_seed = TerrariumSeed {
            identity: OrganismIdentity::synthetic(TerrariumOrganismKind::Seed, 6),
            x: 0.0,
            y: 0.0,
            dormancy_s: 0.0,
            reserve_carbon: 0.18,
            age_s: 0.0,
            genome: deep_seed_genome.clone(),
            cellular: crate::seed_cellular::SeedCellularStateSim::new(
                deep_seed_genome.seed_mass,
                0.18,
                0.0,
            ),
            pose: SeedPose::default(),
            microsite: SeedMicrositeState {
                burial_depth_mm: 0.09,
                surface_exposure: 0.46,
            },
            material_inventory: RegionalMaterialInventory::new("seed:test:deep".into()),
        };

        let weak_feedback = SeedCellularFeedback {
            reserve_carbon: 0.10,
            vitality: 0.44,
            energy_charge: 0.34,
            hydration: 0.46,
            germination_drive: 0.52,
            germination_readiness: 0.50,
            ready_to_germinate: true,
            radicle_extension: 0.30,
            cotyledon_opening: 0.28,
            coat_integrity: 0.76,
        };
        let strong_feedback = SeedCellularFeedback {
            reserve_carbon: 0.18,
            vitality: 0.76,
            energy_charge: 0.72,
            hydration: 0.84,
            germination_drive: 0.96,
            germination_readiness: 0.88,
            ready_to_germinate: true,
            radicle_extension: 0.82,
            cotyledon_opening: 0.74,
            coat_integrity: 0.48,
        };
        let microsite = super::SeedMicrositeFeedback {
            burial_depth_mm: 0.09,
            surface_exposure: 0.46,
            local_light: 0.42,
            moisture: 0.82,
            deep_moisture: 0.86,
            nutrients: 0.30,
            symbionts: 0.10,
            canopy: 0.16,
            litter: 0.12,
            microbial_pressure: 0.04,
            oxygen_gas: 0.30,
            carbon_dioxide: 0.12,
            ammonium: 0.16,
            nitrate: 0.12,
            acidity: 0.06,
            atp_flux: 0.04,
            amino_acids: 0.14,
            nucleotides: 0.08,
            membrane_precursors: 0.06,
            base_saturation: 0.78,
            aluminum_toxicity: 0.03,
            weathering_support: 0.24,
        };

        let weak_capacity = seedling_emergence_capacity_mm(&shallow_seed, &weak_feedback, 0.24);
        let strong_capacity = seedling_emergence_capacity_mm(&deep_seed, &strong_feedback, 0.24);

        assert!(strong_capacity > weak_capacity);
        assert!(!seed_can_emerge(
            &shallow_seed,
            &weak_feedback,
            &microsite,
            0.24
        ));
        assert!(seed_can_emerge(
            &deep_seed,
            &strong_feedback,
            &microsite,
            0.24
        ));
    }

    #[test]
    fn seed_dormancy_cold_no_progress() {
        // Below Eyring TST effective threshold, dormancy reduction is near-zero.
        // At 0°C the CO bond eff=0.88 gives a very low rate.
        let d0 = 10_000.0;
        let d_cold = advance_seed_dormancy(d0, 60.0, 0.8, 0.5, 0.0);
        let d_warm = advance_seed_dormancy(d0, 60.0, 0.8, 0.5, 20.0);
        // Cold should barely reduce dormancy compared to warm.
        let cold_reduction = d0 - d_cold;
        let warm_reduction = d0 - d_warm;
        assert!(
            cold_reduction < warm_reduction * 0.3,
            "At 0°C dormancy reduction ({cold_reduction:.2}) should be <30% of 20°C ({warm_reduction:.2})"
        );
    }

    #[test]
    fn seed_dormancy_warm_moist_germinates() {
        // At 25°C + good moisture (0.8) + light (0.5): dormancy should break.
        // At reference temperature, rate ≈ 1.4 * 0.877 * 1.075 ≈ 1.32 / sim-s.
        // 5000 / 1.32 ≈ 3787s → ~63 minutes of 60s steps.
        let mut dormancy = 5_000.0;
        for _ in 0..100 {
            dormancy = advance_seed_dormancy(dormancy, 60.0, 0.8, 0.5, 25.0);
        }
        assert!(
            dormancy <= 0.0,
            "Warm moist seed should germinate within 100 min at 25°C: dormancy={dormancy:.1}"
        );
    }

    #[test]
    fn seed_dormancy_warm_dry_stalled() {
        // At 20°C but very dry (0.01): moisture gate should nearly block progress.
        let d0 = 10_000.0;
        let d_dry = advance_seed_dormancy(d0, 60.0, 0.01, 0.5, 20.0);
        let d_wet = advance_seed_dormancy(d0, 60.0, 0.8, 0.5, 20.0);
        let dry_reduction = d0 - d_dry;
        let wet_reduction = d0 - d_wet;
        assert!(
            dry_reduction < wet_reduction * 0.05,
            "Dry seed should barely progress: dry={dry_reduction:.2}, wet={wet_reduction:.2}"
        );
    }

    #[test]
    fn seed_dormancy_temperature_proportional() {
        // Higher temperature should break dormancy faster (Eyring TST scaling).
        let d0 = 10_000.0;
        let d_20 = advance_seed_dormancy(d0, 60.0, 0.7, 0.4, 20.0);
        let d_30 = advance_seed_dormancy(d0, 60.0, 0.7, 0.4, 30.0);
        let reduction_20 = d0 - d_20;
        let reduction_30 = d0 - d_30;
        assert!(
            reduction_30 > reduction_20 * 1.3,
            "30°C should break dormancy >1.3× faster than 20°C: 30°C={reduction_30:.2}, 20°C={reduction_20:.2}"
        );
    }
}
