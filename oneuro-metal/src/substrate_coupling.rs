//! Reusable coupling helpers between coarse world fields and explicit substrate state.

use crate::constants::clamp;
use crate::field_coupling::layer_mean_map;

fn packet_surface_factor(cells: f32, packets: f32, target_cells: f32) -> f32 {
    let represented = packets.max(0.0) * target_cells.max(1.0);
    clamp(
        if represented <= 1.0e-6 {
            0.12
        } else {
            (cells.max(0.0) / represented).sqrt()
        },
        0.12,
        1.75,
    )
}

fn trait_match(current: f32, target: f32) -> f32 {
    let diff = (current - target).abs();
    clamp(1.0 - diff * 1.8, 0.18, 1.0)
}

pub fn microbial_copiotroph_target(
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

pub fn nitrifier_aerobic_target(oxygen_factor: f32, aeration_factor: f32, anoxia_factor: f32) -> f32 {
    clamp(
        0.18 + oxygen_factor * 0.56 + aeration_factor * 0.18 - anoxia_factor * 0.22,
        0.05,
        0.98,
    )
}

pub fn denitrifier_anoxic_target(anoxia_factor: f32, deep_moisture: f32, oxygen_factor: f32) -> f32 {
    clamp(
        0.16 + anoxia_factor * 0.60 + deep_moisture * 0.10 - oxygen_factor * 0.20,
        0.05,
        0.98,
    )
}

fn validate_len(name: &str, slice: &[f32], expected: usize) -> Result<(), String> {
    if slice.len() == expected {
        Ok(())
    } else {
        Err(format!(
            "{name} length mismatch: expected {expected}, got {}",
            slice.len()
        ))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubstrateControlConfig {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub daylight: f32,
    pub ownership_threshold: f32,
    pub microbial_packet_target_cells: f32,
    pub nitrifier_packet_target_cells: f32,
    pub denitrifier_packet_target_cells: f32,
}

pub struct SubstrateControlInputs<'a> {
    pub temperature: &'a [f32],
    pub moisture: &'a [f32],
    pub deep_moisture: &'a [f32],
    pub litter_carbon: &'a [f32],
    pub root_exudates: &'a [f32],
    pub organic_matter: &'a [f32],
    pub root_density: &'a [f32],
    pub symbiont_biomass: &'a [f32],
    pub nitrification_potential: &'a [f32],
    pub denitrification_potential: &'a [f32],
    pub explicit_microbe_authority: &'a [f32],
    pub explicit_microbe_activity: &'a [f32],
    pub microbial_cells: &'a [f32],
    pub microbial_packets: &'a [f32],
    pub microbial_copiotroph_fraction: &'a [f32],
    pub microbial_dormancy: &'a [f32],
    pub microbial_vitality: &'a [f32],
    pub microbial_reserve: &'a [f32],
    pub nitrifier_cells: &'a [f32],
    pub nitrifier_packets: &'a [f32],
    pub nitrifier_aerobic_fraction: &'a [f32],
    pub nitrifier_dormancy: &'a [f32],
    pub nitrifier_vitality: &'a [f32],
    pub nitrifier_reserve: &'a [f32],
    pub denitrifier_cells: &'a [f32],
    pub denitrifier_packets: &'a [f32],
    pub denitrifier_anoxic_fraction: &'a [f32],
    pub denitrifier_dormancy: &'a [f32],
    pub denitrifier_vitality: &'a [f32],
    pub denitrifier_reserve: &'a [f32],
}

#[derive(Debug, Clone)]
pub struct SubstrateControlFields {
    pub hydration: Vec<f32>,
    pub soil_temperature: Vec<f32>,
    pub decomposers: Vec<f32>,
    pub nitrifiers: Vec<f32>,
    pub denitrifiers: Vec<f32>,
    pub plant_drive: Vec<f32>,
}

pub fn build_substrate_control_fields(
    config: SubstrateControlConfig,
    inputs: SubstrateControlInputs<'_>,
) -> Result<SubstrateControlFields, String> {
    let plane = config.width * config.height;
    let depth = config.depth.max(1);
    let total = plane * depth;

    validate_len("temperature", inputs.temperature, total)?;
    for (name, slice) in [
        ("moisture", inputs.moisture),
        ("deep_moisture", inputs.deep_moisture),
        ("litter_carbon", inputs.litter_carbon),
        ("root_exudates", inputs.root_exudates),
        ("organic_matter", inputs.organic_matter),
        ("root_density", inputs.root_density),
        ("symbiont_biomass", inputs.symbiont_biomass),
        ("nitrification_potential", inputs.nitrification_potential),
        (
            "denitrification_potential",
            inputs.denitrification_potential,
        ),
        (
            "explicit_microbe_authority",
            inputs.explicit_microbe_authority,
        ),
        (
            "explicit_microbe_activity",
            inputs.explicit_microbe_activity,
        ),
        ("microbial_cells", inputs.microbial_cells),
        ("microbial_packets", inputs.microbial_packets),
        (
            "microbial_copiotroph_fraction",
            inputs.microbial_copiotroph_fraction,
        ),
        ("microbial_dormancy", inputs.microbial_dormancy),
        ("microbial_vitality", inputs.microbial_vitality),
        ("microbial_reserve", inputs.microbial_reserve),
        ("nitrifier_cells", inputs.nitrifier_cells),
        ("nitrifier_packets", inputs.nitrifier_packets),
        (
            "nitrifier_aerobic_fraction",
            inputs.nitrifier_aerobic_fraction,
        ),
        ("nitrifier_dormancy", inputs.nitrifier_dormancy),
        ("nitrifier_vitality", inputs.nitrifier_vitality),
        ("nitrifier_reserve", inputs.nitrifier_reserve),
        ("denitrifier_cells", inputs.denitrifier_cells),
        ("denitrifier_packets", inputs.denitrifier_packets),
        (
            "denitrifier_anoxic_fraction",
            inputs.denitrifier_anoxic_fraction,
        ),
        ("denitrifier_dormancy", inputs.denitrifier_dormancy),
        ("denitrifier_vitality", inputs.denitrifier_vitality),
        ("denitrifier_reserve", inputs.denitrifier_reserve),
    ] {
        validate_len(name, slice, plane)?;
    }

    let mut hydration = vec![0.0f32; total];
    let soil_temperature = inputs.temperature.to_vec();
    let mut decomposers = vec![0.0f32; total];
    let mut nitrifiers = vec![0.0f32; total];
    let mut denitrifiers = vec![0.0f32; total];
    let mut plant_drive = vec![0.0f32; total];

    for z in 0..depth {
        let z_frac = if depth > 1 {
            z as f32 / (depth - 1) as f32
        } else {
            0.0
        };
        for i in 0..plane {
            let gid = z * plane + i;
            let local_substrate = inputs.litter_carbon[i] * 1.10
                + inputs.root_exudates[i] * 1.35
                + inputs.organic_matter[i] * 0.90;
            let moisture_factor = clamp(
                (inputs.moisture[i] + inputs.deep_moisture[i] * 0.35) / 0.48,
                0.0,
                1.6,
            );
            let oxygen_factor = clamp(1.15 - inputs.deep_moisture[i] * 0.55, 0.35, 1.1);
            let aeration_factor = clamp(
                1.10 - inputs.moisture[i] * 0.55 - inputs.deep_moisture[i] * 0.45,
                0.05,
                1.15,
            );
            let anoxia_factor = clamp(
                (inputs.deep_moisture[i] * 0.95 + inputs.moisture[i] * 0.18) - oxygen_factor * 0.28,
                0.02,
                1.3,
            );
            let root_factor = 1.0 + inputs.root_density[i] * 0.08;
            let substrate_gate = clamp(local_substrate / 0.08, 0.0, 1.35);
            let explicit_authority = inputs.explicit_microbe_authority[i].clamp(0.0, 0.95);
            let explicit_activity = inputs.explicit_microbe_activity[i].max(0.0);
            let coarse_biology_factor = if explicit_authority >= config.ownership_threshold {
                0.0
            } else {
                1.0 - explicit_authority
            };
            let decomposer_trait_factor = trait_match(
                inputs.microbial_copiotroph_fraction[i],
                microbial_copiotroph_target(
                    substrate_gate,
                    moisture_factor,
                    oxygen_factor,
                    root_factor,
                ),
            );
            let decomposer_packet_factor = packet_surface_factor(
                inputs.microbial_cells[i],
                inputs.microbial_packets[i],
                config.microbial_packet_target_cells,
            );
            let decomposer_active = inputs.microbial_cells[i]
                * (1.0 - inputs.microbial_dormancy[i]).clamp(0.02, 1.0)
                * (0.25 + 0.75 * inputs.microbial_vitality[i]).clamp(0.0, 1.25)
                * (0.55 + 0.45 * inputs.microbial_reserve[i]).clamp(0.20, 1.25)
                * decomposer_packet_factor
                * decomposer_trait_factor
                * 0.040
                * coarse_biology_factor;
            let nitrifier_trait_factor = trait_match(
                inputs.nitrifier_aerobic_fraction[i],
                nitrifier_aerobic_target(oxygen_factor, aeration_factor, anoxia_factor),
            );
            let nitrifier_packet_factor = packet_surface_factor(
                inputs.nitrifier_cells[i],
                inputs.nitrifier_packets[i],
                config.nitrifier_packet_target_cells,
            );
            let nitrifier_active = inputs.nitrifier_cells[i]
                * (1.0 - inputs.nitrifier_dormancy[i]).clamp(0.02, 1.0)
                * (0.25 + 0.75 * inputs.nitrifier_vitality[i]).clamp(0.0, 1.25)
                * (0.55 + 0.45 * inputs.nitrifier_reserve[i]).clamp(0.20, 1.25)
                * nitrifier_packet_factor
                * nitrifier_trait_factor
                * 0.045
                * coarse_biology_factor;
            let denitrifier_trait_factor = trait_match(
                inputs.denitrifier_anoxic_fraction[i],
                denitrifier_anoxic_target(anoxia_factor, inputs.deep_moisture[i], oxygen_factor),
            );
            let denitrifier_packet_factor = packet_surface_factor(
                inputs.denitrifier_cells[i],
                inputs.denitrifier_packets[i],
                config.denitrifier_packet_target_cells,
            );
            let denitrifier_active = inputs.denitrifier_cells[i]
                * (1.0 - inputs.denitrifier_dormancy[i]).clamp(0.02, 1.0)
                * (0.25 + 0.75 * inputs.denitrifier_vitality[i]).clamp(0.0, 1.25)
                * (0.55 + 0.45 * inputs.denitrifier_reserve[i]).clamp(0.20, 1.25)
                * denitrifier_packet_factor
                * denitrifier_trait_factor
                * 0.045
                * coarse_biology_factor;
            hydration[gid] = clamp(
                inputs.moisture[i] * (1.0 - z_frac * 0.55) + inputs.deep_moisture[i] * z_frac,
                0.02,
                1.0,
            );
            let nitrif_signal = (inputs.nitrification_potential[i] * 9000.0).clamp(0.0, 1.0);
            let denit_signal = (inputs.denitrification_potential[i] * 9000.0).clamp(0.0, 1.0);
            if explicit_authority >= config.ownership_threshold {
                decomposers[gid] = 0.0;
                nitrifiers[gid] = 0.0;
                denitrifiers[gid] = 0.0;
            } else {
                decomposers[gid] = clamp(
                    decomposer_active * (0.65 + inputs.moisture[i] * 0.55)
                        + explicit_activity
                            * (0.42 + inputs.moisture[i] * 0.38)
                            * (0.92 - z_frac * 0.22).clamp(0.35, 1.0)
                        + inputs.symbiont_biomass[i] * (0.16 + z_frac * 0.10),
                    0.02,
                    1.2,
                );
                nitrifiers[gid] = clamp(
                    nitrifier_active
                        * (0.24 + (1.0 - z_frac) * 0.74)
                        * (0.72 + nitrif_signal * 0.75)
                        * (1.06 - inputs.deep_moisture[i] * 0.22),
                    0.0,
                    1.2,
                );
                denitrifiers[gid] = clamp(
                    denitrifier_active
                        * (0.12 + z_frac * 0.82)
                        * (0.70 + denit_signal * 0.85)
                        * (0.48 + inputs.deep_moisture[i] * 0.55),
                    0.0,
                    1.2,
                );
            }
            plant_drive[gid] = clamp(
                inputs.root_density[i] * (1.0 - z_frac * 0.35) * (0.35 + config.daylight * 0.65),
                0.0,
                1.5,
            );
        }
    }

    Ok(SubstrateControlFields {
        hydration,
        soil_temperature,
        decomposers,
        nitrifiers,
        denitrifiers,
        plant_drive,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct OwnedSummaryProjectionConfig {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub ownership_threshold: f32,
}

pub struct OwnedSummaryProjectionInputs<'a> {
    pub explicit_microbe_authority: &'a [f32],
    pub ammonium: &'a [f32],
    pub nitrate: &'a [f32],
    pub phosphorus: &'a [f32],
    pub glucose: &'a [f32],
    pub carbon_dioxide: &'a [f32],
    pub atp_flux: &'a [f32],
}

pub struct OwnedSummaryProjectionOutputs<'a> {
    pub root_exudates: &'a mut [f32],
    pub litter_carbon: &'a mut [f32],
    pub dissolved_nutrients: &'a mut [f32],
    pub shallow_nutrients: &'a mut [f32],
    pub mineral_nitrogen: &'a mut [f32],
    pub organic_matter: &'a mut [f32],
}

pub fn project_owned_summary_pools(
    config: OwnedSummaryProjectionConfig,
    inputs: OwnedSummaryProjectionInputs<'_>,
    outputs: OwnedSummaryProjectionOutputs<'_>,
) -> Result<(), String> {
    let plane = config.width * config.height;
    let depth = config.depth.max(1);
    let total = plane * depth;

    validate_len(
        "explicit_microbe_authority",
        inputs.explicit_microbe_authority,
        plane,
    )?;
    for (name, slice) in [
        ("ammonium", inputs.ammonium),
        ("nitrate", inputs.nitrate),
        ("phosphorus", inputs.phosphorus),
        ("glucose", inputs.glucose),
        ("carbon_dioxide", inputs.carbon_dioxide),
        ("atp_flux", inputs.atp_flux),
    ] {
        validate_len(name, slice, total)?;
    }
    for (name, slice) in [
        ("root_exudates", &*outputs.root_exudates),
        ("litter_carbon", &*outputs.litter_carbon),
        ("dissolved_nutrients", &*outputs.dissolved_nutrients),
        ("shallow_nutrients", &*outputs.shallow_nutrients),
        ("mineral_nitrogen", &*outputs.mineral_nitrogen),
        ("organic_matter", &*outputs.organic_matter),
    ] {
        validate_len(name, slice, plane)?;
    }

    let ammonium_surface = layer_mean_map(
        config.width,
        config.height,
        depth,
        inputs.ammonium,
        0,
        depth.min(2),
    );
    let nitrate_surface = layer_mean_map(
        config.width,
        config.height,
        depth,
        inputs.nitrate,
        0,
        depth.min(2),
    );
    let phosphorus_surface = layer_mean_map(
        config.width,
        config.height,
        depth,
        inputs.phosphorus,
        0,
        depth.min(2),
    );
    let glucose_surface = layer_mean_map(
        config.width,
        config.height,
        depth,
        inputs.glucose,
        0,
        depth.min(2),
    );
    let carbon_dioxide_surface = layer_mean_map(
        config.width,
        config.height,
        depth,
        inputs.carbon_dioxide,
        0,
        depth.min(2),
    );
    let atp_flux_surface = layer_mean_map(
        config.width,
        config.height,
        depth,
        inputs.atp_flux,
        0,
        depth.min(2),
    );

    for flat in 0..plane {
        if inputs.explicit_microbe_authority[flat] < config.ownership_threshold {
            continue;
        }
        let root_exudates = clamp(glucose_surface[flat] * 0.028, 0.0, 1.5);
        let litter_carbon = clamp(
            glucose_surface[flat] * 0.020 + carbon_dioxide_surface[flat] * 0.012,
            0.0,
            1.4,
        );
        let dissolved_nutrients = clamp(
            ammonium_surface[flat] * 0.090 + nitrate_surface[flat] * 0.028,
            0.0,
            1.5,
        );
        let shallow_nutrients = clamp(
            nitrate_surface[flat] * 0.082 + phosphorus_surface[flat] * 0.058,
            0.0,
            1.5,
        );
        let mineral_nitrogen = clamp(
            ammonium_surface[flat] * 0.095 + nitrate_surface[flat] * 0.062,
            0.0,
            1.5,
        );
        let organic_matter = clamp(
            litter_carbon * 0.58 + glucose_surface[flat] * 0.010 + atp_flux_surface[flat] * 48.0,
            0.0,
            2.5,
        );

        outputs.root_exudates[flat] = root_exudates;
        outputs.litter_carbon[flat] = litter_carbon;
        outputs.dissolved_nutrients[flat] = dissolved_nutrients;
        outputs.shallow_nutrients[flat] = shallow_nutrients;
        outputs.mineral_nitrogen[flat] = mineral_nitrogen;
        outputs.organic_matter[flat] = organic_matter;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        build_substrate_control_fields, project_owned_summary_pools, OwnedSummaryProjectionConfig,
        OwnedSummaryProjectionInputs, OwnedSummaryProjectionOutputs, SubstrateControlConfig,
        SubstrateControlInputs,
    };

    #[test]
    fn substrate_controls_disable_coarse_microbes_under_explicit_ownership() {
        let config = SubstrateControlConfig {
            width: 1,
            height: 1,
            depth: 2,
            daylight: 0.5,
            ownership_threshold: 0.5,
            microbial_packet_target_cells: 24.0,
            nitrifier_packet_target_cells: 24.0,
            denitrifier_packet_target_cells: 24.0,
        };
        let temperature = vec![23.0, 21.0];
        let plane = vec![0.6];
        let zero = vec![0.0];
        let one = vec![1.0];
        let fields = build_substrate_control_fields(
            config,
            SubstrateControlInputs {
                temperature: &temperature,
                moisture: &plane,
                deep_moisture: &plane,
                litter_carbon: &one,
                root_exudates: &one,
                organic_matter: &one,
                root_density: &one,
                symbiont_biomass: &zero,
                nitrification_potential: &zero,
                denitrification_potential: &zero,
                explicit_microbe_authority: &one,
                explicit_microbe_activity: &one,
                microbial_cells: &one,
                microbial_packets: &one,
                microbial_copiotroph_fraction: &plane,
                microbial_dormancy: &zero,
                microbial_vitality: &one,
                microbial_reserve: &one,
                nitrifier_cells: &one,
                nitrifier_packets: &one,
                nitrifier_aerobic_fraction: &plane,
                nitrifier_dormancy: &zero,
                nitrifier_vitality: &one,
                nitrifier_reserve: &one,
                denitrifier_cells: &one,
                denitrifier_packets: &one,
                denitrifier_anoxic_fraction: &plane,
                denitrifier_dormancy: &zero,
                denitrifier_vitality: &one,
                denitrifier_reserve: &one,
            },
        )
        .unwrap();

        assert!(fields.decomposers.iter().all(|value| *value == 0.0));
        assert!(fields.nitrifiers.iter().all(|value| *value == 0.0));
        assert!(fields.denitrifiers.iter().all(|value| *value == 0.0));
        assert!(fields.plant_drive.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn owned_summary_projection_only_updates_owned_cells() {
        let config = OwnedSummaryProjectionConfig {
            width: 2,
            height: 1,
            depth: 2,
            ownership_threshold: 0.5,
        };
        let explicit_microbe_authority = vec![0.8, 0.0];
        let ammonium = vec![1.0, 0.5, 1.0, 0.5];
        let nitrate = vec![0.6, 0.2, 0.6, 0.2];
        let phosphorus = vec![0.3, 0.1, 0.3, 0.1];
        let glucose = vec![0.9, 0.4, 0.9, 0.4];
        let carbon_dioxide = vec![0.7, 0.2, 0.7, 0.2];
        let atp_flux = vec![0.02, 0.0, 0.02, 0.0];
        let mut root_exudates = vec![0.0, 9.0];
        let mut litter_carbon = vec![0.0, 9.0];
        let mut dissolved_nutrients = vec![0.0, 9.0];
        let mut shallow_nutrients = vec![0.0, 9.0];
        let mut mineral_nitrogen = vec![0.0, 9.0];
        let mut organic_matter = vec![0.0, 9.0];

        project_owned_summary_pools(
            config,
            OwnedSummaryProjectionInputs {
                explicit_microbe_authority: &explicit_microbe_authority,
                ammonium: &ammonium,
                nitrate: &nitrate,
                phosphorus: &phosphorus,
                glucose: &glucose,
                carbon_dioxide: &carbon_dioxide,
                atp_flux: &atp_flux,
            },
            OwnedSummaryProjectionOutputs {
                root_exudates: &mut root_exudates,
                litter_carbon: &mut litter_carbon,
                dissolved_nutrients: &mut dissolved_nutrients,
                shallow_nutrients: &mut shallow_nutrients,
                mineral_nitrogen: &mut mineral_nitrogen,
                organic_matter: &mut organic_matter,
            },
        )
        .unwrap();

        assert!(root_exudates[0] > 0.0);
        assert!(litter_carbon[0] > 0.0);
        assert!(dissolved_nutrients[0] > 0.0);
        assert_eq!(root_exudates[1], 9.0);
        assert_eq!(litter_carbon[1], 9.0);
        assert_eq!(dissolved_nutrients[1], 9.0);
        assert_eq!(shallow_nutrients[1], 9.0);
        assert_eq!(mineral_nitrogen[1], 9.0);
        assert_eq!(organic_matter[1], 9.0);
    }
}
