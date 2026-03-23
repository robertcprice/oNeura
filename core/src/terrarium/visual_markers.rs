use super::*;
use crate::botany::visual_phenotype::MolecularVisualState;
use std::collections::HashMap;

fn select_hotspots(field: &[f32], width: usize, min_signal: f32, limit: usize) -> Vec<usize> {
    let mut ranked: Vec<usize> = field
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value >= min_signal).then_some(idx))
        .collect();
    ranked.sort_by(|&a, &b| field[b].total_cmp(&field[a]));

    let mut chosen = Vec::new();
    for idx in ranked {
        let x = idx % width.max(1);
        let y = idx / width.max(1);
        let separated = chosen.iter().all(|&existing| {
            let ex = existing % width.max(1);
            let ey = existing / width.max(1);
            x.abs_diff(ex) + y.abs_diff(ey) >= 3
        });
        if separated {
            chosen.push(idx);
            if chosen.len() >= limit {
                break;
            }
        }
    }
    chosen
}

impl TerrariumWorld {
    pub fn fruit_visuals(
        &self,
    ) -> Vec<crate::terrarium::visual_projection::TerrariumFruitVisualResponse> {
        self.fruit_visuals_at_time(self.time_s)
    }

    pub fn fruit_visuals_at_time(
        &self,
        time_s: f32,
    ) -> Vec<crate::terrarium::visual_projection::TerrariumFruitVisualResponse> {
        let atmosphere = self.atmosphere_frame();
        let width = self.config.width;
        let height = self.config.height;
        let parent_visual_state: HashMap<u64, MolecularVisualState> = self
            .plants
            .iter()
            .map(|plant| {
                let gene_snapshot = plant.botanical_genome.expression_snapshot();
                let molecular = MolecularVisualState::from_metabolome(
                    &plant.metabolome,
                    &gene_snapshot,
                    plant.physiology.leaf_biomass() + plant.physiology.stem_biomass(),
                );
                (plant.identity.organism_id, molecular)
            })
            .collect();
        self.fruits
            .iter()
            .enumerate()
            .map(|(i, fruit)| {
                let local_air = crate::terrarium::visual_projection::sample_visual_air(
                    &atmosphere,
                    width,
                    height,
                    fruit.source.x as f32 + 0.5,
                    fruit.source.y as f32 + 0.5,
                );
                let molecular = fruit
                    .identity
                    .parent_organism_id
                    .and_then(|organism_id| parent_visual_state.get(&organism_id).copied());
                crate::terrarium::visual_projection::fruit_visual_response_with_molecular_state(
                    fruit.taxonomy_id,
                    local_air,
                    fruit.source.ripeness,
                    fruit.source.sugar_content,
                    time_s,
                    i as f32,
                    self.config.visual_emergence_blend,
                    molecular,
                )
            })
            .collect()
    }

    pub fn seed_visuals(
        &self,
    ) -> Vec<crate::terrarium::visual_projection::TerrariumSeedVisualResponse> {
        self.seed_visuals_at_time(self.time_s)
    }

    pub fn seed_visuals_at_time(
        &self,
        time_s: f32,
    ) -> Vec<crate::terrarium::visual_projection::TerrariumSeedVisualResponse> {
        let atmosphere = self.atmosphere_frame();
        let width = self.config.width;
        let height = self.config.height;
        self.seeds
            .iter()
            .enumerate()
            .map(|(i, seed)| {
                let local_air = crate::terrarium::visual_projection::sample_visual_air(
                    &atmosphere,
                    width,
                    height,
                    seed.x,
                    seed.y,
                );
                crate::terrarium::visual_projection::seed_visual_response(
                    seed.genome.taxonomy_id,
                    local_air,
                    seed.reserve_carbon,
                    seed.dormancy_s,
                    time_s,
                    i as f32,
                )
            })
            .collect()
    }

    pub fn soil_surface_markers(
        &self,
    ) -> Vec<(
        usize,
        usize,
        crate::terrarium::visual_projection::TerrariumSoilSurfaceVisualResponse,
    )> {
        let atmosphere = self.atmosphere_frame();
        let width = self.config.width;
        let height = self.config.height;
        let atp_field = self.substrate.species_field(TerrariumSpecies::AtpFlux);
        let mut visuals = vec![
            crate::terrarium::visual_projection::TerrariumSoilSurfaceVisualResponse::default();
            width * height
        ];
        let mut signal = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let local_air = crate::terrarium::visual_projection::sample_visual_air(
                    &atmosphere,
                    width,
                    height,
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                );
                let chemistry =
                    crate::terrarium::visual_projection::sample_visual_chemistry(self, x, y);
                let nematode_density = self
                    .nematode_guilds
                    .iter()
                    .map(|guild| guild.population_density.get(idx).copied().unwrap_or(0.0))
                    .sum::<f32>();
                let shoreline_signal = shoreline_water_signal(width, height, &self.water_mask, idx);
                let mut visual = crate::terrarium::visual_projection::soil_surface_visual_response(
                    local_air,
                    chemistry,
                    self.moisture.get(idx).copied().unwrap_or(0.0),
                    self.organic_matter.get(idx).copied().unwrap_or(0.0),
                    self.microbial_biomass.get(idx).copied().unwrap_or(0.0),
                    self.symbiont_biomass.get(idx).copied().unwrap_or(0.0),
                    self.earthworm_population
                        .population_density
                        .get(idx)
                        .copied()
                        .unwrap_or(0.0),
                    nematode_density,
                    atp_field.get(idx).copied().unwrap_or(0.0),
                    self.nitrification_potential
                        .get(idx)
                        .copied()
                        .unwrap_or(0.0),
                    self.denitrification_potential
                        .get(idx)
                        .copied()
                        .unwrap_or(0.0),
                    shoreline_signal,
                );
                let x = idx % width;
                let y = idx / width;
                match self.ownership.get(idx).map(|cell| cell.owner) {
                    Some(SoilOwnershipClass::ExplicitMicrobeCohort { .. }) => {
                        if let Some(cohort) = self.explicit_microbe_at(x, y) {
                            let ecology = match cohort.guild {
                                1 => crate::terrarium::packet::GenotypePacketEcology::Nitrifier,
                                2 => crate::terrarium::packet::GenotypePacketEcology::Denitrifier,
                                _ => crate::terrarium::packet::GenotypePacketEcology::Decomposer,
                            };
                            visual = crate::terrarium::visual_projection::soil_surface_visual_response_from_packet_ecology(
                                ecology,
                                local_air,
                                chemistry,
                                self.moisture.get(idx).copied().unwrap_or(0.0),
                                self.organic_matter.get(idx).copied().unwrap_or(0.0),
                                cohort.smoothed_energy,
                                cohort.identity.catalog.local_bank_share.clamp(0.0, 1.0),
                                cohort.represented_cells,
                                shoreline_signal,
                            );
                        }
                    }
                    Some(SoilOwnershipClass::GenotypePacketRegion { .. }) => {
                        if let Some(pop) = self.packet_population_at(x, y) {
                            if let Some(ecology) = pop.dominant_ecology() {
                                visual = crate::terrarium::visual_projection::soil_surface_visual_response_from_packet_ecology(
                                    ecology,
                                    local_air,
                                    chemistry,
                                    self.moisture.get(idx).copied().unwrap_or(0.0),
                                    self.organic_matter.get(idx).copied().unwrap_or(0.0),
                                    pop.mean_activity(),
                                    pop.ecology_diversity(),
                                    pop.total_cells,
                                    shoreline_signal,
                                );
                            }
                        }
                    }
                    _ => {}
                }
                visuals[idx] = visual;
                signal[idx] = match visual.class {
                    crate::terrarium::visual_projection::TerrariumSoilSurfaceClass::Mineral => 0.0,
                    crate::terrarium::visual_projection::TerrariumSoilSurfaceClass::Humus => {
                        (visual.density - 0.28).max(0.0) * 0.6
                    }
                    crate::terrarium::visual_projection::TerrariumSoilSurfaceClass::WetDetritus => {
                        (visual.density - 0.24).max(0.0) * 0.8
                    }
                    crate::terrarium::visual_projection::TerrariumSoilSurfaceClass::MicrobialMat => {
                        visual.density * 1.2
                    }
                    crate::terrarium::visual_projection::TerrariumSoilSurfaceClass::NitrifierCrust => {
                        visual.density * 1.24
                    }
                    crate::terrarium::visual_projection::TerrariumSoilSurfaceClass::DenitrifierFilm => {
                        visual.density * 1.28
                    }
                    crate::terrarium::visual_projection::TerrariumSoilSurfaceClass::MycorrhizalPatch => {
                        visual.density * 1.1
                    }
                    crate::terrarium::visual_projection::TerrariumSoilSurfaceClass::EarthwormCast => {
                        visual.density * 1.3
                    }
                    crate::terrarium::visual_projection::TerrariumSoilSurfaceClass::NematodeBloom => {
                        visual.density * 1.15
                    }
                };
            }
        }

        select_hotspots(&signal, width, 0.22, 18)
            .into_iter()
            .map(|idx| (idx % width, idx / width, visuals[idx]))
            .collect()
    }

    pub fn earthworm_visual_markers(
        &self,
    ) -> Vec<(
        usize,
        usize,
        crate::terrarium::visual_projection::TerrariumEarthwormVisualResponse,
    )> {
        self.earthworm_visual_markers_at_time(self.time_s)
    }

    pub fn earthworm_visual_markers_at_time(
        &self,
        time_s: f32,
    ) -> Vec<(
        usize,
        usize,
        crate::terrarium::visual_projection::TerrariumEarthwormVisualResponse,
    )> {
        let atmosphere = self.atmosphere_frame();
        let width = self.config.width;
        let height = self.config.height;
        let density = &self.earthworm_population.population_density;
        let max_density = density.iter().copied().fold(0.0f32, f32::max);
        if max_density <= 0.0 {
            return Vec::new();
        }

        select_hotspots(density, width, max_density * 0.45, 4)
            .into_iter()
            .map(|idx| {
                let x = idx % width;
                let y = idx / width;
                let local_air = crate::terrarium::visual_projection::sample_visual_air(
                    &atmosphere,
                    width,
                    height,
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                );
                (
                    x,
                    y,
                    crate::terrarium::visual_projection::earthworm_visual_response(
                        local_air,
                        density.get(idx).copied().unwrap_or(0.0),
                        self.earthworm_population
                            .biomass_per_voxel
                            .get(idx)
                            .copied()
                            .unwrap_or(0.0),
                        self.earthworm_population
                            .bioturbation_rate
                            .get(idx)
                            .copied()
                            .unwrap_or(0.0),
                        time_s,
                        idx as f32,
                    ),
                )
            })
            .collect()
    }

    pub fn nematode_visual_markers(
        &self,
    ) -> Vec<(
        usize,
        usize,
        crate::terrarium::visual_projection::TerrariumNematodeVisualResponse,
    )> {
        self.nematode_visual_markers_at_time(self.time_s)
    }

    pub fn nematode_visual_markers_at_time(
        &self,
        time_s: f32,
    ) -> Vec<(
        usize,
        usize,
        crate::terrarium::visual_projection::TerrariumNematodeVisualResponse,
    )> {
        let atmosphere = self.atmosphere_frame();
        let width = self.config.width;
        let height = self.config.height;
        let mut markers = Vec::new();

        for (guild_idx, guild) in self.nematode_guilds.iter().enumerate() {
            let max_density = guild
                .population_density
                .iter()
                .copied()
                .fold(0.0f32, f32::max);
            if max_density <= 0.0 {
                continue;
            }

            for idx in select_hotspots(&guild.population_density, width, max_density * 0.55, 2) {
                let x = idx % width;
                let y = idx / width;
                let local_air = crate::terrarium::visual_projection::sample_visual_air(
                    &atmosphere,
                    width,
                    height,
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                );
                markers.push((
                    x,
                    y,
                    crate::terrarium::visual_projection::nematode_visual_response(
                        guild.kind,
                        local_air,
                        guild.population_density.get(idx).copied().unwrap_or(0.0),
                        guild.biomass_per_voxel.get(idx).copied().unwrap_or(0.0),
                        time_s,
                        (idx + guild_idx * width * height) as f32,
                    ),
                ));
            }
        }

        markers
    }
}
