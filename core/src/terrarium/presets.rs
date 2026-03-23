use std::fmt;

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerrariumDemoPreset {
    #[default]
    Demo,
    MicroTerrarium,
    MicroAquarium,
}

impl TerrariumDemoPreset {
    /// Pair this preset with a climate scenario for multi-decade evolution runs.
    ///
    /// Returns the preset world config and the climate scenario, ready for
    /// `TerrariumWorld::demo_preset_with_climate()`.
    pub fn with_climate(
        self,
        scenario: ClimateScenario,
    ) -> (Self, ClimateScenario) {
        (self, scenario)
    }

    /// Demo preset with RCP 8.5 (business-as-usual warming).
    ///
    /// Showcases dramatic emergent temperature response: fly metabolism slows in
    /// winter, seed dormancy tracks thermal time, plant VOC emission shifts.
    pub fn warming_demo() -> (TerrariumDemoPreset, ClimateScenario) {
        (Self::Demo, ClimateScenario::Rcp85)
    }

    /// Demo preset with pre-industrial climate (stable baseline).
    ///
    /// Control condition: 13.5 C mean, 280 ppm CO₂, no warming trend.
    pub fn stable_demo() -> (TerrariumDemoPreset, ClimateScenario) {
        (Self::Demo, ClimateScenario::PreIndustrial)
    }

    /// Demo preset with RCP 4.5 (intermediate pathway).
    pub fn moderate_warming_demo() -> (TerrariumDemoPreset, ClimateScenario) {
        (Self::Demo, ClimateScenario::Rcp45)
    }

    pub fn parse(name: &str) -> Option<Self> {
        let normalized = name
            .trim()
            .to_ascii_lowercase()
            .replace('_', "-")
            .replace(' ', "-");
        match normalized.as_str() {
            "demo" | "default" | "native" => Some(Self::Demo),
            "terrarium" | "micro-terrarium" | "microterrarium" | "microcosm" | "rhizosphere" => {
                Some(Self::MicroTerrarium)
            }
            "aquarium" | "micro-aquarium" | "microaquarium" | "tidepool" | "micro-ocean" => {
                Some(Self::MicroAquarium)
            }
            _ => None,
        }
    }

    pub fn cli_name(self) -> &'static str {
        match self {
            Self::Demo => "demo",
            Self::MicroTerrarium => "terrarium",
            Self::MicroAquarium => "aquarium",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Demo => "Native Demo",
            Self::MicroTerrarium => "Micro Terrarium",
            Self::MicroAquarium => "Micro Aquarium",
        }
    }

    pub fn infer_from_config(config: &TerrariumWorldConfig) -> Option<Self> {
        [Self::MicroTerrarium, Self::MicroAquarium, Self::Demo]
            .into_iter()
            .find(|preset| preset.matches_config(config))
    }

    fn matches_config(self, config: &TerrariumWorldConfig) -> bool {
        let expected = self.world_config(config.seed, config.use_gpu_substrate);
        expected.width == config.width
            && expected.height == config.height
            && expected.depth == config.depth
            && expected.substeps == config.substeps
            && expected.max_plants == config.max_plants
            && expected.max_fruits == config.max_fruits
            && expected.max_seeds == config.max_seeds
            && expected.max_explicit_microbes == config.max_explicit_microbes
            && same_f32(expected.cell_size_mm, config.cell_size_mm)
            && same_f32(expected.world_dt_s, config.world_dt_s)
            && same_f32(expected.substrate_dt_ms, config.substrate_dt_ms)
            && same_f32(expected.time_warp, config.time_warp)
    }

    fn world_config(self, seed: u64, use_gpu_substrate: bool) -> TerrariumWorldConfig {
        match self {
            Self::Demo => {
                let mut config = TerrariumWorldConfig::default();
                config.seed = seed;
                config.use_gpu_substrate = use_gpu_substrate;
                config
            }
            Self::MicroTerrarium => TerrariumWorldConfig {
                width: 8,
                height: 8,
                depth: 4,
                cell_size_mm: 0.18,
                seed,
                use_gpu_substrate,
                world_dt_s: 0.04,
                substrate_dt_ms: 10.0,
                substeps: 3,
                time_warp: 120.0,
                max_plants: 4,
                max_fruits: 6,
                max_seeds: 12,
                max_explicit_microbes: 12,
                base_wind_speed_mm_s: 0.2,
                turbulence_intensity: 0.10,
                latitude_deg: 42.0,
                visual_emergence_blend: 1.0,
            },
            Self::MicroAquarium => TerrariumWorldConfig {
                width: 8,
                height: 8,
                depth: 6,
                cell_size_mm: 0.28,
                seed,
                use_gpu_substrate,
                world_dt_s: 0.04,
                substrate_dt_ms: 8.0,
                substeps: 3,
                time_warp: 240.0,
                max_plants: 4,
                max_fruits: 6,
                max_seeds: 10,
                max_explicit_microbes: 16,
                base_wind_speed_mm_s: 0.1,  // sheltered aquarium
                turbulence_intensity: 0.08,
                latitude_deg: 42.0,
                visual_emergence_blend: 1.0,
            },
        }
    }
}

fn same_f32(lhs: f32, rhs: f32) -> bool {
    (lhs - rhs).abs() <= 1.0e-6
}

impl fmt::Display for TerrariumDemoPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl TerrariumWorld {
    pub fn demo_preset(
        seed: u64,
        use_gpu_substrate: bool,
        preset: TerrariumDemoPreset,
    ) -> Result<Self, String> {
        let mut world = Self::new(preset.world_config(seed, use_gpu_substrate))?;
        match preset {
            TerrariumDemoPreset::Demo => world.seed_default_demo_layout(seed)?,
            TerrariumDemoPreset::MicroTerrarium => seed_micro_terrarium(&mut world)?,
            TerrariumDemoPreset::MicroAquarium => seed_micro_aquarium(&mut world)?,
        }
        Ok(world)
    }

    /// Create a demo preset with an attached climate scenario.
    ///
    /// The climate driver is enabled immediately at `start_year` (default 2025).
    /// All temperature-sensitive rates (fly metabolism, seed dormancy, plant
    /// metabolome) respond to the climate trajectory via Eyring TST.
    pub fn demo_preset_with_climate(
        seed: u64,
        use_gpu_substrate: bool,
        preset: TerrariumDemoPreset,
        scenario: ClimateScenario,
        start_year: Option<f64>,
    ) -> Result<Self, String> {
        let mut world = Self::demo_preset(seed, use_gpu_substrate, preset)?;
        world.enable_climate_driver(scenario, seed, start_year.or(Some(2025.0)));
        Ok(world)
    }
}

fn seed_micro_terrarium(world: &mut TerrariumWorld) -> Result<(), String> {
    let width = world.config.width;
    let height = world.config.height;

    sculpt_micro_terrarium_floor(world);
    world.time_s = 43_200.0;
    world.temperature.fill(20.5);
    // Don't override humidity — let it emerge from Priestley-Taylor equilibrium
    // (set during world construction from temperature + soil moisture).
    // Hardcoding 0.76 caused immediate rain feedback loop.
    world.earthworm_population = EarthwormPopulation::new(width, height, &world.organic_matter);
    world.nematode_guilds = crate::soil_fauna::default_nematode_guilds(width, height);

    for &(x, y, volume, emission) in &[
        (2usize, 5usize, 28.0f32, 0.00012f32),
        (5usize, 2usize, 20.0f32, 0.00008f32),
    ] {
        // Carve a depression in the terrain for water to pool in.
        // Larger volume = deeper/wider pond. This is physical —
        // water bodies exist in terrain depressions formed by erosion.
        let depth = (volume / 140.0).clamp(0.1, 0.5);
        let radius = if volume >= 60.0 { 2 } else { 1 };
        for dy in -(radius as i32)..=(radius as i32) {
            for dx in -(radius as i32)..=(radius as i32) {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt() / radius as f32;
                    if dist <= 1.2 {
                        let idx = ny as usize * width + nx as usize;
                        // Gaussian depression: deepest at center, shallow at edges
                        let depression = depth * (-dist * dist * 2.0).exp();
                        world.soil_structure[idx] = (world.soil_structure[idx] - depression).max(0.02);
                    }
                }
            }
        }
        world.add_water(x, y, volume, emission);
    }

    let grass = crate::terrarium::plant_species::genome_for_taxonomy(15368, &mut world.rng);
    let herb = crate::terrarium::plant_species::genome_for_taxonomy(3702, &mut world.rng);
    let w = width;
    let h = height;

    // Pre-compute mutations to avoid borrow conflicts
    let grass_b = grass.mutate(&mut world.rng);
    let grass_c = grass.mutate(&mut world.rng);

    // A few small grass seedlings
    let _ = world.add_plant(w / 3, h / 2, Some(grass.clone()), Some(0.18));
    let _ = world.add_plant(w * 2 / 3, h / 3, Some(grass_b), Some(0.14));

    // Seeds for future growth
    world.add_seed(3.0, 5.0, grass.clone(), Some(0.16), Some(600.0));
    world.add_seed(5.0, 2.0, herb, Some(0.10), Some(2400.0));
    world.add_seed(1.0, 6.0, grass_c, Some(0.12), Some(4800.0));

    let wf = w as f32;
    let hf = h as f32;
    for &(cx, cy, radius, strength) in &[
        (wf * 0.17, hf * 0.67, 2.8f32, 1.0f32),
        (wf * 0.75, hf * 0.25, 2.5f32, 0.88f32),
        (wf * 0.50, hf * 0.50, 2.0f32, 0.72f32),
        (wf * 0.33, hf * 0.21, 1.4f32, 0.58f32),
        (wf * 0.70, hf * 0.68, 1.5f32, 0.62f32),
    ] {
        paint_surface_blend(&mut world.moisture, width, height, cx, cy, radius, 0.42);
        paint_surface_blend(
            &mut world.deep_moisture,
            width,
            height,
            cx,
            cy,
            radius,
            0.46,
        );
        paint_surface_blend(
            &mut world.organic_matter,
            width,
            height,
            cx,
            cy,
            radius,
            0.84,
        );
        paint_surface_blend(
            &mut world.microbial_biomass,
            width,
            height,
            cx,
            cy,
            radius,
            0.52 + strength * 0.08,
        );
        paint_surface_blend(
            &mut world.symbiont_biomass,
            width,
            height,
            cx,
            cy,
            radius,
            0.20 + strength * 0.06,
        );
        paint_surface_blend(
            &mut world.soil_structure,
            width,
            height,
            cx,
            cy,
            radius,
            0.28 + strength * 0.10,
        );
        paint_surface_add(
            &mut world.root_exudates,
            width,
            height,
            cx,
            cy,
            radius,
            0.18 * strength,
        );
        paint_surface_blend(
            &mut world.shallow_nutrients,
            width,
            height,
            cx,
            cy,
            radius,
            0.12,
        );
        paint_surface_blend(
            &mut world.deep_minerals,
            width,
            height,
            cx,
            cy,
            radius,
            0.11,
        );
        paint_surface_blend(
            &mut world.dissolved_nutrients,
            width,
            height,
            cx,
            cy,
            radius,
            0.10,
        );
        paint_surface_blend(
            &mut world.mineral_nitrogen,
            width,
            height,
            cx,
            cy,
            radius,
            0.06,
        );
        paint_surface_add(
            &mut world.litter_carbon,
            width,
            height,
            cx,
            cy,
            radius,
            0.07 * strength,
        );
        paint_rhizosphere_patch(world, cx, cy, radius, strength);
    }

    let probe_a = crate::enzyme_probes::select_enzyme_for_seed(world.config.seed);
    world.spawn_probe(&probe_a, width / 4, height.saturating_sub(2), 1)?;
    let probe_b = crate::enzyme_probes::select_enzyme_for_seed(world.config.seed.wrapping_add(17));
    world.spawn_probe(&probe_b, width * 3 / 4, height / 4, 1)?;
    let probe_c = crate::enzyme_probes::select_enzyme_for_seed(world.config.seed.wrapping_add(53));
    world.spawn_probe(&probe_c, width / 2, height / 2, 2)?;
    seed_mineral_stratigraphy(world, false);
    world.rebuild_water_mask();
    Ok(())
}

fn seed_micro_aquarium(world: &mut TerrariumWorld) -> Result<(), String> {
    let width = world.config.width;
    let height = world.config.height;

    world.time_s = 43_200.0;
    world.temperature.fill(19.0);
    world.humidity.fill(0.99);
    world.earthworm_population = EarthwormPopulation::empty(width, height);
    world.nematode_guilds = crate::soil_fauna::default_nematode_guilds(width, height);
    sculpt_micro_aquarium_basin(world);

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let nx = normalized_axis(x, width);
            let ny = normalized_axis(y, height);
            let radial = (nx * nx + ny * ny).sqrt().clamp(0.0, 1.0);
            let basin_t = (1.0 - radial).clamp(0.0, 1.0);
            let volume = 180.0 + basin_t * 260.0;
            let emission = 0.000012 + basin_t * 0.000008;
            if basin_t > 0.08 {
                world.add_water(x, y, volume, emission);
            }
        }
    }
    world.add_water(width / 2, height / 2, 460.0, 0.000008);

    let duckweed = crate::terrarium::plant_species::genome_for_taxonomy(4472, &mut world.rng);
    let coontail = crate::terrarium::plant_species::genome_for_taxonomy(4428, &mut world.rng);
    let waterweed = crate::terrarium::plant_species::genome_for_taxonomy(100364, &mut world.rng);
    let coontail_edge = coontail.mutate(&mut world.rng);
    let waterweed_edge = waterweed.mutate(&mut world.rng);
    let _ = world.add_plant(
        2,
        height.saturating_sub(3),
        Some(waterweed.clone()),
        Some(0.30),
    );
    let _ = world.add_plant(
        width.saturating_sub(3),
        2,
        Some(coontail.clone()),
        Some(0.34),
    );
    let _ = world.add_plant(
        width / 2,
        height / 2 + 1,
        Some(duckweed.clone()),
        Some(0.22),
    );
    let _ = world.add_plant(
        width / 2 + 1,
        height / 2,
        Some(duckweed.clone()),
        Some(0.20),
    );
    let _ = world.add_plant(width / 2 - 2, 2, Some(coontail_edge), Some(0.31));
    let _ = world.add_plant(
        width.saturating_sub(4),
        height.saturating_sub(3),
        Some(waterweed_edge),
        Some(0.36),
    );

    seed_local_seed_bank(
        world,
        &waterweed,
        &[(2, height.saturating_sub(4)), (width / 2, height / 2 + 2)],
        0.18,
        900.0,
        2200.0,
    );
    seed_local_seed_bank(
        world,
        &coontail,
        &[(width.saturating_sub(3), 2), (width / 2 - 2, 2)],
        0.16,
        1100.0,
        2400.0,
    );
    seed_local_seed_bank(
        world,
        &duckweed,
        &[(width / 2, height / 2), (width / 2 + 1, height / 2)],
        0.10,
        800.0,
        1800.0,
    );

    fill_surface(world, 0.94, 0.98, 0.10, 0.07, 0.18, 0.42, 0.06);
    fill_aquarium_substrate(
        world, 0.98, 2.20, 0.20, 0.24, 0.10, 0.08, 0.02, 0.03, 0.016, 0.012, 0.014,
    );

    for &(cx, cy, radius, strength) in &[
        (width as f32 * 0.5, height as f32 * 0.5, 4.4, 1.0),
        (2.5, height as f32 * 0.6, 2.7, 0.72),
        (width as f32 - 3.0, height as f32 * 0.35, 2.5, 0.66),
        (width as f32 * 0.32, height as f32 * 0.28, 1.8, 0.52),
        (width as f32 * 0.70, height as f32 * 0.72, 1.6, 0.48),
    ] {
        paint_surface_blend(
            &mut world.organic_matter,
            width,
            height,
            cx,
            cy,
            radius,
            0.92,
        );
        paint_surface_blend(
            &mut world.microbial_biomass,
            width,
            height,
            cx,
            cy,
            radius,
            0.66 + strength * 0.08,
        );
        paint_surface_blend(
            &mut world.symbiont_biomass,
            width,
            height,
            cx,
            cy,
            radius,
            0.30 + strength * 0.10,
        );
        paint_surface_add(
            &mut world.litter_carbon,
            width,
            height,
            cx,
            cy,
            radius,
            0.11 * strength,
        );
        paint_surface_add(
            &mut world.root_exudates,
            width,
            height,
            cx,
            cy,
            radius,
            0.05 * strength,
        );
        paint_surface_blend(
            &mut world.dissolved_nutrients,
            width,
            height,
            cx,
            cy,
            radius,
            0.16,
        );
        paint_surface_blend(
            &mut world.mineral_nitrogen,
            width,
            height,
            cx,
            cy,
            radius,
            0.09,
        );
        paint_biofilm_patch(world, cx, cy, radius, strength);
    }

    for (guild_idx, guild) in world.nematode_guilds.iter_mut().enumerate() {
        let biomass_ratio = guild
            .population_density
            .first()
            .copied()
            .filter(|density| *density > 0.0)
            .map(|density| guild.biomass_per_voxel[0] / density)
            .unwrap_or(0.05);
        let hotspots: &[(f32, f32, f32, f32)] = match guild_idx {
            0 => &[
                (width as f32 * 0.50, height as f32 * 0.52, 3.6, 10.0),
                (2.5, height as f32 * 0.62, 1.9, 6.4),
            ],
            1 => &[
                (width as f32 * 0.72, height as f32 * 0.70, 2.0, 4.8),
                (width as f32 * 0.30, height as f32 * 0.28, 1.7, 3.8),
            ],
            _ => &[
                (width as f32 * 0.62, height as f32 * 0.44, 2.3, 3.4),
                (width as f32 * 0.44, height as f32 * 0.64, 1.8, 2.8),
            ],
        };
        for &(cx, cy, radius, amount) in hotspots {
            paint_surface_add(
                &mut guild.population_density,
                width,
                height,
                cx,
                cy,
                radius,
                amount,
            );
            paint_surface_add(
                &mut guild.biomass_per_voxel,
                width,
                height,
                cx,
                cy,
                radius,
                amount * biomass_ratio,
            );
        }
    }

    let probe_a = crate::enzyme_probes::select_enzyme_for_seed(world.config.seed.wrapping_add(101));
    world.spawn_probe(&probe_a, width / 2, height / 2, 2)?;
    let probe_b = crate::enzyme_probes::select_enzyme_for_seed(world.config.seed.wrapping_add(203));
    world.spawn_probe(&probe_b, width / 2 + 1, height / 2, 3)?;
    let probe_c = crate::enzyme_probes::select_enzyme_for_seed(world.config.seed.wrapping_add(307));
    world.spawn_probe(&probe_c, width / 2 - 1, height / 2, 2)?;
    seed_mineral_stratigraphy(world, true);
    world.rebuild_water_mask();
    Ok(())
}

fn sculpt_micro_terrarium_floor(world: &mut TerrariumWorld) {
    let width = world.config.width;
    let height = world.config.height;
    for y in 0..height {
        for x in 0..width {
            let idx = idx2_local(width, x, y);
            let nx = normalized_axis(x, width);
            let ny = normalized_axis(y, height);
            let ridge = (nx * 1.7).sin() * 0.072 + (ny * 2.1).cos() * 0.056;
            let basin = (1.0 - (nx * nx + ny * ny).sqrt()).clamp(0.0, 1.0);
            let pocket = ((x as f32 * 0.91) + (y as f32 * 0.47)).sin() * 0.030
                + ((x as f32 * 0.38) - (y as f32 * 0.73)).cos() * 0.024;
            world.soil_structure[idx] = (0.40 + basin * 0.18 + ridge + pocket).clamp(0.22, 0.84);
        }
    }
}

fn sculpt_micro_aquarium_basin(world: &mut TerrariumWorld) {
    let width = world.config.width;
    let height = world.config.height;
    for y in 0..height {
        for x in 0..width {
            let idx = idx2_local(width, x, y);
            let nx = normalized_axis(x, width);
            let ny = normalized_axis(y, height);
            let radial = (nx * nx + ny * ny).sqrt().clamp(0.0, 1.0);
            let rim_t = radial.powf(1.2);
            let shelf_t = ((radial - 0.42) / 0.50).clamp(0.0, 1.0);
            let ripple = ((x as f32 * 0.63) + (y as f32 * 0.31)).sin() * 0.018
                + ((x as f32 * 0.27) - (y as f32 * 0.52)).cos() * 0.012;
            world.soil_structure[idx] =
                (0.10 + rim_t * 0.34 + shelf_t * 0.18 + ripple).clamp(0.05, 0.82);
        }
    }
}

fn fill_surface(
    world: &mut TerrariumWorld,
    moisture: f32,
    deep_moisture: f32,
    nutrients: f32,
    mineral_nitrogen: f32,
    microbial_biomass: f32,
    organic_matter: f32,
    litter_carbon: f32,
) {
    world.moisture.fill(moisture);
    world.deep_moisture.fill(deep_moisture);
    world.dissolved_nutrients.fill(nutrients);
    world.shallow_nutrients.fill(nutrients);
    world.deep_minerals.fill(nutrients * 0.92);
    world.mineral_nitrogen.fill(mineral_nitrogen);
    world.microbial_biomass.fill(microbial_biomass);
    world.organic_matter.fill(organic_matter);
    world.litter_carbon.fill(litter_carbon);
}

fn seed_mineral_stratigraphy(world: &mut TerrariumWorld, aquatic: bool) {
    let width = world.config.width;
    let height = world.config.height;
    let depth = world.config.depth.max(1);
    let total_voxels = world.substrate.total_voxels();

    for z in 0..depth {
        let depth_t = if depth <= 1 {
            0.0
        } else {
            z as f32 / (depth - 1) as f32
        };
        for y in 0..height {
            for x in 0..width {
                let cell = idx2_local(width, x, y);
                let voxel = idx3_local(width, height, x, y, z);
                let texture = world.soil_structure[cell].clamp(0.0, 1.0);
                let organic = world.organic_matter[cell].clamp(0.0, 1.0);
                let moisture = lerp(
                    world.moisture[cell].clamp(0.0, 1.0),
                    world.deep_moisture[cell].clamp(0.0, 1.0),
                    depth_t,
                );
                let absorbency =
                    crate::soil_broad::soil_texture_absorbency(texture, world.organic_matter[cell]);
                let retention =
                    crate::soil_broad::soil_texture_retention(texture, world.organic_matter[cell]);
                let capillarity = crate::soil_broad::soil_texture_capillarity(
                    texture,
                    world.organic_matter[cell],
                );
                let coarse_fraction = clamp(1.0 - texture * 0.92, 0.0, 1.0);
                let fine_fraction = clamp(texture * 0.90 + retention * 0.24, 0.0, 1.0);
                let drainage = clamp(
                    1.04 - moisture * 0.72 + absorbency * 0.18 - capillarity * 0.10,
                    0.02,
                    1.20,
                );
                let oxidation = clamp(drainage * (0.48 + (1.0 - depth_t) * 0.34), 0.0, 1.15);
                let rim_bias = if aquatic {
                    clamp(texture * 0.82 + drainage * 0.18, 0.0, 1.0)
                } else {
                    clamp(drainage * 0.60 + (1.0 - moisture) * 0.22, 0.0, 1.0)
                };

                let silicate = clamp(
                    0.36 + coarse_fraction * 0.34 + depth_t * 0.28 + rim_bias * 0.10
                        - organic * 0.08,
                    0.10,
                    1.35,
                );
                let clay = clamp(
                    0.08 + fine_fraction * 0.30 + depth_t * 0.22 + moisture * 0.10
                        - absorbency * 0.06,
                    0.02,
                    1.10,
                );
                let carbonate = clamp(
                    0.01 + depth_t * 0.14
                        + drainage * 0.08
                        + rim_bias * if aquatic { 0.02 } else { 0.05 }
                        - organic * 0.07
                        - moisture * if aquatic { 0.09 } else { 0.04 },
                    0.0,
                    if aquatic { 0.18 } else { 0.28 },
                );
                let iron_oxide = clamp(
                    0.02 + oxidation * 0.12 + rim_bias * 0.06
                        - moisture * if aquatic { 0.08 } else { 0.03 },
                    0.0,
                    0.28,
                );
                let dissolved_silicate = clamp(
                    0.006 + silicate * 0.028 + clay * 0.012 + moisture * 0.006,
                    0.0,
                    0.10,
                );
                let bicarbonate = clamp(
                    0.004
                        + carbonate * 0.10
                        + moisture * 0.018
                        + world.substrate.current
                            [TerrariumSpecies::CarbonDioxide as usize * total_voxels + voxel]
                            * if aquatic { 0.16 } else { 0.08 },
                    0.0,
                    0.18,
                );
                let exchangeable_calcium = clamp(
                    0.006
                        + carbonate * 0.32
                        + clay * 0.03
                        + if aquatic { 0.0 } else { drainage * 0.02 },
                    0.0,
                    0.18,
                );
                let exchangeable_magnesium =
                    clamp(0.005 + silicate * 0.040 + clay * 0.032, 0.0, 0.16);
                let exchangeable_potassium =
                    clamp(0.004 + silicate * 0.030 + organic * 0.010, 0.0, 0.14);
                let exchangeable_sodium = clamp(
                    0.003 + silicate * 0.020 + if aquatic { moisture * 0.012 } else { 0.0 },
                    0.0,
                    0.12,
                );
                let exchangeable_aluminum = clamp(
                    0.002 + clay * 0.040 + (1.0 - drainage) * 0.020 + moisture * 0.010,
                    0.0,
                    0.16,
                );
                let aqueous_iron = clamp(
                    0.002 + iron_oxide * 0.030 + (1.0 - oxidation) * 0.008,
                    0.0,
                    0.10,
                );

                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::SilicateMineral,
                    silicate,
                    1.0,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::ClayMineral,
                    clay,
                    1.0,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::CarbonateMineral,
                    carbonate,
                    1.0,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::IronOxideMineral,
                    iron_oxide,
                    1.0,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::DissolvedSilicate,
                    dissolved_silicate,
                    1.0,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::BicarbonatePool,
                    bicarbonate,
                    1.0,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::ExchangeableCalcium,
                    exchangeable_calcium,
                    1.0,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::ExchangeableMagnesium,
                    exchangeable_magnesium,
                    1.0,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::ExchangeablePotassium,
                    exchangeable_potassium,
                    1.0,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::ExchangeableSodium,
                    exchangeable_sodium,
                    1.0,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::ExchangeableAluminum,
                    exchangeable_aluminum,
                    1.0,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::AqueousIronPool,
                    aqueous_iron,
                    1.0,
                );
            }
        }
    }

    world
        .substrate
        .next
        .copy_from_slice(&world.substrate.current);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn fill_aquarium_substrate(
    world: &mut TerrariumWorld,
    hydration: f32,
    water: f32,
    oxygen_gas: f32,
    carbon_dioxide: f32,
    ammonium: f32,
    nitrate: f32,
    glucose: f32,
    atp_flux: f32,
    amino_acids: f32,
    nucleotides: f32,
    membrane_precursors: f32,
) {
    let width = world.config.width;
    let height = world.config.height;
    let depth = world.config.depth.max(1);
    let total_voxels = world.substrate.total_voxels();
    for z in 0..depth {
        let depth_t = if depth <= 1 {
            0.0
        } else {
            z as f32 / (depth - 1) as f32
        };
        for y in 0..height {
            for x in 0..width {
                let voxel = idx3_local(width, height, x, y, z);
                world.substrate.hydration[voxel] = hydration - depth_t * 0.12;
                world.substrate.current
                    [species_slot(TerrariumSpecies::Water, total_voxels, voxel)] =
                    water - depth_t * 0.18;
                world.substrate.current
                    [species_slot(TerrariumSpecies::OxygenGas, total_voxels, voxel)] =
                    oxygen_gas - depth_t * 0.10;
                world.substrate.current
                    [species_slot(TerrariumSpecies::CarbonDioxide, total_voxels, voxel)] =
                    carbon_dioxide + depth_t * 0.08;
                world.substrate.current
                    [species_slot(TerrariumSpecies::Ammonium, total_voxels, voxel)] =
                    ammonium + depth_t * 0.03;
                world.substrate.current
                    [species_slot(TerrariumSpecies::Nitrate, total_voxels, voxel)] =
                    nitrate - depth_t * 0.02;
                world.substrate.current
                    [species_slot(TerrariumSpecies::Glucose, total_voxels, voxel)] =
                    glucose * (1.0 - depth_t * 0.35);
                world.substrate.current
                    [species_slot(TerrariumSpecies::AtpFlux, total_voxels, voxel)] =
                    atp_flux * (1.0 - depth_t * 0.15);
                world.substrate.current
                    [species_slot(TerrariumSpecies::AminoAcidPool, total_voxels, voxel)] =
                    amino_acids * (1.0 - depth_t * 0.22);
                world.substrate.current
                    [species_slot(TerrariumSpecies::NucleotidePool, total_voxels, voxel)] =
                    nucleotides * (1.0 - depth_t * 0.16);
                world.substrate.current
                    [species_slot(TerrariumSpecies::MembranePrecursorPool, total_voxels, voxel)] =
                    membrane_precursors * (1.0 - depth_t * 0.18);
            }
        }
    }
}

fn paint_rhizosphere_patch(
    world: &mut TerrariumWorld,
    cx: f32,
    cy: f32,
    radius: f32,
    strength: f32,
) {
    let width = world.config.width;
    let height = world.config.height;
    let depth = world.config.depth.max(1);
    let total_voxels = world.substrate.total_voxels();
    for z in 0..depth {
        let depth_t = if depth <= 1 {
            0.0
        } else {
            z as f32 / (depth - 1) as f32
        };
        let shallow_weight = 1.0 - depth_t * 0.65;
        for y in 0..height {
            for x in 0..width {
                let weight = radial_weight(x, y, cx, cy, radius) * shallow_weight * strength;
                if weight <= 0.0 {
                    continue;
                }
                let voxel = idx3_local(width, height, x, y, z);
                let hydration_target = 0.68 - depth_t * 0.14;
                blend_scalar(
                    &mut world.substrate.hydration[voxel],
                    hydration_target,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::Water,
                    0.94 - depth_t * 0.18,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::Glucose,
                    0.08 * (1.0 - depth_t * 0.5),
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::OxygenGas,
                    0.50 - depth_t * 0.18,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::Ammonium,
                    0.07 + depth_t * 0.02,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::Nitrate,
                    0.05 + depth_t * 0.05,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::CarbonDioxide,
                    0.10 + depth_t * 0.04,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::AtpFlux,
                    0.018 + weight * 0.03,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::AminoAcidPool,
                    0.018 + shallow_weight * 0.010,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::NucleotidePool,
                    0.010 + (1.0 - depth_t) * 0.006,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::MembranePrecursorPool,
                    0.008 + shallow_weight * 0.007,
                    weight,
                );
            }
        }
    }
}

fn paint_biofilm_patch(world: &mut TerrariumWorld, cx: f32, cy: f32, radius: f32, strength: f32) {
    let width = world.config.width;
    let height = world.config.height;
    let depth = world.config.depth.max(1);
    let total_voxels = world.substrate.total_voxels();
    for z in 0..depth {
        let depth_t = if depth <= 1 {
            0.0
        } else {
            z as f32 / (depth - 1) as f32
        };
        for y in 0..height {
            for x in 0..width {
                let planar = radial_weight(x, y, cx, cy, radius);
                if planar <= 0.0 {
                    continue;
                }
                let layer_weight = (1.0 - (depth_t - 0.55).abs() * 1.6).clamp(0.0, 1.0);
                let weight = planar * layer_weight * strength;
                if weight <= 0.0 {
                    continue;
                }
                let voxel = idx3_local(width, height, x, y, z);
                blend_scalar(
                    &mut world.substrate.hydration[voxel],
                    0.82 - depth_t * 0.08,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::Water,
                    1.70 - depth_t * 0.16,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::Glucose,
                    0.10 * (1.0 - depth_t * 0.25),
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::OxygenGas,
                    0.22 - depth_t * 0.06,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::Ammonium,
                    0.09 + depth_t * 0.03,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::Nitrate,
                    0.07 - depth_t * 0.02,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::CarbonDioxide,
                    0.22 + depth_t * 0.06,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::AtpFlux,
                    0.05 + weight * 0.04,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::AminoAcidPool,
                    0.014 + layer_weight * 0.010,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::NucleotidePool,
                    0.012 + layer_weight * 0.008,
                    weight,
                );
                blend_substrate_species(
                    &mut world.substrate.current,
                    total_voxels,
                    voxel,
                    TerrariumSpecies::MembranePrecursorPool,
                    0.016 + layer_weight * 0.014,
                    weight,
                );
            }
        }
    }
}

fn seed_local_seed_bank(
    world: &mut TerrariumWorld,
    template: &TerrariumPlantGenome,
    anchors: &[(usize, usize)],
    jitter: f32,
    dormancy_lo: f32,
    dormancy_hi: f32,
) {
    for &(sx, sy) in anchors {
        let seed_genome = template.mutate(&mut world.rng);
        let offset_x = world.rng.gen_range(-jitter..jitter);
        let offset_y = world.rng.gen_range(-jitter..jitter);
        let dormancy = world.rng.gen_range(dormancy_lo..dormancy_hi);
        world.add_seed(
            sx as f32 + offset_x,
            sy as f32 + offset_y,
            seed_genome,
            None,
            Some(dormancy),
        );
    }
}

fn paint_surface_blend(
    field: &mut [f32],
    width: usize,
    height: usize,
    cx: f32,
    cy: f32,
    radius: f32,
    target: f32,
) {
    for y in 0..height {
        for x in 0..width {
            let weight = radial_weight(x, y, cx, cy, radius);
            if weight > 0.0 {
                let idx = idx2_local(width, x, y);
                blend_scalar(&mut field[idx], target, weight);
            }
        }
    }
}

fn paint_surface_add(
    field: &mut [f32],
    width: usize,
    height: usize,
    cx: f32,
    cy: f32,
    radius: f32,
    amount: f32,
) {
    for y in 0..height {
        for x in 0..width {
            let weight = radial_weight(x, y, cx, cy, radius);
            if weight > 0.0 {
                let idx = idx2_local(width, x, y);
                field[idx] += amount * weight;
            }
        }
    }
}

fn radial_weight(x: usize, y: usize, cx: f32, cy: f32, radius: f32) -> f32 {
    let dx = x as f32 + 0.5 - cx;
    let dy = y as f32 + 0.5 - cy;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist >= radius {
        0.0
    } else {
        1.0 - dist / radius.max(1.0e-3)
    }
}

fn blend_scalar(slot: &mut f32, target: f32, weight: f32) {
    let weight = weight.clamp(0.0, 1.0);
    *slot += (target - *slot) * weight;
}

fn blend_substrate_species(
    current: &mut [f32],
    total_voxels: usize,
    voxel: usize,
    species: TerrariumSpecies,
    target: f32,
    weight: f32,
) {
    let idx = species_slot(species, total_voxels, voxel);
    blend_scalar(&mut current[idx], target, weight);
}

fn species_slot(species: TerrariumSpecies, total_voxels: usize, voxel: usize) -> usize {
    species as usize * total_voxels + voxel
}

fn idx2_local(width: usize, x: usize, y: usize) -> usize {
    y * width + x
}

fn idx3_local(width: usize, height: usize, x: usize, y: usize, z: usize) -> usize {
    (z * height + y) * width + x
}

fn normalized_axis(pos: usize, extent: usize) -> f32 {
    if extent <= 1 {
        return 0.0;
    }
    let center = (extent - 1) as f32 * 0.5;
    let span = center.max(1.0);
    ((pos as f32 - center) / span).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_terrarium_preset_builds_without_flies() {
        let world = TerrariumWorld::demo_preset(7, false, TerrariumDemoPreset::MicroTerrarium)
            .expect("micro terrarium preset should build");
        assert_eq!(world.config.width, 8);
        assert_eq!(world.config.height, 8);
        assert_eq!(world.config.depth, 4);
        assert!(
            world.flies.is_empty(),
            "terrarium should start fly-free"
        );
        assert!(
            world.seeds.len() >= 2,
            "terrarium should have seeds"
        );
        assert!(
            !world.atomistic_probes.is_empty(),
            "micro terrarium should seed probes"
        );
    }

    #[test]
    fn micro_aquarium_preset_is_water_dominant() {
        let world = TerrariumWorld::demo_preset(11, false, TerrariumDemoPreset::MicroAquarium)
            .expect("micro aquarium preset should build");
        let mean_hydration = world.substrate.hydration.iter().copied().sum::<f32>()
            / world.substrate.hydration.len().max(1) as f32;
        let mean_surface_water =
            world.water_mask.iter().copied().sum::<f32>() / world.water_mask.len().max(1) as f32;
        assert_eq!(world.config.depth, 6);
        assert!(
            world.flies.is_empty(),
            "micro aquarium should start fly-free"
        );
        assert!(
            world.waters.len() >= 20,
            "micro aquarium should seed a dense water field"
        );
        assert!(
            mean_hydration > 0.85,
            "micro aquarium should start significantly wetter than the default terrarium"
        );
        assert!(
            mean_surface_water > 0.70,
            "micro aquarium should present as a flooded basin, not isolated puddles"
        );
    }

    #[test]
    fn micro_terrarium_preset_seeds_depth_stratified_minerals() {
        let world = TerrariumWorld::demo_preset(19, false, TerrariumDemoPreset::MicroTerrarium)
            .expect("micro terrarium preset should build");
        let top_z = 0usize;
        let bottom_z = world.config.depth.saturating_sub(1);
        let mut top_silicate = 0.0f32;
        let mut bottom_silicate = 0.0f32;
        let mut top_clay = 0.0f32;
        let mut bottom_clay = 0.0f32;
        let mut total = 0usize;
        for y in 0..world.config.height {
            for x in 0..world.config.width {
                top_silicate += world.substrate.patch_mean_species(
                    TerrariumSpecies::SilicateMineral,
                    x,
                    y,
                    top_z,
                    1,
                );
                bottom_silicate += world.substrate.patch_mean_species(
                    TerrariumSpecies::SilicateMineral,
                    x,
                    y,
                    bottom_z,
                    1,
                );
                top_clay += world.substrate.patch_mean_species(
                    TerrariumSpecies::ClayMineral,
                    x,
                    y,
                    top_z,
                    1,
                );
                bottom_clay += world.substrate.patch_mean_species(
                    TerrariumSpecies::ClayMineral,
                    x,
                    y,
                    bottom_z,
                    1,
                );
                total += 1;
            }
        }
        let denom = total.max(1) as f32;
        assert!(bottom_silicate / denom > top_silicate / denom);
        assert!(bottom_clay / denom > top_clay / denom);
        assert!(
            world
                .substrate
                .mean_species(TerrariumSpecies::CarbonateMineral)
                > 0.0
        );
        assert!(
            world
                .substrate
                .mean_species(TerrariumSpecies::IronOxideMineral)
                > 0.0
        );
        assert!(
            world
                .substrate
                .mean_species(TerrariumSpecies::DissolvedSilicate)
                > 0.0
        );
        assert!(
            world
                .substrate
                .mean_species(TerrariumSpecies::BicarbonatePool)
                > 0.0
        );
        assert!(
            world
                .substrate
                .mean_species(TerrariumSpecies::ExchangeableCalcium)
                > 0.0
        );
        assert!(
            world
                .substrate
                .mean_species(TerrariumSpecies::ExchangeableMagnesium)
                > 0.0
        );
        assert!(
            world
                .substrate
                .mean_species(TerrariumSpecies::ExchangeablePotassium)
                > 0.0
        );
        assert!(
            world
                .substrate
                .mean_species(TerrariumSpecies::ExchangeableSodium)
                > 0.0
        );
        assert!(
            world
                .substrate
                .mean_species(TerrariumSpecies::ExchangeableAluminum)
                > 0.0
        );
        assert!(
            world
                .substrate
                .mean_species(TerrariumSpecies::AqueousIronPool)
                > 0.0
        );
    }

    #[test]
    fn climate_preset_with_climate_builds() {
        // Verify demo_preset_with_climate produces a world with an active climate driver.
        let world = TerrariumWorld::demo_preset_with_climate(
            42,
            false,
            TerrariumDemoPreset::Demo,
            ClimateScenario::Rcp85,
            Some(2025.0),
        )
        .expect("climate preset should build");
        assert!(
            world.climate_driver.is_some(),
            "climate driver should be enabled"
        );
        let year = world.current_year().expect("should have current year");
        assert!(
            (year - 2025.0).abs() < 1.0,
            "climate year should be ~2025: {year}"
        );
    }

    #[test]
    fn warming_increases_temperature_field() {
        // RCP 8.5 should drive the temperature field higher than pre-industrial
        // after running the same number of frames.
        let mut warm = TerrariumWorld::demo_preset_with_climate(
            7,
            false,
            TerrariumDemoPreset::Demo,
            ClimateScenario::Rcp85,
            Some(2080.0), // deep into warming scenario
        )
        .expect("warming preset");
        let mut stable = TerrariumWorld::demo_preset_with_climate(
            7,
            false,
            TerrariumDemoPreset::Demo,
            ClimateScenario::PreIndustrial,
            Some(2080.0),
        )
        .expect("stable preset");

        for _ in 0..50 {
            warm.step_frame().ok();
            stable.step_frame().ok();
        }

        let mean_temp_warm =
            warm.temperature.iter().copied().sum::<f32>() / warm.temperature.len().max(1) as f32;
        let mean_temp_stable = stable.temperature.iter().copied().sum::<f32>()
            / stable.temperature.len().max(1) as f32;
        assert!(
            mean_temp_warm > mean_temp_stable,
            "RCP 8.5 at 2080 should be warmer: warm={mean_temp_warm:.1} vs stable={mean_temp_stable:.1}"
        );
    }

    #[test]
    fn elevated_co2_propagates_to_substrate() {
        // RCP 8.5 should have higher dissolved CO₂ in the substrate than pre-industrial.
        let mut warm = TerrariumWorld::demo_preset_with_climate(
            13,
            false,
            TerrariumDemoPreset::Demo,
            ClimateScenario::Rcp85,
            Some(2080.0),
        )
        .expect("warming preset");
        let mut stable = TerrariumWorld::demo_preset_with_climate(
            13,
            false,
            TerrariumDemoPreset::Demo,
            ClimateScenario::PreIndustrial,
            Some(2080.0),
        )
        .expect("stable preset");

        for _ in 0..50 {
            warm.step_frame().ok();
            stable.step_frame().ok();
        }

        // The climate driver relaxes odorant CO₂ channel toward target_co2.
        // Under RCP 8.5 at 2080, atmospheric CO₂ is ~1100 ppm vs 280 ppm.
        let co2_warm: f32 = warm.odorants[3].iter().copied().sum::<f32>()
            / warm.odorants[3].len().max(1) as f32;
        let co2_stable: f32 = stable.odorants[3].iter().copied().sum::<f32>()
            / stable.odorants[3].len().max(1) as f32;
        assert!(
            co2_warm > co2_stable,
            "RCP 8.5 CO₂ should exceed pre-industrial: warm={co2_warm:.4} vs stable={co2_stable:.4}"
        );
    }

    #[test]
    fn convenience_presets_produce_correct_scenarios() {
        let (preset, scenario) = TerrariumDemoPreset::warming_demo();
        assert_eq!(preset, TerrariumDemoPreset::Demo);
        assert!(matches!(scenario, ClimateScenario::Rcp85));

        let (preset, scenario) = TerrariumDemoPreset::stable_demo();
        assert_eq!(preset, TerrariumDemoPreset::Demo);
        assert!(matches!(scenario, ClimateScenario::PreIndustrial));

        let (preset, scenario) = TerrariumDemoPreset::moderate_warming_demo();
        assert_eq!(preset, TerrariumDemoPreset::Demo);
        assert!(matches!(scenario, ClimateScenario::Rcp45));
    }
}
