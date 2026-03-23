use super::material_exchange::{
    deposit_species_to_inventory, inventory_component_amount, material_inventory_target_from_patch,
    spill_inventory_to_patch, sync_inventory_with_patch, withdraw_species_from_inventory,
};
use super::*;

fn fruit_source_traits(genome: &TerrariumPlantGenome) -> (f32, f32, Vec<(usize, f32)>) {
    let canopy_t = clamp(genome.canopy_radius_mm / 8.5, 0.0, 1.0);
    let seed_t = clamp(genome.seed_mass / 0.20, 0.0, 1.0);
    let volatile_t = clamp((genome.volatile_scale - 0.45) / 1.4, 0.0, 1.0);
    let root_t = clamp(genome.root_depth_bias / 1.1, 0.0, 1.0);
    let base_radius = clamp(
        0.34 + canopy_t * 0.78 + seed_t * 0.30 + root_t * 0.14,
        0.28,
        1.70,
    );
    let base_ripeness = clamp(
        0.18 + genome.fruiting_threshold * 0.26 + genome.water_use_efficiency * 0.08,
        0.12,
        0.78,
    );
    let odorant_profile = vec![
        (
            ETHYL_ACETATE_IDX,
            clamp(0.28 + volatile_t * 0.58 + seed_t * 0.12, 0.12, 1.0),
        ),
        (
            GERANIOL_IDX,
            clamp(0.16 + (1.0 - volatile_t) * 0.22 + root_t * 0.28, 0.08, 1.0),
        ),
    ];
    (base_radius, base_ripeness, odorant_profile)
}

fn root_zone_depth_from_bias(depth: usize, depth_bias: f32) -> usize {
    let max_z = depth.saturating_sub(1);
    clamp(depth_bias * max_z as f32, 0.0, max_z as f32).round() as usize
}

fn fruit_surface_inventory_target(
    fruit: &TerrariumFruitPatch,
    moisture: f32,
    deep_moisture: f32,
    humidity: f32,
    microbial_biomass: f32,
) -> RegionalMaterialInventory {
    let mut target = RegionalMaterialInventory::new(format!(
        "fruit-surface-target:{}:{}",
        fruit.source.x, fruit.source.y
    ));
    let water = clamp(
        moisture * 0.18
            + deep_moisture * 0.12
            + humidity * 0.18
            + fruit.composition.water_fraction * (0.50 + fruit.radius * 0.18),
        0.02,
        2.4,
    );
    let glucose = clamp(
        fruit.source.sugar_content * (0.58 + fruit.composition.sugar_fraction * 1.20),
        0.0,
        2.4,
    );
    let amino = clamp(
        fruit.source.sugar_content * (0.08 + fruit.composition.amino_fraction * 2.2),
        0.0,
        1.2,
    );
    let nucleotide = clamp(
        fruit
            .reproduction
            .as_ref()
            .map(|repro| repro.reserve_carbon * 0.28)
            .unwrap_or(0.0)
            + fruit.organ.vascular_supply * 0.16
            + glucose * 0.08,
        0.0,
        1.0,
    );
    let membrane = clamp(
        0.06 + fruit.organ.peel_integrity * 0.18
            + fruit.organ.flesh_integrity * 0.14
            + fruit.radius * 0.04,
        0.0,
        1.0,
    );
    let oxygen = clamp(
        (1.0 - fruit.organ.rot_progress.min(1.0)) * 0.24 + humidity * 0.08,
        0.0,
        1.0,
    );
    let carbon_dioxide = clamp(
        fruit.organ.rot_progress * 0.30
            + microbial_biomass.max(0.0) * 0.03
            + (1.0 - fruit.organ.flesh_integrity) * 0.12,
        0.0,
        1.4,
    );
    let ammonium = clamp(
        (1.0 - fruit.organ.flesh_integrity) * 0.08
            + fruit.organ.rot_progress * fruit.composition.amino_fraction * 0.32,
        0.0,
        0.9,
    );
    let atp_flux = clamp(
        fruit.organ.vascular_supply * 0.14 + (1.0 - fruit.organ.rot_progress.min(1.0)) * 0.04,
        0.0,
        0.6,
    );
    deposit_species_to_inventory(&mut target, TerrariumSpecies::Water, water);
    deposit_species_to_inventory(&mut target, TerrariumSpecies::Glucose, glucose);
    deposit_species_to_inventory(&mut target, TerrariumSpecies::AminoAcidPool, amino);
    deposit_species_to_inventory(&mut target, TerrariumSpecies::NucleotidePool, nucleotide);
    deposit_species_to_inventory(
        &mut target,
        TerrariumSpecies::MembranePrecursorPool,
        membrane,
    );
    deposit_species_to_inventory(&mut target, TerrariumSpecies::OxygenGas, oxygen);
    deposit_species_to_inventory(&mut target, TerrariumSpecies::CarbonDioxide, carbon_dioxide);
    deposit_species_to_inventory(&mut target, TerrariumSpecies::Ammonium, ammonium);
    deposit_species_to_inventory(&mut target, TerrariumSpecies::AtpFlux, atp_flux);
    target
}

impl TerrariumWorld {
    pub fn add_water(&mut self, x: usize, y: usize, volume: f32, emission_rate: f32) {
        self.waters.push(WaterSourceState {
            x,
            y,
            z: 0,
            volume,
            evaporation_rate: emission_rate,
            alive: true,
        });
        let amplitude = clamp(volume / 140.0, 0.06, 1.0);
        let radius = if volume >= 120.0 { 3 } else { 2 };
        deposit_2d(
            &mut self.water_mask,
            self.config.width,
            self.config.height,
            x,
            y,
            radius,
            amplitude,
        );

        let (sub_width, sub_height, sub_depth) = self.substrate.shape();
        let width = self.config.width.min(sub_width);
        let height = self.config.height.min(sub_height);
        let depth = sub_depth.max(1).min(3);
        let radius_f = radius as f32 + 0.8;
        for z in 0..depth {
            let depth_t = if depth <= 1 {
                0.0
            } else {
                z as f32 / (depth - 1) as f32
            };
            for dy in -(radius as isize)..=(radius as isize) {
                for dx in -(radius as isize)..=(radius as isize) {
                    let xx = x as isize + dx;
                    let yy = y as isize + dy;
                    if xx < 0 || yy < 0 || xx >= width as isize || yy >= height as isize {
                        continue;
                    }
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    let falloff = (1.0 - dist / radius_f).clamp(0.0, 1.0);
                    if falloff <= 0.0 {
                        continue;
                    }
                    let weight = falloff * (0.42 + amplitude * 0.58);
                    let xx = xx as usize;
                    let yy = yy as usize;
                    let ctrl = z * sub_width * sub_height + yy * sub_width + xx;
                    let hydration_target = 0.88 + amplitude * 0.42 - depth_t * 0.14;
                    self.substrate.hydration[ctrl] = clamp(
                        self.substrate.hydration[ctrl] * (1.0 - weight) + hydration_target * weight,
                        0.0,
                        1.5,
                    );

                    let water_idx = self.substrate.index(TerrariumSpecies::Water, xx, yy, z);
                    let water_target = 1.02 + amplitude * 1.36 - depth_t * 0.22;
                    self.substrate.current[water_idx] = clamp(
                        self.substrate.current[water_idx] * (1.0 - weight) + water_target * weight,
                        0.0,
                        3.0,
                    );
                }
            }
        }
    }

    pub fn add_fruit(&mut self, x: usize, y: usize, sugar: f32, volatile_scale: Option<f32>) {
        let volatile_scale = volatile_scale.unwrap_or(1.0);
        let (source_genome, parent_organism_id) = self
            .plants
            .iter()
            .min_by(|a, b| {
                let da = ((a.x as f32 - x as f32).powi(2) + (a.y as f32 - y as f32).powi(2))
                    .partial_cmp(
                        &((b.x as f32 - x as f32).powi(2) + (b.y as f32 - y as f32).powi(2)),
                    )
                    .unwrap_or(std::cmp::Ordering::Equal);
                da
            })
            .map(|plant| (plant.genome.clone(), Some(plant.identity.organism_id)))
            .unwrap_or_else(|| {
                (
                    crate::terrarium::plant_species::sample_named_plant_genome(&mut self.rng),
                    None,
                )
            });
        let taxonomy_id = source_genome.taxonomy_id;
        let (base_radius, base_ripeness, odorant_profile) = fruit_source_traits(&source_genome);
        let ripeness = clamp(base_ripeness + sugar * 0.04, 0.12, 0.98);
        // Use metabolome-derived composition if parent plant exists, else fall back to genome formula.
        let composition = self
            .plants
            .iter()
            .min_by(|a, b| {
                let da = (a.x as f32 - x as f32).powi(2) + (a.y as f32 - y as f32).powi(2);
                let db = (b.x as f32 - x as f32).powi(2) + (b.y as f32 - y as f32).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| super::fruit_state::fruit_composition_from_metabolome(&p.metabolome, ripeness))
            .unwrap_or_else(|| {
                super::fruit_state::fruit_composition_from_parent(
                    &source_genome,
                    taxonomy_id,
                    sugar,
                    ripeness,
                )
            });
        let identity = self.register_organism_identity(
            TerrariumOrganismKind::Fruit,
            Some(taxonomy_id),
            parent_organism_id,
            None,
            None,
            None,
            None,
        );
        self.fruits.push(TerrariumFruitPatch {
            identity,
            source: FruitSourceState {
                x,
                y,
                z: 1,
                attached: false,
                sugar_content: sugar,
                ripeness,
                odorant_emission_rate: 0.05 * volatile_scale,
                decay_rate: 0.001,
                alive: true,
                odorant_profile,
            },
            taxonomy_id,
            source_genome,
            composition,
            development: None,
            organ: super::fruit_state::detached_fruit_organ_state(ripeness),
            reproduction: None,
            radius: clamp(base_radius * (0.86 + sugar * 0.12), 0.30, 2.10),
            previous_remaining: sugar,
            deposited_all: false,
            material_inventory: RegionalMaterialInventory::new(format!("fruit:{x}:{y}")),
        });
    }

    fn add_attached_fruit(
        &mut self,
        x: usize,
        y: usize,
        parent_x: usize,
        parent_y: usize,
        sugar_capacity: f32,
        volatile_scale: Option<f32>,
    ) {
        let volatile_scale = volatile_scale.unwrap_or(1.0);
        let (source_genome, parent_identity) = self
            .plants
            .iter()
            .find(|plant| plant.x == parent_x && plant.y == parent_y)
            .map(|plant| (plant.genome.clone(), Some(plant.identity.clone())))
            .unwrap_or_else(|| {
                (
                    crate::terrarium::plant_species::sample_named_plant_genome(&mut self.rng),
                    None,
                )
            });
        let taxonomy_id = source_genome.taxonomy_id;
        let (base_radius, _base_ripeness, odorant_profile) = fruit_source_traits(&source_genome);
        let sugar_capacity = clamp(sugar_capacity, 0.12, 1.8);
        let mature_radius = clamp(base_radius * (0.86 + sugar_capacity * 0.12), 0.30, 2.10);
        let growth_progress = 0.08;
        let ripeness = 0.06 + growth_progress * 0.18;
        let sugar_content = sugar_capacity * 0.04;
        // Use metabolome-derived composition from parent plant if available.
        let composition = self
            .plants
            .iter()
            .find(|plant| plant.x == parent_x && plant.y == parent_y)
            .map(|p| super::fruit_state::fruit_composition_from_metabolome(&p.metabolome, ripeness))
            .unwrap_or_else(|| {
                super::fruit_state::fruit_composition_from_parent(
                    &source_genome,
                    taxonomy_id,
                    sugar_content,
                    ripeness,
                )
            });
        let reproduction = Some(super::fruit_reproduction::initialize_fruit_reproduction(
            &self.plants,
            parent_x,
            parent_y,
            &source_genome,
            &mut self.rng,
        ));
        let identity = self.register_organism_identity(
            TerrariumOrganismKind::Fruit,
            Some(taxonomy_id),
            parent_identity
                .as_ref()
                .map(|identity| identity.organism_id),
            None,
            None,
            None,
            None,
        );
        self.fruits.push(TerrariumFruitPatch {
            identity,
            source: FruitSourceState {
                x,
                y,
                z: 1,
                attached: true,
                sugar_content,
                ripeness,
                odorant_emission_rate: 0.03 * volatile_scale,
                decay_rate: 0.001,
                alive: true,
                odorant_profile,
            },
            taxonomy_id,
            source_genome,
            composition,
            development: Some(super::fruit_state::TerrariumFruitDevelopmentState {
                parent_x,
                parent_y,
                mature_radius,
                sugar_capacity,
                growth_progress,
            }),
            organ: super::fruit_state::attached_fruit_organ_state(),
            reproduction,
            radius: clamp(mature_radius * 0.24, 0.18, mature_radius),
            previous_remaining: sugar_content,
            deposited_all: false,
            material_inventory: RegionalMaterialInventory::new(format!("fruit:{x}:{y}")),
        });
    }

    fn release_seed_from_fruit_idx(&mut self, fruit_idx: usize) -> Option<TerrariumSeed> {
        let (x, y, parent_organism_id, co_parent_organism_id, released) = {
            let fruit = self.fruits.get_mut(fruit_idx)?;
            let co_parent_organism_id = fruit
                .reproduction
                .as_ref()
                .and_then(|reproduction| reproduction.paternal_organism_id);
            let released = super::fruit_reproduction::release_seed(fruit.reproduction.as_mut()?)?;
            fruit.source.attached = false;
            fruit.source.z = 0;
            fruit.organ.seed_exposure = 1.0;
            (
                fruit.source.x,
                fruit.source.y,
                fruit.identity.parent_organism_id,
                co_parent_organism_id,
                released,
            )
        };
        let phylo_parent_id = parent_organism_id
            .and_then(|organism_id| self.organism_registry_entry(organism_id))
            .and_then(|entry| entry.identity.phylo_id);
        let identity = self.register_organism_identity(
            TerrariumOrganismKind::Seed,
            Some(released.embryo_genome.taxonomy_id),
            parent_organism_id,
            co_parent_organism_id,
            phylo_parent_id,
            Some(super::organism_identity::plant_genome_hash(
                &released.embryo_genome,
            )),
            Some(super::organism_identity::seed_phylo_traits_from_genome(
                &released.embryo_genome,
                released.reserve_carbon,
                1.0,
            )),
        );

        Some(TerrariumSeed {
            identity,
            x: x as f32,
            y: y as f32,
            dormancy_s: released.dormancy_s,
            reserve_carbon: released.reserve_carbon,
            age_s: 0.0,
            genome: released.embryo_genome,
            cellular: released.embryo_cellular,
            pose: SeedPose::default(),
            microsite: seed_microsite::SeedMicrositeState::default(),
            material_inventory: released.material_inventory,
        })
    }

    fn sync_plant_fruit_loads(&mut self) {
        let mut fruit_loads = std::collections::HashMap::<(usize, usize), u32>::new();
        for fruit in &self.fruits {
            if !fruit.source.attached {
                continue;
            }
            let Some(development) = fruit.development.as_ref() else {
                continue;
            };
            *fruit_loads
                .entry((development.parent_x, development.parent_y))
                .or_insert(0) += 1;
        }
        for plant in &mut self.plants {
            plant
                .physiology
                .set_fruit_count(fruit_loads.get(&(plant.x, plant.y)).copied().unwrap_or(0));
        }
    }

    pub fn add_seed(
        &mut self,
        x: f32,
        y: f32,
        genome: TerrariumPlantGenome,
        reserve_carbon: Option<f32>,
        dormancy_s: Option<f32>,
    ) {
        let reserve_carbon = reserve_carbon
            .unwrap_or_else(|| clamp(genome.seed_mass * self.rng.gen_range(0.9..1.2), 0.03, 0.28));
        let dormancy_s = dormancy_s.unwrap_or_else(|| {
            self.rng.gen_range(9_000.0..24_000.0) * (1.15 - genome.seed_mass * 1.5).max(0.45)
        });
        let identity = self.register_organism_identity(
            TerrariumOrganismKind::Seed,
            Some(genome.taxonomy_id),
            None,
            None,
            None,
            Some(super::organism_identity::plant_genome_hash(&genome)),
            Some(super::organism_identity::seed_phylo_traits_from_genome(
                &genome,
                reserve_carbon,
                1.0,
            )),
        );
        self.seeds.push(TerrariumSeed {
            identity,
            x,
            y,
            dormancy_s,
            reserve_carbon,
            age_s: 0.0,
            cellular: SeedCellularStateSim::new(genome.seed_mass, reserve_carbon, dormancy_s),
            genome,
            pose: SeedPose::default(),
            microsite: seed_microsite::SeedMicrositeState::default(),
            material_inventory: RegionalMaterialInventory::new(format!("seed:{x:.2}:{y:.2}")),
        });
    }

    fn add_plant_with_identity(
        &mut self,
        x: usize,
        y: usize,
        genome: TerrariumPlantGenome,
        scale: f32,
        identity: Option<OrganismIdentity>,
    ) -> Result<usize, String> {
        let identity = if let Some(identity) = identity {
            self.promote_organism_identity_kind(
                &identity,
                TerrariumOrganismKind::Plant,
                Some(genome.taxonomy_id),
            )
        } else {
            self.register_organism_identity(
                TerrariumOrganismKind::Plant,
                Some(genome.taxonomy_id),
                None,
                None,
                None,
                Some(super::organism_identity::plant_genome_hash(&genome)),
                Some(super::organism_identity::plant_phylo_traits_from_genome(
                    &genome, scale,
                )),
            )
        };
        let plant = TerrariumPlant::new(x, y, genome, scale, identity, &mut self.rng);
        let id = self.plants.len();
        self.plants.push(plant);
        Ok(id)
    }

    pub fn add_plant(
        &mut self,
        x: usize,
        y: usize,
        genome: Option<TerrariumPlantGenome>,
        scale: Option<f32>,
    ) -> Result<usize, String> {
        let genome = genome.unwrap_or_else(|| {
            crate::terrarium::plant_species::sample_named_plant_genome(&mut self.rng)
        });
        let scale = scale.unwrap_or(1.0);
        self.add_plant_with_identity(x, y, genome, scale, None)
    }

    pub fn exchange_atmosphere_flux_bundle(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        co2_flux: f32,
        o2_flux: f32,
        humidity_flux: f32,
    ) {
        self.exchange_atmosphere_odorant(ATMOS_CO2_IDX, x, y, z, radius, co2_flux);
        self.exchange_atmosphere_odorant(ATMOS_O2_IDX, x, y, z, radius, o2_flux);
        self.exchange_atmosphere_humidity(x, y, z, radius, humidity_flux);
    }

    pub fn step_plants(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.plants.is_empty() {
            return Ok(());
        }
        self.sync_plant_fruit_loads();
        let depth = self.config.depth.max(1);
        let ammonium_surface = layer_mean_map(
            self.config.width,
            self.config.height,
            depth,
            self.substrate.species_field(TerrariumSpecies::Ammonium),
            0,
            depth.min(2),
        );
        let nitrate_surface = layer_mean_map(
            self.config.width,
            self.config.height,
            depth,
            self.substrate.species_field(TerrariumSpecies::Nitrate),
            0,
            depth.min(2),
        );
        let nitrate_deep = layer_mean_map(
            self.config.width,
            self.config.height,
            depth,
            self.substrate.species_field(TerrariumSpecies::Nitrate),
            depth / 2,
            depth,
        );
        let solar_state = super::solar::compute_solar_state(self.time_s, self.config.latitude_deg, self.weather.cloud_cover);
        let depth = self.config.depth.max(1);
        let mut queued_fruits = Vec::new();
        let mut queued_co2_fluxes: Vec<(usize, usize, usize, usize, f32)> = Vec::new();
        let mut queued_o2_fluxes: Vec<(usize, usize, usize, usize, f32)> = Vec::new();
        let mut queued_humidity_fluxes: Vec<(usize, usize, usize, usize, f32)> = Vec::new();
        let mut queued_defense_voc_emissions: Vec<(usize, f32)> = Vec::new();
        let mut dead_plants = Vec::new();

        // Build per-plant asymmetric competition descriptors.
        let canopy_descriptors: Vec<CanopyDescriptor> = self
            .plants
            .iter()
            .map(|p| {
                let r = p.genome.canopy_radius_mm.max(1.0);
                let leaf = p.physiology.leaf_biomass().max(0.0);
                let lai = leaf * 200.0 / (std::f32::consts::PI * r * r * 0.01);
                CanopyDescriptor {
                    x: p.x as f32,
                    y: p.y as f32,
                    height_mm: p.physiology.height_mm(),
                    canopy_radius_mm: r,
                    lai,
                    // Light extinction from specific leaf area (Monsi & Saeki 1953, Wright+ 2004).
                    // leaf_efficiency ∈ [0,1] maps to SLA 400→200 cm²/g (sun-adapted → shade-adapted)
                    extinction_coeff: super::emergent_rates::emergent_light_extinction(
                        400.0 - p.genome.leaf_efficiency * 200.0
                    ),
                }
            })
            .collect();
        let root_descriptors: Vec<RootDescriptor> = self
            .plants
            .iter()
            .map(|p| RootDescriptor {
                x: p.x as f32,
                y: p.y as f32,
                root_depth_mm: p.genome.root_depth_bias * p.physiology.height_mm() * 0.4,
                root_radius_mm: p.genome.root_radius_mm,
                root_biomass: p.physiology.root_biomass().max(0.0),
            })
            .collect();
        let light_factors =
            compute_light_competition(&canopy_descriptors, self.config.cell_size_mm);
        let photon_fluxes = super::solar::raycast_canopy_photons(
            &canopy_descriptors,
            &solar_state,
            self.config.cell_size_mm,
        );
        let root_factors = compute_root_competition(
            &root_descriptors,
            &self.mineral_nitrogen,
            &self.shallow_nutrients,
            self.config.width,
            self.config.height,
            self.config.cell_size_mm,
        );

        for idx in 0..self.plants.len() {
            let fruit_reset_s = self.rng.gen_range(7200.0..17000.0);
            let _seed_reset_s = self.rng.gen_range(12000.0..30000.0);
            let (
                x,
                y,
                genome,
                _canopy_self,
                _root_self,
                canopy_radius,
                root_radius,
                canopy_z,
                storage_signal,
                health_before,
            ) = {
                let plant = &self.plants[idx];
                (
                    plant.x,
                    plant.y,
                    plant.genome.clone(),
                    plant.canopy_amplitude(),
                    plant.root_amplitude(),
                    plant.canopy_radius_cells(),
                    plant.root_radius_cells(),
                    ((plant.physiology.height_mm() / 3.0).round() as usize).clamp(1, depth - 1),
                    plant.physiology.storage_carbon().max(0.0),
                    plant.physiology.health(),
                )
            };
            let flat = idx2(self.config.width, x, y);
            // Per-plant asymmetric competition from plant_competition module.
            let light_factor = light_factors.get(idx).copied().unwrap_or(1.0);
            let (root_n_factor, _root_p_factor) =
                root_factors.get(idx).copied().unwrap_or((1.0, 1.0));
            let canopy_comp = (1.0 - light_factor).max(0.0);
            let root_comp = (1.0 - root_n_factor).max(0.0);
            let symbionts = self.symbiont_biomass[flat];
            let deep_moisture = self.deep_moisture[flat];
            let litter = self.litter_carbon[flat];

            let oxygen = self.substrate.patch_mean_species(
                TerrariumSpecies::OxygenGas,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let proton = self.substrate.patch_mean_species(
                TerrariumSpecies::Proton,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let soil_nitrate_patch = self.substrate.patch_mean_species(
                TerrariumSpecies::Nitrate,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let carbon_dioxide = self.substrate.patch_mean_species(
                TerrariumSpecies::CarbonDioxide,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let dissolved_silicate = self.substrate.patch_mean_species(
                TerrariumSpecies::DissolvedSilicate,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let exchangeable_calcium = self.substrate.patch_mean_species(
                TerrariumSpecies::ExchangeableCalcium,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let exchangeable_magnesium = self.substrate.patch_mean_species(
                TerrariumSpecies::ExchangeableMagnesium,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let exchangeable_potassium = self.substrate.patch_mean_species(
                TerrariumSpecies::ExchangeablePotassium,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let exchangeable_sodium = self.substrate.patch_mean_species(
                TerrariumSpecies::ExchangeableSodium,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let exchangeable_aluminum = self.substrate.patch_mean_species(
                TerrariumSpecies::ExchangeableAluminum,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let aqueous_iron = self.substrate.patch_mean_species(
                TerrariumSpecies::AqueousIronPool,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let carbonate_mineral = self.substrate.patch_mean_species(
                TerrariumSpecies::CarbonateMineral,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let clay_mineral = self.substrate.patch_mean_species(
                TerrariumSpecies::ClayMineral,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let soil_redox = clamp(
                (oxygen * 1.3 + soil_nitrate_patch * 0.7)
                    / (oxygen * 1.3
                        + soil_nitrate_patch * 0.7
                        + carbon_dioxide * 0.35
                        + proton * 0.45
                        + 1e-9),
                0.0,
                1.0,
            );
            let soil_glucose = self.substrate.patch_mean_species(
                TerrariumSpecies::Glucose,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let soil_ammonium = self.substrate.patch_mean_species(
                TerrariumSpecies::Ammonium,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let atp_flux = self.substrate.patch_mean_species(
                TerrariumSpecies::AtpFlux,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let bicarbonate = self.substrate.patch_mean_species(
                TerrariumSpecies::BicarbonatePool,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let surface_proton_load = self.substrate.patch_mean_species(
                TerrariumSpecies::SurfaceProtonLoad,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let calcium_bicarbonate_complex = self.substrate.patch_mean_species(
                TerrariumSpecies::CalciumBicarbonateComplex,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let base_saturation = soil_base_saturation(
                exchangeable_calcium,
                exchangeable_magnesium,
                exchangeable_potassium,
                exchangeable_sodium,
                exchangeable_aluminum,
                proton,
                surface_proton_load,
            );
            let aluminum_toxicity = soil_aluminum_toxicity(
                exchangeable_aluminum,
                proton,
                surface_proton_load,
                base_saturation,
            );
            let weathering_support = soil_weathering_support(
                dissolved_silicate,
                bicarbonate,
                calcium_bicarbonate_complex,
                exchangeable_calcium,
                exchangeable_magnesium,
                exchangeable_potassium,
                aqueous_iron,
                carbonate_mineral,
                clay_mineral,
            );
            let mineral_buffer = soil_mineral_buffer(
                carbonate_mineral,
                bicarbonate,
                calcium_bicarbonate_complex,
                exchangeable_calcium,
                exchangeable_magnesium,
                proton,
                surface_proton_load,
                base_saturation,
            );
            let root_energy_gate = clamp(0.32 + atp_flux * 1800.0, 0.2, 1.2);
            let water_deficit = clamp(
                0.42 - (self.moisture[flat] + deep_moisture * 0.35),
                0.0,
                1.0,
            );
            let nitrogen_deficit = clamp(
                0.14 - (self.mineral_nitrogen[flat] + soil_ammonium + soil_nitrate_patch) * 0.32,
                0.0,
                1.0,
            );

            let (water_demand, nutrient_demand) = self.plants[idx].physiology.resource_demands(
                eco_dt,
                root_energy_gate,
                water_deficit,
                nitrogen_deficit,
            );
            let extraction = extract_root_resources_with_layers(
                self.config.width,
                self.config.height,
                x as i32,
                y as i32,
                root_radius as i32,
                water_demand,
                nutrient_demand,
                genome.root_depth_bias,
                genome.symbiosis_affinity,
                root_energy_gate,
                &self.moisture,
                &self.deep_moisture,
                &self.dissolved_nutrients,
                &self.mineral_nitrogen,
                &self.shallow_nutrients,
                &self.deep_minerals,
                &self.symbiont_biomass,
                &ammonium_surface,
                &nitrate_surface,
                &nitrate_deep,
            )?;
            self.moisture = extraction.moisture;
            self.deep_moisture = extraction.deep_moisture;
            self.dissolved_nutrients = extraction.dissolved_nutrients;
            self.mineral_nitrogen = extraction.mineral_nitrogen;
            self.shallow_nutrients = extraction.shallow_nutrients;
            self.deep_minerals = extraction.deep_minerals;

            let total_water_take = extraction.surface_water_take + extraction.deep_water_take;
            if total_water_take > 0.0 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Water,
                    x,
                    y,
                    0,
                    root_radius,
                    total_water_take,
                );
            }
            if extraction.ammonium_take > 0.0 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Ammonium,
                    x,
                    y,
                    0,
                    root_radius,
                    extraction.ammonium_take,
                );
            }
            let nitrate_take = extraction.rhizo_nitrate_take + extraction.deep_nitrate_take;
            if nitrate_take > 0.0 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Nitrate,
                    x,
                    y,
                    0,
                    root_radius,
                    nitrate_take,
                );
            }

            let root_zone_z = root_zone_depth_from_bias(depth, genome.root_depth_bias);
            let plant_inventory_target = material_inventory_target_from_patch(
                &self.substrate,
                x,
                y,
                root_zone_z,
                root_radius,
                format!("plant-target:{x}:{y}:{root_zone_z}"),
            );
            let plant_sync_relax = clamp(
                0.24 + root_energy_gate * 0.16 + genome.symbiosis_affinity * 0.10,
                0.12,
                0.46,
            );
            let (
                inventory_water_take,
                inventory_amino_take,
                inventory_nucleotide_take,
                inventory_membrane_take,
                inventory_glucose_signal,
                inventory_base_signal,
                inventory_aluminum_signal,
                inventory_weathering_signal,
            ) = {
                let plant = &mut self.plants[idx];
                sync_inventory_with_patch(
                    &mut self.substrate,
                    &mut plant.material_inventory,
                    x,
                    y,
                    root_zone_z,
                    root_radius,
                    &plant_inventory_target,
                    plant_sync_relax,
                )?;
                let division_drive = plant.cellular.division_signal().clamp(0.0, 2.0);
                let nitrogen_shortage = (0.20 - plant.cellular.nitrogen_pool()).max(0.0);
                let water_take = withdraw_species_from_inventory(
                    &mut plant.material_inventory,
                    TerrariumSpecies::Water,
                    water_demand * clamp(0.18 + root_energy_gate * 0.14, 0.12, 0.38),
                );
                let amino_take = withdraw_species_from_inventory(
                    &mut plant.material_inventory,
                    TerrariumSpecies::AminoAcidPool,
                    nutrient_demand
                        * clamp(
                            0.18 + division_drive * 0.08 + nitrogen_shortage * 0.40,
                            0.06,
                            0.40,
                        ),
                );
                let nucleotide_take = withdraw_species_from_inventory(
                    &mut plant.material_inventory,
                    TerrariumSpecies::NucleotidePool,
                    nutrient_demand
                        * clamp(
                            0.10 + division_drive * 0.10 + plant.cellular.energy_charge() * 0.06,
                            0.04,
                            0.28,
                        ),
                );
                let membrane_take = withdraw_species_from_inventory(
                    &mut plant.material_inventory,
                    TerrariumSpecies::MembranePrecursorPool,
                    nutrient_demand
                        * clamp(
                            0.06 + plant.cellular.vitality() * 0.06
                                + (plant.cellular.total_cells() / 800.0).clamp(0.0, 0.12),
                            0.03,
                            0.20,
                        ),
                );
                let glucose_signal = inventory_component_amount(
                    &plant.material_inventory,
                    TerrariumSpecies::Glucose,
                );
                let base_signal = soil_base_cation_pool(
                    inventory_component_amount(
                        &plant.material_inventory,
                        TerrariumSpecies::ExchangeableCalcium,
                    ),
                    inventory_component_amount(
                        &plant.material_inventory,
                        TerrariumSpecies::ExchangeableMagnesium,
                    ),
                    inventory_component_amount(
                        &plant.material_inventory,
                        TerrariumSpecies::ExchangeablePotassium,
                    ),
                    inventory_component_amount(
                        &plant.material_inventory,
                        TerrariumSpecies::ExchangeableSodium,
                    ),
                );
                let aluminum_signal = inventory_component_amount(
                    &plant.material_inventory,
                    TerrariumSpecies::ExchangeableAluminum,
                );
                let weathering_signal = inventory_component_amount(
                    &plant.material_inventory,
                    TerrariumSpecies::DissolvedSilicate,
                ) + inventory_component_amount(
                    &plant.material_inventory,
                    TerrariumSpecies::AqueousIronPool,
                ) * 0.5;
                (
                    water_take,
                    amino_take,
                    nucleotide_take,
                    membrane_take,
                    glucose_signal,
                    base_signal,
                    aluminum_signal,
                    weathering_signal,
                )
            };
            let plant_water_take = extraction.water_take + inventory_water_take;
            let plant_nutrient_take = extraction.nutrient_take
                + inventory_amino_take * 0.72
                + inventory_nucleotide_take * 0.58
                + inventory_membrane_take * 0.22;
            let effective_base_saturation =
                clamp(base_saturation + inventory_base_signal * 0.12, 0.0, 1.25);
            let effective_aluminum_toxicity = clamp(
                aluminum_toxicity + inventory_aluminum_signal * 0.08,
                0.0,
                1.6,
            );
            let effective_weathering_support = clamp(
                weathering_support + inventory_weathering_signal * 0.08,
                0.0,
                1.6,
            );
            let water_factor = if water_demand > 1.0e-6 {
                clamp(plant_water_take / water_demand, 0.0, 1.18)
            } else {
                1.0
            };
            let nutrient_factor = if nutrient_demand > 1.0e-6 {
                clamp(
                    plant_nutrient_take / nutrient_demand
                        * clamp(
                            0.72 + effective_base_saturation * 0.24
                                + effective_weathering_support * 0.10
                                + mineral_buffer * 0.06
                                - effective_aluminum_toxicity * 0.14,
                            0.42,
                            1.18,
                        ),
                    0.0,
                    1.18,
                )
            } else {
                1.0
            };
            let _soil_glucose_signal = soil_glucose + inventory_glucose_signal * 0.16;
            let local_temp = self.sample_temperature_at(x, y, 1.min(depth - 1));
            let temp_factor = temp_response(local_temp, 24.0, 10.0);
            let local_humidity = self.sample_humidity_at(x, y, canopy_z);
            // Photon raycasting: PAR from solar model attenuated through canopy geometry.
            // The raycast accounts for sun angle, time of day, atmospheric effects, and 3D canopy shading.
            let flux = photon_fluxes.get(idx).copied().unwrap_or_default();
            let normalized_par = flux.par_received / super::solar::PAR_FULL_SUN;
            // Moonlight adds ~0.1% of peak PAR (enough for photoperiodism signaling).
            let moonlight_contribution = if !solar_state.is_daytime {
                self.moonlight() * 0.001
            } else {
                0.0
            };
            let local_light = (normalized_par + moonlight_contribution).clamp(0.03, 1.15);
            let local_air_co2 = self.sample_odorant_patch(
                ATMOS_CO2_IDX,
                x,
                y,
                canopy_z,
                (canopy_radius.max(2) / 2).max(1),
            );
            let air_co2_factor = clamp(local_air_co2 / ATMOS_CO2_BASELINE, 0.35, 1.8);
            let stomatal_open = clamp(
                0.24 + local_light * 0.46 + water_factor * 0.28 + local_humidity * 0.22
                    - water_deficit * 0.32
                    - canopy_comp * 0.02,
                0.12,
                1.45,
            );
            let symbiosis_signal = clamp(
                symbionts * 6.5 * genome.symbiosis_affinity
                    + effective_weathering_support * 0.10
                    + effective_base_saturation * 0.08,
                0.0,
                1.5,
            );
            let symbiosis_bonus = 1.0
                + symbionts * 7.5 * genome.symbiosis_affinity
                + effective_weathering_support * 0.08
                + effective_base_saturation * 0.05
                - effective_aluminum_toxicity * 0.05;
            let stress_signal = clamp(
                water_deficit * 0.75
                    + nitrogen_deficit * 0.58
                    + canopy_comp * 0.018
                    + root_comp * 0.014
                    + (1.0 - soil_redox) * 0.45
                    + effective_aluminum_toxicity * 0.22
                    + (0.48 - effective_base_saturation).max(0.0) * 0.16
                    - mineral_buffer * 0.10,
                0.0,
                1.4,
            );

            // Sample defense VOC concentration from the odorant grid at plant canopy height.
            // This must happen before the mutable borrow of self.plants[idx] because
            // both odorant sampling and plant mutation go through &mut self.
            let defense_voc_concentration = {
                let width = self.config.width;
                let height = self.config.height;
                if DEFENSE_VOC_IDX < self.odorants.len() {
                    let idx_3d = (canopy_z * height + y) * width + x;
                    self.odorants[DEFENSE_VOC_IDX]
                        .get(idx_3d)
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            };

            let (report, new_total_cells, cell_vitality, volatile_scale, root_radius_after, molecular_viability, shade_elongation, shade_branching) = {
                let plant = &mut self.plants[idx];

                // === Molecular Botany: GRN + Metabolome ===
                // Build EnvironmentState from terrarium signals for the GRN.
                // Wires defense signaling from Phase 5: neighbor VOC from odorant grid,
                // internal JA/SA from metabolome pools.
                let env_state = crate::botany::EnvironmentState {
                    temperature_c: local_temp,
                    soil_moisture: water_factor.clamp(0.0, 1.0),
                    light_intensity: local_light.clamp(0.0, 1.0),
                    // Photoperiod: derived from astronomical solar model (latitude + day of year).
                    photoperiod_hours: solar_state.day_length_hours,
                    soil_nitrate: soil_nitrate_patch * 2.0, // scale substrate concentration to ~mM range
                    internal_aba: stress_signal.clamp(0.0, 1.0), // ABA correlates with stress
                    // Spectral R:FR ratio from canopy raycasting (Phase 4).
                    r_fr_ratio: flux.r_fr_ratio,
                    // Mechanical stem damage from wind/turbulence (Phase 1).
                    mechanical_stress: plant.morphology.mechanical_damage,
                    // Defense VOC from odorant grid: neighbor plant damage signal (Phase 5).
                    neighbor_voc: defense_voc_concentration,
                    // Internal jasmonate normalized to [0,1] (10 molecules = saturating).
                    jasmonic_acid: (plant.metabolome.jasmonate_count as f32 / 10.0).clamp(0.0, 1.0),
                    // Internal salicylate normalized to [0,1].
                    salicylic_acid: (plant.metabolome.salicylate_count as f32 / 10.0).clamp(0.0, 1.0),
                };

                // Step the gene regulatory network — Hill-function kinetics update expression levels.
                plant.botanical_genome.step_gene_regulation(&env_state, eco_dt);

                // Collect expression levels for metabolome coupling.
                let gene_expr = plant.botanical_genome.expression_snapshot();

                // Feed CO2 and H2O from the terrarium grid into the metabolome pools.
                // Scale by leaf biomass so larger plants absorb proportionally more.
                let leaf_area_factor = (plant.physiology.leaf_biomass() * 4.0).max(0.1) as f64;
                let co2_from_atmosphere = (air_co2_factor * leaf_area_factor as f32 * stomatal_open * 50.0 * eco_dt) as f64;
                let water_from_soil = (plant_water_take * 200.0) as f64;
                plant.metabolome.replenish_substrates(co2_from_atmosphere, water_from_soil);

                // Run the full metabolic step — photosynthesis, sugar interconversion,
                // malate, ethylene, VOC, respiration, all modulated by gene expression.
                let _metabolic_report = plant.metabolome.full_metabolic_step(
                    local_light,
                    local_temp,
                    &gene_expr,
                    eco_dt,
                );

                // Queue defense VOC emission into the odorant grid (Phase 5).
                // GLV + MeSA from the metabolome are emitted at canopy height.
                let defense_emission = plant.metabolome.defense_voc_emission();
                if defense_emission > 0.001 {
                    let width = self.config.width;
                    let height = self.config.height;
                    let idx_3d = (canopy_z * height + y) * width + x;
                    queued_defense_voc_emissions.push((idx_3d, defense_emission * eco_dt * 0.1));
                }
                // === End Molecular Botany ===

                let cell_feedback = plant.cellular.step(
                    eco_dt,
                    local_light,
                    temp_factor,
                    plant_water_take,
                    plant_nutrient_take,
                    water_factor,
                    nutrient_factor,
                    symbiosis_signal,
                    stress_signal,
                    storage_signal,
                );
                // Compute the molecular drive state from metabolome + gene expression.
                // This replaces ~80 hardcoded thresholds with Hill/MM kinetic functions.
                let molecular_drive = crate::botany::physiology_bridge::compute_molecular_drive(
                    &plant.metabolome,
                    &gene_expr,
                    local_light,
                    water_factor,
                    plant.physiology.leaf_biomass(),
                    plant.physiology.total_biomass(),
                    plant.genome.volatile_scale,
                    eco_dt,
                    plant.morphology.mechanical_damage,
                );

                // Use the molecular-driven step instead of the hardcoded threshold step.
                let report = plant.physiology.step_molecular(
                    eco_dt,
                    plant_water_take,
                    plant_nutrient_take,
                    &molecular_drive,
                    temp_factor,
                    root_energy_gate,
                    symbiosis_bonus,
                    canopy_comp,
                    root_comp,
                    cell_feedback.energy_charge,
                    cell_feedback.vitality,
                    cell_feedback.maintenance_cost,
                    cell_feedback.senescence_mass,
                    plant.cellular.total_cells(),
                    fruit_reset_s,
                );
                (
                    report,
                    plant.cellular.total_cells(),
                    plant.cellular.vitality(),
                    plant.genome.volatile_scale,
                    plant.root_radius_cells(),
                    molecular_drive.metabolic_viability,
                    molecular_drive.shade_elongation_factor,
                    molecular_drive.shade_branching_factor,
                )
            };

            deposit_2d(
                &mut self.root_exudates,
                self.config.width,
                self.config.height,
                x,
                y,
                root_radius_after.max(1) / 2,
                report.exudates,
            );
            deposit_2d(
                &mut self.litter_carbon,
                self.config.width,
                self.config.height,
                x,
                y,
                root_radius_after.max(1) / 2,
                report.litter,
            );
            let hotspot_radius = root_radius_after.max(1);
            {
                let plant = &mut self.plants[idx];
                deposit_species_to_inventory(
                    &mut plant.material_inventory,
                    TerrariumSpecies::Glucose,
                    report.exudates * 12.0,
                );
                deposit_species_to_inventory(
                    &mut plant.material_inventory,
                    TerrariumSpecies::AminoAcidPool,
                    report.exudates * 5.0 + report.litter * 1.6,
                );
                deposit_species_to_inventory(
                    &mut plant.material_inventory,
                    TerrariumSpecies::NucleotidePool,
                    report.exudates * 1.8 + report.litter * 1.4,
                );
                deposit_species_to_inventory(
                    &mut plant.material_inventory,
                    TerrariumSpecies::MembranePrecursorPool,
                    report.exudates * 1.4 + report.litter * 3.4,
                );
                deposit_species_to_inventory(
                    &mut plant.material_inventory,
                    TerrariumSpecies::Ammonium,
                    report.litter * 8.0,
                );
                deposit_species_to_inventory(
                    &mut plant.material_inventory,
                    TerrariumSpecies::CarbonDioxide,
                    report.litter * 4.0,
                );
                let spill_target = material_inventory_target_from_patch(
                    &self.substrate,
                    x,
                    y,
                    root_zone_z,
                    hotspot_radius,
                    format!("plant-spill-target:{x}:{y}:{root_zone_z}"),
                );
                let spill_relax = clamp(
                    0.18 + report.exudates * 0.10 + report.litter * 0.14 + root_energy_gate * 0.10,
                    0.10,
                    0.48,
                );
                sync_inventory_with_patch(
                    &mut self.substrate,
                    &mut plant.material_inventory,
                    x,
                    y,
                    root_zone_z,
                    hotspot_radius,
                    &spill_target,
                    spill_relax,
                )?;
            }

            // ---- Root exudate-driven mineral solubilization ----
            //
            // Organic acids (citrate, malate) chelate metal ions and lower
            // rhizosphere pH, solubilizing locked nutrients from mineral phases.
            //
            // Rate constants derived from Michaelis-Menten kinetics on
            // published dissolution rates in rhizosphere soil (Hinsinger 2001,
            // "Bioavailability of soil inorganic P in the rhizosphere"):
            //
            //   Carbonate: k_CaCO3 ≈ 10⁻³·⁵ mol/m²/s at pH 5 (Chou+ 1989)
            //     → fastest dissolving: rate constant = highest
            //   Silicate:  k_SiO2  ≈ 10⁻⁴·⁵ mol/m²/s at pH 5 (Brady+ 1990)
            //     → ~10x slower than carbonate
            //   Fe-oxide:  k_FeOOH ≈ 10⁻⁵·⁵ mol/m²/s (Schwertmann 1991)
            //     → most refractory: ~100x slower than carbonate
            //
            // Normalized to sim scale: carbonate=1.0, silicate=0.1, iron=0.01
            // Base rate scaled by cell_size² (surface area) / cell_volume.
            //
            // Proton donation: citrate has 3 carboxyl groups (pKa 3.1, 4.8, 6.4),
            // malate has 2 (pKa 3.4, 5.1). Mean ~2.5 protons per molecule at
            // rhizosphere pH ~5-6. Proton flux proportional to exudate strength.
            if report.exudates > 0.001 {
                let exudate_strength =
                    crate::botany::physiology_bridge::hill(report.exudates, 0.01, 2.0);

                // Dissolution rates derived from Eyring/TST via molecular lattice
                // cohesion energy — emergent from mineral structure, not hardcoded.
                // Hierarchy (carbonate >> silicate >> iron oxide) arises naturally
                // from the lattice energy differences.
                let cell_idx = y * self.config.width + x;
                let temp_c = self.temperature.get(cell_idx).copied().unwrap_or(20.0)
                    + self.weather.temperature_offset_c;
                let k_carbonate = eco_dt * super::emergent_rates::emergent_dissolution_rate(
                    TerrariumSpecies::CarbonateMineral, exudate_strength, temp_c,
                );
                let k_silicate = eco_dt * super::emergent_rates::emergent_dissolution_rate(
                    TerrariumSpecies::SilicateMineral, exudate_strength, temp_c,
                );
                let k_iron = eco_dt * super::emergent_rates::emergent_dissolution_rate(
                    TerrariumSpecies::IronOxideMineral, exudate_strength, temp_c,
                );

                // Silicate dissolution: exudate + SilicateMineral → DissolvedSilicate
                let silicate_avail = self.substrate.patch_mean_species(
                    TerrariumSpecies::SilicateMineral,
                    x,
                    y,
                    root_zone_z,
                    1,
                );
                let si_dissolved = silicate_avail * k_silicate;
                self.substrate.deposit_patch_species(
                    TerrariumSpecies::DissolvedSilicate,
                    x,
                    y,
                    root_zone_z,
                    1,
                    si_dissolved,
                );
                self.substrate.deposit_patch_species(
                    TerrariumSpecies::SilicateMineral,
                    x,
                    y,
                    root_zone_z,
                    1,
                    -si_dissolved,
                );

                // Iron oxide dissolution: exudate + IronOxideMineral → AqueousIronPool
                let iron_avail = self.substrate.patch_mean_species(
                    TerrariumSpecies::IronOxideMineral,
                    x,
                    y,
                    root_zone_z,
                    1,
                );
                let fe_dissolved = iron_avail * k_iron;
                self.substrate.deposit_patch_species(
                    TerrariumSpecies::AqueousIronPool,
                    x,
                    y,
                    root_zone_z,
                    1,
                    fe_dissolved,
                );
                self.substrate.deposit_patch_species(
                    TerrariumSpecies::IronOxideMineral,
                    x,
                    y,
                    root_zone_z,
                    1,
                    -fe_dissolved,
                );

                // Carbonate dissolution: exudate + CarbonateMineral → BicarbonatePool
                let carb_avail = self.substrate.patch_mean_species(
                    TerrariumSpecies::CarbonateMineral,
                    x,
                    y,
                    root_zone_z,
                    1,
                );
                let carb_dissolved = carb_avail * k_carbonate;
                self.substrate.deposit_patch_species(
                    TerrariumSpecies::BicarbonatePool,
                    x,
                    y,
                    root_zone_z,
                    1,
                    carb_dissolved,
                );
                self.substrate.deposit_patch_species(
                    TerrariumSpecies::CarbonateMineral,
                    x,
                    y,
                    root_zone_z,
                    1,
                    -carb_dissolved,
                );

                // Proton donation from organic acid dissociation.
                // Citrate/malate donate ~2.5 H⁺ per molecule (mean of 3+2 carboxyl
                // groups at rhizosphere pH 5-6). Proton flux = exudate_strength *
                // stoichiometric_protons * dt.
                let protons_per_acid = 2.5;
                self.substrate.deposit_patch_species(
                    TerrariumSpecies::Proton,
                    x,
                    y,
                    root_zone_z,
                    1,
                    exudate_strength * eco_dt * protons_per_acid * 0.0004,
                );
            }

            deposit_2d(
                &mut self.organic_matter,
                self.config.width,
                self.config.height,
                x,
                y,
                hotspot_radius / 2,
                report.litter * 0.18,
            );
            queued_co2_fluxes.push((
                x,
                y,
                canopy_z,
                (canopy_radius.max(2) / 2).max(1),
                report.co2_flux,
            ));
            queued_o2_fluxes.push((
                x,
                y,
                canopy_z,
                (canopy_radius.max(2) / 2).max(1),
                (-report.co2_flux * 1.05),
            ));
            queued_humidity_fluxes.push((
                x,
                y,
                canopy_z,
                (canopy_radius.max(2) / 2).max(1),
                report.water_vapor_flux,
            ));

            // === Phase 3: Epigenetic Morphology (L-Systems) ===
            {
                let flat_idx = idx2(self.config.width, x, y);
                let local_moisture = self.moisture[flat_idx];
                let plant = &mut self.plants[idx];

                // 1. Modulate geometric parameters based on local moisture/light
                plant
                    .morphology
                    .modulate_epigenetics(local_moisture, local_light);
                let structure = crate::terrarium::shape_projection::plant_structure_descriptor(
                    &plant.genome,
                    &plant.physiology,
                    &plant.cellular,
                    local_moisture,
                    local_light,
                    local_temp,
                );
                // Shade Avoidance Syndrome: SAS gene expression modulates branching
                // and elongation via Hill kinetics (Phase 4). Under canopy shade
                // (low R:FR → high SAS), lateral branching is suppressed and
                // internode length increases — concentrating growth into vertical
                // escape rather than lateral spread.
                plant.morphology.internode_length = structure.internode_length * shade_elongation;
                plant.morphology.branch_angle_rad = structure.branch_angle_rad;
                plant.morphology.thickness_base = structure.thickness_base;
                plant.morphology.fruit_radius_scale = structure.fruit_radius_scale;
                plant.morphology.leaf_radius_scale = structure.leaf_radius_scale;
                plant.morphology.lateral_bias = structure.lateral_bias * shade_branching;
                plant.morphology.droop_factor = structure.droop_factor;
                plant.morphology.branch_twist_rad = structure.branch_twist_rad;
                plant.morphology.branch_depth_attenuation = structure.branch_depth_attenuation;
                plant.morphology.canopy_depth_scale = structure.canopy_depth_scale;
                plant.morphology.leaf_cluster_density = structure.leaf_cluster_density;

                // 2. Molecular visual state → turgor-driven droop
                {
                    let gene_snap = plant.botanical_genome.expression_snapshot();
                    let total_bm = plant.physiology.leaf_biomass() + plant.physiology.stem_biomass();
                    let vis = crate::botany::visual_phenotype::MolecularVisualState::from_metabolome(
                        &plant.metabolome,
                        &gene_snap,
                        total_bm,
                    );
                    plant.morphology.molecular_droop = vis.droop_from_turgor();

                    // 3. Phototropism: light asymmetry → lean direction
                    //    Use the relative light direction from nearest shading neighbor.
                    //    PIN1 expression gates the phototropic strength.
                    let pin1_expr = gene_snap.get("PIN1").copied().unwrap_or(0.5);
                    // Light gradient from raycast: asymmetry vector points toward brighter side.
                    // 4 offset rays (±x, ±y) measure differential shading from canopy neighbors.
                    let asymmetry = photon_fluxes.get(idx).map(|f| f.light_asymmetry).unwrap_or([0.0; 3]);
                    let asym_mag = (asymmetry[0] * asymmetry[0] + asymmetry[2] * asymmetry[2]).sqrt().max(0.001);
                    plant.morphology.phototropic_direction = [asymmetry[0] / asym_mag, 0.0, asymmetry[2] / asym_mag];
                    plant.morphology.phototropic_strength = pin1_expr * asym_mag * 0.5;
                }

                // 4. Growth trigger: Expand L-System if biomass significantly exceeds current iteration depth
                let total_biomass =
                    plant.physiology.leaf_biomass() + plant.physiology.stem_biomass();
                let target_iterations = (total_biomass * 15.0).clamp(0.0, 6.0) as u32;

                while plant.morphology.iterations < target_iterations {
                    plant.morphology.grow();
                }
            }

            if report.spawned_fruit
                && self.fruits.len() + queued_fruits.len() < self.config.max_fruits
            {
                queued_fruits.push((x, y, report.fruit_size, volatile_scale));
            }
            if self.plants[idx].physiology.is_dead_molecular(molecular_viability)
                || (new_total_cells < 24.0 && cell_vitality < 0.08)
            {
                dead_plants.push(idx);
            }

            let _ = (health_before, litter);
        }

        for (x, y, z, radius, flux) in queued_co2_fluxes {
            self.exchange_atmosphere_odorant(ATMOS_CO2_IDX, x, y, z, radius, flux);
        }
        for (x, y, z, radius, flux) in queued_o2_fluxes {
            self.exchange_atmosphere_odorant(ATMOS_O2_IDX, x, y, z, radius, flux);
        }
        for (x, y, z, radius, flux) in queued_humidity_fluxes {
            self.exchange_atmosphere_humidity(x, y, z, radius, flux);
        }
        // Apply queued defense VOC emissions to the odorant grid (Phase 5).
        // GLV + MeSA diffuse from the emitting plant's canopy into the 3D grid,
        // where wind turbulence disperses them to neighboring plants.
        if DEFENSE_VOC_IDX < self.odorants.len() {
            for (idx_3d, amount) in &queued_defense_voc_emissions {
                if let Some(slot) = self.odorants[DEFENSE_VOC_IDX].get_mut(*idx_3d) {
                    *slot = (*slot + amount).min(1.0);
                }
            }
        }

        dead_plants.sort_unstable();
        dead_plants.dedup();
        for idx in dead_plants.into_iter().rev() {
            let plant = self.plants.swap_remove(idx);
            self.mark_organism_dead(plant.identity.organism_id);
            let litter_return = 0.02 + plant.physiology.storage_carbon().max(0.0) * 0.20;
            deposit_2d(
                &mut self.litter_carbon,
                self.config.width,
                self.config.height,
                plant.x,
                plant.y,
                plant.root_radius_cells().max(1) / 2,
                litter_return,
            );
            deposit_2d(
                &mut self.organic_matter,
                self.config.width,
                self.config.height,
                plant.x,
                plant.y,
                plant.root_radius_cells().max(1) / 3,
                litter_return * 0.22,
            );
            self.substrate.add_hotspot(
                TerrariumSpecies::Glucose,
                plant.x,
                plant.y,
                0,
                litter_return * 10.0,
            );
            self.substrate.add_hotspot(
                TerrariumSpecies::Ammonium,
                plant.x,
                plant.y,
                0,
                litter_return * 6.0,
            );
            self.substrate.add_hotspot(
                TerrariumSpecies::AminoAcidPool,
                plant.x,
                plant.y,
                0,
                litter_return * 3.8,
            );
            self.substrate.add_hotspot(
                TerrariumSpecies::NucleotidePool,
                plant.x,
                plant.y,
                0,
                litter_return * 2.4,
            );
            self.substrate.add_hotspot(
                TerrariumSpecies::MembranePrecursorPool,
                plant.x,
                plant.y,
                0,
                litter_return * 2.8,
            );
            self.substrate.add_hotspot(
                TerrariumSpecies::CarbonDioxide,
                plant.x,
                plant.y,
                1.min(self.config.depth.max(1) - 1),
                litter_return * 3.5,
            );
        }
        for (x, y, size, volatile_scale) in queued_fruits {
            let offset = self.rng.gen_range(0..4);
            let (dx, dy) = match offset {
                0 => (2, 1),
                1 => (-2, 1),
                2 => (1, -2),
                _ => (-1, -2),
            };
            let fx = offset_clamped(x, dx, self.config.width);
            let fy = offset_clamped(y, dy, self.config.height);
            self.add_attached_fruit(fx, fy, x, y, size, Some(volatile_scale));
        }
        self.step_mycorrhizal_exchange(eco_dt);
        Ok(())
    }

    /// Mycorrhizal carbon/nutrient sharing between plants with overlapping root zones.
    ///
    /// Plants connected by fungal hyphae share carbon (glucose) along concentration
    /// gradients through a "common mycorrhizal network" (CMN). Connection strength
    /// scales with both plants' `symbiosis_affinity` (heritable genome trait) and
    /// root proximity (Hill kinetics on overlap fraction).
    fn step_mycorrhizal_exchange(&mut self, dt: f32) {
        use crate::terrarium::material_exchange::{
            deposit_species_to_inventory, inventory_component_amount,
        };

        let n = self.plants.len();
        if n < 2 {
            return;
        }

        // Collect pairwise transfer deltas (to avoid aliasing &mut issues)
        let mut transfers: Vec<(usize, usize, f32)> = Vec::new();
        let mut connection_count: usize = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.plants[i].x as f32 - self.plants[j].x as f32;
                let dy = self.plants[i].y as f32 - self.plants[j].y as f32;
                let dist = (dx * dx + dy * dy).sqrt();

                let ri = self.plants[i].root_radius_cells() as f32;
                let rj = self.plants[j].root_radius_cells() as f32;
                let max_reach = ri + rj;
                if dist > max_reach || max_reach < 0.01 {
                    continue;
                }

                // Connection strength: Hill on overlap * geometric mean of symbiosis affinities
                let overlap = (max_reach - dist) / max_reach;
                let affinity = (self.plants[i].genome.symbiosis_affinity
                    * self.plants[j].genome.symbiosis_affinity)
                    .sqrt();
                let connection = crate::botany::physiology_bridge::hill(overlap, 0.3, 2.0)
                    * crate::botany::physiology_bridge::hill(affinity, 0.6, 2.0);

                if connection < 0.01 {
                    continue;
                }

                connection_count += 1;

                // Carbon sharing: glucose flows from surplus to deficit via
                // Fick's first law on the fungal network.
                //
                // Measured CMN carbon flux: ~4-10% of photosynthate over 24h
                // (Simard+ 1997, Nature 388:579). At sim time_warp ~900x,
                // 24h real = ~96s sim. So 7% / 96s ≈ 7.3e-4 /s per unit
                // connection strength. We use the gradient * connection to get
                // Fickian diffusion along the hyphal network.
                let glucose_i = inventory_component_amount(
                    &self.plants[i].material_inventory,
                    TerrariumSpecies::Glucose,
                );
                let glucose_j = inventory_component_amount(
                    &self.plants[j].material_inventory,
                    TerrariumSpecies::Glucose,
                );
                let gradient = glucose_i - glucose_j;
                // Transfer rate derived from Stokes-Einstein D for glucose
                // through porous fungal hyphae (porosity=0.4, tortuosity=1.5).
                // Matches Simard+ 1997 (~7% photosynthate/24h) emergently.
                let pi_idx = self.plants[i].y * self.config.width + self.plants[i].x;
                let temp_c = self.temperature.get(pi_idx).copied().unwrap_or(20.0)
                    + self.weather.temperature_offset_c;
                let cmn_transfer_rate = super::emergent_rates::emergent_hyphal_transfer_rate(
                    TerrariumSpecies::Glucose,
                    temp_c,
                );
                let transfer = gradient * connection * dt * cmn_transfer_rate;

                if transfer.abs() > 1e-6 {
                    transfers.push((i, j, transfer));
                }
            }
        }

        // Apply transfers
        for (i, j, transfer) in transfers {
            // Positive transfer: i → j (i has more glucose)
            deposit_species_to_inventory(
                &mut self.plants[i].material_inventory,
                TerrariumSpecies::Glucose,
                -transfer,
            );
            deposit_species_to_inventory(
                &mut self.plants[j].material_inventory,
                TerrariumSpecies::Glucose,
                transfer,
            );
        }

        self.mycorrhizal_connection_count = connection_count;
    }

    pub(super) fn step_food_patches_native(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.fruits.is_empty() {
            return Ok(());
        }
        let daylight = self.daylight();
        let parent_states: Vec<(usize, usize, f32, f32)> = self
            .plants
            .iter()
            .map(|plant| {
                (
                    plant.x,
                    plant.y,
                    plant.physiology.health(),
                    plant.physiology.storage_carbon(),
                )
            })
            .collect();
        for fruit in self.fruits.iter_mut() {
            let Some((parent_x, parent_y)) = fruit
                .development
                .as_ref()
                .map(|development| (development.parent_x, development.parent_y))
            else {
                continue;
            };
            if !fruit.source.attached || !fruit.source.alive {
                continue;
            }
            let Some((_, _, parent_health, parent_storage_carbon)) = parent_states
                .iter()
                .find(|(x, y, _, _)| *x == parent_x && *y == parent_y)
                .copied()
            else {
                fruit.source.attached = false;
                fruit.source.z = 0;
                fruit.organ.vascular_supply = 0.0;
                fruit.organ.attachment_strength = 0.0;
                continue;
            };
            let flat = idx2(self.config.width, parent_x, parent_y);
            let humidity_idx = idx3(
                self.config.width,
                self.config.height,
                fruit.source.x.min(self.config.width - 1),
                fruit.source.y.min(self.config.height - 1),
                fruit.source.z.min(self.config.depth.max(1) - 1),
            );
            let surface_target = fruit_surface_inventory_target(
                fruit,
                self.moisture[flat],
                self.deep_moisture[flat],
                self.humidity[humidity_idx],
                self.microbial_biomass[flat],
            );
            if fruit.material_inventory.is_empty() {
                fruit.material_inventory = surface_target.scaled(0.90);
            } else {
                let _ = fruit.material_inventory.relax_toward(&surface_target, 0.22);
            }
            let Some(development) = fruit.development.as_mut() else {
                continue;
            };
            let _growth_drive = super::fruit_state::advance_attached_fruit(
                &mut fruit.source,
                &mut fruit.radius,
                development,
                &mut fruit.organ,
                daylight,
                parent_health,
                parent_storage_carbon,
                self.moisture[flat],
                self.deep_moisture[flat],
                self.shallow_nutrients[flat],
                eco_dt,
            );
            if let Some(reproduction) = fruit.reproduction.as_mut() {
                super::fruit_reproduction::step_fruit_reproduction(
                    reproduction,
                    fruit.source_genome.seed_mass,
                    fruit.source.ripeness,
                    fruit.source.sugar_content,
                    self.moisture[flat],
                    self.deep_moisture[flat],
                    self.shallow_nutrients[flat],
                    &fruit.material_inventory,
                    eco_dt,
                );
            }
        }
        let patch_remaining = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.sugar_content.max(0.0))
            .collect::<Vec<_>>();
        let previous_remaining = self
            .fruits
            .iter()
            .map(|fruit| fruit.previous_remaining.max(0.0))
            .collect::<Vec<_>>();
        let deposited_all = self
            .fruits
            .iter()
            .map(|fruit| fruit.deposited_all)
            .collect::<Vec<_>>();
        let has_fruit = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.alive)
            .collect::<Vec<_>>();
        let fruit_ripeness = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.ripeness)
            .collect::<Vec<_>>();
        let fruit_sugar = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.sugar_content)
            .collect::<Vec<_>>();
        let microbial = self
            .fruits
            .iter()
            .map(|fruit| {
                let flat = idx2(
                    self.config.width,
                    fruit.source.x.min(self.config.width - 1),
                    fruit.source.y.min(self.config.height - 1),
                );
                self.microbial_biomass[flat]
                    + self.microbial_biomass[flat] * 0.08
                    + self.microbial_biomass[flat] * 0.08
            })
            .collect::<Vec<_>>();
        let stepped = step_food_patches(
            eco_dt,
            &patch_remaining,
            &previous_remaining,
            &deposited_all,
            &has_fruit,
            &fruit_ripeness,
            &fruit_sugar,
            &microbial,
        )?;
        let mut decay_fluxes = Vec::new();
        let mut seed_release_candidates = Vec::new();
        for (idx, fruit) in self.fruits.iter_mut().enumerate() {
            let current = stepped.remaining[idx].max(0.0);
            fruit.source.sugar_content = current.min(stepped.sugar_content[idx]);
            fruit.source.alive = stepped.fruit_alive[idx];
            fruit.deposited_all = stepped.deposited_all[idx];
            fruit.composition = super::fruit_state::fruit_composition_from_parent(
                &fruit.source_genome,
                fruit.taxonomy_id,
                fruit.source.sugar_content,
                fruit.source.ripeness,
            );
            let x = fruit.source.x;
            let y = fruit.source.y;
            let radius = fruit.radius.round().max(1.0) as usize;
            let flat = idx2(
                self.config.width,
                x.min(self.config.width - 1),
                y.min(self.config.height - 1),
            );
            let humidity_idx = idx3(
                self.config.width,
                self.config.height,
                x.min(self.config.width - 1),
                y.min(self.config.height - 1),
                fruit.source.z.min(self.config.depth.max(1) - 1),
            );
            let surface_target = fruit_surface_inventory_target(
                fruit,
                self.moisture[flat],
                self.deep_moisture[flat],
                self.humidity[humidity_idx],
                microbial[idx],
            );
            if fruit.material_inventory.is_empty() {
                fruit.material_inventory = surface_target.scaled(0.90);
            } else {
                let _ = fruit.material_inventory.relax_toward(&surface_target, 0.18);
            }
            let detritus_total = stepped.decay_detritus[idx]
                + stepped.lost_detritus[idx]
                + stepped.final_detritus[idx];
            if stepped.decay_detritus[idx] > 0.0 {
                deposit_2d(
                    &mut self.litter_carbon,
                    self.config.width,
                    self.config.height,
                    x,
                    y,
                    radius,
                    stepped.decay_detritus[idx],
                );
            }
            if stepped.lost_detritus[idx] > 0.0 {
                deposit_2d(
                    &mut self.litter_carbon,
                    self.config.width,
                    self.config.height,
                    x,
                    y,
                    radius,
                    stepped.lost_detritus[idx],
                );
            }
            if stepped.final_detritus[idx] > 0.0 {
                deposit_2d(
                    &mut self.litter_carbon,
                    self.config.width,
                    self.config.height,
                    x,
                    y,
                    radius,
                    stepped.final_detritus[idx],
                );
            }
            if detritus_total > 0.0 {
                deposit_species_to_inventory(
                    &mut fruit.material_inventory,
                    TerrariumSpecies::Glucose,
                    detritus_total * 14.0,
                );
                deposit_species_to_inventory(
                    &mut fruit.material_inventory,
                    TerrariumSpecies::AminoAcidPool,
                    detritus_total * 2.2,
                );
                deposit_species_to_inventory(
                    &mut fruit.material_inventory,
                    TerrariumSpecies::NucleotidePool,
                    detritus_total * 1.3,
                );
                deposit_species_to_inventory(
                    &mut fruit.material_inventory,
                    TerrariumSpecies::MembranePrecursorPool,
                    detritus_total * 1.9,
                );
                deposit_species_to_inventory(
                    &mut fruit.material_inventory,
                    TerrariumSpecies::Ammonium,
                    detritus_total * 2.8,
                );
                spill_inventory_to_patch(
                    &mut self.substrate,
                    &mut fruit.material_inventory,
                    x,
                    y,
                    0,
                    radius.max(1),
                );
            }
            let decay_flux_driver = stepped.decay_detritus[idx]
                + stepped.lost_detritus[idx] * 0.32
                + stepped.final_detritus[idx] * 0.18;
            if decay_flux_driver > 0.0 {
                let microbial_drive = 0.65 + microbial[idx].max(0.0) * 6.0;
                let decay_co2_flux = clamp(decay_flux_driver * microbial_drive * 0.020, 0.0, 0.004);
                let decay_o2_flux = -decay_co2_flux * 0.88;
                let decay_humidity_flux = decay_co2_flux * 0.24;
                decay_fluxes.push((
                    x,
                    y,
                    fruit.source.z.min(self.config.depth.max(1) - 1),
                    radius.max(1),
                    decay_co2_flux,
                    decay_o2_flux,
                    decay_humidity_flux,
                ));
            }
            super::fruit_state::advance_detached_fruit(
                &mut fruit.source,
                &mut fruit.radius,
                &mut fruit.organ,
                self.moisture[flat],
                self.deep_moisture[flat],
                self.humidity[humidity_idx],
                microbial[idx],
                stepped.decay_detritus[idx],
                stepped.lost_detritus[idx],
                stepped.final_detritus[idx],
                eco_dt,
            );
            if fruit
                .reproduction
                .as_ref()
                .map(|reproduction| {
                    super::fruit_reproduction::fruit_ready_for_seed_release(
                        reproduction,
                        fruit.source.ripeness,
                    ) && super::fruit_state::fruit_ready_to_drop_seed(
                        &fruit.source,
                        &fruit.organ,
                        detritus_total,
                    )
                })
                .unwrap_or(false)
            {
                seed_release_candidates.push(idx);
            }
            fruit.previous_remaining = current;
        }
        for (x, y, z, radius, co2_flux, o2_flux, humidity_flux) in decay_fluxes {
            self.exchange_atmosphere_odorant(ATMOS_CO2_IDX, x, y, z, radius, co2_flux);
            self.exchange_atmosphere_odorant(ATMOS_O2_IDX, x, y, z, radius, o2_flux);
            self.exchange_atmosphere_humidity(x, y, z, radius, humidity_flux);
        }
        let mut released_seeds = Vec::new();
        for fruit_idx in seed_release_candidates {
            if self.seeds.len() + released_seeds.len() >= self.config.max_seeds {
                break;
            }
            if let Some(seed) = self.release_seed_from_fruit_idx(fruit_idx) {
                released_seeds.push(seed);
            }
        }
        self.seeds.extend(released_seeds);
        let removed_fruit_ids: Vec<u64> = self
            .fruits
            .iter()
            .filter(|fruit| {
                !(fruit.source.attached
                    || fruit.source.alive
                    || !fruit.deposited_all
                    || fruit
                        .reproduction
                        .as_ref()
                        .map(|reproduction| !reproduction.seed_released)
                        .unwrap_or(false))
            })
            .map(|fruit| fruit.identity.organism_id)
            .collect();
        for organism_id in removed_fruit_ids {
            self.mark_organism_dead(organism_id);
        }
        self.fruits.retain(|fruit| {
            fruit.source.attached
                || fruit.source.alive
                || !fruit.deposited_all
                || fruit
                    .reproduction
                    .as_ref()
                    .map(|reproduction| !reproduction.seed_released)
                    .unwrap_or(false)
        });
        self.sync_plant_fruit_loads();
        Ok(())
    }

    pub(super) fn step_seeds_native(&mut self, eco_dt: f32) -> Result<(), String> {
        self.step_litter_surface(eco_dt);
        if self.seeds.is_empty() {
            return Ok(());
        }
        let daylight = self.daylight();
        let depth = self.config.depth.max(1);
        let mut next_bank = Vec::new();
        let mut germinations = Vec::new();
        let mut seeds = std::mem::take(&mut self.seeds);

        for mut seed in seeds.drain(..) {
            let transport_x = seed.x.round().clamp(0.0, (self.config.width - 1) as f32) as usize;
            let transport_y = seed.y.round().clamp(0.0, (self.config.height - 1) as f32) as usize;
            let transport_flat = idx2(self.config.width, transport_x, transport_y);
            let litter_surface = self.litter_surface[transport_flat];

            seed_microsite::advance_seed_transport(
                &mut seed,
                eco_dt,
                self.config.width,
                self.config.height,
                seed_microsite::SeedTransportInputs {
                    cell_size_mm: self.config.cell_size_mm,
                    bioturbation_mm2_day: self.earthworm_population.bioturbation_rate
                        [transport_flat],
                    surface_moisture: self.moisture[transport_flat],
                    deep_moisture: self.deep_moisture[transport_flat],
                    cover_fraction: litter_surface.cover_fraction,
                    support_depth_mm: litter_surface.support_depth_mm,
                    pore_exposure: litter_surface.pore_exposure,
                    roughness_mm: litter_surface.roughness_mm,
                    collapse_rate: litter_surface.collapse_rate,
                },
                &mut self.rng,
            );

            let x = seed.x.round().clamp(0.0, (self.config.width - 1) as f32) as usize;
            let y = seed.y.round().clamp(0.0, (self.config.height - 1) as f32) as usize;
            let flat = idx2(self.config.width, x, y);
            let probe_z = seed_microsite::seed_probe_z(&seed, self.config.cell_size_mm, depth);
            let substrate_gid = probe_z * self.config.width * self.config.height + flat;
            let seed_inventory_target = material_inventory_target_from_patch(
                &self.substrate,
                x,
                y,
                probe_z,
                1,
                format!("seed-target:{x}:{y}:{probe_z}"),
            );
            let dormancy_t = clamp(seed.dormancy_s / 26_000.0, 0.0, 1.0);
            let reserve_t = clamp(seed.reserve_carbon / 0.20, 0.0, 1.0);
            let seed_sync_relax = clamp(
                0.22 + seed.microsite.surface_exposure * 0.10
                    + seed.cellular.hydration() * 0.08
                    + seed.cellular.energy_charge() * 0.08
                    + (1.0 - dormancy_t) * 0.10,
                0.12,
                0.54,
            );
            let (
                inventory_glucose_signal,
                inventory_amino_signal,
                inventory_nucleotide_signal,
                inventory_membrane_signal,
            ) = {
                let seed_inventory = &mut seed.material_inventory;
                sync_inventory_with_patch(
                    &mut self.substrate,
                    seed_inventory,
                    x,
                    y,
                    probe_z,
                    1,
                    &seed_inventory_target,
                    seed_sync_relax,
                )?;
                (
                    inventory_component_amount(seed_inventory, TerrariumSpecies::Glucose),
                    inventory_component_amount(seed_inventory, TerrariumSpecies::AminoAcidPool),
                    inventory_component_amount(seed_inventory, TerrariumSpecies::NucleotidePool),
                    inventory_component_amount(
                        seed_inventory,
                        TerrariumSpecies::MembranePrecursorPool,
                    ),
                )
            };
            let soil_glucose =
                self.substrate
                    .patch_mean_species(TerrariumSpecies::Glucose, x, y, probe_z, 1);
            let soil_oxygen_gas =
                self.substrate
                    .patch_mean_species(TerrariumSpecies::OxygenGas, x, y, probe_z, 1);
            let soil_carbon_dioxide = self.substrate.patch_mean_species(
                TerrariumSpecies::CarbonDioxide,
                x,
                y,
                probe_z,
                1,
            );
            let soil_ammonium =
                self.substrate
                    .patch_mean_species(TerrariumSpecies::Ammonium, x, y, probe_z, 1);
            let soil_nitrate =
                self.substrate
                    .patch_mean_species(TerrariumSpecies::Nitrate, x, y, probe_z, 1);
            let soil_proton =
                self.substrate
                    .patch_mean_species(TerrariumSpecies::Proton, x, y, probe_z, 1);
            let soil_atp_flux =
                self.substrate
                    .patch_mean_species(TerrariumSpecies::AtpFlux, x, y, probe_z, 1);
            let soil_amino_acids = self.substrate.patch_mean_species(
                TerrariumSpecies::AminoAcidPool,
                x,
                y,
                probe_z,
                1,
            );
            let soil_nucleotides = self.substrate.patch_mean_species(
                TerrariumSpecies::NucleotidePool,
                x,
                y,
                probe_z,
                1,
            );
            let soil_membrane_precursors = self.substrate.patch_mean_species(
                TerrariumSpecies::MembranePrecursorPool,
                x,
                y,
                probe_z,
                1,
            );
            let soil_dissolved_silicate = self.substrate.patch_mean_species(
                TerrariumSpecies::DissolvedSilicate,
                x,
                y,
                probe_z,
                1,
            );
            let soil_bicarbonate = self.substrate.patch_mean_species(
                TerrariumSpecies::BicarbonatePool,
                x,
                y,
                probe_z,
                1,
            );
            let soil_surface_proton_load = self.substrate.patch_mean_species(
                TerrariumSpecies::SurfaceProtonLoad,
                x,
                y,
                probe_z,
                1,
            );
            let soil_calcium_bicarbonate_complex = self.substrate.patch_mean_species(
                TerrariumSpecies::CalciumBicarbonateComplex,
                x,
                y,
                probe_z,
                1,
            );
            let soil_exchangeable_calcium = self.substrate.patch_mean_species(
                TerrariumSpecies::ExchangeableCalcium,
                x,
                y,
                probe_z,
                1,
            );
            let soil_exchangeable_magnesium = self.substrate.patch_mean_species(
                TerrariumSpecies::ExchangeableMagnesium,
                x,
                y,
                probe_z,
                1,
            );
            let soil_exchangeable_potassium = self.substrate.patch_mean_species(
                TerrariumSpecies::ExchangeablePotassium,
                x,
                y,
                probe_z,
                1,
            );
            let soil_exchangeable_sodium = self.substrate.patch_mean_species(
                TerrariumSpecies::ExchangeableSodium,
                x,
                y,
                probe_z,
                1,
            );
            let soil_exchangeable_aluminum = self.substrate.patch_mean_species(
                TerrariumSpecies::ExchangeableAluminum,
                x,
                y,
                probe_z,
                1,
            );
            let soil_aqueous_iron = self.substrate.patch_mean_species(
                TerrariumSpecies::AqueousIronPool,
                x,
                y,
                probe_z,
                1,
            );
            let microsite = seed_microsite::sample_seed_microsite(
                &seed,
                seed_microsite::SeedMicrositeInputs {
                    cell_size_mm: self.config.cell_size_mm,
                    daylight,
                    surface_moisture: self.moisture[flat],
                    deep_moisture: self.deep_moisture[flat],
                    nutrients: self.shallow_nutrients[flat],
                    symbionts: self.symbiont_biomass[flat],
                    canopy: self.canopy_cover[flat],
                    litter_carbon: self.litter_carbon[flat],
                    organic_matter: self.organic_matter[flat],
                    microbial_biomass: self.microbial_biomass[flat],
                    nitrifier_biomass: self.nitrifier_biomass[flat],
                    denitrifier_biomass: self.denitrifier_biomass[flat],
                    substrate_microbial_activity: self.substrate.microbial_activity[substrate_gid],
                    soil_glucose: soil_glucose + inventory_glucose_signal * 0.18,
                    soil_oxygen_gas,
                    soil_carbon_dioxide,
                    soil_ammonium,
                    soil_nitrate,
                    soil_proton,
                    soil_atp_flux,
                    soil_amino_acids: soil_amino_acids + inventory_amino_signal * 0.28,
                    soil_nucleotides: soil_nucleotides + inventory_nucleotide_signal * 0.28,
                    soil_membrane_precursors: soil_membrane_precursors
                        + inventory_membrane_signal * 0.24,
                    soil_dissolved_silicate,
                    soil_bicarbonate,
                    soil_surface_proton_load,
                    soil_calcium_bicarbonate_complex,
                    soil_exchangeable_calcium,
                    soil_exchangeable_magnesium,
                    soil_exchangeable_potassium,
                    soil_exchangeable_sodium,
                    soil_exchangeable_aluminum,
                    soil_aqueous_iron,
                },
            );
            let (
                inventory_water_take,
                inventory_amino_take,
                inventory_nucleotide_take,
                inventory_membrane_take,
            ) = {
                let hydration_gate = seed.cellular.hydration().clamp(0.0, 1.4);
                let energy_gate = seed.cellular.energy_charge().clamp(0.0, 1.4);
                let vitality_gate = seed.cellular.vitality().clamp(0.0, 1.2);
                let germination_gate = clamp(
                    (1.0 - dormancy_t) * 0.52 + energy_gate * 0.24 + reserve_t * 0.16,
                    0.08,
                    1.2,
                );
                let uptake_access = clamp(
                    microsite.oxygen_gas * 0.62
                        + microsite.moisture * 0.22
                        + microsite.surface_exposure * 0.18
                        + microsite.nutrients * 0.10
                        - microsite.microbial_pressure * 0.34
                        - microsite.acidity * 0.18,
                    0.02,
                    1.0,
                );
                let uptake_dt = eco_dt / 120.0 * uptake_access;
                let seed_inventory = &mut seed.material_inventory;
                let water_take = withdraw_species_from_inventory(
                    seed_inventory,
                    TerrariumSpecies::Water,
                    (0.003 + germination_gate * 0.008 + hydration_gate * 0.003) * uptake_dt,
                );
                let amino_take = withdraw_species_from_inventory(
                    seed_inventory,
                    TerrariumSpecies::AminoAcidPool,
                    (0.0016 + germination_gate * 0.0038 + reserve_t * 0.0018) * uptake_dt,
                );
                let nucleotide_take = withdraw_species_from_inventory(
                    seed_inventory,
                    TerrariumSpecies::NucleotidePool,
                    (0.0012 + germination_gate * 0.0028 + energy_gate * 0.0016) * uptake_dt,
                );
                let membrane_take = withdraw_species_from_inventory(
                    seed_inventory,
                    TerrariumSpecies::MembranePrecursorPool,
                    (0.0008 + germination_gate * 0.0020 + vitality_gate * 0.0012) * uptake_dt,
                );
                (water_take, amino_take, nucleotide_take, membrane_take)
            };

            seed.age_s += eco_dt;
            let seed_temp = self.temperature.get(flat).copied().unwrap_or(20.0)
                + self.weather.temperature_offset_c;
            seed.dormancy_s = seed_microsite::advance_seed_dormancy(
                seed.dormancy_s,
                eco_dt,
                microsite.moisture,
                microsite.local_light,
                seed_temp,
            );

            let seed_moisture = clamp(microsite.moisture + inventory_water_take * 0.45, 0.0, 1.25);
            let seed_deep_moisture = clamp(
                microsite.deep_moisture + inventory_water_take * 0.18,
                0.0,
                1.25,
            );
            let seed_nutrients = clamp(
                microsite.nutrients
                    + inventory_amino_take * 0.72
                    + inventory_nucleotide_take * 0.58
                    + inventory_membrane_take * 0.24,
                0.0,
                1.9,
            );
            let feedback = seed.cellular.step(
                eco_dt,
                seed_moisture,
                seed_deep_moisture,
                seed_nutrients,
                microsite.symbionts,
                microsite.canopy,
                microsite.litter,
                microsite.surface_exposure,
                microsite.oxygen_gas,
                microsite.carbon_dioxide,
                microsite.acidity,
                microsite.microbial_pressure,
                seed.dormancy_s,
                seed.reserve_carbon,
            );
            seed.reserve_carbon = feedback.reserve_carbon;

            if feedback.ready_to_germinate
                && seed_microsite::seed_can_emerge(
                    &seed,
                    &feedback,
                    &microsite,
                    self.config.cell_size_mm,
                )
                && self.plants.len() + germinations.len() < self.config.max_plants
                && !self.plants.iter().any(|plant| plant.x == x && plant.y == y)
            {
                let scale = seed_microsite::seedling_scale(seed.reserve_carbon, &feedback);
                germinations.push((
                    x,
                    y,
                    seed.genome,
                    scale,
                    seed.material_inventory,
                    seed.identity,
                ));
            } else if seed_microsite::seed_should_persist(
                seed.age_s,
                seed.reserve_carbon,
                &feedback,
                &microsite,
            ) {
                next_bank.push(seed);
            } else {
                let detritus = seed_microsite::seed_detritus_return(
                    seed.reserve_carbon,
                    &feedback,
                    &microsite,
                );
                self.litter_carbon[flat] =
                    (self.litter_carbon[flat] + detritus.litter_carbon).max(0.0);
                self.organic_matter[flat] =
                    (self.organic_matter[flat] + detritus.organic_matter).max(0.0);
                deposit_species_to_inventory(
                    &mut seed.material_inventory,
                    TerrariumSpecies::Glucose,
                    detritus.glucose,
                );
                deposit_species_to_inventory(
                    &mut seed.material_inventory,
                    TerrariumSpecies::AminoAcidPool,
                    detritus.amino_acids,
                );
                deposit_species_to_inventory(
                    &mut seed.material_inventory,
                    TerrariumSpecies::NucleotidePool,
                    detritus.nucleotides,
                );
                deposit_species_to_inventory(
                    &mut seed.material_inventory,
                    TerrariumSpecies::MembranePrecursorPool,
                    detritus.membrane_precursors,
                );
                deposit_species_to_inventory(
                    &mut seed.material_inventory,
                    TerrariumSpecies::Ammonium,
                    detritus.ammonium,
                );
                deposit_species_to_inventory(
                    &mut seed.material_inventory,
                    TerrariumSpecies::CarbonDioxide,
                    detritus.carbon_dioxide,
                );
                spill_inventory_to_patch(
                    &mut self.substrate,
                    &mut seed.material_inventory,
                    x,
                    y,
                    probe_z,
                    1,
                );
            }
        }
        self.seeds = next_bank;
        for (x, y, genome, scale, mut material_inventory, identity) in germinations {
            if let Ok(id) = self.add_plant_with_identity(x, y, genome, scale, Some(identity)) {
                material_inventory.name = format!("plant:{x}:{y}");
                self.plants[id].material_inventory = material_inventory;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn added_fruit_inherits_parent_identity_and_composition() {
        let mut world = TerrariumWorld::demo(47, false).unwrap();
        world.plants.clear();
        world.fruits.clear();
        world.seeds.clear();
        world.flies.clear();
        world.explicit_microbes.clear();
        world.waters.clear();

        let x = world.config.width / 2;
        let y = world.config.height / 2;
        let genome = crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        let plant_idx = world
            .add_plant(x, y, Some(genome.clone()), Some(1.25))
            .unwrap();
        let parent_taxonomy = world.plants[plant_idx].genome.taxonomy_id;
        world.add_fruit(x + 1, y, 0.85, Some(1.0));

        assert!(
            !world.fruits.is_empty(),
            "adding fruit near a plant should create an explicit fruit patch"
        );
        assert!(
            world
                .fruits
                .iter()
                .any(|fruit| fruit.taxonomy_id == parent_taxonomy
                    && fruit.source_genome.taxonomy_id == parent_taxonomy
                    && fruit.composition.sugar_fraction > 0.0),
            "fruit should inherit the nearby parent plant identity and explicit composition"
        );
    }

    #[test]
    fn attached_fruit_grows_over_time_under_favorable_conditions() {
        let mut world = TerrariumWorld::demo(53, false).unwrap();
        world.plants.clear();
        world.fruits.clear();
        world.seeds.clear();
        world.flies.clear();
        world.explicit_microbes.clear();
        world.waters.clear();
        world.time_s = 12.0 * 3600.0;

        let x = world.config.width / 2;
        let y = world.config.height / 2;
        let genome = crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        let plant_idx = world.add_plant(x, y, Some(genome), Some(1.3)).unwrap();
        world.plants[plant_idx].physiology = crate::plant_organism::PlantOrganismSim::new(
            18.0, 8.0, 5.8, 1.08, 0.96, 0.88, 1.22, 0.74, 1.02, 1.0, 0.62, 1.08, 0.12, 0.42, 0.55,
            0.48, 1.20, 0.90, 0.72, 60.0, 7200.0,
        );
        let flat = idx2(world.config.width, x, y);
        world.moisture[flat] = 0.96;
        world.deep_moisture[flat] = 0.92;
        world.shallow_nutrients[flat] = 0.84;

        world.add_attached_fruit(x + 1, y, x, y, 0.9, Some(1.0));
        let initial_radius = world.fruits[0].radius;
        let initial_ripeness = world.fruits[0].source.ripeness;
        let initial_sugar = world.fruits[0].source.sugar_content;

        for _ in 0..12 {
            world.step_food_patches_native(120.0).unwrap();
        }

        assert!(world.fruits[0].radius > initial_radius);
        assert!(world.fruits[0].source.ripeness > initial_ripeness);
        assert!(world.fruits[0].source.sugar_content > initial_sugar);
        assert!(
            world.fruits[0].source.attached || world.fruits[0].source.ripeness > 0.8,
            "fruit should still be attached while developing or have reached mature ripeness"
        );
    }

    #[test]
    fn detached_rotting_fruit_releases_seed_from_embryo_state() {
        let mut world = TerrariumWorld::demo(61, false).unwrap();
        world.plants.clear();
        world.fruits.clear();
        world.seeds.clear();
        world.flies.clear();
        world.explicit_microbes.clear();
        world.waters.clear();

        let mut maternal =
            crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        maternal.leaf_efficiency = 0.88;
        let mut donor = maternal.clone();
        donor.leaf_efficiency = 1.31;
        donor.root_depth_bias = 0.92;

        let parent_x = world.config.width / 2;
        let parent_y = world.config.height / 2;
        let _ = world.add_plant(parent_x, parent_y, Some(maternal.clone()), Some(1.2));
        let _ = world.add_plant(parent_x + 1, parent_y, Some(donor.clone()), Some(1.1));
        world.add_attached_fruit(
            parent_x + 1,
            parent_y + 1,
            parent_x,
            parent_y,
            1.0,
            Some(1.0),
        );
        let fruit = world.fruits.get_mut(0).unwrap();
        fruit.source.ripeness = 0.92;
        fruit.source.sugar_content = 0.88;
        fruit.source.attached = false;
        fruit.source.z = 0;
        fruit.organ.seed_exposure = 0.84;
        fruit.organ.rot_progress = 0.62;
        fruit.organ.flesh_integrity = 0.24;
        fruit.organ.peel_integrity = 0.30;
        if let Some(reproduction) = fruit.reproduction.as_mut() {
            reproduction.reserve_carbon = 0.09;
        }

        world.step_food_patches_native(120.0).unwrap();

        let seed = world
            .seeds
            .first()
            .expect("detached rotting fruit should release a seed");

        assert!(
            (seed.genome.leaf_efficiency - maternal.leaf_efficiency).abs() > 1.0e-4
                || (seed.genome.root_depth_bias - maternal.root_depth_bias).abs() > 1.0e-4
        );
        assert!(seed.reserve_carbon > 0.03);
        assert!(
            world.fruits[0]
                .reproduction
                .as_ref()
                .map(|reproduction| reproduction.seed_released)
                .unwrap_or(false),
            "fruit lifecycle should own the seed release event"
        );
    }

    #[test]
    fn earthworm_bioturbation_updates_seed_microsite_state() {
        let mut world = TerrariumWorld::demo(71, false).unwrap();
        world.plants.clear();
        world.fruits.clear();
        world.seeds.clear();
        world.flies.clear();
        world.explicit_microbes.clear();
        world.waters.clear();

        let x = world.config.width / 2;
        let y = world.config.height / 2;
        let flat = idx2(world.config.width, x, y);
        world.moisture[flat] = 0.88;
        world.deep_moisture[flat] = 0.76;
        world.litter_carbon[flat] = 0.18;
        world.organic_matter[flat] = 0.92;
        world.earthworm_population.bioturbation_rate[flat] = 5.0;

        let genome = crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        world.add_seed(
            x as f32 + 0.1,
            y as f32 + 0.1,
            genome,
            Some(0.16),
            Some(8_000.0),
        );

        for _ in 0..12 {
            world.step_seeds_native(120.0).unwrap();
        }

        assert!(!world.seeds.is_empty());
        let seed = &world.seeds[0];
        assert!(seed.microsite.burial_depth_mm > 0.0);
        assert!(seed.microsite.surface_exposure < 1.0);
        assert!(seed.pose.offset_mm[1] < 0.0);
    }

    #[test]
    fn seed_germinates_from_cellular_feedback_under_favorable_conditions() {
        let mut world = TerrariumWorld::demo(73, false).unwrap();
        world.plants.clear();
        world.fruits.clear();
        world.seeds.clear();
        world.flies.clear();
        world.explicit_microbes.clear();
        world.waters.clear();
        world.time_s = 12.0 * 3600.0;

        let x = world.config.width / 2;
        let y = world.config.height / 2;
        let flat = idx2(world.config.width, x, y);
        let genome = crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        world.moisture[flat] = 0.98;
        world.deep_moisture[flat] = 0.84;
        world.shallow_nutrients[flat] = 0.36;
        world.symbiont_biomass[flat] = 0.28;
        world.canopy_cover[flat] = 0.08;
        world.litter_carbon[flat] = 0.16;
        world.organic_matter[flat] = 0.72;
        world.add_seed(x as f32, y as f32, genome, Some(0.18), Some(0.0));

        for _ in 0..10 {
            world.step_seeds_native(120.0).unwrap();
            if !world.plants.is_empty() {
                break;
            }
        }

        assert!(
            !world.plants.is_empty(),
            "seed did not germinate from cellular feedback under favorable local conditions"
        );
        assert!(world.seeds.is_empty());
    }

    #[test]
    fn buried_seed_waits_for_surface_emergence_capacity() {
        let mut world = TerrariumWorld::demo(77, false).unwrap();
        world.plants.clear();
        world.fruits.clear();
        world.seeds.clear();
        world.flies.clear();
        world.explicit_microbes.clear();
        world.waters.clear();
        world.time_s = 12.0 * 3600.0;

        let x = world.config.width / 2;
        let y = world.config.height / 2;
        let flat = idx2(world.config.width, x, y);
        let mut genome = crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        genome.seed_mass = 0.022;
        genome.root_depth_bias = 0.08;
        world.moisture[flat] = 0.98;
        world.deep_moisture[flat] = 0.90;
        world.shallow_nutrients[flat] = 0.36;
        world.symbiont_biomass[flat] = 0.30;
        world.canopy_cover[flat] = 0.06;
        world.litter_carbon[flat] = 0.18;
        world.organic_matter[flat] = 0.76;
        world.earthworm_population.bioturbation_rate[flat] = 0.0;
        world.add_seed(x as f32, y as f32, genome, Some(0.18), Some(0.0));

        world.seeds[0].microsite.burial_depth_mm = world.config.cell_size_mm * 0.60;
        world.seeds[0].microsite.surface_exposure = 0.16;
        world.seeds[0].pose.offset_mm[1] = -world.seeds[0].microsite.burial_depth_mm;

        for _ in 0..24 {
            let dormancy_s = world.seeds[0].dormancy_s;
            let reserve_carbon = world.seeds[0].reserve_carbon;
            let feedback = world.seeds[0].cellular.step(
                120.0,
                0.98,
                0.90,
                0.36,
                0.30,
                0.06,
                0.18,
                0.95,
                0.42,
                0.10,
                0.06,
                0.04,
                dormancy_s,
                reserve_carbon,
            );
            world.seeds[0].reserve_carbon = feedback.reserve_carbon;
            if feedback.ready_to_germinate {
                break;
            }
        }

        assert!(
            world.seeds[0].cellular.last_feedback().ready_to_germinate,
            "test setup failed to bring the buried seed to internal germination readiness"
        );

        world.step_seeds_native(120.0).unwrap();
        assert!(
            world.plants.is_empty(),
            "small buried seed should stay in the bank even after its internal tissues are ready"
        );
        assert!(
            !world.seeds.is_empty(),
            "buried seed should remain in the bank"
        );

        world.seeds[0].microsite.burial_depth_mm = world.config.cell_size_mm * 0.05;
        world.seeds[0].microsite.surface_exposure = 0.92;
        world.seeds[0].pose.offset_mm[1] = -world.seeds[0].microsite.burial_depth_mm;
        world.step_seeds_native(120.0).unwrap();

        assert!(
            !world.plants.is_empty(),
            "once the seed is close enough to the surface, the same germinating seed should emerge"
        );
        assert!(world.seeds.is_empty());
    }

    #[test]
    fn drying_sparse_cover_can_reexpose_and_emerge_buried_seed() {
        let mut world = TerrariumWorld::demo(78, false).unwrap();
        world.plants.clear();
        world.fruits.clear();
        world.seeds.clear();
        world.flies.clear();
        world.explicit_microbes.clear();
        world.waters.clear();
        world.time_s = 12.0 * 3600.0;

        let x = world.config.width / 2;
        let y = world.config.height / 2;
        let flat = idx2(world.config.width, x, y);
        let mut genome = crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        genome.seed_mass = 0.022;
        genome.root_depth_bias = 0.08;
        world.moisture[flat] = 0.24;
        world.deep_moisture[flat] = 0.88;
        world.shallow_nutrients[flat] = 0.34;
        world.symbiont_biomass[flat] = 0.24;
        world.canopy_cover[flat] = 0.08;
        world.litter_carbon[flat] = 0.01;
        world.organic_matter[flat] = 0.06;
        world.earthworm_population.bioturbation_rate[flat] = 2.8;
        world.add_seed(x as f32, y as f32, genome, Some(0.18), Some(0.0));

        for _ in 0..24 {
            let dormancy_s = world.seeds[0].dormancy_s;
            let reserve_carbon = world.seeds[0].reserve_carbon;
            let feedback = world.seeds[0].cellular.step(
                120.0,
                0.98,
                0.90,
                0.36,
                0.30,
                0.06,
                0.18,
                0.95,
                0.42,
                0.10,
                0.06,
                0.04,
                dormancy_s,
                reserve_carbon,
            );
            world.seeds[0].reserve_carbon = feedback.reserve_carbon;
            if feedback.ready_to_germinate {
                break;
            }
        }
        assert!(world.seeds[0].cellular.last_feedback().ready_to_germinate);

        world.seeds[0].microsite.burial_depth_mm = world.config.cell_size_mm * 0.60;
        world.seeds[0].microsite.surface_exposure = 0.16;
        world.seeds[0].pose.offset_mm[1] = -world.seeds[0].microsite.burial_depth_mm;
        let start_depth = world.seeds[0].microsite.burial_depth_mm;
        let mut reexposed = false;

        for _ in 0..120 {
            world.step_seeds_native(120.0).unwrap();
            if !world.seeds.is_empty()
                && world.seeds[0].microsite.burial_depth_mm + 1.0e-4 < start_depth
            {
                reexposed = true;
            }
            if !world.plants.is_empty() {
                break;
            }
        }

        assert!(
            reexposed,
            "dry sparse cover should be able to reduce burial depth for at least some steps"
        );
        assert!(
            !world.plants.is_empty(),
            "a resurfacing ready seed should eventually emerge through the native terrarium loop"
        );
        assert!(world.seeds.is_empty());
    }

    #[test]
    fn hostile_seed_microsite_returns_detritus_to_local_soil() {
        let mut world = TerrariumWorld::demo(79, false).unwrap();
        world.plants.clear();
        world.fruits.clear();
        world.seeds.clear();
        world.flies.clear();
        world.explicit_microbes.clear();
        world.waters.clear();

        let x = world.config.width / 2;
        let y = world.config.height / 2;
        let flat = idx2(world.config.width, x, y);
        let genome = crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        world.moisture[flat] = 0.94;
        world.deep_moisture[flat] = 0.98;
        world.shallow_nutrients[flat] = 0.06;
        world.symbiont_biomass[flat] = 0.0;
        world.canopy_cover[flat] = 0.86;
        world.litter_carbon[flat] = 0.22;
        world.organic_matter[flat] = 0.88;
        world.microbial_biomass[flat] = 2.6;
        world.nitrifier_biomass[flat] = 1.2;
        world.denitrifier_biomass[flat] = 1.0;
        world.earthworm_population.bioturbation_rate[flat] = 0.0;

        world.add_seed(x as f32, y as f32, genome, Some(0.07), Some(0.0));
        world.seeds[0].microsite.burial_depth_mm = world.config.cell_size_mm * 0.68;
        world.seeds[0].microsite.surface_exposure = 0.12;
        world.seeds[0].pose.offset_mm[1] = -world.seeds[0].microsite.burial_depth_mm;

        let probe_z = seed_microsite::seed_probe_z(
            &world.seeds[0],
            world.config.cell_size_mm,
            world.config.depth,
        );
        for yy in y.saturating_sub(1)..=(y + 1).min(world.config.height - 1) {
            for xx in x.saturating_sub(1)..=(x + 1).min(world.config.width - 1) {
                for species_value in [
                    (TerrariumSpecies::OxygenGas, 0.0),
                    (TerrariumSpecies::CarbonDioxide, 1.8),
                    (TerrariumSpecies::Proton, 1.6),
                    (TerrariumSpecies::Glucose, 1.4),
                    (TerrariumSpecies::Ammonium, 0.9),
                    (TerrariumSpecies::Nitrate, 0.7),
                    (TerrariumSpecies::AtpFlux, 0.3),
                ] {
                    let idx = world.substrate.index(species_value.0, xx, yy, probe_z);
                    world.substrate.current[idx] = species_value.1;
                    world.substrate.next[idx] = species_value.1;
                }
                let gid = probe_z * world.config.width * world.config.height
                    + idx2(world.config.width, xx, yy);
                world.substrate.microbial_activity[gid] = 2.4;
            }
        }

        let litter_before = world.litter_carbon[flat];
        let organic_before = world.organic_matter[flat];
        let glucose_before =
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::Glucose, x, y, probe_z, 1);
        let amino_before =
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::AminoAcidPool, x, y, probe_z, 1);
        let nucleotide_before =
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::NucleotidePool, x, y, probe_z, 1);
        let ammonium_before =
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::Ammonium, x, y, probe_z, 1);

        for _ in 0..120 {
            world.step_seeds_native(600.0).unwrap();
            if world.seeds.is_empty() {
                break;
            }
        }

        assert!(
            world.seeds.is_empty(),
            "hostile buried chemistry should eventually rot the seed out of the bank"
        );
        assert!(world.plants.is_empty());
        assert!(world.litter_carbon[flat] > litter_before);
        assert!(world.organic_matter[flat] > organic_before);
        assert!(
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::Glucose, x, y, probe_z, 1)
                > glucose_before
        );
        assert!(
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::AminoAcidPool, x, y, probe_z, 1)
                > amino_before
        );
        assert!(
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::NucleotidePool, x, y, probe_z, 1)
                > nucleotide_before
        );
        assert!(
            world
                .substrate
                .patch_mean_species(TerrariumSpecies::Ammonium, x, y, probe_z, 1)
                > ammonium_before
        );
    }

    #[test]
    fn exudate_dissolves_silicate_mineral() {
        // Test the physics formula directly: Hill(exudate, 0.01, 2.0) drives dissolution
        // rate proportional to mineral availability.
        use crate::botany::physiology_bridge::hill;

        // Strong exudates → high dissolution drive
        let strong = hill(0.05, 0.01, 2.0);
        assert!(strong > 0.9, "High exudate should saturate dissolution drive: {strong}");

        // Weak exudates → low dissolution drive
        let weak = hill(0.002, 0.01, 2.0);
        assert!(weak < 0.1, "Low exudate should have minimal dissolution: {weak}");

        // Zero exudates → no dissolution (gated by exudates > 0.001)
        let none = hill(0.0005, 0.01, 2.0);
        assert!(none < 0.003, "Below threshold exudate should produce near-zero: {none}");

        // Dissolution amount scales with mineral availability
        let mineral_rich = 0.5;
        let mineral_poor = 0.01;
        let dt = 0.05;
        let dissolved_rich = mineral_rich * strong * dt * 0.002;
        let dissolved_poor = mineral_poor * strong * dt * 0.002;
        assert!(
            dissolved_rich > dissolved_poor * 40.0,
            "More mineral → more dissolution: rich={dissolved_rich}, poor={dissolved_poor}"
        );
    }

    #[test]
    fn exudate_deposits_protons() {
        // Test that organic acid dissociation produces protons proportional
        // to exudate strength (Hill kinetics).
        use crate::botany::physiology_bridge::hill;

        let dt = 0.05;
        let strong_exudate = hill(0.04, 0.01, 2.0);
        let proton_deposit = strong_exudate * dt * 0.001;
        assert!(
            proton_deposit > 0.0,
            "Exudate dissociation should produce protons: {proton_deposit}"
        );

        let weak_exudate = hill(0.001, 0.01, 2.0);
        let proton_weak = weak_exudate * dt * 0.001;
        assert!(
            proton_deposit > proton_weak * 5.0,
            "Stronger exudates should donate more protons: strong={proton_deposit}, weak={proton_weak}"
        );
    }

    #[test]
    fn mycorrhizal_equalizes_glucose() {
        use crate::terrarium::material_exchange::{
            deposit_species_to_inventory, inventory_component_amount,
        };

        let mut world = TerrariumWorld::demo(70, false).unwrap();
        world.plants.clear();
        world.fruits.clear();
        world.seeds.clear();
        world.flies.clear();
        world.explicit_microbes.clear();
        world.waters.clear();

        // Place two plants close together with high symbiosis affinity
        let x1 = world.config.width / 2;
        let y1 = world.config.height / 2;
        let x2 = x1 + 1; // adjacent
        let y2 = y1;
        let mut genome1 =
            crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        genome1.symbiosis_affinity = 1.5;
        let mut genome2 =
            crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        genome2.symbiosis_affinity = 1.5;

        let p1 = world
            .add_plant(x1, y1, Some(genome1), Some(1.3))
            .unwrap();
        let p2 = world
            .add_plant(x2, y2, Some(genome2), Some(1.3))
            .unwrap();

        // Give plant 1 lots of glucose, plant 2 very little
        deposit_species_to_inventory(
            &mut world.plants[p1].material_inventory,
            TerrariumSpecies::Glucose,
            2.0,
        );

        let g1_before =
            inventory_component_amount(&world.plants[p1].material_inventory, TerrariumSpecies::Glucose);
        let g2_before =
            inventory_component_amount(&world.plants[p2].material_inventory, TerrariumSpecies::Glucose);
        let gap_before = (g1_before - g2_before).abs();

        // Run mycorrhizal exchange
        for _ in 0..100 {
            world.step_mycorrhizal_exchange(1.0);
        }

        let g1_after =
            inventory_component_amount(&world.plants[p1].material_inventory, TerrariumSpecies::Glucose);
        let g2_after =
            inventory_component_amount(&world.plants[p2].material_inventory, TerrariumSpecies::Glucose);
        let gap_after = (g1_after - g2_after).abs();

        assert!(
            gap_after < gap_before,
            "Mycorrhizal exchange should reduce glucose gradient: gap {gap_before:.4} -> {gap_after:.4}"
        );
    }

    #[test]
    fn mycorrhizal_requires_root_overlap() {
        use crate::terrarium::material_exchange::{
            deposit_species_to_inventory, inventory_component_amount,
        };

        let mut world = TerrariumWorld::demo(71, false).unwrap();
        world.plants.clear();
        world.fruits.clear();
        world.seeds.clear();
        world.flies.clear();
        world.explicit_microbes.clear();
        world.waters.clear();

        // Place two plants far apart — no root overlap
        let x1 = 1;
        let y1 = 1;
        let x2 = world.config.width - 2;
        let y2 = world.config.height - 2;
        let mut genome1 =
            crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        genome1.symbiosis_affinity = 1.5;
        let mut genome2 =
            crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        genome2.symbiosis_affinity = 1.5;

        let p1 = world
            .add_plant(x1, y1, Some(genome1), Some(0.5))
            .unwrap();
        let p2 = world
            .add_plant(x2, y2, Some(genome2), Some(0.5))
            .unwrap();

        // Give plant 1 glucose
        deposit_species_to_inventory(
            &mut world.plants[p1].material_inventory,
            TerrariumSpecies::Glucose,
            2.0,
        );

        let g2_before =
            inventory_component_amount(&world.plants[p2].material_inventory, TerrariumSpecies::Glucose);

        for _ in 0..100 {
            world.step_mycorrhizal_exchange(1.0);
        }

        let g2_after =
            inventory_component_amount(&world.plants[p2].material_inventory, TerrariumSpecies::Glucose);

        assert!(
            (g2_after - g2_before).abs() < 1e-5,
            "Distant plants should not exchange glucose: {g2_before:.6} -> {g2_after:.6}"
        );
    }
}
