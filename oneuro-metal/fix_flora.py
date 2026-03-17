#!/usr/bin/env python3
"""Fix remaining compilation errors in flora.rs.

All changes are in flora.rs only (linter-safe). Fixes:
1. physiology.step() arg count (29→26): remove phosphate, local_temp, air_o2_factor
2. step_seeds_native: replace cellular simulation with simple germination
3. Remove speciation logic (species_id, trait_distance, next_species_id)
4. Remove EcologyTelemetryEvent::{FruitProduced,SeedGerminated}
5. Remove TerrariumSeed cellular/pose fields from construction
"""
import os

SRC = os.path.dirname(os.path.abspath(__file__)) + "/src"

def read(path):
    with open(path) as f:
        return f.read()

def write(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote {path} ({len(content)} bytes)")

fl = read(f"{SRC}/terrarium_world/flora.rs")

# ====================================================================
# Fix 1: physiology.step() — remove 3 extra args (phosphate, local_temp, air_o2_factor)
# ====================================================================
print("Fix 1: physiology.step() arg count ...")

# Remove line: extraction.nutrient_take * 0.067,
fl = fl.replace(
    "                    extraction.nutrient_take,\n"
    "                    extraction.nutrient_take * 0.067,\n"
    "                    local_light,",
    "                    extraction.nutrient_take,\n"
    "                    local_light,",
)

# Remove line: local_temp,
fl = fl.replace(
    "                    temp_factor,\n"
    "                    local_temp,\n"
    "                    root_energy_gate,",
    "                    temp_factor,\n"
    "                    root_energy_gate,",
)

# Remove line: air_o2_factor,
fl = fl.replace(
    "                    air_co2_factor,\n"
    "                    air_o2_factor,\n"
    "                    stomatal_open,",
    "                    air_co2_factor,\n"
    "                    stomatal_open,",
)

# ====================================================================
# Fix 2: Remove air_o2_factor computation (no longer used)
# ====================================================================
print("Fix 2: Remove unused air_o2_factor ...")
fl = fl.replace(
    """            let local_air_o2 = self.sample_odorant_patch(
                ATMOS_O2_IDX,
                x,
                y,
                canopy_z,
                (canopy_radius.max(2) / 2).max(1),
            );
            let air_co2_factor = clamp(local_air_co2 / ATMOS_CO2_BASELINE, 0.35, 1.8);
            let air_o2_factor = clamp(local_air_o2 / ATMOS_O2_BASELINE, 0.25, 1.5);""",
    """            let air_co2_factor = clamp(local_air_co2 / ATMOS_CO2_BASELINE, 0.35, 1.8);""",
)

# ====================================================================
# Fix 3: TerrariumSeed construction — remove cellular and pose fields
# ====================================================================
print("Fix 3: TerrariumSeed construction ...")
fl = fl.replace(
    """                queued_seeds.push(TerrariumSeed {
                    x: sx as f32,
                    y: sy as f32,
                    dormancy_s: dormancy,
                    reserve_carbon: reserve,
                    age_s: 0.0,
                    genome: child_genome,
                    cellular: SeedCellularStateSim::new(seed_mass, reserve, dormancy),
                    pose: TerrariumSeedPose::default(),
                });""",
    """                queued_seeds.push(TerrariumSeed {
                    x: sx as f32,
                    y: sy as f32,
                    dormancy_s: dormancy,
                    reserve_carbon: reserve,
                    age_s: 0.0,
                    genome: child_genome,
                });""",
)

# ====================================================================
# Fix 4: Remove speciation logic
# ====================================================================
print("Fix 4: Remove speciation logic ...")

# Remove the speciation check block
fl = fl.replace(
    """                let mut child_genome = genome.mutate(&mut self.rng);
                // Speciation: if trait drift exceeds threshold, assign new species.
                if genome.trait_distance(&child_genome) > PLANT_SPECIATION_THRESHOLD {
                    child_genome.species_id = self.next_species_id;
                    self.next_species_id += 1;
                }
                queued_seeds.push(TerrariumSeed {""",
    """                let child_genome = genome.mutate(&mut self.rng);
                queued_seeds.push(TerrariumSeed {""",
)

# ====================================================================
# Fix 5: Remove FruitProduced event
# ====================================================================
print("Fix 5: Remove FruitProduced event ...")
fl = fl.replace(
    """            self.add_fruit(fx, fy, size, Some(volatile_scale));
            self.ecology_events.push(super::EcologyTelemetryEvent::FruitProduced {
                x: fx as f32,
                y: fy as f32,
                sugar_content: size,
            });""",
    """            self.add_fruit(fx, fy, size, Some(volatile_scale));""",
)

# ====================================================================
# Fix 6: Replace step_seeds_native with simplified version
#         Remove cellular sim, SeedTissue, and SeedGerminated event
# ====================================================================
print("Fix 6: Simplify step_seeds_native ...")

OLD_SEEDS = """    pub(super) fn step_seeds_native(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.seeds.is_empty() {
            return Ok(());
        }
        let mut dormancy = Vec::with_capacity(self.seeds.len());
        let mut age = Vec::with_capacity(self.seeds.len());
        let mut reserve = Vec::with_capacity(self.seeds.len());
        let mut affinity = Vec::with_capacity(self.seeds.len());
        let mut shade = Vec::with_capacity(self.seeds.len());
        let mut moisture = Vec::with_capacity(self.seeds.len());
        let mut deep_moisture = Vec::with_capacity(self.seeds.len());
        let mut nutrients = Vec::with_capacity(self.seeds.len());
        let mut symbionts = Vec::with_capacity(self.seeds.len());
        let mut canopy = Vec::with_capacity(self.seeds.len());
        let mut litter = Vec::with_capacity(self.seeds.len());
        let mut positions = Vec::with_capacity(self.seeds.len());

        for seed in &self.seeds {
            let x = seed.x.round().clamp(0.0, (self.config.width - 1) as f32) as usize;
            let y = seed.y.round().clamp(0.0, (self.config.height - 1) as f32) as usize;
            let flat = idx2(self.config.width, x, y);
            positions.push((x, y));
            dormancy.push(seed.dormancy_s);
            age.push(seed.age_s);
            reserve.push(seed.reserve_carbon);
            affinity.push(seed.genome.symbiosis_affinity);
            shade.push(seed.genome.shade_tolerance);
            moisture.push(self.moisture[flat]);
            deep_moisture.push(self.deep_moisture[flat]);
            nutrients.push(self.shallow_nutrients[flat]);
            symbionts.push(self.symbiont_biomass[flat]);
            canopy.push(self.canopy_cover[flat]);
            litter.push(self.litter_carbon[flat]);
        }

        let stepped = step_seed_bank(
            eco_dt,
            self.daylight(),
            self.plants.len(),
            self.config.max_plants,
            &dormancy,
            &age,
            &reserve,
            &affinity,
            &shade,
            &moisture,
            &deep_moisture,
            &nutrients,
            &symbionts,
            &canopy,
            &litter,
        )?;

        let mut next_bank = Vec::new();
        let mut germinations = Vec::new();
        for (idx, mut seed) in self.seeds.drain(..).enumerate() {
            seed.age_s = stepped.age_s[idx];
            seed.dormancy_s = stepped.dormancy_s[idx];
            let feedback = seed.cellular.step(
                eco_dt,
                moisture[idx],
                deep_moisture[idx],
                nutrients[idx],
                symbionts[idx],
                canopy[idx],
                litter[idx],
                seed.dormancy_s,
                seed.reserve_carbon,
            );
            seed.reserve_carbon = feedback.reserve_carbon;
            let coat = seed.cellular.cluster_snapshot(SeedTissue::Coat);
            let endosperm = seed.cellular.cluster_snapshot(SeedTissue::Endosperm);
            let radicle = seed.cellular.cluster_snapshot(SeedTissue::Radicle);
            let cotyledon = seed.cellular.cluster_snapshot(SeedTissue::Cotyledon);
            let emergent_germination = feedback.ready_to_germinate
                && self.plants.len() + germinations.len() < self.config.max_plants;
            if emergent_germination {
                let (x, y) = positions[idx];
                let total_cells = coat.cell_count
                    + endosperm.cell_count
                    + radicle.cell_count
                    + cotyledon.cell_count;
                let scale = clamp(
                    0.40 + feedback.germination_drive * 0.22
                        + feedback.vitality * 0.18
                        + feedback.radicle_extension * 0.14
                        + total_cells.sqrt() * 0.012,
                    0.45,
                    1.15,
                );
                germinations.push((x, y, seed.genome, scale));
            } else if stepped.keep[idx]
                && feedback.reserve_carbon > 0.002
                && feedback.vitality > 0.03
            {
                next_bank.push(seed);
            }
        }
        self.seeds = next_bank;
        for (x, y, genome, scale) in germinations {
            let sid = genome.species_id;
            let _ = self.add_plant(x, y, Some(genome), Some(scale));
            self.ecology_events.push(super::EcologyTelemetryEvent::SeedGerminated {
                x: x as f32,
                y: y as f32,
                species_id: sid,
            });
        }
        Ok(())
    }"""

NEW_SEEDS = """    pub(super) fn step_seeds_native(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.seeds.is_empty() {
            return Ok(());
        }
        let mut dormancy = Vec::with_capacity(self.seeds.len());
        let mut age = Vec::with_capacity(self.seeds.len());
        let mut reserve = Vec::with_capacity(self.seeds.len());
        let mut affinity = Vec::with_capacity(self.seeds.len());
        let mut shade = Vec::with_capacity(self.seeds.len());
        let mut moisture = Vec::with_capacity(self.seeds.len());
        let mut deep_moisture = Vec::with_capacity(self.seeds.len());
        let mut nutrients = Vec::with_capacity(self.seeds.len());
        let mut symbionts = Vec::with_capacity(self.seeds.len());
        let mut canopy = Vec::with_capacity(self.seeds.len());
        let mut litter = Vec::with_capacity(self.seeds.len());
        let mut positions = Vec::with_capacity(self.seeds.len());

        for seed in &self.seeds {
            let x = seed.x.round().clamp(0.0, (self.config.width - 1) as f32) as usize;
            let y = seed.y.round().clamp(0.0, (self.config.height - 1) as f32) as usize;
            let flat = idx2(self.config.width, x, y);
            positions.push((x, y));
            dormancy.push(seed.dormancy_s);
            age.push(seed.age_s);
            reserve.push(seed.reserve_carbon);
            affinity.push(seed.genome.symbiosis_affinity);
            shade.push(seed.genome.shade_tolerance);
            moisture.push(self.moisture[flat]);
            deep_moisture.push(self.deep_moisture[flat]);
            nutrients.push(self.shallow_nutrients[flat]);
            symbionts.push(self.symbiont_biomass[flat]);
            canopy.push(self.canopy_cover[flat]);
            litter.push(self.litter_carbon[flat]);
        }

        let stepped = step_seed_bank(
            eco_dt,
            self.daylight(),
            self.plants.len(),
            self.config.max_plants,
            &dormancy,
            &age,
            &reserve,
            &affinity,
            &shade,
            &moisture,
            &deep_moisture,
            &nutrients,
            &symbionts,
            &canopy,
            &litter,
        )?;

        let mut next_bank = Vec::new();
        let mut germinations = Vec::new();
        for (idx, mut seed) in self.seeds.drain(..).enumerate() {
            seed.age_s = stepped.age_s[idx];
            seed.dormancy_s = stepped.dormancy_s[idx];
            if stepped.germinate[idx]
                && self.plants.len() + germinations.len() < self.config.max_plants
            {
                let (x, y) = positions[idx];
                let scale = stepped.seedling_scale[idx].max(0.45);
                germinations.push((x, y, seed.genome, scale));
            } else if stepped.keep[idx] {
                next_bank.push(seed);
            }
        }
        self.seeds = next_bank;
        for (x, y, genome, scale) in germinations {
            let _ = self.add_plant(x, y, Some(genome), Some(scale));
        }
        Ok(())
    }"""

if OLD_SEEDS in fl:
    fl = fl.replace(OLD_SEEDS, NEW_SEEDS)
    print("  Replaced step_seeds_native with simplified version")
else:
    print("  WARNING: Could not find step_seeds_native to replace!")
    # Try to find it and show what's different
    if "pub(super) fn step_seeds_native" in fl:
        print("  (method exists but text doesn't match exactly)")
    else:
        print("  (method not found at all)")

# ====================================================================
# Fix 7: Remove unused constant imports (ATMOS_O2_BASELINE)
# ====================================================================
# ATMOS_O2_BASELINE is no longer used (air_o2_factor removed)
# Keep it anyway — it's harmless and might be useful later

# ====================================================================
# Verify no remaining broken references
# ====================================================================
print("\nVerification:")
problems = []
for bad in [
    "SeedCellularStateSim",
    "TerrariumSeedPose",
    "SeedTissue",
    "seed.cellular",
    "species_id",
    "next_species_id",
    "trait_distance",
    "FruitProduced",
    "SeedGerminated",
    "air_o2_factor",
    "extraction.nutrient_take * 0.067",
    "local_temp,",
]:
    if bad in fl:
        problems.append(bad)
        print(f"  WARNING: '{bad}' still present!")

if not problems:
    print("  All broken references removed!")

write(f"{SRC}/terrarium_world/flora.rs", fl)
