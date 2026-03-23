//! Native helpers for ecological bookkeeping events.
//!
//! These functions keep deterministic food-patch decay and seed-bank
//! progression out of Python loops while leaving object creation and
//! higher-level world wiring in Python.

use crate::constants::clamp;

const FOOD_PATCH_MICROBIAL_DECAY_MIN: f32 = 1.0e-6;
const SEED_MAX_AGE_S: f32 = 300_000.0;
const SEED_MIN_SURFACE_MOISTURE: f32 = 0.08;
const SEED_MIN_DEEP_MOISTURE: f32 = 0.04;
const SEED_MIN_RESERVE_CARBON: f32 = 0.04;

#[derive(Debug, Clone)]
pub struct FoodPatchStepResult {
    pub remaining: Vec<f32>,
    pub sugar_content: Vec<f32>,
    pub fruit_alive: Vec<bool>,
    pub deposited_all: Vec<bool>,
    pub decay_detritus: Vec<f32>,
    pub lost_detritus: Vec<f32>,
    pub final_detritus: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SeedBankStepResult {
    pub dormancy_s: Vec<f32>,
    pub age_s: Vec<f32>,
    pub keep: Vec<bool>,
    pub germinate: Vec<bool>,
    pub seedling_scale: Vec<f32>,
}

fn ensure_len(name: &str, got: usize, expected: usize) -> Result<(), String> {
    if got != expected {
        return Err(format!(
            "{} length mismatch: expected {}, got {}",
            name, expected, got
        ));
    }
    Ok(())
}

pub fn step_food_patches(
    dt: f32,
    patch_remaining: &[f32],
    previous_remaining: &[f32],
    deposited_all: &[bool],
    has_fruit: &[bool],
    fruit_ripeness: &[f32],
    fruit_sugar_content: &[f32],
    microbial_biomass: &[f32],
) -> Result<FoodPatchStepResult, String> {
    let n = patch_remaining.len();
    for (name, len) in [
        ("previous_remaining", previous_remaining.len()),
        ("deposited_all", deposited_all.len()),
        ("has_fruit", has_fruit.len()),
        ("fruit_ripeness", fruit_ripeness.len()),
        ("fruit_sugar_content", fruit_sugar_content.len()),
        ("microbial_biomass", microbial_biomass.len()),
    ] {
        ensure_len(name, len, n)?;
    }

    let mut remaining = Vec::with_capacity(n);
    let mut sugar_content = Vec::with_capacity(n);
    let mut fruit_alive = Vec::with_capacity(n);
    let mut deposited_next = Vec::with_capacity(n);
    let mut decay_detritus = Vec::with_capacity(n);
    let mut lost_detritus = Vec::with_capacity(n);
    let mut final_detritus = Vec::with_capacity(n);

    for i in 0..n {
        let mut current = patch_remaining[i].max(0.0);
        let previous = previous_remaining[i].max(0.0);
        let mut deposited = deposited_all[i];
        let mut sugar = fruit_sugar_content[i].max(0.0);
        let mut alive = false;
        let mut decay_loss = 0.0f32;

        if has_fruit[i] {
            sugar = sugar.min(current);
            let microbial_decay_driver = microbial_biomass[i].max(0.0);
            if current > 0.0 && microbial_decay_driver > FOOD_PATCH_MICROBIAL_DECAY_MIN {
                let decay_driver =
                    (fruit_ripeness[i] - 0.82).max(0.0) + (0.02 - current).max(0.0) * 0.2;
                decay_loss = current
                    .min(decay_driver * (0.00008 * dt) * (1.0 + microbial_decay_driver * 8.0));
                if decay_loss > 0.0 {
                    current = (current - decay_loss).max(0.0);
                    sugar = sugar.min(current);
                }
            }
            alive = current > 0.01;
        }

        let lost = (previous - current).max(0.0);
        let lost_return = if lost > 0.0 { lost * 0.65 } else { 0.0 };
        let final_return = if current <= 0.01 && !deposited {
            deposited = true;
            0.03
        } else {
            0.0
        };

        remaining.push(current);
        sugar_content.push(sugar);
        fruit_alive.push(alive);
        deposited_next.push(deposited);
        decay_detritus.push(decay_loss);
        lost_detritus.push(lost_return);
        final_detritus.push(final_return);
    }

    Ok(FoodPatchStepResult {
        remaining,
        sugar_content,
        fruit_alive,
        deposited_all: deposited_next,
        decay_detritus,
        lost_detritus,
        final_detritus,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn step_seed_bank(
    dt: f32,
    light: f32,
    current_plant_count: usize,
    max_plants: usize,
    dormancy_s: &[f32],
    age_s: &[f32],
    reserve_carbon: &[f32],
    symbiosis_affinity: &[f32],
    shade_tolerance: &[f32],
    moisture: &[f32],
    deep_moisture: &[f32],
    nutrients: &[f32],
    symbionts: &[f32],
    canopy: &[f32],
    litter: &[f32],
) -> Result<SeedBankStepResult, String> {
    let n = dormancy_s.len();
    for (name, len) in [
        ("age_s", age_s.len()),
        ("reserve_carbon", reserve_carbon.len()),
        ("symbiosis_affinity", symbiosis_affinity.len()),
        ("shade_tolerance", shade_tolerance.len()),
        ("moisture", moisture.len()),
        ("deep_moisture", deep_moisture.len()),
        ("nutrients", nutrients.len()),
        ("symbionts", symbionts.len()),
        ("canopy", canopy.len()),
        ("litter", litter.len()),
    ] {
        ensure_len(name, len, n)?;
    }

    let mut dormancy_out = Vec::with_capacity(n);
    let mut age_out = Vec::with_capacity(n);
    let mut keep = Vec::with_capacity(n);
    let mut germinate = Vec::with_capacity(n);
    let mut seedling_scale = Vec::with_capacity(n);
    let mut plant_count = current_plant_count;

    for i in 0..n {
        let age = age_s[i] + dt;
        let dormancy = dormancy_s[i] - dt * (0.55 + moisture[i] * 0.9 + light * 0.15);
        let mut keep_seed = false;
        let mut germinate_seed = false;
        let mut scale = 0.0f32;
        let moisture_ready = moisture[i] >= SEED_MIN_SURFACE_MOISTURE;
        let deep_moisture_ready = deep_moisture[i] >= SEED_MIN_DEEP_MOISTURE;
        let reserve_ready = reserve_carbon[i] >= SEED_MIN_RESERVE_CARBON;

        if age >= SEED_MAX_AGE_S {
            keep_seed = false;
        } else if dormancy > 0.0 {
            keep_seed = true;
        } else {
            let germination_score = moisture[i] * 1.2
                + deep_moisture[i] * 0.5
                + nutrients[i] * 4.0
                + symbionts[i] * 6.0 * symbiosis_affinity[i]
                + litter[i] * 1.6
                + reserve_carbon[i] * 2.0
                - canopy[i] * (1.0 - shade_tolerance[i]).max(0.18);

            if moisture_ready
                && deep_moisture_ready
                && reserve_ready
                && germination_score > 0.72
                && plant_count < max_plants
            {
                germinate_seed = true;
                plant_count += 1;
                scale = clamp(
                    0.30 + reserve_carbon[i].max(0.0).powf(0.75) * 2.2,
                    0.45,
                    1.10,
                );
            } else if dormancy > -28000.0 {
                keep_seed = true;
            }
        }

        dormancy_out.push(dormancy);
        age_out.push(age);
        keep.push(keep_seed);
        germinate.push(germinate_seed);
        seedling_scale.push(scale);
    }

    Ok(SeedBankStepResult {
        dormancy_s: dormancy_out,
        age_s: age_out,
        keep,
        germinate,
        seedling_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::{step_food_patches, step_seed_bank};

    #[test]
    fn food_patch_decay_stays_bounded() {
        let result = step_food_patches(
            45.0,
            &[0.8, 0.01],
            &[0.85, 0.04],
            &[false, false],
            &[true, true],
            &[0.9, 0.95],
            &[0.7, 0.03],
            &[0.02, 0.05],
        )
        .unwrap();
        assert!(result.remaining.iter().all(|v| *v >= 0.0));
        assert_eq!(result.remaining.len(), 2);
    }

    #[test]
    fn seed_bank_progression_respects_capacity() {
        let result = step_seed_bank(
            50.0,
            0.9,
            27,
            28,
            &[0.0, 0.0],
            &[100.0, 100.0],
            &[0.15, 0.15],
            &[1.0, 1.0],
            &[0.8, 0.8],
            &[0.5, 0.5],
            &[0.4, 0.4],
            &[0.1, 0.1],
            &[0.05, 0.05],
            &[0.0, 0.0],
            &[0.02, 0.02],
        )
        .unwrap();
        assert_eq!(result.germinate.iter().filter(|v| **v).count(), 1);
    }
}
