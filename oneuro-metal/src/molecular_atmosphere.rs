//! Native molecular-atmosphere stepping for `MolecularWorld`.
//!
//! This mirrors the Python `MolecularWorld` field update path:
//! - odorant diffusion/advection/decay with buoyancy
//! - temperature update from solar input, radiation, and conduction
//! - humidity diffusion/relaxation with water-source injection
//! - optional fruit/water lifecycle, odorant emission, and wind perturbation

use std::f32::consts::TAU;

const AMBIENT_TEMP_C: f32 = 22.0;
const SOLAR_MAX_TEMP_RATE: f32 = 0.02;
const RADIATIVE_COOLING_RATE: f32 = 0.005;
const CONDUCTION_COEFF: f32 = 0.3;
const HUMIDITY_DIFFUSION_FACTOR: f32 = 0.7;
const LAPSE_RATE_C_PER_MM: f32 = 6.5e-6;
const BUOYANCY_SCALE: f32 = 0.5;
const MW_AIR: f32 = 29.0;
const HUMIDITY_BACKGROUND: f32 = 0.4;
const HUMIDITY_DIFFUSION: f32 = 0.2;
const HUMIDITY_RELAX: f32 = 0.002;
const MAX_DIFFUSION_SUBCYCLES: usize = 30;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdorantChannelParams {
    pub diffusion_mm2_per_s: f32,
    pub decay_per_s: f32,
    pub molecular_weight: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FruitSourceState {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub ripeness: f32,
    pub sugar_content: f32,
    pub odorant_emission_rate: f32,
    pub decay_rate: f32,
    pub alive: bool,
    pub odorant_profile: Vec<(usize, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlantSourceState {
    pub x: usize,
    pub y: usize,
    pub emission_z: usize,
    pub odorant_emission_rate: f32,
    pub alive: bool,
    pub odorant_profile: Vec<(usize, f32)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSourceState {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub volume: f32,
    pub evaporation_rate: f32,
    pub alive: bool,
}

pub fn odorant_channel_params(name: &str) -> Option<OdorantChannelParams> {
    let key = name.to_ascii_lowercase();
    if key.contains("ethanol") {
        return Some(OdorantChannelParams {
            diffusion_mm2_per_s: 11.9,
            decay_per_s: 0.01,
            molecular_weight: 46.07,
        });
    }
    if key.contains("acetic") {
        return Some(OdorantChannelParams {
            diffusion_mm2_per_s: 12.4,
            decay_per_s: 0.015,
            molecular_weight: 60.05,
        });
    }
    if key.contains("ethyl_acetate") || key.contains("ethyl acetate") {
        return Some(OdorantChannelParams {
            diffusion_mm2_per_s: 8.7,
            decay_per_s: 0.008,
            molecular_weight: 88.11,
        });
    }
    if key.contains("geraniol") {
        return Some(OdorantChannelParams {
            diffusion_mm2_per_s: 5.0,
            decay_per_s: 0.02,
            molecular_weight: 154.25,
        });
    }
    if key.contains("ammonia") {
        return Some(OdorantChannelParams {
            diffusion_mm2_per_s: 22.8,
            decay_per_s: 0.005,
            molecular_weight: 17.03,
        });
    }
    if key.contains("carbon_dioxide") || key.contains("carbon dioxide") || key == "co2" {
        return Some(OdorantChannelParams {
            diffusion_mm2_per_s: 16.0,
            decay_per_s: 0.0002,
            molecular_weight: 44.01,
        });
    }
    None
}

fn validate_len(name: &str, got: usize, expected: usize) -> Result<(), String> {
    if got != expected {
        return Err(format!(
            "{name} length mismatch: expected {expected}, got {got}"
        ));
    }
    Ok(())
}

fn idx(width: usize, height: usize, x: usize, y: usize, z: usize) -> usize {
    (z * height + y) * width + x
}

fn bump_profile_channel(profile: &mut Vec<(usize, f32)>, channel_idx: usize, delta: f32) {
    for (idx, amount) in profile.iter_mut() {
        if *idx == channel_idx {
            *amount += delta;
            return;
        }
    }
    profile.push((channel_idx, delta));
}

fn next_u64(state: &mut u64) -> u64 {
    let mut x = if *state == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        *state
    };
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn next_unit_f32(state: &mut u64) -> f32 {
    let bits = (next_u64(state) >> 40) as u32;
    (bits as f32 + 1.0) / 16_777_217.0
}

fn next_normal(state: &mut u64, sigma: f32) -> f32 {
    let u1 = next_unit_f32(state).clamp(1e-7, 1.0 - 1e-7);
    let u2 = next_unit_f32(state);
    sigma * (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

fn laplacian_edge(
    field: &[f32],
    width: usize,
    height: usize,
    depth: usize,
    x: usize,
    y: usize,
    z: usize,
) -> f32 {
    let center = field[idx(width, height, x, y, z)];
    let x_l = if x > 0 { x - 1 } else { x };
    let x_r = if x + 1 < width { x + 1 } else { x };
    let y_l = if y > 0 { y - 1 } else { y };
    let y_r = if y + 1 < height { y + 1 } else { y };
    let z_l = if z > 0 { z - 1 } else { z };
    let z_r = if z + 1 < depth { z + 1 } else { z };
    field[idx(width, height, x_r, y, z)]
        + field[idx(width, height, x_l, y, z)]
        + field[idx(width, height, x, y_r, z)]
        + field[idx(width, height, x, y_l, z)]
        + field[idx(width, height, x, y, z_r)]
        + field[idx(width, height, x, y, z_l)]
        - 6.0 * center
}

pub fn step_molecular_atmosphere_fields(
    width: usize,
    height: usize,
    depth: usize,
    dt: f32,
    cell_size_mm: f32,
    is_3d: bool,
    solar_factor: f32,
    water_sources: &[(usize, usize, usize)],
    odorants: &mut [Vec<f32>],
    odorant_params: &[OdorantChannelParams],
    temperature: &mut [f32],
    humidity: &mut [f32],
    wind_x: &[f32],
    wind_y: &[f32],
    wind_z: &[f32],
) -> Result<(), String> {
    let depth = depth.max(1);
    let total = width * height * depth;
    validate_len("temperature", temperature.len(), total)?;
    validate_len("humidity", humidity.len(), total)?;
    validate_len("wind_x", wind_x.len(), total)?;
    validate_len("wind_y", wind_y.len(), total)?;
    validate_len("wind_z", wind_z.len(), total)?;
    validate_len("odorant_params", odorant_params.len(), odorants.len())?;
    for (channel_idx, channel) in odorants.iter().enumerate() {
        validate_len(&format!("odorant[{channel_idx}]"), channel.len(), total)?;
    }

    let dx = cell_size_mm.max(1e-6);
    let dx2 = dx * dx;
    let cfl_factor = if is_3d && depth > 1 { 1.0 / 6.0 } else { 0.25 };

    for (channel, params) in odorants.iter_mut().zip(odorant_params.iter().copied()) {
        let diffusion = params.diffusion_mm2_per_s.max(0.0);
        let decay = params.decay_per_s.max(0.0);
        let v_field_max = wind_x
            .iter()
            .zip(wind_y.iter())
            .zip(wind_z.iter())
            .fold(0.0f32, |acc, ((vx, vy), vz)| {
                acc.max(vx.abs()).max(vy.abs()).max(vz.abs())
            });
        let v_buoy = if is_3d && depth > 1 {
            (BUOYANCY_SCALE * (MW_AIR - params.molecular_weight) / MW_AIR).abs()
        } else {
            0.0
        };
        let dt_cfl_diff = cfl_factor * dx2 / diffusion.max(1e-12);
        let dt_cfl_adv = dx / v_field_max.max(v_buoy).max(1e-12);
        let dt_sub_guess = dt.min(dt_cfl_diff).min(dt_cfl_adv).max(1e-6);
        let n_sub = usize::max(
            1,
            usize::min(MAX_DIFFUSION_SUBCYCLES, (dt / dt_sub_guess).ceil() as usize),
        );
        let dt_sub = dt / n_sub as f32;
        let buoyancy_vz = if is_3d && depth > 1 {
            BUOYANCY_SCALE * (MW_AIR - params.molecular_weight) / MW_AIR
        } else {
            0.0
        };
        let mut next = vec![0.0f32; total];

        for _ in 0..n_sub {
            for z in 0..depth {
                for y in 0..height {
                    for x in 0..width {
                        let i = idx(width, height, x, y, z);
                        let center = channel[i];
                        let x_l = if x > 0 { x - 1 } else { x };
                        let x_r = if x + 1 < width { x + 1 } else { x };
                        let y_l = if y > 0 { y - 1 } else { y };
                        let y_r = if y + 1 < height { y + 1 } else { y };
                        let z_l = if z > 0 { z - 1 } else { z };
                        let z_r = if z + 1 < depth { z + 1 } else { z };

                        let left = channel[idx(width, height, x_l, y, z)];
                        let right = channel[idx(width, height, x_r, y, z)];
                        let down = channel[idx(width, height, x, y_l, z)];
                        let up = channel[idx(width, height, x, y_r, z)];
                        let back = channel[idx(width, height, x, y, z_l)];
                        let front = channel[idx(width, height, x, y, z_r)];

                        let lap = (right + left + up + down + front + back - 6.0 * center) / dx2;
                        let grad_x = if wind_x[i] >= 0.0 {
                            (center - left) / dx
                        } else {
                            (right - center) / dx
                        };
                        let grad_y = if wind_y[i] >= 0.0 {
                            (center - down) / dx
                        } else {
                            (up - center) / dx
                        };
                        let mut advection = wind_x[i] * grad_x + wind_y[i] * grad_y;

                        if depth > 1 {
                            let eff_vz = wind_z[i] + buoyancy_vz;
                            let grad_z = if eff_vz >= 0.0 {
                                (center - back) / dx
                            } else {
                                (front - center) / dx
                            };
                            advection += eff_vz * grad_z;
                        }

                        let humidity_scale = 1.0 - humidity[i] * (1.0 - HUMIDITY_DIFFUSION_FACTOR);
                        let delta = (diffusion * humidity_scale * lap - advection - decay * center)
                            * dt_sub;
                        next[i] = (center + delta).max(0.0);
                    }
                }
            }
            channel.copy_from_slice(&next);
        }
    }

    let solar = SOLAR_MAX_TEMP_RATE * solar_factor.max(0.0);
    let mut next_temp = vec![0.0f32; total];
    for z in 0..depth {
        let ambient = if is_3d && depth > 1 {
            AMBIENT_TEMP_C - LAPSE_RATE_C_PER_MM * z as f32 * dx
        } else {
            AMBIENT_TEMP_C
        };
        for y in 0..height {
            for x in 0..width {
                let i = idx(width, height, x, y, z);
                let lap = laplacian_edge(temperature, width, height, depth, x, y, z) / dx2;
                let solar_term = if z == 0 { solar } else { 0.0 };
                let radiation = RADIATIVE_COOLING_RATE * (temperature[i] - ambient);
                next_temp[i] =
                    temperature[i] + (solar_term - radiation + CONDUCTION_COEFF * lap) * dt;
            }
        }
    }
    temperature.copy_from_slice(&next_temp);

    for &(x, y, z) in water_sources {
        if x < width && y < height && z < depth {
            let i = idx(width, height, x, y, z);
            humidity[i] = (humidity[i] + 0.01 * dt).min(1.0);
        }
    }

    let mut next_humidity = vec![0.0f32; total];
    for z in 0..depth {
        let background = if is_3d && depth > 1 {
            HUMIDITY_BACKGROUND * (-(z as f32 * dx) / 50.0).exp()
        } else {
            HUMIDITY_BACKGROUND
        };
        for y in 0..height {
            for x in 0..width {
                let i = idx(width, height, x, y, z);
                let lap = laplacian_edge(humidity, width, height, depth, x, y, z) / dx2;
                let next = humidity[i]
                    + (HUMIDITY_DIFFUSION * lap - HUMIDITY_RELAX * (humidity[i] - background)) * dt;
                next_humidity[i] = next.clamp(0.0, 1.0);
            }
        }
    }
    humidity.copy_from_slice(&next_humidity);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn step_molecular_world_fields(
    width: usize,
    height: usize,
    depth: usize,
    dt: f32,
    cell_size_mm: f32,
    is_3d: bool,
    solar_factor: f32,
    ammonia_channel: Option<usize>,
    fruits: &mut [FruitSourceState],
    plants: &[PlantSourceState],
    waters: &mut [WaterSourceState],
    odorants: &mut [Vec<f32>],
    odorant_params: &[OdorantChannelParams],
    temperature: &mut [f32],
    humidity: &mut [f32],
    wind_x: &mut [f32],
    wind_y: &mut [f32],
    wind_z: &mut [f32],
    rng_state: &mut u64,
) -> Result<(), String> {
    let depth = depth.max(1);
    let total = width * height * depth;
    validate_len("wind_x", wind_x.len(), total)?;
    validate_len("wind_y", wind_y.len(), total)?;
    validate_len("wind_z", wind_z.len(), total)?;

    for fruit in fruits.iter_mut() {
        if !fruit.alive {
            continue;
        }
        fruit.ripeness = (fruit.ripeness + fruit.decay_rate * dt).min(1.0);
        if fruit.ripeness >= 1.0 {
            if let Some(channel_idx) = ammonia_channel {
                bump_profile_channel(&mut fruit.odorant_profile, channel_idx, 0.001 * dt);
            }
            fruit.sugar_content = (fruit.sugar_content - 0.0005 * dt).max(0.0);
            if fruit.sugar_content <= 0.0 {
                fruit.alive = false;
            }
        }
    }

    for water in waters.iter_mut() {
        if !water.alive {
            continue;
        }
        water.volume -= water.evaporation_rate * dt;
        if water.volume <= 0.0 {
            water.volume = 0.0;
            water.alive = false;
        }
    }

    for fruit in fruits.iter() {
        if !fruit.alive {
            continue;
        }
        let rate = fruit.odorant_emission_rate * fruit.ripeness * dt;
        let z = fruit.z.min(depth - 1);
        let x = fruit.x.min(width - 1);
        let y = fruit.y.min(height - 1);
        let i = idx(width, height, x, y, z);
        for &(channel_idx, fraction) in fruit.odorant_profile.iter() {
            if let Some(channel) = odorants.get_mut(channel_idx) {
                channel[i] += rate * fraction;
            }
        }
    }

    for plant in plants.iter() {
        if !plant.alive {
            continue;
        }
        let rate = plant.odorant_emission_rate * dt;
        let z = plant.emission_z.min(depth - 1);
        let x = plant.x.min(width - 1);
        let y = plant.y.min(height - 1);
        let i = idx(width, height, x, y, z);
        for &(channel_idx, fraction) in plant.odorant_profile.iter() {
            if let Some(channel) = odorants.get_mut(channel_idx) {
                channel[i] += rate * fraction;
            }
        }
    }

    let active_water_sources: Vec<(usize, usize, usize)> = waters
        .iter()
        .filter(|water| water.alive)
        .map(|water| (water.x, water.y, water.z.min(depth - 1)))
        .collect();

    step_molecular_atmosphere_fields(
        width,
        height,
        depth,
        dt,
        cell_size_mm,
        is_3d,
        solar_factor,
        &active_water_sources,
        odorants,
        odorant_params,
        temperature,
        humidity,
        wind_x,
        wind_y,
        wind_z,
    )?;

    const BASE_VX: f32 = 0.5;
    const BASE_VY: f32 = 0.0;
    const BASE_VZ: f32 = 0.0;
    const REVERT: f32 = 0.001;
    for i in 0..total {
        wind_x[i] += next_normal(rng_state, 0.002) - REVERT * (wind_x[i] - BASE_VX) * dt;
        wind_y[i] += next_normal(rng_state, 0.002) - REVERT * (wind_y[i] - BASE_VY) * dt;
        if is_3d && depth > 1 {
            wind_z[i] += next_normal(rng_state, 0.001) - REVERT * (wind_z[i] - BASE_VZ) * dt;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        odorant_channel_params, step_molecular_atmosphere_fields, step_molecular_world_fields,
        FruitSourceState, OdorantChannelParams, PlantSourceState, WaterSourceState,
    };

    #[test]
    fn odorant_transport_stays_bounded() {
        let width = 8;
        let height = 6;
        let depth = 1;
        let total = width * height * depth;
        let mut odorants = vec![vec![0.0f32; total]];
        odorants[0][3 * width + 4] = 1.0;
        let params = vec![odorant_channel_params("ethanol").unwrap()];
        let mut temperature = vec![22.0f32; total];
        let mut humidity = vec![0.4f32; total];
        let wind_x = vec![0.0f32; total];
        let wind_y = vec![0.0f32; total];
        let wind_z = vec![0.0f32; total];
        step_molecular_atmosphere_fields(
            width,
            height,
            depth,
            0.1,
            1.0,
            false,
            0.8,
            &[],
            &mut odorants,
            &params,
            &mut temperature,
            &mut humidity,
            &wind_x,
            &wind_y,
            &wind_z,
        )
        .unwrap();
        assert!(odorants[0].iter().all(|v| *v >= 0.0));
        assert!(temperature.iter().all(|v| v.is_finite()));
        assert!(humidity.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(odorants[0][3 * width + 4] < 1.0);
    }

    #[test]
    fn water_sources_raise_humidity() {
        let width = 6;
        let height = 6;
        let depth = 1;
        let total = width * height * depth;
        let mut odorants = vec![vec![0.0f32; total]];
        let params = vec![odorant_channel_params("ammonia").unwrap()];
        let mut temperature = vec![22.0f32; total];
        let mut humidity = vec![0.4f32; total];
        let wind_x = vec![0.0f32; total];
        let wind_y = vec![0.0f32; total];
        let wind_z = vec![0.0f32; total];
        step_molecular_atmosphere_fields(
            width,
            height,
            depth,
            0.1,
            1.0,
            false,
            0.0,
            &[(2, 2, 0)],
            &mut odorants,
            &params,
            &mut temperature,
            &mut humidity,
            &wind_x,
            &wind_y,
            &wind_z,
        )
        .unwrap();
        assert!(humidity[2 * width + 2] > 0.4);
    }

    #[test]
    fn world_step_updates_sources_and_wind() {
        let width = 8;
        let height = 6;
        let depth = 1;
        let total = width * height * depth;
        let mut odorants = vec![vec![0.0f32; total]];
        let params: Vec<OdorantChannelParams> = vec![odorant_channel_params("ethanol").unwrap()];
        let mut temperature = vec![22.0f32; total];
        let mut humidity = vec![0.4f32; total];
        let mut wind_x = vec![0.5f32; total];
        let mut wind_y = vec![0.0f32; total];
        let mut wind_z = vec![0.0f32; total];
        let mut rng_state = 12345u64;
        let mut fruits = vec![FruitSourceState {
            x: 4,
            y: 3,
            z: 0,
            ripeness: 0.9,
            sugar_content: 1.0,
            odorant_emission_rate: 0.05,
            decay_rate: 0.001,
            alive: true,
            odorant_profile: vec![(0, 1.0)],
        }];
        let plants = vec![PlantSourceState {
            x: 2,
            y: 2,
            emission_z: 0,
            odorant_emission_rate: 0.03,
            alive: true,
            odorant_profile: vec![(0, 1.0)],
        }];
        let mut waters = vec![WaterSourceState {
            x: 1,
            y: 1,
            z: 0,
            volume: 1.0,
            evaporation_rate: 0.01,
            alive: true,
        }];
        step_molecular_world_fields(
            width,
            height,
            depth,
            0.1,
            1.0,
            false,
            0.8,
            None,
            &mut fruits,
            &plants,
            &mut waters,
            &mut odorants,
            &params,
            &mut temperature,
            &mut humidity,
            &mut wind_x,
            &mut wind_y,
            &mut wind_z,
            &mut rng_state,
        )
        .unwrap();
        assert!(odorants[0].iter().sum::<f32>() > 0.0);
        assert!(waters[0].volume < 1.0);
        assert!(wind_x.iter().any(|v| (*v - 0.5).abs() > 1e-6));
    }
}
