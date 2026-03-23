use std::f32::consts::FRAC_PI_2;

use super::{
    CelegansOrganism, CelegansSensoryInputs, CELEGANS_BODY_DIAMETER_UM, CELEGANS_BODY_LENGTH_UM,
    CELEGANS_PREFERRED_TEMP_C,
};

const SENSOR_FORWARD_OFFSET_UM: f32 = CELEGANS_BODY_LENGTH_UM * 0.08;
const SENSOR_LATERAL_OFFSET_UM: f32 = CELEGANS_BODY_DIAMETER_UM * 0.5;

#[derive(Clone, Debug)]
pub struct CelegansAssayTracePoint {
    pub episode: u32,
    pub step: u32,
    pub x_um: f32,
    pub y_um: f32,
    pub angle_rad: f32,
    pub speed_um_s: f32,
    pub command_bias: f32,
    pub steering_bias: f32,
    pub attractant_left: f32,
    pub attractant_right: f32,
    pub temperature_left_c: f32,
    pub temperature_right_c: f32,
    pub immersion: f32,
    pub food_density: f32,
    pub energy_reserve: f32,
    pub gut_content: f32,
    pub pharyngeal_pumping_hz: f32,
}

#[derive(Clone, Debug)]
pub struct CelegansAssayResult {
    pub name: String,
    pub passed: bool,
    pub metric_name: String,
    pub metric_value: f64,
    pub threshold: f64,
    pub details: String,
    pub traces: Vec<CelegansAssayTracePoint>,
}

impl std::fmt::Display for CelegansAssayResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passed { "PASS" } else { "FAIL" };
        write!(
            f,
            "[{}] {} | {} = {:.4} (threshold: {:.4}) | {}",
            status, self.name, self.metric_name, self.metric_value, self.threshold, self.details
        )
    }
}

#[derive(Clone, Debug)]
pub struct CelegansChemotaxisAssayConfig {
    pub episodes: u32,
    pub steps_per_episode: u32,
    pub dt_ms: f32,
    pub sample_every: u32,
    pub source_x_um: f32,
    pub source_y_um: f32,
    pub source_sigma_um: f32,
    pub start_x_um: f32,
    pub start_lateral_offset_um: f32,
    pub posterior_touch_pulse_steps: u32,
    pub posterior_touch_strength: f32,
}

impl Default for CelegansChemotaxisAssayConfig {
    fn default() -> Self {
        Self {
            episodes: 4,
            steps_per_episode: 4000,
            dt_ms: 1.0,
            sample_every: 20,
            source_x_um: 180.0,
            source_y_um: 0.0,
            source_sigma_um: 140.0,
            start_x_um: -60.0,
            start_lateral_offset_um: 120.0,
            posterior_touch_pulse_steps: 160,
            posterior_touch_strength: 0.6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CelegansThermotaxisAssayConfig {
    pub episodes: u32,
    pub steps_per_episode: u32,
    pub dt_ms: f32,
    pub sample_every: u32,
    pub min_x_um: f32,
    pub max_x_um: f32,
    pub min_temp_c: f32,
    pub max_temp_c: f32,
    pub hot_start_x_um: f32,
    pub cold_start_x_um: f32,
    pub start_lateral_offset_um: f32,
    pub posterior_touch_pulse_steps: u32,
    pub posterior_touch_strength: f32,
    pub baseline_food_density: f32,
}

impl Default for CelegansThermotaxisAssayConfig {
    fn default() -> Self {
        Self {
            episodes: 4,
            steps_per_episode: 4000,
            dt_ms: 1.0,
            sample_every: 20,
            min_x_um: 0.0,
            max_x_um: 1000.0,
            min_temp_c: 14.0,
            max_temp_c: 26.0,
            hot_start_x_um: 920.0,
            cold_start_x_um: 80.0,
            start_lateral_offset_um: 80.0,
            posterior_touch_pulse_steps: 220,
            posterior_touch_strength: 0.7,
            baseline_food_density: 0.35,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CelegansCrawlSwimAssayConfig {
    pub steps_per_mode: u32,
    pub dt_ms: f32,
    pub sample_every: u32,
    pub drive_strength: f32,
    pub posterior_touch_pulse_steps: u32,
    pub posterior_touch_strength: f32,
    pub food_density: f32,
}

impl Default for CelegansCrawlSwimAssayConfig {
    fn default() -> Self {
        Self {
            steps_per_mode: 500,
            dt_ms: 1.0,
            sample_every: 20,
            drive_strength: 0.4,
            posterior_touch_pulse_steps: 80,
            posterior_touch_strength: 0.45,
            food_density: 0.35,
        }
    }
}

pub fn run_all_assays() -> Vec<CelegansAssayResult> {
    vec![
        run_chemotaxis_assay(&CelegansChemotaxisAssayConfig::default()),
        run_thermotaxis_assay(&CelegansThermotaxisAssayConfig::default()),
        run_crawl_swim_assay(&CelegansCrawlSwimAssayConfig::default()),
    ]
}

pub fn run_chemotaxis_assay(config: &CelegansChemotaxisAssayConfig) -> CelegansAssayResult {
    let mut traces = Vec::new();
    let mut attractant_gain_sum = 0.0;
    let mut x_progress_sum = 0.0;
    let mut alignment_sum = 0.0;
    let mut command_bias_sum = 0.0;

    for episode in 0..config.episodes.max(1) {
        let mut worm = CelegansOrganism::new();
        worm.x_um = config.start_x_um;
        worm.y_um = if episode % 2 == 0 {
            config.start_lateral_offset_um
        } else {
            -config.start_lateral_offset_um
        };
        worm.angle_rad = 0.0;

        let start_attractant = gaussian_field(
            worm.x_um,
            worm.y_um,
            config.source_x_um,
            config.source_y_um,
            config.source_sigma_um * 1.3,
        );
        let start_x = worm.x_um;
        let mut sample_alignment = 0.0f32;
        let mut sample_command_bias = 0.0f32;
        let mut sample_count = 0.0f32;

        for step in 0..config.steps_per_episode {
            let ((lx, ly), (rx, ry)) = head_sensor_positions(&worm);
            let attractant_left = gaussian_field(
                lx,
                ly,
                config.source_x_um,
                config.source_y_um,
                config.source_sigma_um,
            );
            let attractant_right = gaussian_field(
                rx,
                ry,
                config.source_x_um,
                config.source_y_um,
                config.source_sigma_um,
            );
            let food_density = gaussian_field(
                worm.x_um,
                worm.y_um,
                config.source_x_um,
                config.source_y_um,
                config.source_sigma_um * 1.3,
            )
            .max(0.05);
            let inputs = CelegansSensoryInputs {
                attractant_left,
                attractant_right,
                posterior_touch: pulse_touch(
                    step,
                    config.posterior_touch_pulse_steps,
                    config.posterior_touch_strength,
                ),
                food_density,
                ..Default::default()
            };
            worm.step_with_inputs(config.dt_ms, &inputs);

            let to_source_x = config.source_x_um - worm.x_um;
            let to_source_y = config.source_y_um - worm.y_um;
            let to_source_norm = (to_source_x * to_source_x + to_source_y * to_source_y)
                .sqrt()
                .max(1.0);
            let heading_alignment = (worm.angle_rad.cos() * (to_source_x / to_source_norm)
                + worm.angle_rad.sin() * (to_source_y / to_source_norm))
                .clamp(-1.0, 1.0);
            sample_alignment += heading_alignment;
            sample_command_bias += worm.command_bias();
            sample_count += 1.0;

            if step % config.sample_every.max(1) == 0 {
                traces.push(CelegansAssayTracePoint {
                    episode,
                    step,
                    x_um: worm.x_um,
                    y_um: worm.y_um,
                    angle_rad: worm.angle_rad,
                    speed_um_s: worm.speed_um_s,
                    command_bias: worm.command_bias(),
                    steering_bias: worm.head_steering_bias(),
                    attractant_left,
                    attractant_right,
                    temperature_left_c: CELEGANS_PREFERRED_TEMP_C,
                    temperature_right_c: CELEGANS_PREFERRED_TEMP_C,
                    immersion: 0.0,
                    food_density,
                    energy_reserve: worm.energy_reserve(),
                    gut_content: worm.gut_content(),
                    pharyngeal_pumping_hz: worm.pharyngeal_pumping_hz(),
                });
            }
        }

        let end_attractant = gaussian_field(
            worm.x_um,
            worm.y_um,
            config.source_x_um,
            config.source_y_um,
            config.source_sigma_um * 1.3,
        );
        attractant_gain_sum += end_attractant - start_attractant;
        x_progress_sum += (worm.x_um - start_x) / (config.source_x_um - start_x).abs().max(1.0);
        alignment_sum += sample_alignment / sample_count.max(1.0);
        command_bias_sum += sample_command_bias / sample_count.max(1.0);
    }

    let episodes = config.episodes.max(1) as f32;
    let mean_attractant_gain = attractant_gain_sum / episodes;
    let mean_x_progress = x_progress_sum / episodes;
    let mean_alignment = alignment_sum / episodes;
    let mean_command_bias = command_bias_sum / episodes;
    let threshold = 0.008;

    CelegansAssayResult {
        name: "chemotaxis".to_string(),
        passed: mean_attractant_gain > threshold && mean_alignment > 0.0,
        metric_name: "mean_local_attractant_gain".to_string(),
        metric_value: mean_attractant_gain as f64,
        threshold: threshold as f64,
        details: format!(
            "attractant gain {:.4}, normalized x progress {:.3}, heading alignment {:.4}, command bias {:.4}",
            mean_attractant_gain,
            mean_x_progress,
            mean_alignment,
            mean_command_bias
        ),
        traces,
    }
}

pub fn run_thermotaxis_assay(config: &CelegansThermotaxisAssayConfig) -> CelegansAssayResult {
    let mut traces = Vec::new();
    let mut error_reduction_sum = 0.0;
    let mut alignment_sum = 0.0;
    let mut command_bias_sum = 0.0;

    for episode in 0..config.episodes.max(1) {
        let mut worm = CelegansOrganism::new();
        let starts_hot_side = episode % 2 == 0;
        worm.x_um = if starts_hot_side {
            config.hot_start_x_um
        } else {
            config.cold_start_x_um
        };
        worm.y_um = if starts_hot_side {
            config.start_lateral_offset_um
        } else {
            -config.start_lateral_offset_um
        };
        worm.angle_rad = if starts_hot_side {
            FRAC_PI_2
        } else {
            -FRAC_PI_2
        };

        let initial_temp = temperature_at(
            worm.x_um,
            config.min_x_um,
            config.max_x_um,
            config.min_temp_c,
            config.max_temp_c,
        );
        let initial_error = (initial_temp - CELEGANS_PREFERRED_TEMP_C).abs();
        let mut sample_alignment = 0.0f32;
        let mut sample_command_bias = 0.0f32;
        let mut sample_count = 0.0f32;

        for step in 0..config.steps_per_episode {
            let ((lx, _ly), (rx, _ry)) = head_sensor_positions(&worm);
            let temp_left = temperature_at(
                lx,
                config.min_x_um,
                config.max_x_um,
                config.min_temp_c,
                config.max_temp_c,
            );
            let temp_right = temperature_at(
                rx,
                config.min_x_um,
                config.max_x_um,
                config.min_temp_c,
                config.max_temp_c,
            );
            let inputs = CelegansSensoryInputs {
                temperature_left_c: temp_left,
                temperature_right_c: temp_right,
                posterior_touch: pulse_touch(
                    step,
                    config.posterior_touch_pulse_steps,
                    config.posterior_touch_strength,
                ),
                food_density: config.baseline_food_density,
                ..Default::default()
            };
            worm.step_with_inputs(config.dt_ms, &inputs);

            let local_temp = temperature_at(
                worm.x_um,
                config.min_x_um,
                config.max_x_um,
                config.min_temp_c,
                config.max_temp_c,
            );
            let desired_heading_x = if local_temp > CELEGANS_PREFERRED_TEMP_C {
                -1.0
            } else {
                1.0
            };
            sample_alignment += (worm.angle_rad.cos() * desired_heading_x).clamp(-1.0, 1.0);
            sample_command_bias += worm.command_bias();
            sample_count += 1.0;

            if step % config.sample_every.max(1) == 0 {
                traces.push(CelegansAssayTracePoint {
                    episode,
                    step,
                    x_um: worm.x_um,
                    y_um: worm.y_um,
                    angle_rad: worm.angle_rad,
                    speed_um_s: worm.speed_um_s,
                    command_bias: worm.command_bias(),
                    steering_bias: worm.head_steering_bias(),
                    attractant_left: 0.0,
                    attractant_right: 0.0,
                    temperature_left_c: temp_left,
                    temperature_right_c: temp_right,
                    immersion: 0.0,
                    food_density: config.baseline_food_density,
                    energy_reserve: worm.energy_reserve(),
                    gut_content: worm.gut_content(),
                    pharyngeal_pumping_hz: worm.pharyngeal_pumping_hz(),
                });
            }
        }

        let final_temp = temperature_at(
            worm.x_um,
            config.min_x_um,
            config.max_x_um,
            config.min_temp_c,
            config.max_temp_c,
        );
        let final_error = (final_temp - CELEGANS_PREFERRED_TEMP_C).abs();
        error_reduction_sum += (initial_error - final_error) / initial_error.max(0.1);
        alignment_sum += sample_alignment / sample_count.max(1.0);
        command_bias_sum += sample_command_bias / sample_count.max(1.0);
    }

    let episodes = config.episodes.max(1) as f32;
    let mean_error_reduction = error_reduction_sum / episodes;
    let mean_alignment = alignment_sum / episodes;
    let mean_command_bias = command_bias_sum / episodes;
    let threshold = 0.02;

    CelegansAssayResult {
        name: "thermotaxis".to_string(),
        passed: mean_error_reduction > threshold && mean_alignment > 0.0,
        metric_name: "mean_temperature_error_reduction".to_string(),
        metric_value: mean_error_reduction as f64,
        threshold: threshold as f64,
        details: format!(
            "temperature error reduction {:.1}%, heading alignment {:.4}, command bias {:.4}",
            mean_error_reduction * 100.0,
            mean_alignment,
            mean_command_bias
        ),
        traces,
    }
}

pub fn run_crawl_swim_assay(config: &CelegansCrawlSwimAssayConfig) -> CelegansAssayResult {
    let (crawl_speed, crawl_command_bias, crawl_traces) = simulate_mode(config, 0.0, 0);
    let (swim_speed, swim_command_bias, swim_traces) = simulate_mode(config, 1.0, 1);
    let speed_ratio = swim_speed / crawl_speed.max(1.0);
    let separation = (speed_ratio - 1.0).abs();

    let mut traces = crawl_traces;
    traces.extend(swim_traces);

    CelegansAssayResult {
        name: "crawl_vs_swim".to_string(),
        passed: separation > 0.1 && crawl_speed > 20.0 && swim_speed > 20.0,
        metric_name: "mode_speed_ratio_separation".to_string(),
        metric_value: separation as f64,
        threshold: 0.1,
        details: format!(
            "crawl {:.2} um/s, swim {:.2} um/s, swim/crawl {:.3}, command biases {:.4}/{:.4}",
            crawl_speed, swim_speed, speed_ratio, crawl_command_bias, swim_command_bias
        ),
        traces,
    }
}

fn simulate_mode(
    config: &CelegansCrawlSwimAssayConfig,
    immersion: f32,
    episode: u32,
) -> (f32, f32, Vec<CelegansAssayTracePoint>) {
    let mut worm = CelegansOrganism::new();
    let mut speed_sum = 0.0f32;
    let mut command_bias_sum = 0.0f32;
    let mut sample_count = 0.0f32;
    let mut traces = Vec::new();

    for step in 0..config.steps_per_mode {
        let inputs = CelegansSensoryInputs {
            attractant_left: config.drive_strength,
            attractant_right: config.drive_strength,
            immersion,
            posterior_touch: pulse_touch(
                step,
                config.posterior_touch_pulse_steps,
                config.posterior_touch_strength,
            ),
            food_density: config.food_density,
            ..Default::default()
        };
        worm.step_with_inputs(config.dt_ms, &inputs);
        speed_sum += worm.speed_um_s;
        command_bias_sum += worm.command_bias();
        sample_count += 1.0;

        if step % config.sample_every.max(1) == 0 {
            traces.push(CelegansAssayTracePoint {
                episode,
                step,
                x_um: worm.x_um,
                y_um: worm.y_um,
                angle_rad: worm.angle_rad,
                speed_um_s: worm.speed_um_s,
                command_bias: worm.command_bias(),
                steering_bias: worm.head_steering_bias(),
                attractant_left: config.drive_strength,
                attractant_right: config.drive_strength,
                temperature_left_c: CELEGANS_PREFERRED_TEMP_C,
                temperature_right_c: CELEGANS_PREFERRED_TEMP_C,
                immersion,
                food_density: config.food_density,
                energy_reserve: worm.energy_reserve(),
                gut_content: worm.gut_content(),
                pharyngeal_pumping_hz: worm.pharyngeal_pumping_hz(),
            });
        }
    }

    (
        speed_sum / sample_count.max(1.0),
        command_bias_sum / sample_count.max(1.0),
        traces,
    )
}

fn head_sensor_positions(worm: &CelegansOrganism) -> ((f32, f32), (f32, f32)) {
    let head_x = worm.x_um + worm.angle_rad.cos() * SENSOR_FORWARD_OFFSET_UM;
    let head_y = worm.y_um + worm.angle_rad.sin() * SENSOR_FORWARD_OFFSET_UM;
    let lateral_x = -worm.angle_rad.sin() * SENSOR_LATERAL_OFFSET_UM;
    let lateral_y = worm.angle_rad.cos() * SENSOR_LATERAL_OFFSET_UM;

    (
        (head_x + lateral_x, head_y + lateral_y),
        (head_x - lateral_x, head_y - lateral_y),
    )
}

fn gaussian_field(x: f32, y: f32, source_x: f32, source_y: f32, sigma_um: f32) -> f32 {
    let dx = x - source_x;
    let dy = y - source_y;
    let variance = (sigma_um * sigma_um).max(1.0);
    (-(dx * dx + dy * dy) / (2.0 * variance))
        .exp()
        .clamp(0.0, 1.0)
}

fn temperature_at(
    x_um: f32,
    min_x_um: f32,
    max_x_um: f32,
    min_temp_c: f32,
    max_temp_c: f32,
) -> f32 {
    let span = (max_x_um - min_x_um).max(1.0);
    let alpha = ((x_um - min_x_um) / span).clamp(0.0, 1.0);
    min_temp_c + (max_temp_c - min_temp_c) * alpha
}

fn pulse_touch(step: u32, pulse_steps: u32, strength: f32) -> f32 {
    if step < pulse_steps {
        strength
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        run_chemotaxis_assay, run_crawl_swim_assay, run_thermotaxis_assay,
        CelegansChemotaxisAssayConfig, CelegansCrawlSwimAssayConfig,
        CelegansThermotaxisAssayConfig,
    };

    #[test]
    fn test_chemotaxis_assay_shows_approach_and_steering_alignment() {
        let result = run_chemotaxis_assay(&CelegansChemotaxisAssayConfig::default());
        assert!(
            result.metric_value > 0.0,
            "chemotaxis should improve local attractant exposure"
        );
        assert!(
            result
                .traces
                .iter()
                .any(|trace| trace.steering_bias.abs() > 0.0),
            "chemotaxis assay should produce explicit steering bias"
        );
    }

    #[test]
    fn test_thermotaxis_assay_reduces_temperature_error() {
        let result = run_thermotaxis_assay(&CelegansThermotaxisAssayConfig::default());
        assert!(
            result.metric_value > 0.0,
            "thermotaxis should reduce distance from preferred temperature"
        );
    }

    #[test]
    fn test_crawl_swim_assay_separates_modes() {
        let result = run_crawl_swim_assay(&CelegansCrawlSwimAssayConfig::default());
        assert!(
            result.metric_value > 0.0,
            "crawl/swim assay should separate locomotor modes"
        );
        assert_eq!(
            result
                .traces
                .iter()
                .filter(|trace| trace.immersion == 0.0)
                .count(),
            result
                .traces
                .iter()
                .filter(|trace| trace.immersion == 1.0)
                .count(),
        );
    }
}
