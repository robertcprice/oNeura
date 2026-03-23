//! Focused assay helpers for the fly lane.
//!
//! These assays keep the hardening pass measurable: local sensory
//! perturbations should first move the relevant brain regions, then shift the
//! downstream motor readout.

use super::{DrosophilaScale, DrosophilaSim, ExperimentResult, TerrariumFlyInputs};
use crate::types::NTType;

const PROBE_WINDOWS: u32 = 4;
const STEERING_BODY_STEPS: u32 = 12;

#[derive(Debug, Clone, Copy)]
pub struct SensoryCircuitProbe {
    pub al_input_current: f32,
    pub ol_lam_left_input_current: f32,
    pub ol_lam_right_input_current: f32,
    pub sez_input_current: f32,
    pub al_activity: f32,
    pub ol_lam_left_activity: f32,
    pub ol_lam_right_activity: f32,
    pub sez_activity: f32,
    pub cx_activity: f32,
    pub dn_activity: f32,
    pub vnc_forward_activity: f32,
    pub vnc_backward_activity: f32,
    pub vnc_left_activity: f32,
    pub vnc_right_activity: f32,
    pub speed: f32,
    pub turn: f32,
}

impl SensoryCircuitProbe {
    pub fn motor_drive(&self) -> f32 {
        self.speed.abs() + self.turn.abs()
    }
}

#[derive(Debug)]
pub struct OlfactoryAssayResult {
    pub baseline_probe: SensoryCircuitProbe,
    pub odor_probe: SensoryCircuitProbe,
    pub behavior: ExperimentResult,
}

#[derive(Debug)]
pub struct PhototaxisAssayResult {
    pub left_light_probe: SensoryCircuitProbe,
    pub right_light_probe: SensoryCircuitProbe,
    pub behavior: ExperimentResult,
}

#[derive(Debug)]
pub struct ThermotaxisAssayResult {
    pub preferred_probe: SensoryCircuitProbe,
    pub warm_probe: SensoryCircuitProbe,
    pub behavior: ExperimentResult,
}

#[derive(Debug)]
pub struct ForagingAssayResult {
    pub behavior: ExperimentResult,
}

#[derive(Debug, Clone, Copy)]
pub struct FeedingReinforcementProbe {
    pub sez_input_current: f32,
    pub dan_input_current: f32,
    pub sez_activity: f32,
    pub dan_activity: f32,
    pub mb_kc_dopamine_nm: f32,
    pub mbon_dopamine_nm: f32,
}

#[derive(Debug)]
pub struct FeedingReinforcementAssayResult {
    pub baseline_probe: FeedingReinforcementProbe,
    pub ingestion_probe: FeedingReinforcementProbe,
}

#[derive(Debug, Clone, Copy)]
pub struct LateralOlfactoryProbe {
    pub al_left_input_current: f32,
    pub al_right_input_current: f32,
    pub lh_left_input_current: f32,
    pub lh_right_input_current: f32,
    pub cx_left_activity: f32,
    pub cx_right_activity: f32,
    pub dn_left_activity: f32,
    pub dn_right_activity: f32,
    pub vnc_left_activity: f32,
    pub vnc_right_activity: f32,
    pub cx_activity: f32,
    pub dn_activity: f32,
    pub turn: f32,
    pub speed: f32,
}

#[derive(Debug)]
pub struct OdorSteeringAssayResult {
    pub left_biased_probe: LateralOlfactoryProbe,
    pub right_biased_probe: LateralOlfactoryProbe,
}

#[derive(Debug, Clone, Copy)]
pub struct CircadianProbe {
    pub phase_hours: f32,
    pub circadian_activity: f32,
    pub mean_firing_rate_hz: f32,
    pub mean_arousal_input_current: f32,
    pub mean_motor_drive: f32,
}

#[derive(Debug)]
pub struct CircadianAssayResult {
    pub day_probe: CircadianProbe,
    pub night_probe: CircadianProbe,
    pub behavior: ExperimentResult,
}

fn configured_assay_sim(scale: DrosophilaScale, seed: u64) -> DrosophilaSim {
    let mut sim = DrosophilaSim::new(scale, seed);
    sim.brain.enable_gpu = false;
    sim.brain.enable_circadian = false;
    sim.brain.enable_pharmacology = false;
    sim.brain.enable_glia = false;
    sim
}

fn reset_probe_state(sim: &mut DrosophilaSim) {
    sim.fep = None;
    sim.reset_episode(32.0, 32.0);
    sim.body.heading = 0.0;
    sim.body.pitch = 0.0;
    sim.body.speed = 0.0;
    sim.body.is_flying = false;
    sim.body.energy = 80.0;
    sim.body.temperature = 18.0;
    sim.brain.reset_spike_counts();
    sim.brain
        .neurons
        .external_current
        .iter_mut()
        .for_each(|current| *current = 0.0);
    sim.brain
        .neurons
        .fired
        .iter_mut()
        .for_each(|fired| *fired = 0);
}

fn normalized_activity(spike_sum: u32, neurons: usize, steps: u32) -> f32 {
    let denom = (neurons.max(1) as f32) * (steps.max(1) as f32);
    spike_sum as f32 / denom
}

fn mean_external_current(sim: &DrosophilaSim, range: std::ops::Range<usize>) -> f32 {
    let len = range.len().max(1) as f32;
    let mut sum = 0.0f32;
    for idx in range {
        sum += sim.brain.neurons.external_current[idx];
    }
    sum / len
}

fn mean_nt_concentration(sim: &DrosophilaSim, range: std::ops::Range<usize>, nt_idx: usize) -> f32 {
    let len = range.len().max(1) as f32;
    let mut sum = 0.0f32;
    for idx in range {
        sum += sim.brain.neurons.nt_conc[idx][nt_idx];
    }
    sum / len
}

pub fn probe_manual_sensory_response(
    sim: &mut DrosophilaSim,
    odorant: f32,
    left_light: f32,
    right_light: f32,
    temperature: f32,
) -> SensoryCircuitProbe {
    reset_probe_state(sim);

    let al_range = sim.layout.range("AL");
    let sez_range = sim.layout.range("SEZ");
    let cx_range = sim.layout.range("CX");
    let dn_range = sim.layout.range("DN");
    let vnc_forward = sim.vnc_forward_range();
    let vnc_backward = sim.vnc_backward_range();
    let vnc_left = sim.vnc_left_range();
    let vnc_right = sim.vnc_right_range();

    let lam_range = sim.layout.range("OL_LAM");
    let lam_half = lam_range.len() / 2;
    let lam_left = lam_range.start..lam_range.start + lam_half;
    let lam_right = lam_range.start + lam_half..lam_range.end;

    let mut al_spikes = 0u32;
    let mut al_input_current = 0.0f32;
    let mut lam_left_input_current = 0.0f32;
    let mut lam_right_input_current = 0.0f32;
    let mut lam_left_spikes = 0u32;
    let mut lam_right_spikes = 0u32;
    let mut sez_input_current = 0.0f32;
    let mut sez_spikes = 0u32;
    let mut cx_spikes = 0u32;
    let mut dn_spikes = 0u32;
    let mut vnc_fwd_spikes = 0u32;
    let mut vnc_bwd_spikes = 0u32;
    let mut vnc_left_spikes = 0u32;
    let mut vnc_right_spikes = 0u32;

    let total_steps = sim.neural_steps_per_body * PROBE_WINDOWS;
    for _ in 0..PROBE_WINDOWS {
        sim.brain
            .neurons
            .external_current
            .iter_mut()
            .for_each(|current| *current = 0.0);
        sim.encode_manual_sensory(odorant, left_light, right_light, temperature);

        al_input_current += mean_external_current(sim, al_range.clone());
        lam_left_input_current += mean_external_current(sim, lam_left.clone());
        lam_right_input_current += mean_external_current(sim, lam_right.clone());
        sez_input_current += mean_external_current(sim, sez_range.clone());
        sim.brain
            .neurons
            .external_current
            .iter_mut()
            .for_each(|current| *current = 0.0);

        let al_prev = sim.brain.spike_count_range_sum(al_range.clone());
        let lam_left_prev = sim.brain.spike_count_range_sum(lam_left.clone());
        let lam_right_prev = sim.brain.spike_count_range_sum(lam_right.clone());
        let sez_prev = sim.brain.spike_count_range_sum(sez_range.clone());
        let cx_prev = sim.brain.spike_count_range_sum(cx_range.clone());
        let dn_prev = sim.brain.spike_count_range_sum(dn_range.clone());
        let vnc_fwd_prev = sim.brain.spike_count_range_sum(vnc_forward.clone());
        let vnc_bwd_prev = sim.brain.spike_count_range_sum(vnc_backward.clone());
        let vnc_left_prev = sim.brain.spike_count_range_sum(vnc_left.clone());
        let vnc_right_prev = sim.brain.spike_count_range_sum(vnc_right.clone());

        sim.run_neural_window_with_drive(sim.neural_steps_per_body, |sim| {
            sim.encode_manual_sensory(odorant, left_light, right_light, temperature);
        });

        al_spikes += sim
            .brain
            .spike_count_range_sum(al_range.clone())
            .saturating_sub(al_prev) as u32;
        lam_left_spikes += sim
            .brain
            .spike_count_range_sum(lam_left.clone())
            .saturating_sub(lam_left_prev) as u32;
        lam_right_spikes += sim
            .brain
            .spike_count_range_sum(lam_right.clone())
            .saturating_sub(lam_right_prev) as u32;
        sez_spikes += sim
            .brain
            .spike_count_range_sum(sez_range.clone())
            .saturating_sub(sez_prev) as u32;
        cx_spikes += sim
            .brain
            .spike_count_range_sum(cx_range.clone())
            .saturating_sub(cx_prev) as u32;
        dn_spikes += sim
            .brain
            .spike_count_range_sum(dn_range.clone())
            .saturating_sub(dn_prev) as u32;
        vnc_fwd_spikes += sim
            .brain
            .spike_count_range_sum(vnc_forward.clone())
            .saturating_sub(vnc_fwd_prev) as u32;
        vnc_bwd_spikes += sim
            .brain
            .spike_count_range_sum(vnc_backward.clone())
            .saturating_sub(vnc_bwd_prev) as u32;
        vnc_left_spikes += sim
            .brain
            .spike_count_range_sum(vnc_left.clone())
            .saturating_sub(vnc_left_prev) as u32;
        vnc_right_spikes += sim
            .brain
            .spike_count_range_sum(vnc_right.clone())
            .saturating_sub(vnc_right_prev) as u32;
    }

    sim.brain.sync_shadow_from_gpu();

    let vnc_left_activity = normalized_activity(vnc_left_spikes, vnc_left.len(), total_steps);
    let vnc_right_activity = normalized_activity(vnc_right_spikes, vnc_right.len(), total_steps);
    let vnc_forward_activity = normalized_activity(vnc_fwd_spikes, vnc_forward.len(), total_steps);
    let vnc_backward_activity =
        normalized_activity(vnc_bwd_spikes, vnc_backward.len(), total_steps);

    let total_lr = vnc_left_activity + vnc_right_activity;
    let turn = if total_lr > 0.0 {
        (vnc_left_activity - vnc_right_activity) / total_lr
    } else {
        0.0
    };
    let speed = ((vnc_forward_activity - vnc_backward_activity * 0.5) / 0.05).clamp(0.0, 1.0);

    SensoryCircuitProbe {
        al_input_current: al_input_current / PROBE_WINDOWS as f32,
        ol_lam_left_input_current: lam_left_input_current / PROBE_WINDOWS as f32,
        ol_lam_right_input_current: lam_right_input_current / PROBE_WINDOWS as f32,
        sez_input_current: sez_input_current / PROBE_WINDOWS as f32,
        al_activity: normalized_activity(al_spikes, al_range.len(), total_steps),
        ol_lam_left_activity: normalized_activity(lam_left_spikes, lam_left.len(), total_steps),
        ol_lam_right_activity: normalized_activity(lam_right_spikes, lam_right.len(), total_steps),
        sez_activity: normalized_activity(sez_spikes, sez_range.len(), total_steps),
        cx_activity: normalized_activity(cx_spikes, cx_range.len(), total_steps),
        dn_activity: normalized_activity(dn_spikes, dn_range.len(), total_steps),
        vnc_forward_activity,
        vnc_backward_activity,
        vnc_left_activity,
        vnc_right_activity,
        speed,
        turn,
    }
}

pub fn run_olfactory_assay(
    scale: DrosophilaScale,
    seed: u64,
    episodes: u32,
) -> OlfactoryAssayResult {
    let mut baseline_sim = configured_assay_sim(scale, seed);
    let baseline_probe = probe_manual_sensory_response(&mut baseline_sim, 0.0, 0.0, 0.0, 18.0);

    let mut odor_sim = configured_assay_sim(scale, seed);
    let odor_probe = probe_manual_sensory_response(&mut odor_sim, 0.8, 0.0, 0.0, 18.0);

    let mut behavior_sim = configured_assay_sim(scale, seed);
    let behavior = behavior_sim.run_olfactory(episodes);

    OlfactoryAssayResult {
        baseline_probe,
        odor_probe,
        behavior,
    }
}

pub fn run_phototaxis_assay(
    scale: DrosophilaScale,
    seed: u64,
    episodes: u32,
) -> PhototaxisAssayResult {
    let mut left_light_sim = configured_assay_sim(scale, seed);
    let left_light_probe = probe_manual_sensory_response(&mut left_light_sim, 0.0, 1.0, 0.0, 18.0);

    let mut right_light_sim = configured_assay_sim(scale, seed);
    let right_light_probe =
        probe_manual_sensory_response(&mut right_light_sim, 0.0, 0.0, 1.0, 18.0);

    let mut behavior_sim = configured_assay_sim(scale, seed);
    let behavior = behavior_sim.run_phototaxis(episodes);

    PhototaxisAssayResult {
        left_light_probe,
        right_light_probe,
        behavior,
    }
}

pub fn run_thermotaxis_assay(
    scale: DrosophilaScale,
    seed: u64,
    episodes: u32,
) -> ThermotaxisAssayResult {
    let mut preferred_sim = configured_assay_sim(scale, seed);
    let preferred_probe = probe_manual_sensory_response(&mut preferred_sim, 0.0, 0.0, 0.0, 18.0);

    let mut warm_sim = configured_assay_sim(scale, seed);
    let warm_probe = probe_manual_sensory_response(&mut warm_sim, 0.0, 0.0, 0.0, 30.0);

    let mut behavior_sim = configured_assay_sim(scale, seed);
    let behavior = behavior_sim.run_thermotaxis(episodes);

    ThermotaxisAssayResult {
        preferred_probe,
        warm_probe,
        behavior,
    }
}

pub fn run_foraging_assay(scale: DrosophilaScale, seed: u64, episodes: u32) -> ForagingAssayResult {
    let mut sim = configured_assay_sim(scale, seed);
    let behavior = sim.run_foraging(episodes);
    ForagingAssayResult { behavior }
}

fn probe_feeding_reinforcement_response(
    sim: &mut DrosophilaSim,
    consumed_food: f32,
    hunger_signal: f32,
) -> FeedingReinforcementProbe {
    reset_probe_state(sim);
    sim.set_energy(45.0);
    sim.set_homeostatic_inputs(hunger_signal, 12.0, 1.0);
    if consumed_food > 1.0e-6 {
        sim.register_ingestion_feedback(consumed_food, hunger_signal);
    }

    let sez_range = sim.layout.range("SEZ");
    let dan_range = sim.layout.range("DAN");
    let mb_kc_range = sim.layout.range("MB_KC");
    let mbon_range = sim.layout.range("MBON");

    let total_steps = sim.neural_steps_per_body * PROBE_WINDOWS;
    let mut sez_input_sum = 0.0f32;
    let mut dan_input_sum = 0.0f32;
    let mut sez_spikes = 0u32;
    let mut dan_spikes = 0u32;
    let mut mb_kc_dopamine_sum = 0.0f32;
    let mut mbon_dopamine_sum = 0.0f32;

    for _ in 0..PROBE_WINDOWS {
        sim.brain
            .neurons
            .external_current
            .iter_mut()
            .for_each(|current| *current = 0.0);
        sim.stimulate_recent_feeding_feedback();
        sim.stimulate_internal_arousal();

        sez_input_sum += mean_external_current(sim, sez_range.clone());
        dan_input_sum += mean_external_current(sim, dan_range.clone());
        sim.brain
            .neurons
            .external_current
            .iter_mut()
            .for_each(|current| *current = 0.0);

        let sez_prev = sim.brain.spike_count_range_sum(sez_range.clone());
        let dan_prev = sim.brain.spike_count_range_sum(dan_range.clone());

        sim.run_neural_window_with_drive(sim.neural_steps_per_body, |sim| {
            sim.stimulate_recent_feeding_feedback();
            sim.stimulate_internal_arousal();
        });

        sez_spikes += sim
            .brain
            .spike_count_range_sum(sez_range.clone())
            .saturating_sub(sez_prev) as u32;
        dan_spikes += sim
            .brain
            .spike_count_range_sum(dan_range.clone())
            .saturating_sub(dan_prev) as u32;
        mb_kc_dopamine_sum +=
            mean_nt_concentration(sim, mb_kc_range.clone(), NTType::Dopamine.index());
        mbon_dopamine_sum +=
            mean_nt_concentration(sim, mbon_range.clone(), NTType::Dopamine.index());
    }

    sim.brain.sync_shadow_from_gpu();

    FeedingReinforcementProbe {
        sez_input_current: sez_input_sum / PROBE_WINDOWS as f32,
        dan_input_current: dan_input_sum / PROBE_WINDOWS as f32,
        sez_activity: normalized_activity(sez_spikes, sez_range.len(), total_steps),
        dan_activity: normalized_activity(dan_spikes, dan_range.len(), total_steps),
        mb_kc_dopamine_nm: mb_kc_dopamine_sum / PROBE_WINDOWS as f32,
        mbon_dopamine_nm: mbon_dopamine_sum / PROBE_WINDOWS as f32,
    }
}

pub fn run_feeding_reinforcement_assay(
    scale: DrosophilaScale,
    seed: u64,
) -> FeedingReinforcementAssayResult {
    let mut baseline_sim = configured_assay_sim(scale, seed);
    let baseline_probe = probe_feeding_reinforcement_response(&mut baseline_sim, 0.0, 0.8);

    let mut ingestion_sim = configured_assay_sim(scale, seed);
    let ingestion_probe = probe_feeding_reinforcement_response(&mut ingestion_sim, 0.03, 0.8);

    FeedingReinforcementAssayResult {
        baseline_probe,
        ingestion_probe,
    }
}

fn probe_manual_odor_geometry_response(
    sim: &mut DrosophilaSim,
    left_odorant: f32,
    right_odorant: f32,
    use_cx_bridge: bool,
) -> LateralOlfactoryProbe {
    reset_probe_state(sim);
    let odorant = left_odorant.max(right_odorant);

    let (al_left, al_right) = sim.layout.bilateral_excitatory_ranges("AL");
    let (lh_left, lh_right) = sim.layout.bilateral_excitatory_ranges("LH");

    let cx_range = sim.layout.range("CX");
    let (cx_left, cx_right) = sim.layout.bilateral_excitatory_ranges("CX");
    let dn_range = sim.layout.range("DN");
    let (dn_left, dn_right) = sim.layout.bilateral_excitatory_ranges("DN");
    let vnc_left = sim.vnc_left_range();
    let vnc_right = sim.vnc_right_range();

    let total_steps = sim.neural_steps_per_body * PROBE_WINDOWS;
    let mut al_left_input_current = 0.0f32;
    let mut al_right_input_current = 0.0f32;
    let mut lh_left_input_current = 0.0f32;
    let mut lh_right_input_current = 0.0f32;
    let mut cx_spikes = 0u32;
    let mut dn_spikes = 0u32;
    let mut cx_left_spikes = 0u32;
    let mut cx_right_spikes = 0u32;
    let mut dn_left_spikes = 0u32;
    let mut dn_right_spikes = 0u32;
    let mut vnc_left_spikes = 0u32;
    let mut vnc_right_spikes = 0u32;

    for _ in 0..PROBE_WINDOWS {
        sim.brain
            .neurons
            .external_current
            .iter_mut()
            .for_each(|current| *current = 0.0);
        sim.encode_manual_sensory(odorant, 0.0, 0.0, 18.0);
        sim.stimulate_manual_odor_geometry_with_bridge(left_odorant, right_odorant, use_cx_bridge);
        sim.stimulate_internal_arousal();

        al_left_input_current += mean_external_current(sim, al_left.clone());
        al_right_input_current += mean_external_current(sim, al_right.clone());
        lh_left_input_current += mean_external_current(sim, lh_left.clone());
        lh_right_input_current += mean_external_current(sim, lh_right.clone());
        sim.brain
            .neurons
            .external_current
            .iter_mut()
            .for_each(|current| *current = 0.0);

        let cx_prev = sim.brain.spike_count_range_sum(cx_range.clone());
        let dn_prev = sim.brain.spike_count_range_sum(dn_range.clone());
        let cx_left_prev = sim.brain.spike_count_range_sum(cx_left.clone());
        let cx_right_prev = sim.brain.spike_count_range_sum(cx_right.clone());
        let dn_left_prev = sim.brain.spike_count_range_sum(dn_left.clone());
        let dn_right_prev = sim.brain.spike_count_range_sum(dn_right.clone());
        let vnc_left_prev = sim.brain.spike_count_range_sum(vnc_left.clone());
        let vnc_right_prev = sim.brain.spike_count_range_sum(vnc_right.clone());

        sim.run_neural_window_with_drive(sim.neural_steps_per_body, |sim| {
            sim.encode_manual_sensory(odorant, 0.0, 0.0, 18.0);
            sim.stimulate_manual_odor_geometry_with_bridge(
                left_odorant,
                right_odorant,
                use_cx_bridge,
            );
            sim.stimulate_internal_arousal();
        });

        cx_spikes += sim
            .brain
            .spike_count_range_sum(cx_range.clone())
            .saturating_sub(cx_prev) as u32;
        dn_spikes += sim
            .brain
            .spike_count_range_sum(dn_range.clone())
            .saturating_sub(dn_prev) as u32;
        cx_left_spikes += sim
            .brain
            .spike_count_range_sum(cx_left.clone())
            .saturating_sub(cx_left_prev) as u32;
        cx_right_spikes += sim
            .brain
            .spike_count_range_sum(cx_right.clone())
            .saturating_sub(cx_right_prev) as u32;
        dn_left_spikes += sim
            .brain
            .spike_count_range_sum(dn_left.clone())
            .saturating_sub(dn_left_prev) as u32;
        dn_right_spikes += sim
            .brain
            .spike_count_range_sum(dn_right.clone())
            .saturating_sub(dn_right_prev) as u32;
        vnc_left_spikes += sim
            .brain
            .spike_count_range_sum(vnc_left.clone())
            .saturating_sub(vnc_left_prev) as u32;
        vnc_right_spikes += sim
            .brain
            .spike_count_range_sum(vnc_right.clone())
            .saturating_sub(vnc_right_prev) as u32;
    }

    reset_probe_state(sim);
    let mut turn_sum = 0.0f32;
    let mut speed_sum = 0.0f32;
    for _ in 0..STEERING_BODY_STEPS {
        let report = sim.body_step_terrarium_inputs(
            TerrariumFlyInputs {
                odorant,
                left_odorant,
                right_odorant,
                temperature: 18.0,
                ..TerrariumFlyInputs::default()
            },
            1.225,
        );
        turn_sum += report.turn;
        speed_sum += report.speed;
    }

    LateralOlfactoryProbe {
        al_left_input_current: al_left_input_current / PROBE_WINDOWS as f32,
        al_right_input_current: al_right_input_current / PROBE_WINDOWS as f32,
        lh_left_input_current: lh_left_input_current / PROBE_WINDOWS as f32,
        lh_right_input_current: lh_right_input_current / PROBE_WINDOWS as f32,
        cx_left_activity: normalized_activity(cx_left_spikes, cx_left.len(), total_steps),
        cx_right_activity: normalized_activity(cx_right_spikes, cx_right.len(), total_steps),
        dn_left_activity: normalized_activity(dn_left_spikes, dn_left.len(), total_steps),
        dn_right_activity: normalized_activity(dn_right_spikes, dn_right.len(), total_steps),
        vnc_left_activity: normalized_activity(vnc_left_spikes, vnc_left.len(), total_steps),
        vnc_right_activity: normalized_activity(vnc_right_spikes, vnc_right.len(), total_steps),
        cx_activity: normalized_activity(cx_spikes, cx_range.len(), total_steps),
        dn_activity: normalized_activity(dn_spikes, dn_range.len(), total_steps),
        turn: turn_sum / STEERING_BODY_STEPS as f32,
        speed: speed_sum / STEERING_BODY_STEPS as f32,
    }
}

pub fn run_odor_steering_assay(scale: DrosophilaScale, seed: u64) -> OdorSteeringAssayResult {
    let mut left_sim = configured_assay_sim(scale, seed);
    let left_biased_probe = probe_manual_odor_geometry_response(&mut left_sim, 1.0, 0.0, false);

    let mut right_sim = configured_assay_sim(scale, seed);
    let right_biased_probe = probe_manual_odor_geometry_response(&mut right_sim, 0.0, 1.0, false);

    OdorSteeringAssayResult {
        left_biased_probe,
        right_biased_probe,
    }
}

fn probe_circadian_response(
    sim: &mut DrosophilaSim,
    phase_hours: f32,
    circadian_activity: f32,
) -> CircadianProbe {
    reset_probe_state(sim);
    sim.set_homeostatic_inputs(0.15, phase_hours, circadian_activity);
    sim.set_energy(70.0);

    let arousal_range = sim.neuromod_arousal_range();
    let mut arousal_input_sum = 0.0f32;
    for _ in 0..PROBE_WINDOWS {
        sim.brain
            .neurons
            .external_current
            .iter_mut()
            .for_each(|current| *current = 0.0);
        sim.stimulate_homeostatic_feeding_drive();
        sim.stimulate_internal_arousal();
        arousal_input_sum += mean_external_current(sim, arousal_range.clone());
        sim.run_neural_window(sim.neural_steps_per_body);
    }

    let mean_firing_rate_hz = sim.mean_firing_rate();

    reset_probe_state(sim);
    sim.set_homeostatic_inputs(0.15, phase_hours, circadian_activity);
    sim.set_energy(70.0);

    let mut motor_drive_sum = 0.0f32;
    for _ in 0..PROBE_WINDOWS {
        let report = sim.body_step_terrarium_inputs(
            TerrariumFlyInputs {
                temperature: 18.0,
                ..TerrariumFlyInputs::default()
            },
            1.225,
        );
        motor_drive_sum += report.activity_score();
    }

    CircadianProbe {
        phase_hours,
        circadian_activity,
        mean_firing_rate_hz,
        mean_arousal_input_current: arousal_input_sum / PROBE_WINDOWS as f32,
        mean_motor_drive: motor_drive_sum / PROBE_WINDOWS as f32,
    }
}

pub fn run_circadian_assay(
    scale: DrosophilaScale,
    seed: u64,
    episodes: u32,
) -> CircadianAssayResult {
    let mut day_sim = configured_assay_sim(scale, seed);
    let day_probe = probe_circadian_response(&mut day_sim, 12.0, 1.15);

    let mut night_sim = configured_assay_sim(scale, seed);
    let night_probe = probe_circadian_response(&mut night_sim, 0.0, 0.2);

    let mut behavior_sim = configured_assay_sim(scale, seed);
    let behavior = behavior_sim.run_circadian(episodes);

    CircadianAssayResult {
        day_probe,
        night_probe,
        behavior,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visual_perturbation_reaches_lamina_before_behavior_baseline() {
        let result = run_phototaxis_assay(DrosophilaScale::Tiny, 42, 2);

        assert!(
            result.left_light_probe.ol_lam_left_input_current
                > result.left_light_probe.ol_lam_right_input_current,
            "left-biased light should preferentially drive the left lamina"
        );
        assert!(
            result.right_light_probe.ol_lam_right_input_current
                > result.right_light_probe.ol_lam_left_input_current,
            "right-biased light should preferentially drive the right lamina"
        );
        assert!(
            result.behavior.metric_value.is_finite(),
            "phototaxis assay should return a finite displacement metric"
        );
        assert!(
            !result.behavior.trajectories.is_empty(),
            "phototaxis assay should emit trajectories"
        );
    }

    #[test]
    fn test_odor_perturbation_reaches_al_before_behavior_baseline() {
        let result = run_olfactory_assay(DrosophilaScale::Tiny, 42, 2);

        assert!(
            result.odor_probe.al_input_current > result.baseline_probe.al_input_current,
            "odor stimulation should increase antennal-lobe activity"
        );
        assert!(
            result.behavior.metric_value.is_finite(),
            "olfactory assay should return a finite improvement metric"
        );
        assert!(
            !result.behavior.trajectories.is_empty(),
            "olfactory assay should emit trajectories"
        );
    }

    #[test]
    fn test_thermal_perturbation_reaches_sez_before_behavior_baseline() {
        let result = run_thermotaxis_assay(DrosophilaScale::Tiny, 42, 2);

        assert!(
            result.warm_probe.sez_input_current > result.preferred_probe.sez_input_current,
            "thermal stress should increase SEZ thermal-drive activity"
        );
        assert!(
            result.behavior.metric_value.is_finite(),
            "thermotaxis assay should return a finite temperature metric"
        );
        assert!(
            !result.behavior.trajectories.is_empty(),
            "thermotaxis assay should emit trajectories"
        );
    }

    #[test]
    fn test_foraging_assay_returns_behavior_baseline() {
        let result = run_foraging_assay(DrosophilaScale::Tiny, 42, 1);
        assert_eq!(result.behavior.metric_name, "mean_food_visits");
        assert!(result.behavior.metric_value.is_finite());
        assert!(!result.behavior.trajectories.is_empty());
    }

    #[test]
    fn test_ingestion_feedback_reaches_dan_without_direct_current() {
        let result = run_feeding_reinforcement_assay(DrosophilaScale::Tiny, 42);

        assert!(
            result.ingestion_probe.sez_input_current > result.baseline_probe.sez_input_current,
            "ingestion-derived feedback should increase SEZ sensory drive before any downstream teaching signal"
        );
        assert!(
            result.ingestion_probe.sez_activity > result.baseline_probe.sez_activity,
            "ingestion-derived feedback should raise SEZ activity above baseline"
        );
        assert!(
            result.ingestion_probe.dan_input_current.abs() <= 1.0e-6,
            "feeding reinforcement must not inject external current directly into DAN (input={:.4})",
            result.ingestion_probe.dan_input_current,
        );
        assert!(
            result.ingestion_probe.dan_activity > result.baseline_probe.dan_activity,
            "SEZ-mediated feeding feedback should recruit DAN through the explicit circuit path (baseline={:.4}, ingestion={:.4})",
            result.baseline_probe.dan_activity,
            result.ingestion_probe.dan_activity,
        );
        assert!(
            result.ingestion_probe.mbon_dopamine_nm > result.baseline_probe.mbon_dopamine_nm
                || result.ingestion_probe.mb_kc_dopamine_nm > result.baseline_probe.mb_kc_dopamine_nm,
            "DAN recruitment should raise downstream dopamine in mushroom-body targets (MBON {:.3}->{:.3}, MB_KC {:.3}->{:.3})",
            result.baseline_probe.mbon_dopamine_nm,
            result.ingestion_probe.mbon_dopamine_nm,
            result.baseline_probe.mb_kc_dopamine_nm,
            result.ingestion_probe.mb_kc_dopamine_nm,
        );
    }

    #[test]
    fn test_lateral_odor_bias_reaches_al_and_lh() {
        let result = run_odor_steering_assay(DrosophilaScale::Tiny, 42);
        assert!(
            result.left_biased_probe.al_left_input_current
                > result.left_biased_probe.al_right_input_current,
            "left-biased odor should drive the left AL half more strongly"
        );
        assert!(
            result.right_biased_probe.al_right_input_current
                > result.right_biased_probe.al_left_input_current,
            "right-biased odor should drive the right AL half more strongly"
        );
        assert!(
            result.left_biased_probe.lh_left_input_current
                > result.left_biased_probe.lh_right_input_current,
            "left-biased odor should drive the left LH half more strongly"
        );
        assert!(
            result.right_biased_probe.lh_right_input_current
                > result.right_biased_probe.lh_left_input_current,
            "right-biased odor should drive the right LH half more strongly"
        );
        assert!(
            result.left_biased_probe.cx_activity.is_finite()
                && result.right_biased_probe.cx_activity.is_finite(),
            "odor bias should keep intermediate CX activity bounded"
        );
        assert!(
            result.left_biased_probe.cx_activity > 0.0
                && result.right_biased_probe.cx_activity > 0.0,
            "odor bias should still recruit the CX through the explicit olfactory path"
        );
        assert!(
            result.left_biased_probe.dn_left_activity > result.left_biased_probe.dn_right_activity,
            "left-biased odor should propagate to the left DN half (left={:.4}, right={:.4})",
            result.left_biased_probe.dn_left_activity,
            result.left_biased_probe.dn_right_activity,
        );
        assert!(
            result.right_biased_probe.dn_right_activity
                > result.right_biased_probe.dn_left_activity,
            "right-biased odor should propagate to the right DN half (left={:.4}, right={:.4})",
            result.right_biased_probe.dn_left_activity,
            result.right_biased_probe.dn_right_activity,
        );
        assert!(
            result.left_biased_probe.vnc_left_activity
                > result.left_biased_probe.vnc_right_activity,
            "left-biased odor should drive the left VNC turn pool more strongly (left={:.4}, right={:.4})",
            result.left_biased_probe.vnc_left_activity,
            result.left_biased_probe.vnc_right_activity,
        );
        assert!(
            result.right_biased_probe.vnc_right_activity
                > result.right_biased_probe.vnc_left_activity,
            "right-biased odor should drive the right VNC turn pool more strongly (left={:.4}, right={:.4})",
            result.right_biased_probe.vnc_left_activity,
            result.right_biased_probe.vnc_right_activity,
        );
        assert!(
            result.left_biased_probe.turn.is_finite() && result.right_biased_probe.turn.is_finite(),
            "odor-geometry assay should keep downstream steering readouts bounded"
        );
        assert!(
            result.left_biased_probe.turn > 0.0,
            "left-biased odor should produce a positive turn without the CX bridge (turn={:.4})",
            result.left_biased_probe.turn,
        );
        assert!(
            result.right_biased_probe.turn < 0.0,
            "right-biased odor should produce a negative turn without the CX bridge (turn={:.4})",
            result.right_biased_probe.turn,
        );
        assert!(
            result.left_biased_probe.speed.is_finite()
                && result.right_biased_probe.speed.is_finite(),
            "odor-geometry assay should keep downstream speed readouts bounded"
        );
    }

    #[test]
    fn test_circadian_assay_uses_homeostatic_state() {
        let result = run_circadian_assay(DrosophilaScale::Tiny, 42, 2);

        assert!(
            result.day_probe.mean_arousal_input_current
                > result.night_probe.mean_arousal_input_current,
            "day homeostatic state should produce higher arousal current than night (day={:.4}, night={:.4})",
            result.day_probe.mean_arousal_input_current,
            result.night_probe.mean_arousal_input_current,
        );
        assert!(
            result.day_probe.mean_motor_drive >= result.night_probe.mean_motor_drive,
            "day homeostatic state should produce at least as much motor drive as night (day={:.4}, night={:.4})",
            result.day_probe.mean_motor_drive,
            result.night_probe.mean_motor_drive,
        );
        assert_eq!(result.behavior.metric_name, "day_night_ratio");
        assert!(result.behavior.metric_value.is_finite());
        assert!(!result.behavior.trajectories.is_empty());
    }
}
