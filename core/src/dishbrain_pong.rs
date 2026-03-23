//! DishBrain-style Pong runner with the hot loop owned entirely by Rust.
//!
//! This keeps the existing `MolecularBrain` and biophysics intact and ports the
//! current Pong orchestration path off the Python control loop.

use crate::network::MolecularBrain;
use crate::types::{NTType, NeuronArchetype};
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PongScale {
    Small,
    Medium,
    Large,
    Mega,
}

impl PongScale {
    pub fn from_str(raw: &str) -> Result<Self, String> {
        match raw.to_ascii_lowercase().as_str() {
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            "mega" => Ok(Self::Mega),
            other => Err(format!("Unsupported scale: {}", other)),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
            Self::Mega => "mega",
        }
    }

    pub fn columns(self) -> usize {
        match self {
            Self::Small => 10,
            Self::Medium => 50,
            Self::Large => 250,
            Self::Mega => 1000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PongOutcome {
    Play,
    Hit,
    Miss,
}

struct SimplePong {
    paddle_half_width: f32,
    ball_speed: f32,
    rng: StdRng,
    ball_y: f32,
    ball_dir: f32,
    paddle_y: f32,
    steps_taken: u32,
}

impl SimplePong {
    fn new(seed: u64) -> Self {
        let mut game = Self {
            paddle_half_width: 0.15,
            ball_speed: 0.08,
            rng: StdRng::seed_from_u64(seed),
            ball_y: 0.5,
            ball_dir: 1.0,
            paddle_y: 0.5,
            steps_taken: 0,
        };
        game.reset();
        game
    }

    fn reset(&mut self) {
        self.ball_y = self.rng.gen_range(0.2f32..0.8f32);
        self.ball_dir = if self.rng.gen::<f32>() > 0.5 {
            1.0
        } else {
            -1.0
        };
        self.paddle_y = 0.5;
        self.steps_taken = 0;
    }

    fn step(&mut self, action: u8) -> PongOutcome {
        let paddle_speed = 0.06f32;
        match action {
            0 => self.paddle_y = (self.paddle_y + paddle_speed).min(1.0),
            1 => self.paddle_y = (self.paddle_y - paddle_speed).max(0.0),
            _ => {}
        }

        self.ball_y += self.ball_dir * self.ball_speed;
        self.steps_taken += 1;

        if self.ball_y >= 1.0 {
            self.ball_y = 2.0 - self.ball_y;
            self.ball_dir = -1.0;
        } else if self.ball_y <= 0.0 {
            self.ball_y = -self.ball_y;
            self.ball_dir = 1.0;
        }

        if self.steps_taken >= 12 {
            let dist = (self.ball_y - self.paddle_y).abs();
            if dist <= self.paddle_half_width {
                PongOutcome::Hit
            } else {
                PongOutcome::Miss
            }
        } else {
            PongOutcome::Play
        }
    }
}

struct SensoryEncoder {
    preferred: Vec<f32>,
    two_sigma_sq: f32,
}

impl SensoryEncoder {
    fn new(n_relay: usize, sigma: f32) -> Self {
        let preferred = if n_relay <= 1 {
            vec![0.5]
        } else {
            (0..n_relay)
                .map(|i| i as f32 / (n_relay.saturating_sub(1)) as f32)
                .collect()
        };
        Self {
            preferred,
            two_sigma_sq: 2.0 * sigma * sigma,
        }
    }

    fn encode(&self, position: f32, intensity: f32) -> Vec<f32> {
        self.preferred
            .iter()
            .map(|pref| {
                let diff = *pref - position;
                (-diff * diff / self.two_sigma_sq).exp() * intensity
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
struct FreeEnergyConfig {
    structured_steps: u64,
    unstructured_steps: u64,
    structured_intensity: f32,
    unstructured_intensity: f32,
    ne_boost: f32,
    hebbian_delta: f32,
    unstructured_fraction: f32,
    miss_settle_steps: u64,
    structured_replay_scale: f32,
    miss_hebbian_scale: f32,
}

impl Default for FreeEnergyConfig {
    fn default() -> Self {
        Self {
            structured_steps: 50,
            unstructured_steps: 100,
            structured_intensity: 40.0,
            unstructured_intensity: 40.0,
            ne_boost: 200.0,
            hebbian_delta: 0.8,
            unstructured_fraction: 0.30,
            miss_settle_steps: 0,
            structured_replay_scale: 0.0,
            miss_hebbian_scale: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
struct PongTopology {
    cortex_ids: Vec<usize>,
    relay_ids: Vec<usize>,
    up_ids: Vec<usize>,
    down_ids: Vec<usize>,
    n_columns: usize,
}

impl PongTopology {
    fn from_columns(n_columns: usize, n_per_layer: usize) -> Self {
        let neurons_per_column = n_per_layer * 4;
        let cortex_total = n_columns * neurons_per_column;
        let relay_start = cortex_total;
        let relay_end = relay_start + usize::max(10, n_columns * 3);

        let mut l5_ids = Vec::with_capacity(n_columns * n_per_layer);
        for col in 0..n_columns {
            let col_start = col * neurons_per_column;
            let l5_start = col_start + (n_per_layer * 2);
            let l5_end = l5_start + n_per_layer;
            l5_ids.extend(l5_start..l5_end);
        }
        let mid = l5_ids.len() / 2;

        Self {
            cortex_ids: (0..cortex_total).collect(),
            relay_ids: (relay_start..relay_end).collect(),
            up_ids: l5_ids[..mid].to_vec(),
            down_ids: l5_ids[mid..].to_vec(),
            n_columns,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DishBrainPongResult {
    pub scale: String,
    pub seed: u64,
    pub rallies: usize,
    pub benchmark_mode: bool,
    pub no_interval_biology: bool,
    pub neurons: usize,
    pub synapses: usize,
    pub outcomes: Vec<u8>,
    pub first_10: f32,
    pub last_10: f32,
    pub total_hits: usize,
    pub total_hitrate: f32,
    pub random_baseline: f32,
    pub improvement_vs_random: f32,
    pub passed: bool,
    pub wall_time_s: f64,
    pub bio_time_ms: f64,
    pub realtime_factor: f64,
    pub slowdown_vs_realtime: f64,
    pub neural_steps: u64,
    pub gpu_available: bool,
    pub gpu_dispatch_active: bool,
    pub gpu_init_error: Option<String>,
}

pub struct DishBrainPongSim {
    pub brain: MolecularBrain,
    topology: PongTopology,
    encoder: SensoryEncoder,
    protocol: FreeEnergyConfig,
    scale: PongScale,
    seed: u64,
    stim_steps: u64,
    inter_frame_settle_steps: u64,
    benchmark_mode: bool,
    rng: StdRng,
}

impl DishBrainPongSim {
    pub fn new(scale: PongScale, seed: u64) -> Self {
        let n_columns = scale.columns();
        let topology = PongTopology::from_columns(n_columns, 20);
        let mut rng = StdRng::seed_from_u64(seed);
        let edges = Self::build_edges(&topology, 20, &mut rng);
        let n_total = Self::total_neurons(n_columns, 20);

        let mut brain = MolecularBrain::from_edges(n_total, &edges);
        brain.psc_scale = 30.0;
        Self::assign_archetypes(&mut brain, n_columns, 20);

        let n_relay = usize::max(10, n_columns * 3);
        Self {
            brain,
            topology,
            encoder: SensoryEncoder::new(n_relay, 0.15),
            protocol: FreeEnergyConfig::default(),
            scale,
            seed,
            stim_steps: 30,
            inter_frame_settle_steps: 5,
            benchmark_mode: false,
            rng,
        }
    }

    pub fn enable_latency_benchmark_mode(&mut self) {
        self.benchmark_mode = true;
        self.brain.enable_latency_benchmark_mode();
    }

    pub fn run_replication(&mut self, n_rallies: usize) -> DishBrainPongResult {
        let mut game = SimplePong::new(self.seed);
        self.brain.run_without_sync(300);

        let start_wall = Instant::now();
        let start_step = self.brain.step_count;
        let mut outcomes = Vec::with_capacity(n_rallies);

        for _ in 0..n_rallies {
            let outcome = self.play_rally(&mut game);
            outcomes.push(if matches!(outcome, PongOutcome::Hit) {
                1
            } else {
                0
            });
        }

        let wall_time_s = start_wall.elapsed().as_secs_f64();
        let neural_steps = self.brain.step_count - start_step;
        let bio_time_ms = neural_steps as f64 * self.brain.dt as f64;
        let realtime_factor = if wall_time_s > 0.0 {
            (bio_time_ms / 1000.0) / wall_time_s
        } else {
            0.0
        };
        let slowdown_vs_realtime = if realtime_factor > 0.0 {
            1.0 / realtime_factor
        } else {
            f64::INFINITY
        };

        let first_10_den = usize::min(10, outcomes.len()).max(1);
        let last_10_den = usize::min(10, outcomes.len()).max(1);
        let first_10 =
            outcomes.iter().take(first_10_den).copied().sum::<u8>() as f32 / first_10_den as f32;
        let last_10 = outcomes.iter().rev().take(last_10_den).copied().sum::<u8>() as f32
            / last_10_den as f32;
        let total_hits = outcomes.iter().copied().map(usize::from).sum::<usize>();
        let total_hitrate = total_hits as f32 / usize::max(outcomes.len(), 1) as f32;
        let random_baseline = 0.30f32;
        let passed = last_10 > random_baseline + 0.10 && last_10 > first_10;

        DishBrainPongResult {
            scale: self.scale.as_str().to_string(),
            seed: self.seed,
            rallies: n_rallies,
            benchmark_mode: self.benchmark_mode,
            no_interval_biology: self.benchmark_mode,
            neurons: self.brain.neuron_count(),
            synapses: self.brain.synapse_count(),
            outcomes,
            first_10,
            last_10,
            total_hits,
            total_hitrate,
            random_baseline,
            improvement_vs_random: total_hitrate - random_baseline,
            passed,
            wall_time_s,
            bio_time_ms,
            realtime_factor,
            slowdown_vs_realtime,
            neural_steps,
            gpu_available: self.brain.gpu_available(),
            gpu_dispatch_active: self.brain.gpu_dispatch_active(),
            gpu_init_error: self.brain.gpu_init_error().map(str::to_string),
        }
    }

    fn play_rally(&mut self, game: &mut SimplePong) -> PongOutcome {
        game.reset();
        #[allow(unused_assignments)]
        let mut last_activation = Vec::new();

        loop {
            let activation = self.encoder.encode(game.ball_y, 60.0);
            last_activation = activation.clone();
            let (up_count, down_count) = self.present_stimulus(&activation);
            let action = if up_count > down_count {
                0
            } else if down_count > up_count {
                1
            } else {
                2
            };

            let outcome = game.step(action);
            self.brain.run_without_sync(self.inter_frame_settle_steps);

            if matches!(outcome, PongOutcome::Hit | PongOutcome::Miss) {
                let (correct_ids, wrong_ids) = if game.ball_y > game.paddle_y {
                    (self.topology.up_ids.clone(), self.topology.down_ids.clone())
                } else {
                    (self.topology.down_ids.clone(), self.topology.up_ids.clone())
                };

                match outcome {
                    PongOutcome::Hit => {
                        self.deliver_hit(&last_activation, &correct_ids, &wrong_ids);
                        return PongOutcome::Hit;
                    }
                    PongOutcome::Miss => {
                        self.deliver_miss(&last_activation, &correct_ids, &wrong_ids);
                        return PongOutcome::Miss;
                    }
                    PongOutcome::Play => {}
                }
            }
        }
    }

    fn present_stimulus(&mut self, activation: &[f32]) -> (usize, usize) {
        let up_before = self.brain.spike_count_subset_sum(&self.topology.up_ids);
        let down_before = self.brain.spike_count_subset_sum(&self.topology.down_ids);

        let pulse_pairs = self.stim_steps / 2;
        for _ in 0..pulse_pairs {
            self.brain
                .stimulate_weighted(&self.topology.relay_ids, activation);
            self.brain.run_without_sync(2);
        }
        if self.stim_steps % 2 != 0 {
            self.brain
                .stimulate_weighted(&self.topology.relay_ids, activation);
            self.brain.step();
        }

        let up_after = self.brain.spike_count_subset_sum(&self.topology.up_ids);
        let down_after = self.brain.spike_count_subset_sum(&self.topology.down_ids);
        (
            up_after.saturating_sub(up_before) as usize,
            down_after.saturating_sub(down_before) as usize,
        )
    }

    fn deliver_hit(&mut self, activation: &[f32], correct_ids: &[usize], wrong_ids: &[usize]) {
        self.brain.add_nt_concentration_many(
            &self.topology.cortex_ids,
            NTType::Norepinephrine.index(),
            self.protocol.ne_boost,
        );

        let replay: Option<Vec<f32>> = (self.protocol.structured_replay_scale > 0.0).then(|| {
            activation
                .iter()
                .map(|value| value * self.protocol.structured_replay_scale)
                .collect()
        });
        let pulse_pairs = self.protocol.structured_steps / 2;
        for _ in 0..pulse_pairs {
            self.brain.stimulate_many(
                &self.topology.cortex_ids,
                self.protocol.structured_intensity,
            );
            if let Some(replay) = replay.as_ref() {
                self.brain
                    .stimulate_weighted(&self.topology.relay_ids, replay);
            }
            self.brain.run_without_sync(2);
        }
        if self.protocol.structured_steps % 2 != 0 {
            self.brain.stimulate_many(
                &self.topology.cortex_ids,
                self.protocol.structured_intensity,
            );
            if let Some(replay) = replay.as_ref() {
                self.brain
                    .stimulate_weighted(&self.topology.relay_ids, replay);
            }
            self.brain.step();
        }

        if self.protocol.hebbian_delta > 0.0 {
            let active_relay = self.active_relay_ids(activation);
            let correct_ids_u32: Vec<u32> = correct_ids.iter().map(|&idx| idx as u32).collect();
            let wrong_ids_u32: Vec<u32> = wrong_ids.iter().map(|&idx| idx as u32).collect();
            self.brain.hebbian_nudge(
                &active_relay,
                &correct_ids_u32,
                &wrong_ids_u32,
                self.protocol.hebbian_delta,
            );
        }
    }

    fn deliver_miss(&mut self, activation: &[f32], correct_ids: &[usize], wrong_ids: &[usize]) {
        let mut active_ids = Vec::new();
        let mut random_intensity = Vec::new();

        for _step in 0..self.protocol.unstructured_steps {
            active_ids.clear();
            random_intensity.clear();
            for &idx in &self.topology.cortex_ids {
                if self.rng.gen::<f32>() < self.protocol.unstructured_fraction {
                    active_ids.push(idx);
                    random_intensity
                        .push(self.rng.gen::<f32>() * self.protocol.unstructured_intensity);
                }
            }
            if !active_ids.is_empty() {
                self.brain
                    .stimulate_weighted(&active_ids, &random_intensity);
            }
            self.brain.step();
        }

        if self.protocol.miss_settle_steps > 0 {
            self.brain.run_without_sync(self.protocol.miss_settle_steps);
        }

        if self.protocol.miss_hebbian_scale > 0.0 && self.protocol.hebbian_delta > 0.0 {
            let active_relay = self.active_relay_ids(activation);
            let correct_ids_u32: Vec<u32> = correct_ids.iter().map(|&idx| idx as u32).collect();
            let wrong_ids_u32: Vec<u32> = wrong_ids.iter().map(|&idx| idx as u32).collect();
            self.brain.hebbian_nudge(
                &active_relay,
                &correct_ids_u32,
                &wrong_ids_u32,
                self.protocol.hebbian_delta * self.protocol.miss_hebbian_scale,
            );
        }
    }

    fn active_relay_ids(&self, activation: &[f32]) -> Vec<u32> {
        let threshold = activation.iter().copied().fold(0.0f32, f32::max) * 0.2;
        self.topology
            .relay_ids
            .iter()
            .zip(activation.iter())
            .filter_map(|(&idx, &value)| (value > threshold).then_some(idx as u32))
            .collect()
    }

    fn total_neurons(n_columns: usize, n_per_layer: usize) -> usize {
        let neurons_per_column = n_per_layer * 4;
        let cortex_total = n_columns * neurons_per_column;
        let thalamus_relay = usize::max(10, n_columns * 3);
        let thalamus_reticular = usize::max(5, n_columns * 2);
        let hippo_dg = usize::max(10, n_columns * 5);
        let hippo_ca3 = usize::max(8, n_columns * 4);
        let hippo_ca1 = usize::max(6, n_columns * 3);
        let bg_d1 = usize::max(5, n_columns * 2);
        let bg_d2 = usize::max(5, n_columns * 2);
        cortex_total
            + thalamus_relay
            + thalamus_reticular
            + hippo_dg
            + hippo_ca3
            + hippo_ca1
            + bg_d1
            + bg_d2
    }

    fn build_edges(
        topology: &PongTopology,
        n_per_layer: usize,
        rng: &mut StdRng,
    ) -> Vec<(u32, u32, NTType)> {
        let n_columns = topology.n_columns;
        let neurons_per_column = n_per_layer * 4;
        let cortex_total = n_columns * neurons_per_column;
        let relay_start = cortex_total;
        let relay_end = relay_start + usize::max(10, n_columns * 3);
        let reticular_start = relay_end;
        let reticular_end = reticular_start + usize::max(5, n_columns * 2);
        let dg_start = reticular_end;
        let dg_end = dg_start + usize::max(10, n_columns * 5);
        let ca3_start = dg_end;
        let ca3_end = ca3_start + usize::max(8, n_columns * 4);
        let ca1_start = ca3_end;
        let ca1_end = ca1_start + usize::max(6, n_columns * 3);
        let d1_start = ca1_end;
        let d1_end = d1_start + usize::max(5, n_columns * 2);
        let d2_start = d1_end;
        let d2_end = d2_start + usize::max(5, n_columns * 2);

        let relay: Vec<usize> = (relay_start..relay_end).collect();
        let reticular: Vec<usize> = (reticular_start..reticular_end).collect();
        let dg: Vec<usize> = (dg_start..dg_end).collect();
        let ca3: Vec<usize> = (ca3_start..ca3_end).collect();
        let ca1: Vec<usize> = (ca1_start..ca1_end).collect();
        let d1: Vec<usize> = (d1_start..d1_end).collect();
        let d2: Vec<usize> = (d2_start..d2_end).collect();

        let mut edges: Vec<(u32, u32, NTType)> = Vec::new();

        let mut random_connections =
            |src: &[usize],
             dst: &[usize],
             prob: f32,
             nt: NTType,
             edges: &mut Vec<(u32, u32, NTType)>| {
                if src.is_empty() || dst.is_empty() {
                    return;
                }
                let n_possible = src.len() * dst.len();
                if n_possible > 100_000 {
                    let n_expected = (n_possible as f32 * prob) as usize;
                    for _ in 0..n_expected {
                        let pre = src[rng.gen_range(0..src.len())];
                        let post = dst[rng.gen_range(0..dst.len())];
                        if pre != post {
                            edges.push((pre as u32, post as u32, nt));
                        }
                    }
                } else {
                    for &pre in src {
                        for &post in dst {
                            if pre != post && rng.gen::<f32>() < prob {
                                edges.push((pre as u32, post as u32, nt));
                            }
                        }
                    }
                }
            };

        for col_idx in 0..n_columns {
            let col_start = col_idx * neurons_per_column;
            let l4: Vec<usize> = (col_start..col_start + n_per_layer).collect();
            let l23: Vec<usize> =
                (col_start + n_per_layer..col_start + (n_per_layer * 2)).collect();
            let l5: Vec<usize> =
                (col_start + (n_per_layer * 2)..col_start + (n_per_layer * 3)).collect();
            let l6: Vec<usize> =
                (col_start + (n_per_layer * 3)..col_start + (n_per_layer * 4)).collect();

            random_connections(&l4, &l23, 0.3, NTType::Glutamate, &mut edges);
            random_connections(&l23, &l5, 0.25, NTType::Glutamate, &mut edges);
            random_connections(&l5, &l6, 0.2, NTType::Glutamate, &mut edges);
            random_connections(&l6, &l4, 0.15, NTType::Glutamate, &mut edges);

            for layer in [&l4, &l23, &l5, &l6] {
                let n_layer = layer.len();
                let n_inhib = usize::max(1, ((n_layer as f32) * 0.2) as usize);
                let excit = &layer[..n_layer - n_inhib];
                let inhib = &layer[n_layer - n_inhib..];
                random_connections(excit, excit, 0.1, NTType::Glutamate, &mut edges);
                random_connections(inhib, excit, 0.5, NTType::GABA, &mut edges);
                random_connections(excit, inhib, 0.3, NTType::Glutamate, &mut edges);
            }
        }

        if n_columns > 1 {
            for i in 0..n_columns {
                let src_l23_start = i * neurons_per_column + n_per_layer;
                let src_l23_end = src_l23_start + n_per_layer;
                let src_l23: Vec<usize> = (src_l23_start..src_l23_end).collect();
                for j in (i + 1)..usize::min(i + 3, n_columns) {
                    let dst_l23_start = j * neurons_per_column + n_per_layer;
                    let dst_l23_end = dst_l23_start + n_per_layer;
                    let dst_l23: Vec<usize> = (dst_l23_start..dst_l23_end).collect();
                    random_connections(&src_l23, &dst_l23, 0.05, NTType::Glutamate, &mut edges);
                    random_connections(&dst_l23, &src_l23, 0.05, NTType::Glutamate, &mut edges);
                }
            }
        }

        random_connections(&relay, &reticular, 0.4, NTType::Glutamate, &mut edges);
        random_connections(&reticular, &relay, 0.5, NTType::GABA, &mut edges);
        random_connections(&dg, &ca3, 0.3, NTType::Glutamate, &mut edges);
        random_connections(&ca3, &ca3, 0.5, NTType::Glutamate, &mut edges);
        random_connections(&ca3, &ca1, 0.4, NTType::Glutamate, &mut edges);
        random_connections(&d1, &d2, 0.3, NTType::GABA, &mut edges);
        random_connections(&d2, &d1, 0.3, NTType::GABA, &mut edges);

        for col_idx in 0..n_columns {
            let col_start = col_idx * neurons_per_column;
            let l4: Vec<usize> = (col_start..col_start + n_per_layer).collect();
            let l23: Vec<usize> =
                (col_start + n_per_layer..col_start + (n_per_layer * 2)).collect();
            let l5: Vec<usize> =
                (col_start + (n_per_layer * 2)..col_start + (n_per_layer * 3)).collect();
            let l6: Vec<usize> =
                (col_start + (n_per_layer * 3)..col_start + (n_per_layer * 4)).collect();

            random_connections(&relay, &l4, 0.3, NTType::Glutamate, &mut edges);
            random_connections(&relay, &l5, 0.5, NTType::Glutamate, &mut edges);
            random_connections(&l6, &relay, 0.2, NTType::Glutamate, &mut edges);
            random_connections(&l5, &d1, 0.15, NTType::Glutamate, &mut edges);
            random_connections(&l23, &dg, 0.1, NTType::Glutamate, &mut edges);
            random_connections(&ca1, &l5, 0.1, NTType::Glutamate, &mut edges);
        }

        edges.sort_by_key(|edge| edge.0);
        edges
    }

    fn assign_archetypes(brain: &mut MolecularBrain, n_columns: usize, n_per_layer: usize) {
        let neurons_per_column = n_per_layer * 4;
        for col_idx in 0..n_columns {
            let col_start = col_idx * neurons_per_column;
            for layer_idx in 0..4 {
                let layer_start = col_start + layer_idx * n_per_layer;
                let layer_end = layer_start + n_per_layer;
                let n_inhib = usize::max(1, ((n_per_layer as f32) * 0.2) as usize);
                for idx in layer_end - n_inhib..layer_end {
                    brain.neurons.archetype[idx] = NeuronArchetype::Interneuron as u8;
                }
            }
        }

        let relay_end = n_columns * neurons_per_column + usize::max(10, n_columns * 3);
        let reticular_end = relay_end + usize::max(5, n_columns * 2);
        for idx in relay_end..reticular_end {
            brain.neurons.archetype[idx] = NeuronArchetype::Interneuron as u8;
        }

        let dg_start = reticular_end;
        let dg_end = dg_start + usize::max(10, n_columns * 5);
        for idx in dg_start..dg_end {
            brain.neurons.archetype[idx] = NeuronArchetype::Granule as u8;
        }

        let d1_start = dg_end + usize::max(8, n_columns * 4) + usize::max(6, n_columns * 3);
        let d2_end = d1_start + usize::max(5, n_columns * 2) * 2;
        for idx in d1_start..d2_end {
            brain.neurons.archetype[idx] = NeuronArchetype::MediumSpiny as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_scale_count_matches_python_path() {
        let topology = PongTopology::from_columns(250, 20);
        assert_eq!(DishBrainPongSim::total_neurons(250, 20), 25_250);
        assert_eq!(topology.relay_ids.len(), 750);
        assert_eq!(topology.up_ids.len() + topology.down_ids.len(), 5_000);
    }

    #[test]
    fn test_small_scale_smoke() {
        let mut sim = DishBrainPongSim::new(PongScale::Small, 42);
        let result = sim.run_replication(2);
        assert_eq!(result.rallies, 2);
        assert_eq!(result.outcomes.len(), 2);
    }

    #[test]
    fn test_small_scale_benchmark_mode_smoke() {
        let mut sim = DishBrainPongSim::new(PongScale::Small, 42);
        sim.enable_latency_benchmark_mode();
        assert!(sim.benchmark_mode);
        assert!(!sim.brain.enable_gene_expression);
        assert!(!sim.brain.enable_metabolism);
        assert!(!sim.brain.enable_microtubules);
        let result = sim.run_replication(2);
        assert!(result.benchmark_mode);
        assert_eq!(result.rallies, 2);
        assert_eq!(result.outcomes.len(), 2);
    }
}
