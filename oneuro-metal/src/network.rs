//! MolecularBrain -- the main simulation orchestrator.
//!
//! Owns all neuron state (`NeuronArrays`), synaptic connectivity (`SynapseArrays`),
//! and subsystem engines (circadian, glia, pharmacology). The [`MolecularBrain::step`]
//! method runs a single dt=0.1ms simulation tick through the full biophysical pipeline:
//!
//! 1. Pre-step clear (fired flags, external currents)
//! 2. Circadian excitability modulation
//! 3. Pharmacology (drug-modified conductance scales)
//! 4. HH gating + receptor binding + membrane integration + calcium + second messengers
//!    (GPU or CPU fallback)
//! 5. Spike propagation (vesicle release + PSC injection)
//! 6. STDP (receptor trafficking for active synapses)
//! 7. Synaptic cleft dynamics (NT clearance)
//! 8. Interval-gated subsystems (gene expression, metabolism, microtubules, glia)

use crate::circadian::CircadianClock;
use crate::constants::*;
use crate::gene_expression;
use crate::glia::GliaState;
use crate::gpu;
use crate::metabolism;
use crate::microtubules;
use crate::neuron_arrays::NeuronArrays;
use crate::pharmacology::PharmacologyEngine;
use crate::synapse_arrays::SynapseArrays;
use crate::types::*;

/// The central simulation struct. Owns all neural state and drives the step loop.
///
/// # GPU Acceleration
///
/// On macOS with Apple Silicon, the hot-path phases (HH gating, receptor binding,
/// membrane integration, calcium dynamics, second messengers) are dispatched to
/// Metal compute shaders via `GpuContext`. On other platforms (or for small networks
/// with < 64 neurons), equivalent CPU fallback implementations are used automatically.
///
/// # External Current Convention
///
/// External currents (`stimulate()`) are consumed and zeroed each step. To sustain
/// stimulation across multiple steps, call `stimulate()` every step or use a callback.
pub struct MolecularBrain {
    /// Structure-of-Arrays neuron state for GPU-friendly memory layout.
    pub neurons: NeuronArrays,
    /// CSR sparse synapse storage with per-synapse molecular state.
    pub synapses: SynapseArrays,

    // GPU context (None if Metal unavailable or not macOS)
    #[cfg(target_os = "macos")]
    gpu: Option<gpu::GpuContext>,
    #[cfg(target_os = "macos")]
    gpu_init_error: Option<String>,
    #[cfg(target_os = "macos")]
    gpu_neurons: Option<gpu::state::MetalNeuronState>,
    #[cfg(target_os = "macos")]
    gpu_synapses: Option<gpu::state::MetalSynapseState>,
    #[cfg(target_os = "macos")]
    gpu_glia: Option<gpu::state::MetalGliaState>,
    #[cfg(target_os = "macos")]
    neuron_shadow_stale: bool,
    #[cfg(target_os = "macos")]
    synapse_shadow_stale: bool,
    #[cfg(target_os = "macos")]
    glia_shadow_stale: bool,
    #[cfg(target_os = "macos")]
    neuron_shadow_dirty: bool,
    #[cfg(target_os = "macos")]
    synapse_shadow_dirty: bool,
    #[cfg(target_os = "macos")]
    glia_shadow_dirty: bool,

    // ===== Subsystems =====
    /// Circadian clock (TTFL ODE -> excitability bias, NT synthesis, alertness).
    pub circadian: CircadianClock,
    /// Glial cells (astrocytes, oligodendrocytes, microglia).
    pub glia: GliaState,
    /// Drug pharmacokinetics/pharmacodynamics engine.
    pub pharmacology: PharmacologyEngine,

    // ===== Configuration =====
    /// PSC amplitude scaling (default 30.0). Higher values produce stronger
    /// postsynaptic currents, enabling cascade propagation.
    pub psc_scale: f32,
    /// Integration timestep in ms (default 0.1).
    pub dt: f32,
    /// Current simulation time in ms.
    pub time: f32,
    /// Total simulation steps executed.
    pub step_count: u64,
    /// Base maximal conductances for the eight channel families (mS/cm²).
    pub channel_g_max: [f32; IonChannelType::COUNT],
    /// Leak reversal potential used for the passive potassium-like channel (mV).
    pub kleak_reversal_mv: f32,
    /// Effective membrane capacitance used by voltage integration (µF/cm²).
    pub membrane_capacitance_uf: f32,
    /// Spike threshold for threshold-crossing detection (mV).
    pub spike_threshold_mv: f32,
    /// Absolute refractory period enforced after a spike (ms).
    pub refractory_period_ms: f32,

    // ===== Feature flags =====
    /// Enable astrocyte/oligodendrocyte/microglia subsystem.
    pub enable_glia: bool,
    /// Enable TTFL circadian clock.
    pub enable_circadian: bool,
    /// Enable pharmacology engine (drug PK/PD).
    pub enable_pharmacology: bool,
    /// Enable interval-gated gene expression updates.
    pub enable_gene_expression: bool,
    /// Enable interval-gated metabolism updates.
    pub enable_metabolism: bool,
    /// Enable interval-gated microtubule / Orch-OR updates.
    pub enable_microtubules: bool,
    /// Enable Metal GPU acceleration (macOS only; falls back to CPU if false).
    pub enable_gpu: bool,

    // ===== G-protein input buffers =====
    // Computed from NT concentrations and metabotropic receptor Hill functions
    // before the second messenger GPU shader dispatch.
    gs_input: Vec<f32>,
    gi_input: Vec<f32>,
    gq_input: Vec<f32>,
}

impl MolecularBrain {
    /// Create a new brain with `n_neurons` neurons at resting state and no synapses.
    ///
    /// Attempts to initialize Metal GPU on macOS. If Metal is unavailable, the brain
    /// will use CPU fallback for all compute phases.
    pub fn new(n_neurons: usize) -> Self {
        #[cfg(target_os = "macos")]
        let (gpu_ctx, gpu_init_error) = match gpu::GpuContext::new() {
            Ok(ctx) => (Some(ctx), None),
            Err(err) => (None, Some(err)),
        };

        let neurons = NeuronArrays::new(n_neurons);
        let synapses = SynapseArrays::new(n_neurons);

        Self {
            neurons,
            synapses,
            #[cfg(target_os = "macos")]
            gpu: gpu_ctx,
            #[cfg(target_os = "macos")]
            gpu_init_error,
            #[cfg(target_os = "macos")]
            gpu_neurons: None,
            #[cfg(target_os = "macos")]
            gpu_synapses: None,
            #[cfg(target_os = "macos")]
            gpu_glia: None,
            #[cfg(target_os = "macos")]
            neuron_shadow_stale: false,
            #[cfg(target_os = "macos")]
            synapse_shadow_stale: false,
            #[cfg(target_os = "macos")]
            glia_shadow_stale: false,
            #[cfg(target_os = "macos")]
            neuron_shadow_dirty: false,
            #[cfg(target_os = "macos")]
            synapse_shadow_dirty: false,
            #[cfg(target_os = "macos")]
            glia_shadow_dirty: false,
            circadian: CircadianClock::new(3600.0), // 1h bio = 1s sim
            glia: GliaState::new(n_neurons),
            pharmacology: PharmacologyEngine::new(),
            psc_scale: DEFAULT_PSC_SCALE,
            dt: DEFAULT_DT,
            time: 0.0,
            step_count: 0,
            channel_g_max: [
                NAV_G_MAX,
                KV_G_MAX,
                KLEAK_G_MAX,
                CAV_G_MAX,
                NMDA_G_MAX,
                AMPA_G_MAX,
                GABAA_G_MAX,
                NACHR_G_MAX,
            ],
            kleak_reversal_mv: KLEAK_E_REV,
            membrane_capacitance_uf: DEFAULT_C_M,
            spike_threshold_mv: AP_THRESHOLD,
            refractory_period_ms: REFRACTORY_PERIOD,
            enable_glia: true,
            enable_circadian: true,
            enable_pharmacology: true,
            enable_gene_expression: true,
            enable_metabolism: true,
            enable_microtubules: true,
            enable_gpu: true,
            gs_input: vec![0.0; n_neurons],
            gi_input: vec![0.0; n_neurons],
            gq_input: vec![0.0; n_neurons],
        }
    }

    /// Main simulation step: advance the entire brain by one `dt` tick.
    ///
    /// This is the hot loop entry point. The step pipeline is:
    ///
    /// 0. Pre-step clear (fired flags, external currents zeroed)
    /// 1. Circadian excitability bias (every step, trivial cost)
    /// 2. Pharmacology (drug PK/PD modifies conductance_scale)
    /// 3. GPU or CPU compute path:
    ///    - HH gating variable integration (Na_v, K_v, Ca_v)
    ///    - Receptor binding (Hill equation -> open fractions)
    ///    - Membrane integration (dV/dt + spike detection)
    ///    - Calcium 4-compartment ODE
    ///    - Second messenger cascades (cAMP/PKA/PKC/CaMKII/CREB/ERK)
    /// 4. Spike propagation (only for fired neurons, via CSR outgoing synapses)
    /// 5. STDP receptor trafficking (only for synapses touching fired neurons)
    /// 6. Synaptic cleft dynamics (NT clearance for all active synapses)
    /// 7. Interval-gated subsystems:
    ///    - Gene expression (every 10 steps)
    ///    - Metabolism (every 5 steps)
    ///    - Microtubules/Orch-OR (every 10 steps)
    ///    - Glia (every 10 steps)
    pub fn step(&mut self) {
        let dt = self.dt;
        let use_gpu = self.should_use_gpu();
        let mut circadian_bias = 0.0f32;

        if self.enable_circadian {
            self.circadian.step(dt);
            circadian_bias = self.circadian.excitability_bias();
        }

        if use_gpu {
            #[cfg(target_os = "macos")]
            {
                self.upload_shadow_to_gpu_if_dirty();
                if self.enable_pharmacology && self.pharmacology.n_active_drugs() > 0 {
                    if let (Some(gpu), Some(gpu_neurons)) =
                        (self.gpu.as_ref(), self.gpu_neurons.as_ref())
                    {
                        self.pharmacology.step_gpu(gpu, gpu_neurons, dt);
                        self.neuron_shadow_stale = true;
                    }
                }
                self.step_gpu(dt, circadian_bias);
            }
        } else {
            self.step_cpu_with_bias(dt, circadian_bias);

            let fired = self.neurons.fired_indices();
            if !fired.is_empty() {
                crate::spike_propagation::propagate_spikes(
                    &mut self.neurons,
                    &mut self.synapses,
                    &fired,
                    self.time,
                    self.psc_scale,
                );

                crate::stdp::update_stdp(&self.neurons, &mut self.synapses, &fired, self.time, dt);

                let step = self.step_count as u32;
                for &idx in &fired {
                    self.neurons.last_fired_step[idx] = step;
                }
            }

            for i in 0..self.synapses.n_synapses {
                self.synapses.update_cleft(i, self.time, dt);
            }
        }

        // 7. Interval-gated subsystems
        let step = self.step_count as u32;
        let gene_due =
            self.enable_gene_expression && step > 0 && step % GENE_EXPRESSION_INTERVAL == 0;
        let metabolism_due = self.enable_metabolism && step > 0 && step % METABOLISM_INTERVAL == 0;
        let cytoskeleton_due =
            self.enable_microtubules && step > 0 && step % CYTOSKELETON_INTERVAL == 0;
        let glia_due = self.enable_glia && step > 0 && step % GLIA_INTERVAL == 0;

        if gene_due {
            let eff_dt = dt * GENE_EXPRESSION_INTERVAL as f32;
            #[cfg(target_os = "macos")]
            if use_gpu {
                if let (Some(gpu), Some(gpu_neurons)) =
                    (self.gpu.as_ref(), self.gpu_neurons.as_ref())
                {
                    gpu::gene_expression::dispatch_gene_expression(gpu, gpu_neurons, eff_dt);
                    self.neuron_shadow_stale = true;
                }
            } else {
                gene_expression::update_gene_expression(&mut self.neurons, eff_dt);
            }
        }

        if metabolism_due {
            let eff_dt = dt * METABOLISM_INTERVAL as f32;
            #[cfg(target_os = "macos")]
            if use_gpu {
                if let (Some(gpu), Some(gpu_neurons)) =
                    (self.gpu.as_ref(), self.gpu_neurons.as_ref())
                {
                    gpu::metabolism::dispatch_metabolism(gpu, gpu_neurons, eff_dt);
                    self.neuron_shadow_stale = true;
                }
            } else {
                metabolism::update_metabolism(&mut self.neurons, eff_dt);
            }
        }

        if cytoskeleton_due {
            let eff_dt = dt * CYTOSKELETON_INTERVAL as f32;
            #[cfg(target_os = "macos")]
            if use_gpu {
                if let (Some(gpu), Some(gpu_neurons)) =
                    (self.gpu.as_ref(), self.gpu_neurons.as_ref())
                {
                    gpu::microtubules::dispatch_microtubules(gpu, gpu_neurons, eff_dt);
                    self.neuron_shadow_stale = true;
                }
            } else {
                microtubules::update_microtubules(&mut self.neurons, eff_dt);
            }
        }

        if glia_due {
            #[cfg(target_os = "macos")]
            if use_gpu {
                let eff_dt = dt * GLIA_INTERVAL as f32;
                if let (Some(gpu), Some(gpu_glia), Some(gpu_neurons), Some(gpu_synapses)) = (
                    self.gpu.as_ref(),
                    self.gpu_glia.as_ref(),
                    self.gpu_neurons.as_ref(),
                    self.gpu_synapses.as_ref(),
                ) {
                    gpu::glia::dispatch_glia_neuron_step(gpu, gpu_glia, gpu_neurons, eff_dt);
                    gpu::glia::dispatch_glia_prune_synapses(gpu, gpu_glia, gpu_synapses);
                    self.neuron_shadow_stale = true;
                    self.synapse_shadow_stale = true;
                    self.glia_shadow_stale = true;
                }
            } else {
                self.glia.update(
                    &mut self.neurons,
                    &mut self.synapses,
                    dt * GLIA_INTERVAL as f32,
                );
            }
        }

        #[cfg(target_os = "macos")]
        if use_gpu {
            self.upload_shadow_to_gpu_if_dirty();
            if let Some(gpu_neurons) = self.gpu_neurons.as_ref() {
                gpu_neurons.clear_external_current();
            }
        }
        self.neurons.clear_external_current();

        // 8. Advance simulation clock
        self.time += dt;
        self.step_count += 1;
    }

    /// GPU hot path: HH gating, receptor binding, membrane integration,
    /// calcium dynamics, and second messenger cascades dispatched to Metal.
    #[cfg(target_os = "macos")]
    fn step_gpu(&mut self, dt: f32, global_bias: f32) {
        if !self.ensure_gpu_resident_state() {
            self.step_cpu_with_bias(dt, global_bias);
            return;
        }

        if let (Some(gpu), Some(gpu_neurons), Some(gpu_synapses)) = (
            self.gpu.as_ref(),
            self.gpu_neurons.as_ref(),
            self.gpu_synapses.as_ref(),
        ) {
            let cmd = gpu.new_command_buffer();
            gpu::hh_gating::encode_hh_gating(gpu, &cmd, gpu_neurons, dt);
            gpu::receptor_binding::encode_receptor_binding(gpu, &cmd, gpu_neurons);
            gpu::membrane_integration::encode_membrane_integration(
                gpu,
                &cmd,
                gpu_neurons,
                dt,
                global_bias,
                &self.channel_g_max,
                self.kleak_reversal_mv,
                self.membrane_capacitance_uf,
                self.spike_threshold_mv,
                self.refractory_period_ms,
            );
            gpu::calcium_dynamics::encode_calcium_dynamics(gpu, &cmd, gpu_neurons, dt);
            gpu::second_messenger::encode_second_messenger(gpu, &cmd, gpu_neurons, dt);
            gpu::synapse_step::encode_synapse_step(
                gpu,
                &cmd,
                gpu_neurons,
                gpu_synapses,
                self.time,
                dt,
                self.psc_scale,
                self.step_count as u32,
            );
            gpu.commit_and_wait(cmd);
            self.neuron_shadow_stale = true;
            self.synapse_shadow_stale = true;
        }
    }

    /// On non-macOS platforms, GPU dispatch falls through to CPU.
    #[cfg(not(target_os = "macos"))]
    fn step_gpu(&mut self, dt: f32, global_bias: f32) {
        self.step_cpu_with_bias(dt, global_bias);
    }

    /// CPU fallback path: identical biophysics, scalar per-neuron loop.
    fn step_cpu_with_bias(&mut self, dt: f32, global_bias: f32) {
        self.neurons.pre_step_clear();
        if self.enable_pharmacology {
            self.pharmacology.step(&mut self.neurons, dt);
        }
        gpu::hh_gating::cpu_hh_gating(&mut self.neurons, dt);
        gpu::receptor_binding::cpu_receptor_binding(&mut self.neurons);
        gpu::membrane_integration::cpu_membrane_integration(
            &mut self.neurons,
            dt,
            global_bias,
            &self.channel_g_max,
            self.kleak_reversal_mv,
            self.membrane_capacitance_uf,
            self.spike_threshold_mv,
            self.refractory_period_ms,
        );
        gpu::calcium_dynamics::cpu_calcium_dynamics(&mut self.neurons, dt);
        // G-protein inputs computed internally from NT concentrations.
        gpu::second_messenger::cpu_second_messenger(&mut self.neurons, dt);
    }

    /// Decide whether to dispatch to GPU or use CPU fallback.
    ///
    /// GPU is used when: (1) macOS, (2) Metal device available, (3) GPU enabled,
    /// (4) at least 64 neurons (below this the dispatch overhead dominates).
    fn should_use_gpu(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.enable_gpu && self.gpu.is_some() && self.neurons.count >= 64
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// Compute G-protein cascade inputs from NT concentrations and metabotropic receptors.
    ///
    /// Metabotropic receptors couple to heterotrimeric G-proteins:
    /// - Gs (stimulatory): D1 receptor (dopamine)
    /// - Gi (inhibitory): D2, 5-HT1A, Alpha2
    /// - Gq (phospholipase C): 5-HT2A, mAChR-M1, Alpha1
    ///
    /// Each is computed as a Hill-equation activation [0, 1] capped at saturation.
    /// Compute G-protein inputs on CPU (used when second messenger GPU dispatch
    /// is unavailable, or for debugging). Currently the GPU dispatch computes
    /// these internally, but this method is retained for CPU-only mode.
    #[allow(dead_code)]
    fn compute_gprotein_inputs(&mut self) {
        for i in 0..self.neurons.count {
            let nt = &self.neurons.nt_conc[i];
            let da = nt[NTType::Dopamine.index()];
            let ht = nt[NTType::Serotonin.index()];
            let ne = nt[NTType::Norepinephrine.index()];
            let ach = nt[NTType::Acetylcholine.index()];

            // Gs: D1 receptor (Gs-coupled, increases cAMP via adenylyl cyclase)
            self.gs_input[i] = hill(da, ReceptorType::D1.ec50_nm(), ReceptorType::D1.hill_n());

            // Gi: D2 + 5-HT1A + Alpha2 (Gi-coupled, inhibits adenylyl cyclase)
            self.gi_input[i] = (hill(da, ReceptorType::D2.ec50_nm(), ReceptorType::D2.hill_n())
                + hill(
                    ht,
                    ReceptorType::HT1A.ec50_nm(),
                    ReceptorType::HT1A.hill_n(),
                )
                + hill(
                    ne,
                    ReceptorType::Alpha2.ec50_nm(),
                    ReceptorType::Alpha2.hill_n(),
                ))
            .min(1.0);

            // Gq: 5-HT2A + mAChR-M1 + Alpha1 (Gq-coupled, activates PLC -> IP3 + DAG)
            self.gq_input[i] = (hill(
                ht,
                ReceptorType::HT2A.ec50_nm(),
                ReceptorType::HT2A.hill_n(),
            ) + hill(
                ach,
                ReceptorType::MAChRM1.ec50_nm(),
                ReceptorType::MAChRM1.hill_n(),
            ) + hill(
                ne,
                ReceptorType::Alpha1.ec50_nm(),
                ReceptorType::Alpha1.hill_n(),
            ))
            .min(1.0);
        }
    }

    /// Run multiple simulation steps.
    pub fn run(&mut self, steps: u64) {
        self.run_without_sync(steps);
        self.sync_shadow_from_gpu();
    }

    /// Run multiple simulation steps without forcing a host shadow sync.
    pub fn run_without_sync(&mut self, steps: u64) {
        #[cfg(target_os = "macos")]
        if self.run_gpu_steps_batched_if_possible(steps) {
            return;
        }

        for _ in 0..steps {
            self.step();
        }
    }

    #[cfg(target_os = "macos")]
    fn can_batch_gpu_steps_without_host(&self) -> bool {
        self.should_use_gpu()
            && !self.enable_circadian
            && !self.enable_pharmacology
            && !self.enable_gene_expression
            && !self.enable_metabolism
            && !self.enable_microtubules
            && !self.enable_glia
    }

    #[cfg(target_os = "macos")]
    fn run_gpu_steps_batched_if_possible(&mut self, steps: u64) -> bool {
        if steps == 0 || !self.can_batch_gpu_steps_without_host() {
            return false;
        }
        if !self.ensure_gpu_resident_state() {
            return false;
        }

        self.upload_shadow_to_gpu_if_dirty();

        let (Some(gpu), Some(gpu_neurons), Some(gpu_synapses)) = (
            self.gpu.as_ref(),
            self.gpu_neurons.as_ref(),
            self.gpu_synapses.as_ref(),
        ) else {
            return false;
        };

        let mut remaining = steps;
        const STEP_BATCH: u64 = 16;
        while remaining > 0 {
            let batch = remaining.min(STEP_BATCH);
            let cmd = gpu.new_command_buffer();
            for _ in 0..batch {
                gpu::hh_gating::encode_hh_gating(gpu, &cmd, gpu_neurons, self.dt);
                gpu::receptor_binding::encode_receptor_binding(gpu, &cmd, gpu_neurons);
                gpu::membrane_integration::encode_membrane_integration(
                    gpu,
                    &cmd,
                    gpu_neurons,
                    self.dt,
                    0.0,
                    &self.channel_g_max,
                    self.kleak_reversal_mv,
                    self.membrane_capacitance_uf,
                    self.spike_threshold_mv,
                    self.refractory_period_ms,
                );
                gpu::calcium_dynamics::encode_calcium_dynamics(gpu, &cmd, gpu_neurons, self.dt);
                gpu::second_messenger::encode_second_messenger(gpu, &cmd, gpu_neurons, self.dt);
                gpu::synapse_step::encode_synapse_step(
                    gpu,
                    &cmd,
                    gpu_neurons,
                    gpu_synapses,
                    self.time,
                    self.dt,
                    self.psc_scale,
                    self.step_count as u32,
                );
                self.time += self.dt;
                self.step_count += 1;
            }
            gpu.commit_and_wait(cmd);
            remaining -= batch;
        }

        self.neuron_shadow_stale = true;
        self.synapse_shadow_stale = true;
        self.neurons.clear_external_current();
        true
    }

    /// Build a brain from a list of directed edges `(pre, post, nt_type)`.
    ///
    /// This is the primary constructor for networks with known connectivity.
    /// Edges are converted to CSR format internally. Edges do not need to be sorted.
    ///
    /// # Example
    /// ```ignore
    /// let edges = vec![
    ///     (0, 1, NTType::Glutamate),
    ///     (0, 2, NTType::Glutamate),
    ///     (1, 0, NTType::GABA),
    /// ];
    /// let mut brain = MolecularBrain::from_edges(3, &edges);
    /// brain.run(1000);
    /// ```
    pub fn from_edges(n_neurons: usize, edges: &[(u32, u32, NTType)]) -> Self {
        // Sort edges by pre neuron for CSR construction
        let mut sorted_edges = edges.to_vec();
        sorted_edges.sort_by_key(|e| e.0);

        let mut brain = Self::new(n_neurons);
        brain.synapses = SynapseArrays::from_edges(n_neurons, &sorted_edges);

        // Resize G-protein buffers to match neuron count
        brain.gs_input.resize(n_neurons, 0.0);
        brain.gi_input.resize(n_neurons, 0.0);
        brain.gq_input.resize(n_neurons, 0.0);

        brain
    }

    /// Inject external current into a neuron for the current step.
    ///
    /// Current is in uA/cm^2 and is consumed (zeroed) after each step.
    /// For sustained stimulation, call this every step.
    ///
    /// Typical values: 5-20 uA/cm^2 for subthreshold, 40-80 uA/cm^2 for
    /// suprathreshold (will fire within a few dt steps).
    pub fn stimulate(&mut self, neuron_idx: usize, current_ua: f32) {
        if neuron_idx < self.neurons.count {
            self.neurons.external_current[neuron_idx] += current_ua;
            #[cfg(target_os = "macos")]
            if let Some(gpu_neurons) = self.gpu_neurons.as_ref() {
                gpu_neurons.add_external_current(neuron_idx, current_ua);
            }
        }
    }

    /// Inject the same external current into many neurons for the current step.
    pub fn stimulate_many(&mut self, neuron_indices: &[usize], current_ua: f32) {
        for &idx in neuron_indices {
            if idx < self.neurons.count {
                self.neurons.external_current[idx] += current_ua;
            }
        }
        #[cfg(target_os = "macos")]
        if let Some(gpu_neurons) = self.gpu_neurons.as_ref() {
            gpu_neurons.add_external_current_many(neuron_indices, current_ua);
        }
    }

    /// Inject per-neuron external currents for the current step.
    pub fn stimulate_weighted(&mut self, neuron_indices: &[usize], currents_ua: &[f32]) {
        for (&idx, &current) in neuron_indices.iter().zip(currents_ua.iter()) {
            if idx < self.neurons.count {
                self.neurons.external_current[idx] += current;
            }
        }
        #[cfg(target_os = "macos")]
        if let Some(gpu_neurons) = self.gpu_neurons.as_ref() {
            gpu_neurons.add_external_current_weighted(neuron_indices, currents_ua);
        }
    }

    /// Get instantaneous firing rate (Hz) for a neuron.
    ///
    /// Computed as total spike count / elapsed simulation time.
    /// Returns 0.0 if no simulation time has elapsed.
    pub fn firing_rate(&self, neuron_idx: usize) -> f32 {
        if neuron_idx >= self.neurons.count {
            return 0.0;
        }
        let spikes = self.neurons.spike_count[neuron_idx] as f32;
        let time_s = self.time / 1000.0;
        if time_s > 0.0 {
            spikes / time_s
        } else {
            0.0
        }
    }

    /// Get mean firing rate (Hz) across all alive neurons.
    pub fn mean_firing_rate(&self) -> f32 {
        let time_s = self.time / 1000.0;
        if time_s <= 0.0 {
            return 0.0;
        }
        let total_spikes: u64 = self
            .neurons
            .spike_count
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.neurons.alive[i] != 0)
            .map(|(_, &s)| s as u64)
            .sum();
        let n_alive = self.neurons.alive.iter().filter(|&&a| a != 0).count() as f32;
        if n_alive > 0.0 {
            total_spikes as f32 / (n_alive * time_s)
        } else {
            0.0
        }
    }

    /// Count how many neurons fired this step.
    pub fn fired_count(&self) -> usize {
        self.neurons.fired.iter().filter(|&&f| f != 0).count()
    }

    /// Count how many neurons in a subset fired this step.
    pub fn fired_count_subset(&self, neuron_indices: &[usize]) -> usize {
        #[cfg(target_os = "macos")]
        if self.should_use_gpu() {
            if let Some(gpu_neurons) = self.gpu_neurons.as_ref() {
                return gpu_neurons.sum_fired_indices(neuron_indices) as usize;
            }
        }

        neuron_indices
            .iter()
            .filter(|&&idx| idx < self.neurons.count && self.neurons.fired[idx] != 0)
            .count()
    }

    /// Sum cumulative spike counters across a subset without forcing full host sync.
    pub fn spike_count_subset_sum(&self, neuron_indices: &[usize]) -> u64 {
        #[cfg(target_os = "macos")]
        if self.should_use_gpu() {
            if let Some(gpu_neurons) = self.gpu_neurons.as_ref() {
                return gpu_neurons.sum_spike_count_indices(neuron_indices);
            }
        }

        neuron_indices
            .iter()
            .filter(|&&idx| idx < self.neurons.count)
            .map(|&idx| self.neurons.spike_count[idx] as u64)
            .sum()
    }

    /// Apply a drug to the pharmacology engine.
    ///
    /// Dose is in mg. The drug's PK model (1-compartment Bateman) determines
    /// plasma concentration over time, and PD (Hill equation) maps concentration
    /// to conductance scale modifications.
    pub fn apply_drug(&mut self, drug: DrugType, dose_mg: f32) {
        self.pharmacology.apply_drug(drug, dose_mg);
    }

    /// Apply general anesthesia (multi-target: GABA-A 8x, NMDA 0.05x, AMPA 0.4x,
    /// Na_v 0.5x, K_leak 2x, PSC 0.1x, Orch-OR 0.05x).
    ///
    /// This should produce > 70% drop in consciousness composite score.
    pub fn apply_anesthesia(&mut self) {
        #[cfg(target_os = "macos")]
        self.prepare_host_neuron_mutation();
        self.pharmacology.apply_anesthesia(&mut self.neurons);
        #[cfg(target_os = "macos")]
        self.mark_neuron_shadow_dirty();
    }

    /// Reset all neuron state to resting values without changing connectivity.
    pub fn reset(&mut self) {
        let n = self.neurons.count;
        self.neurons = NeuronArrays::new(n);
        self.time = 0.0;
        self.step_count = 0;
        self.gs_input = vec![0.0; n];
        self.gi_input = vec![0.0; n];
        self.gq_input = vec![0.0; n];
        self.circadian = CircadianClock::new(3600.0);
        self.pharmacology = PharmacologyEngine::new();
        self.glia = GliaState::new(n);
        self.enable_glia = true;
        self.enable_circadian = true;
        self.enable_pharmacology = true;
        self.enable_gene_expression = true;
        self.enable_metabolism = true;
        self.enable_microtubules = true;
        #[cfg(target_os = "macos")]
        self.invalidate_gpu_resident_state();
    }

    /// Get the number of neurons.
    pub fn neuron_count(&self) -> usize {
        self.neurons.count
    }

    /// Get the number of synapses.
    pub fn synapse_count(&self) -> usize {
        self.synapses.n_synapses
    }

    /// Hebbian weight nudge: strengthen relay→correct_motor, weaken relay→wrong_motor.
    ///
    /// Used by the FEP protocol to accelerate learning after a HIT event.
    /// The nudge is position-aware: synapses from relay neurons to the
    /// correct motor population are strengthened, while synapses to wrong
    /// motor populations are weakened.
    ///
    /// # Arguments
    /// * `relay_ids` - Thalamic relay neuron IDs (presynaptic).
    /// * `correct_ids` - Correct motor population neuron IDs (postsynaptic, strengthen).
    /// * `wrong_ids` - Wrong motor population neuron IDs (postsynaptic, weaken).
    /// * `delta` - Weight update magnitude. Scale-adaptive:
    ///   typically `0.8 * max(1.0, (n_l5 / 200)^0.3)`.
    pub fn hebbian_nudge(
        &mut self,
        relay_ids: &[u32],
        correct_ids: &[u32],
        wrong_ids: &[u32],
        delta: f32,
    ) {
        #[cfg(target_os = "macos")]
        self.prepare_host_synapse_mutation();

        // Build fast lookup sets for correct and wrong populations
        let max_id = self.neurons.count as u32;
        let mut is_correct = vec![false; max_id as usize];
        let mut is_wrong = vec![false; max_id as usize];
        for &id in correct_ids {
            if (id as usize) < is_correct.len() {
                is_correct[id as usize] = true;
            }
        }
        for &id in wrong_ids {
            if (id as usize) < is_wrong.len() {
                is_wrong[id as usize] = true;
            }
        }

        // Iterate outgoing synapses for each relay neuron
        for &pre in relay_ids {
            if pre as usize >= self.neurons.count {
                continue;
            }
            let range = self.synapses.outgoing_range(pre as usize);
            for syn_idx in range {
                let post = self.synapses.col_indices[syn_idx] as usize;
                if post >= max_id as usize {
                    continue;
                }
                if is_correct[post] {
                    // Strengthen relay → correct motor
                    self.synapses.strength[syn_idx] =
                        (self.synapses.strength[syn_idx] + delta).clamp(0.3, 8.0);
                    self.synapses.recompute_weight(syn_idx);
                } else if is_wrong[post] {
                    // Weaken relay → wrong motor (smaller magnitude)
                    self.synapses.strength[syn_idx] =
                        (self.synapses.strength[syn_idx] - delta * 0.15).clamp(0.3, 8.0);
                    self.synapses.recompute_weight(syn_idx);
                }
            }
        }

        #[cfg(target_os = "macos")]
        self.mark_synapse_shadow_dirty();
    }

    /// Set a conductance scale on one neuron and keep resident GPU state coherent.
    pub fn set_conductance_scale(&mut self, neuron_idx: usize, channel: usize, scale: f32) {
        if neuron_idx >= self.neurons.count || channel >= IonChannelType::COUNT {
            return;
        }
        self.neurons.conductance_scale[neuron_idx][channel] = scale;
        #[cfg(target_os = "macos")]
        if let Some(gpu_neurons) = self.gpu_neurons.as_ref() {
            gpu_neurons.set_conductance_scale(neuron_idx, channel, scale);
        }
    }

    /// Set a neurotransmitter concentration on one neuron.
    pub fn set_nt_concentration(&mut self, neuron_idx: usize, nt: usize, concentration_nm: f32) {
        if neuron_idx >= self.neurons.count || nt >= NTType::COUNT {
            return;
        }
        self.neurons.nt_conc[neuron_idx][nt] = concentration_nm;
        #[cfg(target_os = "macos")]
        if let Some(gpu_neurons) = self.gpu_neurons.as_ref() {
            gpu_neurons.set_nt_concentration(neuron_idx, nt, concentration_nm);
        }
    }

    /// Add NT concentration to many neurons without forcing a full shadow sync.
    pub fn add_nt_concentration_many(
        &mut self,
        neuron_indices: &[usize],
        nt: usize,
        delta_nm: f32,
    ) {
        if nt >= NTType::COUNT {
            return;
        }
        for &idx in neuron_indices {
            if idx >= self.neurons.count {
                continue;
            }
            self.neurons.nt_conc[idx][nt] += delta_nm;
            #[cfg(target_os = "macos")]
            if let Some(gpu_neurons) = self.gpu_neurons.as_ref() {
                gpu_neurons.add_nt_concentration(idx, nt, delta_nm);
            }
        }
    }

    /// Adjust selected synapse strengths in both host shadow and resident GPU state.
    pub fn adjust_synapse_strengths(
        &mut self,
        synapse_indices: &[usize],
        delta: f32,
        min_strength: f32,
        max_strength: f32,
    ) {
        for &idx in synapse_indices {
            if idx >= self.synapses.n_synapses {
                continue;
            }
            let updated = (self.synapses.strength[idx] + delta).clamp(min_strength, max_strength);
            self.synapses.strength[idx] = updated;
            self.synapses.recompute_weight(idx);
            #[cfg(target_os = "macos")]
            if let Some(gpu_synapses) = self.gpu_synapses.as_ref() {
                gpu_synapses.set_strength(idx, updated);
            }
        }
    }

    /// Set selected synapse strengths explicitly.
    pub fn set_synapse_strengths(
        &mut self,
        synapse_indices: &[usize],
        strengths: &[f32],
        min_strength: f32,
        max_strength: f32,
    ) {
        for (&idx, &value) in synapse_indices.iter().zip(strengths.iter()) {
            if idx >= self.synapses.n_synapses {
                continue;
            }
            let clamped = value.clamp(min_strength, max_strength);
            self.synapses.strength[idx] = clamped;
            self.synapses.recompute_weight(idx);
            #[cfg(target_os = "macos")]
            if let Some(gpu_synapses) = self.gpu_synapses.as_ref() {
                gpu_synapses.set_strength(idx, clamped);
            }
        }
    }

    /// Clear cumulative spike counters while keeping resident GPU state coherent.
    pub fn reset_spike_counts(&mut self) {
        self.neurons
            .spike_count
            .iter_mut()
            .for_each(|count| *count = 0);
        #[cfg(target_os = "macos")]
        if let Some(gpu_neurons) = self.gpu_neurons.as_ref() {
            gpu_neurons.clear_spike_count();
        }
    }

    /// Refresh the CPU shadow arrays from resident GPU state, if needed.
    pub fn sync_shadow_from_gpu(&mut self) {
        #[cfg(target_os = "macos")]
        self.sync_shadow_from_gpu_internal();
    }

    /// Check whether GPU acceleration is active.
    pub fn gpu_available(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// Check whether the step loop will actually dispatch work to GPU.
    pub fn gpu_dispatch_active(&self) -> bool {
        self.should_use_gpu()
    }

    /// Trim nonessential biology for latency-sensitive closed-loop assays.
    pub fn enable_latency_benchmark_mode(&mut self) {
        self.enable_circadian = false;
        self.enable_pharmacology = false;
        self.enable_gene_expression = false;
        self.enable_metabolism = false;
        self.enable_microtubules = false;
        self.enable_glia = false;
    }

    /// Return the most recent GPU initialization error, if any.
    pub fn gpu_init_error(&self) -> Option<&str> {
        #[cfg(target_os = "macos")]
        {
            self.gpu_init_error.as_deref()
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }

    #[cfg(target_os = "macos")]
    fn invalidate_gpu_resident_state(&mut self) {
        self.gpu_neurons = None;
        self.gpu_synapses = None;
        self.gpu_glia = None;
        self.neuron_shadow_stale = false;
        self.synapse_shadow_stale = false;
        self.glia_shadow_stale = false;
        self.neuron_shadow_dirty = false;
        self.synapse_shadow_dirty = false;
        self.glia_shadow_dirty = false;
    }

    #[cfg(target_os = "macos")]
    fn ensure_gpu_resident_state(&mut self) -> bool {
        let Some(gpu) = self.gpu.as_ref() else {
            return false;
        };

        if self.gpu_neurons.is_none() {
            self.gpu_neurons = Some(gpu::state::MetalNeuronState::from_cpu(gpu, &self.neurons));
        }
        if self.gpu_synapses.is_none() {
            self.gpu_synapses = Some(gpu::state::MetalSynapseState::from_cpu(gpu, &self.synapses));
        }
        if self.gpu_glia.is_none() {
            self.gpu_glia = Some(gpu::state::MetalGliaState::from_cpu(gpu, &self.glia));
        }

        true
    }

    #[cfg(target_os = "macos")]
    fn upload_shadow_to_gpu_if_dirty(&mut self) {
        if !self.ensure_gpu_resident_state() {
            return;
        }

        if self.neuron_shadow_dirty {
            if let Some(gpu_neurons) = self.gpu_neurons.as_ref() {
                gpu_neurons.upload_all_from_cpu(&self.neurons);
            }
            self.neuron_shadow_dirty = false;
            self.neuron_shadow_stale = false;
        }

        if self.synapse_shadow_dirty {
            if let Some(gpu_synapses) = self.gpu_synapses.as_ref() {
                gpu_synapses.upload_all_from_cpu(&self.synapses);
            }
            self.synapse_shadow_dirty = false;
            self.synapse_shadow_stale = false;
        }

        if self.glia_shadow_dirty {
            if let Some(gpu_glia) = self.gpu_glia.as_ref() {
                gpu_glia.upload_all_from_cpu(&self.glia);
            }
            self.glia_shadow_dirty = false;
            self.glia_shadow_stale = false;
        }
    }

    #[cfg(target_os = "macos")]
    fn sync_shadow_from_gpu_internal(&mut self) {
        if self.neuron_shadow_stale {
            if let Some(gpu_neurons) = self.gpu_neurons.as_ref() {
                gpu_neurons.write_back_to_cpu(&mut self.neurons);
            }
            self.neuron_shadow_stale = false;
        }

        if self.synapse_shadow_stale {
            if let Some(gpu_synapses) = self.gpu_synapses.as_ref() {
                gpu_synapses.write_back_to_cpu(&mut self.synapses);
            }
            self.synapse_shadow_stale = false;
        }

        if self.glia_shadow_stale {
            if let Some(gpu_glia) = self.gpu_glia.as_ref() {
                gpu_glia.write_back_to_cpu(&mut self.glia);
            }
            self.glia_shadow_stale = false;
        }
    }

    #[cfg(target_os = "macos")]
    fn prepare_host_neuron_mutation(&mut self) {
        if self.neuron_shadow_stale {
            self.sync_shadow_from_gpu_internal();
        }
    }

    #[cfg(target_os = "macos")]
    fn prepare_host_synapse_mutation(&mut self) {
        if self.synapse_shadow_stale {
            self.sync_shadow_from_gpu_internal();
        }
    }

    #[cfg(target_os = "macos")]
    fn mark_neuron_shadow_dirty(&mut self) {
        self.neuron_shadow_dirty = true;
        self.neuron_shadow_stale = false;
    }

    #[cfg(target_os = "macos")]
    fn mark_synapse_shadow_dirty(&mut self) {
        self.synapse_shadow_dirty = true;
        self.synapse_shadow_stale = false;
    }

    #[cfg(target_os = "macos")]
    fn mark_glia_shadow_dirty(&mut self) {
        self.glia_shadow_dirty = true;
        self.glia_shadow_stale = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_brain_resting_state() {
        let brain = MolecularBrain::new(100);
        assert_eq!(brain.neuron_count(), 100);
        assert_eq!(brain.synapse_count(), 0);
        assert_eq!(brain.step_count, 0);
        assert!((brain.time - 0.0).abs() < 1e-6);
        assert!((brain.dt - DEFAULT_DT).abs() < 1e-6);
        assert!((brain.psc_scale - DEFAULT_PSC_SCALE).abs() < 1e-6);

        // All neurons should be at resting potential
        for i in 0..100 {
            assert!((brain.neurons.voltage[i] - INITIAL_VOLTAGE).abs() < 1e-6);
            assert_eq!(brain.neurons.fired[i], 0);
            assert_eq!(brain.neurons.spike_count[i], 0);
        }
    }

    #[test]
    fn test_from_edges_builds_csr() {
        let edges = vec![
            (0u32, 1, NTType::Glutamate),
            (0, 2, NTType::Glutamate),
            (1, 0, NTType::GABA),
        ];
        let brain = MolecularBrain::from_edges(3, &edges);
        assert_eq!(brain.neuron_count(), 3);
        assert_eq!(brain.synapse_count(), 3);

        // Neuron 0 has 2 outgoing synapses
        let range = brain.synapses.outgoing_range(0);
        assert_eq!(range.len(), 2);

        // Neuron 1 has 1 outgoing synapse
        let range = brain.synapses.outgoing_range(1);
        assert_eq!(range.len(), 1);

        // Neuron 2 has 0 outgoing synapses
        let range = brain.synapses.outgoing_range(2);
        assert_eq!(range.len(), 0);
    }

    #[test]
    fn test_stimulate_injects_current() {
        let mut brain = MolecularBrain::new(3);
        brain.stimulate(1, 50.0);
        assert!((brain.neurons.external_current[1] - 50.0).abs() < 1e-6);
        assert!((brain.neurons.external_current[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_stimulate_out_of_bounds_ignored() {
        let mut brain = MolecularBrain::new(3);
        brain.stimulate(999, 50.0); // should not panic
    }

    #[test]
    fn test_step_increments_time() {
        let mut brain = MolecularBrain::new(10);
        brain.enable_circadian = false;
        brain.enable_pharmacology = false;
        brain.enable_glia = false;
        brain.enable_gpu = false;

        brain.step();
        assert_eq!(brain.step_count, 1);
        assert!((brain.time - brain.dt).abs() < 1e-6);

        brain.step();
        assert_eq!(brain.step_count, 2);
        assert!((brain.time - 2.0 * brain.dt).abs() < 1e-5);
    }

    #[test]
    fn test_run_multiple_steps() {
        let mut brain = MolecularBrain::new(10);
        brain.enable_circadian = false;
        brain.enable_pharmacology = false;
        brain.enable_glia = false;
        brain.enable_gpu = false;

        brain.run(100);
        assert_eq!(brain.step_count, 100);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut brain = MolecularBrain::new(10);
        brain.enable_gpu = false;
        brain.enable_circadian = false;
        brain.enable_pharmacology = false;
        brain.enable_glia = false;
        brain.stimulate(0, 100.0);
        brain.run(10);

        brain.reset();
        assert_eq!(brain.step_count, 0);
        assert!((brain.time - 0.0).abs() < 1e-6);
        for i in 0..10 {
            assert!((brain.neurons.voltage[i] - INITIAL_VOLTAGE).abs() < 1e-6);
        }
    }

    #[test]
    fn test_external_current_consumed_each_step() {
        let mut brain = MolecularBrain::new(3);
        brain.enable_gpu = false;
        brain.enable_circadian = false;
        brain.enable_pharmacology = false;
        brain.enable_glia = false;

        brain.stimulate(0, 50.0);
        brain.step();

        // After step, external current should be zeroed
        assert!((brain.neurons.external_current[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_stimulation_changes_voltage_before_clear() {
        let mut brain = MolecularBrain::new(1);
        brain.enable_gpu = false;
        brain.enable_circadian = false;
        brain.enable_pharmacology = false;
        brain.enable_glia = false;

        let v_before = brain.neurons.voltage[0];
        brain.stimulate(0, 50.0);
        brain.step();

        assert!(
            brain.neurons.voltage[0] > v_before,
            "Injected current should depolarize the neuron: {:.3} -> {:.3}",
            v_before,
            brain.neurons.voltage[0],
        );
    }

    #[test]
    fn test_hebbian_nudge_strengthens_correct() {
        // Build a simple network: relay (0,1) -> motor correct (3,4) + wrong (5,6)
        let edges = vec![
            (0u32, 3, NTType::Glutamate),
            (0, 4, NTType::Glutamate),
            (0, 5, NTType::Glutamate),
            (0, 6, NTType::Glutamate),
            (1, 3, NTType::Glutamate),
            (1, 5, NTType::Glutamate),
        ];
        let mut brain = MolecularBrain::from_edges(8, &edges);

        // Record initial strengths for relay->correct and relay->wrong
        let get_strength = |b: &MolecularBrain, pre: usize, post: usize| -> Option<f32> {
            for syn_idx in b.synapses.outgoing_range(pre) {
                if b.synapses.col_indices[syn_idx] as usize == post {
                    return Some(b.synapses.strength[syn_idx]);
                }
            }
            None
        };

        let s_correct_before = get_strength(&brain, 0, 3).unwrap();
        let s_wrong_before = get_strength(&brain, 0, 5).unwrap();

        // Apply hebbian nudge: relay=[0,1], correct=[3,4], wrong=[5,6], delta=1.0
        brain.hebbian_nudge(&[0, 1], &[3, 4], &[5, 6], 1.0);

        let s_correct_after = get_strength(&brain, 0, 3).unwrap();
        let s_wrong_after = get_strength(&brain, 0, 5).unwrap();

        assert!(
            s_correct_after > s_correct_before,
            "Correct motor should be strengthened: {:.3} -> {:.3}",
            s_correct_before,
            s_correct_after,
        );
        assert!(
            s_wrong_after < s_wrong_before || s_wrong_after <= 0.3 + 1e-6,
            "Wrong motor should be weakened: {:.3} -> {:.3}",
            s_wrong_before,
            s_wrong_after,
        );
    }

    #[test]
    fn test_gprotein_inputs_resting() {
        let mut brain = MolecularBrain::new(5);
        brain.compute_gprotein_inputs();

        // At resting NT concentrations, Gs should be near zero (DA=20nM, D1 EC50=2340nM)
        // Gi should be moderately active (D2 EC50=2.8nM, DA=20nM -> high activation)
        for i in 0..5 {
            // Gs: DA=20, EC50=2340 -> very low activation
            assert!(brain.gs_input[i] < 0.05);
            // Gi: D2 has DA=20, EC50=2.8 -> high activation (~0.88)
            assert!(brain.gi_input[i] > 0.5);
            // All values in [0, 1]
            assert!(brain.gs_input[i] >= 0.0 && brain.gs_input[i] <= 1.0);
            assert!(brain.gi_input[i] >= 0.0 && brain.gi_input[i] <= 1.0);
            assert!(brain.gq_input[i] >= 0.0 && brain.gq_input[i] <= 1.0);
        }
    }
}
