//! CSR sparse format for synaptic connectivity + per-synapse state.
//!
//! Compressed Sparse Row lets us efficiently iterate outgoing synapses
//! per source neuron (critical for spike propagation). Per-synapse state
//! includes vesicle pools, cleft dynamics, STDP traces, and BCM theta.

use crate::constants::*;
use crate::types::*;

/// CSR-formatted synapse storage with per-synapse molecular state.
pub struct SynapseArrays {
    /// Number of neurons (rows in CSR).
    pub n_neurons: usize,
    /// Total number of synapses.
    pub n_synapses: usize,

    // ===== CSR Structure =====
    /// Row offsets: `row_offsets[i]..row_offsets[i+1]` are synapse indices for neuron i.
    pub row_offsets: Vec<u32>,
    /// Target (postsynaptic) neuron ID for each synapse.
    pub col_indices: Vec<u32>,
    /// Source (presynaptic) neuron ID for each synapse. Stored explicitly so
    /// GPU kernels can address synapses without reconstructing CSR rows.
    pub pre_indices: Vec<u32>,

    // ===== Synapse Identity =====
    pub nt_type: Vec<u8>, // NTType as u8
    pub delay: Vec<f32>,  // ms

    // ===== Synaptic Strength =====
    pub weight: Vec<f32>,   // effective weight [0, 2]
    pub strength: Vec<f32>, // synaptic health [0, 1]

    // ===== Vesicle Pool =====
    pub vesicle_rrp: Vec<f32>,       // readily releasable pool
    pub vesicle_recycling: Vec<f32>, // recycling pool
    pub vesicle_reserve: Vec<f32>,   // reserve pool

    // ===== Synaptic Cleft =====
    pub cleft_concentration: Vec<f32>, // NT concentration in cleft (nM)

    // ===== Receptor Counts (for STDP trafficking) =====
    pub ampa_receptors: Vec<u16>,  // AMPA receptor count
    pub nmda_receptors: Vec<u16>,  // NMDA receptor count
    pub gabaa_receptors: Vec<u16>, // GABA-A receptor count

    // ===== STDP Traces =====
    pub last_pre_spike: Vec<f32>,    // time of last presynaptic spike
    pub last_post_spike: Vec<f32>,   // time of last postsynaptic spike
    pub eligibility_trace: Vec<f32>, // for reward-modulated learning

    // ===== BCM Metaplasticity =====
    pub bcm_theta: Vec<f32>, // sliding LTP threshold
    pub post_activity_history: Vec<f32>,

    // ===== Synaptic Tagging =====
    pub tagged: Vec<u8>, // bool
    pub tag_strength: Vec<f32>,

    // ===== Homeostatic Scaling =====
    pub homeostatic_scale: Vec<f32>,

    // ===== Delay Buffer =====
    /// Pending releases: (release_time, amount_nM) per synapse.
    /// Small Vec per synapse — most are empty.
    pub pending_releases: Vec<Vec<(f32, f32)>>,
}

impl SynapseArrays {
    /// Create empty synapse storage for n_neurons.
    pub fn new(n_neurons: usize) -> Self {
        Self {
            n_neurons,
            n_synapses: 0,
            row_offsets: vec![0; n_neurons + 1],
            col_indices: Vec::new(),
            pre_indices: Vec::new(),
            nt_type: Vec::new(),
            delay: Vec::new(),
            weight: Vec::new(),
            strength: Vec::new(),
            vesicle_rrp: Vec::new(),
            vesicle_recycling: Vec::new(),
            vesicle_reserve: Vec::new(),
            cleft_concentration: Vec::new(),
            ampa_receptors: Vec::new(),
            nmda_receptors: Vec::new(),
            gabaa_receptors: Vec::new(),
            last_pre_spike: Vec::new(),
            last_post_spike: Vec::new(),
            eligibility_trace: Vec::new(),
            bcm_theta: Vec::new(),
            post_activity_history: Vec::new(),
            tagged: Vec::new(),
            tag_strength: Vec::new(),
            homeostatic_scale: Vec::new(),
            pending_releases: Vec::new(),
        }
    }

    /// Build CSR from a list of (pre, post, nt_type) edges.
    /// Edges must be sorted by pre neuron ID.
    pub fn from_edges(n_neurons: usize, edges: &[(u32, u32, NTType)]) -> Self {
        let n_synapses = edges.len();
        let mut arrays = Self::new(n_neurons);
        arrays.n_synapses = n_synapses;

        // Build row offsets
        arrays.row_offsets = vec![0u32; n_neurons + 1];
        for &(pre, _, _) in edges {
            arrays.row_offsets[pre as usize + 1] += 1;
        }
        for i in 1..=n_neurons {
            arrays.row_offsets[i] += arrays.row_offsets[i - 1];
        }

        // Fill per-synapse data
        arrays.pre_indices = edges.iter().map(|e| e.0).collect();
        arrays.col_indices = edges.iter().map(|e| e.1).collect();
        arrays.nt_type = edges.iter().map(|e| e.2 as u8).collect();
        arrays.delay = vec![1.0; n_synapses];
        arrays.strength = vec![1.0; n_synapses];

        // Initialize receptor counts based on NT type
        arrays.ampa_receptors = vec![0; n_synapses];
        arrays.nmda_receptors = vec![0; n_synapses];
        arrays.gabaa_receptors = vec![0; n_synapses];
        for (i, &(_, _, nt)) in edges.iter().enumerate() {
            match nt {
                NTType::Glutamate => {
                    arrays.ampa_receptors[i] = 50;
                    arrays.nmda_receptors[i] = 20;
                }
                NTType::GABA => {
                    arrays.gabaa_receptors[i] = 40;
                }
                _ => {
                    arrays.ampa_receptors[i] = 10;
                }
            }
        }

        // Compute initial weights from receptor counts
        arrays.weight = (0..n_synapses)
            .map(|i| {
                let total = arrays.ampa_receptors[i] as f32
                    + arrays.nmda_receptors[i] as f32
                    + arrays.gabaa_receptors[i] as f32;
                (total / 50.0).min(2.0)
            })
            .collect();

        // Vesicle pools
        arrays.vesicle_rrp = vec![VESICLE_RRP_MAX; n_synapses];
        arrays.vesicle_recycling = vec![VESICLE_RECYCLING_MAX; n_synapses];
        arrays.vesicle_reserve = vec![VESICLE_RESERVE_MAX; n_synapses];

        // Cleft
        arrays.cleft_concentration = vec![0.0; n_synapses];

        // STDP
        arrays.last_pre_spike = vec![-1000.0; n_synapses];
        arrays.last_post_spike = vec![-1000.0; n_synapses];
        arrays.eligibility_trace = vec![0.0; n_synapses];

        // BCM
        arrays.bcm_theta = vec![0.5; n_synapses];
        arrays.post_activity_history = vec![0.0; n_synapses];

        // Tagging
        arrays.tagged = vec![0; n_synapses];
        arrays.tag_strength = vec![0.0; n_synapses];

        // Homeostatic
        arrays.homeostatic_scale = vec![1.0; n_synapses];

        // Delay buffer
        arrays.pending_releases = vec![Vec::new(); n_synapses];

        arrays
    }

    /// Iterate outgoing synapse indices for a given source neuron.
    #[inline]
    pub fn outgoing_range(&self, pre_neuron: usize) -> std::ops::Range<usize> {
        let start = self.row_offsets[pre_neuron] as usize;
        let end = self.row_offsets[pre_neuron + 1] as usize;
        start..end
    }

    /// Process a presynaptic spike: release vesicles into delay buffer.
    pub fn presynaptic_spike(&mut self, synapse_idx: usize, time: f32, ca_level_nm: f32) {
        // Ca-dependent release probability
        let ca_factor = (ca_level_nm / 500.0).min(2.0);
        let release_prob = VESICLE_BASE_RELEASE_PROB * ca_factor;

        let vesicles_released = self.vesicle_rrp[synapse_idx] * release_prob;
        let vesicles_released = vesicles_released.min(self.vesicle_rrp[synapse_idx]);
        self.vesicle_rrp[synapse_idx] -= vesicles_released;

        let nt_released = vesicles_released * VESICLE_NT_PER_RELEASE_NM;
        let release_time = time + self.delay[synapse_idx];

        self.pending_releases[synapse_idx].push((release_time, nt_released));
        self.last_pre_spike[synapse_idx] = time;
    }

    /// Update cleft dynamics for a single synapse. Returns current cleft concentration.
    pub fn update_cleft(&mut self, synapse_idx: usize, time: f32, dt: f32) -> f32 {
        // Check delay buffer
        let pending = &mut self.pending_releases[synapse_idx];
        let mut released = 0.0;
        pending.retain(|&(t, amount)| {
            if time >= t {
                released += amount;
                false
            } else {
                true
            }
        });
        self.cleft_concentration[synapse_idx] += released;

        if self.cleft_concentration[synapse_idx] <= 0.0 {
            self.replenish_vesicles(synapse_idx, dt);
            return 0.0;
        }

        // Enzymatic degradation (simplified Michaelis-Menten)
        let nt = self.nt_type[synapse_idx];
        let half_life = NTType::from_u8(nt).half_life_ms();
        let decay_rate = 0.693 / half_life; // ln(2)/t_half
        let mut conc = self.cleft_concentration[synapse_idx];
        conc -= conc * decay_rate * dt;

        // Diffusion + reuptake
        conc *= 1.0 - CLEFT_DIFFUSION_RATE * dt;
        conc *= 1.0 - CLEFT_REUPTAKE_RATE * dt;
        conc = conc.max(0.0);
        self.cleft_concentration[synapse_idx] = conc;

        // Replenish vesicles
        self.replenish_vesicles(synapse_idx, dt);

        conc
    }

    /// Replenish vesicle pools.
    fn replenish_vesicles(&mut self, idx: usize, dt: f32) {
        // Reserve → Recycling
        let to_recycling = (VESICLE_RECYCLING_REFILL_RATE * dt)
            .min(self.vesicle_reserve[idx])
            .min(VESICLE_RECYCLING_MAX - self.vesicle_recycling[idx])
            .max(0.0);
        self.vesicle_reserve[idx] -= to_recycling;
        self.vesicle_recycling[idx] += to_recycling;

        // Recycling → RRP
        let to_rrp = (VESICLE_RRP_REFILL_RATE * dt)
            .min(self.vesicle_recycling[idx])
            .min(VESICLE_RRP_MAX - self.vesicle_rrp[idx])
            .max(0.0);
        self.vesicle_recycling[idx] -= to_rrp;
        self.vesicle_rrp[idx] += to_rrp;
    }

    /// Recompute weight from receptor counts for a synapse.
    pub fn recompute_weight(&mut self, idx: usize) {
        let total = self.ampa_receptors[idx] as f32
            + self.nmda_receptors[idx] as f32
            + self.gabaa_receptors[idx] as f32;
        self.weight[idx] =
            (total / 50.0).min(2.0) * self.strength[idx] * self.homeostatic_scale[idx];
    }

    /// Check if synapse should be pruned.
    pub fn should_prune(&self, idx: usize) -> bool {
        let total = self.ampa_receptors[idx] as u32
            + self.nmda_receptors[idx] as u32
            + self.gabaa_receptors[idx] as u32;
        self.strength[idx] < 0.1 || total < 5
    }

    /// Get indices of synapses with non-zero cleft concentration (active synapses).
    pub fn active_synapse_indices(&self) -> Vec<usize> {
        (0..self.n_synapses)
            .filter(|&i| self.cleft_concentration[i] > 0.0)
            .collect()
    }
}

impl NTType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => NTType::Dopamine,
            1 => NTType::Serotonin,
            2 => NTType::Norepinephrine,
            3 => NTType::Acetylcholine,
            4 => NTType::GABA,
            5 => NTType::Glutamate,
            _ => NTType::Glutamate,
        }
    }
}
