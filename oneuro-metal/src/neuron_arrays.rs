//! Structure-of-Arrays neuron state for GPU-friendly memory layout.
//!
//! Each field is a contiguous `Vec<f32>` so Metal compute shaders can process
//! all neurons in parallel with coalesced memory access. ~80 f32 fields per
//! neuron — at 100K neurons this is ~32 MB, easily fits in Apple unified memory.

use crate::constants::*;
use crate::types::*;

/// SoA neuron state — every per-neuron value in a contiguous array.
pub struct NeuronArrays {
    /// Number of neurons.
    pub count: usize,

    // ===== Position & Identity =====
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub archetype: Vec<u8>, // NeuronArchetype as u8
    pub alive: Vec<u8>,     // bool packed as u8 (1=alive)

    // ===== Membrane =====
    pub voltage: Vec<f32>,          // mV
    pub prev_voltage: Vec<f32>,     // mV (for spike detection)
    pub fired: Vec<u8>,             // bool (1=fired this step)
    pub refractory_timer: Vec<f32>, // ms remaining
    pub spike_count: Vec<u32>,

    // ===== HH Gating Variables =====
    pub nav_m: Vec<f32>, // Na_v activation
    pub nav_h: Vec<f32>, // Na_v inactivation
    pub kv_n: Vec<f32>,  // K_v activation
    pub cav_m: Vec<f32>, // Ca_v activation
    pub cav_h: Vec<f32>, // Ca_v inactivation

    // ===== Conductance Scales (drug-modifiable, per channel type) =====
    pub conductance_scale: Vec<[f32; IonChannelType::COUNT]>, // [Nav,Kv,Kleak,Cav,NMDA,AMPA,GabaA,nAChR]

    // ===== Ligand-Gated Open Fractions =====
    pub ampa_open: Vec<f32>,
    pub nmda_open: Vec<f32>,
    pub gabaa_open: Vec<f32>,
    pub nachr_open: Vec<f32>,

    // ===== Calcium 4-Compartment =====
    pub ca_cytoplasmic: Vec<f32>,
    pub ca_er: Vec<f32>,
    pub ca_mitochondrial: Vec<f32>,
    pub ca_microdomain: Vec<f32>,

    // ===== Second Messengers =====
    pub camp: Vec<f32>,            // cAMP (nM)
    pub gs_active: Vec<f32>,       // G-protein Gs [0,1]
    pub gi_active: Vec<f32>,       // G-protein Gi [0,1]
    pub gq_active: Vec<f32>,       // G-protein Gq [0,1]
    pub pka_activity: Vec<f32>,    // [0,1]
    pub pkc_activity: Vec<f32>,    // [0,1]
    pub camkii_activity: Vec<f32>, // [0,1]
    pub ip3: Vec<f32>,             // nM
    pub dag: Vec<f32>,             // nM
    pub erk_activity: Vec<f32>,    // [0,1]
    pub er_ca_released: Vec<f32>,  // nM released from ER
    pub er_ca_store: Vec<f32>,     // nM ER lumen

    // ===== Phosphorylation State =====
    pub ampa_p: Vec<f32>, // AMPA GluA1 phosphorylation [0,1]
    pub kv_p: Vec<f32>,   // Kv phosphorylation [0,1]
    pub cav_p: Vec<f32>,  // CaV phosphorylation [0,1]
    pub creb_p: Vec<f32>, // CREB Ser133 phosphorylation [0,1]

    // ===== Metabolism =====
    pub atp: Vec<f32>,     // µM
    pub adp: Vec<f32>,     // µM
    pub glucose: Vec<f32>, // µM
    pub oxygen: Vec<f32>,  // µM
    pub energy: Vec<f32>,  // abstract energy (backward compat)

    // ===== NT Concentrations (local ambient) =====
    pub nt_conc: Vec<[f32; NTType::COUNT]>, // [DA, 5-HT, NE, ACh, GABA, Glu]

    // ===== External Current (zeroed after each step) =====
    pub external_current: Vec<f32>, // µA/cm²
    /// Synaptic current generated during the previous step and consumed by the
    /// next membrane integration pass.
    pub synaptic_current: Vec<f32>, // µA/cm²

    // ===== Gene Expression (slow, interval-gated) =====
    pub bdnf_level: Vec<f32>, // neurotrophin level [0,1]
    pub cfos_level: Vec<f32>, // IEG expression [0,1]
    pub arc_level: Vec<f32>,  // activity-regulated [0,1]

    // ===== Microtubule / Orch-OR =====
    pub mt_coherence: Vec<f32>,   // quantum coherence [0,1]
    pub orch_or_events: Vec<u32>, // collapse events

    // ===== Step tracking =====
    pub last_fired_step: Vec<u32>, // step number of last spike

    // ===== Circadian modulation (applied per-step from global) =====
    pub excitability_bias: Vec<f32>, // µA/cm²
}

impl NeuronArrays {
    /// Create arrays for `n` neurons, all at resting state.
    pub fn new(n: usize) -> Self {
        // Compute resting HH gating variables at V=-65 mV
        let v_rest = INITIAL_VOLTAGE;
        let (nav_m_rest, nav_h_rest) = resting_nav_gates(v_rest);
        let kv_n_rest = resting_kv_gate(v_rest);
        let (cav_m_rest, cav_h_rest) = resting_cav_gates(v_rest);

        let default_conductance = [1.0f32; IonChannelType::COUNT];
        let default_nt = [
            NTType::Dopamine.resting_conc_nm(),
            NTType::Serotonin.resting_conc_nm(),
            NTType::Norepinephrine.resting_conc_nm(),
            NTType::Acetylcholine.resting_conc_nm(),
            NTType::GABA.resting_conc_nm(),
            NTType::Glutamate.resting_conc_nm(),
        ];

        Self {
            count: n,
            x: vec![0.0; n],
            y: vec![0.0; n],
            z: vec![0.0; n],
            archetype: vec![NeuronArchetype::Pyramidal as u8; n],
            alive: vec![1; n],

            voltage: vec![v_rest; n],
            prev_voltage: vec![v_rest; n],
            fired: vec![0; n],
            refractory_timer: vec![0.0; n],
            spike_count: vec![0; n],

            nav_m: vec![nav_m_rest; n],
            nav_h: vec![nav_h_rest; n],
            kv_n: vec![kv_n_rest; n],
            cav_m: vec![cav_m_rest; n],
            cav_h: vec![cav_h_rest; n],

            conductance_scale: vec![default_conductance; n],

            ampa_open: vec![0.0; n],
            nmda_open: vec![0.0; n],
            gabaa_open: vec![0.0; n],
            nachr_open: vec![0.0; n],

            ca_cytoplasmic: vec![REST_CA_CYTOPLASMIC_NM; n],
            ca_er: vec![REST_CA_ER_NM; n],
            ca_mitochondrial: vec![REST_CA_MITOCHONDRIAL_NM; n],
            ca_microdomain: vec![REST_CA_MICRODOMAIN_NM; n],

            camp: vec![50.0; n],
            gs_active: vec![0.0; n],
            gi_active: vec![0.0; n],
            gq_active: vec![0.0; n],
            pka_activity: vec![0.0; n],
            pkc_activity: vec![0.0; n],
            camkii_activity: vec![0.0; n],
            ip3: vec![IP3_BASAL; n],
            dag: vec![DAG_BASAL; n],
            erk_activity: vec![0.0; n],
            er_ca_released: vec![0.0; n],
            er_ca_store: vec![SMS_ER_CA_STORE; n],

            ampa_p: vec![0.0; n],
            kv_p: vec![0.0; n],
            cav_p: vec![0.0; n],
            creb_p: vec![0.0; n],

            atp: vec![ATP_RESTING; n],
            adp: vec![ADP_RESTING; n],
            glucose: vec![GLUCOSE_RESTING; n],
            oxygen: vec![OXYGEN_RESTING; n],
            energy: vec![100.0; n],

            nt_conc: vec![default_nt; n],

            external_current: vec![0.0; n],
            synaptic_current: vec![0.0; n],

            bdnf_level: vec![0.0; n],
            cfos_level: vec![0.0; n],
            arc_level: vec![0.0; n],

            mt_coherence: vec![0.0; n],
            orch_or_events: vec![0; n],

            last_fired_step: vec![0; n],
            excitability_bias: vec![0.0; n],
        }
    }

    /// Clear per-step spike bookkeeping for the next integration pass.
    pub fn pre_step_clear(&mut self) {
        for i in 0..self.count {
            self.fired[i] = 0;
            self.prev_voltage[i] = self.voltage[i];
        }
    }

    /// Clear externally injected current after it has been consumed.
    pub fn clear_external_current(&mut self) {
        self.external_current.iter_mut().for_each(|c| *c = 0.0);
    }

    /// Add a neuron at position, returns its index.
    pub fn add_neuron(&mut self, x: f32, y: f32, z: f32, archetype: NeuronArchetype) -> usize {
        let idx = self.count;
        self.count += 1;
        let v_rest = INITIAL_VOLTAGE;
        let (nav_m_rest, nav_h_rest) = resting_nav_gates(v_rest);
        let kv_n_rest = resting_kv_gate(v_rest);
        let (cav_m_rest, cav_h_rest) = resting_cav_gates(v_rest);

        self.x.push(x);
        self.y.push(y);
        self.z.push(z);
        self.archetype.push(archetype as u8);
        self.alive.push(1);

        self.voltage.push(v_rest);
        self.prev_voltage.push(v_rest);
        self.fired.push(0);
        self.refractory_timer.push(0.0);
        self.spike_count.push(0);

        self.nav_m.push(nav_m_rest);
        self.nav_h.push(nav_h_rest);
        self.kv_n.push(kv_n_rest);
        self.cav_m.push(cav_m_rest);
        self.cav_h.push(cav_h_rest);

        self.conductance_scale.push([1.0; IonChannelType::COUNT]);

        self.ampa_open.push(0.0);
        self.nmda_open.push(0.0);
        self.gabaa_open.push(0.0);
        self.nachr_open.push(0.0);

        self.ca_cytoplasmic.push(REST_CA_CYTOPLASMIC_NM);
        self.ca_er.push(REST_CA_ER_NM);
        self.ca_mitochondrial.push(REST_CA_MITOCHONDRIAL_NM);
        self.ca_microdomain.push(REST_CA_MICRODOMAIN_NM);

        self.camp.push(50.0);
        self.gs_active.push(0.0);
        self.gi_active.push(0.0);
        self.gq_active.push(0.0);
        self.pka_activity.push(0.0);
        self.pkc_activity.push(0.0);
        self.camkii_activity.push(0.0);
        self.ip3.push(IP3_BASAL);
        self.dag.push(DAG_BASAL);
        self.erk_activity.push(0.0);
        self.er_ca_released.push(0.0);
        self.er_ca_store.push(SMS_ER_CA_STORE);

        self.ampa_p.push(0.0);
        self.kv_p.push(0.0);
        self.cav_p.push(0.0);
        self.creb_p.push(0.0);

        self.atp.push(ATP_RESTING);
        self.adp.push(ADP_RESTING);
        self.glucose.push(GLUCOSE_RESTING);
        self.oxygen.push(OXYGEN_RESTING);
        self.energy.push(100.0);

        let default_nt = [
            NTType::Dopamine.resting_conc_nm(),
            NTType::Serotonin.resting_conc_nm(),
            NTType::Norepinephrine.resting_conc_nm(),
            NTType::Acetylcholine.resting_conc_nm(),
            NTType::GABA.resting_conc_nm(),
            NTType::Glutamate.resting_conc_nm(),
        ];
        self.nt_conc.push(default_nt);

        self.external_current.push(0.0);
        self.synaptic_current.push(0.0);

        self.bdnf_level.push(0.0);
        self.cfos_level.push(0.0);
        self.arc_level.push(0.0);

        self.mt_coherence.push(0.0);
        self.orch_or_events.push(0);

        self.last_fired_step.push(0);
        self.excitability_bias.push(0.0);

        idx
    }

    /// Get indices of all neurons that fired this step.
    pub fn fired_indices(&self) -> Vec<usize> {
        (0..self.count)
            .filter(|&i| self.fired[i] != 0 && self.alive[i] != 0)
            .collect()
    }

    /// Get indices of all alive neurons.
    pub fn alive_indices(&self) -> Vec<usize> {
        (0..self.count).filter(|&i| self.alive[i] != 0).collect()
    }
}

// ===== HH Resting Gate Calculations =====

/// Na_v resting gates at given voltage.
fn resting_nav_gates(v: f32) -> (f32, f32) {
    let am = alpha_m(v);
    let bm = beta_m(v);
    let ah = alpha_h(v);
    let bh = beta_h(v);
    (am / (am + bm), ah / (ah + bh))
}

/// K_v resting gate at given voltage.
fn resting_kv_gate(v: f32) -> f32 {
    let an = alpha_n(v);
    let bn = beta_n(v);
    an / (an + bn)
}

/// Ca_v resting gates at given voltage.
fn resting_cav_gates(v: f32) -> (f32, f32) {
    let am = alpha_m_ca(v);
    let bm = beta_m_ca(v);
    let ah = alpha_h_ca(v);
    let bh = beta_h_ca(v);
    (am / (am + bm), ah / (ah + bh))
}

// ===== HH Rate Functions (CPU fallback; also used for initialization) =====

#[inline(always)]
pub fn alpha_m(v: f32) -> f32 {
    if (v + 40.0).abs() < 1e-6 {
        1.0
    } else {
        0.1 * (v + 40.0) / (1.0 - (-((v + 40.0) / 10.0)).exp())
    }
}

#[inline(always)]
pub fn beta_m(v: f32) -> f32 {
    4.0 * (-((v + 65.0) / 18.0)).exp()
}

#[inline(always)]
pub fn alpha_h(v: f32) -> f32 {
    0.07 * (-((v + 65.0) / 20.0)).exp()
}

#[inline(always)]
pub fn beta_h(v: f32) -> f32 {
    1.0 / (1.0 + (-((v + 35.0) / 10.0)).exp())
}

#[inline(always)]
pub fn alpha_n(v: f32) -> f32 {
    if (v + 55.0).abs() < 1e-6 {
        0.1
    } else {
        0.01 * (v + 55.0) / (1.0 - (-((v + 55.0) / 10.0)).exp())
    }
}

#[inline(always)]
pub fn beta_n(v: f32) -> f32 {
    0.125 * (-((v + 65.0) / 80.0)).exp()
}

#[inline(always)]
pub fn alpha_m_ca(v: f32) -> f32 {
    if (v + 27.0).abs() < 1e-6 {
        0.5
    } else {
        0.055 * (v + 27.0) / (1.0 - (-((v + 27.0) / 3.8)).exp())
    }
}

#[inline(always)]
pub fn beta_m_ca(v: f32) -> f32 {
    0.94 * (-((v + 75.0) / 17.0)).exp()
}

#[inline(always)]
pub fn alpha_h_ca(v: f32) -> f32 {
    0.000457 * (-((v + 13.0) / 50.0)).exp()
}

#[inline(always)]
pub fn beta_h_ca(v: f32) -> f32 {
    0.0065 / (1.0 + (-((v + 15.0) / 28.0)).exp())
}
