//! Biophysical constants from Hodgkin-Huxley and neuroscience literature.
//!
//! All conductances in mS/cm², potentials in mV, concentrations in nM,
//! time in ms, currents in µA/cm².

// ===== Membrane =====
pub const DEFAULT_C_M: f32 = 1.0; // µF/cm² membrane capacitance
pub const INITIAL_VOLTAGE: f32 = -65.0; // mV resting potential
pub const AP_THRESHOLD: f32 = -20.0; // mV action potential detection
pub const REFRACTORY_PERIOD: f32 = 2.0; // ms
pub const VOLTAGE_MIN: f32 = -100.0; // mV clamp
pub const VOLTAGE_MAX: f32 = 60.0; // mV clamp

// ===== Ion Channel Parameters =====
// Maximal conductances (mS/cm²) and reversal potentials (mV)
pub const NAV_G_MAX: f32 = 120.0;
pub const NAV_E_REV: f32 = 50.0;

pub const KV_G_MAX: f32 = 36.0;
pub const KV_E_REV: f32 = -77.0;

pub const KLEAK_G_MAX: f32 = 0.3;
pub const KLEAK_E_REV: f32 = -77.0;

pub const CAV_G_MAX: f32 = 4.4;
pub const CAV_E_REV: f32 = 120.0;

pub const NMDA_G_MAX: f32 = 0.5;
pub const NMDA_E_REV: f32 = 0.0;

pub const AMPA_G_MAX: f32 = 1.0;
pub const AMPA_E_REV: f32 = 0.0;

pub const GABAA_G_MAX: f32 = 1.0;
pub const GABAA_E_REV: f32 = -80.0;

pub const NACHR_G_MAX: f32 = 0.8;
pub const NACHR_E_REV: f32 = 0.0;

// NMDA Mg2+ block
pub const NMDA_MG_CONC_MM: f32 = 1.0;

// EC50 values for ligand-gated channels (nM)
pub const NMDA_EC50_NM: f32 = 3000.0;
pub const AMPA_EC50_NM: f32 = 500.0;
pub const GABAA_EC50_NM: f32 = 200.0;
pub const NACHR_EC50_NM: f32 = 30.0;

// ===== HH Rate Constants =====
// All α/β functions defined as Metal shader constants — see hh_gating.metal
// Singularity guards: |V + offset| < 1e-6

// ===== Calcium System =====
pub const REST_CA_CYTOPLASMIC_NM: f32 = 50.0;
pub const REST_CA_ER_NM: f32 = 500_000.0;
pub const REST_CA_MITOCHONDRIAL_NM: f32 = 100.0;
pub const REST_CA_MICRODOMAIN_NM: f32 = 50.0;
pub const SPIKE_CA_INFLUX_NM: f32 = 50_000.0;
pub const MICRODOMAIN_DIFFUSION_TAU: f32 = 0.5;
pub const CA_BUFFER_CAPACITY: f32 = 20.0;
pub const CA_PASSIVE_LEAK_RATE: f32 = 210.0;

// IP3R parameters
pub const IP3R_VMAX: f32 = 6000.0;
pub const IP3R_K_IP3: f32 = 300.0;
pub const IP3R_K_ACT: f32 = 200.0;
pub const IP3R_K_INH: f32 = 500.0;
pub const IP3R_HILL_IP3: f32 = 2.0;
pub const IP3R_HILL_ACT: f32 = 2.0;
pub const IP3R_HILL_INH: f32 = 2.0;

// RyR parameters
pub const RYR_VMAX: f32 = 2000.0;
pub const RYR_K_ACT: f32 = 500.0;
pub const RYR_HILL: f32 = 3.0;
pub const RYR_K_INH: f32 = 1500.0;
pub const RYR_HILL_INH: f32 = 3.0;

// SERCA parameters
pub const SERCA_VMAX: f32 = 1500.0;
pub const SERCA_KM: f32 = 200.0;
pub const SERCA_HILL: f32 = 2.0;

// MCU parameters
pub const MCU_VMAX: f32 = 800.0;
pub const MCU_KM: f32 = 10_000.0;
pub const MCU_HILL: f32 = 2.5;

// PMCA parameters
pub const PMCA_VMAX: f32 = 300.0;
pub const PMCA_KM: f32 = 100.0;

// NCX parameters
pub const NCX_VMAX: f32 = 1500.0;
pub const NCX_KM: f32 = 1000.0;

// CaMKII parameters
pub const CAMKII_KD: f32 = 1000.0;
pub const CAMKII_HILL: f32 = 4.0;

// ===== Second Messenger System =====
// Adenylyl cyclase
pub const AC_BASAL_RATE: f32 = 0.05;
pub const AC_GS_VMAX: f32 = 1.5;
pub const AC_GS_KM: f32 = 0.5;
pub const AC_GI_INHIBITION_MAX: f32 = 0.85;
pub const AC_GI_KM: f32 = 0.4;

// PDE
pub const PDE_VMAX: f32 = 2.0;
pub const PDE_KM: f32 = 500.0;

// PKA
pub const PKA_HILL_N: f32 = 1.7;
pub const PKA_KA: f32 = 200.0;

// PLC
pub const PLC_VMAX: f32 = 1.0;
pub const PLC_KM: f32 = 0.4;

// IP3 dynamics
pub const IP3_DEGRADATION_RATE: f32 = 0.003;
pub const IP3_BASAL: f32 = 10.0;

// DAG dynamics
pub const DAG_DEGRADATION_RATE: f32 = 0.002;
pub const DAG_BASAL: f32 = 5.0;

// ER Ca2+ release via second messengers
pub const SMS_ER_CA_STORE: f32 = 200_000.0;
pub const SMS_IP3R_EC50: f32 = 300.0;
pub const SMS_IP3R_HILL_N: f32 = 2.5;
pub const SMS_ER_RELEASE_RATE: f32 = 0.001;
pub const SMS_SERCA_VMAX: f32 = 0.4;
pub const SMS_SERCA_KM: f32 = 200.0;

// PKC
pub const PKC_DAG_EC50: f32 = 100.0;
pub const PKC_CA_EC50: f32 = 600.0;
pub const PKC_HILL_N: f32 = 1.5;

// CaMKII (SMS pathway)
pub const SMS_CAMKII_CA_THRESHOLD: f32 = 500.0;
pub const SMS_CAMKII_CA_EC50: f32 = 800.0;
pub const SMS_CAMKII_HILL_N: f32 = 4.0;
pub const SMS_CAMKII_AUTOPHOSPHO_RATE: f32 = 0.001;
pub const SMS_CAMKII_PHOSPHATASE_RATE: f32 = 0.0003;

// MAPK/ERK
pub const ERK_ACTIVATION_RATE: f32 = 0.0002;
pub const ERK_DEACTIVATION_RATE: f32 = 0.0005;
pub const ERK_PKA_WEIGHT: f32 = 0.3;
pub const ERK_PKC_WEIGHT: f32 = 0.5;
pub const ERK_CAMKII_WEIGHT: f32 = 0.2;

// CREB
pub const CREB_PHOSPHO_RATE: f32 = 0.0005;
pub const CREB_DEPHOSPHO_RATE: f32 = 0.0002;

// Phosphorylation rates
pub const PKA_PHOSPHO_RATE: f32 = 0.002;
pub const PKA_DEPHOSPHO_RATE: f32 = 0.001;
pub const PKC_PHOSPHO_RATE: f32 = 0.002;
pub const PKC_DEPHOSPHO_RATE: f32 = 0.001;
pub const CAMKII_PHOSPHO_RATE: f32 = 0.003;

// ===== Vesicle Pool =====
pub const VESICLE_NT_PER_RELEASE_NM: f32 = 3000.0;
pub const VESICLE_RRP_MAX: f32 = 10.0;
pub const VESICLE_RECYCLING_MAX: f32 = 50.0;
pub const VESICLE_RESERVE_MAX: f32 = 200.0;
pub const VESICLE_RRP_REFILL_RATE: f32 = 0.05;
pub const VESICLE_RECYCLING_REFILL_RATE: f32 = 0.01;
pub const VESICLE_BASE_RELEASE_PROB: f32 = 0.3;

// ===== Synaptic Cleft =====
pub const CLEFT_DIFFUSION_RATE: f32 = 0.5;
pub const CLEFT_REUPTAKE_RATE: f32 = 0.1;

// ===== STDP =====
pub const STDP_WINDOW_MS: f32 = 20.0;
pub const STDP_LTP_RATE: f32 = 0.5;
pub const STDP_LTD_RATE: f32 = 0.5;

// ===== PSC =====
pub const DEFAULT_PSC_SCALE: f32 = 30.0;

// ===== Gene Expression =====
pub const GENE_EXPRESSION_INTERVAL: u32 = 10;
pub const METABOLISM_INTERVAL: u32 = 5;
pub const CYTOSKELETON_INTERVAL: u32 = 10;
pub const GLIA_INTERVAL: u32 = 10;
pub const LAZY_INACTIVE_THRESHOLD: u32 = 50;

// ===== Transcription Factor Dynamics (ms) =====
pub const TF_ACTIVATION_TAU: f32 = 5000.0;
pub const TF_DEACTIVATION_TAU: f32 = 30000.0;
pub const TF_NUCLEAR_TRANSLOCATION_TAU: f32 = 10000.0;

// ===== Metabolism =====
pub const ATP_RESTING: f32 = 5000.0; // µM
pub const ADP_RESTING: f32 = 500.0; // µM
pub const GLUCOSE_RESTING: f32 = 5000.0; // µM
pub const OXYGEN_RESTING: f32 = 100.0; // µM

// ===== Enzyme kinetics =====
// Km in µM, kcat in s⁻¹
pub const ACHE_KM: f32 = 90.0;
pub const ACHE_KCAT: f32 = 14000.0;

pub const MAO_A_KM: f32 = 178.0;
pub const MAO_A_VMAX_REL: f32 = 0.3;

pub const MAO_B_KM: f32 = 220.0;
pub const MAO_B_VMAX_REL: f32 = 0.25;

pub const COMT_KM: f32 = 200.0;
pub const COMT_VMAX_REL: f32 = 0.15;

pub const GABAT_KM: f32 = 1200.0;
pub const GABAT_VMAX_REL: f32 = 0.4;

pub const GLNS_KM: f32 = 3000.0;
pub const GLNS_VMAX_REL: f32 = 0.5;

// ===== Circadian =====
pub const CIRCADIAN_PERIOD_H: f32 = 24.0;

// ===== Global defaults =====
pub const DEFAULT_DT: f32 = 0.1; // ms
pub const DEFAULT_NETWORK_SIZE: [f32; 3] = [10.0, 10.0, 10.0];

/// Inline helper: Hill equation x^n / (ec50^n + x^n).
#[inline(always)]
pub fn hill(x: f32, ec50: f32, n: f32) -> f32 {
    if x <= 0.0 || ec50 <= 0.0 {
        return 0.0;
    }
    let xn = x.powf(n);
    xn / (ec50.powf(n) + xn)
}

/// Inline helper: Michaelis-Menten rate = Vmax * S / (Km + S).
#[inline(always)]
pub fn michaelis_menten(substrate: f32, vmax: f32, km: f32) -> f32 {
    if substrate <= 0.0 {
        return 0.0;
    }
    vmax * substrate / (km + substrate)
}

/// Inline helper: clamp to [lo, hi].
#[inline(always)]
pub fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}
