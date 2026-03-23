//! GPU-resident neural state -- persistent CudaSlice buffers allocated once.
//!
//! Mirrors [`NeuronArrays`] and [`SynapseArrays`] but all data lives on the
//! CUDA device.  No H2D / D2H copies happen during the hot simulation loop;
//! only explicit `download_*` calls transfer results back to host for
//! inspection or Python-side consumption.
//!
//! # Memory layout
//!
//! Array-of-Struct fields from the CPU side (e.g. `conductance_scale: [f32; 8]`
//! per neuron) are transposed into Struct-of-Array (SoA) form on the GPU so
//! that each CUDA warp accesses a contiguous cache line.  The flattened index
//! for channel `ch` of neuron `i` is `ch * N + i`.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::neuron_arrays::NeuronArrays;
#[cfg(feature = "cuda")]
use crate::synapse_arrays::SynapseArrays;
#[cfg(feature = "cuda")]
use crate::types::IonChannelType;

// ---------------------------------------------------------------------------
// CudaNeuronState
// ---------------------------------------------------------------------------

/// GPU-resident neuron state.
///
/// All fields are persistent [`CudaSlice`] buffers allocated once at
/// construction.  The simulation loop operates entirely on these buffers via
/// CUDA kernels -- no per-step host allocation or transfer.
#[cfg(feature = "cuda")]
pub struct CudaNeuronState {
    /// Total neuron count (constant after construction).
    pub n: u32,

    // ===== Membrane =====
    pub voltage: CudaSlice<f32>,
    pub prev_voltage: CudaSlice<f32>,
    pub fired: CudaSlice<u8>,
    pub refractory_timer: CudaSlice<f32>,
    pub spike_count: CudaSlice<u32>,

    // ===== HH Gating Variables =====
    pub nav_m: CudaSlice<f32>,
    pub nav_h: CudaSlice<f32>,
    pub kv_n: CudaSlice<f32>,
    pub cav_m: CudaSlice<f32>,
    pub cav_h: CudaSlice<f32>,

    // ===== Conductance Scales (flattened SoA: cond_scale[ch * N + i]) =====
    /// 8 channels x N neurons = 8N elements.  Channel order matches
    /// [`IonChannelType`]: Nav, Kv, Kleak, Cav, NMDA, AMPA, GabaA, nAChR.
    pub cond_scale: CudaSlice<f32>,

    // ===== Ligand-Gated Receptor Open Fractions =====
    pub ampa_open: CudaSlice<f32>,
    pub nmda_open: CudaSlice<f32>,
    pub gabaa_open: CudaSlice<f32>,
    pub nachr_open: CudaSlice<f32>,

    // ===== Calcium 4-Compartment =====
    pub ca_cyto: CudaSlice<f32>,
    pub ca_er: CudaSlice<f32>,
    pub ca_mito: CudaSlice<f32>,
    pub ca_micro: CudaSlice<f32>,

    // ===== Second Messengers =====
    pub camp: CudaSlice<f32>,
    pub gs_active: CudaSlice<f32>,
    pub gi_active: CudaSlice<f32>,
    pub gq_active: CudaSlice<f32>,
    pub pka_activity: CudaSlice<f32>,
    pub pkc_activity: CudaSlice<f32>,
    pub camkii_activity: CudaSlice<f32>,
    pub ip3: CudaSlice<f32>,
    pub dag: CudaSlice<f32>,
    pub erk_activity: CudaSlice<f32>,
    pub er_ca_released: CudaSlice<f32>,
    pub er_ca_store: CudaSlice<f32>,

    // ===== Phosphorylation State =====
    pub ampa_p: CudaSlice<f32>,
    pub kv_p: CudaSlice<f32>,
    pub cav_p: CudaSlice<f32>,
    pub creb_p: CudaSlice<f32>,

    // ===== Metabolism =====
    pub atp: CudaSlice<f32>,
    pub adp: CudaSlice<f32>,
    pub glucose: CudaSlice<f32>,
    pub oxygen: CudaSlice<f32>,
    pub energy: CudaSlice<f32>,

    // ===== NT Concentrations (flattened SoA: nt_conc[nt * N + i]) =====
    /// 6 NTs x N neurons = 6N elements.  NT order matches [`NTType`]:
    /// DA, 5-HT, NE, ACh, GABA, Glu.
    pub nt_conc: CudaSlice<f32>,

    // ===== External Current (zeroed each step before re-injection) =====
    pub ext_current: CudaSlice<f32>,

    // ===== Identity =====
    pub alive: CudaSlice<u8>,
    pub archetype: CudaSlice<u8>,
}

#[cfg(feature = "cuda")]
impl CudaNeuronState {
    /// Upload a complete [`NeuronArrays`] from CPU to persistent GPU buffers.
    ///
    /// AoS fields (`conductance_scale`, `nt_conc`) are transposed into SoA
    /// layout during upload so that CUDA warps access contiguous memory.
    pub fn from_cpu(device: &Arc<CudaDevice>, neurons: &NeuronArrays) -> Result<Self, String> {
        let n = neurons.count;
        let map_err = |e: cudarc::driver::DriverError| format!("CUDA neuron alloc: {}", e);

        // -- Flatten conductance_scale: AoS [f32; 8] --> SoA (8 * N) ---------
        let n_channels = IonChannelType::COUNT; // 8
        let mut cs_flat = vec![0.0f32; n_channels * n];
        for i in 0..n {
            for ch in 0..n_channels {
                cs_flat[ch * n + i] = neurons.conductance_scale[i][ch];
            }
        }

        // -- Flatten nt_conc: AoS [f32; 6] --> SoA (6 * N) -------------------
        let n_nts = crate::types::NTType::COUNT; // 6
        let mut nt_flat = vec![0.0f32; n_nts * n];
        for i in 0..n {
            for nt in 0..n_nts {
                nt_flat[nt * n + i] = neurons.nt_conc[i][nt];
            }
        }

        Ok(Self {
            n: n as u32,

            // Membrane
            voltage: device.htod_copy(neurons.voltage.clone()).map_err(map_err)?,
            prev_voltage: device
                .htod_copy(neurons.prev_voltage.clone())
                .map_err(map_err)?,
            fired: device.htod_copy(neurons.fired.clone()).map_err(map_err)?,
            refractory_timer: device
                .htod_copy(neurons.refractory_timer.clone())
                .map_err(map_err)?,
            spike_count: device
                .htod_copy(neurons.spike_count.clone())
                .map_err(map_err)?,

            // HH gating
            nav_m: device.htod_copy(neurons.nav_m.clone()).map_err(map_err)?,
            nav_h: device.htod_copy(neurons.nav_h.clone()).map_err(map_err)?,
            kv_n: device.htod_copy(neurons.kv_n.clone()).map_err(map_err)?,
            cav_m: device.htod_copy(neurons.cav_m.clone()).map_err(map_err)?,
            cav_h: device.htod_copy(neurons.cav_h.clone()).map_err(map_err)?,

            // Conductance scales (SoA)
            cond_scale: device.htod_copy(cs_flat).map_err(map_err)?,

            // Receptor open fractions
            ampa_open: device
                .htod_copy(neurons.ampa_open.clone())
                .map_err(map_err)?,
            nmda_open: device
                .htod_copy(neurons.nmda_open.clone())
                .map_err(map_err)?,
            gabaa_open: device
                .htod_copy(neurons.gabaa_open.clone())
                .map_err(map_err)?,
            nachr_open: device
                .htod_copy(neurons.nachr_open.clone())
                .map_err(map_err)?,

            // Calcium
            ca_cyto: device
                .htod_copy(neurons.ca_cytoplasmic.clone())
                .map_err(map_err)?,
            ca_er: device.htod_copy(neurons.ca_er.clone()).map_err(map_err)?,
            ca_mito: device
                .htod_copy(neurons.ca_mitochondrial.clone())
                .map_err(map_err)?,
            ca_micro: device
                .htod_copy(neurons.ca_microdomain.clone())
                .map_err(map_err)?,

            // Second messengers
            camp: device.htod_copy(neurons.camp.clone()).map_err(map_err)?,
            gs_active: device
                .htod_copy(neurons.gs_active.clone())
                .map_err(map_err)?,
            gi_active: device
                .htod_copy(neurons.gi_active.clone())
                .map_err(map_err)?,
            gq_active: device
                .htod_copy(neurons.gq_active.clone())
                .map_err(map_err)?,
            pka_activity: device
                .htod_copy(neurons.pka_activity.clone())
                .map_err(map_err)?,
            pkc_activity: device
                .htod_copy(neurons.pkc_activity.clone())
                .map_err(map_err)?,
            camkii_activity: device
                .htod_copy(neurons.camkii_activity.clone())
                .map_err(map_err)?,
            ip3: device.htod_copy(neurons.ip3.clone()).map_err(map_err)?,
            dag: device.htod_copy(neurons.dag.clone()).map_err(map_err)?,
            erk_activity: device
                .htod_copy(neurons.erk_activity.clone())
                .map_err(map_err)?,
            er_ca_released: device
                .htod_copy(neurons.er_ca_released.clone())
                .map_err(map_err)?,
            er_ca_store: device
                .htod_copy(neurons.er_ca_store.clone())
                .map_err(map_err)?,

            // Phosphorylation
            ampa_p: device.htod_copy(neurons.ampa_p.clone()).map_err(map_err)?,
            kv_p: device.htod_copy(neurons.kv_p.clone()).map_err(map_err)?,
            cav_p: device.htod_copy(neurons.cav_p.clone()).map_err(map_err)?,
            creb_p: device.htod_copy(neurons.creb_p.clone()).map_err(map_err)?,

            // Metabolism
            atp: device.htod_copy(neurons.atp.clone()).map_err(map_err)?,
            adp: device.htod_copy(neurons.adp.clone()).map_err(map_err)?,
            glucose: device.htod_copy(neurons.glucose.clone()).map_err(map_err)?,
            oxygen: device.htod_copy(neurons.oxygen.clone()).map_err(map_err)?,
            energy: device.htod_copy(neurons.energy.clone()).map_err(map_err)?,

            // NT concentrations (SoA)
            nt_conc: device.htod_copy(nt_flat).map_err(map_err)?,

            // External current
            ext_current: device
                .htod_copy(neurons.external_current.clone())
                .map_err(map_err)?,

            // Identity
            alive: device.htod_copy(neurons.alive.clone()).map_err(map_err)?,
            archetype: device
                .htod_copy(neurons.archetype.clone())
                .map_err(map_err)?,
        })
    }

    /// Allocate all neuron buffers on the CUDA device initialised to HH
    /// resting state.  Useful for tests or when CPU-side NeuronArrays is not
    /// available.
    pub fn new_resting(device: &Arc<CudaDevice>, n: u32) -> Result<Self, String> {
        let sz = n as usize;
        let map_err = |e: cudarc::driver::DriverError| format!("CUDA neuron alloc: {}", e);

        // HH steady-state gating at V = -65 mV (values from neuron_arrays.rs)
        let v_rest = -65.0f32;
        let nav_m_rest = crate::neuron_arrays::alpha_m(v_rest)
            / (crate::neuron_arrays::alpha_m(v_rest) + crate::neuron_arrays::beta_m(v_rest));
        let nav_h_rest = crate::neuron_arrays::alpha_h(v_rest)
            / (crate::neuron_arrays::alpha_h(v_rest) + crate::neuron_arrays::beta_h(v_rest));
        let kv_n_rest = crate::neuron_arrays::alpha_n(v_rest)
            / (crate::neuron_arrays::alpha_n(v_rest) + crate::neuron_arrays::beta_n(v_rest));
        let cav_m_rest = crate::neuron_arrays::alpha_m_ca(v_rest)
            / (crate::neuron_arrays::alpha_m_ca(v_rest) + crate::neuron_arrays::beta_m_ca(v_rest));
        let cav_h_rest = crate::neuron_arrays::alpha_h_ca(v_rest)
            / (crate::neuron_arrays::alpha_h_ca(v_rest) + crate::neuron_arrays::beta_h_ca(v_rest));

        let n_channels = IonChannelType::COUNT;
        let n_nts = crate::types::NTType::COUNT;

        Ok(Self {
            n,

            voltage: device.htod_copy(vec![v_rest; sz]).map_err(map_err)?,
            prev_voltage: device.htod_copy(vec![v_rest; sz]).map_err(map_err)?,
            fired: device.alloc_zeros(sz).map_err(map_err)?,
            refractory_timer: device.alloc_zeros(sz).map_err(map_err)?,
            spike_count: device.alloc_zeros::<u32>(sz).map_err(map_err)?,

            nav_m: device.htod_copy(vec![nav_m_rest; sz]).map_err(map_err)?,
            nav_h: device.htod_copy(vec![nav_h_rest; sz]).map_err(map_err)?,
            kv_n: device.htod_copy(vec![kv_n_rest; sz]).map_err(map_err)?,
            cav_m: device.htod_copy(vec![cav_m_rest; sz]).map_err(map_err)?,
            cav_h: device.htod_copy(vec![cav_h_rest; sz]).map_err(map_err)?,

            // Default conductance scales = 1.0 for all channels
            cond_scale: device
                .htod_copy(vec![1.0f32; n_channels * sz])
                .map_err(map_err)?,

            ampa_open: device.alloc_zeros(sz).map_err(map_err)?,
            nmda_open: device.alloc_zeros(sz).map_err(map_err)?,
            gabaa_open: device.alloc_zeros(sz).map_err(map_err)?,
            nachr_open: device.alloc_zeros(sz).map_err(map_err)?,

            ca_cyto: device
                .htod_copy(vec![crate::constants::REST_CA_CYTOPLASMIC_NM; sz])
                .map_err(map_err)?,
            ca_er: device
                .htod_copy(vec![crate::constants::REST_CA_ER_NM; sz])
                .map_err(map_err)?,
            ca_mito: device
                .htod_copy(vec![crate::constants::REST_CA_MITOCHONDRIAL_NM; sz])
                .map_err(map_err)?,
            ca_micro: device
                .htod_copy(vec![crate::constants::REST_CA_MICRODOMAIN_NM; sz])
                .map_err(map_err)?,

            camp: device.htod_copy(vec![50.0f32; sz]).map_err(map_err)?,
            gs_active: device.alloc_zeros(sz).map_err(map_err)?,
            gi_active: device.alloc_zeros(sz).map_err(map_err)?,
            gq_active: device.alloc_zeros(sz).map_err(map_err)?,
            pka_activity: device.alloc_zeros(sz).map_err(map_err)?,
            pkc_activity: device.alloc_zeros(sz).map_err(map_err)?,
            camkii_activity: device.alloc_zeros(sz).map_err(map_err)?,
            ip3: device
                .htod_copy(vec![crate::constants::IP3_BASAL; sz])
                .map_err(map_err)?,
            dag: device
                .htod_copy(vec![crate::constants::DAG_BASAL; sz])
                .map_err(map_err)?,
            erk_activity: device.alloc_zeros(sz).map_err(map_err)?,
            er_ca_released: device.alloc_zeros(sz).map_err(map_err)?,
            er_ca_store: device
                .htod_copy(vec![crate::constants::SMS_ER_CA_STORE; sz])
                .map_err(map_err)?,

            ampa_p: device.alloc_zeros(sz).map_err(map_err)?,
            kv_p: device.alloc_zeros(sz).map_err(map_err)?,
            cav_p: device.alloc_zeros(sz).map_err(map_err)?,
            creb_p: device.alloc_zeros(sz).map_err(map_err)?,

            atp: device
                .htod_copy(vec![crate::constants::ATP_RESTING; sz])
                .map_err(map_err)?,
            adp: device
                .htod_copy(vec![crate::constants::ADP_RESTING; sz])
                .map_err(map_err)?,
            glucose: device
                .htod_copy(vec![crate::constants::GLUCOSE_RESTING; sz])
                .map_err(map_err)?,
            oxygen: device
                .htod_copy(vec![crate::constants::OXYGEN_RESTING; sz])
                .map_err(map_err)?,
            energy: device.htod_copy(vec![100.0f32; sz]).map_err(map_err)?,

            // Resting NT concentrations in SoA layout
            nt_conc: {
                let mut flat = vec![0.0f32; n_nts * sz];
                let resting = [
                    crate::types::NTType::Dopamine.resting_conc_nm(),
                    crate::types::NTType::Serotonin.resting_conc_nm(),
                    crate::types::NTType::Norepinephrine.resting_conc_nm(),
                    crate::types::NTType::Acetylcholine.resting_conc_nm(),
                    crate::types::NTType::GABA.resting_conc_nm(),
                    crate::types::NTType::Glutamate.resting_conc_nm(),
                ];
                for nt in 0..n_nts {
                    for i in 0..sz {
                        flat[nt * sz + i] = resting[nt];
                    }
                }
                device.htod_copy(flat).map_err(map_err)?
            },

            ext_current: device.alloc_zeros(sz).map_err(map_err)?,
            alive: device.htod_copy(vec![1u8; sz]).map_err(map_err)?,
            archetype: device.alloc_zeros(sz).map_err(map_err)?,
        })
    }

    // -----------------------------------------------------------------------
    // D2H downloads (only used for result collection, not in the hot loop)
    // -----------------------------------------------------------------------

    /// Download fired flags from GPU to CPU.
    pub fn download_fired(&self, device: &Arc<CudaDevice>) -> Result<Vec<u8>, String> {
        device
            .dtoh_sync_copy(&self.fired)
            .map_err(|e| format!("D2H fired: {}", e))
    }

    /// Download voltages from GPU to CPU.
    pub fn download_voltages(&self, device: &Arc<CudaDevice>) -> Result<Vec<f32>, String> {
        device
            .dtoh_sync_copy(&self.voltage)
            .map_err(|e| format!("D2H voltage: {}", e))
    }

    /// Download spike counts from GPU to CPU.
    pub fn download_spike_counts(&self, device: &Arc<CudaDevice>) -> Result<Vec<u32>, String> {
        device
            .dtoh_sync_copy(&self.spike_count)
            .map_err(|e| format!("D2H spike_count: {}", e))
    }

    /// Download calcium (cytoplasmic) from GPU to CPU.
    pub fn download_ca_cyto(&self, device: &Arc<CudaDevice>) -> Result<Vec<f32>, String> {
        device
            .dtoh_sync_copy(&self.ca_cyto)
            .map_err(|e| format!("D2H ca_cyto: {}", e))
    }

    /// Download NT concentrations from GPU to CPU (flattened SoA, 6 * N).
    pub fn download_nt_conc(&self, device: &Arc<CudaDevice>) -> Result<Vec<f32>, String> {
        device
            .dtoh_sync_copy(&self.nt_conc)
            .map_err(|e| format!("D2H nt_conc: {}", e))
    }

    /// Download NT concentrations and reshape into per-neuron `[f32; 6]` arrays.
    pub fn download_nt_conc_aos(&self, device: &Arc<CudaDevice>) -> Result<Vec<[f32; 6]>, String> {
        let flat = self.download_nt_conc(device)?;
        let n = self.n as usize;
        let n_nts = crate::types::NTType::COUNT;
        let mut result = vec![[0.0f32; 6]; n];
        for i in 0..n {
            for nt in 0..n_nts {
                result[i][nt] = flat[nt * n + i];
            }
        }
        Ok(result)
    }

    /// Download conductance scales and reshape into per-neuron `[f32; 8]` arrays.
    pub fn download_cond_scale_aos(
        &self,
        device: &Arc<CudaDevice>,
    ) -> Result<Vec<[f32; 8]>, String> {
        let flat: Vec<f32> = device
            .dtoh_sync_copy(&self.cond_scale)
            .map_err(|e| format!("D2H cond_scale: {}", e))?;
        let n = self.n as usize;
        let n_ch = IonChannelType::COUNT;
        let mut result = vec![[0.0f32; 8]; n];
        for i in 0..n {
            for ch in 0..n_ch {
                result[i][ch] = flat[ch * n + i];
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // H2D uploads (for injecting external current, stimulation, etc.)
    // -----------------------------------------------------------------------

    /// Upload external currents from CPU to GPU (overwrites the device buffer).
    pub fn upload_ext_current(
        &mut self,
        device: &Arc<CudaDevice>,
        currents: &[f32],
    ) -> Result<(), String> {
        assert_eq!(
            currents.len(),
            self.n as usize,
            "ext_current length mismatch"
        );
        self.ext_current = device
            .htod_copy(currents.to_vec())
            .map_err(|e| format!("H2D ext_current: {}", e))?;
        Ok(())
    }

    /// Zero the external current buffer by allocating a fresh zeroed buffer.
    ///
    /// Called at the start of each simulation step before sensory encoding and
    /// FEP stimulation re-inject current.
    pub fn clear_ext_current(&mut self, device: &Arc<CudaDevice>) -> Result<(), String> {
        self.ext_current = device
            .alloc_zeros(self.n as usize)
            .map_err(|e| format!("Clear ext_current: {}", e))?;
        Ok(())
    }

    /// Upload conductance scales from CPU (AoS `[f32; 8]` per neuron) to GPU
    /// (SoA flattened).
    pub fn upload_cond_scale(
        &mut self,
        device: &Arc<CudaDevice>,
        scales: &[[f32; 8]],
    ) -> Result<(), String> {
        let n = self.n as usize;
        assert_eq!(scales.len(), n, "cond_scale length mismatch");
        let n_ch = IonChannelType::COUNT;
        let mut flat = vec![0.0f32; n_ch * n];
        for i in 0..n {
            for ch in 0..n_ch {
                flat[ch * n + i] = scales[i][ch];
            }
        }
        self.cond_scale = device
            .htod_copy(flat)
            .map_err(|e| format!("H2D cond_scale: {}", e))?;
        Ok(())
    }

    /// Write the complete neuron state back to a CPU-side [`NeuronArrays`].
    ///
    /// This is the inverse of [`from_cpu`] and is used when the simulation
    /// needs to hand off to CPU-only subsystems (gene expression, metabolism).
    pub fn write_back_to_cpu(
        &self,
        device: &Arc<CudaDevice>,
        neurons: &mut NeuronArrays,
    ) -> Result<(), String> {
        let n = self.n as usize;
        assert_eq!(neurons.count, n, "neuron count mismatch on write-back");

        let map_err =
            |field: &str| move |e: cudarc::driver::DriverError| format!("D2H {}: {}", field, e);

        neurons.voltage = device
            .dtoh_sync_copy(&self.voltage)
            .map_err(map_err("voltage"))?;
        neurons.prev_voltage = device
            .dtoh_sync_copy(&self.prev_voltage)
            .map_err(map_err("prev_voltage"))?;
        neurons.fired = device
            .dtoh_sync_copy(&self.fired)
            .map_err(map_err("fired"))?;
        neurons.refractory_timer = device
            .dtoh_sync_copy(&self.refractory_timer)
            .map_err(map_err("refractory_timer"))?;
        neurons.spike_count = device
            .dtoh_sync_copy(&self.spike_count)
            .map_err(map_err("spike_count"))?;

        neurons.nav_m = device
            .dtoh_sync_copy(&self.nav_m)
            .map_err(map_err("nav_m"))?;
        neurons.nav_h = device
            .dtoh_sync_copy(&self.nav_h)
            .map_err(map_err("nav_h"))?;
        neurons.kv_n = device.dtoh_sync_copy(&self.kv_n).map_err(map_err("kv_n"))?;
        neurons.cav_m = device
            .dtoh_sync_copy(&self.cav_m)
            .map_err(map_err("cav_m"))?;
        neurons.cav_h = device
            .dtoh_sync_copy(&self.cav_h)
            .map_err(map_err("cav_h"))?;

        neurons.ampa_open = device
            .dtoh_sync_copy(&self.ampa_open)
            .map_err(map_err("ampa_open"))?;
        neurons.nmda_open = device
            .dtoh_sync_copy(&self.nmda_open)
            .map_err(map_err("nmda_open"))?;
        neurons.gabaa_open = device
            .dtoh_sync_copy(&self.gabaa_open)
            .map_err(map_err("gabaa_open"))?;
        neurons.nachr_open = device
            .dtoh_sync_copy(&self.nachr_open)
            .map_err(map_err("nachr_open"))?;

        neurons.ca_cytoplasmic = device
            .dtoh_sync_copy(&self.ca_cyto)
            .map_err(map_err("ca_cyto"))?;
        neurons.ca_er = device
            .dtoh_sync_copy(&self.ca_er)
            .map_err(map_err("ca_er"))?;
        neurons.ca_mitochondrial = device
            .dtoh_sync_copy(&self.ca_mito)
            .map_err(map_err("ca_mito"))?;
        neurons.ca_microdomain = device
            .dtoh_sync_copy(&self.ca_micro)
            .map_err(map_err("ca_micro"))?;

        neurons.camp = device.dtoh_sync_copy(&self.camp).map_err(map_err("camp"))?;
        neurons.gs_active = device
            .dtoh_sync_copy(&self.gs_active)
            .map_err(map_err("gs_active"))?;
        neurons.gi_active = device
            .dtoh_sync_copy(&self.gi_active)
            .map_err(map_err("gi_active"))?;
        neurons.gq_active = device
            .dtoh_sync_copy(&self.gq_active)
            .map_err(map_err("gq_active"))?;
        neurons.pka_activity = device
            .dtoh_sync_copy(&self.pka_activity)
            .map_err(map_err("pka_activity"))?;
        neurons.pkc_activity = device
            .dtoh_sync_copy(&self.pkc_activity)
            .map_err(map_err("pkc_activity"))?;
        neurons.camkii_activity = device
            .dtoh_sync_copy(&self.camkii_activity)
            .map_err(map_err("camkii_activity"))?;
        neurons.ip3 = device.dtoh_sync_copy(&self.ip3).map_err(map_err("ip3"))?;
        neurons.dag = device.dtoh_sync_copy(&self.dag).map_err(map_err("dag"))?;
        neurons.erk_activity = device
            .dtoh_sync_copy(&self.erk_activity)
            .map_err(map_err("erk_activity"))?;
        neurons.er_ca_released = device
            .dtoh_sync_copy(&self.er_ca_released)
            .map_err(map_err("er_ca_released"))?;
        neurons.er_ca_store = device
            .dtoh_sync_copy(&self.er_ca_store)
            .map_err(map_err("er_ca_store"))?;

        neurons.ampa_p = device
            .dtoh_sync_copy(&self.ampa_p)
            .map_err(map_err("ampa_p"))?;
        neurons.kv_p = device.dtoh_sync_copy(&self.kv_p).map_err(map_err("kv_p"))?;
        neurons.cav_p = device
            .dtoh_sync_copy(&self.cav_p)
            .map_err(map_err("cav_p"))?;
        neurons.creb_p = device
            .dtoh_sync_copy(&self.creb_p)
            .map_err(map_err("creb_p"))?;

        neurons.atp = device.dtoh_sync_copy(&self.atp).map_err(map_err("atp"))?;
        neurons.adp = device.dtoh_sync_copy(&self.adp).map_err(map_err("adp"))?;
        neurons.glucose = device
            .dtoh_sync_copy(&self.glucose)
            .map_err(map_err("glucose"))?;
        neurons.oxygen = device
            .dtoh_sync_copy(&self.oxygen)
            .map_err(map_err("oxygen"))?;
        neurons.energy = device
            .dtoh_sync_copy(&self.energy)
            .map_err(map_err("energy"))?;

        neurons.external_current = device
            .dtoh_sync_copy(&self.ext_current)
            .map_err(map_err("ext_current"))?;
        neurons.alive = device
            .dtoh_sync_copy(&self.alive)
            .map_err(map_err("alive"))?;
        neurons.archetype = device
            .dtoh_sync_copy(&self.archetype)
            .map_err(map_err("archetype"))?;

        // Unflatten SoA conductance_scale back to AoS
        let cs_flat: Vec<f32> = device
            .dtoh_sync_copy(&self.cond_scale)
            .map_err(map_err("cond_scale"))?;
        let n_ch = IonChannelType::COUNT;
        for i in 0..n {
            for ch in 0..n_ch {
                neurons.conductance_scale[i][ch] = cs_flat[ch * n + i];
            }
        }

        // Unflatten SoA nt_conc back to AoS
        let nt_flat: Vec<f32> = device
            .dtoh_sync_copy(&self.nt_conc)
            .map_err(map_err("nt_conc"))?;
        let n_nts = crate::types::NTType::COUNT;
        for i in 0..n {
            for nt_idx in 0..n_nts {
                neurons.nt_conc[i][nt_idx] = nt_flat[nt_idx * n + i];
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CudaSynapseState
// ---------------------------------------------------------------------------

/// GPU-resident synapse state in CSR format.
///
/// The CSR row_offsets and col_indices are immutable after construction (the
/// topology does not change during the simulation).  Per-synapse weights and
/// STDP traces are updated by CUDA kernels each step.
#[cfg(feature = "cuda")]
pub struct CudaSynapseState {
    /// Number of neurons (rows in the CSR matrix).
    pub n_neurons: u32,
    /// Total synapse count (nnz in the CSR matrix).
    pub n_synapses: u32,

    // ===== CSR Structure (read-only after init) =====
    pub row_offsets: CudaSlice<u32>,
    pub col_indices: CudaSlice<u32>,

    // ===== Synapse Identity =====
    pub nt_type: CudaSlice<u8>,

    // ===== Synaptic Strength =====
    pub weight: CudaSlice<f32>,
    pub strength: CudaSlice<f32>,

    // ===== Vesicle Pool =====
    pub vesicle_rrp: CudaSlice<f32>,

    // ===== Synaptic Cleft =====
    pub cleft_concentration: CudaSlice<f32>,

    // ===== STDP Traces =====
    pub pre_trace: CudaSlice<f32>,
    pub post_trace: CudaSlice<f32>,
    pub eligibility_trace: CudaSlice<f32>,

    // ===== Homeostatic Scaling =====
    pub homeostatic_scale: CudaSlice<f32>,
}

#[cfg(feature = "cuda")]
impl CudaSynapseState {
    /// Upload a complete [`SynapseArrays`] from CPU to persistent GPU buffers.
    pub fn from_cpu(device: &Arc<CudaDevice>, synapses: &SynapseArrays) -> Result<Self, String> {
        let map_err = |e: cudarc::driver::DriverError| format!("CUDA synapse alloc: {}", e);
        let s = synapses.n_synapses;

        // STDP traces initialised to zero on GPU (CPU stores last-spike times
        // but the GPU kernels use exponential trace variables instead).
        let zero_traces = vec![0.0f32; s];

        Ok(Self {
            n_neurons: synapses.n_neurons as u32,
            n_synapses: s as u32,

            row_offsets: device
                .htod_copy(synapses.row_offsets.clone())
                .map_err(map_err)?,
            col_indices: device
                .htod_copy(synapses.col_indices.clone())
                .map_err(map_err)?,
            nt_type: device
                .htod_copy(synapses.nt_type.clone())
                .map_err(map_err)?,
            weight: device.htod_copy(synapses.weight.clone()).map_err(map_err)?,
            strength: device
                .htod_copy(synapses.strength.clone())
                .map_err(map_err)?,
            vesicle_rrp: device
                .htod_copy(synapses.vesicle_rrp.clone())
                .map_err(map_err)?,
            cleft_concentration: device
                .htod_copy(synapses.cleft_concentration.clone())
                .map_err(map_err)?,
            pre_trace: device.htod_copy(zero_traces.clone()).map_err(map_err)?,
            post_trace: device.htod_copy(zero_traces.clone()).map_err(map_err)?,
            eligibility_trace: device
                .htod_copy(synapses.eligibility_trace.clone())
                .map_err(map_err)?,
            homeostatic_scale: device
                .htod_copy(synapses.homeostatic_scale.clone())
                .map_err(map_err)?,
        })
    }

    // -----------------------------------------------------------------------
    // D2H downloads
    // -----------------------------------------------------------------------

    /// Download weights from GPU to CPU.
    pub fn download_weights(&self, device: &Arc<CudaDevice>) -> Result<Vec<f32>, String> {
        device
            .dtoh_sync_copy(&self.weight)
            .map_err(|e| format!("D2H weight: {}", e))
    }

    /// Download cleft concentrations from GPU to CPU.
    pub fn download_cleft(&self, device: &Arc<CudaDevice>) -> Result<Vec<f32>, String> {
        device
            .dtoh_sync_copy(&self.cleft_concentration)
            .map_err(|e| format!("D2H cleft_concentration: {}", e))
    }

    /// Download STDP pre-traces from GPU to CPU.
    pub fn download_pre_trace(&self, device: &Arc<CudaDevice>) -> Result<Vec<f32>, String> {
        device
            .dtoh_sync_copy(&self.pre_trace)
            .map_err(|e| format!("D2H pre_trace: {}", e))
    }

    /// Download STDP post-traces from GPU to CPU.
    pub fn download_post_trace(&self, device: &Arc<CudaDevice>) -> Result<Vec<f32>, String> {
        device
            .dtoh_sync_copy(&self.post_trace)
            .map_err(|e| format!("D2H post_trace: {}", e))
    }

    /// Write the synapse state back to a CPU-side [`SynapseArrays`].
    ///
    /// Only mutable per-synapse fields are written back; the CSR structure
    /// (row_offsets, col_indices) is assumed unchanged.
    pub fn write_back_to_cpu(
        &self,
        device: &Arc<CudaDevice>,
        synapses: &mut SynapseArrays,
    ) -> Result<(), String> {
        assert_eq!(
            synapses.n_synapses, self.n_synapses as usize,
            "synapse count mismatch on write-back"
        );

        let map_err =
            |field: &str| move |e: cudarc::driver::DriverError| format!("D2H {}: {}", field, e);

        synapses.weight = device
            .dtoh_sync_copy(&self.weight)
            .map_err(map_err("weight"))?;
        synapses.strength = device
            .dtoh_sync_copy(&self.strength)
            .map_err(map_err("strength"))?;
        synapses.vesicle_rrp = device
            .dtoh_sync_copy(&self.vesicle_rrp)
            .map_err(map_err("vesicle_rrp"))?;
        synapses.cleft_concentration = device
            .dtoh_sync_copy(&self.cleft_concentration)
            .map_err(map_err("cleft_concentration"))?;
        synapses.eligibility_trace = device
            .dtoh_sync_copy(&self.eligibility_trace)
            .map_err(map_err("eligibility_trace"))?;
        synapses.homeostatic_scale = device
            .dtoh_sync_copy(&self.homeostatic_scale)
            .map_err(map_err("homeostatic_scale"))?;

        Ok(())
    }
}
