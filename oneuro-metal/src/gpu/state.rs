//! Persistent Metal-resident simulation state.
//!
//! This mirrors the CUDA resident-state pattern: long-lived shared-memory
//! buffers are allocated once and reused across many simulation steps.

#[cfg(target_os = "macos")]
use metal::Buffer;

#[cfg(target_os = "macos")]
use crate::glia::GliaState;
#[cfg(target_os = "macos")]
use crate::gpu::GpuContext;
#[cfg(target_os = "macos")]
use crate::neuron_arrays::NeuronArrays;
#[cfg(target_os = "macos")]
use crate::synapse_arrays::SynapseArrays;
#[cfg(target_os = "macos")]
use crate::types::{IonChannelType, NTType};

#[cfg(target_os = "macos")]
unsafe fn copy_slice_to_buffer<T: Copy>(buffer: &Buffer, data: &[T]) {
    let ptr = buffer.contents() as *mut T;
    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
}

#[cfg(target_os = "macos")]
unsafe fn copy_buffer_to_slice<T: Copy>(buffer: &Buffer, out: &mut [T]) {
    let ptr = buffer.contents() as *const T;
    std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len());
}

#[cfg(target_os = "macos")]
unsafe fn fill_buffer_with<T: Copy>(buffer: &Buffer, len: usize, value: T) {
    let ptr = buffer.contents() as *mut T;
    for i in 0..len {
        *ptr.add(i) = value;
    }
}

/// Persistent neuron-side Metal buffers.
#[cfg(target_os = "macos")]
pub struct MetalNeuronState {
    pub count: usize,
    pub voltage: Buffer,
    pub prev_voltage: Buffer,
    pub fired: Buffer,
    pub refractory_timer: Buffer,
    pub spike_count: Buffer,
    pub nav_m: Buffer,
    pub nav_h: Buffer,
    pub kv_n: Buffer,
    pub cav_m: Buffer,
    pub cav_h: Buffer,
    pub conductance_scale: Buffer,
    pub ampa_open: Buffer,
    pub nmda_open: Buffer,
    pub gabaa_open: Buffer,
    pub nachr_open: Buffer,
    pub ca_cytoplasmic: Buffer,
    pub ca_er: Buffer,
    pub ca_mitochondrial: Buffer,
    pub ca_microdomain: Buffer,
    pub camp: Buffer,
    pub gs_active: Buffer,
    pub gi_active: Buffer,
    pub gq_active: Buffer,
    pub pka_activity: Buffer,
    pub pkc_activity: Buffer,
    pub camkii_activity: Buffer,
    pub ip3: Buffer,
    pub dag: Buffer,
    pub erk_activity: Buffer,
    pub er_ca_released: Buffer,
    pub er_ca_store: Buffer,
    pub ampa_p: Buffer,
    pub kv_p: Buffer,
    pub cav_p: Buffer,
    pub creb_p: Buffer,
    pub atp: Buffer,
    pub adp: Buffer,
    pub glucose: Buffer,
    pub oxygen: Buffer,
    pub energy: Buffer,
    pub nt_conc: Buffer,
    pub external_current: Buffer,
    pub synaptic_current: Buffer,
    pub synaptic_current_accum_i32: Buffer,
    pub alive: Buffer,
    pub archetype: Buffer,
    pub bdnf_level: Buffer,
    pub cfos_level: Buffer,
    pub arc_level: Buffer,
    pub mt_coherence: Buffer,
    pub orch_or_events: Buffer,
    pub last_fired_step: Buffer,
    pub excitability_bias: Buffer,
}

#[cfg(target_os = "macos")]
impl MetalNeuronState {
    pub fn from_cpu(gpu: &GpuContext, neurons: &NeuronArrays) -> Self {
        Self {
            count: neurons.count,
            voltage: gpu.buffer_from_slice(&neurons.voltage),
            prev_voltage: gpu.buffer_from_slice(&neurons.prev_voltage),
            fired: gpu.buffer_from_slice(&neurons.fired),
            refractory_timer: gpu.buffer_from_slice(&neurons.refractory_timer),
            spike_count: gpu.buffer_from_slice(&neurons.spike_count),
            nav_m: gpu.buffer_from_slice(&neurons.nav_m),
            nav_h: gpu.buffer_from_slice(&neurons.nav_h),
            kv_n: gpu.buffer_from_slice(&neurons.kv_n),
            cav_m: gpu.buffer_from_slice(&neurons.cav_m),
            cav_h: gpu.buffer_from_slice(&neurons.cav_h),
            conductance_scale: gpu.buffer_from_slice(&neurons.conductance_scale),
            ampa_open: gpu.buffer_from_slice(&neurons.ampa_open),
            nmda_open: gpu.buffer_from_slice(&neurons.nmda_open),
            gabaa_open: gpu.buffer_from_slice(&neurons.gabaa_open),
            nachr_open: gpu.buffer_from_slice(&neurons.nachr_open),
            ca_cytoplasmic: gpu.buffer_from_slice(&neurons.ca_cytoplasmic),
            ca_er: gpu.buffer_from_slice(&neurons.ca_er),
            ca_mitochondrial: gpu.buffer_from_slice(&neurons.ca_mitochondrial),
            ca_microdomain: gpu.buffer_from_slice(&neurons.ca_microdomain),
            camp: gpu.buffer_from_slice(&neurons.camp),
            gs_active: gpu.buffer_from_slice(&neurons.gs_active),
            gi_active: gpu.buffer_from_slice(&neurons.gi_active),
            gq_active: gpu.buffer_from_slice(&neurons.gq_active),
            pka_activity: gpu.buffer_from_slice(&neurons.pka_activity),
            pkc_activity: gpu.buffer_from_slice(&neurons.pkc_activity),
            camkii_activity: gpu.buffer_from_slice(&neurons.camkii_activity),
            ip3: gpu.buffer_from_slice(&neurons.ip3),
            dag: gpu.buffer_from_slice(&neurons.dag),
            erk_activity: gpu.buffer_from_slice(&neurons.erk_activity),
            er_ca_released: gpu.buffer_from_slice(&neurons.er_ca_released),
            er_ca_store: gpu.buffer_from_slice(&neurons.er_ca_store),
            ampa_p: gpu.buffer_from_slice(&neurons.ampa_p),
            kv_p: gpu.buffer_from_slice(&neurons.kv_p),
            cav_p: gpu.buffer_from_slice(&neurons.cav_p),
            creb_p: gpu.buffer_from_slice(&neurons.creb_p),
            atp: gpu.buffer_from_slice(&neurons.atp),
            adp: gpu.buffer_from_slice(&neurons.adp),
            glucose: gpu.buffer_from_slice(&neurons.glucose),
            oxygen: gpu.buffer_from_slice(&neurons.oxygen),
            energy: gpu.buffer_from_slice(&neurons.energy),
            nt_conc: gpu.buffer_from_slice(&neurons.nt_conc),
            external_current: gpu.buffer_from_slice(&neurons.external_current),
            synaptic_current: gpu.buffer_from_slice(&neurons.synaptic_current),
            synaptic_current_accum_i32: {
                let buf = gpu.buffer_of_size((neurons.count * std::mem::size_of::<i32>()) as u64);
                unsafe {
                    fill_buffer_with(&buf, neurons.count, 0i32);
                }
                buf
            },
            alive: gpu.buffer_from_slice(&neurons.alive),
            archetype: gpu.buffer_from_slice(&neurons.archetype),
            bdnf_level: gpu.buffer_from_slice(&neurons.bdnf_level),
            cfos_level: gpu.buffer_from_slice(&neurons.cfos_level),
            arc_level: gpu.buffer_from_slice(&neurons.arc_level),
            mt_coherence: gpu.buffer_from_slice(&neurons.mt_coherence),
            orch_or_events: gpu.buffer_from_slice(&neurons.orch_or_events),
            last_fired_step: gpu.buffer_from_slice(&neurons.last_fired_step),
            excitability_bias: gpu.buffer_from_slice(&neurons.excitability_bias),
        }
    }

    pub fn upload_all_from_cpu(&self, neurons: &NeuronArrays) {
        unsafe {
            copy_slice_to_buffer(&self.voltage, &neurons.voltage);
            copy_slice_to_buffer(&self.prev_voltage, &neurons.prev_voltage);
            copy_slice_to_buffer(&self.fired, &neurons.fired);
            copy_slice_to_buffer(&self.refractory_timer, &neurons.refractory_timer);
            copy_slice_to_buffer(&self.spike_count, &neurons.spike_count);
            copy_slice_to_buffer(&self.nav_m, &neurons.nav_m);
            copy_slice_to_buffer(&self.nav_h, &neurons.nav_h);
            copy_slice_to_buffer(&self.kv_n, &neurons.kv_n);
            copy_slice_to_buffer(&self.cav_m, &neurons.cav_m);
            copy_slice_to_buffer(&self.cav_h, &neurons.cav_h);
            copy_slice_to_buffer(&self.conductance_scale, &neurons.conductance_scale);
            copy_slice_to_buffer(&self.ampa_open, &neurons.ampa_open);
            copy_slice_to_buffer(&self.nmda_open, &neurons.nmda_open);
            copy_slice_to_buffer(&self.gabaa_open, &neurons.gabaa_open);
            copy_slice_to_buffer(&self.nachr_open, &neurons.nachr_open);
            copy_slice_to_buffer(&self.ca_cytoplasmic, &neurons.ca_cytoplasmic);
            copy_slice_to_buffer(&self.ca_er, &neurons.ca_er);
            copy_slice_to_buffer(&self.ca_mitochondrial, &neurons.ca_mitochondrial);
            copy_slice_to_buffer(&self.ca_microdomain, &neurons.ca_microdomain);
            copy_slice_to_buffer(&self.camp, &neurons.camp);
            copy_slice_to_buffer(&self.gs_active, &neurons.gs_active);
            copy_slice_to_buffer(&self.gi_active, &neurons.gi_active);
            copy_slice_to_buffer(&self.gq_active, &neurons.gq_active);
            copy_slice_to_buffer(&self.pka_activity, &neurons.pka_activity);
            copy_slice_to_buffer(&self.pkc_activity, &neurons.pkc_activity);
            copy_slice_to_buffer(&self.camkii_activity, &neurons.camkii_activity);
            copy_slice_to_buffer(&self.ip3, &neurons.ip3);
            copy_slice_to_buffer(&self.dag, &neurons.dag);
            copy_slice_to_buffer(&self.erk_activity, &neurons.erk_activity);
            copy_slice_to_buffer(&self.er_ca_released, &neurons.er_ca_released);
            copy_slice_to_buffer(&self.er_ca_store, &neurons.er_ca_store);
            copy_slice_to_buffer(&self.ampa_p, &neurons.ampa_p);
            copy_slice_to_buffer(&self.kv_p, &neurons.kv_p);
            copy_slice_to_buffer(&self.cav_p, &neurons.cav_p);
            copy_slice_to_buffer(&self.creb_p, &neurons.creb_p);
            copy_slice_to_buffer(&self.atp, &neurons.atp);
            copy_slice_to_buffer(&self.adp, &neurons.adp);
            copy_slice_to_buffer(&self.glucose, &neurons.glucose);
            copy_slice_to_buffer(&self.oxygen, &neurons.oxygen);
            copy_slice_to_buffer(&self.energy, &neurons.energy);
            copy_slice_to_buffer(&self.nt_conc, &neurons.nt_conc);
            copy_slice_to_buffer(&self.external_current, &neurons.external_current);
            copy_slice_to_buffer(&self.synaptic_current, &neurons.synaptic_current);
            copy_slice_to_buffer(&self.alive, &neurons.alive);
            copy_slice_to_buffer(&self.archetype, &neurons.archetype);
            copy_slice_to_buffer(&self.bdnf_level, &neurons.bdnf_level);
            copy_slice_to_buffer(&self.cfos_level, &neurons.cfos_level);
            copy_slice_to_buffer(&self.arc_level, &neurons.arc_level);
            copy_slice_to_buffer(&self.mt_coherence, &neurons.mt_coherence);
            copy_slice_to_buffer(&self.orch_or_events, &neurons.orch_or_events);
            copy_slice_to_buffer(&self.last_fired_step, &neurons.last_fired_step);
            copy_slice_to_buffer(&self.excitability_bias, &neurons.excitability_bias);
        }
    }

    pub fn write_back_to_cpu(&self, neurons: &mut NeuronArrays) {
        unsafe {
            copy_buffer_to_slice(&self.voltage, &mut neurons.voltage);
            copy_buffer_to_slice(&self.prev_voltage, &mut neurons.prev_voltage);
            copy_buffer_to_slice(&self.fired, &mut neurons.fired);
            copy_buffer_to_slice(&self.refractory_timer, &mut neurons.refractory_timer);
            copy_buffer_to_slice(&self.spike_count, &mut neurons.spike_count);
            copy_buffer_to_slice(&self.nav_m, &mut neurons.nav_m);
            copy_buffer_to_slice(&self.nav_h, &mut neurons.nav_h);
            copy_buffer_to_slice(&self.kv_n, &mut neurons.kv_n);
            copy_buffer_to_slice(&self.cav_m, &mut neurons.cav_m);
            copy_buffer_to_slice(&self.cav_h, &mut neurons.cav_h);
            copy_buffer_to_slice(&self.conductance_scale, &mut neurons.conductance_scale);
            copy_buffer_to_slice(&self.ampa_open, &mut neurons.ampa_open);
            copy_buffer_to_slice(&self.nmda_open, &mut neurons.nmda_open);
            copy_buffer_to_slice(&self.gabaa_open, &mut neurons.gabaa_open);
            copy_buffer_to_slice(&self.nachr_open, &mut neurons.nachr_open);
            copy_buffer_to_slice(&self.ca_cytoplasmic, &mut neurons.ca_cytoplasmic);
            copy_buffer_to_slice(&self.ca_er, &mut neurons.ca_er);
            copy_buffer_to_slice(&self.ca_mitochondrial, &mut neurons.ca_mitochondrial);
            copy_buffer_to_slice(&self.ca_microdomain, &mut neurons.ca_microdomain);
            copy_buffer_to_slice(&self.camp, &mut neurons.camp);
            copy_buffer_to_slice(&self.gs_active, &mut neurons.gs_active);
            copy_buffer_to_slice(&self.gi_active, &mut neurons.gi_active);
            copy_buffer_to_slice(&self.gq_active, &mut neurons.gq_active);
            copy_buffer_to_slice(&self.pka_activity, &mut neurons.pka_activity);
            copy_buffer_to_slice(&self.pkc_activity, &mut neurons.pkc_activity);
            copy_buffer_to_slice(&self.camkii_activity, &mut neurons.camkii_activity);
            copy_buffer_to_slice(&self.ip3, &mut neurons.ip3);
            copy_buffer_to_slice(&self.dag, &mut neurons.dag);
            copy_buffer_to_slice(&self.erk_activity, &mut neurons.erk_activity);
            copy_buffer_to_slice(&self.er_ca_released, &mut neurons.er_ca_released);
            copy_buffer_to_slice(&self.er_ca_store, &mut neurons.er_ca_store);
            copy_buffer_to_slice(&self.ampa_p, &mut neurons.ampa_p);
            copy_buffer_to_slice(&self.kv_p, &mut neurons.kv_p);
            copy_buffer_to_slice(&self.cav_p, &mut neurons.cav_p);
            copy_buffer_to_slice(&self.creb_p, &mut neurons.creb_p);
            copy_buffer_to_slice(&self.atp, &mut neurons.atp);
            copy_buffer_to_slice(&self.adp, &mut neurons.adp);
            copy_buffer_to_slice(&self.glucose, &mut neurons.glucose);
            copy_buffer_to_slice(&self.oxygen, &mut neurons.oxygen);
            copy_buffer_to_slice(&self.energy, &mut neurons.energy);
            copy_buffer_to_slice(&self.nt_conc, &mut neurons.nt_conc);
            copy_buffer_to_slice(&self.external_current, &mut neurons.external_current);
            copy_buffer_to_slice(&self.synaptic_current, &mut neurons.synaptic_current);
            copy_buffer_to_slice(&self.alive, &mut neurons.alive);
            copy_buffer_to_slice(&self.archetype, &mut neurons.archetype);
            copy_buffer_to_slice(&self.bdnf_level, &mut neurons.bdnf_level);
            copy_buffer_to_slice(&self.cfos_level, &mut neurons.cfos_level);
            copy_buffer_to_slice(&self.arc_level, &mut neurons.arc_level);
            copy_buffer_to_slice(&self.mt_coherence, &mut neurons.mt_coherence);
            copy_buffer_to_slice(&self.orch_or_events, &mut neurons.orch_or_events);
            copy_buffer_to_slice(&self.last_fired_step, &mut neurons.last_fired_step);
            copy_buffer_to_slice(&self.excitability_bias, &mut neurons.excitability_bias);
        }
    }

    pub fn add_external_current(&self, idx: usize, delta: f32) {
        unsafe {
            let ptr = self.external_current.contents() as *mut f32;
            *ptr.add(idx) += delta;
        }
    }

    pub fn add_external_current_many(&self, indices: &[usize], delta: f32) {
        unsafe {
            let ptr = self.external_current.contents() as *mut f32;
            for &idx in indices {
                if idx < self.count {
                    *ptr.add(idx) += delta;
                }
            }
        }
    }

    pub fn add_external_current_weighted(&self, indices: &[usize], deltas: &[f32]) {
        unsafe {
            let ptr = self.external_current.contents() as *mut f32;
            for (&idx, &delta) in indices.iter().zip(deltas.iter()) {
                if idx < self.count {
                    *ptr.add(idx) += delta;
                }
            }
        }
    }

    pub fn set_conductance_scale(&self, idx: usize, channel: usize, value: f32) {
        unsafe {
            let ptr = self.conductance_scale.contents() as *mut [f32; IonChannelType::COUNT];
            (*ptr.add(idx))[channel] = value;
        }
    }

    pub fn set_nt_concentration(&self, idx: usize, nt: usize, value: f32) {
        unsafe {
            let ptr = self.nt_conc.contents() as *mut [f32; NTType::COUNT];
            (*ptr.add(idx))[nt] = value;
        }
    }

    pub fn add_nt_concentration(&self, idx: usize, nt: usize, delta: f32) {
        unsafe {
            let ptr = self.nt_conc.contents() as *mut [f32; NTType::COUNT];
            (*ptr.add(idx))[nt] += delta;
        }
    }

    pub fn clear_external_current(&self) {
        unsafe {
            fill_buffer_with(&self.external_current, self.count, 0.0f32);
        }
    }

    pub fn clear_spike_count(&self) {
        unsafe {
            fill_buffer_with(&self.spike_count, self.count, 0u32);
        }
    }

    pub fn clear_synaptic_current_accum_i32(&self) {
        unsafe {
            fill_buffer_with(&self.synaptic_current_accum_i32, self.count, 0i32);
        }
    }

    pub fn sum_fired_indices(&self, indices: &[usize]) -> u32 {
        unsafe {
            let ptr = self.fired.contents() as *const u8;
            indices
                .iter()
                .filter(|&&idx| idx < self.count)
                .map(|&idx| *ptr.add(idx) as u32)
                .sum()
        }
    }

    pub fn sum_spike_count_indices(&self, indices: &[usize]) -> u64 {
        unsafe {
            let ptr = self.spike_count.contents() as *const u32;
            indices
                .iter()
                .filter(|&&idx| idx < self.count)
                .map(|&idx| *ptr.add(idx) as u64)
                .sum()
        }
    }
}

/// Persistent synapse-side Metal buffers.
#[cfg(target_os = "macos")]
pub struct MetalSynapseState {
    pub count: usize,
    pub pre_indices: Buffer,
    pub col_indices: Buffer,
    pub nt_type: Buffer,
    pub delay: Buffer,
    pub weight: Buffer,
    pub strength: Buffer,
    pub vesicle_rrp: Buffer,
    pub vesicle_recycling: Buffer,
    pub vesicle_reserve: Buffer,
    pub cleft_concentration: Buffer,
    pub ampa_receptors: Buffer,
    pub nmda_receptors: Buffer,
    pub gabaa_receptors: Buffer,
    pub last_pre_spike: Buffer,
    pub last_post_spike: Buffer,
    pub eligibility_trace: Buffer,
    pub bcm_theta: Buffer,
    pub post_activity_history: Buffer,
    pub tagged: Buffer,
    pub tag_strength: Buffer,
    pub homeostatic_scale: Buffer,
}

/// Persistent glia-side Metal buffers.
#[cfg(target_os = "macos")]
pub struct MetalGliaState {
    pub count: usize,
    pub astrocyte_uptake: Buffer,
    pub astrocyte_lactate: Buffer,
    pub myelin_integrity: Buffer,
    pub microglia_activation: Buffer,
    pub damage_signal: Buffer,
}

#[cfg(target_os = "macos")]
impl MetalSynapseState {
    pub fn from_cpu(gpu: &GpuContext, synapses: &SynapseArrays) -> Self {
        Self {
            count: synapses.n_synapses,
            pre_indices: gpu.buffer_from_slice(&synapses.pre_indices),
            col_indices: gpu.buffer_from_slice(&synapses.col_indices),
            nt_type: gpu.buffer_from_slice(&synapses.nt_type),
            delay: gpu.buffer_from_slice(&synapses.delay),
            weight: gpu.buffer_from_slice(&synapses.weight),
            strength: gpu.buffer_from_slice(&synapses.strength),
            vesicle_rrp: gpu.buffer_from_slice(&synapses.vesicle_rrp),
            vesicle_recycling: gpu.buffer_from_slice(&synapses.vesicle_recycling),
            vesicle_reserve: gpu.buffer_from_slice(&synapses.vesicle_reserve),
            cleft_concentration: gpu.buffer_from_slice(&synapses.cleft_concentration),
            ampa_receptors: gpu.buffer_from_slice(&synapses.ampa_receptors),
            nmda_receptors: gpu.buffer_from_slice(&synapses.nmda_receptors),
            gabaa_receptors: gpu.buffer_from_slice(&synapses.gabaa_receptors),
            last_pre_spike: gpu.buffer_from_slice(&synapses.last_pre_spike),
            last_post_spike: gpu.buffer_from_slice(&synapses.last_post_spike),
            eligibility_trace: gpu.buffer_from_slice(&synapses.eligibility_trace),
            bcm_theta: gpu.buffer_from_slice(&synapses.bcm_theta),
            post_activity_history: gpu.buffer_from_slice(&synapses.post_activity_history),
            tagged: gpu.buffer_from_slice(&synapses.tagged),
            tag_strength: gpu.buffer_from_slice(&synapses.tag_strength),
            homeostatic_scale: gpu.buffer_from_slice(&synapses.homeostatic_scale),
        }
    }

    pub fn upload_all_from_cpu(&self, synapses: &SynapseArrays) {
        unsafe {
            copy_slice_to_buffer(&self.pre_indices, &synapses.pre_indices);
            copy_slice_to_buffer(&self.col_indices, &synapses.col_indices);
            copy_slice_to_buffer(&self.nt_type, &synapses.nt_type);
            copy_slice_to_buffer(&self.delay, &synapses.delay);
            copy_slice_to_buffer(&self.weight, &synapses.weight);
            copy_slice_to_buffer(&self.strength, &synapses.strength);
            copy_slice_to_buffer(&self.vesicle_rrp, &synapses.vesicle_rrp);
            copy_slice_to_buffer(&self.vesicle_recycling, &synapses.vesicle_recycling);
            copy_slice_to_buffer(&self.vesicle_reserve, &synapses.vesicle_reserve);
            copy_slice_to_buffer(&self.cleft_concentration, &synapses.cleft_concentration);
            copy_slice_to_buffer(&self.ampa_receptors, &synapses.ampa_receptors);
            copy_slice_to_buffer(&self.nmda_receptors, &synapses.nmda_receptors);
            copy_slice_to_buffer(&self.gabaa_receptors, &synapses.gabaa_receptors);
            copy_slice_to_buffer(&self.last_pre_spike, &synapses.last_pre_spike);
            copy_slice_to_buffer(&self.last_post_spike, &synapses.last_post_spike);
            copy_slice_to_buffer(&self.eligibility_trace, &synapses.eligibility_trace);
            copy_slice_to_buffer(&self.bcm_theta, &synapses.bcm_theta);
            copy_slice_to_buffer(&self.post_activity_history, &synapses.post_activity_history);
            copy_slice_to_buffer(&self.tagged, &synapses.tagged);
            copy_slice_to_buffer(&self.tag_strength, &synapses.tag_strength);
            copy_slice_to_buffer(&self.homeostatic_scale, &synapses.homeostatic_scale);
        }
    }

    pub fn write_back_to_cpu(&self, synapses: &mut SynapseArrays) {
        unsafe {
            copy_buffer_to_slice(&self.pre_indices, &mut synapses.pre_indices);
            copy_buffer_to_slice(&self.col_indices, &mut synapses.col_indices);
            copy_buffer_to_slice(&self.nt_type, &mut synapses.nt_type);
            copy_buffer_to_slice(&self.delay, &mut synapses.delay);
            copy_buffer_to_slice(&self.weight, &mut synapses.weight);
            copy_buffer_to_slice(&self.strength, &mut synapses.strength);
            copy_buffer_to_slice(&self.vesicle_rrp, &mut synapses.vesicle_rrp);
            copy_buffer_to_slice(&self.vesicle_recycling, &mut synapses.vesicle_recycling);
            copy_buffer_to_slice(&self.vesicle_reserve, &mut synapses.vesicle_reserve);
            copy_buffer_to_slice(&self.cleft_concentration, &mut synapses.cleft_concentration);
            copy_buffer_to_slice(&self.ampa_receptors, &mut synapses.ampa_receptors);
            copy_buffer_to_slice(&self.nmda_receptors, &mut synapses.nmda_receptors);
            copy_buffer_to_slice(&self.gabaa_receptors, &mut synapses.gabaa_receptors);
            copy_buffer_to_slice(&self.last_pre_spike, &mut synapses.last_pre_spike);
            copy_buffer_to_slice(&self.last_post_spike, &mut synapses.last_post_spike);
            copy_buffer_to_slice(&self.eligibility_trace, &mut synapses.eligibility_trace);
            copy_buffer_to_slice(&self.bcm_theta, &mut synapses.bcm_theta);
            copy_buffer_to_slice(
                &self.post_activity_history,
                &mut synapses.post_activity_history,
            );
            copy_buffer_to_slice(&self.tagged, &mut synapses.tagged);
            copy_buffer_to_slice(&self.tag_strength, &mut synapses.tag_strength);
            copy_buffer_to_slice(&self.homeostatic_scale, &mut synapses.homeostatic_scale);
        }
    }

    fn recompute_weight(&self, idx: usize) {
        unsafe {
            let ampa = self.ampa_receptors.contents() as *const u16;
            let nmda = self.nmda_receptors.contents() as *const u16;
            let gabaa = self.gabaa_receptors.contents() as *const u16;
            let strength = self.strength.contents() as *const f32;
            let homeostatic = self.homeostatic_scale.contents() as *const f32;
            let weight = self.weight.contents() as *mut f32;

            let total = *ampa.add(idx) as f32 + *nmda.add(idx) as f32 + *gabaa.add(idx) as f32;
            *weight.add(idx) = (total / 50.0).min(2.0) * *strength.add(idx) * *homeostatic.add(idx);
        }
    }

    pub fn set_strength(&self, idx: usize, value: f32) {
        unsafe {
            let strength = self.strength.contents() as *mut f32;
            *strength.add(idx) = value;
        }
        self.recompute_weight(idx);
    }
}

#[cfg(target_os = "macos")]
impl MetalGliaState {
    pub fn from_cpu(gpu: &GpuContext, glia: &GliaState) -> Self {
        Self {
            count: glia.astrocyte_uptake.len(),
            astrocyte_uptake: gpu.buffer_from_slice(&glia.astrocyte_uptake),
            astrocyte_lactate: gpu.buffer_from_slice(&glia.astrocyte_lactate),
            myelin_integrity: gpu.buffer_from_slice(&glia.myelin_integrity),
            microglia_activation: gpu.buffer_from_slice(&glia.microglia_activation),
            damage_signal: gpu.buffer_from_slice(&glia.damage_signal),
        }
    }

    pub fn upload_all_from_cpu(&self, glia: &GliaState) {
        unsafe {
            copy_slice_to_buffer(&self.astrocyte_uptake, &glia.astrocyte_uptake);
            copy_slice_to_buffer(&self.astrocyte_lactate, &glia.astrocyte_lactate);
            copy_slice_to_buffer(&self.myelin_integrity, &glia.myelin_integrity);
            copy_slice_to_buffer(&self.microglia_activation, &glia.microglia_activation);
            copy_slice_to_buffer(&self.damage_signal, &glia.damage_signal);
        }
    }

    pub fn write_back_to_cpu(&self, glia: &mut GliaState) {
        unsafe {
            copy_buffer_to_slice(&self.astrocyte_uptake, &mut glia.astrocyte_uptake);
            copy_buffer_to_slice(&self.astrocyte_lactate, &mut glia.astrocyte_lactate);
            copy_buffer_to_slice(&self.myelin_integrity, &mut glia.myelin_integrity);
            copy_buffer_to_slice(&self.microglia_activation, &mut glia.microglia_activation);
            copy_buffer_to_slice(&self.damage_signal, &mut glia.damage_signal);
        }
    }
}
