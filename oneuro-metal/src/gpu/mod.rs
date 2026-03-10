//! Metal GPU infrastructure — device init, pipeline cache, dispatch helpers.
//!
//! Uses Apple's unified memory (`StorageModeShared`) for zero-copy CPU↔GPU
//! access on Apple Silicon. Each compute pipeline is compiled once and cached.

#[cfg(target_os = "macos")]
use metal::*;
#[cfg(target_os = "macos")]
use std::mem::ManuallyDrop;

pub mod calcium_dynamics;
pub mod diffusion_3d;
pub mod gene_expression;
pub mod glia;
pub mod hh_gating;
pub mod md_gpu;
pub mod membrane_integration;
pub mod metabolism;
pub mod microtubules;
pub mod pharmacology;
pub mod receptor_binding;
pub mod second_messenger;
pub mod state;
pub mod synapse_cleft;
pub mod synapse_step;
pub mod terrarium_substrate;
pub mod whole_cell_rdme;

/// Metal GPU context — holds device, queue, and compiled pipelines.
#[cfg(target_os = "macos")]
pub struct GpuContext {
    pub device: Device,
    pub queue: ManuallyDrop<CommandQueue>,
    pub pipelines: GpuPipelines,
}

/// All compiled compute pipelines.
#[cfg(target_os = "macos")]
pub struct GpuPipelines {
    pub hh_gating: ComputePipelineState,
    pub membrane_euler: ComputePipelineState,
    pub calcium_ode: ComputePipelineState,
    pub second_messenger: ComputePipelineState,
    pub synapse_step: ComputePipelineState,
    pub clear_i32_buffer: ComputePipelineState,
    pub synaptic_current_commit: ComputePipelineState,
    pub mark_last_fired: ComputePipelineState,
    pub gene_expression: ComputePipelineState,
    pub metabolism: ComputePipelineState,
    pub microtubules: ComputePipelineState,
    pub pharmacology: ComputePipelineState,
    pub glia_neuron_step: ComputePipelineState,
    pub glia_prune_synapses: ComputePipelineState,
    pub hill_binding: ComputePipelineState,
    pub cleft_dynamics: ComputePipelineState,
    pub diffusion_3d: ComputePipelineState,
    pub terrarium_substrate: ComputePipelineState,
    pub whole_cell_rdme: ComputePipelineState,
}

#[cfg(target_os = "macos")]
impl GpuContext {
    /// Initialize Metal device and compile all shaders.
    pub fn new() -> Result<Self, String> {
        let device =
            Device::system_default().ok_or_else(|| "No Metal GPU device found".to_string())?;

        let queue = device.new_command_queue();

        let pipelines = Self::create_pipelines(&device)?;

        Ok(Self {
            device,
            queue: ManuallyDrop::new(queue),
            pipelines,
        })
    }

    fn compile_pipeline(
        device: &Device,
        source: &str,
        function_name: &str,
    ) -> Result<ComputePipelineState, String> {
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(source, &options)
            .map_err(|e| {
                format!(
                    "Metal shader compilation failed for '{}': {}",
                    function_name, e
                )
            })?;
        let func = library
            .get_function(function_name, None)
            .map_err(|e| format!("Shader function '{}': {}", function_name, e))?;
        device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| format!("Pipeline '{}': {}", function_name, e))
    }

    fn create_pipelines(device: &Device) -> Result<GpuPipelines, String> {
        let make = |source: &str, function_name: &str| -> Result<ComputePipelineState, String> {
            Self::compile_pipeline(device, source, function_name)
        };

        Ok(GpuPipelines {
            hh_gating: make(include_str!("../metal/hh_gating.metal"), "hh_gating")?,
            membrane_euler: make(
                include_str!("../metal/membrane_euler.metal"),
                "membrane_euler",
            )?,
            calcium_ode: make(include_str!("../metal/calcium_ode.metal"), "calcium_ode")?,
            second_messenger: make(
                include_str!("../metal/second_messenger.metal"),
                "second_messenger",
            )?,
            synapse_step: make(include_str!("../metal/synapse_step.metal"), "synapse_step")?,
            clear_i32_buffer: make(
                include_str!("../metal/synapse_step.metal"),
                "clear_i32_buffer",
            )?,
            synaptic_current_commit: make(
                include_str!("../metal/synapse_step.metal"),
                "commit_synaptic_current",
            )?,
            mark_last_fired: make(
                include_str!("../metal/synapse_step.metal"),
                "mark_last_fired",
            )?,
            gene_expression: make(
                include_str!("../metal/gene_expression.metal"),
                "gene_expression",
            )?,
            metabolism: make(include_str!("../metal/metabolism.metal"), "metabolism")?,
            microtubules: make(include_str!("../metal/microtubules.metal"), "microtubules")?,
            pharmacology: make(include_str!("../metal/pharmacology.metal"), "pharmacology")?,
            glia_neuron_step: make(include_str!("../metal/glia.metal"), "glia_neuron_step")?,
            glia_prune_synapses: make(include_str!("../metal/glia.metal"), "glia_prune_synapses")?,
            hill_binding: make(include_str!("../metal/hill_binding.metal"), "hill_binding")?,
            cleft_dynamics: make(
                include_str!("../metal/cleft_dynamics.metal"),
                "cleft_dynamics",
            )?,
            diffusion_3d: make(include_str!("../metal/diffusion_3d.metal"), "diffusion_3d")?,
            terrarium_substrate: make(
                include_str!("../metal/terrarium_substrate.metal"),
                "terrarium_substrate",
            )?,
            whole_cell_rdme: make(
                include_str!("../metal/whole_cell_rdme.metal"),
                "whole_cell_rdme_kernel",
            )?,
        })
    }

    /// Create a shared-mode Metal buffer from a slice (zero-copy on unified memory).
    pub fn buffer_from_slice<T>(&self, data: &[T]) -> Buffer {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::StorageModeShared);
        unsafe {
            let ptr = buffer.contents() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        buffer
    }

    /// Create a shared-mode buffer of given byte size.
    pub fn buffer_of_size(&self, bytes: u64) -> Buffer {
        self.device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    }

    /// Create a new command buffer for a batched compute sequence.
    pub fn new_command_buffer(&self) -> &CommandBufferRef {
        self.queue.new_command_buffer()
    }

    /// Encode a 1D compute kernel onto an existing command buffer.
    pub fn encode_1d(
        &self,
        cmd: &CommandBufferRef,
        pipeline: &ComputePipelineState,
        buffers: &[(&Buffer, u64)],
        param_bytes: Option<(&[u8], u64)>,
        n_threads: u64,
    ) {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);

        for (i, (buf, offset)) in buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(buf), *offset);
        }

        if let Some((bytes, binding)) = param_bytes {
            enc.set_bytes(binding, bytes.len() as u64, bytes.as_ptr() as *const _);
        }

        let tpg = pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        let threadgroups = (n_threads + tpg - 1) / tpg;

        enc.dispatch_thread_groups(MTLSize::new(threadgroups, 1, 1), MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
    }

    /// Submit a previously encoded command buffer and wait for completion.
    pub fn commit_and_wait(&self, cmd: &CommandBufferRef) {
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Dispatch a compute kernel with 1D thread grid.
    pub fn dispatch_1d(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[(&Buffer, u64)],        // (buffer, offset) pairs
        param_bytes: Option<(&[u8], u64)>, // (bytes, binding_index)
        n_threads: u64,
    ) {
        let cmd = self.new_command_buffer();
        self.encode_1d(&cmd, pipeline, buffers, param_bytes, n_threads);
        self.commit_and_wait(cmd);
    }
}

/// CPU fallback context for non-macOS platforms.
#[cfg(not(target_os = "macos"))]
pub struct GpuContext;

#[cfg(not(target_os = "macos"))]
impl GpuContext {
    pub fn new() -> Result<Self, String> {
        Ok(Self)
    }
}

/// Whether Metal GPU acceleration is available.
pub fn has_gpu() -> bool {
    #[cfg(target_os = "macos")]
    {
        Device::system_default().is_some()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}
