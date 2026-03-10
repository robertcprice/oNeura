//! GPU-resident world simulation -- odorant diffusion, temperature, light, food.
//!
//! The world is a 2D grid (default 64x64) with C odorant channels.
//! All state lives on GPU; diffusion runs via CUDA kernel each step.
//!
//! # Physical Model
//!
//! Odorant diffusion follows the 2D heat equation with decay:
//!     dC/dt = D * laplacian(C) - decay * C + source
//!
//! where D is the diffusion coefficient (mm^2/s), decay is the first-order
//! degradation rate, and source is the per-cell emission rate. CFL stability
//! requires dt < dx^2 / (4*D); the kernel performs CFL subcycling internally
//! when D is large (e.g. D=22.8 for NH3).
//!
//! Temperature uses a separate diffusion coefficient (thermal conductivity of
//! air ~0.1 mm^2/s) and does not decay. Light is static per step (set from CPU).
//!
//! Food sources are point objects with finite amount. When the organism's body
//! position is within `eat_radius` of a food source, the food amount decreases.

#[cfg(feature = "cuda")]
use crate::cuda::CudaContext;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync};
#[cfg(feature = "cuda")]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default grid width (cells). Each cell is 1 mm.
#[cfg(feature = "cuda")]
pub const DEFAULT_WORLD_W: u32 = 64;

/// Default grid height (cells).
#[cfg(feature = "cuda")]
pub const DEFAULT_WORLD_H: u32 = 64;

/// Maximum number of odorant channels (chemical species tracked simultaneously).
#[cfg(feature = "cuda")]
pub const MAX_ODORANT_CHANNELS: u32 = 4;

/// Maximum number of discrete food sources in the world.
#[cfg(feature = "cuda")]
pub const MAX_FOOD_SOURCES: u32 = 16;

// ---------------------------------------------------------------------------
// GpuWorld
// ---------------------------------------------------------------------------

/// GPU-resident 2D world state for Drosophila-scale experiments.
///
/// All grid buffers use row-major layout: index = channel * H * W + y * W + x.
/// Scalar parameters (diffusion coefficients, decay rate) are passed to kernels
/// as launch arguments so they can be tuned per-experiment without re-upload.
#[cfg(feature = "cuda")]
pub struct GpuWorld {
    /// Grid width in cells.
    pub width: u32,
    /// Grid height in cells.
    pub height: u32,
    /// Number of active odorant channels (1..=MAX_ODORANT_CHANNELS).
    pub n_channels: u32,

    // ===== Grid Buffers (device-resident) =====
    /// Odorant concentration grid. Shape: [C, H, W] flattened.
    /// Units: arbitrary concentration (nM-scale for biological realism).
    pub odorant_grid: CudaSlice<f32>,

    /// Odorant emission rate grid. Shape: [C, H, W] flattened.
    /// Non-zero cells continuously emit odorant into the diffusion field.
    pub odorant_source: CudaSlice<f32>,

    /// Temperature field. Shape: [H, W] flattened. Units: degrees Celsius.
    pub temp_grid: CudaSlice<f32>,

    /// Light intensity field. Shape: [H, W] flattened.
    /// Normalized [0, 1] where 1.0 = full daylight.
    pub light_grid: CudaSlice<f32>,

    // ===== Food Sources =====
    /// X-coordinates of food sources in world units.
    pub food_x: CudaSlice<f32>,
    /// Y-coordinates of food sources in world units.
    pub food_y: CudaSlice<f32>,
    /// Remaining amount of each food source (depletes when eaten).
    pub food_amount: CudaSlice<f32>,
    /// Current number of active food sources.
    pub n_food: u32,

    // ===== Physics Parameters =====
    /// Odorant diffusion coefficient (mm^2/s). Typical: 2.0 (ethanol) to 22.8 (NH3).
    pub d_odorant: f32,
    /// Thermal diffusion coefficient (mm^2/s). Air at 25C ~ 0.1.
    pub d_temp: f32,
    /// First-order odorant decay rate (1/s). Typical: 0.01 for slow degradation.
    pub decay_rate: f32,
}

#[cfg(feature = "cuda")]
impl GpuWorld {
    /// Allocate a new GPU world with all grids zeroed (temperature at 22C, light at 1.0).
    ///
    /// # Arguments
    /// * `device` - CUDA device handle for buffer allocation.
    /// * `w` - Grid width in cells.
    /// * `h` - Grid height in cells.
    /// * `n_channels` - Number of odorant channels (clamped to MAX_ODORANT_CHANNELS).
    pub fn new(device: &Arc<CudaDevice>, w: u32, h: u32, n_channels: u32) -> Result<Self, String> {
        let map_err = |e: cudarc::driver::DriverError| format!("World alloc: {}", e);
        let n_ch = n_channels.min(MAX_ODORANT_CHANNELS);
        let grid_size = (w * h * n_ch) as usize;
        let spatial_size = (w * h) as usize;

        Ok(Self {
            width: w,
            height: h,
            n_channels: n_ch,
            odorant_grid: device.alloc_zeros(grid_size).map_err(map_err)?,
            odorant_source: device.alloc_zeros(grid_size).map_err(map_err)?,
            // 22 C ambient temperature
            temp_grid: device
                .htod_copy(vec![22.0f32; spatial_size])
                .map_err(map_err)?,
            // Uniform daylight
            light_grid: device
                .htod_copy(vec![1.0f32; spatial_size])
                .map_err(map_err)?,
            food_x: device
                .alloc_zeros(MAX_FOOD_SOURCES as usize)
                .map_err(map_err)?,
            food_y: device
                .alloc_zeros(MAX_FOOD_SOURCES as usize)
                .map_err(map_err)?,
            food_amount: device
                .alloc_zeros(MAX_FOOD_SOURCES as usize)
                .map_err(map_err)?,
            n_food: 0,
            d_odorant: 2.0,
            d_temp: 0.1,
            decay_rate: 0.01,
        })
    }

    /// Place an odorant source at grid position (gx, gy) on the given channel.
    ///
    /// The source continuously emits at `rate` concentration units per second
    /// until explicitly removed by setting rate to 0.
    pub fn place_odorant_source(
        &mut self,
        device: &Arc<CudaDevice>,
        gx: u32,
        gy: u32,
        channel: u32,
        rate: f32,
    ) -> Result<(), String> {
        if gx >= self.width || gy >= self.height || channel >= self.n_channels {
            return Ok(());
        }
        let idx = (channel * self.height * self.width + gy * self.width + gx) as usize;
        // Round-trip: download, modify, re-upload. Acceptable for setup-time calls
        // (not called in the hot loop).
        let mut src = device
            .dtoh_sync_copy(&self.odorant_source)
            .map_err(|e| format!("D2H source: {}", e))?;
        src[idx] = rate;
        self.odorant_source = device
            .htod_copy(src)
            .map_err(|e| format!("H2D source: {}", e))?;
        Ok(())
    }

    /// Place a food source at world coordinates (x, y) with the given amount.
    ///
    /// Returns silently if the maximum number of food sources has been reached.
    pub fn place_food(
        &mut self,
        device: &Arc<CudaDevice>,
        x: f32,
        y: f32,
        amount: f32,
    ) -> Result<(), String> {
        if self.n_food >= MAX_FOOD_SOURCES {
            return Ok(());
        }
        let idx = self.n_food as usize;
        let mut fx = device
            .dtoh_sync_copy(&self.food_x)
            .map_err(|e| format!("{}", e))?;
        let mut fy = device
            .dtoh_sync_copy(&self.food_y)
            .map_err(|e| format!("{}", e))?;
        let mut fa = device
            .dtoh_sync_copy(&self.food_amount)
            .map_err(|e| format!("{}", e))?;
        fx[idx] = x;
        fy[idx] = y;
        fa[idx] = amount;
        self.food_x = device.htod_copy(fx).map_err(|e| format!("{}", e))?;
        self.food_y = device.htod_copy(fy).map_err(|e| format!("{}", e))?;
        self.food_amount = device.htod_copy(fa).map_err(|e| format!("{}", e))?;
        self.n_food += 1;
        Ok(())
    }

    /// Set a linear temperature gradient along the x-axis.
    ///
    /// Temperature varies linearly from `temp_low` (at x=0) to `temp_high`
    /// (at x=width-1). Used for thermotaxis experiments.
    pub fn set_temp_gradient(
        &mut self,
        device: &Arc<CudaDevice>,
        temp_low: f32,
        temp_high: f32,
    ) -> Result<(), String> {
        let mut temp = vec![0.0f32; (self.width * self.height) as usize];
        let w_max = (self.width - 1).max(1) as f32;
        for y in 0..self.height {
            for x in 0..self.width {
                let t = x as f32 / w_max;
                temp[(y * self.width + x) as usize] = temp_low + t * (temp_high - temp_low);
            }
        }
        self.temp_grid = device.htod_copy(temp).map_err(|e| format!("{}", e))?;
        Ok(())
    }

    /// Set a radial light pattern centered at (cx, cy) with given radius and intensity.
    ///
    /// Used for phototaxis experiments. Light intensity falls off linearly from
    /// `intensity` at center to 0 at `radius` cells away.
    pub fn set_light_spot(
        &mut self,
        device: &Arc<CudaDevice>,
        cx: u32,
        cy: u32,
        radius: f32,
        intensity: f32,
    ) -> Result<(), String> {
        let mut light = vec![0.0f32; (self.width * self.height) as usize];
        let r2 = radius * radius;
        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                let dist2 = dx * dx + dy * dy;
                if dist2 < r2 {
                    let falloff = 1.0 - (dist2 / r2).sqrt();
                    light[(y * self.width + x) as usize] = intensity * falloff;
                }
            }
        }
        self.light_grid = device.htod_copy(light).map_err(|e| format!("{}", e))?;
        Ok(())
    }

    /// Run one world physics step: odorant diffusion + decay + source emission.
    ///
    /// The CUDA kernel `world_diffusion` handles:
    /// 1. 5-point stencil Laplacian for odorant diffusion (with Neumann BCs).
    /// 2. First-order decay of odorant concentration.
    /// 3. Source emission (additive each step).
    /// 4. Thermal diffusion (same stencil, no decay).
    ///
    /// CFL subcycling is handled inside the kernel when D * dt / dx^2 > 0.25.
    pub fn step(&self, ctx: &CudaContext, dt: f32) -> Result<(), String> {
        let n_cells = self.width * self.height;
        let cfg = CudaContext::launch_cfg(n_cells);

        let func = ctx
            .device
            .get_func("world", "world_diffusion")
            .ok_or("world_diffusion kernel not found")?;

        unsafe {
            func.launch(
                cfg,
                (
                    &self.odorant_grid,
                    &self.temp_grid,
                    &self.odorant_source,
                    self.d_odorant,
                    self.d_temp,
                    self.decay_rate,
                    self.height as i32,
                    self.width as i32,
                    self.n_channels as i32,
                    dt,
                ),
            )
        }
        .map_err(|e| format!("world_diffusion launch: {}", e))?;

        Ok(())
    }

    /// Update food sources based on organism proximity.
    ///
    /// For each food source, checks if the body position is within `eat_radius`.
    /// If so, decreases the food amount by a fixed bite size per step.
    pub fn update_food(&self, ctx: &CudaContext, body: &CudaSlice<f32>) -> Result<(), String> {
        if self.n_food == 0 {
            return Ok(());
        }

        // Download body to get position (body[0]=x, body[1]=y).
        // The body buffer is only 8 floats so this round-trip is negligible.
        let body_host = ctx
            .device
            .dtoh_sync_copy(body)
            .map_err(|e| format!("D2H body: {}", e))?;

        let cfg = CudaContext::launch_cfg(self.n_food);

        let func = ctx
            .device
            .get_func("world", "world_food_update")
            .ok_or("world_food_update kernel not found")?;

        unsafe {
            func.launch(
                cfg,
                (
                    &self.food_x,
                    &self.food_y,
                    &self.food_amount,
                    body_host[0], // body_x
                    body_host[1], // body_y
                    2.0f32,       // eat_radius (cells)
                    self.n_food as i32,
                ),
            )
        }
        .map_err(|e| format!("world_food_update launch: {}", e))?;

        Ok(())
    }

    /// Download the full odorant grid to CPU for visualization or analysis.
    pub fn download_odorant_grid(&self, device: &Arc<CudaDevice>) -> Result<Vec<f32>, String> {
        device
            .dtoh_sync_copy(&self.odorant_grid)
            .map_err(|e| format!("D2H odorant: {}", e))
    }

    /// Download the temperature grid to CPU.
    pub fn download_temp_grid(&self, device: &Arc<CudaDevice>) -> Result<Vec<f32>, String> {
        device
            .dtoh_sync_copy(&self.temp_grid)
            .map_err(|e| format!("D2H temp: {}", e))
    }

    /// Total number of grid cells (H * W).
    pub fn n_cells(&self) -> u32 {
        self.width * self.height
    }
}
