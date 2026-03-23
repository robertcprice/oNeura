// world.cu — Odorant/temperature world simulation kernels
// Compiled at runtime via NVRTC.
//
// Two kernels:
//   1. world_diffusion  — 5-point Laplacian stencil diffusion + decay for odorants and temperature
//   2. world_food_update — Food consumption within eating radius

// ---------------------------------------------------------------------------
// Clamp helper
// ---------------------------------------------------------------------------
__device__ __forceinline__ float clampf(float x, float lo, float hi)
{
    return fminf(fmaxf(x, lo), hi);
}

// ---------------------------------------------------------------------------
// Kernel: world_diffusion
// One thread per grid cell (y, x). Processes all odorant channels + temperature.
//
// Grid layout:
//   odorant_grid: [H * W * C] — row-major, channels innermost: grid[(y*W + x)*C + c]
//   temp_grid:    [H * W]     — row-major: temp[y*W + x]
//   odorant_source: [H * W * C] — source emission rates (nM/ms)
//
// Diffusion via 5-point Laplacian stencil (2D):
//   L(u) = u[y-1][x] + u[y+1][x] + u[y][x-1] + u[y][x+1] - 4*u[y][x]
//   du/dt = D * L(u) - decay * u + source
//
// Boundary: Neumann (zero-flux) — clamp neighbor reads to grid bounds.
// ---------------------------------------------------------------------------
extern "C" __global__ void world_diffusion(
    float* __restrict__ odorant_grid,
    float* __restrict__ temp_grid,
    const float* __restrict__ odorant_source,
    float D_odorant,
    float D_temp,
    float decay_rate,
    int H,
    int W,
    int C,
    float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = H * W;
    if (idx >= total_cells) return;

    int y = idx / W;
    int x = idx % W;

    // Neighbor indices with Neumann boundary (clamp to edges)
    int y_up   = (y > 0)     ? y - 1 : 0;
    int y_down = (y < H - 1) ? y + 1 : H - 1;
    int x_left = (x > 0)     ? x - 1 : 0;
    int x_right= (x < W - 1) ? x + 1 : W - 1;

    // --- Odorant diffusion for each channel ---
    for (int c = 0; c < C; c++) {
        int center = (y * W + x) * C + c;
        float u_c  = odorant_grid[center];

        float u_up    = odorant_grid[(y_up   * W + x)       * C + c];
        float u_down  = odorant_grid[(y_down * W + x)       * C + c];
        float u_left  = odorant_grid[(y * W + x_left)       * C + c];
        float u_right = odorant_grid[(y * W + x_right)      * C + c];

        // 5-point Laplacian
        float laplacian = u_up + u_down + u_left + u_right - 4.0f * u_c;

        // Diffusion + decay + source
        float du = D_odorant * laplacian - decay_rate * u_c + odorant_source[center];
        float new_val = u_c + dt * du;

        // Clamp non-negative
        odorant_grid[center] = fmaxf(new_val, 0.0f);
    }

    // --- Temperature diffusion (no decay, no source in this formulation) ---
    {
        float t_c     = temp_grid[y * W + x];
        float t_up    = temp_grid[y_up   * W + x];
        float t_down  = temp_grid[y_down * W + x];
        float t_left  = temp_grid[y * W + x_left];
        float t_right = temp_grid[y * W + x_right];

        float laplacian = t_up + t_down + t_left + t_right - 4.0f * t_c;
        float dt_temp = D_temp * laplacian;
        temp_grid[y * W + x] = t_c + dt * dt_temp;
    }
}

// ---------------------------------------------------------------------------
// Kernel: world_food_update
// One thread per food source. Checks proximity to body and decrements food.
// ---------------------------------------------------------------------------
extern "C" __global__ void world_food_update(
    float* __restrict__ food_x,
    float* __restrict__ food_y,
    float* __restrict__ food_amount,
    float body_x,
    float body_y,
    float eat_radius,
    int F)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;

    if (food_amount[f] <= 0.0f) return;

    float dx = food_x[f] - body_x;
    float dy = food_y[f] - body_y;
    float dist_sq = dx * dx + dy * dy;
    float radius_sq = eat_radius * eat_radius;

    if (dist_sq < radius_sq) {
        // Consume a fixed bite per step (1.0 unit)
        float bite = fminf(1.0f, food_amount[f]);
        food_amount[f] -= bite;
        if (food_amount[f] < 0.0f) {
            food_amount[f] = 0.0f;
        }
    }
}
