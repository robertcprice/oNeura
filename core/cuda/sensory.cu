// sensory.cu — Bilateral sensory encoding for biophysical organism simulation
// One thread per sensory neuron. Compiled at runtime via NVRTC.
//
// Sensory modalities:
//   - Olfactory (antennal lobe, left/right) — samples odorant grid at antenna positions
//   - Visual (optic lobe, left/right) — samples light grid at eye positions
//   - Thermal (left/right) — samples temperature grid at antenna positions
//
// Region layout via region_starts/region_sizes (12 entries):
//   0: AL_L (antennal lobe left)
//   1: AL_R (antennal lobe right)
//   2: optic_L
//   3: optic_R
//   4: thermal_L
//   5: thermal_R
//   6-11: reserved
//
// Body state vector (8 floats):
//   [0]=x, [1]=y, [2]=heading (rad), [3]=speed, [4]=hp,
//   [5]=prev_temp, [6]=prev_food, [7]=time_of_day

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Antenna/eye offset angle from body heading (radians)
#define SENSOR_ANGLE_OFFSET  (M_PI / 6.0f)   // 30 degrees

// Antenna/eye distance from body center (grid units)
#define SENSOR_DISTANCE      2.0f

// Weber-Fechner log scaling constant (prevents log(0))
#define WEBER_EPSILON        1.0f

// Current scaling factors
#define ODORANT_CURRENT_SCALE  50.0f   // nM -> uA/cm^2
#define LIGHT_CURRENT_SCALE    80.0f   // intensity -> uA/cm^2
#define THERMAL_CURRENT_SCALE  30.0f   // delta_T -> uA/cm^2

// Region type identifiers
#define REGION_AL_L       0
#define REGION_AL_R       1
#define REGION_OPTIC_L    2
#define REGION_OPTIC_R    3
#define REGION_THERMAL_L  4
#define REGION_THERMAL_R  5
#define NUM_REGIONS       6

// ---------------------------------------------------------------------------
// Bilinear interpolation on a 2D grid
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bilinear_sample_2d(
    const float* __restrict__ grid,
    float fx, float fy,
    int H, int W)
{
    // Clamp to grid bounds
    fx = fminf(fmaxf(fx, 0.0f), (float)(W - 1));
    fy = fminf(fmaxf(fy, 0.0f), (float)(H - 1));

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);

    float sx = fx - (float)x0;
    float sy = fy - (float)y0;

    float v00 = grid[y0 * W + x0];
    float v10 = grid[y0 * W + x1];
    float v01 = grid[y1 * W + x0];
    float v11 = grid[y1 * W + x1];

    return (1.0f - sy) * ((1.0f - sx) * v00 + sx * v10)
         +        sy  * ((1.0f - sx) * v01 + sx * v11);
}

// ---------------------------------------------------------------------------
// Bilinear interpolation on a 3D grid (H x W x C), sample specific channel
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bilinear_sample_3d(
    const float* __restrict__ grid,
    float fx, float fy,
    int H, int W, int C, int channel)
{
    fx = fminf(fmaxf(fx, 0.0f), (float)(W - 1));
    fy = fminf(fmaxf(fy, 0.0f), (float)(H - 1));

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);

    float sx = fx - (float)x0;
    float sy = fy - (float)y0;

    float v00 = grid[(y0 * W + x0) * C + channel];
    float v10 = grid[(y0 * W + x1) * C + channel];
    float v01 = grid[(y1 * W + x0) * C + channel];
    float v11 = grid[(y1 * W + x1) * C + channel];

    return (1.0f - sy) * ((1.0f - sx) * v00 + sx * v10)
         +        sy  * ((1.0f - sx) * v01 + sx * v11);
}

// ---------------------------------------------------------------------------
// Find which sensory region a neuron index belongs to
// Returns region type (0-5) or -1 if not in any sensory region.
// Linear scan is fine for 6 regions.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int find_region(
    int neuron_idx,
    const unsigned int* __restrict__ region_starts,
    const unsigned int* __restrict__ region_sizes)
{
    for (int r = 0; r < NUM_REGIONS; r++) {
        unsigned int start = region_starts[r];
        unsigned int size  = region_sizes[r];
        if (size == 0) continue;
        if ((unsigned int)neuron_idx >= start && (unsigned int)neuron_idx < start + size) {
            return r;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Weber-Fechner law: logarithmic scaling of stimulus intensity
// I_out = scale * log(1 + stimulus / epsilon)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float weber_fechner(float stimulus, float scale)
{
    return scale * logf(1.0f + fabsf(stimulus) / WEBER_EPSILON);
}

// ---------------------------------------------------------------------------
// Kernel: sensory_encode
// One thread per sensory neuron. Converts environmental stimuli to neural current.
//
// Total threads should cover all neurons, but only sensory region neurons produce output.
// ---------------------------------------------------------------------------
extern "C" __global__ void sensory_encode(
    const float* __restrict__ body,
    const float* __restrict__ odorant_grid,
    const float* __restrict__ temp_grid,
    const float* __restrict__ light_grid,
    float* __restrict__ ext_current,
    const unsigned int* __restrict__ region_starts,
    const unsigned int* __restrict__ region_sizes,
    int H,
    int W,
    int C,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Determine which sensory region this neuron belongs to
    int region = find_region(i, region_starts, region_sizes);
    if (region < 0) return;  // not a sensory neuron

    // --- Load body state ---
    float bx      = body[0];
    float by      = body[1];
    float heading  = body[2];

    // --- Compute sensor positions ---
    // Left sensor: heading + 30 degrees
    // Right sensor: heading - 30 degrees
    int side = region & 1;  // 0=left (even index), 1=right (odd index)
    float angle_offset = (side == 0) ? SENSOR_ANGLE_OFFSET : -SENSOR_ANGLE_OFFSET;
    float sensor_angle = heading + angle_offset;
    float sensor_x = bx + SENSOR_DISTANCE * cosf(sensor_angle);
    float sensor_y = by + SENSOR_DISTANCE * sinf(sensor_angle);

    // --- Compute index within the sensory region ---
    unsigned int region_start = region_starts[region];
    unsigned int region_size  = region_sizes[region];
    int local_idx = i - (int)region_start;

    float current = 0.0f;

    int region_type = region >> 1;  // 0=olfactory, 1=visual, 2=thermal

    switch (region_type) {
        case 0: {
            // Olfactory (antennal lobe): sample odorant concentration
            // Map local neuron index to odorant channel (distribute neurons across channels)
            int channel = local_idx % C;
            float conc = bilinear_sample_3d(odorant_grid, sensor_x, sensor_y, H, W, C, channel);
            current = weber_fechner(conc, ODORANT_CURRENT_SCALE);
            break;
        }
        case 1: {
            // Visual (optic lobe): sample light intensity
            // Each neuron in the optic region samples a slightly different visual field position
            // Spread neurons across a small receptive field around the eye position
            float rf_spread = 3.0f;  // receptive field radius in grid cells
            float theta = (float)local_idx / (float)region_size * 2.0f * M_PI;
            float rf_r = rf_spread * (float)local_idx / (float)region_size;
            float sample_x = sensor_x + rf_r * cosf(theta);
            float sample_y = sensor_y + rf_r * sinf(theta);

            float intensity = bilinear_sample_2d(light_grid, sample_x, sample_y, H, W);
            current = weber_fechner(intensity, LIGHT_CURRENT_SCALE);
            break;
        }
        case 2: {
            // Thermal: sample temperature
            float temp = bilinear_sample_2d(temp_grid, sensor_x, sensor_y, H, W);
            // Thermal neurons encode deviation from comfortable range (20-25 C)
            float comfort = 22.5f;
            float deviation = temp - comfort;
            current = weber_fechner(deviation, THERMAL_CURRENT_SCALE);
            // Sign: positive current for both hot and cold (absolute deviation)
            // The left/right difference provides directional information
            break;
        }
        default:
            break;
    }

    // --- Inject current ---
    // Use atomicAdd since ext_current may also receive synaptic input
    atomicAdd(&ext_current[i], current);
}
