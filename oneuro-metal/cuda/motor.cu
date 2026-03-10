// motor.cu — Motor decoding + body physics update
// Small grid kernel (can be single-threaded). Compiled at runtime via NVRTC.
//
// Reads voltage from motor neuron populations, computes differential drive,
// and updates body position/heading with toroidal wrapping.
//
// Motor populations (via motor_start/motor_size, 4 entries):
//   0: left_motor   — drives left turning
//   1: right_motor  — drives right turning
//   2: forward      — drives forward locomotion
//   3: backward     — drives backward locomotion (or braking)
//
// Body state vector (8 floats):
//   [0]=x, [1]=y, [2]=heading (rad), [3]=speed, [4]=hp,
//   [5]=prev_temp, [6]=prev_food, [7]=time_of_day

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Motor population indices
#define MOTOR_LEFT     0
#define MOTOR_RIGHT    1
#define MOTOR_FORWARD  2
#define MOTOR_BACKWARD 3
#define NUM_MOTOR_POPS 4

// ---------------------------------------------------------------------------
// Helper: compute mean voltage of a neuron population
// ---------------------------------------------------------------------------
__device__ __forceinline__ float population_mean_voltage(
    const float* __restrict__ voltage,
    unsigned int start,
    unsigned int size)
{
    if (size == 0) return 0.0f;

    float sum = 0.0f;
    for (unsigned int j = 0; j < size; j++) {
        sum += voltage[start + j];
    }
    return sum / (float)size;
}

// ---------------------------------------------------------------------------
// Clamp helper
// ---------------------------------------------------------------------------
__device__ __forceinline__ float clampf(float x, float lo, float hi)
{
    return fminf(fmaxf(x, lo), hi);
}

// ---------------------------------------------------------------------------
// Normalize angle to [-pi, pi]
// ---------------------------------------------------------------------------
__device__ __forceinline__ float normalize_angle(float a)
{
    while (a > M_PI)  a -= 2.0f * M_PI;
    while (a < -M_PI) a += 2.0f * M_PI;
    return a;
}

// ---------------------------------------------------------------------------
// Kernel: motor_decode
// Single thread (or very small grid). Reads motor neuron voltages,
// computes differential drive, updates body state.
//
// Drive model:
//   Differential steering: turn = (right_mean - left_mean) * max_turn_rate * dt
//   Forward/backward:      speed = (forward_mean - backward_mean) * max_speed
//   Speed clamped to [0, max_speed] (no reverse for simplicity)
//
// Physics update:
//   heading += turn
//   x += cos(heading) * speed * dt
//   y += sin(heading) * speed * dt
//   Toroidal wrapping at world boundaries
// ---------------------------------------------------------------------------
extern "C" __global__ void motor_decode(
    float* __restrict__ body,
    const float* __restrict__ voltage,
    const unsigned int* __restrict__ motor_start,
    const unsigned int* __restrict__ motor_size,
    float max_turn_rate,
    float max_speed,
    float world_w,
    float world_h,
    float dt,
    int N)
{
    // Single-threaded kernel — only thread 0 executes
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // --- Compute mean voltages for each motor population ---
    float left_mean  = population_mean_voltage(voltage, motor_start[MOTOR_LEFT],     motor_size[MOTOR_LEFT]);
    float right_mean = population_mean_voltage(voltage, motor_start[MOTOR_RIGHT],    motor_size[MOTOR_RIGHT]);
    float fwd_mean   = population_mean_voltage(voltage, motor_start[MOTOR_FORWARD],  motor_size[MOTOR_FORWARD]);
    float bwd_mean   = population_mean_voltage(voltage, motor_start[MOTOR_BACKWARD], motor_size[MOTOR_BACKWARD]);

    // --- Normalize voltages to [0, 1] range ---
    // Resting potential ~ -65 mV, peak ~ +50 mV
    // Map: -65 -> 0.0, +50 -> 1.0
    float v_min = -65.0f;
    float v_range = 115.0f;  // 50 - (-65)

    float left_norm  = clampf((left_mean  - v_min) / v_range, 0.0f, 1.0f);
    float right_norm = clampf((right_mean - v_min) / v_range, 0.0f, 1.0f);
    float fwd_norm   = clampf((fwd_mean   - v_min) / v_range, 0.0f, 1.0f);
    float bwd_norm   = clampf((bwd_mean   - v_min) / v_range, 0.0f, 1.0f);

    // --- Differential steering ---
    float turn = (right_norm - left_norm) * max_turn_rate * dt;

    // --- Speed from forward/backward balance ---
    float speed = (fwd_norm - bwd_norm) * max_speed;
    speed = clampf(speed, 0.0f, max_speed);

    // --- Load current body state ---
    float x       = body[0];
    float y       = body[1];
    float heading  = body[2];

    // --- Update heading ---
    heading += turn;
    heading = normalize_angle(heading);

    // --- Update position ---
    x += cosf(heading) * speed * dt;
    y += sinf(heading) * speed * dt;

    // --- Toroidal wrapping ---
    if (x < 0.0f)     x += world_w;
    if (x >= world_w)  x -= world_w;
    if (y < 0.0f)     y += world_h;
    if (y >= world_h)  y -= world_h;

    // --- Store updated body state ---
    body[0] = x;
    body[1] = y;
    body[2] = heading;
    body[3] = speed;
}
