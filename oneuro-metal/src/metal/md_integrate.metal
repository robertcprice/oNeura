// ============================================================================
// md_integrate.metal — GPU Molecular Dynamics Integration
// ============================================================================
// Velocity Verlet integration with Langevin thermostat for NPT/NVT ensemble.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
struct MDParams {
    uint n_atoms;
    float dt;              // Timestep (ps)
    float temperature;     // Target temperature (K)
    float gamma;          // Friction coefficient (ps^-1)
    float box_x, box_y, box_z;
    float kBT;            // k_B * T in MD units
    float sqrt_dt;        // sqrt(dt)
    uint seed;            // Random seed
};

// Atom structure
struct Atom {
    float3 position;
    float3 velocity;
    float3 force;
    float mass;
    float charge;
    float sigma;
    float epsilon;
};

// Random number generation (LCG)
struct LCG {
    uint state;

    // Seed
    static LCG create(uint seed) {
        LCG lcg;
        lcg.state = seed;
        return lcg;
    }

    // Next random float [0, 1)
    float next() {
        state = state * 1664525u + 1013904223u;
        return float(state) / float(0xFFFFFFFFu);
    }

    // Box-Muller for normal distribution
    float2 normal() {
        float u1 = next();
        float u2 = next();
        u1 = max(u1, 1e-10f);  // Avoid log(0)
        float r = sqrt(-2.0f * log(u1));
        float phi = 2.0f * M_PI_F * u2;
        return float2(r * cos(phi), r * sin(phi));
    }
};

// ---------------------------------------------------------------------------
// Kernel: Velocity Verlet integration with Langevin thermostat
// ---------------------------------------------------------------------------
kernel void integrate_langevin(
    device Atom* atoms [[buffer(0)]],
    constant MDParams& params [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.n_atoms) return;

    Atom atom = atoms[id];
    float inv_mass = 1.0f / atom.mass;
    float3 force = atom.force;

    // Random number generator per thread
    LCG rng = LCG::create(params.seed + id * 7919u);

    // Langevin friction and random force
    float gamma = params.gamma;
    float3 random_force;
    random_force.x = rng.normal().x * params.sqrt_dt;
    random_force.y = rng.normal().y * params.sqrt_dt;
    random_force.z = rng.normal().x * params.sqrt_dt;  // Reuse for z

    // Scale by sqrt(kBT/m)
    float scale = params.kBT * sqrt(inv_mass);
    random_force *= scale;

    // Velocity Verlet with Langevin
    // v(t + dt/2) = v(t) + (F/m - gamma*v/2) * dt + random * sqrt(dt)
    float3 v_half = atom.velocity;
    v_half += (force * inv_mass - gamma * atom.velocity * 0.5f) * params.dt;
    v_half += random_force;

    // Damping factor
    float damp = exp(-gamma * params.dt * 0.5f);
    v_half *= damp;

    // Position update
    atom.position += v_half * params.dt;

    // Apply periodic boundary conditions
    if (params.box_x > 0.0f) {
        atom.position.x = fmod(atom.position.x, params.box_x);
        if (atom.position.x < 0.0f) atom.position.x += params.box_x;
    }
    if (params.box_y > 0.0f) {
        atom.position.y = fmod(atom.position.y, params.box_y);
        if (atom.position.y < 0.0f) atom.position.y += params.box_y;
    }
    if (params.box_z > 0.0f) {
        atom.position.z = fmod(atom.position.z, params.box_z);
        if (atom.position.z < 0.0f) atom.position.z += params.box_z;
    }

    // Final velocity (for next step)
    atom.velocity = v_half;
    atom.force = float3(0.0f);  // Clear forces for next step

    atoms[id] = atom;
}

// ---------------------------------------------------------------------------
// Kernel: Velocity Verlet (no thermostat) - NVE ensemble
// ---------------------------------------------------------------------------
kernel void integrate_nve(
    device Atom* atoms [[buffer(0)]],
    constant MDParams& params [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.n_atoms) return;

    Atom atom = atoms[id];
    float inv_mass = 1.0f / atom.mass;
    float3 force = atom.force;

    // Velocity Verlet (NVE)
    float3 v_half = atom.velocity + force * inv_mass * params.dt * 0.5f;
    atom.position += v_half * params.dt;

    // Update velocity with new forces (would need force recalculation)
    atom.velocity = v_half + force * inv_mass * params.dt * 0.5f;
    atom.force = float3(0.0f);

    // Apply PBC
    if (params.box_x > 0.0f) {
        atom.position.x = fmod(atom.position.x + params.box_x, params.box_x);
    }
    if (params.box_y > 0.0f) {
        atom.position.y = fmod(atom.position.y + params.box_y, params.box_y);
    }
    if (params.box_z > 0.0f) {
        atom.position.z = fmod(atom.position.z + params.box_z, params.box_z);
    }

    atoms[id] = atom;
}

// ---------------------------------------------------------------------------
// Kernel: Berendsen thermostat - velocity rescaling
// ---------------------------------------------------------------------------
kernel void apply_berendsen_thermostat(
    device Atom* atoms [[buffer(0)]],
    constant MDParams& params [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.n_atoms) return;

    // Note: In practice, you'd compute current temperature first
    // This is a simplified per-atom scaling

    float3 v = atoms[id].velocity;
    float speed = length(v);
    if (speed < 1e-6f) return;

    // Target speed based on temperature (simplified)
    float target_speed = sqrt(params.kBT / atoms[id].mass);
    float scale = target_speed / speed;

    // Gentle scaling toward target
    float tau = 0.1f;  // Time constant
    float lambda = 1.0f + (scale - 1.0f) * params.dt / tau;
    lambda = clamp(lambda, 0.9f, 1.1f);

    atoms[id].velocity = v * lambda;
}
