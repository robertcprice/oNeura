// ============================================================================
// cuda_md.cuh - CUDA Molecular Dynamics Kernels
// ============================================================================
// CUDA compute kernels for GPU-accelerated molecular dynamics on NVIDIA GPUs.
// ============================================================================

#ifndef CUDA_MD_CUH
#define CUDA_MD_CUH

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// Data Structures (must match Rust structs)
// ============================================================================

struct MDParams {
    uint n_atoms;
    float dt;
    float temperature;
    float cutoff;
    float box_x, box_y, box_z;
    float epsilon_lj;
    float epsilon_coulomb;
};

struct MDAtom {
    float3 position;
    float3 velocity;
    float3 force;
    float mass;
    float charge;
    float sigma;
    float epsilon;
};

struct MDBond {
    uint i, j;
    float r0;
    float k;
};

struct EnergyOutput {
    float bond_energy;
    float angle_energy;
    float dihedral_energy;
    float vdw_energy;
    float electrostatic_energy;
    float kinetic_energy;
};

// ============================================================================
// Kernel: Clear forces
// ============================================================================

__global__ void clear_forces(MDAtom* atoms, uint n_atoms) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_atoms) return;

    atoms[idx].force = make_float3(0.0f, 0.0f, 0.0f);
}

// ============================================================================
// Kernel: Bond forces
// ============================================================================

__global__ void compute_bond_forces(
    MDAtom* atoms,
    MDBond* bonds,
    uint n_bonds,
    EnergyOutput* energy
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bonds) return;

    MDBond bond = bonds[idx];

    float3 ri = atoms[bond.i].position;
    float3 rj = atoms[bond.j].position;

    float3 dr = rj - ri;
    float r = length(dr);
    if (r < 1e-6f) return;

    float dr_eq = r - bond.r0;
    float force_mag = -2.0f * bond.k * dr_eq;
    float3 fij = (force_mag / r) * dr;

    // Atomic add for forces
    atomicAdd(&atoms[bond.i].force.x, fij.x);
    atomicAdd(&atoms[bond.i].force.y, fij.y);
    atomicAdd(&atoms[bond.i].force.z, fij.z);

    atomicAdd(&atoms[bond.j].force.x, -fij.x);
    atomicAdd(&atoms[bond.j].force.y, -fij.y);
    atomicAdd(&atoms[bond.j].force.z, -fij.z);

    // Energy
    float e = bond.k * dr_eq * dr_eq;
    atomicAdd(&energy->bond_energy, e);
}

// ============================================================================
// Kernel: Lennard-Jones forces (non-bonded)
// ============================================================================

__global__ void compute_lj_forces(
    MDAtom* atoms,
    MDParams params,
    EnergyOutput* energy
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= params.n_atoms || j >= params.n_atoms || i >= j) return;

    MDAtom atom_i = atoms[i];
    MDAtom atom_j = atoms[j];

    // Minimum image convention
    float3 dr = atom_j.position - atom_i.position;
    dr.x -= params.box_x * roundf(dr.x / params.box_x);
    dr.y -= params.box_y * roundf(dr.y / params.box_y);
    dr.z -= params.box_z * roundf(dr.z / params.box_z);

    float r2 = dot(dr, dr);
    float r = sqrtf(r2);

    if (r > params.cutoff || r < 1e-6f) return;

    // LJ parameters
    float sigma_ij = 0.5f * (atom_i.sigma + atom_j.sigma);
    float epsilon_ij = sqrtf(atom_i.epsilon * atom_j.epsilon) * params.epsilon_lj;

    float sigma_r = sigma_ij / r;
    float sigma_r6 = powf(sigma_r, 6.0f);
    float sigma_r12 = sigma_r6 * sigma_r6;

    // Force
    float force_mag = 12.0f * epsilon_ij * (sigma_r12 - sigma_r6) / r2;
    float3 fij = force_mag * dr / r;

    // Apply forces
    atomicAdd(&atoms[i].force.x, fij.x);
    atomicAdd(&atoms[i].force.y, fij.y);
    atomicAdd(&atoms[i].force.z, fij.z);

    atomicAdd(&atoms[j].force.x, -fij.x);
    atomicAdd(&atoms[j].force.y, -fij.y);
    atomicAdd(&atoms[j].force.z, -fij.z);

    // Energy
    float e = 4.0f * epsilon_ij * (sigma_r12 - sigma_r6);
    atomicAdd(&energy->vdw_energy, e);
}

// ============================================================================
// Kernel: Electrostatic forces (Coulomb)
// ============================================================================

__global__ void compute_coulomb_forces(
    MDAtom* atoms,
    MDParams params,
    EnergyOutput* energy
) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= params.n_atoms || j >= params.n_atoms || i >= j) return;

    MDAtom atom_i = atoms[i];
    MDAtom atom_j = atoms[j];

    if (atom_i.charge == 0.0f || atom_j.charge == 0.0f) return;

    // Minimum image
    float3 dr = atom_j.position - atom_i.position;
    dr.x -= params.box_x * roundf(dr.x / params.box_x);
    dr.y -= params.box_y * roundf(dr.y / params.box_y);
    dr.z -= params.box_z * roundf(dr.z / params.box_z);

    float r2 = dot(dr, dr);
    float r = sqrtf(r2);

    if (r > params.cutoff || r < 1e-6f) return;

    // Coulomb
    float qiqj = atom_i.charge * atom_j.charge;
    float e = params.epsilon_coulomb * qiqj / r;
    float3 fij = (e / r2) * dr;

    atomicAdd(&atoms[i].force.x, fij.x);
    atomicAdd(&atoms[i].force.y, fij.y);
    atomicAdd(&atoms[i].force.z, fij.z);

    atomicAdd(&atoms[j].force.x, -fij.x);
    atomicAdd(&atoms[j].force.y, -fij.y);
    atomicAdd(&atoms[j].force.z, -fij.z);

    atomicAdd(&energy->electrostatic_energy, e);
}

// ============================================================================
// Kernel: Velocity Verlet integration with Langevin thermostat
// ============================================================================

__global__ void integrate_langevin(
    MDAtom* atoms,
    MDParams params,
    uint seed
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.n_atoms) return;

    MDAtom atom = atoms[idx];
    float inv_mass = 1.0f / atom.mass;

    // Random number generator (LCG)
    uint rng_state = seed + idx * 7919u;

    auto rng = [&rng_state]() {
        rng_state = rng_state * 1664525u + 1013904223u;
        return (float)rng_state / (float)0xFFFFFFFFu;
    };

    // Random force (Box-Muller)
    float u1 = fmaxf(rng(), 1e-10f);
    float u2 = rng();
    float rand_normal = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);

    float kBT = 0.001987f * params.temperature;  // kcal/mol
    float gamma = 1.0f;  // ps^-1

    float3 random_force;
    random_force.x = rand_normal * sqrtf(2.0f * gamma * params.dt * kBT * inv_mass);
    random_force.y = rng() * 2.0f - 1.0f;
    random_force.z = rng() * 2.0f - 1.0f;

    // Velocity Verlet with Langevin
    float3 v_half = atom.velocity;
    v_half += (atom.force * inv_mass - gamma * atom.velocity * 0.5f) * params.dt;
    v_half += random_force;

    // Damping
    float damp = expf(-gamma * params.dt * 0.5f);
    v_half *= damp;

    // Position update
    atom.position += v_half * params.dt;

    // Periodic boundary conditions
    if (params.box_x > 0.0f) {
        atom.position.x = fmodf(atom.position.x, params.box_x);
        if (atom.position.x < 0.0f) atom.position.x += params.box_x;
    }
    if (params.box_y > 0.0f) {
        atom.position.y = fmodf(atom.position.y, params.box_y);
        if (atom.position.y < 0.0f) atom.position.y += params.box_y;
    }
    if (params.box_z > 0.0f) {
        atom.position.z = fmodf(atom.position.z, params.box_z);
        if (atom.position.z < 0.0f) atom.position.z += params.box_z;
    }

    // Store
    atom.velocity = v_half;
    atom.force = make_float3(0.0f, 0.0f, 0.0f);

    atoms[idx] = atom;
}

// ============================================================================
// Kernel: Compute kinetic energy
// ============================================================================

__global__ void compute_kinetic_energy(
    MDAtom* atoms,
    uint n_atoms,
    EnergyOutput* energy
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_atoms) return;

    float3 v = atoms[idx].velocity;
    float ke = 0.5f * atoms[idx].mass * dot(v, v);

    atomicAdd(&energy->kinetic_energy, ke);
}

#endif // CUDA_MD_CUH
