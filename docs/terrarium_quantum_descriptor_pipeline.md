# Terrarium Quantum Descriptor Pipeline

Date: 2026-03-20
Codebase: `/Users/bobbyprice/projects/oNeura`

## Purpose

This document describes the current terrarium molecular descriptor path that sits between canonical chemistry assets and higher-level geochemistry, reaction, and rendering code.

The point of this layer is simple:

- load canonical terrarium topology from data assets
- generate representative embedded structures with a species-agnostic embedder
- derive reusable quantum descriptors once per canonical species
- feed those descriptors into terrarium thermodynamics instead of repeating formula-only guesses

Companion status/handoff document:

- [docs/terrarium_ab_initio_status_2026-03-20.md](/Users/bobbyprice/projects/oNeura/docs/terrarium_ab_initio_status_2026-03-20.md)

## Current Runtime Path

### 1. Canonical topology assets

Source:

- [core/specs/terrarium_molecular_assets.json](/Users/bobbyprice/projects/oNeura/core/specs/terrarium_molecular_assets.json)

These assets define:

- terrarium species id
- atom list
- formal charges
- bond graph
- whether the species belongs on the normal runtime `quantum_fast_path`

The assets are authoritative for molecular topology. Species-specific embedded coordinates must not be hardcoded in Rust.

### 2. Generic embedding

Source:

- [core/src/terrarium/inventory_species_registry.rs](/Users/bobbyprice/projects/oNeura/core/src/terrarium/inventory_species_registry.rs)

`EmbeddedMolecule` coordinates are generated from topology by a generic graph embedder that uses:

- bond graph connectivity
- bond order
- periodic-table covalent radii
- weak charge-aware relaxation

This is intentionally species-agnostic.

### 3. Fast-path quantum descriptors

Source:

- [core/src/terrarium/inventory_species_registry.rs](/Users/bobbyprice/projects/oNeura/core/src/terrarium/inventory_species_registry.rs)

For small asset-backed species on the `quantum_fast_path`, the runtime derives and caches:

- ground-state energy
- energy per atom
- dipole magnitude
- effective charge magnitude
- charge span
- mean LDA exchange potential
- frontier occupancy fraction

The cache now has three useful properties:

- a derived precomputed cache asset can prime the runtime before any live solve
- embedded molecules are cached by species
- fast-path descriptors are parallel-warmed on CPU with `rayon`
- snapshots can be exported as either per-species JSON entries or a row-major tensor JSON artifact

Public terrarium hooks:

- `prime_terrarium_quantum_descriptor_cache_from_json(...)`
- `warm_terrarium_quantum_descriptor_cache()`
- `terrarium_quantum_descriptor_cache_json_pretty()`
- `terrarium_quantum_descriptor_tensor_json_pretty()`

CLI materializer:

- `cargo run -p oneura-cli --bin terrarium_descriptor_cache`
- `cargo run -p oneura-cli --bin terrarium_descriptor_cache -- --entries-json -o terrarium_descriptor_cache.json`
- `cargo run -p oneura-cli --bin terrarium_descriptor_cache -- --tensor-json -o terrarium_descriptor_tensor.json`

Bundled derived cache asset:

- [core/specs/terrarium_quantum_descriptor_cache.json](/Users/bobbyprice/projects/oNeura/core/specs/terrarium_quantum_descriptor_cache.json)

The bundled asset is derived data, not a chemistry-authoring table. If canonical topology assets or the fast-path quantum solve changes materially, regenerate it with the CLI instead of editing values by hand.

## Accelerator Contract

The tensor snapshot exists so this layer can move onto more aggressive backends without changing the chemistry asset model again.

Current tensor contract:

- rows: fast-path terrarium species with cached descriptors
- cols: descriptor features
- layout: row-major contiguous `Vec<f32>`

Feature order:

1. `ground_state_energy_ev`
2. `ground_state_energy_per_atom_ev`
3. `dipole_magnitude_e_angstrom`
4. `mean_abs_effective_charge`
5. `charge_span`
6. `mean_lda_exchange_potential_ev`
7. `frontier_occupancy_fraction`

This is the right shape for:

- CPU vectorized/tiled post-processing
- Metal or WGSL upload as a dense structured buffer
- later batch kernels that regress slower motif descriptors from cached lower-level data

## What Is Implemented Today

- Asset-backed molecular topology authority
- Precomputed derived descriptor-cache asset for cold-start priming
- Generic coordinate embedding
- Cached fast-path quantum descriptor derivation
- CPU parallel warmup with `rayon`
- JSON and tensor snapshot materialization
- Geochemistry consuming cached descriptors when available

## What Is Not Implemented Yet

- GPU or shader execution for the quantum descriptor solve itself
- Systolic-array style tiled descriptor kernels over the full terrarium chemistry stack
- Persistent binary cache loading at process start
- Slow motif/regression tier for larger mineral or interfacial structures
- Ab initio voxel-by-voxel runtime chemistry

## Engineering Rules

- No species-specific embedded coordinates in Rust source
- No pretending fallback species are fully emergent
- No claiming GPU acceleration for this descriptor stage until the solver actually runs there
- Any future accelerator path must consume the same canonical asset and tensor contracts instead of reintroducing species-name branches
