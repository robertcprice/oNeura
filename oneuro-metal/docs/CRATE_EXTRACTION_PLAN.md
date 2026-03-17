# Crate Extraction Plan: oneuro-metal → oneura-terrarium

## Goal

Extract the terrarium simulation into a standalone `oneura-terrarium` crate that:
- Compiles independently without GPU/Metal dependencies
- Has zero dependencies on the neural simulation or Python bindings
- Can be published to crates.io as a reusable biology simulation library

## Extraction Boundary

### Moves to `oneura-terrarium`

#### Core Simulation
| Module | Lines | Dependencies | Notes |
|--------|-------|-------------|-------|
| `terrarium_world.rs` | 3207 | terrarium, soil_broad, soil_uptake, drosophila | Main world struct + stepping |
| `terrarium.rs` | ~500 | ndarray | Substrate chemistry grid |
| `terrarium_field.rs` | ~200 | - | Sensory field types |
| `soil_broad.rs` | ~310 | - | Biogeochemistry stepping |
| `soil_uptake.rs` | ~200 | - | Plant root resource extraction |
| `drosophila.rs` | ~800 | - | Fly brain + body simulation |
| `drosophila_population.rs` | 1133 | drosophila | Sharpe-Schoolfield lifecycle |
| `fly_metabolism.rs` | 516 | - | 7-pool Michaelis-Menten |
| `plant_competition.rs` | 615 | - | Beer-Lambert + root splitting |
| `soil_fauna.rs` | 921 | - | Earthworm + nematode guilds |
| `organism_metabolism.rs` | 49 | - | Universal metabolic trait |
| `field_coupling.rs` | 254 | - | 2D/3D Gaussian helpers |
| `seed_cellular.rs` | 550 | - | Seed metabolism |
| `cross_scale_coupling.rs` | ~300 | whole_cell | Explicit microbe bridge |

#### Terrarium World Submodules
| Module | Lines | Status | Notes |
|--------|-------|--------|-------|
| `terrarium_world/genotype.rs` | 547 | Unconditional | Gene weights, PublicSecondaryBanks |
| `terrarium_world/packet.rs` | 356 | Unconditional | GenotypePacketPopulation |
| `terrarium_world/calibrator.rs` | 242 | Unconditional | MD→reaction rate bridge |
| `terrarium_world/flora.rs` | 761 | Unconditional | Advanced plant stepping |
| `terrarium_world/soil.rs` | 799 | Feature-gated | Needs guild infrastructure |
| `terrarium_world/biomechanics.rs` | 561 | Feature-gated | Wind/pose biomechanics |
| `terrarium_world/snapshot.rs` | 1284 | Feature-gated | Full snapshot with guilds |
| `terrarium_world/explicit_microbe_impl.rs` | 2107 | Feature-gated | Whole-cell integration |

#### Evolution Engine
| Module | Lines | Dependencies | Notes |
|--------|-------|-------------|-------|
| `terrarium_evolve.rs` | 3400+ | terrarium_world | NSGA-II + WorldGenome |

#### Stochastic Expression
| Module | Lines | Dependencies | Notes |
|--------|-------|-------------|-------|
| `stochastic_expression.rs` | 400 | - | Gillespie tau-leaping |

### Stays in `oneuro-metal`

| Module | Reason |
|--------|--------|
| `whole_cell.rs` + submodules | Heavy GPU/Metal dependency |
| `whole_cell_quantum_runtime.rs` | Quantum chemistry pipeline |
| `atomistic_chemistry.rs` | PDB/mmCIF parsing, molecule graph |
| `structure_ingest.rs` | Biomolecular file parsing |
| `gpu/` directory | Metal compute shaders |
| `terrarium_substrate.metal` | GPU substrate chemistry |
| `neural.rs`, `neural_*.rs` | Neural simulation |
| All render modules | Bevy/minifb rendering pipeline |

### Shared Interface (trait-based)

The `cross_scale_coupling.rs` module bridges terrarium_world ↔ whole_cell.
After extraction:
- `oneura-terrarium` defines a `WholeCellBridge` trait
- `oneuro-metal` implements it with real WholeCellSimulator
- Standalone terrarium uses a no-op stub implementation

## Dependency Graph

```
oneura-terrarium (new crate)
├── rand, rand_distr
├── ndarray
├── serde, serde_json
└── (no Metal, no GPU, no Bevy, no PyO3)

oneuro-metal (existing)
├── oneura-terrarium (as dependency)
├── metal, objc (GPU)
├── bevy (3D rendering, optional)
├── pyo3 (Python bindings, optional)
└── minifb (software rendering)
```

## Migration Steps

### Phase A: Extract Pure Modules (no dependency changes)
1. Copy `organism_metabolism.rs`, `field_coupling.rs`, `stochastic_expression.rs` → oneura-terrarium
2. Copy `fly_metabolism.rs`, `soil_fauna.rs`, `plant_competition.rs` → oneura-terrarium
3. Copy `seed_cellular.rs`, `drosophila_population.rs` → oneura-terrarium
4. Verify: `cargo check` in oneura-terrarium

### Phase B: Extract Core Types
1. Move `TerrariumSpecies`, substrate grid types → oneura-terrarium
2. Move `TerrariumWorld`, `TerrariumWorldConfig` → oneura-terrarium
3. Define `WholeCellBridge` trait in oneura-terrarium
4. Update oneuro-metal to depend on oneura-terrarium

### Phase C: Extract Evolution Engine
1. Move `terrarium_evolve.rs` → oneura-terrarium binary
2. Move `WorldGenome` + fitness evaluation → oneura-terrarium
3. Verify: `cargo build --bin terrarium_evolve` in oneura-terrarium

### Phase D: Wire Advanced Modules
1. Move feature-gated submodules behind `advanced` feature
2. Move render modules behind `render` feature
3. Verify: all features compile independently

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Circular dependency | WholeCellBridge trait breaks the cycle |
| Borrow checker issues in split modules | Index-based access patterns already used |
| Test breakage | Run full 211-test regression after each phase |
| Build time increase | Workspace-level caching, shared target dir |
| GPU substrate sync | Keep terrarium_substrate.metal in oneuro-metal |

## Success Criteria

- [ ] `oneura-terrarium` compiles standalone with 0 errors
- [ ] All 211 regression tests pass
- [ ] `terrarium_evolve` binary builds from oneura-terrarium
- [ ] `oneuro-metal` depends on oneura-terrarium (no duplication)
- [ ] No GPU/Metal/Bevy dependencies in oneura-terrarium
