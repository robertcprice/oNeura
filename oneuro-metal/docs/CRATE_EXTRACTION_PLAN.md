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
| `terrarium_world/soil.rs` | 799 | **Unconditional** | Guild-aware soil stepping |
| `terrarium_world/biomechanics.rs` | ~14 | **Unconditional** | Wind/pose biomechanics (stubbed) |
| `terrarium_world/snapshot.rs` | 1285 | **Unconditional** | Full snapshot with guilds |
| `terrarium_world/explicit_microbe_impl.rs` | ~220 | **Unconditional** | Whole-cell integration (stubbed) |

#### Evolution Engine
| Module | Lines | Dependencies | Notes |
|--------|-------|-------------|-------|
| `terrarium_evolve.rs` | 3400+ | terrarium_world | NSGA-II + WorldGenome |

#### Stochastic Expression
| Module | Lines | Dependencies | Notes |
|--------|-------|-------------|-------|
| `stochastic_expression.rs` | 400 | - | Gillespie tau-leaping |

#### Whole-Cell & Quantum
| Module | Lines | Dependencies | Notes |
|--------|-------|-------------|-------|
| `whole_cell.rs` + submodules (10) | ~19,780 | whole_cell_data | Full E. coli simulator |
| `whole_cell_data.rs` | 11,445 | - | Biological constants + types |
| `whole_cell_submodels.rs` | 5,300 | whole_cell_data | Sub-cellular models |
| `whole_cell_quantum_runtime.rs` | 8,052 | subatomic_quantum | Quantum chemistry bridge |
| `subatomic_quantum.rs` | 2,680 | - | Hartree-Fock / DFT SCF solver |
| `atomistic_chemistry.rs` | 900+ | - | Molecular topology |
| `atomistic_topology.rs` | ~300 | atomistic_chemistry | Topology utilities |
| `structure_ingest.rs` | 1,200+ | atomistic_chemistry | PDB/mmCIF parser |
| `molecular_dynamics.rs` | ~800 | atomistic_chemistry | Verlet integrator + TIP3P |

#### Advanced Biology
| Module | Lines | Dependencies | Notes |
|--------|-------|-------------|-------|
| `substrate_coupling.rs` | 611 | soil_broad | Substrate ↔ genotype coupling |
| `ecology_events.rs` | ~300 | - | Ecology event types |
| `ecology_fields.rs` | ~250 | - | Ecology field helpers |
| `plant_cellular.rs` | ~400 | - | Plant tissue metabolism |
| `plant_organism.rs` | ~350 | plant_cellular | Plant physiology |
| `drug_discovery.rs` | ~600 | - | Persister cell + drug protocols |
| `enzyme_engineering.rs` | ~500 | - | Directed enzyme evolution |
| `enzyme_probes.rs` | ~400 | enzyme_engineering | Fluorescent probes |
| `probe_coupling.rs` | ~300 | enzyme_probes | Probe ↔ substrate coupling |
| `enzyme_evolution.rs` | ~400 | enzyme_engineering | Enzyme fitness landscapes |
| `bioremediation.rs` | ~500 | enzyme_engineering | Bioremediation simulation |
| `metabolic_flux.rs` | 1,725 | whole_cell_data | Flux balance analysis |
| `phylogenetic_tracker.rs` | ~300 | - | Lineage tracking |
| `climate_scenarios.rs` | ~400 | - | Climate forcing presets |
| `horizontal_gene_transfer.rs` | 1,513 | - | HGT / conjugation |
| `biofilm_dynamics.rs` | 1,456 | - | Biofilm + quorum sensing |
| `guild_latent.rs` | 463 | - | Latent guild bank evolution |
| `microbiome_assembly.rs` | 1,642 | - | Community assembly rules |
| `resistance_evolution.rs` | 1,754 | drug_discovery | AMR evolution |
| `eco_evolutionary_feedback.rs` | 1,686 | - | Eco-evo coupling |
| `nutrient_cycling.rs` | 1,414 | - | Biogeochemical cycling |
| `population_genetics.rs` | 1,514 | - | Pop-gen models |
| `ecosystem_integration.rs` | ~800 | terrarium_world | Integration scenarios |
| `molecular_atmosphere.rs` | ~300 | - | Gas exchange |

**Estimated total**: ~85,000 lines (59% of codebase)

### Stays in `oneuro-metal`

| Module | Reason |
|--------|--------|
| `neuron_arrays.rs`, `synapse_arrays.rs` | Neural simulation core |
| `spike_propagation.rs`, `stdp.rs` | Neural spike routing + plasticity |
| `brain_regions.rs`, `consciousness.rs` | Cortical models + IIT/GNW |
| `network.rs`, `retina.rs` | Network topology + visual processing |
| `doom_brain.rs`, `dishbrain_pong.rs` | Embodied agents |
| `gpu/` directory | Metal compute shaders |
| `terrarium_substrate.metal` | GPU substrate chemistry |
| `python.rs` | PyO3 bindings (bridges both crates) |
| `constants.rs`, `types.rs` | Shared (re-exported by both) |
| Render modules (4 files) | Need `terrarium_render` infrastructure |

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
| Test breakage | Run full 216-test regression after each phase |
| Build time increase | Workspace-level caching, shared target dir |
| GPU substrate sync | Keep terrarium_substrate.metal in oneuro-metal |

## Success Criteria

- [ ] `oneura-terrarium` compiles standalone with 0 errors
- [ ] All 216+ regression tests pass
- [ ] `terrarium_evolve` binary builds from oneura-terrarium
- [ ] `oneuro-metal` depends on oneura-terrarium (no duplication)
- [ ] No GPU/Metal/Bevy dependencies in oneura-terrarium
