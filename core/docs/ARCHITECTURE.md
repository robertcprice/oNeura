# oNeura Terrarium: System Architecture

> **Codebase**: 148,000+ lines Rust | 161 source files | 825+ tests | 23 binaries
> **Last updated**: 2026-03-17

---

## 1. System Overview

oNeura Terrarium is a multi-scale biological simulation framework that integrates
seven scales of biological organization in a single bidirectionally coupled system:

```
Scale 7: EVOLUTIONARY    terrarium_evolve.rs         NSGA-II multi-objective optimization
    ↕                                                 Genome → World → Fitness → Selection
Scale 6: ECOLOGICAL       terrarium_world.rs          Spatial ecosystem (plants, flies, soil, water)
    ↕                     + 8 submodules              Guild-aware biogeochemistry, competition
Scale 5: ORGANISMAL       drosophila.rs               Individual organism physiology
    ↕                     fly_metabolism.rs            Energy budgets, behavior, lifecycle
Scale 4: STOCHASTIC       stochastic_expression.rs    Gillespie tau-leaping gene noise
    ↕                                                 Telegraph promoter model
Scale 3: CELLULAR         whole_cell.rs               Full E. coli WholeCellSimulator
    ↕                     + 10 submodules             Metabolism, chromosome, membrane, division
Scale 2: ATOMISTIC        molecular_dynamics.rs       Verlet integrator, TIP3P water
    ↕                     atomistic_chemistry.rs      Molecular topology from PDB/mmCIF
Scale 1: QUANTUM          subatomic_quantum.rs        Hartree-Fock SCF, Eyring TST
                                                      Reaction barrier → rate constant
```

**Key design principle**: Each scale uses published, peer-reviewed kinetic models.
Behaviors emerge from the simulation rather than being programmed.

---

## 2. Module Dependency Graph

### Scale 1: Quantum Chemistry
```
subatomic_quantum.rs (2,680 lines)
  └── Hartree-Fock / DFT SCF solver
  └── Eyring transition state theory
  └── No external dependencies
```

### Scale 2: Atomistic
```
atomistic_chemistry.rs (900+ lines)
  ├── EmbeddedMolecule topology
  ├── Molecular graph from PDB/mmCIF
  └── Van der Waals radii, covalent bonds

atomistic_topology.rs (~300 lines)
  └── depends on: atomistic_chemistry

structure_ingest.rs (1,200+ lines)
  ├── PDB file parser (19 tests)
  ├── mmCIF file parser
  └── depends on: atomistic_chemistry

molecular_dynamics.rs (~800 lines)
  ├── Verlet velocity integrator
  ├── TIP3P water model (O-H bonds, k=553, r₀=0.9572 Å)
  └── depends on: atomistic_chemistry
```

### Scale 3: Cellular Metabolism
```
whole_cell_data.rs (11,445 lines)
  └── Biological constants, type definitions
  └── 13 WholeCellBulkField variants (ATP..CoA)
  └── No dependencies (foundational)

whole_cell.rs (7,780 lines)
  ├── WholeCellSimulator (main struct)
  ├── Submodules:
  │   ├── chromosome.rs        — DNA replication + segregation
  │   ├── complex_assembly.rs  — Ribosome + polysome assembly
  │   ├── expression.rs        — Transcription + translation
  │   ├── initialization.rs    — Cell state initialization
  │   ├── local_chemistry.rs   — Compartment chemistry
  │   ├── membrane.rs          — Lipid bilayer dynamics
  │   ├── rdme_drive.rs        — Reaction-diffusion master equation
  │   ├── spatial.rs           — 3D lattice geometry
  │   ├── stochastic_expression.rs — Gillespie tau-leaping
  │   └── tests.rs             — 64+ unit tests
  └── depends on: whole_cell_data

whole_cell_submodels.rs (5,300 lines)
  └── depends on: whole_cell_data

whole_cell_quantum_runtime.rs (8,052 lines)
  ├── Quantum chemistry → reaction rate bridge
  ├── Arrhenius/Eyring calibration
  └── depends on: subatomic_quantum, atomistic_chemistry, whole_cell
```

### Scale 4: Stochastic Gene Expression
```
stochastic_expression.rs (400 lines, inside whole_cell/)
  ├── Gillespie tau-leaping algorithm
  ├── Telegraph promoter model
  ├── Fano factor validation (Taniguchi et al. 2010)
  └── Opt-in via WholeCellSimulator::step()
```

### Scale 5: Organismal Physiology
```
organism_metabolism.rs (49 lines)
  └── Universal OrganismMetabolism trait

drosophila.rs (2,418 lines)
  └── Flight dynamics, foraging behavior, body state

drosophila_population.rs (1,133 lines)
  ├── Sharpe-Schoolfield thermal development
  ├── Egg → Larva → Pupa → Adult lifecycle
  └── depends on: drosophila

fly_metabolism.rs (516 lines)
  ├── 7-pool Michaelis-Menten model
  ├── crop → hemolymph → trehalose → muscle ATP
  └── depends on: organism_metabolism

plant_cellular.rs (~400 lines)
  └── Per-tissue metabolite pools (ATP, glucose, starch, water, nitrate)

plant_organism.rs (~350 lines)
  └── depends on: plant_cellular

plant_competition.rs (615 lines)
  ├── Beer-Lambert canopy shading
  ├── Asymmetric root nutrient splitting
  └── No dependencies

seed_cellular.rs (550 lines)
  └── Seed germination metabolism
```

### Scale 6: Ecological Community
```
terrarium.rs (855 lines)
  ├── BatchedAtomTerrarium (GPU substrate grid)
  ├── 14 TerrariumSpecies: Glucose, Oxygen, CO2, Nitrate, Ammonium,
  │   Phosphate, Water, OrganicC, Cellulose, Chitin, Siderophore,
  │   Quorum, AtpFlux, Proton
  └── PDE diffusion on 2D/3D voxel grid

terrarium_world.rs (3,085 lines)
  ├── TerrariumWorld (main simulation struct)
  ├── step_frame() — multi-rate simulation loop
  ├── Submodules (all unconditional):
  │   ├── genotype.rs (547)     — Secondary genotype types
  │   ├── packet.rs (356)       — Genotype packet populations
  │   ├── calibrator.rs (242)   — MD→reaction rate bridge
  │   ├── flora.rs (761)        — Advanced plant stepping
  │   ├── soil.rs (799)         — Guild-aware soil stepping
  │   ├── snapshot.rs (1,285)   — Full world snapshot
  │   ├── biomechanics.rs (~14) — Wind/pose (stubbed)
  │   └── explicit_microbe_impl.rs (~220) — Microbe lifecycle (stubbed)
  └── depends on: terrarium, soil_broad, drosophila, whole_cell

soil_broad.rs (309 lines)
  └── 3-pool C/N/P biogeochemistry (Parton 1988)

soil_fauna.rs (921 lines)
  ├── Earthworm bioturbation
  ├── Nematode Lotka-Volterra predation
  └── N mineralization coupling

substrate_coupling.rs (611 lines)
  └── Substrate ↔ genotype coupling functions

guild_latent.rs (463 lines)
  ├── LatentGuildState, LatentGuildConfig, LatentGuildResult
  └── step_latent_guild_banks() — microbial guild evolution

climate_scenarios.rs (~400 lines)
  └── Temperature/precipitation forcing presets

cross_scale_coupling.rs (~500 lines)
  └── Bridge functions between scales
```

### Scale 7: Evolutionary Optimization
```
terrarium_evolve.rs (5,091 lines)
  ├── WorldGenome (18-parameter genome)
  ├── NSGA-II multi-objective optimizer
  ├── 6 modes: Standard, Pareto, Stress, Coevolution, Bet-Hedging, GRN
  ├── FitnessLandscape scanner
  ├── Telemetry export (CSV, Prometheus)
  └── depends on: terrarium_world
```

### Advanced Biology Modules
```
drug_discovery.rs (~600)            — Persister cell dynamics, drug protocols
resistance_evolution.rs (1,754)     — AMR mutation, fitness costs, plasmid transfer
enzyme_engineering.rs (~500)        — Directed evolution, saturation mutagenesis
enzyme_probes.rs (~400)             — Fluorescent enzyme probes
probe_coupling.rs (~300)            — Probe ↔ substrate coupling
enzyme_evolution.rs (~400)          — Enzyme fitness landscapes
bioremediation.rs (~500)            — Pollutant degradation simulation
metabolic_flux.rs (1,725)           — FBA-style flux balance analysis
horizontal_gene_transfer.rs (1,513) — Conjugation, transformation, transduction
biofilm_dynamics.rs (1,456)         — Biofilm formation, quorum sensing, EPS
microbiome_assembly.rs (1,642)      — Community assembly, priority effects
eco_evolutionary_feedback.rs (1,686)— Eco-evo coupling dynamics
nutrient_cycling.rs (1,414)         — C/N/P biogeochemical cycling
population_genetics.rs (1,514)      — Wright-Fisher, Hardy-Weinberg, drift
phylogenetic_tracker.rs (~300)      — Lineage tracking, speciation detection
ecosystem_integration.rs (~800)     — Cross-module integration scenarios
molecular_atmosphere.rs (~300)      — Gas exchange
```

---

## 3. Data Flow Between Scales

### Upward Causation (Micro → Macro)

```
Quantum (SCF energies)
    │ Eyring TST: ΔG‡ → k = (kT/h)·exp(-ΔG‡/RT)
    ▼
Atomistic (MD trajectories)
    │ Arrhenius calibration: ln(k) vs 1/T → Ea, A
    ▼
Cellular (WholeCellSnapshot)
    │ Metabolite concentrations → organism energy state
    ▼
Stochastic (Gillespie noise)
    │ Gene expression variance → phenotypic heterogeneity
    ▼
Organismal (body state, fitness)
    │ Energy, growth, reproduction → population dynamics
    ▼
Ecological (community structure)
    │ Species abundances, diversity → fitness evaluation
    ▼
Evolutionary (NSGA-II selection)
    │ Fitness → genome selection → next generation
```

### Downward Causation (Macro → Micro)

```
Evolutionary (genome parameters)
    │ WorldGenome.seed_world() → initial conditions
    ▼
Ecological (environmental state)
    │ Temperature, moisture, nutrients → organism context
    ▼
Organismal (metabolic demand)
    │ Activity level, growth → cellular resource demand
    ▼
Cellular (metabolic state)
    │ ATP/substrate levels → gene expression regulation
    ▼
Stochastic (promoter state)
    │ Transcription factor binding → expression noise
    ▼
Atomistic (local chemistry)
    │ Substrate concentrations → reaction conditions
    ▼
Quantum (reaction barriers)
    │ Temperature, pH → barrier height modulation
```

### Cross-Scale Bridge Modules

| Bridge | From → To | Mechanism |
|--------|-----------|-----------|
| `whole_cell_quantum_runtime.rs` | Quantum → Cellular | Hartree-Fock energies → Eyring rate constants |
| `calibrator.rs` | Atomistic → Ecological | MD trajectory → Arrhenius parameters → soil kinetics |
| `cross_scale_coupling.rs` | Cellular ↔ Organismal | WholeCellSnapshot → metabolite pools |
| `substrate_coupling.rs` | Ecological ↔ Genotype | Substrate chemistry ↔ microbial genotype traits |
| `explicit_microbe_impl.rs` | Cellular ↔ Ecological | Individual WCS ↔ terrarium world state |
| `terrarium_evolve.rs` | Ecological → Evolutionary | Fitness evaluation → NSGA-II selection |

---

## 4. Key Structs

### TerrariumWorld (`terrarium_world.rs`)
The central simulation struct. Contains:
- `config: TerrariumWorldConfig` — grid dimensions, physics parameters
- `substrate: BatchedAtomTerrarium` — GPU substrate chemistry grid
- `plants: Vec<TerrariumPlant>` — individual plant entities
- `flies: Vec<DrosophilaSim>` — individual fly entities
- `fly_metabolisms: Vec<FlyMetabolismState>` — parallel metabolic state per fly
- `fly_pop: DrosophilaPopulation` — population-level lifecycle tracking
- `seeds`, `fruits`, `water` — other entity collections
- `explicit_microbes: Vec<TerrariumExplicitMicrobe>` — individually-tracked microbes
- Guild fields: `microbial_secondary`, `nitrifier_secondary`, `denitrifier_secondary`
- Substrate overlays: `moisture_field`, `temperature_field`, `ownership_map`

### BatchedAtomTerrarium (`terrarium.rs`)
GPU-accelerated substrate chemistry grid with 14 chemical species per cell.
Uses Metal compute shaders for PDE diffusion. 5 persistent GPU buffers.

### WholeCellSimulator (`whole_cell.rs`)
Full E. coli (Syn3A) cellular simulator:
- RDME lattice (24×24×12 voxels)
- 13 bulk metabolite pools (ATP, ADP, glucose, amino acids, nucleotides, etc.)
- Chromosome replication + segregation
- Membrane lipid dynamics
- Gillespie stochastic gene expression (opt-in)

### WorldGenome (`terrarium_evolve.rs`)
18-parameter genome encoding ecosystem configuration. Used by NSGA-II to evolve
optimal ecosystem configurations across 7 simultaneous fitness objectives.

### TerrariumWorldSnapshot (`terrarium_world/snapshot.rs`)
Complete ecosystem state capture with 200+ fields covering every subsystem.
Implements `Default` — test code uses `..Default::default()`.

---

## 5. Simulation Loop

`TerrariumWorld::step_frame()` implements multi-rate scheduling:

```
EVERY STEP (dt):
  ├── GPU substrate chemistry PDE diffusion
  ├── Fly metabolism (7-pool MM per individual)
  └── Fly population lifecycle (Sharpe-Schoolfield)

EVERY 5 STEPS:
  └── Plant competition (Beer-Lambert canopy + root splitting)

EVERY 10 STEPS:
  ├── Soil fauna (earthworm bioturbation + nematode predation)
  ├── Explicit microbe stepping (individual WCS)
  └── Broad soil biogeochemistry (guild-aware C/N/P cycling)
```

**Rationale**: Chemistry diffuses fast (small dt needed), plants grow slowly,
soil fauna populations change on longer timescales.

---

## 6. Extension Guide

### Adding a New Chemical Species
1. Add variant to `TerrariumSpecies` enum in `terrarium.rs`
2. Add corresponding field in Metal shader struct (`terrarium_substrate.metal`)
3. Add diffusion coefficient in `substrate_diffusion_coefficient()`
4. Update `snapshot.rs` to capture the new species
5. Add test validating diffusion behavior

### Adding a New Organism Type
1. Create `src/new_organism.rs` with body state + stepping logic
2. Add `pub mod new_organism;` to `lib.rs`
3. Add `Vec<NewOrganism>` field to `TerrariumWorld`
4. Wire `step_new_organism()` into `step_frame()` at appropriate rate
5. Add rendering in `terrarium_3d/` and `terrarium_zoom`

### Adding a New Evolution Objective
1. Add evaluation function in `terrarium_evolve.rs`
2. Add to `OBJECTIVES` array in Pareto evaluation
3. Add CLI flag handling

### Adding a New CLI Binary
1. Create `src/bin/new_binary.rs`
2. Add `[[bin]]` to `Cargo.toml`
3. Build: `cargo build --profile fast --no-default-features --bin new_binary`

---

## 7. Testing Architecture

### Test Coverage by Scale

| Scale | Module | Tests | What's Validated |
|-------|--------|-------|-----------------|
| Quantum | `subatomic_quantum` | 64 (6 ignored) | SCF convergence, Eyring rates |
| Atomistic | `atomistic_chemistry` | 12 | Molecular topology, bond detection |
| Atomistic | `structure_ingest` | 19 | PDB/mmCIF parsing |
| Cellular | `whole_cell/tests` | 64+ | Metabolite dynamics, chromosome, membrane |
| Stochastic | `stochastic_expression` | 10 | Gillespie correctness, Fano factors |
| Organismal | `fly_metabolism` | 10 | MM kinetics, energy conservation |
| Organismal | `drosophila_population` | 7 | Sharpe-Schoolfield thermal response |
| Organismal | `plant_competition` | 10 | Beer-Lambert, root competition |
| Ecological | `soil_fauna` | 8 | Earthworm, nematode dynamics |
| Ecological | `ecosystem_integration` | 15 | Cross-module scenarios |
| Ecological | `guild_latent` | 5+ | Latent guild evolution |
| Evolutionary | `terrarium_evolve` | 49 | NSGA-II, Pareto, stress, coevolution |
| Synbio | `enzyme_*`, `drug_*` | 30+ | Enzyme kinetics, drug protocols |
| Cross-scale | `cross_scale_coupling` | 6 | Scale bridging functions |

### Running Tests

```bash
# Regression suite (216 tests, ~32s)
cargo test --no-default-features --lib -- <filter>

# Full suite (825+ tests, ~2-3 min)
cargo test --no-default-features --lib

# Single module with output
cargo test --no-default-features --lib -- fly_metabolism --nocapture
```

---

## 8. Build Profiles

| Profile | Use Case | LTO | Opt Level |
|---------|----------|-----|-----------|
| `dev` | Development + testing | None | 2 |
| `fast` | Demo binaries (16GB Mac safe) | Thin | 2 |
| `release` | Production | Thin | 3 |

### Memory Considerations (16GB Mac)
- `CARGO_BUILD_JOBS=1` prevents OOM during linking
- `pkill -9 -f rustc` before rebuilds clears stale processes
- `CARGO_TARGET_DIR=/tmp/oneuro-build` isolates concurrent sessions
- Thin LTO saves 2-4x linker memory vs Fat LTO

---

## 9. Feature Flags

| Feature | What It Gates | Status |
|---------|--------------|--------|
| (default: none) | Full CPU simulation | All builds |
| `terrarium_render` | 4 Metal-native render modules | Needs crate infrastructure |
| `web` | REST API server | Optional |
| `python` | PyO3 bindings | Optional (maturin) |

All core simulation, evolution, and biology modules compile with `--no-default-features`.
