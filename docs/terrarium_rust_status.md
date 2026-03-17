# Terrarium Rust Status

## Answer

Yes, the terrarium can become a pure Rust program. The current blocker is no longer missing low-level kernels, and it is no longer the lack of a native world owner either. The remaining blocker is feature-parity ownership: Python still owns the richer demo/runtime shell and some higher-level integration surfaces.

## Native Today

- `oneuro-metal/src/terrarium.rs`
  - Batched atom substrate with CPU and Metal backends.
- `oneuro-metal/src/molecular_atmosphere.rs`
  - Molecular world stepping for odorants, temperature, humidity, source updates, source emission, and wind perturbation.
- `oneuro-metal/src/soil_broad.rs`
  - Broad soil turnover and hydrology update.
- `oneuro-metal/src/soil_uptake.rs`
  - Root-zone resource extraction.
- `oneuro-metal/src/plant_cellular.rs`
  - Native plant cellular state stepping.
- `oneuro-metal/src/cellular_metabolism.rs`
  - Native plant cell metabolism.
- `oneuro-metal/src/plant_organism.rs`
  - Native whole-plant physiology.
- `oneuro-metal/src/ecology_fields.rs`
  - Native canopy and root field rebuilds.
- `oneuro-metal/src/ecology_events.rs`
  - Native food and seed event stepping.
- `oneuro-metal/src/terrarium_field.rs`
  - Native sensory lattice for fly/world coupling.
- `oneuro-metal/src/drosophila.rs`
  - Native fly brain/body step for terrarium sensory input.
- `oneuro-metal/src/terrarium_world.rs`
  - Native terrarium owner for broad soil pools, substrate control sync, plant stepping, seed/food bookkeeping, atmosphere stepping, plant/atmosphere gas exchange, and fly stepping.
- `oneuro-metal/examples/native_terrarium_stack.rs`
  - Headless Rust-only example running the native terrarium owner without Python.
- `oneuro-metal/src/bin/terrarium_native.rs`
  - Rust-native terminal loop for stepping and viewing the terrarium without Python.
- `oneuro-metal/src/bin/terrarium_viewer.rs`
  - Rust-native graphical viewer with a live window, top-down field rendering, keyboard controls, and an in-window stats panel driven directly by `TerrariumWorld`.
- `oneuro-3d/src/bin/terrarium_gpu.rs`
  - Rust-native GPU viewer built on Bevy/wgpu that renders the terrarium through the platform graphics backend while driving the same native `TerrariumWorld`, now with packed raw field textures for terrain/soil/canopy/chemistry/odor/gas layers plus live overlay point data for water, plants, fruit, and flies, all colored on-GPU in WGSL.

## Still Python-Owned

- `src/oneuro/worlds/molecular_world.py`
  - Still owns the Python world object used by the current Python demo path.
- `src/oneuro/ecology/terrarium.py`
  - Still owns the richer Python ecology surface used by the current Python demo path.
- `src/oneuro/organisms/rust_drosophila.py`
  - Still acts as the adapter from the Python world into the native fly sim.
- `demos/demo_actual_molecular_terrarium.py`
  - Still owns the richer graphical demo shell and rendering loop.

## What “Pure Rust” Actually Means Here

To remove Python entirely, the project needs a native terrarium owner, not just native kernels:

1. Keep extending the Rust `TerrariumWorld` so it fully covers the behaviors the Python demo still expects.
2. Move the remaining orchestration in `src/oneuro/ecology/terrarium.py` and the current Python demo loop onto that Rust type.
3. Add Rust-native app entry points:
   - headless CLI
   - terminal viewer
   - graphical viewer
4. Treat PyO3 bindings as optional adapters instead of the main runtime path.

## Practical Next Step

The clean path is:

1. Keep the Python demo as a thin viewer while the Rust `TerrariumWorld` becomes authoritative.
2. Keep `terrarium_native` as the Python-free terminal/runtime path while feature parity improves.
3. Keep `terrarium_viewer` as the lightweight native graphical runtime path for inspecting the world without the Python demo shell.
4. Keep `terrarium_gpu` as the current GPU-backed viewer path when a richer render surface is needed.
5. Deepen the GPU render path further with more simulation-aware shader passes once the current visual contract is stable.

That is how the project gets to "no Python at all" honestly, without pretending the current Python shell is gone when it is not.

## Biology Modules (2026-03-16)

All literature-grounded, all tested, all wired into the terrarium step loop:

| Module | Lines | Tests | Status |
|--------|-------|-------|--------|
| `soil_fauna.rs` | 921 | 8 | Wired: earthworm bioturbation + nematode Lotka-Volterra guilds |
| `plant_competition.rs` | 615 | 10 | Wired: Beer-Lambert asymmetric shading + root nutrient splitting |
| `drosophila_population.rs` | 1133 | 7 | Wired: Sharpe-Schoolfield egg-to-adult lifecycle |
| `fly_metabolism.rs` | 516 | 10 | Wired: 7-pool MM crop-to-ATP biochemistry |
| `field_coupling.rs` | 254 | 5 | Utility: 2D/3D Gaussian field deposit + mean-map |
| `seed_cellular.rs` | 550 | 1 | Standalone: seed germination cellular metabolism |
| `organism_metabolism.rs` | 49 | - | Trait: universal metabolic observation surface |

## Whole-Cell Stochastic Gene Expression (2026-03-16)

| Feature | Status |
|---------|--------|
| Gillespie tau-leaping stochastic stepping | Implemented |
| Telegraph promoter model (ON/OFF states) | Implemented |
| Configurable mRNA burst size | Implemented |
| Protein noise with Fano factor > 1 | Implemented |
| Deterministic↔stochastic bidirectional sync | Implemented |
| Wired into WholeCellSimulator::step() | Wired (opt-in) |
| 10 unit tests | All pass |

Located in `oneuro-metal/src/whole_cell/stochastic_expression.rs`.

## Evolution Engine (2026-03-16)

| Feature | Status |
|---------|--------|
| Standard GA (tournament + crossover + mutation) | Working |
| NSGA-II Pareto multi-objective (7 objectives) | Working |
| Stress-test mode (drought + heat spikes) | Working |
| Fly brain coevolution (PSC scale, neural steps) | Working |
| WorldGenome seeding (plants/water/fruits/flies) | Working |
| Telemetry export to JSON | Working |
| Convergence: fitness 62 to 85 over 5 generations | Verified |
| Noise-driven bet-hedging evolution | Working |
| Persister cell antibiotic survival modeling | Working |
| Per-fly phenotypic noise from stochastic expression | Working |
| Biphasic kill curve detection | Working |
| Environmental variability engine (seasons/drought/weather) | Working |
| Spatial heterogeneity zones (6 zone types) | Working |
| Multi-species bet-hedging (plants + microbes) | Working |
| Drug protocol optimizer (single/pulsed/combination) | Working |
| E. coli persistence validation against literature | Working |
| Synthetic biology gene circuit noise designer | Working |
| 3D ASCII isometric terrarium renderer | Working |
| 49 unit tests | All pass |

## 3D ASCII Renderer (2026-03-16)

| Feature | Status |
|---------|--------|
| Isometric projection with Unicode block characters | Working |
| ANSI truecolor gradient terrain rendering | Working |
| Enhanced top-down colored view | Working |
| Split-screen mode (iso + top-down) | Working |
| Real-time at 15-30 FPS in terminal | Working |
| Animated water, plant canopy, flying insects | Working |

Binary: `cargo run --profile fast --no-default-features --bin terrarium_ascii`

## Test Counts (2026-03-16)

136 tests pass in comprehensive regression (0 failures):
- terrarium_evolve: 49
- terrarium_world: 22
- whole_cell: 3
- stochastic_expression: 10
- plant_competition: 10
- fly_metabolism: 10
- soil_fauna: 8
- drosophila_population: 7
- field_coupling: 5
- soil_atmosphere: 4
- substrate_stays_bounded: 4
- organism_metabolism: 4
- guild_activity: 2
- dishbrain_pong: 1
- seed_cellular: 1
