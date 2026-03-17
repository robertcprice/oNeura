# oNeura-Metal

**The world's first GPU-accelerated molecular brain simulator.**

Not a neural network. A complete molecular brain where every behavior — learning, memory, drug response, consciousness, sleep — *emerges* from biochemistry simulated on Apple Silicon GPU.

**148,000+ lines of Rust** | **214+ regression tests** | **177 source files** | **11 standalone binaries**

## What is this?

oNeura-Metal is a biophysical neural engine written in Rust with Metal compute shaders. Every neuron in the simulation is a full molecular model: Hodgkin-Huxley ion channels, 4-compartment calcium dynamics, second messenger cascades (cAMP/PKA/PKC/CaMKII/CREB), gene expression, vesicle release, STDP via receptor trafficking, and Orch-OR quantum consciousness — all running on GPU.

The key insight: **you don't program behaviors.** You simulate biochemistry, and behaviors emerge. Administer Diazepam and GABA-A conductance increases, inhibition rises, firing rates drop. Apply general anesthesia and consciousness metrics collapse by >70%. Run a sleep cycle and memory consolidation happens through hippocampal replay. None of this is hardcoded — it falls out of the molecular simulation.

## Architecture

```
GPU Phase (Metal compute, 1 thread per neuron):
  1. HH gating variables    — Na_v m³h, K_v n⁴, Ca_v m²h
  2. Receptor binding        — Hill equation → AMPA/NMDA/GABA-A/nAChR open fractions
  3. Membrane integration    — 8-channel I_ion + dV/dt + spike detection
  4. Calcium dynamics        — 4-compartment ODE (microdomain/cyto/ER/mito)
  5. Second messengers       — cAMP/PKA/PKC/CaMKII/CREB/MAPK cascades

CPU Phase (serial, only fired neurons — typically 1-5%):
  6. Spike propagation       — vesicle release + PSC injection via CSR graph
  7. STDP                    — receptor trafficking (LTP=insert AMPA, LTD=remove)
  8. Synaptic cleft dynamics — NT degradation/diffusion/reuptake

CPU Interval-Gated (every N steps):
  9. Gene expression         — CREB→c-Fos→BDNF→AMPA (every 10 steps)
  10. Metabolism              — Glycolysis + OxPhos + ATP pools (every 5 steps)
  11. Microtubules            — Orch-OR quantum coherence/collapse (every 10 steps)
  12. Glia                    — Astrocyte/oligodendrocyte/microglia (every 10 steps)
  13. Circadian               — TTFL oscillator + adenosine homeostasis (every step)
```

Apple Silicon unified memory (`StorageModeShared`) means GPU↔CPU sync is zero-copy — the CPU reads the `fired` array directly after command buffer completion with no DMA transfer.

## 16 Molecular Subsystems

| Subsystem | What It Does | Why It Matters |
|-----------|-------------|----------------|
| **HH Ion Channels** | Na_v, K_v, K_leak, Ca_v voltage-gated channels with exact α/β rate functions | Action potentials emerge from channel kinetics, not threshold crossing |
| **Ligand-Gated Channels** | AMPA, NMDA (with Mg²⁺ block), GABA-A, nAChR | Synaptic transmission is molecular, not a weight multiplication |
| **4-Compartment Calcium** | Cytoplasmic/ER/mitochondrial/microdomain with IP3R, RyR, SERCA, MCU, PMCA, NCX | Calcium is THE intracellular signal — triggers everything from vesicle release to gene expression |
| **Second Messengers** | cAMP, PKA, PKC, CaMKII (bistable switch), MAPK/ERK, CREB, IP3/DAG | Long-term potentiation requires kinase cascades, not just Hebbian rules |
| **Gene Expression** | DNA→RNA→Protein pipeline with c-Fos, Arc, BDNF, Zif268 transcription factors | Memory consolidation requires protein synthesis (anisomycin blocks this — testable!) |
| **Metabolism** | Glycolysis + oxidative phosphorylation → ATP pools, O₂/glucose supply | Neurons that run out of ATP become less excitable — natural activity limiter |
| **Vesicle Pools** | Readily releasable / recycling / reserve pools with Ca²⁺-dependent release | Short-term synaptic plasticity (facilitation, depression) emerges from vesicle depletion |
| **STDP** | Spike-timing-dependent plasticity via AMPA receptor trafficking + BCM metaplasticity | Learning rules emerge from molecular dynamics, not parameter-tuned Hebbian learning |
| **Synaptic Cleft** | NT release, enzymatic degradation, diffusion, transporter reuptake | Drug targets: SSRIs block serotonin reuptake, AChE inhibitors block ACh breakdown |
| **Pharmacology** | 7 drugs + general anesthesia with 1-compartment PK (Bateman) + PD (Hill) | Real dose-response curves: Diazepam enhances GABA-A 5x, Ketamine blocks NMDA 90% |
| **Glia** | Astrocyte (glutamate uptake, lactate shuttle), oligodendrocyte (myelination), microglia (synaptic pruning) | Astrocytes regulate excitotoxicity, oligodendrocytes control conduction velocity |
| **Circadian** | TTFL oscillator (BMAL1/PER-CRY) + adenosine homeostasis (two-process sleep model) | Chronopharmacology: drug effects vary with time of day |
| **Microtubules** | Orch-OR quantum coherence/collapse model | Consciousness metric: anesthetics suppress coherence, matching clinical observations |
| **Consciousness** | 7 metrics: Phi (IIT), PCI, causal density, criticality, global workspace, Orch-OR, composite | Quantitative consciousness measurement — composite drops >70% under anesthesia |
| **Brain Regions** | Cortical columns, thalamic nuclei, hippocampus, basal ganglia with anatomical connectivity | Regional architecture enables sleep replay, thalamocortical loops, striatal learning |
| **Extracellular Space** | 3D voxel grid with Fick's law diffusion (GPU shader) | Volume transmission of neuromodulators (DA, 5-HT diffuse through extracellular space) |

## Competitive Landscape

| Capability | oNeura-Metal | NEURON | NEST | Brian2 | CoreNeuron |
|---|---|---|---|---|---|
| HH ion channels on GPU | **Yes** | No | No | No | Yes (partial) |
| Second messenger cascades | **Yes (GPU)** | MOD files (CPU) | No | No | No |
| Gene expression (CREB→BDNF) | **Yes** | No | No | No | No |
| Quantum consciousness (Orch-OR) | **Yes** | No | No | No | No |
| Psychopharmacology (7 drugs) | **Yes** | No | No | No | No |
| Circadian + chronopharm | **Yes** | No | No | No | No |
| 3D extracellular diffusion (GPU) | **Yes** | 1D | No | No | No |
| Consciousness metrics (7) | **Yes** | No | No | No | No |
| Apple Silicon Metal GPU | **Yes** | No | No | No | No |
| Zero-copy CPU↔GPU (unified memory) | **Yes** | N/A | N/A | N/A | No (CUDA copies) |

## Installation

### Prerequisites
- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- macOS 13+ with Apple Silicon (for GPU acceleration)
- Python 3.10+ (for Python bindings)

### Build from source

```bash
# Clone
git clone https://github.com/bobbyprice/oNeura.git
cd oNeura/oneuro-metal

# Build all binaries (optimized, 16GB Mac safe)
cargo build --profile fast --no-default-features

# Build specific binary
cargo build --profile fast --no-default-features --bin terrarium_3d

# Run regression tests
cargo test --no-default-features --lib

# Build Python extension (requires maturin)
pip install maturin
maturin develop --release
```

### CPU-only (non-macOS)

The crate compiles on any platform with Rust. Without Metal, all compute falls back to optimized CPU code using Rayon for parallelism.

## Standalone Binaries

oNeura-Metal ships 11 standalone binaries covering visualization, evolution, ecology, and synthetic biology:

### 3D Terrarium Viewer — `terrarium_3d` (2,257 lines, 18 modules)

Full software 3D rasterizer with orbit camera, Blinn-Phong lighting, raycasted sunlight shadows, and multi-scale semantic zoom — no game engine required, just `minifb`.

```bash
cargo build --profile fast --no-default-features --bin terrarium_3d
./target/fast/terrarium_3d --seed 7 --fps 30
```

**Rendering Pipeline:**
- Z-buffer triangle rasterization with barycentric edge functions
- Per-pixel Blinn-Phong diffuse + specular lighting
- Heightfield-based raycasted sunlight shadows
- Distance-based atmospheric fog
- Screen-space ambient occlusion (SSAO)
- Day/night cycle with warm sunrise/cool night sky gradients

**Multi-Scale Zoom (4 levels):**
| Zoom Level | Camera Distance | What You See |
|------------|----------------|-------------|
| Ecosystem | >12 units | Full terrain, plant canopies, fly swarms, water bodies |
| Organism | 3–12 units | Individual plant physiology, fly body segments with animated wings |
| Cellular | 0.5–3 units | Per-tissue metabolite pools (ATP, glucose, starch, water, nitrate) |
| Molecular | <0.5 units | CPK-colored atoms with van der Waals radii, bond cylinders |

**Interactive Features:**
| Key | Action |
|-----|--------|
| Left drag | Orbit camera (yaw/pitch) |
| Right drag | Pan target along XZ plane |
| Scroll / +/- | Zoom in/out |
| WASD | Keyboard pan |
| F | Follow selected entity (smooth lerp tracking) |
| T | Auto-orbit demo mode |
| [ / ] | Sim speed 1x/2x/4x/8x |
| Click / Tab | Select entity / cycle selection |
| L | Toggle realistic/flat lighting |
| Space | Pause/resume |
| P | Screenshot (BMP) |
| R | Reset camera |

**Visual Systems:**
- Particle system: dust, soil splash, fly wing dust, trail particles
- Population sparkline: 120-frame rolling graph of plant/fly counts
- Minimap: top-down overhead view with camera frustum indicator
- Entity selection: per-pixel EntityTag picking (terrain, plant, fly, water, fruit, atom, bond, metabolite)
- 3-segment fly bodies with animated wing flutter (frame-based sine oscillation)
- Water with specular highlights and slight elevation animation

**18 Modules:**
`main.rs`, `camera.rs`, `color.rs`, `flies.rs`, `fruits.rs`, `hud.rs`, `input.rs`, `lighting.rs`, `math.rs`, `mesh.rs`, `particles.rs`, `plants.rs`, `rasterizer.rs`, `screenshot.rs`, `selection.rs`, `terrain.rs`, `water.rs`, `zoom.rs`

### Semantic Zoom Renderer — `terrarium_zoom` (1,269 lines)

Terminal-based semantic zoom viewer with cursor selection and split-screen detail panels.

```bash
cargo build --profile fast --no-default-features --bin terrarium_zoom
./target/fast/terrarium_zoom --mode iso --fps 15
```

### Evolution Engine — `terrarium_evolve` (3,400+ lines, 49 tests)

NSGA-II multi-objective optimizer that evolves terrarium ecosystem configurations via 18-parameter genomes.

```bash
cargo build --profile fast --no-default-features --bin terrarium_evolve

# Standard fitness climb
./target/fast/terrarium_evolve --population 8 --generations 5 --frames 100 --fitness biomass --lite

# Pareto multi-objective (7 objectives)
./target/fast/terrarium_evolve --pareto --population 8 --generations 5 --frames 50 --lite

# Stress resilience (drought + heat shocks)
./target/fast/terrarium_evolve --pareto --stress-test --population 8 --generations 5 --frames 50 --lite
```

Modes: Standard, NSGA-II Pareto, Stress-Test, Coevolution (fly brain + ecosystem), Bet-Hedging, GRN (gene regulatory networks).

### Drug Protocol Optimizer — `drug_optimizer` (306 lines)

Antibiotic protocol comparison, validation, optimization, and sensitivity scanning for persister cell dynamics.

```bash
cargo build --profile fast --no-default-features --bin drug_optimizer
./target/fast/drug_optimizer --mode compare    # Compare single vs pulsed vs combination
./target/fast/drug_optimizer --mode validate   # Validate against E. coli literature
./target/fast/drug_optimizer --mode optimize   # Evolutionary protocol optimization
./target/fast/drug_optimizer --mode scan       # Parameter sensitivity scan
```

### Gene Circuit Designer — `gene_circuit` (341 lines)

Telegraph model noise designer for synthetic biology — targets specific Fano factor and mean expression levels.

```bash
cargo build --profile fast --no-default-features --bin gene_circuit
./target/fast/gene_circuit --target-fano 5.0 --target-mean 100.0
```

### REST API Server — `terrarium_web` (112 lines)

Axum-based HTTP/WebSocket server for remote terrarium control, snapshot export, tournament orchestration, and authentication.

```bash
cargo build --profile fast --no-default-features --bin terrarium_web
```

### Demo & Research Binaries

| Binary | Lines | Purpose |
|--------|-------|---------|
| `amr_simulator` | 1,030 | Antimicrobial resistance evolution simulator |
| `soil_nutrient_demo` | 746 | Soil nutrient cycling and guild dynamics demo |
| `evolution_lab` | 1,053 | Eco-evolutionary feedback laboratory |
| `terrarium_viewer` | 697 | 2D pixel viewer (minifb) |
| `terrarium_native` | 235 | Headless terminal simulation |

## Terrarium Ecosystem — Molecular Ecology

A complete soil-plant-insect ecosystem where every chemical reaction uses literature-grounded kinetics. 14,098 lines across 13 core terrarium_world submodules.

### Biology Subsystems (All Wired)

| Subsystem | Lines | Physics | Key Reference |
|-----------|-------|---------|---------------|
| **Soil chemistry** | 3,085 | PDE diffusion of 14 species, 3D voxel grid | Moldrup 2001 tortuosity |
| **Microbial guilds** | 463 | 4 guilds: heterotrophs, nitrifiers, denitrifiers, N-fixers | Monod growth kinetics |
| **Plant physiology** | 761 | Farquhar-FvCB photosynthesis, Beer-Lambert canopy | Bernacchi 2001 |
| **Plant competition** | 615 | Asymmetric vertical shading + root nutrient splitting | Beer-Lambert inter-species |
| **Soil fauna** | 921 | Earthworm bioturbation + nematode Lotka-Volterra | N mineralization coupling |
| **Fly lifecycle** | 1,133 | Egg→larva→pupa→adult, temperature-dependent | Sharpe-Schoolfield thermal |
| **Fly metabolism** | 516 | 7-pool Michaelis-Menten: crop→trehalose→ATP | Molecular hunger model |
| **Atmosphere** | (integrated) | 3D odorant diffusion, wind, Rayleigh-Benard convection | Chapman-Enskog |
| **Stochastic expression** | 400 | Gillespie tau-leaping gene noise | Telegraph promoter model |

### Advanced Ecosystem Modules (15,771 lines)

| Module | Lines | What It Simulates |
|--------|-------|-------------------|
| **resistance_evolution** | 1,754 | AMR mutation, fitness costs, plasmid transfer |
| **eco_evolutionary_feedback** | 1,686 | Eco-evo dynamics, niche construction |
| **metabolic_flux** | 1,725 | FBA-style metabolic network analysis |
| **microbiome_assembly** | 1,642 | Community assembly, priority effects, succession |
| **biomechanics** | 561 | Wind/pose biomechanics, displacement integration |
| **horizontal_gene_transfer** | 1,513 | Conjugation, transformation, transduction |
| **population_genetics** | 1,514 | Wright-Fisher, Hardy-Weinberg, genetic drift |
| **biofilm_dynamics** | 1,456 | Biofilm formation, quorum sensing, EPS matrix |
| **nutrient_cycling** | 1,414 | C/N/P cycling, mineralization, immobilization |
| **climate_scenarios** | 1,199 | Temperature/precipitation forcing, seasonal cycles |
| **ecosystem_integration** | 685 | Cross-scale coupling, telemetry, seasonal events |
| **phylogenetic_tracker** | 720 | Lineage tracking, speciation detection |
| **guild_latent** | 463 | Latent guild state and microbial biomass tracking |

### Terrarium World Submodules

| Module | Lines | Status | Purpose |
|--------|-------|--------|---------|
| genotype.rs | 547 | Unconditional | Gene weights + PublicSecondaryBanks |
| packet.rs | 356 | Unconditional | GenotypePacketPopulation |
| calibrator.rs | 242 | Unconditional | Arrhenius/Eyring MD→reaction rate bridge |
| flora.rs | 761 | Unconditional | Advanced plant stepping, asymmetric competition |
| soil.rs | 799 | Unconditional | Guild-aware soil stepping + gas exchange |
| biomechanics.rs | 561 | Feature-gated | Wind/pose biomechanics |
| snapshot.rs | 1,284 | Feature-gated | Full ecosystem snapshot |
| explicit_microbe_impl.rs | 2,107 | Feature-gated | Explicit microbial cohort model |
| mesh.rs | 403 | Feature-gated | 3D mesh generation |
| render_utils.rs | 444 | Feature-gated | Rendering utilities |
| render_impl.rs | 2,525 | Feature-gated | Core render implementation |
| render_stateful.rs | 1,306 | Feature-gated | Stateful render pipeline |
| tests.rs | 2,863 | Feature-gated | Integration tests |

### Evolution Engine Modes

| Mode | Description | Objectives |
|------|-------------|------------|
| **Standard** | Single-objective fitness climb | Biomass, diversity, or stability |
| **Pareto** | NSGA-II multi-objective | 7 simultaneous objectives |
| **Stress-Test** | Resilience under environmental shocks | Drought + heat survival |
| **Coevolution** | Fly brain + ecosystem co-optimization | Neural + ecological fitness |
| **Bet-Hedging** | Stochastic gene expression exploitation | Phenotypic variance advantage |
| **GRN** | Gene regulatory network evolution | Regulatory topology fitness |

## Whole-Cell Simulation — Syn3A Minimal Cell

Native Rust simulator for JCVI-Syn3A (493 genes). Staged integration:
- RDME lattice diffusion (Metal GPU shader)
- CME stochastic expression (Gillespie tau-leaping)
- ODE metabolic fluxes
- Brownian dynamics for chromosome physics
- Geometry/divisome for cell division
- CASCI quantum chemistry for reaction barriers

7,780 lines with submodules: chromosome, initialization, local_chemistry, membrane, spatial, stochastic_expression, tests.

Anchored to: Thornburg et al., *Cell* (2026), DOI `10.1016/j.cell.2026.02.009`.

### Python example

```python
from oneuro_metal import WholeCellSimulator

cell = WholeCellSimulator(x_dim=24, y_dim=24, z_dim=12, dt_ms=0.25, use_gpu=True)
cell.set_metabolic_load(1.2)
cell.enable_default_syn3a_subsystems()
cell.run(100)
print(cell.snapshot())
print(cell.local_chemistry_report())
```

### Rust example

```rust
use oneuro_metal::{WholeCellConfig, WholeCellSimulator};

let mut cell = WholeCellSimulator::new(WholeCellConfig {
    dt_ms: 0.25,
    use_gpu: true,
    ..WholeCellConfig::default()
});
cell.run(100);
println!("{:?}", cell.snapshot());
```

## Molecular Dynamics

Embedded atomistic molecular dynamics (879 lines) with:
- TIP3P water model (O-H bonds, k=553, r₀=0.9572 Å) stable at 1 fs timestep
- PDB/mmCIF structure ingestion (1,200+ lines, 19 tests)
- EmbeddedMolecule topology from atomistic_chemistry (900+ lines, 12 tests)
- Localized MD probes for ribosome/septum/chromosome subregions

## Python API

```python
from oneuro_metal import MolecularBrain, RegionalBrain

# Simple network
brain = MolecularBrain(n_neurons=1000)
brain.stimulate(0, 50.0)  # 50 µA/cm² to neuron 0
brain.run(10000)           # 10K steps (1 second at dt=0.1ms)

# Regional architecture (cortex + thalamus + hippocampus + basal ganglia)
brain = RegionalBrain.xlarge(seed=42)  # 1018 neurons, 23.5K synapses

# Pharmacology
brain.apply_drug("caffeine", 100.0)
brain.apply_drug("diazepam", 5.0)
brain.run(5000)

# Consciousness monitoring
metrics = brain.consciousness_metrics()
print(f"Phi={metrics.phi:.2f}, Composite={metrics.composite:.3f}")
```

## Rust API

```rust
use oneuro_metal::{MolecularBrain, RegionalBrain, ConsciousnessMonitor, DrugType, NTType};

let edges = vec![
    (0u32, 1, NTType::Glutamate),
    (0, 2, NTType::Glutamate),
    (1, 0, NTType::GABA),
];
let mut brain = MolecularBrain::from_edges(3, &edges);
brain.stimulate(0, 50.0);
brain.run(10000);

let mut monitor = ConsciousnessMonitor::new(brain.neuron_count());
for _ in 0..100 {
    brain.step();
    monitor.record(&brain.neurons);
}
let metrics = monitor.compute(&brain.neurons, &brain.synapses);
println!("Composite consciousness: {:.3}", metrics.composite);
```

## Technical Details

### Memory Layout

All neuron state is stored in Structure-of-Arrays (SoA) format for GPU-coalesced memory access:

```
NeuronArrays (~80 f32 per neuron):
├── Membrane: voltage, prev_voltage, fired, refractory_timer
├── HH gating: nav_m, nav_h, kv_n, cav_m, cav_h
├── Conductance scales: [f32; 8] per ion channel type
├── Ligand open fractions: ampa_open, nmda_open, gabaa_open, nachr_open
├── Calcium 4-compartment: ca_cyto, ca_er, ca_mito, ca_micro
├── Second messengers: cAMP, PKA, PKC, CaMKII, IP3, DAG, ERK (10 floats)
├── Phosphorylation: AMPA_p, Kv_p, CaV_p, CREB_p
├── Metabolism: ATP, ADP, glucose, oxygen
├── NT concentrations: [f32; 6] (DA, 5-HT, NE, ACh, GABA, glutamate)
└── External current, spike count, gene expression levels
```

~320 bytes/neuron. 100K neurons = 32 MB — fits easily in Apple Silicon unified memory.

### Metal Shaders (8 kernels)

| Shader | Threads | What It Computes |
|--------|---------|-----------------|
| `hh_gating.metal` | N neurons | α/β rate functions → m,h,n integration |
| `hill_binding.metal` | N neurons | NT concentration → receptor open fractions |
| `membrane_euler.metal` | N neurons | 8-channel I_ion sum + dV/dt + spike detect |
| `calcium_ode.metal` | N neurons | IP3R/RyR/SERCA/MCU/PMCA/NCX calcium flows |
| `second_messenger.metal` | N neurons | G-protein → cAMP/PKA/PKC/CaMKII/CREB cascades |
| `cleft_dynamics.metal` | S synapses | NT degradation + diffusion + reuptake |
| `diffusion_3d.metal` | V voxels | 6-neighbor Laplacian for 3D NT diffusion |
| `whole_cell_rdme.metal` | V voxels | Intracellular reaction-diffusion |

### Supported Drugs

| Drug | Class | Molecular Target | Max Effect |
|------|-------|-----------------|------------|
| Fluoxetine | SSRI | Serotonin reuptake | 5-HT ×4.0 |
| Diazepam | Benzodiazepine | GABA-A allosteric | Conductance ×5.0 |
| Caffeine | Xanthine | Adenosine receptor | Na_v ×1.3, +excitation |
| Amphetamine | Psychostimulant | DAT/NET reverse | DA ×6.0, NE ×4.0 |
| L-DOPA | DA precursor | DOPA decarboxylase | DA ×5.0 |
| Donepezil | AChE inhibitor | Acetylcholinesterase | ACh ×4.0 |
| Ketamine | NMDA antagonist | NMDA channel block | Conductance ×0.1 |
| *Anesthesia* | *Multi-target* | *GABA-A/NMDA/AMPA/Na_v/K_leak* | *>70% consciousness drop* |

## Commercial Modules

| Module | Capability | Lines |
|--------|-----------|-------|
| **Drug Discovery** | Virtual screening, ADMET prediction (Lipinski + Veber), lead optimization | ~1,200 |
| **Enzyme Engineering** | Directed evolution, saturation mutagenesis, DNA shuffling, MM kinetics | ~1,050 |
| **Drug Protocol Optimizer** | Single/pulsed/combination therapy, E. coli validation data | 306 (standalone binary) |
| **Gene Circuit Designer** | Target noise (Fano, CV), telegraph model, evolutionary optimization | 341 (standalone binary) |

## Tests

**214+ regression tests** covering core terrarium, biology subsystems, evolution engine, ecosystem modules, and synbio tools — all passing.

```bash
# Quick regression suite (214 tests)
cargo test --no-default-features --lib -- substrate_stays_bounded guild_activity \
  soil_atmosphere terrarium_evolve drosophila_population plant_competition \
  soil_fauna fly_metabolism field_coupling seed_cellular terrarium_world \
  organism_metabolism stochastic cross_scale phenotypic persister bet_hedging \
  benchmark seasonal drought tropical arid spatial zone plant_noise \
  microbial_noise multi_species single_drug pulsed combination protocol \
  ecoli circuit enzyme_ bioremediation probe_coupling probe_snapshot \
  drug_enzyme soil_enzyme temperature_coupling remediation guild_latent

# Full test suite (includes whole-cell, quantum, commercial modules)
cargo test --no-default-features --lib
```

## Build Tips

```bash
# 16GB Mac safe build (single job to avoid OOM)
CARGO_BUILD_JOBS=1 cargo build --profile fast --no-default-features --bin terrarium_3d

# Build all binaries at once
cargo build --profile fast --no-default-features

# If concurrent builds cause OOM, kill stale rustc processes
pkill -9 -f rustc

# Build isolation for concurrent sessions
CARGO_TARGET_DIR=/tmp/oneuro-build cargo build --profile fast --no-default-features
```

## Demo Commands

```bash
# 3D viewer with orbit camera, shadows, multi-scale zoom
./target/fast/terrarium_3d --seed 7 --fps 30

# Evolution demo (fitness climb ~62 → ~85 over 5 generations)
./target/fast/terrarium_evolve --population 8 --generations 5 --frames 100 --fitness biomass --lite

# Pareto + stress resilience
./target/fast/terrarium_evolve --pareto --stress-test --population 8 --generations 5 --frames 50 --lite

# Semantic zoom renderer (terminal)
./target/fast/terrarium_zoom --mode iso --fps 15

# Drug protocol comparison
./target/fast/drug_optimizer --mode compare

# Gene circuit noise design
./target/fast/gene_circuit --target-fano 5.0 --target-mean 100.0

# AMR evolution simulator
./target/fast/amr_simulator

# Soil nutrient cycling demo
./target/fast/soil_nutrient_demo

# Eco-evolutionary feedback lab
./target/fast/evolution_lab
```

## License

CC BY-NC 4.0

## Links

- **Website**: [oneura.ai](https://oneura.ai)
- **GitHub**: [github.com/robertcprice/oNeura](https://github.com/robertcprice/oNeura)
- **Commercial licensing**: [hello@oneura.ai](mailto:hello@oneura.ai)

## Author

Bobby Price — [@bobbyprice](https://github.com/bobbyprice)
