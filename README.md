# oNeuro

<div align="center">

![oNeuro Logo](docs/assets/logo.png)

**Build digital organisms with biophysically faithful brains — from molecules to behavior.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

---

## Project Status

oNeuro is an **active working research project**. The repository is usable, but it is moving quickly:

- APIs and demo entrypoints are still changing
- benchmark numbers can change as the GPU backends improve
- some subsystems are productionized, while others are still exploratory or draft-quality

If you want the current measured performance rather than older paper text, start here:

- 25K Pong latency/resource comparison: [`results/pong_compare_20260310/README.md`](results/pong_compare_20260310/README.md)
- Documentation index: [`docs/README.md`](docs/README.md)
- Repository guide: [`docs/repo_structure.md`](docs/repo_structure.md)
- Rust/Metal backend: [`oneuro-metal/README.md`](oneuro-metal/README.md)

This should be read as a **working project**, not a frozen release.

## Special Thanks to Contributors

This project exists because of the incredible people who contribute their time, ideas, and expertise:

- **Eric Reid** (@ereid7) — Low-level capability benchmarks, corticostriatal mechanism assays, D1/D2 MSN pathways, hardened reward-modulated plasticity
- **The oNeuro Community** — For pushing the boundaries of digital biology and neuromorphic computing

*Want to contribute? PRs welcome!*

---

## What Is oNeuro?

oNeuro is a platform for simulating **complete digital organisms** — brains, bodies, and environments — at molecular resolution. Every neuron runs real Hodgkin-Huxley ion channel dynamics, communicates through 6 real neurotransmitters, learns through STDP, and responds to drugs via pharmacokinetic/pharmacodynamic models. Membrane potential **emerges from physics** — it is never a hand-set float.

We build digital flies that smell, navigate, learn, and respond to drugs. We build digital neural cultures that learn to play Pong using the free energy principle. We build digital worlds with real molecular diffusion physics where odorant plumes flow in wind.

`oNeuro` now also carries a native Rust/Metal whole-cell modeling track for minimal-cell-style digital cells. That work is explicitly anchored to Thornburg, Z. R., Maytin, A., Kwon, J., Solomon, K. V., et al., "Bringing the Genetically Minimal Cell to Life on a Computer in 4D," `Cell` (2026), DOI `10.1016/j.cell.2026.02.009`. The current native backend is moving toward a substrate-first architecture where local chemistry, structural order, microdomain placement, reaction activity, higher-level CME/ODE/BD/geometry rates, subsystem readiness reduction, and chemistry exchange targeting are driven by generic reaction, assembly, localization, local-state, scalar process-rule, and affine reducer layers rather than hard-coded whole-cell behaviors or hand-authored global process pools.

**This is not a toy neural network simulator.** This is a molecular-resolution digital biology platform.

## Terminology

| Term | What It Means |
|------|--------------|
| **ONN** | **Organic Neural Network** — real biological neurons on hardware. Cortical Labs' [DishBrain](https://doi.org/10.1016/j.neuron.2022.09.001) (800K neurons playing Pong), FinalSpark's bioprocessors, and future living-tissue compute platforms. |
| **dONN** | **digital Organic Neural Network** — oNeuro's biophysically faithful simulation of an ONN, running on GPU/CPU. Same molecular physics, same emergent behaviors, without the biology lab. |
| **oNeuro** | The software platform for building, running, and experimenting with dONNs — from single neurons to 139K-neuron insect brains. |

A dONN differs from a standard artificial neural network (ANN) in the same way a wind tunnel differs from a paper airplane. In a dONN, action potentials emerge from ion channel kinetics, learning emerges from receptor trafficking and STDP, and drug effects emerge from real pharmacology acting on molecular targets. Nothing is hand-tuned.


## Project Structure

```
oNeuro/
├── src/oneuro/
│   ├── molecular/              # Molecular simulation engine + CUDA backend
│   │   ├── cuda_backend.py     # GPU-accelerated HH brain (CUDAMolecularBrain)
│   │   ├── retina.py           # 3-layer biophysical retina (rods/cones → bipolar → RGC)
│   │   ├── network.py          # MolecularNeuralNetwork (pure Python)
│   │   ├── neuron.py           # HH neuron with all 25 subsystems
│   │   ├── membrane.py         # Hodgkin-Huxley membrane dynamics
│   │   ├── ion_channels.py     # 8 channel types (Na_v, K_v, Ca_v, NMDA, ...)
│   │   ├── neurotransmitters.py # 6 NTs with real molecular identities
│   │   ├── pharmacology.py     # 8 drugs with PK/PD models
│   │   ├── consciousness.py    # IIT Phi, PCI, criticality, GW, Orch-OR
│   │   ├── brain_regions.py    # Cortex, thalamus, hippocampus, BG
│   │   └── ...                 # (gene_expression, calcium, glia, axon, etc.)
│   ├── organisms/              # Complete digital organisms
│   │   └── drosophila.py       # Drosophila brain (15 regions) + body (eyes, legs, wings)
│   ├── worlds/                 # Physics-grounded environments
│   │   └── molecular_world.py  # 2D/3D volumetric odorant diffusion, temperature, wind, buoyancy
│   └── environments/           # Game/navigation environments
│       └── doom_fps.py         # DDA raycasting FPS engine (278 FPS)
├── oneuro-metal/               # Rust/Metal backend and native Pong / whole-cell work
├── demos/
│   ├── demo_drosophila_ecosystem.py  # Drosophila ecosystem and behavioral assays
│   ├── demo_dishbrain_pong.py        # DishBrain Pong / arena / scale experiments
│   ├── demo_doom_arena.py            # 3 experiments: spatial navigation, threat avoidance
│   ├── demo_emergent_cuda.py         # 13 experiments: emergent behaviors (GPU)
│   ├── demo_language_learning.py     # Language acquisition at 5K neurons
│   ├── demo_beyond_ann.py            # 23 capabilities impossible in ANNs
│   └── ...
├── papers/
│   ├── beyond_ann_white_paper.md     # 23 experiments proving dONN capabilities
│   ├── dishbrain_replication_paper.md # DishBrain replication (draft, A100 data)
│   └── data/                         # GPU experiment JSON results
├── scripts/
│   ├── vast_deploy.sh                # Vast.ai GPU deployment & benchmarking
│   └── *_telemetry.py                # Profiling / telemetry capture helpers
├── docs/                             # Docs index, repo guide, design notes, backend status
├── results/                          # Measured benchmark artifacts and comparisons
├── doom_videos/                      # Rendered gameplay/video artifacts
├── tmp_demo_*/                       # Temporary generated demo artifacts
├── pyproject.toml
└── LICENSE                           # CC BY-NC 4.0
```

For a quicker orientation to what is canonical vs exploratory, use
[`docs/repo_structure.md`](docs/repo_structure.md).


## What We've Built

### Digital Organisms

| Organism | Neurons | Brain Regions | Behaviors | File |
|----------|---------|---------------|-----------|------|
| **Drosophila melanogaster** | 1K–139K (FlyWire scale) | 15 (AL, MB, CX, OL, VNC, ...) | Olfactory learning, phototaxis, thermotaxis, walking, flight, feeding | `src/oneuro/organisms/drosophila.py` |
| **DishBrain culture** | 1K–25K | Thalamic relay + L5 cortex | Pong via FEP, arena navigation, drug response | `demos/demo_dishbrain_pong.py` |

### Digital Worlds

| Environment | Physics | Resolution | What It Simulates |
|-------------|---------|------------|-------------------|
| **MolecularWorld** | Real gas-phase diffusion (CRC Handbook), 3D wind advection, CFL-stable subcycling, molecular buoyancy | 1mm cells, 2D or 3D volumetric | Odorant plumes rising/sinking by molecular weight, temperature gradients, vertical wind, day/night, soil chemistry |
| **Doom FPS Engine** | DDA raycasting, BSP dungeon generation | 64×48 @ 278 FPS | Room-corridor environments for spatial navigation experiments |

### Digital Senses

| Sense | Mechanism | File |
|-------|-----------|------|
| **MolecularRetina** | 3-layer biophysical retina: photoreceptors (Govardovskii spectral sensitivity) → bipolar cells (ON/OFF pathways) → RGC (HH spiking output) | `src/oneuro/molecular/retina.py` |
| **Olfactory antennae** | Population-coded odorant receptor activation with real detection thresholds (Hallem & Carlson 2006) | Built into organisms |
| **Taste (gustatory)** | Sugar/bitter receptor activation driving SEZ proboscis extension reflex | Built into Drosophila |

### Molecular Brain Engine

25 subsystems running on every neuron:

| Layer | Components |
|-------|-----------|
| **Ion Channels** | Na_v, K_v, K_leak, Ca_v, NMDA, AMPA, GABA_A, nAChR (HH gating kinetics) |
| **Neurotransmitters** | Dopamine, serotonin, NE, ACh, GABA, glutamate (real molecular identities) |
| **Learning** | NMDA-gated STDP, BCM metaplasticity, synaptic tagging & capture |
| **Pharmacology** | 8 drugs with 1-compartment PK (Bateman) + PD (Hill equation) |
| **Gene Expression** | DNA → RNA → Protein, CREB/c-Fos transcription factors, epigenetics |
| **Second Messengers** | cAMP/PKA/PKC/CaMKII/CREB/MAPK cascades |
| **Calcium** | 4-compartment dynamics (cytoplasmic, ER, mitochondrial, microdomain) |
| **Glia** | Astrocytes (glutamate uptake), oligodendrocytes (myelin), microglia (pruning) |
| **Consciousness** | IIT Phi, PCI, neural complexity, criticality, Global Workspace, Orch-OR |
| **Circadian** | TTFL molecular clock, sleep homeostasis, adenosine pressure |

### CUDA Backend (GPU Scale)

The main HH simulation runs on GPU via PyTorch sparse tensors, with a native Rust/Metal track in `oneuro-metal/` for Apple hardware. For the current measured 25K Pong numbers, use [`results/pong_compare_20260310/README.md`](results/pong_compare_20260310/README.md) rather than relying on older paper snapshots.

| Scale | Neurons | Synapses | Target Hardware |
|-------|---------|----------|----------------|
| tiny | 1K | ~14K | Any CPU |
| small | 5K | ~350K | MPS / any GPU |
| medium | 25K | ~7–8M | Apple Metal / A100 / H200 |
| large | 139K (full FlyWire) | ~54M | Large-memory NVIDIA GPU |

## Validated Experiments

### DishBrain Replication — Working Benchmark Track

Replicates Cortical Labs' DishBrain (Kagan et al. 2022, *Neuron*) — the first demonstration that biological neurons learn to play Pong. Our dONN learns via the **Free Energy Principle**: structured feedback (low entropy) for correct actions, random noise (high entropy) for incorrect ones. No reward. No punishment. Just physics.

This track is still under active optimization. The current live benchmark and utilization comparisons are documented in [`results/pong_compare_20260310/README.md`](results/pong_compare_20260310/README.md).

| # | Experiment | What It Tests | Result |
|---|-----------|--------------|--------|
| 1 | **Pong Replication** | FEP-driven learning | PASS (40%→60% hit rate) |
| 2 | **FEP vs DA vs Random** | Learning protocol comparison | PASS (FEP > DA > Random) |
| 3 | **Drug Effects** | Caffeine enhances, diazepam impairs | PASS (validated at 25K on A100) |
| 4 | **Arena Navigation** | 2D grid world navigation | PASS (36% > 15% random) |
| 5 | **Scale Invariance** | Learning at 1K → 10K neurons | PASS |

### Emergent Behaviors — 13/13 PASS

Behaviors that emerge from molecular dynamics and are **impossible in standard ANNs**:

| Experiment | What Emerges |
|-----------|-------------|
| Forgetting resistance | 9% catastrophic forgetting vs 12% baseline |
| Damage recovery | 60% functional recovery after 20% lesion |
| Sleep consolidation | Gene expression + adenosine clearance |
| Interference effects | Proactive and retroactive memory interference |
| Serial position | Primacy and recency effects in memory lists |
| Critical periods | PNN-mediated developmental window closure |
| Circadian modulation | Drug efficacy varies >90% by time of day |

### Language Learning — 100% Accuracy

5,000-neuron dONN learns 30 English words via discriminative Hebbian learning with weight-based BCI readout. 100% word accuracy, 100% sentence generation.

### Drosophila Ecosystem — 6 Experiments

Complete digital fruit fly in a physics-grounded molecular world:

| # | Experiment | What It Tests |
|---|-----------|--------------|
| 1 | **Olfactory Learning** | Mushroom body conditioning (Tully & Quinn 1985 paradigm) |
| 2 | **Phototaxis** | Positive/negative phototaxis via optic lobe → CX → motor |
| 3 | **Thermotaxis** | Navigate toward preferred 24°C zone |
| 4 | **Foraging** | Multi-source olfactory navigation with FEP learning |
| 5 | **Drug Effects** | Caffeine, diazepam, nicotine on foraging performance |
| 6 | **Day/Night Cycle** | Diurnal activity patterns across circadian cycles |

### Spatial Arena — Doom-inspired Navigation

BSP-generated dungeon environments with 8-directional movement, enemies, health pickups, and FEP-driven threat avoidance.

## Applications

### Neuroscience Research
- **In-silico electrophysiology**: Record from any neuron, any synapse, any time — impossible with real tissue
- **Connectome simulation**: Run the 139K-neuron FlyWire Drosophila connectome as a functional digital twin
- **Learning mechanisms**: Compare FEP, dopamine reward, and Hebbian protocols at molecular resolution
- **Circuit manipulation**: Silence, stimulate, or lesion any brain region and observe system-level effects

### Drug Discovery & Pharmacology
- **Virtual drug screening**: Test compounds against a molecular brain with full dose-response curves
- **Pharmacological specificity**: Drugs act on real molecular targets (GABA-A, nAChR, NMDA, MAO, etc.)
- **Chronopharmacology**: Drug efficacy varies with circadian phase — model timing-dependent dosing
- **Safety screening**: Detect neural side effects before animal testing
- **Insecticide development**: Test neuroactive compounds on digital Drosophila brains

### AI & Robotics
- **Biologically grounded controllers**: Use dONN motor output to drive robots or game agents
- **Embodied cognition**: Digital organisms with complete sensorimotor loops (sense → think → act)
- **Emergent intelligence**: Behaviors arise from physics, not hand-coded rules — no reward shaping needed
- **FEP-based learning**: Alternative to reinforcement learning that matches biological learning dynamics

### Education
- **Digital dissection**: Explore brain regions, apply drugs, measure consciousness — no animals harmed
- **Interactive neuroscience**: Students can interact with live demos and closed-loop tasks, even though the 25K benchmark path is still slower than full biological real time
- **Comparative neurobiology**: Compare C. elegans (302 neurons) to Drosophila (139K) to cortical tissue

## Quick Start

```bash
git clone https://github.com/robertcprice/oNeuro.git
cd oNeuro
python3 -m venv .venv
source .venv/bin/activate

# Recommended for the current working repo
pip install -e .
pip install -e .[viz]
```

For Apple Silicon native benchmarking, build the Rust/Metal backend separately:

```bash
cd oneuro-metal
maturin develop --release
cd ..
```

### Build a Fly Brain

```python
from oneuro.organisms.drosophila import Drosophila, MolecularWorld

# Create world with food sources
world = MolecularWorld(size=(100, 100), seed=42)
world.add_fruit(x=30, y=50, sugar=0.8, ripeness=0.7)
world.add_plant(x=70, y=60, nectar_rate=0.2)

# Create digital Drosophila (5000 HH neurons, 350K synapses)
fly = Drosophila(world=world, scale='small')

# Run organism — sense, think, act
for step in range(1000):
    result = fly.step(world=world)
    print(f"pos=({result['x']:.1f}, {result['y']:.1f}) "
          f"speed={result['motor']['speed']:.3f}")
```

### Drug Screening

```python
from oneuro.molecular.cuda_backend import CUDARegionalBrain

brain = CUDARegionalBrain(n_columns=50, device="cuda", seed=42)

# Baseline measurement
baseline_spikes = sum(brain.step() for _ in range(500))

# Apply diazepam (GABA-A potentiator)
brain.apply_drug("diazepam", dose_mg=10.0)
drug_spikes = sum(brain.step() for _ in range(500))

# Result: ~60-98% spike reduction (dose-dependent)
```

### Run Experiments

```bash
# DishBrain replication (5 experiments)
python3 demos/demo_dishbrain_pong.py

# Drosophila ecosystem (6 experiments)
python3 demos/demo_drosophila_ecosystem.py

# Emergent behaviors (13 experiments)
python3 demos/demo_emergent_cuda.py

# Language learning
python3 demos/demo_language_learning.py

# Spatial Arena (Doom-style navigation)
python3 demos/demo_doom_arena.py

# GPU scale with JSON output and multi-seed
python3 demos/demo_dishbrain_pong.py --scale medium --device cuda --runs 5 --json results.json

# Vast.ai GPU deployment
bash scripts/vast_deploy.sh search          # find cheap A100s
bash scripts/vast_deploy.sh all <id> medium # run everything
```

## How dONNs Differ from ANNs

| Capability | Standard ANN | dONN (oNeuro) |
|-----------|-------------|---------------|
| **Action potentials** | Matrix multiply | Emerge from HH ion channel kinetics |
| **Learning** | Backpropagation | STDP from receptor trafficking |
| **Drug response** | Not possible | 8 drugs with real PK/PD acting on molecular targets |
| **Forgetting** | Catastrophic | Resistant (9% vs 12% loss after 4 tasks) |
| **Damage recovery** | None | 60% recovery after 20% lesion |
| **Sleep** | Not modeled | Gene expression, adenosine clearance, memory replay |
| **Consciousness metrics** | Not applicable | IIT Phi, PCI, criticality, Global Workspace |
| **Circadian rhythms** | Not modeled | TTFL clock, circadian drug efficacy variation |
| **Gene expression** | None | Full DNA→RNA→Protein pipeline |

## Papers

| Paper | Status | Experiments | File |
|-------|--------|-------------|------|
| **Beyond ANN** | Complete | 23/23 PASS | `papers/beyond_ann_white_paper.md` |
| **DishBrain Replication** | Working draft + live benchmark artifacts | 5/5 PASS baseline, actively optimized | `papers/dishbrain_replication_paper.md` |

## Basal Ganglia Learning Benchmark

A validated Go/No-Go benchmark testing dopamine-dependent reinforcement learning in the basal ganglia.

### Results: 30-Seed Standard Scale

| Condition | Pre | Post | Δ | 95% CI |
|-----------|-----|------|---|--------|
| **full_learning** | 90.2% | 100% | **+9.8%** | [+8%, +12%] |
| **no_dopamine** | 83.3% | 50.0% | **-33.3%** | [-36%, -30%] |

**Contrast: +43.1%** | Cohen's d ≈ 1.6 (very large)

### Results: 20-Seed 4 Conditions

| Condition | Pre | Post | Δ | 95% CI |
|-----------|-----|------|---|--------|
| **full_learning** | 90.2% | 100% | **+9.8%** | [+8%, +12%] |
| **nmda_block** | 82.2% | 58.5% | -23.7% | [-28%, -19%] |
| **anti_correlated** | 88.0% | 71.2% | -16.8% | [-21%, -13%] |
| **no_dopamine** | 86.5% | 50.0% | -36.5% | [-40%, -33%] |

### Key Findings
- **Dopamine learning works**: Networks improve accuracy by ~10%
- **NMDA critical**: Blocking NMDA receptors impairs learning by ~24%
- **Contingency proven**: Inverted rewards cause learning of wrong associations
- **Ablation robust**: Removing dopamine causes 33-37% accuracy decline

### Run Benchmark

```bash
# 30-seed confirmatory
PYTHONPATH=src python3 experiments/go_no_go_benchmark.py \
    --conditions full_learning no_dopamine \
    --n-seeds 30 \
    --scale standard \
    --workers 2

# 4 conditions
PYTHONPATH=src python3 experiments/go_no_go_benchmark.py \
    --conditions full_learning no_dopamine nmda_block anti_correlated \
    --n-seeds 20 \
    --scale standard
```

### Generate Figures

```bash
python3 experiments/generate_figures.py results.json figures/
```

### Benchmark Files

| File | Purpose |
|------|---------|
| `experiments/go_no_go_benchmark.py` | Main benchmark script |
| `experiments/generate_figures.py` | Publication figure generator |
| `experiments/results/*.json` | Raw results |
| `experiments/figures/*.png` | Publication figures |
| `docs/benchmarks/RESULTS_SUMMARY.md` | Full results documentation |

## Requirements

- Python 3.11+
- NumPy >= 1.24
- PyTorch >= 2.0 (for CUDA/MPS GPU backend)
- Optional: [nQPU](https://github.com/robertcprice/nqpu-metal) for quantum chemistry

## Citation

```bibtex
@software{oneuro_2026,
  title = {oNeuro: Digital Organic Neural Network Platform for Molecular-Scale Brain Simulation},
  author = {Price, Robert C.},
  year = {2026},
  url = {https://github.com/robertcprice/oNeuro}
}
```

## License

CC BY-NC 4.0 — See [LICENSE](LICENSE)

For commercial licensing: research@entropy.ai
