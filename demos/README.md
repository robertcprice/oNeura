# oNeura Demos

This is a working demo surface, not a frozen product menu. If you want the
current benchmark-critical entrypoints first, start with:

- `demo_dishbrain_pong.py` for the 25K Pong benchmark track
- `demo_drosophila_ecosystem.py` for the fly ecosystem track
- `demo_doom_arena.py` for the spatial arena track

Use `results/` for measured claims and treat the rest of the demos as mixed
stability: some are benchmarked, some are exploratory.

Organized by organism type:

## Fly (Drosophila) - BoB (Brains on Board)

| Demo | Description | File |
|------|-------------|------|
| Ecosystem | Complete fly in physics world | `demo_drosophila_ecosystem.py` |
| Outdoor Flight | Neural drone navigation | `demo_outdoor_flight.py` |
| ONNX Export | Edge deployment for MCUs | `demo_onnx_export.py` |
| C. elegans | 302-neuron connectome | `demo_celegans_ecosystem.py` |

### BoB-Specific Demos

The **Brains on Board** demos implement Nature 2024 paper features:

#### Outdoor Flight Simulation (`demo_outdoor_flight.py`)
Simulated drone navigation using fly brain (no hardware required):
- **Phototaxis**: Fly toward sun/light
- **Homing**: Return to starting position
- **Landmark Navigation**: Use learned visual landmarks
- **Exploration**: Random walk with learning

```bash
python3 demos/demo_outdoor_flight.py --exp phototaxis --visualize
python3 demos/demo_outdoor_flight.py --exp homing
python3 demos/demo_outdoor_flight.py --exp landmark_navigation
```

#### ONNX Export (`demo_onnx_export.py`)
Export brains for edge deployment on:
- Cortex-M0 (64KB flash, 16KB RAM)
- Cortex-M4 (256KB flash, 64KB RAM)
- Cortex-M7 (1MB flash, 256KB RAM)

```bash
python3 demos/demo_onnx_export.py --scale tiny
python3 demos/demo_onnx_export.py --output brain.onnx
```

### BoB Navigation Features
- Ring attractor for heading representation
- Path integration (dead reckoning)
- Landmark learning and recognition
- Compass-based navigation (sun/light)
- Obstacle avoidance

## PetriBrain (DishBrain-style)

| Demo | Description | File |
|------|-------------|------|
| Pong | FEP learning to play Pong | `demo_dishbrain_pong.py` |
| Doom Arena | Spatial navigation | `demo_doom_arena.py` |
| Doom Brain | Brain + retina + Doom | `demo_doom_brain.py` |
| Doom Combat | Combat behaviors | `demo_doom_combat.py` |
| Doom VizDoom | Full VizDoom interface | `demo_doom_vizdoom.py` |

## Brain & Learning

| Demo | Description | File |
|------|-------------|------|
| Beyond ANN | 23 emergent capabilities | `demo_beyond_ann.py` |
| Emergent CUDA | GPU-accelerated emergence | `demo_emergent_cuda.py` |
| Language | Word learning | `demo_language_learning.py` |
| Full Brain | RegionalBrain demo | `demo_full_brain.py` |
| Brain Tasks | Task-based benchmarking | `demo_brain_tasks.py` |

## Molecular

| Demo | Description | File |
|------|-------------|------|
| Molecular Emergence | Single-neuron dynamics | `demo_molecular_emergence.py` |
| Research Platforms | Platform comparison | `demo_research_platforms.py` |

---

## Quick Start

```bash
# Fly ecosystem
python3 demos/demo_drosophila_ecosystem.py

# DishBrain Pong
python3 demos/demo_dishbrain_pong.py

# Doom Arena
python3 demos/demo_doom_arena.py
```
