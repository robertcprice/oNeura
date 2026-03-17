# oNeura Physics Module - Complete Documentation

## Overview

The oNeura physics module provides a complete 3D neuromechanical simulation of *Drosophila melanogaster* using MuJoCo. It combines realistic body physics with simplified neural control to enable closed-loop sensorimotor behavior.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHYSICS MODULE ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │
│   │   PHYSICS    │     │    BRAIN     │     │   SENSORY    │           │
│   │  (MuJoCo)    │◄───►│  (Neural)    │◄───►│  (Sensors)   │           │
│   └──────────────┘     └──────────────┘     └──────────────┘           │
│         │                    │                    │                      │
│         ▼                    ▼                    ▼                      │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │
│   │   drosophila │     │ connectome_  │     │ compound_eye │           │
│   │ _simulator.py│     │   bridge.py  │     │  olfaction.py│           │
│   └──────────────┘     └──────────────┘     └──────────────┘           │
│                                                                          │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │
│   │  physics_    │     │ brain_motor_ │     │ biological_  │           │
│   │ environment.py│    │  interface.py│     │   params.py  │           │
│   └──────────────┘     └──────────────┘     └──────────────┘           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Body Model (`drosophila_mjcf.xml`)

| Feature | Count | Notes |
|---------|-------|-------|
| Bodies | 38 | Head, thorax, 6 legs (4 segments each), wings, antennae |
| Joints | 38 | 6-DOF root + 32 articulated joints |
| Actuators | 26 | 2 wing + 24 leg motors |
| Sensors | 36 | 6 tarsus touch + 30 joint position/velocity |

**Scale**: 1000x (2.5mm → 2.5m) for MuJoCo numerical stability.

### 2. Compound Eye (`compound_eye.py`)

| Feature | Value | Reference |
|---------|-------|-----------|
| Ommatidia/eye | ~300 | Hexagonal lattice with frontal bias |
| Acceptance angle | 5° | Land (1997) |
| Visual field | 300° × 180° | ~150° per eye |
| Motion detection | Hassenstein-Reichardt EMDs | Borst (2009) |
| Spectral channels | UV, Blue, Green | Rh3/Rh4, Rh5, Rh6 |

**Key Classes**:
- `CompoundEye`: Single eye with ommatidia lattice
- `BinocularVisionSystem`: Integrates both eyes
- `MotionDetector`: EMD implementation
- `create_visual_encoder_output()`: Compresses to neural dimension

### 3. Olfactory System (`olfaction.py`)

| Feature | Value | Reference |
|---------|-------|-----------|
| Receptor types | 50 | Hallem & Carlson (2006) |
| Attractive odors | ethyl_acetate, vinegar, phenethyl_alcohol | Fruit/yeast |
| Repellent odors | benzaldehyde, CO2 | Stress signals |
| Response function | Hill equation | Concentration-response |
| Adaptation | τ = 5s | Sensory fatigue |

**Key Classes**:
- `OdorantReceptor`: Single receptor with Hill response
- `Antenna`: Collection of 50 receptors
- `BilateralOlfaction`: Both antennae for chemotaxis
- `create_olfactory_encoder_output()`: Compresses to neural dimension

### 4. Physics Environment (`physics_environment.py`)

| Feature | Description |
|---------|-------------|
| Fruits | Odorant sources with sugar concentration |
| Plants | Stem + flower geometry |
| Obstacles | Box and cylinder collision |
| Wind | Aerodynamic drag force |
| Light | Directional with intensity |

**Preset Scenes**:
- `foraging_arena()`: 3 fruits, 2 plants, 2 obstacles, wind
- `obstacle_course()`: 7 obstacles, no wind
- `empty_arena()`: Minimal environment

### 5. Connectome Bridge (`connectome_bridge.py`)

| Pool | Neurons | Type | Function |
|------|---------|------|----------|
| mechanosensory_tarsi | 50 | Sensory | Touch encoding |
| proprioceptive_legs | 50 | Sensory | Joint angles |
| visual_motion | 50 | Sensory | Optic flow |
| olfactory_antenna | 50 | Sensory | Odor encoding |
| local_circuit_legs | 100 | Interneuron | Leg coordination |
| descending_commands | 50 | Interneuron | Brain→body |
| central_pattern_generator | 50 | Interneuron | Rhythm generation |
| visual_integration | 50 | Interneuron | Visual processing |
| motor_protractor | 50 | Motor | Swing phase |
| motor_retractor | 50 | Motor | Stance phase |
| motor_left | 50 | Motor | Left turn |
| motor_right | 50 | Motor | Right turn |
| wing_motor | 50 | Motor | Wingbeat |

**Total**: 650 neurons, ~6,338 synapses

### 6. Biological Parameters (`biological_params.py`)

| Parameter | Real Value | Scaled Value | Source |
|-----------|------------|--------------|--------|
| Body length | 2.5mm | 2.5m | NeuroMechFly |
| Body mass | ~1mg | ~1kg | NeuroMechFly |
| Wing frequency | 200Hz | 200Hz | Dickinson & Tu |
| Stride frequency | 8Hz | 8Hz | Bidaye et al. |
| Tarsus friction | 1.5 | 1.5 | NeuroMechFly |

### 7. GPU Batch Physics (`gpu_batch_physics.py`)

| Feature | Value | Description |
|---------|-------|-------------|
| Framework | MJX/JAX | GPU-accelerated MuJoCo |
| Batch size | 10-1000 flies | Parallel simulation |
| Gymnasium | ✅ | Standard RL interface |
| Step time | ~2ms | Per fly-step on H200 |

**Key Classes**:
- `DrosophilaBatchEnv`: Gymnasium-compatible vectorized environment
- `MJXPhysicsEngine`: JAX-based physics backend
- `batch_step()`: Vectorized simulation step

### 8. FlyWire Connectome (`flywire_connectome.py`)

| Feature | Value | Description |
|---------|-------|-------------|
| Total neurons | 139K (full) | Dorkenwald et al. 2024 |
| Scale tiers | tiny/small/medium/large | 1K/5K/25K/139K |
| Brain regions | 9 major | AL, MB, CX, OL, LH, SEZ, VNC |
| Validated circuits | 8 | olfactory, phototaxis, chemotaxis, etc. |

**Key Classes**:
- `FlyWireConnectome`: Full connectome data with scaled neuron counts
- `CircuitTemplate`: Validated circuit with connection parameters
- `create_connectome()`: Factory function

**Validated Circuits**:
- `olfactory`: ORN → AL → MB/LH pathway
- `odor_learning`: KC → MBON with DAN gating
- `phototaxis`: photoreceptor → VNC locomotion
- `motion_detection`: Hassenstein-Reichardt EMDs
- `chemotaxis`: bilateral odor → turning
- `compass`: ring attractor heading
- `walking`: CPG for leg coordination
- `flight`: wing motor control

### 9. Multi-Fly Arena (`multi_fly_arena.py`)

| Feature | Value | Description |
|---------|-------|-------------|
| Flies | 10-100 | Parallel simulation |
| Pheromones | 4 types | CVA, male/female CHCs, stress |
| Social | aggregation, courtship, aggression, flocking |
| Sensing | visual + olfactory inter-fly |

**Key Classes**:
- `MultiFlyArena`: Main arena with multiple flies
- `PheromoneSystem`: Chemical communication
- `SocialMetrics`: Behavior quantification
- `create_multi_fly_arena()`: Factory function

### 10. RL Foraging (`rl_foraging.py`)

| Feature | Value | Description |
|---------|-------|-------------|
| Interface | Gymnasium | Compatible with SB3, Tonic |
| Observation | 14-dim | position, velocity, pheromone, energy, fruit dir |
| Action | 3-dim | heading (2) + speed (1) |
| Rewards | gradient + fruit | Odorant climbing + sugar |

**Key Classes**:
- `DrosophilaForagingEnv`: Gymnasium-compatible RL environment
- `ForagingConfig`: Reward/penalty configuration
- `create_foraging_env()`: Factory function

**Reward Shaping**:
- Gradient climbing: +10 per unit toward fruit
- Fruit contact: +100 × sugar concentration
- Time penalty: -0.1 per step
- Energy cost: -0.01 × speed

---

## Test Coverage

### Local Tests (macOS, no MuJoCo)

```
76 tests in 1.12s
```

| Module | Tests | Status |
|--------|-------|--------|
| LIF Neurons | 6 | ✅ PASS |
| ConnectomeBridge | 10 | ✅ PASS |
| PhysicsEnvironment | 12 | ✅ PASS |
| SensorEncoder | 3 | ✅ PASS |
| NeuralPatternDecoder | 2 | ✅ PASS |
| BrainMotorInterface | 2 | ✅ PASS |
| StateVector | 3 | ✅ PASS |
| MotorPatterns | 4 | ✅ PASS |
| CompoundEye | 11 | ✅ PASS |
| BinocularVision | 3 | ✅ PASS |
| VisualIntegration | 2 | ✅ PASS |
| OdorantReceptor | 4 | ✅ PASS |
| Antenna | 3 | ✅ PASS |
| BilateralOlfaction | 4 | ✅ PASS |
| BiologicalParameters | 6 | ✅ PASS |
| GPUBatchPhysics | 6 | ✅ PASS |

### GPU Validation (H200, MuJoCo 3.5.0)

| Component | Status | Details |
|-----------|--------|---------|
| MuJoCo | ✅ PASS | Version 3.5.0 |
| Physics Demo | ✅ PASS | 100 steps, 38 bodies, 26 actuators |
| Compound Eye | ✅ PASS | 314 ommatidia, intensity 0.0-0.46 |
| Olfaction | ✅ PASS | 50 receptors, turn signal working |
| Biological Params | ✅ PASS | All parameters correct |

---

## Usage Examples

### Basic Physics Simulation

```python
from oneuro.physics.drosophila_simulator import DrosophilaSimulator
from oneuro.physics.physics_environment import PhysicsEnvironment

# Create environment
env = PhysicsEnvironment.foraging_arena()
mjcf_xml = env.generate_mjcf()

# Initialize simulator
sim = DrosophilaSimulator(mjcf_xml=mjcf_xml)

# Run simulation
for step in range(1000):
    # Apply wind
    wind = env.compute_wind_force(sim.get_state().body_velocity)
    sim.apply_perturbation(wind)

    # Step physics
    sim.step([])

sim.close()
```

### Compound Eye Sampling

```python
from oneuro.physics.compound_eye import BinocularVisionSystem
import numpy as np

vision = BinocularVisionSystem(num_ommatidia_per_eye=768)

sample = vision.sample(
    light_direction=np.array([0, 0, 1]),
    light_intensity=1.0,
    body_orientation=np.array([1, 0, 0, 0])
)

# Get outputs
left_eye = sample['left_intensities']     # (768,) array
right_eye = sample['right_intensities']   # (768,) array
motion_h, motion_v = sample['global_motion']
yaw, pitch = sample['phototaxis']
```

### Olfactory Sampling

```python
from oneuro.physics.olfaction import BilateralOlfaction

olf = BilateralOlfaction(num_receptors=50)

# Sample odorants
sample = ol.sample(
    odorants={"ethyl_acetate": 1.0, "vinegar": 0.5}
)

# Get outputs
left_rates = sample['left_rates']      # (50,) array
right_rates = sample['right_rates']    # (50,) array
turn_signal = sample['turn_signal']    # -1 to 1
approach_signal = sample['approach_signal']  # -1 to 1
```

---

## File Structure

```
src/oneuro/physics/
├── __init__.py              # Module exports
├── drosophila_mjcf.xml      # Body model
├── drosophila_simulator.py  # Physics wrapper
├── brain_motor_interface.py # Brain↔Physics bridge
├── connectome_bridge.py     # Neural controller
├── compound_eye.py          # Visual system
├── olfaction.py             # Olfactory system
├── physics_environment.py   # 3D world
├── biological_params.py     # Parameter tuning
├── gpu_batch_physics.py     # MJX/JAX GPU batch simulation
├── flywire_connectome.py    # Real FlyWire connectome data
├── multi_fly_arena.py       # Multi-fly social simulation
├── rl_foraging.py          # RL foraging environment
├── demo_mujoco.py           # Full demo
├── demo_environment.py      # Environment demo
├── examples.py              # Code examples
├── README.md                # Module overview
├── VALIDATION_REPORT.md     # GPU validation results
└── PHYSICS_MODULE_SUMMARY.md # This document
```

---

## References

1. Lobato-Rios et al. (2022) "NeuroMechFly, a neuromechanical model of adult Drosophila"
2. Hassenstein & Reichardt (1956) "Systemtheoretische Analyse der Zeit-, Reihenfolgen- und Vorzeichenauswertung"
3. Borst (2009) "Drosophila's view on insect vision"
4. Hallem & Carlson (2006) "Coding of odors by a receptor repertoire"
5. Dickinson & Tu (1997) "The function of dipteran flight muscle"
6. Bidaye et al. (2014) "Two pathways for walking in Drosophila"
