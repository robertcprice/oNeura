# oNeura Physics Module

Physics simulation for Drosophila melanogaster and neural-based organisms.

## Structure

```
physics/
├── __init__.py           # Main exports
├── ecosystem.py           # Terrarium simulation with neural flies
├── drosophila_simulator.py  # MuJoCo-based fly physics
├── brain_motor_interface.py  # Neural to motor translation
├── connectome_bridge.py     # Connectome-based brain
├── physics_environment.py   # Base environment
├── compound_eye.py          # Visual system
├── olfaction.py             # Olfactory system
├── multi_fly_arena.py       # Multi-fly social simulation
├── rl_foraging.py          # RL-based foraging
├── gpu_batch_physics.py    # GPU-accelerated physics
│
├── web/                     # Legacy HTML demos (not authoritative terrarium state)
│   ├── particle_world.html  # GPU particle simulation
│   └── ...
│
└── demos/                   # Demo scripts
    ├── demo_mujoco.py      # MuJoCo physics demo
    ├── demo_environment.py # Environment demo
    └── demo_ecosystem.py   # Ecosystem demo
```

## Quick Start

### Python Ecosystem Simulation

```python
from oneuro.physics.ecosystem import create_ecosystem, Terrarium

# Create ecosystem with 20 neural flies
terrarium = create_ecosystem(n_flies=20)

# Run simulation
for step in range(1000):
    terrarium.step()

# Get stats
stats = terrarium.get_stats()
print(f"Population: {stats['adults']} adults, {stats['larvae']} larvae")
```

### HTML Visualization

Older browser-only terrarium demo pages under `src/oneuro/physics/web/` were
removed because they invented biology and chemistry locally in JavaScript and
were not authoritative entrypoints.

Use the canonical runtime-backed terrarium shell at
`terrarium/web/terrarium.html` through the Rust terrarium web server or the
desktop shell.

## Key Classes

- `Terrarium` - Complete ecosystem simulation
- `FlyOrganism` - Neural fly with brain
- `FoodSource` - Food that gets consumed
- `NeuralActivity` - Real-time brain state
