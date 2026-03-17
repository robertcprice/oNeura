#!/usr/bin/env python3
"""
oNeura Full Integration Demo

This demonstrates the complete neural-molecular simulation pipeline:
1. GPU-accelerated molecular dynamics (Rust/Metal)
2. Drosophila brain simulation with Hodgkin-Huxley neurons
3. Integration between molecular odorants and neural processing
4. Emergent behavior - flies navigate to food via neural dynamics

Run with: python demo_full_integration.py
"""

import sys
import time
import numpy as np

# Try to import the Rust module
try:
    from oneuro_metal import GPUMolecularDynamics, NeuralMDSim, DrosophilaScale
    RUST_AVAILABLE = True
    print("✓ Rust module loaded (oneuro-metal)")
except ImportError:
    RUST_AVAILABLE = False
    print("⚠ Rust module not available - using Python fallback")


def create_water_box(n_atoms: int = 1000):
    """Create a simple water box for MD demonstration."""
    md = GPUMolecularDynamics(n_atoms, "auto")

    # Set random positions
    positions = np.random.rand(n_atoms * 3).astype(np.float32) * 30
    md.set_positions(positions)

    # Set random velocities
    velocities = (np.random.rand(n_atoms * 3) - 0.5).astype(np.float32) * 0.5
    md.set_velocities(velocities)

    # Set masses (oxygen = 16, hydrogen = 1)
    masses = np.array([16.0] * n_atoms, dtype=np.float32)
    md.set_masses(masses)

    # Add some bonds (simplified water model)
    for i in range(0, n_atoms - 2, 3):
        md.add_bond(i, i + 1, 0.96, 450)  # O-H
        md.add_bond(i, i + 2, 0.96, 450)  # O-H

    md.set_temperature(300.0)
    md.set_box([30.0, 30.0, 30.0])
    md.initialize_velocities()

    return md


def create_neural_molecular_sim(n_particles: int = 2000, brain_scale: str = "tiny"):
    """Create integrated neural-molecular simulation."""
    sim = NeuralMDSim(
        n_particles=n_particles,
        brain_scale=brain_scale,
        n_agents=1,
        bounds=[50.0, 50.0]
    )

    # Add food sources
    sim.add_food(15.0, 20.0, 0.5, 5.0, 1.0)
    sim.add_food(40.0, 35.0, 0.5, 4.0, 0.8)
    sim.add_food(30.0, 10.0, 0.5, 3.0, 0.6)

    return sim


def run_md_demo():
    """Run pure molecular dynamics demo."""
    print("\n" + "="*60)
    print("MOLECULAR DYNAMICS DEMO")
    print("="*60)

    if not RUST_AVAILABLE:
        print("Rust module not available - skipping MD demo")
        return

    n_atoms = 500
    n_steps = 100

    print(f"\nInitializing {n_atoms} atom system...")
    md = create_water_box(n_atoms)

    print(f"Running {n_steps} simulation steps...")
    start = time.time()

    for step in range(n_steps):
        stats = md.step(0.001)  # 1 fs timestep

        if step % 20 == 0:
            print(f"  Step {step:3d}: T={stats.temperature:.1f}K, "
                  f"E={stats.total_energy:.2f}, "
                  f"KE={stats.kinetic_energy:.2f}")

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f}s ({n_steps/elapsed:.1f} steps/sec)")

    # Get final positions
    positions = md.positions()
    print(f"Position range: [{min(positions):.2f}, {max(positions):.2f}]")


def run_neural_md_demo():
    """Run neural-molecular integration demo."""
    print("\n" + "="*60)
    print("NEURAL-MOLECULAR INTEGRATION DEMO")
    print("="*60)

    if not RUST_AVAILABLE:
        print("Rust module not available - using Python fallback")
        run_python_fallback()
        return

    n_particles = 1000

    print(f"\nInitializing simulation with {n_particles} particles...")
    sim = create_neural_molecular_sim(n_particles, "tiny")

    print("Food sources added at: (15,20), (40,35), (30,10)")
    print("\nRunning simulation...")

    n_steps = 50
    start = time.time()

    for step in range(n_steps):
        positions, stats = sim.step()

        if step % 10 == 0:
            print(f"  Step {step:3d}: "
                  f"Temp={stats.get('temperature', 0):.1f}K, "
                  f"KE={stats.get('kinetic', 0):.2f}")

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f}s")


def run_python_fallback():
    """Python-only fallback when Rust module unavailable."""
    print("\n" + "="*60)
    print("PYTHON FALLBACK DEMO")
    print("="*60)

    # Simple particle simulation
    n_particles = 500
    positions = np.random.rand(n_particles, 3) * 30
    velocities = (np.random.rand(n_particles, 3) - 0.5) * 0.2

    food_sources = np.array([[15, 20], [40, 35], [30, 10]])

    print(f"\nSimulating {n_particles} particles...")

    for step in range(50):
        # Simple attraction to food
        for i, pos in enumerate(positions):
            nearest = food_sources[np.argmin(np.linalg.norm(food_sources - pos[:2], axis=1))]
            direction = nearest - pos[:2]
            dist = np.linalg.norm(direction)
            if dist > 1:
                velocities[i, :2] += direction / dist * 0.01

        # Random walk
        velocities += (np.random.rand(n_particles, 3) - 0.5) * 0.05
        velocities *= 0.98  # Damping
        positions += velocities

        # Wrap around
        positions = positions % 30

        if step % 10 == 0:
            print(f"  Step {step}: particles near food = {np.sum(np.any(np.abs(positions[:, :2] - food_sources) < 5, axis=1))}")

    print("\n✓ Simulation complete!")


def main():
    print("="*60)
    print("oNeura - Neural Molecular Dynamics")
    print("="*60)

    # Run MD demo
    run_md_demo()

    # Run neural-MD demo
    run_neural_md_demo()

    print("\n" + "="*60)
    print("DEMOS COMPLETE")
    print("="*60)
    print("\nTo run the browser visualization:")
    print("  open oneuro-wasm/web/full_integration.html")
    print("\nOr run a specific demo:")
    print("  python -m http.server 8080")
    print("  # Then open http://localhost:8080/oneuro-wasm/web/")


if __name__ == "__main__":
    main()
