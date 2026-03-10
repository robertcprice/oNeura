#!/usr/bin/env python3
"""
MuJoCo Integration Demo for Drosophila melanogaster

This demo shows the full 3D physics simulation with:
- Realistic fly body (head, thorax, abdomen, 6 legs, 2 wings)
- Articulated joints for walking and flying
- Touch sensors on tarsi
- Connectome-based brain controlling physics

Usage:
    cd /Users/bobbyprice/projects/oNeuro
    PYTHONPATH=src python3 src/oneuro/physics/demo_mujoco.py
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from oneuro.physics.drosophila_simulator import (
    DrosophilaSimulator,
    DrosophilaPhysicsState,
    MotorCommand,
    MotorPatterns,
)
from oneuro.physics.brain_motor_interface import (
    BrainMotorInterface,
    SensorEncoder,
    NeuralPatternDecoder,
)
from oneuro.physics.connectome_bridge import ConnectomeBridge, create_brain_model
from oneuro.physics.physics_environment import PhysicsEnvironment


def main():
    print("=" * 70)
    print("  DROSOPHILA MELANOGASTER - MuJoCo 3D Physics Simulation")
    print("=" * 70)

    # 1. Build 3D environment
    print("\n1. BUILDING 3D ENVIRONMENT")
    print("-" * 40)
    env = PhysicsEnvironment.foraging_arena()
    print(f"  {env}")
    print(f"   - Fruits: {len(env.fruits)}")
    print(f"   - Plants: {len(env.plants)}")
    print(f"   - Obstacles: {len(env.obstacles)}")
    print(f"   - Wind speed: {env.wind_speed:.1f}")

    # Generate augmented MJCF with environment
    import tempfile, os
    mjcf_xml = env.generate_mjcf()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False
    ) as f:
        f.write(mjcf_xml)
        env_mjcf_path = f.name
    print(f"  Environment MJCF generated ({len(mjcf_xml)} chars)")

    # 2. Initialize physics simulator with environment
    print("\n2. INITIALIZING PHYSICS ENGINE")
    print("-" * 40)
    try:
        sim = DrosophilaSimulator(mjcf_path=env_mjcf_path)
    finally:
        os.unlink(env_mjcf_path)
    print(f"  MuJoCo model loaded")
    print(f"   - Scale: 1000x (for numerical stability)")
    print(f"   - Real scale: ~2.5mm body, ~1mg mass")
    print(f"   - Bodies: {sim.model.nbody}")
    print(f"   - Joints: {sim.model.njnt}")
    print(f"   - Actuators: {sim.model.nu}")
    print(f"   - Sensors: {sim.model.nsensor}")

    # 2. Initialize brain-motor interface
    print("\n2. INITIALIZING BRAIN-MOTOR INTERFACE")
    print("-" * 40)
    interface = BrainMotorInterface()
    print(f"  Interface initialized")
    print(f"   - Sensor dim: {interface.get_sensor_dim()}")
    print(f"   - Motor dim: {interface.get_motor_dim()}")

    # 3. Initialize connectome bridge (simplified brain)
    print("\n3. INITIALIZING CONNECTOME BRIDGE")
    print("-" * 40)
    bridge = ConnectomeBridge(num_neurons_per_pool=50)
    print(f"  Connectome bridge initialized")
    print(f"   - Neuron pools: {len(bridge.pools)}")
    print(f"   - Synapses: {len(bridge.connections)}")
    print(f"   - Sensor input dim: {bridge.get_sensor_input_dim()}")
    print(f"   - Motor output dim: {bridge.get_motor_output_dim()}")

    # 4. Run physics simulation with tripod gait + wind + environment
    print("\n4. RUNNING PHYSICS SIMULATION (Tripod Gait + Environment)")
    print("-" * 40)

    state = sim.reset()
    gait_frequency = 10.0  # Hz

    for step in range(500):
        # Generate tripod gait pattern
        phase = (step * sim.timestep * gait_frequency) % 1.0
        targets = MotorPatterns.tripod_gait(phase)

        # Convert to motor commands
        commands = []
        for joint_name, angle in targets.items():
            motor_name = joint_name.replace("_pitch", "_motor")
            commands.append(MotorCommand(
                actuator_name=motor_name,
                control_signal=angle * 0.5,
            ))

        # Apply wind force
        wind_force = env.compute_wind_force(state.body_velocity)
        sim.apply_perturbation(wind_force)

        state = sim.step(commands)

        if step % 100 == 0:
            pos = state.body_position
            env_sample = env.sample_environment(pos)
            print(f"   Step {step}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
                  f" odorant={env_sample['odorant']:.3f}")

    print(f"  Simulation complete ({sim.time:.3f}s sim time)")

    # 5. Show final state
    print("\n5. FINAL STATE")
    print("-" * 40)
    final_state = sim.get_state()
    print(f"   - Body position: {final_state.body_position}")
    print(f"   - Body velocity: {final_state.body_velocity}")
    print(f"   - Touch contacts: {final_state.touch_contacts}")

    # 6. Connectome-driven simulation
    print("\n6. CONNECTOME-DRIVEN SIMULATION")
    print("-" * 40)

    brain_model = create_brain_model()
    interface_with_brain = BrainMotorInterface(brain_model=brain_model)

    sim.reset()
    for step in range(200):
        state = sim.get_state()
        commands = interface_with_brain.process(state, dt=sim.timestep)
        sim.step(commands)

    print(f"  Connectome simulation complete ({sim.time:.3f}s)")
    print(f"   - Final position: {sim.get_state().body_position}")

    # 7. Wingbeat simulation
    print("\n7. WINGBEAT SIMULATION")
    print("-" * 40)
    sim.reset()

    for step in range(200):
        freq = 200.0  # Hz wingbeat frequency
        wing_targets = MotorPatterns.wingbeat(freq, sim.time, amplitude=0.8)
        commands = [
            MotorCommand(f"{wing}_motor", angle)
            for wing, angle in wing_targets.items()
        ]
        sim.step(commands)

    print(f"  Wingbeat simulation complete ({sim.time:.3f}s)")
    print(f"   - Final position: {sim.get_state().body_position}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("  - Full 3D physics simulation operational")
    print("  - Realistic fly body with articulated legs, wings, head")
    print("  - Touch sensors on all 6 tarsi")
    print("  - Brain controls physics via connectome bridge")
    print("=" * 70)

    sim.close()


if __name__ == "__main__":
    main()
