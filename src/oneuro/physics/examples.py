"""
Drosophila MuJoCo Simulation Examples

Demonstrates the integration of:
- MuJoCo physics simulation
- Brain-motor interface
- Connectome-based brain model
- Basic locomotion behaviors

Usage:
    cd /Users/bobbyprice/projects/oNeuro
    python -m oneuro.physics.examples
"""

import numpy as np


def check_mujoco_available():
    """Check if MuJoCo is installed and available."""
    try:
        import mujoco
        return True
    except ImportError:
        return False


def example_basic_simulation():
    """Basic physics simulation without brain model."""
    if not check_mujoco_available():
        print("MuJoCo not available. Install with: pip install mujoco")
        return

    from oneuro.physics.drosophila_simulator import (
        DrosophilaSimulator,
        MotorCommand,
        MotorPatterns,
    )

    print("=" * 60)
    print("Example 1: Basic Physics Simulation")
    print("=" * 60)

    # Create simulator
    with DrosophilaSimulator() as sim:
        # Reset to initial position
        state = sim.reset(position=np.array([0, 0, 0.005]))
        print(f"Initial position: {state.body_position}")
        print(f"Number of actuators: {sim.num_actuators}")
        print(f"Number of sensors: {sim.num_sensors}")

        # Run a few steps with no control
        for _ in range(100):
            state = sim.step([])

        print(f"Position after free fall: {state.body_position}")

        # Apply wing motor commands
        print("\nApplying wing motor commands...")
        commands = [
            MotorCommand("wing_L_motor", 1.0),
            MotorCommand("wing_R_motor", -1.0),
        ]

        for _ in range(1000):
            state = sim.step(commands)

        print(f"Position after wing actuation: {state.body_position}")
        print(f"Touch contacts: {state.touch_contacts}")


def example_brain_motor_interface():
    """Demonstrate brain-motor interface."""
    if not check_mujoco_available():
        print("MuJoCo not available. Install with: pip install mujoco")
        return

    from oneuro.physics.drosophila_simulator import DrosophilaSimulator
    from oneuro.physics.brain_motor_interface import BrainMotorInterface

    print("\n" + "=" * 60)
    print("Example 2: Brain-Motor Interface")
    print("=" * 60)

    with DrosophilaSimulator() as sim:
        interface = BrainMotorInterface()

        state = sim.reset()
        print(f"Sensor dimension: {interface.get_sensor_dim()}")
        print(f"Motor dimension: {interface.get_motor_dim()}")

        # Run simulation with brain-motor interface
        for step in range(1000):
            # Get physics state
            state = sim.get_state()

            # Process through interface (uses reflex controller by default)
            commands = interface.process(state, dt=0.0001)

            # Apply commands
            state = sim.step(commands)

        print(f"Final position: {state.body_position}")
        print(f"Simulation time: {sim.time:.3f}s")


def example_connectome_bridge():
    """Demonstrate connectome-based brain model."""
    print("\n" + "=" * 60)
    print("Example 3: Connectome Bridge")
    print("=" * 60)

    from oneuro.physics.connectome_bridge import (
        ConnectomeBridge,
        create_brain_model,
        NeuronType,
    )

    # Create bridge
    bridge = ConnectomeBridge(num_neurons_per_pool=50)

    print(f"Neuron pools: {list(bridge.pools.keys())}")
    print(f"Number of connections: {len(bridge.connections)}")
    print(f"Sensor input dim: {bridge.get_sensor_input_dim()}")
    print(f"Motor output dim: {bridge.get_motor_output_dim()}")

    # Set some sensory input
    sensory_input = np.random.rand(bridge.get_sensor_input_dim())
    bridge.set_sensory_input("mechanosensory_tarsi", sensory_input[:50])

    # Run simulation steps
    print("\nRunning connectome simulation...")
    for _ in range(100):
        activities = bridge.step(dt=1.0)

    # Get motor output
    motor_output = bridge.get_motor_output()
    print(f"Motor pool activities:")
    for pool_name, activity in motor_output.items():
        mean_activity = np.mean(activity)
        print(f"  {pool_name}: mean={mean_activity:.3f}")

    # Test brain model factory
    print("\nTesting brain model factory...")
    brain_model = create_brain_model(quantum_enhanced=False)

    sensor_input = np.random.rand(400)
    motor_output = brain_model(sensor_input)
    print(f"Brain model output shape: {motor_output.shape}")


def example_full_integration():
    """Full integration: connectome brain -> interface -> physics."""
    if not check_mujoco_available():
        print("MuJoCo not available. Install with: pip install mujoco")
        return

    from oneuro.physics.drosophila_simulator import DrosophilaSimulator
    from oneuro.physics.brain_motor_interface import BrainMotorInterface
    from oneuro.physics.connectome_bridge import create_brain_model

    print("\n" + "=" * 60)
    print("Example 4: Full Integration (Connectome -> Physics)")
    print("=" * 60)

    # Create components
    brain_model = create_brain_model()
    interface = BrainMotorInterface(brain_model=brain_model)

    with DrosophilaSimulator() as sim:
        state = sim.reset()

        print("Running integrated simulation...")
        for step in range(500):
            # Get physics state
            state = sim.get_state()

            # Process through brain-motor interface with connectome brain
            commands = interface.process(state, dt=0.0001)

            # Step physics
            state = sim.step(commands)

            if step % 100 == 0:
                print(f"  Step {step}: pos={state.body_position}, "
                      f"contacts={sum(state.touch_contacts.values())}")

        print(f"\nFinal state:")
        print(f"  Position: {state.body_position}")
        print(f"  Velocity: {state.body_velocity}")
        print(f"  Simulation time: {sim.time:.3f}s")


def example_motor_patterns():
    """Demonstrate predefined motor patterns."""
    if not check_mujoco_available():
        print("MuJoCo not available. Install with: pip install mujoco")
        return

    from oneuro.physics.drosophila_simulator import (
        DrosophilaSimulator,
        MotorCommand,
        MotorPatterns,
        LegID,
    )

    print("\n" + "=" * 60)
    print("Example 5: Motor Patterns (Tripod Gait)")
    print("=" * 60)

    with DrosophilaSimulator() as sim:
        state = sim.reset(position=np.array([0, 0, 0.003]))  # Low start

        print("Executing tripod gait pattern...")
        gait_frequency = 10.0  # Hz
        dt = 0.0001

        for step in range(5000):
            # Generate gait commands
            phase = (step * dt * gait_frequency) % 1.0
            targets = MotorPatterns.tripod_gait(phase)

            # Convert to motor commands
            commands = []
            for joint_name, angle in targets.items():
                motor_name = joint_name.replace("_pitch", "_motor")
                commands.append(MotorCommand(
                    actuator_name=motor_name,
                    control_signal=angle * 0.5,  # Scale down
                ))

            # Add wing beats
            wing_cmd = MotorPatterns.wingbeat(200.0, step * dt)
            for wing, angle in wing_cmd.items():
                commands.append(MotorCommand(
                    actuator_name=f"{wing}_motor",
                    control_signal=angle * 0.3,
                ))

            state = sim.step(commands)

            if step % 500 == 0:
                print(f"  Step {step}: pos={state.body_position[:2]}, "
                      f"yaw={state.joint_positions.get('C1_L_yaw', 0):.2f}")

        print(f"\nFinal position: {state.body_position}")


def main():
    """Run all examples."""
    print("Drosophila MuJoCo Simulation Examples")
    print("=" * 60)

    # Check MuJoCo availability
    if check_mujoco_available():
        print("MuJoCo is available\n")
    else:
        print("WARNING: MuJoCo not available - some examples will be skipped\n")
        print("Install MuJoCo: pip install mujoco")
        print("For visualization: pip install mujoco[viewer]\n")

    # Run examples
    try:
        example_basic_simulation()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_brain_motor_interface()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_connectome_bridge()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_full_integration()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    try:
        example_motor_patterns()
    except Exception as e:
        print(f"Example 5 failed: {e}")

    print("\n" + "=" * 60)
    print("Examples complete!")


if __name__ == "__main__":
    main()
