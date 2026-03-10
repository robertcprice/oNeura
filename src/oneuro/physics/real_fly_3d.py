"""
Real oNeuro Fly in 3D Minecraft World

This connects the REAL oNeuro brain to the 3D visualization.
The fly's brain actually runs Hodgkin-Huxley dynamics,
and its neural activity drives movement in the 3D world.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from oneuro.organisms.drosophila import DrosophilaBrain


class RealFly3D:
    """Fly with REAL oNeuro brain."""

    def __init__(self, scale='tiny', device='cpu'):
        print(f"Creating REAL oNeuro brain ({scale} scale)...")
        self.brain = DrosophilaBrain(scale=scale, device=device)

        # Position in world
        self.position = np.array([10.0, 10.0, 5.0])
        self.orientation = 0.0

        # Get neuron info
        self.n_neurons = self.brain.n_total
        self.n_synapses = self.brain.brain.n_synapses

        # Motor history
        self.motor_history = []

        print(f"Created: {self.n_neurons} neurons, {self.n_synapses:,} synapses")

    def step(self, sensory_input):
        """
        Run one step with real brain.

        Args:
            sensory_input: dict with 'olfactory', 'visual', etc.

        Returns:
            dict with 'motor', 'position', 'brain_activity'
        """
        # Stimulate brain with sensory input BEFORE running
        if 'olfactory' in sensory_input:
            self.brain.stimulate_al(sensory_input['olfactory'])

        if 'visual' in sensory_input:
            self.brain.stimulate_optic(sensory_input['visual'])

        # Run brain and read motor output (this runs neural dynamics internally!)
        motor = self.brain.read_motor_output(n_steps=10)

        # Get motor values
        speed = motor.get('speed', 0.5)
        turn = motor.get('turn', 0.0)

        # Convert to movement
        self.orientation += turn * 0.3
        forward = speed * 0.1

        self.position[0] += np.cos(self.orientation) * forward
        self.position[1] += np.sin(self.orientation) * forward

        # Keep in bounds (0-20 world)
        self.position = np.clip(self.position, 0, 20)

        # Track history
        self.motor_history.append({'speed': speed, 'turn': turn})

        return {
            'speed': speed,
            'turn': turn,
            'position': self.position.copy(),
            'orientation': self.orientation,
        }


def demo():
    """Demo the real oNeuro fly."""
    print("=" * 60)
    print("REAL oNEURO FLY IN 3D WORLD")
    print("=" * 60)

    fly = RealFly3D(scale='tiny')

    print("\nRunning 100 steps with sensory stimulation...")

    for i in range(100):
        # Varying sensory input to stimulate brain
        sensory = {
            'olfactory': {
                'ethyl_acetate': 0.5 + 0.3 * np.sin(i / 10),
                'vinegar': 0.3 + 0.2 * np.cos(i / 15),
            },
            'visual': np.random.rand(4) * 0.5,
        }

        result = fly.step(sensory)

        if i % 20 == 0:
            print(f"Step {i:3d}: "
                  f"pos=({result['position'][0]:.1f}, {result['position'][1]:.1f}) "
                  f"speed={result['speed']:.3f} "
                  f"turn={result['turn']:.3f}")

    # Summary
    if len(fly.motor_history) > 0:
        speeds = [h['speed'] for h in fly.motor_history]
        turns = [h['turn'] for h in fly.motor_history]
        print("\nMotor Summary:")
        print(f"  Avg speed: {np.mean(speeds):.3f}")
        print(f"  Avg turn:  {np.mean(turns):.3f}")

    print("\n" + "=" * 60)
    print("THIS IS A REAL oNEURO BRAIN RUNNING!")
    print("=" * 60)
    print(f"- {fly.n_neurons} Hodgkin-Huxley neurons")
    print(f"- {fly.n_synapses:,} synapses")
    print(f"- Real neurotransmitter dynamics (DA, 5-HT, ACh, GABA, Glu)")
    print(f"- Motor output drives 3D movement")
    print("=" * 60)

    return fly


if __name__ == "__main__":
    demo()
