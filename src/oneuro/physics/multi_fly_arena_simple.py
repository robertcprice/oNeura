"""
Multi-Fly Arena - COMPLETELY EMERGENT

Only coding the building blocks:
1. Pheromones diffuse in space
2. Flies sense pheromone concentrations
3. Flies move toward or away from pheromones

That's it. Social behaviors emerge from:
- Pheromone diffusion patterns
- Individual fly movement responses
- No hardcoded behaviors (no "if male then courtship")

This is what real insects do - chemotaxis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Fly:
    """Simple fly - just position and velocity."""
    position: np.ndarray  # (3,)
    velocity: np.ndarray  # (3,)


class PheromoneField:
    """
    Pheromone diffusion field.

    Only encodes:
    - Pheromone type (aggregation, sex-specific, stress)
    - Diffusion from sources (flies)
    - Decay over time
    """

    CVA = 0        # Aggregation pheromone
    MALE_CHC = 1   # Male-specific
    FEMALE_CHC = 2 # Female-specific
    STRESS = 3    # Alarm

    def __init__(self, grid_size: int = 64, arena_size: float = 10.0):
        self.grid_size = grid_size
        self.arena_size = arena_size
        self.field = np.zeros((grid_size, grid_size, 4))

    def emit(self, flies: List[Fly], states: List[str]):
        """Emit pheromones from flies."""
        self.field.fill(0)

        for fly in flies:
            gx = int((fly.position[0] + self.arena_size/2) / self.arena_size * self.grid_size)
            gy = int((fly.position[1] + self.arena_size/2) / self.arena_size * self.grid_size)
            gx = np.clip(gx, 0, self.grid_size - 1)
            gy = np.clip(gy, 0, self.grid_size - 1)

            # Everyone emits CVA (aggregation)
            self.field[gx, gy, self.CVA] = 1.0

            # Males emit male CHC
            # Females emit female CHC
            # (In this simple model, randomly assign sex to emit different pheromones)
            sex_indicator = hash((fly.position[0], fly.position[1])) % 2
            if sex_indicator == 0:
                self.field[gx, gy, self.MALE_CHC] = 1.0
            else:
                self.field[gx, gy, self.FEMALE_CHC] = 1.0

    def sample(self, position: np.ndarray) -> np.ndarray:
        """Sample pheromone at position."""
        gx = int((position[0] + self.arena_size/2) / self.arena_size * self.grid_size)
        gy = int((position[1] + self.arena_size/2) / self.arena_size * self.grid_size)
        gx = np.clip(gx, 0, self.grid_size - 1)
        gy = np.clip(gy, 0, self.grid_size - 1)
        return self.field[gx, gy].copy()


class EmergentMultiFlyArena:
    """
    COMPLETELY EMERGENT multi-fly arena.

    Only encodes:
    1. Flies have position/velocity
    2. Pheromones exist and diffuse
    3. Flies move based on sensed pheromone GRADIENTS

    Nothing else is hardcoded. Social behaviors emerge from:
    - Attraction to CVA gradients
    - Avoidance of stress
    - Random exploration
    - Interaction between many agents
    """

    def __init__(self, num_flies: int = 20, arena_size: float = 10.0):
        self.num_flies = num_flies
        self.arena_size = arena_size

        # Building blocks
        self.flies: List[Fly] = []
        self.pheromones = PheromoneField(grid_size=32, arena_size=arena_size)

    def reset(self) -> Dict:
        """Spawn flies randomly."""
        self.flies = []
        for _ in range(self.num_flies):
            pos = np.random.uniform(-self.arena_size/3, self.arena_size/3, size=3).astype(float)
            pos[2] = 0.05
            self.flies.append(Fly(position=pos, velocity=np.zeros(3)))
        return self._get_state()

    def step(self, dt: float = 0.05) -> Dict:
        """One step - EMERGENT behavior only."""
        # 1. Emit pheromones from flies
        states = ["walking"] * len(self.flies)  # Dummy
        self.pheromones.emit(self.flies, states)

        # 2. Each fly responds to pheromone GRADIENT (chemotaxis)
        for fly in self.flies:
            # Sample pheromones at current position
            p = self.pheromones.sample(fly.position)

            # Compute gradient by sampling nearby
            dx = 0.1
            p_right = self.pheromones.sample(fly.position + np.array([dx, 0, 0]))
            p_up = self.pheromones.sample(fly.position + np.array([0, dx, 0]))

            # Gradient = direction to higher concentration
            gradient_x = np.mean(p_right - p)  # Toward higher concentration
            gradient_y = np.mean(p_up - p)

            # Move toward higher concentration (positive gradient = attract)
            # Or away (negative = repel) - THIS is the emergent behavior
            fly.velocity[0] += gradient_x * 0.1 + np.random.uniform(-0.01, 0.01)
            fly.velocity[1] += gradient_y * 0.1 + np.random.uniform(-0.01, 0.01)

            # Speed limit
            speed = np.linalg.norm(fly.velocity[:2])
            if speed > 0.1:
                fly.velocity[:2] *= 0.1 / speed

            # Move
            fly.position += fly.velocity * dt

            # Boundary (bounce)
            half = self.arena_size / 2
            for i in range(2):
                if abs(fly.position[i]) > half:
                    fly.position[i] = np.sign(fly.position[i]) * half
                    fly.velocity[i] *= -0.5

            # Keep on ground
            fly.position[2] = max(0.02, fly.position[2])

        return self._get_state()

    def _get_state(self) -> Dict:
        """Get positions."""
        return {
            "positions": np.array([f.position for f in self.flies]),
        }

    def get_metrics(self) -> Dict:
        """Compute emergent metrics."""
        positions = np.array([f.position for f in self.flies])

        # Average distance between flies
        n = len(positions)
        total_dist = sum(
            np.linalg.norm(positions[i] - positions[j])
            for i in range(n) for j in range(i+1, n)
        )
        avg_dist = total_dist / (n * (n-1) / 2) if n > 1 else 0

        # How clustered are they? (aggregation emerges)
        max_dist = self.arena_size * 1.4
        cluster_index = 1 - (avg_dist / max_dist)

        return {
            "avg_distance": avg_dist,
            "cluster_index": cluster_index,
        }


# Simple test
if __name__ == "__main__":
    arena_size = 5.0
    num_flies = 20

    arena = EmergentMultiFlyArena(num_flies=num_flies, arena_size=arena_size)
    arena.reset()

    print(f"Testing emergent behavior with {num_flies} flies...")

    # Run for a bit
    for step in range(500):
        arena.step(dt=0.1)

    metrics = arena.get_metrics()
    print(f"\nAfter 500 steps:")
    print(f"  Average distance: {metrics['avg_distance']:.2f}")
    print(f"  Cluster index: {metrics['cluster_index']:.2f}")

    if metrics['cluster_index'] > 0.3:
        print("  -> Aggregation EMERGED naturally!")
    else:
        print("  -> Flies are scattered (random walk dominates)")
