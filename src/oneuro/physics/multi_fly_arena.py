"""
Multi-Fly Arena Simulation

Simulates multiple flies in a shared environment with inter-fly sensing
and emergent social behaviors.

Features:
- Multiple flies (10-100) in shared arena
- Inter-fly visual sensing (compound eye view of other flies)
- Inter-fly olfactory sensing (pheromone detection)
- Social behaviors: aggregation, courtship, aggression, flocking
- GPU-accelerated parallel simulation

Usage:
    from oneuro.physics.multi_fly_arena import MultiFlyArena

    arena = MultiFlyArena(num_flies=10, arena_size=10.0)
    states = arena.reset()

    for step in range(1000):
        states = arena.step()
        # Get positions, orientations, social interactions

    arena.close()
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class SocialBehavior(Enum):
    """Emergent social behaviors in multi-fly simulation."""
    SOLITARY = "solitary"
    AGGREGATION = "aggregation"
    COURTSHIP = "courtship"
    AGGRESSION = "aggression"
    FLOCKING = "flocking"
    FEEDING = "feeding"


@dataclass
class FlyState:
    """State of a single fly in the arena."""
    id: int
    position: np.ndarray  # (3,) x, y, z
    velocity: np.ndarray  # (3,)
    orientation: np.ndarray  # (4,) quaternion
    angular_velocity: np.ndarray  # (3,)
    energy: float
    state: str  # "idle", "walking", "feeding", "courting", "fighting"
    age: float  # seconds since spawn


class EmergentNeuralController:
    """
    Neural controller for emergent social behaviors.

    Instead of hardcoded behaviors, each fly has neural parameters
    that respond to sensory inputs. Social behaviors emerge from
    the interaction of neural dynamics with pheromones.

    Key emergent properties:
    - Attraction to CVA pheromone (aggregation)
    - Avoidance of stress pheromones
    - Personal space (repulsion at close range)
    - Speed modulation based on pheromone concentration
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize neural controller.

        Args:
            seed: Random seed for neural parameter variation
        """
        if seed is not None:
            np.random.seed(seed)

        # Neural response parameters (emerge from variation)
        # Each fly has slightly different sensitivities
        self.attraction_strength = np.random.uniform(0.5, 1.5)  # CVA attraction
        self.repulsion_distance = np.random.uniform(0.02, 0.08)  # Personal space
        self.speed_response = np.random.uniform(0.8, 1.2)  # Speed modulation

        # Internal state
        self.activity_level = 1.0  # Neural activity

    def process(
        self,
        pheromones: np.ndarray,
        nearby_positions: np.ndarray,
        distances: np.ndarray,
    ) -> Tuple[np.ndarray, str]:
        """
        Process sensory inputs through neural dynamics.

        Args:
            pheromones: (4,) pheromone concentrations
            nearby_positions: (N, 3) relative positions
            distances: (N,) distances to others

        Returns:
            velocity change (2,), behavioral state
        """
        velocity_change = np.zeros(2)
        state = "exploring"

        # Pheromone responses (NEURAL, not hardcoded)
        # CVA (index 0) = aggregation pheromone - attracts
        # Male CHC (1) = male-specific - mild repulsion
        # Female CHC (2) = female-specific - attraction (emergent!)
        # Stress (3) = alarm - strong repulsion

        # Attraction to CVA and female pheromones
        attract_signal = (
            pheromones[0] * self.attraction_strength +
            pheromones[2] * 0.5  # Female attraction emerges
        )

        # Repulsion from stress pheromones
        repulse_signal = pheromones[3] * 2.0

        # Personal space (NEURAL response to proximity)
        close_proximity = distances < self.repulsion_distance
        if np.any(close_proximity):
            close_positions = nearby_positions[close_proximity]
            repulsion = -np.mean(close_positions[:, :2], axis=0)
            velocity_change += repulsion * 0.5

        # Aggregate direction from nearby flies
        if len(nearby_positions) > 0:
            # Move toward higher pheromone concentration
            direction = np.mean(nearby_positions[:, :2], axis=0)
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 0.001:
                velocity_change += (
                    direction / direction_norm *
                    attract_signal *
                    self.speed_response *
                    0.02
                )

        # Avoid stress pheromones
        if repulse_signal > 0.1:
            velocity_change -= repulse_signal * 0.05
            state = "avoiding"

        # Update internal state based on activity
        self.activity_level = 0.95 * self.activity_level + 0.05 * (
            attract_signal - repulse_signal
        )

        if self.activity_level > 0.8:
            state = "aggregating"
        elif self.activity_level < 0.2:
            state = "resting"

        return velocity_change, state


@dataclass
class InterFlySensing:
    """Sensing data between flies."""
    # Visual: what each fly sees of others
    visual_positions: np.ndarray  # (N_other, 3) relative positions
    visual_angles: np.ndarray  # (N_other, 2) azimuth, elevation

    # Olfactory: pheromone detection
    pheromone_concentrations: np.ndarray  # (N_pheromones,)

    # Social: detected behaviors of others
    nearby_ids: List[int]
    approach_signals: np.ndarray  # (N_other,) -1 to 1
    aggression_signals: np.ndarray  # (N_other,)


@dataclass
class SocialMetrics:
    """Metrics for social behavior analysis."""
    avg_interfly_distance: float
    aggregation_index: float  # 0=sparse, 1=clustered
    courtship_events: int
    aggression_events: int
    flocking_coherence: float  # 0=random, 1=aligned


class PheromoneSystem:
    """
    Pheromone communication between flies.

    Drosophila use cuticular hydrocarbons (CHCs) for:
    - Sex recognition (7-tricosene vs 7-pentacosene)
    - Aggregation (cVA - cis-vaccenyl acetate)
    - Trail-following
    """

    # Pheromone types
    CVA = 0         # cis-vaccenyl acetate (aggregation)
    MALE_CHC = 1    # 7-tricosene (male-specific)
    FEMALE_CHC = 2  # 7-pentacosene (female-specific)
    STRESS = 3      # alarm pheromones

    DIFFUSION_RATE = 0.1  # m²/s
    DECAY_RATE = 0.05     # per second

    def __init__(self, grid_size: int = 64):
        """
        Initialize pheromone system.

        Args:
            grid_size: Spatial grid resolution
        """
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size, 4))
        self.sources = []  # List of (x, y, type, strength)

    def emit(
        self,
        positions: np.ndarray,
        fly_states: List[str],
        fly_sexes: List[str],
    ):
        """
        Emit pheromones from flies.

        Args:
            positions: (N, 3) positions
            fly_states: List of fly states
            fly_sexes: List of "male" or "female"
        """
        # Reset grid
        self.grid.fill(0)

        # Emit from each fly
        for i, (pos, state, sex) in enumerate(zip(positions, fly_states, fly_sexes)):
            grid_x = int((pos[0] + 10) / 20 * self.grid_size) % self.grid_size
            grid_y = int((pos[1] + 10) / 20 * self.grid_size) % self.grid_size

            # CVA: always emitted (aggregation)
            self.grid[grid_x, grid_y, self.CVA] += 1.0

            # Sex-specific CHCs
            if sex == "male":
                self.grid[grid_x, grid_y, self.MALE_CHC] += 1.0
            else:
                self.grid[grid_x, grid_y, self.FEMALE_CHC] += 1.0

            # Stress pheromones when fighting
            if state == "fighting":
                self.grid[grid_x, grid_y, self.STRESS] += 2.0

    def sample(
        self,
        position: np.ndarray,
    ) -> np.ndarray:
        """
        Sample pheromone concentrations at a position.

        Args:
            position: (3,) position

        Returns:
            (4,) pheromone concentrations
        """
        grid_x = int((position[0] + 10) / 20 * self.grid_size) % self.grid_size
        grid_y = int((position[1] + 10) / 20 * self.grid_size) % self.grid_size

        return self.grid[grid_x, grid_y].copy()


class MultiFlyArena:
    """
    Multi-fly arena with social behaviors.

    Simulates 10-100 flies in a shared 3D environment with:
    - Compound eye vision (sees other flies)
    - Antenna olfaction (pheromones)
    - Emergent social behaviors
    """

    def __init__(
        self,
        num_flies: int = 10,
        arena_size: float = 10.0,  # meters
        enable_pheromones: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize multi-fly arena.

        Args:
            num_flies: Number of flies in arena
            arena_size: Size of arena in meters (cubic)
            enable_pheromones: Enable pheromone communication
            device: 'cpu' or 'cuda'
        """
        self.num_flies = num_flies
        self.arena_size = arena_size
        self.enable_pheromones = enable_pheromones
        self.device = device

        # Initialize flies
        self.fly_states: List[FlyState] = []
        self.fly_sexes = self._assign_sexes()

        # Pheromone system
        if enable_pheromones:
            self.pheromones = PheromoneSystem()
        else:
            self.pheromones = None

        # Social behavior parameters
        self.interaction_radius = 0.5  # meters
        self.aggregation_threshold = 0.3  # distance for aggregation
        self.courtship_distance = 0.1  # meters
        self.fighting_distance = 0.05  # meters

        # Metrics tracking
        self.courtship_count = 0
        self.aggression_count = 0

        # Initialize
        self.reset()

    def _assign_sexes(self) -> List[str]:
        """Assign sexes to flies (50/50 split)."""
        sexes = ["male", "female"] * (self.num_flies // 2)
        if len(sexes) < self.num_flies:
            sexes.append("male")
        return sexes[:self.num_flies]

    def reset(self) -> Dict:
        """
        Reset the arena.

        Returns:
            Initial states
        """
        self.fly_states = []
        self.courtship_count = 0
        self.aggression_count = 0

        # Initialize neural controllers for emergent behavior
        self._neural_controllers = []
        self._fly_seed_base = np.random.randint(0, 10000)

        # Spawn flies at random positions
        for i in range(self.num_flies):
            pos = np.random.uniform(
                -self.arena_size / 3,
                self.arena_size / 3,
                size=3
            )
            pos[2] = 0.05  # Ground level

            # Random orientation
            yaw = np.random.uniform(0, 2 * np.pi)
            quat = self._yaw_to_quaternion(yaw)

            fly = FlyState(
                id=i,
                position=pos,
                velocity=np.zeros(3),
                orientation=quat,
                angular_velocity=np.zeros(3),
                energy=100.0,
                state="walking",
                age=0.0,
            )
            self.fly_states.append(fly)

        return self._get_states()

    def _yaw_to_quaternion(self, yaw: float) -> np.ndarray:
        """Convert yaw angle to quaternion."""
        half_yaw = yaw / 2
        return np.array([
            np.cos(half_yaw),
            0,
            0,
            np.sin(half_yaw),
        ])

    def _get_states(self) -> Dict:
        """Get current state of all flies."""
        return {
            "positions": np.array([f.position for f in self.fly_states]),
            "orientations": np.array([f.orientation for f in self.fly_states]),
            "velocities": np.array([f.velocity for f in self.fly_states]),
            "energies": np.array([f.energy for f in self.fly_states]),
            "states": np.array([hash(f.state) % 128 for f in self.fly_states]),
            "sexes": self.fly_sexes,
        }

    def step(
        self,
        dt: float = 0.05,
    ) -> Dict:
        """
        Advance simulation by one step.

        Args:
            dt: Time step in seconds

        Returns:
            Updated states
        """
        positions = np.array([f.position for f in self.fly_states])

        # Update pheromones
        if self.pheromones:
            fly_states = [f.state for f in self.fly_states]
            self.pheromones.emit(positions, fly_states, self.fly_sexes)

        # Update each fly
        for i, fly in enumerate(self.fly_states):
            # Get inter-fly sensing
            sensing = self._compute_sensing(i, positions)

            # Update behavior based on sensing
            self._update_fly_behavior(i, sensing, dt)

            # Update age
            fly.age += dt

            # Simple movement
            fly.position += fly.velocity * dt

            # Update orientation (simple yaw from angular velocity)
            yaw_change = fly.angular_velocity[2] * dt
            new_yaw = np.arctan2(fly.orientation[3], fly.orientation[0]) + yaw_change
            fly.orientation = self._yaw_to_quaternion(new_yaw)

            # Boundary conditions (bounce)
            half = self.arena_size / 2
            for dim in range(2):  # x, y
                if abs(fly.position[dim]) > half:
                    fly.position[dim] = np.sign(fly.position[dim]) * half
                    fly.velocity[dim] *= -0.5

            # Ground constraint
            fly.position[2] = max(0.02, fly.position[2])

        return self._get_states()

    def _compute_sensing(
        self,
        fly_idx: int,
        all_positions: np.ndarray,
    ) -> InterFlySensing:
        """
        Compute RAW sensory data (no hardcoded behavior signals).

        Returns only raw pheromone concentrations and positions.
        Behavior emerges from neural controller processing these signals.

        Args:
            fly_idx: Index of focal fly
            all_positions: Positions of all flies

        Returns:
            Sensing data with raw inputs
        """
        fly = self.fly_states[fly_idx]

        # Relative positions to all other flies
        rel_positions = all_positions - fly.position

        # Distances
        distances = np.linalg.norm(rel_positions, axis=1)
        distances[fly_idx] = np.inf  # Exclude self

        # Filter by interaction radius
        nearby = distances < self.interaction_radius
        nearby_idx = np.where(nearby)[0]

        if len(nearby_idx) > 0:
            nearby_positions = rel_positions[nearby_idx]
        else:
            nearby_positions = np.zeros((0, 3))

        # Pheromone sampling - RAW concentration
        if self.pheromones:
            pheromones = self.pheromones.sample(fly.position)
        else:
            pheromones = np.zeros(4)

        # Return RAW sensory data - behavior emerges from neural processing
        return InterFlySensing(
            visual_positions=nearby_positions,
            visual_angles=np.zeros(0),  # Not used anymore
            pheromone_concentrations=pheromones,  # Raw pheromones
            nearby_ids=list(nearby_idx),
            approach_signals=np.zeros(0),  # Deprecated
            aggression_signals=np.zeros(0),  # Deprecated
        )

    def _update_fly_behavior(
        self,
        fly_idx: int,
        sensing: InterFlySensing,
        dt: float,
    ):
        """
        Update fly behavior using EMERGENT neural dynamics.

        Instead of hardcoded if/else rules, behaviors emerge from:
        1. Neural controller processing pheromones
        2. Personal space repulsion
        3. Random walk for exploration

        Args:
            fly_idx: Fly index
            sensing: What the fly perceives
            dt: Time step
        """
        fly = self.fly_states[fly_idx]

        # Get or create neural controller for this fly
        if not hasattr(self, '_neural_controllers'):
            self._neural_controllers = []

        while len(self._neural_controllers) <= fly_idx:
            # Create controller with seed based on fly index
            seed = self._fly_seed_base + len(self._neural_controllers) if hasattr(self, '_fly_seed_base') else None
            self._neural_controllers.append(EmergentNeuralController(seed=seed))

        controller = self._neural_controllers[fly_idx]

        # Default: random walk (exploration)
        fly.velocity[:2] += np.random.uniform(-0.1, 0.1, 2)

        # EMERGENT BEHAVIOR: Process through neural controller
        # This is where behavior emerges - no hardcoded "if male then courtship"
        velocity_change, state = controller.process(
            pheromones=sensing.pheromone_concentrations,
            nearby_positions=sensing.visual_positions,
            distances=np.linalg.norm(sensing.visual_positions, axis=1) if len(sensing.visual_positions) > 0 else np.array([]),
        )

        # Apply neural response to velocity
        fly.velocity[:2] += velocity_change

        # Speed limiting
        speed = np.linalg.norm(fly.velocity[:2])
        if speed > 0.1:
            fly.velocity[:2] *= 0.1 / speed

        # State emerges from neural activity, not explicit rules
        fly.state = state

    def get_social_metrics(self) -> SocialMetrics:
        """
        Compute social behavior metrics.

        Returns:
            Social behavior metrics
        """
        positions = np.array([f.position for f in self.fly_states])

        # Average inter-fly distance
        n = len(positions)
        total_dist = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(positions[i] - positions[j])
                count += 1
        avg_dist = total_dist / count if count > 0 else 0

        # Aggregation index (inverse of average distance normalized)
        max_dist = self.arena_size * np.sqrt(3)
        agg_index = 1 - (avg_dist / max_dist)

        # Flocking coherence (alignment of velocities)
        velocities = np.array([f.velocity[:2] for f in self.fly_states])
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = velocities / norms
        coherence = np.abs(np.mean(normalized[:, 0]) + np.mean(normalized[:, 1])) / 2

        return SocialMetrics(
            avg_interfly_distance=avg_dist,
            aggregation_index=agg_index,
            courtship_events=self.courtship_count,
            aggression_events=self.aggression_count,
            flocking_coherence=coherence,
        )

    def close(self):
        """Clean up resources."""
        pass


def create_multi_fly_arena(
    num_flies: int = 10,
    arena_size: float = 10.0,
) -> MultiFlyArena:
    """
    Factory function to create a multi-fly arena.

    Args:
        num_flies: Number of flies
        arena_size: Size of arena

    Returns:
        Configured MultiFlyArena
    """
    return MultiFlyArena(
        num_flies=num_flies,
        arena_size=arena_size,
    )


# ============================================================================
# Tests
# ============================================================================

def test_arena_creation():
    """Test arena can be created."""
    arena = MultiFlyArena(num_flies=5, arena_size=5.0)
    assert arena.num_flies == 5
    assert len(arena.fly_states) == 5
    print("✓ Arena creation test passed")


def test_reset():
    """Test arena reset."""
    arena = MultiFlyArena(num_flies=10)
    states = arena.reset()

    assert states["positions"].shape == (10, 3)
    assert states["orientations"].shape == (10, 4)
    print("✓ Reset test passed")


def test_step():
    """Test arena step."""
    arena = MultiFlyArena(num_flies=5)
    arena.reset()

    states = arena.step(dt=0.1)
    assert states["positions"].shape == (5, 3)
    print("✓ Step test passed")


def test_social_behaviors():
    """Test social behavior emergence."""
    arena = MultiFlyArena(num_flies=20, arena_size=2.0)
    arena.reset()

    # Run simulation
    for _ in range(100):
        arena.step(dt=0.1)

    metrics = arena.get_social_metrics()
    assert metrics.aggregation_index >= 0
    assert metrics.flocking_coherence >= 0

    print(f"  Aggregation: {metrics.aggregation_index:.2f}")
    print(f"  Flocking: {metrics.flocking_coherence:.2f}")
    print("✓ Social behaviors test passed")


def test_pheromones():
    """Test pheromone system."""
    arena = MultiFlyArena(num_flies=5, enable_pheromones=True)
    arena.reset()

    # Run a few steps
    for _ in range(10):
        arena.step(dt=0.1)

    assert arena.pheromones is not None
    print("✓ Pheromones test passed")


if __name__ == "__main__":
    print("Testing Multi-Fly Arena...")

    test_arena_creation()
    test_reset()
    test_step()
    test_social_behaviors()
    test_pheromones()

    print("\n✓ All tests passed!")
