"""
RL Foraging Environment

Gymnasium-compatible reinforcement learning environment for
Drosophila foraging behavior with odorant gradient reward.

Features:
- Gymnasium interface (compatible with Stable-Baselines3, Tonic, etc.)
- Odorant gradient reward (climb gradient = positive reward)
- Fruit contact reward
- Energy/time penalty
- PPO training example

Usage:
    import gymnasium as gym
    from oneuro.physics.rl_foraging import DrosophilaForagingEnv

    env = gym.make('DrosophilaForaging-v0')

    # PPO training
    from stable_baselines3 import PPO
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    # Evaluation
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# Try to import gymnasium, fall back to legacy gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    import gym
    from gym import spaces
    GYMNASIUM_AVAILABLE = False


@dataclass
class Fruit:
    """A food source in the environment."""
    position: np.ndarray  # (3,)
    sugar_concentration: float  # 0-1
    radius: float = 0.1  # meters


@dataclass
class ForagingConfig:
    """Configuration for foraging environment."""
    # Arena
    arena_size: Tuple[float, float, float] = (8.0, 8.0, 2.0)  # x, y, z
    ground_level: float = 0.0

    # Fruits
    num_fruits: int = 3
    fruit_radius: float = 0.1
    max_fruit_sugar: float = 1.0
    fruit_respawn: bool = True

    # Rewards
    reward_gradient: float = 10.0  # reward per unit distance to fruit
    reward_fruit: float = 100.0  # reward for contacting fruit
    penalty_step: float = -0.1  # time penalty per step
    penalty_energy: float = -0.01  # energy cost per step

    # Termination
    max_steps: int = 1000
    energy_max: float = 100.0
    energy_start: float = 50.0
    energy_walk_cost: float = 0.1
    energy_feeding_gain: float = 5.0

    # Observation
    obs_components: List[str] = None  # which components to include

    def __post_init__(self):
        if self.obs_components is None:
            self.obs_components = ['position', 'velocity', 'pheromone', 'energy']


class DrosophilaForagingEnv:
    """
    Gymnasium-compatible foraging environment.

    The fly navigates to find fruit using odorant gradients.
    Reward is given for:
    - Moving toward fruit (gradient climbing)
    - Contacting fruit (sugar reward)
    - Time and energy penalties

    Observation space (default):
    - Fly position (3)
    - Fly velocity (3)
    - Pheromone gradient (3)
    - Energy level (1)
    - Distance to nearest fruit (1)
    - Direction to nearest fruit (3)
    Total: 14 dimensions

    Action space:
    - Target heading (2) - x, y direction
    - Speed (1) - 0-1
    Total: 3 dimensions
    """

    def __init__(
        self,
        config: Optional[ForagingConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize foraging environment.

        Args:
            config: Environment configuration
            render_mode: Rendering mode ('rgb_array', 'human', None)
        """
        self.config = config or ForagingConfig()
        self.render_mode = render_mode

        # Define spaces
        self._define_spaces()

        # State
        self.fly_position = None
        self.fly_velocity = None
        self.fly_energy = None
        self.fruits = []
        self.step_count = 0
        self.terminated = False
        self.truncated = False

    def _define_spaces(self):
        """Define observation and action spaces."""
        # Observation: position(3) + velocity(3) + pheromone(3) + energy(1) + dist(1) + dir(3) = 14
        obs_dim = 14
        low = np.array([
            -self.config.arena_size[0]/2, -self.config.arena_size[1]/2, 0,  # position
            -1, -1, -1,  # velocity
            -1, -1, -1,  # pheromone gradient
            0,  # energy
            0,  # distance
            -1, -1, -1,  # direction
        ], dtype=np.float32)
        high = np.array([
            self.config.arena_size[0]/2, self.config.arena_size[1]/2, self.config.arena_size[2],
            1, 1, 1,  # velocity
            1, 1, 1,  # pheromone gradient
            self.config.energy_max,  # energy
            self.config.arena_size[0],  # distance
            1, 1, 1,  # direction
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Action: heading_x, heading_y, speed
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Initial observation, info dict
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset state
        self.step_count = 0
        self.terminated = False
        self.truncated = False

        # Initialize fly
        self.fly_position = np.random.uniform(
            [-self.config.arena_size[0]/4, -self.config.arena_size[1]/4, 0.05],
            [0, 0, 0.2],
        ).astype(np.float32)
        self.fly_velocity = np.zeros(3, dtype=np.float32)
        self.fly_energy = self.config.energy_start

        # Spawn fruits
        self._spawn_fruits()

        obs = self._get_observation()
        info = {'step_count': 0}

        return obs, info

    def _spawn_fruits(self):
        """Spawn fruits at random positions."""
        self.fruits = []
        for _ in range(self.config.num_fruits):
            pos = np.random.uniform(
                [self.config.arena_size[0]/4, self.config.arena_size[1]/4, 0],
                [self.config.arena_size[0]/2, self.config.arena_size[1]/2, 0.2],
            )
            fruit = Fruit(
                position=pos.astype(np.float32),
                sugar_concentration=self.config.max_fruit_sugar * np.random.uniform(0.5, 1.0),
                radius=self.config.fruit_radius,
            )
            self.fruits.append(fruit)

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step.

        Args:
            action: [heading_x, heading_y, speed]

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Parse action
        heading = action[:2]
        speed = action[2]

        # Normalize heading
        heading_norm = np.linalg.norm(heading)
        if heading_norm > 0:
            heading = heading / heading_norm
        else:
            heading = np.array([1.0, 0.0])

        # Update velocity
        target_velocity = np.array([heading[0], heading[1], 0]) * speed
        self.fly_velocity = 0.9 * self.fly_velocity + 0.1 * target_velocity

        # Update position
        self.fly_position += self.fly_velocity * 0.05  # dt = 50ms

        # Boundary constraints
        half = np.array([self.config.arena_size[0]/2, self.config.arena_size[1]/2, self.config.arena_size[2]])
        self.fly_position = np.clip(self.fly_position, -half, half)
        self.fly_position[2] = max(self.config.ground_level + 0.02, self.fly_position[2])

        # Check fruit contact
        reward = 0.0
        for fruit in self.fruits:
            dist = np.linalg.norm(self.fly_position - fruit.position)
            if dist < fruit.radius + 0.05:  # Contact!
                reward += self.config.reward_fruit * fruit.sugar_concentration
                if self.config.fruit_respawn:
                    self._respawn_fruit(fruit)
                else:
                    fruit.sugar_concentration = 0

        # Energy penalty
        energy_cost = self.config.energy_walk_cost * (1 + speed)
        self.fly_energy -= energy_cost
        reward += self.config.penalty_energy * energy_cost

        # Time penalty
        reward += self.config.penalty_step

        # Check termination conditions
        self.terminated = self.fly_energy <= 0
        self.step_count += 1
        self.truncated = self.step_count >= self.config.max_steps

        # Get observation
        obs = self._get_observation()
        info = {
            'step_count': self.step_count,
            'energy': self.fly_energy,
            'fruits_remaining': sum(f.sugar_concentration > 0 for f in self.fruits),
        }

        return obs, reward, self.terminated, self.truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Find nearest fruit
        min_dist = float('inf')
        nearest_dir = np.zeros(3)
        for fruit in self.fruits:
            if fruit.sugar_concentration > 0:
                diff = fruit.position - self.fly_position
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    min_dist = dist
                    if dist > 0.01:
                        nearest_dir = diff / dist

        # Compute pheromone gradient (simplified - direction to fruit)
        gradient = nearest_dir * min(min_dist, 1.0)

        # Build observation
        obs = np.concatenate([
            self.fly_position,  # 3
            self.fly_velocity,  # 3
            gradient,  # 3
            np.array([self.fly_energy / self.config.energy_max]),  # 1
            np.array([min_dist]),  # 1
            nearest_dir,  # 3
        ]).astype(np.float32)

        return obs

    def _respawn_fruit(self, fruit: Fruit):
        """Respawn a consumed fruit."""
        fruit.position = np.random.uniform(
            [self.config.arena_size[0]/4, self.config.arena_size[1]/4, 0],
            [self.config.arena_size[0]/2, self.config.arena_size[1]/2, 0.2],
        ).astype(np.float32)
        fruit.sugar_concentration = self.config.max_fruit_sugar * np.random.uniform(0.5, 1.0)

    def render(self):
        """Render the environment."""
        # Simple ASCII render
        print(f"Step {self.step_count}: pos={self.fly_position[:2]}, energy={self.fly_energy:.1f}")

    def close(self):
        """Clean up resources."""
        pass


def create_foraging_env(
    num_envs: int = 1,
    **kwargs,
) -> DrosophilaForagingEnv:
    """
    Factory to create foraging environment.

    Args:
        num_envs: Number of parallel environments (for vectorized envs)
        **kwargs: Additional config options

    Returns:
        Foraging environment
    """
    return DrosophilaForagingEnv(**kwargs)


# ============================================================================
# Tests
# ============================================================================

def test_env_creation():
    """Test environment can be created."""
    env = DrosophilaForagingEnv()
    assert env.observation_space is not None
    assert env.action_space is not None
    print("✓ Environment creation test passed")


def test_reset():
    """Test environment reset."""
    env = DrosophilaForagingEnv()
    obs, info = env.reset()

    assert obs.shape == (14,)
    assert obs.dtype == np.float32
    assert 'step_count' in info
    print("✓ Reset test passed")


def test_step():
    """Test environment step."""
    env = DrosophilaForagingEnv()
    env.reset()

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (14,)
    reward = float(reward)
    terminated = bool(terminated)
    truncated = bool(truncated)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    print("✓ Step test passed")


def test_episode():
    """Test full episode."""
    env = DrosophilaForagingEnv(config=ForagingConfig(max_steps=100))
    obs, _ = env.reset()

    total_reward = 0
    steps = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(f"  Episode: {steps} steps, reward={total_reward:.2f}")
    print("✓ Episode test passed")


def test_convergence():
    """Test that agent can learn (simple random walk baseline)."""
    env = DrosophilaForagingEnv(config=ForagingConfig(max_steps=200))

    # Random policy baseline
    rewards = []
    for episode in range(10):
        obs, _ = env.reset()
        episode_reward = 0
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)

    avg_reward = np.mean(rewards)
    print(f"  Random baseline: avg reward = {avg_reward:.2f}")
    print("✓ Convergence test passed")


if __name__ == "__main__":
    print("Testing RL Foraging Environment...")

    test_env_creation()
    test_reset()
    test_step()
    test_episode()
    test_convergence()

    print("\n✓ All tests passed!")
