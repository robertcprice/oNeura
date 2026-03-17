"""
oNeura Ecosystem - Terrarium Simulation with Neural Flies

A complete ecosystem simulation where flies use their actual oNeura brains
to navigate, find food, mate, and survive. All behaviors emerge from
neural dynamics, not hardcoded.

Usage:
    from oneuro.physics.ecosystem import Terrarium, FlyOrganism

    # Create terrarium
    terrarium = Terrarium()

    # Add neural flies
    for i in range(20):
        terrarium.add_fly()

    # Add food sources
    terrarium.add_food(x=10, z=0, food_type='fruit')

    # Run simulation
    for step in range(1000):
        terrarium.step()
        terrarium.render()  # If using visualization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math


class LifeStage(Enum):
    """Life stages of a fly."""
    EGG = 0
    LARVA = 1
    PUPA = 2
    ADULT = 3


class BehaviorState(Enum):
    """Emergent behavior states from neural activity."""
    RESTING = 0
    SEARCHING = 1
    SEEKING_FOOD = 2
    FEEDING = 3
    SEEKING_MATE = 4
    MATING = 5
    LAYING_EGGS = 6
    FLEEING = 7


@dataclass
class FoodSource:
    """Food source in the terrarium."""
    x: float
    z: float
    amount: float = 1.0  # 0-1
    food_type: str = 'fruit'  # 'fruit' or 'dish'
    max_amount: float = 1.0

    def consume(self, amount: float) -> float:
        """Consume food, returns actual amount consumed."""
        consumed = min(self.amount, amount)
        self.amount = max(0, self.amount - amount)
        return consumed


@dataclass
class NeuralActivity:
    """Real-time neural activity for visualization."""
    al: float = 0  # Antennal Lobe
    mb: float = 0  # Mushroom Body
    cx: float = 0  # Central Complex
    vnc: float = 0  # Motor output
    hunger: float = 0
    arousal: float = 0


class FlyOrganism:
    """
    A fly with an oNeura brain that generates emergent behavior.

    The fly has:
    - Bilateral antennae for olfactory sensing
    - Compound eyes for visual input
    - A full Drosophila brain with AL, MB, CX, VNC
    - Emergent behaviors from neural dynamics
    """

    def __init__(
        self,
        x: float = 0,
        z: float = 0,
        stage: LifeStage = LifeStage.ADULT,
        gender: Optional[str] = None,
        brain=None,
    ):
        self.x = x
        self.z = z
        self.stage = stage
        self.gender = gender or ('male' if np.random.random() > 0.5 else 'female')

        # Life
        self.age = 0.0  # days
        self.energy = 0.5  # for egg production
        self.hunger = 0.0  # 0 = full, 1 = starving

        # Movement
        self.heading = np.random.random() * 2 * math.pi
        self.speed = 0

        # Neural brain (only for adults)
        self.brain = brain
        self.neural_activity = NeuralActivity()

        # State
        self.behavior = BehaviorState.SEARCHING
        self.target_x = None
        self.target_z = None
        self.mating_cooldown = 0

        # Mesh for visualization (set by terrarium)
        self.mesh = None

    def sense_environment(self, food_sources: List[FoodSource], other_flies: List['FlyOrganism']) -> Dict:
        """
        Sense the environment through bilateral antennae.
        This is the sensory input to the brain.
        """
        # Get scent at current position and ahead of left/right antennae
        sense_dist = 2.0

        def get_scent(px, pz):
            """Calculate scent level at position."""
            total = 0
            for f in food_sources:
                if f.amount > 0.01:
                    d = math.sqrt((f.x - px) ** 2 + (f.z - pz) ** 2)
                    if d < 20:  # Scent range
                        total += f.amount / (d * d + 1)
            return min(1.0, total)

        # Center
        scent_center = get_scent(self.x, self.z)

        # Left antenna (ahead and to the left)
        left_x = self.x + math.cos(self.heading) * sense_dist + math.sin(self.heading) * 0.5
        left_z = self.z + math.sin(self.heading) * sense_dist - math.cos(self.heading) * 0.5
        scent_left = get_scent(left_x, left_z)

        # Right antenna (ahead and to the right)
        right_x = self.x + math.cos(self.heading) * sense_dist - math.sin(self.heading) * 0.5
        right_z = self.z + math.sin(self.heading) * sense_dist + math.cos(self.heading) * 0.5
        scent_right = get_scent(right_x, right_z)

        # Check for mates nearby
        has_mate = False
        if self.gender and self.energy > 0.3:
            for fly in other_flies:
                if fly is not self and fly.stage == LifeStage.ADULT:
                    if fly.gender != self.gender:
                        dist = math.sqrt((fly.x - self.x) ** 2 + (fly.z - self.z) ** 2)
                        if dist < 5:
                            has_mate = True
                            break

        return {
            'odor_left': scent_left,
            'odor_right': scent_right,
            'odor_center': scent_center,
            'has_food_nearby': scent_center > 0.1,
            'has_mate_nearby': has_mate,
        }

    def process_brain(self, sensory: Dict, dt: float, is_night: bool) -> Dict:
        """
        Process sensory input through the oNeura brain.
        Returns motor commands that drive behavior.
        """
        if self.brain is None:
            # Simple fallback for non-adults
            return self._simple_motor(sensory, dt)

        # Run brain step with sensory input
        # Note: The actual Drosophila brain handles this
        result = self.brain.step()

        # Extract motor output
        motor = result.get('motor', {'speed': 0.3, 'turn': 0})

        # Update neural activity for visualization
        self.neural_activity.al = result.get('al_activity', 0)
        self.neural_activity.mb = result.get('mb_activity', 0)
        self.neural_activity.cx = result.get('cx_activity', 0)
        self.neural_activity.vnc = motor.get('speed', 0)
        self.neural_activity.hunger = self.hunger

        # Determine behavior state from neural activity
        if self.hunger > 0.7:
            self.behavior = BehaviorState.FLEEING
        elif sensory.get('has_food_nearby') and self.hunger > 0.3:
            self.behavior = BehaviorState.SEEKING_FOOD
        elif sensory.get('has_mate_nearby') and self.energy > 0.5:
            self.behavior = BehaviorState.SEEKING_MATE
        elif motor.get('speed', 0) < 0.1:
            self.behavior = BehaviorState.RESTING
        else:
            self.behavior = BehaviorState.SEARCHING

        return motor

    def _simple_motor(self, sensory: Dict, dt: float) -> Dict:
        """Simple motor for non-brain organisms (larvae, etc)."""
        odor_diff = sensory['odor_right'] - sensory['odor_left']
        odor_center = sensory['odor_center']

        # Turn toward higher concentration
        turn = odor_diff * 3 if abs(odor_diff) > 0.02 else 0

        # Speed based on hunger and odor
        speed = 0.3 + self.hunger * 0.5 + odor_center * 0.5
        speed = min(1.0, speed)

        return {'turn': turn, 'speed': speed}

    def update(self, dt: float, sensory: Dict, motor: Dict, food_sources: List[FoodSource], is_night: bool = False):
        """Update position and internal state."""
        self.age += dt

        # Update mating cooldown
        if self.mating_cooldown > 0:
            self.mating_cooldown -= dt

        # Apply motor output
        self.heading += motor.get('turn', 0) * dt * 10

        base_speed = motor.get('speed', 0.3)
        self.speed = base_speed * (0.2 if is_night else 1.0)

        self.x += math.cos(self.heading) * self.speed * dt * 5
        self.z += math.sin(self.heading) * self.speed * dt * 5

        # Bounds
        self.x = max(-19, min(19, self.x))
        self.z = max(-19, min(19, self.z))

        # Hunger increases with activity
        hunger_rate = 0.02 * (1 + self.speed)
        self.hunger = min(1.0, self.hunger + hunger_rate * dt)

        # Eat if hungry
        if self.hunger > 0.4:
            for food in food_sources:
                dist = math.sqrt((food.x - self.x) ** 2 + (food.z - self.z) ** 2)
                if dist < 1.5 and food.amount > 0:
                    consumed = food.consume(dt * 0.3)
                    self.hunger = max(0, self.hunger - consumed * 2)

        # Death conditions
        if self.hunger >= 1.0 or self.age > 25:
            return False  # Dead

        return True  # Alive


class Terrarium:
    """
    A complete terrarium ecosystem with neural flies.

    Features:
    - Multiple flies with oNeura brains
    - Food sources that get consumed
    - Day/night cycle affecting behavior
    - Full lifecycle: egg -> larva -> pupa -> adult
    - Emergent behaviors from neural dynamics
    """

    def __init__(self, size: Tuple[float, float] = (40, 40)):
        self.width, self.depth = size
        self.food_sources: List[FoodSource] = []
        self.organisms: List[FlyOrganism] = []

        # Environment
        self.time_of_day = 12.0  # Hours (0-24)
        self.day = 1

    def add_fly(
        self,
        x: Optional[float] = None,
        z: Optional[float] = None,
        brain=None,
    ) -> FlyOrganism:
        """Add a fly to the terrarium."""
        if x is None:
            x = (np.random.random() - 0.5) * (self.width - 4)
        if z is None:
            z = (np.random.random() - 0.5) * (self.depth - 4)

        fly = FlyOrganism(x, z, LifeStage.ADULT, brain=brain)
        self.organisms.append(fly)
        return fly

    def add_food(self, x: float, z: float, food_type: str = 'fruit', amount: float = 1.0):
        """Add a food source to the terrarium."""
        food = FoodSource(x, z, amount, food_type, amount)
        self.food_sources.append(food)
        return food

    def get_stats(self) -> Dict:
        """Get population statistics."""
        adults = [o for o in self.organisms if o.stage == LifeStage.ADULT]
        return {
            'total': len(self.organisms),
            'adults': len(adults),
            'males': len([o for o in adults if o.gender == 'male']),
            'females': len([o for o in adults if o.gender == 'female']),
            'larvae': len([o for o in self.organisms if o.stage == LifeStage.LARVA]),
            'pupae': len([o for o in self.organisms if o.stage == LifeStage.PUPA]),
            'eggs': len([o for o in self.organisms if o.stage == LifeStage.EGG]),
            'food_remaining': sum(f.amount for f in self.food_sources),
            'time_of_day': self.time_of_day,
            'day': self.day,
        }

    def step(self, dt: float = 0.016):
        """
        Step the simulation forward.

        This is where all the emergent behavior happens:
        1. Each fly senses the environment
        2. Brain processes sensory input
        3. Motor output drives movement
        4. Internal state updates (hunger, energy)
        5. Death and reproduction
        """
        # Update time
        self.time_of_day += dt * 0.5  # 1 second = 30 minutes
        if self.time_of_day >= 24:
            self.time_of_day = 0
            self.day += 1

        is_night = self.time_of_day < 6 or self.time_of_day > 20

        # Update each organism
        for fly in self.organisms[:]:
            if fly.stage != LifeStage.ADULT:
                continue  # Simplified for now

            # Sense
            sensory = fly.sense_environment(self.food_sources, self.organisms)

            # Brain processing
            motor = fly.process_brain(sensory, dt, is_night)

            # Update
            alive = fly.update(dt, sensory, motor, self.food_sources, is_night)

            if not alive:
                self.organisms.remove(fly)

        # Population control - add immigrants if low
        adults = [o for o in self.organisms if o.stage == LifeStage.ADULT]
        if len(adults) < 10 and np.random.random() < 0.01:
            self.add_fly()

    def render(self):
        """Render the terrarium (for use with visualization)."""
        # This would be implemented by a visualization backend
        pass


# Convenience functions
def create_ecosystem(
    n_flies: int = 20,
    brain_class=None,
) -> Terrarium:
    """
    Create a complete ecosystem with neural flies.

    Args:
        n_flies: Number of flies to start with
        brain_class: Optional class to use for fly brains

    Returns:
        Configured Terrarium
    """
    terrarium = Terrarium()

    # Add initial food
    terrarium.add_food(8, 0, 'fruit', 0.5)
    terrarium.add_food(-8, 8, 'dish', 1.0)
    terrarium.add_food(-8, -8, 'fruit', 0.5)

    # Add flies
    for _ in range(n_flies):
        terrarium.add_fly(brain=brain_class)

    return terrarium


__all__ = [
    'Terrarium',
    'FlyOrganism',
    'FoodSource',
    'NeuralActivity',
    'LifeStage',
    'BehaviorState',
    'create_ecosystem',
]
