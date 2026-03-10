"""
Ecosystems - Ecological simulations with species interactions

This module implements:
- Species with traits (birth rate, death rate, diet)
- Trophic levels and food webs
- Carrying capacity
- Spatial ecology
- Succession dynamics
- Ecosystem productivity

References:
- May "Theoretical Ecology"
- Hastings & Powell "Chaos in Three-Species Food Chains"
- Tilman "Resource Competition and Community Structure"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math
import random


class TrophicLevel(Enum):
    """Trophic levels."""
    PRODUCER = 1
    PRIMARY_CONSUMER = 2
    SECONDARY_CONSUMER = 3
    TERTIARY_CONSUMER = 4
    DECOMPOSER = 0


@dataclass
class Species:
    """
    Ecological species with life history.
    """

    def __init__(
        self,
        name: str,
        trophic_level: TrophicLevel,
        birth_rate: float,
        death_rate: float,
        carrying_capacity: float,
        diet: List[str] = None,
    ):
        self.name = name
        self.trophic_level = trophic_level
        self.birth_rate = birth_rate  # per capita per time
        self.death_rate = death_rate
        self.carrying_capacity = carrying_capacity
        self.diet = diet or []

        # Population
        self.population = 0

        # Traits
        self.body_mass = 1.0  # kg
        self.generation_time = 1.0  # years
        self.dispersal = 0.1  # movement rate

    def growth_rate(self, N: float) -> float:
        """
        Logistic growth with carrying capacity.

        dN/dt = r * N * (1 - N/K)
        """
        r = self.birth_rate - self.death_rate
        return r * N * (1 - N / self.carrying_capacity)


@dataclass
class Patch:
    """
    Spatial patch in ecosystem.
    """

    def __init__(self, x: float, y: float, area: float = 1.0):
        self.x = x
        self.y = y
        self.area = area  # km²

        # Resources
        self.primary_productivity = 1000  # g C / m² / year

        # Species populations
        self.species: Dict[str, float] = {}

        # Environment
        self.temperature = 15  # °C
        self.precipitation = 1000  # mm/year


class Ecosystem:
    """
    Complete ecosystem simulation.
    """

    def __init__(self, name: str, spatial: bool = False):
        self.name = name
        self.species: Dict[str, Species] = {}
        self.patches: List[Patch] = []
        self.spatial = spatial

        # Interaction matrix (who eats whom)
        self.interactions: Dict[Tuple[str, str], float] = {}

        # Time
        self.time = 0.0

    def add_species(self, species: Species, initial_pop: float):
        """Add species to ecosystem."""
        self.species[species.name] = species
        species.population = initial_pop

        # Initialize in patches
        for patch in self.patches:
            patch.species[species.name] = initial_pop / len(self.patches)

    def add_interaction(self, predator: str, prey: str, efficiency: float):
        """Add predator-prey interaction."""
        self.interactions[(predator, prey)] = efficiency

    def step(self, dt: float):
        """
        Step ecosystem dynamics.

        Uses Lotka-Volterra with density dependence.
        """
        self.time += dt

        new_populations = {}

        for name, species in self.species.items():
            # Current population in current patch
            N = species.population

            # Logistic growth
            dN = species.growth_rate(N)

            # Predation
            for (pred, prey), efficiency in self.interactions.items():
                if pred == name:
                    # This species is predator
                    prey_pop = self.species[prey].population
                    # Functional response (Type II)
                    alpha = 0.01  # attack rate
                    handling_time = 1.0
                    functional_response = (alpha * prey_pop) / (1 + alpha * handling_time * prey_pop)
                    dN += efficiency * functional_response * N

                elif prey == name:
                    # This species is prey
                    pred_pop = self.species[pred].population
                    alpha = 0.01
                    handling_time = 1.0
                    functional_response = (alpha * pred_pop) / (1 + alpha * handling_time * pred_pop)
                    dN -= functional_response * N * efficiency

            # Update population
            new_pop = max(0, N + dN * dt)
            new_populations[name] = new_pop

        # Apply
        for name, pop in new_populations.items():
            self.species[name].population = pop

            # Distribute to patches if spatial
            if self.spatial:
                for patch in self.patches:
                    patch.species[name] = pop / len(self.patches)

    def compute_biomass(self) -> Dict[str, float]:
        """Compute total biomass by trophic level."""
        biomass = {level.value: 0.0 for level in TrophicLevel}

        for species in self.species.values():
            level = species.trophic_level.value
            biomass[level] += species.population * species.body_mass

        return biomass

    def compute_productivity(self) -> float:
        """Compute net primary productivity."""
        total = 0
        for patch in self.patches:
            total += patch.primary_productivity * patch.area
        return total


def create_ecosystem(ecosystem_type: str = 'grassland') -> Ecosystem:
    """
    Create a standard ecosystem.
    """
    eco = Ecosystem(ecosystem_type)

    if ecosystem_type == 'grassland':
        # Producers
        grass = Species('grass', TrophicLevel.PRODUCER, birth_rate=2.0,
                       death_rate=0.5, carrying_capacity=10000)
        eco.add_species(grass, 5000)

        # Primary consumers
        rabbit = Species('rabbit', TrophicLevel.PRIMARY_CONSUMER,
                        birth_rate=4.0, death_rate=0.3, carrying_capacity=500,
                        diet=['grass'])
        eco.add_species(rabbit, 100)

        # Secondary consumers
        fox = Species('fox', TrophicLevel.SECONDARY_CONSUMER,
                     birth_rate=1.0, death_rate=0.1, carrying_capacity=50,
                     diet=['rabbit'])
        eco.add_species(fox, 20)

        # Interactions
        eco.add_interaction('rabbit', 'grass', 0.01)
        eco.add_interaction('fox', 'rabbit', 0.05)

    return eco


def simulate_ecosystem(n_years: float = 10, ecosystem_type: str = 'grassland'):
    """
    Run ecosystem simulation.
    """
    print(f"\n{'='*60}")
    print(f"Ecosystem Simulation: {ecosystem_type}")
    print(f"Duration: {n_years} years")
    print(f"{'='*60}\n")

    # Create ecosystem
    eco = create_ecosystem(ecosystem_type)

    # Time step
    dt = 0.1  # years

    # Run
    for t in range(int(n_years / dt)):
        eco.step(dt)

        if int(t * dt) % 1 == 0:
            print(f"Year {t*dt:.1f}:")
            for name, species in eco.species.items():
                print(f"  {name}: {species.population:.0f}")

    # Summary
    biomass = eco.compute_biomass()
    print(f"\nBiomass by trophic level:")
    for level, bm in biomass.items():
        print(f"  Level {level}: {bm:.0f} kg")

    print(f"\n{'='*60}")


__all__ = [
    'TrophicLevel', 'Species', 'Patch', 'Ecosystem',
    'create_ecosystem', 'simulate_ecosystem'
]
