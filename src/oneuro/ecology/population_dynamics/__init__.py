"""
Population Dynamics - Mathematical models of population change

This module implements:
- Exponential and logistic growth
- Predator-prey (Lotka-Volterra)
- Competition (Lotka-Volterra)
- Allee effects
- Age-structured populations (Leslie matrix)
- Stochastic population models
- Extinction risk analysis

References:
- Lotka "Elements of Physical Biology"
- Volterra "Variazioni e fluttuazioni del numero d'individui"
- May (1976) "Simple mathematical models"
- Dennis (1989) "Allee effects"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import math
import random


class GrowthModel(Enum):
    """Population growth models."""
    EXPONENTIAL = 'exponential'
    LOGISTIC = 'logistic'
    GOMPERTZ = 'gompertz'
    ALLEE = 'allee'


@dataclass
class Population:
    """
    Population with demographic parameters.
    """

    def __init__(
        self,
        name: str,
        initial_size: float,
        birth_rate: float,
        death_rate: float,
        carrying_capacity: float = None,
        model: GrowthModel = GrowthModel.EXPONENTIAL,
    ):
        self.name = name
        self.N = initial_size
        self.r = birth_rate  # Intrinsic growth rate
        self.m = death_rate
        self.K = carrying_capacity
        self.model = model

        # History
        self.history = [initial_size]

    def growth_rate(self, N: float = None) -> float:
        """
        Compute per-capita growth rate.

        dN/dt = f(N)
        """
        N = N or self.N

        if self.model == GrowthModel.EXPONENTIAL:
            # dN/dt = rN
            return self.r - self.m

        elif self.model == GrowthModel.LOGISTIC:
            # dN/dt = rN(1 - N/K)
            return (self.r - self.m) * (1 - N / self.K) if self.K else self.r - self.m

        elif self.model == GrowthModel.GOMPERTZ:
            # dN/dt = rN ln(K/N)
            if N > 0 and self.K:
                return (self.r - self.m) * math.log(self.K / N)
            return 0

        elif self.model == GrowthModel.ALLEE:
            # dN/dt = rN (1 - N/K) (N/A - 1)  where A is Allee threshold
            if self.K:
                A = self.K * 0.2  # Allee threshold at 20% of K
                return (self.r - self.m) * (1 - N / self.K) * (N / A - 1)
            return self.r - self.m

        return self.r - self.m

    def step(self, dt: float = 1.0):
        """
        Step population forward.

        Uses Euler method.
        """
        dN = self.N * self.growth_rate() * dt
        self.N = max(0, self.N + dN)
        self.history.append(self.N)

        return self.N


class PredatorPrey:
    """
    Lotka-Volterra predator-prey model.
    """

    def __init__(
        self,
        prey_name: str,
        predator_name: str,
        prey_N0: float,
        predator_N0: float,
        prey_birth: float,
        prey_death: float,
        predator_birth: float,
        predator_death: float,
        attack_rate: float,
        conversion_efficiency: float,
    ):
        self.prey = Population(prey_name, prey_N0, prey_birth, prey_death)
        self.predator = Population(predator_name, predator_N0, predator_birth, predator_death)

        # Interaction parameters
        self.alpha = attack_rate  # Attack rate
        self.efficiency = conversion_efficiency  # Conversion efficiency
        self.handler_time = 1.0  # Handling time

        # History
        self.prey_history = [prey_N0]
        self.predator_history = [predator_N0]

    def step(self, dt: float = 0.1):
        """
        Step predator-prey system.

        dN/dt = αN - βNP (prey)
        dP/dt = δNP - γP (predator)

        Using Type II functional response.
        """
        N = self.prey.N
        P = self.predator.N

        # Functional response: αN / (1 + αhN)
        functional_response = (self.alpha * N) / (1 + self.alpha * self.handler_time * N)

        # Prey growth
        prey_growth = self.prey.r * N  # Birth
        prey_predation = functional_response * P  # Death to predators

        # Predator growth
        predator_growth = self.efficiency * functional_response * P
        predator_death = self.predator.m * P

        # Update
        dN = (prey_growth - prey_predation) * dt
        dP = (predator_growth - predator_death) * dt

        self.prey.N = max(0, self.prey.N + dN)
        self.predator.N = max(0, self.predator.N + dP)

        self.prey_history.append(self.prey.N)
        self.predator_history.append(self.predator.N)


class Competition:
    """
    Lotka-Volterra competition model.
    """

    def __init__(
        self,
        species1: Population,
        species2: Population,
        alpha12: float,
        alpha21: float,
    ):
        self.species = [species1, species2]
        self.alpha12 = alpha12  # Effect of sp2 on sp1
        self.alpha21 = alpha21  # Effect of sp1 on sp2

    def step(self, dt: float = 0.1):
        """Step competition model."""
        N1, N2 = self.species[0].N, self.species[1].N
        K1, K2 = self.species[0].K, self.species[1].K

        r1 = self.species[0].r - self.species[0].m
        r2 = self.species[1].r - self.species[1].m

        # Logistic with competition
        dN1 = r1 * N1 * (1 - (N1 + self.alpha12 * N2) / K1) * dt
        dN2 = r2 * N2 * (1 - (N2 + self.alpha21 * N1) / K2) * dt

        self.species[0].N = max(0, N1 + dN1)
        self.species[1].N = max(0, N2 + dN2)


class AgeStructuredPopulation:
    """
    Leslie matrix population model.
    """

    def __init__(self, fertility_rates: List[float], survival_rates: List[float]):
        self.ages = len(fertility_rates)

        # Fertility by age class
        self.F = np.array(fertility_rates)

        # Survival to next age class
        self.S = np.array(survival_rates)

        # Initial age distribution
        self.population = np.ones(self.ages)

    def step(self):
        """Step using Leslie matrix."""
        # Leslie matrix
        L = np.zeros((self.ages, self.ages))

        # First row: fertility
        L[0] = self.F

        # Subdiagonal: survival
        for i in range(1, self.ages):
            L[i, i-1] = self.S[i-1]

        # Matrix multiplication
        self.population = L @ self.population


class StochasticPopulation:
    """
    Stochastic population model (demographic stochasticity).
    """

    def __init__(
        self,
        initial_size: int,
        birth_prob: float,
        death_prob: float,
    ):
        self.N = initial_size
        self.birth_prob = birth_prob
        self.death_prob = death_prob
        self.history = [initial_size]

    def step(self):
        """
        Discrete-time step with binomial sampling.
        """
        # Each individual births with probability b
        births = np.random.binomial(self.N, self.birth_prob)

        # Each individual dies with probability d
        deaths = np.random.binomial(self.N, self.death_prob)

        self.N = max(0, self.N + births - deaths)
        self.history.append(self.N)

    def extinction_risk(self, n_simulations: int = 1000, max_time: int = 100) -> float:
        """
        Compute probability of extinction.

        Returns:
            Fraction of simulations that go extinct
        """
        extinct_count = 0

        for _ in range(n_simulations):
            N = self.N

            for _ in range(max_time):
                births = np.random.binomial(N, self.birth_prob)
                deaths = np.random.binomial(N, self.death_prob)
                N = max(0, N + births - deaths)

                if N == 0:
                    break

            if N == 0:
                extinct_count += 1

        return extinct_count / n_simulations


def run_population_simulation():
    """
    Run population dynamics simulations.
    """
    print(f"\n{'='*60}")
    print(f"Population Dynamics Simulation")
    print(f"{'='*60}\n")

    # 1. Logistic growth
    print("1. Logistic Growth")
    pop = Population('deer', initial_size=10, birth_rate=0.5,
                    death_rate=0.2, carrying_capacity=1000,
                    model=GrowthModel.LOGISTIC)

    for year in range(20):
        pop.step(1)
        if year % 5 == 0:
            print(f"  Year {year}: N = {pop.N:.0f}")

    # 2. Predator-prey
    print("\n2. Predator-Prey (Lotka-Volterra)")
    pp = PredatorPrey(
        'rabbit', 'fox',
        prey_N0=100, predator_N0=10,
        prey_birth=1.0, prey_death=0.0,
        predator_birth=0.1, predator_death=0.05,
        attack_rate=0.01, conversion_efficiency=0.1
    )

    for t in range(200):
        pp.step(0.1)
        if t % 50 == 0:
            print(f"  t={t}: prey={pp.prey.N:.0f}, predator={pp.predator.N:.0f}")

    # 3. Competition
    print("\n3. Competition")
    sp1 = Population('sp1', 50, 0.5, 0.1, 500)
    sp2 = Population('sp2', 50, 0.5, 0.1, 500)
    comp = Competition(sp1, sp2, alpha12=0.5, alpha21=0.8)

    for gen in range(50):
        comp.step(1)
        if gen % 10 == 0:
            print(f"  Gen {gen}: sp1={sp1.N:.0f}, sp2={sp2.N:.0f}")

    # 4. Stochastic extinction
    print("\n4. Stochastic Extinction Risk")
    stoch = StochasticPopulation(50, birth_prob=0.3, death_prob=0.25)
    risk = stoch.extinction_risk(1000)
    print(f"  Extinction probability (1000 sims): {risk*100:.1f}%")

    print(f"\n{'='*60}")


__all__ = [
    'GrowthModel',
    'Population', 'PredatorPrey', 'Competition',
    'AgeStructuredPopulation', 'StochasticPopulation',
    'run_population_simulation'
]
