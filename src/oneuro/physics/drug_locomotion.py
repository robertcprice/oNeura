"""
Drug-Locomotion Pharmacology Module

Maps pharmacological agents to Drosophila locomotion parameters.
Tests how drugs affect walking, flight, and social behaviors.

Drug Effects on Locomotion:
- GABA agonists (diazepam, alprazolam): Reduce locomotion, increase sleep
- Dopamine agonists (amphetamine): Increase locomotion, arousal
- Caffeine: Increase locomotion, aggression
- SSRIs (fluoxetine): Modulate social behaviors

Usage:
    from oneuro.physics.drug_locomotion import DrugLocomotionExperiment

    experiment = DrugLocomotionExperiment()
    results = experiment.run(
        drug='diazepam',
        dose=10.0,
        duration=60.0,
    )

    print(f"Locomotion change: {results['locomotion_change']:.1f}%")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class DrugType(Enum):
    """Drug categories for locomotion effects."""
    GABA_AGONIST = "gaba_agonist"  # Benzodiazepines
    DOPAMINE_AGONIST = "dopamine_agonist"  # Amphetamine
    SEROTONIN_MODULATOR = "serotonin_modulator"  # Fluoxetine
    STIMULANT = "stimulant"  # Caffeine
    NMDA_ANTAGONIST = "nmda_antagonist"  # Ketamine


# Drug effect mappings to locomotion parameters
# Each drug affects these parameters: (multiplier, baseline)
DRUG_EFFECTS = {
    # GABA agonists - reduce locomotion, increase stillness
    "diazepam": {
        "locomotion_speed": (0.3, 0.7),  # Reduce to 30% of baseline
        "turning_rate": (0.4, 0.6),
        "gait_frequency": (0.5, 0.5),
        "flight_initiation": (0.1, 0.9),
        " aggression": (0.3, 0.7),
        "sleep_probability": (3.0, 0.0),  # 3x more likely to rest
    },
    "alprazolam": {
        "locomotion_speed": (0.25, 0.75),  # Similar to diazepam
        "turning_rate": (0.35, 0.65),
        "gait_frequency": (0.45, 0.55),
        "flight_initiation": (0.05, 0.95),
        "aggression": (0.2, 0.8),
        "sleep_probability": (4.0, 0.0),
    },
    # Dopamine agonists - increase locomotion
    "amphetamine": {
        "locomotion_speed": (1.8, 1.0),  # Increase to 180%
        "turning_rate": (1.5, 1.0),
        "gait_frequency": (1.3, 1.0),
        "flight_initiation": (2.0, 1.0),
        "aggression": (2.5, 1.0),
        "sleep_probability": (0.2, 1.0),
    },
    # Caffeine - moderate stimulant
    "caffeine": {
        "locomotion_speed": (1.4, 1.0),
        "turning_rate": (1.3, 1.0),
        "gait_frequency": (1.2, 1.0),
        "flight_initiation": (1.5, 1.0),
        "aggression": (1.8, 1.0),
        "sleep_probability": (0.3, 1.0),
    },
    # SSRIs - modulate social behaviors
    "fluoxetine": {
        "locomotion_speed": (0.9, 1.1),
        "turning_rate": (0.8, 1.2),
        "gait_frequency": (1.0, 1.0),
        "flight_initiation": (0.9, 1.1),
        "aggression": (0.5, 0.5),  # Reduce aggression
        "courtship": (1.5, 0.5),  # Increase courtship
        "sleep_probability": (1.2, 0.8),
    },
    # NMDA antagonist - dissociative
    "ketamine": {
        "locomotion_speed": (0.2, 0.8),  # Reduce to 20%
        "turning_rate": (0.3, 0.7),
        "gait_frequency": (0.4, 0.6),
        "flight_initiation": (0.05, 0.95),
        "aggression": (0.1, 0.9),
        "sleep_probability": (1.5, 0.5),
    },
    # Control - no drug
    "control": {
        "locomotion_speed": (1.0, 0.0),
        "turning_rate": (1.0, 0.0),
        "gait_frequency": (1.0, 0.0),
        "flight_initiation": (1.0, 0.0),
        "aggression": (1.0, 0.0),
        "sleep_probability": (1.0, 0.0),
    },
}


@dataclass
class LocomotionParams:
    """Locomotion parameters for a fly."""
    speed: float = 1.0
    turning_rate: float = 1.0
    gait_frequency: float = 1.0
    flight_initiation: float = 1.0
    aggression: float = 1.0
    courtship: float = 1.0
    sleep_probability: float = 0.0


@dataclass
class DrugEffect:
    """Observed effect of a drug."""
    drug: str
    dose_mg_kg: float  # mg per kg body weight
    duration_seconds: float

    # Parameter changes (multiplier relative to baseline)
    locomotion_speed_change: float = 0.0
    turning_rate_change: float = 0.0
    gait_frequency_change: float = 0.0
    flight_initiation_change: float = 0.0
    aggression_change: float = 0.0

    # Behavioral observations
    time_moving: float = 0.0
    time_resting: float = 0.0
    distance_traveled: float = 0.0
    num_turns: int = 0
    num_flights: int = 0
    num_social_events: int = 0


class DrugLocomotionSimulator:
    """
    Simulates drug effects on Drosophila locomotion.

    Uses pharmacodynamic models to compute how drugs affect
    neural parameters controlling behavior.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize simulator.

        Args:
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        self.params = LocomotionParams()

    def apply_drug(
        self,
        drug: str,
        dose_mg_kg: float = 1.0,
    ):
        """
        Apply a drug and compute its effects.

        Args:
            drug: Drug name
            dose_mg_kg: Dose in mg per kg
        """
        if drug not in DRUG_EFFECTS:
            raise ValueError(f"Unknown drug: {drug}")

        effects = DRUG_EFFECTS[drug]

        # Compute dose-dependent effect (Hill equation)
        # EC50 ~ 10 mg/kg for most drugs
        dose_response = dose_mg_kg / (10.0 + dose_mg_kg)

        # Apply effects
        self.params.speed = self._compute_param(
            effects.get("locomotion_speed", (1.0, 0.0)),
            dose_response
        )
        self.params.turning_rate = self._compute_param(
            effects.get("turning_rate", (1.0, 0.0)),
            dose_response
        )
        self.params.gait_frequency = self._compute_param(
            effects.get("gait_frequency", (1.0, 0.0)),
            dose_response
        )
        self.params.flight_initiation = self._compute_param(
            effects.get("flight_initiation", (1.0, 0.0)),
            dose_response
        )
        self.params.aggression = self._compute_param(
            effects.get("aggression", (1.0, 0.0)),
            dose_response
        )
        self.params.courtship = self._compute_param(
            effects.get("courtship", (1.0, 0.0)),
            dose_response
        )
        self.params.sleep_probability = self._compute_param(
            effects.get("sleep_probability", (1.0, 0.0)),
            dose_response
        )

    def _compute_param(
        self,
        effect: Tuple[float, float],
        dose_response: float,
    ) -> float:
        """
        Compute parameter value from drug effect.

        Args:
            effect: (max_multiplier, baseline_multiplier)
            dose_response: 0-1

        Returns:
            Parameter value (relative to baseline 1.0)
        """
        max_mult, baseline_mult = effect
        # Start from baseline, scale toward max based on dose
        return baseline_mult + (max_mult - baseline_mult) * dose_response

    def simulate(
        self,
        duration: float = 60.0,
        dt: float = 0.1,
    ) -> DrugEffect:
        """
        Simulate locomotion under drug effects.

        Args:
            duration: Duration in seconds
            dt: Time step

        Returns:
            Drug effect measurements
        """
        num_steps = int(duration / dt)
        position = np.zeros(3)
        direction = np.random.uniform(0, 2 * np.pi)

        time_moving = 0.0
        time_resting = 0.0
        distance = 0.0
        num_turns = 0
        num_flights = 0

        for step in range(num_steps):
            # Check if resting (sleep)
            if np.random.random() < self.params.sleep_probability * dt:
                time_resting += dt
                continue

            # Walking
            speed = self.params.speed * 0.1  # m/s
            distance += speed * dt

            # Turning
            if np.random.random() < self.params.turning_rate * 0.1 * dt:
                direction += np.random.uniform(-0.5, 0.5)
                num_turns += 1

            # Flight initiation (rare)
            if np.random.random() < self.params.flight_initiation * 0.01 * dt:
                num_flights += 1

            time_moving += dt

        # Compute changes relative to baseline
        baseline_speed = 0.1  # m/s
        baseline_moving_time = 0.9 * duration  # 90% of time

        speed_change = (self.params.speed - 1.0) * 100
        turning_change = (self.params.turning_rate - 1.0) * 100
        gait_change = (self.params.gait_frequency - 1.0) * 100
        flight_change = (self.params.flight_initiation - 1.0) * 100
        aggression_change = (self.params.aggression - 1.0) * 100

        return DrugEffect(
            drug="",
            dose_mg_kg=0,
            duration_seconds=duration,
            locomotion_speed_change=speed_change,
            turning_rate_change=turning_change,
            gait_frequency_change=gait_change,
            flight_initiation_change=flight_change,
            aggression_change=aggression_change,
            time_moving=time_moving,
            time_resting=time_resting,
            distance_traveled=distance,
            num_turns=num_turns,
            num_flights=num_flights,
        )

    def get_params(self) -> LocomotionParams:
        """Get current locomotion parameters."""
        return self.params


class DrugLocomotionExperiment:
    """
    Experiment harness for drug-locomotion studies.

    Runs controlled experiments comparing drug effects
    on locomotion parameters.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize experiment.

        Args:
            seed: Random seed
        """
        self.seed = seed
        self.results: List[DrugEffect] = []

    def run(
        self,
        drug: str,
        dose: float = 10.0,
        duration: float = 60.0,
    ) -> DrugEffect:
        """
        Run a single experiment.

        Args:
            drug: Drug name
            dose: Dose in mg/kg
            duration: Duration in seconds

        Returns:
            Drug effect measurements
        """
        sim = DrugLocomotionSimulator(seed=self.seed)
        sim.apply_drug(drug, dose)
        result = sim.simulate(duration)
        result.drug = drug
        result.dose_mg_kg = dose

        self.results.append(result)
        return result

    def run_comparative(
        self,
        drugs: List[str],
        dose: float = 10.0,
        duration: float = 60.0,
    ) -> Dict[str, DrugEffect]:
        """
        Run comparative experiment across drugs.

        Args:
            drugs: List of drug names
            dose: Dose in mg/kg
            duration: Duration in seconds

        Returns:
            Dictionary of drug -> effect
        """
        results = {}
        for drug in drugs:
            result = self.run(drug, dose, duration)
            results[drug] = result
        return results

    def get_summary(self) -> Dict:
        """
        Get summary of all experiments.

        Returns:
            Summary statistics
        """
        if not self.results:
            return {}

        summary = {
            "num_experiments": len(self.results),
            "drugs_tested": list(set(r.drug for r in self.results)),
            "comparisons": {},
        }

        # Find control
        control = None
        for r in self.results:
            if r.drug == "control":
                control = r
                break

        if control:
            for r in self.results:
                if r.drug != "control":
                    summary["comparisons"][r.drug] = {
                        "locomotion_change": r.locomotion_speed_change,
                        "turning_change": r.turning_rate_change,
                        "aggression_change": r.aggression_change,
                        "distance_vs_control": r.distance_traveled - control.distance_traveled,
                    }

        return summary


def create_experiment(seed: Optional[int] = None) -> DrugLocomotionExperiment:
    """
    Factory function to create experiment.

    Args:
        seed: Random seed

    Returns:
        Configured experiment
    """
    return DrugLocomotionExperiment(seed=seed)


# ============================================================================
# Tests
# ============================================================================

def test_simulator_creation():
    """Test simulator can be created."""
    sim = DrugLocomotionSimulator()
    assert sim.params is not None
    print("✓ Simulator creation test passed")


def test_control_simulation():
    """Test control (no drug) simulation."""
    sim = DrugLocomotionSimulator()
    result = sim.simulate(duration=10.0)

    assert result.time_moving > 0
    assert result.distance_traveled > 0
    print("✓ Control simulation test passed")


def test_diazepam_effect():
    """Test diazepam effect."""
    sim = DrugLocomotionSimulator()
    sim.apply_drug("diazepam", dose_mg_kg=10.0)

    # Diazepam should reduce locomotion
    assert sim.params.speed < 1.0
    assert sim.params.sleep_probability > 0

    result = sim.simulate(duration=10.0)
    assert result.locomotion_speed_change < 0

    print("✓ Diazepam effect test passed")


def test_amphetamine_effect():
    """Test amphetamine effect."""
    sim = DrugLocomotionSimulator()
    sim.apply_drug("amphetamine", dose_mg_kg=10.0)

    # Amphetamine should increase locomotion
    assert sim.params.speed > 1.0
    assert sim.params.aggression > 1.0

    result = sim.simulate(duration=10.0)
    assert result.locomotion_speed_change > 0

    print("✓ Amphetamine effect test passed")


def test_comparative_experiment():
    """Test comparative experiment."""
    exp = DrugLocomotionExperiment()

    drugs = ["control", "diazepam", "amphetamine", "caffeine"]
    for drug in drugs:
        exp.run(drug, dose=10.0, duration=5.0)

    summary = exp.get_summary()
    assert summary["num_experiments"] == 4

    # Diazepam should reduce, amphetamine increase
    assert summary["comparisons"]["diazepam"]["locomotion_change"] < 0
    assert summary["comparisons"]["amphetamine"]["locomotion_change"] > 0

    print("✓ Comparative experiment test passed")


if __name__ == "__main__":
    print("Testing Drug-Locomotion Module...")

    test_simulator_creation()
    test_control_simulation()
    test_diazepam_effect()
    test_amphetamine_effect()
    test_comparative_experiment()

    # Print example results
    print("\n" + "="*60)
    print("Example Results: Drug Effects on Locomotion")
    print("="*60)

    exp = DrugLocomotionExperiment()
    drugs = ["control", "diazepam", "amphetamine", "caffeine", "fluoxetine"]
    exp.run_comparative(drugs, dose=10.0, duration=60.0)

    for drug in drugs:
        r = exp.results[-1] if exp.results[-1].drug == drug else None
        if r:
            print(f"\n{drug.upper()}")
            print(f"  Locomotion: {r.locomotion_speed_change:+.1f}%")
            print(f"  Turning: {r.turning_rate_change:+.1f}%")
            print(f"  Aggression: {r.aggression_change:+.1f}%")
            print(f"  Distance: {r.distance_traveled:.2f}m")

    print("\n✓ All tests passed!")
