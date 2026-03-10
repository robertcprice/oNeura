"""
Drug-Locomotion - EMERGENT from pharmacology

No hardcoded behaviors. Drugs affect:
1. Ion channel properties (GABA, dopamine receptors)
2. Neural firing rates
3. Motor output
4. Locomotion emerges from motor activity

This mirrors real biology - drugs don't "set speed to 0.3",
they modulate receptors which affects neural circuits which affects behavior.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class NeuralCircuit:
    """
    Simple neural circuit for locomotion control.

    Drugs don't set behavior - they modulate receptor activity
    which changes neural firing which changes motor output.
    """

    # Receptor sensitivities (modulated by drugs)
    gaba_sensitivity: float = 1.0   # GABA-A receptor
    dopamine_sensitivity: float = 1.0  # D1/D2 receptors
    serotonin_sensitivity: float = 1.0  # 5-HT receptors

    # Neural state
    motor_activity: float = 0.5  # 0-1, motor neuron firing rate
    turning_bias: float = 0.0    # -1 to 1

    def step(self, dt: float = 0.1) -> Dict[str, float]:
        """
        One neural step.

        Locomotion emerges from:
        - Motor activity (product of receptor sensitivities)
        - Random fluctuations
        - Turning bias
        """
        # Motor activity emerges from receptor balance
        # GABA = inhibitory (reduces activity)
        # Dopamine = excitatory (increases activity)
        excitatory = self.dopamine_sensitivity * 0.6
        inhibitory = self.gaba_sensitivity * 0.4

        target_activity = excitatory - inhibitory + 0.3

        # Neural dynamics
        self.motor_activity += (target_activity - self.motor_activity) * dt * 2
        self.motor_activity = np.clip(self.motor_activity, 0, 1)

        # Turning bias
        self.turning_bias += np.random.uniform(-0.1, 0.1)
        self.turning_bias *= 0.95  # Decay

        return {
            "speed": self.motor_activity,
            "turning": self.turning_bias,
        }


class DrugPharmacology:
    """
    Drugs modulate receptor sensitivities, not behavior.

    This is emergent - we don't say "diazepam reduces speed".
    We say "diazepam enhances GABA which reduces motor activity
    which reduces speed".
    """

    @staticmethod
    def apply_drug(circuit: NeuralCircuit, drug: str, dose: float = 1.0):
        """
        Apply drug by modulating receptor sensitivities.

        NO hardcoded speed/behavior - only receptor modulation.
        Behavior emerges from neural dynamics.

        Args:
            circuit: Neural circuit to modulate
            drug: Drug name
            dose: Dose (affects magnitude)
        """
        dose_factor = dose / (10 + dose)  # Hill-like saturation

        if drug == "diazepam":
            # Benzodiazepine: enhances GABA-A
            # GABA is inhibitory, so enhanced GABA = reduced activity
            circuit.gaba_sensitivity += 1.5 * dose_factor

        elif drug == "alprazolam":
            # Similar to diazepam but stronger
            circuit.gaba_sensitivity += 2.0 * dose_factor

        elif drug == "amphetamine":
            # Amphetamine: releases dopamine, inhibits reuptake
            # More dopamine = more excitation = more activity
            circuit.dopamine_sensitivity += 2.0 * dose_factor

        elif drug == "caffeine":
            # Caffeine: adenosine antagonist (wakefulness)
            # Adenosine is inhibitory, so blocking it = disinhibition
            circuit.dopamine_sensitivity += 0.8 * dose_factor

        elif drug == "fluoxetine":
            # SSRI: increases serotonin
            # Complex effects on locomotion
            circuit.serotonin_sensitivity += 1.0 * dose_factor

        elif drug == "ketamine":
            # NMDA antagonist: dissociative
            # Reduces motor activity
            circuit.dopamine_sensitivity *= (1 - 0.8 * dose_factor)

        # No "control" case - just leave sensitivities at baseline

        # Clamp sensitivities
        circuit.gaba_sensitivity = np.clip(circuit.gaba_sensitivity, 0.1, 5.0)
        circuit.dopamine_sensitivity = np.clip(circuit.dopamine_sensitivity, 0.1, 5.0)
        circuit.serotonin_sensitivity = np.clip(circuit.serotonin_sensitivity, 0.1, 5.0)


class EmergentDrugExperiment:
    """
    Run experiments where drug effects EMERGE from pharmacology.
    """

    def __init__(self):
        self.results: List[Dict] = []

    def run(
        self,
        drug: str,
        dose: float = 10.0,
        duration: float = 60.0,
    ) -> Dict:
        """
        Run experiment.

        Behavior emerges from drug -> receptor -> neural -> motor -> locomotion
        """
        circuit = NeuralCircuit()
        DrugPharmacology.apply_drug(circuit, drug, dose)

        speeds = []
        turnings = []

        for _ in range(int(duration)):
            output = circuit.step(dt=0.1)
            speeds.append(output["speed"])
            turnings.append(abs(output["turning"]))

        avg_speed = np.mean(speeds)
        avg_turning = np.mean(turnings)

        # Compare to baseline (no drug)
        baseline_circuit = NeuralCircuit()
        baseline_output = baseline_circuit.step(dt=0.1)
        baseline_speed = baseline_output["speed"]

        result = {
            "drug": drug,
            "dose": dose,
            "avg_speed": avg_speed,
            "speed_change_pct": ((avg_speed - baseline_speed) / baseline_speed) * 100,
            "avg_turning": avg_turning,
        }

        self.results.append(result)
        return result


def test_emergent_drugs():
    """Test that drug effects emerge from pharmacology."""
    exp = EmergentDrugExperiment()

    print("Testing emergent drug effects...")
    print("="*60)

    drugs = ["control", "diazepam", "amphetamine", "caffeine", "ketamine"]

    for drug in drugs:
        result = exp.run(drug, dose=10.0, duration=30.0)
        print(f"\n{drug.upper()}:")
        print(f"  Speed: {result['avg_speed']:.3f}")
        print(f"  Change: {result['speed_change_pct']:+.1f}%")

    # Check emergent behaviors
    diaz = [r for r in exp.results if r['drug'] == 'diazepam'][0]
    amph = [r for r in exp.results if r['drug'] == 'amphetamine'][0]

    print("\n" + "="*60)
    print("Emergent behaviors:")
    print(f"  Diazepam reduces locomotion: {diaz['avg_speed'] < 0.5}")
    print(f"  Amphetamine increases locomotion: {amph['avg_speed'] > 0.5}")

    return exp.results


if __name__ == "__main__":
    test_emergent_drugs()
