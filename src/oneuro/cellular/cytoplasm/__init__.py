"""
Cytoplasm - Cellular biochemistry and metabolism

This module implements:
- Glycolysis pathway
- Gluconeogenesis
- Pentose phosphate pathway
- Fatty acid synthesis/oxidation
- Urea cycle
- Amino acid metabolism
- Metabolite concentrations

References:
- Berg "Biochemistry" (Stryer)
- Voet "Biochemistry"
- Alberts "Molecular Biology of the Cell"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math


# Physical constants
CONSTANTS = {
    'R': 8.314,  # J/(mol·K)
    'T': 310.0,   # Body temperature (K)
    'F': 96485,   # Faraday (C/mol)
    'NA': 6.022e23,
}


# Metabolite concentrations (typical cytosolic, μM)
METABOLITES = {
    'glucose': 5000,
    'g6p': 200,    # Glucose-6-phosphate
    'f6p': 100,    # Fructose-6-phosphate
    'f16bp': 30,   # Fructose-1,6-bisphosphate
    'dhap': 100,   # Dihydroxyacetone phosphate
    'g3p': 50,     # Glyceraldehyde-3-phosphate
    'bpg': 5,      # 1,3-Bisphosphoglycerate
    'pg3': 100,    # 3-Phosphoglycerate
    'pg2': 50,     # 2-Phosphoglycerate
    'pep': 50,     # Phosphoenolpyruvate
    'pyr': 100,    # Pyruvate
    'lac': 1000,   # Lactate
    'atp': 2000,   # ATP
    'adp': 200,    # ADP
    'amp': 50,     # AMP
    'nadh': 100,   # NADH
    'nad': 1000,   # NAD+
    'nadph': 200,  # NADPH
    'nadp': 50,    # NADP+
}


@dataclass
class Metabolite:
    """Metabolite with concentration."""
    name: str
    concentration: float  # μM
    charge: int = 0


class MetabolicPathway:
    """Base class for metabolic pathways."""

    def __init__(self, metabolites: Dict[str, Metabolite]):
        self.metabolites = metabolites

    def equilibrium_constant(self, delta_G0: float) -> float:
        """
        Calculate equilibrium constant from standard free energy.

        K_eq = exp(-ΔG°/RT)
        """
        return math.exp(-delta_G0 / (CONSTANTS['R'] * CONSTANTS['T']))


class Glycolysis(MetabolicPathway):
    """
    Glycolysis (Embden-Meyerhof-Parnas pathway).

    10 steps: glucose → 2 pyruvate + 2 ATP + 2 NADH
    """

    def __init__(self):
        super().__init__({
            name: Metabolite(name, METABOLITES[name])
            for name in METABOLITES if name in METABOLITES
        })

        # Reaction parameters (ΔG° in kJ/mol)
        self.reactions = {
            1: {'name': 'Hexokinase', 'ΔG°': -16.7, 'enzyme': 'HK'},
            2: {'name': 'Phosphoglucose isomerase', 'ΔG°': 1.7, 'enzyme': 'PGI'},
            3: {'name': 'Phosphofructokinase', 'ΔG°': -14.2, 'enzyme': 'PFK-1'},
            4: {'name': 'Aldolase', 'ΔG°': 23.8, 'enzyme': 'ALDO'},
            5: {'name': 'Triose phosphate isomerase', 'ΔG°': 7.5, 'enzyme': 'TPI'},
            6: {'name': 'GAPDH', 'ΔG°': 6.3, 'enzyme': 'GAPDH'},
            7: {'name': 'Phosphoglycerate kinase', 'ΔG°': -18.9, 'enzyme': 'PGK'},
            8: {'name': 'Phosphoglycerate mutase', 'ΔG°': 4.4, 'enzyme': 'PGAM'},
            9: {'name': 'Enolase', 'ΔG°': 1.7, 'enzyme': 'ENO'},
            10: {'name': 'Pyruvate kinase', 'ΔG°': -31.4, 'enzyme': 'PK'},
        }

    def step1_glucose_phosphorylation(self, glucose: float) -> Dict:
        """
        Step 1: Glucose + ATP → G6P + ADP

        Irreversible, regulated.
        """
        # Reaction: glucose + atp → g6p + adp
        atp = self.metabolites['atp'].concentration

        # Michaelis-Menten kinetics
        Vmax = 100  # μM/s
        Km = 100    # μM
        v = Vmax * glucose * atp / (Km**2 + glucose * atp)

        # Products
        g6p = v * 1e-3  # Scale
        adp = v * 1e-3

        return {
            'v': v,
            'g6p': g6p,
            'adp': adp,
            'regulated': True,
        }

    def step3_pfk(self, f6p: float) -> Dict:
        """
        Step 3: F6P + ATP → F1,6BP + ADP

        Key regulatory step.
        """
        atp = self.metabolites['atp'].concentration
        amp = self.metabolites['amp'].concentration
        citrate = 100  # μM (inhibitor)

        # Allosteric regulation
        Vmax = 200  # μM/s
        Km = 50     # μM

        # Activators: AMP; Inhibitors: ATP, citrate
        activation = 1 + amp / (10 + amp)
        inhibition = 1 / (1 + atp / 500 + citrate / 100)

        v = Vmax * f6p / (Km + f6p) * activation * inhibition

        return {
            'v': v,
            'f16bp': v * 1e-3,
            'adp': v * 1e-3,
            'allosteric': True,
        }

    def step6_gapdh(self, g3p: float) -> Dict:
        """
        Step 6: G3P + NAD+ + Pi → 1,3BPG + NADH + H+

        Produces NADH.
        """
        nad = self.metabolites['nad'].concentration
        nadh = self.metabolites['nadh'].concentration

        # Equilibrium
        K_eq = 0.5  # Favor products

        v = 50 * g3p * nad / (1 + nadh / K_eq)

        return {
            'v': v,
            'bpg': v * 1e-3,
            'nadh': v * 1e-3,
        }

    def step10_pyruvate_kinase(self, pep: float) -> Dict:
        """
        Step 10: PEP + ADP → Pyruvate + ATP

        Regulated by feed-forward from F1,6BP.
        """
        adp = self.metabolites['adp'].concentration
        f16bp = self.metabolites['f16bp'].concentration

        # Feed-forward activation by F1,6BP
        activation = 1 + f16bp / (10 + f16bp)

        Vmax = 150
        Km = 50

        v = Vmax * pep * adp / (Km**2 + pep * adp) * activation

        return {
            'v': v,
            'pyr': v * 1e-3,
            'atp': v * 1e-3,
        }

    def run_full_pathway(self, glucose: float) -> Dict:
        """
        Run complete glycolysis.

        Returns:
            Products and energy balance
        """
        # Simulate pathway
        step1 = self.step1_glucose_phosphorylation(glucose)
        step3 = self.step3_pfk(100)  # Simplified
        step6 = self.step6_gapdh(50)
        step10 = self.step10_pyruvate_kinase(50)

        return {
            'input_glucose': glucose,
            'pyruvate': step10['pyr'] * 2,  # 2 per glucose
            'atp_net': 2,  # Net: 2 ATP
            'nadh': step6['nadh'] * 2,  # 2 NADH
            'overall_v': min(step1['v'], step3['v'], step6['v'], step10['v']),
        }


class Gluconeogenesis(Glycolysis):
    """
    Gluconeogenesis - glucose synthesis from pyruvate.

    Bypasses irreversible glycolysis steps.
    """

    def __init__(self):
        super().__init__({})

        self.reactions = {
            'pyruvate_carbamoylase': 'Pyruvate → OAA',
            'pep_carboxykinase': 'OAA → PEP',
            'f16bp_phosphatase': 'F1,6BP → F6P',
            'g6p_phosphatase': 'G6P → Glucose',
        }

    def pyruvate_to_pep(self, pyruvate: float) -> Dict:
        """
        Pyruvate → Phosphoenolpyruvate

        Two steps: pyruvate → OAA (PC) → PEP (PEPCK)
        """
        # Pyruvate carboxylase (requires biotin, CO2)
        atp = 2000
        co2 = 1500  # μM

        # PEPCK
        gtp = 200  # Equivalent to ATP

        return {
            'pyruvate_consumed': pyruvate,
            'pep_produced': pyruvate * 0.9,  # 2 PEP from 2 pyr
            'atp_consumed': pyruvate * 2,  # 2 ATP per pyruvate
            'gtp_consumed': pyruvate,
        }


class PentosePhosphatePathway(Glycolysis):
    """
    Pentose Phosphate Pathway (PPP).

    Generates NADPH and ribose-5-phosphate.
    """

    def __init__(self):
        super().__init__({})

        self.phases = {
            1: 'Oxidative PPP (NADPH generation)',
            2: 'Non-oxidative PPP (sugar interconversion)',
        }

    def oxidative_ppp(self, g6p: float) -> Dict:
        """
        Oxidative branch: G6P → ribulose-5-P + 2 NADPH + CO2
        """
        nadp = 50
        nadph = 200

        # G6PDH (rate-limiting)
        v = 20 * g6p * nadp / (10 + nadp)

        return {
            'g6p': g6p,
            'ribulose5p': v * 1e-3,
            'nadph': v * 2e-3,
            'co2': v * 1e-3,
        }


class FattyAcidMetabolism:
    """
    Fatty acid synthesis and β-oxidation.
    """

    def beta_oxidation(self, fatty_acid: str) -> Dict:
        """
        β-Oxidation in mitochondria.

        Returns:
            Acetyl-CoA, NADH, FADH2, ATP yield
        """
        # Fatty acid carbon length
        carbons = {'palmitate': 16, 'oleate': 18, 'stearate': 18}

        n = carbons.get(fatty_acid.lower(), 16)

        # n/2 acetyl-CoA
        acetyl_coa = n // 2

        # (n/2 - 1) NADH, (n/2 - 1) FADH2
        nadh = n // 2 - 1
        fadh2 = n // 2 - 1

        # ATP equivalents (2.5 per NADH, 1.5 per FADH2)
        atp_equivalent = 2.5 * nadh + 1.5 * fadh2 - 2  # Activation cost

        return {
            'fatty_acid': fatty_acid,
            'carbons': n,
            'acetyl_coa': acetyl_coa,
            'nadh': nadh,
            'fadh2': fadh2,
            'atp': atp_equivalent,
        }

    def fatty_acid_synthesis(self, acetyl_coa: float, nadph: float) -> Dict:
        """
        Fatty acid synthesis (in cytosol).

        Acetyl-CoA → Malonyl-CoA → Fatty acid
        """
        # Palmitate (C16) requires:
        # 8 acetyl-CoA + 7 malonyl-CoA + 14 NADPH
        n_palmitate = min(acetyl_coa // 8, nadph // 14)

        return {
            'acetyl_coa_consumed': n_palmitate * 8,
            'nadph_consumed': n_palmitate * 14,
            'palmitate_produced': n_palmitate,
            'atp_consumed': n_palmitate * 7,
        }


class UreaCycle:
    """
    Urea cycle - ammonia detoxification.
    """

    def __init__(self):
        self.location = 'Liver (mitochondria + cytosol)'

    def run_cycle(self, nh3: float, co2: float) -> Dict:
        """
        Run urea cycle.

        2 NH3 + CO2 + 3 ATP → Urea + 2 ADP + AMP + PPi + 2 H2O
        """
        # Limiting reagent
        urea = min(nh3 / 2, co2, 3 * 2000 / 3)  # ATP limited

        return {
            'nh3': urea * 2,
            'co2': urea,
            'urea': urea,
            'atp_consumed': urea * 3,
            'fumarate': urea,
        }


class MetabolismSimulator:
    """
    Complete cellular metabolism simulator.
    """

    def __init__(self):
        self.glycolysis = Glycolysis()
        self.gluconeogenesis = Gluconeogenesis()
        self.ppp = PentosePhosphatePathway()
        self.fatty_acid = FattyAcidMetabolism()
        self.urea = UreaCycle()

        # Cellular state
        self.atp: float = METABOLITES['atp']
        self.adp: float = METABOLITES['adp']
        self.amp: float = METABOLITES['amp']

    def energy_charge(self) -> float:
        """
        Calculate cellular energy charge.

        EC = ([ATP] + 0.5[ADP]) / ([ATP] + [ADP] + [AMP])
        """
        atp = self.atp
        adp = self.adp
        amp = self.amp

        return (atp + 0.5 * adp) / (atp + adp + amp)

    def run_metabolism(self, glucose: float, state: str = 'fed') -> Dict:
        """
        Run cellular metabolism based on state.

        Args:
            glucose: Glucose input (μM)
            state: 'fed' or 'fasted'
        """
        print(f"\nMetabolism Simulation ({state} state)")
        print(f"  Glucose: {glucose} μM")
        print(f"  Energy charge: {self.energy_charge():.3f}")

        results = {}

        if state == 'fed':
            # Glycolysis active
            glycolysis = self.glycolysis.run_full_pathway(glucose)
            results['glycolysis'] = glycolysis

            # Fatty acid synthesis
            acetyl_coa = glucose * 0.5  # From glycolysis
            fas = self.fatty_acid.fatty_acid_synthesis(acetyl_coa, 200)
            results['fatty_acid_synthesis'] = fas

        else:  # fasting
            # Gluconeogenesis
            # Fatty acid oxidation
            fao = self.fatty_acid.beta_oxidation('palmitate')
            results['beta_oxidation'] = fao

            # Ketogenesis from acetyl-CoA
            results['ketone_bodies'] = {
                'acetyl_coa': fao['acetyl_coa'],
                'ketones': fao['acetyl_coa'] * 0.5,
            }

        return results


def simulate_metabolism(glucose: float = 5000, state: str = 'fed'):
    """
    Run complete metabolism simulation.
    """
    print(f"\n{'='*60}")
    print(f"Cellular Metabolism Simulation")
    print(f"State: {state}")
    print(f"{'='*60}\n")

    sim = MetabolismSimulator()

    results = sim.run_metabolism(glucose, state)

    print("\nResults:")
    for pathway, data in results.items():
        print(f"\n{pathway}:")
        for k, v in data.items():
            print(f"  {k}: {v}")

    print(f"\n{'='*60}")


__all__ = [
    'CONSTANTS', 'METABOLITES', 'Metabolite',
    'Glycolysis', 'Gluconeogenesis', 'PentosePhosphatePathway',
    'FattyAcidMetabolism', 'UreaCycle', 'MetabolismSimulator',
    'simulate_metabolism'
]
