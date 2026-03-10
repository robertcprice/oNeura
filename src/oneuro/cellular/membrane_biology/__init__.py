"""
Membrane Biology - Lipid bilayers and transport

This module implements:
- Phospholipid bilayer structure
- Membrane proteins (channels, pumps, receptors)
- Ion channels and gating
- Active transport (pumps)
- Vesicle trafficking
- Membrane fluidity and phase transitions

References:
- Singer & Nicolson (1972) Science 175, 720
- Alberts "Molecular Biology of the Cell"
- Hille "Ion Channels of Excitable Membranes"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math


# Physical constants
CONSTANTS = {
    'kb': 1.380649e-23,  # Boltzmann (J/K)
    'T': 310.0,           # Body temperature (K)
    'eta': 0.001,         # Water viscosity (Pa·s)
    'eps0': 8.85e-12,    # Vacuum permittivity
    'eps_membrane': 2.1,  # Dielectric constant of membrane
    'F': 96485,          # Faraday (C/mol)
}


# Lipid properties
LIPID_PROPERTIES = {
    'DPPC': {'Tm': 41.4, 'chain': 16, 'head': 'PC', 'charge': 0},
    'DPPE': {'Tm': 54, 'chain': 16, 'head': 'PE', 'charge': 0},
    'DOPC': {'Tm': -17, 'chain': 18, 'head': 'PC', 'charge': 0},
    'POPS': {'Tm': 14, 'chain': 16, 'head': 'PS', 'charge': -1},
    'cholesterol': {'Tm': 0, 'Tm_offset': -30, 'rigid': True},
}


class LipidType(Enum):
    """Phospholipid types."""
    PC = 'phosphatidylcholine'
    PE = 'phosphatidylethanolamine'
    PS = 'phosphatidylserine'
    PI = 'phosphatidylinositol'
    PA = 'phosphatidic acid'
    CHOLESTEROL = 'cholesterol'


@dataclass
class Phospholipid:
    """Phospholipid molecule."""
    lipid_type: LipidType
    sn1_chain: int  # Carbon chain length
    sn2_chain: int
    head_charge: int = 0
    Tm: float = 0  # Melting temperature (°C)

    # Physical properties
    cross_section: float = 0.65  # nm²
    diffusion_coeff: float = 1e-11  # m²/s

    def phase_state(self, T: float) -> str:
        """Determine gel or fluid phase."""
        if T < self.Tm:
            return 'gel'
        else:
            return 'fluid'


@dataclass
class Membrane:
    """
    Phospholipid bilayer membrane.
    """

    def __init__(
        self,
        composition: Dict[str, float] = None,
        thickness: float = 4.0,  # nm
        area: float = 1e-12,  # m² (1 μm²)
    ):
        self.composition = composition or {'DPPC': 0.5, 'DPPE': 0.3, 'cholesterol': 0.2}
        self.thickness = thickness
        self.area = area

        # Derived properties
        self.n_lipids = int(area / 0.65e-18)  # ~0.65 nm² per lipid
        self.lipids = self._build_lipids()

        # Fluidity
        self.fluidity = self._compute_fluidity()

    def _build_lipids(self) -> List[Phospholipid]:
        """Build lipid composition."""
        lipids = []
        for name, frac in self.composition.items():
            n = int(self.n_lipids * frac)
            if name in LIPID_PROPERTIES:
                props = LIPID_PROPERTIES[name]
                lipid = Phospholipid(
                    LipidType.PC if 'PC' in name else LipidType.PE if 'PE' in name else LipidType.PS,
                    props['chain'], props['chain'],
                    charge=props.get('charge', 0),
                    Tm=props['Tm']
                )
                lipids.extend([lipid] * n)
        return lipids

    def _compute_fluidity(self) -> float:
        """Compute membrane fluidity."""
        cholesterol = self.composition.get('cholesterol', 0)
        return 1.0 - cholesterol * 0.5  # Cholesterol decreases fluidity

    def lateral_diffusion(self, D0: float = 1e-11) -> float:
        """
        Calculate lateral diffusion coefficient.

        D = D0 * exp(-Ea/kT) * (1 - φ)
        """
        fluidity = self.fluidity
        return D0 * fluidity

    def permeability(self, solute_radius: float) -> float:
        """
        Calculate membrane permeability to solute.

        Uses Flemming model: P = D * K * A / h
        """
        # Partition coefficient (hydrophobic solutes pass easier)
        K = math.exp(-solute_radius * 2)  # Simplified

        # Diffusion through membrane
        D = self.lateral_diffusion()

        # Permeability
        P = D * K * self.n_lipids / (self.thickness * 1e-9)

        return P


class IonChannel:
    """
    Ion channel - passive transport pore.
    """

    def __init__(self, name: str, ion_type: str, conductance: float):
        self.name = name
        self.ion_type = ion_type  # 'Na+', 'K+', 'Ca2+', 'Cl-'
        self.conductance = conductance  # pS

        # Gating
        self.is_open = False
        self.open_probability = 0.0

        # Current
        self.current = 0.0  # pA

    def open_probability_V(self, V: float) -> float:
        """
        Calculate open probability from voltage.

        Simplified: Boltzmann distribution
        """
        if self.ion_type in ['Na+', 'Ca2+']:
            # Depolarization activates (opens)
            V_half = -40  # mV
            k = 5  # slope
        else:
            # Hyperpolarization activates
            V_half = -80
            k = 5

        p_open = 1 / (1 + math.exp((V - V_half) / k))
        self.open_probability = p_open
        return p_open

    def compute_current(self, V: float, concentrations: Dict[str, float]) -> float:
        """
        Compute ionic current using Goldman-Hodgkin-Katz.

        I = g * P_open * (V - E_rev)
        """
        # Reversal potential (GHK)
        if self.ion_type == 'K+':
            E_rev = -90  # mV
        elif self.ion_type == 'Na+':
            E_rev = 60
        elif self.ion_type == 'Ca2+':
            E_rev = 120
        else:
            E_rev = -70

        p = self.open_probability_V(V)
        driving_force = V - E_rev

        self.current = self.conductance * p * driving_force
        return self.current


class IonPump:
    """
    Active ion pump (Na+/K+ ATPase, Ca2+ ATPase).
    """

    def __init__(self, name: str, stoichiometry: Dict[str, int]):
        self.name = name
        self.stoichiometry = stoichiometry  # e.g., {'Na+': 3, 'K+': 2}

        # Kinetics
        self.Vmax = 100  # μM/s
        self.Km = 1000   # μM

        # State
        self.is_active = True

    def transport_rate(self, Na_in: float, Na_out: float,
                      K_in: float, K_out: float) -> float:
        """
        Calculate transport rate.

        Uses Michaelis-Menten kinetics.
        """
        if not self.is_active:
            return 0.0

        # Saturate with substrates
        f_Na = Na_in / (self.Km + Na_in)
        f_K = K_out / (self.Km + K_out)

        rate = self.Vmax * f_Na * f_K
        return rate

    def atp_consumption(self, rate: float) -> float:
        """
        Calculate ATP consumption.

        Na+/K+ ATPase: 1 ATP per cycle
        """
        return rate  # 1:1 stoichiometry


class VesicleTransport:
    """
    Vesicle-mediated membrane trafficking.
    """

    def __init__(self):
        # Vesicle types
        self.vesicle_types = {
            'clathrin': {'diameter': 50, 'cargo': 'receptor'},
            'caveolin': {'diameter': 70, 'cargo': 'lipid_raft'},
            'secretory': {'diameter': 100, 'cargo': 'secreted'},
            'autophagosome': {'diameter': 500, 'cargo': 'organelle'},
        }

    def vesicle_formation(self, vtype: str) -> Dict:
        """
        Simulate vesicle formation.
        """
        if vtype not in self.vesicle_types:
            return {'error': 'Unknown vesicle type'}

        info = self.vesicle_types[vtype]

        # Clathrin coat assembly
        if vtype == 'clathrin':
            n_clathrin = 36  # triskelions
            # Takes ~1 minute
            assembly_time = 60  # seconds

        return {
            'type': vtype,
            'diameter': info['diameter'],
            'cargo': info['cargo'],
            'coat_proteins': n_clathrin if vtype == 'clathrin' else 0,
            'assembly_time': assembly_time if vtype == 'clathrin' else 30,
        }

    def vesicle_fusion(self, vesicle_size: float, target_membrane: str) -> float:
        """
        Simulate SNARE-mediated fusion.

        Returns fusion time.
        """
        # Larger vesicles fuse slower
        fusion_time = 0.1 + vesicle_size / 500  # seconds

        # Energy barrier
        delta_G = 20 * math.exp(-vesicle_size / 100)  # kT

        return fusion_time


class MembraneReceptor:
    """
    Cell surface receptor (GPCR, RTK, etc.).
    """

    def __init__(self, name: str, receptor_type: str):
        self.name = name
        self.receptor_type = receptor_type  # 'GPCR', 'RTK', 'ionotropic'

        # State
        self.is_bound = False
        self.conformation = 'inactive'

        # Signaling
        self.downstream_effectors = []

    def bind_ligand(self, ligand_conc: float, Kd: float = 10) -> bool:
        """
        Ligand binding using Hill equation.

        Returns: bound = [L]^n / (Kd + [L]^n)
        """
        # Hill coefficient
        n = 2 if self.receptor_type == 'GPCR' else 1

        occupancy = ligand_conc**n / (Kd + ligand_conc**n)
        self.is_bound = occupancy > 0.5

        return self.is_bound

    def activate(self) -> Dict:
        """
        Receptor activation and downstream signaling.
        """
        if not self.is_bound:
            return {'status': 'inactive'}

        # Conformational change
        self.conformation = 'active'

        # GPCR: activate G protein
        if self.receptor_type == 'GPCR':
            gtp_exchange = 1  # molec/s
            return {
                'status': 'active',
                'g_protein_activated': gtp_exchange,
                'second_messengers': ['cAMP', 'IP3', 'DAG']
            }

        # RTK: dimerize and autophosphorylate
        elif self.receptor_type == 'RTK':
            return {
                'status': 'active',
                'dimerized': True,
                'phosphorylation_sites': 10,
                'downstream': ['RAS', 'MAPK', 'PI3K']
            }

        return {'status': 'active'}


def simulate_membrane_transport():
    """
    Simulate membrane transport processes.
    """
    print(f"\n{'='*60}")
    print(f"Membrane Transport Simulation")
    print(f"{'='*60}\n")

    # Create membrane
    membrane = Membrane({'DPPC': 0.4, 'DPPE': 0.3, 'cholesterol': 0.3})
    print(f"Membrane: {membrane.n_lipids} lipids, "
          f"fluidity={membrane.fluidity:.2f}")

    # Ion channel
    channel = IonChannel('NaV', 'Na+', 20)
    V = -70  # mV
    conc = {'Na+': 10}  # internal
    I = channel.compute_current(V, conc)
    print(f"\nIon Channel ({channel.name}):")
    print(f"  V = {V} mV, p_open = {channel.open_probability:.3f}")
    print(f"  Current = {channel.current:.2f} pA")

    # Ion pump
    pump = IonPump('NaK', {'Na+': 3, 'K+': 2})
    rate = pump.transport_rate(10, 140, 140, 5)
    atp = pump.atp_consumption(rate)
    print(f"\nNa+/K+ Pump:")
    print(f"  Transport rate: {rate:.2f} cycles/s")
    print(f"  ATP consumption: {atp:.2f} ATP/s")

    # Receptor
    receptor = MembraneReceptor('β-adrenergic', 'GPCR')
    bound = receptor.bind_ligand(100)
    signaling = receptor.activate()
    print(f"\nMembrane Receptor:")
    print(f"  Ligand bound: {bound}")
    print(f"  Status: {signaling['status']}")

    print(f"\n{'='*60}")


__all__ = [
    'CONSTANTS', 'LIPID_PROPERTIES',
    'LipidType', 'Phospholipid', 'Membrane',
    'IonChannel', 'IonPump', 'VesicleTransport', 'MembraneReceptor',
    'simulate_membrane_transport'
]
