"""
Cellular Organelles - Subcellular structures

This module implements:
- Mitochondria with respiratory chain
- Endoplasmic reticulum (rough and smooth)
- Golgi apparatus
- Nucleus with DNA/RNA
- Ribosomes and protein synthesis
- Cytoskeleton (actin, microtubules)

References:
- Alberts "Molecular Biology of the Cell"
- Lodish "Molecular Cell Biology"
- Nicholls "Bioenergetics"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
import math


class OrganelleType(Enum):
    """Cell organelle types."""
    MITOCHONDRION = "mitochondrion"
    ER_ROUGH = "er_rough"
    ER_SMOOTH = "er_smooth"
    GOLGI = "golgi"
    NUCLEUS = "nucleus"
    RIBOSOME = "ribosome"
    LYSOSOME = "lysosome"
    PEROXISOME = "peroxisome"
    CYTOSOL = "cytosol"


@dataclass
class Membrane:
    """Phospholipid bilayer membrane."""
    composition: Dict[str, float] = field(default_factory=lambda: {
        'PC': 0.5,   # Phosphatidylcholine
        'PE': 0.25,  # Phosphatidylethanolamine
        'PS': 0.1,    # Phosphatidylserine
        'cholesterol': 0.15,
    })
    fluidity: float = 0.5  # 0 = rigid, 1 = fluid
    thickness: float = 5.0  # nm


@dataclass
class Organelle:
    """Base organelle class."""
    organelle_type: OrganelleType
    position: np.ndarray  # 3D position in nm
    volume: float  # nm^3
    membrane: Optional[Membrane] = None
    proteins: List[str] = field(default_factory=list)


# Physical constants
CONSTANTS = {
    'NA': 6.02214076e23,  # Avogadro
    'kb': 1.380649e-23,   # Boltzmann (J/K)
    'T': 310.0,            # Body temperature (K)
    'eta': 0.001,         # Cytosol viscosity (Pa·s)
}


class Mitochondrion(Organelle):
    """
    Mitochondrion with oxidative phosphorylation.

    Structure:
    - Outer membrane (porous)
    - Intermembrane space
    - Inner membrane (cristae,ETC complexes)
    - Matrix (Krebs cycle)
    """

    def __init__(self, position: np.ndarray):
        super().__init__(
            OrganelleType.MITOCHONDRION,
            position,
            volume=0.5,  # μm³
            membrane=Membrane({'PC': 0.3, 'PE': 0.4, 'cardiolipin': 0.3}, fluidity=0.4)
        )

        # ETC Complexes
        self.complexes = {
            'I': {'name': 'NADH dehydrogenase', 'n_electrons': 2},
            'II': {'name': 'Succinate dehydrogenase', 'n_electrons': 2},
            'III': {'name': 'Cytochrome bc1', 'n_electrons': 2},
            'IV': {'name': 'Cytochrome c oxidase', 'n_electrons': 4},
            'V': {'name': 'ATP synthase', 'atp_per_proton': 3},
        }

        # Membrane potential
        self.delta_psi: float = -150.0  # mV
        self.delta_pH: float = 1.0

        # Proton gradient
        self.proton_gradient: float = 3.5  # pH units equivalent

        # Matrix
        self.nadh: float = 10.0  # μM
        self.atp: float = 2500.0  # μM
        self.adp: float = 100.0  # μM

    def electron_transport_chain(self, nadh_available: float) -> Dict:
        """
        Simulate electron transport chain.

        Returns:
            Dict with ATP production, ROS, etc.
        """
        # Complex I: NADH oxidation
        e_flow = min(nadh_available, self.nad)

        # P/O ratio: ~2.5 ATP per NADH, 1.5 per FADH2
        atp_produced = e_flow * 2.5

        # ROS (reactive oxygen species) - ~2% leak
        ros = e_flow * 0.02

        # Protons pumped
        protons_pumped = e_flow * 10  # ~10 H+ per NADH

        # ATP synthesis
        atp = min(atp_produced, protons_pumped / 3)  # 3 H+ per ATP

        return {
            'electrons_transferred': e_flow,
            'atp_produced': atp,
            'ros_generated': ros,
            'protons_pumped': protons_pumped,
            'membrane_potential': self.delta_psi,
        }

    def krebs_cycle(self, acetyl_coa: float) -> Dict:
        """
        Krebs cycle (citric acid cycle).

        Returns:
            Dict with products
        """
        # Per acetyl-CoA:
        products = {
            'nadh': 3,
            'fadh2': 1,
            'gtp': 1,  # (equivalent to ATP)
            'co2': 2,
            'atp': 1,
        }

        # Scale by input
        return {k: v * acetyl_coa for k, v in products.items()}

    def atp_synthesis(self, protons_imported: float) -> float:
        """
        ATP synthase F1F0 complex.

        Returns:
            ATP produced
        """
        # ~3 protons per ATP
        return protons_imported / 3

    def compute_membrane_potential(self) -> float:
        """
        Compute mitochondrial membrane potential.

        Uses Nernst equation: Δψ = -60 * log([H+]_in / [H+]_out)
        """
        delta_pH = self.delta_pH
        delta_psi = -60 * delta_pH  # mV at 37°C

        # Add contribution from proton gradient
        self.delta_psi = delta_psi

        return self.delta_psi


class Nucleus(Organelle):
    """
    Cell nucleus with DNA and transcription machinery.
    """

    def __init__(self, position: np.ndarray):
        super().__init__(
            OrganelleType.NUCLEUS,
            position,
            volume=5.0,  # μm³
        )

        # Nuclear envelope
        self.envelope = Membrane({
            'PC': 0.4, 'PE': 0.2, 'cholesterol': 0.3, 'Lamins': 0.1
        }, fluidity=0.3)

        # DNA content (human diploid)
        self.dna_length: float = 6.4e9  # base pairs
        self.chromosomes: int = 46

        # Nucleolus
        self.nucleolus_volume: float = 0.1  # μm³

    def transcription(self, gene_length: int, transcription_rate: float = 50) -> Dict:
        """
        Simulate transcription.

        Args:
            gene_length: Base pairs
            transcription_rate: Nucleotides/second

        Returns:
            RNA produced
        """
        time = gene_length / transcription_rate  # seconds
        mrna = 1  # One mRNA per transcription event

        return {
            'gene_length': gene_length,
            'transcription_time': time,
            'mrna_produced': mrna,
        }

    def dna_replication(self) -> Dict:
        """
        Simulate DNA replication (S phase).

        Returns:
            Replication metrics
        """
        # Replication speed ~50 kb/min
        speed = 50e3 / 60  # bp/s
        time = self.dna_length / speed / 2  # Both strands

        return {
            'duration_hours': time / 3600,
            'origin_count': 100,  # ~100 origins in human
            'fork_speed': speed,
        }


class EndoplasmicReticulum(Organelle):
    """
    Endoplasmic Reticulum (ER).

    Rough ER: ribosome-studded, protein synthesis
    Smooth ER: lipid synthesis, detoxification
    """

    def __init__(self, position: np.ndarray, rough: bool = True):
        otype = OrganelleType.ER_ROUGH if rough else OrganelleType.ER_SMOOTH
        super().__init__(
            otype,
            position,
            volume=1.0 if rough else 0.3,
        )

        self.rough = rough
        self.membrane = Membrane({
            'PC': 0.5, 'PE': 0.3, 'cholesterol': 0.2
        }, fluidity=0.6)

        # Surface area
        self.surface_area: float = 10.0 if rough else 5.0  # μm²

    def protein_synthesis(self, n_ribosomes: int) -> Dict:
        """
        Protein synthesis on rough ER.

        Returns:
            Protein synthesis metrics
        """
        # Translation speed ~6 aa/s
        translation_rate = 6  # aa/s

        # Each ribosome produces ~1 protein every few minutes
        proteins_per_minute = n_ribosomes * 0.5

        return {
            'ribosomes': n_ribosomes,
            'proteins_per_minute': proteins_per_minute,
            'translation_rate': translation_rate,
        }

    def lipid_synthesis(self) -> Dict:
        """
        Lipid synthesis in smooth ER.

        Returns:
            Lipid synthesis metrics
        """
        return {
            'phospholipids_per_second': 100,
            'cholesterol_per_second': 10,
            'glycerolipids_per_second': 50,
        }


class GolgiApparatus(Organelle):
    """
    Golgi apparatus (Golgi body).

    Cisternae: flattened membrane sacs
    """

    def __init__(self, position: np.ndarray):
        super().__init__(
            OrganelleType.GOLGI,
            position,
            volume=0.2,
        )

        # Stacks of cisternae
        self.n_cisternae: int = 6

        # Vesicle trafficking
        self.cis_to_med: float = 50  # vesicles/min
        self.med_to_trans: float = 50

        # Glycosylation
        self.n_glycosyltransferases: int = 200

    def process_protein(self, protein: str, modifications: List[str]) -> Dict:
        """
        Process and modify protein.

        Returns:
            Processing results
        """
        return {
            'protein': protein,
            'modifications': modifications,
            'glycosylation_sites': len([m for m in modifications if 'glycan' in m]),
            'processing_time': len(modifications) * 10,  # seconds
        }


class Ribosome(Organelle):
    """
    Ribosome - protein synthesis machinery.

    80S in eukaryotes (40S + 60S)
    """

    def __init__(self, position: np.ndarray):
        super().__init__(
            OrganelleType.RIBOSOME,
            position,
            volume=0.000004,  # μm³ (4 MDa)
        )

        # Subunits
        self.small_subunit = {'size': '40S', 'mrna': 1}
        self.large_subunit = {'size': '60S', 'trna': 3}

        # Translation rates
        self.initiation_rate: float = 0.1  # s⁻¹
        self.elongation_rate: float = 6.0  # aa/s
        self.termination_rate: float = 0.2  # s⁻¹

    def translate(self, mrna_length: int) -> Dict:
        """
        Translate mRNA to protein.

        Returns:
            Translation metrics
        """
        # Initiation
        t_init = 1.0 / self.initiation_rate

        # Elongation
        t_elong = mrna_length / self.elongation_rate

        # Termination
        t_term = 1.0 / self.termination_rate

        total_time = t_init + t_elong + t_term

        return {
            'mrna_length': mrna_length,
            'amino_acids': mrna_length // 3,
            'total_time': total_time,
            'rate': mrna_length / 3 / total_time,
        }


class Cytoskeleton:
    """
    Cell cytoskeleton.

    - Actin filaments (microfilaments)
    - Microtubules
    - Intermediate filaments
    """

    def __init__(self):
        self.actin_monomers: int = 0
        self.tubulin_dimers: int = 0

        # Motor proteins
        self.kinesins: int = 0
        self.dyneins: int = 0
        self.myosins: int = 0

    def actin_polymerization(self, g_actin: float) -> Dict:
        """
        Actin filament assembly.

        Returns:
            Polymerization state
        """
        # Critical concentration ~0.1 μM
        cc = 0.1

        if g_actin > cc:
            rate = k_on * g_actin - k_off
            filaments = rate > 0
        else:
            rate = 0
            filaments = 0

        return {
            'g_actin': g_actin,
            'filaments': filaments,
            'rate': rate,
        }

    def microtubule_dynamic_instability(self, g_tubulin: float) -> Dict:
        """
        Microtubule plus-end dynamics.

        Returns:
            Dynamics state
        """
        # Growth and shrinkage rates
        growth_rate = 1.0  # μm/min
        shrink_rate = 15.0  # μm/min
        catastrophe_freq = 0.05  # per s
        rescue_freq = 0.01

        return {
            'g_tubulin': g_tubulin,
            'growth_rate': growth_rate,
            'shrink_rate': shrink_rate,
            'catastrophe_freq': catastrophe_freq,
            'rescue_freq': rescue_freq,
        }


def simulate_mitochondria(nadh: float = 10, acetyl_coa: float = 1):
    """
    Run mitochondrial simulation.

    Args:
        nadh: NADH concentration (μM)
        acetyl_coa: Acetyl-CoA for Krebs cycle
    """
    print(f"\n{'='*60}")
    print(f"Mitochondrion Simulation")
    print(f"{'='*60}\n")

    # Create mitochondrion
    mito = Mitochondrion(np.array([0, 0, 0]))

    print(f"Mitochondrion at {mito.position}")
    print(f"Membrane potential: {mito.delta_psi:.1f} mV")
    print(f"ATP: {mito.atp:.1f} μM, ADP: {mito.adp:.1f} μM\n")

    # Krebs cycle
    krebs = mito.krebs_cycle(acetyl_coa)
    print("Krebs Cycle Products:")
    for k, v in krebs.items():
        print(f"  {k}: {v:.2f}")

    # ETC
    etc = mito.electron_transport_chain(nadh)
    print("\nElectron Transport Chain:")
    for k, v in etc.items():
        print(f"  {k}: {v:.2f}")

    print(f"\n{'='*60}")


__all__ = [
    'OrganelleType', 'Membrane', 'Organelle',
    'Mitochondrion', 'Nucleus', 'EndoplasmicReticulum',
    'GolgiApparatus', 'Ribosome', 'Cytoskeleton',
    'simulate_mitochondria'
]
