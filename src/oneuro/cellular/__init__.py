"""
oNeura Cellular Module

Submodules:
- organelles: Mitochondria, nucleus, ER, Golgi, ribosomes
- cytoplasm: Metabolism (glycolysis, Krebs, PPP), metabolites
- membrane_biology: Lipid bilayers, ion channels, pumps, receptors
- cell_cycle: G1/S/G2/M phases, cyclins, CDKs, checkpoints

Usage:
    from oneuro.cellular import simulate_mitochondria, simulate_metabolism
    from oneuro.cellular.cell_cycle import simulate_cell_cycle
"""

from .organelles import (
    OrganelleType, Membrane, Organelle,
    Mitochondrion, Nucleus, EndoplasmicReticulum,
    GolgiApparatus, Ribosome, Cytoskeleton,
    simulate_mitochondria
)

from .cytoplasm import (
    CONSTANTS, METABOLITES, Metabolite,
    Glycolysis, Gluconeogenesis, PentosePhosphatePathway,
    FattyAcidMetabolism, UreaCycle, MetabolismSimulator,
    simulate_metabolism
)

from .membrane_biology import (
    CONSTANTS, LIPID_PROPERTIES,
    LipidType, Phospholipid, Membrane,
    IonChannel, IonPump, VesicleTransport, MembraneReceptor,
    simulate_membrane_transport
)

from .cell_cycle import (
    CellCyclePhase, Cyclin, CDK, Checkpoint,
    CellCycle, CellPopulation,
    simulate_cell_cycle
)

__all__ = [
    # organelles
    'OrganelleType', 'Membrane', 'Organelle',
    'Mitochondrion', 'Nucleus', 'EndoplasmicReticulum',
    'GolgiApparatus', 'Ribosome', 'Cytoskeleton',
    'simulate_mitochondria',

    # cytoplasm
    'CONSTANTS', 'METABOLITES', 'Metabolite',
    'Glycolysis', 'Gluconeogenesis', 'PentosePhosphatePathway',
    'FattyAcidMetabolism', 'UreaCycle', 'MetabolismSimulator',
    'simulate_metabolism',

    # membrane_biology
    'CONSTANTS', 'LIPID_PROPERTIES',
    'LipidType', 'Phospholipid', 'Membrane',
    'IonChannel', 'IonPump', 'VesicleTransport', 'MembraneReceptor',
    'simulate_membrane_transport',

    # cell_cycle
    'CellCyclePhase', 'Cyclin', 'CDK', 'Checkpoint',
    'CellCycle', 'CellPopulation',
    'simulate_cell_cycle',
]
