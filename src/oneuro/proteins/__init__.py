"""
oNeuro Proteins Module

Submodules:
- protein_folding: AlphaFold-style prediction, Rosetta energy
- molecular_dynamics: AMBER force fields, Verlet integration
- docking: AutoDock Vina-style ligand docking
- sequence_analysis: Alignment, secondary structure, properties

Usage:
    from oneuro.proteins import predict_folding, run_md_simulation
    from oneuro.proteins.sequence_analysis import align_sequences
"""

from .protein_folding import (
    AminoAcid, AA_PROPERTIES,
    Residue, Protein, AlphaFoldAttention, RosettaEnergy,
    predict_folding, fold_with_md
)

from .molecular_dynamics import (
    CONSTANTS,
    Atom, Molecule, ForceField, MolecularDynamics,
    create_alanine_dipeptide, run_md_simulation
)

from .docking import (
    Ligand, Receptor, GridMap,
    VinaScoringFunction, MonteCarloDocking,
    create_sample_ligand, create_sample_receptor, run_docking
)

from .sequence_analysis import (
    BLOSUM62, HYDROPHOBICITY,
    Alignment, SequenceAlignment, SecondaryStructure, SequenceProperties,
    align_sequences, analyze_sequence
)

__all__ = [
    # protein_folding
    'AminoAcid', 'AA_PROPERTIES',
    'Residue', 'Protein', 'AlphaFoldAttention', 'RosettaEnergy',
    'predict_folding', 'fold_with_md',

    # molecular_dynamics
    'CONSTANTS',
    'Atom', 'Molecule', 'ForceField', 'MolecularDynamics',
    'create_alanine_dipeptide', 'run_md_simulation',

    # docking
    'Ligand', 'Receptor', 'GridMap',
    'VinaScoringFunction', 'MonteCarloDocking',
    'create_sample_ligand', 'create_sample_receptor', 'run_docking',

    # sequence_analysis
    'BLOSUM62', 'HYDROPHOBICITY',
    'Alignment', 'SequenceAlignment', 'SecondaryStructure', 'SequenceProperties',
    'align_sequences', 'analyze_sequence',
]
