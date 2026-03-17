"""
oNeura Quantum Module

Submodules:
- quantum_mechanics: Wavefunctions, operators, Schrödinger equation
- electronic_structure: Hartree-Fock, CI, MP2, coupled cluster
- density_functional: DFT, Kohn-Sham, exchange-correlation functionals

Usage:
    from oneuro.quantum import run_hf_calculation, run_dft_calculation
    from oneuro.quantum.quantum_mechanics import QuantumState, Hamiltonian
"""

from .quantum_mechanics import (
    CONSTANTS, AU,
    QuantumState, QuantumOperator, Hamiltonian, TimeEvolution,
    hydrogen_atom, calculate_energy, variational_energy
)

from .electronic_structure import (
    run_hf_calculation, create_water_molecule,
    HartreeFock, MoellerPlessetMP2, CoupledCluster
)

from .density_functional import (
    run_dft_calculation, KohnSham, PlaneWaveDFT,
    ExchangeFunctionals, CorrelationFunctionals
)

__all__ = [
    # quantum_mechanics
    'CONSTANTS', 'AU',
    'QuantumState', 'QuantumOperator', 'Hamiltonian', 'TimeEvolution',
    'hydrogen_atom', 'calculate_energy', 'variational_energy',

    # electronic_structure
    'run_hf_calculation', 'create_water_molecule',
    'HartreeFock', 'MoellerPlessetMP2', 'CoupledCluster',

    # density_functional
    'run_dft_calculation', 'KohnSham', 'PlaneWaveDFT',
    'ExchangeFunctionals', 'CorrelationFunctionals',
]
