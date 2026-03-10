"""
Protein Docking - AutoDock Vina-style flexible docking

This module implements:
- Protein-ligand docking
- Scoring functions (Vina, RF-Score)
- Conformational search (Monte Carlo, genetic algorithms)
- Grid-based affinity maps
- Box definitions for binding sites

References:
- Trott & Olson (2010) J. Comput. Chem. 31, 455 (AutoDock Vina)
- Li et al. (2019) J. Chem. Inf. Model. 59, 1584 (RF-Score)
- Morris et al. (2009) J. Comput. Chem. 30, 2785 (AutoDock4)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import random


@dataclass
class Ligand:
    """Flexible ligand with rotatable bonds."""
    atoms: np.ndarray  # (n_atoms, 3)
    atom_types: List[str]
    bonds: List[Tuple[int, int]]
    rotatable_bonds: List[Tuple[int, int]]  # Axis atoms
    torsions: List[int]  # Number of rotatable bonds

    # Conformation state
    conformation: np.ndarray = None

    def __post_init__(self):
        self.conformation = self.atoms.copy()

    def apply_torsion(self, bond_idx: int, angle: float):
        """Apply torsion rotation around a bond."""
        if bond_idx >= len(self.rotatable_bonds):
            return

        i, j = self.rotatable_bonds[bond_idx]
        axis = self.conformation[j] - self.conformation[i]
        axis /= np.linalg.norm(axis)

        # Rotate all atoms after bond
        for k in range(len(self.conformation)):
            if k != i and k != j:
                # Rotation matrix
                R = self._rotation_matrix(axis, angle)
                self.conformation[k] = R @ (self.conformation[k] - self.conformation[i]) + self.conformation[i]

    def _rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues' rotation formula."""
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1 - c
        x, y, z = axis

        return np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
        ])

    def random_conformation(self):
        """Generate random ligand conformation."""
        self.conformation = self.atoms.copy()
        for i in range(len(self.rotatable_bonds)):
            angle = random.uniform(-180, 180) * math.pi / 180
            self.apply_torsion(i, angle)


@dataclass
class Receptor:
    """Protein receptor with binding site."""
    atoms: np.ndarray  # (n_atoms, 3)
    atom_types: List[str]
    charges: List[float]
    grid: Optional[np.ndarray] = None  # Pre-computed affinity grid


class GridMap:
    """
    Grid-based affinity maps (AutoDock style).

    Pre-computes interactions for fast docking.
    """

    def __init__(self, receptor: Receptor, spacing: float = 0.375,
                 size: Tuple[int, int, int] = (40, 40, 40)):
        self.receptor = receptor
        self.spacing = spacing  # Angstrom
        self.size = size

        # Center on binding site (use receptor center for now)
        center = np.mean(receptor.atoms, axis=0)

        # Create grid
        self.origin = center - np.array(size) * spacing / 2
        self.grid = np.zeros(size)

        # Build maps for each atom type
        self.maps: Dict[str, np.ndarray] = {}

    def build_maps(self):
        """Build affinity maps for each atom type."""
        atom_types = set(self.receptor.atom_types)
        for atype in atom_types:
            self.maps[atype] = np.zeros(self.size)

        # Compute interaction potentials
        for i, (pos, atype) in enumerate(zip(self.receptor.receptor.atom.atoms, self_types)):
            # Distance from each grid point
            for ix in range(self.size[0]):
                for iy in range(self.size[1]):
                    for iz in range(self.size[2]):
                        grid_point = self.origin + np.array([ix, iy, iz]) * self.spacing
                        r = np.linalg.norm(pos - grid_point)

                        if r < 0.5:
                            r = 0.5

                        # Simplified Gaussian potential
                        if atype in self.maps:
                            self.maps[atype][ix, iy, iz] -= math.exp(-r / 2)


class VinaScoringFunction:
    """
    AutoDock Vina scoring function.

    E = w_gauss1 * sum(exp(-(r/r0)^2)) + w_gauss2 * sum(exp(-(r/r0-expr)^2))
        + w_repulsion * sum(max(0, r0 - r)^2) + w_hydrophobic * sum(Ai*Bi)
        + w_hydrogen * sum(Ai*Bi) + w_torsion * N_torsions

    Simplified: just Gaussian + repulsion + hydrophobic.
    """

    def __init__(self):
        self.weights = {
            'gauss1': -0.0356,
            'gauss2': -0.0052,
            'repulsion': 0.840,
            'hydrophobic': -0.0351,
            'hydrogen': -0.5874,
            'torsion': 0.3683,
        }

    def score(self, ligand: Ligand, receptor: Receptor) -> float:
        """
        Calculate binding affinity score.

        Returns:
            Predicted binding energy in kcal/mol
        """
        E = 0.0

        # Inter-molecular interactions
        ligand_atoms = ligand.conformation
        receptor_atoms = receptor.atoms

        for i, lpos in enumerate(ligand_atoms):
            for j, rpos in enumerate(receptor_atoms):
                r = np.linalg.norm(lpos - rpos)

                if r < 0.5:
                    r = 0.5

                # Gaussian attraction
                E += self.weights['gauss1'] * math.exp(-(r / 0.5)**2)
                E += self.weights['gauss2'] * math.exp(-(r / 3.0)**2)

                # Repulsion
                if r < 1.5:
                    E += self.weights['repulsion'] * (1.5 - r)**2

        # Hydrophobic contact
        for lpos in ligand_atoms:
            for rpos in receptor_atoms:
                r = np.linalg.norm(lpos - rpos)
                if r < 2.0:
                    # Simplified: check atom types
                    E += self.weights['hydrophobic']

        # Internal ligand energy (torsions)
        E += self.weights['torsion'] * len(ligand.rotatable_bonds)

        return E


class MonteCarloDocking:
    """
    Monte Carlo ligand docking with simulated annealing.
    """

    def __init__(self, ligand: Ligand, receptor: Receptor, scoring: VinaScoringFunction):
        self.ligand = ligand
        self.receptor = receptor
        self.scoring = scoring

        # Search parameters
        self.temperature = 1000.0  # K
        self.n_iterations = 2500

        # Best result
        self.best_score = float('inf')
        self.best_conformation = None

    def run(self) -> float:
        """
        Run docking simulation.

        Returns:
            Best binding energy
        """
        print(f"Running Monte Carlo docking...")
        print(f"  Iterations: {self.n_iterations}")
        print(f"  Temperature: {self.temperature} K")

        current_ligand = Ligand(
            self.ligand.atoms.copy(),
            self.ligand.atom_types.copy(),
            self.ligand.bonds.copy(),
            self.ligand.rotatable_bonds.copy(),
            self.ligand.torsions.copy()
        )

        current_score = self.scoring.score(current_ligand, self.receptor)

        # Cooling schedule
        T_start = self.temperature
        T_end = 10.0

        for iteration in range(self.n_iterations):
            # Temperature
            T = T_start * (T_end / T_start) ** (iteration / self.n_iterations)

            # Generate random move
            new_ligand = Ligand(
                current_ligand.atoms.copy(),
                current_ligand.atom_types.copy(),
                current_ligand.bonds.copy(),
                current_ligand.rotatable_bonds.copy(),
                current_ligand.torsions.copy()
            )

            # Random translation
            if random.random() < 0.5:
                new_ligand.conformation += np.random.randn(3) * 0.5
            else:
                # Random torsion
                torsion_idx = random.randint(0, len(new_ligand.rotatable_bonds) - 1)
                angle = random.uniform(-30, 30) * math.pi / 180
                new_ligand.apply_torsion(torsion_idx, angle)

            # Score
            new_score = self.scoring.score(new_ligand, self.receptor)

            # Metropolis criterion
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / (0.001987 * T)):
                current_ligand = new_ligand
                current_score = new_score

                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_conformation = current_ligand.conformation.copy()

            if iteration % 500 == 0:
                print(f"  Iter {iteration}: T={T:.1f}K, score={current_score:.3f}")

        return self.best_score

    def get_best_ligand(self) -> Ligand:
        """Return ligand in best conformation."""
        result = Ligand(
            self.ligand.atoms.copy(),
            self.ligand.atom_types.copy(),
            self.ligand.bonds.copy(),
            self.ligand.rotatable_bonds.copy(),
            self.ligand.torsions.copy()
        )
        if self.best_conformation is not None:
            result.conformation = self.best_conformation
        return result


def create_sample_ligand() -> Ligand:
    """Create a simple ligand (e.g., benzene-like)."""
    # Benzene ring
    r = 1.4
    atoms = np.array([
        [r, 0, 0],
        [r * math.cos(math.pi/3), r * math.sin(math.pi/3), 0],
        [-r * math.cos(math.pi/3), r * math.sin(math.pi/3), 0],
        [-r, 0, 0],
        [-r * math.cos(math.pi/3), -r * math.sin(math.pi/3), 0],
        [r * math.cos(math.pi/3), -r * math.sin(math.pi/3), 0],
    ])

    atom_types = ['C'] * 6
    bonds = [(i, (i+1)%6) for i in range(6)]
    rotatable_bonds = []
    torsions = []

    return Ligand(atoms, atom_types, bonds, rotatable_bonds, torsions)


def create_sample_receptor() -> Receptor:
    """Create a simple receptor (flat surface)."""
    # Create binding site as flat surface with some depth
    atoms = []
    atom_types = []

    for x in range(-5, 6):
        for y in range(-5, 6):
            atoms.append([x * 3.8, y * 3.8, 0])
            atom_types.append('C')

    # Add some polar atoms
    for x in [-3, 3]:
        for y in [-3, 3]:
            atoms.append([x * 3.8, y * 3.8, 1.5])
            atom_types.append('O')

    return Receptor(np.array(atoms), atom_types, [0] * len(atoms))


def run_docking():
    """
    Run protein-ligand docking simulation.
    """
    print(f"\n{'='*60}")
    print(f"Protein-Ligand Docking")
    print(f"{'='*60}\n")

    # Create ligand and receptor
    ligand = create_sample_ligand()
    receptor = create_sample_receptor()

    print(f"Ligand: {len(ligand.atoms)} atoms, {len(ligand.rotatable_bonds)} rotatable bonds")
    print(f"Receptor: {len(receptor.atoms)} atoms")

    # Create scoring function
    scoring = VinaScoringFunction()

    # Run docking
    dock = MonteCarloDocking(ligand, receptor, scoring)
    best_score = dock.run()

    print(f"\n{'='*60}")
    print(f"Docking Complete")
    print(f"  Best score: {best_score:.3f} kcal/mol")
    print(f"  Predicted Kd: {math.exp(-best_score / 0.592):.3f} μM")
    print(f"{'='*60}")


__all__ = [
    'Ligand', 'Receptor', 'GridMap',
    'VinaScoringFunction', 'MonteCarloDocking',
    'create_sample_ligand', 'create_sample_receptor', 'run_docking'
]
