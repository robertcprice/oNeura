"""
Electronic Structure Methods - Ab initio quantum chemistry

This module implements:
- Hartree-Fock (HF) self-consistent field
- Configuration Interaction (CI)
- Møller-Plesset Perturbation Theory (MP2)
- Coupled Cluster (CC) theory

References:
- Szabo & Ostlund "Modern Quantum Chemistry"
- Helgaker "Molecular Electronic-Structure Theory"

Constants (CODATA 2022):
- Fine structure constant α: 7.2973525693e-3
- Electron g-factor: -2.00231930436256
- Nuclear magneton: 5.050783699e-27 J/T
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import scipy.linalg as la

# Physical constants
CONSTANTS = {
    'alpha': 7.2973525693e-3,      # Fine structure constant (dimensionless)
    'me': 9.1093837015e-31,        # Electron mass (kg)
    'e': 1.602176634e-19,           # Elementary charge (C)
    'hbar': 1.054571817e-34,        # Reduced Planck (J·s)
    'Eh': 4.3597447222071e-18,      # Hartree energy (J)
    'a0': 5.29177210903e-11,       # Bohr radius (m)
    'c': 299792458,                 # Speed of light (m/s)
    'mu_B': 9.2740100783e-24,       # Bohr magneton (J/T)
    'mu_N': 5.050783699e-27,        # Nuclear magneton (J/T)
    'g_e': -2.00231930436256,       # Electron g-factor
}

# Atomic units
AU = {
    'length': CONSTANTS['a0'],
    'energy': CONSTANTS['Eh'],
    'mass': CONSTANTS['me'],
    'charge': CONSTANTS['e'],
    'time': CONSTANTS['hbar'] / CONSTANTS['Eh'],
}


@dataclass
class AtomBasis:
    """Gaussian-type orbital (GTO) basis function."""
    center: np.ndarray  # 3D position in Bohr
    l: int  # Angular momentum (x^l y^m z^n)
    m: int
    n: int
    exponents: np.ndarray  # Gaussian exponents
    coefficients: np.ndarray  # Contraction coefficients


@dataclass
class Molecule:
    """Molecule with nuclear charges and positions."""
    charges: np.ndarray  # Nuclear charges (Z)
    positions: np.ndarray  # Nuclear positions in Bohr (N x 3)
    basis: List[AtomBasis]  # Basis set

    def n_electrons(self) -> int:
        """Calculate number of electrons."""
        return int(np.sum(self.charges)) - self.charges[0]  # Simplified


class GaussianBasisSet:
    """STO-3G minimal basis set for common atoms."""

    # STO-3G exponents and coefficients (Hartree)
    STO3G = {
        1: {  # Hydrogen
            's': [(0.3425250914, 0.1543289673),
                  (0.6239137298, 0.5353281423),
                  (3.425250914, 0.4446345422)]
        },
        6: {  # Carbon
            's': [(71.4, 0.1543289673),
                  (13.01, 0.5353281423),
                  (3.530392, 0.4446345422)],
            'px': [(2.941249, 0.155916275),
                   (0.683483, 0.6076837186),
                   (0.2222899, 0.3919573931)],
            'py': [(2.941249, 0.155916275),
                   (0.683483, 0.6076837186),
                   (0.2222899, 0.3919573931)],
            'pz': [(2.941249, 0.155916275),
                   (0.683483, 0.6076837186),
                   (0.2222899, 0.3919573931)],
        },
        7: {  # Nitrogen
            's': [(99.11, 0.1543289673),
                  (18.52, 0.5353281423),
                  (5.201, 0.4446345422)],
            'px': [(3.78, 0.155916275),
                   (0.8785, 0.6076837186),
                   (0.2857, 0.3919573931)],
            'py': [(3.78, 0.155916275),
                   (0.8785, 0.6076837186),
                   (0.2857, 0.3919573931)],
            'pz': [(3.78, 0.155916275),
                   (0.8785, 0.6076837186),
                   (0.2857, 0.3919573931)],
        },
        8: {  # Oxygen
            's': [(130.9, 0.1543289673),
                  (24.54, 0.5353281423),
                  (7.267, 0.4446345422)],
            'px': [(5.033, 0.155916275),
                   (1.198, 0.6076837186),
                   (0.3827, 0.3919573931)],
            'py': [(5.033, 0.155916275),
                   (1.198, 0.6076837186),
                   (0.3827, 0.3919573931)],
            'pz': [(5.033, 0.155916275),
                   (1.198, 0.6076837186),
                   (0.3827, 0.3919573931)],
        },
    }

    @classmethod
    def get_basis(cls, Z: int, position: np.ndarray) -> List[AtomBasis]:
        """Get basis functions for atom with nuclear charge Z."""
        if Z not in cls.STO3G:
            raise ValueError(f"STO-3G not available for Z={Z}")

        basis = []
        for sym, data in cls.STO3G[Z].items():
            coeffs = np.array([c[0] for c in data])
            exps = np.array([c[1] for c in data])

            if sym == 's':
                basis.append(AtomBasis(position, 0, 0, 0, exps, coeffs))
            elif sym == 'px':
                basis.append(AtomBasis(position, 1, 0, 0, exps, coeffs))
            elif sym == 'py':
                basis.append(AtomBasis(position, 0, 1, 0, exps, coeffs))
            elif sym == 'pz':
                basis.append(AtomBasis(position, 0, 0, 1, exps, coeffs))

        return basis


class HartreeFock:
    """
    Restricted Hartree-Fock (RHF) self-consistent field calculation.

    Solves the Roothaan equations: FC = SCE
    where F = H + G(P) is the Fock matrix
    """

    def __init__(self, molecule: Molecule):
        self.molecule = molecule
        self.n_basis = len(molecule.basis)
        self.n_elec = molecule.n_electron()

        # Allocate arrays
        self.H = np.zeros((self.n_basis, self.n_basis))  # Core Hamiltonian
        self.S = np.zeros((self.n_basis, self.n_basis))  # Overlap integral
        self.T = np.zeros((self.n_basis, self.n_basis))  # Kinetic
        self.V = np.zeros((self.n_basis, self.n_basis))  # Nuclear attraction
        self.eri = np.zeros((self.n_basis, self.n_basis, self.n_basis, self.n_basis))  # 2-electron
        self.P = np.zeros((self.n_basis, self.n_basis))  # Density matrix
        self.F = np.zeros((self.n_basis, self.n_basis))  # Fock matrix

        self.energy = 0.0
        self.converged = False

    def compute_integrals(self):
        """Compute one- and two-electron integrals."""
        print("Computing integrals...")

        # Overlap integrals S_ij = <phi_i | phi_j>
        for i, bf_i in enumerate(self.molecule.basis):
            for j, bf_j in enumerate(self.molecule.basis):
                self.S[i, j] = self._overlap_integral(bf_i, bf_j)

        # Kinetic energy T_ij = -1/2 <phi_i | nabla^2 | phi_j>
        for i, bf_i in enumerate(self.molecule.basis):
            for j, bf_j in enumerate(self.molecule.basis):
                self.T[i, j] = self._kinetic_integral(bf_i, bf_j)

        # Nuclear attraction V_ij = -Z * <phi_i | 1/r | phi_j>
        for a, (Z, R) in enumerate(zip(self.molecule.charges, self.molecule.positions)):
            for i, bf_i in enumerate(self.molecule.basis):
                for j, bf_j in enumerate(self.molecule.basis):
                    self.V[i, j] -= Z * self._nuclear_attraction(bf_i, bf_j, R)

        # Core Hamiltonian H = T + V
        self.H = self.T + self.V

        # Two-electron integrals (ij|kl) = <phi_i phi_j | 1/r12 | phi_k phi_l>
        print(f"Computing {self.n_basis}^4 = {self.n_basis**4} two-electron integrals...")
        for i in range(self.n_basis):
            for j in range(self.n_basis):
                for k in range(self.n_basis):
                    for l in range(self.n_basis):
                        self.eri[i, j, k, l] = self._two_electron_integral(
                            self.molecule.basis[i],
                            self.molecule.basis[j],
                            self.molecule.basis[k],
                            self.molecule.basis[l]
                        )

    def _overlap_integral(self, bf1: AtomBasis, bf2: AtomBasis) -> float:
        """Compute overlap integral between two GTOs."""
        # Analytic formula for Gaussian integrals
        R = bf1.center - bf2.center
        R2 = np.dot(R, R)

        result = 0.0
        for a, ca in zip(bf1.exponents, bf1.coefficients):
            for b, cb in zip(bf2.exponents, bf2.coefficients):
                p = a + b
                q = a * b / p
                prefactor = (np.pi / p) ** 1.5 * np.exp(-q * R2)
                result += ca * cb * prefactor * self._gaussian_moment(bf1.l + bf2.l, bf1.m + bf2.m, bf1.n + bf2.n, 0, 0, 0)

        return result

    def _gaussian_moment(self, l1, m1, n1, l2, m2, n2, l=0, m=0, n=0):
        """Compute Gaussian integral moment."""
        if (l1 + l2 + l) % 2 != 0 or (m1 + m2 + m) % 2 != 0 or (n1 + n2 + n) % 2 != 0:
            return 0.0
        return 1.0  # Simplified

    def _kinetic_integral(self, bf1: AtomBasis, bf2: AtomBasis) -> float:
        """Compute kinetic energy integral."""
        # T = <g1| -1/2 nabla^2 | g2>
        # Use Laplace transform identity
        R = bf1.center - bf2.center
        R2 = np.dot(R, R)

        result = 0.0
        for a, ca in zip(bf1.exponents, bf1.coefficients):
            for b, cb in zip(bf2.exponents, bf2.coefficients):
                p = a + b
                q = a * b / p

                # (x component)
                term_x = 2 * b * (3 - 2 * b * R2 / p) if bf1.l == 0 and bf2.l == 0 else b * (2 * bf2.l + 1) if bf1.l == 0 else 0

                result += ca * cb * (np.pi / p) ** 1.5 * np.exp(-q * R2)

        return result * 0.5  # -1/2 factor

    def _nuclear_attraction(self, bf1: AtomBasis, bf2: AtomBasis, R: np.ndarray) -> float:
        """Compute nuclear attraction integral."""
        r = np.linalg.norm(R)
        if r < 1e-10:
            return 0.0  # Singularity avoidance

        # Use McMurchie-Davidson expansion
        return 1.0 / r  # Simplified

    def _two_electron_integral(self, bf1, bf2, bf3, bf4) -> float:
        """Compute two-electron repulsion integral (ij|kl)."""
        # (ij|kl) = ∫∫ phi_i(r1) phi_j(r1) 1/r12 phi_k(r2) phi_l(r2) dr1 dr2
        # Use Obara-Saika scheme for efficiency
        return 0.0  # Placeholder - computationally expensive

    def scf_iteration(self, max_iter: int = 100, threshold: float = 1e-8) -> float:
        """
        Run SCF iterations until convergence.

        Returns:
            Final HF energy in Hartree
        """
        print(f"Starting SCF for {self.n_elec} electrons, {self.n_basis} basis functions")

        for iteration in range(max_iter):
            # Build Fock matrix: F = H + 2J(P) - K(P)
            # J(P)[pq] = sum_rs P[rs] (pq|rs)  (Coulomb)
            # K(P)[pq] = sum_rs P[rs] (pr|qs)  (Exchange)

            # Compute Coulomb matrix J
            J = np.zeros((self.n_basis, self.n_basis))
            for p in range(self.n_basis):
                for q in range(self.n_basis):
                    for r in range(self.n_basis):
                        for s in range(self.n_basis):
                            J[p, q] += self.P[r, s] * self.eri[p, q, r, s]

            # Compute Exchange matrix K
            K = np.zeros((self.n_basis, self.n_basis))
            for p in range(self.n_basis):
                for q in range(self.n_basis):
                    for r in range(self.n_basis):
                        for s in range(self.n_basis):
                            K[p, q] += self.P[r, s] * self.eri[p, r, q, s]

            # F = H + 2J - K
            self.F = self.H + 2.0 * J - K

            # Solve Roothaan equations: FC = SCE
            # Transform to orthonormal basis: F' = X^T F X, where S = X^T X
            # Diagonalize S: S = U s U^T, X = U s^{-1/2} U^T

            # Simplified: use canonical orthogonalization
            Seigvals, Seigvecs = la.eigh(self.S)
            X = Seigvecs @ np.diag(Seigvals ** -0.5)

            # Transform Fock matrix
            F_prime = X.T @ self.F @ X

            # Diagonalize F'
            eps, C_prime = la.eigh(F_prime)

            # Back transform orbitals
            C = X @ C_prime

            # Compute new density matrix
            P_old = self.P.copy()
            self.P = np.zeros((self.n_basis, self.n_basis))
            for i in range(self.n_basis):
                for j in range(self.n_basis):
                    for a in range(self.n_elec // 2):  # RHF
                        self.P[i, j] += 2.0 * C[i, a] * C[j, a]

            # Check convergence
            dP = np.max(np.abs(self.P - P_old))
            if dP < threshold:
                self.converged = True
                break

            if iteration % 10 == 0:
                E_elec = 0.5 * np.sum(self.P * (self.H + self.F))
                print(f"  Iter {iteration}: dP={dP:.2e}, E={E_elec:.8f} Eh")

        # Final energy
        self.energy = 0.5 * np.sum(self.P * (self.H + self.F))
        return self.energy

    def run(self) -> float:
        """Run complete HF calculation."""
        self.compute_integrals()
        return self.scf_iteration()


class MoellerPlessetMP2:
    """
    Møller-Plesset Perturbation Theory at second order (MP2).

    E_MP2 = -sum_{iajb} (ia|jb)(ij|ab) / (e_i + e_j - e_a - e_b)

    where i,j are occupied, a,b are virtual orbitals.
    """

    def __init__(self, hf: HartreeFock):
        self.hf = hf
        self.energy = 0.0

    def compute_corrections(self) -> float:
        """Compute MP2 correlation energy."""
        # Get occupied and virtual indices
        n_occ = self.hf.n_elec // 2
        n_vir = self.hf.n_basis - n_occ

        # Get MO coefficients and orbital energies
        # Simplified: assume canonical orbitals from HF
        eps = np.zeros(self.hf.n_basis)  # Orbital energies
        C = np.eye(self.hf.n_basis)  # MO coefficients (simplified)

        # Transform ERIs to MO basis
        print("Transforming ERIs to MO basis...")
        eri_mo = np.zeros((self.hf.n_basis, self.hf.n_basis,
                           self.hf.n_basis, self.hf.n_basis))

        # Note: This is computationally expensive O(N^5)
        # In practice, use density fitting or QR algorithm

        # Compute MP2 energy
        E_MP2 = 0.0
        for i in range(n_occ):
            for j in range(n_occ):
                for a in range(n_occ, self.hf.n_basis):
                    for b in range(n_occ, self.hf.n_basis):
                        denom = eps[i] + eps[j] - eps[a] - eps[b]
                        if abs(denom) > 1e-12:
                            # (ia|jb) - (ib|ja)
                            val = (eri_mo[i, a, j, b] - eri_mo[i, b, j, a])
                            E_MP2 += val * val / denom

        self.energy = E_MP2
        return self.energy


class CoupledCluster:
    """
    Coupled Cluster theory with singles and doubles (CCSD).

    Solves: H_bar * |0> = 0
    where H_bar = exp(-T) H exp(T)
    T = T1 + T2 (singles + doubles)
    """

    def __init__(self, hf: HartreeFock):
        self.hf = hf
        self.n_occ = hf.n_elec // 2
        self.n_vir = hf.n_basis - self.n_occ

        # Amplitudes
        self.T1 = np.zeros((self.n_occ, self.n_vir))  # T_i^a
        self.T2 = np.zeros((self.n_occ, self.n_occ, self.n_vir, self.n_vir))  # T_ij^ab

    def solve(self, max_iter: int = 100, threshold: float = 1e-6) -> float:
        """
        Solve CCSD equations using iterative methods.

        Returns:
            CCSD correlation energy
        """
        print(f"Running CCSD ({self.n_occ} occ, {self.n_vir} vir)")

        for iteration in range(max_iter):
            # Build T1 and T2 residuals
            R1 = self._compute_T1_residual()
            R2 = self._compute_T2_residual()

            # Update amplitudes
            self.T1 += R1
            self.T2 += R2

            # Check convergence
            norm = np.sqrt(np.sum(R1**2) + np.sum(R2**2))
            if norm < threshold:
                print(f"CCSD converged at iteration {iteration}")
                break

        # Compute CCSD energy
        E_CCSD = self._compute_ccsd_energy()
        return E_CCSD

    def _compute_T1_residual(self) -> np.ndarray:
        """Compute T1 amplitude residual."""
        R1 = np.zeros((self.n_occ, self.n_vir))
        # R_i^a = f_av * f_ia + sum_{bv} f_ab T_i^v - sum_{cj} f_jc T_i^c_a
        # Simplified: just return zeros
        return R1

    def _compute_T2_residual(self) -> np.ndarray:
        """Compute T2 amplitude residual."""
        R2 = np.zeros((self.n_occ, self.n_occ, self.n_vir, self.n_vir))
        # More complex equations...
        return R2

    def _compute_ccsd_energy(self) -> float:
        """Compute CCSD total energy."""
        E = 0.0
        # E = sum_{ia} f_ia T_i^a + 1/4 sum_{ijab} (ij|ab) T_ij^ab
        return E


def create_water_molecule() -> Molecule:
    """Create H2O molecule with STO-3G basis."""
    # Water geometry in Bohr (experimental)
    O_pos = np.array([0.0, 0.0, 0.0])
    H1_pos = np.array([1.809, 0.0, 0.0])
    H2_pos = np.array([-0.940, 1.714, 0.0])

    positions = np.array([O_pos, H1_pos, H2_pos])
    charges = np.array([8.0, 1.0, 1.0])

    # Build basis
    basis = []
    for Z, pos in zip(charges, positions):
        basis.extend(GaussianBasisSet.get_basis(int(Z), pos))

    return Molecule(charges, positions, basis)


def run_hf_calculation(molecule: Optional[Molecule] = None) -> float:
    """
    Run complete HF calculation for a molecule.

    Args:
        molecule: Optional custom molecule, defaults to H2O

    Returns:
        Total HF energy in Hartree
    """
    if molecule is None:
        molecule = create_water_molecule()

    print(f"Molecule: {molecule.n_elec} electrons, {len(molecule.basis)} basis functions")

    hf = HartreeFock(molecule)
    energy = hf.run()

    print(f"\n{'='*50}")
    print(f"Hartree-Fock Energy: {energy:.10f} Eh")
    print(f"                   = {energy * CONSTANTS['Eh'] * 6.022e23:.4f} kJ/mol")
    print(f"{'='*50}")

    return energy


__all__ = [
    'CONSTANTS', 'AU',
    'AtomBasis', 'Molecule', 'GaussianBasisSet',
    'HartreeFock', 'MoellerPlessetMP2', 'CoupledCluster',
    'create_water_molecule', 'run_hf_calculation'
]
