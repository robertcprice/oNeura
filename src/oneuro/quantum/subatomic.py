"""Explicit subatomic and quantum-chemistry state below atomistic chemistry.

This module does not treat quantum behavior as a top-down correction factor.
Instead it builds explicit lower-scale state:

- quark composition of protons and neutrons
- nuclear composition, binding energy, and charge
- electron subshell filling with effective nuclear charge estimates
- valence-space molecular Hamiltonians for small explicit molecules
- fixed-particle exact diagonalization for the active electronic space

The electronic Hamiltonian is an explicit interacting fermion model built from
atom-resolved orbitals. It is intentionally approximate, but it is still a
stateful quantum model rather than a coarse biological multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import importlib
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

BOHR_RADIUS_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988
COULOMB_EV_ANGSTROM = 14.3996454784255
MEV_PER_U = 931.49410242

PROTON_REST_MASS_U = 1.007276466621
NEUTRON_REST_MASS_U = 1.00866491595
ELECTRON_REST_MASS_U = 0.000548579909065

_SPATIAL_DEGENERACY = {"s": 1, "p": 3, "d": 5, "f": 7}
_L_BY_SUBSHELL = {"s": 0, "p": 1, "d": 2, "f": 3}

_AUFBAU_ORDER: tuple[tuple[int, str, int], ...] = (
    (1, "s", 2),
    (2, "s", 2),
    (2, "p", 6),
    (3, "s", 2),
    (3, "p", 6),
    (4, "s", 2),
    (3, "d", 10),
    (4, "p", 6),
    (5, "s", 2),
    (4, "d", 10),
    (5, "p", 6),
    (6, "s", 2),
    (4, "f", 14),
    (5, "d", 10),
    (6, "p", 6),
    (7, "s", 2),
    (5, "f", 14),
    (6, "d", 10),
    (7, "p", 6),
)

_ELEMENTS_BY_SYMBOL: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "W": 74,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Pb": 82,
    "U": 92,
}

_ATOMIC_MASS_U: dict[int, float] = {
    1: 1.008,
    2: 4.002602,
    3: 6.94,
    4: 9.0121831,
    5: 10.81,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998403163,
    10: 20.1797,
    11: 22.98976928,
    12: 24.305,
    13: 26.9815385,
    14: 28.085,
    15: 30.973761998,
    16: 32.06,
    17: 35.45,
    18: 39.948,
    19: 39.0983,
    20: 40.078,
    21: 44.955908,
    22: 47.867,
    23: 50.9415,
    24: 51.9961,
    25: 54.938044,
    26: 55.845,
    27: 58.933194,
    28: 58.6934,
    29: 63.546,
    30: 65.38,
    31: 69.723,
    32: 72.63,
    33: 74.921595,
    34: 78.971,
    35: 79.904,
    36: 83.798,
    37: 85.4678,
    38: 87.62,
    39: 88.90584,
    40: 91.224,
    41: 92.90637,
    42: 95.95,
    43: 98.0,
    44: 101.07,
    45: 102.9055,
    46: 106.42,
    47: 107.8682,
    48: 112.414,
    49: 114.818,
    50: 118.71,
    51: 121.76,
    52: 127.6,
    53: 126.90447,
    54: 131.293,
    74: 183.84,
    78: 195.084,
    79: 196.96657,
    80: 200.592,
    82: 207.2,
    92: 238.02891,
}


def _popcount(value: int) -> int:
    return value.bit_count()


def _normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("empty element symbol")
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


def _resolve_atomic_number(value: str | int) -> int:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"atomic number must be positive, got {value}")
        return value
    symbol = _normalize_symbol(value)
    atomic_number = _ELEMENTS_BY_SYMBOL.get(symbol)
    if atomic_number is None:
        raise ValueError(f"unsupported element symbol: {symbol}")
    return atomic_number


def _default_mass_number(atomic_number: int) -> int:
    atomic_mass = _ATOMIC_MASS_U.get(atomic_number)
    if atomic_mass is not None:
        return max(atomic_number, int(round(atomic_mass)))
    return max(atomic_number, int(round(2.1 * atomic_number)))


def _subshell_l(subshell: str) -> int:
    try:
        return _L_BY_SUBSHELL[subshell]
    except KeyError as exc:
        raise ValueError(f"unsupported subshell: {subshell}") from exc


def _spatial_orbitals_for_subshell(subshell: str) -> int:
    try:
        return _SPATIAL_DEGENERACY[subshell]
    except KeyError as exc:
        raise ValueError(f"unsupported subshell: {subshell}") from exc


def _filled_subshells(electron_count: int) -> list[tuple[int, str, int]]:
    if electron_count < 0:
        raise ValueError(f"electron count must be non-negative, got {electron_count}")
    remaining = electron_count
    occupancies: list[tuple[int, str, int]] = []
    for n, subshell, capacity in _AUFBAU_ORDER:
        if remaining <= 0:
            break
        occupancy = min(capacity, remaining)
        occupancies.append((n, subshell, occupancy))
        remaining -= occupancy
    if remaining:
        raise ValueError(f"Aufbau ordering exhausted with {remaining} electrons left")
    return occupancies


def _slater_shielding(
    atomic_number: int,
    occupancies: Sequence[tuple[int, str, int]],
    target_idx: int,
) -> float:
    target_n, target_subshell, target_occ = occupancies[target_idx]
    if target_occ <= 0:
        return 0.0
    target_l = _subshell_l(target_subshell)
    shielding = 0.0
    for idx, (n, subshell, occupancy) in enumerate(occupancies):
        if occupancy <= 0:
            continue
        other_occ = occupancy
        same_group = idx == target_idx
        if same_group:
            other_occ = max(0, occupancy - 1)
        if other_occ == 0:
            continue

        l = _subshell_l(subshell)
        if target_l <= 1:
            if n == target_n and l <= 1:
                shielding += other_occ * (0.30 if target_n == 1 else 0.35)
            elif n == target_n - 1:
                shielding += other_occ * 0.85
            elif n < target_n - 1:
                shielding += other_occ * 1.00
            elif n == target_n and l > 1:
                shielding += other_occ * 1.00
        else:
            if same_group:
                shielding += other_occ * 0.35
            elif n < target_n:
                shielding += other_occ * 1.00
            elif n == target_n and l < target_l:
                shielding += other_occ * 1.00
    return min(float(atomic_number - 1), shielding)


def _subshell_energy_ev(n: int, subshell: str, effective_nuclear_charge: float) -> float:
    l = _subshell_l(subshell)
    effective_n = max(1.0, n + 0.35 * l)
    return -13.605693122994 * (effective_nuclear_charge**2) / (effective_n**2)


def _orbital_radius_angstrom(n: int, subshell: str, effective_nuclear_charge: float) -> float:
    l = _subshell_l(subshell)
    effective_n = max(1.0, n + 0.35 * l)
    return BOHR_RADIUS_ANGSTROM * (effective_n**2) / max(0.25, effective_nuclear_charge)


def _coerce_symbol_or_atomic_number(value: Any) -> tuple[str, int]:
    if isinstance(value, str):
        symbol = _normalize_symbol(value)
        return symbol, _resolve_atomic_number(symbol)
    if isinstance(value, int):
        atomic_number = _resolve_atomic_number(value)
        symbol = next(
            (symbol for symbol, z in _ELEMENTS_BY_SYMBOL.items() if z == atomic_number),
            str(atomic_number),
        )
        return symbol, atomic_number
    if hasattr(value, "symbol"):
        symbol_attr = value.symbol
        symbol = symbol_attr() if callable(symbol_attr) else symbol_attr
        if isinstance(symbol, str):
            normalized = _normalize_symbol(symbol)
            return normalized, _resolve_atomic_number(normalized)
    if hasattr(value, "atomic_number"):
        atomic_number_attr = value.atomic_number
        atomic_number = atomic_number_attr() if callable(atomic_number_attr) else atomic_number_attr
        if isinstance(atomic_number, int):
            return _coerce_symbol_or_atomic_number(atomic_number)
    raise ValueError(f"could not coerce element identity from {value!r}")


@dataclass(frozen=True)
class SubatomicSpecies:
    name: str
    symbol: str
    charge_e: float
    rest_mass_u: float
    spin: float
    baryon_number: float = 0.0
    lepton_number: int = 0


UP_QUARK = SubatomicSpecies("up quark", "u", 2.0 / 3.0, 0.0023 / MEV_PER_U, 0.5, 1.0 / 3.0)
DOWN_QUARK = SubatomicSpecies("down quark", "d", -1.0 / 3.0, 0.0048 / MEV_PER_U, 0.5, 1.0 / 3.0)
ELECTRON = SubatomicSpecies("electron", "e-", -1.0, ELECTRON_REST_MASS_U, 0.5, 0, 1)
PROTON = SubatomicSpecies("proton", "p+", 1.0, PROTON_REST_MASS_U, 0.5, 1)
NEUTRON = SubatomicSpecies("neutron", "n0", 0.0, NEUTRON_REST_MASS_U, 0.5, 1)


@dataclass(frozen=True)
class QuarkInventory:
    up: int
    down: int
    electrons: int = 0

    @property
    def charge_e(self) -> float:
        return (2.0 * self.up - self.down) / 3.0 - float(self.electrons)

    @property
    def baryon_number(self) -> float:
        return (self.up + self.down) / 3.0


@dataclass(frozen=True)
class NuclearComposition:
    protons: int
    neutrons: int

    def __post_init__(self) -> None:
        if self.protons < 0 or self.neutrons < 0:
            raise ValueError("nuclear composition cannot have negative proton or neutron counts")

    @property
    def mass_number(self) -> int:
        return self.protons + self.neutrons

    @property
    def quark_inventory(self) -> QuarkInventory:
        return QuarkInventory(
            up=2 * self.protons + self.neutrons,
            down=self.protons + 2 * self.neutrons,
        )

    @property
    def charge_e(self) -> int:
        return self.protons

    @property
    def binding_energy_mev(self) -> float:
        a = self.mass_number
        z = self.protons
        if a <= 1:
            return 0.0
        n = self.neutrons
        a_v = 15.8
        a_s = 18.3
        a_c = 0.714
        a_a = 23.2
        pairing = 0.0
        if a % 2 == 0:
            pairing = 12.0 / math.sqrt(a) if z % 2 == 0 and n % 2 == 0 else -12.0 / math.sqrt(a)
        binding = (
            a_v * a
            - a_s * (a ** (2.0 / 3.0))
            - a_c * z * max(0, z - 1) / (a ** (1.0 / 3.0))
            - a_a * ((a - 2 * z) ** 2) / a
            + pairing
        )
        return max(0.0, binding)

    @property
    def rest_mass_u(self) -> float:
        return (
            self.protons * PROTON_REST_MASS_U
            + self.neutrons * NEUTRON_REST_MASS_U
            - self.binding_energy_mev / MEV_PER_U
        )

    @property
    def radius_fm(self) -> float:
        if self.mass_number <= 0:
            return 0.0
        return 1.25 * (self.mass_number ** (1.0 / 3.0))


@dataclass(frozen=True)
class ElectronSubshell:
    n: int
    subshell: str
    electrons: int
    capacity: int
    effective_nuclear_charge: float
    orbital_energy_ev: float
    orbital_radius_angstrom: float

    @property
    def label(self) -> str:
        return f"{self.n}{self.subshell}"

    @property
    def spatial_orbital_count(self) -> int:
        return _spatial_orbitals_for_subshell(self.subshell)


@dataclass(frozen=True)
class SpatialOrbital:
    atom_index: int
    atom_symbol: str
    n: int
    subshell: str
    orientation: int
    shell_electrons: int
    shell_capacity: int
    effective_nuclear_charge: float
    orbital_energy_ev: float
    orbital_radius_angstrom: float
    onsite_repulsion_ev: float

    @property
    def label(self) -> str:
        return f"{self.atom_symbol}{self.atom_index}:{self.n}{self.subshell}{self.orientation}"


@dataclass(frozen=True)
class QuantumAtom:
    symbol: str
    atomic_number: int
    formal_charge: int
    mass_number: int
    nucleus: NuclearComposition
    electron_subshells: tuple[ElectronSubshell, ...]

    @classmethod
    def from_symbol(
        cls,
        symbol: str,
        *,
        formal_charge: int = 0,
        isotope_mass_number: int | None = None,
    ) -> "QuantumAtom":
        normalized_symbol, atomic_number = _coerce_symbol_or_atomic_number(symbol)
        mass_number = isotope_mass_number or _default_mass_number(atomic_number)
        if mass_number < atomic_number:
            raise ValueError(
                f"isotope mass number {mass_number} is smaller than atomic number {atomic_number}"
            )
        electron_count = atomic_number - formal_charge
        occupancies = _filled_subshells(electron_count)
        shells: list[ElectronSubshell] = []
        for idx, (n, subshell, occupancy) in enumerate(occupancies):
            shielding = _slater_shielding(atomic_number, occupancies, idx)
            effective_charge = max(0.25, atomic_number - shielding)
            shells.append(
                ElectronSubshell(
                    n=n,
                    subshell=subshell,
                    electrons=occupancy,
                    capacity=2 * _spatial_orbitals_for_subshell(subshell),
                    effective_nuclear_charge=effective_charge,
                    orbital_energy_ev=_subshell_energy_ev(n, subshell, effective_charge),
                    orbital_radius_angstrom=_orbital_radius_angstrom(n, subshell, effective_charge),
                )
            )
        return cls(
            symbol=normalized_symbol,
            atomic_number=atomic_number,
            formal_charge=formal_charge,
            mass_number=mass_number,
            nucleus=NuclearComposition(
                protons=atomic_number,
                neutrons=mass_number - atomic_number,
            ),
            electron_subshells=tuple(shells),
        )

    @classmethod
    def from_atom_payload(cls, atom_like: Any) -> "QuantumAtom":
        if isinstance(atom_like, QuantumAtom):
            return atom_like
        if isinstance(atom_like, Mapping):
            if "symbol" in atom_like:
                return cls.from_symbol(
                    atom_like["symbol"],
                    formal_charge=int(atom_like.get("formal_charge", 0)),
                    isotope_mass_number=atom_like.get("isotope_mass_number"),
                )
            if "atomic_number" in atom_like:
                return cls.from_symbol(
                    int(atom_like["atomic_number"]),
                    formal_charge=int(atom_like.get("formal_charge", 0)),
                    isotope_mass_number=atom_like.get("isotope_mass_number"),
                )
            if "element" in atom_like:
                symbol, _ = _coerce_symbol_or_atomic_number(atom_like["element"])
                return cls.from_symbol(
                    symbol,
                    formal_charge=int(atom_like.get("formal_charge", 0)),
                    isotope_mass_number=atom_like.get("isotope_mass_number"),
                )

        formal_charge = int(getattr(atom_like, "formal_charge", 0))
        isotope_mass_number = getattr(atom_like, "isotope_mass_number", None)
        if hasattr(atom_like, "symbol") or hasattr(atom_like, "atomic_number"):
            symbol, _ = _coerce_symbol_or_atomic_number(atom_like)
            return cls.from_symbol(
                symbol,
                formal_charge=formal_charge,
                isotope_mass_number=isotope_mass_number,
            )
        if hasattr(atom_like, "element"):
            symbol, _ = _coerce_symbol_or_atomic_number(getattr(atom_like, "element"))
            return cls.from_symbol(
                symbol,
                formal_charge=formal_charge,
                isotope_mass_number=isotope_mass_number,
            )
        return cls.from_symbol(atom_like)

    @property
    def electron_count(self) -> int:
        return sum(shell.electrons for shell in self.electron_subshells)

    @property
    def quark_inventory(self) -> QuarkInventory:
        return QuarkInventory(
            up=self.nucleus.quark_inventory.up,
            down=self.nucleus.quark_inventory.down,
            electrons=self.electron_count,
        )

    @property
    def charge_e(self) -> int:
        return self.atomic_number - self.electron_count

    @property
    def valence_electrons(self) -> int:
        if not self.electron_subshells:
            return 0
        highest_n = max(shell.n for shell in self.electron_subshells if shell.electrons > 0)
        valence = sum(
            shell.electrons
            for shell in self.electron_subshells
            if shell.n == highest_n and shell.electrons > 0
        )
        valence += sum(
            shell.electrons
            for shell in self.electron_subshells
            if shell.n == highest_n - 1 and shell.subshell == "d" and shell.electrons > 0
        )
        return valence

    @property
    def frozen_core_electrons(self) -> int:
        return self.electron_count - self.valence_electrons

    @property
    def ionization_proxy_ev(self) -> float:
        occupied = [shell.orbital_energy_ev for shell in self.electron_subshells if shell.electrons > 0]
        return abs(max(occupied)) if occupied else 0.0

    @property
    def total_orbital_energy_ev(self) -> float:
        return sum(shell.electrons * shell.orbital_energy_ev for shell in self.electron_subshells)

    @property
    def total_rest_mass_u(self) -> float:
        return self.nucleus.rest_mass_u + self.electron_count * ELECTRON_REST_MASS_U

    def active_subshells(self) -> tuple[ElectronSubshell, ...]:
        if not self.electron_subshells:
            return ()
        highest_n = max(shell.n for shell in self.electron_subshells if shell.electrons > 0)
        selected = [
            shell
            for shell in self.electron_subshells
            if shell.electrons > 0 and (shell.n == highest_n or (shell.n == highest_n - 1 and shell.subshell == "d"))
        ]
        return tuple(selected)


@dataclass(frozen=True)
class ExactDiagonalizationResult:
    energies_ev: np.ndarray
    eigenvectors: np.ndarray
    basis_states: tuple[int, ...]
    expected_spatial_occupancies: np.ndarray
    expected_electron_count: float
    nuclear_repulsion_ev: float

    @property
    def ground_state_energy_ev(self) -> float:
        return float(self.energies_ev[0])

    @property
    def ground_state_vector(self) -> np.ndarray:
        return self.eigenvectors[:, 0]


@dataclass(frozen=True)
class LocalNQPUSource:
    root: Path | None
    sdk_python_dir: Path | None
    core_dir: Path | None
    available: bool


@dataclass(frozen=True)
class MolecularActiveSpace:
    atoms: tuple[QuantumAtom, ...]
    positions_angstrom: np.ndarray
    spatial_orbitals: tuple[SpatialOrbital, ...]
    active_electrons: int
    frozen_core_electrons: int
    atom_bond_orders: np.ndarray
    one_body_integrals_ev: np.ndarray
    onsite_repulsion_ev: np.ndarray
    interorbital_coulomb_ev: np.ndarray
    nuclear_repulsion_ev: float

    @classmethod
    def from_atoms(
        cls,
        atoms: Sequence[Any],
        positions_angstrom: Sequence[Sequence[float]],
        *,
        bonds: Sequence[Sequence[Any]] | None = None,
        max_spatial_orbitals: int | None = None,
    ) -> "MolecularActiveSpace":
        quantum_atoms = tuple(QuantumAtom.from_atom_payload(atom) for atom in atoms)
        positions = np.asarray(positions_angstrom, dtype=float)
        if positions.shape != (len(quantum_atoms), 3):
            raise ValueError(
                f"positions must have shape ({len(quantum_atoms)}, 3); got {positions.shape}"
            )
        spatial_orbitals = _build_spatial_orbitals(quantum_atoms)
        if max_spatial_orbitals is not None and len(spatial_orbitals) > max_spatial_orbitals:
            ranked = sorted(
                spatial_orbitals,
                key=lambda orbital: (
                    abs(orbital.orbital_energy_ev),
                    orbital.atom_index,
                    orbital.n,
                    orbital.orientation,
                ),
            )
            spatial_orbitals = tuple(ranked[:max_spatial_orbitals])
        if not spatial_orbitals:
            raise ValueError("no active spatial orbitals were constructed")

        active_electrons = sum(atom.valence_electrons for atom in quantum_atoms)
        capacity = 2 * len(spatial_orbitals)
        if active_electrons > capacity:
            raise ValueError(
                f"active electron count {active_electrons} exceeds active-space capacity {capacity}"
            )
        frozen_core = sum(atom.frozen_core_electrons for atom in quantum_atoms)
        bond_orders = _build_atom_bond_order_matrix(len(quantum_atoms), bonds)
        one_body = _build_one_body_integrals(
            quantum_atoms,
            positions,
            spatial_orbitals,
            atom_bond_orders=bond_orders,
        )
        onsite = np.array(
            [orbital.onsite_repulsion_ev for orbital in spatial_orbitals],
            dtype=float,
        )
        interorbital = _build_interorbital_coulomb(
            positions,
            spatial_orbitals,
            atom_bond_orders=bond_orders,
        )
        nuclear_repulsion = _nuclear_repulsion_ev(quantum_atoms, positions)
        return cls(
            atoms=quantum_atoms,
            positions_angstrom=positions,
            spatial_orbitals=spatial_orbitals,
            active_electrons=active_electrons,
            frozen_core_electrons=frozen_core,
            atom_bond_orders=bond_orders,
            one_body_integrals_ev=one_body,
            onsite_repulsion_ev=onsite,
            interorbital_coulomb_ev=interorbital,
            nuclear_repulsion_ev=nuclear_repulsion,
        )

    @classmethod
    def from_symbols(
        cls,
        symbols: Sequence[str],
        positions_angstrom: Sequence[Sequence[float]],
        *,
        bonds: Sequence[Sequence[Any]] | None = None,
        max_spatial_orbitals: int | None = None,
    ) -> "MolecularActiveSpace":
        return cls.from_atoms(
            list(symbols),
            positions_angstrom,
            bonds=bonds,
            max_spatial_orbitals=max_spatial_orbitals,
        )

    @property
    def num_spatial_orbitals(self) -> int:
        return len(self.spatial_orbitals)

    @property
    def num_spin_orbitals(self) -> int:
        return 2 * self.num_spatial_orbitals

    @property
    def total_electrons(self) -> int:
        return self.active_electrons + self.frozen_core_electrons

    def exact_diagonalize(self, *, max_basis_size: int = 4096) -> ExactDiagonalizationResult:
        basis = tuple(_fixed_particle_basis(self.num_spin_orbitals, self.active_electrons))
        if len(basis) > max_basis_size:
            raise ValueError(
                f"active-space basis has {len(basis)} states, larger than max_basis_size={max_basis_size}"
            )
        index_by_state = {state: idx for idx, state in enumerate(basis)}
        matrix = np.zeros((len(basis), len(basis)), dtype=float)

        for basis_idx, state in enumerate(basis):
            occ_by_spatial = np.zeros(self.num_spatial_orbitals, dtype=float)
            diagonal_energy = self.nuclear_repulsion_ev

            for spatial in range(self.num_spatial_orbitals):
                up_occ = (state >> (2 * spatial)) & 1
                down_occ = (state >> (2 * spatial + 1)) & 1
                occupancy = up_occ + down_occ
                occ_by_spatial[spatial] = occupancy
                diagonal_energy += self.one_body_integrals_ev[spatial, spatial] * occupancy
                if up_occ and down_occ:
                    diagonal_energy += self.onsite_repulsion_ev[spatial]

            for i in range(self.num_spatial_orbitals):
                for j in range(i + 1, self.num_spatial_orbitals):
                    diagonal_energy += (
                        self.interorbital_coulomb_ev[i, j]
                        * occ_by_spatial[i]
                        * occ_by_spatial[j]
                    )
            matrix[basis_idx, basis_idx] = diagonal_energy

            for spin in (0, 1):
                for p in range(self.num_spatial_orbitals):
                    for q in range(self.num_spatial_orbitals):
                        if p == q:
                            continue
                        amplitude = self.one_body_integrals_ev[p, q]
                        if abs(amplitude) < 1.0e-12:
                            continue
                        target_state = _apply_hop(state, 2 * q + spin, 2 * p + spin)
                        if target_state is None:
                            continue
                        hopped_state, sign = target_state
                        matrix[basis_idx, index_by_state[hopped_state]] += amplitude * sign

        matrix = 0.5 * (matrix + matrix.T)
        energies, eigenvectors = np.linalg.eigh(matrix)
        occupancies = _expected_spatial_occupancies(
            basis,
            eigenvectors[:, 0],
            self.num_spatial_orbitals,
        )
        return ExactDiagonalizationResult(
            energies_ev=energies,
            eigenvectors=eigenvectors,
            basis_states=basis,
            expected_spatial_occupancies=occupancies,
            expected_electron_count=float(occupancies.sum()),
            nuclear_repulsion_ev=self.nuclear_repulsion_ev,
        )


def _build_spatial_orbitals(atoms: Sequence[QuantumAtom]) -> tuple[SpatialOrbital, ...]:
    orbitals: list[SpatialOrbital] = []
    for atom_index, atom in enumerate(atoms):
        for shell in atom.active_subshells():
            spatial_count = shell.spatial_orbital_count
            onsite = COULOMB_EV_ANGSTROM / max(0.25, 1.5 * shell.orbital_radius_angstrom)
            for orientation in range(spatial_count):
                orbitals.append(
                    SpatialOrbital(
                        atom_index=atom_index,
                        atom_symbol=atom.symbol,
                        n=shell.n,
                        subshell=shell.subshell,
                        orientation=orientation,
                        shell_electrons=shell.electrons,
                        shell_capacity=shell.capacity,
                        effective_nuclear_charge=shell.effective_nuclear_charge,
                        orbital_energy_ev=shell.orbital_energy_ev,
                        orbital_radius_angstrom=shell.orbital_radius_angstrom,
                        onsite_repulsion_ev=onsite,
                    )
                )
    return tuple(orbitals)


def _bond_order_value(order: Any) -> float:
    if isinstance(order, (int, float)):
        return float(order)
    if isinstance(order, str):
        normalized = order.strip().lower()
        if normalized in {"single", "1"}:
            return 1.0
        if normalized in {"double", "2"}:
            return 2.0
        if normalized in {"triple", "3"}:
            return 3.0
        if normalized in {"aromatic", "1.5"}:
            return 1.5
    if hasattr(order, "bond_order"):
        bond_order_attr = order.bond_order
        if callable(bond_order_attr):
            return float(bond_order_attr())
    raise ValueError(f"unsupported bond order payload: {order!r}")


def _build_atom_bond_order_matrix(
    atom_count: int,
    bonds: Sequence[Sequence[Any]] | None,
) -> np.ndarray:
    matrix = np.zeros((atom_count, atom_count), dtype=float)
    if bonds is None:
        return matrix
    for bond in bonds:
        if isinstance(bond, Mapping):
            a = int(bond["a"])
            b = int(bond["b"])
            order = _bond_order_value(bond.get("order", 1.0))
        elif isinstance(bond, (tuple, list)):
            if len(bond) == 2:
                a, b = int(bond[0]), int(bond[1])
                order = 1.0
            elif len(bond) >= 3:
                a, b = int(bond[0]), int(bond[1])
                order = _bond_order_value(bond[2])
            else:
                raise ValueError(f"unsupported bond payload: {bond!r}")
        else:
            a = int(getattr(bond, "a"))
            b = int(getattr(bond, "b"))
            order = _bond_order_value(getattr(bond, "order", 1.0))
        if a < 0 or b < 0 or a >= atom_count or b >= atom_count:
            raise ValueError(f"bond indices out of range for {atom_count} atoms: {(a, b)}")
        if a == b:
            raise ValueError(f"self-bond is not allowed: {(a, b)}")
        matrix[a, b] = order
        matrix[b, a] = order
    return matrix


def _build_one_body_integrals(
    atoms: Sequence[QuantumAtom],
    positions_angstrom: np.ndarray,
    orbitals: Sequence[SpatialOrbital],
    *,
    atom_bond_orders: np.ndarray | None = None,
) -> np.ndarray:
    matrix = np.zeros((len(orbitals), len(orbitals)), dtype=float)
    for i, orbital_i in enumerate(orbitals):
        diagonal = orbital_i.orbital_energy_ev
        origin = positions_angstrom[orbital_i.atom_index]
        for atom_index, atom in enumerate(atoms):
            if atom_index == orbital_i.atom_index:
                continue
            distance = float(np.linalg.norm(origin - positions_angstrom[atom_index]))
            screened = math.sqrt(distance * distance + orbital_i.orbital_radius_angstrom**2)
            diagonal -= 0.15 * COULOMB_EV_ANGSTROM * atom.atomic_number / max(0.25, screened)
        matrix[i, i] = diagonal

    for i, orbital_i in enumerate(orbitals):
        for j in range(i + 1, len(orbitals)):
            orbital_j = orbitals[j]
            if orbital_i.atom_index == orbital_j.atom_index:
                continue
            distance = float(
                np.linalg.norm(
                    positions_angstrom[orbital_i.atom_index]
                    - positions_angstrom[orbital_j.atom_index]
                )
            )
            mean_radius = 0.5 * (
                orbital_i.orbital_radius_angstrom + orbital_j.orbital_radius_angstrom
            )
            overlap = math.exp(-distance / max(0.25, mean_radius))
            energy_scale = math.sqrt(
                abs(orbital_i.orbital_energy_ev * orbital_j.orbital_energy_ev)
            )
            bond_order = 0.0
            if atom_bond_orders is not None:
                bond_order = atom_bond_orders[orbital_i.atom_index, orbital_j.atom_index]
            symmetry = 1.0 if orbital_i.subshell == orbital_j.subshell else 0.65
            bonded_boost = 1.0 + 0.85 * bond_order
            hopping = -0.25 * energy_scale * overlap * symmetry * bonded_boost
            matrix[i, j] = matrix[j, i] = hopping
    return matrix


def _build_interorbital_coulomb(
    positions_angstrom: np.ndarray,
    orbitals: Sequence[SpatialOrbital],
    *,
    atom_bond_orders: np.ndarray | None = None,
) -> np.ndarray:
    matrix = np.zeros((len(orbitals), len(orbitals)), dtype=float)
    for i, orbital_i in enumerate(orbitals):
        for j in range(i + 1, len(orbitals)):
            orbital_j = orbitals[j]
            distance = float(
                np.linalg.norm(
                    positions_angstrom[orbital_i.atom_index]
                    - positions_angstrom[orbital_j.atom_index]
                )
            )
            screened = math.sqrt(
                distance * distance
                + orbital_i.orbital_radius_angstrom**2
                + orbital_j.orbital_radius_angstrom**2
            )
            bond_order = 0.0
            if atom_bond_orders is not None:
                bond_order = atom_bond_orders[orbital_i.atom_index, orbital_j.atom_index]
            bonded_screen = 1.0 + 0.20 * bond_order
            matrix[i, j] = matrix[j, i] = (
                0.75 * COULOMB_EV_ANGSTROM / max(0.35, screened * bonded_screen)
            )
    return matrix


def _nuclear_repulsion_ev(atoms: Sequence[QuantumAtom], positions_angstrom: np.ndarray) -> float:
    energy = 0.0
    for i, atom_i in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            atom_j = atoms[j]
            distance = float(np.linalg.norm(positions_angstrom[i] - positions_angstrom[j]))
            energy += (
                atom_i.atomic_number
                * atom_j.atomic_number
                * COULOMB_EV_ANGSTROM
                / max(0.25, distance)
            )
    return energy


def _fixed_particle_basis(num_spin_orbitals: int, num_electrons: int) -> Iterable[int]:
    if num_electrons < 0 or num_electrons > num_spin_orbitals:
        raise ValueError(
            f"electron count {num_electrons} is incompatible with {num_spin_orbitals} spin orbitals"
        )
    for occupied in combinations(range(num_spin_orbitals), num_electrons):
        state = 0
        for orbital in occupied:
            state |= 1 << orbital
        yield state


def _apply_annihilation(state: int, orbital: int) -> tuple[int, int] | None:
    if ((state >> orbital) & 1) == 0:
        return None
    sign = -1 if _popcount(state & ((1 << orbital) - 1)) % 2 else 1
    return state & ~(1 << orbital), sign


def _apply_creation(state: int, orbital: int) -> tuple[int, int] | None:
    if ((state >> orbital) & 1) == 1:
        return None
    sign = -1 if _popcount(state & ((1 << orbital) - 1)) % 2 else 1
    return state | (1 << orbital), sign


def _apply_hop(state: int, source_orbital: int, target_orbital: int) -> tuple[int, int] | None:
    removed = _apply_annihilation(state, source_orbital)
    if removed is None:
        return None
    state_after_removal, sign_remove = removed
    created = _apply_creation(state_after_removal, target_orbital)
    if created is None:
        return None
    state_after_creation, sign_create = created
    return state_after_creation, sign_remove * sign_create


def _expected_spatial_occupancies(
    basis_states: Sequence[int],
    eigenvector: np.ndarray,
    num_spatial_orbitals: int,
) -> np.ndarray:
    occupancies = np.zeros(num_spatial_orbitals, dtype=float)
    probabilities = np.abs(eigenvector) ** 2
    for weight, state in zip(probabilities, basis_states):
        for spatial in range(num_spatial_orbitals):
            occupancies[spatial] += weight * (
                ((state >> (2 * spatial)) & 1) + ((state >> (2 * spatial + 1)) & 1)
            )
    return occupancies


def discover_local_nqpu_sdk(
    *,
    env: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> LocalNQPUSource:
    env = env or os.environ
    repo_root = repo_root or Path(__file__).resolve().parents[3]
    candidates: list[Path] = []

    if env.get("ONEURO_NQPU_ROOT"):
        candidates.append(Path(env["ONEURO_NQPU_ROOT"]).expanduser())
    candidates.append(repo_root.parent / "nQPU")

    for root in candidates:
        sdk_python = root / "sdk" / "python"
        core_dir = sdk_python / "core"
        if (core_dir / "quantum_backend.py").exists():
            return LocalNQPUSource(
                root=root,
                sdk_python_dir=sdk_python,
                core_dir=core_dir,
                available=True,
            )
    return LocalNQPUSource(
        root=None,
        sdk_python_dir=None,
        core_dir=None,
        available=False,
    )


def load_local_nqpu_quantum_backend(
    *,
    env: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
):
    source = discover_local_nqpu_sdk(env=env, repo_root=repo_root)
    if not source.available or source.core_dir is None:
        return None
    core_dir = str(source.core_dir)
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
    try:
        return importlib.import_module("quantum_backend")
    except Exception:
        return None


def reference_molecule_geometry(
    name: str,
    *,
    bond_length_angstrom: float | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    normalized = name.strip().lower()
    if normalized == "h2":
        bond = 0.74 if bond_length_angstrom is None else bond_length_angstrom
        return ("H", "H"), np.array([[0.0, 0.0, -bond / 2.0], [0.0, 0.0, bond / 2.0]])
    if normalized == "lih":
        bond = 1.596 if bond_length_angstrom is None else bond_length_angstrom
        return ("Li", "H"), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]])
    if normalized == "beh2":
        bond = 1.326 if bond_length_angstrom is None else bond_length_angstrom
        return ("H", "Be", "H"), np.array([[0.0, 0.0, -bond], [0.0, 0.0, 0.0], [0.0, 0.0, bond]])
    raise ValueError(f"unsupported reference molecule: {name}")


def build_reference_active_space(
    name: str,
    *,
    bond_length_angstrom: float | None = None,
) -> MolecularActiveSpace:
    symbols, positions = reference_molecule_geometry(
        name,
        bond_length_angstrom=bond_length_angstrom,
    )
    return MolecularActiveSpace.from_symbols(symbols, positions)


def nqpu_reference_ground_state_energy(
    name: str,
    *,
    bond_length_angstrom: float | None = None,
    env: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> float | None:
    backend = load_local_nqpu_quantum_backend(env=env, repo_root=repo_root)
    if backend is None or not hasattr(backend, "VQEMolecule"):
        return None
    bond = bond_length_angstrom
    if bond is None:
        _, positions = reference_molecule_geometry(name)
        if len(positions) == 2:
            bond = float(np.linalg.norm(positions[0] - positions[1]))
    try:
        solver = backend.VQEMolecule(backend="default.qubit")
        return float(solver.compute_ground_state_energy(name.upper(), bond_length=bond))
    except Exception:
        return None


__all__ = [
    "BOHR_RADIUS_ANGSTROM",
    "COULOMB_EV_ANGSTROM",
    "DOWN_QUARK",
    "ELECTRON",
    "ELECTRON_REST_MASS_U",
    "ElectronSubshell",
    "ExactDiagonalizationResult",
    "LocalNQPUSource",
    "MolecularActiveSpace",
    "NEUTRON",
    "NuclearComposition",
    "PROTON",
    "QuantumAtom",
    "QuarkInventory",
    "SpatialOrbital",
    "SubatomicSpecies",
    "UP_QUARK",
    "build_reference_active_space",
    "discover_local_nqpu_sdk",
    "load_local_nqpu_quantum_backend",
    "nqpu_reference_ground_state_energy",
    "reference_molecule_geometry",
]
