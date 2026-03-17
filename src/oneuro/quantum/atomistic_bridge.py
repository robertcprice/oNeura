"""Bridge explicit atomistic payloads into the subatomic/active-space layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .subatomic import (
    ExactDiagonalizationResult,
    MolecularActiveSpace,
    nqpu_reference_ground_state_energy,
)


@dataclass(frozen=True)
class AtomisticAtomRecord:
    symbol: str
    position_angstrom: tuple[float, float, float]
    formal_charge: int = 0
    isotope_mass_number: int | None = None
    partial_charge: float | None = None


@dataclass(frozen=True)
class QuantumMicrodomainReport:
    atoms: tuple[AtomisticAtomRecord, ...]
    active_space: MolecularActiveSpace
    exact_solution: ExactDiagonalizationResult | None
    nqpu_reference_energy_ev: float | None
    net_formal_charge: int
    active_basis_size: int | None

    @property
    def ground_state_energy_ev(self) -> float | None:
        if self.exact_solution is None:
            return None
        return self.exact_solution.ground_state_energy_ev


def _coerce_bond_payloads(bonds: Sequence[Any]) -> list[tuple[int, int, str]]:
    payloads: list[tuple[int, int, str]] = []
    for bond in bonds:
        if isinstance(bond, Mapping):
            a = int(bond["a"])
            b = int(bond["b"])
            order = str(bond.get("order", "single"))
        elif isinstance(bond, (tuple, list)):
            if len(bond) == 2:
                a, b = int(bond[0]), int(bond[1])
                order = "single"
            elif len(bond) >= 3:
                a, b = int(bond[0]), int(bond[1])
                order = str(bond[2])
            else:
                raise TypeError(f"unsupported bond payload: {bond!r}")
        else:
            a = int(getattr(bond, "a"))
            b = int(getattr(bond, "b"))
            order = str(getattr(bond, "order", "single"))
        payloads.append((a, b, order))
    return payloads


def _coerce_atom_symbol(atom_like: Any) -> str:
    if isinstance(atom_like, Mapping):
        if "symbol" in atom_like:
            return str(atom_like["symbol"])
        if "element" in atom_like:
            element = atom_like["element"]
            if isinstance(element, str):
                return element
            if hasattr(element, "symbol"):
                symbol_attr = element.symbol
                symbol = symbol_attr() if callable(symbol_attr) else symbol_attr
                if isinstance(symbol, str):
                    return symbol
    if hasattr(atom_like, "symbol"):
        symbol_attr = atom_like.symbol
        symbol = symbol_attr() if callable(symbol_attr) else symbol_attr
        if isinstance(symbol, str):
            return symbol
    if hasattr(atom_like, "element"):
        element = atom_like.element
        if isinstance(element, str):
            return element
        if hasattr(element, "symbol"):
            symbol_attr = element.symbol
            symbol = symbol_attr() if callable(symbol_attr) else symbol_attr
            if isinstance(symbol, str):
                return symbol
    raise ValueError(f"could not determine atom symbol from {atom_like!r}")


def _coerce_formal_charge(atom_like: Any) -> int:
    if isinstance(atom_like, Mapping):
        if "formal_charge" in atom_like:
            return int(atom_like["formal_charge"])
        return 0
    if hasattr(atom_like, "formal_charge"):
        return int(getattr(atom_like, "formal_charge"))
    return 0


def _coerce_partial_charge(atom_like: Any) -> float | None:
    if isinstance(atom_like, Mapping):
        if "partial_charge" in atom_like:
            return float(atom_like["partial_charge"])
        if "charge" in atom_like and "formal_charge" not in atom_like:
            return float(atom_like["charge"])
        return None
    if hasattr(atom_like, "partial_charge"):
        return float(getattr(atom_like, "partial_charge"))
    if hasattr(atom_like, "charge") and not hasattr(atom_like, "formal_charge"):
        return float(getattr(atom_like, "charge"))
    return None


def _coerce_isotope(atom_like: Any) -> int | None:
    if isinstance(atom_like, Mapping):
        isotope = atom_like.get("isotope_mass_number")
        return int(isotope) if isotope is not None else None
    isotope = getattr(atom_like, "isotope_mass_number", None)
    return int(isotope) if isotope is not None else None


def atom_records_from_atoms_and_positions(
    atoms: Sequence[Any],
    positions_angstrom: Sequence[Sequence[float]],
) -> tuple[AtomisticAtomRecord, ...]:
    positions = np.asarray(positions_angstrom, dtype=float)
    if positions.shape != (len(atoms), 3):
        raise ValueError(f"positions must have shape ({len(atoms)}, 3); got {positions.shape}")
    records: list[AtomisticAtomRecord] = []
    for atom_like, position in zip(atoms, positions):
        records.append(
            AtomisticAtomRecord(
                symbol=_coerce_atom_symbol(atom_like),
                position_angstrom=(float(position[0]), float(position[1]), float(position[2])),
                formal_charge=_coerce_formal_charge(atom_like),
                isotope_mass_number=_coerce_isotope(atom_like),
                partial_charge=_coerce_partial_charge(atom_like),
            )
        )
    return tuple(records)


def atom_records_from_md_molecule(
    molecule: Any,
    positions_angstrom: Sequence[Sequence[float]],
) -> tuple[AtomisticAtomRecord, ...]:
    atoms = getattr(molecule, "atoms", None)
    if atoms is None:
        raise TypeError("MD molecule does not expose an atoms sequence")
    return atom_records_from_atoms_and_positions(atoms, positions_angstrom)


def atom_records_from_native_molecule_graph(
    molecule_graph: Any,
    positions_angstrom: Sequence[Sequence[float]],
) -> tuple[AtomisticAtomRecord, ...]:
    payloads = molecule_graph.atom_payloads()
    positions = np.asarray(positions_angstrom, dtype=float)
    if positions.shape != (len(payloads), 3):
        raise ValueError(
            f"positions must have shape ({len(payloads)}, 3); got {positions.shape}"
        )
    records: list[AtomisticAtomRecord] = []
    for payload, position in zip(payloads, positions):
        if isinstance(payload, Mapping):
            symbol = str(payload["symbol"])
            formal_charge = int(payload.get("formal_charge", 0))
            isotope_mass_number = payload.get("isotope_mass_number")
        elif isinstance(payload, (tuple, list)) and len(payload) >= 3:
            symbol = str(payload[0])
            formal_charge = int(payload[1])
            isotope_mass_number = payload[2]
        else:
            raise TypeError(f"unsupported native atom payload: {payload!r}")
        records.append(
            AtomisticAtomRecord(
                symbol=symbol,
                position_angstrom=(float(position[0]), float(position[1]), float(position[2])),
                formal_charge=formal_charge,
                isotope_mass_number=(
                    None if isotope_mass_number is None else int(isotope_mass_number)
                ),
            )
        )
    return tuple(records)


def atom_records_from_docking_ligand(
    ligand: Any,
    *,
    formal_charges: Sequence[int] | None = None,
) -> tuple[AtomisticAtomRecord, ...]:
    positions = np.asarray(getattr(ligand, "atoms"), dtype=float)
    atom_types = list(getattr(ligand, "atom_types"))
    if formal_charges is not None and len(formal_charges) != len(atom_types):
        raise ValueError(
            f"formal_charges length {len(formal_charges)} does not match ligand atom count {len(atom_types)}"
        )
    records: list[AtomisticAtomRecord] = []
    for idx, (symbol, position) in enumerate(zip(atom_types, positions)):
        records.append(
            AtomisticAtomRecord(
                symbol=str(symbol),
                position_angstrom=(float(position[0]), float(position[1]), float(position[2])),
                formal_charge=0 if formal_charges is None else int(formal_charges[idx]),
            )
        )
    return tuple(records)


def build_quantum_microdomain(
    atom_records: Sequence[AtomisticAtomRecord],
    *,
    bond_payloads: Sequence[Any] | None = None,
    max_spatial_orbitals: int | None = None,
    diagonalize: bool = True,
    max_basis_size: int = 4096,
    reference_name: str | None = None,
    use_nqpu_reference: bool = False,
) -> QuantumMicrodomainReport:
    symbols = [record.symbol for record in atom_records]
    positions = np.asarray([record.position_angstrom for record in atom_records], dtype=float)
    atom_payload = [
        {
            "symbol": record.symbol,
            "formal_charge": record.formal_charge,
            "isotope_mass_number": record.isotope_mass_number,
        }
        for record in atom_records
    ]
    active_space = MolecularActiveSpace.from_atoms(
        atom_payload,
        positions,
        bonds=None if bond_payloads is None else _coerce_bond_payloads(bond_payloads),
        max_spatial_orbitals=max_spatial_orbitals,
    )
    exact_solution = None
    active_basis_size = None
    if diagonalize:
        exact_solution = active_space.exact_diagonalize(max_basis_size=max_basis_size)
        active_basis_size = len(exact_solution.basis_states)
    nqpu_energy = None
    if use_nqpu_reference and reference_name is not None:
        nqpu_energy = nqpu_reference_ground_state_energy(reference_name)
    return QuantumMicrodomainReport(
        atoms=tuple(atom_records),
        active_space=active_space,
        exact_solution=exact_solution,
        nqpu_reference_energy_ev=nqpu_energy,
        net_formal_charge=sum(record.formal_charge for record in atom_records),
        active_basis_size=active_basis_size,
    )


def build_md_quantum_microdomain(
    molecule: Any,
    positions_angstrom: Sequence[Sequence[float]],
    *,
    max_spatial_orbitals: int | None = None,
    diagonalize: bool = True,
    max_basis_size: int = 4096,
) -> QuantumMicrodomainReport:
    records = atom_records_from_md_molecule(molecule, positions_angstrom)
    bond_payloads = getattr(molecule, "bonds", None)
    return build_quantum_microdomain(
        records,
        bond_payloads=bond_payloads,
        max_spatial_orbitals=max_spatial_orbitals,
        diagonalize=diagonalize,
        max_basis_size=max_basis_size,
    )


def build_native_graph_quantum_microdomain(
    molecule_graph: Any,
    positions_angstrom: Sequence[Sequence[float]],
    *,
    max_spatial_orbitals: int | None = None,
    diagonalize: bool = True,
    max_basis_size: int = 4096,
) -> QuantumMicrodomainReport:
    records = atom_records_from_native_molecule_graph(molecule_graph, positions_angstrom)
    bond_payloads = molecule_graph.bond_payloads() if hasattr(molecule_graph, "bond_payloads") else None
    return build_quantum_microdomain(
        records,
        bond_payloads=bond_payloads,
        max_spatial_orbitals=max_spatial_orbitals,
        diagonalize=diagonalize,
        max_basis_size=max_basis_size,
    )


def build_docking_quantum_microdomain(
    ligand: Any,
    *,
    formal_charges: Sequence[int] | None = None,
    max_spatial_orbitals: int | None = None,
    diagonalize: bool = True,
    max_basis_size: int = 4096,
) -> QuantumMicrodomainReport:
    records = atom_records_from_docking_ligand(ligand, formal_charges=formal_charges)
    return build_quantum_microdomain(
        records,
        bond_payloads=getattr(ligand, "bonds", None),
        max_spatial_orbitals=max_spatial_orbitals,
        diagonalize=diagonalize,
        max_basis_size=max_basis_size,
    )


__all__ = [
    "AtomisticAtomRecord",
    "QuantumMicrodomainReport",
    "atom_records_from_atoms_and_positions",
    "atom_records_from_docking_ligand",
    "atom_records_from_md_molecule",
    "atom_records_from_native_molecule_graph",
    "build_docking_quantum_microdomain",
    "build_md_quantum_microdomain",
    "build_native_graph_quantum_microdomain",
    "build_quantum_microdomain",
]
