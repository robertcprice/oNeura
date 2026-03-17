"""Explicit quantum reaction-site analysis on top of atom graphs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .atomistic_bridge import (
    QuantumMicrodomainReport,
    build_native_graph_quantum_microdomain,
)


@dataclass(frozen=True)
class QuantumReactionDelta:
    before: QuantumMicrodomainReport
    after: QuantumMicrodomainReport
    delta_ground_state_energy_ev: float | None
    delta_nuclear_repulsion_ev: float | None
    delta_net_formal_charge: int


@dataclass
class QuantumReactiveSite:
    """Mutable reaction neighborhood backed by an explicit atom graph."""

    graph: Any
    positions_angstrom: np.ndarray
    max_spatial_orbitals: int | None = None
    max_basis_size: int = 4096
    diagonalize: bool = True

    def __post_init__(self) -> None:
        self.positions_angstrom = np.asarray(self.positions_angstrom, dtype=float)
        atom_payloads = self.graph.atom_payloads()
        if self.positions_angstrom.shape != (len(atom_payloads), 3):
            raise ValueError(
                f"positions must have shape ({len(atom_payloads)}, 3); got {self.positions_angstrom.shape}"
            )

    def analyze(self) -> QuantumMicrodomainReport:
        return build_native_graph_quantum_microdomain(
            self.graph,
            self.positions_angstrom,
            max_spatial_orbitals=self.max_spatial_orbitals,
            diagonalize=self.diagonalize,
            max_basis_size=self.max_basis_size,
        )

    def clone(self) -> "QuantumReactiveSite":
        return QuantumReactiveSite(
            graph=_clone_graph(self.graph),
            positions_angstrom=self.positions_angstrom.copy(),
            max_spatial_orbitals=self.max_spatial_orbitals,
            max_basis_size=self.max_basis_size,
            diagonalize=self.diagonalize,
        )

    def form_bond(self, a: int, b: int, order: str = "single") -> None:
        self.graph.add_bond(a, b, order)

    def break_bond(self, a: int, b: int) -> None:
        if not hasattr(self.graph, "break_bond"):
            raise TypeError("graph does not expose break_bond(...)")
        self.graph.break_bond(a, b)

    def change_bond_order(self, a: int, b: int, order: str) -> None:
        if hasattr(self.graph, "set_bond_order"):
            self.graph.set_bond_order(a, b, order)
            return
        raise TypeError("graph does not expose set_bond_order(...)")

    def set_formal_charge(self, atom_idx: int, formal_charge: int) -> None:
        if hasattr(self.graph, "set_formal_charge"):
            self.graph.set_formal_charge(atom_idx, formal_charge)
            return
        raise TypeError("graph does not expose set_formal_charge(...)")

    def evaluate_edit_sequence(self, edits: Sequence[dict[str, Any]]) -> QuantumReactionDelta:
        before = self.analyze()
        mutated = self.clone()
        for edit in edits:
            kind = edit["kind"]
            if kind == "form_bond":
                mutated.form_bond(int(edit["a"]), int(edit["b"]), str(edit.get("order", "single")))
            elif kind == "break_bond":
                mutated.break_bond(int(edit["a"]), int(edit["b"]))
            elif kind == "change_bond_order":
                mutated.change_bond_order(
                    int(edit["a"]),
                    int(edit["b"]),
                    str(edit["order"]),
                )
            elif kind == "set_formal_charge":
                mutated.set_formal_charge(int(edit["atom"]), int(edit["formal_charge"]))
            else:
                raise ValueError(f"unsupported reaction-site edit kind: {kind}")
        after = mutated.analyze()
        delta_ground = None
        if before.ground_state_energy_ev is not None and after.ground_state_energy_ev is not None:
            delta_ground = after.ground_state_energy_ev - before.ground_state_energy_ev
        delta_nuclear = (
            after.active_space.nuclear_repulsion_ev - before.active_space.nuclear_repulsion_ev
        )
        return QuantumReactionDelta(
            before=before,
            after=after,
            delta_ground_state_energy_ev=delta_ground,
            delta_nuclear_repulsion_ev=delta_nuclear,
            delta_net_formal_charge=after.net_formal_charge - before.net_formal_charge,
        )


def _clone_graph(graph: Any) -> Any:
    if hasattr(graph, "to_json") and hasattr(type(graph), "from_json"):
        return type(graph).from_json(graph.to_json())
    return copy.deepcopy(graph)


__all__ = [
    "QuantumReactionDelta",
    "QuantumReactiveSite",
]
