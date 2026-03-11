"""Native shared state for whole-cell simulation inside oNeuro."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, MutableMapping

from .contracts import WholeCellContract, WholeCellProvenance


class CellCompartment(str, Enum):
    """Compartments commonly used by minimal bacterial whole-cell models."""

    CYTOPLASM = "cytoplasm"
    MEMBRANE = "membrane"
    EXTRACELLULAR = "extracellular"
    NUCLEOID = "nucleoid"
    PERIPLASM = "periplasm"


@dataclass
class ChromosomeState:
    """Coarse chromosome state for native whole-cell scheduling."""

    genome_bp: int
    replicated_bp: int = 0
    chromosome_count: int = 1
    origin_count: int = 1
    terminus_count: int = 1
    segregated_fraction: float = 0.0

    @property
    def replicated_fraction(self) -> float:
        """Fraction of one genome that has been replicated."""

        if self.genome_bp <= 0:
            return 0.0
        return max(0.0, min(1.0, self.replicated_bp / float(self.genome_bp)))


@dataclass
class GeometryState:
    """Minimal cell geometry and division progress."""

    radius_nm: float
    surface_area_nm2: float
    volume_nm3: float
    division_progress: float = 0.0
    morphology: str = "spherical"


@dataclass
class WholeCellState:
    """Shared mutable state for a single digital cell."""

    organism: str
    time_ms: float = 0.0
    compartments: Dict[CellCompartment, Dict[str, float]] = field(default_factory=dict)
    metabolites_mM: Dict[str, float] = field(default_factory=dict)
    proteins: Dict[str, float] = field(default_factory=dict)
    transcripts: Dict[str, float] = field(default_factory=dict)
    chromosome: ChromosomeState = field(
        default_factory=lambda: ChromosomeState(genome_bp=543_000)
    )
    geometry: GeometryState = field(
        default_factory=lambda: GeometryState(
            radius_nm=200.0,
            surface_area_nm2=502_654.0,
            volume_nm3=33_510_321.0,
        )
    )
    contract: WholeCellContract = field(default_factory=WholeCellContract)
    provenance: WholeCellProvenance = field(default_factory=WholeCellProvenance)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_history: list[dict[str, Any]] = field(default_factory=list)

    def advance_time(self, dt_ms: float) -> None:
        """Advance biological time."""

        self.time_ms += float(dt_ms)

    def apply_deltas(
        self,
        *,
        compartment_deltas: Mapping[str, Mapping[str, float]] | None = None,
        metabolite_deltas: Mapping[str, float] | None = None,
        protein_deltas: Mapping[str, float] | None = None,
        transcript_deltas: Mapping[str, float] | None = None,
        chromosome_updates: Mapping[str, Any] | None = None,
        geometry_updates: Mapping[str, Any] | None = None,
    ) -> None:
        """Apply stage-level state updates in-place."""

        if compartment_deltas:
            for compartment_name, deltas in compartment_deltas.items():
                compartment_key = CellCompartment(compartment_name)
                compartment = self.compartments.setdefault(compartment_key, {})
                self._apply_numeric_map_deltas(compartment, deltas)
        if metabolite_deltas:
            self._apply_numeric_map_deltas(self.metabolites_mM, metabolite_deltas)
        if protein_deltas:
            self._apply_numeric_map_deltas(self.proteins, protein_deltas)
        if transcript_deltas:
            self._apply_numeric_map_deltas(self.transcripts, transcript_deltas)
        if chromosome_updates:
            for key, value in chromosome_updates.items():
                setattr(self.chromosome, key, value)
        if geometry_updates:
            for key, value in geometry_updates.items():
                setattr(self.geometry, key, value)

    def clone(self) -> "WholeCellState":
        """Deep copy the state for what-if or checkpoint workflows."""

        return copy.deepcopy(self)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot."""

        payload = asdict(self)
        payload["compartments"] = {
            compartment.value: values
            for compartment, values in self.compartments.items()
        }
        return payload

    @staticmethod
    def _apply_numeric_map_deltas(
        target: MutableMapping[str, float],
        deltas: Mapping[str, float],
    ) -> None:
        for key, delta in deltas.items():
            target[key] = target.get(key, 0.0) + float(delta)


def syn3a_minimal_state() -> WholeCellState:
    """Create a coarse native starting state for a Syn3A-like cell."""

    return WholeCellState(
        organism="JCVI-syn3A",
        compartments={
            CellCompartment.CYTOPLASM: {
                "ATP": 8_000.0,
                "ADP": 800.0,
                "ribosome": 500.0,
                "RNAP": 180.0,
            },
            CellCompartment.MEMBRANE: {
                "lipid": 100_000.0,
                "membrane_protein": 5_000.0,
            },
            CellCompartment.NUCLEOID: {
                "chromosome_bead": 54_300.0,
                "origin_site": 1.0,
            },
        },
        metabolites_mM={
            "ATP": 1.2,
            "GTP": 0.3,
            "UTP": 0.3,
            "CTP": 0.4,
            "glucose": 1.0,
            "dATP": 0.1,
            "dTTP": 0.1,
            "dCTP": 0.1,
            "dGTP": 0.1,
        },
        proteins={
            "DnaA": 80.0,
            "SMC": 25.0,
            "FtsZ": 90.0,
        },
        transcripts={
            "dnaA": 2.0,
            "ftsZ": 3.0,
        },
        chromosome=ChromosomeState(genome_bp=543_000),
        geometry=GeometryState(
            radius_nm=200.0,
            surface_area_nm2=502_654.0,
            volume_nm3=33_510_321.0,
        ),
        metadata={
            "model_family": "native_whole_cell_skeleton",
            "organism_class": "minimal_bacterium",
        },
        provenance=WholeCellProvenance(
            source_dataset="JCVI-syn3A minimal native skeleton",
            backend="python_skeleton",
            notes=(
                "Coarse shared state for scheduler and contract tests.",
            ),
        ),
    )
