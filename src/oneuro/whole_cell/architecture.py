"""Core architecture surfaces for whole-cell simulation.

This package is intentionally separate from the neural stack.  The goal is to
support non-neural digital cells whose state must be coordinated across
multiple solvers and biological timescales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from .contracts import WholeCellContract, WholeCellProvenance


class CouplingStage(str, Enum):
    """Major solver stages in a hybrid whole-cell runtime."""

    RDME = "rdme"
    CME = "cme"
    ODE = "ode"
    BD = "bd"
    GEOMETRY = "geometry"


class ExternalTool(str, Enum):
    """Published external tools used by the current minimal-cell reference stack."""

    LATTICE_MICROBES = "lattice_microbes"
    ODECELL = "odecell"
    BTREE_CHROMO_GPU = "btree_chromo_gpu"
    LAMMPS = "lammps"
    SC_CHAIN_GENERATION = "sc_chain_generation"
    FREEDTS = "freedts"


@dataclass(frozen=True)
class WholeCellConfig:
    """Static runtime configuration for a whole-cell program."""

    organism: str
    lattice_spacing_nm: float
    rdme_step_us: float
    hook_interval_ms: float
    chromosome_bp_per_bead: int
    geometry_mode: str
    target_gpu_count: int
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class WholeCellProgramSpec:
    """High-level program definition for a whole-cell target."""

    name: str
    organism: str
    config: WholeCellConfig
    coupling_stages: Tuple[CouplingStage, ...]
    external_tools: Tuple[ExternalTool, ...]
    reusable_oneuro_surfaces: Tuple[str, ...]
    missing_core_surfaces: Tuple[str, ...]
    contract: WholeCellContract = field(default_factory=WholeCellContract)
    provenance: WholeCellProvenance = field(default_factory=WholeCellProvenance)


def syn3a_reference_program() -> WholeCellProgramSpec:
    """Published minimal-cell baseline used as the current design target."""

    config = WholeCellConfig(
        organism="JCVI-syn3A",
        lattice_spacing_nm=10.0,
        rdme_step_us=50.0,
        hook_interval_ms=12.5,
        chromosome_bp_per_bead=10,
        geometry_mode="membrane-growth-driven division",
        target_gpu_count=2,
        notes=(
            "Reference architecture derived from the published minimal-cell 4D stack.",
            "Native oNeura support does not exist yet; use explicit adapters.",
        ),
    )
    return WholeCellProgramSpec(
        name="syn3a_reference_program",
        organism=config.organism,
        config=config,
        coupling_stages=(
            CouplingStage.RDME,
            CouplingStage.CME,
            CouplingStage.ODE,
            CouplingStage.BD,
            CouplingStage.GEOMETRY,
        ),
        external_tools=(
            ExternalTool.LATTICE_MICROBES,
            ExternalTool.ODECELL,
            ExternalTool.BTREE_CHROMO_GPU,
            ExternalTool.LAMMPS,
            ExternalTool.SC_CHAIN_GENERATION,
            ExternalTool.FREEDTS,
        ),
        reusable_oneuro_surfaces=(
            "step-based subsystem scheduling",
            "dataclass-based model definitions",
            "existing diffusion-oriented coding patterns",
            "experiment/reporting infrastructure",
        ),
        missing_core_surfaces=(
            "whole-cell solver scheduler",
            "intracellular spatial lattice",
            "genome-scale molecular count state",
            "chromosome dynamics interface",
            "growth and division geometry controller",
            "whole-cell validation suite",
        ),
        provenance=WholeCellProvenance(
            source_dataset="JCVI-syn3A reference program",
            backend="external_reference",
            notes=(
                "Contract-frozen whole-cell reference program.",
                "Use this surface for runtime and manifest schema compatibility checks.",
            ),
        ),
    )
