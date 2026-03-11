"""Runtime manifests for hybrid whole-cell simulation programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .architecture import CouplingStage, ExternalTool, WholeCellProgramSpec, syn3a_reference_program
from .contracts import WholeCellContract, WholeCellProvenance


@dataclass(frozen=True)
class SolverCadence:
    """Biological cadence for a solver stage."""

    stage: CouplingStage
    interval_bio_ms: float
    notes: str = ""


@dataclass(frozen=True)
class RuntimeDependency:
    """External dependency required by a whole-cell runtime."""

    tool: ExternalTool
    required: bool = True
    executable: Optional[str] = None
    env_var: Optional[str] = None
    repo_subdir: Optional[Path] = None
    notes: str = ""


@dataclass(frozen=True)
class WholeCellRuntimeManifest:
    """Concrete runtime description for a whole-cell program."""

    program: WholeCellProgramSpec
    repository_env_var: Optional[str]
    entrypoint: Path
    restart_entrypoint: Path
    cadences: Tuple[SolverCadence, ...]
    dependencies: Tuple[RuntimeDependency, ...]
    expected_repo_paths: Tuple[Path, ...]
    contract: WholeCellContract = field(default_factory=WholeCellContract)
    provenance: WholeCellProvenance = field(default_factory=WholeCellProvenance)


def syn3a_reference_manifest() -> WholeCellRuntimeManifest:
    """Return the runtime manifest for the published Syn3A baseline."""

    program = syn3a_reference_program()
    return WholeCellRuntimeManifest(
        program=program,
        repository_env_var="ONEURO_MC4D_REPO",
        entrypoint=Path("Whole_Cell_Minimal_Cell.py"),
        restart_entrypoint=Path("Restart_Whole_Cell_Minimal_Cell.py"),
        cadences=(
            SolverCadence(
                stage=CouplingStage.RDME,
                interval_bio_ms=0.05,
                notes="50 us RDME parent step from the published MC4D flow.",
            ),
            SolverCadence(
                stage=CouplingStage.CME,
                interval_bio_ms=12.5,
                notes="Hook cadence used to communicate global stochastic state.",
            ),
            SolverCadence(
                stage=CouplingStage.ODE,
                interval_bio_ms=12.5,
                notes="Metabolism is updated at hook boundaries in the published stack.",
            ),
            SolverCadence(
                stage=CouplingStage.BD,
                interval_bio_ms=12.5,
                notes="Chromosome BD runs as a coupled side solver.",
            ),
            SolverCadence(
                stage=CouplingStage.GEOMETRY,
                interval_bio_ms=12.5,
                notes="Growth and division geometry are updated from membrane synthesis.",
            ),
        ),
        dependencies=(
            RuntimeDependency(
                tool=ExternalTool.LATTICE_MICROBES,
                required=True,
                notes="Built in the active conda environment, per MC4D README.",
            ),
            RuntimeDependency(
                tool=ExternalTool.ODECELL,
                required=True,
                notes="Metabolic ODE engine used by the published MC4D stack.",
            ),
            RuntimeDependency(
                tool=ExternalTool.LAMMPS,
                required=True,
                executable="lammps",
                notes="LAMMPS must be available on PATH for chromosome dynamics.",
            ),
            RuntimeDependency(
                tool=ExternalTool.BTREE_CHROMO_GPU,
                required=True,
                env_var="ONEURO_MC4D_DNA_SOFTWARE_DIR",
                repo_subdir=Path("btree_chromo_gpu"),
                notes="DNA dynamics software directory must contain this checkout.",
            ),
            RuntimeDependency(
                tool=ExternalTool.SC_CHAIN_GENERATION,
                required=True,
                env_var="ONEURO_MC4D_DNA_SOFTWARE_DIR",
                repo_subdir=Path("sc_chain_generation"),
                notes="Used to initialize chromosome configurations.",
            ),
            RuntimeDependency(
                tool=ExternalTool.FREEDTS,
                required=False,
                notes="Optional for membrane shape workflows in the published stack.",
            ),
        ),
        expected_repo_paths=(
            Path("README.md"),
            Path("Whole_Cell_Minimal_Cell.py"),
            Path("Restart_Whole_Cell_Minimal_Cell.py"),
            Path("Hook.py"),
            Path("MC_RDME_initialization.py"),
            Path("SpatialDnaDynamics.py"),
            Path("input_data"),
        ),
        contract=program.contract,
        provenance=WholeCellProvenance(
            source_dataset="JCVI-syn3A reference manifest",
            run_manifest_hash="syn3a_reference_manifest_v1",
            backend="external_reference",
            notes=(
                "Manifest-level cadence and dependency contract for the MC4D-compatible reference flow.",
            ),
        ),
    )
