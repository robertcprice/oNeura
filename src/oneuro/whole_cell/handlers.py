"""Default native stage handlers for the coarse Syn3A skeleton."""

from __future__ import annotations

from .architecture import CouplingStage
from .manifest import WholeCellRuntimeManifest, syn3a_reference_manifest
from .scheduler import WholeCellScheduler, WholeCellStageResult
from .state import WholeCellState, syn3a_minimal_state


def rdme_stage_handler(
    state: WholeCellState,
    stage_dt_ms: float,
    stage: CouplingStage,
) -> WholeCellStageResult:
    """Very coarse intracellular diffusion and crowding placeholder."""

    atp_leak = -1e-6 * stage_dt_ms
    return WholeCellStageResult(
        stage=stage,
        advanced_ms=stage_dt_ms,
        metabolite_deltas={"ATP": atp_leak},
        notes="coarse RDME placeholder",
    )


def cme_stage_handler(
    state: WholeCellState,
    stage_dt_ms: float,
    stage: CouplingStage,
) -> WholeCellStageResult:
    """Coarse transcription/translation bookkeeping placeholder."""

    gtp = max(0.0, state.metabolites_mM.get("GTP", 0.0))
    utp = max(0.0, state.metabolites_mM.get("UTP", 0.0))
    transcription_scale = min(1.0, (gtp + utp) / 0.6)
    transcript_gain = 0.01 * transcription_scale
    protein_gain = 0.005 * transcription_scale
    return WholeCellStageResult(
        stage=stage,
        advanced_ms=stage_dt_ms,
        transcript_deltas={
            "dnaA": transcript_gain,
            "ftsZ": transcript_gain * 1.2,
        },
        protein_deltas={
            "DnaA": protein_gain,
            "FtsZ": protein_gain * 1.4,
        },
        metabolite_deltas={
            "GTP": -0.001 * transcription_scale,
            "UTP": -0.001 * transcription_scale,
            "ATP": -0.0005 * transcription_scale,
        },
        notes="coarse CME placeholder",
    )


def ode_stage_handler(
    state: WholeCellState,
    stage_dt_ms: float,
    stage: CouplingStage,
) -> WholeCellStageResult:
    """Coarse metabolism placeholder."""

    glucose = max(0.0, state.metabolites_mM.get("glucose", 0.0))
    atp = max(0.0, state.metabolites_mM.get("ATP", 0.0))
    glucose_flux = min(glucose, 0.0025 * (stage_dt_ms / 12.5))
    atp_gain = glucose_flux * 0.25
    homeostatic_bonus = 0.02 if atp < 1.5 else 0.0
    return WholeCellStageResult(
        stage=stage,
        advanced_ms=stage_dt_ms,
        metabolite_deltas={
            "glucose": -glucose_flux,
            "ATP": atp_gain + homeostatic_bonus,
            "dATP": 0.001,
            "dTTP": 0.001,
            "dCTP": 0.001,
            "dGTP": 0.001,
        },
        notes="coarse ODE placeholder",
    )


def bd_stage_handler(
    state: WholeCellState,
    stage_dt_ms: float,
    stage: CouplingStage,
) -> WholeCellStageResult:
    """Coarse chromosome replication and segregation placeholder."""

    replicated_bp = state.chromosome.replicated_bp
    genome_bp = state.chromosome.genome_bp
    dntp_pool = min(
        state.metabolites_mM.get("dATP", 0.0),
        state.metabolites_mM.get("dTTP", 0.0),
        state.metabolites_mM.get("dCTP", 0.0),
        state.metabolites_mM.get("dGTP", 0.0),
    )
    replication_increment = int(250 * max(0.25, dntp_pool / 0.1))
    new_replicated_bp = min(genome_bp, replicated_bp + replication_increment)
    segregated_fraction = state.chromosome.segregated_fraction
    if new_replicated_bp >= genome_bp:
        segregated_fraction = min(1.0, segregated_fraction + 0.01)
    return WholeCellStageResult(
        stage=stage,
        advanced_ms=stage_dt_ms,
        chromosome_updates={
            "replicated_bp": new_replicated_bp,
            "segregated_fraction": segregated_fraction,
            "origin_count": 2 if new_replicated_bp > 0 else 1,
            "chromosome_count": 2 if segregated_fraction >= 1.0 else 1,
        },
        metabolite_deltas={
            "dATP": -0.0005,
            "dTTP": -0.0005,
            "dCTP": -0.0005,
            "dGTP": -0.0005,
            "ATP": -0.0005,
        },
        notes="coarse BD placeholder",
    )


def geometry_stage_handler(
    state: WholeCellState,
    stage_dt_ms: float,
    stage: CouplingStage,
) -> WholeCellStageResult:
    """Coarse membrane growth and division placeholder."""

    membrane_protein = state.proteins.get("FtsZ", 0.0)
    growth_nm2 = 20.0 + membrane_protein * 0.01
    new_surface_area = state.geometry.surface_area_nm2 + growth_nm2
    new_volume = state.geometry.volume_nm3 + 500.0
    division_progress = state.geometry.division_progress
    if state.chromosome.replicated_fraction >= 1.0:
        division_progress = min(1.0, division_progress + 0.005)
    morphology = "dividing" if division_progress > 0.0 else state.geometry.morphology
    return WholeCellStageResult(
        stage=stage,
        advanced_ms=stage_dt_ms,
        compartment_deltas={
            "membrane": {
                "lipid": 10.0,
                "membrane_protein": 0.5,
            }
        },
        geometry_updates={
            "surface_area_nm2": new_surface_area,
            "volume_nm3": new_volume,
            "division_progress": division_progress,
            "morphology": morphology,
        },
        notes="coarse geometry placeholder",
    )


def build_syn3a_skeleton_scheduler(
    state: WholeCellState | None = None,
    manifest: WholeCellRuntimeManifest | None = None,
) -> WholeCellScheduler:
    """Create a scheduler preloaded with coarse native Syn3A handlers."""

    resolved_state = state or syn3a_minimal_state()
    resolved_manifest = manifest or syn3a_reference_manifest()
    scheduler = WholeCellScheduler.from_manifest(resolved_state, resolved_manifest)
    scheduler.register_handler(CouplingStage.RDME, rdme_stage_handler)
    scheduler.register_handler(CouplingStage.CME, cme_stage_handler)
    scheduler.register_handler(CouplingStage.ODE, ode_stage_handler)
    scheduler.register_handler(CouplingStage.BD, bd_stage_handler)
    scheduler.register_handler(CouplingStage.GEOMETRY, geometry_stage_handler)
    return scheduler
