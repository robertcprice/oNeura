"""Native scheduler skeleton for staged whole-cell updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from .architecture import CouplingStage
from .manifest import WholeCellRuntimeManifest
from .state import WholeCellState


@dataclass
class WholeCellStageResult:
    """Output from one scheduled stage update."""

    stage: CouplingStage
    advanced_ms: float
    compartment_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metabolite_deltas: Dict[str, float] = field(default_factory=dict)
    protein_deltas: Dict[str, float] = field(default_factory=dict)
    transcript_deltas: Dict[str, float] = field(default_factory=dict)
    chromosome_updates: Dict[str, object] = field(default_factory=dict)
    geometry_updates: Dict[str, object] = field(default_factory=dict)
    notes: str = ""

    def apply(self, state: WholeCellState) -> None:
        """Apply this stage result to a state."""

        state.apply_deltas(
            compartment_deltas=self.compartment_deltas,
            metabolite_deltas=self.metabolite_deltas,
            protein_deltas=self.protein_deltas,
            transcript_deltas=self.transcript_deltas,
            chromosome_updates=self.chromosome_updates,
            geometry_updates=self.geometry_updates,
        )
        state.stage_history.append(
            {
                "time_ms": state.time_ms,
                "stage": self.stage.value,
                "advanced_ms": self.advanced_ms,
                "notes": self.notes,
            }
        )


StageHandler = Callable[[WholeCellState, float, CouplingStage], Optional[WholeCellStageResult]]


class WholeCellScheduler:
    """Coupled native scheduler for RDME/CME/ODE/BD/geometry stage callbacks."""

    def __init__(
        self,
        state: WholeCellState,
        stage_order: Iterable[CouplingStage],
        stage_intervals_ms: Dict[CouplingStage, float],
    ) -> None:
        self.state = state
        self.stage_order = tuple(stage_order)
        self.stage_intervals_ms = dict(stage_intervals_ms)
        self.handlers: Dict[CouplingStage, StageHandler] = {}
        self._accumulators_ms: Dict[CouplingStage, float] = {
            stage: 0.0 for stage in self.stage_order
        }

    @classmethod
    def from_manifest(
        cls,
        state: WholeCellState,
        manifest: WholeCellRuntimeManifest,
    ) -> "WholeCellScheduler":
        """Build a scheduler from the runtime manifest cadence."""

        stage_intervals_ms = {
            cadence.stage: cadence.interval_bio_ms
            for cadence in manifest.cadences
        }
        return cls(
            state=state,
            stage_order=manifest.program.coupling_stages,
            stage_intervals_ms=stage_intervals_ms,
        )

    def register_handler(self, stage: CouplingStage, handler: StageHandler) -> None:
        """Register one handler for a stage."""

        self.handlers[stage] = handler

    def step(self, dt_ms: float) -> List[WholeCellStageResult]:
        """Advance time and execute any stages whose cadence is due."""

        self.state.advance_time(dt_ms)
        results: List[WholeCellStageResult] = []

        for stage in self.stage_order:
            interval = self.stage_intervals_ms.get(stage)
            if interval is None or interval <= 0.0:
                continue
            self._accumulators_ms[stage] += dt_ms
            while self._accumulators_ms[stage] >= interval:
                self._accumulators_ms[stage] -= interval
                handler = self.handlers.get(stage)
                if handler is None:
                    results.append(
                        WholeCellStageResult(
                            stage=stage,
                            advanced_ms=interval,
                            notes="no handler registered",
                        )
                    )
                    continue
                result = handler(self.state, interval, stage)
                if result is None:
                    continue
                result.apply(self.state)
                results.append(result)
        return results

    def run_for(self, duration_ms: float, dt_ms: float) -> List[WholeCellStageResult]:
        """Run a fixed-duration simulation loop."""

        results: List[WholeCellStageResult] = []
        elapsed = 0.0
        while elapsed < duration_ms:
            step_dt = min(dt_ms, duration_ms - elapsed)
            results.extend(self.step(step_dt))
            elapsed += step_dt
        return results
