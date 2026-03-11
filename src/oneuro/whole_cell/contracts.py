"""Frozen contract metadata for whole-cell runtime and restart surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field

WHOLE_CELL_CONTRACT_VERSION = "whole_cell_phase0"
WHOLE_CELL_PROGRAM_SCHEMA_VERSION = 1
WHOLE_CELL_SAVED_STATE_SCHEMA_VERSION = 1
WHOLE_CELL_RUNTIME_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WholeCellContract:
    """Schema versions shared across whole-cell program and restart contracts."""

    contract_version: str = WHOLE_CELL_CONTRACT_VERSION
    program_schema_version: int = WHOLE_CELL_PROGRAM_SCHEMA_VERSION
    saved_state_schema_version: int = WHOLE_CELL_SAVED_STATE_SCHEMA_VERSION
    runtime_manifest_schema_version: int = WHOLE_CELL_RUNTIME_MANIFEST_SCHEMA_VERSION


@dataclass(frozen=True)
class WholeCellProvenance:
    """Minimal provenance surface for whole-cell program and restart payloads."""

    source_dataset: str | None = None
    organism_asset_hash: str | None = None
    compiled_ir_hash: str | None = None
    calibration_bundle_hash: str | None = None
    run_manifest_hash: str | None = None
    backend: str | None = None
    seed: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

