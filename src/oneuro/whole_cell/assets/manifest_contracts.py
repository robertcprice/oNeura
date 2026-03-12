"""Manifest contract validation helpers for whole-cell bundle compilation."""

from __future__ import annotations

from typing import Any, Dict


def _validate_explicit_asset_contracts(
    manifest: Dict[str, Any],
    source_hashes: Dict[str, str],
) -> None:
    allow_legacy_derived_assets = bool(manifest.get("allow_legacy_derived_assets"))
    require_explicit_asset_entities = bool(
        manifest.get("require_explicit_asset_entities")
    )
    require_explicit_asset_semantics = bool(
        manifest.get("require_explicit_asset_semantics")
    )

    if allow_legacy_derived_assets and (
        require_explicit_asset_entities or require_explicit_asset_semantics
    ):
        raise ValueError(
            "allow_legacy_derived_assets is incompatible with explicit asset entity or semantic requirements"
        )
    if not require_explicit_asset_entities and not allow_legacy_derived_assets:
        raise ValueError(
            "bundle must declare explicit asset entities or set allow_legacy_derived_assets"
        )
    if not require_explicit_asset_semantics and not allow_legacy_derived_assets:
        raise ValueError(
            "bundle must declare explicit asset semantics or set allow_legacy_derived_assets"
        )
    if require_explicit_asset_entities:
        required_entity_keys = {
            "operons_json",
            "rnas_json",
            "proteins_json",
            "complexes_json",
        }
        missing = sorted(key for key in required_entity_keys if key not in source_hashes)
        if missing:
            raise ValueError(
                "bundle requires explicit asset entities but is missing "
                + ", ".join(missing)
            )
    if require_explicit_asset_semantics:
        required_semantic_keys = {
            "operon_semantics_json",
            "protein_semantics_json",
            "complex_semantics_json",
        }
        missing = sorted(key for key in required_semantic_keys if key not in source_hashes)
        if missing:
            raise ValueError(
                "bundle requires explicit asset semantics but is missing "
                + ", ".join(missing)
            )
    if manifest.get("require_explicit_program_defaults"):
        if "program_defaults_json" not in source_hashes:
            raise ValueError(
                "bundle requires explicit program defaults but is missing "
                "program_defaults_json"
            )


def _validate_bundle_compile_entrypoint(
    manifest: Dict[str, Any], *, allow_legacy_derived_assets: bool
) -> None:
    manifest_allows_legacy_derived_assets = bool(
        manifest.get("allow_legacy_derived_assets")
    )
    if manifest_allows_legacy_derived_assets and not allow_legacy_derived_assets:
        raise ValueError(
            "legacy-derived-asset bundles must use compile_legacy_bundle_manifest"
        )
    if allow_legacy_derived_assets and not manifest_allows_legacy_derived_assets:
        raise ValueError(
            "compile_legacy_bundle_manifest requires allow_legacy_derived_assets in the manifest"
        )
