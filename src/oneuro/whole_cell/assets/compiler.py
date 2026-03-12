"""Manifest-driven orchestration for whole-cell organism source bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from .asset_overlays import (
    _apply_asset_entity_overlays,
    _apply_asset_semantic_overlays,
    _empty_genome_asset_package,
    _validate_explicit_asset_entities,
    _validate_explicit_asset_entity_coverage,
    _validate_explicit_asset_semantics,
)
from .derived_assets import (
    _compile_genome_asset_package,
)
from .source_ingress import _load_bundle_manifest_inputs
from .source_normalization import _with_compiled_chromosome_domains

_BUNDLES_ROOT = Path(__file__).resolve().parent / "bundles"


@dataclass(frozen=True)
class CompiledOrganismBundle:
    """Compiled organism spec plus derived runtime asset package."""

    manifest_path: str
    organism: str
    organism_spec: Dict[str, Any]
    genome_asset_package: Dict[str, Any]
    source_hashes: Dict[str, str]

    def summary(self) -> Dict[str, Any]:
        operons = self.genome_asset_package.get("operons", [])
        complexes = self.genome_asset_package.get("complexes", [])
        return {
            "organism": self.organism,
            "gene_count": len(self.organism_spec.get("genes", [])),
            "transcription_unit_count": len(
                self.organism_spec.get("transcription_units", [])
            ),
            "operon_count": len(operons),
            "polycistronic_operon_count": sum(
                1 for operon in operons if operon.get("polycistronic")
            ),
            "rna_count": len(self.genome_asset_package.get("rnas", [])),
            "protein_count": len(self.genome_asset_package.get("proteins", [])),
            "complex_count": len(complexes),
            "targeted_complex_count": sum(
                1
                for complex_spec in complexes
                if complex_spec.get("subsystem_targets")
            ),
        }


def available_bundles() -> tuple[str, ...]:
    """Return bundle names shipped with the package."""

    if not _BUNDLES_ROOT.exists():
        return ()
    return tuple(
        sorted(
            path.name
            for path in _BUNDLES_ROOT.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        )
    )


def compile_named_bundle(name: str) -> CompiledOrganismBundle:
    """Compile one of the packaged organism bundles by name."""

    manifest_path = (_BUNDLES_ROOT / name / "manifest.json").resolve()
    return compile_bundle_manifest(manifest_path)


def compile_legacy_named_bundle(name: str) -> CompiledOrganismBundle:
    """Compile a packaged organism bundle through the legacy derived-asset path."""

    manifest_path = (_BUNDLES_ROOT / name / "manifest.json").resolve()
    return compile_legacy_bundle_manifest(manifest_path)


def compile_bundle_manifest(manifest_path: Path | str) -> CompiledOrganismBundle:
    """Compile an organism bundle manifest into runtime-ready JSON payloads."""

    return _compile_explicit_bundle_manifest(manifest_path)


def compile_legacy_bundle_manifest(manifest_path: Path | str) -> CompiledOrganismBundle:
    """Compile an organism bundle manifest through the legacy derived-asset path."""

    return _compile_legacy_bundle_manifest(manifest_path)


def _compile_explicit_bundle_manifest(
    manifest_path: Path | str,
) -> CompiledOrganismBundle:
    """Compile an organism bundle manifest through the explicit structured path."""

    (
        path,
        manifest,
        source_hashes,
        organism_spec,
        operons_overlay,
        rnas_overlay,
        proteins_overlay,
        complexes_overlay,
        operon_semantics_overlay,
        protein_semantics_overlay,
        complex_semantics_overlay,
    ) = _load_bundle_manifest_inputs(manifest_path)
    _validate_bundle_compile_entrypoint(
        manifest, allow_legacy_derived_assets=False
    )
    _validate_explicit_asset_contracts(manifest, source_hashes)
    require_explicit_asset_entities = bool(manifest.get("require_explicit_asset_entities"))
    require_explicit_asset_semantics = bool(
        manifest.get("require_explicit_asset_semantics")
    )

    organism_spec = _with_compiled_chromosome_domains(organism_spec)
    asset_package = _empty_genome_asset_package(organism_spec)
    asset_package = _apply_asset_entity_overlays(
        asset_package,
        operons_overlay,
        rnas_overlay,
        proteins_overlay,
        complexes_overlay,
        derive_semantics=not require_explicit_asset_semantics,
    )
    if require_explicit_asset_entities:
        _validate_explicit_asset_entities(asset_package)
        _validate_explicit_asset_entity_coverage(organism_spec, asset_package)
    asset_package = _apply_asset_semantic_overlays(
        asset_package,
        operon_semantics_overlay,
        protein_semantics_overlay,
        complex_semantics_overlay,
    )
    if require_explicit_asset_semantics:
        _validate_explicit_asset_semantics(asset_package)
    return CompiledOrganismBundle(
        manifest_path=str(path),
        organism=organism_spec["organism"],
        organism_spec=organism_spec,
        genome_asset_package=asset_package,
        source_hashes=source_hashes,
    )


def _compile_legacy_bundle_manifest(
    manifest_path: Path | str,
) -> CompiledOrganismBundle:
    """Compile an organism bundle manifest through the legacy derived-asset path."""

    (
        path,
        manifest,
        source_hashes,
        organism_spec,
        operons_overlay,
        rnas_overlay,
        proteins_overlay,
        complexes_overlay,
        operon_semantics_overlay,
        protein_semantics_overlay,
        complex_semantics_overlay,
    ) = _load_bundle_manifest_inputs(manifest_path)
    _validate_bundle_compile_entrypoint(
        manifest, allow_legacy_derived_assets=True
    )

    organism_spec = _with_compiled_chromosome_domains(organism_spec)
    asset_package = _compile_genome_asset_package(organism_spec)
    asset_package = _apply_asset_entity_overlays(
        asset_package,
        operons_overlay,
        rnas_overlay,
        proteins_overlay,
        complexes_overlay,
        derive_semantics=True,
    )
    asset_package = _apply_asset_semantic_overlays(
        asset_package,
        operon_semantics_overlay,
        protein_semantics_overlay,
        complex_semantics_overlay,
    )
    return CompiledOrganismBundle(
        manifest_path=str(path),
        organism=organism_spec["organism"],
        organism_spec=organism_spec,
        genome_asset_package=asset_package,
        source_hashes=source_hashes,
    )


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
