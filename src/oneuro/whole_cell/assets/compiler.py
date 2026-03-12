"""Manifest-driven compiler for whole-cell organism source bundles."""

from __future__ import annotations

import hashlib
import json
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
from .source_normalization import (
    _merge_gene_annotation,
    _merge_transcription_unit_semantics,
    _validate_explicit_gene_semantics,
    _validate_explicit_pool_metadata,
    _validate_explicit_transcription_unit_semantics,
    _with_compiled_chromosome_domains,
)

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


def _load_bundle_manifest_inputs(
    manifest_path: Path | str,
) -> tuple[
    Path,
    Dict[str, Any],
    Dict[str, str],
    Dict[str, Any],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_json(path)
    source_hashes: Dict[str, str] = {"manifest.json": _sha256_path(path)}
    _validate_manifest_mode(manifest)
    organism_spec = _compile_structured_bundle(path, manifest, source_hashes)
    operons_overlay = _load_optional_json(path, manifest, "operons_json", source_hashes) or []
    rnas_overlay = _load_optional_json(path, manifest, "rnas_json", source_hashes) or []
    proteins_overlay = (
        _load_optional_json(path, manifest, "proteins_json", source_hashes) or []
    )
    complexes_overlay = (
        _load_optional_json(path, manifest, "complexes_json", source_hashes) or []
    )
    operon_semantics_overlay = (
        _load_optional_json(path, manifest, "operon_semantics_json", source_hashes) or []
    )
    protein_semantics_overlay = (
        _load_optional_json(path, manifest, "protein_semantics_json", source_hashes) or []
    )
    complex_semantics_overlay = (
        _load_optional_json(path, manifest, "complex_semantics_json", source_hashes) or []
    )
    _load_optional_json(path, manifest, "program_defaults_json", source_hashes)
    return (
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
    )


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


def _validate_manifest_mode(manifest: Dict[str, Any]) -> None:
    if "organism_spec_json" in manifest:
        raise ValueError(
            "bundle manifests may not define organism_spec_json; "
            "use explicit structured bundle sources"
        )
    if manifest.get("require_explicit_organism_sources"):
        missing = []
        if "metadata_json" not in manifest:
            missing.append("metadata_json")
        if "gene_features_json" not in manifest and "gene_features_gff" not in manifest:
            missing.append("gene_features_json|gene_features_gff")
        if "gene_products_json" not in manifest:
            missing.append("gene_products_json")
        if "transcription_units_json" not in manifest:
            missing.append("transcription_units_json")
        if "chromosome_domains_json" not in manifest:
            missing.append("chromosome_domains_json")
        if "pools_json" not in manifest:
            missing.append("pools_json")
        if missing:
            raise ValueError(
                "bundle requires explicit organism sources but is missing "
                + ", ".join(missing)
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


def _compile_structured_bundle(
    manifest_path: Path,
    manifest: Dict[str, Any],
    source_hashes: Dict[str, str],
) -> Dict[str, Any]:
    metadata = _load_optional_json(manifest_path, manifest, "metadata_json", source_hashes)
    pools = _load_optional_json(manifest_path, manifest, "pools_json", source_hashes) or []
    transcription_units = (
        _load_optional_json(
            manifest_path, manifest, "transcription_units_json", source_hashes
        )
        or []
    )
    transcription_unit_semantics = {
        entry["name"]: entry
        for entry in (
            _load_optional_json(
                manifest_path,
                manifest,
                "transcription_unit_semantics_json",
                source_hashes,
            )
            or []
        )
    }
    gene_products = {
        entry["gene"]: entry
        for entry in (
            _load_optional_json(
                manifest_path, manifest, "gene_products_json", source_hashes
            )
            or []
        )
    }
    gene_semantics = {
        entry["gene"]: entry
        for entry in (
            _load_optional_json(
                manifest_path, manifest, "gene_semantics_json", source_hashes
            )
            or []
        )
    }

    chromosome_length_bp = metadata.get("chromosome_length_bp")
    if "genome_fasta" in manifest:
        fasta_path = _resolve_manifest_path(manifest_path, manifest["genome_fasta"])
        source_hashes["genome_fasta"] = _sha256_path(fasta_path)
        fasta = _read_fasta(fasta_path)
        if chromosome_length_bp is None:
            chromosome_length_bp = len(fasta["sequence"])
    if chromosome_length_bp is None:
        raise ValueError("bundle metadata must define chromosome_length_bp or genome_fasta")

    if "gene_features_json" in manifest:
        genes = _load_optional_json(
            manifest_path, manifest, "gene_features_json", source_hashes
        ) or []
    elif "gene_features_gff" in manifest:
        gff_path = _resolve_manifest_path(manifest_path, manifest["gene_features_gff"])
        source_hashes["gene_features_gff"] = _sha256_path(gff_path)
        genes = _read_gff_features(gff_path)
    else:
        raise ValueError("bundle manifest must define gene_features_json or gene_features_gff")

    compiled_genes = [
        _merge_gene_annotation(
            gene,
            gene_products.get(gene["gene"], {}),
            gene_semantics.get(gene["gene"], {}),
        )
        for gene in genes
    ]
    compiled_transcription_units = [
        _merge_transcription_unit_semantics(
            unit,
            transcription_unit_semantics.get(unit["name"], {}),
        )
        for unit in transcription_units
    ]
    chromosome_domains = (
        _load_optional_json(
            manifest_path, manifest, "chromosome_domains_json", source_hashes
        )
        or []
    )
    if manifest.get("require_explicit_organism_sources"):
        _validate_explicit_pool_metadata(pools)
    if manifest.get("require_explicit_gene_semantics"):
        _validate_explicit_gene_semantics(compiled_genes)
    if manifest.get("require_explicit_transcription_unit_semantics"):
        _validate_explicit_transcription_unit_semantics(compiled_transcription_units)

    return {
        "organism": manifest.get("organism") or metadata["organism"],
        "chromosome_length_bp": int(chromosome_length_bp),
        "origin_bp": int(metadata.get("origin_bp", 0)),
        "terminus_bp": int(metadata.get("terminus_bp", chromosome_length_bp // 2)),
        "geometry": metadata["geometry"],
        "composition": metadata["composition"],
        "chromosome_domains": chromosome_domains,
        "pools": pools,
        "genes": compiled_genes,
        "transcription_units": compiled_transcription_units,
    }

def _read_fasta(path: Path) -> Dict[str, Any]:
    header = None
    sequence_parts: list[str] = []
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].strip()
            continue
        sequence_parts.append(line)
    sequence = "".join(sequence_parts).upper()
    if header is None or not sequence:
        raise ValueError(f"invalid FASTA file: {path}")
    return {"id": header, "sequence": sequence}


def _read_gff_features(path: Path) -> list[Dict[str, Any]]:
    genes: list[Dict[str, Any]] = []
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 9:
            raise ValueError(f"invalid GFF3 row in {path}: {line}")
        _seqid, _source, feature_type, start, end, _score, strand, _phase, attrs = parts
        if feature_type.lower() not in {"gene", "cds"}:
            continue
        attr_map = _parse_gff_attributes(attrs)
        gene_name = attr_map.get("gene") or attr_map.get("Name") or attr_map.get("ID")
        if not gene_name:
            raise ValueError(f"missing gene identifier in {path}: {line}")
        genes.append(
            {
                "gene": gene_name,
                "start_bp": int(start),
                "end_bp": int(end),
                "strand": -1 if strand == "-" else 1,
            }
        )
    return genes


def _parse_gff_attributes(raw_attributes: str) -> Dict[str, str]:
    attributes: Dict[str, str] = {}
    for entry in raw_attributes.split(";"):
        if not entry:
            continue
        if "=" in entry:
            key, value = entry.split("=", 1)
        elif " " in entry:
            key, value = entry.split(" ", 1)
        else:
            key, value = entry, ""
        attributes[key.strip()] = value.strip()
    return attributes


def _load_optional_json(
    manifest_path: Path,
    manifest: Dict[str, Any],
    key: str,
    source_hashes: Dict[str, str],
) -> Any:
    if key not in manifest:
        return None
    path = _resolve_manifest_path(manifest_path, manifest[key])
    if not path.exists():
        raise ValueError(f"bundle manifest references missing {key}: {path}")
    source_hashes[key] = _sha256_path(path)
    return _load_json(path)


def _resolve_manifest_path(manifest_path: Path, relative_path: str) -> Path:
    return (manifest_path.parent / relative_path).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
