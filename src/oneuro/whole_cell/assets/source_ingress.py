"""Raw manifest and source-ingress helpers for whole-cell bundle compilation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from .source_normalization import (
    _merge_gene_annotation,
    _merge_transcription_unit_semantics,
    _validate_explicit_gene_semantics,
    _validate_explicit_pool_metadata,
    _validate_explicit_transcription_unit_semantics,
)


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
