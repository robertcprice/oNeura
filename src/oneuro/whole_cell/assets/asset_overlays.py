"""Asset overlay and explicit asset contract helpers for whole-cell bundles."""

from __future__ import annotations

from typing import Any, Dict

from .derived_assets import (
    _derive_complex_semantics,
    _derive_operon_semantics,
    _derive_protein_semantics,
)


def _apply_asset_semantic_overlays(
    asset_package: Dict[str, Any],
    operon_semantics_overlay: list[Dict[str, Any]],
    protein_semantics_overlay: list[Dict[str, Any]],
    complex_semantics_overlay: list[Dict[str, Any]],
) -> Dict[str, Any]:
    compiled = dict(asset_package)
    operons = [dict(operon) for operon in compiled.get("operons", [])]
    proteins = [dict(protein) for protein in compiled.get("proteins", [])]
    complexes = [dict(complex_spec) for complex_spec in compiled.get("complexes", [])]

    operon_semantics = {
        semantic["name"]: dict(semantic)
        for semantic in compiled.get("operon_semantics", [])
    }
    for semantic in operon_semantics_overlay:
        merged = operon_semantics.setdefault(semantic["name"], {"name": semantic["name"]})
        if semantic.get("asset_class"):
            merged["asset_class"] = semantic["asset_class"]
        if semantic.get("complex_family"):
            merged["complex_family"] = semantic["complex_family"]
        merged_targets = list(merged.get("subsystem_targets", []))
        for target in semantic.get("subsystem_targets", []):
            if target not in merged_targets:
                merged_targets.append(target)
        merged["subsystem_targets"] = merged_targets
    for operon in operons:
        semantic = operon_semantics.get(operon["name"])
        if semantic is None:
            continue
        operon["asset_class"] = semantic.get("asset_class", operon.get("asset_class"))
        operon["complex_family"] = semantic.get(
            "complex_family", operon.get("complex_family")
        )
        targets = list(operon.get("subsystem_targets", []))
        for target in semantic.get("subsystem_targets", []):
            if target not in targets:
                targets.append(target)
        operon["subsystem_targets"] = targets

    protein_semantics = {
        semantic["id"]: dict(semantic)
        for semantic in compiled.get("protein_semantics", [])
    }
    for semantic in protein_semantics_overlay:
        merged = protein_semantics.setdefault(semantic["id"], {"id": semantic["id"]})
        if semantic.get("asset_class"):
            merged["asset_class"] = semantic["asset_class"]
        merged_targets = list(merged.get("subsystem_targets", []))
        for target in semantic.get("subsystem_targets", []):
            if target not in merged_targets:
                merged_targets.append(target)
        merged["subsystem_targets"] = merged_targets
    for protein in proteins:
        semantic = protein_semantics.get(protein["id"])
        if semantic is None:
            continue
        protein["asset_class"] = semantic.get("asset_class", protein.get("asset_class"))
        targets = list(protein.get("subsystem_targets", []))
        for target in semantic.get("subsystem_targets", []):
            if target not in targets:
                targets.append(target)
        protein["subsystem_targets"] = targets

    complex_semantics = {
        semantic["id"]: dict(semantic)
        for semantic in compiled.get("complex_semantics", [])
    }
    for semantic in complex_semantics_overlay:
        merged = complex_semantics.setdefault(semantic["id"], {"id": semantic["id"]})
        if semantic.get("asset_class"):
            merged["asset_class"] = semantic["asset_class"]
        if semantic.get("family"):
            merged["family"] = semantic["family"]
        for key in ("membrane_inserted", "chromosome_coupled", "division_coupled"):
            if key in semantic:
                merged[key] = bool(semantic[key])
        merged_targets = list(merged.get("subsystem_targets", []))
        for target in semantic.get("subsystem_targets", []):
            if target not in merged_targets:
                merged_targets.append(target)
        merged["subsystem_targets"] = merged_targets
    for complex_spec in complexes:
        semantic = complex_semantics.get(complex_spec["id"])
        if semantic is None:
            continue
        complex_spec["asset_class"] = semantic.get(
            "asset_class", complex_spec.get("asset_class")
        )
        complex_spec["family"] = semantic.get("family", complex_spec.get("family"))
        for key in ("membrane_inserted", "chromosome_coupled", "division_coupled"):
            if key in semantic:
                complex_spec[key] = bool(semantic[key])
        targets = list(complex_spec.get("subsystem_targets", []))
        for target in semantic.get("subsystem_targets", []):
            if target not in targets:
                targets.append(target)
        complex_spec["subsystem_targets"] = targets

    compiled["operons"] = operons
    compiled["proteins"] = proteins
    compiled["complexes"] = complexes
    compiled["operon_semantics"] = sorted(
        operon_semantics.values(), key=lambda semantic: semantic["name"]
    )
    compiled["protein_semantics"] = sorted(
        protein_semantics.values(), key=lambda semantic: semantic["id"]
    )
    compiled["complex_semantics"] = sorted(
        complex_semantics.values(), key=lambda semantic: semantic["id"]
    )
    return compiled


def _apply_asset_entity_overlays(
    asset_package: Dict[str, Any],
    operons_overlay: list[Dict[str, Any]],
    rnas_overlay: list[Dict[str, Any]],
    proteins_overlay: list[Dict[str, Any]],
    complexes_overlay: list[Dict[str, Any]],
    *,
    derive_semantics: bool = True,
) -> Dict[str, Any]:
    compiled = dict(asset_package)
    entity_overrides = False
    if operons_overlay:
        compiled["operons"] = [dict(operon) for operon in operons_overlay]
        entity_overrides = True
    if rnas_overlay:
        compiled["rnas"] = [dict(rna) for rna in rnas_overlay]
        entity_overrides = True
    if proteins_overlay:
        compiled["proteins"] = [dict(protein) for protein in proteins_overlay]
        entity_overrides = True
    if complexes_overlay:
        compiled["complexes"] = [dict(complex_spec) for complex_spec in complexes_overlay]
        entity_overrides = True
    if entity_overrides:
        if derive_semantics:
            compiled["operon_semantics"] = _derive_operon_semantics(compiled)
            compiled["protein_semantics"] = _derive_protein_semantics(compiled)
            compiled["complex_semantics"] = _derive_complex_semantics(compiled)
        else:
            compiled["operon_semantics"] = []
            compiled["protein_semantics"] = []
            compiled["complex_semantics"] = []
    return compiled


def _validate_explicit_asset_entities(asset_package: Dict[str, Any]) -> None:
    operons = asset_package.get("operons", [])
    proteins = asset_package.get("proteins", [])
    complexes = asset_package.get("complexes", [])

    missing_operons = [
        operon["name"]
        for operon in operons
        if not operon.get("asset_class")
        or not operon.get("complex_family")
        or not operon.get("subsystem_targets")
    ]
    if missing_operons:
        raise ValueError(
            "bundle requires explicit asset entities but "
            f"{len(missing_operons)} operon(s) are incomplete: "
            + ", ".join(missing_operons)
        )

    missing_proteins = [
        protein["id"]
        for protein in proteins
        if not protein.get("asset_class") or not protein.get("subsystem_targets")
    ]
    if missing_proteins:
        raise ValueError(
            "bundle requires explicit asset entities but "
            f"{len(missing_proteins)} protein(s) are incomplete: "
            + ", ".join(missing_proteins)
        )

    missing_complexes = [
        complex_spec["id"]
        for complex_spec in complexes
        if not complex_spec.get("asset_class")
        or not complex_spec.get("family")
        or not complex_spec.get("subsystem_targets")
    ]
    if missing_complexes:
        raise ValueError(
            "bundle requires explicit asset entities but "
            f"{len(missing_complexes)} complex(es) are incomplete: "
            + ", ".join(missing_complexes)
        )


def _validate_explicit_asset_semantics(asset_package: Dict[str, Any]) -> None:
    operon_semantics = {
        semantic["name"]: semantic for semantic in asset_package.get("operon_semantics", [])
    }
    missing_operon_semantics = [
        operon["name"]
        for operon in asset_package.get("operons", [])
        if operon["name"] not in operon_semantics
        or not operon_semantics[operon["name"]].get("subsystem_targets")
    ]
    if missing_operon_semantics:
        raise ValueError(
            "bundle requires explicit asset semantics but "
            f"{len(missing_operon_semantics)} operon semantic entry(s) are incomplete: "
            + ", ".join(missing_operon_semantics)
        )

    protein_semantics = {
        semantic["id"]: semantic for semantic in asset_package.get("protein_semantics", [])
    }
    missing_protein_semantics = [
        protein["id"]
        for protein in asset_package.get("proteins", [])
        if protein["id"] not in protein_semantics
        or not protein_semantics[protein["id"]].get("subsystem_targets")
    ]
    if missing_protein_semantics:
        raise ValueError(
            "bundle requires explicit asset semantics but "
            f"{len(missing_protein_semantics)} protein semantic entry(s) are incomplete: "
            + ", ".join(missing_protein_semantics)
        )

    complex_semantics = {
        semantic["id"]: semantic for semantic in asset_package.get("complex_semantics", [])
    }
    missing_complex_semantics = [
        complex_spec["id"]
        for complex_spec in asset_package.get("complexes", [])
        if complex_spec["id"] not in complex_semantics
        or not complex_semantics[complex_spec["id"]].get("subsystem_targets")
    ]
    if missing_complex_semantics:
        raise ValueError(
            "bundle requires explicit asset semantics but "
            f"{len(missing_complex_semantics)} complex semantic entry(s) are incomplete: "
            + ", ".join(missing_complex_semantics)
        )


def _validate_explicit_asset_entity_coverage(
    organism_spec: Dict[str, Any], asset_package: Dict[str, Any]
) -> None:
    genes = list(organism_spec.get("genes", []))
    transcription_units = list(organism_spec.get("transcription_units", []))
    operons = list(asset_package.get("operons", []))
    rnas = list(asset_package.get("rnas", []))
    proteins = list(asset_package.get("proteins", []))
    complexes = list(asset_package.get("complexes", []))

    genes_in_units = {
        gene_name
        for unit in transcription_units
        for gene_name in unit.get("genes", [])
    }
    expected_operons = {
        unit["name"] for unit in transcription_units
    } | {gene["gene"] for gene in genes if gene["gene"] not in genes_in_units}
    operon_names = {operon["name"] for operon in operons}
    missing_operons = sorted(expected_operons - operon_names)
    if missing_operons:
        raise ValueError(
            "bundle requires explicit asset entity coverage but "
            f"{len(missing_operons)} operon(s) are missing: "
            + ", ".join(missing_operons)
        )

    rna_genes = {rna["gene"] for rna in rnas}
    missing_rnas = sorted(gene["gene"] for gene in genes if gene["gene"] not in rna_genes)
    if missing_rnas:
        raise ValueError(
            "bundle requires explicit asset entity coverage but "
            f"{len(missing_rnas)} RNA gene(s) are missing: "
            + ", ".join(missing_rnas)
        )

    protein_genes = {protein["gene"] for protein in proteins}
    missing_proteins = sorted(
        gene["gene"] for gene in genes if gene["gene"] not in protein_genes
    )
    if missing_proteins:
        raise ValueError(
            "bundle requires explicit asset entity coverage but "
            f"{len(missing_proteins)} protein gene(s) are missing: "
            + ", ".join(missing_proteins)
        )

    complex_operons = {complex_spec["operon"] for complex_spec in complexes}
    missing_complexes = sorted(expected_operons - complex_operons)
    if missing_complexes:
        raise ValueError(
            "bundle requires explicit asset entity coverage but "
            f"{len(missing_complexes)} complex operon(s) are missing: "
            + ", ".join(missing_complexes)
        )


def _empty_genome_asset_package(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "organism": spec["organism"],
        "chromosome_length_bp": int(spec["chromosome_length_bp"]),
        "origin_bp": int(spec["origin_bp"]),
        "terminus_bp": int(spec["terminus_bp"]),
        "chromosome_domains": list(spec.get("chromosome_domains", [])),
        "operons": [],
        "operon_semantics": [],
        "rnas": [],
        "proteins": [],
        "protein_semantics": [],
        "complex_semantics": [],
        "complexes": [],
        "pools": list(spec.get("pools", [])),
    }
