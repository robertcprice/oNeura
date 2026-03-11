"""Compiler-driven organism asset ingestion for the whole-cell runtime."""

from .compiler import (
    CompiledOrganismBundle,
    available_bundles,
    compile_bundle_manifest,
    compile_named_bundle,
    write_compiled_bundle,
)

__all__ = [
    "CompiledOrganismBundle",
    "available_bundles",
    "compile_bundle_manifest",
    "compile_named_bundle",
    "write_compiled_bundle",
]

