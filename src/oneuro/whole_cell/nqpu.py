"""Optional nQPU bridge for the Rust whole-cell runtime.

The performance-critical simulator stays in Rust. This module only computes a
small set of quantum-chemistry correction factors using the existing nQPU
helper surfaces and then applies those factors to the native simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oneuro.molecular.backend import HAS_NQPU, quantum_enzyme_tunneling, quantum_protein_fold

# Proxy barriers for coarse whole-cell correction factors. These are not
# intended to replace a full ab initio workflow; they let the existing nQPU
# helpers modulate the Rust runtime without pulling Python into the hot loop.
_COMPLEX_I_BARRIER_EV = 0.30
_COMPLEX_III_BARRIER_EV = 0.25
_TRANSLATION_BARRIER_EV = 0.18
_POLYMERASE_BARRIER_EV = 0.22
_MEMBRANE_SYNTHESIS_BARRIER_EV = 0.16
_SEGREGATION_BARRIER_EV = 0.12


@dataclass(frozen=True)
class NQPUWholeCellProfile:
    """Quantum correction factors for the native whole-cell runtime."""

    oxphos_efficiency: float
    translation_efficiency: float
    nucleotide_polymerization_efficiency: float
    membrane_synthesis_efficiency: float
    chromosome_segregation_efficiency: float
    source: str


def _fold_bonus(sequence: str | None) -> float:
    if not sequence or not HAS_NQPU:
        return 1.0

    result = quantum_protein_fold(sequence)
    if not isinstance(result, dict):
        return 1.0

    confidence = float(result.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    energy = result.get("fold_energy", result.get("energy"))
    energy_bonus = 0.0
    if isinstance(energy, (int, float)) and energy < 0:
        energy_bonus = min(abs(float(energy)) / 200.0, 0.05)

    return min(1.15, 1.0 + 0.05 * confidence + energy_bonus)


def build_nqpu_whole_cell_profile(
    *,
    ftsz_sequence: str | None = None,
    dnaa_sequence: str | None = None,
    temperature_K: float = 310.0,
) -> NQPUWholeCellProfile:
    """Build a quantum correction profile from the existing nQPU helpers.

    This uses the same tunneling infrastructure already used elsewhere in the
    repo and packages the results into a handful of scalar multipliers suitable
    for the Rust whole-cell engine.
    """

    tunnel_i = quantum_enzyme_tunneling(_COMPLEX_I_BARRIER_EV, mass_amu=1.008, temperature_K=temperature_K)
    tunnel_iii = quantum_enzyme_tunneling(
        _COMPLEX_III_BARRIER_EV,
        mass_amu=1.008,
        temperature_K=temperature_K,
    )
    translation_tunnel = quantum_enzyme_tunneling(
        _TRANSLATION_BARRIER_EV,
        mass_amu=1.008,
        temperature_K=temperature_K,
    )
    polymerase_tunnel = quantum_enzyme_tunneling(
        _POLYMERASE_BARRIER_EV,
        mass_amu=1.008,
        temperature_K=temperature_K,
    )
    membrane_tunnel = quantum_enzyme_tunneling(
        _MEMBRANE_SYNTHESIS_BARRIER_EV,
        mass_amu=1.008,
        temperature_K=temperature_K,
    )
    segregation_tunnel = quantum_enzyme_tunneling(
        _SEGREGATION_BARRIER_EV,
        mass_amu=1.008,
        temperature_K=temperature_K,
    )

    ftsz_bonus = _fold_bonus(ftsz_sequence)
    dnaa_bonus = _fold_bonus(dnaa_sequence)

    return NQPUWholeCellProfile(
        oxphos_efficiency=1.0 + 0.15 * ((tunnel_i + tunnel_iii) * 0.5),
        translation_efficiency=(1.0 + 0.10 * translation_tunnel) * ftsz_bonus,
        nucleotide_polymerization_efficiency=(1.0 + 0.12 * polymerase_tunnel) * dnaa_bonus,
        membrane_synthesis_efficiency=1.0 + 0.08 * membrane_tunnel,
        chromosome_segregation_efficiency=1.0 + 0.06 * segregation_tunnel,
        source="nqpu" if HAS_NQPU else "wkb_fallback",
    )


def apply_nqpu_whole_cell_profile(
    simulator: Any,
    profile: NQPUWholeCellProfile | None = None,
    *,
    ftsz_sequence: str | None = None,
    dnaa_sequence: str | None = None,
    temperature_K: float = 310.0,
) -> NQPUWholeCellProfile:
    """Apply a computed nQPU profile to a simulator that exposes
    `set_quantum_profile(...)`.
    """

    if not hasattr(simulator, "set_quantum_profile"):
        raise TypeError("Simulator does not expose set_quantum_profile(...)")

    resolved = profile or build_nqpu_whole_cell_profile(
        ftsz_sequence=ftsz_sequence,
        dnaa_sequence=dnaa_sequence,
        temperature_K=temperature_K,
    )
    simulator.set_quantum_profile(
        oxphos_efficiency=resolved.oxphos_efficiency,
        translation_efficiency=resolved.translation_efficiency,
        nucleotide_polymerization_efficiency=resolved.nucleotide_polymerization_efficiency,
        membrane_synthesis_efficiency=resolved.membrane_synthesis_efficiency,
        chromosome_segregation_efficiency=resolved.chromosome_segregation_efficiency,
    )
    return resolved
