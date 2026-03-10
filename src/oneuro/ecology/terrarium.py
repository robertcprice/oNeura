"""Process-based terrarium ecology for MolecularWorld environments.

This module extends :class:`oneuro.worlds.molecular_world.MolecularWorld`
with a reusable ecology layer that remains honest about its abstraction level:

1. Soil biogeochemistry and hydrology:
   - surface and deep moisture redistribution
   - litter, rhizodeposition, decomposition, mineral release
   - microbial and fungal/symbiont biomass dynamics
   - fruit detritus feeding back into soil carbon and nutrients

2. Plant organisms:
   - inherited genotype-shaped traits
   - canopy competition for light
   - root competition for water and nutrients
   - rhizosphere coupling to soil symbionts
   - fruiting, seed production, dormancy, and recruitment

This is still a coarse process-based ecology model, not a cell-by-cell plant
simulator or granular soil mechanics engine. The goal is to provide a richer,
reusable substrate for terrarium demos without smuggling in fake behavioral
rules as "emergence."
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from oneuro.whole_cell.state import (
    CellCompartment,
    ChromosomeState,
    GeometryState,
    WholeCellState,
)
from oneuro.molecular.metabolism import CellularMetabolism
from oneuro.worlds.molecular_world import MolecularWorld, PlantSource

try:
    from oneuro_metal import BatchedAtomTerrarium as RustBatchedAtomTerrarium
    from oneuro_metal import build_dual_radial_fields as RustBuildDualRadialFields
    from oneuro_metal import build_radial_field as RustBuildRadialField
    from oneuro_metal import CellularMetabolism as RustCellularMetabolism
    from oneuro_metal import extract_root_resources_with_layers as RustExtractRootResourcesWithLayers
    from oneuro_metal import PlantCellularState as RustPlantCellularState
    from oneuro_metal import PlantOrganism as RustPlantOrganism
    from oneuro_metal import step_food_patches as RustStepFoodPatches
    from oneuro_metal import step_seed_bank as RustStepSeedBank
    from oneuro_metal import step_soil_broad_pools as RustStepSoilBroadPools
except ImportError:  # pragma: no cover - optional native extension
    RustBatchedAtomTerrarium = None
    RustBuildDualRadialFields = None
    RustBuildRadialField = None
    RustCellularMetabolism = None
    RustExtractRootResourcesWithLayers = None
    RustPlantCellularState = None
    RustPlantOrganism = None
    RustStepFoodPatches = None
    RustStepSeedBank = None
    RustStepSoilBroadPools = None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for weight, _value in pairs)
    if total_weight <= 1e-9:
        return 0.0
    return sum(weight * value for weight, value in pairs) / total_weight


def _temp_response(temp_c: float, optimum: float = 24.0, width: float = 10.0) -> float:
    delta = (temp_c - optimum) / max(width, 1e-6)
    return math.exp(-(delta * delta))


def _diffuse2d(field: NDArray, rate: float) -> None:
    field += (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    ) * rate


def _window_bounds(h: int, w: int, x: int, y: int, radius: int) -> tuple[int, int, int, int]:
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    return y0, y1, x0, x1


def _radial_kernel(
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    center_x: int,
    center_y: int,
    radius: int,
    *,
    sharpness: float = 0.65,
) -> NDArray:
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (yy - center_y) ** 2 + (xx - center_x) ** 2
    sigma = max(radius * sharpness, 0.75)
    kernel = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
    total = float(kernel.sum())
    if total > 1e-9:
        kernel /= total
    return kernel


@dataclass(frozen=True)
class PlantGenome:
    """Low-dimensional inherited plant trait set."""

    max_height_mm: float
    canopy_radius_mm: float
    root_radius_mm: float
    leaf_efficiency: float
    root_uptake_efficiency: float
    water_use_efficiency: float
    volatile_scale: float
    fruiting_threshold: float
    litter_turnover: float
    shade_tolerance: float
    root_depth_bias: float
    symbiosis_affinity: float
    seed_mass: float

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "PlantGenome":
        return cls(
            max_height_mm=float(rng.uniform(7.0, 18.0)),
            canopy_radius_mm=float(rng.uniform(3.0, 8.0)),
            root_radius_mm=float(rng.uniform(2.0, 5.5)),
            leaf_efficiency=float(rng.uniform(0.7, 1.4)),
            root_uptake_efficiency=float(rng.uniform(0.6, 1.3)),
            water_use_efficiency=float(rng.uniform(0.6, 1.25)),
            volatile_scale=float(rng.uniform(0.6, 1.5)),
            fruiting_threshold=float(rng.uniform(0.45, 1.2)),
            litter_turnover=float(rng.uniform(0.7, 1.4)),
            shade_tolerance=float(rng.uniform(0.55, 1.4)),
            root_depth_bias=float(rng.uniform(0.15, 0.95)),
            symbiosis_affinity=float(rng.uniform(0.6, 1.5)),
            seed_mass=float(rng.uniform(0.045, 0.16)),
        )

    def mutate(self, rng: np.random.Generator) -> "PlantGenome":
        return PlantGenome(
            max_height_mm=float(_clamp(self.max_height_mm + rng.normal(0.0, 0.85), 6.5, 20.0)),
            canopy_radius_mm=float(_clamp(self.canopy_radius_mm + rng.normal(0.0, 0.35), 2.5, 9.5)),
            root_radius_mm=float(_clamp(self.root_radius_mm + rng.normal(0.0, 0.3), 1.8, 7.0)),
            leaf_efficiency=float(_clamp(self.leaf_efficiency + rng.normal(0.0, 0.06), 0.55, 1.65)),
            root_uptake_efficiency=float(_clamp(self.root_uptake_efficiency + rng.normal(0.0, 0.05), 0.45, 1.6)),
            water_use_efficiency=float(_clamp(self.water_use_efficiency + rng.normal(0.0, 0.05), 0.45, 1.5)),
            volatile_scale=float(_clamp(self.volatile_scale + rng.normal(0.0, 0.08), 0.45, 1.8)),
            fruiting_threshold=float(_clamp(self.fruiting_threshold + rng.normal(0.0, 0.06), 0.3, 1.5)),
            litter_turnover=float(_clamp(self.litter_turnover + rng.normal(0.0, 0.06), 0.45, 1.8)),
            shade_tolerance=float(_clamp(self.shade_tolerance + rng.normal(0.0, 0.05), 0.4, 1.7)),
            root_depth_bias=float(_clamp(self.root_depth_bias + rng.normal(0.0, 0.04), 0.05, 1.1)),
            symbiosis_affinity=float(_clamp(self.symbiosis_affinity + rng.normal(0.0, 0.05), 0.35, 1.8)),
            seed_mass=float(_clamp(self.seed_mass + rng.normal(0.0, 0.01), 0.03, 0.2)),
        )


@dataclass
class SeedPropagule:
    """Dormant plant propagule carrying inherited traits."""

    x: float
    y: float
    dormancy_s: float
    genome: PlantGenome
    reserve_carbon: float
    age_s: float = 0.0


def _make_plant_cell_state(tissue: str, cell_count: float, genome: PlantGenome) -> WholeCellState:
    scale = max(0.4, cell_count / 100.0)
    if tissue == "leaf":
        radius_nm = 1800.0
        metabolites = {
            "ATP": 2.6,
            "ADP": 1.0,
            "glucose": 1.7,
            "starch": 1.3,
            "water": 4.8,
            "nitrate": 0.35,
            "amino_acid": 0.6,
            "auxin": 0.06,
        }
        proteins = {"rubisco": 140.0 * scale, "aquaporin": 38.0 * scale, "photosystem": 110.0 * scale}
    elif tissue == "stem":
        radius_nm = 1400.0
        metabolites = {
            "ATP": 1.8,
            "ADP": 0.9,
            "glucose": 1.1,
            "starch": 0.6,
            "water": 3.8,
            "nitrate": 0.25,
            "amino_acid": 0.5,
            "auxin": 0.10,
        }
        proteins = {"cellulose_synthase": 90.0 * scale, "aquaporin": 28.0 * scale, "transportase": 54.0 * scale}
    elif tissue == "root":
        radius_nm = 1500.0
        metabolites = {
            "ATP": 1.7,
            "ADP": 1.0,
            "glucose": 0.9,
            "starch": 0.3,
            "water": 4.4,
            "nitrate": 1.2,
            "amino_acid": 0.7,
            "auxin": 0.08,
        }
        proteins = {"nitrate_transporter": 120.0 * scale, "aquaporin": 64.0 * scale, "pump": 55.0 * scale}
    else:
        radius_nm = 1100.0
        metabolites = {
            "ATP": 2.0,
            "ADP": 0.9,
            "glucose": 1.2,
            "starch": 0.4,
            "water": 3.5,
            "nitrate": 0.5,
            "amino_acid": 0.8,
            "auxin": 0.16,
        }
        proteins = {"cyclin": 28.0 * scale, "histone": 64.0 * scale, "aquaporin": 20.0 * scale}

    surface_area_nm2 = 4.0 * math.pi * radius_nm * radius_nm
    volume_nm3 = (4.0 / 3.0) * math.pi * radius_nm * radius_nm * radius_nm
    return WholeCellState(
        organism=f"coarse_plant_{tissue}_cluster",
        compartments={
            CellCompartment.CYTOPLASM: {
                "water": metabolites["water"] * scale,
                "sucrose": metabolites["glucose"] * 0.8 * scale,
                "nitrate": metabolites["nitrate"] * scale,
            },
            CellCompartment.MEMBRANE: {
                "transporters": 30.0 * scale,
                "channels": 20.0 * scale,
            },
            CellCompartment.EXTRACELLULAR: {
                "apoplast_water": metabolites["water"] * 0.7 * scale,
                "apoplast_nitrate": metabolites["nitrate"] * 0.5 * scale,
            },
        },
        metabolites_mM=metabolites,
        proteins=proteins,
        transcripts={
            "stress_response": 0.2,
            "cell_cycle": 0.3 if tissue == "meristem" else 0.1,
            "transport_program": 0.3 if tissue in {"stem", "root"} else 0.12,
        },
        chromosome=ChromosomeState(genome_bp=135_000_000),
        geometry=GeometryState(
            radius_nm=radius_nm,
            surface_area_nm2=surface_area_nm2,
            volume_nm3=volume_nm3,
            morphology="plant_cell_cluster",
        ),
        metadata={
            "model_family": "coarse_plant_cell_cluster",
            "tissue": tissue,
            "seed_mass": genome.seed_mass,
        },
    )


def _make_cluster_chemistry(tissue: str) -> CellularMetabolism:
    chemistry_cls = RustCellularMetabolism if RustCellularMetabolism is not None else CellularMetabolism
    if tissue == "leaf":
        return chemistry_cls(glucose=5.8, pyruvate=0.14, lactate=0.8, oxygen=0.07, atp=3.2, adp=0.35, amp=0.05, nad_plus=0.55, nadh=0.08)
    if tissue == "stem":
        return chemistry_cls(glucose=4.6, pyruvate=0.12, lactate=0.9, oxygen=0.05, atp=2.6, adp=0.40, amp=0.06, nad_plus=0.50, nadh=0.07)
    if tissue == "root":
        return chemistry_cls(glucose=4.2, pyruvate=0.10, lactate=1.2, oxygen=0.04, atp=2.4, adp=0.45, amp=0.07, nad_plus=0.55, nadh=0.06)
    return chemistry_cls(glucose=4.8, pyruvate=0.10, lactate=1.0, oxygen=0.05, atp=2.7, adp=0.38, amp=0.05, nad_plus=0.52, nadh=0.07)


def _clamp_state_maps(state: WholeCellState) -> None:
    for mapping in (state.metabolites_mM, state.proteins, state.transcripts):
        for key, value in list(mapping.items()):
            mapping[key] = max(0.0, float(value))
    for mapping in state.compartments.values():
        for key, value in list(mapping.items()):
            mapping[key] = max(0.0, float(value))
    state.geometry.division_progress = _clamp(state.geometry.division_progress, 0.0, 1.0)


@dataclass
class PlantCellCluster:
    tissue: str
    cell_count: float
    state: WholeCellState
    chemistry: CellularMetabolism
    vitality: float = 1.0
    division_buffer: float = 0.0

    @property
    def energy_charge(self) -> float:
        atp = self.state.metabolites_mM.get("ATP", 0.0)
        adp = self.state.metabolites_mM.get("ADP", 0.0)
        amp = self.state.metabolites_mM.get("AMP", 0.0) + self.chemistry.amp
        total = atp + adp + amp
        if total <= 1e-9:
            nucleotide_ratio = 0.0
        else:
            nucleotide_ratio = atp / total
        return _clamp(nucleotide_ratio * 0.62 + self.chemistry.energy_ratio * 0.38, 0.0, 1.0)

    @property
    def sugar_pool(self) -> float:
        return max(
            0.0,
            self.state.metabolites_mM.get("glucose", 0.0)
            + self.state.metabolites_mM.get("starch", 0.0)
            + self.state.compartments.get(CellCompartment.CYTOPLASM, {}).get("sucrose", 0.0) * 0.4,
        )

    @property
    def water_pool(self) -> float:
        return max(
            0.0,
            self.state.metabolites_mM.get("water", 0.0)
            + self.state.compartments.get(CellCompartment.CYTOPLASM, {}).get("water", 0.0) * 0.08,
        )

    @property
    def nitrogen_pool(self) -> float:
        return max(
            0.0,
            self.state.metabolites_mM.get("nitrate", 0.0)
            + self.state.metabolites_mM.get("amino_acid", 0.0),
        )

    @property
    def chemistry_vector(self) -> NDArray:
        return np.array(
            [
                self.chemistry.glucose,
                self.chemistry.pyruvate,
                self.chemistry.lactate,
                self.chemistry.oxygen,
                self.chemistry.atp,
                self.chemistry.adp,
                self.chemistry.amp,
                self.chemistry.nad_plus,
                self.chemistry.nadh,
            ],
            dtype=np.float32,
        )


@dataclass
class PlantCellularState:
    leaf: PlantCellCluster
    stem: PlantCellCluster
    root: PlantCellCluster
    meristem: PlantCellCluster
    rust_state: object | None = None
    division_events: float = 0.0
    last_new_cells: float = 0.0
    last_senescence: float = 0.0

    @classmethod
    def from_genome(cls, genome: PlantGenome, biomass_scale: float) -> "PlantCellularState":
        leaf_cells = 160.0 * biomass_scale * (0.88 + genome.leaf_efficiency * 0.22)
        stem_cells = 120.0 * biomass_scale * (0.84 + genome.max_height_mm / 24.0)
        root_cells = 145.0 * biomass_scale * (0.78 + genome.root_depth_bias * 0.55)
        meristem_cells = 70.0 * biomass_scale * (0.85 + genome.seed_mass * 3.8)
        rust_state = None
        if RustPlantCellularState is not None:
            rust_state = RustPlantCellularState(leaf_cells, stem_cells, root_cells, meristem_cells)
        return cls(
            leaf=PlantCellCluster("leaf", leaf_cells, _make_plant_cell_state("leaf", leaf_cells, genome), _make_cluster_chemistry("leaf")),
            stem=PlantCellCluster("stem", stem_cells, _make_plant_cell_state("stem", stem_cells, genome), _make_cluster_chemistry("stem")),
            root=PlantCellCluster("root", root_cells, _make_plant_cell_state("root", root_cells, genome), _make_cluster_chemistry("root")),
            meristem=PlantCellCluster("meristem", meristem_cells, _make_plant_cell_state("meristem", meristem_cells, genome), _make_cluster_chemistry("meristem")),
            rust_state=rust_state,
        )

    def clusters(self) -> tuple[PlantCellCluster, PlantCellCluster, PlantCellCluster, PlantCellCluster]:
        return (self.leaf, self.stem, self.root, self.meristem)

    @property
    def total_cells(self) -> float:
        return sum(cluster.cell_count for cluster in self.clusters())

    @property
    def vitality(self) -> float:
        return _weighted_mean([(cluster.cell_count, cluster.vitality) for cluster in self.clusters()])

    @property
    def energy_charge(self) -> float:
        return _weighted_mean([(cluster.cell_count, cluster.energy_charge) for cluster in self.clusters()])

    @property
    def sugar_pool(self) -> float:
        return sum(cluster.sugar_pool for cluster in self.clusters())

    @property
    def water_pool(self) -> float:
        return sum(cluster.water_pool for cluster in self.clusters())

    @property
    def nitrogen_pool(self) -> float:
        return sum(cluster.nitrogen_pool for cluster in self.clusters())

    @property
    def division_signal(self) -> float:
        return self.meristem.division_buffer

    @property
    def metabolism_backend(self) -> str:
        if self.rust_state is not None:
            return "rust"
        if RustCellularMetabolism is not None and isinstance(self.leaf.chemistry, RustCellularMetabolism):
            return "rust"
        return "python"

    def _sync_cluster_from_rust(self, cluster: PlantCellCluster) -> None:
        if self.rust_state is None:
            return
        (
            cell_count,
            vitality,
            division_buffer,
            state_atp,
            state_adp,
            state_glucose,
            state_starch,
            state_water,
            state_nitrate,
            state_amino_acid,
            state_auxin,
            cytoplasm_water,
            cytoplasm_sucrose,
            cytoplasm_nitrate,
            apoplast_water,
            apoplast_nitrate,
            transcript_stress_response,
            transcript_cell_cycle,
            transcript_transport_program,
            chem_glucose,
            chem_pyruvate,
            chem_lactate,
            chem_oxygen,
            chem_atp,
            chem_adp,
            chem_amp,
            chem_nad_plus,
            chem_nadh,
        ) = self.rust_state.cluster_snapshot(cluster.tissue)

        cluster.cell_count = float(cell_count)
        cluster.vitality = float(vitality)
        cluster.division_buffer = float(division_buffer)

        state = cluster.state.metabolites_mM
        state["ATP"] = float(state_atp)
        state["ADP"] = float(state_adp)
        state["AMP"] = 0.0
        state["glucose"] = float(state_glucose)
        state["starch"] = float(state_starch)
        state["water"] = float(state_water)
        state["nitrate"] = float(state_nitrate)
        state["amino_acid"] = float(state_amino_acid)
        state["auxin"] = float(state_auxin)

        cyt = cluster.state.compartments[CellCompartment.CYTOPLASM]
        cyt["water"] = float(cytoplasm_water)
        cyt["sucrose"] = float(cytoplasm_sucrose)
        cyt["nitrate"] = float(cytoplasm_nitrate)
        ext = cluster.state.compartments[CellCompartment.EXTRACELLULAR]
        ext["apoplast_water"] = float(apoplast_water)
        ext["apoplast_nitrate"] = float(apoplast_nitrate)

        cluster.state.transcripts["stress_response"] = float(transcript_stress_response)
        cluster.state.transcripts["cell_cycle"] = float(transcript_cell_cycle)
        cluster.state.transcripts["transport_program"] = float(transcript_transport_program)
        cluster.state.geometry.division_progress = float(division_buffer)

        chem = cluster.chemistry
        if RustCellularMetabolism is not None and isinstance(chem, RustCellularMetabolism):
            # Rust chemistry is authoritative; keep it in sync through direct supply by rebuilding state.
            pass
        else:
            chem.glucose = float(chem_glucose)
            chem.pyruvate = float(chem_pyruvate)
            chem.lactate = float(chem_lactate)
            chem.oxygen = float(chem_oxygen)
            chem.atp = float(chem_atp)
            chem.adp = float(chem_adp)
            chem.amp = float(chem_amp)
            chem.nad_plus = float(chem_nad_plus)
            chem.nadh = float(chem_nadh)
        _clamp_state_maps(cluster.state)

    def _update_vitality(self, cluster: PlantCellCluster, stress_signal: float) -> None:
        cluster.vitality = _clamp(
            0.10
            + cluster.energy_charge * 0.28
            + min(1.0, cluster.sugar_pool / 2.8) * 0.18
            + min(1.0, cluster.water_pool / 5.0) * 0.15
            + min(1.0, cluster.nitrogen_pool / 2.0) * 0.11
            - stress_signal * 0.12,
            0.0,
            1.0,
        )

    def _homeostatic_recovery(self, cluster: PlantCellCluster, temp_factor: float, stress_signal: float) -> None:
        state = cluster.state.metabolites_mM
        atp = state.get("ATP", 0.0)
        glucose = state.get("glucose", 0.0)
        starch = state.get("starch", 0.0)
        water = state.get("water", 0.0)
        amino = state.get("amino_acid", 0.0)
        available_carbon = glucose + starch * 0.7
        if atp >= 0.18 or available_carbon <= 0.05 or water <= 0.35:
            return
        rescue = min(
            0.08 + amino * 0.015,
            available_carbon * (0.06 + temp_factor * 0.025) * max(0.30, 1.0 - stress_signal * 0.18),
        )
        if rescue <= 0.0:
            return
        glucose_draw = min(glucose, rescue * 0.82)
        starch_draw = min(starch, max(0.0, rescue - glucose_draw))
        state["glucose"] = max(0.0, glucose - glucose_draw)
        state["starch"] = max(0.0, starch - starch_draw)
        state["ATP"] = atp + rescue
        state["ADP"] = max(0.0, state.get("ADP", 0.0) - rescue * 0.34)
        state["water"] = max(0.0, water - rescue * 0.04)

    def _step_cluster_chemistry(
        self,
        cluster: PlantCellCluster,
        *,
        dt: float,
        glucose_flux: float,
        oxygen_flux: float,
        lactate_flux: float,
        protein_load: float,
        transport_load: float,
    ) -> None:
        chem = cluster.chemistry
        state = cluster.state.metabolites_mM
        chem.supply_glucose(max(0.0, glucose_flux) + max(0.0, state.get("glucose", 0.0)) * 0.02)
        chem.supply_oxygen(max(0.0, oxygen_flux))
        chem.supply_lactate(max(0.0, lactate_flux) + max(0.0, state.get("starch", 0.0)) * 0.01)

        total_ms = min(360.0, max(40.0, dt * 5.0))
        n_sub = max(1, min(4, int(total_ms // 110.0) + 1))
        chem_dt = total_ms / n_sub
        for _ in range(n_sub):
            chem.consume_atp(max(0.0, transport_load) * chem_dt * 0.00020, f"{cluster.tissue}_transport")
            chem.protein_synthesis_cost(chem_dt, max(0.0, protein_load) * 0.55)
            chem.step(chem_dt)

        state["ATP"] = state.get("ATP", 0.0) * 0.84 + chem.atp * 0.16
        state["ADP"] = state.get("ADP", 0.0) * 0.84 + chem.adp * 0.16
        state["glucose"] = state.get("glucose", 0.0) * 0.92 + chem.glucose * 0.08
        state["water"] = state.get("water", 0.0) * 0.985 + chem.oxygen * 1.2
        state["amino_acid"] = state.get("amino_acid", 0.0) * 0.985 + chem.pyruvate * 0.015

    def step(
        self,
        dt: float,
        *,
        local_light: float,
        temp_factor: float,
        water_in: float,
        nutrient_in: float,
        water_status: float,
        nutrient_status: float,
        symbiosis: float,
        stress_signal: float,
        storage_signal: float,
    ) -> dict[str, float]:
        for cluster in self.clusters():
            cluster.state.advance_time(dt * 1000.0)

        if self.rust_state is not None:
            (
                photosynthetic_capacity,
                maintenance_cost,
                storage_exchange,
                division_growth,
                senescence_mass,
                energy_charge,
                vitality,
                sugar_pool,
                water_pool,
                nitrogen_pool,
                division_signal,
                new_cells,
            ) = self.rust_state.step(
                float(dt),
                float(local_light),
                float(temp_factor),
                float(water_in),
                float(nutrient_in),
                float(water_status),
                float(nutrient_status),
                float(symbiosis),
                float(stress_signal),
                float(storage_signal),
            )
            for cluster in self.clusters():
                self._sync_cluster_from_rust(cluster)
            self.division_events = float(self.rust_state.division_events)
            self.last_new_cells = float(self.rust_state.last_new_cells)
            self.last_senescence = float(self.rust_state.last_senescence)
            return {
                "photosynthetic_capacity": float(photosynthetic_capacity),
                "maintenance_cost": float(maintenance_cost),
                "storage_exchange": float(storage_exchange),
                "division_growth": float(division_growth),
                "senescence_mass": float(senescence_mass),
                "energy_charge": float(energy_charge),
                "vitality": float(vitality),
                "sugar_pool": float(sugar_pool),
                "water_pool": float(water_pool),
                "nitrogen_pool": float(nitrogen_pool),
                "division_signal": float(division_signal),
                "new_cells": float(new_cells),
            }

        stress = max(0.0, stress_signal)
        leaf_resp = self.leaf.cell_count * (0.0000032 + 0.0000026 * temp_factor + stress * 0.0000010) * dt
        stem_resp = self.stem.cell_count * (0.0000024 + 0.0000018 * temp_factor + stress * 0.0000008) * dt
        root_resp = self.root.cell_count * (0.0000028 + 0.0000020 * temp_factor + stress * 0.0000010) * dt
        meristem_resp = self.meristem.cell_count * (0.0000036 + 0.0000024 * temp_factor + stress * 0.0000012) * dt

        photosynthate = (
            local_light
            * temp_factor
            * min(1.0, water_status)
            * min(1.0, nutrient_status)
            * self.leaf.cell_count
            * 0.000010
            * dt
        )
        starch_release = min(
            self.leaf.state.metabolites_mM.get("starch", 0.0),
            max(0.0, stress - 0.30) * self.leaf.cell_count * 0.0000018 * dt,
        )
        sugar_export = max(0.0, (photosynthate + starch_release) * 0.46)

        root_water_capture = water_in * (0.70 + symbiosis * 3.2)
        root_nitrogen_capture = nutrient_in * (0.68 + symbiosis * 1.3)
        xylem_flow = root_water_capture * 0.60
        stem_to_leaf_water = xylem_flow * 0.64
        stem_to_meristem_water = xylem_flow * 0.12
        transpiration = local_light * (0.00026 + self.leaf.cell_count * 0.00000014) * dt * (1.0 + stress * 0.35)

        nitrate_assim = root_nitrogen_capture * (0.50 + symbiosis * 0.90)
        amino_flux = nitrate_assim * 0.62
        stem_to_root_sugar = sugar_export * 0.20
        stem_to_meristem_sugar = sugar_export * 0.28
        stem_buffer_sugar = sugar_export - stem_to_root_sugar - stem_to_meristem_sugar
        leaf_catabolism = min(
            self.leaf.state.metabolites_mM.get("glucose", 0.0) * 0.0011 * dt * temp_factor,
            0.45,
        )
        stem_catabolism = min(
            (self.stem.state.metabolites_mM.get("glucose", 0.0) + stem_buffer_sugar) * 0.0015 * dt * temp_factor,
            0.42,
        )
        root_catabolism = min(
            (self.root.state.metabolites_mM.get("glucose", 0.0) + stem_to_root_sugar) * 0.0018 * dt * temp_factor,
            0.40,
        )
        meristem_catabolism = min(
            (self.meristem.state.metabolites_mM.get("glucose", 0.0) + stem_to_meristem_sugar) * 0.0021 * dt * temp_factor,
            0.46,
        )

        self.leaf.state.apply_deltas(
            compartment_deltas={
                CellCompartment.CYTOPLASM.value: {
                    "water": stem_to_leaf_water * 0.65 - transpiration * 0.35,
                    "sucrose": photosynthate * 0.32 - sugar_export * 0.18,
                },
                CellCompartment.EXTRACELLULAR.value: {
                    "apoplast_water": stem_to_leaf_water * 0.35 - transpiration * 0.65,
                },
            },
            metabolite_deltas={
                "ATP": photosynthate * 0.42 + leaf_catabolism * 0.84 - leaf_resp * 0.55,
                "ADP": leaf_resp * 0.22 - photosynthate * 0.14 - leaf_catabolism * 0.44,
                "glucose": photosynthate * 0.68 + starch_release * 0.45 - sugar_export * 0.58 - leaf_resp * 0.10 - leaf_catabolism,
                "starch": photosynthate * 0.20 - starch_release,
                "water": stem_to_leaf_water + water_in * 0.14 - transpiration,
                "nitrate": nutrient_in * 0.08 - nitrate_assim * 0.18,
                "amino_acid": amino_flux * 0.12 - leaf_resp * 0.02,
                "auxin": local_light * 0.0004 * dt - stress * 0.0003 * dt,
            },
            protein_deltas={
                "rubisco": photosynthate * 0.010 - stress * 0.0015 * dt,
                "photosystem": photosynthate * 0.006 - stress * 0.0010 * dt,
            },
            transcript_deltas={
                "stress_response": stress * 0.003 * dt - photosynthate * 0.0008,
            },
        )

        self.stem.state.apply_deltas(
            compartment_deltas={
                CellCompartment.CYTOPLASM.value: {
                    "water": xylem_flow - stem_to_leaf_water - stem_to_meristem_water,
                    "sucrose": sugar_export * 0.40 - stem_to_root_sugar - stem_to_meristem_sugar,
                },
            },
            metabolite_deltas={
                "ATP": stem_buffer_sugar * 0.18 + stem_catabolism * 0.92 - stem_resp * 0.48,
                "ADP": stem_resp * 0.18 - stem_buffer_sugar * 0.08 - stem_catabolism * 0.50,
                "glucose": stem_buffer_sugar * 0.54 - stem_resp * 0.10 - stem_catabolism,
                "starch": stem_buffer_sugar * 0.10,
                "water": xylem_flow - stem_to_leaf_water - stem_to_meristem_water,
                "nitrate": root_nitrogen_capture * 0.14 - amino_flux * 0.10,
                "amino_acid": amino_flux * 0.18 - stem_resp * 0.02,
                "auxin": self.meristem.state.metabolites_mM.get("auxin", 0.0) * 0.01,
            },
            protein_deltas={
                "cellulose_synthase": stem_buffer_sugar * 0.012 - stress * 0.0012 * dt,
                "transportase": stem_buffer_sugar * 0.005,
            },
            transcript_deltas={
                "transport_program": stem_buffer_sugar * 0.004 - stress * 0.0015 * dt,
            },
        )

        self.root.state.apply_deltas(
            compartment_deltas={
                CellCompartment.EXTRACELLULAR.value: {
                    "apoplast_water": root_water_capture * 0.55,
                    "apoplast_nitrate": root_nitrogen_capture * 0.50,
                },
                CellCompartment.CYTOPLASM.value: {
                    "water": root_water_capture * 0.45 - xylem_flow * 0.65,
                    "nitrate": root_nitrogen_capture * 0.52,
                },
            },
            metabolite_deltas={
                "ATP": root_nitrogen_capture * 0.18 + stem_to_root_sugar * 0.20 + root_catabolism * 0.96 - root_resp * 0.56,
                "ADP": root_resp * 0.20 - stem_to_root_sugar * 0.10 - root_catabolism * 0.54,
                "glucose": stem_to_root_sugar * 0.66 - root_resp * 0.14 - root_catabolism,
                "starch": stem_to_root_sugar * 0.08,
                "water": root_water_capture - xylem_flow,
                "nitrate": root_nitrogen_capture - nitrate_assim,
                "amino_acid": amino_flux * 0.50 - root_resp * 0.04,
                "auxin": 0.0003 * dt,
            },
            protein_deltas={
                "nitrate_transporter": root_nitrogen_capture * 0.018 - stress * 0.0014 * dt,
                "aquaporin": root_water_capture * 0.010 - stress * 0.0008 * dt,
            },
            transcript_deltas={
                "transport_program": root_nitrogen_capture * 0.008 + root_water_capture * 0.006 - stress * 0.0012 * dt,
            },
        )

        self.meristem.state.apply_deltas(
            compartment_deltas={
                CellCompartment.CYTOPLASM.value: {
                    "water": stem_to_meristem_water,
                    "sucrose": stem_to_meristem_sugar * 0.70,
                },
            },
            metabolite_deltas={
                "ATP": stem_to_meristem_sugar * 0.34 + amino_flux * 0.12 + meristem_catabolism * 1.02 - meristem_resp * 0.58,
                "ADP": meristem_resp * 0.20 - stem_to_meristem_sugar * 0.14 - meristem_catabolism * 0.58,
                "glucose": stem_to_meristem_sugar * 0.72 - meristem_resp * 0.14 - meristem_catabolism,
                "starch": stem_to_meristem_sugar * 0.05,
                "water": stem_to_meristem_water - meristem_resp * 0.04,
                "nitrate": root_nitrogen_capture * 0.08 - amino_flux * 0.06,
                "amino_acid": amino_flux * 0.20 - meristem_resp * 0.03,
                "auxin": stem_to_meristem_sugar * 0.01 + 0.0006 * dt,
            },
            protein_deltas={
                "cyclin": stem_to_meristem_sugar * 0.018 - stress * 0.0015 * dt,
                "histone": stem_to_meristem_sugar * 0.010,
            },
            transcript_deltas={
                "cell_cycle": stem_to_meristem_sugar * 0.010 + amino_flux * 0.005 - stress * 0.0016 * dt,
            },
        )

        self._step_cluster_chemistry(
            self.leaf,
            dt=dt,
            glucose_flux=photosynthate * 0.12,
            oxygen_flux=0.006 + local_light * 0.010,
            lactate_flux=0.002,
            protein_load=0.18 + self.leaf.state.transcripts.get("stress_response", 0.0),
            transport_load=0.10 + self.leaf.state.transcripts.get("transport_program", 0.0),
        )
        self._step_cluster_chemistry(
            self.stem,
            dt=dt,
            glucose_flux=stem_buffer_sugar * 0.18,
            oxygen_flux=0.004 + local_light * 0.004,
            lactate_flux=0.003,
            protein_load=0.14 + self.stem.state.transcripts.get("transport_program", 0.0),
            transport_load=0.16 + self.stem.state.transcripts.get("transport_program", 0.0),
        )
        self._step_cluster_chemistry(
            self.root,
            dt=dt,
            glucose_flux=stem_to_root_sugar * 0.20,
            oxygen_flux=0.003 + water_status * 0.004 + symbiosis * 0.010,
            lactate_flux=0.006 + symbiosis * 0.008,
            protein_load=0.12 + self.root.state.transcripts.get("transport_program", 0.0),
            transport_load=0.18 + self.root.state.transcripts.get("transport_program", 0.0),
        )
        self._step_cluster_chemistry(
            self.meristem,
            dt=dt,
            glucose_flux=stem_to_meristem_sugar * 0.24,
            oxygen_flux=0.004 + local_light * 0.003,
            lactate_flux=0.004,
            protein_load=0.16 + self.meristem.state.transcripts.get("cell_cycle", 0.0),
            transport_load=0.10 + self.meristem.state.transcripts.get("cell_cycle", 0.0),
        )

        resource_ready = min(
            1.0,
            (self.meristem.state.metabolites_mM.get("glucose", 0.0) + stem_to_meristem_sugar * 6.0) / 1.2,
            (self.meristem.state.metabolites_mM.get("water", 0.0) + stem_to_meristem_water * 18.0) / 2.2,
            (self.meristem.state.metabolites_mM.get("amino_acid", 0.0) + amino_flux * 0.8) / 0.7,
            (self.meristem.energy_charge + meristem_catabolism * 0.4) / 0.60,
        )
        division_drive = (
            resource_ready
            * temp_factor
            * (0.32 + storage_signal * 0.12)
            * (1.0 - min(stress, 1.4) * 0.52)
        )
        new_cells = max(0.0, division_drive * self.meristem.cell_count * 0.00012 * dt)
        self.leaf.cell_count += new_cells * 0.36
        self.stem.cell_count += new_cells * 0.24
        self.root.cell_count += new_cells * 0.28
        self.meristem.cell_count += new_cells * 0.12
        self.meristem.division_buffer = _clamp(division_drive, 0.0, 1.0)
        self.meristem.state.geometry.division_progress = self.meristem.division_buffer
        self.division_events += new_cells
        self.last_new_cells = new_cells

        senescence = max(0.0, stress - 0.72) * (0.00005 * dt) * (self.leaf.cell_count + self.root.cell_count)
        leaf_loss = min(self.leaf.cell_count * 0.02, senescence * 0.62)
        root_loss = min(self.root.cell_count * 0.015, senescence * 0.38)
        self.leaf.cell_count = max(8.0, self.leaf.cell_count - leaf_loss)
        self.root.cell_count = max(8.0, self.root.cell_count - root_loss)
        self.last_senescence = leaf_loss + root_loss

        for cluster in self.clusters():
            _clamp_state_maps(cluster.state)
            self._homeostatic_recovery(cluster, temp_factor, stress)
            self._update_vitality(cluster, stress)

        total_resp = leaf_resp + stem_resp + root_resp + meristem_resp
        energy_charge = self.energy_charge
        vitality = self.vitality
        return {
            "photosynthetic_capacity": _clamp(0.66 + self.leaf.energy_charge * 0.36 + vitality * 0.16, 0.45, 1.4),
            "maintenance_cost": total_resp * 0.0045,
            "storage_exchange": photosynthate * 0.0018 - total_resp * 0.0012,
            "division_growth": new_cells * 0.00042,
            "senescence_mass": (leaf_loss + root_loss) * 0.00026,
            "energy_charge": energy_charge,
            "vitality": vitality,
            "sugar_pool": self.sugar_pool,
            "water_pool": self.water_pool,
            "nitrogen_pool": self.nitrogen_pool,
            "division_signal": self.division_signal,
            "new_cells": new_cells,
        }


@dataclass
class PlantOrganism:
    """A process-based plant linked to a PlantSource in MolecularWorld."""

    source: PlantSource
    genome: PlantGenome
    leaf_biomass: float
    stem_biomass: float
    root_biomass: float
    storage_carbon: float
    water_buffer: float
    nitrogen_buffer: float
    fruit_timer_s: float
    seed_timer_s: float
    cellular_state: PlantCellularState
    rust_state: object | None = None
    age_s: float = 0.0
    health: float = 1.0
    fruit_count: int = 0

    @property
    def total_biomass(self) -> float:
        return self.leaf_biomass + self.stem_biomass + self.root_biomass + self.storage_carbon

    @property
    def root_radius_cells(self) -> int:
        dynamic_radius = self.genome.root_radius_mm + self.root_biomass * 3.4 + self.cellular_state.root.cell_count * 0.0025
        return max(1, int(round(dynamic_radius)))

    @property
    def canopy_radius_cells(self) -> int:
        dynamic_radius = self.genome.canopy_radius_mm + self.leaf_biomass * 3.1 + self.cellular_state.leaf.cell_count * 0.0022
        return max(1, int(round(dynamic_radius)))

    @property
    def total_cells(self) -> float:
        return self.cellular_state.total_cells

    @property
    def cell_vitality(self) -> float:
        return self.cellular_state.vitality

    @property
    def cellular_energy_charge(self) -> float:
        return self.cellular_state.energy_charge

    @property
    def cellular_division_pressure(self) -> float:
        return self.cellular_state.division_signal

    @property
    def physiology_backend(self) -> str:
        return "rust" if self.rust_state is not None else "python"

    def _sync_from_rust(self) -> None:
        if self.rust_state is None:
            return
        self.leaf_biomass = float(self.rust_state.leaf_biomass)
        self.stem_biomass = float(self.rust_state.stem_biomass)
        self.root_biomass = float(self.rust_state.root_biomass)
        self.storage_carbon = float(self.rust_state.storage_carbon)
        self.water_buffer = float(self.rust_state.water_buffer)
        self.nitrogen_buffer = float(self.rust_state.nitrogen_buffer)
        self.fruit_timer_s = float(self.rust_state.fruit_timer_s)
        self.seed_timer_s = float(self.rust_state.seed_timer_s)
        self.age_s = float(self.rust_state.age_s)
        self.health = float(self.rust_state.health)
        self.fruit_count = int(self.rust_state.fruit_count)
        self.source.height = float(self.rust_state.height_mm)
        self.source.nectar_production_rate = float(self.rust_state.nectar_production_rate)
        self.source.odorant_profile = {
            "geraniol": float(self.rust_state.odorant_geraniol),
            "ethyl_acetate": float(self.rust_state.odorant_ethyl_acetate),
        }
        self.source.odorant_emission_rate = float(self.rust_state.odorant_emission_rate)

    def step(self, ecology: "TerrariumEcology", dt: float) -> None:
        self.age_s += dt
        world = ecology.world
        soil = ecology.soil
        x = int(np.clip(self.source.x, 0, world.W - 1))
        y = int(np.clip(self.source.y, 0, world.H - 1))

        microhabitat = ecology.microhabitat_at(self)
        root_cluster = self.cellular_state.root
        root_energy_gate = _clamp(0.35 + root_cluster.energy_charge * 0.85, 0.18, 1.2)
        root_water_deficit = max(0.0, 5.0 - root_cluster.water_pool)
        root_nitrogen_deficit = max(0.0, 1.6 - root_cluster.nitrogen_pool)
        if self.rust_state is not None:
            water_demand, nutrient_demand = self.rust_state.resource_demands(
                float(dt),
                float(root_energy_gate),
                float(root_water_deficit),
                float(root_nitrogen_deficit),
            )
        else:
            water_demand = (
                0.00082
                + self.root_biomass * 0.00060
                + root_water_deficit * 0.00008
            ) * dt * self.genome.root_uptake_efficiency * root_energy_gate
            nutrient_demand = (
                0.00034
                + self.root_biomass * 0.00020
                + root_nitrogen_deficit * 0.00010
            ) * dt * self.genome.root_uptake_efficiency * root_energy_gate
        water_uptake, nutrient_uptake = soil.extract_resources(
            x=x,
            y=y,
            radius=self.root_radius_cells,
            water_demand=float(water_demand),
            nutrient_demand=float(nutrient_demand),
            deep_fraction=self.genome.root_depth_bias,
            symbiosis_factor=self.genome.symbiosis_affinity,
        )
        self.water_buffer = _clamp(self.water_buffer + water_uptake, 0.0, 3.2)
        self.nitrogen_buffer = _clamp(self.nitrogen_buffer + nutrient_uptake, 0.0, 2.8)

        light = float(world.sample_light(x, y))
        temp = float(world.sample_temperature(x, y))
        temp_factor = _temp_response(temp, optimum=23.5, width=11.0)

        shading_penalty = math.exp(
            -microhabitat["canopy_competition"] * max(0.16, 1.18 - self.genome.shade_tolerance)
        )
        local_light = light * shading_penalty
        root_pressure = 1.0 / (1.0 + microhabitat["root_competition"] * 0.32)
        symbiosis_bonus = 1.0 + microhabitat["symbionts"] * 7.5 * self.genome.symbiosis_affinity

        water_factor = _clamp(
            (self.water_buffer + microhabitat["deep_moisture"] * 0.08) / (0.12 + self.total_biomass * 0.05),
            0.0,
            1.6,
        )
        nutrient_factor = _clamp(
            (
                self.nitrogen_buffer
                + microhabitat["symbionts"] * 0.06
                + microhabitat["soil_ammonium"] * 0.9
                + microhabitat["soil_nitrate"] * 0.7
            ) / (0.07 + self.total_biomass * 0.025),
            0.0,
            1.6,
        )
        redox_stress = max(0.0, 0.42 - microhabitat["soil_redox"]) * 0.75
        stress_signal = (
            microhabitat["canopy_competition"] * 0.18
            + microhabitat["root_competition"] * 0.12
            + max(0.0, 0.70 - water_factor) * 0.55
            + max(0.0, 0.70 - nutrient_factor) * 0.45
            + redox_stress
        )
        cell_feedback = self.cellular_state.step(
            dt,
            local_light=local_light,
            temp_factor=temp_factor,
            water_in=water_uptake,
            nutrient_in=nutrient_uptake,
            water_status=water_factor,
            nutrient_status=nutrient_factor,
            symbiosis=microhabitat["symbionts"],
            stress_signal=stress_signal,
            storage_signal=_clamp(self.storage_carbon, 0.0, 1.8),
        )

        if self.rust_state is not None:
            fruit_reset_s = float(ecology.rng.uniform(7200.0, 17000.0))
            seed_reset_s = float(ecology.rng.uniform(12000.0, 30000.0))
            exudates, litter, spawned_fruit, fruit_size, spawned_seed = self.rust_state.step(
                float(dt),
                float(water_uptake),
                float(nutrient_uptake),
                float(local_light),
                float(temp_factor),
                float(root_pressure),
                float(symbiosis_bonus),
                float(water_factor),
                float(nutrient_factor),
                float(microhabitat["canopy_competition"]),
                float(microhabitat["root_competition"]),
                float(microhabitat["soil_glucose"]),
                float(cell_feedback["photosynthetic_capacity"]),
                float(cell_feedback["maintenance_cost"]),
                float(cell_feedback["storage_exchange"]),
                float(cell_feedback["division_growth"]),
                float(cell_feedback["senescence_mass"]),
                float(cell_feedback["energy_charge"]),
                float(cell_feedback["vitality"]),
                float(cell_feedback["sugar_pool"]),
                float(cell_feedback["division_signal"]),
                float(self.total_cells),
                fruit_reset_s,
                seed_reset_s,
            )
            self._sync_from_rust()
            if exudates > 0.0:
                soil.deposit_root_exudates(x, y, self.root_radius_cells, float(exudates))
            if litter > 0.0:
                soil.deposit_litter(x, y, self.root_radius_cells, float(litter))
            if spawned_fruit > 0.5:
                ecology.spawn_fruit(x, y, size=float(fruit_size), parent=self)
            if spawned_seed > 0.5:
                ecology.spawn_seed_from(self)
            return

        photosynthesis = (
            self.leaf_biomass
            * self.genome.leaf_efficiency
            * local_light
            * temp_factor
            * min(1.0, water_factor)
            * min(1.0, nutrient_factor)
            * symbiosis_bonus
            * root_pressure
            * cell_feedback["photosynthetic_capacity"]
            * 0.0023
            * dt
        )

        maintenance = (
            self.total_biomass
            * (0.00003 + 0.00005 * temp_factor + 0.000012 * microhabitat["canopy_competition"])
            * dt
        ) + cell_feedback["maintenance_cost"]
        self.storage_carbon = _clamp(
            self.storage_carbon + photosynthesis - maintenance + cell_feedback["storage_exchange"],
            -0.28,
            6.0,
        )

        water_used = min(
            self.water_buffer,
            (0.00018 + self.leaf_biomass * 0.00012) * dt / max(self.genome.water_use_efficiency, 1e-6),
        )
        nitrogen_used = min(self.nitrogen_buffer, (0.00010 + self.leaf_biomass * 0.00006) * dt / max(symbiosis_bonus, 1.0))
        self.water_buffer -= water_used
        self.nitrogen_buffer -= nitrogen_used

        if self.storage_carbon > 0.03:
            allocation = min(
                self.storage_carbon,
                0.00042 * dt * root_pressure * (0.75 + local_light * 0.55) + cell_feedback["division_growth"],
            )
            self.storage_carbon -= allocation
            leaf_share = _clamp(0.37 + self.genome.shade_tolerance * 0.05 - microhabitat["canopy_competition"] * 0.03, 0.24, 0.52)
            root_share = _clamp(0.34 + self.genome.root_depth_bias * 0.10 + microhabitat["root_competition"] * 0.03, 0.24, 0.50)
            stem_share = _clamp(1.0 - leaf_share - root_share, 0.14, 0.36)
            total_share = leaf_share + root_share + stem_share
            self.leaf_biomass += allocation * (leaf_share / total_share)
            self.root_biomass += allocation * (root_share / total_share)
            self.stem_biomass += allocation * (stem_share / total_share)
        elif self.storage_carbon < -0.03:
            stress = min(abs(self.storage_carbon), 0.00022 * dt * (1.0 + microhabitat["canopy_competition"] * 0.2))
            self.leaf_biomass = max(0.04, self.leaf_biomass - stress * 0.52)
            self.stem_biomass = max(0.04, self.stem_biomass - stress * 0.20)
            self.root_biomass = max(0.04, self.root_biomass - stress * 0.28)

        self.health = _clamp(
            0.14
            + min(1.0, local_light) * 0.24
            + min(1.0, water_factor) * 0.24
            + min(1.0, nutrient_factor) * 0.18
            + _clamp(self.storage_carbon, 0.0, 0.9) * 0.40
            + _clamp(symbiosis_bonus - 1.0, 0.0, 1.0) * 0.12
            + cell_feedback["vitality"] * 0.16
            + cell_feedback["energy_charge"] * 0.08
            - microhabitat["canopy_competition"] * 0.05,
            0.0,
            1.6,
        )

        height = _clamp(
            2.0
            + (self.leaf_biomass * 1.6 + self.stem_biomass * 3.6) * (2.2 + self.health * 0.25)
            + self.total_cells * 0.0015,
            2.0,
            self.genome.max_height_mm,
        )
        self.source.height = height
        self.source.nectar_production_rate = _clamp(
            0.02 + self.health * 0.12 + cell_feedback["sugar_pool"] * 0.002,
            0.0,
            0.26,
        )
        self.source.odorant_profile = {
            "geraniol": _clamp(0.78 + self.health * 0.16 + cell_feedback["vitality"] * 0.10, 0.50, 1.30),
            "ethyl_acetate": _clamp(
                self.fruit_count * 0.015 + cell_feedback["sugar_pool"] * 0.002 + microhabitat["soil_glucose"] * 0.08,
                0.0,
                0.16,
            ),
        }
        self.source.odorant_emission_rate = _clamp(
            (
                0.003
                + self.health * 0.010
                + self.fruit_count * 0.0012
                + self.leaf_biomass * 0.003
                + cell_feedback["division_signal"] * 0.003
            )
            * self.genome.volatile_scale,
            0.002,
            0.09,
        )

        exudates = min(
            self.storage_carbon + 0.22,
            (0.00005 + self.root_biomass * 0.00006) * dt * (0.8 + self.genome.symbiosis_affinity * 0.4),
        )
        if exudates > 0.0:
            self.storage_carbon -= exudates * 0.42
            soil.deposit_root_exudates(x, y, self.root_radius_cells, exudates)

        litter = (
            (self.leaf_biomass + self.root_biomass)
            * (0.00002 + (1.0 - min(self.health, 1.0)) * 0.00008)
            * dt
            * self.genome.litter_turnover
        ) + cell_feedback["senescence_mass"]
        if litter > 0.0:
            soil.deposit_litter(x, y, self.root_radius_cells, litter)

        self.fruit_timer_s -= dt
        self.seed_timer_s -= dt

        if (
            self.storage_carbon > self.genome.fruiting_threshold
            and self.fruit_timer_s <= 0.0
            and self.health > 0.52
            and cell_feedback["sugar_pool"] > 2.0
        ):
            ecology.spawn_fruit(x, y, size=_clamp(0.45 + self.storage_carbon * 0.46, 0.35, 1.5), parent=self)
            self.storage_carbon = max(-0.12, self.storage_carbon - 0.18)
            self.fruit_count += 1
            self.fruit_timer_s = float(ecology.rng.uniform(7200.0, 17000.0))

        seed_threshold = self.genome.fruiting_threshold + 0.12 + self.genome.seed_mass * 0.8
        if (
            self.storage_carbon > seed_threshold
            and self.seed_timer_s <= 0.0
            and self.health > 0.58
            and cell_feedback["division_signal"] > 0.02
        ):
            ecology.spawn_seed_from(self)
            self.storage_carbon = max(-0.12, self.storage_carbon - (0.06 + self.genome.seed_mass * 0.55))
            self.seed_timer_s = float(ecology.rng.uniform(12000.0, 30000.0))

    def is_dead(self) -> bool:
        return self.total_biomass < 0.09 or (self.health < 0.03 and self.storage_carbon < -0.2)


@dataclass
class SoilChemistryField:
    """Compressed reaction-diffusion chemistry state for the rhizosphere."""

    glucose: NDArray
    pyruvate: NDArray
    lactate: NDArray
    oxygen: NDArray
    ammonium: NDArray
    nitrate: NDArray
    carbon_dioxide: NDArray
    proton_load: NDArray
    atp_flux: NDArray

    @classmethod
    def from_world(cls, world: MolecularWorld, rng: np.random.Generator) -> "SoilChemistryField":
        shape = (world.H, world.W)
        return cls(
            glucose=rng.uniform(0.002, 0.015, shape).astype(np.float32),
            pyruvate=rng.uniform(0.0008, 0.004, shape).astype(np.float32),
            lactate=rng.uniform(0.001, 0.008, shape).astype(np.float32),
            oxygen=rng.uniform(0.028, 0.060, shape).astype(np.float32),
            ammonium=rng.uniform(0.001, 0.006, shape).astype(np.float32),
            nitrate=rng.uniform(0.004, 0.012, shape).astype(np.float32),
            carbon_dioxide=rng.uniform(0.010, 0.030, shape).astype(np.float32),
            proton_load=rng.uniform(0.002, 0.010, shape).astype(np.float32),
            atp_flux=np.zeros(shape, dtype=np.float32),
        )

    def vector_at(self, x: int, y: int) -> NDArray:
        return np.array(
            [
                self.glucose[y, x],
                self.pyruvate[y, x],
                self.lactate[y, x],
                self.oxygen[y, x],
                self.ammonium[y, x],
                self.nitrate[y, x],
                self.carbon_dioxide[y, x],
                self.proton_load[y, x],
                self.atp_flux[y, x],
            ],
            dtype=np.float32,
        )

    @property
    def redox_balance(self) -> NDArray:
        oxidized = self.oxygen * 1.3 + self.nitrate * 0.7
        reduced = self.lactate * 1.1 + self.carbon_dioxide * 0.35 + self.proton_load * 0.45
        return np.clip(oxidized / (oxidized + reduced + 1e-9), 0.0, 1.0)

    @property
    def acidity(self) -> NDArray:
        return np.clip(self.proton_load / (0.015 + self.proton_load), 0.0, 1.0)


class SoilBiogeochemistry:
    """Layered soil process model wrapped around MolecularWorld.soil."""

    def __init__(
        self,
        world: MolecularWorld,
        rng: np.random.Generator,
        *,
        prefer_rust_substrate: bool = True,
    ):
        self.world = world
        self.rng = rng

        self.moisture = world.soil.surface_moisture
        self.shallow_nutrients = world.soil.shallow_nutrients
        self.deep_minerals = world.soil.deep_minerals
        self.organic_matter = world.soil.organic_matter

        self.deep_moisture = rng.uniform(0.24, 0.58, (world.H, world.W)).astype(np.float32)
        self.litter_carbon = rng.uniform(0.003, 0.035, (world.H, world.W)).astype(np.float32)
        self.microbial_biomass = rng.uniform(0.010, 0.050, (world.H, world.W)).astype(np.float32)
        self.symbiont_biomass = rng.uniform(0.004, 0.020, (world.H, world.W)).astype(np.float32)
        self.root_exudates = np.zeros((world.H, world.W), dtype=np.float32)
        self.dissolved_nutrients = rng.uniform(0.003, 0.016, (world.H, world.W)).astype(np.float32)
        self.mineral_nitrogen = rng.uniform(0.002, 0.010, (world.H, world.W)).astype(np.float32)
        self.soil_structure = rng.uniform(0.35, 0.90, (world.H, world.W)).astype(np.float32)
        self.chemistry = SoilChemistryField.from_world(world, rng)

        self.canopy_cover = np.zeros((world.H, world.W), dtype=np.float32)
        self.root_density = np.zeros((world.H, world.W), dtype=np.float32)
        self._rust_layers = 3
        self.rust_substrate = None
        self._rust_ammonium_layers: NDArray | None = None
        self._rust_nitrate_layers: NDArray | None = None
        self.broad_backend = "rust" if RustStepSoilBroadPools is not None else "python"
        self.uptake_backend = "rust" if RustExtractRootResourcesWithLayers is not None else "python"
        self.substrate_backend = "python"
        self.substrate_time_ms = 0.0
        self.substrate_step_count = 0

        if prefer_rust_substrate and RustBatchedAtomTerrarium is not None:
            try:
                self.rust_substrate = RustBatchedAtomTerrarium(
                    world.W,
                    world.H,
                    self._rust_layers,
                    voxel_size_mm=0.65,
                    use_gpu=True,
                )
                self.substrate_backend = f"rust-{self.rust_substrate.backend}"
                self._push_rust_control_fields(light=float(world._light_intensity()))
                self._pull_rust_chemistry_fields()
            except Exception:
                self.rust_substrate = None
                self.substrate_backend = "python"

    def set_ecology_fields(self, canopy_cover: NDArray, root_density: NDArray) -> None:
        self.canopy_cover[:, :] = canopy_cover
        self.root_density[:, :] = root_density

    @property
    def uses_rust_substrate(self) -> bool:
        return self.rust_substrate is not None

    def _stack_rust_layers(self, surface: NDArray, rhizosphere: NDArray, deep: NDArray) -> NDArray:
        return np.stack((surface, rhizosphere, deep), axis=0).astype(np.float32, copy=False)

    def _push_rust_control_fields(self, *, light: float) -> None:
        if self.rust_substrate is None:
            return

        surface_hydration = np.clip(self.moisture * 0.98, 0.02, 1.4)
        rhizosphere_hydration = np.clip(self.moisture * 0.58 + self.deep_moisture * 0.42, 0.02, 1.4)
        deep_hydration = np.clip(self.deep_moisture * 0.96, 0.02, 1.4)
        hydration = self._stack_rust_layers(surface_hydration, rhizosphere_hydration, deep_hydration)

        litter_drive = self.litter_carbon * 6.5 + self.root_exudates * 8.5 + self.organic_matter * 2.0
        microbial_surface = np.clip(self.microbial_biomass * (0.78 + litter_drive * 0.24), 0.02, 2.0)
        microbial_rhizosphere = np.clip(
            self.microbial_biomass * (0.92 + litter_drive * 0.22)
            + self.symbiont_biomass * 0.55
            + self.root_density * 0.040,
            0.02,
            2.0,
        )
        microbial_deep = np.clip(
            self.microbial_biomass * 0.42 + self.symbiont_biomass * 0.72 + self.deep_moisture * 0.06,
            0.02,
            2.0,
        )
        microbes = self._stack_rust_layers(microbial_surface, microbial_rhizosphere, microbial_deep)

        light_drive = float(_clamp(light, 0.0, 1.5))
        plant_surface = np.clip(self.canopy_cover * (0.22 + light_drive * 0.82), 0.0, 1.5)
        plant_rhizosphere = np.clip(
            self.root_density * 0.92 + self.canopy_cover * 0.10 + self.symbiont_biomass * 0.10,
            0.0,
            1.5,
        )
        plant_deep = np.clip(self.root_density * 0.54 + self.deep_moisture * 0.12, 0.0, 1.5)
        plant_drive = self._stack_rust_layers(plant_surface, plant_rhizosphere, plant_deep)

        self.rust_substrate.set_hydration_field(hydration.reshape(-1).tolist())
        self.rust_substrate.set_microbial_activity_field(microbes.reshape(-1).tolist())
        self.rust_substrate.set_plant_drive_field(plant_drive.reshape(-1).tolist())

    def _rust_species_layers(self, species: str) -> NDArray:
        field = np.asarray(self.rust_substrate.species_field(species), dtype=np.float32)
        return field.reshape((self._rust_layers, self.world.H, self.world.W))

    def _pull_rust_chemistry_fields(self) -> None:
        if self.rust_substrate is None:
            return

        glucose = self._rust_species_layers("glucose")
        oxygen = self._rust_species_layers("oxygen_gas")
        ammonium = self._rust_species_layers("ammonium")
        nitrate = self._rust_species_layers("nitrate")
        carbon_dioxide = self._rust_species_layers("carbon_dioxide")
        proton = self._rust_species_layers("proton")
        atp_flux = self._rust_species_layers("atp_flux")
        self._rust_ammonium_layers = ammonium
        self._rust_nitrate_layers = nitrate

        self.chemistry.glucose[:] = np.clip(glucose[0] * 0.24 + glucose[1] * 0.56 + glucose[2] * 0.20, 0.0, 0.6)
        self.chemistry.pyruvate[:] = np.clip(glucose[1] * 0.10 + atp_flux[1] * 0.008, 0.0, 0.4)
        self.chemistry.lactate[:] = np.clip(glucose[1] * 0.06 + carbon_dioxide[1] * 0.07 + proton[1] * 0.18, 0.0, 0.6)
        self.chemistry.oxygen[:] = np.clip(oxygen[0] * 0.42 + oxygen[1] * 0.38 + oxygen[2] * 0.20, 0.0, 0.15)
        self.chemistry.ammonium[:] = np.clip(ammonium[0] * 0.18 + ammonium[1] * 0.54 + ammonium[2] * 0.28, 0.0, 0.3)
        self.chemistry.nitrate[:] = np.clip(nitrate[0] * 0.20 + nitrate[1] * 0.46 + nitrate[2] * 0.34, 0.0, 0.4)
        self.chemistry.carbon_dioxide[:] = np.clip(
            carbon_dioxide[0] * 0.14 + carbon_dioxide[1] * 0.50 + carbon_dioxide[2] * 0.36,
            0.0,
            0.5,
        )
        self.chemistry.proton_load[:] = np.clip(proton[0] * 0.10 + proton[1] * 0.56 + proton[2] * 0.34, 0.0, 0.25)
        self.chemistry.atp_flux[:] = np.clip(atp_flux[0] * 0.14 + atp_flux[1] * 0.58 + atp_flux[2] * 0.28, 0.0, 4.0)
        self.substrate_backend = f"rust-{self.rust_substrate.backend}"
        self.substrate_time_ms = float(self.rust_substrate.time_ms)
        self.substrate_step_count = int(self.rust_substrate.step_count)

    def _rust_patch_mean(self, species: str, x: int, y: int, z: int, radius: int) -> float:
        if self.rust_substrate is None:
            return 0.0
        return float(self.rust_substrate.patch_mean_species(species, x, y, z, max(1, radius)))

    def _rust_extract_patch(self, species: str, x: int, y: int, z: int, radius: int, amount: float) -> float:
        if self.rust_substrate is None or amount <= 1e-9:
            return 0.0
        return float(self.rust_substrate.extract_patch_species(species, x, y, z, max(1, radius), float(amount)))

    def _couple_broad_pools_from_chemistry(self) -> None:
        chem = self.chemistry
        self.dissolved_nutrients[:] = (
            self.dissolved_nutrients * 0.985
            + (chem.ammonium * 0.18 + chem.pyruvate * 0.05 + chem.glucose * 0.04) * 0.015
        )
        self.mineral_nitrogen[:] = (
            self.mineral_nitrogen * 0.980
            + (chem.ammonium * 0.55 + chem.nitrate * 0.70) * 0.020
        )
        self.shallow_nutrients[:] = (
            self.shallow_nutrients * 0.986
            + (chem.nitrate * 0.24 + chem.ammonium * 0.12) * 0.014
        )

    def _add_rust_hotspots(
        self,
        field: NDArray,
        species: str,
        z: int,
        amplitude_scale: float,
        *,
        threshold: float,
        limit: int = 10,
    ) -> None:
        if self.rust_substrate is None:
            return
        flat = field.reshape(-1)
        active = np.flatnonzero(flat > threshold)
        if active.size == 0:
            return
        take = min(limit, int(active.size))
        top_idx = active[np.argpartition(flat[active], active.size - take)[-take:]]
        for idx in top_idx:
            amplitude = float(flat[idx] * amplitude_scale)
            if amplitude <= 1e-7:
                continue
            y, x = divmod(int(idx), self.world.W)
            self.rust_substrate.add_hotspot(species, x, y, z, amplitude)

    def _step_rust_substrate(
        self,
        dt: float,
        *,
        light: float,
        temp_factor: float,
        litter_used: NDArray,
        exudate_used: NDArray,
        organic_used: NDArray,
        microbial_turnover: NDArray,
        sym_turnover: NDArray,
    ) -> None:
        if self.rust_substrate is None:
            return

        self._push_rust_control_fields(light=light * temp_factor)
        self._add_rust_hotspots(litter_used, "glucose", 0, 28.0, threshold=2e-5, limit=8)
        self._add_rust_hotspots(litter_used, "ammonium", 1, 9.0, threshold=2e-5, limit=8)
        self._add_rust_hotspots(exudate_used, "glucose", 1, 36.0, threshold=1e-5, limit=10)
        self._add_rust_hotspots(exudate_used, "carbon_dioxide", 1, 10.0, threshold=1e-5, limit=8)
        self._add_rust_hotspots(organic_used, "carbon_dioxide", 2, 14.0, threshold=2e-5, limit=8)
        self._add_rust_hotspots(organic_used, "nitrate", 2, 7.0, threshold=2e-5, limit=8)
        self._add_rust_hotspots(microbial_turnover, "ammonium", 1, 18.0, threshold=1e-5, limit=8)
        self._add_rust_hotspots(sym_turnover, "nitrate", 1, 12.0, threshold=1e-5, limit=8)

        microsteps = max(2, min(96, int(round(dt * 1.5))))
        self.rust_substrate.run(microsteps, 0.45 + temp_factor * 0.20)
        self._pull_rust_chemistry_fields()
        self._couple_broad_pools_from_chemistry()

    def _water_source_mask(self) -> NDArray:
        mask = np.zeros((self.world.H, self.world.W), dtype=np.float32)
        for src in getattr(self.world, "water_sources", []):
            if not src.alive:
                continue
            y0, y1, x0, x1 = _window_bounds(self.world.H, self.world.W, src.x, src.y, 2)
            mask[y0:y1, x0:x1] += 1.0
        return mask

    def _diffuse_chemistry(self) -> None:
        for field, rate in (
            (self.chemistry.glucose, 0.026),
            (self.chemistry.pyruvate, 0.021),
            (self.chemistry.lactate, 0.024),
            (self.chemistry.oxygen, 0.045),
            (self.chemistry.ammonium, 0.020),
            (self.chemistry.nitrate, 0.018),
            (self.chemistry.carbon_dioxide, 0.032),
            (self.chemistry.proton_load, 0.014),
        ):
            _diffuse2d(field, rate)

    def _step_chemistry(
        self,
        dt: float,
        *,
        light: float,
        temp_factor: float,
        moisture_factor: NDArray,
        litter_used: NDArray,
        exudate_used: NDArray,
        organic_used: NDArray,
        microbial_turnover: NDArray,
        sym_turnover: NDArray,
    ) -> None:
        chem = self.chemistry
        chem.atp_flux *= 0.0

        aeration = np.clip(1.10 - self.moisture * 0.50 - self.deep_moisture * 0.24 - self.canopy_cover * 0.02, 0.08, 1.0)
        surface_oxygen = 0.040 + light * 0.014
        chem.oxygen += (surface_oxygen - chem.oxygen) * (0.0048 * dt) * aeration
        chem.carbon_dioxide *= 1.0 - (0.0012 * dt) * aeration

        chem.glucose += litter_used * 0.28 + exudate_used * 0.44 + organic_used * 0.08
        chem.pyruvate += exudate_used * 0.06 + organic_used * 0.015
        chem.lactate += exudate_used * 0.05
        chem.ammonium += litter_used * 0.045 + organic_used * 0.028 + microbial_turnover * 0.06 + sym_turnover * 0.03

        biomass_drive = self.microbial_biomass * moisture_factor * temp_factor * (0.84 + self.root_density * 0.06)
        glucose_gate = chem.glucose / (0.015 + chem.glucose)
        glycolysis = np.minimum(
            chem.glucose,
            biomass_drive * glucose_gate * (0.0012 * dt),
        )
        chem.glucose -= glycolysis
        chem.pyruvate += glycolysis * 0.82
        chem.atp_flux += glycolysis * 2.0

        oxygen_gate = chem.oxygen / (0.008 + chem.oxygen)
        oxidation_capacity = biomass_drive * oxygen_gate * (0.00082 * dt)
        oxidation = np.minimum(
            chem.pyruvate,
            oxidation_capacity * (chem.pyruvate / (0.010 + chem.pyruvate)),
        )
        oxidation_o2 = np.minimum(chem.oxygen, oxidation * 0.42)
        oxidation_scale = np.minimum(1.0, oxidation_o2 / (oxidation * 0.42 + 1e-9))
        oxidation *= oxidation_scale
        chem.pyruvate -= oxidation
        chem.oxygen -= oxidation * 0.42
        chem.carbon_dioxide += oxidation * 0.88
        chem.atp_flux += oxidation * 8.5

        anaerobic_gate = np.clip(1.0 - oxygen_gate * 1.10, 0.0, 1.0)
        fermentation = np.minimum(
            chem.pyruvate,
            biomass_drive * anaerobic_gate * (0.00055 * dt) * (chem.pyruvate / (0.010 + chem.pyruvate)),
        )
        chem.pyruvate -= fermentation
        chem.lactate += fermentation * 0.86
        chem.proton_load += fermentation * 0.10
        chem.atp_flux += fermentation * 1.4

        sym_drive = self.symbiont_biomass * moisture_factor * temp_factor * (0.88 + self.root_density * 0.12)
        lactate_gate = chem.lactate / (0.012 + chem.lactate)
        lactate_use = np.minimum(
            chem.lactate,
            sym_drive * oxygen_gate * (0.00064 * dt) * lactate_gate,
        )
        lactate_o2 = np.minimum(chem.oxygen, lactate_use * 0.30)
        lactate_scale = np.minimum(1.0, lactate_o2 / (lactate_use * 0.30 + 1e-9))
        lactate_use *= lactate_scale
        chem.lactate -= lactate_use
        chem.oxygen -= lactate_use * 0.30
        chem.carbon_dioxide += lactate_use * 0.62
        chem.proton_load = np.maximum(0.0, chem.proton_load - lactate_use * 0.06)
        chem.atp_flux += lactate_use * 5.0

        nitrifier_drive = (self.microbial_biomass + self.symbiont_biomass * 0.35) * moisture_factor * temp_factor
        ammonium_gate = chem.ammonium / (0.006 + chem.ammonium)
        nitrification = np.minimum(
            chem.ammonium,
            nitrifier_drive * oxygen_gate * (0.00042 * dt) * ammonium_gate,
        )
        chem.ammonium -= nitrification
        chem.nitrate += nitrification * 0.94
        chem.proton_load += nitrification * 0.08

        nitrate_gate = chem.nitrate / (0.006 + chem.nitrate)
        denitrification = np.minimum(
            chem.nitrate,
            biomass_drive * anaerobic_gate * (0.00018 * dt) * nitrate_gate,
        )
        chem.nitrate -= denitrification
        chem.carbon_dioxide += denitrification * 0.05
        chem.atp_flux += denitrification * 1.8

        self._couple_broad_pools_from_chemistry()

    def step(self, dt: float) -> None:
        light = self.world._light_intensity()
        center_temp = float(self.world.sample_temperature(self.world.W // 2, self.world.H // 2))
        temp_factor = _temp_response(center_temp, optimum=24.0, width=12.0)

        prepared_python_broad = False
        if RustStepSoilBroadPools is None:
            _diffuse2d(self.moisture, 0.045)
            _diffuse2d(self.deep_moisture, 0.012)
            _diffuse2d(self.dissolved_nutrients, 0.020)
            _diffuse2d(self.mineral_nitrogen, 0.015)
            _diffuse2d(self.symbiont_biomass, 0.006)
            prepared_python_broad = True
        if self.rust_substrate is None:
            self._diffuse_chemistry()

        water_mask = self._water_source_mask()
        used_rust_broad = False
        if RustStepSoilBroadPools is not None:
            try:
                broad = RustStepSoilBroadPools(
                    self.world.W,
                    self.world.H,
                    float(dt),
                    float(light),
                    float(temp_factor),
                    water_mask.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.canopy_cover.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.root_density.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.moisture.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.deep_moisture.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.dissolved_nutrients.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.mineral_nitrogen.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.shallow_nutrients.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.deep_minerals.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.organic_matter.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.litter_carbon.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.microbial_biomass.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.symbiont_biomass.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.root_exudates.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.soil_structure.reshape(-1).astype(np.float32, copy=False).tolist(),
                )
                shape = (self.world.H, self.world.W)
                self.moisture[:, :] = np.asarray(broad["moisture"], dtype=np.float32).reshape(shape)
                self.deep_moisture[:, :] = np.asarray(broad["deep_moisture"], dtype=np.float32).reshape(shape)
                self.dissolved_nutrients[:, :] = np.asarray(broad["dissolved_nutrients"], dtype=np.float32).reshape(shape)
                self.mineral_nitrogen[:, :] = np.asarray(broad["mineral_nitrogen"], dtype=np.float32).reshape(shape)
                self.shallow_nutrients[:, :] = np.asarray(broad["shallow_nutrients"], dtype=np.float32).reshape(shape)
                self.deep_minerals[:, :] = np.asarray(broad["deep_minerals"], dtype=np.float32).reshape(shape)
                self.organic_matter[:, :] = np.asarray(broad["organic_matter"], dtype=np.float32).reshape(shape)
                self.litter_carbon[:, :] = np.asarray(broad["litter_carbon"], dtype=np.float32).reshape(shape)
                self.microbial_biomass[:, :] = np.asarray(broad["microbial_biomass"], dtype=np.float32).reshape(shape)
                self.symbiont_biomass[:, :] = np.asarray(broad["symbiont_biomass"], dtype=np.float32).reshape(shape)
                self.root_exudates[:, :] = np.asarray(broad["root_exudates"], dtype=np.float32).reshape(shape)
                decomposition = np.asarray(broad["decomposition"], dtype=np.float32).reshape(shape)
                mineralized = np.asarray(broad["mineralized"], dtype=np.float32).reshape(shape)
                litter_used = np.asarray(broad["litter_used"], dtype=np.float32).reshape(shape)
                exudate_used = np.asarray(broad["exudate_used"], dtype=np.float32).reshape(shape)
                organic_used = np.asarray(broad["organic_used"], dtype=np.float32).reshape(shape)
                microbial_turnover = np.asarray(broad["microbial_turnover"], dtype=np.float32).reshape(shape)
                sym_turnover = np.asarray(broad["sym_turnover"], dtype=np.float32).reshape(shape)
                moisture_factor = np.clip((self.moisture + self.deep_moisture * 0.35) / 0.48, 0.0, 1.6)
                self.broad_backend = "rust"
                used_rust_broad = True
            except Exception:
                if not prepared_python_broad:
                    _diffuse2d(self.moisture, 0.045)
                    _diffuse2d(self.deep_moisture, 0.012)
                    _diffuse2d(self.dissolved_nutrients, 0.020)
                    _diffuse2d(self.mineral_nitrogen, 0.015)
                    _diffuse2d(self.symbiont_biomass, 0.006)
                    prepared_python_broad = True
                self.broad_backend = "python"

        if not used_rust_broad:
            if np.any(water_mask):
                self.moisture += water_mask * (0.00016 * dt)
                self.deep_moisture += water_mask * (0.00008 * dt)

            infiltration = np.maximum(self.moisture - (0.34 + self.soil_structure * 0.14), 0.0)
            infiltration *= (0.0040 * dt) * (0.55 + self.soil_structure)
            self.moisture -= infiltration
            self.deep_moisture += infiltration

            capillary = np.maximum(self.deep_moisture - 0.18, 0.0) * np.maximum(0.30 - self.moisture, 0.0)
            capillary *= 0.012 * dt
            self.deep_moisture -= capillary
            self.moisture += capillary

            canopy_damp = 1.0 - np.clip(self.canopy_cover * 0.40, 0.0, 0.55)
            self.moisture *= 1.0 - dt * 0.000018 * (0.40 + light) * canopy_damp
            self.deep_moisture *= 1.0 - dt * 0.0000035

            weathering = self.deep_minerals * (0.000010 * dt) * (0.45 + self.deep_moisture)
            self.deep_minerals -= weathering
            self.dissolved_nutrients += weathering * 0.34
            self.shallow_nutrients += weathering * 0.30

            substrate = self.litter_carbon * 1.10 + self.root_exudates * 1.35 + self.organic_matter * 0.90
            moisture_factor = np.clip((self.moisture + self.deep_moisture * 0.35) / 0.48, 0.0, 1.6)
            oxygen_factor = np.clip(1.15 - self.deep_moisture * 0.55, 0.35, 1.1)
            root_factor = 1.0 + self.root_density * 0.08

            activity = self.microbial_biomass * substrate * moisture_factor * temp_factor * oxygen_factor * root_factor
            decomposition = activity * (0.00016 * dt)

            litter_used = np.minimum(self.litter_carbon, decomposition * 0.43)
            exudate_used = np.minimum(self.root_exudates, decomposition * 0.30)
            organic_used = np.minimum(self.organic_matter, decomposition * 0.27)

            self.litter_carbon -= litter_used
            self.root_exudates -= exudate_used
            self.organic_matter -= organic_used

            immobilization_demand = self.microbial_biomass * (0.000024 * dt) * (1.0 + self.root_density * 0.05)
            immobilized = np.minimum(self.dissolved_nutrients, immobilization_demand)
            self.dissolved_nutrients -= immobilized

            mineralized = litter_used * 0.17 + exudate_used * 0.24 + organic_used * 0.10
            dissolved = litter_used * 0.21 + exudate_used * 0.15 + organic_used * 0.18
            humified = litter_used * 0.10 + organic_used * 0.06

            microbial_growth = decomposition * 0.085 + immobilized * 0.32
            microbial_turnover = self.microbial_biomass * (0.000038 * dt + np.maximum(0.0, self.deep_moisture - 0.85) * 0.00003 * dt)
            self.microbial_biomass += microbial_growth - microbial_turnover

            symbiont_substrate = self.root_exudates * (0.65 + self.root_density * 0.20)
            sym_growth = symbiont_substrate * (0.00022 * dt) * temp_factor * np.clip(self.moisture / 0.35, 0.3, 1.5)
            sym_turnover = self.symbiont_biomass * (0.000028 * dt)
            self.symbiont_biomass += sym_growth - sym_turnover

            self.mineral_nitrogen += mineralized + sym_turnover * 0.18
            self.dissolved_nutrients += dissolved + weathering * 0.24
            self.shallow_nutrients += dissolved * 0.20 + mineralized * 0.36
            self.litter_carbon += microbial_turnover * 0.42 + sym_turnover * 0.22
            self.organic_matter += humified + microbial_turnover * 0.12
            self.broad_backend = "python"
        if self.rust_substrate is None:
            self._step_chemistry(
                dt,
                light=light,
                temp_factor=temp_factor,
                moisture_factor=moisture_factor,
                litter_used=litter_used,
                exudate_used=exudate_used,
                organic_used=organic_used,
                microbial_turnover=microbial_turnover,
                sym_turnover=sym_turnover,
            )
        else:
            self._step_rust_substrate(
                dt,
                light=light,
                temp_factor=temp_factor,
                litter_used=litter_used,
                exudate_used=exudate_used,
                organic_used=organic_used,
                microbial_turnover=microbial_turnover,
                sym_turnover=sym_turnover,
            )

        if np.any(decomposition > 1e-9):
            self.world.odorant_grids["acetic_acid"][0] += decomposition * 0.10
            self.world.odorant_grids["ammonia"][0] += mineralized * 0.08
            self.world.odorant_grids["ethanol"][0] += exudate_used * 0.05

        np.clip(self.moisture, 0.0, 1.0, out=self.moisture)
        np.clip(self.deep_moisture, 0.02, 1.2, out=self.deep_moisture)
        np.clip(self.litter_carbon, 0.0, 2.0, out=self.litter_carbon)
        np.clip(self.root_exudates, 0.0, 2.0, out=self.root_exudates)
        np.clip(self.dissolved_nutrients, 0.0, 2.0, out=self.dissolved_nutrients)
        np.clip(self.mineral_nitrogen, 0.0, 2.0, out=self.mineral_nitrogen)
        np.clip(self.microbial_biomass, 0.001, 2.0, out=self.microbial_biomass)
        np.clip(self.symbiont_biomass, 0.001, 2.0, out=self.symbiont_biomass)
        np.clip(self.shallow_nutrients, 0.0, 2.0, out=self.shallow_nutrients)
        np.clip(self.deep_minerals, 0.0, 2.0, out=self.deep_minerals)
        np.clip(self.organic_matter, 0.0, 2.0, out=self.organic_matter)
        np.clip(self.chemistry.glucose, 0.0, 0.6, out=self.chemistry.glucose)
        np.clip(self.chemistry.pyruvate, 0.0, 0.4, out=self.chemistry.pyruvate)
        np.clip(self.chemistry.lactate, 0.0, 0.6, out=self.chemistry.lactate)
        np.clip(self.chemistry.oxygen, 0.0, 0.15, out=self.chemistry.oxygen)
        np.clip(self.chemistry.ammonium, 0.0, 0.3, out=self.chemistry.ammonium)
        np.clip(self.chemistry.nitrate, 0.0, 0.4, out=self.chemistry.nitrate)
        np.clip(self.chemistry.carbon_dioxide, 0.0, 0.5, out=self.chemistry.carbon_dioxide)
        np.clip(self.chemistry.proton_load, 0.0, 0.25, out=self.chemistry.proton_load)
        np.clip(self.chemistry.atp_flux, 0.0, 4.0, out=self.chemistry.atp_flux)

    def extract_resources(
        self,
        x: int,
        y: int,
        radius: int,
        water_demand: float,
        nutrient_demand: float,
        *,
        deep_fraction: float = 0.35,
        symbiosis_factor: float = 1.0,
    ) -> tuple[float, float]:
        y0, y1, x0, x1 = _window_bounds(self.world.H, self.world.W, x, y, radius)
        kernel = _radial_kernel(y0, y1, x0, x1, x, y, radius, sharpness=0.8)

        window_m = self.moisture[y0:y1, x0:x1]
        window_dm = self.deep_moisture[y0:y1, x0:x1]
        window_d = self.dissolved_nutrients[y0:y1, x0:x1]
        window_n = self.mineral_nitrogen[y0:y1, x0:x1]
        window_s = self.shallow_nutrients[y0:y1, x0:x1]
        window_p = self.deep_minerals[y0:y1, x0:x1]
        window_f = self.symbiont_biomass[y0:y1, x0:x1]

        if self.rust_substrate is None:
            root_oxygen = float((self.chemistry.oxygen[y0:y1, x0:x1] * kernel).sum())
            root_proton = float((self.chemistry.proton_load[y0:y1, x0:x1] * kernel).sum())
        else:
            root_oxygen = self._rust_patch_mean("oxygen_gas", x, y, 1, radius)
            root_proton = self._rust_patch_mean("proton", x, y, 1, radius)
        root_respiration = _clamp(
            root_oxygen / (0.010 + root_oxygen) * (1.0 - root_proton / (0.040 + root_proton) * 0.24),
            0.18,
            1.15,
        )

        if self.rust_substrate is not None and RustExtractRootResourcesWithLayers is not None:
            try:
                ammonium_layers = self._rust_species_layers("ammonium")
                nitrate_layers = self._rust_species_layers("nitrate")
                extracted = RustExtractRootResourcesWithLayers(
                    self.world.W,
                    self.world.H,
                    int(x),
                    int(y),
                    int(max(1, radius)),
                    float(water_demand),
                    float(nutrient_demand),
                    float(deep_fraction),
                    float(symbiosis_factor),
                    float(root_respiration),
                    self.moisture.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.deep_moisture.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.dissolved_nutrients.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.mineral_nitrogen.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.shallow_nutrients.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.deep_minerals.reshape(-1).astype(np.float32, copy=False).tolist(),
                    self.symbiont_biomass.reshape(-1).astype(np.float32, copy=False).tolist(),
                    ammonium_layers[1].reshape(-1).astype(np.float32, copy=False).tolist(),
                    nitrate_layers[1].reshape(-1).astype(np.float32, copy=False).tolist(),
                    nitrate_layers[2].reshape(-1).astype(np.float32, copy=False).tolist(),
                )
                shape = (self.world.H, self.world.W)
                self.moisture[:, :] = np.asarray(extracted["moisture"], dtype=np.float32).reshape(shape)
                self.deep_moisture[:, :] = np.asarray(extracted["deep_moisture"], dtype=np.float32).reshape(shape)
                self.dissolved_nutrients[:, :] = np.asarray(extracted["dissolved_nutrients"], dtype=np.float32).reshape(shape)
                self.mineral_nitrogen[:, :] = np.asarray(extracted["mineral_nitrogen"], dtype=np.float32).reshape(shape)
                self.shallow_nutrients[:, :] = np.asarray(extracted["shallow_nutrients"], dtype=np.float32).reshape(shape)
                self.deep_minerals[:, :] = np.asarray(extracted["deep_minerals"], dtype=np.float32).reshape(shape)
                water_take = float(extracted["water_take"])
                nutrient_take = float(extracted["nutrient_take"])
                surface_take = float(extracted["surface_water_take"])
                deep_take = float(extracted["deep_water_take"])
                ammonium_take = float(extracted["ammonium_take"])
                rhizo_nitrate_take = float(extracted["rhizo_nitrate_take"])
                deep_nitrate_take = float(extracted["deep_nitrate_take"])
                if surface_take > 0.0 or deep_take > 0.0:
                    self._rust_extract_patch("water", x, y, 0, radius, surface_take * 0.55)
                    self._rust_extract_patch("water", x, y, 1, radius, surface_take * 0.18 + deep_take * 0.14)
                    self._rust_extract_patch("water", x, y, 2, radius, deep_take * 0.48)
                if ammonium_take > 0.0:
                    self._rust_extract_patch("ammonium", x, y, 1, radius, ammonium_take)
                if rhizo_nitrate_take > 0.0:
                    self._rust_extract_patch("nitrate", x, y, 1, radius, rhizo_nitrate_take)
                if deep_nitrate_take > 0.0:
                    self._rust_extract_patch("nitrate", x, y, 2, radius, deep_nitrate_take)
                self.uptake_backend = "rust"
                return float(max(water_take, 0.0)), float(max(nutrient_take, 0.0))
            except Exception:
                self.uptake_backend = "python"
                pass

        surface_avail = window_m * kernel
        deep_avail = window_dm * kernel * float(_clamp(deep_fraction, 0.0, 1.2))
        surface_total = float(surface_avail.sum())
        deep_total = float(deep_avail.sum())
        water_take = min(surface_total + deep_total, water_demand * (0.58 + root_respiration * 0.42))

        if water_take > 0.0:
            water_total = surface_total + deep_total
            surface_share = surface_total / max(water_total, 1e-9)
            take_surface = water_take * surface_share
            take_deep = water_take - take_surface
            if surface_total > 1e-9:
                window_m -= take_surface * (surface_avail / surface_total)
            if deep_total > 1e-9:
                window_dm -= take_deep * (deep_avail / deep_total)
            if self.rust_substrate is not None:
                self._rust_extract_patch("water", x, y, 0, radius, take_surface * 0.55)
                self._rust_extract_patch("water", x, y, 1, radius, take_surface * 0.18 + take_deep * 0.14)
                self._rust_extract_patch("water", x, y, 2, radius, take_deep * 0.48)

        fungal_bonus = 1.0 + float((window_f * kernel).sum()) * 9.0 * max(symbiosis_factor, 0.1)
        dissolved_avail = window_d * kernel
        mineral_avail = window_n * kernel
        shallow_avail = window_s * kernel * 0.28
        deep_mineral_avail = window_p * kernel * (0.04 + 0.16 * _clamp(deep_fraction, 0.0, 1.0)) * fungal_bonus

        dissolved_total = float(dissolved_avail.sum())
        mineral_total = float(mineral_avail.sum())
        shallow_total = float(shallow_avail.sum())
        deep_mineral_total = float(deep_mineral_avail.sum())
        nutrient_budget = nutrient_demand * fungal_bonus * root_respiration
        rust_chem_take = 0.0

        if self.rust_substrate is None:
            window_a = self.chemistry.ammonium[y0:y1, x0:x1]
            window_no3 = self.chemistry.nitrate[y0:y1, x0:x1]
            ammonium_avail = window_a * kernel * (0.55 + fungal_bonus * 0.08)
            nitrate_avail = window_no3 * kernel * (0.72 + fungal_bonus * 0.12)
            ammonium_total = float(ammonium_avail.sum())
            nitrate_total = float(nitrate_avail.sum())
            available = dissolved_total + mineral_total + shallow_total + deep_mineral_total + ammonium_total + nitrate_total
            nutrient_take = min(available, nutrient_budget)

            if nutrient_take > 0.0 and available > 1e-9:
                take_d = nutrient_take * (dissolved_total / available)
                take_n = nutrient_take * (mineral_total / available)
                take_s = nutrient_take * (shallow_total / available)
                take_p = nutrient_take * (deep_mineral_total / available)
                take_a = nutrient_take * (ammonium_total / available)
                take_no3 = nutrient_take * (nitrate_total / available)

                if dissolved_total > 1e-9:
                    window_d -= take_d * (dissolved_avail / dissolved_total)
                if mineral_total > 1e-9:
                    window_n -= take_n * (mineral_avail / mineral_total)
                if shallow_total > 1e-9:
                    window_s -= take_s * ((window_s * kernel) / max(float((window_s * kernel).sum()), 1e-9))
                if deep_mineral_total > 1e-9:
                    window_p -= take_p * (deep_mineral_avail / deep_mineral_total)
                if ammonium_total > 1e-9:
                    window_a -= take_a * (ammonium_avail / ammonium_total)
                if nitrate_total > 1e-9:
                    window_no3 -= take_no3 * (nitrate_avail / nitrate_total)
            self.uptake_backend = "python"
            return float(max(water_take, 0.0)), float(max(nutrient_take, 0.0))

        if nutrient_budget > 0.0:
            local_ammonium = self._rust_patch_mean("ammonium", x, y, 1, radius)
            ammonium_bias = _clamp(0.32 + local_ammonium * 3.2, 0.18, 0.72)
            ammonium_quota = nutrient_budget * ammonium_bias
            rust_chem_take += self._rust_extract_patch("ammonium", x, y, 1, radius, ammonium_quota)
            remaining = max(0.0, nutrient_budget - rust_chem_take)
            rhizo_nitrate_quota = remaining * _clamp(0.54 - deep_fraction * 0.16, 0.20, 0.58)
            rust_chem_take += self._rust_extract_patch("nitrate", x, y, 1, radius, rhizo_nitrate_quota)
            remaining = max(0.0, nutrient_budget - rust_chem_take)
            rust_chem_take += self._rust_extract_patch("nitrate", x, y, 2, radius, remaining * (0.48 + deep_fraction * 0.24))

        available = dissolved_total + mineral_total + shallow_total + deep_mineral_total
        mineral_take = min(available, max(0.0, nutrient_budget - rust_chem_take))
        nutrient_take = rust_chem_take + mineral_take

        if mineral_take > 0.0 and available > 1e-9:
            take_d = mineral_take * (dissolved_total / available)
            take_n = mineral_take * (mineral_total / available)
            take_s = mineral_take * (shallow_total / available)
            take_p = mineral_take * (deep_mineral_total / available)

            if dissolved_total > 1e-9:
                window_d -= take_d * (dissolved_avail / dissolved_total)
            if mineral_total > 1e-9:
                window_n -= take_n * (mineral_avail / mineral_total)
            if shallow_total > 1e-9:
                window_s -= take_s * ((window_s * kernel) / max(float((window_s * kernel).sum()), 1e-9))
            if deep_mineral_total > 1e-9:
                window_p -= take_p * (deep_mineral_avail / deep_mineral_total)

        self.uptake_backend = "python"
        return float(max(water_take, 0.0)), float(max(nutrient_take, 0.0))

    def _deposit_with_kernel(self, field: NDArray, x: int, y: int, radius: int, amount: float, *, sharpness: float = 0.65) -> None:
        if amount == 0.0:
            return
        if RustBuildRadialField is not None:
            try:
                overlay = RustBuildRadialField(
                    self.world.W,
                    self.world.H,
                    [float(x), float(y), float(radius), float(amount), float(sharpness)],
                )
                field += np.asarray(overlay, dtype=np.float32).reshape((self.world.H, self.world.W))
                return
            except Exception:
                pass
        y0, y1, x0, x1 = _window_bounds(self.world.H, self.world.W, x, y, radius)
        kernel = _radial_kernel(y0, y1, x0, x1, x, y, radius, sharpness=sharpness)
        field[y0:y1, x0:x1] += kernel * amount

    def deposit_litter(self, x: int, y: int, radius: int, amount: float) -> None:
        self._deposit_with_kernel(self.litter_carbon, x, y, radius, amount, sharpness=0.70)
        self._deposit_with_kernel(self.chemistry.glucose, x, y, radius, amount * 0.10, sharpness=0.72)
        self._deposit_with_kernel(self.chemistry.ammonium, x, y, radius, amount * 0.02, sharpness=0.72)
        if self.rust_substrate is not None:
            self.rust_substrate.add_hotspot("glucose", x, y, 0, max(0.0, amount * 0.22))
            self.rust_substrate.add_hotspot("ammonium", x, y, 1, max(0.0, amount * 0.06))

    def deposit_root_exudates(self, x: int, y: int, radius: int, amount: float) -> None:
        self._deposit_with_kernel(self.root_exudates, x, y, radius, amount, sharpness=0.95)
        self._deposit_with_kernel(self.chemistry.glucose, x, y, radius, amount * 0.26, sharpness=0.95)
        self._deposit_with_kernel(self.chemistry.lactate, x, y, radius, amount * 0.10, sharpness=0.92)
        self._deposit_with_kernel(self.chemistry.pyruvate, x, y, radius, amount * 0.04, sharpness=0.90)
        if self.rust_substrate is not None:
            self.rust_substrate.add_hotspot("glucose", x, y, 1, max(0.0, amount * 0.40))
            self.rust_substrate.add_hotspot("carbon_dioxide", x, y, 1, max(0.0, amount * 0.09))

    def deposit_fruit_detritus(self, x: int, y: int, radius: int, amount: float) -> None:
        if amount <= 0.0:
            return
        self._deposit_with_kernel(self.litter_carbon, x, y, radius, amount * 0.48, sharpness=0.8)
        self._deposit_with_kernel(self.organic_matter, x, y, radius, amount * 0.24, sharpness=0.8)
        self._deposit_with_kernel(self.dissolved_nutrients, x, y, radius, amount * 0.16, sharpness=0.9)
        self._deposit_with_kernel(self.root_exudates, x, y, radius, amount * 0.08, sharpness=0.95)
        self._deposit_with_kernel(self.microbial_biomass, x, y, radius, amount * 0.03, sharpness=0.8)
        self._deposit_with_kernel(self.symbiont_biomass, x, y, radius, amount * 0.01, sharpness=0.9)
        self._deposit_with_kernel(self.chemistry.glucose, x, y, radius, amount * 0.24, sharpness=0.86)
        self._deposit_with_kernel(self.chemistry.lactate, x, y, radius, amount * 0.06, sharpness=0.88)
        self._deposit_with_kernel(self.chemistry.ammonium, x, y, radius, amount * 0.04, sharpness=0.82)
        if self.rust_substrate is not None:
            self.rust_substrate.add_hotspot("glucose", x, y, 0, amount * 0.38)
            self.rust_substrate.add_hotspot("carbon_dioxide", x, y, 1, amount * 0.12)
            self.rust_substrate.add_hotspot("ammonium", x, y, 1, amount * 0.09)
        self.world.odorant_grids["ethanol"][0, y, x] += amount * 0.18
        self.world.odorant_grids["acetic_acid"][0, y, x] += amount * 0.10

    def chemistry_vector_at(self, x: int, y: int) -> NDArray:
        xi = int(np.clip(x, 0, self.world.W - 1))
        yi = int(np.clip(y, 0, self.world.H - 1))
        return self.chemistry.vector_at(xi, yi)

    def mean_microbial_biomass(self) -> float:
        return float(self.microbial_biomass.mean())


@dataclass
class _FoodPatchState:
    remaining: float
    deposited_all: bool = False


class TerrariumEcology:
    """Reusable ecology manager for MolecularWorld terraria."""

    def __init__(self, world: MolecularWorld, seed: int = 0, *, prefer_rust_substrate: bool = True):
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.soil = SoilBiogeochemistry(world, self.rng, prefer_rust_substrate=prefer_rust_substrate)
        self.plants: list[PlantOrganism] = []
        self.seed_bank: list[SeedPropagule] = []
        self.canopy_cover = np.zeros((world.H, world.W), dtype=np.float32)
        self.root_density = np.zeros((world.H, world.W), dtype=np.float32)
        self._food_patch_state: list[_FoodPatchState] = []
        self.field_backend = "rust" if RustBuildDualRadialFields is not None else "python"
        self.food_backend = "rust" if RustStepFoodPatches is not None else "python"
        self.seed_backend = "rust" if RustStepSeedBank is not None else "python"

    def _add_radial_influence(self, field: NDArray, x: int, y: int, radius: int, amplitude: float, *, sharpness: float) -> None:
        y0, y1, x0, x1 = _window_bounds(self.world.H, self.world.W, x, y, radius)
        kernel = _radial_kernel(y0, y1, x0, x1, x, y, radius, sharpness=sharpness)
        field[y0:y1, x0:x1] += kernel * amplitude

    def _rebuild_ecology_fields(self) -> None:
        self.canopy_cover.fill(0.0)
        self.root_density.fill(0.0)
        if RustBuildDualRadialFields is not None and self.plants:
            canopy_sources: list[float] = []
            root_sources: list[float] = []
            for plant in self.plants:
                x = int(np.clip(plant.source.x, 0, self.world.W - 1))
                y = int(np.clip(plant.source.y, 0, self.world.H - 1))
                canopy_sources.extend(
                    (
                        float(x),
                        float(y),
                        float(plant.canopy_radius_cells),
                        float(plant.leaf_biomass * (1.2 + plant.source.height * 0.04)),
                        0.72,
                    )
                )
                root_sources.extend(
                    (
                        float(x),
                        float(y),
                        float(plant.root_radius_cells),
                        float(plant.root_biomass * (1.0 + plant.genome.root_depth_bias * 0.45)),
                        0.95,
                    )
                )
            try:
                canopy, root = RustBuildDualRadialFields(self.world.W, self.world.H, canopy_sources, root_sources)
                self.canopy_cover[:, :] = np.asarray(canopy, dtype=np.float32).reshape((self.world.H, self.world.W))
                self.root_density[:, :] = np.asarray(root, dtype=np.float32).reshape((self.world.H, self.world.W))
                self.field_backend = "rust"
                self.soil.set_ecology_fields(self.canopy_cover, self.root_density)
                return
            except Exception:
                self.field_backend = "python"
        for plant in self.plants:
            x = int(np.clip(plant.source.x, 0, self.world.W - 1))
            y = int(np.clip(plant.source.y, 0, self.world.H - 1))
            self._add_radial_influence(
                self.canopy_cover,
                x,
                y,
                plant.canopy_radius_cells,
                plant.leaf_biomass * (1.2 + plant.source.height * 0.04),
                sharpness=0.72,
            )
            self._add_radial_influence(
                self.root_density,
                x,
                y,
                plant.root_radius_cells,
                plant.root_biomass * (1.0 + plant.genome.root_depth_bias * 0.45),
                sharpness=0.95,
            )
        if RustBuildDualRadialFields is None:
            self.field_backend = "python"
        self.soil.set_ecology_fields(self.canopy_cover, self.root_density)

    def microhabitat_at(self, plant: PlantOrganism) -> dict[str, float]:
        x = int(np.clip(plant.source.x, 0, self.world.W - 1))
        y = int(np.clip(plant.source.y, 0, self.world.H - 1))
        canopy_self = plant.leaf_biomass * (1.2 + plant.source.height * 0.04)
        root_self = plant.root_biomass * (1.0 + plant.genome.root_depth_bias * 0.45)
        if self.soil.rust_substrate is None:
            soil_redox = float(self.soil.chemistry.redox_balance[y, x])
            soil_glucose = float(self.soil.chemistry.glucose[y, x])
            soil_ammonium = float(self.soil.chemistry.ammonium[y, x])
            soil_nitrate = float(self.soil.chemistry.nitrate[y, x])
        else:
            radius = plant.root_radius_cells
            oxygen = self.soil._rust_patch_mean("oxygen_gas", x, y, 1, radius)
            proton = self.soil._rust_patch_mean("proton", x, y, 1, radius)
            nitrate = self.soil._rust_patch_mean("nitrate", x, y, 1, radius)
            carbon_dioxide = self.soil._rust_patch_mean("carbon_dioxide", x, y, 1, radius)
            soil_redox = float(
                _clamp(
                    (oxygen * 1.3 + nitrate * 0.7)
                    / (oxygen * 1.3 + nitrate * 0.7 + carbon_dioxide * 0.35 + proton * 0.45 + 1e-9),
                    0.0,
                    1.0,
                )
            )
            soil_glucose = self.soil._rust_patch_mean("glucose", x, y, 1, radius)
            soil_ammonium = self.soil._rust_patch_mean("ammonium", x, y, 1, radius)
            soil_nitrate = self.soil._rust_patch_mean("nitrate", x, y, 1, radius)
        return {
            "canopy_competition": max(0.0, float(self.canopy_cover[y, x]) - canopy_self),
            "root_competition": max(0.0, float(self.root_density[y, x]) - root_self),
            "symbionts": float(self.soil.symbiont_biomass[y, x]),
            "deep_moisture": float(self.soil.deep_moisture[y, x]),
            "litter": float(self.soil.litter_carbon[y, x]),
            "soil_redox": float(soil_redox),
            "soil_glucose": float(soil_glucose),
            "soil_ammonium": float(soil_ammonium),
            "soil_nitrate": float(soil_nitrate),
        }

    def add_plant(
        self,
        x: int,
        y: int,
        *,
        genome: Optional[PlantGenome] = None,
        biomass_scale: Optional[float] = None,
    ) -> PlantOrganism:
        genome = genome if genome is not None else PlantGenome.sample(self.rng)
        if biomass_scale is None:
            biomass_scale = float(self.rng.uniform(0.75, 1.25) * (0.75 + genome.seed_mass * 4.5))
        src = self.world.add_plant(int(x), int(y), height=6.0, emission_rate=0.02)
        rust_state = None
        if RustPlantOrganism is not None:
            rust_state = RustPlantOrganism(
                genome.max_height_mm,
                genome.canopy_radius_mm,
                genome.root_radius_mm,
                genome.leaf_efficiency,
                genome.root_uptake_efficiency,
                genome.water_use_efficiency,
                genome.volatile_scale,
                genome.fruiting_threshold,
                genome.litter_turnover,
                genome.shade_tolerance,
                genome.root_depth_bias,
                genome.symbiosis_affinity,
                genome.seed_mass,
                0.14 * biomass_scale,
                0.18 * biomass_scale,
                0.17 * biomass_scale,
                0.12 * biomass_scale,
                0.22 + genome.seed_mass * 0.4,
                0.11 + genome.seed_mass * 0.2,
                float(self.rng.uniform(7200.0, 16000.0)),
                float(self.rng.uniform(14000.0, 30000.0)),
            )
        plant = PlantOrganism(
            source=src,
            genome=genome,
            leaf_biomass=0.14 * biomass_scale,
            stem_biomass=0.18 * biomass_scale,
            root_biomass=0.17 * biomass_scale,
            storage_carbon=0.12 * biomass_scale,
            water_buffer=0.22 + genome.seed_mass * 0.4,
            nitrogen_buffer=0.11 + genome.seed_mass * 0.2,
            fruit_timer_s=float(rust_state.fruit_timer_s) if rust_state is not None else float(self.rng.uniform(7200.0, 16000.0)),
            seed_timer_s=float(rust_state.seed_timer_s) if rust_state is not None else float(self.rng.uniform(14000.0, 30000.0)),
            cellular_state=PlantCellularState.from_genome(genome, biomass_scale),
            rust_state=rust_state,
        )
        if rust_state is not None:
            plant._sync_from_rust()
        self.plants.append(plant)
        self._rebuild_ecology_fields()
        return plant

    def spawn_fruit(self, x: int, y: int, size: float = 1.0, parent: Optional[PlantOrganism] = None) -> None:
        fx = int(np.clip(x + self.rng.integers(-2, 3), 1, self.world.W - 2))
        fy = int(np.clip(y + self.rng.integers(-2, 3), 1, self.world.H - 2))
        radius = float(_clamp(2.0 + size, 2.0, 4.2))
        remaining = float(_clamp(0.32 + size, 0.32, 1.9))
        self.world.add_food(fx, fy, radius=radius, remaining=remaining)
        fruit = self.world.fruit_sources[-1]
        volatile_scale = parent.genome.volatile_scale if parent is not None else 1.0
        fruit.odorant_emission_rate = 0.05 + size * 0.10 * volatile_scale
        fruit.ripeness = 0.88
        fruit.decay_rate = 0.00025 + size * 0.00003
        self._ensure_patch_tracking()

    def spawn_seed_from(self, plant: PlantOrganism) -> None:
        if len(self.seed_bank) > 64:
            return
        dispersal = max(2, int(round(7.0 - plant.genome.seed_mass * 18.0 + plant.health * 1.5)))
        x = float(np.clip(plant.source.x + self.rng.integers(-dispersal, dispersal + 1), 1, self.world.W - 2))
        y = float(np.clip(plant.source.y + self.rng.integers(-dispersal, dispersal + 1), 1, self.world.H - 2))
        dormancy = float(self.rng.uniform(9000.0, 26000.0) * (1.18 - plant.genome.seed_mass * 1.5))
        genome = plant.genome.mutate(self.rng)
        reserve = float(_clamp(plant.genome.seed_mass * self.rng.uniform(0.85, 1.25) + max(0.0, plant.storage_carbon) * 0.05, 0.03, 0.28))
        self.seed_bank.append(SeedPropagule(x=x, y=y, dormancy_s=dormancy, genome=genome, reserve_carbon=reserve))

    def _ensure_patch_tracking(self) -> None:
        foods = getattr(self.world, "food_patches", [])
        while len(self._food_patch_state) < len(foods):
            self._food_patch_state.append(_FoodPatchState(remaining=float(foods[len(self._food_patch_state)]["remaining"])))
        if len(self._food_patch_state) > len(foods):
            self._food_patch_state = self._food_patch_state[:len(foods)]

    def sync_external_food_state(self) -> None:
        """Align tracked food-patch state after external consumption.

        Flies may consume food through the shared Rust sensory field between
        ecology ticks. The Python food web tracker needs the new remaining mass
        as its baseline so that fly consumption is not reclassified as decay
        detritus on the next ecology step.
        """
        foods = getattr(self.world, "food_patches", [])
        fruits = getattr(self.world, "fruit_sources", [])
        self._ensure_patch_tracking()
        for idx, patch in enumerate(foods):
            current = float(patch["remaining"])
            state = self._food_patch_state[idx]
            state.remaining = current
            state.deposited_all = state.deposited_all or current <= 0.01
            if idx < len(fruits):
                fruits[idx].sugar_content = min(float(fruits[idx].sugar_content), current)
                if current <= 0.01:
                    fruits[idx].alive = False

    def _step_food_web(self, dt: float) -> None:
        foods = getattr(self.world, "food_patches", [])
        fruits = getattr(self.world, "fruit_sources", [])
        self._ensure_patch_tracking()

        if not foods:
            self.food_backend = "rust-idle" if RustStepFoodPatches is not None else "python-idle"
            return

        if foods and RustStepFoodPatches is not None:
            try:
                remaining_in: list[float] = []
                previous_in: list[float] = []
                deposited_in: list[bool] = []
                has_fruit_in: list[bool] = []
                ripeness_in: list[float] = []
                sugar_in: list[float] = []
                microbes_in: list[float] = []
                coords: list[tuple[int, int, int]] = []

                for idx, patch in enumerate(foods):
                    state = self._food_patch_state[idx]
                    x = int(np.clip(round(patch["x"]), 0, self.world.W - 1))
                    y = int(np.clip(round(patch["y"]), 0, self.world.H - 1))
                    radius = max(1, int(round(patch["radius"])))
                    fruit = fruits[idx] if idx < len(fruits) else None
                    coords.append((x, y, radius))
                    remaining_in.append(float(patch["remaining"]))
                    previous_in.append(float(state.remaining))
                    deposited_in.append(bool(state.deposited_all))
                    has_fruit_in.append(fruit is not None)
                    ripeness_in.append(float(fruit.ripeness) if fruit is not None else 0.0)
                    sugar_in.append(float(fruit.sugar_content) if fruit is not None else 0.0)
                    microbes_in.append(float(self.soil.microbial_biomass[y, x]))

                stepped = RustStepFoodPatches(
                    float(dt),
                    remaining_in,
                    previous_in,
                    deposited_in,
                    has_fruit_in,
                    ripeness_in,
                    sugar_in,
                    microbes_in,
                )

                for idx, patch in enumerate(foods):
                    current = float(stepped["remaining"][idx])
                    patch["remaining"] = current
                    state = self._food_patch_state[idx]
                    state.remaining = current
                    state.deposited_all = bool(stepped["deposited_all"][idx])
                    x, y, radius = coords[idx]
                    fruit = fruits[idx] if idx < len(fruits) else None
                    if fruit is not None:
                        fruit.sugar_content = min(float(stepped["sugar_content"][idx]), current)
                        fruit.alive = bool(stepped["fruit_alive"][idx])
                    decay_detritus = float(stepped["decay_detritus"][idx])
                    lost_detritus = float(stepped["lost_detritus"][idx])
                    final_detritus = float(stepped["final_detritus"][idx])
                    if decay_detritus > 0.0:
                        self.soil.deposit_fruit_detritus(x, y, radius, decay_detritus)
                    if lost_detritus > 0.0:
                        self.soil.deposit_fruit_detritus(x, y, radius, lost_detritus)
                    if final_detritus > 0.0:
                        self.soil.deposit_fruit_detritus(x, y, radius, final_detritus)
                self.food_backend = "rust"
                return
            except Exception:
                self.food_backend = "python"

        for idx, patch in enumerate(foods):
            state = self._food_patch_state[idx]
            current = float(patch["remaining"])
            x = int(np.clip(round(patch["x"]), 0, self.world.W - 1))
            y = int(np.clip(round(patch["y"]), 0, self.world.H - 1))
            radius = max(1, int(round(patch["radius"])))
            microbes = float(self.soil.microbial_biomass[y, x])

            fruit = fruits[idx] if idx < len(fruits) else None
            if fruit is not None:
                fruit.sugar_content = min(fruit.sugar_content, current)

                if current > 0.0:
                    decay_driver = max(0.0, fruit.ripeness - 0.82) + max(0.0, 0.02 - current) * 0.2
                    decay_loss = min(current, decay_driver * (0.00008 * dt) * (1.0 + microbes * 8.0))
                    if decay_loss > 0.0:
                        patch["remaining"] = max(0.0, current - decay_loss)
                        current = float(patch["remaining"])
                        self.soil.deposit_fruit_detritus(x, y, radius, decay_loss)
                        fruit.sugar_content = min(fruit.sugar_content, current)

                if current <= 0.01:
                    fruit.alive = False

            lost = max(0.0, state.remaining - current)
            if lost > 0.0:
                self.soil.deposit_fruit_detritus(x, y, radius, lost * 0.65)

            if current <= 0.01 and not state.deposited_all:
                self.soil.deposit_fruit_detritus(x, y, radius, 0.03)
                state.deposited_all = True

            state.remaining = current
        self.food_backend = "python"

    def _step_seeds(self, dt: float) -> None:
        if not self.seed_bank:
            self.seed_backend = "rust-idle" if RustStepSeedBank is not None else "python-idle"
            return

        if self.seed_bank and RustStepSeedBank is not None:
            try:
                light = float(self.world._light_intensity())
                dormancy_in: list[float] = []
                age_in: list[float] = []
                reserve_in: list[float] = []
                affinity_in: list[float] = []
                shade_in: list[float] = []
                moisture_in: list[float] = []
                deep_moisture_in: list[float] = []
                nutrients_in: list[float] = []
                sym_in: list[float] = []
                canopy_in: list[float] = []
                litter_in: list[float] = []
                positions: list[tuple[int, int]] = []

                for seed in self.seed_bank:
                    xi = int(np.clip(seed.x, 0, self.world.W - 1))
                    yi = int(np.clip(seed.y, 0, self.world.H - 1))
                    positions.append((xi, yi))
                    dormancy_in.append(float(seed.dormancy_s))
                    age_in.append(float(seed.age_s))
                    reserve_in.append(float(seed.reserve_carbon))
                    affinity_in.append(float(seed.genome.symbiosis_affinity))
                    shade_in.append(float(seed.genome.shade_tolerance))
                    moisture_in.append(float(self.soil.moisture[yi, xi]))
                    deep_moisture_in.append(float(self.soil.deep_moisture[yi, xi]))
                    nutrients_in.append(float(self.soil.shallow_nutrients[yi, xi] + self.soil.dissolved_nutrients[yi, xi]))
                    sym_in.append(float(self.soil.symbiont_biomass[yi, xi]))
                    canopy_in.append(float(self.canopy_cover[yi, xi]))
                    litter_in.append(float(self.soil.litter_carbon[yi, xi]))

                stepped = RustStepSeedBank(
                    float(dt),
                    light,
                    int(len(self.plants)),
                    28,
                    dormancy_in,
                    age_in,
                    reserve_in,
                    affinity_in,
                    shade_in,
                    moisture_in,
                    deep_moisture_in,
                    nutrients_in,
                    sym_in,
                    canopy_in,
                    litter_in,
                )

                next_bank: list[SeedPropagule] = []
                for idx, seed in enumerate(self.seed_bank):
                    seed.age_s = float(stepped["age_s"][idx])
                    seed.dormancy_s = float(stepped["dormancy_s"][idx])
                    xi, yi = positions[idx]
                    if bool(stepped["germinate"][idx]):
                        self.add_plant(xi, yi, genome=seed.genome, biomass_scale=float(stepped["seedling_scale"][idx]))
                    elif bool(stepped["keep"][idx]):
                        next_bank.append(seed)
                self.seed_bank = next_bank
                self.seed_backend = "rust"
                return
            except Exception:
                self.seed_backend = "python"

        next_bank: list[SeedPropagule] = []
        for seed in self.seed_bank:
            seed.age_s += dt
            xi = int(np.clip(seed.x, 0, self.world.W - 1))
            yi = int(np.clip(seed.y, 0, self.world.H - 1))
            dormancy_loss = dt * (0.55 + self.soil.moisture[yi, xi] * 0.9 + self.world._light_intensity() * 0.15)
            seed.dormancy_s -= dormancy_loss

            if seed.dormancy_s > 0.0:
                next_bank.append(seed)
                continue

            moist = float(self.soil.moisture[yi, xi])
            deep_moist = float(self.soil.deep_moisture[yi, xi])
            nutr = float(self.soil.shallow_nutrients[yi, xi] + self.soil.dissolved_nutrients[yi, xi])
            sym = float(self.soil.symbiont_biomass[yi, xi])
            canopy = float(self.canopy_cover[yi, xi])
            litter = float(self.soil.litter_carbon[yi, xi])

            germination_score = (
                moist * 1.2
                + deep_moist * 0.5
                + nutr * 4.0
                + sym * 6.0 * seed.genome.symbiosis_affinity
                + litter * 1.6
                + seed.reserve_carbon * 2.0
                - canopy * max(0.18, 1.0 - seed.genome.shade_tolerance)
            )

            if germination_score > 0.72 and len(self.plants) < 28:
                seedling_scale = float(_clamp(0.50 + seed.reserve_carbon * 2.8, 0.45, 1.10))
                self.add_plant(xi, yi, genome=seed.genome, biomass_scale=seedling_scale)
            elif seed.dormancy_s > -28000.0 and seed.age_s < 200000.0:
                next_bank.append(seed)

        self.seed_bank = next_bank
        self.seed_backend = "python"

    def step(self, dt: float) -> None:
        self._rebuild_ecology_fields()
        self.soil.step(dt)
        self._step_food_web(dt)
        self._step_seeds(dt)

        for plant in list(self.plants):
            plant.step(self, dt)
            if plant.is_dead():
                x = int(np.clip(plant.source.x, 0, self.world.W - 1))
                y = int(np.clip(plant.source.y, 0, self.world.H - 1))
                self.soil.deposit_litter(x, y, plant.root_radius_cells, max(0.0, plant.total_biomass) * 0.80)
                self.soil.deposit_fruit_detritus(x, y, max(1, plant.root_radius_cells // 2), 0.02 + max(0.0, plant.storage_carbon) * 0.2)
                plant.source.alive = False
                self.plants.remove(plant)

        self._rebuild_ecology_fields()

    def stats(self) -> dict[str, float | str]:
        total_cells = sum(plant.total_cells for plant in self.plants)
        mean_cell_vitality = (
            sum(plant.cell_vitality for plant in self.plants) / len(self.plants)
            if self.plants
            else 0.0
        )
        mean_cell_energy = (
            sum(plant.cellular_energy_charge for plant in self.plants) / len(self.plants)
            if self.plants
            else 0.0
        )
        mean_division = (
            sum(plant.cellular_division_pressure for plant in self.plants) / len(self.plants)
            if self.plants
            else 0.0
        )
        cell_backend = self.plants[0].cellular_state.metabolism_backend if self.plants else (
            "rust" if RustCellularMetabolism is not None else "python"
        )
        plant_backend = self.plants[0].physiology_backend if self.plants else (
            "rust" if RustPlantOrganism is not None else "python"
        )
        return {
            "plant_count": float(len(self.plants)),
            "seed_bank": float(len(self.seed_bank)),
            "mean_microbes": self.soil.mean_microbial_biomass(),
            "mean_symbionts": float(self.soil.symbiont_biomass.mean()),
            "mean_soil_moisture": float(self.soil.moisture.mean()),
            "mean_deep_moisture": float(self.soil.deep_moisture.mean()),
            "mean_nutrients": float(self.soil.shallow_nutrients.mean()),
            "mean_litter": float(self.soil.litter_carbon.mean()),
            "mean_canopy": float(self.canopy_cover.mean()),
            "mean_root_density": float(self.root_density.mean()),
            "total_plant_cells": float(total_cells),
            "mean_cell_vitality": float(mean_cell_vitality),
            "mean_cell_energy": float(mean_cell_energy),
            "mean_division_pressure": float(mean_division),
            "mean_soil_glucose": float(self.soil.chemistry.glucose.mean()),
            "mean_soil_oxygen": float(self.soil.chemistry.oxygen.mean()),
            "mean_soil_ammonium": float(self.soil.chemistry.ammonium.mean()),
            "mean_soil_nitrate": float(self.soil.chemistry.nitrate.mean()),
            "mean_soil_redox": float(self.soil.chemistry.redox_balance.mean()),
            "mean_soil_atp_flux": float(self.soil.chemistry.atp_flux.mean()),
            "broad_backend": self.soil.broad_backend,
            "uptake_backend": self.soil.uptake_backend,
            "substrate_backend": self.soil.substrate_backend,
            "substrate_time_ms": float(self.soil.substrate_time_ms),
            "substrate_steps": float(self.soil.substrate_step_count),
            "cell_metabolism_backend": cell_backend,
            "plant_backend": plant_backend,
            "field_backend": self.field_backend,
            "food_backend": self.food_backend,
            "seed_backend": self.seed_backend,
        }


__all__ = [
    "PlantGenome",
    "SeedPropagule",
    "PlantCellCluster",
    "PlantCellularState",
    "PlantOrganism",
    "SoilChemistryField",
    "SoilBiogeochemistry",
    "RustBatchedAtomTerrarium",
    "RustCellularMetabolism",
    "RustPlantCellularState",
    "RustPlantOrganism",
    "TerrariumEcology",
]
