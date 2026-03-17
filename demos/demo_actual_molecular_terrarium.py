#!/usr/bin/env python3
"""
Actual molecular terrarium demo built on the real oNeura substrate.

This demo intentionally uses the organism and world models that already exist
in the repository instead of hand-authored "emergent" behavior rules:

 - organism: oneuro.organisms.drosophila.Drosophila
 - environment: oneuro.worlds.molecular_world.MolecularWorld

The remaining code in this file is ecological scaffolding and rendering:
plant growth, fruiting, soil moisture turnover, a top-down terrarium view,
and some convenience controls. It does NOT replace the fly's sensory/motor
loop with custom steering logic.

Run:
    python3 demos/demo_actual_molecular_terrarium.py

Headless verification:
    python3 demos/demo_actual_molecular_terrarium.py --headless-frames 120
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oneuro.ecology.terrarium import PlantOrganism, TerrariumEcology
from oneuro.organisms.drosophila import Drosophila
from oneuro.organisms.rust_drosophila import RustTerrariumDrosophila
from oneuro.worlds.molecular_world import MolecularWorld


WORLD_H = 32
WORLD_W = 44
SCALE = 18
PANEL_W = 320
SCREEN_W = WORLD_W * SCALE + PANEL_W
SCREEN_H = WORLD_H * SCALE
FPS = 30

PHYS_DT = 0.05
SUBSTEPS = 2
TIME_WARP = 900.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class TerrariumWorld(MolecularWorld):
    """Compatibility adapter for Drosophila feeding."""

    def deplete_food(self, x: float, y: float) -> None:
        self.deplete_food_near(x, y, eat_radius=2.0, amount=0.03)


class ActualTerrarium:
    def __init__(
        self,
        seed: int = 7,
        *,
        prefer_rust_substrate: bool = True,
        prefer_rust_fly: bool = True,
    ):
        self.seed = seed
        self.prefer_rust_substrate = prefer_rust_substrate
        self.prefer_rust_fly = prefer_rust_fly and RustTerrariumDrosophila is not None
        self.rng = np.random.default_rng(seed)
        self.world = TerrariumWorld(size=(WORLD_H, WORLD_W), seed=seed)
        self.world.wind_vx[:] = 0.0
        self.world.wind_vy[:] = 0.0
        self.world.time = 8 * 3600.0
        self.ecology = TerrariumEcology(self.world, seed=seed, prefer_rust_substrate=prefer_rust_substrate)
        self.flies: list[object] = []
        self.fly_logs: list[dict] = []
        self.fly_backend = "rust" if self.prefer_rust_fly else "python"
        self.paused = False
        self.view_mode = "terrain"
        self._build_world()

    @property
    def plants(self) -> list[PlantOrganism]:
        return self.ecology.plants

    def _build_world(self) -> None:
        self.world.water_sources.clear()
        self.world.plant_sources.clear()
        self.world.fruit_sources.clear()
        self.world.food_patches = []
        self.world.odorant_grids = {
            name: np.zeros((self.world.D, self.world.H, self.world.W), dtype=np.float64)
            for name in self.world.odorant_grids
        }
        self.world.soil.surface_moisture[:, :] = self.rng.uniform(0.18, 0.42, (self.world.H, self.world.W))
        self.world.soil.shallow_nutrients[:, :] = self.rng.uniform(0.02, 0.08, (self.world.H, self.world.W))
        self.world.soil.organic_matter[:, :] = self.rng.uniform(0.002, 0.03, (self.world.H, self.world.W))
        self.ecology = TerrariumEcology(self.world, seed=self.seed, prefer_rust_substrate=self.prefer_rust_substrate)

        for x, y, volume in ((8, 25, 180.0), (21, 14, 110.0), (35, 22, 160.0)):
            self.world.add_water(x, y, volume=volume, evaporation_rate=0.0008)
            self.world.soil.surface_moisture[max(0, y - 2):min(self.world.H, y + 3), max(0, x - 2):min(self.world.W, x + 3)] += 0.18

        for x, y in ((6, 7), (14, 11), (26, 8), (34, 15), (37, 25), (11, 24)):
            self.add_plant(x, y)

        for x, y in ((22, 16), (15, 11), (30, 22)):
            self.world.add_food(x, y, radius=3.0, remaining=1.0)
            fruit = self.world.fruit_sources[-1]
            fruit.odorant_emission_rate = 0.18
            fruit.ripeness = 0.92
            fruit.decay_rate = 0.0003

        self.flies.clear()
        self.fly_logs.clear()
        fly_cls = RustTerrariumDrosophila if self.prefer_rust_fly else Drosophila
        for i, (x, y) in enumerate(((20, 16), (24, 18))):
            fly = fly_cls(world=self.world, scale="tiny", device="cpu", seed=self.seed + i, n_ommatidia=48)
            fly.body.x = float(x)
            fly.body.y = float(y)
            fly.body.z = 1.0
            fly.body.heading = float(self.rng.uniform(0.0, math.tau))
            if hasattr(fly.body, "land"):
                fly.body.land()
            else:
                fly.body.z = 0.0
                fly.body.pitch = 0.0
                fly.body.is_flying = False
            self.flies.append(fly)
            self.fly_logs.append({})

    def reset(self) -> None:
        self.__init__(
            self.seed,
            prefer_rust_substrate=self.prefer_rust_substrate,
            prefer_rust_fly=self.prefer_rust_fly,
        )

    def add_plant(self, x: float, y: float) -> None:
        self.ecology.add_plant(int(x), int(y))

    def spawn_fruit_near(self, x: int, y: int) -> None:
        self.ecology.spawn_fruit(int(x), int(y), size=float(self.rng.uniform(0.6, 1.3)))

    def add_water_patch(self, x: int, y: int) -> None:
        self.world.add_water(int(x), int(y), volume=100.0, evaporation_rate=0.001)
        self.world.soil.surface_moisture[max(0, y - 2):min(self.world.H, y + 3), max(0, x - 2):min(self.world.W, x + 3)] += 0.12

    def _constrain_fly(self, fly: Drosophila) -> None:
        fly.body.x = clamp(fly.body.x, 1.0, self.world.W - 2.0)
        fly.body.y = clamp(fly.body.y, 1.0, self.world.H - 2.0)
        fly.body.z = clamp(fly.body.z, 0.0, 8.0)

    def step(self) -> None:
        for _ in range(SUBSTEPS):
            eco_dt = PHYS_DT * TIME_WARP
            self.world.step(PHYS_DT)
            self.world.time += PHYS_DT * (TIME_WARP - 1.0)
            self.ecology.step(eco_dt)

            shared_fly_field = None
            if self.prefer_rust_fly and self.flies:
                shared_fly_field = RustTerrariumDrosophila.prepare_world_field(self.world)
            for i, fly in enumerate(self.flies):
                if isinstance(fly, RustTerrariumDrosophila) and shared_fly_field is not None:
                    result = fly.step(self.world, sensory_field=shared_fly_field)
                else:
                    result = fly.step(self.world)
                self._constrain_fly(fly)
                self.fly_logs[i] = result
            if shared_fly_field is not None:
                RustTerrariumDrosophila.sync_world_from_field(self.world, shared_fly_field)
                self.ecology.sync_external_food_state()

    def stats(self) -> dict[str, float | str]:
        foods = getattr(self.world, "food_patches", [])
        remaining = sum(p["remaining"] for p in foods)
        avg_energy = sum(fly.body.energy for fly in self.flies) / max(1, len(self.flies))
        avg_z = sum(fly.body.z for fly in self.flies) / max(1, len(self.flies))
        eco = self.ecology.stats()
        fly_env_backend = getattr(self.flies[0], "sensory_backend", "python") if self.flies else "python"
        fly_food_backend = getattr(self.flies[0], "food_mutation_backend", "python") if self.flies else "python"
        return {
            "plants": float(len(self.plants)),
            "fruit_patches": float(len([p for p in foods if p["remaining"] > 0.02])),
            "food_remaining": float(remaining),
            "avg_fly_energy": float(avg_energy),
            "avg_altitude": float(avg_z),
            "light": float(self.world._light_intensity()),
            "temperature": float(self.world.sample_temperature(self.world.W // 2, self.world.H // 2)),
            "humidity": float(self.world.sample_humidity(self.world.W // 2, self.world.H // 2)),
            "soil_microbes": float(eco["mean_microbes"]),
            "soil_symbionts": float(eco["mean_symbionts"]),
            "soil_surface_water": float(eco["mean_soil_moisture"]),
            "soil_deep_water": float(eco["mean_deep_moisture"]),
            "soil_nutrients": float(eco["mean_nutrients"]),
            "soil_litter": float(eco["mean_litter"]),
            "canopy_cover": float(eco["mean_canopy"]),
            "root_density": float(eco["mean_root_density"]),
            "plant_cells": float(eco["total_plant_cells"]),
            "cell_vitality": float(eco["mean_cell_vitality"]),
            "cell_energy": float(eco["mean_cell_energy"]),
            "division_pressure": float(eco["mean_division_pressure"]),
            "seed_bank": float(eco["seed_bank"]),
            "soil_glucose": float(eco["mean_soil_glucose"]),
            "soil_oxygen": float(eco["mean_soil_oxygen"]),
            "soil_ammonium": float(eco["mean_soil_ammonium"]),
            "soil_nitrate": float(eco["mean_soil_nitrate"]),
            "soil_redox": float(eco["mean_soil_redox"]),
            "soil_atp_flux": float(eco["mean_soil_atp_flux"]),
            "world_backend": str(getattr(self.world, "world_backend", getattr(self.world, "atmosphere_backend", "python"))),
            "world_time_ms": float(getattr(self.world, "world_step_time_ms", getattr(self.world, "atmosphere_step_time_ms", 0.0))),
            "atmosphere_backend": str(getattr(self.world, "atmosphere_backend", "python")),
            "atmosphere_time_ms": float(getattr(self.world, "atmosphere_step_time_ms", 0.0)),
            "broad_backend": str(eco["broad_backend"]),
            "uptake_backend": str(eco["uptake_backend"]),
            "substrate_backend": str(eco["substrate_backend"]),
            "substrate_steps": float(eco["substrate_steps"]),
            "substrate_time_ms": float(eco["substrate_time_ms"]),
            "cell_metabolism_backend": str(eco["cell_metabolism_backend"]),
            "plant_backend": str(eco["plant_backend"]),
            "field_backend": str(eco["field_backend"]),
            "food_backend": str(eco["food_backend"]),
            "seed_backend": str(eco["seed_backend"]),
            "fly_backend": self.fly_backend,
            "fly_env_backend": str(fly_env_backend),
            "fly_food_backend": str(fly_food_backend),
        }

    def time_label(self) -> str:
        seconds = self.world.time % 86400.0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours:02d}:{minutes:02d}"


class Renderer:
    def __init__(self) -> None:
        self.world_rect = pygame.Rect(0, 0, WORLD_W * SCALE, WORLD_H * SCALE)
        self.font = pygame.font.SysFont("menlo", 15)
        self.small = pygame.font.SysFont("menlo", 13)
        self.title = pygame.font.SysFont("menlo", 20, bold=True)

    def _normalize(self, field: np.ndarray) -> np.ndarray:
        peak = float(field.max())
        if peak <= 1e-9:
            return np.zeros_like(field, dtype=np.float32)
        return (field / peak).astype(np.float32)

    def _plant_scalar_field(self, terrarium: ActualTerrarium, attr: str, *, radius_scale: float = 1.0) -> np.ndarray:
        field = np.zeros((terrarium.world.H, terrarium.world.W), dtype=np.float32)
        for plant in terrarium.plants:
            x = int(np.clip(round(plant.source.x), 0, terrarium.world.W - 1))
            y = int(np.clip(round(plant.source.y), 0, terrarium.world.H - 1))
            radius = max(1, int(round(plant.canopy_radius_cells * radius_scale)))
            y0 = max(0, y - radius)
            y1 = min(terrarium.world.H, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(terrarium.world.W, x + radius + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            sigma = max(radius * 0.75, 1.0)
            kernel = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
            field[y0:y1, x0:x1] += kernel * float(getattr(plant, attr))
        return self._normalize(field)

    def _base_world_surface(self, terrarium: ActualTerrarium) -> pygame.Surface:
        world = terrarium.world
        soil = terrarium.ecology.soil
        light = world._light_intensity()
        dim = 0.30 + light * 0.70
        sky = np.array(
            [
                int(12 + light * 70),
                int(18 + light * 102),
                int(28 + light * 118),
            ],
            dtype=np.uint8,
        )

        if terrarium.view_mode == "soil":
            moist = world.soil.surface_moisture
            deep = self._normalize(soil.deep_moisture)
            nutrients = self._normalize(soil.shallow_nutrients + soil.dissolved_nutrients)
            litter = self._normalize(soil.litter_carbon)
            organic = self._normalize(soil.organic_matter)
            ground = np.stack(
                [
                    24 + litter * 145 + nutrients * 42,
                    20 + moist * 95 + deep * 58,
                    18 + deep * 112 + organic * 52,
                ],
                axis=-1,
            )
        elif terrarium.view_mode == "roots":
            roots = self._normalize(terrarium.ecology.root_density)
            canopy = self._normalize(terrarium.ecology.canopy_cover)
            fungi = self._normalize(soil.symbiont_biomass)
            ground = np.stack(
                [
                    20 + roots * 155 + canopy * 36,
                    18 + fungi * 150 + canopy * 84,
                    18 + canopy * 122 + roots * 28,
                ],
                axis=-1,
            )
        elif terrarium.view_mode == "odor":
            ethanol = self._normalize(np.log1p(world.odorant_grids["ethanol"][0] * 10.0))
            acetate = self._normalize(np.log1p(world.odorant_grids["ethyl_acetate"][0] * 10.0))
            geraniol = self._normalize(np.log1p(world.odorant_grids["geraniol"][0] * 10.0))
            ammonia = self._normalize(np.log1p(world.odorant_grids["ammonia"][0] * 10.0))
            ground = np.stack(
                [
                    18 + ethanol * 160 + ammonia * 70,
                    18 + geraniol * 165,
                    20 + acetate * 150 + ammonia * 38,
                ],
                axis=-1,
            )
        elif terrarium.view_mode == "cells":
            vitality = self._plant_scalar_field(terrarium, "cell_vitality")
            energy = self._plant_scalar_field(terrarium, "cellular_energy_charge")
            division = self._plant_scalar_field(terrarium, "cellular_division_pressure", radius_scale=0.65)
            ground = np.stack(
                [
                    18 + division * 165 + energy * 42,
                    18 + vitality * 170 + division * 48,
                    20 + energy * 160,
                ],
                axis=-1,
            )
        elif terrarium.view_mode == "chemistry":
            chem = terrarium.ecology.soil.chemistry
            glucose = self._normalize(np.log1p(chem.glucose * 60.0))
            oxygen = self._normalize(chem.oxygen)
            nitrate = self._normalize(chem.nitrate + chem.ammonium * 0.45)
            acidity = self._normalize(chem.acidity)
            redox = self._normalize(chem.redox_balance)
            ground = np.stack(
                [
                    18 + glucose * 150 + acidity * 80,
                    18 + oxygen * 150 + redox * 55,
                    18 + nitrate * 160 + glucose * 18,
                ],
                axis=-1,
            )
        else:
            odor = (
                world.odorant_grids["ethanol"][0]
                + world.odorant_grids["ethyl_acetate"][0] * 1.2
                + world.odorant_grids["geraniol"][0] * 0.8
                + world.odorant_grids["acetic_acid"][0] * 0.5
            )
            odor = self._normalize(np.log1p(odor * 10.0))
            moist = world.soil.surface_moisture
            nutrients = self._normalize(soil.shallow_nutrients)
            microbes = self._normalize(soil.microbial_biomass)
            fungi = self._normalize(soil.symbiont_biomass)
            ground = np.stack(
                [
                    26 + moist * 20 + odor * 110 + nutrients * 22,
                    20 + moist * 42 + odor * 42 + microbes * 34 + fungi * 18,
                    16 + moist * 18 + microbes * 12 + fungi * 26,
                ],
                axis=-1,
            )
        ground *= dim
        arr = np.clip(ground, 0, 255).astype(np.uint8)
        arr[:2, :, :] = sky
        return pygame.surfarray.make_surface(arr.swapaxes(0, 1))

    def _xy(self, x: float, y: float) -> tuple[int, int]:
        return int(x * SCALE), int(y * SCALE)

    def _draw_water(self, screen: pygame.Surface, terrarium: ActualTerrarium) -> None:
        for src in terrarium.world.water_sources:
            x, y = self._xy(src.x, src.y)
            r = 18 if src.volume > 140 else 12
            pygame.draw.circle(screen, (30, 92, 170), (x, y), r)
            pygame.draw.circle(screen, (90, 170, 230), (x - 5, y - 4), max(4, r // 3))

    def _draw_plants(self, screen: pygame.Surface, terrarium: ActualTerrarium) -> None:
        for plant in terrarium.plants:
            x, y = self._xy(plant.source.x, plant.source.y)
            h = int(plant.source.height * 2.2)
            canopy = int(8 + plant.leaf_biomass * 34)
            stem_top = (x, max(4, y - h))
            if terrarium.view_mode == "cells":
                energy = int(clamp(70 + plant.cellular_energy_charge * 180.0, 70, 220))
                vitality = int(clamp(70 + plant.cell_vitality * 120.0, 70, 220))
                division = int(clamp(40 + plant.cellular_division_pressure * 220.0, 40, 240))
                stem_color = (division, 90, energy)
                canopy_color = (division, vitality, energy)
                shade_color = (max(20, division - 34), max(20, vitality - 46), energy)
            else:
                stem_color = (38, 130, 62)
                canopy_color = (62, 176, 92)
                shade_color = (34, 118, 64)
            pygame.draw.line(screen, stem_color, (x, y), stem_top, 3)
            pygame.draw.circle(screen, canopy_color, stem_top, canopy)
            pygame.draw.circle(screen, shade_color, (stem_top[0] - canopy // 2, stem_top[1] + 2), max(4, canopy // 2))
            root_w = max(4, int(plant.genome.root_radius_mm * 3))
            pygame.draw.line(screen, (88, 64, 42), (x, y), (x - root_w // 2, y + root_w // 2), 2)
            pygame.draw.line(screen, (88, 64, 42), (x, y), (x + root_w // 2, y + root_w // 2), 2)

    def _draw_food(self, screen: pygame.Surface, terrarium: ActualTerrarium) -> None:
        for patch in getattr(terrarium.world, "food_patches", []):
            if patch["remaining"] <= 0.02:
                continue
            x, y = self._xy(patch["x"], patch["y"])
            soil = terrarium.ecology.soil
            xi = int(np.clip(round(patch["x"]), 0, terrarium.world.W - 1))
            yi = int(np.clip(round(patch["y"]), 0, terrarium.world.H - 1))
            detritus = float(soil.litter_carbon[yi, xi])
            r = int(max(6, patch["radius"] * SCALE * 0.45 * (0.45 + patch["remaining"] * 0.55)))
            ring = int(clamp(18 + detritus * 220.0, 18, 118))
            pygame.draw.circle(screen, (ring, 44, 18), (x, y), r + 3, 1)
            pygame.draw.circle(screen, (255, 162, 58), (x, y), r)
            pygame.draw.circle(screen, (255, 214, 112), (x - r // 4, y - r // 4), max(3, r // 3))

    def _draw_seeds(self, screen: pygame.Surface, terrarium: ActualTerrarium) -> None:
        for seed in terrarium.ecology.seed_bank[:80]:
            x, y = self._xy(seed.x, seed.y)
            intensity = int(clamp(150 + seed.reserve_carbon * 380.0, 150, 235))
            pygame.draw.circle(screen, (intensity, intensity - 18, 102), (x, y), 2)

    def _draw_flies(self, screen: pygame.Surface, terrarium: ActualTerrarium) -> None:
        for i, fly in enumerate(terrarium.flies):
            x, y = self._xy(fly.body.x, fly.body.y)
            shadow_y = int(y + 4)
            pygame.draw.ellipse(screen, (10, 10, 10), pygame.Rect(x - 9, shadow_y - 4, 18, 8))

            altitude_scale = 1.0 + fly.body.z * 0.08
            body_w = int(14 * altitude_scale)
            body_h = int(8 * altitude_scale)
            body_rect = pygame.Rect(x - body_w // 2, y - body_h // 2, body_w, body_h)
            pygame.draw.ellipse(screen, (42, 42, 48), body_rect)

            wing_alpha = 110 if fly.body.is_flying else 55
            wing_surface = pygame.Surface((34, 20), pygame.SRCALPHA)
            pygame.draw.ellipse(wing_surface, (210, 220, 255, wing_alpha), pygame.Rect(2, 3, 14, 8))
            pygame.draw.ellipse(wing_surface, (210, 220, 255, wing_alpha), pygame.Rect(18, 3, 14, 8))
            angle = -math.degrees(fly.body.heading)
            rotated = pygame.transform.rotate(wing_surface, angle)
            screen.blit(rotated, rotated.get_rect(center=(x, y)))

            hx = x + int(math.cos(fly.body.heading) * 10)
            hy = y + int(math.sin(fly.body.heading) * 10)
            pygame.draw.line(screen, (255, 120, 72), (x, y), (hx, hy), 2)

            tag = self.small.render(str(i + 1), True, (220, 230, 240))
            screen.blit(tag, (x + 8, y - 14))

    def _draw_panel(self, screen: pygame.Surface, terrarium: ActualTerrarium) -> None:
        panel_x = self.world_rect.width
        pygame.draw.rect(screen, (10, 12, 16), (panel_x, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(screen, (44, 50, 58), (panel_x, 0), (panel_x, SCREEN_H), 1)

        screen.blit(self.title.render("Actual Terrarium", True, (226, 235, 220)), (panel_x + 18, 18))
        screen.blit(self.small.render("MolecularWorld + Drosophila + Ecology", True, (114, 152, 122)), (panel_x + 18, 46))

        stats = terrarium.stats()
        stat_lines = [
            ("Clock", terrarium.time_label()),
            ("View", terrarium.view_mode),
            ("World Phys", f"{stats['world_backend']} / {stats['world_time_ms']:.2f}"),
            ("Substrate", f"{stats['substrate_backend']} / {int(stats['substrate_steps'])}"),
            ("Soil Broad", str(stats["broad_backend"])),
            ("Root Uptake", str(stats["uptake_backend"])),
            ("Cell Chem", str(stats["cell_metabolism_backend"])),
            ("Plant Phys", str(stats["plant_backend"])),
            ("Eco Fields", str(stats["field_backend"])),
            ("Eco Events", f"{stats['food_backend']}/{stats['seed_backend']}"),
            ("Fly Loop", str(stats["fly_backend"])),
            ("Fly Env", str(stats["fly_env_backend"])),
            ("Fly Food", str(stats["fly_food_backend"])),
            ("Light", f"{stats['light'] * 100:.0f}%"),
            ("Temp", f"{stats['temperature']:.1f} C"),
            ("Plants / Seeds", f"{int(stats['plants'])} / {int(stats['seed_bank'])}"),
            ("Fruit / Food", f"{int(stats['fruit_patches'])} / {stats['food_remaining']:.2f}"),
            ("Water S / D", f"{stats['soil_surface_water']:.3f} / {stats['soil_deep_water']:.3f}"),
            ("Nutr / Litter", f"{stats['soil_nutrients']:.3f} / {stats['soil_litter']:.3f}"),
            ("Micro / Sym", f"{stats['soil_microbes']:.3f} / {stats['soil_symbionts']:.3f}"),
            ("Cells / Vital", f"{stats['plant_cells']:.0f} / {stats['cell_vitality']:.2f}"),
            ("Eng / Divide", f"{stats['cell_energy']:.2f} / {stats['division_pressure']:.2f}"),
            ("Chem G / O2", f"{stats['soil_glucose']:.3f} / {stats['soil_oxygen']:.3f}"),
            ("NH4 / NO3", f"{stats['soil_ammonium']:.3f} / {stats['soil_nitrate']:.3f}"),
            ("Redox / ATP", f"{stats['soil_redox']:.2f} / {stats['soil_atp_flux']:.3f}"),
            ("Mean Fly Energy", f"{stats['avg_fly_energy']:.2f}"),
        ]

        y = 92
        for key, value in stat_lines:
            screen.blit(self.small.render(key, True, (138, 146, 156)), (panel_x + 18, y))
            screen.blit(self.small.render(value, True, (214, 224, 236)), (panel_x + 170, y))
            y += 16

        y += 8
        screen.blit(self.font.render("Fly State", True, (186, 196, 210)), (panel_x + 18, y))
        y += 20
        for idx, (fly, log) in enumerate(zip(terrarium.flies, terrarium.fly_logs)):
            motor = log.get("motor", {}) if log else {}
            line = (
                f"Fly {idx + 1}: E={fly.body.energy:.1f} "
                f"v={motor.get('speed', 0.0):.2f} "
                f"t={motor.get('turn', 0.0):.2f} "
                f"{'fly' if fly.body.is_flying else 'walk'}"
            )
            screen.blit(self.small.render(line, True, (148, 164, 176)), (panel_x + 18, y))
            y += 16

        y += 10
        controls = [
            "Space pause  R reset",
            "F fruit  P plant  W water",
            "1 terrain  2 soil  3 roots",
            "4 odor  5 cells  6 chemistry",
            "Esc quit",
        ]
        screen.blit(self.font.render("Controls", True, (186, 196, 210)), (panel_x + 18, y))
        y += 20
        for text in controls:
            screen.blit(self.small.render(text, True, (144, 160, 176)), (panel_x + 18, y))
            y += 16

        y += 8
        notes = [
            "Fly loop can run on Python or the Rust native adapter.",
            "Plants inherit genomes plus coarse cell clusters.",
            "Soil chemistry can run on the Rust batched substrate.",
            "Metal is used when available; CPU fallback stays honest.",
            "This is coarse-grained chemistry, not atomistic MD.",
        ]
        for text in notes:
            screen.blit(self.small.render(text, True, (110, 126, 118)), (panel_x + 18, y))
            y += 16

    def draw(self, screen: pygame.Surface, terrarium: ActualTerrarium) -> None:
        low = self._base_world_surface(terrarium)
        scaled = pygame.transform.scale(low, self.world_rect.size)
        screen.blit(scaled, self.world_rect.topleft)

        self._draw_water(screen, terrarium)
        self._draw_plants(screen, terrarium)
        self._draw_food(screen, terrarium)
        self._draw_seeds(screen, terrarium)
        self._draw_flies(screen, terrarium)
        pygame.draw.rect(screen, (66, 78, 92), self.world_rect, 2)
        self._draw_panel(screen, terrarium)


def run_demo(
    seed: int,
    headless_frames: int | None = None,
    *,
    prefer_rust_substrate: bool = True,
    prefer_rust_fly: bool = True,
) -> None:
    if headless_frames is not None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    pygame.init()
    pygame.display.set_caption("oNeura Actual Molecular Terrarium")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()

    terrarium = ActualTerrarium(
        seed=seed,
        prefer_rust_substrate=prefer_rust_substrate,
        prefer_rust_fly=prefer_rust_fly,
    )
    renderer = Renderer()
    running = True
    frames = 0

    while running:
        if headless_frames is None:
            dt_render = min(clock.tick(FPS) / 1000.0, 0.05)
        else:
            dt_render = 1.0 / FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    terrarium.paused = not terrarium.paused
                elif event.key == pygame.K_f:
                    terrarium.spawn_fruit_near(
                        int(terrarium.rng.integers(4, terrarium.world.W - 4)),
                        int(terrarium.rng.integers(4, terrarium.world.H - 4)),
                    )
                elif event.key == pygame.K_p:
                    terrarium.add_plant(
                        terrarium.rng.integers(4, terrarium.world.W - 4),
                        terrarium.rng.integers(4, terrarium.world.H - 4),
                    )
                elif event.key == pygame.K_w:
                    terrarium.add_water_patch(
                        int(terrarium.rng.integers(4, terrarium.world.W - 4)),
                        int(terrarium.rng.integers(4, terrarium.world.H - 4)),
                    )
                elif event.key == pygame.K_1:
                    terrarium.view_mode = "terrain"
                elif event.key == pygame.K_2:
                    terrarium.view_mode = "soil"
                elif event.key == pygame.K_3:
                    terrarium.view_mode = "roots"
                elif event.key == pygame.K_4:
                    terrarium.view_mode = "odor"
                elif event.key == pygame.K_5:
                    terrarium.view_mode = "cells"
                elif event.key == pygame.K_6:
                    terrarium.view_mode = "chemistry"
                elif event.key == pygame.K_r:
                    terrarium.reset()

        if not terrarium.paused:
            terrarium.step()
            frames += 1

        if headless_frames is None:
            renderer.draw(screen, terrarium)
            pygame.display.flip()

        if headless_frames is not None and frames >= headless_frames:
            stats = terrarium.stats()
            print("actual_molecular_terrarium_summary")
            for key, value in stats.items():
                print(f"{key}={value}")
            print(f"time={terrarium.time_label()}")
            break

        if headless_frames is None:
            _ = dt_render

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Actual molecular terrarium demo")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--headless-frames", type=int, default=None, help="run without a visible window")
    parser.add_argument("--python-substrate", action="store_true", help="disable the Rust/Metal soil substrate")
    parser.add_argument("--python-fly", action="store_true", help="disable the Rust terrarium fly adapter")
    args = parser.parse_args()
    run_demo(
        seed=args.seed,
        headless_frames=args.headless_frames,
        prefer_rust_substrate=not args.python_substrate,
        prefer_rust_fly=not args.python_fly,
    )


if __name__ == "__main__":
    main()
