"""Native terrarium adapter for the Rust Drosophila simulator.

This adapter bridges the Python `MolecularWorld` into the native
`oneuro-metal` Drosophila loop. The Rust side owns the neural stepping,
motor decode, body-state update, shared sensory-field snapshot, and
in-field food consumption; Python only owns the authoritative world model
and the final sync of shared state back into that model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    from oneuro_metal import DrosophilaSim as RustDrosophilaSim
    from oneuro_metal import TerrariumSensoryField as RustTerrariumSensoryField
except ImportError:  # pragma: no cover - optional native extension
    RustDrosophilaSim = None
    RustTerrariumSensoryField = None


_SCALE_NEURONS = {
    "tiny": 1_000,
    "small": 5_000,
    "medium": 25_000,
    "large": 139_000,
}


@dataclass
class RustDrosophilaBody:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0
    pitch: float = 0.0
    speed: float = 0.0
    energy: float = 100.0
    temperature: float = 22.0
    is_flying: bool = False


class RustTerrariumDrosophila:
    """Terrarium-facing wrapper around the Rust `DrosophilaSim`.

    The native fly owns the sense -> brain -> motor -> body loop. This
    wrapper only compresses the terrarium state into the current native
    sensory surface and mirrors the resulting body state.
    """

    backend = "rust"
    _shared_fields: dict[int, dict[str, Any]] = {}

    def __init__(
        self,
        world=None,
        scale: str = "tiny",
        device: str = "auto",
        seed: int = 42,
        n_ommatidia: int = 48,
    ):
        del device, n_ommatidia
        if RustDrosophilaSim is None:
            raise ImportError("oneuro_metal.DrosophilaSim is not available")
        if scale not in _SCALE_NEURONS:
            raise ValueError(f"Unknown Drosophila scale '{scale}'")

        self.world = world
        self.scale = scale
        self.seed = seed
        self.native = RustDrosophilaSim(_SCALE_NEURONS[scale], seed)
        self.body = RustDrosophilaBody()
        self.step_count = 0
        self.sensory_backend = "rust" if RustTerrariumSensoryField is not None else "python"
        self.food_mutation_backend = "rust" if RustTerrariumSensoryField is not None else "python"
        self._sync_from_native()

    @staticmethod
    def _world_bounds(world) -> tuple[float, float]:
        width = float(getattr(world, "W", getattr(world, "width", 64.0)))
        height = float(getattr(world, "H", getattr(world, "height", 64.0)))
        return width, height

    @staticmethod
    def _world_depth(world) -> int:
        return int(getattr(world, "D", 1))

    @staticmethod
    def _ensure_world_revision(world) -> int:
        if world is None:
            return 0
        if not hasattr(world, "_rust_fly_field_revision"):
            world._rust_fly_field_revision = 0
        return int(world._rust_fly_field_revision)

    @staticmethod
    def _bump_world_revision(world) -> None:
        if world is None:
            return
        world._rust_fly_field_revision = RustTerrariumDrosophila._ensure_world_revision(world) + 1

    @classmethod
    def _field_entry(cls, world) -> Optional[dict[str, Any]]:
        if world is None or RustTerrariumSensoryField is None:
            return None
        key = id(world)
        entry = cls._shared_fields.get(key)
        shape = (int(world.W), int(world.H), cls._world_depth(world))
        if entry is None or entry.get("shape") != shape:
            entry = {
                "field": RustTerrariumSensoryField(shape[0], shape[1], shape[2]),
                "shape": shape,
                "stamp": None,
            }
            cls._shared_fields[key] = entry
        return entry

    @classmethod
    def _refresh_shared_field(cls, world) -> Optional[Any]:
        entry = cls._field_entry(world)
        if entry is None:
            return None
        stamp = (float(getattr(world, "time", 0.0)), cls._ensure_world_revision(world))
        if entry["stamp"] == stamp:
            return entry["field"]

        entry["field"].load_world_state(world)
        entry["stamp"] = stamp
        return entry["field"]

    @classmethod
    def prepare_world_field(cls, world) -> Optional[Any]:
        """Refresh or reuse the shared native sensory field for a world."""
        return cls._refresh_shared_field(world)

    @classmethod
    def sync_world_from_field(cls, world, field: Optional[Any] = None) -> bool:
        """Mirror shared field food patches back into the Python world.

        During a fly substep, the native sensory field owns food depletion so
        multiple flies can consume against one shared mutable substrate. This
        method writes the remaining patch mass back once after that loop.
        """
        entry = cls._field_entry(world)
        if entry is None:
            return False
        active_field = field if field is not None else entry["field"]
        if hasattr(active_field, "sync_food_to_world"):
            return bool(active_field.sync_food_to_world(world))
        return False

    def _sample_odorant(self, world) -> float:
        if world is None:
            return 0.0
        ax = self.body.x + math.cos(self.body.heading) * 0.5
        ay = self.body.y + math.sin(self.body.heading) * 0.5
        az = self.body.z + 0.3
        if hasattr(world, "sample_odorants"):
            odorants = world.sample_odorants(int(round(ax)), int(round(ay)), int(round(az)))
            return float(np.clip(sum(float(v) for v in odorants.values()), 0.0, 1.0))
        if hasattr(world, "odorants_at"):
            odorants = world.odorants_at(ax, ay)
            return float(np.clip(sum(float(v) for v in odorants.values()), 0.0, 1.0))
        return 0.0

    def _sample_light_pair(self, world) -> tuple[float, float]:
        if world is None:
            return 0.5, 0.5
        ambient = float(world._light_intensity()) if hasattr(world, "_light_intensity") else 0.5
        return ambient, ambient

    def _sample_temperature(self, world) -> float:
        if world is None:
            return 22.0
        if hasattr(world, "sample_temperature"):
            return float(world.sample_temperature(int(round(self.body.x)), int(round(self.body.y)), int(round(self.body.z))))
        if hasattr(world, "temperature_at"):
            return float(world.temperature_at(self.body.x, self.body.y))
        return 22.0

    def _sample_wind(self, world) -> tuple[float, float, float]:
        if world is None:
            return 0.0, 0.0, 0.0
        if hasattr(world, "sample_wind"):
            wind = world.sample_wind(int(round(self.body.x)), int(round(self.body.y)), int(round(self.body.z)))
            if len(wind) == 3:
                return float(wind[0]), float(wind[1]), float(wind[2])
            return float(wind[0]), float(wind[1]), 0.0
        if hasattr(world, "wind_at"):
            wind = world.wind_at(self.body.x, self.body.y)
            return float(wind[0]), float(wind[1]), 0.0
        return 0.0, 0.0, 0.0

    def _sample_taste(self, world) -> tuple[float, float, float, float]:
        if world is None or self.body.is_flying:
            return 0.0, 0.0, 0.0, 0.0
        sugar = 0.0
        bitter = 0.0
        amino = 0.0
        food_available = 0.0
        if hasattr(world, "taste_at"):
            taste = world.taste_at(self.body.x, self.body.y)
            sugar = float(taste.get("sugar", 0.0))
            bitter = float(taste.get("bitter", 0.0))
            amino = float(taste.get("amino_acid", taste.get("amino", 0.0)))
        if hasattr(world, "food_at"):
            food_available = float(np.clip(world.food_at(self.body.x, self.body.y), 0.0, 1.0))
            sugar = max(sugar, food_available)
        return sugar, bitter, amino, food_available

    def _sync_to_native(self, world) -> None:
        width, height = self._world_bounds(world)
        time_of_day = None
        if world is not None and hasattr(world, "time"):
            time_of_day = float(world.time % 86400.0) / 3600.0
        self.native.set_body_state(
            self.body.x,
            self.body.y,
            self.body.heading,
            z=float(self.body.z),
            pitch=float(self.body.pitch),
            is_flying=bool(self.body.is_flying),
            speed=float(self.body.speed),
            energy=float(self.body.energy),
            temperature=float(self.body.temperature),
            time_of_day=time_of_day,
            world_width=width,
            world_height=height,
        )

    def _sync_from_native(self) -> None:
        self.body.x = float(self.native.body_x)
        self.body.y = float(self.native.body_y)
        self.body.z = float(self.native.body_z)
        self.body.heading = float(self.native.body_heading)
        self.body.pitch = float(self.native.body_pitch)
        self.body.speed = float(self.native.body_speed)
        self.body.energy = float(self.native.body_energy)
        self.body.is_flying = bool(self.native.body_is_flying)

    def step(self, world=None, *, sensory_field=None) -> Dict[str, Any]:
        world = world if world is not None else self.world
        self.world = world
        self.step_count += 1
        self._sync_to_native(world)
        field = sensory_field if sensory_field is not None else type(self)._refresh_shared_field(world)
        if field is not None:
            sensory = field.sample_fly(
                self.body.x,
                self.body.y,
                z=self.body.z,
                heading=self.body.heading,
                is_flying=self.body.is_flying,
            )
            self.sensory_backend = "rust"
        else:
            sensory = {
                "odorant": self._sample_odorant(world),
                "left_light": self._sample_light_pair(world)[0],
                "right_light": self._sample_light_pair(world)[1],
                "temperature": self._sample_temperature(world),
                "wind_x": self._sample_wind(world)[0],
                "wind_y": self._sample_wind(world)[1],
                "wind_z": self._sample_wind(world)[2],
                "sugar_taste": self._sample_taste(world)[0],
                "bitter_taste": self._sample_taste(world)[1],
                "amino_taste": self._sample_taste(world)[2],
                "food_available": self._sample_taste(world)[3],
            }
            self.sensory_backend = "python"
        report = self.native.step_terrarium(
            float(sensory["odorant"]),
            float(sensory["left_light"]),
            float(sensory["right_light"]),
            float(sensory["temperature"]),
            sugar_taste=float(sensory["sugar_taste"]),
            bitter_taste=float(sensory["bitter_taste"]),
            amino_taste=float(sensory["amino_taste"]),
            wind_x=float(sensory["wind_x"]),
            wind_y=float(sensory["wind_y"]),
            wind_z=float(sensory["wind_z"]),
            food_available=float(sensory["food_available"]),
            reward_valence=0.0,
        )
        if float(report.get("consumed_food", 0.0)) > 0.0:
            if field is not None and hasattr(field, "consume_food_near"):
                field.consume_food_near(self.body.x, self.body.y, eat_radius=2.0, amount=0.03)
                self.food_mutation_backend = "rust"
            elif world is not None and hasattr(world, "deplete_food"):
                world.deplete_food(self.body.x, self.body.y)
                type(self)._bump_world_revision(world)
                self.food_mutation_backend = "python"
        self.body.x = float(report["x"])
        self.body.y = float(report["y"])
        self.body.z = float(report["z"])
        self.body.heading = float(report["heading"])
        self.body.pitch = float(report["pitch"])
        self.body.speed = float(report["speed"])
        self.body.energy = float(report["energy"])
        self.body.is_flying = bool(report["is_flying"])
        self.body.temperature = float(sensory["temperature"])
        return {
            "step": self.step_count,
            "x": self.body.x,
            "y": self.body.y,
            "z": self.body.z,
            "heading": self.body.heading,
            "pitch": self.body.pitch,
            "is_flying": self.body.is_flying,
            "energy": self.body.energy,
            "backend": self.backend,
            "sensory_backend": self.sensory_backend,
            "food_mutation_backend": self.food_mutation_backend,
            "motor": {
                "speed": float(report["speed"]),
                "turn": float(report["turn"]),
                "fly": float(report["fly"]),
                "feed": float(report["feed"]),
                "climb": float(report["climb"]),
            },
        }


__all__ = ["RustTerrariumDrosophila", "RustDrosophilaBody"]
