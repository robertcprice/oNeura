"""
Physics Environment

3D environment for Drosophila MuJoCo simulation with fruits, plants,
obstacles, wind, and light. Generates MJCF geometry for environmental
objects and applies external forces during simulation.

The environment operates at the same 1000x scale as the fly model.
"""

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Fruit:
    """A fruit object providing odorant stimulus.

    Attributes:
        position: [x, y, z] position in world coordinates
        radius: Fruit radius (scaled)
        sugar_concentration: Odorant intensity [0, 1]
        color: RGBA color for rendering
    """
    position: np.ndarray
    radius: float = 0.1
    sugar_concentration: float = 0.5
    color: tuple = (1.0, 0.3, 0.0, 1.0)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)


@dataclass
class Plant:
    """A plant object with stem and flower.

    Attributes:
        position: [x, y] base position (z is ground level)
        height: Plant height (scaled)
        stem_radius: Stem radius (scaled)
        color: RGBA color for stem
    """
    position: np.ndarray
    height: float = 2.0
    stem_radius: float = 0.05
    color: tuple = (0.2, 0.6, 0.1, 1.0)

    def __post_init__(self):
        self.position = np.asarray(self.position[:2], dtype=float)


@dataclass
class Obstacle:
    """A static obstacle for collision.

    Attributes:
        position: [x, y, z] center position
        size: [sx, sy, sz] half-extents for box, or [r, h] for cylinder
        shape: "box" or "cylinder"
        color: RGBA color
    """
    position: np.ndarray
    size: np.ndarray
    shape: str = "box"
    color: tuple = (0.5, 0.4, 0.3, 1.0)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.size = np.asarray(self.size, dtype=float)


class PhysicsEnvironment:
    """3D environment for Drosophila MuJoCo simulation.

    Generates MJCF geometry for environmental objects and applies
    external forces (wind, gravity perturbations) during simulation.

    Example:
        >>> env = PhysicsEnvironment(arena_size=(8.0, 8.0))
        >>> env.add_fruit(pos=[2, 3, 0.1], sugar=0.9)
        >>> env.add_plant(pos=[0, -2], height=3.0)
        >>> env.set_wind(direction=[1, 0, 0], speed=0.5)
        >>> mjcf_xml = env.generate_mjcf()
    """

    def __init__(self, arena_size: tuple = (10.0, 10.0)):
        """Initialize physics environment.

        Args:
            arena_size: (width, depth) of the arena in scaled units
        """
        self.arena_size = arena_size
        self.fruits: list[Fruit] = []
        self.plants: list[Plant] = []
        self.obstacles: list[Obstacle] = []
        self.wind: np.ndarray = np.zeros(3)
        self.wind_speed: float = 0.0
        self.light_direction: np.ndarray = np.array([0.0, 0.0, -1.0])
        self.temperature_base: float = 25.0  # Celsius
        self.temperature_gradient: float = -2.0  # degrees per unit height

    def add_fruit(
        self,
        pos: list,
        radius: float = 0.1,
        sugar: float = 0.5,
        color: tuple = (1.0, 0.3, 0.0, 1.0),
    ):
        """Add a fruit to the environment.

        Args:
            pos: [x, y, z] position
            radius: Fruit radius
            sugar: Sugar/odorant concentration [0, 1]
            color: RGBA color
        """
        self.fruits.append(Fruit(
            position=pos,
            radius=radius,
            sugar_concentration=sugar,
            color=color,
        ))

    def add_plant(
        self,
        pos: list,
        height: float = 2.0,
        stem_radius: float = 0.05,
        color: tuple = (0.2, 0.6, 0.1, 1.0),
    ):
        """Add a plant to the environment.

        Args:
            pos: [x, y] base position
            height: Plant height
            stem_radius: Stem radius
            color: RGBA stem color
        """
        self.plants.append(Plant(
            position=pos,
            height=height,
            stem_radius=stem_radius,
            color=color,
        ))

    def add_obstacle(
        self,
        pos: list,
        size: list,
        shape: str = "box",
        color: tuple = (0.5, 0.4, 0.3, 1.0),
    ):
        """Add a static obstacle.

        Args:
            pos: [x, y, z] center position
            size: Half-extents [sx, sy, sz] for box, or [radius, half_height] for cylinder
            shape: "box" or "cylinder"
            color: RGBA color
        """
        self.obstacles.append(Obstacle(
            position=pos,
            size=size,
            shape=shape,
            color=color,
        ))

    def set_wind(self, direction: list, speed: float):
        """Set wind direction and speed.

        Args:
            direction: [dx, dy, dz] wind direction (will be normalized)
            speed: Wind speed in scaled force units
        """
        d = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(d)
        if norm > 0:
            d = d / norm
        self.wind = d
        self.wind_speed = speed

    def set_light(self, direction: list):
        """Set light direction.

        Args:
            direction: [dx, dy, dz] direction light comes FROM
        """
        d = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(d)
        if norm > 0:
            d = d / norm
        self.light_direction = d

    def generate_mjcf(
        self,
        base_mjcf_path: Optional[str] = None,
    ) -> str:
        """Generate complete MJCF XML with environment objects injected.

        Parses the base Drosophila MJCF model and adds environment
        geometry (fruits, plants, obstacles, arena walls) as static bodies.

        Args:
            base_mjcf_path: Path to base MJCF XML. Defaults to included model.

        Returns:
            Complete MJCF XML string with environment
        """
        if base_mjcf_path is None:
            base_mjcf_path = str(Path(__file__).parent / "drosophila_mjcf.xml")

        tree = ET.parse(base_mjcf_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")

        # Add environment materials to asset
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")

        ET.SubElement(asset, "material", {
            "name": "fruit_mat", "rgba": "1 0.3 0 1",
        })
        ET.SubElement(asset, "material", {
            "name": "plant_mat", "rgba": "0.2 0.6 0.1 1",
        })
        ET.SubElement(asset, "material", {
            "name": "flower_mat", "rgba": "0.9 0.2 0.6 1",
        })
        ET.SubElement(asset, "material", {
            "name": "obstacle_mat", "rgba": "0.5 0.4 0.3 1",
        })
        ET.SubElement(asset, "material", {
            "name": "wall_mat", "rgba": "0.3 0.3 0.3 0.5",
        })

        # Add arena walls
        hw, hd = self.arena_size[0] / 2, self.arena_size[1] / 2
        wall_height = 3.0
        wall_thickness = 0.1

        walls = [
            {"name": "wall_north", "pos": f"0 {hd} {wall_height/2}",
             "size": f"{hw} {wall_thickness} {wall_height/2}"},
            {"name": "wall_south", "pos": f"0 {-hd} {wall_height/2}",
             "size": f"{hw} {wall_thickness} {wall_height/2}"},
            {"name": "wall_east", "pos": f"{hw} 0 {wall_height/2}",
             "size": f"{wall_thickness} {hd} {wall_height/2}"},
            {"name": "wall_west", "pos": f"{-hw} 0 {wall_height/2}",
             "size": f"{wall_thickness} {hd} {wall_height/2}"},
        ]
        for wall in walls:
            ET.SubElement(worldbody, "geom", {
                "name": wall["name"],
                "type": "box",
                "pos": wall["pos"],
                "size": wall["size"],
                "material": "wall_mat",
                "contype": "1",
                "conaffinity": "1",
            })

        # Add fruits
        for i, fruit in enumerate(self.fruits):
            body = ET.SubElement(worldbody, "body", {
                "name": f"fruit_{i}",
                "pos": f"{fruit.position[0]} {fruit.position[1]} {fruit.position[2]}",
            })
            ET.SubElement(body, "geom", {
                "name": f"fruit_{i}_geom",
                "type": "sphere",
                "size": str(fruit.radius),
                "rgba": " ".join(str(c) for c in fruit.color),
                "contype": "1",
                "conaffinity": "1",
                "mass": "0.01",
            })

        # Add plants (stem cylinder + flower sphere)
        for i, plant in enumerate(self.plants):
            body = ET.SubElement(worldbody, "body", {
                "name": f"plant_{i}",
                "pos": f"{plant.position[0]} {plant.position[1]} 0",
            })
            # Stem
            ET.SubElement(body, "geom", {
                "name": f"plant_{i}_stem",
                "type": "cylinder",
                "size": f"{plant.stem_radius} {plant.height / 2}",
                "pos": f"0 0 {plant.height / 2}",
                "rgba": " ".join(str(c) for c in plant.color),
                "contype": "1",
                "conaffinity": "1",
                "mass": "0.05",
            })
            # Flower/leaf cluster at top
            ET.SubElement(body, "geom", {
                "name": f"plant_{i}_flower",
                "type": "sphere",
                "size": str(plant.stem_radius * 4),
                "pos": f"0 0 {plant.height}",
                "material": "flower_mat",
                "contype": "0",
                "conaffinity": "0",
                "mass": "0.01",
            })

        # Add obstacles
        for i, obs in enumerate(self.obstacles):
            geom_attrs = {
                "name": f"obstacle_{i}",
                "type": obs.shape,
                "pos": f"{obs.position[0]} {obs.position[1]} {obs.position[2]}",
                "rgba": " ".join(str(c) for c in obs.color),
                "contype": "1",
                "conaffinity": "1",
                "mass": "1.0",
            }
            if obs.shape == "box":
                geom_attrs["size"] = " ".join(str(s) for s in obs.size)
            elif obs.shape == "cylinder":
                geom_attrs["size"] = f"{obs.size[0]} {obs.size[1]}"
            ET.SubElement(worldbody, "geom", geom_attrs)

        # Add light source matching direction
        light_pos = -self.light_direction * 10
        ET.SubElement(worldbody, "light", {
            "name": "env_light",
            "pos": f"{light_pos[0]} {light_pos[1]} {light_pos[2]}",
            "dir": f"{self.light_direction[0]} {self.light_direction[1]} {self.light_direction[2]}",
            "diffuse": "0.8 0.8 0.8",
            "specular": "0.3 0.3 0.3",
        })

        return ET.tostring(root, encoding="unicode")

    def compute_wind_force(self, fly_velocity: np.ndarray) -> np.ndarray:
        """Compute aerodynamic drag force from wind on the fly.

        Uses a simplified drag model: F = 0.5 * Cd * A * rho * v_rel^2

        Args:
            fly_velocity: Current fly velocity [vx, vy, vz]

        Returns:
            Force vector [fx, fy, fz]
        """
        if self.wind_speed == 0:
            return np.zeros(3)

        wind_velocity = self.wind * self.wind_speed
        relative_velocity = wind_velocity - fly_velocity

        # Simplified drag: proportional to relative velocity squared
        drag_coefficient = 0.01  # Scaled for simulation
        speed_sq = np.dot(relative_velocity, relative_velocity)
        if speed_sq < 1e-10:
            return np.zeros(3)

        speed = np.sqrt(speed_sq)
        drag_direction = relative_velocity / speed
        force = drag_coefficient * speed_sq * drag_direction
        return force

    def sample_environment(self, position: np.ndarray) -> dict:
        """Sample environmental conditions at a 3D position.

        Returns odorant concentration, temperature, and light intensity
        based on distance to environmental objects.

        Args:
            position: [x, y, z] sample position

        Returns:
            Dictionary with keys: odorant, temperature, light_intensity, wind
        """
        position = np.asarray(position, dtype=float)

        # Odorant: sum of contributions from all fruits (inverse-square falloff)
        odorant = 0.0
        for fruit in self.fruits:
            dist = np.linalg.norm(position - fruit.position)
            if dist < 0.01:
                dist = 0.01  # Avoid division by zero
            odorant += fruit.sugar_concentration / (1.0 + dist * dist)

        # Temperature: base + height gradient
        temperature = self.temperature_base + self.temperature_gradient * position[2]

        # Light intensity: cosine of angle between surface normal (up) and light
        up = np.array([0.0, 0.0, 1.0])
        light_intensity = max(0.0, -np.dot(self.light_direction, up))

        # Wind at position (uniform for now)
        wind_at_pos = self.wind * self.wind_speed

        return {
            "odorant": float(np.clip(odorant, 0.0, 1.0)),
            "temperature": float(temperature),
            "light_intensity": float(light_intensity),
            "wind": wind_at_pos.copy(),
        }

    def get_nearest_fruit(self, position: np.ndarray) -> Optional[tuple]:
        """Find the nearest fruit and its distance.

        Args:
            position: [x, y, z] query position

        Returns:
            Tuple of (fruit_index, distance) or None if no fruits
        """
        if not self.fruits:
            return None

        position = np.asarray(position, dtype=float)
        distances = [
            np.linalg.norm(position - f.position)
            for f in self.fruits
        ]
        idx = int(np.argmin(distances))
        return idx, distances[idx]

    @staticmethod
    def foraging_arena() -> "PhysicsEnvironment":
        """Create a standard foraging arena with fruits, plants, and obstacles.

        Returns:
            Pre-configured PhysicsEnvironment
        """
        env = PhysicsEnvironment(arena_size=(8.0, 8.0))

        # Fruits at various locations
        env.add_fruit(pos=[2.0, 3.0, 0.1], radius=0.15, sugar=0.9)
        env.add_fruit(pos=[-1.0, 2.0, 0.1], radius=0.1, sugar=0.5)
        env.add_fruit(pos=[3.0, -1.0, 0.1], radius=0.12, sugar=0.7)

        # Plants
        env.add_plant(pos=[0.0, -2.0], height=3.0)
        env.add_plant(pos=[-2.0, 1.0], height=2.5)

        # Obstacles
        env.add_obstacle(
            pos=[1.0, 1.0, 0.3], size=[0.5, 0.5, 0.6], shape="box"
        )
        env.add_obstacle(
            pos=[-2.0, -2.0, 0.4], size=[0.3, 0.8], shape="cylinder"
        )

        # Mild wind from east
        env.set_wind(direction=[1, 0, 0], speed=0.5)

        # Sunlight from above-right
        env.set_light(direction=[0.3, 0.0, -0.95])

        return env

    @staticmethod
    def empty_arena(size: float = 10.0) -> "PhysicsEnvironment":
        """Create an empty arena with just walls.

        Args:
            size: Arena size (width and depth)

        Returns:
            Empty PhysicsEnvironment with walls
        """
        return PhysicsEnvironment(arena_size=(size, size))

    @staticmethod
    def obstacle_course() -> "PhysicsEnvironment":
        """Create an obstacle course for locomotion testing.

        Returns:
            PhysicsEnvironment with dense obstacles
        """
        env = PhysicsEnvironment(arena_size=(10.0, 10.0))

        # Row of boxes
        for x in range(-3, 4, 2):
            env.add_obstacle(
                pos=[float(x), 0.0, 0.2],
                size=[0.3, 0.3, 0.4],
                shape="box",
            )

        # Scattered cylinders
        env.add_obstacle(pos=[0.0, 2.0, 0.3], size=[0.2, 0.6], shape="cylinder")
        env.add_obstacle(pos=[2.0, -2.0, 0.3], size=[0.2, 0.6], shape="cylinder")
        env.add_obstacle(pos=[-2.0, 2.0, 0.3], size=[0.15, 0.5], shape="cylinder")

        return env

    def __repr__(self) -> str:
        return (
            f"PhysicsEnvironment("
            f"arena={self.arena_size}, "
            f"fruits={len(self.fruits)}, "
            f"plants={len(self.plants)}, "
            f"obstacles={len(self.obstacles)}, "
            f"wind_speed={self.wind_speed:.1f})"
        )
