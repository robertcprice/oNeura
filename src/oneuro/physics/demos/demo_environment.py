#!/usr/bin/env python3
"""
Physics Environment Demo

Standalone demo showing 3D environment creation, MJCF generation,
and environment sampling — all without requiring MuJoCo to be installed.

Usage:
    cd /Users/bobbyprice/projects/oNeuro
    PYTHONPATH=src python3 src/oneuro/physics/demo_environment.py
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from oneuro.physics.physics_environment import PhysicsEnvironment


def demo_environment_creation():
    """Demonstrate building environments programmatically."""
    print("=" * 60)
    print("  PHYSICS ENVIRONMENT DEMO")
    print("=" * 60)

    # 1. Create foraging arena
    print("\n1. FORAGING ARENA (preset)")
    print("-" * 40)
    env = PhysicsEnvironment.foraging_arena()
    print(f"  {env}")
    for i, fruit in enumerate(env.fruits):
        print(f"  Fruit {i}: pos={fruit.position}, sugar={fruit.sugar_concentration}")
    for i, plant in enumerate(env.plants):
        print(f"  Plant {i}: pos={plant.position}, height={plant.height}")
    for i, obs in enumerate(env.obstacles):
        print(f"  Obstacle {i}: pos={obs.position}, shape={obs.shape}")

    # 2. Generate MJCF
    print("\n2. MJCF GENERATION")
    print("-" * 40)
    mjcf_xml = env.generate_mjcf()
    print(f"  Generated XML: {len(mjcf_xml)} characters")
    print(f"  Contains 'fruit_0': {'fruit_0' in mjcf_xml}")
    print(f"  Contains 'plant_0': {'plant_0' in mjcf_xml}")
    print(f"  Contains 'wall_north': {'wall_north' in mjcf_xml}")
    print(f"  Contains 'obstacle_0': {'obstacle_0' in mjcf_xml}")

    # 3. Environment sampling
    print("\n3. ENVIRONMENT SAMPLING")
    print("-" * 40)

    # Sample at various positions
    positions = [
        [0.0, 0.0, 2.5],   # Center, fly height
        [2.0, 3.0, 0.1],   # Near fruit 0 (sugar=0.9)
        [-1.0, 2.0, 0.1],  # Near fruit 1 (sugar=0.5)
        [5.0, 5.0, 0.1],   # Far corner
    ]

    for pos in positions:
        sample = env.sample_environment(pos)
        print(f"  pos={pos}")
        print(f"    odorant={sample['odorant']:.4f}, "
              f"temp={sample['temperature']:.1f}C, "
              f"light={sample['light_intensity']:.2f}")

    # 4. Wind force computation
    print("\n4. WIND FORCES")
    print("-" * 40)
    velocities = [
        np.zeros(3),           # Stationary fly
        np.array([0.5, 0, 0]), # Moving with wind
        np.array([-1, 0, 0]),  # Moving against wind
    ]
    for vel in velocities:
        force = env.compute_wind_force(vel)
        print(f"  fly_vel={vel} -> wind_force={force}")

    # 5. Nearest fruit finding
    print("\n5. NEAREST FRUIT")
    print("-" * 40)
    query_pos = np.array([0.0, 0.0, 0.1])
    result = env.get_nearest_fruit(query_pos)
    if result:
        idx, dist = result
        print(f"  From {query_pos}: nearest fruit #{idx} at distance {dist:.2f}")
        print(f"    sugar_concentration={env.fruits[idx].sugar_concentration}")

    # 6. Obstacle course preset
    print("\n6. OBSTACLE COURSE (preset)")
    print("-" * 40)
    course = PhysicsEnvironment.obstacle_course()
    print(f"  {course}")
    course_xml = course.generate_mjcf()
    print(f"  Generated XML: {len(course_xml)} characters")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("  All environment features work without MuJoCo installed!")
    print("=" * 60)


if __name__ == "__main__":
    demo_environment_creation()
