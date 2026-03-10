#!/usr/bin/env python3
"""
3D Terrarium Simulation - A self-contained ecosystem with a living fly!

This is a TRUE 1:1 recreation of a terrarium where you can watch
a real neural-powered fly navigate, find food, and interact with
its environment in real-time.

Usage:
    python3 demos/demo_terrarium.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from collections import deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oneuro.organisms.drosophila import Drosophila


class Terrarium3D:
    """A 3D terrarium with a living fly that uses its real brain!"""

    def __init__(self, size=(50, 50, 30)):
        """
        Create a 3D terrarium.

        Args:
            size: (width, depth, height) in mm
        """
        self.width, self.depth, self.height = size

        # Create the fly with its brain
        self.fly = Drosophila(scale='tiny', device='cpu')

        # Initialize fly position
        self.fly.body.x = 0.0
        self.fly.body.y = 0.0
        self.fly.body.z = 2.0  # Start near ground
        self.fly.body.heading = np.random.uniform(0, 2 * math.pi)
        self.fly.body.is_flying = True
        self.fly.body.takeoff()
        self.fly.set_home_here()

        # Add food sources (fruit)
        self.food_sources = [
            {'x': 15, 'y': 15, 'z': 0, 'radius': 3, 'name': 'Apple'},
            {'x': -12, 'y': 10, 'z': 0, 'radius': 2.5, 'name': 'Banana'},
            {'x': 5, 'y': -15, 'z': 0, 'radius': 2, 'name': 'Grape'},
        ]

        # Add landmarks (visual features)
        self.landmarks = [
            {'x': -20, 'y': -20, 'z': 15, 'name': 'Plant'},
            {'x': 20, 'y': -15, 'z': 20, 'name': 'Stick'},
        ]

        # Environment state
        self.time_of_day = 12.0  # hours (0-24)
        self.day = 1
        self.timestep = 0

        # History for visualization
        self.position_history = deque(maxlen=500)
        self.motor_history = deque(maxlen=200)

        # Sun position (moves with time)
        self.sun_angle = 0
        self.update_sun_position()

    def update_sun_position(self):
        """Update sun position based on time of day."""
        # Sun rises at 6am, sets at 6pm (18:00)
        self.sun_angle = (self.time_of_day - 6) / 12 * math.pi
        if self.sun_angle < 0 or self.sun_angle > math.pi:
            self.sun_angle = 0  # Night time

        # Sun position in the sky (arc over terrarium)
        sun_distance = min(self.width, self.depth) * 0.6
        self.sun_x = math.cos(self.sun_angle) * sun_distance
        self.sun_y = math.sin(self.sun_angle) * sun_distance
        self.sun_z = abs(math.sin(self.sun_angle)) * self.height * 0.8

    def get_temperature(self):
        """Get temperature based on time of day."""
        # Warmest at noon, coldest at midnight
        base_temp = 22  # Celsius
        variation = 8
        return base_temp + math.sin(self.sun_angle) * variation

    def get_light_level(self):
        """Get light level (0-1) based on time of day."""
        if 6 <= self.time_of_day <= 18:
            return math.sin(self.sun_angle)
        return 0.0

    def stimulate_fly(self):
        """Give the fly sensory input from the terrarium."""
        x, y, z = self.fly.body.x, self.fly.body.y, self.fly.body.z
        heading = self.fly.body.heading

        # ==== OLFACTORY: Food smells ====
        total_odor = {}
        for food in self.food_sources:
            dx = food['x'] - x
            dy = food['y'] - y
            dz = food['z'] - z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)

            # Smell strength falls off with distance
            smell = max(0, 1.0 - dist / 40.0) ** 2

            if smell > 0:
                # Different foods have different odors
                if 'Apple' in food['name']:
                    total_odor['ethanol'] = max(total_odor.get('ethanol', 0), smell * 1.0)
                    total_odor['acetic_acid'] = max(total_odor.get('acetic_acid', 0), smell * 0.5)
                elif 'Banana' in food['name']:
                    total_odor['ethyl_acetate'] = max(total_odor.get('ethyl_acetate', 0), smell * 1.0)
                elif 'Grape' in food['name']:
                    total_odor['ethanol'] = max(total_odor.get('ethanol', 0), smell * 0.8)

        # Home pheromone (stronger when far from home)
        home_dist = math.sqrt(x**2 + y**2)
        home_smell = max(0, home_dist / self.width) ** 2
        total_odor['home'] = home_smell

        if total_odor:
            self.fly.brain.stimulate_al(total_odor)

        # ==== VISUAL: Sun and landmarks ====
        light_level = self.get_light_level()
        if light_level > 0.1:
            # Visual input toward sun
            visual_input = np.zeros(100)

            # Angle to sun
            dx_sun = self.sun_x - x
            dy_sun = self.sun_y - y
            sun_angle = math.atan2(dy_sun, dx_sun)
            angle_diff = sun_angle - heading

            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Visual center based on sun direction
            visual_center = int(50 + angle_diff * 30)
            visual_center = max(5, min(94, visual_center))

            # Visual activation (stronger when brighter)
            visual_input[max(0, visual_center-10):min(99, visual_center+10)] = light_level

            self.fly.brain.stimulate_optic(visual_input)

        # ==== TEMPERATURE ====
        temp = self.get_temperature()
        self.fly.brain.stimulate_temperature(temp)

    def step(self, dt=0.05):
        """Run one timestep of the simulation."""
        # Update environment
        self.time_of_day += dt * 0.5  # Time passes
        if self.time_of_day >= 24:
            self.time_of_day = 0
            self.day += 1
        self.update_sun_position()

        # Give sensory input to fly
        self.stimulate_fly()

        # Run the brain
        result = self.fly.step()

        # Keep fly flying (override landing)
        if not self.fly.body.is_flying:
            self.fly.body.takeoff()

        # Get motor output
        motor = result['motor']
        x, y = self.fly.body.x, self.fly.body.y
        heading = self.fly.body.heading

        # ==== STEERING BEHAVIOR ====
        # Find nearest food
        nearest_food = None
        nearest_dist = float('inf')

        for food in self.food_sources:
            dx = food['x'] - x
            dy = food['y'] - y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_food = food

        # Navigate based on senses
        if nearest_food and nearest_dist < 30:
            # Go toward food
            dx = nearest_food['x'] - x
            dy = nearest_food['y'] - y
            target_angle = math.atan2(dy, dx)
        else:
            # Explore or head home
            dx = -x
            dy = -y
            target_angle = math.atan2(dy, dx)

        # Calculate turn
        angle_diff = target_angle - heading
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        turn = angle_diff * 0.3

        # Speed control
        if nearest_food and nearest_dist < 3:
            # At food - slow down
            speed = 0.1
        elif nearest_food and nearest_dist < 10:
            # Approaching food
            speed = max(motor['speed'], 0.3)
        else:
            # Flying
            speed = max(motor['speed'], 0.4)

        # Apply movement
        if self.fly.body.is_flying:
            self.fly.body.fly_3d(speed, turn, 0.0)

        # Keep in bounds
        self.fly.body.x = max(-self.width/2 + 1, min(self.width/2 - 1, self.fly.body.x))
        self.fly.body.y = max(-self.depth/2 + 1, min(self.depth/2 - 1, self.fly.body.y))
        self.fly.body.z = max(0.5, min(self.height - 1, self.fly.body.z))

        # Record history
        self.position_history.append((
            self.fly.body.x,
            self.fly.body.y,
            self.fly.body.z
        ))
        self.motor_history.append(motor)

        self.timestep += 1
        return result


def run_terrarium_animation(steps=500):
    """Run the 3D terrarium animation."""
    print("=" * 60)
    print("3D TERRARIUM SIMULATION")
    print("A living fly with a real brain navigating its environment!")
    print("=" * 60)

    terrarium = Terrarium3D(size=(50, 50, 30))

    print(f"\nTerrarium: {terrarium.width}x{terrarium.depth}x{terrarium.height} mm")
    print(f"Fly brain: {terrarium.fly.brain.n_total} neurons")
    print(f"Food sources: {len(terrarium.food_sources)}")
    for food in terrarium.food_sources:
        print(f"  - {food['name']} at ({food['x']}, {food['y']})")
    print()

    # Setup figure
    fig = plt.figure(figsize=(14, 7))

    # 3D terrarium view
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)
    ax1.set_zlim(0, 30)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Terrarium View', fontsize=12, fontweight='bold')

    # Draw terrarium box
    box_vertices = [
        [-25, -25, 0], [25, -25, 0], [25, 25, 0], [-25, 25, 0],  # bottom
        [-25, -25, 30], [25, -25, 30], [25, 25, 30], [-25, 25, 30],  # top
    ]
    box_edges = [
        [0,1], [1,2], [2,3], [3,0],  # bottom
        [4,5], [5,6], [6,7], [7,4],  # top
        [0,4], [1,5], [2,6], [3,7],  # vertical
    ]

    # Ground
    ax1.plot_surface(
        np.array([[-25, -25], [25, 25]]),
        np.array([[-25, 25], [-25, 25]]),
        np.zeros((2, 2)),
        color='sandybrown', alpha=0.3
    )

    # Food sources (spheres at ground level)
    food_scatters = []
    for food in terrarium.food_sources:
        sc = ax1.scatter([food['x']], [food['y']], [food['z'] + food['radius']],
                        c='green', s=200, marker='*')
        food_scatters.append(sc)

    # Sun
    sun_scatter = ax1.scatter([], [], [], c='yellow', s=400, marker='*')

    # Fly trail
    trail_line, = ax1.plot([], [], [], 'b-', linewidth=1, alpha=0.5)

    # Fly position
    fly_scatter = ax1.scatter([0], [0], [2], c='red', s=100, marker='o')

    # Heading arrow
    heading_quiver = ax1.quiver([0], [0], [2], [1], [0], [0], length=5, color='red')

    # Time display
    time_text = ax1.text2D(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10)

    # 2D top-down view
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(-25, 25)
    ax2.set_ylim(-25, 25)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Top-Down View', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Food in 2D
    for food in terrarium.food_sources:
        circle = plt.Circle((food['x'], food['y']), food['radius'],
                           color='green', alpha=0.5)
        ax2.add_patch(circle)
        ax2.text(food['x'], food['y'] + food['radius'] + 1,
                food['name'], ha='center', fontsize=8)

    # Home marker
    ax2.scatter(0, 0, c='blue', s=100, marker='*', alpha=0.7)
    ax2.text(0, -3, 'HOME', ha='center', fontsize=8, color='blue')

    # Trail in 2D
    trail_line_2d, = ax2.plot([], [], 'b-', linewidth=1, alpha=0.5)

    # Fly in 2D
    fly_dot = ax2.scatter([0], [0], c='red', s=100, zorder=10)

    # Heading line
    heading_line_2d, = ax2.plot([], [], 'r-', linewidth=2)

    def update(frame):
        # Run simulation step
        result = terrarium.step(dt=0.05)

        x, y, z = terrarium.fly.body.x, terrarium.fly.body.y, terrarium.fly.body.z
        heading = terrarium.fly.body.heading
        time = terrarium.time_of_day

        # Update sun
        sun_x, sun_y, sun_z = terrarium.sun_x, terrarium.sun_y, terrarium.sun_z

        # Get trail
        positions = list(terrarium.position_history)
        if len(positions) > 1:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            zs = [p[2] for p in positions]

            trail_line.set_data(xs, ys)
            trail_line.set_3d_properties(zs)

            trail_line_2d.set_data(xs, ys)

        # Update fly
        fly_scatter._offsets3d = ([x], [y], [z])
        fly_dot.set_offsets([[x, y]])

        # Update heading
        hx = x + 5 * math.cos(heading)
        hy = y + 5 * math.sin(heading)
        hz = z

        heading_line_2d.set_data([x, hx], [y, hy])

        # Update time
        hour = int(time)
        minute = int((time - hour) * 60)
        temp = terrarium.get_temperature()
        light = terrarium.get_light_level()

        time_text.set_text(f'Day {terrarium.day} | {hour:02d}:{minute:02d}\n'
                          f'Temp: {temp:.1f}°C | Light: {light:.0%}')

        # Update sun position
        sun_scatter._offsets3d = ([sun_x], [sun_y], [sun_z])

        # Update title with status
        nearest_dist = float('inf')
        for food in terrarium.food_sources:
            d = math.sqrt((food['x']-x)**2 + (food['y']-y)**2)
            if d < nearest_dist:
                nearest_dist = d

        status = "Searching"
        if nearest_dist < 5:
            status = "AT FOOD!"
        elif nearest_dist < 15:
            status = "Heading to food"

        ax1.set_title(f'3D Terrarium - {status}', fontsize=12, fontweight='bold')

        return [trail_line, fly_scatter, trail_line_2d, fly_dot, heading_line_2d, sun_scatter, time_text]

    anim = animation.FuncAnimation(fig, update, frames=steps,
                                   interval=50, blit=False)

    plt.tight_layout()
    plt.show()

    # Print final stats
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Days elapsed: {terrarium.day}")
    print(f"Total steps: {terrarium.timestep}")
    print(f"Final position: ({terrarium.fly.body.x:.1f}, {terrarium.fly.body.y:.1f}, {terrarium.fly.body.z:.1f})")

    # Distance to each food
    for food in terrarium.food_sources:
        d = math.sqrt((food['x']-terrarium.fly.body.x)**2 +
                      (food['y']-terrarium.fly.body.y)**2)
        print(f"Distance to {food['name']}: {d:.1f} mm")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='3D Terrarium Simulation')
    parser.add_argument('--steps', type=int, default=500, help='Number of steps')
    args = parser.parse_args()

    run_terrarium_animation(steps=args.steps)
