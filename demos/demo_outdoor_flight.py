#!/usr/bin/env python3
"""
Outdoor Flight Simulation Demo -- Neural-Controlled Drone Navigation.

This demo simulates a drone with a fly brain navigating outdoors using:
- GPS-based position tracking (simulated)
- Compass-based heading (sun navigation)
- Landmark-based homing
- Path integration

No actual drone hardware required -- runs entirely in simulation.

Usage:
    python3 demos/demo_outdoor_flight.py
    python3 demos/demo_outdoor_flight.py --experiment phototaxis
    python3 demos/demo_outdoor_flight.py --experiment homing
    python3 demos/demo_outdoor_flight.py --experiment exploration
"""

import argparse
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.animation as animation
from collections import deque

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oneuro.organisms.drosophila import Drosophila
from oneuro.robot import SensorFusion


class SimulatedDrone:
    """Simulated drone for outdoor navigation experiments."""

    def __init__(self, start_pos=(0.0, 0.0, 10.0), start_heading=0.0):
        self.x, self.y, self.z = start_pos
        self.heading = start_heading
        self.velocity = (0.0, 0.0, 0.0)

        # Simulated sensors
        self.gps_noise = 0.5  # meters
        self.imu_noise = 0.01

        # Environment
        self.wind = (0.0, 0.0, 0.0)  # wind vector
        self.sun_angle = 0.0  # sun azimuth (compass direction)

    def set_wind(self, wx, wy):
        """Set wind vector."""
        self.wind = (wx, wy, 0.0)

    def set_sun(self, angle):
        """Set sun position for compass."""
        self.sun_angle = angle

    def apply_control(self, roll, pitch, yaw_rate, thrust, dt=0.02):
        """Apply control inputs and update position.

        Args:
            roll: Roll angle in degrees
            pitch: Pitch angle in degrees
            yaw_rate: Yaw rate in degrees/sec
            thrust: Thrust 0-1
            dt: Time step
        """
        # Update heading
        self.heading += math.radians(yaw_rate) * dt

        # Normalize heading
        while self.heading > math.pi:
            self.heading -= 2 * math.pi
        while self.heading < -math.pi:
            self.heading += 2 * math.pi

        # Simple physics model
        speed = thrust * 10.0  # m/s at full thrust

        # Account for wind
        wind_eff = 0.5
        vx = speed * math.cos(self.heading) - self.wind[0] * wind_eff
        vy = speed * math.sin(self.heading) - self.wind[1] * wind_eff

        # Update position
        self.x += vx * dt
        self.y += vy * dt
        self.velocity = (vx, vy, 0.0)

    def get_gps(self):
        """Get GPS reading with noise."""
        noise = np.random.randn(3) * self.gps_noise
        return (
            self.x + noise[0],
            self.y + noise[1],
            self.z + noise[2]
        )

    def get_imu(self):
        """Get IMU readings with noise."""
        gyro_noise = np.random.randn(3) * self.imu_noise
        return {
            'gyro': gyro_noise,
            'accel': (0, 0, -9.8),  # Simplified
        }


class Landmark:
    """Outdoor landmark for visual navigation."""

    def __init__(self, x, y, name, color='green', size=5.0):
        self.x = x
        self.y = y
        self.name = name
        self.color = color
        self.size = size
        # Visual features (simplified as random for simulation)
        self.features = np.random.rand(32).astype(np.float32)


class OutdoorWorld:
    """Outdoor environment simulation."""

    def __init__(self, size=500.0):
        self.size = size
        self.landmarks = []
        self.target = None

        # Add some landmarks
        self.add_landmark(50, 50, "Tree1", "darkgreen", 8.0)
        self.add_landmark(-30, 80, "Tree2", "darkgreen", 7.0)
        self.add_landmark(80, -20, "Building", "gray", 10.0)
        self.add_landmark(-50, -40, "Rock", "sienna", 4.0)
        self.add_landmark(0, 100, "Nest", "brown", 5.0)

    def add_landmark(self, x, y, name, color="green", size=5.0):
        """Add a landmark."""
        self.landmarks.append(Landmark(x, y, name, color, size))

    def set_target(self, x, y):
        """Set navigation target."""
        self.target = (x, y)

    def get_visual_features(self, drone_x, drone_y, drone_heading):
        """Get visual features from current position (simplified)."""
        features = np.zeros(32, dtype=np.float32)

        # Features based on visible landmarks
        for i, lm in enumerate(self.landmarks[:8]):
            dx = lm.x - drone_x
            dy = lm.y - drone_y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < 100:  # Visible range
                angle = math.atan2(dy, dx) - drone_heading
                features[i * 4] = lm.features[0]
                features[i * 4 + 1] = 1.0 / (dist + 1)
                features[i * 4 + 2] = math.sin(angle)
                features[i * 4 + 3] = math.cos(angle)

        return features


def run_experiment(experiment_type="phototaxis", scale="tiny", steps=500):
    """Run an outdoor navigation experiment."""

    print(f"\n{'='*60}")
    print(f"  OUTDOOR FLIGHT SIMULATION -- {experiment_type.upper()}")
    print(f"{'='*60}\n")

    # Create simulated drone
    drone = SimulatedDrone(start_pos=(0.0, 0.0, 10.0))

    # Create world
    world = OutdoorWorld(size=500)
    world.set_target(0, 100)  # Default target

    # Create fly brain
    fly = Drosophila(scale=scale)
    fly.body.x = drone.x
    fly.body.y = drone.y
    fly.body.heading = drone.heading

    # Set sun for compass navigation
    drone.set_sun(math.pi / 4)  # Sun at NE
    fly.set_sun_heading(drone.sun_angle)

    # Set home position
    fly.set_home_here()
    home_x, home_y = drone.x, drone.y

    # Add some landmarks to brain
    for lm in world.landmarks[:5]:
        features = world.get_visual_features(drone.x, drone.y, drone.heading)
        fly.learn_landmark(lm.name, features)

    # Experiment-specific setup
    if experiment_type == "phototaxis":
        drone.set_sun(math.pi / 2)  # Sun at east
        fly.set_sun_heading(drone.sun_angle)
        print("Experiment: Phototaxis (fly toward sun)")
        print(f"  Sun position: {math.degrees(drone.sun_angle):.1f}°\n")

    elif experiment_type == "homing":
        world.set_target(home_x, home_y)
        print("Experiment: Homing (return to start)")
        print(f"  Home position: ({home_x:.1f}, {home_y:.1f})\n")

    elif experiment_type == "exploration":
        print("Experiment: Exploration (random walk with landmark learning)")
        print("  Learning landmarks along the way\n")

    elif experiment_type == "landmark_navigation":
        print("Experiment: Landmark Navigation (use learned landmarks)")
        # Set target to a landmark
        if world.landmarks:
            target_lm = world.landmarks[0]
            world.set_target(target_lm.x, target_lm.y)
            print(f"  Target: {target_lm.name} at ({target_lm.x:.1f}, {target_lm.y:.1f})\n")

    # Data recording
    positions = deque(maxlen=steps)
    headings = []
    motor_outputs = []
    landmarks_seen = []

    # Run simulation
    print("Running simulation...")
    dt = 0.02

    for step in range(steps):
        # Get sensor data
        gps = drone.get_gps()
        imu = drone.get_imu()

        # Update body position to match drone
        fly.body.x = drone.x
        fly.body.y = drone.y
        fly.body.heading = drone.heading

        # Update brain navigation state
        fly.update_navigation_from_movement(dt)

        # Get visual features
        visual_features = world.get_visual_features(
            drone.x, drone.y, drone.heading
        )

        # Try to recognize landmarks
        match = fly.recognize_landmark(visual_features)
        if match:
            landmarks_seen.append((step, match[0]))

        # --- Neural Control ---
        if experiment_type == "phototaxis":
            # Turn toward sun
            turn = fly.head_toward_compass(drone.sun_angle)
            speed = 0.7

        elif experiment_type == "homing":
            # Head toward home
            turn = fly.head_toward_home()
            dist_home = math.sqrt((drone.x - home_x)**2 + (drone.y - home_y)**2)
            speed = 0.8 if dist_home > 10 else 0.2

        elif experiment_type == "exploration":
            # Random walk with occasional turns
            if step % 50 == 0:
                turn = np.random.uniform(-1, 1)
            else:
                turn = turn * 0.9 if 'turn' in locals() else 0.0
            speed = 0.6

            # Learn new landmarks occasionally
            if step % 30 == 0:
                features = world.get_visual_features(drone.x, drone.y, drone.heading)
                fly.learn_landmark(f"landmark_{step}", features)

        elif experiment_type == "landmark_navigation":
            # Use landmark matching
            if match:
                # Turn toward matched landmark
                lm_id, lm_x, lm_y = match
                dx = lm_x - drone.x
                dy = lm_y - drone.y
                target_angle = math.atan2(dy, dx)
                turn = fly.head_toward_compass(target_angle)
                speed = 0.7
            else:
                # Explore
                turn = np.random.uniform(-0.5, 0.5)
                speed = 0.5

        else:
            turn = 0.0
            speed = 0.5

        # Apply to drone
        roll = turn * 15  # Convert to degrees
        yaw_rate = turn * 30
        thrust = speed * 0.7

        drone.apply_control(roll, 0, yaw_rate, thrust, dt)

        # Record data
        positions.append((drone.x, drone.y, drone.z))
        headings.append(drone.heading)
        motor_outputs.append((speed, turn))

        # Progress output
        if step % 100 == 0:
            print(f"  Step {step}/{steps}: "
                  f"pos=({drone.x:.1f}, {drone.y:.1f}, {drone.z:.1f}), "
                  f"heading={math.degrees(drone.heading):.1f}°")

    # Results
    print("\n" + "="*60)
    print("  SIMULATION RESULTS")
    print("="*60)

    # Calculate final distance to target
    if world.target:
        target_x, target_y = world.target
        final_dist = math.sqrt((drone.x - target_x)**2 + (drone.y - target_y)**2)
        print(f"\nTarget: ({target_x:.1f}, {target_y:.1f})")
        print(f"Final position: ({drone.x:.1f}, {drone.y:.1f}, {drone.z:.1f})")
        print(f"Distance to target: {final_dist:.1f}m")
        print(f"Success: {'YES' if final_dist < 20 else 'NO'}")

    # Distance from home
    dist_home = math.sqrt((drone.x - home_x)**2 + (drone.y - home_y)**2)
    print(f"\nDistance from home: {dist_home:.1f}m")

    # Path length
    path_length = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        path_length += math.sqrt(dx*dx + dy*dy)
    print(f"Total path length: {path_length:.1f}m")

    # Efficiency
    if world.target:
        efficiency = math.sqrt((target_x - home_x)**2 + (target_y - home_y)**2) / max(path_length, 1)
        print(f"Path efficiency: {efficiency:.2f}")

    print(f"\nLandmarks recognized: {len(landmarks_seen)}")
    if landmarks_seen:
        print(f"  First recognition at step: {landmarks_seen[0][0]}")

    return {
        'positions': list(positions),
        'headings': headings,
        'motor_outputs': motor_outputs,
        'landmarks_seen': landmarks_seen,
        'target': world.target,
        'home': (home_x, home_y),
        'experiment': experiment_type,
    }


def visualize_results(results):
    """Create visualization of the simulation results."""

    positions = results['positions']
    if not positions:
        print("No positions to visualize")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Outdoor Flight Simulation: {results['experiment'].title()}",
                 fontsize=16, fontweight='bold')

    # 1. 2D Trajectory
    ax1 = axes[0, 0]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    # Draw trail
    ax1.plot(xs, ys, 'b-', alpha=0.5, linewidth=1, label='Flight path')

    # Draw start and end
    ax1.scatter(xs[0], ys[0], c='green', s=200, marker='^', zorder=5, label='Start')
    ax1.scatter(xs[-1], ys[-1], c='red', s=200, marker='v', zorder=5, label='End')

    # Draw target if exists
    if results['target']:
        ax1.scatter(results['target'][0], results['target'][1],
                   c='orange', s=300, marker='*', zorder=4, label='Target')

    # Draw home
    ax1.scatter(results['home'][0], results['home'][1],
               c='blue', s=200, marker='s', zorder=4, label='Home', alpha=0.5)

    # Draw landmarks as boxes
    world = OutdoorWorld()
    for lm in world.landmarks:
        rect = FancyBboxPatch((lm.x - lm.size/2, lm.y - lm.size/2),
                              lm.size, lm.size,
                              boxstyle="round,pad=0.1",
                              facecolor=lm.color, alpha=0.5, edgecolor='black')
        ax1.add_patch(rect)
        ax1.annotate(lm.name, (lm.x, lm.y), fontsize=8, ha='center')

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Flight Trajectory (Top View)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 2. Heading over time
    ax2 = axes[0, 1]
    headings = results['headings']
    ax2.plot(np.degrees(headings), 'b-', alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=180, color='r', linestyle='--', alpha=0.3)
    ax2.axhline(y=-180, color='r', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Heading (degrees)')
    ax2.set_title('Heading Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-200, 200)

    # 3. Motor outputs
    ax3 = axes[1, 0]
    motor_outputs = results['motor_outputs']
    speeds = [m[0] for m in motor_outputs]
    turns = [m[1] for m in motor_outputs]

    ax3.plot(speeds, 'g-', alpha=0.7, label='Speed')
    ax3.plot(turns, 'b-', alpha=0.7, label='Turn')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Control Output')
    ax3.set_title('Motor Outputs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 3D trajectory
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    zs = [p[2] for p in positions]
    ax4.plot(xs, ys, zs, 'b-', alpha=0.5)
    ax4.scatter(xs[0], ys[0], zs[0], c='green', s=100, marker='^')
    ax4.scatter(xs[-1], ys[-1], zs[-1], c='red', s=100, marker='v')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title('3D Flight Path')

    plt.tight_layout()
    plt.savefig('outdoor_flight_result.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: outdoor_flight_result.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Outdoor Flight Simulation -- Neural-Controlled Drone"
    )
    parser.add_argument(
        '--exp', '--experiment',
        choices=['phototaxis', 'homing', 'exploration', 'landmark_navigation'],
        default='phototaxis',
        help='Type of navigation experiment'
    )
    parser.add_argument(
        '--scale', '-s',
        choices=['tiny', 'small', 'medium'],
        default='tiny',
        help='Brain scale'
    )
    parser.add_argument(
        '--steps', '-n',
        type=int,
        default=500,
        help='Number of simulation steps'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Show visualization'
    )

    args = parser.parse_args()

    # Run experiment
    results = run_experiment(
        experiment_type=args.exp,
        scale=args.scale,
        steps=args.steps
    )

    # Visualize if requested
    if args.visualize:
        visualize_results(results)


if __name__ == '__main__':
    main()
