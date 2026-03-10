#!/usr/bin/env python3
"""
BoB Features Visualization -- Attention-Grabbing Demo.

This demo showcases the BoB (Brains on Board) navigation features
in a visually compelling way that demonstrates the capabilities.

Usage:
    python3 demos/demo_bob_features.py
    python3 demos/demo_bob_features.py --demo ring_attractor
    python3 demos/demo_bob_features.py --demo path_integration
    python3 demos/demo_bob_features.py --demo landmarks
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from collections import deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oneuro.organisms.drosophila import Drosophila


class BoBVisualization:
    """Visualization of BoB navigation features."""

    def __init__(self, demo_type='all'):
        self.demo_type = demo_type
        self.fly = Drosophila(scale='tiny')

        # Visualization state
        self.ring_attractor_data = deque(maxlen=100)
        self.path_data = deque(maxlen=500)
        self.landmark_data = []

    def run_ring_attractor_demo(self):
        """Visualize ring attractor heading representation."""
        print("\n" + "="*60)
        print("  RING ATTRACTOR VISUALIZATION")
        print("  (Compass neurons in Central Complex)")
        print("="*60)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('BoB: Ring Attractor Heading System', fontsize=16, fontweight='bold')

        # Initialize data
        n_points = 200
        angular_velocities = np.concatenate([
            np.zeros(30),  # Still
            np.ones(30) * 0.5,  # Turn right
            np.zeros(30),  # Still
            np.ones(30) * -0.5,  # Turn left
            np.zeros(30),  # Still
            np.ones(40) * 0.3,  # Slow right
        ])

        # Run simulation
        headings = []
        ring_states = []

        for i, ang_vel in enumerate(angular_velocities):
            self.fly.brain.update_heading_ring_attractor(ang_vel, dt=0.1)
            headings.append(self.fly.brain._heading)

            # Get ring attractor state
            if self.fly.brain._ring_attractor_state is not None:
                state = self.fly.brain._ring_attractor_state.cpu().numpy()
                ring_states.append(state)
            else:
                ring_states.append(np.zeros(16))

        # Plot 1: Heading over time
        ax1.plot(np.degrees(headings), 'b-', linewidth=2)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.fill_between(range(len(headings)), np.degrees(headings), 0, alpha=0.3)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Heading (degrees)', fontsize=12)
        ax1.set_title('Heading Response to Angular Velocity', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Ring attractor activity (snapshot)
        if ring_states:
            # Show final state
            final_state = ring_states[-1]
            angles = np.linspace(0, 2*np.pi, len(final_state), endpoint=False)

            # Polar plot of ring attractor
            ax2 = fig.add_subplot(1, 2, 2, projection='polar')
            ax2.plot(angles, final_state, 'b-', linewidth=2)
            ax2.fill(angles, final_state, alpha=0.3)
            ax2.set_title('Ring Attractor Activity\n(Compass Neurons)', fontsize=14)
            ax2.set_theta_zero_location('E')

        plt.tight_layout()
        plt.savefig('bob_ring_attractor.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: bob_ring_attractor.png")
        plt.show()

    def run_path_integration_demo(self):
        """Visualize path integration (dead reckoning)."""
        print("\n" + "="*60)
        print("  PATH INTEGRATION VISUALIZATION")
        print("  (Dead Reckoning in Central Complex)")
        print("="*60)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('BoB: Path Integration (Dead Reckoning)', fontsize=16, fontweight='bold')

        # Simulate a foraging path
        np.random.seed(42)
        n_points = 300

        # Generate realistic movement
        positions = [(0, 0)]
        velocities = []
        heading = 0.0
        x, y = 0.0, 0.0

        for i in range(n_points):
            # Random walk with momentum
            if i % 50 < 25:
                # Straight-ish
                turn = np.random.normal(0, 0.1)
            else:
                # Turn
                turn = np.random.normal(0.3, 0.1)

            heading += turn
            heading = heading % (2 * np.pi)

            speed = 0.1 + np.random.uniform(-0.02, 0.02)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)

            x += vx
            y += vy

            positions.append((x, y))
            velocities.append((vx, vy))

            # Update path integration
            self.fly.brain.update_path_integration(vx, vy, dt=0.1)

        # Extract data
        path_x = [p[0] for p in positions]
        path_y = [p[1] for p in positions]

        # Path integration estimate
        int_x = [self.fly.brain._path_integration_x] * len(positions)
        int_y = [self.fly.brain._path_integration_y] * len(positions)

        # Plot 1: Actual path vs integrated
        ax1.plot(path_x, path_y, 'b-', linewidth=1.5, label='Actual Path', alpha=0.7)
        ax1.scatter(path_x[0], path_y[0], c='green', s=200, marker='^', zorder=5, label='Start')
        ax1.scatter(path_x[-1], path_y[-1], c='red', s=200, marker='v', zorder=5, label='End')
        ax1.set_xlabel('X Position (m)', fontsize=12)
        ax1.set_ylabel('Y Position (m)', fontsize=12)
        ax1.set_title('Actual Movement Path', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Plot 2: Integrated position over time
        ax2.plot(range(len(positions)), path_x, 'b-', linewidth=1.5, label='Actual X', alpha=0.7)
        ax2.plot(range(len(positions)), [p[0] for p in positions], 'g-', linewidth=1.5, label='Actual Y', alpha=0.7)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Position (m)', fontsize=12)
        ax2.set_title('Path Integration Estimates', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('bob_path_integration.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: bob_path_integration.png")
        plt.show()

    def run_landmark_demo(self):
        """Visualize landmark learning and recognition."""
        print("\n" + "="*60)
        print("  LANDMARK VISUALIZATION")
        print("  (Visual Place Recognition)")
        print("="*60)

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.suptitle('BoB: Landmark-Based Navigation', fontsize=16, fontweight='bold')

        # Add landmarks at known positions
        landmarks = [
            ('Tree1', 50, 50, 'green'),
            ('Tree2', -30, 80, 'darkgreen'),
            ('Rock', -50, -40, 'sienna'),
            ('Building', 80, -20, 'gray'),
            ('Nest', 0, 100, 'brown'),
        ]

        # Draw landmarks
        for name, lx, ly, color in landmarks:
            # Create feature vector
            features = np.random.rand(32).astype(np.float32)
            features[0] = lx / 100  # Position encoding

            # Learn landmark
            self.fly.brain.add_landmark(name, lx, ly, features)

            # Draw
            rect = FancyBboxPatch((lx-5, ly-5), 10, 10,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color, alpha=0.6, edgecolor='black')
            ax.add_patch(rect)
            ax.annotate(name, (lx, ly+12), ha='center', fontsize=10, fontweight='bold')

        # Simulate fly visiting landmarks
        visits = [(0, 0), (50, 50), (80, -20), (0, 100), (-50, -40)]

        for i, (vx, vy) in enumerate(visits):
            # Generate visual features at this position
            features = np.random.rand(32).astype(np.float32)
            features[0] = vx / 100
            features[1] = vy / 100

            # Try to match
            match = self.fly.brain.match_landmarks(features)

            # Draw visit
            ax.scatter(vx, vy, c='blue', s=100, marker='o', zorder=5)
            ax.annotate(f'{i+1}', (vx+3, vy+3), fontsize=8)

            if match:
                # Draw line to matched landmark
                lm_id, lmx, lmy = match
                ax.plot([vx, lmx], [vy, lmy], 'r--', linewidth=2, alpha=0.7)
                ax.annotate(f'→{lm_id}', ((vx+lmx)/2, (vy+lmy)/2), fontsize=9, color='red')

        # Draw fly trajectory
        traj_x = [v[0] for v in visits]
        traj_y = [v[1] for v in visits]
        ax.plot(traj_x, traj_y, 'b-', linewidth=2, alpha=0.5)

        ax.set_xlim(-80, 120)
        ax.set_ylim(-80, 130)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title('Landmark Recognition During Navigation', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig('bob_landmarks.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: bob_landmarks.png")
        plt.show()

    def run_all_demos(self):
        """Run all BoB feature demonstrations."""
        print("\n" + "="*60)
        print("  BoB FEATURES DEMONSTRATION")
        print("  (Brains on Board - Nature 2024)")
        print("="*60)
        print()
        print("1. Ring Attractor (Heading/Compass)")
        print("2. Path Integration (Dead Reckoning)")
        print("3. Landmark Navigation")
        print()

        self.run_ring_attractor_demo()
        self.run_path_integration_demo()
        self.run_landmark_demo()

        print("\n" + "="*60)
        print("  ALL DEMOS COMPLETE!")
        print("="*60)
        print("\nGenerated visualizations:")
        print("  - bob_ring_attractor.png")
        print("  - bob_path_integration.png")
        print("  - bob_landmarks.png")


def main():
    parser = argparse.ArgumentParser(description="BoB Features Visualization")
    parser.add_argument('--demo', choices=['all', 'ring_attractor', 'path_integration', 'landmarks'],
                       default='all', help='Demo to run')
    args = parser.parse_args()

    viz = BoBVisualization(demo_type=args.demo)

    if args.demo == 'all' or args.demo == 'ring_attractor':
        viz.run_ring_attractor_demo()

    if args.demo == 'all' or args.demo == 'path_integration':
        viz.run_path_integration_demo()

    if args.demo == 'all' or args.demo == 'landmarks':
        viz.run_landmark_demo()

    if args.demo != 'all':
        print(f"\n✓ Demo '{args.demo}' complete!")


if __name__ == '__main__':
    main()
