#!/usr/bin/env python3
"""
Live 3D Animation Demo - Watch the fly navigate in a 3D world!

Usage:
    python3 demos/demo_3d_flight.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oneuro.organisms.drosophila import Drosophila


class Live3DFlight:
    """Real-time 3D animated fly flight."""

    def __init__(self):
        self.fly = Drosophila(scale='tiny')

        # State
        self.positions_3d = deque(maxlen=500)
        self.timestep = 0

        # Initial
        self.fly.body.x = 0.0
        self.fly.body.y = 0.0
        self.fly.body.z = 10.0
        self.fly.body.heading = 0.0
        self.fly.body.pitch = 0.0
        self.fly.body.is_flying = True

        # Home
        self.home = (0, 0, 10)

        # Targets
        self.fly.set_home_here()

    def update(self, frame):
        """Update one frame."""
        dt = 0.02

        # Simple flight behavior
        if self.timestep % 100 < 50:
            # Circle pattern
            turn = 0.5
            speed = 0.6
            climb = 0.0
        else:
            # Figure 8
            turn = -0.5
            speed = 0.5
            climb = 0.2 if self.timestep % 100 < 75 else -0.2

        # Update heading
        self.fly.body.heading += turn * 0.05

        # Move
        move_speed = speed * 0.3
        self.fly.body.x += move_speed * math.cos(self.fly.body.heading)
        self.fly.body.y += move_speed * math.sin(self.fly.body.heading)
        self.fly.body.z += climb * 0.1
        self.fly.body.z = max(1, min(20, self.fly.body.z))  # Clamp altitude

        # Record
        self.positions_3d.append((self.fly.body.x, self.fly.body.y, self.fly.body.z))
        self.timestep += 1

        return self.plot()

    def plot(self):
        """Update 3D plot."""
        ax = self.ax
        ax.clear()

        # Trail
        if len(self.positions_3d) > 1:
            xs = [p[0] for p in self.positions_3d]
            ys = [p[1] for p in self.positions_3d]
            zs = [p[2] for p in self.positions_3d]
            ax.plot(xs, ys, zs, 'b-', alpha=0.6, linewidth=2)

        # Current position
        x, y, z = self.fly.body.x, self.fly.body.y, self.fly.body.z

        # Draw fly
        ax.scatter([x], [y], [z], c='red', s=200, marker='o')

        # Heading vector
        hx = x + 3 * math.cos(self.fly.body.heading)
        hy = y + 3 * math.sin(self.fly.body.heading)
        hz = z + math.sin(self.fly.body.pitch) * 2
        ax.plot([x, hx], [y, hy], [z, hz], 'r-', linewidth=3)

        # Home point
        ax.scatter([self.home[0]], [self.home[1]], [self.home[2]],
                  c='green', s=100, marker='*')

        # Ground
        xx, yy = np.meshgrid(np.linspace(-50, 50, 10), np.linspace(-50, 50, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='brown')

        # Settings
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(0, 25)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title(f'3D Flight - Step {self.timestep}', fontsize=14, fontweight='bold')

        return []


def main():
    print("\n🎬 Starting LIVE 3D FLIGHT animation...")
    print("Watch the fly fly!\n")

    demo = Live3DFlight()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    demo.ax = ax

    anim = animation.FuncAnimation(
        fig, demo.update, frames=200, interval=50, blit=False
    )

    plt.show()
    print(f"\n✅ 3D flight complete! {demo.timestep} steps")


if __name__ == '__main__':
    main()
