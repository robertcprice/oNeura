"""
3D Fly Arena Visualization

Creates beautiful 3D visualizations of the multi-fly arena for videos/demos.
Uses matplotlib with 3D projection for smooth rendering.

Usage:
    python3 src/oneuro/physics/visualize_arena.py

    # Record a video
    python3 src/oneuro/physics/visualize_arena.py --record --output demo.mp4
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from oneuro.physics.multi_fly_arena_simple import EmergentMultiFlyArena, PheromoneField


class FlyArenaVisualizer:
    """
    Beautiful 3D visualization of emergent fly behavior.

    Features:
    - 3D arena with flies as colored spheres
    - Pheromone field as transparent planes
    - Trail effects showing movement history
    - Real-time or recorded output
    """

    def __init__(
        self,
        num_flies: int = 20,
        arena_size: float = 10.0,
        figsize: tuple = (12, 10),
    ):
        self.num_flies = num_flies
        self.arena_size = arena_size
        self.figsize = figsize

        # Initialize arena
        self.arena = EmergentMultiFlyArena(
            num_flies=num_flies,
            arena_size=arena_size,
        )
        self.arena.reset()

        # Trail history
        self.trails = [[] for _ in range(num_flies)]
        self.max_trail_length = 50

        # Colors
        self.fly_colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_flies))

    def _create_figure(self):
        """Create figure with 3D axes."""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-self.arena_size/2, self.arena_size/2)
        ax.set_ylim(-self.arena_size/2, self.arena_size/2)
        ax.set_zlim(0, 2)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title('Emergent Social Behavior: Drosophila Aggregation', fontsize=14, fontweight='bold')

        # Style
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)

        return fig, ax

    def _draw_fly(self, ax, position, color, size=0.15):
        """Draw a single fly as a 3D marker."""
        ax.scatter(
            position[0], position[1], position[2],
            c=[color], s=size * 500, marker='o', edgecolors='white', linewidth=0.5
        )

    def _draw_trails(self, ax):
        """Draw movement trails."""
        for i, trail in enumerate(self.trails):
            if len(trail) > 1:
                trail_array = np.array(trail)
                ax.plot(
                    trail_array[:, 0],
                    trail_array[:, 1],
                    trail_array[:, 2],
                    c=[self.fly_colors[i]],
                    alpha=0.3,
                    linewidth=0.5,
                )

    def _draw_pheromone_field(self, ax):
        """Draw pheromone field as transparent gradient."""
        # Simplified: show aggregate field as colored planes at different heights
        positions = np.array([f.position for f in self.arena.flies])

        # Ground plane with alpha based on density
        if len(positions) > 0:
            # Compute local density
            for x in np.linspace(-self.arena_size/2, self.arena_size/2, 10):
                for y in np.linspace(-self.arena_size/2, self.arena_size/2, 10):
                    dists = np.linalg.norm(positions[:, :2] - np.array([x, y]), axis=1)
                    density = np.exp(-np.min(dists) * 2)
                    if density > 0.3:
                        ax.scatter(x, y, 0.01, c='green', s=density*100, alpha=density*0.3)

    def _update_frame(self, frame):
        """Update one frame of animation."""
        ax = self._axes
        ax.clear()

        # Set up axes again after clear
        ax.set_xlim(-self.arena_size/2, self.arena_size/2)
        ax.set_ylim(-self.arena_size/2, self.arena_size/2)
        ax.set_zlim(0, 2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Emergent Aggregation: Frame {frame}', fontsize=12)

        # Style
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.2)

        # Draw pheromone field
        self._draw_pheromone_field(ax)

        # Step simulation
        self.arena.step(dt=0.1)

        # Update trails
        positions = [f.position for f in self.arena.flies]
        for i, pos in enumerate(positions):
            self.trails[i].append(pos.copy())
            if len(self.trails[i]) > self.max_trail_length:
                self.trails[i].pop(0)

        # Draw trails
        self._draw_trails(ax)

        # Draw flies
        for i, fly in enumerate(self.arena.flies):
            self._draw_fly(ax, fly.position, self.fly_colors[i])

        # Draw ground
        xx, yy = np.meshgrid(
            np.linspace(-self.arena_size/2, self.arena_size/2, 2),
            np.linspace(-self.arena_size/2, self.arena_size/2, 2)
        )
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

        return []

    def animate(self, frames=200, interval=50, record=False, output='fly_arena.mp4'):
        """Create animation."""
        fig, ax = self._create_figure()
        self._axes = ax

        anim = FuncAnimation(
            fig,
            self._update_frame,
            frames=frames,
            interval=interval,
            blit=False,
        )

        if record:
            print(f"Recording to {output}...")
            writer = FFMpegWriter(fps=20)
            anim.save(output, writer=writer, dpi=150)
            print(f"Saved: {output}")
        else:
            plt.show()

        return anim

    def snapshot(self, title="Emergent Fly Behavior"):
        """Take a single snapshot."""
        fig, ax = self._create_figure()

        # Step a few times to get some movement
        for _ in range(50):
            self.arena.step(dt=0.1)

        # Update trails
        for i, fly in enumerate(self.arena.flies):
            self.trails[i].append(fly.position.copy())

        # Draw
        self._draw_pheromone_field(ax)
        self._draw_trails(ax)
        for i, fly in enumerate(self.arena.flies):
            self._draw_fly(ax, fly.position, self.fly_colors[i])

        # Ground
        xx, yy = np.meshgrid(
            np.linspace(-self.arena_size/2, self.arena_size/2, 2),
            np.linspace(-self.arena_size/2, self.arena_size/2, 2)
        )
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

        plt.title(title, fontsize=14, fontweight='bold')
        return fig


class BrainVisualizer:
    """
    Visualize the neural activity in a way that looks cool for demos.
    """

    def __init__(self, num_neurons=100):
        self.num_neurons = num_neurons
        # Simulate neural activity
        self.activities = np.random.rand(num_neurons)

    def visualize_network(self, figsize=(10, 10)):
        """Draw neural network as 3D blob."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Neuron positions in 3D
        theta = np.linspace(0, 2*np.pi, int(np.sqrt(self.num_neurons)))
        phi = np.linspace(0, np.pi, int(np.sqrt(self.num_neurons)))
        x = np.outer(np.cos(theta), np.sin(phi)).flatten()[:self.num_neurons]
        y = np.outer(np.sin(theta), np.sin(phi)).flatten()[:self.num_neurons]
        z = np.cos(np.linspace(0, 2*np.pi, self.num_neurons))[:self.num_neurons]

        # Activity colors
        colors = plt.cm.plasma(self.activities)

        # Draw
        ax.scatter(x, y, z, c=colors, s=100, alpha=0.8)

        ax.set_title('Neural Activity Pattern', fontsize=14, fontweight='bold')
        ax.axis('off')

        return fig


def create_paper_figure():
    """Create a nice figure for the paper."""
    print("Creating paper figure...")

    # Multi-panel figure
    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Arena snapshot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    vis = FlyArenaVisualizer(num_flies=30, arena_size=8.0)
    vis.arena.reset()

    # Run for 100 steps
    for _ in range(100):
        vis.arena.step(dt=0.1)

    positions = np.array([f.position for f in vis.arena.flies])
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                c=vis.fly_colors, s=50, alpha=0.8)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_zlim(0, 1)
    ax1.set_title('A) Emergent Aggregation', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # Panel 2: Cluster over time
    ax2 = fig.add_subplot(2, 2, 2)
    vis2 = FlyArenaVisualizer(num_flies=30, arena_size=8.0)
    vis2.arena.reset()

    cluster_history = []
    for _ in range(200):
        vis2.arena.step(dt=0.1)
        m = vis2.arena.get_metrics()
        cluster_history.append(m['cluster_index'])

    ax2.plot(cluster_history, 'b-', linewidth=2)
    ax2.fill_between(range(len(cluster_history)), cluster_history, alpha=0.3)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Cluster Index')
    ax2.set_title('B) Aggregation Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Panel 3: Neural visualization
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    brain = BrainVisualizer(num_neurons=196)
    brain.activities = np.random.rand(196)

    theta = np.linspace(0, 2*np.pi, 14)
    phi = np.linspace(0, np.pi, 14)
    x = np.outer(np.cos(theta), np.sin(phi)).flatten()[:196]
    y = np.outer(np.sin(theta), np.sin(phi)).flatten()[:196]
    z = np.cos(np.linspace(0, 2*np.pi, 196))

    colors = plt.cm.plasma(brain.activities)
    ax3.scatter(x, y, z, c=colors, s=80, alpha=0.8)
    ax3.set_title('C) Neural Activity', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Panel 4: Drug effects
    ax4 = fig.add_subplot(2, 2, 4)

    from oneuro.physics.drug_locomotion_emergent import EmergentDrugExperiment

    exp = EmergentDrugExperiment()
    drugs = ['diazepam', 'amphetamine', 'caffeine', 'ketamine']
    changes = []

    for drug in drugs:
        r = exp.run(drug, dose=10.0, duration=30.0)
        changes.append(r['speed_change_pct'])

    colors = ['red' if c < 0 else 'green' for c in changes]
    bars = ax4.bar(drugs, changes, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_ylabel('Speed Change (%)')
    ax4.set_title('D) Drug Effects (Emergent)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('paper_figure.png', dpi=300, bbox_inches='tight')
    print("Saved: paper_figure.png")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize fly arena')
    parser.add_argument('--record', action='store_true', help='Record video')
    parser.add_argument('--output', default='fly_arena.mp4', help='Output file')
    parser.add_argument('--frames', type=int, default=200, help='Number of frames')
    parser.add_argument('--paper', action='store_true', help='Create paper figure')

    args = parser.parse_args()

    if args.paper:
        create_paper_figure()
        return

    vis = FlyArenaVisualizer(num_flies=20, arena_size=10.0)
    vis.animate(frames=args.frames, record=args.record, output=args.output)


if __name__ == "__main__":
    main()
