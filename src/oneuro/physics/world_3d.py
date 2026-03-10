"""
3D Minecraft-Style World Visualization

Creates a voxel-style 3D world with:
- Day/night cycle with sun
- Terrain with grass, dirt, sky
- Fly navigating through the world
- Beautiful rendering for video demos

Usage:
    python3 src/oneuro/physics/world_3d.py --demo
    python3 src/oneuro/physics/world_3d.py --record --output world.mp4
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource
import matplotlib.patches as mpatches


class VoxelWorld:
    """
    Minecraft-style voxel world.

    Features:
    - 3D voxel terrain
    - Day/night lighting
    - Sun position
    - Fly agent navigation
    """

    # Block types
    AIR = 0
    GRASS = 1
    DIRT = 2
    STONE = 3
    LEAVES = 4
    FRUIT = 5

    COLORS = {
        AIR: 'white',
        GRASS: '#4CAF50',
        DIRT: '#8B4513',
        STONE: '#808080',
        LEAVES: '#228B22',
        FRUIT: '#FF6B6B',
    }

    def __init__(
        self,
        size: tuple = (20, 20, 15),
        seed: int = 42,
    ):
        self.size = size
        np.random.seed(seed)

        # Create terrain
        self.voxels = self._generate_terrain()

        # Sky gradient
        self.sky_color = '#87CEEB'  # Day sky
        self.sun_position = (10, 10, 15)

    def _generate_terrain(self):
        """Generate voxel terrain."""
        w, h, d = self.size
        voxels = np.zeros((w, h, d), dtype=int)

        # Simple terrain: grass on top, dirt below, stone at bottom
        for x in range(w):
            for y in range(h):
                # Height map with some noise
                base_height = h // 3 + int(np.sin(x / 3) * 2 + np.cos(y / 4) * 2)
                height = max(1, min(d - 3, base_height + np.random.randint(-1, 2)))

                for z in range(d):
                    if z < height - 2:
                        voxels[x, y, z] = self.STONE
                    elif z < height:
                        voxels[x, y, z] = self.DIRT
                    else:
                        voxels[x, y, z] = self.GRASS

                # Add some trees
                if np.random.random() < 0.05 and height < h - 5:
                    self._add_tree(voxels, x, y, height)

                # Add fruits
                if np.random.random() < 0.1:
                    voxels[x, y, height] = self.FRUIT

        return voxels

    def _add_tree(self, voxels, x, y, ground_z):
        """Add a simple tree."""
        h = self.size[2]
        if ground_z + 4 < h:
            # Trunk
            for z in range(ground_z, ground_z + 3):
                if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
                    voxels[x, y, z] = self.DIRT

            # Leaves
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [3, 4]:
                        nx, ny = x + dx, y + dy
                        nz = ground_z + dz
                        if (0 <= nx < self.size[0] and
                            0 <= ny < self.size[1] and
                            nz < h and
                            voxels[nx, ny, nz] == self.AIR):
                            voxels[nx, ny, nz] = self.LEAVES


class Fly3D:
    """
    3D Fly with physics-based movement.
    """

    def __init__(self, world: VoxelWorld, start_pos: tuple = None):
        self.world = world
        if start_pos is None:
            # Start above terrain
            start_pos = (
                world.size[0] // 2,
                world.size[1] // 2,
                world.size[2] - 2,
            )
        self.position = np.array(start_pos, dtype=float)
        self.velocity = np.zeros(3)
        self.orientation = 0  # yaw angle

    def update(self, dt: float = 0.1):
        """Update fly physics."""
        # Simple flight physics
        gravity = -9.8

        # Apply gravity
        self.velocity[2] += gravity * dt

        # Random perturbation (bug-like flight)
        self.velocity[:2] += np.random.uniform(-0.5, 0.5, 2)

        # Lift when moving forward (simplified aerodynamics)
        speed = np.linalg.norm(self.velocity[:2])
        if speed > 0.1:
            self.velocity[2] += speed * 0.5

        # Update position
        self.position += self.velocity * dt

        # Boundary constraints
        for i in range(3):
            self.position[i] = np.clip(
                self.position[i],
                0,
                self.world.size[i] - 1
            )

        # Ground collision
        if self.position[2] < 1:
            self.position[2] = 1
            self.velocity[2] = abs(self.velocity[2]) * 0.5

        # Update orientation
        if np.linalg.norm(self.velocity[:2]) > 0.01:
            self.orientation = np.arctan2(
                self.velocity[1],
                self.velocity[0]
            )


class World3DVisualizer:
    """
    Beautiful 3D world visualization with fly.
    """

    # References to VoxelWorld constants
    AIR = VoxelWorld.AIR
    GRASS = VoxelWorld.GRASS
    DIRT = VoxelWorld.DIRT
    STONE = VoxelWorld.STONE
    LEAVES = VoxelWorld.LEAVES
    FRUIT = VoxelWorld.FRUIT
    COLORS = VoxelWorld.COLORS

    def __init__(
        self,
        world_size=(20, 20, 15),
        num_flies=1,
        show_voxel=True,
    ):
        self.world = VoxelWorld(size=world_size)
        self.flies = [Fly3D(self.world) for _ in range(num_flies)]
        self.show_voxel = show_voxel
        self.frame = 0

    def _create_figure(self):
        """Create figure."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((1, 1, 0.75))

        return fig, ax

    def _draw_world(self, ax):
        """Draw voxel world."""
        if not self.show_voxel:
            # Just draw ground plane
            w, h, d = self.world.size
            xx, yy = np.meshgrid(range(w), range(h))
            zz = np.ones_like(xx) * 0.5
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='green')
            return

        w, h, d = self.world.size

        # Draw each voxel type
        for block_type in [self.GRASS, self.DIRT, self.STONE, self.LEAVES, self.FRUIT]:
            voxels = np.where(self.world.voxels == block_type)

            if len(voxels[0]) > 0:
                color = self.COLORS[block_type]
                alpha = 0.6 if block_type != self.FRUIT else 1.0

                # Subsample for performance
                indices = np.random.choice(
                    len(voxels[0]),
                    min(len(voxels[0]), 500),
                    replace=False
                )

                ax.scatter(
                    voxels[0][indices],
                    voxels[1][indices],
                    voxels[2][indices],
                    c=color,
                    s=50 if block_type == self.FRUIT else 20,
                    alpha=alpha,
                    marker='s' if block_type != self.FRUIT else 'o',
                )

    def _draw_fly(self, ax, fly: Fly3D, color='red'):
        """Draw fly as a stylized marker."""
        pos = fly.position

        # Body (larger, more visible)
        ax.scatter(
            [pos[0]], [pos[1]], [pos[2]],
            c=color, s=200, marker='o',
            edgecolors='white', linewidth=2,
            alpha=0.9,
        )

        # Wings (simplified)
        wing_offset = 0.3
        wing_angle = fly.orientation

        # Left wing
        ax.scatter(
            [pos[0] - wing_offset * np.cos(wing_angle)],
            [pos[1] - wing_offset * np.sin(wing_angle)],
            [pos[2]],
            c='lightblue', s=50, alpha=0.7,
        )

        # Right wing
        ax.scatter(
            [pos[0] + wing_offset * np.cos(wing_angle)],
            [pos[1] + wing_offset * np.sin(wing_angle)],
            [pos[2]],
            c='lightblue', s=50, alpha=0.7,
        )

    def _draw_sun(self, ax):
        """Draw sun in the sky."""
        sun_pos = self.world.sun_position
        ax.scatter(
            [sun_pos[0]], [sun_pos[1]], [sun_pos[2]],
            c='yellow', s=500, marker='o',
            edgecolors='orange', linewidth=2,
        )

        # Sun rays
        for angle in np.linspace(0, 2*np.pi, 8):
            ray_end = (
                sun_pos[0] + 2 * np.cos(angle),
                sun_pos[1] + 2 * np.sin(angle),
                sun_pos[2],
            )
            ax.plot(
                [sun_pos[0], ray_end[0]],
                [sun_pos[1], ray_end[1]],
                [sun_pos[2], ray_end[2]],
                c='yellow', alpha=0.3,
            )

    def _draw_atmosphere(self, ax):
        """Draw sky gradient effect."""
        w, h, d = self.world.size

        # Top face - sky
        xx, yy = np.meshgrid(range(w), range(h))
        zz = np.full_like(xx, d - 0.5)

        # Simple gradient
        colors = np.zeros((h, w, 4))
        colors[:, :, 0] = 0.53  # R
        colors[:, :, 1] = 0.81  # G
        colors[:, :, 2] = 0.92  # B
        colors[:, :, 3] = 0.1  # Alpha

        ax.plot_surface(xx, yy, zz, facecolors=colors, alpha=0.1)

    def _update_frame(self, frame):
        """Update one frame."""
        ax = self._axes
        ax.clear()

        # Set view
        w, h, d = self.world.size
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_zlim(0, d)

        # Draw world
        self._draw_world(ax)
        self._draw_atmosphere(ax)
        self._draw_sun(ax)

        # Update and draw flies
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.flies)))
        for i, fly in enumerate(self.flies):
            fly.update(dt=0.1)
            self._draw_fly(ax, fly, color=colors[i])

        # Style
        ax.set_xlabel('X (blocks)', fontsize=10)
        ax.set_ylabel('Y (blocks)', fontsize=10)
        ax.set_zlabel('Z (blocks)', fontsize=10)

        # Title with info
        fly = self.flies[0]
        ax.set_title(
            f'Minecraft-Style Fly World | '
            f'Pos: ({fly.position[0]:.1f}, {fly.position[1]:.1f}, {fly.position[2]:.1f}) | '
            f'Frame: {frame}',
            fontsize=12
        )

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.2)

        # Set view angle
        ax.view_init(elev=35, azim=45 + frame * 0.5)

        self.frame = frame
        return []

    def animate(self, frames=300, interval=50, record=False, output='world.mp4'):
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
            try:
                writer = FFMpegWriter(fps=20)
                anim.save(output, writer=writer, dpi=120)
                print(f"Saved: {output}")
            except Exception as e:
                print(f"Could not save video: {e}")
                print("Showing animation instead...")
                plt.show()
        else:
            plt.show()

        return anim

    def snapshot(self, title="Minecraft-Style Fly World"):
        """Take a snapshot."""
        fig, ax = self._create_figure()

        # Run for a bit
        for _ in range(50):
            for fly in self.flies:
                fly.update(dt=0.1)

        # Draw
        self._draw_world(ax)
        self._draw_atmosphere(ax)
        self._draw_sun(ax)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.flies)))
        for i, fly in enumerate(self.flies):
            self._draw_fly(ax, fly, color=colors[i])

        w, h, d = self.world.size
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_zlim(0, d)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.view_init(elev=30, azim=45)

        return fig


def create_demo_figure():
    """Create a nice multi-panel figure."""
    print("Creating world demo figure...")

    fig = plt.figure(figsize=(16, 12))

    # Panel 1: World snapshot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    world = World3DVisualizer(world_size=(15, 15, 10), num_flies=3)
    world._axes = ax1
    world._draw_world(ax1)
    world._draw_atmosphere(ax1)
    world._draw_sun(ax1)
    for i, fly in enumerate(world.flies):
        for _ in range(30):
            fly.update(dt=0.1)
        colors = plt.cm.rainbow(np.linspace(0, 1, 3))
        world._draw_fly(ax1, fly, color=colors[i])

    ax1.set_xlim(0, 15)
    ax1.set_ylim(0, 15)
    ax1.set_zlim(0, 10)
    ax1.set_title('A) 3D Voxel World with Flies', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    ax1.axis('off')

    # Panel 2: Fly trajectory
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    world2 = World3DVisualizer(world_size=(15, 15, 10), num_flies=1)

    trajectory = []
    for _ in range(100):
        world2.flies[0].update(dt=0.1)
        trajectory.append(world2.flies[0].position.copy())

    trajectory = np.array(trajectory)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
             'b-', linewidth=2, alpha=0.7)
    ax2.scatter([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]],
               c='green', s=100, marker='o', label='Start')
    ax2.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]],
               c='red', s=100, marker='s', label='End')

    ax2.set_xlim(0, 15)
    ax2.set_ylim(0, 15)
    ax2.set_zlim(0, 10)
    ax2.set_title('B) Fly Trajectory (100 steps)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.view_init(elev=25, azim=45)

    # Panel 3: Multi-fly paths
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    world3 = World3DVisualizer(world_size=(15, 15, 10), num_flies=5)

    for fly_idx, fly in enumerate(world3.flies):
        traj = []
        for _ in range(80):
            fly.update(dt=0.1)
            traj.append(fly.position.copy())
        traj = np.array(traj)
        color = plt.cm.rainbow(fly_idx / 5)
        ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2], c=color, linewidth=1.5, alpha=0.8)

    ax3.set_xlim(0, 15)
    ax3.set_ylim(0, 15)
    ax3.set_zlim(0, 10)
    ax3.set_title('C) Multi-Fly Navigation Paths', fontsize=12, fontweight='bold')
    ax3.view_init(elev=30, azim=45)

    # Panel 4: Statistics
    ax4 = fig.add_subplot(2, 2, 4)

    # Height over time
    world4 = World3DVisualizer(world_size=(15, 15, 10), num_flies=1)
    heights = []
    distances = []

    start_pos = world4.flies[0].position.copy()
    for _ in range(200):
        world4.flies[0].update(dt=0.1)
        heights.append(world4.flies[0].position[2])
        distances.append(np.linalg.norm(
            world4.flies[0].position[:2] - start_pos[:2]
        ))

    ax4.plot(heights, 'b-', label='Height', linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(distances, 'r-', label='Distance', linewidth=2)

    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Height (blocks)', color='blue')
    ax4_twin.set_ylabel('Distance (blocks)', color='red')
    ax4.set_title('D) Flight Statistics', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('world_demo_figure.png', dpi=200, bbox_inches='tight')
    print("Saved: world_demo_figure.png")

    return fig


def main():
    parser = argparse.ArgumentParser(description='3D Minecraft-style world')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--record', action='store_true', help='Record video')
    parser.add_argument('--output', default='world.mp4', help='Output file')
    parser.add_argument('--frames', type=int, default=300, help='Frames')
    parser.add_argument('--figure', action='store_true', help='Create figure')

    args = parser.parse_args()

    if args.figure:
        create_demo_figure()
        return

    vis = World3DVisualizer(
        world_size=(20, 20, 15),
        num_flies=3,
    )
    vis.animate(frames=args.frames, record=args.record, output=args.output)


if __name__ == "__main__":
    main()
