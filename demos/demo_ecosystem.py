#!/usr/bin/env python3
"""
oNeura Ecosystem Demo - Neural Fly Terrarium

Run a terrarium simulation where flies use their actual oNeura brains
to navigate, find food, mate, and survive.

Usage:
    python3 demos/demo_ecosystem.py
    python3 demos/demo_ecosystem.py --flies 50
    python3 demos/demo_ecosystem.py --render  # If matplotlib available
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from oneuro.organisms.drosophila import Drosophila
from oneuro.physics.ecosystem import Terrarium, FoodSource, LifeStage


class NeuralEcosystem:
    """Ecosystem with actual oNeura Drosophila brains."""

    def __init__(self, n_flies=20):
        self.terrarium = Terrarium()

        # Add food
        self.terrarium.add_food(10, 0, 'fruit', 0.8)
        self.terrarium.add_food(-10, 10, 'dish', 1.0)
        self.terrarium.add_food(-10, -10, 'fruit', 0.8)

        # Add flies with actual Drosophila brains
        for i in range(n_flies):
            brain = Drosophila(scale='tiny', device='cpu')
            self.terrarium.add_fly(brain=brain)

    def step(self):
        """Step simulation."""
        self.terrarium.step(dt=0.1)

    def get_fly_data(self):
        """Get data for visualization."""
        flies = [o for o in self.terrarium.organisms if o.stage == LifeStage.ADULT]
        return flies


def run_visualization(n_flies=20, steps=500):
    """Run animated visualization."""
    print(f"🧠 Creating ecosystem with {n_flies} neural flies...")

    eco = NeuralEcosystem(n_flies)

    fig = plt.figure(figsize=(14, 8))

    # Top: Terrarium view
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)
    ax1.set_aspect('equal')
    ax1.set_title('Terrarium View', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Food sources
    food_x = [f.x for f in eco.terrarium.food_sources]
    food_z = [f.z for f in eco.terrarium.food_sources]
    ax1.scatter(food_x, food_z, c='orange', s=200, marker='s', zorder=5, label='Food')

    # Bottom: Neural activity
    ax2 = fig.add_subplot(222)
    ax2.set_title('Neural Activity (First Fly)', fontsize=10)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Activity')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(224)
    ax3.set_title('Population', fontsize=10)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Count')
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, n_flies * 1.5)
    ax3.grid(True, alpha=0.3)

    # Data storage
    al_data, mb_data, vnc_data = [], [], []
    adult_data, larva_data = [], []

    # Scatter plots for flies
    male_scatter = ax1.scatter([], [], c='red', s=80, alpha=0.8, label='Male', zorder=10)
    female_scatter = ax1.scatter([], [], c='magenta', s=80, alpha=0.8, label='Female', zorder=10)

    # Lines for neural activity
    al_line, = ax2.plot([], [], 'g-', linewidth=2, label='AL')
    mb_line, = ax2.plot([], [], 'b-', linewidth=2, label='MB')
    vnc_line, = ax2.plot([], [], 'r-', linewidth=2, label='VNC')

    # Lines for population
    adult_line, = ax3.plot([], [], 'b-', linewidth=2, label='Adults')
    larva_line, = ax3.plot([], [], 'y-', linewidth=2, label='Larvae')

    ax2.legend(loc='upper right', fontsize=8)
    ax3.legend(loc='upper right', fontsize=8)
    ax1.legend(loc='upper right', fontsize=8)

    def update(frame):
        # Step simulation
        eco.step()

        # Get fly data
        flies = eco.get_fly_data()

        # Update fly positions
        male_x = [f.x for f in flies if f.gender == 'male']
        male_z = [f.z for f in flies if f.gender == 'male']
        female_x = [f.x for f in flies if f.gender == 'female']
        female_z = [f.z for f in flies if f.gender == 'female']

        male_scatter.set_offsets(np.column_stack([male_x, male_z]))
        female_scatter.set_offsets(np.column_stack([female_x, female_z]))

        # Get neural activity from first fly
        if flies:
            first_fly = flies[0]
            if first_fly.neural_activity:
                al_data.append(first_fly.neural_activity.al)
                mb_data.append(first_fly.neural_activity.mb)
                vnc_data.append(first_fly.neural_activity.vnc)

                # Keep only last 100 points
                if len(al_data) > 100:
                    al_data.pop(0)
                    mb_data.pop(0)
                    vnc_data.pop(0)

                al_line.set_data(range(len(al_data)), al_data)
                mb_line.set_data(range(len(mb_data)), mb_data)
                vnc_line.set_data(range(len(vnc_data)), vnc_data)

        # Population
        stats = eco.terrarium.get_stats()
        adult_data.append(stats['adults'])
        larva_data.append(stats['larvae'] + stats['pupae'] + stats['eggs'])

        if len(adult_data) > 100:
            adult_data.pop(0)
            larva_data.pop(0)

        adult_line.set_data(range(len(adult_data)), adult_data)
        larva_line.set_data(range(len(larva_data)), larva_data)

        # Title with stats
        ax1.set_title(f'Terrarium - Day {stats["day"]} @ {int(stats["time_of_day"]):02d}:00 | Flies: {stats["adults"]}',
                     fontsize=12, fontweight='bold')

        return male_scatter, female_scatter, al_line, mb_line, vnc_line, adult_line, larva_line

    print("🎬 Running animation...")
    anim = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

    # Final stats
    stats = eco.terrarium.get_stats()
    print(f"\n✅ Simulation complete!")
    print(f"   Final population: {stats['adults']} adults, {stats['larvae']} larvae")
    print(f"   Days elapsed: {stats['day']}")


def run_headless(n_flies=20, steps=200):
    """Run without visualization."""
    print(f"🧠 Creating ecosystem with {n_flies} neural flies...")

    eco = NeuralEcosystem(n_flies)

    print("⏳ Running simulation...")
    for i in range(steps):
        eco.step()

        if (i + 1) % 50 == 0:
            stats = eco.terrarium.get_stats()
            print(f"   Step {i+1}: {stats['adults']} adults, {stats['larvae']} larvae, "
                  f"food: {stats['food_remaining']:.1f}")

    stats = eco.terrarium.get_stats()
    print(f"\n✅ Simulation complete!")
    print(f"   Final population: {stats['adults']} adults")
    print(f"   Larvae: {stats['larvae']}")
    print(f"   Food remaining: {stats['food_remaining']:.1f}")
    print(f"   Days elapsed: {stats['day']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='oNeura Ecosystem Demo')
    parser.add_argument('--flies', type=int, default=20, help='Number of flies')
    parser.add_argument('--steps', type=int, default=200, help='Simulation steps')
    parser.add_argument('--render', action='store_true', help='Show visualization')
    args = parser.parse_args()

    if args.render:
        run_visualization(args.flies, args.steps)
    else:
        run_headless(args.flies, args.steps)
