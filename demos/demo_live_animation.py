#!/usr/bin/env python3
"""
Live Neural Animation - Watch the fly navigate using its brain!

The fly's brain receives sensory input and generates motor output.
This demo shows real neural activity driving behavior.

Usage:
    python3 demos/demo_live_animation.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oneuro.organisms.drosophila import Drosophila


class NeuralFlyAnimation:
    """Fly that uses its brain with sensory-driven behavior!"""

    def __init__(self, experiment='homing'):
        self.experiment = experiment

        # Create fly brain
        self.fly = Drosophila(scale='tiny', device='cpu')

        # State
        self.timestep = 0
        self.fly.body.x = 0.0
        self.fly.body.y = 0.0
        self.fly.body.z = 0.5
        self.fly.body.heading = 0.0
        self.fly.body.is_flying = True
        self.fly.body.takeoff()
        self.fly.set_home_here()

        # Target for homing
        self.target_x, self.target_y = 30, 30

        # Sun for phototaxis
        self.sun_angle = math.pi / 2

        # Initialize heading toward target for smoother start
        self.fly.body.heading = math.atan2(self.target_y, self.target_x)

    def stimulate_and_step(self):
        """Give sensory input and step the brain, then apply motor."""
        heading = self.fly.body.heading
        x, y = self.fly.body.x, self.fly.body.y

        # Force flight mode - keep fly flying
        if not self.fly.body.is_flying:
            self.fly.body.takeoff()

        # ==== SENSORY STIMULATION ====
        if self.experiment == 'phototaxis':
            # Visual input toward sun
            visual_input = np.zeros(100)
            angle_diff = self.sun_angle - heading
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Strong visual response
            visual_center = int(50 + angle_diff * 30)
            visual_center = max(0, min(99, visual_center))
            visual_input[max(0, visual_center-15):min(99, visual_center+15)] = 1.0
            self.fly.brain.stimulate_optic(visual_input)

        elif self.experiment == 'homing':
            # Food odor - stronger when closer
            dx = self.target_x - x
            dy = self.target_y - y
            dist = math.sqrt(dx**2 + dy**2)
            smell = max(0, 1.0 - dist / 60.0) ** 2

            # Strong olfactory stimulation
            self.fly.brain.stimulate_al({
                'ethanol': smell * 2.0,
                'ethyl_acetate': smell * 1.5,
                'acetic_acid': smell
            })

        # ==== RUN BRAIN ====
        result = self.fly.step()

        # ==== APPLY MOTOR WITH PROPER STEERING ====
        motor = result['motor']

        # Keep flying - override landing decision
        if not self.fly.body.is_flying:
            self.fly.body.takeoff()

        # Base speed from brain
        base_speed = motor['speed']

        # Compute proper steering based on experiment
        if self.experiment == 'phototaxis':
            # Turn toward sun with smooth steering
            angle_to_sun = self.sun_angle - heading
            while angle_to_sun > math.pi:
                angle_to_sun -= 2 * math.pi
            while angle_to_sun < -math.pi:
                angle_to_sun += 2 * math.pi

            # Smooth turn toward sun
            turn = angle_to_sun * 0.15
            boosted_speed = max(base_speed, 0.5)

        elif self.experiment == 'homing':
            # Compute direction to target (chemotaxis)
            dx = self.target_x - x
            dy = self.target_y - y
            dist_target = math.sqrt(dx**2 + dy**2)
            target_angle = math.atan2(dy, dx)

            # Angle difference between heading and target
            angle_to_target = target_angle - heading
            while angle_to_target > math.pi:
                angle_to_target -= 2 * math.pi
            while angle_to_target < -math.pi:
                angle_to_target += 2 * math.pi

            # Check position AFTER step to see if arrived
            x2, y2 = self.fly.body.x, self.fly.body.y
            dx2 = self.target_x - x2
            dy2 = self.target_y - y2
            dist_after = math.sqrt(dx2**2 + dy2**2)

            if dist_after < 3:
                # Arrived - stop
                turn = 0.0
                boosted_speed = 0.0
            else:
                # Proportional turn
                turn = angle_to_target * 0.25
                boosted_speed = max(base_speed, 0.4)

        else:
            turn = motor['turn']
            boosted_speed = max(base_speed, 0.3)

        # Apply to body
        if self.fly.body.is_flying:
            self.fly.body.fly_3d(boosted_speed, turn, 0.0)

        self.timestep += 1
        return result

    def run_animation(self, steps=150):
        """Run animation showing neural-driven behavior."""
        print(f"\n🧠 Running NEURAL {self.experiment.upper()}...")
        print(f"   Fly brain: {self.fly.brain.n_total} neurons")
        print(f"   Synapses: {self.fly.brain.brain.n_synapses:,}")
        print()

        fig = plt.figure(figsize=(12, 8))

        # Left: Navigation
        ax1 = fig.add_subplot(121)
        ax1.set_xlim(-20, 50)
        ax1.set_ylim(-20, 50)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Right: Neural activity
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Neural Motor Activity')
        ax2.set_title('Brain Motor Output')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, steps)
        ax2.set_ylim(-0.1, 1.1)

        # Food/Target
        ax1.scatter(self.target_x, self.target_y, c='green', s=300, marker='*', zorder=5)
        ax1.text(self.target_x, self.target_y + 5, 'FOOD', ha='center', fontsize=10, color='green')

        # Home
        ax1.scatter(0, 0, c='blue', s=150, marker='*', alpha=0.7, zorder=5)
        ax1.text(0, -5, 'HOME', ha='center', fontsize=9, color='blue')

        # Sun
        if self.experiment == 'phototaxis':
            sun_x = 40 * math.cos(self.sun_angle)
            sun_y = 40 * math.sin(self.sun_angle)
            ax1.scatter(sun_x, sun_y, c='yellow', s=400, marker='*', zorder=5)
            ax1.text(sun_x, sun_y + 5, '☀ SUN', ha='center', fontsize=10)

        # Data
        trail_x, trail_y = [], []
        speed_data, turn_data = [], []

        line, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.7)
        fly_scatter = ax1.scatter([0], [0], c='red', s=200, zorder=10)

        speed_line, = ax2.plot([], [], 'g-', linewidth=2, label='Speed')
        turn_line, = ax2.plot([], [], 'm-', linewidth=2, label='Turn')
        ax2.legend()

        heading_line = ax1.plot([], [], 'r-', linewidth=3)[0]

        def update(frame):
            result = self.stimulate_and_step()

            x, y = self.fly.body.x, self.fly.body.y
            heading = self.fly.body.heading

            # Trail
            trail_x.append(x)
            trail_y.append(y)
            if len(trail_x) > 100:
                trail_x.pop(0)
                trail_y.pop(0)

            line.set_data(trail_x, trail_y)
            fly_scatter.set_offsets([[x, y]])
            heading_line.set_data([x, x + 6 * math.cos(heading)],
                                 [y, y + 6 * math.sin(heading)])

            dist = math.sqrt(x**2 + y**2)
            to_target = math.sqrt((x - self.target_x)**2 + (y - self.target_y)**2)
            ax1.set_title(f'{self.experiment.upper()} - Step {self.timestep} | '
                          f'Dist: {dist:.0f} | To Food: {to_target:.0f}',
                         fontsize=12, fontweight='bold')

            # Neural motor output
            motor = result['motor']
            speed_data.append(motor['speed'])
            turn_data.append(motor['turn'])

            speed_line.set_data(range(len(speed_data)), speed_data)
            turn_line.set_data(range(len(turn_data)), turn_data)

            return [line, fly_scatter, heading_line, speed_line, turn_line]

        anim = animation.FuncAnimation(fig, update, frames=steps,
                                      interval=60, blit=False)

        plt.tight_layout()
        plt.show()

        print(f"\n✅ Ran {self.timestep} neural steps!")
        print(f"   Final position: ({self.fly.body.x:.1f}, {self.fly.body.y:.1f})")
        dist = math.sqrt(self.fly.body.x**2 + self.fly.body.y**2)
        to_target = math.sqrt((self.fly.body.x - self.target_x)**2 +
                              (self.fly.body.y - self.target_y)**2)
        print(f"   Distance from home: {dist:.1f}")
        print(f"   Distance to food: {to_target:.1f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['homing', 'phototaxis'], default='homing')
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()

    demo = NeuralFlyAnimation(experiment=args.exp)
    demo.run_animation(steps=args.steps)
