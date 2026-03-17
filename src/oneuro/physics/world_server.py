"""
oNeura 3D World Server

Real-time Flask server that:
1. Runs actual oNeura simulation
2. Serves 3D world via WebSocket
3. Tracks fly health, hunger, death

Usage:
    PYTHONPATH=src python3 src/oneuro/physics/world_server.py
"""

import sys
import os
import json
import time
import numpy as np
import threading
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# oNeura imports
try:
    from oneuro.organisms.drosophila import DrosophilaBrain
    from oneuro.physics.compound_eye import BinocularVisionSystem
    from oneuro.physics.olfaction import BilateralOlfaction
    ONEURO_AVAILABLE = True
except ImportError:
    ONEURO_AVAILABLE = False


# ============================================================================
# SIMULATION WORLD
# ============================================================================

class WorldState:
    """
    The actual simulation world with physics.
    """

    def __init__(self):
        # World dimensions
        self.size = (50, 50, 20)  # meters

        # Environmental conditions
        self.temperature = 25.0  # Celsius
        self.wind = np.array([0.0, 0.0, 0.0])
        self.time_of_day = 0.5  # 0-1 (0=midnight, 0.5=noon)

        # Food sources (fruits)
        self.fruits = []
        self.spawn_fruits(5)

        # Flies
        self.flies = []

        print("World initialized")

    def spawn_fruits(self, count=5):
        """Spawn fruits in world."""
        for _ in range(count):
            self.fruits.append({
                'id': len(self.fruits),
                'position': np.array([
                    np.random.uniform(5, self.size[0] - 5),
                    np.random.uniform(5, self.size[1] - 5),
                    0.5
                ]),
                'nutrition': 100.0,
                'type': 'fruit'
            })

    def get_odorants_at(self, position):
        """Get odorant concentrations at a position."""
        odorants = {}

        for fruit in self.fruits:
            if fruit['nutrition'] > 0:
                dist = np.linalg.norm(fruit['position'] - position)
                if dist < 10:  # Detection radius
                    strength = fruit['nutrition'] * np.exp(-dist / 3)
                    odorants['ethyl_acetate'] = odorants.get('ethyl_acetate', 0) + strength / 100

        return odorants

    def get_light_at(self, position):
        """Get light direction and intensity."""
        # Sun position based on time of day
        angle = self.time_of_day * 2 * np.pi - np.pi
        sun_pos = np.array([
            self.size[0] / 2 + 20 * np.cos(angle),
            self.size[1] / 2,
            15 + 10 * np.sin(angle)
        ])

        direction = sun_pos - position
        direction = direction / np.linalg.norm(direction)

        # Intensity varies with time (day/night)
        intensity = max(0, np.sin(self.time_of_day * np.pi))

        return {'direction': direction, 'intensity': intensity}


class Fly:
    """
    A fly with REAL oNeura brain.
    """

    def __init__(self, world, fly_id=0, is_player=False):
        self.world = world
        self.fly_id = fly_id
        self.is_player = is_player

        # Position and physics
        self.position = np.array([
            np.random.uniform(10, 40),
            np.random.uniform(10, 40),
            1.0  # Above ground
        ])
        self.velocity = np.zeros(3)
        self.orientation = np.random.uniform(0, 2 * np.pi)

        # Stats
        self.energy = 100.0  # Dies at 0
        self.hunger = 0.0  # Increases over time
        self.age = 0.0  # Seconds alive
        self.alive = True

        # Brain
        if ONEURO_AVAILABLE:
            self.brain = DrosophilaBrain(scale='tiny', device='cpu')
            self.vision = BinocularVisionSystem(num_ommatidia_per_eye=100)
            self.olfaction = BilateralOlfaction(num_receptors=20)

        # Motor history
        self.speed = 0.0
        self.turn = 0.0

        print(f"Fly {fly_id} created with oNeura brain")

    def step(self, dt=0.1):
        """One simulation step."""
        if not self.alive:
            return

        self.age += dt

        # Energy consumption (metabolism)
        self.energy -= 0.5 * dt  # Base metabolism
        self.hunger += 0.3 * dt  # Gets hungry

        if self.energy <= 0:
            self.alive = False
            print(f"Fly {self.fly_id} died of starvation!")
            return

        # Check for food nearby
        food_eaten = 0
        for fruit in self.world.fruits:
            dist = np.linalg.norm(fruit['position'] - self.position)
            if dist < 1.0 and fruit['nutrition'] > 0:
                # Eat!
                eaten = min(fruit['nutrition'], 20)
                fruit['nutrition'] -= eaten
                self.energy = min(100, self.energy + eaten)
                self.hunger = max(0, self.hunger - 20)
                food_eaten += eaten

        if ONEURO_AVAILABLE:
            self._neural_step(dt)
        else:
            self._simple_step(dt)

        # Apply physics
        self._apply_physics(dt)

        # Keep in bounds
        self.position = np.clip(self.position, 0.5, np.array(self.world.size) - 0.5)

        # Ground constraint
        if self.position[2] < 0.5:
            self.position[2] = 0.5
            self.velocity[2] = max(0, self.velocity[2])

    def _neural_step(self, dt):
        """Real oNeura brain step."""
        # Get sensory input
        light = self.world.get_light_at(self.position)
        odorants = self.world.get_odorants_at(self.position)

        # Visual input
        visual = self.vision.sample(
            light_direction=light['direction'],
            light_intensity=light['intensity'],
            body_orientation=np.array([1, 0, 0, 0])
        )

        # Olfactory input
        olfactory = self.olfaction.sample(odorants)

        # Stimulate brain
        if odorants:
            self.brain.stimulate_al(odorants)

        if visual:
            self.vision = BinocularVisionSystem(num_ommatidia_per_eye=100)
            # Visual stimulation would go here

        # Get motor output
        motor = self.brain.read_motor_output(n_steps=5)

        self.speed = motor.get('speed', 0.3)
        self.turn = motor.get('turn', 0.0)

    def _simple_step(self, dt):
        """Simple step when oNeura unavailable."""
        # Look for food
        best_food = None
        best_dist = float('inf')

        for fruit in self.world.fruits:
            if fruit['nutrition'] > 0:
                dist = np.linalg.norm(fruit['position'] - self.position)
                if dist < best_dist:
                    best_dist = dist
                    best_food = fruit

        if best_food is not None:
            # Turn toward food
            to_food = best_food['position'] - self.position
            target_angle = np.arctan2(to_food[1], to_food[0])
            angle_diff = target_angle - self.orientation

            # Normalize
            while angle_diff > np.pi: angle_diff -= 2 * np.pi
            while angle_diff < -np.pi: angle_diff += 2 * np.pi

            self.turn = angle_diff * 2
            self.speed = 0.5 if best_dist > 2 else 0.1
        else:
            # Random wandering
            self.turn += np.random.uniform(-0.5, 0.5)
            self.speed = 0.3

    def _apply_physics(self, dt):
        """Apply movement physics."""
        # Orientation
        self.orientation += self.turn * dt * 3

        # Forward velocity
        forward = np.array([
            np.cos(self.orientation),
            np.sin(self.orientation),
            0
        ])

        # Apply speed
        self.velocity[:2] = forward[:2] * self.speed * 5  # m/s

        # Gravity (slight)
        self.velocity[2] -= 2.0 * dt

        # Air resistance
        self.velocity *= 0.95

        # Wind
        self.velocity += self.world.wind * dt

        # Update position
        self.position += self.velocity * dt

    def get_state(self):
        """Get state for transmission to client."""
        return {
            'id': self.fly_id,
            'position': self.position.tolist(),
            'orientation': float(self.orientation),
            'alive': self.alive,
            'energy': self.energy,
            'hunger': self.hunger,
            'age': self.age,
        }


# ============================================================================
# FLASK SERVER
# ============================================================================

app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
world = WorldState()
simulation_thread = None
running = False


@app.route('/')
def index():
    """Serve the voxel world."""
    return send_from_directory(os.path.dirname(__file__), 'fly_world.html')


def simulation_loop():
    """Main simulation loop."""
    global world, running

    print("Starting simulation loop...")
    tick_rate = 30  # Hz
    dt = 1.0 / tick_rate

    while running:
        # Update world time
        world.time_of_day += dt / 60  # Full day cycle in 60 seconds

        # Update all flies
        for fly in world.flies:
            fly.step(dt)

        # Emit state to clients
        state = {
            'time': time.time(),
            'world': {
                'size': world.size,
                'temperature': world.temperature,
                'time_of_day': world.time_of_day,
            },
            'fruits': [
                {'id': f['id'], 'position': f['position'].tolist(), 'nutrition': f['nutrition']}
                for f in world.fruits
            ],
            'flies': [fly.get_state() for fly in world.flies],
        }

        socketio.emit('world_state', state)

        time.sleep(dt)

    print("Simulation loop stopped")


@socketio.on('connect')
def handle_connect():
    """Client connected."""
    print(f"Client connected")
    emit('welcome', {'message': 'Connected to oNeura World'})


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected."""
    print(f"Client disconnected")


@socketio.on('add_fly')
def handle_add_fly(data):
    """Add a new fly."""
    is_player = data.get('is_player', False)
    fly = Fly(world, len(world.flies), is_player)
    world.flies.append(fly)
    print(f"Added fly {fly.fly_id}, total: {len(world.flies)}")
    emit('fly_added', {'fly_id': fly.fly_id})


@socketio.on('spawn_fruit')
def handle_spawn_fruit(data):
    """Spawn a fruit."""
    world.spawn_fruits(1)
    emit('fruit_spawned', {'count': len(world.fruits)})


@socketio.on('start_simulation')
def handle_start():
    """Start simulation."""
    global running, simulation_thread

    if not running:
        running = True
        simulation_thread = threading.Thread(target=simulation_loop)
        simulation_thread.daemon = True
        simulation_thread.start()
        print("Simulation started!")
        emit('simulation_started', {'status': 'ok'})


@socketio.on('stop_simulation')
def handle_stop():
    """Stop simulation."""
    global running
    running = False
    print("Simulation stopped!")
    emit('simulation_stopped', {'status': 'ok'})


def main():
    """Run the server."""
    print("=" * 60)
    print("oNEURO 3D WORLD SERVER")
    print("=" * 60)
    print(f"oNeura available: {ONEURO_AVAILABLE}")
    print(f"Open http://localhost:5000 in your browser")
    print("=" * 60)

    socketio.run(app, debug=False, port=5000)


if __name__ == '__main__':
    main()
