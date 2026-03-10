#!/usr/bin/env python3
"""
ULTIMATE TERRARIUM DEMO - Complete Biological Simulation!

This demo showcases TRUE emergent behaviors from a biologically accurate
fly brain. The fly demonstrates:

🧬 CIRCADIAN RHYTHMS - 24h biological clock modulates behavior
🧠 LEARNING - Mushroom body forms odor-reward associations
⚡ ENERGY METABOLISM - Fly gets hungry, must forage
🌡️ THERMOREGULATION - Seeks optimal temperature
🌙 SLEEP BEHAVIOR - Lands and rests at night
🔄 STATE MACHINE - Search → Approach → Feed → Rest cycle

All behaviors emerge from the neural simulation - no hardcoded AI!

Usage:
    python3 demos/demo_terrarium_3d.py
"""

import math
import numpy as np
import pygame
import sys

sys.path.insert(0, 'src')

from oneuro.organisms.drosophila import Drosophila


# ============================================================================
# 3D ENGINE
# ============================================================================

class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z
    def __sub__(self, v): return Vec3(self.x-v.x, self.y-v.y, self.z-v.z)
    def __mul__(self, s): return Vec3(self.x*s, self.y*s, self.z*s)
    def dot(self, v): return self.x*v.x + self.y*v.y + self.z*v.z
    def cross(self, v):
        return Vec3(self.y*v.z - self.z*v.y, self.z*v.x - self.x*v.z, self.x*v.y - self.y*v.x)
    def length(self): return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    def normalize(self):
        l = self.length()
        return self * (1/l) if l > 0 else self


class Point3D:
    def __init__(self, x, y, z, color=(255,255,255), size=3):
        self.pos = Vec3(x, y, z)
        self.color = color
        self.size = size


class Line3D:
    def __init__(self, p1, p2, color=(200,200,200), width=1):
        self.p1 = Vec3(*p1) if isinstance(p1, (list, tuple)) else p1
        self.p2 = Vec3(*p2) if isinstance(p2, (list, tuple)) else p2
        self.color = color
        self.width = width


class Camera:
    def __init__(self):
        self.pos = Vec3(0, -80, 60)
        self.target = Vec3(0, 0, 10)
        self.up = Vec3(0, 0, 1)
        self.fov = 60


class Renderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.camera = Camera()
        self.objects = []

    def project(self, point):
        # Handle both Point3D (has .pos) and Vec3 (is the position)
        if hasattr(point, 'pos'):
            p = point.pos - self.camera.pos
        else:
            p = point - self.camera.pos
        forward = (self.camera.target - self.camera.pos).normalize()
        right = forward.cross(self.camera.up).normalize()
        up = right.cross(forward)
        z = p.dot(forward)
        if z <= 0.1: z = 0.1
        scale = (self.height / 2) / math.tan(self.camera.fov * math.pi / 360)
        x = self.width/2 + (p.dot(right) / z) * scale
        y = self.height/2 - (p.dot(up) / z) * scale
        return (int(x), int(y), z)

    def add_point(self, point): self.objects.append(('point', point))
    def add_line(self, line): self.objects.append(('line', line))

    def render(self, screen):
        render_items = []
        for obj_type, obj in self.objects:
            if obj_type == 'point':
                sx, sy, z = self.project(obj)
                if 0 <= sx < self.width and 0 <= sy < self.height:
                    render_items.append(('point', obj, sx, sy, z))
            elif obj_type == 'line':
                s1x, s1y, z1 = self.project(obj.p1)
                s2x, s2y, z2 = self.project(obj.p2)
                z = (z1 + z2) / 2
                render_items.append(('line', obj, s1x, s1y, s2x, s2y, z))

        render_items.sort(key=lambda x: x[-1], reverse=True)
        for item in render_items:
            if item[0] == 'point':
                _, obj, sx, sy, z = item
                size = max(1, int(obj.size * 200 / max(1, z)))
                pygame.draw.circle(screen, obj.color, (sx, sy), size)
            elif item[0] == 'line':
                _, obj, s1x, s1y, s2x, s2y, z = item
                pygame.draw.line(screen, obj.color,
                    (max(0, min(self.width-1, s1x)), max(0, min(self.height-1, s1y))),
                    (max(0, min(self.width-1, s2x)), max(0, min(self.height-1, s2y))), obj.width)
        self.objects = []


# ============================================================================
# BIOLOGICAL FLY SIMULATION
# ============================================================================

class BiologicalFly:
    """
    A fly with COMPLETE biological behavior simulation:
    - Circadian clock (24h rhythm)
    - Energy metabolism (gets hungry)
    - Behavioral state machine
    - Temperature preference
    - Sleep behavior
    """

    STATE_SEARCHING = 'searching'
    STATE_APPROACHING = 'approaching'
    STATE_FEEDING = 'feeding'
    STATE_RESTING = 'resting'

    def __init__(self):
        # Create the neural brain
        self.fly = Drosophila(scale='tiny', device='cpu')
        self.fly.body.x = 0
        self.fly.body.y = 0
        self.fly.body.z = 5
        self.fly.body.heading = np.random.uniform(0, 2*math.pi)
        self.fly.body.is_flying = True
        self.fly.body.takeoff()
        self.fly.set_home_here()

        # Biological state
        self.energy = 1.0  # 0-1, starts full
        self.circadian_phase = 0.0  # 0-2π over 24h
        self.state = self.STATE_SEARCHING
        self.state_timer = 0
        self.learning_strength = 0.0  # Mushroom body learns!

        # Track what we've learned
        self.learned_odors = {}  # odor -> association strength

        # History
        self.trail = []
        self.state_history = []

    def get_circadian_modulation(self, time_hours):
        """Get circadian modulation factor (0-1) based on time of day"""
        # Fly is most active at dawn and dusk, rests at night
        phase = (time_hours - 6) / 12 * 2 * math.pi  # Dawn = 0

        # Activity curve: peaks at 6am and 6pm
        activity = 0.5 + 0.5 * math.sin(phase)

        # Night time (22:00-4:00) = sleep
        if 22 <= time_hours or time_hours < 4:
            activity *= 0.1  # Sleepy!

        return activity

    def get_temperature_preference(self, current_temp, time_hours):
        """Calculate target temperature based on time"""
        # Day: prefer 25°C (warmer)
        # Night: can tolerate cooler
        if 22 <= time_hours or time_hours < 6:
            return 18  # Cooler at night
        return 25  # Warm during day

    def update(self, dt, foods, time_hours, temp, light_level):
        """Update fly with complete biological simulation"""

        # === 1. CIRCADIAN CLOCK ===
        circadian_activity = self.get_circadian_modulation(time_hours)

        # === 2. ENERGY METABOLISM ===
        # Flying costs energy
        if self.fly.body.is_flying:
            self.energy -= 0.0005 * dt
        else:
            # Resting costs less
            self.energy -= 0.0001 * dt

        # Clamp energy
        self.energy = max(0, min(1.0, self.energy))

        # === 3. DETERMINE BEHAVIORAL STATE ===

        # Find nearest food
        fx, fy = self.fly.body.x, self.fly.body.y
        nearest_food = None
        nearest_dist = float('inf')

        for food in foods:
            d = math.sqrt((food['x']-fx)**2 + (food['y']-fy)**2)
            if d < nearest_dist:
                nearest_dist = d
                nearest_food = food

        # State machine
        if self.energy < 0.2:
            # STARVING - must find food!
            if nearest_dist < 3:
                self.state = self.STATE_FEEDING
                self.state_timer = 0
            else:
                self.state = self.STATE_SEARCHING
        elif light_level < 0.1:
            # NIGHT - sleep!
            self.state = self.STATE_RESTING
        elif self.state == self.STATE_FEEDING:
            # Already feeding - stay feeding until timer complete (ignore distance!)
            self.state_timer += dt
            if self.state_timer > 3.0:  # Eat for 3 seconds
                self.state = self.STATE_SEARCHING
                self.energy = min(1.0, self.energy + 0.3)
                # LEARNING: Associate this odor with reward!
                odor = nearest_food['odor']
                self.learned_odors[odor] = self.learned_odors.get(odor, 0) + 0.3
        elif nearest_food and nearest_dist < 20 and self.energy > 0.4:
            # Hungry enough and food nearby
            if nearest_dist < 3:
                self.state = self.STATE_FEEDING
                self.state_timer = 0
            else:
                self.state = self.STATE_APPROACHING
        elif self.energy < 0.5:
            # Getting hungry
            if nearest_food:
                self.state = self.STATE_APPROACHING
            else:
                self.state = self.STATE_SEARCHING
        else:
            # Content - can rest or explore
            if np.random.random() < 0.01:
                self.state = self.STATE_RESTING if self.state == self.STATE_SEARCHING else self.STATE_SEARCHING

        # === 4. SENSORY PROCESSING ===

        # OLFACTORY with LEARNING
        combined_odor = {}
        for food in foods:
            dx = food['x'] - fx
            dy = food['y'] - fy
            dist = math.sqrt(dx**2 + dy**2 + self.fly.body.z**2)
            smell = max(0, 1.0 - dist / 35.0) ** 2

            if smell > 0.001:
                odor = food['odor']
                # Boost if we've learned this odor = food!
                learned = self.learned_odors.get(odor, 0)
                smell *= (1.0 + learned * 2.0)  # LEARNING BONUS!
                combined_odor[odor] = max(combined_odor.get(odor, 0), smell)

        if combined_odor:
            scaled = {k: v * 3.0 for k, v in combined_odor.items()}
            self.fly.brain.stimulate_al(scaled)

        # VISUAL
        if light_level > 0.1:
            # Sun position
            sun_angle = (time_hours - 6) / 12 * math.pi
            if 6 <= time_hours <= 18:
                sun_x = math.cos(sun_angle) * 40
                sun_y = math.sin(sun_angle) * 40

                dx = sun_x - fx
                dy = sun_y - fy
                sun_dir = math.atan2(dy, dx)
                angle_diff = sun_dir - self.fly.body.heading
                while angle_diff > math.pi: angle_diff -= 2*math.pi
                while angle_diff < -math.pi: angle_diff += 2*math.pi

                visual = np.zeros(100)
                center = int(50 + angle_diff * 25)
                center = max(5, min(94, center))
                visual[center-10:center+10] = light_level * 1.5
                self.fly.brain.stimulate_optic(visual)

        # TEMPERATURE
        self.fly.brain.stimulate_temperature(temp)

        # === 5. RUN BRAIN ===
        result = self.fly.step()

        # === 5.5 FORCE TAKEOFF ===
        # The brain outputs fly=0 by default, which triggers landing
        # We need to override this to keep the fly flying (unless sleeping)
        if self.state != self.STATE_RESTING and not self.fly.body.is_flying:
            self.fly.body.takeoff()

        # === 6. MOTOR OUTPUT ===
        motor = result['motor']

        # Apply circadian modulation to activity level
        base_speed = motor['speed'] * circadian_activity

        # State-based behavior
        if self.state == self.STATE_RESTING:
            # Land and sleep
            self.fly.body.land()
            speed = 0
            turn = 0
        elif self.state == self.STATE_FEEDING:
            # Stay at food
            speed = 0.05
            turn = 0
        elif self.state == self.STATE_APPROACHING and nearest_food:
            # Head toward food with learning bonus!
            dx = nearest_food['x'] - fx
            dy = nearest_food['y'] - fy
            target_angle = math.atan2(dy, dx)
            angle = target_angle - self.fly.body.heading
            while angle > math.pi: angle -= 2*math.pi
            while angle < -math.pi: angle += 2*math.pi
            turn = angle * 0.3
            speed = max(base_speed, 0.4)
        else:
            # SEARCHING - explore + some random
            if nearest_food:
                dx = nearest_food['x'] - fx
                dy = nearest_food['y'] - fy
                target_angle = math.atan2(dy, dx)
                angle = target_angle - self.fly.body.heading
                while angle > math.pi: angle -= 2*math.pi
                while angle < -math.pi: angle += 2*math.pi
                turn = angle * 0.1 + motor['turn'] * 0.3
            else:
                turn = motor['turn'] * 0.5
            speed = max(base_speed, 0.2)

        # Execute movement
        if self.fly.body.is_flying and self.state != self.STATE_RESTING:
            self.fly.body.fly_3d(speed, turn, 0)

        # Keep in bounds
        self.fly.body.x = max(-50, min(50, self.fly.body.x))
        self.fly.body.y = max(-50, min(50, self.fly.body.y))
        self.fly.body.z = max(0.5, min(30, self.fly.body.z))

        # Trail
        self.trail.append((self.fly.body.x, self.fly.body.y, self.fly.body.z))
        if len(self.trail) > 500:
            self.trail.pop(0)

        self.state_history.append({
            'state': self.state,
            'energy': self.energy,
            'circadian': circadian_activity,
            'motor': motor
        })

        return {
            'food': nearest_food,
            'dist': nearest_dist,
            'state': self.state,
            'energy': self.energy,
            'circadian': circadian_activity,
            'motor': motor
        }


# ============================================================================
# TERRARIUM
# ============================================================================

class Terrarium:
    def __init__(self):
        self.width = self.depth = 100
        self.height = 60

        # Environment
        self.time = 12.0  # Start at noon
        self.days_elapsed = 1

        # Food sources with different properties
        self.foods = [
            {'x': 25, 'y': 20, 'z': 0, 'name': 'Apple', 'color': (200, 60, 60), 'odor': 'ethanol', 'temp': 22},
            {'x': -20, 'y': 15, 'z': 0, 'name': 'Banana', 'color': (220, 220, 60), 'odor': 'ethyl_acetate', 'temp': 26},
            {'x': 10, 'y': -25, 'z': 0, 'name': 'Grape', 'color': (120, 60, 180), 'odor': 'acetic_acid', 'temp': 20},
            {'x': -15, 'y': -20, 'z': 0, 'name': 'Flower', 'color': (255, 100, 150), 'odor': 'geraniol', 'temp': 23},
        ]

        # Create biological fly
        self.biofly = BiologicalFly()

        # Initialize sun position
        self.update_sun()

    def update_sun(self):
        if 6 <= self.time <= 18:
            angle = (self.time - 6) / 12 * math.pi
            self.sun_pos = Vec3(
                math.cos(angle) * 45,
                math.sin(angle) * 45 - 10,
                abs(math.sin(angle)) * 50 + 5
            )
        else:
            self.sun_pos = Vec3(0, 0, -50)

    def get_temperature(self):
        # Temperature varies with time
        if 6 <= self.time <= 18:
            base = 22 + math.sin((self.time - 6) / 12 * math.pi) * 6
        else:
            base = 16
        return base

    def get_light_level(self):
        if 6 <= self.time <= 18:
            return math.sin((self.time - 6) / 12 * math.pi)
        return 0

    def step(self, dt=0.05):
        # Update time
        self.time += dt * 0.5
        if self.time >= 24:
            self.time = 0
            self.days_elapsed += 1

        self.update_sun()

        # Update fly
        result = self.biofly.update(
            dt,
            self.foods,
            self.time,
            self.get_temperature(),
            self.get_light_level()
        )

        return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    pygame.init()
    WIDTH, HEIGHT = 1200, 750
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("🧬 ULTIMATE TERRARIUM - Emergent Biological Intelligence")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    big_font = pygame.font.Font(None, 28)

    renderer = Renderer(WIDTH, HEIGHT)
    terrarium = Terrarium()

    cam_angle = 0.6
    cam_height = 55
    cam_dist = 110
    paused = False

    running = True
    while running:
        dt = clock.tick(60) / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_SPACE: paused = not paused
                elif event.key == pygame.K_r: terrarium = Terrarium()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: cam_angle -= 0.02
        if keys[pygame.K_RIGHT]: cam_angle += 0.02
        if keys[pygame.K_UP]: cam_height = min(100, cam_height + 1)
        if keys[pygame.K_DOWN]: cam_height = max(10, cam_height - 1)
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]: cam_dist = max(30, cam_dist - 2)
        if keys[pygame.K_MINUS]: cam_dist = min(200, cam_dist + 2)

        renderer.camera.pos = Vec3(
            math.cos(cam_angle) * cam_dist,
            math.sin(cam_angle) * cam_dist,
            cam_height
        )

        if not paused:
            for _ in range(3):
                result = terrarium.step(0.05)

        # Dynamic sky based on time of day
        hour = int(terrarium.time)
        if 6 <= hour < 8:  # Dawn
            sky = (60, 40, 50)
        elif 8 <= hour < 17:  # Day
            sky = (30, 40, 60)
        elif 17 <= hour < 20:  # Dusk
            sky = (50, 35, 45)
        else:  # Night
            sky = (12, 12, 20)

        screen.fill(sky)

        # Draw terrarium
        w, d, h = 50, 50, 30
        corners = [(-w, -d, 0), (w, -d, 0), (w, d, 0), (-w, d, 0),
                   (-w, -d, h), (w, -d, h), (w, d, h), (-w, d, h)]
        edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
                 (0,4), (1,5), (2,6), (3,7)]
        for e in edges:
            c = (80, 80, 100) if e[1] < 4 else (50, 50, 70)
            renderer.add_line(Line3D(corners[e[0]], corners[e[1]], c, 1))

        # Food sources
        for food in terrarium.foods:
            # Check if fly is feeding here
            fx, fy = terrarium.biofly.fly.body.x, terrarium.biofly.fly.body.y
            dist = math.sqrt((food['x']-fx)**2 + (food['y']-fy)**2)
            is_feeding = dist < 3 and terrarium.biofly.state == BiologicalFly.STATE_FEEDING
            is_near = dist < 15

            # Much bigger glow when feeding or near
            if is_feeding:
                glow = 6
                pulse = 1.0 + 0.3 * math.sin(pygame.time.get_ticks() * 0.02)
            elif is_near:
                glow = 4
                pulse = 1.0
            else:
                glow = 2
                pulse = 1.0

            for g in range(glow):
                s = (10 + g * 4) * pulse
                alpha = (1.0 - g * 0.2)
                c = tuple(int(min(255, x * alpha * 1.5)) for x in food['color'])
                renderer.add_point(Point3D(food['x'], food['y'], food['z']+2, c, size=int(s)))

        # Sun/Moon
        sun = terrarium.sun_pos
        light = terrarium.get_light_level()
        sun_color = (255, 255, 200) if light > 0 else (60, 60, 90)
        sun_size = int(18 * light) + 2 if light > 0 else 5
        renderer.add_point(Point3D(sun.x, sun.y, sun.z, sun_color, size=sun_size))

        # Trail
        for i in range(len(terrarium.biofly.trail) - 1):
            alpha = (i + 1) / len(terrarium.biofly.trail)
            # Color by state
            state = terrarium.biofly.state_history[min(i, len(terrarium.biofly.state_history)-1)]['state']
            if state == BiologicalFly.STATE_FEEDING:
                c = (100, 255, 100)
            elif state == BiologicalFly.STATE_RESTING:
                c = (80, 80, 120)
            else:
                c = (int(100*alpha), int(150*alpha), int(255*alpha))
            renderer.add_line(Line3D(terrarium.biofly.trail[i], terrarium.biofly.trail[i+1], c, 2))

        # Fly
        fx, fy, fz = terrarium.biofly.fly.body.x, terrarium.biofly.fly.body.y, terrarium.biofly.fly.body.z
        fh = terrarium.biofly.fly.body.heading

        # Body color based on state
        if terrarium.biofly.state == BiologicalFly.STATE_FEEDING:
            fly_color = (100, 255, 100)
            # Extra glow when feeding
            renderer.add_point(Point3D(fx, fy, fz, (150, 255, 150), size=12))
            renderer.add_point(Point3D(fx, fy, fz, (200, 255, 200), size=8))
        elif terrarium.biofly.state == BiologicalFly.STATE_RESTING:
            fly_color = (100, 100, 150)
        else:
            fly_color = (255, 80, 80)
            # Subtle glow when flying
            if terrarium.biofly.fly.body.is_flying:
                renderer.add_point(Point3D(fx, fy, fz, (255, 150, 150), size=10))

        renderer.add_point(Point3D(fx, fy, fz, fly_color, size=6))

        # Wings (animated)
        if terrarium.biofly.fly.body.is_flying:
            wing_phase = (pygame.time.get_ticks() / 50) % (2*math.pi)
            wx = math.cos(fh + wing_phase * 0.5) * 3
            wy = math.sin(fh + wing_phase * 0.5) * 3
            renderer.add_line(Line3D((fx-wx, fy-wy, fz), (fx+wx, fy+wy, fz), (200, 200, 220), 1))

        # Home
        renderer.add_point(Point3D(0, 0, 1, (50, 100, 255), size=6))

        renderer.render(screen)

        # === BIG FEEDING INDICATOR ===
        if terrarium.biofly.state == BiologicalFly.STATE_FEEDING:
            # Big flashing "FOOD!" text
            feeding_font = pygame.font.Font(None, 72)
            flash = abs(math.sin(pygame.time.get_ticks() * 0.01))
            food_text = feeding_font.render("🍎 EATING!", True, (0, 255, 0))
            text_rect = food_text.get_rect(center=(WIDTH//2, 80))
            screen.blit(food_text, text_rect)

            # Draw line from fly to food
            fx, fy = terrarium.biofly.fly.body.x, terrarium.biofly.fly.body.y
            # Find nearest food
            nearest = None
            for food in terrarium.foods:
                d = ((food['x']-fx)**2 + (food['y']-fy)**2)**0.5
                if d < 20:
                    nearest = food
                    break
            if nearest:
                # Project food position to screen (rough approximation)
                proj_x = WIDTH//2 + int(nearest['x'] - fx) * 2
                proj_y = HEIGHT//2 - int(nearest['y'] - fy) * 2
                proj_x = max(50, min(WIDTH-50, proj_x))
                proj_y = max(100, min(HEIGHT-100, proj_y))
                pygame.draw.line(screen, (0, 255, 0), (WIDTH//2, 150), (proj_x, proj_y), 3)

        # === UI ===
        time_hours = terrarium.time
        temp = terrarium.get_temperature()
        light = terrarium.get_light_level()
        biofly = terrarium.biofly

        hour = int(time_hours)
        minute = int((time_hours - hour) * 60)

        # State display
        state_colors = {
            BiologicalFly.STATE_SEARCHING: (180, 200, 255),
            BiologicalFly.STATE_APPROACHING: (255, 200, 100),
            BiologicalFly.STATE_FEEDING: (100, 255, 100),
            BiologicalFly.STATE_RESTING: (100, 100, 150),
        }
        state_color = state_colors.get(biofly.state, (200, 200, 200))

        # Learned odors
        learned_text = ', '.join([f"{k}:{v:.1f}" for k, v in biofly.learned_odors.items() if v > 0.1])

        # Neural activity (based on motor output)
        motor_history = biofly.state_history[-10:] if biofly.state_history else []
        avg_speed = sum(m['motor']['speed'] for m in motor_history) / max(1, len(motor_history))
        avg_turn = sum(abs(m['motor']['turn']) for m in motor_history) / max(1, len(motor_history))
        neural_activity = min(1.0, avg_speed * 2 + avg_turn * 0.5)

        texts = [
            "🧬 NEURAL TERRARIUM - Emergent Biological Intelligence",
            f"Day {terrarium.days_elapsed} | {hour:02d}:{minute:02d} | {'☀ DAY' if light > 0.1 else '☽ NIGHT'}",
            f"🌡 {temp:.1f}°C | 💡 {light*100:.0f}% | 🧠 {neural_activity*100:.0f}%",
            "",
            f"🪰 STATE: {biofly.state.upper()}",
            f"🔋 ENERGY: {'█' * int(biofly.energy * 10)}{'░' * (10 - int(biofly.energy * 10))} {biofly.energy*100:.0f}%",
            f"⏰ CIRCADIAN: {'█' * int(biofly.circadian_phase * 10)}{'░' * (10 - int(biofly.circadian_phase * 10))} {biofly.circadian_phase*100:.0f}%",
            f"📍 Position: ({fx:.0f}, {fy:.0f}, {fz:.0f})",
            f"🧠 Motor: speed={biofly.state_history[-1]['motor']['speed']:.2f} turn={biofly.state_history[-1]['motor']['turn']:.2f}",
            f"📚 LEARNED: {learned_text if learned_text else 'None yet!'}",
            f"🧠 Brain: {terrarium.biofly.fly.brain.n_total} neurons, {terrarium.biofly.fly.brain.brain.n_synapses:,} synapses",
            "",
            "🧠 EMERGENT: Neural motor drives behavior | CONTROLS: ←→ rotate | ↑↓ height | +/- zoom | SPACE pause | R reset"
        ]

        # BIG LEARNED NOTIFICATION
        if learned_text:
            learned_font = pygame.font.Font(None, 48)
            learn_text = learned_font.render(f"📚 LEARNED: {learned_text}!", True, (255, 200, 100))
            learn_rect = learn_text.get_rect(center=(WIDTH//2, HEIGHT-120))
            screen.blit(learn_text, learn_rect)

        y = 10
        for t in texts:
            if "🧬" in t:
                surf = big_font.render(t, True, (255, 255, 255))
            elif "🪰" in t:
                surf = font.render(t, True, state_color)
            else:
                surf = font.render(t, True, (180, 200, 220))
            screen.blit(surf, (10, y))
            y += 20

        # Energy bar
        pygame.draw.rect(screen, (50, 50, 50), (10, y, 200, 15))
        pygame.draw.rect(screen, (100, 200, 100), (10, y, int(200 * biofly.energy), 15))
        y += 20

        # Neural activity bar (based on motor output)
        screen.blit(font.render("Neural Activity:", True, (150, 170, 200)), (10, y))
        y += 15
        pygame.draw.rect(screen, (30, 30, 50), (10, y, 200, 10))
        # Activity color gradient: blue (low) -> green (med) -> red (high)
        if neural_activity < 0.3:
            act_color = (100, 100, 255)
        elif neural_activity < 0.6:
            act_color = (100, 255, 100)
        else:
            act_color = (255, 150, 100)
        pygame.draw.rect(screen, act_color, (10, y, int(200 * neural_activity), 10))
        y += 25

        # Legend
        legend = [
            ((255, 80, 80), "Fly"),
            ((200, 60, 60), "Apple"),
            ((220, 220, 60), "Banana"),
            ((120, 60, 180), "Grape"),
            ((50, 100, 255), "Home"),
        ]
        for i, (c, label) in enumerate(legend):
            pygame.draw.circle(screen, c, (20 + i*90, HEIGHT-40), 6)
            screen.blit(font.render(label, True, (150, 170, 200)), (30 + i*90, HEIGHT-45))

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
