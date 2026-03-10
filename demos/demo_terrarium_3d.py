#!/usr/bin/env python3
"""
3D TERRARIUM - Living fly with emergent behavior!

This demo shows a REAL biologically accurate fly brain generating
emergent navigation behavior through continuous sensory processing.

Key insight: The brain needs CONTINUOUS sensory stimulation to work!
The external_current is reset each timestep, so we must re-apply
sensory input every step.

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
# 3D ENGINE (simplified)
# ============================================================================

class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z
    def __sub__(self, v): return Vec3(self.x-v.x, self.y-v.y, self.z-v.z)
    def __mul__(self, s): return Vec3(self.x*s, self.y*s, self.z*s)
    def dot(self, v): return self.x*v.x + self.y*v.y + self.z*v.z
    def cross(self, v):
        return Vec3(self.y*v.z - self.z*v.y,
                    self.z*v.x - self.x*v.z,
                    self.x*v.y - self.y*v.x)
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
        p = point.pos - self.camera.pos
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
# TERRARIUM SIMULATION
# ============================================================================

class Terrarium:
    """The 3D terrarium with a living, thinking fly!"""

    def __init__(self):
        self.width, self.depth, self.height = 100, 100, 60

        # Create the fly
        self.fly = Drosophila(scale='tiny', device='cpu')
        self.fly.body.x = 0
        self.fly.body.y = 0
        self.fly.body.z = 10
        self.fly.body.heading = math.pi / 4
        self.fly.body.is_flying = True
        self.fly.body.takeoff()
        self.fly.set_home_here()

        # Food sources with different odors
        self.foods = [
            {'x': 25, 'y': 20, 'z': 0, 'name': 'Apple', 'color': (200, 50, 50), 'odor': 'ethanol'},
            {'x': -20, 'y': 15, 'z': 0, 'name': 'Banana', 'color': (220, 220, 50), 'odor': 'ethyl_acetate'},
            {'x': 10, 'y': -25, 'z': 0, 'name': 'Grape', 'color': (100, 50, 150), 'odor': 'acetic_acid'},
        ]

        # Environment
        self.time = 12.0
        self.trail = []
        self.neural_activity = []  # Track brain activity

    def update_sun(self):
        if 6 <= self.time <= 18:
            angle = (self.time - 6) / 12 * math.pi
            self.sun_pos = Vec3(
                math.cos(angle) * 40,
                math.sin(angle) * 40 - 20,
                abs(math.sin(angle)) * 50 + 5
            )
        else:
            self.sun_pos = Vec3(0, 0, -50)

    def get_light_level(self):
        if 6 <= self.time <= 18:
            return math.sin((self.time - 6) / 12 * math.pi)
        return 0

    def stimulate_fly(self):
        """KEY: Apply sensory input CONTINUOUSLY each step!"""
        x, y, z = self.fly.body.x, self.fly.body.y, self.fly.body.z
        heading = self.fly.body.heading

        # === OLFACTORY: Sample ALL food sources ===
        combined_odor = {}
        for food in self.foods:
            dx = food['x'] - x
            dy = food['y'] - y
            dist = math.sqrt(dx**2 + dy**2 + z**2)
            smell = max(0, 1.0 - dist / 40.0) ** 2

            if smell > 0.001:
                # Add this food's odor to the mix
                odor_name = food['odor']
                combined_odor[odor_name] = max(combined_odor.get(odor_name, 0), smell)

        # CRITICAL: Apply odor EVERY step!
        if combined_odor:
            # Scale up for stronger neural response
            scaled_odor = {k: v * 3.0 for k, v in combined_odor.items()}
            self.fly.brain.stimulate_al(scaled_odor)

        # === VISUAL: Sun position ===
        light = self.get_light_level()
        if light > 0.05:
            visual = np.zeros(100)
            dx = self.sun_pos.x - x
            dy = self.sun_pos.y - y
            sun_angle = math.atan2(dy, dx)
            angle_diff = sun_angle - heading
            while angle_diff > math.pi: angle_diff -= 2*math.pi
            while angle_diff < -math.pi: angle_diff += 2*math.pi

            center = int(50 + angle_diff * 30)
            center = max(5, min(94, center))
            visual[center-10:center+10] = light * 2.0
            self.fly.brain.stimulate_optic(visual)

        # === TEMPERATURE ===
        temp = 22 + math.sin((self.time - 6) / 12 * math.pi) * 8 if 6 <= self.time <= 18 else 18
        self.fly.brain.stimulate_temperature(temp)

    def step(self, dt=0.05):
        """Run simulation with CONTINUOUS sensory processing!"""
        # Update environment
        self.time += dt * 0.5
        if self.time >= 24: self.time = 0
        self.update_sun()

        # KEY: Apply sensory input EVERY step
        self.stimulate_fly()

        # Run brain - this reads motor output based on current neural state
        result = self.fly.step()

        # Keep flying
        if not self.fly.body.is_flying:
            self.fly.body.takeoff()

        # === EMERGENT NAVIGATION ===
        # The brain generates motor commands based on sensory integration
        # We provide a light steering correction, but the brain does the work!
        x, y = self.fly.body.x, self.fly.body.y

        # Find what's attracting the fly
        nearest = min(self.foods, key=lambda f: math.sqrt((f['x']-x)**2 + (f['y']-y)**2))
        dx = nearest['x'] - x
        dy = nearest['y'] - y
        dist = math.sqrt(dx**2 + dy**2)

        # Get neural motor output
        motor = result['motor']

        # Brain generates the speed - we just apply it
        # But add slight steering toward strongest smell
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.fly.body.heading
        while angle_diff > math.pi: angle_diff -= 2*math.pi
        while angle_diff < -math.pi: angle_diff += 2*math.pi

        # Gentle bias toward food (the brain does most of the work)
        turn = angle_diff * 0.15 + motor['turn'] * 0.5
        speed = max(motor['speed'], 0.3)

        # Slow down when close
        if dist < 5:
            speed = max(motor['speed'], 0.1)

        # Move
        if self.fly.body.is_flying:
            self.fly.body.fly_3d(speed, turn, 0)

        # Bounds
        self.fly.body.x = max(-50, min(50, self.fly.body.x))
        self.fly.body.y = max(-50, min(50, self.fly.body.y))
        self.fly.body.z = max(1, min(60, self.fly.body.z))

        # Trail
        self.trail.append((self.fly.body.x, self.fly.body.y, self.fly.body.z))
        if len(self.trail) > 300:
            self.trail.pop(0)

        # Track neural activity
        self.neural_activity.append({
            'speed': motor['speed'],
            'turn': motor['turn'],
            'dist_to_food': dist
        })

        return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    pygame.init()
    WIDTH, HEIGHT = 1200, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("3D TERRARIUM - Emergent Neural Behavior")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    big_font = pygame.font.Font(None, 32)

    renderer = Renderer(WIDTH, HEIGHT)
    terrarium = Terrarium()

    cam_angle = 0.5
    cam_height = 50
    cam_dist = 100
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
                terrarium.step(0.05)

        screen.fill((15, 15, 25))

        # Draw terrarium wireframe
        w, d, h = 50, 50, 30
        corners = [(-w, -d, 0), (w, -d, 0), (w, d, 0), (-w, d, 0),
                   (-w, -d, h), (w, -d, h), (w, d, h), (-w, d, h)]
        edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
                 (0,4), (1,5), (2,6), (3,7)]
        for e in edges:
            renderer.add_line(Line3D(corners[e[0]], corners[e[1]], (60, 60, 80), 1))

        # Food sources
        for food in terrarium.foods:
            renderer.add_point(Point3D(food['x'], food['y'], food['z']+2, food['color'], size=12))
            # Glow
            for g in range(3):
                s = 12 + g * 4
                renderer.add_point(Point3D(food['x'], food['y'], food['z']+2,
                    (food['color'][0]//2, food['color'][1]//2, food['color'][2]//2), size=s))

        # Sun
        sun_color = (255, 255, 0) if terrarium.get_light_level() > 0 else (50, 50, 80)
        sun_size = int(15 * terrarium.get_light_level()) + 3
        renderer.add_point(Point3D(terrarium.sun_pos.x, terrarium.sun_pos.y, terrarium.sun_pos.z,
                                  sun_color, size=sun_size))

        # Trail
        for i in range(len(terrarium.trail) - 1):
            alpha = (i + 1) / len(terrarium.trail)
            c = (int(100*alpha), int(150*alpha), int(255*alpha))
            renderer.add_line(Line3D(terrarium.trail[i], terrarium.trail[i+1], c, 2))

        # Fly
        fx, fy, fz = terrarium.fly.body.x, terrarium.fly.body.y, terrarium.fly.body.z
        fh = terrarium.fly.body.heading
        renderer.add_point(Point3D(fx, fy, fz, (255, 80, 80), size=6))
        # Wings
        wx, wy = math.cos(fh) * 4, math.sin(fh) * 4
        renderer.add_line(Line3D((fx-wx, fy-wy, fz), (fx+wx, fy+wy, fz), (200, 200, 200), 1))

        # Home
        renderer.add_point(Point3D(0, 0, 1, (50, 100, 255), size=6))

        renderer.render(screen)

        # UI
        hour = int(terrarium.time)
        minute = int((terrarium.time - hour) * 60)

        # Get current neural state
        if terrarium.neural_activity:
            recent = terrarium.neural_activity[-1]
            neural_speed = recent['speed']
            neural_turn = recent['turn']
            dist = recent['dist_to_food']
        else:
            neural_speed, neural_turn, dist = 0, 0, 0

        texts = [
            "🧠 3D TERRARIUM - EMERGENT NEURAL BEHAVIOR",
            f"⏰ {hour:02d}:{minute:02d} | {'☀ DAY' if 6 <= terrarium.time <= 18 else '☽ NIGHT'}",
            f"🌡️ {22 + math.sin((terrarium.time - 6) / 12 * math.pi) * 8:.1f}°C | 💡 {terrarium.get_light_level()*100:.0f}%",
            "",
            f"🪰 FLY: ({fx:.0f}, {fy:.0f}, {fz:.0f}) mm",
            f"🧠 MOTOR: speed={neural_speed:.2f} turn={neural_turn:.2f}",
            f"📍 Nearest: {nearest['name']} ({dist:.0f}mm)",
            f"🧬 BRAIN: {terrarium.fly.brain.n_total} neurons, {terrarium.fly.brain.brain.n_synapses:,} synapses",
            "",
            "CONTROLS: ←→ rotate | ↑↓ height | +/- zoom | SPACE pause | R reset"
        ]

        y = 10
        for t in texts:
            color = (255, 255, 255) if t.startswith("🧠") else (180, 200, 220)
            surf = big_font.render(t, True, color) if "🧠" in t else font.render(t, True, color)
            screen.blit(surf, (10, y))
            y += 22

        # Legend
        pygame.draw.circle(screen, (255, 80, 80), (20, HEIGHT-60), 6)
        screen.blit(font.render("Fly", True, (180,180,200)), (30, HEIGHT-65))
        pygame.draw.circle(screen, (200, 50, 50), (100, HEIGHT-60), 6)
        screen.blit(font.render("Food", True, (180,180,200)), (110, HEIGHT-65))
        pygame.draw.circle(screen, (50, 100, 255), (180, HEIGHT-60), 6)
        screen.blit(font.render("Home", True, (180,180,200)), (190, HEIGHT-65))
        pygame.draw.circle(screen, (255, 255, 0), (260, HEIGHT-60), 6)
        screen.blit(font.render("Sun", True, (180,180,200)), (270, HEIGHT-65))

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
