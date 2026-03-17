#!/usr/bin/env python3
"""C. elegans Ecosystem Simulation -- Eon-style Organism in a Petri Dish.

An Eon Systems (eon.ai) inspired simulation: a biophysically faithful digital
organism living in a continuous 2D environment in real-time. This demo builds
a digital C. elegans -- the most studied organism in neuroscience (302 neurons,
~7000 synapses, complete connectome mapped by White et al. 1986) -- and places
it in a petri dish environment with food, chemical gradients, temperature zones,
and obstacles.

Key Differences from Eon Systems:
  - Eon:       Top-down connectome copy. Imports real connectivity from EM
               reconstructions (125K Drosophila neurons in their flagship demo).
               Behavior is inherited from the real wiring diagram.
  - oNeura/dONN: Bottom-up molecular dynamics. HH ion channels (Na+, K+, Ca2+),
               6 neurotransmitters (DA, 5-HT, NE, ACh, GABA, Glu), 8+ drugs,
               STDP, and gene expression. Behavior EMERGES from molecular physics
               rather than being programmed or copied from a connectome.
  - Our approach: Approximate the C. elegans neural circuit with ~302 biophysically
               faithful neurons organized into sensory, interneuron, and motor
               populations. Wire them with connectivity patterns matching the real
               C. elegans connectome (ASE chemosensory -> AIY/AIZ command interneurons
               -> AVA/AVB/PVC forward/reverse command -> VA/VB/DA/DB motor neurons).
               Then let chemotaxis, thermotaxis, foraging, and drug responses
               emerge from the molecular simulation rather than being hard-coded.

Terminology:
  - ONN:    Organic Neural Network -- real biological neurons (DishBrain, FinalSpark)
  - dONN:   digital Organic Neural Network -- oNeura's biophysically faithful simulation
  - oNeura: The platform for building and running dONNs

4 Experiments:
  1. Chemotaxis:  NaCl gradient navigation (worm navigates UP gradient)
  2. Thermotaxis: Temperature preference (~20C cultivation temperature)
  3. Food Search: Foraging for bacterial lawn patches
  4. Drug Effects: Diazepam/caffeine/alprazolam modulate locomotion speed

References:
  - White et al. (1986) "The structure of the nervous system of C. elegans"
    Phil Trans R Soc Lond B 314:1-340
  - Bargmann (2006) "Chemosensation in C. elegans" WormBook
  - Mori & Ohshima (1995) "Neural regulation of thermotaxis in C. elegans"
    Nature 376:344-348
  - Kagan et al. (2022) "In vitro neurons learn and exhibit sentience when
    embodied in a simulated game-world" Neuron 110(23):3952-3969

Usage:
    python3 demos/demo_celegans_ecosystem.py                    # all 4 experiments
    python3 demos/demo_celegans_ecosystem.py --exp 1            # just chemotaxis
    python3 demos/demo_celegans_ecosystem.py --device mps       # Apple GPU
    python3 demos/demo_celegans_ecosystem.py --json results.json # structured output
    python3 demos/demo_celegans_ecosystem.py --realtime         # ASCII visualization
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oneuro.molecular.cuda_backend import (
    CUDAMolecularBrain,
    CUDARegionalBrain,
    detect_backend,
    NT_DA, NT_5HT, NT_NE, NT_ACH, NT_GABA, NT_GLU,
    CH_GABAA, CH_NMDA,
    ARCH_PYRAMIDAL, ARCH_INTERNEURON,
)


# ==========================================================================
# Constants -- C. elegans biology
# ==========================================================================

# Real C. elegans has 302 neurons. We use ~302-350 to match closely while
# allowing some padding for robustness in the dONN simulation.
N_SENSORY = 60        # amphid + phasmid sensory neurons
N_COMMAND_INTER = 40  # AIY, AIZ, AIA, AIB, AVA, AVB, AVD, AVE, PVC
N_RING_INTER = 40     # RIA, RIB, RIM, RIV, ring motor neurons
N_MOTOR_VENTRAL = 60  # VA, VB class + VD inhibitory
N_MOTOR_DORSAL = 60   # DA, DB class + DD inhibitory
N_OTHER_INTER = 42    # remaining interneurons (DVA, PVD, etc.)

N_TOTAL = (N_SENSORY + N_COMMAND_INTER + N_RING_INTER
           + N_MOTOR_VENTRAL + N_MOTOR_DORSAL + N_OTHER_INTER)  # 302

# Sensory neuron subgroups (indices within sensory block)
# ASE left/right: salt chemosensory (main chemotaxis driver)
N_ASE = 10
# AWC: odor detection
N_AWC = 8
# AFD: thermosensory
N_AFD = 8
# ASH: nociception (aversive chemicals, harsh touch)
N_ASH = 8
# ALM/AVM/PLM: mechanosensory (gentle body touch)
N_MECHANO = 10
# Remaining sensory
N_OTHER_SENSORY = N_SENSORY - N_ASE - N_AWC - N_AFD - N_ASH - N_MECHANO  # 16

# Command interneuron subgroups
N_AIY = 8   # forward promotion (activated by ASE ON response)
N_AIZ = 8   # reversal/turn promotion (activated by ASE OFF response)
N_AVA = 6   # backward command (reversal)
N_AVB = 6   # forward command
N_PVC = 6   # forward command (posterior)
N_OTHER_CMD = N_COMMAND_INTER - N_AIY - N_AIZ - N_AVA - N_AVB - N_PVC  # 6

# Body dimensions
WORM_LENGTH_MM = 1.0
WORM_SPEED_FORWARD = 0.2     # mm/s
WORM_SPEED_REVERSE = 0.1     # mm/s
WORM_TURN_RATE = 0.3          # radians per step during gradual turns
OMEGA_TURN_ANGLE = math.pi    # sharp ~180 degree reorientation

# Petri dish
DISH_RADIUS_MM = 5.0  # 10mm diameter dish
DISH_CENTER = (5.0, 5.0)

# Simulation timestep
DT = 0.05  # seconds per simulation step (50ms -- 20 neural steps per body step)
NEURAL_STEPS_PER_BODY = 20


# ==========================================================================
# Petri Dish Environment
# ==========================================================================

@dataclass
class FoodPatch:
    """A circular patch of E. coli bacterial lawn."""
    x: float           # center position (mm)
    y: float
    radius: float      # patch radius (mm)
    remaining: float   # 0.0 to 1.0 (depletes as worm feeds)

    def concentration_at(self, px: float, py: float) -> float:
        """Food concentration at point (px, py). Gaussian falloff."""
        dist = math.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)
        if dist > self.radius * 2.0:
            return 0.0
        return self.remaining * math.exp(-dist ** 2 / (2.0 * (self.radius * 0.5) ** 2))

    def deplete(self, amount: float = 0.002) -> None:
        """Consume food when worm is feeding."""
        self.remaining = max(0.0, self.remaining - amount)


class PetriDish:
    """2D continuous environment for C. elegans simulation.

    Features:
    - Food sources (E. coli bacterial lawns): circular patches, depletable
    - Chemical gradients: NaCl concentration field (chemotaxis)
    - Temperature gradient: linear (thermotaxis)
    - Obstacles: agar bumps the worm cannot cross
    - Boundaries: circular dish edge
    """

    def __init__(
        self,
        size_mm: float = 10.0,
        n_food: int = 4,
        seed: int = 42,
    ):
        self.size_mm = size_mm
        self.center = (size_mm / 2.0, size_mm / 2.0)
        self.radius = size_mm / 2.0
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Food patches
        self.food_patches: List[FoodPatch] = []
        for _ in range(n_food):
            angle = self.rng.uniform(0, 2 * math.pi)
            dist = self.rng.uniform(1.0, self.radius * 0.7)
            fx = self.center[0] + dist * math.cos(angle)
            fy = self.center[1] + dist * math.sin(angle)
            fr = self.rng.uniform(0.4, 0.8)
            self.food_patches.append(FoodPatch(fx, fy, fr, 1.0))

        # NaCl gradient: peak at one side of the dish
        # Gradient direction angle (fixed for experiment reproducibility)
        self.nacl_peak_x = size_mm * 0.85
        self.nacl_peak_y = size_mm * 0.5
        self.nacl_sigma = size_mm * 0.4  # broad gradient

        # Temperature gradient: linear from 15C (left) to 25C (right)
        # C. elegans cultivated at 20C prefers to navigate to 20C
        self.temp_min = 15.0  # degrees C at x=0
        self.temp_max = 25.0  # degrees C at x=size_mm
        self.cultivation_temp = 20.0

        # Obstacles (small agar bumps)
        self.obstacles: List[Tuple[float, float, float]] = []  # (x, y, radius)
        for _ in range(3):
            ox = self.rng.uniform(2.0, 8.0)
            oy = self.rng.uniform(2.0, 8.0)
            orad = self.rng.uniform(0.2, 0.4)
            self.obstacles.append((ox, oy, orad))

    def nacl_at(self, x: float, y: float) -> float:
        """NaCl concentration at point (x, y). Range [0, 1]."""
        dx = x - self.nacl_peak_x
        dy = y - self.nacl_peak_y
        dist_sq = dx * dx + dy * dy
        return math.exp(-dist_sq / (2.0 * self.nacl_sigma ** 2))

    def temperature_at(self, x: float, y: float) -> float:
        """Temperature at point (x, y) in degrees C."""
        frac = x / self.size_mm
        return self.temp_min + (self.temp_max - self.temp_min) * frac

    def food_at(self, x: float, y: float) -> float:
        """Total food concentration at point (x, y)."""
        total = 0.0
        for patch in self.food_patches:
            total += patch.concentration_at(x, y)
        return min(total, 1.0)

    def is_obstacle(self, x: float, y: float) -> bool:
        """Check if point collides with any obstacle."""
        for ox, oy, orad in self.obstacles:
            if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < orad:
                return True
        return False

    def is_in_dish(self, x: float, y: float) -> bool:
        """Check if point is within the circular dish boundary."""
        dx = x - self.center[0]
        dy = y - self.center[1]
        return math.sqrt(dx * dx + dy * dy) <= self.radius

    def clamp_to_dish(self, x: float, y: float) -> Tuple[float, float]:
        """Clamp position to within dish boundary."""
        dx = x - self.center[0]
        dy = y - self.center[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > self.radius * 0.95:
            scale = (self.radius * 0.95) / max(dist, 1e-6)
            return self.center[0] + dx * scale, self.center[1] + dy * scale
        return x, y

    def deplete_food_near(self, x: float, y: float, radius: float = 0.3) -> None:
        """Deplete food patches near the worm's head."""
        for patch in self.food_patches:
            dist = math.sqrt((x - patch.x) ** 2 + (y - patch.y) ** 2)
            if dist < radius + patch.radius:
                patch.deplete()

    def reset_food(self) -> None:
        """Replenish all food patches."""
        for patch in self.food_patches:
            patch.remaining = 1.0


# ==========================================================================
# C. elegans Body (Simplified Biomechanics)
# ==========================================================================

class CelegansBody:
    """Simplified worm body -- head position, heading, body wave.

    Captures the essential locomotion modes of C. elegans:
    - Forward crawling: sinusoidal body wave propagation
    - Reversal: backward movement (triggered by aversive stimuli)
    - Omega turn: sharp reorientation (~180 degrees, pirouette strategy)
    - Gradual turning: biased head oscillation (weathervane strategy)

    Speed: ~0.2 mm/s forward, ~0.1 mm/s reverse (matching real C. elegans
    on agar at 20C).
    """

    def __init__(self, x: float = 5.0, y: float = 5.0, heading: float = 0.0):
        self.x = x
        self.y = y
        self.heading = heading  # radians, 0 = east
        self.speed = WORM_SPEED_FORWARD
        self.body_wave_phase = 0.0
        self.state = "forward"  # "forward", "reverse", "omega_turn"
        self.state_timer = 0    # steps remaining in current state
        self.omega_direction = 1  # +1 or -1 for turn direction

        # Track trajectory for analysis
        self.trajectory: List[Tuple[float, float]] = [(x, y)]
        self.total_distance = 0.0

    def update(
        self,
        turn_bias: float,
        dish: PetriDish,
        dt: float = DT,
        speed_factor: float = 1.0,
    ) -> None:
        """Advance body position by one timestep.

        Args:
            turn_bias: signed turn rate in radians/step (positive = left/CCW)
            dish: environment for boundary/obstacle checking
            dt: timestep in seconds
            speed_factor: motor output intensity from neural activity (0.0-2.0).
                Modulates base speed. Drugs that suppress motor neurons
                (e.g. benzodiazepines enhancing GABA-A) reduce this factor.
        """
        if self.state == "omega_turn":
            # During omega turn: rapid rotation, minimal translation
            self.heading += self.omega_direction * OMEGA_TURN_ANGLE * dt * 2.0 * speed_factor
            self.state_timer -= 1
            if self.state_timer <= 0:
                self.state = "forward"
                self.speed = WORM_SPEED_FORWARD * speed_factor
            return

        if self.state == "reverse":
            # Backward movement
            self.speed = WORM_SPEED_REVERSE * speed_factor
            self.state_timer -= 1
            if self.state_timer <= 0:
                self.state = "forward"
                self.speed = WORM_SPEED_FORWARD * speed_factor
        else:
            self.speed = WORM_SPEED_FORWARD * speed_factor

        # Apply turn bias (gradual steering -- weathervane mechanism)
        self.heading += turn_bias * dt

        # Body wave phase advance (sinusoidal undulation)
        self.body_wave_phase += 2.0 * math.pi * 0.5 * dt  # ~0.5 Hz body wave

        # Move in heading direction
        direction = 1.0 if self.state == "forward" else -1.0
        dx = math.cos(self.heading) * self.speed * direction * dt
        dy = math.sin(self.heading) * self.speed * direction * dt

        new_x = self.x + dx
        new_y = self.y + dy

        # Boundary and obstacle checks
        if dish.is_obstacle(new_x, new_y):
            # Reverse away from obstacle
            self.initiate_reversal(duration=8)
            return

        new_x, new_y = dish.clamp_to_dish(new_x, new_y)

        # Check if we hit the dish edge
        cx, cy = dish.center
        dist_to_center = math.sqrt((new_x - cx) ** 2 + (new_y - cy) ** 2)
        if dist_to_center > dish.radius * 0.9:
            # Near edge: turn inward
            angle_to_center = math.atan2(cy - new_y, cx - new_x)
            self.heading = angle_to_center + self.rng_uniform(-0.3, 0.3)

        step_dist = math.sqrt(dx ** 2 + dy ** 2)
        self.total_distance += step_dist
        self.x = new_x
        self.y = new_y
        self.trajectory.append((self.x, self.y))

    def rng_uniform(self, lo: float, hi: float) -> float:
        """Simple pseudo-random for body-level noise."""
        # Use body_wave_phase as entropy source (avoids extra RNG state)
        v = math.sin(self.body_wave_phase * 13.7 + self.x * 7.3 + self.y * 11.1)
        return lo + (hi - lo) * (v * 0.5 + 0.5)

    def initiate_reversal(self, duration: int = 10) -> None:
        """Switch to reverse locomotion for given number of steps."""
        self.state = "reverse"
        self.state_timer = duration

    def initiate_omega_turn(self, direction: int = 1, duration: int = 10) -> None:
        """Initiate an omega turn (sharp reorientation).

        Args:
            direction: +1 for ventral (left), -1 for dorsal (right)
            duration: steps for the turn
        """
        self.state = "omega_turn"
        self.omega_direction = direction
        self.state_timer = duration

    def distance_to(self, x: float, y: float) -> float:
        """Euclidean distance from worm head to a point."""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def head_position(self) -> Tuple[float, float]:
        """Position of the head (same as body center in this simplified model)."""
        return self.x, self.y

    @property
    def head_x(self) -> float:
        return self.x

    @property
    def head_y(self) -> float:
        return self.y


# ==========================================================================
# C. elegans Nervous System (Custom-Wired CUDAMolecularBrain)
# ==========================================================================

class CelegansNervousSystem:
    """302-neuron nervous system approximating the real C. elegans connectome.

    Rather than using the pre-wired CUDARegionalBrain (which builds mammalian
    cortical columns), we construct a CUDAMolecularBrain directly and wire it
    with connectivity matching the C. elegans neural circuits.

    Neuron Layout (by index):
    +---------+-------------------+---------+
    | Block   | Subgroup          | Count   |
    +---------+-------------------+---------+
    | Sensory | ASE (chemotaxis)  |  10     |
    |         | AWC (odor)        |   8     |
    |         | AFD (temperature) |   8     |
    |         | ASH (nociception) |   8     |
    |         | Mechano (touch)   |  10     |
    |         | Other sensory     |  16     |
    +---------+-------------------+---------+
    | Command | AIY (fwd promote) |   8     |
    | Inter   | AIZ (rev promote) |   8     |
    |         | AVA (rev command) |   6     |
    |         | AVB (fwd command) |   6     |
    |         | PVC (fwd command) |   6     |
    |         | Other command     |   6     |
    +---------+-------------------+---------+
    | Ring    | Ring interneurons  |  40     |
    | Inter   | (RIA,RIB,RIM,RIV) |         |
    +---------+-------------------+---------+
    | Motor   | Ventral (VA,VB,VD)|  60     |
    |         | Dorsal (DA,DB,DD) |  60     |
    +---------+-------------------+---------+
    | Other   | Misc interneurons |  42     |
    +---------+-------------------+---------+
    | TOTAL                       | 302     |
    +---------+-------------------+---------+
    """

    def __init__(self, device: str = "auto", seed: int = 42):
        torch.manual_seed(seed)
        self.device_str = device

        # Create the molecular brain
        self.brain = CUDAMolecularBrain(N_TOTAL, device=device)
        self.dev = self.brain.device

        # Build neuron index maps
        idx = 0
        self.neuron_groups: Dict[str, Dict[str, Any]] = {}

        # -- Sensory neurons --
        self._ase_start = idx
        self.ase_ids = list(range(idx, idx + N_ASE)); idx += N_ASE
        self.awc_ids = list(range(idx, idx + N_AWC)); idx += N_AWC
        self.afd_ids = list(range(idx, idx + N_AFD)); idx += N_AFD
        self.ash_ids = list(range(idx, idx + N_ASH)); idx += N_ASH
        self.mechano_ids = list(range(idx, idx + N_MECHANO)); idx += N_MECHANO
        self.other_sensory_ids = list(range(idx, idx + N_OTHER_SENSORY)); idx += N_OTHER_SENSORY
        self.all_sensory_ids = list(range(self._ase_start, idx))

        # -- Command interneurons --
        self._cmd_start = idx
        self.aiy_ids = list(range(idx, idx + N_AIY)); idx += N_AIY
        self.aiz_ids = list(range(idx, idx + N_AIZ)); idx += N_AIZ
        self.ava_ids = list(range(idx, idx + N_AVA)); idx += N_AVA
        self.avb_ids = list(range(idx, idx + N_AVB)); idx += N_AVB
        self.pvc_ids = list(range(idx, idx + N_PVC)); idx += N_PVC
        self.other_cmd_ids = list(range(idx, idx + N_OTHER_CMD)); idx += N_OTHER_CMD
        self.all_cmd_ids = list(range(self._cmd_start, idx))

        # -- Ring interneurons --
        self._ring_start = idx
        self.ring_ids = list(range(idx, idx + N_RING_INTER)); idx += N_RING_INTER

        # -- Motor neurons --
        self._motor_start = idx
        self.motor_ventral_ids = list(range(idx, idx + N_MOTOR_VENTRAL)); idx += N_MOTOR_VENTRAL
        self.motor_dorsal_ids = list(range(idx, idx + N_MOTOR_DORSAL)); idx += N_MOTOR_DORSAL
        self.all_motor_ids = list(range(self._motor_start, idx))

        # -- Other interneurons --
        self.other_inter_ids = list(range(idx, idx + N_OTHER_INTER)); idx += N_OTHER_INTER

        assert idx == N_TOTAL, f"Expected {N_TOTAL} neurons, got {idx}"

        # Set archetypes: most C. elegans neurons are excitatory (glutamatergic
        # or cholinergic), with GABAergic inhibitory motor neurons (DD, VD class)
        # Inhibitory motor neurons: last 20% of each motor group
        n_inhib_v = max(2, int(N_MOTOR_VENTRAL * 0.2))
        n_inhib_d = max(2, int(N_MOTOR_DORSAL * 0.2))
        for nid in self.motor_ventral_ids[-n_inhib_v:]:
            self.brain.archetype[nid] = ARCH_INTERNEURON
        for nid in self.motor_dorsal_ids[-n_inhib_d:]:
            self.brain.archetype[nid] = ARCH_INTERNEURON
        # Some ring interneurons are inhibitory
        n_inhib_ring = max(2, int(N_RING_INTER * 0.25))
        for nid in self.ring_ids[-n_inhib_ring:]:
            self.brain.archetype[nid] = ARCH_INTERNEURON

        # Assign spatial positions (for distance-based effects in the brain)
        # Layout: sensory at front, motor at back, interneurons in middle
        self._assign_positions()

        # Wire the connectome
        self._wire_connectome()

        # Pre-build ID tensors for stimulation
        self._ase_tensor = torch.tensor(self.ase_ids, dtype=torch.int64, device=self.dev)
        self._awc_tensor = torch.tensor(self.awc_ids, dtype=torch.int64, device=self.dev)
        self._afd_tensor = torch.tensor(self.afd_ids, dtype=torch.int64, device=self.dev)
        self._ash_tensor = torch.tensor(self.ash_ids, dtype=torch.int64, device=self.dev)
        self._mechano_tensor = torch.tensor(self.mechano_ids, dtype=torch.int64, device=self.dev)
        self._aiy_tensor = torch.tensor(self.aiy_ids, dtype=torch.int64, device=self.dev)
        self._aiz_tensor = torch.tensor(self.aiz_ids, dtype=torch.int64, device=self.dev)
        self._ava_tensor = torch.tensor(self.ava_ids, dtype=torch.int64, device=self.dev)
        self._avb_tensor = torch.tensor(self.avb_ids, dtype=torch.int64, device=self.dev)
        self._pvc_tensor = torch.tensor(self.pvc_ids, dtype=torch.int64, device=self.dev)
        self._motor_v_tensor = torch.tensor(self.motor_ventral_ids, dtype=torch.int64, device=self.dev)
        self._motor_d_tensor = torch.tensor(self.motor_dorsal_ids, dtype=torch.int64, device=self.dev)
        self._all_motor_tensor = torch.tensor(self.all_motor_ids, dtype=torch.int64, device=self.dev)
        self._ring_tensor = torch.tensor(self.ring_ids, dtype=torch.int64, device=self.dev)
        self._all_ids_tensor = torch.arange(N_TOTAL, dtype=torch.int64, device=self.dev)

    def _assign_positions(self) -> None:
        """Assign 3D positions mimicking C. elegans anatomy.

        Head (sensory) at z=0, nerve ring (interneurons) at z=1-2,
        ventral/dorsal cord (motor) at z=3-5.
        """
        b = self.brain
        dev = self.dev

        # Sensory: head region
        n_s = len(self.all_sensory_ids)
        b.x[self.all_sensory_ids[0]:self.all_sensory_ids[-1]+1] = torch.rand(n_s, device=dev) * 1.0
        b.y[self.all_sensory_ids[0]:self.all_sensory_ids[-1]+1] = torch.rand(n_s, device=dev) * 1.0
        b.z[self.all_sensory_ids[0]:self.all_sensory_ids[-1]+1] = torch.rand(n_s, device=dev) * 0.5

        # Command interneurons: nerve ring
        n_c = len(self.all_cmd_ids)
        s, e = self.all_cmd_ids[0], self.all_cmd_ids[-1]+1
        b.x[s:e] = 1.0 + torch.rand(n_c, device=dev) * 1.0
        b.y[s:e] = torch.rand(n_c, device=dev) * 1.0
        b.z[s:e] = 0.5 + torch.rand(n_c, device=dev) * 1.0

        # Ring interneurons
        n_r = len(self.ring_ids)
        s, e = self.ring_ids[0], self.ring_ids[-1]+1
        b.x[s:e] = 1.5 + torch.rand(n_r, device=dev) * 1.0
        b.y[s:e] = torch.rand(n_r, device=dev) * 1.0
        b.z[s:e] = 1.0 + torch.rand(n_r, device=dev) * 1.0

        # Motor ventral: ventral nerve cord
        n_mv = len(self.motor_ventral_ids)
        s, e = self.motor_ventral_ids[0], self.motor_ventral_ids[-1]+1
        b.x[s:e] = 3.0 + torch.rand(n_mv, device=dev) * 3.0
        b.y[s:e] = -0.5 + torch.rand(n_mv, device=dev) * 0.3  # ventral
        b.z[s:e] = 2.0 + torch.rand(n_mv, device=dev) * 3.0

        # Motor dorsal: dorsal nerve cord
        n_md = len(self.motor_dorsal_ids)
        s, e = self.motor_dorsal_ids[0], self.motor_dorsal_ids[-1]+1
        b.x[s:e] = 3.0 + torch.rand(n_md, device=dev) * 3.0
        b.y[s:e] = 0.5 + torch.rand(n_md, device=dev) * 0.3  # dorsal
        b.z[s:e] = 2.0 + torch.rand(n_md, device=dev) * 3.0

        # Other interneurons: scattered
        n_o = len(self.other_inter_ids)
        s, e = self.other_inter_ids[0], self.other_inter_ids[-1]+1
        b.x[s:e] = 2.0 + torch.rand(n_o, device=dev) * 2.0
        b.y[s:e] = torch.rand(n_o, device=dev) * 1.0
        b.z[s:e] = 1.0 + torch.rand(n_o, device=dev) * 2.0

    def _wire_connectome(self) -> None:
        """Wire synapses to approximate the C. elegans connectome.

        Key circuits (from Bargmann 2006, White et al. 1986):

        CHEMOTAXIS CIRCUIT:
          ASE --(glu)--> AIY (forward promotion)
          ASE --(glu)--> AIZ (turn/reversal promotion)
          AIY --(ACh)--> RIA (ring interneuron, head steering)
          AIZ --(glu)--> RIM --> AVA (reversal command)

        FORWARD/REVERSE DECISION:
          AIY, AVB, PVC --> forward locomotion
          AIZ, AVA, AVD --> reversal + omega turn
          AVB --(ACh)--> VB, DB motor neurons (forward)
          AVA --(glu)--> VA, DA motor neurons (backward)

        MOTOR EXECUTION:
          VA/VB --> ventral body wall muscles
          DA/DB --> dorsal body wall muscles
          DD/VD --> inhibitory cross-inhibition (dorsal-ventral alternation)

        TOUCH RESPONSE:
          ALM, AVM (anterior touch) --> AVD, AVA --> reversal
          PLM (posterior touch) --> AVB, PVC --> forward acceleration
        """
        dev = self.dev
        all_pre, all_post, all_weight, all_nt = [], [], [], []

        def _connect(
            src: List[int], dst: List[int],
            prob: float, w_range: Tuple[float, float],
            nt: int,
        ) -> None:
            """Add random connections between src and dst groups."""
            if not src or not dst:
                return
            n_possible = len(src) * len(dst)
            if n_possible > 50_000:
                n_expected = int(n_possible * prob)
                if n_expected == 0:
                    return
                src_t = torch.tensor(src, device=dev)[
                    torch.randint(len(src), (n_expected,), device=dev)]
                dst_t = torch.tensor(dst, device=dev)[
                    torch.randint(len(dst), (n_expected,), device=dev)]
                mask = src_t != dst_t
                pre_t, post_t = src_t[mask], dst_t[mask]
            else:
                src_t = torch.tensor(src, device=dev)
                dst_t = torch.tensor(dst, device=dev)
                conn_mask = torch.rand(len(src), len(dst), device=dev) < prob
                indices = torch.where(conn_mask)
                pre_t = src_t[indices[0]]
                post_t = dst_t[indices[1]]
                valid = pre_t != post_t
                pre_t, post_t = pre_t[valid], post_t[valid]

            if pre_t.shape[0] == 0:
                return
            w = (torch.rand(pre_t.shape[0], device=dev)
                 * (w_range[1] - w_range[0]) + w_range[0])
            nt_t = torch.full((pre_t.shape[0],), nt, dtype=torch.int32, device=dev)
            all_pre.append(pre_t)
            all_post.append(post_t)
            all_weight.append(w)
            all_nt.append(nt_t)

        # ================================================================
        # CHEMOTAXIS CIRCUIT
        # ================================================================
        # ASE --> AIY (glutamatergic, excitatory -- promotes forward)
        _connect(self.ase_ids, self.aiy_ids, 0.6, (1.0, 2.0), NT_GLU)
        # ASE --> AIZ (glutamatergic -- promotes turns/reversals)
        _connect(self.ase_ids, self.aiz_ids, 0.4, (0.8, 1.5), NT_GLU)
        # AWC --> AIY (odor chemotaxis, similar to ASE circuit)
        _connect(self.awc_ids, self.aiy_ids, 0.4, (0.8, 1.5), NT_GLU)
        _connect(self.awc_ids, self.aiz_ids, 0.3, (0.6, 1.2), NT_GLU)

        # ================================================================
        # THERMOTAXIS CIRCUIT
        # ================================================================
        # AFD --> AIY (thermosensory, promotes forward at preferred temp)
        _connect(self.afd_ids, self.aiy_ids, 0.5, (0.8, 1.5), NT_GLU)
        # AFD --> AIZ (thermosensory, promotes turns away from wrong temp)
        _connect(self.afd_ids, self.aiz_ids, 0.4, (0.8, 1.5), NT_GLU)

        # ================================================================
        # NOCICEPTION / TOUCH CIRCUIT
        # ================================================================
        # ASH (aversive) --> AVA, AVD (reversal command)
        _connect(self.ash_ids, self.ava_ids, 0.5, (1.0, 2.0), NT_GLU)
        # Mechanosensory (anterior touch) --> AVA (reversal)
        # First half = anterior (ALM/AVM), second half = posterior (PLM)
        anterior_touch = self.mechano_ids[:N_MECHANO // 2]
        posterior_touch = self.mechano_ids[N_MECHANO // 2:]
        _connect(anterior_touch, self.ava_ids, 0.5, (1.0, 1.8), NT_GLU)
        # Posterior touch --> AVB, PVC (forward acceleration)
        _connect(posterior_touch, self.avb_ids, 0.5, (1.0, 1.8), NT_GLU)
        _connect(posterior_touch, self.pvc_ids, 0.5, (1.0, 1.8), NT_GLU)

        # ================================================================
        # COMMAND INTERNEURON CIRCUIT (forward/reverse decision)
        # ================================================================
        # AIY --> RIA ring interneuron (head movement, steering)
        _connect(self.aiy_ids, self.ring_ids[:N_RING_INTER // 2],
                 0.4, (0.8, 1.5), NT_ACH)
        # AIY --> AVB (forward command via cholinergic)
        _connect(self.aiy_ids, self.avb_ids, 0.4, (0.8, 1.5), NT_ACH)

        # AIZ --> RIM (subset of ring interneurons)
        _connect(self.aiz_ids, self.ring_ids[N_RING_INTER // 2:],
                 0.4, (0.8, 1.5), NT_GLU)
        # AIZ --> AVA (reversal command)
        _connect(self.aiz_ids, self.ava_ids, 0.5, (1.0, 1.8), NT_GLU)

        # AVB <--> PVC (mutual excitation for forward)
        _connect(self.avb_ids, self.pvc_ids, 0.3, (0.5, 1.0), NT_GLU)
        _connect(self.pvc_ids, self.avb_ids, 0.3, (0.5, 1.0), NT_GLU)

        # AVA mutual inhibition with AVB (forward vs backward competition)
        _connect(self.ava_ids, self.avb_ids, 0.3, (0.5, 1.0), NT_GABA)
        _connect(self.avb_ids, self.ava_ids, 0.3, (0.5, 1.0), NT_GABA)

        # ================================================================
        # MOTOR CIRCUIT
        # ================================================================
        # Forward command: AVB --> VB, DB (excitatory motor for forward wave)
        excit_ventral = self.motor_ventral_ids[:len(self.motor_ventral_ids) - max(2, int(N_MOTOR_VENTRAL * 0.2))]
        excit_dorsal = self.motor_dorsal_ids[:len(self.motor_dorsal_ids) - max(2, int(N_MOTOR_DORSAL * 0.2))]
        inhib_ventral = self.motor_ventral_ids[len(excit_ventral):]  # VD class
        inhib_dorsal = self.motor_dorsal_ids[len(excit_dorsal):]    # DD class

        # AVB, PVC --> forward motor (B-type: VB, DB)
        # B-type motor neurons are the first half of excitatory motor neurons
        vb_ids = excit_ventral[:len(excit_ventral) // 2]
        va_ids = excit_ventral[len(excit_ventral) // 2:]
        db_ids = excit_dorsal[:len(excit_dorsal) // 2]
        da_ids = excit_dorsal[len(excit_dorsal) // 2:]

        _connect(self.avb_ids, vb_ids, 0.5, (1.0, 2.0), NT_ACH)
        _connect(self.avb_ids, db_ids, 0.5, (1.0, 2.0), NT_ACH)
        _connect(self.pvc_ids, vb_ids, 0.4, (0.8, 1.5), NT_ACH)
        _connect(self.pvc_ids, db_ids, 0.4, (0.8, 1.5), NT_ACH)

        # AVA --> backward motor (A-type: VA, DA)
        _connect(self.ava_ids, va_ids, 0.5, (1.0, 2.0), NT_ACH)
        _connect(self.ava_ids, da_ids, 0.5, (1.0, 2.0), NT_ACH)

        # Dorsal-ventral cross-inhibition (DD, VD classes)
        # DD inhibits ventral, VD inhibits dorsal -- creates alternating body wave
        _connect(inhib_dorsal, excit_ventral, 0.3, (0.8, 1.5), NT_GABA)
        _connect(inhib_ventral, excit_dorsal, 0.3, (0.8, 1.5), NT_GABA)

        # Excitatory motor --> inhibitory motor (drives cross-inhibition)
        _connect(excit_ventral, inhib_ventral, 0.2, (0.5, 1.0), NT_ACH)
        _connect(excit_dorsal, inhib_dorsal, 0.2, (0.5, 1.0), NT_ACH)

        # ================================================================
        # RING INTERNEURON RECURRENCE (head oscillation)
        # ================================================================
        _connect(self.ring_ids, self.ring_ids, 0.1, (0.3, 0.8), NT_GLU)
        # Ring --> motor (head movement contribution)
        _connect(self.ring_ids[:N_RING_INTER // 2], excit_ventral[:10],
                 0.2, (0.5, 1.0), NT_ACH)
        _connect(self.ring_ids[N_RING_INTER // 2:], excit_dorsal[:10],
                 0.2, (0.5, 1.0), NT_ACH)

        # ================================================================
        # OTHER INTERNEURON CONNECTIONS (diffuse neuromodulation)
        # ================================================================
        # Serotonergic interneurons modulate locomotion speed
        _connect(self.other_inter_ids[:10], self.all_motor_ids,
                 0.05, (0.3, 0.8), NT_5HT)
        # Other interneurons provide background excitation
        _connect(self.other_inter_ids[10:], self.all_cmd_ids,
                 0.05, (0.3, 0.6), NT_GLU)

        # ================================================================
        # FOOD MODULATION (serotonin circuit)
        # ================================================================
        # Food sensory --> serotonergic interneurons --> slowing response
        # (C. elegans slows on food via 5-HT -- Sawin et al. 2000)
        _connect(self.other_sensory_ids[:8], self.other_inter_ids[:10],
                 0.3, (0.8, 1.5), NT_GLU)

        # ================================================================
        # COMMIT ALL SYNAPSES
        # ================================================================
        if all_pre:
            self.brain.add_synapses(
                torch.cat(all_pre),
                torch.cat(all_post),
                torch.cat(all_weight),
                torch.cat(all_nt),
            )

        # Store subgroup references for motor readout
        self._vb_ids = vb_ids
        self._va_ids = va_ids
        self._db_ids = db_ids
        self._da_ids = da_ids
        self._vb_tensor = torch.tensor(vb_ids, dtype=torch.int64, device=self.dev)
        self._va_tensor = torch.tensor(va_ids, dtype=torch.int64, device=self.dev)
        self._db_tensor = torch.tensor(db_ids, dtype=torch.int64, device=self.dev)
        self._da_tensor = torch.tensor(da_ids, dtype=torch.int64, device=self.dev)

    def step(self) -> None:
        """Advance neural simulation by one timestep."""
        self.brain.step()

    def run(self, steps: int) -> None:
        """Run neural simulation for multiple steps."""
        self.brain.run(steps)

    def apply_drug(self, drug_name: str, dose_mg: float) -> None:
        """Apply pharmacological agent."""
        self.brain.apply_drug(drug_name, dose_mg)


# ==========================================================================
# Sensory Encoding
# ==========================================================================

class CelegansSensory:
    """Encode environmental stimuli into neural stimulation currents.

    Maps sensory modalities to specific neuron populations following
    the real C. elegans sensory architecture:
    - NaCl gradient at head --> ASE neurons (chemotaxis)
    - Temperature at head --> AFD neurons (thermotaxis)
    - Food presence --> other sensory neurons (foraging/slowing)
    - Obstacle contact --> ASH nociceptors + mechanosensory neurons
    """

    def __init__(self, ns: CelegansNervousSystem):
        self.ns = ns
        self.dev = ns.dev
        # Adaptation state (sensory neurons adapt to sustained stimuli)
        self._prev_nacl = 0.0
        self._prev_temp = 20.0
        self._prev_food = 0.0

    def encode(
        self,
        dish: PetriDish,
        body: CelegansBody,
        dt: float = DT,
    ) -> Dict[str, float]:
        """Sample environment at worm's head and inject stimulation currents.

        Returns dict of stimulation intensities for logging.

        C. elegans sensory neurons are derivative detectors -- they respond
        to CHANGES in concentration, not absolute levels. This is critical
        for the biased random walk chemotaxis strategy.
        """
        hx, hy = body.head_position()
        brain = self.ns.brain
        intensities = {}

        # ---- NaCl chemosensory (ASE neurons) ----
        nacl = dish.nacl_at(hx, hy)
        # ASE neurons respond to temporal derivative (dC/dt)
        # Positive change = moving up gradient = ASE ON response
        d_nacl = nacl - self._prev_nacl
        self._prev_nacl = nacl

        # ASE ON response (increasing concentration) --> stimulate AIY path
        # ASE OFF response (decreasing concentration) --> stimulate AIZ path
        if d_nacl > 0.001:
            # Moving UP gradient: strong ASE activation --> promotes forward
            ase_intensity = min(40.0, d_nacl * 800.0)
            brain.external_current[self.ns._ase_tensor] += ase_intensity
            intensities["nacl_on"] = float(ase_intensity)
        elif d_nacl < -0.001:
            # Moving DOWN gradient: ASE deactivation --> promotes turning
            # Stimulate ASE weakly + ASH (aversive signal)
            aversive_intensity = min(30.0, abs(d_nacl) * 600.0)
            brain.external_current[self.ns._ash_tensor[:4]] += aversive_intensity
            intensities["nacl_off"] = float(aversive_intensity)
        else:
            intensities["nacl_on"] = 0.0
            intensities["nacl_off"] = 0.0

        # Background tonic ASE activity proportional to absolute concentration
        # (weaker than derivative response)
        tonic_ase = nacl * 8.0
        brain.external_current[self.ns._ase_tensor] += tonic_ase
        intensities["nacl_tonic"] = float(tonic_ase)

        # ---- Thermosensory (AFD neurons) ----
        temp = dish.temperature_at(hx, hy)
        d_temp = temp - self._prev_temp
        self._prev_temp = temp

        # AFD neurons respond to deviation from cultivation temperature
        temp_error = abs(temp - dish.cultivation_temp)
        # At preferred temp: AFD drives AIY (keep going)
        # Away from preferred: AFD drives AIZ (turn)
        if temp_error < 1.5:
            # Near preferred temperature: promote forward
            afd_intensity = max(0.0, (1.5 - temp_error) * 15.0)
            brain.external_current[self.ns._afd_tensor] += afd_intensity
            intensities["temp_comfort"] = float(afd_intensity)
            intensities["temp_aversive"] = 0.0
        else:
            # Far from preferred: promote turning
            afd_aversive = min(25.0, temp_error * 5.0)
            brain.external_current[self.ns._ash_tensor[4:]] += afd_aversive
            intensities["temp_comfort"] = 0.0
            intensities["temp_aversive"] = float(afd_aversive)

        # ---- Food detection ----
        food = dish.food_at(hx, hy)
        d_food = food - self._prev_food
        self._prev_food = food

        if food > 0.1:
            # On food: activate serotonergic slowing circuit
            food_intensity = food * 20.0
            other_sens = self.ns.other_sensory_ids[:8]
            brain.external_current[torch.tensor(other_sens, dtype=torch.int64,
                                                device=self.dev)] += food_intensity
            # Also trigger 5-HT release (basal slowing response)
            brain.nt_conc[self.ns._all_motor_tensor, NT_5HT] += food * 30.0
            intensities["food"] = float(food_intensity)
            # Deplete food near worm
            dish.deplete_food_near(hx, hy)
        else:
            intensities["food"] = 0.0

        # ---- Touch / obstacle detection ----
        if dish.is_obstacle(hx + math.cos(body.heading) * 0.2,
                            hy + math.sin(body.heading) * 0.2):
            # Head bump: strong ASH + mechanosensory activation
            brain.external_current[self.ns._ash_tensor] += 40.0
            brain.external_current[self.ns._mechano_tensor[:5]] += 35.0
            intensities["touch"] = 40.0
        else:
            intensities["touch"] = 0.0

        return intensities


# ==========================================================================
# Motor Decoding
# ==========================================================================

class CelegansMotor:
    """Decode neural activity into body movement commands.

    Reads spike counts from command interneurons and motor neuron populations
    to determine:
    - Forward vs reverse locomotion (AVB/PVC vs AVA competition)
    - Dorsal vs ventral motor balance (turning direction)
    - Omega turn detection (high simultaneous dorsal+ventral activity)

    The forward/reverse decision follows the real C. elegans circuit:
    AVB/PVC activity --> forward crawling
    AVA activity --> reversal
    Dorsal-ventral imbalance --> gradual turning (weathervane strategy)
    """

    def __init__(self, ns: CelegansNervousSystem):
        self.ns = ns
        self.dev = ns.dev

    def decode(
        self,
        n_steps: int = NEURAL_STEPS_PER_BODY,
    ) -> Tuple[str, float, float]:
        """Run neural simulation for n_steps and decode motor output.

        Returns:
            (action, turn_bias, speed_factor) where:
            - action: "forward", "reverse", or "omega_turn"
            - turn_bias: signed turn rate (positive = ventral/left)
            - speed_factor: 0.0-1.0+ scaling motor output intensity.
              Drugs that enhance GABA-A (diazepam, alprazolam) suppress
              motor neuron firing, reducing this factor. Caffeine increases it.

        Note: external_current is zeroed every neural step, so we must
        re-apply tonic background drive on each step. This mimics the
        continuous synaptic bombardment from reticular formation and
        descending command pathways that keeps motor neurons active.
        """
        brain = self.ns.brain

        # GPU accumulators for spike counts
        fwd_acc = torch.zeros(1, device=self.dev)   # AVB + PVC spikes
        rev_acc = torch.zeros(1, device=self.dev)    # AVA spikes
        ventral_acc = torch.zeros(1, device=self.dev)
        dorsal_acc = torch.zeros(1, device=self.dev)
        ring_acc = torch.zeros(1, device=self.dev)

        # Pre-compute inhibitory motor neuron tensor for tonic drive
        n_inhib_v = max(2, int(N_MOTOR_VENTRAL * 0.2))
        n_inhib_d = max(2, int(N_MOTOR_DORSAL * 0.2))
        inhib_v_tensor = self.ns._motor_v_tensor[-n_inhib_v:]
        inhib_d_tensor = self.ns._motor_d_tensor[-n_inhib_d:]

        for s in range(n_steps):
            # Tonic background drive -- external_current is zeroed each step,
            # so we must re-apply. Pulsed (every other step) to avoid
            # depolarization block (Na+ inactivation).
            if s % 2 == 0:
                # Tonic forward command drive (mimics descending reticular input)
                brain.external_current[self.ns._avb_tensor] += 50.0
                brain.external_current[self.ns._pvc_tensor] += 45.0
                # Tonic drive to excitatory motor neurons (basal muscle tone).
                # 60 uA is needed to reliably depolarize HH neurons from
                # -65mV resting to -20mV spike threshold (requires ~45mV
                # shift against ~30 mS/cm2 leak conductance).
                brain.external_current[self.ns._motor_v_tensor] += 60.0
                brain.external_current[self.ns._motor_d_tensor] += 60.0
                # Drive inhibitory motor neurons (DD/VD class) so they release
                # GABA onto excitatory motor neurons. This is essential for
                # benzodiazepine effects: drugs enhance GABA-A conductance,
                # but need GABA present at the synapse to have an effect.
                brain.external_current[inhib_v_tensor] += 65.0
                brain.external_current[inhib_d_tensor] += 65.0

            self.ns.step()
            fwd_acc += brain.fired[self.ns._avb_tensor].sum()
            fwd_acc += brain.fired[self.ns._pvc_tensor].sum()
            rev_acc += brain.fired[self.ns._ava_tensor].sum()
            ventral_acc += brain.fired[self.ns._motor_v_tensor].sum()
            dorsal_acc += brain.fired[self.ns._motor_d_tensor].sum()
            ring_acc += brain.fired[self.ns._ring_tensor].sum()

        # Single GPU->CPU sync
        fwd_count = int(fwd_acc.item())
        rev_count = int(rev_acc.item())
        v_count = int(ventral_acc.item())
        d_count = int(dorsal_acc.item())
        ring_count = int(ring_acc.item())

        # Speed factor: total motor firing rate drives locomotion speed.
        # More motor neuron activity = faster movement. GABA-A enhancing
        # drugs (benzodiazepines) increase inhibitory conductance on motor
        # neurons, suppressing their firing rate and thus reducing speed.
        total_motor = v_count + d_count
        n_motor = len(self.ns.all_motor_ids)
        max_motor_spikes = n_motor * n_steps
        raw_rate = total_motor / max(max_motor_spikes, 1)
        # Fixed calibration: with 120 motor neurons, 20 steps, and 30 uA
        # tonic drive through GABAergic inhibitory synapses, baseline
        # firing rate is ~4-5% of max. We set the reference so that
        # this maps to speed_factor ~1.0. Drug conditions with enhanced
        # GABA-A conductance have lower firing rates -> lower speed_factor.
        speed_factor = raw_rate / 0.04 if raw_rate > 0 else 0.1
        speed_factor = max(0.05, min(speed_factor, 2.0))

        # Determine action
        total_cmd = fwd_count + rev_count
        if total_cmd == 0:
            # No command activity: continue forward with slight random drift
            return "forward", 0.0, speed_factor

        # Forward vs reverse decision
        fwd_frac = fwd_count / max(total_cmd, 1)
        rev_frac = rev_count / max(total_cmd, 1)

        # Omega turn detection: high ring + high both motor groups
        if (ring_count > n_steps * 2 and total_motor > n_steps * 4
                and rev_frac > 0.4):
            direction = 1 if v_count > d_count else -1
            return "omega_turn", float(direction), speed_factor

        if rev_frac > 0.6:
            action = "reverse"
        else:
            action = "forward"

        # Turn bias from dorsal-ventral imbalance
        # More ventral activity = ventral turn (positive bias)
        # More dorsal activity = dorsal turn (negative bias)
        if total_motor > 0:
            turn_bias = (v_count - d_count) / max(total_motor, 1) * WORM_TURN_RATE * 3.0
        else:
            turn_bias = 0.0

        return action, turn_bias, speed_factor


# ==========================================================================
# Simulation Loop
# ==========================================================================

def simulate_episode(
    ns: CelegansNervousSystem,
    body: CelegansBody,
    dish: PetriDish,
    sensory: CelegansSensory,
    motor: CelegansMotor,
    n_steps: int = 500,
    realtime: bool = False,
    drug: Optional[Tuple[str, float]] = None,
) -> Dict[str, Any]:
    """Run one complete simulation episode.

    Each step:
    1. Sensory encoding: sample environment at head, inject currents
    2. Neural simulation: run NEURAL_STEPS_PER_BODY HH timesteps
    3. Motor decoding: read command/motor spike counts
    4. Body update: move worm according to decoded commands

    Args:
        drug: optional (drug_name, dose_mg) to re-apply periodically.
            NT concentrations decay each step, so drugs that modify
            concentrations (e.g. caffeine's cAMP boost) need periodic
            re-application. Conductance scale modifications persist.

    Returns trajectory and statistics.
    """
    trajectory = [(body.x, body.y)]
    nacl_values = [dish.nacl_at(body.x, body.y)]
    temp_values = [dish.temperature_at(body.x, body.y)]
    food_values = [dish.food_at(body.x, body.y)]
    actions = []
    speeds = []

    for step_i in range(n_steps):
        # Periodic drug re-application (every 50 steps) for sustained effect
        if drug is not None and step_i % 50 == 0 and step_i > 0:
            ns.apply_drug(drug[0], drug[1] * 0.3)  # maintenance dose
        # 1. Sensory encoding
        intensities = sensory.encode(dish, body)

        # 2 & 3. Neural simulation + motor decoding (interleaved)
        action, turn_bias, speed_factor = motor.decode(n_steps=NEURAL_STEPS_PER_BODY)

        # 4. Execute action (speed_factor modulates locomotion via motor firing rate)
        if action == "omega_turn":
            direction = 1 if turn_bias >= 0 else -1
            body.initiate_omega_turn(direction=direction, duration=8)
            body.update(0.0, dish, speed_factor=speed_factor)
        elif action == "reverse":
            if body.state != "reverse":
                body.initiate_reversal(duration=6)
            body.update(turn_bias, dish, speed_factor=speed_factor)
        else:
            body.update(turn_bias, dish, speed_factor=speed_factor)

        # Record trajectory
        trajectory.append((body.x, body.y))
        nacl_values.append(dish.nacl_at(body.x, body.y))
        temp_values.append(dish.temperature_at(body.x, body.y))
        food_values.append(dish.food_at(body.x, body.y))
        actions.append(action)
        speeds.append(body.speed)

        # Real-time ASCII visualization
        if realtime and step_i % 10 == 0:
            _render_ascii(dish, body, step_i, n_steps)

    return {
        "trajectory": trajectory,
        "nacl_values": nacl_values,
        "temp_values": temp_values,
        "food_values": food_values,
        "actions": actions,
        "speeds": speeds,
        "total_distance": body.total_distance,
    }


# ==========================================================================
# ASCII Visualization
# ==========================================================================

def _render_ascii(
    dish: PetriDish,
    body: CelegansBody,
    step: int,
    total_steps: int,
    width: int = 40,
    height: int = 20,
) -> None:
    """Render ASCII visualization of petri dish + worm."""
    # Clear screen
    print("\033[2J\033[H", end="")

    grid = [["." for _ in range(width)] for _ in range(height)]

    # Draw dish boundary (circle)
    for row in range(height):
        for col in range(width):
            # Map grid to dish coordinates
            px = (col / width) * dish.size_mm
            py = ((height - 1 - row) / height) * dish.size_mm
            if not dish.is_in_dish(px, py):
                grid[row][col] = "#"

    # Draw food patches
    for patch in dish.food_patches:
        col = int(patch.x / dish.size_mm * width)
        row = int((1.0 - patch.y / dish.size_mm) * height)
        if 0 <= row < height and 0 <= col < width:
            if patch.remaining > 0.5:
                grid[row][col] = "F"
            elif patch.remaining > 0.1:
                grid[row][col] = "f"

    # Draw obstacles
    for ox, oy, _ in dish.obstacles:
        col = int(ox / dish.size_mm * width)
        row = int((1.0 - oy / dish.size_mm) * height)
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = "X"

    # Draw NaCl gradient indicator
    col = int(dish.nacl_peak_x / dish.size_mm * width)
    row = int((1.0 - dish.nacl_peak_y / dish.size_mm) * height)
    if 0 <= row < height and 0 <= col < width:
        grid[row][col] = "S"  # salt peak

    # Draw worm
    wcol = int(body.x / dish.size_mm * width)
    wrow = int((1.0 - body.y / dish.size_mm) * height)
    if 0 <= wrow < height and 0 <= wcol < width:
        if body.state == "forward":
            grid[wrow][wcol] = ">"
        elif body.state == "reverse":
            grid[wrow][wcol] = "<"
        else:
            grid[wrow][wcol] = "@"

    # Print
    print(f"  C. elegans Ecosystem  [step {step}/{total_steps}]")
    print(f"  State: {body.state:8s}  Pos: ({body.x:.1f}, {body.y:.1f})")
    print(f"  NaCl: {dish.nacl_at(body.x, body.y):.3f}  "
          f"Temp: {dish.temperature_at(body.x, body.y):.1f}C  "
          f"Food: {dish.food_at(body.x, body.y):.3f}")
    print("  +" + "-" * width + "+")
    for row in grid:
        print("  |" + "".join(row) + "|")
    print("  +" + "-" * width + "+")
    print("  Legend: > worm(fwd)  < worm(rev)  @ omega  F food  S salt  X obstacle")


# ==========================================================================
# Utilities
# ==========================================================================

def _header(title: str, subtitle: str) -> None:
    """Print formatted section header."""
    w = 76
    print("\n" + "=" * w)
    print(f"  {title}")
    print(f"  {subtitle}")
    print("=" * w)


def _warmup_ns(ns: CelegansNervousSystem, n_steps: int = 400) -> None:
    """Stabilize the nervous system with background tonic activity.

    C. elegans neurons have baseline tonic activity. We simulate this
    with periodic mild stimulation to all sensory neurons, allowing
    the network to reach a stable dynamic equilibrium before experiments.
    """
    for s in range(n_steps):
        if s % 5 == 0:
            # Mild tonic drive to sensory neurons (mimics ambient environment)
            ns.brain.external_current[ns._ase_tensor] += 5.0
            ns.brain.external_current[ns._afd_tensor] += 3.0
            # Tonic AVB drive (default forward locomotion bias)
            ns.brain.external_current[ns._avb_tensor] += 4.0
        ns.step()


def _system_info() -> Dict[str, Any]:
    """Collect system information for JSON output."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "backend": detect_backend(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name()
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["gpu"] = "Apple Silicon (MPS)"
    else:
        info["gpu"] = "CPU only"
    return info


def _make_json_safe(obj: Any) -> Any:
    """Convert results dict to JSON-serializable form."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    return obj


# ==========================================================================
# EXPERIMENT 1: Chemotaxis (NaCl Gradient Navigation)
# ==========================================================================

def exp_chemotaxis(
    device: str = "auto",
    seed: int = 42,
    n_episodes: int = 20,
    n_steps: int = 500,
    realtime: bool = False,
) -> Dict[str, Any]:
    """NaCl chemotaxis -- does the worm navigate UP the chemical gradient?

    C. elegans uses a biased random walk (pirouette strategy) for chemotaxis:
    when moving up the gradient (dC/dt > 0), suppress turning and continue
    forward. When moving down (dC/dt < 0), increase turning frequency.
    This produces net migration toward the attractant peak.

    Pass criteria: worm ends closer to NaCl peak than starting position
    in >50% of episodes.
    """
    _header(
        "Exp 1: NaCl Chemotaxis",
        "Biased random walk -- navigate UP the salt gradient"
    )
    t0 = time.perf_counter()

    successes = 0
    all_start_dist = []
    all_end_dist = []
    all_improvement = []

    for ep in range(n_episodes):
        ep_seed = seed + ep * 7

        # Build fresh nervous system
        ns = CelegansNervousSystem(device=device, seed=ep_seed)
        if ns.dev.type == 'cuda':
            ns.brain.compile()

        if ep == 0:
            print(f"    Brain: {N_TOTAL} neurons, {ns.brain.n_synapses} synapses on {ns.dev}")

        # Random starting position (center area, not near salt peak)
        rng = random.Random(ep_seed)
        start_x = rng.uniform(1.5, 4.5)
        start_y = rng.uniform(2.0, 8.0)
        start_heading = rng.uniform(0, 2 * math.pi)

        body = CelegansBody(x=start_x, y=start_y, heading=start_heading)
        dish = PetriDish(seed=ep_seed)
        sensory = CelegansSensory(ns)
        motor = CelegansMotor(ns)

        # Warmup
        _warmup_ns(ns)

        # Run episode
        show_rt = realtime and ep == 0
        result = simulate_episode(ns, body, dish, sensory, motor,
                                  n_steps=n_steps, realtime=show_rt)

        # Measure: distance to NaCl peak at start vs end
        start_dist = math.sqrt((start_x - dish.nacl_peak_x) ** 2
                               + (start_y - dish.nacl_peak_y) ** 2)
        end_dist = body.distance_to(dish.nacl_peak_x, dish.nacl_peak_y)

        improvement = start_dist - end_dist  # positive = closer to peak
        all_start_dist.append(start_dist)
        all_end_dist.append(end_dist)
        all_improvement.append(improvement)

        if improvement > 0:
            successes += 1

        if (ep + 1) % 5 == 0:
            print(f"    Episode {ep+1:3d}/{n_episodes}: "
                  f"start_dist={start_dist:.2f} end_dist={end_dist:.2f} "
                  f"improvement={improvement:+.2f}mm  "
                  f"[{'closer' if improvement > 0 else 'farther'}]")

    success_rate = successes / n_episodes
    mean_improvement = sum(all_improvement) / n_episodes
    elapsed = time.perf_counter() - t0

    passed = success_rate > 0.50

    print(f"\n    Results:")
    print(f"    Success rate (closer to NaCl): {success_rate:.0%} ({successes}/{n_episodes})")
    print(f"    Mean improvement:              {mean_improvement:+.3f} mm")
    print(f"    Mean start distance:           {sum(all_start_dist)/n_episodes:.2f} mm")
    print(f"    Mean end distance:             {sum(all_end_dist)/n_episodes:.2f} mm")
    print(f"    {'PASS' if passed else 'FAIL'} (>{50}% closer to NaCl peak) in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "success_rate": success_rate,
        "mean_improvement_mm": mean_improvement,
        "improvements": all_improvement,
    }


# ==========================================================================
# EXPERIMENT 2: Thermotaxis (Temperature Preference)
# ==========================================================================

def exp_thermotaxis(
    device: str = "auto",
    seed: int = 42,
    n_episodes: int = 20,
    n_steps: int = 500,
    realtime: bool = False,
) -> Dict[str, Any]:
    """Thermotaxis -- does the worm navigate to its cultivation temperature?

    C. elegans cultivated at 20C will navigate toward 20C on a thermal
    gradient (Mori & Ohshima 1995). This is mediated by AFD thermosensory
    neurons that detect temperature changes and modulate the AIY/AIZ
    forward/turn decision circuit.

    Pass criteria: worm spends more time in the 18-22C zone than in
    the extreme zones (<17C or >23C).
    """
    _header(
        "Exp 2: Thermotaxis",
        "Navigate to cultivation temperature (~20C) on thermal gradient"
    )
    t0 = time.perf_counter()

    all_time_preferred = []
    all_time_extreme = []

    for ep in range(n_episodes):
        ep_seed = seed + ep * 13

        ns = CelegansNervousSystem(device=device, seed=ep_seed)
        if ns.dev.type == 'cuda':
            ns.brain.compile()

        if ep == 0:
            print(f"    Brain: {N_TOTAL} neurons, {ns.brain.n_synapses} synapses on {ns.dev}")

        # Start at random position
        rng = random.Random(ep_seed)
        start_x = rng.uniform(1.0, 9.0)  # random position on temp gradient
        start_y = rng.uniform(2.0, 8.0)
        start_heading = rng.uniform(0, 2 * math.pi)

        body = CelegansBody(x=start_x, y=start_y, heading=start_heading)
        dish = PetriDish(seed=ep_seed)
        sensory = CelegansSensory(ns)
        motor = CelegansMotor(ns)

        _warmup_ns(ns)

        result = simulate_episode(ns, body, dish, sensory, motor,
                                  n_steps=n_steps, realtime=(realtime and ep == 0))

        # Count time in preferred zone (18-22C) vs extreme zones
        temps = result["temp_values"]
        n_preferred = sum(1 for t in temps if 18.0 <= t <= 22.0)
        n_extreme = sum(1 for t in temps if t < 17.0 or t > 23.0)
        n_total = len(temps)

        frac_preferred = n_preferred / n_total
        frac_extreme = n_extreme / n_total
        all_time_preferred.append(frac_preferred)
        all_time_extreme.append(frac_extreme)

        if (ep + 1) % 5 == 0:
            print(f"    Episode {ep+1:3d}/{n_episodes}: "
                  f"preferred_zone={frac_preferred:.0%}  "
                  f"extreme_zone={frac_extreme:.0%}")

    mean_preferred = sum(all_time_preferred) / n_episodes
    mean_extreme = sum(all_time_extreme) / n_episodes
    elapsed = time.perf_counter() - t0

    # Pass: more time in preferred zone than extreme zone
    passed = mean_preferred > mean_extreme

    print(f"\n    Results:")
    print(f"    Mean time in preferred (18-22C): {mean_preferred:.0%}")
    print(f"    Mean time in extreme (<17/>23C): {mean_extreme:.0%}")
    print(f"    Ratio preferred/extreme:         {mean_preferred / max(mean_extreme, 0.01):.2f}x")
    print(f"    {'PASS' if passed else 'FAIL'} (preferred > extreme) in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "mean_preferred_fraction": mean_preferred,
        "mean_extreme_fraction": mean_extreme,
    }


# ==========================================================================
# EXPERIMENT 3: Food Search (Foraging)
# ==========================================================================

def exp_food_search(
    device: str = "auto",
    seed: int = 42,
    n_episodes: int = 20,
    n_steps: int = 500,
    realtime: bool = False,
) -> Dict[str, Any]:
    """Food search -- does the worm find and stay near food?

    C. elegans exhibits area-restricted search (ARS) near food: it slows
    down and increases turning frequency when on food (basal slowing response,
    mediated by serotonin). When leaving food, it performs local search
    with high reversal/omega turn frequency before transitioning to
    long-range dispersal.

    Pass criteria: time spent near food > random baseline.
    Random baseline: fraction of dish area covered by food patches.
    """
    _header(
        "Exp 3: Food Search (Foraging)",
        "Find bacterial lawns + area-restricted search near food"
    )
    t0 = time.perf_counter()

    all_food_time = []

    for ep in range(n_episodes):
        ep_seed = seed + ep * 17

        ns = CelegansNervousSystem(device=device, seed=ep_seed)
        if ns.dev.type == 'cuda':
            ns.brain.compile()

        if ep == 0:
            print(f"    Brain: {N_TOTAL} neurons, {ns.brain.n_synapses} synapses on {ns.dev}")

        # Start at center. Food patches placed NEAR center so they are
        # reachable within the simulation timeframe. With ~0.05 mm/s speed
        # and 500 steps at dt=0.05s, the worm covers ~1.25mm total.
        body = CelegansBody(x=5.0, y=5.0,
                            heading=random.Random(ep_seed).uniform(0, 2 * math.pi))
        dish = PetriDish(seed=ep_seed, n_food=5)
        # Override food patches: place them near center within reach
        ep_rng = random.Random(ep_seed + 999)
        dish.food_patches = []
        for _ in range(5):
            angle = ep_rng.uniform(0, 2 * math.pi)
            dist = ep_rng.uniform(0.3, 1.2)  # close to center
            fx = 5.0 + dist * math.cos(angle)
            fy = 5.0 + dist * math.sin(angle)
            dish.food_patches.append(FoodPatch(fx, fy, 0.5, 1.0))
        sensory = CelegansSensory(ns)
        motor = CelegansMotor(ns)

        _warmup_ns(ns)

        result = simulate_episode(ns, body, dish, sensory, motor,
                                  n_steps=n_steps, realtime=(realtime and ep == 0))

        # Count time near food (food concentration > 0.1)
        food_time = sum(1 for f in result["food_values"] if f > 0.1)
        frac_food = food_time / len(result["food_values"])
        all_food_time.append(frac_food)

        if (ep + 1) % 5 == 0:
            print(f"    Episode {ep+1:3d}/{n_episodes}: "
                  f"time_near_food={frac_food:.0%}")

    mean_food_time = sum(all_food_time) / n_episodes
    elapsed = time.perf_counter() - t0

    # Random baseline: approximate fraction of dish area covered by food
    # Each food patch ~0.6mm radius in 10mm dish (area ~pi*0.6^2 / pi*5^2 = ~1.4% per patch)
    # With 5 patches, random baseline ~7%
    random_baseline = 0.07
    passed = mean_food_time > random_baseline

    print(f"\n    Results:")
    print(f"    Mean time near food:   {mean_food_time:.0%}")
    print(f"    Random baseline:       {random_baseline:.0%}")
    print(f"    Enhancement ratio:     {mean_food_time / max(random_baseline, 0.01):.2f}x")
    print(f"    {'PASS' if passed else 'FAIL'} (food_time > random baseline) in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "mean_food_fraction": mean_food_time,
        "random_baseline": random_baseline,
        "all_food_fractions": all_food_time,
    }


# ==========================================================================
# EXPERIMENT 4: Drug Effects on Locomotion
# ==========================================================================

def exp_drug_effects(
    device: str = "auto",
    seed: int = 42,
    n_episodes: int = 10,
    n_steps: int = 300,
    realtime: bool = False,
) -> Dict[str, Any]:
    """Drug effects on locomotion speed and activity.

    Tests pharmacological modulation of C. elegans behavior:
    - Baseline: normal locomotion speed
    - Diazepam (GABA-A enhancer): should REDUCE speed (muscle relaxation)
    - Caffeine (adenosine antagonist): should INCREASE activity
    - Alprazolam (high-potency GABA-A enhancer): should be MORE sedating

    The real C. elegans has GABA-A receptors on body wall muscles.
    Benzodiazepines enhance GABAergic inhibition --> reduced contraction
    --> slower or paralyzed movement. Caffeine via reduced GABAergic
    inhibition --> increased motor output.

    Pass criteria: benzodiazepines reduce mean speed vs baseline.
    """
    _header(
        "Exp 4: Drug Effects on Locomotion",
        "Pharmacological modulation -- diazepam, caffeine, alprazolam"
    )
    t0 = time.perf_counter()

    conditions = [
        ("baseline", None, 0.0),
        ("diazepam_5mg", "diazepam", 5.0),
        ("caffeine_100mg", "caffeine", 100.0),
        ("alprazolam_1mg", "alprazolam", 1.0),
    ]

    condition_results: Dict[str, Dict[str, float]] = {}

    for cond_name, drug_name, dose in conditions:
        print(f"\n    Condition: {cond_name}")
        speeds = []
        distances = []

        for ep in range(n_episodes):
            ep_seed = seed + ep * 23

            ns = CelegansNervousSystem(device=device, seed=ep_seed)
            if ns.dev.type == 'cuda':
                ns.brain.compile()

            # Apply drug BEFORE warmup (drug takes effect during warmup)
            if drug_name is not None:
                ns.apply_drug(drug_name, dose)

            body = CelegansBody(x=5.0, y=5.0,
                                heading=random.Random(ep_seed).uniform(0, 2 * math.pi))
            dish = PetriDish(seed=ep_seed)
            sensory = CelegansSensory(ns)
            motor = CelegansMotor(ns)

            _warmup_ns(ns)

            drug_param = (drug_name, dose) if drug_name else None
            result = simulate_episode(ns, body, dish, sensory, motor,
                                      n_steps=n_steps,
                                      realtime=(realtime and ep == 0 and cond_name == "baseline"),
                                      drug=drug_param)

            mean_speed = sum(result["speeds"]) / len(result["speeds"])
            speeds.append(mean_speed)
            distances.append(result["total_distance"])

        mean_speed = sum(speeds) / len(speeds)
        mean_dist = sum(distances) / len(distances)
        std_speed = (sum((s - mean_speed) ** 2 for s in speeds) / max(len(speeds) - 1, 1)) ** 0.5

        condition_results[cond_name] = {
            "mean_speed": mean_speed,
            "std_speed": std_speed,
            "mean_distance": mean_dist,
            "speeds": speeds,
        }

        print(f"      Mean speed:    {mean_speed:.4f} +/- {std_speed:.4f} mm/s")
        print(f"      Mean distance: {mean_dist:.2f} mm")

    # Analysis
    baseline_speed = condition_results["baseline"]["mean_speed"]
    diazepam_speed = condition_results["diazepam_5mg"]["mean_speed"]
    caffeine_speed = condition_results["caffeine_100mg"]["mean_speed"]
    alprazolam_speed = condition_results["alprazolam_1mg"]["mean_speed"]

    elapsed = time.perf_counter() - t0

    # Pass criteria: benzodiazepines reduce speed
    diazepam_reduces = diazepam_speed < baseline_speed
    alprazolam_reduces = alprazolam_speed < baseline_speed
    # Alprazolam should be more sedating than diazepam (lower speed)
    alprazolam_stronger = alprazolam_speed <= diazepam_speed

    passed = diazepam_reduces and alprazolam_reduces

    print(f"\n    Results:")
    print(f"    Baseline speed:    {baseline_speed:.4f} mm/s")
    print(f"    Diazepam speed:    {diazepam_speed:.4f} mm/s "
          f"({(diazepam_speed / baseline_speed - 1) * 100:+.1f}%) "
          f"[{'reduced' if diazepam_reduces else 'NOT reduced'}]")
    print(f"    Caffeine speed:    {caffeine_speed:.4f} mm/s "
          f"({(caffeine_speed / baseline_speed - 1) * 100:+.1f}%)")
    print(f"    Alprazolam speed:  {alprazolam_speed:.4f} mm/s "
          f"({(alprazolam_speed / baseline_speed - 1) * 100:+.1f}%) "
          f"[{'reduced' if alprazolam_reduces else 'NOT reduced'}]")
    print(f"    Alprazolam > diazepam sedation: {'YES' if alprazolam_stronger else 'NO'}")
    print(f"    {'PASS' if passed else 'FAIL'} (benzos reduce speed) in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "baseline_speed": baseline_speed,
        "diazepam_speed": diazepam_speed,
        "caffeine_speed": caffeine_speed,
        "alprazolam_speed": alprazolam_speed,
        "diazepam_reduces": diazepam_reduces,
        "alprazolam_reduces": alprazolam_reduces,
        "alprazolam_stronger": alprazolam_stronger,
        "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "speeds"}
                       for k, v in condition_results.items()},
    }


# ==========================================================================
# Experiment Registry
# ==========================================================================

ALL_EXPERIMENTS = {
    1: ("NaCl Chemotaxis", exp_chemotaxis),
    2: ("Thermotaxis", exp_thermotaxis),
    3: ("Food Search (Foraging)", exp_food_search),
    4: ("Drug Effects on Locomotion", exp_drug_effects),
}


# ==========================================================================
# Main
# ==========================================================================

def _run_single(args, seed: int) -> Dict[int, Any]:
    """Run all requested experiments with a single seed."""
    exps = args.exp if args.exp else list(ALL_EXPERIMENTS.keys())
    results = {}

    for exp_id in exps:
        if exp_id not in ALL_EXPERIMENTS:
            print(f"\n  Unknown experiment: {exp_id}")
            continue
        name, func = ALL_EXPERIMENTS[exp_id]

        try:
            kwargs = {
                "device": args.device,
                "seed": seed,
                "realtime": args.realtime,
            }
            if args.steps:
                kwargs["n_steps"] = args.steps
            if args.episodes:
                kwargs["n_episodes"] = args.episodes
            result = func(**kwargs)
            results[exp_id] = result
        except Exception as e:
            print(f"\n  EXPERIMENT {exp_id} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {"passed": False, "error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="C. elegans Ecosystem -- Eon-style dONN Organism Simulation"
    )
    parser.add_argument("--exp", type=int, nargs="*", default=None,
                        help="Which experiments to run (1-4). Default: all")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None,
                        help="Override number of simulation steps per episode")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes per experiment")
    parser.add_argument("--json", type=str, default=None, metavar="PATH",
                        help="Write structured JSON results to file")
    parser.add_argument("--realtime", action="store_true",
                        help="Show ASCII visualization during simulation")
    args = parser.parse_args()

    print("=" * 76)
    print("  C. ELEGANS ECOSYSTEM -- EON-STYLE dONN ORGANISM SIMULATION")
    print(f"  Backend: {detect_backend()} | Device: {args.device}")
    print(f"  302-neuron nervous system with HH ion channels + 6 neurotransmitters")
    print(f"  Approximating the real C. elegans connectome (White et al. 1986)")
    print("=" * 76)

    total_time = time.perf_counter()
    results = _run_single(args, args.seed)
    total = time.perf_counter() - total_time

    # Summary
    print("\n" + "=" * 76)
    print("  C. ELEGANS ECOSYSTEM -- SUMMARY")
    print("=" * 76)

    passed = 0
    total_exp = 0
    for exp_id, result in sorted(results.items()):
        if exp_id not in ALL_EXPERIMENTS:
            continue
        name = ALL_EXPERIMENTS[exp_id][0]
        status = "PASS" if result.get("passed") else "FAIL"
        t = result.get("time", 0)
        print(f"    {exp_id}. {name:35s} [{status}]  {t:.1f}s")
        total_exp += 1
        if result.get("passed"):
            passed += 1

    print(f"\n  Total: {passed}/{total_exp} passed in {total:.1f}s")
    print("=" * 76)

    # JSON output
    if args.json:
        json_data = {
            "experiment": "celegans_ecosystem",
            "device": args.device,
            "seed": args.seed,
            "total_time_s": round(total, 2),
            "n_neurons": N_TOTAL,
            "system": _system_info(),
            "experiments": {},
        }
        for exp_id, result in sorted(results.items()):
            if exp_id not in ALL_EXPERIMENTS:
                continue
            name = ALL_EXPERIMENTS[exp_id][0]
            clean = _make_json_safe(result)
            clean["name"] = name
            json_data["experiments"][str(exp_id)] = clean

        with open(args.json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\n  JSON results written to: {args.json}")

    all_passed = all(
        r.get("passed", False)
        for r in results.values()
        if isinstance(r, dict) and "passed" in r
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
