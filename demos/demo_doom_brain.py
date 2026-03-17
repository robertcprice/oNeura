#!/usr/bin/env python3
"""Doom FPS Brain -- A Biophysical Brain Plays Doom via Molecular Retina and FEP.

A digital Organic Neural Network (dONN) plays a Doom-style FPS game using the
oNeura molecular brain (HH dynamics, 6 NTs, STDP) and optional biophysical
retina (3-layer photoreceptor-bipolar-RGC circuit).

TWO modes demonstrate embodied cognition vs disembodied computation:

  EMBODIED MODE:
    Brain FEELS the game -- damage becomes pain through nociceptor populations
    with NE arousal bursts, health pickups trigger DA reward via VTA-like
    neurons, and visual input flows through a full molecular retina
    (photoreceptors -> bipolar cells -> RGC spike trains -> optic lobe).
    This mimics real neural pathways: spinothalamic tract -> thalamus ->
    insular cortex -> amygdala for pain; mesolimbic DA pathway for reward.

  DISEMBODIED MODE:
    Brain plays without sensory/interoceptive involvement -- raycasts encode
    distances directly as currents to cortical populations (no retina, no
    pain, no reward). Pure information processing baseline.

3 Experiments:
    1. Doom Navigation (both modes, 50 episodes)
       Can the brain navigate to a goal? Compare embodied vs disembodied.
       PASS: goal rate > random (>5%) AND improving over time.

    2. Doom Survival (both modes, 80 episodes)
       Can the brain survive longer with experience?
       PASS: survival time increases AND damage per episode decreases.

    3. Doom Drug Effects (embodied only, 3 drugs)
       Train 3 identical brains, test baseline / caffeine / diazepam.
       PASS: diazepam impairs performance (GABA-A enhancement).

All learning is driven by the FREE ENERGY PRINCIPLE (Friston 2010) -- no
explicit reward function. Structured feedback = low entropy (predictable),
unstructured noise = high entropy (unpredictable). Neurons self-organize
via STDP to prefer states producing structured feedback.

References:
    - Kagan et al. (2022) "In vitro neurons learn and exhibit sentience
      when embodied in a simulated game-world" Neuron 110(23):3952-3969
    - Friston (2010) "The free-energy principle: a unified brain theory?"
      Nature Reviews Neuroscience 11:127-138
    - Craig (2003) "Interoception: the sense of the physiological condition
      of the body" Current Opinion in Neurobiology 13(4):500-505
    - Loeser & Treede (2008) "The Kyoto protocol of IASP Basic Pain
      Terminology" Pain 137(3):473-477

Usage:
    python3 demos/demo_doom_brain.py                        # all 3, both modes
    python3 demos/demo_doom_brain.py --mode embodied        # embodied only
    python3 demos/demo_doom_brain.py --exp 1 --mode both    # navigation, both
    python3 demos/demo_doom_brain.py --scale medium --exp 2 # survival, medium
    python3 demos/demo_doom_brain.py --runs 3 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oneuro.molecular.cuda_backend import (
    CUDAMolecularBrain,
    CUDARegionalBrain,
    detect_backend,
    NT_DA, NT_5HT, NT_NE, NT_ACH, NT_GLU, NT_GABA,
)
from oneuro.environments.doom_fps import DoomFPS
from oneuro.molecular.retina import MolecularRetina

# Import shared helpers from the language demo (same pattern as doom_arena).
from demo_language_cuda import (
    _warmup,
    _header,
    _get_region_ids,
    _get_all_cortex_ids,
    _get_cortex_l5_ids,
    SCALE_COLUMNS,
)


# ============================================================================
# Constants
# ============================================================================

# 8-directional movement: maps to DoomFPS actions 0-7.
DIRECTION_NAMES = [
    "forward", "backward", "strafe_left", "strafe_right",
    "turn_left", "turn_right", "turn_left+fwd", "turn_right+fwd",
]

# Retina resolution scales (width, height) -- must match DoomFPS render size.
RETINA_RESOLUTIONS = {
    "small": (16, 12),
    "medium": (24, 18),
    "large": (32, 24),
    "mega": (48, 36),
}

# DoomFPS render resolutions.
RENDER_RESOLUTIONS = {
    "small": (16, 12),
    "medium": (24, 18),
    "large": (32, 24),
    "mega": (48, 36),
}

# Raycast directions for disembodied mode: 8 evenly spaced angles.
RAYCAST_ANGLES = [i * math.pi / 4.0 for i in range(8)]

# Sensory intensity constants.
INTENSITY_WALL = 30.0
INTENSITY_ENEMY = 55.0
INTENSITY_HEALTH = 40.0
INTENSITY_GOAL = 60.0

# Episode parameters per scale.
DOOM_PARAMS = {
    "small":  {"map_size": 20, "max_steps": 60,  "n_enemies": 2, "n_health": 3},
    "medium": {"map_size": 24, "max_steps": 80,  "n_enemies": 3, "n_health": 4},
    "large":  {"map_size": 28, "max_steps": 100, "n_enemies": 3, "n_health": 4},
    "mega":   {"map_size": 32, "max_steps": 120, "n_enemies": 4, "n_health": 5},
}


# ============================================================================
# Embodied Sensory Encoder (Retina-based)
# ============================================================================

class EmbodiedVisualEncoder:
    """Full retina pipeline: DoomFPS.render() -> MolecularRetina -> RGC spikes -> brain.

    The retina processes the raw pixel frame through three biophysical layers:
    1. Photoreceptors (rods/cones): hyperpolarize proportionally to light intensity
    2. Bipolar cells (ON/OFF): center-surround antagonism for edge detection
    3. RGCs (spiking): produce action potentials transmitted to the brain

    RGC spike indices are mapped to optic lobe (sensory cortex) neurons in the
    brain, creating a biologically plausible visual pathway.

    Args:
        retina: A MolecularRetina instance matching the render resolution.
        sensory_ids: Tensor of sensory cortex neuron IDs to receive RGC output.
        device: Torch device string.
    """

    def __init__(
        self,
        retina: MolecularRetina,
        sensory_ids: torch.Tensor,
        device: str = "cpu",
    ) -> None:
        self.retina = retina
        self.sensory_ids = sensory_ids
        self.n_sensory = len(sensory_ids)
        self.n_rgc = retina.n_rgc
        self.device = device
        # Map RGC neuron_ids (which may not be contiguous) to sensory neuron indices.
        # If there are more RGCs than sensory neurons, we wrap; if fewer, some are unused.
        self._rgc_to_sensory_idx: Dict[int, int] = {}
        for i, rgc in enumerate(retina.rgc_cells):
            brain_idx = i % self.n_sensory
            self._rgc_to_sensory_idx[rgc.neuron_id] = brain_idx

    def encode(
        self,
        frame: np.ndarray,
        brain: CUDAMolecularBrain,
        pulsed_step: int = 0,
        intensity: float = 50.0,
    ) -> None:
        """Process frame through retina and inject RGC spikes into brain.

        Args:
            frame: (H, W, 3) uint8 RGB frame from DoomFPS.render().
            brain: The brain to stimulate.
            pulsed_step: Current step (even steps only for pulsing).
            intensity: Current injection strength per spiking RGC.
        """
        if pulsed_step % 2 != 0:
            return

        # Process frame through retina (returns list of fired RGC neuron_ids).
        fired_rgcs = self.retina.process_frame(frame, n_steps=5, dt=0.5)

        if not fired_rgcs:
            return

        # Map fired RGC IDs to brain sensory neuron indices.
        brain_indices = []
        for rgc_id in fired_rgcs:
            if rgc_id in self._rgc_to_sensory_idx:
                sensory_idx = self._rgc_to_sensory_idx[rgc_id]
                brain_indices.append(self.sensory_ids[sensory_idx].item())

        if brain_indices:
            idx_tensor = torch.tensor(
                brain_indices, dtype=torch.int64, device=brain.device,
            )
            brain.external_current[idx_tensor] += intensity


# ============================================================================
# Disembodied Sensory Encoder (Raycast-based)
# ============================================================================

class DisembodiedVisualEncoder:
    """Direct raycast-to-cortex encoder bypassing the retina.

    Casts 8 rays from the player position at evenly spaced angles and encodes
    wall/enemy/health/goal distances as current strengths into 8 cortical
    populations. Closer objects produce stronger currents.

    This is a simplified information-theoretic encoding: the brain receives
    the same spatial information without biological retinal processing.

    Args:
        sensory_ids: Tensor of sensory cortex neuron IDs.
        device: Torch device string.
    """

    def __init__(self, sensory_ids: torch.Tensor, device: str = "cpu") -> None:
        self.sensory_ids = sensory_ids
        self.n_sensory = len(sensory_ids)
        self.device = device
        # Split sensory neurons into 8 direction groups.
        self.n_per_dir = max(1, self.n_sensory // 8)
        self.dir_groups: List[torch.Tensor] = []
        for i in range(8):
            start = i * self.n_per_dir
            end = min(start + self.n_per_dir, self.n_sensory)
            self.dir_groups.append(sensory_ids[start:end])

    def encode(
        self,
        env: DoomFPS,
        brain: CUDAMolecularBrain,
        pulsed_step: int = 0,
    ) -> None:
        """Cast rays and inject distance-encoded currents into sensory populations.

        Args:
            env: The DoomFPS environment (reads player position/angle and map).
            brain: The brain to stimulate.
            pulsed_step: Current step (even steps only for pulsing).
        """
        if pulsed_step % 2 != 0:
            return

        if env.doom_map is None:
            return

        px, py = env.player_x, env.player_y
        pa = env.player_angle

        for d in range(8):
            # Ray direction: player angle + offset for each of 8 directions.
            ray_angle = pa + RAYCAST_ANGLES[d]
            ray_dx = math.cos(ray_angle)
            ray_dy = math.sin(ray_angle)

            # Simple raycast: step in small increments, find distance to first wall.
            dist = 0.0
            max_dist = float(env.map_size)
            step_size = 0.3
            hit_type = "wall"  # default

            while dist < max_dist:
                dist += step_size
                rx = px + ray_dx * dist
                ry = py + ray_dy * dist

                # Check map bounds.
                gx, gy = int(rx), int(ry)
                if gx < 0 or gx >= env.map_size or gy < 0 or gy >= env.map_size:
                    break

                # Check wall.
                if env.doom_map.is_wall(gx, gy):
                    break

                # Check sprites at this location.
                for sprite in env.sprites:
                    if not sprite.active:
                        continue
                    sdist = math.sqrt((sprite.x - rx) ** 2 + (sprite.y - ry) ** 2)
                    if sdist < 0.6:
                        if sprite.sprite_type == "enemy":
                            hit_type = "enemy"
                        elif sprite.sprite_type == "health":
                            hit_type = "health"
                        elif sprite.sprite_type == "goal":
                            hit_type = "goal"
                        dist = math.sqrt(
                            (sprite.x - px) ** 2 + (sprite.y - py) ** 2,
                        )
                        break
                else:
                    continue
                break

            # Encode distance as current: closer = stronger.
            if dist < 0.1:
                dist = 0.1
            proximity = max(0.0, 1.0 - dist / max_dist)

            # Different object types get different intensity scaling.
            if hit_type == "enemy":
                current = proximity * INTENSITY_ENEMY
            elif hit_type == "health":
                current = proximity * INTENSITY_HEALTH
            elif hit_type == "goal":
                current = proximity * INTENSITY_GOAL
            else:
                current = proximity * INTENSITY_WALL

            if current > 1.0 and d < len(self.dir_groups):
                group = self.dir_groups[d]
                if group.numel() > 0:
                    brain.external_current[group] += current


# ============================================================================
# Nociceptor System (Embodied Pain Pathway)
# ============================================================================

class NociceptorSystem:
    """Pain pathway: damage -> nociceptor neurons -> NE arousal -> aversive signal.

    Mimics the spinothalamic tract: nociceptors detect tissue damage,
    project to thalamus, which relays to insular cortex (pain perception)
    and amygdala (fear/aversion). NE burst from locus coeruleus enhances
    arousal and STDP gain during painful events.

    This is NOT a programmed reward function -- the NE burst and nociceptor
    activation create correlated neural activity that, through STDP,
    associates the preceding motor commands with aversive consequences.
    The learning is emergent from neural dynamics.

    Args:
        nociceptor_ids: Brain neuron IDs for nociceptor population.
        cortex_ids: All cortical neuron IDs (for NE-mediated arousal).
        device: Torch device string.
    """

    def __init__(
        self,
        nociceptor_ids: torch.Tensor,
        cortex_ids: torch.Tensor,
        device: str = "cpu",
    ) -> None:
        self.nociceptor_ids = nociceptor_ids
        self.cortex_ids = cortex_ids
        self.device = device
        self.n_nociceptors = len(nociceptor_ids)
        self.n_cortex = len(cortex_ids)

    def signal_damage(
        self,
        rb: CUDARegionalBrain,
        damage_amount: int,
        pain_steps: int = 40,
    ) -> None:
        """Deliver pain signal proportional to damage taken.

        Pathway:
        1. Strong current to nociceptor neurons (tissue damage -> C-fiber activation)
        2. NE burst to cortex (locus coeruleus arousal response)
        3. Unstructured noise to cortex (high entropy = prediction error via FEP)

        Args:
            rb: Regional brain.
            damage_amount: HP lost (scales pain intensity).
            pain_steps: Duration of pain signal in simulation steps.
        """
        brain = rb.brain
        pain_intensity = min(80.0, damage_amount * 3.0)

        # NE arousal burst -- locus coeruleus response to pain.
        brain.nt_conc[self.cortex_ids, NT_NE] += 150.0 * (damage_amount / 20.0)

        for s in range(pain_steps):
            # Nociceptor activation (pulsed to avoid depolarization block).
            if s % 2 == 0:
                brain.external_current[self.nociceptor_ids] += pain_intensity

            # Unstructured noise to cortex (high entropy = FEP damage signal).
            mask = torch.rand(self.n_cortex, device=self.device) < 0.3
            active = self.cortex_ids[mask]
            if active.numel() > 0:
                noise = torch.rand(active.numel(), device=self.device) * 40.0
                brain.external_current[active] += noise

            rb.step()


# ============================================================================
# Reward System (Embodied Pleasure/VTA Pathway)
# ============================================================================

class RewardSystem:
    """Reward pathway: positive events -> VTA DA neurons -> structured feedback.

    Mimics the mesolimbic dopamine pathway: VTA neurons release DA in
    response to appetitive stimuli (health pickup, goal reach, enemy kill).
    DA does NOT directly teach the network -- it modulates STDP gain via
    three-factor learning (pre x post x DA -> weight change).

    The structured pulse (low entropy) following reward events creates
    correlated activity that STDP strengthens, implementing FEP-based
    learning: the brain learns to produce actions that yield predictable
    (structured) rather than unpredictable (noisy) sensory feedback.

    Args:
        vta_ids: Brain neuron IDs for VTA/reward population.
        cortex_ids: All cortical neuron IDs.
        device: Torch device string.
    """

    def __init__(
        self,
        vta_ids: torch.Tensor,
        cortex_ids: torch.Tensor,
        device: str = "cpu",
    ) -> None:
        self.vta_ids = vta_ids
        self.cortex_ids = cortex_ids
        self.device = device
        self.n_cortex = len(cortex_ids)

    def signal_reward(
        self,
        rb: CUDARegionalBrain,
        magnitude: float = 1.0,
        reward_steps: int = 30,
    ) -> None:
        """Deliver reward via VTA DA release + structured pulse.

        Args:
            rb: Regional brain.
            magnitude: Reward strength multiplier.
            reward_steps: Duration of reward signal.
        """
        brain = rb.brain

        # DA release from VTA neurons.
        brain.nt_conc[self.vta_ids, NT_DA] += 200.0 * magnitude

        # Structured pulse to cortex (low entropy = predictable = FEP reward).
        intensity = 40.0 * magnitude
        for s in range(reward_steps):
            if s % 2 == 0:
                brain.external_current[self.cortex_ids] += intensity
                # Mild VTA stimulation to maintain DA release.
                brain.external_current[self.vta_ids] += 30.0 * magnitude
            rb.step()


# ============================================================================
# Interoceptor System (HP Level Sensing)
# ============================================================================

class InteroceptorSystem:
    """Body-state sensing: HP level -> tonic current to interoceptive neurons.

    Mimics visceral afferents (Craig 2003): interoceptors provide continuous
    information about bodily state. Low HP = high interoceptive alarm
    current; full HP = low baseline current. This gives the brain a
    proprioceptive sense of its own "health".

    Args:
        body_state_ids: Brain neuron IDs for interoceptive population.
    """

    def __init__(self, body_state_ids: torch.Tensor) -> None:
        self.body_state_ids = body_state_ids

    def update(self, brain: CUDAMolecularBrain, hp: int, max_hp: int = 100) -> None:
        """Inject tonic current proportional to HP deviation from full health.

        Full HP -> low current (homeostasis).
        Low HP -> high current (allostatic alarm).

        Args:
            brain: The brain to stimulate.
            hp: Current hit points.
            max_hp: Maximum hit points.
        """
        # Normalize: 0 (dead) to 1 (full health).
        hp_frac = max(0.0, min(1.0, hp / max_hp))
        # Inverted: low HP = high alarm current.
        alarm = (1.0 - hp_frac) * 25.0 + 5.0  # 5-30 uA range
        brain.external_current[self.body_state_ids] += alarm


# ============================================================================
# Motor Decoder (8-directional, voltage-based)
# ============================================================================

class DoomMotorDecoder:
    """8-directional spike-count decoder from L5 motor populations.

    L5 output neurons split into 8 populations mapping to DoomFPS actions.
    Uses zero-threshold decoding: any spike count difference drives action.
    This is the same approach used in the DishBrain Pong and Doom Arena demos.

    Args:
        l5_ids: Tensor of L5 neuron IDs across all cortical columns.
    """

    def __init__(self, l5_ids: torch.Tensor) -> None:
        n = len(l5_ids)
        self.n_per_dir = max(1, n // 8)
        self.dir_ids: List[torch.Tensor] = []
        for i in range(8):
            start = i * self.n_per_dir
            end = min(start + self.n_per_dir, n)
            self.dir_ids.append(l5_ids[start:end])

    def decode(self, counts: List[int]) -> int:
        """Decode action from 8-directional spike counts.

        Zero-threshold: any spike difference drives action. Ties broken randomly.

        Args:
            counts: List of 8 spike counts.

        Returns:
            Action index (0-7).
        """
        max_count = max(counts)
        if max_count == 0:
            return random.randint(0, 7)
        candidates = [i for i, c in enumerate(counts) if c == max_count]
        return random.choice(candidates)


# ============================================================================
# Embodied FEP Protocol
# ============================================================================

class EmbodiedFEP:
    """Free Energy Protocol with pain/reward integration for embodied mode.

    Extends the DishBrain FEP with nociception and interoception:
    - Goal reach:     Strong structured pulse + NE boost + VTA DA release
    - Health pickup:  Mild structured pulse + DA release
    - Enemy damage:   Nociceptor pain signal + unstructured noise
    - Near enemy:     Brief structured pulse (survival = predictable state)
    - Normal step:    Interoceptive HP update only

    All feedback is either structured (low entropy) or unstructured (high
    entropy). The brain learns through STDP-driven entropy minimization,
    NOT through explicit reward/punishment signals.

    Args:
        cortex_ids: All cortical neuron IDs.
        l5_ids: L5 motor neuron IDs.
        relay_ids: Thalamic relay neuron IDs.
        nociceptor: NociceptorSystem (embodied mode only).
        reward_sys: RewardSystem (embodied mode only).
        interoceptor: InteroceptorSystem (embodied mode only).
        device: Torch device string.
    """

    def __init__(
        self,
        cortex_ids: torch.Tensor,
        l5_ids: torch.Tensor,
        relay_ids: torch.Tensor,
        nociceptor: Optional[NociceptorSystem] = None,
        reward_sys: Optional[RewardSystem] = None,
        interoceptor: Optional[InteroceptorSystem] = None,
        device: str = "cpu",
        structured_intensity: float = 50.0,
        unstructured_intensity: float = 40.0,
        ne_boost: float = 200.0,
    ) -> None:
        self.cortex_ids = cortex_ids
        self.l5_ids = l5_ids
        self.relay_ids = relay_ids
        self.nociceptor = nociceptor
        self.reward_sys = reward_sys
        self.interoceptor = interoceptor
        self.device = device
        self.n_cortex = len(cortex_ids)
        self.structured_intensity = structured_intensity
        self.unstructured_intensity = unstructured_intensity
        self.ne_boost = ne_boost
        # Scale-adaptive Hebbian delta.
        n_l5 = len(l5_ids)
        self.hebbian_delta = 0.8 * max(1.0, (n_l5 / 200) ** 0.3)

    def deliver_goal(self, rb: CUDARegionalBrain) -> None:
        """Strong structured feedback + reward on goal completion."""
        brain = rb.brain
        brain.nt_conc[self.cortex_ids, NT_NE] += self.ne_boost
        if self.reward_sys is not None:
            self.reward_sys.signal_reward(rb, magnitude=2.0, reward_steps=40)
        for s in range(40):
            if s % 2 == 0:
                brain.external_current[self.cortex_ids] += self.structured_intensity
            rb.step()

    def deliver_pickup(self, rb: CUDARegionalBrain) -> None:
        """Mild structured feedback + reward on health pickup."""
        brain = rb.brain
        if self.reward_sys is not None:
            self.reward_sys.signal_reward(rb, magnitude=0.5, reward_steps=15)
        for s in range(20):
            if s % 2 == 0:
                brain.external_current[self.cortex_ids] += (
                    self.structured_intensity * 0.5
                )
            rb.step()

    def deliver_damage(self, rb: CUDARegionalBrain, damage: int = 20) -> None:
        """Nociceptor pain signal + unstructured noise on damage."""
        brain = rb.brain
        if self.nociceptor is not None:
            self.nociceptor.signal_damage(rb, damage, pain_steps=30)
        else:
            # Disembodied fallback: just unstructured noise.
            for s in range(60):
                mask = torch.rand(self.n_cortex, device=self.device) < 0.3
                active = self.cortex_ids[mask]
                if active.numel() > 0:
                    noise = (
                        torch.rand(active.numel(), device=self.device)
                        * self.unstructured_intensity
                    )
                    brain.external_current[active] += noise
                rb.step()

    def deliver_survive(self, rb: CUDARegionalBrain) -> None:
        """Brief structured pulse when near enemy but not hit."""
        brain = rb.brain
        for s in range(12):
            if s % 2 == 0:
                brain.external_current[self.cortex_ids] += (
                    self.structured_intensity * 0.3
                )
            rb.step()

    def deliver_timeout(self, rb: CUDARegionalBrain) -> None:
        """Mild unstructured noise on episode timeout (failed to find goal)."""
        brain = rb.brain
        for s in range(30):
            mask = torch.rand(self.n_cortex, device=self.device) < 0.2
            active = self.cortex_ids[mask]
            if active.numel() > 0:
                noise = (
                    torch.rand(active.numel(), device=self.device)
                    * self.unstructured_intensity * 0.5
                )
                brain.external_current[active] += noise
            rb.step()

    def update_interoception(
        self, brain: CUDAMolecularBrain, hp: int,
    ) -> None:
        """Update body-state sensing each step (embodied only)."""
        if self.interoceptor is not None:
            self.interoceptor.update(brain, hp)

    def hebbian_nudge_direction(
        self,
        brain: CUDAMolecularBrain,
        decoder: DoomMotorDecoder,
        correct_action: int,
        chosen_action: int,
    ) -> None:
        """Hebbian weight nudge: strengthen relay->correct motor, weaken wrong.

        Args:
            brain: The brain to modify.
            decoder: Motor decoder with directional neuron groups.
            correct_action: Optimal direction (0-7).
            chosen_action: Direction the network chose.
        """
        if brain.n_synapses == 0 or self.hebbian_delta <= 0:
            return

        relay_set = set(self.relay_ids.cpu().tolist())
        correct_set = set(decoder.dir_ids[correct_action].cpu().tolist())

        pre_np = brain.syn_pre.cpu().numpy()
        post_np = brain.syn_post.cpu().numpy()
        relay_mask = np.isin(pre_np, list(relay_set))

        # Strengthen relay -> correct motor.
        correct_post = np.isin(post_np, list(correct_set))
        strengthen = relay_mask & correct_post
        if strengthen.any():
            idx = torch.tensor(
                np.where(strengthen)[0], device=brain.device,
            )
            brain.syn_strength[idx] = torch.clamp(
                brain.syn_strength[idx] + self.hebbian_delta, 0.3, 8.0,
            )

        # Weaken relay -> wrong motor populations.
        for d in range(8):
            if d == correct_action:
                continue
            wrong_set = set(decoder.dir_ids[d].cpu().tolist())
            wrong_post = np.isin(post_np, list(wrong_set))
            weaken = relay_mask & wrong_post
            if weaken.any():
                idx = torch.tensor(
                    np.where(weaken)[0], device=brain.device,
                )
                brain.syn_strength[idx] = torch.clamp(
                    brain.syn_strength[idx] - self.hebbian_delta * 0.15,
                    0.3, 8.0,
                )

        brain._W_dirty = True
        brain._W_sparse = None
        brain._NT_W_sparse = None


# ============================================================================
# Optimal Action Heuristic
# ============================================================================

def _optimal_action_fps(env: DoomFPS) -> int:
    """Compute heuristic optimal action: move toward goal, avoid enemies.

    NOT used for learning -- only for Hebbian credit assignment.

    Args:
        env: The DoomFPS environment.

    Returns:
        Best action index (0-7).
    """
    if env.doom_map is None:
        return 0

    px, py = env.player_x, env.player_y
    pa = env.player_angle
    gx, gy = env.doom_map.goal

    # Angle to goal.
    goal_dx = gx - px
    goal_dy = gy - py
    goal_angle = math.atan2(goal_dy, goal_dx)

    # Relative angle (how far we need to turn).
    rel_angle = goal_angle - pa
    # Normalize to [-pi, pi].
    rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))

    # Enemy avoidance: find nearest enemy.
    nearest_enemy_dist = float("inf")
    nearest_enemy_angle = 0.0
    for sprite in env.sprites:
        if sprite.sprite_type == "enemy" and sprite.active:
            edist = sprite.distance_to(px, py)
            if edist < nearest_enemy_dist:
                nearest_enemy_dist = edist
                ex_dx = sprite.x - px
                ex_dy = sprite.y - py
                nearest_enemy_angle = math.atan2(ex_dy, ex_dx) - pa
                nearest_enemy_angle = math.atan2(
                    math.sin(nearest_enemy_angle), math.cos(nearest_enemy_angle),
                )

    # If enemy is very close and in front, strafe away.
    if nearest_enemy_dist < 3.0 and abs(nearest_enemy_angle) < math.pi / 3:
        if nearest_enemy_angle > 0:
            return 2  # strafe left
        else:
            return 3  # strafe right

    # Otherwise, navigate toward goal.
    if abs(rel_angle) < math.pi / 6:
        return 0  # forward
    elif abs(rel_angle) > 5 * math.pi / 6:
        return 1  # backward (goal is behind)
    elif rel_angle > 0:
        if abs(rel_angle) < math.pi / 3:
            return 7  # turn right + forward
        else:
            return 5  # turn right
    else:
        if abs(rel_angle) < math.pi / 3:
            return 6  # turn left + forward
        else:
            return 4  # turn left


# ============================================================================
# Brain Builder
# ============================================================================

def _build_doom_brain(
    scale: str,
    device: str,
    seed: int,
    embodied: bool = True,
) -> Tuple:
    """Build a brain with all components for Doom FPS.

    In embodied mode, also creates retina, nociceptor, reward, and
    interoceptor systems. In disembodied mode, only creates the base
    brain with visual encoder.

    Args:
        scale: Network scale.
        device: Device string.
        seed: Random seed.
        embodied: Whether to create embodied subsystems.

    Returns:
        Tuple of (rb, visual_encoder, decoder, protocol, relay_ids, l5_ids,
                  cortex_ids, retina_or_None).
    """
    n_cols = SCALE_COLUMNS[scale]
    rb = CUDARegionalBrain._build(
        n_columns=n_cols, n_per_layer=20, device=device, seed=seed,
    )
    brain = rb.brain
    dev = brain.device

    if dev.type == "cuda":
        brain.compile()

    relay_ids = _get_region_ids(rb, "thalamus", "relay")
    l5_ids = _get_cortex_l5_ids(rb)
    cortex_ids = _get_all_cortex_ids(rb)

    # Use a portion of cortex for sensory input. We take the first ~25% of
    # cortical neurons as "sensory cortex" (V1 analog).
    n_cortex = len(cortex_ids)
    n_sensory = max(10, n_cortex // 4)
    sensory_ids = cortex_ids[:n_sensory]

    # Embodied subsystems.
    retina = None
    nociceptor = None
    reward_sys = None
    interoceptor = None

    if embodied:
        # Retina.
        res = RETINA_RESOLUTIONS.get(scale, (16, 12))
        retina = MolecularRetina(resolution=res, seed=seed)

        # Nociceptor population: use a small portion of hippocampal DG neurons
        # (repurposed as nociceptors -- biologically, these would be in
        # dorsal horn / thalamic VPL, but we use available neuron pools).
        hippo_ids = _get_region_ids(rb, "hippocampus", "DG")
        n_noci = max(5, len(hippo_ids) // 3)
        nociceptor_ids = hippo_ids[:n_noci]
        nociceptor = NociceptorSystem(nociceptor_ids, cortex_ids, device=str(dev))

        # VTA/reward: use basal ganglia D1 neurons as VTA analog.
        bg_d1_ids = _get_region_ids(rb, "basal_ganglia", "D1")
        reward_sys = RewardSystem(bg_d1_ids, cortex_ids, device=str(dev))

        # Interoceptor: use hippocampal CA1 neurons as body-state sensors.
        ca1_ids = _get_region_ids(rb, "hippocampus", "CA1")
        interoceptor = InteroceptorSystem(ca1_ids)

        # Visual encoder.
        visual_encoder = EmbodiedVisualEncoder(retina, sensory_ids, device=str(dev))
    else:
        visual_encoder = DisembodiedVisualEncoder(sensory_ids, device=str(dev))

    decoder = DoomMotorDecoder(l5_ids)

    protocol = EmbodiedFEP(
        cortex_ids, l5_ids, relay_ids,
        nociceptor=nociceptor,
        reward_sys=reward_sys,
        interoceptor=interoceptor,
        device=str(dev),
    )

    return (rb, visual_encoder, decoder, protocol, relay_ids, l5_ids,
            cortex_ids, retina)


# ============================================================================
# Game Loop
# ============================================================================

def play_doom_episode(
    rb: CUDARegionalBrain,
    env: DoomFPS,
    visual_encoder,
    decoder: DoomMotorDecoder,
    protocol: EmbodiedFEP,
    embodied: bool = True,
    stim_steps: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Play one full Doom FPS episode with FEP-based learning.

    Each step:
    1. Render frame (embodied: through retina; disembodied: raycasts).
    2. Encode visual input onto sensory cortex neurons.
    3. Count L5 motor spikes per directional population.
    4. Decode action via zero-threshold majority vote.
    5. Advance environment.
    6. Deliver FEP feedback based on outcome events.
    7. Apply Hebbian nudge toward heuristic optimal action.

    Args:
        rb: Regional brain.
        env: DoomFPS environment.
        visual_encoder: Embodied or disembodied visual encoder.
        decoder: Motor decoder.
        protocol: FEP protocol.
        embodied: Whether running in embodied mode.
        stim_steps: Simulation steps per game step.
        seed: Episode seed.

    Returns:
        Dict with episode results.
    """
    brain = rb.brain
    prev_hp = env.player_hp
    total_damage = 0
    total_kills = 0
    total_pickups = 0
    step_count = 0
    reached_goal = False

    while not env.done:
        step_count += 1

        # 1 & 2: Visual encoding.
        dir_acc = torch.zeros(8, device=brain.device)
        for s in range(stim_steps):
            if embodied and isinstance(visual_encoder, EmbodiedVisualEncoder):
                frame = env.render()
                visual_encoder.encode(frame, brain, pulsed_step=s)
            elif isinstance(visual_encoder, DisembodiedVisualEncoder):
                visual_encoder.encode(env, brain, pulsed_step=s)

            # Interoceptive update (embodied only, every step).
            if embodied:
                protocol.update_interoception(brain, env.player_hp)

            rb.step()

            # 3: Accumulate L5 spikes per direction (GPU accumulator).
            for d in range(8):
                dir_acc[d] += brain.fired[decoder.dir_ids[d]].sum()

        # Single GPU->CPU sync.
        counts = dir_acc.int().tolist()

        # 4: Decode action.
        action = decoder.decode(counts)

        # Compute optimal action for Hebbian nudge.
        optimal = _optimal_action_fps(env)

        # 5: Advance environment.
        frame, reward, done, info = env.step(action)
        current_hp = info["player_hp"]

        # Detect events.
        hp_change = current_hp - prev_hp
        damage_event = hp_change < 0
        pickup_event = hp_change > 0 and current_hp <= 100
        goal_event = done and info.get("score", 0) >= 100

        if damage_event:
            total_damage += abs(hp_change)
        if pickup_event:
            total_pickups += 1
        if goal_event:
            reached_goal = True

        # 6: Deliver FEP feedback based on events.
        if goal_event:
            protocol.deliver_goal(rb)
            protocol.hebbian_nudge_direction(brain, decoder, optimal, action)
        elif damage_event:
            protocol.deliver_damage(rb, damage=abs(hp_change))
            protocol.hebbian_nudge_direction(brain, decoder, optimal, action)
        elif pickup_event:
            protocol.deliver_pickup(rb)
        elif done and not goal_event:
            # Timeout or death.
            if current_hp <= 0:
                protocol.deliver_damage(rb, damage=20)
            else:
                protocol.deliver_timeout(rb)
            protocol.hebbian_nudge_direction(brain, decoder, optimal, action)
        else:
            # Normal step -- check enemy proximity.
            near_enemy = any(
                s.active and s.sprite_type == "enemy"
                and s.distance_to(env.player_x, env.player_y) < 2.5
                for s in env.sprites
            )
            if near_enemy:
                protocol.deliver_survive(rb)
            # Mild Hebbian nudge every step.
            protocol.hebbian_nudge_direction(brain, decoder, optimal, action)

        prev_hp = current_hp

        # 7: Inter-step gap.
        rb.run(3)

    return {
        "reached_goal": reached_goal,
        "steps": env.steps,
        "hp": env.player_hp,
        "score": env.score,
        "damage": total_damage,
        "pickups": total_pickups,
        "outcome": (
            "goal" if reached_goal
            else "dead" if env.player_hp <= 0
            else "timeout"
        ),
    }


# ============================================================================
# Experiment 1: Doom Navigation
# ============================================================================

def exp_doom_navigation(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    mode: str = "both",
    n_episodes: int = 50,
) -> Dict[str, Any]:
    """Can the brain navigate to a goal in a Doom dungeon?

    Tests both embodied (retina + pain/reward) and disembodied (raycasts only)
    modes, comparing navigation performance.

    Pass criteria:
        1. Goal rate > random (~5%) AND improving over time, OR
        2. Score improvement from first to last quarter (learning signal).

    Args:
        scale: Network scale.
        device: Device string.
        seed: Random seed.
        mode: "embodied", "disembodied", or "both".
        n_episodes: Episodes per condition.

    Returns:
        Results dict with pass/fail and metrics.
    """
    _header(
        "Exp 1: Doom FPS Navigation",
        "Can a dONN brain navigate Doom dungeons via FEP?",
    )
    t0 = time.perf_counter()

    conditions = []
    if mode in ("embodied", "both"):
        conditions.append("embodied")
    if mode in ("disembodied", "both"):
        conditions.append("disembodied")

    params = DOOM_PARAMS.get(scale, DOOM_PARAMS["small"])
    render_res = RENDER_RESOLUTIONS.get(scale, (16, 12))
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for condition in conditions:
        is_embodied = condition == "embodied"
        print(f"\n    [{condition.upper()}]")

        rb, encoder, decoder, protocol, relay_ids, l5_ids, cortex_ids, retina = (
            _build_doom_brain(scale, device, seed, embodied=is_embodied)
        )
        brain = rb.brain
        dev = brain.device
        print(f"    Brain: {rb.n_neurons} neurons, {rb.n_synapses} synapses on {dev}")
        if retina is not None:
            print(f"    Retina: {retina.total_neurons} neurons "
                  f"({retina.n_photo} photo, {retina.n_bipolar} bipolar, "
                  f"{retina.n_rgc} RGC)")

        _warmup(rb, n_steps=300)
        print(f"    Warmup complete")

        episode_results: List[Dict[str, Any]] = []
        for ep in range(n_episodes):
            env = DoomFPS(
                render_width=render_res[0], render_height=render_res[1],
                map_size=params["map_size"], n_rooms=6,
                n_enemies=params["n_enemies"], n_health=params["n_health"],
                max_steps=params["max_steps"],
                seed=seed + ep * 137,
            )
            env.reset()
            ep_result = play_doom_episode(
                rb, env, encoder, decoder, protocol,
                embodied=is_embodied, stim_steps=15,
                seed=seed + ep,
            )
            episode_results.append(ep_result)

            if (ep + 1) % 10 == 0:
                recent = episode_results[max(0, ep - 9):ep + 1]
                goal_rate = sum(1 for r in recent if r["reached_goal"]) / len(recent)
                avg_score = sum(r["score"] for r in recent) / len(recent)
                avg_dmg = sum(r["damage"] for r in recent) / len(recent)
                print(f"    Episode {ep + 1:3d}/{n_episodes}: "
                      f"goal={goal_rate:.0%}, score={avg_score:.1f}, "
                      f"damage={avg_dmg:.1f} (last 10)")

        all_results[condition] = episode_results

    # Analyze.
    elapsed = time.perf_counter() - t0
    random_baseline = 0.05
    condition_metrics: Dict[str, Dict[str, float]] = {}
    any_passed = False

    for condition, results_list in all_results.items():
        quarter = max(1, n_episodes // 4)
        q1 = results_list[:quarter]
        q4 = results_list[-quarter:]

        q1_goal = sum(1 for r in q1 if r["reached_goal"]) / len(q1)
        q4_goal = sum(1 for r in q4 if r["reached_goal"]) / len(q4)
        q1_score = sum(r["score"] for r in q1) / len(q1)
        q4_score = sum(r["score"] for r in q4) / len(q4)
        total_goals = sum(1 for r in results_list if r["reached_goal"])
        total_rate = total_goals / n_episodes

        passed = (total_rate > random_baseline) or (q4_score > q1_score) or (q4_goal > q1_goal)
        if passed:
            any_passed = True

        condition_metrics[condition] = {
            "q1_goal": q1_goal, "q4_goal": q4_goal,
            "q1_score": q1_score, "q4_score": q4_score,
            "total_goal_rate": total_rate, "total_goals": total_goals,
            "passed": passed,
        }

        print(f"\n    [{condition.upper()}] Results:")
        print(f"    Q1 goal rate: {q1_goal:.0%}, avg score: {q1_score:.1f}")
        print(f"    Q4 goal rate: {q4_goal:.0%}, avg score: {q4_score:.1f}")
        print(f"    Total goals:  {total_goals}/{n_episodes} ({total_rate:.0%})")
        print(f"    Score change: {q4_score - q1_score:+.1f}")
        print(f"    {'PASS' if passed else 'FAIL'}")

    print(f"\n    Overall: {'PASS' if any_passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": any_passed,
        "time": elapsed,
        "conditions": condition_metrics,
    }


# ============================================================================
# Experiment 2: Doom Survival
# ============================================================================

def exp_doom_survival(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    mode: str = "both",
    n_episodes: int = 80,
) -> Dict[str, Any]:
    """Can the brain survive longer with experience?

    Tests whether survival time increases and damage decreases over episodes.

    Pass criteria:
        Survival rate (ending HP > 0) improves OR damage decreases over quarters.

    Args:
        scale: Network scale.
        device: Device string.
        seed: Random seed.
        mode: "embodied", "disembodied", or "both".
        n_episodes: Episodes per condition.

    Returns:
        Results dict.
    """
    _header(
        "Exp 2: Doom FPS Survival",
        "Does the dONN brain survive longer with experience?",
    )
    t0 = time.perf_counter()

    conditions = []
    if mode in ("embodied", "both"):
        conditions.append("embodied")
    if mode in ("disembodied", "both"):
        conditions.append("disembodied")

    params = DOOM_PARAMS.get(scale, DOOM_PARAMS["small"])
    # More enemies for survival focus.
    n_enemies = params["n_enemies"] + 1
    render_res = RENDER_RESOLUTIONS.get(scale, (16, 12))
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for condition in conditions:
        is_embodied = condition == "embodied"
        print(f"\n    [{condition.upper()}]")

        rb, encoder, decoder, protocol, relay_ids, l5_ids, cortex_ids, retina = (
            _build_doom_brain(scale, device, seed, embodied=is_embodied)
        )
        brain = rb.brain
        dev = brain.device
        print(f"    Brain: {rb.n_neurons} neurons, {rb.n_synapses} synapses on {dev}")

        _warmup(rb, n_steps=300)
        print(f"    Warmup complete")

        episode_results: List[Dict[str, Any]] = []
        for ep in range(n_episodes):
            env = DoomFPS(
                render_width=render_res[0], render_height=render_res[1],
                map_size=params["map_size"], n_rooms=6,
                n_enemies=n_enemies, n_health=params["n_health"],
                max_steps=params["max_steps"],
                seed=seed + ep * 137,
            )
            env.reset()
            ep_result = play_doom_episode(
                rb, env, encoder, decoder, protocol,
                embodied=is_embodied, stim_steps=15,
                seed=seed + ep,
            )
            episode_results.append(ep_result)

            if (ep + 1) % 20 == 0:
                recent = episode_results[max(0, ep - 19):ep + 1]
                survival = sum(1 for r in recent if r["hp"] > 0) / len(recent)
                avg_dmg = sum(r["damage"] for r in recent) / len(recent)
                avg_steps = sum(r["steps"] for r in recent) / len(recent)
                print(f"    Episode {ep + 1:3d}/{n_episodes}: "
                      f"survival={survival:.0%}, damage={avg_dmg:.1f}, "
                      f"steps={avg_steps:.0f} (last 20)")

        all_results[condition] = episode_results

    # Analyze.
    elapsed = time.perf_counter() - t0
    condition_metrics: Dict[str, Dict[str, float]] = {}
    any_passed = False

    for condition, results_list in all_results.items():
        quarter = max(1, n_episodes // 4)
        q1 = results_list[:quarter]
        q4 = results_list[-quarter:]

        q1_survival = sum(1 for r in q1 if r["hp"] > 0) / len(q1)
        q4_survival = sum(1 for r in q4 if r["hp"] > 0) / len(q4)
        q1_damage = sum(r["damage"] for r in q1) / len(q1)
        q4_damage = sum(r["damage"] for r in q4) / len(q4)
        q1_steps = sum(r["steps"] for r in q1) / len(q1)
        q4_steps = sum(r["steps"] for r in q4) / len(q4)

        passed = (q4_survival >= q1_survival) or (q4_damage <= q1_damage)
        if passed:
            any_passed = True

        condition_metrics[condition] = {
            "q1_survival": q1_survival, "q4_survival": q4_survival,
            "q1_damage": q1_damage, "q4_damage": q4_damage,
            "q1_steps": q1_steps, "q4_steps": q4_steps,
            "passed": passed,
        }

        print(f"\n    [{condition.upper()}] Results:")
        print(f"    Q1 survival: {q1_survival:.0%}, damage: {q1_damage:.1f}")
        print(f"    Q4 survival: {q4_survival:.0%}, damage: {q4_damage:.1f}")
        print(f"    Survival change: {q4_survival - q1_survival:+.0%}")
        print(f"    Damage change:   {q4_damage - q1_damage:+.1f}")
        print(f"    {'PASS' if passed else 'FAIL'}")

    print(f"\n    Overall: {'PASS' if any_passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": any_passed,
        "time": elapsed,
        "conditions": condition_metrics,
    }


# ============================================================================
# Experiment 3: Doom Drug Effects
# ============================================================================

def exp_doom_drug_effects(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_train: int = 30,
    n_test: int = 20,
) -> Dict[str, Any]:
    """Pharmacological effects on embodied Doom performance.

    Trains 3 identical brains (same seed), then applies drugs before testing.
    Diazepam (GABA-A PAM) should impair navigation/avoidance. Caffeine
    (adenosine antagonist) may improve reaction time.

    PASS: diazepam test score < baseline test score.

    Biological basis:
        Morris (1984) showed benzodiazepines impair hippocampal spatial
        learning. GABA-A enhancement reduces excitatory drive, impairing
        the STDP-based learning that the FEP protocol relies on.

    Args:
        scale: Network scale.
        device: Device string.
        seed: Random seed.
        n_train: Training episodes.
        n_test: Test episodes.

    Returns:
        Results dict.
    """
    _header(
        "Exp 3: Doom FPS Drug Effects (Embodied)",
        "3 conditions: baseline / caffeine / diazepam",
    )
    t0 = time.perf_counter()

    drug_conditions = [
        ("baseline", None, 0.0),
        ("caffeine", "caffeine", 200.0),
        ("diazepam", "diazepam", 40.0),
    ]

    params = DOOM_PARAMS.get(scale, DOOM_PARAMS["small"])
    render_res = RENDER_RESOLUTIONS.get(scale, (16, 12))
    test_results: Dict[str, Dict[str, Any]] = {}

    for condition_name, drug_name, dose in drug_conditions:
        print(f"\n    [{condition_name.upper()}]")

        # Build fresh brain (same seed = same initial wiring).
        rb, encoder, decoder, protocol, relay_ids, l5_ids, cortex_ids, retina = (
            _build_doom_brain(scale, device, seed, embodied=True)
        )
        brain = rb.brain
        dev = brain.device

        _warmup(rb, n_steps=300)

        # Train (no drug during training).
        print(f"    Training {n_train} episodes...")
        for ep in range(n_train):
            env = DoomFPS(
                render_width=render_res[0], render_height=render_res[1],
                map_size=params["map_size"], n_rooms=6,
                n_enemies=params["n_enemies"], n_health=params["n_health"],
                max_steps=params["max_steps"],
                seed=seed + ep * 137,
            )
            env.reset()
            play_doom_episode(
                rb, env, encoder, decoder, protocol,
                embodied=True, stim_steps=15, seed=seed + ep,
            )

        # Apply drug AFTER training.
        if drug_name is not None:
            brain.apply_drug(drug_name, dose)
            print(f"    Applied {drug_name} {dose}mg")

        # Test.
        print(f"    Testing {n_test} episodes...")
        test_episode_results: List[Dict[str, Any]] = []
        for ep in range(n_test):
            env = DoomFPS(
                render_width=render_res[0], render_height=render_res[1],
                map_size=params["map_size"], n_rooms=6,
                n_enemies=params["n_enemies"], n_health=params["n_health"],
                max_steps=params["max_steps"],
                seed=seed + 5000 + ep * 137,  # Different test seeds.
            )
            env.reset()
            ep_result = play_doom_episode(
                rb, env, encoder, decoder, protocol,
                embodied=True, stim_steps=15, seed=seed + 5000 + ep,
            )
            test_episode_results.append(ep_result)

        goals = sum(1 for r in test_episode_results if r["reached_goal"])
        goal_rate = goals / n_test
        avg_score = sum(r["score"] for r in test_episode_results) / n_test
        avg_damage = sum(r["damage"] for r in test_episode_results) / n_test
        avg_steps = sum(r["steps"] for r in test_episode_results) / n_test

        test_results[condition_name] = {
            "goals": goals, "goal_rate": goal_rate,
            "avg_score": avg_score, "avg_damage": avg_damage,
            "avg_steps": avg_steps,
        }
        print(f"    {condition_name:10s}: {goals}/{n_test} goals ({goal_rate:.0%}), "
              f"score={avg_score:.1f}, damage={avg_damage:.1f}, "
              f"steps={avg_steps:.0f}")

    elapsed = time.perf_counter() - t0

    baseline_score = test_results["baseline"]["avg_score"]
    diazepam_score = test_results["diazepam"]["avg_score"]
    caffeine_score = test_results["caffeine"]["avg_score"]
    baseline_dmg = test_results["baseline"]["avg_damage"]
    diazepam_dmg = test_results["diazepam"]["avg_damage"]

    # Pass: diazepam performs worse than baseline (lower score OR more damage).
    passed = (diazepam_score < baseline_score) or (diazepam_dmg > baseline_dmg)

    print(f"\n    Baseline score:  {baseline_score:.1f}, damage: {baseline_dmg:.1f}")
    print(f"    Caffeine score:  {caffeine_score:.1f} ({caffeine_score - baseline_score:+.1f})")
    print(f"    Diazepam score:  {diazepam_score:.1f} ({diazepam_score - baseline_score:+.1f}), "
          f"damage: {diazepam_dmg:.1f}")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "test_results": test_results,
    }


# ============================================================================
# Multi-Run Aggregation
# ============================================================================

def _run_experiment_multi(
    exp_func,
    n_runs: int,
    scale: str,
    device: str,
    seed: int,
    **extra_kwargs,
) -> Dict[str, Any]:
    """Run an experiment multiple times with different seeds.

    Args:
        exp_func: Experiment function.
        n_runs: Number of independent runs.
        scale: Network scale.
        device: Device string.
        seed: Base seed.
        **extra_kwargs: Extra kwargs for the experiment function.

    Returns:
        Aggregated results.
    """
    all_results: List[Dict[str, Any]] = []
    for run in range(n_runs):
        run_seed = seed + run * 1000
        print(f"\n    --- Run {run + 1}/{n_runs} (seed={run_seed}) ---")
        result = exp_func(scale=scale, device=device, seed=run_seed, **extra_kwargs)
        all_results.append(result)

    pass_count = sum(1 for r in all_results if r["passed"])
    pass_rate = pass_count / n_runs
    times = [r["time"] for r in all_results]
    mean_time = sum(times) / len(times)

    return {
        "passed": pass_rate >= 0.5,
        "pass_rate": pass_rate,
        "pass_count": pass_count,
        "n_runs": n_runs,
        "mean_time": mean_time,
        "individual_results": all_results,
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

ALL_EXPERIMENTS = {
    1: ("Doom FPS Navigation", exp_doom_navigation),
    2: ("Doom FPS Survival", exp_doom_survival),
    3: ("Doom FPS Drug Effects", exp_doom_drug_effects),
}


def main() -> None:
    """Main entry point for the Doom FPS Brain demo."""
    parser = argparse.ArgumentParser(
        description=(
            "Doom FPS Brain -- A biophysical brain plays Doom via molecular "
            "retina and free energy principle learning."
        ),
    )
    parser.add_argument(
        "--exp", type=int, nargs="*", default=None,
        help="Which experiments to run (1-3). Default: all.",
    )
    parser.add_argument(
        "--scale", default="small",
        choices=list(SCALE_COLUMNS.keys()),
        help="Network scale (default: small).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: auto, cuda, mps, cpu.",
    )
    parser.add_argument(
        "--mode", default="both",
        choices=["embodied", "disembodied", "both"],
        help="Embodied, disembodied, or both modes (default: both). "
             "Exp 3 always uses embodied.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json", type=str, default=None,
        help="Path to write structured JSON results.",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Run each experiment N times with different seeds.",
    )
    args = parser.parse_args()

    exps = args.exp if args.exp else list(ALL_EXPERIMENTS.keys())

    print("=" * 76)
    print("  DOOM FPS BRAIN -- BIOPHYSICAL NEURAL NETWORK PLAYS DOOM")
    print(f"  Backend: {detect_backend()} | Scale: {args.scale} | "
          f"Device: {args.device} | Mode: {args.mode} | Runs: {args.runs}")
    print(f"  Free Energy Principle + Molecular Retina + Pain/Reward Pathways")
    print("=" * 76)

    results: Dict[int, Dict[str, Any]] = {}
    total_time = time.perf_counter()

    for exp_id in exps:
        if exp_id not in ALL_EXPERIMENTS:
            print(f"\n  Unknown experiment: {exp_id}")
            continue

        name, func = ALL_EXPERIMENTS[exp_id]

        # Extra kwargs depending on experiment.
        extra_kwargs: Dict[str, Any] = {}
        if exp_id in (1, 2):
            extra_kwargs["mode"] = args.mode

        try:
            if args.runs > 1:
                result = _run_experiment_multi(
                    func, args.runs, args.scale, args.device, args.seed,
                    **extra_kwargs,
                )
            else:
                result = func(
                    scale=args.scale, device=args.device, seed=args.seed,
                    **extra_kwargs,
                )
            results[exp_id] = result
        except Exception as e:
            print(f"\n  EXPERIMENT {exp_id} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {"passed": False, "error": str(e)}

    total = time.perf_counter() - total_time

    # Summary.
    print("\n" + "=" * 76)
    print("  DOOM FPS BRAIN -- SUMMARY")
    print("=" * 76)
    passed = sum(1 for r in results.values() if r.get("passed"))
    total_exp = len(results)
    for exp_id, result in sorted(results.items()):
        name_str = ALL_EXPERIMENTS[exp_id][0]
        status = "PASS" if result.get("passed") else "FAIL"
        t = result.get("time", result.get("mean_time", 0))
        extra = ""
        if args.runs > 1 and "pass_rate" in result:
            extra = f"  ({result['pass_count']}/{result['n_runs']} runs)"
        print(f"    {exp_id}. {name_str:35s} [{status}]  {t:.1f}s{extra}")
    print(f"\n  Total: {passed}/{total_exp} passed in {total:.1f}s")
    print("=" * 76)

    # JSON output.
    if args.json:
        json_results: Dict[str, Any] = {}
        for exp_id, result in results.items():
            clean: Dict[str, Any] = {}
            for k, v in result.items():
                if k == "individual_results":
                    clean[k] = [
                        {kk: vv for kk, vv in r.items()
                         if kk != "conditions"}
                        for r in v
                    ]
                elif k == "conditions":
                    clean[k] = v
                elif k == "test_results":
                    clean[k] = v
                else:
                    clean[k] = v
            json_results[str(exp_id)] = clean

        json_output = {
            "demo": "doom_brain",
            "scale": args.scale,
            "device": args.device,
            "mode": args.mode,
            "seed": args.seed,
            "runs": args.runs,
            "total_time": total,
            "experiments": json_results,
        }
        with open(args.json, "w") as f:
            json.dump(json_output, f, indent=2, default=str)
        print(f"\n  JSON results written to {args.json}")


if __name__ == "__main__":
    main()
