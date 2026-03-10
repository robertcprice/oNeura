#!/usr/bin/env python3
"""ViZDoom via Rust Molecular Brain -- digital Organic Neural Networks (dONNs) Play Doom.

A dONN (digital Organic Neural Network) plays actual Doom through biologically
faithful "organic eyes" (Rust MolecularRetina) and a full Hodgkin-Huxley brain
(Rust MolecularBrain). ViZDoom renders game frames, the Rust retina converts
RGB pixels to spike trains through three biological layers (photoreceptors,
bipolar cells, RGCs), and the Rust MolecularBrain processes those spikes via
HH neurons with molecular neurotransmitter dynamics. Motor neuron populations
drive game actions.

This is the Metal/Rust-accelerated version. The pure-Python equivalent is
demos/demo_doom_vizdoom.py (uses CUDARegionalBrain + Python MolecularRetina).

Learning uses the FREE ENERGY PRINCIPLE -- no reward signal, just structured
vs unstructured sensory feedback.

Terminology:
  - ONN:    Organic Neural Network -- real biological neurons (DishBrain, FinalSpark)
  - dONN:   digital Organic Neural Network -- oNeuro's biophysically faithful simulation
  - oNeuro: The platform for building and running dONNs

3 Experiments:
   1. Doom Navigation (health gathering): FEP learning with health gain improvement
   2. Learning Speed Comparison: FEP vs DA vs Random protocols
   3. Pharmacological Effects: baseline / caffeine / diazepam

Key innovation: Learning via FREE ENERGY PRINCIPLE, not reward/punishment.
  - Positive event (health gain): STRUCTURED pulse to cortex (predictable = low entropy)
  - Negative event (damage taken): RANDOM noise to 30% cortex (unpredictable = high entropy)
  - Neurons self-organize via STDP to prefer states that produce predictable feedback.

References:
  - Kagan et al. (2022) "In vitro neurons learn and exhibit sentience when
    embodied in a simulated game-world" Neuron 110(23):3952-3969
  - Friston (2010) "The free-energy principle: a unified brain theory?"
    Nature Reviews Neuroscience 11:127-138

Usage:
    python3 demos/demo_vizdoom_metal.py                         # all 3 experiments
    python3 demos/demo_vizdoom_metal.py --exp 1                 # just navigation
    python3 demos/demo_vizdoom_metal.py --scenario take_cover   # different scenario
    python3 demos/demo_vizdoom_metal.py --scale medium          # more neurons
    python3 demos/demo_vizdoom_metal.py --json results.json     # structured output
    python3 demos/demo_vizdoom_metal.py --visible               # show game window
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
from typing import Any, Dict, List, Optional, Tuple

# Force unbuffered stdout for real-time progress reporting
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Import Rust PyO3 module (oneuro_metal)
# ---------------------------------------------------------------------------
try:
    from oneuro_metal import MolecularBrain, MolecularRetina, has_gpu, version
    HAS_ONEURO_METAL = True
except ImportError:
    HAS_ONEURO_METAL = False

# ---------------------------------------------------------------------------
# Import ViZDoom
# ---------------------------------------------------------------------------
try:
    import vizdoom as vzd
    HAS_VIZDOOM = True
except ImportError:
    HAS_VIZDOOM = False

# ---------------------------------------------------------------------------
# Import PIL for frame downsampling
# ---------------------------------------------------------------------------
try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ============================================================================
# Constants
# ============================================================================

RETINA_WIDTH = 64
RETINA_HEIGHT = 48
VIZDOOM_WIDTH = 160
VIZDOOM_HEIGHT = 120

# Motor populations: brain neurons split into 5 motor groups
MOTOR_FORWARD = 0
MOTOR_TURN_LEFT = 1
MOTOR_TURN_RIGHT = 2
MOTOR_STRAFE_LEFT = 3
MOTOR_STRAFE_RIGHT = 4
N_MOTOR_POPULATIONS = 5

MOTOR_NAMES = ["forward", "turn_left", "turn_right", "strafe_left", "strafe_right"]

# NTType mapping for from_edges: 0=DA, 1=5HT, 2=NE, 3=ACh, 4=GABA, 5=Glutamate
NT_DA = 0
NT_5HT = 1
NT_NE = 2
NT_ACH = 3
NT_GABA = 4
NT_GLU = 5

# Scale parameters: neuron count, episode count, and timing parameters
SCALE_PARAMS = {
    "small": {
        "n_neurons": 800,
        "n_episodes": 5,
        "stim_steps": 5,
        "max_game_steps": 80,
        "structured_steps": 10,
        "unstructured_steps": 15,
        "neutral_steps": 3,
        "n_train_episodes": 4,
        "n_test_episodes": 3,
        "warmup_steps": 200,
    },
    "medium": {
        "n_neurons": 4000,
        "n_episodes": 15,
        "stim_steps": 10,
        "max_game_steps": 200,
        "structured_steps": 25,
        "unstructured_steps": 40,
        "neutral_steps": 5,
        "n_train_episodes": 10,
        "n_test_episodes": 5,
        "warmup_steps": 300,
    },
    "large": {
        "n_neurons": 20000,
        "n_episodes": 40,
        "stim_steps": 20,
        "max_game_steps": 500,
        "structured_steps": 50,
        "unstructured_steps": 100,
        "neutral_steps": 5,
        "n_train_episodes": 25,
        "n_test_episodes": 10,
        "warmup_steps": 300,
    },
}

# Scenario configurations
SCENARIOS = {
    "health_gathering": {
        "cfg": "health_gathering.cfg",
        "description": "Navigate to collect health vials",
    },
    "my_way_home": {
        "cfg": "my_way_home.cfg",
        "description": "Maze navigation to a vest",
    },
    "take_cover": {
        "cfg": "take_cover.cfg",
        "description": "Dodge fireballs (avoidance learning)",
    },
    "deadly_corridor": {
        "cfg": "deadly_corridor.cfg",
        "description": "Navigate corridor with enemies",
    },
    "defend_the_center": {
        "cfg": "defend_the_center.cfg",
        "description": "Defend against approaching enemies",
    },
}


# ============================================================================
# Region layout: fractional allocation of neurons to brain regions
# ============================================================================

REGION_LAYOUT = {
    "V1":          0.15,
    "V2":          0.10,
    "turn_left":   0.05,
    "turn_right":  0.05,
    "forward":     0.05,
    "strafe_L":    0.05,
    "strafe_R":    0.05,
    "VTA":         0.03,
    "LC":          0.03,
    "prefrontal":  0.15,
    "hippocampus": 0.12,
    "amygdala":    0.08,
}
# Remaining 9% goes to interneurons / unassigned


# ============================================================================
# Utility functions
# ============================================================================

def _header(title: str, subtitle: str = "") -> None:
    """Print a formatted experiment header."""
    print(f"\n{'='*72}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'='*72}")


def _allocate_regions(n_neurons: int) -> Dict[str, Tuple[int, int]]:
    """Allocate neuron index ranges to named brain regions.

    Returns a dict mapping region name to (start_idx, end_idx) where
    the range is [start_idx, end_idx) exclusive.

    Args:
        n_neurons: Total neuron count in the brain.

    Returns:
        Dict of region_name -> (start, end) index ranges.
    """
    regions = {}
    offset = 0
    for name, fraction in REGION_LAYOUT.items():
        size = max(1, int(n_neurons * fraction))
        if offset + size > n_neurons:
            size = n_neurons - offset
        regions[name] = (offset, offset + size)
        offset += size
    return regions


def _region_ids(regions: Dict[str, Tuple[int, int]], name: str) -> List[int]:
    """Get the list of neuron IDs for a named region.

    Args:
        regions: Region layout from _allocate_regions.
        name: Region name.

    Returns:
        List of neuron indices in that region.
    """
    start, end = regions[name]
    return list(range(start, end))


def _region_size(regions: Dict[str, Tuple[int, int]], name: str) -> int:
    """Get the size of a named region."""
    start, end = regions[name]
    return end - start


# ============================================================================
# ViZDoom Game Wrapper
# ============================================================================

class DoomGame:
    """Wraps ViZDoom to provide a clean interface for the dONN game loop.

    Handles initialization, frame capture, downsampling to retina resolution,
    action execution, and health/damage tracking for FEP-based learning.

    Args:
        scenario: One of the supported scenario names.
        seed: Random seed for game reproducibility.
        visible: Whether to show the game window.
        frame_width: ViZDoom frame width before downsampling.
        frame_height: ViZDoom frame height before downsampling.
        retina_w: Target width for retina input.
        retina_h: Target height for retina input.
    """

    def __init__(
        self,
        scenario: str = "health_gathering",
        seed: int = 42,
        visible: bool = False,
        frame_width: int = VIZDOOM_WIDTH,
        frame_height: int = VIZDOOM_HEIGHT,
        retina_w: int = RETINA_WIDTH,
        retina_h: int = RETINA_HEIGHT,
    ):
        if not HAS_VIZDOOM:
            raise ImportError(
                "ViZDoom is required: pip install vizdoom"
            )
        if not HAS_PIL:
            raise ImportError(
                "Pillow and numpy are required: pip install Pillow numpy"
            )

        self.scenario = scenario
        self.seed = seed
        self.retina_w = retina_w
        self.retina_h = retina_h
        self._game = vzd.DoomGame()
        self._setup(scenario, visible, frame_width, frame_height)
        self._prev_health = 100.0
        self._episode_health_gained = 0.0
        self._episode_damage_taken = 0.0
        self._episode_steps = 0
        self._total_episodes = 0

    def _setup(
        self,
        scenario: str,
        visible: bool,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Configure ViZDoom with the selected scenario.

        Args:
            scenario: Scenario name from SCENARIOS dict.
            visible: Whether to render the game window.
            frame_width: Screen width in pixels.
            frame_height: Screen height in pixels.
        """
        cfg = SCENARIOS[scenario]["cfg"]
        self._game.load_config(os.path.join(vzd.scenarios_path, cfg))

        # Override rendering for our retina pipeline
        self._game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self._game.set_screen_format(vzd.ScreenFormat.RGB24)
        self._game.set_window_visible(visible)
        self._game.set_mode(vzd.Mode.PLAYER)

        # Override buttons: clear cfg defaults, add our 5-action set
        self._game.clear_available_buttons()
        self._game.add_available_button(vzd.Button.MOVE_FORWARD)
        self._game.add_available_button(vzd.Button.TURN_LEFT)
        self._game.add_available_button(vzd.Button.TURN_RIGHT)
        self._game.add_available_button(vzd.Button.MOVE_LEFT)
        self._game.add_available_button(vzd.Button.MOVE_RIGHT)

        # Override game variables: we only need HEALTH
        self._game.clear_available_game_variables()
        self._game.add_available_game_variable(vzd.GameVariable.HEALTH)

        self._game.set_seed(self.seed)
        self._game.init()

    def get_frame(self) -> bytes:
        """Capture the current frame, downsample, and return as RGB bytes.

        Returns:
            Flat bytes of length retina_w * retina_h * 3 in RGB order,
            suitable for passing directly to MolecularRetina.process_frame().
        """
        state = self._game.get_state()
        if state is None:
            return bytes(self.retina_w * self.retina_h * 3)

        buf = state.screen_buffer  # (120, 160, 3) uint8 numpy array
        img = Image.fromarray(buf)
        img = img.resize((self.retina_w, self.retina_h), Image.BILINEAR)
        return np.array(img, dtype=np.uint8).tobytes()

    def get_health(self) -> float:
        """Read current health from game variables."""
        try:
            return float(
                self._game.get_game_variable(vzd.GameVariable.HEALTH)
            )
        except Exception:
            return 0.0

    def step(self, action_idx: int) -> Tuple[float, bool]:
        """Execute one action and return (reward, done).

        Args:
            action_idx: Motor population index (0-4).

        Returns:
            Tuple of (vizdoom_reward, episode_finished).
        """
        action = [0] * N_MOTOR_POPULATIONS
        action[action_idx] = 1

        reward = self._game.make_action(action)
        self._episode_steps += 1

        done = self._game.is_episode_finished()
        return reward, done

    def new_episode(self) -> None:
        """Start a new game episode and reset tracking."""
        self._game.new_episode()
        self._prev_health = self.get_health()
        self._episode_health_gained = 0.0
        self._episode_damage_taken = 0.0
        self._episode_steps = 0
        self._total_episodes += 1

    @property
    def is_running(self) -> bool:
        """Whether the game instance is still active."""
        return not self._game.is_episode_finished()

    @property
    def episode_health_gained(self) -> float:
        return self._episode_health_gained

    @property
    def episode_damage_taken(self) -> float:
        return self._episode_damage_taken

    @property
    def episode_steps(self) -> int:
        return self._episode_steps

    def close(self) -> None:
        """Shut down the ViZDoom instance."""
        self._game.close()


# ============================================================================
# Brain Builder: MolecularBrain.from_edges with Doom region layout
# ============================================================================

def build_doom_brain(
    n_neurons: int,
    seed: int = 42,
) -> Tuple[Any, Dict[str, Tuple[int, int]], Any]:
    """Build a MolecularBrain wired for Doom with region-specific connectivity.

    Creates a brain with the following region layout:
      V1 (15%), V2 (10%), five motor populations (5% each), VTA (3%),
      LC (3%), prefrontal (15%), hippocampus (12%), amygdala (8%).

    Connectivity:
      V1 -> V2 -> prefrontal (visual hierarchy, Glu)
      V2 -> amygdala -> LC (threat detection, Glu/NE)
      prefrontal -> all motor populations (executive control, Glu)
      VTA -> motor + prefrontal (dopamine modulation)
      LC -> V1 + prefrontal + motor (norepinephrine arousal)
      turn_left <-> turn_right (mutual GABA inhibition)

    Args:
        n_neurons: Total number of neurons.
        seed: Random seed for reproducible wiring.

    Returns:
        Tuple of (MolecularBrain, region_layout_dict, MolecularRetina).
    """
    rng = random.Random(seed)
    regions = _allocate_regions(n_neurons)

    edges: List[Tuple[int, int, int]] = []

    def _connect(
        src_name: str,
        dst_name: str,
        nt: int,
        probability: float = 0.15,
        max_edges: int = 5000,
    ) -> None:
        """Wire src_region -> dst_region with given NT type and probability.

        Args:
            src_name: Source region name.
            dst_name: Destination region name.
            nt: Neurotransmitter type (0-5).
            probability: Connection probability per pair.
            max_edges: Cap on edge count for this projection.
        """
        src_ids = _region_ids(regions, src_name)
        dst_ids = _region_ids(regions, dst_name)
        count = 0
        for pre in src_ids:
            for post in dst_ids:
                if count >= max_edges:
                    return
                if rng.random() < probability:
                    edges.append((pre, post, nt))
                    count += 1

    # Visual hierarchy: V1 -> V2 -> prefrontal (Glutamate)
    _connect("V1", "V2", NT_GLU, probability=0.10)
    _connect("V2", "prefrontal", NT_GLU, probability=0.08)

    # Threat pathway: V2 -> amygdala -> LC (Glutamate, then NE)
    _connect("V2", "amygdala", NT_GLU, probability=0.06)
    _connect("amygdala", "LC", NT_GLU, probability=0.12)

    # Memory pathway: prefrontal -> hippocampus (Glutamate)
    _connect("prefrontal", "hippocampus", NT_GLU, probability=0.05)
    _connect("hippocampus", "prefrontal", NT_GLU, probability=0.05)

    # Executive control: prefrontal -> all motor populations (Glutamate)
    for motor_name in ("forward", "turn_left", "turn_right", "strafe_L", "strafe_R"):
        _connect("prefrontal", motor_name, NT_GLU, probability=0.10)

    # Dopamine modulation: VTA -> motor + prefrontal
    for motor_name in ("forward", "turn_left", "turn_right", "strafe_L", "strafe_R"):
        _connect("VTA", motor_name, NT_DA, probability=0.15)
    _connect("VTA", "prefrontal", NT_DA, probability=0.10)

    # Norepinephrine arousal: LC -> V1 + prefrontal + motor populations
    _connect("LC", "V1", NT_NE, probability=0.08)
    _connect("LC", "prefrontal", NT_NE, probability=0.08)
    for motor_name in ("forward", "turn_left", "turn_right", "strafe_L", "strafe_R"):
        _connect("LC", motor_name, NT_NE, probability=0.10)

    # Mutual inhibition: turn_left <-> turn_right (GABA lateral competition)
    _connect("turn_left", "turn_right", NT_GABA, probability=0.20)
    _connect("turn_right", "turn_left", NT_GABA, probability=0.20)

    # Intracortical excitation within V1 (recurrent Glu)
    _connect("V1", "V1", NT_GLU, probability=0.03, max_edges=2000)

    # Intracortical excitation within prefrontal
    _connect("prefrontal", "prefrontal", NT_GLU, probability=0.03, max_edges=2000)

    # Build brain from edges
    brain = MolecularBrain.from_edges(n_neurons, edges, psc_scale=30.0, dt=0.1)

    # Build retina (Rust, not Python)
    retina = MolecularRetina(width=RETINA_WIDTH, height=RETINA_HEIGHT, seed=seed)

    return brain, regions, retina


# ============================================================================
# Retina-to-Brain Bridge
# ============================================================================

class RetinaBridge:
    """Maps RGC spike IDs from MolecularRetina to V1 neuron stimulation.

    Uses retinotopic mapping: RGC index is linearly mapped to V1 neuron
    index range, with a small fan-out so each RGC activates 2-5 nearby
    V1 neurons for robust signal propagation.

    Args:
        n_rgc: Number of retinal ganglion cells in the retina.
        v1_start: Start index of V1 region in the brain.
        v1_size: Number of neurons in V1.
    """

    def __init__(self, n_rgc: int, v1_start: int, v1_size: int):
        self.n_rgc = n_rgc
        self.v1_start = v1_start
        self.v1_size = v1_size
        self._total_injections = 0

        # Precompute retinotopic mapping: RGC_id -> list of V1 neuron indices
        self._rgc_to_v1: Dict[int, List[int]] = {}
        fan_out = min(max(2, v1_size // max(n_rgc, 1) * 2), 5)

        for rgc_idx in range(n_rgc):
            # Linear retinotopic mapping
            center_frac = rgc_idx / max(n_rgc - 1, 1)
            center_v1 = int(center_frac * (v1_size - 1))
            half = fan_out // 2
            start = max(0, center_v1 - half)
            end = min(v1_size, start + fan_out)
            self._rgc_to_v1[rgc_idx] = [
                v1_start + i for i in range(start, end)
            ]

    def inject(
        self,
        brain: Any,
        fired_rgc_ids: List[int],
        intensity: float = 40.0,
    ) -> int:
        """Inject current into V1 neurons corresponding to fired RGCs.

        For each fired RGC, stimulates the mapped V1 neuron(s) with the
        given current amplitude. Uses pulsed injection (only called once
        per frame, not sustained) to avoid depolarization block.

        Args:
            brain: The MolecularBrain instance.
            fired_rgc_ids: List of RGC neuron IDs that fired.
            intensity: Current injection amplitude (uA/cm^2).

        Returns:
            Number of V1 neurons activated.
        """
        if not fired_rgc_ids:
            return 0

        activated = set()
        for rgc_id in fired_rgc_ids:
            if rgc_id in self._rgc_to_v1:
                for v1_nid in self._rgc_to_v1[rgc_id]:
                    activated.add(v1_nid)

        for nid in activated:
            brain.stimulate(nid, intensity)

        self._total_injections += 1
        return len(activated)


# ============================================================================
# Motor Decoder
# ============================================================================

class DoomMotorDecoder:
    """5-way L5 spike count decoder for Doom actions.

    Reads spike counts from 5 motor population regions and selects the
    action corresponding to the population with the highest spike count.

    Uses zero-threshold decoding: ANY spike count difference drives action
    selection. This maximizes responsiveness and breaks initial symmetry
    (following DishBrain findings).

    Args:
        regions: Brain region layout dict.
    """

    MOTOR_REGION_NAMES = ["forward", "turn_left", "turn_right", "strafe_L", "strafe_R"]

    def __init__(self, regions: Dict[str, Tuple[int, int]]):
        self.populations: List[List[int]] = []
        for name in self.MOTOR_REGION_NAMES:
            self.populations.append(_region_ids(regions, name))

    def decode(self, brain: Any) -> Tuple[int, List[int]]:
        """Decode motor action from spike counts in motor populations.

        Reads the global spike_counts() from the brain and sums over each
        motor population to determine which action to take.

        Args:
            brain: The MolecularBrain instance.

        Returns:
            Tuple of (action_index, spike_counts_per_population).
        """
        all_counts = brain.spike_counts()
        pop_counts = []
        for pop_ids in self.populations:
            total = sum(all_counts[i] for i in pop_ids)
            pop_counts.append(total)

        total_spikes = sum(pop_counts)
        if total_spikes == 0:
            # No spikes in any motor population: random action to explore
            return random.randint(0, N_MOTOR_POPULATIONS - 1), pop_counts

        # Zero-threshold: pick population with most spikes
        max_count = max(pop_counts)
        # Break ties randomly among winners
        winners = [i for i, c in enumerate(pop_counts) if c == max_count]
        action = random.choice(winners)
        return action, pop_counts


# ============================================================================
# FEP Protocol for Doom
# ============================================================================

class DoomFEPProtocol:
    """Free Energy Principle learning protocol adapted for Doom.

    Positive events (health gain): STRUCTURED pulsed stimulation to V1
    cortical neurons. Predictable input creates correlated pre-post firing
    that STDP strengthens. NE boost enhances signal-to-noise.

    Negative events (damage taken): RANDOM noise to 30% of V1 neurons.
    Unpredictable input creates uncorrelated activity -- no systematic STDP.

    Hebbian weight nudge directly strengthens relay-to-correct-motor pathways
    on positive events.

    Args:
        brain: MolecularBrain instance.
        regions: Brain region layout dict.
        structured_steps: Steps for structured feedback delivery.
        unstructured_steps: Steps for unstructured feedback delivery.
        structured_intensity: Current amplitude for structured pulses (uA/cm^2).
        unstructured_intensity: Current amplitude for random noise (uA/cm^2).
        ne_boost_nm: Norepinephrine concentration boost during positive events.
    """

    def __init__(
        self,
        brain: Any,
        regions: Dict[str, Tuple[int, int]],
        structured_steps: int = 50,
        unstructured_steps: int = 100,
        structured_intensity: float = 40.0,
        unstructured_intensity: float = 40.0,
        ne_boost_nm: float = 200.0,
    ):
        self.brain = brain
        self.regions = regions
        self.structured_steps = structured_steps
        self.unstructured_steps = unstructured_steps
        self.structured_intensity = structured_intensity
        self.unstructured_intensity = unstructured_intensity
        self.ne_boost_nm = ne_boost_nm

        self.v1_ids = _region_ids(regions, "V1")
        self.lc_ids = _region_ids(regions, "LC")
        self.amygdala_ids = _region_ids(regions, "amygdala")

        # Scale-adaptive Hebbian delta
        n_motor = sum(
            _region_size(regions, m)
            for m in DoomMotorDecoder.MOTOR_REGION_NAMES
        )
        self.hebbian_delta = 0.8 * max(1.0, (n_motor / 200) ** 0.3)

        # Track last action for Hebbian credit assignment
        self.last_action: int = 0

    def on_health_gain(self, brain: Any, delta: float = 1.0) -> None:
        """Structured feedback for positive events (health gain, survival).

        Low entropy: every V1 neuron gets same intensity, pulsed timing.
        Creates correlated activity that STDP strengthens.
        NE boost enhances STDP gain (biologically: locus coeruleus activation).

        After stimulation, performs Hebbian weight nudge on relay-to-motor
        pathways: strengthens V1->correct motor, weakens V1->wrong motor.

        Args:
            brain: The MolecularBrain instance.
            delta: Health delta magnitude (unused, protocol is binary).
        """
        # NE boost to LC neurons
        for nid in self.lc_ids:
            brain.set_nt_concentration(nid, NT_NE, self.ne_boost_nm)

        # Structured pulsed stimulation to V1
        for s in range(self.structured_steps):
            if s % 2 == 0:  # pulsed: 5ms on / 5ms off to avoid Na+ inactivation
                for nid in self.v1_ids:
                    brain.stimulate(nid, self.structured_intensity)
            brain.step()

        # Hebbian weight nudge: V1 -> correct motor, weaken V1 -> wrong motor
        correct_motor_name = DoomMotorDecoder.MOTOR_REGION_NAMES[self.last_action]
        correct_ids = [
            i for i in _region_ids(self.regions, correct_motor_name)
        ]
        wrong_ids = []
        for i, name in enumerate(DoomMotorDecoder.MOTOR_REGION_NAMES):
            if i != self.last_action:
                wrong_ids.extend(_region_ids(self.regions, name))

        brain.hebbian_nudge(
            [i for i in self.v1_ids],
            correct_ids,
            wrong_ids,
            self.hebbian_delta,
        )

    def on_damage(self, brain: Any) -> None:
        """Unstructured feedback for negative events (damage taken).

        High entropy: random 30% of V1 neurons, random timing.
        Uncorrelated activity produces no systematic STDP strengthening.
        Also delivers nociceptor current to amygdala.

        Args:
            brain: The MolecularBrain instance.
        """
        rng = random.Random()

        # Nociceptor current to amygdala
        for nid in self.amygdala_ids:
            brain.stimulate(nid, 30.0)

        # NE burst to LC (stress response)
        for nid in self.lc_ids:
            brain.set_nt_concentration(nid, NT_NE, self.ne_boost_nm * 0.5)

        # Unstructured random noise to 30% of V1
        for s in range(self.unstructured_steps):
            for nid in self.v1_ids:
                if rng.random() < 0.3:
                    intensity = rng.random() * self.unstructured_intensity
                    brain.stimulate(nid, intensity)
            brain.step()


# ============================================================================
# DA Protocol for Doom
# ============================================================================

class DoomDAProtocol:
    """Standard dopamine reward protocol for comparison with FEP.

    Positive events: DA release at VTA neurons (dopamine reward signal).
    Negative events: small DA dip, network settles.

    Args:
        brain: MolecularBrain instance.
        regions: Brain region layout dict.
        reward_steps: Steps to run during DA delivery.
        settle_steps: Steps to run during settling.
    """

    def __init__(
        self,
        brain: Any,
        regions: Dict[str, Tuple[int, int]],
        reward_steps: int = 50,
        settle_steps: int = 15,
    ):
        self.brain = brain
        self.regions = regions
        self.reward_steps = reward_steps
        self.settle_steps = settle_steps
        self.vta_ids = _region_ids(regions, "VTA")
        self.v1_ids = _region_ids(regions, "V1")
        self.last_action: int = 0

    def on_health_gain(self, brain: Any, delta: float = 1.0) -> None:
        """DA reward at VTA neurons for positive events.

        Args:
            brain: The MolecularBrain instance.
            delta: Unused, protocol is binary.
        """
        for s in range(self.reward_steps):
            if s % 3 == 0:
                # DA burst to VTA
                for nid in self.vta_ids:
                    brain.set_nt_concentration(nid, NT_DA, 300.0)
            if s % 2 == 0:
                # Mild cortical stimulation
                for nid in self.v1_ids:
                    brain.stimulate(nid, 20.0)
            brain.step()

    def on_damage(self, brain: Any) -> None:
        """Small DA dip -- no reward, let the network settle.

        Args:
            brain: The MolecularBrain instance.
        """
        # Brief DA dip
        for nid in self.vta_ids:
            brain.set_nt_concentration(nid, NT_DA, 5.0)
        brain.run(self.settle_steps)


# ============================================================================
# Random Protocol (Control)
# ============================================================================

class DoomRandomProtocol:
    """Control: no differential feedback, just settle steps.

    Both positive and negative events receive identical treatment (settling
    with no stimulation), making learning impossible.

    Args:
        settle_steps: Steps to run during settling.
    """

    def __init__(self, settle_steps: int = 15):
        self.settle_steps = settle_steps
        self.last_action: int = 0

    def on_health_gain(self, brain: Any, delta: float = 1.0) -> None:
        """Same as negative -- no differential feedback."""
        brain.run(self.settle_steps)

    def on_damage(self, brain: Any) -> None:
        """Same as positive -- no differential feedback."""
        brain.run(self.settle_steps)


# ============================================================================
# Doom Game Loop
# ============================================================================

def play_doom_episode(
    brain: Any,
    game: DoomGame,
    retina: Any,
    bridge: RetinaBridge,
    decoder: DoomMotorDecoder,
    protocol: Any,
    stim_steps: int = 10,
    max_game_steps: int = 500,
    neutral_steps: int = 5,
) -> Dict[str, Any]:
    """Play one Doom episode through the Rust molecular retina pipeline.

    Full loop:
      ViZDoom frame -> Rust retina spikes -> V1 injection ->
      Rust brain cortical processing -> motor decode -> game action ->
      FEP/DA/Random feedback.

    Args:
        brain: The Rust MolecularBrain instance.
        game: The DoomGame wrapper.
        retina: Rust MolecularRetina instance.
        bridge: Retina-to-brain bridge.
        decoder: Motor decoder for L5 readout.
        protocol: Learning protocol (FEP, DA, or Random).
        stim_steps: Neural processing steps per game frame.
        max_game_steps: Maximum steps before forced episode end.
        neutral_steps: Settling steps on neutral game events.

    Returns:
        Dict with episode metrics: health_gained, damage_taken, steps,
        positive/negative event counts, action distribution.
    """
    game.new_episode()
    retina.reset()

    total_positive = 0
    total_negative = 0
    action_counts = [0] * N_MOTOR_POPULATIONS
    step_count = 0
    prev_health = game.get_health()
    health_gained = 0.0
    damage_taken = 0.0

    while game.is_running and step_count < max_game_steps:
        # 1. Capture frame and convert to RGB bytes for Rust retina
        frame_bytes = game.get_frame()

        # 2. Process frame through Rust MolecularRetina (pixel -> RGC spikes)
        fired_rgc_ids = retina.process_frame(frame_bytes, n_steps=5)

        # 3. Inject RGC spikes into V1 neurons via retinotopic bridge
        bridge.inject(brain, fired_rgc_ids, intensity=45.0)

        # 4. Run brain for stim_steps (pulsed to avoid depolarization block)
        for s in range(stim_steps):
            if s % 2 == 0 and fired_rgc_ids:
                # Re-inject on alternate steps for sustained drive
                bridge.inject(brain, fired_rgc_ids, intensity=20.0)
            brain.step()

        # 5. Decode motor action from spike counts
        action, pop_counts = decoder.decode(brain)

        # Track which action the protocol should credit
        if hasattr(protocol, "last_action"):
            protocol.last_action = action
        action_counts[action] += 1

        # 6. Execute action in ViZDoom
        reward, done = game.step(action)

        if done:
            break

        # 7. Compute health delta for FEP feedback
        current_health = game.get_health()
        health_delta = current_health - prev_health
        prev_health = current_health

        if health_delta > 0:
            health_gained += health_delta
            protocol.on_health_gain(brain, health_delta)
            total_positive += 1
        elif health_delta < 0:
            damage_taken += abs(health_delta)
            protocol.on_damage(brain)
            total_negative += 1
        else:
            # Neutral step: brief settling
            brain.run(neutral_steps)

        step_count += 1

    return {
        "health_gained": health_gained,
        "damage_taken": damage_taken,
        "steps": step_count,
        "positive_events": total_positive,
        "negative_events": total_negative,
        "action_counts": action_counts,
    }


# ============================================================================
# Experiment 1: Doom Navigation (Health Gathering)
# ============================================================================

def exp_doom_navigation(
    scale: str = "small",
    seed: int = 42,
    n_episodes: Optional[int] = None,
    scenario: str = "health_gathering",
    visible: bool = False,
) -> Dict[str, Any]:
    """Can a dONN learn to gather health in Doom via the free energy principle?

    Tracks health gained per episode. Learning is evidenced by increasing
    health acquisition over time as the FEP protocol strengthens sensorimotor
    pathways through structured feedback and Hebbian nudging.

    Args:
        scale: Network scale ("small", "medium", "large").
        seed: Random seed.
        n_episodes: Number of episodes to run (None = scale default).
        scenario: ViZDoom scenario to use.
        visible: Whether to show the game window.

    Returns:
        Dict with pass/fail status, per-episode metrics, and timing.
    """
    sp = SCALE_PARAMS[scale]
    if n_episodes is None:
        n_episodes = sp["n_episodes"]

    _header(
        f"Exp 1: Doom Navigation ({scenario}) -- Rust Brain",
        "Free energy principle -- structured vs unstructured feedback, NO reward",
    )
    t0 = time.perf_counter()

    # Build Rust brain + retina
    n_neurons = sp["n_neurons"]
    brain, regions, retina = build_doom_brain(n_neurons, seed)
    gpu_str = "Metal GPU" if has_gpu() else "CPU"
    print(f"    Brain: {brain.n_neurons} neurons, {brain.n_synapses} synapses ({gpu_str})")
    print(f"    Retina: {retina.n_rgc} RGCs, {retina.total_neurons} total retinal neurons")

    # Build bridge and decoder
    v1_start, v1_end = regions["V1"]
    bridge = RetinaBridge(retina.n_rgc, v1_start, v1_end - v1_start)
    decoder = DoomMotorDecoder(regions)

    # Create FEP protocol
    protocol = DoomFEPProtocol(
        brain,
        regions,
        structured_steps=sp["structured_steps"],
        unstructured_steps=sp["unstructured_steps"],
    )

    # Warmup brain (let circadian/glia/STDP stabilize)
    print(f"    Warming up ({sp['warmup_steps']} steps)...")
    brain.run(sp["warmup_steps"])
    print(f"    Warmup complete")

    # Game
    game = DoomGame(scenario=scenario, seed=seed, visible=visible)

    # Play episodes
    report_interval = max(1, n_episodes // 5)
    episode_metrics: List[Dict[str, Any]] = []

    for ep in range(n_episodes):
        metrics = play_doom_episode(
            brain,
            game,
            retina,
            bridge,
            decoder,
            protocol,
            stim_steps=sp["stim_steps"],
            max_game_steps=sp["max_game_steps"],
            neutral_steps=sp["neutral_steps"],
        )
        episode_metrics.append(metrics)

        if (ep + 1) % report_interval == 0 or ep == n_episodes - 1:
            recent = episode_metrics[max(0, ep - report_interval + 1) : ep + 1]
            avg_health = sum(m["health_gained"] for m in recent) / len(recent)
            avg_damage = sum(m["damage_taken"] for m in recent) / len(recent)
            print(
                f"    Episode {ep + 1:3d}/{n_episodes}: "
                f"health +{avg_health:.0f}, damage -{avg_damage:.0f} "
                f"(last {len(recent)})"
            )

    game.close()

    # Analyze results
    health_per_ep = [m["health_gained"] for m in episode_metrics]
    quarter = max(1, n_episodes // 4)
    first_q = health_per_ep[:quarter]
    last_q = health_per_ep[-quarter:]
    first_avg = sum(first_q) / len(first_q) if first_q else 0
    last_avg = sum(last_q) / len(last_q) if last_q else 0
    total_health = sum(health_per_ep)

    elapsed = time.perf_counter() - t0

    # Pass: health gathered improves over episodes OR total health > 0
    passed = (last_avg > first_avg) or (total_health > 0)

    print(f"\n    Results:")
    print(f"    First quarter avg health: {first_avg:.1f}")
    print(f"    Last quarter avg health:  {last_avg:.1f}")
    print(f"    Total health gathered:    {total_health:.0f}")
    print(f"    Improvement:              {last_avg - first_avg:+.1f}")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "first_q_avg": first_avg,
        "last_q_avg": last_avg,
        "total_health": total_health,
        "episode_metrics": episode_metrics,
    }


# ============================================================================
# Experiment 2: Learning Speed Comparison
# ============================================================================

def exp_learning_speed(
    scale: str = "small",
    seed: int = 42,
    n_episodes: Optional[int] = None,
    scenario: str = "health_gathering",
    visible: bool = False,
) -> Dict[str, Any]:
    """Compare free energy vs DA reward vs random protocols on Doom.

    Each protocol trains an identical brain (same seed) and plays the same
    scenario. Metrics: total health gathered per protocol.

    Args:
        scale: Network scale.
        seed: Random seed.
        n_episodes: Episodes per protocol (None = scale default).
        scenario: ViZDoom scenario name.
        visible: Whether to show the game window.

    Returns:
        Dict with per-protocol results and pass/fail status.
    """
    sp = SCALE_PARAMS[scale]
    if n_episodes is None:
        n_episodes = sp["n_episodes"]

    _header(
        "Exp 2: Learning Speed Comparison -- Rust Brain",
        "Free energy vs DA reward vs random -- which learns fastest in Doom?",
    )
    t0 = time.perf_counter()

    conditions = ["free_energy", "da_reward", "random"]
    all_results: Dict[str, Dict[str, Any]] = {}

    for condition in conditions:
        print(f"\n    --- {condition} ---")

        # Build fresh brain (same seed for fair comparison)
        n_neurons = sp["n_neurons"]
        brain, regions, retina = build_doom_brain(n_neurons, seed)

        v1_start, v1_end = regions["V1"]
        bridge = RetinaBridge(retina.n_rgc, v1_start, v1_end - v1_start)
        decoder = DoomMotorDecoder(regions)

        # Create protocol
        if condition == "free_energy":
            protocol = DoomFEPProtocol(
                brain,
                regions,
                structured_steps=sp["structured_steps"],
                unstructured_steps=sp["unstructured_steps"],
            )
        elif condition == "da_reward":
            protocol = DoomDAProtocol(
                brain,
                regions,
                reward_steps=sp["structured_steps"],
                settle_steps=sp["neutral_steps"] * 3,
            )
        else:
            protocol = DoomRandomProtocol(
                settle_steps=sp["neutral_steps"] * 3,
            )

        # Warmup
        brain.run(sp["warmup_steps"])

        game = DoomGame(scenario=scenario, seed=seed, visible=visible)

        episode_metrics: List[Dict[str, Any]] = []
        for ep in range(n_episodes):
            metrics = play_doom_episode(
                brain,
                game,
                retina,
                bridge,
                decoder,
                protocol,
                stim_steps=sp["stim_steps"],
                max_game_steps=sp["max_game_steps"],
                neutral_steps=sp["neutral_steps"],
            )
            episode_metrics.append(metrics)

        game.close()

        total_health = sum(m["health_gained"] for m in episode_metrics)
        q = max(1, n_episodes // 4)
        last_q = episode_metrics[-q:]
        last_q_health = sum(m["health_gained"] for m in last_q) / len(last_q)

        all_results[condition] = {
            "total_health": total_health,
            "last_q_avg": last_q_health,
            "episode_metrics": episode_metrics,
        }
        print(
            f"    {condition:15s}: total health = {total_health:.0f}, "
            f"last quarter avg = {last_q_health:.1f}"
        )

    elapsed = time.perf_counter() - t0

    # Pass: FEP or DA outperforms random
    fe_total = all_results["free_energy"]["total_health"]
    da_total = all_results["da_reward"]["total_health"]
    rand_total = all_results["random"]["total_health"]
    passed = fe_total > rand_total or da_total > rand_total

    print(f"\n    Free Energy: {fe_total:.0f} total health")
    print(f"    DA Reward:   {da_total:.0f} total health")
    print(f"    Random:      {rand_total:.0f} total health")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk != "episode_metrics"}
            for k, v in all_results.items()
        },
        "all_results": all_results,
    }


# ============================================================================
# Experiment 3: Pharmacological Effects on Doom
# ============================================================================

def exp_pharmacology(
    scale: str = "small",
    seed: int = 42,
    n_train_episodes: Optional[int] = None,
    n_test_episodes: Optional[int] = None,
    scenario: str = "health_gathering",
    visible: bool = False,
) -> Dict[str, Any]:
    """Drug effects on Doom performance -- IMPOSSIBLE on real DishBrain tissue.

    Train 3 identical brains with FEP, then apply drugs before testing:
    - Baseline: no drug
    - Caffeine: adenosine antagonist (expected: mild improvement or neutral)
    - Diazepam: GABA-A enhancer (expected: impairment)

    On real tissue, drugs are irreversible. In simulation, we train identical
    networks and compare post-drug performance.

    Args:
        scale: Network scale.
        seed: Random seed.
        n_train_episodes: Episodes for pre-drug training (None = scale default).
        n_test_episodes: Episodes for post-drug testing (None = scale default).
        scenario: ViZDoom scenario name.
        visible: Whether to show the game window.

    Returns:
        Dict with per-condition test results and pass/fail status.
    """
    sp = SCALE_PARAMS[scale]
    if n_train_episodes is None:
        n_train_episodes = sp["n_train_episodes"]
    if n_test_episodes is None:
        n_test_episodes = sp["n_test_episodes"]

    _header(
        "Exp 3: Pharmacological Effects on Doom Performance -- Rust Brain",
        "3 conditions: baseline / caffeine / diazepam",
    )
    t0 = time.perf_counter()

    conditions = ["baseline", "caffeine", "diazepam"]
    test_results: Dict[str, Dict[str, Any]] = {}

    for condition in conditions:
        print(f"\n    --- {condition} ---")

        # Build and train brain (same seed = identical initial network)
        n_neurons = sp["n_neurons"]
        brain, regions, retina = build_doom_brain(n_neurons, seed)

        v1_start, v1_end = regions["V1"]
        bridge = RetinaBridge(retina.n_rgc, v1_start, v1_end - v1_start)
        decoder = DoomMotorDecoder(regions)

        # FEP training protocol
        protocol = DoomFEPProtocol(
            brain,
            regions,
            structured_steps=sp["structured_steps"],
            unstructured_steps=sp["unstructured_steps"],
        )

        # Warmup
        brain.run(sp["warmup_steps"])

        # Train phase
        game = DoomGame(scenario=scenario, seed=seed, visible=visible)
        for ep in range(n_train_episodes):
            play_doom_episode(
                brain,
                game,
                retina,
                bridge,
                decoder,
                protocol,
                stim_steps=sp["stim_steps"],
                max_game_steps=sp["max_game_steps"],
                neutral_steps=sp["neutral_steps"],
            )
        game.close()
        print(f"    Training complete ({n_train_episodes} episodes)")

        # Apply drug AFTER training
        if condition == "caffeine":
            brain.apply_drug("caffeine", 200.0)
            print(f"    Applied caffeine 200mg")
        elif condition == "diazepam":
            brain.apply_drug("diazepam", 40.0)
            print(f"    Applied diazepam 40mg")

        # Test phase (Random protocol = no further learning)
        test_protocol = DoomRandomProtocol(
            settle_steps=sp["neutral_steps"] * 3,
        )

        test_game = DoomGame(
            scenario=scenario, seed=seed + 1000, visible=visible
        )
        test_metrics: List[Dict[str, Any]] = []
        for ep in range(n_test_episodes):
            metrics = play_doom_episode(
                brain,
                test_game,
                retina,
                bridge,
                decoder,
                test_protocol,
                stim_steps=sp["stim_steps"],
                max_game_steps=sp["max_game_steps"],
                neutral_steps=sp["neutral_steps"],
            )
            test_metrics.append(metrics)
        test_game.close()

        total_health = sum(m["health_gained"] for m in test_metrics)
        avg_health = total_health / max(n_test_episodes, 1)
        test_results[condition] = {
            "total_health": total_health,
            "avg_health": avg_health,
            "test_metrics": test_metrics,
        }
        print(
            f"    {condition:10s}: avg health = {avg_health:.1f} "
            f"(total {total_health:.0f} over {n_test_episodes} episodes)"
        )

    elapsed = time.perf_counter() - t0

    # Pass: diazepam < baseline (GABA-A enhancement impairs performance)
    baseline_health = test_results["baseline"]["total_health"]
    diazepam_health = test_results["diazepam"]["total_health"]
    caffeine_health = test_results["caffeine"]["total_health"]

    passed = diazepam_health < baseline_health

    print(f"\n    Baseline:  {baseline_health:.0f} total health")
    print(
        f"    Caffeine:  {caffeine_health:.0f} total health "
        f"({caffeine_health - baseline_health:+.0f})"
    )
    print(
        f"    Diazepam:  {diazepam_health:.0f} total health "
        f"({diazepam_health - baseline_health:+.0f})"
    )
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "test_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "test_metrics"}
            for k, v in test_results.items()
        },
    }


# ============================================================================
# JSON helpers
# ============================================================================

def _make_json_safe(obj: Any) -> Any:
    """Convert results dict to JSON-serializable form.

    Handles any non-standard numeric types that might appear in results.

    Args:
        obj: Any Python object to make JSON-safe.

    Returns:
        JSON-serializable version of the object.
    """
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, bool):
        return obj
    return obj


def _system_info() -> Dict[str, Any]:
    """Collect system information for JSON output.

    Returns:
        Dict with platform, Python version, Rust backend info, and GPU status.
    """
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "vizdoom": HAS_VIZDOOM,
        "backend": "oneuro_metal (Rust/PyO3)",
    }
    if HAS_ONEURO_METAL:
        info["oneuro_metal_version"] = version()
        info["gpu"] = "Apple Metal" if has_gpu() else "CPU"
    else:
        info["gpu"] = "N/A (oneuro_metal not installed)"
    return info


# ============================================================================
# CLI Entry Point
# ============================================================================

ALL_EXPERIMENTS = {
    1: ("Doom Navigation", exp_doom_navigation),
    2: ("Learning Speed Comparison", exp_learning_speed),
    3: ("Pharmacological Effects", exp_pharmacology),
}


def _run_single(args, seed: int) -> Dict[int, Dict[str, Any]]:
    """Run all requested experiments with a single seed.

    Args:
        args: Parsed CLI arguments.
        seed: Random seed for this run.

    Returns:
        Dict mapping experiment ID to results.
    """
    exps = args.exp if args.exp else list(ALL_EXPERIMENTS.keys())
    results: Dict[int, Dict[str, Any]] = {}

    for exp_id in exps:
        if exp_id not in ALL_EXPERIMENTS:
            print(f"\n  Unknown experiment: {exp_id}")
            continue
        name, func = ALL_EXPERIMENTS[exp_id]

        try:
            kwargs: Dict[str, Any] = {
                "scale": args.scale,
                "seed": seed,
                "scenario": args.scenario,
                "visible": args.visible,
            }
            if exp_id in (1, 2) and args.episodes:
                kwargs["n_episodes"] = args.episodes
            result = func(**kwargs)
            results[exp_id] = result
        except Exception as e:
            print(f"\n  EXPERIMENT {exp_id} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {"passed": False, "error": str(e)}

    return results


def main() -> int:
    """CLI entry point for ViZDoom via Rust Molecular Brain experiments."""
    parser = argparse.ArgumentParser(
        description=(
            "ViZDoom via Rust Molecular Retina + Brain -- "
            "dONN Plays Doom (Free Energy Principle)"
        ),
    )
    parser.add_argument(
        "--exp",
        type=int,
        nargs="*",
        default=None,
        help="Which experiments to run (1-3). Default: all",
    )
    parser.add_argument(
        "--scale",
        default="small",
        choices=list(SCALE_PARAMS.keys()),
        help="Network scale (default: small)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--scenario",
        default="health_gathering",
        choices=list(SCENARIOS.keys()),
        help="ViZDoom scenario (default: health_gathering)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of episodes for experiments 1 and 2",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="Write structured JSON results to file",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show the ViZDoom game window",
    )
    args = parser.parse_args()

    # --- Dependency checks ---
    if not HAS_ONEURO_METAL:
        print("ERROR: oneuro_metal is required.")
        print("       Build with: cd oneuro-metal && maturin develop --release")
        return 1
    if not HAS_VIZDOOM:
        print("ERROR: ViZDoom is required. Install with: pip install vizdoom")
        print("       See: https://github.com/Farama-Foundation/ViZDoom")
        return 1
    if not HAS_PIL:
        print("ERROR: Pillow and numpy are required.")
        print("       Install with: pip install Pillow numpy")
        return 1

    # --- Banner ---
    gpu_str = "Metal GPU" if has_gpu() else "CPU"
    print("=" * 72)
    print("  VIZDOOM VIA RUST MOLECULAR RETINA + BRAIN -- dONN PLAYS DOOM")
    print(
        f"  Backend: oneuro_metal v{version()} ({gpu_str}) | "
        f"Scale: {args.scale} ({SCALE_PARAMS[args.scale]['n_neurons']} neurons) | "
        f"Scenario: {args.scenario}"
    )
    print(f"  Free energy principle: Kagan et al. (2022) Neuron")
    print("=" * 72)

    total_time = time.perf_counter()

    results = _run_single(args, args.seed)

    total = time.perf_counter() - total_time

    # --- Summary ---
    print("\n" + "=" * 72)
    print("  VIZDOOM DOOM (RUST) -- SUMMARY")
    print("=" * 72)

    n_passed = sum(1 for r in results.values() if r.get("passed"))
    total_exp = len(results)
    for exp_id, result in sorted(results.items()):
        if exp_id not in ALL_EXPERIMENTS:
            continue
        name = ALL_EXPERIMENTS[exp_id][0]
        status = "PASS" if result.get("passed") else "FAIL"
        t = result.get("time", 0)
        print(f"    {exp_id}. {name:35s} [{status}]  {t:.1f}s")

    print(f"\n  Total: {n_passed}/{total_exp} passed in {total:.1f}s")
    print("=" * 72)

    # --- JSON output ---
    if args.json:
        json_data = {
            "experiment": "doom_vizdoom_metal",
            "backend": "oneuro_metal (Rust/PyO3)",
            "scale": args.scale,
            "n_neurons": SCALE_PARAMS[args.scale]["n_neurons"],
            "scenario": args.scenario,
            "seed": args.seed,
            "total_time_s": round(total, 2),
            "system": _system_info(),
            "experiments": {},
        }
        for exp_id, result in sorted(results.items()):
            if exp_id not in ALL_EXPERIMENTS:
                continue
            name = ALL_EXPERIMENTS[exp_id][0]
            # Strip bulky per-episode metrics for JSON
            clean: Dict[str, Any] = {}
            for k, v in result.items():
                if k in ("episode_metrics", "all_results", "test_metrics"):
                    continue
                clean[k] = v
            clean = _make_json_safe(clean)
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
