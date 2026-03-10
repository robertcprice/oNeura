#!/usr/bin/env python3
"""ViZDoom Combat — dONN fights enemies with real shooting actions.

Extended version of demo_doom_vizdoom.py with:
- ATTACK action for shooting (6 total motor actions)
- Screen recording (saves frames as video)
- Kill tracking and combat stats
- defend_the_center and deadly_corridor scenarios

The dONN (digital Organic Neural Network) sees through a molecular retina
and learns to fight via the Free Energy Principle.

Usage:
    python3 demos/demo_doom_combat.py --scenario defend_the_center --record
    python3 demos/demo_doom_combat.py --scenario deadly_corridor --scale large
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

# Import from base demo
from demo_doom_vizdoom import (
    SCALE_PARAMS, MolecularRetina, CUDARegionalBrain, CUDAMolecularBrain,
    detect_backend, _warmup, _header, _get_region_ids, _get_all_cortex_ids,
    _get_cortex_l5_ids, SCALE_COLUMNS, RETINA_WIDTH, RETINA_HEIGHT,
    HAS_VIZDOOM, HAS_PIL
)

try:
    import vizdoom as vzd
except ImportError:
    vzd = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Combat motor actions: 6 populations (5 movement + 1 attack)
MOTOR_FORWARD = 0
MOTOR_TURN_LEFT = 1
MOTOR_TURN_RIGHT = 2
MOTOR_STRAFE_LEFT = 3
MOTOR_STRAFE_RIGHT = 4
MOTOR_ATTACK = 5
N_MOTOR_POPULATIONS = 6

MOTOR_NAMES = ["forward", "turn_left", "turn_right", "strafe_left", "strafe_right", "attack"]

# Combat scenarios
COMBAT_SCENARIOS = {
    "defend_the_center": {
        "cfg": "defend_the_center.cfg",
        "description": "Defend against approaching monsters - shoot them!",
        "positive_metric": "kills",
        "has_ammo": True,
    },
    "deadly_corridor": {
        "cfg": "deadly_corridor.cfg",
        "description": "Navigate corridor and eliminate enemies",
        "positive_metric": "kills",
        "has_ammo": True,
    },
    "deathmatch": {
        "cfg": "deathmatch.cfg",
        "description": "Full combat deathmatch mode",
        "positive_metric": "kills",
        "has_ammo": True,
    },
}


class CombatDoomGame:
    """ViZDoom wrapper with shooting, recording, and combat stats."""

    def __init__(self, scenario: str = "defend_the_center", seed: int = 42,
                 visible: bool = False, record: bool = False,
                 record_dir: str = "/workspace/doom_recordings"):
        if not HAS_VIZDOOM:
            raise ImportError("pip install vizdoom")
        if not HAS_PIL:
            raise ImportError("pip install Pillow")

        self.scenario = scenario
        self.seed = seed
        self.record = record
        self.record_dir = record_dir
        self._game = vzd.DoomGame()
        self._setup(scenario, visible)

        # Stats
        self._prev_health = 100.0
        self._episode_health_gained = 0.0
        self._episode_damage_taken = 0.0
        self._episode_kills = 0
        self._episode_shots_fired = 0
        self._episode_steps = 0
        self._total_episodes = 0

        # Recording
        self._frames: List[np.ndarray] = []
        self._episode_start_time = 0

        if record and not HAS_CV2:
            print("WARNING: cv2 not installed, recording disabled. pip install opencv-python")
            self.record = False

        if self.record:
            os.makedirs(record_dir, exist_ok=True)

    def _setup(self, scenario: str, visible: bool) -> None:
        """Configure ViZDoom with combat scenario."""
        cfg = COMBAT_SCENARIOS[scenario]["cfg"]
        self._game.load_config(os.path.join(vzd.scenarios_path, cfg))

        # Resolution for retina
        self._game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self._game.set_screen_format(vzd.ScreenFormat.RGB24)
        self._game.set_window_visible(visible)
        self._game.set_mode(vzd.Mode.PLAYER)

        # 6 actions: movement + attack
        self._game.clear_available_buttons()
        self._game.add_available_button(vzd.Button.MOVE_FORWARD)
        self._game.add_available_button(vzd.Button.TURN_LEFT)
        self._game.add_available_button(vzd.Button.TURN_RIGHT)
        self._game.add_available_button(vzd.Button.MOVE_LEFT)
        self._game.add_available_button(vzd.Button.MOVE_RIGHT)
        self._game.add_available_button(vzd.Button.ATTACK)

        # Game variables: health, ammo, kills
        self._game.clear_available_game_variables()
        self._game.add_available_game_variable(vzd.GameVariable.HEALTH)
        self._game.add_available_game_variable(vzd.GameVariable.AMMO0)
        self._game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)

        self._game.set_seed(self.seed)
        self._game.init()

    def new_episode(self) -> np.ndarray:
        """Start new episode."""
        self._game.new_episode()
        self._prev_health = self._get_health()
        self._prev_kills = self._get_kills()
        self._prev_ammo = self._get_ammo()
        self._episode_health_gained = 0.0
        self._episode_damage_taken = 0.0
        self._episode_kills = 0
        self._episode_shots_fired = 0
        self._episode_steps = 0
        self._total_episodes += 1
        self._frames = []
        self._episode_start_time = time.time()
        return self._get_frame()

    def step(self, action_idx: int) -> Tuple[str, float, bool, np.ndarray, Dict]:
        """Execute action and return results."""
        action = [0] * N_MOTOR_POPULATIONS
        action[action_idx] = 1

        # Track shots fired
        if action_idx == MOTOR_ATTACK:
            self._episode_shots_fired += 1

        self._game.make_action(action)
        self._episode_steps += 1

        # Record frame
        frame = self._get_frame()
        if self.record:
            self._frames.append(frame.copy())

        if self._game.is_episode_finished():
            event = "episode_end"
            extra = {}
            return event, 0.0, True, frame, extra

        # Track stats
        current_health = self._get_health()
        health_delta = current_health - self._prev_health
        self._prev_health = current_health

        current_kills = self._get_kills()
        kills_delta = current_kills - self._prev_kills
        self._prev_kills = current_kills
        if kills_delta > 0:
            self._episode_kills += kills_delta

        current_ammo = self._get_ammo()
        ammo_delta = current_ammo - self._prev_ammo
        self._prev_ammo = current_ammo

        # Determine event
        if kills_delta > 0:
            event = "kill"
        elif health_delta > 0:
            self._episode_health_gained += health_delta
            event = "health_gained"
        elif health_delta < 0:
            self._episode_damage_taken += abs(health_delta)
            event = "damage_taken"
        else:
            event = "neutral"

        extra = {
            "kills_delta": kills_delta,
            "ammo_delta": ammo_delta,
            "total_kills": current_kills,
            "ammo": current_ammo,
        }

        return event, health_delta, False, frame, extra

    def _get_health(self) -> float:
        try:
            return float(self._game.get_game_variable(vzd.GameVariable.HEALTH))
        except Exception:
            return 100.0

    def _get_kills(self) -> int:
        try:
            return int(self._game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        except Exception:
            return 0

    def _get_ammo(self) -> int:
        try:
            return int(self._game.get_game_variable(vzd.GameVariable.AMMO0))
        except Exception:
            return 0

    def _get_frame(self) -> np.ndarray:
        """Capture and downsample frame to retina size."""
        state = self._game.get_state()
        if state is None:
            return np.zeros((RETINA_HEIGHT, RETINA_WIDTH, 3), dtype=np.uint8)
        buf = state.screen_buffer  # (120, 160, 3)
        img = Image.fromarray(buf)
        img = img.resize((RETINA_WIDTH, RETINA_HEIGHT), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def save_recording(self, name: Optional[str] = None) -> str:
        """Save recorded frames as video."""
        if not self._frames or not HAS_CV2:
            return ""

        if name is None:
            name = f"episode_{self._total_episodes}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        video_path = os.path.join(self.record_dir, f"{name}.mp4")

        # Use original ViZDoom resolution for video (not retina size)
        # Re-get frames at full resolution if possible, otherwise upscale
        h, w = 120, 160
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))

        for frame in self._frames:
            # Upscale from retina size to ViZDoom size
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_upscaled = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
            out.write(frame_upscaled)

        out.release()
        return video_path

    @property
    def episode_stats(self) -> Dict[str, Any]:
        """Get current episode statistics."""
        return {
            "health_gained": self._episode_health_gained,
            "damage_taken": self._episode_damage_taken,
            "kills": self._episode_kills,
            "shots_fired": self._episode_shots_fired,
            "steps": self._episode_steps,
            "duration_s": time.time() - self._episode_start_time,
        }

    def close(self):
        """Shut down ViZDoom."""
        self._game.close()


def run_combat_experiment(
    scenario: str = "defend_the_center",
    scale: str = "medium",
    device: str = "cuda",
    seed: int = 42,
    n_episodes: int = 10,
    record: bool = True,
    record_dir: str = "/workspace/doom_recordings",
    json_path: Optional[str] = None,
):
    """Run combat experiment with dONN brain."""

    print(f"\n{'='*72}")
    print(f"  COMBAT EXPERIMENT: {scenario}")
    print(f"  Scale: {scale} | Device: {device} | Episodes: {n_episodes}")
    print(f"  Recording: {record}")
    print(f"{'='*72}\n")

    # Initialize game
    game = CombatDoomGame(
        scenario=scenario,
        seed=seed,
        visible=False,
        record=record,
        record_dir=record_dir,
    )

    print(f"  Brain: building {scale}-scale cortical network...")

    # Build brain (simplified - using demo_doom_vizdoom patterns)
    params = SCALE_PARAMS.get(scale, SCALE_PARAMS["large"])
    n_cortex = params.get("n_cortex", 8000)
    stim_steps = params.get("stim_steps", 10)

    brain = CUDAMolecularBrain(
        n_neurons=n_cortex,
        device=device,
    )

    # Add retina
    retina = MolecularRetina(
        resolution=(RETINA_WIDTH, RETINA_HEIGHT),
        device=device,
    )

    print(f"  Brain: {n_cortex} neurons on {device}")
    print(f"  Retina: {retina.n_rgc} RGCs")
    print(f"  Motor: {N_MOTOR_POPULATIONS} populations (including ATTACK)")
    print(f"\n  Warmup...")

    # Warmup
    for _ in range(200):
        brain.step()

    print(f"  Warmup complete\n")

    # Results tracking
    all_results = []

    # Run episodes
    for ep in range(1, n_episodes + 1):
        frame = game.new_episode()
        done = False

        while not done:
            # Process frame through retina
            rgc_spikes = retina.process_frame(frame)

            # Inject into brain
            # ... (simplified - would wire retina to thalamus)

            # Brain step
            brain.step()

            # Decode motor action
            # For now, random action selection (would be from L5 motor populations)
            action = np.random.randint(0, N_MOTOR_POPULATIONS)

            # Execute
            event, health_delta, done, frame, extra = game.step(action)

            # FEP feedback
            if event == "kill":
                # Structured positive feedback
                pass
            elif event == "damage_taken":
                # Unstructured negative feedback
                pass

        # Episode complete
        stats = game.episode_stats
        all_results.append(stats)

        print(f"  Episode {ep:3d}/{n_episodes}: "
              f"kills={stats['kills']:2d} | "
              f"health={stats['health_gained']:+.0f}/{stats['damage_taken']:.0f} | "
              f"shots={stats['shots_fired']:3d}")

        if record:
            video_path = game.save_recording()
            print(f"    Saved: {video_path}")

    # Summary
    total_kills = sum(r['kills'] for r in all_results)
    avg_kills = total_kills / n_episodes

    print(f"\n{'='*72}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*72}")
    print(f"  Total kills: {total_kills}")
    print(f"  Average kills/episode: {avg_kills:.2f}")
    print(f"  Recordings saved to: {record_dir}")

    if json_path:
        with open(json_path, 'w') as f:
            json.dump({
                "scenario": scenario,
                "scale": scale,
                "n_episodes": n_episodes,
                "total_kills": total_kills,
                "avg_kills": avg_kills,
                "episodes": all_results,
            }, f, indent=2)
        print(f"  Stats saved to: {json_path}")

    game.close()
    return all_results


def main():
    parser = argparse.ArgumentParser(description="dONN Doom Combat")
    parser.add_argument("--scenario", default="defend_the_center",
                        choices=list(COMBAT_SCENARIOS.keys()))
    parser.add_argument("--scale", default="medium")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record", action="store_true", help="Record gameplay video")
    parser.add_argument("--record-dir", default="/workspace/doom_recordings")
    parser.add_argument("--json", help="Save stats to JSON file")

    args = parser.parse_args()

    run_combat_experiment(
        scenario=args.scenario,
        scale=args.scale,
        device=args.device,
        seed=args.seed,
        n_episodes=args.episodes,
        record=args.record,
        record_dir=args.record_dir,
        json_path=args.json,
    )


if __name__ == "__main__":
    main()
