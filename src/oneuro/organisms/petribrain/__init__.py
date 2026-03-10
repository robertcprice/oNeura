"""
PetriBrain: In-vitro brain simulation (DishBrain-style)

This module consolidates:
- RegionalBrain: Cortical + thalamic brain simulation
- Retina: Biophysical eye simulation
- DoomEnvironment: FPS game environment for behavioral tasks

Based on Cortical Labs' DishBrain experiments (Kagan et al., 2022)
"""

from oneuro.molecular.brain_regions import RegionalBrain
from oneuro.molecular.retina import MolecularRetina
from oneuro.environments.doom_fps import DoomFPS as DoomEnvironment

__all__ = [
    "RegionalBrain",
    "MolecularRetina",
    "DoomEnvironment",
]
