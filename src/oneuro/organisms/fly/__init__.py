"""
Fly: Drosophila melanogaster digital organism

This module consolidates:
- Drosophila: Complete fly with brain, body, sensors
- FlyBrain: Fly-specific brain regions (mushroom body, central complex)
- FlyBody: Eyes, antennae, legs, wings
- MolecularWorld: Physics-based environment with odor diffusion

Based on the FlyWire Drosophila connectome (~139K neurons)
"""

from oneuro.organisms.drosophila import Drosophila, DrosophilaBrain, DrosophilaBody

__all__ = [
    "Drosophila",
    "DrosophilaBrain",
    "DrosophilaBody",
]
