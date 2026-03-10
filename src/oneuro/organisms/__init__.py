"""oNeuro organism models -- biophysically faithful digital organisms.

Each organism consists of:
  - A nervous system built on CUDAMolecularBrain (HH neurons, NTs, STDP)
  - A body with sensory and motor systems
  - A combined Organism class that runs the sense-think-act loop

Available organisms:

  Fly (Drosophila):
    - Drosophila melanogaster (fruit fly): ~139K neurons, compound eyes, olfaction
    - Complete with wings, legs, antennae

  PetriBrain (In-vitro):
    - RegionalBrain: Cortical + thalamic brain simulation
    - MolecularRetina: Biophysical eye simulation
    - DoomEnvironment: FPS game for behavioral tasks

  Other:
    - C. elegans: 302-neuron connectome (coming soon)
"""

from oneuro.organisms.drosophila import Drosophila, DrosophilaBrain, DrosophilaBody
try:
    from oneuro.organisms.rust_drosophila import RustTerrariumDrosophila, RustDrosophilaBody
except ImportError:  # pragma: no cover - optional native extension
    RustTerrariumDrosophila = None
    RustDrosophilaBody = None
from oneuro.organisms.petribrain import RegionalBrain, MolecularRetina, DoomEnvironment
from oneuro.organisms.fly import *
