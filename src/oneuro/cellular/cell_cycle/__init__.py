"""
Cell Cycle - Cell division and reproduction

This module implements:
- Cell cycle phases (G1, S, G2, M)
- Checkpoint regulation
- Cyclins and CDKs
- DNA replication
- Mitosis and cytokinesis
- Cell growth modeling

References:
- Morgan "The Cell Cycle: Principles of Control"
- Alberts "Molecular Biology of the Cell"
- Nurse (2000) Cell 100, 71
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math


class CellCyclePhase(Enum):
    """Cell cycle phases."""
    G0 = 'G0'           # Quiescent
    G1 = 'G1'           # Gap 1
    S = 'S'             # DNA synthesis
    G2 = 'G2'           # Gap 2
    M = 'M'             # Mitosis
    PROPHASE = 'P'
    METAPHASE = 'META'
    ANAPHASE = 'A'
    TELOPHASE = 'T'
    CYTOKINESIS = 'C'


@dataclass
class Cyclin:
    """Cyclin regulatory protein."""
    name: str
    concentration: float  # μM
    targets_cdk: List[str]
    degradation_rate: float  # per hour


@dataclass
class CDK:
    """Cyclin-dependent kinase."""
    name: str
    activity: float  # 0-1
    phosphorylation: float  # 0-1 (activation)
    inhibitory_phos: float  # 0-1 (inhibition)


@dataclass
class Checkpoint:
    """Cell cycle checkpoint."""
    name: str  # G1/S, G2/M, spindle
    is_engaged: bool = False
    arrest_point: Optional[CellCyclePhase] = None


class CellCycle:
    """
    Cell cycle regulatory network.

    Based on the classic cell cycle model:
    - G1/S transition: Rb-E2F, cyclin D-CDK4/6, cyclin E-CDK2
    - S phase: cyclin A-CDK2
    - G2/M: cyclin B-CDK1
    """

    def __init__(self, cell_type: str = 'somatic'):
        self.cell_type = cell_type

        # Cell state
        self.phase = CellCyclePhase.G1
        self.cycle_time = 24.0  # hours for typical human cell
        self.phase_times = {
            'G1': 11,  # hours
            'S': 8,
            'G2': 4,
            'M': 1,
        }

        # Cell size
        self.radius = 10  # μm
        self.volume = 4/3 * math.pi * self.radius**3

        # DNA content (relative to 2N)
        self.dna_content = 2.0

        # Cyclins
        self.cyclins = {
            'D': Cyclin('cyclin D', 0.5, ['CDK4', 'CDK6'], 0.1),
            'E': Cyclin('cyclin E', 0.2, ['CDK2'], 0.3),
            'A': Cyclin('cyclin A', 0.3, ['CDK2', 'CDK1'], 0.2),
            'B': Cyclin('cyclin B', 0.5, ['CDK1'], 0.5),
        }

        # CDKs
        self.cdks = {
            'CDK4': CDK('CDK4', 0.1, 0.0, 0.0),
            'CDK6': CDK('CDK6', 0.1, 0.0, 0.0),
            'CDK2': CDK('CDK2', 0.1, 0.0, 0.0),
            'CDK1': CDK('CDK1', 0.1, 0.0, 0.0),
        }

        # Checkpoints
        self.checkpoints = {
            'G1/S': Checkpoint('G1/S', False, CellCyclePhase.G1),
            'G2/M': Checkpoint('G2/M', False, CellCyclePhase.G2),
            'METAPHASE': Checkpoint('METAPHASE', False, CellCyclePhase.METAPHASE),
        }

        # Time in current phase
        self.time_in_phase = 0.0

    def update_cyclin_levels(self, dt: float):
        """Update cyclin concentrations based on phase."""
        phase = self.phase

        # G1: cyclin D accumulates
        if phase in [CellCyclePhase.G0, CellCyclePhase.G1]:
            self.cyclins['D'].concentration += 0.01 * dt
            # Cyclin E rises at G1/S
            if self.time_in_phase > self.phase_times['G1'] * 0.9:
                self.cyclins['E'].concentration += 0.05 * dt

        # S: cyclin A rises
        elif phase == CellCyclePhase.S:
            self.cyclins['A'].concentration += 0.03 * dt
            self.cyclins['E'].concentration *= 0.95  # Degrade

        # G2/M: cyclin B rises
        elif phase == CellCyclePhase.G2:
            self.cyclins['B'].concentration += 0.05 * dt

        # M: cyclins degrade
        elif phase == CellCyclePhase.M:
            self.cyclins['B'].concentration *= 0.9
            self.cyclins['A'].concentration *= 0.9

        # Cap concentrations
        for cyclin in self.cyclins.values():
            cyclin.concentration = max(0, min(1.0, cyclin.concentration))

    def compute_cdk_activity(self):
        """Compute CDK activities based on cyclin levels."""
        # Cyclin D → CDK4/6
        cycD = self.cyclins['D'].concentration
        self.cdks['CDK4'].activity = cycD * 0.8
        self.cdks['CDK6'].activity = cycD * 0.8

        # Cyclin E → CDK2
        cycE = self.cyclins['E'].concentration
        self.cdks['CDK2'].activity = cycE * 0.9

        # Cyclin A → CDK2 (S phase)
        cycA = self.cyclins['A'].concentration
        if self.phase == CellCyclePhase.S:
            self.cdks['CDK2'].activity = max(self.cdks['CDK2'].activity, cycA)

        # Cyclin B → CDK1 (G2/M)
        cycB = self.cyclins['B'].concentration
        self.cdks['CDK1'].activity = cycB * 0.95

    def check_g1_s(self) -> bool:
        """G1/S checkpoint - check for DNA damage, nutrients."""
        # Simplified: pass if no DNA damage
        # In reality: check p53, p21, Rb status

        # Size check
        if self.volume < 0.5 * self._initial_volume():
            return False

        # Nutrient check
        nutrients_sufficient = True

        return nutrients_sufficient

    def check_g2_m(self) -> bool:
        """G2/M checkpoint - check DNA replication complete."""
        # DNA content check
        if self.dna_content < 1.95:
            return False

        # DNA damage check
        dna_damage = False

        return not dna_damage

    def check_metaphase(self) -> bool:
        """Metaphase checkpoint - check spindle attachment."""
        # All chromosomes attached
        chromosomes_attached = True

        return chromosomes_attached

    def progress_phase(self, dt: float):
        """
        Progress through cell cycle.
        """
        # Update cyclins
        self.update_cyclin_levels(dt)
        self.compute_cdk_activity()

        # Progress based on CDK activity thresholds
        current_time = self.phase_times.get(self.phase.value, 1)
        self.time_in_phase += dt

        # Phase transitions
        if self.phase == CellCyclePhase.G1:
            # G1/S transition requires cyclin E-CDK2
            if (self.cdks['CDK2'].activity > 0.3 and
                self.check_g1_s() and
                self.time_in_phase > current_time):
                self.phase = CellCyclePhase.S
                self.time_in_phase = 0

        elif self.phase == CellCyclePhase.S:
            # S phase continues until DNA replicated
            if self.dna_content >= 2.0 and self.time_in_phase > current_time:
                self.phase = CellCyclePhase.G2
                self.time_in_phase = 0

        elif self.phase == CellCyclePhase.G2:
            # G2/M requires cyclin B-CDK1
            if (self.cdks['CDK1'].activity > 0.5 and
                self.check_g2_m() and
                self.time_in_phase > current_time):
                self.phase = CellCyclePhase.M
                self.time_in_phase = 0

        elif self.phase == CellCyclePhase.M:
            # M phase progression
            if self.time_in_phase > 0.3:  # Metaphase
                if self.check_metaphase():
                    self.phase = CellCyclePhase.METAPHASE
            if self.time_in_phase > 0.6:  # Anaphase
                self.phase = CellCyclePhase.ANAPHASE
                self.dna_content = 1.0  # Sister chromatids separated
            if self.time_in_phase > 0.8:  # Telophase
                self.phase = CellCyclePhase.TELOPHASE
            if self.time_in_phase > 1.0:  # Cytokinesis
                self.phase = CellCyclePhase.G1
                self.time_in_phase = 0
                self.dna_content = 2.0  # Reset for daughter cell
                return True  # Cell division complete

        return False

    def _initial_volume(self) -> float:
        """Initial cell volume."""
        return 4/3 * math.pi * self.radius**3

    def grow(self, dt: float):
        """Cell growth during G1."""
        if self.phase in [CellCyclePhase.G1, CellCyclePhase.G0]:
            growth_rate = 0.01  # per hour
            self.volume *= (1 + growth_rate * dt)
            self.radius = (3 * self.volume / (4 * math.pi)) ** (1/3)


class CellPopulation:
    """
    Population of dividing cells.
    """

    def __init__(self, initial_cells: int = 100):
        self.cells = [CellCycle() for _ in range(initial_cells)]
        self.generation = 0

    def step(self, dt: float):
        """Step population forward."""
        new_cells = []

        for cell in self.cells:
            # Progress cell cycle
            cell.grow(dt)
            divided = cell.progress_phase(dt)

            if divided:
                new_cells.append(CellCycle())

        self.cells.extend(new_cells)

    def count_by_phase(self) -> Dict[str, int]:
        """Count cells in each phase."""
        counts = {phase.value: 0 for phase in CellCyclePhase}
        for cell in self.cells:
            counts[cell.phase.value] += 1
        return counts


def simulate_cell_cycle(n_hours: float = 48, n_cells: int = 10):
    """
    Simulate cell cycle progression.

    Args:
        n_hours: Simulation duration
        n_cells: Initial cell count
    """
    print(f"\n{'='*60}")
    print(f"Cell Cycle Simulation")
    print(f"Duration: {n_hours} hours, Initial cells: {n_cells}")
    print(f"{'='*60}\n")

    # Create population
    pop = CellPopulation(n_cells)

    # Time step
    dt = 0.1  # hours

    # Run simulation
    for t in range(int(n_hours / dt)):
        pop.step(dt)

        if int(t * dt) % 10 == 0:
            counts = pop.count_by_phase()
            print(f"  t={t*dt:.1f}h: {len(pop.cells)} cells, "
                  f"G1={counts['G1']}, S={counts['S']}, "
                  f"G2={counts['G2']}, M={counts['M']}")

    print(f"\nFinal population: {len(pop.cells)} cells")
    print(f"Growth: {len(pop.cells) / n_cells:.1f}x in {n_hours}h")
    print(f"\n{'='*60}")


__all__ = [
    'CellCyclePhase', 'Cyclin', 'CDK', 'Checkpoint',
    'CellCycle', 'CellPopulation',
    'simulate_cell_cycle'
]
