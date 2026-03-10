"""
Food Webs - Trophic networks and energy flow

This module implements:
- Food web structure
- Trophic cascades
- Energy transfer efficiency
- Species interaction networks
- Keystone species identification
- Network analysis (centrality, modularity)

References:
- May (1973) "Stability and Complexity in Model Ecosystems"
- Pimm (1982) "Food Webs"
- Allesina & Pascual (2009) "Food Webs"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
import math
import random


@dataclass
class TrophicNode:
    """Node in food web (species)."""
    id: str
    species_name: str
    trophic_level: float
    biomass: float  # kg
    biomass_fraction: float = 0.01  # Fraction of energy to next level


class FoodWeb:
    """
    Food web with species and interactions.
    """

    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, TrophicNode] = {}
        self.edges: Dict[Tuple[str, str], float] = {}  # (predator, prey): interaction_strength
        self.adjacency: Dict[str, Set[str]] = {}  # Adjacency list

    def add_species(self, species: str, trophic_level: float, biomass: float):
        """Add species to food web."""
        node = TrophicNode(len(self.nodes), species, trophic_level, biomass)
        self.nodes[species] = node
        self.adjacency[species] = set()

    def add_trophic_link(self, predator: str, prey: str, strength: float = 1.0):
        """Add predator-prey interaction."""
        self.edges[(predator, prey)] = strength
        self.adjacency[predator].add(prey)

    def compute_trophic_levels(self):
        """Compute trophic levels from food web."""
        for species, node in self.nodes.items():
            # Count number of prey items
            prey = [p for (p, pr) in self.edges.items() if pr == species]
            if prey:
                # Trophic level = 1 + average prey level
                avg_prey_level = np.mean([self.nodes[p].trophic_level for p in prey])
                node.trophic_level = 1 + avg_prey_level

    def compute_energy_efficiency(self) -> float:
        """
        Compute average trophic efficiency.

        Typically ~10% energy transfer between levels.
        """
        levels = {}

        for species, node in self.nodes.items():
            level = int(node.trophic_level)
            if level not in levels:
                levels[level] = []
            levels[level].append(node.biomass)

        # Compute efficiency between levels
        efficiencies = []
        sorted_levels = sorted(levels.keys())

        for i in range(len(sorted_levels) - 1):
            level1 = sorted_levels[i]
            level2 = sorted_levels[i + 1]

            biomass1 = sum(levels[level1])
            biomass2 = sum(levels[level2])

            if biomass1 > 0:
                eff = biomass2 / biomass1
                efficiencies.append(eff)

        return np.mean(efficiencies) if efficiencies else 0.1

    def find_keystone_species(self) -> List[str]:
        """
        Identify keystone species using network analysis.

        Species whose removal causes large ecosystem change.
        """
        keystone_scores = {}

        for species in self.nodes.keys():
            # Compute impact of removal
            original_biomass = self.nodes[species].biomass
            connections = len(self.adjacency[species])

            # Keystone if: high impact, few connections
            keystone_scores[species] = original_biomass * (1.0 / (connections + 1))

        # Sort by score
        sorted_species = sorted(keystone_scores.items(), key=lambda x: x[1], reverse=True)

        return [s[0] for s in sorted_species[:3]]

    def compute_centrality(self) -> Dict[str, float]:
        """
        Compute betweenness centrality.

        Species with high centrality are important for energy flow.
        """
        centrality = {s: 0.0 for s in self.nodes}

        # Simplified: degree centrality
        for species, adj in self.adjacency.items():
            centrality[species] = len(adj)

        # Normalize
        max_cent = max(centrality.values()) if centrality else 1
        return {s: c / max_cent for s, c in centrality.items()}

    def cascade_effects(self, perturbation: str, magnitude: float) -> Dict[str, float]:
        """
        Simulate trophic cascade.

        Returns biomass change for all species.
        """
        changes = {perturbation: magnitude}

        # Propagate through network
        affected = {perturbation}

        while affected:
            new_affected = set()

            for species in affected:
                # Find predators
                for (pred, prey), strength in self.edges.items():
                    if prey == species:
                        if pred not in changes:
                            # Impact propagates
                            impact = magnitude * strength * 0.5
                            changes[pred] = changes.get(pred, 0) + impact
                            new_affected.add(pred)

            affected = new_affected

        return changes


class TrophicCascade:
    """
    Top-down vs bottom-up control in ecosystems.
    """

    def __init__(self, web: FoodWeb):
        self.web = web

    def apply_top_down(self, predator: str, factor: float):
        """
        Top-down (trophic cascade): predator increases cascade to prey.

        Predators suppress prey populations → prey of prey increases.
        """
        changes = {}

        for (pred, prey), _ in self.web.edges.items():
            if pred == predator:
                # Predator reduces prey
                changes[prey] = -factor * self.web.nodes[pred].biomass

                # And increases prey's prey (indirect)
                for (p2, pr2), _ in self.web.edges.items():
                    if pr2 == prey:
                        changes[p2] = factor * 0.5 * self.web.nodes[pred].biomass

        return changes

    def apply_bottom_up(self, producer: str, factor: float):
        """
        Bottom-up: producer affects all higher trophic levels.
        """
        changes = {producer: factor}

        # Propagate up
        for level in range(1, 5):
            for species, node in self.web.nodes.items():
                if int(node.trophic_level) == level:
                    # Affected by lower level
                    below = [s for s, n in self.web.nodes.items()
                            if int(n.trophic_level) == level - 1]

                    if any(b in changes for b in below):
                        changes[species] = factor * 0.1 ** level

        return changes


def build_example_food_web() -> FoodWeb:
    """
    Build example food web.
    """
    web = FoodWeb('Example Ecosystem')

    # Producers (trophic level 1)
    web.add_species('grass', 1, 5000)
    web.add_species('shrub', 1, 2000)
    web.add_species('algae', 1, 3000)

    # Primary consumers (level 2)
    web.add_species('rabbit', 2, 500)
    web.add_species('deer', 2, 300)
    web.add_species('zooplankton', 2, 1000)

    # Secondary consumers (level 3)
    web.add_species('fox', 3, 50)
    web.add_species('wolf', 3, 30)
    web.add_species('fish', 3, 200)

    # Tertiary (level 4)
    web.add_species('eagle', 4, 5)
    web.add_species('bear', 4, 10)

    # Interactions
    web.add_trophic_link('rabbit', 'grass', 0.8)
    web.add_trophic_link('rabbit', 'shrub', 0.5)
    web.add_trophic_link('deer', 'grass', 0.6)
    web.add_trophic_link('deer', 'shrub', 0.7)
    web.add_trophic_link('zooplankton', 'algae', 0.9)

    web.add_trophic_link('fox', 'rabbit', 0.4)
    web.add_trophic_link('wolf', 'deer', 0.5)
    web.add_trophic_link('fish', 'zooplankton', 0.6)

    web.add_trophic_link('eagle', 'fox', 0.2)
    web.add_trophic_link('bear', 'deer', 0.1)
    web.add_trophic_link('bear', 'fish', 0.3)

    return web


def analyze_food_web():
    """
    Analyze food web structure and dynamics.
    """
    print(f"\n{'='*60}")
    print(f"Food Web Analysis")
    print(f"{'='*60}\n")

    # Build web
    web = build_example_food_web()

    print(f"Food web: {web.name}")
    print(f"  Species: {len(web.nodes)}")
    print(f"  Trophic links: {len(web.edges)}")

    # Compute properties
    efficiency = web.compute_energy_efficiency()
    print(f"\n  Trophic efficiency: {efficiency*100:.1f}%")

    centrality = web.compute_centrality()
    print(f"\nCentrality (degree):")
    for species, cent in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {species}: {cent:.2f}")

    keystone = web.find_keystone_species()
    print(f"\nKeystone species: {keystone}")

    # Trophic cascade
    cascade = TrophicCascade(web)
    top_down = cascade.apply_top_down('wolf', 0.3)
    print(f"\nTop-down effect (wolf +30%):")
    for species, change in top_down.items():
        print(f"  {species}: {change*100:.1f}%")

    print(f"\n{'='*60}")


__all__ = [
    'TrophicNode', 'FoodWeb', 'TrophicCascade',
    'build_example_food_web', 'analyze_food_web'
]
