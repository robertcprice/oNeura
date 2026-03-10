"""
oNeuro Ecology Module

Submodules:
- ecosystems: Species, trophic levels, patches, logistic growth
- food_webs: Trophic networks, cascades, centrality, keystone species
- population_dynamics: Exponential, logistic, Lotka-Volterra, stochastic

Usage:
    from oneuro.ecology import simulate_ecosystem, analyze_food_web
    from oneuro.ecology.population_dynamics import run_population_simulation
"""

from .ecosystems import (
    TrophicLevel, Species, Patch, Ecosystem,
    create_ecosystem, simulate_ecosystem
)

from .food_webs import (
    TrophicNode, FoodWeb, TrophicCascade,
    build_example_food_web, analyze_food_web
)

from .population_dynamics import (
    GrowthModel,
    Population, PredatorPrey, Competition,
    AgeStructuredPopulation, StochasticPopulation,
    run_population_simulation
)
from .terrarium import (
    PlantGenome, SeedPropagule, PlantCellCluster, PlantCellularState,
    PlantOrganism, SoilChemistryField, SoilBiogeochemistry, RustBatchedAtomTerrarium,
    RustCellularMetabolism, RustPlantCellularState, RustPlantOrganism, TerrariumEcology
)

__all__ = [
    # ecosystems
    'TrophicLevel', 'Species', 'Patch', 'Ecosystem',
    'create_ecosystem', 'simulate_ecosystem',

    # food_webs
    'TrophicNode', 'FoodWeb', 'TrophicCascade',
    'build_example_food_web', 'analyze_food_web',

    # population_dynamics
    'GrowthModel',
    'Population', 'PredatorPrey', 'Competition',
    'AgeStructuredPopulation', 'StochasticPopulation',
    'run_population_simulation',

    # terrarium ecology
    'PlantGenome', 'SeedPropagule', 'PlantCellCluster', 'PlantCellularState',
    'PlantOrganism', 'SoilChemistryField', 'SoilBiogeochemistry',
    'RustBatchedAtomTerrarium', 'RustCellularMetabolism', 'RustPlantCellularState',
    'RustPlantOrganism', 'TerrariumEcology',
]
