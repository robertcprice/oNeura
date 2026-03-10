"""
oNeuro Physics Module - MuJoCo Integration

Full 3D physics simulation for Drosophila melanogaster using MuJoCo.

This module provides:
- Realistic fly body with articulated legs (4 joints each, including tarsus),
  wings, head, and abdomen — 26 actuators total
- Touch sensors on all 6 tarsi
- 3D environment with fruits, plants, obstacles, wind, and light
- Connectome-based neural controller with tuned LIF dynamics
- Brain-motor interface for sensorimotor integration

The physics model is scaled 1000x (~2.5mm real → ~2.5m simulated) for MuJoCo
numerical stability, but all forces and dynamics are computed at biological scale.

Environment features (PhysicsEnvironment) work without MuJoCo installed.

Reference: NeuroMechFly (Lobato-Rios et al., 2022)
"""

from .drosophila_simulator import (
    DrosophilaSimulator,
    DrosophilaPhysicsState,
    MotorCommand,
    SensorReading,
    MotorPatterns,
    LegID,
    JointType,
)
from .brain_motor_interface import (
    BrainMotorInterface,
    NeuralPatternDecoder,
    SensorEncoder,
    MotorPrimitive,
    MotorPrimitiveType,
)
from .connectome_bridge import (
    ConnectomeBridge,
    NeuronPool,
    SynapseModel,
    NeuronType,
    SynapseType,
    create_brain_model,
)
from .physics_environment import (
    PhysicsEnvironment,
    Fruit,
    Plant,
    Obstacle,
)
from .flywire_connectome import (
    FlyWireConnectome,
    CircuitTemplate,
    Region,
    CellType,
    create_connectome,
)
from .multi_fly_arena import (
    MultiFlyArena,
    SocialBehavior,
    FlyState,
    InterFlySensing,
    SocialMetrics,
    PheromoneSystem,
    create_multi_fly_arena,
)
from .rl_foraging import (
    DrosophilaForagingEnv,
    ForagingConfig,
    Fruit,
    create_foraging_env,
)
from .ecosystem import (
    Terrarium,
    FlyOrganism,
    FoodSource,
    NeuralActivity,
    LifeStage,
    BehaviorState,
    create_ecosystem,
)

__all__ = [
    "DrosophilaSimulator",
    "DrosophilaPhysicsState",
    "MotorCommand",
    "SensorReading",
    "MotorPatterns",
    "LegID",
    "JointType",
    "BrainMotorInterface",
    "NeuralPatternDecoder",
    "SensorEncoder",
    "MotorPrimitive",
    "MotorPrimitiveType",
    "ConnectomeBridge",
    "NeuronPool",
    "SynapseModel",
    "NeuronType",
    "SynapseType",
    "create_brain_model",
    "PhysicsEnvironment",
    "Fruit",
    "Plant",
    "Obstacle",
    "FlyWireConnectome",
    "CircuitTemplate",
    "Region",
    "CellType",
    "create_connectome",
    "MultiFlyArena",
    "SocialBehavior",
    "FlyState",
    "InterFlySensing",
    "SocialMetrics",
    "PheromoneSystem",
    "create_multi_fly_arena",
    "DrosophilaForagingEnv",
    "ForagingConfig",
    "Fruit",
    "create_foraging_env",
    "Terrarium",
    "FlyOrganism",
    "FoodSource",
    "NeuralActivity",
    "LifeStage",
    "BehaviorState",
    "create_ecosystem",
]
