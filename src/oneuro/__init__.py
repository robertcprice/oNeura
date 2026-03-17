"""oNeura — Biologically-inspired neural networks.

Core:
    from oneuro.organic_neural_network import OrganicNeuron, OrganicSynapse, OrganicNeuralNetwork
    from oneuro.multi_tissue_network import MultiTissueNetwork

Molecular (requires nqpu for quantum features):
    from oneuro.molecular import MolecularNeuron, MolecularSynapse, MolecularNeuralNetwork

Organisms:
    from oneuro.organisms import Drosophila, DrosophilaBrain
    from oneuro.organisms.drosophila import Drosophila, DrosophilaBrain, DrosophilaBody

Export (edge deployment):
    from oneuro.export import ONNXExporter, ModelOptimizer

Robot (real-world integration):
    from oneuro.robot import DroneInterface, SensorFusion, OutdoorNavigationExperiment

Whole-cell (experimental):
    from oneuro.whole_cell import WholeCellConfig, WholeCellProgramSpec, MC4DAdapter, MC4DRunner
"""

from oneuro.organic_neural_network import (
    OrganicNeuron,
    OrganicSynapse,
    OrganicNeuralNetwork,
    NeuronState,
    TrainingTask,
    XORTask,
    PatternRecognitionTask,
    EmergenceTracker,
)
from oneuro.multi_tissue_network import MultiTissueNetwork, TissueType, TissueConfig

# Export and robot modules (optional imports with graceful failures)
try:
    from oneuro.export import ONNXExporter, ModelOptimizer
except ImportError:
    pass

try:
    from oneuro.robot import DroneInterface, SensorFusion, OutdoorNavigationExperiment
except ImportError:
    pass

# Re-export commonly used classes
try:
    from oneuro.organisms.drosophila import Drosophila, DrosophilaBrain, DrosophilaBody
except ImportError:
    pass

try:
    from oneuro.whole_cell import (
        CellCompartment,
        MC4DAdapter,
        MC4DRunConfig,
        MC4DRunner,
        WholeCellScheduler,
        WholeCellConfig,
        WholeCellProgramSpec,
        WholeCellArtifactIngestor,
        WholeCellState,
        build_syn3a_skeleton_scheduler,
        syn3a_reference_manifest,
        syn3a_minimal_state,
        syn3a_reference_program,
    )
except ImportError:
    pass
