"""
Connectome Bridge

Provides integration between Drosophila connectome models and
the brain-motor interface. This module enables connection of
realistic neural network models (based on the Drosophila connectome)
to the physics simulation.

The Drosophila connectome contains ~139,000 neurons and ~50 million
synapses. This module provides abstractions for:
- Neuron pools representing functional groups
- Synapse models for signal transmission
- Optional quantum-enhanced neural processing

References:
- Dorkenwald et al. (2024) - Complete connectome of the adult fly
- Takemura et al. (2023) - Hemibrain connectome
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import numpy as np


class NeuronType(Enum):
    """Types of neurons in the Drosophila nervous system."""
    SENSORY = "sensory"
    INTERNEURON = "interneuron"
    MOTOR = "motor"
    NEUROMODULATORY = "neuromodulatory"
    CLOCK = "clock"
    VISUAL = "visual"
    OLFACTORY = "olfactory"
    MECHANOSENSORY = "mechanosensory"
    PROPRIOCEPTIVE = "proprioceptive"


class SynapseType(Enum):
    """Types of synaptic connections."""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    ELECTRICAL = "electrical"
    NEUROMODULATORY = "neuromodulatory"


@dataclass
class SynapseModel:
    """
    Model of a synaptic connection.

    Attributes:
        pre_pool: Name of presynaptic pool
        post_pool: Name of postsynaptic pool
        pre_neuron: Presynaptic neuron index within pool
        post_neuron: Postsynaptic neuron index within pool
        weight: Synaptic weight (positive=excitatory, negative=inhibitory)
        delay: Transmission delay in milliseconds
        synapse_type: Type of synapse
        plasticity_rate: Rate of synaptic plasticity
    """
    pre_pool: str
    post_pool: str
    pre_neuron: int
    post_neuron: int
    weight: float
    delay: float = 1.0  # ms
    synapse_type: SynapseType = SynapseType.EXCITATORY
    plasticity_rate: float = 0.0

    def transmit(self, pre_activity: float) -> float:
        """Compute postsynaptic response."""
        return self.weight * pre_activity


@dataclass
class NeuronPool:
    """
    A pool of neurons representing a functional group.

    Neuron pools are used to simplify the connectome by grouping
    neurons with similar function (e.g., "leg motor neurons",
    "visual motion detectors").

    Attributes:
        name: Pool name
        neuron_type: Type of neurons in pool
        num_neurons: Number of neurons
        resting_potential: Resting membrane potential (mV)
        threshold: Firing threshold (mV)
        membrane_time_constant: Membrane time constant (ms)
        refractory_period: Refractory period (ms)
    """
    name: str
    neuron_type: NeuronType
    num_neurons: int = 100
    resting_potential: float = -70.0  # mV
    threshold: float = -50.0  # mV
    membrane_time_constant: float = 10.0  # ms
    refractory_period: float = 2.0  # ms

    # State
    membrane_potentials: np.ndarray = field(default=None, repr=False)
    spike_times: np.ndarray = field(default=None, repr=False)
    firing_rates: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize neural state arrays."""
        self.membrane_potentials = np.full(
            self.num_neurons, self.resting_potential
        )
        self.spike_times = np.full(self.num_neurons, -1000.0)  # Last spike time
        self.firing_rates = np.zeros(self.num_neurons)

    def update(
        self,
        input_current: np.ndarray,
        dt: float,
        time: float,
    ) -> np.ndarray:
        """
        Update neuron states for one time step.

        Uses leaky integrate-and-fire model.

        Args:
            input_current: Input current for each neuron (nA)
            dt: Time step (ms)
            time: Current time (ms)

        Returns:
            Binary spike array
        """
        # Ensure input is correct size
        if len(input_current) != self.num_neurons:
            padded = np.zeros(self.num_neurons)
            padded[:len(input_current)] = input_current
            input_current = padded

        # Membrane potential dynamics (leaky integrate)
        decay = np.exp(-dt / self.membrane_time_constant)
        self.membrane_potentials = (
            decay * self.membrane_potentials +
            (1 - decay) * self.resting_potential +
            input_current * dt / 2.0  # Scale factor (tuned for small pools)
        )

        # Check for spikes
        spikes = np.zeros(self.num_neurons)
        for i in range(self.num_neurons):
            # Check refractory period
            if time - self.spike_times[i] < self.refractory_period:
                self.membrane_potentials[i] = self.resting_potential
                continue

            # Threshold crossing
            if self.membrane_potentials[i] >= self.threshold:
                spikes[i] = 1.0
                self.spike_times[i] = time
                self.membrane_potentials[i] = self.resting_potential

        # Update firing rate estimate (exponential moving average)
        self.firing_rates = 0.9 * self.firing_rates + 0.1 * spikes / (dt / 1000.0)

        return spikes

    def get_activity(self) -> np.ndarray:
        """Get current firing rates as activity pattern."""
        return self.firing_rates.copy()


class ConnectomeBridge:
    """
    Bridge between connectome models and brain-motor interface.

    This class provides:
    1. Neuron pool management
    2. Synaptic connectivity
    3. Signal routing between physics sensors and motor outputs
    4. Optional quantum enhancement

    Architecture:
        Sensors -> Sensory Pools -> Interneurons -> Motor Pools -> Motors
    """

    # Named neuron pools based on Drosophila anatomy
    POOL_DEFINITIONS = {
        # Sensory pools
        "mechanosensory_tarsi": NeuronType.MECHANOSENSORY,
        "proprioceptive_legs": NeuronType.PROPRIOCEPTIVE,
        "visual_motion": NeuronType.VISUAL,
        "olfactory_antenna": NeuronType.OLFACTORY,

        # Interneuron pools
        "local_circuit_legs": NeuronType.INTERNEURON,
        "descending_commands": NeuronType.INTERNEURON,
        "central_pattern_generator": NeuronType.CLOCK,
        "visual_integration": NeuronType.INTERNEURON,

        # Motor pools
        "leg_motor_front": NeuronType.MOTOR,
        "leg_motor_middle": NeuronType.MOTOR,
        "leg_motor_hind": NeuronType.MOTOR,
        "wing_motor": NeuronType.MOTOR,
        "neck_motor": NeuronType.MOTOR,
    }

    def __init__(
        self,
        num_neurons_per_pool: int = 100,
        enable_plasticity: bool = False,
        quantum_enhanced: bool = False,
    ):
        """
        Initialize connectome bridge.

        Args:
            num_neurons_per_pool: Default number of neurons per pool
            enable_plasticity: Enable synaptic plasticity
            quantum_enhanced: Enable quantum enhancement (experimental)
        """
        self.num_neurons_per_pool = num_neurons_per_pool
        self.enable_plasticity = enable_plasticity
        self.quantum_enhanced = quantum_enhanced

        # Create neuron pools
        self.pools: dict[str, NeuronPool] = {}
        for name, ntype in self.POOL_DEFINITIONS.items():
            self.pools[name] = NeuronPool(
                name=name,
                neuron_type=ntype,
                num_neurons=num_neurons_per_pool,
            )

        # Synaptic connections between pools
        self.connections: list[SynapseModel] = []

        # Build default connectivity
        self._build_default_connectivity()

        # Quantum enhancement (lazy import)
        self._quantum_backend = None
        if quantum_enhanced:
            self._init_quantum_backend()

        # Simulation time
        self.time = 0.0  # ms

    def _build_default_connectivity(self):
        """Build default synaptic connectivity between pools."""
        # Sensory -> Local circuit
        self._connect_pools(
            "mechanosensory_tarsi", "local_circuit_legs",
            weight=1.0, connection_prob=0.3
        )
        self._connect_pools(
            "proprioceptive_legs", "local_circuit_legs",
            weight=0.8, connection_prob=0.4
        )
        self._connect_pools(
            "visual_motion", "visual_integration",
            weight=1.2, connection_prob=0.2
        )

        # Local circuit -> CPG
        self._connect_pools(
            "local_circuit_legs", "central_pattern_generator",
            weight=0.6, connection_prob=0.2
        )

        # CPG -> Descending commands
        self._connect_pools(
            "central_pattern_generator", "descending_commands",
            weight=1.4, connection_prob=0.3
        )

        # Descending -> Motor pools
        self._connect_pools(
            "descending_commands", "leg_motor_front",
            weight=1.0, connection_prob=0.3
        )
        self._connect_pools(
            "descending_commands", "leg_motor_middle",
            weight=1.0, connection_prob=0.3
        )
        self._connect_pools(
            "descending_commands", "leg_motor_hind",
            weight=1.0, connection_prob=0.3
        )

        # Cross-inhibition between motor pools (for alternating gait)
        self._connect_pools(
            "leg_motor_front", "leg_motor_middle",
            weight=-0.4, connection_prob=0.1, synapse_type=SynapseType.INHIBITORY
        )
        self._connect_pools(
            "leg_motor_middle", "leg_motor_hind",
            weight=-0.4, connection_prob=0.1, synapse_type=SynapseType.INHIBITORY
        )

    def _connect_pools(
        self,
        pre_pool_name: str,
        post_pool_name: str,
        weight: float,
        connection_prob: float,
        synapse_type: SynapseType = SynapseType.EXCITATORY,
    ):
        """Create random connections between pools."""
        pre = self.pools[pre_pool_name]
        post = self.pools[post_pool_name]

        for i in range(pre.num_neurons):
            for j in range(post.num_neurons):
                if np.random.random() < connection_prob:
                    self.connections.append(SynapseModel(
                        pre_pool=pre_pool_name,
                        post_pool=post_pool_name,
                        pre_neuron=i,
                        post_neuron=j,
                        weight=weight * (0.5 + np.random.random()),
                        synapse_type=synapse_type,
                    ))

    def _init_quantum_backend(self):
        """Initialize quantum backend for enhanced computation."""
        # Placeholder for future quantum-enhanced neural processing
        self._quantum_backend = "initialized"

    def set_sensory_input(
        self,
        pool_name: str,
        input_pattern: np.ndarray,
    ):
        """
        Set input to a sensory neuron pool.

        Args:
            pool_name: Name of sensory pool
            input_pattern: Input pattern (will be broadcast to pool size)
        """
        if pool_name not in self.pools:
            raise ValueError(f"Unknown pool: {pool_name}")

        pool = self.pools[pool_name]
        if pool.neuron_type not in [
            NeuronType.SENSORY, NeuronType.MECHANOSENSORY,
            NeuronType.PROPRIOCEPTIVE, NeuronType.VISUAL,
            NeuronType.OLFACTORY
        ]:
            raise ValueError(f"{pool_name} is not a sensory pool")

        # Broadcast input to pool
        input_current = np.zeros(pool.num_neurons)
        input_current[:len(input_pattern)] = input_pattern[:pool.num_neurons]

        # Directly set firing rates (simplified)
        pool.firing_rates = np.clip(input_current, 0, 1)

    def step(self, dt: float = 0.1) -> dict[str, np.ndarray]:
        """
        Advance simulation by one time step.

        Args:
            dt: Time step in milliseconds

        Returns:
            Dictionary of pool_name -> activity pattern
        """
        self.time += dt

        # Build input currents for each pool from connections
        input_currents = {name: np.zeros(pool.num_neurons)
                         for name, pool in self.pools.items()}

        for conn in self.connections:
            pre_pool = self.pools[conn.pre_pool]
            pre_activity = pre_pool.firing_rates[conn.pre_neuron]
            input_currents[conn.post_pool][conn.post_neuron] += conn.transmit(pre_activity)

        # Update each pool
        activities = {}
        sensory_types = {NeuronType.SENSORY, NeuronType.MECHANOSENSORY,
                         NeuronType.PROPRIOCEPTIVE, NeuronType.VISUAL,
                         NeuronType.OLFACTORY}

        for pool_name, pool in self.pools.items():
            if pool.neuron_type not in sensory_types:
                pool.update(input_currents[pool_name], dt, self.time)
            activities[pool_name] = pool.get_activity()

        # Apply plasticity if enabled
        if self.enable_plasticity:
            self._apply_plasticity(activities, dt)

        return activities

    def _apply_plasticity(self, activities: dict, dt: float):
        """Apply synaptic plasticity rules."""
        for conn in self.connections:
            if conn.plasticity_rate == 0.0:
                continue

            pre_act = activities[conn.pre_pool][conn.pre_neuron]
            post_act = activities[conn.post_pool][conn.post_neuron]

            # Hebbian update
            conn.weight += conn.plasticity_rate * pre_act * post_act * dt

            # Bound weights
            if conn.synapse_type == SynapseType.EXCITATORY:
                conn.weight = np.clip(conn.weight, 0, 2.0)
            elif conn.synapse_type == SynapseType.INHIBITORY:
                conn.weight = np.clip(conn.weight, -2.0, 0)

    def get_motor_output(self) -> dict[str, np.ndarray]:
        """
        Get motor pool activities as output.

        Returns:
            Dictionary of motor_pool_name -> activity pattern
        """
        motor_pools = [
            "leg_motor_front", "leg_motor_middle", "leg_motor_hind",
            "wing_motor", "neck_motor"
        ]

        outputs = {}
        for pool_name in motor_pools:
            if pool_name in self.pools:
                outputs[pool_name] = self.pools[pool_name].get_activity()

        return outputs

    def get_brain_output_vector(self) -> np.ndarray:
        """
        Get combined brain output as a single vector.

        This can be fed to the NeuralPatternDecoder.

        Returns:
            Flattened motor activity vector
        """
        motor_outputs = self.get_motor_output()
        vectors = [output.flatten() for output in motor_outputs.values()]
        if vectors:
            return np.concatenate(vectors)
        return np.zeros(self.num_neurons_per_pool * 5)

    def get_sensor_input_dim(self) -> int:
        """Get dimension of sensor input space."""
        sensory_pools = [
            "mechanosensory_tarsi", "proprioceptive_legs",
            "visual_motion", "olfactory_antenna"
        ]
        return sum(
            self.pools[name].num_neurons
            for name in sensory_pools
            if name in self.pools
        )

    def get_motor_output_dim(self) -> int:
        """Get dimension of motor output space."""
        motor_pools = [
            "leg_motor_front", "leg_motor_middle", "leg_motor_hind",
            "wing_motor", "neck_motor"
        ]
        return sum(
            self.pools[name].num_neurons
            for name in motor_pools
            if name in self.pools
        )

    def reset(self):
        """Reset all pools to resting state."""
        self.time = 0.0
        for pool in self.pools.values():
            pool.membrane_potentials = np.full(
                pool.num_neurons, pool.resting_potential
            )
            pool.spike_times = np.full(pool.num_neurons, -1000.0)
            pool.firing_rates = np.zeros(pool.num_neurons)

    def save_state(self) -> dict:
        """Save current state for checkpointing."""
        return {
            "time": self.time,
            "pools": {
                name: {
                    "membrane_potentials": pool.membrane_potentials.copy(),
                    "spike_times": pool.spike_times.copy(),
                    "firing_rates": pool.firing_rates.copy(),
                }
                for name, pool in self.pools.items()
            },
        }

    def load_state(self, state: dict):
        """Load state from checkpoint."""
        self.time = state["time"]
        for name, pool_state in state["pools"].items():
            if name in self.pools:
                pool = self.pools[name]
                pool.membrane_potentials = pool_state["membrane_potentials"]
                pool.spike_times = pool_state["spike_times"]
                pool.firing_rates = pool_state["firing_rates"]


def create_brain_model(
    quantum_enhanced: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factory function to create a brain model callable.

    This creates a brain model that can be used with the
    BrainMotorInterface.

    Args:
        quantum_enhanced: Whether to use quantum enhancement

    Returns:
        Callable that takes sensor input and returns motor output
    """
    bridge = ConnectomeBridge(
        num_neurons_per_pool=100,
        enable_plasticity=False,
        quantum_enhanced=quantum_enhanced,
    )

    def brain_model(sensor_input: np.ndarray) -> np.ndarray:
        """Process sensor input through connectome model."""
        # Set sensory inputs
        # Split input among sensory pools
        idx = 0
        sensory_pools = [
            "mechanosensory_tarsi", "proprioceptive_legs",
            "visual_motion", "olfactory_antenna"
        ]

        for pool_name in sensory_pools:
            pool = bridge.pools[pool_name]
            end_idx = min(idx + pool.num_neurons, len(sensor_input))
            if end_idx > idx:
                bridge.set_sensory_input(
                    pool_name,
                    sensor_input[idx:end_idx]
                )
            idx = end_idx

        # Run multiple simulation steps for dynamics to settle
        for _ in range(10):
            bridge.step(dt=1.0)

        # Get motor output
        return bridge.get_brain_output_vector()

    return brain_model
