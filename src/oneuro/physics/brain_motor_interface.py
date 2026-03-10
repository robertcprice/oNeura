"""
Brain-Motor Interface

Connects neural/quantum brain models to the MuJoCo physics simulation.
Translates neural activity patterns into motor commands and encodes
sensory feedback into neural representations.

Architecture:
    Brain (neural/quantum) <-> BrainMotorInterface <-> Physics (MuJoCo)

Key Components:
    - NeuralPatternDecoder: Neural activity -> Motor primitives
    - SensorEncoder: Sensor readings -> Neural patterns
    - MotorPrimitive: Predefined motor patterns (walking, grooming, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

import numpy as np


class MotorPrimitiveType(Enum):
    """Types of motor primitives."""
    STANCE = "stance"
    SWING = "swing"
    TRIPOD_GAIT = "tripod_gait"
    WINGBEAT = "wingbeat"
    GROOMING = "grooming"
    TAKEOFF = "takeoff"
    LANDING = "landing"
    IDLE = "idle"


@dataclass
class MotorPrimitive:
    """
    A motor primitive represents a coordinated movement pattern.

    Attributes:
        name: Primitive name
        primitive_type: Type of primitive
        joint_targets: Dictionary of joint_name -> target angle
        duration: Duration of primitive in seconds
        phase_offset: Phase offset for cyclic primitives
        intensity: Intensity multiplier [0, 1]
    """
    name: str
    primitive_type: MotorPrimitiveType
    joint_targets: dict[str, float]
    duration: float = 0.1
    phase_offset: float = 0.0
    intensity: float = 1.0


class SensorEncoder:
    """
    Encodes sensor readings into neural patterns.

    Transforms physics sensor data (joint angles, touch, vision)
    into spike rates or neural activity patterns suitable for
    brain model input.
    """

    def __init__(
        self,
        num_proprioceptive: int = 48,  # 24 joint angles + 24 velocities
        num_touch: int = 6,            # Tarsus contact sensors
        num_visual: int = 64,          # Simplified visual input
        encoding_type: str = "rate",
    ):
        """
        Initialize sensor encoder.

        Args:
            num_proprioceptive: Number of proprioceptive neurons
            num_touch: Number of touch-sensitive neurons
            num_visual: Number of visual input neurons
            encoding_type: "rate" for firing rate, "spike" for binary spikes
        """
        self.num_proprioceptive = num_proprioceptive
        self.num_touch = num_touch
        self.num_visual = num_visual
        self.encoding_type = encoding_type

        # Total output dimension
        self.output_dim = num_proprioceptive + num_touch + num_visual

        # Sensor normalization ranges
        self.joint_angle_range = (-2.0, 2.0)  # radians
        self.joint_velocity_range = (-20.0, 20.0)  # rad/s

        # Adaptive normalization parameters
        self._running_mean = np.zeros(self.output_dim)
        self._running_var = np.ones(self.output_dim)
        self._adaptation_rate = 0.01

    def encode(
        self,
        joint_positions: dict[str, float],
        joint_velocities: dict[str, float],
        touch_contacts: dict,
        visual_input: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Encode sensor readings into neural pattern.

        Args:
            joint_positions: Dict of joint_name -> angle
            joint_velocities: Dict of joint_name -> angular velocity
            touch_contacts: Dict of leg_id -> bool
            visual_input: Optional visual input array

        Returns:
            Neural activity pattern (firing rates or spikes)
        """
        neural_pattern = np.zeros(self.output_dim)
        idx = 0

        # Encode proprioceptive (joint angles and velocities, including tarsus)
        joint_order = [
            "C1_L_pitch", "F1_L_pitch", "T1_L_pitch", "Ta1_L_pitch",
            "C2_L_pitch", "F2_L_pitch", "T2_L_pitch", "Ta2_L_pitch",
            "C3_L_pitch", "F3_L_pitch", "T3_L_pitch", "Ta3_L_pitch",
            "C1_R_pitch", "F1_R_pitch", "T1_R_pitch", "Ta1_R_pitch",
            "C2_R_pitch", "F2_R_pitch", "T2_R_pitch", "Ta2_R_pitch",
            "C3_R_pitch", "F3_R_pitch", "T3_R_pitch", "Ta3_R_pitch",
        ]

        for jname in joint_order:
            # Joint angle -> normalized firing rate
            angle = joint_positions.get(jname, 0.0)
            normalized = self._normalize(angle, self.joint_angle_range)
            if idx < self.num_proprioceptive:
                neural_pattern[idx] = self._to_rate(normalized)
                idx += 1

        # Encode joint velocities
        for jname in joint_order:
            vel = joint_velocities.get(jname, 0.0)
            normalized = self._normalize(vel, self.joint_velocity_range)
            if idx < self.num_proprioceptive:
                neural_pattern[idx] = self._to_rate(normalized)
                idx += 1

        # Encode touch contacts
        from .drosophila_simulator import LegID
        touch_idx = self.num_proprioceptive
        for leg in [LegID.LEFT_FRONT, LegID.LEFT_MIDDLE, LegID.LEFT_HIND,
                    LegID.RIGHT_FRONT, LegID.RIGHT_MIDDLE, LegID.RIGHT_HIND]:
            if touch_idx < self.num_proprioceptive + self.num_touch:
                contact = touch_contacts.get(leg, False)
                neural_pattern[touch_idx] = 1.0 if contact else 0.0
                touch_idx += 1

        # Encode visual input
        if visual_input is not None:
            visual_start = self.num_proprioceptive + self.num_touch
            visual_end = min(visual_start + len(visual_input), self.output_dim)
            neural_pattern[visual_start:visual_end] = visual_input[:visual_end - visual_start]
        else:
            # Default visual input (no stimulus)
            visual_start = self.num_proprioceptive + self.num_touch
            neural_pattern[visual_start:] = 0.0

        # Adaptive normalization
        self._update_running_stats(neural_pattern)

        return neural_pattern

    def _normalize(self, value: float, range_: tuple) -> float:
        """Normalize value to [0, 1] range."""
        min_val, max_val = range_
        return (value - min_val) / (max_val - min_val)

    def _to_rate(self, normalized: float) -> float:
        """Convert normalized value to firing rate."""
        # Map [0, 1] to [0, 1] with some nonlinearity
        return np.clip(normalized, 0.0, 1.0)

    def _update_running_stats(self, pattern: np.ndarray):
        """Update running mean and variance for adaptive normalization."""
        self._running_mean = (
            (1 - self._adaptation_rate) * self._running_mean +
            self._adaptation_rate * pattern
        )
        self._running_var = (
            (1 - self._adaptation_rate) * self._running_var +
            self._adaptation_rate * (pattern - self._running_mean) ** 2
        )

    def get_normalized(self, pattern: np.ndarray) -> np.ndarray:
        """Get z-normalized pattern using running statistics."""
        return (pattern - self._running_mean) / (np.sqrt(self._running_var) + 1e-8)


class NeuralPatternDecoder:
    """
    Decodes neural activity patterns into motor commands.

    Transforms brain output (neural activity, spike rates, or
    quantum measurement outcomes) into motor primitive selection
    and execution parameters.
    """

    def __init__(
        self,
        num_output_neurons: int = 32,
        num_motor_primitives: int = 8,
        decoding_type: str = "population",
    ):
        """
        Initialize neural pattern decoder.

        Args:
            num_output_neurons: Number of neurons in brain output layer
            num_motor_primitives: Number of available motor primitives
            decoding_type: "population" for population vector, "wta" for winner-take-all
        """
        self.num_output_neurons = num_output_neurons
        self.num_motor_primitives = num_motor_primitives
        self.decoding_type = decoding_type

        # Population vectors for each motor primitive
        # Each row is the preferred direction in neural space
        self._population_vectors = np.random.randn(
            num_motor_primitives, num_output_neurons
        )
        self._population_vectors /= np.linalg.norm(
            self._population_vectors, axis=1, keepdims=True
        )

        # Primitive intensity weights
        self._intensity_weights = np.random.randn(num_output_neurons)
        self._intensity_bias = 0.0

        # Smoothing for temporal stability
        self._prev_primitive = None
        self._smoothing = 0.7

    def decode(
        self,
        neural_pattern: np.ndarray,
        available_primitives: Optional[list[MotorPrimitive]] = None,
    ) -> tuple[MotorPrimitiveType, float, dict[str, float]]:
        """
        Decode neural pattern into motor output.

        Args:
            neural_pattern: Neural activity pattern from brain
            available_primitives: List of available primitives (optional)

        Returns:
            Tuple of (primitive_type, intensity, joint_modulations)
        """
        if available_primitives is None:
            available_primitives = list(MotorPrimitiveType)

        # Ensure pattern is correct size
        if len(neural_pattern) != self.num_output_neurons:
            # Pad or truncate
            pattern = np.zeros(self.num_output_neurons)
            pattern[:min(len(neural_pattern), self.num_output_neurons)] = \
                neural_pattern[:min(len(neural_pattern), self.num_output_neurons)]
        else:
            pattern = neural_pattern

        # Decode primitive type
        if self.decoding_type == "population":
            primitive_idx = self._decode_population(pattern)
        elif self.decoding_type == "wta":
            primitive_idx = self._decode_wta(pattern)
        else:
            primitive_idx = self._decode_direct(pattern)

        primitive_type = MotorPrimitiveType(
            list(MotorPrimitiveType)[primitive_idx % len(MotorPrimitiveType)]
        )

        # Apply temporal smoothing
        if self._prev_primitive is not None and self._smoothing > 0:
            if primitive_idx != self._prev_primitive:
                # Only switch if confidence is high enough
                confidence = self._get_confidence(pattern, primitive_idx)
                if confidence < 0.6:
                    primitive_type = MotorPrimitiveType(
                        list(MotorPrimitiveType)[self._prev_primitive]
                    )
                    primitive_idx = self._prev_primitive

        self._prev_primitive = primitive_idx

        # Decode intensity
        intensity = self._decode_intensity(pattern)

        # Decode joint modulations (fine adjustments to primitive)
        joint_modulations = self._decode_modulations(pattern)

        return primitive_type, intensity, joint_modulations

    def _decode_population(self, pattern: np.ndarray) -> int:
        """Decode using population vector method."""
        # Compute cosine similarity with each primitive's population vector
        similarities = self._population_vectors @ pattern
        return int(np.argmax(similarities))

    def _decode_wta(self, pattern: np.ndarray) -> int:
        """Decode using winner-take-all."""
        # Each neuron votes for a primitive
        votes = np.zeros(self.num_motor_primitives)
        for i, activation in enumerate(pattern):
            primitive_vote = i % self.num_motor_primitives
            votes[primitive_vote] += activation
        return int(np.argmax(votes))

    def _decode_direct(self, pattern: np.ndarray) -> int:
        """Direct decoding from first few neurons."""
        # Use softmax on first num_motor_primitives neurons
        logits = pattern[:self.num_motor_primitives]
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        return int(np.argmax(probs))

    def _decode_intensity(self, pattern: np.ndarray) -> float:
        """Decode movement intensity from neural pattern."""
        intensity = np.dot(self._intensity_weights, pattern) + self._intensity_bias
        return float(np.clip(1 / (1 + np.exp(-intensity)), 0.0, 1.0))  # Sigmoid

    def _decode_modulations(self, pattern: np.ndarray) -> dict[str, float]:
        """Decode fine joint modulations."""
        # Use remaining neurons to modulate joint angles
        modulations = {}

        joint_names = [
            "C1_L_pitch", "F1_L_pitch", "T1_L_pitch",
            "C1_R_pitch", "F1_R_pitch", "T1_R_pitch",
        ]

        for i, jname in enumerate(joint_names):
            if self.num_motor_primitives + i < len(pattern):
                # Small modulation (-0.1 to 0.1 radians)
                modulations[jname] = 0.1 * pattern[self.num_motor_primitives + i]

        return modulations

    def _get_confidence(self, pattern: np.ndarray, primitive_idx: int) -> float:
        """Get confidence score for selected primitive."""
        similarities = self._population_vectors @ pattern
        similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities) + 1e-8)
        return float(similarities[primitive_idx])

    def train_population_vectors(
        self,
        patterns: list[np.ndarray],
        primitive_labels: list[int],
        learning_rate: float = 0.1,
    ):
        """
        Train population vectors using labeled examples.

        Args:
            patterns: List of neural patterns
            primitive_labels: Corresponding primitive indices
            learning_rate: Learning rate for updates
        """
        for pattern, label in zip(patterns, primitive_labels):
            # Move population vector toward pattern
            self._population_vectors[label] += learning_rate * (
                pattern - self._population_vectors[label]
            )
            # Renormalize
            self._population_vectors[label] /= np.linalg.norm(
                self._population_vectors[label]
            ) + 1e-8


class BrainMotorInterface:
    """
    Main interface connecting brain models to physics simulation.

    This class provides the bidirectional interface between:
    - Upstream: Neural/quantum brain models (connectome simulation)
    - Downstream: MuJoCo physics simulation

    Data flow:
        1. Physics sensors -> SensorEncoder -> Neural pattern
        2. Neural pattern -> Brain model -> Output pattern
        3. Output pattern -> NeuralPatternDecoder -> Motor commands
        4. Motor commands -> Physics actuators

    Usage:
        >>> interface = BrainMotorInterface()
        >>> physics_state = simulator.get_state()
        >>> motor_commands = interface.process(physics_state, brain_output)
        >>> new_state = simulator.step(motor_commands)
    """

    def __init__(
        self,
        sensor_encoder: Optional[SensorEncoder] = None,
        pattern_decoder: Optional[NeuralPatternDecoder] = None,
        brain_model: Optional[Callable] = None,
    ):
        """
        Initialize brain-motor interface.

        Args:
            sensor_encoder: Custom sensor encoder
            pattern_decoder: Custom pattern decoder
            brain_model: Callable brain model (takes neural pattern, returns output)
        """
        self.sensor_encoder = sensor_encoder or SensorEncoder()
        self.pattern_decoder = pattern_decoder or NeuralPatternDecoder()
        self.brain_model = brain_model

        # Motor primitive library
        self._primitives = self._build_primitive_library()

        # State tracking
        self._current_primitive = MotorPrimitiveType.IDLE
        self._primitive_time = 0.0
        self._gait_phase = 0.0

        # History for analysis
        self._sensor_history = []
        self._motor_history = []

    def _build_primitive_library(self) -> dict[MotorPrimitiveType, MotorPrimitive]:
        """Build library of motor primitives."""
        from .drosophila_simulator import MotorPatterns

        primitives = {}

        # Tripod gait primitive
        primitives[MotorPrimitiveType.TRIPOD_GAIT] = MotorPrimitive(
            name="tripod_gait",
            primitive_type=MotorPrimitiveType.TRIPOD_GAIT,
            joint_targets={},
            duration=0.05,  # 50ms per step
        )

        # Stance and swing for each leg
        for leg in ["L1", "L2", "L3", "R1", "R2", "R3"]:
            primitives[MotorPrimitiveType.STANCE] = MotorPrimitive(
                name="stance",
                primitive_type=MotorPrimitiveType.STANCE,
                joint_targets=MotorPatterns.stance_phase(
                    self._leg_str_to_id(leg)
                ),
                duration=0.05,
            )
            primitives[MotorPrimitiveType.SWING] = MotorPrimitive(
                name="swing",
                primitive_type=MotorPrimitiveType.SWING,
                joint_targets=MotorPatterns.swing_phase(
                    self._leg_str_to_id(leg)
                ),
                duration=0.05,
            )

        # Wingbeat
        primitives[MotorPrimitiveType.WINGBEAT] = MotorPrimitive(
            name="wingbeat",
            primitive_type=MotorPrimitiveType.WINGBEAT,
            joint_targets={},
            duration=0.004,  # 250 Hz = 4ms period
        )

        # Idle
        primitives[MotorPrimitiveType.IDLE] = MotorPrimitive(
            name="idle",
            primitive_type=MotorPrimitiveType.IDLE,
            joint_targets={},
            duration=float('inf'),
        )

        return primitives

    def _leg_str_to_id(self, leg_str: str):
        """Convert leg string to LegID."""
        from .drosophila_simulator import LegID
        mapping = {
            "L1": LegID.LEFT_FRONT,
            "L2": LegID.LEFT_MIDDLE,
            "L3": LegID.LEFT_HIND,
            "R1": LegID.RIGHT_FRONT,
            "R2": LegID.RIGHT_MIDDLE,
            "R3": LegID.RIGHT_HIND,
        }
        return mapping.get(leg_str)

    def process(
        self,
        physics_state,
        brain_output: Optional[np.ndarray] = None,
        dt: float = 0.0001,
    ) -> list:
        """
        Process one cycle of sensor -> brain -> motor.

        Args:
            physics_state: DrosophilaPhysicsState from simulator
            brain_output: Optional pre-computed brain output
            dt: Time step in seconds

        Returns:
            List of MotorCommand objects
        """
        # Step 1: Encode sensors to neural pattern
        sensor_pattern = self.sensor_encoder.encode(
            joint_positions=physics_state.joint_positions,
            joint_velocities=physics_state.joint_velocities,
            touch_contacts=physics_state.touch_contacts,
        )

        # Step 2: Process through brain model
        if brain_output is None and self.brain_model is not None:
            brain_output = self.brain_model(sensor_pattern)
        elif brain_output is None:
            # Default: reflexive behavior based on sensors
            brain_output = self._reflex_controller(sensor_pattern, physics_state)

        # Step 3: Decode to motor primitive
        primitive_type, intensity, modulations = self.pattern_decoder.decode(
            brain_output
        )

        # Step 4: Generate motor commands
        commands = self._generate_commands(
            primitive_type,
            intensity,
            modulations,
            physics_state.timestamp,
            dt,
        )

        # Update state
        self._current_primitive = primitive_type
        self._primitive_time += dt

        # Store history
        self._sensor_history.append(sensor_pattern.copy())
        self._motor_history.append([c.__dict__.copy() for c in commands])

        # Limit history length
        max_history = 1000
        if len(self._sensor_history) > max_history:
            self._sensor_history = self._sensor_history[-max_history:]
            self._motor_history = self._motor_history[-max_history:]

        return commands

    def _reflex_controller(
        self,
        sensor_pattern: np.ndarray,
        physics_state,
    ) -> np.ndarray:
        """
        Simple reflex controller for basic locomotion.

        Generates walking behavior based on touch feedback.
        """
        output = np.zeros(self.pattern_decoder.num_output_neurons)

        # Check for ground contact
        touch_start = self.sensor_encoder.num_proprioceptive
        has_contact = [
            sensor_pattern[touch_start + i] > 0.5
            for i in range(6)
        ]

        # Simple tripod gait trigger
        # Use index of the enum member, not string .value
        primitive_list = list(MotorPrimitiveType)
        num_contacts = sum(has_contact)
        if num_contacts >= 3:
            # Walking
            tripod_idx = primitive_list.index(MotorPrimitiveType.TRIPOD_GAIT)
            output[tripod_idx] = 1.0
        else:
            # Might be falling, try to stabilize
            stance_idx = primitive_list.index(MotorPrimitiveType.STANCE)
            output[stance_idx] = 0.5

        return output

    def _generate_commands(
        self,
        primitive_type: MotorPrimitiveType,
        intensity: float,
        modulations: dict[str, float],
        timestamp: float,
        dt: float,
    ) -> list:
        """Generate motor commands for selected primitive."""
        from .drosophila_simulator import MotorPatterns, MotorCommand

        commands = []

        if primitive_type == MotorPrimitiveType.TRIPOD_GAIT:
            # Advance gait phase
            step_duration = 0.05  # 50ms per half-cycle
            self._gait_phase = (self._gait_phase + dt / step_duration) % 1.0

            # Get joint targets for current phase
            targets = MotorPatterns.tripod_gait(self._gait_phase)

            for joint_name, target_angle in targets.items():
                motor_name = joint_name.replace("_pitch", "_motor")
                if "C" in joint_name:
                    motor_name = motor_name.replace("C", "C").replace("_L_", "_L_")
                commands.append(MotorCommand(
                    actuator_name=motor_name,
                    control_signal=target_angle * intensity,
                    timestamp=timestamp,
                ))

        elif primitive_type == MotorPrimitiveType.WINGBEAT:
            # Wingbeat at 250 Hz
            wingbeat_freq = 250.0
            targets = MotorPatterns.wingbeat(wingbeat_freq, timestamp)

            for wing_name, angle in targets.items():
                commands.append(MotorCommand(
                    actuator_name=f"{wing_name}_motor",
                    control_signal=angle * intensity,
                    timestamp=timestamp,
                ))

        elif primitive_type == MotorPrimitiveType.STANCE:
            targets = MotorPatterns.stance_phase(None)
            for joint_name, angle in targets.items():
                motor_name = joint_name.replace("_pitch", "_motor")
                commands.append(MotorCommand(
                    actuator_name=motor_name,
                    control_signal=angle * intensity,
                    timestamp=timestamp,
                ))

        # Apply modulations
        for joint_name, modulation in modulations.items():
            motor_name = joint_name.replace("_pitch", "_motor")
            for cmd in commands:
                if cmd.actuator_name == motor_name:
                    cmd.control_signal += modulation
                    cmd.control_signal = np.clip(cmd.control_signal, -1.0, 1.0)
                    break

        return commands

    def get_sensor_dim(self) -> int:
        """Get sensor encoding dimension."""
        return self.sensor_encoder.output_dim

    def get_motor_dim(self) -> int:
        """Get motor output dimension."""
        return len(self._primitives)

    def reset(self):
        """Reset interface state."""
        self._current_primitive = MotorPrimitiveType.IDLE
        self._primitive_time = 0.0
        self._gait_phase = 0.0
        self._sensor_history = []
        self._motor_history = []
