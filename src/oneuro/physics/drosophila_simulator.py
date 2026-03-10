"""
Drosophila Physics Simulator

MuJoCo-based physics simulation for Drosophila melanogaster.
Provides realistic body dynamics, leg kinematics, and wing actuation
at biological scale (~1mg, ~2.5mm).

Reference: NeuroMechFly (Lobato-Rios et al., 2022)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

# MuJoCo import with graceful fallback
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None


class LegID(Enum):
    """Drosophila leg identifiers."""
    LEFT_FRONT = "L1"
    LEFT_MIDDLE = "L2"
    LEFT_HIND = "L3"
    RIGHT_FRONT = "R1"
    RIGHT_MIDDLE = "R2"
    RIGHT_HIND = "R3"


class JointType(Enum):
    """Leg joint types."""
    COXA = "C"      # Body to coxa (yaw + pitch)
    FEMUR = "F"     # Coxa to femur (pitch)
    TIBIA = "T"     # Femur to tibia (pitch)
    TARUS = "Ta"    # Tibia to tarsus (pitch)


@dataclass
class MotorCommand:
    """
    Motor command for a single actuator.

    Attributes:
        actuator_name: MuJoCo actuator name
        control_signal: Normalized control signal [-1, 1]
        timestamp: Command timestamp in seconds
    """
    actuator_name: str
    control_signal: float
    timestamp: float = 0.0

    def __post_init__(self):
        self.control_signal = np.clip(self.control_signal, -1.0, 1.0)


@dataclass
class SensorReading:
    """
    Sensor reading from the physics simulation.

    Attributes:
        sensor_name: MuJoCo sensor name
        value: Sensor value (scalar or array)
        timestamp: Reading timestamp in seconds
    """
    sensor_name: str
    value: np.ndarray
    timestamp: float = 0.0


@dataclass
class DrosophilaPhysicsState:
    """
    Complete physics state of the simulated Drosophila.

    Attributes:
        body_position: 3D position [x, y, z] in meters
        body_quaternion: Orientation [w, x, y, z]
        body_velocity: Linear velocity [vx, vy, vz] in m/s
        body_angular_velocity: Angular velocity [wx, wy, wz] in rad/s
        joint_positions: Dict of joint_name -> angle (radians)
        joint_velocities: Dict of joint_name -> angular velocity (rad/s)
        touch_contacts: Dict of leg_id -> bool (ground contact)
        wing_angles: Dict of wing_name -> angle (radians)
        timestamp: Simulation time in seconds
    """
    body_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    body_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joint_positions: dict = field(default_factory=dict)
    joint_velocities: dict = field(default_factory=dict)
    touch_contacts: dict = field(default_factory=dict)
    wing_angles: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert state to flat vector for neural network input."""
        components = [
            self.body_position,
            self.body_quaternion,
            self.body_velocity,
            self.body_angular_velocity,
        ]

        # Add joint positions in consistent order (all 24 leg joints)
        joint_order = [
            "C1_L_pitch", "F1_L_pitch", "T1_L_pitch", "Ta1_L_pitch",
            "C2_L_pitch", "F2_L_pitch", "T2_L_pitch", "Ta2_L_pitch",
            "C3_L_pitch", "F3_L_pitch", "T3_L_pitch", "Ta3_L_pitch",
            "C1_R_pitch", "F1_R_pitch", "T1_R_pitch", "Ta1_R_pitch",
            "C2_R_pitch", "F2_R_pitch", "T2_R_pitch", "Ta2_R_pitch",
            "C3_R_pitch", "F3_R_pitch", "T3_R_pitch", "Ta3_R_pitch",
        ]
        for jname in joint_order:
            components.append(np.array([self.joint_positions.get(jname, 0.0)]))

        # Add wing angles
        components.append(np.array([self.wing_angles.get("wing_L", 0.0)]))
        components.append(np.array([self.wing_angles.get("wing_R", 0.0)]))

        # Add touch contacts
        for leg in LegID:
            components.append(np.array([1.0 if self.touch_contacts.get(leg, False) else 0.0]))

        return np.concatenate(components)

    @property
    def observation_dim(self) -> int:
        """Dimension of the observation vector."""
        return len(self.to_vector())


class DrosophilaSimulator:
    """
    MuJoCo-based Drosophila melanogaster physics simulator.

    Provides:
    - Realistic body dynamics at biological scale
    - 6 legs with 4 actuated joints each (24 DoF)
    - 2 wings with position actuators
    - Ground contact sensing via touch sensors
    - Integration with brain motor interface

    Example:
        >>> simulator = DrosophilaSimulator()
        >>> state = simulator.reset()
        >>> # Apply motor commands
        >>> commands = [MotorCommand("F1_L_motor", 0.5)]
        >>> new_state = simulator.step(commands)
    """

    # Actuator names matching drosophila_mjcf.xml
    WING_ACTUATORS = ["wing_L_motor", "wing_R_motor"]
    LEG_ACTUATORS = [
        # Left legs (coxa, femur, tibia, tarsus each)
        "C1_L_motor", "F1_L_motor", "T1_L_motor", "Ta1_L_motor",
        "C2_L_motor", "F2_L_motor", "T2_L_motor", "Ta2_L_motor",
        "C3_L_motor", "F3_L_motor", "T3_L_motor", "Ta3_L_motor",
        # Right legs
        "C1_R_motor", "F1_R_motor", "T1_R_motor", "Ta1_R_motor",
        "C2_R_motor", "F2_R_motor", "T2_R_motor", "Ta2_R_motor",
        "C3_R_motor", "F3_R_motor", "T3_R_motor", "Ta3_R_motor",
    ]

    ALL_ACTUATORS = WING_ACTUATORS + LEG_ACTUATORS

    # Sensor names
    TOUCH_SENSORS = ["touch_L1", "touch_L2", "touch_L3",
                     "touch_R1", "touch_R2", "touch_R3"]

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        timestep: float = 1e-4,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the Drosophila simulator.

        Args:
            mjcf_path: Path to MJCF XML file. Defaults to included model.
            timestep: Physics timestep in seconds.
            render_mode: "human" for real-time display, None for headless.
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError(
                "MuJoCo is not installed. Install with: pip install mujoco"
            )

        # Load MJCF model
        if mjcf_path is None:
            mjcf_path = Path(__file__).parent / "drosophila_mjcf.xml"

        self.mjcf_path = Path(mjcf_path)
        self.timestep = timestep
        self.render_mode = render_mode

        # Load model and create data
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)

        # Override timestep if specified
        if timestep != self.model.opt.timestep:
            self.model.opt.timestep = timestep

        # Build actuator and sensor indices for fast lookup
        self._actuator_indices = {
            name: i for i, name in enumerate(self.ALL_ACTUATORS)
            if i < self.model.nu
        }
        self._sensor_indices = {}
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                self._sensor_indices[name] = i

        # Create actuator index array
        self._actuator_idx_map = {}
        for name in self.ALL_ACTUATORS:
            try:
                idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self._actuator_idx_map[name] = idx
            except:
                pass

        # Viewer for rendering
        self._viewer = None

        # Simulation state
        self._time = 0.0
        self._step_count = 0

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self._time

    @property
    def num_actuators(self) -> int:
        """Total number of actuators."""
        return len(self._actuator_idx_map)

    @property
    def num_sensors(self) -> int:
        """Total number of sensors."""
        return self.model.nsensor

    def reset(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> DrosophilaPhysicsState:
        """
        Reset simulation to initial state.

        Args:
            position: Initial body position [x, y, z]. Default: [0, 0, 0.005]
            orientation: Initial orientation quaternion [w, x, y, z]. Default: identity

        Returns:
            Initial physics state
        """
        mujoco.mj_resetData(self.model, self.data)

        # Set initial position
        if position is not None:
            self.data.qpos[:3] = position
        else:
            self.data.qpos[:3] = [0, 0, 0.005]  # 5mm above ground

        # Set initial orientation
        if orientation is not None:
            self.data.qpos[3:7] = orientation
        else:
            self.data.qpos[3:7] = [1, 0, 0, 0]  # Identity quaternion

        # Forward dynamics to initialize derived quantities
        mujoco.mj_forward(self.model, self.data)

        self._time = 0.0
        self._step_count = 0

        return self.get_state()

    def step(
        self,
        motor_commands: list[MotorCommand],
        num_substeps: int = 1,
    ) -> DrosophilaPhysicsState:
        """
        Apply motor commands and advance simulation.

        Args:
            motor_commands: List of motor commands to apply
            num_substeps: Number of physics substeps

        Returns:
            Physics state after stepping
        """
        # Apply motor commands
        for cmd in motor_commands:
            if cmd.actuator_name in self._actuator_idx_map:
                idx = self._actuator_idx_map[cmd.actuator_name]
                self.data.ctrl[idx] = cmd.control_signal

        # Step simulation
        for _ in range(num_substeps):
            mujoco.mj_step(self.model, self.data)

        self._time += self.timestep * num_substeps
        self._step_count += num_substeps

        return self.get_state()

    def step_with_vector(
        self,
        control_vector: np.ndarray,
        num_substeps: int = 1,
    ) -> DrosophilaPhysicsState:
        """
        Step simulation with a control vector.

        Args:
            control_vector: Flat array of control signals matching actuator order
            num_substeps: Number of physics substeps

        Returns:
            Physics state after stepping
        """
        # Clip and apply control
        control_vector = np.clip(control_vector, -1.0, 1.0)
        self.data.ctrl[:len(control_vector)] = control_vector

        # Step simulation
        for _ in range(num_substeps):
            mujoco.mj_step(self.model, self.data)

        self._time += self.timestep * num_substeps
        self._step_count += num_substeps

        return self.get_state()

    def get_state(self) -> DrosophilaPhysicsState:
        """
        Get current physics state.

        Returns:
            Complete physics state of the simulated fly
        """
        state = DrosophilaPhysicsState()

        # Body state
        state.body_position = self.data.qpos[:3].copy()
        state.body_quaternion = self.data.qpos[3:7].copy()
        state.body_velocity = self.data.qvel[:3].copy()
        state.body_angular_velocity = self.data.qvel[3:6].copy()

        # Joint positions and velocities (matching drosophila_mjcf.xml)
        joint_names = [
            "neck_pitch", "proboscis_extend",
            "wing_L", "wing_R",
            "C1_L_yaw", "C1_L_pitch", "F1_L_pitch", "T1_L_pitch", "Ta1_L_pitch",
            "C2_L_yaw", "C2_L_pitch", "F2_L_pitch", "T2_L_pitch", "Ta2_L_pitch",
            "C3_L_yaw", "C3_L_pitch", "F3_L_pitch", "T3_L_pitch", "Ta3_L_pitch",
            "C1_R_yaw", "C1_R_pitch", "F1_R_pitch", "T1_R_pitch", "Ta1_R_pitch",
            "C2_R_yaw", "C2_R_pitch", "F2_R_pitch", "T2_R_pitch", "Ta2_R_pitch",
            "C3_R_yaw", "C3_R_pitch", "F3_R_pitch", "T3_R_pitch", "Ta3_R_pitch",
            "abdomen_pitch",
        ]

        for i, name in enumerate(joint_names):
            try:
                jnt_adr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qpos_adr = self.model.jnt_qposadr[jnt_adr]
                qvel_adr = self.model.jnt_dofadr[jnt_adr]
                state.joint_positions[name] = self.data.qpos[qpos_adr]
                state.joint_velocities[name] = self.data.qvel[qvel_adr]
            except:
                pass

        # Wing angles
        state.wing_angles["wing_L"] = state.joint_positions.get("wing_L", 0.0)
        state.wing_angles["wing_R"] = state.joint_positions.get("wing_R", 0.0)

        # Touch contacts
        touch_mapping = {
            "touch_L1": LegID.LEFT_FRONT,
            "touch_L2": LegID.LEFT_MIDDLE,
            "touch_L3": LegID.LEFT_HIND,
            "touch_R1": LegID.RIGHT_FRONT,
            "touch_R2": LegID.RIGHT_MIDDLE,
            "touch_R3": LegID.RIGHT_HIND,
        }

        for sensor_name, leg_id in touch_mapping.items():
            if sensor_name in self._sensor_indices:
                idx = self._sensor_indices[sensor_name]
                adr = self.model.sensor_adr[idx]
                # Touch sensor: positive value indicates contact
                state.touch_contacts[leg_id] = self.data.sensordata[adr] > 0.0

        state.timestamp = self._time
        return state

    def get_sensor_readings(self) -> list[SensorReading]:
        """
        Get all sensor readings.

        Returns:
            List of sensor readings
        """
        readings = []

        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            adr = self.model.sensor_adr[i]
            dim = self.model.sensor_dim[i]

            value = self.data.sensordata[adr:adr+dim].copy()
            readings.append(SensorReading(
                sensor_name=name,
                value=value,
                timestamp=self._time,
            ))

        return readings

    def render(self) -> Optional[np.ndarray]:
        """
        Render current state.

        Returns:
            RGB image array if render_mode is "rgb_array", else None
        """
        if self._viewer is None and self.render_mode == "human":
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        if self._viewer is not None:
            self._viewer.sync()

        return None

    def close(self):
        """Clean up resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def apply_perturbation(
        self,
        force: np.ndarray,
        torque: np.ndarray = None,
    ):
        """
        Apply external perturbation to the body.

        Args:
            force: External force [fx, fy, fz] in Newtons
            torque: External torque [tx, ty, tz] in Nm (optional)
        """
        self.data.xfrc_applied[0, :3] = force
        if torque is not None:
            self.data.xfrc_applied[0, 3:6] = torque

    def get_body_velocity_world(self) -> np.ndarray:
        """Get body velocity in world frame."""
        return self.data.qvel[:3].copy()

    def get_body_velocity_body(self) -> np.ndarray:
        """Get body velocity in body frame."""
        # Transform world velocity to body frame
        quat = self.data.qpos[3:7]
        vel_world = self.data.qvel[:3]

        # Inverse rotation
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        vel_body = rot.T @ vel_world

        return vel_body


# Predefined motor patterns for common behaviors
class MotorPatterns:
    """Predefined motor patterns for Drosophila behaviors."""

    @staticmethod
    def stance_phase(leg: LegID) -> dict[str, float]:
        """Get joint angles for stance phase (ground contact)."""
        patterns = {
            LegID.LEFT_FRONT: {"C1_L_pitch": 0.3, "F1_L_pitch": 0.8, "T1_L_pitch": -0.5, "Ta1_L_pitch": 0.2},
            LegID.LEFT_MIDDLE: {"C2_L_pitch": 0.2, "F2_L_pitch": 0.9, "T2_L_pitch": -0.6, "Ta2_L_pitch": 0.2},
            LegID.LEFT_HIND: {"C3_L_pitch": 0.1, "F3_L_pitch": 1.0, "T3_L_pitch": -0.7, "Ta3_L_pitch": 0.2},
            LegID.RIGHT_FRONT: {"C1_R_pitch": 0.3, "F1_R_pitch": 0.8, "T1_R_pitch": -0.5, "Ta1_R_pitch": 0.2},
            LegID.RIGHT_MIDDLE: {"C2_R_pitch": 0.2, "F2_R_pitch": 0.9, "T2_R_pitch": -0.6, "Ta2_R_pitch": 0.2},
            LegID.RIGHT_HIND: {"C3_R_pitch": 0.1, "F3_R_pitch": 1.0, "T3_R_pitch": -0.7, "Ta3_R_pitch": 0.2},
        }
        return patterns.get(leg, {})

    @staticmethod
    def swing_phase(leg: LegID) -> dict[str, float]:
        """Get joint angles for swing phase (leg in air)."""
        patterns = {
            LegID.LEFT_FRONT: {"C1_L_pitch": -0.2, "F1_L_pitch": 1.5, "T1_L_pitch": -1.2, "Ta1_L_pitch": -0.3},
            LegID.LEFT_MIDDLE: {"C2_L_pitch": -0.1, "F2_L_pitch": 1.6, "T2_L_pitch": -1.3, "Ta2_L_pitch": -0.3},
            LegID.LEFT_HIND: {"C3_L_pitch": 0.0, "F3_L_pitch": 1.7, "T3_L_pitch": -1.4, "Ta3_L_pitch": -0.3},
            LegID.RIGHT_FRONT: {"C1_R_pitch": -0.2, "F1_R_pitch": 1.5, "T1_R_pitch": -1.2, "Ta1_R_pitch": -0.3},
            LegID.RIGHT_MIDDLE: {"C2_R_pitch": -0.1, "F2_R_pitch": 1.6, "T2_R_pitch": -1.3, "Ta2_R_pitch": -0.3},
            LegID.RIGHT_HIND: {"C3_R_pitch": 0.0, "F3_R_pitch": 1.7, "T3_R_pitch": -1.4, "Ta3_R_pitch": -0.3},
        }
        return patterns.get(leg, {})

    @staticmethod
    def tripod_gait(phase: float) -> dict[str, float]:
        """
        Generate tripod gait pattern.

        Tripod gait: L1+R2+L3 in phase, R1+L2+R3 opposite phase.

        Args:
            phase: Gait phase [0, 1]

        Returns:
            Dictionary of joint angle targets
        """
        angles = {}

        # Group 1: L1, R2, L3
        group1_phase = phase
        # Group 2: R1, L2, R3
        group2_phase = (phase + 0.5) % 1.0

        for leg, ph in [(LegID.LEFT_FRONT, group1_phase),
                        (LegID.RIGHT_MIDDLE, group1_phase),
                        (LegID.LEFT_HIND, group1_phase)]:
            if ph < 0.5:
                angles.update(MotorPatterns.stance_phase(leg))
            else:
                angles.update(MotorPatterns.swing_phase(leg))

        for leg, ph in [(LegID.RIGHT_FRONT, group2_phase),
                        (LegID.LEFT_MIDDLE, group2_phase),
                        (LegID.RIGHT_HIND, group2_phase)]:
            if ph < 0.5:
                angles.update(MotorPatterns.stance_phase(leg))
            else:
                angles.update(MotorPatterns.swing_phase(leg))

        return angles

    @staticmethod
    def wingbeat(frequency: float, time: float, amplitude: float = 2.0) -> dict[str, float]:
        """
        Generate wingbeat pattern.

        Args:
            frequency: Wingbeat frequency in Hz (~250 Hz for Drosophila)
            time: Current time in seconds
            amplitude: Wing amplitude in radians

        Returns:
            Dictionary with wing angle targets
        """
        phase = 2 * np.pi * frequency * time
        angle = amplitude * np.sin(phase)
        return {
            "wing_L": angle,
            "wing_R": -angle,  # Opposite phase
        }
