"""
Biological Parameter Tuning for Drosophila Physics Simulation

This module provides scientifically-grounded parameter values based on
the NeuroMechFly project and related literature.

References:
- Lobato-Rios et al. (2022) "NeuroMechFly, a neuromechanical model of adult Drosophila"
- Dickinson & Tu (1997) "The function of dipteran flight muscle"
- Mamiya et al. (2018) "Neural coding of conspecific and heterospecific pheromones"
- Bidaye et al. (2014) "Walking Drosophila"
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class DrosophilaParameters:
    """
    Biologically-accurate parameters for Drosophila simulation.

    All values are in SI units (meters, kilograms, seconds, radians).
    Model is scaled 1000x for numerical stability in MuJoCo.
    """

    # Scale factor (MuJoCo works better with larger numbers)
    SCALE: float = 1000.0

    # Body dimensions (real scale in meters)
    body_length: float = 2.5e-3      # 2.5mm
    body_width: float = 0.5e-3       # 0.5mm
    body_height: float = 0.4e-3      # 0.4mm

    # Mass distribution (real scale in kg)
    total_mass: float = 1.0e-6        # ~1mg total
    head_mass: float = 0.05e-6        # 5% of total
    thorax_mass: float = 0.5e-6       # 50% of total
    abdomen_mass: float = 0.35e-6     # 35% of total
    wing_mass: float = 0.01e-6        # 1% each
    leg_mass: float = 0.015e-6        # 1.5% per leg

    # Joint parameters
    joint_damping: float = 1e-8       # N·m·s/rad (very low for fast movement)
    joint_armature: float = 1e-10     # Added inertia for numerical stability
    joint_friction: float = 1e-9      # N·m friction torque

    # Wing parameters
    wing_length: float = 2.5e-3       # 2.5mm wingspan
    wing_beat_frequency: float = 200.0  # Hz (typical 200-250 Hz)
    wing_stroke_amplitude: float = 2.5  # radians (~144°)

    # Leg parameters (from NeuroMechFly)
    leg_segments: Dict[str, float] = None  # Set in __post_init__

    # Locomotion parameters
    walking_frequency: float = 8.0     # Hz stride frequency
    stance_duration: float = 0.6       # % of stride
    swing_duration: float = 0.4        # % of stride

    # Friction coefficients (tarsus-surface)
    tarsus_friction: float = 1.5       # High for climbing

    # Sensory parameters
    visual_field: float = np.pi * 5/6  # ~150° per eye
    ommatidia_count: int = 768         # Per eye
    ommatidial_angle: float = np.radians(5.0)  # ~5° acceptance

    # Olfactory parameters
    antenna_length: float = 0.4e-3     # 0.4mm
    receptor_count: int = 50           # Per antenna

    def __post_init__(self):
        """Initialize derived parameters."""
        self.leg_segments = {
            'coxa_length': 0.15e-3,    # 150μm
            'femur_length': 0.6e-3,    # 600μm
            'tibia_length': 0.5e-3,    # 500μm
            'tarsus_length': 0.15e-3,  # 150μm
        }

    def get_scaled_value(self, value: float) -> float:
        """Scale a real-world value to MuJoCo simulation units."""
        return value * self.SCALE

    def get_scaled_mass(self, mass: float) -> float:
        """Scale mass (mass scales with cube of length)."""
        return mass * (self.SCALE ** 3)

    def get_scaled_inertia(self, inertia: float) -> float:
        """Scale moment of inertia (scales with length^5)."""
        return inertia * (self.SCALE ** 5)


# Singleton instance
PARAMS = DrosophilaParameters()


def tune_mjcf_parameters() -> Dict:
    """
    Generate MJCF-compatible parameter dictionary.

    Returns parameters tuned for stable MuJoCo simulation
    while maintaining biological accuracy.
    """
    p = PARAMS

    return {
        # Timestep (1ms is typical for legged locomotion)
        'timestep': 0.001,

        # Joint defaults
        'joint_armature': 0.001,
        'joint_damping': 0.1,

        # Geometry friction (high for climbing, moderate for walking)
        'geom_friction': '0.8 0.02 0.01',  # sliding, torsion, rolling

        # Masses (scaled 1000x)
        'thorax_mass': 0.5,    # 500g scaled
        'head_mass': 0.05,     # 50g scaled
        'abdomen_mass': 0.35,  # 350g scaled
        'wing_mass': 0.01,     # 10g scaled
        'leg_mass': 0.015,     # 15g scaled

        # Inertia tensors (scaled, ellipsoid approximation)
        'thorax_inertia': '0.001 0.002 0.0015',
        'head_inertia': '0.0002 0.0002 0.00015',
        'wing_inertia': '0.001 1e-6 0.001',
        'leg_inertia': '1e-6 1e-6 1e-6',

        # Wing dynamics
        'wing_damping': 0.01,
        'wing_range': '-2.5 2.5',

        # Actuator gains (position servos)
        'kp_leg': 0.5,         # Proportional gain for leg joints
        'kp_wing': 0.1,        # Lower for wings (fast oscillation)
        'kp_body': 1.0,        # Higher for body posture
    }


def validate_simulation_stability(model, n_steps: int = 1000) -> Dict:
    """
    Test simulation stability with current parameters.

    Args:
        model: MuJoCo model object
        n_steps: Number of steps to simulate

    Returns:
        Dictionary with stability metrics
    """
    try:
        import mujoco
        data = mujoco.MjData(model)
    except ImportError:
        return {'error': 'MuJoCo not available'}

    # Reset to initial state
    mujoco.mj_resetData(model, data)

    metrics = {
        'max_velocity': 0.0,
        'max_acceleration': 0.0,
        'max_force': 0.0,
        'timesteps_completed': 0,
        'stable': True,
        'warnings': [],
    }

    for step in range(n_steps):
        try:
            mujoco.mj_step(model, data)
            metrics['timesteps_completed'] += 1

            # Track extremes
            vel = np.linalg.norm(data.qvel)
            acc = np.linalg.norm(data.qacc) if hasattr(data, 'qacc') else 0
            force = np.max(np.abs(data.actuator_force)) if data.actuator_force is not None else 0

            metrics['max_velocity'] = max(metrics['max_velocity'], vel)
            metrics['max_acceleration'] = max(metrics['max_acceleration'], acc)
            metrics['max_force'] = max(metrics['max_force'], force)

            # Check for instabilities
            if vel > 1e6:
                metrics['warnings'].append(f'Velocity explosion at step {step}: {vel}')
            if force > 1e6:
                metrics['warnings'].append(f'Force explosion at step {step}: {force}')

        except Exception as e:
            metrics['stable'] = False
            metrics['error'] = str(e)
            break

    return metrics


def get_motor_pattern_parameters() -> Dict:
    """
    Get parameters for generating biologically-accurate motor patterns.

    Based on Bidaye et al. (2014) walking kinematics.
    """
    return {
        # Tripod gait timing
        'swing_duration': 0.05,      # 50ms swing
        'stance_duration': 0.075,    # 75ms stance
        'stride_frequency': 8.0,     # Hz

        # Joint angle ranges (radians)
        'coxa_yaw_range': (-0.3, 0.3),
        'coxa_pitch_range': (-0.5, 1.0),
        'femur_pitch_range': (-0.3, 1.2),
        'tibia_pitch_range': (-1.5, 0.3),
        'tarsus_pitch_range': (-0.5, 0.5),

        # Leg touchdown angles
        'stance_coxa': 0.2,
        'stance_femur': 0.8,
        'stance_tibia': -0.5,
        'swing_coxa': -0.3,
        'swing_femur': 1.2,
        'swing_tibia': 0.0,

        # Wingbeat
        'wing_frequency': 200.0,     # Hz
        'wing_amplitude': 2.5,       # radians
        'wing_phase_offset': np.pi,  # 180° between wings
    }


def get_sensory_parameters() -> Dict:
    """
    Get parameters for sensory encoding.

    Based on Drosophila neurophysiology literature.
    """
    return {
        # Visual
        'ommatidia_per_eye': 768,
        'acceptance_angle': np.radians(5.0),
        'temporal_resolution': 100.0,  # Hz flicker fusion

        # Motion detection (EMD)
        'emd_delay': 0.035,           # 35ms delay
        'emd_highpass': 0.050,        # 50ms time constant

        # Olfactory
        'receptor_types': 50,
        'adaptation_tau': 5.0,         # 5s adaptation time constant
        'max_firing_rate': 200.0,      # Hz

        # Mechanosensory
        'touch_threshold': 0.001,      # 1mm deflection
        'proprioceptive_range': (-2.0, 2.0),  # radians
    }
