"""
oNeura Robot Module - Real-world neural navigation interfaces.

This module provides:
- DroneInterface: MAVLink-based drone control
- SensorFusion: GPS/IMU sensor fusion
- OutdoorNavigationExperiment: Outdoor validation experiments

Based on BoB (Brains on Board) paper methodology.
"""

from oneuro.robot.drone_interface import (
    DroneInterface,
    SensorFusion,
    OutdoorNavigationExperiment
)

__all__ = [
    "DroneInterface",
    "SensorFusion",
    "OutdoorNavigationExperiment",
]
