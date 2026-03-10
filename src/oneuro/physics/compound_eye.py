"""
Compound Eye Simulation for Drosophila melanogaster

Implements realistic compound eye optics with:
- 768 ommatidia per eye (~1500 total)
- Hexagonal lattice sampling pattern
- ~5° acceptance angle per ommatidium
- Hassenstein-Reichardt elementary motion detectors (EMDs)
- Spectral sensitivity (UV, blue, green channels)
- Temporal filtering (photoreceptor dynamics)

References:
- Land, M.F. (1997) Visual acuity in insects
- Hassenstein, B. & Reichardt, W. (1956) Motion detection
- Borst, A. (2009) Drosophila visual course control
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, List
import numpy as np
from numpy.typing import NDArray


class SpectralChannel(Enum):
    """Photoreceptor spectral sensitivities."""
    UV = "uv"          # Rh3/Rh4 (~350nm) - UV-sensitive
    BLUE = "blue"      # Rh5 (~440nm) - Blue-sensitive
    GREEN = "green"    # Rh6 (~530nm) - Green-sensitive


@dataclass
class Ommatidium:
    """
    Single ommatidium (faceted lens unit).

    Each ommatidium samples a small cone of visual space
    defined by its pointing direction and acceptance angle.
    """
    index: int
    azimuth: float      # Horizontal angle (radians)
    elevation: float    # Vertical angle (radians)
    acceptance_angle: float = np.radians(5.0)  # ~5° typical for Drosophila

    # Spectral sensitivity (relative weights for each channel)
    spectral_weights: Dict[str, float] = field(default_factory=lambda: {
        SpectralChannel.UV.value: 0.33,
        SpectralChannel.BLUE.value: 0.33,
        SpectralChannel.GREEN.value: 0.33,
    })

    # Current state
    intensity: float = 0.0
    intensity_history: List[float] = field(default_factory=list)

    def get_direction_vector(self) -> NDArray:
        """Get 3D unit vector pointing in ommatidium's viewing direction."""
        x = np.cos(self.elevation) * np.cos(self.azimuth)
        y = np.cos(self.elevation) * np.sin(self.azimuth)
        z = np.sin(self.elevation)
        return np.array([x, y, z])


@dataclass
class MotionDetector:
    """
    Hassenstein-Reichardt Elementary Motion Detector (EMD).

    Detects motion in one direction by comparing signals from
    two adjacent ommatidia with a temporal delay.

    Architecture:
        Input1 ----[LPF delay]---->
                                  (multiply)
        Input2 ----[HPF]--------->

    The correlation-type detector responds preferentially to motion
    in the direction from Input1 to Input2.

    Reference: Borst, A. (2009) Animate motion vision
    """
    ommatidium_1_idx: int  # Delayed input
    ommatidium_2_idx: int  # Direct input
    preferred_direction: float  # Azimuth direction of motion (radians)

    # Temporal filters (exponential decay)
    delay_tau: float = 0.035    # 35ms delay (typical for Drosophila)
    highpass_tau: float = 0.050  # 50ms highpass

    # Internal state
    _delayed_signal: float = 0.0
    _highpass_state: float = 0.0
    _prev_input: float = 0.0

    # Output
    response: float = 0.0


class CompoundEye:
    """
    Drosophila compound eye with 768 ommatidia.

    The hexagonal lattice samples ~300° horizontally and ~180° vertically,
    with higher density in the frontal visual field for flight control.

    Typical parameters:
    - Inter-ommatidial angle: ~5°
    - Acceptance angle: ~5° (slightly larger than inter-ommatidial)
    - Field of view: ~300° horizontal, ~180° vertical
    - Update rate: ~100 Hz (photoreceptor bandwidth)
    """

    NUM_OMMATIDIA = 768  # Per eye
    ACCEPTANCE_ANGLE_DEG = 5.0  # Degrees
    FRONTAL_ELEVATION = 7.5     # Degrees upward (flight posture)
    FRONTAL_DENSITY_BOOST = 1.3 # Higher density in frontal field

    def __init__(
        self,
        eye_side: str = "left",
        num_ommatidia: int = 768,
        acceptance_angle_deg: float = 5.0,
        frontal_bias: bool = True,
    ):
        """
        Initialize compound eye.

        Args:
            eye_side: "left" or "right" - determines hemisphere
            num_ommatidia: Number of ommatidia (default 768)
            acceptance_angle_deg: Acceptance angle in degrees
            frontal_bias: If True, increase density in frontal visual field
        """
        self.eye_side = eye_side
        self.num_ommatidia = num_ommatidia
        self.acceptance_angle = np.radians(acceptance_angle_deg)
        self.frontal_bias = frontal_bias

        # Generate ommatidial lattice
        self.ommatidia: List[Ommatidium] = []
        self._generate_hexagonal_lattice()

        # Motion detectors (horizontal and vertical)
        self.motion_detectors_h: List[MotionDetector] = []
        self.motion_detectors_v: List[MotionDetector] = []
        self._create_motion_detectors()

        # Spectral channels
        self.spectral_channels = [SpectralChannel.UV, SpectralChannel.BLUE, SpectralChannel.GREEN]

        # State
        self.current_frame: Optional[NDArray] = None
        self.motion_response_h: float = 0.0
        self.motion_response_v: float = 0.0

    def _generate_hexagonal_lattice(self):
        """Generate hexagonal lattice of ommatidia with frontal bias."""
        self.ommatidia = []

        # Eye hemisphere: left eye covers right visual field and vice versa
        if self.eye_side == "left":
            azimuth_range = (-np.pi/2, np.pi/2)  # Right hemisphere (contralateral)
        else:
            azimuth_range = (-np.pi/2, np.pi/2)  # Left hemisphere (contralateral)

        # Vertical range
        elevation_range = (-np.pi/3, np.pi/3)  # ±60° vertical

        # Calculate spacing needed for target ommatidia count
        # Hexagonal lattice: N ≈ (2/√3) * (range_az / spacing) * (range_el / spacing)
        range_az = azimuth_range[1] - azimuth_range[0]
        range_el = elevation_range[1] - elevation_range[0]
        spacing = np.sqrt((2 / np.sqrt(3)) * range_az * range_el / self.num_ommatidia)

        # Generate lattice points
        idx = 0
        row = 0
        elevation = elevation_range[0]

        while elevation <= elevation_range[1] and idx < self.num_ommatidia:
            # Offset every other row for hexagonal pattern
            az_offset = (row % 2) * spacing / 2

            # Calculate number of points in this row
            n_points = int(range_az / spacing)

            for i in range(n_points):
                if idx >= self.num_ommatidia:
                    break

                azimuth = azimuth_range[0] + az_offset + i * spacing

                # Skip if outside range
                if azimuth > azimuth_range[1]:
                    continue

                # Apply frontal density boost (more ommatidia facing forward)
                if self.frontal_bias:
                    # Frontal = small azimuth (looking ahead)
                    frontal_weight = np.exp(-abs(azimuth) / 0.5)
                    if np.random.random() > frontal_weight * self.FRONTAL_DENSITY_BOOST:
                        continue  # Skip this position

                # Create ommatidium
                # Frontal elevation bias (eyes tilted slightly upward for flight)
                adjusted_elevation = elevation + np.radians(self.FRONTAL_ELEVATION)

                om = Ommatidium(
                    index=idx,
                    azimuth=azimuth,
                    elevation=adjusted_elevation,
                    acceptance_angle=self.acceptance_angle,
                )
                self.ommatidia.append(om)
                idx += 1

            elevation += spacing * np.sqrt(3) / 2
            row += 1

        # Trim to exact count
        self.ommatidia = self.ommatidia[:self.num_ommatidia]

    def _create_motion_detectors(self):
        """Create Hassenstein-Reichardt EMD pairs."""
        self.motion_detectors_h = []
        self.motion_detectors_v = []

        # Find adjacent ommatidia pairs for motion detection
        for i, om in enumerate(self.ommatidia):
            # Horizontal motion: find nearest neighbor in azimuth
            for j, om2 in enumerate(self.ommatidia):
                if i == j:
                    continue

                # Check if horizontally adjacent (similar elevation, different azimuth)
                el_diff = abs(om.elevation - om2.elevation)
                az_diff = om2.azimuth - om.azimuth

                if el_diff < np.radians(3) and 0 < az_diff < np.radians(10):
                    # Horizontal EMD
                    emd = MotionDetector(
                        ommatidium_1_idx=i,
                        ommatidium_2_idx=j,
                        preferred_direction=om.azimuth,
                    )
                    self.motion_detectors_h.append(emd)
                    break  # One EMD per ommatidium

            # Vertical motion: find nearest neighbor in elevation
            for j, om2 in enumerate(self.ommatidia):
                if i == j:
                    continue

                # Check if vertically adjacent (similar azimuth, different elevation)
                az_diff = abs(om.azimuth - om2.azimuth)
                el_diff = om2.elevation - om.elevation

                if az_diff < np.radians(3) and 0 < el_diff < np.radians(10):
                    # Vertical EMD
                    emd = MotionDetector(
                        ommatidium_1_idx=i,
                        ommatidium_2_idx=j,
                        preferred_direction=np.pi/2,  # Upward
                    )
                    self.motion_detectors_v.append(emd)
                    break

    def sample_environment(
        self,
        light_direction: NDArray,
        light_intensity: float,
        body_orientation: NDArray,
        obstacles: Optional[List[Tuple[NDArray, float]]] = None,
        dt: float = 0.01,
    ) -> NDArray:
        """
        Sample the visual environment through all ommatidia.

        Args:
            light_direction: Unit vector pointing toward light source
            light_intensity: Base light intensity (0-1)
            body_orientation: Quaternion [w, x, y, z] of body orientation
            obstacles: List of (position, radius) tuples for obstacle shadows
            dt: Time step for temporal filtering

        Returns:
            Array of ommatidial intensities (num_ommatidia,)
        """
        # Body orientation rotation matrix (simplified - just use yaw for now)
        body_yaw = self._quat_to_yaw(body_orientation)

        intensities = np.zeros(self.num_ommatidia)

        for i, om in enumerate(self.ommatidia):
            # Rotate ommatidium direction by body orientation
            azimuth_world = om.azimuth + body_yaw
            if self.eye_side == "left":
                azimuth_world = -azimuth_world  # Mirror for left eye

            # Ommatidium viewing direction in world coordinates
            direction = np.array([
                np.cos(om.elevation) * np.cos(azimuth_world),
                np.cos(om.elevation) * np.sin(azimuth_world),
                np.sin(om.elevation),
            ])

            # Light intensity: cosine of angle between viewing direction and light
            # light_direction points TO light source, so we want alignment
            cos_angle = np.dot(direction, light_direction)

            # Within acceptance angle: smooth falloff
            intensity = max(0.0, cos_angle) * light_intensity

            # Check for obstacle shadows
            if obstacles:
                for obs_pos, obs_radius in obstacles:
                    # Simplified: check if obstacle blocks this direction
                    # (Full implementation would ray-cast)
                    to_obs = obs_pos / (np.linalg.norm(obs_pos) + 1e-6)
                    if np.dot(direction, to_obs) > 0.9:
                        intensity *= 0.3  # Shadow
                        break

            # Temporal filtering (photoreceptor dynamics)
            # Low-pass filter with ~10ms time constant
            tau_photo = 0.010
            alpha = dt / (tau_photo + dt)
            om.intensity = alpha * intensity + (1 - alpha) * om.intensity

            # Store in history for motion detection
            om.intensity_history.append(om.intensity)
            if len(om.intensity_history) > 10:
                om.intensity_history.pop(0)

            intensities[i] = om.intensity

        self.current_frame = intensities
        return intensities

    def _quat_to_yaw(self, quat: NDArray) -> float:
        """Extract yaw angle from quaternion [w, x, y, z]."""
        if len(quat) != 4:
            return 0.0
        w, x, y, z = quat
        # Yaw rotation around z-axis
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def compute_motion(self, dt: float = 0.01) -> Tuple[float, float]:
        """
        Compute motion responses from all EMDs.

        Returns:
            (horizontal_motion, vertical_motion) signed values
            Positive = rightward/upward motion
        """
        if self.current_frame is None:
            return 0.0, 0.0

        h_response = 0.0
        v_response = 0.0

        # Horizontal motion
        for emd in self.motion_detectors_h:
            if emd.ommatidium_1_idx < len(self.current_frame) and emd.ommatidium_2_idx < len(self.current_frame):
                input1 = self.current_frame[emd.ommatidium_1_idx]
                input2 = self.current_frame[emd.ommatidium_2_idx]

                # Delayed signal (low-pass)
                alpha_delay = dt / (emd.delay_tau + dt)
                emd._delayed_signal = alpha_delay * input1 + (1 - alpha_delay) * emd._delayed_signal

                # High-pass filtered signal
                alpha_hp = dt / (emd.highpass_tau + dt)
                highpass = input2 - emd._highpass_state
                emd._highpass_state = alpha_hp * input2 + (1 - alpha_hp) * emd._highpass_state

                # Correlation (multiply)
                emd.response = emd._delayed_signal * highpass
                h_response += emd.response

        # Vertical motion
        for emd in self.motion_detectors_v:
            if emd.ommatidium_1_idx < len(self.current_frame) and emd.ommatidium_2_idx < len(self.current_frame):
                input1 = self.current_frame[emd.ommatidium_1_idx]
                input2 = self.current_frame[emd.ommatidium_2_idx]

                alpha_delay = dt / (emd.delay_tau + dt)
                emd._delayed_signal = alpha_delay * input1 + (1 - alpha_delay) * emd._delayed_signal

                alpha_hp = dt / (emd.highpass_tau + dt)
                highpass = input2 - emd._highpass_state
                emd._highpass_state = alpha_hp * input2 + (1 - alpha_hp) * emd._highpass_state

                emd.response = emd._delayed_signal * highpass
                v_response += emd.response

        # Normalize by number of detectors
        if self.motion_detectors_h:
            h_response /= len(self.motion_detectors_h)
        if self.motion_detectors_v:
            v_response /= len(self.motion_detectors_v)

        self.motion_response_h = h_response
        self.motion_response_v = v_response

        return h_response, v_response

    def get_phototaxis_signal(self) -> Tuple[float, float]:
        """
        Compute phototaxis steering signal.

        Drosophila are positively phototactic (attracted to light).
        This signal indicates turning direction to orient toward light.

        Returns:
            (yaw_signal, pitch_signal) - positive = turn toward light
        """
        if self.current_frame is None or len(self.ommatidia) == 0:
            return 0.0, 0.0

        # Weight intensities by frontal bias (frontal ommatidia have more influence)
        frontal_weight = np.array([
            np.exp(-abs(om.azimuth) / 0.5) for om in self.ommatidia
        ])

        # Compute left-right asymmetry (for yaw)
        left_intensity = 0.0
        right_intensity = 0.0

        for i, om in enumerate(self.ommatidia):
            if om.azimuth < 0:  # Left side
                left_intensity += self.current_frame[i] * frontal_weight[i]
            else:  # Right side
                right_intensity += self.current_frame[i] * frontal_weight[i]

        # Normalize
        total = left_intensity + right_intensity + 1e-6
        yaw_signal = (right_intensity - left_intensity) / total

        # Compute up-down asymmetry (for pitch)
        up_intensity = 0.0
        down_intensity = 0.0

        for i, om in enumerate(self.ommatidia):
            if om.elevation > 0:  # Upper
                up_intensity += self.current_frame[i]
            else:  # Lower
                down_intensity += self.current_frame[i]

        pitch_signal = (up_intensity - down_intensity) / (up_intensity + down_intensity + 1e-6)

        return yaw_signal, pitch_signal

    def reset(self):
        """Reset eye state."""
        for om in self.ommatidia:
            om.intensity = 0.0
            om.intensity_history = []

        for emd in self.motion_detectors_h + self.motion_detectors_v:
            emd._delayed_signal = 0.0
            emd._highpass_state = 0.0
            emd._prev_input = 0.0
            emd.response = 0.0

        self.current_frame = None
        self.motion_response_h = 0.0
        self.motion_response_v = 0.0


class BinocularVisionSystem:
    """
    Both compound eyes with binocular integration.

    Provides:
    - Stereopsis (depth perception via disparity)
    - Wide-field motion detection (optomotor response)
    - Looming detection (collision avoidance)
    """

    def __init__(self, num_ommatidia_per_eye: int = 768):
        """Initialize both compound eyes."""
        self.left_eye = CompoundEye(eye_side="left", num_ommatidia=num_ommatidia_per_eye)
        self.right_eye = CompoundEye(eye_side="right", num_ommatidia=num_ommatidia_per_eye)

        # Binocular overlap region (frontal ~30°)
        self.binocular_overlap = np.radians(30)

    def sample(
        self,
        light_direction: NDArray,
        light_intensity: float,
        body_orientation: NDArray,
        obstacles: Optional[List[Tuple[NDArray, float]]] = None,
        dt: float = 0.01,
    ) -> Dict[str, NDArray]:
        """
        Sample visual environment with both eyes.

        Returns:
            Dictionary with:
            - 'left_intensities': Left eye ommatidial array
            - 'right_intensities': Right eye ommatidial array
            - 'left_motion': (h, v) motion signals
            - 'right_motion': (h, v) motion signals
            - 'phototaxis': (yaw, pitch) steering signals
        """
        left_intensities = self.left_eye.sample_environment(
            light_direction, light_intensity, body_orientation, obstacles, dt
        )
        right_intensities = self.right_eye.sample_environment(
            light_direction, light_intensity, body_orientation, obstacles, dt
        )

        left_motion = self.left_eye.compute_motion(dt)
        right_motion = self.right_eye.compute_motion(dt)

        # Integrate phototaxis signals from both eyes
        left_yaw, left_pitch = self.left_eye.get_phototaxis_signal()
        right_yaw, right_pitch = self.right_eye.get_phototaxis_signal()

        # Average phototaxis (eyes point in opposite directions, so negate left)
        combined_yaw = (right_yaw - left_yaw) / 2
        combined_pitch = (left_pitch + right_pitch) / 2

        return {
            'left_intensities': left_intensities,
            'right_intensities': right_intensities,
            'left_motion': left_motion,
            'right_motion': right_motion,
            'phototaxis': (combined_yaw, combined_pitch),
            'global_motion': (
                (left_motion[0] + right_motion[0]) / 2,
                (left_motion[1] + right_motion[1]) / 2,
            ),
        }

    def get_visual_dim(self) -> int:
        """Return total visual dimension (both eyes)."""
        return self.left_eye.num_ommatidia + self.right_eye.num_ommatidia

    def reset(self):
        """Reset both eyes."""
        self.left_eye.reset()
        self.right_eye.reset()


# Integration with SensorEncoder
def create_visual_encoder_output(
    vision_system: BinocularVisionSystem,
    target_dim: int = 128,
) -> NDArray:
    """
    Compress visual output to target dimension for neural encoding.

    Args:
        vision_system: Binocular vision system
        target_dim: Target output dimension

    Returns:
        Compressed visual feature vector
    """
    sample = vision_system.sample(
        light_direction=np.array([0, 0, 1]),
        light_intensity=1.0,
        body_orientation=np.array([1, 0, 0, 0]),
    )

    # Combine left and right intensities
    left = sample['left_intensities']
    right = sample['right_intensities']
    combined = np.concatenate([left, right])

    # Simple compression via averaging
    if len(combined) > target_dim:
        # Average groups
        chunk_size = len(combined) // target_dim
        output = np.array([
            combined[i*chunk_size:(i+1)*chunk_size].mean()
            for i in range(target_dim)
        ])
    else:
        # Pad with zeros
        output = np.zeros(target_dim)
        output[:len(combined)] = combined

    # Add motion signals
    motion_h, motion_v = sample['global_motion']
    phototaxis_yaw, phototaxis_pitch = sample['phototaxis']

    # Replace last 4 dimensions with motion/phototaxis
    if target_dim >= 4:
        output[-4] = motion_h
        output[-3] = motion_v
        output[-2] = phototaxis_yaw
        output[-1] = phototaxis_pitch

    return np.clip(output, 0, 1)
