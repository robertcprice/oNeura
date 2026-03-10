"""
Antenna Olfactory System for Drosophila melanogaster

Implements realistic olfactory sensing with:
- 50 odorant receptor types (OrX genes)
- Bilateral antennae for chemotaxis
- Hill function concentration-response encoding
- Temporal adaptation (sensory fatigue)
- Integration with PhysicsEnvironment odorant field

References:
- Hallem, E.A. & Carlson, J.R. (2006) Coding of odors by a receptor repertoire
- Fishilevich, E. et al. (2005) Chemotaxis behavior
- Bhandawat, V. et al. (2007) Sensory neural coding
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray


class OdorantType(Enum):
    """Common Drosophila odorants and their receptor mappings."""
    # Fruit odors (attractive)
    ETHYL_ACETATE = "ethyl_acetate"      # Or42a, Or42b - fermented fruit
    ISOAMYL_ACETATE = "isoamyl_acetate"  # Or42a - banana
    VINEGAR = "vinegar"                  # Or42a, Or85a - acetic acid
    CITRONELLAL = "citronellal"          # Or2a - citrus

    # Yeast odors (attractive)
    PHENETHYL_ALCOHOL = "phenethyl_alcohol"  # Or43a - rose/floral
    METHYL_PHENOL = "methyl_phenol"          # Or46a - yeast

    # Repellent odors
    BENZALDEHYDE = "benzaldehyde"        # Or7a - almond (avoidance)
    LINALOOL = "linalool"                # Or22a - floral

    # Pheromones
    CIS_VACCENYL_ACETATE = "cVA"         # Or67d - aggregation pheromone
    CIS_11_VACCENYL_ACETATE = "11cVA"    # Or65a - male marking

    # CO2 (strong avoidance)
    CO2 = "co2"                          # Gr21a/Gr63a - stress signal


@dataclass
class OdorantReceptor:
    """
    Single odorant receptor neuron (ORN).

    Each receptor expresses a specific OrX gene and responds to
    a subset of odorants with characteristic sensitivity.
    """
    name: str                    # e.g., "Or42a"
    sensitivity: Dict[str, float]  # odorant_name -> EC50 (half-max concentration)
    hill_coefficient: float = 1.5  # Cooperativity
    max_firing_rate: float = 200.0  # Hz
    spontaneous_rate: float = 5.0    # Hz baseline

    # Adaptation state
    _adaptation_level: float = field(default=0.0, repr=False)
    adaptation_tau: float = 5.0      # seconds

    def response(self, concentration: float, odorant: str, dt: float = 0.01) -> float:
        """
        Compute receptor response using Hill function.

        Args:
            concentration: Odorant concentration (ppm or arbitrary units)
            odorant: Name of the odorant
            dt: Time step for adaptation

        Returns:
            Firing rate (Hz)
        """
        # Get EC50 for this odorant (default to insensitive)
        ec50 = self.sensitivity.get(odorant)
        if ec50 is None or ec50 <= 0:
            return self.spontaneous_rate

        # Hill function: R = R_max * (C^n / (EC50^n + C^n))
        n = self.hill_coefficient
        hill_response = (concentration ** n) / (ec50 ** n + concentration ** n)
        response = self.spontaneous_rate + (self.max_firing_rate - self.spontaneous_rate) * hill_response

        # Adaptation: reduce response over time with sustained stimulus
        # Fast adaptation for high concentrations
        adaptation_rate = hill_response * (1 - self._adaptation_level) * dt / self.adaptation_tau
        self._adaptation_level += adaptation_rate
        self._adaptation_level = min(self._adaptation_level, 0.9)  # Max 90% adaptation

        # Apply adaptation
        adapted_response = response * (1 - self._adaptation_level)

        # Recovery from adaptation when concentration drops
        if concentration < ec50 * 0.1:
            self._adaptation_level *= np.exp(-dt / (self.adaptation_tau * 2))

        return adapted_response

    def reset(self):
        """Reset adaptation state."""
        self._adaptation_level = 0.0


class Antenna:
    """
    Drosophila antenna with ~50 odorant receptor types.

    The antenna contains olfactory sensory neurons (OSNs) that project
    to the antennal lobe. Each antenna samples the local odor environment
    and encodes concentration as neural firing rates.
    """

    # Receptor sensitivities (EC50 values) based on Hallem & Carlson 2006
    # Format: (odorant_name, EC50 in arbitrary units)
    RECEPTOR_SENSITIVITIES = {
        "Or42a": {"ethyl_acetate": 0.1, "isoamyl_acetate": 0.2, "vinegar": 0.3},
        "Or42b": {"ethyl_acetate": 0.5},
        "Or43a": {"phenethyl_alcohol": 0.1},
        "Or46a": {"methyl_phenol": 0.2},
        "Or7a": {"benzaldehyde": 0.05},
        "Or22a": {"linalool": 0.1, "ethyl_acetate": 1.0},
        "Or2a": {"citronellal": 0.2},
        "Or85a": {"vinegar": 0.5},
        "Or67d": {"cVA": 0.01},  # Very sensitive to pheromone
        "Or65a": {"11cVA": 0.02},
        "Gr21a": {"co2": 0.1},  # CO2 receptor (actually gustatory)
        "Gr63a": {"co2": 0.15},
    }

    def __init__(self, side: str = "left", num_receptors: int = 50):
        """
        Initialize antenna.

        Args:
            side: "left" or "right"
            num_receptors: Number of receptor types (max ~50 in Drosophila)
        """
        self.side = side
        self.num_receptors = num_receptors

        # Create receptors
        self.receptors: List[OdorantReceptor] = []
        self._create_receptors()

        # State
        self.current_rates: NDArray = np.zeros(num_receptors)
        self.current_concentrations: Dict[str, float] = {}

    def _create_receptors(self):
        """Create odorant receptors based on known sensitivities."""
        self.receptors = []

        for i in range(self.num_receptors):
            # Cycle through known receptors
            known_names = list(self.RECEPTOR_SENSITIVITIES.keys())
            if i < len(known_names):
                name = known_names[i]
                sensitivities = self.RECEPTOR_SENSITIVITIES[name].copy()
            else:
                # Generic receptor with random sensitivities
                name = f"Or{i+1}"
                # Assign random sensitivity to some odorants
                odorants = ["ethyl_acetate", "phenethyl_alcohol", "benzaldehyde"]
                sensitivities = {o: np.random.uniform(0.1, 2.0) for o in odorants}

            receptor = OdorantReceptor(
                name=name,
                sensitivity=sensitivities,
                hill_coefficient=np.random.uniform(1.2, 2.0),
            )
            self.receptors.append(receptor)

    def sample(
        self,
        odorants: Dict[str, float],
        dt: float = 0.01,
    ) -> NDArray:
        """
        Sample odorant environment and compute receptor responses.

        Args:
            odorants: Dictionary mapping odorant_name -> concentration
            dt: Time step for adaptation

        Returns:
            Array of firing rates for all receptors (Hz)
        """
        self.current_concentrations = odorants.copy()
        rates = np.zeros(self.num_receptors)

        for i, receptor in enumerate(self.receptors):
            # Sum responses to all present odorants
            # (simplified - real ORNs have complex integration)
            total_rate = receptor.spontaneous_rate
            for odorant, concentration in odorants.items():
                if concentration > 0:
                    rate = receptor.response(concentration, odorant, dt)
                    total_rate = max(total_rate, rate)  # Take max (winner-take-all)

            rates[i] = total_rate

        self.current_rates = rates
        return rates

    def get_chemotaxis_signal(self) -> Tuple[float, float]:
        """
        Compute chemotaxis steering signal based on receptor activation.

        Returns:
            (turn_signal, approach_signal)
            - turn_signal: positive = turn right (more odor on right antenna)
            - approach_signal: positive = attractive odor, negative = repellent
        """
        # Categorize receptors as attractive or repellent
        attractive_keywords = ["ethyl_acetate", "isoamyl", "vinegar", "phenethyl", "methyl_phenol", "citronellal"]
        repellent_keywords = ["benzaldehyde", "linalool", "co2"]

        attractive_rate = 0.0
        repellent_rate = 0.0

        for i, receptor in enumerate(self.receptors):
            rate = self.current_rates[i]
            for odorant in receptor.sensitivity.keys():
                if any(kw in odorant.lower() for kw in attractive_keywords):
                    attractive_rate = max(attractive_rate, rate)
                elif any(kw in odorant.lower() for kw in repellent_keywords):
                    repellent_rate = max(repellent_rate, rate)

        # Net approach signal (attractive - repellent)
        approach_signal = (attractive_rate - repellent_rate) / 200.0  # Normalize
        approach_signal = np.clip(approach_signal, -1, 1)

        return 0.0, approach_signal  # Turn signal computed bilaterally

    def reset(self):
        """Reset all receptors."""
        for receptor in self.receptors:
            receptor.reset()
        self.current_rates = np.zeros(self.num_receptors)
        self.current_concentrations = {}


class BilateralOlfaction:
    """
    Both antennae for bilateral chemotaxis.

    Drosophila compare left vs right antenna signals to navigate
    up odor gradients (tropotaxis).
    """

    def __init__(self, num_receptors: int = 50):
        """Initialize both antennae."""
        self.left_antenna = Antenna(side="left", num_receptors=num_receptors)
        self.right_antenna = Antenna(side="right", num_receptors=num_receptors)

        # Antenna spacing (for gradient estimation)
        self.antenna_spacing = 0.0005  # 0.5mm in meters

    def sample(
        self,
        odorants: Dict[str, float],
        odorant_gradient: Optional[Dict[str, NDArray]] = None,
        dt: float = 0.01,
    ) -> Dict[str, NDArray]:
        """
        Sample odorants with both antennae.

        Args:
            odorants: Base concentration at body center
            odorant_gradient: Dict mapping odorant -> gradient vector [dx, dy, dz]
            dt: Time step

        Returns:
            Dictionary with left/right rates and chemotaxis signals
        """
        # Compute left/right concentrations from gradient
        left_odorants = odorants.copy()
        right_odorants = odorants.copy()

        if odorant_gradient:
            for odorant, gradient in odorant_gradient.items():
                # Concentration difference between antennae
                # gradient points UP the gradient
                # left antenna is -y direction, right is +y
                delta = gradient[1] * self.antenna_spacing  # y-component
                left_odorants[odorant] = odorants.get(odorant, 0) - delta
                right_odorants[odorant] = odorants.get(odorant, 0) + delta

        # Sample both antennae
        left_rates = self.left_antenna.sample(left_odorants, dt)
        right_rates = self.right_antenna.sample(right_odorants, dt)

        # Compute chemotaxis
        left_turn, left_approach = self.left_antenna.get_chemotaxis_signal()
        right_turn, right_approach = self.right_antenna.get_chemotaxis_signal()

        # Bilateral comparison for turning
        # More odor on right -> turn right (positive)
        left_mean = np.mean(left_rates)
        right_mean = np.mean(right_rates)
        turn_signal = (right_mean - left_mean) / (left_mean + right_mean + 1e-6)
        turn_signal = np.clip(turn_signal, -1, 1)

        # Approach signal (average of both)
        approach_signal = (left_approach + right_approach) / 2

        return {
            'left_rates': left_rates,
            'right_rates': right_rates,
            'turn_signal': turn_signal,
            'approach_signal': approach_signal,
            'combined_rates': np.concatenate([left_rates, right_rates]),
        }

    def get_olfaction_dim(self) -> int:
        """Return total olfactory dimension (both antennae)."""
        return self.left_antenna.num_receptors * 2

    def reset(self):
        """Reset both antennae."""
        self.left_antenna.reset()
        self.right_antenna.reset()


# Integration helper
def create_olfactory_encoder_output(
    olfaction: BilateralOlfaction,
    odorants: Dict[str, float],
    target_dim: int = 100,
    dt: float = 0.01,
) -> NDArray:
    """
    Sample olfaction and compress to target dimension.

    Args:
        olfaction: BilateralOlfaction system
        odorants: Current odorant concentrations
        target_dim: Target output dimension
        dt: Time step

    Returns:
        Compressed olfactory feature vector
    """
    sample = olfaction.sample(odorants, dt=dt)
    combined = sample['combined_rates']

    # Normalize firing rates to [0, 1]
    combined = combined / 200.0  # Max firing rate
    combined = np.clip(combined, 0, 1)

    # Compress or pad to target dimension
    if len(combined) > target_dim:
        chunk_size = len(combined) // target_dim
        output = np.array([
            combined[i*chunk_size:(i+1)*chunk_size].mean()
            for i in range(target_dim)
        ])
    else:
        output = np.zeros(target_dim)
        output[:len(combined)] = combined

    # Append turn and approach signals
    if target_dim >= 2:
        output[-2] = sample['turn_signal']
        output[-1] = sample['approach_signal']

    return output
