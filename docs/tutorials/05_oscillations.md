# Tutorial 5: Neural Oscillations and Rhythms

## Overview

Neural oscillations - rhythmic patterns of electrical activity in the brain - are fundamental to cognition. From the gamma rhythms of focused attention to the slow waves of deep sleep, oscillations coordinate neural activity across scales.

This tutorial covers:
- Types of neural oscillations (delta, theta, alpha, beta, gamma)
- Generating oscillations in neural tissue
- Cross-frequency coupling
- Phase-amplitude relationships
- Computing EEG-like signals from tissue activity

## Neuroscience Background

### Brain Rhythm Bands

| Band | Frequency | Associated State | Function |
|------|-----------|------------------|----------|
| **Delta** | 0.5-4 Hz | Deep sleep | Memory consolidation, healing |
| **Theta** | 4-8 Hz | Drowsy, REM | Memory encoding, navigation |
| **Alpha** | 8-13 Hz | Relaxed, eyes closed | Inhibition, idling |
| **Beta** | 13-30 Hz | Active thinking | Focus, cognition |
| **Gamma** | 30-100 Hz | Perception | Binding, consciousness |

### Origin of Oscillations

Oscillations arise from:
1. **Single neuron oscillations**: Intrinsic membrane potential oscillations
2. **Network oscillations**: Feedback loops between excitatory and inhibitory populations
3. **PAC (Phase-Amplitude Coupling)**: Slower rhythms modulating faster ones

### Why Oscillations Matter

- **Temporal Coordination**: Neurons firing in phase are more likely to communicate
- **Phase Coding**: Information encoded in spike timing relative to oscillation phase
- **Communication Through Coherence**: Brain regions synchronize to share information

## Simulating Oscillations

### Natural Emergence

Oscillations can emerge naturally from network dynamics:

```python
from organic_neural_network import OrganicNeuralNetwork
import numpy as np

# Create tissue
tissue = OrganicNeuralNetwork(
    size=(10.0, 10.0, 5.0),
    initial_neurons=50,
    energy_supply=2.0
)

# Record activity over time
activity_trace = []

for step in range(1000):
    tissue.step(dt=0.5)

    # Measure global activity
    active_count = sum(1 for n in tissue.neurons.values()
                       if n.alive and n.state.name == 'ACTIVE')
    activity_trace.append(active_count)

# Analyze for oscillations
activity_trace = np.array(activity_trace)
```

### Forcing Oscillations with Periodic Input

```python
def apply_oscillatory_stimulation(tissue, frequency, amplitude, duration_ms):
    """
    Apply oscillatory stimulation to generate entrained rhythms.

    Args:
        tissue: Neural tissue
        frequency: Oscillation frequency in Hz
        amplitude: Stimulation strength
        duration_ms: Total duration in milliseconds
    """
    period_ms = 1000.0 / frequency
    dt = 0.5  # Time step

    activity_trace = []
    time_trace = []
    time = 0.0

    while time < duration_ms:
        # Calculate stimulation phase
        phase = 2 * np.pi * frequency * (time / 1000.0)
        current_intensity = amplitude * max(0, np.sin(phase))

        # Apply stimulation
        tissue.stimulate(
            position=(5.0, 5.0, 2.5),
            intensity=current_intensity,
            radius=3.0
        )

        # Step simulation
        tissue.step(dt)

        # Record
        active = sum(1 for n in tissue.neurons.values()
                     if n.alive and n.state.name == 'ACTIVE')
        activity_trace.append(active)
        time_trace.append(time)

        time += dt

    return np.array(time_trace), np.array(activity_trace)

# Generate 10 Hz alpha rhythm
time, activity = apply_oscillatory_stimulation(tissue, frequency=10.0,
                                                amplitude=15.0, duration_ms=500.0)
```

## Analyzing Oscillations

### Power Spectrum Analysis

```python
from scipy import signal
import numpy as np

def compute_power_spectrum(activity_trace, dt=0.5):
    """
    Compute power spectral density of activity trace.

    Args:
        activity_trace: Array of activity values over time
        dt: Time step in milliseconds

    Returns:
        frequencies: Frequency array in Hz
        psd: Power spectral density
    """
    # Convert to samples per second
    fs = 1000.0 / dt  # Sampling frequency

    # Compute PSD using Welch's method
    frequencies, psd = signal.welch(activity_trace, fs=fs, nperseg=256)

    return frequencies, psd

# Compute spectrum
freqs, psd = compute_power_spectrum(activity_trace)

# Find peak frequency
peak_idx = np.argmax(psd[(freqs >= 1) & (freqs <= 100)])
peak_freq = freqs[(freqs >= 1) & (freqs <= 100)][peak_idx]
print(f"Peak frequency: {peak_freq:.1f} Hz")
```

### Band Power Analysis

```python
def compute_band_power(frequencies, psd, band):
    """Compute power in a specific frequency band."""
    low, high = band
    mask = (frequencies >= low) & (frequencies <= high)
    return np.trapz(psd[mask], frequencies[mask])

# Define bands
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

# Compute powers
for name, band in bands.items():
    power = compute_band_power(freqs, psd, band)
    print(f"{name}: {power:.2f}")
```

## Generating EEG-like Signals

### From Tissue to EEG

```python
def compute_eeg_signal(tissue, electrode_position, radius=5.0):
    """
    Compute EEG-like signal from tissue activity.

    Models a surface electrode measuring summed
    dendritic currents from nearby neurons.

    Args:
        tissue: Neural tissue
        electrode_position: (x, y, z) position
        radius: Recording radius

    Returns:
        eeg_value: Scalar EEG-like value
    """
    eeg = 0.0
    total_weight = 0.0

    for neuron in tissue.neurons.values():
        if not neuron.alive:
            continue

        # Distance from electrode
        dist = np.sqrt(
            (neuron.x - electrode_position[0])**2 +
            (neuron.y - electrode_position[1])**2 +
            (neuron.z - electrode_position[2])**2
        )

        if dist < radius:
            # Weight falls off with distance
            weight = 1.0 - (dist / radius)
            weight **= 2  # Square for sharper falloff

            # Contribution is weighted membrane potential
            # EEG primarily measures dendritic currents, not spikes
            eeg += weight * neuron.membrane_potential
            total_weight += weight

    if total_weight > 0:
        eeg /= total_weight

    return eeg

# Record EEG over time
eeg_trace = []
for step in range(1000):
    tissue.step(dt=0.5)

    # Record from "scalp" electrode
    eeg = compute_eeg_signal(tissue, electrode_position=(5.0, 5.0, 10.0))
    eeg_trace.append(eeg)

eeg_trace = np.array(eeg_trace)
```

### Multi-Channel EEG

```python
def compute_multi_channel_eeg(tissue, electrode_positions):
    """Compute EEG from multiple electrode positions."""
    eeg_signals = []

    for pos in electrode_positions:
        eeg = compute_eeg_signal(tissue, pos)
        eeg_signals.append(eeg)

    return np.array(eeg_signals)

# Define electrode grid
electrodes = [
    (2.5, 2.5, 10.0),  # Front-left
    (7.5, 2.5, 10.0),  # Front-right
    (2.5, 7.5, 10.0),  # Back-left
    (7.5, 7.5, 10.0),  # Back-right
    (5.0, 5.0, 10.0),  # Center
]

# Record multi-channel EEG
multi_eeg = []
for step in range(1000):
    tissue.step(dt=0.5)
    eeg = compute_multi_channel_eeg(tissue, electrodes)
    multi_eeg.append(eeg)

multi_eeg = np.array(multi_eeg)  # Shape: (time_steps, n_electrodes)
```

## Cross-Frequency Coupling

### Phase-Amplitude Coupling (PAC)

PAC measures how the phase of a slow oscillation modulates the amplitude of a faster one:

```python
def compute_phase_amplitude_coupling(signal, fs, phase_freq, amp_freq):
    """
    Compute phase-amplitude coupling.

    Args:
        signal: Raw signal
        fs: Sampling frequency
        phase_freq: Frequency for phase (slow)
        amp_freq: Frequency for amplitude (fast)

    Returns:
        pac_value: Coupling strength (0-1)
    """
    # Filter for phase (slow oscillation)
    phase_band = (phase_freq - 1, phase_freq + 1)
    phase_signal = bandpass_filter(signal, fs, phase_band)

    # Filter for amplitude (fast oscillation)
    amp_band = (amp_freq - 5, amp_freq + 5)
    amp_signal = bandpass_filter(signal, fs, amp_band)

    # Extract phase using Hilbert transform
    analytic_phase = signal.hilbert(phase_signal)
    phase = np.angle(analytic_phase)

    # Extract amplitude using Hilbert transform
    analytic_amp = signal.hilbert(amp_signal)
    amplitude = np.abs(analytic_amp)

    # Compute coupling (phase-locking of amplitude to phase)
    # Use Kullback-Leibler divergence method
    n_bins = 18
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    amp_by_phase = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
        if np.sum(mask) > 0:
            amp_by_phase[i] = np.mean(amplitude[mask])

    # Normalize
    amp_by_phase /= np.sum(amp_by_phase)

    # KL divergence from uniform
    uniform = np.ones(n_bins) / n_bins
    pac = np.sum(amp_by_phase * np.log(amp_by_phase / uniform + 1e-10))

    return pac

def bandpass_filter(data, fs, band):
    """Apply bandpass filter."""
    low, high = band
    nyq = fs / 2
    b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, data)

# Compute PAC: theta phase modulating gamma amplitude
fs = 1000.0 / 0.5  # 2000 Hz
pac = compute_phase_amplitude_coupling(eeg_trace, fs,
                                        phase_freq=6.0,  # Theta
                                        amp_freq=40.0)   # Gamma
print(f"Theta-Gamma PAC: {pac:.3f}")
```

## Entrainment and Resonance

### Entraining Neural Tissue

```python
def entrain_to_frequency(tissue, target_freq, duration_ms, amplitude):
    """
    Entrain neural tissue to a specific frequency.

    Uses rhythmic stimulation to synchronize network activity.
    """
    dt = 0.5
    n_steps = int(duration_ms / dt)
    activity_trace = []

    for step in range(n_steps):
        time_ms = step * dt
        phase = 2 * np.pi * target_freq * (time_ms / 1000.0)

        # Rhythmic stimulation
        stim = amplitude * (0.5 + 0.5 * np.sin(phase))

        tissue.stimulate((5.0, 5.0, 2.5), intensity=stim, radius=3.0)
        tissue.step(dt)

        # Record
        active = sum(1 for n in tissue.neurons.values()
                     if n.alive and n.state.name == 'ACTIVE')
        activity_trace.append(active)

    return np.array(activity_trace)

# Entrain to 10 Hz alpha
activity = entrain_to_frequency(tissue, target_freq=10.0,
                                 duration_ms=1000.0, amplitude=10.0)

# Check entrainment
freqs, psd = compute_power_spectrum(activity)
print(f"Peak frequency after entrainment: {freqs[np.argmax(psd)]:.1f} Hz")
```

### Resonance Frequency

```python
def find_resonance_frequency(tissue, freq_range=(1, 50), duration_ms=500):
    """
    Find the natural resonance frequency of the tissue.

    The tissue will respond most strongly at its resonance.
    """
    frequencies = np.linspace(freq_range[0], freq_range[1], 20)
    responses = []

    for freq in frequencies:
        # Reset tissue state
        for neuron in tissue.neurons.values():
            neuron.membrane_potential = -70.0

        # Apply stimulation at this frequency
        activity = []
        dt = 0.5

        for step in range(int(duration_ms / dt)):
            time_ms = step * dt
            phase = 2 * np.pi * freq * (time_ms / 1000.0)
            stim = 10.0 * np.sin(phase)

            tissue.stimulate((5.0, 5.0, 2.5), intensity=max(0, stim), radius=3.0)
            tissue.step(dt)

            active = sum(1 for n in tissue.neurons.values()
                         if n.alive and n.state.name == 'ACTIVE')
            activity.append(active)

        # Measure response power at stimulation frequency
        freqs, psd = compute_power_spectrum(np.array(activity))
        idx = np.argmin(np.abs(freqs - freq))
        responses.append(psd[idx])

    # Find peak response
    resonance_freq = frequencies[np.argmax(responses)]
    return resonance_freq, frequencies, responses

# Find resonance
res_freq, freqs, responses = find_resonance_frequency(tissue)
print(f"Tissue resonance frequency: {res_freq:.1f} Hz")
```

## Visualization

### Plotting Oscillations

```python
import matplotlib.pyplot as plt

def plot_oscillation_analysis(time_trace, activity_trace, fs):
    """Create comprehensive oscillation analysis plot."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Raw trace
    axes[0].plot(time_trace, activity_trace)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Activity')
    axes[0].set_title('Neural Activity Trace')

    # 2. Power spectrum
    freqs, psd = signal.welch(activity_trace, fs=fs, nperseg=256)
    axes[1].semilogy(freqs, psd)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('PSD')
    axes[1].set_title('Power Spectrum')
    axes[1].set_xlim(0, 50)

    # Add band annotations
    bands = [(0.5, 4, 'delta'), (4, 8, 'theta'), (8, 13, 'alpha'),
             (13, 30, 'beta'), (30, 50, 'gamma')]
    for low, high, name in bands:
        axes[1].axvspan(low, high, alpha=0.2, label=name)
    axes[1].legend(loc='upper right')

    # 3. Spectrogram
    f, t, Sxx = signal.spectrogram(activity_trace, fs=fs, nperseg=64)
    axes[2].pcolormesh(t * 1000, f, 10 * np.log10(Sxx), shading='gouraud')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_title('Spectrogram')
    axes[2].set_ylim(0, 50)

    plt.tight_layout()
    plt.savefig('oscillation_analysis.png')
    plt.close()
```

## Complete Example: Theta-Gamma Coupling

```python
from organic_neural_network import OrganicNeuralNetwork
import numpy as np

def demonstrate_theta_gamma_coupling():
    """
    Demonstrate theta-gamma coupling, important for memory.
    """
    # Create tissue
    tissue = OrganicNeuralNetwork(
        size=(10.0, 10.0, 5.0),
        initial_neurons=40,
        energy_supply=2.0
    )

    # Record activity during theta-modulated stimulation
    dt = 0.5
    duration_ms = 2000
    theta_freq = 6.0  # Hz

    activity_trace = []
    time_trace = []

    for step in range(int(duration_ms / dt)):
        time_ms = step * dt

        # Theta-modulated input
        theta_phase = 2 * np.pi * theta_freq * (time_ms / 1000.0)

        # Gamma bursts at theta peaks
        if np.sin(theta_phase) > 0.7:  # Near theta peak
            # Add gamma-frequency modulation
            gamma_phase = 2 * np.pi * 40.0 * (time_ms / 1000.0)
            gamma_mod = 0.3 * np.sin(gamma_phase)
            stim = 10.0 * (1 + gamma_mod)
        else:
            stim = 5.0  # Baseline

        tissue.stimulate((5.0, 5.0, 2.5), intensity=stim, radius=3.0)
        tissue.step(dt)

        # Record
        active = sum(1 for n in tissue.neurons.values()
                     if n.alive and n.state.name == 'ACTIVE')
        activity_trace.append(active)
        time_trace.append(time_ms)

    activity_trace = np.array(activity_trace)
    time_trace = np.array(time_trace)

    # Analyze
    fs = 1000.0 / dt
    freqs, psd = compute_power_spectrum(activity_trace, dt)

    print("Theta-Gamma Coupling Analysis")
    print("=" * 40)

    # Theta power
    theta_power = compute_band_power(freqs, psd, (4, 8))
    print(f"Theta power (4-8 Hz): {theta_power:.2f}")

    # Gamma power
    gamma_power = compute_band_power(freqs, psd, (30, 50))
    print(f"Gamma power (30-50 Hz): {gamma_power:.2f}")

    # Peak frequencies
    theta_peak = freqs[(freqs >= 4) & (freqs <= 8)][np.argmax(psd[(freqs >= 4) & (freqs <= 8)])]
    gamma_peak = freqs[(freqs >= 30) & (freqs <= 50)][np.argmax(psd[(freqs >= 30) & (freqs <= 50)])]
    print(f"Theta peak: {theta_peak:.1f} Hz")
    print(f"Gamma peak: {gamma_peak:.1f} Hz")

    return tissue, time_trace, activity_trace

# Run demonstration
tissue, time, activity = demonstrate_theta_gamma_coupling()
```

## References

- Buzsaki, G. (2006). "Rhythms of the Brain" - Oxford University Press
- Fries, P. (2005). "Communication Through Coherence" - Trends in Cognitive Sciences
- Canolty, R. T. & Knight, R. T. (2010). "The Functional Role of Cross-Frequency Coupling" - Trends in Cognitive Sciences

## Summary

In this tutorial, you learned:

1. **Brain Rhythm Bands**: Delta, theta, alpha, beta, gamma and their cognitive correlates
2. **Generating Oscillations**: Both emergent and driven rhythmic activity
3. **Analysis Methods**: Power spectrum, band power, spectrograms
4. **EEG-like Signals**: Computing scalp-like recordings from tissue activity
5. **Cross-Frequency Coupling**: How slow rhythms modulate fast ones
6. **Entrainment and Resonance**: Finding and using natural tissue frequencies

The key insight is that oscillations are not just epiphenomena - they are **computationally meaningful** patterns that coordinate neural activity across space and time, enabling functions like attention, memory encoding, and conscious awareness.
