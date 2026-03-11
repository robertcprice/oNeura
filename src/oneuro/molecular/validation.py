"""Electrophysiology-style validation protocols for the molecular GPU backend.

This module provides repeatable measurements for core single-neuron and
synaptic behaviors, plus comparisons against literature-derived reference
profiles. The emphasis is benchmarkability: measurements should be stable and
easy to rerun as the model is tuned.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .cuda_backend import CUDAMolecularBrain, NT_DA, NT_GABA, NT_GLU

try:
    import oneuro_metal
except ImportError:  # pragma: no cover - optional runtime dependency
    oneuro_metal = None


METAL_GPU_NEURON_THRESHOLD = 64


@dataclass
class ReferenceRange:
    """Inclusive reference window for a scalar benchmark metric."""

    lower: float
    upper: float
    unit: str = ""

    def contains(self, value: Optional[float]) -> Optional[bool]:
        if value is None:
            return None
        return self.lower <= float(value) <= self.upper


@dataclass
class ValidationCheck:
    """One physiology-style benchmark check."""

    name: str
    value: Optional[float]
    unit: str
    target: Optional[ReferenceRange]
    passed: Optional[bool]
    note: str = ""


@dataclass
class TargetCellProfile:
    """Reference physiology profile derived from published measurements."""

    key: str
    name: str
    source_label: str
    source_url: str
    targets: Dict[str, ReferenceRange]


@dataclass
class ValidationBackendInfo:
    """Execution backend used for a validation run."""

    requested_device: str
    resolved_backend: str
    n_sim_neurons: int
    gpu_active: bool
    gpu_dispatch_active: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class CurrentClampMetrics:
    """Current-clamp style measurements for a single model neuron."""

    resting_potential_mv: float
    subthreshold_current_ua: Optional[float]
    subthreshold_delta_mv: float
    input_gain_mv_per_ua: Optional[float]
    membrane_tau_ms: Optional[float]
    rheobase_current_ua: Optional[float]
    fi_curve_hz: Dict[float, float] = field(default_factory=dict)
    first_spike_threshold_mv: Optional[float] = None
    first_spike_peak_mv: Optional[float] = None
    first_spike_half_width_ms: Optional[float] = None
    first_spike_ahp_amplitude_mv: Optional[float] = None
    adaptation_probe_current_ua: Optional[float] = None
    frequency_adaptation_ratio: Optional[float] = None
    absolute_refractory_ms: Optional[float] = None


@dataclass
class SynapticResponseMetrics:
    """Postsynaptic response to a brief presynaptic spike train."""

    synapse_type: str
    baseline_post_mv: float
    peak_delta_mv: float
    peak_time_ms: Optional[float]
    half_decay_ms: Optional[float]
    pre_spike_count: int
    post_spike_count: int


@dataclass
class PlasticityMetrics:
    """Three-factor plasticity measurements from paired pre/post activation."""

    pre_before_post_no_da_delta: float
    pre_before_post_da_delta: float
    post_before_pre_da_delta: float
    rewarded_pairing_final_strength: float
    unrewarded_pairing_final_strength: float
    reverse_pairing_final_strength: float


@dataclass
class ValidationReport:
    """Aggregate neuron validation report."""

    backend: ValidationBackendInfo
    current_clamp: CurrentClampMetrics
    excitatory_synapse: SynapticResponseMetrics
    inhibitory_synapse: SynapticResponseMetrics
    plasticity: PlasticityMetrics
    checks: List[ValidationCheck] = field(default_factory=list)
    reference_profiles: List[TargetCellProfile] = field(default_factory=list)
    reference_comparisons: Dict[str, List[ValidationCheck]] = field(default_factory=dict)
    reference_suggestions: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for check in data["checks"]:
            target = check.get("target")
            if target is None:
                continue
            check["target"] = dict(target)
        for key, checks in data.get("reference_comparisons", {}).items():
            for check in checks:
                target = check.get("target")
                if target is None:
                    continue
                check["target"] = dict(target)
        return data


DEFAULT_REFERENCE_WINDOWS: Dict[str, ReferenceRange] = {
    "resting_potential_mv": ReferenceRange(-80.0, -55.0, "mV"),
    "first_spike_threshold_mv": ReferenceRange(-50.0, -15.0, "mV"),
    "absolute_refractory_ms": ReferenceRange(1.0, 5.0, "ms"),
}

REFERENCE_PROFILES: Dict[str, TargetCellProfile] = {
    "l5_regular_spiking_pyramidal": TargetCellProfile(
        key="l5_regular_spiking_pyramidal",
        name="Layer 5 Regular-Spiking Pyramidal",
        source_label="van Aerde & Feldmeyer 2015 / postrhinal regular-spiking pyramidal",
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC4563020/",
        targets={
            "resting_potential_mv": ReferenceRange(-66.0, -60.0, "mV"),
            "membrane_tau_ms": ReferenceRange(48.0, 60.0, "ms"),
            "first_spike_threshold_mv": ReferenceRange(-49.0, -37.0, "mV"),
            "first_spike_half_width_ms": ReferenceRange(2.0, 3.0, "ms"),
            "first_spike_ahp_amplitude_mv": ReferenceRange(5.0, 8.0, "mV"),
            "frequency_adaptation_ratio": ReferenceRange(0.20, 0.40, "ratio"),
        },
    ),
    "neocortical_pyramidal_wt": TargetCellProfile(
        key="neocortical_pyramidal_wt",
        name="Neocortical Pyramidal WT",
        source_label="Hedrich et al. 2014 / WT pyramidal neurons",
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC4036256/",
        targets={
            "resting_potential_mv": ReferenceRange(-60.0, -56.0, "mV"),
            "first_spike_threshold_mv": ReferenceRange(-35.0, -29.0, "mV"),
            "first_spike_half_width_ms": ReferenceRange(1.8, 2.4, "ms"),
            "frequency_adaptation_ratio": ReferenceRange(0.75, 0.95, "ratio"),
        },
    ),
}


class _ValidationBrain:
    """Minimal interface shared by validation backends."""

    dt: float
    backend_info: ValidationBackendInfo

    def step(self) -> None:
        raise NotImplementedError

    def stimulate(self, neuron_idx: int, current_ua: float) -> None:
        raise NotImplementedError

    def voltage(self, neuron_idx: int) -> float:
        raise NotImplementedError

    def prev_voltage(self, neuron_idx: int) -> float:
        raise NotImplementedError

    def fired(self, neuron_idx: int) -> bool:
        raise NotImplementedError

    def spike_count(self, neuron_idx: int) -> int:
        raise NotImplementedError

    def synapse_strength(self, synapse_idx: int) -> float:
        raise NotImplementedError

    @property
    def is_metal(self) -> bool:
        return self.backend_info.resolved_backend == "rust_metal"

    def add_dopamine(self, neuron_indices: Sequence[int], delta_nm: float) -> None:
        raise NotImplementedError

    def set_conductance_scale_all(self, channel_idx: int, scale: float) -> None:
        """Optionally tune a conductance family across the assay."""

    def set_membrane_capacitance(self, capacitance_uf: float) -> None:
        """Optionally tune the effective membrane capacitance."""

    def set_spike_threshold(self, threshold_mv: float) -> None:
        """Optionally tune spike detection threshold."""

    def set_refractory_period(self, refractory_period_ms: float) -> None:
        """Optionally tune refractory period."""


class _TorchValidationBrain(_ValidationBrain):
    """Validation wrapper over the PyTorch backend."""

    def __init__(
        self,
        n_neurons: int,
        device: str,
        psc_scale: float,
        edges: Optional[Sequence[tuple[int, int, int]]] = None,
    ) -> None:
        self.inner = CUDAMolecularBrain(n_neurons, device=device, psc_scale=psc_scale)
        self.inner.disable_interval_biology()
        self.inner.set_triton_enabled(False)
        if edges:
            pre = torch.tensor([edge[0] for edge in edges], dtype=torch.int64)
            post = torch.tensor([edge[1] for edge in edges], dtype=torch.int64)
            weights = torch.tensor([1.0 for _ in edges], dtype=torch.float32)
            nt_types = torch.tensor([edge[2] for edge in edges], dtype=torch.int32)
            self.inner.add_synapses(pre, post, weights, nt_types)
        device_type = str(getattr(self.inner.device, "type", self.inner.device))
        self.dt = float(self.inner.dt)
        self.backend_info = ValidationBackendInfo(
            requested_device=device,
            resolved_backend=f"torch:{device_type}",
            n_sim_neurons=n_neurons,
            gpu_active=device_type != "cpu",
            gpu_dispatch_active=device_type != "cpu",
            notes=[],
        )

    def step(self) -> None:
        self.inner.step()

    def stimulate(self, neuron_idx: int, current_ua: float) -> None:
        self.inner.stimulate(neuron_idx, current_ua)

    def voltage(self, neuron_idx: int) -> float:
        return float(self.inner.voltage[neuron_idx])

    def prev_voltage(self, neuron_idx: int) -> float:
        return float(self.inner.prev_voltage[neuron_idx])

    def fired(self, neuron_idx: int) -> bool:
        return bool(self.inner.fired[neuron_idx])

    def spike_count(self, neuron_idx: int) -> int:
        return int(self.inner.spike_count[neuron_idx])

    def synapse_strength(self, synapse_idx: int) -> float:
        return float(self.inner.syn_strength[synapse_idx])

    def set_synapse_strength(self, synapse_idx: int, strength: float) -> None:
        self.inner.syn_strength[synapse_idx] = float(strength)

    def add_dopamine(self, neuron_indices: Sequence[int], delta_nm: float) -> None:
        self.inner.nt_conc[list(neuron_indices), NT_DA] += float(delta_nm)

    def set_conductance_scale_all(self, channel_idx: int, scale: float) -> None:
        self.inner.conductance_scale[:, int(channel_idx)] = float(scale)


class _MetalValidationBrain(_ValidationBrain):
    """Validation wrapper over the Rust/Metal backend."""

    def __init__(
        self,
        n_neurons: int,
        requested_device: str,
        psc_scale: float,
        edges: Optional[Sequence[tuple[int, int, int]]] = None,
    ) -> None:
        if oneuro_metal is None:
            raise RuntimeError("oneuro_metal is not available in this interpreter")
        n_sim_neurons = max(int(n_neurons), METAL_GPU_NEURON_THRESHOLD)
        rust_edges = [(int(pre), int(post), int(nt)) for pre, post, nt in (edges or [])]
        if not rust_edges:
            rust_edges = [(1, 2, NT_GLU)]
        if rust_edges:
            self.inner = oneuro_metal.MolecularBrain.from_edges(
                n_sim_neurons,
                rust_edges,
                psc_scale=float(psc_scale),
                dt=0.1,
            )
        else:
            self.inner = oneuro_metal.MolecularBrain(
                n_sim_neurons,
                psc_scale=float(psc_scale),
                dt=0.1,
            )
        self.inner.set_gpu_enabled(True)
        self.inner.set_glia_enabled(False)
        self.inner.set_circadian_enabled(False)
        if hasattr(self.inner, "set_pharmacology_enabled"):
            self.inner.set_pharmacology_enabled(False)
        if hasattr(self.inner, "set_gene_expression_enabled"):
            self.inner.set_gene_expression_enabled(False)
        if hasattr(self.inner, "set_metabolism_enabled"):
            self.inner.set_metabolism_enabled(False)
        if hasattr(self.inner, "set_microtubules_enabled"):
            self.inner.set_microtubules_enabled(False)
        if hasattr(self.inner, "enable_latency_benchmark_mode"):
            self.inner.enable_latency_benchmark_mode()
        gpu_active = bool(self.inner.gpu_active())
        gpu_dispatch_active = bool(self.inner.gpu_dispatch_active())
        notes: List[str] = []
        if n_sim_neurons > n_neurons:
            notes.append(
                f"Expanded assay from {n_neurons} to {n_sim_neurons} neurons to satisfy the Metal dispatch threshold."
            )
        if not gpu_dispatch_active:
            raise RuntimeError("Metal validation requested but GPU dispatch is not active")
        self.dt = float(self.inner.dt)
        self.backend_info = ValidationBackendInfo(
            requested_device=requested_device,
            resolved_backend="rust_metal",
            n_sim_neurons=n_sim_neurons,
            gpu_active=gpu_active,
            gpu_dispatch_active=gpu_dispatch_active,
            notes=notes,
        )

    def step(self) -> None:
        self.inner.step()

    def stimulate(self, neuron_idx: int, current_ua: float) -> None:
        self.inner.stimulate(neuron_idx, float(current_ua))

    def voltage(self, neuron_idx: int) -> float:
        return float(self.inner.voltages()[neuron_idx])

    def prev_voltage(self, neuron_idx: int) -> float:
        if hasattr(self.inner, "prev_voltages"):
            return float(self.inner.prev_voltages()[neuron_idx])
        return self.voltage(neuron_idx)

    def fired(self, neuron_idx: int) -> bool:
        return bool(self.inner.fired()[neuron_idx])

    def spike_count(self, neuron_idx: int) -> int:
        return int(self.inner.spike_counts()[neuron_idx])

    def synapse_strength(self, synapse_idx: int) -> float:
        if hasattr(self.inner, "synapse_weight"):
            return float(self.inner.synapse_weight(synapse_idx))
        return float(self.inner.synapse_strength(synapse_idx))

    def set_synapse_strength(self, synapse_idx: int, strength: float) -> None:
        self.inner.set_synapse_strengths([int(synapse_idx)], [float(strength)])

    def add_dopamine(self, neuron_indices: Sequence[int], delta_nm: float) -> None:
        self.inner.add_nt_concentration_many(list(map(int, neuron_indices)), NT_DA, float(delta_nm))

    def set_conductance_scale_all(self, channel_idx: int, scale: float) -> None:
        scale = float(scale)
        for neuron_idx in range(int(self.backend_info.n_sim_neurons)):
            self.inner.set_conductance_scale(int(neuron_idx), int(channel_idx), scale)

    def set_membrane_capacitance(self, capacitance_uf: float) -> None:
        if hasattr(self.inner, "set_membrane_capacitance"):
            self.inner.set_membrane_capacitance(float(capacitance_uf))

    def set_spike_threshold(self, threshold_mv: float) -> None:
        if hasattr(self.inner, "set_spike_threshold"):
            self.inner.set_spike_threshold(float(threshold_mv))

    def set_refractory_period(self, refractory_period_ms: float) -> None:
        if hasattr(self.inner, "set_refractory_period"):
            self.inner.set_refractory_period(float(refractory_period_ms))


def _resolve_validation_backend(device: str) -> str:
    """Choose the validation backend for a requested device string."""
    requested = device.lower()
    if requested in {"metal", "gpu", "rust", "metal_gpu"}:
        if oneuro_metal is None:
            raise RuntimeError("Metal backend requested but oneuro_metal is unavailable")
        if not bool(oneuro_metal.has_gpu()):
            raise RuntimeError("Metal backend requested but no Metal GPU is available")
        return "metal"
    if requested == "auto":
        if oneuro_metal is not None and bool(oneuro_metal.has_gpu()):
            return "metal"
        return "torch"
    return "torch"


def _make_validation_brain(
    n_neurons: int,
    device: str = "auto",
    psc_scale: float = 300.0,
    edges: Optional[Sequence[tuple[int, int, int]]] = None,
    apply_profile: bool = True,
) -> _ValidationBrain:
    """Create a deterministic validation brain on the requested backend."""
    backend = _resolve_validation_backend(device)
    if backend == "metal":
        brain = _MetalValidationBrain(
            n_neurons=n_neurons,
            requested_device=device,
            psc_scale=psc_scale,
            edges=edges,
        )
    else:
        brain = _TorchValidationBrain(
            n_neurons=n_neurons,
            device=device,
            psc_scale=psc_scale,
            edges=edges,
        )
    if apply_profile:
        _apply_cortical_validation_profile(brain)
    return brain


def _apply_cortical_validation_profile(brain: _ValidationBrain) -> None:
    """Bias the assay toward a cortical regular-spiking regime."""
    if not brain.is_metal:
        return
    brain.set_membrane_capacitance(6.0)
    brain.set_spike_threshold(-42.0)
    brain.set_refractory_period(2.5)


def _probe_backend(device: str) -> ValidationBackendInfo:
    """Create a small validation brain and report the active backend."""
    return _make_validation_brain(2, device=device, psc_scale=300.0).backend_info


def _pulse_neuron(
    brain: _ValidationBrain,
    neuron_idx: int,
    current_ua: float,
    pulse_steps: int,
) -> int:
    """Inject repeated current steps and return the number of emitted spikes."""
    spikes = 0
    for _ in range(max(1, int(pulse_steps))):
        brain.stimulate(neuron_idx, current_ua)
        brain.step()
        spikes += int(brain.fired(neuron_idx))
    return spikes


def _mean_tail(values: Sequence[float], window: int = 200) -> float:
    """Return the mean of the trailing window, or of the full list if shorter."""
    if not values:
        return 0.0
    tail = values[-min(len(values), int(window)) :]
    return float(sum(tail) / max(1, len(tail)))


def _first_crossing_time_ms(
    trace: Sequence[float],
    target: float,
    dt_ms: float,
) -> Optional[float]:
    """Return first time the trace crosses target."""
    for idx, value in enumerate(trace):
        if value >= target:
            return float((idx + 1) * dt_ms)
    return None


def _extract_first_spike_shape(
    voltage_trace: Sequence[float],
    prev_trace: Sequence[float],
    fired_trace: Sequence[bool],
    dt_ms: float,
) -> Dict[str, Optional[float]]:
    """Estimate threshold, peak, half-width, and AHP from the first spike."""
    first_idx = next((idx for idx, fired in enumerate(fired_trace) if fired), None)
    if first_idx is None:
        return {
            "threshold_mv": None,
            "peak_mv": None,
            "half_width_ms": None,
            "ahp_amplitude_mv": None,
        }

    threshold_mv = float(prev_trace[first_idx])
    search_end = min(len(voltage_trace), first_idx + 40)
    spike_window = list(voltage_trace[first_idx:search_end])
    if not spike_window:
        return {
            "threshold_mv": threshold_mv,
            "peak_mv": None,
            "half_width_ms": None,
            "ahp_amplitude_mv": None,
        }

    peak_mv = float(max(spike_window))
    peak_idx = first_idx + spike_window.index(peak_mv)
    half_width_ms = None
    half_level = threshold_mv + 0.5 * (peak_mv - threshold_mv)

    rise_cross = None
    for idx in range(max(0, first_idx - 20), peak_idx):
        left = float(voltage_trace[idx])
        right = float(voltage_trace[idx + 1])
        if left < half_level <= right:
            rise_cross = idx + 1
    fall_cross = None
    for idx in range(peak_idx, min(len(voltage_trace) - 1, peak_idx + 80)):
        left = float(voltage_trace[idx])
        right = float(voltage_trace[idx + 1])
        if left >= half_level > right:
            fall_cross = idx + 1
            break
    if rise_cross is not None and fall_cross is not None and fall_cross > rise_cross:
        half_width_ms = float((fall_cross - rise_cross) * dt_ms)

    ahp_window = voltage_trace[peak_idx : min(len(voltage_trace), peak_idx + 120)]
    ahp_amplitude_mv = None
    if ahp_window:
        min_after_spike = float(min(ahp_window))
        ahp_amplitude_mv = float(threshold_mv - min_after_spike)

    return {
        "threshold_mv": threshold_mv,
        "peak_mv": peak_mv,
        "half_width_ms": half_width_ms,
        "ahp_amplitude_mv": ahp_amplitude_mv,
    }


def _compute_frequency_adaptation_ratio(
    fired_trace: Sequence[bool],
    dt_ms: float,
) -> Optional[float]:
    """Compute adaptation as the 4th-interval frequency over the 2nd-interval frequency."""
    spike_steps = [idx for idx, fired in enumerate(fired_trace) if fired]
    if len(spike_steps) < 5:
        return None
    intervals_ms = [(spike_steps[i + 1] - spike_steps[i]) * dt_ms for i in range(len(spike_steps) - 1)]
    if len(intervals_ms) < 4 or intervals_ms[1] <= 0.0 or intervals_ms[3] <= 0.0:
        return None
    freq2 = 1000.0 / intervals_ms[1]
    freq4 = 1000.0 / intervals_ms[3]
    return float(freq4 / max(freq2, 1e-6))


def _report_metric_value(report: ValidationReport, metric_name: str) -> Optional[float]:
    """Extract a named scalar metric from a validation report."""
    if hasattr(report.current_clamp, metric_name):
        value = getattr(report.current_clamp, metric_name)
        return None if value is None else float(value)
    if hasattr(report.excitatory_synapse, metric_name):
        value = getattr(report.excitatory_synapse, metric_name)
        return None if value is None else float(value)
    if hasattr(report.inhibitory_synapse, metric_name):
        value = getattr(report.inhibitory_synapse, metric_name)
        return None if value is None else float(value)
    if hasattr(report.plasticity, metric_name):
        value = getattr(report.plasticity, metric_name)
        return None if value is None else float(value)
    return None


def compare_report_to_profile(
    report: ValidationReport,
    profile: TargetCellProfile,
) -> List[ValidationCheck]:
    """Compare a measured report against one reference profile."""
    checks: List[ValidationCheck] = []
    for metric_name, target in profile.targets.items():
        value = _report_metric_value(report, metric_name)
        checks.append(
            ValidationCheck(
                name=metric_name,
                value=value,
                unit=target.unit,
                target=target,
                passed=target.contains(value),
                note=f"Reference profile: {profile.name}",
            )
        )
    return checks


def generate_calibration_suggestions(
    report: ValidationReport,
    profile: TargetCellProfile,
) -> List[str]:
    """Generate coarse parameter-tuning suggestions from profile mismatches."""
    suggestions: List[str] = []

    tau = report.current_clamp.membrane_tau_ms
    tau_target = profile.targets.get("membrane_tau_ms")
    if tau is not None and tau_target is not None:
        if tau < tau_target.lower:
            suggestions.append(
                "Increase membrane time constant by raising capacitance or reducing leak conductance."
            )
        elif tau > tau_target.upper:
            suggestions.append(
                "Decrease membrane time constant by lowering capacitance or increasing leak conductance."
            )

    threshold = report.current_clamp.first_spike_threshold_mv
    threshold_target = profile.targets.get("first_spike_threshold_mv")
    if threshold is not None and threshold_target is not None:
        if threshold > threshold_target.upper:
            suggestions.append(
                "Make spike initiation more excitable by increasing sodium activation or reducing outward current near threshold."
            )
        elif threshold < threshold_target.lower:
            suggestions.append(
                "Make spike initiation less excitable by reducing sodium drive or increasing outward current near threshold."
            )

    half_width = report.current_clamp.first_spike_half_width_ms
    half_width_target = profile.targets.get("first_spike_half_width_ms")
    if half_width is not None and half_width_target is not None:
        if half_width < half_width_target.lower:
            suggestions.append(
                "Broaden spikes by increasing capacitance or weakening fast potassium repolarization."
            )
        elif half_width > half_width_target.upper:
            suggestions.append(
                "Narrow spikes by strengthening repolarizing potassium conductance or reducing capacitance."
            )

    ahp = report.current_clamp.first_spike_ahp_amplitude_mv
    ahp_target = profile.targets.get("first_spike_ahp_amplitude_mv")
    if ahp is not None and ahp_target is not None:
        if ahp > ahp_target.upper:
            suggestions.append(
                "Reduce after-hyperpolarization depth by weakening fast/slow potassium currents after spikes."
            )
        elif ahp < ahp_target.lower:
            suggestions.append(
                "Increase after-hyperpolarization depth with stronger repolarizing potassium currents."
            )

    adaptation = report.current_clamp.frequency_adaptation_ratio
    adaptation_target = profile.targets.get("frequency_adaptation_ratio")
    if adaptation is not None and adaptation_target is not None:
        if adaptation > adaptation_target.upper:
            suggestions.append(
                "Add stronger spike-frequency adaptation, such as a slow potassium or calcium-dependent adaptation current."
            )
        elif adaptation < adaptation_target.lower:
            suggestions.append(
                "Reduce adaptation strength so late-spike frequency remains closer to early-spike frequency."
            )

    if not suggestions:
        suggestions.append("Measured metrics fall within the current target windows.")
    return suggestions


def measure_current_clamp(
    device: str = "auto",
    settle_steps: int = 800,
    pulse_steps: int = 1000,
    recovery_steps: int = 400,
    current_candidates_ua: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0),
    spike_pulse_current_ua: float = 50.0,
    spike_pulse_steps: int = 10,
) -> CurrentClampMetrics:
    """Measure basic current-clamp metrics on a single neuron."""
    if _resolve_validation_backend(device) == "metal":
        settle_steps = max(int(settle_steps), 2000)
    if _resolve_validation_backend(device) == "metal" and spike_pulse_current_ua == 50.0:
        spike_pulse_current_ua = 200.0
    traces: Dict[float, Dict[str, Any]] = {}
    subthreshold_current = None
    rheobase_current = None

    for current in current_candidates_ua:
        brain = _make_validation_brain(1, device=device, psc_scale=300.0)
        baseline: List[float] = []
        pulse_trace: List[float] = []
        pulse_prev: List[float] = []
        pulse_fired: List[bool] = []

        for step_idx in range(settle_steps + pulse_steps + recovery_steps):
            if settle_steps <= step_idx < (settle_steps + pulse_steps):
                brain.stimulate(0, float(current))
            brain.step()
            voltage = brain.voltage(0)
            if step_idx < settle_steps:
                baseline.append(voltage)
            elif step_idx < (settle_steps + pulse_steps):
                pulse_trace.append(voltage)
                pulse_prev.append(brain.prev_voltage(0))
                pulse_fired.append(brain.fired(0))

        resting_mv = _mean_tail(baseline)
        steady_mv = _mean_tail(pulse_trace)
        delta_mv = steady_mv - resting_mv
        spike_count = int(sum(int(v) for v in pulse_fired))

        traces[float(current)] = {
            "resting_mv": resting_mv,
            "steady_mv": steady_mv,
            "delta_mv": delta_mv,
            "pulse_trace": pulse_trace,
            "pulse_prev": pulse_prev,
            "pulse_fired": pulse_fired,
            "spike_count": spike_count,
        }

        if spike_count == 0:
            subthreshold_current = float(current)
        elif rheobase_current is None:
            rheobase_current = float(current)

    if subthreshold_current is None:
        subthreshold_current = float(current_candidates_ua[0])
    sub = traces[subthreshold_current]

    membrane_tau_ms = None
    steady_mv = float(sub["steady_mv"])
    resting_mv = float(sub["resting_mv"])
    if steady_mv > resting_mv:
        target_mv = resting_mv + 0.632 * (steady_mv - resting_mv)
        membrane_tau_ms = _first_crossing_time_ms(
            sub["pulse_trace"], target_mv, dt_ms=0.1
        )

    fi_curve_hz: Dict[float, float] = {}
    for current, payload in traces.items():
        pulse_seconds = pulse_steps * 0.0001
        fi_curve_hz[float(current)] = float(payload["spike_count"]) / pulse_seconds

    first_spike_threshold_mv = None
    first_spike_peak_mv = None
    first_spike_half_width_ms = None
    first_spike_ahp_amplitude_mv = None
    adaptation_probe_current_ua = None
    frequency_adaptation_ratio = None
    if rheobase_current is not None:
        rheo = traces[rheobase_current]
        spike_shape = _extract_first_spike_shape(
            rheo["pulse_trace"],
            rheo["pulse_prev"],
            rheo["pulse_fired"],
            dt_ms=0.1,
        )
        first_spike_threshold_mv = spike_shape["threshold_mv"]
        first_spike_peak_mv = spike_shape["peak_mv"]
        first_spike_half_width_ms = spike_shape["half_width_ms"]
        first_spike_ahp_amplitude_mv = spike_shape["ahp_amplitude_mv"]

    for current in current_candidates_ua:
        payload = traces[float(current)]
        ratio = _compute_frequency_adaptation_ratio(payload["pulse_fired"], dt_ms=0.1)
        if ratio is not None:
            adaptation_probe_current_ua = float(current)
            frequency_adaptation_ratio = ratio
            break

    # Double-pulse refractory probe.
    refractory_ms = None
    for gap_steps in range(0, 25):
        brain = _make_validation_brain(1, device=device, psc_scale=300.0)
        spike_steps: List[int] = []
        for _ in range(20):
            brain.step()
        cursor = 20
        for _ in range(spike_pulse_steps):
            brain.stimulate(0, spike_pulse_current_ua)
            brain.step()
            if brain.fired(0):
                spike_steps.append(cursor)
            cursor += 1
        for _ in range(gap_steps):
            brain.step()
            if brain.fired(0):
                spike_steps.append(cursor)
            cursor += 1
        for _ in range(spike_pulse_steps):
            brain.stimulate(0, spike_pulse_current_ua)
            brain.step()
            if brain.fired(0):
                spike_steps.append(cursor)
            cursor += 1
        if len(spike_steps) >= 2:
            refractory_ms = float((spike_steps[1] - spike_steps[0]) * brain.dt)
            break

    input_gain = None
    if subthreshold_current and subthreshold_current > 0.0:
        input_gain = float(sub["delta_mv"]) / float(subthreshold_current)

    return CurrentClampMetrics(
        resting_potential_mv=float(sub["resting_mv"]),
        subthreshold_current_ua=float(subthreshold_current),
        subthreshold_delta_mv=float(sub["delta_mv"]),
        input_gain_mv_per_ua=input_gain,
        membrane_tau_ms=membrane_tau_ms,
        rheobase_current_ua=rheobase_current,
        fi_curve_hz=fi_curve_hz,
        first_spike_threshold_mv=first_spike_threshold_mv,
        first_spike_peak_mv=first_spike_peak_mv,
        first_spike_half_width_ms=first_spike_half_width_ms,
        first_spike_ahp_amplitude_mv=first_spike_ahp_amplitude_mv,
        adaptation_probe_current_ua=adaptation_probe_current_ua,
        frequency_adaptation_ratio=frequency_adaptation_ratio,
        absolute_refractory_ms=refractory_ms,
    )


def measure_synaptic_response(
    inhibitory: bool = False,
    device: str = "auto",
    syn_weight: float = 5.0,
    pulse_current_ua: float = 50.0,
    pulse_steps: int = 10,
) -> SynapticResponseMetrics:
    """Measure the sign and decay of a one-synapse postsynaptic response."""
    nt_type = NT_GABA if inhibitory else NT_GLU
    settle_steps = 40
    effective_syn_weight = float(syn_weight)
    effective_pulse_current = float(pulse_current_ua)
    if _resolve_validation_backend(device) == "metal":
        settle_steps = 800
        if syn_weight == 5.0:
            effective_syn_weight = 1.0 if inhibitory else 0.5
        if pulse_current_ua == 50.0:
            effective_pulse_current = 8.0
    brain = _make_validation_brain(
        2,
        device=device,
        psc_scale=300.0,
        edges=[(0, 1, nt_type)],
    )
    brain.set_synapse_strength(0, effective_syn_weight)

    baseline_trace: List[float] = []
    response_trace: List[float] = []
    for _ in range(settle_steps):
        brain.step()
    for _ in range(40):
        brain.step()
        baseline_trace.append(brain.voltage(1))
    for _ in range(max(1, int(pulse_steps))):
        brain.stimulate(0, effective_pulse_current)
        brain.step()
        response_trace.append(brain.voltage(1))
    for _ in range(120):
        brain.step()
        response_trace.append(brain.voltage(1))

    baseline_mv = _mean_tail(baseline_trace, window=20)
    if inhibitory:
        peak_value = float(min(response_trace))
        peak_index = int(min(range(len(response_trace)), key=lambda i: response_trace[i]))
    else:
        peak_value = float(max(response_trace))
        peak_index = int(max(range(len(response_trace)), key=lambda i: response_trace[i]))
    peak_delta = peak_value - baseline_mv

    half_decay_ms = None
    if peak_delta != 0.0:
        half_value = baseline_mv + 0.5 * peak_delta
        for idx in range(peak_index + 1, len(response_trace)):
            value = response_trace[idx]
            if inhibitory:
                if value >= half_value:
                    half_decay_ms = float((idx - peak_index) * brain.dt)
                    break
            else:
                if value <= half_value:
                    half_decay_ms = float((idx - peak_index) * brain.dt)
                    break

    return SynapticResponseMetrics(
        synapse_type="inhibitory" if inhibitory else "excitatory",
        baseline_post_mv=baseline_mv,
        peak_delta_mv=peak_delta,
        peak_time_ms=float((peak_index + 1) * brain.dt),
        half_decay_ms=half_decay_ms,
        pre_spike_count=brain.spike_count(0),
        post_spike_count=brain.spike_count(1),
    )


def measure_dopamine_gated_plasticity(
    device: str = "auto",
    pairing_trials: int = 12,
    inter_trial_steps: int = 20,
    post_pair_steps: int = 40,
    pair_delay_steps: int = 3,
    pulse_current_ua: float = 50.0,
    pulse_steps: int = 10,
    dopamine_amount: float = 200.0,
) -> PlasticityMetrics:
    """Measure eligibility-trace learning with and without dopamine."""

    def _run_pairing(pre_before_post: bool, dopamine: bool) -> float:
        brain = _make_validation_brain(
            2,
            device=device,
            psc_scale=300.0,
            edges=[(0, 1, NT_GLU)],
            apply_profile=False,
        )
        brain.set_synapse_strength(0, 2.0)
        base_strength = brain.synapse_strength(0)
        for _ in range(max(1, int(pairing_trials))):
            for _ in range(max(0, int(inter_trial_steps))):
                brain.step()
            if pre_before_post:
                _pulse_neuron(brain, 0, pulse_current_ua, pulse_steps)
                for _ in range(max(0, int(pair_delay_steps))):
                    brain.step()
                _pulse_neuron(brain, 1, pulse_current_ua, pulse_steps)
            else:
                _pulse_neuron(brain, 1, pulse_current_ua, pulse_steps)
                for _ in range(max(0, int(pair_delay_steps))):
                    brain.step()
                _pulse_neuron(brain, 0, pulse_current_ua, pulse_steps)
            if dopamine:
                brain.add_dopamine([1], dopamine_amount)
            for _ in range(max(0, int(post_pair_steps))):
                brain.step()
        final_strength = brain.synapse_strength(0)
        return float(final_strength - base_strength), float(final_strength)

    pre_no_da, pre_no_da_final = _run_pairing(pre_before_post=True, dopamine=False)
    pre_da, pre_da_final = _run_pairing(pre_before_post=True, dopamine=True)
    post_da, post_da_final = _run_pairing(pre_before_post=False, dopamine=True)

    return PlasticityMetrics(
        pre_before_post_no_da_delta=pre_no_da,
        pre_before_post_da_delta=pre_da,
        post_before_pre_da_delta=post_da,
        rewarded_pairing_final_strength=pre_da_final,
        unrewarded_pairing_final_strength=pre_no_da_final,
        reverse_pairing_final_strength=post_da_final,
    )


def _build_validation_checks(report: ValidationReport) -> List[ValidationCheck]:
    """Attach broad physiology-style checks to the measured report."""
    checks = [
        ValidationCheck(
            name="resting_potential_mv",
            value=report.current_clamp.resting_potential_mv,
            unit="mV",
            target=DEFAULT_REFERENCE_WINDOWS["resting_potential_mv"],
            passed=DEFAULT_REFERENCE_WINDOWS["resting_potential_mv"].contains(
                report.current_clamp.resting_potential_mv
            ),
        ),
        ValidationCheck(
            name="first_spike_threshold_mv",
            value=report.current_clamp.first_spike_threshold_mv,
            unit="mV",
            target=DEFAULT_REFERENCE_WINDOWS["first_spike_threshold_mv"],
            passed=DEFAULT_REFERENCE_WINDOWS["first_spike_threshold_mv"].contains(
                report.current_clamp.first_spike_threshold_mv
            ),
        ),
        ValidationCheck(
            name="absolute_refractory_ms",
            value=report.current_clamp.absolute_refractory_ms,
            unit="ms",
            target=DEFAULT_REFERENCE_WINDOWS["absolute_refractory_ms"],
            passed=DEFAULT_REFERENCE_WINDOWS["absolute_refractory_ms"].contains(
                report.current_clamp.absolute_refractory_ms
            ),
        ),
        ValidationCheck(
            name="first_spike_half_width_ms",
            value=report.current_clamp.first_spike_half_width_ms,
            unit="ms",
            target=None,
            passed=(
                report.current_clamp.first_spike_half_width_ms is not None
                and report.current_clamp.first_spike_half_width_ms > 0.05
            ),
            note="First spike should have a measurable nonzero half-width.",
        ),
        ValidationCheck(
            name="first_spike_ahp_amplitude_mv",
            value=report.current_clamp.first_spike_ahp_amplitude_mv,
            unit="mV",
            target=None,
            passed=(
                report.current_clamp.first_spike_ahp_amplitude_mv is not None
                and report.current_clamp.first_spike_ahp_amplitude_mv > 1.0
            ),
            note="The first spike should be followed by an after-hyperpolarization.",
        ),
        ValidationCheck(
            name="frequency_adaptation_ratio",
            value=report.current_clamp.frequency_adaptation_ratio,
            unit="ratio",
            target=None,
            passed=(
                report.current_clamp.frequency_adaptation_ratio is None
                or report.current_clamp.frequency_adaptation_ratio > 0.0
            ),
            note="When enough spikes are available, the adaptation ratio should be positive.",
        ),
        ValidationCheck(
            name="excitatory_peak_delta_mv",
            value=report.excitatory_synapse.peak_delta_mv,
            unit="mV",
            target=None,
            passed=report.excitatory_synapse.peak_delta_mv > 0.1,
            note="Excitatory synapse should depolarize the postsynaptic cell.",
        ),
        ValidationCheck(
            name="inhibitory_peak_delta_mv",
            value=report.inhibitory_synapse.peak_delta_mv,
            unit="mV",
            target=None,
            passed=report.inhibitory_synapse.peak_delta_mv < -0.1,
            note="Inhibitory synapse should hyperpolarize the postsynaptic cell.",
        ),
        ValidationCheck(
            name="dopamine_gated_ltp",
            value=report.plasticity.pre_before_post_da_delta,
            unit="syn_strength",
            target=None,
            passed=report.plasticity.pre_before_post_da_delta > 0.25,
            note="Rewarded pre-before-post pairing should strengthen the synapse.",
        ),
        ValidationCheck(
            name="unrewarded_pairing_stability",
            value=report.plasticity.pre_before_post_no_da_delta,
            unit="syn_strength",
            target=None,
            passed=abs(report.plasticity.pre_before_post_no_da_delta) < 0.1,
            note="Pre-before-post pairing without dopamine should remain near baseline.",
        ),
        ValidationCheck(
            name="reverse_pairing_weaker_than_rewarded",
            value=report.plasticity.post_before_pre_da_delta,
            unit="syn_strength",
            target=None,
            passed=(
                report.plasticity.post_before_pre_da_delta
                < report.plasticity.pre_before_post_da_delta
            ),
            note="Rewarded post-before-pre should not outperform rewarded pre-before-post.",
        ),
    ]
    return checks


def run_validation_suite(device: str = "auto") -> ValidationReport:
    """Run the full validation suite on the selected device."""
    report = ValidationReport(
        backend=_probe_backend(device),
        current_clamp=measure_current_clamp(device=device),
        excitatory_synapse=measure_synaptic_response(inhibitory=False, device=device),
        inhibitory_synapse=measure_synaptic_response(inhibitory=True, device=device),
        plasticity=measure_dopamine_gated_plasticity(device=device),
    )
    report.checks = _build_validation_checks(report)
    report.reference_profiles = list(REFERENCE_PROFILES.values())
    report.reference_comparisons = {
        profile.key: compare_report_to_profile(report, profile)
        for profile in report.reference_profiles
    }
    report.reference_suggestions = {
        profile.key: generate_calibration_suggestions(report, profile)
        for profile in report.reference_profiles
    }
    return report


def main() -> int:
    """CLI entry point for printing the validation suite as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="auto",
        help="Validation backend/device: auto, metal, cpu, cuda, or mps.",
    )
    args = parser.parse_args()
    payload = run_validation_suite(device=args.device).to_dict()
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
