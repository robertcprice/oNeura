# Tutorial 7: Consciousness Theories and Tests

## Overview

The Organic Neural framework includes a comprehensive quantum consciousness module that integrates multiple leading theories: Integrated Information Theory (IIT), Global Workspace Theory (GWT), and Orchestrated Objective Reduction (Orch-OR). This tutorial explores how to measure and test consciousness-like properties in neural systems.

This tutorial covers:
- Integrated Information Theory (IIT) and Phi calculation
- Global Workspace Theory (GWT) and broadcasting
- Orch-OR quantum consciousness
- Self-model and mirror tests
- Composite consciousness metrics

## Neuroscience Background

### Integrated Information Theory (IIT)

Proposed by Giulio Tononi, IIT identifies consciousness with integrated information (Phi). A system is conscious to the extent that:

1. It has many possible states (information)
2. Those states cannot be reduced to independent parts (integration)

Phi measures the "information that is irreducible" - how much the whole exceeds the sum of its parts.

### Global Workspace Theory (GWT)

Proposed by Bernard Baars, GWT views consciousness as a "global workspace" where:

1. Specialized modules process information in parallel (unconscious)
2. Information that gains access to the workspace becomes globally available (conscious)
3. This broadcast enables flexible, coordinated behavior

### Orchestrated Objective Reduction (Orch-OR)

Proposed by Penrose and Hameroff, Orch-OR suggests:

1. Microtubules in neurons support quantum computation
2. Quantum superposition states have gravitational self-energy
3. When this energy exceeds a threshold, objective reduction occurs
4. Each reduction event is a moment of "proto-consciousness"

## The Quantum Consciousness Module

### Basic Setup

```python
from quantum_consciousness import (
    ConsciousnessSystem,
    ConsciousnessMeasure,
    OrchORSimulator,
    IITSystem,
    GlobalWorkspace,
    SelfModel
)

# Create unified consciousness system
consciousness = ConsciousnessSystem(
    num_units=8,        # For IIT calculation
    num_tubulins=8,     # For Orch-OR
    workspace_capacity=4  # For GWT
)

print("Consciousness system initialized")
```

### Running Simulation

```python
# Run simulation step
metrics = consciousness.step()

print(f"Phi (IIT): {metrics.phi:.3f}")
print(f"Coherence (Orch-OR): {metrics.coherence:.3f}")
print(f"Workspace Occupancy (GWT): {metrics.workspace_occupancy:.3f}")
print(f"Composite Score: {metrics.composite_score():.3f}")
```

## Integrated Information Theory (IIT)

### Phi Calculation

```python
# Access IIT system directly
iit = consciousness.iit_system

# Get current Phi
phi = iit.phi()
print(f"Phi = {phi:.3f}")

# Explore partitions
partitions = iit.all_partitions()
print(f"Number of bipartitions: {len(partitions)}")

# For each partition, we compute effective information
for part_a, part_b in partitions[:3]:
    ei = iit.effective_information((part_a, part_b))
    print(f"  Partition {part_a} | {part_b}: EI = {ei:.3f}")
```

### Understanding Phi

Phi is the minimum effective information across all partitions:

```
Phi = min(EI) over all possible partitions

High Phi = Information is irreducible (conscious)
Low Phi = Information decomposes easily (unconscious)
```

```python
def analyze_phi_components(iit_system):
    """Analyze which partitions contribute most to Phi."""

    partitions = iit_system.all_partitions()
    ei_values = []

    for part_a, part_b in partitions:
        ei = iit_system.effective_information((part_a, part_b))
        ei_values.append((part_a, part_b, ei))

    # Sort by EI
    ei_values.sort(key=lambda x: x[2])

    print("Minimum Information Bipartitions (limiting Phi):")
    for part_a, part_b, ei in ei_values[:5]:
        print(f"  {part_a} | {part_b}: EI = {ei:.3f}")

    print("\nMaximum Information Bipartitions:")
    for part_a, part_b, ei in ei_values[-3:]:
        print(f"  {part_a} | {part_b}: EI = {ei:.3f}")

    return ei_values

ei_values = analyze_phi_components(consciousness.iit_system)
```

## Global Workspace Theory (GWT)

### Workspace Dynamics

```python
# Access workspace
workspace = consciousness.workspace

# Inject stimuli into different modules
from quantum_consciousness import ModuleType

workspace.inject_input(ModuleType.VISUAL, "bright_light", strength=0.9)
workspace.inject_input(ModuleType.AUDITORY, "loud_sound", strength=0.7)

# Process cycle
summary = workspace.process_cycle()

print("Workspace Cycle Summary:")
print(f"  Winners: {summary['winners']}")
print(f"  Focus: {summary['focus']}")
print(f"  Broadcast content: {summary['broadcast_content']}")
```

### Module Competition

```python
def demonstrate_workspace_competition(workspace):
    """Show how modules compete for workspace access."""

    # Inject competing stimuli
    workspace.inject_input(ModuleType.VISUAL, "face", strength=0.8)
    workspace.inject_input(ModuleType.AUDITORY, "name", strength=0.6)
    workspace.inject_input(ModuleType.MEMORY, "association", strength=0.7)

    # Multiple cycles
    for cycle in range(10):
        summary = workspace.process_cycle()

        if cycle % 3 == 0:
            print(f"\nCycle {cycle}:")
            print(f"  Current focus: {summary['focus']}")
            print(f"  Workspace items: {len(summary['winners'])}")

            # Show module activations
            for mt, module in workspace.modules.items():
                print(f"    {mt.name}: {module.activation:.3f}")

demonstrate_workspace_competition(consciousness.workspace)
```

### Workspace Occupancy

```python
# Measure workspace occupancy over time
occupancy_history = []

for _ in range(100):
    consciousness.step()
    occupancy = consciousness.workspace.workspace_occupancy()
    occupancy_history.append(occupancy)

avg_occupancy = sum(occupancy_history) / len(occupancy_history)
print(f"Average workspace occupancy: {avg_occupancy:.3f}")
```

## Orch-OR Quantum Consciousness

### Microtubule Quantum States

```python
# Access Orch-OR simulator
orch_or = consciousness.orch_or

# Initialize superposition
orch_or.initialize_superposition()

# Check tubulin states
for i, tubulin in enumerate(orch_or.microtubule.tubulins[:4]):
    print(f"Tubulin {i}:")
    print(f"  State: {tubulin.state.name}")
    print(f"  Alpha prob: {tubulin.alpha_prob:.3f}")
    print(f"  Beta prob: {tubulin.beta_prob:.3f}")
    print(f"  Coherence: {tubulin.coherence:.3f}")
```

### Reduction Events

```python
# Run and track reduction events
consciousness.orch_or.initialize_superposition()

reduction_times = []
for step in range(200):
    metrics = consciousness.step()

    # Check for reduction
    if consciousness.orch_or.reduction_history:
        last_reduction = consciousness.orch_or.reduction_history[-1]
        if last_reduction.time_ns not in reduction_times:
            reduction_times.append(last_reduction.time_ns)
            print(f"Reduction at t={last_reduction.time_ns:.1f}ns")
            print(f"  Tubulins involved: {last_reduction.num_tubulins_involved}")
            print(f"  Gravitational energy: {last_reduction.energy_difference:.2e} J")
            print(f"  Phi at collapse: {last_reduction.integrated_information:.3f}")

print(f"\nTotal reductions: {len(consciousness.orch_or.reduction_history)}")
```

### Anesthetic Suppression

```python
def demonstrate_anesthetic_effect(consciousness):
    """Show how anesthetic suppresses consciousness measures."""

    # Baseline
    baseline_scores = []
    for _ in range(50):
        metrics = consciousness.step()
        baseline_scores.append(metrics.composite_score())

    print(f"Baseline composite: {sum(baseline_scores)/len(baseline_scores):.3f}")

    # Apply anesthetic
    consciousness.apply_anesthetic(0.9)  # 90% concentration

    anesthetic_scores = []
    for _ in range(50):
        metrics = consciousness.step()
        anesthetic_scores.append(metrics.composite_score())

    print(f"Under anesthetic: {sum(anesthetic_scores)/len(anesthetic_scores):.3f}")

    # Recovery
    consciousness.apply_anesthetic(0.0)

    recovery_scores = []
    for _ in range(50):
        metrics = consciousness.step()
        recovery_scores.append(metrics.composite_score())

    print(f"After recovery: {sum(recovery_scores)/len(recovery_scores):.3f}")

    ratio = (sum(baseline_scores)/len(baseline_scores)) / \
           max(0.001, sum(anesthetic_scores)/len(anesthetic_scores))
    print(f"Conscious/Unconscious ratio: {ratio:.2f}x")

demonstrate_anesthetic_effect(consciousness)
```

## Self-Model and Mirror Test

### Self-Model Structure

```python
# Access self-model
self_model = consciousness.self_model

print("Initial self-state:")
for key, value in self_model.self_state.items():
    print(f"  {key}: {value:.3f}")
```

Self-state dimensions:
- **Agency**: Sense of being cause of actions
- **Continuity**: Sense of persistence over time
- **Distinctness**: Sense of being separate from environment
- **Ownership**: Sense of owning body/thoughts
- **Perspective**: Sense of first-person viewpoint

### Mirror Self-Recognition Test

```python
def run_mirror_test(consciousness, n_exposures=10):
    """
    Run mirror self-recognition test.

    Animals that pass recognize themselves in mirrors,
    indicating self-awareness.
    """
    self_model = consciousness.self_model

    recognition_scores = []

    for exposure in range(n_exposures):
        # Simulate seeing reflection
        reflection = {
            'agency': self_model.self_state.get('agency', 0.5) + 0.05 * random.random(),
            'continuity': self_model.self_state.get('continuity', 0.5),
            'distinctness': self_model.self_state.get('distinctness', 0.5),
        }

        # Test recognition
        recognition = self_model.mirror_test(reflection, is_self=True)
        recognition_scores.append(recognition)

        # Evolve system
        consciousness.step()

        if (exposure + 1) % 3 == 0:
            print(f"Exposure {exposure + 1}: recognition = {recognition:.3f}, "
                  f"self-distinctness = {self_model.self_state.get('distinctness', 0):.3f}")

    final_score = self_model.self_recognition_score
    print(f"\nFinal self-recognition score: {final_score:.3f}")

    return recognition_scores, final_score

scores, final = run_mirror_test(consciousness)
```

### Self-Other Distinction

```python
def test_self_other_distinction(consciousness):
    """Test ability to distinguish self from other."""

    self_model = consciousness.self_model

    # Simulate observing another agent
    self_model.observe_other("other_agent", {
        'agency': 0.3,  # Different from self
        'continuity': 0.5,
        'distinctness': 0.8,
    })

    # Update self-state
    self_model.update_self_state({
        'agency': 0.7,
        'distinctness': 0.6,
    })

    # Compute distinction
    distinction = self_model.self_other_distinction()
    print(f"Self-other distinction: {distinction:.3f}")

    # Higher values = better self-awareness
    if distinction > 0.5:
        print("Good self-other differentiation")
    else:
        print("Poor self-other differentiation")

    return distinction

distinction = test_self_other_distinction(consciousness)
```

### Introspection

```python
def demonstrate_introspection(consciousness):
    """Show introspective capabilities."""

    # Run some experience
    for _ in range(50):
        consciousness.step()

    # Introspect
    intro = consciousness.self_model.introspect(depth=2)

    print("Introspection Results:")
    print(f"  Self-state: {intro['self_state']}")
    print(f"  Confidence: {intro['confidence']:.3f}")
    print(f"  Recognition ability: {intro['recognition_ability']:.3f}")
    print(f"  Narrative coherence: {intro['narrative_coherence']:.3f}")

    if 'meta' in intro:
        print(f"  Introspection depth: {intro['meta']['introspection_depth']}")
        print(f"  Certainty about self: {intro['meta']['certainty_about_self']:.3f}")

demonstrate_introspection(consciousness)
```

## Composite Consciousness Metrics

### Understanding the Composite Score

```python
def explain_composite_score():
    """Explain how the composite score is computed."""

    print("Composite Consciousness Score Components:")
    print("=" * 50)

    weights = {
        'coherence': 0.15,
        'entanglement': 0.10,
        'phi': 0.20,
        'workspace_occupancy': 0.15,
        'self_complexity': 0.15,
        'orchestration_level': 0.10,
        'attention_focus': 0.10,
        'self_other_distinction': 0.05,
    }

    for metric, weight in weights.items():
        print(f"  {metric}: weight = {weight}")

    print("\nFormula:")
    print("  composite = sum(metric * weight) * (1 - anesthetic)")

    print("\nInterpretation:")
    print("  > 0.7: High consciousness")
    print("  0.4-0.7: Moderate consciousness")
    print("  < 0.4: Low consciousness")

explain_composite_score()
```

### Tracking Consciousness Over Time

```python
def track_consciousness_evolution(consciousness, n_steps=100):
    """Track how consciousness measures evolve."""

    history = {
        'composite': [],
        'phi': [],
        'coherence': [],
        'workspace': [],
        'self_complexity': [],
    }

    for _ in range(n_steps):
        metrics = consciousness.step()

        history['composite'].append(metrics.composite_score())
        history['phi'].append(metrics.phi)
        history['coherence'].append(metrics.coherence)
        history['workspace'].append(metrics.workspace_occupancy)
        history['self_complexity'].append(metrics.self_complexity)

    # Print summary
    print("Consciousness Evolution Summary:")
    print("=" * 40)

    for metric, values in history.items():
        avg = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        print(f"{metric}:")
        print(f"  Average: {avg:.3f}")
        print(f"  Range: {min_val:.3f} - {max_val:.3f}")

    return history

history = track_consciousness_evolution(consciousness)
```

## Complete Example: Full Consciousness Report

```python
from quantum_consciousness import ConsciousnessSystem, ModuleType

def generate_consciousness_report(seed=42):
    """Generate a comprehensive consciousness report."""

    print("=" * 60)
    print("CONSCIOUSNESS SYSTEM REPORT")
    print("=" * 60)

    # Initialize
    system = ConsciousnessSystem(num_units=6, num_tubulins=6, seed=seed)
    system.orch_or.initialize_superposition()

    # Phase 1: Baseline
    print("\n[Phase 1] Baseline Consciousness (50 steps)")
    baseline = []
    for _ in range(50):
        metrics = system.step()
        baseline.append(metrics.composite_score())
    print(f"  Average: {sum(baseline)/len(baseline):.3f}")

    # Phase 2: Attention stimulus
    print("\n[Phase 2] Attention Stimulus")
    system.inject_stimulus(ModuleType.VISUAL, "attention_test", strength=0.9)
    system.inject_stimulus(ModuleType.SELF_MODEL, "self_reflection", strength=0.8)

    stimulus = []
    for _ in range(30):
        metrics = system.step()
        stimulus.append(metrics.composite_score())
    print(f"  Average with stimulus: {sum(stimulus)/len(stimulus):.3f}")

    # Phase 3: Anesthetic (unconscious)
    print("\n[Phase 3] Anesthetic State")
    system.apply_anesthetic(0.9)

    unconscious = []
    for _ in range(30):
        metrics = system.step()
        unconscious.append(metrics.composite_score())
    print(f"  Average under anesthetic: {sum(unconscious)/len(unconscious):.3f}")

    # Phase 4: Recovery
    print("\n[Phase 4] Recovery")
    system.apply_anesthetic(0.0)

    recovery = []
    for _ in range(30):
        metrics = system.step()
        recovery.append(metrics.composite_score())
    print(f"  Average after recovery: {sum(recovery)/len(recovery):.3f}")

    # Phase 5: Mirror test
    print("\n[Phase 5] Mirror Self-Recognition")
    for i in range(5):
        reflection = {
            'agency': system.self_model.self_state.get('agency', 0.5),
            'continuity': system.self_model.self_state.get('continuity', 0.5),
            'distinctness': system.self_model.self_state.get('distinctness', 0.5),
        }
        recognition = system.self_model.mirror_test(reflection, is_self=True)
        system.step()

    print(f"  Self-recognition score: {system.self_model.self_recognition_score:.3f}")

    # Final report
    report = system.get_consciousness_report()
    print("\n" + "=" * 60)
    print("FINAL CONSCIOUSNESS REPORT")
    print("=" * 60)

    print(f"\nTime step: {report['time_step']}")
    print(f"Composite score: {report['composite_score']:.3f}")

    print("\nComponent Metrics:")
    for metric, value in report['metrics'].items():
        print(f"  {metric}: {value:.3f}")

    print(f"\nQuantum reduction events: {report['reductions']}")
    print(f"Workspace focus: {report['workspace_focus']}")

    print("\nSelf-Model:")
    intro = report['self_model']
    print(f"  Self-state: {intro['self_state']}")
    print(f"  Confidence: {intro['confidence']:.3f}")
    print(f"  Narrative coherence: {intro['narrative_coherence']:.3f}")

    # Summary statistics
    baseline_avg = sum(baseline) / len(baseline)
    unconscious_avg = sum(unconscious) / len(unconscious)
    ratio = baseline_avg / max(0.001, unconscious_avg)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Conscious/Unconscious ratio: {ratio:.2f}x")
    print(f"Self-recognition: {system.self_model.self_recognition_score:.3f}")
    print(f"Self-complexity: {system.self_model.complexity():.3f}")

    return system, report

# Generate report
system, report = generate_consciousness_report()
```

## References

- Tononi, G. (2004). "An Information Integration Theory of Consciousness" - BMC Neuroscience
- Baars, B. J. (1988). "A Cognitive Theory of Consciousness" - Cambridge University Press
- Penrose, R. & Hameroff, S. (2014). "Consciousness in the Universe" - Physics of Life Reviews
- Gallup, G. G. (1970). "Chimpanzees: Self-recognition" - Science

## Summary

In this tutorial, you learned:

1. **IIT**: How integrated information (Phi) measures irreducibility
2. **GWT**: How the global workspace enables broadcasting
3. **Orch-OR**: How quantum effects in microtubules might relate to consciousness
4. **Self-Model**: How systems develop self-representation
5. **Mirror Test**: Testing self-recognition capability
6. **Composite Metrics**: Combining multiple theories into unified scores

The key insight is that **consciousness is multi-faceted** - no single measure captures it fully. By integrating IIT (information structure), GWT (global availability), and Orch-OR (quantum effects), we get a richer picture of what consciousness might be and how to measure it in artificial systems.
