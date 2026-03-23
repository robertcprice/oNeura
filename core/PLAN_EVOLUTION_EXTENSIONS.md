# Terrarium Evolution Engine Extensions - Implementation Plan

## Overview

4 features extending the Terrarium Evolution Engine based on thorough codebase exploration.

## Current Architecture

- **Library**: `oneuro-metal/src/terrarium_evolve.rs` (4170 lines, all evolution logic)
- **CLI**: `oneuro-metal/src/bin/terrarium_evolve.rs` (840 lines, arg parsing + dispatch)

Core evaluation primitives:
- `run_single_world()` (line 1159): Normal world evaluation → WorldResult
- `run_single_world_stressed()` (line 2928): Stress-injected evaluation → StressTestResult
- `run_single_coevolution_world()` (line 1640): Coev evaluation → CoevWorldResult
- `run_generation()`: Parallel harness for run_single_world()

---

## Feature 4: Extended Telemetry (IMPLEMENT FIRST)

**Why first**: Features 2 and 3 need to emit telemetry. Currently `telemetry_from_result()` only works on `EvolutionResult`.

### Changes to `terrarium_evolve.rs`

1. **Extend GenerationTelemetry struct** (~line 2682):
```rust
pub struct GenerationTelemetry {
    // existing fields...
    pub mode: Option<String>,
    pub multi_objective_fitness: Option<MultiObjectiveFitness>,
    pub stress_metrics: Option<StressTelemetryMetrics>,
}

pub struct StressTelemetryMetrics {
    pub pre_stress_biomass: f32,
    pub min_stress_biomass: f32,
    pub post_recovery_biomass: f32,
}
```

2. **Update telemetry_from_result()** to accept mode name:
```rust
pub fn telemetry_from_result(result: &EvolutionResult, mode: Option<&str>) -> Vec<GenerationTelemetry>
```

3. **Add telemetry_from_coev_result()** for coevolution mode

4. **Add optional stress_metrics to GenerationResult**

### Changes to CLI

- Update all telemetry export calls to pass mode name
- Wire Pareto and Coev telemetry export

### Tests
- `extended_telemetry_has_mode_name`
- `stress_telemetry_has_stress_metrics`
- `coev_telemetry_produces_records`

---

## Feature 2: Stress-Resilient Pareto Front

Combines NSGA-II with stress perturbations. CLI: `--pareto --stress-test`

### New Structures

```rust
pub struct StressedMultiFitness {
    pub pre_stress: MultiObjectiveFitness,
    pub min_stress: MultiObjectiveFitness,
    pub post_recovery: MultiObjectiveFitness,
    pub weighted: MultiObjectiveFitness,  // stress-weighted combination
}
```

### New Functions

1. **run_single_world_stressed_multi()**: Returns StressTestResult + StressedMultiFitness
2. **evolve_pareto_stressed()**: NSGA-II with stress evaluation
3. **pareto_breed_next_gen()**: Shared breeding helper (extract from evolve_pareto)

### CLI Changes
Handle `--pareto --stress-test` combination before individual mode checks

### Tests
- `stress_pareto_front_has_resilient_solutions`
- `pareto_breed_next_gen_preserves_population_size`
- `stressed_multi_fitness_weights_sum_correctly`

---

## Feature 3: Coevolution + Stress

Co-evolve fly brains alongside stress-tested worlds. CLI: `--coevolve --stress-test`

### New Functions

1. **run_single_coevolution_world_stressed()**: Merges coev + stress logic
2. **run_coevolution_generation_stressed()**: Parallel harness
3. **evolve_coevolution_stressed()**: Full evolution loop
4. **coev_breed_next_gen()**: Shared breeding helper

### CLI Changes
Handle `--coevolve --stress-test` combination

### Tests
- `coev_stressed_produces_valid_result`
- `coev_breed_helper_preserves_population`

---

## Feature 1: Convergence Visualization Dashboard (Python)

Location: `experiments/evolve_dashboard.py`

### Input Files (any combination)
- Telemetry JSON (`--telemetry`)
- Landscape JSON (`--landscape`)
- Pareto JSON (`--pareto`)

### Plots

1. **Fitness Convergence Curve**: best/mean/worst over generations
2. **Population Diversity**: diversity over generations
3. **Parameter Evolution Heatmap**: 15 params × generations
4. **Fitness Landscape FDC Scatter**: distance vs fitness
5. **Pareto Front 2D Projections**: pairwise objective scatter
6. **Stress Recovery Timeline** (if stress metrics present)

---

## Implementation Sequence

1. **Feature 4** (Extended Telemetry) - No dependencies
2. **Feature 2** (Stress-Resilient Pareto) - Depends on Feature 4
3. **Feature 3** (Coevolution + Stress) - Depends on Feature 4
4. **Feature 1** (Python Dashboard) - Depends on all for JSON schema

## Refactoring Opportunities

- `pareto_breed_next_gen()`: Shared NSGA-II breeding
- `coev_breed_next_gen()`: Shared coevolution breeding
- `run_generation_stressed()`: Generic parallel stress harness
- `StressTestConfig::from_frames()`: Stress config constructor

## New CLI Flag Combinations

| Flags | Mode | Function |
|-------|------|----------|
| `--pareto --stress-test` | Stress-resilient Pareto | `evolve_pareto_stressed()` |
| `--coevolve --stress-test` | Stress-resilient Coev | `evolve_coevolution_stressed()` |

## Test Runtime

All integration tests use `lite: true`, small populations (2-3), short generations (2), keeping runtime <60s each.
