# Terrarium Evolution Engine

## Overview

The evolution engine discovers optimal terrarium configurations by treating environmental parameters as a genome and running evolutionary optimization across parallel world simulations. Instead of hand-tuning soil pH, temperature, and organism counts, you specify what "good" means (maximize biodiversity, maximize biomass, etc.) and the engine breeds world configurations that produce the desired outcome.

## Architecture

```
WorldGenome (15 parameters)
    │
    ├──▸ build_world() ──▸ TerrariumWorld
    │                           │
    │                           ├──▸ step_frame() × N
    │                           │
    │                           └──▸ snapshot() ──▸ TerrariumWorldSnapshot
    │                                                    │
    └────────────────────────────────────────────────────▸ evaluate_fitness()
                                                              │
                                                              ▼
                                                         f32 (scalar fitness)
```

Each generation:
1. Construct N worlds from N genomes (in parallel threads)
2. Run each world for `frames_per_world` steps, collecting periodic snapshots
3. Evaluate fitness on the final snapshot (+ periodic history for stability)
4. Select parents via tournament selection
5. Breed next generation via crossover + Gaussian mutation (with elitism)
6. Save checkpoint after each generation

## WorldGenome Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `initial_proton_scale` | 0.3–3.0 | 1.0 | Soil pH scaling (higher = more acidic) |
| `soil_temperature_c` | 10–40 | 25.0 | Mean soil temperature (°C) |
| `water_source_count` | 1–6 | 2 | Number of water sources placed |
| `water_volume` | 50–300 | 150.0 | Volume per water source (mL) |
| `initial_moisture_scale` | 0.5–2.0 | 1.0 | Starting soil moisture multiplier |
| `plant_count` | 2–16 | 6 | Initial plant count |
| `fruit_count` | 0–8 | 3 | Initial fruit count |
| `fly_count` | 0–6 | 2 | Initial fly count |
| `microbe_cohort_count` | 0–6 | 0 | Explicit whole-cell microbes (expensive) |
| `respiration_vmax_scale` | 0.3–3.0 | 1.0 | Respiration enzyme rate multiplier |
| `nitrification_vmax_scale` | 0.3–3.0 | 1.0 | Nitrification enzyme rate multiplier |
| `photosynthesis_vmax_scale` | 0.3–3.0 | 1.0 | Photosynthesis enzyme rate multiplier |
| `mineralization_vmax_scale` | 0.3–3.0 | 1.0 | Mineralization enzyme rate multiplier |
| `seed` | u64 | — | RNG seed for world construction |
| `time_warp` | 100–2000 | 900.0 | Simulation speed multiplier |

## Fitness Objectives

| Objective | CLI flag | Formula |
|-----------|----------|---------|
| **MaxBiomass** | `biomass` | plants × canopy + plant_cells × 0.01 |
| **MaxBiodiversity** | `biodiversity` | 1 − Σ(guild_fraction²) (Simpson diversity) |
| **MaxStability** | `stability` | mean(biomass) / σ(biomass) over periodic snapshots |
| **MaxCarbonSequestration** | `carbon` | soil_glucose + plant_cells × 0.005 |
| **MaxFruitProduction** | `fruit` | fruit_count + food_remaining × 2 |
| **MaxMicrobialHealth** | `microbial` | vitality × (1 + Simpson_diversity) + explicit_microbes × 0.1 |
| **Composite** | `composite` | w₁·biomass + w₂·biodiversity + w₃·stability + w₄·nutrient_cycling |

Default composite weights: biomass=0.3, biodiversity=0.3, stability=0.2, nutrient_cycling=0.2.

## Search Strategies

### Evolutionary (default)
Tournament selection → uniform crossover → Gaussian mutation → elitism.

- **Tournament size**: 3 (configurable)
- **Mutation rate**: 0.15 — each gene has 15% chance of Gaussian perturbation (±15% of range)
- **Crossover rate**: 0.7 — 70% chance of recombination vs cloning
- **Elitism**: 2 — top 2 genomes pass to next generation unchanged

### Random
Each generation samples entirely new random genomes. Useful as a baseline.

### Grid
Systematic sweep over 1–2 parameters. Specified as `field_name:min:max:steps`.

Example: `--strategy grid --grid-param soil_temperature_c:15:35:5` sweeps temperature at 15, 20, 25, 30, 35°C.

## CLI Usage

```bash
# Basic evolutionary run
cargo run --release --no-default-features --bin terrarium_evolve -- \
  --fitness composite --generations 10 --population 8 \
  --output results/evolution_run.json \
  --checkpoint results/evolution_checkpoint.json

# Resume interrupted run
cargo run --release --no-default-features --bin terrarium_evolve -- \
  --resume results/evolution_checkpoint.json \
  --output results/evolution_run.json

# Grid search over temperature
cargo run --release --no-default-features --bin terrarium_evolve -- \
  --strategy grid \
  --grid-param soil_temperature_c:15:35:5 \
  --fitness biomass --frames 50 \
  --output results/temperature_sweep.json

# Custom composite weights (favor stability)
cargo run --release --no-default-features --bin terrarium_evolve -- \
  --fitness composite \
  --w-biomass 0.2 --w-biodiversity 0.2 \
  --w-stability 0.4 --w-nutrient 0.2

# Visualize best genome from a completed run
cargo run --release --no-default-features --bin terrarium_viewer -- \
  --replay-result results/evolution_run.json

# Visualize a specific saved genome
cargo run --release --no-default-features --bin terrarium_viewer -- \
  --replay results/best_genome.json
```

## CLI Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--population <N>` | 8 | Worlds per generation |
| `--generations <N>` | 5 | Number of generations |
| `--frames <N>` | 100 | Simulation frames per world |
| `--strategy <S>` | evolutionary | `random`, `evolutionary`, or `grid` |
| `--fitness <S>` | biomass | Objective: `biomass`, `biodiversity`, `stability`, `carbon`, `fruit`, `microbial`, `composite` |
| `--output <PATH>` | — | Write final EvolutionResult JSON |
| `--checkpoint <PATH>` | — | Save checkpoint after each generation |
| `--resume <PATH>` | — | Resume from checkpoint |
| `--threads <N>` | num_cpus | Max parallel threads |
| `--seed <N>` | 42 | Master RNG seed |
| `--mutation-rate <F>` | 0.15 | Per-gene mutation probability |
| `--elitism <N>` | 2 | Elite genomes preserved per generation |
| `--grid-param <SPEC>` | — | Grid param: `field:min:max:steps` (up to 2) |
| `--w-biomass <F>` | 0.3 | Composite weight for biomass |
| `--w-biodiversity <F>` | 0.3 | Composite weight for biodiversity |
| `--w-stability <F>` | 0.2 | Composite weight for stability |
| `--w-nutrient <F>` | 0.2 | Composite weight for nutrient cycling |

## Persistence

### Checkpoint Format
After each generation, an `EvolutionCheckpoint` JSON is saved containing:
- Full config
- All completed generation results
- Current population genomes
- Global best genome and fitness
- RNG state offset for deterministic resume

### Evolution Result Format
Final output is an `EvolutionResult` JSON:
- Per-generation stats (best/mean/worst fitness, best genome, wall time)
- Global best genome and fitness
- Total worlds evaluated and wall time

### Genome Format
A `WorldGenome` is a flat JSON object with all 15 parameters. Can be loaded into `terrarium_viewer --replay` to visually replay the evolved configuration.

## NSGA-II Multi-Objective Optimization

For problems where you want to optimize multiple objectives simultaneously without pre-specifying weights, use NSGA-II Pareto mode. Instead of a single best genome, you get a **Pareto front** — a set of non-dominated solutions where improving one objective necessarily worsens another.

```bash
# Optimize biomass + biodiversity + stability simultaneously
cargo run --release --no-default-features --bin terrarium_evolve -- \
  --pareto --pareto-obj 0,1,2 \
  --generations 10 --population 12 \
  --output results/pareto_front.json

# Optimize all 6 objectives
cargo run --release --no-default-features --bin terrarium_evolve -- \
  --pareto --pareto-obj 0,1,2,3,4,5 \
  --generations 20 --population 16
```

Objective indices: 0=biomass, 1=biodiversity, 2=stability, 3=carbon, 4=fruit, 5=microbial.

The output includes the full Pareto front with fitness vectors, allowing you to pick the solution that best matches your preferences post-hoc.

### Algorithm Details

1. **Non-dominated sorting**: Partitions population into fronts. Front 0 = solutions not dominated by any other. Front 1 = dominated only by front 0, etc.
2. **Crowding distance**: Within each front, measures how spread out solutions are in objective space. Boundary solutions get infinite distance. Preserves diversity.
3. **Selection**: Binary tournament using (rank, crowding distance) — lower rank wins, ties broken by higher crowding distance.
4. **Crossover + mutation**: Same operators as single-objective evolutionary mode.

## Fly Brain Coevolution

The `CoevolutionGenome` jointly evolves world parameters alongside fly brain parameters:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `olfactory_gain` | 0.1–5.0 | 1.0 | Odor gradient sensitivity multiplier |
| `exploration_rate` | 0.01–0.5 | 0.15 | Random flight probability |
| `reward_learning_rate` | 0.001–0.1 | 0.01 | Hunger→reward modulation speed |
| `hunger_threshold` | 0.1–0.8 | 0.3 | Energy level triggering hunger signal |
| `flight_duration_scale` | 0.3–3.0 | 1.0 | Flight bout length multiplier |

These parameters map directly to the fly brain's neural circuitry:

| Parameter | Neural Target | Effect |
|-----------|--------------|--------|
| `olfactory_gain` | AL (Antennal Lobe) stimulation current | Higher = stronger odor response, better food tracking |
| `exploration_rate` | Stochastic takeoff probability | Higher = more random flights, better area coverage |
| `reward_learning_rate` | DAN (Dopaminergic) amplitude | Higher = stronger reward/punishment signals |
| `hunger_threshold` | SEZ/MB hunger signal threshold | Higher = earlier hunger onset, more feeding motivation |
| `flight_duration_scale` | Takeoff/landing sigmoid thresholds | Higher = easier takeoff, harder landing = longer flights |

Coevolution discovers environment-behavior pairs — e.g., a terrarium with abundant fruit may evolve flies with low exploration (since food is easy to find), while a sparse terrarium may evolve flies with high olfactory gain and longer flights.

```bash
# Run coevolution
cargo run --release --no-default-features --bin terrarium_evolve -- \
  --coevolve --fitness composite --generations 10 --population 8 \
  --output results/coevolution_run.json
```

## Performance Notes

- Each world takes ~30–60s to construct + run 100 frames (CPU substrate, release build, no flies)
- Worlds WITH flies take significantly longer due to neural simulation (~5–50min depending on fly count and scale)
- For evolutionary sweeps, set `fly_count=0` unless specifically studying fly behavior
- A generation of 8 worlds on an 8-core machine takes ~60–120s (limited by world construction, not stepping)
- Whole-cell microbes (`microbe_cohort_count > 0`) add ~10s per microbe per world — default to 0 for sweeps
- Checkpoint save is O(1) — just JSON serialization of current state
- LTO release build (`--release`) is required — debug builds are 50x slower
- Test binary linking may OOM under LTO; use dev mode for unit tests: `cargo test --no-default-features --lib`

## Tests

| Test | Time | Description |
|------|------|-------------|
| `genome_serialization_roundtrip` | instant | JSON serialize/deserialize genome |
| `checkpoint_save_and_load` | instant | Checkpoint persistence roundtrip |
| `mutation_preserves_bounds` | instant | 50 mutations all stay in range |
| `world_genome_random_stays_in_bounds` | instant | 20 random genomes all in range |
| `composite_fitness_returns_finite` | instant | All fitness objectives produce finite values |
| `snapshot_serialization_roundtrip` | instant | Snapshot JSON roundtrip via `#[serde(default)]` |
| `non_dominated_sort_basic` | instant | 3 non-dominated solutions → single front |
| `non_dominated_sort_with_dominated` | instant | Trade-off front vs dominated solution |
| `crowding_distance_boundary_infinite` | instant | Boundary solutions get infinite distance |
| `nsga2_select_produces_valid_indices` | instant | Selection returns valid population indices |
| `multi_objective_fitness_all_finite` | instant | All 6 objectives produce finite values |
| `fly_brain_genome_random_in_bounds` | instant | 20 random brain genomes in range |
| `fly_brain_mutation_preserves_bounds` | instant | 50 brain mutations all stay in range |
| `coevolution_genome_roundtrip` | instant | Joint genome JSON serialize/deserialize |
| `coevolution_crossover_and_mutate` | instant | Joint crossover + mutation preserves bounds |
| `fitness_evaluation_returns_finite` | ~5min | Build world, run 10 frames, evaluate all objectives |
| `crossover_produces_valid_genome` | ~30s | Crossover child builds and runs |
| `parallel_worlds_produce_different_results` | ~2min | 3 parallel worlds with different seeds |
| `evolution_improves_over_generations` | ~4min | 2-gen evolution, elitism prevents regression |
| `evolution_with_checkpoint_saves_file` | ~4min | Full checkpoint persistence (ignored in CI) |
| `grid_strategy_covers_parameter_space` | ~2min | Grid search produces correct number of worlds |

Run all 15 instant tests (dev mode, avoids LTO OOM):
```bash
cargo test --no-default-features --lib -- \
  terrarium_evolve::tests::genome_serialization \
  terrarium_evolve::tests::checkpoint_save \
  terrarium_evolve::tests::composite_fitness \
  terrarium_evolve::tests::snapshot_serialization \
  terrarium_evolve::tests::mutation_preserves \
  terrarium_evolve::tests::world_genome_random \
  terrarium_evolve::tests::non_dominated_sort \
  terrarium_evolve::tests::crowding \
  terrarium_evolve::tests::nsga2 \
  terrarium_evolve::tests::multi_objective \
  terrarium_evolve::tests::fly_brain \
  terrarium_evolve::tests::coevolution
```
