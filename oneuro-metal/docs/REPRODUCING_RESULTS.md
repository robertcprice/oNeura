# Reproducing Results from the oNeura Methods Paper

> **Paper**: "oNeura Terrarium: A Multi-Scale Biological Simulation Framework
> Bridging Quantum Chemistry to Evolutionary Ecology"
>
> **Paper file**: `docs/METHODS_MULTISCALE_PAPER.md`
>
> **Last verified**: 2026-03-17 — all commands produce expected output

---

## 1. Prerequisites

### Hardware
- Any machine with 8+ GB RAM (16 GB recommended)
- macOS with Apple Silicon for GPU acceleration (optional — CPU fallback works everywhere)
- ~2 GB disk space for build artifacts

### Software
- **Rust 1.70+** — install via [rustup.rs](https://rustup.rs/)
- **Git** — for cloning the repository
- **Python 3.10+** (optional) — for generating publication figures

### Clone and Build

```bash
git clone https://github.com/robertcprice/oNeura.git
cd oNeura/oneuro-metal

# Build all binaries (optimized, 16GB Mac safe)
CARGO_BUILD_JOBS=1 cargo build --profile fast --no-default-features

# Or build specific binaries
cargo build --profile fast --no-default-features \
  --bin terrarium_evolve --bin drug_optimizer --bin gene_circuit
```

### Verify Build

```bash
# Should complete with 0 errors
cargo check --no-default-features --lib
```

---

## 2. Full Test Suite Verification

### Regression Tests (216 tests, ~32 seconds)

```bash
cargo test --no-default-features --lib -- \
  substrate_stays_bounded guild_activity soil_atmosphere terrarium_evolve \
  drosophila_population plant_competition soil_fauna fly_metabolism \
  field_coupling seed_cellular terrarium_world organism_metabolism \
  stochastic cross_scale phenotypic persister bet_hedging benchmark \
  seasonal drought tropical arid spatial zone plant_noise microbial_noise \
  multi_species single_drug pulsed combination protocol ecoli circuit \
  enzyme_ bioremediation probe_coupling probe_snapshot drug_enzyme \
  soil_enzyme temperature_coupling remediation guild_latent
```

**Expected**: `test result: ok. 216 passed; 0 failed; 0 ignored`

### Full Test Suite (825+ tests, ~2-3 minutes)

```bash
cargo test --no-default-features --lib
```

**Expected**: 825+ passed, 0 failed, 6 ignored (quantum tests requiring specific hardware configuration)

---

## 3. Reproducing Paper Results

### 3.1 Evolutionary Fitness Convergence (Paper Section 8.1)

**Claim**: NSGA-II evolution drives fitness from random initialization (~62) to
optimized ecosystem (~85) over 5 generations.

```bash
./target/fast/terrarium_evolve \
  --population 8 --generations 5 --frames 100 \
  --fitness biomass --lite --seed 42
```

**Expected output** (approximate):
```
Generation 0: best fitness ~60-65 (random genomes)
Generation 1: best fitness ~68-72
Generation 2: best fitness ~74-78
Generation 3: best fitness ~78-82
Generation 4: best fitness ~82-86
```

**Verification**: Fitness should monotonically increase. Final best > initial best by >20%.

### 3.2 Pareto Front — Multi-Objective Tradeoffs (Paper Section 8.2)

**Claim**: Genuine Pareto tradeoffs exist between biomass, diversity, and stress resilience.

```bash
./target/fast/terrarium_evolve \
  --pareto --population 8 --generations 5 --frames 50 \
  --lite --seed 42
```

**Expected**: 7-objective Pareto front. No single genome dominates all objectives simultaneously. Tradeoff between biomass maximization and species diversity is visible.

### 3.3 Stress Resilience Under Environmental Shocks (Paper Section 8.3)

**Claim**: Evolved ecosystems show >40% higher survival under drought + heat stress
compared to random configurations.

```bash
./target/fast/terrarium_evolve \
  --pareto --stress-test --population 8 --generations 5 --frames 50 \
  --lite --seed 42
```

**Expected**: Stress-tested genomes develop dormancy strategies. Fitness under stress
is lower than unstressed, but evolved genomes significantly outperform random ones.

### 3.4 Michaelis-Menten Steady States (Paper Section 5.5)

**Claim**: 7-pool fly metabolism reaches correct MM steady states.

```bash
cargo test --no-default-features --lib -- fly_metabolism --nocapture
```

**Expected**: 10 tests pass. Tests verify:
- Crop → hemolymph trehalose conversion follows MM kinetics
- Muscle ATP production matches expected V_max and K_m
- Energy conservation: total metabolite mass is preserved
- Starvation cascade: when food = 0, ATP depletes within expected timeframe

### 3.5 Sharpe-Schoolfield Thermal Response (Paper Section 5.6)

**Claim**: Development rate follows Sharpe-Schoolfield curve with thermal optimum
and high-temperature inactivation.

```bash
cargo test --no-default-features --lib -- drosophila_population --nocapture
```

**Expected**: 7 tests pass. Tests verify:
- Development accelerates with temperature up to thermal optimum (~25°C)
- High-temperature inactivation reduces rate above ~32°C
- Egg → larva → pupa → adult progression timing matches Drosophila literature
- Population grows when food is available, declines under starvation

### 3.6 Beer-Lambert Light Competition (Paper Section 5.7)

**Claim**: Taller plants shade shorter ones following Beer-Lambert law, producing
asymmetric competitive outcomes.

```bash
cargo test --no-default-features --lib -- plant_competition --nocapture
```

**Expected**: 10 tests pass. Tests verify:
- Light attenuation follows I = I₀ × exp(-k × LAI)
- Taller plants receive more light than shorter competitors
- Root nutrient splitting scales with root biomass fraction
- Competitive exclusion occurs under extreme asymmetry

### 3.7 Stochastic Gene Expression / Fano Factors (Paper Section 5.4)

**Claim**: Gillespie tau-leaping produces expression noise matching
Taniguchi et al. 2010 single-cell measurements.

```bash
cargo test --no-default-features --lib -- stochastic --nocapture
```

**Expected**: 10 tests pass. Tests verify:
- Fano factor (variance/mean) > 1 for bursty promoters
- Mean expression level converges to analytical prediction
- Telegraph model: promoter switching between ON/OFF states
- Tau-leaping matches exact Gillespie within statistical tolerance

### 3.8 TIP3P Water Structure (Paper Section 4)

**Claim**: Molecular dynamics with TIP3P water model maintains stable O-H bond
length of 0.9572 Å at 1 fs timestep.

```bash
cargo test --no-default-features --lib -- molecular_dynamics --nocapture
```

**Expected**: Tests verify O-H bond stability, H-O-H angle preservation,
and energy conservation over 1000+ timesteps.

### 3.9 Soil Fauna Dynamics (Paper Section 5.8)

**Claim**: Earthworm bioturbation enhances nitrogen mineralization;
nematode predation follows Lotka-Volterra dynamics.

```bash
cargo test --no-default-features --lib -- soil_fauna --nocapture
```

**Expected**: 8 tests pass. Tests verify:
- Earthworm populations bounded and positive
- Nematode-bacteria predator-prey oscillations
- N mineralization increases with earthworm density
- Guild biomass remains within realistic ranges

### 3.10 Drug Protocol Optimization (Paper Section 10)

**Claim**: Pulsed antibiotic protocols outperform continuous dosing for persister
cell elimination, producing biphasic kill curves.

```bash
# Compare treatment protocols
./target/fast/drug_optimizer --mode compare

# Validate against E. coli literature data
./target/fast/drug_optimizer --mode validate

# Evolutionary protocol optimization
./target/fast/drug_optimizer --mode optimize

# Parameter sensitivity scan
./target/fast/drug_optimizer --mode scan
```

**Expected**:
- `compare`: Shows single < pulsed < combination effectiveness
- `validate`: Kill curve matches E. coli biphasic dynamics (fast initial kill, slow persister phase)
- `optimize`: Evolutionary search finds protocols with >99% kill efficiency
- `scan`: Sensitivity heatmap shows dose × interval landscape

### 3.11 Gene Circuit Noise Design (Paper Section 11)

**Claim**: Telegraph model circuit designer achieves target Fano factor
and mean expression level.

```bash
./target/fast/gene_circuit --target-fano 5.0 --target-mean 100.0
```

**Expected**: Designed circuit parameters (k_on, k_off, transcription rate, degradation rate)
that produce:
- Fano factor ≈ 5.0 (within 10% tolerance)
- Mean expression ≈ 100 proteins (within 10% tolerance)

### 3.12 Cross-Scale Coupling (Paper Section 7)

**Claim**: Quantum rate corrections propagate to macroscopic metabolism through
the Arrhenius/Eyring bridge.

```bash
cargo test --no-default-features --lib -- cross_scale --nocapture
```

**Expected**: 6 tests pass verifying bidirectional coupling between scales.

### 3.13 Quantum Chemistry (Paper Section 3)

**Claim**: Hartree-Fock SCF solver converges for small molecules;
Eyring TST produces correct rate constants.

```bash
cargo test --no-default-features --lib -- subatomic quantum --nocapture
```

**Expected**: 64 tests pass, 6 ignored. Tests cover:
- SCF convergence for H₂, HeH⁺
- Eyring rate constant calculation
- Basis set integral evaluation
- Fock matrix construction

### 3.14 Ecosystem Integration Scenarios (Paper Results)

```bash
cargo test --no-default-features --lib -- ecosystem_integration --nocapture
```

**Expected**: 15 tests pass covering AMR evolution, soil nutrient cycling,
eco-evolutionary feedback, and cross-module integration.

### 3.15 Environment-Specific Evolution (Paper Section 8.3)

**Claim**: Arid environments produce harder evolutionary challenges than temperate.

```bash
cargo test --no-default-features --lib -- evolve_with_environment --nocapture
```

**Expected**: Tests confirm that arid fitness < temperate fitness for equivalent genomes.

---

## 4. Telemetry Export for Figures

### CSV Export

The evolution engine can export telemetry data suitable for plotting:

```rust
use oneuro_metal::{telemetry_to_csv, TerrariumWorldSnapshot};

let snapshots: Vec<TerrariumWorldSnapshot> = /* from simulation */;
let csv = telemetry_to_csv(&snapshots);
std::fs::write("telemetry.csv", csv).unwrap();
```

### Prometheus Metrics

```rust
use oneuro_metal::telemetry_to_prometheus;

let snapshots: Vec<TerrariumWorldSnapshot> = /* from simulation */;
let prom = telemetry_to_prometheus(&snapshots);
// Push to Prometheus or write to file
```

### JSON Snapshots via REST API

```bash
# Start the web server
./target/fast/terrarium_web

# Fetch current snapshot
curl http://localhost:3000/api/snapshot | python3 -m json.tool > snapshot.json

# Stream via WebSocket
websocat ws://localhost:3000/ws/stream
```

---

## 5. Generating Publication Figures

### Suggested Figure Scripts (Python/matplotlib)

**Figure 1: Fitness convergence over generations**
```python
import matplotlib.pyplot as plt
import json

# Load telemetry from evolution run
# ./target/fast/terrarium_evolve --pareto --population 8 --generations 10 --frames 100 --lite --seed 42 > evolution.json

data = json.load(open("evolution.json"))
generations = range(len(data["best_fitness"]))
plt.plot(generations, data["best_fitness"])
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("NSGA-II Fitness Convergence")
plt.savefig("fig1_fitness_convergence.pdf")
```

**Figure 2: Pareto front (biomass vs diversity)**
```python
# Extract from pareto run output
plt.scatter(biomass_values, diversity_values, c=stress_values, cmap='viridis')
plt.xlabel("Biomass")
plt.ylabel("Species Diversity")
plt.colorbar(label="Stress Resilience")
plt.title("7-Objective Pareto Front (2D Projection)")
plt.savefig("fig2_pareto_front.pdf")
```

**Figure 3: Arrhenius plot (ln(k) vs 1/T)**
```python
# From quantum_runtime test output
temps = [273, 283, 293, 303, 313, 323]
rates = [/* from test output */]
plt.plot([1/T for T in temps], [np.log(k) for k in rates])
plt.xlabel("1/T (K⁻¹)")
plt.ylabel("ln(k)")
plt.title("Arrhenius Plot: Eyring TST Rate Constants")
plt.savefig("fig3_arrhenius.pdf")
```

---

## 6. Environment Scenarios

### Temperate (baseline)
```bash
./target/fast/terrarium_evolve --population 8 --generations 5 --frames 100 \
  --fitness biomass --lite --seed 42
```

### Tropical (warm, wet)
```bash
cargo test --no-default-features --lib -- evolve_with_environment_tropical
```

### Arid (hot, dry — harder)
```bash
cargo test --no-default-features --lib -- evolve_with_environment_arid
```

### Drought Stress
```bash
./target/fast/terrarium_evolve --pareto --stress-test --population 8 \
  --generations 5 --frames 50 --lite --seed 42
```

---

## 7. Known Limitations and Reproducibility Notes

### Stochastic Elements
- **Gillespie tau-leaping**: Uses seeded RNG. Same seed → same trajectory.
- **Fly behavior**: Random foraging decisions. Use `--seed` for determinism.
- **NSGA-II**: Tournament selection uses RNG. Fixed seeds reproduce exactly.

### Platform Differences
- **GPU vs CPU**: Metal GPU substrate chemistry may differ from CPU fallback
  by ~1e-6 (float32 rounding). All claims hold within this tolerance.
- **macOS vs Linux**: CPU-only mode produces identical results across platforms.

### Numerical Precision
- All simulations use `f32` for performance. Results are reproducible to
  ~6 significant digits with the same seed on the same platform.
- Cross-platform reproducibility is within ~1e-4 due to FMA instruction differences.

### Test Determinism
- All 216 regression tests are fully deterministic (seeded RNG).
- The 6 ignored quantum tests require specific molecular geometry data.

### Approximate Timings (Apple M1, 16GB)
| Operation | Time |
|-----------|------|
| `cargo check` | ~5s (incremental) |
| Regression tests (216) | ~32s |
| Full test suite (825+) | ~2-3 min |
| 5-gen evolution run | ~10s (--lite) |
| Drug optimizer compare | ~2s |
| Gene circuit design | ~1s |

---

## 8. Supplementary Data

### Parameter Tables

All model parameters are defined in source code with literature references:

| Parameter File | Parameters | Reference |
|---------------|-----------|-----------|
| `whole_cell_data.rs` | 100+ cellular constants | Thornburg et al. 2026 |
| `fly_metabolism.rs` | MM V_max, K_m values | Drosophila biochemistry literature |
| `drosophila_population.rs` | Sharpe-Schoolfield thermal params | Sharpe & DeMichele 1977 |
| `plant_competition.rs` | Beer-Lambert k, LAI scaling | Monsi & Saeki 1953 |
| `soil_broad.rs` | C/N/P pool turnover rates | Parton 1988 |
| `soil_fauna.rs` | Earthworm/nematode carrying capacities | Lavelle 1988 |
| `molecular_dynamics.rs` | TIP3P force field params | Jorgensen 1983 |
| `subatomic_quantum.rs` | STO-3G basis set exponents | Hehre 1969 |

### Code Availability

- **Repository**: https://github.com/robertcprice/oNeura
- **License**: CC BY-NC 4.0 (academic use free; commercial via oneura.ai)
- **Archived version**: Tag `v1.0-paper` (to be created upon submission)
- **Contact**: hello@oneura.ai
