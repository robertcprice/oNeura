# oNeura Terrarium: Complete Next Steps & Roadmap

> **Last updated**: 2026-03-17
> **Codebase**: 144,569 lines Rust | 161 files | 825+ tests (216 regression) | 23 binaries
> **Status**: All builds green (0 errors, 25 warnings), 216 regression tests pass, 0 failures

---

## Current State Summary

### What's Working Now
| Component | Status | Lines | Tests |
|-----------|--------|-------|-------|
| Terrarium core (`terrarium.rs`, `terrarium_world.rs`) | Compiling, tested | 5,700+ | 20+ |
| Evolution engine (`terrarium_evolve.rs`) | Full binary, 4 modes + fitness landscape | 5,091 | 49+ |
| Fly metabolism (7-pool MM) | Wired into step loop | 516 | 10 |
| Plant competition (Beer-Lambert) | Wired into step loop | 615 | 10 |
| Soil fauna (earthworm + nematode) | Wired into step loop | 921 | 8 |
| Fly population (Sharpe-Schoolfield) | Wired into step loop | 1,133 | 7 |
| Stochastic gene expression (Gillespie) | Wired into whole_cell | 400 | 10 |
| Cross-scale coupling | Compiling, tested | 300+ | 6 |
| Molecular dynamics (TIP3P) | GPU-accelerated | 864 | 12 |
| Atomistic chemistry (PDB/mmCIF) | Parser + topology | 2,100+ | 31 |
| Quantum runtime (Eyring/TST) | Compiling, **64 pass** | 2,680 | 64 pass, 6 ignored |
| Drug optimizer CLI | Binary ships | 306 | via terrarium_evolve |
| Gene circuit CLI | Binary ships | 341 | via terrarium_evolve |
| Software 3D renderer (minifb) | Binary ships | ~2,500 | manual |
| Semantic zoom (terrarium_zoom) | Binary ships | 1,269 | manual |
| REST API (axum/WebSocket) | Feature-gated (`web`) | 112+ | manual |
| Enzyme engineering | Compiling, tested | 500+ | 8+ |
| Bioremediation | Compiling, tested | 400+ | 6+ |
| Drug discovery | Compiling, tested | 500+ | 8+ |
| Guild latent banks | Compiling, tested | 463 | 5+ |
| Ecosystem integration | Compiling, tested | 800+ | 15+ |
| Advanced ecology (9 modules) | Compiling, tested | 12,000+ | 30+ |
| Terrarium analytics CLI | New binary | ~500 | manual |
| Terrarium profiler CLI | New binary | ~300 | manual |
| Terrarium sensitivity CLI | New binary | ~300 | manual |
| Terrarium stress suite CLI | New binary | ~400 | manual |

### Recently Completed (2026-03-17)
- [x] **Ungated soil.rs** (799 lines) — guild-aware soil stepping compiles unconditionally
- [x] **Ungated snapshot.rs** (1,285 lines) — full world snapshot with guild metrics
- [x] **Ungated biomechanics.rs** (~14 lines stubbed) — wind/pose biomechanics
- [x] **Ungated explicit_microbe_impl.rs** (~220 lines stubbed) — explicit microbe lifecycle
- [x] **Fixed quantum tests** — 64 pass, 0 fail, 6 ignored (was 4 failures)
- [x] **CI/CD pipeline** — `.github/workflows/ci.yml` (check, test, build jobs)
- [x] **Nature Methods paper** — full draft with Introduction, Methods, Results, Discussion (2,127 lines)
- [x] **Novel opportunities doc** — publication strategy, products, research directions (676 lines)
- [x] **Crate extraction plan** — `docs/CRATE_EXTRACTION_PLAN.md` with module classification
- [x] **Fitness landscape scanner** — `FitnessLandscape` + `scan_fitness_landscape()` added to evolution engine
- [x] **Telemetry export** — `telemetry_to_csv()`, `telemetry_to_prometheus()` for observability
- [x] **4 new CLI binaries** — analytics, profiler, sensitivity analysis, stress suite

### What's Still Blocked
| Component | Lines | Blocker | Severity |
|-----------|-------|---------|----------|
| `terrarium_world/render_*.rs` (4 files) | 4,678 | Needs `terrarium_render` crate infrastructure | Low |
| `terrarium_world/tests.rs` | 2,863 | Depends on render modules | Low |
| `biomechanics.rs` real impl | 561 | Needs pose fields on Plant/Seed/Fruit structs | Low |
| `explicit_microbe_impl.rs` real impl | 2,107 | Needs SubstrateKinetics API + material inventory | Low |

---

## Tier 1: Remaining High-Impact Work

### 1.1 Submit the Multi-Scale Biology Paper
**Status**: DRAFT COMPLETE (2,127 lines, 50+ references)
**File**: `docs/METHODS_MULTISCALE_PAPER.md`

**Remaining work**:
- [ ] Generate publication-quality figures from JSON telemetry exports (matplotlib/plotly)
- [ ] Pareto front visualizations from `terrarium_evolve --pareto` output
- [ ] Cross-scale coupling figure (quantum rate correction → macroscopic metabolism)
- [ ] Final copyediting for Nature Methods formatting
- [ ] Supplementary materials (code availability, parameter tables)
- [ ] Submit to Nature Methods or PLOS Computational Biology

### 1.2 Extract `oneura-terrarium` Standalone Crate
**Status**: PLAN COMPLETE (see `docs/CRATE_EXTRACTION_PLAN.md`)

**Ready to execute**: ~85,000 lines move to standalone crate
- [ ] Create workspace layout
- [ ] Move Tier 1 modules (core simulation)
- [ ] Move Tier 2 modules (binaries)
- [ ] Wire backwards compatibility
- [ ] Publish to crates.io

### 1.3 Push to Remote + Enable CI
**Status**: 60 commits ahead of origin
- [ ] Review and squash/organize commits
- [ ] Push to remote
- [ ] Verify CI pipeline passes on GitHub Actions
- [ ] Add status badge to README

---

## Tier 2: Product Development

### 2.1 Cloud Deployment of REST API
- [ ] Dockerize `terrarium_web` binary
- [ ] CORS + rate limiting + API key management
- [ ] Deploy to Fly.io/AWS with WebSocket support
- [ ] OpenAPI/Swagger spec

### 2.2 Web Dashboard
- [ ] React/Svelte frontend consuming WebSocket stream
- [ ] Real-time ecosystem visualization
- [ ] Scenario builder UI for education

### 2.3 Synbio Web Services
- [ ] REST endpoints for drug protocol optimization
- [ ] Gene circuit design web UI
- [ ] Batch screening for circuit design libraries

---

## Tier 3: Research Directions

### 3.1 Antibiotic Resistance Evolution Simulator
Builds on: `PersisterCellSimulator`, `DrugProtocol`, `resistance_evolution.rs`

### 3.2 Synthetic Ecosystem Design
Builds on: `evolve_coevolution()`, `WorldGenome`, multi-objective optimization

### 3.3 Digital Twin Calibration Against Real Soil Data
Builds on: `TerrariumWorldConfig`, Arrhenius temperature scaling

### 3.4 Full Whole-Cell Integration
Builds on: `WholeCellSimulator`, stubbed `explicit_microbe_impl.rs`

### 3.5 CUDA Port for Cloud/NVIDIA
Builds on: `terrarium.rs` substrate chemistry, `cuda` feature flag

---

## Tier 4: Low Priority

### 4.1 Wire Render Pipeline (4,678 lines)
Needs: `terrarium_render`, `terrarium_scene_query` crate infrastructure
Already have: Software 3D renderer + Bevy viewer as alternatives

### 4.2 Unstub biomechanics.rs
Needs: Pose fields on Plant/Seed/Fruit structs + `integrate_displacement()`

### 4.3 Unstub explicit_microbe_impl.rs
Needs: Full SubstrateKinetics API + material inventory infrastructure

### 4.4 Wire tests.rs (2,863 lines)
Depends on: Render pipeline (4.1)

---

## Quick Reference: Build & Test Commands

```bash
# Check compilation
cargo check --no-default-features --lib

# Build all key binaries
cargo build --profile fast --no-default-features \
  --bin terrarium_evolve --bin drug_optimizer --bin gene_circuit \
  --bin terrarium_zoom --bin terrarium_3d

# Run 216 regression tests
cargo test --no-default-features --lib -- \
  substrate_stays_bounded guild_activity soil_atmosphere terrarium_evolve \
  drosophila_population plant_competition soil_fauna fly_metabolism \
  field_coupling seed_cellular terrarium_world organism_metabolism \
  stochastic cross_scale phenotypic persister bet_hedging benchmark \
  seasonal drought tropical arid spatial zone plant_noise microbial_noise \
  multi_species single_drug pulsed combination protocol ecoli circuit \
  enzyme_ bioremediation probe_coupling probe_snapshot drug_enzyme \
  soil_enzyme temperature_coupling remediation guild_latent

# Run ALL tests (825+ pass, 0 fail)
cargo test --no-default-features --lib

# Demo commands
./target/fast/terrarium_evolve --population 8 --generations 5 --frames 100 --fitness biomass --lite
./target/fast/drug_optimizer --mode compare
./target/fast/gene_circuit --target-fano 5.0 --target-mean 100.0
./target/fast/terrarium_3d --seed 7 --fps 30
./target/fast/terrarium_zoom --mode iso --fps 15
```

---

## File Index

| Document | Purpose |
|----------|---------|
| `docs/NEXT_STEPS.md` | This file — complete roadmap |
| `docs/METHODS_MULTISCALE_PAPER.md` | Nature Methods paper draft (2,127 lines) |
| `docs/CRATE_EXTRACTION_PLAN.md` | Crate extraction boundary & plan |
| `NOVEL_OPPORTUNITIES.md` | Business opportunities & monetization strategy |
| `.github/workflows/ci.yml` | CI/CD pipeline (check, test, build) |
