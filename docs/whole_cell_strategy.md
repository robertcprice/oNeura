# Whole-Cell Strategy

## Decision

We should do both of the following:

1. Use the published minimal-cell stack as an external reference and baseline.
2. Expand `oNeuro` with a separate whole-cell subsystem instead of trying to force the neural engine to do bacterial whole-cell simulation.

This is not an either/or choice.

## Primary Reference

This whole-cell work is explicitly anchored to:

- Thornburg, Z. R., Maytin, A., Kwon, J., Solomon, K. V., et al. "Bringing the Genetically Minimal Cell to Life on a Computer in 4D." `Cell` (2026). DOI: `10.1016/j.cell.2026.02.009`
- BibTeX entry in `docs/papers/minimal_cell_4d_reference.bib`

## What Is Already Available

The minimal-cell team has already released:

- A public GitHub repository for the `Minimal_Cell_4DWCM` codebase.
- A frozen Zenodo software/data release for the production runs.
- A paper describing the hybrid RDME/CME/ODE/BD architecture.

That means we do **not** need to rediscover their entire method from scratch just to reproduce a baseline result.

## What We Can Reuse

Short-term reuse should focus on the external reference stack:

- `Lattice Microbes` for RDME reaction-diffusion.
- `odecell` for metabolic ODE integration.
- `btree_chromo_gpu` and `LAMMPS` for chromosome Brownian dynamics.
- The published orchestration pattern that couples RDME, CME, ODE, BD, and geometry updates.
- Their frozen Syn3A production configuration as a baseline target.

Within `oNeuro`, the reusable surfaces are mostly engineering patterns, not domain-complete biology:

- Step-based subsystem updates.
- Dataclass-heavy state definitions.
- Existing diffusion/environment coding style.
- Existing experiment/reporting conventions.

## What We Should Not Assume

We should **not** assume that the external release means `oNeuro` already has this capability.

The current repo is still neural-first:

- `MolecularNeuron` is the core unit, not a bacterial cell state.
- The current cellular layer is generic and largely eukaryotic.
- There is no native whole-cell scheduler coupling RDME, CME, ODE, BD, and geometry.
- There is no bacterial chromosome dynamics layer, intracellular 10 nm lattice, or membrane-growth-driven cytokinesis runtime.

## Licensing And Reuse Policy

The safest current policy is:

- Treat the Zenodo release as the canonical reusable artifact for the published software package and data.
- Do not vendor code directly from the GitHub `main` branch until repository-level licensing is explicit.
- Keep any near-term integration as an adapter or bridge, not a silent code copy.

Reason:

- The Zenodo record declares `CC-BY-4.0`.
- The GitHub repository metadata currently does not expose a repository license file.

## Recommended Build Strategy

### Track A: External Baseline

Goal: reproduce and inspect the published pipeline with as little reinvention as possible.

Scope:

- Stand up the published MC4D environment separately from `oNeuro`.
- Verify that we can run a small or partial reference workload.
- Use the external runtime as the benchmark target for architecture and output shape.

### Track B: Native `oNeuro` Whole-Cell Layer

Goal: build a first-class whole-cell subsystem inside `oNeuro`.

Scope:

- New `oneuro.whole_cell` package.
- Native program specification for whole-cell targets.
- Clear separation from the neural stack.
- Adapter surfaces for external solvers and published datasets.
- Rust-native execution inside `oneuro-metal` for the performance-critical path.

## Current Native Status

The native path is now split intentionally:

- `oneuro.whole_cell` remains the Python-side integration, adapter, artifact, and orchestration surface.
- `oneuro-metal` now contains the Rust whole-cell runtime for fast local simulation.

Current native capabilities in `oneuro-metal`:

- `WholeCellSimulator` implemented in Rust.
- Voxelized intracellular lattice in structure-of-arrays form.
- Custom Metal compute shader for intracellular RDME on macOS.
- Stability-preserving RDME substepping for the default diffusion regime.
- Rayon CPU fallback for non-Metal platforms.
- PyO3 exposure so the Rust simulator can still be driven from Python without moving the engine back into Python.
- Optional `nQPU` bridge that feeds sparse quantum-chemistry correction factors into the Rust runtime instead of trying to run whole-cell dynamics in Python.
- Optional local chemistry microdomain bridge built on the existing batched terrarium substrate.
- Optional localized MD probes that feed ATP, translation, replication, segregation, membrane, and constriction recommendations back into the whole-cell runtime.
- Persistent Syn3A subsystem states for ATP-synthase bands, ribosome clusters, replisome tracks, and FtsZ septum rings.
- 9 quantum hotspot kinds with lifted ED headroom (CIPSI=8000, orbitals=48, atom budgets 8-12).
- Live-state quantum carving: `refresh_quantum_corrections_from_live_state()` modulates discovered quantum reaction corrections from complex assembly counts, subsystem activity, and energy charge every 100 steps.
- Probe-driven refinement: `refine_quantum_corrections_from_probe()` maps 6 MD probe thermodynamic observables to 5 quantum efficiency channels with exponential blending (α=0.12).
- Broadened quantum auto-discovery: 12/17 reaction classes now eligible for quantum corrections (was 5/17).
- Extended surrogate fast path: uses direct inventory from named_complexes and complex_assembly in addition to organism_assets.
- Site-resolved local chemistry reports so each subsystem consumes its own weighted microdomain support rather than only a global chemistry mean.
- Localized substrate demand/depletion so active subsystem sites draw down their own terrarium patches before support is evaluated.
- Persistent microdomain chemistry memory so local depletion and waste fields survive across scheduler updates instead of being rebuilt from the coarse state each step.
- Generic bulk-exchange relaxation from coarse concentrations back into the terrarium, so replenishment is driven by the field dynamics rather than preset-specific refill rules.
- Generic substrate-level reaction IR now sits underneath the local chemistry bridge, so site chemistry is executed from stoichiometric rules instead of bespoke hotspot seeding branches.
- Generic assembly/occupancy evaluation now also sits under the local chemistry bridge, so structural order is derived from weighted component availability, local pressure, and crowding rather than hard-coded site behavior.
- Generic spatial localization now resolves subsystem patches from substrate chemistry, geometry cues, continuity memory, and exclusion pressure instead of fixed preset anchors.
- Local reaction activity and catalyst drive are now derived from resolved patch chemistry, local assembly state, and persistent subsystem state instead of snapshot-level subsystem counters like ribosome, DnaA, or FtsZ counts.
- Native CME/ODE/BD/geometry stages now consume substrate-derived assembly inventories and process-capacity signals, with ribosome/RNAP/DnaA/FtsZ pool fields retained only as smoothed diagnostics for bindings and observability.
- Those higher-level inventories and rates are now evaluated through a generic scalar process-rule IR, so the native scheduler is reading interpreted rule tables rather than bespoke arithmetic blocks.
- Subsystem readiness reducers, coarse resource-to-signal mappings, and terrarium snapshot-exchange targets are now also evaluated through rule tables, which removes another layer of inline whole-cell heuristics from the stack.
- Local chemistry crowding, derived patch signals, assembly context scaling, subsystem structural targets, and aggregate chemistry support are now evaluated through a generic affine reducer layer instead of bespoke local weighted sums.
- The native intracellular lattice now starts from neutral bulk fields instead of injecting preset ATP hotspots at initialization; spatial structure must come from later dynamics or explicit perturbations.
- Local depletion and byproduct pressure now feed back into subsystem scaling and effective metabolic load instead of remaining passive diagnostics.
- Per-subsystem differences in chemistry and scaling are now expressed as profile data tables instead of branch-specific support formulas, which keeps the runtime closer to a substrate-first architecture.

## What Must Be Built In `oNeuro`

These surfaces do not currently exist in the repo and should be treated as new work:

- Whole-cell orchestration scheduler.
- Genome-scale molecular count state.
- Intracellular spatial lattice model.
- Chromosome replication and segregation interface.
- Growth and division geometry controller.
- Data interchange layer between stochastic, deterministic, and spatial solvers.
- Minimal-cell-specific validation suite.

## Integration Wiring Table (Phases 1-8)

| Phase | Subsystem | File | Lines | Tests |
|-------|-----------|------|-------|-------|
| 1 | Chromosome physics | whole_cell/chromosome.rs | ~420 | ✅ |
| 2 | Divisome/FtsZ | whole_cell/stages.rs | wired | ✅ |
| 3 | Membrane geometry | whole_cell/membrane.rs | ~180 | ✅ |
| 4 | Multirate scheduler | whole_cell.rs | wired | ✅ |
| 5 | Local chemistry bridge | whole_cell/local_chemistry.rs | ~370 | ✅ |
| 6 | Quantum rate bridge | whole_cell.rs | wired | ✅ |
| 7 | Spatial coupling | whole_cell/spatial.rs | ~150 | ✅ |
| 8 | Quantum auto-discovery | whole_cell.rs | ~120 new | 5 tests ✅ |
| 9 | GPU Monod dynamics | terrarium_substrate.metal | already done | ✅ |

### Phase 8: Quantum Auto-Discovery (NEW)
- `QuantumDiscoveredReaction` + `QuantumProfileChannel` types
- `run_quantum_auto_discovery()`: scans organism reactions by class + subsystem targets
- `refine_quantum_corrections_from_probe()`: EMA (α=0.12) updates from MD probe thermodynamics
- `apply_quantum_discovery_to_profile()`: geometric mean per channel → quantum_profile
- Wired into `new()`, `from_program_spec()`, and local_chemistry MD probe cycle

### Observable Auto-Calibration Engine (NOVEL)
- `ObservableKind` enum: 13 measurable quantities (ATP, amino acids, ribosomes, volume, growth rate, etc.)
- `ObservableTarget`: experimental target with tolerance and weight
- `calibrate_from_observables()`: coordinate-descent optimizer over 5 quantum profile efficiencies
  - Multi-scale grid search (step sizes 0.20, 0.10, 0.05, 0.02)
  - Creates fresh simulators per evaluation (no Clone dependency)
  - Returns `ObservableCalibrationResult` with per-target errors and fitted profile
- 3 unit tests pass (loss computation, observe all kinds)

## Remaining Critical Path

- [ ] Wire orphaned submodules in `terrarium_world/` directory (14k lines uncompiled)
- [ ] Wire `whole_cell_quantum_runtime.rs` (324KB, not in lib.rs)
- [ ] Full doubling-time validation against Syn3A literature (~65 min)

## Product Direction

The recommended product direction is:

- Use their released system to avoid wasting time rebuilding a known baseline.
- Expand the library anyway, because owning the whole-cell abstraction inside `oNeuro` is what enables differentiation.

In practice:

- Reproduce with their stack.
- Integrate around it.
- Replace selectively where `oNeuro` needs native control, portability, or new biology.
