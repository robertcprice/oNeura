# Terrarium Ab Initio Status

Date: 2026-03-21
Workspace: `/Users/bobbyprice/projects/oNeura`
Previous: `docs/terrarium_ab_initio_status_2026-03-20.md`

## What Changed Since 2026-03-20

### Temperature-Coupled Fly Metabolism (Step 1)

Six fly metabolism Vmax constants (`CROP_VMAX`, `TREHALASE_VMAX`, `GLYCOLYSIS_VMAX`, `GLYCOGENOLYSIS_VMAX`, `LIPID_MOB_VMAX`, `GLYCOGEN_STORAGE_VMAX`) are now wired through `metabolome_rate()` with Eyring TST temperature scaling. Insect metabolism responds emergently to environmental temperature — cold suppresses, warm accelerates.

Files: `fly_metabolism.rs`, `emergent_rates.rs`, `fauna.rs`
New tests: 4 (cold slows, warm speeds, Q10 check, extreme cold survival)

### Thermal-Time Seed Dormancy (Step 2)

Linear countdown dormancy (`dormancy_s -= constant * dt`) replaced with `metabolome_rate("seed_germination", T)`. No hardcoded T_base cutoff — temperature sensitivity emerges from C-O bond physics (Eyring TST). Seeds stall naturally below ~5°C because the Boltzmann factor drops to near zero.

Files: `seed_microsite.rs`, `flora.rs`, `emergent_rates.rs`
New tests: 4 (cold no progress, warm moist germinates, dry stalled, temperature proportional)

### Visual Emergence Blend Wiring (Step 3)

The `visual_emergence_blend: f32` config knob was dead code — present in config but never threaded to the render pipeline. Now wired through:

- `terrain_visual_response_with_chemistry(blend)` — lerps base terrain color with emergent mineral/chemistry-derived color
- `fruit_visual_response(blend)` — lerps fruit palette with emergent ripeness color

Users can A/B compare legacy vs fully emergent visuals by setting `visual_emergence_blend` from 0.0 (legacy) to 1.0 (fully emergent).

Files: `visual_projection.rs`, `visual_markers.rs`, `terrarium_web_handlers.rs`, `terrarium_web_cutaway.rs`
New tests: 2 (blend=0 preserves base, blend=1 uses emergent chemistry)

### Climate Scenario Presets (Step 4)

Because ALL metabolic rates now derive from temperature via Eyring TST, climate scenarios produce genuinely emergent ecosystem response:

- `TerrariumDemoPreset::warming_demo()` — Demo + RCP 8.5
- `TerrariumDemoPreset::stable_demo()` — Demo + PreIndustrial
- `TerrariumDemoPreset::moderate_warming_demo()` — Demo + RCP 4.5
- `TerrariumWorld::demo_preset_with_climate()` — General climate-attached constructor

The climate driver relaxes temperature, humidity, CO₂, and wind fields toward IPCC AR5-calibrated trajectories. Elevated CO₂ propagates to substrate dissolved CO₂. Temperature changes cascade through Eyring TST to every metabolic rate.

Files: `presets.rs`
New tests: 4 (builds, warming > stable temperature, CO₂ propagation, convenience methods)

### Ecosystem Telemetry API (Step 5)

Molecular-scale inspect now includes a `derivation_chain` field that traces how atomic properties propagate through the emergent pipeline:

**Optical derivation**: atom composition → CPK colors → molar extinction → scattering cross-section
**Rate derivation**: pathway → bond type → BDE (eV) → enzyme efficiency → Vmax@25°C → citation

This is the first user-facing API that lets researchers see the complete atoms→rates→behavior chain.

Files: `scale_level.rs` (new types), `emergent_rates.rs` (derivation helper), `terrarium_web_inspect.rs` (builder + tests)
New tests: 2 (glucose derivation chain, CO₂ derivation chain)

## Emergent Rate Engine Coverage

All temperature-sensitive rates now derive from `metabolome_rate(pathway, temperature_c)` via Eyring TST:

| Domain | Pathways | Source |
|--------|----------|--------|
| **Plant metabolome** | photosynthesis, fructose, sucrose, starch, malate, citrate, ethylene, benzaldehyde, limonene, anthocyanin, carotenoid, voc, jasmonate, salicylate, glv, mesa | Eyring TST + literature Vmax |
| **Plant respiration** | respiration | Eyring TST (replaced Q10) |
| **Fly metabolism** | fly_crop, fly_trehalase, fly_glycolysis, fly_glycogenolysis, fly_lipid_mobilization, fly_glycogen_storage | Eyring TST + insect enzyme Vmax |
| **Seed dormancy** | seed_germination | Eyring TST (replaced linear countdown) |

Total pathways: 27 (all temperature-coupled via single Eyring TST framework)

## Test Suite

Full suite: `cargo test -p oneura-core --features web`

- **1186+ tests** (1170 original + 16 new from this session)
- 0 failures (excluding known flaky `macro_checkpoint_json_roundtrip_serializes` under parallel execution)

New test groups:
- Fly metabolism temperature: 4 tests
- Seed dormancy thermal time: 4 tests
- Visual blend wiring: 2 tests
- Climate presets: 4 tests
- Telemetry derivation chain: 2 tests

## Fully Emergent Pipeline Status

All 6 phases complete + 5 novel physics systems + 5 new steps from this session:

| Phase | Status | Tests |
|-------|--------|-------|
| 1. Visual (emergent_color.rs) | Complete | 12 |
| 2. Stoichiometry (SubstrateStoichiometryTable) | Complete | 8 |
| 3. Initial Concentrations (Jenny/Gapon/Nernst) | Complete | varies |
| 4. Botany (Eyring TST + allometry) | Complete | ~37 |
| 5. Soil/Weather (van Genuchten, Clausius-Clapeyron) | Complete | varies |
| 6. Remaining (fly metabolism named+cited) | Complete | varies |
| **NEW: Fly temperature coupling** | Complete | 4 |
| **NEW: Seed thermal time** | Complete | 4 |
| **NEW: Visual blend wiring** | Complete | 2 |
| **NEW: Climate presets** | Complete | 4 |
| **NEW: Telemetry API** | Complete | 2 |

## What Is Still Not First-Principles

Unchanged from 2026-03-20 status doc:
- Formula/statistic heuristics in `geochemistry.rs`
- Reaction/environment heuristics in `inventory_geochemistry_registry.rs`
- Larger motifs still fallback
- Rendering not literal biophysical morphology

## Companion Files

- `docs/terrarium_ab_initio_status_2026-03-20.md` — previous status
- `docs/terrarium_quantum_descriptor_pipeline.md`
- `docs/terrarium_five_physics_systems.md`
- `docs/terrarium_novel_opportunities.md`
- `plans/ab-initio-terrarium-roadmap-2026-03-20.md`
