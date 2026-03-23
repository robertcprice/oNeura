# Terrarium Ab Initio Status And Handoff

Date: 2026-03-20
Workspace: `/Users/bobbyprice/projects/oNeura`

## Why This Document Exists

This is the current technical status of the terrarium and aquarium chemistry pipeline after the shift away from hand-authored species tables toward canonical assets, derived descriptors, and cache/tensor contracts.

This document is not marketing copy. It is meant to tell the next engineer exactly:

- what is already real
- what is still fallback or heuristic
- what must not be reintroduced
- what order the remaining work should happen in

## Current Implemented Stack

### Canonical structure authority

Canonical terrarium molecular topology now lives in data, not Rust coordinate tables:

- `core/specs/terrarium_molecular_assets.json`

That asset registry defines:

- species identity
- atoms
- formal charges
- bond graph
- whether a species is on the normal runtime `quantum_fast_path`

Species-specific embedded coordinates in Rust source are explicitly disallowed.

### Generic molecular embedding

Representative `EmbeddedMolecule` values are generated from topology using the generic embedder in:

- `core/src/terrarium/inventory_species_registry.rs`

The embedder is intentionally species-agnostic and uses:

- graph connectivity
- bond order
- covalent radii
- weak charge-aware relaxation

### Derived descriptor cache

Fast-path molecular descriptors now exist in two forms:

1. Live derivation:
   - runtime path in `core/src/terrarium/inventory_species_registry.rs`
   - uses canonical embedded molecules plus the existing quantum machinery

2. Precomputed derived asset:
   - `core/specs/terrarium_quantum_descriptor_cache.json`
   - primes fresh processes before any live fast-path solve

The derived descriptor payload currently includes:

- ground-state energy
- ground-state energy per atom
- dipole magnitude
- mean absolute effective charge
- charge span
- mean LDA exchange potential
- frontier occupancy fraction

### Runtime cache behavior

The cache path in `core/src/terrarium/inventory_species_registry.rs` now does the following:

1. Prime from the bundled derived cache asset.
2. If a descriptor is already cached, return it immediately.
3. If the species is on the explicit `quantum_fast_path`, warm the fast-path cache in parallel on CPU with `rayon`.
4. Fall back to a live solve only when the primed cache does not already satisfy the query.

This makes cold-start behavior dramatically cheaper for the species currently covered by the derived asset.

### Export and accelerator contract

The same registry now exposes:

- per-species JSON snapshot export
- row-major tensor snapshot export

These are surfaced publicly through `core/src/terrarium/mod.rs` and via:

- `terrarium_descriptor_cache` in `terrarium/cli/src/bin/terrarium_descriptor_cache.rs`

The tensor contract is deliberate. It gives later Metal/WGSL or tiled CPU work a stable dense input layout without needing another species-name branch layer.

### Geochemistry consumption

Terrarium geochemistry continues to consume these descriptors in:

- `core/src/terrarium/geochemistry.rs`

That means the thermodynamic profile path driving chemistry-sensitive behavior no longer depends only on formula-level proxies for the covered fast-path species.

### Visual impact

Terrarium and aquarium appearance already reacts to chemistry through the visual projection layer and the web/desktop presentation code. The new cache path matters for visuals because it removes a large first-use penalty from the descriptor-backed chemistry queries that feed those projections.

## Verified Results

These checks were run after the precomputed descriptor asset was wired in:

- `cargo fmt --all`
- `cargo test -p oneura-core --features web terrarium::inventory_species_registry::tests -- --nocapture`
- `cargo test -p oneura-core --features web terrarium::geochemistry::tests::derived_geochemistry_profiles_publish_thermodynamic_potentials -- --exact --nocapture`
- `cargo build -p oneura-cli --bin terrarium_descriptor_cache`
- `cargo run -p oneura-cli --bin terrarium_descriptor_cache -- --tensor-json -o /tmp/terrarium_descriptor_tensor.json`
- `cargo build --bin terrarium_web -p oneura-cli --features web`

Measured behavior from this environment:

- before the precomputed descriptor asset, the fast-path materializer was taking about 42-43 seconds on a fresh process
- after priming from `core/specs/terrarium_quantum_descriptor_cache.json`, the same materializer warmed 12 fast-path terrarium descriptors in about 1.06 milliseconds

That does not mean the underlying quantum solve became cheap. It means fresh processes no longer need to run it for the covered fast-path species.

## Non-Negotiable Engineering Rules

These rules are active and should be preserved:

- No species-specific embedded coordinates hardcoded in Rust source.
- No species-specific chemistry coefficients added unless clearly labeled fallback.
- Canonical molecular or material structure must come from data assets or generic generators.
- If a species or phase is unresolved, leave it on an explicit fallback path instead of pretending it is emergent.
- Do not claim GPU/shader acceleration for the descriptor solve until it is actually implemented.
- Do not replace the canonical asset/tensor contract with ad hoc species-name logic.

## What Is Still Not Real Yet

The stack is stricter and faster now, but several major pieces are still not first-principles:

### Formula/statistic heuristics still exist

Large parts of `core/src/terrarium/geochemistry.rs` still derive chemistry behavior from:

- formula statistics
- class flags
- authored coefficient mixes

This is better than per-species tables, but it is still heuristic.

### Reaction/environment heuristics still exist

`core/src/terrarium/inventory_geochemistry_registry.rs` still contains descriptor-to-drive logic that is authored by us, even when it is no longer species-specific.

### Larger motifs are still fallback

The current fast-path descriptor asset only covers the smaller explicit species that fit the chosen runtime budget. Larger mineral/interfacial motifs still need:

- slower descriptor-regression tiers
- richer material motifs
- or actual heavier solves with stronger caching

### Rendering is not first-principles morphology

Visual response is chemistry-coupled, but it is still not a literal biophysical renderer. Some terrain, deposit, fog, water, and scene-art decisions remain authored mapping logic.

## What Should Happen Next

The remaining work should follow this order.

### 1. Inventory the residual heuristics

Audit and label the remaining authored logic in:

- `core/src/terrarium/geochemistry.rs`
- `core/src/terrarium/inventory_geochemistry_registry.rs`
- `core/src/terrarium/visual_projection.rs`

Every remaining heuristic should be categorized as:

- temporary fallback
- missing-data placeholder
- irreducible presentation mapping

### 2. Add persistent binary cache loading

The JSON asset is enough for correctness and visibility, but not the final runtime format. Add:

- a binary descriptor cache artifact format
- checksum or version validation against the topology asset registry
- a regeneration path through the CLI

Do not replace the JSON asset until the binary path is real and verified.

### 3. Build the slower motif/regression tier

Add a second tier for larger species and motifs that are not on the fast path yet. This should consume canonical structures and produce derived descriptors without reintroducing species-name tables.

### 4. Push more chemistry into environmental embedding

Move more of the geochemistry and reaction drive away from authored coefficient blends and toward environment-aware chemical potentials.

### 5. Only then accelerate the heavier tiers

Once the cache/tensor contract is stable for the slower tier, add actual accelerator work:

- Metal or WGSL kernels
- or clearly justified tiled/vectorized CPU kernels

Do not start by writing shaders for a moving target.

### 6. Keep the visuals coupled to material state

As chemistry gets more structural and less heuristic, continue feeding those outputs into appearance rather than inventing new art tables.

## Five Physics Systems

All five physics systems are implemented and passing. For detailed equations, interconnections, and test coverage, see [`docs/terrarium_five_physics_systems.md`](terrarium_five_physics_systems.md).

| System | What It Does | Source | Tests |
|--------|-------------|--------|-------|
| **Wind & Turbulence** | 3-octave sinusoidal wind field, drag force F=0.5ρCdAv², cantilever beam stress, Hill damage accumulation, pollen wind boost | `biomechanics.rs`, `atmosphere.rs` | 22 |
| **Soil Depth** | 4-layer vertical profile [2-100mm], Fick's-law percolation, capillary rise, rainfall infiltration, root-weighted moisture | `soil_profile.rs` | 10 |
| **Weather** | Emergent from physics: humidity→cloud cover (Clausius-Clapeyron + Hill), cloud+humidity→precipitation, cloud albedo + evaporative cooling→temperature offset. WeatherRegime is diagnostic label only. | `biomechanics.rs` (WeatherState) | 12 |
| **Spectral Light** | Astronomical solar position, R:FR decomposition (direct 1.2, diffuse 1.0), per-plant canopy raycast, PHYB/SAS gene circuits, SAS→growth pipeline (elongation/branching) | `solar.rs`, `physiology_bridge.rs`, `flora.rs` | 23 + 5 |
| **VOC Signaling** | Damage→JA→GLV/MeSA emission→odorant ch5→wind dispersal→neighbor SA_RESPONSE/DEFENSE_PRIMING, AND-gate priming, herbivore grazing with Hill deterrence | `metabolome.rs`, `flora.rs`, `fauna.rs` | 11 + 5 |

Key integration points:
- Wind disperses VOCs through the odorant grid
- Weather emerges from simulation state (humidity, temperature, soil moisture, wind) — not a state machine
- Spectral R:FR feeds SAS gene expression → height elongation and branching suppression
- Both wind damage and herbivore grazing trigger the same JA defense cascade
- Soil depth profile provides root-weighted moisture → stomatal openness → VOC emission rate

Total test count with physics systems: 1111 tests, 0 failures.

## Companion Files

Use these together:

- `docs/terrarium_quantum_descriptor_pipeline.md`
- `docs/terrarium_five_physics_systems.md`
- `docs/terrarium_novel_opportunities.md`
- `plans/ab-initio-terrarium-roadmap-2026-03-20.md`
- `plans/claude_terrarium_ab_initio_orchestrator_prompt_2026-03-20.md`

## Final Honest Statement

This is not a literal ab initio terrarium.

It is, however, a materially more honest stack than before:

- canonical topology in data
- generic embedding instead of handwritten coordinates
- derived molecular descriptors
- cached and exportable descriptor tensors
- cold-start priming from derived assets
- chemistry-backed visual response

The next engineer should preserve that direction and make the remaining heuristics smaller, more explicit, and more derived, not hide them under new names.
