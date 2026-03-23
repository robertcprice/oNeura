# Claude Terrarium Ab Initio Orchestrator Prompt

Date: 2026-03-20

## Read These First

Before writing any code, read these documents in order:

1. `docs/terrarium_ab_initio_status_2026-03-20.md` — what is real, what is
   heuristic, what must not be reintroduced
2. `docs/terrarium_quantum_descriptor_pipeline.md` — the runtime path from
   canonical topology to cached descriptors
3. `plans/ab-initio-terrarium-roadmap-2026-03-20.md` — ordered phases and
   current completion state
4. The heuristic audit labels at the top of:
   - `core/src/terrarium/geochemistry.rs`
   - `core/src/terrarium/inventory_geochemistry_registry.rs`
   - `core/src/terrarium/visual_projection.rs`

## Active Constraints

These constraints are non-negotiable and should be treated as hard invariants:

1. **No species-specific embedded coordinates in Rust source.** Canonical
   molecular topology lives in `core/specs/terrarium_molecular_assets.json`.
   Generic embedding generates coordinates from topology.

2. **No species-specific chemistry coefficients** unless clearly labeled as a
   temporary fallback. If you must add one, annotate it as
   `// TEMPORARY_FALLBACK: <reason>`.

3. **No claiming GPU/shader acceleration** for the descriptor solve until the
   solver actually runs there. The tensor contract exists to enable this later.

4. **No replacing the canonical asset/tensor contract** with ad-hoc species-name
   logic. Any new descriptor tier must consume the same asset registry.

5. **No pretending fallback species are fully emergent.** If a species or phase
   is unresolved, leave it on an explicit fallback path.

6. **Tests must stay green.** The full suite is 910 tests, 0 failures. Any
   change that breaks a test must be fixed before merging.

## Current Starting State

### What compiles

The entire workspace compiles clean:

- `oneura-core` (with and without `web` feature)
- `oneura-cli` (with and without `web` feature)
- `oneura-web` (wasm target)
- `oneura-desktop` (bevy 3D target)

### What tests pass

910 tests pass, 0 fail (`cargo test -p oneura-core --features web`).

### What runs

- `terrarium_web` — serves HTML frontend at localhost:8420 with live WebSocket
  streaming of simulation state
- `terrarium run --seed 42 --frames N` — simple CLI simulation loop
- `terrarium_descriptor_cache --tensor-json -o file.json` — materializes the
  quantum descriptor tensor
- `terrarium_ascii` — terminal-based isometric 3D renderer

### Measured performance

- Fast-path descriptor priming: 12 species in ~0.3ms from JSON cache
- Simulation: ~45 fps for demo preset
- Cold-start without cache: ~42s (this is why the cache matters)

## Your Mission

Work through these phases in order. Do not skip ahead.

### Phase 1: Residual heuristic audit ✓ COMPLETE

Labels are inline in the three core files. See the `// ── Heuristic audit`
comment blocks at the top of each file.

### Phase 2: Add persistent binary descriptor cache

**Files to modify:**
- `core/src/terrarium/inventory_species_registry.rs` — add binary serialization
- `terrarium/cli/src/bin/terrarium_descriptor_cache.rs` — add `--binary` flag
- `core/specs/` — output location for the binary artifact

**Acceptance criteria:**
- Binary format loads faster than JSON
- Checksum validation against `terrarium_molecular_assets.json` hash
- Existing JSON path still works unchanged
- Tests remain green

### Phase 3: Build slower motif/regression tier

**Files to create or modify:**
- `core/src/terrarium/inventory_species_registry.rs` — add a second descriptor
  tier
- `core/specs/terrarium_molecular_assets.json` — add larger species entries

**Acceptance criteria:**
- Larger mineral/interfacial species get descriptors from canonical structures
- No species-name tables introduced
- Fast-path species unaffected

### Phase 4: Push chemistry into environmental embedding

**Target files (from Phase 1 audit):**
- `core/src/terrarium/geochemistry.rs` lines 285-486 (ion exchange)
- `core/src/terrarium/geochemistry.rs` lines 865-958 (thermodynamic potentials)
- `core/src/terrarium/inventory_geochemistry_registry.rs` lines 896-952
  (environmental affinity)

**Acceptance criteria:**
- Fewer TEMPORARY_FALLBACK labels remain
- Chemistry behavior is quantitatively similar (regression tests)
- No new species-name logic introduced

### Phase 5: Accelerator execution

Only begin after Phases 2-4 are stable.

**Acceptance criteria:**
- The accelerator consumes the tensor contract from Phase 2
- Real measured speedup on the target platform
- No regression in chemistry behavior

## Explicit Anti-Patterns

Do NOT:

- Add hardcoded species coordinates in Rust source
- Write GPU/shader code before the tensor contract is stable
- Replace the JSON/binary asset pipeline with in-memory construction
- Introduce species-name branching anywhere in the descriptor or chemistry path
- Skip phases or start Phase 5 before Phase 4 is solid
- Leave failing tests
- Add classification tables that duplicate information from the asset registry

## Expected Output Style

When reporting completed work:

1. **What was built** — plain English explanation of the change
2. **Why it matters** — what this unlocks or improves
3. **Test results** — actual numbers from running the suite
4. **What's next** — concrete next steps in the roadmap
5. **Measured impact** — before/after numbers if applicable

Do not produce file-by-file changelogs. Explain what changed and why.
