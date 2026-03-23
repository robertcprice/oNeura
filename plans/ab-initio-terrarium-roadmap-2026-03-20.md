# Ab Initio Terrarium Roadmap

Date: 2026-03-20
Workspace: `/Users/bobbyprice/projects/oNeura`

## Current State

The terrarium simulation compiles across the entire workspace, passes 910 tests
with 0 failures, and has three verified demo paths:

- `terrarium_web` — browser-based real-time terrarium with WebSocket streaming
- `terrarium run` — CLI simulation runner (~45 fps)
- `terrarium_descriptor_cache` — quantum descriptor pipeline materializer

The molecular descriptor pipeline (canonical topology → generic embedding →
fast-path quantum descriptors → geochemistry) is real and working. Cold-start
priming from the derived cache takes ~0.3ms for 12 species.

## Ordered Phases

### Phase 1: Residual heuristic audit ✓ COMPLETE

Every authored heuristic in the three core chemistry/visual files has been
categorized as one of:

- **TEMPORARY_FALLBACK** — should eventually be replaced by derived data
- **MISSING_DATA_PLACEHOLDER** — reasonable default until data source exists
- **IRREDUCIBLE_PRESENTATION** — art/mapping decisions that cannot derive from
  first principles

Audit labels are now inline at the top of:

- `core/src/terrarium/geochemistry.rs`
- `core/src/terrarium/inventory_geochemistry_registry.rs`
- `core/src/terrarium/visual_projection.rs`

### Phase 2: Persistent binary descriptor cache

- Add a binary descriptor cache artifact format (e.g., bincode or
  zero-copy-friendly).
- Add checksum validation against the topology asset registry.
- Add a CLI regeneration path.
- Do not remove the JSON asset until the binary path is verified.

### Phase 3: Slower motif/regression tier

- Add a second descriptor tier for larger mineral and interfacial species.
- Must consume canonical structures from the asset registry.
- Must produce derived descriptors without species-name tables.
- May use a regression model trained on existing fast-path descriptors.

### Phase 4: Environmental embedding for chemistry

- Replace authored coefficient blends in `geochemistry.rs` with environment-
  aware chemical potentials where possible.
- The TEMPORARY_FALLBACK labels from Phase 1 define the target set.
- Priority: ion exchange (lines 285-486) and thermodynamic potentials (865-958).

### Phase 5: Accelerator execution

- Only start after the cache/tensor contract is stable for the slower tier.
- Metal or WGSL kernels for the descriptor tensor.
- Or tiled/vectorized CPU kernels with clear justification.
- Must consume the same canonical asset and tensor contracts.

### Phase 6: Keep visuals coupled to material state

- As chemistry becomes more structural and less heuristic, continue feeding
  those outputs into appearance rather than inventing new art tables.
- The IRREDUCIBLE_PRESENTATION labels from Phase 1 mark decisions that stay.

## Companion Files

- `docs/terrarium_ab_initio_status_2026-03-20.md` — technical status/handoff
- `docs/terrarium_quantum_descriptor_pipeline.md` — descriptor pipeline docs
- `plans/claude_terrarium_ab_initio_orchestrator_prompt_2026-03-20.md` — Claude
  orchestration prompt
