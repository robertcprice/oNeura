# Salvage Analysis Report - oNeura Project
## Generated: 2026-03-16
## Working Baseline: `fb092cb` "Calibrate Metal neuron resting-state receptor and leak dynamics"
## Broken Commits: 118 commits between fb092cb and HEAD

---

## Executive Summary

The - **Working Baseline**: `fb092cb` compiles with only warnings
    - **Broken HEAD**: 118 commits of whole-cell refactoring
    - **Root Cause**: Missing module declarations in `lib.rs` for new helper modules
    - **Salvage Strategy**: Incremental integration, independent modules first

---

## Category 1: FULLY SALVAGEABLE (No Dependencies)

These files can be added to `lib.rs` immediately and will compile:

### `organism_metabolism.rs`
- **Purpose**: Universal metabolic observation trait for all organisms
- **Lines**: ~50 lines
- **Dependencies**: NONE - pure trait definition
- **Tests**: None (trait only)
- **Risk**: ZERO - can add immediately
- **Action**:
  ```rust
  // In lib.rs, add:
  pub mod organism_metabolism;
  ```

### `field_coupling.rs`
- **Purpose**: 2D/3D Gaussian field deposit and mean-map helpers
- **Lines**: ~150 lines
- **Dependencies**: NONE - standalone math functions
- **Tests**: 5 tests in file
- **Risk**: LOW - pure functions, no state
- **Action**:
  ```rust
  // In lib.rs, add:
  pub mod field_coupling;
  ```

---

## Category 2: SALVAGEABLE WITH MINOR DEPENDENCIES

### `fly_metabolism.rs`
- **Purpose**: Drosophila 7-pool Michaelis-Menten metabolism model
- **Lines**: ~500 lines
- **Dependencies**:
  - `crate::constants::michaelis_menten` (EXISTS in baseline)
  - `crate::organism_metabolism::OrganismMetabolism` (needs Category 1)
- **Tests**: 9 tests - ALL PASS (verified)
- **Risk**: LOW - depends only on Category 1
- **Action**: Add `organism_metabolism` first, then this

### `substrate_coupling.rs`
- **Purpose**: Coarse world field ↔ explicit substrate coupling
- **Lines**: ~200 lines
- **Dependencies**:
  - `crate::constants` (EXISTS)
  - `crate::terrarium` (EXISTS)
- **Tests**: 2 tests in file
- **Risk**: LOW
- **Action**: Add after Category 1

### `terrarium_contact.rs`
- **Purpose**: Fly-terrain contact/collision physics (BVH raycast)
- **Lines**: ~300 lines
- **Dependencies**: `crate::constants` (EXISTS)
- **Tests**: None
- **Risk**: LOW
- **Action**: Add directly

### `terrarium_scene_query.rs`
- **Purpose**: BVH scene raycasting utilities
- **Lines**: ~250 lines
- **Dependencies**: `crate::constants` (EXISTS)
- **Tests**: None
- **Risk**: LOW
- **Action**: Add directly

### `terrarium_render.rs`
- **Purpose**: Render descriptors, topdown view projections
- **Lines**: ~200 lines
- **Dependencies**: `crate::constants` (EXISTS)
- **Tests**: None
- **Risk**: LOW
- **Action**: Add directly

### `terrarium_render_pipeline.rs`
- **Purpose**: Render pipeline orchestration
- **Lines**: ~300 lines
- **Dependencies**: `crate::constants` (EXISTS)
- **Tests**: None
- **Risk**: LOW
- **Action**: Add directly

### `soil_fauna.rs`
- **Purpose**: Soil ecology (nematodes, protozoa, etc.)
- **Lines**: ~400 lines
- **Dependencies**: `crate::constants` (EXISTS)
- **Tests**: None
- **Risk**: LOW
- **Action**: Add directly

### `plant_competition.rs`
- **Purpose**: Beer-Lambert shading competition
- **Lines**: ~200 lines
- **Dependencies**: `crate::constants` (EXISTS), `crate::plant_organism` (EXISTS)
- **Tests**: None
- **Risk**: LOW
- **Action**: Add directly

### `seed_cellular.rs`
- **Purpose**: Seed cellular metabolism
- **Lines**: ~200 lines
- **Dependencies**: `crate::constants` (EXISTS), `crate::cellular_metabolism` (EXISTS)
- **Tests**: None
- **Risk**: LOW
- **Action**: Add directly

### `drosophila_population.rs`
- **Purpose**: Fly population dynamics
- **Lines**: ~300 lines
- **Dependencies**: `crate::constants` (EXISTS)
- **Tests**: None
- **Risk**: LOW
- **Action**: Add directly

---

## Category 3: COMPLEX MODULES (Need Careful Integration)

### `terrarium_evolve.rs` (lib) + `bin/terrarium_evolve.rs`
- **Purpose**: Evolution engine with 10 modes
- **Lines**: ~4,200 lines (lib) + ~700 lines (bin)
- **Tests**: 33 passed, 1 ignored
- **Dependencies**: Many - requires `TerrariumWorld`, fitness functions, etc.
- **Risk**: MEDIUM - needs module declarations and integration
- **Note**: BINARY EXISTS and works independently
- **Action**:
  1. Add module declaration
  2. Verify all imports resolve
  3. May need minor fixes for missing APIs

### `terrarium_world/` directory
- **Purpose**: Modularized terrarium world implementation
- **Files**: 13 submodules (flora.rs, soil.rs, mesh.rs, etc.)
- **Lines**: ~2,500 lines main + submodules
- **Dependencies**: Deep integration with existing `terrarium_world.rs`
- **Risk**: HIGH - significant refactoring required
- **Action**:
  - The files extend `impl super::TerrariumWorld` blocks
  - Need to merge into existing `terrarium_world.rs` or restructure

---

## Category 4: DEEPLY INTEGRATED (High Risk)
### `atomistic_chemistry/` directory
- **Purpose**: Explicit periodic elements, atom/bond graphs, molecular stoichiometry
- **Files**: mod.rs, molecule.rs, material.rs, law_based_enumeration.rs
- **Lines**: ~7,000+ lines total
- **Dependencies**:
  - `crate::molecular_dynamics` (EXISTS)
  - `crate::subatomic_quantum` (needs this)
  - `crate::whole_cell` (needs this)
- **Risk**: HIGH - complex interdependencies
- **Note**: Uses `pub(super)` and `pub(crate)` visibility extensively
- **Action**:
  - Requires careful module structure planning
  - Must declare in lib.rs with proper visibility

### `subatomic_quantum.rs`
- **Purpose**: Explicit subatomic quantum-chemistry state
- **Lines**: ~500 lines
- **Dependencies**: `crate::atomistic_chemistry` (needs atomistic_chemistry first)
- **Risk**: HIGH
- **Action**: Must integrate atomistic_chemistry first

### `atomistic_topology.rs`
- **Purpose**: Atomistic local-topology templates for whole-cell microdomain refinement
- **Lines**: ~380 lines
- **Dependencies**: `crate::molecular_dynamics` (EXISTS)
- **Risk**: MEDIUM
- **Action**: Can add independently, needs JSON spec file

### `whole_cell_quantum_runtime.rs`
- **Purpose**: Quantum chemistry runtime for whole-cell simulation
- **Lines**: ~3,200 lines
- **Dependencies**:
  - `crate::whole_cell_data` (broken commit)
  - `crate::atomistic_chemistry` (needs this)
  - `crate::subatomic_quantum` (needs this)
- **Tests**: 55 passed, 1 ignored (when integrated properly)
- **Risk**: VERY HIGH - deeply integrated
- **Action**: Requires working whole_cell_data.rs first

### `whole_cell/` directory
- **Purpose**: Modular whole-cell implementation
- **Files**: mod.rs, types.rs, scheduler.rs, runtime_quantum.rs, etc.
- **Lines**: ~20,000+ lines total
- **Dependencies**:
  - `crate::whole_cell_data` (broken commit)
  - `crate::whole_cell_quantum_runtime` (needs this)
  - Many others
- **Risk**: VERY HIGH - this is the core of the broken commits
- **Action**:
  - The `whole_cell.rs` file in HEAD (182KB) conflicts with `whole_cell/mod.rs`
  - Need to decide: keep monolithic or split modular
  - Root cause of compile failures

### `whole_cell_data.rs`
- **Purpose**: Whole-cell data structures and asset compilation
- **Lines**: ~11,000 lines
- **Dependencies**: serde, various internal modules
- **Risk**: HIGH - but appears to be self-contained
- **Note**: The KEY file - many others depend on this
- **Action**:
  - Must declare in lib.rs as `pub mod whole_cell_data;`
  - This is the foundation for the whole-cell refactor

---

## Supporting Modules (Need to be declared in lib.rs)
These are untracked but referenced by whole_cell.rs:

```
mod whole_cell_assembly_fallbacks;
mod whole_cell_assembly_projection;
mod whole_cell_asset_fallbacks;
mod whole_cell_chromosome_math;
mod whole_cell_complex_channels;
mod whole_cell_inventory_authority;
mod whole_cell_named_complex_dynamics;
mod whole_cell_process_occupancy;
mod whole_cell_process_weights;
mod whole_cell_rule_math;
mod whole_cell_scale_reducers;
mod whole_cell_signal_estimators;
```

**Note**: These are currently NOT declared in lib.rs, which causes the E0433 errors.

---

## Integration Order (Recommended)

### Phase 1: Add Independent Modules (Zero Risk)
```
1. organism_metabolism.rs     <- Trait only, no deps
2. field_coupling.rs           <- Pure functions
3. terrarium_contact.rs       <- Collision physics
4. terrarium_scene_query.rs   <- BVH utilities
5. terrarium_render.rs        <- Render descriptors
6. terrarium_render_pipeline.rs <- Render orchest
7. soil_fauna.rs               <- Soil ecology
8. plant_competition.rs       <- Beer-Lambert
9. seed_cellular.rs           <- Seed metabolism
10. drosophila_population.rs  <- Fly population
```

### Phase 2: Add Modules With Minor Dependencies
```
1. fly_metabolism.rs          <- needs organism_metabolism
2. substrate_coupling.rs      <- needs terrarium
```

### Phase 3: Add Evolution Engine (Test Binary)
```
1. Test binary compilation: src/bin/terrarium_evolve.rs
2. Library support: src/terrarium_evolve.rs
   - Needs: TerrariumWorld, various imports
   - Has 33 passing tests
```

### Phase 4: Core Data Layer (Foundation for Whole-Cell)
```
1. whole_cell_data.rs         <- Must be declared in lib.rs
   - ~11,000 lines
   - Self-contained, no crate dependencies
```

### Phase 5: Supporting Modules (Need lib.rs declarations)
```
1. All whole_cell_*.rs helper modules
   - Must add `mod` declarations to lib.rs
```

### Phase 6: Atomistic Chemistry (Complex)
```
1. atomistic_chemistry/ directory
   - Must integrate carefully
   - Dependencies on molecular_dynamics, subatomic_quantum
2. subatomic_quantum.rs
   - Depends on atomistic_chemistry
3. atomistic_topology.rs
   - Can add independently
```

### Phase 7: Whole-Cell Refactor (Highest Risk)
```
1. whole_cell_quantum_runtime.rs
   - Depends on whole_cell_data, atomistic_chemistry, subatomic_quantum
2. whole_cell/ directory OR whole_cell.rs
   - CHOICE: Keep monolithic or split modular
   - This is where the breakage occurred
```

---

## Critical Decision Points

### 1. Whole-Cell Architecture
**Option A**: Keep monolithic `whole_cell.rs` (182KB)
- Simpler integration
- Existing working code
- Miss out on modularity benefits

**Option B**: Use `whole_cell/` directory structure
- More maintainable
- Better separation of concerns
- Requires significant integration work
- ROOT CAUSE of current breakage

**Recommendation**: Start with Option A (monolithic), migrate to Option B incrementally

### 2. Module Visibility Strategy
The helper modules need visibility declarations:
```rust
// Option 1: Make public (simplest)
pub mod whole_cell_data;
pub mod whole_cell_quantum_runtime;
// etc.

// Option 2: Keep private, use pub(crate)
mod whole_cell_data;
mod whole_cell_quantum_runtime;
// etc.
```

**Recommendation**: Option 1 for now (public), can refine later

---

## Test Coverage Summary

| Module | Tests | Status |
|-------|------|--------|
| fly_metabolism | 9 | ALL PASS (verified) |
| terrarium_evolve | 34 (33+1 ignored) | PASS when integrated |
| whole_cell_quantum_runtime | 56 | PASS when properly integrated |
| field_coupling | 5 | In file |
| substrate_coupling | 2 | In file |

---

## Estimated Effort

| Category | Files | Risk | Time |
|----------|-------|------|------|
| Phase 1: Independent | 10 | LOW | 30 min |
| Phase 2: Minor Deps | 2 | LOW | 15 min |
| Phase 3: Evolution | 2 | MEDIUM | 1 hour |
| Phase 4: Data Layer | 1 | MEDIUM | 30 min |
| Phase 5: Helpers | 12 | MEDIUM | 1 hour |
| Phase 6: Atomistic | 3 | HIGH | 2-3 hours |
| Phase 7: Whole-Cell | 2 | VERY HIGH | 4+ hours |

---

## Recommended Path Forward

### Immediate (Today)
1. Integrate Category 1 modules (10 files, ~30 min)
2. Integrate Category 2 modules (2 files, ~15 min)
3. Verify all tests pass

### Next Session
1. Integrate evolution engine (Phase 3)
2. Test binary functionality
3. Run full test suite

### Future Sessions
1. Core data layer (Phase 4)
2. Helper modules (Phase 5)
3. Atomistic chemistry (Phase 6)
4. Whole-cell refactor (Phase 7) - BIGGEST EFFORT

---

## Files to Preserve (Copy to Safe Location)

```bash
# These are working, tested files that should be preserved
cp src/fly_metabolism.rs /tmp/salvage_safe/
cp src/organism_metabolism.rs /tmp/salvage_safe/
cp src/field_coupling.rs /tmp/salvage_safe/
cp src/substrate_coupling.rs /tmp/salvage_safe/
cp src/bin/terrarium_evolve.rs /tmp/salvage_safe/
# ... etc
```

---

## Important Discovery

**The untracked files in `src/` have dependencies on code that DOESN'T EXIST in the baseline!**

For example:
- `substrate_coupling.rs` imports `crate::terrarium_world::fly_body_state_from_world_translation` - doesn't exist in baseline
- `terrarium_contact.rs` imports `crate::terrarium_world::TerrariumRaycastBvhNode` - doesn't exist in baseline
- `plant_competition.rs` imports `crate::plant_organism::beer_lambert_transmitted_fraction` - doesn't exist in baseline

**This means Phase 1-2 cannot be done independently!** The files are interdependent with the broken commits.

---

## Revised Strategy

### Option A: Cherry-pick from broken commits
1. Use `git cherry-pick` to selectively apply commits that add infrastructure without breaking
2. Start with data structure commits, then build up

### Option B: Manual extraction
1. Extract just the truly independent code (constants, pure functions)
2. Carefully integrate one module at a time

### Option C: Full integration of broken branch
1. Fix all module declarations in lib.rs
2. Resolve all missing imports
3. This is 4+ hours of work

---

## Recommended Next Steps

1. **First**: Decide which option (A, B, or C)
2. **If Option B**: Start with `organism_metabolism.rs` (trait only, truly no deps)
3. **Then**: Progress to modules with satisfied dependencies
4. **Avoid**: Don't try to add all files at once - they're interconnected
