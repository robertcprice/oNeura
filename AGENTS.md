# oNeura Agent Instructions

## Architecture Contract

This repository is building an atoms-first, bottom-up simulation stack.

The intended causal direction is:

`atoms -> molecules -> compounds/materials -> macromolecules/genome state -> cells -> tissues/neural systems -> organisms/ecosystems`

Do not reverse that direction by making higher-level wrappers, schedulers,
terrarium fields, process scales, or compatibility reducers the scientific
source of truth when a lower-scale explicit model exists or is being added.

## Hard Rules

- No dual authority. If explicit chemistry, explicit cell state, explicit
  genome state, or an explicit microdomain owns a phenomenon, coarse overlays
  may not also be authoritative for that same local state.
- Compatibility, fallback, and scalar-rule paths are temporary scaffolds only.
  Keep them clearly labeled, keep them subordinate to explicit state, and do
  not expand them into new biological authority.
- When explicit lower-scale state exists, higher layers may reduce, summarize,
  schedule, or boundary-couple it. They may not overwrite it with new top-down
  surrogate state.
- Prefer removing or isolating top-down shortcuts over extending them.
- Prefer extracting cohesive logic into dedicated modules instead of continuing
  to grow monolithic files like `whole_cell.rs` and `terrarium_world.rs`.
- When a change touches a giant file, first check whether the new logic can
  live in a sibling module with a narrow responsibility, and prefer that
  extraction over adding another long in-file section.
- Do not describe a hybrid scaffold as "bottom-up" unless the explicit
  lower-scale layer is actually authoritative.
- If a requested change would introduce a new top-down authority path, stop and
  ask instead of implementing it silently.

## Required Reading By Area

- Whole-cell, chemistry, atomistic, molecular dynamics:
  `docs/atoms_first_execution_plan.md`
- Whole-cell roadmap and multiscale direction:
  `docs/whole_cell_atomic_roadmap.md`
- Terrarium, ecology, explicit microbial regions, plant regions:
  `docs/terrarium_bottom_up_execution_plan.md`

## Working Default

When in doubt, move authority downward into explicit state and keep coarse
layers as boundary conditions, accelerators, or reporting projections.
