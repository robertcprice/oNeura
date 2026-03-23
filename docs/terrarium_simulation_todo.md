# Terrarium / Aquarium Simulation Todo

This list is the active realism and restartability tranche for the native terrarium stack.

## In Progress

- [x] Add seed provenance and natural entropy seeding for interactive terrarium entrypoints.
- [x] Add a macro-ecology checkpoint format that can restore explicit world state without pretending exact neural replay.
- [x] Upgrade fly checkpointing from construction-seed rebuilds to exact neural/body replay from serialized lower-scale state.
- [x] Make checkpoint JSON save/load work for explicit material inventories instead of only in-memory checkpoint structs.
- [x] Exercise checkpoint save/load from the CLI with natural entropy provenance preserved in the saved record.
- [x] Preserve lineage/name/climate/archive metadata in saveable world records.
- [x] Add explicit whole-cell / explicit-microbe checkpoint support so restart fidelity does not drop those cohorts.
- [x] Add packet-population and secondary-genotype-bank checkpoint coverage for live genotype-region authority.
- [x] Add atomistic probe checkpoint support from live MD state instead of dropping embedded probe regions.
- [x] Replace resume-seed checkpoint fallbacks with serialized live stochastic-engine state for terrarium world RNG, whole-cell stochastic expression, fly/population RNGs, and MD Langevin noise.

## Next Highest-Value Work

- [x] Add CLI and desktop UX for naming organisms, querying lineage, and saving/loading archives/checkpoints without manual file editing.
- [x] Add live web-shell UX for browsing organisms, querying lineage, renaming tracked organisms, and exporting archives from the served terrarium frontend.
- [x] Add canonical save/load tests that compare pre-save and post-load world observables after a short replay window.
- [x] Add an in-app web-shell transfer console and non-blocking operator notices so save/restore/archive flows stay visible during live demos.
- [x] Add drag-and-drop JSON import in the web shell with auto-detection for checkpoint restore, archive inspection, and replay bundles.
- [x] Add a browser-side chooser for dropped multi-payload bundles so operators can explicitly pick live restore, archive inspection, or replay import.
- [x] Reuse the same multi-payload bundle chooser for file-picked checkpoint/archive/replay imports so drag-and-drop and picker flows stay consistent.
- [x] Add browser-side transfer progress reporting for large imports and exports, including file read, JSON parse, download, upload, and post-restore sync stages.
- [x] Add bundle/checkpoint/archive metadata previews in the chooser and transfer log so operators can see payload mix, size, fidelity, preset, seed, tracked organisms, replay coverage, and climate-year context before or after a transfer.
- [x] Add a real browser-level import/export smoke harness for the demo shell so archive inspect, bundle replay, and bundle restore are validated end to end against the live `terrarium_web` server.
- [x] Extend the browser smoke harness to cover persisted layout state, chooser-driven restore confirmation, archive-inspect reset guards, and share/recent-world state, and fix shell startup so a fresh tab does not clobber saved layout prefs during hydration.
- [x] Add role-focused surface presets for the demo shell, thread them through persistence/share URLs/guide actions/command palette, and verify that fresh tabs preserve the selected surface without clobbering the saved layout.

## Scientific Fidelity Work

- [x] Add first-pass conservation audits for carbon, nitrogen, phosphorus, oxygen, water, and energy-equivalent pools, with explicit substrate/inventory/atmosphere totals separated from coarse state proxies.
- [x] Expose live web-shell export surfaces for archive and restart-grade checkpoint downloads, and include checkpoint state in the full demo export bundle.
- [x] Add live web-shell checkpoint restore from uploaded checkpoint/bundle JSON, plus archive inspection loading without pretending archives are exact world restarts.
- [x] Clarify live-vs-archive mode in the web shell, add in-flight import/export operator feedback, and make checkpoint HTTP export/import round-trip directly through the served demo server.
- [x] Reorganize the web-shell stats rail into overview/operations/biology/chemistry/history clusters and tighten the visual shell so the demo UI reads cleaner under live use.
- [x] Add first-class keyboard navigation and shortcut discoverability to the web shell, including camera movement, zoom, tool hotkeys, and a visible shortcut overlay.
- [x] Add a searchable command palette and reliable focus-safe hotkeys so the demo shell can be driven quickly without relying on dense visible controls.
- [x] Persist the demo-shell layout across reloads and add confirmation guards around destructive live-world replacement actions like checkpoint restore.
- [x] Add shareable preset/seed/layout URLs plus recent-worlds quick relaunch controls so operators can recover and hand off exact demo states.
- [ ] Tighten conservation audits by replacing coarse proxy domains with lower-scale authoritative chemistry as more organism and detritus state moves downward.
- [ ] Push more plant authority into explicit tissue/cellular state so morphology, fruiting, senescence, and abscission depend less on coarse coupled summaries.
- [ ] Replace remaining compatibility reward/search surfaces with explicit sensor chemistry and body-contact pathways.
- [ ] Extend the same atoms-first authority tightening to the aquarium path, especially producer/decomposer/consumer coupling and sediment chemistry.
- [ ] Add explicit litter architecture / particle state if burial and resurfacing dynamics need to move below the current litter-surface summary layer.

## Identity / Provenance Work

- [ ] Extend organism identity to any fauna that becomes individually modeled instead of population-bucketed.
- [ ] Persist user-assigned names and lineage queries through archive/checkpoint round-trips in every frontend.
- [ ] Add year/location/scenario provenance to world start records and expose it in UI snapshots.

## Boundary Notes

- Macro checkpoint coverage now includes fly neural/body state, explicit whole-cell microbe state, packet/genotype-region state, secondary genotype banks, ownership state, and atomistic probe topology/kinematics.
- Exact restart now preserves the live stochastic-engine state for terrarium world RNG, whole-cell stochastic expression, fly/population RNGs, and atomistic MD Langevin noise.
- Remaining exact-replay limits are now lower-level and backend-specific: cross-version stability of serialized third-party RNG layouts, GPU/CPU floating-point drift, and any solver state that still lives outside the explicit checkpoint contracts.
- New high-level behavior shortcuts should not be added to “fill gaps” in these systems. The right move remains pushing authority downward into explicit state.
