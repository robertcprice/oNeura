# Subatomic And Quantum Execution Plan

This document tracks the explicit layer below the Rust atomistic chemistry
stack and how it is being threaded into whole-cell execution.

## Architecture Rule

The authority chain for this tranche is:

`quarks -> nucleons -> nuclei -> electron subshells/orbitals -> atoms -> molecules -> materials -> cells`

Higher layers may summarize, schedule, or boundary-couple this state. They may
not invent competing top-down authority for the same local chemistry.

## What Exists Now

### Native Lower-Scale Core

`oneuro-metal/src/subatomic_quantum.rs` now provides:

1. Explicit quark and nuclear composition under each atom.
2. Explicit electron subshell filling with approximate shielding.
3. Valence active-space construction from native `MoleculeGraph`.
4. Bond-topology-aware interacting electronic Hamiltonians.
5. Fixed-particle exact diagonalization for bounded active spaces.
6. Explicit quantum reaction deltas before/after structural edits.
7. Solved observables used for coupling upward:
   - spatial occupancies
   - atom effective charges
   - fragment dipole moments
   - one-particle density response between atom pairs

### Native Atomistic Integration

`oneuro-metal/src/atomistic_chemistry.rs` now uses that layer for:

1. `EmbeddedMolecule` exact diagonalization.
2. Embedded structural reaction execution with explicit `QuantumReactionDelta`.
3. Embedded material-mixture transforms and structural reactions.

### Native Whole-Cell Integration

`oneuro-metal/src/whole_cell.rs` and
`oneuro-metal/src/whole_cell_quantum_runtime.rs` now provide:

1. Persistent explicit runtime quantum microdomains for selected subsystem
   chemistries.
2. Save/restore of those runtime quantum processes.
3. Automatic queueing of quantum summaries into live local chemistry.
4. Multi-fragment runtime microdomains for supported sites.
5. Self-consistent fragment coupling with explicit charge/dipole embedding.
6. Solved-density-derived bond-local polarization dipoles for fragment boundary
   exchange.
7. Pre-solve orbital-response coupling that perturbs local hopping terms before
   diagonalization.
8. Live-state-driven fragment count, fragment atom budget, and reactant
   replenishment floors for runtime quantum microdomains.
9. Primitive/law-based fragment anchor selection and fragment growth inside the
   existing runtime atom graphs instead of coordinate-ranked scaffold cuts.
10. Primitive/law-based placement of reactive molecules from the selected local
    chemical primitive and outward bond geometry instead of template-edge
    offsets.
11. Same-site scaffold expansion/contraction now preserves existing reactive
    inventory and microdomain chemistry instead of resetting the process from a
    fresh template fragment when the live atom budget changes.
12. Bare simulator startup and empty runtime-quantum restore payloads no longer
    pre-seed template-default microdomains before live whole-cell ownership
    exists.
13. Reactive runtime replenishment now gates on live bulk-pool ownership, and
    unsupported fragment applications skip instead of recreating missing
    reactive substrates from runtime-only defaults.
14. Covered whole-cell hotspot reactions now scale their realized runtime flux
    from explicit runtime quantum microdomain execution instead of always
    executing at full coarse extent and only emitting a quantum sidecar.
15. Dual-target runtime hotspot reactions now split requested quantum extent
    across their explicit local site kinds instead of cloning full demand into
    each targeted microdomain.
16. Live symbolic whole-cell state now compiles into a local embedded
    microdomain before template-only runtime seeding, using named-complex,
    species, and reaction ownership as the material source and the site
    template only as fallback geometry.
17. Missing subsystem embedded microdomains are now materialized from that
    live symbolic compilation before runtime process rebuild, so the explicit
    local microdomain becomes the reusable authority instead of recompiling a
    transient symbolic seed on every sync. Shared presets aggregate their
    compatible runtime chemistry inputs into one owned local mixture and now
    preserve distinct kind-local scaffold families instead of collapsing them
    into one surrogate scaffold.
18. Runtime quantum chemistry now promotes reacted local molecule/material
    state back into the subsystem-owned embedded microdomain, so missing
    runtime-process payloads can rebuild from the last explicit reacted state
    instead of only from the original seed or coarse live assembly.
19. Runtime-owned embedded microdomains now outrank live symbolic projection
    on rebuild when the live runtime process payload is missing, but they no
    longer override already-live process state during ordinary sync; live
    processes remain authoritative in place.
20. Missing runtime process payloads rebuilt from an explicit embedded
    microdomain now preserve multiple owned local scaffold molecules instead of
    collapsing the local neighborhood to one duplicated scaffold source.
21. The remaining symbolic/runtime fallback scaffold source now prefers the
    existing atomistic site-template export for each subsystem kind, so the
    fallback geometry is site-local atomistic structure instead of a detached
    two-primitive placeholder. The disconnected primitive source remains only
    as a last-resort fallback when a site template cannot be exported.
22. When explicit embedded microdomains are absent and live whole-cell symbolic
    ownership is the only source, runtime rebuild now compiles exact local
    fragment processes directly from that live symbolic inventory and only then
    promotes the resulting explicit fragment mixture back into owned subsystem
    state. The live runtime path no longer has to detour through a transient
    symbolic embedded microdomain just to recreate the same fragment processes.
23. When no subsystem embedded microdomain exists yet, that same live symbolic
    support still auto-materializes a runtime-owned explicit embedded
    microdomain during sync, but newly materialized symbolic seeds no longer
    override same-sync per-kind rebuilds. Direct symbolic compilation now seeds
    every covered shared-preset chemistry first, then the combined runtime
    result is promoted back into owned local state.
24. When a preset already has live runtime quantum processes and later grows to
    a higher desired fragment count, runtime-owned explicit embedded
    microdomains now contribute only the additional same-kind processes. The
    already-live processes remain authoritative instead of being rewritten by
    the growth scaffold source.
25. Initial runtime-owned microdomain seeding now first compiles an explicit
    embedded mixture from live structural whole-cell owners: targeted
    `named_complexes` and targeted runtime species are lowered into owner-tagged
    scaffold families, while canonical reactive molecules are seeded from the
    same live lower-scale support. The older scalar symbolic seed path remains
    only as fallback when no structural owner exists yet.
26. Those initial owner-tagged scaffold families now prefer explicit local
    fragment families carved from the exported atomistic site templates rather
    than disconnected primitive motif compilation. Complex stoichiometry,
    protein lengths, species stage, and live abundance still size how many of
    those local families each live owner contributes, so the entry path stays
    tied to explicit atom graphs even before fully live-owned molecular
    neighborhoods exist.
27. Live owner seeding now writes those local owner fragments directly into the
    owned subsystem microdomain instead of first collapsing each owner into one
    disconnected aggregate scaffold molecule. Runtime rebuild therefore starts
    from the same local fragment family the lower quantum layer actually uses,
    rather than adding and then undoing an extra scaffold aggregation step.
28. Explicit embedded microdomains now keep their seeded scaffold/reactant
    inventories through sync instead of being overwritten by compatibility
    inventory floors, and same-kind reactants are partitioned across multiple
    owned scaffold families by per-reactant site placement anchors rather than
    whole-scaffold centroid distance.
29. Explicit subsystem embedded transforms and structural reactions now mutate
    an already-owned `EmbeddedMaterialMixture` in place when that lower-scale
    state owns the reactants. Untouched local neighborhood molecules therefore
    remain authoritative across explicit chemistry edits instead of being
    discarded and replaced by products-only subsystem state.
30. Those explicit subsystem embedded edits now also resolve graph-equivalent
    reactants from the owned embedded microdomain and run the quantum event on
    that owned local geometry and inventory when present. A caller-supplied
    reactant copy therefore no longer overrides the subsystem-owned lower-scale
    state just because its coordinates or graph label differ.
31. The lower atomistic `EmbeddedMaterialMixture` layer now resolves
    graph-equivalent embedded reactants directly, and whole-cell now exposes
    subsystem-owned embedded material-transform and material-structural-reaction
    entry points with explicit extent. Local subsystem chemistry therefore no
    longer needs to route only through single-event caller-molecule wrappers to
    stay connected to the authoritative owned microdomain.
32. The legacy single-event subsystem embedded wrappers now delegate into that
    owned material layer whenever a subsystem microdomain exists. Once local
    chemistry is owned below, missing reactants therefore fail against the
    owned microdomain instead of silently falling back to caller-supplied
    embedded copies and overwriting subsystem-owned state.
33. Asset-backed live explicit owner seeds now build runtime quantum scaffold
    components from owner composition itself before any site-template fallback.
    When the whole-cell runtime already has explicit protein/complex
    composition, the first runtime-owned microdomain for that owner therefore
    starts from an owner-derived fragment family instead of inheriting the site
    template scaffold family.
34. Initial owner/live-state microdomains now also lower live bulk/reaction
    support into explicit representative local molecules such as amino-acid-,
    ATP/ADP-, nucleotide-, glucose/oxygen-, and membrane-precursor-like
    neighborhoods instead of seeding only the older canonical primitive
    reactants. Runtime rebuild now maps those live support molecules back into
    canonical local primitive inventory while preserving the support molecules
    themselves inside the embedded microdomain, and support molecules no longer
    get mis-ranked as scaffold sources during rebuild.

## Current Focus

The next upgrade is to move from live-state-driven template reconfiguration to
true live-state-carved atomistic neighborhoods owned directly by explicit
whole-cell molecular state.

Current in-flight slice:

1. Keep live-state-driven runtime fragment resizing inside the current ED
   budget.
2. Replace the remaining compiled representative scaffold/support neighborhoods
   with atom graphs carved from explicit live assembly/material neighborhoods
   where that ownership exists.
3. Preserve the current charge/dipole/orbital-response self-consistent coupling
   path when those live-carved neighborhoods arrive.
4. Keep runtime fragment carving tied to explicit chemical primitives and graph
   laws rather than template-layout heuristics.
5. Keep scaffold resizing subordinate to existing lower-scale state rather than
   resetting chemistry whenever the fragment budget changes.
6. Keep runtime microdomain creation subordinate to live state instead of
   silently restoring template-default processes when no explicit microdomain
   exists; where fallback geometry is still needed, keep it on the exported
   atomistic site templates rather than detached primitive representatives.
   was saved.
7. Keep rebuild precedence ordered so missing runtime processes recover from
   the last explicit owned microdomain before falling back to live symbolic
   compilation.
8. Keep reactive substrate replenishment subordinate to actual live pool
   support instead of runtime-only fallback amounts.
9. Keep covered reaction execution subordinate to explicit runtime quantum
   realization instead of treating the quantum layer as reporting-only.

## Current Hard Limits

1. The native ED kernel is bounded by CIPSI_MAX_VARIATIONAL_DIM=8000 and
   MAX_SPATIAL_ORBITALS=48 — large enough for realistic fragment chemistry
   but not full-enzyme active sites.
2. Runtime quantum regions are still bounded local fragments rather than large
   joint subsystem solves.
3. Live-state quantum carving now modulates corrections from cell state every
   100 steps, but the underlying fragment geometries still come from template
   scaffolds where explicit atomistic ownership is absent.
4. 12/17 reaction classes are now quantum-authoritative (up from 5/17), but
   5 classes remain unclassified (BulkPoolDecay, BulkPoolSynthesis,
   BulkPoolInterconversion, EnvironmentExchange, RegulatoryBinding).

## Next Execution Steps

### Completed (2026-03-16)

1. ✅ Expand the native ED headroom or add a larger-fragment fallback path.
   - CIPSI_MAX_VARIATIONAL_DIM=8000, MAX_BASIS_SIZE=2048, MAX_SPATIAL_ORBITALS=48
   - Per-kind fragment atom budgets lifted: 12/10/8/10/10 (was 8/6/4/6/4)
2. ✅ Promote runtime site builders from representative templates to live explicit
   atomistic neighborhoods where the owning assembly/material state exists.
   - `refresh_quantum_corrections_from_live_state()` reads complex assembly counts,
     subsystem activity scalars, and bulk energy charge to modulate discovered
     quantum reaction correction factors every 100 steps.
   - `refine_quantum_corrections_from_probe()` maps MD probe thermodynamics
     (thermal stability, structural/electrostatic order) to 5 quantum efficiency
     channels with α=0.12 exponential blending.
3. ✅ Expand the supported quantum-authoritative reaction families.
   - Auto-discovery broadened from 5/17 to 12/17 reaction classes.
   - 9 quantum hotspot kinds (7 original + AtpBandElectronTransfer +
     MembraneProteinInsertion).
   - Surrogate pool diagnostics fast-path extended to include named_complexes
     and complex_assembly inventory.

### Immediate (Next)

4. Add broader Rust-native topology/parameter ingestion for real biomolecular
   subsystems.
5. Add local MD under the same authority chain for structured atomistic
   neighborhoods.

### After That

6. Feed larger parts of whole-cell local chemistry directly from explicit
   molecule/material state.
7. Continue removing authored scheduler/process surrogates where lower-scale
   explicit state is already present.

## Out Of Scope For This Layer

1. Full QCD field evolution.
2. Full relativistic QED for the whole simulator.
3. Full-cell all-electron trajectories.
4. Replacing the Rust-native bottom-up runtime with Python-side authority.
