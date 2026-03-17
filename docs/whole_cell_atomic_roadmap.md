# Whole-Cell Atomic Roadmap

## Goal

Build from the current Rust whole-cell runtime to a genome-explicit, chemically explicit, multiscale microbial simulator where:

1. the organism program is defined from real microbial genome assets
2. molecular species and complexes are explicit runtime state
3. local atomistic chemistry is a native source of truth for selected assemblies
4. whole-cell dynamics are driven by explicit state plus generic multiscale interaction kernels, with any retained reductions compiled from lower-scale dynamics rather than hand-authored process surrogates
5. the system reaches and then surpasses the robustness of the published MC4D/Syn3A stack

## Reality Constraint

A literal full-cell all-atom trajectory for an entire living cell cycle is not the first production target. The practical target is:

- explicit genome and molecule state at the whole-cell level
- explicit stoichiometric assembly state for macromolecular complexes
- atomistic local refinement for the most important assemblies and events
- compiled multiscale reductions that are fit back to those lower-scale simulations

That is the shortest technically credible path from the current codebase to something stronger than the cited Cell/MC4D system.

## Execution Rule

This roadmap is meant to remove the stop-and-go loop.

- Work should continue through these phases in order without asking for a new prompt each time.
- Stop only for a real blocker: missing licensed data, impossible hardware assumption, contradictory biological target, or a decision that would permanently narrow the architecture.

## Non-Negotiable Guardrails

- The experiment is bottom-up. Do not turn scheduler layers, terrarium fields,
  process-scale reducers, or compatibility math into biological source of
  truth.
- If explicit lower-scale state exists for a phenomenon, higher layers may only
  summarize, schedule, or boundary-couple that state. They may not reclaim
  authority with a new aggregate shortcut.
- Compatibility and fallback layers are allowed only as temporary scaffolds in
  uncovered areas. Every such layer should stay visibly subordinate to explicit
  chemistry and remain a candidate for deletion.
- Do not create dual ownership for the same local biology. Explicit chemistry,
  explicit assembly state, explicit operon/chromosome state, and explicit
  microdomain ownership win over coarse overlays.
- The target direction remains `atoms -> molecules -> compounds/materials ->
  genome/macromolecules -> cells -> tissues -> organisms`, not the reverse.

## Starting Point

Current native status already includes:

- Rust whole-cell runtime in `oneuro-metal/src/whole_cell.rs`
- descriptor-driven local chemistry and atomistic probe surfaces in `oneuro-metal/src/whole_cell_submodels.rs`
- Syn3A organism/program specs in `oneuro-metal/specs/whole_cell_syn3a_organism.json` and `oneuro-metal/specs/whole_cell_syn3a_reference.json`
- persistent transcription-unit state, transcript/protein inventories, and assembled-complex state
- compiled genome asset packaging from the Syn3A descriptor into explicit operon/RNA/protein/complex assets carried through the native runtime

## Current Progress

The first Phase 1 slice is now implemented:

- explicit genome asset schema exists in Rust
- bundled Syn3A now compiles into operon, RNA, protein, and complex asset packages
- those compiled assets now hydrate through the program spec and restart payloads
- the runtime now exposes organism asset summaries instead of only raw descriptor counts
- those compiled assets now directly drive named live complex state, aggregate assembly capacity, and restartable Rust/Python inspection surfaces
- those named complex surfaces now include explicit subunit-pool, nucleation, and elongation intermediates, so phase-3 assembly state is no longer only a mature-abundance scalar
- the aggregate assembly cache now projects those staged named-complex states directly into explicit complex channels instead of collapsing them through legacy occupancy/inventory compatibility math
- those compiled assets now also expand into an explicit derived species/reaction registry, so phase-2 chemistry now has a concrete canonical artifact for pools, RNAs, proteins, mature complexes, assembly intermediates, and staged assembly reactions
- that genome-derived registry now also seeds restartable live species-count and reaction-flux state in the native runtime, so the process-registry layer is no longer only an exposed compiled artifact
- that live substrate layer now also uses explicit pool bindings and generic stoichiometric/catalyst execution instead of name-matched pool anchors and reaction-class-specific runtime hints
- that live substrate layer now also derives a generic process-occupancy state which the scheduler reads ahead of the legacy complex summary, so more CME/ODE/BD/geometry support now comes directly from explicit live species/reaction state instead of fixed complex-capacity channels
- the runtime now also carries explicit chromosome fork and locus state through save/restore, and organism-expression copy gain now comes from those chromosome entities instead of only a scalar replicated-fraction shortcut
- that runtime registry now also binds reactions back to explicit operons, and the transcription layer now reads fork support, fork pressure, and strand alignment from chromosome geometry so operon activity and flux depend on explicit fork-feature state rather than copy number alone
- the runtime now also carries explicit operon occupancy/accessibility state, and RNA/protein runtime anchors now read operon-bound flux/extents plus local operon accessibility instead of only transcription-unit aggregate totals
- transcript/protein inventory dynamics now also follow live operon-bound reaction activity plus operon-local occupancy before falling back to stage-level transcription/translation scalars
- named-complex support/satisfaction now also reduce from runtime species counts, operon-local state, and operon-bound reaction activity rather than transcription-unit aggregate fallbacks, and named-complex component abundance / structural-support reducers now prefer explicit runtime operon species/reaction signal over operon heuristics when those direct channels exist
- derived complex-assembly targets now also reduce from live species/reaction occupancy plus operon-local activity rather than transcription-unit abundance totals
- operon runtime drive, support, and refreshed operon-state surfaces now reduce from explicit per-operon species/reaction state when that lower-scale signal exists and only fall back to operon-level heuristic occupancy/copy reductions or prior operon cache when runtime signal is absent; expression-refresh abundance seeding likewise prefers live per-operon species state over the transcription-unit cache
- runtime RNA/protein anchor counts in `sync_runtime_process_species(...)` now also prefer live per-operon species state over transcription-unit projections
- expression refresh now rebuilds its operon/transcription-unit compatibility surface from compiled operon/genome assets before consulting `transcription_units`, and runtime RNA/protein anchoring now preserves live intra-operon distribution instead of re-splitting totals by basal abundance
- inventory dynamics now rebuild transcript/protein abundance and rate fields from operon state plus live runtime species before projecting them back onto the `transcription_units` compatibility surface, and the higher-level process guards now key off operons rather than unit-cache emptiness
- operon state now carries basal/effective/support/stress/fork/inventory/rate metadata as the live authority, expression refresh and inventory dynamics now prefer direct runtime/reaction targets over prior operon cache when live channels exist and otherwise fall back to prior operon metadata instead of unit-cache state, inventory dynamics now also prefers live transcript/protein synthesis and turnover rates over authored rate formulas when those channels exist, and inventory dynamics now rewrites both operon and unit views from that operon authority
- restart/runtime flows now rebuild the `transcription_units` projection from operons when the compatibility cache is missing, so an empty unit cache no longer forces an unnecessary expression refresh when operon state is intact
- the last `unit_*` reducers are now out of expression refresh, with support and inventory scaling reading live operon occupancy, runtime species, reaction extent, and fork geometry directly before contributing to global process scales
- program construction now recompiles genome asset packages from explicit organism descriptors before initializing named-complex state, so descriptor edits stay authoritative even when a stale compiled asset bundle is present in the incoming spec
- scheduler-facing process scales now derive from live occupancy/species/reaction/operon/chromosome state rather than using cached expression-scale summaries as the primary authority, pushing stage capacity closer to explicit substrate state
- the cached `organism_expression.process_scales` surface now resynchronizes from that same live runtime-derived scale path after refresh/init/update transitions, so the compatibility/reporting layer no longer drifts behind the scheduler
- the live `organism_process_scales()` projection now reads directly from the substrate-capacity bridge instead of mostly from authored occupancy reducers, with membrane scaling now explicitly coupled to the membrane-plus-energy live path
- `process_rule_context(...)` now sources ATP/ribosome/RNAP/replisome/membrane/FtsZ/DnaA rule channels from explicit complex assembly state when present, otherwise from direct runtime species/reaction projection when available, and only then from channel-direct occupancy projection, instead of inflating them through legacy inventory backprojections
- the no-organism complex-assembly fallback now seeds directly from live bulk, chromosome, local-chemistry, and quantum state instead of scalar inventory rules, and its derived targets now prefer explicit bulk/support fields over MD recommendation scales while leaving those MD scales as missing-bulk fallback only
- `process_occupancy_state_from_inventory(...)` now keeps inventory-to-occupancy backprojection as fallback only, so live named-complex/runtime occupancy remains authoritative when present
- scheduler/seeded assembly inventory fallback now projects direct complex channels from runtime species/reaction state when explicit assembly caches are absent, leaving only channel-direct occupancy projection as the last-resort compatibility surface
- compiled registry process weights now flow through into live runtime species/reaction state, so runtime occupancy and inventory projection prefer those lower-scale compiled weights before falling back to asset-class templates
- runtime species anchoring plus operon/complex structural-support reduction now also prefer explicit process weights first, then subsystem targets, and only then any asset-class fallback
- registry process-drive reduction now also stays on those explicit projected channels and direct abundance/flux instead of inventing extra process signal or weighting from species-class or reaction-class labels
- the capacity-rule layer now uses direct process signals for energy, transcription, translation, replication, segregation, membrane, and constriction whenever those live signals are present, falling back to the compatibility-era complex slots only when the direct process signal is absent
- `process_fluxes(...)` now takes scheduler-facing capacity directly from the live substrate-capacity bridge whenever that lower-scale signal is present instead of returning or blending rule output
- stage-rule complex signals now prefer explicit assembly state over occupancy-to-legacy backprojection, while the lower-level capacity reducer still reads occupancy directly so live substrate support remains visible below the scheduler
- BD chromosome initiation now also prefers a direct lower-scale initiation signal from runtime occupancy/species/reaction/chromosome state when that signal is present, instead of relying only on the stage-rule `DnaaSignal` surrogate
- CME/ODE/BD stage targets now take direct live substrate drive from runtime species/reaction/operon/chromosome state whenever that lower-scale path is present rather than depending on scalar-rule outputs
- scheduler capacities still prefer live substrate/species/reaction/operon/chromosome signals, and stage flux/drive updates now use those direct lower-scale signals outright when available, demoting the rule layer toward fallback behavior instead of primary execution authority
- geometry membrane growth and constriction-flux targets now blend that same direct substrate drive, while division progression itself remains on the explicit assembly/rule path so descriptor-driven organism differences are not flattened
- the compiled Syn3A reaction registry now carries explicit ATP/ADP pool participants for transcription and translation, and CME now pays ATP/amino-acid/nucleotide bulk costs directly from those live reaction deltas while leaving scalar fallback only for cases where runtime projection is absent
- the compiled Syn3A reaction registry now also derives explicit energy-recharge and nucleotide-salvage reactions from the current asset mix, and ODE now projects glucose/oxygen/ATP/ADP/nucleotide bulk movement directly from live reaction flux while leaving scalar fallback only for cases where runtime projection is absent
- the compiled Syn3A reaction registry no longer constrains mature complexes to a single catalytic asset-class family; mixed-function complexes now emit explicit energy-recharge, nucleotide-salvage, and membrane-precursor synthesis reactions from their compiled process weights, and ODE now projects membrane-precursor bulk movement directly from that live chemistry layer without scalar membrane backfill
- the direct stage-drive bridge now pulls membrane growth signal from explicit membrane reaction flux/extents, membrane species state, and projected membrane-precursor bulk movement before leaning on membrane-capacity summaries, reducing one more geometry dependency on top-down process scaffolding
- geometry-stage membrane depletion now resolves through an explicit membrane materialization limiter tied to live membrane reaction/species state plus current precursor pool, replacing the old fixed area-to-pool drain as the primary authority for realized membrane growth
- geometry now advances surface area directly from explicit membrane reaction/species materialization instead of routing that update through a separate membrane reservoir layer
- geometry no longer reads a membrane-growth rule output either; membrane area change is now controlled only by explicit membrane materialization, and ODE membrane precursor bulk updates now also stay on the explicit runtime reaction path

The next execution target is to keep replacing scheduler shortcuts with explicit state, especially by shrinking the remaining authored process-scale reducers and pushing more replication/transcription behavior through live species, operon-local state, and chromosome entities so stage logic stops leaning on fixed aggregate process channels.

## Tracker

### Accomplished

- Bottom-up architecture guardrails are now explicit in the repo and roadmap, so new work is not supposed to reintroduce top-down biological authority.
- Syn3A organism descriptors now compile into explicit operon/RNA/protein/complex asset packages that flow through runtime initialization, restart, and inspection surfaces.
- The runtime now carries explicit species, reactions, named-complex state, assembly intermediates, chromosome fork/locus state, operon occupancy/accessibility state, and runtime quantum process state.
- Scheduler-facing capacity, stage-drive, bulk-pool, and geometry membrane updates now prefer explicit lower-scale state when it exists; scalar/rule layers are fallback only in the covered paths.
- Membrane geometry growth no longer depends on a membrane-growth rule output, membrane reservoir, or ODE scalar backfill; it now follows explicit membrane reaction/species materialization.
- Runtime occupancy, assembly inventory, rule-context channels, process-scale reduction, and registry drive now prefer explicit process weights and direct lower-scale projection over legacy inventory, class-label, or authored compatibility inflation.
- Live runtime occupancy, assembly inventory, and registry drive now also ignore bare asset-class labels for species/reactions that have no explicit process weights or subsystem targets, leaving those asset-class templates behind as non-runtime fallback only.
- Operon runtime drive and support reduction now use explicit per-operon species/reaction signal as authority and only fall back to promoter/copy/occupancy heuristics when runtime signal is absent.
- Expression refresh and inventory dynamics now rebuild operon promoter/accessibility/RNAP/elongation/blockage surfaces from direct runtime species/reaction targets whenever those live channels exist, leaving prior operon cache as fallback only when the direct channels are absent.
- Inventory dynamics now also uses live transcript/protein synthesis and turnover rates as authority whenever those runtime rate channels exist, leaving the authored operon-rate formulas as fallback only when the direct rate channels are absent.
- Named-complex component abundance and structural-support reducers now also use explicit runtime operon species/reaction signal as authority and only fall back to operon heuristics when those direct channels are absent.
- The no-organism assembly-target path now also prefers explicit bulk/support fields over MD recommendation scales and only uses those MD scales when the corresponding bulk pools are actually absent.
- The substrate-stage constriction bridge, the no-organism seeded assembly fallback, and the geometry-stage constriction gate now treat `division_progress` as fallback-only for constriction capacity/drive, seeded FtsZ assembly, and direct geometry-stage constriction updates, so live membrane/constriction runtime state or explicit bulk/chromosome state cannot be inflated or throttled by late scalar progress when those lower-scale signals already exist.
- Base rule-context process/division scales now also prefer explicit lower-scale bulk/runtime/chromosome state, the surviving septum/division rule surfaces use direct lower-scale state when available, and the dead ATP/ribosome/replisome inventory-only rule signals are gone.
- Stage-rule `DnaaSignal` and `ReplisomeAssemblySignal` now also prefer direct lower-scale replisome scales when explicit bulk/runtime/chromosome state exists, leaving authored replisome subsystem scales as fallback only when those direct channels are absent.
- Stage-rule `ConstrictionSignal` now also prefers direct lower-scale membrane/constriction state when that state exists, leaving FtsZ assembly inventory as fallback only when those direct channels are absent.
- Stage-rule `TranscriptionDriveMix` and `TranslationDriveMix` now also prefer direct lower-scale operon/species/reaction/chromosome drive when that state exists, leaving authored mix formulas as fallback only when those direct channels are absent.
- Runtime-quantum sync now preserves existing runtime-owned microdomain state when no explicit assembly or explicit embedded override exists, rebuilds from live symbolic or explicit embedded sources only when runtime state is actually missing, and keeps runtime-owned embedded microdomains in a cache/fallback role so live assembly updates still win.
- Runtime-quantum fragment-count, replenish, atom-budget, and inventory-floor reducers now zero subsystem compatibility signal whenever explicit assembly, named-complex, species, or reaction state exists, leaving subsystem compatibility as a no-runtime fallback only.
- Compiled operon process-weight reduction now also ignores bare RNA/protein/complex asset-class labels when those compiled assets carry no explicit process weights or subsystem targets, so operon drive/support inherit only explicit lower-scale weights or subsystem targets from the compiled asset package.
- Compiled registry-drive reduction now also ignores bare asset-class labels for derived species/reactions when those compiled entries carry no explicit process weights or subsystem targets, so non-runtime registry drive no longer authors process channels from class labels alone.
- Regression coverage has been added around those bottom-up cuts so explicit process weights, subsystem targets, runtime assembly inventory, and live stage-drive authority are locked down by tests.
- Pure whole-cell reducers are now being pulled out of `oneuro-metal/src/whole_cell.rs` into focused sibling modules, starting with process-weight helpers, assembly-inventory projection (including explicit runtime species/reaction reduction), process-occupancy projection (including explicit runtime species/reaction reduction), chromosome/circular-genome utility math, generic scale/operon-signal reducers, resource/subsystem signal-estimator tables, complex-channel assembly step helpers, named-complex seed/update/aggregation dynamics, rule-signal/blend math, base/stage rule-context setters, compiled-asset fallback reducers, bare lower-scale assembly inventory math, fallback assembly-target/seeding math, fallback assembly support reducers, fallback assembly channel-advance math, and assembly-inventory authority/diagnostic reducers so new logic does not keep accreting into the monolith.

### Remaining For The Vision

- The runtime is still not fully chemistry-explicit for metabolism, membrane/division mechanics, chromosome mechanics, and atomistic feedback at MC4D-like depth.
- Most operon/expression and stage-drive authored reductions are now fallback-only; the remaining work is concentrated in late division/chromosome mechanics, residual no-runtime compatibility surfaces, surviving non-runtime asset/template fallbacks, and deeper chemistry-explicit metabolism/membrane behavior that still has no explicit lower-scale authority.
- The atomistic refinement path exists, but it still needs stronger automatic domain construction, parameter ingestion, orbital-response coupling stability, and calibrated feedback into whole-cell rates.
- Data/calibration infrastructure is still shallow relative to the target vision: reference datasets, residual reporting, held-out validation, and uncertainty tracking are not complete.
- Cross-organism compilation, robust multirate solver orchestration, and cluster-scale execution are still future work.

### Current Task Inventory

- `[in progress]` Remove the remaining late-division, chromosome, no-runtime compatibility, and non-runtime asset/template fallback seams plus any leftover authored process-drive surfaces that still sit above explicit operon/species/reaction/chromosome state.
- `[in progress]` Keep whole-cell verification green while codegen and quantum/atomistic/terrarium APIs evolve underneath the runtime.
- `[in progress]` Keep breaking pure helper/reducer clusters out of `oneuro-metal/src/whole_cell.rs` and `oneuro-metal/src/terrarium_world.rs` into focused modules so bottom-up logic stays inspectable and testable instead of expanding the monoliths.
- `[todo]` Continue the modularity pass with the next low-risk pure slices: residual non-runtime fallback reducers and remaining runtime stage-rule projection scaffolding that still sit inline in `oneuro-metal/src/whole_cell.rs`.
- `[todo]` Reduce the remaining fallback-only late division, chromosome, and residual no-runtime compatibility heuristics that still infer subsystem state after explicit bulk/runtime assembly authority is exhausted.
- `[todo]` Replace remaining scheduler/division shortcuts with explicit divisome assembly, membrane remodeling, chromosome occlusion, and late-stage constriction mechanics.
- `[todo]` Finish pushing metabolism, transport, degradation, repair, and membrane synthesis onto explicit compiled reaction networks with explicit enzyme/state dependence.
- `[todo]` Deepen explicit chromosome mechanics: topology, compaction, tethering, collision handling, and fork-domain dynamics.
- `[todo]` Deepen explicit membrane composition and inserted-protein state so geometry and transport depend on real membrane material, not coarse capacity summaries.
- `[todo]` Expand atomistic local-domain builders, force-field ingestion, and atomistic benchmark coverage for ribosome, replisome, ATP synthase, membrane insertion, and divisome neighborhoods.
- `[todo]` Feed atomistic outputs back as calibrated rates, energies, conformational states, diffusion modifiers, and assembly/degradation propensities.
- `[todo]` Build the Syn3A reference dataset bundle, calibration pipelines, held-out validation suites, residual reports, and uncertainty analysis.
- `[todo]` Reproduce the published Syn3A observables that matter most, then exceed them on explicit assembly/division/atomistic fidelity.
- `[todo]` Compile at least one additional microbial organism through the same asset and runtime pipeline so the system stops being Syn3A-specialized.

## Phase 0: Freeze The Architecture Contract

1. Define the primary target organism set, starting with `JCVI-syn3A` and one second microbial reference with a larger annotated genome.
2. Freeze the whole-cell unit system for counts, concentrations, lattice geometry, energies, forces, and timestamps across every Rust subsystem.
3. Define the canonical runtime state layers: atoms, molecules, complexes, genome features, chromosomes, membranes, compartments, and observables.
4. Define the canonical IR layers: chemistry graph IR, assembly graph IR, genome/transcription IR, chromosome/polymer IR, and solver schedule IR.
5. Freeze the serialization contract for restart, provenance, dataset hashes, and reproducibility metadata.
6. Add a roadmap progress tracker that maps each implementation milestone to tests, datasets, and benchmark outputs.

## Phase 1: Make The Organism Program Explicit

7. Add an organism asset pipeline that ingests `FASTA`, `GenBank`/`GFF`, protein products, operons, transcription units, promoters, terminators, and essentiality annotations.
8. Normalize those assets into Rust-owned specs for genes, RNAs, proteins, complexes, reactions, and compartments.
9. Add explicit operon/polycistronic transcription support instead of treating transcription units as shallow labels.
10. Add genome feature classes for coding genes, ncRNAs, promoters, operators, ribosome-binding sites, origins, termini, and structural motifs.
11. Add sequence-derived feature extraction for GC content, codon usage, motif density, replication timing priors, and transcription directionality.
12. Add a build step that compiles those assets into bundled organism packages instead of hand-maintained JSON descriptors.

## Phase 2: Make Chemistry Explicit

13. Add a canonical species registry for metabolites, ions, cofactors, lipids, nucleotides, amino acids, RNAs, proteins, complexes, and membrane species.
14. Add molecular graph support for explicit small-molecule chemistry using standardized identifiers plus graph/topology payloads.
15. Add reaction network ingestion for metabolism, gene expression, transport, membrane synthesis, degradation, repair, and division chemistry.
16. Replace remaining high-level flux surrogates with compiled reaction sets wherever whole-cell rates are still hand-authored.
17. Add enzyme-linked reaction execution where rates depend on explicit enzyme abundance and state rather than aggregate process scales.
18. Add explicit molecule-count bookkeeping for every compiled species participating in genome, metabolism, and assembly dynamics.

## Phase 3: Make Assembly Explicit

19. Add a first-class macromolecular assembly graph runtime for ribosomes, RNAP, replisomes, ATP synthase, transporters, membrane enzymes, and FtsZ/divisome components.
20. Add stoichiometric assembly interactions, subunit recruitment, failure states, incomplete intermediates, and turnover/degradation paths.
21. Replace aggregate complex classes with named complex inventories and named assembly intermediates.
22. Tie protein inventories to actual assembly demand, competition, and sequestration instead of only scaling target pools.
23. Add chaperone-assisted folding, misfolding, stress damage, and repair pathways for key complexes.
24. Add assembly-aware capacity exports so CME/ODE/BD stages consume concrete named complex state instead of generic readiness channels.

## Phase 4: Make Chromosomes And Membranes Explicit

25. Add explicit circular chromosome sequence state with binding sites, fork positions, replication initiation logic, and collision-aware transcription/replication bookkeeping.
26. Add polymer-level chromosome dynamics with native Rust data structures for loci, domains, tethering, compaction, and segregation.
27. Add DNA topology state for supercoiling, strand separation cost, and local torsional stress where it materially changes expression or replication.
28. Add explicit membrane composition state with lipid classes, inserted proteins, curvature fields, and septum-local membrane remodeling.
29. Add divisome assembly state and geometry coupling so constriction follows assembled divisome mechanics rather than only scalar progress shortcuts.
30. Add cell-shape and growth updates that are driven by membrane synthesis, wall mechanics if applicable, osmotic load, and chromosome occlusion constraints.

## Phase 5: Make Atomistic Chemistry A Native Truth Source

31. Add a Rust-native topology/parameter pipeline for local atomistic systems: proteins, nucleic-acid segments, protein-RNA complexes, lipids, metabolites, ions, and membrane patches.
32. Add force-field ingestion and validation for bonded, electrostatic, van der Waals, and solvent/implicit-environment terms needed by the local MD path.
33. Add automatic local-domain builders that carve atomistic subsystems out of whole-cell state around ribosomes, replisomes, ATP synthase bands, membrane insertion sites, and FtsZ/division zones.
34. Add adaptive local refinement triggers based on uncertainty, rare events, instability, or major assembly transitions.
35. Add atomistic outputs that feed back into the whole-cell runtime as calibrated rates, energies, conformational states, diffusion changes, and assembly/degradation propensities.
36. Add a library of validated atomistic microbenchmarks so local MD is measured against known structures and not just used as an internal heuristic.

## Phase 6: Make The Solver Stack Properly Multiscale

37. Replace the current staged scheduler with an explicit multirate orchestration layer that supports RDME, CME, ODE, BD/polymer, membrane mechanics, and atomistic refinement as separate clocks.
38. Add event-driven scheduling for replication initiation, fork collisions, division checkpoint transitions, assembly completion, and rare chemistry events.
39. Add uncertainty-aware coupling so lower-scale outputs update higher-scale reductions only when they materially change predicted behavior.
40. Add native checkpoint/restart slices for each solver layer so long-running whole-cell jobs can be resumed without losing multiscale consistency.
41. Add deterministic replay and provenance capture for random seeds, spec versions, compiled bundles, and calibrated reduction snapshots.
42. Add distributed execution support for larger local-atomistic workloads while keeping the Rust whole-cell core authoritative.

## Phase 7: Make The Data And Calibration Story Real

43. Build a reference dataset bundle for Syn3A: genome, transcript/protein measurements, metabolite concentrations, growth curves, replication timing, division timing, and perturbation data.
44. Build calibration pipelines that fit whole-cell reductions against experimental data, local chemistry outputs, and atomistic refinement outputs separately.
45. Add held-out validation suites so fitting cannot silently overfit the same reference observables.
46. Add per-module residual reports for expression, metabolism, chromosome behavior, geometry, and division.
47. Add uncertainty bands and sensitivity analysis so we know which subsystems dominate prediction error.
48. Add regression suites that compare native outputs against the published MC4D baselines wherever observables overlap.

## Phase 8: Reach MC4D Parity

49. Reproduce the published Syn3A observables that matter most: cell-cycle timing, gene-expression distributions, metabolic support, chromosome behavior, and division progression.
50. Match or exceed MC4D restartability, artifact completeness, dataset packaging, and reproducibility.
51. Match or exceed their solver coverage for genome expression, metabolism, chromosome dynamics, and geometry updates at the observable level.
52. Replace remaining coarse whole-cell heuristics that have no explicit lower-scale or data-driven justification.
53. Add end-to-end benchmark runs that report compute cost, fidelity, uncertainty, and calibration status together.

## Phase 9: Surpass MC4D

54. Add explicit operon/polycistronic mechanics, named assembly intermediates, and local atomistic truth in places where MC4D still relies on coarse abstractions.
55. Add deeper division physics: divisome assembly order, membrane remodeling, chromosome occlusion, and late-stage constriction failure modes.
56. Add better membrane biophysics, transport, and protein insertion than the current published whole-cell reference.
57. Add explicit genotype-to-phenotype perturbation workflows so genome edits can propagate through chemistry, assembly, geometry, and fitness automatically.
58. Add cross-organism generalization by compiling a second and third microbial organism through the same pipeline instead of hard-specializing to Syn3A.
59. Add high-throughput design loops for genome edits, media changes, and drug perturbations using the same substrate-first runtime.

## Phase 10: Push Toward Broader Atomistic Coverage

60. Expand atomistic refinement from selected hotspots to a larger fraction of the proteome and membrane machinery as compute allows.
61. Add automatic surrogate compilation from repeated atomistic neighborhoods so expensive local chemistry becomes reusable learned physical reductions.
62. Add adaptive partitioning that can promote or demote a region between coarse and atomistic treatment during a live run.
63. Add hardware-specialized backends in Rust for Metal and CUDA compute kernels rather than moving the hot loop back into Python.
64. Add cluster-scale job orchestration for large multiscale runs, with artifact ingestion kept inside the existing `oNeura` experiment model.

## Immediate Build Order From The Current Codebase

65. Add a compiled genome asset schema and loader for explicit gene, operon, RNA, protein, and complex definitions.
66. Replace the current bundled Syn3A descriptor with a compiled asset package produced by that loader.
67. Add named complex inventories and assembly intermediates on top of the new complex state already present in `oneuro-metal/src/whole_cell.rs`.
68. Add a reaction compiler that maps genome/program assets into explicit species and reaction registries.
69. Add the first explicit chromosome/fork state layer.
70. Add the first atomistic topology/parameter ingestion path for a real Syn3A subsystem rather than template-only probes.

## Exit Criteria For The Program

We should consider the roadmap complete only when all of the following are true:

- the organism is compiled from explicit microbial genome assets
- whole-cell state contains explicit named molecules and complexes
- at least the key high-value subsystems are grounded by native atomistic refinement
- the whole-cell runtime can reproduce and checkpoint a full validated microbial cell cycle
- the system matches or exceeds the cited MC4D stack on robustness and observable fidelity
