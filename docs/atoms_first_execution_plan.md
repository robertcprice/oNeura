# Atoms-First Execution Plan

This document is the source of truth for the bottom-up path from atoms to
molecules, compounds, materials, cells, and only then higher-level ecosystems.

## Current Truth

The repo now has four distinct layers:

1. `oneuro-metal/src/atomistic_chemistry.rs`
   - Explicit periodic elements
   - Atom nodes
   - Bond graph
   - Molecule graph
   - Compound/material mixture bookkeeping
   - Molecular formula, empirical formula, mass, charge, stoichiometric inventory
   - MD compatibility checks against the current Rust MD engine

2. `oneuro-metal/src/molecular_dynamics.rs`
   - Real atomistic MD engine
   - Fixed supported MD element subset only
   - Non-reactive force-field dynamics

3. `oneuro-metal/src/atomistic_topology.rs`
   - Template-driven local atomistic assemblies
   - Useful for structured microdomains
   - Not a general molecule/material construction engine

4. `oneuro-metal/src/terrarium.rs`
   - Coarse batched chemistry lattice
   - Not atomistic world authority

## nQPU Boundary

`nQPU` is not the missing bottom-up chemistry engine.

What it can contribute when present:
- molecule helper / SMILES-sidecar surfaces
- small-molecule quantum helpers
- tunneling / channel / folding correction terms

What it does not currently provide:
- authoritative atom graph runtime
- general reactive chemistry
- compound/material synthesis engine
- full atom -> molecule -> compound -> material -> cell authority

On this machine, `nqpu_metal` is not installed in the active Python environment.

## Locked Execution Order

### T1. Atom/Bond/Molecule Foundation
Status: done

- add explicit periodic element registry
- add atom nodes and bond orders
- add molecule graph bookkeeping
- add compound/material mixture bookkeeping
- add mass / charge / formula / stoichiometric inventory tests

### T2. MD Coupling Surface
Status: done

- add graph-to-MD conversion helpers for supported elements
- add coordinate-bearing molecule assembly records
- add bond/angle/dihedral generation helpers for simple molecules
- make unsupported elements fail explicitly with useful diagnostics

### T3. Reactive Chemistry Foundation
Status: active

- add bond formation/breaking rule representation
- add reaction participants and stoichiometric transforms on molecule graphs
- add element/mass/charge conservation checks for reactions
- add local chemistry events that mutate explicit molecule inventories

Current slice already implemented:
- conserved molecule-level reactions
- element and charge balance validation
- finite-extent mutation of explicit material mixtures
- bond form / break / bond-order change transforms on molecule graphs
- connected-component product splitting after structural transforms
- finite-extent replacement of explicit mixture components by transformed products
- scoped atom references across multiple reactant molecules
- cross-reactant structural reaction execution with explicit product graphs
- explicit formal-charge edits on molecule and structural reactions
- proton-transfer / ionization-style structural reactions with charge conservation
- atom-mapped product construction from explicit reactant atom provenance
- material-level execution of atom-mapped reactions over explicit mixtures
- generic valence admissibility checks on explicit product graphs
- heuristic bond-energy estimates for admissible reactions and transforms
- law-based bond-formation proposal/ranking from valence headroom and element properties
- law-based bond-breaking and bond-order-shift proposal/ranking from the same primitive layer
- law-based proton-transfer proposal/ranking from donor-H bonding, electronegativity, and valence headroom
- law-based electron-transfer proposal/ranking from electronegativity gap and post-transfer charge admissibility
- law-based terminal-atom transfer proposal/ranking from terminality, electronegativity improvement, and valence headroom
- multi-step structural path search over primitive law-based moves toward explicit target product sets
- automatic conversion of law-generated structural rearrangements into explicit atom-mapped reaction templates with source provenance
- provenance-aware multi-step path search that emits a single net atom-mapped reaction template from the discovered primitive path
- conflict-checked batched structural chemistry so compatible primitive reactions can execute concurrently instead of only as a serialized queue
- multi-step batch-path search so law-generated chemistry can progress in concurrent reaction waves rather than only one primitive move per step
- provenance-aware batch-path search that collapses multi-wave chemistry into a single net atom-mapped reaction template
- primitive Lewis-style shell-electron bookkeeping with duet/octet/expanded-shell targets folded into law-based reaction scoring
- oxidation-state and partial-charge heuristics from explicit atom graphs folded into law-based reaction scoring
- local acid/base and electron donor/acceptor propensities derived from explicit atom-graph electron allocation, used directly in proton/electron transfer ranking
- explicit material-mixture execution of structural and atom-mapped batch waves so compounds can react concurrently at the inventory layer too
- inventory-driven law-based wave enumeration/execution directly from explicit material multiplicity, without pre-supplying reaction batches by hand
- inventory-driven multi-wave path enumeration/execution directly from explicit material multiplicity, so compound pools can discover and run concurrent reaction sequences instead of only a single best wave
- whole-state batch and path ranking from resulting electron-allocation state, so concurrent waves and search frontiers prefer globally stabilized reactant sets instead of only summing local primitive scores
- law-based heterolytic polarized bond cleavage from atom-graph partial charge, electronegativity, and donor/acceptor propensity, so ionic bond dissociation can emerge without hardcoding named acid/base cases
- batch metadata now tracks primitive-family diversity and mixed-wave score, so concurrent chemistry can promote genuinely mixed law-driven waves instead of only same-family batches
- batch metadata now also tracks explicit redox-route score from whole-wave polarization, oxidation-span, and state-gain deltas
- direct mixed-redox batch generation now seeds concurrent ionic/redox waves from electron-allocation pressure instead of relying only on generic proposal DFS
- structural batch candidate selection now keeps both high-admissibility and high-redox-pressure proposals, so redox-active waves stay discoverable under truncation
- direct mixed-redox batch generation now extends beyond pairwise seeds and can assemble larger concurrent waves with three distinct primitive families from compatible graph-state pressure
- molecule-level donor, acceptor, and intramolecular redox-pressure scores are now derived directly from atom-graph partial charge, oxidation state, and radical bookkeeping and fed back into electron-transfer / ionic-cleavage ranking
- mixed-redox wave search no longer suppresses repeated primitive families during construction, so repeated bond-form or other compatible moves can survive inside a genuinely mixed redox-active wave
- mixed-redox anchoring and acceptance now use proposal-level redox activity derived from reaction-state deltas instead of relying on a hardcoded “redox family” label set
- mixed-redox wave construction now carries an explicit pairwise mini-wave score for `current + candidate`, so DFS ordering responds to combined redox-route completion and resulting-state gain instead of ranking candidates only in isolation
- mixed-redox wave construction now also scores partial-wave redox-field gain from the actual intermediate product set, so DFS can prefer candidates that improve donor/acceptor closure and oxidation-field balance in the in-progress wave itself
- mixed-redox wave construction now also uses touched-atom hotspot alignment, so DFS can favor candidates whose atom-level donor/acceptor and signed charge/oxidation centers match the current wave’s active sites
- mixed-redox anchor and companion ranking now also use reactant-set hotspot extrema from continuous donor/acceptor, partial-charge, and oxidation fields, so wave generation is no longer driven only by the flat proposal list ordering

Current focus within `T3`:
- landed: batches and path/frontier search now carry explicit resulting/final state scores derived from whole-reactant electron allocation
- landed: ranking proofs cover closed-shell preference, positive H2 dimerization state gain, descending batch state ordering, and descending batch-path final-state ordering
- landed: polarized heterolytic cleavage now generates ionic bond-breaking routes directly from bond polarization and electron donor/acceptor propensities
- landed: concurrent batches now expose primitive diversity and mixed-wave score, with direct proof of a disjoint ionic-cleavage plus electron-transfer wave
- landed: batches now expose explicit redox-route score and sort mixed redox waves by whole-wave electron-state improvement rather than only local primitive admissibility
- landed: direct mixed-redox batch generation now finds ionic-plus-electron-transfer waves from graph-state redox pressure
- landed: mixed-redox generation now builds larger concurrent waves beyond pairwise seeds, with direct proof of a three-proposal heterolytic/electron-transfer/bond-form wave
- landed: molecule-level donor/acceptor/redox-pressure scores now come directly from electron-density and oxidation heuristics and influence electron-transfer plus heterolytic-cleavage ranking
- landed: mixed-redox wave construction now allows repeated primitive families so long as the final batch remains genuinely mixed and redox-active
- landed: mixed-redox detection now keys off proposal-level redox activity from state deltas, not only enum-family labels
- landed: mixed-redox DFS now uses pairwise mini-wave state for `current + candidate`, and the proof suite verifies that a heterolytic anchor boosts a complementary electron-transfer partner more than an unrelated H-H closure
- landed: mixed-redox DFS now also uses partial-wave redox-field gain from the actual intermediate product set, and the proof suite verifies that a heterolytic partial wave gains more field closure from a complementary electron-transfer follow-up than from an unrelated H-H closure
- landed: mixed-redox DFS now also uses touched-atom hotspot alignment, and the proof suite verifies that a heterolytic anchor aligns more strongly with a complementary electron-transfer candidate than with an unrelated H-H closure
- landed: mixed-redox anchor and companion ranking now use reactant-set hotspot extrema from continuous donor/acceptor, partial-charge, and oxidation fields, and the proof suite verifies that those extrema prefer the Na/Cl electron-transfer route over detached H-H closure
- landed: hotspot-seeded redox proposal generation now builds electron-transfer, heterolytic-cleavage, proton-transfer, and field-biased bond-formation candidates directly from reactant-set extrema before the engine falls back to the full primitive proposal pool
- landed: hotspot-seeded redox proposal generation now respects the same basic electronegativity and donor/acceptor-law gates as the family enumerators, so it no longer fabricates hydrogen-only redox routes
- landed: mixed-redox batch search now runs a hotspot-seeded path first, and the proof suite verifies that the ionic HCl/Na/Cl route can be constructed from hotspot-seeded proposals alone
- landed: mixed-redox wave search now also has a direct field-seeded batch-construction path, so hotspot-guided anchors can assemble redox waves before proposal-DFS fallback runs
- landed: field-seeded interactions now lower into multiple admissible realizations instead of a fixed one-interaction/one-primitive mapping, and direct field-wave construction now consumes that richer realized proposal set
- landed: field-seeded interactions now also harvest admissible realizations from the local one-hop law pool around their core atoms, so terminal-atom transfer and other families can emerge from field seeds without being hand-mapped to that interaction kind first
- landed: direct field-wave construction now runs the same DFS mixed-wave search over field-derived proposals instead of a brittle greedy companion pick, so valid ionic waves are not dropped when the field-local proposal set gets richer
- landed: covalent-closure field seeds now gate charge-transfer realizations on real electronegativity / partial-charge gradients, so hydrogen-only closures no longer fabricate bogus H↔H electron-transfer routes
- landed: field-wave search now also has an interaction-first path, so complementary hotspot interactions are assembled directly from field state before their lowered proposal realizations are sent through mixed-redox batch construction
- landed: interaction-first field-wave proofs now cover the direct ionic HCl/Na/Cl interaction wave, closing another gap between continuous field state and concurrent batch generation
- landed: interaction-first field waves now also harvest cross-interaction local realizations from the generic law pool, so proposals that span multiple hotspot interactions can enter the wave even when no single interaction lowering emitted them directly
- landed: direct wave-topology proposal generation now synthesizes electron-transfer, heterolytic-cleavage, proton-transfer, and closure proposals from the union local atoms of a field interaction wave itself, before generic-pool harvest
- landed: direct wave-topology generation now has an explicit field-topology link layer, so continuous interaction-wave structure can be inspected and ranked before it is lowered into primitive structural proposals
- landed: field-topology links now compose into scored link waves before lowering, so continuous topology can combine and rank multi-link local reaction structure before it materializes as primitive move proposals
- landed: explicit interaction-graph edges now connect complementary hotspot interactions by continuous donor/acceptor, charge, oxidation, and local-bridge closure, and graph-seeded interaction waves now run before the broader field-wave DFS path
- landed: field-topology link waves now build an explicit continuous wave-field state before mutation, so donor/acceptor, partial-charge, proton-acceptor, and local-redox pressure are harvested directly from the wave-local atom field rather than only from already-discretized link labels
- landed: mutated field-topology links now emerge from that wave-field state, including charge-topology-derived polarization bridges, so ionic routes can be surfaced from continuous wave state before primitive lowering
- landed: field-topology wave state now lowers directly into executable mixed reaction waves, so heterolytic cleavage, electron transfer, proton transfer, and closure can be batched from continuous wave state before the older link-lowering route runs
- landed: continuous field / wave state now also emits direct reaction-template steps, so reaction-graph seeds can be applied from wave state before the primitive-family batch path is consulted
- landed: continuous field / wave state now also emits explicit reaction-graph edges and multistep reaction-graph paths, so direct graph search can proceed from field-state template steps without dropping back to the older primitive batch path first
- landed: continuous field / wave state now also emits direct reaction-wave graph edges from mixed field-state reaction batches, so graph search can begin from direct concurrent wave chemistry without first crystallizing through fixed reaction-template seed shapes
- landed: multistep reaction-wave graph search now uses a bounded beam with cached per-signature edge expansions and one best path per final signature, so wave-graph exploration no longer tries to enumerate the full concurrent-wave tree
- landed: the bounded reaction-wave graph layer is now verified to stay finite and productive on duplicate inventories without exploding the concurrent-wave search tree
- remains: finish exact repeated-route and serialized chained-wave equivalence verification for the bounded wave-graph search, and keep reducing dependence on the current pre-declared primitive move set by letting continuous field / wave state evolve into richer reaction graphs without first lowering through the current fixed primitive-family proposal set

### T4. Compound/Material Assembly
Status: active

- landed: phase-aware material descriptors now exist for gas, liquid, aqueous, solid, amorphous, crystalline, and interfacial matter
- landed: phased compound portions and phased material mixtures now preserve the same compound across distinct material phases instead of collapsing everything into one flat molar bucket
- landed: phased material mixtures can collapse back to the flat inventory layer, so the new material phase bookkeeping is additive instead of breaking existing reaction/material APIs
- landed: regional material inventories now track phased compounds separately across pore-water, gas-phase, mineral-surface, bulk-solid, and biofilm-matrix regions
- landed: explicit solution, polymer, and crystal assembly bookkeeping now exists on the atoms-first layer, with flattening back to molecule inventories so higher layers can consume material structure without losing inventory conservation
- landed: regional material inventories now support explicit region-to-region transfer rules with phase selectors, so compounds can actually move between pore water, gas, mineral surface, bulk solid, and biofilm inventories instead of living in static buckets
- landed: equilibrium-style partition rules now relax compounds toward target inter-region distributions while conserving explicit molecule inventory, so adsorption/desorption and volatilization-style bookkeeping can be expressed directly at the material layer
- landed: regional material inventories can now ingest flat mixtures plus solution, polymer, and crystal assemblies directly with explicit phase assignment, so material structures populate pore-water, mineral-surface, bulk-solid, and biofilm regions without manual flattening by callers
- add material phase descriptors
- add crystal/polymer/solution mixture bookkeeping
- add pore-water / gas / mineral surface material inventories
- connect explicit compounds to substrate/material regions

### T5. Explicit Cellular Chemistry Inputs
Status: active

- landed: representative atoms-first support molecules now exist for glucose, oxygen gas, amino-acid pool, nucleotide pool, and membrane-precursor pool, so higher layers can seed explicit cell chemistry from molecule graphs instead of only anonymous scalar buckets
- landed: regional material inventories now project abundance-sensitive `WholeCellEnvironmentInputs`, so explicit cell chemistry can be estimated from explicit molecule/compound/material inventories instead of only weighted composition means
- landed: the explicit microbial whole-cell seam now builds nutrient inputs from regional material inventories first and only blends local environmental stress into metabolic load afterward
- landed: explicit microbial cohorts now own persistent `RegionalMaterialInventory` state instead of rebuilding a fresh material inventory for every whole-cell input call
- landed: owned explicit-microbe material inventories now relax conservatively toward the surrounding patch chemistry instead of snapping back to scalar substrate reconstruction each step
- landed: in owned cells, explicit microbial material inventories now exchange conserved glucose/oxygen/amino/nucleotide/membrane support directly against local substrate and atmospheric chemistry instead of only drifting toward a patch-derived target
- landed: explicit microbial nutrient uptake now mutates the owned regional material inventory directly before substrate writeback, so whole-cell chemistry depletes persistent molecule/material state instead of only transient projections
- landed: explicit microbial product release now accumulates carbon-dioxide and proton-pool molecules into the owned regional material inventory before coarse environment writeback, so persistent molecule/material state tracks both uptake and release
- landed: owned explicit-microbe steps no longer immediately mirror nutrient draws and product release back into coarse substrate / atmosphere when the cell already owns that patch; those molecules now remain in persistent owned material state first
- landed: owned carbon-dioxide and proton products now spill back into substrate / atmosphere through conservative owned-inventory boundary exchange during sync instead of immediate coarse writeback
- landed: owned nutrient-support molecules now maintain internal reserve bands instead of relaxing to exact patch-derived targets every sync, so local persistent material state can stay authoritative even when the surrounding patch is temporarily depleted
- landed: ATP-like support is now represented in owned regional material inventories, so explicit microbial energy output can remain in local material state instead of going straight to the coarse `AtpFlux` field
- landed: owned membrane support now consumes local ATP-like inventory before drawing coarse `AtpFlux`, pushing another part of the whole-cell support loop onto persistent molecule/material state
- landed: owned cells now cap nutrient consumption to what the material inventory actually holds, so the whole-cell step is genuinely rate-limited by local material stock instead of consuming arbitrarily from abstract mM deltas
- landed: batch and incremental explicit-microbe steppers now share a single `step_single_explicit_microbe` helper, eliminating ~200 lines of duplicated flux/growth/substrate code
- stop inventing chemistry only as coarse named scalar pools where explicit chemistry exists
- promote local whole-cell chemistry to read from molecule/compound/material state

### T6. Multiscale Coupling
Status: locked

- use coarse terrarium fields only outside explicit atomistic / molecular regions
- use boundary exchange between explicit chemistry regions and coarse background
- keep atomistic/molecular regions authoritative where they exist

## Hard Rules

- Do not call a higher-level ownership wrapper "bottom-up".
- Do not add new coarse biological authority in cells already owned by explicit chemistry.
- Do not present terrarium abstractions as the scientific source of truth.
- Every tranche must move authority downward, not just add a new overlay.
