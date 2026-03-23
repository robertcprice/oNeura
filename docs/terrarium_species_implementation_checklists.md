# Terrarium Species Implementation Checklists

This document converts the ordered backlog in
`docs/terrarium_species_backlog.md` into execution checklists.

The order is locked unless there is a clear scientific or architectural reason
to change it.

Rules carried over from the terrarium and atoms-first plans:

- explicit lower-scale state stays authoritative where it exists
- higher layers may summarize, schedule, or boundary-couple only
- behavior should emerge from explicit sensory, circuit, cellular, or tissue
  state rather than direct organism-level scripting

## Wave A: Harden Existing Anchors

### `Caenorhabditis elegans`

Files:

- `core/src/celegans.rs`
- `core/src/organism_catalog.rs`

Checklist:

- [x] Build the neuron index before canonical synapse construction so named
  connectome edges actually exist.
- [x] Add canonical neuromuscular wiring so body-wall muscles are driven by
  explicit motor neurons.
- [x] Add an explicit local sensory-input interface for attractant, repellent,
  temperature, touch, and immersion.
- [x] Route anterior and posterior touch through distinct locomotor pathways
  instead of treating all touch as the same command.
- [x] Make body translation depend on muscle activation and locomotion state
  rather than command-neuron labels alone.
- [x] Extract a dedicated `celegans::assays` module with chemotaxis,
  thermotaxis, and crawl-versus-swim assay harnesses.
- [x] Validate and tune the chemotaxis assay so it shows reliable approach from
  sensory asymmetry and circuit state.
- [x] Validate and tune the thermotaxis assay so `AFD`-centered input produces
  reliable preference movement.
- [x] Validate and tune the crawl-versus-swim assay so the two modes remain
  behaviorally distinct under the current locomotor decoder.
- [x] Add feeding and energy-state modulation of locomotion.
- [x] Expand beyond the current simplified connectome subset into a better
  sensorimotor slice for turns and reversals.

### `Drosophila melanogaster`

Files:

- `core/src/drosophila.rs`
- `core/src/fly_metabolism.rs`
- `core/src/terrarium/fauna.rs`

Checklist:

- [ ] Audit every experiment path so actions remain downstream of explicit
  sensory encoding, neural state, and motor decode.
- [x] Add regression tests proving that visual, odor, and thermal perturbations
  move the relevant brain regions before they change behavior.
- [x] Tighten the coupling between energy state, hunger, and action selection.
- [x] Tighten the coupling between circadian state and excitability/action.
- [ ] Separate any remaining task-specific reward or steering shortcuts from
  circuit-authoritative pathways.
- [x] Add clearer assay baselines for phototaxis, thermotaxis, foraging, and
  circadian modulation.
- [x] Start reducing region-level shortcuts where explicit neuron-class state is
  available.

Current status:

- `core/src/drosophila/assays.rs` now exposes probe-level olfactory,
  phototaxis, and thermotaxis checks plus a foraging baseline, with regression
  tests that verify the sensory perturbation lands in the correct fly brain
  regions before the behavior assay is evaluated.
- The common fly body-step path now batches neural windows through the
  GPU-capable `MolecularBrain::run_without_sync` path when no per-step FEP loop
  is active, and motor decode reads contiguous spike ranges through
  GPU-capable reducers instead of forcing a full host shadow sync.
- The terrarium-specific fly step now re-encodes sensory and explicit internal
  arousal state across coarse neural windows, then decodes motor output from
  cumulative contiguous spike-range deltas. That removes the previous
  host-driven `DN`/`VNC` stimulation shortcut while keeping the hot path
  batched for GPU execution.
- Terrarium coupling no longer writes hunger current directly into `SEZ` and
  `MBON`. Explicit `FlyMetabolism` now synchronizes energy, hunger, and
  circadian phase/activity into the fly sim as homeostatic inputs, and the fly
  consumes those inputs inside the batched neural window.
- `core/src/terrarium/fauna.rs` now owns a real metabolism step instead of a
  stub: locomotor activity is classified from explicit fly body state,
  circadian state is advanced there, reserve depletion and ATP stress are
  stepped there, and starvation/ATP-crash telemetry is emitted on threshold
  crossings.
- `core/src/drosophila/assays.rs` now includes a circadian assay that first
  verifies day/night homeostatic state changes the arousal-region input current
  delivered by the explicit body-step pathway, then checks the downstream
  locomotor activity baseline. `run_circadian()` now scores locomotor activity
  instead of using an unstable whole-brain spike ratio.
- `core/src/drosophila/assays.rs` also now includes a lateral odor-geometry
  probe that verifies left/right odor bias lands in the correct `AL` and `LH`
  halves without destabilizing downstream motor readouts. The connectome now
  includes extra same-side lateralized wiring for `AL` -> `LH`, `LH` -> `CX/DN`,
  `CX` -> `DN`, and `DN` -> the left/right turn pools of `VNC`, and the fly
  now runs with a stronger synaptic current scale so that explicit `DN -> VNC`
  transfer actually reaches the motor pools at `Tiny` scale.
- `core/src/drosophila.rs` now also exposes explicit excitatory/inhibitory
  region partitions that the fly lane uses for odor injection and left/right
  motor-pool readout. `core/src/drosophila/assays.rs` records side-specific
  `CX`, `DN`, and `VNC` activity for the lateral odor probe. That deeper
  diagnostic now shows downstream left/right asymmetry all the way through the
  scaffold-free `AL/LH -> CX/DN -> VNC` path at `Tiny` scale, and the common
  terrarium body-step path now runs without the direct `CX` odor-asymmetry
  bridge.
- The key same-side `CX/LH -> DN -> VNC` route now includes deterministic
  topographic fan-in scaffolds in addition to the stochastic reduced connectome
  edges, so left/right olfactory steering no longer depends on lucky RNG draws
  inside the reduced scaffold.
- The terrarium fly turn decoder now reads left/right `VNC` population firing
  rates rather than a sparse-spike ratio that could saturate to `+/-1` from a
  handful of spikes. `decode_motor()` now uses the same left-positive turn
  convention as the terrarium path.
- The live terrarium fly runtime now enters through an explicit
  `TerrariumFlyInputs` sensory slice instead of passing a coarse reward scalar
  through the hot body-step path. The old `reward_valence` signature is still
  retained only as a compatibility wrapper that queues explicit peripheral
  taste state before the authoritative sensory path runs.
- The common terrarium body-step path also no longer writes wind directly into
  `CX`. Wind now enters through bilateral mechanosensory `OTHER` partitions,
  and the connectome carries that signal forward through stronger explicit
  `OTHER -> CX/DN` coupling. The regression now checks the authoritative path
  directly: left-lateral wind first biases `OTHER`, then biases `CX`, then
  produces the correct signed turn through the live body-step path.
- The old queued scalar reward scaffold has now been replaced by a short-lived
  explicit peripheral feeding/taste state inside `core/src/drosophila.rs`.
  Positive and negative reinforcement compatibility calls are still supported,
  but they now only buffer peripheral taste state rather than writing region
  current, and real terrarium feeding registers ingestion-derived sensory
  feedback from the fly's own body-step path instead of a top-down reward write
  from `core/src/terrarium/fauna.rs`.
- The reduced fly scaffold now also has an explicit `SEZ -> DAN -> MB` teaching
  route, and `core/src/drosophila/assays.rs` includes a feeding-reinforcement
  probe that checks ingestion-derived sensory state first raises `SEZ`, then
  recruits `DAN`, then raises dopamine in mushroom-body targets, all without
  any direct external current injection into `DAN`.
- `run_drug_response()` no longer injects baseline activity directly into `AL`;
  it now uses an odor-evoked manual sensory probe through the normal fly input
  path before measuring TTX suppression.
- The remaining shortcut cleanup is now less about direct region current
  injection and more about shrinking the remaining compatibility reinforcer API
  while adding more explicit peripheral state and plasticity. Circadian-specific
  assay hardening beyond the current baseline also remains open.

## Wave B: Land Terrarium Core

### `Arabidopsis thaliana`

Target files:

- `core/src/plant_cellular.rs`
- `core/src/plant_organism.rs`
- `core/src/terrarium/flora.rs`
- new root and leaf tissue modules under `core/src/terrarium/` as needed

Checklist:

- [ ] Add an `OrganismType` entry and catalog stats once the first explicit
  implementation slice exists.
- [ ] Define root tissue cell classes and their authoritative local state.
- [ ] Add explicit water and nitrate uptake at tissue-owned patches.
- [ ] Add auxin transport and growth anisotropy as the primary local steering
  mechanism for root behavior.
- [ ] Add gravitropism from local hormone and growth state, not a root steering
  vector.
- [ ] Add root exudation coupled to nearby microbial microdomains.
- [ ] Add leaf tissue gas-exchange state after root authority is stable.
- [ ] Add drought and nutrient-stress assays.

### `Bacillus subtilis`

Target files:

- `core/src/terrarium/explicit_microbe_impl.rs`
- `core/src/terrarium/packet.rs`
- `core/src/whole_cell.rs`
- new species-specific microbial modules under `core/src/terrarium/`

Checklist:

- [ ] Define a `B. subtilis` genotype/module record that feeds explicit packet
  and whole-cell state.
- [ ] Add nutrient uptake and respiration as authoritative local microbial state.
- [ ] Add sporulation and dormancy state transitions.
- [ ] Add quorum-signaling state and matrix production.
- [ ] Add explicit biofilm-local material bookkeeping.
- [ ] Add patch-scale growth, stress, death, and dispersal assays.
- [ ] Keep coarse guild scalars suppressed in owned `B. subtilis` patches.

### `Dictyostelium discoideum`

Target files:

- new `core/src/dictyostelium.rs`
- `core/src/terrarium/explicit_microbe_impl.rs`
- `core/src/terrarium/mod.rs`

Checklist:

- [ ] Add explicit cell state for cAMP production, sensing, and refractory phase.
- [ ] Add motility and polarity state per cell.
- [ ] Add starvation-triggered switching from feeding to aggregation.
- [ ] Add aggregation and streaming from local cAMP relay.
- [ ] Add prestalk and prespore differentiation state.
- [ ] Add slug migration mechanics.
- [ ] Add fruiting-body and spore persistence only after aggregation is stable.

## Wave C: Aquatic Bottom-Up Benchmarks

### `Paramecium tetraurelia`

Checklist:

- [ ] Add explicit membrane and calcium excitability state.
- [ ] Add ciliary beat and reversal state.
- [ ] Add mechanosensory avoiding reaction.
- [ ] Add simple chemotaxis and feeding-state modulation.
- [ ] Add assay harnesses for swimming speed, reversal frequency, and obstacle
  avoidance.

### `Chlamydomonas reinhardtii`

Checklist:

- [ ] Add explicit photoreception and eyespot-linked sensory state.
- [ ] Add two-flagellum beat asymmetry.
- [ ] Add phototaxis and photokinesis from sensory and motility state.
- [ ] Add chloroplast energy state coupled to growth and motility.
- [ ] Add nutrient and circadian modulation later in the lane.

## Wave D: Aquatic Neural Lane

### `Hydra vulgaris`

Checklist:

- [ ] Add neuron subtype and distributed nerve-net state.
- [ ] Add myoepithelial and body-wall actuation state.
- [ ] Add contraction and elongation behaviors.
- [ ] Add mechanosensory and feeding loops.
- [ ] Add regeneration only after baseline body control is stable.

### `Ciona robusta` larva

Checklist:

- [ ] Add compact CNS neuron and sensory-cell graph.
- [ ] Add tail-muscle actuation and swim-bout generation.
- [ ] Add phototaxis and gravitaxis from circuit state.
- [ ] Add developmental transition hooks only after larval behavior is stable.

## Current Focus

The active implementation focus in `core/src/celegans.rs` is now complete for
the first hardening slice:

- connectome initialization happens in the correct order
- canonical muscles are wired to explicit motor neurons
- local sensory inputs are explicit runtime state
- touch pathways distinguish anterior reversal from posterior escape
- body motion depends on muscle activation instead of command labels alone
- feeding state modulates pharyngeal pumping, gut loading, and locomotor drive
- assay defaults now pass for chemotaxis, thermotaxis, and crawl-versus-swim

Latest validation notes:

- assay runs are currently verified through a built-library probe because
  `cargo test -p oneura-core --lib` is still blocked by pre-existing duplicate
  `tests` modules in `core/src/terrarium/mod.rs`
- the locomotor command slice now routes electrical coupling through the
  dedicated gap-junction pool instead of misclassifying it as chemical drive

Next code target:

- start the `Drosophila melanogaster` hardening pass with the same rule:
  behavior must stay downstream of explicit sensory, circuit, and metabolic
  state
