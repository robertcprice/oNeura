# Terrarium Bottom-Up Execution Plan

This plan replaces the previous terrarium-first refinement path.

The core rule is simple:

- explicit biology owns local state where it exists
- coarse fields only fill the background outside those owned regions
- boundary exchange is allowed
- dual ownership is not allowed

This is a multiscale plan, but it is not top-down in the scientific sense. The
authoritative biology should move downward into explicit genomes, cells, local
chemistry, and finally atomistic probes. The terrarium should become the
container and boundary-condition layer, not the source of truth for life.

## Invariants

Every tranche must preserve these invariants:

1. No dual authority
   - If a voxel/cell is owned by an explicit microdomain, broad microbial
     turnover cannot also be authoritative there.

2. Boundary-only coupling
   - Exchange between explicit and coarse regions must happen through explicit
     flux transfer or relaxation at the boundary, not by letting both models
     update the same state directly.

3. Persistent explicit state
   - Whole-cell simulators, genotype banks, and explicit packets must persist
     across world steps. Recreating them each frame is not allowed.

4. Scientific honesty
   - Do not describe hybrid regions as fully bottom-up.
   - Do not claim atomistic simulation unless the active local region is
     actually running the atomistic/MD path.

5. Verification per tranche
   - Every stage adds focused tests and runtime diagnostics before the next
     tranche starts.

## Target Architecture

The target runtime has four nested authority layers:

1. Global background fields
   - atmosphere
   - hydrology
   - mineral pools
   - large-scale transport

2. Explicit microbial microdomains
   - local ownership map
   - explicit genotype-ID packet populations
   - explicit whole-cell cohorts for active subsets
   - explicit local chemistry authority

3. Explicit plant tissue microdomains
   - root and leaf tissue cells as owners of local plant biochemistry
   - explicit uptake/exudation and gas exchange

4. Atomistic local probes
   - membrane, enzyme, mineral, and reactive hotspot patches
   - parameterize or directly perturb local mesoscale dynamics

## Phase 1: Ownership Infrastructure

Goal: make explicit biological regions first-class owners.

### Tasks

1. Add a microdomain ownership map to `oneuro-metal/src/terrarium_world.rs`
   - ownership mask for each 2D soil position
   - ownership strength field
   - ownership source metadata for debugging

2. Define ownership classes
   - none/background
   - explicit microbe cohort
   - explicit genotype packet region
   - plant tissue region
   - atomistic probe region

3. Add owned-region rebuild logic
   - rebuild after explicit cohort creation/removal
   - rebuild after explicit packet migration/division
   - rebuild after plant tissue microdomain changes

4. Add diagnostics
   - owned soil fraction
   - explicit microbial owned fraction
   - overlap checks
   - maximum ownership strength

### Stopping criteria

- owned regions appear in snapshot/runtime metrics
- no owned cell remains unmarked
- no illegal overlap occurs

## Phase 2: Authority Suppression

Goal: coarse soil biology stops owning explicit regions.

### Tasks

1. Suppress broad decomposer authority in owned regions
2. Suppress broad microbial respiration in owned regions
3. Suppress broad nitrifier/denitrifier biological authority in owned regions
4. Keep coarse hydrology/mineral transport active globally
5. Keep only boundary/background microbial behavior outside owned regions

### Subtasks

1. Split broad-soil update outputs into:
   - background-safe outputs
   - explicit-biological outputs

2. Gate explicit-biological outputs with ownership mask
3. Add tests that owned cells reduce broad microbial drive locally
4. Add tests that background cells still behave normally

### Stopping criteria

- broad microbial activity is attenuated to near zero in fully owned cells
- atmosphere/substrate changes in owned cells are measurably influenced by
  explicit cohorts

## Phase 3: Explicit Microbial Local Authority

Goal: explicit cohorts become true local biological owners, not probes.

### Tasks

1. Expand `TerrariumExplicitMicrobe`
   - local patch state
   - represented cell count
   - genotype identity or lineage ID
   - local flux cache
   - ownership radius

2. Make explicit cohorts the primary microbial source in owned cells
   - glucose uptake
   - oxygen uptake
   - ammonium/nitrate draw
   - CO2/proton/ATP-flux release
   - local activity signal

3. Add cohort lifecycle
   - growth
   - stress
   - dormancy
   - death
   - migration or budding into nearby cells

4. Add cohort-local chemistry ledger
   - before/after snapshot deltas
   - cumulative uptake/release
   - energy stress metrics

### Stopping criteria

- owned patches can remain biologically active with broad microbial ownership
  suppressed
- explicit cohorts produce stable local flux signals over many frames

## Phase 4: Genotype-ID Packet Regions

Goal: replace aggregate microbial fields in owned regions with explicit packet
populations keyed by genotype identity.

### Tasks

1. Add explicit genotype-ID packet container
   - packet count
   - genotype ID
   - lineage ID
   - activity state
   - dormancy
   - reserve
   - damage

2. Tie genotype ID to reusable genotype catalog entries
   - no duplicated raw gene vectors as primary evolving state
   - packets reference catalog/genotype identity

3. Replace broad microbial region state in owned cells
   - owned cells use packet populations
   - background still uses coarse fields

4. Add packet-local mutation and selection
   - mutation operates on genotype identities/catalog state
   - not on anonymous duplicated arrays

### Stopping criteria

- owned cells can be simulated without the broad microbial state as primary
  microbial storage
- packet distributions evolve over time and survive checkpoint/restart

## Phase 5: Whole-Cell Promotion Path

Goal: active explicit packets can promote to real whole-cell agents.

### Tasks

1. Define promotion rules
   - high activity
   - steep gradients
   - division attempt
   - high uncertainty
   - user-selected ROI

2. Promote genotype-ID packets into `WholeCellSimulator` instances
3. Keep demotion rules
   - low activity
   - stable environment
   - budget pressure

4. Add state translation
   - packet -> whole-cell initialization
   - whole-cell -> packet summary on demotion

### Stopping criteria

- active owned patches can host persistent promoted whole-cell agents
- promoted and demoted states conserve mass/state within tolerance

## Phase 6: Local Chemistry Authority

Goal: owned regions stop using broad chemical heuristics as their primary
biological chemistry.

### Tasks

1. Introduce explicit local chemistry ownership in owned regions
2. Couple explicit packets/whole-cells directly to local substrate chemistry
3. Remove aggregate guild-rate formulas from owned regions
4. Keep coarse chemistry only as boundary/background support

### Subtasks

1. Add owned-region chemistry masks
2. Separate owned vs background chemistry update passes
3. Add boundary flux transfer between owned regions and coarse field
4. Verify mass conservation across owned/background boundary

### Stopping criteria

- biology in owned regions is driven by explicit packets/cells + local chemistry
- broad guild rate laws are no longer primary inside owned cells

## Phase 7: Explicit Genome Modules

Goal: move from compact gene axes to real genotype modules.

### Tasks

1. Define genotype module records
   - carbon uptake
   - oxygen respiration
   - nitrogen handling
   - stress response
   - dormancy
   - scavenging
   - membrane maintenance
   - repair
   - regulation

2. Decode phenotype from module composition
3. Use modules directly in selection and packet fitness
4. Add lineage/genotype persistence and mutation tracking

### Stopping criteria

- genotype modules, not compact axes, determine explicit packet and whole-cell
  behavior in owned regions

## Phase 8: Plant Bottom-Up Regions

Goal: start moving plant authority downward too.

### Tasks

1. Add owned plant root microdomains
2. Move local uptake/exudation from aggregate root formulas toward explicit
   tissue-cell owners
3. Couple plant cell state directly to owned local chemistry
4. Use coarse plant physiology only outside explicit plant-owned regions

### Stopping criteria

- at least root microdomains become explicit local owners

## Phase 9: Atomistic Microdomains

Goal: true atomistic detail where it matters.

### Tasks

1. Select hotspot classes
   - membrane interface
   - mineral surface
   - transport site
   - high-reactivity microdomain

2. Promote those hotspots into local MD/atomistic probes
3. Feed probe outputs back into local chemistry/transport parameters
4. Keep atomistic probes persistent or cached where appropriate

### Stopping criteria

- some owned regions are genuinely using atomistic probes as part of their
  active local authority

## Phase 10: Calibration And Validation

Goal: make this a scientific tool rather than a plausible story.

### Tasks

1. Add invariants
   - mass conservation
   - boundary flux conservation
   - ownership exclusivity

2. Add biological checks
   - explicit microbe survives or dies under expected local chemistry
   - packet competition behaves sensibly
   - promotion/demotion remains stable

3. Add long-run regression cases
   - owned patch under stress
   - mixed owned/background region
   - explicit region takeover
   - explicit region collapse

## Immediate Active Sequence

This is the order to execute now:

1. ownership map
2. suppress broad microbial biological authority in owned regions
3. explicit cohorts become primary local source in owned regions
4. diagnostics and tests
5. replace owned-region aggregate microbial state with genotype-ID packet
   populations

## What Not To Do

Do not do these unless they are directly required by the active phase:

- do not add more broad microbial trait heuristics as the main state
- do not expand aggregate guild summaries and call it bottom-up
- do not let coarse and explicit biology co-own the same local patch
- do not describe selected explicit patches as full atomistic life simulation

## Current Status

Already done before this plan:

- native terrarium runtime in Rust
- explicit whole-cell microbial cohort insertion
- local cohort/environment coupling surface
- first pass of local authority suppression fields

Current active tranche:

- make ownership explicit and authoritative
- keep coarse microbial biology out of owned local patches

## Execution Board

This is the concrete work queue to execute without renegotiating the direction
each step.

### Status legend

- `done`: implemented and verified
- `active`: current implementation tranche
- `queued`: next tranche after the current one
- `blocked`: cannot proceed honestly without a prerequisite

### Phase 1 Board: Ownership Infrastructure

1. Ownership strength field in `TerrariumWorld`
   - status: `done`
   - deliverables:
     - `explicit_microbe_authority`
     - `explicit_microbe_activity`
     - snapshot metrics for owned fraction / max authority / activity
   - verification:
     - focused explicit microbe tests
     - native runtime summary

2. Ownership rebuild after explicit cohort edits
   - status: `done`
   - deliverables:
     - rebuild after demo seeding
     - rebuild after explicit microbe insertion
     - rebuild after explicit microbe stepping

3. Ownership class/source metadata
   - status: `active`
   - deliverables:
     - explicit owner class map
     - owner source/debug metadata
     - overlap validation
   - exit:
     - no owned patch is anonymous
     - ownership source is visible in runtime diagnostics

### Phase 2 Board: Authority Suppression

1. Suppress broad decomposer control in owned cells
   - status: `done`
   - deliverables:
     - substrate control attenuation under explicit ownership

2. Suppress broad respiration influence in owned cells
   - status: `done`
   - deliverables:
     - surface respiration attenuation under explicit ownership

3. Suppress broad-soil biological state updates at the source
   - status: `done`
   - deliverables:
     - owned-region mask passed into `step_soil_broad_pools_grouped(...)`
     - coarse microbial state frozen in owned cells
     - coarse microbial mutation/turnover/potential outputs zeroed in owned cells
   - exit:
     - owned cells no longer evolve through the broad microbial state path

4. Split biological vs background chemistry in owned cells
   - status: `active`
   - deliverables:
     - owned/background chemistry masks
     - broad hydrology remains active
     - broad microbial chemistry is removed from owned cells
   - exit:
     - no dual chemistry authority in owned cells

5. Suppress broad nitrifier/denitrifier ownership in owned cells
   - status: `queued`
   - deliverables:
     - nitrifier/denitrifier state frozen or removed in owned cells
     - explicit nitrogen-handling owners can replace them later

### Phase 3 Board: Explicit Microbial Local Authority

1. Persistent explicit cohort ledger
   - status: `queued`
   - deliverables:
     - local uptake/release ledger
     - per-cohort stress/energy history
     - persistent ownership radius

2. Explicit cohort lifecycle
   - status: `queued`
   - deliverables:
     - growth
     - dormancy
     - death
     - budding / migration

3. Explicit cohort primary ownership in owned cells
   - status: `queued`
   - deliverables:
     - glucose / oxygen / nitrogen use from explicit state
     - coarse background disabled there

### Phase 4 Board: Genotype-ID Packet Regions

1. Introduce genotype-ID packet populations for owned cells
   - status: `done`
   - deliverables:
     - `GenotypePacket` struct with catalog_slot, genotype_id, lineage_id,
       activity, dormancy, reserve, damage, and cumulative flux ledger
     - `GenotypePacketPopulation` container per owned cell with competition,
       lifecycle stepping, and promotion candidate detection
     - `recruit_packet_populations()` seeds from local secondary bank state
     - `step_packet_populations()` runs lightweight metabolism per packet,
       reads/writes local substrate chemistry, handles dormancy/damage/growth
     - Snapshot metrics: population count, total cells, mean activity/dormancy,
       total packets, promotion candidates
     - Viewer panel displays PACKETS section when populations exist
   - verification:
     - compiles cleanly, integrated into step_frame loop
     - packet populations appear at owned cells alongside explicit cohorts
     - chemistry flux (glucose extract, CO2 deposit) wired to substrate

2. Replace coarse microbial storage in owned cells
   - status: `queued`
   - deliverables:
     - owned cells use packet populations as primary state
     - coarse microbial fields become background-only

3. Mutation/selection on genotype identities
   - status: `queued`
   - deliverables:
     - mutation on reusable genotype/catalog identities
     - no duplicated raw vectors as primary evolving state

### Phase 5 Board: Whole-Cell Promotion

1. Promote active genotype-ID packets into `WholeCellSimulator`
   - status: `queued`

2. Demote stable cells back into packet summaries
   - status: `queued`

3. Verify conservation across promotion/demotion
   - status: `queued`

### Phase 6 Board: Local Chemistry Authority

1. Owned-region chemistry mask
   - status: `queued`

2. Owned/background chemistry split
   - status: `queued`

3. Boundary flux transfer
   - status: `queued`

4. Mass-conservation regression
   - status: `queued`

### Phase 7 Board: Explicit Genome Modules

1. Replace compact axes with explicit module records
   - status: `queued`

2. Decode packet fitness and chemistry from modules
   - status: `queued`

3. Persist lineage and genotype mutation state
   - status: `queued`

### Phase 8 Board: Plant Bottom-Up Regions

1. Root-owned microdomains
   - status: `queued`

2. Explicit tissue-cell uptake/exudation
   - status: `queued`

### Phase 9 Board: Atomistic Microdomains

1. Persistent hotspot selection
   - status: `queued`

2. MD/atomistic authority in selected owned regions
   - status: `queued`

3. Back-coupling into mesoscale chemistry
   - status: `queued`

### Phase 10 Board: Validation

1. Ownership exclusivity regression
   - status: `active`

2. Explicit-owned patch survival/collapse cases
   - status: `queued`

3. Long-run mixed owned/background regression
   - status: `queued`

## Current Tranche

### Tranche T1: Source-Level Suppression Of Broad Microbial State

Objective:
- move from downstream attenuation to source-level suppression inside owned
  cells

Scope:
- pass an owned-region microbial mask into the native grouped broad-soil step
- freeze coarse microbial/nitrifier/denitrifier state in owned cells
- zero coarse biological outputs there
- keep the rest of the world runtime stable

Files:
- `oneuro-metal/src/soil_broad.rs`
- `oneuro-metal/src/terrarium_world.rs`
- `docs/terrarium_bottom_up_execution_plan.md`

Verification commands:
- `cargo test --manifest-path oneuro-metal/Cargo.toml soil_broad -- --nocapture`
- `cargo test --manifest-path oneuro-metal/Cargo.toml terrarium_world -- --nocapture`
- `cargo run --manifest-path oneuro-metal/Cargo.toml --bin terrarium_native -- --frames 10 --no-render --summary-every 5`
- `PYTHONPATH="/Users/bobbyprice/projects/oNeura/.venv-codex/lib/python3.14/site-packages:src" python3 demos/demo_actual_molecular_terrarium.py --headless-frames 30`

Status:
- `done` for source-level state suppression
- verification completed through the new owned-cell regression and native loop

### Tranche T2: Owned/Background Chemistry Split

Objective:
- remove the remaining coarse biological chemistry authority inside owned cells
  without disabling background transport globally

Scope:
- separate broad-soil biological chemistry deltas from background transport
- keep hydrology/mineral transport available as boundary/background support
- stop applying coarse microbial chemistry deltas inside owned cells

Files:
- `oneuro-metal/src/soil_broad.rs`
- `oneuro-metal/src/terrarium_world.rs`

Verification commands:
- `cargo test --manifest-path oneuro-metal/Cargo.toml terrarium_world -- --nocapture`
- `cargo run --manifest-path oneuro-metal/Cargo.toml --bin terrarium_native -- --frames 10 --no-render --summary-every 5`

Progress inside `T2`:
- `done`: owned cells keep background transport/weathering support
- `done`: owned cells branch out of the broad-soil biology loop instead of
  computing coarse biology and restoring afterward
- `done`: coarse nitrifier/denitrifier substrate-control and respiration
  influence is zeroed in owned cells
- `done`: explicit owned/background boundary-flux transfer now replaces the
  last restore-and-freeze chemistry seam
- `done`: soil-core regression verifies conservative owned/background boundary
  exchange directly
- `done`: world-level regressions verify owned cells receive chemistry only
  through explicit background boundary support while coarse biology remains off

Status:
- `complete`

### Tranche T3: Explicit Local Chemistry Authority

Objective:
- make owned regions consume and emit local chemistry through bottom-up
  explicit cohorts first, with the coarse broad-soil layer acting only as
  background support outside those owned cells

Scope:
- move more local microbial chemistry authority from broad-soil summaries into
  explicit whole-cell or genotype-ID packet state
- keep coarse pools as non-owned background support only
- tighten explicit cohort coupling to terrarium substrate chemistry before any
  new top-down ecology refinement

Files:
- `oneuro-metal/src/terrarium_world.rs`
- `oneuro-metal/src/whole_cell.rs`
- `oneuro-metal/src/terrarium.rs`

Verification commands:
- `cargo test --manifest-path oneuro-metal/Cargo.toml explicit_microbes -- --nocapture`
- `cargo test --manifest-path oneuro-metal/Cargo.toml terrarium_world -- --nocapture`
- `cargo run --manifest-path oneuro-metal/Cargo.toml --bin terrarium_native -- --frames 10 --no-render --summary-every 5`

Next locked tasks inside `T3`:
- [done] step explicit cohorts before substrate stepping so owned-cell uptake and
  release land before lattice chemistry transport/reaction
- [done] zero coarse decomposer/nitrifier/denitrifier substrate control inside
  explicitly owned cells
- [done] suppress coarse soil respiration inside explicitly owned cells
- [done] make explicit whole-cell environment inputs ignore coarse
  `root_exudates`, `litter_carbon`, `dissolved_nutrients`,
  `shallow_nutrients`, and `organic_matter` inside owned cells
- [done] route plant coarse exudate/litter deposits into background cells only
  while leaving substrate hotspots as the owned-cell chemistry carrier
- [done] route fruit decay and dead-plant detritus away from owned coarse
  litter/organic pools and into substrate-first plus background-only paths
- [done] add owned-cell regressions for pre-substrate explicit chemistry,
  control suppression, and coarse-input suppression
- [done] reconcile owned-cell coarse summary arrays from substrate chemistry
  after lattice stepping so remaining consumers read a derived summary layer,
  not an authority
- [done] recruit explicit cohorts dynamically from live microbial hotspots so
  ownership can grow out of the field instead of staying fixed to the demo seed
  set
- [done] carry dominant local microbial genotype/lineage identity into new
  explicit cohorts and maintain per-cohort uptake/stress/energy ledger state
- [done] let explicit cohorts update represented biomass/radius from whole-cell
  outputs so owned regions are carried by explicit state rather than a static
  cell count

Next locked tasks after this sub-tranche:
- expand owned-cell explicit biology from the fixed whole-cell cohort set toward
  genotype-ID packet ownership so more local patches are bottom-up owned
- extend owned-cell bottom-up ownership to more ecology consumers that still
  read coarse summaries indirectly (`seed-bank germination`, additional plant
  detrital contexts, and any remaining background-only heuristics)
