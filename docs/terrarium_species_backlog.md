# Terrarium Species Backlog

This document turns the species-selection research pass into an implementation
backlog that matches the bottom-up terrarium contract:

- explicit biology owns local state where it exists
- coarse fields only act as background or boundary conditions
- behavior should emerge from explicit cell, tissue, and circuit state instead
  of being hand-scripted at the organism layer

It is intentionally biased toward organisms with unusually strong public data
coverage from genome and cell state upward to measurable behavior.

## Scope And Scoring

This backlog mixes three kinds of species:

1. existing anchor species already present in the repo
2. new land-terrarium species that fit the current plant and soil direction
3. aquatic benchmark species that are valuable for bottom-up validation even if
   they are not the first ecological fit for the current terrarium

Scores are qualitative and meant to help ordering, not claim exactness.

- `Fit`: ecological fit to the current terrarium direction
- `Cell`: depth of public genome, physiology, and cell-state resources
- `Circuit`: depth of nervous-system data for animals, or signaling/control
  architecture for non-animals
- `Behavior`: richness of quantitative assay literature
- `Cost`: implementation cost, where lower is easier

## Ordered Backlog Summary

| Order | Species | Lane | Fit | Cell | Circuit | Behavior | Cost | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `Caenorhabditis elegans` | Soil neural animal | 5 | 5 | 5 | 5 | 2 | Existing anchor |
| 0 | `Drosophila melanogaster` | Insect neural animal | 4 | 4 | 5 | 5 | 4 | Existing anchor |
| 1 | `Arabidopsis thaliana` | Plant tissue | 5 | 5 | 4 | 4 | 3 | Next land species |
| 2 | `Bacillus subtilis` | Soil bacterium | 5 | 5 | 4 | 4 | 2 | Next land species |
| 3 | `Dictyostelium discoideum` | Soil social amoeba | 4 | 5 | 4 | 5 | 2 | Next land species |
| 4 | `Paramecium tetraurelia` | Excitable unicell | 2 | 5 | 4 | 4 | 2 | Aquatic benchmark |
| 5 | `Chlamydomonas reinhardtii` | Phototactic alga | 2 | 4 | 4 | 4 | 2 | Aquatic benchmark |
| 6 | `Hydra vulgaris` | Nerve-net animal | 1 | 4 | 4 | 4 | 3 | Aquatic neural lane |
| 7 | `Ciona robusta` larva | Compact chordate | 1 | 4 | 5 | 4 | 3 | Aquatic neural lane |

## Order Rationale

The order is deliberate.

- `C. elegans` and `Drosophila` should be hardened first because the repo
  already carries explicit species-specific code for them. They are the current
  truth anchors for "behavior emerges from biology" in animals.
- `Arabidopsis`, `Bacillus`, and `Dictyostelium` are the best next land
  additions because they map directly onto the current terrarium architecture:
  plant tissue microdomains, explicit microbial microdomains, and signaling-led
  collective behavior.
- `Paramecium` and `Chlamydomonas` are excellent bottom-up benchmarks because
  they let the project test cell-biophysics-to-behavior emergence without the
  implementation burden of a large nervous system.
- `Hydra` and `Ciona` are scientifically strong, but they should wait until the
  codebase is ready for an aquatic microhabitat lane.

## Common Implementation Contract

Every new species brief below assumes the same rules.

- The first authoritative state must live at the lowest explicit layer we can
  actually simulate.
- Organism-level behavior modules may summarize or read lower-scale state, but
  they may not overwrite it once explicit lower-scale state exists.
- If a species requires neural behavior, the correct target is explicit sensory
  input -> explicit circuit state -> explicit body actuation, not a direct
  "task controller."
- If a species has no nervous system, behavior must still emerge from explicit
  signaling, excitability, gene regulation, cytoskeletal mechanics, or growth.

## Species Briefs

### 0. `Caenorhabditis elegans`

Why it stays first:

- It is still the strongest fully tractable animal for explicit
  cell-to-circuit-to-behavior modeling.
- The repo already contains a dedicated connectome-oriented implementation.

Minimal explicit state:

- neuron identities, neurotransmitter types, synapses, and graded membrane state
- body-wall muscle state and neuromuscular coupling
- mechanosensory, chemosensory, and thermosensory channels
- body posture, cuticle mechanics, and substrate contact
- reproductive and metabolic state only after the sensorimotor loop is stable

Must-have assays:

- crawling and omega turns
- swimming gait
- chemotaxis and klinokinesis
- tap withdrawal and mechanosensory reversals
- temperature preference

Canonical environmental variables:

- temperature
- substrate stiffness and moisture
- attractant and repellent gradients
- bacterial food density
- oxygen

What not to fake top-down:

- do not hand-script reversals or turns once sensory neurons, interneurons,
  motor neurons, and body mechanics exist
- do not collapse chemotaxis into a steering vector if gradient sensing is
  already explicit at the sensory layer

First tranche:

1. lock the connectome and body loop to emergent crawl and swim assays
2. make chemotaxis depend on explicit sensory adaptation and motor asymmetry
3. couple feeding and energy state into locomotion and state transitions

Key sources:

- https://openworm.org/ConnectomeToolbox/Neurons_OpenWormUnified_data/
- https://parasite.wormbase.org/Caenorhabditis_elegans_prjna13758/Info/Index
- https://pubmed.ncbi.nlm.nih.gov/37352352/

### 0. `Drosophila melanogaster`

Why it stays first:

- It is the strongest current route to a richer animal brain than `C. elegans`
  without jumping to vertebrate complexity.
- The repo already carries a dedicated fly brain simulation lane.

Minimal explicit state:

- sensory channels for odor, light, temperature, and mechanosensation
- explicit region or cell-class state for antennal lobe, mushroom body,
  central complex, optic lobes, and ventral nerve cord
- neuromodulators such as dopamine, serotonin, and octopamine
- body and wing actuation state
- energy, circadian phase, and hunger state

Must-have assays:

- phototaxis
- olfactory approach and avoidance
- thermotaxis
- foraging
- locomotor circadian modulation

Canonical environmental variables:

- light direction and intensity
- odor plume geometry
- temperature field
- food patch quality
- humidity

What not to fake top-down:

- do not hand-author "approach food" once sensory encoding, mushroom body
  plasticity, and central-complex action selection exist
- do not treat phototaxis as a direct sign bit if visual input and downstream
  circuit state already exist

First tranche:

1. harden sensory-input -> circuit-state -> action-output behavior loops
2. keep region-scale fly models visibly subordinate to explicit neuron or
   neuron-class state
3. add better metabolic and circadian modulation before expanding task count

Key sources:

- https://flybase.org/
- https://virtualflybrain.org/
- https://www.janelia.org/project-team/flyem/hemibrain
- https://pubmed.ncbi.nlm.nih.gov/32880371/
- https://pubmed.ncbi.nlm.nih.gov/35239393/

### 1. `Arabidopsis thaliana`

Why it is the first new land species:

- It is the strongest plant model for genome, cell identity, development, and
  quantitative phenotype resources.
- It fits the current terrarium direction directly through explicit plant tissue
  microdomains.

Minimal explicit state:

- genome and regulatory identity for major root and shoot cell types
- root epidermis, cortex, endodermis, pericycle, vasculature, and meristem state
- auxin transport and local hormone fields
- turgor, water potential, nutrient uptake, and cell-wall growth state
- stomatal and photosynthetic state for leaf tissues in later tranches

Must-have assays:

- root growth rate
- root system architecture
- gravitropism
- phototropism
- drought response
- stomatal response and diurnal growth in later phases

Canonical environmental variables:

- light direction and intensity
- soil water potential
- nitrate and phosphate
- temperature
- CO2 and humidity
- mechanical impedance

What not to fake top-down:

- do not apply a hand-authored "root seeks water" steering field once auxin,
  growth anisotropy, and local uptake are explicit
- do not let a canopy-level scalar override cell-level carbon or water state in
  owned plant tissues

First tranche:

1. explicit root tissue cell classes plus water and nitrate uptake
2. auxin-led growth anisotropy and gravitropism
3. local root exudation coupled to microbial patches
4. later: stomata, leaf gas exchange, and circadian modulation

Key sources:

- https://www.jcvi.org/research/arabidopsis-information-portal-araport
- https://pubmed.ncbi.nlm.nih.gov/35134336/
- https://pubmed.ncbi.nlm.nih.gov/40830271/
- https://pubmed.ncbi.nlm.nih.gov/22303208/
- https://pubmed.ncbi.nlm.nih.gov/24692421/

### 2. `Bacillus subtilis`

Why it is the first new soil microbe:

- It is a strong soil bacterium with unusually mature resources for genetics,
  metabolism, differentiation, sporulation, motility, and biofilms.
- It maps cleanly onto the explicit microbial microdomain plan.

Minimal explicit state:

- genotype identity and strain-specific module set
- nutrient uptake and respiration state
- sporulation decision state
- quorum and competence signaling state
- motility and chemotaxis state
- matrix production and biofilm architecture state

Must-have assays:

- growth curves under nutrient and oxygen variation
- chemotaxis
- colony expansion
- sporulation timing
- biofilm formation and matrix production

Canonical environmental variables:

- glucose and alternate carbon sources
- oxygen
- ammonium and nitrate
- temperature
- hydration
- pH
- crowding and quorum signal concentration

What not to fake top-down:

- do not keep a coarse guild scalar authoritative in owned patches once
  explicit uptake, sporulation, and biofilm state exist
- do not hand-switch biofilm mode from the terrarium layer if the quorum and
  nutrient state already imply it

First tranche:

1. explicit patch-local cells with nutrient uptake, growth, and death
2. explicit sporulation and dormancy path
3. quorum and matrix production
4. later: chemotactic movement and colony morphology

Key sources:

- https://subtiwiki.uni-goettingen.de/wiki/index.php/Main_Page
- https://subtiwiki.uni-goettingen.de/wiki/index.php/Metabolism
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8332661/

### 3. `Dictyostelium discoideum`

Why it is the next land species after plant and bacterium:

- It is one of the cleanest examples of collective behavior emerging from local
  signaling, chemotaxis, adhesion, and differentiation.
- It gives the terrarium a bridge between microbial and multicellular behavior
  without requiring a nervous system.

Minimal explicit state:

- genotype identity
- cAMP production, sensing, and relay state
- excitability and refractory state
- motility and polarity state
- cell-cell adhesion state
- differentiation state for prestalk and prespore programs

Must-have assays:

- aggregation center emergence
- streaming
- slug migration
- fruiting-body formation
- starvation-triggered developmental switching

Canonical environmental variables:

- food bacterial density
- starvation history
- hydration
- temperature
- substrate stiffness
- extracellular cAMP landscape

What not to fake top-down:

- do not seed aggregation centers by script once cAMP relay and chemotaxis are
  explicit
- do not force developmental stage transitions from a global scheduler when
  local starvation and signaling state can drive them

First tranche:

1. explicit chemotactic cell state and cAMP relay
2. aggregation and stream formation
3. differentiation and slug mechanics
4. later: fruiting-body construction and spore persistence

Key sources:

- https://dictybase.org/Dicty_Info/genome_statistics.html
- https://pubmed.ncbi.nlm.nih.gov/14681427/
- https://pubmed.ncbi.nlm.nih.gov/15875012/
- https://www.mdpi.com/2218-273X/14/7/830

### 4. `Paramecium tetraurelia`

Why it is the first aquatic benchmark:

- It is one of the best cell-biophysics-to-behavior organisms available.
- A single large excitable cell can produce clear measurable behavior without a
  nervous system, which is ideal for validating the project's low-level control
  philosophy.

Minimal explicit state:

- membrane voltage and ionic channels
- Ca-dependent excitability
- ciliary beat and reversal state
- metabolic reserve and feeding state
- local mechanosensory and chemical input channels

Must-have assays:

- straight swimming
- avoiding reaction
- backward swimming bursts
- attractant and repellent navigation
- speed modulation with medium chemistry

Canonical environmental variables:

- ionic composition of the medium
- temperature
- viscosity
- food particle density
- pH
- mechanical obstacles

What not to fake top-down:

- do not script avoidance turns directly if membrane excitability and ciliary
  reversal are explicit
- do not convert electrophysiology into a hidden "behavior state machine" once
  ion-channel state exists

First tranche:

1. membrane and calcium excitability
2. ciliary reversal and swimming kinematics
3. mechanosensory avoidance
4. later: chemotaxis and feeding-linked modulation

Key sources:

- https://paramecium.i2bc.paris-saclay.fr/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9946239/
- https://pubmed.ncbi.nlm.nih.gov/38757880/

### 5. `Chlamydomonas reinhardtii`

Why it follows `Paramecium`:

- It adds phototaxis and photosynthetic metabolism to the unicellular aquatic
  lane.
- It is a strong benchmark for flagellar control, light sensing, and energy
  coupling without nervous tissue.

Minimal explicit state:

- light-sensing state including eyespot and photoreceptor coupling
- chloroplast energy and redox state
- two-flagellum beat state and asymmetry
- circadian modulation
- nutrient and carbon state

Must-have assays:

- positive and negative phototaxis
- photokinesis
- recovery after light-step changes
- flagellar beat asymmetry
- growth across nutrient and light conditions

Canonical environmental variables:

- light intensity and direction
- wavelength if practical
- temperature
- acetate and carbon availability
- nitrogen and phosphorus
- medium viscosity

What not to fake top-down:

- do not hand-flip a heading toward light if photoreception and flagellar
  asymmetry are already explicit
- do not treat photosynthetic state as a cosmetic output once growth and
  motility depend on it

First tranche:

1. explicit light sensing and flagellar asymmetry
2. phototaxis and photokinesis
3. energy coupling to growth and motility
4. later: circadian and nutrient-state modulation

Key sources:

- https://www.chlamycollection.org/resources/about-chlamydomonas/
- https://www.chlamycollection.org/version-6-of-the-chlamydomonas-reference-genome-is-released/
- https://pubmed.ncbi.nlm.nih.gov/8137921/
- https://pubmed.ncbi.nlm.nih.gov/27122315/
- https://pubmed.ncbi.nlm.nih.gov/30862615/

### 6. `Hydra vulgaris`

Why it is the first added neural aquatic animal:

- It has a small but real nerve net, regeneration, and measurable whole-body
  behavior with strong public molecular resources.
- It is a better first step into aquatic neural animals than larger bilaterian
  systems.

Minimal explicit state:

- neuron subtype identity and distributed nerve-net state
- epithelial and myoepithelial actuation state
- pacemaker-like network activity
- feeding apparatus and tentacle state
- regenerative state in later tranches

Must-have assays:

- contraction bursts
- elongation
- bending
- feeding response
- mechanosensory response

Canonical environmental variables:

- temperature
- light-dark cycle
- prey or food density
- mechanical stimulation
- water chemistry

What not to fake top-down:

- do not trigger contraction pulses from a body-level scheduler if distributed
  neural and myoepithelial state already exists
- do not represent regeneration as an instant morphological switch

First tranche:

1. distributed nerve-net and body wall excitability
2. contraction and elongation behavior
3. mechanosensory and feeding loops
4. later: regeneration and structural remodeling

Key sources:

- https://research.nhgri.nih.gov/hydra/
- https://pubmed.ncbi.nlm.nih.gov/41071669/
- https://pubmed.ncbi.nlm.nih.gov/34328079/

### 7. `Ciona robusta` larva

Why it is last in the current order:

- It is scientifically excellent, but it belongs after the codebase is ready
  for an aquatic chordate lane.
- It is the cleanest bridge toward a vertebrate-like nervous system without
  jumping to zebrafish-scale model burden.

Minimal explicit state:

- photoreceptor and sensory neuron state
- compact CNS neuron identities and synapses
- tail muscle and notochord-linked body dynamics
- developmental stage and metamorphic competence
- energy state and swimming history

Must-have assays:

- phototaxis
- gravitaxis
- spontaneous swims
- startle responses
- simple state-dependent tactic switching

Canonical environmental variables:

- light direction and intensity
- gravity reference
- temperature
- salinity
- flow field

What not to fake top-down:

- do not hard-code tactic mode switches if the sensory and central circuit can
  already produce them
- do not reduce the body to a point mass once tail-muscle wave generation is
  explicit

First tranche:

1. compact larval sensory and CNS graph
2. explicit tail-muscle actuation and swim bouts
3. phototaxis and gravitaxis from circuit state
4. later: developmental transitions toward metamorphosis

Key sources:

- https://www.aniseed.fr/
- https://pubmed.ncbi.nlm.nih.gov/27921996/
- https://pubmed.ncbi.nlm.nih.gov/30392840/
- https://pubmed.ncbi.nlm.nih.gov/37162881/

## Deferred Species

These are not rejected. They are deferred because their public data stacks are
less favorable for the current bottom-up objective than the ordered list above.

- `Formica` ants: rich behavior, weaker integrated cell-to-connectome path
- `Lumbricus terrestris`: ecologically attractive, but weaker cell and neural
  tractability than current priorities
- `Daphnia`, `Schmidtea`, zebrafish: scientifically rich, but either too large,
  too aquatic-specialized for the current land roadmap, or too expensive for
  the immediate bottom-up return
- `E. coli` and yeast: extraordinary data resources, but weaker ecological fit
  than `B. subtilis` and `Dictyostelium` for the current terrarium direction

## Recommended Execution Waves

### Wave A: Harden Current Anchors

- keep `C. elegans` and `Drosophila` behavior explicitly downstream of sensory,
  circuit, muscle, and body state
- add regression assays proving that behavior shifts when lower-scale state
  changes

### Wave B: Land Terrarium Core

- `Arabidopsis thaliana`
- `Bacillus subtilis`
- `Dictyostelium discoideum`

This is the highest-value new wave because it matches the current terrarium
ownership model directly.

### Wave C: Aquatic Bottom-Up Benchmarks

- `Paramecium tetraurelia`
- `Chlamydomonas reinhardtii`

These species are especially useful for validating that behavior can emerge from
explicit excitability, motility, and energy state without a hand-authored brain
layer.

### Wave D: Aquatic Neural Lane

- `Hydra vulgaris`
- `Ciona robusta` larva

Only start this wave after the project is ready to support aquatic habitat
physics and organism-specific body coupling there.
