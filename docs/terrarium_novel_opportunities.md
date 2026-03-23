# oNeura Terrarium: Novel Opportunities

## 1. First-Principles Ecosystem Simulator

The oNeura terrarium is — to our knowledge — the only ecosystem simulator where soil chemistry, atmospheric dynamics, organism metabolism, and molecular interactions **all derive from quantum-mechanical descriptors** with zero empirical lookup tables.

The derivation chain:

```
Physical constants (Bohr radius, Coulomb constant, rest masses)
  → Quantum element descriptors (Slater shielding, orbital energies)
    → Molecular embeddings (3-D coordinates from covalent radii)
      → Quantum molecular descriptors (ED/CI ground state, dipole, charge span)
        → Geochemistry (activity, acidity, redox, selectivity — all continuous)
```

Every reaction rate, selectivity coefficient, and visual property traces back to the Schrödinger equation through this chain. This distinguishes oNeura from empirical models like CENTURY, DNDC, or DayCent that rely on fitted parameters.

**Opportunity**: Publish as a methodology paper (e.g., "Ab Initio Ecosystem Simulation: Deriving Soil Chemistry from Quantum Mechanics") in a computational ecology or environmental modeling journal. The novelty is not the quantum chemistry itself, but the unbroken derivation chain from physical constants to ecosystem-scale observables.

## 2. Reproducible Computation Pattern

Given a deterministic seed and the ab initio pipeline, the terrarium produces **bit-exact ecosystem state**. The same seed + same code = same soil chemistry, same organism populations, same atmospheric composition, at every timestep.

This is rare in ecosystem modeling, where stochastic processes are typically not controlled. Our `ChaCha12Rng` + explicit seed provenance + entropy tracking makes every simulation run a reproducible scientific experiment.

**Opportunity**: Frame as a reproducibility framework for computational ecology. Researchers could share seed + configuration and reproduce results exactly, addressing the reproducibility crisis in ecological modeling.

## 3. Extractable Quantum Descriptor Crate

The regression tier — an OLS model trained on 6 molecular structure features (atom count, bond count, heavy atom ratio, mean electronegativity, charge span, degree of unsaturation) with a Cholesky solver — predicts all 7 quantum molecular descriptors without running DFT or CI calculations.

This is a standalone capability: given any molecule's SMILES or atom/bond graph, predict its ground-state energy, dipole moment, effective charges, etc. in microseconds rather than hours.

**Opportunity**: Extract as an independent Rust crate (`quantum-descriptor-regression`) and publish on crates.io. Target cheminformatics users who need fast molecular property estimates for virtual screening, QSAR modeling, or materials discovery. Include the binary cache format (585 bytes for 22 species) as a distribution mechanism.

## 4. Multi-Scale Rendering from Unified Data Model

The same quantum descriptors that drive chemistry also provide atom colors (CPK convention) and radii (Van der Waals) for visualization. When a user zooms from ecosystem scale (terrain, populations) to molecular scale (bond networks) to atomic scale (electron configuration), the data comes from the same pipeline — no separate rendering database.

This "Powers of Ten" capability — seamlessly zooming from kilometers to picometers — is novel in ecological simulation. Most simulators either show macroscopic views OR molecular views, never both from the same data source.

**Opportunity**: Build an interactive web experience (WebGL/WebGPU) that lets users click any organism and zoom through tissue → cell → molecule → atom, all backed by live simulation data. This has strong educational and outreach potential.

## 5. Educational Platform

The combination of:
- Real quantum mechanics at the atomic level
- First-principles chemistry at the molecular level
- Emergent organism behavior at the macroscopic level
- Interactive drill-down between all scales

...creates a unique educational tool for teaching biology, chemistry, and physics as an integrated whole rather than isolated disciplines.

**Opportunity**: Partner with science museums or educational platforms (Khan Academy, Brilliant, etc.) for an interactive "Powers of Ten" exhibit. The terrarium could demonstrate how quantum-level properties (electron orbitals) determine molecular-level behavior (chemical bonds) which determines ecosystem-level observables (soil pH, plant growth).

## 6. Digital Twin for Soil Science

The first-principles soil chemistry model — with descriptor-derived selectivity for ion exchange, quantum-informed mineral weathering, and ab initio reaction thermodynamics — could be calibrated against USDA NRCS soil survey data for precision agriculture applications.

Current precision agriculture relies on empirical soil models with limited transferability. An ab initio model that derives soil behavior from mineral composition and quantum chemistry could:
- Predict fertilizer response from soil mineralogy
- Model long-term soil health under different management practices
- Estimate carbon sequestration potential from mineral surface chemistry

**Opportunity**: Develop a calibration pipeline against NRCS Soil Survey Geographic Database (SSURGO). If the ab initio predictions correlate well with measured soil properties (pH, CEC, base saturation), this becomes a publishable validation and a potential commercial tool for precision agriculture advisory services.

## 7. Spectral Ecology: First-Principles Shade Avoidance

The terrarium implements R:FR (red to far-red ratio) spectral decomposition from first principles: direct sunlight carries R:FR ≈ 1.2, diffuse/scattered light carries R:FR ≈ 1.0. Each plant receives its own R:FR ratio computed by separate red and far-red extinction through the canopy raycast — not a threshold lookup.

This spectral signal feeds PHYB and SAS gene circuits (Hill-function kinetics), producing shade avoidance elongation (up to 1.8× height increase) and branching suppression (down to 0.4× lateral growth) that are wired directly into the growth pipeline. The shade avoidance response *emerges* from gene expression dynamics, not from hardcoded "if shaded, grow taller" rules.

**Opportunity**: This is the first ecosystem simulator where shade avoidance syndrome emerges from spectral physics → phytochrome equilibrium → gene regulatory networks → morphological response. Publish as a methods paper in functional ecology ("Emergent Shade Avoidance from First-Principles Spectral Decomposition in Ecosystem Simulation"). The canopy raycast approach with separate red/FR extinction is novel and could interest the plant ecophysiology community.

## 8. Chemical Ecology: Volatile Signaling Cascade

The terrarium implements the full jasmonate defense signaling cascade:

1. Mechanical damage (wind or herbivory) → JA_RESPONSE gene activation
2. Jasmonate synthesis → GLV (green leaf volatiles) + MeSA (methyl salicylate) emission
3. VOC dispersal through the wind-advected odorant grid
4. Neighbor plant detection via SA_RESPONSE and DEFENSE_PRIMING gene circuits
5. AND-gate defense priming: requires BOTH neighbor VOC AND own jasmonic acid

Every step uses Hill-function kinetics with zero hardcoded thresholds. The defense priming AND-gate — where a plant must both detect neighbor distress signals AND have its own jasmonate pathway active — matches the biological requirement for "self + non-self" recognition in plant immunity.

Herbivore deterrence also follows Hill kinetics: jasmonate + salicylate accumulation creates a sigmoidal deterrence curve that repels fly grazing, creating a negative feedback loop where damage → defense → deterrence → recovery.

**Opportunity**: The combination of abiotic (wind) and biotic (herbivory) damage triggers feeding into the same defense cascade, with emergent neighbor-to-neighbor volatile communication, is novel in ecosystem simulation. Publish as "Ab Initio Chemical Ecology: Volatile Defense Signaling Without Empirical Parameters" in a chemical ecology journal. The AND-gate priming mechanism could interest the plant immunology community.

## 9. Emergent Weather from Simulation Physics

Weather in the terrarium is **fully emergent** — not prescribed as external forcing or Markov state transitions, but arising from the same thermodynamics that govern real atmospheric moisture:

- **Cloud cover** emerges from atmospheric humidity approaching saturation (Clausius-Clapeyron + Hill kinetics), with contributions from soil evaporation and convective instability (spatial temperature variance across the grid).
- **Precipitation** emerges when BOTH cloud cover AND humidity surplus exceed condensation thresholds (multiplicative Hill gate). Rain intensity scales with wind speed (orographic effect proxy).
- **Temperature offset** emerges from cloud albedo cooling, evaporative cooling from wet soil, latent heat release from precipitation, and seasonal solar heating.
- **Negative feedback**: precipitation removes atmospheric moisture, which reduces cloud cover, which stops precipitation — producing natural weather cycling without any external forcing.

The `WeatherRegime` enum (Clear/PartlyCloudy/Overcast/Rain/Storm) is a **diagnostic label** derived from the continuous emergent state for display purposes. It has zero effect on the simulation.

**Opportunity**: This is the only ecosystem simulator where weather itself is emergent from simulation physics. Most models treat weather as external forcing (reanalysis data, stochastic generators). The self-regulating humidity → cloud → rain → drying cycle, where weather arises from and feeds back into soil, atmosphere, and biology, is publishable as "Emergent Weather in Ab Initio Ecosystem Simulation" for atmospheric or environmental modeling venues. The negative feedback loop producing natural weather cycling without any state machine is particularly novel.

## 10. Aquatic Physics Adaptation

The same physics equations (drag, bending stress, soil transport, spectral light) apply to aquatic environments through species-specific parameters from `BotanicalSpeciesProfile`:

- Wood density: 100 kg/m³ (aquatic) vs 400-600 kg/m³ (terrestrial)
- Drag coefficient: 0.80 (aquatic, higher in water) vs 0.30-0.45 (terrestrial)
- Wind/turbulence intensity: 0.1/0.08 (aquatic) vs 0.3/0.15 (terrestrial)

No separate "aquatic physics engine" exists. The same Hill/Michaelis-Menten kinetics produce physically appropriate behavior in both environments because the parameters derive from species morphology, not from environment-specific rules.

**Opportunity**: Demonstrate that a single set of first-principles equations can simulate both terrestrial and aquatic ecosystems by varying only species-specific parameters. This "universal physics" approach could interest limnology and marine ecology communities, where separate models are typically used for aquatic vs terrestrial systems. Package as a comparative ecology tool for teaching ecosystem physics across biomes.

---

## Technical Foundation

All opportunities above rest on the completed 5-phase ab initio pipeline:

| Phase | Deliverable | Status |
|-------|------------|--------|
| 1 | Heuristic audit labels | Complete |
| 2 | Binary descriptor cache (585 bytes) | Complete |
| 3 | Regression descriptor tier (OLS, Cholesky) | Complete |
| 4 | Environmental embedding (continuous affinities) | Complete |
| 5 | Accelerator verification (1099+ tests, 0 failures) | Complete |
| 6 | Five physics systems (Wind, Soil, Weather, Spectral, VOC) | Complete |
| 7 | SAS growth pipeline wiring | Complete |
| 8 | Herbivore damage → defense cascade | Complete |

The multi-scale inspection system (ScaleLevel enum, AtomVisual/BondVisual/MolecularDetail/CellularDetail/OrganismComponentDetail structs) provides the data model for all visualization and educational applications.
