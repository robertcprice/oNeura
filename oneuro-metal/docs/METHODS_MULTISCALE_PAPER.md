# Methods: A Multi-Scale Biological Simulation Framework Bridging Quantum Chemistry to Evolutionary Ecology

## Abstract

We present oNeura Terrarium, a 126,000-line Rust framework that integrates seven
scales of biological organization — from quantum electron transfer through
molecular dynamics, cellular metabolism, organismal physiology, population
dynamics, ecological community structure, to evolutionary optimization — in a
single bidirectionally coupled simulation. Each scale employs published,
peer-reviewed kinetic models. The framework produces emergent behaviors including
fitness convergence over evolutionary generations, stress-resilient dormancy
strategies, and genuine Pareto tradeoffs between biomass accumulation and species
diversity. To our knowledge, no existing simulation platform spans all seven
scales in an integrated, spatially explicit system.

---

## 1. Overview of Multi-Scale Architecture

The simulation is organized as a hierarchy of coupled subsystems, each operating
at a characteristic spatiotemporal scale. Data flows bidirectionally through a
central substrate voxel grid that mediates chemical and physical signals between
scales.

| Scale | Spatial Resolution | Temporal Resolution | Model Basis | Lines of Code |
|-------|-------------------|---------------------|-------------|---------------|
| **Quantum** | Orbital (~0.5 Å) | Femtoseconds | Eyring TST, Hartree-Fock profiling | 2,680 |
| **Atomistic** | Atomic (~1 Å) | 1 fs | AMBER/TIP3P, Velocity Verlet | 864 |
| **Metabolic** | Cellular (~1 µm) | Milliseconds | 7-pool Michaelis-Menten | 566 |
| **Cellular** | Gene circuit | Seconds | Gillespie tau-leaping, telegraph model | 469 |
| **Organismal** | Individual (~mm) | Hours | Sharpe-Schoolfield temperature kinetics | 1,143 |
| **Ecological** | Population (~m²) | Days | Lotka-Volterra, Beer-Lambert | 6,339 |
| **Evolutionary** | Meta-population | Generations | NSGA-II multi-objective GA | 4,803 |

The substrate chemistry layer (implemented on GPU via Apple Metal or CUDA)
maintains a 3D voxel grid of chemical species concentrations that all scales
read from and write to, providing the coupling medium.

---

## 2. Quantum Scale: Electron Transfer and Transition-State Theory

### 2.1 Ground-State Electron Configuration

Atomic electron configurations are computed via Aufbau filling of atomic
orbitals (1s, 2s, 2p, 3s, 3p, 4s, 3d, ...) with spatial degeneracy assigned
per subshell (s: 1, p: 3, d: 5, f: 7). The implementation uses:

- Bohr radius: a₀ = 0.529177 Å
- Coulomb constant: k_e = 14.3996 eV·Å/e²
- Proton rest mass: 1.007277 u
- Electron rest mass: 0.000549 u

Hartree-Fock-inspired orbital response fields (ERFs) model the electronic
contribution to reaction barriers. Fragment embedding uses a 20.0 Å cutoff for
collecting candidate atomic sources, and Coulomb contributions below kT/100
are truncated for convergence.

### 2.2 Eyring Transition-State Theory

Reaction rates connecting quantum-derived barrier heights to macroscopic
kinetics are computed via Eyring's equation:

$$k = \frac{k_B T}{h} \exp\left(-\frac{\Delta G^\ddagger}{RT}\right)$$

where $\Delta G^\ddagger = \Delta H^\ddagger - T\Delta S^\ddagger$ is the
activation free energy derived from orbital overlap and binding energy
calculations. This provides the fundamental link between quantum chemistry
and metabolic rate constants used at higher scales.

**References:**
- Eyring, H. (1935) "The Activated Complex in Chemical Reactions." *J. Chem. Phys.* 3:107.
- Marcus, R.A. (1956) "On the theory of oxidation-reduction reactions." *J. Chem. Phys.* 24:966.

---

## 3. Atomistic Scale: Molecular Dynamics with TIP3P Water

### 3.1 Force Field

Non-bonded interactions follow the AMBER force field convention:

**Lennard-Jones 12-6:**
$$V_{LJ}(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]$$

**Coulomb:**
$$V_C(r) = 332.0 \frac{q_i q_j}{r} \quad \text{(kcal/mol)}$$

with cutoff r_c = 12.0 Å and vacuum dielectric (ε = 1.0).

Van der Waals parameters for biologically relevant elements:

| Element | σ (Å) | ε (kcal/mol) | Mass (u) |
|---------|-------|-------------|----------|
| H | 1.20 | 0.0157 | 1.008 |
| C | 1.70 | 0.0860 | 12.011 |
| N | 1.55 | 0.1700 | 14.007 |
| O | 1.52 | 0.1520 | 15.999 |
| S | 1.80 | 0.2500 | 32.065 |
| P | 1.80 | 0.2000 | 30.974 |

### 3.2 Bonded Interactions

**Harmonic bond potential:**
$$V_b(r) = 2k_b(r - r_0)^2$$

For TIP3P water: O–H bond length r₀ = 0.9572 Å, spring constant k_b = 553 kcal/(mol·Å²).
H–O–H angle: θ₀ = 104.52°, k_θ ≈ 100 kcal/(mol·rad²).

### 3.3 Integration

Equations of motion are integrated using the Velocity Verlet algorithm with a
Langevin thermostat for temperature control:

- **Timestep:** Δt = 1 fs (10⁻¹⁵ s)
- **Thermostat:** Langevin friction, configurable damping time
- **Ensemble:** NVT (canonical) or NPT (isothermal-isobaric)

Atomistic probes can be embedded at specific positions within the terrarium
grid, claiming ownership of surrounding voxels. Kinetic energy statistics from
probe MD runs calibrate Arrhenius pre-factors for whole-cell metabolic rates.

**References:**
- Jorgensen, W.L. et al. (1983) "Comparison of simple potential functions for
  simulating liquid water." *J. Chem. Phys.* 79:926.
- Case, D.A. et al. (2005) AMBER 8. University of California, San Francisco.

---

## 4. Metabolic Scale: Seven-Pool Michaelis-Menten Kinetics

### 4.1 Drosophila Metabolic Model

Individual *Drosophila* organisms maintain a 7-pool metabolic state representing
the major energy compartments:

| Pool | Units | Initial State | K_m | V_max |
|------|-------|--------------|-----|-------|
| Crop sugar | mg | 0.0 | 0.1 mg | 0.5 mM/s |
| Hemolymph trehalose | mM | 25.0 | 5.0 mM | 2.0 mM/s |
| Hemolymph glucose | mM | 2.0 | 0.5 mM | 1.0 mM/s |
| Fat body glycogen | mg | 0.015 | 0.01 mg | 0.005 mg/s |
| Fat body lipid | mg | 0.045 | 0.02 mg | 0.003 mg/s |
| Muscle ATP | mM | 4.0 | — | — |
| Muscle ADP | mM | 1.0 | — | — |

All inter-pool fluxes follow Michaelis-Menten kinetics:

$$v = V_{\max} \frac{[S]}{K_m + [S]}$$

### 4.2 Energy Budget

Energy conversions use published thermodynamic values:

- Glycogen energy density: 17.1 J/mg (Wigglesworth, 1949)
- Lipid energy density: 39.3 J/mg
- ATP yield per glucose (aerobic): 36 molecules
- ATP hydrolysis energy: 54 kJ/mol

Activity-dependent metabolic rates:

| Activity State | Metabolic Rate (µW) |
|---------------|---------------------|
| Basal (resting) | 20 |
| Walking | 30 |
| Flight | 100 |

Trehalose satiation threshold: 30 mM. ATP crash telemetry events are emitted
when muscle ATP drops below 0.5 mM.

**References:**
- Edgecomb, R.S. et al. (1994) "Regulation of feeding behavior in adult
  *Drosophila*." *J. Exp. Biol.* 197:215.
- Wigglesworth, V.B. (1949) "The utilization of reserve substances in
  *Drosophila* during flight." *J. Exp. Biol.* 26:150.
- Lehmann, F.-O. & Dickinson, M.H. (2000) "The production of elevated flight
  force compromises manoeuvrability." *J. Exp. Biol.* 203:1613.
- Rulifson, E.J. et al. (2002) "Ablation of insulin-producing neurons in
  flies." *Science* 296:1118.

---

## 5. Cellular Scale: Stochastic Gene Expression via Gillespie Tau-Leaping

### 5.1 Telegraph Promoter Model

Gene expression noise is modeled using a two-state telegraph promoter:

$$\text{OFF} \underset{k_{\text{off}}}{\overset{k_{\text{on}}}{\rightleftharpoons}} \text{ON}$$

When ON, transcription produces mRNA in geometrically distributed bursts.
The analytical steady-state Fano factor is:

$$F = 1 + b \cdot \frac{k_{\text{off}}}{k_{\text{on}} + k_{\text{off}}}$$

where $b$ is the mean burst size.

### 5.2 Default Parameters

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| k_on | 0.01 | s⁻¹ | Telegraph model |
| k_off | 0.1 | s⁻¹ | Telegraph model |
| mRNA burst size | 4.0 | transcripts/burst | Taniguchi et al. 2010 |
| mRNA half-life | 180 | s | Syn3A-scale |
| Protein half-life | 3600 | s | Typical bacterial |
| Translation rate | 0.04 | s⁻¹ per mRNA | Standard |

### 5.3 Algorithm

Propensity-weighted tau-leaping (Gillespie, 1977) with adaptive timestep
ceiling (max_τ = 10.0 s). Five reaction channels: promoter ON→OFF, OFF→ON,
transcription, mRNA decay, translation, protein decay. A fast Xorshift64
PRNG provides deterministic, reproducible noise.

The stochastic layer communicates with the deterministic whole-cell simulator
through an exponential moving average of mRNA burst variance. When the
measured Fano factor exceeds 1.05, protein abundance modulation factors
are applied to downstream metabolic rates.

**References:**
- Gillespie, D.T. (1977) "Exact stochastic simulation of coupled chemical
  reactions." *J. Phys. Chem.* 81:2340.
- Paulsson, J. (2005) "Models of stochastic gene expression." *Phys. Life Rev.* 2:157.
- Raj, A. & van Oudenaarden, A. (2008) "Nature, nurture, or chance." *Cell* 135:216.
- Taniguchi, Y. et al. (2010) "Quantifying E. coli proteome and
  transcriptome." *Science* 329:533.

---

## 6. Organismal Scale: Temperature-Dependent Development

### 6.1 Sharpe-Schoolfield Equation

Development rate as a function of temperature:

$$r(T) = r_{25} \cdot \frac{T}{T_{\text{ref}}} \cdot \frac{\exp\left[\frac{\Delta H_A}{R}\left(\frac{1}{T_{\text{ref}}} - \frac{1}{T}\right)\right]}{1 + \exp\left[\frac{\Delta H_H}{R}\left(\frac{1}{T_H} - \frac{1}{T}\right)\right]}$$

### 6.2 Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| T_ref | 298.15 (25°C) | K |
| ΔH_A (activation enthalpy) | 75 | kJ/mol |
| T_lower (developmental floor) | 12 | °C |
| T_upper (thermal ceiling) | 32 | °C |
| R (gas constant) | 8.314 | J/(mol·K) |

### 6.3 *Drosophila* Lifecycle Stages at 25°C

| Stage | Duration (h) | Notes |
|-------|-------------|-------|
| Embryo | 22 | Oviposition to hatching |
| Larval instar 1 | 24 | |
| Larval instar 2 | 24 | |
| Larval instar 3 | 48 | |
| Pupa | 96 | Metamorphosis |
| Adult | up to 1440 (60 d) | Max lifespan |

Effective timestep: $\Delta t_{\text{eff}} = \Delta t \cdot \exp\left[\frac{\Delta H_A}{R}\left(\frac{1}{T_{\text{ref}}} - \frac{1}{T}\right)\right]$

**Reproductive parameters:**
- Peak oviposition: 60 eggs/day at 25°C
- Lifetime fecundity: ~400 eggs per female
- Egg cost: 2.0 µJ per egg
- Oviposition begins 36 hours post-eclosion

**References:**
- Sharpe, P.J.H. & DeMichele, D.W. (1977) "Reaction kinetics of poikilotherm
  development." *J. Theor. Biol.* 64:649.
- Schoolfield, R.M. et al. (1981) "Non-linear regression of biological
  temperature-dependent rate models." *J. Theor. Biol.* 88:719.
- Ashburner, M. et al. (2005) *Drosophila: A Laboratory Handbook*, 3rd ed.

---

## 7. Ecological Scale: Community Dynamics

### 7.1 Plant Competition: Beer-Lambert Light Attenuation

Light availability at depth z through a plant canopy:

$$I(z) = I_0 \cdot \exp(-k \cdot \text{LAI} \cdot z)$$

- Extinction coefficient: k ≈ 0.5 (mixed canopy)
- **Asymmetric competition**: Only taller plants shade shorter neighbors
  (Weiner, 1990). Canopy overlap decays linearly from full (distance = 0)
  to zero (distance ≥ r₁ + r₂).

**Root competition** follows symmetric nutrient splitting: overlapping root
zones share the local nutrient pool in proportion to root biomass fraction
(Casper & Jackson, 1997).

### 7.2 Soil Fauna: Earthworm Bioturbation

Earthworm population dynamics follow logistic growth:

$$\frac{dN}{dt} = rN\left(1 - \frac{N}{K}\right) - dN$$

| Parameter | Value | Source |
|-----------|-------|--------|
| r (growth rate) | 0.02 d⁻¹ | Lavelle 1988 |
| K (carrying capacity) | 200 ind/m² | Edwards & Bohlen 1996 |
| Feeding rate | 0.02 g OM/(g worm·d) | Field measurements |
| Bioturbation diffusivity | 5.0 mm²/d | Boudreau 1997 |
| Assimilation efficiency | 15% | |
| Death rate | 0.003 d⁻¹ | |

Temperature response: Gaussian centered at T_opt = 12.5°C, σ = 7.0°C.
Hydration response: triangular with minimum 0.15, optimum 0.55, maximum 0.90.

### 7.3 Nematode Grazing: Lotka-Volterra Dynamics

Bacterivorous nematode populations interact with soil microbes through
predator-prey dynamics:

$$\frac{dP}{dt} = \alpha P - \beta PM$$
$$\frac{dM}{dt} = \delta PM - \gamma M$$

| Parameter | Value | Source |
|-----------|-------|--------|
| Carrying capacity | 40 ind/g soil | Yeates et al. 1993 |
| Feeding rate | 5.0 body-weights/d | |
| r_max (growth rate) | 0.10 d⁻¹ | |
| Assimilation efficiency | 0.25 | |
| Death rate | 0.02 d⁻¹ | |
| NH₄⁺ excretion | 0.15 g N/g C consumed | Ingham et al. 1985 |

Nematode excretion accounts for 10–30% of total soil nitrogen flux,
representing a major pathway for N mineralization.

**References:**
- Weiner, J. (1990) "Asymmetric competition in plant populations."
  *Trends Ecol. Evol.* 5:360.
- Casper, B.B. & Jackson, R.B. (1997) "Plant competition underground."
  *Annu. Rev. Ecol. Syst.* 28:545.
- Edwards, C.A. & Bohlen, P.J. (1996) *Biology and Ecology of Earthworms*.
- Boudreau, B.P. (1997) *Diagenetic Models and Their Implementation*.
- Yeates, G.W. et al. (1993) *Soil Biol. Biochem.* 25:869.
- Ingham, R.E. et al. (1985) *Ecol. Monogr.* 55:119.

---

## 8. Evolutionary Scale: NSGA-II Multi-Objective Optimization

### 8.1 Algorithm

The evolutionary engine uses NSGA-II (Deb et al., 2002) to evolve terrarium
world configurations toward multi-objective fitness targets.

**Operators:**
- **Selection:** Binary tournament (size 3)
- **Crossover:** Uniform blend of parameter pairs, rate = 0.7
- **Mutation:** Gaussian perturbation per allele, rate = 0.15
- **Elitism:** Top 2 individuals advance unconditionally

### 8.2 Genome Representation

Each `WorldGenome` encodes 14+ continuous parameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| proton_scale | 0.3–3.0 | Initial soil pH/redox |
| soil_temp | 10–40°C | Baseline temperature |
| water_count | 1–6 | Number of water sources |
| moisture_scale | 0.5–2.0 | Initial soil moisture |
| plant_count | 2–16 | Initial plant population |
| fly_count | 0–6 | Initial Drosophila adults |
| respiration_vmax_scale | 0.3–3.0 | Microbial metabolism scaling |
| photosynthesis_vmax_scale | 0.3–3.0 | Plant photosynthesis scaling |

### 8.3 Fitness Objectives

Seven fitness functions are available for single- or multi-objective
optimization:

1. **MaxBiomass**: plant count × mean canopy + cellular biomass
2. **MaxBiodiversity**: Simpson diversity index across microbial guilds
3. **MaxStability**: inverse variance of biomass over periodic snapshots
4. **MaxCarbonSequestration**: soil organic carbon + plant carbon pools
5. **MaxFruitProduction**: ripe fruit count × resource value
6. **MaxMicrobialHealth**: mean microbial vitality + enzyme probe stability
7. **MaxFlyMetabolism**: fly population × ATP reserves

**Pareto ranking** uses non-dominated sorting with crowding distance to
identify Pareto-optimal tradeoff fronts.

### 8.4 Stress Testing

An environmental variability engine applies:
- **Seasonal sinusoids**: temperature and humidity with configurable
  amplitude, period, and phase
- **Stochastic droughts**: Poisson-triggered events with severity and
  duration parameters
- **Climate presets**: temperate, tropical, and arid regimes

Stress-resilient evolution (`evolve_pareto_stressed`) combines NSGA-II
with periodic environmental perturbations.

**Reference:**
- Deb, K. et al. (2002) "A fast and elitist multiobjective genetic
  algorithm: NSGA-II." *IEEE Trans. Evol. Comput.* 6:182.

---

## 9. Cross-Scale Coupling

### 9.1 Bidirectional Data Flow

```
Quantum (orbital energies, TST barriers)
    ↕ [barrier heights → rate constants]
Atomistic MD (force integration, thermal equilibrium)
    ↕ [kinetic energy → Arrhenius calibration]
Metabolic (Michaelis-Menten pools)
    ↕ [ATP/metabolite concentrations → growth rates]
Cellular (gene expression noise)
    ↕ [protein modulation factors → phenotypic variation]
Organismal (lifecycle, energy budget)
    ↕ [foraging, respiration, reproduction → substrate fluxes]
Ecological (population dynamics)
    ↕ [community fitness → genome evaluation]
Evolutionary (NSGA-II optimization)
```

### 9.2 Temperature Coupling: Arrhenius Q₁₀ Scaling

All metabolic rates are modulated by temperature via Q₁₀ scaling:

$$r(T) = r_{\text{ref}} \cdot 2^{(T - 37°C) / 10°C}$$

Clamped to [0.3, 3.0] for biological realism. Applied to oxidative
phosphorylation efficiency, translation rate, polymerization rate, and
other enzymatic processes.

### 9.3 pH Response

Gaussian response centered at physiological optimum:

$$f(\text{pH}) = \exp\left[-0.5 \cdot \left(\frac{\text{pH} - 7.2}{1.2}\right)^2\right]$$

Clamped to [0.3, 1.0].

### 9.4 Oxygen Availability

$$f(O_2) = \left(\frac{[O_2]}{0.21}\right) \quad \text{clamped to [0.1, 1.5]}$$

Modulates OXPHOS efficiency linearly. Respiratory coupling: 6 mol O₂
consumed per glucose oxidized, with respiratory quotient = 1.0.

### 9.5 Spatial Field Coupling

Chemical signals are deposited on the substrate grid using Gaussian kernels:

$$\text{deposit}(x, y) = A \cdot \exp\left(-\frac{\Delta x^2 + \Delta y^2}{2\sigma^2}\right)$$

where σ = 0.72 × radius. This mediates plant root exudates, earthworm
fecal pellets, microbial metabolite diffusion, and atmospheric gas exchange.

### 9.6 Microdomain Ownership

A per-cell authority map (`SoilOwnershipClass`) assigns spatial control:
- Plant root zones claim surrounding voxels
- Explicit microbial cohorts set spatial authority masks
- Atomistic probes claim rectangular patches for MD simulation
- Background soil biogeochemistry operates only on unclaimed cells

---

## 10. Implementation

### 10.1 Language and Performance

The framework is implemented in Rust (126,000 lines) for memory safety and
performance. GPU substrate chemistry runs on Apple Metal or CUDA. The
`#[repr(C)]` struct layout ensures byte-exact correspondence between CPU
and GPU data structures.

### 10.2 Testing

748 test functions cover all scales. A regression suite of 165 tests verifies
cross-scale coupling integrity. Tests are deterministic (seeded RNG) and run
without GPU hardware (CPU fallback mode).

### 10.3 Reproducibility

All simulations use seeded pseudo-random number generators (`StdRng` with
user-specified seeds). Evolution experiments are fully reproducible given
identical seeds and parameters. Results can be exported as JSON bundles for
analysis in R or Python.

---

## References

1. Ashburner, M., Golic, K.G. & Hawley, R.S. (2005) *Drosophila: A Laboratory Handbook*. 3rd ed. Cold Spring Harbor Laboratory Press.
2. Boudreau, B.P. (1997) *Diagenetic Models and Their Implementation*. Springer.
3. Case, D.A. et al. (2005) "The Amber biomolecular simulation programs." *J. Comput. Chem.* 26:1668.
4. Casper, B.B. & Jackson, R.B. (1997) "Plant competition underground." *Annu. Rev. Ecol. Syst.* 28:545.
5. Deb, K. et al. (2002) "A fast and elitist multiobjective genetic algorithm: NSGA-II." *IEEE Trans. Evol. Comput.* 6:182.
6. Edgecomb, R.S. et al. (1994) "Regulation of feeding behavior in adult *Drosophila melanogaster*." *J. Exp. Biol.* 197:215.
7. Edwards, C.A. & Bohlen, P.J. (1996) *Biology and Ecology of Earthworms*. 3rd ed. Chapman & Hall.
8. Eyring, H. (1935) "The activated complex in chemical reactions." *J. Chem. Phys.* 3:107.
9. Gillespie, D.T. (1977) "Exact stochastic simulation of coupled chemical reactions." *J. Phys. Chem.* 81:2340.
10. Ingham, R.E. et al. (1985) "Interactions of bacteria, fungi, and their nematode grazers." *Ecol. Monogr.* 55:119.
11. Jorgensen, W.L. et al. (1983) "Comparison of simple potential functions for simulating liquid water." *J. Chem. Phys.* 79:926.
12. Lavelle, P. (1988) "Earthworm activities and the soil system." *Biol. Fertil. Soils* 6:237.
13. Lehmann, F.-O. & Dickinson, M.H. (2000) "The production of elevated flight force compromises manoeuvrability in *Drosophila*." *J. Exp. Biol.* 203:1613.
14. Marcus, R.A. (1956) "On the theory of oxidation-reduction reactions involving electron transfer." *J. Chem. Phys.* 24:966.
15. Monsi, M. & Saeki, T. (1953) "Über den Lichtfaktor in den Pflanzengesellschaften und seine Bedeutung für die Stoffproduktion." *Jpn. J. Bot.* 14:22.
16. Paulsson, J. (2005) "Models of stochastic gene expression." *Phys. Life Rev.* 2:157.
17. Raj, A. & van Oudenaarden, A. (2008) "Nature, nurture, or chance: stochastic gene expression and its consequences." *Cell* 135:216.
18. Rulifson, E.J. et al. (2002) "Ablation of insulin-producing neurons in flies: growth and diabetic phenotypes." *Science* 296:1118.
19. Schoolfield, R.M. et al. (1981) "Non-linear regression of biological temperature-dependent rate models." *J. Theor. Biol.* 88:719.
20. Sharpe, P.J.H. & DeMichele, D.W. (1977) "Reaction kinetics of poikilotherm development." *J. Theor. Biol.* 64:649.
21. Taniguchi, Y. et al. (2010) "Quantifying *E. coli* proteome and transcriptome with single-molecule sensitivity in single cells." *Science* 329:533.
22. Weiner, J. (1990) "Asymmetric competition in plant populations." *Trends Ecol. Evol.* 5:360.
23. Wigglesworth, V.B. (1949) "The utilization of reserve substances in *Drosophila* during flight." *J. Exp. Biol.* 26:150.
24. Yeates, G.W. et al. (1993) "Feeding habits in soil nematode families and genera." *Soil Biol. Biochem.* 25:869.

---

## Supplementary Information

### S1. Software Availability

The oNeura Terrarium source code is available at
https://github.com/bobbyprice/oNeura under the MIT license. Compiled binaries
are provided for macOS (ARM64) and Linux (x86_64).

### S2. Standalone Tools

Two standalone CLI tools are packaged from the framework:

**Drug Protocol Optimizer** (`drug_optimizer`): Compares single, pulsed, and
combination antibiotic treatment strategies against bacterial persister cells.
Validated against *E. coli* persistence data from Balaban et al. (2004).

**Gene Circuit Noise Designer** (`gene_circuit`): Designs synthetic gene
circuits with target noise properties (Fano factor, mean protein, CV) using
evolutionary optimization over the telegraph promoter model.

### S3. Computational Requirements

Tested on Apple M1 (16 GB RAM). Full evolution experiment (8 population,
5 generations, 100 frames per evaluation): ~15 seconds. GPU substrate
chemistry (64×64 grid): ~2 ms per frame on Metal.
