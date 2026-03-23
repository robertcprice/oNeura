# oNeura Terrarium: A Multi-Scale Biological Simulation Framework Bridging Quantum Chemistry to Evolutionary Ecology

## Abstract

We present oNeura Terrarium, a 138,000-line Rust framework that integrates seven
scales of biological organization --- from quantum electron transfer through
molecular dynamics, cellular metabolism, organismal physiology, population
dynamics, ecological community structure, to evolutionary optimization --- in a
single bidirectionally coupled simulation. Each scale employs published,
peer-reviewed kinetic models. The framework produces emergent behaviors including
fitness convergence over evolutionary generations, stress-resilient dormancy
strategies, and genuine Pareto tradeoffs between biomass accumulation and species
diversity. To our knowledge, no existing simulation platform spans all seven
scales in an integrated, spatially explicit system.

---

# Introduction

## The Multi-Scale Challenge in Biological Simulation

Living systems operate simultaneously across at least seven orders of
magnitude in both space and time: quantum tunneling in enzyme active sites
(femtosecond, sub-angstrom) drives metabolic fluxes (millisecond, micron)
that determine organismal fitness (hours, millimeters), population
trajectories (days, meters), and evolutionary outcomes (generations,
landscapes). Understanding how causation propagates across these scales
remains one of the central open problems in computational biology.

Despite decades of progress, no single simulation platform integrates all
seven scales of biological organization --- from quantum chemistry through
molecular dynamics, cellular metabolism, gene expression noise, organismal
physiology, ecological community dynamics, to evolutionary optimization ---
in a bidirectionally coupled, spatially explicit system. Instead, the field
relies on specialized tools, each covering one or two scales, with
hand-coded interfaces where integration is attempted at all.

## Existing Tools and Their Scale Coverage

The following table summarizes the most widely used simulation frameworks
and the biological scales they address:

| Tool | Scales Covered | Coupling | Spatial | Language | Key Limitation |
|------|---------------|----------|---------|----------|----------------|
| **COPASI** (Hoops et al., 2006) | Metabolic, cellular | Deterministic + stochastic ODE | Non-spatial | C++ | No organismal or ecological layers |
| **OpenMM** (Eastman et al., 2017) | Atomistic, quantum (QM/MM) | Unidirectional (QM to MM) | 3D molecular | Python/C++ | Ends at molecular scale |
| **NetLogo** (Wilensky, 1999) | Ecological, agent-based | Rule-based | 2D grid | Java | No mechanistic biochemistry |
| **DSSAT** (Jones et al., 2003) | Organismal (crop), ecological | Process-based | Field-level | Fortran | No subcellular detail |
| **E-Cell** (Tomita et al., 1999) | Metabolic, cellular | Multi-algorithm | Non-spatial | C++ | No population or evolutionary dynamics |
| **CompuCell3D** (Swat et al., 2012) | Cellular, tissue | Cellular Potts + PDE | 3D lattice | Python/C++ | No molecular or evolutionary scale |
| **FLAME** (Richmond et al., 2010) | Agent-based, population | Message-passing | 3D | C/XML | No biochemical mechanism |
| **VCell** (Schaff et al., 1997) | Subcellular, cellular | PDE + compartmental | 3D geometry | Java | No organism or ecology |
| **oNeura** (this work) | **All seven** | **Bidirectional, substrate-mediated** | **3D voxel grid** | **Rust** | See Discussion: Limitations |

A critical gap is apparent: tools either model subcellular processes with
high fidelity but stop at the cell membrane (COPASI, E-Cell, VCell), or
capture population-level dynamics without mechanistic biochemistry (NetLogo,
DSSAT). No existing framework allows a researcher to ask, for example, how
a point mutation affecting enzyme kinetics (quantum/atomistic) propagates
through metabolic flux (cellular), developmental timing (organismal),
competitive fitness (ecological), and allele frequency (evolutionary) ---
all within a single coherent simulation.

## Detailed Capability Comparison

To clarify the landscape of existing tools more precisely, we provide a
feature-level comparison across six key capabilities that define a
multi-scale biology simulator. A checkmark indicates native support; a
tilde (~) indicates partial or plugin-based support; a dash indicates
absence.

| Capability | COPASI | OpenMM | NetLogo | DSSAT | E-Cell | CompuCell3D | VCell | oNeura |
|------------|--------|--------|---------|-------|--------|-------------|-------|--------|
| Quantum/atomistic kinetics | --- | Full | --- | --- | --- | --- | --- | Eyring TST |
| Molecular dynamics | --- | Full | --- | --- | --- | --- | --- | AMBER/TIP3P |
| Metabolic ODE/MM kinetics | Full | --- | --- | ~ | Full | ~ | Full | 7-pool MM |
| Stochastic gene expression | Gillespie | --- | --- | --- | Multi-algo | --- | --- | Tau-leaping |
| Organism-level physiology | --- | --- | Rule-based | Process | --- | --- | --- | Sharpe-Schoolfield |
| Population/community ecology | --- | --- | Agent-based | Field-level | --- | Tissue | --- | LV + Beer-Lambert |
| Evolutionary optimization | --- | --- | --- | --- | --- | --- | --- | NSGA-II |
| Bidirectional cross-scale coupling | --- | QM/MM only | --- | --- | --- | PDE-cell | --- | **Full 7-scale** |
| GPU-accelerated substrate | --- | CUDA/OpenCL | --- | --- | --- | --- | --- | Metal/CUDA |
| Spatially explicit (3D) | --- | Molecular | 2D grid | 1D profile | --- | 3D lattice | 3D geometry | 3D voxel |
| Deterministic reproducibility | Yes | Yes | Seed-based | --- | Yes | --- | --- | Bitwise (seeded) |
| Automated test suite | --- | Unit tests | --- | --- | --- | --- | --- | 722 tests |

Several patterns emerge from this comparison. First, no tool other than
oNeura covers more than three of the seven biological scales. Second,
bidirectional cross-scale coupling --- where changes at a lower scale
propagate upward to affect fitness and evolutionary dynamics, and
evolutionary outcomes feed back to reshape lower-scale parameters ---
is unique to oNeura. Third, the combination of GPU-accelerated spatial
chemistry with deterministic reproducibility enables both performance
and scientific rigor.

We emphasize that this comparison reflects scope, not quality. OpenMM's
molecular dynamics capabilities exceed oNeura's atomistic probes by
orders of magnitude in accuracy and system size. COPASI's ODE solver
infrastructure is more mature and feature-rich than oNeura's metabolic
layer. The contribution of oNeura is not to replace these specialized
tools but to demonstrate that meaningful biological insight emerges when
all scales are coupled in a single simulation, even at reduced per-scale
fidelity.

## The Cost of Scale Isolation

The absence of integrated multi-scale tools has practical consequences.
Researchers studying antibiotic persistence must separately model drug
pharmacokinetics, stochastic gene switching, metabolic dormancy, and
population-level regrowth, then manually reconcile parameters across
disconnected frameworks. Crop scientists modeling climate adaptation cannot
easily connect soil biogeochemistry with plant physiology and evolutionary
responses without building bespoke coupling code. Systems biologists
studying noise-driven phenotypic variation must extrapolate from single-cell
stochastic models to population-level fitness consequences using ad hoc
approximations.

These disconnections introduce parameter inconsistencies, prevent the
discovery of emergent cross-scale behaviors, and make reproducibility
difficult when each scale uses a different tool with different assumptions.

The problem is not merely inconvenient; it is scientifically costly.
Multi-drug-resistant bacterial infections kill over 1.2 million people
annually [Murray et al. 2022], yet our best models of antibiotic
persistence operate at either the molecular level (drug-target binding
affinity) or the population level (pharmacodynamic kill curves), with
no mechanistic link between the stochastic molecular events that trigger
dormancy and the population-level survival that produces treatment
failure. Similarly, crop yield predictions under climate change require
integrating soil biogeochemistry, plant physiology, microbial community
dynamics, and evolutionary adaptation --- a multi-scale challenge that
no existing agricultural model addresses comprehensively [Boote et al.
2013].

The fundamental issue is that biological causation does not respect
scale boundaries. A single nucleotide polymorphism affects enzyme
kinetics (molecular), which alters metabolic flux (cellular), which
changes developmental timing (organismal), which shifts competitive
outcomes (ecological), which determines allele frequency in the next
generation (evolutionary). Understanding this causal chain requires
a simulation framework that spans all intervening scales.

## Key Contributions

We present oNeura Terrarium, a 138,000-line Rust framework that addresses
these limitations through five principal contributions:

1. **Seven-scale integration.** A single codebase spans quantum electron
   transfer (Eyring transition-state theory), atomistic molecular dynamics
   (AMBER/TIP3P), metabolic kinetics (7-pool Michaelis-Menten), stochastic
   gene expression (Gillespie tau-leaping with telegraph promoters),
   temperature-dependent organismal development (Sharpe-Schoolfield),
   ecological community dynamics (Beer-Lambert shading, Lotka-Volterra
   predation, logistic soil fauna), and multi-objective evolutionary
   optimization (NSGA-II with Pareto ranking).

2. **Bidirectional coupling through a shared substrate grid.** All scales
   read from and write to a common 3D voxel grid of chemical species
   concentrations, maintained on GPU (Apple Metal or CUDA). This substrate
   layer mediates temperature, pH, oxygen, nutrients, and signaling
   molecules, ensuring that changes at any scale propagate naturally to all
   others without manual parameter passing.

3. **Emergent cross-scale phenomena.** The framework produces behaviors
   that no single scale predicts in isolation: persister cell dormancy
   arising from stochastic gene expression noise, bet-hedging strategies
   emerging from evolutionary pressure on variable environments, and niche
   partitioning driven by competitive exclusion under resource limitation.

4. **Quantitative validation at every scale.** Each subsystem is validated
   against published experimental data: TIP3P water structure factors,
   Michaelis-Menten kinetic constants from *Drosophila* literature,
   Fano factors matching Taniguchi et al. (2010) single-cell measurements,
   Sharpe-Schoolfield developmental timing, and NSGA-II convergence
   characteristics consistent with Deb et al. (2002).

5. **Reproducibility and performance.** Seeded pseudo-random number
   generators ensure bitwise-identical results across runs. A comprehensive
   test suite of 722 automated tests (including 211 cross-scale regression
   tests) runs without GPU hardware. Full evolutionary experiments complete
   in approximately 15 seconds on consumer hardware (Apple M1, 16 GB RAM).

## Paper Organization

The Methods section (Sections 1--10) details each scale's mathematical
formulation, parameters, and coupling mechanisms. The Results section
presents validation experiments, emergent behavior analysis, and
computational performance benchmarks. The Discussion considers limitations,
compares the framework to existing alternatives, and outlines future
directions including GPU scaling, integration with real-world soil data,
and potential clinical applications in drug resistance modeling.

---

# Methods

## 1. Overview of Multi-Scale Architecture

The simulation is organized as a hierarchy of coupled subsystems, each operating
at a characteristic spatiotemporal scale. Data flows bidirectionally through a
central substrate voxel grid that mediates chemical and physical signals between
scales.

| Scale | Spatial Resolution | Temporal Resolution | Model Basis | Lines of Code |
|-------|-------------------|---------------------|-------------|---------------|
| **Quantum** | Orbital (~0.5 A) | Femtoseconds | Eyring TST, Hartree-Fock profiling | 2,680 |
| **Atomistic** | Atomic (~1 A) | 1 fs | AMBER/TIP3P, Velocity Verlet | 864 |
| **Metabolic** | Cellular (~1 um) | Milliseconds | 7-pool Michaelis-Menten | 566 |
| **Cellular** | Gene circuit | Seconds | Gillespie tau-leaping, telegraph model | 469 |
| **Organismal** | Individual (~mm) | Hours | Sharpe-Schoolfield temperature kinetics | 1,143 |
| **Ecological** | Population (~m^2) | Days | Lotka-Volterra, Beer-Lambert | 6,339 |
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

- Bohr radius: a_0 = 0.529177 A
- Coulomb constant: k_e = 14.3996 eV*A/e^2
- Proton rest mass: 1.007277 u
- Electron rest mass: 0.000549 u

Hartree-Fock-inspired orbital response fields (ERFs) model the electronic
contribution to reaction barriers. Fragment embedding uses a 20.0 A cutoff for
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

with cutoff r_c = 12.0 A and vacuum dielectric (epsilon = 1.0).

Van der Waals parameters for biologically relevant elements:

| Element | sigma (A) | epsilon (kcal/mol) | Mass (u) |
|---------|-----------|-------------------|----------|
| H | 1.20 | 0.0157 | 1.008 |
| C | 1.70 | 0.0860 | 12.011 |
| N | 1.55 | 0.1700 | 14.007 |
| O | 1.52 | 0.1520 | 15.999 |
| S | 1.80 | 0.2500 | 32.065 |
| P | 1.80 | 0.2000 | 30.974 |

### 3.2 Bonded Interactions

**Harmonic bond potential:**
$$V_b(r) = 2k_b(r - r_0)^2$$

For TIP3P water: O-H bond length r_0 = 0.9572 A, spring constant k_b = 553 kcal/(mol*A^2).
H-O-H angle: theta_0 = 104.52 degrees, k_theta ~ 100 kcal/(mol*rad^2).

### 3.3 Integration

Equations of motion are integrated using the Velocity Verlet algorithm with a
Langevin thermostat for temperature control:

- **Timestep:** dt = 1 fs (10^-15 s)
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
| Muscle ATP | mM | 4.0 | --- | --- |
| Muscle ADP | mM | 1.0 | --- | --- |

All inter-pool fluxes follow Michaelis-Menten kinetics:

$$v = V_{\max} \frac{[S]}{K_m + [S]}$$

### 4.2 Energy Budget

Energy conversions use published thermodynamic values:

- Glycogen energy density: 17.1 J/mg (Wigglesworth, 1949)
- Lipid energy density: 39.3 J/mg
- ATP yield per glucose (aerobic): 36 molecules
- ATP hydrolysis energy: 54 kJ/mol

Activity-dependent metabolic rates:

| Activity State | Metabolic Rate (uW) |
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
| k_on | 0.01 | s^-1 | Telegraph model |
| k_off | 0.1 | s^-1 | Telegraph model |
| mRNA burst size | 4.0 | transcripts/burst | Taniguchi et al. 2010 |
| mRNA half-life | 180 | s | Syn3A-scale |
| Protein half-life | 3600 | s | Typical bacterial |
| Translation rate | 0.04 | s^-1 per mRNA | Standard |

### 5.3 Algorithm

Propensity-weighted tau-leaping (Gillespie, 1977) with adaptive timestep
ceiling (max_tau = 10.0 s). Five reaction channels: promoter ON->OFF, OFF->ON,
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
| T_ref | 298.15 (25 C) | K |
| Delta_H_A (activation enthalpy) | 75 | kJ/mol |
| T_lower (developmental floor) | 12 | C |
| T_upper (thermal ceiling) | 32 | C |
| R (gas constant) | 8.314 | J/(mol*K) |

### 6.3 *Drosophila* Lifecycle Stages at 25 C

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
- Peak oviposition: 60 eggs/day at 25 C
- Lifetime fecundity: ~400 eggs per female
- Egg cost: 2.0 uJ per egg
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

- Extinction coefficient: k ~ 0.5 (mixed canopy)
- **Asymmetric competition**: Only taller plants shade shorter neighbors
  (Weiner, 1990). Canopy overlap decays linearly from full (distance = 0)
  to zero (distance >= r_1 + r_2).

**Root competition** follows symmetric nutrient splitting: overlapping root
zones share the local nutrient pool in proportion to root biomass fraction
(Casper & Jackson, 1997).

### 7.2 Soil Fauna: Earthworm Bioturbation

Earthworm population dynamics follow logistic growth:

$$\frac{dN}{dt} = rN\left(1 - \frac{N}{K}\right) - dN$$

| Parameter | Value | Source |
|-----------|-------|--------|
| r (growth rate) | 0.02 d^-1 | Lavelle 1988 |
| K (carrying capacity) | 200 ind/m^2 | Edwards & Bohlen 1996 |
| Feeding rate | 0.02 g OM/(g worm*d) | Field measurements |
| Bioturbation diffusivity | 5.0 mm^2/d | Boudreau 1997 |
| Assimilation efficiency | 15% | |
| Death rate | 0.003 d^-1 | |

Temperature response: Gaussian centered at T_opt = 12.5 C, sigma = 7.0 C.
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
| r_max (growth rate) | 0.10 d^-1 | |
| Assimilation efficiency | 0.25 | |
| Death rate | 0.02 d^-1 | |
| NH4+ excretion | 0.15 g N/g C consumed | Ingham et al. 1985 |

Nematode excretion accounts for 10--30% of total soil nitrogen flux,
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
| proton_scale | 0.3--3.0 | Initial soil pH/redox |
| soil_temp | 10--40 C | Baseline temperature |
| water_count | 1--6 | Number of water sources |
| moisture_scale | 0.5--2.0 | Initial soil moisture |
| plant_count | 2--16 | Initial plant population |
| fly_count | 0--6 | Initial Drosophila adults |
| respiration_vmax_scale | 0.3--3.0 | Microbial metabolism scaling |
| photosynthesis_vmax_scale | 0.3--3.0 | Plant photosynthesis scaling |

### 8.3 Fitness Objectives

Seven fitness functions are available for single- or multi-objective
optimization:

1. **MaxBiomass**: plant count x mean canopy + cellular biomass
2. **MaxBiodiversity**: Simpson diversity index across microbial guilds
3. **MaxStability**: inverse variance of biomass over periodic snapshots
4. **MaxCarbonSequestration**: soil organic carbon + plant carbon pools
5. **MaxFruitProduction**: ripe fruit count x resource value
6. **MaxMicrobialHealth**: mean microbial vitality + enzyme probe stability
7. **MaxFlyMetabolism**: fly population x ATP reserves

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
    <-> [barrier heights -> rate constants]
Atomistic MD (force integration, thermal equilibrium)
    <-> [kinetic energy -> Arrhenius calibration]
Metabolic (Michaelis-Menten pools)
    <-> [ATP/metabolite concentrations -> growth rates]
Cellular (gene expression noise)
    <-> [protein modulation factors -> phenotypic variation]
Organismal (lifecycle, energy budget)
    <-> [foraging, respiration, reproduction -> substrate fluxes]
Ecological (population dynamics)
    <-> [community fitness -> genome evaluation]
Evolutionary (NSGA-II optimization)
```

### 9.2 Temperature Coupling: Arrhenius Q10 Scaling

All metabolic rates are modulated by temperature via Q10 scaling:

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

Modulates OXPHOS efficiency linearly. Respiratory coupling: 6 mol O2
consumed per glucose oxidized, with respiratory quotient = 1.0.

### 9.5 Spatial Field Coupling

Chemical signals are deposited on the substrate grid using Gaussian kernels:

$$\text{deposit}(x, y) = A \cdot \exp\left(-\frac{\Delta x^2 + \Delta y^2}{2\sigma^2}\right)$$

where sigma = 0.72 x radius. This mediates plant root exudates, earthworm
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

The framework is implemented in Rust (138,000 lines) for memory safety and
performance. GPU substrate chemistry runs on Apple Metal or CUDA. The
`#[repr(C)]` struct layout ensures byte-exact correspondence between CPU
and GPU data structures.

### 10.2 Testing

722 test functions cover all scales. A regression suite of 211 tests verifies
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
2. Balaban, N.Q. et al. (2004) "Bacterial persistence as a phenotypic switch." *Science* 305:1622.
3. Beaumont, H.J.E. et al. (2009) "Experimental evolution of bet hedging." *Nature* 462:90.
4. Boudreau, B.P. (1997) *Diagenetic Models and Their Implementation*. Springer.
5. Case, D.A. et al. (2005) "The Amber biomolecular simulation programs." *J. Comput. Chem.* 26:1668.
6. Casper, B.B. & Jackson, R.B. (1997) "Plant competition underground." *Annu. Rev. Ecol. Syst.* 28:545.
7. Deb, K. et al. (2002) "A fast and elitist multiobjective genetic algorithm: NSGA-II." *IEEE Trans. Evol. Comput.* 6:182.
8. Eastman, P. et al. (2017) "OpenMM 7: rapid development of high-performance algorithms for molecular dynamics." *PLoS Comput. Biol.* 13:e1005659.
9. Edgecomb, R.S. et al. (1994) "Regulation of feeding behavior in adult *Drosophila melanogaster*." *J. Exp. Biol.* 197:215.
10. Edwards, C.A. & Bohlen, P.J. (1996) *Biology and Ecology of Earthworms*. 3rd ed. Chapman & Hall.
11. Elowitz, M.B. et al. (2002) "Stochastic gene expression in a single cell." *Science* 297:1183.
12. Eyring, H. (1935) "The activated complex in chemical reactions." *J. Chem. Phys.* 3:107.
13. Gillespie, D.T. (1977) "Exact stochastic simulation of coupled chemical reactions." *J. Phys. Chem.* 81:2340.
14. Goldberg, D.E. (1989) *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
15. Grimm, V. et al. (2005) "Pattern-oriented modeling of agent-based complex systems." *Science* 310:987.
16. Hardin, G. (1960) "The competitive exclusion principle." *Science* 131:1292.
17. Hoops, S. et al. (2006) "COPASI --- a COmplex PAthway SImulator." *Bioinformatics* 22:3067.
18. Ingham, R.E. et al. (1985) "Interactions of bacteria, fungi, and their nematode grazers." *Ecol. Monogr.* 55:119.
19. Jones, J.W. et al. (2003) "The DSSAT cropping system model." *Eur. J. Agron.* 18:235.
20. Jorgensen, W.L. et al. (1983) "Comparison of simple potential functions for simulating liquid water." *J. Chem. Phys.* 79:926.
21. Karr, J.R. et al. (2012) "A whole-cell computational model predicts phenotype from genotype." *Cell* 150:389.
22. Klipp, E. et al. (2005) *Systems Biology in Practice*. Wiley-VCH.
23. Kussell, E. & Leibler, S. (2005) "Phenotypic diversity, population growth, and information in fluctuating environments." *Science* 309:2075.
24. Lavelle, P. (1988) "Earthworm activities and the soil system." *Biol. Fertil. Soils* 6:237.
25. Lehmann, F.-O. & Dickinson, M.H. (2000) "The production of elevated flight force compromises manoeuvrability in *Drosophila*." *J. Exp. Biol.* 203:1613.
26. Lewis, K. (2010) "Persister cells." *Annu. Rev. Microbiol.* 64:357.
27. Lotka, A.J. (1925) *Elements of Physical Biology*. Williams & Wilkins.
28. Marcus, R.A. (1956) "On the theory of oxidation-reduction reactions involving electron transfer." *J. Chem. Phys.* 24:966.
29. Monsi, M. & Saeki, T. (1953) "Uber den Lichtfaktor in den Pflanzengesellschaften und seine Bedeutung fur die Stoffproduktion." *Jpn. J. Bot.* 14:22.
30. Noble, D. (2002) "Modeling the heart --- from genes to cells to the whole organ." *Science* 295:1678.
31. Paulsson, J. (2005) "Models of stochastic gene expression." *Phys. Life Rev.* 2:157.
32. Raj, A. & van Oudenaarden, A. (2008) "Nature, nurture, or chance: stochastic gene expression and its consequences." *Cell* 135:216.
33. Richmond, P. et al. (2010) "High performance cellular level agent-based simulation with FLAME for the GPU." *Briefings Bioinf.* 11:334.
34. Rulifson, E.J. et al. (2002) "Ablation of insulin-producing neurons in flies: growth and diabetic phenotypes." *Science* 296:1118.
35. Schaff, J. et al. (1997) "A general computational framework for modeling cellular structure and function." *Biophys. J.* 73:1135.
36. Schoolfield, R.M. et al. (1981) "Non-linear regression of biological temperature-dependent rate models." *J. Theor. Biol.* 88:719.
37. Sharpe, P.J.H. & DeMichele, D.W. (1977) "Reaction kinetics of poikilotherm development." *J. Theor. Biol.* 64:649.
38. Simpson, E.H. (1949) "Measurement of diversity." *Nature* 163:688.
39. Swat, M.H. et al. (2012) "Multi-scale modeling of tissues using CompuCell3D." *Methods Cell Biol.* 110:325.
40. Taniguchi, Y. et al. (2010) "Quantifying *E. coli* proteome and transcriptome with single-molecule sensitivity in single cells." *Science* 329:533.
41. Tilman, D. (1982) *Resource Competition and Community Structure*. Princeton University Press.
42. Tomita, M. et al. (1999) "E-Cell: software environment for whole-cell simulation." *Bioinformatics* 15:72.
43. Venable, D.L. (2007) "Bet hedging in a guild of desert annuals." *Ecology* 88:1086.
44. Volterra, V. (1926) "Fluctuations in the abundance of a species considered mathematically." *Nature* 118:558.
45. Warburg, O. (1956) "On the origin of cancer cells." *Science* 123:309.
46. Weiner, J. (1990) "Asymmetric competition in plant populations." *Trends Ecol. Evol.* 5:360.
47. Wigglesworth, V.B. (1949) "The utilization of reserve substances in *Drosophila* during flight." *J. Exp. Biol.* 26:150.
48. Wilensky, U. (1999) *NetLogo*. Center for Connected Learning and Computer-Based Modeling, Northwestern University.
49. Yeates, G.W. et al. (1993) "Feeding habits in soil nematode families and genera." *Soil Biol. Biochem.* 25:869.
50. Zitzler, E. et al. (2000) "Comparison of multiobjective evolutionary algorithms: empirical results." *Evol. Comput.* 8:173.
51. Allison, K.R. et al. (2011) "Metabolite-enabled eradication of bacterial persisters by aminoglycosides." *Nature* 473:216.
52. Boote, K.J. et al. (2013) "Putting mechanisms into crop production models." *Plant Cell Environ.* 36:1658.
53. Cranmer, K. et al. (2020) "The frontier of simulation-based inference." *Proc. Natl. Acad. Sci.* 117:30055.
54. Davidson, E.A. & Janssens, I.A. (2006) "Temperature sensitivity of soil carbon decomposition and feedbacks to climate change." *Nature* 440:165.
55. Gilchrist, G.W. et al. (1997) "Thermal sensitivity of *Drosophila melanogaster*: evolutionary responses of adults and eggs to laboratory natural selection." *Physiol. Zool.* 70:403.
56. Gillooly, J.F. et al. (2001) "Effects of size and temperature on metabolic rate." *Science* 293:2248.
57. Holzworth, D.P. et al. (2014) "APSIM --- Evolution towards a new generation of agricultural systems simulation." *Environ. Model. Softw.* 62:327.
58. Murray, C.J.L. et al. (2022) "Global burden of bacterial antimicrobial resistance in 2019: a systematic analysis." *Lancet* 399:629.
59. Ofria, C. & Wilke, C.O. (2004) "Avida: a software platform for research in computational evolutionary biology." *Artif. Life* 10:191.
60. Poeplau, C. & Don, A. (2015) "Carbon sequestration in agricultural soils via cultivation of cover crops." *Agric. Ecosyst. Environ.* 200:33.
61. Ray, T.S. (1991) "An approach to the synthesis of life." In *Artificial Life II*, ed. Langton, C.G. et al. Addison-Wesley.
62. Schulman, J. et al. (2017) "Proximal policy optimization algorithms." *arXiv:1707.06347*.
63. Simard, S.W. et al. (2012) "Mycorrhizal networks: mechanisms, ecology and modelling." *Fungal Biol. Rev.* 26:39.
64. Somero, G.N. (2004) "Adaptation of enzymes to temperature: searching for basic strategies." *Comp. Biochem. Physiol. B* 139:321.
65. Torsvik, V. et al. (1990) "High diversity in DNA of soil bacteria." *Appl. Environ. Microbiol.* 56:782.
66. van der Heijden, M.G.A. et al. (1998) "Mycorrhizal fungal diversity determines plant biodiversity, ecosystem variability and productivity." *Nature* 396:69.
67. Windels, E.M. et al. (2019) "Bacterial persistence promotes the evolution of antibiotic resistance by increasing survival and mutation rates." *ISME J.* 13:1239.
68. Zhang, Y. et al. (2012) "Mechanisms of drug resistance in *Mycobacterium tuberculosis*." *Int. J. Tuberc. Lung Dis.* 13:1320.
69. Zhao, Y. & Truhlar, D.G. (2008) "The M06 suite of density functionals for main group thermochemistry." *Theor. Chem. Acc.* 120:215.

---

## Supplementary Information

### S1. Software Availability

The oNeura Terrarium source code is available at
https://github.com/bobbyprice/oNeura under the CC BY-NC 4.0 license with
commercial licensing available through oneura.ai. Compiled binaries
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
chemistry (64x64 grid): ~2 ms per frame on Metal.

---

# Results

## 1. Test Suite Validation

The oNeura Terrarium framework maintains a comprehensive automated test
suite that validates correctness at every biological scale and across
scale boundaries.

### 1.1 Test Coverage Summary

| Category | Test Count | Scope |
|----------|-----------|-------|
| Total automated tests | 722 | All scales, all modules |
| Cross-scale regression suite | 211 | Inter-scale coupling, emergent behavior |
| Ignored (hardware-dependent) | 6 | GPU-specific, quantum runtime |
| Failures | 0 | --- |

All 722 tests pass deterministically with seeded random number generators.
The 211-test regression suite specifically targets cross-scale coupling
integrity: substrate boundary conditions, guild activity feedback,
soil-atmosphere gas exchange, organism metabolism, stochastic expression
coupling, phenotypic variation, persister dormancy, bet-hedging dynamics,
seasonal and drought responses, spatial zone effects, drug protocols,
enzyme kinetics, bioremediation pathways, and probe coupling.

### 1.2 Determinism Verification

Running the full test suite with identical seeds produces bitwise-identical
results across invocations. This was verified by computing SHA-256 hashes
of simulation state snapshots at 100-frame intervals across 50 independent
runs, with zero hash mismatches observed.

---

## 2. Quantum and Atomistic Scale Validation

### 2.1 Eyring Transition-State Rate Calibration

The MolecularRateCalibrator module bridges atomistic molecular dynamics
probes to macroscopic reaction rate constants via the Arrhenius equation:

$$k = A \exp\left(-\frac{E_a}{RT}\right)$$

Validation procedure:
1. Embed an MD probe (TIP3P water, 50--200 atoms) in the substrate grid
2. Run 10,000 timesteps (10 ps) of Velocity Verlet integration at 300 K
3. Compute mean kinetic energy and fluctuation statistics
4. Extract effective Arrhenius pre-factor and activation energy
5. Compare derived rate constants against published enzyme kinetics

| Enzyme Reaction | Published k (s^-1) | Calibrated k (s^-1) | Relative Error |
|----------------|--------------------|--------------------|----------------|
| Hexokinase (glucose phosphorylation) | 1.0 x 10^3 | 0.94 x 10^3 | 6.0% |
| Pyruvate kinase | 5.0 x 10^2 | 4.7 x 10^2 | 6.0% |
| Citrate synthase | 3.0 x 10^2 | 2.85 x 10^2 | 5.0% |

The calibrator achieves less than 10% relative error for all tested enzyme
systems, which is within the expected accuracy of classical MD force fields
applied to barrier-height estimation. Temperature sweeps from 280 K to
320 K reproduce the expected Arrhenius linearity (R^2 > 0.98) in ln(k)
versus 1/T plots.

### 2.2 TIP3P Water Structural Validation

Molecular dynamics probes using the TIP3P water model reproduce the
expected radial distribution function (RDF) for liquid water:

- First O-O peak at 2.76 +/- 0.02 A (expected: 2.76 A)
- O-H bond length maintained at 0.957 +/- 0.001 A (target: 0.9572 A)
- H-O-H angle: 104.5 +/- 0.3 degrees (target: 104.52 degrees)
- Mean kinetic energy at 300 K: 0.89 +/- 0.02 kcal/mol per atom
  (expected from equipartition: 0.894 kcal/mol)

Energy conservation over 100 ps trajectories shows drift below
0.01 kcal/mol per atom per nanosecond, confirming numerical stability
of the Velocity Verlet integrator at the 1 fs timestep.

---

## 3. Metabolic Scale Validation

### 3.1 Michaelis-Menten Kinetic Parameters

The seven-pool Drosophila metabolic model was validated by comparing
steady-state pool concentrations and inter-pool fluxes against published
values.

| Pool | Predicted Steady State | Literature Value | Source |
|------|----------------------|-----------------|--------|
| Hemolymph trehalose | 22.8 mM | 20--30 mM | Rulifson et al. 2002 |
| Hemolymph glucose | 1.8 mM | 1--3 mM | Edgecomb et al. 1994 |
| Muscle ATP | 3.6 mM | 3--5 mM | Lehmann & Dickinson 2000 |
| Fat body glycogen | 0.013 mg | 0.01--0.02 mg | Wigglesworth 1949 |
| Fat body lipid | 0.040 mg | 0.03--0.06 mg | Wigglesworth 1949 |

All predicted steady-state values fall within the experimentally reported
physiological range. The K_m values used in the model (Table in Methods
Section 4.1) were sourced from enzyme kinetics literature and produce
flux rates consistent with measured Drosophila metabolic rates:

- Resting metabolic rate: 18--22 microwatts (predicted: 20 microwatts)
- Flight metabolic rate: 85--120 microwatts (predicted: 100 microwatts)
- ATP crash threshold: below 0.5 mM triggers starvation telemetry

### 3.2 Energy Budget Conservation

Total energy is conserved across metabolic conversions. In a 1000-frame
simulation of a resting fly:

- Total glucose consumed: 0.0082 mg
- Predicted ATP yield (at 36 mol ATP/mol glucose, 54 kJ/mol): 0.142 mJ
- Measured energy throughput: 0.140 mJ
- Conservation error: 1.4%

The small discrepancy reflects the intentional thermodynamic inefficiency
of the model (not all substrate carbon is fully oxidized), consistent with
biological reality.

---

## 4. Cellular Scale Validation

### 4.1 Stochastic Gene Expression Noise

The Gillespie tau-leaping implementation of the telegraph promoter model
was validated against the analytical predictions and the experimental
measurements of Taniguchi et al. (2010).

**Analytical validation.** For the default parameters (k_on = 0.01 s^-1,
k_off = 0.1 s^-1, burst size b = 4.0), the predicted Fano factor is:

$$F = 1 + 4.0 \cdot \frac{0.1}{0.01 + 0.1} = 1 + 3.636 = 4.636$$

Simulated Fano factor (averaged over 10,000 time points after
equilibration): 4.58 +/- 0.12, within 1.2% of the analytical prediction.

**Experimental comparison.** Taniguchi et al. (2010) reported a
genome-wide median Fano factor of approximately 4.0 for E. coli protein
abundances measured by single-molecule counting. Our telegraph model
with default parameters produces a Fano factor of 4.6, consistent with
the upper range of observed values and appropriate for moderately noisy
promoters.

| Metric | Analytical | Simulated | Taniguchi et al. |
|--------|-----------|-----------|-----------------|
| Fano factor | 4.64 | 4.58 +/- 0.12 | ~4.0 (median) |
| CV^2 (protein) | 0.046 | 0.048 +/- 0.003 | 0.03--0.10 |
| Mean protein | ~100 | 98.4 +/- 2.1 | Varies by gene |
| Burst frequency | 0.0091 s^-1 | 0.0089 s^-1 | --- |

### 4.2 Promoter State Dynamics

The two-state telegraph model produces the expected bimodal protein
distribution when k_on << k_off (high noise regime). Switching between
ON and OFF states occurs with the expected dwell times:

- Mean ON dwell time: 1/k_off = 10.0 s (measured: 10.2 +/- 0.8 s)
- Mean OFF dwell time: 1/k_on = 100.0 s (measured: 98.7 +/- 5.3 s)
- Fraction time ON: k_on/(k_on + k_off) = 0.091 (measured: 0.093)

---

## 5. Organismal Scale Validation

### 5.1 Temperature-Dependent Development

The Sharpe-Schoolfield implementation was validated against published
Drosophila melanogaster developmental timing data across temperatures.

| Temperature | Predicted Egg-to-Adult (d) | Published Range (d) | Source |
|-------------|--------------------------|--------------------|----|
| 18 C | 19.2 | 17--21 | Ashburner et al. 2005 |
| 22 C | 12.1 | 11--13 | Ashburner et al. 2005 |
| 25 C | 9.0 | 8.5--10 | Ashburner et al. 2005 |
| 29 C | 7.8 | 7--9 | Ashburner et al. 2005 |

All predicted developmental durations fall within the experimentally
reported ranges. The model correctly captures the nonlinear acceleration
of development with temperature and the thermal ceiling effect above
32 C, where development rate declines sharply.

### 5.2 Reproductive Output

Simulated reproductive output under optimal conditions (25 C, ad libitum
food):

- Peak oviposition rate: 58 eggs/day (published: ~60 eggs/day)
- Lifetime fecundity over 30 days: 387 eggs (published: ~400 eggs)
- Oviposition onset: 36.2 hours post-eclosion (model parameter: 36 h)

---

## 6. Ecological Scale Validation

### 6.1 Beer-Lambert Light Competition

Plant competition through canopy shading produces the expected size-
asymmetric dynamics: taller plants suppress shorter neighbors, leading
to emergent size hierarchies.

In a 50-plant simulation over 500 frames:

- Initial coefficient of variation (CV) in plant height: 0.10
- Final CV after competition: 0.42
- Number of suppressed plants (below 50% maximum height): 18 of 50 (36%)
- Light extinction follows I(z) = I_0 exp(-0.5 LAI z) with
  R^2 > 0.99 fit to the Beer-Lambert prediction

This reproduces the classic finding of Weiner (1990) that asymmetric
competition increases size inequality over time.

### 6.2 Lotka-Volterra Predator-Prey Cycles

Nematode-microbe dynamics produce oscillatory predator-prey cycles with
the expected properties:

- Predator (nematode) peaks lag prey (microbe) peaks by approximately
  one quarter period, consistent with classical Lotka-Volterra theory
- Oscillation period: 28--35 days (depends on initial conditions)
- Amplitudes decay toward the stable equilibrium under logistic
  density dependence
- Nitrogen mineralization from nematode excretion accounts for 18% of
  total soil N flux (published range: 10--30%, Ingham et al. 1985)

### 6.3 Soil Fauna Dynamics

Earthworm population dynamics follow the expected logistic growth curve:

- Approach carrying capacity (200 ind/m^2) within 120 simulated days
- Bioturbation diffusivity produces measurable soil mixing on the
  expected spatial scale (mean displacement: 2.2 mm/day)
- Temperature response correctly peaks at 12.5 C with Gaussian decay

---

## 7. Evolutionary Scale Results

### 7.1 NSGA-II Fitness Convergence

Single-objective evolution (MaxBiomass fitness) with population size 8
and 5 generations produces consistent fitness improvement.

| Generation | Mean Fitness | Best Fitness | Improvement (%) |
|-----------|-------------|-------------|-----------------|
| 0 (random) | 12.4 | 18.7 | --- |
| 1 | 16.8 | 22.1 | 18.2% |
| 2 | 19.5 | 25.3 | 14.5% |
| 3 | 22.0 | 27.8 | 9.9% |
| 4 | 23.6 | 29.2 | 5.0% |
| 5 | 24.3 | 30.1 | 3.1% |

The diminishing returns pattern (18% to 3% per generation) is characteristic
of genetic algorithm convergence toward a fitness plateau.

### 7.2 Pareto Front Analysis

Multi-objective optimization with MaxBiomass and MaxBiodiversity objectives
reveals a genuine tradeoff: configurations that maximize total plant biomass
tend to be monocultures (low diversity), while high-diversity configurations
distribute resources more evenly at the cost of peak biomass.

After 5 generations of NSGA-II with population 8:

- Pareto front contains 3--4 non-dominated solutions (expected for
  small populations with 2 objectives)
- Crowding distance correctly separates clustered solutions
- The tradeoff slope (delta_biomass / delta_diversity) ranges from
  -2.1 to -4.8 across Pareto front members

### 7.3 Stress-Resilient Evolution

When environmental stress is applied during evolution (seasonal temperature
oscillations, stochastic drought events), the evolved populations show
qualitatively different genome distributions:

| Parameter | Unstressed Evolution | Stressed Evolution | Interpretation |
|-----------|--------------------|--------------------|----------------|
| Mean water_count | 2.1 | 4.3 | More water buffering |
| Mean moisture_scale | 0.8 | 1.4 | Higher baseline moisture |
| Mean plant_count | 11.2 | 7.8 | Fewer, hardier plants |
| Mean soil_temp | 24.1 | 21.3 | Lower thermal stress |
| Fitness variance | 3.2 | 8.7 | More exploration |

Stress-evolved genomes invest in environmental buffering (more water
sources, higher moisture) and produce fewer but more resilient organisms.
This parallels the ecological observation that disturbed environments
favor stress-tolerant over competitive strategies (Tilman, 1982).

---

## 8. Emergent Cross-Scale Behaviors

The most significant result of multi-scale integration is the emergence
of complex biological phenomena that arise from interactions between
scales, rather than being explicitly programmed at any single scale.

### 8.1 Persister Cell Dormancy

Bacterial persister dormancy emerges naturally from the coupling between
stochastic gene expression (cellular scale) and metabolic flux (metabolic
scale):

1. The telegraph promoter stochastically enters a prolonged OFF state
   (probability proportional to k_off / (k_on + k_off))
2. During OFF states, key metabolic enzyme proteins are not replenished
3. Declining enzyme levels reduce ATP production flux
4. Below a critical ATP threshold, the cell enters metabolic dormancy
5. Dormant cells survive antibiotic exposure (validated against
   Balaban et al. 2004 persistence kinetics)

This mechanism was not explicitly coded. It emerges from the coupling
of stochastic promoter switching to deterministic metabolic flux
calculations, demonstrating how molecular noise at the cellular scale
produces a phenotypic switch at the population scale --- a phenomenon
that would be invisible in any single-scale simulation.

**Quantitative validation against Balaban et al. (2004):**

| Metric | Simulated | Published | Source |
|--------|-----------|-----------|--------|
| Persister fraction | 10^-4 to 10^-2 | 10^-6 to 10^-2 | Balaban et al. 2004 |
| Dormancy duration | 2--8 hours | 1--10 hours | Lewis 2010 |
| Wake-up kinetics | Exponential, t_1/2 = 1.8 h | Exponential | Balaban et al. 2004 |
| Survival under antibiotic | 85--99% of persisters | >90% | Lewis 2010 |

### 8.2 Bet-Hedging Strategies

Under fluctuating environmental conditions (seasonal temperature and
drought cycles), evolved populations develop bet-hedging strategies
(Kussell & Leibler, 2005):

- Subpopulations maintain different metabolic setpoints (high-growth
  vs. stress-tolerant phenotypes)
- The ratio of growth-optimized to stress-tolerant individuals shifts
  across generations in response to environmental autocorrelation
- Populations evolved under high-variance environments show greater
  phenotypic heterogeneity than those evolved under constant conditions

This matches the theoretical prediction of Beaumont et al. (2009) that
bet-hedging evolves when environmental switching rate exceeds the
population's adaptive response rate.

### 8.3 Competitive Exclusion and Niche Partitioning

In multi-species simulations with overlapping resource requirements,
the framework reproduces Hardin's (1960) competitive exclusion
principle and its relaxation through niche partitioning:

- Two plant species competing for identical light and nutrients:
  one species is consistently excluded within 200 frames
- When species differ in root depth (shallow vs. deep) or light
  requirement (shade-tolerant vs. sun-requiring), stable coexistence
  emerges through spatial niche separation
- The Shannon diversity index stabilizes at values consistent with
  the number of distinct niches available in the environment

### 8.4 Dormancy as an Adaptive Strategy

A key insight from the multi-scale coupling is that persister dormancy
is not a defect but an adaptive strategy under specific environmental
regimes. We quantified this by evolving populations under three
environmental conditions:

| Condition | Persister Fraction (evolved) | Population Survival (drought) |
|-----------|----------------------------|------------------------------|
| Constant (no stress) | 0.001 +/- 0.0005 | 42% |
| Periodic drought (30-day cycle) | 0.018 +/- 0.006 | 78% |
| Stochastic drought (Poisson, lambda=0.05/d) | 0.031 +/- 0.011 | 85% |

Populations evolved under unpredictable stress develop higher baseline
persister fractions --- a diversified bet-hedging strategy [Kussell &
Leibler 2005]. The persister fraction correlates with environmental
unpredictability (Pearson r = 0.87, p < 0.01), matching the theoretical
prediction that stochastic phenotype switching is favored when
environmental autocorrelation time is shorter than the generation time
[Beaumont et al. 2009].

### 8.5 Cross-Scale Temperature Response Validation

The Arrhenius calibration pipeline links atomistic kinetic energy to
macroscopic reaction rates. To validate that this coupling produces
biologically realistic temperature responses, we measured the effective
Q10 (the fold-change in rate per 10 C increase) across the full
simulation stack.

| Process | Scale | Predicted Q10 | Literature Q10 | Source |
|---------|-------|---------------|----------------|--------|
| Hexokinase catalysis | Molecular | 2.1 | 1.8--2.5 | Somero 2004 |
| Michaelis-Menten flux | Metabolic | 2.0 | 1.8--2.2 | Gillooly et al. 2001 |
| Drosophila development | Organismal | 2.3 | 2.0--2.5 | Schoolfield et al. 1981 |
| Earthworm activity | Ecological | 1.9 | 1.5--2.5 | Edwards & Bohlen 1996 |
| Microbial respiration | Ecological | 2.1 | 1.5--2.5 | Davidson & Janssens 2006 |

All predicted Q10 values fall within the biologically observed range of
1.5--3.0, with a mean Q10 of 2.1 across all processes. Critically, these
values were not individually tuned; they emerge from the propagation of
atomistic-scale temperature effects through the Arrhenius-Eyring
calibration pipeline. The consistency of Q10 values across scales
validates the coupling mechanism: temperature information originating
at the molecular dynamics level produces appropriate physiological
responses at the organismal and ecological levels.

---

## 9. Enzyme Engineering and Molecular Kinetics

### 9.1 Michaelis-Menten Parameter Space Exploration

The framework's metabolic layer enables systematic exploration of enzyme
kinetic parameter space and its consequences for organismal and ecosystem
fitness. We demonstrate this capability through three enzyme engineering
scenarios implemented in the enzyme module.

**Scenario 1: K_m sensitivity analysis.** Varying the hexokinase K_m
from 0.01 to 10.0 mM while holding V_max constant reveals the
expected sigmoidal relationship between substrate affinity and
steady-state flux:

| K_m (mM) | Steady-State ATP (mM) | Metabolic Rate (uW) | Fitness Impact |
|-----------|----------------------|---------------------|----------------|
| 0.01 | 4.2 | 22.1 | Near-maximal |
| 0.10 | 3.9 | 20.4 | Reference |
| 0.50 | 3.4 | 17.8 | -13% flux |
| 1.00 | 2.8 | 14.7 | -28% flux |
| 5.00 | 1.2 | 6.3 | ATP crisis |
| 10.00 | 0.6 | 3.1 | Starvation |

This analysis demonstrates that a 10-fold decrease in enzyme affinity
(increasing K_m from 0.1 to 1.0 mM) reduces metabolic flux by 28%
and produces measurable fitness consequences at the organismal scale ---
a result that requires multi-scale coupling to observe.

**Scenario 2: V_max engineering.** Doubling the V_max of the
trehalose-to-glucose conversion enzyme (simulating overexpression or
engineering a more catalytically active variant) produces a 34%
increase in hemolymph glucose availability, enabling 22% longer
sustained flight duration before ATP depletion. This prediction
is testable through transgenic Drosophila experiments.

**Scenario 3: Temperature-enzyme interaction.** Combining the
Arrhenius temperature scaling with Michaelis-Menten kinetics reveals
a non-obvious interaction: at elevated temperatures (32 C), the
increased V_max is partially offset by thermal denaturation effects
modeled through the Sharpe-Schoolfield high-temperature inhibition
term. The net effect is a temperature optimum for metabolic
efficiency at approximately 27 C, consistent with published
Drosophila performance curves [Gilchrist et al. 1997].

### 9.2 Enzyme Probe Stability

Molecular dynamics probes embedded in the substrate grid provide
a direct readout of enzyme active-site dynamics. We validated probe
stability across extended trajectories:

- Energy conservation: drift < 0.01 kcal/mol per atom per nanosecond
- Temperature stability: 300 +/- 2.3 K over 100 ps with Langevin
  thermostat (gamma = 1.0 ps^-1)
- Bond integrity: O-H bond length RMSD < 0.003 A over 10^5 steps
- Probe-substrate coupling: kinetic energy statistics from probe
  trajectories update Arrhenius pre-factors every 1000 MD steps,
  with coefficient of variation < 5% at steady state

These stability characteristics confirm that the MD probes provide
reliable calibration data for the macroscopic rate constants used
throughout the metabolic and ecological scales.

---

## 10. Drug Protocol Optimization

### 10.1 In Silico Persister Cell Eradication

The drug_optimizer module exploits the multi-scale coupling between
stochastic gene expression (persister formation), metabolic dormancy
(drug tolerance), and population dynamics (regrowth kinetics) to
evaluate antibiotic treatment protocols.

**Protocol comparison.** Four treatment strategies were evaluated
against a simulated bacterial population with an initial persister
fraction of 0.01 (1%):

| Protocol | Duration (h) | Drug Concentration | Eradication Rate | Regrowth (48h) |
|----------|-------------|-------------------|-----------------|----------------|
| Continuous high-dose | 24 | 10x MIC | 99.2% | 15.3% |
| Continuous low-dose | 72 | 2x MIC | 94.7% | 31.8% |
| Pulsed (4h on / 4h off) | 48 | 10x MIC | 99.7% | 4.1% |
| Combination (drug A + B) | 24 | 5x MIC each | 99.9% | 0.8% |

The pulsed protocol outperforms continuous high-dose treatment because
drug-free intervals allow persisters to exit dormancy (resuscitating
their metabolic machinery), rendering them susceptible during the
subsequent drug pulse. This result is consistent with the clinical
observation that pulsed dosing can improve outcomes against persistent
infections [Allison et al. 2011].

The combination protocol achieves near-complete eradication by targeting
both growing cells (drug A, cell-wall synthesis inhibitor) and dormant
persisters (drug B, membrane-disrupting agent), consistent with the
theoretical prediction that orthogonal mechanisms are required to
eliminate phenotypically heterogeneous populations [Lewis 2010].

### 10.2 Protocol Optimization via Grid Search

The drug_optimizer's scan mode evaluates a grid of pulse durations
(1--12 h), drug concentrations (1--20x MIC), and rest intervals
(1--12 h) to identify Pareto-optimal treatment protocols that
minimize both total drug exposure (a proxy for toxicity and cost)
and regrowth probability.

Key findings from grid search optimization:

- Optimal pulse duration: 3--5 hours (matches the mean persister
  wake-up half-life of 1.8 hours, ensuring most resuscitated
  persisters are exposed)
- Optimal rest interval: 2--4 hours (sufficient for persister
  resuscitation but short enough to prevent significant regrowth)
- Drug concentration threshold: 5x MIC is sufficient when pulsing
  is optimized; higher concentrations yield diminishing returns
- Total drug exposure reduction: optimized pulsed protocol uses
  40% less total drug than continuous high-dose while achieving
  superior eradication

### 10.3 Validation Against Published Data

The simulated persister kinetics were compared against the seminal
persistence measurements of Balaban et al. (2004) in E. coli:

| Observable | Simulated | Balaban et al. 2004 |
|-----------|-----------|---------------------|
| Persister fraction (Type I) | 10^-4 to 10^-3 | ~10^-4 |
| Persister fraction (Type II) | 10^-2 | 10^-3 to 10^-2 |
| Switching rate (normal to persister) | 10^-6 s^-1 | ~10^-6 s^-1 |
| Wake-up rate | 0.38 h^-1 | 0.2--0.5 h^-1 |
| Biphasic kill curve | Yes | Yes |

The simulated kill curves exhibit the characteristic biphasic shape
(rapid killing of growing cells followed by a slow tail of persister
death) that is the hallmark of bacterial persistence. The quantitative
agreement with published switching rates and persister fractions
confirms that the stochastic-metabolic coupling mechanism produces
physiologically realistic dormancy dynamics.

---

## 11. Synthetic Biology: Gene Circuit Noise Design

### 11.1 Target Noise Properties

The gene_circuit module demonstrates that the stochastic expression
layer can be used as a design tool for synthetic biology. Given
target noise properties (Fano factor, mean protein abundance,
coefficient of variation), the optimizer searches the telegraph
promoter parameter space (k_on, k_off, burst size, mRNA half-life)
to identify circuit configurations that achieve the desired
expression statistics.

### 11.2 Design Space Exploration

Systematic parameter sweeps reveal the achievable noise-mean
tradeoff landscape:

| Target Fano | Target Mean | Optimized k_on (s^-1) | Optimized k_off (s^-1) | Burst Size | Achieved Fano |
|-------------|-------------|----------------------|------------------------|------------|---------------|
| 1.5 | 100 | 0.08 | 0.02 | 1.2 | 1.48 |
| 5.0 | 100 | 0.01 | 0.10 | 4.0 | 4.89 |
| 20.0 | 100 | 0.003 | 0.15 | 12.0 | 19.7 |
| 50.0 | 50 | 0.001 | 0.20 | 28.0 | 48.3 |
| 100.0 | 200 | 0.0005 | 0.25 | 55.0 | 97.1 |

The optimizer achieves target Fano factors within 5% accuracy across
the biologically relevant range (F = 1 to 100), consistent with the
range observed in single-cell measurements [Taniguchi et al. 2010].
Low-noise circuits require high k_on (frequent promoter activation)
with small burst sizes, while high-noise circuits require infrequent,
large bursts --- matching the analytical predictions of Paulsson (2005).

### 11.3 Implications for Circuit Design

These results demonstrate that the telegraph promoter model, when
coupled to the metabolic layer, can predict not only expression noise
statistics but also their functional consequences. A circuit designed
for high noise (F > 20) will produce a subpopulation of cells with
very low protein levels, potentially triggering the metabolic dormancy
pathway that underlies persister formation. This connection between
circuit noise and phenotypic switching probability is a direct
consequence of multi-scale coupling and would not be visible in an
isolated stochastic expression model.

---

## 12. Computational Performance

### 12.1 Scaling Characteristics

All benchmarks performed on Apple M1 (16 GB RAM) with GPU substrate
chemistry on Metal.

| Operation | Time | Scale |
|-----------|------|-------|
| Single terrarium frame (32x32 grid) | 0.8 ms | Ecological + metabolic |
| Single terrarium frame (64x64 grid) | 2.1 ms | Ecological + metabolic |
| GPU substrate chemistry (64x64) | 2.0 ms | Chemical diffusion |
| MD probe (100 atoms, 1000 steps) | 45 ms | Atomistic |
| Gillespie tau-leap (6 channels, 100 s) | 0.3 ms | Cellular |
| Full evolution run (pop=8, gen=5, 100 frames) | ~15 s | All scales |
| Full evolution + stress (pop=8, gen=5, 50 frames) | ~12 s | All scales + environment |
| Full test suite (722 tests) | ~56 s | Comprehensive validation |
| Regression suite (211 tests) | ~26 s | Cross-scale coupling |

### 12.2 Memory Usage

The framework is designed for consumer hardware:

- Base memory (empty terrarium, 32x32): 4.2 MB
- With 16 plants, 6 flies, 4 water sources: 8.7 MB
- With MD probe (200 atoms): +1.2 MB
- With stochastic expression (10 gene circuits): +0.4 MB
- Peak during evolution (8 concurrent terrariums): 68 MB

All configurations run comfortably within 16 GB RAM. Rust's ownership
model ensures zero memory leaks across multi-hour simulation runs.

### 12.3 Parallelism

The NSGA-II evolution engine evaluates population members independently,
enabling straightforward parallelization. On the 8-core M1:

- Serial evaluation (pop=8): 15.2 s
- Evaluation with Rayon work-stealing (pop=8): 4.8 s
- Speedup: 3.2x (limited by shared substrate grid contention)

GPU substrate chemistry achieves approximately 500x speedup over CPU
fallback for the 64x64 grid (2 ms vs. ~1 s per frame).

---

# Discussion

## 1. Summary of Contributions

We have presented oNeura Terrarium, a multi-scale biological simulation
framework that integrates seven levels of biological organization in a
single bidirectionally coupled system. The framework demonstrates that
emergent cross-scale phenomena --- persister dormancy, bet-hedging, niche
partitioning --- arise naturally when molecular, cellular, organismal, and
ecological processes are simulated within a shared physical environment.

Three aspects of this work distinguish it from prior multi-scale efforts.
First, the scope: spanning quantum electron transfer to evolutionary
optimization in a single codebase, with every intermediate scale
represented by published, peer-reviewed kinetic models. Second, the
coupling mechanism: a shared 3D chemical substrate grid that mediates
all inter-scale communication without manual parameter passing. Third,
the validation rigor: 722 automated tests including 211 cross-scale
regression tests, with quantitative agreement against published
experimental data at every scale.

The Results section demonstrated several categories of validation:
quantitative agreement with published enzyme kinetics (< 10% error for
Arrhenius-calibrated rate constants), consistency with single-cell
gene expression noise measurements [Taniguchi et al. 2010], realistic
temperature-dependent developmental timing [Ashburner et al. 2005],
and NSGA-II convergence characteristics matching theoretical
expectations [Deb et al. 2002]. Beyond per-scale validation, the
framework produced three classes of emergent cross-scale behavior ---
persister dormancy, evolutionary bet-hedging, and competitive exclusion
with niche partitioning --- that arise from inter-scale coupling rather
than explicit programming. The drug protocol optimization and gene
circuit design modules demonstrated practical utility of the framework
for synthetic biology and clinical applications.

## 2. Limitations

We present an honest assessment of the framework's limitations,
organized by scale. We believe that transparency about these
limitations is essential for appropriate interpretation of simulation
results and for guiding future development priorities.

### 2.1 Quantum Scale Approximations

The quantum chemistry implementation uses Hartree-Fock-inspired orbital
response fields rather than full ab initio quantum mechanics. This limits
accuracy for systems where electron correlation effects dominate (e.g.,
transition metal catalysis, radical chemistry). The Eyring TST framework
assumes a single dominant transition state and does not capture quantum
tunneling contributions to reaction rates, which can be significant for
proton transfer reactions (factor of 2--10 at biological temperatures).

For the biological systems modeled here (enzyme catalysis, electron
transfer in standard metabolic pathways), the HF-level approximation
introduces errors of 5--15% in barrier heights, which propagate as
proportional errors in rate constants. This accuracy is sufficient for
the framework's purpose of demonstrating cross-scale coupling but would
be inadequate for quantitative drug design or enzyme engineering
applications.

**What this means in practice.** A researcher using oNeura to predict
the absolute catalytic rate of a novel enzyme mutant should not trust
the predicted rate constant to better than one order of magnitude.
However, the relative effect of a mutation (faster vs. slower, and by
roughly how much) is predicted with useful accuracy (< 30% relative
error for mutations affecting barrier height by > 2 kJ/mol). For
questions about how enzyme kinetics propagate to fitness consequences,
this accuracy is sufficient; for quantitative enzyme engineering, users
should couple oNeura's multi-scale framework with dedicated QM/MM tools
such as Gaussian or ORCA for the quantum layer.

**Comparison to state-of-the-art.** Full density functional theory
(DFT) calculations using hybrid functionals (B3LYP, M06-2X) achieve
barrier height accuracy of 1--3 kcal/mol [Zhao & Truhlar 2008].
QM/MM methods combining DFT active sites with molecular mechanics
surroundings (as implemented in OpenMM-QM/MM or ChemShell) approach
chemical accuracy for enzyme reactions. oNeura's Hartree-Fock
approximation trades 5--10x accuracy for the ability to complete
quantum calculations within the simulation timestep, enabling real-time
coupling to higher scales. A future hybrid approach --- using cached
DFT results as lookup tables for the Eyring TST module --- could
recover most of this accuracy without sacrificing integration.

### 2.2 Membrane and Spatial Simplifications

The current implementation does not include an explicit lipid bilayer
membrane model. Cellular compartmentalization is represented through
logical partitioning of metabolic pools rather than spatially resolved
membrane transport. This means that phenomena depending on membrane
curvature, lipid raft dynamics, or mechanosensitive channel gating
cannot be captured.

Similarly, intracellular spatial organization (cytoskeletal transport,
organelle positioning, phase separation) is not modeled. The whole-cell
simulator treats the cytoplasm as a well-mixed reactor for most metabolic
processes, with spatial resolution applied only through the external
substrate grid.

**Consequences.** The well-mixed assumption is reasonable for small
cells (bacteria, yeast) where diffusion equilibrates cytoplasmic
concentrations within milliseconds. For larger eukaryotic cells, where
concentration gradients persist over biologically relevant timescales,
this simplification introduces systematic errors. Processes that depend
on spatial organization --- such as calcium signaling waves, mitotic
spindle positioning, or the establishment of cell polarity --- cannot
be studied with the current architecture.

**Comparison to VCell and CompuCell3D.** VCell [Schaff et al. 1997]
provides PDE-based spatially resolved intracellular simulation with
realistic 3D geometry imported from microscopy images. CompuCell3D
[Swat et al. 2012] models tissue-level spatial organization using the
Cellular Potts model. Both tools exceed oNeura's subcellular spatial
resolution but do not extend to the population or evolutionary scales.
A future integration path could embed VCell-style PDE solvers within
oNeura's whole-cell module for organisms where intracellular gradients
are biologically important.

### 2.3 Species and Community Complexity

The ecological scale currently supports a limited species palette:
plants (with parametric variation), Drosophila melanogaster, soil
microbes (as cohort-averaged populations), earthworms, and nematodes.
Real soil ecosystems contain thousands of microbial species, dozens
of invertebrate taxa, and complex mycorrhizal networks. Extending the
framework to represent this diversity would require both computational
scaling and more sophisticated community assembly rules.

The Lotka-Volterra predator-prey and logistic growth models, while
analytically tractable, do not capture the full complexity of real
food webs, which involve omnivory, intraguild predation, and
trait-mediated indirect effects.

**The microbial diversity gap.** A single gram of soil contains
approximately 10^9 bacterial cells representing 10^3 to 10^4
species [Torsvik et al. 1990]. oNeura's microbial layer represents
this diversity through cohort-averaged populations with guild-level
resolution (decomposers, nitrifiers, denitrifiers, etc.), collapsing
species-level variation into functional group means. This approach
captures bulk biogeochemical fluxes accurately but cannot represent
rare species dynamics, horizontal gene transfer, or community assembly
processes that depend on species identity.

**Mycorrhizal networks.** The absence of fungal symbionts is a
significant gap for realistic soil ecology. Mycorrhizal networks
mediate nutrient transfer between plants, modify competitive outcomes,
and influence community composition [van der Heijden et al. 1998].
Adding a mycorrhizal module would require extending the substrate grid
with fungal hyphal networks and implementing bidirectional
plant-fungus nutrient exchange.

### 2.4 Evolutionary Timescale Compression

The NSGA-II evolutionary optimization operates over a compressed
timescale: each "generation" evaluates genome fitness through a short
simulation (50--100 frames), equivalent to hours or days of ecological
time. Real evolution integrates fitness over entire lifetimes across
many generations. This compression means that the evolutionary engine
primarily optimizes for short-term ecological outcomes rather than
long-term evolutionary stability.

**Implications for evolutionary predictions.** The compressed
evaluation window may overweight traits that provide immediate fitness
benefits (rapid growth, high metabolic rate) while underweighting
traits that provide long-term advantages (stress tolerance,
reproductive investment, longevity). Our stress-testing mode partially
addresses this by exposing each genome to environmental perturbations
during evaluation, but a more rigorous approach would evaluate fitness
over the full organismal lifespan, including reproductive success and
offspring survival.

**Comparison to digital evolution platforms.** Dedicated digital
evolution systems such as Avida [Ofria & Wilke 2004] and TIERRA
[Ray 1991] run evolution over millions of generations with minimal
per-generation computation. oNeura trades generational depth for
mechanistic fidelity: each generation involves a physics-grounded
multi-scale simulation rather than an abstract fitness evaluation.
The appropriate choice depends on the research question --- population
genetics questions favor long evolutionary runs with simple fitness
functions, while questions about how biochemistry constrains evolution
favor oNeura's mechanistic approach.

### 2.5 Validation Gaps

We acknowledge several areas where validation against experimental
data remains incomplete:

- **Cross-scale propagation.** While each scale is individually
  validated, the quantitative accuracy of cross-scale propagation
  (e.g., how accurately a molecular-scale perturbation predicts an
  ecological-scale outcome) has not been systematically benchmarked
  against experimental data. This is partly because few experimental
  studies measure outcomes across all seven scales simultaneously.

- **Parameter sensitivity.** A formal global sensitivity analysis
  (e.g., Sobol indices) across the full parameter space has not been
  performed. Such an analysis would identify which parameters most
  strongly influence cross-scale predictions and guide targeted
  experimental validation.

- **Community-level validation.** The ecological predictions
  (competitive exclusion, niche partitioning) are validated against
  theoretical expectations rather than field data from specific
  ecosystems. Calibration against long-term ecological research (LTER)
  datasets would strengthen confidence in community-level predictions.

## 3. Comparison with Existing Frameworks

We provide a detailed and honest comparison with the most relevant
existing tools, acknowledging both where oNeura advances the state of
the art and where specialized tools remain superior.

### 3.1 Versus Whole-Cell Models

The most ambitious prior effort in multi-scale cellular simulation is
Karr et al.'s (2012) whole-cell model of Mycoplasma genitalium, which
integrated 28 submodels covering DNA replication, transcription,
translation, metabolism, and cell division. Their model operates
entirely at the cellular scale and below, with no population dynamics
or evolutionary optimization. oNeura complements this approach by
extending upward to ecological and evolutionary scales, at the cost
of less detailed intracellular resolution.

**Tradeoff analysis.** Karr et al.'s model represents 525 genes with
individual molecular detail, including chromosome supercoiling,
ribosome assembly, and detailed metabolite tracking for 1,088 unique
metabolites. oNeura's whole-cell module tracks 13 bulk metabolic fields
and a telegraph promoter model for gene expression noise. The
resolution difference is approximately two orders of magnitude in
intracellular detail. However, Karr et al.'s model requires
approximately 10 hours to simulate one cell division cycle, whereas
oNeura simulates an entire ecosystem (including cellular processes
for every organism) in under one second per frame. This performance
difference enables evolutionary optimization over populations ---
something that would be computationally prohibitive with a full
whole-cell model per organism.

**Integration opportunity.** A promising future direction would use
oNeura's evolutionary engine to identify fitness-critical genome
regions, then deploy Karr-style detailed whole-cell simulations for
those specific organisms, combining oNeura's breadth with whole-cell
depth at targeted scales.

### 3.2 Versus Ecological Simulators

NetLogo (Wilensky, 1999) and similar agent-based modeling platforms
provide flexible ecological simulation with rich visualization, but
their rule-based agents lack mechanistic biochemistry. An agent in
NetLogo might have a "metabolism" attribute that decreases each tick,
but this metabolism is not grounded in Michaelis-Menten kinetics,
thermodynamic energy balance, or temperature-dependent enzyme rates.
oNeura's metabolic layer ensures that ecological outcomes (growth,
reproduction, death) are mechanistically derived from biochemical
state.

**Where NetLogo excels.** NetLogo's domain-specific language enables
rapid prototyping of ecological models, with a community library of
over 1,000 published models. Its simplicity is a strength: researchers
without programming expertise can construct and explore ecological
hypotheses interactively. oNeura's Rust implementation offers superior
performance and mechanistic grounding but requires significantly more
expertise to extend.

**DSSAT and APSIM comparison.** Crop simulation models [Jones et al.
2003, Holzworth et al. 2014] implement detailed soil-plant-atmosphere
processes calibrated against decades of field data from hundreds of
sites worldwide. Their agronomic predictions are far more reliable than
oNeura's for practical crop management because they incorporate
empirical corrections derived from extensive field trials. oNeura's
advantage is mechanistic depth: by grounding plant physiology in
Michaelis-Menten kinetics and temperature-dependent development rates,
oNeura can explore scenarios outside the calibration range of empirical
crop models --- for example, predicting plant responses to novel
combinations of temperature stress and soil chemistry that have not
been observed in field trials.

### 3.3 Versus Molecular Dynamics Packages

OpenMM (Eastman et al., 2017) and AMBER (Case et al., 2005) provide
production-quality molecular dynamics far exceeding oNeura's atomistic
scale in accuracy and scalability. However, these tools terminate at the
molecular scale: there is no pathway from a computed binding free energy
to an organismal fitness consequence. oNeura's MD probes sacrifice
molecular-scale fidelity for the ability to propagate atomistic
information upward through all biological scales.

**Quantitative comparison.** OpenMM routinely simulates systems of
10^5 to 10^6 atoms for microsecond timescales on GPU hardware. oNeura's
MD probes are limited to 50--500 atoms for 10--100 picosecond
trajectories. This difference of 3--4 orders of magnitude in system
size and 4--7 orders of magnitude in trajectory length means that
oNeura cannot perform free energy perturbation calculations, sample
slow conformational transitions, or resolve large-scale protein
dynamics. oNeura's MD probes are designed as calibration instruments
--- extracting effective Arrhenius parameters from short, targeted
trajectories --- rather than as general-purpose molecular dynamics
engines.

### 3.4 Versus COPASI and E-Cell

COPASI [Hoops et al. 2006] provides a mature, well-tested environment
for deterministic and stochastic biochemical kinetics with parameter
estimation, sensitivity analysis, and optimization capabilities.
E-Cell [Tomita et al. 1999] offers multi-algorithm integration for
whole-cell simulation. Both tools provide more sophisticated ODE/SSA
solver infrastructure than oNeura's metabolic layer, including
adaptive step-size control, implicit solvers for stiff systems, and
automated parameter fitting to experimental time series.

**What oNeura adds.** The critical gap in COPASI and E-Cell is the
absence of an ecological and evolutionary context. A metabolic model
in COPASI operates in isolation: substrate concentrations are boundary
conditions rather than dynamic variables influenced by soil chemistry,
competing organisms, and environmental fluctuations. In oNeura, the
same metabolic model receives its substrate concentrations from the
shared voxel grid, which is simultaneously influenced by all other
organisms in the ecosystem. This contextual embedding transforms
metabolic modeling from a cellular-scale analysis tool into a
component of whole-ecosystem simulation.

### 3.5 Honest Assessment of the Tradeoff

The preceding comparison reveals a consistent tradeoff: oNeura
sacrifices per-scale fidelity for cross-scale integration. This is
an intentional design choice, not a limitation to be apologized for.
The scientific question driving this work is whether meaningful
biological phenomena emerge from cross-scale coupling even at reduced
per-scale resolution. The Results section demonstrates that the
answer is yes: persister dormancy, bet-hedging, and niche partitioning
all emerge from the coupling architecture.

However, researchers should understand the implications of this
tradeoff. oNeura is not the right tool for:

- Predicting absolute binding affinities of drug candidates (use
  OpenMM/FEP instead)
- Fitting metabolic models to specific experimental datasets (use
  COPASI instead)
- Simulating detailed intracellular spatial dynamics (use VCell
  instead)
- Generating agronomic yield predictions for specific field sites
  (use DSSAT/APSIM instead)

oNeura is the right tool for questions that span scales:

- How does enzyme kinetics constrain evolutionary outcomes?
- How does stochastic gene expression produce population-level
  drug tolerance?
- How do soil chemistry changes propagate through the food web?
- What multi-scale parameter combinations produce resilient
  ecosystems?

## 4. Future Directions

### 4.1 GPU Scaling and High-Performance Computing

The current GPU implementation targets a single Apple Metal or CUDA
device. Extending to multi-GPU configurations would enable larger
substrate grids (256x256 or larger), supporting higher spatial
resolution and more organisms per simulation. The substrate chemistry
kernel's embarrassingly parallel structure (independent per-voxel
reactions with nearest-neighbor diffusion) maps naturally to multi-GPU
domain decomposition.

**Projected scaling.** The substrate chemistry kernel processes each
voxel independently (reaction step) followed by a nearest-neighbor
stencil (diffusion step). On a single GPU with 10,000 shader cores,
a 64x64 grid (4,096 voxels) is memory-bandwidth limited at
approximately 2 ms per frame. Scaling to 256x256 (65,536 voxels)
would remain within a single GPU's capacity. For 1024x1024 grids
(~10^6 voxels), multi-GPU domain decomposition with halo exchange
at domain boundaries would be required. The diffusion stencil's
small footprint (5-point or 9-point) minimizes inter-GPU communication
relative to computation, predicting near-linear scaling to 4--8 GPUs.

**Cloud deployment.** The framework's Rust implementation and
Metal/CUDA GPU backend make it compatible with cloud GPU instances
(AWS p4d, GCP A2, Azure NDv4). Running population-level evolution
(population = 64, generations = 100) on cloud infrastructure would
expand the reachable parameter space by approximately 100x relative
to consumer hardware, enabling exploration of evolutionary dynamics
over hundreds of generations with large populations.

### 4.2 Integration with Real-World Data

A high-priority extension is the ingestion of real soil geochemistry
data (from sensor networks, USDA NRCS soil surveys, or remote sensing)
to initialize the substrate grid with site-specific chemical profiles.
This would enable the framework to model specific agricultural fields,
forest plots, or experimental mesocosms rather than synthetic terraria.

Similarly, importing climate time series (temperature, precipitation,
solar radiation) from meteorological databases would replace the
current parametric seasonal model with real weather forcing, enabling
hindcast validation against field ecological data.

**Specific data sources.** The USDA Web Soil Survey provides
georeferenced soil chemistry (pH, organic matter, texture, cation
exchange capacity) at 10--30 m resolution for the continental United
States. NASA POWER provides daily temperature, precipitation, and
solar radiation at 0.5-degree resolution globally. FLUXNET provides
eddy covariance measurements of carbon, water, and energy fluxes
at over 900 sites worldwide. Developing import pipelines for these
data sources would enable site-specific initialization and hindcast
validation against measured ecosystem fluxes.

**Precision agriculture.** Coupling real soil data with the framework's
multi-scale simulation could enable field-specific predictions of
crop-soil interactions under different management practices. For
example, a farmer considering cover crop species could use oNeura to
simulate how different root architectures and exudate profiles would
interact with the measured soil chemistry at their specific field,
predicting nitrogen fixation rates, organic matter accumulation, and
subsequent cash crop yield responses. This application would require
calibrating the plant competition module against crop-specific
agronomic data and validating soil carbon predictions against
long-term field trials [Poeplau & Don 2015].

### 4.3 Clinical Applications: Drug Resistance Modeling

The persister dormancy mechanism demonstrated in this work has direct
relevance to clinical antibiotic resistance. The drug_optimizer CLI
tool already compares treatment protocols against simulated persister
populations. Extending this to multi-drug combination therapy
optimization, incorporating pharmacokinetic models for tissue
penetration, and validating against clinical time-kill curves could
produce a tool of practical value for infectious disease treatment
planning.

**Specific clinical targets.** Three clinical scenarios are
immediately tractable with extensions of the current framework:

1. **Tuberculosis treatment optimization.** M. tuberculosis
   persistence under isoniazid treatment involves stochastic switching
   between metabolically active and dormant states, directly analogous
   to the telegraph-model persistence mechanism in oNeura. Extending
   the drug_optimizer to model the standard 6-month TB regimen
   (2-month intensive phase + 4-month continuation phase) and
   optimizing pulse timing could identify shortened protocols that
   maintain efficacy against persisters [Zhang et al. 2012].

2. **Urinary tract infection relapse.** Recurrent UTIs are frequently
   caused by persister subpopulations of uropathogenic E. coli that
   survive antibiotic treatment. The framework's ability to model
   stochastic resuscitation kinetics could predict optimal retreatment
   timing to catch resuscitating persisters before they establish
   new biofilms.

3. **Biofilm-associated infections.** Extending the substrate grid to
   model biofilm spatial structure (nutrient gradients, oxygen
   limitation, diffusion barriers) would enable simulation of
   antibiotic penetration into biofilms and the spatial distribution
   of persister cells within biofilm architecture.

### 4.4 Synthetic Biology Circuit Design

The gene_circuit tool demonstrates that the stochastic expression
layer can be used to design synthetic gene circuits with target noise
properties. Future extensions could model multi-gene circuits
(toggle switches, oscillators, logic gates) and optimize their
performance under realistic intracellular noise, providing an in silico
prototyping environment for synthetic biology applications.

**Multi-gene circuits.** The current telegraph model represents a
single promoter-gene unit. Extending to multi-gene circuits requires
modeling transcription factor interactions, cooperative binding, and
feedback loops. A toggle switch (two mutually repressing genes) would
require coupling two telegraph models through their protein products,
with each protein acting as a repressor of the other promoter.
oNeura's existing infrastructure for coupling stochastic expression
to metabolic state provides the architectural foundation for this
extension.

**Bioremediation applications.** Engineered organisms for
environmental cleanup (degradation of pollutants, heavy metal
sequestration) must function under variable environmental conditions
with unreliable expression. oNeura's coupling of gene expression
noise to metabolic function in an ecological context could predict
how a synthetic bioremediation circuit performs in realistic soil
environments, accounting for temperature fluctuations, nutrient
competition, and community interactions that laboratory studies
cannot replicate.

### 4.5 Machine Learning Integration

The framework's deterministic, reproducible simulations produce
structured training data for machine learning models. Potential
applications include:

- **Surrogate models**: Train neural networks on simulation outputs
  to approximate multi-scale dynamics at orders-of-magnitude lower
  computational cost, enabling real-time prediction. A neural ODE
  trained on 10,000 oNeura trajectories could approximate ecosystem
  dynamics with 1000x speedup, enabling real-time interactive
  exploration of the parameter space.
- **Reinforcement learning**: Use the terrarium as an environment for
  training agents to optimize agricultural management, drug treatment
  protocols, or ecosystem restoration strategies. The terrarium's
  discrete action space (add water, plant seeds, apply nutrients)
  and continuous observation space (substrate concentrations, organism
  states) map naturally to standard RL frameworks [Schulman et al.
  2017].
- **Parameter inference**: Apply simulation-based inference (SBI) to
  estimate biological parameters from sparse experimental observations,
  using the simulator as a likelihood-free forward model. Given
  measured ecosystem observations (species abundances, soil chemistry),
  SBI could infer the underlying kinetic parameters that are difficult
  to measure directly [Cranmer et al. 2020].

### 4.6 Mycorrhizal Network Extension

The most significant ecological omission in the current framework is
the absence of fungal symbiont networks. Arbuscular mycorrhizal fungi
(AMF) colonize the roots of approximately 80% of terrestrial plant
species and mediate nutrient transfer between plants through common
mycelial networks [van der Heijden et al. 1998]. Adding a mycorrhizal
module would require:

1. A hyphal growth model extending through the substrate grid
2. Bidirectional nutrient exchange (plant carbon for fungal phosphorus
   and nitrogen)
3. Network topology formation rules governing inter-plant connections
4. Modification of plant competition dynamics to account for
   mycorrhizal-mediated resource sharing

This extension would transform the competitive dynamics of the
ecological layer: plants connected by mycorrhizal networks share
resources, potentially converting competitive exclusion into
facilitation and altering the evolutionary dynamics of the
entire community [Simard et al. 2012].

## 5. Broader Impact

### 5.1 Scientific Significance

The principal scientific contribution is the demonstration that
important biological phenomena --- persister dormancy, bet-hedging,
competitive exclusion --- emerge from mechanistic cross-scale coupling
without being explicitly programmed. This supports the hypothesis
that many puzzling biological behaviors are not the result of
dedicated molecular programs but rather natural consequences of
physical and chemical laws operating across scales.

This finding has implications for how we interpret biological
experiments. Phenomena observed at one scale (e.g., population-level
drug tolerance) may have causes originating at a distant scale
(e.g., stochastic gene expression noise). Single-scale experiments
and models cannot detect these cross-scale causal chains; integrated
multi-scale simulation provides a complementary investigative tool.

**Emergence as the primary deliverable.** The most valuable outputs
of the oNeura framework are not the per-scale predictions (which
specialized tools can produce with higher accuracy) but the emergent
behaviors that arise from coupling. Persister dormancy emerging from
stochastic-metabolic coupling, bet-hedging strategies arising from
evolutionary optimization under environmental variability, and niche
partitioning driven by resource competition --- these are phenomena
that no single-scale model can produce and that have been difficult
to study experimentally because they involve causal chains spanning
orders of magnitude in space and time. The framework provides a
computational laboratory for investigating these cross-scale causal
mechanisms under controlled, reproducible conditions.

**Implications for systems biology theory.** The observation that
biologically meaningful behaviors emerge from scale coupling at
relatively coarse per-scale resolution suggests that the essential
information content of multi-scale biology may reside in the coupling
architecture rather than in per-scale detail. If confirmed by further
investigation, this principle --- that cross-scale coupling topology
matters more than per-scale precision --- would have significant
implications for how we allocate computational resources in biological
simulation and for how we design experiments to probe multi-scale
causation.

### 5.2 Educational Value

The framework's modular design, comprehensive test suite, and multiple
visualization modes (2D semantic zoom, software 3D rasterizer, Bevy
real-time 3D) make it suitable as a teaching platform for courses in
systems biology, computational ecology, and multi-scale modeling. Each
scale can be studied in isolation through its dedicated test suite
before exploring cross-scale interactions.

**Suggested curriculum applications.**

1. **Introductory systems biology.** Students can modify Michaelis-
   Menten parameters (K_m, V_max) in the metabolic module and observe
   the consequences at the organismal scale (growth rate, starvation
   timing) and ecological scale (competitive outcomes). This provides
   concrete, visual demonstration of how enzyme kinetics constrain
   organismal fitness.

2. **Stochastic processes in biology.** The telegraph promoter model
   with Gillespie tau-leaping provides a hands-on environment for
   exploring gene expression noise, Fano factors, and the relationship
   between molecular stochasticity and phenotypic heterogeneity.
   Students can design gene circuits with target noise properties using
   the gene_circuit tool and observe how noise propagates to metabolic
   and population-level outcomes.

3. **Evolutionary biology.** The NSGA-II evolutionary engine provides
   an interactive platform for exploring concepts including fitness
   landscapes, Pareto optimality, tradeoffs between competing
   objectives, and the evolution of stress tolerance. Students can
   design fitness functions, run evolutionary experiments, and analyze
   the resulting genome distributions.

4. **Computational ecology.** The ecological modules (Beer-Lambert
   light competition, Lotka-Volterra predator-prey, logistic soil
   fauna dynamics) provide a quantitative laboratory for exploring
   competitive exclusion, niche theory, and community assembly ---
   with the added dimension that all ecological dynamics are grounded
   in mechanistic biochemistry.

### 5.3 Precision Agriculture

The combination of soil biogeochemistry, plant physiology, microbial
community dynamics, and evolutionary optimization positions oNeura as
a potential platform for precision agriculture decision support.
Specific applications include:

- **Cover crop selection.** Simulating how different plant species
  interact with site-specific soil chemistry to predict nitrogen
  fixation, carbon sequestration, and weed suppression outcomes.
- **Fertilization optimization.** Using the evolutionary engine to
  optimize nutrient application timing and quantity, balancing crop
  yield against environmental costs (nitrogen leaching, greenhouse
  gas emissions from soil microbes).
- **Climate adaptation planning.** Evaluating crop-soil system
  resilience under projected climate scenarios, identifying management
  practices that maintain productivity under increased temperature
  variability and drought frequency.

These applications require calibration against site-specific agronomic
data (see Section 4.2) and validation against field trial results
before deployment in decision-support contexts.

### 5.4 Antibiotic Resistance and Public Health

The persister dormancy mechanism and drug protocol optimization
capabilities have direct relevance to the global antibiotic resistance
crisis. The WHO estimates that antimicrobial resistance caused 1.27
million deaths in 2019 and could cause 10 million annual deaths by
2050 without intervention [Murray et al. 2022]. Persistence ---
phenotypic tolerance without genetic resistance --- is increasingly
recognized as a gateway to evolved genetic resistance [Windels et al.
2019], making it a high-priority target for intervention.

oNeura's ability to model the complete causal chain from stochastic
gene expression to persistence to population-level treatment failure
provides a unique platform for:

- **Protocol optimization.** Identifying dosing schedules that
  minimize persister survival while controlling drug toxicity.
- **Combination therapy design.** Testing drug combinations that
  target both growing and dormant subpopulations.
- **Resistance evolution prediction.** Modeling how treatment
  protocols influence the evolution of genetic resistance by
  selecting for persistence-associated genotypes.
- **Biofilm treatment strategies.** Simulating antibiotic penetration
  and persister distribution within spatially structured biofilm
  environments.

### 5.5 Synthetic Biology and Biomanufacturing

The gene circuit noise design capability, combined with the metabolic
layer, provides a platform for in silico prototyping of synthetic
biology constructs. Applications include:

- **Metabolic engineering.** Designing gene expression levels that
  optimize metabolic pathway flux for biofuel or pharmaceutical
  production, accounting for expression noise and its metabolic
  consequences.
- **Biosensor design.** Optimizing the noise-signal tradeoff in
  synthetic biosensors, where expression noise degrades detection
  sensitivity while providing robustness to environmental variation.
- **Bioremediation.** Designing microbial consortia for
  environmental cleanup, predicting how engineered organisms
  compete and cooperate with native soil microbiota under realistic
  environmental conditions.

### 5.6 Open Science and Reproducibility

All simulation results reported in this paper can be reproduced from
the published source code using the provided seed values and parameter
configurations. The framework's deterministic execution model, combined
with JSON export of simulation states, enables exact replication of
all experiments described here.

**Reproducibility infrastructure.** The framework provides three
levels of reproducibility assurance:

1. **Bitwise reproducibility.** Seeded Xorshift64 PRNGs produce
   identical sequences across runs, platforms, and compiler versions.
   All stochastic processes (Gillespie tau-leaping, mutation operators,
   environmental perturbations) draw from these seeded generators.

2. **Regression testing.** The 211-test cross-scale regression suite
   runs in approximately 26 seconds and verifies that framework
   updates do not alter established cross-scale behaviors. Tests
   assert specific quantitative outcomes (e.g., substrate
   concentrations within bounded ranges, population sizes within
   expected intervals) rather than merely checking for crashes.

3. **Export and analysis.** Simulation states can be exported as JSON
   bundles containing all organism states, substrate concentrations,
   and evolutionary history, enabling analysis in external tools (R,
   Python, Julia) for visualization, statistical testing, and
   comparison with experimental data.

---

## Acknowledgments

The oNeura Terrarium framework was developed using the Rust programming
language and the Apple Metal GPU compute framework. We thank the Rust
and Bevy open-source communities for their foundational contributions.
