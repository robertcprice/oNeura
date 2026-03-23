#![allow(dead_code)]
//! Biofilm formation, quorum sensing, and multi-species biofilm dynamics.
//!
//! This is a fully self-contained module (no `crate::` imports) that simulates
//! bacterial biofilm lifecycle including:
//!
//! - **Monod growth kinetics**: nutrient-limited bacterial growth with species-specific
//!   half-saturation constants (K_s) and maximum growth rates (mu_max).
//! - **EPS matrix production**: extracellular polymeric substance secretion forming the
//!   structural scaffold of the biofilm. Production is upregulated by quorum sensing.
//! - **Quorum sensing**: diffusible signal molecules (modeled on N-acyl homoserine
//!   lactones / AHL) that coordinate population-level gene expression when
//!   concentrations exceed a threshold (typically 1-10 nM for LuxR-type systems).
//! - **Antibiotic tolerance**: biofilm-embedded cells exhibit 10-1000x higher tolerance
//!   than planktonic cells due to EPS diffusion barrier and metabolic dormancy.
//! - **Detachment**: shear-mediated erosion and sloughing inversely proportional to
//!   local EPS density.
//! - **Mushroom structure detection**: mature biofilms form characteristic mushroom-shaped
//!   tower structures (Klausen et al. 2003, *Molecular Microbiology*).
//!
//! # Literature
//!
//! - Monod J (1949) "The growth of bacterial cultures", *Annual Review of Microbiology* 3:371.
//! - Davies DG et al. (1998) "The involvement of cell-to-cell signals in the development of
//!   a bacterial biofilm", *Science* 280:295.
//! - Costerton JW, Stewart PS, Greenberg EP (1999) "Bacterial biofilms: a common cause of
//!   persistent infections", *Science* 284:1318.
//! - Klausen M et al. (2003) "Biofilm formation by *Pseudomonas aeruginosa* wild type,
//!   flagella and type IV pili mutants", *Molecular Microbiology* 48:1511.
//! - Stewart PS (2003) "Diffusion in biofilms", *Journal of Bacteriology* 185:1485.
//! - Flemming HC, Wingender J (2010) "The biofilm matrix", *Nature Reviews Microbiology* 8:623.
//! - Fuqua C, Parsek MR, Greenberg EP (2001) "Regulation of gene expression by cell-to-cell
//!   communication: acyl-homoserine lactone quorum sensing", *Annual Review of Genetics* 35:439.
//! - Balaban NQ et al. (2004) "Bacterial persistence as a phenotypic switch", *Science* 305:1622.

use std::collections::HashMap;

// ── Inline xorshift64 RNG ────────────────────────────────────────────────

/// Xorshift64 pseudo-random number generator.
///
/// Fast, deterministic, zero-dependency RNG suitable for stochastic simulation.
/// Period 2^64 - 1.
#[derive(Debug, Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new RNG from a seed. Seed of 0 is remapped to 1 to avoid
    /// the degenerate all-zeros fixed point.
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Advance state and return a raw u64.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        s
    }

    /// Uniform f64 in [0, 1).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Uniform f64 in [lo, hi).
    #[inline]
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Poisson-distributed random variate (Knuth algorithm for small lambda,
    /// normal approximation for large lambda).
    fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda <= 0.0 {
            return 0;
        }
        if lambda < 30.0 {
            // Knuth's algorithm
            let l = (-lambda).exp();
            let mut k = 0u64;
            let mut p = 1.0f64;
            loop {
                k += 1;
                p *= self.next_f64();
                if p <= l {
                    break;
                }
            }
            k.saturating_sub(1)
        } else {
            // Normal approximation for large lambda
            let u1 = self.next_f64().max(1e-15);
            let u2 = self.next_f64();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let val = lambda + z * lambda.sqrt();
            val.max(0.0).round() as u64
        }
    }
}

// ── Utility ──────────────────────────────────────────────────────────────

/// Clamp a value to [lo, hi].
#[inline]
fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Maximum number of explicit diffusion substeps before switching to a
/// quasi-steady relaxation solve. For ecosystem-scale day/week steps, nutrient
/// diffusion equilibrates far faster than biomass changes, so we accelerate the
/// lower-scale field toward its explicit limit rather than iterating millions
/// of explicit microsteps on the host.
const MAX_EXPLICIT_DIFFUSION_SUBSTEPS: usize = 256;
const MAX_DIFFUSION_RELAX_SWEEPS: usize = 512;
const DIFFUSION_RELAXATION_FACTOR: f64 = 0.85;

// ── Quorum sensing ───────────────────────────────────────────────────────

/// A diffusible quorum-sensing signal molecule.
///
/// Modeled on N-acyl homoserine lactones (AHLs) used by Gram-negative bacteria.
/// Production rate is in nM/cell/s, threshold is the concentration at which
/// cognate receptors activate (typically 1-10 nM for LuxR-type systems).
///
/// Diffusion follows Fick's law with the effective diffusivity reduced by
/// EPS matrix density (Stewart 2003).
#[derive(Debug, Clone)]
pub struct QuorumSignal {
    /// Current bulk concentration (nM).
    pub concentration: f64,
    /// Production rate per active cell (nM/cell/s).
    pub production_rate: f64,
    /// First-order decay rate (1/s). AHL lactonolysis half-life ~4-8 h at pH 7.
    pub decay_rate: f64,
    /// Activation threshold (nM). Below this, cognate receptor is inactive.
    pub threshold: f64,
    /// Free diffusion coefficient (um^2/s). AHL in water: ~500 um^2/s.
    pub diffusion_rate: f64,
}

impl QuorumSignal {
    /// Default AHL-type signal with parameters from Fuqua et al. (2001).
    pub fn ahl_default() -> Self {
        Self {
            concentration: 0.0,
            // ~0.001 nM per cell per second (order-of-magnitude for LasI)
            production_rate: 0.001,
            // Half-life ~6 h → k = ln(2)/21600 ≈ 3.2e-5 /s
            decay_rate: 3.2e-5,
            // LuxR activation threshold ~5 nM
            threshold: 5.0,
            // Free diffusion in water
            diffusion_rate: 500.0,
        }
    }
}

// ── Biofilm cell ─────────────────────────────────────────────────────────

/// A single bacterial cell within the biofilm or in planktonic state.
///
/// Each cell has species identity, spatial position, metabolic parameters,
/// and biofilm-specific traits (EPS production, quorum response, antibiotic
/// tolerance).
#[derive(Debug, Clone)]
pub struct BiofilmCell {
    /// Unique cell identifier.
    pub id: u64,
    /// Species index (0-255).
    pub species: u8,
    /// X position in the simulation domain (um).
    pub x: f64,
    /// Y position in the simulation domain (um).
    pub y: f64,
    /// Maximum specific growth rate mu_max (1/s). Typical: 0.0002-0.0006 for
    /// P. aeruginosa (doubling time 20-60 min).
    pub growth_rate: f64,
    /// EPS production rate (arbitrary units per second per cell).
    pub eps_production: f64,
    /// Quorum sensing response level [0, 1]. 0 = no response, 1 = fully activated.
    pub quorum_response: f64,
    /// Antibiotic tolerance factor. 1.0 = planktonic baseline, 10-1000 = biofilm.
    pub antibiotic_tolerance: f64,
    /// Nutrient uptake rate (Monod V_max analog, um^3/cell/s).
    pub nutrient_uptake: f64,
    /// Whether the cell is alive.
    pub alive: bool,
}

// ── EPS matrix ───────────────────────────────────────────────────────────

/// Extracellular polymeric substance (EPS) matrix on a 2D grid.
///
/// The EPS matrix is the structural scaffold of the biofilm, composed primarily
/// of polysaccharides (Psl, Pel in P. aeruginosa), proteins, and eDNA.
/// It provides mechanical cohesion, antibiotic diffusion barrier, and
/// nutrient gradients (Flemming & Wingender 2010).
#[derive(Debug, Clone)]
pub struct EpsMatrix {
    /// EPS density at each grid cell (arbitrary units, 0 = empty).
    pub density: Vec<Vec<f64>>,
    /// Matrix viscosity affecting cell motility (Pa*s). Biofilm EPS: ~0.01-1 Pa*s.
    pub viscosity: f64,
    /// Diffusion barrier coefficient. Effective D = D_free * exp(-density * barrier).
    /// Typical values 0.1-0.5 (Stewart 2003).
    pub diffusion_barrier: f64,
    /// EPS degradation rate (1/s). Enzymatic and abiotic breakdown.
    pub degradation_rate: f64,
}

impl EpsMatrix {
    /// Create an empty EPS matrix for the given grid dimensions.
    fn new(width: usize, height: usize) -> Self {
        Self {
            density: vec![vec![0.0; width]; height],
            viscosity: 0.05,
            diffusion_barrier: 0.3,
            degradation_rate: 1.0e-5,
        }
    }
}

// ── Time series output ───────────────────────────────────────────────────

/// Time series data collected during a biofilm simulation run.
///
/// Captures key biofilm observables at each recorded time point for
/// post-hoc analysis and visualization.
#[derive(Debug, Clone)]
pub struct BiofilmTimeSeries {
    /// Simulation time at each sample point (s).
    pub times: Vec<f64>,
    /// Total living biomass (cell count) at each time point.
    pub biomass: Vec<f64>,
    /// Fraction of grid covered by EPS above threshold at each time point.
    pub eps_coverage: Vec<f64>,
    /// Per-species cell counts over time.
    pub species_counts: HashMap<u8, Vec<usize>>,
    /// Mean quorum activation fraction over time.
    pub quorum_activation: Vec<f64>,
}

// ── Nutrient field ───────────────────────────────────────────────────────

/// 2D nutrient concentration grid.
///
/// Nutrient (e.g., glucose) diffuses from the bulk medium into the biofilm
/// and is consumed by cells via Monod kinetics. The effective diffusion
/// coefficient is reduced by local EPS density.
#[derive(Debug, Clone)]
struct NutrientField {
    /// Concentration at each grid cell (mM). Typical bulk glucose: 5-20 mM.
    concentration: Vec<Vec<f64>>,
    /// Bulk concentration at the top boundary (mM).
    bulk_concentration: f64,
    /// Free diffusion coefficient in water (um^2/s). Glucose: ~600 um^2/s.
    diffusion_coeff: f64,
    /// Monod half-saturation constant K_s (mM). Glucose for E. coli: ~0.01 mM,
    /// P. aeruginosa: ~0.05 mM.
    half_saturation: f64,
}

impl NutrientField {
    fn new(width: usize, height: usize, bulk: f64, ks: f64) -> Self {
        Self {
            concentration: vec![vec![bulk; width]; height],
            bulk_concentration: bulk,
            diffusion_coeff: 600.0,
            half_saturation: ks,
        }
    }
}

// ── BiofilmSimulator ─────────────────────────────────────────────────────

/// Simulator for spatially-explicit biofilm growth, quorum sensing, and
/// antibiotic response on a 2D grid.
///
/// The grid represents a vertical cross-section of a biofilm with
/// the substratum at y=0 and the bulk medium at y=height. Each grid cell
/// is `cell_size` um on a side (default 5 um, approximately one bacterial
/// cell diameter for P. aeruginosa).
///
/// # Simulation Loop (`step`)
///
/// 1. **Quorum signal update**: production, decay, and diffusion of signal molecules.
/// 2. **Quorum response**: cells update their response level based on local signal.
/// 3. **Nutrient diffusion**: Fick's law with EPS-reduced effective diffusivity.
/// 4. **Growth**: Monod kinetics with nutrient consumption.
/// 5. **EPS production**: secretion proportional to growth rate and quorum activation.
/// 6. **Division**: cells above a biomass threshold divide into adjacent grid positions.
/// 7. **Detachment**: shear-driven removal of surface cells.
/// 8. **Maintenance**: EPS degradation and dead cell lysis.
#[derive(Debug, Clone)]
pub struct BiofilmSimulator {
    /// Grid width in cells.
    width: usize,
    /// Grid height in cells.
    height: usize,
    /// Grid cell edge length (um).
    cell_size: f64,
    /// All bacterial cells (living and dead).
    cells: Vec<BiofilmCell>,
    /// EPS matrix on the grid.
    eps: EpsMatrix,
    /// Quorum-sensing signals.
    signals: Vec<QuorumSignal>,
    /// Nutrient field.
    nutrients: NutrientField,
    /// Deterministic RNG.
    rng: Xorshift64,
    /// Next cell ID to assign.
    next_cell_id: u64,
    /// Simulation clock (s).
    time: f64,
    /// Base shear stress for detachment (Pa). 0 = no flow.
    shear_stress: f64,
    /// EPS threshold for "covered" classification in `eps_coverage()`.
    eps_coverage_threshold: f64,
    /// Antibiotic tolerance multiplier for cells within EPS (vs planktonic = 1.0).
    biofilm_tolerance_factor: f64,
}

impl BiofilmSimulator {
    /// Create an empty biofilm simulator with the given grid dimensions.
    ///
    /// The grid is initialized with uniform nutrient at bulk concentration
    /// (10 mM glucose) and no cells or EPS. The `seed` parameter ensures
    /// deterministic, reproducible simulations.
    ///
    /// # Arguments
    ///
    /// * `width` - Grid width in cells (each cell ~ 5 um).
    /// * `height` - Grid height in cells.
    /// * `seed` - RNG seed for reproducibility.
    pub fn new(width: usize, height: usize, seed: u64) -> Self {
        Self {
            width,
            height,
            cell_size: 5.0, // um, approximately one P. aeruginosa cell diameter
            cells: Vec::new(),
            eps: EpsMatrix::new(width, height),
            signals: Vec::new(),
            nutrients: NutrientField::new(width, height, 10.0, 0.05),
            rng: Xorshift64::new(seed),
            next_cell_id: 0,
            time: 0.0,
            shear_stress: 0.0,
            eps_coverage_threshold: 0.1,
            biofilm_tolerance_factor: 100.0, // 100x baseline for biofilm cells
        }
    }

    /// Seed bacterial cells of a given species at random positions on the
    /// substratum (y = 0..2 rows, initial attachment zone).
    ///
    /// # Arguments
    ///
    /// * `species` - Species index.
    /// * `count` - Number of cells to seed.
    /// * `growth_rate` - Maximum specific growth rate mu_max (1/s).
    pub fn seed_cells(&mut self, species: u8, count: usize, growth_rate: f64) {
        for _ in 0..count {
            let x = self.rng.next_range(0.0, self.width as f64 * self.cell_size);
            // Initial attachment near substratum (y=0..2 cell heights)
            let y = self.rng.next_range(0.0, 2.0 * self.cell_size);
            let cell = BiofilmCell {
                id: self.next_cell_id,
                species,
                x,
                y,
                growth_rate,
                eps_production: 0.01,
                quorum_response: 0.0,
                antibiotic_tolerance: 1.0, // planktonic baseline
                nutrient_uptake: 0.1,
                alive: true,
            };
            self.cells.push(cell);
            self.next_cell_id += 1;
        }
    }

    /// Register a quorum-sensing signal and return its index.
    pub fn add_signal(&mut self, signal: QuorumSignal) -> usize {
        let id = self.signals.len();
        self.signals.push(signal);
        id
    }

    /// Advance the simulation by one time step of duration `dt` seconds.
    ///
    /// Executes the full biofilm lifecycle: quorum signal dynamics, nutrient
    /// diffusion and consumption, Monod growth, EPS secretion, cell division,
    /// and detachment.
    pub fn step(&mut self, dt: f64) {
        self.step_quorum_signals(dt);
        self.step_quorum_response();
        self.step_nutrient_diffusion(dt);
        self.step_growth_and_consumption(dt);
        self.step_eps_production(dt);
        self.step_division();
        self.step_eps_degradation(dt);
        self.step_detachment(self.shear_stress);
        self.update_antibiotic_tolerance();
        self.time += dt;
    }

    // ── Quorum sensing dynamics ──────────────────────────────────────

    /// Update quorum signal concentrations: production by living cells,
    /// first-order decay, and simplified spatial averaging (well-mixed
    /// approximation within diffusion length).
    fn step_quorum_signals(&mut self, dt: f64) {
        let alive_count = self.cells.iter().filter(|c| c.alive).count() as f64;
        for signal in &mut self.signals {
            // Production: each living cell contributes production_rate * dt
            let production = signal.production_rate * alive_count * dt;
            // Decay: first-order
            let decay = signal.concentration * signal.decay_rate * dt;
            signal.concentration = (signal.concentration + production - decay).max(0.0);
        }
    }

    /// Update each cell's quorum response based on local signal concentration.
    /// Uses a Hill-function response: response = C^n / (K^n + C^n) with n=2.
    fn step_quorum_response(&mut self) {
        if self.signals.is_empty() {
            return;
        }
        // Use the maximum signal activation across all signals
        let max_activation: f64 = self
            .signals
            .iter()
            .map(|s| {
                let c = s.concentration;
                let k = s.threshold;
                // Hill function with cooperativity n=2
                let c2 = c * c;
                let k2 = k * k;
                c2 / (k2 + c2)
            })
            .fold(0.0f64, |a, b| a.max(b));

        for cell in &mut self.cells {
            if cell.alive {
                // Cells update response with some inertia (exponential moving average)
                cell.quorum_response += 0.1 * (max_activation - cell.quorum_response);
                cell.quorum_response = clamp_f64(cell.quorum_response, 0.0, 1.0);
            }
        }
    }

    // ── Nutrient dynamics ────────────────────────────────────────────

    /// Diffuse nutrients through the grid using explicit finite differences.
    ///
    /// Effective diffusion is reduced by local EPS density:
    ///   D_eff = D_free * exp(-eps_density * barrier_coeff)
    /// (Stewart 2003, *Journal of Bacteriology*).
    ///
    /// Top boundary (y = height-1) is held at bulk concentration (Dirichlet).
    /// Bottom (substratum) and sides use no-flux (Neumann) boundary conditions.
    fn step_nutrient_diffusion(&mut self, dt: f64) {
        let h = self.cell_size;
        let h2 = h * h;

        // Stability criterion for explicit 2D diffusion: dt_sub <= h^2 / (4 * D_max).
        // D_max is the free diffusion coefficient (EPS only reduces it).
        let d_max = self.nutrients.diffusion_coeff;
        let dt_stable = h2 / (4.0 * d_max + 1.0e-30); // avoid div-by-zero
        let n_sub = ((dt / dt_stable).ceil() as usize).max(1);
        if n_sub <= MAX_EXPLICIT_DIFFUSION_SUBSTEPS {
            let dt_sub = dt / n_sub as f64;
            for _ in 0..n_sub {
                self.step_nutrient_diffusion_explicit(dt_sub);
            }
            return;
        }

        let relax_sweeps = ((dt / dt_stable).sqrt().ceil() as usize)
            .clamp(MAX_EXPLICIT_DIFFUSION_SUBSTEPS, MAX_DIFFUSION_RELAX_SWEEPS);
        self.relax_nutrient_diffusion(relax_sweeps);
    }

    fn step_nutrient_diffusion_explicit(&mut self, dt_sub: f64) {
        let h = self.cell_size;
        let h2 = h * h;
        let rows = self.height;
        let cols = self.width;
        let mut new_conc = self.nutrients.concentration.clone();

        for r in 0..rows {
            for c in 0..cols {
                // Effective diffusivity at this cell
                let eps_here = self.eps.density[r][c];
                let d_eff =
                    self.nutrients.diffusion_coeff * (-eps_here * self.eps.diffusion_barrier).exp();

                // Neighbors with boundary conditions
                let up = if r + 1 < rows {
                    self.nutrients.concentration[r + 1][c]
                } else {
                    self.nutrients.bulk_concentration // Dirichlet top
                };
                let down = if r > 0 {
                    self.nutrients.concentration[r - 1][c]
                } else {
                    self.nutrients.concentration[r][c] // no-flux bottom
                };
                let left = if c > 0 {
                    self.nutrients.concentration[r][c - 1]
                } else {
                    self.nutrients.concentration[r][c] // no-flux
                };
                let right = if c + 1 < cols {
                    self.nutrients.concentration[r][c + 1]
                } else {
                    self.nutrients.concentration[r][c] // no-flux
                };

                let laplacian =
                    (up + down + left + right - 4.0 * self.nutrients.concentration[r][c]) / h2;

                new_conc[r][c] =
                    (self.nutrients.concentration[r][c] + d_eff * laplacian * dt_sub).max(0.0);
            }
        }

        // Enforce top boundary
        if rows > 0 {
            for c in 0..cols {
                new_conc[rows - 1][c] = self.nutrients.bulk_concentration;
            }
        }

        self.nutrients.concentration = new_conc;
    }

    fn relax_nutrient_diffusion(&mut self, sweeps: usize) {
        let rows = self.height;
        let cols = self.width;
        let d_max = self.nutrients.diffusion_coeff.max(1.0e-30);

        for _ in 0..sweeps {
            let mut new_conc = self.nutrients.concentration.clone();

            for r in 0..rows {
                for c in 0..cols {
                    let eps_here = self.eps.density[r][c];
                    let d_eff = self.nutrients.diffusion_coeff
                        * (-eps_here * self.eps.diffusion_barrier).exp();
                    let local_relax =
                        (DIFFUSION_RELAXATION_FACTOR * (d_eff / d_max)).clamp(0.0, 1.0);

                    let up = if r + 1 < rows {
                        self.nutrients.concentration[r + 1][c]
                    } else {
                        self.nutrients.bulk_concentration
                    };
                    let down = if r > 0 {
                        self.nutrients.concentration[r - 1][c]
                    } else {
                        self.nutrients.concentration[r][c]
                    };
                    let left = if c > 0 {
                        self.nutrients.concentration[r][c - 1]
                    } else {
                        self.nutrients.concentration[r][c]
                    };
                    let right = if c + 1 < cols {
                        self.nutrients.concentration[r][c + 1]
                    } else {
                        self.nutrients.concentration[r][c]
                    };

                    let neighbor_mean = 0.25 * (up + down + left + right);
                    let here = self.nutrients.concentration[r][c];
                    new_conc[r][c] = (here + local_relax * (neighbor_mean - here)).max(0.0);
                }
            }

            if rows > 0 {
                for c in 0..cols {
                    new_conc[rows - 1][c] = self.nutrients.bulk_concentration;
                }
            }

            self.nutrients.concentration = new_conc;
        }
    }

    // ── Growth and nutrient consumption ──────────────────────────────

    /// Monod-limited growth: mu = mu_max * S / (K_s + S).
    ///
    /// Each living cell consumes nutrient from its grid cell proportional to
    /// its growth rate. Division is handled separately in `step_division`.
    fn step_growth_and_consumption(&mut self, dt: f64) {
        let cell_size = self.cell_size;
        let width = self.width;
        let height = self.height;
        let ks = self.nutrients.half_saturation;

        for i in 0..self.cells.len() {
            if !self.cells[i].alive {
                continue;
            }

            // Map cell position to grid coordinates
            let gc = ((self.cells[i].x / cell_size) as usize).min(width.saturating_sub(1));
            let gr = ((self.cells[i].y / cell_size) as usize).min(height.saturating_sub(1));

            let s = self.nutrients.concentration[gr][gc];

            // Monod kinetics
            let mu = self.cells[i].growth_rate * s / (ks + s);

            // Nutrient consumption (yield coefficient ~0.5)
            let consumption = mu * self.cells[i].nutrient_uptake * dt;
            self.nutrients.concentration[gr][gc] =
                (self.nutrients.concentration[gr][gc] - consumption).max(0.0);

            // Stochastic death at very low nutrient (starvation)
            if s < ks * 0.001 {
                let death_prob = 1.0e-5 * dt; // very low starvation death rate
                if self.rng.next_f64() < death_prob {
                    self.cells[i].alive = false;
                }
            }
        }
    }

    // ── EPS production ───────────────────────────────────────────────

    /// EPS secretion by living cells. Production rate is upregulated by
    /// quorum sensing activation (Davies et al. 1998).
    fn step_eps_production(&mut self, dt: f64) {
        let cell_size = self.cell_size;
        let width = self.width;
        let height = self.height;

        for i in 0..self.cells.len() {
            if !self.cells[i].alive {
                continue;
            }

            let gc = ((self.cells[i].x / cell_size) as usize).min(width.saturating_sub(1));
            let gr = ((self.cells[i].y / cell_size) as usize).min(height.saturating_sub(1));

            // Base EPS production + quorum-enhanced production (up to 5x)
            let quorum_boost = 1.0 + 4.0 * self.cells[i].quorum_response;
            let eps_added = self.cells[i].eps_production * quorum_boost * dt;
            self.eps.density[gr][gc] += eps_added;
        }
    }

    /// EPS matrix degradation (enzymatic + abiotic).
    fn step_eps_degradation(&mut self, dt: f64) {
        let rate = self.eps.degradation_rate;
        for row in &mut self.eps.density {
            for val in row.iter_mut() {
                *val = (*val - *val * rate * dt).max(0.0);
            }
        }
    }

    // ── Cell division ────────────────────────────────────────────────

    /// Cells divide when a stochastic check succeeds, weighted by local
    /// nutrient availability (Monod growth rate). Daughter cells are placed
    /// adjacent to the mother cell, pushing the biofilm upward.
    fn step_division(&mut self) {
        let ks = self.nutrients.half_saturation;
        let cell_size = self.cell_size;
        let width = self.width;
        let height = self.height;
        let max_y = (height as f64) * cell_size;

        let mut new_cells: Vec<BiofilmCell> = Vec::new();

        for i in 0..self.cells.len() {
            if !self.cells[i].alive {
                continue;
            }

            let gc = ((self.cells[i].x / cell_size) as usize).min(width.saturating_sub(1));
            let gr = ((self.cells[i].y / cell_size) as usize).min(height.saturating_sub(1));
            let s = self.nutrients.concentration[gr][gc];
            let mu = self.cells[i].growth_rate * s / (ks + s);

            // Division probability per step scales with growth rate
            // (dt is baked into growth_rate as 1/s, but division is checked per step)
            let div_prob = mu * 0.5; // conservative division probability
            if self.rng.next_f64() < div_prob {
                // Daughter cell offset: slightly upward and random lateral jitter
                let dx = self.rng.next_range(-cell_size * 0.5, cell_size * 0.5);
                let dy = self.rng.next_range(0.0, cell_size);

                let new_x = clamp_f64(self.cells[i].x + dx, 0.0, (width as f64) * cell_size);
                let new_y = clamp_f64(self.cells[i].y + dy, 0.0, max_y);

                let daughter = BiofilmCell {
                    id: self.next_cell_id,
                    species: self.cells[i].species,
                    x: new_x,
                    y: new_y,
                    growth_rate: self.cells[i].growth_rate,
                    eps_production: self.cells[i].eps_production,
                    quorum_response: self.cells[i].quorum_response * 0.5, // partial inheritance
                    antibiotic_tolerance: 1.0, // daughter starts planktonic
                    nutrient_uptake: self.cells[i].nutrient_uptake,
                    alive: true,
                };
                new_cells.push(daughter);
                self.next_cell_id += 1;
            }
        }

        self.cells.extend(new_cells);
    }

    // ── Detachment ───────────────────────────────────────────────────

    /// Shear-mediated detachment. All living cells are subject to shear-driven
    /// removal with probability proportional to shear stress and inversely
    /// proportional to local EPS density. Cells at higher positions (further
    /// from the substratum) have additional exposure.
    ///
    /// Returns the number of cells detached.
    fn step_detachment(&mut self, shear_stress: f64) -> usize {
        if shear_stress <= 0.0 {
            return 0;
        }

        let cell_size = self.cell_size;
        let width = self.width;
        let height = self.height;
        let max_y = (height as f64) * cell_size;
        let mut detached = 0usize;

        for i in 0..self.cells.len() {
            if !self.cells[i].alive {
                continue;
            }

            // Base exposure factor: even cells at the substratum are exposed to shear.
            // Higher cells get additional exposure.
            let height_factor = 0.3 + 0.7 * (self.cells[i].y / max_y);

            // EPS protection: high EPS density reduces detachment
            let gc = ((self.cells[i].x / cell_size) as usize).min(width.saturating_sub(1));
            let gr = ((self.cells[i].y / cell_size) as usize).min(height.saturating_sub(1));
            let eps_here = self.eps.density[gr][gc];
            let eps_protection = (-eps_here * 2.0).exp(); // higher EPS -> lower detachment

            let detach_prob = shear_stress * height_factor * eps_protection * 0.01;
            if self.rng.next_f64() < detach_prob {
                self.cells[i].alive = false;
                detached += 1;
            }
        }

        detached
    }

    // ── Antibiotic tolerance update ──────────────────────────────────

    /// Update antibiotic tolerance for each cell based on local EPS density.
    /// Cells embedded in thick EPS matrix exhibit 10-1000x tolerance vs
    /// planktonic cells (Costerton et al. 1999).
    fn update_antibiotic_tolerance(&mut self) {
        let cell_size = self.cell_size;
        let width = self.width;
        let height = self.height;
        let base_factor = self.biofilm_tolerance_factor;

        for cell in &mut self.cells {
            if !cell.alive {
                continue;
            }
            let gc = ((cell.x / cell_size) as usize).min(width.saturating_sub(1));
            let gr = ((cell.y / cell_size) as usize).min(height.saturating_sub(1));
            let eps_here = self.eps.density[gr][gc];

            // Tolerance scales with local EPS density, capped at biofilm_tolerance_factor
            // A cell in dense EPS (density > 1.0) gets ~full biofilm tolerance
            let eps_fraction = clamp_f64(eps_here / 1.0, 0.0, 1.0);
            cell.antibiotic_tolerance = 1.0 + (base_factor - 1.0) * eps_fraction;
        }
    }

    // ── Public query methods ─────────────────────────────────────────

    /// Number of living cells.
    pub fn cell_count(&self) -> usize {
        self.cells.iter().filter(|c| c.alive).count()
    }

    /// Total biomass (living cell count as f64).
    pub fn biomass(&self) -> f64 {
        self.cell_count() as f64
    }

    /// Fraction of grid cells covered by EPS above the coverage threshold.
    pub fn eps_coverage(&self) -> f64 {
        let total = (self.width * self.height) as f64;
        if total == 0.0 {
            return 0.0;
        }
        let covered = self
            .eps
            .density
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v > self.eps_coverage_threshold)
            .count() as f64;
        covered / total
    }

    /// Distribution of living cells by species.
    pub fn species_distribution(&self) -> HashMap<u8, usize> {
        let mut dist = HashMap::new();
        for cell in &self.cells {
            if cell.alive {
                *dist.entry(cell.species).or_insert(0) += 1;
            }
        }
        dist
    }

    /// Fraction of living cells with quorum response above 0.5 for
    /// the given signal.
    pub fn quorum_activated_fraction(&self, signal_id: usize) -> f64 {
        if signal_id >= self.signals.len() {
            return 0.0;
        }
        let alive: Vec<&BiofilmCell> = self.cells.iter().filter(|c| c.alive).collect();
        if alive.is_empty() {
            return 0.0;
        }
        let activated = alive.iter().filter(|c| c.quorum_response > 0.5).count();
        activated as f64 / alive.len() as f64
    }

    /// Apply an antibiotic challenge at the given concentration (ug/mL) for
    /// duration `dt` seconds. Returns the number of cells killed.
    ///
    /// Kill probability for each cell is:
    ///   P_kill = concentration / (concentration + MIC * tolerance) * dt * k_kill
    ///
    /// where MIC = 1.0 ug/mL (baseline) and k_kill = 0.001 /s (killing rate constant).
    /// Biofilm-embedded cells with high tolerance are far less likely to be killed.
    pub fn antibiotic_challenge(&mut self, concentration: f64, dt: f64) -> usize {
        let mic = 1.0; // baseline MIC (ug/mL)
        let k_kill = 0.001; // killing rate constant (1/s)
        let mut killed = 0usize;

        for cell in &mut self.cells {
            if !cell.alive {
                continue;
            }
            let effective_mic = mic * cell.antibiotic_tolerance;
            let kill_prob = concentration / (concentration + effective_mic) * dt * k_kill;
            // Use a fresh random draw for each cell
            // (Inline RNG advancement — safe because we own &mut self)
            let r = {
                let mut s = self.rng.state;
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                self.rng.state = s;
                (s >> 11) as f64 / ((1u64 << 53) as f64)
            };
            if r < kill_prob {
                cell.alive = false;
                killed += 1;
            }
        }

        killed
    }

    /// Apply shear stress and return the number of cells detached.
    pub fn detachment_events(&mut self, shear_stress: f64) -> usize {
        self.step_detachment(shear_stress)
    }

    /// Return the current nutrient concentration grid.
    pub fn nutrient_profile(&self) -> Vec<Vec<f64>> {
        self.nutrients.concentration.clone()
    }

    /// Run a complete simulation for the given number of steps, recording
    /// time series data at each step.
    pub fn run_simulation(&mut self, steps: usize, dt: f64) -> BiofilmTimeSeries {
        let mut ts = BiofilmTimeSeries {
            times: Vec::with_capacity(steps),
            biomass: Vec::with_capacity(steps),
            eps_coverage: Vec::with_capacity(steps),
            species_counts: HashMap::new(),
            quorum_activation: Vec::with_capacity(steps),
        };

        // Initialize species tracking for all species currently present
        for cell in &self.cells {
            ts.species_counts
                .entry(cell.species)
                .or_insert_with(Vec::new);
        }

        for _ in 0..steps {
            self.step(dt);

            ts.times.push(self.time);
            ts.biomass.push(self.biomass());
            ts.eps_coverage.push(self.eps_coverage());

            // Per-species counts
            let dist = self.species_distribution();
            for (species, counts_vec) in ts.species_counts.iter_mut() {
                counts_vec.push(*dist.get(species).unwrap_or(&0));
            }
            // Track new species that appeared via mutation/etc
            for (species, count) in &dist {
                if !ts.species_counts.contains_key(species) {
                    let mut v = vec![0usize; ts.times.len() - 1];
                    v.push(*count);
                    ts.species_counts.insert(*species, v);
                }
            }

            // Mean quorum activation
            let qa = if !self.signals.is_empty() {
                self.quorum_activated_fraction(0)
            } else {
                0.0
            };
            ts.quorum_activation.push(qa);
        }

        ts
    }

    /// Detect mushroom-shaped biofilm structures.
    ///
    /// Mushroom structures are characteristic of mature *P. aeruginosa* biofilms
    /// (Klausen et al. 2003). They consist of a narrow stalk near the substratum
    /// with a wider cap in the upper portion. We detect these by scanning columns
    /// for regions where the cell/EPS density profile narrows then widens moving
    /// upward from the substratum.
    ///
    /// A column qualifies as a mushroom if:
    /// 1. There is EPS density at the base (stalk).
    /// 2. There is a constriction (local minimum in horizontal extent).
    /// 3. Above the constriction, density expands again (cap).
    pub fn mushroom_structure_count(&self) -> usize {
        // Build a column density profile: for each x column, sum EPS across a
        // vertical window and look for the stalk-constriction-cap pattern.
        let mut mushroom_count = 0usize;

        // Aggregate EPS density per column
        for c in 0..self.width {
            let col_profile: Vec<f64> = (0..self.height).map(|r| self.eps.density[r][c]).collect();

            // Need at least 5 rows to detect a mushroom pattern
            if col_profile.len() < 5 {
                continue;
            }

            // Look for: base > 0, then decrease, then increase
            let base_density = col_profile[0] + col_profile[1];
            if base_density < self.eps_coverage_threshold {
                continue; // no stalk
            }

            // Find a local minimum above the base
            let mut found_constriction = false;
            let mut constriction_idx = 0usize;
            for r in 2..col_profile.len().saturating_sub(2) {
                if col_profile[r] < col_profile[r - 1]
                    && col_profile[r] < col_profile[r + 1]
                    && col_profile[r] < base_density * 0.5
                {
                    found_constriction = true;
                    constriction_idx = r;
                    break;
                }
            }

            if !found_constriction {
                continue;
            }

            // Look for cap expansion above the constriction
            let constriction_val = col_profile[constriction_idx];
            let mut found_cap = false;
            for r in (constriction_idx + 1)..col_profile.len() {
                if col_profile[r] > constriction_val * 2.0
                    && col_profile[r] > self.eps_coverage_threshold
                {
                    found_cap = true;
                    break;
                }
            }

            if found_cap {
                mushroom_count += 1;
            }
        }

        mushroom_count
    }
}

// ── Preset configurations ────────────────────────────────────────────────

/// Single-species *Pseudomonas aeruginosa*-like biofilm with LasI/LasR-type
/// AHL quorum sensing.
///
/// Parameters from Davies et al. (1998) and Klausen et al. (2003):
/// - Doubling time ~35 min → mu_max ≈ 0.00033 /s
/// - AHL threshold ~5 nM (LasR activation)
/// - Biofilm forms within 4-8 hours under flow conditions
pub fn pseudomonas_biofilm(seed: u64) -> BiofilmSimulator {
    let mut sim = BiofilmSimulator::new(50, 30, seed);

    // P. aeruginosa growth rate: doubling time ~35 min
    // mu_max = ln(2) / (35 * 60) ≈ 3.3e-4 /s
    sim.seed_cells(0, 50, 3.3e-4);

    // LasI/LasR AHL quorum sensing
    sim.add_signal(QuorumSignal {
        concentration: 0.0,
        production_rate: 0.002, // nM/cell/s (LasI production)
        decay_rate: 3.2e-5,     // AHL half-life ~6 h
        threshold: 5.0,         // nM (LasR activation)
        diffusion_rate: 500.0,  // um^2/s in water
    });

    // Pseudomonas-specific EPS parameters (Psl/Pel polysaccharides)
    sim.eps.viscosity = 0.1; // Pa*s
    sim.eps.diffusion_barrier = 0.3; // moderate barrier
    sim.eps.degradation_rate = 5.0e-6;

    // Nutrient: glucose at 10 mM, K_s ~ 0.05 mM
    sim.nutrients.bulk_concentration = 10.0;
    sim.nutrients.half_saturation = 0.05;

    sim.biofilm_tolerance_factor = 500.0; // Pseudomonas biofilms: up to 500x tolerance

    sim
}

/// Three-species mixed biofilm with interspecies competition and cooperation.
///
/// Species 0: Primary colonizer (fast growth, low EPS)
/// Species 1: EPS producer (slow growth, high matrix production)
/// Species 2: Opportunist (moderate growth, high quorum response)
///
/// This models the ecological succession observed in environmental biofilms
/// and chronic wound infections.
pub fn mixed_species_biofilm(seed: u64) -> BiofilmSimulator {
    let mut sim = BiofilmSimulator::new(60, 40, seed);

    // Species 0: Primary colonizer — fast grower, low EPS
    sim.seed_cells(0, 40, 5.0e-4); // doubling ~23 min

    // Species 1: EPS producer — slow grower, heavy matrix
    sim.seed_cells(1, 20, 2.0e-4); // doubling ~58 min

    // Species 2: Opportunist — moderate growth, quorum-responsive
    sim.seed_cells(2, 15, 3.5e-4); // doubling ~33 min

    // Modify EPS production rates by species
    for cell in &mut sim.cells {
        match cell.species {
            0 => {
                cell.eps_production = 0.005; // low EPS
                cell.nutrient_uptake = 0.15; // efficient uptake
            }
            1 => {
                cell.eps_production = 0.05; // high EPS
                cell.nutrient_uptake = 0.08; // less competitive uptake
            }
            2 => {
                cell.eps_production = 0.015; // moderate EPS
                cell.nutrient_uptake = 0.12;
            }
            _ => {}
        }
    }

    // Shared quorum signal (AI-2 type, interspecies)
    sim.add_signal(QuorumSignal {
        concentration: 0.0,
        production_rate: 0.001,
        decay_rate: 5.0e-5,
        threshold: 3.0, // lower threshold for AI-2
        diffusion_rate: 600.0,
    });

    // Species-specific signal (AHL-type)
    sim.add_signal(QuorumSignal {
        concentration: 0.0,
        production_rate: 0.0015,
        decay_rate: 3.2e-5,
        threshold: 8.0,
        diffusion_rate: 500.0,
    });

    sim.nutrients.bulk_concentration = 15.0; // richer medium for multi-species
    sim.biofilm_tolerance_factor = 200.0;

    sim
}

/// Medical device (catheter) biofilm with periodic antibiotic exposure.
///
/// Models urinary catheter biofilm formation with parameters from
/// Stickler (2008) *Nature Clinical Practice Urology*:
/// - Low nutrient (urine ~2-5 mM glucose equivalent)
/// - Moderate shear from urine flow
/// - Periodic ciprofloxacin exposure
pub fn catheter_biofilm(seed: u64) -> BiofilmSimulator {
    let mut sim = BiofilmSimulator::new(40, 25, seed);

    // E. coli — common uropathogen
    sim.seed_cells(0, 30, 4.0e-4); // doubling ~29 min

    // Enterococcus — slow-growing, highly tolerant
    sim.seed_cells(1, 15, 1.5e-4); // doubling ~77 min

    for cell in &mut sim.cells {
        match cell.species {
            0 => {
                cell.eps_production = 0.012;
                cell.nutrient_uptake = 0.1;
            }
            1 => {
                cell.eps_production = 0.02;
                cell.nutrient_uptake = 0.06;
            }
            _ => {}
        }
    }

    sim.add_signal(QuorumSignal::ahl_default());

    // Low nutrient environment (urine)
    sim.nutrients.bulk_concentration = 3.0; // mM glucose equivalent
    sim.nutrients.half_saturation = 0.02; // E. coli K_s for glucose

    // Moderate shear from urine flow
    sim.shear_stress = 0.1; // Pa

    sim.biofilm_tolerance_factor = 1000.0; // catheter biofilms: extremely tolerant

    sim
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_biofilm() {
        let sim = BiofilmSimulator::new(20, 10, 42);
        assert_eq!(sim.cell_count(), 0);
        assert_eq!(sim.biomass(), 0.0);
        assert_eq!(sim.eps_coverage(), 0.0);
        assert!(sim.species_distribution().is_empty());
    }

    #[test]
    fn seed_cells_adds_cells() {
        let mut sim = BiofilmSimulator::new(20, 10, 42);
        sim.seed_cells(0, 25, 3.3e-4);
        assert_eq!(sim.cell_count(), 25);
        assert_eq!(sim.biomass(), 25.0);

        let dist = sim.species_distribution();
        assert_eq!(dist[&0], 25);

        // All cells should be near the substratum (y < 2 * cell_size)
        for cell in &sim.cells {
            assert!(
                cell.y < 2.0 * sim.cell_size + 0.01,
                "Cell y={} exceeds initial attachment zone",
                cell.y
            );
            assert!(cell.alive);
        }
    }

    #[test]
    fn cells_grow_with_nutrients() {
        let mut sim = BiofilmSimulator::new(30, 20, 123);
        sim.seed_cells(0, 20, 5.0e-4); // fast growers
        sim.nutrients.bulk_concentration = 20.0; // rich medium

        let initial = sim.cell_count();

        // Run enough steps for division events
        for _ in 0..500 {
            sim.step(1.0);
        }

        assert!(
            sim.cell_count() > initial,
            "Expected growth: initial={}, final={}",
            initial,
            sim.cell_count()
        );
    }

    #[test]
    fn eps_production_increases_over_time() {
        let mut sim = BiofilmSimulator::new(20, 15, 77);
        sim.seed_cells(0, 30, 3.3e-4);

        let initial_coverage = sim.eps_coverage();

        // Simulate for a while
        for _ in 0..200 {
            sim.step(1.0);
        }

        let final_coverage = sim.eps_coverage();
        assert!(
            final_coverage >= initial_coverage,
            "EPS coverage should not decrease: initial={}, final={}",
            initial_coverage,
            final_coverage
        );

        // Verify some EPS was actually deposited
        let total_eps: f64 = sim.eps.density.iter().flat_map(|row| row.iter()).sum();
        assert!(
            total_eps > 0.0,
            "Total EPS density should be positive after simulation"
        );
    }

    #[test]
    fn quorum_sensing_activates_at_threshold() {
        let mut sim = BiofilmSimulator::new(20, 10, 99);
        sim.seed_cells(0, 100, 3.3e-4); // many cells for fast signal accumulation

        let signal_id = sim.add_signal(QuorumSignal {
            concentration: 0.0,
            production_rate: 0.01, // high production for fast activation
            decay_rate: 1.0e-5,    // slow decay
            threshold: 5.0,
            diffusion_rate: 500.0,
        });

        // Initially no activation
        assert_eq!(sim.quorum_activated_fraction(signal_id), 0.0);

        // Run until signal accumulates above threshold
        for _ in 0..200 {
            sim.step(1.0);
        }

        // Signal should have accumulated: 100 cells * 0.01 nM/cell/s * 200s = 200 nM >> 5 nM threshold
        assert!(
            sim.signals[signal_id].concentration > sim.signals[signal_id].threshold,
            "Signal concentration ({:.2} nM) should exceed threshold ({:.2} nM)",
            sim.signals[signal_id].concentration,
            sim.signals[signal_id].threshold
        );

        let activated = sim.quorum_activated_fraction(signal_id);
        assert!(
            activated > 0.0,
            "Some cells should be quorum-activated, got {}",
            activated
        );
    }

    #[test]
    fn antibiotic_kills_planktonic_more_than_biofilm() {
        // Create two simulators: one with EPS (biofilm), one without (planktonic analog)
        let mut biofilm = BiofilmSimulator::new(20, 10, 42);
        biofilm.seed_cells(0, 100, 3.3e-4);

        // Build up EPS in the biofilm
        for _ in 0..300 {
            biofilm.step(1.0);
        }

        let biofilm_count_before = biofilm.cell_count();

        // Clone for planktonic comparison — strip all EPS
        let mut planktonic = biofilm.clone();
        for row in &mut planktonic.eps.density {
            for val in row.iter_mut() {
                *val = 0.0;
            }
        }
        // Reset tolerance to planktonic baseline
        for cell in &mut planktonic.cells {
            cell.antibiotic_tolerance = 1.0;
        }

        let planktonic_count_before = planktonic.cell_count();

        // Apply strong antibiotic challenge
        let biofilm_killed = biofilm.antibiotic_challenge(10.0, 100.0);
        let planktonic_killed = planktonic.antibiotic_challenge(10.0, 100.0);

        // Planktonic cells should die more than biofilm cells
        let biofilm_survival =
            (biofilm_count_before - biofilm_killed) as f64 / biofilm_count_before as f64;
        let planktonic_survival =
            (planktonic_count_before - planktonic_killed) as f64 / planktonic_count_before as f64;

        assert!(
            biofilm_survival >= planktonic_survival,
            "Biofilm should survive better: biofilm={:.2}%, planktonic={:.2}%",
            biofilm_survival * 100.0,
            planktonic_survival * 100.0
        );
    }

    #[test]
    fn detachment_at_high_shear() {
        let mut sim = BiofilmSimulator::new(30, 20, 55);
        sim.seed_cells(0, 80, 3.3e-4);

        // Let biofilm grow a bit
        for _ in 0..100 {
            sim.step(1.0);
        }

        let before = sim.cell_count();

        // Apply very high shear repeatedly to ensure statistical detachment
        let mut total_detached = 0usize;
        for _ in 0..20 {
            total_detached += sim.detachment_events(50.0); // 50 Pa — extreme shear
        }

        assert!(
            total_detached > 0 || before == 0,
            "Repeated high shear should detach some cells (before={}, detached={})",
            before,
            total_detached
        );
    }

    #[test]
    fn nutrient_gradient_forms() {
        let mut sim = BiofilmSimulator::new(20, 15, 33);
        sim.seed_cells(0, 60, 5.0e-4);
        sim.nutrients.bulk_concentration = 10.0;

        // Run simulation to create nutrient gradients via consumption
        for _ in 0..300 {
            sim.step(1.0);
        }

        let profile = sim.nutrient_profile();

        // Top row should be near bulk concentration (Dirichlet boundary)
        let top_mean: f64 =
            profile.last().unwrap().iter().sum::<f64>() / profile.last().unwrap().len() as f64;

        // Bottom rows should be depleted (cells consume nutrients)
        let bottom_mean: f64 = profile[0].iter().sum::<f64>() / profile[0].len() as f64;

        assert!(
            top_mean >= bottom_mean,
            "Nutrient gradient: top ({:.4} mM) should >= bottom ({:.4} mM)",
            top_mean,
            bottom_mean
        );

        // Top should be close to bulk concentration
        assert!(
            top_mean > sim.nutrients.bulk_concentration * 0.5,
            "Top nutrient ({:.4} mM) should be near bulk ({:.4} mM)",
            top_mean,
            sim.nutrients.bulk_concentration
        );
    }

    #[test]
    fn mixed_species_coexist() {
        let mut sim = mixed_species_biofilm(42);

        let initial_dist = sim.species_distribution();
        assert_eq!(initial_dist.len(), 3, "Should have 3 species initially");

        // Run for a while
        for _ in 0..200 {
            sim.step(1.0);
        }

        let final_dist = sim.species_distribution();

        // At least 2 species should survive
        let surviving_species = final_dist.len();
        assert!(
            surviving_species >= 2,
            "At least 2 species should coexist, got {}",
            surviving_species
        );

        // Total biomass should have increased
        assert!(
            sim.cell_count() > 50,
            "Mixed biofilm should grow: count={}",
            sim.cell_count()
        );
    }

    #[test]
    fn pseudomonas_preset_runs() {
        let mut sim = pseudomonas_biofilm(42);

        assert_eq!(sim.cell_count(), 50);
        assert_eq!(sim.signals.len(), 1);

        // Run simulation
        let ts = sim.run_simulation(100, 1.0);

        assert_eq!(ts.times.len(), 100);
        assert_eq!(ts.biomass.len(), 100);
        assert_eq!(ts.eps_coverage.len(), 100);
        assert_eq!(ts.quorum_activation.len(), 100);

        // Biomass should increase over 100 seconds
        let first_biomass = ts.biomass[0];
        let last_biomass = *ts.biomass.last().unwrap();
        assert!(
            last_biomass >= first_biomass,
            "Biomass should grow: first={}, last={}",
            first_biomass,
            last_biomass
        );
    }

    #[test]
    fn catheter_preset_has_shear() {
        let sim = catheter_biofilm(42);
        assert!(
            sim.shear_stress > 0.0,
            "Catheter biofilm should have flow shear"
        );
        assert_eq!(sim.species_distribution().len(), 2, "Should have 2 species");
        assert!(
            sim.biofilm_tolerance_factor >= 1000.0,
            "Catheter biofilm should have extreme tolerance"
        );
    }

    #[test]
    fn mushroom_structures_form_at_maturity() {
        // Manually construct a biofilm with a mushroom-like EPS profile
        let mut sim = BiofilmSimulator::new(10, 10, 42);

        // Create a mushroom pattern in column 3:
        // Base (rows 0-1): moderate density (stalk)
        // Constriction (row 3): low density
        // Cap (rows 5-7): high density
        sim.eps.density[0][3] = 1.0; // stalk base
        sim.eps.density[1][3] = 0.8; // stalk
        sim.eps.density[2][3] = 0.3; // narrowing
        sim.eps.density[3][3] = 0.05; // constriction (< 50% of base)
        sim.eps.density[4][3] = 0.2; // cap expanding
        sim.eps.density[5][3] = 0.8; // cap
        sim.eps.density[6][3] = 1.2; // cap peak
        sim.eps.density[7][3] = 0.6; // cap edge

        let count = sim.mushroom_structure_count();
        assert!(
            count >= 1,
            "Should detect at least 1 mushroom structure, got {}",
            count
        );
    }

    #[test]
    fn xorshift_deterministic() {
        // Two RNGs with the same seed must produce identical sequences
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn run_simulation_records_time_series() {
        let mut sim = BiofilmSimulator::new(15, 10, 99);
        sim.seed_cells(0, 20, 3.3e-4);
        sim.add_signal(QuorumSignal::ahl_default());

        let ts = sim.run_simulation(50, 1.0);

        assert_eq!(ts.times.len(), 50);
        assert_eq!(ts.biomass.len(), 50);
        assert_eq!(ts.eps_coverage.len(), 50);
        assert_eq!(ts.quorum_activation.len(), 50);

        // Time should be monotonically increasing
        for i in 1..ts.times.len() {
            assert!(
                ts.times[i] > ts.times[i - 1],
                "Time must increase monotonically"
            );
        }

        // Species counts should be tracked
        assert!(
            ts.species_counts.contains_key(&0),
            "Species 0 should be tracked"
        );
    }

    #[test]
    fn large_dt_diffusion_relaxation_stays_finite() {
        let mut sim = BiofilmSimulator::new(30, 20, 7);
        sim.seed_cells(0, 30, 3.3e-4);
        sim.add_signal(QuorumSignal::ahl_default());

        // One ecosystem-scale weekly step should not require millions of host
        // diffusion substeps or produce invalid field values.
        sim.step(7.0 * 86_400.0);

        let profile = sim.nutrient_profile();
        for row in &profile {
            for &value in row {
                assert!(value.is_finite(), "nutrient concentration must stay finite");
                assert!(
                    value >= 0.0,
                    "nutrient concentration must stay non-negative"
                );
            }
        }
    }
}
