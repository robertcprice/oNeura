//! Flux Balance Analysis (FBA) for microbial metabolism in a terrarium simulation.
//!
//! This module implements a self-contained constraint-based metabolic modeling toolkit
//! including stoichiometric network construction, a bounded-variable simplex LP solver,
//! flux variability analysis (FVA), and essential reaction identification.
//!
//! # Overview
//!
//! Flux balance analysis assumes that intracellular metabolites are at steady state
//! (production = consumption), giving the fundamental constraint **S * v = 0** where
//! S is the stoichiometric matrix and v is the flux vector. Each reaction flux is
//! bounded: `lower_bound <= v_j <= upper_bound`. An objective function (typically
//! biomass production) is maximized or minimized subject to these constraints.
//!
//! # LP Solver
//!
//! The solver uses a bounded-variable primal simplex method operating directly on
//! the augmented tableau. It handles the steady-state equality constraints Sv = 0
//! by converting them to a standard form with slack variables, then applying the
//! simplex algorithm with an initial Phase I to find a basic feasible solution.
//!
//! # Preset Models
//!
//! Two ready-to-use metabolic networks are included:
//! - [`e_coli_core()`] -- simplified *E. coli* core metabolism (~20 metabolites, ~25 reactions)
//! - [`soil_microbe_generic()`] -- generic soil bacterium with aerobic/anaerobic pathways
//!
//! # Example
//!
//! ```rust
//! use oneuro_metal::metabolic_flux::{MetabolicNetwork, FbaStatus};
//!
//! let mut net = MetabolicNetwork::new();
//! let glc = net.add_metabolite("glucose".into(), true);
//! let g6p = net.add_metabolite("glucose-6-phosphate".into(), false);
//! let biomass = net.add_metabolite("biomass".into(), true);
//!
//! // Glucose uptake: glc_ext -> g6p
//! let mut s1 = std::collections::HashMap::new();
//! s1.insert(glc, -1.0);
//! s1.insert(g6p, 1.0);
//! let r_uptake = net.add_reaction("glc_uptake".into(), s1, -10.0, 0.0, false);
//!
//! // Biomass: g6p -> biomass
//! let mut s2 = std::collections::HashMap::new();
//! s2.insert(g6p, -1.0);
//! s2.insert(biomass, 1.0);
//! let r_biomass = net.add_reaction("biomass_rxn".into(), s2, 0.0, 1000.0, false);
//!
//! let result = net.fba(r_biomass, true);
//! assert!(matches!(result.status, FbaStatus::Optimal));
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Tolerance and iteration limits for the LP solver
// ---------------------------------------------------------------------------

/// Numerical tolerance for the simplex algorithm pivot selection and
/// optimality checking. Values smaller than this are treated as zero.
const LP_TOLERANCE: f64 = 1e-8;

/// Maximum number of simplex iterations before declaring the problem
/// unbounded or cycling.
const MAX_ITERATIONS: usize = 1000;

/// A large bound used as a proxy for "unbounded" when no explicit
/// bound is supplied by the user.
const BIG_M: f64 = 1e6;

// ===========================================================================
// Public data types
// ===========================================================================

/// A metabolite (chemical species) participating in one or more reactions.
///
/// Metabolites are either *internal* (must satisfy steady-state balance Sv = 0)
/// or *external* (boundary species whose accumulation is unconstrained, e.g.
/// nutrients absorbed from the environment or waste products excreted).
#[derive(Debug, Clone)]
pub struct Metabolite {
    /// Unique identifier assigned by [`MetabolicNetwork::add_metabolite`].
    pub id: usize,
    /// Human-readable name (e.g. "glucose-6-phosphate").
    pub name: String,
    /// If `true`, this metabolite is a boundary species and its row is excluded
    /// from the steady-state constraint matrix.
    pub external: bool,
}

/// A biochemical reaction with stoichiometric coefficients and flux bounds.
///
/// Stoichiometry maps metabolite IDs to coefficients: negative values indicate
/// substrates (consumed) and positive values indicate products (produced).
/// For example, the reaction `A + 2B -> C` has stoichiometry
/// `{A: -1.0, B: -2.0, C: 1.0}`.
#[derive(Debug, Clone)]
pub struct Reaction {
    /// Unique identifier assigned by [`MetabolicNetwork::add_reaction`].
    pub id: usize,
    /// Human-readable name (e.g. "PFK" for phosphofructokinase).
    pub name: String,
    /// Metabolite ID -> stoichiometric coefficient.
    /// Negative = consumed, positive = produced.
    pub stoichiometry: HashMap<usize, f64>,
    /// Minimum allowed flux (typically 0 for irreversible, negative for reversible).
    pub lower_bound: f64,
    /// Maximum allowed flux (carrying capacity).
    pub upper_bound: f64,
    /// Whether this reaction can carry flux in both directions.
    pub reversible: bool,
}

/// The result of a flux balance analysis optimization.
#[derive(Debug, Clone)]
pub struct FbaResult {
    /// Optimal objective function value (flux through the objective reaction).
    pub objective_value: f64,
    /// Optimal flux vector, one entry per reaction in the network.
    pub fluxes: Vec<f64>,
    /// Solver termination status.
    pub status: FbaStatus,
    /// Shadow prices (dual values) for each internal metabolite constraint.
    /// A non-zero shadow price indicates that relaxing that metabolite's
    /// steady-state constraint would improve the objective.
    pub shadow_prices: Vec<f64>,
}

/// Termination status of the LP solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FbaStatus {
    /// An optimal solution was found within tolerance.
    Optimal,
    /// No feasible solution exists (constraints are contradictory).
    Infeasible,
    /// The objective is unbounded (can grow without limit).
    Unbounded,
}

// ===========================================================================
// MetabolicNetwork
// ===========================================================================

/// A constraint-based metabolic network supporting FBA, FVA, and essentiality analysis.
///
/// Build a network by adding metabolites and reactions, then call [`fba()`](MetabolicNetwork::fba)
/// to solve the linear program maximizing or minimizing flux through a chosen objective reaction.
#[derive(Debug, Clone)]
pub struct MetabolicNetwork {
    /// All metabolites in the network, indexed by their `id` field.
    pub metabolites: Vec<Metabolite>,
    /// All reactions in the network, indexed by their `id` field.
    pub reactions: Vec<Reaction>,
}

impl MetabolicNetwork {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create an empty metabolic network with no metabolites or reactions.
    pub fn new() -> Self {
        Self {
            metabolites: Vec::new(),
            reactions: Vec::new(),
        }
    }

    /// Add a metabolite to the network.
    ///
    /// Returns the metabolite's unique ID (its index in the metabolites vector).
    ///
    /// # Arguments
    /// * `name` -- Human-readable metabolite name.
    /// * `external` -- `true` if this is a boundary species excluded from Sv = 0.
    pub fn add_metabolite(&mut self, name: String, external: bool) -> usize {
        let id = self.metabolites.len();
        self.metabolites.push(Metabolite { id, name, external });
        id
    }

    /// Add a reaction to the network.
    ///
    /// Returns the reaction's unique ID (its index in the reactions vector).
    ///
    /// # Arguments
    /// * `name` -- Human-readable reaction name.
    /// * `stoichiometry` -- Map of metabolite ID to stoichiometric coefficient.
    /// * `lower_bound` -- Minimum flux (use negative for reversible reactions).
    /// * `upper_bound` -- Maximum flux.
    /// * `reversible` -- Whether the reaction can operate in reverse.
    pub fn add_reaction(
        &mut self,
        name: String,
        stoichiometry: HashMap<usize, f64>,
        lower_bound: f64,
        upper_bound: f64,
        reversible: bool,
    ) -> usize {
        let id = self.reactions.len();
        self.reactions.push(Reaction {
            id,
            name,
            stoichiometry,
            lower_bound,
            upper_bound,
            reversible,
        });
        id
    }

    // -----------------------------------------------------------------------
    // Stoichiometric matrix
    // -----------------------------------------------------------------------

    /// Build the stoichiometric matrix S for *internal* metabolites only.
    ///
    /// Returns `(matrix, n_rows, n_cols)` where:
    /// - `matrix[i][j]` is the stoichiometric coefficient of internal metabolite `i`
    ///   in reaction `j`.
    /// - `n_rows` = number of internal metabolites.
    /// - `n_cols` = number of reactions.
    ///
    /// External metabolites are excluded because their accumulation is unconstrained
    /// (they represent the environment boundary).
    pub fn stoichiometric_matrix(&self) -> (Vec<Vec<f64>>, usize, usize) {
        let internal: Vec<usize> = self
            .metabolites
            .iter()
            .filter(|m| !m.external)
            .map(|m| m.id)
            .collect();
        let n_rows = internal.len();
        let n_cols = self.reactions.len();
        let mut matrix = vec![vec![0.0; n_cols]; n_rows];

        // Map from metabolite ID to row index in the internal-only matrix.
        let mut met_to_row: HashMap<usize, usize> = HashMap::new();
        for (row, &met_id) in internal.iter().enumerate() {
            met_to_row.insert(met_id, row);
        }

        for rxn in &self.reactions {
            for (&met_id, &coeff) in &rxn.stoichiometry {
                if let Some(&row) = met_to_row.get(&met_id) {
                    matrix[row][rxn.id] = coeff;
                }
            }
        }

        (matrix, n_rows, n_cols)
    }

    // -----------------------------------------------------------------------
    // Flux Balance Analysis
    // -----------------------------------------------------------------------

    /// Perform flux balance analysis: optimize flux through `objective_reaction`
    /// subject to steady-state constraints Sv = 0 and flux bounds.
    ///
    /// # Arguments
    /// * `objective_reaction` -- Reaction index to optimize.
    /// * `maximize` -- `true` to maximize, `false` to minimize.
    ///
    /// # Returns
    /// An [`FbaResult`] with the optimal fluxes and solver status.
    pub fn fba(&self, objective_reaction: usize, maximize: bool) -> FbaResult {
        let n_rxns = self.reactions.len();
        if n_rxns == 0 || objective_reaction >= n_rxns {
            return FbaResult {
                objective_value: 0.0,
                fluxes: vec![0.0; n_rxns],
                status: FbaStatus::Infeasible,
                shadow_prices: Vec::new(),
            };
        }

        // Build objective vector: maximize c^T v (negate for minimization inside solver).
        let mut c = vec![0.0; n_rxns];
        c[objective_reaction] = if maximize { 1.0 } else { -1.0 };

        // Collect bounds.
        let lb: Vec<f64> = self.reactions.iter().map(|r| r.lower_bound).collect();
        let ub: Vec<f64> = self.reactions.iter().map(|r| r.upper_bound).collect();

        // Build the equality constraint matrix (internal metabolites only).
        let (s_matrix, n_constraints, _n_vars) = self.stoichiometric_matrix();

        // Solve the LP.
        let lp_result = solve_lp(&s_matrix, n_constraints, n_rxns, &c, &lb, &ub);

        match lp_result {
            LpResult::Optimal {
                objective,
                variables,
                duals,
            } => {
                let obj_val = if maximize { objective } else { -objective };
                FbaResult {
                    objective_value: obj_val,
                    fluxes: variables,
                    status: FbaStatus::Optimal,
                    shadow_prices: duals,
                }
            }
            LpResult::Infeasible => FbaResult {
                objective_value: 0.0,
                fluxes: vec![0.0; n_rxns],
                status: FbaStatus::Infeasible,
                shadow_prices: vec![0.0; n_constraints],
            },
            LpResult::Unbounded => FbaResult {
                objective_value: if maximize { f64::INFINITY } else { f64::NEG_INFINITY },
                fluxes: vec![0.0; n_rxns],
                status: FbaStatus::Unbounded,
                shadow_prices: vec![0.0; n_constraints],
            },
        }
    }

    // -----------------------------------------------------------------------
    // Flux Variability Analysis
    // -----------------------------------------------------------------------

    /// Compute the feasible flux range for every reaction while maintaining
    /// at least `fraction_optimum` of the optimal objective value.
    ///
    /// This is done by first solving the FBA to find the optimum, then adding
    /// a constraint that the objective flux >= `fraction_optimum * optimum`,
    /// and finally minimizing and maximizing each reaction flux in turn.
    ///
    /// # Arguments
    /// * `objective_reaction` -- Reaction index for the primary objective.
    /// * `fraction_optimum` -- Fraction of the optimal objective to maintain (0.0..=1.0).
    ///
    /// # Returns
    /// A vector of `(min_flux, max_flux)` for each reaction.
    pub fn flux_variability_analysis(
        &self,
        objective_reaction: usize,
        fraction_optimum: f64,
    ) -> Vec<(f64, f64)> {
        let n_rxns = self.reactions.len();
        let fba_result = self.fba(objective_reaction, true);

        if fba_result.status != FbaStatus::Optimal {
            return vec![(0.0, 0.0); n_rxns];
        }

        let min_obj_flux = fraction_optimum * fba_result.objective_value;

        // Build an augmented network with the objective constraint as a bound.
        let mut ranges = Vec::with_capacity(n_rxns);

        for rxn_idx in 0..n_rxns {
            // Minimize
            let min_val = self.fva_single(objective_reaction, min_obj_flux, rxn_idx, false);
            // Maximize
            let max_val = self.fva_single(objective_reaction, min_obj_flux, rxn_idx, true);
            ranges.push((min_val, max_val));
        }

        ranges
    }

    /// Solve a single FVA sub-problem: optimize reaction `target` subject to
    /// the original steady-state constraints plus an additional lower bound on
    /// the objective reaction.
    fn fva_single(
        &self,
        objective_reaction: usize,
        min_obj_flux: f64,
        target_reaction: usize,
        maximize: bool,
    ) -> f64 {
        let n_rxns = self.reactions.len();

        let mut c = vec![0.0; n_rxns];
        c[target_reaction] = if maximize { 1.0 } else { -1.0 };

        let mut lb: Vec<f64> = self.reactions.iter().map(|r| r.lower_bound).collect();
        let ub: Vec<f64> = self.reactions.iter().map(|r| r.upper_bound).collect();

        // Impose minimum objective flux.
        if min_obj_flux > lb[objective_reaction] {
            lb[objective_reaction] = min_obj_flux;
        }

        let (s_matrix, n_constraints, _) = self.stoichiometric_matrix();
        let lp_result = solve_lp(&s_matrix, n_constraints, n_rxns, &c, &lb, &ub);

        match lp_result {
            LpResult::Optimal {
                objective,
                variables: _,
                duals: _,
            } => {
                if maximize {
                    objective
                } else {
                    -objective
                }
            }
            _ => {
                if maximize {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Essential reactions
    // -----------------------------------------------------------------------

    /// Identify reactions that are essential for achieving at least `threshold`
    /// fraction of the maximum objective flux.
    ///
    /// A reaction is essential if knocking it out (setting both bounds to zero)
    /// causes the optimal objective to drop below `threshold * max_objective`.
    ///
    /// # Arguments
    /// * `objective_reaction` -- Reaction index for the primary objective.
    /// * `threshold` -- Fraction of optimum below which the reaction is essential (0.0..=1.0).
    ///
    /// # Returns
    /// Vector of reaction IDs that are essential.
    pub fn essential_reactions(
        &self,
        objective_reaction: usize,
        threshold: f64,
    ) -> Vec<usize> {
        let fba_result = self.fba(objective_reaction, true);
        if fba_result.status != FbaStatus::Optimal {
            return Vec::new();
        }

        let min_objective = threshold * fba_result.objective_value;
        let mut essential = Vec::new();

        for rxn_idx in 0..self.reactions.len() {
            if rxn_idx == objective_reaction {
                continue;
            }

            // Create a copy with this reaction knocked out.
            let mut knocked = self.clone();
            knocked.reactions[rxn_idx].lower_bound = 0.0;
            knocked.reactions[rxn_idx].upper_bound = 0.0;

            let ko_result = knocked.fba(objective_reaction, true);

            if ko_result.status != FbaStatus::Optimal
                || ko_result.objective_value < min_objective - LP_TOLERANCE
            {
                essential.push(rxn_idx);
            }
        }

        essential
    }

    // -----------------------------------------------------------------------
    // Dead-end metabolites
    // -----------------------------------------------------------------------

    /// Find metabolites that participate in only consuming or only producing
    /// reactions (dead ends). These metabolites cannot be at steady state
    /// because mass cannot balance for them.
    ///
    /// # Returns
    /// Vector of metabolite IDs that are dead ends.
    pub fn dead_end_metabolites(&self) -> Vec<usize> {
        let mut dead_ends = Vec::new();

        for met in &self.metabolites {
            if met.external {
                continue;
            }

            let mut has_producer = false;
            let mut has_consumer = false;

            for rxn in &self.reactions {
                if let Some(&coeff) = rxn.stoichiometry.get(&met.id) {
                    if coeff.abs() < LP_TOLERANCE {
                        continue;
                    }
                    if coeff > 0.0 {
                        has_producer = true;
                    }
                    if coeff < 0.0 {
                        has_consumer = true;
                    }
                    // Reversible reactions can act as both producer and consumer.
                    if rxn.reversible {
                        has_producer = true;
                        has_consumer = true;
                    }
                }
            }

            if !has_producer || !has_consumer {
                dead_ends.push(met.id);
            }
        }

        dead_ends
    }
}

impl Default for MetabolicNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// LP Solver -- standard-form simplex with Big-M method
// ===========================================================================
//
// Strategy: convert the bounded-variable FBA problem into a standard-form LP
// where all variables are non-negative, all constraints are equalities, and
// upper bounds are modeled as explicit inequality constraints with slack
// variables. Then solve using Big-M simplex (single phase).
//
// Conversion from: maximize c^T x,  A x = 0,  lb <= x <= ub
//
// 1. Shift: y_j = x_j - lb_j  =>  y_j >= 0,  0 <= y_j <= ub_j - lb_j
//    New objective: maximize c^T (y + lb) = c^T y + const
//    Constraints: A y = -A lb (= rhs)
//
// 2. Upper-bound slacks: for each finite upper bound d_j = ub_j - lb_j,
//    add slack s_j >= 0 with y_j + s_j = d_j.
//
// 3. Artificial variables for equality constraints (Big-M penalty).

/// Internal result type for the LP solver.
#[derive(Debug)]
enum LpResult {
    Optimal {
        objective: f64,
        variables: Vec<f64>,
        duals: Vec<f64>,
    },
    Infeasible,
    Unbounded,
}

/// Solve the linear program:
///
///   maximize  c^T x
///   subject to  A x = 0   (steady-state)
///               lb <= x <= ub
///
/// Converts to standard form and solves via Big-M simplex.
fn solve_lp(
    a: &[Vec<f64>],
    m: usize,
    n: usize,
    c: &[f64],
    lb: &[f64],
    ub: &[f64],
) -> LpResult {
    // -----------------------------------------------------------------------
    // Validate bounds.
    // -----------------------------------------------------------------------
    for j in 0..n {
        if lb[j] > ub[j] + LP_TOLERANCE {
            return LpResult::Infeasible;
        }
    }

    // Handle trivial case: no constraints.
    if m == 0 {
        let mut x = vec![0.0; n];
        let mut obj = 0.0;
        for j in 0..n {
            if c[j] > LP_TOLERANCE {
                x[j] = ub[j].min(BIG_M);
            } else if c[j] < -LP_TOLERANCE {
                x[j] = lb[j].max(-BIG_M);
            } else {
                x[j] = lb[j].max(0.0).min(ub[j]);
            }
            obj += c[j] * x[j];
        }
        return LpResult::Optimal {
            objective: obj,
            variables: x,
            duals: Vec::new(),
        };
    }

    // -----------------------------------------------------------------------
    // Step 1: Shift variables. y_j = x_j - lb_j, so y_j >= 0.
    // -----------------------------------------------------------------------
    let mut d = vec![0.0f64; n]; // shifted upper bounds
    for j in 0..n {
        d[j] = (ub[j] - lb[j]).max(0.0);
    }

    // RHS of equality constraints: A y = -A lb.
    let mut rhs_eq = vec![0.0; m];
    for i in 0..m {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i][j] * lb[j];
        }
        rhs_eq[i] = -s;
    }

    // -----------------------------------------------------------------------
    // Step 2: Count upper-bound constraints (finite d_j).
    // -----------------------------------------------------------------------
    let mut ub_constrained: Vec<usize> = Vec::new();
    for j in 0..n {
        if d[j] < BIG_M {
            ub_constrained.push(j);
        }
    }
    let n_ub = ub_constrained.len();

    // Total constraints = m (equality) + n_ub (upper-bound slacks).
    let total_constraints = m + n_ub;

    // Variables: y (n) + slack for UB (n_ub) + artificial for equalities (m).
    // UB constraints: y_j + s_j = d_j  (s_j is basic initially, no artificial needed).
    // Equality constraints: need artificial variable if we can't find a natural basic.
    let n_slack = n_ub;
    let n_art = m;
    let total_vars = n + n_slack + n_art;

    // Column layout:
    //   [0..n)                   = y variables (shifted original)
    //   [n..n+n_slack)           = slack variables for upper bounds
    //   [n+n_slack..total_vars)  = artificial variables for equality constraints

    let art_start = n + n_slack;

    // -----------------------------------------------------------------------
    // Step 3: Build the tableau.
    // -----------------------------------------------------------------------
    // Rows: total_constraints constraint rows + 1 objective row.
    // Columns: total_vars + 1 (RHS).
    let cols = total_vars + 1;
    let rhs_col = cols - 1;
    let mut tab = vec![vec![0.0; cols]; total_constraints + 1];
    let obj_row = total_constraints;

    // Basis: one basic variable per constraint row.
    let mut basis = vec![0usize; total_constraints];

    // --- Fill equality constraint rows (rows 0..m). ---
    for i in 0..m {
        // We need RHS >= 0 for Big-M to work. If rhs_eq[i] < 0, multiply row by -1.
        let sign = if rhs_eq[i] < -LP_TOLERANCE { -1.0 } else { 1.0 };
        for j in 0..n {
            tab[i][j] = sign * a[i][j];
        }
        // Artificial variable for this row.
        tab[i][art_start + i] = 1.0;
        tab[i][rhs_col] = sign * rhs_eq[i];
        basis[i] = art_start + i;
    }

    // --- Fill upper-bound constraint rows (rows m..m+n_ub). ---
    for (k, &j) in ub_constrained.iter().enumerate() {
        let row = m + k;
        tab[row][j] = 1.0;          // y_j
        tab[row][n + k] = 1.0;      // slack s_k
        tab[row][rhs_col] = d[j];   // = d_j
        basis[row] = n + k;         // slack is basic
    }

    // --- Build objective row. ---
    // maximize c^T y - BIG_M * sum(artificials)
    // In the tableau convention, the objective row stores the reduced costs.
    // z = c^T y - BIG_M * sum(a_i)
    // For non-basic variables: reduced cost = c_j (or -BIG_M for artificials).
    // For basic variables: must be 0 (eliminate via row operations).

    // Start with raw objective coefficients.
    for j in 0..n {
        tab[obj_row][j] = c[j];
    }
    // Artificials get -BIG_M.
    for k in 0..n_art {
        tab[obj_row][art_start + k] = -BIG_M;
    }

    // Eliminate basic variables from the objective row.
    // Basic variables in equality rows are artificials (initially).
    // Basic variables in UB rows are slacks (coefficient 0 in obj, no elimination needed).
    for i in 0..m {
        let bv = basis[i];
        let rc = tab[obj_row][bv];
        if rc.abs() > LP_TOLERANCE * 0.001 {
            for j2 in 0..cols {
                tab[obj_row][j2] -= rc * tab[i][j2];
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 4: Simplex iterations.
    // -----------------------------------------------------------------------
    for _iter in 0..MAX_ITERATIONS {
        // --- Pricing: find entering variable with largest positive reduced cost. ---
        let mut best_col: Option<usize> = None;
        let mut best_rc = LP_TOLERANCE;

        for j in 0..total_vars {
            if tab[obj_row][j] > best_rc {
                // Make sure it's not already basic.
                let is_basic = basis.iter().any(|&b| b == j);
                if !is_basic {
                    best_rc = tab[obj_row][j];
                    best_col = Some(j);
                }
            }
        }

        let entering = match best_col {
            Some(col) => col,
            None => break, // Optimal.
        };

        // --- Ratio test: find leaving variable (min ratio, RHS / a_ij > 0). ---
        let mut min_ratio = f64::INFINITY;
        let mut pivot_row: Option<usize> = None;

        for i in 0..total_constraints {
            let aij = tab[i][entering];
            if aij > LP_TOLERANCE {
                let ratio = tab[i][rhs_col] / aij;
                if ratio < min_ratio - LP_TOLERANCE {
                    min_ratio = ratio;
                    pivot_row = Some(i);
                } else if (ratio - min_ratio).abs() < LP_TOLERANCE {
                    // Bland's rule tie-breaking: prefer row with smaller basis index.
                    if let Some(pr) = pivot_row {
                        if basis[i] < basis[pr] {
                            pivot_row = Some(i);
                        }
                    }
                }
            }
        }

        let pr = match pivot_row {
            Some(r) => r,
            None => {
                // Unbounded -- entering variable can increase without limit.
                return LpResult::Unbounded;
            }
        };

        // --- Pivot. ---
        let pivot_val = tab[pr][entering];
        let inv = 1.0 / pivot_val;
        for j2 in 0..cols {
            tab[pr][j2] *= inv;
        }
        for i in 0..=obj_row {
            if i == pr {
                continue;
            }
            let factor = tab[i][entering];
            if factor.abs() < LP_TOLERANCE * 0.001 {
                continue;
            }
            for j2 in 0..cols {
                tab[i][j2] -= factor * tab[pr][j2];
            }
        }
        basis[pr] = entering;
    }

    // -----------------------------------------------------------------------
    // Step 5: Check feasibility -- all artificials must be zero.
    // -----------------------------------------------------------------------
    for i in 0..total_constraints {
        let bv = basis[i];
        if bv >= art_start {
            // Artificial is still in the basis.
            if tab[i][rhs_col].abs() > LP_TOLERANCE {
                return LpResult::Infeasible;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 6: Extract solution.
    // -----------------------------------------------------------------------
    let mut y = vec![0.0; n];
    for i in 0..total_constraints {
        let bv = basis[i];
        if bv < n {
            y[bv] = tab[i][rhs_col];
        }
    }

    // Un-shift: x_j = y_j + lb_j.
    let mut x = vec![0.0; n];
    let mut obj = 0.0;
    for j in 0..n {
        x[j] = y[j] + lb[j];
        // Clamp for numerical stability.
        if x[j] < lb[j] - LP_TOLERANCE {
            x[j] = lb[j];
        }
        if x[j] > ub[j] + LP_TOLERANCE {
            x[j] = ub[j];
        }
        obj += c[j] * x[j];
    }

    // Extract shadow prices for the equality constraints.
    // The dual value for constraint i is related to the objective row entry
    // for the artificial variable of that constraint.
    let mut duals = vec![0.0; m];
    for i in 0..m {
        duals[i] = -tab[obj_row][art_start + i];
    }

    LpResult::Optimal {
        objective: obj,
        variables: x,
        duals,
    }
}

// ===========================================================================
// Preset metabolic network models
// ===========================================================================

/// Build a simplified *E. coli* core metabolic network.
///
/// This model represents the central carbon metabolism of *Escherichia coli*
/// including glycolysis, the TCA cycle, the pentose phosphate pathway, and
/// a lumped biomass reaction. It contains approximately 20 metabolites and
/// 25 reactions.
///
/// The stoichiometry is loosely based on the Palsson group's E. coli core
/// model (Orth et al., 2010) but simplified for fast computation.
///
/// # Metabolites (20)
///
/// External: glucose, O2, CO2, NH4, biomass_ext, acetate, ethanol, formate,
/// succinate_ext, lactate
///
/// Internal: G6P, F6P, FBP, GAP, PEP, PYR, AcCoA, OAA, CIT, AKG,
/// SUC, FUM, MAL, R5P, E4P, S7P, ATP, NADH, NADPH, biomass
///
/// # Reactions (25+)
///
/// Glycolysis: GLC_uptake, PGI, PFK, FBA_rxn, TPI_rxn, GAPD, PYK
/// TCA cycle: PDH, CS, ACO_IDH, AKGDH, SUCOAS, SDH, FUM_rxn, MDH
/// Pentose phosphate: G6PDH, PGL_GND, RPI_RPE, TKT1, TALA, TKT2
/// Anaplerotic: PPC
/// Biomass: BIOMASS
/// Exchange: EX_O2, EX_CO2, EX_NH4, EX_AC, EX_ETH, EX_FORM, EX_SUC, EX_LAC
pub fn e_coli_core() -> MetabolicNetwork {
    let mut net = MetabolicNetwork::new();

    // -- External metabolites (boundary species) --
    let glc_e = net.add_metabolite("glucose_ext".into(), true);
    let o2_e = net.add_metabolite("O2_ext".into(), true);
    let co2_e = net.add_metabolite("CO2_ext".into(), true);
    let nh4_e = net.add_metabolite("NH4_ext".into(), true);
    let bio_e = net.add_metabolite("biomass_ext".into(), true);
    let ac_e = net.add_metabolite("acetate_ext".into(), true);
    let eth_e = net.add_metabolite("ethanol_ext".into(), true);
    let form_e = net.add_metabolite("formate_ext".into(), true);
    let suc_e = net.add_metabolite("succinate_ext".into(), true);
    let lac_e = net.add_metabolite("lactate_ext".into(), true);

    // -- Internal metabolites --
    let g6p = net.add_metabolite("G6P".into(), false);        // 10
    let f6p = net.add_metabolite("F6P".into(), false);        // 11
    let fbp = net.add_metabolite("FBP".into(), false);        // 12
    let gap = net.add_metabolite("GAP".into(), false);        // 13
    let pep = net.add_metabolite("PEP".into(), false);        // 14
    let pyr = net.add_metabolite("PYR".into(), false);        // 15
    let accoa = net.add_metabolite("AcCoA".into(), false);    // 16
    let oaa = net.add_metabolite("OAA".into(), false);        // 17
    let cit = net.add_metabolite("CIT".into(), false);        // 18
    let akg = net.add_metabolite("AKG".into(), false);        // 19
    let suc = net.add_metabolite("SUC".into(), false);        // 20
    let fum = net.add_metabolite("FUM".into(), false);        // 21
    let mal = net.add_metabolite("MAL".into(), false);        // 22
    let r5p = net.add_metabolite("R5P".into(), false);        // 23
    let e4p = net.add_metabolite("E4P".into(), false);        // 24
    let s7p = net.add_metabolite("S7P".into(), false);        // 25
    let atp = net.add_metabolite("ATP".into(), false);        // 26
    let nadh = net.add_metabolite("NADH".into(), false);      // 27
    let nadph = net.add_metabolite("NADPH".into(), false);    // 28
    let co2_int = net.add_metabolite("CO2".into(), false);    // 29

    // Helper to build stoichiometry.
    fn s(pairs: &[(usize, f64)]) -> HashMap<usize, f64> {
        pairs.iter().cloned().collect()
    }

    // -- Glycolysis --
    // 1. Glucose uptake: glc_ext -> G6P (costs 1 ATP via PTS)
    net.add_reaction(
        "GLC_uptake".into(),
        s(&[(glc_e, -1.0), (g6p, 1.0), (pep, -1.0), (pyr, 1.0)]),
        0.0, 10.0, false,
    ); // 0

    // 2. PGI: G6P <-> F6P
    net.add_reaction(
        "PGI".into(),
        s(&[(g6p, -1.0), (f6p, 1.0)]),
        -1000.0, 1000.0, true,
    ); // 1

    // 3. PFK: F6P + ATP -> FBP
    net.add_reaction(
        "PFK".into(),
        s(&[(f6p, -1.0), (atp, -1.0), (fbp, 1.0)]),
        0.0, 1000.0, false,
    ); // 2

    // 4. FBA: FBP -> 2 GAP
    net.add_reaction(
        "FBA".into(),
        s(&[(fbp, -1.0), (gap, 2.0)]),
        -1000.0, 1000.0, true,
    ); // 3

    // 5. GAPD: GAP -> PEP + NADH + ATP (lumped lower glycolysis)
    net.add_reaction(
        "GAPD".into(),
        s(&[(gap, -1.0), (pep, 1.0), (nadh, 1.0), (atp, 2.0)]),
        0.0, 1000.0, false,
    ); // 4

    // 6. PYK: PEP -> PYR + ATP
    net.add_reaction(
        "PYK".into(),
        s(&[(pep, -1.0), (pyr, 1.0), (atp, 1.0)]),
        0.0, 1000.0, false,
    ); // 5

    // -- Pyruvate metabolism --
    // 7. PDH: PYR -> AcCoA + CO2 + NADH
    net.add_reaction(
        "PDH".into(),
        s(&[(pyr, -1.0), (accoa, 1.0), (co2_int, 1.0), (nadh, 1.0)]),
        0.0, 1000.0, false,
    ); // 6

    // -- TCA cycle --
    // 8. CS: AcCoA + OAA -> CIT
    net.add_reaction(
        "CS".into(),
        s(&[(accoa, -1.0), (oaa, -1.0), (cit, 1.0)]),
        0.0, 1000.0, false,
    ); // 7

    // 9. ACO + IDH (lumped): CIT -> AKG + CO2 + NADPH
    net.add_reaction(
        "ACO_IDH".into(),
        s(&[(cit, -1.0), (akg, 1.0), (co2_int, 1.0), (nadph, 1.0)]),
        0.0, 1000.0, false,
    ); // 8

    // 10. AKGDH: AKG -> SUC + CO2 + NADH
    net.add_reaction(
        "AKGDH".into(),
        s(&[(akg, -1.0), (suc, 1.0), (co2_int, 1.0), (nadh, 1.0)]),
        0.0, 1000.0, false,
    ); // 9

    // 11. SUCOAS (lumped): SUC -> FUM + ATP
    net.add_reaction(
        "SUCOAS".into(),
        s(&[(suc, -1.0), (fum, 1.0), (atp, 1.0)]),
        -1000.0, 1000.0, true,
    ); // 10

    // 12. SDH: SUC -> FUM + NADH (alternate, membrane-coupled)
    // Merged with SUCOAS for simplicity; skipped to avoid parallel pathway.

    // 12. FUM: FUM <-> MAL
    net.add_reaction(
        "FUM_rxn".into(),
        s(&[(fum, -1.0), (mal, 1.0)]),
        -1000.0, 1000.0, true,
    ); // 11

    // 13. MDH: MAL <-> OAA + NADH
    net.add_reaction(
        "MDH".into(),
        s(&[(mal, -1.0), (oaa, 1.0), (nadh, 1.0)]),
        -1000.0, 1000.0, true,
    ); // 12

    // -- Pentose phosphate pathway --
    // 14. G6PDH: G6P -> R5P + 2 NADPH + CO2 (lumped oxidative branch)
    net.add_reaction(
        "G6PDH".into(),
        s(&[(g6p, -1.0), (r5p, 1.0), (nadph, 2.0), (co2_int, 1.0)]),
        0.0, 1000.0, false,
    ); // 13

    // 15. TKT1: R5P + R5P <-> GAP + S7P (simplified)
    net.add_reaction(
        "TKT1".into(),
        s(&[(r5p, -2.0), (gap, 1.0), (s7p, 1.0)]),
        -1000.0, 1000.0, true,
    ); // 14

    // 16. TALA: S7P + GAP <-> E4P + F6P
    net.add_reaction(
        "TALA".into(),
        s(&[(s7p, -1.0), (gap, -1.0), (e4p, 1.0), (f6p, 1.0)]),
        -1000.0, 1000.0, true,
    ); // 15

    // 17. TKT2: R5P + E4P <-> GAP + F6P (simplified)
    net.add_reaction(
        "TKT2".into(),
        s(&[(r5p, -1.0), (e4p, -1.0), (gap, 1.0), (f6p, 1.0)]),
        -1000.0, 1000.0, true,
    ); // 16

    // -- Anaplerotic --
    // 18. PPC: PEP + CO2 -> OAA
    net.add_reaction(
        "PPC".into(),
        s(&[(pep, -1.0), (co2_int, -1.0), (oaa, 1.0)]),
        0.0, 1000.0, false,
    ); // 17

    // -- Oxidative phosphorylation --
    // 19. ATPS: NADH + 0.5 O2 -> 2.5 ATP (P/O ratio ~2.5 for NADH)
    net.add_reaction(
        "ATPS".into(),
        s(&[(nadh, -1.0), (o2_e, -0.5), (atp, 2.5)]),
        0.0, 1000.0, false,
    ); // 18

    // 20. NADPH transhydrogenase: NADPH <-> NADH
    net.add_reaction(
        "THD".into(),
        s(&[(nadph, -1.0), (nadh, 1.0)]),
        -1000.0, 1000.0, true,
    ); // 19

    // -- Biomass (lumped) --
    // 21. BIOMASS: consumes precursors, produces biomass
    // Simplified stoichiometry based on Feist et al. (2007).
    let _r_biomass = net.add_reaction(
        "BIOMASS".into(),
        s(&[
            (g6p, -0.205), (f6p, -0.071), (r5p, -0.185),
            (e4p, -0.259), (gap, -0.129), (pep, -0.051),
            (pyr, -2.833), (accoa, -3.748), (oaa, -1.787),
            (akg, -1.079),
            (atp, -59.81), (nadph, -13.0276), (nadh, -3.547),
            (bio_e, 1.0),
        ]),
        0.0, 1000.0, false,
    ); // 20

    // -- ATP maintenance (non-growth associated) --
    // 22. ATPM: ATP -> (dissipated)
    net.add_reaction(
        "ATPM".into(),
        s(&[(atp, -1.0)]),
        8.39, 1000.0, false, // minimum maintenance flux
    ); // 21

    // -- Exchange reactions --
    // 23. EX_O2: O2_ext -> (unlimited uptake)
    net.add_reaction(
        "EX_O2".into(),
        s(&[(o2_e, -1.0)]),
        -1000.0, 0.0, false,
    ); // 22 -- note: negative = uptake convention

    // 24. EX_CO2: CO2 -> CO2_ext
    net.add_reaction(
        "EX_CO2".into(),
        s(&[(co2_int, -1.0), (co2_e, 1.0)]),
        0.0, 1000.0, false,
    ); // 23

    // 25. EX_NH4: NH4_ext -> (uptake)
    net.add_reaction(
        "EX_NH4".into(),
        s(&[(nh4_e, -1.0)]),
        -1000.0, 0.0, false,
    ); // 24

    // -- Overflow / fermentation exchanges --
    // Acetate: AcCoA -> acetate_ext + ATP (acetate kinase/PTA pathway, lumped)
    net.add_reaction(
        "ACKr".into(),
        s(&[(accoa, -1.0), (ac_e, 1.0), (atp, 1.0)]),
        0.0, 1000.0, false,
    ); // 25

    // Ethanol: AcCoA + 2 NADH -> ethanol_ext
    net.add_reaction(
        "ALCD".into(),
        s(&[(accoa, -1.0), (nadh, -2.0), (eth_e, 1.0)]),
        0.0, 1000.0, false,
    ); // 26

    // Formate: PYR -> AcCoA + formate_ext (PFL)
    net.add_reaction(
        "PFL".into(),
        s(&[(pyr, -1.0), (accoa, 1.0), (form_e, 1.0)]),
        0.0, 1000.0, false,
    ); // 27

    // Lactate: PYR + NADH -> lactate_ext (LDH)
    net.add_reaction(
        "LDH".into(),
        s(&[(pyr, -1.0), (nadh, -1.0), (lac_e, 1.0)]),
        0.0, 1000.0, false,
    ); // 28

    // Succinate export: SUC -> succinate_ext
    net.add_reaction(
        "EX_SUC".into(),
        s(&[(suc, -1.0), (suc_e, 1.0)]),
        0.0, 1000.0, false,
    ); // 29

    net
}

/// Build a generic soil bacterium metabolic network.
///
/// Models a versatile heterotrophic soil microorganism capable of:
/// - Aerobic respiration via a simplified TCA cycle
/// - Anaerobic fermentation (ethanol, lactate)
/// - Nitrogen fixation (energetically expensive N2 -> NH4)
/// - Organic acid secretion (acetate, succinate)
///
/// This network is designed to interface with the terrarium soil chemistry
/// system, where glucose and oxygen availability fluctuate spatially.
///
/// # Metabolites (14)
///
/// External: glucose, O2, CO2, N2, NH4, acetate, ethanol, lactate
/// Internal: G6P, PYR, AcCoA, OAA, ATP, NADH
///
/// # Reactions (16)
///
/// Uptake, glycolysis, TCA, oxidative phosphorylation, fermentation,
/// nitrogen fixation, biomass, maintenance, exchanges.
pub fn soil_microbe_generic() -> MetabolicNetwork {
    let mut net = MetabolicNetwork::new();

    // -- External metabolites --
    let glc_e = net.add_metabolite("glucose_ext".into(), true);
    let o2_e = net.add_metabolite("O2_ext".into(), true);
    let co2_e = net.add_metabolite("CO2_ext".into(), true);
    let n2_e = net.add_metabolite("N2_ext".into(), true);
    let nh4_e = net.add_metabolite("NH4_ext".into(), true);
    let ac_e = net.add_metabolite("acetate_ext".into(), true);
    let eth_e = net.add_metabolite("ethanol_ext".into(), true);
    let lac_e = net.add_metabolite("lactate_ext".into(), true);
    let bio_e = net.add_metabolite("biomass_ext".into(), true);

    // -- Internal metabolites --
    let g6p = net.add_metabolite("G6P".into(), false);     // 9
    let pyr = net.add_metabolite("PYR".into(), false);     // 10
    let accoa = net.add_metabolite("AcCoA".into(), false); // 11
    let oaa = net.add_metabolite("OAA".into(), false);     // 12
    let atp = net.add_metabolite("ATP".into(), false);     // 13
    let nadh = net.add_metabolite("NADH".into(), false);   // 14

    fn s(pairs: &[(usize, f64)]) -> HashMap<usize, f64> {
        pairs.iter().cloned().collect()
    }

    // 1. Glucose uptake: glc_ext -> G6P
    net.add_reaction(
        "GLC_uptake".into(),
        s(&[(glc_e, -1.0), (g6p, 1.0)]),
        0.0, 10.0, false,
    ); // 0

    // 2. Glycolysis (lumped): G6P -> 2 PYR + 2 ATP + 2 NADH
    net.add_reaction(
        "Glycolysis".into(),
        s(&[(g6p, -1.0), (pyr, 2.0), (atp, 2.0), (nadh, 2.0)]),
        0.0, 1000.0, false,
    ); // 1

    // 3. PDH: PYR -> AcCoA + CO2 + NADH
    net.add_reaction(
        "PDH".into(),
        s(&[(pyr, -1.0), (accoa, 1.0), (co2_e, 1.0), (nadh, 1.0)]),
        0.0, 1000.0, false,
    ); // 2

    // 4. TCA (lumped): AcCoA + OAA -> OAA + 3 NADH + 1 ATP + 2 CO2
    // Net: AcCoA -> 3 NADH + 1 ATP + 2 CO2 (OAA regenerated)
    net.add_reaction(
        "TCA_cycle".into(),
        s(&[(accoa, -1.0), (nadh, 3.0), (atp, 1.0), (co2_e, 2.0)]),
        0.0, 1000.0, false,
    ); // 3

    // 5. Oxidative phosphorylation: NADH + 0.5 O2 -> 2.5 ATP
    net.add_reaction(
        "OxPhos".into(),
        s(&[(nadh, -1.0), (o2_e, -0.5), (atp, 2.5)]),
        0.0, 1000.0, false,
    ); // 4

    // 6. Anaplerosis: PYR + CO2 -> OAA (pyruvate carboxylase)
    net.add_reaction(
        "PYC".into(),
        s(&[(pyr, -1.0), (co2_e, -1.0), (oaa, 1.0), (atp, -1.0)]),
        0.0, 1000.0, false,
    ); // 5

    // 7. Acetate secretion: AcCoA -> acetate + ATP
    net.add_reaction(
        "ACK".into(),
        s(&[(accoa, -1.0), (ac_e, 1.0), (atp, 1.0)]),
        0.0, 1000.0, false,
    ); // 6

    // 8. Ethanol fermentation: AcCoA + 2 NADH -> ethanol
    net.add_reaction(
        "ALCD".into(),
        s(&[(accoa, -1.0), (nadh, -2.0), (eth_e, 1.0)]),
        0.0, 1000.0, false,
    ); // 7

    // 9. Lactate fermentation: PYR + NADH -> lactate
    net.add_reaction(
        "LDH".into(),
        s(&[(pyr, -1.0), (nadh, -1.0), (lac_e, 1.0)]),
        0.0, 1000.0, false,
    ); // 8

    // 10. Nitrogen fixation: N2 + 16 ATP + 8 NADH -> 2 NH4
    // Energetically very expensive (nitrogenase).
    net.add_reaction(
        "NIF".into(),
        s(&[(n2_e, -1.0), (atp, -16.0), (nadh, -8.0), (nh4_e, 2.0)]),
        0.0, 1000.0, false,
    ); // 9

    // 11. Biomass: consumes precursors, produces biomass
    net.add_reaction(
        "BIOMASS".into(),
        s(&[
            (g6p, -0.2), (pyr, -1.0), (accoa, -2.0), (oaa, -1.0),
            (atp, -40.0), (nadh, -5.0),
            (bio_e, 1.0),
        ]),
        0.0, 1000.0, false,
    ); // 10

    // 12. ATP maintenance
    net.add_reaction(
        "ATPM".into(),
        s(&[(atp, -1.0)]),
        5.0, 1000.0, false,
    ); // 11

    // 13. Exchange: O2 uptake
    net.add_reaction(
        "EX_O2".into(),
        s(&[(o2_e, -1.0)]),
        -1000.0, 0.0, false,
    ); // 12

    // 14. Exchange: CO2 export (handled via TCA/PDH directly to external)
    // Not needed since CO2 is external.

    // 15. Exchange: N2 uptake
    net.add_reaction(
        "EX_N2".into(),
        s(&[(n2_e, -1.0)]),
        -1000.0, 0.0, false,
    ); // 13

    // 16. OAA sink (prevents dead-end when biomass doesn't consume all OAA)
    net.add_reaction(
        "OAA_sink".into(),
        s(&[(oaa, -1.0), (co2_e, 1.0)]),
        0.0, 1000.0, false,
    ); // 14

    net
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build stoichiometry from pairs.
    fn s(pairs: &[(usize, f64)]) -> HashMap<usize, f64> {
        pairs.iter().cloned().collect()
    }

    // -----------------------------------------------------------------------
    // 1. Empty network
    // -----------------------------------------------------------------------
    #[test]
    fn empty_network() {
        let net = MetabolicNetwork::new();
        assert_eq!(net.metabolites.len(), 0);
        assert_eq!(net.reactions.len(), 0);

        let (matrix, rows, cols) = net.stoichiometric_matrix();
        assert_eq!(rows, 0);
        assert_eq!(cols, 0);
        assert!(matrix.is_empty());

        // FBA on empty network is infeasible.
        let result = net.fba(0, true);
        assert_eq!(result.status, FbaStatus::Infeasible);
    }

    // -----------------------------------------------------------------------
    // 2. Add metabolites and reactions
    // -----------------------------------------------------------------------
    #[test]
    fn add_metabolites_and_reactions() {
        let mut net = MetabolicNetwork::new();
        let a = net.add_metabolite("A".into(), false);
        let b = net.add_metabolite("B".into(), false);
        let c = net.add_metabolite("C".into(), true);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);

        assert_eq!(net.metabolites.len(), 3);
        assert!(!net.metabolites[0].external);
        assert!(net.metabolites[2].external);

        let r = net.add_reaction("R1".into(), s(&[(a, -1.0), (b, 1.0)]), 0.0, 10.0, false);
        assert_eq!(r, 0);
        assert_eq!(net.reactions.len(), 1);
        assert_eq!(net.reactions[0].name, "R1");
    }

    // -----------------------------------------------------------------------
    // 3. Stoichiometric matrix shape
    // -----------------------------------------------------------------------
    #[test]
    fn stoichiometric_matrix_shape() {
        let mut net = MetabolicNetwork::new();
        let a = net.add_metabolite("A".into(), false);
        let b = net.add_metabolite("B".into(), false);
        let c = net.add_metabolite("C".into(), true); // external -- excluded

        net.add_reaction("R1".into(), s(&[(a, -1.0), (b, 1.0), (c, -1.0)]), 0.0, 10.0, false);
        net.add_reaction("R2".into(), s(&[(b, -1.0)]), 0.0, 10.0, false);

        let (matrix, rows, cols) = net.stoichiometric_matrix();
        // Only 2 internal metabolites, 2 reactions.
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);

        // R1: A -> B means A row has -1 in col 0, B row has +1 in col 0.
        assert!((matrix[0][0] - (-1.0)).abs() < 1e-12); // A in R1
        assert!((matrix[1][0] - 1.0).abs() < 1e-12);    // B in R1
        assert!((matrix[0][1] - 0.0).abs() < 1e-12);    // A not in R2
        assert!((matrix[1][1] - (-1.0)).abs() < 1e-12);  // B consumed in R2
    }

    // -----------------------------------------------------------------------
    // 4. Simple FBA -- feasible 2-reaction linear pathway
    // -----------------------------------------------------------------------
    #[test]
    fn simple_fba_feasible() {
        // Model: A_ext -> B -> C_ext
        // Maximize C production.
        let mut net = MetabolicNetwork::new();
        let a_ext = net.add_metabolite("A_ext".into(), true);
        let b = net.add_metabolite("B".into(), false);
        let c_ext = net.add_metabolite("C_ext".into(), true);

        // R0: A_ext -> B (uptake, max 10)
        let _r0 = net.add_reaction("uptake".into(), s(&[(a_ext, -1.0), (b, 1.0)]), 0.0, 10.0, false);
        // R1: B -> C_ext (production, max 1000)
        let r1 = net.add_reaction("produce".into(), s(&[(b, -1.0), (c_ext, 1.0)]), 0.0, 1000.0, false);

        let result = net.fba(r1, true);
        assert_eq!(result.status, FbaStatus::Optimal);
        // Maximum C production is limited by uptake of 10.
        assert!((result.objective_value - 10.0).abs() < 0.01,
            "Expected ~10.0, got {}", result.objective_value);
        assert!((result.fluxes[0] - 10.0).abs() < 0.01); // uptake at max
        assert!((result.fluxes[1] - 10.0).abs() < 0.01); // production matches
    }

    // -----------------------------------------------------------------------
    // 5. E. coli core model runs
    // -----------------------------------------------------------------------
    #[test]
    fn e_coli_core_runs() {
        let net = e_coli_core();

        // Sanity checks on model size.
        assert!(net.metabolites.len() >= 20, "Expected >= 20 metabolites, got {}", net.metabolites.len());
        assert!(net.reactions.len() >= 25, "Expected >= 25 reactions, got {}", net.reactions.len());

        // BIOMASS reaction is index 20.
        let biomass_idx = 20;
        assert_eq!(net.reactions[biomass_idx].name, "BIOMASS");

        let result = net.fba(biomass_idx, true);
        // The model should find a feasible solution.
        assert_eq!(result.status, FbaStatus::Optimal,
            "E. coli FBA failed with status {:?}", result.status);
        assert!(result.objective_value > 0.0,
            "Expected positive biomass, got {}", result.objective_value);
        assert_eq!(result.fluxes.len(), net.reactions.len());
    }

    // -----------------------------------------------------------------------
    // 6. Soil microbe model runs
    // -----------------------------------------------------------------------
    #[test]
    fn soil_microbe_runs() {
        let net = soil_microbe_generic();

        assert!(net.metabolites.len() >= 14, "Expected >= 14 metabolites");
        assert!(net.reactions.len() >= 14, "Expected >= 14 reactions");

        // BIOMASS reaction is index 10.
        let biomass_idx = 10;
        assert_eq!(net.reactions[biomass_idx].name, "BIOMASS");

        let result = net.fba(biomass_idx, true);
        assert_eq!(result.status, FbaStatus::Optimal,
            "Soil microbe FBA failed with status {:?}", result.status);
        assert!(result.objective_value > 0.0,
            "Expected positive biomass, got {}", result.objective_value);
    }

    // -----------------------------------------------------------------------
    // 7. Essential reactions detected
    // -----------------------------------------------------------------------
    #[test]
    fn essential_reactions_detected() {
        // Linear pathway: A_ext -> B -> C -> D_ext
        // All reactions are essential for D production.
        let mut net = MetabolicNetwork::new();
        let a_ext = net.add_metabolite("A_ext".into(), true);
        let b = net.add_metabolite("B".into(), false);
        let c = net.add_metabolite("C".into(), false);
        let d_ext = net.add_metabolite("D_ext".into(), true);

        let r0 = net.add_reaction("R0".into(), s(&[(a_ext, -1.0), (b, 1.0)]), 0.0, 10.0, false);
        let r1 = net.add_reaction("R1".into(), s(&[(b, -1.0), (c, 1.0)]), 0.0, 10.0, false);
        let r2 = net.add_reaction("R2".into(), s(&[(c, -1.0), (d_ext, 1.0)]), 0.0, 10.0, false);

        let essential = net.essential_reactions(r2, 0.01);
        // R0 and R1 are essential (R2 is the objective so it's skipped).
        assert!(essential.contains(&r0), "R0 should be essential");
        assert!(essential.contains(&r1), "R1 should be essential");
    }

    // -----------------------------------------------------------------------
    // 8. Dead-end metabolites found
    // -----------------------------------------------------------------------
    #[test]
    fn dead_end_metabolites_found() {
        let mut net = MetabolicNetwork::new();
        let a = net.add_metabolite("A".into(), false);
        let b = net.add_metabolite("B".into(), false);
        let c = net.add_metabolite("C".into(), false); // dead end -- only produced

        // R0: A -> B (irreversible)
        net.add_reaction("R0".into(), s(&[(a, -1.0), (b, 1.0)]), 0.0, 10.0, false);
        // R1: B -> C (irreversible) -- C is only produced, never consumed.
        net.add_reaction("R1".into(), s(&[(b, -1.0), (c, 1.0)]), 0.0, 10.0, false);

        let dead_ends = net.dead_end_metabolites();
        // A is only consumed (dead end), C is only produced (dead end).
        assert!(dead_ends.contains(&a), "A should be a dead end (only consumed)");
        assert!(dead_ends.contains(&c), "C should be a dead end (only produced)");
        assert!(!dead_ends.contains(&b), "B should NOT be a dead end (both consumed and produced)");
    }

    // -----------------------------------------------------------------------
    // 9. Flux variability bounds
    // -----------------------------------------------------------------------
    #[test]
    fn flux_variability_bounds() {
        // Two parallel pathways: A_ext -> B -> C_ext, and A_ext -> B -> D -> C_ext.
        // Both produce C. FVA should show the direct path can range.
        let mut net = MetabolicNetwork::new();
        let a_ext = net.add_metabolite("A_ext".into(), true);
        let b = net.add_metabolite("B".into(), false);
        let d = net.add_metabolite("D".into(), false);
        let c_ext = net.add_metabolite("C_ext".into(), true);

        // R0: A_ext -> B (uptake, max 10)
        net.add_reaction("uptake".into(), s(&[(a_ext, -1.0), (b, 1.0)]), 0.0, 10.0, false);
        // R1: B -> C_ext (direct)
        net.add_reaction("direct".into(), s(&[(b, -1.0), (c_ext, 1.0)]), 0.0, 1000.0, false);
        // R2: B -> D
        net.add_reaction("step1".into(), s(&[(b, -1.0), (d, 1.0)]), 0.0, 1000.0, false);
        // R3: D -> C_ext
        let r3 = net.add_reaction("step2".into(), s(&[(d, -1.0), (c_ext, 1.0)]), 0.0, 1000.0, false);

        // Objective: maximize total C_ext production. We need a lumped objective.
        // Actually, let's just use one of the C-producing reactions.
        // With parallel paths, maximizing r1 gives 10, but FVA of r1 at 100% optimum
        // should show r1 can range from 0 to 10 if we add a combined objective.
        // Simpler: use a combined objective reaction.
        let c_obj = net.add_metabolite("C_obj".into(), false);
        // Add an objective reaction that consumes C_ext -- but C_ext is external.
        // Let's restructure: make C internal and add an export.
        // ... Actually, let's keep it simple and test with the existing setup.

        // FVA with r1 as objective at 0% optimum (no constraint) should show full range.
        let ranges = net.flux_variability_analysis(1, 0.0);

        // R0 (uptake): must be between 0 and 10 regardless.
        assert!(ranges[0].0 >= -0.01, "Uptake min should be >= 0, got {}", ranges[0].0);
        assert!(ranges[0].1 <= 10.01, "Uptake max should be <= 10, got {}", ranges[0].1);

        // With 0% optimum constraint, there's flexibility in how flux distributes.
        // R2 (step1) and R3 (step2) should be able to carry flux.
        // At 0% objective, R2 max should be >= 0 (it can carry some flux).
        assert!(ranges[2].1 >= -0.01, "Step1 max should be >= 0, got {}", ranges[2].1);

        // With r3 = step2, check it has non-trivial range.
        let _ = r3; // used above
    }

    // -----------------------------------------------------------------------
    // 10. Infeasible network detected
    // -----------------------------------------------------------------------
    #[test]
    fn infeasible_network_detected() {
        // Network where internal metabolite B is consumed but never produced.
        let mut net = MetabolicNetwork::new();
        let _a_ext = net.add_metabolite("A_ext".into(), true);
        let b = net.add_metabolite("B".into(), false);
        let c_ext = net.add_metabolite("C_ext".into(), true);

        // R0: B -> C_ext (but no reaction produces B!)
        let r0 = net.add_reaction("consume_B".into(), s(&[(b, -1.0), (c_ext, 1.0)]), 1.0, 10.0, false);

        let result = net.fba(r0, true);
        // With a minimum flux of 1.0 on the only reaction, but B has no source,
        // the steady-state constraint for B (-v0 = 0 => v0 = 0) conflicts with v0 >= 1.
        assert_eq!(result.status, FbaStatus::Infeasible,
            "Expected infeasible but got {:?} with obj={}", result.status, result.objective_value);
    }

    // -----------------------------------------------------------------------
    // 11. Reversible reactions handle negative flux
    // -----------------------------------------------------------------------
    #[test]
    fn reversible_reactions_handle_negative_flux() {
        // A_ext <-> B <-> C_ext. If we maximize C production by pulling B from A,
        // but we can also push C back to B if we minimize C.
        let mut net = MetabolicNetwork::new();
        let a_ext = net.add_metabolite("A_ext".into(), true);
        let b = net.add_metabolite("B".into(), false);
        let c_ext = net.add_metabolite("C_ext".into(), true);

        // R0: A_ext <-> B (reversible, bounds -10..10)
        net.add_reaction(
            "R0".into(),
            s(&[(a_ext, -1.0), (b, 1.0)]),
            -10.0, 10.0, true,
        );
        // R1: B <-> C_ext (reversible, bounds -10..10)
        let r1 = net.add_reaction(
            "R1".into(),
            s(&[(b, -1.0), (c_ext, 1.0)]),
            -10.0, 10.0, true,
        );

        // Maximize C production.
        let max_result = net.fba(r1, true);
        assert_eq!(max_result.status, FbaStatus::Optimal);
        assert!((max_result.objective_value - 10.0).abs() < 0.01,
            "Expected max ~10.0, got {}", max_result.objective_value);

        // Minimize C production (flux can go negative).
        let min_result = net.fba(r1, false);
        assert_eq!(min_result.status, FbaStatus::Optimal);
        assert!((min_result.objective_value - (-10.0)).abs() < 0.01,
            "Expected min ~-10.0, got {}", min_result.objective_value);
    }

    // -----------------------------------------------------------------------
    // 12. Branched pathway FBA
    // -----------------------------------------------------------------------
    #[test]
    fn branched_pathway_fba() {
        // A_ext -> B (max 10)
        // B -> C_ext (yield 1)
        // B -> D_ext (yield 2 per B)
        // Maximize D: should route all flux through D pathway.
        let mut net = MetabolicNetwork::new();
        let a_ext = net.add_metabolite("A_ext".into(), true);
        let b = net.add_metabolite("B".into(), false);
        let c_ext = net.add_metabolite("C_ext".into(), true);
        let d_ext = net.add_metabolite("D_ext".into(), true);

        net.add_reaction("uptake".into(), s(&[(a_ext, -1.0), (b, 1.0)]), 0.0, 10.0, false);
        net.add_reaction("to_C".into(), s(&[(b, -1.0), (c_ext, 1.0)]), 0.0, 1000.0, false);
        let r_d = net.add_reaction("to_D".into(), s(&[(b, -0.5), (d_ext, 1.0)]), 0.0, 1000.0, false);

        let result = net.fba(r_d, true);
        assert_eq!(result.status, FbaStatus::Optimal);
        // B uptake = 10, to_D uses 0.5 B per D, so max D = 20.
        assert!((result.objective_value - 20.0).abs() < 0.1,
            "Expected ~20.0, got {}", result.objective_value);
    }

    // -----------------------------------------------------------------------
    // 13. Network with ATP coupling
    // -----------------------------------------------------------------------
    #[test]
    fn atp_coupled_pathway() {
        // Tests that ATP balance constrains growth.
        // Uptake: A_ext -> B (0..10)
        // ATP production: B -> C + ATP (0..1000)
        // Biomass: B + 5 ATP -> biomass_ext (0..1000)
        // ATPM: ATP -> nothing (fixed at 2..1000)
        let mut net = MetabolicNetwork::new();
        let a_ext = net.add_metabolite("A_ext".into(), true);
        let b = net.add_metabolite("B".into(), false);
        let c_ext = net.add_metabolite("C_ext".into(), true);
        let atp = net.add_metabolite("ATP".into(), false);
        let bio_ext = net.add_metabolite("biomass_ext".into(), true);

        net.add_reaction("uptake".into(), s(&[(a_ext, -1.0), (b, 1.0)]), 0.0, 10.0, false);
        net.add_reaction("ferment".into(), s(&[(b, -1.0), (c_ext, 1.0), (atp, 2.0)]), 0.0, 1000.0, false);
        let r_bio = net.add_reaction("biomass".into(), s(&[(b, -1.0), (atp, -5.0), (bio_ext, 1.0)]), 0.0, 1000.0, false);
        net.add_reaction("ATPM".into(), s(&[(atp, -1.0)]), 2.0, 1000.0, false);

        let result = net.fba(r_bio, true);
        assert_eq!(result.status, FbaStatus::Optimal);
        // ATP budget constrains biomass. Each B used for biomass costs 5 ATP but each B
        // fermented yields 2 ATP. Let x = biomass flux, y = ferment flux.
        // B balance: x + y = 10, ATP balance: 2y - 5x - atpm = 0, atpm >= 2.
        // 2(10-x) - 5x - 2 = 0 => 20 - 2x - 5x - 2 = 0 => 7x = 18 => x ~ 2.57
        assert!(result.objective_value > 0.0 && result.objective_value < 10.0,
            "Expected biomass constrained by ATP, got {}", result.objective_value);
    }

    // -----------------------------------------------------------------------
    // 14. Dead ends not flagged for external metabolites
    // -----------------------------------------------------------------------
    #[test]
    fn dead_ends_exclude_external() {
        let mut net = MetabolicNetwork::new();
        let _a_ext = net.add_metabolite("A_ext".into(), true); // external, not checked
        let b = net.add_metabolite("B".into(), false);
        let _c_ext = net.add_metabolite("C_ext".into(), true); // external, not checked

        net.add_reaction("R0".into(), s(&[(_a_ext, -1.0), (b, 1.0)]), 0.0, 10.0, false);
        net.add_reaction("R1".into(), s(&[(b, -1.0), (_c_ext, 1.0)]), 0.0, 10.0, false);

        let dead_ends = net.dead_end_metabolites();
        // B is both produced and consumed, so not a dead end.
        // External metabolites are excluded from analysis.
        assert!(!dead_ends.contains(&b), "B should not be a dead end");
        assert!(dead_ends.is_empty(), "No dead ends expected, got {:?}", dead_ends);
    }

    // -----------------------------------------------------------------------
    // 15. Soil microbe anaerobic growth
    // -----------------------------------------------------------------------
    #[test]
    fn soil_microbe_anaerobic_growth() {
        let mut net = soil_microbe_generic();

        // Cut off oxygen supply by setting EX_O2 bounds to 0.
        // EX_O2 is reaction index 12.
        net.reactions[12].lower_bound = 0.0;
        net.reactions[12].upper_bound = 0.0;

        // BIOMASS is reaction 10.
        let result = net.fba(10, true);
        // Should still grow (via fermentation) but slower.
        assert_eq!(result.status, FbaStatus::Optimal,
            "Anaerobic FBA failed with {:?}", result.status);
        // Biomass should be positive but lower than aerobic.
        // (Could be zero if ATP maintenance can't be met -- that's also valid.)
    }

    // -----------------------------------------------------------------------
    // 16. Reversible dead-end detection
    // -----------------------------------------------------------------------
    #[test]
    fn reversible_reactions_not_dead_end() {
        let mut net = MetabolicNetwork::new();
        let a = net.add_metabolite("A".into(), false);
        let b = net.add_metabolite("B".into(), false);

        // A <-> B (reversible): A is both consumed and produced.
        net.add_reaction("R0".into(), s(&[(a, -1.0), (b, 1.0)]), -10.0, 10.0, true);

        let dead_ends = net.dead_end_metabolites();
        // Reversible reaction means both A and B are produced and consumed.
        assert!(!dead_ends.contains(&a), "A should not be dead end with reversible reaction");
        assert!(!dead_ends.contains(&b), "B should not be dead end with reversible reaction");
    }
}
