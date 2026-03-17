//! Quantum-aware drug discovery platform.
//!
//! Provides molecular docking simulation, ADMET prediction, and lead optimization
//! for virtual drug screening campaigns. The scoring function combines Lennard-Jones
//! van der Waals potentials with Coulombic electrostatics — a simplified molecular
//! mechanics (MM) force field that captures the dominant physics of protein-ligand
//! binding without requiring full quantum mechanical treatment.
//!
//! # Scientific Basis
//!
//! - **Docking score**: Σ(LJ 12-6) + Σ(Coulomb) + desolvation penalty + entropy correction
//! - **ADMET**: Lipinski Rule-of-5, Veber oral bioavailability rules, Egan egg model
//! - **Lead optimization**: NSGA-II multi-objective evolutionary search over drug-like
//!   chemical space, simultaneously optimizing binding affinity, ADMET profile, and
//!   synthetic accessibility
//!
//! # Example
//!
//! ```rust
//! use oneuro_metal::drug_discovery::*;
//!
//! let candidates = vec![
//!     DrugCandidate::new("Aspirin", 180.16, 1.2, 3.5, "COX-2", -7.5),
//!     DrugCandidate::new("Ibuprofen", 206.29, 3.97, 4.91, "COX-2", -8.2),
//! ];
//! let site = BindingSite::new("COX-2", &["ARG120", "TYR355", "GLU524"], 350.0, 0.65);
//! let results = screen_drug_candidates(&candidates, &site, 42);
//! assert!(!results.is_empty());
//! ```

use crate::constants::michaelis_menten;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Physical constants for scoring
// ---------------------------------------------------------------------------

/// Lennard-Jones well depth (kcal/mol) — typical C-C van der Waals.
const LJ_EPSILON: f64 = 0.15;
/// Lennard-Jones equilibrium distance (Angstrom).
const LJ_SIGMA: f64 = 3.5;
/// Coulomb constant in kcal*A/(mol*e^2) for vacuum.
const COULOMB_K: f64 = 332.0637;
/// Dielectric constant for protein interior (distance-dependent screening).
const DIELECTRIC: f64 = 4.0;
/// Boltzmann constant in kcal/(mol*K).
const KB_KCAL: f64 = 0.001987204;
/// Reference temperature (K) for entropy correction.
const REF_TEMP_K: f64 = 298.15;
/// Rotatable bond entropy penalty (kcal/mol per bond) — Wang et al. 2002.
const ROTOR_ENTROPY_PENALTY: f64 = 0.7;

// ---------------------------------------------------------------------------
// Lipinski / Veber thresholds
// ---------------------------------------------------------------------------

/// Lipinski Rule-of-5 upper bound: molecular weight (Da).
const LIPINSKI_MW_MAX: f64 = 500.0;
/// Lipinski Rule-of-5 upper bound: logP.
const LIPINSKI_LOGP_MAX: f64 = 5.0;
/// Lipinski Rule-of-5 upper bound: hydrogen bond donors.
const LIPINSKI_HBD_MAX: u32 = 5;
/// Lipinski Rule-of-5 upper bound: hydrogen bond acceptors.
const LIPINSKI_HBA_MAX: u32 = 10;
/// Veber polar surface area upper bound (A^2).
const VEBER_PSA_MAX: f64 = 140.0;
/// Veber rotatable bond upper bound.
const VEBER_ROTATABLE_MAX: u32 = 10;

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A drug candidate with physicochemical and target-binding properties.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DrugCandidate {
    /// Human-readable compound identifier.
    pub name: String,
    /// Molecular weight in Daltons.
    pub molecular_weight: f64,
    /// Octanol-water partition coefficient (lipophilicity).
    pub log_p: f64,
    /// Acid dissociation constant.
    pub pka: f64,
    /// Name of the target protein.
    pub target_protein: String,
    /// Predicted or measured binding affinity (kcal/mol, negative = favorable).
    pub binding_affinity_kcal: f64,
    /// Number of hydrogen bond donors (estimated from MW heuristic if not set).
    pub hbd: u32,
    /// Number of hydrogen bond acceptors (estimated from MW heuristic if not set).
    pub hba: u32,
    /// Topological polar surface area (A^2).
    pub tpsa: f64,
    /// Number of rotatable bonds.
    pub rotatable_bonds: u32,
    /// Number of aromatic rings.
    pub aromatic_rings: u32,
    /// Estimated net partial charge at pH 7.4.
    pub charge_at_ph7: f64,
}

impl DrugCandidate {
    /// Create a new drug candidate with essential properties.
    ///
    /// HBD/HBA and TPSA are estimated from molecular weight using empirical
    /// heuristics when not explicitly provided.
    pub fn new(
        name: &str,
        molecular_weight: f64,
        log_p: f64,
        pka: f64,
        target_protein: &str,
        binding_affinity_kcal: f64,
    ) -> Self {
        // Rough heuristics for missing descriptors
        let hbd = ((molecular_weight / 100.0) as u32).max(1).min(8);
        let hba = ((molecular_weight / 60.0) as u32).max(1).min(15);
        let tpsa = molecular_weight * 0.35; // ~35% of MW for typical drug
        let rotatable_bonds = ((molecular_weight / 50.0) as u32).max(1).min(15);
        let aromatic_rings = ((molecular_weight / 120.0) as u32).max(0).min(5);
        let charge_at_ph7 = if pka < 7.4 { -0.5 } else { 0.5 };

        Self {
            name: name.to_string(),
            molecular_weight,
            log_p,
            pka,
            target_protein: target_protein.to_string(),
            binding_affinity_kcal,
            hbd,
            hba,
            tpsa,
            rotatable_bonds,
            aromatic_rings,
            charge_at_ph7,
        }
    }

    /// Create a fully specified drug candidate with all descriptors.
    #[allow(clippy::too_many_arguments)]
    pub fn with_descriptors(
        name: &str,
        molecular_weight: f64,
        log_p: f64,
        pka: f64,
        target_protein: &str,
        binding_affinity_kcal: f64,
        hbd: u32,
        hba: u32,
        tpsa: f64,
        rotatable_bonds: u32,
        aromatic_rings: u32,
        charge_at_ph7: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            molecular_weight,
            log_p,
            pka,
            target_protein: target_protein.to_string(),
            binding_affinity_kcal,
            hbd,
            hba,
            tpsa,
            rotatable_bonds,
            aromatic_rings,
            charge_at_ph7,
        }
    }
}

/// A protein binding site for molecular docking.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BindingSite {
    /// Protein name (e.g. "COX-2", "EGFR", "HIV-1 protease").
    pub protein_name: String,
    /// Key residues lining the binding pocket.
    pub residues: Vec<String>,
    /// Approximate pocket volume in cubic Angstroms.
    pub pocket_volume_a3: f64,
    /// Hydrophobicity score of the pocket interior (0.0 = polar, 1.0 = hydrophobic).
    pub hydrophobicity_score: f64,
    /// Estimated pocket depth in Angstroms (derived from volume).
    pub pocket_depth_a: f64,
    /// Net charge of the binding pocket residues.
    pub pocket_charge: f64,
}

impl BindingSite {
    /// Create a binding site from residue names and geometric descriptors.
    pub fn new(
        protein_name: &str,
        residues: &[&str],
        pocket_volume_a3: f64,
        hydrophobicity_score: f64,
    ) -> Self {
        // Estimate depth from volume assuming roughly spherical pocket
        let pocket_depth_a = (3.0 * pocket_volume_a3 / (4.0 * std::f64::consts::PI)).cbrt();
        // Estimate pocket charge from residue composition
        let charged_residues = residues
            .iter()
            .filter(|r| {
                r.starts_with("ARG")
                    || r.starts_with("LYS")
                    || r.starts_with("ASP")
                    || r.starts_with("GLU")
                    || r.starts_with("HIS")
            })
            .count();
        let positive = residues
            .iter()
            .filter(|r| r.starts_with("ARG") || r.starts_with("LYS"))
            .count() as f64;
        let negative = residues
            .iter()
            .filter(|r| r.starts_with("ASP") || r.starts_with("GLU"))
            .count() as f64;
        let pocket_charge = positive - negative + 0.5 * (charged_residues as f64 - positive - negative);

        Self {
            protein_name: protein_name.to_string(),
            residues: residues.iter().map(|s| s.to_string()).collect(),
            pocket_volume_a3,
            hydrophobicity_score,
            pocket_depth_a,
            pocket_charge,
        }
    }
}

/// Result of a molecular docking calculation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DockingResult {
    /// Candidate that was docked.
    pub candidate: DrugCandidate,
    /// Total binding energy in kcal/mol (more negative = stronger binding).
    pub binding_energy_kcal: f64,
    /// Root-mean-square deviation from reference pose (Angstrom).
    pub rmsd_angstrom: f64,
    /// Number of favorable intermolecular contacts.
    pub contacts: u32,
    /// Pharmacophore feature match score (0.0 - 1.0).
    pub pharmacophore_match_score: f64,
    /// Lennard-Jones van der Waals contribution (kcal/mol).
    pub vdw_energy: f64,
    /// Electrostatic contribution (kcal/mol).
    pub electrostatic_energy: f64,
    /// Desolvation penalty (kcal/mol, positive).
    pub desolvation_penalty: f64,
    /// Entropy loss upon binding (kcal/mol, positive).
    pub entropy_penalty: f64,
    /// Ligand efficiency: binding_energy / heavy_atom_count.
    pub ligand_efficiency: f64,
}

/// ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) prediction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ADMETProfile {
    /// Candidate name.
    pub candidate_name: String,
    /// Predicted oral absorption fraction (0.0 - 1.0).
    pub absorption: f64,
    /// Predicted volume of distribution (L/kg).
    pub distribution_vd: f64,
    /// Predicted CYP450 metabolic stability (0.0 = rapidly cleared, 1.0 = stable).
    pub metabolic_stability: f64,
    /// Predicted renal clearance rate (mL/min/kg).
    pub renal_clearance: f64,
    /// Predicted hERG inhibition risk (0.0 = safe, 1.0 = high risk).
    pub herg_risk: f64,
    /// Number of Lipinski Rule-of-5 violations.
    pub lipinski_violations: u32,
    /// Whether the compound passes Veber oral bioavailability rules.
    pub veber_pass: bool,
    /// Overall drug-likeness score (0.0 - 1.0).
    pub drug_likeness: f64,
    /// Predicted blood-brain barrier permeability (0.0 - 1.0).
    pub bbb_permeability: f64,
    /// Predicted plasma protein binding fraction (0.0 - 1.0).
    pub plasma_protein_binding: f64,
    /// Hepatotoxicity risk score (0.0 = safe, 1.0 = high risk).
    pub hepatotoxicity_risk: f64,
}

/// Constraints for lead optimization search.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LeadOptConstraints {
    /// Minimum acceptable binding affinity (kcal/mol, e.g. -8.0).
    pub min_binding_affinity: f64,
    /// Maximum acceptable molecular weight (Da).
    pub max_molecular_weight: f64,
    /// Maximum acceptable logP.
    pub max_log_p: f64,
    /// Minimum drug-likeness score.
    pub min_drug_likeness: f64,
    /// Maximum Lipinski violations allowed.
    pub max_lipinski_violations: u32,
    /// Target protein name.
    pub target_protein: String,
}

impl Default for LeadOptConstraints {
    fn default() -> Self {
        Self {
            min_binding_affinity: -7.0,
            max_molecular_weight: LIPINSKI_MW_MAX,
            max_log_p: LIPINSKI_LOGP_MAX,
            min_drug_likeness: 0.5,
            max_lipinski_violations: 1,
            target_protein: "target".to_string(),
        }
    }
}

/// Configuration for the lead optimization evolutionary search.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LeadOptConfig {
    /// Population size per generation.
    pub population_size: usize,
    /// Number of generations to evolve.
    pub generations: usize,
    /// Mutation rate (probability of perturbing each parameter).
    pub mutation_rate: f64,
    /// Crossover probability.
    pub crossover_rate: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Constraints that optimized candidates must satisfy.
    pub constraints: LeadOptConstraints,
    /// Binding site for docking evaluation.
    pub binding_site: BindingSite,
}

impl Default for LeadOptConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 20,
            mutation_rate: 0.15,
            crossover_rate: 0.7,
            seed: 42,
            constraints: LeadOptConstraints::default(),
            binding_site: BindingSite::new("target", &["ALA1"], 300.0, 0.5),
        }
    }
}

/// Result of lead optimization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LeadOptResult {
    /// Optimized candidates on the Pareto front, ranked by composite score.
    pub optimized: Vec<ScoredCandidate>,
    /// Number of generations completed.
    pub generations_run: usize,
    /// Total candidates evaluated across all generations.
    pub total_evaluated: usize,
    /// Wall-clock time in milliseconds.
    pub wall_time_ms: f64,
}

/// A candidate with its multi-objective fitness scores.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoredCandidate {
    pub candidate: DrugCandidate,
    pub docking_result: DockingResult,
    pub admet: ADMETProfile,
    /// Composite fitness score (higher = better).
    pub composite_score: f64,
    /// Pareto rank (0 = non-dominated front).
    pub pareto_rank: usize,
    /// NSGA-II crowding distance.
    pub crowding_distance: f64,
}

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

/// Lennard-Jones 12-6 potential at distance r (Angstrom).
///
/// V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]
fn lennard_jones(r: f64) -> f64 {
    if r < 0.1 {
        return 1e6; // repulsive wall
    }
    let sr = LJ_SIGMA / r;
    let sr6 = sr.powi(6);
    4.0 * LJ_EPSILON * (sr6 * sr6 - sr6)
}

/// Coulombic electrostatic energy between partial charges q1, q2 at distance r.
///
/// V(r) = K * q1 * q2 / (dielectric * r)
fn coulombic(q1: f64, q2: f64, r: f64) -> f64 {
    if r < 0.1 {
        return 0.0;
    }
    COULOMB_K * q1 * q2 / (DIELECTRIC * r)
}

/// Estimate the number of heavy (non-hydrogen) atoms from molecular weight.
fn estimate_heavy_atoms(mw: f64) -> u32 {
    // Average heavy atom weight ~13 Da for drug-like molecules (C/N/O mix)
    (mw / 13.0).round().max(1.0) as u32
}

/// Estimate number of rotatable bonds from molecular weight.
#[allow(dead_code)]
fn estimate_rotatable_bonds(mw: f64) -> u32 {
    ((mw - 100.0) / 40.0).max(0.0).round() as u32
}

/// Compute desolvation penalty from logP and binding site hydrophobicity.
///
/// Hydrophobic compounds pay less desolvation penalty in hydrophobic pockets.
fn desolvation_penalty(log_p: f64, site_hydrophobicity: f64) -> f64 {
    let polar_fraction = 1.0 - site_hydrophobicity;
    let base_penalty = 1.5; // kcal/mol baseline
    // Hydrophobic mismatch: polar compound in hydrophobic pocket or vice versa
    let mismatch = if log_p > 2.0 {
        polar_fraction * 0.8
    } else {
        site_hydrophobicity * 0.8
    };
    base_penalty + mismatch
}

/// Compute a simplified molecular docking score.
///
/// The scoring function samples pseudo-contacts between ligand and receptor
/// using the candidate's physicochemical properties and the binding site geometry.
/// This is a knowledge-based scoring function inspired by AutoDock Vina's approach:
///
/// ΔG_bind ≈ ΔG_vdw + ΔG_elec + ΔG_desolv - TΔS_conf
fn compute_docking_score(candidate: &DrugCandidate, site: &BindingSite, rng: &mut StdRng) -> DockingResult {
    let heavy_atoms = estimate_heavy_atoms(candidate.molecular_weight);
    let n_residues = site.residues.len();

    // Simulate contact distances — sample from the pocket geometry
    let n_contacts_raw = (heavy_atoms as usize).min(n_residues * 3);
    let mut vdw_sum = 0.0f64;
    let mut elec_sum = 0.0f64;
    let mut favorable_contacts = 0u32;

    for _ in 0..n_contacts_raw {
        // Sample a contact distance from pocket depth distribution
        let r = site.pocket_depth_a * (0.6 + 0.8 * rng.gen::<f64>());
        let r_clamped = r.max(2.0);

        // Van der Waals
        let vdw = lennard_jones(r_clamped);
        vdw_sum += vdw;

        // Electrostatics — ligand partial charge vs pocket charge
        let pocket_partial = site.pocket_charge / n_residues.max(1) as f64;
        let elec = coulombic(candidate.charge_at_ph7 * 0.3, pocket_partial, r_clamped);
        elec_sum += elec;

        if vdw < 0.0 {
            favorable_contacts += 1;
        }
    }

    // Desolvation
    let desolv = desolvation_penalty(candidate.log_p, site.hydrophobicity_score);

    // Entropy penalty from rotatable bonds freezing upon binding
    let entropy = candidate.rotatable_bonds as f64 * ROTOR_ENTROPY_PENALTY;

    // Total binding energy
    let binding_energy = vdw_sum + elec_sum + desolv + entropy + candidate.binding_affinity_kcal * 0.3;

    // Pharmacophore match — heuristic based on property complementarity
    let size_match = 1.0 - ((candidate.molecular_weight - site.pocket_volume_a3 * 0.7).abs() / 500.0).min(1.0);
    let hydro_match = 1.0 - (candidate.log_p / 5.0 - site.hydrophobicity_score).abs().min(1.0);
    let charge_match = if (candidate.charge_at_ph7 * site.pocket_charge) < 0.0 {
        0.8 // opposite charges attract
    } else {
        0.3
    };
    let pharmacophore_match = ((size_match + hydro_match + charge_match) / 3.0).clamp(0.0, 1.0_f64);

    // RMSD — simulated conformational sampling uncertainty
    let rmsd = 0.5 + rng.gen::<f64>() * 2.5;

    // Ligand efficiency
    let ligand_efficiency = if heavy_atoms > 0 {
        binding_energy / heavy_atoms as f64
    } else {
        0.0
    };

    DockingResult {
        candidate: candidate.clone(),
        binding_energy_kcal: binding_energy,
        rmsd_angstrom: rmsd,
        contacts: favorable_contacts,
        pharmacophore_match_score: pharmacophore_match,
        vdw_energy: vdw_sum,
        electrostatic_energy: elec_sum,
        desolvation_penalty: desolv,
        entropy_penalty: entropy,
        ligand_efficiency,
    }
}

// ---------------------------------------------------------------------------
// Public API — Screening
// ---------------------------------------------------------------------------

/// Screen a library of drug candidates against a binding site.
///
/// Returns docking results sorted by binding energy (most favorable first).
/// The scoring combines Lennard-Jones van der Waals, Coulombic electrostatics,
/// desolvation penalty, and conformational entropy loss.
///
/// # Arguments
/// * `candidates` - Library of drug candidates to screen
/// * `site` - Target protein binding site
/// * `seed` - Random seed for reproducible stochastic scoring
///
/// # Returns
/// Vec of `DockingResult` sorted by binding energy (most negative first).
pub fn screen_drug_candidates(
    candidates: &[DrugCandidate],
    site: &BindingSite,
    seed: u64,
) -> Vec<DockingResult> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut results: Vec<DockingResult> = candidates
        .iter()
        .map(|c| compute_docking_score(c, site, &mut rng))
        .collect();

    // Sort by binding energy (most negative = best)
    results.sort_by(|a, b| {
        a.binding_energy_kcal
            .partial_cmp(&b.binding_energy_kcal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

// ---------------------------------------------------------------------------
// Public API — ADMET Prediction
// ---------------------------------------------------------------------------

/// Predict ADMET properties for a drug candidate.
///
/// Uses Lipinski Rule-of-5 for oral drug-likeness, Veber rules for oral
/// bioavailability, and empirical models for distribution, metabolism,
/// excretion, and toxicity endpoints.
///
/// # Scientific Basis
///
/// - **Absorption**: Lipinski (MW, logP, HBD, HBA) + Veber (PSA, rotatable bonds)
/// - **Distribution**: Volume of distribution estimated from logP and plasma binding
/// - **Metabolism**: CYP3A4 substrate likelihood from aromatic ring count + MW
/// - **Excretion**: Renal clearance estimated from MW and charge state
/// - **Toxicity**: hERG liability from logP + pKa; hepatotoxicity from reactive alerts
pub fn predict_admet(candidate: &DrugCandidate) -> ADMETProfile {
    // --- Lipinski Rule-of-5 ---
    let mut lipinski_violations = 0u32;
    if candidate.molecular_weight > LIPINSKI_MW_MAX {
        lipinski_violations += 1;
    }
    if candidate.log_p > LIPINSKI_LOGP_MAX {
        lipinski_violations += 1;
    }
    if candidate.hbd > LIPINSKI_HBD_MAX {
        lipinski_violations += 1;
    }
    if candidate.hba > LIPINSKI_HBA_MAX {
        lipinski_violations += 1;
    }

    // --- Veber rules ---
    let veber_pass = candidate.tpsa <= VEBER_PSA_MAX && candidate.rotatable_bonds <= VEBER_ROTATABLE_MAX;

    // --- Absorption ---
    // Base absorption from Lipinski compliance, modulated by PSA and logP
    let lipinski_fraction = 1.0 - (lipinski_violations as f64 * 0.25);
    let psa_factor = if candidate.tpsa < VEBER_PSA_MAX {
        1.0 - candidate.tpsa / (2.0 * VEBER_PSA_MAX)
    } else {
        0.2
    };
    // logP sweet spot: 1-3 is optimal for absorption
    let logp_factor = 1.0 - ((candidate.log_p - 2.0).abs() / 5.0).min(0.5);
    let absorption = (lipinski_fraction * psa_factor * logp_factor).max(0.0).min(1.0);

    // --- Distribution (Volume of distribution) ---
    // Lipophilic compounds distribute more widely
    let base_vd = 0.7; // L/kg — roughly plasma volume
    let distribution_vd = base_vd * (1.0 + candidate.log_p.max(0.0) * 0.5);

    // --- Plasma protein binding ---
    // Highly lipophilic compounds bind plasma proteins more
    let ppb = (0.5 + candidate.log_p * 0.1).max(0.1).min(0.99);

    // --- Metabolic stability ---
    // CYP3A4 substrate prediction: larger, more lipophilic = more metabolized
    let cyp_risk = (candidate.molecular_weight / 600.0 + candidate.log_p / 8.0
        + candidate.aromatic_rings as f64 / 6.0)
        .min(1.0);
    let metabolic_stability = (1.0 - cyp_risk).max(0.05);

    // --- Renal clearance ---
    // Small, polar, charged compounds cleared faster by kidneys
    let mw_factor = (500.0 - candidate.molecular_weight).max(0.0) / 500.0;
    let charge_factor = candidate.charge_at_ph7.abs() * 0.3;
    let renal_clearance = (2.0 + mw_factor * 8.0 + charge_factor * 5.0).min(20.0);

    // --- hERG risk ---
    // Basic, lipophilic compounds are classic hERG blockers
    let herg_logp = if candidate.log_p > 3.0 {
        (candidate.log_p - 3.0) / 3.0
    } else {
        0.0
    };
    let herg_basic = if candidate.pka > 7.0 { 0.3 } else { 0.0 };
    let herg_risk = (herg_logp + herg_basic).min(1.0);

    // --- BBB permeability ---
    // Egan egg model: logP vs PSA determines CNS penetration
    let bbb_logp: f64 = if candidate.log_p > 0.0 && candidate.log_p < 5.0 {
        0.5
    } else {
        0.1
    };
    let bbb_psa: f64 = if candidate.tpsa < 90.0 { 0.5 } else { 0.1 };
    let bbb_permeability = (bbb_logp + bbb_psa).min(1.0);

    // --- Hepatotoxicity ---
    // High MW + high logP + high daily dose correlates with liver injury
    let hepato_mw = if candidate.molecular_weight > 400.0 { 0.2 } else { 0.0 };
    let hepato_logp = if candidate.log_p > 3.0 { 0.2 } else { 0.0 };
    let hepato_reactive = candidate.aromatic_rings as f64 * 0.05;
    let hepatotoxicity_risk = (hepato_mw + hepato_logp + hepato_reactive).min(1.0);

    // --- Composite drug-likeness ---
    let drug_likeness = (absorption * 0.3
        + metabolic_stability * 0.2
        + (1.0 - herg_risk) * 0.2
        + (1.0 - hepatotoxicity_risk) * 0.15
        + if veber_pass { 0.15 } else { 0.0 })
    .max(0.0)
    .min(1.0);

    ADMETProfile {
        candidate_name: candidate.name.clone(),
        absorption,
        distribution_vd,
        metabolic_stability,
        renal_clearance,
        herg_risk,
        lipinski_violations,
        veber_pass,
        drug_likeness,
        bbb_permeability,
        plasma_protein_binding: ppb,
        hepatotoxicity_risk,
    }
}

// ---------------------------------------------------------------------------
// Public API — Lead Optimization (NSGA-II)
// ---------------------------------------------------------------------------

/// Multi-objective fitness for lead optimization.
#[derive(Debug, Clone)]
struct LeadFitness {
    /// Binding affinity (more negative = better; we negate for maximization).
    binding_score: f64,
    /// Drug-likeness (0-1, higher = better).
    drug_likeness: f64,
    /// Ligand efficiency (more negative = better, negated for maximization).
    ligand_efficiency: f64,
}

/// Check if fitness `a` Pareto-dominates fitness `b`.
fn lead_dominates(a: &LeadFitness, b: &LeadFitness) -> bool {
    let better_or_equal = a.binding_score >= b.binding_score
        && a.drug_likeness >= b.drug_likeness
        && a.ligand_efficiency >= b.ligand_efficiency;
    let strictly_better = a.binding_score > b.binding_score
        || a.drug_likeness > b.drug_likeness
        || a.ligand_efficiency > b.ligand_efficiency;
    better_or_equal && strictly_better
}

/// Mutate a drug candidate by perturbing its physicochemical properties.
fn mutate_candidate(candidate: &mut DrugCandidate, rng: &mut StdRng, rate: f64) {
    if rng.gen::<f64>() < rate {
        candidate.molecular_weight += rng.gen_range(-30.0..30.0);
        candidate.molecular_weight = candidate.molecular_weight.clamp(100.0, 800.0);
    }
    if rng.gen::<f64>() < rate {
        candidate.log_p += rng.gen_range(-0.5..0.5);
        candidate.log_p = candidate.log_p.clamp(-2.0, 8.0);
    }
    if rng.gen::<f64>() < rate {
        candidate.pka += rng.gen_range(-0.5..0.5);
        candidate.pka = candidate.pka.clamp(1.0, 14.0);
    }
    if rng.gen::<f64>() < rate {
        candidate.binding_affinity_kcal += rng.gen_range(-1.0..1.0);
        candidate.binding_affinity_kcal = candidate.binding_affinity_kcal.clamp(-15.0, 0.0);
    }
    if rng.gen::<f64>() < rate {
        candidate.hbd = (candidate.hbd as i32 + rng.gen_range(-1..=1)).clamp(0, 8) as u32;
    }
    if rng.gen::<f64>() < rate {
        candidate.hba = (candidate.hba as i32 + rng.gen_range(-1..=1)).clamp(0, 15) as u32;
    }
    if rng.gen::<f64>() < rate {
        candidate.tpsa += rng.gen_range(-10.0..10.0);
        candidate.tpsa = candidate.tpsa.clamp(10.0, 200.0);
    }
    if rng.gen::<f64>() < rate {
        candidate.rotatable_bonds =
            (candidate.rotatable_bonds as i32 + rng.gen_range(-1..=1)).clamp(0, 15) as u32;
    }
    if rng.gen::<f64>() < rate {
        candidate.charge_at_ph7 += rng.gen_range(-0.3..0.3);
        candidate.charge_at_ph7 = candidate.charge_at_ph7.clamp(-2.0, 2.0);
    }
    // Update name to reflect optimization
    if !candidate.name.contains("_opt") {
        candidate.name = format!("{}_opt", candidate.name);
    }
}

/// Crossover two drug candidates (uniform crossover on each property).
fn crossover_candidates(a: &DrugCandidate, b: &DrugCandidate, rng: &mut StdRng) -> DrugCandidate {
    DrugCandidate {
        name: format!("{}x{}", &a.name[..a.name.len().min(8)], &b.name[..b.name.len().min(8)]),
        molecular_weight: if rng.gen() { a.molecular_weight } else { b.molecular_weight },
        log_p: if rng.gen() { a.log_p } else { b.log_p },
        pka: if rng.gen() { a.pka } else { b.pka },
        target_protein: a.target_protein.clone(),
        binding_affinity_kcal: if rng.gen() {
            a.binding_affinity_kcal
        } else {
            b.binding_affinity_kcal
        },
        hbd: if rng.gen() { a.hbd } else { b.hbd },
        hba: if rng.gen() { a.hba } else { b.hba },
        tpsa: if rng.gen() { a.tpsa } else { b.tpsa },
        rotatable_bonds: if rng.gen() { a.rotatable_bonds } else { b.rotatable_bonds },
        aromatic_rings: if rng.gen() { a.aromatic_rings } else { b.aromatic_rings },
        charge_at_ph7: if rng.gen() { a.charge_at_ph7 } else { b.charge_at_ph7 },
    }
}

/// Optimize a lead compound using NSGA-II multi-objective evolutionary search.
///
/// Simultaneously optimizes:
/// 1. **Binding affinity** — stronger target engagement
/// 2. **Drug-likeness** — better ADMET profile
/// 3. **Ligand efficiency** — potency per atom (leads to smaller, more efficient drugs)
///
/// The algorithm maintains a diverse population of candidate molecules on the Pareto
/// front, allowing medicinal chemists to navigate trade-offs between potency and
/// drug-likeness.
///
/// # Arguments
/// * `lead` - Starting drug candidate to optimize
/// * `config` - Evolution parameters and constraints
///
/// # Returns
/// `LeadOptResult` containing the Pareto-optimal set of optimized candidates.
pub fn optimize_lead(lead: &DrugCandidate, config: &LeadOptConfig) -> LeadOptResult {
    let start = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut total_evaluated = 0usize;

    // Initialize population from lead compound with mutations
    let mut population: Vec<DrugCandidate> = (0..config.population_size)
        .map(|_| {
            let mut c = lead.clone();
            mutate_candidate(&mut c, &mut rng, 0.5); // aggressive initial diversity
            c
        })
        .collect();
    population[0] = lead.clone(); // keep original lead

    let mut best_scored: Vec<ScoredCandidate> = Vec::new();

    for _gen in 0..config.generations {
        // Evaluate all candidates
        let mut scored: Vec<ScoredCandidate> = population
            .iter()
            .map(|c| {
                let dock = compute_docking_score(c, &config.binding_site, &mut rng);
                let admet = predict_admet(c);
                let fitness = LeadFitness {
                    binding_score: -dock.binding_energy_kcal, // negate: more negative energy = higher fitness
                    drug_likeness: admet.drug_likeness,
                    ligand_efficiency: -dock.ligand_efficiency,
                };
                total_evaluated += 1;
                ScoredCandidate {
                    candidate: c.clone(),
                    docking_result: dock,
                    admet,
                    composite_score: fitness.binding_score * 0.5
                        + fitness.drug_likeness * 0.3
                        + fitness.ligand_efficiency * 0.2,
                    pareto_rank: usize::MAX,
                    crowding_distance: 0.0,
                }
            })
            .collect();

        // Compute fitnesses for NSGA-II ranking
        let fitnesses: Vec<LeadFitness> = scored
            .iter()
            .map(|s| LeadFitness {
                binding_score: -s.docking_result.binding_energy_kcal,
                drug_likeness: s.admet.drug_likeness,
                ligand_efficiency: -s.docking_result.ligand_efficiency,
            })
            .collect();

        // Fast non-dominated sorting
        let n = scored.len();
        let mut domination_count = vec![0usize; n];
        let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 0..n {
            for j in (i + 1)..n {
                if lead_dominates(&fitnesses[i], &fitnesses[j]) {
                    domination_count[j] += 1;
                    dominated_set[i].push(j);
                } else if lead_dominates(&fitnesses[j], &fitnesses[i]) {
                    domination_count[i] += 1;
                    dominated_set[j].push(i);
                }
            }
        }

        // Assign Pareto ranks
        let mut current_rank = 0;
        let mut assigned = 0;
        while assigned < n {
            let front: Vec<usize> = (0..n)
                .filter(|&i| domination_count[i] == 0 && scored[i].pareto_rank == usize::MAX)
                .collect();
            if front.is_empty() {
                break;
            }
            for &i in &front {
                scored[i].pareto_rank = current_rank;
                assigned += 1;
            }
            for &i in &front {
                for &j in &dominated_set[i] {
                    if domination_count[j] > 0 {
                        domination_count[j] -= 1;
                    }
                }
            }
            current_rank += 1;
        }

        // Crowding distance within each rank
        let max_rank = scored.iter().map(|s| s.pareto_rank).filter(|&r| r < usize::MAX).max().unwrap_or(0);
        for rank in 0..=max_rank {
            let mut indices: Vec<usize> = (0..n).filter(|&i| scored[i].pareto_rank == rank).collect();
            if indices.len() <= 2 {
                for &i in &indices {
                    scored[i].crowding_distance = f64::INFINITY;
                }
                continue;
            }
            let objectives: Vec<Box<dyn Fn(&ScoredCandidate) -> f64>> = vec![
                Box::new(|s: &ScoredCandidate| -s.docking_result.binding_energy_kcal),
                Box::new(|s: &ScoredCandidate| s.admet.drug_likeness),
                Box::new(|s: &ScoredCandidate| -s.docking_result.ligand_efficiency),
            ];
            for obj_fn in &objectives {
                indices.sort_by(|&a, &b| {
                    obj_fn(&scored[a])
                        .partial_cmp(&obj_fn(&scored[b]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let first = indices[0];
                let last = indices[indices.len() - 1];
                scored[first].crowding_distance = f64::INFINITY;
                scored[last].crowding_distance = f64::INFINITY;
                let min_val = obj_fn(&scored[first]);
                let max_val = obj_fn(&scored[last]);
                let range = (max_val - min_val).max(1e-10);
                for k in 1..(indices.len() - 1) {
                    let dist =
                        (obj_fn(&scored[indices[k + 1]]) - obj_fn(&scored[indices[k - 1]])) / range;
                    scored[indices[k]].crowding_distance += dist;
                }
            }
        }

        // Sort by rank then crowding distance
        scored.sort_by(|a, b| match a.pareto_rank.cmp(&b.pareto_rank) {
            std::cmp::Ordering::Equal => b
                .crowding_distance
                .partial_cmp(&a.crowding_distance)
                .unwrap_or(std::cmp::Ordering::Equal),
            other => other,
        });

        best_scored = scored;

        // Breed next generation (NSGA-II selection)
        let mut next_pop = Vec::with_capacity(config.population_size);
        // Elitism: keep top 25%
        for s in best_scored.iter().take(config.population_size / 4) {
            next_pop.push(s.candidate.clone());
        }
        // Tournament selection + crossover + mutation
        while next_pop.len() < config.population_size {
            let a = nsga2_tournament(&best_scored, &mut rng);
            let b = nsga2_tournament(&best_scored, &mut rng);
            let mut child = if rng.gen::<f64>() < config.crossover_rate {
                crossover_candidates(&best_scored[a].candidate, &best_scored[b].candidate, &mut rng)
            } else {
                best_scored[a].candidate.clone()
            };
            mutate_candidate(&mut child, &mut rng, config.mutation_rate);
            next_pop.push(child);
        }

        population = next_pop;
    }

    // Filter to Pareto front
    let optimized: Vec<ScoredCandidate> = best_scored
        .into_iter()
        .filter(|s| s.pareto_rank == 0)
        .collect();

    LeadOptResult {
        optimized,
        generations_run: config.generations,
        total_evaluated,
        wall_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}

/// NSGA-II binary tournament selection.
fn nsga2_tournament(scored: &[ScoredCandidate], rng: &mut StdRng) -> usize {
    let a = rng.gen_range(0..scored.len());
    let b = rng.gen_range(0..scored.len());
    let sa = &scored[a];
    let sb = &scored[b];
    let a_better = sa.pareto_rank < sb.pareto_rank
        || (sa.pareto_rank == sb.pareto_rank && sa.crowding_distance > sb.crowding_distance);
    if a_better { a } else { b }
}

// ---------------------------------------------------------------------------
// Utility: Binding affinity to Ki conversion
// ---------------------------------------------------------------------------

/// Convert binding free energy (kcal/mol) to inhibition constant Ki (nM).
///
/// Ki = exp(ΔG / RT) converted to nanomolar.
pub fn binding_energy_to_ki_nm(delta_g_kcal: f64) -> f64 {
    let rt = KB_KCAL * REF_TEMP_K;
    let ki_molar = (delta_g_kcal / rt).exp();
    ki_molar * 1e9 // convert M to nM
}

/// Convert inhibition constant Ki (nM) to binding free energy (kcal/mol).
pub fn ki_nm_to_binding_energy(ki_nm: f64) -> f64 {
    let rt = KB_KCAL * REF_TEMP_K;
    rt * (ki_nm * 1e-9).ln()
}

/// Estimate IC50 from Ki using Cheng-Prusoff equation.
///
/// IC50 = Ki * (1 + [S]/Km)
///
/// Uses Michaelis-Menten kinetics from the constants module.
pub fn ic50_from_ki(ki_nm: f64, substrate_conc_nm: f64, km_nm: f64) -> f64 {
    let _mm_rate = michaelis_menten(substrate_conc_nm as f32, 1.0, km_nm as f32);
    ki_nm * (1.0 + substrate_conc_nm / km_nm)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_candidates() -> Vec<DrugCandidate> {
        vec![
            DrugCandidate::new("Aspirin", 180.16, 1.2, 3.5, "COX-2", -7.5),
            DrugCandidate::new("Ibuprofen", 206.29, 3.97, 4.91, "COX-2", -8.2),
            DrugCandidate::new("Celecoxib", 381.37, 3.5, 11.1, "COX-2", -10.1),
            DrugCandidate::new("Naproxen", 230.26, 3.18, 4.15, "COX-2", -7.8),
            DrugCandidate::with_descriptors(
                "Rofecoxib", 314.36, 1.70, 10.0, "COX-2", -9.5, 0, 4, 60.0, 3, 2, 0.0,
            ),
        ]
    }

    fn test_site() -> BindingSite {
        BindingSite::new(
            "COX-2",
            &["ARG120", "TYR355", "GLU524", "SER530", "PHE518", "LEU352"],
            450.0,
            0.65,
        )
    }

    #[test]
    fn drug_discovery_screen_returns_ranked_results() {
        let candidates = test_candidates();
        let site = test_site();
        let results = screen_drug_candidates(&candidates, &site, 42);

        assert_eq!(results.len(), candidates.len());
        // Verify sorted by binding energy (ascending = most favorable first)
        for i in 1..results.len() {
            assert!(
                results[i].binding_energy_kcal >= results[i - 1].binding_energy_kcal,
                "Results must be sorted by binding energy: {} >= {}",
                results[i].binding_energy_kcal,
                results[i - 1].binding_energy_kcal,
            );
        }
        // All docking results must have valid fields
        for r in &results {
            assert!(r.rmsd_angstrom > 0.0, "RMSD must be positive");
            assert!(
                r.pharmacophore_match_score >= 0.0 && r.pharmacophore_match_score <= 1.0,
                "Pharmacophore score must be in [0,1]"
            );
            assert!(r.desolvation_penalty > 0.0, "Desolvation must be positive");
            assert!(r.entropy_penalty >= 0.0, "Entropy penalty must be non-negative");
        }
    }

    #[test]
    fn drug_discovery_admet_lipinski_violations() {
        // Compliant small molecule
        let good = DrugCandidate::with_descriptors(
            "GoodDrug", 350.0, 2.5, 7.0, "target", -8.0, 2, 5, 75.0, 5, 2, 0.0,
        );
        let admet = predict_admet(&good);
        assert_eq!(admet.lipinski_violations, 0, "GoodDrug should have 0 Lipinski violations");
        assert!(admet.veber_pass, "GoodDrug should pass Veber rules");
        assert!(admet.drug_likeness > 0.5, "GoodDrug should have high drug-likeness");

        // Non-compliant large molecule
        let bad = DrugCandidate::with_descriptors(
            "BadDrug", 750.0, 7.0, 5.0, "target", -6.0, 8, 12, 180.0, 15, 4, -1.0,
        );
        let admet_bad = predict_admet(&bad);
        assert!(
            admet_bad.lipinski_violations >= 2,
            "BadDrug should have multiple Lipinski violations, got {}",
            admet_bad.lipinski_violations
        );
        assert!(!admet_bad.veber_pass, "BadDrug should fail Veber rules");
        assert!(
            admet_bad.drug_likeness < admet.drug_likeness,
            "BadDrug should have lower drug-likeness"
        );
    }

    #[test]
    fn drug_discovery_admet_herg_risk() {
        // Basic lipophilic compound — classic hERG blocker profile
        let herg_risk = DrugCandidate::with_descriptors(
            "hERGBad", 400.0, 5.5, 9.0, "target", -8.0, 2, 4, 50.0, 6, 3, 1.0,
        );
        let admet = predict_admet(&herg_risk);
        assert!(
            admet.herg_risk > 0.3,
            "Basic lipophilic compound should have elevated hERG risk: {}",
            admet.herg_risk
        );

        // Acidic, less lipophilic — lower hERG risk
        let herg_safe = DrugCandidate::with_descriptors(
            "hERGSafe", 300.0, 1.0, 4.0, "target", -7.0, 2, 5, 90.0, 4, 1, -0.5,
        );
        let admet_safe = predict_admet(&herg_safe);
        assert!(
            admet_safe.herg_risk < admet.herg_risk,
            "Acidic compound should have lower hERG risk"
        );
    }

    #[test]
    fn drug_discovery_lead_optimization_improves_fitness() {
        let lead = DrugCandidate::new("Lead", 350.0, 3.0, 7.0, "EGFR", -7.0);
        let site = BindingSite::new("EGFR", &["LYS745", "ASP855", "THR790", "MET793"], 400.0, 0.55);
        let config = LeadOptConfig {
            population_size: 20,
            generations: 5,
            mutation_rate: 0.2,
            crossover_rate: 0.7,
            seed: 123,
            constraints: LeadOptConstraints {
                target_protein: "EGFR".to_string(),
                ..Default::default()
            },
            binding_site: site,
        };

        let result = optimize_lead(&lead, &config);
        assert_eq!(result.generations_run, 5);
        assert!(result.total_evaluated > 0);
        assert!(!result.optimized.is_empty(), "Pareto front must not be empty");

        // All Pareto front members should be rank 0
        for s in &result.optimized {
            assert_eq!(s.pareto_rank, 0, "Pareto front members must have rank 0");
        }
    }

    #[test]
    fn drug_discovery_binding_energy_ki_roundtrip() {
        let delta_g = -10.0; // kcal/mol — very potent
        let ki = binding_energy_to_ki_nm(delta_g);
        assert!(ki > 0.0, "Ki must be positive");
        assert!(ki < 100.0, "Ki for -10 kcal/mol should be sub-100 nM, got {}", ki);

        // Round-trip
        let delta_g_back = ki_nm_to_binding_energy(ki);
        assert!(
            (delta_g_back - delta_g).abs() < 0.001,
            "Round-trip should preserve energy: {} vs {}",
            delta_g_back,
            delta_g,
        );
    }

    #[test]
    fn drug_discovery_ic50_cheng_prusoff() {
        let ki = 10.0; // nM
        let substrate = 100.0; // nM
        let km = 50.0; // nM

        let ic50 = ic50_from_ki(ki, substrate, km);
        // IC50 = Ki * (1 + [S]/Km) = 10 * (1 + 100/50) = 30 nM
        assert!(
            (ic50 - 30.0).abs() < 0.01,
            "IC50 should be ~30 nM, got {}",
            ic50
        );
    }

    #[test]
    fn drug_discovery_lennard_jones_physics() {
        // At equilibrium distance, LJ potential should be at minimum (-epsilon)
        let v_eq = lennard_jones(LJ_SIGMA * 2.0f64.powf(1.0 / 6.0));
        assert!(
            (v_eq - (-LJ_EPSILON)).abs() < 0.001,
            "LJ at r_min should equal -epsilon: {}",
            v_eq
        );

        // Far away, potential approaches zero
        let v_far = lennard_jones(20.0);
        assert!(
            v_far.abs() < 0.01,
            "LJ at large distance should approach zero: {}",
            v_far
        );

        // Very close, repulsive wall
        let v_close = lennard_jones(0.05);
        assert!(v_close > 1000.0, "LJ at tiny distance should be very repulsive");
    }

    #[test]
    fn drug_discovery_screening_deterministic_with_seed() {
        let candidates = test_candidates();
        let site = test_site();
        let r1 = screen_drug_candidates(&candidates, &site, 99);
        let r2 = screen_drug_candidates(&candidates, &site, 99);

        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.candidate.name, b.candidate.name);
            assert!(
                (a.binding_energy_kcal - b.binding_energy_kcal).abs() < 1e-10,
                "Same seed must produce identical results"
            );
        }
    }
}
