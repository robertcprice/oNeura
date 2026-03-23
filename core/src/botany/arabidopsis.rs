//! Arabidopsis thaliana explicit root tissue simulation.
//!
//! Models the primary root with cellular resolution, including:
//! - Seven anatomical cell types (epidermis through quiescent center)
//! - PIN-mediated polar auxin transport (reverse fountain model)
//! - Gravitropism via amyloplast sedimentation in columella cells
//! - Water uptake driven by water potential gradients (Ohm's law analogy)
//! - Dual-affinity nitrate transport (NRT1 low-affinity, NRT2 high-affinity)
//! - Lockhart-equation cell elongation
//!
//! Biological parameters are drawn from published Arabidopsis literature; see
//! inline citations for sources.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants — all values cite primary literature
// ---------------------------------------------------------------------------

/// PIN efflux carrier transport rate (~15 um/s).
/// Kramer (2004) Plant Cell 16:1368-1378; Grieneisen et al. (2007) Nature 449:1008.
const PIN_TRANSPORT_RATE_UM_S: f32 = 15.0;

/// Passive auxin diffusion coefficient in plant tissue (~300 um^2/s).
/// Kramer (2006) Trends Plant Sci 11:382-386.
const AUXIN_DIFFUSION_COEFF: f32 = 300.0;

/// Auxin decay rate constant (s^-1). Half-life ~25 min in root tissue.
/// Ljung et al. (2005) Plant Cell 17:1090-1104.
const AUXIN_DECAY_RATE: f32 = 0.00046; // ln(2) / (25 * 60)

/// Typical cell length in the elongation zone (~15 um before elongation).
/// Verbelen et al. (2006) New Phytol 170:723-732.
const CELL_LENGTH_UM: f32 = 15.0;

/// Lockhart yield threshold for Arabidopsis root cells (MPa).
/// Cosgrove (2005) Nat Rev Mol Cell Biol 6:850-861.
const WALL_YIELD_THRESHOLD_MPA: f32 = 0.3;

/// Default cell wall extensibility (MPa^-1 h^-1).
/// Cosgrove (2005). Typical range 0.5-5.0 for growing cells.
const DEFAULT_EXTENSIBILITY: f32 = 2.0;

/// Hydraulic conductivity of root tissue (m s^-1 MPa^-1).
/// Steudle (2000) J Exp Bot 51:1531-1542.
const ROOT_HYDRAULIC_CONDUCTIVITY: f32 = 1.5e-7;

/// NRT1 (low-affinity) Km for nitrate (mM).
/// Liu et al. (1999) Plant Cell 11:865-874; Tsay et al. (1993) Cell 72:705-713.
const NRT1_KM_MM: f32 = 4.0;

/// NRT1 Vmax (umol g^-1 FW h^-1).
/// Liu et al. (1999).
const NRT1_VMAX: f32 = 80.0;

/// NRT2 (high-affinity) Km for nitrate (mM).
/// Filleur et al. (2001) J Exp Bot 52:2243-2250.
const NRT2_KM_MM: f32 = 0.05;

/// NRT2 Vmax (umol g^-1 FW h^-1).
/// Filleur et al. (2001).
const NRT2_VMAX: f32 = 8.0;

/// Gravitropic response gain.  Arabidopsis primary root bends at ~40 deg/hr
/// when displaced 90 deg from vertical.
/// Mullen et al. (2000) Plant Physiol 123:1089-1098.
const GRAVITROPISM_GAIN_DEG_PER_HR: f32 = 40.0;

/// Auxin concentration that half-maximally inhibits elongation (nM).
/// Based on dose-response from Thimann (1939) and Arabidopsis data in
/// Evans et al. (1994) Planta 194:215-222.
const _AUXIN_INHIBITION_KD_NM: f32 = 50.0;

/// Cell division interval in the quiescent center (~24 h).
/// Dolan et al. (1993) Development 119:71-84.
const _QC_DIVISION_INTERVAL_HR: f32 = 24.0;

/// Meristematic cell division interval (~12 h for transit-amplifying cells).
/// Beemster & Baskin (1998) Plant Physiol 116:1515-1526.
const MERISTEM_DIVISION_INTERVAL_HR: f32 = 12.0;

/// Maximum elongation rate in the elongation zone (um/hr).
/// Beemster & Baskin (1998).
const MAX_ELONGATION_RATE_UM_HR: f32 = 300.0;

/// Casparian strip reflection coefficient (0 = fully permeable, 1 = perfect barrier).
/// Steudle (2000) J Exp Bot 51:1531-1542.
const CASPARIAN_REFLECTION_COEFF: f32 = 0.95;

// ---------------------------------------------------------------------------
// Cell types
// ---------------------------------------------------------------------------

/// Anatomical cell types in the Arabidopsis primary root.
///
/// Ordered radially from outside to center, plus the root cap.
/// Matches the canonical description in Dolan et al. (1993)
/// Development 119:71-84.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArabidopsisRootCellType {
    /// Root hairs for water and nutrient absorption from soil.
    Epidermis,
    /// Parenchyma cells for radial solute transport and storage.
    Cortex,
    /// Barrier tissue with Casparian strip controlling apoplastic flow.
    Endodermis,
    /// Origin of lateral root primordia; retains division competence.
    Pericycle,
    /// Xylem and phloem for long-distance transport of water and sugars.
    Vasculature,
    /// Gravity-sensing cells at the root tip containing amyloplast statoliths.
    RootCapColumella,
    /// Organizer of the root stem cell niche; rarely divides.
    QuiescentCenter,
}

// ---------------------------------------------------------------------------
// Per-cell state
// ---------------------------------------------------------------------------

/// State for a single Arabidopsis root cell.
///
/// Concentrations, pressures, and growth rates are in physiological units
/// to allow direct comparison with published experimental data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArabidopsisRootCell {
    /// Anatomical identity of this cell.
    pub cell_type: ArabidopsisRootCellType,

    /// Auxin (indole-3-acetic acid) concentration in nM.
    /// Typical root-tip maximum is ~500 nM; background ~5-20 nM.
    pub auxin_concentration: f32,

    /// Water potential (MPa). Negative in plant tissue; soil is typically
    /// -0.01 to -1.5 MPa depending on moisture.
    pub water_potential_mpa: f32,

    /// Turgor pressure (MPa). Drives cell expansion when above yield threshold.
    pub turgor_pressure_mpa: f32,

    /// Nitrate absorption rate for this cell (umol g^-1 FW h^-1).
    pub nitrate_uptake_rate: f32,

    /// Cell wall extensibility (MPa^-1 h^-1). Higher in the elongation zone.
    pub cell_wall_extensibility: f32,

    /// Current cell elongation rate (um/hr). Product of the Lockhart equation.
    pub growth_rate_um_hr: f32,
}

impl Default for ArabidopsisRootCell {
    fn default() -> Self {
        Self {
            cell_type: ArabidopsisRootCellType::Cortex,
            auxin_concentration: 10.0,
            water_potential_mpa: -0.5,
            turgor_pressure_mpa: 0.5,
            nitrate_uptake_rate: 0.0,
            cell_wall_extensibility: DEFAULT_EXTENSIBILITY,
            growth_rate_um_hr: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Whole-root state
// ---------------------------------------------------------------------------

/// The complete Arabidopsis primary root.
///
/// Cells are stored tip-to-base: index 0 is the root cap/columella,
/// and the last index is the oldest tissue near the root-shoot junction.
/// This ordering matches the auxin flow direction (acropetal in the
/// vasculature, basipetal in the epidermis).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArabidopsisRoot {
    /// Ordered cell array (tip-to-base).
    pub cells: Vec<ArabidopsisRootCell>,

    /// Total root length in mm.
    pub root_length_mm: f32,

    /// Root angle from vertical in degrees (0 = straight down).
    pub root_angle_deg: f32,

    /// Running total of auxin in the root (nM * cell count, diagnostic).
    pub total_auxin: f32,

    /// Cumulative water uptake this timestep (mm^3 s^-1, for coupling to terrarium).
    pub total_water_uptake: f32,

    /// Cumulative nitrate uptake this timestep (umol h^-1).
    pub total_nitrate_uptake: f32,

    /// Length of the elongation zone (mm).
    pub growth_zone_length_mm: f32,

    /// Positions (mm from tip) of lateral root primordia initiated by the pericycle.
    pub lateral_root_primordia: Vec<f32>,

    /// Accumulated time since last cell division (hours).
    pub time_since_division_hr: f32,
}

impl ArabidopsisRoot {
    /// Build a default primary root with the canonical Arabidopsis radial
    /// anatomy repeated along the longitudinal axis.
    ///
    /// `n_longitudinal` sets the number of cell tiers along the root axis.
    /// Each tier contains the 7 cell types, giving `n_longitudinal * 7` total
    /// cells, except that the tip tier replaces the inner cells with columella
    /// and quiescent center.
    pub fn new(n_longitudinal: usize) -> Self {
        let n_long = n_longitudinal.max(3);
        let mut cells = Vec::with_capacity(n_long * 7);

        for tier in 0..n_long {
            if tier == 0 {
                // Root tip tier: columella + QC
                cells.push(ArabidopsisRootCell {
                    cell_type: ArabidopsisRootCellType::RootCapColumella,
                    auxin_concentration: 200.0, // high auxin at the tip
                    ..Default::default()
                });
                cells.push(ArabidopsisRootCell {
                    cell_type: ArabidopsisRootCellType::QuiescentCenter,
                    auxin_concentration: 500.0, // maximum at QC
                    cell_wall_extensibility: 0.1, // barely grows
                    ..Default::default()
                });
            } else {
                // Standard radial anatomy from outside in
                let is_elongation_zone = tier >= 2 && tier < n_long / 2;
                let base_auxin = if tier < 3 {
                    150.0 - tier as f32 * 40.0
                } else {
                    20.0
                };
                let extensibility = if is_elongation_zone {
                    DEFAULT_EXTENSIBILITY
                } else {
                    0.3
                };

                for cell_type in &[
                    ArabidopsisRootCellType::Epidermis,
                    ArabidopsisRootCellType::Cortex,
                    ArabidopsisRootCellType::Endodermis,
                    ArabidopsisRootCellType::Pericycle,
                    ArabidopsisRootCellType::Vasculature,
                ] {
                    let auxin = match cell_type {
                        // Vasculature carries auxin shootward -> rootward
                        ArabidopsisRootCellType::Vasculature => base_auxin * 1.5,
                        // Epidermis carries the reflux stream
                        ArabidopsisRootCellType::Epidermis => base_auxin * 0.8,
                        _ => base_auxin,
                    };
                    cells.push(ArabidopsisRootCell {
                        cell_type: *cell_type,
                        auxin_concentration: auxin,
                        cell_wall_extensibility: extensibility,
                        ..Default::default()
                    });
                }
            }
        }

        let root_length = n_long as f32 * CELL_LENGTH_UM / 1000.0; // convert um to mm

        Self {
            cells,
            root_length_mm: root_length,
            root_angle_deg: 0.0,
            total_auxin: 0.0,
            total_water_uptake: 0.0,
            total_nitrate_uptake: 0.0,
            growth_zone_length_mm: root_length * 0.4,
            lateral_root_primordia: Vec::new(),
            time_since_division_hr: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Auxin transport — "reverse fountain" model
// ---------------------------------------------------------------------------

/// Advance polar auxin transport by one timestep.
///
/// Implements the reverse fountain model (Grieneisen et al. 2007):
/// - Auxin flows **rootward** through the vasculature via PIN1
/// - At the tip (columella), auxin is redirected laterally
/// - Auxin flows **shootward** through the epidermis/cortex via PIN2
///
/// `dt` is the timestep in seconds.
pub fn auxin_transport_step(cells: &mut [ArabidopsisRootCell], dt: f32) {
    if cells.len() < 4 {
        return;
    }

    // Collect current concentrations so we can compute fluxes without aliasing.
    let old: Vec<f32> = cells.iter().map(|c| c.auxin_concentration).collect();
    let n = old.len();

    // Transport rate scaled to a per-cell basis.
    // PIN velocity (um/s) / cell_length (um) = fraction transported per second.
    let pin_flux_frac = (PIN_TRANSPORT_RATE_UM_S / CELL_LENGTH_UM) * dt;
    let diffusion_frac = (AUXIN_DIFFUSION_COEFF / (CELL_LENGTH_UM * CELL_LENGTH_UM)) * dt;

    for i in 0..n {
        let cell = &cells[i];
        let ct = cell.cell_type;

        // ---- Active PIN-mediated polar transport ----
        match ct {
            // Vasculature: PIN1 drives auxin rootward (toward lower indices = tip).
            ArabidopsisRootCellType::Vasculature => {
                if i >= 2 {
                    // Donate auxin toward the tip (lower index vasculature or QC).
                    let flux = old[i] * pin_flux_frac.min(0.5);
                    cells[i].auxin_concentration -= flux;
                    // Find the nearest tip-side cell (previous vasculature or QC).
                    let target = find_rootward_target(cells, i);
                    cells[target].auxin_concentration += flux;
                }
            }
            // Epidermis: PIN2 drives auxin shootward (toward higher indices).
            ArabidopsisRootCellType::Epidermis => {
                if i + 1 < n {
                    let flux = old[i] * pin_flux_frac.min(0.5);
                    cells[i].auxin_concentration -= flux;
                    let target = find_shootward_target(cells, i);
                    cells[target].auxin_concentration += flux;
                }
            }
            // Columella: redistribute laterally to epidermis (the tip reflux).
            ArabidopsisRootCellType::RootCapColumella => {
                let flux = old[i] * pin_flux_frac.min(0.4) * 0.5;
                if flux > 0.0 {
                    cells[i].auxin_concentration -= flux;
                    // Find nearest epidermis cell to receive the reflux.
                    if let Some(epi_idx) = find_nearest_type(cells, i, ArabidopsisRootCellType::Epidermis) {
                        cells[epi_idx].auxin_concentration += flux;
                    }
                }
            }
            _ => {}
        }

        // ---- Passive diffusion between neighbors ----
        let _diff_flux = old[i] * diffusion_frac.min(0.3);
        if i > 0 && i + 1 < n {
            let to_prev = (old[i] - old[i - 1]).max(0.0) * diffusion_frac.min(0.15);
            let to_next = (old[i] - old[i + 1]).max(0.0) * diffusion_frac.min(0.15);
            cells[i].auxin_concentration -= to_prev + to_next;
            cells[i - 1].auxin_concentration += to_prev;
            cells[i + 1].auxin_concentration += to_next;
        }

        // ---- Metabolic decay (GH3/DAO-mediated IAA conjugation) ----
        cells[i].auxin_concentration -= old[i] * AUXIN_DECAY_RATE * dt;

        // Clamp to non-negative.
        if cells[i].auxin_concentration < 0.0 {
            cells[i].auxin_concentration = 0.0;
        }
    }

    // We deliberately suppress the unused-variable warning for diff_flux
    // which is computed per-cell but consumed in the neighbor block.
    let _ = pin_flux_frac;
}

/// Find the nearest rootward (lower-index) vasculature cell or QC.
fn find_rootward_target(cells: &[ArabidopsisRootCell], from: usize) -> usize {
    for i in (0..from).rev() {
        match cells[i].cell_type {
            ArabidopsisRootCellType::Vasculature
            | ArabidopsisRootCellType::QuiescentCenter
            | ArabidopsisRootCellType::RootCapColumella => return i,
            _ => {}
        }
    }
    0 // fallback to tip
}

/// Find the nearest shootward (higher-index) epidermis cell.
fn find_shootward_target(cells: &[ArabidopsisRootCell], from: usize) -> usize {
    for i in (from + 1)..cells.len() {
        if cells[i].cell_type == ArabidopsisRootCellType::Epidermis {
            return i;
        }
    }
    from // no further epidermis — stay put
}

/// Find the nearest cell of a given type starting from `from`.
fn find_nearest_type(
    cells: &[ArabidopsisRootCell],
    from: usize,
    target_type: ArabidopsisRootCellType,
) -> Option<usize> {
    // Search outward from `from` in both directions.
    let mut lo = from as isize - 1;
    let mut hi = from + 1;
    while lo >= 0 || hi < cells.len() {
        if hi < cells.len() && cells[hi].cell_type == target_type {
            return Some(hi);
        }
        if lo >= 0 && cells[lo as usize].cell_type == target_type {
            return Some(lo as usize);
        }
        lo -= 1;
        hi += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// Gravitropism
// ---------------------------------------------------------------------------

/// Compute the growth-rate differential caused by gravitropism.
///
/// When the root deviates from vertical (`root_angle_deg != 0`), amyloplast
/// sedimentation in columella cells triggers asymmetric auxin redistribution.
/// The resulting auxin asymmetry inhibits elongation on the lower side,
/// causing the root to bend downward.
///
/// Returns the angular correction rate in **degrees per hour** (positive =
/// bending toward vertical).
///
/// # Parameters
/// - `root_angle_deg`: current deviation from vertical (0 = straight down).
/// - `columella_auxin`: auxin concentration in columella cells (nM), which
///   modulates the response magnitude.
///
/// # Biology
/// Mullen et al. (2000) Plant Physiol 123:1089-1098 measured ~40 deg/hr
/// at 90-degree displacement. The response is roughly sinusoidal in the
/// displacement angle, and saturates at high auxin due to receptor saturation.
pub fn gravitropism_response(root_angle_deg: f32, columella_auxin: f32) -> f32 {
    // Sinusoidal displacement sensing (statolith sedimentation is proportional
    // to the sine of the angle from vertical).
    let angle_rad = root_angle_deg.to_radians();
    let displacement_signal = angle_rad.sin();

    // Auxin-dependent gain: columella auxin modulates sensitivity.
    // Half-saturation at ~100 nM; linear below, saturating above.
    let auxin_factor = columella_auxin / (columella_auxin + 100.0);

    // Response magnitude: maximum ~40 deg/hr at 90 degrees.
    GRAVITROPISM_GAIN_DEG_PER_HR * displacement_signal * auxin_factor
}

// ---------------------------------------------------------------------------
// Water uptake
// ---------------------------------------------------------------------------

/// Compute whole-root water uptake rate using the Ohm's law analogy.
///
/// Water flow = conductance * (soil water potential - root water potential).
///
/// The Casparian strip in the endodermis limits apoplastic bypass, so
/// effective conductance is reduced by the reflection coefficient.
///
/// Returns water uptake in mm^3 s^-1 (a volumetric flux suitable for
/// coupling to the terrarium soil water pool).
///
/// # Parameters
/// - `soil_water_potential`: soil psi in MPa (typically -0.01 to -1.5).
/// - `root_cells`: the root cell array.
///
/// # Biology
/// Steudle (2000) J Exp Bot 51:1531-1542.
pub fn water_uptake(soil_water_potential: f32, root_cells: &[ArabidopsisRootCell]) -> f32 {
    if root_cells.is_empty() {
        return 0.0;
    }

    // Average root water potential from all cells.
    let avg_root_psi: f32 =
        root_cells.iter().map(|c| c.water_potential_mpa).sum::<f32>() / root_cells.len() as f32;

    // Driving force: soil must be wetter (less negative) than root for inflow.
    let delta_psi = (soil_water_potential - avg_root_psi).max(0.0);

    // Number of absorbing cells (epidermis).
    let epidermis_count = root_cells
        .iter()
        .filter(|c| c.cell_type == ArabidopsisRootCellType::Epidermis)
        .count() as f32;

    // Has the root passed through an endodermis with intact Casparian strip?
    let has_endodermis = root_cells
        .iter()
        .any(|c| c.cell_type == ArabidopsisRootCellType::Endodermis);
    let barrier_factor = if has_endodermis {
        1.0 - CASPARIAN_REFLECTION_COEFF
    } else {
        1.0
    };

    // Total conductance scales with absorbing surface (epidermis count).
    let conductance = ROOT_HYDRAULIC_CONDUCTIVITY * epidermis_count * barrier_factor;

    // Volumetric flux. The 1e6 converts m to um for dimensional consistency
    // with the cell-count-based surface approximation; the result is in
    // arbitrary volumetric units suitable for the terrarium coupling layer.
    conductance * delta_psi * 1e6
}

// ---------------------------------------------------------------------------
// Nitrate uptake — dual-affinity transport
// ---------------------------------------------------------------------------

/// Compute whole-root nitrate uptake rate using Michaelis-Menten kinetics
/// for both the low-affinity (NRT1/NPF6.3) and high-affinity (NRT2.1)
/// transporter systems.
///
/// Returns total uptake in umol h^-1.
///
/// # Parameters
/// - `soil_nitrate_mm`: external nitrate concentration in mM.
/// - `root_cells`: the root cell array.
///
/// # Biology
/// - NRT1 (low affinity): Km ~4 mM, Vmax ~80 umol/g/h.
///   Tsay et al. (1993) Cell 72:705-713.
/// - NRT2 (high affinity): Km ~0.05 mM, Vmax ~8 umol/g/h.
///   Filleur et al. (2001) J Exp Bot 52:2243-2250.
pub fn nitrate_uptake(soil_nitrate_mm: f32, root_cells: &[ArabidopsisRootCell]) -> f32 {
    if soil_nitrate_mm <= 0.0 || root_cells.is_empty() {
        return 0.0;
    }

    // Count absorbing cells (epidermis are the primary nitrate-absorbing tissue
    // via root hairs, but cortex also participates).
    let absorbing_fraction: f32 = root_cells
        .iter()
        .map(|c| match c.cell_type {
            ArabidopsisRootCellType::Epidermis => 1.0,
            ArabidopsisRootCellType::Cortex => 0.3,
            _ => 0.0,
        })
        .sum();

    // NRT1: low-affinity transporter — dominant at [NO3-] > ~0.5 mM.
    let nrt1_rate = NRT1_VMAX * soil_nitrate_mm / (NRT1_KM_MM + soil_nitrate_mm);

    // NRT2: high-affinity transporter — dominant at [NO3-] < ~0.5 mM.
    let nrt2_rate = NRT2_VMAX * soil_nitrate_mm / (NRT2_KM_MM + soil_nitrate_mm);

    // Total rate scaled by absorbing cell count (each cell ~1 ug fresh weight).
    let per_cell_rate = nrt1_rate + nrt2_rate;

    per_cell_rate * absorbing_fraction
}

// ---------------------------------------------------------------------------
// Root growth — Lockhart equation
// ---------------------------------------------------------------------------

/// Advance root growth by one timestep.
///
/// Cell elongation follows the Lockhart equation:
///   dL/dt = extensibility * (turgor - yield_threshold)   [when turgor > yield]
///
/// Auxin modulates extensibility: moderate auxin promotes growth while
/// excess auxin (as at the root tip) inhibits it. New cells are added
/// from the quiescent center at a species-specific division rate.
///
/// `dt` is in **hours** (matching growth rate units of um/hr).
pub fn root_growth_step(root: &mut ArabidopsisRoot, dt: f32) {
    let mut total_elongation_um = 0.0_f32;

    for cell in root.cells.iter_mut() {
        // Only cells in the elongation zone (non-tip, non-QC, extensibility > threshold)
        // actually grow.
        let turgor_excess = (cell.turgor_pressure_mpa - WALL_YIELD_THRESHOLD_MPA).max(0.0);

        // Auxin modulation: moderate auxin (~10-30 nM) promotes growth.
        // High auxin (>50 nM) progressively inhibits elongation.
        // Low auxin (<5 nM) also reduces growth (optimal window).
        let auxin = cell.auxin_concentration;
        let auxin_promotion = if auxin < 1.0 {
            0.1
        } else {
            // Bell-shaped dose-response: peaks near 15-25 nM, inhibits above.
            let promotion = (auxin / 20.0) * (-((auxin - 20.0) / 40.0).powi(2)).exp();
            promotion.clamp(0.0, 1.0)
        };

        // Lockhart growth rate.
        let raw_rate = cell.cell_wall_extensibility * turgor_excess * auxin_promotion;
        cell.growth_rate_um_hr = raw_rate.min(MAX_ELONGATION_RATE_UM_HR);

        total_elongation_um += cell.growth_rate_um_hr * dt;
    }

    // Update root length (um -> mm).
    root.root_length_mm += total_elongation_um / 1000.0;

    // ---- Cell division from QC / meristem ----
    root.time_since_division_hr += dt;
    if root.time_since_division_hr >= MERISTEM_DIVISION_INTERVAL_HR {
        root.time_since_division_hr -= MERISTEM_DIVISION_INTERVAL_HR;

        // Insert a new tier of cells just above the tip (index 2, after QC+columella).
        let insert_at = 2.min(root.cells.len());
        for cell_type in [
            ArabidopsisRootCellType::Epidermis,
            ArabidopsisRootCellType::Cortex,
            ArabidopsisRootCellType::Endodermis,
            ArabidopsisRootCellType::Pericycle,
            ArabidopsisRootCellType::Vasculature,
        ] {
            root.cells.insert(
                insert_at,
                ArabidopsisRootCell {
                    cell_type,
                    auxin_concentration: 80.0, // near-tip cells have moderate auxin
                    cell_wall_extensibility: DEFAULT_EXTENSIBILITY,
                    ..Default::default()
                },
            );
        }
    }

    // ---- Gravitropism: adjust root angle toward vertical ----
    let columella_auxin = root
        .cells
        .iter()
        .find(|c| c.cell_type == ArabidopsisRootCellType::RootCapColumella)
        .map(|c| c.auxin_concentration)
        .unwrap_or(0.0);

    let angle_correction = gravitropism_response(root.root_angle_deg, columella_auxin) * dt;
    root.root_angle_deg -= angle_correction; // reduce deviation from vertical

    // Clamp angle to [-180, 180].
    root.root_angle_deg = root.root_angle_deg.clamp(-180.0, 180.0);

    // ---- Update diagnostics ----
    root.total_auxin = root.cells.iter().map(|c| c.auxin_concentration).sum();
}

// ---------------------------------------------------------------------------
// Full simulation step
// ---------------------------------------------------------------------------

/// Run one full simulation tick for the root system.
///
/// Integrates auxin transport, water uptake, nitrate uptake, and growth.
///
/// # Parameters
/// - `root`: mutable reference to the root.
/// - `soil_water_potential`: soil psi in MPa.
/// - `soil_nitrate_mm`: external nitrate in mM.
/// - `dt_seconds`: timestep in seconds (auxin transport operates on seconds).
pub fn root_simulation_step(
    root: &mut ArabidopsisRoot,
    soil_water_potential: f32,
    soil_nitrate_mm: f32,
    dt_seconds: f32,
) {
    // 1. Auxin transport (operates on seconds timescale).
    auxin_transport_step(&mut root.cells, dt_seconds);

    // 2. Water uptake.
    root.total_water_uptake = water_uptake(soil_water_potential, &root.cells);

    // Update cell water potentials based on uptake feedback.
    if root.total_water_uptake > 0.0 {
        for cell in root.cells.iter_mut() {
            // Cells hydrate slightly from water inflow.
            cell.water_potential_mpa += root.total_water_uptake * 1e-4;
            cell.water_potential_mpa = cell.water_potential_mpa.min(0.0); // cannot be positive
            // Turgor increases with hydration.
            cell.turgor_pressure_mpa =
                (cell.turgor_pressure_mpa + root.total_water_uptake * 1e-4).clamp(0.0, 1.5);
        }
    }

    // 3. Nitrate uptake.
    root.total_nitrate_uptake = nitrate_uptake(soil_nitrate_mm, &root.cells);

    // 4. Growth (Lockhart equation, operates on hours timescale).
    let dt_hours = dt_seconds / 3600.0;
    root_growth_step(root, dt_hours);
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// After multiple transport steps, auxin should concentrate at the root tip
    /// (QC/columella) due to the reverse fountain model: vasculature delivers
    /// auxin rootward, and the columella/QC region accumulates the maximum.
    #[test]
    fn test_auxin_transport_creates_tip_maximum() {
        let mut root = ArabidopsisRoot::new(10);

        // Set a uniform-ish auxin baseline, then let transport run.
        for cell in root.cells.iter_mut() {
            cell.auxin_concentration = 50.0;
        }

        // Run 100 transport steps at 1-second intervals.
        for _ in 0..100 {
            auxin_transport_step(&mut root.cells, 1.0);
        }

        // The QC / columella cells (indices 0 and 1) should have the highest auxin.
        let tip_auxin: f32 = root.cells[..2]
            .iter()
            .map(|c| c.auxin_concentration)
            .sum::<f32>()
            / 2.0;

        // The mid-root region should have lower auxin.
        let mid_start = root.cells.len() / 2;
        let mid_end = (mid_start + 5).min(root.cells.len());
        let mid_auxin: f32 = root.cells[mid_start..mid_end]
            .iter()
            .map(|c| c.auxin_concentration)
            .sum::<f32>()
            / (mid_end - mid_start) as f32;

        assert!(
            tip_auxin > mid_auxin,
            "Tip auxin ({:.2} nM) should exceed mid-root ({:.2} nM) after transport",
            tip_auxin,
            mid_auxin
        );
    }

    /// A root displaced from vertical should bend back toward vertical.
    /// The gravitropism response should produce a positive correction
    /// (reducing root_angle_deg).
    #[test]
    fn test_gravitropism_bends_root_downward() {
        let mut root = ArabidopsisRoot::new(8);
        root.root_angle_deg = 45.0; // tilted 45 degrees from vertical

        // Give the columella enough auxin to drive a response.
        if let Some(col) = root
            .cells
            .iter_mut()
            .find(|c| c.cell_type == ArabidopsisRootCellType::RootCapColumella)
        {
            col.auxin_concentration = 200.0;
        }

        let initial_angle = root.root_angle_deg;

        // Run growth for 10 steps of 0.1 hours each.
        for _ in 0..10 {
            root_growth_step(&mut root, 0.1);
        }

        assert!(
            root.root_angle_deg < initial_angle,
            "Root should bend toward vertical: angle went from {:.1} to {:.1}",
            initial_angle,
            root.root_angle_deg
        );
        assert!(
            root.root_angle_deg > 0.0,
            "Root should not overshoot past vertical in 1 hour"
        );
    }

    /// Water uptake should follow Ohm's law: higher soil water potential
    /// (less negative = wetter soil) drives greater uptake.
    #[test]
    fn test_water_uptake_follows_gradient() {
        let root = ArabidopsisRoot::new(8);

        // Wet soil: psi = -0.01 MPa (field capacity).
        let uptake_wet = water_uptake(-0.01, &root.cells);

        // Dry soil: psi = -1.0 MPa (approaching wilting point).
        let uptake_dry = water_uptake(-1.0, &root.cells);

        // Very dry soil: psi = -1.5 MPa (permanent wilting point).
        let uptake_wilting = water_uptake(-1.5, &root.cells);

        assert!(
            uptake_wet > uptake_dry,
            "Wet soil ({:.6}) should yield more uptake than dry ({:.6})",
            uptake_wet,
            uptake_dry
        );
        assert!(
            uptake_dry >= uptake_wilting,
            "Dry soil ({:.6}) should yield >= uptake than wilting ({:.6})",
            uptake_dry,
            uptake_wilting
        );
        assert!(uptake_wet > 0.0, "Uptake from wet soil should be positive");
    }

    /// Nitrate uptake should show the dual-affinity kinetics:
    /// - At low [NO3-] (<0.1 mM), NRT2 dominates (high affinity).
    /// - At high [NO3-] (>5 mM), NRT1 dominates (low affinity, high capacity).
    /// - The combined curve should be monotonically increasing and saturating.
    #[test]
    fn test_nitrate_uptake_dual_affinity() {
        let root = ArabidopsisRoot::new(8);

        let uptake_low = nitrate_uptake(0.01, &root.cells); // 10 uM
        let uptake_mid = nitrate_uptake(0.5, &root.cells); // 500 uM
        let uptake_high = nitrate_uptake(10.0, &root.cells); // 10 mM
        let uptake_vhigh = nitrate_uptake(50.0, &root.cells); // 50 mM

        // Should be monotonically increasing.
        assert!(
            uptake_mid > uptake_low,
            "Mid ({:.4}) > Low ({:.4})",
            uptake_mid,
            uptake_low
        );
        assert!(
            uptake_high > uptake_mid,
            "High ({:.4}) > Mid ({:.4})",
            uptake_high,
            uptake_mid
        );
        assert!(
            uptake_vhigh > uptake_high,
            "VHigh ({:.4}) > High ({:.4})",
            uptake_vhigh,
            uptake_high
        );

        // At very low concentration, NRT2 should contribute significantly.
        // Compute NRT2's fractional contribution at 10 uM.
        let nrt2_at_low = NRT2_VMAX * 0.01 / (NRT2_KM_MM + 0.01);
        let nrt1_at_low = NRT1_VMAX * 0.01 / (NRT1_KM_MM + 0.01);
        let nrt2_fraction = nrt2_at_low / (nrt1_at_low + nrt2_at_low);
        assert!(
            nrt2_fraction > 0.5,
            "At 10 uM, NRT2 should dominate (fraction = {:.3})",
            nrt2_fraction
        );

        // Saturation check: doubling from 50 to 100 mM should give diminishing returns.
        let uptake_100 = nitrate_uptake(100.0, &root.cells);
        let marginal = (uptake_100 - uptake_vhigh) / uptake_vhigh;
        assert!(
            marginal < 0.5,
            "Saturation: marginal gain at 100 mM vs 50 mM should be < 50%, got {:.1}%",
            marginal * 100.0
        );
    }

    /// Growth rate should follow the Lockhart equation:
    ///   rate = extensibility * (turgor - yield_threshold)
    /// Cells with turgor below yield threshold should not grow.
    #[test]
    fn test_root_growth_lockhart() {
        let mut root = ArabidopsisRoot::new(6);

        // Set up a controlled scenario: one cell with known parameters.
        // Find the first epidermis cell in the elongation zone.
        let target_idx = root
            .cells
            .iter()
            .position(|c| {
                c.cell_type == ArabidopsisRootCellType::Epidermis
                    && c.cell_wall_extensibility > 1.0
            })
            .expect("Should have an extensible epidermis cell");

        // Set precise values for Lockhart testing.
        root.cells[target_idx].turgor_pressure_mpa = 0.6; // above yield threshold of 0.3
        root.cells[target_idx].cell_wall_extensibility = 2.0;
        root.cells[target_idx].auxin_concentration = 20.0; // near-optimal for growth

        // Also test a non-growing cell: turgor below yield.
        let non_growing_idx = root
            .cells
            .iter()
            .position(|c| {
                c.cell_type == ArabidopsisRootCellType::Cortex && c.cell_wall_extensibility > 1.0
            })
            .expect("Should have an extensible cortex cell");
        root.cells[non_growing_idx].turgor_pressure_mpa = 0.2; // below yield threshold
        root.cells[non_growing_idx].cell_wall_extensibility = 2.0;
        root.cells[non_growing_idx].auxin_concentration = 20.0;

        let initial_length = root.root_length_mm;

        // Grow for one step of 0.1 hours.
        root_growth_step(&mut root, 0.1);

        // The growing cell should have a positive growth rate.
        assert!(
            root.cells[target_idx].growth_rate_um_hr > 0.0,
            "Cell above yield threshold should grow, rate = {:.3}",
            root.cells[target_idx].growth_rate_um_hr
        );

        // The non-growing cell should have zero growth.
        assert!(
            root.cells[non_growing_idx].growth_rate_um_hr == 0.0,
            "Cell below yield threshold should NOT grow, rate = {:.3}",
            root.cells[non_growing_idx].growth_rate_um_hr
        );

        // Root length should increase.
        assert!(
            root.root_length_mm > initial_length,
            "Root should have elongated: {:.4} -> {:.4}",
            initial_length,
            root.root_length_mm
        );
    }

    /// Verify that cell division adds new cells to the root over time.
    #[test]
    fn test_cell_division_adds_cells() {
        let mut root = ArabidopsisRoot::new(6);
        let initial_count = root.cells.len();

        // Advance 24 hours (should trigger at least one division at 12h interval).
        let dt_hr = 0.5;
        for _ in 0..(24.0 / dt_hr) as usize {
            root_growth_step(&mut root, dt_hr);
        }

        assert!(
            root.cells.len() > initial_count,
            "After 24 hours, root should have more cells: {} -> {}",
            initial_count,
            root.cells.len()
        );
    }

    /// Full simulation step should integrate all subsystems without panicking,
    /// and should produce non-degenerate state.
    #[test]
    fn test_full_simulation_step_integration() {
        let mut root = ArabidopsisRoot::new(10);
        root.root_angle_deg = 30.0;

        // Run 60 seconds of simulation.
        for _ in 0..60 {
            root_simulation_step(&mut root, -0.1, 1.0, 1.0);
        }

        assert!(root.total_auxin > 0.0, "Total auxin should be positive");
        assert!(
            root.total_water_uptake > 0.0,
            "Water uptake should be positive for wet soil"
        );
        assert!(
            root.total_nitrate_uptake > 0.0,
            "Nitrate uptake should be positive"
        );
        assert!(
            root.root_angle_deg < 30.0,
            "Root should have bent toward vertical"
        );
    }

    /// Serialization roundtrip for root state.
    #[test]
    fn test_root_serde_roundtrip() {
        let root = ArabidopsisRoot::new(6);
        let json = serde_json::to_string(&root).expect("should serialize");
        let deser: ArabidopsisRoot = serde_json::from_str(&json).expect("should deserialize");

        assert_eq!(root.cells.len(), deser.cells.len());
        assert!((root.root_length_mm - deser.root_length_mm).abs() < 1e-6);
    }

    /// Verify the constructor builds the expected cell type distribution.
    #[test]
    fn test_root_anatomy_structure() {
        let root = ArabidopsisRoot::new(8);

        // Should have exactly 1 columella and 1 QC.
        let columella_count = root
            .cells
            .iter()
            .filter(|c| c.cell_type == ArabidopsisRootCellType::RootCapColumella)
            .count();
        let qc_count = root
            .cells
            .iter()
            .filter(|c| c.cell_type == ArabidopsisRootCellType::QuiescentCenter)
            .count();

        assert_eq!(columella_count, 1, "Should have exactly 1 columella cell");
        assert_eq!(qc_count, 1, "Should have exactly 1 quiescent center cell");

        // Each non-tip tier should have 5 cell types.
        let non_tip_cells = root.cells.len() - 2; // minus columella + QC
        assert_eq!(
            non_tip_cells % 5,
            0,
            "Non-tip cells ({}) should be a multiple of 5 (one per radial type)",
            non_tip_cells
        );

        // Epidermis count should equal the number of non-tip tiers.
        let epi_count = root
            .cells
            .iter()
            .filter(|c| c.cell_type == ArabidopsisRootCellType::Epidermis)
            .count();
        assert_eq!(
            epi_count,
            (non_tip_cells / 5),
            "Each tier should have one epidermis cell"
        );
    }

    /// Water uptake should be zero when root is drier than soil (reversed gradient).
    #[test]
    fn test_water_uptake_zero_when_gradient_reversed() {
        let mut root = ArabidopsisRoot::new(6);
        // Make root very wet (close to zero).
        for cell in root.cells.iter_mut() {
            cell.water_potential_mpa = -0.01;
        }
        // Soil is drier than root.
        let uptake = water_uptake(-1.0, &root.cells);
        assert!(
            uptake == 0.0,
            "No water should flow from dry soil to wet root, got {:.6}",
            uptake
        );
    }
}
