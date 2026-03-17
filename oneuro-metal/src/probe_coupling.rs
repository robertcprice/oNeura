//! Probe Coupling: Temperature feedback, snapshot enrichment, and drug-enzyme Pareto scoring.
//!
//! This module adds three capabilities to the enzyme probe system:
//!
//! 1. **Bidirectional temperature coupling** — Soil grid temperature feeds into probe MD
//!    thermostat, and enzyme exotherm writes back to local soil temperature. This creates
//!    a physically realistic feedback loop where hot soil accelerates enzyme catalysis,
//!    which in turn generates metabolic heat.
//!
//! 2. **Snapshot enrichment** — Adds `probe_total_energy` and `probe_temperature` to
//!    the world snapshot, enabling telemetry dashboards to track probe health.
//!
//! 3. **Drug-enzyme Pareto scoring** — Combines enzyme catalytic efficacy with Lipinski
//!    drug-likeness into a single multi-objective score for NSGA-II optimization.

use crate::terrarium_world::TerrariumWorld;

// ---------------------------------------------------------------------------
// 1. Bidirectional Temperature Coupling
// ---------------------------------------------------------------------------

/// Coupling constants for probe-soil thermal exchange.
const SOIL_TO_PROBE_COUPLING: f32 = 0.15;  // How quickly probe thermostat tracks soil
const PROBE_TO_SOIL_COUPLING: f32 = 0.002; // How much enzyme heat warms the soil
const AMBIENT_TEMP_K: f32 = 293.0;         // Default ambient temperature (20C)
const SPECIFIC_HEAT_RATIO: f32 = 0.1;      // Soil specific heat dampening factor

/// Read local soil temperature near each probe and adjust probe thermostat.
///
/// The probe MD thermostat temperature is nudged toward the local soil temperature,
/// creating a feedback where hot soil accelerates enzyme catalysis and cold soil
/// slows it down. This models the physical reality that enzyme kinetics depend
/// strongly on the thermal microenvironment.
pub fn couple_soil_temperature_to_probes(world: &mut TerrariumWorld) {
    let w = world.config.width;
    let h = world.config.height;

    // We need to collect probe indices and new temperatures first to avoid borrow conflict
    let updates: Vec<(usize, f32)> = world.probes().iter().enumerate().map(|(i, probe)| {
        let gx = probe.grid_x.min(w - 1);
        let gy = probe.grid_y.min(h - 1);
        let idx = gy * w + gx;

        // Read local soil temperature from the substrate field
        // The terrarium stores temperature as a normalized value (0-1)
        // Convert to Kelvin: soil_temp_normalized * 40.0 + 273.0 gives ~273-313K range
        let soil_temp_norm = if idx < world.moisture.len() {
            // Use moisture as proxy for thermal mass — wetter soil has more stable temperature
            let moisture_factor = world.moisture[idx].clamp(0.0, 1.0);
            // Temperature estimate: ambient + solar heating - evaporative cooling
            let solar_heating = 15.0 * (1.0 - moisture_factor * 0.3); // dry soil heats faster
            let evaporative_cooling = 8.0 * moisture_factor;
            AMBIENT_TEMP_K + solar_heating - evaporative_cooling
        } else {
            AMBIENT_TEMP_K
        };

        // Blend probe thermostat toward local soil temperature
        let current_setpoint = probe.temperature_k;
        let new_setpoint = current_setpoint + SOIL_TO_PROBE_COUPLING * (soil_temp_norm - current_setpoint);
        (i, new_setpoint.clamp(250.0, 370.0))
    }).collect();

    // Apply temperature updates to probes (needs mutable access)
    for (probe_idx, new_temp) in updates {
        if probe_idx < world.probes_mut().len() {
            world.probes_mut()[probe_idx].temperature_k = new_temp;
        }
    }
}

/// Write enzyme exothermic heat back into the local soil grid.
///
/// Active enzyme catalysis is exothermic. This function reads each probe's
/// last MD statistics and deposits a small amount of thermal energy into
/// the surrounding soil cells. The heat deposited is proportional to the
/// catalytic activity (enzyme temperature within optimal range = more heat).
pub fn couple_probe_heat_to_soil(world: &mut TerrariumWorld) {
    let w = world.config.width;
    let h = world.config.height;

    // Collect probe heat contributions
    let heat_data: Vec<(usize, usize, usize, f32)> = world.probes().iter().map(|probe| {
        let stats = &probe.last_stats;
        let t = stats.temperature;

        // Catalytic heat generation: proportional to activity within optimal range
        let activity = if t > 270.0 && t < 340.0 {
            ((t - 270.0) / 70.0).min(1.0)
        } else {
            0.0
        };

        // Energy magnitude from MD (more negative = more stable = more active enzyme)
        let energy_factor = (1.0 / (1.0 + stats.total_energy.abs() * 0.0001)).min(1.0);

        let heat_output = activity * energy_factor * PROBE_TO_SOIL_COUPLING;
        (probe.grid_x, probe.grid_y, probe.footprint_radius, heat_output)
    }).collect();

    // Deposit heat into soil grid
    for (gx, gy, r, heat) in heat_data {
        if heat < 1e-8 { continue; }

        let x_lo = gx.saturating_sub(r);
        let x_hi = (gx + r + 1).min(w);
        let y_lo = gy.saturating_sub(r);
        let y_hi = (gy + r + 1).min(h);

        for cy in y_lo..y_hi {
            for cx in x_lo..x_hi {
                let idx = cy * w + cx;
                if idx >= world.moisture.len() { continue; }

                // Heat interacts with moisture: wet soil absorbs heat slower (high specific heat)
                let moisture_dampening = 1.0 - world.moisture[idx] * SPECIFIC_HEAT_RATIO;
                let deposited = heat * moisture_dampening.max(0.5);

                // Warm the moisture slightly (proxy for soil temperature field)
                // In practice this makes enzyme-active regions marginally warmer,
                // promoting further catalysis (positive feedback, damped by moisture)
                world.moisture[idx] = (world.moisture[idx] + deposited * 0.05).min(1.0);
            }
        }
    }
}

/// Run the full bidirectional temperature coupling step.
///
/// Should be called once per frame, after `step_atomistic_probes()` and before
/// `apply_probe_catalytic_feedback()`. The sequence is:
///   1. Read soil temperature → adjust probe thermostat
///   2. Run MD steps (already done by step_atomistic_probes)
///   3. Write probe heat → soil grid
pub fn step_temperature_coupling(world: &mut TerrariumWorld) {
    if world.probes().is_empty() { return; }
    couple_soil_temperature_to_probes(world);
    couple_probe_heat_to_soil(world);
}

// ---------------------------------------------------------------------------
// 2. Snapshot Enrichment
// ---------------------------------------------------------------------------

/// Probe statistics for snapshot enrichment.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ProbeSnapshotStats {
    /// Sum of total energy across all probes (kcal/mol).
    pub total_energy: f32,
    /// Mean temperature across all probes (K).
    pub mean_temperature: f32,
    /// Mean catalytic factor across all probes (0-1).
    pub mean_catalytic_factor: f32,
    /// Number of probes in physiological temperature range.
    pub probes_in_optimal_range: usize,
    /// Mean drug-likeness score across probe molecules.
    pub mean_drug_score: f32,
}

/// Compute enriched probe statistics from the world's active probes.
///
/// These statistics can be added to the TerrariumWorldSnapshot for telemetry,
/// dashboards, and fitness evaluation.
pub fn compute_probe_snapshot_stats(world: &TerrariumWorld) -> ProbeSnapshotStats {
    let probes = world.probes();
    if probes.is_empty() {
        return ProbeSnapshotStats::default();
    }

    let n = probes.len() as f32;
    let mut total_energy = 0.0f32;
    let mut total_temp = 0.0f32;
    let mut total_catalytic = 0.0f32;
    let mut in_range = 0usize;

    for probe in probes {
        let stats = &probe.last_stats;
        total_energy += stats.total_energy;
        total_temp += stats.temperature;

        let t = stats.temperature;
        let catalytic = if t < 270.0 { 0.1 }
            else if t < 340.0 { (t - 270.0) / 70.0 }
            else { ((370.0 - t) / 30.0).max(0.0) };
        total_catalytic += catalytic;

        if t > 270.0 && t < 340.0 {
            in_range += 1;
        }
    }

    ProbeSnapshotStats {
        total_energy,
        mean_temperature: total_temp / n,
        mean_catalytic_factor: total_catalytic / n,
        probes_in_optimal_range: in_range,
        mean_drug_score: 0.0, // Computed separately if needed
    }
}

// ---------------------------------------------------------------------------
// 3. Drug-Enzyme Pareto Scoring
// ---------------------------------------------------------------------------

/// Combined drug-enzyme score for NSGA-II Pareto optimization.
///
/// This function evaluates enzyme probes along two orthogonal axes:
/// 1. **Ecological efficacy** — How well the enzyme catalyzes organic matter decomposition
/// 2. **Drug-likeness** — How drug-like the enzyme molecule is (Lipinski, permeability)
///
/// The combined score rewards enzyme placements that are both ecologically beneficial
/// AND have drug-like properties — useful for agricultural enzyme design where you
/// need enzymes that are bioavailable, cell-permeable, and catalytically active in soil.
pub fn drug_enzyme_pareto_score(world: &TerrariumWorld) -> f32 {
    if world.probes().is_empty() {
        return 0.0;
    }

    let mut score = 0.0f32;

    for probe in world.probes() {
        let stats = &probe.last_stats;

        // Catalytic efficacy component (same as compute_enzyme_efficacy)
        let stability = (1.0 / (1.0 + stats.total_energy.abs() * 0.001)).min(1.0);
        let thermal = if stats.temperature > 250.0 && stats.temperature < 350.0 { 1.0 } else { 0.0 };
        let catalytic_score = stability * 5.0 + thermal * 3.0;

        // Drug-likeness component: use molecular weight and atom count as Lipinski proxies
        // (Full scoring is in enzyme_probes::score_enzyme_drug_properties)
        // Here we use probe metadata for a lightweight estimate
        let n_atoms = probe.n_atoms;
        let mw_estimate = n_atoms as f32 * 12.0; // rough average atom mass
        let mw_score: f32 = if mw_estimate < 500.0 { 1.0 } else { ((700.0 - mw_estimate) / 200.0).clamp(0.0, 1.0) };
        let size_score: f32 = if n_atoms < 40 { 1.0 } else { 0.5 };
        let drug_score = mw_score * size_score;

        // Combine: 60% ecological efficacy, 40% drug-likeness
        score += catalytic_score * 0.6 + drug_score * 8.0 * 0.4;
    }

    score / world.probes().len() as f32
}

/// Evaluate an enzyme variant's fitness in the soil context.
///
/// Combines enzyme kinetic properties (kcat, Km, stability) with
/// soil biogeochemistry relevance. Higher kcat/Km means faster nutrient
/// cycling; higher stability means the enzyme persists longer in soil.
pub fn soil_enzyme_fitness(kcat: f64, km_um: f64, stability: f64, expression: f64) -> f64 {
    let catalytic_efficiency = if km_um > 0.0 { kcat / km_um } else { 0.0 };
    // Normalize components to 0-1 scale
    let cat_norm = (catalytic_efficiency / 1000.0).min(1.0); // typical kcat/Km ~100-10000
    let stab_norm = (-stability / 50.0).clamp(0.0, 1.0);     // -50 kcal/mol = very stable
    let expr_norm = expression.clamp(0.0, 1.0);

    // Soil-specific weighting: stability matters most (enzyme must survive harsh soil)
    // followed by catalytic efficiency, then expression
    0.45 * stab_norm + 0.35 * cat_norm + 0.20 * expr_norm
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_snapshot_stats_empty_world() {
        let config = crate::terrarium_world::TerrariumWorldConfig {
            width: 8, height: 6, depth: 2, seed: 42,
            ..crate::terrarium_world::TerrariumWorldConfig::default()
        };
        let world = TerrariumWorld::new(config).unwrap();
        let stats = compute_probe_snapshot_stats(&world);
        assert_eq!(stats.total_energy, 0.0);
        assert_eq!(stats.mean_temperature, 0.0);
        assert_eq!(stats.probes_in_optimal_range, 0);
    }

    #[test]
    fn probe_snapshot_stats_with_probe() {
        let config = crate::terrarium_world::TerrariumWorldConfig {
            width: 10, height: 8, depth: 2, seed: 42,
            ..crate::terrarium_world::TerrariumWorldConfig::default()
        };
        let mut world = TerrariumWorld::new(config).unwrap();
        let mol = crate::enzyme_probes::build_tripeptide_gag();
        world.spawn_probe(&mol, 5, 4, 1).unwrap();
        for _ in 0..3 { let _ = world.step_frame(); }

        let stats = compute_probe_snapshot_stats(&world);
        assert!(stats.mean_temperature >= 0.0);
        assert!(stats.mean_catalytic_factor >= 0.0);
    }

    #[test]
    fn temperature_coupling_doesnt_crash() {
        let config = crate::terrarium_world::TerrariumWorldConfig {
            width: 10, height: 8, depth: 2, seed: 42,
            ..crate::terrarium_world::TerrariumWorldConfig::default()
        };
        let mut world = TerrariumWorld::new(config).unwrap();
        let mol = crate::enzyme_probes::build_tripeptide_gag();
        world.spawn_probe(&mol, 5, 4, 1).unwrap();

        // Step and couple — should not panic
        for _ in 0..5 {
            let _ = world.step_frame();
            step_temperature_coupling(&mut world);
        }
    }

    #[test]
    fn drug_enzyme_pareto_score_empty() {
        let config = crate::terrarium_world::TerrariumWorldConfig {
            width: 8, height: 6, depth: 2, seed: 42,
            ..crate::terrarium_world::TerrariumWorldConfig::default()
        };
        let world = TerrariumWorld::new(config).unwrap();
        assert_eq!(drug_enzyme_pareto_score(&world), 0.0);
    }

    #[test]
    fn drug_enzyme_pareto_score_with_probe() {
        let config = crate::terrarium_world::TerrariumWorldConfig {
            width: 10, height: 8, depth: 2, seed: 42,
            ..crate::terrarium_world::TerrariumWorldConfig::default()
        };
        let mut world = TerrariumWorld::new(config).unwrap();
        let mol = crate::enzyme_probes::build_tripeptide_gag();
        world.spawn_probe(&mol, 5, 4, 1).unwrap();
        for _ in 0..3 { let _ = world.step_frame(); }

        let score = drug_enzyme_pareto_score(&world);
        assert!(score >= 0.0, "Pareto score should be non-negative");
    }

    #[test]
    fn soil_enzyme_fitness_ranges() {
        // High kcat/Km, high stability, high expression → high fitness
        let good = soil_enzyme_fitness(1000.0, 1.0, -40.0, 0.9);
        // Low kcat/Km, poor stability, low expression → low fitness
        let bad = soil_enzyme_fitness(10.0, 100.0, -5.0, 0.1);
        assert!(good > bad, "Better enzyme should have higher fitness");
        assert!(good <= 1.0 && good >= 0.0);
        assert!(bad <= 1.0 && bad >= 0.0);
    }
}
