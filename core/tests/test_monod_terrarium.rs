#![cfg(feature = "satellite_tests")]
//! Integration tests for Monod population dynamics in the terrarium.
//!
//! These live in tests/ so they compile separately from the lib test target,
//! avoiding interference from pre-existing compilation issues in other test
//! modules.

use oneura_core::{BatchedAtomTerrarium, TerrariumSpecies};

// ── substrate_stays_bounded (mirrors the in-crate test) ─────────────────

#[test]
fn monod_substrate_stays_bounded() {
    let mut terrarium = BatchedAtomTerrarium::new(12, 12, 8, 0.5, false);
    terrarium.run(200, 0.25);
    let snap = terrarium.snapshot();
    assert!(
        snap.mean_glucose >= 0.0,
        "glucose went negative: {}",
        snap.mean_glucose
    );
    assert!(
        snap.mean_oxygen_gas >= 0.0,
        "O2 went negative: {}",
        snap.mean_oxygen_gas
    );
    assert!(
        snap.mean_atp_flux >= 0.0,
        "ATP went negative: {}",
        snap.mean_atp_flux
    );
    assert!(
        snap.mean_hydration > 0.0,
        "hydration went to zero: {}",
        snap.mean_hydration
    );
    // Activity fields should stay in the Monod-allowed range [0.001, 5.0].
    assert!(
        snap.mean_microbes >= 0.001,
        "heterotroph activity collapsed: {}",
        snap.mean_microbes
    );
    assert!(
        snap.mean_microbes <= 5.0,
        "heterotroph activity exploded: {}",
        snap.mean_microbes
    );
}

// ── guild_activity_controls_nitrogen_transforms (mirrors in-crate test) ─

#[test]
fn monod_guild_activity_controls_nitrogen_transforms() {
    let total = 6 * 6 * 4;

    // Nitrifier-dominant scenario: high NH4, high O2, high nitrifier activity.
    let mut nitrify = BatchedAtomTerrarium::new(6, 6, 4, 0.5, false);
    nitrify.fill_species(TerrariumSpecies::Ammonium, 0.55);
    nitrify.fill_species(TerrariumSpecies::Nitrate, 0.04);
    nitrify.fill_species(TerrariumSpecies::OxygenGas, 0.95);
    nitrify.fill_species(TerrariumSpecies::Glucose, 0.18);
    nitrify.set_hydration_field(&vec![0.72; total]).unwrap();
    nitrify
        .set_microbial_activity_field(&vec![0.08; total])
        .unwrap();
    nitrify
        .set_nitrifier_activity_field(&vec![1.0; total])
        .unwrap();
    nitrify
        .set_denitrifier_activity_field(&vec![0.0; total])
        .unwrap();

    let nitrify_nitrate_before = nitrify.mean_species(TerrariumSpecies::Nitrate);
    let nitrify_ammonium_before = nitrify.mean_species(TerrariumSpecies::Ammonium);
    nitrify.run(160, 0.25);
    assert!(
        nitrify.mean_species(TerrariumSpecies::Nitrate) > nitrify_nitrate_before,
        "nitrifier-dominant substrate did not raise nitrate: before={}, after={}",
        nitrify_nitrate_before,
        nitrify.mean_species(TerrariumSpecies::Nitrate)
    );
    assert!(
        nitrify.mean_species(TerrariumSpecies::Ammonium) < nitrify_ammonium_before,
        "nitrifier-dominant substrate did not lower ammonium: before={}, after={}",
        nitrify_ammonium_before,
        nitrify.mean_species(TerrariumSpecies::Ammonium)
    );

    // Denitrifier-dominant scenario: high NO3, low O2, high denitrifier activity.
    let mut denit = BatchedAtomTerrarium::new(6, 6, 4, 0.5, false);
    denit.fill_species(TerrariumSpecies::Ammonium, 0.06);
    denit.fill_species(TerrariumSpecies::Nitrate, 0.52);
    denit.fill_species(TerrariumSpecies::OxygenGas, 0.03);
    denit.fill_species(TerrariumSpecies::Glucose, 0.24);
    denit.set_hydration_field(&vec![0.92; total]).unwrap();
    denit
        .set_microbial_activity_field(&vec![0.10; total])
        .unwrap();
    denit
        .set_nitrifier_activity_field(&vec![0.0; total])
        .unwrap();
    denit
        .set_denitrifier_activity_field(&vec![1.0; total])
        .unwrap();

    let denit_nitrate_before = denit.mean_species(TerrariumSpecies::Nitrate);
    denit.run(160, 0.25);
    assert!(
        denit.mean_species(TerrariumSpecies::Nitrate) < denit_nitrate_before,
        "denitrifier-dominant substrate did not lower nitrate: before={}, after={}",
        denit_nitrate_before,
        denit.mean_species(TerrariumSpecies::Nitrate)
    );
}

// ── Monod-specific: starved voxel activity decays ───────────────────────

#[test]
fn monod_starved_heterotrophs_decay() {
    // A terrarium with zero glucose AND zero carbon (no mineralization source)
    // AND zero O2 should see heterotroph activity decline because Monod
    // growth rate stays at zero while death/maintenance consume biomass.
    //
    // IMPORTANT: we must NOT call set_hydration_field() or any set_*_field()
    // method because those enable external_controls which bypasses Monod.
    // fill_species() does NOT set external_controls.
    //
    // Monod rates are in h^-1, so we need biologically meaningful simulated
    // time.  4000 steps at 1000 ms = 4000 s ~ 1.1 hours.  With combined
    // death + maintenance rate of 0.015 h^-1 the expected decay is
    // exp(-0.015 * 1.1) ~ 0.984, i.e. ~1.6 % loss -- small but detectable.
    let mut t = BatchedAtomTerrarium::new(6, 6, 4, 0.5, false);
    // Remove all substrate sources that could regenerate glucose.
    t.fill_species(TerrariumSpecies::Glucose, 0.0);
    t.fill_species(TerrariumSpecies::Carbon, 0.0); // no mineralization
    t.fill_species(TerrariumSpecies::CarbonDioxide, 0.0); // no photosynthesis
    t.fill_species(TerrariumSpecies::OxygenGas, 0.0); // no O2 for growth
    t.fill_species(TerrariumSpecies::Water, 0.0); // no photosynthesis
    let initial_activity = t.snapshot().mean_microbes;
    assert!(
        initial_activity > 0.01,
        "initial activity too low to test: {}",
        initial_activity
    );
    t.run(4000, 1000.0);
    let final_activity = t.snapshot().mean_microbes;
    // With zero glucose AND zero O2, heterotroph Monod growth is 0.
    // Death + maintenance = 0.015 h^-1 drives decay.
    assert!(
        final_activity < initial_activity,
        "starved heterotroph activity did not decay: before={}, after={}",
        initial_activity,
        final_activity
    );
}

// ── Monod-specific: fed voxels maintain or grow activity ────────────────

#[test]
fn monod_fed_heterotrophs_do_not_collapse() {
    // A terrarium with abundant glucose and O2 should maintain or grow
    // heterotroph activity above the dormant floor.
    let mut t = BatchedAtomTerrarium::new(6, 6, 4, 0.5, false);
    t.fill_species(TerrariumSpecies::Glucose, 1.0);
    t.fill_species(TerrariumSpecies::OxygenGas, 0.8);
    t.run(200, 0.25);
    let activity = t.snapshot().mean_microbes;
    assert!(
        activity > 0.01,
        "fed heterotroph activity collapsed to near-zero: {}",
        activity
    );
}
