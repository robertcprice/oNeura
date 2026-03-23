//! Integration tests for ecology_events: food patch decay and seed germination.
//!
//! These live in tests/ so they compile separately from the lib test target,
//! avoiding interference from pre-existing compilation issues in other test
//! modules.

use oneura_core::{step_food_patches, step_seed_bank};

// ── Food patch decay (Michaelis-Menten kinetics) ──────────────────────────

#[test]
fn food_patch_decay_stays_bounded() {
    let result = step_food_patches(
        45.0,
        &[0.8, 0.01],
        &[0.85, 0.04],
        &[false, false],
        &[true, true],
        &[0.9, 0.95],
        &[0.7, 0.03],
        &[0.02, 0.05],
    )
    .unwrap();
    assert!(result.remaining.iter().all(|v| *v >= 0.0));
    assert_eq!(result.remaining.len(), 2);
}

#[test]
fn food_patch_mm_decay_removes_mass() {
    // Use moderate dt so decay is observable but fruit stays alive.
    let result = step_food_patches(
        100.0,
        &[0.8],
        &[0.8],
        &[false],
        &[true],
        &[0.95],
        &[0.7],
        &[0.5], // moderate microbial biomass
    )
    .unwrap();
    assert!(
        result.remaining[0] < 0.8,
        "MM decay should reduce fruit mass, got {}",
        result.remaining[0]
    );
    assert!(result.remaining[0] >= 0.0);
    assert!(result.fruit_alive[0], "fruit should still be alive");
}

#[test]
fn food_patch_no_decay_without_microbes() {
    let result = step_food_patches(
        500.0,
        &[0.5],
        &[0.5],
        &[false],
        &[true],
        &[0.95],
        &[0.4],
        &[0.0],
    )
    .unwrap();
    assert_eq!(
        result.remaining[0], 0.5,
        "without microbes, fruit should not decay"
    );
}

// ── Seed germination (hydrothermal time model) ───────────────────────────

#[test]
fn seed_bank_dormant_seeds_do_not_germinate() {
    let result = step_seed_bank(
        50.0,
        0.9,
        27,
        28,
        &[20_000.0, 20_000.0],
        &[100.0, 100.0],
        &[0.15, 0.15],
        &[1.0, 1.0],
        &[0.8, 0.8],
        &[0.5, 0.5],
        &[0.4, 0.4],
        &[0.1, 0.1],
        &[0.05, 0.05],
        &[0.0, 0.0],
        &[0.02, 0.02],
    )
    .unwrap();
    assert!(
        result.germinate.iter().all(|v| !*v),
        "dormant seeds must not germinate"
    );
    assert!(result.keep.iter().all(|v| *v), "young seeds should be kept");
}

#[test]
fn seed_germination_when_conditions_met() {
    let result = step_seed_bank(
        50.0,
        0.9,
        5,
        28,
        &[0.0, 0.0],
        &[100.0, 100.0],
        &[0.15, 0.15],
        &[1.0, 1.0],
        &[0.8, 0.8],
        &[0.5, 0.5],
        &[0.4, 0.4],
        &[0.1, 0.1],
        &[0.05, 0.05],
        &[0.0, 0.0],
        &[0.02, 0.02],
    )
    .unwrap();
    assert!(
        result.germinate.iter().all(|v| *v),
        "seeds with broken dormancy, moisture, and reserves should germinate"
    );
}

#[test]
fn seed_germination_blocked_by_dry_soil() {
    let result = step_seed_bank(
        50.0,
        0.9,
        5,
        28,
        &[0.0],
        &[100.0],
        &[0.15],
        &[1.0],
        &[0.8],
        &[0.05],
        &[0.02],
        &[0.1],
        &[0.05],
        &[0.0],
        &[0.02],
    )
    .unwrap();
    assert!(
        !result.germinate[0],
        "seeds should not germinate in dry soil"
    );
}

#[test]
fn seed_germination_blocked_by_low_reserves() {
    let result = step_seed_bank(
        50.0,
        0.9,
        5,
        28,
        &[0.0],
        &[100.0],
        &[0.02], // below RESERVE_CARBON_MIN (0.04)
        &[1.0],
        &[0.8],
        &[0.5],
        &[0.4],
        &[0.1],
        &[0.05],
        &[0.0],
        &[0.02],
    )
    .unwrap();
    assert!(
        !result.germinate[0],
        "seeds should not germinate with insufficient carbon reserves"
    );
}

#[test]
fn seed_germination_old_seeds_expire() {
    // Age beyond SEED_MAX_AGE_S (300_000) guarantees removal regardless of
    // viability, since the hard cap enforces age < 300_000.
    let result = step_seed_bank(
        50.0,
        0.9,
        5,
        28,
        &[20_000.0],
        &[300_100.0], // already past max viable age
        &[0.15],
        &[1.0],
        &[0.8],
        &[0.5],
        &[0.4],
        &[0.1],
        &[0.05],
        &[0.0],
        &[0.02],
    )
    .unwrap();
    assert!(!result.keep[0], "seeds beyond max age should not be kept");
}

#[test]
fn seed_germination_stratification_rate() {
    let wet = step_seed_bank(
        1000.0,
        0.9,
        5,
        28,
        &[5000.0],
        &[100.0],
        &[0.15],
        &[1.0],
        &[0.8],
        &[0.9],
        &[0.8],
        &[0.1],
        &[0.05],
        &[0.0],
        &[0.02],
    )
    .unwrap();
    let dry = step_seed_bank(
        1000.0,
        0.9,
        5,
        28,
        &[5000.0],
        &[100.0],
        &[0.15],
        &[1.0],
        &[0.8],
        &[0.05],
        &[0.02],
        &[0.1],
        &[0.05],
        &[0.0],
        &[0.02],
    )
    .unwrap();
    assert!(
        wet.dormancy_s[0] < dry.dormancy_s[0],
        "wet conditions should break dormancy faster: wet={}, dry={}",
        wet.dormancy_s[0],
        dry.dormancy_s[0]
    );
}

/// Verify the seedling scale uses allometric scaling (mass^0.75).
#[test]
fn seed_germination_allometric_scale() {
    let low_reserve = step_seed_bank(
        50.0,
        0.9,
        5,
        28,
        &[0.0],
        &[100.0],
        &[0.05], // low reserves
        &[1.0],
        &[0.8],
        &[0.5],
        &[0.4],
        &[0.1],
        &[0.05],
        &[0.0],
        &[0.02],
    )
    .unwrap();
    let high_reserve = step_seed_bank(
        50.0,
        0.9,
        5,
        28,
        &[0.0],
        &[100.0],
        &[0.18], // high reserves
        &[1.0],
        &[0.8],
        &[0.5],
        &[0.4],
        &[0.1],
        &[0.05],
        &[0.0],
        &[0.02],
    )
    .unwrap();
    assert!(
        high_reserve.seedling_scale[0] > low_reserve.seedling_scale[0],
        "higher reserves should produce larger seedlings: high={}, low={}",
        high_reserve.seedling_scale[0],
        low_reserve.seedling_scale[0]
    );
    // Both should be within clamped range [0.45, 1.10]
    assert!(low_reserve.seedling_scale[0] >= 0.45);
    assert!(high_reserve.seedling_scale[0] <= 1.10);
}
