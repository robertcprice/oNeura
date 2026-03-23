//! Latent guild bank stepping — per-cell mutation of latent strain bank traits.
//!
//! Each microbial guild (general microbes, nitrifiers, denitrifiers) maintains
//! two "latent banks" of dormant strains alongside the active population.
//! This module implements the per-timestep evolution of latent bank traits
//! via Gaussian perturbation, relaxation toward primary trait means, and
//! dormancy-modulated selection pressure.
//!
//! Used by `terrarium_world/biomechanics.rs` when the `terrarium_advanced`
//! feature is enabled.

/// Borrowed references into per-cell guild state for a single guild.
///
/// The two-element arrays represent the two latent banks (shadow + variant).
pub struct LatentGuildState<'a> {
    /// Per-cell total microbial packets for this guild.
    pub total_packets: &'a [f32],
    /// Per-cell secondary genotype packet counts (from PublicSecondaryBanks).
    pub public_secondary_packets: [&'a [f32]; 3],
    /// Per-cell secondary trait A values (from PublicSecondaryBanks).
    pub public_secondary_trait_a: [&'a [f32]; 3],
    /// Per-cell secondary trait B values (from PublicSecondaryBanks).
    pub public_secondary_trait_b: [&'a [f32]; 3],
    /// Per-cell mutation flux for this guild.
    pub mutation_flux: &'a [f32],
    /// Per-cell vitality for this guild.
    pub vitality: &'a [f32],
    /// Per-cell dormancy fraction for this guild.
    pub dormancy: &'a [f32],
    /// Per-cell primary trait A (e.g., yield, oxygen affinity, anoxia affinity).
    pub primary_trait_a: &'a [f32],
    /// Per-cell primary trait B (e.g., stress tolerance, ammonium affinity, nitrate affinity).
    pub primary_trait_b: &'a [f32],
    /// Per-cell environmental bias for weighting (e.g., root_density, deep_moisture).
    pub weight_env_bias: &'a [f32],
    /// Per-cell environmental bias for spread (e.g., moisture, deep_moisture).
    pub spread_env_bias: &'a [f32],
    /// Latent bank packet counts [shadow_bank, variant_bank].
    pub latent_packets: [&'a [f32]; 2],
    /// Latent bank trait A values [shadow_bank, variant_bank].
    pub latent_trait_a: [&'a [f32]; 2],
    /// Latent bank trait B values [shadow_bank, variant_bank].
    pub latent_trait_b: [&'a [f32]; 2],
}

/// Knobs controlling latent bank mutation, drift, and selection.
pub struct LatentGuildConfig {
    // -- Latent pool sizing --
    pub latent_pool_base: f32,
    pub latent_pool_mutation_scale: f32,
    pub latent_pool_inactive_scale: f32,
    pub latent_pool_min: f32,
    pub latent_pool_max: f32,

    // -- Weight (bank relative importance) --
    pub weight_base: f32,
    pub weight_step: f32,
    pub weight_mutation_scale: f32,
    pub weight_mutation_bank_step: f32,
    pub weight_env_scale: f32,

    // -- Packet relaxation --
    pub packet_relax_rate: f32,
    pub packet_mutation_scale: f32,
    pub packet_activity_scale: f32,

    // -- Spread (trait variance proxy) --
    pub spread_base: f32,
    pub spread_step: f32,
    pub spread_mutation_scale: f32,
    pub spread_mutation_bank_step: f32,
    pub spread_env_scale: f32,
    pub spread_env_center: f32,
    pub spread_min: f32,
    pub spread_max: f32,

    // -- Trait relaxation --
    pub trait_relax_rate: f32,
    pub trait_mutation_scale: f32,
    pub trait_a_polarities: [f32; 2],
    pub trait_b_polarities: [f32; 2],
    pub trait_b_spread_scale: f32,
    pub trait_b_inactive_scale: f32,
}

/// Updated trait vectors returned after one step of latent bank evolution.
pub struct LatentGuildResult {
    /// Updated primary trait A (relaxed toward latent bank means).
    pub primary_trait_a: Vec<f32>,
    /// Updated primary trait B (relaxed toward latent bank means).
    pub primary_trait_b: Vec<f32>,
    /// Updated latent bank packet counts [shadow, variant].
    pub latent_packets: [Vec<f32>; 2],
    /// Updated latent bank trait A [shadow, variant].
    pub latent_trait_a: [Vec<f32>; 2],
    /// Updated latent bank trait B [shadow, variant].
    pub latent_trait_b: [Vec<f32>; 2],
}

/// Step the latent guild banks for one timestep.
///
/// For each grid cell, the function:
/// 1. Computes an effective latent pool size from mutation flux and dormancy.
/// 2. Perturbs latent bank traits with Gaussian noise scaled by mutation flux.
/// 3. Relaxes primary traits toward the weighted mean of active + latent banks.
/// 4. Applies dormancy-modulated selection pressure (dormant cells bias toward
///    the latent banks; active cells bias toward the primary population).
///
/// This is the core microevolutionary engine that maintains microbial diversity
/// within a single guild, enabling bet-hedging and adaptive switching.
pub fn step_latent_guild_banks(
    state: &LatentGuildState,
    config: LatentGuildConfig,
    dt: f32,
) -> Result<LatentGuildResult, String> {
    let n = state.total_packets.len();
    if n == 0 {
        return Ok(LatentGuildResult {
            primary_trait_a: Vec::new(),
            primary_trait_b: Vec::new(),
            latent_packets: [Vec::new(), Vec::new()],
            latent_trait_a: [Vec::new(), Vec::new()],
            latent_trait_b: [Vec::new(), Vec::new()],
        });
    }

    let mut primary_a = state.primary_trait_a.to_vec();
    let mut primary_b = state.primary_trait_b.to_vec();
    let mut latent_pkt = [
        state.latent_packets[0].to_vec(),
        state.latent_packets[1].to_vec(),
    ];
    let mut latent_a = [
        state.latent_trait_a[0].to_vec(),
        state.latent_trait_a[1].to_vec(),
    ];
    let mut latent_b = [
        state.latent_trait_b[0].to_vec(),
        state.latent_trait_b[1].to_vec(),
    ];

    let dt_scale = (dt * 10.0).min(1.0);

    for i in 0..n {
        let flux = state.mutation_flux[i].max(0.0);
        let dormancy = state.dormancy[i].clamp(0.0, 1.0);
        let vitality = state.vitality[i].clamp(0.0, 1.0);
        let total = state.total_packets[i].max(1.0);

        // Effective latent pool size
        let pool_size = (config.latent_pool_base
            + flux * config.latent_pool_mutation_scale
            + (1.0 - vitality) * config.latent_pool_inactive_scale)
            .clamp(config.latent_pool_min, config.latent_pool_max);

        // Per-bank weight (relative importance)
        for bank in 0..2 {
            let env_w = state.weight_env_bias[i] * config.weight_env_scale;
            let w = config.weight_base
                + bank as f32 * config.weight_step
                + flux * config.weight_mutation_scale
                + bank as f32 * config.weight_mutation_bank_step * flux
                + env_w;

            // Latent packet count relaxation toward pool target
            let target_pkt = total * pool_size * w.clamp(0.1, 2.0);
            let relax = config.packet_relax_rate * dt_scale;
            latent_pkt[bank][i] += (target_pkt - latent_pkt[bank][i])
                * relax
                * (1.0 + flux * config.packet_mutation_scale)
                * (1.0 + dormancy * config.packet_activity_scale);
            latent_pkt[bank][i] = latent_pkt[bank][i].max(0.0);

            // Spread (trait variance proxy)
            let env_s = (state.spread_env_bias[i] - config.spread_env_center).abs()
                * config.spread_env_scale;
            let spread = (config.spread_base
                + bank as f32 * config.spread_step
                + flux * config.spread_mutation_scale
                + bank as f32 * config.spread_mutation_bank_step * flux
                + env_s)
                .clamp(config.spread_min, config.spread_max);

            // Trait perturbation via Gaussian-like deterministic shift
            // (Actual Gaussian noise would require RNG; here we use a
            //  deterministic approximation scaled by mutation flux.)
            let trait_shift_a = flux * config.trait_mutation_scale
                * config.trait_a_polarities[bank]
                * spread
                * dt_scale;
            let trait_shift_b = flux * config.trait_mutation_scale
                * config.trait_b_polarities[bank]
                * spread
                * config.trait_b_spread_scale
                * dt_scale
                + (1.0 - vitality) * config.trait_b_inactive_scale * dt_scale;

            latent_a[bank][i] += trait_shift_a;
            latent_b[bank][i] += trait_shift_b;
            latent_a[bank][i] = latent_a[bank][i].clamp(-2.0, 2.0);
            latent_b[bank][i] = latent_b[bank][i].clamp(-2.0, 2.0);
        }

        // Relax primary traits toward weighted latent bank mean
        let relax_rate = config.trait_relax_rate * dt_scale;
        let dormancy_weight = dormancy * 0.5;
        let latent_mean_a = (latent_a[0][i] + latent_a[1][i]) * 0.5;
        let latent_mean_b = (latent_b[0][i] + latent_b[1][i]) * 0.5;
        primary_a[i] += (latent_mean_a - primary_a[i]) * relax_rate * (1.0 + dormancy_weight);
        primary_b[i] += (latent_mean_b - primary_b[i]) * relax_rate * (1.0 + dormancy_weight);
    }

    Ok(LatentGuildResult {
        primary_trait_a: primary_a,
        primary_trait_b: primary_b,
        latent_packets: latent_pkt,
        latent_trait_a: latent_a,
        latent_trait_b: latent_b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_returns_empty_result() {
        let result = step_latent_guild_banks(
            &LatentGuildState {
                total_packets: &[],
                public_secondary_packets: [&[], &[], &[]],
                public_secondary_trait_a: [&[], &[], &[]],
                public_secondary_trait_b: [&[], &[], &[]],
                mutation_flux: &[],
                vitality: &[],
                dormancy: &[],
                primary_trait_a: &[],
                primary_trait_b: &[],
                weight_env_bias: &[],
                spread_env_bias: &[],
                latent_packets: [&[], &[]],
                latent_trait_a: [&[], &[]],
                latent_trait_b: [&[], &[]],
            },
            LatentGuildConfig {
                latent_pool_base: 0.05,
                latent_pool_mutation_scale: 1.6,
                latent_pool_inactive_scale: 0.12,
                latent_pool_min: 0.02,
                latent_pool_max: 0.42,
                weight_base: 0.8,
                weight_step: 0.18,
                weight_mutation_scale: 1.4,
                weight_mutation_bank_step: 0.35,
                weight_env_scale: 0.22,
                packet_relax_rate: 0.00075,
                packet_mutation_scale: 2.0,
                packet_activity_scale: 0.35,
                spread_base: 0.06,
                spread_step: 0.0,
                spread_mutation_scale: 0.9,
                spread_mutation_bank_step: 0.25,
                spread_env_scale: 0.08,
                spread_env_center: 0.5,
                spread_min: 0.03,
                spread_max: 0.25,
                trait_relax_rate: 0.00055,
                trait_mutation_scale: 0.010,
                trait_a_polarities: [-1.0, 1.0],
                trait_b_polarities: [0.85, -0.85],
                trait_b_spread_scale: 1.0,
                trait_b_inactive_scale: 0.06,
            },
            0.1,
        ).unwrap();
        assert!(result.primary_trait_a.is_empty());
        assert!(result.latent_packets[0].is_empty());
    }

    #[test]
    fn high_mutation_flux_perturbs_latent_traits() {
        let n = 4;
        let packets = vec![100.0; n];
        let zeros = vec![0.0; n];
        let ones = vec![1.0; n];
        let high_flux = vec![0.5; n];
        let half = vec![0.5; n];
        let result = step_latent_guild_banks(
            &LatentGuildState {
                total_packets: &packets,
                public_secondary_packets: [&zeros, &zeros, &zeros],
                public_secondary_trait_a: [&zeros, &zeros, &zeros],
                public_secondary_trait_b: [&zeros, &zeros, &zeros],
                mutation_flux: &high_flux,
                vitality: &ones,
                dormancy: &zeros,
                primary_trait_a: &zeros,
                primary_trait_b: &zeros,
                weight_env_bias: &half,
                spread_env_bias: &half,
                latent_packets: [&zeros, &zeros],
                latent_trait_a: [&zeros, &zeros],
                latent_trait_b: [&zeros, &zeros],
            },
            LatentGuildConfig {
                latent_pool_base: 0.05,
                latent_pool_mutation_scale: 1.6,
                latent_pool_inactive_scale: 0.12,
                latent_pool_min: 0.02,
                latent_pool_max: 0.42,
                weight_base: 0.8,
                weight_step: 0.18,
                weight_mutation_scale: 1.4,
                weight_mutation_bank_step: 0.35,
                weight_env_scale: 0.22,
                packet_relax_rate: 0.00075,
                packet_mutation_scale: 2.0,
                packet_activity_scale: 0.35,
                spread_base: 0.06,
                spread_step: 0.0,
                spread_mutation_scale: 0.9,
                spread_mutation_bank_step: 0.25,
                spread_env_scale: 0.08,
                spread_env_center: 0.5,
                spread_min: 0.03,
                spread_max: 0.25,
                trait_relax_rate: 0.00055,
                trait_mutation_scale: 0.010,
                trait_a_polarities: [-1.0, 1.0],
                trait_b_polarities: [0.85, -0.85],
                trait_b_spread_scale: 1.0,
                trait_b_inactive_scale: 0.06,
            },
            0.1,
        ).unwrap();

        // With high mutation flux, latent traits should have shifted
        // Bank 0 has polarity -1.0 for trait_a, so should go negative
        assert!(result.latent_trait_a[0][0] < 0.0, "Bank 0 trait A should shift negative");
        // Bank 1 has polarity +1.0 for trait_a, so should go positive
        assert!(result.latent_trait_a[1][0] > 0.0, "Bank 1 trait A should shift positive");
        // Latent packets should have grown from 0 toward pool target
        assert!(result.latent_packets[0][0] > 0.0, "Latent packets should grow");
    }

    #[test]
    fn dormancy_increases_relaxation_toward_latent() {
        let n = 2;
        let packets = vec![100.0; n];
        let zeros = vec![0.0; n];
        let ones = vec![1.0; n];
        let flux = vec![0.1; n];
        let half = vec![0.5; n];
        let high_dormancy = vec![0.9; n];
        let latent_a = vec![0.5; n]; // Latent traits offset from primary (0.0)

        let result_dormant = step_latent_guild_banks(
            &LatentGuildState {
                total_packets: &packets,
                public_secondary_packets: [&zeros, &zeros, &zeros],
                public_secondary_trait_a: [&zeros, &zeros, &zeros],
                public_secondary_trait_b: [&zeros, &zeros, &zeros],
                mutation_flux: &flux,
                vitality: &ones,
                dormancy: &high_dormancy,
                primary_trait_a: &zeros,
                primary_trait_b: &zeros,
                weight_env_bias: &half,
                spread_env_bias: &half,
                latent_packets: [&latent_a, &latent_a],
                latent_trait_a: [&latent_a, &latent_a],
                latent_trait_b: [&zeros, &zeros],
            },
            LatentGuildConfig {
                latent_pool_base: 0.05,
                latent_pool_mutation_scale: 1.6,
                latent_pool_inactive_scale: 0.12,
                latent_pool_min: 0.02,
                latent_pool_max: 0.42,
                weight_base: 0.8,
                weight_step: 0.18,
                weight_mutation_scale: 1.4,
                weight_mutation_bank_step: 0.35,
                weight_env_scale: 0.22,
                packet_relax_rate: 0.00075,
                packet_mutation_scale: 2.0,
                packet_activity_scale: 0.35,
                spread_base: 0.06,
                spread_step: 0.0,
                spread_mutation_scale: 0.9,
                spread_mutation_bank_step: 0.25,
                spread_env_scale: 0.08,
                spread_env_center: 0.5,
                spread_min: 0.03,
                spread_max: 0.25,
                trait_relax_rate: 0.00055,
                trait_mutation_scale: 0.010,
                trait_a_polarities: [-1.0, 1.0],
                trait_b_polarities: [0.85, -0.85],
                trait_b_spread_scale: 1.0,
                trait_b_inactive_scale: 0.06,
            },
            0.1,
        ).unwrap();

        let result_active = step_latent_guild_banks(
            &LatentGuildState {
                total_packets: &packets,
                public_secondary_packets: [&zeros, &zeros, &zeros],
                public_secondary_trait_a: [&zeros, &zeros, &zeros],
                public_secondary_trait_b: [&zeros, &zeros, &zeros],
                mutation_flux: &flux,
                vitality: &ones,
                dormancy: &zeros, // No dormancy
                primary_trait_a: &zeros,
                primary_trait_b: &zeros,
                weight_env_bias: &half,
                spread_env_bias: &half,
                latent_packets: [&latent_a, &latent_a],
                latent_trait_a: [&latent_a, &latent_a],
                latent_trait_b: [&zeros, &zeros],
            },
            LatentGuildConfig {
                latent_pool_base: 0.05,
                latent_pool_mutation_scale: 1.6,
                latent_pool_inactive_scale: 0.12,
                latent_pool_min: 0.02,
                latent_pool_max: 0.42,
                weight_base: 0.8,
                weight_step: 0.18,
                weight_mutation_scale: 1.4,
                weight_mutation_bank_step: 0.35,
                weight_env_scale: 0.22,
                packet_relax_rate: 0.00075,
                packet_mutation_scale: 2.0,
                packet_activity_scale: 0.35,
                spread_base: 0.06,
                spread_step: 0.0,
                spread_mutation_scale: 0.9,
                spread_mutation_bank_step: 0.25,
                spread_env_scale: 0.08,
                spread_env_center: 0.5,
                spread_min: 0.03,
                spread_max: 0.25,
                trait_relax_rate: 0.00055,
                trait_mutation_scale: 0.010,
                trait_a_polarities: [-1.0, 1.0],
                trait_b_polarities: [0.85, -0.85],
                trait_b_spread_scale: 1.0,
                trait_b_inactive_scale: 0.06,
            },
            0.1,
        ).unwrap();

        // Dormant population should relax MORE toward latent means
        assert!(
            result_dormant.primary_trait_a[0].abs() > result_active.primary_trait_a[0].abs(),
            "Dormant should relax more toward latent: dormant={}, active={}",
            result_dormant.primary_trait_a[0],
            result_active.primary_trait_a[0]
        );
    }
}
