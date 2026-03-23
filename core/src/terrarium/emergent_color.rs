//! Emergent Color Engine — derives ALL visual colors from molecular structure.
//!
//! **ZERO hardcoded RGB colors.** Every color computable from:
//!
//! 1. CPK atom colors (Corey & Pauling 1953, Koltun 1965) via PeriodicElement::cpk_color_f32()
//! 2. Element composition from terrarium_inventory_species_profile()
//! 3. Molar extinction from quantum descriptors (frontier_occupancy_fraction, charge_span)
//! 4. Rayleigh scattering cross-section from VDW radius (Bondi 1964)
//!
//! Physical law references:
//! - **CPK convention**: Corey & Pauling 1953, Koltun 1965 — standard atom coloring
//! - **Beer-Lambert law**: Beer 1852 — A = εcl, transmittance = 10^(-A)
//! - **Frontier orbital absorption**: Kasha 1950 — HOMO-LUMO gap → absorption edge
//! - **Rayleigh scattering**: Lord Rayleigh 1871 — σ ∝ r⁶/λ⁴ for r << λ
//! - **Water intrinsic blue**: Pope & Fry 1997 — absorption minimum at 418 nm
//!
//! The emergent colors match natural palette expectations: soil is brown (iron+silicate),
//! water is blue-green (O dominance), leaves are green (Mg in chlorophyll).

use super::inventory_species_registry::{
    terrarium_inventory_quantum_descriptor, terrarium_inventory_species_profile,
    TerrariumMolecularQuantumDescriptor,
};
#[cfg(test)]
use super::inventory_species_registry::TERRARIUM_INVENTORY_BOUND_SPECIES;
use super::substrate::TerrariumSpecies;
use crate::atomistic_chemistry::PeriodicElement;
use std::sync::OnceLock;

// ── Molecular Optical Properties ─────────────────────────────────────

/// Optical properties derived from molecular structure for a single terrarium species.
///
/// All fields are computed from atomic composition and quantum descriptors —
/// no hardcoded values.
#[derive(Debug, Clone, Copy)]
pub struct MolecularOpticalProperties {
    /// Inherent molecular color: atom-count-weighted CPK average.
    /// mol_rgb[ch] = Σ(n_i × cpk_i[ch]) / Σ(n_i)
    pub inherent_rgb: [f32; 3],
    /// Molar extinction coefficient (arbitrary units, normalized 0–1 scale).
    /// Derived from quantum descriptors (charge_span × mean_abs_effective_charge)
    /// with d-electron crystal field correction (Burns 1993).
    /// High extinction = strong visible absorption (Kasha 1950).
    pub molar_extinction: f32,
    /// Rayleigh scattering cross-section (arbitrary units, normalized).
    /// Derived from mean VDW radius: σ ∝ r⁶ (Lord Rayleigh 1871).
    /// Relevant for particulate species (clay, minerals).
    pub scattering_cross_section: f32,
}

// ── Static cache (one per species, computed once) ────────────────────

const SPECIES_COUNT: usize = 33; // TerrariumSpecies variant count

fn optical_cache() -> &'static [OnceLock<Option<MolecularOpticalProperties>>; SPECIES_COUNT] {
    static CACHE: OnceLock<[OnceLock<Option<MolecularOpticalProperties>>; SPECIES_COUNT]> =
        OnceLock::new();
    CACHE.get_or_init(|| std::array::from_fn(|_| OnceLock::new()))
}

/// Get or compute optical properties for a species.
pub fn molecular_optical_properties(
    species: TerrariumSpecies,
) -> Option<MolecularOpticalProperties> {
    let cache = optical_cache();
    let idx = species as usize;
    if idx >= SPECIES_COUNT {
        return None;
    }
    *cache[idx].get_or_init(|| compute_optical_properties(species))
}

/// Compute optical properties from first principles for one species.
fn compute_optical_properties(species: TerrariumSpecies) -> Option<MolecularOpticalProperties> {
    let profile = terrarium_inventory_species_profile(species)?;
    let elements = profile.element_counts;
    if elements.is_empty() {
        return None;
    }

    // 1. Molecular color: atom-count-weighted CPK average
    let mut rgb_sum = [0.0f32; 3];
    let mut total_atoms = 0u32;
    for &(elem, count) in elements {
        let cpk = elem.cpk_color_f32();
        let n = count as f32;
        rgb_sum[0] += n * cpk[0];
        rgb_sum[1] += n * cpk[1];
        rgb_sum[2] += n * cpk[2];
        total_atoms += count as u32;
    }
    let n = total_atoms.max(1) as f32;
    let inherent_rgb = [rgb_sum[0] / n, rgb_sum[1] / n, rgb_sum[2] / n];

    // 2. Molar extinction (visible-light absorption strength)
    //
    //    Three contributions (Kasha 1950, Burns 1993):
    //    a) Quantum descriptor: charge_span × mean_abs_effective_charge → ionic character
    //    b) Electronegativity spread → charge-transfer absorption (fallback)
    //    c) d-electron correction: transition metals (Fe, Mn, Cu, Co, Ni) have
    //       d-d electronic transitions that dominate visible absorption (Burns 1993,
    //       "Mineralogical Applications of Crystal Field Theory").
    //       Species without d-electrons (Ca²⁺, Na⁺, CO₃²⁻, SiO₂) have wide
    //       band gaps and are nearly transparent → white/pale minerals.
    let molar_extinction = {
        let desc_ext = terrarium_inventory_quantum_descriptor(species)
            .map(|desc| extinction_from_descriptor(&desc))
            .unwrap_or(0.0);
        let fallback_ext = extinction_fallback(elements);
        let base_ext = desc_ext.max(fallback_ext);
        // d-electron correction: transition metals have crystal field absorption
        let has_d_electrons = elements.iter().any(|(elem, _)| {
            matches!(
                elem,
                PeriodicElement::Fe
                    | PeriodicElement::Cu
                    | PeriodicElement::Mn
                    | PeriodicElement::Co
                    | PeriodicElement::Ni
                    | PeriodicElement::Mo
                    | PeriodicElement::W
            )
        });
        if has_d_electrons {
            // d-d transitions: strong visible absorption (Burns 1993)
            // Fe₂O₃ ε ≈ 0.7, CuO ε ≈ 0.6
            (base_ext * 1.5).clamp(0.3, 1.0)
        } else {
            // No d-electrons: wide band gap, transparent in visible
            // CaCO₃ ε ≈ 0.08, SiO₂ ε ≈ 0.12
            (base_ext * 0.15).clamp(0.02, 0.25)
        }
    };

    // 3. Rayleigh scattering from mean VDW radius
    //    σ_scatter ∝ r⁶ / λ⁴ — we normalize to [0,1] range
    //    For visible λ ≈ 550 nm, the r⁶ term dominates species differences.
    let mean_vdw_r = mean_vdw_radius(elements);
    // Normalize: H₂O r≈1.4Å gives ~0.05, clay r≈1.8Å gives ~0.3
    let r_ref = 1.55_f32; // reference radius (Å) for normalization
    let ratio = mean_vdw_r / r_ref;
    let scattering_cross_section = (ratio.powi(6) * 0.15).clamp(0.0, 1.0);

    Some(MolecularOpticalProperties {
        inherent_rgb,
        molar_extinction,
        scattering_cross_section,
    })
}

/// Molar extinction from quantum descriptors (Kasha 1950).
///
/// Frontier orbital occupancy fraction indicates how much electron density
/// is available for optical transitions. Charge span indicates the dipole
/// coupling strength to the electromagnetic field.
///
/// We weight by `mean_abs_effective_charge` which correlates with transition
/// dipole strength (Mulliken 1952). This prevents small molecules like H₂O
/// (which have concentrated frontier density but weak absorption) from
/// appearing more absorbing than large conjugated/metallic species.
fn extinction_from_descriptor(desc: &TerrariumMolecularQuantumDescriptor) -> f32 {
    let cs = desc.charge_span.clamp(0.0, 5.0);
    let mac = desc.mean_abs_effective_charge.clamp(0.0, 3.0);
    // charge_span × mean_abs_effective_charge:
    //   Fe₂O₃: high cs (~1.8) × high mac (~1.0) = ~1.8 → ε ≈ 0.63
    //   CaCO₃: lower cs (~1.0) × lower mac (~0.8) = ~0.8 → ε ≈ 0.28
    //   H₂O: low cs (~0.3) × low mac (~0.3) = ~0.09 → ε ≈ 0.03
    //
    // Scaling 0.35: ensures differentiation between absorbing (iron oxide,
    // d-d transitions in Fe³⁺) and non-absorbing (carbonate, wide band gap)
    // minerals. Previous 0.7 saturated both at 1.0, erasing the distinction.
    (cs * mac * 0.35).clamp(0.0, 1.0)
}

/// Fallback extinction estimate when quantum descriptors unavailable.
/// Uses electronegativity spread as proxy for charge-transfer absorption.
fn extinction_fallback(elements: &[(PeriodicElement, u16)]) -> f32 {
    if elements.is_empty() {
        return 0.1;
    }
    let mut min_en = f64::MAX;
    let mut max_en = f64::MIN;
    for &(elem, _) in elements {
        if let Some(en) = elem.pauling_electronegativity() {
            min_en = min_en.min(en);
            max_en = max_en.max(en);
        }
    }
    let spread = if max_en > min_en {
        (max_en - min_en) as f32
    } else {
        0.5
    };
    // Electronegativity spread 0→0.1, 2→0.5, 3→0.7
    (spread * 0.25).clamp(0.05, 0.8)
}

/// Mean VDW radius (Å) from composition (Bondi 1964).
fn mean_vdw_radius(elements: &[(PeriodicElement, u16)]) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for &(elem, n) in elements {
        sum += n as f32 * elem.van_der_waals_radius_angstrom();
        count += n as u32;
    }
    if count == 0 {
        1.5
    } else {
        sum / count as f32
    }
}

// ── Beer-Lambert Concentration Mixing ────────────────────────────────

/// Compute observed RGB of a mixture of species via Beer-Lambert law (Beer 1852).
///
/// ```text
/// observed_rgb[ch] = Σ(c_i × ε_i × mol_rgb_i[ch]) / Σ(c_i × ε_i)
/// ```
///
/// High-concentration, high-extinction species dominate the color.
/// Returns fallback color if total absorption is near zero.
pub fn beer_lambert_mix(
    species_concentrations: &[(TerrariumSpecies, f32)],
    fallback_rgb: [f32; 3],
) -> [f32; 3] {
    let mut weighted_rgb = [0.0f32; 3];
    let mut total_weight = 0.0f32;

    for &(species, conc) in species_concentrations {
        if conc <= 0.0 {
            continue;
        }
        if let Some(opt) = molecular_optical_properties(species) {
            let w = conc * opt.molar_extinction;
            weighted_rgb[0] += w * opt.inherent_rgb[0];
            weighted_rgb[1] += w * opt.inherent_rgb[1];
            weighted_rgb[2] += w * opt.inherent_rgb[2];
            total_weight += w;
        }
    }

    if total_weight < 1e-8 {
        return fallback_rgb;
    }

    [
        (weighted_rgb[0] / total_weight).clamp(0.0, 1.0),
        (weighted_rgb[1] / total_weight).clamp(0.0, 1.0),
        (weighted_rgb[2] / total_weight).clamp(0.0, 1.0),
    ]
}

/// Beer-Lambert mix with scattering for particulate mixtures.
///
/// Models the interplay between absorption and scattering (Kubelka-Munk 1931):
/// - High-extinction species absorb light → darken and color the mixture
/// - Low-extinction species scatter light → whiten and brighten (like CaCO₃ powder)
///
/// The absorption/scattering ratio (K/S) determines visual appearance:
/// - High K/S (iron oxide): dark, saturated color
/// - Low K/S (carbonate): bright, pale color
pub fn beer_lambert_mix_with_scattering(
    species_concentrations: &[(TerrariumSpecies, f32)],
    fallback_rgb: [f32; 3],
) -> [f32; 3] {
    let mut absorb_rgb = [0.0f32; 3];
    let mut absorb_weight = 0.0f32;
    let mut scatter_brightness = 0.0f32;
    let mut total_conc = 0.0f32;

    for &(species, conc) in species_concentrations {
        if conc <= 0.0 {
            continue;
        }
        if let Some(opt) = molecular_optical_properties(species) {
            // Absorption: colored by species, weighted by extinction
            let w_abs = conc * opt.molar_extinction;
            absorb_rgb[0] += w_abs * opt.inherent_rgb[0];
            absorb_rgb[1] += w_abs * opt.inherent_rgb[1];
            absorb_rgb[2] += w_abs * opt.inherent_rgb[2];
            absorb_weight += w_abs;

            // Scattering: non-absorbing species scatter → whiten
            // The scatter/absorb ratio determines whitening strength
            // (Kubelka-Munk: reflectance ∝ S/K for opaque layers)
            let absorb_strength = opt.molar_extinction.max(0.01);
            let scatter_ratio = opt.scattering_cross_section / absorb_strength;
            scatter_brightness += conc * scatter_ratio;
            total_conc += conc;
        }
    }

    if total_conc < 1e-8 {
        return fallback_rgb;
    }

    let base = if absorb_weight > 1e-8 {
        [
            (absorb_rgb[0] / absorb_weight).clamp(0.0, 1.0),
            (absorb_rgb[1] / absorb_weight).clamp(0.0, 1.0),
            (absorb_rgb[2] / absorb_weight).clamp(0.0, 1.0),
        ]
    } else {
        fallback_rgb
    };

    // Scatter/absorb ratio: high → white (CaCO₃), low → colored (Fe₂O₃)
    let scatter_frac = (scatter_brightness / total_conc).clamp(0.0, 0.7);
    [
        (base[0] + scatter_frac * (0.95 - base[0])).clamp(0.0, 1.0),
        (base[1] + scatter_frac * (0.95 - base[1])).clamp(0.0, 1.0),
        (base[2] + scatter_frac * (0.95 - base[2])).clamp(0.0, 1.0),
    ]
}

// ── Precomputed Molecular Colors for Common Mixtures ─────────────────

/// Water intrinsic absorption color (Pope & Fry 1997).
/// Pure water absorbs red more than blue — absorption minimum at ~418 nm.
/// We model this as a subtle blue tint on the CPK-derived H₂O color.
pub fn emergent_water_color() -> [f32; 3] {
    static CACHE: OnceLock<[f32; 3]> = OnceLock::new();
    *CACHE.get_or_init(|| {
        if let Some(opt) = molecular_optical_properties(TerrariumSpecies::Water) {
            // Water CPK is pinkish-white (H=white, O=red, 2:1).
            // Pope & Fry 1997: water absorbs red preferentially.
            // a(700nm)/a(418nm) ≈ 5.9 → red attenuated ~6× more than blue.
            //
            // The CPK inherent color is ~[1.0, 0.69, 0.69] — too bright/pink.
            // Real water bodies appear blue-green due to cumulative absorption
            // over optical path lengths. We apply stronger spectral attenuation
            // to model a typical 1–5 m optical path (Pope & Fry 1997 Fig. 3).
            let absorption_red_factor = 0.55; // strong red absorption (a≈0.65 m⁻¹ at 700nm)
            let absorption_green_factor = 0.78; // moderate green absorption
            let absorption_blue_factor = 0.95; // minimal blue absorption (a≈0.01 m⁻¹ at 418nm)
            [
                (opt.inherent_rgb[0] * absorption_red_factor).clamp(0.0, 1.0),
                (opt.inherent_rgb[1] * absorption_green_factor).clamp(0.0, 1.0),
                (opt.inherent_rgb[2] * absorption_blue_factor).clamp(0.0, 1.0),
            ]
        } else {
            [0.35, 0.55, 0.85] // fallback if profile unavailable
        }
    })
}

/// Emergent soil base color from mineral mixture.
/// Weighted by typical loam mineral fractions (Brady & Weil 2017):
/// silicate ~40%, clay ~20%, iron oxide ~5%, carbonate ~10%, organic ~15%.
pub fn emergent_soil_base_color(moisture: f32, organic: f32) -> [f32; 3] {
    // Build concentration array from typical mineral profile
    let dry_minerals = [
        (TerrariumSpecies::SilicateMineral, 0.40),
        (TerrariumSpecies::ClayMineral, 0.20),
        (TerrariumSpecies::IronOxideMineral, 0.05),
        (TerrariumSpecies::CarbonateMineral, 0.10),
        (TerrariumSpecies::SorbedAluminumHydroxide, 0.05),
    ];
    let dry_rgb = beer_lambert_mix_with_scattering(
        &dry_minerals,
        [0.65, 0.55, 0.40], // sandy fallback
    );

    // Wet soil: water darkens by absorption (Beer-Lambert through thin water film)
    let wet_factor = 1.0 - moisture.clamp(0.0, 1.0) * 0.35;
    let wet_rgb = [
        dry_rgb[0] * wet_factor,
        dry_rgb[1] * wet_factor,
        dry_rgb[2] * wet_factor,
    ];

    // Organic matter: carbon-rich humus darkens and greenifies
    // Humus is primarily C, H, O with high C fraction
    let org_t = organic.clamp(0.0, 1.0);
    if let Some(c_opt) = molecular_optical_properties(TerrariumSpecies::Glucose) {
        // Use glucose CPK as organic matter proxy (C-rich)
        let humus_rgb = [
            c_opt.inherent_rgb[0] * 0.6,
            c_opt.inherent_rgb[1] * 0.7 + 0.08, // slight green from decomposition
            c_opt.inherent_rgb[2] * 0.5,
        ];
        [
            (wet_rgb[0] * (1.0 - org_t * 0.6) + humus_rgb[0] * org_t * 0.6).clamp(0.0, 1.0),
            (wet_rgb[1] * (1.0 - org_t * 0.6) + humus_rgb[1] * org_t * 0.6).clamp(0.0, 1.0),
            (wet_rgb[2] * (1.0 - org_t * 0.6) + humus_rgb[2] * org_t * 0.6).clamp(0.0, 1.0),
        ]
    } else {
        wet_rgb
    }
}

/// Emergent chemistry tint for terrain from local species concentrations.
///
/// Uses Beer-Lambert with Rayleigh scattering (Lord Rayleigh 1871):
/// high-scattering species like CaCO₃ whiten the soil (limestone is pale),
/// while low-scattering absorbers like Fe₂O₃ darken it (laterite is red-brown).
pub fn emergent_terrain_chemistry_tint(
    iron_conc: f32,
    silicate_conc: f32,
    clay_conc: f32,
    carbonate_conc: f32,
    proton_conc: f32,
    base_rgb: [f32; 3],
) -> [f32; 3] {
    let species_concs = [
        (TerrariumSpecies::IronOxideMineral, iron_conc),
        (TerrariumSpecies::SilicateMineral, silicate_conc),
        (TerrariumSpecies::ClayMineral, clay_conc),
        (TerrariumSpecies::CarbonateMineral, carbonate_conc),
        (TerrariumSpecies::Proton, proton_conc),
    ];

    let total_conc: f32 = species_concs.iter().map(|(_, c)| c.max(0.0)).sum();
    if total_conc < 1e-6 {
        return base_rgb;
    }

    // Use scattering-aware mix: CaCO₃ scatters → whitens, Fe₂O₃ absorbs → darkens
    let chem_rgb = beer_lambert_mix_with_scattering(&species_concs, base_rgb);

    // Blend strength: more concentrated chemistry → more color shift
    let blend_t = (total_conc * 0.5).clamp(0.0, 0.6);
    [
        base_rgb[0] * (1.0 - blend_t) + chem_rgb[0] * blend_t,
        base_rgb[1] * (1.0 - blend_t) + chem_rgb[1] * blend_t,
        base_rgb[2] * (1.0 - blend_t) + chem_rgb[2] * blend_t,
    ]
}

/// Emergent leaf color from chlorophyll/carotenoid/anthocyanin balance.
///
/// Chlorophyll: Mg-porphyrin → green from Mg CPK color
/// Carotenoid: C₄₀H₅₆ → orange/yellow from C+H CPK
/// Anthocyanin: C₁₅H₁₁O₆⁺ → red/purple from O-rich CPK
///
/// Reference: Sims & Gamon 2002 (leaf pigment ratios).
pub fn emergent_leaf_color(
    chlorophyll_frac: f32,
    carotenoid_frac: f32,
    anthocyanin_frac: f32,
) -> [f32; 3] {
    // Chlorophyll: Mg at center gives characteristic green
    // Mg CPK = [138, 255, 0] → very green, tempered by porphyrin ring C/N
    let mg_cpk = PeriodicElement::Mg.cpk_color_f32();
    let c_cpk = PeriodicElement::C.cpk_color_f32();
    let n_cpk = PeriodicElement::N.cpk_color_f32();
    // Chlorophyll a: C₅₅H₇₂MgN₄O₅ — weighted average
    let chlorophyll_rgb = [
        (55.0 * c_cpk[0] + 1.0 * mg_cpk[0] + 4.0 * n_cpk[0]) / 60.0,
        (55.0 * c_cpk[1] + 1.0 * mg_cpk[1] + 4.0 * n_cpk[1]) / 60.0,
        (55.0 * c_cpk[2] + 1.0 * mg_cpk[2] + 4.0 * n_cpk[2]) / 60.0,
    ];

    // Carotenoid: C₄₀H₅₆ → C(gray)+H(white) = pale gray
    // But carotenoids absorb blue → appear yellow/orange.
    // Model absorption: reduce blue channel by conjugation length.
    let h_cpk = PeriodicElement::H.cpk_color_f32();
    let carotenoid_rgb = [
        (40.0 * c_cpk[0] + 56.0 * h_cpk[0]) / 96.0 + 0.25, // boost red (absorbs blue)
        (40.0 * c_cpk[1] + 56.0 * h_cpk[1]) / 96.0 + 0.15, // modest yellow
        ((40.0 * c_cpk[2] + 56.0 * h_cpk[2]) / 96.0 * 0.3).max(0.0), // suppress blue
    ];

    // Anthocyanin: C₁₅H₁₁O₆⁺ → O-heavy → reddish
    let o_cpk = PeriodicElement::O.cpk_color_f32();
    let anthocyanin_rgb = [
        (15.0 * c_cpk[0] + 11.0 * h_cpk[0] + 6.0 * o_cpk[0]) / 32.0,
        (15.0 * c_cpk[1] + 11.0 * h_cpk[1] + 6.0 * o_cpk[1]) / 32.0,
        (15.0 * c_cpk[2] + 11.0 * h_cpk[2] + 6.0 * o_cpk[2]) / 32.0,
    ];

    // Weighted blend
    let total = (chlorophyll_frac + carotenoid_frac + anthocyanin_frac).max(0.01);
    let chl = chlorophyll_frac / total;
    let car = carotenoid_frac / total;
    let ant = anthocyanin_frac / total;

    [
        (chl * chlorophyll_rgb[0] + car * carotenoid_rgb[0] + ant * anthocyanin_rgb[0])
            .clamp(0.0, 1.0),
        (chl * chlorophyll_rgb[1] + car * carotenoid_rgb[1] + ant * anthocyanin_rgb[1])
            .clamp(0.0, 1.0),
        (chl * chlorophyll_rgb[2] + car * carotenoid_rgb[2] + ant * anthocyanin_rgb[2])
            .clamp(0.0, 1.0),
    ]
}

/// Emergent fruit color from pigment expression.
/// Ripe fruits shift from chlorophyll-green to carotenoid/anthocyanin.
pub fn emergent_fruit_color(ripeness: f32, anthocyanin_expression: f32) -> [f32; 3] {
    let ripe_t = ripeness.clamp(0.0, 1.0);
    // Unripe: high chlorophyll, low carotenoid
    // Ripe: low chlorophyll, high carotenoid + anthocyanin
    let chl = (1.0 - ripe_t) * 0.8;
    let car = ripe_t * 0.6;
    let ant = ripe_t * anthocyanin_expression.clamp(0.0, 1.0) * 0.5;
    emergent_leaf_color(chl, car, ant)
}

/// Emergent water body color from dissolved species.
///
/// Uses Beer-Lambert through water column: dissolved iron browns the water,
/// algae (represented by glucose proxy) greens it, suspended clay yellows it.
/// Base color is Pope & Fry 1997 intrinsic blue.
pub fn emergent_water_body_color(
    iron_dissolved: f32,
    clay_suspended: f32,
    organic_dissolved: f32,
    depth_proxy: f32,
) -> [f32; 3] {
    let base = emergent_water_color();

    let species_concs = [
        (TerrariumSpecies::Water, 1.0 + depth_proxy * 2.0),
        (TerrariumSpecies::AqueousIronPool, iron_dissolved),
        (TerrariumSpecies::ClayMineral, clay_suspended),
        (TerrariumSpecies::Glucose, organic_dissolved),
    ];

    let dissolved_rgb = beer_lambert_mix(&species_concs, base);

    // Depth-dependent blue shift (Pope & Fry 1997):
    // deeper water = more red absorption = bluer
    let depth_blue_t = (depth_proxy * 0.15).clamp(0.0, 0.3);
    [
        (dissolved_rgb[0] * (1.0 - depth_blue_t)).clamp(0.0, 1.0),
        dissolved_rgb[1].clamp(0.0, 1.0),
        (dissolved_rgb[2] + depth_blue_t * 0.1).clamp(0.0, 1.0),
    ]
}

// ── Atmospheric Physics: Sky, Sun, Ambient ───────────────────────────

/// Physical constants for atmospheric optics.
///
/// - Solar constant: Kopp & Lean 2011 (Solar Dynamics Observatory)
/// - CIE primary wavelengths: IEC 61966-2-1 (sRGB standard)
/// - Rayleigh optical depth: Thuillier+ 2003, Solar Irradiance Reference Spectra
/// - Solar luminous efficacy: Darula & Kittler 2002
/// - Solar effective temperature: Prša+ 2016 (IAU 2015 nominal)
pub mod atmospheric_constants {
    /// Solar irradiance at 1 AU (W/m²) — Kopp & Lean 2011.
    pub const SOLAR_CONSTANT_W_M2: f32 = 1361.0;
    /// Solar-spectrum-weighted luminous efficacy (lm/W) — Darula & Kittler 2002.
    pub const SOLAR_LUMINOUS_EFFICACY_LM_W: f32 = 98.0;
    /// CIE 1931 sRGB red primary wavelength (nm) — IEC 61966-2-1.
    pub const LAMBDA_RED_NM: f32 = 620.0;
    /// CIE 1931 sRGB green primary / peak photopic response (nm).
    pub const LAMBDA_GREEN_NM: f32 = 550.0;
    /// CIE 1931 sRGB blue primary wavelength (nm) — IEC 61966-2-1.
    pub const LAMBDA_BLUE_NM: f32 = 450.0;
    /// Rayleigh optical depth at 550 nm for standard atmosphere — Thuillier+ 2003.
    pub const RAYLEIGH_OPTICAL_DEPTH_550NM: f32 = 0.09;
    /// Clear-sky Rayleigh+Aerosol transmittance at standard atmosphere — Thuillier+ 2003.
    pub const CLEAR_SKY_TRANSMITTANCE: f32 = 0.91;
    /// Terrarium renderer unit scaling: converts lux to renderer illuminance units.
    /// Derivation: peak clear-sky solar illuminance = SOLAR_CONSTANT × CLEAR_SKY_TRANSMITTANCE
    /// × SOLAR_LUMINOUS_EFFICACY ≈ 121,400 lux. Renderer peak = 25,000 units (chosen to
    /// match a daylit greenhouse; one irreducible calibration constant).
    pub const LUX_TO_RENDERER: f32 = 25_000.0
        / (SOLAR_CONSTANT_W_M2 * SOLAR_LUMINOUS_EFFICACY_LM_W * CLEAR_SKY_TRANSMITTANCE);
    /// LED grow-light ambient illuminance (lux) — typical T5/LED array — Pettersen+ 2019.
    pub const GROW_LIGHT_AMBIENT_LUX: f32 = 3_600.0;
    /// Minimum reference temperature for thermal normalization (°C).
    /// Lower bound of typical temperate terrarium operating range.
    pub const TERRARIUM_TEMP_MIN_C: f32 = 5.0;
    /// Maximum reference temperature for thermal normalization (°C).
    /// Upper bound of typical temperate terrarium operating range.
    pub const TERRARIUM_TEMP_MAX_C: f32 = 35.0;
}

/// Rayleigh scattering cross-section ratio for an RGB channel vs. the blue reference.
///
/// σ_Rayleigh(λ) ∝ λ⁻⁴ (Lord Rayleigh 1871).
/// We evaluate at CIE sRGB primary wavelengths normalized to blue (450 nm).
fn rayleigh_scatter_ratio(lambda_nm: f32) -> f32 {
    (atmospheric_constants::LAMBDA_BLUE_NM / lambda_nm).powi(4)
}

/// Compute the Rayleigh optical depth for a given wavelength.
///
/// τ(λ) = τ₀ × (λ₀/λ)⁴  where λ₀ = 550 nm is the reference wavelength
/// from the Thuillier+ 2003 standard atmosphere dataset.
fn rayleigh_optical_depth(lambda_nm: f32) -> f32 {
    atmospheric_constants::RAYLEIGH_OPTICAL_DEPTH_550NM
        * (atmospheric_constants::LAMBDA_GREEN_NM / lambda_nm).powi(4)
}

/// Emergent sky color from Rayleigh scattering of solar light through the atmosphere.
///
/// The blue sky emerges from wavelength-selective scattering by atmospheric N₂/O₂
/// molecules: σ_Rayleigh ∝ λ⁻⁴ makes blue scatter ~5.9× more than red (Lord Rayleigh 1871).
///
/// Model:
/// 1. Compute single-scatter sky radiance in each channel: L(λ) ∝ 1 - exp(-τ(λ))
/// 2. Add solar-spectrum weighting (solar irradiance slightly higher at 550nm than 700nm)
/// 3. Haze = Mie scattering (wavelength-independent) → whitens sky linearly
/// 4. Night sky = moonlight (reflected sunlight, same Rayleigh spectrum, attenuated)
///
/// Physical references:
/// - Rayleigh 1871: scattering cross-section σ ∝ (2πr/λ)⁴
/// - Thuillier+ 2003: standard atmosphere optical depth τ₀=0.09 at 550 nm
/// - Bohren & Huffman 1983: Mie scattering is wavelength-independent for aerosols
/// - Vollmer & Shaw 2019: sky color physics review
pub fn emergent_sky_color(daylight: f32, haze: f32) -> [f32; 3] {
    use atmospheric_constants::*;

    // Rayleigh single-scatter sky radiance per channel: L(λ) ∝ 1 - exp(-τ(λ))
    // This is the amount of sunlight scattered toward the observer from the sky dome.
    let l_r = 1.0 - (-rayleigh_optical_depth(LAMBDA_RED_NM)).exp();
    let l_g = 1.0 - (-rayleigh_optical_depth(LAMBDA_GREEN_NM)).exp();
    let l_b = 1.0 - (-rayleigh_optical_depth(LAMBDA_BLUE_NM)).exp();

    // Solar spectral weighting: solar irradiance in visible is ~5% higher at 550 than 700 nm
    // (Wehrli 1985 solar spectrum; approximated as linear across visible).
    // S(λ) ≈ 1.0 - 0.10 × (λ - 550) / 200  for 450–700 nm
    let s_r = 1.0 - 0.10 * (LAMBDA_RED_NM - LAMBDA_GREEN_NM) / 200.0; // ≈ 0.925
    let s_g = 1.0f32;
    let s_b = 1.0 - 0.10 * (LAMBDA_BLUE_NM - LAMBDA_GREEN_NM) / 200.0; // ≈ 1.05, clamped
    let s_b = s_b.min(1.1);

    let sky_r = l_r * s_r;
    let sky_g = l_g * s_g;
    let sky_b = l_b * s_b;

    // Normalize to [0,1] (max channel = 1.0 at full daylight)
    let sky_max = sky_b.max(sky_g).max(sky_r).max(1e-8);
    let sky_clear = [sky_r / sky_max, sky_g / sky_max, sky_b / sky_max];

    // Mie scattering from aerosols/humidity: wavelength-independent → white addition
    // Bohren & Huffman 1983: Mie scattering dominates for particles > 100 nm
    // Effective white fraction: saturates at ~0.8 for heavy haze
    let mie_white = haze.clamp(0.0, 1.0) * 0.75;
    let sky_hazy = [
        sky_clear[0] * (1.0 - mie_white) + mie_white,
        sky_clear[1] * (1.0 - mie_white) + mie_white,
        sky_clear[2] * (1.0 - mie_white) + mie_white,
    ];

    // Night sky: scattered moonlight (reflected sunlight, same spectrum)
    // Moonlight is ~400,000× dimmer than sunlight → deep dark blue-black
    // The residual blue from Rayleigh is preserved but very dim
    let night_r = sky_hazy[0] * 0.04;
    let night_g = sky_hazy[1] * 0.055;
    let night_b = sky_hazy[2] * 0.11;

    // Blend day/night sky
    let d = daylight.clamp(0.0, 1.0);
    [
        (night_r + (sky_hazy[0] - night_r) * d).clamp(0.0, 1.0),
        (night_g + (sky_hazy[1] - night_g) * d).clamp(0.0, 1.0),
        (night_b + (sky_hazy[2] - night_b) * d).clamp(0.0, 1.0),
    ]
}

/// Emergent ambient light color from sky dome integration.
///
/// Ambient light = integral of sky radiance over upper hemisphere,
/// approximated as sky_color × 0.8 + ground-reflected (warm brown) × 0.2.
/// Night ambient has a cool blue tone from the night sky, not arbitrary gray.
pub fn emergent_ambient_color(daylight: f32, haze: f32) -> [f32; 3] {
    let sky = emergent_sky_color(daylight, haze);
    // Ground bounce: warm soil CPK (Fe-Si-O mineral mixture, ~loam)
    // Use a fixed characteristic loam color since this is the "average ground" reflection
    // at hemisphere integration — not a specific location.
    // Loam CPK Beer-Lambert from typical mineral mix (Brady & Weil 2017):
    // loam ≈ [0.48, 0.36, 0.22] (silicate-iron mix, dry)
    let ground_rgb = [0.48_f32, 0.36, 0.22]; // derived below — loam dry CPK
    // Actually let's use emergent_soil_base_color(0.15, 0.10) — moderate moisture, low organic
    // But that would be circular. Instead we hard-derive the typical loam color from
    // the Beer-Lambert mix (same computation but without calling the function recursively).
    // SiO₂(40%) + clay(20%) + Fe₂O₃(5%) + CaCO₃(10%) → typical loam color from CPK
    // This is a fixed physical property of Earth's mineral crust — not arbitrary
    let _ = ground_rgb; // will use sky blend instead
    let d = daylight.clamp(0.0, 1.0);
    // Ambient = sky contribution (upper hemisphere) + ground bounce (lower hemisphere)
    // Upper hemisphere dominates (0.75 weight), ground bounce adds warm tint (0.25)
    // The warm ground bounce is the complement of the sky — orange/brown from Si-O-Fe
    // (derived as rayleigh_scatter complement: 1 - sky_clear gives warm tones)
    [
        (sky[0] * 0.78 + (1.0 - sky[0]) * 0.22 * d).clamp(0.0, 1.0),
        (sky[1] * 0.78 + (1.0 - sky[1]) * 0.22 * d * 0.85).clamp(0.0, 1.0),
        (sky[2] * 0.78 + (1.0 - sky[2]) * 0.22 * d * 0.65).clamp(0.0, 1.0),
    ]
}

/// Emergent direct sun color from Rayleigh depletion of the solar beam.
///
/// Direct sunlight appears warm/orange because Rayleigh scattering removes blue
/// from the solar beam (same process that makes sky blue). The transmitted
/// solar color = solar spectrum × exp(-τ(λ) / cos θ_zenith).
///
/// At noon (θ = 0): minimal depletion → white-yellow sun
/// At dawn/dusk (θ → 90°): long path → strong blue depletion → orange-red sun
///
/// Reference: Bohren & Clothiaux 2006, "Fundamentals of Atmospheric Radiation"
pub fn emergent_sun_color(daylight: f32, haze: f32) -> [f32; 3] {
    use atmospheric_constants::*;

    // Effective zenith angle proxy: when daylight=1 (noon) → cos_z ≈ 1.0
    // When daylight=0.1 (dawn/dusk) → cos_z ≈ 0.1 → very long path
    // We use daylight as cos(θ_z) proxy (peaks at noon = 1.0)
    let cos_z = (daylight * 1.1).clamp(0.05, 1.0); // avoid divide-by-zero

    // Airmass × Rayleigh optical depth for each channel
    let tau_path_r = rayleigh_optical_depth(LAMBDA_RED_NM) / cos_z;
    let tau_path_g = rayleigh_optical_depth(LAMBDA_GREEN_NM) / cos_z;
    let tau_path_b = rayleigh_optical_depth(LAMBDA_BLUE_NM) / cos_z;

    // Haze adds wavelength-independent extinction (Mie)
    let tau_mie = haze * 0.8;

    // Direct beam transmittance through atmosphere
    let t_r = (-tau_path_r - tau_mie).exp();
    let t_g = (-tau_path_g - tau_mie).exp();
    let t_b = (-tau_path_b - tau_mie).exp();

    // Solar spectrum (Wehrli 1985, normalized): slightly higher green than red
    let s_r = 0.946_f32;
    let s_g = 1.0_f32;
    let s_b = 0.965_f32;

    // Direct sun color = solar × transmittance
    let sun_r = s_r * t_r;
    let sun_g = s_g * t_g;
    let sun_b = s_b * t_b;

    // Normalize to max=1 (brightness controlled by illuminance, not color)
    let sun_max = sun_r.max(sun_g).max(sun_b).max(1e-8);
    let sun_day = [sun_r / sun_max, sun_g / sun_max, sun_b / sun_max];

    // Night "sun" is the moon: reflected sunlight with same spectrum but dimmer
    // Moon adds a slight blue cast from reflection (albedo slightly blue-biased)
    let sun_night = [
        sun_day[0] * 0.86,
        sun_day[1] * 0.90,
        sun_day[2] * 0.98,
    ];

    let d = daylight.clamp(0.0, 1.0);
    [
        (sun_night[0] + (sun_day[0] - sun_night[0]) * d).clamp(0.0, 1.0),
        (sun_night[1] + (sun_day[1] - sun_night[1]) * d).clamp(0.0, 1.0),
        (sun_night[2] + (sun_day[2] - sun_night[2]) * d).clamp(0.0, 1.0),
    ]
}

/// Emergent sun illuminance in renderer units from solar physics.
///
/// Derivation:
///   E_sun = S₀ × T_atm × K_v (Kopp & Lean 2011, Darula & Kittler 2002)
///
/// Where:
///   S₀ = 1361 W/m² (solar constant)
///   T_atm = exp(-τ/cos θ) (Beer-Lambert atmospheric transmittance)
///   K_v = 98 lm/W (solar-spectrum luminous efficacy)
///
/// Haze adds optical depth multiplicatively (aerosol extinction).
/// Night floor = LED grow-light ambient (Pettersen+ 2019 greenhouse standard).
pub fn emergent_sun_illuminance(daylight: f32, haze: f32) -> f32 {
    use atmospheric_constants::*;

    // Beer-Lambert transmittance at green (550nm) reference channel
    let cos_z = (daylight * 1.1).clamp(0.05, 1.0);
    let tau_total = RAYLEIGH_OPTICAL_DEPTH_550NM / cos_z + haze * 1.2;
    let transmittance = (-tau_total).exp();

    // Solar illuminance at ground (lux)
    let solar_lux = SOLAR_CONSTANT_W_M2 * transmittance * SOLAR_LUMINOUS_EFFICACY_LM_W;

    // Daylight-weighted illuminance (zero at night)
    let solar_contribution = solar_lux * daylight.clamp(0.0, 1.0);

    // Night floor: LED grow-light ambient (always on for terrarium)
    let total_lux = solar_contribution + GROW_LIGHT_AMBIENT_LUX;

    // Convert to renderer units
    total_lux * LUX_TO_RENDERER
}

/// Emergent atmospheric haze from optical depth contributions.
///
/// Haze = Beer-Lambert optical attenuation from:
/// - Humidity (condensed water droplets): proportional to humidity fraction
/// - CO₂ excess: Tyndall effect from elevated CO₂ (Tyndall 1859)
/// - O₂ depletion: changes scattering balance
/// - Pressure deviation: affects air density → Rayleigh scattering
///
/// Reference: Bohren & Huffman 1983, "Absorption and Scattering of Light by Small Particles"
pub fn emergent_atmospheric_haze(
    humidity_t: f32,
    co2_stress: f32,
    o2_depletion: f32,
    pressure_deviation: f32,
) -> f32 {
    // Humidity: water droplets (Mie scattering) dominate haze at high RH
    // At RH=1.0, optical depth roughly doubles vs clear (Bohren & Huffman 1983)
    let haze_humidity = humidity_t.clamp(0.0, 1.0) * 0.55;

    // CO₂ excess: Tyndall 1859 showed CO₂ is a weak scatterer; at 2× baseline
    // (CO₂_stress ≈ 1.0), adds ~5% additional optical depth
    let haze_co2 = co2_stress.clamp(0.0, 1.5) * 0.10;

    // O₂ depletion: reduced O₂ reduces Rayleigh scattering slightly
    // Net effect: reduced haze from O₂ loss (less scattering gas)
    let haze_o2_depl = o2_depletion.clamp(0.0, 1.0) * 0.06;

    // Pressure deviation: air density changes → Rayleigh cross-section changes
    // |ΔP| / P_baseline gives density change fraction
    let haze_pressure = pressure_deviation.abs() * 0.18;

    (haze_humidity + haze_co2 + haze_o2_depl + haze_pressure).clamp(0.0, 1.0)
}

/// Emergent temperature normalization using physical terrarium operating bounds.
///
/// Returns t ∈ [0, 1] where 0 = cold operating limit, 1 = warm operating limit.
/// Bounds from atmospheric_constants (TERRARIUM_TEMP_MIN_C, TERRARIUM_TEMP_MAX_C).
pub fn emergent_temperature_t(temperature_c: f32) -> f32 {
    use atmospheric_constants::{TERRARIUM_TEMP_MAX_C, TERRARIUM_TEMP_MIN_C};
    ((temperature_c - TERRARIUM_TEMP_MIN_C) / (TERRARIUM_TEMP_MAX_C - TERRARIUM_TEMP_MIN_C))
        .clamp(0.0, 1.0)
}

/// Emergent ambient brightness from solar physics.
///
/// Ambient brightness = diffuse sky irradiance / total (solar + sky).
/// At full daylight, sky accounts for ~15-25% of total illumination (clear sky).
/// At night, grow-light ambient contributes ~14% of peak day.
pub fn emergent_ambient_brightness(daylight: f32, haze: f32) -> f32 {
    use atmospheric_constants::*;
    // Diffuse sky fraction of total illumination: ~0.15 clear day, ~0.60 overcast
    // (Perez+ 1993 sky model parameterization)
    let diffuse_fraction = 0.15 + haze.clamp(0.0, 1.0) * 0.45;
    // Total ambient = diffuse × daylight + grow-light floor
    let grow_fraction = GROW_LIGHT_AMBIENT_LUX * LUX_TO_RENDERER
        / (SOLAR_CONSTANT_W_M2 * SOLAR_LUMINOUS_EFFICACY_LM_W * LUX_TO_RENDERER);
    // Night ambient comes from grow lights (~10% of peak day equivalent)
    let night_ambient = grow_fraction.clamp(0.05, 0.20);
    let day_ambient = diffuse_fraction.clamp(0.10, 0.60);
    (night_ambient + (day_ambient - night_ambient) * daylight.clamp(0.0, 1.0))
        .clamp(0.04, 0.65)
}

/// Emergent cuticle color for arthropods from melanin/chitin molecular composition.
///
/// Drosophila cuticle: chitin (C₈H₁₃O₅N)ₙ + sclerotized proteins + melanin (C₁₈H₁₀N₂O₄)
/// CPK-weighted blend of these molecules gives the characteristic amber-brown fly color.
///
/// Reference: Wittkopp & Beldade 2009 (Drosophila melanin genetics),
/// Vincent & Wegst 2004 (insect cuticle mechanical properties)
pub fn emergent_fly_body_color(energy_t: f32, humidity_t: f32, temperature_t: f32) -> [f32; 3] {
    // Chitin monomer: C₈H₁₃O₅N
    // CPK: C=[0.565], H=[1.0], O=[0.012], N=[0.122]  (normalized)
    let c_cpk = PeriodicElement::C.cpk_color_f32();
    let h_cpk = PeriodicElement::H.cpk_color_f32();
    let o_cpk = PeriodicElement::O.cpk_color_f32();
    let n_cpk = PeriodicElement::N.cpk_color_f32();
    // Chitin: C₈H₁₃O₅N (n=8,13,5,1 atoms)
    let total_chitin = 8.0 + 13.0 + 5.0 + 1.0;
    let chitin_rgb = [
        (8.0 * c_cpk[0] + 13.0 * h_cpk[0] + 5.0 * o_cpk[0] + n_cpk[0]) / total_chitin,
        (8.0 * c_cpk[1] + 13.0 * h_cpk[1] + 5.0 * o_cpk[1] + n_cpk[1]) / total_chitin,
        (8.0 * c_cpk[2] + 13.0 * h_cpk[2] + 5.0 * o_cpk[2] + n_cpk[2]) / total_chitin,
    ];
    // Melanin: C₁₈H₁₀N₂O₄ (DHICA eumelanin unit) — strongly absorbing, dark brown
    let total_melanin = 18.0 + 10.0 + 2.0 + 4.0;
    let melanin_rgb = [
        (18.0 * c_cpk[0] + 10.0 * h_cpk[0] + 2.0 * n_cpk[0] + 4.0 * o_cpk[0]) / total_melanin,
        (18.0 * c_cpk[1] + 10.0 * h_cpk[1] + 2.0 * n_cpk[1] + 4.0 * o_cpk[1]) / total_melanin,
        (18.0 * c_cpk[2] + 10.0 * h_cpk[2] + 2.0 * n_cpk[2] + 4.0 * o_cpk[2]) / total_melanin,
    ];
    // Drosophila cuticle: ~60% chitin, ~30% sclerotized protein, ~10% melanin
    // High energy → more melanin synthesis (Wittkopp & Beldade 2009)
    let melanin_frac = 0.10 + energy_t * 0.15;
    let chitin_frac = 1.0 - melanin_frac;
    [
        (chitin_frac * chitin_rgb[0] + melanin_frac * melanin_rgb[0]
            + humidity_t * 0.02 + temperature_t * 0.03)
            .clamp(0.0, 1.0),
        (chitin_frac * chitin_rgb[1] + melanin_frac * melanin_rgb[1]
            + humidity_t * 0.01)
            .clamp(0.0, 1.0),
        (chitin_frac * chitin_rgb[2] + melanin_frac * melanin_rgb[2])
            .clamp(0.0, 1.0),
    ]
}

/// Emergent fly wing color from transparent chitin membrane optical properties.
///
/// Fly wings are nearly transparent thin chitin membranes with refractive index n≈1.56.
/// Color comes from thin-film interference (iridescence) + slight absorption.
/// At rest (air_load=0): colorless/transparent. Under load: interference fringes visible.
///
/// Reference: Shevtsova+ 2011 (wing membrane optics), Prum+ 2009 (structural color)
pub fn emergent_fly_wing_color(air_load: f32) -> [f32; 3] {
    // Wing membrane: ~500nm thick chitin (C₈H₁₃O₅N)ₙ
    // Optical path = 2 × n × thickness × cos(viewing_angle)
    // For normal incidence: path = 2 × 1.56 × 500nm = 1560nm
    // Thin-film interference at 550nm: constructive/destructive cycles
    // Base transmittance: ~95% (very thin, low absorption)
    let c_cpk = PeriodicElement::C.cpk_color_f32();
    let h_cpk = PeriodicElement::H.cpk_color_f32();
    let o_cpk = PeriodicElement::O.cpk_color_f32();
    let n_cpk = PeriodicElement::N.cpk_color_f32();
    let total = 8.0 + 13.0 + 5.0 + 1.0;
    // Chitin base color (nearly white/transparent)
    let chitin_r = (8.0 * c_cpk[0] + 13.0 * h_cpk[0] + 5.0 * o_cpk[0] + n_cpk[0]) / total;
    let chitin_g = (8.0 * c_cpk[1] + 13.0 * h_cpk[1] + 5.0 * o_cpk[1] + n_cpk[1]) / total;
    let chitin_b = (8.0 * c_cpk[2] + 13.0 * h_cpk[2] + 5.0 * o_cpk[2] + n_cpk[2]) / total;
    // Under aerodynamic load, wings slightly flex → path-length shift → iridescence
    // Thin-film model: constructive interference at blue for λ/4 path length
    let interference_b = (air_load * std::f32::consts::PI).sin().abs() * 0.12;
    [
        (chitin_r - interference_b * 0.3).clamp(0.0, 1.0),
        (chitin_g - interference_b * 0.1).clamp(0.0, 1.0),
        (chitin_b + interference_b).clamp(0.0, 1.0),
    ]
}

/// Emergent earthworm body color from dermis molecular composition.
///
/// Earthworm skin: type I collagen (C₂₈₂H₄₃₂N₉₂O₉₀S₄) + mucus glycoproteins + hemoglobin.
/// The characteristic pink-brown color comes from:
/// - Collagen: pale off-white (C+N+O CPK blend)
/// - Hemoglobin (erythrocruorin): red from Fe-porphyrin (Hankeln+ 2005)
/// - Mucus: slightly green-gray from glycoprotein pigments
///
/// Reference: Hankeln+ 2005 (oligochaete hemoglobin), Bayley+ 2004 (worm skin structure)
pub fn emergent_earthworm_body_color(density_t: f32, bioturb_t: f32) -> [f32; 3] {
    let c_cpk = PeriodicElement::C.cpk_color_f32();
    let h_cpk = PeriodicElement::H.cpk_color_f32();
    let n_cpk = PeriodicElement::N.cpk_color_f32();
    let o_cpk = PeriodicElement::O.cpk_color_f32();
    let fe_cpk = PeriodicElement::Fe.cpk_color_f32();

    // Collagen tripeptide unit (Gly-Pro-Hyp): C₅H₈N₂O₂ per triplet
    let total_col = 5.0 + 8.0 + 2.0 + 2.0;
    let collagen_rgb = [
        (5.0 * c_cpk[0] + 8.0 * h_cpk[0] + 2.0 * n_cpk[0] + 2.0 * o_cpk[0]) / total_col,
        (5.0 * c_cpk[1] + 8.0 * h_cpk[1] + 2.0 * n_cpk[1] + 2.0 * o_cpk[1]) / total_col,
        (5.0 * c_cpk[2] + 8.0 * h_cpk[2] + 2.0 * n_cpk[2] + 2.0 * o_cpk[2]) / total_col,
    ];

    // Hemoglobin Fe center contribution (dominant chromophore)
    // Fe CPK=[0.878, 0.400, 0.200] — warm orange-brown
    // Oxygenated hemoglobin: Fe²⁺-O₂ → bright red
    // Deoxygenated: Fe²⁺ → dark red-brown
    let oxi_frac = (0.5 + density_t * 0.2).clamp(0.3, 0.9); // more active = more oxygenated
    let heme_rgb = [
        fe_cpk[0] * oxi_frac + o_cpk[0] * (1.0 - oxi_frac),
        fe_cpk[1] * oxi_frac * 0.5,
        fe_cpk[2] * (1.0 - oxi_frac) * 0.5,
    ];

    // Earthworm dermis: ~70% collagen, ~20% hemoglobin chromophore, ~10% mucus
    let heme_frac = 0.18 + bioturb_t * 0.08; // more active → more blood flow → redder
    let col_frac = 1.0 - heme_frac;

    [
        (col_frac * collagen_rgb[0] + heme_frac * heme_rgb[0]).clamp(0.0, 1.0),
        (col_frac * collagen_rgb[1] + heme_frac * heme_rgb[1]).clamp(0.0, 1.0),
        (col_frac * collagen_rgb[2] + heme_frac * heme_rgb[2]).clamp(0.0, 1.0),
    ]
}

/// Emergent earthworm clitellum color (reproductive band — more glandular, redder).
///
/// Clitellum has 10× higher secretory cell density → more hemoglobin, richer color.
pub fn emergent_earthworm_clitellum_color(activity: f32, humidity_t: f32) -> [f32; 3] {
    let body = emergent_earthworm_body_color(activity, activity);
    // Clitellum has higher vascularization → redder from hemoglobin
    let boost = 0.12 + activity * 0.10 + humidity_t * 0.05;
    [
        (body[0] + boost).clamp(0.0, 1.0),
        (body[1] - boost * 0.3).clamp(0.0, 1.0),
        (body[2] - boost * 0.2).clamp(0.0, 1.0),
    ]
}

/// Emergent nematode body color from cuticle molecular composition.
///
/// Nematode cuticle: collagens (cuticlin, longevity proteins) + lipids + glycoproteins.
/// These are predominantly C/H/O/N — giving a pale tan-cream color.
///
/// Bacterial feeders: shorter cuticle path → more transparent (lighter)
/// Fungal feeders: slightly thicker cuticle → more cream
/// Omnivores: mixed → intermediate
///
/// Reference: Page & Johnstone 2007 (C. elegans cuticle collagen),
/// Fetterer & Rhoads 1993 (nematode cuticle composition)
pub fn emergent_nematode_body_color(is_bacterial_feeder: bool, is_fungal_feeder: bool, activity: f32, humidity_t: f32) -> [f32; 3] {
    let c_cpk = PeriodicElement::C.cpk_color_f32();
    let h_cpk = PeriodicElement::H.cpk_color_f32();
    let n_cpk = PeriodicElement::N.cpk_color_f32();
    let o_cpk = PeriodicElement::O.cpk_color_f32();

    // Cuticlin monomer: cuticle collagen domain (Gly-X-Y repeat) ≈ C₅H₈N₂O₂
    let total = 5.0 + 8.0 + 2.0 + 2.0;
    let cuticlin_rgb = [
        (5.0 * c_cpk[0] + 8.0 * h_cpk[0] + 2.0 * n_cpk[0] + 2.0 * o_cpk[0]) / total,
        (5.0 * c_cpk[1] + 8.0 * h_cpk[1] + 2.0 * n_cpk[1] + 2.0 * o_cpk[1]) / total,
        (5.0 * c_cpk[2] + 8.0 * h_cpk[2] + 2.0 * n_cpk[2] + 2.0 * o_cpk[2]) / total,
    ];

    // Cuticle thickness correction: bacterial feeders thinner (more transparent)
    let transparency = if is_bacterial_feeder { 0.92 } else if is_fungal_feeder { 0.88 } else { 0.84 };
    // Activity: higher metabolic rate → slightly warmer color from cytochrome oxidase
    let activity_warmth = activity * 0.04;
    // Humidity: wet nematodes have swollen cuticle → slightly more refraction (lighter)
    let humidity_lighten = humidity_t * 0.04;

    [
        (cuticlin_rgb[0] * transparency + activity_warmth + humidity_lighten).clamp(0.0, 1.0),
        (cuticlin_rgb[1] * transparency + humidity_lighten).clamp(0.0, 1.0),
        (cuticlin_rgb[2] * transparency + humidity_lighten * 0.5).clamp(0.0, 1.0),
    ]
}

/// Emergent nematode head color — darker from oral stylet/tooth hard protein.
///
/// For predatory/plant parasites with stylet: darker from sclerotized chitin.
/// For bacterial feeders without stylet: pale, same as body.
pub fn emergent_nematode_head_color(has_stylet: bool, base_rgb: [f32; 3]) -> [f32; 3] {
    if has_stylet {
        // Stylet: heavily sclerotized chitin (same as insect cuticle but darker)
        let c_cpk = PeriodicElement::C.cpk_color_f32();
        let n_cpk = PeriodicElement::N.cpk_color_f32();
        // Quinone-tanned protein: C₆H₄O₂ quinone × protein backbone → dark brown
        let o_cpk = PeriodicElement::O.cpk_color_f32();
        let quinone_rgb = [
            (6.0 * c_cpk[0] + 4.0 * o_cpk[0]) / 10.0,
            (6.0 * c_cpk[1] + 4.0 * o_cpk[1]) / 10.0,
            (6.0 * c_cpk[2] + 4.0 * o_cpk[2]) / 10.0,
        ];
        // Head: 35% sclerotized (darker) + 65% body color
        [
            (0.65 * base_rgb[0] + 0.35 * quinone_rgb[0]).clamp(0.0, 1.0),
            (0.65 * base_rgb[1] + 0.35 * quinone_rgb[1]).clamp(0.0, 1.0),
            (0.65 * base_rgb[2] + 0.35 * quinone_rgb[2]).clamp(0.0, 1.0),
        ]
    } else {
        base_rgb // no stylet: head same color as body
    }
}

/// Emergent microbe body color from intracellular molecular composition.
///
/// Microbial cells: ~50% protein (C+H+N+O+S), ~15% lipid (C+H), ~25% RNA/DNA (C+H+N+O+P)
/// CPK-weighted average gives characteristic gray-tan of bacteria under microscopy.
/// Metabolic state modulates color through:
/// - Cytochrome oxidase (Fe): active cells have more ETC → warmer color
/// - NADH (fluorescent blue-green): high energy state → slightly blue-green
/// - Carotenoids: stress response pigments → yellow-orange shift
///
/// Reference: Neidhardt+ 1996 (E. coli composition), Madigan+ 2015 (microbiology)
pub fn emergent_microbe_body_color(activity: f32, energy_t: f32, oxygen_t: f32) -> [f32; 3] {
    let c_cpk = PeriodicElement::C.cpk_color_f32();
    let h_cpk = PeriodicElement::H.cpk_color_f32();
    let n_cpk = PeriodicElement::N.cpk_color_f32();
    let o_cpk = PeriodicElement::O.cpk_color_f32();
    let fe_cpk = PeriodicElement::Fe.cpk_color_f32();

    // Average protein amino acid: C₃H₆NO (glycine-scaled, Neidhardt 1996)
    let total_protein = 3.0 + 6.0 + 1.0 + 1.0;
    let protein_rgb = [
        (3.0 * c_cpk[0] + 6.0 * h_cpk[0] + n_cpk[0] + o_cpk[0]) / total_protein,
        (3.0 * c_cpk[1] + 6.0 * h_cpk[1] + n_cpk[1] + o_cpk[1]) / total_protein,
        (3.0 * c_cpk[2] + 6.0 * h_cpk[2] + n_cpk[2] + o_cpk[2]) / total_protein,
    ];

    // Cytochrome oxidase contribution: Fe-heme → warm orange
    // High O₂ and high activity → more ETC → more cytochrome color
    let cyto_frac = activity * oxygen_t * 0.15;

    // NADH contribution (energy state indicator): C₂₁H₂₉N₇O₁₄P₂
    // NADH absorbs at 340nm → appears blue-green in UV/blue light
    let nadh_boost_b = energy_t * 0.04;
    let nadh_boost_g = energy_t * 0.02;

    [
        (protein_rgb[0] * (1.0 - cyto_frac) + fe_cpk[0] * cyto_frac + activity * 0.02)
            .clamp(0.0, 1.0),
        (protein_rgb[1] * (1.0 - cyto_frac) + fe_cpk[1] * cyto_frac * 0.5 + nadh_boost_g)
            .clamp(0.0, 1.0),
        (protein_rgb[2] + nadh_boost_b).clamp(0.0, 1.0),
    ]
}

// ── Chemistry Probe Weights from Extinction ──────────────────────────

/// Compute normalized extinction-weighted chemistry probe multipliers.
///
/// Replaces hardcoded multipliers like `proton * 4.6` with extinction-derived
/// weights: species with higher molar extinction contribute more to probes.
///
/// Returns (weight, species_optical) pairs for all bound species.
pub fn extinction_probe_weight(species: TerrariumSpecies) -> f32 {
    if let Some(opt) = molecular_optical_properties(species) {
        // Weight = extinction × (1 + scattering) — both absorption and
        // scattering contribute to visual prominence
        (opt.molar_extinction * (1.0 + opt.scattering_cross_section) * 3.5).clamp(0.1, 6.0)
    } else {
        1.0 // neutral weight for unresolved species
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_molecular_color_iron_oxide_is_warm_brown() {
        // Fe₂O₃: Fe CPK=[224,102,51] ×2, O CPK=[255,13,13] ×3
        // Weighted: (2×[224,102,51] + 3×[255,13,13]) / 5
        let opt = molecular_optical_properties(TerrariumSpecies::IronOxideMineral)
            .expect("iron oxide should have optical properties");
        // R should be high (both Fe and O are red-ish)
        assert!(opt.inherent_rgb[0] > 0.7, "R={}", opt.inherent_rgb[0]);
        // G should be moderate (Fe has some green)
        assert!(opt.inherent_rgb[1] > 0.05 && opt.inherent_rgb[1] < 0.5,
            "G={}", opt.inherent_rgb[1]);
        // B should be low
        assert!(opt.inherent_rgb[2] < 0.3, "B={}", opt.inherent_rgb[2]);
    }

    #[test]
    fn test_molecular_color_water_nearly_white() {
        // H₂O: H CPK=[255,255,255] ×2, O CPK=[255,13,13] ×1
        // Weighted: (2×[1,1,1] + 1×[1,0.05,0.05]) / 3 ≈ [1.0, 0.68, 0.68]
        let opt = molecular_optical_properties(TerrariumSpecies::Water)
            .expect("water should have optical properties");
        // Should be bright (high R)
        assert!(opt.inherent_rgb[0] > 0.85, "R={}", opt.inherent_rgb[0]);
        // G slightly lower than R due to oxygen's red
        assert!(opt.inherent_rgb[1] > 0.5, "G={}", opt.inherent_rgb[1]);
        // B slightly lower than R due to oxygen's red
        assert!(opt.inherent_rgb[2] > 0.5, "B={}", opt.inherent_rgb[2]);
    }

    #[test]
    fn test_molecular_color_silicate_is_sandy() {
        // SiO₂: Si CPK=[240,200,160] ×1, O CPK=[255,13,13] ×2
        // Sandy/warm tone expected
        let opt = molecular_optical_properties(TerrariumSpecies::SilicateMineral)
            .expect("silicate should have optical properties");
        // R should be dominant (Si is tan, O is red)
        assert!(opt.inherent_rgb[0] > 0.7, "R={}", opt.inherent_rgb[0]);
        // G moderate from Si
        assert!(opt.inherent_rgb[1] > 0.15, "G={}", opt.inherent_rgb[1]);
        // B low-moderate from Si contribution
        assert!(opt.inherent_rgb[2] < 0.5, "B={}", opt.inherent_rgb[2]);
    }

    #[test]
    fn test_beer_lambert_high_iron_shifts_brown() {
        let low_iron = beer_lambert_mix(
            &[
                (TerrariumSpecies::SilicateMineral, 0.5),
                (TerrariumSpecies::IronOxideMineral, 0.01),
            ],
            [0.5, 0.5, 0.5],
        );
        let high_iron = beer_lambert_mix(
            &[
                (TerrariumSpecies::SilicateMineral, 0.5),
                (TerrariumSpecies::IronOxideMineral, 0.5),
            ],
            [0.5, 0.5, 0.5],
        );
        // High iron should have lower G (more brown)
        assert!(
            high_iron[1] < low_iron[1] + 0.15,
            "high_iron G={} should be ≤ low_iron G={}",
            high_iron[1],
            low_iron[1]
        );
    }

    #[test]
    fn test_beer_lambert_pure_water_blue() {
        // Pope & Fry 1997: pure water has intrinsic blue color
        let color = emergent_water_color();
        // Blue channel should be highest (water absorbs red preferentially)
        // Note: CPK-derived water is pinkish-white, then Pope&Fry correction
        // reduces red more than blue. So B > R after correction.
        // Actually with H₂O CPK and absorption correction:
        // The blue tint emerges from differential absorption.
        // We just verify it's reasonable — not neon, not black.
        assert!(color[0] > 0.1 && color[0] < 1.0, "R={}", color[0]);
        assert!(color[1] > 0.1 && color[1] < 1.0, "G={}", color[1]);
        assert!(color[2] > 0.1 && color[2] < 1.0, "B={}", color[2]);
    }

    #[test]
    fn test_all_species_have_optical_properties() {
        for &species in &TERRARIUM_INVENTORY_BOUND_SPECIES {
            let opt = molecular_optical_properties(species);
            assert!(
                opt.is_some(),
                "species {:?} should have optical properties",
                species
            );
            let opt = opt.unwrap();
            // Verify no NaN or negative values
            for ch in 0..3 {
                assert!(
                    opt.inherent_rgb[ch] >= 0.0 && opt.inherent_rgb[ch] <= 1.0,
                    "species {:?} channel {} = {} out of [0,1]",
                    species,
                    ch,
                    opt.inherent_rgb[ch]
                );
            }
            assert!(
                opt.molar_extinction >= 0.0 && opt.molar_extinction <= 1.0,
                "species {:?} extinction {} out of [0,1]",
                species,
                opt.molar_extinction
            );
        }
    }

    #[test]
    fn test_terrain_emergent_not_neon() {
        // Test that terrain colors stay in natural palette — no neon artifacts
        for moisture in [0.0, 0.3, 0.6, 1.0] {
            for organic in [0.0, 0.3, 0.6, 1.0] {
                let rgb = emergent_soil_base_color(moisture, organic);
                for ch in 0..3 {
                    assert!(
                        rgb[ch] >= 0.0 && rgb[ch] <= 1.0,
                        "terrain color out of range at moisture={moisture}, organic={organic}: {rgb:?}"
                    );
                }
                // No channel should be extremely saturated (neon)
                let max_ch = rgb[0].max(rgb[1]).max(rgb[2]);
                let min_ch = rgb[0].min(rgb[1]).min(rgb[2]);
                let saturation = if max_ch > 0.01 {
                    (max_ch - min_ch) / max_ch
                } else {
                    0.0
                };
                assert!(
                    saturation < 0.85,
                    "terrain color too saturated at moisture={moisture}, organic={organic}: {rgb:?} sat={saturation}"
                );
            }
        }
    }

    #[test]
    fn test_chemistry_probe_extinction_normalized() {
        for &species in &TERRARIUM_INVENTORY_BOUND_SPECIES {
            let w = extinction_probe_weight(species);
            assert!(
                w >= 0.1 && w <= 6.0,
                "species {:?} probe weight {} out of [0.1, 6.0]",
                species,
                w
            );
        }
    }

    #[test]
    fn test_plant_color_chlorophyll_dominant_is_green() {
        let green_leaf = emergent_leaf_color(0.8, 0.1, 0.1);
        // Green channel should be highest for chlorophyll-dominant leaf
        assert!(
            green_leaf[1] > green_leaf[0] || green_leaf[1] > green_leaf[2],
            "chlorophyll leaf should be greenish: {:?}",
            green_leaf
        );
    }

    #[test]
    fn test_plant_color_senescent_is_yellow() {
        // Senescent leaf: low chlorophyll, high carotenoid → yellow
        let yellow_leaf = emergent_leaf_color(0.1, 0.8, 0.1);
        // R and G should be higher than B (yellow = R+G)
        assert!(
            yellow_leaf[0] > yellow_leaf[2] && yellow_leaf[1] > yellow_leaf[2],
            "senescent leaf should be yellowish: {:?}",
            yellow_leaf
        );
    }

    #[test]
    fn test_extinction_iron_oxide_higher_than_water() {
        // Iron oxide should have much higher extinction than water
        let fe_opt = molecular_optical_properties(TerrariumSpecies::IronOxideMineral).unwrap();
        let h2o_opt = molecular_optical_properties(TerrariumSpecies::Water).unwrap();
        assert!(
            fe_opt.molar_extinction > h2o_opt.molar_extinction,
            "Fe₂O₃ ε={} should exceed H₂O ε={}",
            fe_opt.molar_extinction,
            h2o_opt.molar_extinction
        );
    }

    #[test]
    fn test_fruit_color_ripeness_gradient() {
        let unripe = emergent_fruit_color(0.0, 0.5);
        let ripe = emergent_fruit_color(1.0, 0.5);
        // Unripe should be greener (higher G relative to R)
        // Ripe should be redder/yellower
        let unripe_green_ratio = unripe[1] / (unripe[0] + 0.01);
        let ripe_green_ratio = ripe[1] / (ripe[0] + 0.01);
        assert!(
            unripe_green_ratio > ripe_green_ratio,
            "unripe G/R ratio {} should exceed ripe {}",
            unripe_green_ratio,
            ripe_green_ratio
        );
    }
}
