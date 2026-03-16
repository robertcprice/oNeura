//! Multi-species plant competition for light and soil nutrients.
//!
//! Implements two mechanisms of resource competition:
//!
//! 1. **Light competition (asymmetric)**: Taller plants shade shorter ones via
//!    Beer-Lambert attenuation through their canopy. This is inherently asymmetric
//!    -- a tall plant shades a short one but not vice versa (Weiner 1990).
//!
//! 2. **Root competition (symmetric)**: Plants with overlapping root zones share
//!    the same soil nutrient pool, splitting resources in proportion to root
//!    biomass (Casper & Jackson 1997, Schenk 2006).
//!
//! Both mechanisms return per-plant availability factors in [0.0, 1.0] that
//! the caller multiplies into the plant's resource uptake.
//!
//! # References
//!
//! - Weiner J (1990) "Asymmetric competition in plant populations",
//!   Trends in Ecology & Evolution 5:360-364.
//! - Tilman D (1988) "Plant Strategies and the Dynamics and Structure of
//!   Plant Communities", Princeton University Press.
//! - Casper BB, Jackson RB (1997) "Plant competition underground",
//!   Annual Review of Ecology and Systematics 28:545-570.
//! - Schenk HJ (2006) "Root competition: beyond resource depletion",
//!   Journal of Ecology 94:725-739.

use crate::plant_organism::{beer_lambert_transmitted_fraction, EXTINCTION_COEFF_DEFAULT};

// ---------------------------------------------------------------------------
// Data structures for competition inputs
// ---------------------------------------------------------------------------

/// Descriptor for a single plant's canopy, used in light competition.
#[derive(Debug, Clone, Copy)]
pub struct CanopyDescriptor {
    /// Grid x position (cell index).
    pub x: f32,
    /// Grid y position (cell index).
    pub y: f32,
    /// Total height of the plant (mm).
    pub height_mm: f32,
    /// Canopy radius (mm).
    pub canopy_radius_mm: f32,
    /// Leaf area index (m^2 leaf / m^2 ground).
    pub lai: f32,
    /// Beer-Lambert extinction coefficient.
    pub extinction_coeff: f32,
}

/// Descriptor for a single plant's root zone, used in nutrient competition.
#[derive(Debug, Clone, Copy)]
pub struct RootDescriptor {
    /// Grid x position (cell index).
    pub x: f32,
    /// Grid y position (cell index).
    pub y: f32,
    /// Root depth (mm), determines vertical reach into soil.
    pub root_depth_mm: f32,
    /// Root horizontal radius (mm), determines lateral extent.
    pub root_radius_mm: f32,
    /// Root biomass, used to weight nutrient partitioning.
    pub root_biomass: f32,
}

// ---------------------------------------------------------------------------
// Light competition
// ---------------------------------------------------------------------------

/// Compute light competition factors for a set of plants.
///
/// For each plant, finds all taller neighbors whose canopy overlaps
/// horizontally, sums their LAI contribution above this plant's crown,
/// and applies Beer-Lambert attenuation.
///
/// The algorithm is O(n^2) but terraria have few plants (typically <50),
/// so this is not a bottleneck.
///
/// # Arguments
/// * `plants` - canopy descriptors for all plants in the scene
/// * `voxel_size_mm` - size of a single grid cell in mm (for distance calculation)
///
/// # Returns
/// Per-plant light availability factor in [0.0, 1.0]. A value of 1.0 means
/// the plant receives full unattenuated sunlight; 0.0 means it is completely
/// shaded.
///
/// # References
/// Weiner (1990), Tilman (1988), Monsi & Saeki (1953).
pub fn compute_light_competition(
    plants: &[CanopyDescriptor],
    voxel_size_mm: f32,
) -> Vec<f32> {
    let n = plants.len();
    if n == 0 {
        return Vec::new();
    }

    let mut light_factors = vec![1.0f32; n];
    let voxel = voxel_size_mm.max(0.01);

    for i in 0..n {
        let me = &plants[i];
        // Accumulate the total LAI of taller neighbors that shade this plant.
        let mut shading_lai = 0.0f32;

        for j in 0..n {
            if i == j {
                continue;
            }
            let other = &plants[j];
            // Only taller plants cast shade downward.
            if other.height_mm <= me.height_mm {
                continue;
            }

            // Horizontal distance between canopy centers.
            let dx = (me.x - other.x) * voxel;
            let dy = (me.y - other.y) * voxel;
            let dist = (dx * dx + dy * dy).sqrt();

            // The shading plant's canopy radius defines its shadow footprint.
            // A plant is shaded if the distance is less than the sum of the
            // shading plant's canopy radius and the shaded plant's canopy radius
            // (partial overlap).
            let overlap_threshold = other.canopy_radius_mm + me.canopy_radius_mm;
            if dist >= overlap_threshold {
                continue;
            }

            // Overlap fraction: full overlap when dist=0, linear decay to edge.
            let overlap_frac = if overlap_threshold > 0.0 {
                (1.0 - dist / overlap_threshold).clamp(0.0, 1.0)
            } else {
                1.0
            };

            // The effective LAI contribution from this neighbor is proportional
            // to the overlap fraction. This models partial canopy overlap.
            shading_lai += other.lai * overlap_frac;
        }

        // Apply Beer-Lambert attenuation from cumulative shading LAI.
        // Use a default k for the shading canopy (mixed species).
        let transmitted = beer_lambert_transmitted_fraction(shading_lai, EXTINCTION_COEFF_DEFAULT);
        light_factors[i] = transmitted.clamp(0.0, 1.0);
    }

    light_factors
}

// ---------------------------------------------------------------------------
// Root competition
// ---------------------------------------------------------------------------

/// Compute root zone nutrient competition among overlapping plants.
///
/// For each plant, identifies neighbors with overlapping root zones and
/// splits the available nutrient pool in proportion to root biomass.
/// This implements symmetric competition: two equal plants split resources
/// 50/50 regardless of other traits (Casper & Jackson 1997).
///
/// # Arguments
/// * `plants` - root zone descriptors for all plants
/// * `soil_nitrogen` - per-cell available nitrogen (flat 2D grid)
/// * `soil_phosphorus` - per-cell available phosphorus (flat 2D grid)
/// * `grid_width` - width of the 2D soil grid
/// * `grid_height` - height of the 2D soil grid
/// * `voxel_size_mm` - size of a single grid cell in mm
///
/// # Returns
/// Per-plant (N_fraction, P_fraction) in [0.0, 1.0].
/// A fraction of 1.0 means the plant has sole access to its root zone nutrients;
/// 0.5 means it shares equally with one identical competitor.
///
/// # References
/// Casper & Jackson (1997), Schenk (2006).
pub fn compute_root_competition(
    plants: &[RootDescriptor],
    soil_nitrogen: &[f32],
    soil_phosphorus: &[f32],
    grid_width: usize,
    grid_height: usize,
    voxel_size_mm: f32,
) -> Vec<(f32, f32)> {
    let n = plants.len();
    if n == 0 {
        return Vec::new();
    }

    let voxel = voxel_size_mm.max(0.01);
    let mut result = vec![(1.0f32, 1.0f32); n];

    for i in 0..n {
        let me = &plants[i];
        let my_biomass = me.root_biomass.max(1e-9);

        // Total competing root biomass in this plant's root zone (including self).
        let mut total_biomass = my_biomass;
        // Weighted available N and P in the plant's root zone.
        let mut zone_n = 0.0f32;
        let mut zone_p = 0.0f32;
        let mut zone_cells = 0usize;

        // Sample the soil nutrient pool within the root zone.
        let rx = (me.root_radius_mm / voxel).ceil() as i32;
        let cx = (me.x / voxel).round() as i32;
        let cy = (me.y / voxel).round() as i32;

        for dy in -rx..=rx {
            for dx in -rx..=rx {
                let gx = cx + dx;
                let gy = cy + dy;
                if gx < 0 || gy < 0 || gx >= grid_width as i32 || gy >= grid_height as i32 {
                    continue;
                }
                let dist_sq = (dx as f32 * voxel) * (dx as f32 * voxel)
                    + (dy as f32 * voxel) * (dy as f32 * voxel);
                if dist_sq > me.root_radius_mm * me.root_radius_mm {
                    continue;
                }
                let flat = gy as usize * grid_width + gx as usize;
                if flat < soil_nitrogen.len() {
                    zone_n += soil_nitrogen[flat];
                }
                if flat < soil_phosphorus.len() {
                    zone_p += soil_phosphorus[flat];
                }
                zone_cells += 1;
            }
        }

        // Find overlapping competitors and sum their root biomass.
        for j in 0..n {
            if i == j {
                continue;
            }
            let other = &plants[j];
            let dx_mm = (me.x - other.x) * voxel;
            let dy_mm = (me.y - other.y) * voxel;
            let dist = (dx_mm * dx_mm + dy_mm * dy_mm).sqrt();

            // Root zones overlap if distance < sum of radii.
            let overlap_threshold = me.root_radius_mm + other.root_radius_mm;
            if dist >= overlap_threshold {
                continue;
            }

            // Weight the competitor's contribution by overlap fraction.
            let overlap_frac = if overlap_threshold > 0.0 {
                (1.0 - dist / overlap_threshold).clamp(0.0, 1.0)
            } else {
                1.0
            };
            total_biomass += other.root_biomass.max(1e-9) * overlap_frac;
        }

        // This plant's share of available nutrients.
        let my_share = my_biomass / total_biomass;

        // If the zone has nutrients, compute the fraction available to this plant.
        // The competition factor is the plant's biomass share of the total demand.
        let n_fraction = if zone_cells > 0 && zone_n > 0.0 {
            my_share
        } else {
            1.0
        };
        let p_fraction = if zone_cells > 0 && zone_p > 0.0 {
            my_share
        } else {
            1.0
        };

        result[i] = (n_fraction.clamp(0.0, 1.0), p_fraction.clamp(0.0, 1.0));
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beer_lambert_full_sun_no_canopy() {
        // LAI = 0 means no leaves: all light transmitted, nothing intercepted.
        use crate::plant_organism::beer_lambert_par_intercepted;
        let intercepted = beer_lambert_par_intercepted(2000.0, 0.0, 0.5);
        assert!(
            intercepted.abs() < 1e-4,
            "no canopy should intercept nothing: got {}",
            intercepted,
        );
    }

    #[test]
    fn beer_lambert_dense_canopy() {
        // LAI = 6, k = 0.5 -> transmitted = exp(-3.0) ~ 0.0498
        // So intercepted fraction ~ 95%.
        use crate::plant_organism::beer_lambert_par_intercepted;
        let incident = 2000.0;
        let intercepted = beer_lambert_par_intercepted(incident, 6.0, 0.5);
        let transmitted = incident - intercepted;
        let transmitted_frac = transmitted / incident;
        assert!(
            (transmitted_frac - (-3.0f32).exp()).abs() < 0.001,
            "dense canopy LAI=6 k=0.5: expected transmitted fraction ~{:.4}, got {:.4}",
            (-3.0f32).exp(),
            transmitted_frac,
        );
        // About 95% intercepted
        assert!(
            intercepted / incident > 0.94,
            "dense canopy should intercept >94%: got {:.2}%",
            intercepted / incident * 100.0,
        );
    }

    #[test]
    fn taller_plant_shades_shorter() {
        // Two plants at the same position: one tall (100mm), one short (50mm).
        // The short one should receive less light.
        let plants = vec![
            CanopyDescriptor {
                x: 5.0,
                y: 5.0,
                height_mm: 100.0,
                canopy_radius_mm: 10.0,
                lai: 4.0,
                extinction_coeff: 0.5,
            },
            CanopyDescriptor {
                x: 5.0,
                y: 5.0,
                height_mm: 50.0,
                canopy_radius_mm: 8.0,
                lai: 3.0,
                extinction_coeff: 0.5,
            },
        ];
        let light = compute_light_competition(&plants, 3.0);
        assert_eq!(light.len(), 2);
        // Tall plant: no one taller -> full light.
        assert!(
            (light[0] - 1.0).abs() < 1e-6,
            "tall plant should get full light: got {}",
            light[0],
        );
        // Short plant: shaded by tall plant's LAI=4.
        // transmitted = exp(-0.5 * 4.0) = exp(-2.0) ~ 0.135
        let expected = (-0.5f32 * 4.0).exp();
        assert!(
            (light[1] - expected).abs() < 0.02,
            "short plant should be shaded: expected ~{:.3}, got {:.3}",
            expected,
            light[1],
        );
        assert!(
            light[1] < light[0],
            "short plant ({:.3}) should receive less light than tall plant ({:.3})",
            light[1],
            light[0],
        );
    }

    #[test]
    fn root_competition_splits_nutrients() {
        // Two identical plants at the same spot should split nutrients ~50/50.
        let plants = vec![
            RootDescriptor {
                x: 5.0,
                y: 5.0,
                root_depth_mm: 20.0,
                root_radius_mm: 15.0,
                root_biomass: 1.0,
            },
            RootDescriptor {
                x: 5.0,
                y: 5.0,
                root_depth_mm: 20.0,
                root_radius_mm: 15.0,
                root_biomass: 1.0,
            },
        ];
        let width = 20;
        let height = 20;
        let soil_n = vec![0.1f32; width * height];
        let soil_p = vec![0.05f32; width * height];
        let result = compute_root_competition(&plants, &soil_n, &soil_p, width, height, 3.0);
        assert_eq!(result.len(), 2);
        // Both should get ~50%.
        assert!(
            (result[0].0 - 0.5).abs() < 0.05,
            "plant 0 N fraction should be ~0.5: got {}",
            result[0].0,
        );
        assert!(
            (result[1].0 - 0.5).abs() < 0.05,
            "plant 1 N fraction should be ~0.5: got {}",
            result[1].0,
        );
        assert!(
            (result[0].1 - 0.5).abs() < 0.05,
            "plant 0 P fraction should be ~0.5: got {}",
            result[0].1,
        );
        assert!(
            (result[1].1 - 0.5).abs() < 0.05,
            "plant 1 P fraction should be ~0.5: got {}",
            result[1].1,
        );
    }

    #[test]
    fn isolated_plant_no_competition() {
        // A single plant should get full light and full nutrients.
        let canopy = vec![CanopyDescriptor {
            x: 10.0,
            y: 10.0,
            height_mm: 80.0,
            canopy_radius_mm: 8.0,
            lai: 3.0,
            extinction_coeff: 0.5,
        }];
        let light = compute_light_competition(&canopy, 3.0);
        assert_eq!(light.len(), 1);
        assert!(
            (light[0] - 1.0).abs() < 1e-6,
            "isolated plant should get full light: got {}",
            light[0],
        );

        let roots = vec![RootDescriptor {
            x: 10.0,
            y: 10.0,
            root_depth_mm: 20.0,
            root_radius_mm: 12.0,
            root_biomass: 0.8,
        }];
        let width = 20;
        let height = 20;
        let soil_n = vec![0.1f32; width * height];
        let soil_p = vec![0.05f32; width * height];
        let result = compute_root_competition(&roots, &soil_n, &soil_p, width, height, 3.0);
        assert_eq!(result.len(), 1);
        assert!(
            (result[0].0 - 1.0).abs() < 1e-6,
            "isolated plant should get full N: got {}",
            result[0].0,
        );
        assert!(
            (result[0].1 - 1.0).abs() < 1e-6,
            "isolated plant should get full P: got {}",
            result[0].1,
        );
    }

    #[test]
    fn distant_plants_no_shading() {
        // Two plants far apart should not shade each other.
        let plants = vec![
            CanopyDescriptor {
                x: 0.0,
                y: 0.0,
                height_mm: 100.0,
                canopy_radius_mm: 5.0,
                lai: 5.0,
                extinction_coeff: 0.5,
            },
            CanopyDescriptor {
                x: 50.0,
                y: 50.0,
                height_mm: 40.0,
                canopy_radius_mm: 5.0,
                lai: 3.0,
                extinction_coeff: 0.5,
            },
        ];
        let light = compute_light_competition(&plants, 3.0);
        assert!(
            (light[0] - 1.0).abs() < 1e-6,
            "distant tall plant should get full light: got {}",
            light[0],
        );
        assert!(
            (light[1] - 1.0).abs() < 1e-6,
            "distant short plant should get full light: got {}",
            light[1],
        );
    }

    #[test]
    fn unequal_roots_weighted_split() {
        // Plant with 3x root biomass should get ~75% of nutrients.
        let plants = vec![
            RootDescriptor {
                x: 5.0,
                y: 5.0,
                root_depth_mm: 20.0,
                root_radius_mm: 15.0,
                root_biomass: 3.0,
            },
            RootDescriptor {
                x: 5.0,
                y: 5.0,
                root_depth_mm: 20.0,
                root_radius_mm: 15.0,
                root_biomass: 1.0,
            },
        ];
        let width = 20;
        let height = 20;
        let soil_n = vec![0.1f32; width * height];
        let soil_p = vec![0.05f32; width * height];
        let result = compute_root_competition(&plants, &soil_n, &soil_p, width, height, 3.0);
        // Plant 0: 3.0 / (3.0 + 1.0) = 0.75
        assert!(
            (result[0].0 - 0.75).abs() < 0.05,
            "big-root plant N fraction should be ~0.75: got {}",
            result[0].0,
        );
        // Plant 1: 1.0 / (3.0 + 1.0) = 0.25
        assert!(
            (result[1].0 - 0.25).abs() < 0.05,
            "small-root plant N fraction should be ~0.25: got {}",
            result[1].0,
        );
    }

    #[test]
    fn empty_plant_list() {
        let light = compute_light_competition(&[], 3.0);
        assert!(light.is_empty());
        let result = compute_root_competition(&[], &[], &[], 10, 10, 3.0);
        assert!(result.is_empty());
    }

    #[test]
    fn same_height_no_shading() {
        // Two plants of equal height should not shade each other (asymmetric rule).
        let plants = vec![
            CanopyDescriptor {
                x: 5.0,
                y: 5.0,
                height_mm: 60.0,
                canopy_radius_mm: 8.0,
                lai: 4.0,
                extinction_coeff: 0.5,
            },
            CanopyDescriptor {
                x: 5.0,
                y: 5.0,
                height_mm: 60.0,
                canopy_radius_mm: 8.0,
                lai: 4.0,
                extinction_coeff: 0.5,
            },
        ];
        let light = compute_light_competition(&plants, 3.0);
        assert!(
            (light[0] - 1.0).abs() < 1e-6,
            "equal-height plant 0 should get full light: got {}",
            light[0],
        );
        assert!(
            (light[1] - 1.0).abs() < 1e-6,
            "equal-height plant 1 should get full light: got {}",
            light[1],
        );
    }

    #[test]
    fn partial_canopy_overlap_reduces_shading() {
        // A plant partially overlapping should be less shaded than one directly beneath.
        let tall = CanopyDescriptor {
            x: 5.0,
            y: 5.0,
            height_mm: 100.0,
            canopy_radius_mm: 10.0,
            lai: 4.0,
            extinction_coeff: 0.5,
        };
        let directly_beneath = CanopyDescriptor {
            x: 5.0,
            y: 5.0,
            height_mm: 40.0,
            canopy_radius_mm: 6.0,
            lai: 2.0,
            extinction_coeff: 0.5,
        };
        let partially_overlapping = CanopyDescriptor {
            x: 10.0, // further away
            y: 5.0,
            height_mm: 40.0,
            canopy_radius_mm: 6.0,
            lai: 2.0,
            extinction_coeff: 0.5,
        };

        let plants_direct = vec![tall, directly_beneath];
        let plants_partial = vec![tall, partially_overlapping];
        let light_direct = compute_light_competition(&plants_direct, 3.0);
        let light_partial = compute_light_competition(&plants_partial, 3.0);

        assert!(
            light_partial[1] > light_direct[1],
            "partially overlapping plant ({:.3}) should get more light than directly beneath ({:.3})",
            light_partial[1],
            light_direct[1],
        );
    }
}
