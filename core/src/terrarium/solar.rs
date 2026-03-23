//! Solar position model and photon raycasting for physically-accurate light delivery.
//!
//! Implements:
//! 1. Astronomical solar position from simulation time and latitude
//!    (simplified Meeus algorithm for declination, hour angle, elevation, azimuth)
//! 2. Clear-sky PAR (Photosynthetically Active Radiation) with air-mass correction
//! 3. Ray marching through 3D canopy geometry with Beer-Lambert attenuation
//! 4. Per-plant photon flux in µmol/m²/s with light asymmetry for phototropism
//!
//! # References
//! - Meeus J (1991) "Astronomical Algorithms", Willmann-Bell
//! - Monteith JL, Unsworth MH (2013) "Principles of Environmental Physics", Academic Press
//! - Campbell GS, Norman JM (1998) "An Introduction to Biophysical Ecology", Springer

use std::f32::consts::PI;

use crate::plant_competition::CanopyDescriptor;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Peak clear-sky PAR at solar noon, sea level (µmol photons/m²/s).
pub const PAR_FULL_SUN: f32 = 2000.0;

/// Earth's axial tilt (degrees).
const OBLIQUITY_DEG: f32 = 23.44;

/// Minimum sun elevation for meaningful PAR (radians, ~1°).
const MIN_SUN_ELEVATION: f32 = 0.017;

/// Crown-depth fraction: canopy occupies top fraction of plant height.
/// For most growth forms, the leaf-bearing crown is roughly the top 40%.
const CROWN_DEPTH_FRACTION: f32 = 0.40;

// ---------------------------------------------------------------------------
// Solar astronomy
// ---------------------------------------------------------------------------

/// Solar declination (radians) for a given day of year (0-365).
///
/// δ = −ε × cos(2π(d + 10)/365)
fn solar_declination_rad(day_of_year: f32) -> f32 {
    -OBLIQUITY_DEG.to_radians() * (2.0 * PI / 365.0 * (day_of_year + 10.0)).cos()
}

/// Solar elevation angle (radians) above horizon.
///
/// sin(α) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(h)
fn solar_elevation_rad(latitude_rad: f32, declination_rad: f32, hour_angle_rad: f32) -> f32 {
    let sin_elev = latitude_rad.sin() * declination_rad.sin()
        + latitude_rad.cos() * declination_rad.cos() * hour_angle_rad.cos();
    sin_elev.clamp(-1.0, 1.0).asin()
}

/// Solar azimuth angle (radians, 0 = north, π/2 = east, π = south).
fn solar_azimuth_rad(
    latitude_rad: f32,
    declination_rad: f32,
    elevation_rad: f32,
    hour_angle_rad: f32,
) -> f32 {
    let cos_elev = elevation_rad.cos().max(1e-6);
    let lat_cos = latitude_rad.cos().max(1e-6);
    let cos_az =
        (declination_rad.sin() - elevation_rad.sin() * latitude_rad.sin()) / (cos_elev * lat_cos);
    let az = cos_az.clamp(-1.0, 1.0).acos();
    if hour_angle_rad > 0.0 {
        2.0 * PI - az
    } else {
        az
    }
}

/// Day length in hours for given latitude (radians) and day of year.
///
/// Uses the standard sunrise equation: cos(ω₀) = −tan(φ)tan(δ)
pub fn day_length_hours(latitude_rad: f32, day_of_year: f32) -> f32 {
    let decl = solar_declination_rad(day_of_year);
    let cos_hour_angle = -(latitude_rad.tan() * decl.tan());
    if cos_hour_angle <= -1.0 {
        return 24.0; // polar day
    }
    if cos_hour_angle >= 1.0 {
        return 0.0; // polar night
    }
    2.0 * cos_hour_angle.acos() * 12.0 / PI
}

/// Clear-sky PAR at ground level (µmol photons/m²/s) for a given solar elevation.
///
/// Uses Beer-Lambert atmospheric attenuation with air mass approximation:
/// PAR = PAR₀ × τ^(AM) × sin(α)
///
/// where AM = 1/sin(α) is the air mass and τ ≈ 0.76 is the clear-sky transmittance.
fn clear_sky_par(elevation_rad: f32) -> f32 {
    if elevation_rad <= MIN_SUN_ELEVATION {
        return 0.0;
    }
    let sin_elev = elevation_rad.sin();
    let air_mass = (1.0 / sin_elev).min(40.0);
    let transmittance = 0.76f32.powf(air_mass);
    PAR_FULL_SUN * sin_elev * transmittance
}

/// Sun direction unit vector in terrarium coordinates [x, y, z].
///
/// Convention: +x = east, +y = north, +z = up.
/// Azimuth: 0 = north, π/2 = east, π = south, 3π/2 = west.
fn sun_direction(elevation_rad: f32, azimuth_rad: f32) -> [f32; 3] {
    let cos_elev = elevation_rad.cos();
    [
        cos_elev * azimuth_rad.sin(), // east-west
        cos_elev * azimuth_rad.cos(), // north-south
        elevation_rad.sin(),          // vertical
    ]
}

// ---------------------------------------------------------------------------
// Solar state
// ---------------------------------------------------------------------------

/// Complete solar state for a given moment.
#[derive(Debug, Clone, Copy)]
pub struct SolarState {
    /// Solar elevation above horizon (radians).
    pub elevation_rad: f32,
    /// Solar azimuth (radians, 0=north clockwise).
    pub azimuth_rad: f32,
    /// Unit vector pointing toward the sun [x, y, z].
    pub direction: [f32; 3],
    /// Clear-sky PAR at canopy top (µmol/m²/s).
    pub par_incident: f32,
    /// Hours of daylight for this day of year.
    pub day_length_hours: f32,
    /// Whether the sun is above the horizon.
    pub is_daytime: bool,
    /// Day of year (0-365).
    pub day_of_year: f32,
    /// Red PAR component (600-700nm, µmol/m²/s).
    pub par_red: f32,
    /// Far-red component (700-800nm, µmol/m²/s).
    pub par_far_red: f32,
    /// Red:Far-Red ratio at canopy top.
    pub r_fr_ratio: f32,
}

/// Compute solar state from simulation time, latitude, and cloud cover.
///
/// # Arguments
/// * `time_s` - elapsed simulation time in seconds
/// * `latitude_deg` - observer latitude in degrees (positive = north)
/// * `cloud_cover` - fractional cloud cover [0.0, 1.0] attenuating PAR up to 75%
pub fn compute_solar_state(time_s: f32, latitude_deg: f32, cloud_cover: f32) -> SolarState {
    let day_of_year = (time_s / 86400.0) % 365.0;
    let hour_of_day = (time_s % 86400.0) / 3600.0;
    let latitude_rad = latitude_deg.to_radians();

    let decl = solar_declination_rad(day_of_year);
    let hour_angle = (hour_of_day - 12.0) * (PI / 12.0); // 15°/hour
    let elevation = solar_elevation_rad(latitude_rad, decl, hour_angle);
    let azimuth = solar_azimuth_rad(latitude_rad, decl, elevation, hour_angle);
    let par = clear_sky_par(elevation);
    // Clouds attenuate up to 75% of PAR
    let par = par * (1.0 - cloud_cover.clamp(0.0, 1.0) * 0.75);
    let day_len = day_length_hours(latitude_rad, day_of_year);
    let dir = sun_direction(elevation, azimuth);

    // Spectral decomposition: Red (600-700nm) and Far-Red (700-800nm).
    // Direct sunlight: R:FR ≈ 1.2 (55% Red, 45.8% Far-Red of PAR-adjacent band).
    // Diffuse/cloud light: R:FR ≈ 1.0 (50% R, 50% FR).
    // Holmes & Smith (1977, Photochem Photobiol 25:539) measured R:FR 1.15-1.25 in direct sun.
    let direct_fraction = (1.0 - cloud_cover.clamp(0.0, 1.0)).max(0.0);
    let par_red = par * (0.55 * direct_fraction + 0.50 * (1.0 - direct_fraction));
    let par_far_red = par * (0.458 * direct_fraction + 0.50 * (1.0 - direct_fraction));
    let r_fr_ratio = if par_far_red > 1e-6 { par_red / par_far_red } else { 1.2 };

    SolarState {
        elevation_rad: elevation,
        azimuth_rad: azimuth,
        direction: dir,
        par_incident: par,
        day_length_hours: day_len,
        is_daytime: elevation > MIN_SUN_ELEVATION,
        day_of_year,
        par_red,
        par_far_red,
        r_fr_ratio,
    }
}

// ---------------------------------------------------------------------------
// Photon raycasting
// ---------------------------------------------------------------------------

/// Per-plant photon delivery result from raycasting.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhotonFlux {
    /// PAR reaching this plant after canopy attenuation (µmol/m²/s).
    pub par_received: f32,
    /// Fraction of incident PAR transmitted (0 = full shade, 1 = full sun).
    pub shade_fraction: f32,
    /// Light gradient vector for phototropism [x, y, z].
    /// Points toward the direction of strongest illumination.
    pub light_asymmetry: [f32; 3],
    /// Red PAR after canopy attenuation (µmol/m²/s).
    pub par_red: f32,
    /// Far-red PAR after canopy attenuation (µmol/m²/s).
    pub par_far_red: f32,
    /// Red:Far-Red ratio at this plant's position.
    pub r_fr_ratio: f32,
}

/// Trace a single ray from origin toward the sun through canopy geometry.
///
/// Returns the fraction of light transmitted (0.0 to 1.0).
fn trace_ray(
    origin_x_mm: f32,
    origin_y_mm: f32,
    origin_z_mm: f32,
    dir: [f32; 3],
    plants: &[CanopyDescriptor],
    skip_idx: usize,
    voxel_mm: f32,
) -> f32 {
    let [dx, dy, dz] = dir;
    if dz.abs() < 1e-6 {
        return 1.0; // horizontal sun → no vertical traverse
    }

    let mut transmission = 1.0f32;

    for (j, blocker) in plants.iter().enumerate() {
        if j == skip_idx {
            continue;
        }

        let canopy_top = blocker.height_mm;
        if canopy_top <= origin_z_mm {
            continue;
        }

        let canopy_bottom = canopy_top * (1.0 - CROWN_DEPTH_FRACTION);
        let eff_bottom = canopy_bottom.max(origin_z_mm);
        if eff_bottom >= canopy_top {
            continue;
        }

        // Find ray parameter t at canopy bottom and top.
        let t_bottom = (eff_bottom - origin_z_mm) / dz;
        let t_top = (canopy_top - origin_z_mm) / dz;
        let t_enter = t_bottom.min(t_top).max(0.0);
        let t_exit = t_bottom.max(t_top);
        if t_exit <= t_enter {
            continue;
        }

        // Find the ray parameter where horizontal distance to blocker is minimized.
        // This handles low sun angles where the midpoint of the height segment
        // can overshoot the canopy horizontally.
        let bx = blocker.x * voxel_mm;
        let by = blocker.y * voxel_mm;
        let rel_x = origin_x_mm - bx;
        let rel_y = origin_y_mm - by;
        let horiz_len_sq = dx * dx + dy * dy;
        let t_closest = if horiz_len_sq > 1e-10 {
            -(rel_x * dx + rel_y * dy) / horiz_len_sq
        } else {
            (t_enter + t_exit) * 0.5
        };
        let t_check = t_closest.clamp(t_enter, t_exit);
        let hit_x = origin_x_mm + t_check * dx;
        let hit_y = origin_y_mm + t_check * dy;
        let dist_sq = (hit_x - bx) * (hit_x - bx) + (hit_y - by) * (hit_y - by);
        let r = blocker.canopy_radius_mm;
        if dist_sq > r * r {
            continue;
        }

        // Radial falloff: full LAI at center, linear decrease to edge.
        let dist = dist_sq.sqrt();
        let radial_factor = (1.0 - dist / r.max(0.01)).clamp(0.0, 1.0);

        // Angular correction: oblique rays traverse more leaf area.
        // cos(zenith) = dz for normalized sun direction.
        let cos_zenith = dz.abs().max(0.05);
        let angular_factor = (1.0 / cos_zenith).min(5.0);

        // Beer-Lambert: transmission *= exp(−k × LAI × radial × angular)
        let effective_lai = blocker.lai * radial_factor * angular_factor;
        transmission *= (-blocker.extinction_coeff * effective_lai).exp();
    }

    transmission.clamp(0.0, 1.0)
}

/// Compute per-plant photon flux by raycasting through the canopy.
///
/// For each plant, traces rays from its crown toward the sun. Taller plants'
/// canopies attenuate the light via Beer-Lambert. Also computes a light
/// asymmetry vector by tracing offset rays for phototropism.
///
/// # Arguments
/// * `plants` - canopy descriptors for all plants
/// * `solar` - current solar state (position, PAR, direction)
/// * `cell_size_mm` - grid cell size in mm
///
/// # Returns
/// Per-plant `PhotonFlux` with PAR, shade fraction, and light asymmetry.
pub fn raycast_canopy_photons(
    plants: &[CanopyDescriptor],
    solar: &SolarState,
    cell_size_mm: f32,
) -> Vec<PhotonFlux> {
    let n = plants.len();
    let voxel = cell_size_mm.max(0.01);

    if n == 0 || !solar.is_daytime {
        return vec![PhotonFlux::default(); n];
    }

    let dir = solar.direction;
    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let me = &plants[i];
        let crown_x = me.x * voxel;
        let crown_y = me.y * voxel;
        let crown_z = me.height_mm;

        // Primary ray from crown center.
        let primary = trace_ray(crown_x, crown_y, crown_z, dir, plants, i, voxel);

        // Asymmetry: 4 offset rays for phototropism gradient.
        let offset = me.canopy_radius_mm.max(0.5) * 0.5;
        let east = trace_ray(crown_x + offset, crown_y, crown_z, dir, plants, i, voxel);
        let west = trace_ray(crown_x - offset, crown_y, crown_z, dir, plants, i, voxel);
        let north = trace_ray(crown_x, crown_y + offset, crown_z, dir, plants, i, voxel);
        let south = trace_ray(crown_x, crown_y - offset, crown_z, dir, plants, i, voxel);

        let grad_x = east - west;
        let grad_y = north - south;

        // Spectral splitting through canopy: Red absorbed more strongly, Far-Red transmitted.
        // Chlorophyll absorbs heavily in 600-700nm (red) but transmits 700-800nm (far-red).
        // For a total transmission T, the spectral transmissions are:
        //   T_red = T^1.3   (red attenuated more by chlorophyll absorption)
        //   T_fr  = T^0.4   (far-red passes through leaves more easily)
        // (Smith, 1982, Ann Rev Plant Physiol 33:481; Ballaré, 1999, Trends Plant Sci)
        let t_red = primary.powf(1.3);
        let t_fr = primary.powf(0.4);
        let plant_par_red = solar.par_red * t_red;
        let plant_par_fr = solar.par_far_red * t_fr;
        let plant_rfr = if plant_par_fr > 1e-6 {
            plant_par_red / plant_par_fr
        } else {
            solar.r_fr_ratio
        };

        results.push(PhotonFlux {
            par_received: solar.par_incident * primary,
            shade_fraction: primary,
            light_asymmetry: [grad_x, 0.0, grad_y],
            par_red: plant_par_red,
            par_far_red: plant_par_fr,
            r_fr_ratio: plant_rfr,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solar_declination_summer_solstice() {
        // Day ~172 (June 21): declination ≈ +23.44°
        let decl = solar_declination_rad(172.0);
        let decl_deg = decl.to_degrees();
        assert!(
            (decl_deg - 23.44).abs() < 1.0,
            "summer solstice declination should be ~+23.44°, got {:.2}°",
            decl_deg,
        );
    }

    #[test]
    fn test_solar_declination_winter_solstice() {
        // Day ~355 (Dec 21): declination ≈ -23.44°
        let decl = solar_declination_rad(355.0);
        let decl_deg = decl.to_degrees();
        assert!(
            (decl_deg + 23.44).abs() < 1.0,
            "winter solstice declination should be ~-23.44°, got {:.2}°",
            decl_deg,
        );
    }

    #[test]
    fn test_solar_elevation_noon_equator_equinox() {
        // Equator (0°), equinox (day 80), solar noon: elevation ≈ 90°
        let decl = solar_declination_rad(80.0);
        let elev = solar_elevation_rad(0.0, decl, 0.0);
        let elev_deg = elev.to_degrees();
        assert!(
            elev_deg > 80.0,
            "equator equinox noon should have elevation ~90°, got {:.1}°",
            elev_deg,
        );
    }

    #[test]
    fn test_solar_elevation_night() {
        // 42°N, midnight (hour_angle = π): sun should be below horizon
        let decl = solar_declination_rad(80.0);
        let lat = 42.0f32.to_radians();
        let elev = solar_elevation_rad(lat, decl, PI);
        assert!(
            elev < 0.0,
            "midnight sun should be below horizon, got {:.2} rad",
            elev,
        );
    }

    #[test]
    fn test_day_length_equinox() {
        let lat = 42.0f32.to_radians();
        let dlen = day_length_hours(lat, 80.0);
        assert!(
            (dlen - 12.0).abs() < 1.0,
            "equinox day length at 42°N should be ~12h, got {:.1}h",
            dlen,
        );
    }

    #[test]
    fn test_day_length_summer_longer() {
        let lat = 42.0f32.to_radians();
        let summer = day_length_hours(lat, 172.0);
        let winter = day_length_hours(lat, 355.0);
        assert!(
            summer > winter + 4.0,
            "summer ({:.1}h) should be much longer than winter ({:.1}h)",
            summer,
            winter,
        );
    }

    #[test]
    fn test_clear_sky_par_noon_vs_dawn() {
        let noon_par = clear_sky_par(60.0f32.to_radians());
        let dawn_par = clear_sky_par(10.0f32.to_radians());
        assert!(
            noon_par > dawn_par * 3.0,
            "noon PAR ({:.0}) should be much higher than dawn PAR ({:.0})",
            noon_par,
            dawn_par,
        );
        assert!(noon_par > 500.0, "noon PAR should be substantial: {:.0}", noon_par);
    }

    #[test]
    fn test_clear_sky_par_below_horizon() {
        let par = clear_sky_par(-0.1);
        assert!(par == 0.0, "below-horizon PAR should be 0, got {}", par);
    }

    #[test]
    fn test_compute_solar_state_midday() {
        let time_s = 172.0 * 86400.0 + 43200.0; // day 172, noon
        let state = compute_solar_state(time_s, 42.0, 0.0);
        assert!(state.is_daytime, "noon should be daytime");
        assert!(state.par_incident > 500.0, "noon PAR should be high: {:.0}", state.par_incident);
        assert!(state.elevation_rad > 0.5, "noon elevation should be high: {:.2}", state.elevation_rad);
        assert!(
            state.day_length_hours > 14.0,
            "summer day should be >14h: {:.1}",
            state.day_length_hours,
        );
    }

    #[test]
    fn test_compute_solar_state_midnight() {
        let time_s = 172.0 * 86400.0; // midnight of day 172
        let state = compute_solar_state(time_s, 42.0, 0.0);
        assert!(!state.is_daytime, "midnight should not be daytime");
        assert!(state.par_incident == 0.0, "midnight PAR should be 0");
    }

    #[test]
    fn test_raycast_single_plant_full_sun() {
        let plants = vec![CanopyDescriptor {
            x: 10.0,
            y: 10.0,
            height_mm: 15.0,
            canopy_radius_mm: 5.0,
            lai: 3.0,
            extinction_coeff: 0.5,
        }];
        let solar = SolarState {
            elevation_rad: 1.0,
            azimuth_rad: PI,
            direction: sun_direction(1.0, PI),
            par_incident: 1500.0,
            day_length_hours: 14.0,
            is_daytime: true,
            day_of_year: 172.0,
            par_red: 1500.0 * 0.55,
            par_far_red: 1500.0 * 0.458,
            r_fr_ratio: 1.2,
        };
        let flux = raycast_canopy_photons(&plants, &solar, 0.5);
        assert_eq!(flux.len(), 1);
        assert!(
            (flux[0].shade_fraction - 1.0).abs() < 0.001,
            "single plant should get full sun: {:.3}",
            flux[0].shade_fraction,
        );
        assert!(
            (flux[0].par_received - 1500.0).abs() < 1.0,
            "single plant PAR should match incident: {:.0}",
            flux[0].par_received,
        );
    }

    #[test]
    fn test_raycast_tall_shades_short() {
        let plants = vec![
            CanopyDescriptor {
                x: 10.0,
                y: 10.0,
                height_mm: 18.0,
                canopy_radius_mm: 6.0,
                lai: 4.0,
                extinction_coeff: 0.5,
            },
            CanopyDescriptor {
                x: 10.0,
                y: 10.0,
                height_mm: 8.0,
                canopy_radius_mm: 4.0,
                lai: 2.0,
                extinction_coeff: 0.5,
            },
        ];
        let solar = SolarState {
            elevation_rad: 1.2,
            azimuth_rad: PI,
            direction: sun_direction(1.2, PI),
            par_incident: 1800.0,
            day_length_hours: 14.0,
            is_daytime: true,
            day_of_year: 172.0,
            par_red: 1800.0 * 0.55,
            par_far_red: 1800.0 * 0.458,
            r_fr_ratio: 1.2,
        };
        let flux = raycast_canopy_photons(&plants, &solar, 0.5);
        assert_eq!(flux.len(), 2);
        assert!(
            flux[0].shade_fraction > 0.95,
            "tall plant should get near-full sun: {:.3}",
            flux[0].shade_fraction,
        );
        assert!(
            flux[1].shade_fraction < flux[0].shade_fraction,
            "short plant ({:.3}) should be more shaded than tall ({:.3})",
            flux[1].shade_fraction,
            flux[0].shade_fraction,
        );
    }

    #[test]
    fn test_raycast_low_sun_casts_longer_shadows() {
        // Tall plant to the south, short plant far to the north.
        // At high sun the shadow barely reaches the short plant; at low sun
        // the shadow extends further north and shades it more.
        let tall = CanopyDescriptor {
            x: 10.0,
            y: 8.0,  // south
            height_mm: 16.0,
            canopy_radius_mm: 5.0,
            lai: 4.0,
            extinction_coeff: 0.5,
        };
        let short = CanopyDescriptor {
            x: 10.0,
            y: 30.0, // far north — outside shadow at high sun, inside at low
            height_mm: 6.0,
            canopy_radius_mm: 3.0,
            lai: 2.0,
            extinction_coeff: 0.5,
        };
        let plants = vec![tall, short];

        // High sun: nearly overhead, shadow is short
        let solar_high = SolarState {
            elevation_rad: 1.2,
            azimuth_rad: PI,
            direction: sun_direction(1.2, PI),
            par_incident: 1800.0,
            day_length_hours: 14.0,
            is_daytime: true,
            day_of_year: 172.0,
            par_red: 1800.0 * 0.55,
            par_far_red: 1800.0 * 0.458,
            r_fr_ratio: 1.2,
        };
        // Low sun from south: shadow extends far northward
        let solar_low = SolarState {
            elevation_rad: 0.3,
            azimuth_rad: PI,
            direction: sun_direction(0.3, PI),
            par_incident: 500.0,
            day_length_hours: 14.0,
            is_daytime: true,
            day_of_year: 172.0,
            par_red: 500.0 * 0.55,
            par_far_red: 500.0 * 0.458,
            r_fr_ratio: 1.2,
        };

        let flux_high = raycast_canopy_photons(&plants, &solar_high, 0.5);
        let flux_low = raycast_canopy_photons(&plants, &solar_low, 0.5);

        // At low sun, shadow extends northward and should shade the distant short plant
        // more (or equally) compared to the high-sun case.
        assert!(
            flux_low[1].shade_fraction <= flux_high[1].shade_fraction + 0.01,
            "low sun should shade distant plant more: low={:.3}, high={:.3}",
            flux_low[1].shade_fraction,
            flux_high[1].shade_fraction,
        );
    }

    #[test]
    fn test_raycast_nighttime_zero_flux() {
        let plants = vec![CanopyDescriptor {
            x: 10.0,
            y: 10.0,
            height_mm: 15.0,
            canopy_radius_mm: 5.0,
            lai: 3.0,
            extinction_coeff: 0.5,
        }];
        let solar = SolarState {
            elevation_rad: -0.2,
            azimuth_rad: 0.0,
            direction: [0.0, 0.0, -1.0],
            par_incident: 0.0,
            day_length_hours: 14.0,
            is_daytime: false,
            day_of_year: 172.0,
            par_red: 0.0,
            par_far_red: 0.0,
            r_fr_ratio: 1.2,
        };
        let flux = raycast_canopy_photons(&plants, &solar, 0.5);
        assert_eq!(flux[0].par_received, 0.0, "nighttime PAR should be 0");
    }

    #[test]
    fn test_distant_plants_no_shading() {
        let plants = vec![
            CanopyDescriptor {
                x: 0.0,
                y: 0.0,
                height_mm: 18.0,
                canopy_radius_mm: 5.0,
                lai: 5.0,
                extinction_coeff: 0.5,
            },
            CanopyDescriptor {
                x: 40.0,
                y: 40.0,
                height_mm: 8.0,
                canopy_radius_mm: 4.0,
                lai: 2.0,
                extinction_coeff: 0.5,
            },
        ];
        let solar = SolarState {
            elevation_rad: 1.0,
            azimuth_rad: PI,
            direction: sun_direction(1.0, PI),
            par_incident: 1500.0,
            day_length_hours: 14.0,
            is_daytime: true,
            day_of_year: 172.0,
            par_red: 1500.0 * 0.55,
            par_far_red: 1500.0 * 0.458,
            r_fr_ratio: 1.2,
        };
        let flux = raycast_canopy_photons(&plants, &solar, 0.5);
        assert!(
            flux[1].shade_fraction > 0.99,
            "distant plant should not be shaded: {:.3}",
            flux[1].shade_fraction,
        );
    }

    #[test]
    fn test_light_asymmetry_shaded_from_east() {
        // Tall plant to the east shades short plant's east side.
        let plants = vec![
            CanopyDescriptor {
                x: 12.0,
                y: 10.0,
                height_mm: 18.0,
                canopy_radius_mm: 5.0,
                lai: 5.0,
                extinction_coeff: 0.5,
            },
            CanopyDescriptor {
                x: 10.0,
                y: 10.0,
                height_mm: 6.0,
                canopy_radius_mm: 3.0,
                lai: 2.0,
                extinction_coeff: 0.5,
            },
        ];
        let solar = SolarState {
            elevation_rad: 1.0,
            azimuth_rad: PI,
            direction: sun_direction(1.0, PI),
            par_incident: 1500.0,
            day_length_hours: 14.0,
            is_daytime: true,
            day_of_year: 172.0,
            par_red: 1500.0 * 0.55,
            par_far_red: 1500.0 * 0.458,
            r_fr_ratio: 1.2,
        };
        let flux = raycast_canopy_photons(&plants, &solar, 0.5);
        // East side more shaded → asymmetry[0] (east - west) should be ≤ 0.
        assert!(
            flux[1].light_asymmetry[0] <= 0.0,
            "shaded from east should have non-positive x asymmetry: {:.4}",
            flux[1].light_asymmetry[0],
        );
    }

    #[test]
    fn test_day_length_seasonal_variation() {
        let lat = 42.0f32.to_radians();
        let spring = day_length_hours(lat, 80.0);
        let summer = day_length_hours(lat, 172.0);
        let autumn = day_length_hours(lat, 265.0);
        let winter = day_length_hours(lat, 355.0);

        assert!(summer > spring, "summer > spring");
        assert!(spring > winter, "spring > winter");
        assert!((spring - autumn).abs() < 1.0, "equinoxes similar");
        assert!(summer > 14.0, "summer >14h at 42°N: {:.1}", summer);
        assert!(winter < 10.0, "winter <10h at 42°N: {:.1}", winter);
    }

    #[test]
    fn test_par_reduced_by_clouds() {
        let time_s = 172.0 * 86400.0 + 43200.0; // day 172, noon
        let clear = compute_solar_state(time_s, 42.0, 0.0);
        let cloudy = compute_solar_state(time_s, 42.0, 0.8);
        assert!(
            cloudy.par_incident < clear.par_incident,
            "cloudy PAR ({:.0}) should be less than clear PAR ({:.0})",
            cloudy.par_incident,
            clear.par_incident,
        );
        assert!(
            cloudy.par_incident > 0.0,
            "even with 80% cloud cover there should be some PAR",
        );
    }

    #[test]
    fn test_full_clouds_reduce_par_75_percent() {
        let time_s = 172.0 * 86400.0 + 43200.0; // day 172, noon
        let clear = compute_solar_state(time_s, 42.0, 0.0);
        let full_cloud = compute_solar_state(time_s, 42.0, 1.0);
        // Full cloud cover should reduce PAR by ~75%
        let ratio = full_cloud.par_incident / clear.par_incident;
        assert!(
            (ratio - 0.25).abs() < 0.01,
            "full cloud cover should leave ~25% PAR, got ratio={:.4}",
            ratio,
        );
    }

    // -- Spectral R:FR tests (Phase 4) --

    #[test]
    fn test_solar_state_includes_spectral() {
        let time_s = 172.0 * 86400.0 + 43200.0;
        let state = compute_solar_state(time_s, 42.0, 0.0);
        assert!(state.par_red > 0.0, "Should have red PAR");
        assert!(state.par_far_red > 0.0, "Should have far-red PAR");
        assert!(
            state.r_fr_ratio > 1.0,
            "Direct sun R:FR should be > 1.0: {}",
            state.r_fr_ratio,
        );
    }

    #[test]
    fn test_direct_sun_rfr_about_1_2() {
        let time_s = 172.0 * 86400.0 + 43200.0;
        let state = compute_solar_state(time_s, 42.0, 0.0);
        assert!(
            (state.r_fr_ratio - 1.2).abs() < 0.1,
            "Direct sun R:FR ≈ 1.2, got {}",
            state.r_fr_ratio,
        );
    }

    #[test]
    fn test_cloud_diffuse_rfr_about_1_0() {
        let time_s = 172.0 * 86400.0 + 43200.0;
        let state = compute_solar_state(time_s, 42.0, 1.0);
        assert!(
            (state.r_fr_ratio - 1.0).abs() < 0.1,
            "Full cloud R:FR ≈ 1.0, got {}",
            state.r_fr_ratio,
        );
    }

    #[test]
    fn test_canopy_shade_lowers_rfr() {
        // Two plants at same position, tall shades short
        let plants = vec![
            CanopyDescriptor {
                x: 10.0,
                y: 10.0,
                height_mm: 18.0,
                canopy_radius_mm: 6.0,
                lai: 4.0,
                extinction_coeff: 0.5,
            },
            CanopyDescriptor {
                x: 10.0,
                y: 10.0,
                height_mm: 8.0,
                canopy_radius_mm: 4.0,
                lai: 2.0,
                extinction_coeff: 0.5,
            },
        ];
        let solar = SolarState {
            elevation_rad: 1.2,
            azimuth_rad: PI,
            direction: sun_direction(1.2, PI),
            par_incident: 1800.0,
            day_length_hours: 14.0,
            is_daytime: true,
            day_of_year: 172.0,
            par_red: 1800.0 * 0.55,
            par_far_red: 1800.0 * 0.458,
            r_fr_ratio: 1.2,
        };
        let flux = raycast_canopy_photons(&plants, &solar, 0.5);
        assert!(
            flux[1].r_fr_ratio < flux[0].r_fr_ratio,
            "Shaded plant should have lower R:FR: shaded={}, unshaded={}",
            flux[1].r_fr_ratio,
            flux[0].r_fr_ratio,
        );
    }
}
