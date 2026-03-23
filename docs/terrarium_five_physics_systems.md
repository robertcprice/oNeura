# oNeura Terrarium: Five Physics Systems

Technical reference for the five coupled physical simulation systems that govern the
oNeura terrarium's abiotic environment. Each system is grounded in published biophysical
equations and uses Hill-kinetic or Michaelis-Menten rate functions as the universal
nonlinear response primitive.

**Last updated**: 2026-03-21
**Test suite**: 1099 tests (1086 unit + 10 integration + 3 doc), 0 failures

---

## Table of Contents

1. [Wind and Turbulence (Phase 1)](#1-wind-and-turbulence-phase-1)
2. [Soil Depth Profile (Phase 2)](#2-soil-depth-profile-phase-2)
3. [Weather State Machine (Phase 3)](#3-weather-state-machine-phase-3)
4. [Spectral Light and R:FR (Phase 4)](#4-spectral-light-and-rfr-phase-4)
5. [VOC Signaling and Chemical Ecology (Phase 5)](#5-voc-signaling-and-chemical-ecology-phase-5)
6. [System Interconnections](#system-interconnections)
7. [Aquarium vs Terrarium Differences](#aquarium-vs-terrarium-differences)
8. [Simulation Step Order](#simulation-step-order)
9. [Common Kinetic Primitives](#common-kinetic-primitives)
10. [Test Summary](#test-summary)

---

## Common Kinetic Primitives

Before describing the five systems, it is important to understand the two rate functions
used throughout. Both are defined in `core/src/botany/physiology_bridge.rs`.

### Hill function

```
hill(C, Km, n) = C^n / (Km^n + C^n)
```

Returns a value in `[0, 1]`. At `C = Km`, the output is exactly `0.5`. The exponent `n`
controls the steepness of the switch: `n = 1` gives the hyperbolic Michaelis-Menten
curve, while `n >= 3` produces a steep sigmoidal threshold.

### Hill repression

```
hill_repression(C, Km, n) = 1 - hill(C, Km, n)
```

Inverts the Hill function: output is `1.0` at `C = 0` and falls toward `0.0` as `C`
rises past `Km`.

### Michaelis-Menten

```
michaelis_menten(S, Km) = hill(S, Km, 1)
```

Special case of Hill with `n = 1`. Used where cooperative binding is not expected.

---

## 1. Wind and Turbulence (Phase 1)

**Source files**:
- `core/src/terrarium/biomechanics.rs` -- drag, bending stress, damage accumulation
- `core/src/terrarium/atmosphere.rs` -- wind field advection, 3-octave turbulence

### 1.1 Turbulent Wind Field

The wind field is a 3D grid (`wind_x`, `wind_y`, `wind_z`) updated each timestep with
three-octave sinusoidal noise that produces deterministic, spatially coherent turbulence.

```
fluctuation = (octave1 + octave2 + octave3) / 1.75

octave1 = sin(t * 0.5  + phase * 6.28)
octave2 = sin(t * 1.3  + phase * 12.56 + 1.7) * 0.5
octave3 = sin(t * 3.7  + phase * 25.12 + 3.1) * 0.25
```

- `phase` is a deterministic spatial hash: `sin(x * 0.73 + y * 1.17 + z * 2.31)`
- A logarithmic height profile scales wind speed by elevation:
  `height_factor = 0.5 + 0.5 * (z + 1) / depth`
- Local wind speed: `base_speed * height_factor * (1 + turbulence_intensity * fluctuation)`
- Wind direction drifts slowly: `dir_angle = t * 0.02 + phase * 0.5`

The wind vector at each cell relaxes toward the target via exponential approach:

```
wind_x[i] += (target_x - wind_x[i]) * dt.min(0.5)
```

### 1.2 Aerodynamic Drag Force

Standard aerodynamic drag applied to each plant's projected canopy cross-section.

```
F = 0.5 * rho * Cd * A * v^2
```

| Symbol | Meaning | Value / Source |
|--------|---------|----------------|
| `rho` | Air density | 1.2 kg/m^3 (sea-level, ~22 C) |
| `Cd` | Drag coefficient | Species-specific via `BotanicalSpeciesProfile.drag_coefficient` |
| `A` | Projected area | `canopy_radius * height` (mm^2, converted to m^2) |
| `v` | Wind speed | Sampled at canopy height from wind grid (mm/s, converted to m/s) |

### 1.3 Stem Bending Stress (Cantilever Beam Model)

The stem is modeled as a cantilever beam with circular cross-section. Wind force is
applied at the canopy center of mass (half height).

```
sigma = M / S

M = F * h / 2           (bending moment, N*m)
S = pi * d^3 / 32       (section modulus, m^3)
```

| Symbol | Meaning |
|--------|---------|
| `F` | Drag force from Section 1.2 (Newtons) |
| `h` | Plant height (m) |
| `d` | Stem diameter = `height * stem_diameter_fraction` (m) |

### 1.4 Yield Strength (Niklas 1992 Allometry)

Stem yield strength is estimated from wood density using an empirical allometric
relationship from *Plant Biomechanics* (Niklas, 1992):

```
sigma_yield = 0.0005 * rho_wood^1.5
```

Herbaceous stems (density ~150-200 kg/m^3) are weak; hardwood (500-700 kg/m^3) is
substantially stronger.

### 1.5 Elastic Deflection Angle

```
theta = F * L^2 / (3 * E * I)

E = 0.01 * rho_wood^2        (Young's modulus approximation, Pa)
I = pi * d^4 / 64             (second moment of area, m^4)
```

Deflection is clamped to `[0, pi/2]` (90 degrees maximum).

### 1.6 Damage Accumulation (Hill Kinetics)

Damage accumulates continuously based on the stress-to-yield ratio, with no hardcoded
thresholds. A recovery term allows healing when stress is low.

```
stress_ratio   = sigma / sigma_yield
damage_rate    = hill(stress_ratio, 0.7, 4)      # steep onset above 70% of yield
recovery_rate  = hill_repression(stress_ratio, 0.3, 2)

damage += (damage_rate * 0.1  -  recovery_rate * 0.02 * damage) * dt
damage = clamp(damage, 0, 1)
```

The resulting `mechanical_damage` in `[0, 1]` feeds directly into plant viability via
another Hill repression:

```
viability_modifier = hill_repression(damage, 0.5, 3)
```

### 1.7 Pollen Wind Boost

Wind speed and direction alignment multiply pollen dispersal distance. Downwind
direction is determined by the cosine of the angle between the wind vector and the
donor-receiver displacement:

```
cos_angle = dot(displacement, wind) / (|displacement| * |wind|)
wind_boost = max(cos_angle, 0) * wind_speed * 2.0
```

Upwind donors receive zero boost; downwind donors receive a range extension proportional
to wind speed.

### 1.8 Tests (12 total)

- `wind_drag_force_zero_speed` -- F = 0 when v = 0
- `wind_drag_force_quadratic` -- F scales with v^2
- `stem_bending_stress_thin_stem` -- thin stems have higher stress
- `stem_yield_strength_wood_density` -- hardwood >> softwood
- `wind_deflection_small_force` -- small force gives small deflection
- `damage_accumulation_below_yield` -- low stress => near-zero damage rate
- `damage_accumulation_above_yield` -- above yield => high damage rate
- `damage_recovery_low_stress` -- low stress allows recovery
- `damage_does_not_exceed_one` -- 1000 steps of extreme wind stays <= 1.0
- `mechanical_viability_modifier_healthy` -- zero damage => viability 1.0
- `mechanical_viability_modifier_damaged` -- 0.8 damage => low viability
- `pollen_wind_boost_concept` -- downwind gets boost, upwind gets zero

---

## 2. Soil Depth Profile (Phase 2)

**Source file**: `core/src/terrarium/soil_profile.rs`

### 2.1 Four-Layer Vertical Structure

The soil column is divided into four layers with fixed depth boundaries.

| Layer | Bottom Depth (mm) | Thickness (mm) | Typical Role |
|-------|-------------------|-----------------|-------------|
| 0 | 2 | 2 | Surface litter / evaporation zone |
| 1 | 10 | 8 | Shallow root zone |
| 2 | 30 | 20 | Main root zone |
| 3 | 100 | 70 | Deep moisture reservoir |

Each cell in the 2D grid maintains its own four-layer moisture array `[f32; 4]`.

### 2.2 Vertical Transport: Percolation and Capillary Rise

Soil texture is encoded as a continuous value: `0 = sand`, `1 = clay`.

**Percolation** (gravity-driven, downward):

```
percolation_coeff = 0.05 * (1 + (1 - texture) * 2)     # sand: 0.15, clay: 0.05
percolation_rate  = hill(upper_moisture, 0.4, 2) * percolation_coeff
percolation_flux  = min(percolation_rate * dt, upper_moisture * 0.5)
```

**Capillary rise** (deficit-driven, upward):

```
capillary_coeff = 0.02 * (1 + texture * 3)               # sand: 0.02, clay: 0.08
capillary_rate  = michaelis_menten(lower_moisture, 0.3) * capillary_coeff * (1 - upper_moisture)
capillary_flux  = min(capillary_rate * dt, lower_moisture * 0.3 * thickness_ratio)
```

Net flux between adjacent layers: `net = percolation - capillary`. Transferred volume
is scaled by the thickness ratio of the two layers. Moisture is clamped to `[0, 1]`.

### 2.3 Rainfall Infiltration

Precipitation enters the top layer proportional to available pore space:

```
available_pore = (1 - layers[0])
moisture_add   = precip_rate_mm_h * dt / 3600 / layer_0_depth
infiltrated    = michaelis_menten(available_pore, 0.3) * moisture_add
```

### 2.4 Surface Evaporation

Surface moisture loss depends on temperature, humidity deficit, and current moisture:

```
humidity_deficit = (1 - humidity)
temp_factor     = hill(temperature / 40, 0.5, 1.5)
evap_rate       = hill(layers[0], 0.2, 2) * temp_factor * humidity_deficit * 0.005
evap            = min(evap_rate * dt, layers[0] * 0.3)
```

### 2.5 Root-Weighted Moisture Access

Plants access moisture based on root depth. An exponential decay kernel weights deeper
layers more heavily for deep-rooted species.

```
char_depth = 2 + root_depth_bias * 98       # mm: [2, 100]
weight_i   = exp(-midpoint_i / char_depth) * thickness_i

root_moisture = sum(layers[i] * weight_i) / sum(weight_i)
```

`root_depth_bias` ranges from `0` (shallow, surface-only) to `1` (deep, all layers).

### 2.6 Legacy Field Synchronization

After each step, the four-layer state syncs to legacy scalar fields:

```
surface_moisture = layers[0]
deep_moisture    = (layers[1] + layers[2]) / 2
```

### 2.7 Tests (10 total)

- `layer_thickness_sums_to_total` -- thicknesses sum to 100 mm
- `percolation_wet_to_dry` -- wet surface loses water downward
- `capillary_rise_dry_to_wet` -- dry surface gains water from below (clay texture)
- `infiltration_fills_top_layer` -- rainfall increases top-layer moisture
- `evaporation_dries_surface` -- warm air dries surface
- `root_shallow_bias_reads_surface` -- shallow roots see mostly layer 0
- `root_deep_bias_reads_all` -- deep roots access deep moisture
- `sandy_soil_percolates_faster` -- sand drains faster than clay
- `clay_soil_holds_capillary` -- clay has stronger capillary rise
- `backward_compat_legacy_fields` -- root moisture stays in `[0, 1]`

---

## 3. Emergent Weather (Phase 3)

**Source file**: `core/src/terrarium/biomechanics.rs`

Weather is **fully emergent** from the simulation's actual physical state fields. There
is no state machine, no Markov chain, no random transitions. Cloud cover, precipitation,
and temperature offset are continuous functions of humidity, temperature, soil moisture,
and wind speed — derived every frame from the live simulation grid.

### 3.1 Weather State

```rust
pub struct WeatherState {
    pub cloud_cover: f32,              // [0, 1] — emergent from humidity
    pub precipitation_rate_mm_h: f32,  // mm/h — emergent from cloud + humidity
    pub temperature_offset_c: f32,     // deg C — emergent from cloud albedo + evaporation
    pub regime: WeatherRegime,         // DIAGNOSTIC LABEL ONLY — derived, not a driver
    pub regime_duration_s: f32,
}
```

`WeatherRegime` (Clear/PartlyCloudy/Overcast/Rain/Storm) is classified from the
continuous state via `classify_weather_regime()` for display and checkpoint readability.
It has **zero effect** on the simulation.

### 3.2 Cloud Cover (Clausius-Clapeyron + Hill Kinetics)

Cloud formation emerges from atmospheric humidity approaching saturation. The physics:

- Clausius-Clapeyron: warmer air holds more moisture before saturating
- Soil evaporation feeds atmospheric humidity (wet soil → more moisture source)
- Convective instability (spatial temperature variance) drives cloud formation

```
sat_humidity_norm   = mean_temp / 40                          # rough saturation proxy
evap_contribution   = hill(mean_soil_moisture, 0.4, 1.5) * 0.15
effective_humidity   = clamp(mean_humidity + evap_contribution, 0, 1)

saturation_ease     = hill_repression(sat_humidity_norm, 0.5, 2)  # cooler = easier
cloud_tendency      = hill(effective_humidity, 0.35 + saturation_ease * 0.2, 3)
convective_lift     = hill(temp_variance, 4, 2) * 0.2
winter_cloud_bias   = (1 - season_summer) * 0.1

target_cloud_cover  = clamp(cloud_tendency + convective_lift + winter_bias, 0, 1)
```

### 3.3 Precipitation (Cloud + Humidity Surplus)

Rain requires BOTH high cloud cover AND sufficient humidity surplus — a multiplicative
Hill gate (same AND-gate pattern as VOC defense priming):

```
precip_readiness   = hill(target_cloud, 0.6, 4) * hill(effective_humidity, 0.5, 3)
wind_precip_boost  = hill(mean_wind_speed, 0.5, 2) * 0.3    # orographic proxy
humidity_surplus    = max(effective_humidity - 0.5, 0) * 2    # [0, 1]

target_precip      = precip_readiness * (1 + wind_boost) * humidity_surplus * 20
```

Maximum precipitation intensity is ~20 mm/h under extreme conditions (full cloud cover,
high humidity surplus, strong wind).

### 3.4 Temperature Offset (Emergent Thermodynamics)

Temperature offset emerges from four physical effects:

```
cloud_cooling       = -target_cloud * 5.0           # albedo: up to -5°C at full overcast
evaporative_cooling = -hill(soil_moisture, 0.3, 2) * 1.5  # latent heat of evaporation
precip_warming      = hill(precip/20, 0.3, 2) * 0.5       # latent heat release
summer_warming      = season_summer * 1.5                   # seasonal solar heating

target_temp_offset  = cloud_cooling + evap_cooling + precip_warming + summer_warming
```

### 3.5 Negative Feedback: Precipitation Removes Humidity

Rain creates a self-limiting negative feedback loop — precipitation removes atmospheric
moisture, which eventually reduces cloud cover, which stops precipitation:

```
if precipitation > 0.1 mm/h:
    humidity -= precipitation * dt / 3600 * 0.001    # per cell, clamped to 0.05 min
```

This produces natural weather cycling without any external forcing or state machine.

### 3.6 Exponential Relaxation

All three weather outputs relax smoothly toward their emergent targets:

```
relax_factor = exp(-0.3 * dt)
cloud_cover  = target + (current - target) * relax_factor
```

### 3.7 Diagnostic Regime Classification

`classify_weather_regime()` reads the continuous state and names it:

| Condition | Label |
|-----------|-------|
| precip > 8 mm/h, or (precip > 3 and wind > 1) | Storm |
| precip > 0.5 mm/h | Rain |
| cloud > 0.65 | Overcast |
| cloud > 0.25 | PartlyCloudy |
| otherwise | Clear |

### 3.8 Downstream Effects

- **Cloud cover** attenuates PAR by up to 75%: `PAR *= (1 - cloud_cover * 0.75)`
- **Precipitation** drives rainfall infiltration into the Phase 2 soil profile
- **Temperature offset** modifies metabolic rates (respiration Q10, VOC emission)

### 3.9 Tests (12 total)

- `weather_default_clear` -- default state has low cloud cover
- `classify_low_cloud_clear` -- low cloud → Clear label
- `classify_moderate_cloud_partly` -- moderate cloud → PartlyCloudy label
- `classify_high_cloud_overcast` -- high cloud → Overcast label
- `classify_precipitation_rain` -- moderate precip → Rain label
- `classify_heavy_precip_storm` -- heavy precip + wind → Storm label
- `high_humidity_produces_clouds` -- Hill on humidity yields more cloud tendency
- `cloud_cover_causes_cooling` -- overcast cooler than clear
- `precipitation_requires_clouds_and_humidity` -- AND-gate: both needed for rain
- `exponential_relaxation_converges` -- smooth convergence to target
- `soil_evaporation_feeds_clouds` -- wet soil → more atmospheric humidity → clouds
- `precipitation_removes_humidity` -- negative feedback: rain dries the air

---

## 4. Spectral Light and R:FR (Phase 4)

**Source files**:
- `core/src/terrarium/solar.rs` -- solar astronomy, raycasting, spectral decomposition
- `core/src/botany/genome.rs` -- PHYB and SAS gene circuits
- `core/src/botany/physiology_bridge.rs` -- shade avoidance growth modifiers

### 4.1 Solar Astronomy

Solar position is computed from simulation time and observer latitude using a simplified
Meeus algorithm.

**Declination** (axial tilt projection):

```
delta = -23.44 deg * cos(2 * pi * (day_of_year + 10) / 365)
```

**Elevation** (angle above horizon):

```
sin(alpha) = sin(phi) * sin(delta) + cos(phi) * cos(delta) * cos(h)
```

where `phi` is latitude (radians) and `h` is the hour angle (`(hour - 12) * 15 deg/hr`).

**Day length**:

```
cos(omega_0) = -tan(phi) * tan(delta)
day_length   = 2 * acos(omega_0) * 12 / pi       (hours)
```

### 4.2 Clear-Sky PAR (Beer-Lambert Atmospheric Attenuation)

```
AM  = 1 / sin(alpha)           (air mass, capped at 40)
tau = 0.76                      (clear-sky transmittance)
PAR = PAR_FULL_SUN * sin(alpha) * tau^AM
```

`PAR_FULL_SUN = 2000 umol/m^2/s` (peak clear-sky PAR at solar noon, sea level).

Cloud attenuation:

```
PAR *= (1 - cloud_cover * 0.75)
```

### 4.3 Spectral R:FR Decomposition

The incident PAR is split into red (600-700 nm) and far-red (700-800 nm) components
based on the direct/diffuse light ratio.

| Light Source | Red Fraction | Far-Red Fraction | R:FR Ratio |
|-------------|-------------|-----------------|-----------|
| Direct sunlight | 55.0% | 45.8% | ~1.20 |
| Diffuse / cloud | 50.0% | 50.0% | ~1.00 |

Mixed conditions use a weighted blend:

```
direct_fraction = (1 - cloud_cover)
par_red    = PAR * (0.55 * direct_fraction + 0.50 * (1 - direct_fraction))
par_far_red = PAR * (0.458 * direct_fraction + 0.50 * (1 - direct_fraction))
r_fr_ratio = par_red / par_far_red
```

**Reference**: Holmes & Smith (1977), *Photochemistry and Photobiology* 25:539 -- measured
R:FR 1.15-1.25 in direct sun.

### 4.4 Canopy Raycasting

For each plant, rays are traced from the crown toward the sun through the canopy geometry
of all other plants. Taller plants' canopies attenuate light via Beer-Lambert:

```
transmission *= exp(-k * LAI * radial_factor * angular_factor)
```

| Parameter | Meaning |
|-----------|---------|
| `k` | Extinction coefficient (species-specific) |
| `LAI` | Leaf area index of the blocking canopy |
| `radial_factor` | Linear falloff from canopy center: `(1 - dist / radius)` |
| `angular_factor` | Oblique ray correction: `min(1 / cos(zenith), 5)` |

**Closest-approach geometry**: The ray-to-blocker distance is computed using the
closest-approach formula (line-to-point distance on the ray), not the midpoint method.
This correctly handles low sun angles where the midpoint of the height segment can
overshoot the canopy horizontally.

**Crown depth**: The leaf-bearing crown occupies the top 40% of plant height
(`CROWN_DEPTH_FRACTION = 0.40`).

### 4.5 Spectral Splitting Through Canopy

Chlorophyll absorbs red light more strongly than far-red, which shifts the R:FR ratio
lower under canopy shade.

```
T_red     = T^1.3     (red attenuated more by chlorophyll)
T_far_red = T^0.4     (far-red passes through leaves more easily)

plant_R:FR = (par_red * T_red) / (par_far_red * T_far_red)
```

**Reference**: Smith (1982), *Annual Review of Plant Physiology* 33:481.

A plant at full sun receives R:FR ~1.2; a plant under a dense canopy may see R:FR ~0.3-0.5.

### 4.6 Phototropism (Light Asymmetry)

For each plant, four offset rays (east, west, north, south) are traced at half the
canopy radius from center. The gradient vector drives phototropism:

```
grad_x = transmission_east  - transmission_west
grad_y = transmission_north - transmission_south
```

### 4.7 PHYB and SAS Gene Circuits

Defined in `core/src/botany/genome.rs`, these circuits read the `RedFarRedRatio`
environmental signal and produce gene expression levels that the physiology bridge
converts into growth modifiers.

**PHYB (Phytochrome B)**:

```
PHYB = activator(R:FR_signal, Km=0.4, n=3, max=1.0, decay=0.002)
```

Active in full sun (R:FR ~1.2, signal ~0.8), drops under canopy shade.
*Reference*: Quail (2002), *Nature Reviews Molecular Cell Biology* 3:85.

**SAS (Shade Avoidance Syndrome)**:

```
SAS = repressor(R:FR_signal, Km=0.4, n=3, max=0.85, decay=0.002)
```

Activated when R:FR is LOW (shade). Drives stem elongation and branching suppression.
*Reference*: Ballare (1999), *Trends in Plant Science* 4:97.

### 4.8 Growth Modifiers (Physiology Bridge)

Defined in `core/src/botany/physiology_bridge.rs`:

**Shade avoidance elongation**:

```
elongation_factor = 1.0 + hill(SAS, 0.5, 2) * 0.8
```

Range: `[1.0, 1.8]`. At full SAS expression, stems elongate up to 80% faster.

**Shade avoidance branching**:

```
branching_factor = 1.0 - hill(SAS, 0.4, 3) * 0.6
```

Range: `[0.4, 1.0]`. High SAS suppresses lateral branching by up to 60%.

These factors are wired into the growth pipeline: `height *= elongation_factor`,
`lateral_bias *= branching_factor`.

### 4.9 Tests (23 total, including spectral and integration)

Solar astronomy (10):
- `test_solar_declination_summer_solstice` -- +23.44 deg at day 172
- `test_solar_declination_winter_solstice` -- -23.44 deg at day 355
- `test_solar_elevation_noon_equator_equinox` -- ~90 deg at equator equinox
- `test_solar_elevation_night` -- below horizon at midnight
- `test_day_length_equinox` -- ~12 h at 42 N
- `test_day_length_summer_longer` -- summer > winter + 4 h
- `test_clear_sky_par_noon_vs_dawn` -- noon PAR > dawn * 3
- `test_clear_sky_par_below_horizon` -- PAR = 0 below horizon
- `test_compute_solar_state_midday` -- noon: daytime, high PAR
- `test_compute_solar_state_midnight` -- midnight: not daytime, PAR = 0

Raycasting (6):
- `test_raycast_single_plant_full_sun` -- lone plant gets 100% PAR
- `test_raycast_tall_shades_short` -- tall plant shades short
- `test_raycast_low_sun_casts_longer_shadows` -- low sun extends shadows
- `test_raycast_nighttime_zero_flux` -- no flux at night
- `test_distant_plants_no_shading` -- far plants do not shade each other
- `test_light_asymmetry_shaded_from_east` -- east blocker creates negative x gradient

Seasonal/spectral (7):
- `test_day_length_seasonal_variation` -- spring < summer, equinoxes similar
- `test_par_reduced_by_clouds` -- cloudy < clear
- `test_full_clouds_reduce_par_75_percent` -- ratio = 0.25
- `test_solar_state_includes_spectral` -- red and far-red PAR present
- `test_direct_sun_rfr_about_1_2` -- clear-sky R:FR ~1.2
- `test_cloud_diffuse_rfr_about_1_0` -- full cloud R:FR ~1.0
- `test_canopy_shade_lowers_rfr` -- shaded plant has lower R:FR

---

## 5. VOC Signaling and Chemical Ecology (Phase 5)

**Source files**:
- `core/src/botany/metabolome.rs` -- JA/SA synthesis, GLV/MeSA emission
- `core/src/botany/genome.rs` -- JA_RESPONSE, SA_RESPONSE, DEFENSE_PRIMING circuits
- `core/src/terrarium/flora.rs` -- defense VOC emission into odorant grid
- `core/src/terrarium/fauna.rs` -- herbivore grazing trigger

### 5.1 The Defense Cascade (Overview)

```
mechanical_damage --> JA_RESPONSE gene --> jasmonate synthesis
                                              |
                                              v
                                     GLV emission + MeSA emission
                                              |
                                              v
                                  odorant channel 5 (DEFENSE_VOC)
                                              |
                                     [wind dispersal via odorant grid]
                                              |
                                              v
                              neighbor plant detects VOC signal
                                              |
                                              v
                                  SA_RESPONSE + DEFENSE_PRIMING
                                    (AND-gate: VOC + own JA)
```

### 5.2 Damage Sources

Two sources of mechanical damage trigger the JA cascade:

1. **Wind damage** (Phase 1): `mechanical_damage` accumulates from stem bending stress
2. **Herbivore grazing** (this phase): flies graze leaves when hungry

### 5.3 Herbivore Grazing (Fauna)

Fly grazing probability is controlled by hunger via Hill kinetics:

```
grazing_drive = hill(hunger, Km=0.6, n=2)
graze_prob    = grazing_drive * 0.003        # ~0.3% per step at max hunger
```

Grazing requires the fly to be walking (not flying), have found no fruit to eat, and be
within 2 cells of a plant.

**Bite size** via Michaelis-Menten:

```
bite_fraction = michaelis_menten(leaf_biomass, 0.3)
leaf_consumed = min(bite_fraction * 0.008, leaf_biomass * 0.05)
```

**Defense deterrence**: Defended plants repel herbivores.

```
defense_pool = jasmonate_count + salicylate_count
deterrence   = hill(defense_pool, Km=5.0, n=2)
```

A random roll against `deterrence` determines whether the fly avoids the plant. At 5
molecules of combined JA+SA, deterrence is 50%. Well-defended plants with high JA+SA
pools will repel most grazing attempts.

On successful grazing, `mechanical_damage += 0.04`, which feeds into the JA cascade.

### 5.4 Jasmonate Synthesis

Jasmonate is synthesized from JA_RESPONSE gene expression using Hill kinetics:

```
ja_synth_rate = hill(JA_RESPONSE_expr, Km=0.3, n=3) * 2.0 * dt
```

Requires glucose as substrate (linolenic acid precursor). Jasmonate decays with
`half-life ~ 2.3 min`:

```
jasmonate *= (1 - 0.005 * dt)
```

*Reference*: Wasternack & Hause (2013), *Annals of Botany* 111:1021-1058.

### 5.5 Salicylate Synthesis

Salicylate is synthesized from SA_RESPONSE gene expression:

```
sa_synth_rate = hill(SA_RESPONSE_expr, Km=0.4, n=2) * 1.5 * dt
```

Synthesized via the isochorismate pathway (costs glucose via shikimate). SA decays more
slowly than JA (half-life ~ 3.8 min), accumulating for systemic resistance:

```
salicylate *= (1 - 0.003 * dt)
```

*Reference*: Vlot et al. (2009), *Annual Review of Phytopathology* 47:177-206.

### 5.6 Green Leaf Volatile (GLV) Emission

GLVs are C6 aldehydes/alcohols released upon tissue damage via the lipoxygenase pathway.
Produced from jasmonate:

```
ja_norm  = min(jasmonate / 10, 1)
glv_rate = hill(ja_norm, Km=0.5, n=2) * 3.0 * dt
```

GLV is highly volatile with fast atmospheric decay:

```
GLV *= (1 - 0.02 * dt)
```

*Reference*: Matsui (2006), *Current Opinion in Plant Biology* 9:274-280.

### 5.7 Methyl Salicylate (MeSA) Emission

MeSA is the volatile form of salicylic acid, methylated by SA methyltransferase (SAMT):

```
sa_norm   = min(salicylate / 10, 1)
mesa_rate = hill(sa_norm, Km=0.4, n=2) * 2.0 * dt
```

MeSA volatile decay:

```
MeSA *= (1 - 0.015 * dt)
```

*Reference*: Park et al. (2007), *Science* 318:113-116.

### 5.8 Defense VOC Emission Rate

The combined emission intensity (used for odorant grid coupling):

```
defense_voc_emission = min((GLV_count + MeSA_count) * 0.01, 1.0)
```

This value is written to odorant channel `DEFENSE_VOC_IDX = 5` at the canopy height
of each emitting plant. Wind turbulence then disperses the signal through the 3D grid.

### 5.9 Neighbor Detection: AND-Gate Priming

The DEFENSE_PRIMING gene circuit in `genome.rs` uses geometric-mean AND-gate logic,
requiring BOTH signals to be present for full activation:

```
DEFENSE_PRIMING = AND_gate(NeighborVOC_signal, JasmonicAcid_signal)
```

The geometric mean ensures that if either input is zero, the output is zero. A neighbor
plant must both:
1. Detect defense VOCs from the odorant grid (non-zero `NeighborVOC`)
2. Have its own internal JA signal (from `JA_RESPONSE`)

This models the biological reality that defense priming requires the plant to have
already initiated its own damage response pathway before neighbor signals can amplify it.

*Reference*: Engelberth et al. (2004), *PNAS* 101:1781-1785.

### 5.10 Odorant Channel Layout

| Index | Constant | Gas |
|-------|----------|-----|
| 0 | `ETHYL_ACETATE_IDX` | Ethyl acetate (fruit attractant) |
| 1 | `GERANIOL_IDX` | Geraniol (floral) |
| 2 | `AMMONIA_IDX` | Ammonia (decomposition) |
| 3 | `ATMOS_CO2_IDX` | Carbon dioxide |
| 4 | `ATMOS_O2_IDX` | Oxygen |
| 5 | `DEFENSE_VOC_IDX` | Defense VOCs (GLV + MeSA) |

### 5.11 Tests

**VOC signaling (11)**:
- `test_jasmonate_synthesis_from_damage` -- JA_RESPONSE expression produces jasmonate
- `test_jasmonate_decay_without_damage` -- JA decays without continued activation
- `test_glv_emission_requires_ja` -- no GLV without jasmonate
- `test_glv_rises_with_ja` -- high JA produces GLV
- `test_mesa_from_salicylate` -- SA produces MeSA
- `test_defense_voc_emission_rate` -- emission rate capped at 1.0
- `test_no_voc_without_damage` -- no defense VOC without damage cascade
- Additional integration tests for VOC dispersion through odorant grid
- SA_RESPONSE gene circuit tests
- DEFENSE_PRIMING AND-gate tests
- Full cascade integration tests

**Herbivore grazing (5)**: Located in `fauna.rs` integration tests.
- Grazing drive correlates with hunger
- Defense deterrence reduces grazing success
- Leaf damage increments `mechanical_damage`
- Fly gains energy from consumed leaf tissue
- Telemetry event emitted on grazing

---

## System Interconnections

The five physics systems form a coupled network where each system feeds signals into
one or more other systems.

```
                          +--------------+
                          |  3. Weather  |
                          |  (Emergent)  |
                          +---+--+--+----+
                              |  |  |
                  +-----------+  |  +-----------+
                  |              |              |
                  v              v              v
          precipitation    cloud_cover    temp_offset
                  |              |              |
                  v              |              v
        +----------------+      |      metabolic rates
        | 2. Soil Profile |      |      (respiration,
        | (4-layer Fick)  |      |       VOC Q10)
        +-------+--------+      |
                |                |
                v                v
         root moisture    PAR attenuation
                |                |
                v                v
         water stress    +------------------+
                |        | 4. Spectral/R:FR |
                |        | (raycast + PHYB) |
                |        +--------+---------+
                |                 |
                v                 v
         stomatal            SAS expression
         openness                 |
              |                   v
              |           height elongation
              |           branch suppression
              v
       VOC emission rate
              |
              v
   +----------------------+          +-------------------+
   | 5. VOC / Chemical    |  <----   | 1. Wind / Turbulence |
   |    Ecology           |          | (3-octave noise)     |
   |  (JA/SA/GLV/MeSA)    |  ---->   +-----+---------------+
   +----------+-----------+                |
              |                            |
              v                            v
     neighbor defense              wind drag -> stem stress
     priming (AND-gate)            -> damage accumulation
              |                            |
              +----------------------------+
                         |
                         v
               mechanical_damage
                         |
                         v
                JA_RESPONSE activation
                         |
                         v
               defense VOC cascade
```

### Key Cross-System Pathways

| Pathway | From | To | Mechanism |
|---------|------|----|-----------|
| Precipitation -> soil | Weather (3) | Soil (2) | `infiltrate_rainfall()` |
| Cloud -> light | Weather (3) | Spectral (4) | `PAR *= (1 - cloud * 0.75)` |
| Wind -> VOC dispersal | Wind (1) | VOC (5) | Odorant grid advection |
| Wind -> stem damage | Wind (1) | VOC (5) | `mechanical_damage` -> JA cascade |
| R:FR -> SAS -> growth | Spectral (4) | Growth | `elongation_factor`, `branching_factor` |
| Herbivore -> JA cascade | VOC (5) | VOC (5) | Grazing -> `mechanical_damage += 0.04` |
| Soil -> water stress | Soil (2) | VOC (5) | Root moisture -> stomatal openness -> VOC rate |
| Weather -> wind regime | Weather (3) | Wind (1) | Storm increases base wind speed |

---

## Aquarium vs Terrarium Differences

The same equations govern both environments. The only differences are species-specific
parameters and world configuration constants.

### World Configuration

| Parameter | Terrarium | Aquarium |
|-----------|-----------|----------|
| `base_wind_speed_mm_s` | 0.3 | 0.1 (sheltered) |
| `turbulence_intensity` | 0.15 | 0.08 |
| Grid size | 18 x 14 | 14 x 10 |
| Cell size (mm) | 0.25 | 0.28 |
| Depth layers | 6 | 6 |
| Humidity | ~0.6 | 0.99 |
| Surface water | Puddles | Flooded basin |

### Aquatic Species Biomechanics

| Parameter | Terrestrial | Aquatic |
|-----------|------------|---------|
| `wood_density` | 400-600 kg/m^3 | 100 kg/m^3 |
| `drag_coefficient` | 0.40-0.55 | 0.80 (higher in water) |
| `stem_diameter_fraction` | 0.05-0.10 | 0.03 |

Aquatic species (kelp, seagrass, Posidonia) have very low wood density (buoyant, flexible
tissue) and high drag coefficients (water is ~800x denser than air). Because
`sigma_yield = 0.0005 * rho^1.5`, their low wood density means low yield strength, but
this is compensated by their extreme flexibility (low Young's modulus `E = 0.01 * rho^2`)
which prevents the stress ratio from exceeding yield under normal current speeds.

---

## Simulation Step Order

The five systems execute in a fixed order within `step_frame()`:

```
1. step_climate_driver()        -- external climate forcing (if configured)
2. step_weather(dt)             -- Phase 3: emergent weather from humidity/temp/soil/wind
3. step_wind_turbulence(dt)     -- Phase 1: 3-octave wind field update
4. step_atmosphere()            -- gas diffusion, odorant advection
5. step_biomechanics(dt)        -- Phase 1: drag, stress, damage per plant
6. step_soil_profile(dt)        -- Phase 2: vertical transport + infiltration
7. step_plants(dt)              -- includes Phases 4+5:
   a. compute_solar_state()     -- Phase 4: solar position
   b. raycast_canopy_photons()  -- Phase 4: spectral light per plant
   c. step_gene_regulation()    -- Phase 4: PHYB/SAS gene circuits
   d. full_metabolic_step()     -- Phase 5: JA/SA/GLV/MeSA synthesis
   e. defense VOC emission      -- Phase 5: write to odorant grid ch5
   f. neighbor VOC detection    -- Phase 5: read from odorant grid ch5
8. step_fauna(dt)               -- Phase 5: herbivore grazing
```

This ordering ensures that weather updates propagate to wind, soil, and light before
plants and fauna react to them.

---

## Test Summary

| Phase | System | Tests | Run Command |
|-------|--------|-------|-------------|
| 1 | Wind and Turbulence | 12 | `cargo test -p oneura-core --features web -- "biomechanics::tests"` |
| 2 | Soil Depth Profile | 10 | `cargo test -p oneura-core --features web -- "soil_profile::tests"` |
| 3 | Weather State Machine | 12 | (included in biomechanics::tests above) |
| 4 | Spectral Light / R:FR | 23 | `cargo test -p oneura-core --features web -- "terrarium::solar::tests"` |
| 5a | VOC Signaling | 11 | `cargo test -p oneura-core --features web -- "botany::metabolome::tests"` |
| 5b | Herbivore Grazing | 5 | (included in fauna integration tests) |
| 4/5 | Physiology Bridge | 15 | `cargo test -p oneura-core --features web -- "physiology_bridge::tests"` |
| All | Full Suite | 1099 | `cargo test -p oneura-core --features web` |

All 1099 tests pass with 0 failures as of 2026-03-21.

---

## References

- Adams DO, Yang SF (1979) Ethylene biosynthesis. *PNAS* 76:170-174.
- Amthor JS (2000) The McCree-de Wit-Penning de Vries-Thornley respiration paradigms. *Plant Cell Environ* 23:1241-1257.
- Ballare CL (1999) Keeping up with the neighbours: phytochrome sensing. *Trends Plant Sci* 4:97-102.
- Campbell GS, Norman JM (1998) *An Introduction to Biophysical Ecology*. Springer.
- Casal JJ (2012) Shade avoidance. *Plant Cell Environ* 35:271-285.
- Dong JG et al. (1992) Purification and characterization of ACC oxidase. *Planta* 188:439-444.
- Engelberth J et al. (2004) Airborne signals prime plants against insect herbivore attack. *PNAS* 101:1781-1785.
- Farquhar GD et al. (1980) A biochemical model of photosynthetic CO2 assimilation. *Planta* 149:78-90.
- Franklin KA (2008) Shade avoidance. *New Phytol* 179:930-944.
- Franklin KA, Whitelam GC (2005) Phytochromes and shade-avoidance. *J Exp Bot* 56:3271-3282.
- Guenther A et al. (1993) Isoprene and monoterpene emission. *J Geophys Res* 98:12609-12617.
- Holmes MG, Smith H (1977) The function of phytochrome. *Photochem Photobiol* 25:539-545.
- Jury WA, Horton R (2004) *Soil Physics*. 6th ed. Wiley.
- Lu N, Likos WJ (2004) *Unsaturated Soil Mechanics*. Wiley.
- Matsui K (2006) Green leaf volatiles. *Curr Opin Plant Biol* 9:274-280.
- Meeus J (1991) *Astronomical Algorithms*. Willmann-Bell.
- Monteith JL, Unsworth MH (2013) *Principles of Environmental Physics*. 4th ed. Academic Press.
- Niklas KJ (1992) *Plant Biomechanics*. Univ. Chicago Press.
- Park SW et al. (2007) Methyl salicylate is a critical signal. *Science* 318:113-116.
- Quail PH (2002) Phytochrome photosensory signalling networks. *Nat Rev Mol Cell Biol* 3:85-93.
- Smith H (1982) Light quality, photoperception, and plant strategy. *Annu Rev Plant Physiol* 33:481-518.
- Vlot AC et al. (2009) Salicylic acid in plant defence. *Annu Rev Phytopathol* 47:177-206.
- Wasternack C, Hause B (2013) Jasmonates: biosynthesis, perception, signal transduction. *Ann Bot* 111:1021-1058.
