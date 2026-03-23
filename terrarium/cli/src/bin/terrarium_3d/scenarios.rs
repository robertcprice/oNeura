//! Preset education scenarios for the BioSim Explorer education MVP.
//!
//! Each scenario configures a TerrariumWorld with specific parameters and
//! provides phase-based narration text for guided learning.

use oneura_core::drosophila::DrosophilaScale;
use oneura_core::terrarium::TerrariumWorld;

/// Available preset scenarios.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scenario {
    Default,
    DroughtSurvival,
    CompetitiveExclusion,
    NutrientCycling,
    PopulationBoom,
    NightEcology,
    StressResilience,
    MicrobialWorld,
    ClimateImpact,
    AmrEmergence,
    SoilHealth,
}

impl Scenario {
    pub fn label(&self) -> &'static str {
        match self {
            Scenario::Default => "FREE EXPLORE",
            Scenario::DroughtSurvival => "DROUGHT SURVIVAL",
            Scenario::CompetitiveExclusion => "COMPETITION",
            Scenario::NutrientCycling => "NUTRIENT CYCLING",
            Scenario::PopulationBoom => "POPULATION BOOM",
            Scenario::NightEcology => "NIGHT ECOLOGY",
            Scenario::StressResilience => "STRESS RESILIENCE",
            Scenario::MicrobialWorld => "MICROBIAL WORLD",
            Scenario::ClimateImpact => "CLIMATE IMPACT",
            Scenario::AmrEmergence => "AMR EMERGENCE",
            Scenario::SoilHealth => "SOIL HEALTH",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Scenario::Default => "Free exploration mode. Observe a balanced terrarium ecosystem.",
            Scenario::DroughtSurvival => "Water sources are scarce. Watch plants compete for moisture and observe how drought stress propagates through the food web.",
            Scenario::CompetitiveExclusion => "Dense planting with limited light. Beer-Lambert canopy shading creates winners and losers. Size hierarchies emerge from asymmetric competition.",
            Scenario::NutrientCycling => "Nutrient-rich soil with many organisms. Track how nitrogen and carbon flow between soil microbes, plants, and decomposers.",
            Scenario::PopulationBoom => "Abundant resources trigger rapid population growth. Watch boom-bust dynamics as flies exhaust food supplies and populations crash.",
            Scenario::NightEcology => "Start at dusk. Observe how moonlight, tidal moisture, and nocturnal activity patterns shape ecosystem behavior after dark.",
            Scenario::StressResilience => "Harsh conditions with temperature extremes. Organisms that survive demonstrate stress tolerance mechanisms.",
            Scenario::MicrobialWorld => "Minimal macro-organisms, rich substrate chemistry. Zoom to molecular level to observe soil chemistry dynamics.",
            Scenario::ClimateImpact => "RCP4.5 climate scenario. Rising CO2 and temperature drive nutrient cycling, microbial shifts, and eco-evolutionary feedbacks across 11 integrated modules.",
            Scenario::AmrEmergence => "Antibiotic resistance under selective pressure. Horizontal gene transfer, biofilm formation, and resistance evolution emerge from microbial community dynamics.",
            Scenario::SoilHealth => "Soil biogeochemistry focus. Carbon, nitrogen, and phosphorus cycling coupled to microbial community assembly, metabolic flux, and phylogenetic diversification.",
        }
    }

    /// Cycle to next scenario.
    pub fn next(&self) -> Self {
        match self {
            Scenario::Default => Scenario::DroughtSurvival,
            Scenario::DroughtSurvival => Scenario::CompetitiveExclusion,
            Scenario::CompetitiveExclusion => Scenario::NutrientCycling,
            Scenario::NutrientCycling => Scenario::PopulationBoom,
            Scenario::PopulationBoom => Scenario::NightEcology,
            Scenario::NightEcology => Scenario::StressResilience,
            Scenario::StressResilience => Scenario::MicrobialWorld,
            Scenario::MicrobialWorld => Scenario::ClimateImpact,
            Scenario::ClimateImpact => Scenario::AmrEmergence,
            Scenario::AmrEmergence => Scenario::SoilHealth,
            Scenario::SoilHealth => Scenario::Default,
        }
    }

    /// Get narration text for the current frame count within this scenario.
    /// Returns None if no narration is due at this frame.
    pub fn narration(&self, frame: usize) -> Option<&'static str> {
        match self {
            Scenario::Default => match frame {
                0..=60 => Some("Welcome to oNeura Terrarium 3D. Use WASD to pan, mouse to rotate, scroll to zoom."),
                61..=180 => Some("Green dots are plants. Yellow are flies. Blue pools are water sources."),
                181..=300 => Some("Try zooming in (+/= key) to see organism details. Select entities by clicking."),
                301..=420 => Some("Press 2-0 to switch overlay modes and visualize soil moisture, temperature, chemistry."),
                _ => None,
            },
            Scenario::DroughtSurvival => match frame {
                0..=90 => Some("DROUGHT: Only 1 water source in this arid landscape. Plants must compete for scarce moisture."),
                91..=240 => Some("Watch soil moisture (press 2). Plants near water thrive; distant ones wilt."),
                241..=420 => Some("Flies depend on plant-produced fruits. As plants die, the food web collapses."),
                421..=600 => Some("Surviving plants have deeper roots and better water-use efficiency — natural selection in action."),
                601..=900 => Some("Notice: soil microbes near dead plants release nutrients. Decomposition fuels recovery."),
                _ => None,
            },
            Scenario::CompetitiveExclusion => match frame {
                0..=90 => Some("COMPETITION: 24 plants packed into limited space. Beer-Lambert law governs light."),
                91..=240 => Some("Taller plants shade shorter neighbors. Watch the canopy close and suppress understory growth."),
                241..=420 => Some("Competitive exclusion: weaker plants lose vitality as light drops below compensation point."),
                421..=600 => Some("Size hierarchy emerges — CV in height increases as winners separate from losers."),
                601..=900 => Some("Root competition adds a second dimension. Plants with better root uptake survive even in shade."),
                _ => None,
            },
            Scenario::NutrientCycling => match frame {
                0..=90 => Some("NUTRIENTS: Rich soil with active microbial guilds. Press 7 for O2 overlay, 0 for nitrogen."),
                91..=240 => Some("Nitrifiers convert ammonium → nitrate. Denitrifiers close the loop under low oxygen."),
                241..=420 => Some("Plant roots absorb nitrate. Root exudates feed heterotrophic microbes. A cycle forms."),
                421..=600 => Some("Earthworm bioturbation mixes soil layers, redistributing nutrients vertically."),
                601..=900 => Some("Zoom to molecular level (+++ key) to see individual chemical species concentrations."),
                _ => None,
            },
            Scenario::PopulationBoom => match frame {
                0..=90 => Some("BOOM: Abundant water and fruits attract many flies. Population grows exponentially."),
                91..=240 => Some("Watch the fly population sparkline in the panel. Classic J-curve growth phase."),
                241..=420 => Some("Resources deplete. Flies compete for diminishing fruit supply."),
                421..=600 => Some("Population crashes as carrying capacity is exceeded. Boom-bust dynamics."),
                601..=900 => Some("Surviving flies are those with better energy efficiency — evolutionary pressure selects."),
                _ => None,
            },
            Scenario::NightEcology => match frame {
                0..=90 => Some("NIGHT: Simulation starts near dusk. Watch the sky darken and moon rise."),
                91..=240 => Some("Photosynthesis stops at night. Plants switch to respiration-only metabolism."),
                241..=420 => Some("Moonlight affects tidal moisture. Spring tides increase soil water availability."),
                421..=600 => Some("Some flies are nocturnal-adapted. Activity patterns shift with the light cycle."),
                601..=900 => Some("Dawn approaches. CO2 accumulated overnight will be fixed by photosynthesis at sunrise."),
                _ => None,
            },
            Scenario::StressResilience => match frame {
                0..=90 => Some("STRESS: Extreme temperature and low moisture. Only stress-tolerant organisms survive."),
                91..=240 => Some("Sharpe-Schoolfield thermal response: enzyme activity drops at temperature extremes."),
                241..=420 => Some("Plants with higher water-use efficiency maintain turgor longer under drought."),
                421..=600 => Some("Flies conserve energy by reducing flight speed. Metabolism slows at temperature extremes."),
                601..=900 => Some("The ecosystem reaches a new equilibrium — fewer organisms, but each well-adapted to stress."),
                _ => None,
            },
            Scenario::MicrobialWorld => match frame {
                0..=90 => Some("MICROBES: Minimal plants/flies. Focus on soil chemistry and microbial dynamics."),
                91..=240 => Some("Press 5 for chemistry overlay. Watch substrate reaction-diffusion patterns."),
                241..=420 => Some("Zoom to cellular level (++ key) to see individual soil cell chemistry."),
                421..=600 => Some("At molecular zoom (+++ key), each TerrariumSpecies shows as a density bar."),
                601..=900 => Some("Michaelis-Menten kinetics govern all reactions. Substrate concentrations drive rates."),
                _ => None,
            },
            Scenario::ClimateImpact => match frame {
                0..=90 => Some("CLIMATE: RCP4.5 scenario — rising CO2 and temperature drive ecosystem change across 11 integrated modules."),
                91..=240 => Some("Nutrient cycling accelerates with warming: C/N/P mineralization rates increase, CO2 flux rises."),
                241..=420 => Some("Microbial community shifts: thermophilic taxa gain advantage as mesophiles decline."),
                421..=600 => Some("Eco-evolutionary feedback: traits under selection shift as environmental conditions change."),
                601..=900 => Some("Phylogenetic diversification: new lineages emerge under novel selective pressures."),
                _ => None,
            },
            Scenario::AmrEmergence => match frame {
                0..=90 => Some("AMR: Antibiotic pressure selects for resistance. Watch MDR strains emerge through HGT and mutation."),
                91..=240 => Some("Horizontal gene transfer spreads resistance plasmids through the biofilm community."),
                241..=420 => Some("Biofilm formation: cells produce EPS matrix, enabling quorum sensing and cooperative resistance."),
                421..=600 => Some("MIC fold-change increases: resistant strains tolerate higher antibiotic concentrations."),
                601..=900 => Some("Resistance is costly: fitness trade-offs shape the balance between resistant and susceptible populations."),
                _ => None,
            },
            Scenario::SoilHealth => match frame {
                0..=90 => Some("SOIL: Biogeochemistry focus — carbon, nitrogen, and phosphorus cycling in a living soil."),
                91..=240 => Some("Microbiome assembly: diverse taxa partition resources through metabolic specialization."),
                241..=420 => Some("FBA metabolic flux analysis estimates microbial growth yields from available substrates."),
                421..=600 => Some("Guild dynamics: latent ecological variables track unobserved community structure."),
                601..=900 => Some("Soil health emerges from the interaction of all 11 modules — no single process dominates."),
                _ => None,
            },
        }
    }
}

/// Configure a TerrariumWorld for a specific scenario.
/// Called after world creation to modify parameters and entity placement.
pub fn apply_scenario(world: &mut TerrariumWorld, scenario: Scenario, seed: u64) {
    match scenario {
        Scenario::Default => {
            // Default demo layout — add extra entities for visual richness
            for i in 0..20 {
                let x = (i * 7 + 3) % (world.config.width - 2);
                let y = (i * 11 + 2) % (world.config.height - 2);
                let _ = world.add_plant(x, y, None, None);
            }
            for i in 0..12 {
                let x = 3.0 + (i as f32 * 3.1) % (world.config.width as f32 - 6.0);
                let y = 3.0 + (i as f32 * 2.3) % (world.config.height as f32 - 6.0);
                world.add_fly(DrosophilaScale::Tiny, x, y, seed + i as u64);
            }
        }
        Scenario::DroughtSurvival => {
            // Sparse water, scattered plants
            world.waters.clear();
            world.add_water(world.config.width / 2, world.config.height / 2, 80.0, 0.001);
            // Plant ring around the single water source
            let cx = world.config.width / 2;
            let cy = world.config.height / 2;
            for i in 0..16 {
                let angle = i as f32 * std::f32::consts::TAU / 16.0;
                let r = 3.0 + (i as f32 * 0.7);
                let x = (cx as f32 + angle.cos() * r).clamp(1.0, (world.config.width - 2) as f32)
                    as usize;
                let y = (cy as f32 + angle.sin() * r).clamp(1.0, (world.config.height - 2) as f32)
                    as usize;
                let _ = world.add_plant(x, y, None, None);
            }
            // Few flies
            for i in 0..4 {
                let x = cx as f32 + (i as f32 - 1.5) * 2.0;
                let y = cy as f32 + (i as f32 - 1.5);
                world.add_fly(DrosophilaScale::Tiny, x, y, seed + i as u64);
            }
            // Reduce moisture globally
            for m in world.moisture_field_mut().iter_mut() {
                *m *= 0.3;
            }
        }
        Scenario::CompetitiveExclusion => {
            // Dense planting, normal water
            for i in 0..24 {
                let x = 2 + (i % 6) * 6;
                let y = 2 + (i / 6) * 6;
                let x = x.min(world.config.width - 2);
                let y = y.min(world.config.height - 2);
                let _ = world.add_plant(x, y, None, None);
            }
            // Few flies for pollination
            for i in 0..6 {
                let x = 5.0 + i as f32 * 5.0;
                let y = 5.0 + i as f32 * 3.0;
                world.add_fly(DrosophilaScale::Tiny, x, y, seed + i);
            }
        }
        Scenario::NutrientCycling => {
            // Moderate plants, boost initial substrate concentrations
            for i in 0..12 {
                let x = 3 + (i * 5) % (world.config.width - 4);
                let y = 3 + (i * 7) % (world.config.height - 4);
                let _ = world.add_plant(x, y, None, None);
            }
            for i in 0..8 {
                let x = 4.0 + i as f32 * 4.5;
                let y = 4.0 + i as f32 * 3.2;
                world.add_fly(DrosophilaScale::Tiny, x, y, seed + i);
            }
            // Boost moisture for nutrient cycling
            for m in world.moisture_field_mut().iter_mut() {
                *m = (*m * 1.5).min(0.95);
            }
        }
        Scenario::PopulationBoom => {
            // Lots of water and fruit, many flies
            for i in 0..4 {
                let x = 5 + i * 10;
                let y = 5 + i * 7;
                world.add_water(
                    x.min(world.config.width - 2),
                    y.min(world.config.height - 2),
                    200.0,
                    0.001,
                );
            }
            for i in 0..10 {
                let x = 3 + (i * 4) % (world.config.width - 4);
                let y = 3 + (i * 5) % (world.config.height - 4);
                let _ = world.add_plant(x, y, None, None);
            }
            // Many flies for boom dynamics
            for i in 0..20 {
                let x = 2.0 + (i as f32 * 2.1) % (world.config.width as f32 - 4.0);
                let y = 2.0 + (i as f32 * 1.7) % (world.config.height as f32 - 4.0);
                world.add_fly(DrosophilaScale::Tiny, x, y, seed + i);
            }
        }
        Scenario::NightEcology => {
            // Standard layout but advance time to near dusk
            for i in 0..15 {
                let x = (i * 5 + 3) % (world.config.width - 2);
                let y = (i * 7 + 2) % (world.config.height - 2);
                let _ = world.add_plant(x, y, None, None);
            }
            for i in 0..10 {
                let x = 3.0 + (i as f32 * 3.5) % (world.config.width as f32 - 6.0);
                let y = 3.0 + (i as f32 * 2.8) % (world.config.height as f32 - 6.0);
                world.add_fly(DrosophilaScale::Tiny, x, y, seed + i);
            }
            // Advance simulation to evening (light cycle ~0.7 = dusk)
            for _ in 0..200 {
                let _ = world.step_frame();
            }
        }
        Scenario::StressResilience => {
            // Few entities, harsh conditions
            for i in 0..8 {
                let x = 4 + (i * 6) % (world.config.width - 4);
                let y = 4 + (i * 5) % (world.config.height - 4);
                let _ = world.add_plant(x, y, None, None);
            }
            for i in 0..4 {
                let x = 5.0 + i as f32 * 8.0;
                let y = 5.0 + i as f32 * 6.0;
                world.add_fly(DrosophilaScale::Tiny, x, y, seed + i);
            }
            // Reduce moisture heavily for stress
            for m in world.moisture_field_mut().iter_mut() {
                *m *= 0.2;
            }
        }
        Scenario::MicrobialWorld => {
            // Minimal plants/flies, focus on substrate
            for i in 0..4 {
                let x = 8 + i * 10;
                let y = 8 + i * 6;
                let _ = world.add_plant(
                    x.min(world.config.width - 2),
                    y.min(world.config.height - 2),
                    None,
                    None,
                );
            }
            world.add_fly(DrosophilaScale::Tiny, 10.0, 10.0, seed);
            // Boost moisture for microbial activity
            for m in world.moisture_field_mut().iter_mut() {
                *m = (*m * 2.0).min(0.98);
            }
        }
        Scenario::ClimateImpact => {
            // Rich ecosystem to observe climate-driven changes
            for i in 0..16 {
                let x = 3 + (i * 5) % (world.config.width - 4);
                let y = 3 + (i * 6) % (world.config.height - 4);
                let _ = world.add_plant(x, y, None, None);
            }
            for i in 0..3 {
                let x = 5 + i * 12;
                let y = 5 + i * 8;
                world.add_water(
                    x.min(world.config.width - 2),
                    y.min(world.config.height - 2),
                    120.0,
                    0.0006,
                );
            }
            for i in 0..8 {
                let x = 4.0 + (i as f32 * 4.5) % (world.config.width as f32 - 8.0);
                let y = 4.0 + (i as f32 * 3.2) % (world.config.height as f32 - 8.0);
                world.add_fly(DrosophilaScale::Tiny, x, y, seed + i);
            }
        }
        Scenario::AmrEmergence => {
            // Dense microbial environment with few macro-organisms
            for i in 0..6 {
                let x = 5 + (i * 7) % (world.config.width - 4);
                let y = 5 + (i * 9) % (world.config.height - 4);
                let _ = world.add_plant(x, y, None, None);
            }
            world.add_fly(DrosophilaScale::Tiny, 15.0, 10.0, seed);
            // High moisture for biofilm formation
            for m in world.moisture_field_mut().iter_mut() {
                *m = (*m * 1.8).min(0.95);
            }
        }
        Scenario::SoilHealth => {
            // Diverse plant community with rich soil
            for i in 0..20 {
                let x = 2 + (i * 4) % (world.config.width - 3);
                let y = 2 + (i * 5) % (world.config.height - 3);
                let _ = world.add_plant(x, y, None, None);
            }
            for i in 0..4 {
                let x = 8 + i * 10;
                let y = 6 + i * 8;
                world.add_water(
                    x.min(world.config.width - 2),
                    y.min(world.config.height - 2),
                    180.0,
                    0.0005,
                );
            }
            for i in 0..6 {
                let x = 5.0 + (i as f32 * 5.5) % (world.config.width as f32 - 10.0);
                let y = 5.0 + (i as f32 * 4.3) % (world.config.height as f32 - 10.0);
                world.add_fly(DrosophilaScale::Tiny, x, y, seed + i);
            }
            // Rich moisture for nutrient cycling
            for m in world.moisture_field_mut().iter_mut() {
                *m = (*m * 1.6).min(0.92);
            }
        }
    }
}
