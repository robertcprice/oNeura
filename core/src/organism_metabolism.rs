//! Universal metabolic observation surface for all terrarium organisms.
//!
//! Every organism -- microbe, plant, insect -- implements [`OrganismMetabolism`]
//! with its own internal biochemistry. The trait exposes six read-only metrics
//! that let higher-level systems (brain, evolution engine, telemetry) observe
//! metabolic state without knowing organism-specific internals.
//!
//! The six metrics are grounded in universal biochemistry:
//! - **Energy charge** (Atkinson 1968): [ATP + 0.5*ADP] / [ATP+ADP+AMP], the
//!   universal currency of every living cell from E. coli to Drosophila.
//! - **Biomass**: total dry weight, the physical manifestation of growth.
//! - **Substrate saturation**: primary carbon source availability.
//! - **Oxygen status**: aerobic/anaerobic state.
//! - **Stress**: boolean composite of critically low pools.
//! - **Metabolic rate fraction**: current throughput vs. theoretical maximum.

/// Universal metabolic observation surface for all terrarium organisms.
///
/// Each organism implements this with its own internal biochemistry.
/// All methods are read-only and allocation-free.
pub trait OrganismMetabolism {
    /// Adenylate energy charge: \[ATP + 0.5·ADP\] / \[ATP+ADP+AMP\].
    /// Range 0.0–1.0. Values <0.5 = stressed, >0.8 = healthy (Atkinson 1968).
    fn energy_charge(&self) -> f32;

    /// Total biomass in mg. Organism-specific: dry weight for microbes,
    /// tissue mass for plants, body mass for insects.
    fn biomass_mg(&self) -> f32;

    /// Primary carbon substrate saturation (0.0–1.0).
    /// Glucose/trehalose for animals, sucrose for plants, glucose for microbes.
    fn substrate_saturation(&self) -> f32;

    /// Oxygen availability (0.0–1.0). 1.0 = normoxic, 0.0 = anoxic.
    fn oxygen_status(&self) -> f32;

    /// Metabolic stress indicator. `true` = at least one pool critically low.
    fn is_stressed(&self) -> bool;

    /// Current metabolic rate as fraction of maximum (0.0–1.0).
    /// Driven by substrate availability and demand.
    fn metabolic_rate_fraction(&self) -> f32;
}
