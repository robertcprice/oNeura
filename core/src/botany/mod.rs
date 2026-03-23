//! Molecular Botany subsystem.
//!
//! Provides genomic and metabolic simulation for specific plant species.

pub mod arabidopsis;
pub mod genome;
pub mod metabolome;
pub mod morphology;
pub mod physiology_bridge;
pub mod species;
pub mod visual_phenotype;

pub use arabidopsis::{
    ArabidopsisRoot, ArabidopsisRootCell, ArabidopsisRootCellType,
    auxin_transport_step, gravitropism_response, nitrate_uptake, root_growth_step,
    root_simulation_step, water_uptake,
};
pub use genome::{
    BotanicalGenome, BotanicalSpecies, EnvironmentState, EnvironmentalSignal, GenomicCatalog,
    PlantGeneCircuit,
};
pub use metabolome::{MetabolicReport, PlantMetabolome, SubstrateReport};
pub use morphology::{MorphNode, NodeType, PlantMorphology};
pub use species::{
    botanical_species_profiles, sample_species_profile, species_profile_by_taxonomy,
    BotanicalGrowthForm, BotanicalSpeciesProfile,
};
