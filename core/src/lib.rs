//! oNeura Core - GPU-accelerated molecular brain simulator.
//!
//! This crate provides the main simulation engine, including the biophysical
//! neural model, the molecular atmosphere, and the integrated terrarium world.

pub mod constants;
pub mod neuron_arrays;
pub mod synapse_arrays;
pub mod types;

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod gpu;

pub mod spike_propagation;
pub mod stdp;

pub mod botany;
pub mod cellular_metabolism;
pub mod circadian;
pub mod gene_expression;
pub mod glia;
pub mod metabolism;
pub mod microtubules;
pub mod molecular_atmosphere;
pub mod pharmacology;
pub mod plant_cellular;
pub mod plant_organism;
pub mod substrate_ir;
pub mod terrarium;
pub use molecular_atmosphere::FruitSourceState;
pub use terrarium as terrarium_world;
pub use terrarium::evolve as terrarium_evolve;
pub use terrarium::{
    EcologyTelemetryEvent, TerrariumFruitPatch, TerrariumPlantSnapshot, TerrariumTopdownView,
    TerrariumWorld, TerrariumWorldConfig, TerrariumWorldSnapshot, WorldGenome,
};
pub mod ant_colony;
pub mod atomistic_chemistry;
pub mod atomistic_topology;
pub mod brain_regions;
pub mod celegans;
pub mod consciousness;
pub mod doom_brain;
pub mod drosophila;
pub mod drosophila_population;
pub mod ecology_events;
pub use ecology_events::{
    step_food_patches, step_seed_bank, FoodPatchStepResult, SeedBankStepResult,
};
pub mod ecology_fields;
pub mod ecology_telemetry;
pub mod enzyme_probes;
pub mod field_coupling;
pub mod fly_metabolism;
pub mod molecular_dynamics;
pub mod network;
pub mod neural_molecular_simulator;
pub mod nutrient_cycling;
pub mod organism_catalog;
pub mod organism_metabolism;
pub mod plant_competition;
#[cfg(feature = "python")]
pub mod python;
pub mod retina;
pub mod seed_cellular;
pub mod sensory_mapping;
pub mod soil_broad;
pub mod soil_fauna;
pub mod soil_uptake;
pub mod subatomic_quantum;
pub mod substrate_coupling;
pub mod terrarium_render;
pub mod terrarium_render_pipeline;
pub mod terrarium_scene_query;

pub mod biofilm_dynamics;
pub mod climate_scenarios;
pub mod dishbrain_pong;
pub mod drug_discovery;
pub mod enzyme_engineering;
pub mod pharma_lab;
pub mod structure_ingest;
pub mod eco_evolutionary_feedback;
pub mod ecosystem_integration;
pub mod horizontal_gene_transfer;
pub mod metabolic_flux;
pub mod microbiome_assembly;
pub mod phylogenetic_tracker;
pub mod population_genetics;
pub mod resistance_evolution;

#[cfg(feature = "web")]
pub mod terrarium_web_analysis;
#[cfg(feature = "web")]
pub mod terrarium_web_annotations;
#[cfg(feature = "web")]
pub mod terrarium_web_auth;
#[cfg(feature = "web")]
pub mod terrarium_web_cutaway;
#[cfg(feature = "web")]
pub mod terrarium_web_evolution;
#[cfg(feature = "web")]
pub mod terrarium_web_handlers;
#[cfg(feature = "web")]
pub mod terrarium_web_inspect;
#[cfg(feature = "web")]
pub mod terrarium_web_protocol;
#[cfg(feature = "web")]
pub mod terrarium_web_state;
#[cfg(feature = "web")]
pub mod terrarium_web_tournament;

pub mod whole_cell;
pub mod whole_cell_data;
pub mod whole_cell_submodels;

#[allow(dead_code)]
pub(crate) mod whole_cell_assembly_fallbacks;
#[allow(dead_code)]
pub(crate) mod whole_cell_assembly_projection;
#[allow(dead_code)]
pub(crate) mod whole_cell_asset_fallbacks;
#[allow(dead_code)]
pub(crate) mod whole_cell_chromosome_math;
#[allow(dead_code)]
pub(crate) mod whole_cell_complex_channels;
#[allow(dead_code)]
pub(crate) mod whole_cell_inventory_authority;
#[allow(dead_code)]
pub(crate) mod whole_cell_named_complex_dynamics;
#[allow(dead_code)]
pub(crate) mod whole_cell_process_occupancy;
#[allow(dead_code)]
pub(crate) mod whole_cell_process_weights;
#[allow(dead_code)]
pub(crate) mod whole_cell_rule_math;
#[allow(dead_code)]
pub(crate) mod whole_cell_scale_reducers;
#[allow(dead_code)]
pub(crate) mod whole_cell_signal_estimators;
