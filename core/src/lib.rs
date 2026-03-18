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

pub mod cellular_metabolism;
pub mod circadian;
pub mod gene_expression;
pub mod glia;
pub mod metabolism;
pub mod microtubules;
pub mod molecular_atmosphere;
pub mod botany;
pub mod pharmacology;
pub mod plant_cellular;
pub mod plant_organism;
pub mod substrate_ir;
pub mod terrarium;
pub mod terrarium_render;
pub mod terrarium_render_pipeline;
pub mod terrarium_scene_query;
pub mod drosophila;
pub mod drosophila_population;
pub mod ecology_events;
pub mod ecology_fields;
pub mod ecology_telemetry;
pub mod nutrient_cycling;
pub mod soil_fauna;
pub mod celegans;
pub mod ant_colony;
pub mod soil_broad;
pub mod soil_uptake;
pub mod substrate_coupling;
pub mod plant_competition;
pub mod seed_cellular;
pub mod organism_metabolism;
pub mod fly_metabolism;
pub mod molecular_dynamics;
pub mod network;
pub mod brain_regions;
pub mod consciousness;
pub mod sensory_mapping;
pub mod enzyme_probes;
pub mod atomistic_chemistry;
pub mod atomistic_topology;
pub mod doom_brain;
pub mod retina;
pub mod field_coupling;
pub mod neural_molecular_simulator;
pub mod subatomic_quantum;
pub mod python;

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
