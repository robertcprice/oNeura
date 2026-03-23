use super::*;
use crate::whole_cell_data::{
    WholeCellAssetClass, WholeCellBulkField, WholeCellComplexComponentSpec, WholeCellOperonSpec,
    WholeCellProteinProductSpec,
};
use crate::whole_cell_submodels::{
    LocalChemistryReport, LocalChemistrySiteReport, LocalMDProbeRequest, Syn3ASubsystemPreset,
    WholeCellChemistrySite,
};

fn distribute_total_by_weights(weights: &[f32], mean_value: f32) -> Vec<f32> {
    let total = mean_value.max(0.0) * weights.len() as f32;
    let weight_sum = weights
        .iter()
        .map(|weight| weight.max(0.0))
        .sum::<f32>()
        .max(1.0e-6);
    weights
        .iter()
        .map(|weight| total * weight.max(0.0) / weight_sum)
        .collect()
}

#[test]
fn test_cpu_whole_cell_progresses_state() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        dt_ms: 0.5,
        ..WholeCellConfig::default()
    });
    let start = sim.snapshot();

    sim.run(40);

    let end = sim.snapshot();
    let complex = sim.complex_assembly_state();
    assert_eq!(sim.backend(), WholeCellBackend::Cpu);
    assert!(end.time_ms > start.time_ms);
    assert!(end.replicated_bp > start.replicated_bp);
    assert!(end.surface_area_nm2 > start.surface_area_nm2);
    assert!(end.division_progress >= start.division_progress);
    assert!(end.atp_mm > 0.0);
    assert!(complex.total_complexes() > 0.0);
    assert!(complex.ftsz_target > 0.0);
}

#[test]
fn checkpoint_preserves_next_stochastic_expression_step() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        dt_ms: 1.0,
        ..WholeCellConfig::default()
    });
    sim.stochastic_config.enabled = true;
    sim.stochastic_config.mean_burst_size = 3.5;
    sim.stochastic_config.promoter_on_rate = 0.12;
    sim.stochastic_config.promoter_off_rate = 0.08;
    sim.organism_expression.transcription_units = vec![WholeCellTranscriptionUnitState {
        name: "stochastic_test".to_string(),
        gene_count: 1,
        copy_gain: 1.0,
        basal_activity: 1.0,
        effective_activity: 1.0,
        support_level: 1.0,
        stress_penalty: 1.0,
        transcript_abundance: 6.0,
        protein_abundance: 4.0,
        transcript_synthesis_rate: 0.5,
        protein_synthesis_rate: 0.4,
        transcript_turnover_rate: 0.1,
        protein_turnover_rate: 0.05,
        promoter_open_fraction: 0.0,
        active_rnap_occupancy: 0.0,
        transcription_length_nt: 300.0,
        transcription_progress_nt: 0.0,
        nascent_transcript_abundance: 0.0,
        mature_transcript_abundance: 6.0,
        damaged_transcript_abundance: 0.0,
        active_ribosome_occupancy: 0.0,
        translation_length_aa: 100.0,
        translation_progress_aa: 0.0,
        nascent_protein_abundance: 0.0,
        mature_protein_abundance: 4.0,
        damaged_protein_abundance: 0.0,
        process_drive: WholeCellProcessWeights::default(),
    }];

    sim.run(1);
    let checkpoint = sim.checkpoint_state().expect("checkpoint whole-cell");
    let mut restored =
        WholeCellSimulator::from_checkpoint_state(checkpoint).expect("restore whole-cell");

    sim.run(1);
    restored.run(1);

    assert_eq!(
        sim.stochastic_operon_states,
        restored.stochastic_operon_states
    );
    assert_eq!(
        sim.organism_expression.transcription_units,
        restored.organism_expression.transcription_units
    );
    assert_eq!(sim.snapshot().step_count, restored.snapshot().step_count);
}

#[test]
fn test_registry_transport_reaction_updates_authoritative_glucose_pool() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    sim.glucose_mm = 0.0;
    sim.organism_species = vec![WholeCellSpeciesRuntimeState {
        id: "pool_glucose".to_string(),
        name: "glucose".to_string(),
        species_class: WholeCellSpeciesClass::Pool,
        compartment: "cytosol".to_string(),
        asset_class: WholeCellAssetClass::Energy,
        basal_abundance: 24.0,
        bulk_field: Some(WholeCellBulkField::Glucose),
        spatial_scope: WholeCellSpatialScope::WellMixed,
        patch_domain: WholeCellPatchDomain::Distributed,
        chromosome_domain: None,
        operon: None,
        parent_complex: None,
        subsystem_targets: Vec::new(),
        count: 0.0,
        anchor_count: 24.0,
        synthesis_rate: 0.0,
        turnover_rate: 0.0,
        process_weights: WholeCellProcessWeights::default(),
    }];
    sim.organism_reactions = vec![WholeCellReactionRuntimeState {
        id: "pool_glucose_transport".to_string(),
        name: "glucose transport".to_string(),
        reaction_class: WholeCellReactionClass::PoolTransport,
        asset_class: WholeCellAssetClass::Energy,
        nominal_rate: 8.0,
        catalyst: None,
        operon: None,
        reactants: Vec::new(),
        products: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
            species_id: "pool_glucose".to_string(),
            stoichiometry: 1.0,
        }],
        subsystem_targets: Vec::new(),
        spatial_scope: WholeCellSpatialScope::WellMixed,
        patch_domain: WholeCellPatchDomain::Distributed,
        chromosome_domain: None,
        current_flux: 0.0,
        cumulative_extent: 0.0,
        reactant_satisfaction: 1.0,
        catalyst_support: 1.0,
        process_weights: WholeCellProcessWeights::default(),
    }];

    sim.update_runtime_process_reactions(1.0, 0.0, 0.0);

    assert!(sim.glucose_mm > 0.0);
    assert!(sim.organism_species[0].count > 0.0);
    assert!(sim.organism_reactions[0].current_flux > 0.0);
}

#[test]
fn test_registry_degradation_reaction_returns_mass_to_nucleotide_lattice() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    sim.lattice
        .fill_species(IntracellularSpecies::Nucleotides, 0.05);
    sim.sync_from_lattice();
    sim.organism_expression.transcription_units = vec![WholeCellTranscriptionUnitState {
        name: "test_operon".to_string(),
        gene_count: 1,
        copy_gain: 1.0,
        basal_activity: 1.0,
        effective_activity: 1.0,
        support_level: 1.0,
        stress_penalty: 1.20,
        transcript_abundance: 16.0,
        protein_abundance: 10.0,
        transcript_synthesis_rate: 0.0,
        protein_synthesis_rate: 0.0,
        transcript_turnover_rate: 0.0,
        protein_turnover_rate: 0.0,
        promoter_open_fraction: 0.0,
        active_rnap_occupancy: 0.0,
        transcription_length_nt: 900.0,
        transcription_progress_nt: 0.0,
        nascent_transcript_abundance: 0.0,
        mature_transcript_abundance: 16.0,
        damaged_transcript_abundance: 0.0,
        active_ribosome_occupancy: 0.0,
        translation_length_aa: 300.0,
        translation_progress_aa: 0.0,
        nascent_protein_abundance: 0.0,
        mature_protein_abundance: 10.0,
        damaged_protein_abundance: 0.0,
        process_drive: WholeCellProcessWeights::default(),
    }];
    sim.refresh_expression_inventory_totals();
    sim.organism_species = vec![
        WholeCellSpeciesRuntimeState {
            id: "pool_nucleotides".to_string(),
            name: "nucleotides".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "cytosol".to_string(),
            asset_class: WholeCellAssetClass::Replication,
            basal_abundance: 32.0,
            bulk_field: Some(WholeCellBulkField::Nucleotides),
            spatial_scope: WholeCellSpatialScope::NucleoidLocal,
            patch_domain: WholeCellPatchDomain::NucleoidTrack,
            chromosome_domain: None,
            operon: None,
            parent_complex: None,
            subsystem_targets: Vec::new(),
            count: 3.2,
            anchor_count: 3.2,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
            process_weights: WholeCellProcessWeights::default(),
        },
        WholeCellSpeciesRuntimeState {
            id: "test_rna".to_string(),
            name: "test_rna".to_string(),
            species_class: WholeCellSpeciesClass::Rna,
            compartment: "cytosol".to_string(),
            asset_class: WholeCellAssetClass::Homeostasis,
            basal_abundance: 4.0,
            bulk_field: None,
            spatial_scope: WholeCellSpatialScope::WellMixed,
            patch_domain: WholeCellPatchDomain::Distributed,
            chromosome_domain: None,
            operon: Some("test_operon".to_string()),
            parent_complex: None,
            subsystem_targets: Vec::new(),
            count: 16.0,
            anchor_count: 16.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
            process_weights: WholeCellProcessWeights::default(),
        },
    ];
    sim.organism_reactions = vec![WholeCellReactionRuntimeState {
        id: "test_rna_degradation".to_string(),
        name: "test_rna degradation".to_string(),
        reaction_class: WholeCellReactionClass::RnaDegradation,
        asset_class: WholeCellAssetClass::Homeostasis,
        nominal_rate: 6.0,
        catalyst: None,
        operon: Some("test_operon".to_string()),
        reactants: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
            species_id: "test_rna".to_string(),
            stoichiometry: 1.0,
        }],
        products: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
            species_id: "pool_nucleotides".to_string(),
            stoichiometry: 2.5,
        }],
        subsystem_targets: Vec::new(),
        spatial_scope: WholeCellSpatialScope::NucleoidLocal,
        patch_domain: WholeCellPatchDomain::NucleoidTrack,
        chromosome_domain: None,
        current_flux: 0.0,
        cumulative_extent: 0.0,
        reactant_satisfaction: 1.0,
        catalyst_support: 1.0,
        process_weights: WholeCellProcessWeights::default(),
    }];
    let before_nucleotides = sim.lattice.mean_species(IntracellularSpecies::Nucleotides);

    sim.update_runtime_process_reactions(1.0, 0.0, 0.0);

    let after_nucleotides = sim.lattice.mean_species(IntracellularSpecies::Nucleotides);
    let rna_state = sim
        .organism_species
        .iter()
        .find(|species| species.id == "test_rna")
        .expect("RNA species");
    let unit_state = sim
        .organism_expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == "test_operon")
        .expect("unit state");
    assert!(after_nucleotides > before_nucleotides);
    assert!(rna_state.count < 16.0);
    assert!(sim.nucleotides_mm > before_nucleotides);
    assert!(unit_state.transcript_abundance < 16.0);
    assert!(sim.organism_expression.total_transcript_abundance < 16.0);
}

#[test]
fn test_registry_stress_response_reduces_metabolic_load_and_operon_stress() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    sim.metabolic_load = 2.2;
    sim.lattice.fill_species(IntracellularSpecies::ATP, 0.50);
    sim.sync_from_lattice();
    sim.organism_expression.transcription_units = vec![WholeCellTranscriptionUnitState {
        name: "stress_operon".to_string(),
        gene_count: 2,
        copy_gain: 1.0,
        basal_activity: 1.0,
        effective_activity: 1.0,
        support_level: 0.72,
        stress_penalty: 1.45,
        transcript_abundance: 12.0,
        protein_abundance: 16.0,
        transcript_synthesis_rate: 0.0,
        protein_synthesis_rate: 0.0,
        transcript_turnover_rate: 0.0,
        protein_turnover_rate: 0.0,
        promoter_open_fraction: 0.0,
        active_rnap_occupancy: 0.0,
        transcription_length_nt: 1800.0,
        transcription_progress_nt: 0.0,
        nascent_transcript_abundance: 0.0,
        mature_transcript_abundance: 12.0,
        damaged_transcript_abundance: 0.0,
        active_ribosome_occupancy: 0.0,
        translation_length_aa: 600.0,
        translation_progress_aa: 0.0,
        nascent_protein_abundance: 0.0,
        mature_protein_abundance: 16.0,
        damaged_protein_abundance: 0.0,
        process_drive: WholeCellProcessWeights::default(),
    }];
    sim.organism_species = vec![WholeCellSpeciesRuntimeState {
        id: "pool_atp".to_string(),
        name: "atp".to_string(),
        species_class: WholeCellSpeciesClass::Pool,
        compartment: "cytosol".to_string(),
        asset_class: WholeCellAssetClass::Energy,
        basal_abundance: 28.0,
        bulk_field: Some(WholeCellBulkField::ATP),
        spatial_scope: WholeCellSpatialScope::WellMixed,
        patch_domain: WholeCellPatchDomain::Distributed,
        chromosome_domain: None,
        operon: None,
        parent_complex: None,
        subsystem_targets: Vec::new(),
        count: 28.0,
        anchor_count: 28.0,
        synthesis_rate: 0.0,
        turnover_rate: 0.0,
        process_weights: WholeCellProcessWeights::default(),
    }];
    sim.organism_reactions = vec![WholeCellReactionRuntimeState {
        id: "stress_operon_stress_response".to_string(),
        name: "stress response".to_string(),
        reaction_class: WholeCellReactionClass::StressResponse,
        asset_class: WholeCellAssetClass::Homeostasis,
        nominal_rate: 8.0,
        catalyst: None,
        operon: Some("stress_operon".to_string()),
        reactants: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
            species_id: "pool_atp".to_string(),
            stoichiometry: 1.0,
        }],
        products: Vec::new(),
        subsystem_targets: Vec::new(),
        spatial_scope: WholeCellSpatialScope::WellMixed,
        patch_domain: WholeCellPatchDomain::Distributed,
        chromosome_domain: None,
        current_flux: 0.0,
        cumulative_extent: 0.0,
        reactant_satisfaction: 1.0,
        catalyst_support: 1.0,
        process_weights: WholeCellProcessWeights::default(),
    }];
    let before_load = sim.metabolic_load;
    let before_atp = sim.atp_mm;

    sim.update_runtime_process_reactions(1.0, 0.0, 0.0);

    let unit_state = sim
        .organism_expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == "stress_operon")
        .expect("stress unit");
    assert!(sim.metabolic_load < before_load);
    assert!(sim.atp_mm < before_atp);
    assert!(unit_state.stress_penalty < 1.45);
    assert!(unit_state.support_level > 0.72);
}

#[test]
fn test_registry_complex_repair_updates_named_complex_state() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    sim.lattice
        .fill_species(IntracellularSpecies::AminoAcids, 0.50);
    sim.sync_from_lattice();
    sim.named_complexes = vec![WholeCellNamedComplexState {
        id: "test_complex".to_string(),
        operon: "repair_operon".to_string(),
        asset_class: WholeCellAssetClass::Membrane,
        family: WholeCellAssemblyFamily::MembraneEnzyme,
        subsystem_targets: Vec::new(),
        subunit_pool: 10.0,
        nucleation_intermediate: 1.0,
        elongation_intermediate: 1.0,
        abundance: 2.0,
        target_abundance: 8.0,
        assembly_rate: 0.0,
        degradation_rate: 0.0,
        nucleation_rate: 0.0,
        elongation_rate: 0.0,
        maturation_rate: 0.0,
        component_satisfaction: 0.8,
        structural_support: 0.8,
        assembly_progress: 0.4,
        stalled_intermediate: 0.0,
        damaged_abundance: 0.0,
        limiting_component_signal: 0.8,
        shared_component_pressure: 0.0,
        insertion_progress: 0.6,
        failure_count: 0.0,
    }];
    sim.organism_species = vec![
        WholeCellSpeciesRuntimeState {
            id: "test_complex_subunit_pool".to_string(),
            name: "test complex subunit pool".to_string(),
            species_class: WholeCellSpeciesClass::ComplexSubunitPool,
            compartment: "cytosol".to_string(),
            asset_class: WholeCellAssetClass::Membrane,
            basal_abundance: 10.0,
            bulk_field: None,
            spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
            patch_domain: WholeCellPatchDomain::MembraneBand,
            chromosome_domain: None,
            operon: Some("repair_operon".to_string()),
            parent_complex: Some("test_complex".to_string()),
            subsystem_targets: Vec::new(),
            count: 10.0,
            anchor_count: 10.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
            process_weights: WholeCellProcessWeights::default(),
        },
        WholeCellSpeciesRuntimeState {
            id: "test_complex_mature".to_string(),
            name: "test complex".to_string(),
            species_class: WholeCellSpeciesClass::ComplexMature,
            compartment: "membrane".to_string(),
            asset_class: WholeCellAssetClass::Membrane,
            basal_abundance: 8.0,
            bulk_field: None,
            spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
            patch_domain: WholeCellPatchDomain::MembraneBand,
            chromosome_domain: None,
            operon: Some("repair_operon".to_string()),
            parent_complex: Some("test_complex".to_string()),
            subsystem_targets: Vec::new(),
            count: 2.0,
            anchor_count: 8.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
            process_weights: WholeCellProcessWeights::default(),
        },
        WholeCellSpeciesRuntimeState {
            id: "pool_amino_acids".to_string(),
            name: "amino acids".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "cytosol".to_string(),
            asset_class: WholeCellAssetClass::Translation,
            basal_abundance: 32.0,
            bulk_field: Some(WholeCellBulkField::AminoAcids),
            spatial_scope: WholeCellSpatialScope::WellMixed,
            patch_domain: WholeCellPatchDomain::Distributed,
            chromosome_domain: None,
            operon: None,
            parent_complex: None,
            subsystem_targets: Vec::new(),
            count: 32.0,
            anchor_count: 32.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
            process_weights: WholeCellProcessWeights::default(),
        },
    ];
    sim.organism_reactions = vec![WholeCellReactionRuntimeState {
        id: "test_complex_repair".to_string(),
        name: "test complex repair".to_string(),
        reaction_class: WholeCellReactionClass::ComplexRepair,
        asset_class: WholeCellAssetClass::Membrane,
        nominal_rate: 6.0,
        catalyst: None,
        operon: Some("repair_operon".to_string()),
        reactants: vec![
            crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "test_complex_subunit_pool".to_string(),
                stoichiometry: 1.0,
            },
            crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "pool_amino_acids".to_string(),
                stoichiometry: 0.5,
            },
        ],
        products: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
            species_id: "test_complex_mature".to_string(),
            stoichiometry: 1.0,
        }],
        subsystem_targets: Vec::new(),
        spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
        patch_domain: WholeCellPatchDomain::MembraneBand,
        chromosome_domain: None,
        current_flux: 0.0,
        cumulative_extent: 0.0,
        reactant_satisfaction: 1.0,
        catalyst_support: 1.0,
        process_weights: WholeCellProcessWeights::default(),
    }];

    sim.update_runtime_process_reactions(1.0, 0.0, 0.0);

    let complex_state = sim
        .named_complexes
        .iter()
        .find(|state| state.id == "test_complex")
        .expect("complex state");
    assert!(complex_state.abundance > 2.0);
    assert!(complex_state.subunit_pool < 10.0);
}

#[test]
fn test_atp_hotspot_diffuses_on_cpu() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        x_dim: 8,
        y_dim: 8,
        z_dim: 4,
        use_gpu: false,
        dt_ms: 0.25,
        ..WholeCellConfig::default()
    });
    let center = (4, 4, 2);
    let neighbor_idx = center.2 * 8 * 8 + center.1 * 8 + (center.0 + 1);
    let center_idx = center.2 * 8 * 8 + center.1 * 8 + center.0;

    sim.add_hotspot(IntracellularSpecies::ATP, center.0, center.1, center.2, 4.0);
    let before = sim.atp_lattice();

    sim.step();

    let after = sim.atp_lattice();
    assert!(after[center_idx] < before[center_idx]);
    assert!(after[neighbor_idx] > before[neighbor_idx]);
}

#[test]
fn test_initial_lattice_has_no_seeded_hotspots() {
    let sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    let atp = sim.atp_lattice();
    let first = atp.first().copied().expect("atp lattice");
    assert!(atp.iter().all(|value| (*value - first).abs() < 1.0e-6));
}

#[test]
fn test_spatial_fields_round_trip_through_saved_state() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    sim.run(4);
    let saved = sim.save_state_json().expect("saved state json");
    let restored = WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");

    let original_membrane = sim
        .spatial_fields
        .clone_field(IntracellularSpatialField::MembraneAdjacency);
    let restored_membrane = restored
        .spatial_fields
        .clone_field(IntracellularSpatialField::MembraneAdjacency);
    let original_nucleoid = sim
        .spatial_fields
        .clone_field(IntracellularSpatialField::NucleoidOccupancy);
    let restored_nucleoid = restored
        .spatial_fields
        .clone_field(IntracellularSpatialField::NucleoidOccupancy);
    let original_band = sim
        .spatial_fields
        .clone_field(IntracellularSpatialField::MembraneBandZone);
    let restored_band = restored
        .spatial_fields
        .clone_field(IntracellularSpatialField::MembraneBandZone);
    let original_poles = sim
        .spatial_fields
        .clone_field(IntracellularSpatialField::PoleZone);
    let restored_poles = restored
        .spatial_fields
        .clone_field(IntracellularSpatialField::PoleZone);

    assert_eq!(original_membrane.len(), restored_membrane.len());
    assert_eq!(original_nucleoid.len(), restored_nucleoid.len());
    assert_eq!(original_band.len(), restored_band.len());
    assert_eq!(original_poles.len(), restored_poles.len());
    assert!(original_membrane
        .iter()
        .zip(restored_membrane.iter())
        .all(|(lhs, rhs)| (lhs - rhs).abs() < 1.0e-6));
    assert!(original_nucleoid
        .iter()
        .zip(restored_nucleoid.iter())
        .all(|(lhs, rhs)| (lhs - rhs).abs() < 1.0e-6));
    assert!(original_band
        .iter()
        .zip(restored_band.iter())
        .all(|(lhs, rhs)| (lhs - rhs).abs() < 1.0e-6));
    assert!(original_poles
        .iter()
        .zip(restored_poles.iter())
        .all(|(lhs, rhs)| (lhs - rhs).abs() < 1.0e-6));
}

#[test]
fn test_nucleoid_localization_biases_nucleotide_signal() {
    let mut nucleoid_loaded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let mut pole_loaded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let nucleoid_weights = nucleoid_loaded
        .spatial_fields
        .clone_field(IntracellularSpatialField::NucleoidOccupancy);
    let pole_weights: Vec<f32> = nucleoid_weights
        .iter()
        .map(|weight| (1.0 - *weight).max(0.02))
        .collect();
    let nucleoid_values = distribute_total_by_weights(&nucleoid_weights, 0.80);
    let pole_values = distribute_total_by_weights(&pole_weights, 0.80);

    nucleoid_loaded
        .lattice
        .set_species(IntracellularSpecies::Nucleotides, &nucleoid_values)
        .expect("nucleoid-targeted nucleotides");
    pole_loaded
        .lattice
        .set_species(IntracellularSpecies::Nucleotides, &pole_values)
        .expect("pole-targeted nucleotides");
    nucleoid_loaded.sync_from_lattice();
    pole_loaded.sync_from_lattice();

    let nucleoid_ctx = nucleoid_loaded.base_rule_context(0.0);
    let pole_ctx = pole_loaded.base_rule_context(0.0);

    assert!((nucleoid_loaded.nucleotides_mm - pole_loaded.nucleotides_mm).abs() < 1.0e-6);
    assert!(
        nucleoid_loaded.localized_nucleotide_pool_mm() > pole_loaded.localized_nucleotide_pool_mm()
    );
    assert!(
        nucleoid_ctx.get(WholeCellRuleSignal::NucleotideSignal)
            > pole_ctx.get(WholeCellRuleSignal::NucleotideSignal)
    );
}

#[test]
fn test_nucleoid_atp_localization_biases_replication_unit_support() {
    let mut nucleoid_loaded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let mut pole_loaded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let nucleoid_weights = nucleoid_loaded
        .spatial_fields
        .clone_field(IntracellularSpatialField::NucleoidOccupancy);
    let pole_weights: Vec<f32> = nucleoid_weights
        .iter()
        .map(|weight| (1.0 - *weight).max(0.02))
        .collect();
    let nucleoid_values = distribute_total_by_weights(&nucleoid_weights, 1.10);
    let pole_values = distribute_total_by_weights(&pole_weights, 1.10);

    nucleoid_loaded
        .lattice
        .set_species(IntracellularSpecies::ATP, &nucleoid_values)
        .expect("nucleoid-targeted ATP");
    pole_loaded
        .lattice
        .set_species(IntracellularSpecies::ATP, &pole_values)
        .expect("pole-targeted ATP");
    nucleoid_loaded.sync_from_lattice();
    pole_loaded.sync_from_lattice();
    nucleoid_loaded.refresh_organism_expression_state();
    pole_loaded.refresh_organism_expression_state();

    let nucleoid_expression = nucleoid_loaded
        .organism_expression_state()
        .expect("nucleoid expression");
    let pole_expression = pole_loaded
        .organism_expression_state()
        .expect("pole expression");
    let nucleoid_unit = nucleoid_expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == "replication_cycle_operon")
        .expect("nucleoid replication unit");
    let pole_unit = pole_expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == "replication_cycle_operon")
        .expect("pole replication unit");

    assert!(
        nucleoid_loaded.localized_nucleoid_atp_pool_mm()
            > pole_loaded.localized_nucleoid_atp_pool_mm()
    );
    assert!(nucleoid_unit.support_level > pole_unit.support_level);
    assert!(nucleoid_unit.effective_activity > pole_unit.effective_activity);
}

#[test]
fn test_chromosome_domain_support_tracks_domain_loaded_pools() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let domain_zero_weights = sim.chromosome_domain_weights(0);
    let domain_three_weights = sim.chromosome_domain_weights(3);
    let domain_zero_atp = distribute_total_by_weights(&domain_zero_weights, 1.10);
    let domain_zero_nucleotides = distribute_total_by_weights(&domain_zero_weights, 0.80);

    sim.lattice
        .set_species(IntracellularSpecies::ATP, &domain_zero_atp)
        .expect("domain-zero ATP");
    sim.lattice
        .set_species(IntracellularSpecies::Nucleotides, &domain_zero_nucleotides)
        .expect("domain-zero nucleotides");
    sim.sync_from_lattice();

    assert!(
        sim.weighted_species_mean(IntracellularSpecies::ATP, &domain_zero_weights)
            > sim.weighted_species_mean(IntracellularSpecies::ATP, &domain_three_weights)
    );
    assert!(sim.chromosome_domain_energy_support(0) > sim.chromosome_domain_energy_support(3));
    assert!(
        sim.chromosome_domain_nucleotide_support(0) > sim.chromosome_domain_nucleotide_support(3)
    );
}

#[test]
fn test_compiled_chromosome_domain_centers_bias_weight_peaks() {
    let sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let organism = sim.organism_data.as_ref().expect("bundled organism data");
    assert!(organism.chromosome_domains.len() >= 4);

    let weighted_x_center = |weights: &[f32], x_dim: usize| -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        for (gid, weight) in weights.iter().enumerate() {
            let x = gid % x_dim;
            weighted_sum += *weight * (x as f32 + 0.5) / x_dim as f32;
            total_weight += *weight;
        }
        if total_weight <= 1.0e-6 {
            0.5
        } else {
            weighted_sum / total_weight
        }
    };

    let domain_zero = organism
        .chromosome_domains
        .first()
        .expect("first chromosome domain");
    let domain_last = organism
        .chromosome_domains
        .last()
        .expect("last chromosome domain");
    let domain_zero_center =
        weighted_x_center(&sim.chromosome_domain_weights(0), sim.lattice.x_dim.max(1));
    let domain_last_center = weighted_x_center(
        &sim.chromosome_domain_weights((organism.chromosome_domains.len() - 1) as u32),
        sim.lattice.x_dim.max(1),
    );

    assert!(domain_zero_center < domain_last_center);
    assert!(
        (domain_zero_center - domain_zero.axial_center_fraction).abs()
            < (domain_zero_center - domain_last.axial_center_fraction).abs()
    );
    assert!(
        (domain_last_center - domain_last.axial_center_fraction).abs()
            < (domain_last_center - domain_zero.axial_center_fraction).abs()
    );
}

#[test]
fn test_registry_chromosome_domains_bias_weight_peaks_without_assets() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig::default());
    sim.organism_data = None;
    sim.organism_assets = None;
    sim.organism_process_registry = Some(WholeCellGenomeProcessRegistry {
        organism: "registry-domain-test".to_string(),
        chromosome_domains: vec![
            WholeCellChromosomeDomainSpec {
                id: "domain_left".to_string(),
                start_bp: 0,
                end_bp: 399,
                axial_center_fraction: 0.22,
                axial_spread_fraction: 0.14,
                genes: Vec::new(),
                transcription_units: Vec::new(),
                operons: Vec::new(),
            },
            WholeCellChromosomeDomainSpec {
                id: "domain_right".to_string(),
                start_bp: 400,
                end_bp: 799,
                axial_center_fraction: 0.78,
                axial_spread_fraction: 0.14,
                genes: Vec::new(),
                transcription_units: Vec::new(),
                operons: Vec::new(),
            },
        ],
        species: Vec::new(),
        reactions: Vec::new(),
    });

    let weighted_x_center = |weights: &[f32], x_dim: usize| -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        for (gid, weight) in weights.iter().enumerate() {
            let x = gid % x_dim;
            weighted_sum += *weight * (x as f32 + 0.5) / x_dim as f32;
            total_weight += *weight;
        }
        if total_weight <= 1.0e-6 {
            0.5
        } else {
            weighted_sum / total_weight
        }
    };

    let left_center =
        weighted_x_center(&sim.chromosome_domain_weights(0), sim.lattice.x_dim.max(1));
    let right_center =
        weighted_x_center(&sim.chromosome_domain_weights(1), sim.lattice.x_dim.max(1));

    assert_eq!(sim.chromosome_domain_count(), 2);
    assert_eq!(sim.chromosome_domain_index_by_id("domain_left"), Some(0));
    assert_eq!(sim.chromosome_domain_index_by_id("domain_right"), Some(1));
    assert_eq!(sim.chromosome_domain_index(100, 800), 0);
    assert_eq!(sim.chromosome_domain_index(700, 800), 1);
    assert!(left_center < right_center);
    assert!((left_center - 0.22).abs() < (left_center - 0.78).abs());
    assert!((right_center - 0.78).abs() < (right_center - 0.22).abs());
}

#[test]
fn test_chromosome_domain_loading_biases_expression_by_midpoint() {
    let mut left_loaded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let mut right_loaded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");

    let domain_zero_weights = left_loaded.chromosome_domain_weights(0);
    let domain_three_weights = left_loaded.chromosome_domain_weights(3);
    let left_atp = distribute_total_by_weights(&domain_zero_weights, 1.10);
    let left_nucleotides = distribute_total_by_weights(&domain_zero_weights, 0.80);
    let right_atp = distribute_total_by_weights(&domain_three_weights, 1.10);
    let right_nucleotides = distribute_total_by_weights(&domain_three_weights, 0.80);

    left_loaded
        .lattice
        .set_species(IntracellularSpecies::ATP, &left_atp)
        .expect("left-domain ATP");
    left_loaded
        .lattice
        .set_species(IntracellularSpecies::Nucleotides, &left_nucleotides)
        .expect("left-domain nucleotides");
    right_loaded
        .lattice
        .set_species(IntracellularSpecies::ATP, &right_atp)
        .expect("right-domain ATP");
    right_loaded
        .lattice
        .set_species(IntracellularSpecies::Nucleotides, &right_nucleotides)
        .expect("right-domain nucleotides");
    left_loaded.sync_from_lattice();
    right_loaded.sync_from_lattice();
    left_loaded.refresh_organism_expression_state();
    right_loaded.refresh_organism_expression_state();

    let organism = left_loaded
        .organism_data
        .as_ref()
        .expect("bundled organism data");
    let mut unit_midpoints: Vec<(String, u32)> = organism
        .transcription_units
        .iter()
        .map(|unit| {
            let mut midpoint_sum = 0.0;
            let mut midpoint_count = 0usize;
            for gene_name in &unit.genes {
                if let Some(feature) = organism
                    .genes
                    .iter()
                    .find(|feature| feature.gene == *gene_name)
                {
                    midpoint_sum += 0.5 * (feature.start_bp as f32 + feature.end_bp as f32);
                    midpoint_count += 1;
                }
            }
            let midpoint_bp = if midpoint_count > 0 {
                (midpoint_sum / midpoint_count as f32).round() as u32
            } else {
                organism.origin_bp.min(organism.chromosome_length_bp.max(1))
            };
            (unit.name.clone(), midpoint_bp)
        })
        .collect();
    unit_midpoints.sort_by_key(|(_, midpoint_bp)| *midpoint_bp);
    let low_name = unit_midpoints
        .first()
        .map(|(name, _)| name.clone())
        .expect("low-midpoint unit");
    let high_name = unit_midpoints
        .last()
        .map(|(name, _)| name.clone())
        .expect("high-midpoint unit");

    let left_expression = left_loaded
        .organism_expression_state()
        .expect("left-domain expression");
    let right_expression = right_loaded
        .organism_expression_state()
        .expect("right-domain expression");
    let left_low = left_expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == low_name)
        .expect("left low-midpoint unit");
    let right_low = right_expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == low_name)
        .expect("right low-midpoint unit");
    let left_high = left_expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == high_name)
        .expect("left high-midpoint unit");
    let right_high = right_expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == high_name)
        .expect("right high-midpoint unit");

    assert!(left_low.support_level > right_low.support_level);
    assert!(right_high.support_level > left_high.support_level);
}

#[test]
fn test_nucleoid_localization_biases_chromosome_progress() {
    let mut nucleoid_loaded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let mut pole_loaded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    for sim in [&mut nucleoid_loaded, &mut pole_loaded] {
        sim.dnaa = 72.0;
        sim.complex_assembly.replisome_complexes = 12.0;
        sim.chemistry_report.atp_support = 0.20;
        sim.chemistry_report.nucleotide_support = 0.15;
        sim.chromosome_state.replicated_bp = 0;
        sim.chromosome_state.forks.clear();
        sim.chromosome_state.initiation_potential = 0.30;
        sim.chromosome_state = sim.normalize_chromosome_state(sim.chromosome_state.clone());
        sim.synchronize_chromosome_summary();
    }

    let nucleoid_weights = nucleoid_loaded
        .spatial_fields
        .clone_field(IntracellularSpatialField::NucleoidOccupancy);
    let pole_weights: Vec<f32> = nucleoid_weights
        .iter()
        .map(|weight| (1.0 - *weight).max(0.02))
        .collect();
    let nucleoid_atp = distribute_total_by_weights(&nucleoid_weights, 1.10);
    let pole_atp = distribute_total_by_weights(&pole_weights, 1.10);
    let nucleoid_nucleotides = distribute_total_by_weights(&nucleoid_weights, 0.80);
    let pole_nucleotides = distribute_total_by_weights(&pole_weights, 0.80);

    nucleoid_loaded
        .lattice
        .set_species(IntracellularSpecies::ATP, &nucleoid_atp)
        .expect("nucleoid-targeted ATP");
    pole_loaded
        .lattice
        .set_species(IntracellularSpecies::ATP, &pole_atp)
        .expect("pole-targeted ATP");
    nucleoid_loaded
        .lattice
        .set_species(IntracellularSpecies::Nucleotides, &nucleoid_nucleotides)
        .expect("nucleoid-targeted nucleotides");
    pole_loaded
        .lattice
        .set_species(IntracellularSpecies::Nucleotides, &pole_nucleotides)
        .expect("pole-targeted nucleotides");
    nucleoid_loaded.sync_from_lattice();
    pole_loaded.sync_from_lattice();

    nucleoid_loaded.refresh_organism_expression_state();
    pole_loaded.refresh_organism_expression_state();
    nucleoid_loaded.advance_chromosome_state(nucleoid_loaded.config.dt_ms, 3.5, 0.5);
    pole_loaded.advance_chromosome_state(pole_loaded.config.dt_ms, 3.5, 0.5);

    let nucleoid_state = nucleoid_loaded.chromosome_state();
    let pole_state = pole_loaded.chromosome_state();

    assert!(
        nucleoid_loaded.chromosome_domain_energy_support(0)
            > pole_loaded.chromosome_domain_energy_support(0)
    );
    assert!(
        nucleoid_loaded.chromosome_domain_nucleotide_support(0)
            > pole_loaded.chromosome_domain_nucleotide_support(0)
    );
    assert!(
        nucleoid_state.initiation_potential > pole_state.initiation_potential,
        "initiation nucleoid={} pole={}",
        nucleoid_state.initiation_potential,
        pole_state.initiation_potential
    );
    assert!(nucleoid_state.initiation_events >= pole_state.initiation_events);
    assert!(nucleoid_state.forks.len() >= pole_state.forks.len());
}

#[test]
fn test_septum_localization_biases_membrane_constriction_support() {
    let mut septum_loaded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let mut pole_loaded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let septum_weights = septum_loaded
        .spatial_fields
        .clone_field(IntracellularSpatialField::SeptumZone);
    let pole_weights: Vec<f32> = septum_loaded
        .spatial_fields
        .clone_field(IntracellularSpatialField::MembraneAdjacency)
        .into_iter()
        .zip(septum_weights.iter().copied())
        .map(|(membrane, septum)| (membrane - septum).max(0.02))
        .collect();
    let septum_values = distribute_total_by_weights(&septum_weights, 0.35);
    let pole_values = distribute_total_by_weights(&pole_weights, 0.35);

    septum_loaded
        .lattice
        .set_species(IntracellularSpecies::MembranePrecursors, &septum_values)
        .expect("septum-targeted membrane precursors");
    pole_loaded
        .lattice
        .set_species(IntracellularSpecies::MembranePrecursors, &pole_values)
        .expect("pole-targeted membrane precursors");
    septum_loaded.sync_from_lattice();
    pole_loaded.sync_from_lattice();
    assert!(
        (septum_loaded.membrane_precursors_mm - pole_loaded.membrane_precursors_mm).abs() < 1.0e-4
    );
    septum_loaded.refresh_organism_expression_state();
    pole_loaded.refresh_organism_expression_state();

    for _ in 0..8 {
        septum_loaded.geometry_stage(septum_loaded.config.dt_ms);
        pole_loaded.geometry_stage(pole_loaded.config.dt_ms);
    }

    assert!(
        septum_loaded.localized_membrane_precursor_pool_mm()
            > pole_loaded.localized_membrane_precursor_pool_mm()
    );
    assert!(
        septum_loaded.membrane_division_state.constriction_force
            > pole_loaded.membrane_division_state.constriction_force
    );
    assert!(
        septum_loaded
            .membrane_division_state
            .septal_lipid_inventory_nm2
            > pole_loaded
                .membrane_division_state
                .septal_lipid_inventory_nm2
    );
}

#[test]
fn test_membrane_band_zone_exceeds_poles_for_band_localized_drive() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    sim.organism_species = vec![WholeCellSpeciesRuntimeState {
        id: "atp_band_complex_mature".to_string(),
        name: "atp band complex".to_string(),
        species_class: WholeCellSpeciesClass::ComplexMature,
        compartment: "membrane".to_string(),
        asset_class: WholeCellAssetClass::Energy,
        basal_abundance: 8.0,
        bulk_field: None,
        spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
        patch_domain: WholeCellPatchDomain::MembraneBand,
        chromosome_domain: None,
        operon: Some("energy_operon".to_string()),
        parent_complex: Some("atp_band_complex".to_string()),
        subsystem_targets: vec![Syn3ASubsystemPreset::AtpSynthaseMembraneBand],
        count: 8.0,
        anchor_count: 8.0,
        synthesis_rate: 0.0,
        turnover_rate: 0.0,
        process_weights: WholeCellProcessWeights::default(),
    }];
    sim.organism_reactions.clear();
    sim.refresh_spatial_fields();
    sim.refresh_rdme_drive_fields();

    let band_energy = sim.localized_drive_mean(
        WholeCellRdmeDriveField::EnergySource,
        IntracellularSpatialField::MembraneBandZone,
    );
    let pole_energy = sim.localized_drive_mean(
        WholeCellRdmeDriveField::EnergySource,
        IntracellularSpatialField::PoleZone,
    );

    assert!(band_energy > pole_energy);
}

#[test]
fn test_localized_pool_transfer_moves_membrane_precursors_into_band_zone() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    sim.lattice
        .fill_species(IntracellularSpecies::MembranePrecursors, 0.22);
    sim.sync_from_lattice();
    sim.organism_species = vec![
        WholeCellSpeciesRuntimeState {
            id: "pool_membrane_precursors".to_string(),
            name: "membrane precursors".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "cytosol".to_string(),
            asset_class: WholeCellAssetClass::Membrane,
            basal_abundance: 48.0,
            bulk_field: Some(WholeCellBulkField::MembranePrecursors),
            spatial_scope: WholeCellSpatialScope::WellMixed,
            patch_domain: WholeCellPatchDomain::Distributed,
            chromosome_domain: None,
            operon: None,
            parent_complex: None,
            subsystem_targets: Vec::new(),
            count: 48.0,
            anchor_count: 48.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
            process_weights: WholeCellProcessWeights::default(),
        },
        WholeCellSpeciesRuntimeState {
            id: "pool_membrane_band_membrane_precursors".to_string(),
            name: "membrane band precursors".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "membrane".to_string(),
            asset_class: WholeCellAssetClass::Membrane,
            basal_abundance: 8.0,
            bulk_field: Some(WholeCellBulkField::MembranePrecursors),
            spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
            patch_domain: WholeCellPatchDomain::MembraneBand,
            chromosome_domain: None,
            operon: None,
            parent_complex: None,
            subsystem_targets: vec![Syn3ASubsystemPreset::AtpSynthaseMembraneBand],
            count: 2.0,
            anchor_count: 2.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
            process_weights: WholeCellProcessWeights::default(),
        },
    ];
    sim.organism_reactions = vec![WholeCellReactionRuntimeState {
        id: "pool_membrane_band_membrane_precursors_localized_transfer".to_string(),
        name: "membrane band precursors localized transfer".to_string(),
        reaction_class: WholeCellReactionClass::LocalizedPoolTransfer,
        asset_class: WholeCellAssetClass::Membrane,
        nominal_rate: 9.0,
        catalyst: None,
        operon: None,
        reactants: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
            species_id: "pool_membrane_precursors".to_string(),
            stoichiometry: 1.0,
        }],
        products: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
            species_id: "pool_membrane_band_membrane_precursors".to_string(),
            stoichiometry: 1.0,
        }],
        subsystem_targets: vec![Syn3ASubsystemPreset::AtpSynthaseMembraneBand],
        spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
        patch_domain: WholeCellPatchDomain::MembraneBand,
        chromosome_domain: None,
        current_flux: 0.0,
        cumulative_extent: 0.0,
        reactant_satisfaction: 1.0,
        catalyst_support: 1.0,
        process_weights: WholeCellProcessWeights::default(),
    }];

    let before_band = sim.localized_membrane_band_precursor_pool_mm();

    sim.update_runtime_process_reactions(1.0, 0.0, 0.0);

    let after_band = sim.localized_membrane_band_precursor_pool_mm();
    let after_pole = sim.localized_polar_precursor_pool_mm();
    assert!(after_band > before_band);
    assert!(after_band > after_pole);
    assert!(sim.organism_reactions[0].current_flux > 0.0);
}

#[test]
fn test_localized_pool_transfer_moves_nucleotides_into_nucleoid_track() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    sim.lattice
        .fill_species(IntracellularSpecies::Nucleotides, 0.18);
    sim.sync_from_lattice();
    sim.organism_species = vec![
        WholeCellSpeciesRuntimeState {
            id: "pool_nucleotides".to_string(),
            name: "nucleotides".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "cytosol".to_string(),
            asset_class: WholeCellAssetClass::Replication,
            basal_abundance: 42.0,
            bulk_field: Some(WholeCellBulkField::Nucleotides),
            spatial_scope: WholeCellSpatialScope::WellMixed,
            patch_domain: WholeCellPatchDomain::Distributed,
            chromosome_domain: None,
            operon: None,
            parent_complex: None,
            subsystem_targets: Vec::new(),
            count: 42.0,
            anchor_count: 42.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
            process_weights: WholeCellProcessWeights::default(),
        },
        WholeCellSpeciesRuntimeState {
            id: "pool_nucleoid_track_nucleotides".to_string(),
            name: "nucleoid track nucleotides pool".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "chromosome".to_string(),
            asset_class: WholeCellAssetClass::Replication,
            basal_abundance: 10.0,
            bulk_field: Some(WholeCellBulkField::Nucleotides),
            spatial_scope: WholeCellSpatialScope::NucleoidLocal,
            patch_domain: WholeCellPatchDomain::NucleoidTrack,
            chromosome_domain: None,
            operon: None,
            parent_complex: None,
            subsystem_targets: vec![Syn3ASubsystemPreset::ReplisomeTrack],
            count: 3.0,
            anchor_count: 3.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
            process_weights: WholeCellProcessWeights::default(),
        },
    ];
    sim.organism_reactions = vec![WholeCellReactionRuntimeState {
        id: "pool_nucleoid_track_nucleotides_localized_transfer".to_string(),
        name: "nucleoid track nucleotides localized transfer".to_string(),
        reaction_class: WholeCellReactionClass::LocalizedPoolTransfer,
        asset_class: WholeCellAssetClass::Replication,
        nominal_rate: 8.5,
        catalyst: None,
        operon: None,
        reactants: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
            species_id: "pool_nucleotides".to_string(),
            stoichiometry: 1.0,
        }],
        products: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
            species_id: "pool_nucleoid_track_nucleotides".to_string(),
            stoichiometry: 1.0,
        }],
        subsystem_targets: vec![Syn3ASubsystemPreset::ReplisomeTrack],
        spatial_scope: WholeCellSpatialScope::NucleoidLocal,
        patch_domain: WholeCellPatchDomain::NucleoidTrack,
        chromosome_domain: None,
        current_flux: 0.0,
        cumulative_extent: 0.0,
        reactant_satisfaction: 1.0,
        catalyst_support: 1.0,
        process_weights: WholeCellProcessWeights::default(),
    }];
    let mut nucleotide_demand = vec![0.0; sim.lattice.total_voxels()];
    let nucleoid_weights = sim
        .spatial_fields
        .clone_field(IntracellularSpatialField::NucleoidOccupancy);
    for (value, weight) in nucleotide_demand.iter_mut().zip(nucleoid_weights.iter()) {
        *value = 1.6 * weight.max(0.0);
    }
    let _ = sim.rdme_drive_fields.set_field(
        WholeCellRdmeDriveField::NucleotideDemand,
        &nucleotide_demand,
    );

    let before_nucleoid = sim.localized_nucleotide_pool_mm();

    sim.update_runtime_process_reactions(1.0, 0.0, 0.0);

    let after_nucleoid = sim.localized_nucleotide_pool_mm();
    let membrane_nucleotides = sim.spatial_species_mean(
        IntracellularSpecies::Nucleotides,
        IntracellularSpatialField::MembraneAdjacency,
    );
    assert!(after_nucleoid > before_nucleoid);
    assert!(after_nucleoid > membrane_nucleotides);
    assert!(sim.organism_reactions[0].current_flux > 0.0);
}

#[test]
fn test_localized_pool_transfer_preserves_compiled_chromosome_domain() {
    let mut left_sim =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let mut right_sim =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let organism = left_sim
        .organism_data
        .as_ref()
        .expect("bundled organism data");
    let left_domain = organism
        .chromosome_domains
        .first()
        .expect("first chromosome domain")
        .id
        .clone();
    let right_domain = organism
        .chromosome_domains
        .last()
        .expect("last chromosome domain")
        .id
        .clone();

    for (sim, domain_id) in [(&mut left_sim, left_domain), (&mut right_sim, right_domain)] {
        sim.lattice
            .fill_species(IntracellularSpecies::Nucleotides, 0.18);
        sim.sync_from_lattice();
        sim.organism_species = vec![
            WholeCellSpeciesRuntimeState {
                id: "pool_nucleotides".to_string(),
                name: "nucleotides".to_string(),
                species_class: WholeCellSpeciesClass::Pool,
                compartment: "cytosol".to_string(),
                asset_class: WholeCellAssetClass::Replication,
                basal_abundance: 42.0,
                bulk_field: Some(WholeCellBulkField::Nucleotides),
                spatial_scope: WholeCellSpatialScope::WellMixed,
                patch_domain: WholeCellPatchDomain::Distributed,
                chromosome_domain: None,
                operon: None,
                parent_complex: None,
                subsystem_targets: Vec::new(),
                count: 42.0,
                anchor_count: 42.0,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
                process_weights: WholeCellProcessWeights::default(),
            },
            WholeCellSpeciesRuntimeState {
                id: "pool_nucleoid_track_nucleotides".to_string(),
                name: "nucleoid track nucleotides pool".to_string(),
                species_class: WholeCellSpeciesClass::Pool,
                compartment: "chromosome".to_string(),
                asset_class: WholeCellAssetClass::Replication,
                basal_abundance: 10.0,
                bulk_field: Some(WholeCellBulkField::Nucleotides),
                spatial_scope: WholeCellSpatialScope::NucleoidLocal,
                patch_domain: WholeCellPatchDomain::NucleoidTrack,
                chromosome_domain: None,
                operon: None,
                parent_complex: None,
                subsystem_targets: vec![Syn3ASubsystemPreset::ReplisomeTrack],
                count: 3.0,
                anchor_count: 3.0,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
                process_weights: WholeCellProcessWeights::default(),
            },
        ];
        sim.organism_reactions = vec![WholeCellReactionRuntimeState {
            id: "pool_nucleoid_track_nucleotides_localized_transfer".to_string(),
            name: "nucleoid track nucleotides localized transfer".to_string(),
            reaction_class: WholeCellReactionClass::LocalizedPoolTransfer,
            asset_class: WholeCellAssetClass::Replication,
            nominal_rate: 8.5,
            catalyst: None,
            operon: Some("replication_cycle_operon".to_string()),
            reactants: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "pool_nucleotides".to_string(),
                stoichiometry: 1.0,
            }],
            products: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "pool_nucleoid_track_nucleotides".to_string(),
                stoichiometry: 1.0,
            }],
            subsystem_targets: vec![Syn3ASubsystemPreset::ReplisomeTrack],
            spatial_scope: WholeCellSpatialScope::NucleoidLocal,
            patch_domain: WholeCellPatchDomain::NucleoidTrack,
            chromosome_domain: Some(domain_id),
            current_flux: 0.0,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
            process_weights: WholeCellProcessWeights::default(),
        }];
        sim.refresh_spatial_fields();
        sim.refresh_rdme_drive_fields();
        sim.update_runtime_process_reactions(1.0, 0.0, 0.0);
    }

    let left_domain_zero =
        left_sim.chromosome_domain_species_mean(IntracellularSpecies::Nucleotides, 0);
    let left_domain_last = left_sim.chromosome_domain_species_mean(
        IntracellularSpecies::Nucleotides,
        (left_sim.chromosome_domain_count().saturating_sub(1)) as u32,
    );
    let right_domain_zero =
        right_sim.chromosome_domain_species_mean(IntracellularSpecies::Nucleotides, 0);
    let right_domain_last = right_sim.chromosome_domain_species_mean(
        IntracellularSpecies::Nucleotides,
        (right_sim.chromosome_domain_count().saturating_sub(1)) as u32,
    );

    assert!(left_sim.organism_reactions[0].current_flux > 0.0);
    assert!(right_sim.organism_reactions[0].current_flux > 0.0);
    assert!(left_domain_zero > left_domain_last);
    assert!(right_domain_last > right_domain_zero);
}

#[test]
fn test_rdme_drive_fields_follow_compiled_scopes() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    sim.organism_species = vec![
        WholeCellSpeciesRuntimeState {
            id: "pool_nucleotides".to_string(),
            name: "nucleotides".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "cytosol".to_string(),
            asset_class: WholeCellAssetClass::Replication,
            basal_abundance: 32.0,
            bulk_field: Some(WholeCellBulkField::Nucleotides),
            spatial_scope: WholeCellSpatialScope::NucleoidLocal,
            patch_domain: WholeCellPatchDomain::NucleoidTrack,
            chromosome_domain: None,
            operon: None,
            parent_complex: None,
            subsystem_targets: Vec::new(),
            count: 24.0,
            anchor_count: 24.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
            process_weights: WholeCellProcessWeights::default(),
        },
        WholeCellSpeciesRuntimeState {
            id: "atp_band_complex_mature".to_string(),
            name: "atp band complex".to_string(),
            species_class: WholeCellSpeciesClass::ComplexMature,
            compartment: "membrane".to_string(),
            asset_class: WholeCellAssetClass::Energy,
            basal_abundance: 6.0,
            bulk_field: None,
            spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
            patch_domain: WholeCellPatchDomain::MembraneBand,
            chromosome_domain: None,
            operon: Some("energy_operon".to_string()),
            parent_complex: Some("atp_band_complex".to_string()),
            subsystem_targets: vec![Syn3ASubsystemPreset::AtpSynthaseMembraneBand],
            count: 6.0,
            anchor_count: 6.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
            process_weights: WholeCellProcessWeights::default(),
        },
    ];
    sim.organism_reactions = vec![WholeCellReactionRuntimeState {
        id: "replication_drive".to_string(),
        name: "replication drive".to_string(),
        reaction_class: WholeCellReactionClass::Transcription,
        asset_class: WholeCellAssetClass::Replication,
        nominal_rate: 9.0,
        catalyst: None,
        operon: Some("replication_operon".to_string()),
        reactants: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
            species_id: "pool_nucleotides".to_string(),
            stoichiometry: 1.6,
        }],
        products: Vec::new(),
        subsystem_targets: vec![Syn3ASubsystemPreset::ReplisomeTrack],
        spatial_scope: WholeCellSpatialScope::NucleoidLocal,
        patch_domain: WholeCellPatchDomain::NucleoidTrack,
        chromosome_domain: None,
        current_flux: 6.0,
        cumulative_extent: 0.0,
        reactant_satisfaction: 1.0,
        catalyst_support: 1.0,
        process_weights: WholeCellProcessWeights::default(),
    }];

    sim.refresh_spatial_fields();
    sim.refresh_rdme_drive_fields();

    let membrane_weights = sim
        .spatial_fields
        .clone_field(IntracellularSpatialField::MembraneAdjacency);
    let nucleoid_weights = sim
        .spatial_fields
        .clone_field(IntracellularSpatialField::NucleoidOccupancy);

    let membrane_energy = sim
        .rdme_drive_fields
        .weighted_mean(WholeCellRdmeDriveField::EnergySource, &membrane_weights);
    let nucleoid_energy = sim
        .rdme_drive_fields
        .weighted_mean(WholeCellRdmeDriveField::EnergySource, &nucleoid_weights);
    let nucleoid_demand = sim
        .rdme_drive_fields
        .weighted_mean(WholeCellRdmeDriveField::NucleotideDemand, &nucleoid_weights);
    let membrane_demand = sim
        .rdme_drive_fields
        .weighted_mean(WholeCellRdmeDriveField::NucleotideDemand, &membrane_weights);
    let band_weights = sim
        .spatial_fields
        .clone_field(IntracellularSpatialField::MembraneBandZone);
    let pole_weights = sim
        .spatial_fields
        .clone_field(IntracellularSpatialField::PoleZone);
    let band_energy = sim
        .rdme_drive_fields
        .weighted_mean(WholeCellRdmeDriveField::EnergySource, &band_weights);
    let pole_energy = sim
        .rdme_drive_fields
        .weighted_mean(WholeCellRdmeDriveField::EnergySource, &pole_weights);

    assert!(membrane_energy > nucleoid_energy);
    assert!(nucleoid_demand > membrane_demand);
    assert!(band_energy > pole_energy);
}

#[test]
fn test_rdme_drive_fields_follow_compiled_chromosome_domains() {
    let mut left_sim =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let mut right_sim =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let organism = left_sim
        .organism_data
        .as_ref()
        .expect("bundled organism data");
    let left_domain = organism
        .chromosome_domains
        .first()
        .expect("first chromosome domain")
        .id
        .clone();
    let right_domain = organism
        .chromosome_domains
        .last()
        .expect("last chromosome domain")
        .id
        .clone();

    for (sim, domain_id) in [(&mut left_sim, left_domain), (&mut right_sim, right_domain)] {
        sim.organism_species = vec![WholeCellSpeciesRuntimeState {
            id: "pool_nucleoid_track_nucleotides".to_string(),
            name: "nucleoid track nucleotides pool".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "chromosome".to_string(),
            asset_class: WholeCellAssetClass::Replication,
            basal_abundance: 12.0,
            bulk_field: Some(WholeCellBulkField::Nucleotides),
            spatial_scope: WholeCellSpatialScope::NucleoidLocal,
            patch_domain: WholeCellPatchDomain::NucleoidTrack,
            chromosome_domain: None,
            operon: None,
            parent_complex: None,
            subsystem_targets: vec![Syn3ASubsystemPreset::ReplisomeTrack],
            count: 12.0,
            anchor_count: 12.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
            process_weights: WholeCellProcessWeights::default(),
        }];
        sim.organism_reactions = vec![WholeCellReactionRuntimeState {
            id: "domain_replication_drive".to_string(),
            name: "domain replication drive".to_string(),
            reaction_class: WholeCellReactionClass::Transcription,
            asset_class: WholeCellAssetClass::Replication,
            nominal_rate: 10.0,
            catalyst: None,
            operon: Some("replication_cycle_operon".to_string()),
            reactants: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "pool_nucleoid_track_nucleotides".to_string(),
                stoichiometry: 1.8,
            }],
            products: Vec::new(),
            subsystem_targets: vec![Syn3ASubsystemPreset::ReplisomeTrack],
            spatial_scope: WholeCellSpatialScope::NucleoidLocal,
            patch_domain: WholeCellPatchDomain::NucleoidTrack,
            chromosome_domain: Some(domain_id),
            current_flux: 7.5,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
            process_weights: WholeCellProcessWeights::default(),
        }];
        sim.refresh_spatial_fields();
        sim.refresh_rdme_drive_fields();
    }

    let left_domain_zero = left_sim.rdme_drive_fields.weighted_mean(
        WholeCellRdmeDriveField::NucleotideDemand,
        &left_sim.chromosome_domain_weights(0),
    );
    let left_domain_last = left_sim.rdme_drive_fields.weighted_mean(
        WholeCellRdmeDriveField::NucleotideDemand,
        &left_sim.chromosome_domain_weights(
            (left_sim.chromosome_domain_count().saturating_sub(1)) as u32,
        ),
    );
    let right_domain_zero = right_sim.rdme_drive_fields.weighted_mean(
        WholeCellRdmeDriveField::NucleotideDemand,
        &right_sim.chromosome_domain_weights(0),
    );
    let right_domain_last = right_sim.rdme_drive_fields.weighted_mean(
        WholeCellRdmeDriveField::NucleotideDemand,
        &right_sim.chromosome_domain_weights(
            (right_sim.chromosome_domain_count().saturating_sub(1)) as u32,
        ),
    );

    assert!(left_domain_zero > left_domain_last);
    assert!(right_domain_last > right_domain_zero);
}

#[test]
fn test_membrane_patch_turnover_prefers_band_loaded_precursors() {
    let mut band_loaded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let mut pole_loaded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let band_weights = band_loaded
        .spatial_fields
        .clone_field(IntracellularSpatialField::MembraneBandZone);
    let pole_weights = pole_loaded
        .spatial_fields
        .clone_field(IntracellularSpatialField::PoleZone);
    let band_values = distribute_total_by_weights(&band_weights, 0.35);
    let pole_values = distribute_total_by_weights(&pole_weights, 0.35);

    band_loaded
        .lattice
        .set_species(IntracellularSpecies::MembranePrecursors, &band_values)
        .expect("band-targeted membrane precursors");
    pole_loaded
        .lattice
        .set_species(IntracellularSpecies::MembranePrecursors, &pole_values)
        .expect("pole-targeted membrane precursors");
    band_loaded.sync_from_lattice();
    pole_loaded.sync_from_lattice();

    for _ in 0..8 {
        band_loaded.geometry_stage(band_loaded.config.dt_ms);
        pole_loaded.geometry_stage(pole_loaded.config.dt_ms);
    }

    assert!(
        band_loaded
            .membrane_division_state
            .membrane_band_lipid_inventory_nm2
            > pole_loaded
                .membrane_division_state
                .membrane_band_lipid_inventory_nm2
    );
    assert!(
        band_loaded.membrane_division_state.band_turnover_pressure
            < pole_loaded.membrane_division_state.band_turnover_pressure
    );
}

#[test]
fn test_resource_estimators_ingest_local_chemistry_context() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    let baseline = sim.base_rule_context(0.0);

    sim.chemistry_report = LocalChemistryReport {
        atp_support: 1.12,
        translation_support: 1.08,
        nucleotide_support: 1.10,
        membrane_support: 1.06,
        crowding_penalty: 0.95,
        mean_glucose: 0.85,
        mean_oxygen: 0.78,
        mean_atp_flux: 0.92,
        mean_carbon_dioxide: 0.18,
    };

    let enriched = sim.base_rule_context(0.0);
    assert!(
        enriched.get(WholeCellRuleSignal::GlucoseSignal)
            > baseline.get(WholeCellRuleSignal::GlucoseSignal)
    );
    assert!(
        enriched.get(WholeCellRuleSignal::OxygenSignal)
            > baseline.get(WholeCellRuleSignal::OxygenSignal)
    );
    assert!(
        enriched.get(WholeCellRuleSignal::EnergySignal)
            > baseline.get(WholeCellRuleSignal::EnergySignal)
    );
}

#[test]
fn test_quantum_profile_accelerates_growth() {
    let config = WholeCellConfig {
        use_gpu: false,
        dt_ms: 0.5,
        ..WholeCellConfig::default()
    };
    let mut baseline = WholeCellSimulator::new(config.clone());
    let mut accelerated = WholeCellSimulator::new(config);
    accelerated.set_quantum_profile(WholeCellQuantumProfile {
        oxphos_efficiency: 1.60,
        translation_efficiency: 1.45,
        nucleotide_polymerization_efficiency: 1.50,
        membrane_synthesis_efficiency: 1.35,
        chromosome_segregation_efficiency: 1.30,
    });

    baseline.run(120);
    accelerated.run(120);

    let baseline_snapshot = baseline.snapshot();
    let accelerated_snapshot = accelerated.snapshot();
    let baseline_complex = baseline.complex_assembly_state();
    let accelerated_complex = accelerated.complex_assembly_state();
    let baseline_membrane = baseline.membrane_division_state();
    let accelerated_membrane = accelerated.membrane_division_state();

    assert!(accelerated_snapshot.atp_mm >= baseline_snapshot.atp_mm);
    assert!(accelerated_complex.ftsz_polymer >= baseline_complex.ftsz_polymer);
    assert!(accelerated_snapshot.surface_area_nm2 > baseline_snapshot.surface_area_nm2);
    assert!(accelerated_membrane.constriction_force > baseline_membrane.constriction_force);
}

#[test]
fn test_surrogate_pools_are_diagnostics_not_stage_drivers() {
    let config = WholeCellConfig {
        use_gpu: false,
        dt_ms: 0.25,
        ..WholeCellConfig::default()
    };
    let mut baseline = WholeCellSimulator::new(config.clone());
    let mut perturbed = WholeCellSimulator::new(config);

    perturbed.active_rnap = 256.0;
    perturbed.active_ribosomes = 320.0;
    perturbed.dnaa = 256.0;
    perturbed.ftsz = 384.0;

    baseline.run(16);
    perturbed.run(16);

    let baseline_snapshot = baseline.snapshot();
    let perturbed_snapshot = perturbed.snapshot();

    assert_eq!(
        perturbed_snapshot.replicated_bp,
        baseline_snapshot.replicated_bp
    );
    assert!(
        (perturbed_snapshot.division_progress - baseline_snapshot.division_progress).abs() < 1.0e-6
    );
    assert!(
        (perturbed_snapshot.surface_area_nm2 - baseline_snapshot.surface_area_nm2).abs() < 1.0e-4
    );
    assert!(perturbed_snapshot.active_rnap < 256.0);
    assert!(perturbed_snapshot.active_ribosomes < 320.0);
    assert!(perturbed_snapshot.dnaa < 256.0);
    assert!(perturbed_snapshot.ftsz < 384.0);
}

#[test]
fn test_explicit_asset_diagnostics_follow_inventory_not_flux_surrogates() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    sim.named_complexes.clear();
    sim.complex_assembly = WholeCellComplexAssemblyState {
        atp_band_complexes: 2.0,
        ribosome_complexes: 15.0,
        rnap_complexes: 9.0,
        replisome_complexes: 4.0,
        membrane_complexes: 7.0,
        ftsz_polymer: 21.0,
        dnaa_activity: 6.0,
        ..WholeCellComplexAssemblyState::default()
    };

    let inventory = sim.assembly_inventory();
    sim.refresh_surrogate_pool_diagnostics(inventory, 10.0, 9.0, 8.0, 7.0, 6.0);

    assert!((sim.active_rnap - 9.0_f32.clamp(8.0, 256.0)).abs() < 1.0e-6);
    assert!((sim.active_ribosomes - 15.0_f32.clamp(12.0, 320.0)).abs() < 1.0e-6);
    assert!((sim.dnaa - 6.0_f32.clamp(8.0, 256.0)).abs() < 1.0e-6);
    assert!((sim.ftsz - 21.0_f32.clamp(12.0, 384.0)).abs() < 1.0e-6);
}

#[test]
fn test_hot_path_accessors_prefer_explicit_chromosome_and_membrane_state() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        dt_ms: 0.25,
        ..WholeCellConfig::default()
    });
    sim.genome_bp = 1000;
    sim.replicated_bp = 0;
    sim.chromosome_separation_nm = 0.0;
    sim.radius_nm = 120.0;
    sim.surface_area_nm2 = WholeCellSimulator::surface_area_from_radius(sim.radius_nm);
    sim.volume_nm3 = WholeCellSimulator::volume_from_radius(sim.radius_nm);
    sim.division_progress = 0.0;

    sim.chromosome_state.chromosome_length_bp = 1000;
    sim.chromosome_state.replicated_bp = 500;
    sim.chromosome_state.replicated_fraction = 0.5;
    sim.chromosome_state.segregation_progress = 0.6;

    sim.membrane_division_state.membrane_area_nm2 = sim.surface_area_nm2 * 1.25;
    sim.membrane_division_state.preferred_membrane_area_nm2 = sim.surface_area_nm2 * 1.25;
    sim.membrane_division_state.septum_radius_fraction = 0.25;
    sim.membrane_division_state.osmotic_balance = 1.0;

    assert!((sim.current_replicated_fraction() - 0.5).abs() < 1.0e-6);
    assert!((sim.current_division_progress() - 0.75).abs() < 1.0e-6);
    assert!(sim.current_radius_nm() > sim.radius_nm);
    assert!(sim.current_chromosome_separation_nm() > 0.0);

    let ctx = sim.base_rule_context(0.0);
    assert!((ctx.get(WholeCellRuleSignal::ReplicatedFraction) - 0.5).abs() < 1.0e-6);
}

#[test]
fn test_local_chemistry_bridge_updates_report_and_md_probe() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        dt_ms: 0.25,
        ..WholeCellConfig::default()
    });
    sim.enable_local_chemistry(10, 10, 6, 0.5, false);

    sim.run(8);
    let chemistry = sim
        .local_chemistry_report()
        .expect("local chemistry report");
    assert!(chemistry.atp_support > 0.0);
    assert!(chemistry.translation_support > 0.0);
    assert!(chemistry.crowding_penalty > 0.0);

    let probe = sim
        .run_local_md_probe(LocalMDProbeRequest {
            site: WholeCellChemistrySite::RibosomeCluster,
            n_atoms: 16,
            steps: 8,
            dt_ps: 0.001,
            box_size_angstrom: 14.0,
            temperature_k: 310.0,
        })
        .expect("md probe");
    assert!(probe.structural_order > 0.0);
    assert!(probe.crowding_penalty > 0.0);
    assert!(sim.last_md_probe().is_some());
}

#[test]
fn test_default_syn3a_subsystems_schedule_and_run() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        dt_ms: 0.25,
        ..WholeCellConfig::default()
    });
    sim.enable_default_syn3a_subsystems();

    let scheduled = sim.scheduled_syn3a_subsystem_probes();
    assert_eq!(scheduled.len(), Syn3ASubsystemPreset::all().len());

    sim.run(12);

    assert!(sim.local_chemistry_report().is_some());
    assert!(sim.last_md_probe().is_some());
    assert!(sim.md_translation_scale() > 0.0);
    assert!(sim.md_membrane_scale() > 0.0);
}

#[test]
fn test_derivation_calibration_is_exposed_on_simulator() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        dt_ms: 0.25,
        ..WholeCellConfig::default()
    });
    sim.enable_local_chemistry(12, 12, 6, 0.5, false);

    let samples = sim.derivation_calibration_samples(0.25, 2);
    assert_eq!(samples.len(), Syn3ASubsystemPreset::all().len());

    let fit = sim
        .fit_derivation_calibration(0.25, 2)
        .expect("fit result")
        .expect("bridge-enabled fit");
    assert_eq!(fit.sample_count, Syn3ASubsystemPreset::all().len());
    assert!(fit.fitted_loss < fit.baseline_loss);
}

#[test]
fn test_subsystem_states_capture_probe_couplings() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        dt_ms: 0.25,
        ..WholeCellConfig::default()
    });
    sim.enable_default_syn3a_subsystems();
    sim.run(16);

    let states = sim.subsystem_states();
    assert_eq!(states.len(), Syn3ASubsystemPreset::all().len());

    let atp_band = states
        .iter()
        .find(|state| state.preset == Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
        .expect("ATP synthase state");
    assert!(atp_band.atp_scale > 0.0);
    assert!(atp_band.membrane_scale > 0.0);
    assert!(atp_band.last_probe_step.is_some());

    let replisome = states
        .iter()
        .find(|state| state.preset == Syn3ASubsystemPreset::ReplisomeTrack)
        .expect("replisome state");
    assert!(replisome.replication_scale > 0.0);
    assert!(replisome.segregation_scale > 0.0);
    assert!(replisome.last_probe_step.is_some());
}

#[test]
fn test_local_chemistry_sites_are_exposed_and_site_resolved() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        dt_ms: 0.25,
        ..WholeCellConfig::default()
    });
    sim.enable_default_syn3a_subsystems();
    sim.run(8);

    let site_reports = sim.local_chemistry_sites();
    assert_eq!(site_reports.len(), Syn3ASubsystemPreset::all().len());

    let atp_band = site_reports
        .iter()
        .find(|report| report.preset == Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
        .expect("ATP site report");
    let replisome = site_reports
        .iter()
        .find(|report| report.preset == Syn3ASubsystemPreset::ReplisomeTrack)
        .expect("replisome site report");
    let ribosome = site_reports
        .iter()
        .find(|report| report.preset == Syn3ASubsystemPreset::RibosomePolysomeCluster)
        .expect("ribosome site report");

    assert!(atp_band.patch_radius > 0);
    assert!(atp_band.localization_score != 0.0);
    assert!(atp_band.site_z < sim.config.z_dim);
    assert!(replisome.nucleotide_support > 0.0);
    assert!(ribosome.substrate_draw > 0.0);
    assert!(replisome.biosynthetic_draw > 0.0);
    assert!(atp_band.demand_satisfaction > 0.0);
    assert!(atp_band.assembly_occupancy > 0.0);
    assert!(ribosome.assembly_stability > 0.0);
    let unique_sites = site_reports
        .iter()
        .map(|report| (report.site_x, report.site_y, report.site_z))
        .collect::<std::collections::HashSet<_>>();
    assert!(unique_sites.len() > 1);
    assert!(
        atp_band.mean_oxygen != replisome.mean_oxygen
            || atp_band.mean_atp_flux != replisome.mean_atp_flux
    );
}

#[test]
fn test_localized_resource_pressure_increases_effective_metabolic_load() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        dt_ms: 0.25,
        ..WholeCellConfig::default()
    });
    sim.chemistry_site_reports = vec![LocalChemistrySiteReport {
        preset: Syn3ASubsystemPreset::RibosomePolysomeCluster,
        site: WholeCellChemistrySite::RibosomeCluster,
        patch_radius: 2,
        site_x: 4,
        site_y: 4,
        site_z: 2,
        localization_score: 0.92,
        atp_support: 0.95,
        translation_support: 0.90,
        nucleotide_support: 0.92,
        membrane_support: 0.94,
        crowding_penalty: 0.88,
        mean_glucose: 0.10,
        mean_oxygen: 0.08,
        mean_atp_flux: 0.06,
        mean_carbon_dioxide: 0.14,
        mean_nitrate: 0.05,
        mean_ammonium: 0.07,
        mean_proton: 0.02,
        mean_phosphorus: 0.04,
        assembly_component_availability: 0.76,
        assembly_occupancy: 0.72,
        assembly_stability: 0.70,
        assembly_turnover: 0.38,
        substrate_draw: 0.60,
        energy_draw: 0.55,
        biosynthetic_draw: 0.24,
        byproduct_load: 0.42,
        demand_satisfaction: 0.46,
    }];

    assert!(sim.effective_metabolic_load() > sim.metabolic_load);
    assert!(sim.localized_supply_scale() < 1.0);
    assert!(sim.localized_resource_pressure() > 0.0);
}

#[test]
fn test_replisome_probe_accelerates_replication() {
    let config = WholeCellConfig {
        use_gpu: false,
        dt_ms: 0.25,
        ..WholeCellConfig::default()
    };
    let mut baseline = WholeCellSimulator::new(config.clone());
    baseline.enable_local_chemistry(12, 12, 6, 0.5, false);

    let mut targeted = WholeCellSimulator::new(config);
    targeted.enable_local_chemistry(12, 12, 6, 0.5, false);
    targeted.schedule_syn3a_subsystem_probe(Syn3ASubsystemPreset::ReplisomeTrack, 1);

    baseline.run(24);
    targeted.run(24);

    let baseline_snapshot = baseline.snapshot();
    let targeted_snapshot = targeted.snapshot();
    let baseline_replisome = baseline_snapshot
        .subsystem_states
        .iter()
        .find(|state| state.preset == Syn3ASubsystemPreset::ReplisomeTrack)
        .expect("baseline replisome state");
    let targeted_replisome = targeted_snapshot
        .subsystem_states
        .iter()
        .find(|state| state.preset == Syn3ASubsystemPreset::ReplisomeTrack)
        .expect("targeted replisome state");

    assert!(
        targeted_snapshot.replicated_bp >= baseline_snapshot.replicated_bp,
        "replication baseline={} targeted={}",
        baseline_snapshot.replicated_bp,
        targeted_snapshot.replicated_bp
    );
    assert!(
        targeted_replisome.replication_scale > baseline_replisome.replication_scale,
        "replication scale baseline={} targeted={}",
        baseline_replisome.replication_scale,
        targeted_replisome.replication_scale
    );
    assert!(
        targeted_snapshot.chromosome_separation_nm > baseline_snapshot.chromosome_separation_nm,
        "segregation baseline={} targeted={}",
        baseline_snapshot.chromosome_separation_nm,
        targeted_snapshot.chromosome_separation_nm
    );
}

#[test]
fn test_membrane_and_septum_probes_accelerate_division() {
    let config = WholeCellConfig {
        use_gpu: false,
        dt_ms: 0.25,
        ..WholeCellConfig::default()
    };
    let mut baseline = WholeCellSimulator::new(config.clone());
    baseline.enable_local_chemistry(12, 12, 6, 0.5, false);

    let mut targeted = WholeCellSimulator::new(config);
    targeted.enable_local_chemistry(12, 12, 6, 0.5, false);
    targeted.schedule_syn3a_subsystem_probe(Syn3ASubsystemPreset::AtpSynthaseMembraneBand, 1);
    targeted.schedule_syn3a_subsystem_probe(Syn3ASubsystemPreset::RibosomePolysomeCluster, 1);
    targeted.schedule_syn3a_subsystem_probe(Syn3ASubsystemPreset::FtsZSeptumRing, 1);

    baseline.run(12);
    targeted.run(12);

    let baseline_snapshot = baseline.snapshot();
    let targeted_snapshot = targeted.snapshot();

    assert!(
        targeted_snapshot.division_progress > baseline_snapshot.division_progress,
        "division baseline={} targeted={}",
        baseline_snapshot.division_progress,
        targeted_snapshot.division_progress
    );
    assert!(
        targeted_snapshot.surface_area_nm2 > baseline_snapshot.surface_area_nm2,
        "surface baseline={} targeted={}",
        baseline_snapshot.surface_area_nm2,
        targeted_snapshot.surface_area_nm2
    );
}

#[test]
fn test_bundled_syn3a_reference_spec_builds_native_simulator() {
    let sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");

    assert_eq!(sim.genome_bp, 543_000);
    assert!(sim.chemistry_bridge.is_some());
    assert_eq!(sim.scheduled_subsystem_probes.len(), 4);
    assert!(sim.lattice.mean_species(IntracellularSpecies::ATP) > 0.0);
    let summary = sim.organism_summary().expect("organism summary");
    let asset_summary = sim
        .organism_asset_summary()
        .expect("organism asset summary");
    assert_eq!(summary.organism, "JCVI-syn3A");
    assert!(summary.gene_count >= 10);
    assert!(summary.transcription_unit_count >= 4);
    assert!(asset_summary.operon_count >= summary.transcription_unit_count);
    assert_eq!(asset_summary.protein_count, summary.gene_count);
    assert!(asset_summary.targeted_complex_count >= 4);
    let registry_summary = sim
        .organism_process_registry_summary()
        .expect("organism process registry summary");
    let registry = sim
        .organism_process_registry()
        .expect("organism process registry");
    assert!(registry_summary.species_count > asset_summary.protein_count);
    assert_eq!(registry_summary.rna_species_count, asset_summary.rna_count);
    assert_eq!(
        registry_summary.protein_species_count,
        asset_summary.protein_count
    );
    assert_eq!(
        registry_summary.complex_species_count,
        asset_summary.complex_count
    );
    assert!(
        registry_summary.assembly_intermediate_species_count >= asset_summary.complex_count * 3
    );
    assert!(registry_summary.transcription_reaction_count >= summary.transcription_unit_count);
    assert!(registry_summary.translation_reaction_count >= asset_summary.protein_count);
    assert!(registry_summary.assembly_reaction_count >= asset_summary.complex_count * 4);
    assert!(registry
        .species
        .iter()
        .any(|species| species.id == "ribosome_biogenesis_operon_complex_mature"));
    assert!(registry
        .reactions
        .iter()
        .any(|reaction| reaction.id == "ribosome_biogenesis_operon_complex_maturation"));
    assert!(sim.provenance.compiled_ir_hash.is_some());
    let profile = sim.organism_profile().expect("organism profile");
    assert!(profile.process_scales.translation > 0.9);
    assert!(profile.metabolic_burden_scale > 0.9);
    let expression = sim
        .organism_expression_state()
        .expect("organism expression state");
    assert!(expression.global_activity > 0.0);
    assert!(expression.transcription_units.len() >= 4);
    assert!(expression.total_transcript_abundance > 0.0);
    assert!(expression.total_protein_abundance > 0.0);
}

#[test]
fn test_apply_pool_seed_prefers_explicit_bulk_field_metadata() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig::default());
    let baseline_atp = sim.lattice.mean_species(IntracellularSpecies::ATP);
    let baseline_glucose = sim.glucose_mm;
    let baseline_oxygen = sim.oxygen_mm;
    let baseline_adp = sim.adp_mm;
    sim.apply_pool_seed(&WholeCellMoleculePoolSpec {
        species: "custom_energy_buffer".to_string(),
        bulk_field: Some(WholeCellBulkField::ATP),
        role: None,
        concentration_mm: 1.75,
        count: 0.0,
    });
    sim.apply_pool_seed(&WholeCellMoleculePoolSpec {
        species: "fallback_carrier".to_string(),
        bulk_field: Some(WholeCellBulkField::Glucose),
        role: None,
        concentration_mm: 1.45,
        count: 0.0,
    });
    sim.apply_pool_seed(&WholeCellMoleculePoolSpec {
        species: "respiratory_buffer".to_string(),
        bulk_field: Some(WholeCellBulkField::Oxygen),
        role: None,
        concentration_mm: 1.05,
        count: 0.0,
    });
    sim.apply_pool_seed(&WholeCellMoleculePoolSpec {
        species: "spent_energy_buffer".to_string(),
        bulk_field: Some(WholeCellBulkField::ADP),
        role: None,
        concentration_mm: 0.42,
        count: 0.0,
    });

    assert!(sim.lattice.mean_species(IntracellularSpecies::ATP) > baseline_atp);
    assert!(sim.glucose_mm > baseline_glucose);
    assert!(sim.oxygen_mm > baseline_oxygen);
    assert!(sim.adp_mm > baseline_adp);
}

#[test]
fn test_runtime_pool_bulk_fields_backfill_from_registry_metadata() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig::default());
    sim.organism_process_registry = Some(WholeCellGenomeProcessRegistry {
        organism: "test".to_string(),
        chromosome_domains: Vec::new(),
        species: vec![crate::whole_cell_data::WholeCellSpeciesSpec {
            id: "custom_energy_buffer".to_string(),
            name: "custom energy buffer".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "cytosol".to_string(),
            asset_class: WholeCellAssetClass::Energy,
            basal_abundance: 12.0,
            bulk_field: Some(WholeCellBulkField::ATP),
            operon: None,
            parent_complex: None,
            subsystem_targets: Vec::new(),
            spatial_scope: WholeCellSpatialScope::WellMixed,
            patch_domain: WholeCellPatchDomain::Distributed,
            chromosome_domain: None,
            process_weights: WholeCellProcessWeights::default(),
        }],
        reactions: Vec::new(),
    });
    sim.organism_species = vec![WholeCellSpeciesRuntimeState {
        id: "custom_energy_buffer".to_string(),
        name: "opaque pool".to_string(),
        species_class: WholeCellSpeciesClass::Pool,
        compartment: "cytosol".to_string(),
        asset_class: WholeCellAssetClass::Energy,
        basal_abundance: 12.0,
        bulk_field: None,
        operon: None,
        parent_complex: None,
        subsystem_targets: Vec::new(),
        spatial_scope: WholeCellSpatialScope::WellMixed,
        patch_domain: WholeCellPatchDomain::Distributed,
        chromosome_domain: None,
        count: 12.0,
        anchor_count: 12.0,
        synthesis_rate: 0.0,
        turnover_rate: 0.0,
        process_weights: WholeCellProcessWeights::default(),
    }];

    sim.normalize_runtime_species_bulk_fields();

    assert_eq!(
        sim.organism_species[0].bulk_field,
        Some(WholeCellBulkField::ATP)
    );
}

#[test]
fn test_apply_organism_data_initialization_uses_explicit_pool_bulk_fields() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig::default());
    let mut organism =
        crate::whole_cell_data::bundled_syn3a_organism_spec().expect("bundled organism");
    let atp_pool = organism
        .pools
        .iter_mut()
        .find(|pool| pool.bulk_field == Some(WholeCellBulkField::ATP))
        .expect("ATP pool");
    atp_pool.concentration_mm = 1.65;

    let baseline_atp = sim.lattice.mean_species(IntracellularSpecies::ATP);
    sim.organism_data = Some(organism);
    sim.apply_organism_data_initialization();

    assert!(sim.lattice.mean_species(IntracellularSpecies::ATP) > baseline_atp);
    let stored_pool = sim
        .organism_data
        .as_ref()
        .expect("stored organism")
        .pools
        .iter()
        .find(|pool| pool.species.eq_ignore_ascii_case("atp"))
        .expect("stored ATP pool");
    assert_eq!(stored_pool.bulk_field, Some(WholeCellBulkField::ATP));
}

#[test]
fn test_apply_pool_seed_uses_explicit_pool_roles_for_diagnostics() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig::default());
    sim.apply_pool_seed(&WholeCellMoleculePoolSpec {
        species: "opaque_diagnostic_pool".to_string(),
        bulk_field: None,
        role: Some(WholeCellPoolRole::ActiveRibosomes),
        concentration_mm: 0.0,
        count: 88.0,
    });
    sim.apply_pool_seed(&WholeCellMoleculePoolSpec {
        species: "opaque_polymer_pool".to_string(),
        bulk_field: None,
        role: Some(WholeCellPoolRole::Ftsz),
        concentration_mm: 0.0,
        count: 42.0,
    });

    assert!((sim.active_ribosomes - 88.0).abs() < 1.0e-6);
    assert!((sim.ftsz - 42.0).abs() < 1.0e-6);
}

#[test]
fn test_saved_state_round_trip_preserves_core_progress() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    sim.run(8);

    let saved = sim.save_state_json().expect("serialize saved state");
    let restored = WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");

    let original = sim.snapshot();
    let reloaded = restored.snapshot();

    assert_eq!(reloaded.step_count, original.step_count);
    assert_eq!(reloaded.replicated_bp, original.replicated_bp);
    assert_eq!(reloaded.genome_bp, original.genome_bp);
    assert!((reloaded.time_ms - original.time_ms).abs() < 1.0e-6);
    assert!((reloaded.atp_mm - original.atp_mm).abs() < 1.0e-6);
    assert_eq!(
        restored.scheduled_subsystem_probes.len(),
        sim.scheduled_subsystem_probes.len()
    );
    assert_eq!(restored.organism_summary(), sim.organism_summary());
    assert_eq!(
        restored.organism_asset_summary(),
        sim.organism_asset_summary()
    );
    assert_eq!(
        restored.organism_process_registry_summary(),
        sim.organism_process_registry_summary()
    );
    assert_eq!(
        restored
            .organism_expression_state()
            .expect("restored organism expression")
            .transcription_units
            .len(),
        sim.organism_expression_state()
            .expect("original organism expression")
            .transcription_units
            .len()
    );
    assert!(
        restored
            .organism_expression_state()
            .expect("restored expression")
            .total_transcript_abundance
            > 0.0
    );
    let restored_complex = restored.complex_assembly_state();
    let original_complex = sim.complex_assembly_state();
    assert!(restored_complex.total_complexes() > 0.0);
    assert!(
        (restored_complex.ribosome_complexes - original_complex.ribosome_complexes).abs() < 1.0e-6
    );
}

#[test]
fn test_boundary_snapshots_and_save_state_prefer_explicit_state() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    sim.genome_bp = 1000;
    sim.replicated_bp = 0;
    sim.chromosome_separation_nm = 0.0;
    sim.radius_nm = 100.0;
    sim.surface_area_nm2 = WholeCellSimulator::surface_area_from_radius(sim.radius_nm);
    sim.volume_nm3 = WholeCellSimulator::volume_from_radius(sim.radius_nm);
    sim.division_progress = 0.0;
    sim.ftsz = 0.0;
    sim.dnaa = 0.0;
    sim.active_ribosomes = 0.0;
    sim.active_rnap = 0.0;

    sim.chromosome_state.chromosome_length_bp = 1000;
    sim.chromosome_state.replicated_bp = 640;
    sim.chromosome_state.replicated_fraction = 0.64;
    sim.chromosome_state.segregation_progress = 0.5;

    sim.membrane_division_state.membrane_area_nm2 = sim.surface_area_nm2 * 1.40;
    sim.membrane_division_state.preferred_membrane_area_nm2 = sim.surface_area_nm2 * 1.40;
    sim.membrane_division_state.septum_radius_fraction = 0.20;
    sim.membrane_division_state.osmotic_balance = 1.0;

    sim.named_complexes.clear();
    sim.complex_assembly = WholeCellComplexAssemblyState {
        ribosome_complexes: 18.0,
        rnap_complexes: 11.0,
        dnaa_activity: 7.0,
        ftsz_polymer: 23.0,
        ..WholeCellComplexAssemblyState::default()
    };

    let snapshot = sim.snapshot();
    assert_eq!(snapshot.replicated_bp, 640);
    assert!((snapshot.division_progress - 0.80).abs() < 1.0e-6);
    assert!(snapshot.radius_nm > 100.0);
    assert!(snapshot.chromosome_separation_nm > 0.0);
    assert!((snapshot.active_ribosomes - 18.0).abs() < 1.0e-6);
    assert!((snapshot.active_rnap - 11.0).abs() < 1.0e-6);
    assert!((snapshot.ftsz - 23.0).abs() < 1.0e-6);
    assert!((snapshot.dnaa - 8.0).abs() < 1.0e-6);

    let saved = parse_saved_state_json(&sim.save_state_json().expect("serialize saved state"))
        .expect("parse saved state");
    assert_eq!(saved.core.replicated_bp, 640);
    assert!((saved.core.division_progress - 0.80).abs() < 1.0e-6);
    assert!(saved.core.radius_nm > 100.0);
    assert!(saved.core.chromosome_separation_nm > 0.0);
    assert!((saved.core.active_ribosomes - 18.0).abs() < 1.0e-6);
    assert!((saved.core.active_rnap - 11.0).abs() < 1.0e-6);
    assert!((saved.core.ftsz - 23.0).abs() < 1.0e-6);
    assert!((saved.core.dnaa - 8.0).abs() < 1.0e-6);
}

#[test]
fn test_bundleless_boundary_diagnostics_prefer_explicit_complex_state() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig::default());
    sim.active_rnap = 1.0;
    sim.active_ribosomes = 2.0;
    sim.dnaa = 3.0;
    sim.ftsz = 4.0;
    sim.complex_assembly = WholeCellComplexAssemblyState {
        ribosome_complexes: 17.0,
        rnap_complexes: 10.0,
        dnaa_activity: 8.5,
        ftsz_polymer: 24.0,
        ..WholeCellComplexAssemblyState::default()
    };

    let snapshot = sim.snapshot();
    assert!((sim.ftsz() - 24.0).abs() < 1.0e-6);
    assert!((snapshot.active_rnap - 10.0).abs() < 1.0e-6);
    assert!((snapshot.active_ribosomes - 17.0).abs() < 1.0e-6);
    assert!((snapshot.dnaa - 8.5).abs() < 1.0e-6);
    assert!((snapshot.ftsz - 24.0).abs() < 1.0e-6);

    let saved = parse_saved_state_json(&sim.save_state_json().expect("serialize saved state"))
        .expect("parse saved state");
    assert!((saved.core.active_rnap - 10.0).abs() < 1.0e-6);
    assert!((saved.core.active_ribosomes - 17.0).abs() < 1.0e-6);
    assert!((saved.core.dnaa - 8.5).abs() < 1.0e-6);
    assert!((saved.core.ftsz - 24.0).abs() < 1.0e-6);
}

#[test]
fn test_bundleless_boundary_diagnostics_prefer_explicit_named_complex_state() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig::default());
    sim.active_rnap = 1.0;
    sim.active_ribosomes = 2.0;
    sim.dnaa = 3.0;
    sim.ftsz = 4.0;
    sim.complex_assembly = WholeCellComplexAssemblyState::default();
    sim.named_complexes = vec![
        WholeCellNamedComplexState {
            id: "ribosome".to_string(),
            operon: "ribosome".to_string(),
            asset_class: WholeCellAssetClass::Translation,
            family: WholeCellAssemblyFamily::Ribosome,
            subsystem_targets: vec![Syn3ASubsystemPreset::RibosomePolysomeCluster],
            subunit_pool: 0.0,
            nucleation_intermediate: 0.0,
            elongation_intermediate: 0.0,
            abundance: 17.0,
            target_abundance: 20.0,
            assembly_rate: 1.2,
            degradation_rate: 0.2,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 1.0,
            structural_support: 1.0,
            assembly_progress: 0.0,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 0.0,
            shared_component_pressure: 0.0,
            insertion_progress: 1.0,
            failure_count: 0.0,
        },
        WholeCellNamedComplexState {
            id: "rnap".to_string(),
            operon: "rnap".to_string(),
            asset_class: WholeCellAssetClass::Homeostasis,
            family: WholeCellAssemblyFamily::RnaPolymerase,
            subsystem_targets: Vec::new(),
            subunit_pool: 0.0,
            nucleation_intermediate: 0.0,
            elongation_intermediate: 0.0,
            abundance: 10.0,
            target_abundance: 12.0,
            assembly_rate: 0.9,
            degradation_rate: 0.15,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 1.0,
            structural_support: 1.0,
            assembly_progress: 0.0,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 0.0,
            shared_component_pressure: 0.0,
            insertion_progress: 1.0,
            failure_count: 0.0,
        },
        WholeCellNamedComplexState {
            id: "divisome".to_string(),
            operon: "divisome".to_string(),
            asset_class: WholeCellAssetClass::Constriction,
            family: WholeCellAssemblyFamily::Divisome,
            subsystem_targets: vec![Syn3ASubsystemPreset::FtsZSeptumRing],
            subunit_pool: 0.0,
            nucleation_intermediate: 0.0,
            elongation_intermediate: 0.0,
            abundance: 24.0,
            target_abundance: 30.0,
            assembly_rate: 1.0,
            degradation_rate: 0.1,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 1.0,
            structural_support: 1.0,
            assembly_progress: 0.0,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 0.0,
            shared_component_pressure: 0.0,
            insertion_progress: 1.0,
            failure_count: 0.0,
        },
        WholeCellNamedComplexState {
            id: "replisome".to_string(),
            operon: "replisome".to_string(),
            asset_class: WholeCellAssetClass::Replication,
            family: WholeCellAssemblyFamily::Replisome,
            subsystem_targets: vec![Syn3ASubsystemPreset::ReplisomeTrack],
            subunit_pool: 0.0,
            nucleation_intermediate: 0.0,
            elongation_intermediate: 0.0,
            abundance: 20.0,
            target_abundance: 22.0,
            assembly_rate: 0.8,
            degradation_rate: 0.1,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 1.0,
            structural_support: 1.0,
            assembly_progress: 0.0,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 0.0,
            shared_component_pressure: 0.0,
            insertion_progress: 1.0,
            failure_count: 0.0,
        },
    ];

    let expected = sim.aggregate_named_complex_assembly_state_without_assets();
    let snapshot = sim.snapshot();
    assert!((snapshot.active_rnap - expected.rnap_complexes).abs() < 1.0e-6);
    assert!((snapshot.active_ribosomes - expected.ribosome_complexes).abs() < 1.0e-6);
    assert!((snapshot.dnaa - expected.dnaa_activity.clamp(8.0, 256.0)).abs() < 1.0e-6);
    assert!((snapshot.ftsz - expected.ftsz_polymer).abs() < 1.0e-6);

    let saved = parse_saved_state_json(&sim.save_state_json().expect("serialize saved state"))
        .expect("parse saved state");
    assert!((saved.core.active_rnap - expected.rnap_complexes).abs() < 1.0e-6);
    assert!((saved.core.active_ribosomes - expected.ribosome_complexes).abs() < 1.0e-6);
    assert!((saved.core.dnaa - expected.dnaa_activity.clamp(8.0, 256.0)).abs() < 1.0e-6);
    assert!((saved.core.ftsz - expected.ftsz_polymer).abs() < 1.0e-6);
}

#[test]
fn test_restore_saved_state_prefers_explicit_state_over_stale_core_summary() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    sim.genome_bp = 1000;
    sim.replicated_bp = 0;
    sim.chromosome_separation_nm = 0.0;
    sim.radius_nm = 100.0;
    sim.surface_area_nm2 = WholeCellSimulator::surface_area_from_radius(sim.radius_nm);
    sim.volume_nm3 = WholeCellSimulator::volume_from_radius(sim.radius_nm);
    sim.division_progress = 0.0;
    sim.ftsz = 0.0;
    sim.dnaa = 0.0;
    sim.active_ribosomes = 0.0;
    sim.active_rnap = 0.0;

    sim.chromosome_state.chromosome_length_bp = 1000;
    sim.chromosome_state.replicated_bp = 580;
    sim.chromosome_state.replicated_fraction = 0.58;
    sim.chromosome_state.segregation_progress = 0.45;

    sim.membrane_division_state.membrane_area_nm2 = sim.surface_area_nm2 * 1.30;
    sim.membrane_division_state.preferred_membrane_area_nm2 = sim.surface_area_nm2 * 1.30;
    sim.membrane_division_state.septum_radius_fraction = 0.30;
    sim.membrane_division_state.osmotic_balance = 1.0;

    for complex in &mut sim.named_complexes {
        match complex.family {
            WholeCellAssemblyFamily::Ribosome => complex.abundance = 19.0,
            WholeCellAssemblyFamily::RnaPolymerase => complex.abundance = 12.0,
            WholeCellAssemblyFamily::Divisome => complex.abundance = 27.0,
            WholeCellAssemblyFamily::Replisome => complex.abundance = 9.0,
            _ => complex.abundance = 0.0,
        }
    }
    let expected = sim.snapshot();

    let mut saved = parse_saved_state_json(&sim.save_state_json().expect("serialize saved"))
        .expect("parse saved state");
    saved.core.replicated_bp = 12;
    saved.core.chromosome_separation_nm = 0.0;
    saved.core.radius_nm = 55.0;
    saved.core.surface_area_nm2 = WholeCellSimulator::surface_area_from_radius(55.0);
    saved.core.volume_nm3 = WholeCellSimulator::volume_from_radius(55.0);
    saved.core.division_progress = 0.05;
    saved.core.active_rnap = 3.0;
    saved.core.active_ribosomes = 4.0;
    saved.core.dnaa = 2.0;
    saved.core.ftsz = 5.0;

    let restored = WholeCellSimulator::from_saved_state_json(
        &saved_state_to_json(&saved).expect("re-serialize saved state"),
    )
    .expect("restore saved state");
    let snapshot = restored.snapshot();

    assert_eq!(snapshot.replicated_bp, expected.replicated_bp);
    assert!((snapshot.division_progress - expected.division_progress).abs() < 1.0e-6);
    assert!((snapshot.radius_nm - expected.radius_nm).abs() < 1.0e-6);
    assert!((snapshot.chromosome_separation_nm - expected.chromosome_separation_nm).abs() < 1.0e-6);
    assert!((snapshot.active_rnap - expected.active_rnap).abs() < 1.0e-6);
    assert!((snapshot.active_ribosomes - expected.active_ribosomes).abs() < 1.0e-6);
    assert!((snapshot.dnaa - expected.dnaa).abs() < 1.0e-6);
    assert!((snapshot.ftsz - expected.ftsz).abs() < 1.0e-6);
}

#[test]
fn test_saved_state_round_trip_preserves_scheduler_state() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    sim.enable_default_syn3a_subsystems();
    sim.run(10);

    let original = sim.snapshot();
    let saved = sim.save_state_json().expect("serialize saved state");
    let restored = WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");
    let reloaded = restored.snapshot();

    assert_eq!(reloaded.scheduler_state, original.scheduler_state);
    assert!(reloaded
        .scheduler_state
        .stage_clocks
        .iter()
        .any(|clock| clock.run_count > 0));
    assert!(
        reloaded
            .scheduler_state
            .stage_clocks
            .iter()
            .find(|clock| clock.stage == WholeCellSolverStage::AtomisticRefinement)
            .expect("atomistic clock")
            .run_count
            > 0
    );
}

#[test]
fn test_multirate_scheduler_allows_stress_driven_early_cme_reexecution() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    sim.config.cme_interval = 8;
    sim.config.ode_interval = 8;
    sim.atp_mm = 0.08;
    sim.adp_mm = 2.10;
    sim.glucose_mm = 0.12;
    sim.oxygen_mm = 0.14;
    sim.metabolic_load = 2.8;
    sim.chemistry_report.translation_support = 0.58;
    sim.chemistry_report.nucleotide_support = 0.60;
    sim.chemistry_report.crowding_penalty = 0.74;
    sim.refresh_organism_expression_state();
    sim.refresh_multirate_scheduler();

    let cme_clock = sim
        .scheduler_clock(WholeCellSolverStage::Cme)
        .expect("CME clock")
        .clone();
    let ode_clock = sim
        .scheduler_clock(WholeCellSolverStage::Ode)
        .expect("ODE clock")
        .clone();
    assert!(cme_clock.dynamic_interval_steps < sim.config.cme_interval);
    assert!(ode_clock.dynamic_interval_steps < sim.config.ode_interval);

    sim.step();
    let initial_cme_runs = sim
        .scheduler_clock(WholeCellSolverStage::Cme)
        .expect("CME clock after first step")
        .run_count;
    let initial_ode_runs = sim
        .scheduler_clock(WholeCellSolverStage::Ode)
        .expect("ODE clock after first step")
        .run_count;
    assert_eq!(initial_cme_runs, 1);
    assert_eq!(initial_ode_runs, 1);

    let followup_steps = cme_clock
        .dynamic_interval_steps
        .max(ode_clock.dynamic_interval_steps);
    sim.run(followup_steps);

    let cme_after = sim
        .scheduler_clock(WholeCellSolverStage::Cme)
        .expect("CME clock after followup");
    let ode_after = sim
        .scheduler_clock(WholeCellSolverStage::Ode)
        .expect("ODE clock after followup");
    assert!(cme_after.run_count >= 2);
    assert!(ode_after.run_count >= 2);
    assert!(sim.step_count < sim.config.cme_interval + 2);
}

#[test]
fn test_organism_expression_state_responds_to_energy_and_load_stress() {
    let mut baseline =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    baseline.refresh_organism_expression_state();
    let baseline_expression = baseline
        .organism_expression_state()
        .expect("baseline expression");

    let mut stressed =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    stressed.atp_mm = 0.10;
    stressed.adp_mm = 1.80;
    stressed.glucose_mm = 0.15;
    stressed.oxygen_mm = 0.12;
    stressed.metabolic_load = 2.4;
    stressed.chemistry_report.translation_support = 0.62;
    stressed.chemistry_report.nucleotide_support = 0.58;
    stressed.chemistry_report.membrane_support = 0.64;
    stressed.chemistry_report.crowding_penalty = 0.72;
    stressed.refresh_organism_expression_state();
    let stressed_expression = stressed
        .organism_expression_state()
        .expect("stressed expression");
    let baseline_ribosome = baseline_expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == "ribosome_biogenesis_operon")
        .expect("baseline ribosome operon");
    let stressed_ribosome = stressed_expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == "ribosome_biogenesis_operon")
        .expect("stressed ribosome operon");

    assert!(stressed_expression.global_activity < baseline_expression.global_activity);
    assert!(stressed_ribosome.effective_activity < baseline_ribosome.effective_activity);
    assert!(stressed_ribosome.stress_penalty > baseline_ribosome.stress_penalty);
}

#[test]
fn test_runtime_process_scales_consume_compiled_registry_signals() {
    let baseline = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let baseline_ribosome_rna = baseline
        .organism_species
        .iter()
        .cloned()
        .into_iter()
        .find(|species| species.id == "ribosome_small_subunit_core_rna")
        .expect("baseline ribosome RNA species");
    let baseline_ftsz_rna = baseline
        .organism_species
        .iter()
        .cloned()
        .into_iter()
        .find(|species| species.id == "ftsz_ring_polymerization_core_rna")
        .expect("baseline FtsZ RNA species");

    let mut registry_boosted =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    {
        let assets = registry_boosted
            .organism_assets
            .as_mut()
            .expect("compiled genome asset package");
        for rna in &mut assets.rnas {
            if rna.operon == "ribosome_biogenesis_operon" {
                rna.basal_abundance *= 18.0;
            }
            if rna.operon == "division_ring_operon" {
                rna.basal_abundance *= 16.0;
            }
        }
    }
    assert!(registry_boosted.recompile_process_registry_from_assets());
    registry_boosted.refresh_organism_expression_state();
    let boosted_ribosome_rna = registry_boosted
        .organism_species
        .iter()
        .cloned()
        .into_iter()
        .find(|species| species.id == "ribosome_small_subunit_core_rna")
        .expect("boosted ribosome RNA species");
    let boosted_ftsz_rna = registry_boosted
        .organism_species
        .iter()
        .cloned()
        .into_iter()
        .find(|species| species.id == "ftsz_ring_polymerization_core_rna")
        .expect("boosted FtsZ RNA species");

    assert!(boosted_ribosome_rna.basal_abundance > baseline_ribosome_rna.basal_abundance);
    assert!(boosted_ftsz_rna.basal_abundance > baseline_ftsz_rna.basal_abundance);
}

#[test]
fn test_organism_inventory_state_accumulates_transcripts_and_proteins() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let start_expression = sim
        .organism_expression_state()
        .expect("initial organism expression");
    let start_ribosome = start_expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == "ribosome_biogenesis_operon")
        .expect("initial ribosome unit");

    sim.run(12);

    let end_expression = sim
        .organism_expression_state()
        .expect("updated organism expression");
    let end_ribosome = end_expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == "ribosome_biogenesis_operon")
        .expect("updated ribosome unit");

    assert!(end_expression.total_transcript_abundance > 0.0);
    assert!(end_expression.total_protein_abundance > 0.0);
    assert!(end_ribosome.transcript_synthesis_rate > 0.0);
    assert!(end_ribosome.protein_synthesis_rate > 0.0);
    assert!(end_ribosome.transcript_abundance != start_ribosome.transcript_abundance);
    assert!(end_ribosome.protein_abundance != start_ribosome.protein_abundance);
}

#[test]
fn test_expression_execution_state_tracks_occupancy_and_product_pools() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");

    sim.run(12);

    let expression = sim
        .organism_expression_state()
        .expect("updated organism expression");
    let ribosome_unit = expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == "ribosome_biogenesis_operon")
        .expect("ribosome operon");

    assert!(ribosome_unit.transcription_length_nt >= 90.0);
    assert!(ribosome_unit.translation_length_aa >= 30.0);
    assert!(ribosome_unit.promoter_open_fraction > 0.0);
    assert!(ribosome_unit.active_rnap_occupancy > 0.0);
    assert!(ribosome_unit.active_ribosome_occupancy > 0.0);
    assert!(ribosome_unit.mature_transcript_abundance > 0.0);
    assert!(ribosome_unit.mature_protein_abundance > 0.0);
    assert!(ribosome_unit.transcript_abundance >= ribosome_unit.mature_transcript_abundance);
    assert!(ribosome_unit.protein_abundance >= ribosome_unit.mature_protein_abundance);
}

#[test]
fn test_saved_state_round_trip_preserves_expression_execution_state() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");

    sim.run(10);

    let saved = sim.save_state_json().expect("serialize saved state");
    let restored = WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");
    let original = sim
        .organism_expression_state()
        .expect("original expression state");
    let reloaded = restored
        .organism_expression_state()
        .expect("restored expression state");
    let original_unit = original
        .transcription_units
        .iter()
        .find(|unit| unit.name == "ribosome_biogenesis_operon")
        .expect("original ribosome unit");
    let reloaded_unit = reloaded
        .transcription_units
        .iter()
        .find(|unit| unit.name == "ribosome_biogenesis_operon")
        .expect("reloaded ribosome unit");

    assert!(
        (reloaded_unit.promoter_open_fraction - original_unit.promoter_open_fraction).abs()
            < 1.0e-6
    );
    assert!(
        (reloaded_unit.active_rnap_occupancy - original_unit.active_rnap_occupancy).abs() < 1.0e-6
    );
    assert!(
        (reloaded_unit.transcription_progress_nt - original_unit.transcription_progress_nt).abs()
            < 1.0e-6
    );
    assert!(
        (reloaded_unit.mature_transcript_abundance - original_unit.mature_transcript_abundance)
            .abs()
            < 1.0e-6
    );
    assert!(
        (reloaded_unit.active_ribosome_occupancy - original_unit.active_ribosome_occupancy).abs()
            < 1.0e-6
    );
    assert!(
        (reloaded_unit.translation_progress_aa - original_unit.translation_progress_aa).abs()
            < 1.0e-6
    );
    assert!(
        (reloaded_unit.mature_protein_abundance - original_unit.mature_protein_abundance).abs()
            < 1.0e-6
    );
}

#[test]
fn test_chromosome_state_tracks_forks_loci_and_restart() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    sim.dnaa = 128.0;
    sim.complex_assembly.replisome_complexes = 24.0;
    sim.chromosome_state.replicated_bp = 2;
    sim.chromosome_state.forks = WholeCellSimulator::chromosome_forks_from_progress(
        sim.chromosome_state.chromosome_length_bp,
        sim.chromosome_state.origin_bp,
        sim.chromosome_state.terminus_bp,
        2,
    );
    sim.synchronize_chromosome_summary();
    let start = sim.chromosome_state();

    assert_eq!(start.chromosome_length_bp, sim.genome_bp);
    assert!(!start.loci.is_empty());

    sim.run(24);

    let end = sim.chromosome_state();
    assert!(end.initiation_events >= start.initiation_events);
    assert!(!end.forks.is_empty());
    assert!(end.replicated_bp > 0);
    assert!(end.mean_locus_accessibility > 0.0);
    assert!(end
        .loci
        .iter()
        .zip(start.loci.iter())
        .any(|(end_locus, start_locus)| {
            (end_locus.accessibility - start_locus.accessibility).abs() > 1.0e-6
                || (end_locus.torsional_stress - start_locus.torsional_stress).abs() > 1.0e-6
        }));

    let saved = sim.save_state_json().expect("serialize saved state");
    let restored = WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");
    let restored_state = restored.chromosome_state();

    assert_eq!(restored_state.loci.len(), end.loci.len());
    assert_eq!(restored_state.forks.len(), end.forks.len());
    assert_eq!(restored_state.initiation_events, end.initiation_events);
    assert_eq!(restored_state.completion_events, end.completion_events);
    assert_eq!(restored_state.replicated_bp, end.replicated_bp);
    assert!(
        (restored_state.mean_locus_accessibility - end.mean_locus_accessibility).abs() < 1.0e-6
    );
    if let (Some(restored_fork), Some(end_fork)) = (restored_state.forks.first(), end.forks.first())
    {
        assert_eq!(restored_fork.pause_events, end_fork.pause_events);
        assert_eq!(restored_fork.traveled_bp, end_fork.traveled_bp);
        assert!((restored_fork.collision_pressure - end_fork.collision_pressure).abs() < 1.0e-6);
    }
}

#[test]
fn test_head_on_transcription_increases_chromosome_collision_pressure() {
    let mut baseline =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    baseline.dnaa = 128.0;
    baseline.complex_assembly.replisome_complexes = 24.0;
    baseline.chromosome_state.replicated_bp = 2;
    baseline.chromosome_state.forks = WholeCellSimulator::chromosome_forks_from_progress(
        baseline.chromosome_state.chromosome_length_bp,
        baseline.chromosome_state.origin_bp,
        baseline.chromosome_state.terminus_bp,
        2,
    );
    baseline.synchronize_chromosome_summary();
    baseline.run(24);
    let baseline_state = baseline.chromosome_state();
    let baseline_collision = baseline_state
        .forks
        .iter()
        .map(|fork| fork.collision_pressure)
        .sum::<f32>();
    let baseline_pauses = baseline_state
        .forks
        .iter()
        .map(|fork| fork.pause_events)
        .sum::<u32>();

    let mut stressed =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    stressed.dnaa = 128.0;
    stressed.complex_assembly.replisome_complexes = 24.0;
    stressed.chromosome_state.replicated_bp = 2;
    stressed.chromosome_state.forks = WholeCellSimulator::chromosome_forks_from_progress(
        stressed.chromosome_state.chromosome_length_bp,
        stressed.chromosome_state.origin_bp,
        stressed.chromosome_state.terminus_bp,
        2,
    );
    stressed.synchronize_chromosome_summary();
    if let Some(organism) = stressed.organism_data.as_mut() {
        for feature in &mut organism.genes {
            feature.strand = -1;
            feature.basal_expression *= 18.0;
        }
    }
    stressed.run(24);
    let stressed_state = stressed.chromosome_state();
    let stressed_collision = stressed_state
        .forks
        .iter()
        .map(|fork| fork.collision_pressure)
        .sum::<f32>();
    let stressed_pauses = stressed_state
        .forks
        .iter()
        .map(|fork| fork.pause_events)
        .sum::<u32>();

    assert!(stressed_collision >= baseline_collision);
    assert!(stressed_pauses >= baseline_pauses);
    assert!(stressed_state.torsional_stress >= baseline_state.torsional_stress);
}

#[test]
fn test_membrane_division_state_persists_across_restart() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");

    sim.run(16);

    let original = sim.membrane_division_state();
    assert!(original.membrane_area_nm2 > 0.0);
    assert!(original.preferred_membrane_area_nm2 > 0.0);
    assert!(original.envelope_integrity > 0.0);

    let saved = sim.save_state_json().expect("serialize saved state");
    let restored = WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");
    let reloaded = restored.membrane_division_state();

    assert!((reloaded.membrane_area_nm2 - original.membrane_area_nm2).abs() < 1.0e-6);
    assert!(
        (reloaded.preferred_membrane_area_nm2 - original.preferred_membrane_area_nm2).abs()
            < 1.0e-6
    );
    assert!((reloaded.divisome_occupancy - original.divisome_occupancy).abs() < 1.0e-6);
    assert!((reloaded.ring_tension - original.ring_tension).abs() < 1.0e-6);
    assert!((reloaded.chromosome_occlusion - original.chromosome_occlusion).abs() < 1.0e-6);
    assert!((reloaded.failure_pressure - original.failure_pressure).abs() < 1.0e-6);
}

#[test]
fn test_chromosome_occlusion_penalizes_division_mechanics() {
    let mut occluded =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let mut cleared =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");

    for sim in [&mut occluded, &mut cleared] {
        let genome_bp = sim.chromosome_state.chromosome_length_bp.max(1);
        sim.chromosome_state.replicated_bp = genome_bp / 2;
        sim.chromosome_state.forks = WholeCellSimulator::chromosome_forks_from_progress(
            genome_bp,
            sim.chromosome_state.origin_bp,
            sim.chromosome_state.terminus_bp,
            genome_bp / 2,
        );
        sim.chromosome_state = sim.normalize_chromosome_state(sim.chromosome_state.clone());
        sim.synchronize_chromosome_summary();
        sim.membrane_division_state = sim.seeded_membrane_division_state();
        sim.synchronize_membrane_division_summary();
    }

    occluded.chromosome_state.segregation_progress = 0.0;
    occluded.chromosome_state.compaction_fraction = 0.85;
    occluded.chromosome_state =
        occluded.normalize_chromosome_state(occluded.chromosome_state.clone());
    occluded.synchronize_chromosome_summary();

    cleared.chromosome_state.segregation_progress = 0.90;
    cleared.chromosome_state.compaction_fraction = 0.35;
    cleared.chromosome_state = cleared.normalize_chromosome_state(cleared.chromosome_state.clone());
    cleared.synchronize_chromosome_summary();

    occluded.run(24);
    cleared.run(24);

    let occluded_state = occluded.membrane_division_state();
    let cleared_state = cleared.membrane_division_state();

    assert!(occluded_state.chromosome_occlusion > cleared_state.chromosome_occlusion);
    assert!(cleared_state.divisome_occupancy > occluded_state.divisome_occupancy);
    assert!(cleared_state.ring_occupancy > occluded_state.ring_occupancy);
    assert!(cleared.snapshot().division_progress > occluded.snapshot().division_progress);
}

#[test]
fn test_complex_assembly_state_accumulates_from_inventory_and_preserves_targets() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let start = sim.complex_assembly_state();

    sim.run(12);

    let end = sim.complex_assembly_state();
    assert!(end.total_complexes() > 0.0);
    assert!(end.ribosome_target > 0.0);
    assert!(end.ribosome_assembly_rate > 0.0);
    assert!(end.replisome_assembly_rate > 0.0);
    assert!(end.ribosome_complexes != start.ribosome_complexes);
    assert!(end.replisome_complexes != start.replisome_complexes);
}

#[test]
fn test_named_complex_state_is_compiled_from_assets_and_persists_across_restart() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let start_named = sim.named_complexes_state();
    let asset_package = sim
        .organism_asset_package()
        .expect("compiled genome asset package");

    assert_eq!(start_named.len(), asset_package.complexes.len());
    let start_ribosome = start_named
        .iter()
        .find(|state| state.id == "ribosome_biogenesis_operon_complex")
        .expect("ribosome complex state");
    assert_eq!(start_ribosome.family, WholeCellAssemblyFamily::Ribosome);
    assert!(start_ribosome.subunit_pool > 0.0);
    assert!(start_ribosome.nucleation_intermediate > 0.0);
    assert!(start_ribosome.elongation_intermediate > 0.0);
    assert!(start_ribosome.abundance > 0.0);
    assert!(start_ribosome.component_satisfaction > 0.0);
    assert!(start_ribosome.structural_support > 0.0);
    assert!(start_ribosome.assembly_progress > 0.0);
    assert!(start_ribosome.limiting_component_signal > 0.0);
    assert!(start_ribosome.insertion_progress >= 0.0);

    sim.run(12);

    let end_named = sim.named_complexes_state();
    let end_ribosome = end_named
        .iter()
        .find(|state| state.id == "ribosome_biogenesis_operon_complex")
        .expect("updated ribosome complex state");
    assert!(end_ribosome.target_abundance > 0.0);
    assert!(end_ribosome.assembly_rate > 0.0);
    assert!(end_ribosome.nucleation_rate > 0.0);
    assert!(end_ribosome.elongation_rate > 0.0);
    assert!(end_ribosome.maturation_rate > 0.0);
    assert!(end_ribosome.abundance != start_ribosome.abundance);
    assert!(end_ribosome.assembly_progress != start_ribosome.assembly_progress);
    assert!(end_ribosome.failure_count >= start_ribosome.failure_count);

    let saved = sim.save_state_json().expect("serialize saved state");
    let restored = WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");
    let restored_named = restored.named_complexes_state();
    let restored_ribosome = restored_named
        .iter()
        .find(|state| state.id == "ribosome_biogenesis_operon_complex")
        .expect("restored ribosome complex state");

    assert_eq!(restored_named.len(), end_named.len());
    assert!((restored_ribosome.subunit_pool - end_ribosome.subunit_pool).abs() < 1.0e-6);
    assert!(
        (restored_ribosome.nucleation_intermediate - end_ribosome.nucleation_intermediate).abs()
            < 1.0e-6
    );
    assert!(
        (restored_ribosome.elongation_intermediate - end_ribosome.elongation_intermediate).abs()
            < 1.0e-6
    );
    assert!((restored_ribosome.abundance - end_ribosome.abundance).abs() < 1.0e-6);
    assert!((restored_ribosome.target_abundance - end_ribosome.target_abundance).abs() < 1.0e-6);
    assert!(
        (restored_ribosome.shared_component_pressure - end_ribosome.shared_component_pressure)
            .abs()
            < 1.0e-6
    );
    assert!(
        (restored_ribosome.limiting_component_signal - end_ribosome.limiting_component_signal)
            .abs()
            < 1.0e-6
    );
}

#[test]
fn test_named_complex_state_tracks_stall_and_damage_under_stress() {
    let mut baseline =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let mut stressed =
        WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    stressed.metabolic_load = 2.5;
    stressed.chemistry_report.atp_support = 0.62;
    stressed.chemistry_report.translation_support = 0.60;
    stressed.chemistry_report.membrane_support = 0.58;
    stressed.chemistry_report.crowding_penalty = 0.72;

    baseline.run(16);
    stressed.run(16);

    let stressed_complex = stressed
        .named_complexes_state()
        .into_iter()
        .find(|state| {
            matches!(
                state.family,
                WholeCellAssemblyFamily::AtpSynthase
                    | WholeCellAssemblyFamily::Transporter
                    | WholeCellAssemblyFamily::MembraneEnzyme
                    | WholeCellAssemblyFamily::Divisome
            )
        })
        .expect("stressed membrane-coupled complex");
    let baseline_complex = baseline
        .named_complexes_state()
        .into_iter()
        .find(|state| state.id == stressed_complex.id)
        .expect("baseline matching complex");

    assert!(stressed_complex.stalled_intermediate >= baseline_complex.stalled_intermediate);
    assert!(
        stressed_complex.shared_component_pressure >= baseline_complex.shared_component_pressure
    );
    assert!(stressed_complex.failure_count >= baseline_complex.failure_count);
    assert!(stressed_complex.insertion_progress <= baseline_complex.insertion_progress + 1.0e-6);
}

#[test]
fn test_named_complex_target_tracks_component_capacity_over_static_prior() {
    let package = WholeCellGenomeAssetPackage {
        organism: "Capacity-demo".to_string(),
        chromosome_length_bp: 800,
        origin_bp: 0,
        terminus_bp: 400,
        chromosome_domains: Vec::new(),
        operons: vec![WholeCellOperonSpec {
            name: "assembly_operon".to_string(),
            genes: vec!["gene_a".to_string()],
            promoter_bp: 10,
            terminator_bp: 120,
            basal_activity: 1.0,
            polycistronic: false,
            process_weights: WholeCellProcessWeights {
                translation: 1.0,
                ..WholeCellProcessWeights::default()
            },
            subsystem_targets: Vec::new(),
            asset_class: Some(WholeCellAssetClass::Translation),
            complex_family: Some(WholeCellAssemblyFamily::Generic),
        }],
        operon_semantics: Vec::new(),
        rnas: Vec::new(),
        proteins: vec![WholeCellProteinProductSpec {
            id: "protein_a".to_string(),
            gene: "gene_a".to_string(),
            operon: "assembly_operon".to_string(),
            rna_id: "rna_a".to_string(),
            aa_length: 220,
            basal_abundance: 4.0,
            translation_cost: 1.0,
            nucleotide_cost: 1.0,
            asset_class: WholeCellAssetClass::Translation,
            process_weights: WholeCellProcessWeights {
                translation: 1.0,
                ..WholeCellProcessWeights::default()
            },
            subsystem_targets: Vec::new(),
        }],
        protein_semantics: Vec::new(),
        complex_semantics: Vec::new(),
        complexes: vec![WholeCellComplexSpec {
            id: "assembly_complex".to_string(),
            name: "assembly complex".to_string(),
            operon: "assembly_operon".to_string(),
            components: vec![WholeCellComplexComponentSpec {
                protein_id: "protein_a".to_string(),
                stoichiometry: 2,
            }],
            basal_abundance: 128.0,
            asset_class: WholeCellAssetClass::Translation,
            family: WholeCellAssemblyFamily::Generic,
            process_weights: WholeCellProcessWeights {
                translation: 1.0,
                ..WholeCellProcessWeights::default()
            },
            subsystem_targets: Vec::new(),
            membrane_inserted: false,
            chromosome_coupled: false,
            division_coupled: false,
        }],
        pools: Vec::new(),
    };

    let build_sim = |protein_abundance: f32| {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        sim.organism_assets = Some(package.clone());
        sim.organism_expression.transcription_units = vec![WholeCellTranscriptionUnitState {
            name: "assembly_operon".to_string(),
            gene_count: 1,
            copy_gain: 1.0,
            basal_activity: 1.0,
            effective_activity: 1.0,
            support_level: 1.0,
            stress_penalty: 1.0,
            transcript_abundance: protein_abundance,
            protein_abundance,
            transcript_synthesis_rate: 0.0,
            protein_synthesis_rate: 0.0,
            transcript_turnover_rate: 0.0,
            protein_turnover_rate: 0.0,
            promoter_open_fraction: 0.0,
            active_rnap_occupancy: 0.0,
            transcription_length_nt: 660.0,
            transcription_progress_nt: 0.0,
            nascent_transcript_abundance: 0.0,
            mature_transcript_abundance: protein_abundance,
            damaged_transcript_abundance: 0.0,
            active_ribosome_occupancy: 0.0,
            translation_length_aa: 220.0,
            translation_progress_aa: 0.0,
            nascent_protein_abundance: 0.0,
            mature_protein_abundance: protein_abundance,
            damaged_protein_abundance: 0.0,
            process_drive: WholeCellProcessWeights {
                translation: 1.0,
                ..WholeCellProcessWeights::default()
            },
        }];
        sim.refresh_expression_inventory_totals();
        assert!(sim.initialize_named_complexes_state());
        sim.named_complexes_state()
            .into_iter()
            .find(|state| state.id == "assembly_complex")
            .expect("assembly complex state")
    };

    let low = build_sim(4.0);
    let high = build_sim(40.0);

    assert!(high.target_abundance > low.target_abundance);
    assert!(high.subunit_pool > low.subunit_pool);
    assert!(low.target_abundance < 16.0);
}

#[test]
fn test_legacy_derived_complex_targets_prefer_persistent_complex_inventory() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    sim.atp_mm = 1.4;
    sim.amino_acids_mm = 1.3;
    sim.nucleotides_mm = 1.2;
    sim.membrane_precursors_mm = 1.1;
    sim.glucose_mm = 1.0;
    sim.oxygen_mm = 1.0;
    sim.md_translation_scale = 1.2;
    sim.md_membrane_scale = 1.1;
    sim.complex_assembly = WholeCellComplexAssemblyState {
        atp_band_complexes: 5.0,
        ribosome_complexes: 19.0,
        rnap_complexes: 11.0,
        replisome_complexes: 7.0,
        membrane_complexes: 13.0,
        ftsz_polymer: 17.0,
        dnaa_activity: 9.0,
        ..WholeCellComplexAssemblyState::default()
    };

    let scalar_prior = sim.prior_assembly_inventory();
    let target = sim.derived_complex_assembly_target();

    assert!((target.ribosome_complexes - 19.0).abs() < 1.0e-6);
    assert!((target.rnap_complexes - 11.0).abs() < 1.0e-6);
    assert!((target.replisome_complexes - 7.0).abs() < 1.0e-6);
    assert!((target.membrane_complexes - 13.0).abs() < 1.0e-6);
    assert!((target.ftsz_polymer - 17.0).abs() < 1.0e-6);
    assert!((scalar_prior.ribosome_complexes - target.ribosome_complexes).abs() > 1.0e-3);
    assert!((scalar_prior.membrane_complexes - target.membrane_complexes).abs() > 1.0e-3);
    assert!(target.ribosome_target > 0.0);
    assert!(target.membrane_target > 0.0);
}

#[test]
fn test_process_capacity_tracks_persistent_complex_assembly_state() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    sim.refresh_organism_expression_state();
    sim.initialize_complex_assembly_state();

    let baseline_inventory = sim.assembly_inventory();
    let baseline_fluxes = sim.process_fluxes(baseline_inventory);

    for state in &mut sim.named_complexes {
        state.subunit_pool *= 0.10;
        state.nucleation_intermediate *= 0.10;
        state.elongation_intermediate *= 0.10;
        state.abundance *= 0.10;
        state.target_abundance *= 0.10;
    }
    let assets = sim
        .organism_assets
        .clone()
        .expect("compiled genome asset package");
    sim.complex_assembly = sim.aggregate_named_complex_assembly_state(&assets);
    let reduced_inventory = sim.assembly_inventory();
    let reduced_fluxes = sim.process_fluxes(reduced_inventory);

    assert!(reduced_inventory.ribosome_complexes < baseline_inventory.ribosome_complexes);
    assert!(reduced_inventory.replisome_complexes < baseline_inventory.replisome_complexes);
    assert!(reduced_inventory.ftsz_polymer < baseline_inventory.ftsz_polymer);
    assert!(reduced_fluxes.translation_capacity <= baseline_fluxes.translation_capacity);
    assert!(reduced_fluxes.transcription_capacity <= baseline_fluxes.transcription_capacity);
    assert!(reduced_fluxes.replication_capacity <= baseline_fluxes.replication_capacity);
}

#[test]
fn test_process_fluxes_follow_explicit_channel_inventory() {
    let mut sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    sim.atp_mm = 1.4;
    sim.amino_acids_mm = 1.2;
    sim.nucleotides_mm = 1.3;
    sim.membrane_precursors_mm = 1.1;
    sim.oxygen_mm = 1.0;
    sim.glucose_mm = 1.0;
    sim.chemistry_report.atp_support = 1.0;
    sim.chemistry_report.translation_support = 1.0;
    sim.chemistry_report.nucleotide_support = 1.0;
    sim.chemistry_report.membrane_support = 1.0;

    let translation_fluxes = sim.process_fluxes(WholeCellAssemblyInventory {
        ribosome_complexes: 96.0,
        ..WholeCellAssemblyInventory::default()
    });
    let transcription_fluxes = sim.process_fluxes(WholeCellAssemblyInventory {
        rnap_complexes: 72.0,
        ..WholeCellAssemblyInventory::default()
    });
    let replication_fluxes = sim.process_fluxes(WholeCellAssemblyInventory {
        replisome_complexes: 56.0,
        ..WholeCellAssemblyInventory::default()
    });

    assert!(translation_fluxes.translation_capacity > transcription_fluxes.translation_capacity);
    assert!(
        translation_fluxes.transcription_capacity <= transcription_fluxes.transcription_capacity
    );
    assert!(replication_fluxes.replication_capacity > translation_fluxes.replication_capacity);
    assert!((translation_fluxes.transcription_capacity - 0.35).abs() < 1.0e-6);
    assert!((transcription_fluxes.translation_capacity - 0.35).abs() < 1.0e-6);
}

#[test]
fn test_cme_stage_follows_explicit_inventory_channels() {
    let mut translation = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    let mut transcription = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });

    for sim in [&mut translation, &mut transcription] {
        sim.atp_mm = 1.4;
        sim.amino_acids_mm = 1.2;
        sim.nucleotides_mm = 1.3;
        sim.membrane_precursors_mm = 1.1;
        sim.oxygen_mm = 1.0;
        sim.glucose_mm = 1.0;
        sim.active_ribosomes = 16.0;
        sim.active_rnap = 16.0;
        sim.chemistry_report.atp_support = 1.0;
        sim.chemistry_report.translation_support = 1.0;
        sim.chemistry_report.nucleotide_support = 1.0;
        sim.chemistry_report.membrane_support = 1.0;
    }

    translation.complex_assembly = WholeCellAssemblyInventory {
        ribosome_complexes: 96.0,
        ribosome_target: 96.0,
        ..WholeCellAssemblyInventory::default()
    };
    transcription.complex_assembly = WholeCellAssemblyInventory {
        rnap_complexes: 72.0,
        rnap_target: 72.0,
        ..WholeCellAssemblyInventory::default()
    };

    translation.cme_stage(1.0);
    transcription.cme_stage(1.0);

    assert!(translation.active_ribosomes > transcription.active_ribosomes);
    assert!(transcription.active_rnap > translation.active_rnap);
}

#[test]
fn test_assembly_inventory_prefers_named_complex_state_when_assets_exist() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    let assets = sim
        .organism_assets
        .clone()
        .expect("compiled genome asset package");
    sim.initialize_complex_assembly_state();
    for state in &mut sim.named_complexes {
        state.subunit_pool *= 0.25;
        state.nucleation_intermediate *= 0.15;
        state.elongation_intermediate *= 0.20;
        state.abundance *= 0.30;
        state.target_abundance *= 0.35;
    }
    let expected = sim.aggregate_named_complex_assembly_state(&assets);
    sim.complex_assembly = WholeCellComplexAssemblyState::default();

    let inventory = sim.assembly_inventory();

    assert!((inventory.ribosome_complexes - expected.ribosome_complexes).abs() < 1.0e-6);
    assert!((inventory.rnap_complexes - expected.rnap_complexes).abs() < 1.0e-6);
    assert!((inventory.replisome_complexes - expected.replisome_complexes).abs() < 1.0e-6);
    assert!((inventory.ftsz_target - expected.ftsz_target).abs() < 1.0e-6);
}

#[test]
fn test_named_complex_aggregation_prefers_explicit_family_channels() {
    let sim = WholeCellSimulator::new(WholeCellConfig {
        use_gpu: false,
        ..WholeCellConfig::default()
    });
    let assets = WholeCellGenomeAssetPackage {
        organism: "Aggregation-demo".to_string(),
        chromosome_length_bp: 800,
        origin_bp: 0,
        terminus_bp: 400,
        chromosome_domains: Vec::new(),
        operons: Vec::new(),
        operon_semantics: Vec::new(),
        rnas: Vec::new(),
        proteins: Vec::new(),
        protein_semantics: Vec::new(),
        complex_semantics: Vec::new(),
        complexes: vec![
            WholeCellComplexSpec {
                id: "ribosome_complex".to_string(),
                name: "ribosome complex".to_string(),
                operon: "ribosome_operon".to_string(),
                components: Vec::new(),
                basal_abundance: 12.0,
                asset_class: WholeCellAssetClass::Replication,
                family: WholeCellAssemblyFamily::Ribosome,
                process_weights: WholeCellProcessWeights {
                    replication: 5.0,
                    segregation: 3.0,
                    ..WholeCellProcessWeights::default()
                },
                subsystem_targets: Vec::new(),
                membrane_inserted: false,
                chromosome_coupled: false,
                division_coupled: false,
            },
            WholeCellComplexSpec {
                id: "rnap_complex".to_string(),
                name: "rnap complex".to_string(),
                operon: "rnap_operon".to_string(),
                components: Vec::new(),
                basal_abundance: 6.0,
                asset_class: WholeCellAssetClass::QualityControl,
                family: WholeCellAssemblyFamily::RnaPolymerase,
                process_weights: WholeCellProcessWeights {
                    replication: 4.0,
                    translation: 2.0,
                    ..WholeCellProcessWeights::default()
                },
                subsystem_targets: Vec::new(),
                membrane_inserted: false,
                chromosome_coupled: false,
                division_coupled: false,
            },
            WholeCellComplexSpec {
                id: "divisome_complex".to_string(),
                name: "divisome complex".to_string(),
                operon: "division_operon".to_string(),
                components: Vec::new(),
                basal_abundance: 10.0,
                asset_class: WholeCellAssetClass::Energy,
                family: WholeCellAssemblyFamily::Divisome,
                process_weights: WholeCellProcessWeights {
                    energy: 4.0,
                    membrane: 2.0,
                    ..WholeCellProcessWeights::default()
                },
                subsystem_targets: Vec::new(),
                membrane_inserted: false,
                chromosome_coupled: false,
                division_coupled: false,
            },
            WholeCellComplexSpec {
                id: "dnaa_complex".to_string(),
                name: "dnaa complex".to_string(),
                operon: "dnaa_operon".to_string(),
                components: Vec::new(),
                basal_abundance: 5.0,
                asset_class: WholeCellAssetClass::Replication,
                family: WholeCellAssemblyFamily::ReplicationInitiator,
                process_weights: WholeCellProcessWeights {
                    replication: 3.0,
                    segregation: 1.0,
                    ..WholeCellProcessWeights::default()
                },
                subsystem_targets: Vec::new(),
                membrane_inserted: false,
                chromosome_coupled: true,
                division_coupled: false,
            },
        ],
        pools: Vec::new(),
    };
    let mut sim = sim;
    sim.named_complexes = vec![
        WholeCellNamedComplexState {
            id: "ribosome_complex".to_string(),
            operon: "ribosome_operon".to_string(),
            asset_class: WholeCellAssetClass::Replication,
            family: WholeCellAssemblyFamily::Ribosome,
            subsystem_targets: Vec::new(),
            subunit_pool: 0.0,
            nucleation_intermediate: 0.0,
            elongation_intermediate: 0.0,
            abundance: 10.0,
            target_abundance: 12.0,
            assembly_rate: 3.0,
            degradation_rate: 1.0,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 1.0,
            structural_support: 1.0,
            assembly_progress: 1.0,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 1.0,
            shared_component_pressure: 0.0,
            insertion_progress: 1.0,
            failure_count: 0.0,
        },
        WholeCellNamedComplexState {
            id: "rnap_complex".to_string(),
            operon: "rnap_operon".to_string(),
            asset_class: WholeCellAssetClass::QualityControl,
            family: WholeCellAssemblyFamily::RnaPolymerase,
            subsystem_targets: Vec::new(),
            subunit_pool: 0.0,
            nucleation_intermediate: 0.0,
            elongation_intermediate: 0.0,
            abundance: 6.0,
            target_abundance: 7.0,
            assembly_rate: 1.5,
            degradation_rate: 0.25,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 1.0,
            structural_support: 1.0,
            assembly_progress: 1.0,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 1.0,
            shared_component_pressure: 0.0,
            insertion_progress: 1.0,
            failure_count: 0.0,
        },
        WholeCellNamedComplexState {
            id: "divisome_complex".to_string(),
            operon: "division_operon".to_string(),
            asset_class: WholeCellAssetClass::Energy,
            family: WholeCellAssemblyFamily::Divisome,
            subsystem_targets: Vec::new(),
            subunit_pool: 0.0,
            nucleation_intermediate: 0.0,
            elongation_intermediate: 0.0,
            abundance: 8.0,
            target_abundance: 9.0,
            assembly_rate: 2.0,
            degradation_rate: 0.5,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 1.0,
            structural_support: 1.0,
            assembly_progress: 1.0,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 1.0,
            shared_component_pressure: 0.0,
            insertion_progress: 1.0,
            failure_count: 0.0,
        },
        WholeCellNamedComplexState {
            id: "dnaa_complex".to_string(),
            operon: "dnaa_operon".to_string(),
            asset_class: WholeCellAssetClass::Replication,
            family: WholeCellAssemblyFamily::ReplicationInitiator,
            subsystem_targets: Vec::new(),
            subunit_pool: 0.0,
            nucleation_intermediate: 0.0,
            elongation_intermediate: 0.0,
            abundance: 5.0,
            target_abundance: 6.0,
            assembly_rate: 1.2,
            degradation_rate: 0.3,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 1.0,
            structural_support: 1.0,
            assembly_progress: 1.0,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 1.0,
            shared_component_pressure: 0.0,
            insertion_progress: 1.0,
            failure_count: 0.0,
        },
    ];

    let aggregate = sim.aggregate_named_complex_assembly_state(&assets);

    assert!((aggregate.ribosome_complexes - 10.0).abs() < 1.0e-6);
    assert!((aggregate.ribosome_target - 12.0).abs() < 1.0e-6);
    assert!((aggregate.ribosome_assembly_rate - 3.0).abs() < 1.0e-6);
    assert!((aggregate.ribosome_degradation_rate - 1.0).abs() < 1.0e-6);
    assert!((aggregate.rnap_complexes - 6.0).abs() < 1.0e-6);
    assert!((aggregate.rnap_target - 7.0).abs() < 1.0e-6);
    assert!((aggregate.rnap_assembly_rate - 1.5).abs() < 1.0e-6);
    assert!((aggregate.rnap_degradation_rate - 0.25).abs() < 1.0e-6);
    assert!((aggregate.ftsz_polymer - 8.0).abs() < 1.0e-6);
    assert!((aggregate.ftsz_target - 9.0).abs() < 1.0e-6);
    assert!((aggregate.ftsz_assembly_rate - 2.0).abs() < 1.0e-6);
    assert!((aggregate.ftsz_degradation_rate - 0.5).abs() < 1.0e-6);
    assert!((aggregate.dnaa_activity - 5.0).abs() < 1.0e-6);
    assert!((aggregate.dnaa_target - 6.0).abs() < 1.0e-6);
    assert!((aggregate.dnaa_assembly_rate - 1.2).abs() < 1.0e-6);
    assert!((aggregate.dnaa_degradation_rate - 0.3).abs() < 1.0e-6);
    assert!(aggregate.replisome_complexes.abs() < 1.0e-6);
    assert!(aggregate.atp_band_complexes.abs() < 1.0e-6);
}

#[test]
fn test_explicit_asset_inventory_does_not_fall_back_to_derived_targets() {
    let mut sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
    sim.named_complexes.clear();
    sim.complex_assembly = WholeCellComplexAssemblyState {
        atp_band_complexes: 3.0,
        ribosome_complexes: 17.0,
        rnap_complexes: 11.0,
        replisome_complexes: 5.0,
        membrane_complexes: 9.0,
        ftsz_polymer: 13.0,
        dnaa_activity: 7.0,
        atp_band_target: 4.0,
        ribosome_target: 18.0,
        rnap_target: 12.0,
        replisome_target: 6.0,
        membrane_target: 10.0,
        ftsz_target: 14.0,
        dnaa_target: 8.0,
        ..WholeCellComplexAssemblyState::default()
    };

    let inventory = sim.assembly_inventory();

    assert!((inventory.atp_band_complexes - 3.0).abs() < 1.0e-6);
    assert!((inventory.ribosome_complexes - 17.0).abs() < 1.0e-6);
    assert!((inventory.rnap_complexes - 11.0).abs() < 1.0e-6);
    assert!((inventory.replisome_complexes - 5.0).abs() < 1.0e-6);
    assert!((inventory.membrane_complexes - 9.0).abs() < 1.0e-6);
    assert!((inventory.ftsz_polymer - 13.0).abs() < 1.0e-6);
    assert!((inventory.dnaa_activity - 7.0).abs() < 1.0e-6);
    assert!((inventory.ribosome_target - 18.0).abs() < 1.0e-6);
    assert!((inventory.ftsz_target - 14.0).abs() < 1.0e-6);
}

#[test]
fn test_organism_descriptor_drives_division_and_replication_scales() {
    let baseline_spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
    let mut constrained_spec = baseline_spec.clone();
    let organism = constrained_spec
        .organism_data
        .as_mut()
        .expect("bundled organism data");
    for gene in &mut organism.genes {
        gene.process_weights.replication *= 0.35;
        gene.process_weights.segregation *= 0.35;
        gene.process_weights.membrane *= 0.40;
        gene.process_weights.constriction *= 0.30;
    }
    for unit in &mut organism.transcription_units {
        unit.process_weights.replication *= 0.35;
        unit.process_weights.segregation *= 0.35;
        unit.process_weights.membrane *= 0.40;
        unit.process_weights.constriction *= 0.30;
    }

    let mut baseline = WholeCellSimulator::from_program_spec(baseline_spec);
    let mut constrained = WholeCellSimulator::from_program_spec(constrained_spec);
    let baseline_expression = baseline
        .organism_expression_state()
        .expect("baseline organism expression");
    let constrained_expression = constrained
        .organism_expression_state()
        .expect("constrained organism expression");
    let baseline_complex = baseline.complex_assembly_state();
    let constrained_complex = constrained.complex_assembly_state();

    assert!(
        baseline_expression.process_scales.replication
            > constrained_expression.process_scales.replication
    );
    assert!(
        baseline_expression.process_scales.membrane
            > constrained_expression.process_scales.membrane
    );
    assert!(
        baseline_expression.process_scales.constriction
            > constrained_expression.process_scales.constriction
    );
    assert!(baseline_complex.replisome_target > constrained_complex.replisome_target);
    assert!(baseline_complex.ftsz_target > constrained_complex.ftsz_target);

    baseline.run(96);
    constrained.run(96);

    let baseline_snapshot = baseline.snapshot();
    let constrained_snapshot = constrained.snapshot();
    let baseline_membrane = baseline.membrane_division_state();
    let constrained_membrane = constrained.membrane_division_state();

    assert!(baseline_snapshot.replicated_bp >= constrained_snapshot.replicated_bp);
    assert!(baseline_membrane.divisome_occupancy > constrained_membrane.divisome_occupancy);
    assert!(baseline_membrane.ring_tension > constrained_membrane.ring_tension);
    assert!(baseline_snapshot.division_progress >= constrained_snapshot.division_progress);
    assert!(baseline_snapshot.surface_area_nm2 > constrained_snapshot.surface_area_nm2);
}

#[test]
fn test_from_program_spec_preserves_missing_assets_and_registry() {
    let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
    spec.organism_data_ref = None;
    spec.organism_assets = None;
    spec.organism_process_registry = None;

    let simulator = WholeCellSimulator::from_program_spec(spec);

    assert!(simulator.organism_data.is_some());
    assert!(simulator.organism_assets.is_none());
    assert!(simulator.organism_process_registry.is_none());
}

#[test]
fn test_from_program_spec_json_preserves_missing_assets_and_registry() {
    let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
    spec.organism_data_ref = None;
    spec.organism_assets = None;
    spec.organism_process_registry = None;

    let simulator = WholeCellSimulator::from_program_spec_json(
        &serde_json::to_string(&spec).expect("serialize explicit program spec"),
    )
    .expect("restore explicit program spec");

    assert!(simulator.organism_data.is_some());
    assert!(simulator.organism_assets.is_none());
    assert!(simulator.organism_process_registry.is_none());
    assert!(simulator.organism_process_registry().is_none());
}

#[test]
fn test_from_program_spec_preserves_explicit_complex_assembly_without_assets() {
    let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
    spec.organism_data_ref = None;
    spec.organism_assets = None;
    spec.organism_process_registry = None;
    spec.named_complexes.clear();
    spec.complex_assembly = Some(WholeCellComplexAssemblyState {
        ribosome_complexes: 17.0,
        rnap_complexes: 10.0,
        dnaa_activity: 8.5,
        ftsz_polymer: 24.0,
        ..WholeCellComplexAssemblyState::default()
    });

    let simulator = WholeCellSimulator::from_program_spec(spec);
    let snapshot = simulator.snapshot();

    assert!(simulator.organism_assets.is_none());
    assert!((snapshot.active_rnap - 10.0).abs() < 1.0e-6);
    assert!((snapshot.active_ribosomes - 17.0).abs() < 1.0e-6);
    assert!((snapshot.dnaa - 8.5).abs() < 1.0e-6);
    assert!((snapshot.ftsz - 24.0).abs() < 1.0e-6);
}

#[test]
fn test_from_program_spec_preserves_explicit_expression_state() {
    let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
    let mut expression = WholeCellOrganismExpressionState::default();
    expression.global_activity = 1.35;
    expression.energy_support = 1.10;
    expression.translation_support = 0.92;
    expression.nucleotide_support = 1.18;
    expression.membrane_support = 0.87;
    expression.crowding_penalty = 0.94;
    expression.metabolic_burden_scale = 1.08;
    expression.process_scales = WholeCellProcessWeights {
        energy: 1.12,
        transcription: 0.88,
        translation: 1.06,
        replication: 0.79,
        segregation: 0.83,
        membrane: 0.91,
        constriction: 0.86,
    };
    expression.amino_cost_scale = 1.11;
    expression.nucleotide_cost_scale = 0.93;
    expression.total_transcript_abundance = 42.0;
    expression.total_protein_abundance = 84.0;
    expression.transcription_units = vec![WholeCellTranscriptionUnitState {
        name: "synthetic_unit".to_string(),
        gene_count: 2,
        copy_gain: 1.10,
        basal_activity: 1.00,
        effective_activity: 1.25,
        support_level: 1.05,
        stress_penalty: 0.95,
        transcript_abundance: 7.0,
        protein_abundance: 12.0,
        transcript_synthesis_rate: 0.14,
        protein_synthesis_rate: 0.12,
        transcript_turnover_rate: 0.03,
        protein_turnover_rate: 0.02,
        promoter_open_fraction: 0.45,
        active_rnap_occupancy: 0.30,
        transcription_length_nt: 120.0,
        transcription_progress_nt: 40.0,
        nascent_transcript_abundance: 1.5,
        mature_transcript_abundance: 5.5,
        damaged_transcript_abundance: 0.2,
        active_ribosome_occupancy: 0.35,
        translation_length_aa: 45.0,
        translation_progress_aa: 14.0,
        nascent_protein_abundance: 2.0,
        mature_protein_abundance: 10.0,
        damaged_protein_abundance: 0.3,
        process_drive: WholeCellProcessWeights {
            energy: 0.9,
            transcription: 1.0,
            translation: 1.1,
            replication: 0.7,
            segregation: 0.7,
            membrane: 0.8,
            constriction: 0.75,
        },
    }];
    spec.organism_expression = Some(expression.clone());

    let simulator = WholeCellSimulator::from_program_spec(spec);
    let restored = simulator
        .organism_expression_state()
        .expect("restored organism expression");

    assert!((restored.global_activity - expression.global_activity).abs() < 1.0e-6);
    assert!((restored.translation_support - expression.translation_support).abs() < 1.0e-6);
    assert!(
        (restored.process_scales.replication - expression.process_scales.replication).abs()
            < 1.0e-6
    );
    assert!(
        (restored.total_transcript_abundance - expression.total_transcript_abundance).abs()
            < 1.0e-6
    );
    assert_eq!(restored.transcription_units.len(), 1);
    assert_eq!(restored.transcription_units[0].name, "synthetic_unit");
    assert!((restored.transcription_units[0].transcript_abundance - 7.0).abs() < 1.0e-6);
}

#[test]
fn test_from_program_spec_preserves_explicit_runtime_process_and_scheduler_state() {
    let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
    let seeded = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");

    let mut species = seeded.organism_species.clone();
    species[0].count = 123.0;
    species[0].anchor_count = 17.0;
    species[0].synthesis_rate = 0.44;
    species[0].turnover_rate = 0.12;

    let mut reactions = seeded.organism_reactions.clone();
    reactions[0].current_flux = 0.81;
    reactions[0].cumulative_extent = 4.5;
    reactions[0].reactant_satisfaction = 0.73;
    reactions[0].catalyst_support = 1.24;

    let mut scheduler_state = seeded.scheduler_state.clone();
    let cme_clock = scheduler_state
        .stage_clocks
        .iter_mut()
        .find(|clock| clock.stage == WholeCellSolverStage::Cme)
        .expect("CME clock");
    cme_clock.dynamic_interval_steps = 7;
    cme_clock.next_due_step = 19;
    cme_clock.run_count = 3;
    cme_clock.last_run_step = Some(12);
    cme_clock.last_run_time_ms = 6.5;

    spec.organism_species = Some(species.clone());
    spec.organism_reactions = Some(reactions.clone());
    spec.scheduler_state = Some(scheduler_state.clone());

    let simulator = WholeCellSimulator::from_program_spec(spec);
    let restored_cme_clock = simulator
        .scheduler_clock(WholeCellSolverStage::Cme)
        .expect("restored CME clock");

    assert_eq!(simulator.organism_species.len(), species.len());
    assert_eq!(simulator.organism_reactions.len(), reactions.len());
    assert_eq!(simulator.organism_species[0].id, species[0].id);
    assert!((simulator.organism_species[0].count - 123.0).abs() < 1.0e-6);
    assert!((simulator.organism_species[0].anchor_count - 17.0).abs() < 1.0e-6);
    assert!((simulator.organism_species[0].synthesis_rate - 0.44).abs() < 1.0e-6);
    assert!((simulator.organism_species[0].turnover_rate - 0.12).abs() < 1.0e-6);
    assert_eq!(simulator.organism_reactions[0].id, reactions[0].id);
    assert!((simulator.organism_reactions[0].current_flux - 0.81).abs() < 1.0e-6);
    assert!((simulator.organism_reactions[0].cumulative_extent - 4.5).abs() < 1.0e-6);
    assert!((simulator.organism_reactions[0].reactant_satisfaction - 0.73).abs() < 1.0e-6);
    assert!((simulator.organism_reactions[0].catalyst_support - 1.24).abs() < 1.0e-6);
    assert_eq!(restored_cme_clock.dynamic_interval_steps, 7);
    assert_eq!(restored_cme_clock.next_due_step, 19);
    assert_eq!(restored_cme_clock.run_count, 3);
    assert_eq!(restored_cme_clock.last_run_step, Some(12));
    assert!((restored_cme_clock.last_run_time_ms - 6.5).abs() < 1.0e-6);
}

#[test]
fn test_from_program_spec_preserves_explicit_local_chemistry_runtime_state() {
    let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
    spec.local_chemistry = Some(WholeCellLocalChemistrySpec {
        x_dim: 12,
        y_dim: 12,
        z_dim: 6,
        voxel_size_au: 0.5,
        use_gpu: false,
        enable_default_syn3a_subsystems: false,
        scheduled_subsystem_probes: Vec::new(),
    });
    spec.chemistry_report = Some(LocalChemistryReport {
        atp_support: 0.82,
        translation_support: 0.91,
        nucleotide_support: 0.87,
        membrane_support: 1.08,
        crowding_penalty: 0.79,
        mean_glucose: 0.33,
        mean_oxygen: 0.27,
        mean_atp_flux: 0.41,
        mean_carbon_dioxide: 0.18,
    });
    spec.chemistry_site_reports = vec![LocalChemistrySiteReport {
        preset: Syn3ASubsystemPreset::ReplisomeTrack,
        site: WholeCellChemistrySite::ChromosomeTrack,
        patch_radius: 2,
        site_x: 4,
        site_y: 5,
        site_z: 1,
        localization_score: 0.86,
        atp_support: 0.78,
        translation_support: 0.74,
        nucleotide_support: 0.92,
        membrane_support: 0.66,
        crowding_penalty: 0.83,
        mean_glucose: 0.25,
        mean_oxygen: 0.21,
        mean_atp_flux: 0.38,
        mean_carbon_dioxide: 0.19,
        mean_nitrate: 0.12,
        mean_ammonium: 0.09,
        mean_proton: 0.04,
        mean_phosphorus: 0.07,
        assembly_component_availability: 0.91,
        assembly_occupancy: 0.64,
        assembly_stability: 0.72,
        assembly_turnover: 0.18,
        substrate_draw: 0.26,
        energy_draw: 0.31,
        biosynthetic_draw: 0.22,
        byproduct_load: 0.14,
        demand_satisfaction: 0.77,
    }];
    spec.last_md_probe = Some(LocalMDProbeReport {
        site: WholeCellChemistrySite::ChromosomeTrack,
        mean_temperature: 301.0,
        mean_total_energy: -18.0,
        mean_vdw_energy: -8.0,
        mean_electrostatic_energy: -5.0,
        structural_order: 0.84,
        crowding_penalty: 0.81,
        compactness: 0.62,
        shell_order: 0.58,
        axis_anisotropy: 0.19,
        thermal_stability: 0.87,
        electrostatic_order: 0.74,
        vdw_cohesion: 0.71,
        polar_fraction: 0.44,
        phosphate_fraction: 0.16,
        hydrogen_fraction: 0.22,
        bond_density: 0.52,
        angle_density: 0.47,
        dihedral_density: 0.29,
        charge_density: 0.12,
        recommended_atp_scale: 1.05,
        recommended_translation_scale: 1.18,
        recommended_replication_scale: 1.09,
        recommended_segregation_scale: 1.04,
        recommended_membrane_scale: 0.93,
        recommended_constriction_scale: 0.97,
    });
    spec.scheduled_subsystem_probes = vec![ScheduledSubsystemProbe {
        preset: Syn3ASubsystemPreset::ReplisomeTrack,
        interval_steps: 9,
    }];
    spec.subsystem_states = vec![WholeCellSubsystemState {
        preset: Syn3ASubsystemPreset::ReplisomeTrack,
        site: WholeCellChemistrySite::ChromosomeTrack,
        site_x: 4,
        site_y: 5,
        site_z: 1,
        localization_score: 0.86,
        structural_order: 0.79,
        crowding_penalty: 0.81,
        assembly_component_availability: 0.88,
        assembly_occupancy: 0.63,
        assembly_stability: 0.69,
        assembly_turnover: 0.21,
        substrate_draw: 0.24,
        energy_draw: 0.29,
        biosynthetic_draw: 0.20,
        byproduct_load: 0.13,
        demand_satisfaction: 0.75,
        atp_scale: 1.06,
        translation_scale: 1.17,
        replication_scale: 1.11,
        segregation_scale: 1.03,
        membrane_scale: 0.95,
        constriction_scale: 0.98,
        last_probe_step: Some(14),
    }];
    spec.md_translation_scale = Some(1.18);
    spec.md_membrane_scale = Some(0.93);

    let simulator = WholeCellSimulator::from_program_spec(spec);
    let restored_report = simulator
        .local_chemistry_report()
        .expect("restored local chemistry report");
    let restored_site_reports = simulator.local_chemistry_sites();
    let restored_subsystem_states = simulator.subsystem_states();
    let restored_md_probe = simulator.last_md_probe().expect("restored md probe");
    let restored_probes = simulator.scheduled_syn3a_subsystem_probes();

    assert!((restored_report.atp_support - 0.82).abs() < 1.0e-6);
    assert!((restored_report.crowding_penalty - 0.79).abs() < 1.0e-6);
    assert_eq!(restored_site_reports.len(), 1);
    assert_eq!(
        restored_site_reports[0].preset,
        Syn3ASubsystemPreset::ReplisomeTrack
    );
    assert!((restored_site_reports[0].localization_score - 0.86).abs() < 1.0e-6);
    assert_eq!(restored_subsystem_states.len(), 1);
    assert_eq!(
        restored_subsystem_states[0].preset,
        Syn3ASubsystemPreset::ReplisomeTrack
    );
    assert_eq!(restored_subsystem_states[0].last_probe_step, Some(14));
    assert!((restored_subsystem_states[0].replication_scale - 1.11).abs() < 1.0e-6);
    assert_eq!(restored_probes.len(), 1);
    assert_eq!(restored_probes[0].interval_steps, 9);
    assert_eq!(
        restored_md_probe.site,
        WholeCellChemistrySite::ChromosomeTrack
    );
    assert!((simulator.md_translation_scale() - 1.18).abs() < 1.0e-6);
    assert!((simulator.md_membrane_scale() - 0.93).abs() < 1.0e-6);
}

#[test]
fn test_local_chemistry_getters_expose_explicit_state_without_bridge() {
    let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
    spec.local_chemistry = None;
    spec.chemistry_report = Some(LocalChemistryReport {
        atp_support: 0.84,
        translation_support: 0.88,
        nucleotide_support: 0.86,
        membrane_support: 0.93,
        crowding_penalty: 0.77,
        mean_glucose: 0.22,
        mean_oxygen: 0.18,
        mean_atp_flux: 0.36,
        mean_carbon_dioxide: 0.11,
    });
    spec.chemistry_site_reports = vec![LocalChemistrySiteReport {
        preset: Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
        site: WholeCellChemistrySite::AtpSynthaseBand,
        patch_radius: 2,
        site_x: 6,
        site_y: 3,
        site_z: 2,
        localization_score: 0.91,
        atp_support: 0.94,
        translation_support: 0.83,
        nucleotide_support: 0.80,
        membrane_support: 0.89,
        crowding_penalty: 0.78,
        mean_glucose: 0.29,
        mean_oxygen: 0.31,
        mean_atp_flux: 0.45,
        mean_carbon_dioxide: 0.12,
        mean_nitrate: 0.08,
        mean_ammonium: 0.07,
        mean_proton: 0.03,
        mean_phosphorus: 0.05,
        assembly_component_availability: 0.90,
        assembly_occupancy: 0.68,
        assembly_stability: 0.74,
        assembly_turnover: 0.16,
        substrate_draw: 0.20,
        energy_draw: 0.34,
        biosynthetic_draw: 0.18,
        byproduct_load: 0.10,
        demand_satisfaction: 0.81,
    }];
    spec.subsystem_states = vec![WholeCellSubsystemState {
        preset: Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
        site: WholeCellChemistrySite::AtpSynthaseBand,
        site_x: 6,
        site_y: 3,
        site_z: 2,
        localization_score: 0.91,
        structural_order: 0.82,
        crowding_penalty: 0.78,
        assembly_component_availability: 0.90,
        assembly_occupancy: 0.68,
        assembly_stability: 0.74,
        assembly_turnover: 0.16,
        substrate_draw: 0.20,
        energy_draw: 0.34,
        biosynthetic_draw: 0.18,
        byproduct_load: 0.10,
        demand_satisfaction: 0.81,
        atp_scale: 1.14,
        translation_scale: 1.03,
        replication_scale: 1.00,
        segregation_scale: 1.00,
        membrane_scale: 1.08,
        constriction_scale: 0.98,
        last_probe_step: Some(8),
    }];
    spec.md_translation_scale = Some(1.07);
    spec.md_membrane_scale = Some(1.09);

    let simulator = WholeCellSimulator::from_program_spec(spec);
    let report = simulator
        .local_chemistry_report()
        .expect("explicit chemistry report");
    let sites = simulator.local_chemistry_sites();
    let snapshot = simulator.snapshot();

    assert!(simulator.chemistry_bridge.is_none());
    assert!((report.atp_support - 0.84).abs() < 1.0e-6);
    assert_eq!(sites.len(), 1);
    assert_eq!(
        sites[0].preset,
        Syn3ASubsystemPreset::AtpSynthaseMembraneBand
    );
    assert!(snapshot.local_chemistry.is_some());
    assert_eq!(snapshot.local_chemistry_sites.len(), 1);
    assert!((simulator.md_translation_scale() - 1.07).abs() < 1.0e-6);
    assert!((simulator.md_membrane_scale() - 1.09).abs() < 1.0e-6);
}

#[test]
fn test_from_program_spec_preserves_explicit_spatial_fields() {
    let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
    let total_voxels = spec.config.x_dim * spec.config.y_dim * spec.config.z_dim;
    let spatial = WholeCellSpatialFieldState {
        membrane_adjacency: (0..total_voxels)
            .map(|index| 0.05 + 0.0005 * index as f32)
            .collect(),
        septum_zone: (0..total_voxels)
            .map(|index| 0.15 + 0.0004 * index as f32)
            .collect(),
        nucleoid_occupancy: (0..total_voxels)
            .map(|index| 0.25 + 0.0003 * index as f32)
            .collect(),
        membrane_band_zone: (0..total_voxels)
            .map(|index| 0.35 + 0.0002 * index as f32)
            .collect(),
        pole_zone: (0..total_voxels)
            .map(|index| 0.45 + 0.0001 * index as f32)
            .collect(),
    };
    spec.spatial_fields = Some(spatial.clone());

    let simulator = WholeCellSimulator::from_program_spec(spec);
    let membrane = simulator
        .spatial_fields
        .field_slice(IntracellularSpatialField::MembraneAdjacency);
    let septum = simulator
        .spatial_fields
        .field_slice(IntracellularSpatialField::SeptumZone);
    let nucleoid = simulator
        .spatial_fields
        .field_slice(IntracellularSpatialField::NucleoidOccupancy);
    let membrane_band = simulator
        .spatial_fields
        .field_slice(IntracellularSpatialField::MembraneBandZone);
    let poles = simulator
        .spatial_fields
        .field_slice(IntracellularSpatialField::PoleZone);

    assert_eq!(membrane.len(), total_voxels);
    assert!((membrane[0] - spatial.membrane_adjacency[0]).abs() < 1.0e-6);
    assert!((septum[17] - spatial.septum_zone[17]).abs() < 1.0e-6);
    assert!(
        (nucleoid[total_voxels - 1] - spatial.nucleoid_occupancy[total_voxels - 1]).abs() < 1.0e-6
    );
    assert!((membrane_band[9] - spatial.membrane_band_zone[9]).abs() < 1.0e-6);
    assert!((poles[23] - spatial.pole_zone[23]).abs() < 1.0e-6);
}

#[test]
fn test_from_program_spec_preserves_explicit_named_complexes_with_assets() {
    let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
    let mut simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    for complex in &mut simulator.named_complexes {
        match complex.family {
            WholeCellAssemblyFamily::Ribosome => complex.abundance = 13.0,
            WholeCellAssemblyFamily::RnaPolymerase => complex.abundance = 9.0,
            WholeCellAssemblyFamily::Divisome => complex.abundance = 21.0,
            WholeCellAssemblyFamily::Replisome => complex.abundance = 6.0,
            _ => complex.abundance = 0.0,
        }
    }
    let expected = simulator.snapshot();
    spec.complex_assembly = None;
    spec.named_complexes = simulator.named_complexes;

    let restored = WholeCellSimulator::from_program_spec(spec);
    let snapshot = restored.snapshot();

    assert!(restored.organism_assets.is_some());
    assert!((snapshot.active_rnap - expected.active_rnap).abs() < 1.0e-6);
    assert!((snapshot.active_ribosomes - expected.active_ribosomes).abs() < 1.0e-6);
    assert!((snapshot.dnaa - expected.dnaa).abs() < 1.0e-6);
    assert!((snapshot.ftsz - expected.ftsz).abs() < 1.0e-6);
}

#[test]
fn test_from_legacy_program_spec_json_restores_missing_assets_and_registry() {
    let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
    spec.organism_data_ref = None;
    spec.organism_assets = None;
    spec.organism_process_registry = None;

    let simulator = WholeCellSimulator::from_legacy_program_spec_json(
        &serde_json::to_string(&spec).expect("serialize legacy program spec"),
    )
    .expect("restore legacy program spec");

    assert!(simulator.organism_data.is_some());
    assert!(simulator.organism_assets.is_some());
    assert!(simulator.organism_process_registry.is_some());
    assert!(simulator.organism_process_registry().is_some());
}

#[test]
fn test_from_saved_state_json_preserves_missing_assets_and_registry() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.organism_data_ref = None;
    saved.organism_assets = None;
    saved.organism_process_registry = None;

    let restored = WholeCellSimulator::from_saved_state_json(
        &saved_state_to_json(&saved).expect("serialize explicit saved state"),
    )
    .expect("restore explicit saved state");

    assert!(restored.organism_data.is_some());
    assert!(restored.organism_assets.is_none());
    assert!(restored.organism_process_registry.is_none());
    assert!(restored.organism_process_registry().is_none());
}

#[test]
fn test_from_saved_state_json_without_organism_prefers_explicit_saved_state() {
    let mut simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    simulator.genome_bp = 1000;
    simulator.replicated_bp = 0;
    simulator.chromosome_separation_nm = 0.0;
    simulator.radius_nm = 100.0;
    simulator.surface_area_nm2 = WholeCellSimulator::surface_area_from_radius(simulator.radius_nm);
    simulator.volume_nm3 = WholeCellSimulator::volume_from_radius(simulator.radius_nm);
    simulator.division_progress = 0.0;
    simulator.ftsz = 0.0;
    simulator.dnaa = 0.0;
    simulator.active_ribosomes = 0.0;
    simulator.active_rnap = 0.0;

    simulator.chromosome_state.chromosome_length_bp = 1000;
    simulator.chromosome_state.replicated_bp = 610;
    simulator.chromosome_state.replicated_fraction = 0.61;
    simulator.chromosome_state.segregation_progress = 0.55;

    simulator.membrane_division_state.membrane_area_nm2 = simulator.surface_area_nm2 * 1.35;
    simulator
        .membrane_division_state
        .preferred_membrane_area_nm2 = simulator.surface_area_nm2 * 1.35;
    simulator.membrane_division_state.septum_radius_fraction = 0.22;
    simulator.membrane_division_state.osmotic_balance = 1.0;

    simulator.complex_assembly = WholeCellComplexAssemblyState {
        ribosome_complexes: 17.0,
        rnap_complexes: 10.0,
        dnaa_activity: 8.5,
        ftsz_polymer: 24.0,
        ..WholeCellComplexAssemblyState::default()
    };

    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.organism_data_ref = None;
    saved.organism_data = None;
    saved.organism_assets = None;
    saved.organism_process_registry = None;
    saved.named_complexes.clear();
    saved.core.replicated_bp = 9;
    saved.core.chromosome_separation_nm = 0.0;
    saved.core.radius_nm = 55.0;
    saved.core.surface_area_nm2 = WholeCellSimulator::surface_area_from_radius(55.0);
    saved.core.volume_nm3 = WholeCellSimulator::volume_from_radius(55.0);
    saved.core.division_progress = 0.08;
    saved.core.active_rnap = 2.0;
    saved.core.active_ribosomes = 3.0;
    saved.core.dnaa = 1.0;
    saved.core.ftsz = 4.0;

    let restored = WholeCellSimulator::from_saved_state_json(
        &saved_state_to_json(&saved).expect("serialize stripped saved state"),
    )
    .expect("restore stripped saved state");
    let snapshot = restored.snapshot();

    assert!(restored.organism_data.is_none());
    assert!(restored.organism_assets.is_none());
    assert!(restored.organism_process_registry().is_none());
    assert_eq!(snapshot.replicated_bp, 610);
    assert!((snapshot.division_progress - 0.78).abs() < 1.0e-6);
    assert!((snapshot.radius_nm - (100.0 * 1.35_f32.sqrt())).abs() < 1.0e-4);
    assert!(snapshot.chromosome_separation_nm > 0.0);
    assert!(snapshot.chromosome_separation_nm > saved.core.chromosome_separation_nm);
    assert!((snapshot.active_rnap - 10.0).abs() < 1.0e-6);
    assert!((snapshot.active_ribosomes - 17.0).abs() < 1.0e-6);
    assert!((snapshot.dnaa - 8.5).abs() < 1.0e-6);
    assert!((snapshot.ftsz - 24.0).abs() < 1.0e-6);
}

#[test]
fn test_bundleless_restore_preserves_explicit_expression_runtime_and_named_complex_state() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.organism_data_ref = None;
    saved.organism_data = None;
    saved.organism_assets = None;
    saved.organism_expression.global_activity = 0.83;
    saved.organism_expression.energy_support = 1.17;
    saved.organism_expression.translation_support = 1.09;
    saved.organism_expression.nucleotide_support = 1.11;
    saved.organism_expression.transcription_units[0].promoter_open_fraction = 0.71;
    saved.organism_expression.transcription_units[0].copy_gain = 1.23;
    saved.organism_expression.transcription_units[0].support_level = 1.08;
    saved.organism_species[0].count = 42.0;
    saved.organism_reactions[0].current_flux = 1.25;
    saved.named_complexes = vec![
        WholeCellNamedComplexState {
            id: "ribosome".to_string(),
            operon: "ribosome".to_string(),
            asset_class: WholeCellAssetClass::Translation,
            family: WholeCellAssemblyFamily::Ribosome,
            subsystem_targets: vec![Syn3ASubsystemPreset::RibosomePolysomeCluster],
            subunit_pool: 0.0,
            nucleation_intermediate: 0.0,
            elongation_intermediate: 0.0,
            abundance: 19.0,
            target_abundance: 21.0,
            assembly_rate: 1.2,
            degradation_rate: 0.2,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 1.0,
            structural_support: 1.0,
            assembly_progress: 0.0,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 0.0,
            shared_component_pressure: 0.0,
            insertion_progress: 1.0,
            failure_count: 0.0,
        },
        WholeCellNamedComplexState {
            id: "rnap".to_string(),
            operon: "rnap".to_string(),
            asset_class: WholeCellAssetClass::Homeostasis,
            family: WholeCellAssemblyFamily::RnaPolymerase,
            subsystem_targets: Vec::new(),
            subunit_pool: 0.0,
            nucleation_intermediate: 0.0,
            elongation_intermediate: 0.0,
            abundance: 11.0,
            target_abundance: 12.0,
            assembly_rate: 0.9,
            degradation_rate: 0.15,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 1.0,
            structural_support: 1.0,
            assembly_progress: 0.0,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 0.0,
            shared_component_pressure: 0.0,
            insertion_progress: 1.0,
            failure_count: 0.0,
        },
        WholeCellNamedComplexState {
            id: "divisome".to_string(),
            operon: "divisome".to_string(),
            asset_class: WholeCellAssetClass::Constriction,
            family: WholeCellAssemblyFamily::Divisome,
            subsystem_targets: vec![Syn3ASubsystemPreset::FtsZSeptumRing],
            subunit_pool: 0.0,
            nucleation_intermediate: 0.0,
            elongation_intermediate: 0.0,
            abundance: 26.0,
            target_abundance: 29.0,
            assembly_rate: 1.0,
            degradation_rate: 0.1,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 1.0,
            structural_support: 1.0,
            assembly_progress: 0.0,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 0.0,
            shared_component_pressure: 0.0,
            insertion_progress: 1.0,
            failure_count: 0.0,
        },
    ];
    saved.complex_assembly = WholeCellComplexAssemblyState::default();
    saved.core.active_rnap = 2.0;
    saved.core.active_ribosomes = 3.0;
    saved.core.dnaa = 4.0;
    saved.core.ftsz = 5.0;

    let restored = WholeCellSimulator::from_saved_state_json(
        &saved_state_to_json(&saved).expect("serialize stripped saved state"),
    )
    .expect("restore stripped saved state");
    let expected_inventory = restored.aggregate_named_complex_assembly_state_without_assets();

    assert!(restored.organism_data.is_none());
    assert!(restored.organism_assets.is_none());
    assert!(restored.organism_process_registry().is_some());
    assert_eq!(
        restored.organism_species.len(),
        saved.organism_species.len()
    );
    assert_eq!(
        restored.organism_reactions.len(),
        saved.organism_reactions.len()
    );
    assert_eq!(restored.organism_species[0].count, 42.0);
    assert_eq!(restored.organism_reactions[0].current_flux, 1.25);
    assert_eq!(
        restored.named_complexes_state().len(),
        saved.named_complexes.len()
    );

    let expression = restored
        .organism_expression_state()
        .expect("bundle-less explicit expression state");
    assert_eq!(expression.global_activity, 0.83);
    assert_eq!(expression.energy_support, 1.17);
    assert_eq!(expression.translation_support, 1.09);
    assert_eq!(expression.nucleotide_support, 1.11);
    assert_eq!(
        expression.transcription_units[0].promoter_open_fraction,
        0.71
    );
    assert_eq!(expression.transcription_units[0].copy_gain, 1.23);
    assert_eq!(expression.transcription_units[0].support_level, 1.08);

    let snapshot = restored.snapshot();
    assert!((snapshot.active_rnap - expected_inventory.rnap_complexes).abs() < 1.0e-6);
    assert!((snapshot.active_ribosomes - expected_inventory.ribosome_complexes).abs() < 1.0e-6);
    assert!((snapshot.ftsz - expected_inventory.ftsz_polymer).abs() < 1.0e-6);
    assert!(snapshot.active_rnap > saved.core.active_rnap);
    assert!(snapshot.active_ribosomes > saved.core.active_ribosomes);
    assert!(snapshot.ftsz > saved.core.ftsz);

    let boundary = parse_saved_state_json(
        &restored
            .save_state_json()
            .expect("serialize restored bundle-less state"),
    )
    .expect("parse restored bundle-less state");
    assert_eq!(boundary.organism_species[0].count, 42.0);
    assert_eq!(boundary.organism_reactions[0].current_flux, 1.25);
    assert_eq!(boundary.named_complexes.len(), saved.named_complexes.len());
    assert_eq!(boundary.organism_expression.global_activity, 0.83);
    assert_eq!(
        boundary.organism_expression.transcription_units[0].promoter_open_fraction,
        0.71
    );

    let mut stepped = restored;
    stepped.step();
    assert!(stepped.organism_process_registry().is_some());
    assert_eq!(stepped.organism_species.len(), saved.organism_species.len());
    assert_eq!(
        stepped.organism_reactions.len(),
        saved.organism_reactions.len()
    );
    assert_eq!(
        stepped.named_complexes_state().len(),
        saved.named_complexes.len()
    );
    assert_eq!(
        stepped
            .organism_expression_state()
            .expect("expression state after step")
            .transcription_units
            .len(),
        boundary.organism_expression.transcription_units.len()
    );
    let stepped_boundary = parse_saved_state_json(
        &stepped
            .save_state_json()
            .expect("serialize stepped bundle-less state"),
    )
    .expect("parse stepped bundle-less state");
    assert_eq!(
        stepped_boundary.organism_species.len(),
        saved.organism_species.len()
    );
    assert_eq!(
        stepped_boundary.organism_reactions.len(),
        saved.organism_reactions.len()
    );
}

#[test]
fn test_bundleless_restore_supplements_missing_named_complex_carriers_from_assembly() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.organism_data_ref = None;
    saved.organism_data = None;
    saved.organism_assets = None;
    saved.named_complexes = vec![WholeCellNamedComplexState {
        id: "rnap".to_string(),
        operon: "rnap".to_string(),
        asset_class: WholeCellAssetClass::Homeostasis,
        family: WholeCellAssemblyFamily::RnaPolymerase,
        subsystem_targets: Vec::new(),
        subunit_pool: 0.0,
        nucleation_intermediate: 0.0,
        elongation_intermediate: 0.0,
        abundance: 11.0,
        target_abundance: 12.0,
        assembly_rate: 0.9,
        degradation_rate: 0.15,
        nucleation_rate: 0.0,
        elongation_rate: 0.0,
        maturation_rate: 0.0,
        component_satisfaction: 1.0,
        structural_support: 1.0,
        assembly_progress: 0.0,
        stalled_intermediate: 0.0,
        damaged_abundance: 0.0,
        limiting_component_signal: 0.0,
        shared_component_pressure: 0.0,
        insertion_progress: 1.0,
        failure_count: 0.0,
    }];
    saved.complex_assembly = WholeCellComplexAssemblyState {
        atp_band_complexes: 7.0,
        ribosome_complexes: 18.0,
        rnap_complexes: 11.0,
        replisome_complexes: 6.5,
        membrane_complexes: 9.0,
        ftsz_polymer: 23.0,
        dnaa_activity: 8.0,
        atp_band_target: 8.0,
        ribosome_target: 19.0,
        rnap_target: 12.0,
        replisome_target: 7.0,
        membrane_target: 10.0,
        ftsz_target: 24.0,
        dnaa_target: 8.5,
        atp_band_assembly_rate: 0.7,
        ribosome_assembly_rate: 1.1,
        rnap_assembly_rate: 0.9,
        replisome_assembly_rate: 0.8,
        membrane_assembly_rate: 0.6,
        ftsz_assembly_rate: 1.0,
        dnaa_assembly_rate: 0.5,
        atp_band_degradation_rate: 0.1,
        ribosome_degradation_rate: 0.2,
        rnap_degradation_rate: 0.15,
        replisome_degradation_rate: 0.1,
        membrane_degradation_rate: 0.08,
        ftsz_degradation_rate: 0.12,
        dnaa_degradation_rate: 0.05,
    };

    let restored = WholeCellSimulator::from_saved_state_json(
        &saved_state_to_json(&saved).expect("serialize stripped saved state"),
    )
    .expect("restore stripped saved state");

    assert!(restored.organism_assets.is_none());
    assert!(restored.named_complexes_state().len() > saved.named_complexes.len());
    assert!(restored
        .named_complexes_state()
        .iter()
        .any(|state| state.id == "legacy_ribosome_complex"));
    assert!(restored
        .named_complexes_state()
        .iter()
        .any(|state| state.id == "legacy_replisome_complex"));
    assert!(restored
        .named_complexes_state()
        .iter()
        .any(|state| state.id == "legacy_membrane_complex"));
    assert!(restored
        .named_complexes_state()
        .iter()
        .any(|state| state.id == "legacy_divisome_complex"));
    assert!(restored
        .named_complexes_state()
        .iter()
        .any(|state| state.id == "legacy_dnaa_complex"));

    let aggregate = restored.aggregate_named_complex_assembly_state_without_assets();
    let inventory = restored.assembly_inventory();
    assert!((inventory.ribosome_complexes - aggregate.ribosome_complexes).abs() < 1.0e-6);
    assert!((inventory.replisome_complexes - aggregate.replisome_complexes).abs() < 1.0e-6);
    assert!((inventory.membrane_complexes - aggregate.membrane_complexes).abs() < 1.0e-6);
    assert!((inventory.ftsz_polymer - aggregate.ftsz_polymer).abs() < 1.0e-6);
    assert!((inventory.dnaa_activity - aggregate.dnaa_activity).abs() < 1.0e-6);

    let boundary = parse_saved_state_json(
        &restored
            .save_state_json()
            .expect("serialize restored bundle-less state"),
    )
    .expect("parse restored bundle-less state");
    assert!(boundary.named_complexes.len() > saved.named_complexes.len());
    assert!(boundary
        .named_complexes
        .iter()
        .any(|state| state.id == "legacy_dnaa_complex"));
}

#[test]
fn test_bundleless_restore_synthesizes_expression_from_runtime_process_state() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.organism_data_ref = None;
    saved.organism_data = None;
    saved.organism_assets = None;
    saved.organism_expression = WholeCellOrganismExpressionState::default();

    let mut operon_rna_counts: HashMap<String, usize> = HashMap::new();
    let mut operon_protein_counts: HashMap<String, usize> = HashMap::new();
    for species in &saved.organism_species {
        let Some(operon) = species.operon.as_ref() else {
            continue;
        };
        match species.species_class {
            WholeCellSpeciesClass::Rna => {
                *operon_rna_counts.entry(operon.clone()).or_insert(0) += 1;
            }
            WholeCellSpeciesClass::Protein => {
                *operon_protein_counts.entry(operon.clone()).or_insert(0) += 1;
            }
            _ => {}
        }
    }
    let mut operon_candidates = operon_rna_counts
        .iter()
        .filter_map(|(operon, rna_count)| {
            if *rna_count > 0 && operon_protein_counts.get(operon).copied().unwrap_or(0) > 0 {
                Some(operon.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    operon_candidates.sort();
    let target_operon = operon_candidates
        .into_iter()
        .next()
        .expect("runtime operon with RNA and protein species");

    let mut expected_transcript = 0.0f32;
    let mut expected_protein = 0.0f32;
    for species in &mut saved.organism_species {
        if species.operon.as_deref() != Some(target_operon.as_str()) {
            continue;
        }
        match species.species_class {
            WholeCellSpeciesClass::Rna => {
                species.count = 17.0;
                expected_transcript += species.count;
            }
            WholeCellSpeciesClass::Protein => {
                species.count = 29.0;
                expected_protein += species.count;
            }
            _ => {}
        }
    }
    assert!(expected_transcript > 0.0);
    assert!(expected_protein > 0.0);

    let mut expected_transcription_flux = 0.0f32;
    let mut expected_translation_flux = 0.0f32;
    for reaction in &mut saved.organism_reactions {
        if reaction.operon.as_deref() != Some(target_operon.as_str()) {
            continue;
        }
        match reaction.reaction_class {
            WholeCellReactionClass::Transcription => {
                reaction.current_flux = 1.4;
                expected_transcription_flux += reaction.current_flux;
            }
            WholeCellReactionClass::Translation => {
                reaction.current_flux = 1.1;
                expected_translation_flux += reaction.current_flux;
            }
            _ => {}
        }
    }
    assert!(expected_transcription_flux > 0.0);
    assert!(expected_translation_flux > 0.0);

    let restored = WholeCellSimulator::from_saved_state_json(
        &saved_state_to_json(&saved).expect("serialize stripped saved state"),
    )
    .expect("restore stripped saved state");

    let expression = restored
        .organism_expression_state()
        .expect("bundle-less synthesized expression state");
    let unit = expression
        .transcription_units
        .iter()
        .find(|unit| unit.name == target_operon)
        .expect("synthesized operon unit");

    assert!(!expression.transcription_units.is_empty());
    assert!((unit.transcript_abundance - expected_transcript).abs() < 1.0e-6);
    assert!((unit.protein_abundance - expected_protein).abs() < 1.0e-6);
    assert!((unit.transcript_synthesis_rate - expected_transcription_flux).abs() < 1.0e-6);
    assert!((unit.protein_synthesis_rate - expected_translation_flux).abs() < 1.0e-6);
    assert!(unit.promoter_open_fraction > 0.0);
    assert!(unit.support_level > 0.0);
    assert!(expression.total_transcript_abundance >= expected_transcript);
    assert!(expression.total_protein_abundance >= expected_protein);

    let boundary = parse_saved_state_json(
        &restored
            .save_state_json()
            .expect("serialize synthesized bundle-less state"),
    )
    .expect("parse synthesized bundle-less state");
    assert!(boundary
        .organism_expression
        .transcription_units
        .iter()
        .any(|unit| unit.name == target_operon));

    let mut stepped = restored;
    stepped.step();
    let stepped_expression = stepped
        .organism_expression_state()
        .expect("expression state after step");
    assert!(stepped_expression
        .transcription_units
        .iter()
        .any(|unit| unit.name == target_operon));
}

#[test]
fn test_bundleless_registry_bootstraps_runtime_process_state_without_assets() {
    let mut saved = parse_saved_state_json(
        &WholeCellSimulator::bundled_syn3a_reference()
            .expect("bundled Syn3A")
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.organism_data_ref = None;
    saved.organism_data = None;
    saved.organism_assets = None;
    saved.organism_species.clear();
    saved.organism_reactions.clear();

    let mut restored = WholeCellSimulator::from_saved_state_json(
        &saved_state_to_json(&saved).expect("serialize stripped saved state"),
    )
    .expect("restore stripped saved state");

    assert!(restored.organism_process_registry().is_some());
    assert!(restored.organism_species.is_empty());
    assert!(restored.organism_reactions.is_empty());

    restored.step();

    assert!(!restored.organism_species.is_empty());
    assert!(!restored.organism_reactions.is_empty());
    let stepped_boundary = parse_saved_state_json(
        &restored
            .save_state_json()
            .expect("serialize stepped bundle-less state"),
    )
    .expect("parse stepped bundle-less state");
    assert!(!stepped_boundary.organism_species.is_empty());
    assert!(!stepped_boundary.organism_reactions.is_empty());
}

#[test]
fn test_from_legacy_saved_state_json_restores_missing_assets_and_registry() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.organism_data_ref = None;
    saved.organism_assets = None;
    saved.organism_process_registry = None;

    let restored = WholeCellSimulator::from_legacy_saved_state_json(
        &saved_state_to_json(&saved).expect("serialize legacy saved state"),
    )
    .expect("restore legacy saved state");

    assert!(restored.organism_data.is_some());
    assert!(restored.organism_assets.is_some());
    assert!(restored.organism_process_registry.is_some());
    assert!(restored.organism_process_registry().is_some());
}

#[test]
fn test_from_legacy_saved_state_json_promotes_core_summary_to_explicit_state() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.organism_data_ref = None;
    saved.organism_data = None;
    saved.organism_assets = None;
    saved.organism_process_registry = None;
    saved.chromosome_state = WholeCellChromosomeState::default();
    saved.membrane_division_state = WholeCellMembraneDivisionState::default();
    saved.complex_assembly = WholeCellComplexAssemblyState::default();
    saved.scheduler_state = WholeCellSchedulerState::default();
    saved.named_complexes.clear();
    saved.core.genome_bp = 1000;
    saved.core.replicated_bp = 610;
    saved.core.chromosome_separation_nm = 75.0;
    saved.core.radius_nm = 115.0;
    saved.core.surface_area_nm2 = 170_000.0;
    saved.core.volume_nm3 = 800_000.0;
    saved.core.division_progress = 0.28;
    saved.core.step_count = 23;
    saved.core.time_ms = 11.5;
    saved.core.active_rnap = 12.0;
    saved.core.active_ribosomes = 19.0;
    saved.core.dnaa = 9.5;
    saved.core.ftsz = 24.0;
    saved.core.glucose_mm = 2.2;
    saved.core.oxygen_mm = 1.6;
    saved.core.adp_mm = 0.55;
    saved.core.metabolic_load = 1.32;
    saved.chemistry_report = LocalChemistryReport::default();
    saved.chemistry_site_reports.clear();
    saved.organism_expression = WholeCellOrganismExpressionState::default();
    saved.last_md_probe = None;
    saved.scheduled_subsystem_probes.clear();
    saved.subsystem_states.clear();
    saved.md_translation_scale = 1.0;
    saved.md_membrane_scale = 1.0;
    saved.lattice.atp.fill(3.0);
    saved.lattice.amino_acids.fill(2.4);
    saved.lattice.nucleotides.fill(2.0);
    saved.lattice.membrane_precursors.fill(1.6);

    let saved_json = saved_state_to_json(&saved).expect("serialize legacy saved state");
    let expected_saved =
        parse_legacy_saved_state_json(&saved_json).expect("parse promoted legacy saved state");
    let expected_chemistry = expected_saved.chemistry_report;
    let expected_sites = expected_saved.chemistry_site_reports.clone();
    let expected_expression = expected_saved.organism_expression.clone();
    let expected_md_probe = expected_saved.last_md_probe;
    let expected_scheduled_probes = expected_saved.scheduled_subsystem_probes.clone();
    let restored = WholeCellSimulator::from_legacy_saved_state_json(&saved_json)
        .expect("restore legacy saved state");
    let snapshot = restored.snapshot();
    let chromosome = restored.chromosome_state();
    let membrane = restored.membrane_division_state();
    let complex = restored.complex_assembly_state();

    assert!(restored.organism_assets.is_none());
    assert_eq!(chromosome.chromosome_length_bp, 1000);
    assert_eq!(chromosome.replicated_bp, 610);
    assert!(chromosome.segregation_progress > 0.0);
    assert!(!chromosome.forks.is_empty());
    assert!(membrane.preferred_membrane_area_nm2 >= 170_000.0);
    assert!((membrane.septum_radius_fraction - 0.72).abs() < 1.0e-6);
    assert!((complex.rnap_complexes - 12.0).abs() < 1.0e-6);
    assert!((complex.ribosome_complexes - 19.0).abs() < 1.0e-6);
    assert!((complex.dnaa_activity - 9.5).abs() < 1.0e-6);
    assert!((complex.ftsz_polymer - 24.0).abs() < 1.0e-6);
    assert!(!restored.named_complexes_state().is_empty());
    assert!(restored
        .named_complexes_state()
        .iter()
        .any(|complex| complex.family == WholeCellAssemblyFamily::Ribosome));
    assert!(restored
        .named_complexes_state()
        .iter()
        .any(|complex| complex.family == WholeCellAssemblyFamily::RnaPolymerase));
    assert!(restored
        .named_complexes_state()
        .iter()
        .any(|complex| complex.family == WholeCellAssemblyFamily::ReplicationInitiator));
    let chemistry = restored
        .local_chemistry_report()
        .expect("promoted legacy local chemistry report");
    assert_eq!(chemistry, expected_chemistry);
    let sites = restored.local_chemistry_sites();
    assert_eq!(sites, expected_sites);
    assert_eq!(
        restored.scheduled_syn3a_subsystem_probes(),
        expected_scheduled_probes
    );
    assert_eq!(restored.last_md_probe(), expected_md_probe);
    assert_eq!(
        restored
            .organism_expression_state()
            .expect("promoted legacy expression state"),
        expected_expression
    );
    let replisome_state = restored
        .subsystem_states()
        .into_iter()
        .find(|state| state.preset == Syn3ASubsystemPreset::ReplisomeTrack)
        .expect("replisome subsystem state");
    let replisome_site = sites
        .iter()
        .find(|site| site.preset == Syn3ASubsystemPreset::ReplisomeTrack)
        .expect("replisome chemistry site");
    assert_eq!(replisome_state.site_x, replisome_site.site_x);
    assert_eq!(replisome_state.site_y, replisome_site.site_y);
    assert_eq!(replisome_state.site_z, replisome_site.site_z);
    assert!(replisome_state.demand_satisfaction > 0.7);
    assert!((snapshot.active_rnap - 12.0).abs() < 1.0e-6);
    assert!((snapshot.active_ribosomes - 19.0).abs() < 1.0e-6);
    assert!((snapshot.ftsz - 24.0).abs() < 1.0e-6);
    assert_eq!(snapshot.replicated_bp, 610);
    assert!((snapshot.division_progress - 0.28).abs() < 1.0e-6);
    assert_eq!(
        snapshot.scheduler_state.stage_clocks.len(),
        WholeCellSimulator::SOLVER_STAGE_ORDER.len()
    );
    assert!(snapshot
        .scheduler_state
        .stage_clocks
        .iter()
        .any(|clock| clock.run_count > 0));
}

#[test]
fn test_from_legacy_saved_state_json_promotes_expression_from_assets_without_runtime_or_registry() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.organism_data_ref = None;
    saved.organism_data = None;
    saved.organism_process_registry = None;
    saved.organism_species.clear();
    saved.organism_reactions.clear();
    saved.organism_expression = WholeCellOrganismExpressionState::default();
    saved.chemistry_report = LocalChemistryReport::default();
    saved.chemistry_site_reports.clear();
    saved.core.metabolic_load = 1.22;
    saved.lattice.atp.fill(2.9);
    saved.lattice.amino_acids.fill(2.3);
    saved.lattice.nucleotides.fill(2.0);
    saved.lattice.membrane_precursors.fill(1.5);

    let saved_json = saved_state_to_json(&saved).expect("serialize legacy saved state");
    let expected_saved =
        parse_legacy_saved_state_json(&saved_json).expect("parse promoted legacy saved state");
    let restored = WholeCellSimulator::from_legacy_saved_state_json(&saved_json)
        .expect("restore legacy saved state");

    assert!(restored.organism_assets.is_some());
    assert_eq!(
        restored
            .organism_expression_state()
            .expect("asset-promoted legacy expression state"),
        expected_saved.organism_expression
    );
}

#[test]
fn test_from_legacy_saved_state_json_promotes_expression_from_assembly_without_assets() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.organism_data_ref = None;
    saved.organism_data = None;
    saved.organism_assets = None;
    saved.organism_process_registry = None;
    saved.organism_species.clear();
    saved.organism_reactions.clear();
    saved.organism_expression = WholeCellOrganismExpressionState::default();
    saved.named_complexes.clear();
    saved.complex_assembly = WholeCellComplexAssemblyState {
        rnap_complexes: 12.0,
        ribosome_complexes: 18.0,
        dnaa_activity: 7.0,
        ftsz_polymer: 9.0,
        rnap_target: 14.0,
        ribosome_target: 20.0,
        dnaa_target: 8.0,
        ftsz_target: 11.0,
        rnap_assembly_rate: 1.2,
        ribosome_assembly_rate: 1.5,
        dnaa_assembly_rate: 0.8,
        ftsz_assembly_rate: 0.7,
        ..WholeCellComplexAssemblyState::default()
    };

    let saved_json = saved_state_to_json(&saved).expect("serialize legacy saved state");
    let expected_saved =
        parse_legacy_saved_state_json(&saved_json).expect("parse promoted legacy saved state");
    let restored = WholeCellSimulator::from_legacy_saved_state_json(&saved_json)
        .expect("restore legacy saved state");
    let expression = restored
        .organism_expression_state()
        .expect("assembly-promoted legacy expression state");

    assert!(restored.organism_assets.is_none());
    assert_eq!(expression, expected_saved.organism_expression);
    assert!(expression
        .transcription_units
        .iter()
        .any(|unit| unit.name == "legacy_rnap_complex" && unit.process_drive.transcription > 0.0));
    assert!(
        expression
            .transcription_units
            .iter()
            .any(|unit| unit.name == "legacy_ribosome_complex"
                && unit.process_drive.translation > 0.0)
    );
    assert!(expression
        .transcription_units
        .iter()
        .any(|unit| unit.name == "legacy_dnaa_complex" && unit.process_drive.replication > 0.0));
}

#[test]
fn test_from_legacy_saved_state_json_promotes_expression_from_core_without_assets() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.organism_data_ref = None;
    saved.organism_data = None;
    saved.organism_assets = None;
    saved.organism_process_registry = None;
    saved.organism_species.clear();
    saved.organism_reactions.clear();
    saved.organism_expression = WholeCellOrganismExpressionState::default();
    saved.named_complexes.clear();
    saved.complex_assembly = WholeCellComplexAssemblyState::default();
    saved.core.active_rnap = 11.0;
    saved.core.active_ribosomes = 17.0;
    saved.core.dnaa = 6.0;
    saved.core.ftsz = 8.0;

    let saved_json = saved_state_to_json(&saved).expect("serialize legacy saved state");
    let expected_saved =
        parse_legacy_saved_state_json(&saved_json).expect("parse promoted legacy saved state");
    let restored = WholeCellSimulator::from_legacy_saved_state_json(&saved_json)
        .expect("restore legacy saved state");
    let expression = restored
        .organism_expression_state()
        .expect("core-promoted legacy expression state");

    assert!(restored.organism_assets.is_none());
    assert_eq!(expression, expected_saved.organism_expression);
    assert!(expression
        .transcription_units
        .iter()
        .any(|unit| unit.name == "legacy_rnap_complex" && unit.process_drive.transcription > 0.0));
    assert!(
        expression
            .transcription_units
            .iter()
            .any(|unit| unit.name == "legacy_ribosome_complex"
                && unit.process_drive.translation > 0.0)
    );
    assert!(expression
        .transcription_units
        .iter()
        .any(|unit| unit.name == "legacy_dnaa_complex" && unit.process_drive.replication > 0.0));
    assert!(expression.transcription_units.iter().any(|unit| unit.name
        == "legacy_divisome_complex"
        && unit.process_drive.constriction > 0.0));
}

#[test]
fn test_from_legacy_saved_state_json_prefers_site_reports_for_missing_chemistry_report() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.chemistry_report = LocalChemistryReport::default();
    saved.chemistry_site_reports = vec![
        LocalChemistrySiteReport {
            preset: Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
            site: WholeCellChemistrySite::AtpSynthaseBand,
            patch_radius: 2,
            site_x: 3,
            site_y: 2,
            site_z: 1,
            localization_score: 0.82,
            atp_support: 1.34,
            translation_support: 0.88,
            nucleotide_support: 0.91,
            membrane_support: 1.26,
            crowding_penalty: 0.74,
            mean_glucose: 1.9,
            mean_oxygen: 1.4,
            mean_atp_flux: 1.3,
            mean_carbon_dioxide: 0.7,
            mean_nitrate: 0.2,
            mean_ammonium: 0.1,
            mean_proton: 0.3,
            mean_phosphorus: 0.15,
            assembly_component_availability: 0.9,
            assembly_occupancy: 0.7,
            assembly_stability: 0.85,
            assembly_turnover: 0.12,
            substrate_draw: 0.4,
            energy_draw: 0.5,
            biosynthetic_draw: 0.3,
            byproduct_load: 0.2,
            demand_satisfaction: 0.92,
        },
        LocalChemistrySiteReport {
            preset: Syn3ASubsystemPreset::ReplisomeTrack,
            site: WholeCellChemistrySite::ChromosomeTrack,
            patch_radius: 2,
            site_x: 4,
            site_y: 2,
            site_z: 1,
            localization_score: 0.68,
            atp_support: 0.96,
            translation_support: 0.79,
            nucleotide_support: 1.28,
            membrane_support: 0.82,
            crowding_penalty: 0.81,
            mean_glucose: 1.1,
            mean_oxygen: 0.9,
            mean_atp_flux: 1.5,
            mean_carbon_dioxide: 0.5,
            mean_nitrate: 0.3,
            mean_ammonium: 0.2,
            mean_proton: 0.2,
            mean_phosphorus: 0.18,
            assembly_component_availability: 0.86,
            assembly_occupancy: 0.64,
            assembly_stability: 0.78,
            assembly_turnover: 0.16,
            substrate_draw: 0.45,
            energy_draw: 0.42,
            biosynthetic_draw: 0.48,
            byproduct_load: 0.24,
            demand_satisfaction: 0.88,
        },
    ];
    saved.core.glucose_mm = 9.0;
    saved.core.oxygen_mm = 8.0;
    saved.lattice.atp.fill(0.2);

    let saved_json = saved_state_to_json(&saved).expect("serialize legacy saved state");
    let expected_saved =
        parse_legacy_saved_state_json(&saved_json).expect("parse promoted legacy saved state");
    let restored = WholeCellSimulator::from_legacy_saved_state_json(&saved_json)
        .expect("restore legacy saved state");

    assert_eq!(
        restored
            .local_chemistry_report()
            .expect("site-derived chemistry report"),
        expected_saved.chemistry_report
    );
    assert_eq!(
        restored.local_chemistry_sites(),
        saved.chemistry_site_reports
    );
}

#[test]
fn test_from_legacy_saved_state_json_promotes_last_md_probe_into_subsystem_state() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.subsystem_states.clear();
    saved.scheduled_subsystem_probes.clear();
    saved.last_md_probe = Some(LocalMDProbeReport {
        site: WholeCellChemistrySite::ChromosomeTrack,
        mean_temperature: 304.0,
        mean_total_energy: -18.0,
        mean_vdw_energy: -7.0,
        mean_electrostatic_energy: -4.5,
        structural_order: 0.91,
        crowding_penalty: 0.79,
        compactness: 0.73,
        shell_order: 0.69,
        axis_anisotropy: 0.41,
        thermal_stability: 0.84,
        electrostatic_order: 0.66,
        vdw_cohesion: 0.71,
        polar_fraction: 0.24,
        phosphate_fraction: 0.31,
        hydrogen_fraction: 0.27,
        bond_density: 0.46,
        angle_density: 0.39,
        dihedral_density: 0.28,
        charge_density: 0.21,
        recommended_atp_scale: 1.08,
        recommended_translation_scale: 0.94,
        recommended_replication_scale: 1.29,
        recommended_segregation_scale: 1.18,
        recommended_membrane_scale: 0.91,
        recommended_constriction_scale: 0.88,
    });

    let saved_json = saved_state_to_json(&saved).expect("serialize legacy saved state");
    let expected_saved =
        parse_legacy_saved_state_json(&saved_json).expect("parse promoted legacy saved state");
    let restored = WholeCellSimulator::from_legacy_saved_state_json(&saved_json)
        .expect("restore legacy saved state");

    let replisome = restored
        .subsystem_states()
        .into_iter()
        .find(|state| state.preset == Syn3ASubsystemPreset::ReplisomeTrack)
        .expect("promoted replisome subsystem");
    let expected_replisome = expected_saved
        .subsystem_states
        .into_iter()
        .find(|state| state.preset == Syn3ASubsystemPreset::ReplisomeTrack)
        .expect("expected replisome subsystem");

    assert_eq!(replisome, expected_replisome);
    assert_eq!(replisome.last_probe_step, Some(saved.core.step_count));
}

#[test]
fn test_from_legacy_saved_state_json_prefers_explicit_local_probe_schedule() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.chemistry_site_reports.clear();
    saved.subsystem_states.clear();
    saved.last_md_probe = None;
    saved.scheduled_subsystem_probes.clear();
    saved.local_chemistry = Some(WholeCellLocalChemistrySpec {
        x_dim: 10,
        y_dim: 8,
        z_dim: 4,
        voxel_size_au: 0.5,
        use_gpu: true,
        enable_default_syn3a_subsystems: false,
        scheduled_subsystem_probes: vec![
            ScheduledSubsystemProbe {
                preset: Syn3ASubsystemPreset::ReplisomeTrack,
                interval_steps: 17,
            },
            ScheduledSubsystemProbe {
                preset: Syn3ASubsystemPreset::FtsZSeptumRing,
                interval_steps: 11,
            },
        ],
    });

    let saved_json = saved_state_to_json(&saved).expect("serialize legacy saved state");
    let expected_saved =
        parse_legacy_saved_state_json(&saved_json).expect("parse promoted legacy saved state");
    let restored = WholeCellSimulator::from_legacy_saved_state_json(&saved_json)
        .expect("restore legacy saved state");

    assert_eq!(
        restored.scheduled_syn3a_subsystem_probes(),
        expected_saved.scheduled_subsystem_probes
    );
    assert_eq!(
        restored.scheduled_syn3a_subsystem_probes(),
        saved
            .local_chemistry
            .as_ref()
            .expect("local chemistry")
            .scheduled_subsystem_probes
    );
}

#[test]
fn test_restore_saved_state_refreshes_subsystem_state_from_explicit_site_reports() {
    let simulator = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A");
    let mut saved = parse_saved_state_json(
        &simulator
            .save_state_json()
            .expect("serialize explicit saved state"),
    )
    .expect("parse explicit saved state");
    saved.subsystem_states.clear();
    saved.chemistry_report = LocalChemistryReport {
        atp_support: 1.08,
        translation_support: 1.04,
        nucleotide_support: 1.22,
        membrane_support: 0.94,
        crowding_penalty: 0.81,
        mean_glucose: 2.1,
        mean_oxygen: 1.7,
        mean_atp_flux: 1.5,
        mean_carbon_dioxide: 0.42,
    };
    saved.chemistry_site_reports = vec![LocalChemistrySiteReport {
        preset: Syn3ASubsystemPreset::ReplisomeTrack,
        site: WholeCellChemistrySite::ChromosomeTrack,
        patch_radius: 2,
        site_x: 7,
        site_y: 5,
        site_z: 3,
        localization_score: 0.84,
        atp_support: 1.12,
        translation_support: 0.92,
        nucleotide_support: 1.34,
        membrane_support: 0.88,
        crowding_penalty: 0.79,
        mean_glucose: 1.9,
        mean_oxygen: 1.4,
        mean_atp_flux: 1.7,
        mean_carbon_dioxide: 0.46,
        mean_nitrate: 0.31,
        mean_ammonium: 0.22,
        mean_proton: 0.19,
        mean_phosphorus: 0.41,
        assembly_component_availability: 0.83,
        assembly_occupancy: 0.78,
        assembly_stability: 0.91,
        assembly_turnover: 0.18,
        substrate_draw: 0.36,
        energy_draw: 0.28,
        biosynthetic_draw: 0.44,
        byproduct_load: 0.17,
        demand_satisfaction: 0.88,
    }];

    let restored = WholeCellSimulator::from_saved_state_json(
        &saved_state_to_json(&saved).expect("serialize refreshed saved state"),
    )
    .expect("restore explicit saved state");

    let replisome_state = restored
        .subsystem_states()
        .into_iter()
        .find(|state| state.preset == Syn3ASubsystemPreset::ReplisomeTrack)
        .expect("replisome subsystem state");
    assert_eq!(replisome_state.site_x, 7);
    assert_eq!(replisome_state.site_y, 5);
    assert_eq!(replisome_state.site_z, 3);
    assert!(replisome_state.replication_scale > 1.0);
    assert!(replisome_state.demand_satisfaction > 0.8);
}

// ---- Phase 8: Quantum Auto-Discovery ----

#[test]
fn test_quantum_auto_discovery_empty() {
    let sim = WholeCellSimulator::new(WholeCellConfig::default());
    assert!(sim.discovered_quantum_reactions().is_empty());
}

#[test]
fn test_quantum_auto_discovery_finds_translation() {
    use crate::whole_cell_data::WholeCellReactionRuntimeState;
    let mut sim = WholeCellSimulator::new(WholeCellConfig::default());
    sim.organism_reactions = vec![WholeCellReactionRuntimeState {
        id: "tr_ftsZ".into(),
        name: "FtsZ translation".into(),
        reaction_class: WholeCellReactionClass::Translation,
        asset_class: WholeCellAssetClass::Constriction,
        nominal_rate: 1.2,
        catalyst: None,
        operon: None,
        reactants: vec![],
        products: vec![],
        subsystem_targets: vec![Syn3ASubsystemPreset::RibosomePolysomeCluster],
        spatial_scope: Default::default(),
        patch_domain: Default::default(),
        chromosome_domain: None,
        current_flux: 0.0,
        cumulative_extent: 0.0,
        reactant_satisfaction: 1.0,
        catalyst_support: 1.0,
        process_weights: WholeCellProcessWeights::default(),
    }];
    sim.run_quantum_auto_discovery();
    assert_eq!(sim.discovered_quantum_reactions().len(), 1);
    assert_eq!(
        sim.discovered_quantum_reactions()[0].profile_channel,
        QuantumProfileChannel::Translation
    );
}

#[test]
fn test_quantum_auto_discovery_energy_maps_oxphos() {
    use crate::whole_cell_data::WholeCellReactionRuntimeState;
    let mut sim = WholeCellSimulator::new(WholeCellConfig::default());
    sim.organism_reactions = vec![WholeCellReactionRuntimeState {
        id: "atp_syn".into(),
        name: "ATP synthase transport".into(),
        reaction_class: WholeCellReactionClass::PoolTransport,
        asset_class: WholeCellAssetClass::Energy,
        nominal_rate: 80.0,
        catalyst: None,
        operon: None,
        reactants: vec![],
        products: vec![],
        subsystem_targets: vec![Syn3ASubsystemPreset::AtpSynthaseMembraneBand],
        spatial_scope: Default::default(),
        patch_domain: Default::default(),
        chromosome_domain: None,
        current_flux: 0.0,
        cumulative_extent: 0.0,
        reactant_satisfaction: 1.0,
        catalyst_support: 1.0,
        process_weights: WholeCellProcessWeights::default(),
    }];
    sim.run_quantum_auto_discovery();
    assert_eq!(sim.discovered_quantum_reactions().len(), 1);
    assert_eq!(
        sim.discovered_quantum_reactions()[0].profile_channel,
        QuantumProfileChannel::Oxphos
    );
}

#[test]
fn test_quantum_auto_discovery_ignores_ineligible() {
    use crate::whole_cell_data::WholeCellReactionRuntimeState;
    let mut sim = WholeCellSimulator::new(WholeCellConfig::default());
    sim.organism_reactions = vec![WholeCellReactionRuntimeState {
        id: "rna_deg".into(),
        name: "RNA degradation".into(),
        reaction_class: WholeCellReactionClass::RnaDegradation,
        asset_class: WholeCellAssetClass::QualityControl,
        nominal_rate: 0.5,
        catalyst: None,
        operon: None,
        reactants: vec![],
        products: vec![],
        subsystem_targets: vec![Syn3ASubsystemPreset::RibosomePolysomeCluster],
        spatial_scope: Default::default(),
        patch_domain: Default::default(),
        chromosome_domain: None,
        current_flux: 0.0,
        cumulative_extent: 0.0,
        reactant_satisfaction: 1.0,
        catalyst_support: 1.0,
        process_weights: WholeCellProcessWeights::default(),
    }];
    sim.run_quantum_auto_discovery();
    assert!(sim.discovered_quantum_reactions().is_empty());
}

#[test]
fn test_quantum_probe_refinement() {
    use crate::whole_cell_data::WholeCellReactionRuntimeState;
    let mut sim = WholeCellSimulator::new(WholeCellConfig::default());
    sim.organism_reactions = vec![WholeCellReactionRuntimeState {
        id: "atp_t".into(),
        name: "ATP transport".into(),
        reaction_class: WholeCellReactionClass::PoolTransport,
        asset_class: WholeCellAssetClass::Energy,
        nominal_rate: 80.0,
        catalyst: None,
        operon: None,
        reactants: vec![],
        products: vec![],
        subsystem_targets: vec![Syn3ASubsystemPreset::AtpSynthaseMembraneBand],
        spatial_scope: Default::default(),
        patch_domain: Default::default(),
        chromosome_domain: None,
        current_flux: 0.0,
        cumulative_extent: 0.0,
        reactant_satisfaction: 1.0,
        catalyst_support: 1.0,
        process_weights: WholeCellProcessWeights::default(),
    }];
    sim.run_quantum_auto_discovery();
    let probe = LocalMDProbeReport {
        site: WholeCellChemistrySite::AtpSynthaseBand,
        mean_temperature: 310.0,
        mean_total_energy: -50.0,
        mean_vdw_energy: -30.0,
        mean_electrostatic_energy: -20.0,
        structural_order: 1.2,
        crowding_penalty: 0.1,
        compactness: 0.85,
        shell_order: 0.9,
        axis_anisotropy: 0.1,
        thermal_stability: 1.1,
        electrostatic_order: 1.0,
        vdw_cohesion: 0.8,
        polar_fraction: 0.4,
        phosphate_fraction: 0.1,
        hydrogen_fraction: 0.3,
        bond_density: 0.5,
        angle_density: 0.3,
        dihedral_density: 0.2,
        charge_density: 0.05,
        recommended_atp_scale: 1.15,
        recommended_translation_scale: 1.10,
        recommended_replication_scale: 1.0,
        recommended_segregation_scale: 1.0,
        recommended_membrane_scale: 1.05,
        recommended_constriction_scale: 1.0,
    };
    sim.refine_quantum_corrections_from_probe(&probe);
    assert!(sim
        .discovered_quantum_reactions()
        .iter()
        .all(|r| r.probe_refined));
    assert!((sim.quantum_profile().oxphos_efficiency - 1.0).abs() > 1e-4);
}

// ---- Observable Auto-Calibration Engine ----

#[test]
fn test_observable_loss_perfect_match() {
    let sim = WholeCellSimulator::new(WholeCellConfig::default());
    let targets = vec![ObservableTarget {
        kind: ObservableKind::AtpMm,
        target_value: sim.atp_mm,
        tolerance: 0.1,
        weight: 1.0,
    }];
    let (loss, per) = sim.observable_loss(&targets);
    assert!(loss < 1e-6, "loss should be ~0 for perfect match: {}", loss);
    assert!(per[0] < 1e-6);
}

#[test]
fn test_observable_loss_off_target() {
    let sim = WholeCellSimulator::new(WholeCellConfig::default());
    let targets = vec![ObservableTarget {
        kind: ObservableKind::AtpMm,
        target_value: sim.atp_mm + 10.0,
        tolerance: 1.0,
        weight: 1.0,
    }];
    let (loss, _) = sim.observable_loss(&targets);
    assert!(loss > 1.0, "loss should be large for off-target: {}", loss);
}

#[test]
fn test_observable_observe_all_kinds() {
    let sim = WholeCellSimulator::new(WholeCellConfig::default());
    let kinds = [
        ObservableKind::AtpMm,
        ObservableKind::AminoAcidsMm,
        ObservableKind::NucleotidesMm,
        ObservableKind::MembranePrecursorsMm,
        ObservableKind::GlucoseMm,
        ObservableKind::OxygenMm,
        ObservableKind::ActiveRibosomes,
        ObservableKind::ActiveRnap,
        ObservableKind::RadiusNm,
        ObservableKind::SurfaceAreaNm2,
        ObservableKind::VolumeNm3,
        ObservableKind::DivisionProgress,
        ObservableKind::GrowthRatePerMs,
    ];
    for kind in &kinds {
        let val = sim.observe(*kind);
        assert!(val.is_finite(), "{:?} returned non-finite: {}", kind, val);
    }
}
