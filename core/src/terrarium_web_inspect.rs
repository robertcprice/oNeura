//! Authoritative browser inspection data built from live terrarium state.

use crate::atomistic_chemistry::PeriodicElement;
use crate::constants::clamp;
use crate::drosophila::{self, genome_stats as fly_genome_stats};
use crate::drosophila_population::{EggCluster, Fly, FlyEmbryoState, FlyLifeStage, FLY_ENERGY_MAX};
use crate::organism_metabolism::OrganismMetabolism;
use crate::seed_cellular::SeedTissue;
use crate::soil_broad::{shoreline_water_signal, soil_texture_absorbency, soil_texture_retention};
use crate::soil_fauna::{earthworm_genome_stats, nematode_genome_stats, NematodeKind};
use crate::terrarium::geochemistry::{
    normalize_terrarium_geochemistry_for_inspect, SUBSTRATE_CHEMISTRY_GRID_SPECIES,
};
use crate::terrarium::material_exchange::inventory_component_amount;
use crate::terrarium::scale_level::{
    AtomVisual, BondVisual, CellularDetail, DerivationChain, GeneInfo, MolecularDetail,
    MolecularRef, OpticalDerivation, OrganelleInfo, OrganismComponentDetail, QuantumAtomSummary,
    QuantumMolecularDescriptorSummary, RateDerivation, ScaleInspectResponse, ScaleLevel,
};
use crate::terrarium::visual_projection::sample_visual_air;
use crate::terrarium::{
    soil_aluminum_toxicity, soil_base_saturation, soil_weathering_support,
    terrarium_molecular_asset_by_name, PlantTissue, SoilOwnershipClass, TerrariumDemoPreset,
    TerrariumSpecies, TerrariumWorld, EXPLICIT_MICROBE_MAX_REPRESENTED_CELLS,
};
use crate::terrarium_web_protocol::{
    InspectComposition, InspectData, InspectGrid, InspectMetric, InspectPosition,
};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct InspectQuery {
    pub kind: String,
    pub index: Option<usize>,
    pub x: Option<f32>,
    pub y: Option<f32>,
    /// Target scale level: "ecosystem"|"organism"|"cellular"|"molecular"|"atomic"
    #[serde(default)]
    pub scale: Option<String>,
    /// Which tissue/component to drill into at cellular scale.
    #[serde(default)]
    pub tissue: Option<String>,
    /// Which molecule to show at molecular scale.
    #[serde(default)]
    pub molecule: Option<String>,
    /// Which atom index to focus at atomic scale.
    #[serde(default)]
    pub atom_index: Option<usize>,
}

pub fn build_inspect_data(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    query: &InspectQuery,
) -> Result<InspectData, String> {
    match query.kind.trim().to_ascii_lowercase().as_str() {
        "plant" => inspect_plant(world, preset, required_index(query)?),
        "fly" => inspect_fly(world, preset, required_index(query)?),
        "fly_egg" => inspect_fly_egg(world, preset, required_index(query)?),
        "fly_embryo" => inspect_fly_embryo(world, preset, required_index(query)?),
        "fly_larva" => inspect_fly_larva(world, preset, required_index(query)?),
        "fly_pupa" => inspect_fly_pupa(world, preset, required_index(query)?),
        "fruit" => inspect_fruit(world, preset, required_index(query)?),
        "seed" => inspect_seed(world, preset, required_index(query)?),
        "water" => inspect_water(world, preset, query.index, query.x, query.y),
        "earthworm" => inspect_earthworm(world, preset, required_index(query)?),
        "nematode" => inspect_nematode(world, preset, required_index(query)?),
        "soil" | "terrain" => {
            let (x, y) = required_xy(world, query.x, query.y)?;
            inspect_soil(world, preset, x, y)
        }
        other => Err(format!("Unsupported inspect kind: {other}")),
    }
}

/// Build a scale-aware inspection response. If no scale is specified in the
/// query, returns a `ScaleInspectResponse` with the base `InspectData` and
/// `ScaleLevel::Ecosystem` (backward compatible).
pub fn build_scale_inspect_data(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    query: &InspectQuery,
) -> Result<ScaleInspectResponse, String> {
    let base = build_inspect_data(world, preset, query)?;
    let scale = query
        .scale
        .as_deref()
        .and_then(ScaleLevel::from_str_opt)
        .unwrap_or(ScaleLevel::Ecosystem);

    let mut response = ScaleInspectResponse {
        base,
        scale,
        organism_components: None,
        cellular_detail: None,
        molecular_detail: None,
        atomic_detail: None,
    };

    match scale {
        ScaleLevel::Ecosystem => {} // base data is sufficient
        ScaleLevel::Organism => {
            response.organism_components =
                Some(build_organism_components(world, query)?);
        }
        ScaleLevel::Cellular => {
            response.cellular_detail =
                Some(build_cellular_detail(world, query)?);
        }
        ScaleLevel::Molecular => {
            let plane = world.config.width * world.config.height;
            let mean_temp_c = if plane > 0 {
                Some(world.temperature.iter().take(plane).sum::<f32>() / plane as f32)
            } else {
                None
            };
            response.molecular_detail =
                Some(build_molecular_detail(query, mean_temp_c)?);
        }
        ScaleLevel::Atomic => {
            response.atomic_detail =
                Some(build_atomic_detail(query)?);
        }
    }

    Ok(response)
}

// ---------------------------------------------------------------------------
// Organism-level builders
// ---------------------------------------------------------------------------

fn build_organism_components(
    world: &TerrariumWorld,
    query: &InspectQuery,
) -> Result<Vec<OrganismComponentDetail>, String> {
    match query.kind.trim().to_ascii_lowercase().as_str() {
        "plant" => build_plant_organism_components(world, required_index(query)?),
        "fly" => build_fly_organism_components(world, required_index(query)?),
        "earthworm" => Ok(build_earthworm_organism_components()),
        "nematode" => Ok(build_nematode_organism_components()),
        _ => Err(format!(
            "Organism-level inspection not supported for kind '{}'",
            query.kind
        )),
    }
}

fn build_plant_organism_components(
    world: &TerrariumWorld,
    index: usize,
) -> Result<Vec<OrganismComponentDetail>, String> {
    let plant = world
        .plants
        .get(index)
        .ok_or_else(|| format!("Plant index {index} out of range"))?;
    let leaf = plant.cellular.cluster_snapshot(PlantTissue::Leaf);
    let stem = plant.cellular.cluster_snapshot(PlantTissue::Stem);
    let root = plant.cellular.cluster_snapshot(PlantTissue::Root);
    let meristem = plant.cellular.cluster_snapshot(PlantTissue::Meristem);

    Ok(vec![
        OrganismComponentDetail {
            component_name: "leaf".into(),
            component_type: "tissue".into(),
            cell_count: leaf.cell_count as u32,
            cell_types: vec!["mesophyll".into(), "epidermal".into(), "guard".into()],
            molecular_inventory: vec![
                MolecularRef { name: "glucose".into(), role: "photosynthetic product".into() },
                MolecularRef { name: "water".into(), role: "photolysis substrate".into() },
                MolecularRef { name: "oxygen_gas".into(), role: "photosynthetic byproduct".into() },
            ],
            metrics: vec![
                metric("Cell Count", format!("{:.0}", leaf.cell_count), Some((leaf.cell_count / 200.0).clamp(0.0, 1.0))),
                metric("Energy Charge", format!("{:.3}", cluster_energy_charge(&leaf)), Some(cluster_energy_charge(&leaf))),
            ],
        },
        OrganismComponentDetail {
            component_name: "stem".into(),
            component_type: "tissue".into(),
            cell_count: stem.cell_count as u32,
            cell_types: vec!["xylem".into(), "phloem".into(), "cortex".into()],
            molecular_inventory: vec![
                MolecularRef { name: "glucose".into(), role: "transported sugar".into() },
                MolecularRef { name: "water".into(), role: "xylem fluid".into() },
            ],
            metrics: vec![
                metric("Cell Count", format!("{:.0}", stem.cell_count), Some((stem.cell_count / 200.0).clamp(0.0, 1.0))),
                metric("Energy Charge", format!("{:.3}", cluster_energy_charge(&stem)), Some(cluster_energy_charge(&stem))),
            ],
        },
        OrganismComponentDetail {
            component_name: "root".into(),
            component_type: "tissue".into(),
            cell_count: root.cell_count as u32,
            cell_types: vec!["root_hair".into(), "cortex".into(), "endodermis".into()],
            molecular_inventory: vec![
                MolecularRef { name: "water".into(), role: "absorbed from soil".into() },
                MolecularRef { name: "nitrate".into(), role: "nitrogen uptake".into() },
                MolecularRef { name: "ammonium".into(), role: "nitrogen uptake".into() },
            ],
            metrics: vec![
                metric("Cell Count", format!("{:.0}", root.cell_count), Some((root.cell_count / 200.0).clamp(0.0, 1.0))),
                metric("Energy Charge", format!("{:.3}", cluster_energy_charge(&root)), Some(cluster_energy_charge(&root))),
            ],
        },
        OrganismComponentDetail {
            component_name: "meristem".into(),
            component_type: "tissue".into(),
            cell_count: meristem.cell_count as u32,
            cell_types: vec!["stem_cell".into(), "dividing".into()],
            molecular_inventory: vec![
                MolecularRef { name: "atp_flux".into(), role: "energy for division".into() },
                MolecularRef { name: "nucleotide_pool".into(), role: "DNA replication".into() },
            ],
            metrics: vec![
                metric("Cell Count", format!("{:.0}", meristem.cell_count), Some((meristem.cell_count / 50.0).clamp(0.0, 1.0))),
                metric("Energy Charge", format!("{:.3}", cluster_energy_charge(&meristem)), Some(cluster_energy_charge(&meristem))),
            ],
        },
    ])
}

fn build_fly_organism_components(
    world: &TerrariumWorld,
    index: usize,
) -> Result<Vec<OrganismComponentDetail>, String> {
    let fly = world
        .flies
        .get(index)
        .ok_or_else(|| format!("Fly index {index} out of range"))?;
    let body = fly.body_state();

    Ok(vec![
        OrganismComponentDetail {
            component_name: "head".into(),
            component_type: "organ".into(),
            cell_count: 0,
            cell_types: vec!["neuron".into(), "photoreceptor".into(), "antennal_sensory".into()],
            molecular_inventory: vec![
                MolecularRef { name: "atp_flux".into(), role: "neural energy".into() },
            ],
            metrics: vec![
                metric("Orientation", format!("{:.1} deg", body.heading.to_degrees()), None),
            ],
        },
        OrganismComponentDetail {
            component_name: "thorax".into(),
            component_type: "organ".into(),
            cell_count: 0,
            cell_types: vec!["flight_muscle".into(), "motor_neuron".into()],
            molecular_inventory: vec![
                MolecularRef { name: "atp_flux".into(), role: "flight energy".into() },
                MolecularRef { name: "glucose".into(), role: "metabolic fuel".into() },
            ],
            metrics: vec![
                metric("Wing Beat", format!("{:.1} Hz", body.wing_beat_freq), None),
            ],
        },
        OrganismComponentDetail {
            component_name: "abdomen".into(),
            component_type: "organ".into(),
            cell_count: 0,
            cell_types: vec!["fat_body".into(), "gut_epithelial".into(), "malpighian_tubule".into()],
            molecular_inventory: vec![
                MolecularRef { name: "glucose".into(), role: "energy reserve".into() },
                MolecularRef { name: "amino_acid_pool".into(), role: "protein synthesis".into() },
            ],
            metrics: vec![
                metric("Energy", format!("{:.3} uJ", body.energy), Some((body.energy / 5000.0).clamp(0.0, 1.0))),
            ],
        },
        OrganismComponentDetail {
            component_name: "wings".into(),
            component_type: "organ".into(),
            cell_count: 0,
            cell_types: vec!["cuticle".into(), "vein".into()],
            molecular_inventory: vec![],
            metrics: vec![
                metric("Wing Stroke", format!("{:.3}", body.wing_stroke), None),
            ],
        },
        OrganismComponentDetail {
            component_name: "neural_circuit".into(),
            component_type: "system".into(),
            cell_count: 198,
            cell_types: vec!["sensory_neuron".into(), "interneuron".into(), "motor_neuron".into()],
            molecular_inventory: vec![
                MolecularRef { name: "atp_flux".into(), role: "synaptic transmission".into() },
            ],
            metrics: vec![
                metric("Neurons", "198".into(), None),
                metric("Connections", "302".into(), None),
            ],
        },
    ])
}

fn build_earthworm_organism_components() -> Vec<OrganismComponentDetail> {
    vec![
        OrganismComponentDetail {
            component_name: "segments".into(),
            component_type: "tissue".into(),
            cell_count: 0,
            cell_types: vec!["muscle".into(), "epithelial".into(), "nerve_cord".into()],
            molecular_inventory: vec![
                MolecularRef { name: "glucose".into(), role: "metabolic fuel".into() },
            ],
            metrics: vec![],
        },
        OrganismComponentDetail {
            component_name: "clitellum".into(),
            component_type: "organ".into(),
            cell_count: 0,
            cell_types: vec!["glandular".into()],
            molecular_inventory: vec![],
            metrics: vec![],
        },
    ]
}

fn build_nematode_organism_components() -> Vec<OrganismComponentDetail> {
    vec![
        OrganismComponentDetail {
            component_name: "pharynx".into(),
            component_type: "organ".into(),
            cell_count: 0,
            cell_types: vec!["muscle".into(), "epithelial".into()],
            molecular_inventory: vec![],
            metrics: vec![],
        },
        OrganismComponentDetail {
            component_name: "neural_circuit".into(),
            component_type: "system".into(),
            cell_count: 302,
            cell_types: vec!["sensory_neuron".into(), "interneuron".into(), "motor_neuron".into()],
            molecular_inventory: vec![
                MolecularRef { name: "atp_flux".into(), role: "synaptic energy".into() },
            ],
            metrics: vec![
                metric("Neurons", "302".into(), None),
            ],
        },
        OrganismComponentDetail {
            component_name: "body_wall".into(),
            component_type: "tissue".into(),
            cell_count: 0,
            cell_types: vec!["muscle".into(), "hypodermis".into()],
            molecular_inventory: vec![],
            metrics: vec![],
        },
    ]
}

// ---------------------------------------------------------------------------
// Cellular-level builders
// ---------------------------------------------------------------------------

fn build_cellular_detail(
    world: &TerrariumWorld,
    query: &InspectQuery,
) -> Result<CellularDetail, String> {
    match query.kind.trim().to_ascii_lowercase().as_str() {
        "plant" => build_plant_cellular_detail(world, required_index(query)?, query.tissue.as_deref()),
        "fly" => Ok(build_fly_cellular_detail(query.tissue.as_deref())),
        "earthworm" => Ok(build_earthworm_cellular_detail(query.tissue.as_deref())),
        "nematode" => Ok(build_nematode_cellular_detail(query.tissue.as_deref())),
        _ => Err(format!(
            "Cellular-level inspection not supported for kind '{}'",
            query.kind
        )),
    }
}

fn build_plant_cellular_detail(
    world: &TerrariumWorld,
    index: usize,
    tissue: Option<&str>,
) -> Result<CellularDetail, String> {
    let plant = world
        .plants
        .get(index)
        .ok_or_else(|| format!("Plant index {index} out of range"))?;

    // Guard cell is a specialized leaf cell type — use leaf cluster data
    let is_guard = tissue.unwrap_or("leaf") == "guard";
    let tissue_type = match tissue.unwrap_or("leaf") {
        "stem" => PlantTissue::Stem,
        "root" => PlantTissue::Root,
        "meristem" => PlantTissue::Meristem,
        "guard" => PlantTissue::Leaf, // guard cells live in the leaf epidermis
        _ => PlantTissue::Leaf,
    };
    let cluster = plant.cellular.cluster_snapshot(tissue_type);
    let energy_charge = cluster_energy_charge(&cluster);
    let cell_type_name = if is_guard {
        "guard cell"
    } else {
        match tissue_type {
            PlantTissue::Leaf => "mesophyll",
            PlantTissue::Stem => "xylem/phloem",
            PlantTissue::Root => "root cortex",
            PlantTissue::Meristem => "stem cell",
        }
    };

    let organelles = if is_guard {
        vec![
            OrganelleInfo { name: "chloroplast".into(), count: 12, function: "photosynthesis + blue-light sensing".into() },
            OrganelleInfo { name: "mitochondrion".into(), count: 60, function: "ATP for ion pumping".into() },
            OrganelleInfo { name: "central_vacuole".into(), count: 1, function: "osmotic turgor regulation".into() },
            OrganelleInfo { name: "cell_wall".into(), count: 1, function: "asymmetric thickening for stomatal opening".into() },
            OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
        ]
    } else {
        match tissue_type {
            PlantTissue::Leaf => vec![
                OrganelleInfo { name: "chloroplast".into(), count: 40, function: "photosynthesis".into() },
                OrganelleInfo { name: "mitochondrion".into(), count: 100, function: "respiration".into() },
                OrganelleInfo { name: "central_vacuole".into(), count: 1, function: "turgor pressure".into() },
                OrganelleInfo { name: "cell_wall".into(), count: 1, function: "structural support".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
            ],
            PlantTissue::Stem => vec![
                OrganelleInfo { name: "mitochondrion".into(), count: 50, function: "respiration".into() },
                OrganelleInfo { name: "cell_wall".into(), count: 1, function: "structural support (lignified)".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
            ],
            PlantTissue::Root => vec![
                OrganelleInfo { name: "mitochondrion".into(), count: 80, function: "respiration for active transport".into() },
                OrganelleInfo { name: "central_vacuole".into(), count: 1, function: "ion storage".into() },
                OrganelleInfo { name: "cell_wall".into(), count: 1, function: "structural support".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
            ],
            PlantTissue::Meristem => vec![
                OrganelleInfo { name: "mitochondrion".into(), count: 200, function: "energy for division".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "active replication".into() },
                OrganelleInfo { name: "ribosome_cluster".into(), count: 500, function: "protein synthesis".into() },
            ],
        }
    };

    let (genes, membrane) = if is_guard {
        (
            vec![
                GeneInfo { name: "SLAC1".into(), expression_level: energy_charge * 0.85, function: "anion channel (stomatal closure)".into() },
                GeneInfo { name: "KAT1".into(), expression_level: energy_charge * 0.78, function: "K+ inward rectifier (stomatal opening)".into() },
            ],
            vec![
                MolecularRef { name: "membrane_precursor_pool".into(), role: "phospholipid bilayer".into() },
            ],
        )
    } else {
        (
            vec![
                GeneInfo { name: "RbcL".into(), expression_level: energy_charge, function: "RuBisCO large subunit".into() },
                GeneInfo { name: "PsbA".into(), expression_level: energy_charge * 0.8, function: "photosystem II D1 protein".into() },
            ],
            vec![
                MolecularRef { name: "membrane_precursor_pool".into(), role: "phospholipid bilayer".into() },
                MolecularRef { name: "glucose".into(), role: "glycolipid component".into() },
            ],
        )
    };

    Ok(CellularDetail {
        cell_type: cell_type_name.into(),
        organelles,
        membrane_composition: membrane,
        active_genes: genes,
        metabolic_state: vec![
            metric("ATP Level", format!("{:.3}", energy_charge), Some(energy_charge)),
            metric("Cell Count", format!("{:.0}", cluster.cell_count), None),
            metric("Glucose", format!("{:.3}", cluster.state_glucose), Some((cluster.state_glucose / 80.0).clamp(0.0, 1.0))),
        ],
    })
}

fn build_fly_cellular_detail(tissue: Option<&str>) -> CellularDetail {
    match tissue.unwrap_or("flight_muscle") {
        "compound_eye" => CellularDetail {
            cell_type: "photoreceptor (ommatidium)".into(),
            organelles: vec![
                OrganelleInfo { name: "rhabdomere".into(), count: 1, function: "microvillar photoreceptive membrane".into() },
                OrganelleInfo { name: "pigment_granule".into(), count: 8, function: "optical isolation between ommatidia".into() },
                OrganelleInfo { name: "mitochondrion".into(), count: 400, function: "ATP for phototransduction".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
            ],
            membrane_composition: vec![
                MolecularRef { name: "membrane_precursor_pool".into(), role: "phospholipid bilayer".into() },
            ],
            active_genes: vec![
                GeneInfo { name: "Rh1".into(), expression_level: 0.92, function: "rhodopsin 1 — primary visual pigment".into() },
                GeneInfo { name: "ninaE".into(), expression_level: 0.88, function: "opsin (neither inactivation nor afterpotential E)".into() },
            ],
            metabolic_state: vec![
                metric("ATP Demand", "High (phototransduction)".into(), Some(0.85)),
            ],
        },
        "neural_circuit" => CellularDetail {
            cell_type: "neuron (Kenyon cell)".into(),
            organelles: vec![
                OrganelleInfo { name: "synaptic_vesicle".into(), count: 200, function: "neurotransmitter storage and release".into() },
                OrganelleInfo { name: "endoplasmic_reticulum".into(), count: 1, function: "protein folding and Ca²⁺ store".into() },
                OrganelleInfo { name: "mitochondrion".into(), count: 150, function: "ATP for synaptic transmission".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
            ],
            membrane_composition: vec![
                MolecularRef { name: "membrane_precursor_pool".into(), role: "phospholipid bilayer".into() },
            ],
            active_genes: vec![
                GeneInfo { name: "elav".into(), expression_level: 0.91, function: "pan-neuronal RNA-binding protein".into() },
                GeneInfo { name: "Syt1".into(), expression_level: 0.87, function: "synaptotagmin 1 — Ca²⁺ sensor for vesicle fusion".into() },
            ],
            metabolic_state: vec![
                metric("ATP Demand", "High (synaptic)".into(), Some(0.82)),
            ],
        },
        _ => CellularDetail {
            cell_type: "flight_muscle".into(),
            organelles: vec![
                OrganelleInfo { name: "mitochondrion".into(), count: 1000, function: "ATP for flight".into() },
                OrganelleInfo { name: "sarcomere".into(), count: 200, function: "contractile unit".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
            ],
            membrane_composition: vec![
                MolecularRef { name: "membrane_precursor_pool".into(), role: "phospholipid bilayer".into() },
            ],
            active_genes: vec![
                GeneInfo { name: "Actin88F".into(), expression_level: 0.95, function: "indirect flight muscle actin".into() },
                GeneInfo { name: "Myosin".into(), expression_level: 0.90, function: "motor protein".into() },
            ],
            metabolic_state: vec![
                metric("ATP Demand", "High (flight)".into(), Some(0.9)),
            ],
        },
    }
}

fn build_earthworm_cellular_detail(tissue: Option<&str>) -> CellularDetail {
    match tissue.unwrap_or("segment_muscle") {
        "chloragogenous" => CellularDetail {
            cell_type: "chloragocyte (fat body)".into(),
            organelles: vec![
                OrganelleInfo { name: "granular_inclusion".into(), count: 200, function: "glycogen and lipid storage".into() },
                OrganelleInfo { name: "mitochondrion".into(), count: 80, function: "respiration".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
            ],
            membrane_composition: vec![
                MolecularRef { name: "membrane_precursor_pool".into(), role: "phospholipid bilayer".into() },
            ],
            active_genes: vec![
                GeneInfo { name: "vitellogenin".into(), expression_level: 0.72, function: "yolk protein precursor / lipid transport".into() },
                GeneInfo { name: "metallothionein".into(), expression_level: 0.65, function: "heavy metal detoxification".into() },
            ],
            metabolic_state: vec![
                metric("ATP Demand", "Moderate (storage)".into(), Some(0.55)),
            ],
        },
        "nephridium" => CellularDetail {
            cell_type: "excretory cell (nephridium)".into(),
            organelles: vec![
                OrganelleInfo { name: "flame_cell".into(), count: 1, function: "ciliary filtration".into() },
                OrganelleInfo { name: "tubule".into(), count: 1, function: "reabsorption / secretion".into() },
                OrganelleInfo { name: "mitochondrion".into(), count: 120, function: "active transport for excretion".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
            ],
            membrane_composition: vec![
                MolecularRef { name: "membrane_precursor_pool".into(), role: "phospholipid bilayer".into() },
            ],
            active_genes: vec![
                GeneInfo { name: "aquaporin".into(), expression_level: 0.80, function: "water channel for osmoregulation".into() },
            ],
            metabolic_state: vec![
                metric("ATP Demand", "Moderate (excretion)".into(), Some(0.60)),
            ],
        },
        _ => CellularDetail {
            cell_type: "circular/longitudinal muscle".into(),
            organelles: vec![
                OrganelleInfo { name: "mitochondrion".into(), count: 300, function: "ATP for peristaltic contraction".into() },
                OrganelleInfo { name: "myofilament".into(), count: 50, function: "actin-myosin contractile assembly".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
            ],
            membrane_composition: vec![
                MolecularRef { name: "membrane_precursor_pool".into(), role: "phospholipid bilayer".into() },
            ],
            active_genes: vec![
                GeneInfo { name: "mhc-1".into(), expression_level: 0.88, function: "myosin heavy chain — muscle contraction".into() },
            ],
            metabolic_state: vec![
                metric("ATP Demand", "Moderate (peristalsis)".into(), Some(0.65)),
            ],
        },
    }
}

fn build_nematode_cellular_detail(tissue: Option<&str>) -> CellularDetail {
    match tissue.unwrap_or("body_wall_muscle") {
        "pharyngeal_muscle" => CellularDetail {
            cell_type: "pharynx pump cell".into(),
            organelles: vec![
                OrganelleInfo { name: "sarcomere".into(), count: 80, function: "rhythmic pumping contraction".into() },
                OrganelleInfo { name: "mitochondrion".into(), count: 200, function: "ATP for continuous pharyngeal pumping".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
            ],
            membrane_composition: vec![
                MolecularRef { name: "membrane_precursor_pool".into(), role: "phospholipid bilayer".into() },
            ],
            active_genes: vec![
                GeneInfo { name: "myo-2".into(), expression_level: 0.93, function: "pharyngeal myosin — feeding pump motor".into() },
            ],
            metabolic_state: vec![
                metric("ATP Demand", "High (pharyngeal pump)".into(), Some(0.85)),
            ],
        },
        "sensory_neuron" => CellularDetail {
            cell_type: "amphid sensory neuron".into(),
            organelles: vec![
                OrganelleInfo { name: "cilium".into(), count: 1, function: "chemosensory signal transduction".into() },
                OrganelleInfo { name: "synaptic_vesicle".into(), count: 50, function: "neurotransmitter release".into() },
                OrganelleInfo { name: "mitochondrion".into(), count: 30, function: "ATP for signal transduction".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
            ],
            membrane_composition: vec![
                MolecularRef { name: "membrane_precursor_pool".into(), role: "phospholipid bilayer".into() },
            ],
            active_genes: vec![
                GeneInfo { name: "odr-10".into(), expression_level: 0.86, function: "odorant receptor (diacetyl)".into() },
                GeneInfo { name: "tax-4".into(), expression_level: 0.81, function: "cyclic nucleotide-gated channel".into() },
            ],
            metabolic_state: vec![
                metric("ATP Demand", "Moderate (sensory)".into(), Some(0.55)),
            ],
        },
        _ => CellularDetail {
            cell_type: "obliquely striated muscle".into(),
            organelles: vec![
                OrganelleInfo { name: "sarcomere".into(), count: 40, function: "oblique striation for sinusoidal locomotion".into() },
                OrganelleInfo { name: "mitochondrion".into(), count: 60, function: "ATP for body wall contraction".into() },
                OrganelleInfo { name: "nucleus".into(), count: 1, function: "gene regulation".into() },
            ],
            membrane_composition: vec![
                MolecularRef { name: "membrane_precursor_pool".into(), role: "phospholipid bilayer".into() },
            ],
            active_genes: vec![
                GeneInfo { name: "unc-54".into(), expression_level: 0.90, function: "body wall myosin heavy chain B".into() },
                GeneInfo { name: "unc-15".into(), expression_level: 0.84, function: "paramyosin — thick filament core".into() },
            ],
            metabolic_state: vec![
                metric("ATP Demand", "Moderate (locomotion)".into(), Some(0.60)),
            ],
        },
    }
}

// ---------------------------------------------------------------------------
// Molecular-level builders
// ---------------------------------------------------------------------------

fn build_molecular_detail(query: &InspectQuery, mean_temp_c: Option<f32>) -> Result<MolecularDetail, String> {
    let molecule_name = query
        .molecule
        .as_deref()
        .unwrap_or("water");

    let embedded = terrarium_molecular_asset_by_name(molecule_name)
        .ok_or_else(|| format!("No molecular asset found for '{molecule_name}'"))?;

    let graph = &embedded.graph;
    let positions = &embedded.positions_angstrom;

    // Build atom visuals
    let atoms: Vec<AtomVisual> = graph
        .atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| {
            let elem = atom.element;
            AtomVisual {
                element: format!("{elem}"),
                symbol: elem.symbol().to_string(),
                atomic_number: elem.atomic_number(),
                position: positions[i],
                vdw_radius: elem.van_der_waals_radius_angstrom(),
                cpk_color: elem.cpk_color_rgb(),
                formal_charge: atom.formal_charge,
                electron_config: elem.electron_configuration_short().to_string(),
                quantum_state: build_quantum_atom_summary(elem, atom.formal_charge),
            }
        })
        .collect();

    // Build bond visuals
    let bonds: Vec<BondVisual> = graph
        .bonds
        .iter()
        .map(|bond| {
            let pi = positions[bond.i];
            let pj = positions[bond.j];
            let dx = pj[0] - pi[0];
            let dy = pj[1] - pi[1];
            let dz = pj[2] - pi[2];
            let length = (dx * dx + dy * dy + dz * dz).sqrt();
            BondVisual {
                atom_i: bond.i,
                atom_j: bond.j,
                order: match bond.order {
                    crate::atomistic_chemistry::BondOrder::Single => "single",
                    crate::atomistic_chemistry::BondOrder::Double => "double",
                    crate::atomistic_chemistry::BondOrder::Triple => "triple",
                    crate::atomistic_chemistry::BondOrder::Aromatic => "aromatic",
                }
                .into(),
                length_angstrom: length,
                midpoint: [
                    (pi[0] + pj[0]) / 2.0,
                    (pi[1] + pj[1]) / 2.0,
                    (pi[2] + pj[2]) / 2.0,
                ],
            }
        })
        .collect();

    // Build formula string from element composition
    let comp = graph.element_composition();
    let mut formula_parts: Vec<(String, usize)> = comp
        .iter()
        .map(|(elem, count)| (elem.symbol().to_string(), *count))
        .collect();
    // Hill system order: C first, H second, then alphabetical
    formula_parts.sort_by(|a, b| {
        let order = |s: &str| match s {
            "C" => 0,
            "H" => 1,
            _ => 2,
        };
        order(&a.0).cmp(&order(&b.0)).then(a.0.cmp(&b.0))
    });
    let formula: String = formula_parts
        .iter()
        .map(|(sym, count)| {
            if *count == 1 {
                sym.clone()
            } else {
                format!("{sym}{count}")
            }
        })
        .collect();

    // Attach quantum descriptor if available
    let species = TerrariumSpecies::from_name(molecule_name);
    let quantum_descriptor = species.and_then(|sp| {
        crate::terrarium::inventory_species_registry::terrarium_inventory_quantum_descriptor(sp)
            .map(|d| QuantumMolecularDescriptorSummary {
                ground_state_energy_ev: d.ground_state_energy_ev,
                ground_state_energy_per_atom_ev: d.ground_state_energy_per_atom_ev,
                dipole_magnitude_e_angstrom: d.dipole_magnitude_e_angstrom,
                mean_abs_effective_charge: d.mean_abs_effective_charge,
                charge_span: d.charge_span,
                mean_lda_exchange_potential_ev: d.mean_lda_exchange_potential_ev,
                frontier_occupancy_fraction: d.frontier_occupancy_fraction,
            })
    });

    // Build derivation chain: optical properties + rate derivations
    let derivation_chain = build_derivation_chain(species, molecule_name, mean_temp_c);

    // Build backbone ribbon mesh from atom positions for molecular zoom visualization.
    let backbone_mesh = if atoms.len() >= 3 {
        Some(crate::terrarium::ribbon::build_protein_backbone_ribbon(
            &atoms,
            &crate::terrarium::ribbon::RibbonConfig {
                subdivisions: 4,
                radial_segments: 6,
                min_taper_ratio: 0.3,
                caps: true,
            },
        ))
    } else {
        None
    };

    Ok(MolecularDetail {
        name: graph.name.clone(),
        formula,
        molecular_weight: graph.molecular_mass_daltons(),
        atoms,
        bonds,
        quantum_descriptor,
        derivation_chain,
        backbone_mesh,
    })
}

/// Build the atoms → optics → color + bond → Eyring → rate derivation chain.
fn build_derivation_chain(
    species: Option<TerrariumSpecies>,
    molecule_name: &str,
    mean_temp_c: Option<f32>,
) -> Option<DerivationChain> {
    use crate::terrarium::emergent_color::molecular_optical_properties;
    use crate::terrarium::emergent_rates::{metabolome_rate, metabolome_rate_derivation};

    let optical = species.and_then(|sp| {
        molecular_optical_properties(sp).map(|props| OpticalDerivation {
            cpk_rgb: props.inherent_rgb,
            molar_extinction: props.molar_extinction,
            scattering_cross_section: props.scattering_cross_section,
        })
    });

    // Map species to their associated metabolic pathways
    let pathways: &[&str] = match molecule_name {
        "glucose" => &["photosynthesis", "respiration", "fly_glycolysis"],
        "trehalose" => &["trehalase", "fly_trehalase"],
        "atp" | "atp_flux" => &["respiration", "fly_glycolysis"],
        "ethylene" => &["ethylene"],
        "jasmonate" | "jasmonic_acid" => &["jasmonate"],
        "salicylate" | "salicylic_acid" => &["salicylate"],
        "sucrose" => &["sucrose"],
        "starch" => &["starch"],
        "fructose" => &["fructose"],
        "malate" | "malic_acid" => &["malate"],
        "citrate" | "citric_acid" => &["citrate"],
        "carbon_dioxide" | "co2" => &["photosynthesis", "respiration"],
        "oxygen" | "oxygen_gas" | "o2" => &["respiration", "photosynthesis"],
        _ => &[],
    };

    let rates: Vec<RateDerivation> = pathways
        .iter()
        .filter_map(|pathway| {
            let (bond_type, bond_energy_ev, enzyme_efficiency, vmax_25, citation) =
                metabolome_rate_derivation(pathway)?;
            // Compute live rate at current world temperature via Eyring TST
            let rate_at_current_temp = mean_temp_c
                .map(|t| metabolome_rate(pathway, t) as f32);
            Some(RateDerivation {
                pathway: pathway.to_string(),
                bond_type: bond_type.to_string(),
                bond_energy_ev,
                enzyme_efficiency,
                vmax_25,
                rate_at_current_temp,
                citation: citation.to_string(),
            })
        })
        .collect();

    if optical.is_none() && rates.is_empty() {
        return None;
    }

    Some(DerivationChain { optical, rates })
}

fn build_quantum_atom_summary(
    elem: PeriodicElement,
    formal_charge: i8,
) -> Option<QuantumAtomSummary> {
    use crate::subatomic_quantum::quantum_element_descriptor;
    let desc = quantum_element_descriptor(elem, formal_charge).ok()?;
    Some(QuantumAtomSummary {
        valence_electrons: desc.valence_electrons as u8,
        ionization_proxy_ev: desc.ionization_proxy_ev as f32,
        mean_valence_orbital_energy_ev: desc.mean_valence_orbital_energy_ev as f32,
        mean_effective_nuclear_charge: desc.mean_effective_nuclear_charge as f32,
        mean_orbital_radius_angstrom: desc.mean_orbital_radius_angstrom as f32,
    })
}

// ---------------------------------------------------------------------------
// Atomic-level builders
// ---------------------------------------------------------------------------

fn build_atomic_detail(query: &InspectQuery) -> Result<Vec<AtomVisual>, String> {
    // If a molecule is specified, return all atoms of that molecule
    if let Some(molecule_name) = query.molecule.as_deref() {
        let detail = build_molecular_detail(query, None)?;
        // If a specific atom index is requested, return just that atom
        if let Some(idx) = query.atom_index {
            return detail
                .atoms
                .get(idx)
                .cloned()
                .map(|a| vec![a])
                .ok_or_else(|| format!("Atom index {idx} out of range for '{molecule_name}'"));
        }
        return Ok(detail.atoms);
    }
    // Without a molecule, return a single element's atomic data
    let elem = PeriodicElement::from_symbol_or_name(
        query.kind.trim(),
    )
    .unwrap_or(PeriodicElement::C);

    Ok(vec![AtomVisual {
        element: format!("{elem}"),
        symbol: elem.symbol().to_string(),
        atomic_number: elem.atomic_number(),
        position: [0.0, 0.0, 0.0],
        vdw_radius: elem.van_der_waals_radius_angstrom(),
        cpk_color: elem.cpk_color_rgb(),
        formal_charge: 0,
        electron_config: elem.electron_configuration_short().to_string(),
        quantum_state: build_quantum_atom_summary(elem, 0),
    }])
}

// ---------------------------------------------------------------------------
// Original inspect entry point
// ---------------------------------------------------------------------------

fn inspect_plant(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    index: usize,
) -> Result<InspectData, String> {
    let plant = world
        .plants
        .get(index)
        .ok_or_else(|| format!("Plant index {index} out of range"))?;
    let x = plant.x;
    let y = plant.y;
    let local_air = local_air(world, x as f32 + 0.5, y as f32 + 0.5);
    let root_z = root_zone_depth(world, plant.genome.root_depth_bias);
    let root_zone = substrate_patch_metrics(world, x, y, root_z);
    let root_inventory = inventory_patch_metrics(&plant.material_inventory);
    let root_zone_proton =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Proton, x, y, root_z, 1);
    let root_zone_base_saturation = soil_base_saturation(
        root_zone.exchangeable_calcium,
        root_zone.exchangeable_magnesium,
        root_zone.exchangeable_potassium,
        root_zone.exchangeable_sodium,
        root_zone.exchangeable_aluminum,
        root_zone_proton,
        root_zone.surface_proton_load,
    );
    let root_zone_al_toxicity = soil_aluminum_toxicity(
        root_zone.exchangeable_aluminum,
        root_zone_proton,
        root_zone.surface_proton_load,
        root_zone_base_saturation,
    );
    let root_zone_weathering_support = soil_weathering_support(
        root_zone.dissolved_silicate,
        root_zone.bicarbonate,
        root_zone.calcium_bicarbonate_complex,
        root_zone.exchangeable_calcium,
        root_zone.exchangeable_magnesium,
        root_zone.exchangeable_potassium,
        root_zone.aqueous_iron,
        root_zone.carbonate_mineral,
        root_zone.clay_mineral,
    );
    let leaf = plant.cellular.cluster_snapshot(PlantTissue::Leaf);
    let stem = plant.cellular.cluster_snapshot(PlantTissue::Stem);
    let root = plant.cellular.cluster_snapshot(PlantTissue::Root);
    let meristem = plant.cellular.cluster_snapshot(PlantTissue::Meristem);
    let leaf_energy = cluster_energy_charge(&leaf);
    let approx_photosynthetic_capacity = clamp(
        0.66 + leaf_energy * 0.36 + plant.cellular.vitality() * 0.16,
        0.45,
        1.4,
    );
    let fruit_load = plant.physiology.fruit_count();
    let title = format!(
        "{} ({})",
        crate::terrarium::plant_species::plant_common_name(plant.genome.taxonomy_id),
        crate::terrarium::plant_species::plant_scientific_name(plant.genome.taxonomy_id)
    );

    Ok(InspectData {
        kind: "plant".into(),
        title,
        subtitle: format!(
            "{} in {}",
            format!("{:?}", plant.morphology.growth_form).replace('_', " "),
            preset.label()
        ),
        preset: preset.cli_name().into(),
        position: Some(InspectPosition {
            x: x as f32 + 0.5,
            y: y as f32 + 0.5,
            z: Some(root_z as f32),
        }),
        summary: vec![
            metric("Height", format!("{:.2} mm", plant.physiology.height_mm()), Some(plant.physiology.height_mm() / plant.genome.max_height_mm.max(1.0))),
            metric("Vitality", format!("{:.3}", plant.cellular.vitality()), Some(plant.cellular.vitality())),
            metric("Health", format!("{:.3}", plant.physiology.health()), Some(plant.physiology.health())),
            metric("Fruit Load", fruit_load.to_string(), Some((fruit_load as f32 / 12.0).clamp(0.0, 1.0))),
            metric("Storage Carbon", format!("{:.3}", plant.physiology.storage_carbon()), Some((plant.physiology.storage_carbon() / 0.5).clamp(0.0, 1.0))),
        ],
        scene: vec![
            metric("Canopy Radius", format!("{} cells", plant.canopy_radius_cells()), Some((plant.canopy_radius_cells() as f32 / world.config.width.max(1) as f32).clamp(0.0, 1.0))),
            metric("Root Radius", format!("{} cells", plant.root_radius_cells()), Some((plant.root_radius_cells() as f32 / world.config.width.max(1) as f32).clamp(0.0, 1.0))),
            metric("Leaf Area Index", format!("{:.3}", plant.physiology.lai()), Some((plant.physiology.lai() / 4.5).clamp(0.0, 1.0))),
            metric("Daylight", format!("{:.3}", world.daylight()), Some(world.daylight())),
            metric("Temperature", format!("{:.1} C", local_air.temperature_c), Some(((local_air.temperature_c - 5.0) / 30.0).clamp(0.0, 1.0))),
            metric("Humidity", format!("{:.2}", local_air.humidity), Some(local_air.humidity)),
            metric("Air Pressure", format!("{:.2} kPa", local_air.pressure_kpa), Some(((local_air.pressure_kpa - 95.0) / 12.0).clamp(0.0, 1.0))),
            metric("Wind Speed", format!("{:.2}", local_air.wind_speed), Some((local_air.wind_speed / 2.0).clamp(0.0, 1.0))),
        ],
        cellular: vec![
            metric("Total Cells", format!("{:.0}", plant.cellular.total_cells()), Some((plant.cellular.total_cells() / 600.0).clamp(0.0, 1.0))),
            metric("Cell Energy", format!("{:.3}", plant.cellular.energy_charge()), Some(plant.cellular.energy_charge())),
            metric("Photosynth Capacity", format!("{:.3}", approx_photosynthetic_capacity), Some((approx_photosynthetic_capacity / 1.4).clamp(0.0, 1.0))),
            metric("Sugar Pool", format!("{:.3}", plant.cellular.sugar_pool()), Some((plant.cellular.sugar_pool() / 80.0).clamp(0.0, 1.0))),
            metric("Water Pool", format!("{:.3}", plant.cellular.water_pool()), Some((plant.cellular.water_pool() / 120.0).clamp(0.0, 1.0))),
            metric("Nitrogen Pool", format!("{:.3}", plant.cellular.nitrogen_pool()), Some((plant.cellular.nitrogen_pool() / 30.0).clamp(0.0, 1.0))),
            metric("Division Signal", format!("{:.3}", plant.cellular.division_signal()), Some((plant.cellular.division_signal() / 2.0).clamp(0.0, 1.0))),
            metric("Leaf ATP", format!("{:.3}", leaf.state_atp), Some((leaf.state_atp / 5.0).clamp(0.0, 1.0))),
            metric("Root Water", format!("{:.3}", root.state_water), Some((root.state_water / 8.0).clamp(0.0, 1.0))),
            metric("Meristem Cycle Tx", format!("{:.3}", meristem.transcript_cell_cycle), Some((meristem.transcript_cell_cycle / 2.0).clamp(0.0, 1.0))),
        ],
        molecular: vec![
            metric("Root-Zone Water", format!("{:.3}", root_zone.water), Some((root_zone.water / 1.5).clamp(0.0, 1.0))),
            metric("Root-Zone Glucose", format!("{:.4}", root_zone.glucose), Some((root_zone.glucose / 0.12).clamp(0.0, 1.0))),
            metric("Root-Zone Amino", format!("{:.4}", root_zone.amino_acids), Some((root_zone.amino_acids / 0.028).clamp(0.0, 1.0))),
            metric("Root-Zone Nucleotide", format!("{:.4}", root_zone.nucleotides), Some((root_zone.nucleotides / 0.020).clamp(0.0, 1.0))),
            metric("Root-Zone Membrane Px", format!("{:.4}", root_zone.membrane_precursors), Some((root_zone.membrane_precursors / 0.018).clamp(0.0, 1.0))),
            metric("Root-Zone O2", format!("{:.4}", root_zone.oxygen), Some((root_zone.oxygen / 0.12).clamp(0.0, 1.0))),
            metric("Root-Zone CO2", format!("{:.4}", root_zone.co2), Some((root_zone.co2 / 0.12).clamp(0.0, 1.0))),
            metric("Root-Zone NH4+", format!("{:.4}", root_zone.ammonium), Some((root_zone.ammonium / 0.08).clamp(0.0, 1.0))),
            metric("Root-Zone NO3-", format!("{:.4}", root_zone.nitrate), Some((root_zone.nitrate / 0.08).clamp(0.0, 1.0))),
            metric("ATP Flux", format!("{:.4}", root_zone.atp_flux), Some((root_zone.atp_flux / 0.08).clamp(0.0, 1.0))),
            metric("Root-Zone Si(aq)", format!("{:.4}", root_zone.dissolved_silicate), Some((root_zone.dissolved_silicate / 0.09).clamp(0.0, 1.0))),
            metric("Root-Zone HCO3-", format!("{:.4}", root_zone.bicarbonate), Some((root_zone.bicarbonate / 0.18).clamp(0.0, 1.0))),
            metric("Root-Zone H+ surf", format!("{:.4}", root_zone.surface_proton_load), Some((root_zone.surface_proton_load / 0.18).clamp(0.0, 1.0))),
            metric("Root-Zone Ca(HCO3)2", format!("{:.4}", root_zone.calcium_bicarbonate_complex), Some((root_zone.calcium_bicarbonate_complex / 0.16).clamp(0.0, 1.0))),
            metric("Root-Zone Al(OH)3(s)", format!("{:.4}", root_zone.sorbed_aluminum_hydroxide), Some((root_zone.sorbed_aluminum_hydroxide / 0.12).clamp(0.0, 1.0))),
            metric("Root-Zone Fe(OH)3(s)", format!("{:.4}", root_zone.sorbed_ferric_hydroxide), Some((root_zone.sorbed_ferric_hydroxide / 0.12).clamp(0.0, 1.0))),
            metric("Root-Zone Ca exch", format!("{:.4}", root_zone.exchangeable_calcium), Some((root_zone.exchangeable_calcium / 0.18).clamp(0.0, 1.0))),
            metric("Root-Zone Al exch", format!("{:.4}", root_zone.exchangeable_aluminum), Some((root_zone.exchangeable_aluminum / 0.14).clamp(0.0, 1.0))),
            metric("Root-Zone Fe(aq)", format!("{:.4}", root_zone.aqueous_iron), Some((root_zone.aqueous_iron / 0.10).clamp(0.0, 1.0))),
            metric("Root-Zone Base Sat", format!("{:.3}", root_zone_base_saturation), Some(root_zone_base_saturation)),
            metric("Root-Zone Al Toxicity", format!("{:.3}", root_zone_al_toxicity), Some((root_zone_al_toxicity / 1.4).clamp(0.0, 1.0))),
            metric("Root-Zone Weathering", format!("{:.3}", root_zone_weathering_support), Some((root_zone_weathering_support / 1.6).clamp(0.0, 1.0))),
            metric("Root Inventory Water", format!("{:.4}", root_inventory.water), Some((root_inventory.water / 1.5).clamp(0.0, 1.0))),
            metric("Root Inventory Glucose", format!("{:.4}", root_inventory.glucose), Some((root_inventory.glucose / 0.12).clamp(0.0, 1.0))),
            metric("Root Inventory Amino", format!("{:.4}", root_inventory.amino_acids), Some((root_inventory.amino_acids / 0.028).clamp(0.0, 1.0))),
            metric("Root Inventory Nucleotide", format!("{:.4}", root_inventory.nucleotides), Some((root_inventory.nucleotides / 0.020).clamp(0.0, 1.0))),
            metric("Root Inventory Membrane Px", format!("{:.4}", root_inventory.membrane_precursors), Some((root_inventory.membrane_precursors / 0.018).clamp(0.0, 1.0))),
            metric("Metabolome Glucose", format!("{:.1}", plant.metabolome.glucose_count), Some((plant.metabolome.glucose_count as f32 / 600.0).clamp(0.0, 1.0))),
            metric("Metabolome O2", format!("{:.1}", plant.metabolome.oxygen_count), Some((plant.metabolome.oxygen_count as f32 / 1200.0).clamp(0.0, 1.0))),
            metric("Metabolome CO2", format!("{:.1}", plant.metabolome.co2_count), Some((plant.metabolome.co2_count as f32 / 1200.0).clamp(0.0, 1.0))),
            metric("Ethylene", format!("{:.2}", plant.metabolome.ethylene_count), Some((plant.metabolome.ethylene_count as f32 / 80.0).clamp(0.0, 1.0))),
        ],
        scene_grid: plant_morphology_grid(plant),
        cellular_grid: Some(plant_cellular_grid(&leaf, &stem, &root, &meristem)),
        molecular_grid: Some(plant_molecular_grid(&leaf, &stem, &root, &meristem)),
        composition: vec![
            composition("Water Pool", plant.cellular.water_pool(), "a.u.", "cellular"),
            composition("Sugar Pool", plant.cellular.sugar_pool(), "a.u.", "cellular"),
            composition("Nitrogen Pool", plant.cellular.nitrogen_pool(), "a.u.", "cellular"),
            composition("Stored Carbon", plant.physiology.storage_carbon(), "a.u.", "organism"),
            composition("Leaf Sucrose", leaf.cytoplasm_sucrose, "a.u.", "cellular"),
            composition("Stem Glucose", stem.state_glucose, "a.u.", "cellular"),
            composition("Root Water", root.state_water, "a.u.", "cellular"),
            composition("Meristem ATP", meristem.state_atp, "a.u.", "energy"),
            composition("Root-Zone Water", root_zone.water, "a.u.", "substrate"),
            composition("Root-Zone Glucose", root_zone.glucose, "a.u.", "substrate"),
            composition("Root-Zone Amino Acids", root_zone.amino_acids, "a.u.", "substrate"),
            composition("Root-Zone Nucleotides", root_zone.nucleotides, "a.u.", "substrate"),
            composition("Root-Zone Membrane Precursors", root_zone.membrane_precursors, "a.u.", "substrate"),
            composition("Root-Zone O2", root_zone.oxygen, "a.u.", "substrate"),
            composition("Root-Zone NH4+", root_zone.ammonium, "a.u.", "substrate"),
            composition("Root-Zone NO3-", root_zone.nitrate, "a.u.", "substrate"),
            composition("Root-Zone Dissolved Silicate", root_zone.dissolved_silicate, "a.u.", "substrate"),
            composition("Root-Zone Bicarbonate", root_zone.bicarbonate, "a.u.", "substrate"),
            composition("Root-Zone Surface Proton Load", root_zone.surface_proton_load, "a.u.", "substrate"),
            composition("Root-Zone Calcium Bicarbonate Complex", root_zone.calcium_bicarbonate_complex, "a.u.", "substrate"),
            composition("Root-Zone Sorbed Aluminum Hydroxide", root_zone.sorbed_aluminum_hydroxide, "a.u.", "substrate"),
            composition("Root-Zone Sorbed Ferric Hydroxide", root_zone.sorbed_ferric_hydroxide, "a.u.", "substrate"),
            composition("Root-Zone Exchangeable Calcium", root_zone.exchangeable_calcium, "a.u.", "substrate"),
            composition("Root-Zone Exchangeable Magnesium", root_zone.exchangeable_magnesium, "a.u.", "substrate"),
            composition("Root-Zone Exchangeable Potassium", root_zone.exchangeable_potassium, "a.u.", "substrate"),
            composition("Root-Zone Exchangeable Aluminum", root_zone.exchangeable_aluminum, "a.u.", "substrate"),
            composition("Root-Zone Aqueous Iron", root_zone.aqueous_iron, "a.u.", "substrate"),
            composition("Root Inventory Water", root_inventory.water, "a.u.", "inventory"),
            composition("Root Inventory Glucose", root_inventory.glucose, "a.u.", "inventory"),
            composition("Root Inventory Amino Acids", root_inventory.amino_acids, "a.u.", "inventory"),
            composition("Root Inventory Nucleotides", root_inventory.nucleotides, "a.u.", "inventory"),
            composition("Root Inventory Membrane Precursors", root_inventory.membrane_precursors, "a.u.", "inventory"),
            composition("Root Inventory Exchangeable Calcium", root_inventory.exchangeable_calcium, "a.u.", "inventory"),
            composition("Root Inventory Exchangeable Magnesium", root_inventory.exchangeable_magnesium, "a.u.", "inventory"),
            composition("Root Inventory Exchangeable Potassium", root_inventory.exchangeable_potassium, "a.u.", "inventory"),
            composition("Root Inventory Exchangeable Aluminum", root_inventory.exchangeable_aluminum, "a.u.", "inventory"),
            composition("Root Inventory Surface Proton Load", root_inventory.surface_proton_load, "a.u.", "inventory"),
            composition("Root Inventory Calcium Bicarbonate Complex", root_inventory.calcium_bicarbonate_complex, "a.u.", "inventory"),
            composition("Root Inventory Sorbed Aluminum Hydroxide", root_inventory.sorbed_aluminum_hydroxide, "a.u.", "inventory"),
            composition("Root Inventory Sorbed Ferric Hydroxide", root_inventory.sorbed_ferric_hydroxide, "a.u.", "inventory"),
            composition("Root Inventory Dissolved Silicate", root_inventory.dissolved_silicate, "a.u.", "inventory"),
            composition("Metabolome Glucose", plant.metabolome.glucose_count as f32, "molecules", "molecular"),
            composition("Metabolome Water", plant.metabolome.water_count as f32, "molecules", "molecular"),
            composition("Metabolome CO2", plant.metabolome.co2_count as f32, "molecules", "molecular"),
            composition("Metabolome O2", plant.metabolome.oxygen_count as f32, "molecules", "molecular"),
            composition("Starch Reserve", plant.metabolome.starch_reserve as f32, "molecules", "reserve"),
        ],
        notes: vec![
            format!("Time state: {} / {}", world.time_label(), world.circadian_label()),
            format!("Lunar phase: {}", world.moon_phase_name()),
            "Molecular counts come from the explicit plant metabolome inventory; cellular pools come from live tissue-cluster state.".into(),
            "Root inventory metrics come from the plant-owned local chemistry ledger that boundary-couples to the surrounding substrate patch.".into(),
            "Plant physiology remains authoritative; this panel only summarizes live cellular and root-zone chemistry.".into(),
        ],
    })
}

fn inspect_fly(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    index: usize,
) -> Result<InspectData, String> {
    let fly = world
        .flies
        .get(index)
        .ok_or_else(|| format!("Fly index {index} out of range"))?;
    let lifecycle_fly = world.fly_identities.get(index).and_then(|identity| {
        world
            .fly_population()
            .fly_by_organism_id(identity.organism_id)
    });
    let metabolism = world
        .fly_metabolisms
        .get(index)
        .ok_or_else(|| format!("Fly metabolism index {index} out of range"))?;
    let body = fly.body_state();
    let local_air = local_air(world, body.x, body.y);
    let visual = crate::terrarium::visual_projection::fly_visual_response(
        local_air,
        body,
        metabolism.energy_charge(),
        world.time_s,
        index as f32,
    );
    let linked_inventory =
        lifecycle_fly.map(|fly| inventory_patch_metrics(&fly.material_inventory));
    let mut summary = vec![
        metric(
            "Energy",
            format!("{:.1} uJ", body.energy),
            Some((body.energy / 5000.0).clamp(0.0, 1.0)),
        ),
        metric(
            "Flight State",
            if body.is_flying {
                "Airborne"
            } else {
                "Grounded"
            }
            .into(),
            Some(if body.is_flying { 1.0 } else { 0.0 }),
        ),
        metric(
            "Neurons",
            fly.scale.neuron_count().to_string(),
            Some(
                (fly.scale.neuron_count() as f32
                    / drosophila::DrosophilaScale::Large.neuron_count() as f32)
                    .clamp(0.0, 1.0),
            ),
        ),
        metric(
            "Wing Beat",
            format!("{:.1} Hz", body.wing_beat_freq),
            Some((body.wing_beat_freq / 350.0).clamp(0.0, 1.0)),
        ),
    ];
    let mut cellular = vec![
        metric(
            "Energy Charge",
            format!("{:.3}", metabolism.energy_charge()),
            Some(metabolism.energy_charge()),
        ),
        metric(
            "Substrate Saturation",
            format!("{:.3}", metabolism.substrate_saturation()),
            Some(metabolism.substrate_saturation()),
        ),
        metric(
            "Oxygen Status",
            format!("{:.3}", metabolism.oxygen_status()),
            Some(metabolism.oxygen_status()),
        ),
        metric(
            "Biomass",
            format!("{:.3} mg", metabolism.biomass_mg()),
            Some((metabolism.biomass_mg() / 1.0).clamp(0.0, 1.0)),
        ),
        metric(
            "Metabolic Rate",
            format!("{:.3}", metabolism.metabolic_rate_fraction()),
            Some(metabolism.metabolic_rate_fraction()),
        ),
        metric(
            "Hunger",
            format!("{:.3}", metabolism.hunger()),
            Some(metabolism.hunger()),
        ),
    ];
    let mut molecular = vec![
        metric(
            "Trehalose",
            format!("{:.3} mM", metabolism.hemolymph_trehalose_mm),
            Some((metabolism.hemolymph_trehalose_mm / 25.0).clamp(0.0, 1.0)),
        ),
        metric(
            "Glucose",
            format!("{:.3} mM", metabolism.hemolymph_glucose_mm),
            Some((metabolism.hemolymph_glucose_mm / 10.0).clamp(0.0, 1.0)),
        ),
        metric(
            "Glycogen",
            format!("{:.4} mg", metabolism.fat_body_glycogen_mg),
            Some((metabolism.fat_body_glycogen_mg / 0.03).clamp(0.0, 1.0)),
        ),
        metric(
            "Lipid",
            format!("{:.4} mg", metabolism.fat_body_lipid_mg),
            Some((metabolism.fat_body_lipid_mg / 0.06).clamp(0.0, 1.0)),
        ),
        metric(
            "ATP",
            format!("{:.3} mM", metabolism.muscle_atp_mm),
            Some((metabolism.muscle_atp_mm / 5.0).clamp(0.0, 1.0)),
        ),
        metric(
            "ADP",
            format!("{:.3} mM", metabolism.muscle_adp_mm),
            Some((metabolism.muscle_adp_mm / 5.0).clamp(0.0, 1.0)),
        ),
        metric(
            "Ambient O2",
            format!("{:.3}", metabolism.ambient_o2_fraction),
            Some((metabolism.ambient_o2_fraction / 0.21).clamp(0.0, 1.0)),
        ),
    ];
    let mut compositions = vec![
        composition(
            "Hemolymph Trehalose",
            metabolism.hemolymph_trehalose_mm,
            "mM",
            "circulation",
        ),
        composition(
            "Hemolymph Glucose",
            metabolism.hemolymph_glucose_mm,
            "mM",
            "circulation",
        ),
        composition(
            "Fat Body Glycogen",
            metabolism.fat_body_glycogen_mg,
            "mg",
            "reserve",
        ),
        composition(
            "Fat Body Lipid",
            metabolism.fat_body_lipid_mg,
            "mg",
            "reserve",
        ),
        composition("Muscle ATP", metabolism.muscle_atp_mm, "mM", "energy"),
        composition("Muscle ADP", metabolism.muscle_adp_mm, "mM", "energy"),
    ];
    let mut notes = vec![
        format!(
            "Reference genome: {:.0} Mb, {} genes, {} neurons full-fly reference.",
            fly_genome_stats::GENOME_MB,
            fly_genome_stats::PROTEIN_GENES,
            fly_genome_stats::FULL_NEURON_COUNT
        ),
        format!(
            "Active neural scaffold: {} neurons, {} neural windows per body step.",
            fly.scale.neuron_count(),
            fly.neural_steps_per_body
        ),
        "Behavior here is summarized from live body and metabolism state; inspection does not invent a new behavior controller.".into(),
    ];
    if let Some(linked_fly) = lifecycle_fly {
        summary.push(metric(
            "Life Stage",
            format!("{:?}", linked_fly.stage),
            None,
        ));
        summary.push(metric(
            "Sex",
            match linked_fly.sex {
                crate::drosophila_population::FlySex::Male => "Male",
                crate::drosophila_population::FlySex::Female => "Female",
            }
            .into(),
            Some(
                if matches!(linked_fly.sex, crate::drosophila_population::FlySex::Female) {
                    1.0
                } else {
                    0.0
                },
            ),
        ));
        cellular.push(metric(
            "Egg Capacity",
            linked_fly.eggs_remaining.to_string(),
            Some(
                (linked_fly.eggs_remaining as f32
                    / crate::drosophila_population::LIFETIME_EGGS as f32)
                    .clamp(0.0, 1.0),
            ),
        ));
        cellular.push(metric(
            "Mated",
            if linked_fly.mated { "Yes" } else { "No" }.into(),
            Some(if linked_fly.mated { 1.0 } else { 0.0 }),
        ));
        if let Some(inventory) = linked_inventory {
            molecular.push(metric(
                "Reproductive Ledger Water",
                format!("{:.3}", inventory.water),
                Some((inventory.water / 2.0).clamp(0.0, 1.0)),
            ));
            molecular.push(metric(
                "Reproductive Ledger Glucose",
                format!("{:.3}", inventory.glucose),
                Some((inventory.glucose / 2.0).clamp(0.0, 1.0)),
            ));
            molecular.push(metric(
                "Reproductive Ledger Amino",
                format!("{:.3}", inventory.amino_acids),
                Some((inventory.amino_acids / 1.5).clamp(0.0, 1.0)),
            ));
            molecular.push(metric(
                "Reproductive Ledger Nucleotide",
                format!("{:.3}", inventory.nucleotides),
                Some((inventory.nucleotides / 1.0).clamp(0.0, 1.0)),
            ));
            compositions.push(composition(
                "Reproductive Ledger Water",
                inventory.water,
                "a.u.",
                "ledger",
            ));
            compositions.push(composition(
                "Reproductive Ledger Glucose",
                inventory.glucose,
                "a.u.",
                "ledger",
            ));
            compositions.push(composition(
                "Reproductive Ledger Amino Acids",
                inventory.amino_acids,
                "a.u.",
                "ledger",
            ));
            compositions.push(composition(
                "Reproductive Ledger Nucleotides",
                inventory.nucleotides,
                "a.u.",
                "ledger",
            ));
        }
        notes.push(
            "Lifecycle and reproductive chemistry are linked to this neural fly by shared terrarium organism identity; the inspect panel is reading the same adult state the ecology step uses."
                .into(),
        );
    }

    Ok(InspectData {
        kind: "fly".into(),
        title: "Fruit Fly (Drosophila melanogaster)".into(),
        subtitle: format!("{:?} neural scale in {}", fly.scale, preset.label()),
        preset: preset.cli_name().into(),
        position: Some(InspectPosition {
            x: body.x,
            y: body.y,
            z: Some(body.z),
        }),
        summary,
        scene: vec![
            metric("Heading", format!("{:.2} rad", body.heading), None),
            metric(
                "Speed",
                format!("{:.2}", body.speed),
                Some((body.speed / 3.5).clamp(0.0, 1.0)),
            ),
            metric(
                "Altitude",
                format!("{:.2}", body.z),
                Some((body.z / 2.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Temperature",
                format!("{:.1} C", body.temperature),
                Some(((body.temperature - 5.0) / 30.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Local Humidity",
                format!("{:.2}", local_air.humidity),
                Some(local_air.humidity),
            ),
            metric(
                "Air Pressure",
                format!("{:.2} kPa", local_air.pressure_kpa),
                Some(((local_air.pressure_kpa - 95.0) / 12.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Proboscis",
                if body.proboscis_extended {
                    "Extended"
                } else {
                    "Retracted"
                }
                .into(),
                Some(if body.proboscis_extended { 1.0 } else { 0.0 }),
            ),
        ],
        cellular,
        molecular,
        scene_grid: Some(fly_scene_grid(body, &visual)),
        cellular_grid: Some(fly_cellular_grid(body, metabolism)),
        molecular_grid: Some(fly_molecular_grid(metabolism)),
        composition: compositions,
        notes,
    })
}

fn inspect_fly_egg(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    index: usize,
) -> Result<InspectData, String> {
    let cluster = world
        .fly_population()
        .egg_clusters
        .get(index)
        .ok_or_else(|| format!("Fly egg cluster index {index} out of range"))?;
    let (x, y) = fly_population_world_xy(world, (cluster.position.0, cluster.position.1, 0.0));
    let local_air = local_air(world, x, y);
    let inventory = inventory_patch_metrics(&cluster.material_inventory);
    let x_cell = clamp(x.floor(), 0.0, world.config.width.saturating_sub(1) as f32) as usize;
    let y_cell = clamp(y.floor(), 0.0, world.config.height.saturating_sub(1) as f32) as usize;
    let patch = substrate_patch_metrics(world, x_cell, y_cell, 0);
    let progress = clamp(cluster.age_hours / 22.0, 0.0, 1.0);

    Ok(InspectData {
        kind: "fly_egg".into(),
        title: "Fruit Fly Egg Cluster".into(),
        subtitle: format!("{} eggs in {}", cluster.count, preset.label()),
        preset: preset.cli_name().into(),
        position: Some(InspectPosition {
            x,
            y,
            z: Some(0.0),
        }),
        summary: vec![
            metric("Egg Count", cluster.count.to_string(), Some((cluster.count as f32 / 20.0).clamp(0.0, 1.0))),
            metric("Development Age", format!("{:.2} h", cluster.age_hours), Some(progress)),
            metric("Development Progress", format!("{:.3}", progress), Some(progress)),
            metric("Substrate Quality", format!("{:.3}", cluster.substrate_quality), Some((cluster.substrate_quality / 1.4).clamp(0.0, 1.0))),
            metric("Mean Viability", format!("{:.3}", cluster.mean_embryo_viability()), Some(cluster.mean_embryo_viability())),
        ],
        scene: vec![
            metric("Surface Moisture", format!("{:.3}", world.moisture[y_cell * world.config.width + x_cell]), Some(world.moisture[y_cell * world.config.width + x_cell])),
            metric("Local Humidity", format!("{:.2}", local_air.humidity), Some(local_air.humidity)),
            metric("Temperature", format!("{:.1} C", local_air.temperature_c), Some(((local_air.temperature_c - 5.0) / 30.0).clamp(0.0, 1.0))),
            metric("Air Pressure", format!("{:.2} kPa", local_air.pressure_kpa), Some(((local_air.pressure_kpa - 95.0) / 12.0).clamp(0.0, 1.0))),
            metric("Fruit-Zone Glucose", format!("{:.3}", patch.glucose), Some((patch.glucose / 1.5).clamp(0.0, 1.0))),
            metric("Fruit-Zone O2", format!("{:.3}", patch.oxygen), Some((patch.oxygen / 0.3).clamp(0.0, 1.0))),
        ],
        cellular: vec![
            metric("Development Progress", format!("{:.3}", progress), Some(progress)),
            metric("Cluster Water", format!("{:.3}", inventory.water), Some((inventory.water / 2.0).clamp(0.0, 1.0))),
            metric("Cluster Glucose", format!("{:.3}", inventory.glucose), Some((inventory.glucose / 2.0).clamp(0.0, 1.0))),
            metric("Cluster Amino", format!("{:.3}", inventory.amino_acids), Some((inventory.amino_acids / 1.5).clamp(0.0, 1.0))),
            metric("Cluster O2", format!("{:.3}", inventory.oxygen), Some((inventory.oxygen / 0.3).clamp(0.0, 1.0))),
            metric("Substrate Quality", format!("{:.3}", cluster.substrate_quality), Some((cluster.substrate_quality / 1.4).clamp(0.0, 1.0))),
        ],
        molecular: vec![
            metric("Ledger Water", format!("{:.3}", inventory.water), Some((inventory.water / 2.0).clamp(0.0, 1.0))),
            metric("Ledger Glucose", format!("{:.3}", inventory.glucose), Some((inventory.glucose / 2.0).clamp(0.0, 1.0))),
            metric("Ledger Amino", format!("{:.3}", inventory.amino_acids), Some((inventory.amino_acids / 1.5).clamp(0.0, 1.0))),
            metric("Ledger Nucleotide", format!("{:.3}", inventory.nucleotides), Some((inventory.nucleotides / 1.0).clamp(0.0, 1.0))),
            metric("Ledger Membrane", format!("{:.3}", inventory.membrane_precursors), Some((inventory.membrane_precursors / 1.0).clamp(0.0, 1.0))),
            metric("Ledger O2", format!("{:.3}", inventory.oxygen), Some((inventory.oxygen / 0.3).clamp(0.0, 1.0))),
        ],
        scene_grid: Some(fly_egg_scene_grid(cluster)),
        cellular_grid: Some(fly_egg_cellular_grid(cluster, &inventory)),
        molecular_grid: Some(fly_inventory_molecular_grid(
            "Egg-cluster chemistry ledger: H2O / glucose / amino / nucleotide / membrane / O2",
            &inventory,
        )),
        composition: vec![
            composition("Egg Cluster Water", inventory.water, "a.u.", "ledger"),
            composition("Egg Cluster Glucose", inventory.glucose, "a.u.", "ledger"),
            composition("Egg Cluster Amino Acids", inventory.amino_acids, "a.u.", "ledger"),
            composition("Egg Cluster Nucleotides", inventory.nucleotides, "a.u.", "ledger"),
            composition("Egg Cluster Membrane Precursors", inventory.membrane_precursors, "a.u.", "ledger"),
            composition("Egg Cluster O2", inventory.oxygen, "a.u.", "ledger"),
        ],
        notes: vec![
            "This inspect target comes from the explicit fly egg-cluster population layer, not from the adult fly renderer.".into(),
            "The clutch owns a fruit-surface microdomain, while each embryo carries its own internal chemistry and development state underneath this summary.".into(),
        ],
    })
}

fn inspect_fly_embryo(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    index: usize,
) -> Result<InspectData, String> {
    let (cluster_index, _, cluster, embryo) = world
        .fly_population()
        .embryo_at_flat_index(index)
        .ok_or_else(|| format!("Fly embryo index {index} out of range"))?;
    let position_mm = cluster.embryo_position_mm(embryo);
    let (x, y) = fly_population_world_xy(world, position_mm);
    let local_air = local_air(world, x, y);
    let inventory = inventory_patch_metrics(&embryo.material_inventory);
    let x_cell = clamp(x.floor(), 0.0, world.config.width.saturating_sub(1) as f32) as usize;
    let y_cell = clamp(y.floor(), 0.0, world.config.height.saturating_sub(1) as f32) as usize;
    let patch = substrate_patch_metrics(world, x_cell, y_cell, 0);
    let progress = clamp(embryo.age_hours / 22.0, 0.0, 1.0);

    Ok(InspectData {
        kind: "fly_embryo".into(),
        title: "Fruit Fly Embryo".into(),
        subtitle: format!("Embryo #{} in clutch {} ({})", embryo.id, cluster_index, preset.label()),
        preset: preset.cli_name().into(),
        position: Some(InspectPosition {
            x,
            y,
            z: Some(0.0),
        }),
        summary: vec![
            metric("Embryo ID", embryo.id.to_string(), None),
            metric(
                "Sex",
                format!("{:?}", embryo.sex),
                Some(if matches!(embryo.sex, crate::drosophila_population::FlySex::Female) {
                    1.0
                } else {
                    0.0
                }),
            ),
            metric("Development Age", format!("{:.2} h", embryo.age_hours), Some(progress)),
            metric("Development Progress", format!("{:.3}", progress), Some(progress)),
            metric("Viability", format!("{:.3}", embryo.viability), Some(embryo.viability)),
        ],
        scene: vec![
            metric("Clutch Size", cluster.count.to_string(), Some((cluster.count as f32 / 20.0).clamp(0.0, 1.0))),
            metric("Surface Moisture", format!("{:.3}", world.moisture[y_cell * world.config.width + x_cell]), Some(world.moisture[y_cell * world.config.width + x_cell])),
            metric("Local Humidity", format!("{:.2}", local_air.humidity), Some(local_air.humidity)),
            metric("Temperature", format!("{:.1} C", local_air.temperature_c), Some(((local_air.temperature_c - 5.0) / 30.0).clamp(0.0, 1.0))),
            metric("Fruit-Zone Glucose", format!("{:.3}", patch.glucose), Some((patch.glucose / 1.5).clamp(0.0, 1.0))),
            metric("Fruit-Zone O2", format!("{:.3}", patch.oxygen), Some((patch.oxygen / 0.3).clamp(0.0, 1.0))),
        ],
        cellular: vec![
            metric("Viability", format!("{:.3}", embryo.viability), Some(embryo.viability)),
            metric("Internal Water", format!("{:.3}", inventory.water), Some((inventory.water / 0.2).clamp(0.0, 1.0))),
            metric("Internal Glucose", format!("{:.3}", inventory.glucose), Some((inventory.glucose / 0.2).clamp(0.0, 1.0))),
            metric("Internal Amino", format!("{:.3}", inventory.amino_acids), Some((inventory.amino_acids / 0.15).clamp(0.0, 1.0))),
            metric("Internal Nucleotide", format!("{:.3}", inventory.nucleotides), Some((inventory.nucleotides / 0.12).clamp(0.0, 1.0))),
            metric("Internal O2", format!("{:.3}", inventory.oxygen), Some((inventory.oxygen / 0.1).clamp(0.0, 1.0))),
        ],
        molecular: vec![
            metric("Ledger Water", format!("{:.3}", inventory.water), Some((inventory.water / 0.2).clamp(0.0, 1.0))),
            metric("Ledger Glucose", format!("{:.3}", inventory.glucose), Some((inventory.glucose / 0.2).clamp(0.0, 1.0))),
            metric("Ledger Amino", format!("{:.3}", inventory.amino_acids), Some((inventory.amino_acids / 0.15).clamp(0.0, 1.0))),
            metric("Ledger Nucleotide", format!("{:.3}", inventory.nucleotides), Some((inventory.nucleotides / 0.12).clamp(0.0, 1.0))),
            metric("Ledger Membrane", format!("{:.3}", inventory.membrane_precursors), Some((inventory.membrane_precursors / 0.10).clamp(0.0, 1.0))),
            metric("Ledger O2", format!("{:.3}", inventory.oxygen), Some((inventory.oxygen / 0.1).clamp(0.0, 1.0))),
        ],
        scene_grid: Some(fly_embryo_scene_grid(embryo)),
        cellular_grid: Some(fly_embryo_cellular_grid(embryo, &inventory)),
        molecular_grid: Some(fly_inventory_molecular_grid(
            "Embryo chemistry ledger: H2O / glucose / amino / nucleotide / membrane / O2",
            &inventory,
        )),
        composition: vec![
            composition("Embryo Water", inventory.water, "a.u.", "ledger"),
            composition("Embryo Glucose", inventory.glucose, "a.u.", "ledger"),
            composition("Embryo Amino Acids", inventory.amino_acids, "a.u.", "ledger"),
            composition("Embryo Nucleotides", inventory.nucleotides, "a.u.", "ledger"),
            composition("Embryo Membrane Precursors", inventory.membrane_precursors, "a.u.", "ledger"),
            composition("Embryo O2", inventory.oxygen, "a.u.", "ledger"),
        ],
        notes: vec![
            "This inspect target is a single embryo owned by the explicit clutch, not a renderer-inferred sub-selection.".into(),
            "Embryo chemistry comes from its persistent internal ledger while the clutch remains the shared fruit-surface microdomain.".into(),
        ],
    })
}

fn inspect_fly_larva(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    index: usize,
) -> Result<InspectData, String> {
    let larva = nth_fly_lifecycle_stage(world, index, |fly| match fly.stage {
        FlyLifeStage::Larva { .. } => Some(fly),
        _ => None,
    })
    .ok_or_else(|| format!("Fly larva index {index} out of range"))?;
    inspect_fly_immature(world, preset, "fly_larva", "Fruit Fly Larva", larva)
}

fn inspect_fly_pupa(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    index: usize,
) -> Result<InspectData, String> {
    let pupa = nth_fly_lifecycle_stage(world, index, |fly| match fly.stage {
        FlyLifeStage::Pupa { .. } => Some(fly),
        _ => None,
    })
    .ok_or_else(|| format!("Fly pupa index {index} out of range"))?;
    inspect_fly_immature(world, preset, "fly_pupa", "Fruit Fly Pupa", pupa)
}

fn inspect_fly_immature(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    kind: &str,
    title: &str,
    fly: &Fly,
) -> Result<InspectData, String> {
    let (x, y) = fly_population_world_xy(world, fly.position);
    let z = fly.position.2 / world.config.cell_size_mm.max(1.0e-3);
    let local_air = local_air(world, x, y);
    let inventory = inventory_patch_metrics(&fly.material_inventory);
    let x_cell = clamp(x.floor(), 0.0, world.config.width.saturating_sub(1) as f32) as usize;
    let y_cell = clamp(y.floor(), 0.0, world.config.height.saturating_sub(1) as f32) as usize;
    let patch = substrate_patch_metrics(world, x_cell, y_cell, 0);
    let (stage_label, age_hours, scene_grid, progress) = match fly.stage {
        FlyLifeStage::Larva { instar, age_hours } => (
            format!("Larva L{}", instar),
            age_hours,
            fly_larva_scene_grid(instar, fly.energy / FLY_ENERGY_MAX),
            clamp(age_hours / fly_larval_duration_hours(instar), 0.0, 1.0),
        ),
        FlyLifeStage::Pupa { age_hours } => (
            "Pupa".into(),
            age_hours,
            fly_pupa_scene_grid(age_hours / 96.0),
            clamp(age_hours / 96.0, 0.0, 1.0),
        ),
        _ => return Err(format!("Unsupported immature fly stage for {kind}")),
    };

    Ok(InspectData {
        kind: kind.into(),
        title: title.into(),
        subtitle: format!("{stage_label} in {}", preset.label()),
        preset: preset.cli_name().into(),
        position: Some(InspectPosition { x, y, z: Some(z) }),
        summary: vec![
            metric("Stage", stage_label, None),
            metric("Sex", format!("{:?}", fly.sex), Some(if matches!(fly.sex, crate::drosophila_population::FlySex::Female) { 1.0 } else { 0.0 })),
            metric("Energy", format!("{:.1} uJ", fly.energy), Some((fly.energy / FLY_ENERGY_MAX).clamp(0.0, 1.0))),
            metric("Development Progress", format!("{:.3}", progress), Some(progress)),
        ],
        scene: vec![
            metric("Age", format!("{:.2} h", age_hours), Some(progress)),
            metric("Surface Moisture", format!("{:.3}", world.moisture[y_cell * world.config.width + x_cell]), Some(world.moisture[y_cell * world.config.width + x_cell])),
            metric("Local Humidity", format!("{:.2}", local_air.humidity), Some(local_air.humidity)),
            metric("Temperature", format!("{:.1} C", local_air.temperature_c), Some(((local_air.temperature_c - 5.0) / 30.0).clamp(0.0, 1.0))),
            metric("Fruit-Zone Glucose", format!("{:.3}", patch.glucose), Some((patch.glucose / 1.5).clamp(0.0, 1.0))),
            metric("Fruit-Zone Amino", format!("{:.3}", patch.amino_acids), Some((patch.amino_acids / 1.2).clamp(0.0, 1.0))),
        ],
        cellular: vec![
            metric("Energy Fraction", format!("{:.3}", fly.energy / FLY_ENERGY_MAX), Some((fly.energy / FLY_ENERGY_MAX).clamp(0.0, 1.0))),
            metric("Ledger Water", format!("{:.3}", inventory.water), Some((inventory.water / 2.0).clamp(0.0, 1.0))),
            metric("Ledger Glucose", format!("{:.3}", inventory.glucose), Some((inventory.glucose / 2.0).clamp(0.0, 1.0))),
            metric("Ledger Amino", format!("{:.3}", inventory.amino_acids), Some((inventory.amino_acids / 1.5).clamp(0.0, 1.0))),
            metric("Ledger O2", format!("{:.3}", inventory.oxygen), Some((inventory.oxygen / 0.3).clamp(0.0, 1.0))),
            metric("Development Progress", format!("{:.3}", progress), Some(progress)),
        ],
        molecular: vec![
            metric("Ledger Water", format!("{:.3}", inventory.water), Some((inventory.water / 2.0).clamp(0.0, 1.0))),
            metric("Ledger Glucose", format!("{:.3}", inventory.glucose), Some((inventory.glucose / 2.0).clamp(0.0, 1.0))),
            metric("Ledger Amino", format!("{:.3}", inventory.amino_acids), Some((inventory.amino_acids / 1.5).clamp(0.0, 1.0))),
            metric("Ledger Nucleotide", format!("{:.3}", inventory.nucleotides), Some((inventory.nucleotides / 1.0).clamp(0.0, 1.0))),
            metric("Ledger Membrane", format!("{:.3}", inventory.membrane_precursors), Some((inventory.membrane_precursors / 1.0).clamp(0.0, 1.0))),
            metric("Ledger O2", format!("{:.3}", inventory.oxygen), Some((inventory.oxygen / 0.3).clamp(0.0, 1.0))),
        ],
        scene_grid: Some(scene_grid),
        cellular_grid: Some(fly_immature_cellular_grid(fly, &inventory, progress)),
        molecular_grid: Some(fly_inventory_molecular_grid(
            "Immature fly chemistry ledger: H2O / glucose / amino / nucleotide / membrane / O2",
            &inventory,
        )),
        composition: vec![
            composition("Immature Fly Water", inventory.water, "a.u.", "ledger"),
            composition("Immature Fly Glucose", inventory.glucose, "a.u.", "ledger"),
            composition("Immature Fly Amino Acids", inventory.amino_acids, "a.u.", "ledger"),
            composition("Immature Fly Nucleotides", inventory.nucleotides, "a.u.", "ledger"),
            composition("Immature Fly Membrane Precursors", inventory.membrane_precursors, "a.u.", "ledger"),
            composition("Immature Fly O2", inventory.oxygen, "a.u.", "ledger"),
        ],
        notes: vec![
            "This inspect target comes from the explicit fly lifecycle population layer, so larvae and pupae can be inspected directly instead of through an adult proxy.".into(),
            "Chemistry is read from the same persistent local ledger that governs immature feeding, metamorphosis, and survival.".into(),
        ],
    })
}

fn inspect_fruit(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    index: usize,
) -> Result<InspectData, String> {
    let fruit = world
        .fruits
        .get(index)
        .ok_or_else(|| format!("Fruit index {index} out of range"))?;
    let local_air = local_air(
        world,
        fruit.source.x as f32 + 0.5,
        fruit.source.y as f32 + 0.5,
    );
    let patch = substrate_patch_metrics(
        world,
        fruit.source.x,
        fruit.source.y,
        fruit.source.z.min(world.config.depth.saturating_sub(1)),
    );
    let surface_inventory = inventory_patch_metrics(&fruit.material_inventory);
    let embryo_inventory = fruit
        .reproduction
        .as_ref()
        .map(|reproduction| inventory_patch_metrics(&reproduction.material_inventory));
    let shape = crate::terrarium::shape_projection::fruit_shape_descriptor(
        &fruit.source_genome,
        fruit.radius,
        fruit.source.ripeness,
        fruit.source.sugar_content,
    );
    let title = format!(
        "{} ({})",
        crate::terrarium::plant_species::plant_common_name(fruit.taxonomy_id),
        crate::terrarium::plant_species::plant_scientific_name(fruit.taxonomy_id)
    );

    Ok(InspectData {
        kind: "fruit".into(),
        title,
        subtitle: format!("Fruit body in {}", preset.label()),
        preset: preset.cli_name().into(),
        position: Some(InspectPosition {
            x: fruit.source.x as f32 + 0.5,
            y: fruit.source.y as f32 + 0.5,
            z: Some(fruit.source.z as f32),
        }),
        summary: vec![
            metric(
                "Ripeness",
                format!("{:.3}", fruit.source.ripeness),
                Some(fruit.source.ripeness),
            ),
            metric(
                "Sugar",
                format!("{:.3}", fruit.source.sugar_content),
                Some((fruit.source.sugar_content / 1.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Radius",
                format!("{:.2}", fruit.radius),
                Some((fruit.radius / 1.2).clamp(0.0, 1.0)),
            ),
            metric(
                "Alive",
                if fruit.source.alive { "Yes" } else { "No" }.into(),
                Some(if fruit.source.alive { 1.0 } else { 0.0 }),
            ),
        ],
        scene: vec![
            metric(
                "Temperature",
                format!("{:.1} C", local_air.temperature_c),
                Some(((local_air.temperature_c - 5.0) / 30.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Humidity",
                format!("{:.2}", local_air.humidity),
                Some(local_air.humidity),
            ),
            metric(
                "Odor Emission",
                format!("{:.4}", fruit.source.odorant_emission_rate),
                Some((fruit.source.odorant_emission_rate / 0.02).clamp(0.0, 1.0)),
            ),
            metric(
                "Decay Rate",
                format!("{:.4}", fruit.source.decay_rate),
                Some((fruit.source.decay_rate / 0.02).clamp(0.0, 1.0)),
            ),
        ],
        cellular: vec![
            metric(
                "Seed Mass Trait",
                format!("{:.3}", fruit.source_genome.seed_mass),
                Some((fruit.source_genome.seed_mass / 0.2).clamp(0.0, 1.0)),
            ),
            metric(
                "Leaf Efficiency Trait",
                format!("{:.3}", fruit.source_genome.leaf_efficiency),
                Some((fruit.source_genome.leaf_efficiency / 1.6).clamp(0.0, 1.0)),
            ),
            metric(
                "Volatile Trait",
                format!("{:.3}", fruit.source_genome.volatile_scale),
                Some((fruit.source_genome.volatile_scale / 1.8).clamp(0.0, 1.0)),
            ),
        ],
        molecular: vec![
            metric(
                "Patch Water",
                format!("{:.3}", patch.water),
                Some((patch.water / 1.5).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch Glucose",
                format!("{:.4}", patch.glucose),
                Some((patch.glucose / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch Amino",
                format!("{:.4}", patch.amino_acids),
                Some((patch.amino_acids / 0.028).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch Nucleotide",
                format!("{:.4}", patch.nucleotides),
                Some((patch.nucleotides / 0.020).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch Membrane Px",
                format!("{:.4}", patch.membrane_precursors),
                Some((patch.membrane_precursors / 0.018).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch O2",
                format!("{:.4}", patch.oxygen),
                Some((patch.oxygen / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch CO2",
                format!("{:.4}", patch.co2),
                Some((patch.co2 / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "Surface Inventory Glucose",
                format!("{:.4}", surface_inventory.glucose),
                Some((surface_inventory.glucose / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "Surface Inventory Amino",
                format!("{:.4}", surface_inventory.amino_acids),
                Some((surface_inventory.amino_acids / 0.028).clamp(0.0, 1.0)),
            ),
            metric(
                "Embryo Inventory Glucose",
                format!(
                    "{:.4}",
                    embryo_inventory.map(|inventory| inventory.glucose).unwrap_or(0.0)
                ),
                Some(
                    (embryo_inventory.map(|inventory| inventory.glucose).unwrap_or(0.0) / 0.12)
                        .clamp(0.0, 1.0),
                ),
            ),
            metric(
                "Embryo Inventory Nucleotide",
                format!(
                    "{:.4}",
                    embryo_inventory
                        .map(|inventory| inventory.nucleotides)
                        .unwrap_or(0.0)
                ),
                Some(
                    (embryo_inventory
                        .map(|inventory| inventory.nucleotides)
                        .unwrap_or(0.0)
                        / 0.020)
                        .clamp(0.0, 1.0),
                ),
            ),
        ],
        scene_grid: Some(fruit_scene_grid(shape, fruit.source.ripeness)),
        cellular_grid: Some(fruit_cellular_grid(fruit)),
        molecular_grid: Some(substrate_vertical_grid(
            world,
            fruit.source.x,
            fruit.source.y,
            "Depth x chemistry slice below fruit: H2O / glucose / amino / nucleotide / membrane / O2 / CO2 / NH4 / NO3 / ATP / Si / HCO3 / H+surf / Ca(HCO3)2 / AlOH3(s) / FeOH3(s) / Ca / Mg / K / Na / Al / Fe",
        )),
        composition: vec![
            composition("Fruit Sugar", fruit.source.sugar_content, "a.u.", "fruit"),
            composition("Fruit Water", patch.water, "a.u.", "substrate"),
            composition("Fruit-zone Glucose", patch.glucose, "a.u.", "substrate"),
            composition("Fruit-zone Amino Acids", patch.amino_acids, "a.u.", "substrate"),
            composition("Fruit-zone Nucleotides", patch.nucleotides, "a.u.", "substrate"),
            composition("Fruit-zone Membrane Precursors", patch.membrane_precursors, "a.u.", "substrate"),
            composition("Fruit-zone O2", patch.oxygen, "a.u.", "substrate"),
            composition("Fruit-zone CO2", patch.co2, "a.u.", "substrate"),
            composition("Surface Inventory Water", surface_inventory.water, "a.u.", "inventory"),
            composition("Surface Inventory Glucose", surface_inventory.glucose, "a.u.", "inventory"),
            composition("Surface Inventory Amino Acids", surface_inventory.amino_acids, "a.u.", "inventory"),
            composition(
                "Embryo Inventory Glucose",
                embryo_inventory.map(|inventory| inventory.glucose).unwrap_or(0.0),
                "a.u.",
                "inventory",
            ),
            composition(
                "Embryo Inventory Nucleotides",
                embryo_inventory
                    .map(|inventory| inventory.nucleotides)
                    .unwrap_or(0.0),
                "a.u.",
                "inventory",
            ),
        ],
        notes: vec![
            "Fruit chemistry is read from live source state plus surrounding substrate patch."
                .into(),
            "Surface and embryo inventory metrics come from persistent local chemistry ledgers, not renderer-side placeholders."
                .into(),
        ],
    })
}

fn inspect_seed(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    index: usize,
) -> Result<InspectData, String> {
    let seed = world
        .seeds
        .get(index)
        .ok_or_else(|| format!("Seed index {index} out of range"))?;
    let (x, y) = clamped_cell(world, seed.x, seed.y);
    let probe_z = crate::terrarium::seed_microsite::seed_probe_z(
        seed,
        world.config.cell_size_mm,
        world.config.depth,
    );
    let patch = substrate_patch_metrics(world, x, y, probe_z);
    let inventory = inventory_patch_metrics(&seed.material_inventory);
    let patch_proton =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Proton, x, y, probe_z, 1);
    let patch_base_saturation = soil_base_saturation(
        patch.exchangeable_calcium,
        patch.exchangeable_magnesium,
        patch.exchangeable_potassium,
        patch.exchangeable_sodium,
        patch.exchangeable_aluminum,
        patch_proton,
        patch.surface_proton_load,
    );
    let patch_al_toxicity = soil_aluminum_toxicity(
        patch.exchangeable_aluminum,
        patch_proton,
        patch.surface_proton_load,
        patch_base_saturation,
    );
    let shape = crate::terrarium::shape_projection::seed_shape_descriptor(
        &seed.genome,
        seed.reserve_carbon,
        seed.dormancy_s,
    );
    let feedback = seed.cellular.last_feedback();
    let coat = seed.cellular.cluster_snapshot(SeedTissue::Coat);
    let endosperm = seed.cellular.cluster_snapshot(SeedTissue::Endosperm);
    let radicle = seed.cellular.cluster_snapshot(SeedTissue::Radicle);
    let cotyledon = seed.cellular.cluster_snapshot(SeedTissue::Cotyledon);
    Ok(InspectData {
        kind: "seed".into(),
        title: format!(
            "{} ({})",
            crate::terrarium::plant_species::plant_common_name(seed.genome.taxonomy_id),
            crate::terrarium::plant_species::plant_scientific_name(seed.genome.taxonomy_id)
        ),
        subtitle: format!("Seed state in {}", preset.label()),
        preset: preset.cli_name().into(),
        position: Some(InspectPosition {
            x: seed.x,
            y: seed.y,
            z: Some(probe_z as f32),
        }),
        summary: vec![
            metric("Reserve Carbon", format!("{:.3}", seed.reserve_carbon), Some((seed.reserve_carbon / 0.2).clamp(0.0, 1.0))),
            metric("Dormancy", format!("{:.0} s", seed.dormancy_s), Some((seed.dormancy_s / 20_000.0).clamp(0.0, 1.0))),
            metric("Age", format!("{:.0} s", seed.age_s), Some((seed.age_s / 20_000.0).clamp(0.0, 1.0))),
        ],
        scene: vec![
            metric("Hydration", format!("{:.3}", seed.cellular.hydration()), Some(seed.cellular.hydration())),
            metric("Patch Moisture", format!("{:.3}", world.moisture[y * world.config.width + x]), Some((world.moisture[y * world.config.width + x] / 1.0).clamp(0.0, 1.0))),
            metric("Canopy Cover", format!("{:.3}", world.canopy_cover[y * world.config.width + x]), Some(world.canopy_cover[y * world.config.width + x].clamp(0.0, 1.0))),
            metric("Burial Depth", format!("{:.3} mm", seed.microsite.burial_depth_mm), Some((seed.microsite.burial_depth_mm / world.config.cell_size_mm.max(1.0e-3)).clamp(0.0, 1.0))),
            metric("Surface Exposure", format!("{:.3}", seed.microsite.surface_exposure), Some(seed.microsite.surface_exposure)),
        ],
        cellular: vec![
            metric("Vitality", format!("{:.3}", seed.cellular.vitality()), Some(seed.cellular.vitality())),
            metric("Energy Charge", format!("{:.3}", seed.cellular.energy_charge()), Some(seed.cellular.energy_charge())),
            metric("Reserve Carbon Eq.", format!("{:.3}", seed.cellular.reserve_carbon_equivalent()), Some((seed.cellular.reserve_carbon_equivalent() / 0.2).clamp(0.0, 1.0))),
        ],
        molecular: vec![
            metric("Patch Water", format!("{:.3}", patch.water), Some((patch.water / 1.5).clamp(0.0, 1.0))),
            metric("Patch Glucose", format!("{:.4}", patch.glucose), Some((patch.glucose / 0.12).clamp(0.0, 1.0))),
            metric("Patch Amino", format!("{:.4}", patch.amino_acids), Some((patch.amino_acids / 0.028).clamp(0.0, 1.0))),
            metric("Patch Nucleotide", format!("{:.4}", patch.nucleotides), Some((patch.nucleotides / 0.020).clamp(0.0, 1.0))),
            metric("Patch Membrane Px", format!("{:.4}", patch.membrane_precursors), Some((patch.membrane_precursors / 0.018).clamp(0.0, 1.0))),
            metric("Patch NH4+", format!("{:.4}", patch.ammonium), Some((patch.ammonium / 0.08).clamp(0.0, 1.0))),
            metric("Patch NO3-", format!("{:.4}", patch.nitrate), Some((patch.nitrate / 0.08).clamp(0.0, 1.0))),
            metric("Patch HCO3-", format!("{:.4}", patch.bicarbonate), Some((patch.bicarbonate / 0.18).clamp(0.0, 1.0))),
            metric("Patch H+ surf", format!("{:.4}", patch.surface_proton_load), Some((patch.surface_proton_load / 0.18).clamp(0.0, 1.0))),
            metric("Patch Ca(HCO3)2", format!("{:.4}", patch.calcium_bicarbonate_complex), Some((patch.calcium_bicarbonate_complex / 0.16).clamp(0.0, 1.0))),
            metric("Patch Al(OH)3(s)", format!("{:.4}", patch.sorbed_aluminum_hydroxide), Some((patch.sorbed_aluminum_hydroxide / 0.12).clamp(0.0, 1.0))),
            metric("Patch Fe(OH)3(s)", format!("{:.4}", patch.sorbed_ferric_hydroxide), Some((patch.sorbed_ferric_hydroxide / 0.12).clamp(0.0, 1.0))),
            metric("Patch Ca exch", format!("{:.4}", patch.exchangeable_calcium), Some((patch.exchangeable_calcium / 0.18).clamp(0.0, 1.0))),
            metric("Patch Al exch", format!("{:.4}", patch.exchangeable_aluminum), Some((patch.exchangeable_aluminum / 0.14).clamp(0.0, 1.0))),
            metric("Patch Si(aq)", format!("{:.4}", patch.dissolved_silicate), Some((patch.dissolved_silicate / 0.09).clamp(0.0, 1.0))),
            metric("Patch Base Sat", format!("{:.3}", patch_base_saturation), Some(patch_base_saturation)),
            metric("Patch Al Toxicity", format!("{:.3}", patch_al_toxicity), Some((patch_al_toxicity / 1.4).clamp(0.0, 1.0))),
            metric("Seed Inventory Water", format!("{:.4}", inventory.water), Some((inventory.water / 1.5).clamp(0.0, 1.0))),
            metric("Seed Inventory Glucose", format!("{:.4}", inventory.glucose), Some((inventory.glucose / 0.12).clamp(0.0, 1.0))),
            metric("Seed Inventory Amino", format!("{:.4}", inventory.amino_acids), Some((inventory.amino_acids / 0.028).clamp(0.0, 1.0))),
            metric("Seed Inventory Nucleotide", format!("{:.4}", inventory.nucleotides), Some((inventory.nucleotides / 0.020).clamp(0.0, 1.0))),
            metric("Seed Inventory Membrane Px", format!("{:.4}", inventory.membrane_precursors), Some((inventory.membrane_precursors / 0.018).clamp(0.0, 1.0))),
        ],
        scene_grid: Some(seed_scene_grid(shape, feedback)),
        cellular_grid: Some(seed_cellular_grid(&coat, &endosperm, &radicle, &cotyledon)),
        molecular_grid: Some(seed_molecular_grid(&coat, &endosperm, &radicle, &cotyledon)),
        composition: vec![
            composition("Reserve Carbon", seed.reserve_carbon, "a.u.", "seed"),
            composition("Reserve Carbon Eq.", seed.cellular.reserve_carbon_equivalent(), "a.u.", "seed"),
            composition("Patch Water", patch.water, "a.u.", "substrate"),
            composition("Patch Glucose", patch.glucose, "a.u.", "substrate"),
            composition("Patch Amino Acids", patch.amino_acids, "a.u.", "substrate"),
            composition("Patch Nucleotides", patch.nucleotides, "a.u.", "substrate"),
            composition("Patch Membrane Precursors", patch.membrane_precursors, "a.u.", "substrate"),
            composition("Patch NH4+", patch.ammonium, "a.u.", "substrate"),
            composition("Patch Dissolved Silicate", patch.dissolved_silicate, "a.u.", "substrate"),
            composition("Patch Bicarbonate", patch.bicarbonate, "a.u.", "substrate"),
            composition("Patch Surface Proton Load", patch.surface_proton_load, "a.u.", "substrate"),
            composition("Patch Calcium Bicarbonate Complex", patch.calcium_bicarbonate_complex, "a.u.", "substrate"),
            composition("Patch Sorbed Aluminum Hydroxide", patch.sorbed_aluminum_hydroxide, "a.u.", "substrate"),
            composition("Patch Sorbed Ferric Hydroxide", patch.sorbed_ferric_hydroxide, "a.u.", "substrate"),
            composition("Patch Exchangeable Calcium", patch.exchangeable_calcium, "a.u.", "substrate"),
            composition("Patch Exchangeable Magnesium", patch.exchangeable_magnesium, "a.u.", "substrate"),
            composition("Patch Exchangeable Potassium", patch.exchangeable_potassium, "a.u.", "substrate"),
            composition("Patch Exchangeable Aluminum", patch.exchangeable_aluminum, "a.u.", "substrate"),
            composition("Seed Inventory Water", inventory.water, "a.u.", "inventory"),
            composition("Seed Inventory Glucose", inventory.glucose, "a.u.", "inventory"),
            composition("Seed Inventory Amino Acids", inventory.amino_acids, "a.u.", "inventory"),
            composition("Seed Inventory Nucleotides", inventory.nucleotides, "a.u.", "inventory"),
            composition("Seed Inventory Membrane Precursors", inventory.membrane_precursors, "a.u.", "inventory"),
            composition("Seed Inventory Surface Proton Load", inventory.surface_proton_load, "a.u.", "inventory"),
            composition("Seed Inventory Calcium Bicarbonate Complex", inventory.calcium_bicarbonate_complex, "a.u.", "inventory"),
            composition("Seed Inventory Sorbed Aluminum Hydroxide", inventory.sorbed_aluminum_hydroxide, "a.u.", "inventory"),
            composition("Seed Inventory Sorbed Ferric Hydroxide", inventory.sorbed_ferric_hydroxide, "a.u.", "inventory"),
        ],
        notes: vec![
            "Seed readiness is still summarized from explicit seed cellular state plus local substrate chemistry.".into(),
            "Inventory metrics come from the seed-owned local chemistry ledger at the current burial depth rather than a renderer-only cache.".into(),
        ],
    })
}

fn inspect_water(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    index: Option<usize>,
    x: Option<f32>,
    y: Option<f32>,
) -> Result<InspectData, String> {
    let water = if let Some(idx) = index {
        world
            .waters
            .get(idx)
            .ok_or_else(|| format!("Water index {idx} out of range"))?
    } else {
        let (cx, cy) = required_xy(world, x, y)?;
        world
            .waters
            .iter()
            .find(|water| water.x == cx && water.y == cy)
            .ok_or_else(|| format!("No water source at {cx},{cy}"))?
    };
    let patch = substrate_patch_metrics(
        world,
        water.x,
        water.y,
        water.z.min(world.config.depth.saturating_sub(1)),
    );
    let flat = water.y * world.config.width + water.x;
    Ok(InspectData {
        kind: "water".into(),
        title: "Water Body".into(),
        subtitle: format!("Hydrated basin in {}", preset.label()),
        preset: preset.cli_name().into(),
        position: Some(InspectPosition {
            x: water.x as f32 + 0.5,
            y: water.y as f32 + 0.5,
            z: Some(water.z as f32),
        }),
        summary: vec![
            metric("Volume", format!("{:.2}", water.volume), Some((water.volume / 240.0).clamp(0.0, 1.0))),
            metric("Open Water", format!("{:.3}", world.water_mask[flat]), Some(world.water_mask[flat].clamp(0.0, 1.0))),
            metric("Moisture", format!("{:.3}", world.moisture[flat]), Some(world.moisture[flat].clamp(0.0, 1.0))),
            metric("Alive", if water.alive { "Yes" } else { "No" }.into(), Some(if water.alive { 1.0 } else { 0.0 })),
        ],
        scene: vec![
            metric("Evaporation", format!("{:.6}", water.evaporation_rate), Some((water.evaporation_rate / 0.002).clamp(0.0, 1.0))),
            metric("Deep Moisture", format!("{:.3}", world.deep_moisture[flat]), Some(world.deep_moisture[flat].clamp(0.0, 1.0))),
            metric("Organic Matter", format!("{:.3}", world.organic_matter[flat]), Some(world.organic_matter[flat].clamp(0.0, 1.0))),
        ],
        cellular: vec![
            metric("Nearby Microbes", format!("{:.2}", world.microbial_biomass[flat]), Some((world.microbial_biomass[flat] / 50.0).clamp(0.0, 1.0))),
            metric("Nearby Symbionts", format!("{:.2}", world.symbiont_biomass[flat]), Some((world.symbiont_biomass[flat] / 50.0).clamp(0.0, 1.0))),
        ],
        molecular: vec![
            metric("Water Species", format!("{:.3}", patch.water), Some((patch.water / 1.5).clamp(0.0, 1.0))),
            metric("Glucose", format!("{:.4}", patch.glucose), Some((patch.glucose / 0.12).clamp(0.0, 1.0))),
            metric("Amino Acids", format!("{:.4}", patch.amino_acids), Some((patch.amino_acids / 0.028).clamp(0.0, 1.0))),
            metric("Nucleotides", format!("{:.4}", patch.nucleotides), Some((patch.nucleotides / 0.020).clamp(0.0, 1.0))),
            metric("Membrane Px", format!("{:.4}", patch.membrane_precursors), Some((patch.membrane_precursors / 0.018).clamp(0.0, 1.0))),
            metric("O2", format!("{:.4}", patch.oxygen), Some((patch.oxygen / 0.12).clamp(0.0, 1.0))),
            metric("CO2", format!("{:.4}", patch.co2), Some((patch.co2 / 0.12).clamp(0.0, 1.0))),
            metric("NH4+", format!("{:.4}", patch.ammonium), Some((patch.ammonium / 0.08).clamp(0.0, 1.0))),
            metric("NO3-", format!("{:.4}", patch.nitrate), Some((patch.nitrate / 0.08).clamp(0.0, 1.0))),
            metric("Si(aq)", format!("{:.4}", patch.dissolved_silicate), Some((patch.dissolved_silicate / 0.09).clamp(0.0, 1.0))),
            metric("HCO3-", format!("{:.4}", patch.bicarbonate), Some((patch.bicarbonate / 0.18).clamp(0.0, 1.0))),
            metric("Al(OH)3(s)", format!("{:.4}", patch.sorbed_aluminum_hydroxide), Some((patch.sorbed_aluminum_hydroxide / 0.12).clamp(0.0, 1.0))),
            metric("Fe(OH)3(s)", format!("{:.4}", patch.sorbed_ferric_hydroxide), Some((patch.sorbed_ferric_hydroxide / 0.12).clamp(0.0, 1.0))),
            metric("Ca exch", format!("{:.4}", patch.exchangeable_calcium), Some((patch.exchangeable_calcium / 0.18).clamp(0.0, 1.0))),
            metric("Al exch", format!("{:.4}", patch.exchangeable_aluminum), Some((patch.exchangeable_aluminum / 0.14).clamp(0.0, 1.0))),
            metric("Fe(aq)", format!("{:.4}", patch.aqueous_iron), Some((patch.aqueous_iron / 0.10).clamp(0.0, 1.0))),
        ],
        scene_grid: Some(neighborhood_grid(
            world,
            water.x,
            water.y,
            4,
            "Surface water occupancy neighborhood",
            "water_body",
            |gx, gy| world.water_mask[gy * world.config.width + gx],
        )),
        cellular_grid: Some(neighborhood_grid(
            world,
            water.x,
            water.y,
            4,
            "Nearby microbial activity neighborhood",
            "biology",
            |gx, gy| {
                let flat = gy * world.config.width + gx;
                clamp(
                    world.microbial_biomass[flat] / 50.0 * 0.70
                        + world.symbiont_biomass[flat] / 50.0 * 0.30,
                    0.0,
                    1.0,
                )
            },
        )),
        molecular_grid: Some(substrate_vertical_grid(
            world,
            water.x,
            water.y,
            "Depth x chemistry slice: H2O / glucose / amino / nucleotide / membrane / O2 / CO2 / NH4 / NO3 / ATP / Si / HCO3 / H+surf / Ca(HCO3)2 / AlOH3(s) / FeOH3(s) / Ca / Mg / K / Na / Al / Fe",
        )),
        composition: vec![
            composition("Water", patch.water, "a.u.", "fluid"),
            composition("Dissolved Glucose", patch.glucose, "a.u.", "solute"),
            composition("Dissolved Amino Acids", patch.amino_acids, "a.u.", "solute"),
            composition("Dissolved Nucleotides", patch.nucleotides, "a.u.", "solute"),
            composition("Membrane Precursors", patch.membrane_precursors, "a.u.", "colloid"),
            composition("Dissolved O2", patch.oxygen, "a.u.", "gas"),
            composition("Dissolved CO2", patch.co2, "a.u.", "gas"),
            composition("Dissolved NH4+", patch.ammonium, "a.u.", "nutrient"),
            composition("Dissolved NO3-", patch.nitrate, "a.u.", "nutrient"),
            composition("Dissolved Silicate", patch.dissolved_silicate, "a.u.", "solute"),
            composition("Bicarbonate", patch.bicarbonate, "a.u.", "solute"),
            composition("Surface Proton Load", patch.surface_proton_load, "a.u.", "interfacial"),
            composition("Calcium Bicarbonate Complex", patch.calcium_bicarbonate_complex, "a.u.", "solute"),
            composition("Sorbed Aluminum Hydroxide", patch.sorbed_aluminum_hydroxide, "a.u.", "mineral"),
            composition("Sorbed Ferric Hydroxide", patch.sorbed_ferric_hydroxide, "a.u.", "mineral"),
            composition("Exchangeable Calcium", patch.exchangeable_calcium, "a.u.", "mineral"),
            composition("Exchangeable Aluminum", patch.exchangeable_aluminum, "a.u.", "mineral"),
            composition("Aqueous Iron", patch.aqueous_iron, "a.u.", "mineral"),
        ],
        notes: vec!["Water remains sampled from the authoritative substrate lattice and surface-water mask.".into()],
    })
}

fn inspect_earthworm(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    index: usize,
) -> Result<InspectData, String> {
    let markers = world.earthworm_visual_markers();
    let (x, y, visual) = markers
        .get(index)
        .copied()
        .ok_or_else(|| format!("Earthworm index {index} out of range"))?;
    let flat = y * world.config.width + x;
    let patch = substrate_patch_metrics(world, x, y, 0);
    Ok(InspectData {
        kind: "earthworm".into(),
        title: "Earthworm (Lumbricus terrestris reference)".into(),
        subtitle: format!("Soil fauna in {}", preset.label()),
        preset: preset.cli_name().into(),
        position: Some(InspectPosition {
            x: x as f32 + 0.5,
            y: y as f32 + 0.5,
            z: Some(0.0),
        }),
        summary: vec![
            metric(
                "Activity",
                format!("{:.3}", visual.activity),
                Some(visual.activity),
            ),
            metric(
                "Segments",
                visual.segment_count.to_string(),
                Some((visual.segment_count as f32 / 180.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Length Scale",
                format!("{:.3}", visual.length_scale),
                Some((visual.length_scale / 2.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Population Density",
                format!("{:.2}", world.earthworm_population.population_density[flat]),
                Some((world.earthworm_population.population_density[flat] / 200.0).clamp(0.0, 1.0)),
            ),
        ],
        scene: vec![
            metric(
                "Biomass",
                format!("{:.3}", world.earthworm_population.biomass_per_voxel[flat]),
                Some((world.earthworm_population.biomass_per_voxel[flat] / 20.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Bioturbation",
                format!("{:.3}", world.earthworm_population.bioturbation_rate[flat]),
                Some((world.earthworm_population.bioturbation_rate[flat] / 5.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Moisture",
                format!("{:.3}", world.moisture[flat]),
                Some(world.moisture[flat].clamp(0.0, 1.0)),
            ),
            metric(
                "Organic Matter",
                format!("{:.3}", world.organic_matter[flat]),
                Some(world.organic_matter[flat].clamp(0.0, 1.0)),
            ),
        ],
        cellular: vec![
            metric(
                "Nearby Microbes",
                format!("{:.2}", world.microbial_biomass[flat]),
                Some((world.microbial_biomass[flat] / 50.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Nearby Symbionts",
                format!("{:.2}", world.symbiont_biomass[flat]),
                Some((world.symbiont_biomass[flat] / 50.0).clamp(0.0, 1.0)),
            ),
        ],
        molecular: vec![
            metric(
                "Patch Water",
                format!("{:.3}", patch.water),
                Some((patch.water / 1.5).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch Glucose",
                format!("{:.4}", patch.glucose),
                Some((patch.glucose / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch Amino",
                format!("{:.4}", patch.amino_acids),
                Some((patch.amino_acids / 0.028).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch Nucleotide",
                format!("{:.4}", patch.nucleotides),
                Some((patch.nucleotides / 0.020).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch Membrane Px",
                format!("{:.4}", patch.membrane_precursors),
                Some((patch.membrane_precursors / 0.018).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch O2",
                format!("{:.4}", patch.oxygen),
                Some((patch.oxygen / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch NH4+",
                format!("{:.4}", patch.ammonium),
                Some((patch.ammonium / 0.08).clamp(0.0, 1.0)),
            ),
        ],
        scene_grid: Some(earthworm_scene_grid(&visual)),
        cellular_grid: Some(earthworm_state_grid(world, flat, &visual)),
        molecular_grid: Some(substrate_vertical_grid(
            world,
            x,
            y,
            "Depth x chemistry slice below earthworm hotspot: H2O / glucose / amino / nucleotide / membrane / O2 / CO2 / NH4 / NO3 / ATP / Si / HCO3 / H+surf / Ca(HCO3)2 / AlOH3(s) / FeOH3(s) / Ca / Mg / K / Na / Al / Fe",
        )),
        composition: vec![
            composition("Organic Matter", world.organic_matter[flat], "a.u.", "soil"),
            composition("Water", patch.water, "a.u.", "soil"),
            composition("Amino Acids", patch.amino_acids, "a.u.", "soil"),
            composition("Membrane Precursors", patch.membrane_precursors, "a.u.", "soil"),
            composition("Oxygen", patch.oxygen, "a.u.", "soil"),
            composition("Glucose", patch.glucose, "a.u.", "soil"),
        ],
        notes: vec![format!(
            "Reference genome: {:.0} Mb, {} genes.",
            earthworm_genome_stats::GENOME_MB,
            earthworm_genome_stats::PROTEIN_GENES
        )],
    })
}

fn inspect_nematode(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    index: usize,
) -> Result<InspectData, String> {
    let markers = world.nematode_visual_markers();
    let (x, y, visual) = markers
        .get(index)
        .copied()
        .ok_or_else(|| format!("Nematode index {index} out of range"))?;
    let flat = y * world.config.width + x;
    let patch = substrate_patch_metrics(world, x, y, 0);
    let density = world
        .nematode_guilds
        .iter()
        .map(|guild| guild.population_density[flat])
        .sum::<f32>();
    let biomass = world
        .nematode_guilds
        .iter()
        .map(|guild| guild.biomass_per_voxel[flat])
        .sum::<f32>();
    let dominant_kind = world
        .nematode_guilds
        .iter()
        .max_by(|a, b| {
            a.population_density[flat]
                .partial_cmp(&b.population_density[flat])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|guild| match guild.kind {
            NematodeKind::BacterialFeeder => "Bacterial feeder",
            NematodeKind::FungalFeeder => "Fungal feeder",
            NematodeKind::Omnivore => "Omnivore",
        })
        .unwrap_or("Mixed");

    Ok(InspectData {
        kind: "nematode".into(),
        title: "Nematode (C. elegans reference)".into(),
        subtitle: format!("{dominant_kind} guild in {}", preset.label()),
        preset: preset.cli_name().into(),
        position: Some(InspectPosition {
            x: x as f32 + 0.5,
            y: y as f32 + 0.5,
            z: Some(0.0),
        }),
        summary: vec![
            metric(
                "Activity",
                format!("{:.3}", visual.activity),
                Some(visual.activity),
            ),
            metric(
                "Length Scale",
                format!("{:.3}", visual.length_scale),
                Some((visual.length_scale / 2.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Density",
                format!("{:.2}", density),
                Some((density / 40.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Stylet Scale",
                format!("{:.3}", visual.stylet_length_scale),
                Some((visual.stylet_length_scale / 2.0).clamp(0.0, 1.0)),
            ),
        ],
        scene: vec![
            metric(
                "Microbial Biomass",
                format!("{:.2}", world.microbial_biomass[flat]),
                Some((world.microbial_biomass[flat] / 50.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Moisture",
                format!("{:.3}", world.moisture[flat]),
                Some(world.moisture[flat].clamp(0.0, 1.0)),
            ),
            metric(
                "Canopy Cover",
                format!("{:.3}", world.canopy_cover[flat]),
                Some(world.canopy_cover[flat].clamp(0.0, 1.0)),
            ),
        ],
        cellular: vec![metric(
            "Guild Count",
            world.nematode_guilds.len().to_string(),
            Some((world.nematode_guilds.len() as f32 / 3.0).clamp(0.0, 1.0)),
        )],
        molecular: vec![
            metric(
                "Patch Water",
                format!("{:.3}", patch.water),
                Some((patch.water / 1.5).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch Glucose",
                format!("{:.4}", patch.glucose),
                Some((patch.glucose / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch Amino",
                format!("{:.4}", patch.amino_acids),
                Some((patch.amino_acids / 0.028).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch Nucleotide",
                format!("{:.4}", patch.nucleotides),
                Some((patch.nucleotides / 0.020).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch Membrane Px",
                format!("{:.4}", patch.membrane_precursors),
                Some((patch.membrane_precursors / 0.018).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch NH4+",
                format!("{:.4}", patch.ammonium),
                Some((patch.ammonium / 0.08).clamp(0.0, 1.0)),
            ),
            metric(
                "Patch NO3-",
                format!("{:.4}", patch.nitrate),
                Some((patch.nitrate / 0.08).clamp(0.0, 1.0)),
            ),
        ],
        scene_grid: Some(nematode_scene_grid(&visual)),
        cellular_grid: Some(nematode_state_grid(
            world,
            flat,
            density,
            biomass,
            &visual,
        )),
        molecular_grid: Some(substrate_vertical_grid(
            world,
            x,
            y,
            "Depth x chemistry slice below nematode hotspot: H2O / glucose / amino / nucleotide / membrane / O2 / CO2 / NH4 / NO3 / ATP / Si / HCO3 / H+surf / Ca(HCO3)2 / AlOH3(s) / FeOH3(s) / Ca / Mg / K / Na / Al / Fe",
        )),
        composition: vec![
            composition("Water", patch.water, "a.u.", "soil"),
            composition("Glucose", patch.glucose, "a.u.", "soil"),
            composition("Amino Acids", patch.amino_acids, "a.u.", "soil"),
            composition("Nucleotides", patch.nucleotides, "a.u.", "soil"),
            composition("Ammonium", patch.ammonium, "a.u.", "soil"),
            composition("Nitrate", patch.nitrate, "a.u.", "soil"),
        ],
        notes: vec![format!(
            "Reference genome: {:.0} Mb, {} genes, {} neurons.",
            nematode_genome_stats::GENOME_MB,
            nematode_genome_stats::PROTEIN_GENES,
            nematode_genome_stats::NEURON_COUNT
        )],
    })
}

fn inspect_soil(
    world: &TerrariumWorld,
    preset: TerrariumDemoPreset,
    x: usize,
    y: usize,
) -> Result<InspectData, String> {
    let flat = y * world.config.width + x;
    let probe_z = (world.config.depth / 3).min(world.config.depth.saturating_sub(1));
    let patch = substrate_patch_metrics(world, x, y, probe_z);
    let patch_proton =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Proton, x, y, probe_z, 1);
    let base_saturation = soil_base_saturation(
        patch.exchangeable_calcium,
        patch.exchangeable_magnesium,
        patch.exchangeable_potassium,
        patch.exchangeable_sodium,
        patch.exchangeable_aluminum,
        patch_proton,
        patch.surface_proton_load,
    );
    let aluminum_toxicity = soil_aluminum_toxicity(
        patch.exchangeable_aluminum,
        patch_proton,
        patch.surface_proton_load,
        base_saturation,
    );
    let weathering_support = soil_weathering_support(
        patch.dissolved_silicate,
        patch.bicarbonate,
        patch.calcium_bicarbonate_complex,
        patch.exchangeable_calcium,
        patch.exchangeable_magnesium,
        patch.exchangeable_potassium,
        patch.aqueous_iron,
        patch.carbonate_mineral,
        patch.clay_mineral,
    );
    let local_air = local_air(world, x as f32 + 0.5, y as f32 + 0.5);
    let owner = describe_owner(world.ownership[flat].owner);
    let nematode_density = world
        .nematode_guilds
        .iter()
        .map(|guild| guild.population_density[flat])
        .sum::<f32>();
    let shore_contact = shoreline_water_signal(
        world.config.width,
        world.config.height,
        &world.water_mask,
        flat,
    );
    let absorbency =
        soil_texture_absorbency(world.soil_structure[flat], world.organic_matter[flat]);
    let retention = soil_texture_retention(world.soil_structure[flat], world.organic_matter[flat]);
    let explicit_microbe = if matches!(
        world.ownership[flat].owner,
        SoilOwnershipClass::ExplicitMicrobeCohort { .. }
    ) {
        world.explicit_microbe_at(x, y)
    } else {
        None
    };
    let packet_population = if matches!(
        world.ownership[flat].owner,
        SoilOwnershipClass::GenotypePacketRegion { .. }
            | SoilOwnershipClass::ExplicitMicrobeCohort { .. }
    ) {
        world.packet_population_at(x, y)
    } else {
        None
    };
    let packet_totals = packet_population
        .map(|pop| pop.ecology_totals())
        .unwrap_or([0.0; 3]);
    let packet_total_cells = packet_population.map(|pop| pop.total_cells).unwrap_or(0.0);
    let packet_diversity = packet_population
        .map(|pop| pop.ecology_diversity())
        .unwrap_or(0.0);
    let packet_mean_activity = packet_population
        .map(|pop| pop.mean_activity())
        .unwrap_or(0.0);
    let packet_mean_dormancy = packet_population
        .map(|pop| pop.mean_dormancy())
        .unwrap_or(0.0);
    let dominant_packet = packet_population.and_then(|pop| pop.dominant_packet());
    let dominant_packet_ecology = packet_population.and_then(|pop| pop.dominant_ecology());
    let dominant_microbiome = if world.denitrification_potential[flat]
        > world.nitrification_potential[flat] * 1.05
        && world.denitrification_potential[flat] > 0.00002
    {
        "anoxic denitrifier film"
    } else if world.nitrification_potential[flat] > 0.00002 {
        "oxic nitrifier crust"
    } else if world.symbiont_biomass[flat] > world.microbial_biomass[flat] * 0.35 {
        "root-symbiont patch"
    } else if world.microbial_biomass[flat] > 0.08 {
        "heterotroph mat"
    } else {
        "mineral or humus background"
    };
    let mut cellular = vec![
        metric(
            "Microbial Biomass",
            format!("{:.2}", world.microbial_biomass[flat]),
            Some((world.microbial_biomass[flat] / 50.0).clamp(0.0, 1.0)),
        ),
        metric(
            "Symbiont Biomass",
            format!("{:.2}", world.symbiont_biomass[flat]),
            Some((world.symbiont_biomass[flat] / 50.0).clamp(0.0, 1.0)),
        ),
        metric(
            "Earthworm Density",
            format!("{:.2}", world.earthworm_population.population_density[flat]),
            Some((world.earthworm_population.population_density[flat] / 200.0).clamp(0.0, 1.0)),
        ),
        metric(
            "Nematode Density",
            format!("{:.2}", nematode_density),
            Some((nematode_density / 40.0).clamp(0.0, 1.0)),
        ),
        metric(
            "Nitrification",
            format!("{:.5}", world.nitrification_potential[flat]),
            Some((world.nitrification_potential[flat] * 9000.0).clamp(0.0, 1.0)),
        ),
        metric(
            "Denitrification",
            format!("{:.5}", world.denitrification_potential[flat]),
            Some((world.denitrification_potential[flat] * 9000.0).clamp(0.0, 1.0)),
        ),
    ];
    if let Some(pop) = packet_population {
        cellular.splice(
            0..2,
            [
                metric(
                    "Packet Count",
                    format!("{}", pop.packets.len()),
                    Some((pop.packets.len() as f32 / 8.0).clamp(0.0, 1.0)),
                ),
                metric(
                    "Represented Cells",
                    format!("{:.0}", packet_total_cells),
                    Some((packet_total_cells / 1.2e6).sqrt().clamp(0.0, 1.0)),
                ),
                metric(
                    "Packet Activity",
                    format!("{:.3}", packet_mean_activity),
                    Some(packet_mean_activity.clamp(0.0, 1.0)),
                ),
                metric(
                    "Packet Dormancy",
                    format!("{:.3}", packet_mean_dormancy),
                    Some(packet_mean_dormancy.clamp(0.0, 1.0)),
                ),
                metric(
                    "Ecology Diversity",
                    format!("{:.3}", packet_diversity),
                    Some(packet_diversity),
                ),
                metric(
                    "Nitrifier Share",
                    format!(
                        "{:.2}%",
                        packet_totals[1] * 100.0 / packet_total_cells.max(1.0)
                    ),
                    Some((packet_totals[1] / packet_total_cells.max(1.0)).clamp(0.0, 1.0)),
                ),
            ],
        );
    }
    if let Some(cohort) = explicit_microbe {
        cellular.splice(
            0..0,
            [
                metric(
                    "Cohort Cells",
                    format!("{:.0}", cohort.represented_cells),
                    Some(
                        (cohort.represented_cells
                            / EXPLICIT_MICROBE_MAX_REPRESENTED_CELLS.max(1.0))
                        .sqrt()
                        .clamp(0.0, 1.0),
                    ),
                ),
                metric(
                    "Cohort Packets",
                    format!("{:.1}", cohort.represented_packets),
                    Some((cohort.represented_packets / 8.0).clamp(0.0, 1.0)),
                ),
                metric(
                    "Cohort Energy",
                    format!("{:.3}", cohort.smoothed_energy),
                    Some((cohort.smoothed_energy / 1.2).clamp(0.0, 1.0)),
                ),
                metric(
                    "Cohort Stress",
                    format!("{:.3}", cohort.smoothed_stress),
                    Some((cohort.smoothed_stress / 1.2).clamp(0.0, 1.0)),
                ),
                metric(
                    "Whole-Cell ATP",
                    format!("{:.3}", cohort.last_snapshot.atp_mm),
                    Some((cohort.last_snapshot.atp_mm / 2.0).clamp(0.0, 1.0)),
                ),
                metric(
                    "Division Progress",
                    format!("{:.3}", cohort.last_snapshot.division_progress),
                    Some(cohort.last_snapshot.division_progress.clamp(0.0, 1.0)),
                ),
            ],
        );
    }
    let mut notes = vec![
        format!(
            "Time state: {} / {}",
            world.time_label(),
            world.circadian_label()
        ),
        format!("Lunar phase: {}", world.moon_phase_name()),
    ];
    if let Some(cohort) = explicit_microbe {
        let ecology = match cohort.guild {
            1 => crate::terrarium::packet::GenotypePacketEcology::Nitrifier,
            2 => crate::terrarium::packet::GenotypePacketEcology::Denitrifier,
            _ => crate::terrarium::packet::GenotypePacketEcology::Decomposer,
        };
        notes.push(format!(
            "Promoted whole-cell shoreline cohort: {} lineage #{}, bank {}, {:.0} represented cells.",
            ecology.label(),
            cohort.identity.record.lineage_id,
            cohort.identity.bank_idx,
            cohort.represented_cells
        ));
        notes.push(format!(
            "Whole-cell state: ATP {:.3} mM / glucose {:.3} mM / oxygen {:.3} mM / division {:.3}.",
            cohort.last_snapshot.atp_mm,
            cohort.last_snapshot.glucose_mm,
            cohort.last_snapshot.oxygen_mm,
            cohort.last_snapshot.division_progress,
        ));
    } else if let Some(pop) = packet_population {
        let dominant_ecology = dominant_packet_ecology
            .map(|ecology| ecology.label())
            .unwrap_or("mixed");
        let dominant_genotype = dominant_packet
            .map(|packet| packet.genotype_id)
            .unwrap_or(0);
        notes.push(format!(
            "Explicit shoreline packet ecology: {} community, {} packets, dominant genotype #{}.",
            dominant_ecology,
            pop.packets.len(),
            dominant_genotype
        ));
        notes.push(format!(
            "Guild shares: decomposer {:.0}% / nitrifier {:.0}% / denitrifier {:.0}%.",
            packet_totals[0] * 100.0 / packet_total_cells.max(1.0),
            packet_totals[1] * 100.0 / packet_total_cells.max(1.0),
            packet_totals[2] * 100.0 / packet_total_cells.max(1.0),
        ));
    } else {
        notes.push(format!(
            "Dominant background microbiome: {dominant_microbiome}."
        ));
    }

    Ok(InspectData {
        kind: "soil".into(),
        title: format!("Soil Cell ({x}, {y})"),
        subtitle: format!("{} / {}", preset.label(), owner),
        preset: preset.cli_name().into(),
        position: Some(InspectPosition {
            x: x as f32 + 0.5,
            y: y as f32 + 0.5,
            z: Some(probe_z as f32),
        }),
        summary: vec![
            metric(
                "Surface Moisture",
                format!("{:.3}", world.moisture[flat]),
                Some(world.moisture[flat].clamp(0.0, 1.0)),
            ),
            metric(
                "Deep Moisture",
                format!("{:.3}", world.deep_moisture[flat]),
                Some(world.deep_moisture[flat].clamp(0.0, 1.0)),
            ),
            metric(
                "Open Water",
                format!("{:.3}", world.water_mask[flat]),
                Some(world.water_mask[flat].clamp(0.0, 1.0)),
            ),
            metric(
                "Ownership Strength",
                format!("{:.3}", world.ownership[flat].strength),
                Some(world.ownership[flat].strength),
            ),
            metric(
                "Shore Contact",
                format!("{:.3}", shore_contact),
                Some(shore_contact),
            ),
        ],
        scene: vec![
            metric(
                "Organic Matter",
                format!("{:.3}", world.organic_matter[flat]),
                Some(world.organic_matter[flat].clamp(0.0, 1.0)),
            ),
            metric(
                "Soil Structure",
                format!("{:.3}", world.soil_structure[flat]),
                Some(world.soil_structure[flat].clamp(0.0, 1.0)),
            ),
            metric(
                "Canopy Cover",
                format!("{:.3}", world.canopy_cover[flat]),
                Some(world.canopy_cover[flat].clamp(0.0, 1.0)),
            ),
            metric(
                "Root Density",
                format!("{:.3}", world.root_density[flat]),
                Some(world.root_density[flat].clamp(0.0, 1.0)),
            ),
            metric("Absorbency", format!("{:.3}", absorbency), Some(absorbency)),
            metric("Retention", format!("{:.3}", retention), Some(retention)),
            metric(
                "Temperature",
                format!("{:.1} C", local_air.temperature_c),
                Some(((local_air.temperature_c - 5.0) / 30.0).clamp(0.0, 1.0)),
            ),
            metric(
                "Humidity",
                format!("{:.2}", local_air.humidity),
                Some(local_air.humidity),
            ),
            metric(
                "Wind Speed",
                format!("{:.2}", local_air.wind_speed),
                Some((local_air.wind_speed / 2.0).clamp(0.0, 1.0)),
            ),
        ],
        cellular,
        molecular: vec![
            metric(
                "Water",
                format!("{:.3}", patch.water),
                Some((patch.water / 1.5).clamp(0.0, 1.0)),
            ),
            metric(
                "Glucose",
                format!("{:.4}", patch.glucose),
                Some((patch.glucose / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "Amino Acids",
                format!("{:.4}", patch.amino_acids),
                Some((patch.amino_acids / 0.028).clamp(0.0, 1.0)),
            ),
            metric(
                "Nucleotides",
                format!("{:.4}", patch.nucleotides),
                Some((patch.nucleotides / 0.020).clamp(0.0, 1.0)),
            ),
            metric(
                "Membrane Px",
                format!("{:.4}", patch.membrane_precursors),
                Some((patch.membrane_precursors / 0.018).clamp(0.0, 1.0)),
            ),
            metric(
                "O2",
                format!("{:.4}", patch.oxygen),
                Some((patch.oxygen / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "CO2",
                format!("{:.4}", patch.co2),
                Some((patch.co2 / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "NH4+",
                format!("{:.4}", patch.ammonium),
                Some((patch.ammonium / 0.08).clamp(0.0, 1.0)),
            ),
            metric(
                "NO3-",
                format!("{:.4}", patch.nitrate),
                Some((patch.nitrate / 0.08).clamp(0.0, 1.0)),
            ),
            metric(
                "ATP Flux",
                format!("{:.4}", patch.atp_flux),
                Some((patch.atp_flux / 0.08).clamp(0.0, 1.0)),
            ),
            metric(
                "Si(aq)",
                format!("{:.4}", patch.dissolved_silicate),
                Some((patch.dissolved_silicate / 0.09).clamp(0.0, 1.0)),
            ),
            metric(
                "HCO3-",
                format!("{:.4}", patch.bicarbonate),
                Some((patch.bicarbonate / 0.18).clamp(0.0, 1.0)),
            ),
            metric(
                "H+ surf",
                format!("{:.4}", patch.surface_proton_load),
                Some((patch.surface_proton_load / 0.18).clamp(0.0, 1.0)),
            ),
            metric(
                "Ca(HCO3)2",
                format!("{:.4}", patch.calcium_bicarbonate_complex),
                Some((patch.calcium_bicarbonate_complex / 0.16).clamp(0.0, 1.0)),
            ),
            metric(
                "Al(OH)3(s)",
                format!("{:.4}", patch.sorbed_aluminum_hydroxide),
                Some((patch.sorbed_aluminum_hydroxide / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "Fe(OH)3(s)",
                format!("{:.4}", patch.sorbed_ferric_hydroxide),
                Some((patch.sorbed_ferric_hydroxide / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "Ca exch",
                format!("{:.4}", patch.exchangeable_calcium),
                Some((patch.exchangeable_calcium / 0.18).clamp(0.0, 1.0)),
            ),
            metric(
                "Mg exch",
                format!("{:.4}", patch.exchangeable_magnesium),
                Some((patch.exchangeable_magnesium / 0.14).clamp(0.0, 1.0)),
            ),
            metric(
                "K exch",
                format!("{:.4}", patch.exchangeable_potassium),
                Some((patch.exchangeable_potassium / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "Na exch",
                format!("{:.4}", patch.exchangeable_sodium),
                Some((patch.exchangeable_sodium / 0.12).clamp(0.0, 1.0)),
            ),
            metric(
                "Al exch",
                format!("{:.4}", patch.exchangeable_aluminum),
                Some((patch.exchangeable_aluminum / 0.14).clamp(0.0, 1.0)),
            ),
            metric(
                "Fe(aq)",
                format!("{:.4}", patch.aqueous_iron),
                Some((patch.aqueous_iron / 0.10).clamp(0.0, 1.0)),
            ),
            metric(
                "Base Saturation",
                format!("{:.3}", base_saturation),
                Some(base_saturation),
            ),
            metric(
                "Al Toxicity",
                format!("{:.3}", aluminum_toxicity),
                Some((aluminum_toxicity / 1.4).clamp(0.0, 1.0)),
            ),
            metric(
                "Weathering Support",
                format!("{:.3}", weathering_support),
                Some((weathering_support / 1.6).clamp(0.0, 1.0)),
            ),
        ],
        scene_grid: Some(neighborhood_grid(
            world,
            x,
            y,
            4,
            "Surface moisture neighborhood",
            "soil_moisture",
            |gx, gy| {
                let flat = gy * world.config.width + gx;
                clamp(
                    world.moisture[flat] * 0.62 + world.water_mask[flat] * 0.38,
                    0.0,
                    1.0,
                )
            },
        )),
        cellular_grid: Some(neighborhood_grid(
            world,
            x,
            y,
            4,
            "Microbiome and fauna neighborhood",
            "biology",
            |gx, gy| {
                let flat = gy * world.config.width + gx;
                let nematodes = world
                    .nematode_guilds
                    .iter()
                    .map(|guild| guild.population_density[flat])
                    .sum::<f32>();
                clamp(
                    world.microbial_biomass[flat] / 50.0 * 0.52
                        + world.symbiont_biomass[flat] / 50.0 * 0.18
                        + world.earthworm_population.population_density[flat] / 200.0 * 0.12
                        + nematodes / 40.0 * 0.18,
                    0.0,
                    1.0,
                )
            },
        )),
        molecular_grid: Some(substrate_vertical_grid(
            world,
            x,
            y,
            "Depth x chemistry slice: H2O / glucose / amino / nucleotide / membrane / O2 / CO2 / NH4 / NO3 / ATP / Si / HCO3 / H+surf / Ca(HCO3)2 / AlOH3(s) / FeOH3(s) / Ca / Mg / K / Na / Al / Fe",
        )),
        composition: vec![
            composition("Water", patch.water, "a.u.", "substrate"),
            composition("Glucose", patch.glucose, "a.u.", "substrate"),
            composition("Amino Acids", patch.amino_acids, "a.u.", "substrate"),
            composition("Nucleotides", patch.nucleotides, "a.u.", "substrate"),
            composition("Membrane Precursors", patch.membrane_precursors, "a.u.", "substrate"),
            composition("Oxygen", patch.oxygen, "a.u.", "substrate"),
            composition("CO2", patch.co2, "a.u.", "substrate"),
            composition("Ammonium", patch.ammonium, "a.u.", "substrate"),
            composition("Nitrate", patch.nitrate, "a.u.", "substrate"),
            composition("Dissolved Silicate", patch.dissolved_silicate, "a.u.", "substrate"),
            composition("Bicarbonate", patch.bicarbonate, "a.u.", "substrate"),
            composition("Surface Proton Load", patch.surface_proton_load, "a.u.", "substrate"),
            composition("Calcium Bicarbonate Complex", patch.calcium_bicarbonate_complex, "a.u.", "substrate"),
            composition(
                "Sorbed Aluminum Hydroxide",
                patch.sorbed_aluminum_hydroxide,
                "a.u.",
                "substrate",
            ),
            composition(
                "Sorbed Ferric Hydroxide",
                patch.sorbed_ferric_hydroxide,
                "a.u.",
                "substrate",
            ),
            composition(
                "Exchangeable Calcium",
                patch.exchangeable_calcium,
                "a.u.",
                "substrate",
            ),
            composition(
                "Exchangeable Magnesium",
                patch.exchangeable_magnesium,
                "a.u.",
                "substrate",
            ),
            composition(
                "Exchangeable Potassium",
                patch.exchangeable_potassium,
                "a.u.",
                "substrate",
            ),
            composition(
                "Exchangeable Sodium",
                patch.exchangeable_sodium,
                "a.u.",
                "substrate",
            ),
            composition(
                "Exchangeable Aluminum",
                patch.exchangeable_aluminum,
                "a.u.",
                "substrate",
            ),
            composition("Aqueous Iron", patch.aqueous_iron, "a.u.", "substrate"),
            composition("Carbon", patch.carbon, "a.u.", "element"),
            composition("Nitrogen", patch.nitrogen, "a.u.", "element"),
            composition("Phosphorus", patch.phosphorus, "a.u.", "element"),
        ],
        notes,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct SubstratePatchMetrics {
    water: f32,
    glucose: f32,
    amino_acids: f32,
    nucleotides: f32,
    membrane_precursors: f32,
    oxygen: f32,
    co2: f32,
    ammonium: f32,
    nitrate: f32,
    atp_flux: f32,
    dissolved_silicate: f32,
    bicarbonate: f32,
    sorbed_aluminum_hydroxide: f32,
    sorbed_ferric_hydroxide: f32,
    exchangeable_calcium: f32,
    exchangeable_magnesium: f32,
    exchangeable_potassium: f32,
    exchangeable_sodium: f32,
    exchangeable_aluminum: f32,
    aqueous_iron: f32,
    clay_mineral: f32,
    carbonate_mineral: f32,
    surface_proton_load: f32,
    calcium_bicarbonate_complex: f32,
    carbon: f32,
    nitrogen: f32,
    phosphorus: f32,
}

fn substrate_patch_metrics(
    world: &TerrariumWorld,
    x: usize,
    y: usize,
    z: usize,
) -> SubstratePatchMetrics {
    SubstratePatchMetrics {
        water: world
            .substrate
            .patch_mean_species(TerrariumSpecies::Water, x, y, z, 1),
        glucose: world
            .substrate
            .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, 1),
        amino_acids: world.substrate.patch_mean_species(
            TerrariumSpecies::AminoAcidPool,
            x,
            y,
            z,
            1,
        ),
        nucleotides: world.substrate.patch_mean_species(
            TerrariumSpecies::NucleotidePool,
            x,
            y,
            z,
            1,
        ),
        membrane_precursors: world.substrate.patch_mean_species(
            TerrariumSpecies::MembranePrecursorPool,
            x,
            y,
            z,
            1,
        ),
        oxygen: world
            .substrate
            .patch_mean_species(TerrariumSpecies::OxygenGas, x, y, z, 1),
        co2: world
            .substrate
            .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, 1),
        ammonium: world
            .substrate
            .patch_mean_species(TerrariumSpecies::Ammonium, x, y, z, 1),
        nitrate: world
            .substrate
            .patch_mean_species(TerrariumSpecies::Nitrate, x, y, z, 1),
        atp_flux: world
            .substrate
            .patch_mean_species(TerrariumSpecies::AtpFlux, x, y, z, 1),
        dissolved_silicate: world.substrate.patch_mean_species(
            TerrariumSpecies::DissolvedSilicate,
            x,
            y,
            z,
            1,
        ),
        bicarbonate: world.substrate.patch_mean_species(
            TerrariumSpecies::BicarbonatePool,
            x,
            y,
            z,
            1,
        ),
        surface_proton_load: world.substrate.patch_mean_species(
            TerrariumSpecies::SurfaceProtonLoad,
            x,
            y,
            z,
            1,
        ),
        calcium_bicarbonate_complex: world.substrate.patch_mean_species(
            TerrariumSpecies::CalciumBicarbonateComplex,
            x,
            y,
            z,
            1,
        ),
        sorbed_aluminum_hydroxide: world.substrate.patch_mean_species(
            TerrariumSpecies::SorbedAluminumHydroxide,
            x,
            y,
            z,
            1,
        ),
        sorbed_ferric_hydroxide: world.substrate.patch_mean_species(
            TerrariumSpecies::SorbedFerricHydroxide,
            x,
            y,
            z,
            1,
        ),
        exchangeable_calcium: world.substrate.patch_mean_species(
            TerrariumSpecies::ExchangeableCalcium,
            x,
            y,
            z,
            1,
        ),
        exchangeable_magnesium: world.substrate.patch_mean_species(
            TerrariumSpecies::ExchangeableMagnesium,
            x,
            y,
            z,
            1,
        ),
        exchangeable_potassium: world.substrate.patch_mean_species(
            TerrariumSpecies::ExchangeablePotassium,
            x,
            y,
            z,
            1,
        ),
        exchangeable_sodium: world.substrate.patch_mean_species(
            TerrariumSpecies::ExchangeableSodium,
            x,
            y,
            z,
            1,
        ),
        exchangeable_aluminum: world.substrate.patch_mean_species(
            TerrariumSpecies::ExchangeableAluminum,
            x,
            y,
            z,
            1,
        ),
        aqueous_iron: world.substrate.patch_mean_species(
            TerrariumSpecies::AqueousIronPool,
            x,
            y,
            z,
            1,
        ),
        clay_mineral: world
            .substrate
            .patch_mean_species(TerrariumSpecies::ClayMineral, x, y, z, 1),
        carbonate_mineral: world.substrate.patch_mean_species(
            TerrariumSpecies::CarbonateMineral,
            x,
            y,
            z,
            1,
        ),
        carbon: world
            .substrate
            .patch_mean_species(TerrariumSpecies::Carbon, x, y, z, 1),
        nitrogen: world
            .substrate
            .patch_mean_species(TerrariumSpecies::Nitrogen, x, y, z, 1),
        phosphorus: world
            .substrate
            .patch_mean_species(TerrariumSpecies::Phosphorus, x, y, z, 1),
    }
}

fn inventory_patch_metrics(
    inventory: &crate::terrarium::RegionalMaterialInventory,
) -> SubstratePatchMetrics {
    SubstratePatchMetrics {
        water: inventory_component_amount(inventory, TerrariumSpecies::Water),
        glucose: inventory_component_amount(inventory, TerrariumSpecies::Glucose),
        amino_acids: inventory_component_amount(inventory, TerrariumSpecies::AminoAcidPool),
        nucleotides: inventory_component_amount(inventory, TerrariumSpecies::NucleotidePool),
        membrane_precursors: inventory_component_amount(
            inventory,
            TerrariumSpecies::MembranePrecursorPool,
        ),
        oxygen: inventory_component_amount(inventory, TerrariumSpecies::OxygenGas),
        co2: inventory_component_amount(inventory, TerrariumSpecies::CarbonDioxide),
        ammonium: inventory_component_amount(inventory, TerrariumSpecies::Ammonium),
        nitrate: inventory_component_amount(inventory, TerrariumSpecies::Nitrate),
        atp_flux: inventory_component_amount(inventory, TerrariumSpecies::AtpFlux),
        dissolved_silicate: inventory_component_amount(
            inventory,
            TerrariumSpecies::DissolvedSilicate,
        ),
        bicarbonate: inventory_component_amount(inventory, TerrariumSpecies::BicarbonatePool),
        surface_proton_load: inventory_component_amount(
            inventory,
            TerrariumSpecies::SurfaceProtonLoad,
        ),
        calcium_bicarbonate_complex: inventory_component_amount(
            inventory,
            TerrariumSpecies::CalciumBicarbonateComplex,
        ),
        sorbed_aluminum_hydroxide: inventory_component_amount(
            inventory,
            TerrariumSpecies::SorbedAluminumHydroxide,
        ),
        sorbed_ferric_hydroxide: inventory_component_amount(
            inventory,
            TerrariumSpecies::SorbedFerricHydroxide,
        ),
        exchangeable_calcium: inventory_component_amount(
            inventory,
            TerrariumSpecies::ExchangeableCalcium,
        ),
        exchangeable_magnesium: inventory_component_amount(
            inventory,
            TerrariumSpecies::ExchangeableMagnesium,
        ),
        exchangeable_potassium: inventory_component_amount(
            inventory,
            TerrariumSpecies::ExchangeablePotassium,
        ),
        exchangeable_sodium: inventory_component_amount(
            inventory,
            TerrariumSpecies::ExchangeableSodium,
        ),
        exchangeable_aluminum: inventory_component_amount(
            inventory,
            TerrariumSpecies::ExchangeableAluminum,
        ),
        aqueous_iron: inventory_component_amount(inventory, TerrariumSpecies::AqueousIronPool),
        clay_mineral: inventory_component_amount(inventory, TerrariumSpecies::ClayMineral),
        carbonate_mineral: inventory_component_amount(
            inventory,
            TerrariumSpecies::CarbonateMineral,
        ),
        carbon: 0.0,
        nitrogen: 0.0,
        phosphorus: 0.0,
    }
}

fn local_air(
    world: &TerrariumWorld,
    x: f32,
    y: f32,
) -> crate::terrarium::visual_projection::TerrariumVisualAirSample {
    let atmosphere = world.atmosphere_frame();
    sample_visual_air(&atmosphere, world.config.width, world.config.height, x, y)
}

fn neighborhood_grid<F>(
    world: &TerrariumWorld,
    center_x: usize,
    center_y: usize,
    radius: usize,
    label: impl Into<String>,
    palette: impl Into<String>,
    sample: F,
) -> InspectGrid
where
    F: FnMut(usize, usize) -> f32,
{
    let values = neighborhood_values(
        world.config.width,
        world.config.height,
        center_x,
        center_y,
        radius,
        sample,
    );
    let width = radius * 2 + 1;
    let height = radius * 2 + 1;
    InspectGrid {
        label: label.into(),
        width,
        height,
        palette: palette.into(),
        values,
    }
}

fn neighborhood_values<F>(
    world_width: usize,
    world_height: usize,
    center_x: usize,
    center_y: usize,
    radius: usize,
    mut sample: F,
) -> Vec<f32>
where
    F: FnMut(usize, usize) -> f32,
{
    let width = radius * 2 + 1;
    let height = radius * 2 + 1;
    let mut values = Vec::with_capacity(width * height);
    let center_x = center_x as isize;
    let center_y = center_y as isize;
    let radius = radius as isize;

    for oy in -radius..=radius {
        for ox in -radius..=radius {
            let gx = center_x + ox;
            let gy = center_y + oy;
            let value =
                if gx < 0 || gy < 0 || gx >= world_width as isize || gy >= world_height as isize {
                    0.0
                } else {
                    clamp(sample(gx as usize, gy as usize), 0.0, 1.0)
                };
            values.push(value);
        }
    }

    values
}

fn inspect_grid(
    label: impl Into<String>,
    palette: impl Into<String>,
    width: usize,
    height: usize,
    mut values: Vec<f32>,
) -> InspectGrid {
    let target_len = width.saturating_mul(height);
    values.truncate(target_len);
    while values.len() < target_len {
        values.push(0.0);
    }
    for value in &mut values {
        *value = if value.is_finite() {
            clamp(*value, 0.0, 1.0)
        } else {
            0.0
        };
    }
    InspectGrid {
        label: label.into(),
        width,
        height,
        palette: palette.into(),
        values,
    }
}

fn raster_grid<F>(
    width: usize,
    height: usize,
    label: impl Into<String>,
    palette: impl Into<String>,
    mut sample: F,
) -> InspectGrid
where
    F: FnMut(f32, f32) -> f32,
{
    let mut values = Vec::with_capacity(width.saturating_mul(height));
    for gy in 0..height {
        let ny = if height <= 1 {
            0.0
        } else {
            ((gy as f32 + 0.5) / height as f32) * 2.0 - 1.0
        };
        for gx in 0..width {
            let nx = if width <= 1 {
                0.0
            } else {
                ((gx as f32 + 0.5) / width as f32) * 2.0 - 1.0
            };
            values.push(sample(nx, ny));
        }
    }
    inspect_grid(label, palette, width, height, values)
}

fn ellipse_intensity(nx: f32, ny: f32, cx: f32, cy: f32, rx: f32, ry: f32) -> f32 {
    if rx <= 1.0e-6 || ry <= 1.0e-6 {
        return 0.0;
    }
    let dx = (nx - cx) / rx;
    let dy = (ny - cy) / ry;
    let dist_sq = dx * dx + dy * dy;
    if dist_sq >= 1.0 {
        0.0
    } else {
        (1.0 - dist_sq).sqrt()
    }
}

fn segment_intensity(nx: f32, ny: f32, ax: f32, ay: f32, bx: f32, by: f32, radius: f32) -> f32 {
    let abx = bx - ax;
    let aby = by - ay;
    let apx = nx - ax;
    let apy = ny - ay;
    let denom = abx * abx + aby * aby;
    let t = if denom <= 1.0e-6 {
        0.0
    } else {
        clamp((apx * abx + apy * aby) / denom, 0.0, 1.0)
    };
    let px = ax + abx * t;
    let py = ay + aby * t;
    let dist = ((nx - px) * (nx - px) + (ny - py) * (ny - py)).sqrt();
    if dist >= radius || radius <= 1.0e-6 {
        0.0
    } else {
        1.0 - dist / radius
    }
}

fn fly_scene_grid(
    body: &crate::drosophila::BodyState,
    visual: &crate::terrarium::visual_projection::TerrariumFlyVisualResponse,
) -> InspectGrid {
    let thorax_rx = 0.20 + visual.thorax_scale * 0.10;
    let thorax_ry = 0.16 + visual.thorax_scale * 0.08;
    let abdomen_rx = 0.16 + visual.abdomen_scale * 0.08;
    let abdomen_ry = 0.26 + visual.abdomen_scale * 0.12;
    let head_r = 0.10 + visual.head_scale * 0.06;
    let wing_rx = (0.18 + visual.wing_span * 0.24).clamp(0.20, 0.54);
    let wing_ry = (0.05 + visual.wing_width * 0.08).clamp(0.05, 0.16);
    let wing_y = -0.18 - visual.wing_angle.abs() * 0.10;
    let leg_span = (0.26 + visual.leg_span * 0.10).clamp(0.22, 0.44);
    let proboscis_len = if body.proboscis_extended {
        0.10 + visual.proboscis_extension.clamp(0.0, 0.4) * 1.2
    } else {
        0.0
    };

    raster_grid(
        18,
        18,
        "Live fly body slice from body state and render kinematics",
        "fauna_body",
        |nx, ny| {
            let thorax = ellipse_intensity(nx, ny, 0.0, -0.04, thorax_rx, thorax_ry) * 0.42;
            let abdomen = ellipse_intensity(nx, ny, 0.0, 0.32, abdomen_rx, abdomen_ry) * 0.28;
            let head = ellipse_intensity(nx, ny, 0.0, -0.54, head_r, head_r * 0.95) * 0.70;
            let wing_left = ellipse_intensity(nx, ny, 0.36, wing_y, wing_rx, wing_ry)
                * if body.is_flying { 0.58 } else { 0.34 };
            let wing_right = ellipse_intensity(nx, ny, -0.36, wing_y, wing_rx, wing_ry)
                * if body.is_flying { 0.58 } else { 0.34 };
            let leg_left = segment_intensity(nx, ny, -0.06, 0.02, -leg_span, 0.42, 0.06) * 0.22;
            let leg_right = segment_intensity(nx, ny, 0.06, 0.02, leg_span, 0.42, 0.06) * 0.22;
            let proboscis = if proboscis_len > 0.0 {
                segment_intensity(nx, ny, 0.0, -0.62, 0.0, -0.62 - proboscis_len, 0.05) * 0.94
            } else {
                0.0
            };
            thorax
                .max(abdomen)
                .max(head)
                .max(wing_left)
                .max(wing_right)
                .max(leg_left)
                .max(leg_right)
                .max(proboscis)
        },
    )
}

fn fly_cellular_grid(
    body: &crate::drosophila::BodyState,
    metabolism: &crate::fly_metabolism::FlyMetabolism,
) -> InspectGrid {
    inspect_grid(
        "Body-state matrix: energy / substrate / oxygen / metabolic rate / hunger / biomass / flight / speed",
        "cellular",
        4,
        2,
        vec![
            metabolism.energy_charge(),
            metabolism.substrate_saturation(),
            metabolism.oxygen_status(),
            metabolism.metabolic_rate_fraction(),
            metabolism.hunger(),
            clamp(metabolism.biomass_mg() / 1.0, 0.0, 1.0),
            if body.is_flying { 1.0 } else { 0.0 },
            clamp(body.speed / 3.5, 0.0, 1.0),
        ],
    )
}

fn fly_molecular_grid(metabolism: &crate::fly_metabolism::FlyMetabolism) -> InspectGrid {
    inspect_grid(
        "Fly metabolite vector: trehalose / glucose / glycogen / lipid / ATP / ADP / ambient O2",
        "molecular",
        7,
        1,
        vec![
            metabolism.hemolymph_trehalose_mm / 25.0,
            metabolism.hemolymph_glucose_mm / 10.0,
            metabolism.fat_body_glycogen_mg / 0.03,
            metabolism.fat_body_lipid_mg / 0.06,
            metabolism.muscle_atp_mm / 5.0,
            metabolism.muscle_adp_mm / 5.0,
            metabolism.ambient_o2_fraction / 0.21,
        ],
    )
}

fn fly_population_world_xy(world: &TerrariumWorld, position_mm: (f32, f32, f32)) -> (f32, f32) {
    let cell_size_mm = world.config.cell_size_mm.max(1.0e-3);
    (position_mm.0 / cell_size_mm, position_mm.1 / cell_size_mm)
}

fn nth_fly_lifecycle_stage<'a, F>(
    world: &'a TerrariumWorld,
    index: usize,
    mut filter: F,
) -> Option<&'a Fly>
where
    F: FnMut(&'a Fly) -> Option<&'a Fly>,
{
    world
        .fly_population()
        .flies
        .iter()
        .filter_map(|fly| filter(fly))
        .nth(index)
}

fn fly_larval_duration_hours(instar: u8) -> f32 {
    match instar {
        1 => 24.0,
        2 => 24.0,
        _ => 48.0,
    }
}

fn fly_egg_scene_grid(cluster: &EggCluster) -> InspectGrid {
    raster_grid(
        16,
        10,
        "Egg-cluster slice from explicit clutch state",
        "fauna_body",
        |nx, ny| {
            let mut best: f32 = 0.0;
            let extent_x = cluster
                .embryos
                .iter()
                .map(|embryo| embryo.offset_mm.0.abs())
                .fold(0.18, f32::max)
                .max(0.18);
            let extent_y = cluster
                .embryos
                .iter()
                .map(|embryo| embryo.offset_mm.1.abs())
                .fold(0.18, f32::max)
                .max(0.18);
            for embryo in &cluster.embryos {
                let cx = (embryo.offset_mm.0 / extent_x).clamp(-1.0, 1.0) * 0.62;
                let cy = (embryo.offset_mm.1 / extent_y).clamp(-1.0, 1.0) * 0.34;
                let egg_rx = 0.13;
                let egg_ry = 0.20;
                best = best.max(
                    ellipse_intensity(nx, ny, cx, cy, egg_rx, egg_ry)
                        * (0.58 + embryo.viability * 0.24),
                );
            }
            best
        },
    )
}

fn fly_egg_cellular_grid(cluster: &EggCluster, inventory: &SubstratePatchMetrics) -> InspectGrid {
    let mean_viability = cluster.mean_embryo_viability();
    inspect_grid(
        "Egg-clutch state matrix: age / quality / water / glucose / amino / O2 / viability / nucleotide",
        "cellular",
        4,
        2,
        vec![
            cluster.age_hours / 22.0,
            cluster.substrate_quality / 1.4,
            inventory.water / 2.0,
            inventory.glucose / 2.0,
            inventory.amino_acids / 1.5,
            inventory.oxygen / 0.3,
            mean_viability,
            inventory.nucleotides / 1.0,
        ],
    )
}

fn fly_embryo_scene_grid(embryo: &FlyEmbryoState) -> InspectGrid {
    let progress = clamp(embryo.age_hours / 22.0, 0.0, 1.0);
    raster_grid(
        12,
        12,
        "Single-embryo slice from explicit embryonic state",
        "fauna_body",
        |nx, ny| {
            let body =
                ellipse_intensity(nx, ny, 0.0, 0.02, 0.24, 0.42) * (0.56 + embryo.viability * 0.24);
            let yolk = ellipse_intensity(nx, ny, 0.0, 0.16, 0.14, 0.18) * (0.18 + progress * 0.18);
            let dorsal =
                segment_intensity(nx, ny, 0.0, -0.30, 0.0, 0.28, 0.05) * (0.08 + progress * 0.16);
            body.max(yolk).max(dorsal)
        },
    )
}

fn fly_embryo_cellular_grid(
    embryo: &FlyEmbryoState,
    inventory: &SubstratePatchMetrics,
) -> InspectGrid {
    inspect_grid(
        "Embryo state matrix: age / viability / water / glucose / amino / nucleotide / membrane / O2",
        "cellular",
        4,
        2,
        vec![
            embryo.age_hours / 22.0,
            embryo.viability,
            inventory.water / 0.2,
            inventory.glucose / 0.2,
            inventory.amino_acids / 0.15,
            inventory.nucleotides / 0.12,
            inventory.membrane_precursors / 0.10,
            inventory.oxygen / 0.10,
        ],
    )
}

fn fly_inventory_molecular_grid(
    label: impl Into<String>,
    inventory: &SubstratePatchMetrics,
) -> InspectGrid {
    inspect_grid(
        label,
        "molecular",
        6,
        1,
        vec![
            inventory.water / 2.0,
            inventory.glucose / 2.0,
            inventory.amino_acids / 1.5,
            inventory.nucleotides / 1.0,
            inventory.membrane_precursors / 1.0,
            inventory.oxygen / 0.3,
        ],
    )
}

fn fly_larva_scene_grid(instar: u8, energy_frac: f32) -> InspectGrid {
    let segments = match instar {
        1 => 4,
        2 => 5,
        _ => 6,
    };
    raster_grid(
        20,
        8,
        "Larval body slice from explicit instar state",
        "fauna_body",
        |nx, ny| {
            let mut best: f32 = 0.0;
            for i in 0..segments {
                let t = if segments <= 1 {
                    0.0
                } else {
                    i as f32 / (segments - 1) as f32
                };
                let cx = -0.72 + t * 1.44;
                let cy = (t * std::f32::consts::PI).sin() * 0.10;
                let rx = 0.16 + energy_frac * 0.04;
                let ry = 0.20 - t * 0.02;
                best = best.max(ellipse_intensity(nx, ny, cx, cy, rx, ry) * (0.72 - t * 0.10));
            }
            best.max(segment_intensity(nx, ny, -0.76, 0.0, 0.78, 0.04, 0.10) * 0.18)
        },
    )
}

fn fly_pupa_scene_grid(progress: f32) -> InspectGrid {
    raster_grid(
        18,
        10,
        "Pupal body slice from explicit metamorphosis state",
        "fauna_body",
        |nx, ny| {
            let body = ellipse_intensity(nx, ny, 0.0, 0.02, 0.34, 0.54) * 0.86;
            let taper = ellipse_intensity(nx, ny, 0.0, -0.50, 0.14, 0.20) * 0.42;
            let seam = segment_intensity(nx, ny, 0.0, -0.44, 0.0, 0.42, 0.04)
                * (0.18 + clamp(progress, 0.0, 1.0) * 0.18);
            body.max(taper).max(seam)
        },
    )
}

fn fly_immature_cellular_grid(
    fly: &Fly,
    inventory: &SubstratePatchMetrics,
    progress: f32,
) -> InspectGrid {
    let instar_t = match fly.stage {
        FlyLifeStage::Larva { instar, .. } => instar as f32 / 3.0,
        FlyLifeStage::Pupa { .. } => 1.0,
        _ => 0.0,
    };
    inspect_grid(
        "Immature fly state matrix: progress / energy / stage / water / glucose / amino / nucleotide / O2",
        "cellular",
        4,
        2,
        vec![
            progress,
            fly.energy / FLY_ENERGY_MAX,
            instar_t,
            inventory.water / 2.0,
            inventory.glucose / 2.0,
            inventory.amino_acids / 1.5,
            inventory.nucleotides / 1.0,
            inventory.oxygen / 0.3,
        ],
    )
}

fn fruit_scene_grid(
    shape: crate::terrarium::shape_projection::TerrariumFruitShapeDescriptor,
    ripeness: f32,
) -> InspectGrid {
    let body_rx = clamp(0.24 + shape.width_scale * 0.26, 0.22, 0.70);
    let body_ry = clamp(0.24 + shape.height_scale * 0.24, 0.20, 0.74);
    let crown_rx = clamp(body_rx * (0.34 + shape.top_taper * 0.36), 0.12, 0.46);
    let crown_ry = clamp(0.08 + shape.top_taper * 0.18, 0.06, 0.26);
    let stem_len = clamp(0.10 + shape.stem_length * 1.8, 0.08, 0.44);
    let ripe_t = ripeness.clamp(0.0, 1.0);

    raster_grid(
        16,
        16,
        "Fruit cross-section from live shape descriptor",
        "fruit_body",
        |nx, ny| {
            let stem = segment_intensity(
                nx,
                ny,
                0.0,
                -body_ry - 0.16,
                0.0,
                -body_ry - 0.16 - stem_len,
                0.05,
            ) * 0.24;
            let body =
                ellipse_intensity(nx, ny, 0.0, 0.06, body_rx, body_ry) * (0.52 + ripe_t * 0.18);
            let crown = ellipse_intensity(nx, ny, 0.0, -body_ry * 0.58, crown_rx, crown_ry) * 0.82;
            stem.max(body).max(crown)
        },
    )
}

fn fruit_cellular_grid(fruit: &crate::terrarium::TerrariumFruitPatch) -> InspectGrid {
    inspect_grid(
        "Fruit state matrix: ripeness / sugar / odor / decay / seed mass / leaf efficiency / water-use / volatile",
        "cellular",
        4,
        2,
        vec![
            fruit.source.ripeness,
            fruit.source.sugar_content,
            fruit.source.odorant_emission_rate / 0.08,
            fruit.source.decay_rate / 0.02,
            fruit.source_genome.seed_mass / 0.20,
            fruit.source_genome.leaf_efficiency / 1.65,
            fruit.source_genome.water_use_efficiency / 1.50,
            fruit.source_genome.volatile_scale / 1.85,
        ],
    )
}

fn seed_scene_grid(
    shape: crate::terrarium::shape_projection::TerrariumSeedShapeDescriptor,
    feedback: &crate::seed_cellular::SeedCellularFeedback,
) -> InspectGrid {
    let body_rx = clamp(0.26 + shape.width_scale * 0.22, 0.22, 0.66);
    let body_ry = clamp(0.18 + shape.height_scale * 0.18, 0.14, 0.46);
    let awn_len = clamp(shape.awn_length * 2.2, 0.0, 0.62);
    let radicle_len = clamp(feedback.radicle_extension * 1.4, 0.0, 0.58);
    let cotyledon_open = clamp(feedback.cotyledon_opening, 0.0, 1.0);

    raster_grid(
        16,
        12,
        "Seed cross-section from live seed shape and germination readiness",
        "seed_body",
        |nx, ny| {
            let body = ellipse_intensity(nx, ny, -0.04, 0.0, body_rx, body_ry) * 0.54;
            let awn = if awn_len > 0.02 {
                segment_intensity(nx, ny, -body_rx * 0.18, -body_ry * 0.26, -0.92, -0.62, 0.05)
                    * (0.68 + awn_len * 0.24)
            } else {
                0.0
            };
            let radicle = if radicle_len > 0.02 {
                segment_intensity(
                    nx,
                    ny,
                    body_rx * 0.72,
                    0.02,
                    body_rx * 0.72 + radicle_len,
                    0.18,
                    0.06,
                ) * 0.92
            } else {
                0.0
            };
            let cotyledon = if cotyledon_open > 0.04 {
                ellipse_intensity(
                    nx,
                    ny,
                    0.10,
                    -body_ry * 0.18,
                    body_rx * (0.24 + cotyledon_open * 0.14),
                    body_ry * (0.22 + cotyledon_open * 0.18),
                ) * 0.74
            } else {
                0.0
            };
            body.max(awn).max(radicle).max(cotyledon)
        },
    )
}

fn seed_cellular_grid(
    coat: &crate::seed_cellular::SeedClusterSnapshot,
    endosperm: &crate::seed_cellular::SeedClusterSnapshot,
    radicle: &crate::seed_cellular::SeedClusterSnapshot,
    cotyledon: &crate::seed_cellular::SeedClusterSnapshot,
) -> InspectGrid {
    let tissues = [coat, endosperm, radicle, cotyledon];
    let mut values = Vec::with_capacity(4 * 6);
    for row in 0..6 {
        for tissue in tissues {
            let value = match row {
                0 => tissue.vitality,
                1 => tissue.hydration,
                2 => tissue.energy_charge,
                3 => clamp(tissue.sugar_pool / 6.0, 0.0, 1.0),
                4 => clamp(tissue.nitrogen_pool / 3.0, 0.0, 1.0),
                _ => clamp(
                    tissue.transcript_germination_program * 0.72
                        + tissue.transcript_stress_response * 0.28,
                    0.0,
                    1.0,
                ),
            };
            values.push(value);
        }
    }
    inspect_grid(
        "Seed tissue matrix: vitality / hydration / energy / sugars / nitrogen / germination",
        "cellular",
        4,
        6,
        values,
    )
}

fn seed_molecular_grid(
    coat: &crate::seed_cellular::SeedClusterSnapshot,
    endosperm: &crate::seed_cellular::SeedClusterSnapshot,
    radicle: &crate::seed_cellular::SeedClusterSnapshot,
    cotyledon: &crate::seed_cellular::SeedClusterSnapshot,
) -> InspectGrid {
    let tissues = [coat, endosperm, radicle, cotyledon];
    let mut values = Vec::with_capacity(3 * 4);
    for tissue in tissues {
        values.push(clamp(tissue.chem_glucose / 3.0, 0.0, 1.0));
        values.push(clamp(tissue.chem_oxygen / 4.0, 0.0, 1.0));
        values.push(clamp(tissue.chem_atp / 5.0, 0.0, 1.0));
    }
    inspect_grid(
        "Seed tissue chemistry matrix: glucose / O2 / ATP",
        "molecular",
        3,
        4,
        values,
    )
}

fn earthworm_scene_grid(
    visual: &crate::terrarium::visual_projection::TerrariumEarthwormVisualResponse,
) -> InspectGrid {
    let segment_count = visual.segment_count.max(4) as usize;
    raster_grid(
        20,
        8,
        "Earthworm body slice from live segment marker",
        "fauna_body",
        |nx, ny| {
            let mut value = 0.0f32;
            for idx in 0..segment_count {
                let t = if segment_count <= 1 {
                    0.0
                } else {
                    idx as f32 / (segment_count - 1) as f32
                };
                let center_x = -0.82 + t * 1.64;
                let center_y = visual.curl * (std::f32::consts::PI * t).sin() * 0.28;
                let taper = 0.72 + (1.0 - (t * 2.0 - 1.0).abs()) * 0.28;
                let seg = ellipse_intensity(
                    nx,
                    ny,
                    center_x,
                    center_y,
                    0.12 * visual.thickness_scale * taper,
                    0.18 * visual.thickness_scale * taper,
                ) * if (0.38..=0.58).contains(&t) {
                    0.78
                } else {
                    0.42
                };
                value = value.max(seg);
            }
            value
        },
    )
}

fn earthworm_state_grid(
    world: &TerrariumWorld,
    flat: usize,
    visual: &crate::terrarium::visual_projection::TerrariumEarthwormVisualResponse,
) -> InspectGrid {
    inspect_grid(
        "Earthworm hotspot matrix: activity / density / biomass / bioturbation / moisture / organic / microbes / symbionts",
        "biology",
        4,
        2,
        vec![
            visual.activity,
            world.earthworm_population.population_density[flat] / 200.0,
            world.earthworm_population.biomass_per_voxel[flat] / 0.6,
            world.earthworm_population.bioturbation_rate[flat] / 5.0,
            world.moisture[flat],
            world.organic_matter[flat],
            world.microbial_biomass[flat] / 50.0,
            world.symbiont_biomass[flat] / 50.0,
        ],
    )
}

fn nematode_scene_grid(
    visual: &crate::terrarium::visual_projection::TerrariumNematodeVisualResponse,
) -> InspectGrid {
    raster_grid(
        20,
        7,
        "Nematode body slice from live guild marker",
        "fauna_body",
        |nx, ny| {
            let mut value = 0.0f32;
            let segment_count = 5usize;
            for idx in 0..segment_count {
                let t = if segment_count <= 1 {
                    0.0
                } else {
                    idx as f32 / (segment_count - 1) as f32
                };
                let center_x = -0.78 + t * 1.56;
                let center_y = visual.curl * (std::f32::consts::PI * t).sin() * 0.22;
                let taper = 1.0 - t * 0.42;
                let seg = ellipse_intensity(
                    nx,
                    ny,
                    center_x,
                    center_y,
                    0.09 * visual.thickness_scale * taper,
                    0.14 * visual.thickness_scale * taper,
                ) * if idx == segment_count - 1 { 0.72 } else { 0.38 };
                value = value.max(seg);
            }
            if visual.stylet_length_scale > 0.05 {
                value = value.max(
                    segment_intensity(
                        nx,
                        ny,
                        0.76,
                        0.0,
                        0.76 + visual.stylet_length_scale * 0.34,
                        0.0,
                        0.04,
                    ) * 0.94,
                );
            }
            value
        },
    )
}

fn nematode_state_grid(
    world: &TerrariumWorld,
    flat: usize,
    density: f32,
    biomass: f32,
    visual: &crate::terrarium::visual_projection::TerrariumNematodeVisualResponse,
) -> InspectGrid {
    inspect_grid(
        "Nematode hotspot matrix: activity / density / biomass / stylet / microbes / moisture / canopy / water",
        "biology",
        4,
        2,
        vec![
            visual.activity,
            density / 40.0,
            biomass / 0.0025,
            visual.stylet_length_scale,
            world.microbial_biomass[flat] / 50.0,
            world.moisture[flat],
            world.canopy_cover[flat],
            world.water_mask[flat],
        ],
    )
}

fn substrate_vertical_grid(
    world: &TerrariumWorld,
    x: usize,
    y: usize,
    label: impl Into<String>,
) -> InspectGrid {
    let mut values =
        Vec::with_capacity(SUBSTRATE_CHEMISTRY_GRID_SPECIES.len() * world.config.depth);
    for z in 0..world.config.depth {
        for species_id in SUBSTRATE_CHEMISTRY_GRID_SPECIES {
            let sample = world.substrate.patch_mean_species(species_id, x, y, z, 1);
            let normalized = normalize_terrarium_geochemistry_for_inspect(species_id, sample);
            values.push(normalized);
        }
    }
    InspectGrid {
        label: label.into(),
        width: SUBSTRATE_CHEMISTRY_GRID_SPECIES.len(),
        height: world.config.depth,
        palette: "substrate".into(),
        values,
    }
}

fn plant_morphology_grid(plant: &crate::terrarium::TerrariumPlant) -> Option<InspectGrid> {
    let nodes = plant
        .morphology
        .generate_nodes_with_context(plant.physiology.fruit_count(), plant.cellular.vitality());
    if nodes.is_empty() {
        return None;
    }

    let width = 28usize;
    let height = 22usize;
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for node in &nodes {
        min_x = min_x.min(node.position[0] - node.radius);
        max_x = max_x.max(node.position[0] + node.radius);
        min_y = min_y.min(node.position[1] - node.radius);
        max_y = max_y.max(node.position[1] + node.radius);
    }
    let span_x = (max_x - min_x).max(0.25);
    let span_y = (max_y - min_y).max(0.25);
    let mut values = vec![0.0f32; width * height];

    for node in &nodes {
        let encoded = match node.node_type {
            crate::botany::morphology::NodeType::Trunk => 0.18,
            crate::botany::morphology::NodeType::Branch => 0.34,
            crate::botany::morphology::NodeType::Bud => 0.50,
            crate::botany::morphology::NodeType::Leaf => 0.72,
            crate::botany::morphology::NodeType::Fruit => 0.92,
        };
        let px = ((node.position[0] - min_x) / span_x * (width as f32 - 1.0)).round() as isize;
        let py =
            ((1.0 - (node.position[1] - min_y) / span_y) * (height as f32 - 1.0)).round() as isize;
        let pr = ((node.radius / span_x.max(span_y)) * width.max(height) as f32 * 4.5)
            .clamp(1.0, 4.0) as isize;
        for oy in -pr..=pr {
            for ox in -pr..=pr {
                if ox * ox + oy * oy > pr * pr {
                    continue;
                }
                let gx = px + ox;
                let gy = py + oy;
                if gx < 0 || gy < 0 || gx >= width as isize || gy >= height as isize {
                    continue;
                }
                let idx = gy as usize * width + gx as usize;
                values[idx] = values[idx].max(encoded);
            }
        }
    }

    Some(InspectGrid {
        label: "Live morphology slice from plant node graph".into(),
        width,
        height,
        palette: "botany".into(),
        values,
    })
}

fn plant_cellular_grid(
    leaf: &crate::plant_cellular::PlantClusterSnapshot,
    stem: &crate::plant_cellular::PlantClusterSnapshot,
    root: &crate::plant_cellular::PlantClusterSnapshot,
    meristem: &crate::plant_cellular::PlantClusterSnapshot,
) -> InspectGrid {
    let tissues = [leaf, stem, root, meristem];
    let width = tissues.len();
    let height = 6usize;
    let mut values = Vec::with_capacity(width * height);
    for row in 0..height {
        for tissue in tissues {
            let value = match row {
                0 => tissue.vitality,
                1 => cluster_energy_charge(tissue),
                2 => clamp(tissue.state_water / 8.0, 0.0, 1.0),
                3 => clamp((tissue.state_glucose + tissue.state_starch) / 4.0, 0.0, 1.0),
                4 => clamp(
                    (tissue.state_nitrate + tissue.state_amino_acid) / 2.0,
                    0.0,
                    1.0,
                ),
                _ => clamp(
                    (tissue.division_buffer + tissue.transcript_cell_cycle * 0.6) / 1.5,
                    0.0,
                    1.0,
                ),
            };
            values.push(value);
        }
    }
    InspectGrid {
        label: "Tissue state matrix: vitality / energy / water / sugars / nitrogen / division"
            .into(),
        width,
        height,
        palette: "cellular".into(),
        values,
    }
}

fn plant_molecular_grid(
    leaf: &crate::plant_cellular::PlantClusterSnapshot,
    stem: &crate::plant_cellular::PlantClusterSnapshot,
    root: &crate::plant_cellular::PlantClusterSnapshot,
    meristem: &crate::plant_cellular::PlantClusterSnapshot,
) -> InspectGrid {
    let tissues = [leaf, stem, root, meristem];
    let width = 6usize;
    let height = tissues.len();
    let mut values = Vec::with_capacity(width * height);
    for tissue in tissues {
        values.push(clamp(tissue.state_atp / 5.0, 0.0, 1.0));
        values.push(clamp(tissue.state_adp / 5.0, 0.0, 1.0));
        values.push(clamp(tissue.state_glucose / 3.0, 0.0, 1.0));
        values.push(clamp(tissue.state_water / 8.0, 0.0, 1.0));
        values.push(clamp(tissue.state_nitrate / 2.0, 0.0, 1.0));
        values.push(clamp(tissue.chem_oxygen / 4.0, 0.0, 1.0));
    }
    InspectGrid {
        label: "Tissue metabolite matrix: ATP / ADP / glucose / water / nitrate / O2".into(),
        width,
        height,
        palette: "molecular".into(),
        values,
    }
}

fn root_zone_depth(world: &TerrariumWorld, depth_bias: f32) -> usize {
    let max_z = world.config.depth.saturating_sub(1);
    clamp(depth_bias * max_z as f32, 0.0, max_z as f32).round() as usize
}

fn clamped_cell(world: &TerrariumWorld, x: f32, y: f32) -> (usize, usize) {
    (
        clamp(x.floor(), 0.0, world.config.width.saturating_sub(1) as f32) as usize,
        clamp(y.floor(), 0.0, world.config.height.saturating_sub(1) as f32) as usize,
    )
}

fn required_index(query: &InspectQuery) -> Result<usize, String> {
    query.index.ok_or_else(|| "Missing inspect index".into())
}

fn required_xy(
    world: &TerrariumWorld,
    x: Option<f32>,
    y: Option<f32>,
) -> Result<(usize, usize), String> {
    let x = x.ok_or_else(|| "Missing x".to_string())?;
    let y = y.ok_or_else(|| "Missing y".to_string())?;
    Ok(clamped_cell(world, x, y))
}

fn describe_owner(owner: SoilOwnershipClass) -> String {
    match owner {
        SoilOwnershipClass::Background => "background domain".into(),
        SoilOwnershipClass::ExplicitMicrobeCohort { cohort_id } => {
            format!("explicit microbe cohort #{cohort_id}")
        }
        SoilOwnershipClass::GenotypePacketRegion { genotype_id } => {
            format!("packet region genotype #{genotype_id}")
        }
        SoilOwnershipClass::PlantTissueRegion { plant_id } => {
            format!("plant tissue region #{plant_id}")
        }
        SoilOwnershipClass::AtomisticProbeRegion { probe_id } => {
            format!("atomistic probe #{probe_id}")
        }
    }
}

fn cluster_energy_charge(snapshot: &crate::plant_cellular::PlantClusterSnapshot) -> f32 {
    let total = snapshot.state_atp + snapshot.state_adp + snapshot.chem_amp.max(0.0);
    let nucleotide_ratio = if total <= 1.0e-9 {
        0.0
    } else {
        snapshot.state_atp / total
    };
    clamp(
        nucleotide_ratio * 0.62
            + snapshot.chem_atp.max(0.0)
                / (snapshot.chem_atp.max(0.0)
                    + snapshot.chem_adp.max(0.0)
                    + snapshot.chem_amp.max(0.0)
                    + 1.0e-6)
                * 0.38,
        0.0,
        1.0,
    )
}

fn metric(label: impl Into<String>, value: String, fraction: Option<f32>) -> InspectMetric {
    InspectMetric {
        label: label.into(),
        value,
        fraction: fraction.map(|value| value.clamp(0.0, 1.0)),
    }
}

fn composition(
    label: impl Into<String>,
    amount: f32,
    unit: impl Into<String>,
    category: impl Into<String>,
) -> InspectComposition {
    InspectComposition {
        label: label.into(),
        amount: if amount.is_finite() {
            amount.max(0.0)
        } else {
            0.0
        },
        unit: unit.into(),
        category: category.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        inspect_earthworm, inspect_fly, inspect_fly_egg, inspect_fly_embryo, inspect_fly_larva,
        inspect_fly_pupa, inspect_fruit, inspect_nematode, inspect_plant, inspect_seed,
        inspect_soil, inspect_water, neighborhood_values,
    };
    use crate::drosophila::DrosophilaScale;
    use crate::drosophila_population::{
        EggCluster, Fly, FlyEmbryoState, FlyLifeStage, FlySex, FLY_ENERGY_MAX,
    };
    use crate::terrarium::material_exchange::deposit_species_to_inventory;
    use crate::terrarium::{
        RegionalMaterialInventory, TerrariumDemoPreset, TerrariumSpecies, TerrariumWorld,
    };

    fn assert_grid_shape(
        grid: &Option<crate::terrarium_web_protocol::InspectGrid>,
        width: usize,
        height: usize,
        palette: &str,
    ) {
        let grid = grid.as_ref().expect("expected inspect grid");
        assert_eq!(grid.width, width);
        assert_eq!(grid.height, height);
        assert_eq!(grid.palette, palette);
        assert_eq!(grid.values.len(), width * height);
    }

    fn assert_grid_has_signal(grid: &Option<crate::terrarium_web_protocol::InspectGrid>) {
        let grid = grid.as_ref().expect("expected inspect grid");
        assert!(
            grid.values.iter().any(|value| *value > 0.0),
            "grid should contain some live signal: {} [{}]",
            grid.label,
            grid.palette
        );
    }

    fn test_embryo(id: u32, age_hours: f32, viability: f32) -> FlyEmbryoState {
        let mut material_inventory = RegionalMaterialInventory::new(format!("embryo:test:{id}"));
        deposit_species_to_inventory(&mut material_inventory, TerrariumSpecies::Water, 0.08);
        deposit_species_to_inventory(&mut material_inventory, TerrariumSpecies::Glucose, 0.06);
        deposit_species_to_inventory(
            &mut material_inventory,
            TerrariumSpecies::AminoAcidPool,
            0.04,
        );
        deposit_species_to_inventory(
            &mut material_inventory,
            TerrariumSpecies::NucleotidePool,
            0.03,
        );
        deposit_species_to_inventory(
            &mut material_inventory,
            TerrariumSpecies::MembranePrecursorPool,
            0.02,
        );
        FlyEmbryoState {
            id,
            sex: if id % 2 == 0 {
                FlySex::Female
            } else {
                FlySex::Male
            },
            offset_mm: (
                (id % 3) as f32 * 0.08 - 0.08,
                (id % 2) as f32 * 0.09 - 0.045,
            ),
            age_hours,
            viability,
            material_inventory,
        }
    }

    #[test]
    fn neighborhood_values_keep_selected_cell_centered_at_edges() {
        let values = neighborhood_values(3, 3, 0, 0, 1, |x, y| (y * 3 + x + 1) as f32 / 8.0);

        assert_eq!(
            values,
            vec![0.0, 0.0, 0.0, 0.0, 0.125, 0.25, 0.0, 0.5, 0.625,]
        );
    }

    #[test]
    fn neighborhood_values_preserve_full_window_away_from_edges() {
        let values = neighborhood_values(5, 5, 2, 2, 1, |x, y| (y * 5 + x) as f32 / 32.0);

        assert_eq!(
            values,
            vec![0.1875, 0.21875, 0.25, 0.34375, 0.375, 0.40625, 0.5, 0.53125, 0.5625,]
        );
    }

    #[test]
    fn fruit_seed_and_soil_fauna_inspect_use_grid_backed_views() {
        let world = TerrariumWorld::demo_preset(7, false, TerrariumDemoPreset::Demo)
            .expect("demo preset should build");

        let fruit = inspect_fruit(&world, TerrariumDemoPreset::Demo, 0)
            .expect("fruit inspect should exist");
        assert_grid_shape(&fruit.scene_grid, 16, 16, "fruit_body");
        assert_grid_has_signal(&fruit.scene_grid);
        assert_grid_shape(&fruit.cellular_grid, 4, 2, "cellular");
        assert_grid_shape(&fruit.molecular_grid, 22, world.config.depth, "substrate");
        assert!(
            fruit
                .molecular
                .iter()
                .any(|metric| metric.label == "Surface Inventory Glucose"),
            "fruit inspect should expose fruit-surface inventory chemistry"
        );

        let seed = inspect_seed(&world, TerrariumDemoPreset::Demo, 0)
            .expect("seed inspect should exist");
        assert_grid_shape(&seed.scene_grid, 16, 12, "seed_body");
        assert_grid_has_signal(&seed.scene_grid);
        assert_grid_shape(&seed.cellular_grid, 4, 6, "cellular");
        assert_grid_shape(&seed.molecular_grid, 3, 4, "molecular");

        let earthworm = inspect_earthworm(&world, TerrariumDemoPreset::Demo, 0)
            .expect("earthworm inspect should exist");
        assert_grid_shape(&earthworm.scene_grid, 20, 8, "fauna_body");
        assert_grid_has_signal(&earthworm.scene_grid);
        assert_grid_shape(&earthworm.cellular_grid, 4, 2, "biology");
        assert_grid_shape(
            &earthworm.molecular_grid,
            22,
            world.config.depth,
            "substrate",
        );

        let nematode = inspect_nematode(&world, TerrariumDemoPreset::Demo, 0)
            .expect("nematode inspect should exist");
        assert_grid_shape(&nematode.scene_grid, 20, 7, "fauna_body");
        assert_grid_has_signal(&nematode.scene_grid);
        assert_grid_shape(&nematode.cellular_grid, 4, 2, "biology");
        assert_grid_shape(
            &nematode.molecular_grid,
            22,
            world.config.depth,
            "substrate",
        );
    }

    #[test]
    fn fly_inspect_uses_live_body_and_metabolism_grids() {
        let mut world = TerrariumWorld::demo(7, false).expect("demo world should build");
        world.add_fruit(5, 5, 1.2, Some(1.0));
        world.add_fly(DrosophilaScale::Tiny, 5.0, 5.0, 42);
        let _ = world.step_flies();
        world.step_fly_population(3600.0);

        let fly =
            inspect_fly(&world, TerrariumDemoPreset::Demo, 0).expect("fly inspect should exist");
        assert_grid_shape(&fly.scene_grid, 18, 18, "fauna_body");
        assert_grid_has_signal(&fly.scene_grid);
        assert_grid_shape(&fly.cellular_grid, 4, 2, "cellular");
        assert_grid_shape(&fly.molecular_grid, 7, 1, "molecular");
        assert!(
            fly.summary
                .iter()
                .any(|metric| metric.label == "Life Stage"),
            "fly inspect should surface the linked lifecycle state"
        );
        assert!(
            fly.molecular
                .iter()
                .any(|metric| metric.label == "Reproductive Ledger Glucose"),
            "fly inspect should expose the linked reproductive chemistry ledger"
        );
    }

    #[test]
    fn fly_lifecycle_inspect_supports_egg_larva_and_pupa_targets() {
        let mut world = TerrariumWorld::demo(17, false).expect("demo world should build");
        let cs = world.config.cell_size_mm.max(1.0e-3);

        let mut egg_inventory = RegionalMaterialInventory::new("egg:test".into());
        deposit_species_to_inventory(&mut egg_inventory, TerrariumSpecies::Water, 0.8);
        deposit_species_to_inventory(&mut egg_inventory, TerrariumSpecies::Glucose, 0.6);
        deposit_species_to_inventory(&mut egg_inventory, TerrariumSpecies::NucleotidePool, 0.2);
        let mut egg_cluster = EggCluster {
            position: (5.0 * cs, 5.0 * cs),
            count: 0,
            age_hours: 0.0,
            substrate_quality: 0.9,
            material_inventory: egg_inventory.clone(),
            embryos: (0..9)
                .map(|i| test_embryo(200 + i, 8.0 + i as f32 * 0.1, 0.78))
                .collect(),
        };
        egg_cluster.refresh_summary();
        world.fly_pop.egg_clusters.push(egg_cluster);

        let mut larva = Fly::new_adult(900, FlySex::Female, (6.0 * cs, 6.0 * cs, 0.0));
        larva.stage = FlyLifeStage::Larva {
            instar: 2,
            age_hours: 11.0,
        };
        larva.energy = FLY_ENERGY_MAX * 0.42;
        larva.material_inventory = egg_inventory.clone();
        world.fly_pop.add_fly(larva);

        let mut pupa = Fly::new_adult(901, FlySex::Male, (7.0 * cs, 7.0 * cs, 0.0));
        pupa.stage = FlyLifeStage::Pupa { age_hours: 40.0 };
        pupa.energy = FLY_ENERGY_MAX * 0.36;
        pupa.material_inventory = egg_inventory;
        world.fly_pop.add_fly(pupa);

        let egg = inspect_fly_egg(&world, TerrariumDemoPreset::Demo, 0)
            .expect("fly egg inspect should exist");
        assert_grid_shape(&egg.scene_grid, 16, 10, "fauna_body");
        assert_grid_has_signal(&egg.scene_grid);
        assert_grid_shape(&egg.cellular_grid, 4, 2, "cellular");
        assert_grid_shape(&egg.molecular_grid, 6, 1, "molecular");
        assert!(
            egg.summary
                .iter()
                .any(|metric| metric.label == "Mean Viability"),
            "clutch inspect should surface embryo-backed viability"
        );

        let embryo = inspect_fly_embryo(&world, TerrariumDemoPreset::Demo, 0)
            .expect("fly embryo inspect should exist");
        assert_grid_shape(&embryo.scene_grid, 12, 12, "fauna_body");
        assert_grid_has_signal(&embryo.scene_grid);
        assert_grid_shape(&embryo.cellular_grid, 4, 2, "cellular");
        assert_grid_shape(&embryo.molecular_grid, 6, 1, "molecular");
        assert!(
            embryo
                .summary
                .iter()
                .any(|metric| metric.label == "Embryo ID"),
            "individual embryo inspect should expose stable embryo identity"
        );

        let larva = inspect_fly_larva(&world, TerrariumDemoPreset::Demo, 0)
            .expect("fly larva inspect should exist");
        assert_grid_shape(&larva.scene_grid, 20, 8, "fauna_body");
        assert_grid_has_signal(&larva.scene_grid);
        assert_grid_shape(&larva.cellular_grid, 4, 2, "cellular");
        assert_grid_shape(&larva.molecular_grid, 6, 1, "molecular");

        let pupa = inspect_fly_pupa(&world, TerrariumDemoPreset::Demo, 0)
            .expect("fly pupa inspect should exist");
        assert_grid_shape(&pupa.scene_grid, 18, 10, "fauna_body");
        assert_grid_has_signal(&pupa.scene_grid);
        assert_grid_shape(&pupa.cellular_grid, 4, 2, "cellular");
        assert_grid_shape(&pupa.molecular_grid, 6, 1, "molecular");
    }

    #[test]
    fn plant_soil_and_water_inspect_expose_expanded_substrate_channels() {
        let world = TerrariumWorld::demo_preset(11, false, TerrariumDemoPreset::Demo)
            .expect("micro terrarium preset should build");

        let plant = inspect_plant(&world, TerrariumDemoPreset::Demo, 0)
            .expect("plant inspect should exist");
        assert!(
            plant
                .molecular
                .iter()
                .any(|metric| metric.label == "Root-Zone Amino"),
            "plant inspect should expose amino-acid root-zone chemistry"
        );
        assert!(
            plant
                .composition
                .iter()
                .any(|entry| entry.label == "Root Inventory Amino Acids"),
            "plant inspect should expose plant-owned inventory chemistry"
        );

        let seed = inspect_seed(&world, TerrariumDemoPreset::Demo, 0)
            .expect("seed inspect should exist");
        assert!(
            seed.molecular
                .iter()
                .any(|metric| metric.label == "Seed Inventory Amino"),
            "seed inspect should expose seed-owned inventory chemistry"
        );
        assert!(
            seed.position.as_ref().and_then(|pos| pos.z).unwrap_or(-1.0) >= 0.0,
            "seed inspect should report burial-aware depth"
        );

        let soil = inspect_soil(&world, TerrariumDemoPreset::Demo, 0, 0)
            .expect("soil inspect should exist");
        assert_grid_shape(&soil.molecular_grid, 22, world.config.depth, "substrate");
        assert!(
            soil.composition
                .iter()
                .any(|entry| entry.label == "Membrane Precursors"),
            "soil inspect should expose membrane precursor composition"
        );
        assert!(
            soil.molecular
                .iter()
                .any(|metric| metric.label == "Base Saturation"),
            "soil inspect should expose exchange chemistry state"
        );
        assert!(
            soil.composition
                .iter()
                .any(|entry| entry.label == "Exchangeable Calcium"),
            "soil inspect should expose exchangeable mineral chemistry"
        );
        assert!(
            soil.composition
                .iter()
                .any(|entry| entry.label == "Bicarbonate"),
            "soil inspect should expose local alkalinity chemistry"
        );
        assert!(
            soil.composition
                .iter()
                .any(|entry| entry.label == "Sorbed Ferric Hydroxide"),
            "soil inspect should expose hydroxide-sequestered metal chemistry"
        );

        let water = inspect_water(
            &world,
            TerrariumDemoPreset::Demo,
            Some(0),
            None,
            None,
        )
        .expect("water inspect should exist");
        assert_grid_shape(&water.molecular_grid, 22, world.config.depth, "substrate");
        assert!(
            water
                .composition
                .iter()
                .any(|entry| entry.label == "Dissolved Nucleotides"),
            "water inspect should expose nucleotide composition"
        );
    }

    // -----------------------------------------------------------------------
    // Scale-level inspection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_molecular_detail_water() {
        let query = super::InspectQuery {
            kind: "plant".into(),
            index: Some(0),
            x: None,
            y: None,
            scale: Some("molecular".into()),
            tissue: None,
            molecule: Some("water".into()),
            atom_index: None,
        };
        let detail = super::build_molecular_detail(&query, None).expect("water molecular detail");
        assert_eq!(detail.atoms.len(), 3, "water should have 3 atoms");
        assert_eq!(detail.bonds.len(), 2, "water should have 2 bonds");
        assert!(detail.formula.contains('H'), "formula should contain H");
        assert!(detail.formula.contains('O'), "formula should contain O");
    }

    #[test]
    fn test_molecular_detail_glucose() {
        let query = super::InspectQuery {
            kind: "plant".into(),
            index: Some(0),
            x: None,
            y: None,
            scale: Some("molecular".into()),
            tissue: None,
            molecule: Some("glucose".into()),
            atom_index: None,
        };
        let detail = super::build_molecular_detail(&query, None).expect("glucose molecular detail");
        assert_eq!(detail.atoms.len(), 24, "glucose pyranose should have 24 atoms");
        assert_eq!(detail.bonds.len(), 24, "glucose pyranose should have 24 bonds");
        assert!(detail.molecular_weight > 170.0 && detail.molecular_weight < 190.0,
            "glucose MW should be ~180, got {}", detail.molecular_weight);
    }

    #[test]
    fn test_molecular_detail_atp() {
        let query = super::InspectQuery {
            kind: "plant".into(),
            index: Some(0),
            x: None,
            y: None,
            scale: Some("molecular".into()),
            tissue: None,
            molecule: Some("atp".into()),
            atom_index: None,
        };
        let detail = super::build_molecular_detail(&query, None).expect("ATP molecular detail");
        assert_eq!(detail.atoms.len(), 47, "ATP should have 47 atoms");
        assert!(detail.molecular_weight > 480.0, "ATP MW should be >480, got {}", detail.molecular_weight);
    }

    #[test]
    fn test_molecular_asset_by_name() {
        // These should all resolve to embedded molecules
        for name in &["water", "glucose", "atp", "amino_acids", "nucleotide"] {
            assert!(
                crate::terrarium::terrarium_molecular_asset_by_name(name).is_some(),
                "terrarium_molecular_asset_by_name('{name}') should return Some"
            );
        }
    }

    #[test]
    fn test_atomic_detail_from_molecule() {
        let query = super::InspectQuery {
            kind: "C".into(),
            index: None,
            x: None,
            y: None,
            scale: Some("atomic".into()),
            tissue: None,
            molecule: Some("water".into()),
            atom_index: None,
        };
        let atoms = super::build_atomic_detail(&query).expect("atomic detail from water");
        assert_eq!(atoms.len(), 3);
        // First atom should be oxygen
        assert_eq!(atoms[0].symbol, "O");
        assert_eq!(atoms[0].atomic_number, 8);
        assert!(atoms[0].quantum_state.is_some(), "should have quantum state");
    }

    #[test]
    fn test_atomic_detail_single_element() {
        let query = super::InspectQuery {
            kind: "C".into(),
            index: None,
            x: None,
            y: None,
            scale: Some("atomic".into()),
            tissue: None,
            molecule: None,
            atom_index: None,
        };
        let atoms = super::build_atomic_detail(&query).expect("atomic detail for C");
        assert_eq!(atoms.len(), 1);
        assert_eq!(atoms[0].symbol, "C");
        assert_eq!(atoms[0].atomic_number, 6);
        assert!((atoms[0].vdw_radius - 1.70).abs() < 0.01);
        assert_eq!(atoms[0].cpk_color, [80, 80, 80]);
    }

    #[test]
    fn test_backward_compat_inspect() {
        let world = TerrariumWorld::demo(42, false).expect("demo world should build");
        // Query without scale field should still work
        let query = super::InspectQuery {
            kind: "soil".into(),
            index: None,
            x: Some(2.0),
            y: Some(2.0),
            scale: None,
            tissue: None,
            molecule: None,
            atom_index: None,
        };
        let response = super::build_scale_inspect_data(
            &world,
            TerrariumDemoPreset::Demo,
            &query,
        )
        .expect("backward-compatible inspect");
        assert_eq!(
            response.scale,
            crate::terrarium::scale_level::ScaleLevel::Ecosystem
        );
        assert!(response.organism_components.is_none());
        assert!(response.cellular_detail.is_none());
        assert!(response.molecular_detail.is_none());
        assert!(response.atomic_detail.is_none());
        // Base inspect data should be populated
        assert!(!response.base.summary.is_empty());
    }

    #[test]
    fn test_organism_components_plant() {
        let world = TerrariumWorld::demo(42, false).expect("demo world should build");
        if world.plants.is_empty() {
            return; // skip if no plants in demo preset
        }
        let query = super::InspectQuery {
            kind: "plant".into(),
            index: Some(0),
            x: None,
            y: None,
            scale: Some("organism".into()),
            tissue: None,
            molecule: None,
            atom_index: None,
        };
        let components = super::build_organism_components(&world, &query)
            .expect("plant organism components");
        let names: Vec<&str> = components.iter().map(|c| c.component_name.as_str()).collect();
        assert!(names.contains(&"leaf"), "should have leaf component");
        assert!(names.contains(&"stem"), "should have stem component");
        assert!(names.contains(&"root"), "should have root component");
        assert!(names.contains(&"meristem"), "should have meristem component");
    }

    #[test]
    fn test_cellular_detail_plant() {
        let world = TerrariumWorld::demo(42, false).expect("demo world should build");
        if world.plants.is_empty() {
            return;
        }
        let query = super::InspectQuery {
            kind: "plant".into(),
            index: Some(0),
            x: None,
            y: None,
            scale: Some("cellular".into()),
            tissue: Some("leaf".into()),
            molecule: None,
            atom_index: None,
        };
        let detail = super::build_cellular_detail(&world, &query)
            .expect("plant cellular detail");
        assert_eq!(detail.cell_type, "mesophyll");
        let organelle_names: Vec<&str> = detail.organelles.iter().map(|o| o.name.as_str()).collect();
        assert!(organelle_names.contains(&"chloroplast"), "leaf should have chloroplasts");
        assert!(organelle_names.contains(&"cell_wall"), "plant cell should have cell wall");
    }

    #[test]
    fn test_cellular_detail_fly_compound_eye() {
        let detail = super::build_fly_cellular_detail(Some("compound_eye"));
        assert_eq!(detail.cell_type, "photoreceptor (ommatidium)");
        let names: Vec<&str> = detail.organelles.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"rhabdomere"), "compound eye should have rhabdomere");
        assert!(names.contains(&"pigment_granule"), "compound eye should have pigment granules");
        let gene_names: Vec<&str> = detail.active_genes.iter().map(|g| g.name.as_str()).collect();
        assert!(gene_names.contains(&"Rh1"), "compound eye should express Rh1 rhodopsin");
        assert!(gene_names.contains(&"ninaE"), "compound eye should express ninaE opsin");
    }

    #[test]
    fn test_cellular_detail_fly_neural_circuit() {
        let detail = super::build_fly_cellular_detail(Some("neural_circuit"));
        assert_eq!(detail.cell_type, "neuron (Kenyon cell)");
        let names: Vec<&str> = detail.organelles.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"synaptic_vesicle"), "neuron should have synaptic vesicles");
        let gene_names: Vec<&str> = detail.active_genes.iter().map(|g| g.name.as_str()).collect();
        assert!(gene_names.contains(&"elav"), "neuron should express elav");
        assert!(gene_names.contains(&"Syt1"), "neuron should express Syt1");
    }

    #[test]
    fn test_cellular_detail_earthworm_segment_muscle() {
        let detail = super::build_earthworm_cellular_detail(None);
        assert_eq!(detail.cell_type, "circular/longitudinal muscle");
        let names: Vec<&str> = detail.organelles.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"myofilament"), "segment muscle should have myofilaments");
        assert!(names.contains(&"mitochondrion"), "segment muscle should have mitochondria");
        let gene_names: Vec<&str> = detail.active_genes.iter().map(|g| g.name.as_str()).collect();
        assert!(gene_names.contains(&"mhc-1"), "segment muscle should express mhc-1");

        // Also test chloragogenous tissue
        let chlor = super::build_earthworm_cellular_detail(Some("chloragogenous"));
        assert_eq!(chlor.cell_type, "chloragocyte (fat body)");
        let chlor_genes: Vec<&str> = chlor.active_genes.iter().map(|g| g.name.as_str()).collect();
        assert!(chlor_genes.contains(&"vitellogenin"));
        assert!(chlor_genes.contains(&"metallothionein"));

        // Nephridium tissue
        let neph = super::build_earthworm_cellular_detail(Some("nephridium"));
        assert_eq!(neph.cell_type, "excretory cell (nephridium)");
        let neph_names: Vec<&str> = neph.organelles.iter().map(|o| o.name.as_str()).collect();
        assert!(neph_names.contains(&"flame_cell"));
    }

    #[test]
    fn test_cellular_detail_nematode_body_wall() {
        let detail = super::build_nematode_cellular_detail(None);
        assert_eq!(detail.cell_type, "obliquely striated muscle");
        let gene_names: Vec<&str> = detail.active_genes.iter().map(|g| g.name.as_str()).collect();
        assert!(gene_names.contains(&"unc-54"), "body wall should express unc-54 myosin");
        assert!(gene_names.contains(&"unc-15"), "body wall should express unc-15 paramyosin");

        // Pharyngeal muscle
        let phar = super::build_nematode_cellular_detail(Some("pharyngeal_muscle"));
        assert_eq!(phar.cell_type, "pharynx pump cell");
        let phar_genes: Vec<&str> = phar.active_genes.iter().map(|g| g.name.as_str()).collect();
        assert!(phar_genes.contains(&"myo-2"));

        // Sensory neuron
        let sens = super::build_nematode_cellular_detail(Some("sensory_neuron"));
        assert_eq!(sens.cell_type, "amphid sensory neuron");
        let sens_names: Vec<&str> = sens.organelles.iter().map(|o| o.name.as_str()).collect();
        assert!(sens_names.contains(&"cilium"), "amphid sensory should have cilium");
        let sens_genes: Vec<&str> = sens.active_genes.iter().map(|g| g.name.as_str()).collect();
        assert!(sens_genes.contains(&"odr-10"));
        assert!(sens_genes.contains(&"tax-4"));
    }

    #[test]
    fn test_cellular_detail_plant_guard_cell() {
        let world = TerrariumWorld::demo(42, false).expect("demo world should build");
        if world.plants.is_empty() {
            return;
        }
        let query = super::InspectQuery {
            kind: "plant".into(),
            index: Some(0),
            x: None,
            y: None,
            scale: Some("cellular".into()),
            tissue: Some("guard".into()),
            molecule: None,
            atom_index: None,
        };
        let detail = super::build_cellular_detail(&world, &query)
            .expect("plant guard cell detail");
        assert_eq!(detail.cell_type, "guard cell");
        let organelle_names: Vec<&str> = detail.organelles.iter().map(|o| o.name.as_str()).collect();
        assert!(organelle_names.contains(&"chloroplast"), "guard cell should have chloroplasts");
        assert!(organelle_names.contains(&"central_vacuole"), "guard cell should have vacuole for turgor");
        let gene_names: Vec<&str> = detail.active_genes.iter().map(|g| g.name.as_str()).collect();
        assert!(gene_names.contains(&"SLAC1"), "guard cell should express SLAC1 anion channel");
        assert!(gene_names.contains(&"KAT1"), "guard cell should express KAT1 K+ channel");
    }

    #[test]
    fn inspect_molecular_shows_derivation_chain() {
        let query = super::InspectQuery {
            kind: "plant".into(),
            index: Some(0),
            x: None,
            y: None,
            scale: Some("molecular".into()),
            tissue: None,
            molecule: Some("glucose".into()),
            atom_index: None,
        };
        let detail = super::build_molecular_detail(&query, None)
            .expect("glucose molecular detail");
        let chain = detail
            .derivation_chain
            .as_ref()
            .expect("glucose should have derivation chain");

        // Optical derivation: glucose has CPK colors and extinction
        let optical = chain.optical.as_ref().expect("glucose should have optical derivation");
        assert!(optical.cpk_rgb[0] >= 0.0 && optical.cpk_rgb[0] <= 1.0);
        assert!(optical.molar_extinction >= 0.0);

        // Rate derivation: glucose is involved in photosynthesis, respiration, fly_glycolysis
        assert!(
            !chain.rates.is_empty(),
            "glucose should have associated rate derivations"
        );
        let pathway_names: Vec<&str> = chain.rates.iter().map(|r| r.pathway.as_str()).collect();
        assert!(
            pathway_names.contains(&"photosynthesis"),
            "glucose should reference photosynthesis pathway"
        );
        assert!(
            pathway_names.contains(&"respiration"),
            "glucose should reference respiration pathway"
        );

        // Each rate should have valid derivation data
        for rate in &chain.rates {
            assert!(rate.bond_energy_ev > 0.0, "bond energy should be positive");
            assert!(
                rate.enzyme_efficiency > 0.0 && rate.enzyme_efficiency <= 1.0,
                "efficiency should be in (0,1]"
            );
            assert!(rate.vmax_25 > 0.0, "Vmax_25 should be positive");
            assert!(!rate.citation.is_empty(), "citation should not be empty");
        }
    }

    #[test]
    fn inspect_rate_derivation_for_ethylene() {
        let query = super::InspectQuery {
            kind: "plant".into(),
            index: Some(0),
            x: None,
            y: None,
            scale: Some("molecular".into()),
            tissue: None,
            molecule: Some("carbon_dioxide".into()),
            atom_index: None,
        };
        let detail = super::build_molecular_detail(&query, None)
            .expect("CO₂ molecular detail");
        let chain = detail
            .derivation_chain
            .as_ref()
            .expect("CO₂ should have derivation chain");

        // CO₂ is involved in both photosynthesis (consumed) and respiration (produced)
        let pathway_names: Vec<&str> = chain.rates.iter().map(|r| r.pathway.as_str()).collect();
        assert!(
            pathway_names.contains(&"photosynthesis"),
            "CO₂ should reference photosynthesis"
        );
        assert!(
            pathway_names.contains(&"respiration"),
            "CO₂ should reference respiration"
        );

        // Photosynthesis Vmax should be 100.0 (Farquhar+ 1980)
        let photo = chain.rates.iter().find(|r| r.pathway == "photosynthesis").unwrap();
        assert!((photo.vmax_25 - 100.0).abs() < 0.1);
        assert_eq!(photo.citation, "Farquhar+ 1980");
        assert_eq!(photo.bond_type, "C-O");

        // Optical properties should be present
        let optical = chain.optical.as_ref().expect("CO₂ should have optical derivation");
        assert!(optical.cpk_rgb[0] >= 0.0);
    }
}
