use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use crate::phylogenetic_tracker::PhyloTraits;

use super::{TerrariumFruitPatch, TerrariumPlant, TerrariumPlantGenome, TerrariumSeed};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum TerrariumOrganismKind {
    Plant,
    Seed,
    Fruit,
    Fly,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrganismIdentity {
    pub organism_id: u64,
    pub kind: TerrariumOrganismKind,
    pub birth_time_s: f32,
    pub generation: u32,
    pub parent_organism_id: Option<u64>,
    pub co_parent_organism_id: Option<u64>,
    pub phylo_id: Option<u64>,
    pub display_name: Option<String>,
}

impl OrganismIdentity {
    pub fn synthetic(kind: TerrariumOrganismKind, organism_id: u64) -> Self {
        Self {
            organism_id,
            kind,
            birth_time_s: 0.0,
            generation: 0,
            parent_organism_id: None,
            co_parent_organism_id: None,
            phylo_id: None,
            display_name: None,
        }
    }

    pub fn preferred_name(&self, default_common_name: &str) -> String {
        self.display_name
            .clone()
            .unwrap_or_else(|| default_common_name.to_string())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrganismRegistryEntry {
    pub identity: OrganismIdentity,
    pub taxonomy_id: Option<u32>,
    pub common_name: String,
    pub scientific_name: String,
    pub death_time_s: Option<f32>,
}

impl OrganismRegistryEntry {
    pub fn preferred_name(&self) -> String {
        self.identity.preferred_name(&self.common_name)
    }
}

pub type OrganismRegistry = BTreeMap<u64, OrganismRegistryEntry>;

fn hash_f32(value: f32, hasher: &mut DefaultHasher) {
    value.to_bits().hash(hasher);
}

pub fn plant_genome_hash(genome: &TerrariumPlantGenome) -> u64 {
    let mut hasher = DefaultHasher::new();
    genome.species_id.hash(&mut hasher);
    genome.taxonomy_id.hash(&mut hasher);
    hash_f32(genome.max_height_mm, &mut hasher);
    hash_f32(genome.canopy_radius_mm, &mut hasher);
    hash_f32(genome.root_radius_mm, &mut hasher);
    hash_f32(genome.leaf_efficiency, &mut hasher);
    hash_f32(genome.root_uptake_efficiency, &mut hasher);
    hash_f32(genome.water_use_efficiency, &mut hasher);
    hash_f32(genome.volatile_scale, &mut hasher);
    hash_f32(genome.fruiting_threshold, &mut hasher);
    hash_f32(genome.litter_turnover, &mut hasher);
    hash_f32(genome.shade_tolerance, &mut hasher);
    hash_f32(genome.root_depth_bias, &mut hasher);
    hash_f32(genome.symbiosis_affinity, &mut hasher);
    hash_f32(genome.seed_mass, &mut hasher);
    hasher.finish()
}

pub fn plant_phylo_traits(plant: &TerrariumPlant) -> PhyloTraits {
    let reproductive_rate = (plant.genome.fruiting_threshold
        + plant.physiology.fruit_count() as f32 * 0.04)
        .clamp(0.0, 4.0);
    PhyloTraits {
        biomass: plant.physiology.total_biomass(),
        drought_tolerance: plant.genome.water_use_efficiency.clamp(0.0, 2.0) * 0.5,
        enzyme_efficacy: plant.genome.leaf_efficiency.clamp(0.0, 2.0),
        reproductive_rate,
        niche_width: ((plant.genome.shade_tolerance + plant.genome.symbiosis_affinity) * 0.5)
            .clamp(0.0, 2.0),
    }
}

pub fn seed_phylo_traits(seed: &TerrariumSeed) -> PhyloTraits {
    let reserve = seed.reserve_carbon.max(seed.genome.seed_mass * 0.5);
    PhyloTraits {
        biomass: reserve,
        drought_tolerance: (seed.genome.water_use_efficiency * 0.55
            + seed.microsite.surface_exposure * 0.25)
            .clamp(0.0, 2.0),
        enzyme_efficacy: (seed.cellular.energy_charge() + seed.genome.leaf_efficiency * 0.5)
            .clamp(0.0, 2.0),
        reproductive_rate: 0.0,
        niche_width: ((seed.genome.root_depth_bias + seed.genome.symbiosis_affinity) * 0.5)
            .clamp(0.0, 2.0),
    }
}

pub fn fruit_parent_organism_id(fruit: &TerrariumFruitPatch) -> Option<u64> {
    fruit.identity.parent_organism_id
}

pub fn plant_phylo_traits_from_genome(
    genome: &TerrariumPlantGenome,
    biomass_scale: f32,
) -> PhyloTraits {
    PhyloTraits {
        biomass: (genome.seed_mass * 3.0 + genome.max_height_mm * 0.02) * biomass_scale.max(0.2),
        drought_tolerance: genome.water_use_efficiency.clamp(0.0, 2.0) * 0.5,
        enzyme_efficacy: genome.leaf_efficiency.clamp(0.0, 2.0),
        reproductive_rate: genome.fruiting_threshold.clamp(0.0, 2.0),
        niche_width: ((genome.shade_tolerance + genome.symbiosis_affinity) * 0.5).clamp(0.0, 2.0),
    }
}

pub fn seed_phylo_traits_from_genome(
    genome: &TerrariumPlantGenome,
    reserve_carbon: f32,
    surface_exposure: f32,
) -> PhyloTraits {
    PhyloTraits {
        biomass: reserve_carbon.max(genome.seed_mass * 0.5),
        drought_tolerance: (genome.water_use_efficiency * 0.55 + surface_exposure * 0.25)
            .clamp(0.0, 2.0),
        enzyme_efficacy: genome.leaf_efficiency.clamp(0.0, 2.0),
        reproductive_rate: 0.0,
        niche_width: ((genome.root_depth_bias + genome.symbiosis_affinity) * 0.5).clamp(0.0, 2.0),
    }
}

fn registry_names(kind: TerrariumOrganismKind, taxonomy_id: Option<u32>) -> (String, String) {
    match (kind, taxonomy_id) {
        (TerrariumOrganismKind::Fly, _) => ("fruit fly".into(), "Drosophila melanogaster".into()),
        (_, Some(taxonomy_id)) => (
            crate::terrarium::plant_species::plant_common_name(taxonomy_id).to_string(),
            crate::terrarium::plant_species::plant_scientific_name(taxonomy_id).to_string(),
        ),
        _ => ("unnamed organism".into(), "unknown lineage".into()),
    }
}

impl super::TerrariumWorld {
    pub(crate) fn register_organism_identity(
        &mut self,
        kind: TerrariumOrganismKind,
        taxonomy_id: Option<u32>,
        parent_organism_id: Option<u64>,
        co_parent_organism_id: Option<u64>,
        phylo_parent_id: Option<u64>,
        genome_hash: Option<u64>,
        phylo_traits: Option<PhyloTraits>,
    ) -> OrganismIdentity {
        let organism_id = self.next_organism_id;
        self.next_organism_id += 1;
        let generation = parent_organism_id
            .and_then(|id| self.organism_registry.get(&id))
            .map(|entry| entry.identity.generation.saturating_add(1))
            .unwrap_or(0);
        let phylo_id = match (genome_hash, phylo_traits) {
            (Some(genome_hash), Some(phylo_traits)) => Some(self.organism_phylogeny.add_node(
                phylo_parent_id,
                generation,
                1.0,
                genome_hash,
                self.time_s,
                phylo_traits,
            )),
            _ => None,
        };
        let identity = OrganismIdentity {
            organism_id,
            kind,
            birth_time_s: self.time_s,
            generation,
            parent_organism_id,
            co_parent_organism_id,
            phylo_id,
            display_name: None,
        };
        let (common_name, scientific_name) = registry_names(kind, taxonomy_id);
        self.organism_registry.insert(
            organism_id,
            OrganismRegistryEntry {
                identity: identity.clone(),
                taxonomy_id,
                common_name,
                scientific_name,
                death_time_s: None,
            },
        );
        identity
    }

    pub(crate) fn promote_organism_identity_kind(
        &mut self,
        identity: &OrganismIdentity,
        kind: TerrariumOrganismKind,
        taxonomy_id: Option<u32>,
    ) -> OrganismIdentity {
        let mut promoted = identity.clone();
        promoted.kind = kind;
        if let Some(entry) = self.organism_registry.get_mut(&identity.organism_id) {
            entry.identity.kind = kind;
            entry.taxonomy_id = taxonomy_id;
            let (common_name, scientific_name) = registry_names(kind, taxonomy_id);
            entry.common_name = common_name;
            entry.scientific_name = scientific_name;
        }
        promoted
    }

    pub(crate) fn mark_organism_dead(&mut self, organism_id: u64) {
        if let Some(entry) = self.organism_registry.get_mut(&organism_id) {
            entry.death_time_s = Some(self.time_s);
            if let Some(phylo_id) = entry.identity.phylo_id {
                self.organism_phylogeny.mark_dead(phylo_id, self.time_s);
            }
        }
    }

    pub fn organism_registry_entry(&self, organism_id: u64) -> Option<&OrganismRegistryEntry> {
        self.organism_registry.get(&organism_id)
    }

    pub fn organism_lineage(&self, organism_id: u64) -> Vec<OrganismRegistryEntry> {
        let mut lineage = Vec::new();
        let mut current = Some(organism_id);
        while let Some(id) = current {
            let Some(entry) = self.organism_registry.get(&id) else {
                break;
            };
            lineage.push(entry.clone());
            current = entry.identity.parent_organism_id;
        }
        lineage.reverse();
        lineage
    }

    pub fn set_organism_name(&mut self, organism_id: u64, name: String) -> bool {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return false;
        }
        let mut updated = false;
        if let Some(entry) = self.organism_registry.get_mut(&organism_id) {
            entry.identity.display_name = Some(trimmed.to_string());
            updated = true;
        }
        for plant in &mut self.plants {
            if plant.identity.organism_id == organism_id {
                plant.identity.display_name = Some(trimmed.to_string());
            }
        }
        for fruit in &mut self.fruits {
            if fruit.identity.organism_id == organism_id {
                fruit.identity.display_name = Some(trimmed.to_string());
            }
        }
        for seed in &mut self.seeds {
            if seed.identity.organism_id == organism_id {
                seed.identity.display_name = Some(trimmed.to_string());
            }
        }
        for identity in &mut self.fly_identities {
            if identity.organism_id == organism_id {
                identity.display_name = Some(trimmed.to_string());
            }
        }
        updated
    }
}
