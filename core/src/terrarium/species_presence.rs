use std::collections::BTreeMap;

use crate::botany::species_profile_by_taxonomy;
use crate::drosophila::genome_stats as drosophila_genome;
use crate::soil_fauna::{earthworm_genome_stats, NematodeKind};

use super::TerrariumWorld;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TerrariumSpeciesDomain {
    Plant,
    Insect,
    Annelid,
    Nematode,
    Fungus,
    Bacterium,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TerrariumSpeciesAuthority {
    ExplicitSpecies,
    ReferenceSpecies,
    GuildReference,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TerrariumSpeciesPresence {
    pub common_name: String,
    pub scientific_name: String,
    pub domain: TerrariumSpeciesDomain,
    pub authority: TerrariumSpeciesAuthority,
    pub reference_genome_mb: Option<f32>,
    pub reference_gene_count: Option<usize>,
    pub reference_chromosome_count: Option<usize>,
    pub reference_neuron_count: Option<usize>,
    pub genetics_model: String,
    pub neural_model: String,
    pub evidence: String,
    pub presence_score: f32,
}

fn mean(field: &[f32]) -> f32 {
    if field.is_empty() {
        0.0
    } else {
        field.iter().sum::<f32>() / field.len() as f32
    }
}

fn plant_reference_genome(taxonomy_id: u32) -> (Option<f32>, Option<usize>) {
    match taxonomy_id {
        3702 => (Some(135.0), Some(27_655)),
        3750 => (Some(651.0), Some(40_624)),
        23211 => (Some(577.0), Some(43_419)),
        3760 => (Some(227.0), None),
        42229 => (Some(353.0), None),
        2711 => (Some(320.0), None),
        2708 => (Some(313.0), None),
        15368 => (Some(272.0), Some(31_029)),
        _ => (None, None),
    }
}

fn push_plant_species(
    entries: &mut Vec<TerrariumSpeciesPresence>,
    taxonomy_id: u32,
    count: usize,
    total_plants: usize,
) {
    let Some(profile) = species_profile_by_taxonomy(taxonomy_id) else {
        return;
    };
    let (reference_genome_mb, reference_gene_count) = plant_reference_genome(taxonomy_id);
    entries.push(TerrariumSpeciesPresence {
        common_name: profile.common_name.to_string(),
        scientific_name: profile.scientific_name.to_string(),
        domain: TerrariumSpeciesDomain::Plant,
        authority: TerrariumSpeciesAuthority::ExplicitSpecies,
        reference_genome_mb,
        reference_gene_count,
        reference_chromosome_count: Some(profile.symbolic_chromosomes as usize),
        reference_neuron_count: None,
        genetics_model: "species-resolved morphology/physiology with taxon-specific genome scaffold".to_string(),
        neural_model: "none".to_string(),
        evidence: format!("{count} plants live in world state"),
        presence_score: if total_plants == 0 {
            0.0
        } else {
            count as f32 / total_plants as f32
        },
    });
}

fn push_soil_reference(
    entries: &mut Vec<TerrariumSpeciesPresence>,
    common_name: &str,
    scientific_name: &str,
    domain: TerrariumSpeciesDomain,
    authority: TerrariumSpeciesAuthority,
    evidence: String,
    presence_score: f32,
    genetics_model: &str,
    neural_model: &str,
    reference_genome_mb: Option<f32>,
    reference_gene_count: Option<usize>,
    reference_chromosome_count: Option<usize>,
    reference_neuron_count: Option<usize>,
) {
    entries.push(TerrariumSpeciesPresence {
        common_name: common_name.to_string(),
        scientific_name: scientific_name.to_string(),
        domain,
        authority,
        reference_genome_mb,
        reference_gene_count,
        reference_chromosome_count,
        reference_neuron_count,
        genetics_model: genetics_model.to_string(),
        neural_model: neural_model.to_string(),
        evidence,
        presence_score,
    });
}

impl TerrariumWorld {
    pub fn species_presence(&self) -> Vec<TerrariumSpeciesPresence> {
        let mut entries = Vec::new();

        let mut plant_counts: BTreeMap<u32, usize> = BTreeMap::new();
        for plant in &self.plants {
            *plant_counts.entry(plant.genome.taxonomy_id).or_insert(0) += 1;
        }
        let total_plants = self.plants.len();
        for (taxonomy_id, count) in plant_counts {
            push_plant_species(&mut entries, taxonomy_id, count, total_plants);
        }

        let fly_census = self.fly_pop.stage_census();
        let total_fly_population = fly_census.total_individuals() as usize;
        if total_fly_population > 0 || !self.flies.is_empty() {
            push_soil_reference(
                &mut entries,
                "Fruit fly",
                "Drosophila melanogaster",
                TerrariumSpeciesDomain::Insect,
                TerrariumSpeciesAuthority::ExplicitSpecies,
                format!(
                    "{} active agents | {} total lifecycle individuals",
                    self.flies.len(),
                    total_fly_population
                ),
                total_fly_population.max(self.flies.len()) as f32,
                "explicit fly genome stats + metabolism + body state",
                "explicit Drosophila neural model drives behavior",
                Some(drosophila_genome::GENOME_MB),
                Some(drosophila_genome::PROTEIN_GENES),
                Some(drosophila_genome::CHROMOSOME_COUNT),
                Some(drosophila_genome::FULL_NEURON_COUNT),
            );
        }

        let earthworm_density = mean(&self.earthworm_population.population_density);
        if earthworm_density > 0.01 {
            let hotspots = self
                .earthworm_population
                .population_density
                .iter()
                .filter(|&&density| density > 0.5)
                .count();
            push_soil_reference(
                &mut entries,
                "Common earthworm",
                "Lumbricus terrestris",
                TerrariumSpeciesDomain::Annelid,
                TerrariumSpeciesAuthority::ReferenceSpecies,
                format!("{hotspots} density hotspots | mean {:.2} ind./g soil", earthworm_density),
                earthworm_density,
                "species-specific earthworm population field + reference genome constants",
                "segmental ganglia and ventral nerve cord exist biologically; explicit neural sim pending",
                Some(earthworm_genome_stats::GENOME_MB),
                Some(earthworm_genome_stats::PROTEIN_GENES),
                Some(earthworm_genome_stats::CHROMOSOME_COUNT),
                None,
            );
        }

        for guild in &self.nematode_guilds {
            let mean_density = mean(&guild.population_density);
            if mean_density <= 0.05 {
                continue;
            }
            let (common_name, scientific_name) = match guild.kind {
                NematodeKind::BacterialFeeder => (
                    "Bacterivorous nematode guild",
                    "Caenorhabditis elegans (reference anatomy)",
                ),
                NematodeKind::FungalFeeder => (
                    "Fungivorous nematode guild",
                    "Aphelenchus avenae (reference guild)",
                ),
                NematodeKind::Omnivore => ("Omnivorous nematode guild", "Nematoda spp. omnivores"),
            };
            let neural_model = match guild.kind {
                NematodeKind::BacterialFeeder => {
                    "nematodes have ganglia/neurons; 302-neuron reference connectome exists in C. elegans"
                }
                _ => "nematodes have ganglia/neurons biologically; terrarium currently tracks guild density only",
            };
            push_soil_reference(
                &mut entries,
                common_name,
                scientific_name,
                TerrariumSpeciesDomain::Nematode,
                TerrariumSpeciesAuthority::GuildReference,
                format!("mean {:.2} ind./g soil", mean_density),
                mean_density,
                "guild-density soil fauna model, not species-resolved individuals",
                neural_model,
                None,
                None,
                None,
                if matches!(guild.kind, NematodeKind::BacterialFeeder) {
                    Some(302)
                } else {
                    None
                },
            );
        }

        let mean_symbiont = mean(&self.symbiont_biomass);
        if mean_symbiont > 0.002 {
            push_soil_reference(
                &mut entries,
                "Arbuscular mycorrhiza",
                "Rhizophagus irregularis (reference guild)",
                TerrariumSpeciesDomain::Fungus,
                TerrariumSpeciesAuthority::GuildReference,
                format!("mean symbiont biomass {:.3}", mean_symbiont),
                mean_symbiont,
                "symbiont biomass field calibrated to AM-fungal behavior",
                "none",
                None,
                None,
                None,
                None,
            );
        }

        let mean_nitrifier = mean(&self.nitrifier_biomass);
        if mean_nitrifier > 0.002 {
            push_soil_reference(
                &mut entries,
                "Ammonia-oxidizing bacteria",
                "Nitrosomonas europaea (reference guild)",
                TerrariumSpeciesDomain::Bacterium,
                TerrariumSpeciesAuthority::GuildReference,
                format!("mean nitrifier biomass {:.3}", mean_nitrifier),
                mean_nitrifier,
                "nitrifier guild field coupled to NH4+/O2 chemistry",
                "none",
                None,
                None,
                Some(1),
                None,
            );
        }

        let mean_denitrifier = mean(&self.denitrifier_biomass);
        if mean_denitrifier > 0.002 {
            push_soil_reference(
                &mut entries,
                "Denitrifying bacteria",
                "Paracoccus denitrificans (reference guild)",
                TerrariumSpeciesDomain::Bacterium,
                TerrariumSpeciesAuthority::GuildReference,
                format!("mean denitrifier biomass {:.3}", mean_denitrifier),
                mean_denitrifier,
                "denitrifier guild field coupled to NO3-/redox chemistry",
                "none",
                None,
                None,
                Some(2),
                None,
            );
        }

        entries.sort_by(|a, b| {
            b.presence_score
                .partial_cmp(&a.presence_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.common_name.cmp(&b.common_name))
        });
        entries
    }
}
