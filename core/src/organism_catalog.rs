//! Organism Catalog - Unified interface for all simulated organisms.
//!
//! This module provides a common catalog and trait system for accessing
//! organism data across the simulation, including molecular/DNA fidelity,
//! genome statistics, and body anatomy.
//!
//! # Organisms Included
//! - C. elegans (nematode)
//! - Drosophila melanogaster (fruit fly)
//! - Ant (Formica species)
//! - Earthworm (Lumbricus terrestris)
//! - Nematode soil fauna (bacterial/fungal feeders)

use std::collections::HashMap;

// ============================================================================
// Organism Type Enumeration
// ============================================================================

/// All supported organism types in the simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OrganismType {
    /// Nematode: C. elegans (model organism).
    Celegans,
    /// Fruit fly: Drosophila melanogaster.
    Drosophila,
    /// Ant: Formica species (worker caste).
    AntWorker,
    /// Ant: Formica species (queen caste).
    AntQueen,
    /// Earthworm: Lumbricus terrestris.
    Earthworm,
    /// Soil nematode: bacterial feeder.
    NematodeBacterial,
    /// Soil nematode: fungal feeder.
    NematodeFungal,
    /// Soil nematode: plant parasite.
    NematodePlant,
}

impl OrganismType {
    /// Get the common name of the organism.
    pub fn common_name(&self) -> &'static str {
        match self {
            Self::Celegans => "C. elegans",
            Self::Drosophila => "Fruit fly",
            Self::AntWorker => "Ant (worker)",
            Self::AntQueen => "Ant (queen)",
            Self::Earthworm => "Earthworm",
            Self::NematodeBacterial => "Nematode (bacterial feeder)",
            Self::NematodeFungal => "Nematode (fungal feeder)",
            Self::NematodePlant => "Nematode (plant parasite)",
        }
    }

    /// Get the scientific name of the organism.
    pub fn scientific_name(&self) -> &'static str {
        match self {
            Self::Celegans => "Caenorhabditis elegans",
            Self::Drosophila => "Drosophila melanogaster",
            Self::AntWorker | Self::AntQueen => "Formica spp.",
            Self::Earthworm => "Lumbricus terrestris",
            Self::NematodeBacterial | Self::NematodeFungal | Self::NematodePlant => {
                "Caenorhabditis spp."
            }
        }
    }

    /// Get the organism category.
    pub fn category(&self) -> OrganismCategory {
        match self {
            Self::Celegans
            | Self::NematodeBacterial
            | Self::NematodeFungal
            | Self::NematodePlant => OrganismCategory::Nematode,
            Self::Drosophila => OrganismCategory::Insect,
            Self::AntWorker | Self::AntQueen => OrganismCategory::Insect,
            Self::Earthworm => OrganismCategory::Annelid,
        }
    }

    /// Check if this organism has neural simulation.
    pub fn has_neural_sim(&self) -> bool {
        matches!(
            self,
            Self::Celegans | Self::Drosophila | Self::AntWorker | Self::AntQueen
        )
    }

    /// Check if this organism has full connectome data.
    pub fn has_connectome(&self) -> bool {
        matches!(self, Self::Celegans)
    }
}

/// Broad organism category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OrganismCategory {
    /// Roundworms (phylum Nematoda).
    Nematode,
    /// Insects (class Insecta).
    Insect,
    /// Segmented worms (phylum Annelida).
    Annelid,
}

// ============================================================================
// Genome Statistics Trait
// ============================================================================

/// Genome statistics for an organism.
#[derive(Clone, Debug, Default)]
pub struct GenomeStats {
    /// Total genome size in megabases.
    pub genome_mb: f32,
    /// Number of protein-coding genes.
    pub protein_genes: usize,
    /// Number of chromosomes.
    pub chromosome_count: usize,
    /// GC content percentage.
    pub gc_content: f32,
    /// Repeat content percentage (if known).
    pub repeat_content: Option<f32>,
}

/// Trait for organisms with genome data.
pub trait HasGenome {
    /// Get genome statistics.
    fn genome_stats(&self) -> GenomeStats;
    /// Get chromosome lengths in Mb.
    fn chromosome_lengths(&self) -> Vec<f32>;
}

impl GenomeStats {
    /// C. elegans genome statistics.
    pub fn celegans() -> Self {
        Self {
            genome_mb: 100.0,
            protein_genes: 20_470,
            chromosome_count: 6,
            gc_content: 35.0,
            repeat_content: Some(12.0),
        }
    }

    /// Drosophila genome statistics.
    pub fn drosophila() -> Self {
        Self {
            genome_mb: 180.0,
            protein_genes: 13_600,
            chromosome_count: 4,
            gc_content: 41.5,
            repeat_content: Some(20.0),
        }
    }

    /// Ant genome statistics (Formica).
    pub fn ant() -> Self {
        Self {
            genome_mb: 280.0,
            protein_genes: 18_000,
            chromosome_count: 26,
            gc_content: 38.0,
            repeat_content: Some(25.0),
        }
    }

    /// Earthworm genome statistics.
    pub fn earthworm() -> Self {
        Self {
            genome_mb: 1060.0,
            protein_genes: 32_000,
            chromosome_count: 36,
            gc_content: 37.0,
            repeat_content: Some(58.0),
        }
    }
}

// ============================================================================
// Neural Statistics
// ============================================================================

/// Neural system statistics for an organism.
#[derive(Clone, Debug, Default)]
pub struct NeuralStats {
    /// Total neuron count.
    pub neuron_count: usize,
    /// Muscle cell count.
    pub muscle_count: Option<usize>,
    /// Brain regions (name, fraction of neurons).
    pub brain_regions: Vec<(&'static str, f32)>,
    /// Primary neurotransmitters.
    pub neurotransmitters: Vec<&'static str>,
}

impl NeuralStats {
    /// C. elegans neural statistics.
    pub fn celegans() -> Self {
        Self {
            neuron_count: 302,
            muscle_count: Some(95),
            brain_regions: vec![("sensory", 0.20), ("interneurons", 0.50), ("motor", 0.30)],
            neurotransmitters: vec!["ACh", "GABA", "Glutamate", "Serotonin", "Dopamine"],
        }
    }

    /// Drosophila neural statistics.
    pub fn drosophila() -> Self {
        Self {
            neuron_count: 139_000,
            muscle_count: None,
            brain_regions: vec![
                ("AL", 0.05),    // Antennal Lobe
                ("MB_KC", 0.10), // Mushroom Body Kenyon Cells
                ("CX", 0.06),    // Central Complex
                ("OL", 0.33),    // Optic Lobes
                ("VNC", 0.15),   // Ventral Nerve Cord
                ("other", 0.31),
            ],
            neurotransmitters: vec![
                "ACh",
                "GABA",
                "Glutamate",
                "Dopamine",
                "Serotonin",
                "Octopamine",
            ],
        }
    }

    /// Ant neural statistics.
    pub fn ant_worker() -> Self {
        Self {
            neuron_count: 250_000,
            muscle_count: None,
            brain_regions: vec![
                ("mushroom_body", 0.20),
                ("antennal_lobe", 0.10),
                ("optic_lobe", 0.35),
                ("central_complex", 0.05),
                ("subesophageal", 0.10),
                ("other", 0.20),
            ],
            neurotransmitters: vec!["ACh", "GABA", "Dopamine", "Octopamine", "Serotonin"],
        }
    }
}

// ============================================================================
// Body Anatomy
// ============================================================================

/// Body segment description.
#[derive(Clone, Debug)]
pub struct BodySegment {
    /// Segment name.
    pub name: String,
    /// Segment count (if multiple).
    pub count: usize,
    /// Key features.
    pub features: Vec<String>,
}

/// Body anatomy summary for an organism.
#[derive(Clone, Debug, Default)]
pub struct BodyAnatomy {
    /// Total length in mm.
    pub length_mm: f32,
    /// Diameter/width in mm.
    pub diameter_mm: f32,
    /// Number of segments (if segmented).
    pub segment_count: Option<usize>,
    /// Body segments.
    pub segments: Vec<BodySegment>,
    /// Locomotion type.
    pub locomotion: LocomotionType,
}

/// Locomotion mechanism.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LocomotionType {
    /// No active locomotion.
    #[default]
    None,
    /// Peristaltic crawling (earthworm, nematode).
    Peristaltic,
    /// Walking on legs.
    Walking,
    /// Flight (insects).
    Flight,
    /// Swimming.
    Swimming,
    /// Combined walking/flight.
    WalkingAndFlight,
}

impl BodyAnatomy {
    /// C. elegans body anatomy.
    pub fn celegans() -> Self {
        Self {
            length_mm: 1.0,
            diameter_mm: 0.08,
            segment_count: None, // Not truly segmented
            segments: vec![],
            locomotion: LocomotionType::Peristaltic,
        }
    }

    /// Drosophila body anatomy.
    pub fn drosophila() -> Self {
        Self {
            length_mm: 3.0,
            diameter_mm: 0.5,
            segment_count: Some(3),
            segments: vec![
                BodySegment {
                    name: "Head".into(),
                    count: 1,
                    features: vec!["800 ommatidia/eye".into(), "6 antennal segments".into()],
                },
                BodySegment {
                    name: "Thorax".into(),
                    count: 1,
                    features: vec!["3 leg pairs".into(), "2 wings".into(), "2 halteres".into()],
                },
                BodySegment {
                    name: "Abdomen".into(),
                    count: 1,
                    features: vec!["6 visible segments".into()],
                },
            ],
            locomotion: LocomotionType::WalkingAndFlight,
        }
    }

    /// Earthworm body anatomy.
    pub fn earthworm() -> Self {
        Self {
            length_mm: 200.0,
            diameter_mm: 6.0,
            segment_count: Some(150),
            segments: vec![
                BodySegment {
                    name: "Prostomium".into(),
                    count: 1,
                    features: vec!["Mouth".into()],
                },
                BodySegment {
                    name: "Clitellum".into(),
                    count: 6,
                    features: vec!["Reproductive".into()],
                },
                BodySegment {
                    name: "Body segments".into(),
                    count: 143,
                    features: vec!["8 setae each".into(), "nephridia".into()],
                },
            ],
            locomotion: LocomotionType::Peristaltic,
        }
    }
}

// ============================================================================
// Gene Information
// ============================================================================

/// Gene function category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeneCategory {
    /// Ion channel.
    IonChannel,
    /// Neurotransmitter system.
    Neurotransmitter,
    /// Learning and memory.
    LearningMemory,
    /// Circadian rhythm.
    Circadian,
    /// Digestive enzyme.
    Digestive,
    /// Stress response.
    StressResponse,
    /// Development.
    Development,
    /// Structural.
    Structural,
    /// Other.
    Other,
}

/// Gene information.
#[derive(Clone, Debug)]
pub struct GeneInfo {
    /// Gene symbol.
    pub symbol: String,
    /// Full name.
    pub full_name: String,
    /// Function category.
    pub category: GeneCategory,
    /// Chromosome (if known).
    pub chromosome: Option<u8>,
    /// Essential gene.
    pub essential: bool,
}

// ============================================================================
// Organism Catalog
// ============================================================================

/// Catalog of all organisms with their molecular data.
#[derive(Clone, Debug, Default)]
pub struct OrganismCatalog {
    /// Organism genome stats.
    genomes: HashMap<OrganismType, GenomeStats>,
    /// Organism neural stats.
    neural: HashMap<OrganismType, NeuralStats>,
    /// Organism body anatomy.
    anatomy: HashMap<OrganismType, BodyAnatomy>,
}

impl OrganismCatalog {
    /// Create a new catalog populated with all known organisms.
    pub fn new() -> Self {
        let mut catalog = Self {
            genomes: HashMap::new(),
            neural: HashMap::new(),
            anatomy: HashMap::new(),
        };

        // C. elegans
        catalog
            .genomes
            .insert(OrganismType::Celegans, GenomeStats::celegans());
        catalog
            .neural
            .insert(OrganismType::Celegans, NeuralStats::celegans());
        catalog
            .anatomy
            .insert(OrganismType::Celegans, BodyAnatomy::celegans());

        // Drosophila
        catalog
            .genomes
            .insert(OrganismType::Drosophila, GenomeStats::drosophila());
        catalog
            .neural
            .insert(OrganismType::Drosophila, NeuralStats::drosophila());
        catalog
            .anatomy
            .insert(OrganismType::Drosophila, BodyAnatomy::drosophila());

        // Ant
        catalog
            .genomes
            .insert(OrganismType::AntWorker, GenomeStats::ant());
        catalog
            .genomes
            .insert(OrganismType::AntQueen, GenomeStats::ant());
        catalog
            .neural
            .insert(OrganismType::AntWorker, NeuralStats::ant_worker());

        // Earthworm
        catalog
            .genomes
            .insert(OrganismType::Earthworm, GenomeStats::earthworm());
        catalog
            .anatomy
            .insert(OrganismType::Earthworm, BodyAnatomy::earthworm());

        // Soil nematodes (use C. elegans reference)
        for nt in [
            OrganismType::NematodeBacterial,
            OrganismType::NematodeFungal,
            OrganismType::NematodePlant,
        ] {
            catalog.genomes.insert(nt, GenomeStats::celegans());
        }

        catalog
    }

    /// Get genome stats for an organism.
    pub fn genome(&self, org: OrganismType) -> Option<&GenomeStats> {
        self.genomes.get(&org)
    }

    /// Get neural stats for an organism.
    pub fn neural(&self, org: OrganismType) -> Option<&NeuralStats> {
        self.neural.get(&org)
    }

    /// Get body anatomy for an organism.
    pub fn anatomy(&self, org: OrganismType) -> Option<&BodyAnatomy> {
        self.anatomy.get(&org)
    }

    /// List all organisms with neural simulation.
    pub fn neural_organisms(&self) -> Vec<OrganismType> {
        self.neural.keys().copied().collect()
    }

    /// Get summary string for an organism.
    pub fn summary(&self, org: OrganismType) -> String {
        let mut lines = vec![
            format!("{} ({})", org.common_name(), org.scientific_name()),
            format!("Category: {:?}", org.category()),
        ];

        if let Some(g) = self.genome(org) {
            lines.push(format!(
                "Genome: {:.0} Mb, {} genes, {} chromosomes",
                g.genome_mb, g.protein_genes, g.chromosome_count
            ));
        }

        if let Some(n) = self.neural(org) {
            lines.push(format!("Neurons: {}", n.neuron_count));
        }

        if let Some(a) = self.anatomy(org) {
            lines.push(format!(
                "Body: {:.1} mm, {:?} locomotion",
                a.length_mm, a.locomotion
            ));
        }

        lines.join("\n")
    }

    /// Generate a comparison table of all organisms.
    pub fn comparison_table(&self) -> String {
        let mut rows = vec!["| Organism | Genome (Mb) | Genes | Neurons |".to_string()];
        rows.push("|----------|-------------|-------|---------|".to_string());

        let organisms = [
            (OrganismType::Celegans, "C. elegans"),
            (OrganismType::Drosophila, "Drosophila"),
            (OrganismType::AntWorker, "Ant"),
            (OrganismType::Earthworm, "Earthworm"),
        ];

        for (org, name) in organisms {
            let genome = self
                .genome(org)
                .map(|g| format!("{:.0}", g.genome_mb))
                .unwrap_or("-".into());
            let genes = self
                .genome(org)
                .map(|g| format!("{}", g.protein_genes))
                .unwrap_or("-".into());
            let neurons = self
                .neural(org)
                .map(|n| format!("{}", n.neuron_count))
                .unwrap_or("-".into());
            rows.push(format!(
                "| {} | {} | {} | {} |",
                name, genome, genes, neurons
            ));
        }

        rows.join("\n")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_creation() {
        let catalog = OrganismCatalog::new();
        assert!(
            catalog.genomes.len() >= 5,
            "At least 5 organisms with genomes"
        );
    }

    #[test]
    fn test_celegans_genome() {
        let stats = GenomeStats::celegans();
        assert!((stats.genome_mb - 100.0).abs() < 10.0, "Genome ~100Mb");
        assert_eq!(stats.chromosome_count, 6, "6 chromosomes");
        assert!(stats.protein_genes >= 20_000, "At least 20,000 genes");
    }

    #[test]
    fn test_drosophila_neural() {
        let stats = NeuralStats::drosophila();
        assert_eq!(stats.neuron_count, 139_000, "139k neurons from FlyWire");
        assert!(stats.brain_regions.len() >= 5, "At least 5 brain regions");
    }

    #[test]
    fn test_organism_type_names() {
        assert_eq!(OrganismType::Celegans.common_name(), "C. elegans");
        assert_eq!(
            OrganismType::Celegans.scientific_name(),
            "Caenorhabditis elegans"
        );
    }

    #[test]
    fn test_organism_categories() {
        assert_eq!(
            OrganismType::Celegans.category(),
            OrganismCategory::Nematode
        );
        assert_eq!(
            OrganismType::Drosophila.category(),
            OrganismCategory::Insect
        );
        assert_eq!(
            OrganismType::Earthworm.category(),
            OrganismCategory::Annelid
        );
    }

    #[test]
    fn test_neural_sim_flag() {
        assert!(OrganismType::Celegans.has_neural_sim());
        assert!(OrganismType::Drosophila.has_neural_sim());
        assert!(!OrganismType::Earthworm.has_neural_sim());
    }

    #[test]
    fn test_catalog_summary() {
        let catalog = OrganismCatalog::new();
        let summary = catalog.summary(OrganismType::Celegans);
        assert!(summary.contains("C. elegans"));
        assert!(summary.contains("100"));
        assert!(summary.contains("302"));
    }

    #[test]
    fn test_comparison_table() {
        let catalog = OrganismCatalog::new();
        let table = catalog.comparison_table();
        assert!(table.contains("C. elegans"));
        assert!(table.contains("Drosophila"));
        assert!(table.contains("Earthworm"));
    }

    #[test]
    fn test_body_anatomy() {
        let celegans = BodyAnatomy::celegans();
        assert_eq!(celegans.locomotion, LocomotionType::Peristaltic);

        let drosophila = BodyAnatomy::drosophila();
        assert_eq!(drosophila.locomotion, LocomotionType::WalkingAndFlight);
        assert_eq!(drosophila.segments.len(), 3, "Head, thorax, abdomen");
    }

    #[test]
    fn test_genome_size_ordering() {
        // Earthworm has the largest genome
        let ew = GenomeStats::earthworm();
        let ce = GenomeStats::celegans();
        assert!(
            ew.genome_mb > ce.genome_mb * 5.0,
            "Earthworm genome >5x C. elegans"
        );
    }
}
