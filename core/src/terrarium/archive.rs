use std::path::Path;

use super::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TerrariumWorldArchive {
    pub config: TerrariumWorldConfig,
    pub time_s: f32,
    #[serde(default)]
    pub seed_provenance: crate::terrarium::TerrariumSeedProvenance,
    pub climate_driver: Option<crate::terrarium::climate_driver::TerrariumClimateDriver>,
    pub snapshot: TerrariumWorldSnapshot,
    pub organism_registry: crate::terrarium::organism_identity::OrganismRegistry,
    pub organism_phylogeny: crate::phylogenetic_tracker::PhyloTree,
}

impl TerrariumWorldArchive {
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let json = self
            .to_json_pretty()
            .map_err(|error| format!("serialize terrarium archive: {error}"))?;
        std::fs::write(path.as_ref(), json).map_err(|error| {
            format!(
                "write terrarium archive {}: {error}",
                path.as_ref().display()
            )
        })
    }

    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let json = std::fs::read_to_string(path.as_ref()).map_err(|error| {
            format!(
                "read terrarium archive {}: {error}",
                path.as_ref().display()
            )
        })?;
        Self::from_json_str(&json).map_err(|error| {
            format!(
                "parse terrarium archive {}: {error}",
                path.as_ref().display()
            )
        })
    }
}

impl TerrariumWorld {
    pub fn archive(&self) -> TerrariumWorldArchive {
        TerrariumWorldArchive {
            config: self.config.clone(),
            time_s: self.time_s,
            seed_provenance: self.seed_provenance.clone(),
            climate_driver: self.climate_driver.clone(),
            snapshot: self.snapshot(),
            organism_registry: self.organism_registry.clone(),
            organism_phylogeny: self.organism_phylogeny.clone(),
        }
    }

    pub fn save_archive<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        self.archive().save_to_path(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drosophila_population::{EggCluster, FlyEmbryoState, FlySex};
    use crate::terrarium::material_exchange::deposit_species_to_inventory;

    #[test]
    fn archive_round_trip_preserves_full_fly_embryos() {
        let mut world =
            TerrariumWorld::demo(42, false).expect("demo world should build for archive test");
        let cs = world.config.cell_size_mm.max(1.0e-3);
        let mut clutch_inventory = RegionalMaterialInventory::new("archive:egg-clutch".into());
        deposit_species_to_inventory(&mut clutch_inventory, TerrariumSpecies::Water, 0.8);
        deposit_species_to_inventory(&mut clutch_inventory, TerrariumSpecies::Glucose, 0.6);
        let mut embryo_inventory = RegionalMaterialInventory::new("archive:embryo".into());
        deposit_species_to_inventory(&mut embryo_inventory, TerrariumSpecies::Water, 0.09);
        deposit_species_to_inventory(&mut embryo_inventory, TerrariumSpecies::Glucose, 0.05);
        deposit_species_to_inventory(
            &mut embryo_inventory,
            TerrariumSpecies::NucleotidePool,
            0.03,
        );
        let mut clutch = EggCluster {
            position: (5.0 * cs, 4.0 * cs),
            count: 0,
            age_hours: 0.0,
            substrate_quality: 0.9,
            material_inventory: clutch_inventory,
            embryos: vec![FlyEmbryoState {
                id: 910,
                sex: FlySex::Female,
                offset_mm: (0.04, -0.03),
                age_hours: 4.0,
                viability: 0.81,
                material_inventory: embryo_inventory,
            }],
        };
        clutch.refresh_summary();
        world.fly_pop.egg_clusters.push(clutch);

        let archive = world.archive();
        let json = archive
            .to_json_pretty()
            .expect("archive should serialize with explicit embryo snapshots");
        let restored = TerrariumWorldArchive::from_json_str(&json)
            .expect("archive should deserialize with explicit embryo snapshots");

        assert_eq!(restored.snapshot.full_fly_embryos.len(), 1);
        let embryo = &restored.snapshot.full_fly_embryos[0];
        assert_eq!(embryo.embryo_id, 910);
        assert_eq!(embryo.cluster_index, 0);
        assert_eq!(embryo.sex, "Female");
        assert!(embryo.viability > 0.0);
        assert!(embryo.water > 0.0);
        assert!(embryo.glucose > 0.0);
        assert!(embryo.nucleotides > 0.0);
    }
}
