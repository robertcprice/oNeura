use super::*;
use std::collections::HashMap;

mod material_inventory_components_serde {
    use super::{MaterialInventoryEntry, MaterialInventoryKey};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S>(
        components: &HashMap<MaterialInventoryKey, MaterialInventoryEntry>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let entries: Vec<_> = components.iter().collect();
        entries.serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<HashMap<MaterialInventoryKey, MaterialInventoryEntry>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entries =
            Vec::<(MaterialInventoryKey, MaterialInventoryEntry)>::deserialize(deserializer)?;
        Ok(entries.into_iter().collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
struct MaterialInventoryKey {
    region: MaterialRegionKind,
    molecule_name: String,
    phase_kind: MaterialPhaseKind,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MaterialInventoryEntry {
    phase: MaterialPhaseDescriptor,
    amount_moles: f64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct RegionalMaterialInventory {
    pub name: String,
    pub regions: Vec<(MaterialRegionKind, f32)>,
    #[serde(with = "material_inventory_components_serde")]
    components: HashMap<MaterialInventoryKey, MaterialInventoryEntry>,
}

#[allow(dead_code)]
impl RegionalMaterialInventory {
    pub fn new_empty() -> Self {
        Self::default()
    }

    pub fn new(name: String) -> Self {
        Self {
            name,
            ..Self::default()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.components
            .values()
            .all(|entry| entry.amount_moles <= 1.0e-12)
    }

    pub fn total_amount_moles(&self) -> f64 {
        self.components
            .values()
            .map(|entry| entry.amount_moles)
            .sum()
    }

    pub fn estimate_whole_cell_environment_inputs(
        &self,
        regions: &[MaterialRegionKind],
    ) -> WholeCellEnvironmentInputs {
        let include_all = regions.is_empty();
        let includes = |region: MaterialRegionKind| include_all || regions.contains(&region);
        let aqueous = MaterialPhaseSelector::Kind(MaterialPhaseKind::Aqueous);
        let gas = MaterialPhaseSelector::Kind(MaterialPhaseKind::Gas);

        let glucose = if includes(MaterialRegionKind::PoreWater) {
            self.total_amount_for_component(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_glucose(),
                &aqueous,
            )
        } else {
            0.0
        };
        let oxygen = if includes(MaterialRegionKind::GasPhase) {
            self.total_amount_for_component(
                MaterialRegionKind::GasPhase,
                &MoleculeGraph::representative_oxygen_gas(),
                &gas,
            )
        } else {
            0.0
        };
        let amino_acids = if includes(MaterialRegionKind::PoreWater) {
            self.total_amount_for_component(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_amino_acid_pool(),
                &aqueous,
            )
        } else {
            0.0
        };
        let nucleotides = if includes(MaterialRegionKind::PoreWater) {
            self.total_amount_for_component(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_nucleotide_pool(),
                &aqueous,
            )
        } else {
            0.0
        };
        let ammonium = if includes(MaterialRegionKind::PoreWater) {
            self.total_amount_for_component(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_ammonium(),
                &aqueous,
            )
        } else {
            0.0
        };
        let nitrate = if includes(MaterialRegionKind::PoreWater) {
            self.total_amount_for_component(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_nitrate(),
                &aqueous,
            )
        } else {
            0.0
        };
        let atp = if includes(MaterialRegionKind::BiofilmMatrix) {
            self.total_amount_for_component(
                MaterialRegionKind::BiofilmMatrix,
                &MoleculeGraph::representative_atp(),
                &aqueous,
            )
        } else {
            0.0
        };
        let membrane_precursors = if includes(MaterialRegionKind::BiofilmMatrix) {
            self.total_amount_for_component(
                MaterialRegionKind::BiofilmMatrix,
                &MoleculeGraph::representative_membrane_precursor_pool(),
                &MaterialPhaseSelector::Kind(MaterialPhaseKind::Amorphous),
            )
        } else {
            0.0
        };
        let carbon_dioxide = if includes(MaterialRegionKind::GasPhase) {
            self.total_amount_for_component(
                MaterialRegionKind::GasPhase,
                &MoleculeGraph::representative_carbon_dioxide(),
                &gas,
            )
        } else {
            0.0
        };
        let proton = if includes(MaterialRegionKind::PoreWater) {
            self.total_amount_for_component(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_proton_pool(),
                &aqueous,
            )
        } else {
            0.0
        };

        WholeCellEnvironmentInputs {
            glucose_mm: clamp(glucose, 0.0, 8.0),
            oxygen_mm: clamp(oxygen, 0.0, 8.0),
            amino_acids_mm: clamp(amino_acids + ammonium * 0.18 + glucose * 0.05, 0.0, 6.0),
            nucleotides_mm: clamp(nucleotides + nitrate * 0.16 + atp * 0.04, 0.0, 6.0),
            membrane_precursors_mm: clamp(
                membrane_precursors + glucose * 0.10 + atp * 0.03,
                0.0,
                5.0,
            ),
            metabolic_load: clamp(0.22 + carbon_dioxide * 0.06 + proton * 0.26, 0.0, 2.2),
            temperature_c: 22.0,
            proton_concentration: clamp(1.0e-7 + proton * 1.0e-6, 1.0e-8, 1.0e-3),
        }
    }

    pub fn total_amount_for_component(
        &self,
        region: MaterialRegionKind,
        molecule: &MoleculeDescriptor,
        selector: &MaterialPhaseSelector,
    ) -> f32 {
        self.components
            .iter()
            .filter(|(key, entry)| {
                key.region == region
                    && key.molecule_name == molecule.name
                    && key.phase_kind == selector.kind
                    && entry.phase.fraction >= selector.min_fraction
            })
            .map(|(_, entry)| entry.amount_moles)
            .sum::<f64>() as f32
    }

    pub fn add_component(
        &mut self,
        region: MaterialRegionKind,
        molecule: MoleculeDescriptor,
        amount: f64,
        phase: MaterialPhaseDescriptor,
    ) {
        self.deposit_component(region, molecule, amount, phase);
    }

    pub fn deposit_component(
        &mut self,
        region: MaterialRegionKind,
        molecule: MoleculeDescriptor,
        amount: f64,
        phase: MaterialPhaseDescriptor,
    ) {
        if amount <= 1.0e-12 {
            return;
        }
        let key = Self::component_key(region, &molecule, phase.kind);
        let entry = self
            .components
            .entry(key)
            .or_insert(MaterialInventoryEntry {
                phase: phase.clone(),
                amount_moles: 0.0,
            });
        entry.amount_moles += amount.max(0.0);
        entry.phase = phase;
        self.refresh_region_totals();
    }

    pub fn set_component_amount(
        &mut self,
        region: MaterialRegionKind,
        molecule: MoleculeDescriptor,
        phase: MaterialPhaseDescriptor,
        amount: f64,
    ) -> Result<(), String> {
        let key = Self::component_key(region, &molecule, phase.kind);
        if amount <= 1.0e-12 {
            self.components.remove(&key);
        } else {
            self.components.insert(
                key,
                MaterialInventoryEntry {
                    phase,
                    amount_moles: amount.max(0.0),
                },
            );
        }
        self.refresh_region_totals();
        Ok(())
    }

    pub fn remove_component_amount(
        &mut self,
        region: MaterialRegionKind,
        molecule: &MoleculeDescriptor,
        amount: f64,
    ) -> Result<f64, String> {
        Ok(self.withdraw_component(region, molecule, amount))
    }

    pub fn relax_toward(
        &mut self,
        target: &RegionalMaterialInventory,
        relaxation: f64,
    ) -> Result<(), String> {
        let relaxation = relaxation.clamp(0.0, 1.0);
        let keys = self
            .components
            .keys()
            .cloned()
            .chain(target.components.keys().cloned())
            .collect::<Vec<_>>();
        for key in keys {
            let current = self
                .components
                .get(&key)
                .map(|entry| entry.amount_moles)
                .unwrap_or(0.0);
            let target_amount = target
                .components
                .get(&key)
                .map(|entry| entry.amount_moles)
                .unwrap_or(0.0);
            let next = (current + (target_amount - current) * relaxation).max(0.0);
            if next <= 1.0e-12 {
                self.components.remove(&key);
                continue;
            }
            let phase = target
                .components
                .get(&key)
                .map(|entry| entry.phase.clone())
                .or_else(|| self.components.get(&key).map(|entry| entry.phase.clone()))
                .unwrap_or_else(|| MaterialPhaseDescriptor::ambient(key.phase_kind));
            self.components.insert(
                key,
                MaterialInventoryEntry {
                    phase,
                    amount_moles: next,
                },
            );
        }
        self.refresh_region_totals();
        Ok(())
    }

    pub fn withdraw_component(
        &mut self,
        region: MaterialRegionKind,
        molecule: &MoleculeDescriptor,
        amount: f64,
    ) -> f64 {
        let target = amount.max(0.0);
        if target <= 1.0e-12 {
            return 0.0;
        }
        let mut removed = 0.0;
        let keys = self
            .components
            .keys()
            .filter(|key| key.region == region && key.molecule_name == molecule.name)
            .cloned()
            .collect::<Vec<_>>();
        for key in keys {
            if removed >= target {
                break;
            }
            let Some(entry) = self.components.get_mut(&key) else {
                continue;
            };
            let take = (target - removed).min(entry.amount_moles.max(0.0));
            entry.amount_moles -= take;
            removed += take;
        }
        self.components
            .retain(|_, entry| entry.amount_moles > 1.0e-12);
        self.refresh_region_totals();
        removed
    }

    pub fn scale_in_place(&mut self, factor: f64) {
        let factor = factor.max(0.0);
        for entry in self.components.values_mut() {
            entry.amount_moles = (entry.amount_moles * factor).max(0.0);
        }
        self.components
            .retain(|_, entry| entry.amount_moles > 1.0e-12);
        self.refresh_region_totals();
    }

    pub fn scaled(&self, factor: f64) -> Self {
        let mut copy = self.clone();
        copy.scale_in_place(factor);
        copy
    }

    fn component_key(
        region: MaterialRegionKind,
        molecule: &MoleculeDescriptor,
        phase_kind: MaterialPhaseKind,
    ) -> MaterialInventoryKey {
        MaterialInventoryKey {
            region,
            molecule_name: molecule.name.clone(),
            phase_kind,
        }
    }

    fn refresh_region_totals(&mut self) {
        let mut per_region: HashMap<MaterialRegionKind, f64> = HashMap::new();
        for (key, entry) in &self.components {
            *per_region.entry(key.region).or_default() += entry.amount_moles.max(0.0);
        }
        let mut regions = per_region
            .into_iter()
            .map(|(region, amount)| (region, amount as f32))
            .collect::<Vec<_>>();
        regions.sort_by_key(|(region, _)| *region as u8);
        self.regions = regions;
    }
}
