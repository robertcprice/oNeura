use super::*;

impl TerrariumWorld {
    /// Rebuild explicit microbe authority/activity fields from live cohorts.
    pub(super) fn rebuild_explicit_microbe_fields(&mut self) {
        let plane = self.config.width * self.config.height;
        self.explicit_microbe_authority.resize(plane, 0.0);
        self.explicit_microbe_activity.resize(plane, 0.0);
        self.explicit_microbe_authority.fill(0.0);
        self.explicit_microbe_activity.fill(0.0);
        for em in &self.explicit_microbes {
            let flat = em.y * self.config.width + em.x;
            if flat < plane {
                self.explicit_microbe_authority[flat] =
                    (self.explicit_microbe_authority[flat] + em.represented_cells * 0.002).min(1.0);
                self.explicit_microbe_activity[flat] =
                    (self.explicit_microbe_activity[flat] + em.smoothed_energy).min(2.0);
            }
        }
    }

    /// Number of whole-cell steps per ecology step for a given simulator.
    pub(crate) fn explicit_microbe_step_count(
        _simulator: &crate::whole_cell::WholeCellSimulator,
        eco_dt: f32,
    ) -> usize {
        // Default: 1ms per WCS step, eco_dt in seconds
        ((eco_dt * 1000.0) as usize).max(1).min(50)
    }

    /// Build identity record for a new explicit microbe at the given cell.
    pub(super) fn explicit_microbe_identity_at(
        &self,
        flat: usize,
    ) -> TerrariumExplicitMicrobeIdentity {
        let _ = flat;
        TerrariumExplicitMicrobeIdentity::default()
    }

    /// Compute growth signal from whole-cell snapshot.
    pub(super) fn explicit_microbe_growth_signal(
        &self,
        _snapshot: &WholeCellSnapshot,
        _represented_cells: f32,
    ) -> f32 {
        0.0
    }

    /// Recruit new explicit microbes from high-activity soil cells.
    pub(super) fn recruit_explicit_microbes_from_soil(&mut self) -> Result<(), String> {
        // Stub: recruitment logic requires full material inventory infrastructure.
        Ok(())
    }

    /// Build environment inputs for a whole-cell simulator from local soil chemistry.
    pub(super) fn explicit_microbe_environment_inputs(
        &self,
        _x: usize, _y: usize, _z: usize,
        _material_inputs: &WholeCellEnvironmentInputs,
    ) -> WholeCellEnvironmentInputs {
        WholeCellEnvironmentInputs::default()
    }

    /// Build material inventory for an explicit microbe's patch.
    pub(super) fn explicit_microbe_material_inventory(
        &self,
        _x: usize, _y: usize, _z: usize,
        _radius: usize,
    ) -> RegionalMaterialInventory {
        RegionalMaterialInventory::new(String::new())
    }

    /// Build material inventory target from surrounding patch.
    pub(super) fn explicit_microbe_material_inventory_target_from_patch(
        &self,
        _x: usize, _y: usize, _z: usize,
        _radius: usize,
        _represented_cells: f32,
    ) -> RegionalMaterialInventory {
        RegionalMaterialInventory::new(String::new())
    }

    /// Fraction of material exchange allowed by ownership authority.
    pub(super) fn explicit_microbe_material_exchange_fraction(authority: f32) -> f64 {
        (authority as f64).clamp(0.0, 1.0)
    }

    /// Normalize component shares with fallback.
    pub(super) fn normalize_component_shares(potentials: &[f64], fallback: &[f64]) -> Vec<f64> {
        let total: f64 = potentials.iter().sum();
        if total > 1e-12 {
            potentials.iter().map(|p| p / total).collect()
        } else {
            fallback.to_vec()
        }
    }

    /// Reserve-band helper for clamped substrate extraction/deposit.
    pub(super) fn reserve_band(
        value: f32,
        lo: f32,
        hi: f32,
        fraction: f64,
    ) -> f32 {
        let range = (hi - lo).max(0.0);
        let within = ((value - lo) / range.max(1e-9)).clamp(0.0, 1.0);
        within * range * fraction as f32
    }

    /// Sync a single species component between substrate and explicit microbe.
    pub(super) fn sync_owned_component_single_species(
        &mut self,
        _idx: usize,
        _species: crate::terrarium::TerrariumSpecies,
        _fraction: f64,
        _target_mm: f32,
    ) {
        // Stub: full substrate coupling needed
    }

    /// Sync amino acid pool between substrate and explicit microbe.
    pub(super) fn sync_owned_component_amino_pool(
        &mut self, _idx: usize, _fraction: f64, _target_mm: f32,
    ) {
        // Stub
    }

    /// Sync nucleotide pool between substrate and explicit microbe.
    pub(super) fn sync_owned_component_nucleotide_pool(
        &mut self, _idx: usize, _fraction: f64, _target_mm: f32,
    ) {
        // Stub
    }

    /// Sync oxygen pool between substrate and explicit microbe.
    pub(super) fn sync_owned_component_oxygen_pool(
        &mut self, _idx: usize, _fraction: f64, _target_mm: f32,
    ) {
        // Stub
    }

    /// Sync membrane precursor pool between substrate and explicit microbe.
    pub(super) fn sync_owned_component_membrane_pool(
        &mut self, _idx: usize, _fraction: f64, _target_mm: f32,
    ) {
        // Stub
    }

    /// Sync ATP pool between substrate and explicit microbe.
    pub(super) fn sync_owned_component_atp_pool(
        &mut self, _idx: usize, _fraction: f64, _target_mm: f32,
    ) {
        // Stub
    }

    /// Spill CO2 from explicit microbe to substrate.
    pub(super) fn spill_owned_component_carbon_dioxide_pool(
        &mut self, _idx: usize, _fraction: f64, _release: f32,
    ) {
        // Stub
    }

    /// Spill protons from explicit microbe to substrate.
    pub(super) fn spill_owned_component_proton_pool(
        &mut self, _idx: usize, _fraction: f64, _release: f32,
    ) {
        // Stub
    }

    /// Sync full material inventory for an explicit microbe with substrate.
    pub(super) fn sync_owned_explicit_microbe_material_inventory(
        &mut self,
        _idx: usize,
        _inventory: &RegionalMaterialInventory,
        _target: &RegionalMaterialInventory,
        _fraction: f64,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Sync explicit microbe material inventory (high-level).
    pub(super) fn sync_explicit_microbe_material_inventory(
        &mut self,
        _idx: usize,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Check available material in owned substrate region.
    pub(super) fn owned_inventory_available(
        &self,
        _x: usize, _y: usize, _z: usize,
        _radius: usize,
        _species: crate::terrarium::TerrariumSpecies,
    ) -> f32 {
        0.0
    }

    /// Apply material fluxes computed by whole-cell simulation.
    pub(super) fn apply_explicit_microbe_material_fluxes(
        &mut self,
        _idx: usize,
        _fluxes: &[(crate::terrarium::TerrariumSpecies, f32)],
    ) -> Result<(), String> {
        Ok(())
    }

    /// Compute material fluxes for a single explicit microbe.
    pub(super) fn compute_explicit_microbe_fluxes(
        &self,
        _idx: usize,
        _snapshot: &WholeCellSnapshot,
        _eco_dt: f32,
    ) -> Vec<(crate::terrarium::TerrariumSpecies, f32)> {
        Vec::new()
    }

    /// Step a single explicit microbe: run whole-cell sim + material exchange.
    pub(super) fn step_single_explicit_microbe(
        &mut self,
        _idx: usize,
        _eco_dt: f32,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Bridge explicit microbe state back to coarse packet representation.
    pub(super) fn bridge_explicit_to_coarse_packet(
        &mut self,
        _idx: usize,
    ) {
        // Stub: bridges explicit microbe metrics back to coarse guild fields
    }

    /// Step all explicit microbes (full iteration).
    pub(super) fn step_explicit_microbes(&mut self, _eco_dt: f32) -> Result<(), String> {
        if self.explicit_microbes.is_empty() {
            return Ok(());
        }
        // Stub: full explicit microbe lifecycle needs material inventory + WCS stepping
        self.rebuild_explicit_microbe_fields();
        Ok(())
    }

    /// Step explicit microbes incrementally (budget-limited).
    pub(super) fn step_explicit_microbes_incremental(&mut self, _eco_dt: f32) -> Result<(), String> {
        if self.explicit_microbes.is_empty() {
            return Ok(());
        }
        self.rebuild_explicit_microbe_fields();
        Ok(())
    }
}
