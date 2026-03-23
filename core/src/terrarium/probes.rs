//! Terrarium Probes: Atomistic MD probe management.

use super::*;

impl TerrariumWorld {
    /// Spawn an MD probe from an `EmbeddedMolecule` at the given grid cell.
    pub fn spawn_probe(
        &mut self,
        mol: &crate::atomistic_chemistry::EmbeddedMolecule,
        grid_x: usize,
        grid_y: usize,
        footprint_radius: usize,
    ) -> Result<u32, String> {
        let w = self.config.width;
        let h = self.config.height;
        if grid_x >= w || grid_y >= h {
            return Err(format!(
                "probe center ({grid_x},{grid_y}) out of bounds ({w}x{h})"
            ));
        }
        let id = self.next_probe_id;
        self.next_probe_id += 1;

        // Claim ownership cells.
        let r = footprint_radius;
        let x_lo = grid_x.saturating_sub(r);
        let x_hi = (grid_x + r + 1).min(w);
        let y_lo = grid_y.saturating_sub(r);
        let y_hi = (grid_y + r + 1).min(h);
        for cy in y_lo..y_hi {
            for cx in x_lo..x_hi {
                self.claim_ownership(
                    cx,
                    cy,
                    SoilOwnershipClass::AtomisticProbeRegion { probe_id: id },
                    1.0,
                );
            }
        }

        let probe =
            AtomisticProbe::from_embedded_molecule(id, mol, grid_x, grid_y, footprint_radius);
        self.atomistic_probes.push(probe);
        Ok(id)
    }

    /// Remove a probe by id. Releases its ownership cells.
    pub fn remove_probe(&mut self, probe_id: u32) -> bool {
        let idx = self.atomistic_probes.iter().position(|p| p.id == probe_id);
        if let Some(i) = idx {
            // Release ownership cells claimed by this probe.
            for cell in &mut self.ownership {
                if matches!(cell.owner, SoilOwnershipClass::AtomisticProbeRegion { probe_id: pid } if pid == probe_id)
                {
                    *cell = SoilOwnershipCell::default();
                }
            }
            self.atomistic_probes.swap_remove(i);
            true
        } else {
            false
        }
    }

    /// Step all atomistic probes. Each probe advances its MD engine.
    pub fn step_atomistic_probes(&mut self) {
        const MD_STEPS_PER_FRAME: usize = 10;
        for probe in &mut self.atomistic_probes {
            probe.step(MD_STEPS_PER_FRAME);
        }
    }

    pub fn probe_count(&self) -> usize {
        self.atomistic_probes.len()
    }

    pub fn probes(&self) -> &[AtomisticProbe] {
        &self.atomistic_probes
    }

    pub fn probes_mut(&mut self) -> &mut Vec<AtomisticProbe> {
        &mut self.atomistic_probes
    }
}
