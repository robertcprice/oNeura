//! Pharmaceutical Lab — interactive molecule builder + MD simulation + drug discovery.
//!
//! Provides a "lab bench" where users can:
//! - Build molecules atom-by-atom or from SMILES
//! - Run molecular dynamics simulations
//! - Detect and visualize reactions
//! - Dock ligands to protein targets
//! - Compute ADMET profiles

pub mod library;
pub mod smiles;

use crate::atomistic_chemistry::{
    BondOrder, EmbeddedMolecule, MoleculeGraph, PeriodicElement, StructuralReactionTemplate,
};
use crate::drug_discovery::{self, ADMETProfile, BindingSite, DockingResult, DrugCandidate};
use crate::molecular_dynamics::{GPUMolecularDynamics, MDStats};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabConfig {
    pub box_size: [f32; 3],
    pub temperature_kelvin: f32,
    pub solvent: SolventModel,
}

impl Default for LabConfig {
    fn default() -> Self {
        Self {
            box_size: [80.0, 80.0, 80.0],
            temperature_kelvin: 300.0,
            solvent: SolventModel::Vacuum,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolventModel {
    Vacuum,
    ImplicitWater { dielectric: f32 },
}

// ---------------------------------------------------------------------------
// Lab molecule
// ---------------------------------------------------------------------------

pub type LabMoleculeId = u64;

/// A molecule on the lab bench with its graph, 3D positions, and MD state.
#[derive(Debug, Clone)]
pub struct LabMolecule {
    pub id: LabMoleculeId,
    pub name: String,
    pub graph: MoleculeGraph,
    pub positions: Vec<[f32; 3]>,
    pub velocities: Vec<[f32; 3]>,
    /// Offset of this molecule's atoms in the consolidated MD array.
    pub md_offset: usize,
    /// If true, this molecule is being edited and excluded from MD.
    pub locked: bool,
}

impl LabMolecule {
    fn atom_count(&self) -> usize {
        self.graph.atom_count()
    }

    fn molecular_formula(&self) -> String {
        let comp = self.graph.element_composition();
        let mut parts: Vec<(String, usize)> = comp
            .into_iter()
            .map(|(e, n)| (e.symbol().to_string(), n))
            .collect();
        // Hill system: C first, then H, then alphabetical
        parts.sort_by(|a, b| {
            let order = |s: &str| match s {
                "C" => 0,
                "H" => 1,
                _ => 2,
            };
            order(&a.0).cmp(&order(&b.0)).then(a.0.cmp(&b.0))
        });
        let mut formula = String::new();
        for (sym, count) in parts {
            formula.push_str(&sym);
            if count > 1 {
                formula.push_str(&count.to_string());
            }
        }
        formula
    }
}

// ---------------------------------------------------------------------------
// Reaction event
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionEvent {
    pub template_name: String,
    pub molecule_ids: Vec<LabMoleculeId>,
    pub atoms_involved: Vec<usize>,
    pub energy_change_kcal: f64,
    pub reaction_type: String,
}

// ---------------------------------------------------------------------------
// Docking session
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DockingSession {
    pub ligand_id: LabMoleculeId,
    pub target: BindingSite,
    pub result: Option<DockingResult>,
    pub admet: Option<ADMETProfile>,
}

// ---------------------------------------------------------------------------
// Serializable snapshot data (for WebSocket)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct PharmaLabSnapshot {
    pub molecules: Vec<PharmaLabMoleculeSnapshot>,
    pub md_stats: Option<MDStats>,
    pub temperature_kelvin: f32,
    pub box_size: [f32; 3],
    pub step_count: u64,
    pub md_running: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PharmaLabMoleculeSnapshot {
    pub id: u64,
    pub name: String,
    pub formula: String,
    pub molecular_weight: f32,
    pub atom_count: usize,
    pub bond_count: usize,
    pub locked: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PharmaLabFrame {
    pub atoms: Vec<PharmaLabAtomFrame>,
    pub bonds: Vec<PharmaLabBondFrame>,
    pub md_stats: MDStats,
    pub step_count: u64,
    pub reactions: Vec<ReactionEvent>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PharmaLabAtomFrame {
    pub position: [f32; 3],
    pub element: String,
    pub cpk_color: [u8; 3],
    pub radius: f32,
    pub molecule_id: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PharmaLabBondFrame {
    pub atom_a: usize,
    pub atom_b: usize,
    pub order: f32,
}

// ---------------------------------------------------------------------------
// PharmaLab — the main state machine
// ---------------------------------------------------------------------------

pub struct PharmaLab {
    pub config: LabConfig,
    pub molecules: Vec<LabMolecule>,
    next_id: LabMoleculeId,

    // Consolidated MD engine (all unlocked molecules)
    md: Option<GPUMolecularDynamics>,
    md_dirty: bool,
    md_running: bool,

    // Reaction templates
    reaction_templates: Vec<StructuralReactionTemplate>,

    // Stats
    step_count: u64,
    last_stats: MDStats,
}

impl PharmaLab {
    pub fn new(config: LabConfig) -> Self {
        Self {
            config,
            molecules: Vec::new(),
            next_id: 1,
            md: None,
            md_dirty: false,
            md_running: false,
            reaction_templates: Vec::new(),
            step_count: 0,
            last_stats: MDStats::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Molecule management
    // -----------------------------------------------------------------------

    /// Add a single atom as a new molecule. Returns (molecule_id, atom_index).
    pub fn add_atom(
        &mut self,
        element: PeriodicElement,
        position: [f32; 3],
    ) -> (LabMoleculeId, usize) {
        let id = self.next_id;
        self.next_id += 1;
        let mut graph = MoleculeGraph::new(element.symbol());
        graph.add_element(element);
        self.molecules.push(LabMolecule {
            id,
            name: element.symbol().to_string(),
            graph,
            positions: vec![position],
            velocities: vec![[0.0; 3]],
            md_offset: 0,
            locked: true, // new atoms start locked (user is building)
        });
        self.md_dirty = true;
        (id, 0)
    }

    /// Add a bond between two atoms in the same molecule.
    pub fn add_bond(
        &mut self,
        mol_id: LabMoleculeId,
        atom_a: usize,
        atom_b: usize,
        order: BondOrder,
    ) -> Result<(), String> {
        let mol = self
            .molecules
            .iter_mut()
            .find(|m| m.id == mol_id)
            .ok_or_else(|| format!("molecule {} not found", mol_id))?;
        mol.graph.add_bond(atom_a, atom_b, order)?;
        self.md_dirty = true;
        Ok(())
    }

    /// Remove an atom from a molecule.
    pub fn remove_atom(&mut self, mol_id: LabMoleculeId, atom_idx: usize) -> Result<(), String> {
        let mol = self
            .molecules
            .iter_mut()
            .find(|m| m.id == mol_id)
            .ok_or_else(|| format!("molecule {} not found", mol_id))?;
        if atom_idx >= mol.graph.atoms.len() {
            return Err(format!("atom index {} out of range", atom_idx));
        }
        // Remove the atom
        mol.graph.atoms.remove(atom_idx);
        mol.positions.remove(atom_idx);
        mol.velocities.remove(atom_idx);
        // Remove bonds referencing this atom, fix indices
        mol.graph.bonds.retain(|b| b.i != atom_idx && b.j != atom_idx);
        for bond in &mut mol.graph.bonds {
            if bond.i > atom_idx { bond.i -= 1; }
            if bond.j > atom_idx { bond.j -= 1; }
        }
        // Remove molecule if empty
        if mol.graph.atoms.is_empty() {
            self.molecules.retain(|m| m.id != mol_id);
        }
        self.md_dirty = true;
        Ok(())
    }

    /// Remove a bond from a molecule.
    pub fn remove_bond(&mut self, mol_id: LabMoleculeId, bond_idx: usize) -> Result<(), String> {
        let mol = self
            .molecules
            .iter_mut()
            .find(|m| m.id == mol_id)
            .ok_or_else(|| format!("molecule {} not found", mol_id))?;
        if bond_idx >= mol.graph.bonds.len() {
            return Err(format!("bond index {} out of range", bond_idx));
        }
        mol.graph.bonds.remove(bond_idx);
        self.md_dirty = true;
        Ok(())
    }

    /// Remove an entire molecule.
    pub fn remove_molecule(&mut self, id: LabMoleculeId) -> Result<(), String> {
        let before = self.molecules.len();
        self.molecules.retain(|m| m.id != id);
        if self.molecules.len() == before {
            return Err(format!("molecule {} not found", id));
        }
        self.md_dirty = true;
        Ok(())
    }

    /// Add molecule from SMILES string.
    pub fn add_molecule_from_smiles(&mut self, smiles_str: &str) -> Result<LabMoleculeId, String> {
        let (graph, positions) = smiles::parse_smiles(smiles_str)?;
        let id = self.next_id;
        self.next_id += 1;
        let n = graph.atom_count();
        self.molecules.push(LabMolecule {
            id,
            name: smiles_str.to_string(),
            graph,
            positions,
            velocities: vec![[0.0; 3]; n],
            md_offset: 0,
            locked: false,
        });
        self.md_dirty = true;
        Ok(id)
    }

    /// Add molecule from the built-in library.
    pub fn add_library_molecule(&mut self, name: &str) -> Result<LabMoleculeId, String> {
        let entry = library::get_library_molecule(name)
            .ok_or_else(|| format!("library molecule '{}' not found", name))?;
        self.add_molecule_from_smiles(&entry.smiles)
    }

    /// Merge two standalone atom-molecules when a bond is drawn between them.
    pub fn merge_molecules(
        &mut self,
        mol_a: LabMoleculeId,
        mol_b: LabMoleculeId,
    ) -> Result<LabMoleculeId, String> {
        if mol_a == mol_b {
            return Ok(mol_a);
        }
        let idx_b = self.molecules.iter().position(|m| m.id == mol_b)
            .ok_or_else(|| format!("molecule {} not found", mol_b))?;
        let b = self.molecules.remove(idx_b);
        let a = self.molecules.iter_mut().find(|m| m.id == mol_a)
            .ok_or_else(|| format!("molecule {} not found", mol_a))?;
        let offset = a.graph.atoms.len();
        // Merge atoms
        for atom in b.graph.atoms {
            a.graph.atoms.push(atom);
        }
        a.positions.extend_from_slice(&b.positions);
        a.velocities.extend_from_slice(&b.velocities);
        // Merge bonds (shift indices)
        for mut bond in b.graph.bonds {
            bond.i += offset;
            bond.j += offset;
            a.graph.bonds.push(bond);
        }
        a.name = a.molecular_formula();
        self.md_dirty = true;
        Ok(mol_a)
    }

    // -----------------------------------------------------------------------
    // MD simulation
    // -----------------------------------------------------------------------

    /// Rebuild the consolidated MD system from unlocked molecules.
    pub fn rebuild_md_system(&mut self) {
        let unlocked: Vec<usize> = self.molecules.iter()
            .enumerate()
            .filter(|(_, m)| !m.locked)
            .map(|(i, _)| i)
            .collect();

        let total_atoms: usize = unlocked.iter()
            .map(|&i| self.molecules[i].atom_count())
            .sum();

        if total_atoms == 0 {
            self.md = None;
            self.md_dirty = false;
            return;
        }

        let mut md = GPUMolecularDynamics::new(total_atoms, "cpu");
        md.set_box(self.config.box_size);
        md.set_temperature(self.config.temperature_kelvin);

        let mut positions = Vec::with_capacity(total_atoms * 3);
        let mut masses = Vec::with_capacity(total_atoms);
        let mut sigmas = Vec::with_capacity(total_atoms);
        let mut epsilons = Vec::with_capacity(total_atoms);
        let mut charges = vec![0.0f32; total_atoms];

        let mut offset = 0usize;
        for &mol_idx in &unlocked {
            let mol = &mut self.molecules[mol_idx];
            mol.md_offset = offset;

            for (i, atom) in mol.graph.atoms.iter().enumerate() {
                let pos = mol.positions[i];
                positions.extend_from_slice(&pos);
                masses.push(atom.element.mass_daltons());
                let (sigma, eps) = atom.element.uff_lj_params();
                sigmas.push(sigma);
                epsilons.push(eps);

                // Simple Gasteiger-like charge estimate from electronegativity
                if let Some(en) = atom.element.pauling_electronegativity() {
                    charges[offset + i] = ((en - 2.5) * 0.1) as f32;
                }
            }

            // Add bonds
            for bond in &mol.graph.bonds {
                let r0 = mol.graph.atoms[bond.i].element.covalent_radius_angstrom()
                    + mol.graph.atoms[bond.j].element.covalent_radius_angstrom();
                let k = match bond.order {
                    BondOrder::Single => 300.0,
                    BondOrder::Double => 600.0,
                    BondOrder::Triple => 900.0,
                    BondOrder::Aromatic => 450.0,
                };
                md.add_bond(offset + bond.i, offset + bond.j, r0, k);
            }

            // Generate angles from bond graph (all i-j-k triplets)
            let n = mol.graph.atoms.len();
            let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
            for bond in &mol.graph.bonds {
                adj[bond.i].push(bond.j);
                adj[bond.j].push(bond.i);
            }
            for j in 0..n {
                for &i in &adj[j] {
                    for &k in &adj[j] {
                        if i < k {
                            // Default angle: ~109.5° (tetrahedral) or ~120° depending on hybridization
                            let theta0 = if adj[j].len() <= 2 { 180.0f32.to_radians() }
                                else if adj[j].len() == 3 { 120.0f32.to_radians() }
                                else { 109.47f32.to_radians() };
                            md.add_angle(offset + i, offset + j, offset + k, theta0, 50.0);
                        }
                    }
                }
            }

            offset += mol.atom_count();
        }

        md.set_positions(&positions);
        md.set_masses(&masses);
        md.set_lj_params(&sigmas, &epsilons);
        md.set_charges(&charges);
        md.initialize_velocities();

        self.md = Some(md);
        self.md_dirty = false;
    }

    /// Step the MD simulation. Returns stats.
    pub fn step_md(&mut self, n_steps: u32) -> MDStats {
        if self.md_dirty {
            self.rebuild_md_system();
        }
        let stats = if let Some(md) = &mut self.md {
            let dt = 0.002; // 2 fs timestep
            let mut last_stats = MDStats::default();
            for _ in 0..n_steps {
                last_stats = md.step(dt);
            }
            self.step_count += n_steps as u64;

            // Write positions back to molecules
            let positions = md.positions();
            for mol in &mut self.molecules {
                if mol.locked { continue; }
                for i in 0..mol.atom_count() {
                    let base = (mol.md_offset + i) * 3;
                    if base + 2 < positions.len() {
                        mol.positions[i] = [positions[base], positions[base + 1], positions[base + 2]];
                    }
                }
            }
            last_stats
        } else {
            MDStats::default()
        };
        self.last_stats = stats.clone();
        stats
    }

    pub fn set_temperature(&mut self, kelvin: f32) {
        self.config.temperature_kelvin = kelvin;
        if let Some(md) = &mut self.md {
            md.set_temperature(kelvin);
        }
    }

    pub fn set_md_running(&mut self, running: bool) {
        self.md_running = running;
        // Unlock all molecules when starting MD
        if running {
            for mol in &mut self.molecules {
                mol.locked = false;
            }
            self.md_dirty = true;
        }
    }

    pub fn is_md_running(&self) -> bool {
        self.md_running
    }

    // -----------------------------------------------------------------------
    // Drug discovery
    // -----------------------------------------------------------------------

    /// Dock a molecule against a named target.
    pub fn dock_ligand(
        &self,
        ligand_id: LabMoleculeId,
        target_name: &str,
    ) -> Result<DockingResult, String> {
        let mol = self.molecules.iter().find(|m| m.id == ligand_id)
            .ok_or_else(|| format!("molecule {} not found", ligand_id))?;

        let candidate = DrugCandidate::new(
            &mol.name,
            mol.graph.molecular_mass_daltons() as f64,
            1.5, // estimated logP
            7.0, // estimated pKa
            target_name,
            -5.0, // initial estimate
        );

        let site = BindingSite::new(
            target_name,
            &["ARG120", "TYR355", "HIS90", "GLU524"],
            500.0,
            0.6,
        );

        let results = drug_discovery::screen_drug_candidates(&[candidate], &site, 42);
        results.into_iter().next().ok_or_else(|| "docking failed".to_string())
    }

    /// Compute ADMET profile for a molecule.
    pub fn compute_admet(&self, mol_id: LabMoleculeId) -> Result<ADMETProfile, String> {
        let mol = self.molecules.iter().find(|m| m.id == mol_id)
            .ok_or_else(|| format!("molecule {} not found", mol_id))?;

        let mass = mol.graph.molecular_mass_daltons() as f64;
        let comp = mol.graph.element_composition();
        let n_atoms = mol.graph.atom_count();
        let n_bonds = mol.graph.bond_count();

        // Estimate descriptors from molecular graph
        let hbd = comp.get(&PeriodicElement::N).copied().unwrap_or(0) as u32
            + comp.get(&PeriodicElement::O).copied().unwrap_or(0) as u32;
        let hba = hbd + comp.get(&PeriodicElement::F).copied().unwrap_or(0) as u32;
        let rotatable = n_bonds.saturating_sub(n_atoms) as u32;
        let aromatic = mol.graph.bonds.iter()
            .filter(|b| b.order == BondOrder::Aromatic)
            .count() as u32 / 2;

        let candidate = DrugCandidate::with_descriptors(
            &mol.name, mass, 1.5, 7.0, "generic", -5.0,
            hbd, hba, 60.0, rotatable, aromatic, 0.0,
        );

        Ok(drug_discovery::predict_admet(&candidate))
    }

    // -----------------------------------------------------------------------
    // Reaction detection
    // -----------------------------------------------------------------------

    /// Check for proximity-based reactions between molecules.
    pub fn check_proximity_reactions(&self) -> Vec<ReactionEvent> {
        let mut events = Vec::new();
        let n = self.molecules.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if self.molecules[i].locked || self.molecules[j].locked { continue; }
                // Check if any atom pair is within reaction distance
                for (ai, pos_a) in self.molecules[i].positions.iter().enumerate() {
                    for (aj, pos_b) in self.molecules[j].positions.iter().enumerate() {
                        let dx = pos_a[0] - pos_b[0];
                        let dy = pos_a[1] - pos_b[1];
                        let dz = pos_a[2] - pos_b[2];
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                        let elem_a = self.molecules[i].graph.atoms[ai].element;
                        let elem_b = self.molecules[j].graph.atoms[aj].element;
                        let threshold = (elem_a.van_der_waals_radius_angstrom()
                            + elem_b.van_der_waals_radius_angstrom()) * 0.8;

                        if dist < threshold {
                            events.push(ReactionEvent {
                                template_name: "proximity".to_string(),
                                molecule_ids: vec![self.molecules[i].id, self.molecules[j].id],
                                atoms_involved: vec![ai, aj],
                                energy_change_kcal: -1.0,
                                reaction_type: "approach".to_string(),
                            });
                        }
                    }
                }
            }
        }
        events
    }

    // -----------------------------------------------------------------------
    // Snapshot
    // -----------------------------------------------------------------------

    /// Build a snapshot for sending to the frontend.
    pub fn snapshot(&self) -> PharmaLabSnapshot {
        PharmaLabSnapshot {
            molecules: self.molecules.iter().map(|m| PharmaLabMoleculeSnapshot {
                id: m.id,
                name: m.name.clone(),
                formula: m.molecular_formula(),
                molecular_weight: m.graph.molecular_mass_daltons(),
                atom_count: m.atom_count(),
                bond_count: m.graph.bond_count(),
                locked: m.locked,
            }).collect(),
            md_stats: if self.step_count > 0 { Some(self.last_stats.clone()) } else { None },
            temperature_kelvin: self.config.temperature_kelvin,
            box_size: self.config.box_size,
            step_count: self.step_count,
            md_running: self.md_running,
        }
    }

    /// Build a frame for streaming to the frontend.
    pub fn build_frame(&self, reactions: &[ReactionEvent]) -> PharmaLabFrame {
        let mut atoms = Vec::new();
        let mut bonds = Vec::new();
        let mut global_offset = 0usize;

        for mol in &self.molecules {
            for (i, atom) in mol.graph.atoms.iter().enumerate() {
                atoms.push(PharmaLabAtomFrame {
                    position: mol.positions[i],
                    element: atom.element.symbol().to_string(),
                    cpk_color: atom.element.cpk_color_rgb(),
                    radius: atom.element.van_der_waals_radius_angstrom(),
                    molecule_id: mol.id,
                });
            }
            for bond in &mol.graph.bonds {
                bonds.push(PharmaLabBondFrame {
                    atom_a: global_offset + bond.i,
                    atom_b: global_offset + bond.j,
                    order: bond.order.bond_order(),
                });
            }
            global_offset += mol.atom_count();
        }

        PharmaLabFrame {
            atoms,
            bonds,
            md_stats: self.last_stats.clone(),
            step_count: self.step_count,
            reactions: reactions.to_vec(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_lab_and_add_atoms() {
        let mut lab = PharmaLab::new(LabConfig::default());
        let (id1, _) = lab.add_atom(PeriodicElement::C, [0.0, 0.0, 0.0]);
        let (id2, _) = lab.add_atom(PeriodicElement::O, [1.5, 0.0, 0.0]);
        assert_eq!(lab.molecules.len(), 2);
        // Merge them
        let merged = lab.merge_molecules(id1, id2).unwrap();
        assert_eq!(lab.molecules.len(), 1);
        assert_eq!(lab.molecules[0].id, merged);
        assert_eq!(lab.molecules[0].atom_count(), 2);
    }

    #[test]
    fn add_bond_between_atoms() {
        let mut lab = PharmaLab::new(LabConfig::default());
        let (id1, _) = lab.add_atom(PeriodicElement::C, [0.0, 0.0, 0.0]);
        let (id2, _) = lab.add_atom(PeriodicElement::O, [1.2, 0.0, 0.0]);
        let merged = lab.merge_molecules(id1, id2).unwrap();
        lab.add_bond(merged, 0, 1, BondOrder::Double).unwrap();
        assert_eq!(lab.molecules[0].graph.bond_count(), 1);
    }

    #[test]
    fn remove_atom_cleans_up_bonds() {
        let mut lab = PharmaLab::new(LabConfig::default());
        let (id, _) = lab.add_atom(PeriodicElement::C, [0.0, 0.0, 0.0]);
        // Add more atoms to the same molecule by merging
        let (id2, _) = lab.add_atom(PeriodicElement::H, [1.0, 0.0, 0.0]);
        let (id3, _) = lab.add_atom(PeriodicElement::H, [0.0, 1.0, 0.0]);
        let merged = lab.merge_molecules(id, id2).unwrap();
        let merged = lab.merge_molecules(merged, id3).unwrap();
        lab.add_bond(merged, 0, 1, BondOrder::Single).unwrap();
        lab.add_bond(merged, 0, 2, BondOrder::Single).unwrap();
        assert_eq!(lab.molecules[0].graph.bond_count(), 2);

        // Remove the carbon (index 0) — both bonds should vanish
        lab.remove_atom(merged, 0).unwrap();
        assert_eq!(lab.molecules[0].atom_count(), 2);
        assert_eq!(lab.molecules[0].graph.bond_count(), 0);
    }

    #[test]
    fn md_rebuild_and_step() {
        let mut lab = PharmaLab::new(LabConfig::default());
        let (id1, _) = lab.add_atom(PeriodicElement::O, [0.0, 0.0, 0.0]);
        let (id2, _) = lab.add_atom(PeriodicElement::H, [0.96, 0.0, 0.0]);
        let (id3, _) = lab.add_atom(PeriodicElement::H, [-0.24, 0.93, 0.0]);
        let merged = lab.merge_molecules(id1, id2).unwrap();
        let merged = lab.merge_molecules(merged, id3).unwrap();
        lab.add_bond(merged, 0, 1, BondOrder::Single).unwrap();
        lab.add_bond(merged, 0, 2, BondOrder::Single).unwrap();

        // Unlock for MD
        lab.molecules[0].locked = false;
        lab.rebuild_md_system();
        assert!(lab.md.is_some());

        let stats = lab.step_md(10);
        assert!(stats.total_energy.is_finite());
        assert!(lab.step_count == 10);
    }

    #[test]
    fn snapshot_contains_molecule_data() {
        let mut lab = PharmaLab::new(LabConfig::default());
        lab.add_atom(PeriodicElement::Au, [0.0, 0.0, 0.0]);
        let snap = lab.snapshot();
        assert_eq!(snap.molecules.len(), 1);
        assert_eq!(snap.molecules[0].formula, "Au");
        assert!((snap.molecules[0].molecular_weight - 196.967).abs() < 0.1);
    }

    #[test]
    fn library_molecule_loads() {
        let mut lab = PharmaLab::new(LabConfig::default());
        let result = lab.add_library_molecule("water");
        assert!(result.is_ok());
        assert_eq!(lab.molecules.len(), 1);
        assert!(lab.molecules[0].atom_count() >= 3);
    }

    #[test]
    fn admet_computes() {
        let mut lab = PharmaLab::new(LabConfig::default());
        let (id1, _) = lab.add_atom(PeriodicElement::C, [0.0, 0.0, 0.0]);
        let (id2, _) = lab.add_atom(PeriodicElement::O, [1.2, 0.0, 0.0]);
        let merged = lab.merge_molecules(id1, id2).unwrap();
        let admet = lab.compute_admet(merged);
        assert!(admet.is_ok());
        let profile = admet.unwrap();
        assert!(profile.drug_likeness >= 0.0);
    }
}
