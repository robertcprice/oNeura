//! Biomolecular-level chemistry types: elements, bond orders, molecular graphs.
//!
//! This module provides the structural types used by [`crate::structure_ingest`]
//! to represent parsed PDB/mmCIF structures with full biomolecular topology
//! (residues, chains, secondary structure).  The types here sit between raw
//! coordinate files and the MD engine's low-level atom arrays.
//!
//! # Design Notes
//!
//! - [`PeriodicElement`] covers biologically relevant elements (wider than
//!   `molecular_dynamics::Element` which only carries LJ params).
//! - [`MoleculeGraph`] stores atoms + bonds + optional residue topology.
//! - [`EmbeddedMolecule`] pairs a graph with 3-D coordinates.
//! - [`BiomolecularTopology`] adds chain/residue/secondary-structure annotation.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Periodic element
// ---------------------------------------------------------------------------

/// Biologically relevant elements for structure ingestion.
///
/// Covers all elements found in PDB protein/nucleic-acid/ligand structures.
/// Use [`from_symbol_or_name`](PeriodicElement::from_symbol_or_name) for
/// case-insensitive lookup from PDB/mmCIF element columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PeriodicElement {
    H,
    C,
    N,
    O,
    S,
    P,
    Fe,
    Mg,
    Ca,
    Zn,
    Na,
    K,
    Cl,
    Mn,
    Cu,
    Se,
    // Less common but found in PDB structures
    Co,
    Ni,
    Mo,
    W,
    F,
    Br,
    I,
    Si,
    B,
}

impl PeriodicElement {
    /// Look up element from a 1- or 2-character symbol (case-insensitive).
    pub fn from_symbol_or_name(s: &str) -> Option<Self> {
        match s.trim().to_uppercase().as_str() {
            "H" => Some(Self::H),
            "C" => Some(Self::C),
            "N" => Some(Self::N),
            "O" => Some(Self::O),
            "S" => Some(Self::S),
            "P" => Some(Self::P),
            "FE" => Some(Self::Fe),
            "MG" => Some(Self::Mg),
            "CA" => Some(Self::Ca),
            "ZN" => Some(Self::Zn),
            "NA" => Some(Self::Na),
            "K" => Some(Self::K),
            "CL" => Some(Self::Cl),
            "MN" => Some(Self::Mn),
            "CU" => Some(Self::Cu),
            "SE" => Some(Self::Se),
            "CO" => Some(Self::Co),
            "NI" => Some(Self::Ni),
            "MO" => Some(Self::Mo),
            "W" => Some(Self::W),
            "F" => Some(Self::F),
            "BR" => Some(Self::Br),
            "I" => Some(Self::I),
            "SI" => Some(Self::Si),
            "B" => Some(Self::B),
            _ => None,
        }
    }

    /// Atomic mass in daltons (approximate, for biomolecular use).
    pub fn mass_daltons(self) -> f32 {
        match self {
            Self::H => 1.008,
            Self::C => 12.011,
            Self::N => 14.007,
            Self::O => 15.999,
            Self::S => 32.065,
            Self::P => 30.974,
            Self::Fe => 55.845,
            Self::Mg => 24.305,
            Self::Ca => 40.078,
            Self::Zn => 65.38,
            Self::Na => 22.990,
            Self::K => 39.098,
            Self::Cl => 35.453,
            Self::Mn => 54.938,
            Self::Cu => 63.546,
            Self::Se => 78.971,
            Self::Co => 58.933,
            Self::Ni => 58.693,
            Self::Mo => 95.95,
            Self::W => 183.84,
            Self::F => 18.998,
            Self::Br => 79.904,
            Self::I => 126.904,
            Self::Si => 28.086,
            Self::B => 10.811,
        }
    }

    /// Standard covalent radius in angstroms (Cordero et al., Dalton Trans. 2008).
    pub fn covalent_radius_angstrom(self) -> f32 {
        match self {
            Self::H => 0.31,
            Self::C => 0.76,
            Self::N => 0.71,
            Self::O => 0.66,
            Self::S => 1.05,
            Self::P => 1.07,
            Self::Fe => 1.32,
            Self::Mg => 1.41,
            Self::Ca => 1.76,
            Self::Zn => 1.22,
            Self::Na => 1.66,
            Self::K => 2.03,
            Self::Cl => 1.02,
            Self::Mn => 1.39,
            Self::Cu => 1.32,
            Self::Se => 1.20,
            Self::Co => 1.26,
            Self::Ni => 1.24,
            Self::Mo => 1.54,
            Self::W => 1.62,
            Self::F => 0.57,
            Self::Br => 1.20,
            Self::I => 1.39,
            Self::Si => 1.11,
            Self::B => 0.84,
        }
    }

    /// Short element symbol (1-2 chars, title case).
    pub fn symbol(self) -> &'static str {
        match self {
            Self::H => "H",
            Self::C => "C",
            Self::N => "N",
            Self::O => "O",
            Self::S => "S",
            Self::P => "P",
            Self::Fe => "Fe",
            Self::Mg => "Mg",
            Self::Ca => "Ca",
            Self::Zn => "Zn",
            Self::Na => "Na",
            Self::K => "K",
            Self::Cl => "Cl",
            Self::Mn => "Mn",
            Self::Cu => "Cu",
            Self::Se => "Se",
            Self::Co => "Co",
            Self::Ni => "Ni",
            Self::Mo => "Mo",
            Self::W => "W",
            Self::F => "F",
            Self::Br => "Br",
            Self::I => "I",
            Self::Si => "Si",
            Self::B => "B",
        }
    }
}

impl PeriodicElement {
    /// Atomic number (number of protons).
    pub fn atomic_number(self) -> u8 {
        match self {
            Self::H => 1,
            Self::B => 5,
            Self::C => 6,
            Self::N => 7,
            Self::O => 8,
            Self::F => 9,
            Self::Si => 14,
            Self::P => 15,
            Self::S => 16,
            Self::Cl => 17,
            Self::K => 19,
            Self::Ca => 20,
            Self::Mn => 25,
            Self::Fe => 26,
            Self::Co => 27,
            Self::Ni => 28,
            Self::Cu => 29,
            Self::Zn => 30,
            Self::Se => 34,
            Self::Br => 35,
            Self::Mo => 42,
            Self::I => 53,
            Self::W => 74,
            Self::Na => 11,
            Self::Mg => 12,
        }
    }
}

impl std::fmt::Display for PeriodicElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

// ---------------------------------------------------------------------------
// Bond order
// ---------------------------------------------------------------------------

/// Covalent bond multiplicity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BondOrder {
    Single,
    Double,
    Triple,
    Aromatic,
}

impl BondOrder {
    /// Numeric bond order (1.0 for single, 2.0 for double, etc.).
    pub fn bond_order(self) -> f32 {
        match self {
            Self::Single => 1.0,
            Self::Double => 2.0,
            Self::Triple => 3.0,
            Self::Aromatic => 1.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Atom node
// ---------------------------------------------------------------------------

/// A single atom in a [`MoleculeGraph`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtomNode {
    pub element: PeriodicElement,
    /// PDB atom name (e.g. "CA", "N", "OG1").  Empty if not from a structure file.
    #[serde(default)]
    pub atom_name: String,
    /// Residue annotation, if this atom belongs to a biomolecular chain.
    #[serde(default)]
    pub residue: Option<ResidueInfo>,
    /// Formal charge on this atom (e.g. -1 for carboxylate O, +1 for ammonium N).
    #[serde(default)]
    pub formal_charge: i8,
    /// Isotope mass number override (e.g. 13 for 13C). If `None`, use the natural
    /// abundance mass rounded to the nearest integer.
    #[serde(default)]
    pub isotope_mass_number: Option<u16>,
}

impl AtomNode {
    /// Create a minimal atom node from just an element (no residue, no charge).
    pub fn new(element: PeriodicElement) -> Self {
        Self {
            element,
            atom_name: String::new(),
            residue: None,
            formal_charge: 0,
            isotope_mass_number: None,
        }
    }

    /// Atomic mass in daltons as f64 (uses isotope override if present,
    /// otherwise falls back to the element's natural abundance mass).
    pub fn atomic_mass(&self) -> f64 {
        if let Some(mass_number) = self.isotope_mass_number {
            f64::from(mass_number)
        } else {
            f64::from(self.element.mass_daltons())
        }
    }
}

/// Residue-level annotation for an atom.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidueInfo {
    /// Three-letter residue name (e.g. "ALA", "DA", "HOH").
    pub name: String,
    /// Chain identifier (e.g. "A", "B").
    pub chain_id: String,
    /// Residue sequence number.
    pub seq_num: i32,
    /// PDB insertion code (usually empty).
    #[serde(default)]
    pub ins_code: String,
}

// ---------------------------------------------------------------------------
// Bond edge
// ---------------------------------------------------------------------------

/// A bond between two atoms in a [`MoleculeGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BondEdge {
    pub i: usize,
    pub j: usize,
    pub order: BondOrder,
}

// ---------------------------------------------------------------------------
// Secondary structure
// ---------------------------------------------------------------------------

/// A secondary structure element parsed from PDB HELIX/SHEET records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecondaryStructureElement {
    pub kind: SecondaryStructureKind,
    pub chain_id: String,
    pub start_seq: i32,
    pub end_seq: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecondaryStructureKind {
    AlphaHelix,
    Helix310,
    PiHelix,
    BetaStrand,
    Turn,
}

// ---------------------------------------------------------------------------
// Disulfide bridge
// ---------------------------------------------------------------------------

/// A disulfide bond between two cysteine residues.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisulfideBridge {
    pub chain_id_1: String,
    pub seq_num_1: i32,
    pub chain_id_2: String,
    pub seq_num_2: i32,
}

// ---------------------------------------------------------------------------
// Molecule graph
// ---------------------------------------------------------------------------

/// Labeled molecular graph: atoms + bonds + optional residue topology.
///
/// This is a purely structural representation — no coordinates.  Pair with
/// [`EmbeddedMolecule`] for spatial information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoleculeGraph {
    pub name: String,
    pub atoms: Vec<AtomNode>,
    pub bonds: Vec<BondEdge>,
}

impl MoleculeGraph {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            atoms: Vec::new(),
            bonds: Vec::new(),
        }
    }

    /// Add an atom with element only (no residue annotation). Returns the atom index.
    pub fn add_element(&mut self, element: PeriodicElement) -> usize {
        let idx = self.atoms.len();
        self.atoms.push(AtomNode::new(element));
        idx
    }

    /// Add a fully constructed [`AtomNode`] and return its index.
    pub fn add_atom_node(&mut self, node: AtomNode) -> usize {
        let idx = self.atoms.len();
        self.atoms.push(node);
        idx
    }

    /// Add an atom with full biomolecular annotation.
    pub fn add_atom(&mut self, element: PeriodicElement, atom_name: &str, residue: Option<ResidueInfo>) -> usize {
        let idx = self.atoms.len();
        self.atoms.push(AtomNode {
            element,
            atom_name: atom_name.to_string(),
            residue,
            formal_charge: 0,
            isotope_mass_number: None,
        });
        idx
    }

    /// Add a bond between atoms i and j.
    pub fn add_bond(&mut self, i: usize, j: usize, order: BondOrder) -> Result<(), String> {
        if i >= self.atoms.len() || j >= self.atoms.len() {
            return Err(format!(
                "bond indices out of range: ({i}, {j}) for {} atoms",
                self.atoms.len()
            ));
        }
        if i == j {
            return Err(format!("self-bond not allowed: ({i}, {j})"));
        }
        self.bonds.push(BondEdge { i, j, order });
        Ok(())
    }

    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    pub fn bond_count(&self) -> usize {
        self.bonds.len()
    }

    /// Unique chain IDs found in atom residue annotations.
    pub fn chain_ids(&self) -> Vec<String> {
        let mut chains: Vec<String> = self
            .atoms
            .iter()
            .filter_map(|a| a.residue.as_ref().map(|r| r.chain_id.clone()))
            .collect();
        chains.sort();
        chains.dedup();
        chains
    }

    /// Count of unique residues (by chain_id + seq_num).
    pub fn residue_count(&self) -> usize {
        let mut residues: Vec<(String, i32)> = self
            .atoms
            .iter()
            .filter_map(|a| {
                a.residue
                    .as_ref()
                    .map(|r| (r.chain_id.clone(), r.seq_num))
            })
            .collect();
        residues.sort();
        residues.dedup();
        residues.len()
    }

    /// Total molecular mass in daltons.
    pub fn molecular_mass_daltons(&self) -> f32 {
        self.atoms.iter().map(|a| a.element.mass_daltons()).sum()
    }

    /// Atom composition as element → count.
    pub fn element_composition(&self) -> std::collections::HashMap<PeriodicElement, usize> {
        let mut counts = std::collections::HashMap::new();
        for atom in &self.atoms {
            *counts.entry(atom.element).or_insert(0) += 1;
        }
        counts
    }

    // -----------------------------------------------------------------------
    // Representative molecule stubs for quantum-runtime mapping
    // -----------------------------------------------------------------------

    /// Stub ATP graph (adenosine triphosphate).
    pub fn representative_atp() -> Self { Self::new("ATP") }
    /// Stub ADP graph (adenosine diphosphate).
    pub fn representative_adp() -> Self { Self::new("ADP") }
    /// Stub orthophosphoric acid graph (Pi / H3PO4).
    pub fn representative_orthophosphoric_acid() -> Self { Self::new("Pi") }
    /// Stub glucose graph.
    pub fn representative_glucose() -> Self { Self::new("glucose") }
    /// Stub O2 graph.
    pub fn representative_oxygen_gas() -> Self { Self::new("O2") }
    /// Stub amino-acid pool graph.
    pub fn representative_amino_acid_pool() -> Self { Self::new("amino_acid_pool") }
    /// Stub nucleotide pool graph.
    pub fn representative_nucleotide_pool() -> Self { Self::new("nucleotide_pool") }
    /// Stub membrane precursor pool graph.
    pub fn representative_membrane_precursor_pool() -> Self { Self::new("membrane_precursor_pool") }
    /// Stub NAD+ (oxidized) graph.
    pub fn representative_nad_oxidized() -> Self { Self::new("NAD_oxidized") }
    /// Stub NADH (reduced) graph.
    pub fn representative_nad_reduced() -> Self { Self::new("NAD_reduced") }
    /// Stub coenzyme A graph.
    pub fn representative_coenzyme_a() -> Self { Self::new("CoA") }

    /// Sum of all formal charges on atoms in the graph.
    pub fn net_charge(&self) -> i32 {
        self.atoms.iter().map(|a| i32::from(a.formal_charge)).sum()
    }
}

// ---------------------------------------------------------------------------
// Embedded molecule
// ---------------------------------------------------------------------------

/// A molecular graph with 3-D coordinates attached to each atom.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddedMolecule {
    pub graph: MoleculeGraph,
    pub positions_angstrom: Vec<[f32; 3]>,
}

impl EmbeddedMolecule {
    pub fn new(graph: MoleculeGraph, positions: Vec<[f32; 3]>) -> Result<Self, String> {
        if graph.atom_count() != positions.len() {
            return Err(format!(
                "atom count mismatch: graph has {} atoms, positions has {}",
                graph.atom_count(),
                positions.len()
            ));
        }
        if positions.is_empty() {
            return Err("empty molecule".to_string());
        }
        Ok(Self {
            graph,
            positions_angstrom: positions,
        })
    }

    /// Bounding box as ([min_x, min_y, min_z], [max_x, max_y, max_z]).
    pub fn bounding_box(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for pos in &self.positions_angstrom {
            for d in 0..3 {
                lo[d] = lo[d].min(pos[d]);
                hi[d] = hi[d].max(pos[d]);
            }
        }
        (lo, hi)
    }

    /// Maximum extent in any dimension (angstroms).
    pub fn max_extent(&self) -> f32 {
        let (lo, hi) = self.bounding_box();
        (0..3).map(|d| hi[d] - lo[d]).fold(0.0f32, f32::max)
    }

    /// Extract atoms belonging to a specific chain.
    pub fn chain_subset(&self, chain_id: &str) -> Self {
        let mut new_graph = MoleculeGraph::new(&format!("{}_{}", self.graph.name, chain_id));
        let mut new_positions = Vec::new();
        let mut old_to_new: std::collections::HashMap<usize, usize> = Default::default();

        for (old_idx, atom) in self.graph.atoms.iter().enumerate() {
            let matches = atom
                .residue
                .as_ref()
                .map_or(false, |r| r.chain_id == chain_id);
            if matches {
                let new_idx = new_graph.atoms.len();
                old_to_new.insert(old_idx, new_idx);
                new_graph.atoms.push(atom.clone());
                new_positions.push(self.positions_angstrom[old_idx]);
            }
        }

        for bond in &self.graph.bonds {
            if let (Some(&ni), Some(&nj)) = (old_to_new.get(&bond.i), old_to_new.get(&bond.j)) {
                let _ = new_graph.add_bond(ni, nj, bond.order);
            }
        }

        Self {
            graph: new_graph,
            positions_angstrom: new_positions,
        }
    }

    /// Return a copy of this molecule with all positions shifted by `delta`.
    pub fn translated(&self, delta: [f32; 3]) -> Self {
        Self {
            graph: self.graph.clone(),
            positions_angstrom: self
                .positions_angstrom
                .iter()
                .map(|p| [p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]])
                .collect(),
        }
    }

    /// Extract atoms within a radius (angstroms) of a reference point.
    pub fn sphere_subset(&self, center: [f32; 3], radius: f32) -> Self {
        let r2 = radius * radius;
        let mut new_graph = MoleculeGraph::new(&format!("{}_sphere", self.graph.name));
        let mut new_positions = Vec::new();
        let mut old_to_new: std::collections::HashMap<usize, usize> = Default::default();

        for (old_idx, pos) in self.positions_angstrom.iter().enumerate() {
            let dx = pos[0] - center[0];
            let dy = pos[1] - center[1];
            let dz = pos[2] - center[2];
            if dx * dx + dy * dy + dz * dz <= r2 {
                let new_idx = new_graph.atoms.len();
                old_to_new.insert(old_idx, new_idx);
                new_graph.atoms.push(self.graph.atoms[old_idx].clone());
                new_positions.push(*pos);
            }
        }

        for bond in &self.graph.bonds {
            if let (Some(&ni), Some(&nj)) = (old_to_new.get(&bond.i), old_to_new.get(&bond.j)) {
                let _ = new_graph.add_bond(ni, nj, bond.order);
            }
        }

        Self {
            graph: new_graph,
            positions_angstrom: new_positions,
        }
    }
}

// ---------------------------------------------------------------------------
// Biomolecular topology
// ---------------------------------------------------------------------------

/// Full biomolecular annotation for a parsed structure.
///
/// Wraps an [`EmbeddedMolecule`] with chain/residue/secondary-structure
/// metadata parsed from PDB HELIX/SHEET/SSBOND records (or mmCIF
/// equivalents).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BiomolecularTopology {
    pub molecule: EmbeddedMolecule,
    pub secondary_structure: Vec<SecondaryStructureElement>,
    pub disulfide_bridges: Vec<DisulfideBridge>,
    pub title: String,
}

impl BiomolecularTopology {
    pub fn new(molecule: EmbeddedMolecule) -> Self {
        Self {
            molecule,
            secondary_structure: Vec::new(),
            disulfide_bridges: Vec::new(),
            title: String::new(),
        }
    }

    /// Look up secondary structure assignment for a residue.
    pub fn secondary_structure_at(&self, chain_id: &str, seq_num: i32) -> Option<SecondaryStructureKind> {
        self.secondary_structure
            .iter()
            .find(|ss| ss.chain_id == chain_id && seq_num >= ss.start_seq && seq_num <= ss.end_seq)
            .map(|ss| ss.kind)
    }

    /// Find all atom indices belonging to a specific residue.
    pub fn residue_atom_indices(&self, chain_id: &str, seq_num: i32) -> Vec<usize> {
        self.molecule
            .graph
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(idx, atom)| {
                atom.residue.as_ref().and_then(|r| {
                    if r.chain_id == chain_id && r.seq_num == seq_num {
                        Some(idx)
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Count of unique chains.
    pub fn chain_count(&self) -> usize {
        self.molecule.graph.chain_ids().len()
    }

    /// Count of unique residues.
    pub fn residue_count(&self) -> usize {
        self.molecule.graph.residue_count()
    }
}

// ---------------------------------------------------------------------------
// Standard residue bond-order templates
// ---------------------------------------------------------------------------

/// Backbone bond orders for standard amino acids.
///
/// Returns (atom_name_1, atom_name_2, BondOrder) for intra-residue bonds.
pub fn amino_acid_backbone_bonds() -> &'static [(&'static str, &'static str, BondOrder)] {
    &[
        ("N", "CA", BondOrder::Single),
        ("CA", "C", BondOrder::Single),
        ("C", "O", BondOrder::Double),
        ("CA", "CB", BondOrder::Single),
    ]
}

/// Peptide bond between residues (C of residue i, N of residue i+1).
pub const PEPTIDE_BOND_ORDER: BondOrder = BondOrder::Single;

/// Nucleotide backbone bond orders for standard DNA/RNA.
pub fn nucleotide_backbone_bonds() -> &'static [(&'static str, &'static str, BondOrder)] {
    &[
        ("P", "O5'", BondOrder::Single),
        ("O5'", "C5'", BondOrder::Single),
        ("C5'", "C4'", BondOrder::Single),
        ("C4'", "C3'", BondOrder::Single),
        ("C3'", "O3'", BondOrder::Single),
        ("C4'", "O4'", BondOrder::Single),
        ("O4'", "C1'", BondOrder::Single),
        ("C1'", "C2'", BondOrder::Single),
    ]
}

// ---------------------------------------------------------------------------
// Scoped atom reference (for structural reactions across reactant molecules)
// ---------------------------------------------------------------------------

/// Reference to a specific atom within a multi-reactant reaction context.
/// `reactant_idx` selects which reactant molecule, `atom_idx` selects the atom
/// within that molecule's graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScopedAtomRef {
    pub reactant_idx: usize,
    pub atom_idx: usize,
}

impl ScopedAtomRef {
    pub fn new(reactant_idx: usize, atom_idx: usize) -> Self {
        Self {
            reactant_idx,
            atom_idx,
        }
    }
}

// ---------------------------------------------------------------------------
// Structural reaction edit operations
// ---------------------------------------------------------------------------

/// An elementary topological edit in a structural reaction: break a bond,
/// form a bond, change bond order, or set a formal charge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StructuralReactionEdit {
    BreakBond {
        a: ScopedAtomRef,
        b: ScopedAtomRef,
    },
    FormBond {
        a: ScopedAtomRef,
        b: ScopedAtomRef,
        order: BondOrder,
    },
    ChangeBondOrder {
        a: ScopedAtomRef,
        b: ScopedAtomRef,
        order: BondOrder,
    },
    SetFormalCharge {
        atom: ScopedAtomRef,
        formal_charge: i8,
    },
}

// ---------------------------------------------------------------------------
// Structural reaction template
// ---------------------------------------------------------------------------

/// A named template describing a multi-reactant structural reaction:
/// a set of reactant molecule graphs and a sequence of topological edits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralReactionTemplate {
    pub name: String,
    pub reactants: Vec<MoleculeGraph>,
    pub edits: Vec<StructuralReactionEdit>,
}

impl StructuralReactionTemplate {
    pub fn new(
        name: &str,
        reactants: Vec<MoleculeGraph>,
        edits: Vec<StructuralReactionEdit>,
    ) -> Self {
        Self {
            name: name.to_string(),
            reactants,
            edits,
        }
    }
}

// ---------------------------------------------------------------------------
// Embedded material mixture
// ---------------------------------------------------------------------------

/// A named collection of embedded molecules, each with an associated amount
/// (e.g. moles or a concentration proxy). Used by the quantum runtime to
/// represent multi-component microdomains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddedMaterialMixtureComponent {
    pub molecule: EmbeddedMolecule,
    pub amount: f64,
}

/// A named mixture of embedded molecules for microdomain quantum chemistry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddedMaterialMixture {
    pub name: String,
    pub components: Vec<EmbeddedMaterialMixtureComponent>,
}

impl EmbeddedMaterialMixture {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            components: Vec::new(),
        }
    }

    pub fn add_component(&mut self, molecule: EmbeddedMolecule, amount: f64) {
        self.components.push(EmbeddedMaterialMixtureComponent {
            molecule,
            amount,
        });
    }
}

// ---------------------------------------------------------------------------
// Embedded material structural reaction error
// ---------------------------------------------------------------------------

/// Error type for structural-reaction operations on embedded material mixtures.
#[derive(Debug, Clone)]
pub struct EmbeddedMaterialStructuralReactionError {
    pub message: String,
}

impl std::fmt::Display for EmbeddedMaterialStructuralReactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "structural reaction error: {}", self.message)
    }
}

impl std::error::Error for EmbeddedMaterialStructuralReactionError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_from_symbol() {
        assert_eq!(PeriodicElement::from_symbol_or_name("C"), Some(PeriodicElement::C));
        assert_eq!(PeriodicElement::from_symbol_or_name("fe"), Some(PeriodicElement::Fe));
        assert_eq!(PeriodicElement::from_symbol_or_name("ZN"), Some(PeriodicElement::Zn));
        assert_eq!(PeriodicElement::from_symbol_or_name("XX"), None);
    }

    #[test]
    fn element_mass_reasonable() {
        assert!((PeriodicElement::C.mass_daltons() - 12.011).abs() < 0.01);
        assert!((PeriodicElement::Fe.mass_daltons() - 55.845).abs() < 0.01);
    }

    #[test]
    fn graph_basic_operations() {
        let mut g = MoleculeGraph::new("test");
        g.add_element(PeriodicElement::C);
        g.add_element(PeriodicElement::O);
        g.add_bond(0, 1, BondOrder::Double).unwrap();
        assert_eq!(g.atom_count(), 2);
        assert_eq!(g.bond_count(), 1);
        assert_eq!(g.bonds[0].order, BondOrder::Double);
    }

    #[test]
    fn graph_bond_out_of_range() {
        let mut g = MoleculeGraph::new("test");
        g.add_element(PeriodicElement::C);
        assert!(g.add_bond(0, 5, BondOrder::Single).is_err());
    }

    #[test]
    fn embedded_molecule_bounding_box() {
        let mut g = MoleculeGraph::new("test");
        g.add_element(PeriodicElement::C);
        g.add_element(PeriodicElement::O);
        let mol = EmbeddedMolecule::new(g, vec![[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]]).unwrap();
        let (lo, hi) = mol.bounding_box();
        assert_eq!(lo, [0.0, 0.0, 0.0]);
        assert_eq!(hi, [3.0, 4.0, 5.0]);
        assert!((mol.max_extent() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn embedded_molecule_size_mismatch() {
        let mut g = MoleculeGraph::new("test");
        g.add_element(PeriodicElement::C);
        assert!(EmbeddedMolecule::new(g, vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).is_err());
    }

    #[test]
    fn chain_subset() {
        let mut g = MoleculeGraph::new("test");
        g.add_atom(PeriodicElement::N, "N", Some(ResidueInfo {
            name: "ALA".to_string(),
            chain_id: "A".to_string(),
            seq_num: 1,
            ins_code: String::new(),
        }));
        g.add_atom(PeriodicElement::C, "CA", Some(ResidueInfo {
            name: "ALA".to_string(),
            chain_id: "A".to_string(),
            seq_num: 1,
            ins_code: String::new(),
        }));
        g.add_atom(PeriodicElement::N, "N", Some(ResidueInfo {
            name: "GLY".to_string(),
            chain_id: "B".to_string(),
            seq_num: 1,
            ins_code: String::new(),
        }));
        g.add_bond(0, 1, BondOrder::Single).unwrap();
        let mol = EmbeddedMolecule::new(
            g,
            vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0]],
        )
        .unwrap();

        let chain_a = mol.chain_subset("A");
        assert_eq!(chain_a.graph.atom_count(), 2);
        assert_eq!(chain_a.graph.bond_count(), 1);

        let chain_b = mol.chain_subset("B");
        assert_eq!(chain_b.graph.atom_count(), 1);
        assert_eq!(chain_b.graph.bond_count(), 0);
    }

    #[test]
    fn sphere_subset() {
        let mut g = MoleculeGraph::new("test");
        g.add_element(PeriodicElement::C);
        g.add_element(PeriodicElement::N);
        g.add_element(PeriodicElement::O);
        g.add_bond(0, 1, BondOrder::Single).unwrap();
        g.add_bond(1, 2, BondOrder::Single).unwrap();
        let mol = EmbeddedMolecule::new(
            g,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        )
        .unwrap();

        let sub = mol.sphere_subset([0.0, 0.0, 0.0], 2.0);
        assert_eq!(sub.graph.atom_count(), 2); // C and N within 2 A
        assert_eq!(sub.graph.bond_count(), 1); // C-N bond preserved
    }

    #[test]
    fn residue_and_chain_counting() {
        let mut g = MoleculeGraph::new("test");
        for i in 0..4 {
            g.add_atom(
                PeriodicElement::C,
                "CA",
                Some(ResidueInfo {
                    name: "ALA".to_string(),
                    chain_id: if i < 2 { "A" } else { "B" }.to_string(),
                    seq_num: (i % 2) + 1,
                    ins_code: String::new(),
                }),
            );
        }
        assert_eq!(g.chain_ids(), vec!["A", "B"]);
        assert_eq!(g.residue_count(), 4); // A:1, A:2, B:1, B:2
    }

    #[test]
    fn biomolecular_topology_secondary_structure() {
        let mut g = MoleculeGraph::new("test");
        g.add_atom(PeriodicElement::N, "N", Some(ResidueInfo {
            name: "ALA".to_string(),
            chain_id: "A".to_string(),
            seq_num: 5,
            ins_code: String::new(),
        }));
        let mol = EmbeddedMolecule::new(g, vec![[0.0, 0.0, 0.0]]).unwrap();
        let mut topo = BiomolecularTopology::new(mol);
        topo.secondary_structure.push(SecondaryStructureElement {
            kind: SecondaryStructureKind::AlphaHelix,
            chain_id: "A".to_string(),
            start_seq: 3,
            end_seq: 10,
        });
        assert_eq!(
            topo.secondary_structure_at("A", 5),
            Some(SecondaryStructureKind::AlphaHelix)
        );
        assert_eq!(topo.secondary_structure_at("A", 11), None);
    }

    #[test]
    fn molecular_mass() {
        let mut g = MoleculeGraph::new("water");
        g.add_element(PeriodicElement::O);
        g.add_element(PeriodicElement::H);
        g.add_element(PeriodicElement::H);
        let mass = g.molecular_mass_daltons();
        assert!((mass - 18.015).abs() < 0.01, "water mass = {mass}");
    }

    #[test]
    fn backbone_bond_templates_nonempty() {
        assert!(amino_acid_backbone_bonds().len() >= 3);
        assert!(nucleotide_backbone_bonds().len() >= 5);
    }
}
