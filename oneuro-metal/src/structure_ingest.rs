//! PDB / mmCIF structure ingestion → `EmbeddedMolecule`.
//!
//! Parses coordinate files (ATOM/HETATM records for PDB, `_atom_site` loops
//! for mmCIF) into [`EmbeddedMolecule`] instances.  Bonds are inferred from
//! inter-atomic distances using covalent radii when the file does not supply
//! explicit CONECT records.
//!
//! # Design Notes
//!
//! Force field parameters are NOT imported wholesale — the quantum layer
//! derives them.  This module's job is purely structural: atoms + coordinates
//! + connectivity.  Bond orders default to Single; a future refinement can
//! assign orders from geometry or from the quantum solution itself.

use crate::atomistic_chemistry::{
    BondOrder, EmbeddedMolecule, MoleculeGraph, PeriodicElement,
};

// ---------------------------------------------------------------------------
// Covalent radii (Å) for distance-based bond inference.
// Cordero et al., Dalton Trans. 2008 — "Covalent radii revisited".
// ---------------------------------------------------------------------------

fn covalent_radius_angstrom(element: PeriodicElement) -> f32 {
    match element {
        PeriodicElement::H => 0.31,
        PeriodicElement::C => 0.76,
        PeriodicElement::N => 0.71,
        PeriodicElement::O => 0.66,
        PeriodicElement::S => 1.05,
        PeriodicElement::P => 1.07,
        PeriodicElement::Fe => 1.32,
        PeriodicElement::Mg => 1.41,
        PeriodicElement::Ca => 1.76,
        PeriodicElement::Zn => 1.22,
        PeriodicElement::Na => 1.66,
        PeriodicElement::K => 2.03,
        PeriodicElement::Cl => 1.02,
        PeriodicElement::Mn => 1.39,
        PeriodicElement::Cu => 1.32,
        PeriodicElement::Se => 1.20,
        _ => 1.50, // conservative fallback
    }
}

/// Tolerance factor for covalent bond detection (1.0 + TOLERANCE).
const BOND_TOLERANCE: f32 = 0.40; // 40 % tolerance

// ---------------------------------------------------------------------------
// PDB parsing
// ---------------------------------------------------------------------------

/// Errors that can occur during structure ingestion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructureIngestError {
    /// An ATOM/HETATM line has an unrecognised element symbol.
    UnknownElement(String),
    /// Coordinate parse failure on a specific line.
    CoordinateParse { line_number: usize, detail: String },
    /// The file contained no atoms.
    EmptyStructure,
    /// mmCIF loop column not found.
    MissingColumn(String),
}

impl std::fmt::Display for StructureIngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownElement(s) => write!(f, "unknown element: {s}"),
            Self::CoordinateParse { line_number, detail } => {
                write!(f, "line {line_number}: {detail}")
            }
            Self::EmptyStructure => write!(f, "no atoms found"),
            Self::MissingColumn(c) => write!(f, "missing mmCIF column: {c}"),
        }
    }
}

impl std::error::Error for StructureIngestError {}

/// Options controlling structure ingestion.
#[derive(Debug, Clone)]
pub struct IngestOptions {
    /// Whether to include HETATM records (ligands, waters, ions).
    pub include_hetatm: bool,
    /// Specific chain IDs to include.  Empty = all chains.
    pub chain_filter: Vec<String>,
    /// Whether to infer bonds from distances when CONECT records are absent.
    pub infer_bonds: bool,
    /// Maximum bond length tolerance factor (default 0.40).
    pub bond_tolerance: f32,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            include_hetatm: true,
            chain_filter: Vec::new(),
            infer_bonds: true,
            bond_tolerance: BOND_TOLERANCE,
        }
    }
}

/// Parse a PDB-format string into an `EmbeddedMolecule`.
///
/// Reads ATOM / HETATM records for coordinates and element identity.
/// CONECT records are used for bonds when present; otherwise bonds are
/// inferred from inter-atomic distances using covalent radii.
pub fn embedded_molecule_from_pdb(
    pdb_text: &str,
    options: &IngestOptions,
) -> Result<EmbeddedMolecule, StructureIngestError> {
    let mut elements: Vec<PeriodicElement> = Vec::new();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut conect_pairs: Vec<(usize, usize)> = Vec::new();
    // Map from PDB serial number → our index.
    let mut serial_to_idx: std::collections::HashMap<usize, usize> = Default::default();

    for (line_no, line) in pdb_text.lines().enumerate() {
        let record = &line[..6.min(line.len())];
        let is_atom = record.starts_with("ATOM  ");
        let is_hetatm = record.starts_with("HETATM");

        if is_atom || (is_hetatm && options.include_hetatm) {
            if line.len() < 54 {
                continue; // malformed short line
            }

            // Chain ID (column 22, 1-indexed → index 21).
            if !options.chain_filter.is_empty() && line.len() > 21 {
                let chain = line[21..22].trim();
                if !chain.is_empty()
                    && !options
                        .chain_filter
                        .iter()
                        .any(|c| c.eq_ignore_ascii_case(chain))
                {
                    continue;
                }
            }

            // Element symbol: PDB columns 77-78, fallback to atom name columns 13-16.
            let element_str = if line.len() >= 78 {
                line[76..78].trim()
            } else {
                // Fall back: first non-digit, non-space char in columns 13-16.
                let name = &line[12..16.min(line.len())];
                name.trim()
                    .trim_start_matches(|c: char| c.is_ascii_digit())
                    .split(|c: char| c.is_ascii_digit())
                    .next()
                    .unwrap_or("")
                    .trim()
            };

            let element = PeriodicElement::from_symbol_or_name(element_str).ok_or_else(|| {
                StructureIngestError::UnknownElement(element_str.to_string())
            })?;

            // Coordinates: columns 31-38, 39-46, 47-54 (1-indexed).
            let x = line[30..38]
                .trim()
                .parse::<f32>()
                .map_err(|e| StructureIngestError::CoordinateParse {
                    line_number: line_no + 1,
                    detail: format!("x: {e}"),
                })?;
            let y = line[38..46]
                .trim()
                .parse::<f32>()
                .map_err(|e| StructureIngestError::CoordinateParse {
                    line_number: line_no + 1,
                    detail: format!("y: {e}"),
                })?;
            let z = line[46..54]
                .trim()
                .parse::<f32>()
                .map_err(|e| StructureIngestError::CoordinateParse {
                    line_number: line_no + 1,
                    detail: format!("z: {e}"),
                })?;

            // PDB serial number: columns 7-11.
            if let Ok(serial) = line[6..11].trim().parse::<usize>() {
                serial_to_idx.insert(serial, elements.len());
            }

            elements.push(element);
            positions.push([x, y, z]);
        } else if record.starts_with("CONECT") && line.len() >= 11 {
            // CONECT records: first serial at 7-11, bonded serials at 12-16, 17-21, ...
            if let Ok(base_serial) = line[6..11].trim().parse::<usize>() {
                let mut col = 11;
                while col + 5 <= line.len() {
                    if let Ok(bonded) = line[col..col + 5].trim().parse::<usize>() {
                        if bonded != base_serial {
                            conect_pairs.push((base_serial, bonded));
                        }
                    }
                    col += 5;
                }
            }
        }
    }

    if elements.is_empty() {
        return Err(StructureIngestError::EmptyStructure);
    }

    let mut graph = MoleculeGraph::new("pdb_structure");
    for &elem in &elements {
        graph.add_element(elem);
    }

    // Add bonds: from CONECT records first, then infer remaining.
    let mut bonded_pairs: std::collections::HashSet<(usize, usize)> = Default::default();

    for &(s1, s2) in &conect_pairs {
        if let (Some(&i1), Some(&i2)) = (serial_to_idx.get(&s1), serial_to_idx.get(&s2)) {
            let pair = if i1 < i2 { (i1, i2) } else { (i2, i1) };
            if bonded_pairs.insert(pair) {
                let _ = graph.add_bond(pair.0, pair.1, BondOrder::Single);
            }
        }
    }

    // Distance-based bond inference.
    if options.infer_bonds && bonded_pairs.is_empty() {
        infer_bonds_from_distances(
            &elements,
            &positions,
            options.bond_tolerance,
            &mut graph,
        );
    }

    EmbeddedMolecule::new(graph, positions).map_err(|_e| StructureIngestError::EmptyStructure)
}

// ---------------------------------------------------------------------------
// mmCIF parsing
// ---------------------------------------------------------------------------

/// Parse an mmCIF-format string into an `EmbeddedMolecule`.
///
/// Reads the `_atom_site` loop for coordinates and element identity.
/// Bonds are always inferred from distances (mmCIF bond tables not yet
/// supported).
pub fn embedded_molecule_from_mmcif(
    cif_text: &str,
    options: &IngestOptions,
) -> Result<EmbeddedMolecule, StructureIngestError> {
    // Find the _atom_site loop.
    let lines: Vec<&str> = cif_text.lines().collect();
    let mut in_atom_site;
    let mut columns: Vec<String> = Vec::new();
    let mut data_rows: Vec<Vec<String>> = Vec::new();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        if line.starts_with("loop_") {
            // Check if the next lines define _atom_site columns.
            in_atom_site = false;
            columns.clear();
            i += 1;
            while i < lines.len() && lines[i].trim().starts_with("_atom_site.") {
                columns.push(lines[i].trim().to_string());
                in_atom_site = true;
                i += 1;
            }
            if in_atom_site {
                // Now read data rows until a line starts with '_', 'loop_', '#', or is empty.
                while i < lines.len() {
                    let row = lines[i].trim();
                    if row.is_empty()
                        || row.starts_with('_')
                        || row.starts_with("loop_")
                        || row.starts_with('#')
                    {
                        break;
                    }
                    data_rows.push(split_cif_row(row));
                    i += 1;
                }
                break; // We found our atom_site block.
            }
        } else {
            i += 1;
        }
    }

    if columns.is_empty() || data_rows.is_empty() {
        return Err(StructureIngestError::EmptyStructure);
    }

    // Find column indices.
    let col_idx = |name: &str| -> Result<usize, StructureIngestError> {
        columns
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| StructureIngestError::MissingColumn(name.to_string()))
    };

    let type_col = col_idx("_atom_site.type_symbol")?;
    let x_col = col_idx("_atom_site.Cartn_x")?;
    let y_col = col_idx("_atom_site.Cartn_y")?;
    let z_col = col_idx("_atom_site.Cartn_z")?;
    let group_col = col_idx("_atom_site.group_PDB").ok();
    let chain_col = col_idx("_atom_site.auth_asym_id")
        .or_else(|_| col_idx("_atom_site.label_asym_id"))
        .ok();

    let mut elements: Vec<PeriodicElement> = Vec::new();
    let mut positions: Vec<[f32; 3]> = Vec::new();

    for (row_no, row) in data_rows.iter().enumerate() {
        if row.len() <= type_col || row.len() <= x_col || row.len() <= y_col || row.len() <= z_col
        {
            continue;
        }

        // Filter HETATM if requested.
        if let Some(gc) = group_col {
            if row.len() > gc && row[gc] == "HETATM" && !options.include_hetatm {
                continue;
            }
        }

        // Chain filter.
        if let Some(cc) = chain_col {
            if !options.chain_filter.is_empty() && row.len() > cc {
                let chain = &row[cc];
                if !options
                    .chain_filter
                    .iter()
                    .any(|c| c.eq_ignore_ascii_case(chain))
                {
                    continue;
                }
            }
        }

        let element_str = &row[type_col];
        let element =
            PeriodicElement::from_symbol_or_name(element_str).ok_or_else(|| {
                StructureIngestError::UnknownElement(element_str.clone())
            })?;

        let x = row[x_col]
            .parse::<f32>()
            .map_err(|e| StructureIngestError::CoordinateParse {
                line_number: row_no + 1,
                detail: format!("x: {e}"),
            })?;
        let y = row[y_col]
            .parse::<f32>()
            .map_err(|e| StructureIngestError::CoordinateParse {
                line_number: row_no + 1,
                detail: format!("y: {e}"),
            })?;
        let z = row[z_col]
            .parse::<f32>()
            .map_err(|e| StructureIngestError::CoordinateParse {
                line_number: row_no + 1,
                detail: format!("z: {e}"),
            })?;

        elements.push(element);
        positions.push([x, y, z]);
    }

    if elements.is_empty() {
        return Err(StructureIngestError::EmptyStructure);
    }

    let mut graph = MoleculeGraph::new("mmcif_structure");
    for &elem in &elements {
        graph.add_element(elem);
    }

    if options.infer_bonds {
        infer_bonds_from_distances(&elements, &positions, options.bond_tolerance, &mut graph);
    }

    EmbeddedMolecule::new(graph, positions).map_err(|_e| StructureIngestError::EmptyStructure)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Infer covalent bonds from inter-atomic distances using covalent radii.
///
/// Two atoms are bonded if distance < (r_i + r_j) * (1 + tolerance).
/// Uses O(n²) brute force — fine for typical structures (< 50 000 atoms).
fn infer_bonds_from_distances(
    elements: &[PeriodicElement],
    positions: &[[f32; 3]],
    tolerance: f32,
    graph: &mut MoleculeGraph,
) {
    let n = elements.len();
    for i in 0..n {
        let ri = covalent_radius_angstrom(elements[i]);
        let pi = positions[i];
        for j in (i + 1)..n {
            let rj = covalent_radius_angstrom(elements[j]);
            let max_dist = (ri + rj) * (1.0 + tolerance);
            let max_dist_sq = max_dist * max_dist;

            let dx = pi[0] - positions[j][0];
            let dy = pi[1] - positions[j][1];
            let dz = pi[2] - positions[j][2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq <= max_dist_sq && dist_sq > 0.01 {
                let _ = graph.add_bond(i, j, BondOrder::Single);
            }
        }
    }
}

/// Split a mmCIF data row by whitespace, respecting single-quoted fields.
fn split_cif_row(row: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = row.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c == '\'' {
            chars.next(); // consume opening quote
            let mut token = String::new();
            while let Some(&inner) = chars.peek() {
                if inner == '\'' {
                    chars.next();
                    break;
                }
                token.push(inner);
                chars.next();
            }
            tokens.push(token);
        } else {
            let mut token = String::new();
            while let Some(&inner) = chars.peek() {
                if inner.is_whitespace() {
                    break;
                }
                token.push(inner);
                chars.next();
            }
            tokens.push(token);
        }
    }
    tokens
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MINI_PDB: &str = "\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.460   2.000   3.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.010   3.420   3.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.240   4.380   3.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.960   1.260   1.780  1.00  0.00           C
ATOM      6  H   ALA A   1       0.600   1.100   3.000  1.00  0.00           H
END
";

    #[test]
    fn test_pdb_parse_basic() {
        let mol = embedded_molecule_from_pdb(MINI_PDB, &IngestOptions::default()).unwrap();
        assert_eq!(mol.graph.atom_count(), 6);
        assert_eq!(mol.positions_angstrom.len(), 6);
        // N is first
        assert_eq!(mol.graph.atoms[0].element, PeriodicElement::N);
        // H is last
        assert_eq!(mol.graph.atoms[5].element, PeriodicElement::H);
    }

    #[test]
    fn test_pdb_bond_inference() {
        let mol = embedded_molecule_from_pdb(MINI_PDB, &IngestOptions::default()).unwrap();
        // N-CA bond should be inferred (~1.46 Å, well within covalent radius sum)
        assert!(
            mol.graph.bonds.len() >= 4,
            "expected at least 4 bonds, got {}",
            mol.graph.bonds.len()
        );
    }

    #[test]
    fn test_pdb_empty_returns_error() {
        let result = embedded_molecule_from_pdb("REMARK empty\nEND\n", &IngestOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_pdb_chain_filter() {
        let opts = IngestOptions {
            chain_filter: vec!["B".to_string()],
            ..IngestOptions::default()
        };
        // All atoms are chain A, so filtering for B should give empty.
        let result = embedded_molecule_from_pdb(MINI_PDB, &opts);
        assert!(result.is_err()); // EmptyStructure
    }

    #[test]
    fn test_pdb_with_conect() {
        let pdb_with_conect = "\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.460   2.000   3.000  1.00  0.00           C
CONECT    1    2
END
";
        let mol =
            embedded_molecule_from_pdb(pdb_with_conect, &IngestOptions::default()).unwrap();
        assert_eq!(mol.graph.atom_count(), 2);
        assert_eq!(mol.graph.bonds.len(), 1);
    }

    #[test]
    fn test_covalent_radii_reasonable() {
        // C-C bond ~ 1.54 Å, sum of radii = 1.52 Å — should be detected.
        let r_cc = covalent_radius_angstrom(PeriodicElement::C) * 2.0;
        assert!(r_cc > 1.4 && r_cc < 1.6, "C-C covalent sum = {r_cc}");

        // N-H bond ~ 1.01 Å, sum of radii = 1.02 Å.
        let r_nh =
            covalent_radius_angstrom(PeriodicElement::N) + covalent_radius_angstrom(PeriodicElement::H);
        assert!(r_nh > 0.9 && r_nh < 1.2, "N-H covalent sum = {r_nh}");
    }

    const MINI_MMCIF: &str = "\
data_test
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM N  N   ALA A 1.000 2.000 3.000
ATOM C  CA  ALA A 2.460 2.000 3.000
ATOM C  C   ALA A 3.010 3.420 3.000
ATOM O  O   ALA A 2.240 4.380 3.000
";

    #[test]
    fn test_mmcif_parse_basic() {
        let mol =
            embedded_molecule_from_mmcif(MINI_MMCIF, &IngestOptions::default()).unwrap();
        assert_eq!(mol.graph.atom_count(), 4);
        assert_eq!(mol.graph.atoms[0].element, PeriodicElement::N);
        assert_eq!(mol.graph.atoms[3].element, PeriodicElement::O);
    }

    #[test]
    fn test_mmcif_bond_inference() {
        let mol =
            embedded_molecule_from_mmcif(MINI_MMCIF, &IngestOptions::default()).unwrap();
        assert!(
            mol.graph.bonds.len() >= 2,
            "expected at least 2 bonds from distance inference, got {}",
            mol.graph.bonds.len()
        );
    }

    #[test]
    fn test_split_cif_row() {
        let tokens = split_cif_row("ATOM N  N   ALA A 1.000 2.000 3.000");
        assert_eq!(tokens.len(), 8);
        assert_eq!(tokens[0], "ATOM");
        assert_eq!(tokens[1], "N");
    }

    #[test]
    fn test_split_cif_row_quoted() {
        let tokens = split_cif_row("'hello world' foo");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], "hello world");
        assert_eq!(tokens[1], "foo");
    }
}
