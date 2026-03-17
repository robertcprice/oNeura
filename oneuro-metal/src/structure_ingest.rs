//! PDB / mmCIF structure ingestion → [`EmbeddedMolecule`] and [`BiomolecularTopology`].
//!
//! Parses coordinate files (ATOM/HETATM records for PDB, `_atom_site` loops
//! for mmCIF) into biomolecular structures with full residue-level topology,
//! secondary structure, and disulfide bridge annotations.
//!
//! # Design Notes
//!
//! Force field parameters are NOT imported wholesale — the quantum layer
//! derives them.  This module's job is purely structural: atoms + coordinates
//! + connectivity + biomolecular annotation.  Bond orders default to Single
//! for distance-inferred bonds; standard residue templates assign correct
//! orders for backbone C=O double bonds.

use crate::atomistic_chemistry::{
    BiomolecularTopology, BondOrder, DisulfideBridge, EmbeddedMolecule, MoleculeGraph,
    PeriodicElement, ResidueInfo, SecondaryStructureElement, SecondaryStructureKind,
};

/// Tolerance factor for covalent bond detection (1.0 + TOLERANCE).
const BOND_TOLERANCE: f32 = 0.40; // 40 % tolerance

// ---------------------------------------------------------------------------
// Covalent radii (Å) for distance-based bond inference.
// Cordero et al., Dalton Trans. 2008 — "Covalent radii revisited".
// ---------------------------------------------------------------------------

fn covalent_radius_angstrom(element: PeriodicElement) -> f32 {
    element.covalent_radius_angstrom()
}

// ---------------------------------------------------------------------------
// Errors
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

// ---------------------------------------------------------------------------
// Ingest options
// ---------------------------------------------------------------------------

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
    /// Whether to parse HELIX/SHEET records for secondary structure.
    pub parse_secondary_structure: bool,
    /// Whether to parse SSBOND records for disulfide bridges.
    pub parse_disulfide_bridges: bool,
    /// Whether to store residue-level atom annotations.
    pub store_residue_info: bool,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            include_hetatm: true,
            chain_filter: Vec::new(),
            infer_bonds: true,
            bond_tolerance: BOND_TOLERANCE,
            parse_secondary_structure: true,
            parse_disulfide_bridges: true,
            store_residue_info: true,
        }
    }
}

// ---------------------------------------------------------------------------
// PDB parsing — raw EmbeddedMolecule
// ---------------------------------------------------------------------------

/// Parse a PDB-format string into an `EmbeddedMolecule`.
///
/// Reads ATOM / HETATM records for coordinates and element identity.
/// CONECT records are used for bonds when present; otherwise bonds are
/// inferred from inter-atomic distances using covalent radii.
pub fn embedded_molecule_from_pdb(
    pdb_text: &str,
    options: &IngestOptions,
) -> Result<EmbeddedMolecule, StructureIngestError> {
    let mut graph = MoleculeGraph::new("pdb_structure");
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut conect_pairs: Vec<(usize, usize)> = Vec::new();
    let mut serial_to_idx: std::collections::HashMap<usize, usize> = Default::default();

    for (line_no, line) in pdb_text.lines().enumerate() {
        let record = &line[..6.min(line.len())];
        let is_atom = record.starts_with("ATOM  ");
        let is_hetatm = record.starts_with("HETATM");

        if is_atom || (is_hetatm && options.include_hetatm) {
            if line.len() < 54 {
                continue;
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
                serial_to_idx.insert(serial, graph.atom_count());
            }

            // Residue-level annotation.
            let residue = if options.store_residue_info && line.len() >= 27 {
                let atom_name_str = line[12..16.min(line.len())].trim();
                let res_name = line[17..20.min(line.len())].trim();
                let chain_id = line[21..22].trim();
                let seq_num = line[22..26].trim().parse::<i32>().unwrap_or(0);
                let ins_code = if line.len() > 26 {
                    line[26..27].trim().to_string()
                } else {
                    String::new()
                };
                graph.add_atom(
                    element,
                    atom_name_str,
                    Some(ResidueInfo {
                        name: res_name.to_string(),
                        chain_id: chain_id.to_string(),
                        seq_num,
                        ins_code,
                    }),
                );
                true
            } else {
                graph.add_element(element);
                false
            };
            let _ = residue; // used for control flow

            positions.push([x, y, z]);
        } else if record.starts_with("CONECT") && line.len() >= 11 {
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

    if graph.atom_count() == 0 {
        return Err(StructureIngestError::EmptyStructure);
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
        let elements: Vec<PeriodicElement> = graph.atoms.iter().map(|a| a.element).collect();
        infer_bonds_from_distances(&elements, &positions, options.bond_tolerance, &mut graph);
    }

    // Upgrade backbone C=O bonds to double based on standard residue templates.
    upgrade_backbone_bond_orders(&mut graph);

    EmbeddedMolecule::new(graph, positions).map_err(|_e| StructureIngestError::EmptyStructure)
}

// ---------------------------------------------------------------------------
// PDB parsing — full BiomolecularTopology
// ---------------------------------------------------------------------------

/// Parse a PDB-format string into a full [`BiomolecularTopology`].
///
/// This is the high-level entry point that produces an annotated structure
/// with secondary structure, disulfide bridges, and residue-level metadata.
pub fn biomolecular_topology_from_pdb(
    pdb_text: &str,
    options: &IngestOptions,
) -> Result<BiomolecularTopology, StructureIngestError> {
    let molecule = embedded_molecule_from_pdb(pdb_text, options)?;
    let mut topo = BiomolecularTopology::new(molecule);

    // Parse TITLE.
    for line in pdb_text.lines() {
        if line.starts_with("TITLE ") && line.len() > 10 {
            let title_part = line[10..].trim();
            if !title_part.is_empty() {
                if !topo.title.is_empty() {
                    topo.title.push(' ');
                }
                topo.title.push_str(title_part);
            }
        }
    }

    // Parse HELIX records.
    if options.parse_secondary_structure {
        for line in pdb_text.lines() {
            if line.starts_with("HELIX ") && line.len() >= 38 {
                let helix_class = line[38..40.min(line.len())].trim().parse::<u32>().unwrap_or(1);
                let kind = match helix_class {
                    1 => SecondaryStructureKind::AlphaHelix,
                    5 => SecondaryStructureKind::Helix310,
                    3 => SecondaryStructureKind::PiHelix,
                    _ => SecondaryStructureKind::AlphaHelix,
                };
                let chain_id = line[19..20].trim().to_string();
                let start_seq = line[21..25].trim().parse::<i32>().unwrap_or(0);
                let end_seq = line[33..37].trim().parse::<i32>().unwrap_or(0);
                if start_seq <= end_seq {
                    topo.secondary_structure.push(SecondaryStructureElement {
                        kind,
                        chain_id,
                        start_seq,
                        end_seq,
                    });
                }
            }

            if line.starts_with("SHEET ") && line.len() >= 38 {
                let chain_id = line[21..22].trim().to_string();
                let start_seq = line[22..26].trim().parse::<i32>().unwrap_or(0);
                let end_seq = line[33..37].trim().parse::<i32>().unwrap_or(0);
                if start_seq <= end_seq {
                    topo.secondary_structure.push(SecondaryStructureElement {
                        kind: SecondaryStructureKind::BetaStrand,
                        chain_id,
                        start_seq,
                        end_seq,
                    });
                }
            }
        }
    }

    // Parse SSBOND records.
    if options.parse_disulfide_bridges {
        for line in pdb_text.lines() {
            if line.starts_with("SSBOND") && line.len() >= 35 {
                let chain1 = line[15..16].trim().to_string();
                let seq1 = line[17..21].trim().parse::<i32>().unwrap_or(0);
                let chain2 = line[29..30].trim().to_string();
                let seq2 = line[31..35].trim().parse::<i32>().unwrap_or(0);
                topo.disulfide_bridges.push(DisulfideBridge {
                    chain_id_1: chain1,
                    seq_num_1: seq1,
                    chain_id_2: chain2,
                    seq_num_2: seq2,
                });
            }
        }
    }

    Ok(topo)
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
    let lines: Vec<&str> = cif_text.lines().collect();
    let mut in_atom_site;
    let mut columns: Vec<String> = Vec::new();
    let mut data_rows: Vec<Vec<String>> = Vec::new();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        if line.starts_with("loop_") {
            in_atom_site = false;
            columns.clear();
            i += 1;
            while i < lines.len() && lines[i].trim().starts_with("_atom_site.") {
                columns.push(lines[i].trim().to_string());
                in_atom_site = true;
                i += 1;
            }
            if in_atom_site {
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
                break;
            }
        } else {
            i += 1;
        }
    }

    if columns.is_empty() || data_rows.is_empty() {
        return Err(StructureIngestError::EmptyStructure);
    }

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
    let atom_name_col = col_idx("_atom_site.label_atom_id")
        .or_else(|_| col_idx("_atom_site.auth_atom_id"))
        .ok();
    let res_name_col = col_idx("_atom_site.label_comp_id")
        .or_else(|_| col_idx("_atom_site.auth_comp_id"))
        .ok();
    let seq_num_col = col_idx("_atom_site.auth_seq_id")
        .or_else(|_| col_idx("_atom_site.label_seq_id"))
        .ok();

    let mut graph = MoleculeGraph::new("mmcif_structure");
    let mut positions: Vec<[f32; 3]> = Vec::new();

    for (row_no, row) in data_rows.iter().enumerate() {
        if row.len() <= type_col || row.len() <= x_col || row.len() <= y_col || row.len() <= z_col
        {
            continue;
        }

        if let Some(gc) = group_col {
            if row.len() > gc && row[gc] == "HETATM" && !options.include_hetatm {
                continue;
            }
        }

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

        // Residue-level annotation from mmCIF columns.
        if options.store_residue_info {
            let atom_name = atom_name_col
                .and_then(|c| row.get(c))
                .map(|s| s.as_str())
                .unwrap_or("");
            let res_name = res_name_col
                .and_then(|c| row.get(c))
                .map(|s| s.as_str())
                .unwrap_or("");
            let chain_id = chain_col
                .and_then(|c| row.get(c))
                .map(|s| s.as_str())
                .unwrap_or("");
            let seq_num = seq_num_col
                .and_then(|c| row.get(c))
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0);

            graph.add_atom(
                element,
                atom_name,
                Some(ResidueInfo {
                    name: res_name.to_string(),
                    chain_id: chain_id.to_string(),
                    seq_num,
                    ins_code: String::new(),
                }),
            );
        } else {
            graph.add_element(element);
        }

        positions.push([x, y, z]);
    }

    if graph.atom_count() == 0 {
        return Err(StructureIngestError::EmptyStructure);
    }

    if options.infer_bonds {
        let elements: Vec<PeriodicElement> = graph.atoms.iter().map(|a| a.element).collect();
        infer_bonds_from_distances(&elements, &positions, options.bond_tolerance, &mut graph);
    }

    upgrade_backbone_bond_orders(&mut graph);

    EmbeddedMolecule::new(graph, positions).map_err(|_e| StructureIngestError::EmptyStructure)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Infer covalent bonds from inter-atomic distances using covalent radii.
///
/// Two atoms are bonded if distance < (r_i + r_j) * (1 + tolerance).
/// Uses O(n^2) brute force — fine for typical structures (< 50 000 atoms).
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

/// Upgrade backbone C=O bonds from Single to Double using residue annotation.
///
/// For standard amino acids, the backbone carbonyl (atom name "C" bonded to "O")
/// should be a double bond.  This runs after distance-based bond inference.
fn upgrade_backbone_bond_orders(graph: &mut MoleculeGraph) {
    // Build lookup: which atoms have residue info and are backbone C or O?
    let carbonyl_c_indices: Vec<usize> = graph
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| {
            a.atom_name == "C"
                && a.residue.is_some()
        })
        .map(|(i, _)| i)
        .collect();

    let carbonyl_o_indices: Vec<usize> = graph
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| {
            a.atom_name == "O"
                && a.residue.is_some()
        })
        .map(|(i, _)| i)
        .collect();

    // For each C-O pair in the same residue, upgrade to double bond.
    for bond in &mut graph.bonds {
        let (bi, bj) = (bond.i, bond.j);
        let is_co = (carbonyl_c_indices.contains(&bi) && carbonyl_o_indices.contains(&bj))
            || (carbonyl_c_indices.contains(&bj) && carbonyl_o_indices.contains(&bi));
        if is_co {
            // Check same residue.
            let res_i = graph.atoms[bi].residue.as_ref();
            let res_j = graph.atoms[bj].residue.as_ref();
            if let (Some(ri), Some(rj)) = (res_i, res_j) {
                if ri.chain_id == rj.chain_id && ri.seq_num == rj.seq_num {
                    bond.order = BondOrder::Double;
                }
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
            chars.next();
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
        assert_eq!(mol.graph.atoms[0].element, PeriodicElement::N);
        assert_eq!(mol.graph.atoms[5].element, PeriodicElement::H);
    }

    #[test]
    fn test_pdb_residue_annotation() {
        let mol = embedded_molecule_from_pdb(MINI_PDB, &IngestOptions::default()).unwrap();
        let res = mol.graph.atoms[0].residue.as_ref().unwrap();
        assert_eq!(res.name, "ALA");
        assert_eq!(res.chain_id, "A");
        assert_eq!(res.seq_num, 1);
        assert_eq!(mol.graph.atoms[1].atom_name, "CA");
    }

    #[test]
    fn test_pdb_bond_inference() {
        let mol = embedded_molecule_from_pdb(MINI_PDB, &IngestOptions::default()).unwrap();
        assert!(
            mol.graph.bonds.len() >= 4,
            "expected at least 4 bonds, got {}",
            mol.graph.bonds.len()
        );
    }

    #[test]
    fn test_pdb_carbonyl_double_bond() {
        let mol = embedded_molecule_from_pdb(MINI_PDB, &IngestOptions::default()).unwrap();
        // Find the C=O bond (atom names "C" and "O" in same residue ALA 1).
        let c_idx = mol.graph.atoms.iter().position(|a| a.atom_name == "C").unwrap();
        let o_idx = mol.graph.atoms.iter().position(|a| a.atom_name == "O").unwrap();
        let co_bond = mol.graph.bonds.iter().find(|b| {
            (b.i == c_idx && b.j == o_idx) || (b.i == o_idx && b.j == c_idx)
        });
        assert!(co_bond.is_some(), "C-O bond not found");
        assert_eq!(co_bond.unwrap().order, BondOrder::Double, "C=O should be double bond");
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
        let result = embedded_molecule_from_pdb(MINI_PDB, &opts);
        assert!(result.is_err()); // EmptyStructure — all atoms are chain A
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
        let r_cc = covalent_radius_angstrom(PeriodicElement::C) * 2.0;
        assert!(r_cc > 1.4 && r_cc < 1.6, "C-C covalent sum = {r_cc}");

        let r_nh =
            covalent_radius_angstrom(PeriodicElement::N) + covalent_radius_angstrom(PeriodicElement::H);
        assert!(r_nh > 0.9 && r_nh < 1.2, "N-H covalent sum = {r_nh}");
    }

    // -- Secondary structure and disulfide --

    const PDB_WITH_ANNOTATIONS: &str = "\
TITLE     TEST STRUCTURE WITH ANNOTATIONS
HELIX    1 H1  ALA A    3  ALA A   10  1                               8
SHEET    1 S1  GLY A   15  GLY A   20  0
SSBOND   1 CYS A    5    CYS A   20
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.460   2.000   3.000  1.00  0.00           C
END
";

    #[test]
    fn test_biomolecular_topology_from_pdb() {
        let topo = biomolecular_topology_from_pdb(
            PDB_WITH_ANNOTATIONS,
            &IngestOptions::default(),
        )
        .unwrap();
        assert_eq!(topo.molecule.graph.atom_count(), 2);
        assert_eq!(topo.title, "TEST STRUCTURE WITH ANNOTATIONS");
    }

    #[test]
    fn test_secondary_structure_parsing() {
        let topo = biomolecular_topology_from_pdb(
            PDB_WITH_ANNOTATIONS,
            &IngestOptions::default(),
        )
        .unwrap();
        assert_eq!(topo.secondary_structure.len(), 2);
        assert_eq!(topo.secondary_structure[0].kind, SecondaryStructureKind::AlphaHelix);
        assert_eq!(topo.secondary_structure[0].start_seq, 3);
        assert_eq!(topo.secondary_structure[0].end_seq, 10);
        assert_eq!(topo.secondary_structure[1].kind, SecondaryStructureKind::BetaStrand);
    }

    #[test]
    fn test_disulfide_bridge_parsing() {
        let topo = biomolecular_topology_from_pdb(
            PDB_WITH_ANNOTATIONS,
            &IngestOptions::default(),
        )
        .unwrap();
        assert_eq!(topo.disulfide_bridges.len(), 1);
        assert_eq!(topo.disulfide_bridges[0].chain_id_1, "A");
        assert_eq!(topo.disulfide_bridges[0].seq_num_1, 5);
        assert_eq!(topo.disulfide_bridges[0].seq_num_2, 20);
    }

    #[test]
    fn test_no_residue_info_when_disabled() {
        let opts = IngestOptions {
            store_residue_info: false,
            ..IngestOptions::default()
        };
        let mol = embedded_molecule_from_pdb(MINI_PDB, &opts).unwrap();
        assert!(mol.graph.atoms[0].residue.is_none());
    }

    #[test]
    fn test_chain_and_residue_counting() {
        let mol = embedded_molecule_from_pdb(MINI_PDB, &IngestOptions::default()).unwrap();
        assert_eq!(mol.graph.chain_ids(), vec!["A"]);
        assert_eq!(mol.graph.residue_count(), 1); // all atoms are ALA A 1
    }

    #[test]
    fn test_molecular_mass() {
        let mol = embedded_molecule_from_pdb(MINI_PDB, &IngestOptions::default()).unwrap();
        let mass = mol.graph.molecular_mass_daltons();
        // N + 2C + O + C + H = 14.007 + 2*12.011 + 15.999 + 12.011 + 1.008 = 67.047
        assert!(mass > 60.0 && mass < 75.0, "mass = {mass}");
    }

    // -- mmCIF tests --

    const MINI_MMCIF: &str = "\
data_test
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM N  N   ALA A 1 1.000 2.000 3.000
ATOM C  CA  ALA A 1 2.460 2.000 3.000
ATOM C  C   ALA A 1 3.010 3.420 3.000
ATOM O  O   ALA A 1 2.240 4.380 3.000
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
    fn test_mmcif_residue_annotation() {
        let mol =
            embedded_molecule_from_mmcif(MINI_MMCIF, &IngestOptions::default()).unwrap();
        let res = mol.graph.atoms[0].residue.as_ref().unwrap();
        assert_eq!(res.name, "ALA");
        assert_eq!(res.chain_id, "A");
        assert_eq!(res.seq_num, 1);
    }

    #[test]
    fn test_mmcif_bond_inference() {
        let mol =
            embedded_molecule_from_mmcif(MINI_MMCIF, &IngestOptions::default()).unwrap();
        assert!(
            mol.graph.bonds.len() >= 2,
            "expected at least 2 bonds, got {}",
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
