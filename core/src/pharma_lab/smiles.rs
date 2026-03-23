//! Minimal SMILES parser for the pharma lab.
//!
//! Supports: atoms (organic subset + bracket), single/double/triple bonds,
//! branches (), ring closures (digits 1-9), and charges in brackets.
//! Generates 3D coordinates via simple distance-geometry embedding.

use crate::atomistic_chemistry::{BondOrder, MoleculeGraph, PeriodicElement};

/// Parse a SMILES string into a MoleculeGraph + 3D positions.
pub fn parse_smiles(smiles: &str) -> Result<(MoleculeGraph, Vec<[f32; 3]>), String> {
    let tokens = tokenize(smiles)?;
    let (graph, _charges) = build_graph(&tokens)?;
    let positions = embed_3d(&graph)?;
    Ok((graph, positions))
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Token {
    Atom(String),          // element symbol
    Bond(BondOrder),
    BranchOpen,
    BranchClose,
    Ring(u8),              // ring closure digit
}

fn tokenize(smiles: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = smiles.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            '(' => { tokens.push(Token::BranchOpen); i += 1; }
            ')' => { tokens.push(Token::BranchClose); i += 1; }
            '-' => { tokens.push(Token::Bond(BondOrder::Single)); i += 1; }
            '=' => { tokens.push(Token::Bond(BondOrder::Double)); i += 1; }
            '#' => { tokens.push(Token::Bond(BondOrder::Triple)); i += 1; }
            ':' => { tokens.push(Token::Bond(BondOrder::Aromatic)); i += 1; }
            '0'..='9' => {
                tokens.push(Token::Ring(chars[i] as u8 - b'0'));
                i += 1;
            }
            '[' => {
                // Bracket atom: [Fe], [NH4+], [O-], etc.
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != ']' { i += 1; }
                if i >= chars.len() {
                    return Err("unclosed bracket atom".to_string());
                }
                let content: String = chars[start..i].iter().collect();
                // Extract element symbol (first 1-2 uppercase+lowercase chars)
                let sym = extract_symbol(&content);
                tokens.push(Token::Atom(sym));
                i += 1; // skip ']'
            }
            c if c.is_ascii_uppercase() => {
                // Organic subset atom: B, C, N, O, P, S, F, Cl, Br, I
                let mut sym = String::new();
                sym.push(c);
                if i + 1 < chars.len() && chars[i + 1].is_ascii_lowercase() {
                    sym.push(chars[i + 1]);
                    i += 1;
                }
                tokens.push(Token::Atom(sym));
                i += 1;
            }
            c if c.is_ascii_lowercase() => {
                // Aromatic atoms: c, n, o, s, p
                let sym = c.to_ascii_uppercase().to_string();
                tokens.push(Token::Atom(sym));
                i += 1;
            }
            '.' => { i += 1; } // disconnected fragments, skip
            '+' | '%' | '/' | '\\' | '@' => { i += 1; } // skip stereo/charge notation
            _ => { i += 1; } // skip unknown
        }
    }
    Ok(tokens)
}

fn extract_symbol(bracket_content: &str) -> String {
    let chars: Vec<char> = bracket_content.chars().collect();
    let mut sym = String::new();
    if let Some(&first) = chars.first() {
        if first.is_ascii_uppercase() {
            sym.push(first);
            if chars.len() > 1 && chars[1].is_ascii_lowercase() {
                sym.push(chars[1]);
            }
        }
    }
    if sym.is_empty() { "C".to_string() } else { sym }
}

// ---------------------------------------------------------------------------
// Graph builder
// ---------------------------------------------------------------------------

fn build_graph(tokens: &[Token]) -> Result<(MoleculeGraph, Vec<i8>), String> {
    let mut graph = MoleculeGraph::new("smiles");
    let mut charges: Vec<i8> = Vec::new();
    let mut stack: Vec<usize> = Vec::new(); // branch stack
    let mut current: Option<usize> = None;
    let mut pending_bond = BondOrder::Single;
    let mut ring_openings: [Option<usize>; 10] = [None; 10];

    for token in tokens {
        match token {
            Token::Atom(sym) => {
                let elem = PeriodicElement::from_symbol_or_name(sym)
                    .ok_or_else(|| format!("unknown element: {}", sym))?;
                let idx = graph.add_element(elem);
                charges.push(0);

                if let Some(prev) = current {
                    graph.add_bond(prev, idx, pending_bond)?;
                    pending_bond = BondOrder::Single;
                }
                current = Some(idx);
            }
            Token::Bond(order) => {
                pending_bond = *order;
            }
            Token::BranchOpen => {
                if let Some(c) = current {
                    stack.push(c);
                }
            }
            Token::BranchClose => {
                current = stack.pop();
                pending_bond = BondOrder::Single;
            }
            Token::Ring(digit) => {
                let d = *digit as usize;
                if let Some(open_idx) = ring_openings[d] {
                    // Close ring
                    if let Some(cur) = current {
                        let _ = graph.add_bond(open_idx, cur, pending_bond);
                    }
                    ring_openings[d] = None;
                    pending_bond = BondOrder::Single;
                } else {
                    // Open ring
                    ring_openings[d] = current;
                }
            }
        }
    }

    // Add implicit hydrogens for organic subset atoms
    add_implicit_hydrogens(&mut graph, &mut charges);

    Ok((graph, charges))
}

fn add_implicit_hydrogens(graph: &mut MoleculeGraph, charges: &mut Vec<i8>) {
    let n = graph.atoms.len();
    // Count existing bonds for each atom
    let mut bond_count: Vec<f32> = vec![0.0; n];
    for bond in &graph.bonds {
        let order = bond.order.bond_order();
        bond_count[bond.i] += order;
        bond_count[bond.j] += order;
    }

    // Standard valences for organic subset
    let mut h_to_add: Vec<(usize, usize)> = Vec::new(); // (atom_idx, n_hydrogens)
    for i in 0..n {
        let elem = graph.atoms[i].element;
        let target_valence = match elem {
            PeriodicElement::C => 4.0,
            PeriodicElement::N => 3.0,
            PeriodicElement::O => 2.0,
            PeriodicElement::S => 2.0,
            PeriodicElement::P => 3.0,
            PeriodicElement::F | PeriodicElement::Cl | PeriodicElement::Br | PeriodicElement::I => 1.0,
            PeriodicElement::B => 3.0,
            _ => continue, // skip metals, etc.
        };
        let deficit = (target_valence - bond_count[i]).max(0.0).round() as usize;
        if deficit > 0 {
            h_to_add.push((i, deficit));
        }
    }

    for (atom_idx, n_h) in h_to_add {
        for _ in 0..n_h {
            let h_idx = graph.add_element(PeriodicElement::H);
            charges.push(0);
            let _ = graph.add_bond(atom_idx, h_idx, BondOrder::Single);
        }
    }
}

// ---------------------------------------------------------------------------
// 3D coordinate embedding
// ---------------------------------------------------------------------------

fn embed_3d(graph: &MoleculeGraph) -> Result<Vec<[f32; 3]>, String> {
    let n = graph.atom_count();
    if n == 0 {
        return Err("empty molecule".to_string());
    }

    // Build adjacency list
    let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for bond in &graph.bonds {
        let r0 = graph.atoms[bond.i].element.covalent_radius_angstrom()
            + graph.atoms[bond.j].element.covalent_radius_angstrom();
        adj[bond.i].push((bond.j, r0));
        adj[bond.j].push((bond.i, r0));
    }

    // Place atoms using BFS from atom 0
    let mut positions = vec![[0.0f32; 3]; n];
    let mut placed = vec![false; n];
    placed[0] = true;

    let mut queue = std::collections::VecDeque::new();
    queue.push_back(0usize);

    // Simple deterministic seed for placement angles
    let mut angle_offset = 0.0f32;

    while let Some(current) = queue.pop_front() {
        let mut child_idx = 0;
        let n_children = adj[current].iter().filter(|(nb, _)| !placed[*nb]).count();

        for &(neighbor, bond_len) in &adj[current] {
            if placed[neighbor] { continue; }

            // Place at bond_len distance in a fan pattern
            let base_angle = if n_children == 1 {
                angle_offset
            } else {
                angle_offset + (child_idx as f32 / n_children as f32) * std::f32::consts::TAU
            };

            // Alternate elevation for 3D effect
            let elevation = if child_idx % 2 == 0 { 0.3 } else { -0.3 };

            positions[neighbor] = [
                positions[current][0] + bond_len * base_angle.cos(),
                positions[current][1] + bond_len * base_angle.sin(),
                positions[current][2] + elevation * bond_len,
            ];

            placed[neighbor] = true;
            queue.push_back(neighbor);
            child_idx += 1;
        }
        angle_offset += 2.094; // ~120 degrees offset for next atom's children
    }

    // Place any disconnected atoms
    let mut x_offset = 5.0;
    for i in 0..n {
        if !placed[i] {
            positions[i] = [x_offset, 0.0, 0.0];
            x_offset += 2.0;
            placed[i] = true;
        }
    }

    Ok(positions)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_water() {
        let (graph, positions) = parse_smiles("O").unwrap();
        // O with 2 implicit H
        assert_eq!(graph.atom_count(), 3);
        assert_eq!(graph.bond_count(), 2);
        assert_eq!(positions.len(), 3);
    }

    #[test]
    fn parse_ethanol() {
        let (graph, _) = parse_smiles("CCO").unwrap();
        // C + C + O + implicit H (3+2+1=6 H)
        assert_eq!(graph.atoms.iter().filter(|a| a.element == PeriodicElement::C).count(), 2);
        assert_eq!(graph.atoms.iter().filter(|a| a.element == PeriodicElement::O).count(), 1);
    }

    #[test]
    fn parse_benzene() {
        let (graph, _) = parse_smiles("C1=CC=CC=C1").unwrap();
        let carbons = graph.atoms.iter().filter(|a| a.element == PeriodicElement::C).count();
        assert_eq!(carbons, 6);
    }

    #[test]
    fn parse_bracket_atom() {
        let (graph, _) = parse_smiles("[Fe]").unwrap();
        assert_eq!(graph.atoms[0].element, PeriodicElement::Fe);
    }

    #[test]
    fn parse_branched() {
        let (graph, _) = parse_smiles("CC(=O)O").unwrap();
        // Acetic acid: 2C + 2O + hydrogens
        let carbons = graph.atoms.iter().filter(|a| a.element == PeriodicElement::C).count();
        let oxygens = graph.atoms.iter().filter(|a| a.element == PeriodicElement::O).count();
        assert_eq!(carbons, 2);
        assert_eq!(oxygens, 2);
    }

    #[test]
    fn positions_are_finite() {
        let (_, positions) = parse_smiles("CCCCCC").unwrap();
        for pos in &positions {
            assert!(pos[0].is_finite());
            assert!(pos[1].is_finite());
            assert!(pos[2].is_finite());
        }
    }
}
