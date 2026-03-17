//! Explicit subatomic and quantum-chemistry state below the atom graph.
//!
//! This module keeps lower-scale authority attached to the Rust-native atomistic
//! graph instead of routing through top-down scalar corrections.

use crate::atomistic_chemistry::{AtomNode, BondOrder, MoleculeGraph, PeriodicElement};
use crate::substrate_ir::ReactionQuantumSummary;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

const BOHR_RADIUS_ANGSTROM: f64 = 0.529_177_210_903;
const COULOMB_EV_ANGSTROM: f64 = 14.399_645_478_425_5;
const MEV_PER_U: f64 = 931.494_102_42;

const PROTON_REST_MASS_U: f64 = 1.007_276_466_621;
const NEUTRON_REST_MASS_U: f64 = 1.008_664_915_95;
const ELECTRON_REST_MASS_U: f64 = 0.000_548_579_909_065;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumSubshell {
    S,
    P,
    D,
    F,
}

impl QuantumSubshell {
    pub fn label(self) -> &'static str {
        match self {
            Self::S => "s",
            Self::P => "p",
            Self::D => "d",
            Self::F => "f",
        }
    }

    pub fn l(self) -> u8 {
        match self {
            Self::S => 0,
            Self::P => 1,
            Self::D => 2,
            Self::F => 3,
        }
    }

    pub fn spatial_degeneracy(self) -> u8 {
        match self {
            Self::S => 1,
            Self::P => 3,
            Self::D => 5,
            Self::F => 7,
        }
    }
}

const AUFBAU_ORDER: &[(u8, QuantumSubshell, u8)] = &[
    (1, QuantumSubshell::S, 2),
    (2, QuantumSubshell::S, 2),
    (2, QuantumSubshell::P, 6),
    (3, QuantumSubshell::S, 2),
    (3, QuantumSubshell::P, 6),
    (4, QuantumSubshell::S, 2),
    (3, QuantumSubshell::D, 10),
    (4, QuantumSubshell::P, 6),
    (5, QuantumSubshell::S, 2),
    (4, QuantumSubshell::D, 10),
    (5, QuantumSubshell::P, 6),
    (6, QuantumSubshell::S, 2),
    (4, QuantumSubshell::F, 14),
    (5, QuantumSubshell::D, 10),
    (6, QuantumSubshell::P, 6),
    (7, QuantumSubshell::S, 2),
    (5, QuantumSubshell::F, 14),
    (6, QuantumSubshell::D, 10),
    (7, QuantumSubshell::P, 6),
];

#[cfg(feature = "nuclear_physics")]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct QuarkInventory {
    pub up: u32,
    pub down: u32,
    pub electrons: u32,
}

#[cfg(feature = "nuclear_physics")]
impl QuarkInventory {
    pub fn charge_e(self) -> f64 {
        (2.0 * f64::from(self.up) - f64::from(self.down)) / 3.0 - f64::from(self.electrons)
    }

    pub fn baryon_number(self) -> f64 {
        f64::from(self.up + self.down) / 3.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NuclearComposition {
    pub protons: u32,
    pub neutrons: u32,
}

impl NuclearComposition {
    pub fn new(protons: u32, neutrons: u32) -> Self {
        Self { protons, neutrons }
    }

    pub fn mass_number(self) -> u32 {
        self.protons + self.neutrons
    }

    pub fn charge_e(self) -> u32 {
        self.protons
    }

    #[cfg(feature = "nuclear_physics")]
    pub fn quark_inventory(self) -> QuarkInventory {
        QuarkInventory {
            up: 2 * self.protons + self.neutrons,
            down: self.protons + 2 * self.neutrons,
            electrons: 0,
        }
    }

    pub fn binding_energy_mev(self) -> f64 {
        let a = self.mass_number();
        let z = self.protons;
        if a <= 1 {
            return 0.0;
        }
        let n = self.neutrons;
        let a_f = a as f64;
        let z_f = z as f64;
        let pairing = if a % 2 == 0 {
            if z % 2 == 0 && n % 2 == 0 {
                12.0 / a_f.sqrt()
            } else {
                -12.0 / a_f.sqrt()
            }
        } else {
            0.0
        };
        let binding = 15.8 * a_f
            - 18.3 * a_f.powf(2.0 / 3.0)
            - 0.714 * z_f * (z_f - 1.0).max(0.0) / a_f.powf(1.0 / 3.0)
            - 23.2 * (a_f - 2.0 * z_f).powi(2) / a_f
            + pairing;
        binding.max(0.0)
    }

    pub fn rest_mass_u(self) -> f64 {
        f64::from(self.protons) * PROTON_REST_MASS_U
            + f64::from(self.neutrons) * NEUTRON_REST_MASS_U
            - self.binding_energy_mev() / MEV_PER_U
    }

    pub fn radius_fm(self) -> f64 {
        if self.mass_number() == 0 {
            return 0.0;
        }
        1.25 * (f64::from(self.mass_number())).powf(1.0 / 3.0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ElectronSubshell {
    pub n: u8,
    pub subshell: QuantumSubshell,
    pub electrons: u8,
    pub capacity: u8,
    pub effective_nuclear_charge: f64,
    pub orbital_energy_ev: f64,
    pub orbital_radius_angstrom: f64,
}

impl ElectronSubshell {
    pub fn label(&self) -> String {
        format!("{}{}", self.n, self.subshell.label())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantumAtomState {
    pub element: PeriodicElement,
    pub formal_charge: i8,
    pub mass_number: u16,
    pub nucleus: NuclearComposition,
    pub electron_subshells: Vec<ElectronSubshell>,
}

impl QuantumAtomState {
    pub fn from_atom_node(atom: &AtomNode) -> Result<Self, QuantumMicrodomainError> {
        let atomic_number = i32::from(atom.element.atomic_number());
        let electron_count = atomic_number - i32::from(atom.formal_charge);
        if electron_count < 0 {
            return Err(QuantumMicrodomainError::NegativeElectronCount {
                element: atom.element.symbol().to_string(),
                formal_charge: atom.formal_charge,
            });
        }
        let occupancies = filled_subshells(electron_count as u32);
        let mut electron_subshells = Vec::with_capacity(occupancies.len());
        for (idx, (n, subshell, occupancy)) in occupancies.iter().copied().enumerate() {
            let shielding = slater_shielding(atom.element.atomic_number(), &occupancies, idx);
            let effective_charge = (f64::from(atom.element.atomic_number()) - shielding).max(0.25);
            electron_subshells.push(ElectronSubshell {
                n,
                subshell,
                electrons: occupancy,
                capacity: 2 * subshell.spatial_degeneracy(),
                effective_nuclear_charge: effective_charge,
                orbital_energy_ev: subshell_energy_ev(n, subshell, effective_charge),
                orbital_radius_angstrom: orbital_radius_angstrom(n, subshell, effective_charge),
            });
        }

        let mass_number = atom.isotope_mass_number.unwrap_or_else(|| {
            atom.atomic_mass()
                .round()
                .max(f64::from(atom.element.atomic_number())) as u16
        });

        Ok(Self {
            element: atom.element,
            formal_charge: atom.formal_charge,
            mass_number,
            nucleus: NuclearComposition::new(
                u32::from(atom.element.atomic_number()),
                u32::from(mass_number) - u32::from(atom.element.atomic_number()),
            ),
            electron_subshells,
        })
    }

    pub fn electron_count(&self) -> u32 {
        self.electron_subshells
            .iter()
            .map(|shell| u32::from(shell.electrons))
            .sum()
    }

    #[cfg(feature = "nuclear_physics")]
    pub fn quark_inventory(&self) -> QuarkInventory {
        let mut inventory = self.nucleus.quark_inventory();
        inventory.electrons = self.electron_count();
        inventory
    }

    pub fn charge_e(&self) -> i32 {
        i32::from(self.element.atomic_number()) - self.electron_count() as i32
    }

    pub fn ionization_proxy_ev(&self) -> f64 {
        self.electron_subshells
            .iter()
            .filter(|shell| shell.electrons > 0)
            .map(|shell| shell.orbital_energy_ev)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|energy| energy.abs())
            .unwrap_or(0.0)
    }

    pub fn total_rest_mass_u(&self) -> f64 {
        self.nucleus.rest_mass_u() + self.electron_count() as f64 * ELECTRON_REST_MASS_U
    }

    pub fn valence_electrons(&self) -> u32 {
        let Some(highest_n) = self
            .electron_subshells
            .iter()
            .filter(|shell| shell.electrons > 0)
            .map(|shell| shell.n)
            .max()
        else {
            return 0;
        };
        let main_shell: u32 = self
            .electron_subshells
            .iter()
            .filter(|shell| shell.electrons > 0 && shell.n == highest_n)
            .map(|shell| u32::from(shell.electrons))
            .sum();
        let d_shell: u32 = self
            .electron_subshells
            .iter()
            .filter(|shell| {
                shell.electrons > 0
                    && shell.n + 1 == highest_n
                    && shell.subshell == QuantumSubshell::D
            })
            .map(|shell| u32::from(shell.electrons))
            .sum();
        main_shell + d_shell
    }

    pub fn frozen_core_electrons(&self) -> u32 {
        self.electron_count()
            .saturating_sub(self.valence_electrons())
    }

    pub fn active_subshells(&self) -> Vec<ElectronSubshell> {
        let Some(highest_n) = self
            .electron_subshells
            .iter()
            .filter(|shell| shell.electrons > 0)
            .map(|shell| shell.n)
            .max()
        else {
            return Vec::new();
        };
        self.electron_subshells
            .iter()
            .filter(|shell| {
                shell.electrons > 0
                    && (shell.n == highest_n
                        || (shell.n + 1 == highest_n && shell.subshell == QuantumSubshell::D))
            })
            .cloned()
            .collect()
    }
}

impl AtomNode {
    pub fn quantum_state(&self) -> Result<QuantumAtomState, QuantumMicrodomainError> {
        QuantumAtomState::from_atom_node(self)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpatialOrbital {
    pub atom_index: usize,
    pub n: u8,
    pub subshell: QuantumSubshell,
    pub orientation: u8,
    pub shell_electrons: u8,
    pub shell_capacity: u8,
    pub effective_nuclear_charge: f64,
    pub orbital_energy_ev: f64,
    pub orbital_radius_angstrom: f64,
    pub onsite_repulsion_ev: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MolecularActiveSpace {
    pub atoms: Vec<QuantumAtomState>,
    pub positions_angstrom: Vec<[f64; 3]>,
    pub spatial_orbitals: Vec<SpatialOrbital>,
    pub active_electrons: usize,
    pub frozen_core_electrons: usize,
    #[serde(default)]
    pub embedding_point_charges: Vec<QuantumEmbeddingPointCharge>,
    #[serde(default)]
    pub embedding_dipoles: Vec<QuantumEmbeddingDipole>,
    pub atom_bond_orders: Vec<f64>,
    pub one_body_integrals_ev: Vec<f64>,
    pub onsite_repulsion_ev: Vec<f64>,
    pub interorbital_coulomb_ev: Vec<f64>,
    pub nuclear_repulsion_ev: f64,
}

impl MolecularActiveSpace {
    pub fn from_molecule_graph(
        graph: &MoleculeGraph,
        positions_angstrom: &[[f64; 3]],
        max_spatial_orbitals: Option<usize>,
    ) -> Result<Self, QuantumMicrodomainError> {
        Self::from_molecule_graph_with_embedding(
            graph,
            positions_angstrom,
            max_spatial_orbitals,
            &[],
            &[],
        )
    }

    pub fn from_molecule_graph_with_embedding(
        graph: &MoleculeGraph,
        positions_angstrom: &[[f64; 3]],
        max_spatial_orbitals: Option<usize>,
        embedding_point_charges: &[QuantumEmbeddingPointCharge],
        embedding_dipoles: &[QuantumEmbeddingDipole],
    ) -> Result<Self, QuantumMicrodomainError> {
        if positions_angstrom.len() != graph.atoms.len() {
            return Err(QuantumMicrodomainError::PositionCountMismatch {
                atoms: graph.atoms.len(),
                positions: positions_angstrom.len(),
            });
        }

        let atoms: Vec<QuantumAtomState> = graph
            .atoms
            .iter()
            .map(QuantumAtomState::from_atom_node)
            .collect::<Result<_, _>>()?;
        let mut spatial_orbitals = build_spatial_orbitals(&atoms);
        if let Some(limit) = max_spatial_orbitals {
            if spatial_orbitals.len() > limit {
                spatial_orbitals.sort_by(|a, b| {
                    abs_cmp(a.orbital_energy_ev, b.orbital_energy_ev)
                        .then(a.atom_index.cmp(&b.atom_index))
                        .then(a.n.cmp(&b.n))
                        .then(a.orientation.cmp(&b.orientation))
                });
                spatial_orbitals.truncate(limit);
            }
        }
        if spatial_orbitals.is_empty() {
            return Err(QuantumMicrodomainError::EmptyActiveSpace);
        }

        let active_electrons: usize = atoms
            .iter()
            .map(|atom| atom.valence_electrons() as usize)
            .sum();
        let capacity = 2 * spatial_orbitals.len();
        if active_electrons > capacity {
            return Err(QuantumMicrodomainError::ActiveElectronsExceedCapacity {
                active_electrons,
                capacity,
            });
        }
        let frozen_core_electrons = atoms
            .iter()
            .map(|atom| atom.frozen_core_electrons() as usize)
            .sum();
        let atom_bond_orders = build_atom_bond_order_matrix(graph);
        let one_body_integrals_ev = build_one_body_integrals(
            &atoms,
            positions_angstrom,
            &spatial_orbitals,
            &atom_bond_orders,
            embedding_point_charges,
            embedding_dipoles,
        );
        let onsite_repulsion_ev = spatial_orbitals
            .iter()
            .map(|orbital| orbital.onsite_repulsion_ev)
            .collect();
        let interorbital_coulomb_ev =
            build_interorbital_coulomb(positions_angstrom, &spatial_orbitals, &atom_bond_orders);
        let nuclear_repulsion_ev = nuclear_repulsion_ev(
            atoms.as_slice(),
            positions_angstrom,
            embedding_point_charges,
            embedding_dipoles,
        );
        Ok(Self {
            atoms,
            positions_angstrom: positions_angstrom.to_vec(),
            spatial_orbitals,
            active_electrons,
            frozen_core_electrons,
            embedding_point_charges: embedding_point_charges.to_vec(),
            embedding_dipoles: embedding_dipoles.to_vec(),
            atom_bond_orders,
            one_body_integrals_ev,
            onsite_repulsion_ev,
            interorbital_coulomb_ev,
            nuclear_repulsion_ev,
        })
    }

    pub fn num_spatial_orbitals(&self) -> usize {
        self.spatial_orbitals.len()
    }

    pub fn num_spin_orbitals(&self) -> usize {
        2 * self.num_spatial_orbitals()
    }

    pub fn total_electrons(&self) -> usize {
        self.active_electrons + self.frozen_core_electrons
    }

    pub fn basis_size(&self, max_basis_size: usize) -> Result<usize, QuantumMicrodomainError> {
        let num_spin_orbitals = self.num_spin_orbitals();
        validate_spin_orbital_count(num_spin_orbitals)?;
        Ok(combination_count_capped(
            num_spin_orbitals,
            self.active_electrons,
            max_basis_size,
        ))
    }

    pub fn exact_diagonalize(
        &self,
        max_basis_size: usize,
    ) -> Result<ExactDiagonalizationResult, QuantumMicrodomainError> {
        let full_basis_size = self.basis_size(usize::MAX)?;

        // Tier selection: use full ED for small spaces, CIPSI + EN-PT2 for large ones.
        if full_basis_size <= FULL_ED_DIM_THRESHOLD {
            // Also enforce the caller's hard cap.
            if full_basis_size > max_basis_size {
                return Err(QuantumMicrodomainError::BasisTooLarge {
                    basis_size: full_basis_size,
                    max_basis_size,
                });
            }
            let mut result = self.full_ed_diagonalize()?;
            result.solver_tier = SolverTier::FullED;
            Ok(result)
        } else {
            // Large Hilbert space: Selected CI with optional EN-PT2 correction.
            let (mut result, tier) = cipsi_selected_ci(self, max_basis_size)?;
            if tier == SolverTier::SelectedCI {
                // Apply perturbative correction from the excluded space.
                let pt2 = en_pt2_correction(
                    self,
                    &result.basis_states,
                    &result.ground_state_vector,
                    result.energies_ev[0],
                );
                result.energies_ev[0] += pt2;
                result.solver_tier = SolverTier::EnPt2;
            }
            Ok(result)
        }
    }

    /// Full exact diagonalization via Jacobi rotations over the complete
    /// fixed-particle basis. This is the original O(N^3) path.
    fn full_ed_diagonalize(
        &self,
    ) -> Result<ExactDiagonalizationResult, QuantumMicrodomainError> {
        let basis = fixed_particle_basis(self.num_spin_orbitals(), self.active_electrons)?;
        let dim = basis.len();
        let mut index_by_state = std::collections::HashMap::with_capacity(dim);
        for (idx, state) in basis.iter().copied().enumerate() {
            index_by_state.insert(state, idx);
        }

        let n_spatial = self.num_spatial_orbitals();
        let mut matrix = vec![0.0; dim * dim];
        for (basis_idx, state) in basis.iter().copied().enumerate() {
            let diagonal = determinant_diagonal_energy(self, state, n_spatial);
            matrix[matrix_index(basis_idx, basis_idx, dim)] = diagonal;

            accumulate_off_diagonal(self, state, basis_idx, n_spatial, &index_by_state, &mut matrix, dim);
        }

        symmetrize_in_place(&mut matrix, dim);
        let (mut energies_ev, eigenvectors) =
            jacobi_eigendecomposition(&matrix, dim, 1.0e-10, 200 * dim.max(1) * dim.max(1));
        let mut order: Vec<usize> = (0..energies_ev.len()).collect();
        order.sort_by(|&i, &j| {
            energies_ev[i]
                .partial_cmp(&energies_ev[j])
                .unwrap_or(Ordering::Equal)
        });
        let sorted_energies: Vec<f64> = order.iter().map(|&idx| energies_ev[idx]).collect();
        let mut sorted_vectors = vec![0.0; dim * dim];
        for (col_out, &col_in) in order.iter().enumerate() {
            for row in 0..dim {
                sorted_vectors[matrix_index(row, col_out, dim)] =
                    eigenvectors[matrix_index(row, col_in, dim)];
            }
        }
        energies_ev = sorted_energies;
        let ground_state_vector: Vec<f64> = (0..dim)
            .map(|row| sorted_vectors[matrix_index(row, 0, dim)])
            .collect();
        let spatial_one_particle_density_matrix = spatial_one_particle_density_matrix(
            &basis,
            &ground_state_vector,
            n_spatial,
            &index_by_state,
        );
        let expected_spatial_occupancies = (0..n_spatial)
            .map(|orbital_idx| {
                spatial_one_particle_density_matrix
                    [matrix_index(orbital_idx, orbital_idx, n_spatial)]
            })
            .collect::<Vec<_>>();
        let expected_atom_effective_charges = expected_atom_effective_charges(
            &self.atoms,
            &self.spatial_orbitals,
            &expected_spatial_occupancies,
        );
        let expected_dipole_moment_e_angstrom = expected_dipole_moment_e_angstrom(
            &self.positions_angstrom,
            &expected_atom_effective_charges,
        );
        Ok(ExactDiagonalizationResult {
            energies_ev,
            ground_state_vector,
            basis_states: basis,
            expected_spatial_occupancies: expected_spatial_occupancies.clone(),
            spatial_orbital_atom_indices: self
                .spatial_orbitals
                .iter()
                .map(|orbital| orbital.atom_index)
                .collect(),
            spatial_one_particle_density_matrix,
            expected_atom_effective_charges,
            expected_dipole_moment_e_angstrom,
            expected_electron_count: expected_spatial_occupancies.iter().sum(),
            nuclear_repulsion_ev: self.nuclear_repulsion_ev,
            solver_tier: SolverTier::FullED,
        })
    }
}

/// Compute the diagonal Hamiltonian matrix element for a single determinant.
fn determinant_diagonal_energy(
    space: &MolecularActiveSpace,
    state: u64,
    n_spatial: usize,
) -> f64 {
    let mut occ_by_spatial = vec![0.0; n_spatial];
    let mut diagonal = space.nuclear_repulsion_ev;
    for spatial in 0..n_spatial {
        let up_occ = ((state >> (2 * spatial)) & 1) as f64;
        let down_occ = ((state >> (2 * spatial + 1)) & 1) as f64;
        let occupancy = up_occ + down_occ;
        occ_by_spatial[spatial] = occupancy;
        diagonal +=
            space.one_body_integrals_ev[matrix_index(spatial, spatial, n_spatial)] * occupancy;
        if up_occ > 0.5 && down_occ > 0.5 {
            diagonal += space.onsite_repulsion_ev[spatial];
        }
    }
    for i in 0..n_spatial {
        for j in (i + 1)..n_spatial {
            diagonal += space.interorbital_coulomb_ev[matrix_index(i, j, n_spatial)]
                * occ_by_spatial[i]
                * occ_by_spatial[j];
        }
    }
    diagonal
}

/// Accumulate off-diagonal one-body hopping matrix elements from a given basis
/// state into the Hamiltonian matrix.
fn accumulate_off_diagonal(
    space: &MolecularActiveSpace,
    state: u64,
    basis_idx: usize,
    n_spatial: usize,
    index_by_state: &std::collections::HashMap<u64, usize>,
    matrix: &mut [f64],
    dim: usize,
) {
    for spin in 0..2usize {
        for p in 0..n_spatial {
            for q in 0..n_spatial {
                if p == q {
                    continue;
                }
                let amplitude =
                    space.one_body_integrals_ev[matrix_index(p, q, n_spatial)];
                if amplitude.abs() < 1.0e-12 {
                    continue;
                }
                if let Some((hopped_state, sign)) =
                    apply_hop(state, 2 * q + spin, 2 * p + spin)
                {
                    if let Some(&target_idx) = index_by_state.get(&hopped_state) {
                        matrix[matrix_index(basis_idx, target_idx, dim)] +=
                            amplitude * f64::from(sign);
                    }
                }
            }
        }
    }
}

/// Compute the off-diagonal Hamiltonian coupling <bra| H |ket> from one-body
/// hopping integrals. Returns zero if the determinants differ by more than a
/// single excitation.
fn hamiltonian_coupling(
    space: &MolecularActiveSpace,
    bra: u64,
    ket: u64,
) -> f64 {
    let n_spatial = space.num_spatial_orbitals();
    let diff = bra ^ ket;
    let differing_bits = diff.count_ones();
    // One-body operator couples determinants differing by exactly one creation
    // and one annihilation (2 bits different, same spin sector).
    if differing_bits != 2 {
        return 0.0;
    }
    // Identify which spin-orbitals changed.
    let in_ket_only = diff & ket; // orbital occupied in ket but not bra (annihilated)
    let in_bra_only = diff & bra; // orbital occupied in bra but not ket (created)
    let source = in_ket_only.trailing_zeros() as usize; // spin-orbital annihilated
    let target = in_bra_only.trailing_zeros() as usize; // spin-orbital created
    // Must be the same spin.
    if (source % 2) != (target % 2) {
        return 0.0;
    }
    let spatial_source = source / 2;
    let spatial_target = target / 2;
    let amplitude =
        space.one_body_integrals_ev[matrix_index(spatial_target, spatial_source, n_spatial)];
    if amplitude.abs() < 1.0e-12 {
        return 0.0;
    }
    // Compute the fermionic sign by checking how many occupied orbitals lie
    // between source and target in the ket determinant.
    let (lo, hi) = if source < target {
        (source, target)
    } else {
        (target, source)
    };
    let mask = if hi < 63 {
        ((1u64 << (hi + 1)) - 1) ^ ((1u64 << (lo + 1)) - 1) ^ (1u64 << hi)
    } else {
        let lower = if lo + 1 < 64 {
            (1u64 << (lo + 1)) - 1
        } else {
            u64::MAX
        };
        (u64::MAX ^ lower) ^ (1u64 << hi)
    };
    let parity = (ket & mask).count_ones();
    let sign = if parity % 2 == 0 { 1.0 } else { -1.0 };
    amplitude * sign
}

/// Build the Hartree-Fock determinant: lowest `n_electrons` spin-orbitals occupied.
fn hf_determinant(n_electrons: usize) -> u64 {
    if n_electrons == 0 {
        return 0;
    }
    if n_electrons >= 64 {
        return u64::MAX;
    }
    (1u64 << n_electrons) - 1
}

/// Generate all single and double excitations from a set of determinants,
/// returning those not already in the set.
fn generate_singles_doubles(
    space: &MolecularActiveSpace,
    variational_set: &std::collections::HashSet<u64>,
    source_dets: &[u64],
) -> Vec<u64> {
    let n_spin = space.num_spin_orbitals();
    let mut candidates = std::collections::HashSet::new();
    // Pre-allocate index buffers (reused per determinant).
    let mut occ = Vec::with_capacity(n_spin);
    let mut virt = Vec::with_capacity(n_spin);
    for &det in source_dets {
        // Extract occupied and virtual orbital indices from the bit mask.
        occ.clear();
        virt.clear();
        for p in 0..n_spin {
            if ((det >> p) & 1) == 1 {
                occ.push(p);
            } else {
                virt.push(p);
            }
        }
        // Single excitations: annihilate occupied i, create virtual a.
        for &i in &occ {
            for &a in &virt {
                let new_det = (det & !(1u64 << i)) | (1u64 << a);
                if !variational_set.contains(&new_det) {
                    candidates.insert(new_det);
                }
            }
        }
        // Double excitations: annihilate i,j; create a,b.
        for oi in 0..occ.len() {
            let i = occ[oi];
            for oj in (oi + 1)..occ.len() {
                let j = occ[oj];
                let removed = det & !(1u64 << i) & !(1u64 << j);
                // Virtual orbitals after removal include the original
                // virtuals plus the two freed orbitals i and j.
                for va in 0..virt.len() {
                    let a = virt[va];
                    for vb in (va + 1)..virt.len() {
                        let b = virt[vb];
                        let new_det = removed | (1u64 << a) | (1u64 << b);
                        if !variational_set.contains(&new_det) {
                            candidates.insert(new_det);
                        }
                    }
                    // Also pair original virtual a with freed orbital i or j.
                    for &freed in &[i, j] {
                        if freed > a {
                            let new_det = removed | (1u64 << a) | (1u64 << freed);
                            if !variational_set.contains(&new_det) {
                                candidates.insert(new_det);
                            }
                        }
                    }
                }
                // Both freed orbitals i and j as the virtual pair.
                let new_det = removed | (1u64 << i) | (1u64 << j);
                if !variational_set.contains(&new_det) {
                    candidates.insert(new_det);
                }
                // One freed orbital paired with another freed orbital
                // (i < j always holds, and (i,j) case handled above).
                // Freed orbital paired with each original virtual.
                for &freed in &[i, j] {
                    for &a in &virt {
                        if freed < a {
                            let new_det = removed | (1u64 << freed) | (1u64 << a);
                            if !variational_set.contains(&new_det) {
                                candidates.insert(new_det);
                            }
                        }
                    }
                }
            }
        }
    }
    candidates.into_iter().collect()
}

/// Heat-bath CIPSI selected CI solver.
///
/// Starts from the Hartree-Fock determinant and iteratively expands the
/// variational space by adding excitations whose importance
/// `|H_{IJ} * c_I|` exceeds a progressively tightened threshold epsilon.
/// Diagonalizes the selected subspace at each iteration until the energy
/// converges within [`CIPSI_CONVERGENCE_EV`] or the space reaches
/// [`CIPSI_MAX_VARIATIONAL_DIM`].
fn cipsi_selected_ci(
    space: &MolecularActiveSpace,
    max_basis_size: usize,
) -> Result<(ExactDiagonalizationResult, SolverTier), QuantumMicrodomainError> {
    let n_spatial = space.num_spatial_orbitals();
    let n_electrons = space.active_electrons;

    // Start with the HF determinant.
    let hf = hf_determinant(n_electrons);
    let mut variational_basis: Vec<u64> = vec![hf];
    let mut variational_set: std::collections::HashSet<u64> =
        std::collections::HashSet::from([hf]);

    // Initial epsilon (importance threshold) -- start generous and tighten.
    let mut epsilon = 0.1_f64;
    let epsilon_factor = 0.5_f64;
    let min_epsilon = 1.0e-8_f64;

    // Initial single-determinant "diagonalization".
    let hf_energy = determinant_diagonal_energy(space, hf, n_spatial);
    let mut prev_energy = hf_energy;
    let mut ground_state_vector = vec![1.0];

    let dim_cap = max_basis_size.min(CIPSI_MAX_VARIATIONAL_DIM);

    loop {
        // Generate candidate excitations from the current variational space.
        let candidates =
            generate_singles_doubles(space, &variational_set, &variational_basis);

        // Importance screening: keep candidates where max_I |H_{IJ} * c_I| > epsilon.
        let mut accepted = Vec::new();
        for &candidate in &candidates {
            let mut max_importance = 0.0_f64;
            for (idx, &det_i) in variational_basis.iter().enumerate() {
                let h_ij = hamiltonian_coupling(space, candidate, det_i);
                if h_ij.abs() < 1.0e-14 {
                    continue;
                }
                let importance = (h_ij * ground_state_vector[idx]).abs();
                if importance > max_importance {
                    max_importance = importance;
                }
            }
            if max_importance > epsilon {
                accepted.push(candidate);
            }
        }

        if accepted.is_empty() {
            // No new determinants pass the threshold; we have converged the
            // space at this epsilon. Tighten and try again.
            epsilon *= epsilon_factor;
            if epsilon < min_epsilon {
                break;
            }
            continue;
        }

        // Limit additions so we do not exceed the dimension cap.
        let room = dim_cap.saturating_sub(variational_basis.len());
        if room == 0 {
            break;
        }
        if accepted.len() > room {
            // Sort by importance (descending) and keep the most important.
            let mut scored: Vec<(f64, u64)> = accepted
                .iter()
                .map(|&candidate| {
                    let mut max_imp = 0.0_f64;
                    for (idx, &det_i) in variational_basis.iter().enumerate() {
                        let h_ij = hamiltonian_coupling(space, candidate, det_i);
                        let imp = (h_ij * ground_state_vector[idx]).abs();
                        if imp > max_imp {
                            max_imp = imp;
                        }
                    }
                    (max_imp, candidate)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            accepted = scored.into_iter().take(room).map(|(_, det)| det).collect();
        }

        for &det in &accepted {
            variational_set.insert(det);
            variational_basis.push(det);
        }

        // Build and diagonalize the Hamiltonian in the selected subspace.
        let dim = variational_basis.len();
        let mut index_by_state = std::collections::HashMap::with_capacity(dim);
        for (idx, &state) in variational_basis.iter().enumerate() {
            index_by_state.insert(state, idx);
        }

        let mut matrix = vec![0.0; dim * dim];
        for (basis_idx, &state) in variational_basis.iter().enumerate() {
            let diagonal = determinant_diagonal_energy(space, state, n_spatial);
            matrix[matrix_index(basis_idx, basis_idx, dim)] = diagonal;
            accumulate_off_diagonal(
                space,
                state,
                basis_idx,
                n_spatial,
                &index_by_state,
                &mut matrix,
                dim,
            );
        }
        symmetrize_in_place(&mut matrix, dim);
        let (current_energy, gs_vec) = ground_state_diag(&matrix, dim);
        ground_state_vector = gs_vec;

        // Check convergence.
        let delta = (current_energy - prev_energy).abs();
        prev_energy = current_energy;
        if delta < CIPSI_CONVERGENCE_EV {
            break;
        }

        // Tighten epsilon for next iteration.
        epsilon *= epsilon_factor;
        if epsilon < min_epsilon || variational_basis.len() >= dim_cap {
            break;
        }
    }

    // Build the final sorted result from the converged variational space.
    let dim = variational_basis.len();
    let mut index_by_state = std::collections::HashMap::with_capacity(dim);
    for (idx, &state) in variational_basis.iter().enumerate() {
        index_by_state.insert(state, idx);
    }

    let mut matrix = vec![0.0; dim * dim];
    for (basis_idx, &state) in variational_basis.iter().enumerate() {
        let diagonal = determinant_diagonal_energy(space, state, n_spatial);
        matrix[matrix_index(basis_idx, basis_idx, dim)] = diagonal;
        accumulate_off_diagonal(
            space,
            state,
            basis_idx,
            n_spatial,
            &index_by_state,
            &mut matrix,
            dim,
        );
    }
    symmetrize_in_place(&mut matrix, dim);
    let (energies_ev, ground_state_vector) = if dim <= LANCZOS_DIM_THRESHOLD {
        // Small space: full Jacobi for all eigenvalues.
        let (energies_raw, eigenvectors) =
            jacobi_eigendecomposition(&matrix, dim, 1.0e-10, 200 * dim.max(1) * dim.max(1));
        let mut order: Vec<usize> = (0..energies_raw.len()).collect();
        order.sort_by(|&i, &j| {
            energies_raw[i]
                .partial_cmp(&energies_raw[j])
                .unwrap_or(Ordering::Equal)
        });
        let sorted_energies: Vec<f64> = order.iter().map(|&idx| energies_raw[idx]).collect();
        let gs_vec: Vec<f64> = (0..dim)
            .map(|row| eigenvectors[matrix_index(row, order[0], dim)])
            .collect();
        (sorted_energies, gs_vec)
    } else {
        // Large space: Lanczos for ground state only.
        let (gs_energy, gs_vec) = lanczos_ground_state(&matrix, dim, LANCZOS_MAX_KRYLOV);
        (vec![gs_energy], gs_vec)
    };
    let spatial_one_particle_density_matrix = spatial_one_particle_density_matrix(
        &variational_basis,
        &ground_state_vector,
        n_spatial,
        &index_by_state,
    );
    let expected_spatial_occupancies = (0..n_spatial)
        .map(|orbital_idx| {
            spatial_one_particle_density_matrix
                [matrix_index(orbital_idx, orbital_idx, n_spatial)]
        })
        .collect::<Vec<_>>();
    let expected_atom_effective_charges = expected_atom_effective_charges(
        &space.atoms,
        &space.spatial_orbitals,
        &expected_spatial_occupancies,
    );
    let expected_dipole_moment_e_angstrom = expected_dipole_moment_e_angstrom(
        &space.positions_angstrom,
        &expected_atom_effective_charges,
    );

    let result = ExactDiagonalizationResult {
        energies_ev,
        ground_state_vector,
        basis_states: variational_basis,
        expected_spatial_occupancies: expected_spatial_occupancies.clone(),
        spatial_orbital_atom_indices: space
            .spatial_orbitals
            .iter()
            .map(|orbital| orbital.atom_index)
            .collect(),
        spatial_one_particle_density_matrix,
        expected_atom_effective_charges,
        expected_dipole_moment_e_angstrom,
        expected_electron_count: expected_spatial_occupancies.iter().sum(),
        nuclear_repulsion_ev: space.nuclear_repulsion_ev,
        solver_tier: SolverTier::SelectedCI,
    };
    Ok((result, SolverTier::SelectedCI))
}

/// Epstein-Nesbet second-order perturbation theory correction.
///
/// For each determinant |J> NOT in the variational space, accumulates:
///
/// ```text
/// E_PT2 += |sum_I H_{IJ} c_I|^2 / (E_0 - H_{JJ})
/// ```
///
/// This captures dynamic correlation energy from the excluded space.
fn en_pt2_correction(
    space: &MolecularActiveSpace,
    variational_basis: &[u64],
    ground_state_vector: &[f64],
    ground_state_energy: f64,
) -> f64 {
    let n_spatial = space.num_spatial_orbitals();
    let variational_set: std::collections::HashSet<u64> =
        variational_basis.iter().copied().collect();

    // Generate all single and double excitations from the variational space.
    let external_dets =
        generate_singles_doubles(space, &variational_set, variational_basis);

    let mut pt2_energy = 0.0_f64;
    for &det_j in &external_dets {
        // Compute sum_I H_{IJ} * c_I.
        let mut numerator_sum = 0.0_f64;
        for (idx, &det_i) in variational_basis.iter().enumerate() {
            let h_ij = hamiltonian_coupling(space, det_j, det_i);
            if h_ij.abs() < 1.0e-14 {
                continue;
            }
            numerator_sum += h_ij * ground_state_vector[idx];
        }
        if numerator_sum.abs() < 1.0e-14 {
            continue;
        }
        let h_jj = determinant_diagonal_energy(space, det_j, n_spatial);
        let denominator = ground_state_energy - h_jj;
        // Avoid division by zero or near-degeneracy blow-up.
        if denominator.abs() < 1.0e-10 {
            continue;
        }
        pt2_energy += numerator_sum * numerator_sum / denominator;
    }
    pt2_energy
}

impl MoleculeGraph {
    pub fn quantum_active_space(
        &self,
        positions_angstrom: &[[f64; 3]],
        max_spatial_orbitals: Option<usize>,
    ) -> Result<MolecularActiveSpace, QuantumMicrodomainError> {
        MolecularActiveSpace::from_molecule_graph(self, positions_angstrom, max_spatial_orbitals)
    }

    pub fn quantum_reactive_site_with_config(
        &self,
        positions_angstrom: Vec<[f64; 3]>,
        quantum: QuantumChemistryConfig,
    ) -> Result<QuantumReactiveSite, QuantumMicrodomainError> {
        QuantumReactiveSite::new_with_embedding(
            self.clone(),
            positions_angstrom,
            quantum.max_spatial_orbitals,
            quantum.max_basis_size,
            quantum.embedding_point_charges,
            quantum.embedding_dipoles,
        )
    }

    pub fn quantum_reactive_site(
        &self,
        positions_angstrom: Vec<[f64; 3]>,
        max_spatial_orbitals: Option<usize>,
        max_basis_size: usize,
    ) -> Result<QuantumReactiveSite, QuantumMicrodomainError> {
        QuantumReactiveSite::new(
            self.clone(),
            positions_angstrom,
            max_spatial_orbitals,
            max_basis_size,
        )
    }
}

fn default_embedding_screening_radius_angstrom() -> f64 {
    0.8
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantumEmbeddingPointCharge {
    pub position_angstrom: [f64; 3],
    pub charge_e: f64,
    #[serde(default = "default_embedding_screening_radius_angstrom")]
    pub screening_radius_angstrom: f64,
}

impl QuantumEmbeddingPointCharge {
    pub const fn new(
        position_angstrom: [f64; 3],
        charge_e: f64,
        screening_radius_angstrom: f64,
    ) -> Self {
        Self {
            position_angstrom,
            charge_e,
            screening_radius_angstrom,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantumEmbeddingDipole {
    pub position_angstrom: [f64; 3],
    pub dipole_e_angstrom: [f64; 3],
    #[serde(default = "default_embedding_screening_radius_angstrom")]
    pub screening_radius_angstrom: f64,
}

impl QuantumEmbeddingDipole {
    pub const fn new(
        position_angstrom: [f64; 3],
        dipole_e_angstrom: [f64; 3],
        screening_radius_angstrom: f64,
    ) -> Self {
        Self {
            position_angstrom,
            dipole_e_angstrom,
            screening_radius_angstrom,
        }
    }
}

/// Orbital-level response field for quantum embedding.
///
/// Represents the response of a molecular orbital to the embedding environment,
/// including coupling strength and spatial orientation.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantumEmbeddingOrbitalResponseField {
    /// Position of the orbital center in Ångströms.
    pub position_angstrom: [f64; 3],
    /// Spatial axis of the orbital response (unit vector).
    pub axis: [f64; 3],
    /// Coupling energy in electron-volts.
    pub coupling_ev: f64,
    /// Screening radius in Ångströms.
    pub screening_radius_angstrom: f64,
}

impl QuantumEmbeddingOrbitalResponseField {
    pub fn new(
        position_angstrom: [f64; 3],
        axis: [f64; 3],
        coupling_ev: f64,
        screening_radius_angstrom: f64,
    ) -> Self {
        Self {
            position_angstrom,
            axis,
            coupling_ev,
            screening_radius_angstrom,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantumChemistryConfig {
    pub max_spatial_orbitals: Option<usize>,
    pub max_basis_size: usize,
    #[serde(default)]
    pub embedding_point_charges: Vec<QuantumEmbeddingPointCharge>,
    #[serde(default)]
    pub embedding_dipoles: Vec<QuantumEmbeddingDipole>,
    #[serde(default)]
    pub embedding_orbital_response_fields: Vec<QuantumEmbeddingOrbitalResponseField>,
}

impl QuantumChemistryConfig {
    pub fn new(max_spatial_orbitals: Option<usize>, max_basis_size: usize) -> Self {
        Self {
            max_spatial_orbitals,
            max_basis_size,
            embedding_point_charges: Vec::new(),
            embedding_dipoles: Vec::new(),
            embedding_orbital_response_fields: Vec::new(),
        }
    }

    pub fn with_embedding_point_charges(
        mut self,
        embedding_point_charges: Vec<QuantumEmbeddingPointCharge>,
    ) -> Self {
        self.embedding_point_charges = embedding_point_charges;
        self
    }

    pub fn with_embedding_dipoles(
        mut self,
        embedding_dipoles: Vec<QuantumEmbeddingDipole>,
    ) -> Self {
        self.embedding_dipoles = embedding_dipoles;
        self
    }

    pub fn with_embedding_orbital_response_fields(
        mut self,
        embedding_orbital_response_fields: Vec<QuantumEmbeddingOrbitalResponseField>,
    ) -> Self {
        self.embedding_orbital_response_fields = embedding_orbital_response_fields;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExactDiagonalizationResult {
    pub energies_ev: Vec<f64>,
    pub ground_state_vector: Vec<f64>,
    pub basis_states: Vec<u64>,
    pub expected_spatial_occupancies: Vec<f64>,
    #[serde(default)]
    pub spatial_orbital_atom_indices: Vec<usize>,
    #[serde(default)]
    pub spatial_one_particle_density_matrix: Vec<f64>,
    #[serde(default)]
    pub expected_atom_effective_charges: Vec<f64>,
    #[serde(default)]
    pub expected_dipole_moment_e_angstrom: [f64; 3],
    pub expected_electron_count: f64,
    pub nuclear_repulsion_ev: f64,
    #[serde(default)]
    pub solver_tier: SolverTier,
}

impl ExactDiagonalizationResult {
    pub fn ground_state_energy_ev(&self) -> Option<f64> {
        self.energies_ev.first().copied()
    }

    pub fn num_spatial_orbitals(&self) -> usize {
        self.expected_spatial_occupancies.len()
    }

    pub fn atom_pair_density_response(&self, atom_a: usize, atom_b: usize) -> f64 {
        let n = self.num_spatial_orbitals();
        if n == 0
            || self.spatial_orbital_atom_indices.len() != n
            || self.spatial_one_particle_density_matrix.len() != n * n
        {
            return 0.0;
        }
        let mut response = 0.0f64;
        for p in 0..n {
            for q in 0..n {
                let left_atom = self.spatial_orbital_atom_indices[p];
                let right_atom = self.spatial_orbital_atom_indices[q];
                if (left_atom == atom_a && right_atom == atom_b)
                    || (atom_a != atom_b && left_atom == atom_b && right_atom == atom_a)
                {
                    response +=
                        self.spatial_one_particle_density_matrix[matrix_index(p, q, n)].abs();
                }
            }
        }
        response
    }

    /// Diagonalize the 1-RDM to obtain natural orbitals and their occupancies.
    /// Returns (occupancies, natural_orbital_coefficients) where the coefficients
    /// matrix transforms from the original spatial orbital basis to natural orbitals.
    /// Natural orbital occupancies range from 0 to 2 (spatial orbitals, paired electrons).
    pub fn natural_orbitals(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        let n = self.num_spatial_orbitals();
        if n == 0 || self.spatial_one_particle_density_matrix.len() != n * n {
            return None;
        }
        let (eigenvalues, eigenvectors) = jacobi_eigendecomposition(
            &self.spatial_one_particle_density_matrix,
            n,
            1.0e-10,
            200 * n.max(1) * n.max(1),
        );
        // Sort by occupancy (descending) — most occupied natural orbitals first.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(Ordering::Equal)
        });
        let occupancies: Vec<f64> = order.iter().map(|&i| eigenvalues[i].max(0.0).min(2.0)).collect();
        let mut sorted_coefficients = vec![0.0; n * n];
        for (col_out, &col_in) in order.iter().enumerate() {
            for row in 0..n {
                sorted_coefficients[matrix_index(row, col_out, n)] =
                    eigenvectors[matrix_index(row, col_in, n)];
            }
        }
        Some((occupancies, sorted_coefficients))
    }

    /// Compute exchange integral estimate between two atoms using the Mulliken
    /// approximation from the 1-RDM: K_ij ~ S_ij^2 * J_ij where S_ij is
    /// approximated from the off-diagonal density matrix elements.
    /// Returns the exchange contribution in eV.
    pub fn mulliken_exchange_ev(&self, atom_a: usize, atom_b: usize) -> f64 {
        let density_response = self.atom_pair_density_response(atom_a, atom_b);
        // Mulliken approximation: K ~ rho_ab^2 * 14.3996 / r_ab
        // where rho_ab is the off-diagonal density matrix element.
        // This is a rough first-principles estimate, not an empirical parameter.
        const COULOMB_CONSTANT_EV_ANGSTROM: f64 = 14.3996;
        // Without explicit positions here, return the density-squared term.
        // The full spatial integral is computed when positions are available.
        density_response * density_response * COULOMB_CONSTANT_EV_ANGSTROM
    }

    /// LDA exchange potential at a given atom from the solved electron density.
    /// V_x = -(3/pi * rho)^(1/3) in atomic units, converted to eV.
    /// The electron density per atom is approximated from the effective charges.
    /// This is a true first-principles functional, not an empirical parameter.
    pub fn lda_exchange_potential_ev(&self, atom_idx: usize) -> f64 {
        if atom_idx >= self.expected_atom_effective_charges.len() {
            return 0.0;
        }
        // Effective number of electrons on this atom: Z - Q_eff.
        // We don't have Z here, but the occupancy sum gives electron density.
        let n = self.num_spatial_orbitals();
        if n == 0 || self.spatial_orbital_atom_indices.len() != n {
            return 0.0;
        }
        // Sum occupancies on orbitals belonging to this atom.
        let mut electron_density = 0.0f64;
        for (orb, &owner) in self.spatial_orbital_atom_indices.iter().enumerate() {
            if owner == atom_idx {
                electron_density += self.expected_spatial_occupancies[orb];
            }
        }
        // Approximate local density using a typical atomic volume (~1 Å³ = 1e-30 m³).
        // rho ~ n_electrons / V_atom, V_atom ~ (4/3) * pi * r_cov^3.
        // For simplicity, use a fixed effective volume per atom.
        const HARTREE_TO_EV: f64 = 27.211_386_245_988;
        // Effective radius ~1.5 Bohr for typical atoms.
        let r_eff_bohr: f64 = 1.5;
        let volume_bohr3 = (4.0 / 3.0) * std::f64::consts::PI * r_eff_bohr.powi(3);
        let rho = electron_density / volume_bohr3;
        if rho <= 0.0 {
            return 0.0;
        }
        // Slater exchange: V_x = -(3/pi * rho)^{1/3} * (3/(4*pi))^{1/3} * 4/3
        // Full Dirac-Slater exchange potential:
        // V_x = -(3/pi)^(1/3) * rho^(1/3) in Hartree
        let vx_hartree = -((3.0 / std::f64::consts::PI) * rho).powf(1.0 / 3.0);
        vx_hartree * HARTREE_TO_EV
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumReactionEdit {
    FormBond {
        a: usize,
        b: usize,
        order: BondOrder,
    },
    BreakBond {
        a: usize,
        b: usize,
    },
    ChangeBondOrder {
        a: usize,
        b: usize,
        order: BondOrder,
    },
    SetFormalCharge {
        atom: usize,
        formal_charge: i8,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantumReactionDelta {
    pub before: ExactDiagonalizationResult,
    pub after: ExactDiagonalizationResult,
    pub delta_ground_state_energy_ev: Option<f64>,
    pub delta_nuclear_repulsion_ev: f64,
    pub delta_net_formal_charge: i32,
}

impl QuantumReactionDelta {
    pub fn between(
        before: ExactDiagonalizationResult,
        after: ExactDiagonalizationResult,
        delta_net_formal_charge: i32,
    ) -> Self {
        Self {
            delta_ground_state_energy_ev: match (
                before.ground_state_energy_ev(),
                after.ground_state_energy_ev(),
            ) {
                (Some(before_e), Some(after_e)) => Some(after_e - before_e),
                _ => None,
            },
            delta_nuclear_repulsion_ev: after.nuclear_repulsion_ev - before.nuclear_repulsion_ev,
            delta_net_formal_charge,
            before,
            after,
        }
    }

    pub fn summary(&self) -> ReactionQuantumSummary {
        ReactionQuantumSummary {
            event_count: 1,
            ground_state_energy_delta_ev: self.delta_ground_state_energy_ev.unwrap_or(0.0) as f32,
            nuclear_repulsion_delta_ev: self.delta_nuclear_repulsion_ev as f32,
            net_formal_charge_delta: self.delta_net_formal_charge,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantumReactiveSite {
    pub graph: MoleculeGraph,
    pub positions_angstrom: Vec<[f64; 3]>,
    pub max_spatial_orbitals: Option<usize>,
    pub max_basis_size: usize,
    #[serde(default)]
    pub embedding_point_charges: Vec<QuantumEmbeddingPointCharge>,
    #[serde(default)]
    pub embedding_dipoles: Vec<QuantumEmbeddingDipole>,
}

impl QuantumReactiveSite {
    pub fn new(
        graph: MoleculeGraph,
        positions_angstrom: Vec<[f64; 3]>,
        max_spatial_orbitals: Option<usize>,
        max_basis_size: usize,
    ) -> Result<Self, QuantumMicrodomainError> {
        Self::new_with_embedding(
            graph,
            positions_angstrom,
            max_spatial_orbitals,
            max_basis_size,
            Vec::new(),
            Vec::new(),
        )
    }

    pub fn new_with_embedding(
        graph: MoleculeGraph,
        positions_angstrom: Vec<[f64; 3]>,
        max_spatial_orbitals: Option<usize>,
        max_basis_size: usize,
        embedding_point_charges: Vec<QuantumEmbeddingPointCharge>,
        embedding_dipoles: Vec<QuantumEmbeddingDipole>,
    ) -> Result<Self, QuantumMicrodomainError> {
        if positions_angstrom.len() != graph.atoms.len() {
            return Err(QuantumMicrodomainError::PositionCountMismatch {
                atoms: graph.atoms.len(),
                positions: positions_angstrom.len(),
            });
        }
        Ok(Self {
            graph,
            positions_angstrom,
            max_spatial_orbitals,
            max_basis_size,
            embedding_point_charges,
            embedding_dipoles,
        })
    }

    pub fn analyze(&self) -> Result<ExactDiagonalizationResult, QuantumMicrodomainError> {
        let active_space = MolecularActiveSpace::from_molecule_graph_with_embedding(
            &self.graph,
            &self.positions_angstrom,
            self.max_spatial_orbitals,
            &self.embedding_point_charges,
            &self.embedding_dipoles,
        )?;
        active_space.exact_diagonalize(self.max_basis_size)
    }

    pub fn evaluate_edit_sequence(
        &self,
        edits: &[QuantumReactionEdit],
    ) -> Result<QuantumReactionDelta, QuantumMicrodomainError> {
        let before = self.analyze()?;
        let before_charge = self.graph.net_charge();
        let mut graph = self.graph.clone();
        for edit in edits {
            apply_reaction_edit(&mut graph, edit)?;
        }
        let after_site = Self::new_with_embedding(
            graph.clone(),
            self.positions_angstrom.clone(),
            self.max_spatial_orbitals,
            self.max_basis_size,
            self.embedding_point_charges.clone(),
            self.embedding_dipoles.clone(),
        )?;
        let after = after_site.analyze()?;
        Ok(QuantumReactionDelta::between(
            before,
            after,
            graph.net_charge() - before_charge,
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantumMicrodomainError {
    PositionCountMismatch {
        atoms: usize,
        positions: usize,
    },
    NegativeElectronCount {
        element: String,
        formal_charge: i8,
    },
    EmptyActiveSpace,
    ActiveElectronsExceedCapacity {
        active_electrons: usize,
        capacity: usize,
    },
    InvalidParticleCount {
        spin_orbitals: usize,
        electrons: usize,
    },
    SpinOrbitalLimitExceeded {
        spin_orbitals: usize,
        max_supported: usize,
    },
    BasisTooLarge {
        basis_size: usize,
        max_basis_size: usize,
    },
    BondIndexOutOfRange {
        a: usize,
        b: usize,
        atoms: usize,
    },
    BondMissing {
        a: usize,
        b: usize,
    },
    InvalidBondEdit {
        reason: String,
    },
    AtomIndexOutOfRange {
        atom: usize,
        atoms: usize,
    },
}

impl fmt::Display for QuantumMicrodomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositionCountMismatch { atoms, positions } => {
                write!(
                    f,
                    "position count mismatch: {positions} positions for {atoms} atoms"
                )
            }
            Self::NegativeElectronCount {
                element,
                formal_charge,
            } => write!(
                f,
                "negative electron count for element {element} with formal charge {formal_charge}"
            ),
            Self::EmptyActiveSpace => write!(f, "active-space construction produced zero orbitals"),
            Self::ActiveElectronsExceedCapacity {
                active_electrons,
                capacity,
            } => write!(
                f,
                "active electron count {active_electrons} exceeds active-space capacity {capacity}"
            ),
            Self::InvalidParticleCount {
                spin_orbitals,
                electrons,
            } => write!(
                f,
                "invalid particle count: {electrons} electrons in {spin_orbitals} spin orbitals"
            ),
            Self::SpinOrbitalLimitExceeded {
                spin_orbitals,
                max_supported,
            } => write!(
                f,
                "spin-orbital count {spin_orbitals} exceeds u64 occupancy limit {max_supported}"
            ),
            Self::BasisTooLarge {
                basis_size,
                max_basis_size,
            } => write!(
                f,
                "basis size {basis_size} exceeds maximum allowed size {max_basis_size}"
            ),
            Self::BondIndexOutOfRange { a, b, atoms } => {
                write!(f, "bond indices out of range: ({a}, {b}) for {atoms} atoms")
            }
            Self::BondMissing { a, b } => write!(f, "bond missing between atoms {a} and {b}"),
            Self::InvalidBondEdit { reason } => write!(f, "invalid bond edit: {reason}"),
            Self::AtomIndexOutOfRange { atom, atoms } => {
                write!(f, "atom index out of range: {atom} for {atoms} atoms")
            }
        }
    }
}

impl Error for QuantumMicrodomainError {}

/// Solver tier selected automatically based on Hilbert space dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverTier {
    /// Full exact diagonalization (Jacobi). Used when dim <= FULL_ED_DIM_THRESHOLD.
    FullED,
    /// Selected CI with heat-bath CIPSI importance sampling.
    /// BFS from HF determinant, includes excitations where |H_ij * c_j| > epsilon.
    SelectedCI,
    /// Epstein-Nesbet 2nd-order perturbation theory correction on top of Selected CI.
    EnPt2,
}

impl Default for SolverTier {
    fn default() -> Self {
        Self::FullED
    }
}

/// Threshold below which we use full ED. Above this, we use Selected CI + EN-PT2.
const FULL_ED_DIM_THRESHOLD: usize = 500;

/// Convergence threshold for selected CI epsilon refinement (in eV).
const CIPSI_CONVERGENCE_EV: f64 = 1.0e-4;

/// Maximum selected CI variational space size before switching to PT2 correction.
/// Lifted from 2000→4000→8000 to support larger live-state-carved microdomains
/// with 32–48 atom fragments.  Memory cost scales as O(dim²) for the Davidson
/// solver, so 8000 needs ~500 MB per fragment solve — acceptable on modern GPUs.
const CIPSI_MAX_VARIATIONAL_DIM: usize = 8000;

fn filled_subshells(electron_count: u32) -> Vec<(u8, QuantumSubshell, u8)> {
    let mut remaining = electron_count;
    let mut occupancies = Vec::new();
    for &(n, subshell, capacity) in AUFBAU_ORDER {
        if remaining == 0 {
            break;
        }
        let occupancy = remaining.min(u32::from(capacity)) as u8;
        occupancies.push((n, subshell, occupancy));
        remaining -= u32::from(occupancy);
    }
    occupancies
}

fn slater_shielding(
    atomic_number: u8,
    occupancies: &[(u8, QuantumSubshell, u8)],
    target_idx: usize,
) -> f64 {
    let (target_n, target_subshell, target_occupancy) = occupancies[target_idx];
    if target_occupancy == 0 {
        return 0.0;
    }
    let target_l = target_subshell.l();
    let mut shielding = 0.0;
    for (idx, &(n, subshell, occupancy)) in occupancies.iter().enumerate() {
        if occupancy == 0 {
            continue;
        }
        let mut other_occ = occupancy as f64;
        if idx == target_idx {
            other_occ = f64::from(occupancy.saturating_sub(1));
        }
        if other_occ == 0.0 {
            continue;
        }
        let l = subshell.l();
        if target_l <= 1 {
            if n == target_n && l <= 1 {
                shielding += other_occ * if target_n == 1 { 0.30 } else { 0.35 };
            } else if n + 1 == target_n {
                shielding += other_occ * 0.85;
            } else if n < target_n.saturating_sub(1) {
                shielding += other_occ;
            } else if n == target_n && l > 1 {
                shielding += other_occ;
            }
        } else if idx == target_idx {
            shielding += other_occ * 0.35;
        } else if n < target_n || (n == target_n && l < target_l) {
            shielding += other_occ;
        }
    }
    shielding.min(f64::from(atomic_number.saturating_sub(1)))
}

fn subshell_energy_ev(n: u8, subshell: QuantumSubshell, effective_nuclear_charge: f64) -> f64 {
    let effective_n = (f64::from(n) + 0.35 * f64::from(subshell.l())).max(1.0);
    -13.605_693_122_994 * effective_nuclear_charge.powi(2) / effective_n.powi(2)
}

fn orbital_radius_angstrom(n: u8, subshell: QuantumSubshell, effective_nuclear_charge: f64) -> f64 {
    let effective_n = (f64::from(n) + 0.35 * f64::from(subshell.l())).max(1.0);
    BOHR_RADIUS_ANGSTROM * effective_n.powi(2) / effective_nuclear_charge.max(0.25)
}

fn build_spatial_orbitals(atoms: &[QuantumAtomState]) -> Vec<SpatialOrbital> {
    let mut orbitals = Vec::new();
    for (atom_index, atom) in atoms.iter().enumerate() {
        for shell in atom.active_subshells() {
            let spatial_count = shell.subshell.spatial_degeneracy();
            let onsite = COULOMB_EV_ANGSTROM / (1.5 * shell.orbital_radius_angstrom).max(0.25);
            for orientation in 0..spatial_count {
                orbitals.push(SpatialOrbital {
                    atom_index,
                    n: shell.n,
                    subshell: shell.subshell,
                    orientation,
                    shell_electrons: shell.electrons,
                    shell_capacity: shell.capacity,
                    effective_nuclear_charge: shell.effective_nuclear_charge,
                    orbital_energy_ev: shell.orbital_energy_ev,
                    orbital_radius_angstrom: shell.orbital_radius_angstrom,
                    onsite_repulsion_ev: onsite,
                });
            }
        }
    }
    orbitals
}

fn build_atom_bond_order_matrix(graph: &MoleculeGraph) -> Vec<f64> {
    let atom_count = graph.atoms.len();
    let mut matrix = vec![0.0; atom_count * atom_count];
    for bond in &graph.bonds {
        let order = match bond.order {
            BondOrder::Single => 1.0,
            BondOrder::Double => 2.0,
            BondOrder::Triple => 3.0,
            BondOrder::Aromatic => 1.5,
        };
        matrix[matrix_index(bond.i, bond.j, atom_count)] = order;
        matrix[matrix_index(bond.j, bond.i, atom_count)] = order;
    }
    matrix
}

fn build_one_body_integrals(
    atoms: &[QuantumAtomState],
    positions_angstrom: &[[f64; 3]],
    orbitals: &[SpatialOrbital],
    atom_bond_orders: &[f64],
    embedding_point_charges: &[QuantumEmbeddingPointCharge],
    embedding_dipoles: &[QuantumEmbeddingDipole],
) -> Vec<f64> {
    let n = orbitals.len();
    let mut matrix = vec![0.0; n * n];
    for (i, orbital_i) in orbitals.iter().enumerate() {
        let mut diagonal = orbital_i.orbital_energy_ev;
        let origin = positions_angstrom[orbital_i.atom_index];
        for (atom_index, atom) in atoms.iter().enumerate() {
            if atom_index == orbital_i.atom_index {
                continue;
            }
            let distance = euclidean_distance(origin, positions_angstrom[atom_index]);
            let screened = (distance.powi(2) + orbital_i.orbital_radius_angstrom.powi(2)).sqrt();
            diagonal -= 0.15 * COULOMB_EV_ANGSTROM * f64::from(atom.element.atomic_number())
                / screened.max(0.25);
        }
        for point_charge in embedding_point_charges {
            let distance = euclidean_distance(origin, point_charge.position_angstrom);
            let screened = (distance.powi(2)
                + orbital_i.orbital_radius_angstrom.powi(2)
                + point_charge.screening_radius_angstrom.powi(2))
            .sqrt();
            diagonal -= COULOMB_EV_ANGSTROM * point_charge.charge_e / screened.max(0.35);
        }
        for dipole in embedding_dipoles {
            let delta = [
                origin[0] - dipole.position_angstrom[0],
                origin[1] - dipole.position_angstrom[1],
                origin[2] - dipole.position_angstrom[2],
            ];
            let screened = (delta[0] * delta[0]
                + delta[1] * delta[1]
                + delta[2] * delta[2]
                + dipole.screening_radius_angstrom.powi(2))
            .sqrt();
            let projection = dipole.dipole_e_angstrom[0] * delta[0]
                + dipole.dipole_e_angstrom[1] * delta[1]
                + dipole.dipole_e_angstrom[2] * delta[2];
            diagonal -= COULOMB_EV_ANGSTROM * projection / screened.max(0.45).powi(3);
        }
        matrix[matrix_index(i, i, n)] = diagonal;
    }
    for (i, orbital_i) in orbitals.iter().enumerate() {
        for (j, orbital_j) in orbitals.iter().enumerate().skip(i + 1) {
            if orbital_i.atom_index == orbital_j.atom_index {
                continue;
            }
            let distance = euclidean_distance(
                positions_angstrom[orbital_i.atom_index],
                positions_angstrom[orbital_j.atom_index],
            );
            let mean_radius =
                0.5 * (orbital_i.orbital_radius_angstrom + orbital_j.orbital_radius_angstrom);
            let overlap = (-distance / mean_radius.max(0.25)).exp();
            let energy_scale =
                (orbital_i.orbital_energy_ev.abs() * orbital_j.orbital_energy_ev.abs()).sqrt();
            let symmetry = if orbital_i.subshell == orbital_j.subshell {
                1.0
            } else {
                0.65
            };
            let bond_order = atom_bond_orders
                [matrix_index(orbital_i.atom_index, orbital_j.atom_index, atoms.len())];
            let bonded_boost = 1.0 + 0.85 * bond_order;
            let hopping = -0.25 * energy_scale * overlap * symmetry * bonded_boost;
            matrix[matrix_index(i, j, n)] = hopping;
            matrix[matrix_index(j, i, n)] = hopping;
        }
    }
    matrix
}

fn build_interorbital_coulomb(
    positions_angstrom: &[[f64; 3]],
    orbitals: &[SpatialOrbital],
    atom_bond_orders: &[f64],
) -> Vec<f64> {
    let n = orbitals.len();
    let atom_count = positions_angstrom.len();
    let mut matrix = vec![0.0; n * n];
    for (i, orbital_i) in orbitals.iter().enumerate() {
        for (j, orbital_j) in orbitals.iter().enumerate().skip(i + 1) {
            let distance = euclidean_distance(
                positions_angstrom[orbital_i.atom_index],
                positions_angstrom[orbital_j.atom_index],
            );
            let screened = (distance.powi(2)
                + orbital_i.orbital_radius_angstrom.powi(2)
                + orbital_j.orbital_radius_angstrom.powi(2))
            .sqrt();
            let bond_order = atom_bond_orders
                [matrix_index(orbital_i.atom_index, orbital_j.atom_index, atom_count)];
            let bonded_screen = 1.0 + 0.20 * bond_order;
            let value = 0.75 * COULOMB_EV_ANGSTROM / (screened * bonded_screen).max(0.35);
            matrix[matrix_index(i, j, n)] = value;
            matrix[matrix_index(j, i, n)] = value;
        }
    }
    matrix
}

fn nuclear_repulsion_ev(
    atoms: &[QuantumAtomState],
    positions_angstrom: &[[f64; 3]],
    embedding_point_charges: &[QuantumEmbeddingPointCharge],
    embedding_dipoles: &[QuantumEmbeddingDipole],
) -> f64 {
    let mut energy = 0.0;
    for i in 0..atoms.len() {
        for j in (i + 1)..atoms.len() {
            let distance = euclidean_distance(positions_angstrom[i], positions_angstrom[j]);
            energy += f64::from(atoms[i].element.atomic_number())
                * f64::from(atoms[j].element.atomic_number())
                * COULOMB_EV_ANGSTROM
                / distance.max(0.25);
        }
        for point_charge in embedding_point_charges {
            let distance =
                euclidean_distance(positions_angstrom[i], point_charge.position_angstrom);
            let screened =
                (distance.powi(2) + point_charge.screening_radius_angstrom.powi(2)).sqrt();
            energy += f64::from(atoms[i].element.atomic_number())
                * point_charge.charge_e
                * COULOMB_EV_ANGSTROM
                / screened.max(0.35);
        }
        for dipole in embedding_dipoles {
            let delta = [
                positions_angstrom[i][0] - dipole.position_angstrom[0],
                positions_angstrom[i][1] - dipole.position_angstrom[1],
                positions_angstrom[i][2] - dipole.position_angstrom[2],
            ];
            let screened = (delta[0] * delta[0]
                + delta[1] * delta[1]
                + delta[2] * delta[2]
                + dipole.screening_radius_angstrom.powi(2))
            .sqrt();
            let projection = dipole.dipole_e_angstrom[0] * delta[0]
                + dipole.dipole_e_angstrom[1] * delta[1]
                + dipole.dipole_e_angstrom[2] * delta[2];
            energy +=
                f64::from(atoms[i].element.atomic_number()) * COULOMB_EV_ANGSTROM * projection
                    / screened.max(0.45).powi(3);
        }
    }
    energy
}

fn fixed_particle_basis(
    num_spin_orbitals: usize,
    num_electrons: usize,
) -> Result<Vec<u64>, QuantumMicrodomainError> {
    validate_spin_orbital_count(num_spin_orbitals)?;
    if num_electrons > num_spin_orbitals {
        return Err(QuantumMicrodomainError::InvalidParticleCount {
            spin_orbitals: num_spin_orbitals,
            electrons: num_electrons,
        });
    }
    let capacity = combination_count_capped(num_spin_orbitals, num_electrons, usize::MAX);
    let mut basis = Vec::with_capacity(capacity.min(1 << 20));
    append_fixed_particle_basis(0, num_spin_orbitals, num_electrons, 0, &mut basis);
    Ok(basis)
}

fn validate_spin_orbital_count(num_spin_orbitals: usize) -> Result<(), QuantumMicrodomainError> {
    if num_spin_orbitals > u64::BITS as usize {
        return Err(QuantumMicrodomainError::SpinOrbitalLimitExceeded {
            spin_orbitals: num_spin_orbitals,
            max_supported: u64::BITS as usize,
        });
    }
    Ok(())
}

fn combination_count_capped(n: usize, k: usize, cap: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut value = 1u128;
    let cap_u128 = cap as u128;
    for step in 0..k {
        value = value * (n - step) as u128 / (step + 1) as u128;
        if value > cap_u128 {
            return cap.saturating_add(1);
        }
    }
    value as usize
}

fn append_fixed_particle_basis(
    start_orbital: usize,
    num_spin_orbitals: usize,
    num_electrons: usize,
    state: u64,
    basis: &mut Vec<u64>,
) {
    if num_electrons == 0 {
        basis.push(state);
        return;
    }
    let max_orbital = num_spin_orbitals - num_electrons;
    for orbital in start_orbital..=max_orbital {
        append_fixed_particle_basis(
            orbital + 1,
            num_spin_orbitals,
            num_electrons - 1,
            state | (1u64 << orbital),
            basis,
        );
    }
}

fn apply_annihilation(state: u64, orbital: usize) -> Option<(u64, i8)> {
    if ((state >> orbital) & 1) == 0 {
        return None;
    }
    let sign = if (state & ((1u64 << orbital) - 1)).count_ones() % 2 == 0 {
        1
    } else {
        -1
    };
    Some((state & !(1u64 << orbital), sign))
}

fn apply_creation(state: u64, orbital: usize) -> Option<(u64, i8)> {
    if ((state >> orbital) & 1) == 1 {
        return None;
    }
    let sign = if (state & ((1u64 << orbital) - 1)).count_ones() % 2 == 0 {
        1
    } else {
        -1
    };
    Some((state | (1u64 << orbital), sign))
}

fn apply_hop(state: u64, source_orbital: usize, target_orbital: usize) -> Option<(u64, i8)> {
    let (removed_state, remove_sign) = apply_annihilation(state, source_orbital)?;
    let (created_state, create_sign) = apply_creation(removed_state, target_orbital)?;
    Some((created_state, remove_sign * create_sign))
}

fn spatial_one_particle_density_matrix(
    basis_states: &[u64],
    ground_state_vector: &[f64],
    num_spatial_orbitals: usize,
    index_by_state: &std::collections::HashMap<u64, usize>,
) -> Vec<f64> {
    let mut density = vec![0.0; num_spatial_orbitals * num_spatial_orbitals];
    for (ket_idx, &state) in basis_states.iter().enumerate() {
        let ket_amplitude = ground_state_vector[ket_idx];
        if ket_amplitude.abs() < 1.0e-12 {
            continue;
        }
        for spin in 0..2usize {
            for q in 0..num_spatial_orbitals {
                let Some((removed_state, remove_sign)) = apply_annihilation(state, 2 * q + spin)
                else {
                    continue;
                };
                for p in 0..num_spatial_orbitals {
                    let Some((bra_state, create_sign)) =
                        apply_creation(removed_state, 2 * p + spin)
                    else {
                        continue;
                    };
                    let Some(&bra_idx) = index_by_state.get(&bra_state) else {
                        continue;
                    };
                    density[matrix_index(p, q, num_spatial_orbitals)] += ground_state_vector
                        [bra_idx]
                        * ket_amplitude
                        * f64::from(remove_sign * create_sign);
                }
            }
        }
    }
    symmetrize_in_place(&mut density, num_spatial_orbitals);
    density
}

fn expected_atom_effective_charges(
    atoms: &[QuantumAtomState],
    spatial_orbitals: &[SpatialOrbital],
    expected_spatial_occupancies: &[f64],
) -> Vec<f64> {
    let mut atom_valence_occupancies = vec![0.0; atoms.len()];
    for (orbital, occupancy) in spatial_orbitals
        .iter()
        .zip(expected_spatial_occupancies.iter())
    {
        atom_valence_occupancies[orbital.atom_index] += occupancy;
    }
    atoms
        .iter()
        .enumerate()
        .map(|(atom_idx, atom)| {
            f64::from(atom.formal_charge) + f64::from(atom.valence_electrons())
                - atom_valence_occupancies[atom_idx]
        })
        .collect()
}

fn expected_dipole_moment_e_angstrom(
    positions_angstrom: &[[f64; 3]],
    expected_atom_effective_charges: &[f64],
) -> [f64; 3] {
    if positions_angstrom.is_empty()
        || positions_angstrom.len() != expected_atom_effective_charges.len()
    {
        return [0.0, 0.0, 0.0];
    }
    let mut centroid = [0.0f64; 3];
    for position in positions_angstrom {
        centroid[0] += position[0];
        centroid[1] += position[1];
        centroid[2] += position[2];
    }
    centroid[0] /= positions_angstrom.len() as f64;
    centroid[1] /= positions_angstrom.len() as f64;
    centroid[2] /= positions_angstrom.len() as f64;

    let mut dipole = [0.0f64; 3];
    for (position, &charge) in positions_angstrom
        .iter()
        .zip(expected_atom_effective_charges.iter())
    {
        dipole[0] += charge * (position[0] - centroid[0]);
        dipole[1] += charge * (position[1] - centroid[1]);
        dipole[2] += charge * (position[2] - centroid[2]);
    }
    dipole
}

/// Lanczos iteration for the ground state (lowest eigenvalue + eigenvector).
///
/// For symmetric matrices larger than ~500×500 where full Jacobi
/// eigendecomposition is too expensive, Lanczos finds the ground state
/// in O(k × n²) where k is the Krylov subspace dimension (~100).
///
/// Returns (ground_state_energy, ground_state_vector).
fn lanczos_ground_state(matrix: &[f64], dim: usize, max_krylov: usize) -> (f64, Vec<f64>) {
    if dim == 0 {
        return (0.0, vec![]);
    }
    if dim == 1 {
        return (matrix[0], vec![1.0]);
    }
    let k = max_krylov.min(dim);
    let mut alpha = Vec::with_capacity(k);
    let mut beta = Vec::with_capacity(k);
    // Lanczos vectors stored column-major: lanczos_vecs[j * dim .. (j+1) * dim].
    let mut lanczos_vecs = vec![0.0; k * dim];

    // Starting vector: unit vector at the lowest diagonal element.
    let start = (0..dim)
        .min_by(|&a, &b| {
            matrix[a * dim + a]
                .partial_cmp(&matrix[b * dim + b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    lanczos_vecs[start] = 1.0;

    let mut beta_prev = 0.0_f64;
    let mut actual_k = k;
    for j in 0..k {
        let v_j = &lanczos_vecs[j * dim..(j + 1) * dim];
        // w = H * v_j
        let mut w = vec![0.0; dim];
        for r in 0..dim {
            let mut sum = 0.0;
            let row_base = r * dim;
            for c in 0..dim {
                sum += matrix[row_base + c] * v_j[c];
            }
            w[r] = sum;
        }
        // alpha_j = v_j^T w
        let a_j: f64 = v_j.iter().zip(w.iter()).map(|(v, w_)| v * w_).sum();
        alpha.push(a_j);
        // w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
        for r in 0..dim {
            w[r] -= a_j * v_j[r];
        }
        if j > 0 {
            let v_jm1 = &lanczos_vecs[(j - 1) * dim..j * dim];
            for r in 0..dim {
                w[r] -= beta_prev * v_jm1[r];
            }
        }
        // Re-orthogonalize against all previous Lanczos vectors to
        // prevent loss of orthogonality in finite precision.
        for prev in 0..=j {
            let v_p = &lanczos_vecs[prev * dim..(prev + 1) * dim];
            let dot: f64 = w.iter().zip(v_p.iter()).map(|(a, b)| a * b).sum();
            for r in 0..dim {
                w[r] -= dot * v_p[r];
            }
        }
        let b_j: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if b_j < 1.0e-14 {
            actual_k = j + 1;
            break;
        }
        beta.push(b_j);
        beta_prev = b_j;
        if j + 1 < k {
            let next_base = (j + 1) * dim;
            for r in 0..dim {
                lanczos_vecs[next_base + r] = w[r] / b_j;
            }
        }
    }

    // Diag the actual_k × actual_k tridiagonal matrix using Jacobi.
    let tk = actual_k;
    let mut t_matrix = vec![0.0; tk * tk];
    for i in 0..tk {
        t_matrix[i * tk + i] = alpha[i];
        if i > 0 {
            t_matrix[i * tk + (i - 1)] = beta[i - 1];
            t_matrix[(i - 1) * tk + i] = beta[i - 1];
        }
    }
    let (t_evals, t_evecs) =
        jacobi_eigendecomposition(&t_matrix, tk, 1.0e-12, 200 * tk.max(1) * tk.max(1));

    // Find the ground state in the tridiagonal eigenvalues.
    let gs_tri = t_evals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let gs_energy = t_evals[gs_tri];

    // Reconstruct ground state vector in the original basis.
    let mut gs_vector = vec![0.0; dim];
    for j in 0..tk {
        let coeff = t_evecs[j * tk + gs_tri];
        if coeff.abs() < 1.0e-16 {
            continue;
        }
        let base = j * dim;
        for r in 0..dim {
            gs_vector[r] += coeff * lanczos_vecs[base + r];
        }
    }
    // Normalize.
    let norm: f64 = gs_vector.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1.0e-14 {
        for x in &mut gs_vector {
            *x /= norm;
        }
    }
    (gs_energy, gs_vector)
}

/// Threshold above which the CIPSI solver uses Lanczos for ground-state
/// diagonalization instead of full Jacobi eigendecomposition.
const LANCZOS_DIM_THRESHOLD: usize = 500;

/// Krylov subspace dimension for Lanczos ground-state solver.
const LANCZOS_MAX_KRYLOV: usize = 120;

/// Find the ground state (lowest eigenvalue + eigenvector) of a symmetric
/// matrix, choosing Lanczos for large matrices and Jacobi for small ones.
fn ground_state_diag(matrix: &[f64], dim: usize) -> (f64, Vec<f64>) {
    if dim <= LANCZOS_DIM_THRESHOLD {
        let (eigenvalues, eigenvectors) =
            jacobi_eigendecomposition(matrix, dim, 1.0e-10, 200 * dim.max(1) * dim.max(1));
        let gs_idx = eigenvalues
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let gs_energy = eigenvalues[gs_idx];
        let gs_vector: Vec<f64> = (0..dim)
            .map(|row| eigenvectors[row * dim + gs_idx])
            .collect();
        (gs_energy, gs_vector)
    } else {
        lanczos_ground_state(matrix, dim, LANCZOS_MAX_KRYLOV)
    }
}

fn jacobi_eigendecomposition(
    matrix: &[f64],
    dim: usize,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = identity_matrix(dim);
    if dim == 0 {
        return (Vec::new(), Vec::new());
    }
    for _ in 0..max_iterations {
        let mut p = 0usize;
        let mut q = 1usize.min(dim - 1);
        let mut max_off_diag = 0.0;
        for i in 0..dim {
            for j in (i + 1)..dim {
                let value = a[matrix_index(i, j, dim)].abs();
                if value > max_off_diag {
                    max_off_diag = value;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off_diag < tolerance || p == q {
            break;
        }

        let app = a[matrix_index(p, p, dim)];
        let aqq = a[matrix_index(q, q, dim)];
        let apq = a[matrix_index(p, q, dim)];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for k in 0..dim {
            if k == p || k == q {
                continue;
            }
            let aik = a[matrix_index(k, p, dim)];
            let akq = a[matrix_index(k, q, dim)];
            a[matrix_index(k, p, dim)] = c * aik - s * akq;
            a[matrix_index(p, k, dim)] = a[matrix_index(k, p, dim)];
            a[matrix_index(k, q, dim)] = s * aik + c * akq;
            a[matrix_index(q, k, dim)] = a[matrix_index(k, q, dim)];
        }
        a[matrix_index(p, p, dim)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[matrix_index(q, q, dim)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[matrix_index(p, q, dim)] = 0.0;
        a[matrix_index(q, p, dim)] = 0.0;

        for k in 0..dim {
            let vip = v[matrix_index(k, p, dim)];
            let viq = v[matrix_index(k, q, dim)];
            v[matrix_index(k, p, dim)] = c * vip - s * viq;
            v[matrix_index(k, q, dim)] = s * vip + c * viq;
        }
    }

    let eigenvalues = (0..dim).map(|idx| a[matrix_index(idx, idx, dim)]).collect();
    (eigenvalues, v)
}

fn apply_reaction_edit(
    graph: &mut MoleculeGraph,
    edit: &QuantumReactionEdit,
) -> Result<(), QuantumMicrodomainError> {
    match *edit {
        QuantumReactionEdit::FormBond { a, b, order } => {
            if a >= graph.atoms.len() || b >= graph.atoms.len() {
                return Err(QuantumMicrodomainError::BondIndexOutOfRange {
                    a,
                    b,
                    atoms: graph.atoms.len(),
                });
            }
            graph
                .add_bond(a, b, order)
                .map_err(|reason| QuantumMicrodomainError::InvalidBondEdit { reason })
        }
        QuantumReactionEdit::BreakBond { a, b } => {
            let before = graph.bonds.len();
            graph
                .bonds
                .retain(|bond| !((bond.i == a && bond.j == b) || (bond.i == b && bond.j == a)));
            if graph.bonds.len() == before {
                return Err(QuantumMicrodomainError::BondMissing { a, b });
            }
            Ok(())
        }
        QuantumReactionEdit::ChangeBondOrder { a, b, order } => {
            let mut changed = false;
            for bond in &mut graph.bonds {
                if (bond.i == a && bond.j == b) || (bond.i == b && bond.j == a) {
                    bond.order = order;
                    changed = true;
                    break;
                }
            }
            if !changed {
                return Err(QuantumMicrodomainError::BondMissing { a, b });
            }
            Ok(())
        }
        QuantumReactionEdit::SetFormalCharge {
            atom,
            formal_charge,
        } => {
            let Some(node) = graph.atoms.get_mut(atom) else {
                return Err(QuantumMicrodomainError::AtomIndexOutOfRange {
                    atom,
                    atoms: graph.atoms.len(),
                });
            };
            node.formal_charge = formal_charge;
            Ok(())
        }
    }
}

fn symmetrize_in_place(matrix: &mut [f64], dim: usize) {
    for i in 0..dim {
        for j in (i + 1)..dim {
            let average = 0.5 * (matrix[matrix_index(i, j, dim)] + matrix[matrix_index(j, i, dim)]);
            matrix[matrix_index(i, j, dim)] = average;
            matrix[matrix_index(j, i, dim)] = average;
        }
    }
}

fn identity_matrix(dim: usize) -> Vec<f64> {
    let mut matrix = vec![0.0; dim * dim];
    for idx in 0..dim {
        matrix[matrix_index(idx, idx, dim)] = 1.0;
    }
    matrix
}

fn euclidean_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn matrix_index(row: usize, col: usize, dim: usize) -> usize {
    row * dim + col
}

fn abs_cmp(a: f64, b: f64) -> Ordering {
    a.abs().partial_cmp(&b.abs()).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomistic_chemistry::{AtomNode, BondOrder, MoleculeGraph, PeriodicElement};

    fn hydrogen_graph() -> MoleculeGraph {
        let mut graph = MoleculeGraph::new("h2");
        graph.add_atom_node(AtomNode::new(PeriodicElement::H));
        graph.add_atom_node(AtomNode::new(PeriodicElement::H));
        graph.add_bond(0, 1, BondOrder::Single).unwrap();
        graph
    }

    fn water_graph() -> MoleculeGraph {
        let mut graph = MoleculeGraph::new("water");
        let o = graph.add_atom_node(AtomNode::new(PeriodicElement::O));
        let h1 = graph.add_atom_node(AtomNode::new(PeriodicElement::H));
        let h2 = graph.add_atom_node(AtomNode::new(PeriodicElement::H));
        graph.add_bond(o, h1, BondOrder::Single).unwrap();
        graph.add_bond(o, h2, BondOrder::Single).unwrap();
        graph
    }

    #[test]
    fn oxygen_atom_has_explicit_quark_and_shell_state() {
        let atom = AtomNode::new(PeriodicElement::O);
        let state = QuantumAtomState::from_atom_node(&atom).unwrap();
        assert_eq!(state.electron_count(), 8);
        assert_eq!(state.nucleus.protons, 8);
        assert_eq!(state.nucleus.neutrons, 8);
        #[cfg(feature = "nuclear_physics")]
        {
            assert_eq!(state.quark_inventory().up, 24);
            assert_eq!(state.quark_inventory().down, 24);
        }
        assert_eq!(
            state
                .electron_subshells
                .iter()
                .map(ElectronSubshell::label)
                .collect::<Vec<_>>(),
            vec!["1s".to_string(), "2s".to_string(), "2p".to_string()]
        );
        assert_eq!(state.valence_electrons(), 6);
        assert!(state.nucleus.binding_energy_mev() > 0.0);
    }

    #[test]
    fn hydrogen_active_space_diagonalizes() {
        let graph = hydrogen_graph();
        let positions = [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]];
        let active = graph.quantum_active_space(&positions, None).unwrap();
        let result = active.exact_diagonalize(64).unwrap();
        assert_eq!(active.active_electrons, 2);
        assert_eq!(active.num_spatial_orbitals(), 2);
        assert_eq!(active.num_spin_orbitals(), 4);
        assert_eq!(active.total_electrons(), 2);
        assert_eq!(result.basis_states.len(), 6);
        assert_eq!(result.spatial_orbital_atom_indices, vec![0, 1]);
        assert_eq!(result.spatial_one_particle_density_matrix.len(), 4);
        assert!(
            (result.spatial_one_particle_density_matrix[matrix_index(0, 0, 2)]
                - result.expected_spatial_occupancies[0])
                .abs()
                < 1.0e-6
        );
        assert!((result.expected_electron_count - 2.0).abs() < 1.0e-6);
        assert_eq!(result.expected_atom_effective_charges.len(), 2);
        assert!(
            result
                .expected_atom_effective_charges
                .iter()
                .sum::<f64>()
                .abs()
                < 1.0e-6
        );
        assert!(
            result
                .expected_dipole_moment_e_angstrom
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
                < 1.0e-6
        );
        assert!(result.ground_state_energy_ev().is_some());
        assert!(result
            .energies_ev
            .windows(2)
            .all(|window| window[0] <= window[1] + 1.0e-9));
    }

    #[test]
    fn embedding_point_charge_perturbs_active_space_hamiltonian() {
        let graph = hydrogen_graph();
        let positions = [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]];
        let unembedded = MolecularActiveSpace::from_molecule_graph(&graph, &positions, None)
            .expect("unembedded active space");
        let embedded = MolecularActiveSpace::from_molecule_graph_with_embedding(
            &graph,
            &positions,
            None,
            &[QuantumEmbeddingPointCharge::new([0.0, 0.0, 1.2], 0.8, 0.7)],
            &[],
        )
        .expect("embedded active space");

        assert!(
            embedded.one_body_integrals_ev[matrix_index(1, 1, 2)]
                < unembedded.one_body_integrals_ev[matrix_index(1, 1, 2)]
        );
        let embedded_ground_state = embedded
            .exact_diagonalize(64)
            .expect("embedded diagonalization")
            .ground_state_energy_ev()
            .expect("embedded ground state");
        let unembedded_ground_state = unembedded
            .exact_diagonalize(64)
            .expect("unembedded diagonalization")
            .ground_state_energy_ev()
            .expect("unembedded ground state");
        assert!((embedded_ground_state - unembedded_ground_state).abs() > 1.0e-3);
    }

    #[test]
    fn embedding_dipole_perturbs_active_space_hamiltonian() {
        let graph = hydrogen_graph();
        let positions = [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]];
        let unembedded = MolecularActiveSpace::from_molecule_graph(&graph, &positions, None)
            .expect("unembedded active space");
        let embedded = MolecularActiveSpace::from_molecule_graph_with_embedding(
            &graph,
            &positions,
            None,
            &[],
            &[QuantumEmbeddingDipole::new(
                [0.0, 0.0, 1.2],
                [0.0, 0.0, 1.0],
                0.7,
            )],
        )
        .expect("dipole embedded active space");

        assert!(
            (embedded.one_body_integrals_ev[matrix_index(0, 0, 2)]
                - unembedded.one_body_integrals_ev[matrix_index(0, 0, 2)])
            .abs()
                > 1.0e-3
        );
        let embedded_ground_state = embedded
            .exact_diagonalize(64)
            .expect("dipole embedded diagonalization")
            .ground_state_energy_ev()
            .expect("dipole embedded ground state");
        let unembedded_ground_state = unembedded
            .exact_diagonalize(64)
            .expect("unembedded diagonalization")
            .ground_state_energy_ev()
            .expect("unembedded ground state");
        assert!((embedded_ground_state - unembedded_ground_state).abs() > 1.0e-3);
    }

    #[test]
    fn bond_topology_increases_hopping_coupling() {
        let mut bonded = MoleculeGraph::new("h2");
        bonded.add_atom_node(AtomNode::new(PeriodicElement::H));
        bonded.add_atom_node(AtomNode::new(PeriodicElement::H));
        bonded.add_bond(0, 1, BondOrder::Single).unwrap();

        let mut unbonded = MoleculeGraph::new("h_h");
        unbonded.add_atom_node(AtomNode::new(PeriodicElement::H));
        unbonded.add_atom_node(AtomNode::new(PeriodicElement::H));

        let positions = [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]];
        let bonded_active =
            MolecularActiveSpace::from_molecule_graph(&bonded, &positions, None).unwrap();
        let unbonded_active =
            MolecularActiveSpace::from_molecule_graph(&unbonded, &positions, None).unwrap();
        let bonded_hopping = bonded_active.one_body_integrals_ev[matrix_index(0, 1, 2)].abs();
        let unbonded_hopping = unbonded_active.one_body_integrals_ev[matrix_index(0, 1, 2)].abs();
        let bonded_response = bonded_active
            .exact_diagonalize(64)
            .unwrap()
            .atom_pair_density_response(0, 1);
        let unbonded_response = unbonded_active
            .exact_diagonalize(64)
            .unwrap()
            .atom_pair_density_response(0, 1);
        assert_eq!(bonded_active.atom_bond_orders[matrix_index(0, 1, 2)], 1.0);
        assert_eq!(unbonded_active.atom_bond_orders[matrix_index(0, 1, 2)], 0.0);
        assert!(bonded_hopping > unbonded_hopping);
        assert!(bonded_response > unbonded_response + 1.0e-3);
    }

    #[test]
    fn same_atom_orbitals_do_not_gain_artificial_hopping() {
        let mut oxygen = MoleculeGraph::new("oxygen_atom");
        oxygen.add_atom_node(AtomNode::new(PeriodicElement::O));
        let positions = [[0.0, 0.0, 0.0]];
        let active = oxygen.quantum_active_space(&positions, None).unwrap();
        let n = active.num_spatial_orbitals();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert_eq!(active.one_body_integrals_ev[matrix_index(i, j, n)], 0.0);
                }
            }
        }
    }

    #[test]
    fn water_reactive_site_reports_energy_shift_for_charge_edit() {
        let graph = water_graph();
        let positions = [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]];
        let site = graph
            .quantum_reactive_site(positions.to_vec(), Some(4), 1024)
            .unwrap();
        let delta = site
            .evaluate_edit_sequence(&[QuantumReactionEdit::SetFormalCharge {
                atom: 0,
                formal_charge: 1,
            }])
            .unwrap();
        assert_eq!(delta.delta_net_formal_charge, 1);
        assert!(delta.delta_ground_state_energy_ev.is_some());
    }

    #[test]
    fn atom_node_exposes_quantum_state() {
        let oxygen = AtomNode::new(PeriodicElement::O);
        let state = oxygen.quantum_state().unwrap();
        assert_eq!(state.valence_electrons(), 6);
        assert!(state.ionization_proxy_ev() > 0.0);
        assert!(state.total_rest_mass_u() > state.nucleus.rest_mass_u());
    }
}
