import math

import numpy as np
import pytest

from oneuro.proteins.docking import Ligand
from oneuro.proteins.molecular_dynamics import Atom, Molecule
from oneuro.quantum.atomistic_bridge import build_native_graph_quantum_microdomain
from oneuro.quantum.subatomic import (
    MolecularActiveSpace,
    QuantumAtom,
    build_reference_active_space,
    discover_local_nqpu_sdk,
    reference_molecule_geometry,
)


def test_quantum_atom_builds_explicit_nuclear_and_quark_state():
    oxygen = QuantumAtom.from_symbol("O")

    assert oxygen.atomic_number == 8
    assert oxygen.electron_count == 8
    assert oxygen.nucleus.protons == 8
    assert oxygen.nucleus.neutrons == 8
    assert oxygen.quark_inventory.up == 24
    assert oxygen.quark_inventory.down == 24
    assert oxygen.quark_inventory.charge_e == pytest.approx(0.0)
    assert [shell.label for shell in oxygen.electron_subshells] == ["1s", "2s", "2p"]
    assert oxygen.valence_electrons == 6
    assert oxygen.ionization_proxy_ev > 0.0
    assert oxygen.nucleus.binding_energy_mev > 0.0


def test_hydrogen_molecule_active_space_builds_and_diagonalizes():
    h2 = build_reference_active_space("H2")
    solution = h2.exact_diagonalize()

    assert h2.num_spatial_orbitals == 2
    assert h2.num_spin_orbitals == 4
    assert h2.active_electrons == 2
    assert h2.total_electrons == 2
    assert h2.one_body_integrals_ev.shape == (2, 2)
    assert len(solution.basis_states) == math.comb(h2.num_spin_orbitals, h2.active_electrons)
    assert np.isclose(np.linalg.norm(solution.ground_state_vector), 1.0)
    assert np.all(np.diff(solution.energies_ev[:4]) >= -1.0e-9)
    assert solution.expected_electron_count == pytest.approx(2.0, abs=1.0e-6)
    assert np.all(solution.expected_spatial_occupancies >= -1.0e-9)


def test_water_valence_active_space_is_explicit_and_not_scalar():
    symbols = ("O", "H", "H")
    positions = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ]
    )
    water = MolecularActiveSpace.from_symbols(symbols, positions)

    assert water.num_spatial_orbitals == 6
    assert water.num_spin_orbitals == 12
    assert water.active_electrons == 8
    assert water.frozen_core_electrons == 2
    assert water.total_electrons == 10
    assert water.one_body_integrals_ev.shape == (6, 6)
    assert water.interorbital_coulomb_ev.shape == (6, 6)
    assert np.allclose(water.one_body_integrals_ev, water.one_body_integrals_ev.T)
    assert np.allclose(water.interorbital_coulomb_ev, water.interorbital_coulomb_ev.T)
    assert water.nuclear_repulsion_ev > 0.0


def test_reference_molecule_geometry_provides_explicit_positions():
    symbols, positions = reference_molecule_geometry("LiH")

    assert symbols == ("Li", "H")
    assert positions.shape == (2, 3)
    assert float(np.linalg.norm(positions[0] - positions[1])) == pytest.approx(1.596)


def test_discover_local_nqpu_sdk_respects_environment_override(tmp_path):
    root = tmp_path / "nQPU"
    core_dir = root / "sdk" / "python" / "core"
    core_dir.mkdir(parents=True)
    (core_dir / "quantum_backend.py").write_text("class VQEMolecule:\n    pass\n", encoding="ascii")

    discovered = discover_local_nqpu_sdk(env={"ONEURO_NQPU_ROOT": str(root)})

    assert discovered.available
    assert discovered.root == root
    assert discovered.core_dir == core_dir


def test_md_molecule_integrates_with_quantum_microdomain_bridge():
    molecule = Molecule(
        atoms=[
            Atom(id=0, element="H", mass=1.008, charge=0.0, sigma=2.886, epsilon=0.0157),
            Atom(id=1, element="H", mass=1.008, charge=0.0, sigma=2.886, epsilon=0.0157),
        ],
        bonds=[(0, 1)],
        angles=[],
        dihedrals=[],
    )
    positions = np.array([[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]])

    report = molecule.to_quantum_microdomain(positions)

    assert report.active_space.active_electrons == 2
    assert report.active_space.num_spatial_orbitals == 2
    assert report.exact_solution is not None
    assert report.active_basis_size == 6
    assert report.ground_state_energy_ev is not None


def test_docking_ligand_integrates_with_quantum_microdomain_bridge():
    ligand = Ligand(
        atoms=np.array(
            [
                [0.0000, 0.0000, 0.0000],
                [1.0900, 0.0000, 0.0000],
                [2.2500, 0.0000, 0.0000],
            ]
        ),
        atom_types=["H", "C", "N"],
        bonds=[(0, 1), (1, 2)],
        rotatable_bonds=[],
        torsions=[],
    )

    report = ligand.to_quantum_microdomain(max_spatial_orbitals=5)

    assert report.active_space.total_electrons == 14
    assert report.active_space.num_spatial_orbitals == 5
    assert report.exact_solution is not None
    assert report.active_basis_size is not None


def test_native_graph_payload_bridge_builds_quantum_microdomain():
    class NativeGraphStub:
        def atom_payloads(self):
            return [("H", 0, None), ("O", 0, None), ("H", 0, None)]

        def bond_payloads(self):
            return [(0, 1, "single"), (1, 2, "single")]

    positions = np.array(
        [
            [0.9572, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ]
    )

    report = build_native_graph_quantum_microdomain(NativeGraphStub(), positions)

    assert report.active_space.total_electrons == 10
    assert report.active_space.active_electrons == 8
    assert report.exact_solution is not None
    assert report.active_basis_size is not None


def test_explicit_bond_topology_changes_active_space_coupling():
    positions = np.array([[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]])
    bonded = MolecularActiveSpace.from_symbols(
        ("H", "H"),
        positions,
        bonds=[(0, 1, "single")],
    )
    unbonded = MolecularActiveSpace.from_symbols(("H", "H"), positions)

    bonded_hopping = abs(bonded.one_body_integrals_ev[0, 1])
    unbonded_hopping = abs(unbonded.one_body_integrals_ev[0, 1])

    assert bonded.atom_bond_orders[0, 1] == pytest.approx(1.0)
    assert unbonded.atom_bond_orders[0, 1] == pytest.approx(0.0)
    assert bonded_hopping > unbonded_hopping
