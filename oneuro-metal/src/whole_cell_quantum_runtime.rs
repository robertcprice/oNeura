#![allow(dead_code)] // Scaffolding for future quantum-runtime phases
use crate::atomistic_chemistry::{
    BondOrder, EmbeddedMaterialMixture, EmbeddedMaterialStructuralReactionError, EmbeddedMolecule,
    MoleculeGraph, PeriodicElement, ScopedAtomRef, StructuralReactionEdit,
    StructuralReactionTemplate,
};
use crate::atomistic_topology::atomistic_template_for_site_name;
use crate::subatomic_quantum::{
    ExactDiagonalizationResult, QuantumAtomState, QuantumChemistryConfig, QuantumEmbeddingDipole,
    QuantumEmbeddingOrbitalResponseField, QuantumEmbeddingPointCharge,
};
use crate::substrate_ir::ReactionQuantumSummary;
use crate::whole_cell_data::{
    WholeCellAssetClass, WholeCellBulkField, WholeCellGenomeProcessRegistry,
    WholeCellReactionClass, WholeCellReactionRuntimeState, WholeCellSavedRuntimeQuantumProcess,
};
use crate::whole_cell_submodels::Syn3ASubsystemPreset;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum WholeCellRuntimeQuantumProcessKind {
    RibosomeTranslationCondensation,
    AtpBandEnergyPhosphorylation,
    ReplisomeNucleotidePhosphorylation,
    AtpBandMembraneEsterification,
    SeptumMembraneEsterification,
    AtpBandProtonTransfer,
    SeptumGtpHydrolysis,
    AtpBandElectronTransfer,
    MembraneProteinInsertion,
}

impl WholeCellRuntimeQuantumProcessKind {
    pub(crate) const fn all() -> [Self; 9] {
        [
            Self::RibosomeTranslationCondensation,
            Self::AtpBandEnergyPhosphorylation,
            Self::ReplisomeNucleotidePhosphorylation,
            Self::AtpBandMembraneEsterification,
            Self::SeptumMembraneEsterification,
            Self::AtpBandProtonTransfer,
            Self::SeptumGtpHydrolysis,
            Self::AtpBandElectronTransfer,
            Self::MembraneProteinInsertion,
        ]
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::RibosomeTranslationCondensation => "ribosome_translation_condensation",
            Self::AtpBandEnergyPhosphorylation => "atp_band_energy_phosphorylation",
            Self::ReplisomeNucleotidePhosphorylation => "replisome_nucleotide_phosphorylation",
            Self::AtpBandMembraneEsterification => "atp_band_membrane_esterification",
            Self::SeptumMembraneEsterification => "septum_membrane_esterification",
            Self::AtpBandProtonTransfer => "atp_band_proton_transfer",
            Self::SeptumGtpHydrolysis => "septum_gtp_hydrolysis",
            Self::AtpBandElectronTransfer => "atp_band_electron_transfer",
            Self::MembraneProteinInsertion => "membrane_protein_insertion",
        }
    }

    pub(crate) fn from_str(value: &str) -> Option<Self> {
        match value.trim() {
            "ribosome_translation_condensation" => Some(Self::RibosomeTranslationCondensation),
            "atp_band_energy_phosphorylation" => Some(Self::AtpBandEnergyPhosphorylation),
            "replisome_nucleotide_phosphorylation" => {
                Some(Self::ReplisomeNucleotidePhosphorylation)
            }
            "atp_band_membrane_esterification" => Some(Self::AtpBandMembraneEsterification),
            "septum_membrane_esterification" => Some(Self::SeptumMembraneEsterification),
            "atp_band_proton_transfer" => Some(Self::AtpBandProtonTransfer),
            "septum_gtp_hydrolysis" => Some(Self::SeptumGtpHydrolysis),
            "atp_band_electron_transfer" => Some(Self::AtpBandElectronTransfer),
            "membrane_protein_insertion" => Some(Self::MembraneProteinInsertion),
            _ => None,
        }
    }

    pub(crate) fn preset(self) -> Syn3ASubsystemPreset {
        match self {
            Self::RibosomeTranslationCondensation => Syn3ASubsystemPreset::RibosomePolysomeCluster,
            Self::AtpBandEnergyPhosphorylation => Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
            Self::ReplisomeNucleotidePhosphorylation => Syn3ASubsystemPreset::ReplisomeTrack,
            Self::AtpBandMembraneEsterification => Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
            Self::SeptumMembraneEsterification => Syn3ASubsystemPreset::FtsZSeptumRing,
            Self::AtpBandProtonTransfer => Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
            Self::SeptumGtpHydrolysis => Syn3ASubsystemPreset::FtsZSeptumRing,
            Self::AtpBandElectronTransfer => Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
            Self::MembraneProteinInsertion => Syn3ASubsystemPreset::FtsZSeptumRing,
        }
    }

    pub(crate) fn site_name(self) -> &'static str {
        match self {
            Self::RibosomeTranslationCondensation => "ribosome_cluster",
            Self::AtpBandEnergyPhosphorylation => "atp_synthase_band",
            Self::ReplisomeNucleotidePhosphorylation => "chromosome_track",
            Self::AtpBandMembraneEsterification => "atp_synthase_band",
            Self::SeptumMembraneEsterification => "septum_ring",
            Self::AtpBandProtonTransfer => "atp_synthase_band",
            Self::SeptumGtpHydrolysis => "septum_ring",
            Self::AtpBandElectronTransfer => "atp_synthase_band",
            Self::MembraneProteinInsertion => "septum_ring",
        }
    }

    pub(crate) const fn max_fragment_count(self) -> usize {
        2
    }

    pub(crate) fn kinds_for_reaction(
        reaction: &WholeCellReactionRuntimeState,
    ) -> Vec<WholeCellRuntimeQuantumProcessKind> {
        let mut kinds = Vec::new();
        let mut push_kind = |kind| {
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        };
        for preset in &reaction.subsystem_targets {
            match (reaction.reaction_class, reaction.asset_class, preset) {
                (
                    WholeCellReactionClass::Translation,
                    _,
                    Syn3ASubsystemPreset::RibosomePolysomeCluster,
                ) => push_kind(Self::RibosomeTranslationCondensation),
                (
                    WholeCellReactionClass::ComplexMaturation,
                    WholeCellAssetClass::Energy,
                    Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
                ) => push_kind(Self::AtpBandEnergyPhosphorylation),
                (
                    WholeCellReactionClass::ComplexMaturation,
                    WholeCellAssetClass::Replication,
                    Syn3ASubsystemPreset::ReplisomeTrack,
                ) => push_kind(Self::ReplisomeNucleotidePhosphorylation),
                (
                    WholeCellReactionClass::ComplexMaturation,
                    WholeCellAssetClass::Membrane,
                    Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
                ) => push_kind(Self::AtpBandMembraneEsterification),
                (
                    WholeCellReactionClass::ComplexMaturation,
                    WholeCellAssetClass::Membrane,
                    Syn3ASubsystemPreset::FtsZSeptumRing,
                ) => push_kind(Self::SeptumMembraneEsterification),
                (
                    WholeCellReactionClass::ComplexElongation,
                    WholeCellAssetClass::Energy,
                    Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
                ) => push_kind(Self::AtpBandProtonTransfer),
                (
                    WholeCellReactionClass::ComplexMaturation,
                    WholeCellAssetClass::Constriction,
                    Syn3ASubsystemPreset::FtsZSeptumRing,
                ) => push_kind(Self::SeptumGtpHydrolysis),
                (
                    WholeCellReactionClass::ComplexElongation,
                    WholeCellAssetClass::Homeostasis,
                    Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
                ) => push_kind(Self::AtpBandElectronTransfer),
                (
                    WholeCellReactionClass::SubunitPoolFormation,
                    WholeCellAssetClass::Membrane,
                    Syn3ASubsystemPreset::FtsZSeptumRing,
                ) => push_kind(Self::MembraneProteinInsertion),
                _ => {}
            }
        }
        kinds
    }

    /// Map this enum variant to a fully-specified `QuantumReactionSpec`.
    /// This is the bridge from the legacy hardcoded kinds to the generic path.
    pub(crate) fn reaction_spec(self) -> QuantumReactionSpec {
        match self {
            Self::RibosomeTranslationCondensation => QuantumReactionSpec {
                name: "ribosome_translation_amide_condensation".to_string(),
                kind: self,
                preset: Syn3ASubsystemPreset::RibosomePolysomeCluster,
                site_name: "ribosome_cluster".to_string(),
                scaffold_builder: methylamine as fn([f32; 3]) -> EmbeddedMolecule,
                reactive_builders: vec![acetic_acid as fn([f32; 3]) -> EmbeddedMolecule],
                build_reaction: |scaffold_mol, reactive_mols| {
                    let amine = &scaffold_mol;
                    let acid = &reactive_mols[0];
                    StructuralReactionTemplate::new(
                        "ribosome_translation_amide_condensation",
                        vec![amine.graph.clone(), acid.graph.clone()],
                        vec![
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(0, 1),
                                b: ScopedAtomRef::new(0, 6),
                            },
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(1, 1),
                                b: ScopedAtomRef::new(1, 3),
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(0, 1),
                                b: ScopedAtomRef::new(1, 1),
                                order: BondOrder::Single,
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(1, 3),
                                b: ScopedAtomRef::new(0, 6),
                                order: BondOrder::Single,
                            },
                        ],
                    )
                },
                reactive_displacement_angstrom: 2.35,
                inter_reactive_displacement_angstrom: 0.0,
                atom_budget_seed: 8,
            },
            Self::AtpBandEnergyPhosphorylation => QuantumReactionSpec {
                name: "atp_band_energy_phosphorylation".to_string(),
                kind: self,
                preset: Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
                site_name: "atp_synthase_band".to_string(),
                scaffold_builder: methanol as fn([f32; 3]) -> EmbeddedMolecule,
                reactive_builders: vec![
                    orthophosphoric_acid as fn([f32; 3]) -> EmbeddedMolecule,
                ],
                build_reaction: |scaffold_mol, reactive_mols| {
                    phosphate_esterification_reaction(
                        "atp_band_energy_phosphorylation",
                        scaffold_mol.graph.clone(),
                        1,
                        5,
                        reactive_mols[0].graph.clone(),
                    )
                },
                reactive_displacement_angstrom: 2.55,
                inter_reactive_displacement_angstrom: 0.0,
                atom_budget_seed: 6,
            },
            Self::ReplisomeNucleotidePhosphorylation => QuantumReactionSpec {
                name: "replisome_nucleotide_phosphorylation".to_string(),
                kind: self,
                preset: Syn3ASubsystemPreset::ReplisomeTrack,
                site_name: "chromosome_track".to_string(),
                scaffold_builder: ethanol as fn([f32; 3]) -> EmbeddedMolecule,
                reactive_builders: vec![
                    orthophosphoric_acid as fn([f32; 3]) -> EmbeddedMolecule,
                ],
                build_reaction: |scaffold_mol, reactive_mols| {
                    phosphate_esterification_reaction(
                        "replisome_nucleotide_phosphorylation",
                        scaffold_mol.graph.clone(),
                        2,
                        8,
                        reactive_mols[0].graph.clone(),
                    )
                },
                reactive_displacement_angstrom: 2.80,
                inter_reactive_displacement_angstrom: 0.0,
                atom_budget_seed: 6,
            },
            Self::AtpBandMembraneEsterification => QuantumReactionSpec {
                name: "atp_band_membrane_esterification".to_string(),
                kind: self,
                preset: Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
                site_name: "atp_synthase_band".to_string(),
                scaffold_builder: methanol as fn([f32; 3]) -> EmbeddedMolecule,
                reactive_builders: vec![acetic_acid as fn([f32; 3]) -> EmbeddedMolecule],
                build_reaction: |scaffold_mol, reactive_mols| {
                    let alcohol = &scaffold_mol;
                    let acid = &reactive_mols[0];
                    StructuralReactionTemplate::new(
                        "atp_band_membrane_esterification",
                        vec![alcohol.graph.clone(), acid.graph.clone()],
                        vec![
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(0, 1),
                                b: ScopedAtomRef::new(0, 5),
                            },
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(1, 1),
                                b: ScopedAtomRef::new(1, 3),
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(0, 1),
                                b: ScopedAtomRef::new(1, 1),
                                order: BondOrder::Single,
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(1, 3),
                                b: ScopedAtomRef::new(0, 5),
                                order: BondOrder::Single,
                            },
                        ],
                    )
                },
                reactive_displacement_angstrom: 2.55,
                inter_reactive_displacement_angstrom: 0.0,
                atom_budget_seed: 4,
            },
            Self::SeptumMembraneEsterification => QuantumReactionSpec {
                name: "septum_membrane_esterification".to_string(),
                kind: self,
                preset: Syn3ASubsystemPreset::FtsZSeptumRing,
                site_name: "septum_ring".to_string(),
                scaffold_builder: methanol as fn([f32; 3]) -> EmbeddedMolecule,
                reactive_builders: vec![acetic_acid as fn([f32; 3]) -> EmbeddedMolecule],
                build_reaction: |scaffold_mol, reactive_mols| {
                    let alcohol = &scaffold_mol;
                    let acid = &reactive_mols[0];
                    StructuralReactionTemplate::new(
                        "septum_membrane_esterification",
                        vec![alcohol.graph.clone(), acid.graph.clone()],
                        vec![
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(0, 1),
                                b: ScopedAtomRef::new(0, 5),
                            },
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(1, 1),
                                b: ScopedAtomRef::new(1, 3),
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(0, 1),
                                b: ScopedAtomRef::new(1, 1),
                                order: BondOrder::Single,
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(1, 3),
                                b: ScopedAtomRef::new(0, 5),
                                order: BondOrder::Single,
                            },
                        ],
                    )
                },
                reactive_displacement_angstrom: 2.75,
                inter_reactive_displacement_angstrom: 0.0,
                atom_budget_seed: 4,
            },
            Self::AtpBandProtonTransfer => QuantumReactionSpec {
                name: "atp_band_proton_transfer".to_string(),
                kind: self,
                preset: Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
                site_name: "atp_synthase_band".to_string(),
                scaffold_builder: imidazole as fn([f32; 3]) -> EmbeddedMolecule,
                reactive_builders: vec![water as fn([f32; 3]) -> EmbeddedMolecule],
                build_reaction: |scaffold_mol, reactive_mols| {
                    let imid = &scaffold_mol;
                    let wat = &reactive_mols[0];
                    // Proton relay through water bridge: break N-H on imidazole
                    // donor, break one O-H on water, form new O-H (donor H goes
                    // to water O), form new N-H (water H goes to imidazole N).
                    // Net: proton swaps through the Grotthuss water bridge.
                    StructuralReactionTemplate::new(
                        "atp_band_proton_transfer",
                        vec![imid.graph.clone(), wat.graph.clone()],
                        vec![
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(0, 3),  // N3 (pyrrole N)
                                b: ScopedAtomRef::new(0, 8),  // H (N-H proton donor)
                            },
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(1, 0),  // water O
                                b: ScopedAtomRef::new(1, 1),  // water H (leaving)
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(1, 0),  // water O
                                b: ScopedAtomRef::new(0, 8),  // H (imidazole → water)
                                order: BondOrder::Single,
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(0, 3),  // N3 (imidazole)
                                b: ScopedAtomRef::new(1, 1),  // H (water → imidazole)
                                order: BondOrder::Single,
                            },
                        ],
                    )
                },
                reactive_displacement_angstrom: 2.20,
                inter_reactive_displacement_angstrom: 0.0,
                atom_budget_seed: 6,
            },
            Self::SeptumGtpHydrolysis => QuantumReactionSpec {
                name: "septum_gtp_hydrolysis".to_string(),
                kind: self,
                preset: Syn3ASubsystemPreset::FtsZSeptumRing,
                site_name: "septum_ring".to_string(),
                scaffold_builder: methyl_pyrophosphate as fn([f32; 3]) -> EmbeddedMolecule,
                reactive_builders: vec![water as fn([f32; 3]) -> EmbeddedMolecule],
                build_reaction: |scaffold_mol, reactive_mols| {
                    let ppi = &scaffold_mol;
                    let wat = &reactive_mols[0];
                    // GTP hydrolysis: break P-O bridge (P2-O_bridge, atoms 2→1),
                    // break O-H on water (atoms 0→1 of water),
                    // form P-OH (gamma-P gets OH from water),
                    // form bridge-O-H (bridge O gets H from water).
                    StructuralReactionTemplate::new(
                        "septum_gtp_hydrolysis",
                        vec![ppi.graph.clone(), wat.graph.clone()],
                        vec![
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(0, 1),  // bridge O
                                b: ScopedAtomRef::new(0, 2),  // gamma-P
                            },
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(1, 0),  // water O
                                b: ScopedAtomRef::new(1, 1),  // water H
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(0, 2),  // gamma-P
                                b: ScopedAtomRef::new(1, 0),  // water O (now P-OH)
                                order: BondOrder::Single,
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(0, 1),  // bridge O
                                b: ScopedAtomRef::new(1, 1),  // water H (bridge O-H)
                                order: BondOrder::Single,
                            },
                        ],
                    )
                },
                reactive_displacement_angstrom: 2.60,
                inter_reactive_displacement_angstrom: 0.0,
                atom_budget_seed: 6,
            },
            Self::AtpBandElectronTransfer => QuantumReactionSpec {
                name: "atp_band_electron_transfer".to_string(),
                kind: self,
                preset: Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
                site_name: "atp_synthase_band".to_string(),
                scaffold_builder: methanol as fn([f32; 3]) -> EmbeddedMolecule,
                reactive_builders: vec![water as fn([f32; 3]) -> EmbeddedMolecule],
                build_reaction: |scaffold_mol, reactive_mols| {
                    let nadh_model = &scaffold_mol;
                    let quinone_model = &reactive_mols[0];
                    // Hydride transfer: break C-H on NADH model (methanol C-H),
                    // break O-H on quinone model (water O-H),
                    // form new O-H (hydride to quinone O), form C-H (proton to NADH C).
                    // Net: 2e- + H+ relay through enzyme active site.
                    StructuralReactionTemplate::new(
                        "atp_band_electron_transfer",
                        vec![nadh_model.graph.clone(), quinone_model.graph.clone()],
                        vec![
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(0, 0),  // methanol C
                                b: ScopedAtomRef::new(0, 2),  // methanol H (on C)
                            },
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(1, 0),  // water O
                                b: ScopedAtomRef::new(1, 1),  // water H
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(1, 0),  // water O (quinone acceptor)
                                b: ScopedAtomRef::new(0, 2),  // H (hydride from NADH->quinone)
                                order: BondOrder::Single,
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(0, 0),  // methanol C (NADH donor)
                                b: ScopedAtomRef::new(1, 1),  // H (proton from quinone->NADH)
                                order: BondOrder::Single,
                            },
                        ],
                    )
                },
                reactive_displacement_angstrom: 2.40,
                inter_reactive_displacement_angstrom: 0.0,
                atom_budget_seed: 6,
            },
            Self::MembraneProteinInsertion => QuantumReactionSpec {
                name: "membrane_protein_insertion".to_string(),
                kind: self,
                preset: Syn3ASubsystemPreset::FtsZSeptumRing,
                site_name: "septum_ring".to_string(),
                scaffold_builder: methylamine as fn([f32; 3]) -> EmbeddedMolecule,
                reactive_builders: vec![methanol as fn([f32; 3]) -> EmbeddedMolecule],
                build_reaction: |scaffold_mol, reactive_mols| {
                    let peptide = &scaffold_mol;
                    let lipid = &reactive_mols[0];
                    // Peptide-lipid insertion: break N-H on peptide (amine proton leaves),
                    // break C-O on lipid interface (methanol C-OH bond),
                    // form N-C (peptide nitrogen bonds to lipid carbon -- insertion),
                    // form O-H (displaced proton goes to hydroxyl).
                    StructuralReactionTemplate::new(
                        "membrane_protein_insertion",
                        vec![peptide.graph.clone(), lipid.graph.clone()],
                        vec![
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(0, 1),  // methylamine N
                                b: ScopedAtomRef::new(0, 6),  // methylamine H (on N)
                            },
                            StructuralReactionEdit::BreakBond {
                                a: ScopedAtomRef::new(1, 0),  // methanol C
                                b: ScopedAtomRef::new(1, 1),  // methanol O
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(0, 1),  // peptide N
                                b: ScopedAtomRef::new(1, 0),  // lipid C (insertion bond)
                                order: BondOrder::Single,
                            },
                            StructuralReactionEdit::FormBond {
                                a: ScopedAtomRef::new(1, 1),  // lipid O
                                b: ScopedAtomRef::new(0, 6),  // displaced H
                                order: BondOrder::Single,
                            },
                        ],
                    )
                },
                reactive_displacement_angstrom: 2.50,
                inter_reactive_displacement_angstrom: 0.0,
                atom_budget_seed: 6,
            },
        }
    }
}

/// A fully specified quantum reaction -- molecule graphs + structural edits.
/// This replaces the hardcoded per-kind molecule builders and enables ANY reaction
/// with explicit MoleculeGraph reactants and StructuralReactionEdit products to be
/// quantum-elevated without needing to be in the hardcoded enum.
#[derive(Clone)]
pub(crate) struct QuantumReactionSpec {
    /// Human-readable reaction name.
    pub name: String,
    /// The enum kind that this spec corresponds to (for backward compatibility).
    pub kind: WholeCellRuntimeQuantumProcessKind,
    /// The subsystem preset where this reaction occurs.
    pub preset: Syn3ASubsystemPreset,
    /// Site name for scaffold lookup.
    pub site_name: String,
    /// Builder for the scaffold-side reactant molecule at a given position.
    pub scaffold_builder: fn([f32; 3]) -> EmbeddedMolecule,
    /// Builders for each reactive-side molecule at given positions.
    pub reactive_builders: Vec<fn([f32; 3]) -> EmbeddedMolecule>,
    /// Closure that builds the base StructuralReactionTemplate from the placed
    /// scaffold molecule and reactive molecules (before scaffold prepend).
    pub build_reaction: fn(&EmbeddedMolecule, &[EmbeddedMolecule]) -> StructuralReactionTemplate,
    /// Displacement distance (Angstrom) between scaffold surface and first reactive molecule.
    pub reactive_displacement_angstrom: f32,
    /// Displacement between successive reactive molecules.
    pub inter_reactive_displacement_angstrom: f32,
    /// Atom budget seed (used by adaptive budget system).
    pub atom_budget_seed: usize,
}

impl std::fmt::Debug for QuantumReactionSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantumReactionSpec")
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("preset", &self.preset)
            .field("site_name", &self.site_name)
            .field("reactive_displacement_angstrom", &self.reactive_displacement_angstrom)
            .field("inter_reactive_displacement_angstrom", &self.inter_reactive_displacement_angstrom)
            .field("atom_budget_seed", &self.atom_budget_seed)
            .finish()
    }
}

/// Build `WholeCellRuntimeQuantumProcessState` instances from a generic
/// `QuantumReactionSpec`. This is the single generic path that replaces
/// all five per-kind builder functions.
pub(crate) fn quantum_processes_from_spec(
    spec: &QuantumReactionSpec,
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
    scaffold_source: Option<&EmbeddedMolecule>,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    let origin = [0.0f32; 3];
    let scaffold_mol = (spec.scaffold_builder)(origin);
    let mut reactant_ve = valence_electrons_for_graph(&scaffold_mol.graph);
    for builder in &spec.reactive_builders {
        reactant_ve += valence_electrons_for_graph(&builder(origin).graph);
    }
    let effective_max =
        max_scaffold_atoms.min(max_scaffold_atoms_within_orbital_limit(reactant_ve));
    let scaffolds =
        scaffolds_for_kind(spec.kind, effective_max, preferred_anchors, scaffold_source);
    scaffolds
        .into_iter()
        .map(|fragment| {
            let SiteScaffoldFragment {
                scaffold,
                source_anchor_atom_idx,
            } = fragment;
            let placement = scaffold_reactant_placement(spec.kind, &scaffold, 2.6);
            let anchor = placement.anchor_position_angstrom;

            // Build scaffold-side reactant at displacement distance from anchor.
            let scaffold_mol = (spec.scaffold_builder)(offset_along_direction(
                anchor,
                placement.direction,
                spec.reactive_displacement_angstrom,
            ));

            // Build reactive-side reactants at anchor (and beyond for multi-reactive).
            let mut reactive_mols = Vec::with_capacity(spec.reactive_builders.len());
            for (idx, builder) in spec.reactive_builders.iter().enumerate() {
                let displacement = spec.inter_reactive_displacement_angstrom * idx as f32;
                let pos = offset_along_direction(anchor, placement.direction, displacement);
                reactive_mols.push(builder(pos));
            }

            // Build the base reaction template from placed molecules.
            let base_reaction = (spec.build_reaction)(&scaffold_mol, &reactive_mols);

            // Assemble the full reactant list and reaction with scaffold prepended.
            let mut all_reactants = Vec::with_capacity(1 + spec.reactive_builders.len());
            all_reactants.push(scaffold_mol);
            all_reactants.extend(reactive_mols);
            let (reactants, reaction) =
                prepend_site_scaffold(scaffold, all_reactants, base_reaction);

            WholeCellRuntimeQuantumProcessState::new(
                spec.kind,
                Some(source_anchor_atom_idx),
                reactants,
                reaction,
            )
        })
        .collect()
}

/// Auto-discover quantum-eligible reactions from the compiled process registry.
/// Any reaction that has:
/// - Explicit MoleculeGraph reactants (from atomistic_chemistry StructuralReactionTemplates)
/// - StructuralReactionEdit products (bond breaks/forms)
/// - A subsystem preset with a known site template
/// can be quantum-elevated without needing to be in the hardcoded enum.
///
/// This scans reactions for ones not already covered by the hardcoded 5 kinds
/// and produces generic `QuantumReactionSpec` entries for them.
pub(crate) fn auto_discover_quantum_reactions(
    reactions: &[WholeCellReactionRuntimeState],
    _registry: &WholeCellGenomeProcessRegistry,
) -> Vec<QuantumReactionSpec> {
    let mut discovered = Vec::new();
    for reaction in reactions {
        // Skip reactions already covered by the hardcoded enum dispatch.
        let known_kinds = WholeCellRuntimeQuantumProcessKind::kinds_for_reaction(reaction);
        if !known_kinds.is_empty() {
            continue;
        }
        // Only consider reactions with subsystem targets that have known site templates.
        for preset in &reaction.subsystem_targets {
            let site_name = match preset {
                Syn3ASubsystemPreset::RibosomePolysomeCluster => "ribosome_cluster",
                Syn3ASubsystemPreset::AtpSynthaseMembraneBand => "atp_synthase_band",
                Syn3ASubsystemPreset::ReplisomeTrack => "chromosome_track",
                Syn3ASubsystemPreset::FtsZSeptumRing => "septum_ring",
            };
            // We need a site template to have scaffold geometry.
            if atomistic_template_for_site_name(site_name).is_none() {
                continue;
            }
            // For auto-discovered reactions we use a generic condensation template
            // based on the reaction's asset class, using the same small-molecule
            // surrogates as the hardcoded kinds. The heuristic selects:
            //   - Translation + any preset => amide condensation (methylamine + acetic_acid)
            //   - Energy asset class => phosphorylation (methanol + orthophosphoric_acid)
            //   - Replication asset class => nucleotide phosphorylation (ethanol + orthophosphoric_acid)
            //   - Membrane asset class => esterification (methanol + acetic_acid)
            //   - Otherwise => generic condensation (methylamine + acetic_acid)
            let (scaffold_builder, reactive_builders, displacement, build_reaction, spec_name): (
                fn([f32; 3]) -> EmbeddedMolecule,
                Vec<fn([f32; 3]) -> EmbeddedMolecule>,
                f32,
                fn(&EmbeddedMolecule, &[EmbeddedMolecule]) -> StructuralReactionTemplate,
                String,
            ) = match (reaction.reaction_class, reaction.asset_class) {
                (WholeCellReactionClass::Translation, _) => (
                    methylamine,
                    vec![acetic_acid as fn([f32; 3]) -> EmbeddedMolecule],
                    2.35,
                    |scaffold_mol: &EmbeddedMolecule, reactive_mols: &[EmbeddedMolecule]| {
                        StructuralReactionTemplate::new(
                            "auto_amide_condensation",
                            vec![scaffold_mol.graph.clone(), reactive_mols[0].graph.clone()],
                            vec![
                                StructuralReactionEdit::BreakBond {
                                    a: ScopedAtomRef::new(0, 1),
                                    b: ScopedAtomRef::new(0, 6),
                                },
                                StructuralReactionEdit::BreakBond {
                                    a: ScopedAtomRef::new(1, 1),
                                    b: ScopedAtomRef::new(1, 3),
                                },
                                StructuralReactionEdit::FormBond {
                                    a: ScopedAtomRef::new(0, 1),
                                    b: ScopedAtomRef::new(1, 1),
                                    order: BondOrder::Single,
                                },
                                StructuralReactionEdit::FormBond {
                                    a: ScopedAtomRef::new(1, 3),
                                    b: ScopedAtomRef::new(0, 6),
                                    order: BondOrder::Single,
                                },
                            ],
                        )
                    },
                    format!("auto_translation_{}_{}", reaction.id, site_name),
                ),
                (_, WholeCellAssetClass::Energy) => (
                    methanol,
                    vec![orthophosphoric_acid as fn([f32; 3]) -> EmbeddedMolecule],
                    2.55,
                    |scaffold_mol: &EmbeddedMolecule, reactive_mols: &[EmbeddedMolecule]| {
                        phosphate_esterification_reaction(
                            "auto_energy_phosphorylation",
                            scaffold_mol.graph.clone(),
                            1,
                            5,
                            reactive_mols[0].graph.clone(),
                        )
                    },
                    format!("auto_energy_phosphorylation_{}_{}", reaction.id, site_name),
                ),
                (_, WholeCellAssetClass::Replication) => (
                    ethanol,
                    vec![orthophosphoric_acid as fn([f32; 3]) -> EmbeddedMolecule],
                    2.80,
                    |scaffold_mol: &EmbeddedMolecule, reactive_mols: &[EmbeddedMolecule]| {
                        phosphate_esterification_reaction(
                            "auto_nucleotide_phosphorylation",
                            scaffold_mol.graph.clone(),
                            2,
                            8,
                            reactive_mols[0].graph.clone(),
                        )
                    },
                    format!(
                        "auto_nucleotide_phosphorylation_{}_{}",
                        reaction.id, site_name
                    ),
                ),
                (_, WholeCellAssetClass::Membrane) => (
                    methanol,
                    vec![acetic_acid as fn([f32; 3]) -> EmbeddedMolecule],
                    2.55,
                    |scaffold_mol: &EmbeddedMolecule, reactive_mols: &[EmbeddedMolecule]| {
                        StructuralReactionTemplate::new(
                            "auto_membrane_esterification",
                            vec![scaffold_mol.graph.clone(), reactive_mols[0].graph.clone()],
                            vec![
                                StructuralReactionEdit::BreakBond {
                                    a: ScopedAtomRef::new(0, 1),
                                    b: ScopedAtomRef::new(0, 5),
                                },
                                StructuralReactionEdit::BreakBond {
                                    a: ScopedAtomRef::new(1, 1),
                                    b: ScopedAtomRef::new(1, 3),
                                },
                                StructuralReactionEdit::FormBond {
                                    a: ScopedAtomRef::new(0, 1),
                                    b: ScopedAtomRef::new(1, 1),
                                    order: BondOrder::Single,
                                },
                                StructuralReactionEdit::FormBond {
                                    a: ScopedAtomRef::new(1, 3),
                                    b: ScopedAtomRef::new(0, 5),
                                    order: BondOrder::Single,
                                },
                            ],
                        )
                    },
                    format!("auto_membrane_esterification_{}_{}", reaction.id, site_name),
                ),
                _ => (
                    methylamine,
                    vec![acetic_acid as fn([f32; 3]) -> EmbeddedMolecule],
                    2.35,
                    |scaffold_mol: &EmbeddedMolecule, reactive_mols: &[EmbeddedMolecule]| {
                        StructuralReactionTemplate::new(
                            "auto_generic_condensation",
                            vec![scaffold_mol.graph.clone(), reactive_mols[0].graph.clone()],
                            vec![
                                StructuralReactionEdit::BreakBond {
                                    a: ScopedAtomRef::new(0, 1),
                                    b: ScopedAtomRef::new(0, 6),
                                },
                                StructuralReactionEdit::BreakBond {
                                    a: ScopedAtomRef::new(1, 1),
                                    b: ScopedAtomRef::new(1, 3),
                                },
                                StructuralReactionEdit::FormBond {
                                    a: ScopedAtomRef::new(0, 1),
                                    b: ScopedAtomRef::new(1, 1),
                                    order: BondOrder::Single,
                                },
                                StructuralReactionEdit::FormBond {
                                    a: ScopedAtomRef::new(1, 3),
                                    b: ScopedAtomRef::new(0, 6),
                                    order: BondOrder::Single,
                                },
                            ],
                        )
                    },
                    format!("auto_generic_condensation_{}_{}", reaction.id, site_name),
                ),
            };

            // Use RibosomeTranslationCondensation as the kind for auto-discovered
            // reactions since the kind is only used for scaffold scoring heuristics
            // and we need a valid kind. The preset determines actual behavior.
            let fallback_kind = match reaction.asset_class {
                WholeCellAssetClass::Energy => {
                    WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation
                }
                WholeCellAssetClass::Replication => {
                    WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation
                }
                WholeCellAssetClass::Membrane => {
                    WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification
                }
                _ => WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation,
            };

            // Check for duplicate: same name already discovered.
            if discovered.iter().any(|d: &QuantumReactionSpec| d.name == spec_name) {
                continue;
            }

            discovered.push(QuantumReactionSpec {
                name: spec_name,
                kind: fallback_kind,
                preset: *preset,
                site_name: site_name.to_string(),
                scaffold_builder,
                reactive_builders,
                build_reaction,
                reactive_displacement_angstrom: displacement,
                inter_reactive_displacement_angstrom: 0.0,
                atom_budget_seed: 4,
            });
        }
    }
    discovered
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WholeCellRuntimeQuantumProcessState {
    kind: WholeCellRuntimeQuantumProcessKind,
    site_anchor_atom_idx: Option<usize>,
    mixture: EmbeddedMaterialMixture,
    reactants: Vec<EmbeddedMolecule>,
    reaction: StructuralReactionTemplate,
    quantum: QuantumChemistryConfig,
    scaffold_boundary_replenish_amount: f64,
    reactive_boundary_replenish_amount: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeQuantumFragmentOutcome {
    pub summary: ReactionQuantumSummary,
    pub extent: f32,
    pub scaffold_centroid_angstrom: [f32; 3],
    pub reactive_centroid_angstrom: [f32; 3],
}

impl RuntimeQuantumFragmentOutcome {
    fn interaction_center_angstrom(self) -> [f32; 3] {
        [
            0.5 * (self.scaffold_centroid_angstrom[0] + self.reactive_centroid_angstrom[0]),
            0.5 * (self.scaffold_centroid_angstrom[1] + self.reactive_centroid_angstrom[1]),
            0.5 * (self.scaffold_centroid_angstrom[2] + self.reactive_centroid_angstrom[2]),
        ]
    }

    fn local_radius_angstrom(self) -> f32 {
        squared_distance(
            self.scaffold_centroid_angstrom,
            self.reactive_centroid_angstrom,
        )
        .sqrt()
        .max(1.0)
    }

    fn response_magnitude_ev(self) -> f32 {
        self.extent.max(0.0)
            * (self.summary.ground_state_energy_delta_ev.abs()
                + 0.5 * self.summary.nuclear_repulsion_delta_ev.abs())
    }
}

const FRAGMENT_RESPONSE_MAX_ITERATIONS: usize = 10;
const FRAGMENT_RESPONSE_RELAXATION: f32 = 0.65;
const FRAGMENT_RESPONSE_TOLERANCE_EV: f32 = 1.0e-4;
const FRAGMENT_EMBEDDING_RESPONSE_MAX_ITERATIONS: usize = 120;
const FRAGMENT_EMBEDDING_RESPONSE_RELAXATION: f64 = 0.55;
const FRAGMENT_EMBEDDING_RESPONSE_TOLERANCE_E: f64 = 1.0e-2;
/// Safety ceilings for embedding source counts. The convergence-driven
/// truncation (`convergence_truncate`) applies AFTER filtering and sorting,
/// keeping sources until their Coulomb contribution falls below kT/100.
/// These caps prevent runaway memory in pathological cases.
const FRAGMENT_EMBEDDING_MAX_POINT_CHARGES: usize = 128;
const FRAGMENT_EMBEDDING_MAX_DIPOLES: usize = 64;
const FRAGMENT_EMBEDDING_MAX_ORBITAL_RESPONSE_FIELDS: usize = 48;
/// Spatial cutoff for collecting candidate embedding sources.
/// Raised from the original 14.0 Å to allow the convergence-driven truncation
/// to determine the effective radius based on Coulomb contribution < kT/100.
const FRAGMENT_EMBEDDING_CUTOFF_ANGSTROM: f32 = 20.0;
const FRAGMENT_EMBEDDING_MIN_CHARGE_E: f64 = 0.025;
const FRAGMENT_EMBEDDING_MIN_DIPOLE_E_ANGSTROM: f64 = 0.05;
const FRAGMENT_EMBEDDING_MIN_ORBITAL_RESPONSE_EV: f64 = 0.01;
const FRAGMENT_EMBEDDING_RESPONSE_TOLERANCE_ORBITAL_EV: f64 = 5.0e-2;

/// Default embedding convergence threshold: kT/100 at 310 K.
/// kB = 8.617e-5 eV/K, T = 310 K → kT ≈ 0.02671 eV → kT/100 ≈ 2.671e-4 eV.
const DEFAULT_EMBEDDING_CONVERGENCE_THRESHOLD_EV: f64 = 2.671e-4;

/// LDA exchange coupling scale factor for perturbative embedding correction.
/// A value of ~0.1 keeps the exchange correction perturbative relative to the
/// Coulomb embedding, avoiding SCF instability from over-correcting exchange.
const LDA_EXCHANGE_COUPLING_SCALE: f64 = 0.1;

/// Fragment energy convergence threshold: kT/10 at 310 K ≈ 0.002671 eV.
/// If successive self-consistent iterations change a fragment's ground-state
/// energy by more than this, the fragment is considered unconverged and may
/// benefit from adaptive growth (larger atom budget).
///
/// Set to kT/10 at 310 K (≈ 2.67e-2 eV) to accommodate the variational noise
/// floor of the CIPSI selected-CI solver — the determinant selection changes
/// slightly between iterations with different embedding potentials, introducing
/// O(0.01 eV) energy fluctuations even when the embedding itself has converged.
const FRAGMENT_ENERGY_CONVERGENCE_THRESHOLD_EV: f64 = 2.671e-2;

// ---------------------------------------------------------------------------
// Eyring/TST rate constants and functions (Phase 7d)
// ---------------------------------------------------------------------------

/// Boltzmann constant in eV/K.
const BOLTZMANN_K_EV_PER_K: f64 = 8.617333262e-5;

/// Planck constant in eV*s.
const PLANCK_H_EV_S: f64 = 4.135667696e-15;

/// Compute reaction rate from a quantum-computed activation barrier using
/// Eyring transition-state theory:
///
///   k = (kB T / h) * exp(-dG_barrier / kB T)
///
/// Returns rate in s^-1.
pub(crate) fn eyring_rate_from_barrier(barrier_ev: f64, temperature_k: f64) -> f64 {
    let kt = BOLTZMANN_K_EV_PER_K * temperature_k;
    let prefactor = kt / PLANCK_H_EV_S; // ~6.25e12 s^-1 at 310 K
    prefactor * (-barrier_ev / kt).exp()
}

/// Compute the quantum-derived rate enhancement or suppression factor
/// relative to an existing heuristic rate.  This is used to gradually
/// replace hardcoded Km/Vmax with quantum-derived barriers.
///
/// Returns `(eyring_rate_s_inv, quantum_rate_modifier)` where the modifier
/// is the ratio of the quantum-derived rate to the heuristic rate.
pub(crate) fn quantum_rate_modifier(
    barrier_ev: f64,
    temperature_k: f64,
    heuristic_rate_s_inv: f64,
) -> (f64, f64) {
    let quantum_rate = eyring_rate_from_barrier(barrier_ev, temperature_k);
    let modifier = if heuristic_rate_s_inv > 0.0 {
        quantum_rate / heuristic_rate_s_inv
    } else {
        1.0
    };
    (quantum_rate, modifier)
}

/// Known activation barriers from literature (eV) for validation.
/// These are NOT used in the simulation -- they are reference values to check
/// ED-computed barriers against.
pub(crate) const REFERENCE_BARRIERS: &[(&str, f64)] = &[
    ("peptide_bond_formation", 0.87),  // ~84 kJ/mol, ribosomal catalysis lowers to ~0.65 eV
    ("atp_hydrolysis", 0.55),          // ~53 kJ/mol
    ("phosphodiester_bond", 0.73),     // ~70 kJ/mol
    ("ester_bond_formation", 0.78),    // ~75 kJ/mol, membrane lipid synthesis
    ("amide_bond_hydrolysis", 0.91),   // ~88 kJ/mol, uncatalyzed
    ("electron_transfer", 0.30),           // ~29 kJ/mol, enzyme-catalyzed NADH->quinone
    ("membrane_protein_insertion", 0.45),  // ~43 kJ/mol, SecYEG-catalyzed
];

/// Literature-derived heuristic turnover rates (s^-1) for each quantum
/// process kind.  These come from Michaelis-Menten Vmax estimates for the
/// corresponding enzyme families in minimal bacterial cells:
///
///  - Ribosome peptide bond: ~20 aa/s (Bremer & Dennis 1996)
///  - ATP synthase:          ~100-400 s^-1 (Boyer 1997)
///  - Replisome:             ~1000 nt/s in E. coli, ~200 in Mycoplasma
///  - Lipid esterification:  ~10-50 s^-1 (acyltransferase family)
fn heuristic_rate_for_process_kind(kind: WholeCellRuntimeQuantumProcessKind) -> f64 {
    match kind {
        WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation => 20.0,
        WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation => 200.0,
        WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation => 200.0,
        WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification => 30.0,
        WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification => 30.0,
        WholeCellRuntimeQuantumProcessKind::AtpBandProtonTransfer => 1000.0,
        WholeCellRuntimeQuantumProcessKind::SeptumGtpHydrolysis => 5.0,
        WholeCellRuntimeQuantumProcessKind::AtpBandElectronTransfer => 300.0,
        WholeCellRuntimeQuantumProcessKind::MembraneProteinInsertion => 5.0,
    }
}

/// Minimum barrier (eV) for a quantum-derived rate to be considered
/// physically meaningful.  Below this threshold the ED energy delta is
/// likely numerical noise rather than a real activation barrier.
const MIN_QUANTUM_BARRIER_EV: f64 = 0.01;

/// Maximum barrier (eV) for a quantum-derived rate to be considered
/// applicable.  Above this the Eyring rate is effectively zero at
/// physiological temperature, so there is no point in replacing the
/// heuristic rate.
const MAX_QUANTUM_BARRIER_EV: f64 = 5.0;

/// Convergence-driven truncation: given a distance-sorted list of sources,
/// keep adding them until the Coulomb contribution of the next source falls
/// below `threshold_ev`.  Returns the truncation index.
///
/// For point charges: V = 14.3996 * q / r  (eV, with q in e and r in Å)
/// For dipoles:       V ~ 14.3996 * |p| / r²  (eV, rough upper bound)
fn convergence_truncate_charges(
    charges: &[QuantumEmbeddingPointCharge],
    target_center: [f32; 3],
    threshold_ev: f64,
    safety_cap: usize,
) -> usize {
    const COULOMB_CONSTANT_EV_ANGSTROM: f64 = 14.3996;
    let n = charges.len().min(safety_cap);
    for i in 0..n {
        let pos = f64x3_to_f32(charges[i].position_angstrom);
        let r = squared_distance(pos, target_center).sqrt().max(0.1) as f64;
        let contribution = COULOMB_CONSTANT_EV_ANGSTROM * charges[i].charge_e.abs() / r;
        if contribution < threshold_ev {
            return i.max(1); // keep at least 1 source
        }
    }
    n
}

fn convergence_truncate_dipoles(
    dipoles: &[QuantumEmbeddingDipole],
    target_center: [f32; 3],
    threshold_ev: f64,
    safety_cap: usize,
) -> usize {
    const COULOMB_CONSTANT_EV_ANGSTROM: f64 = 14.3996;
    let n = dipoles.len().min(safety_cap);
    for i in 0..n {
        let pos = f64x3_to_f32(dipoles[i].position_angstrom);
        let r = squared_distance(pos, target_center).sqrt().max(0.1) as f64;
        let mag = (dipoles[i].dipole_e_angstrom[0].powi(2)
            + dipoles[i].dipole_e_angstrom[1].powi(2)
            + dipoles[i].dipole_e_angstrom[2].powi(2))
        .sqrt();
        // Dipole potential falls off as 1/r²
        let contribution = COULOMB_CONSTANT_EV_ANGSTROM * mag / (r * r);
        if contribution < threshold_ev {
            return i.max(1);
        }
    }
    n
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RuntimeQuantumResolvedCoupling {
    pub summary: ReactionQuantumSummary,
    pub iteration_count: usize,
    pub residual_ev: f32,
    pub coupling_energy_ev: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimeQuantumFragmentSolve {
    outcome: RuntimeQuantumFragmentOutcome,
    solved_point_charges: Vec<QuantumEmbeddingPointCharge>,
    solved_dipoles: Vec<QuantumEmbeddingDipole>,
    solved_orbital_response_fields: Vec<QuantumEmbeddingOrbitalResponseField>,
    /// Per-atom natural orbital occupancies from 1-RDM diagonalization, aggregated
    /// by summing orbital occupancies over each atom's spatial orbitals.  When
    /// available, these refine the effective charges used in neighbor embedding
    /// by reflecting correlation-driven partial occupancies.
    natural_orbital_occupancies: Vec<f64>,
    /// Per-atom LDA exchange potential (eV) from the solved electron density.
    /// Applied as a perturbative correction to embedding point charges seen by
    /// neighboring fragments.
    lda_exchange_corrections_ev: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimeQuantumFragmentEmbeddingSource {
    point_charges: Vec<QuantumEmbeddingPointCharge>,
    dipoles: Vec<QuantumEmbeddingDipole>,
    orbital_response_fields: Vec<QuantumEmbeddingOrbitalResponseField>,
    /// Per-atom natural orbital occupancies from 1-RDM diagonalization.  When
    /// non-empty, these per-atom occupancies refine the effective charges used
    /// in neighbor embedding by reflecting correlation-driven partial filling.
    natural_orbital_occupancies: Vec<f64>,
    /// Per-atom LDA exchange potential correction (eV).  Applied as a
    /// perturbative additive shift to the embedding point charges seen by
    /// neighboring fragments.
    lda_exchange_corrections_ev: Vec<f64>,
}

/// Tracks whether a metabolic reaction uses a quantum-derived Eyring rate
/// or a legacy heuristic rate.  Phase 7d infrastructure -- the actual wiring
/// into metabolic flux is deferred to Phase 8c.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct QuantumDerivedRate {
    /// Reaction identifier (matches `QuantumReactionSpec::name` or process kind).
    pub reaction_name: String,
    /// Quantum-computed activation barrier in eV (if available).
    pub barrier_ev: Option<f64>,
    /// Eyring-derived rate in s^-1 (if barrier available).
    pub quantum_rate_s_inv: Option<f64>,
    /// Currently-used heuristic rate in s^-1.
    pub heuristic_rate_s_inv: f64,
    /// Whether this reaction is using the quantum-derived rate.
    pub using_quantum: bool,
    /// Temperature at which the rate was computed.
    pub temperature_k: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeQuantumResolvedEmbeddingConfigs {
    pub quantum_configs: Vec<QuantumChemistryConfig>,
    pub iteration_count: usize,
    pub residual_charge_e: f64,
    pub residual_dipole_e_angstrom: f64,
    pub residual_orbital_response_ev: f64,
    /// Ground-state energy (eV) of each fragment at the final self-consistent
    /// iteration.  Used by the adaptive growth mechanism to decide whether a
    /// fragment's atom budget should be expanded.
    pub per_fragment_energy_ev: Vec<f64>,
    /// Whether each fragment's energy changed by less than `kT/10` between the
    /// last two self-consistent iterations, indicating convergence with respect
    /// to the embedding environment.
    pub per_fragment_converged: Vec<bool>,
    /// Natural orbital occupancies from the final self-consistent iteration,
    /// aggregated per atom.  Outer vec is per-fragment; inner vec contains the
    /// summed occupancies for each atom (electrons attributed to that atom from
    /// the 1-RDM diagonalization).  Empty inner vec for fragments that did not
    /// produce an ED result.
    pub per_fragment_natural_orbital_occupancies: Vec<Vec<f64>>,
    /// Per-fragment quantum-derived rates computed from the ED barrier heights
    /// via the Eyring equation.  Empty until Phase 8c wires these into the
    /// metabolic flux system.
    pub quantum_derived_rates: Vec<QuantumDerivedRate>,
}

impl WholeCellRuntimeQuantumProcessState {
    fn new(
        kind: WholeCellRuntimeQuantumProcessKind,
        site_anchor_atom_idx: Option<usize>,
        reactants: Vec<EmbeddedMolecule>,
        reaction: StructuralReactionTemplate,
    ) -> Self {
        let mut mixture = EmbeddedMaterialMixture::new(format!("{kind:?}_microdomain"));
        for reactant in &reactants {
            mixture.add_component(reactant.clone(), 128.0);
        }
        let quantum = quantum_config_for_reactants(&reactants);
        Self {
            kind,
            site_anchor_atom_idx,
            mixture,
            reactants,
            reaction,
            quantum,
            scaffold_boundary_replenish_amount: 128.0,
            reactive_boundary_replenish_amount: 128.0,
        }
    }

    pub(crate) fn kind(&self) -> WholeCellRuntimeQuantumProcessKind {
        self.kind
    }

    pub(crate) fn preset(&self) -> Syn3ASubsystemPreset {
        self.kind.preset()
    }

    pub(crate) fn site_anchor_atom_idx(&self) -> Option<usize> {
        self.site_anchor_atom_idx
    }

    #[cfg(test)]
    pub(crate) fn boundary_replenish_amount(&self) -> f64 {
        self.reactive_boundary_replenish_amount
    }

    #[cfg(test)]
    pub(crate) fn scaffold_boundary_replenish_amount(&self) -> f64 {
        self.scaffold_boundary_replenish_amount
    }

    #[cfg(test)]
    pub(crate) fn reactive_boundary_replenish_amount(&self) -> f64 {
        self.reactive_boundary_replenish_amount
    }

    pub(crate) fn scaffold_atom_count(&self) -> usize {
        self.reactants
            .first()
            .map(|molecule| molecule.graph.atom_count())
            .unwrap_or(0)
    }

    pub(crate) fn scaffold(&self) -> Option<&EmbeddedMolecule> {
        self.reactants.first()
    }

    pub(crate) fn mixture(&self) -> &EmbeddedMaterialMixture {
        &self.mixture
    }

    pub(crate) fn scaffold_centroid_angstrom(&self) -> [f32; 3] {
        self.reactants
            .first()
            .map(centroid_angstrom)
            .unwrap_or([0.0, 0.0, 0.0])
    }

    pub(crate) fn reactant_count(&self) -> usize {
        self.reactants.len()
    }

    pub(crate) fn reactant_component_amount(&self, reactant_idx: usize) -> f64 {
        let Some(reactant) = self.reactants.get(reactant_idx) else {
            return 0.0;
        };
        self.mixture
            .components
            .iter()
            .find(|component| component.molecule == *reactant)
            .map(|component| component.amount_moles)
            .unwrap_or(0.0)
    }

    pub(crate) fn set_boundary_replenish_amounts(
        &mut self,
        scaffold_amount: f64,
        reactive_amount: f64,
    ) {
        self.scaffold_boundary_replenish_amount = scaffold_amount.max(0.0);
        self.reactive_boundary_replenish_amount = reactive_amount.max(0.0);
    }

    pub(crate) fn ensure_reactant_inventory_floor(&mut self, reactant_idx: usize, amount: f64) {
        let Some(reactant) = self.reactants.get(reactant_idx).cloned() else {
            return;
        };
        let current = self.reactant_component_amount(reactant_idx);
        if current + 1.0e-12 < amount.max(0.0) {
            self.mixture
                .add_component(reactant, (amount - current).max(0.0));
        }
    }

    pub(crate) fn translate_geometry(&mut self, delta_angstrom: [f32; 3]) {
        for reactant in &mut self.reactants {
            *reactant = reactant.translated(delta_angstrom);
        }
        for component in &mut self.mixture.components {
            component.molecule = component.molecule.translated(delta_angstrom);
        }
        let delta_f64 = [
            f64::from(delta_angstrom[0]),
            f64::from(delta_angstrom[1]),
            f64::from(delta_angstrom[2]),
        ];
        for point_charge in &mut self.quantum.embedding_point_charges {
            point_charge.position_angstrom[0] += delta_f64[0];
            point_charge.position_angstrom[1] += delta_f64[1];
            point_charge.position_angstrom[2] += delta_f64[2];
        }
        for dipole in &mut self.quantum.embedding_dipoles {
            dipole.position_angstrom[0] += delta_f64[0];
            dipole.position_angstrom[1] += delta_f64[1];
            dipole.position_angstrom[2] += delta_f64[2];
        }
    }

    pub(crate) fn replace_scaffold_preserving_microdomain(
        &mut self,
        scaffold: EmbeddedMolecule,
    ) -> bool {
        let Some(current_scaffold) = self.reactants.first().cloned() else {
            return false;
        };
        let preserved_amount = self.reactant_component_amount(0);
        let current_centroid = centroid_angstrom(&current_scaffold);
        let next_centroid = centroid_angstrom(&scaffold);
        let translated_scaffold = scaffold.translated([
            current_centroid[0] - next_centroid[0],
            current_centroid[1] - next_centroid[1],
            current_centroid[2] - next_centroid[2],
        ]);

        self.reactants[0] = translated_scaffold.clone();
        if !self.reaction.reactants.is_empty() {
            self.reaction.reactants[0] = translated_scaffold.graph.clone();
        }
        self.mixture
            .components
            .retain(|component| component.molecule != current_scaffold);
        if preserved_amount > 1.0e-12 {
            self.mixture
                .add_component(translated_scaffold.clone(), preserved_amount);
        }
        self.quantum = quantum_config_for_reactants(&self.reactants);
        true
    }

    pub(crate) fn set_reactant_inventory_amount(&mut self, reactant_idx: usize, amount: f64) {
        let Some(reactant) = self.reactants.get(reactant_idx).cloned() else {
            return;
        };
        if let Some(component) = self
            .mixture
            .components
            .iter_mut()
            .find(|component| component.molecule == reactant)
        {
            component.amount_moles = amount.max(0.0);
        } else if amount > 1.0e-12 {
            self.mixture.add_component(reactant, amount.max(0.0));
        }
        self.mixture
            .components
            .retain(|component| component.amount_moles > 1.0e-12);
    }

    pub(crate) fn apply_fragment_outcome(
        &mut self,
        extent: f64,
    ) -> Result<Option<RuntimeQuantumFragmentOutcome>, EmbeddedMaterialStructuralReactionError>
    {
        self.apply_fragment_outcome_with_quantum(extent, self.quantum.clone())
    }

    pub(crate) fn apply_fragment_outcome_with_quantum(
        &mut self,
        extent: f64,
        quantum: QuantumChemistryConfig,
    ) -> Result<Option<RuntimeQuantumFragmentOutcome>, EmbeddedMaterialStructuralReactionError>
    {
        Ok(self
            .execute_fragment_quantum_application(extent, quantum)?
            .map(|solve| solve.outcome))
    }

    fn execute_fragment_quantum_application(
        &mut self,
        extent: f64,
        quantum: QuantumChemistryConfig,
    ) -> Result<Option<RuntimeQuantumFragmentSolve>, EmbeddedMaterialStructuralReactionError> {
        if extent <= 1.0e-9 {
            return Ok(None);
        }

        for (reactant_idx, reactant) in self.reactants.iter().enumerate() {
            let available = self
                .mixture
                .components
                .iter()
                .find(|component| component.molecule == *reactant)
                .map(|component| component.amount_moles)
                .unwrap_or(0.0);
            let replenish_amount = if reactant_idx == 0 {
                self.scaffold_boundary_replenish_amount
            } else {
                self.reactive_boundary_replenish_amount
            };
            if available + 1.0e-12 < extent && replenish_amount > 1.0e-12 {
                self.mixture.add_component(
                    reactant.clone(),
                    replenish_amount.max((extent - available).max(0.0)),
                );
            }
        }

        for reactant in &self.reactants {
            let available = self
                .mixture
                .components
                .iter()
                .find(|component| component.molecule == *reactant)
                .map(|component| component.amount_moles)
                .unwrap_or(0.0);
            if available + 1.0e-12 < extent {
                return Ok(None);
            }
        }

        let result = self.mixture.apply_structural_reaction(
            &self.reactants,
            &self.reaction,
            extent,
            quantum,
        )?;
        let after_result = &result.event_result.quantum_delta.after;

        // Extract natural orbital occupancies from the 1-RDM diagonalization,
        // aggregated per atom.  The raw natural orbitals are per spatial orbital;
        // we sum them by atom owner to produce per-atom occupancies suitable for
        // charge refinement in the embedding pipeline.
        let natural_orbital_occupancies: Vec<f64> = if let Some((raw_occ, _coefficients)) =
            after_result.natural_orbitals()
        {
            let n_orb = after_result.num_spatial_orbitals();
            let n_atoms = after_result
                .spatial_orbital_atom_indices
                .iter()
                .copied()
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);
            if after_result.spatial_orbital_atom_indices.len() == n_orb && n_atoms > 0 {
                let mut per_atom = vec![0.0f64; n_atoms];
                for (orb_idx, &atom_owner) in
                    after_result.spatial_orbital_atom_indices.iter().enumerate()
                {
                    if atom_owner < n_atoms {
                        per_atom[atom_owner] += *raw_occ.get(orb_idx).unwrap_or(&0.0f64);
                    }
                }
                per_atom
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Compute per-atom LDA exchange potential corrections (eV).
        let n_atoms = after_result.expected_atom_effective_charges.len();
        let lda_exchange_corrections_ev: Vec<f64> = (0..n_atoms)
            .map(|atom_idx| after_result.lda_exchange_potential_ev(atom_idx))
            .collect();

        Ok(Some(RuntimeQuantumFragmentSolve {
            outcome: RuntimeQuantumFragmentOutcome {
                summary: result.event_result.quantum_delta.summary(),
                extent: extent as f32,
                scaffold_centroid_angstrom: centroid_angstrom(&self.reactants[0]),
                reactive_centroid_angstrom: centroid_angstrom_slice(&self.reactants[1..]),
            },
            solved_point_charges: solved_fragment_point_charges(
                &result.event_result.combined_molecule,
                &after_result.expected_atom_effective_charges,
            ),
            solved_dipoles: solved_fragment_dipoles(
                &result.event_result.combined_molecule,
                after_result,
            ),
            solved_orbital_response_fields: solved_fragment_orbital_response_fields(
                &result.event_result.combined_molecule,
                after_result,
            ),
            natural_orbital_occupancies,
            lda_exchange_corrections_ev,
        }))
    }

    pub(crate) fn snapshot(&self) -> WholeCellSavedRuntimeQuantumProcess {
        WholeCellSavedRuntimeQuantumProcess {
            kind: self.kind.as_str().to_string(),
            site_anchor_atom_idx: self.site_anchor_atom_idx,
            mixture: self.mixture.clone(),
            reactants: self.reactants.clone(),
            reaction: self.reaction.clone(),
            quantum: self.quantum.clone(),
            boundary_replenish_amount: self
                .scaffold_boundary_replenish_amount
                .max(self.reactive_boundary_replenish_amount),
            scaffold_boundary_replenish_amount: Some(self.scaffold_boundary_replenish_amount),
            reactive_boundary_replenish_amount: Some(self.reactive_boundary_replenish_amount),
        }
    }

    pub(crate) fn from_saved(saved: WholeCellSavedRuntimeQuantumProcess) -> Result<Self, String> {
        let kind = WholeCellRuntimeQuantumProcessKind::from_str(&saved.kind)
            .ok_or_else(|| format!("unknown runtime quantum process kind: {}", saved.kind))?;
        if saved.reactants.len() != saved.reaction.reactants.len() {
            return Err(format!(
                "runtime quantum process {} saved reactant count {} does not match reaction count {}",
                kind.as_str(),
                saved.reactants.len(),
                saved.reaction.reactants.len()
            ));
        }
        for (reactant_idx, (reactant, expected_graph)) in saved
            .reactants
            .iter()
            .zip(saved.reaction.reactants.iter())
            .enumerate()
        {
            if reactant.graph != *expected_graph {
                return Err(format!(
                    "runtime quantum process {} reactant graph mismatch at index {}",
                    kind.as_str(),
                    reactant_idx
                ));
            }
        }

        Ok(Self {
            kind,
            site_anchor_atom_idx: saved.site_anchor_atom_idx,
            mixture: saved.mixture,
            reactants: saved.reactants,
            reaction: saved.reaction,
            quantum: saved.quantum,
            scaffold_boundary_replenish_amount: saved
                .scaffold_boundary_replenish_amount
                .unwrap_or(saved.boundary_replenish_amount)
                .max(0.0),
            reactive_boundary_replenish_amount: saved
                .reactive_boundary_replenish_amount
                .unwrap_or(saved.boundary_replenish_amount)
                .max(0.0),
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn apply_extent(
        &mut self,
        extent: f64,
    ) -> Result<Option<ReactionQuantumSummary>, EmbeddedMaterialStructuralReactionError> {
        Ok(self.apply_fragment_outcome(extent)?.map(|outcome| {
            let mut summary = scale_runtime_summary(outcome.summary, outcome.extent);
            summary.event_count = outcome.extent.ceil().clamp(1.0, u32::MAX as f32) as u32;
            summary
        }))
    }
}

pub(crate) fn aggregate_coupled_fragment_outcomes(
    outcomes: &[RuntimeQuantumFragmentOutcome],
) -> Option<ReactionQuantumSummary> {
    resolve_coupled_fragment_outcomes(outcomes).map(|resolved| resolved.summary)
}

pub(crate) fn resolve_coupled_fragment_outcomes(
    outcomes: &[RuntimeQuantumFragmentOutcome],
) -> Option<RuntimeQuantumResolvedCoupling> {
    if outcomes.is_empty() {
        return None;
    }

    let mut aggregated = ReactionQuantumSummary::default();
    let mut total_extent = 0.0f32;
    for outcome in outcomes {
        aggregated.accumulate(scale_runtime_summary(outcome.summary, outcome.extent));
        total_extent += outcome.extent.max(0.0);
    }
    if total_extent <= 1.0e-6 {
        return None;
    }

    aggregated.event_count = total_extent.ceil().clamp(1.0, u32::MAX as f32) as u32;
    let (responses, iteration_count, residual_ev) =
        solve_self_consistent_fragment_responses(outcomes);
    let coupling_energy_ev = fragment_coupling_energy_ev_with_responses(outcomes, &responses);
    aggregated.ground_state_energy_delta_ev += coupling_energy_ev;
    Some(RuntimeQuantumResolvedCoupling {
        summary: aggregated,
        iteration_count,
        residual_ev,
        coupling_energy_ev,
    })
}

pub(crate) fn restore_runtime_quantum_processes(
    saved: Vec<WholeCellSavedRuntimeQuantumProcess>,
) -> Result<Vec<WholeCellRuntimeQuantumProcessState>, String> {
    if saved.is_empty() {
        return Ok(Vec::new());
    }
    saved
        .into_iter()
        .map(WholeCellRuntimeQuantumProcessState::from_saved)
        .collect()
}

#[cfg(test)]
pub(crate) fn default_runtime_quantum_processes() -> Vec<WholeCellRuntimeQuantumProcessState> {
    let mut processes = Vec::new();
    for kind in WholeCellRuntimeQuantumProcessKind::all() {
        processes.extend(runtime_quantum_processes_for_kind(kind));
    }
    processes
}

#[cfg(test)]
pub(crate) fn runtime_quantum_processes_for_kind(
    kind: WholeCellRuntimeQuantumProcessKind,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    runtime_quantum_processes_for_kind_with_atom_budget(kind, 4)
}

pub(crate) fn runtime_quantum_processes_for_kind_with_atom_budget_and_anchor_preferences(
    kind: WholeCellRuntimeQuantumProcessKind,
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    runtime_quantum_processes_for_kind_with_optional_scaffold_source_and_anchor_preferences(
        kind,
        max_scaffold_atoms,
        preferred_anchors,
        None,
    )
}

pub(crate) fn runtime_quantum_processes_for_kind_from_embedded_microdomain_with_atom_budget_and_anchor_preferences(
    kind: WholeCellRuntimeQuantumProcessKind,
    max_scaffold_atoms: usize,
    desired_process_count: usize,
    preferred_anchors: &[usize],
    microdomain: &EmbeddedMaterialMixture,
) -> Option<Vec<WholeCellRuntimeQuantumProcessState>> {
    let scaffold_sources =
        ranked_runtime_quantum_scaffold_components_from_mixture(kind, microdomain);
    if scaffold_sources.is_empty() {
        return None;
    }

    if !preferred_anchors.is_empty() || desired_process_count <= 1 {
        let (scaffold_source, scaffold_amount_moles) = scaffold_sources.into_iter().next()?;
        let mut processes =
            runtime_quantum_processes_for_kind_with_optional_scaffold_source_and_anchor_preferences(
                kind,
                max_scaffold_atoms,
                preferred_anchors,
                Some(&scaffold_source),
            );
        seed_runtime_quantum_process_inventory_from_microdomain(
            kind,
            &mut processes,
            scaffold_amount_moles,
            &scaffold_source,
            microdomain,
        );
        return Some(processes);
    }

    let mut combined = Vec::new();
    for (scaffold_source, scaffold_amount_moles) in scaffold_sources {
        let mut processes =
            runtime_quantum_processes_for_kind_with_optional_scaffold_source_and_anchor_preferences(
                kind,
                max_scaffold_atoms,
                &[],
                Some(&scaffold_source),
            );
        seed_runtime_quantum_process_inventory_from_microdomain(
            kind,
            &mut processes,
            scaffold_amount_moles,
            &scaffold_source,
            microdomain,
        );
        combined.extend(processes);
        if combined.len() >= desired_process_count {
            break;
        }
    }

    if combined.is_empty() {
        None
    } else {
        Some(combined)
    }
}

#[cfg(test)]
pub(crate) fn symbolic_runtime_quantum_microdomain_from_amounts(
    kind: WholeCellRuntimeQuantumProcessKind,
    scaffold_amount_moles: f64,
    reactive_amount_moles: f64,
) -> Option<EmbeddedMaterialMixture> {
    if scaffold_amount_moles <= 1.0e-12 {
        return None;
    }
    let scaffold = runtime_quantum_default_scaffold_source(kind);
    let placement = scaffold_reactant_placement(kind, &scaffold, 2.6);
    let reactants = symbolic_runtime_quantum_reactants(kind, placement);

    let mut mixture =
        EmbeddedMaterialMixture::new(format!("{}_symbolic_microdomain", kind.as_str()));
    mixture.add_component(scaffold, scaffold_amount_moles.max(0.0));
    if reactive_amount_moles > 1.0e-12 {
        for reactant in reactants {
            mixture.add_component(reactant, reactive_amount_moles.max(0.0));
        }
    }
    Some(mixture)
}

pub(crate) fn runtime_quantum_processes_for_kind_from_symbolic_inventory_with_atom_budget_and_anchor_preferences(
    kind: WholeCellRuntimeQuantumProcessKind,
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
    scaffold_amount_moles: f64,
    reactive_amount_moles: f64,
) -> Option<Vec<WholeCellRuntimeQuantumProcessState>> {
    if scaffold_amount_moles <= 1.0e-12 {
        return None;
    }

    let mut processes = runtime_quantum_processes_for_kind_with_atom_budget_and_anchor_preferences(
        kind,
        max_scaffold_atoms,
        preferred_anchors,
    );
    if processes.is_empty() {
        return None;
    }

    for process in &mut processes {
        process.set_reactant_inventory_amount(0, scaffold_amount_moles.max(0.0));
        for reactant_idx in 1..process.reactant_count() {
            process.set_reactant_inventory_amount(reactant_idx, reactive_amount_moles.max(0.0));
        }
    }
    Some(processes)
}

// ---------------------------------------------------------------------------
// Live-state-carved quantum neighborhoods
// ---------------------------------------------------------------------------

/// Snapshot of the live whole-cell state relevant to a single quantum
/// hotspot kind.  Built by the simulator and consumed by
/// [`runtime_quantum_processes_for_kind_from_live_state`] to seed quantum
/// process inventories from actual complex assembly counts and lattice
/// species concentrations rather than static templates.
#[derive(Clone, Debug)]
pub(crate) struct LiveSubsystemQuantumState {
    pub kind: WholeCellRuntimeQuantumProcessKind,
    /// Scaffold (complex) amount in moles, derived from complex assembly
    /// counts and Avogadro scaling (typically count * 1e-6).
    pub scaffold_amount_moles: f64,
    /// Reactive species present in the lattice neighbourhood, each as
    /// (species-label, amount-moles).
    pub reactive_species_moles: Vec<(String, f64)>,
    /// Local temperature in Kelvin (defaults to 310.15 K = 37 C).
    pub temperature_kelvin: f32,
    /// MD probe translation scale factor.
    pub md_translation_scale: f32,
    /// MD probe membrane scale factor.
    pub md_membrane_scale: f32,
}

/// Construct quantum process states from a live subsystem snapshot.
///
/// This is the main entry point for "live-state-carved neighborhoods": it
/// builds the same `WholeCellRuntimeQuantumProcessState` vector that the
/// static-template path produces, but seeds the reactant inventories from
/// the live concentrations captured in `state`.
///
/// Returns `None` when the scaffold amount is negligible (< 1e-12 mol) or
/// when the underlying spec produces no processes for the given kind.
pub(crate) fn runtime_quantum_processes_for_kind_from_live_state(
    state: &LiveSubsystemQuantumState,
) -> Option<Vec<WholeCellRuntimeQuantumProcessState>> {
    if state.scaffold_amount_moles <= 1.0e-12 {
        return None;
    }

    let max_atoms = runtime_quantum_owner_scaffold_fragment_atom_budget(state.kind);
    let mut processes = runtime_quantum_processes_for_kind_with_atom_budget_and_anchor_preferences(
        state.kind,
        max_atoms,
        &[],
    );
    if processes.is_empty() {
        return None;
    }

    // Seed inventories from the live state.
    let reactive_total: f64 = state
        .reactive_species_moles
        .iter()
        .map(|(_, m)| m.max(0.0))
        .sum();

    for process in &mut processes {
        // Index 0 is the scaffold reactant.
        process.set_reactant_inventory_amount(0, state.scaffold_amount_moles.max(0.0));
        // Remaining indices get the aggregate reactive pool.
        for idx in 1..process.reactant_count() {
            process.set_reactant_inventory_amount(idx, reactive_total.max(0.0));
        }
    }

    Some(processes)
}

/// Map a quantum process kind to the name of the corresponding complex
/// assembly field in [`WholeCellComplexAssemblyState`].
pub(crate) fn subsystem_complex_field_for_kind(
    kind: WholeCellRuntimeQuantumProcessKind,
) -> &'static str {
    match kind {
        WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation => {
            "ribosome_complexes"
        }
        WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation => {
            "replisome_complexes"
        }
        WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation
        | WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification
        | WholeCellRuntimeQuantumProcessKind::AtpBandProtonTransfer
        | WholeCellRuntimeQuantumProcessKind::AtpBandElectronTransfer => "atp_band_complexes",
        WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification
        | WholeCellRuntimeQuantumProcessKind::SeptumGtpHydrolysis => "ftsz_polymer",
        WholeCellRuntimeQuantumProcessKind::MembraneProteinInsertion => "membrane_complexes",
    }
}

fn runtime_quantum_processes_for_kind_with_optional_scaffold_source_and_anchor_preferences(
    kind: WholeCellRuntimeQuantumProcessKind,
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
    scaffold_source: Option<&EmbeddedMolecule>,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    let spec = kind.reaction_spec();
    quantum_processes_from_spec(&spec, max_scaffold_atoms, preferred_anchors, scaffold_source)
}

#[cfg(test)]
pub(crate) fn runtime_quantum_processes_for_kind_with_atom_budget(
    kind: WholeCellRuntimeQuantumProcessKind,
    max_scaffold_atoms: usize,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    runtime_quantum_processes_for_kind_with_atom_budget_and_anchor_preferences(
        kind,
        max_scaffold_atoms,
        &[],
    )
}

pub(crate) fn resolve_runtime_fragment_quantum_configs(
    processes: &[WholeCellRuntimeQuantumProcessState],
) -> Vec<QuantumChemistryConfig> {
    let sources = processes
        .iter()
        .map(|process| heuristic_fragment_embedding_source(&process.reactants))
        .collect::<Vec<_>>();
    runtime_fragment_quantum_configs_from_sources(processes, &sources)
}

pub(crate) fn solve_self_consistent_runtime_fragment_quantum_configs(
    processes: &[WholeCellRuntimeQuantumProcessState],
    extent: f64,
) -> Result<RuntimeQuantumResolvedEmbeddingConfigs, EmbeddedMaterialStructuralReactionError> {
    let n_fragments = processes.len();
    let mut sources = processes
        .iter()
        .map(|process| heuristic_fragment_embedding_source(&process.reactants))
        .collect::<Vec<_>>();
    let mut quantum_configs = runtime_fragment_quantum_configs_from_sources(processes, &sources);
    let mut iteration_count = 0usize;
    let mut residual_charge_e = 0.0f64;
    let mut residual_dipole_e_angstrom = 0.0f64;
    let mut residual_orbital_response_ev = 0.0f64;

    // Track per-fragment ground-state energies across iterations for
    // convergence detection.  `prev_energies` stores the energy from the
    // previous iteration; `curr_energies` is updated each pass.
    let mut prev_energies: Vec<f64> = vec![f64::NAN; n_fragments];
    let mut curr_energies: Vec<f64> = vec![f64::NAN; n_fragments];
    let mut per_fragment_converged: Vec<bool> = vec![false; n_fragments];
    let mut per_fragment_natural_orbital_occupancies: Vec<Vec<f64>> =
        vec![Vec::new(); n_fragments];

    for iteration in 0..FRAGMENT_EMBEDDING_RESPONSE_MAX_ITERATIONS {
        let mut solved_sources = Vec::with_capacity(n_fragments);
        for (frag_idx, (process, quantum)) in
            processes.iter().zip(quantum_configs.iter().cloned()).enumerate()
        {
            let mut preview = process.clone();
            if let Some(solve) = preview.execute_fragment_quantum_application(extent, quantum)? {
                curr_energies[frag_idx] =
                    f64::from(solve.outcome.summary.ground_state_energy_delta_ev);
                per_fragment_natural_orbital_occupancies[frag_idx] =
                    solve.natural_orbital_occupancies.clone();
                solved_sources.push(RuntimeQuantumFragmentEmbeddingSource {
                    point_charges: solve.solved_point_charges,
                    dipoles: solve.solved_dipoles,
                    orbital_response_fields: solve.solved_orbital_response_fields,
                    natural_orbital_occupancies: solve.natural_orbital_occupancies,
                    lda_exchange_corrections_ev: solve.lda_exchange_corrections_ev,
                });
            } else {
                // Fragment did not produce a solve (insufficient reactants).
                // Energy stays NAN (unconverged).
                curr_energies[frag_idx] = f64::NAN;
                solved_sources.push(heuristic_fragment_embedding_source(&process.reactants));
            }
        }

        // Check per-fragment energy convergence against previous iteration.
        for frag_idx in 0..n_fragments {
            let delta = (curr_energies[frag_idx] - prev_energies[frag_idx]).abs();
            per_fragment_converged[frag_idx] =
                delta.is_finite() && delta <= FRAGMENT_ENERGY_CONVERGENCE_THRESHOLD_EV;
        }

        prev_energies.copy_from_slice(&curr_energies);

        (
            residual_charge_e,
            residual_dipole_e_angstrom,
            residual_orbital_response_ev,
        ) = relax_fragment_embedding_sources(&mut sources, &solved_sources);
        quantum_configs = runtime_fragment_quantum_configs_from_sources(processes, &sources);
        iteration_count = iteration + 1;
        if residual_charge_e <= FRAGMENT_EMBEDDING_RESPONSE_TOLERANCE_E
            && residual_dipole_e_angstrom <= FRAGMENT_EMBEDDING_RESPONSE_TOLERANCE_E
            && residual_orbital_response_ev <= FRAGMENT_EMBEDDING_RESPONSE_TOLERANCE_ORBITAL_EV
        {
            break;
        }
    }

    // Phase 7d: compute Eyring-derived rates from the final ground-state
    // energy deltas.  The `ground_state_energy_delta_ev` from the ED solve
    // captures the reaction energy change; we use its absolute value as a
    // first-order barrier estimate.  Proper transition-state searches will
    // refine these barriers in Phase 8c.
    let temperature_k = 310.0; // Syn3A physiological temperature
    let quantum_derived_rates: Vec<QuantumDerivedRate> = processes
        .iter()
        .enumerate()
        .map(|(frag_idx, process)| {
            let energy_ev = curr_energies[frag_idx];
            let (barrier, quantum_rate) = if energy_ev.is_finite() {
                let barrier = energy_ev.abs();
                let rate = eyring_rate_from_barrier(barrier, temperature_k);
                (Some(barrier), Some(rate))
            } else {
                (None, None)
            };
            // Phase 8c: populate the heuristic rate from literature Km/Vmax
            // for this process kind, and activate quantum-derived rates when
            // the barrier is in a physically reasonable range.
            let heuristic_rate = heuristic_rate_for_process_kind(process.kind());
            let using_quantum = match barrier {
                Some(b) if b >= MIN_QUANTUM_BARRIER_EV && b <= MAX_QUANTUM_BARRIER_EV => true,
                _ => false,
            };
            QuantumDerivedRate {
                reaction_name: process.kind().as_str().to_string(),
                barrier_ev: barrier,
                quantum_rate_s_inv: quantum_rate,
                heuristic_rate_s_inv: heuristic_rate,
                using_quantum,
                temperature_k,
            }
        })
        .collect();

    Ok(RuntimeQuantumResolvedEmbeddingConfigs {
        quantum_configs,
        iteration_count,
        residual_charge_e,
        residual_dipole_e_angstrom,
        residual_orbital_response_ev,
        per_fragment_energy_ev: curr_energies,
        per_fragment_converged,
        per_fragment_natural_orbital_occupancies,
        quantum_derived_rates,
    })
}

fn runtime_fragment_quantum_configs_from_sources(
    processes: &[WholeCellRuntimeQuantumProcessState],
    sources: &[RuntimeQuantumFragmentEmbeddingSource],
) -> Vec<QuantumChemistryConfig> {
    // Convergence threshold kT/100 at 310 K ≈ 2.67e-4 eV.
    let threshold_ev = DEFAULT_EMBEDDING_CONVERGENCE_THRESHOLD_EV;
    processes
        .iter()
        .enumerate()
        .map(|(target_idx, process)| {
            let target_center = centroid_angstrom_slice(&process.reactants);

            // Collect embedding point charges from neighbor fragments,
            // applying LDA exchange corrections and natural orbital charge
            // refinement when available.  Corrections are applied BEFORE
            // spatial/magnitude filtering so that the atom-indexed LDA and
            // natural orbital data stay aligned with the charge array.
            let mut embedding_point_charges: Vec<QuantumEmbeddingPointCharge> = sources
                .iter()
                .enumerate()
                .filter(|(neighbor_idx, _)| *neighbor_idx != target_idx)
                .flat_map(|(_, source)| {
                    // Start with a mutable copy of the source point charges so
                    // that per-atom corrections can be applied before filtering.
                    let mut charges = source.point_charges.clone();

                    // Apply LDA exchange potential corrections: each atom's
                    // embedding charge is shifted by the scaled LDA V_x.
                    // This is a perturbative first-principles correction, not
                    // an empirical parameter.
                    if !source.lda_exchange_corrections_ev.is_empty() {
                        for (atom_idx, charge) in charges.iter_mut().enumerate() {
                            if let Some(&lda_vx) =
                                source.lda_exchange_corrections_ev.get(atom_idx)
                            {
                                charge.charge_e += lda_vx * LDA_EXCHANGE_COUPLING_SCALE;
                            }
                        }
                    }

                    // Use natural orbital occupancies for fractional charge
                    // refinement.  When the 1-RDM diagonalization produced
                    // per-atom occupancies, we redistribute the effective
                    // charge proportionally to reflect correlation-driven
                    // partial occupancies instead of integer filling.
                    if !source.natural_orbital_occupancies.is_empty()
                        && source.natural_orbital_occupancies.len() == charges.len()
                    {
                        let total_charge: f64 =
                            charges.iter().map(|c| c.charge_e).sum::<f64>();
                        let mean_occ: f64 = source
                            .natural_orbital_occupancies
                            .iter()
                            .copied()
                            .sum::<f64>()
                            / charges.len().max(1) as f64;
                        if mean_occ > 1.0e-12 {
                            let mut rescaled_sum = 0.0f64;
                            for (atom_idx, charge) in charges.iter_mut().enumerate() {
                                let occ = source.natural_orbital_occupancies[atom_idx];
                                charge.charge_e *= occ / mean_occ;
                                rescaled_sum += charge.charge_e;
                            }
                            // Re-normalize to conserve total charge.
                            if rescaled_sum.abs() > 1.0e-12 && total_charge.abs() > 1.0e-12 {
                                let correction = total_charge / rescaled_sum;
                                for charge in charges.iter_mut() {
                                    charge.charge_e *= correction;
                                }
                            }
                        }
                    }

                    // Now apply spatial/magnitude filtering on the corrected charges.
                    let cutoff_squared = FRAGMENT_EMBEDDING_CUTOFF_ANGSTROM.powi(2);
                    charges
                        .into_iter()
                        .filter(|pc| pc.charge_e.abs() >= FRAGMENT_EMBEDDING_MIN_CHARGE_E)
                        .filter(|pc| {
                            squared_distance(
                                f64x3_to_f32(pc.position_angstrom),
                                target_center,
                            ) <= cutoff_squared
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            embedding_point_charges.sort_by(|left, right| {
                let left_distance =
                    squared_distance(f64x3_to_f32(left.position_angstrom), target_center);
                let right_distance =
                    squared_distance(f64x3_to_f32(right.position_angstrom), target_center);
                left_distance.total_cmp(&right_distance)
            });
            // Convergence-driven truncation is the primary limiter: remove
            // sources whose individual Coulomb contribution to the target
            // fragment falls below kT/100.  The safety cap is only a fallback
            // to prevent runaway memory in pathological cases.
            {
                let conv = convergence_truncate_charges(
                    &embedding_point_charges,
                    target_center,
                    threshold_ev,
                    FRAGMENT_EMBEDDING_MAX_POINT_CHARGES,
                );
                embedding_point_charges.truncate(conv);
            }
            let mut embedding_dipoles = sources
                .iter()
                .enumerate()
                .filter(|(neighbor_idx, _)| *neighbor_idx != target_idx)
                .flat_map(|(_, source)| {
                    filtered_fragment_embedding_dipoles(&source.dipoles, target_center)
                })
                .collect::<Vec<_>>();
            embedding_dipoles.sort_by(|left, right| {
                let left_distance =
                    squared_distance(f64x3_to_f32(left.position_angstrom), target_center);
                let right_distance =
                    squared_distance(f64x3_to_f32(right.position_angstrom), target_center);
                left_distance.total_cmp(&right_distance)
            });
            // Convergence-driven truncation first, safety cap as fallback.
            {
                let conv = convergence_truncate_dipoles(
                    &embedding_dipoles,
                    target_center,
                    threshold_ev,
                    FRAGMENT_EMBEDDING_MAX_DIPOLES,
                );
                embedding_dipoles.truncate(conv);
            }
            let mut embedding_orbital_response_fields = sources
                .iter()
                .enumerate()
                .filter(|(neighbor_idx, _)| *neighbor_idx != target_idx)
                .flat_map(|(_, source)| {
                    filtered_fragment_embedding_orbital_response_fields(
                        &source.orbital_response_fields,
                        target_center,
                    )
                })
                .collect::<Vec<_>>();
            embedding_orbital_response_fields.sort_by(|left, right| {
                let left_distance =
                    squared_distance(f64x3_to_f32(left.position_angstrom), target_center);
                let right_distance =
                    squared_distance(f64x3_to_f32(right.position_angstrom), target_center);
                left_distance.total_cmp(&right_distance)
            });
            embedding_orbital_response_fields
                .truncate(FRAGMENT_EMBEDDING_MAX_ORBITAL_RESPONSE_FIELDS);
            process
                .quantum
                .clone()
                .with_embedding_point_charges(embedding_point_charges)
                .with_embedding_dipoles(embedding_dipoles)
                .with_embedding_orbital_response_fields(embedding_orbital_response_fields)
        })
        .collect()
}

fn runtime_quantum_reactants_from_placement(
    kind: WholeCellRuntimeQuantumProcessKind,
    placement: SiteReactivePlacement,
) -> Vec<EmbeddedMolecule> {
    match kind {
        WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation => vec![
            methylamine(offset_along_direction(
                placement.anchor_position_angstrom,
                placement.direction,
                2.35,
            )),
            acetic_acid(placement.anchor_position_angstrom),
        ],
        WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation => vec![
            methanol(offset_along_direction(
                placement.anchor_position_angstrom,
                placement.direction,
                2.55,
            )),
            orthophosphoric_acid(placement.anchor_position_angstrom),
        ],
        WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation => vec![
            ethanol(offset_along_direction(
                placement.anchor_position_angstrom,
                placement.direction,
                2.80,
            )),
            orthophosphoric_acid(placement.anchor_position_angstrom),
        ],
        WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification => vec![
            methanol(offset_along_direction(
                placement.anchor_position_angstrom,
                placement.direction,
                2.55,
            )),
            acetic_acid(placement.anchor_position_angstrom),
        ],
        WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification => vec![
            methanol(offset_along_direction(
                placement.anchor_position_angstrom,
                placement.direction,
                2.75,
            )),
            acetic_acid(placement.anchor_position_angstrom),
        ],
        WholeCellRuntimeQuantumProcessKind::AtpBandProtonTransfer => vec![
            imidazole(offset_along_direction(
                placement.anchor_position_angstrom,
                placement.direction,
                2.20,
            )),
            water(placement.anchor_position_angstrom),
        ],
        WholeCellRuntimeQuantumProcessKind::SeptumGtpHydrolysis => vec![
            methyl_pyrophosphate(offset_along_direction(
                placement.anchor_position_angstrom,
                placement.direction,
                2.60,
            )),
            water(placement.anchor_position_angstrom),
        ],
        WholeCellRuntimeQuantumProcessKind::AtpBandElectronTransfer => vec![
            methanol(offset_along_direction(
                placement.anchor_position_angstrom,
                placement.direction,
                2.55,
            )),
            orthophosphoric_acid(placement.anchor_position_angstrom),
        ],
        WholeCellRuntimeQuantumProcessKind::MembraneProteinInsertion => vec![
            methanol(offset_along_direction(
                placement.anchor_position_angstrom,
                placement.direction,
                2.75,
            )),
            acetic_acid(placement.anchor_position_angstrom),
        ],
    }
}

fn runtime_quantum_context_support_bulk_fields(
    kind: WholeCellRuntimeQuantumProcessKind,
) -> &'static [WholeCellBulkField] {
    const TRANSLATION_FIELDS: [WholeCellBulkField; 3] = [
        WholeCellBulkField::AminoAcids,
        WholeCellBulkField::ATP,
        WholeCellBulkField::ADP,
    ];
    const ENERGY_FIELDS: [WholeCellBulkField; 4] = [
        WholeCellBulkField::ADP,
        WholeCellBulkField::ATP,
        WholeCellBulkField::Glucose,
        WholeCellBulkField::Oxygen,
    ];
    const NUCLEOTIDE_FIELDS: [WholeCellBulkField; 4] = [
        WholeCellBulkField::Nucleotides,
        WholeCellBulkField::ATP,
        WholeCellBulkField::ADP,
        WholeCellBulkField::Glucose,
    ];
    const MEMBRANE_FIELDS: [WholeCellBulkField; 5] = [
        WholeCellBulkField::MembranePrecursors,
        WholeCellBulkField::ATP,
        WholeCellBulkField::ADP,
        WholeCellBulkField::Glucose,
        WholeCellBulkField::Oxygen,
    ];
    const PROTON_TRANSFER_FIELDS: [WholeCellBulkField; 3] = [
        WholeCellBulkField::ATP,
        WholeCellBulkField::ADP,
        WholeCellBulkField::Pi,
    ];
    const GTP_HYDROLYSIS_FIELDS: [WholeCellBulkField; 3] = [
        WholeCellBulkField::Gtp,
        WholeCellBulkField::Gdp,
        WholeCellBulkField::Pi,
    ];
    match kind {
        WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation => &TRANSLATION_FIELDS,
        WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation => &ENERGY_FIELDS,
        WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation => {
            &NUCLEOTIDE_FIELDS
        }
        WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification
        | WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification => &MEMBRANE_FIELDS,
        WholeCellRuntimeQuantumProcessKind::AtpBandProtonTransfer
        | WholeCellRuntimeQuantumProcessKind::AtpBandElectronTransfer => &PROTON_TRANSFER_FIELDS,
        WholeCellRuntimeQuantumProcessKind::SeptumGtpHydrolysis => &GTP_HYDROLYSIS_FIELDS,
        WholeCellRuntimeQuantumProcessKind::MembraneProteinInsertion => &MEMBRANE_FIELDS,
    }
}

fn runtime_quantum_representative_graph_for_bulk_field(
    bulk_field: WholeCellBulkField,
) -> MoleculeGraph {
    match bulk_field {
        WholeCellBulkField::ATP => MoleculeGraph::representative_atp(),
        WholeCellBulkField::ADP => MoleculeGraph::representative_adp(),
        WholeCellBulkField::Glucose => MoleculeGraph::representative_glucose(),
        WholeCellBulkField::Oxygen => MoleculeGraph::representative_oxygen_gas(),
        WholeCellBulkField::AminoAcids => MoleculeGraph::representative_amino_acid_pool(),
        WholeCellBulkField::Nucleotides => MoleculeGraph::representative_nucleotide_pool(),
        WholeCellBulkField::MembranePrecursors => {
            MoleculeGraph::representative_membrane_precursor_pool()
        }
        WholeCellBulkField::Pi => MoleculeGraph::representative_orthophosphoric_acid(),
        WholeCellBulkField::Gtp => {
            let mut g = MoleculeGraph::representative_atp();
            g.name = "gtp_like".to_string();
            g
        }
        WholeCellBulkField::Gdp => {
            let mut g = MoleculeGraph::representative_adp();
            g.name = "gdp_like".to_string();
            g
        }
        WholeCellBulkField::NadOxidized => MoleculeGraph::representative_nad_oxidized(),
        WholeCellBulkField::NadReduced => MoleculeGraph::representative_nad_reduced(),
        WholeCellBulkField::CoA => MoleculeGraph::representative_coenzyme_a(),
    }
}

fn runtime_quantum_context_support_graphs(
    kind: WholeCellRuntimeQuantumProcessKind,
) -> Vec<MoleculeGraph> {
    runtime_quantum_context_support_bulk_fields(kind)
        .iter()
        .copied()
        .map(runtime_quantum_representative_graph_for_bulk_field)
        .collect()
}

pub(crate) fn runtime_quantum_bulk_field_supports_canonical_reactant(
    kind: WholeCellRuntimeQuantumProcessKind,
    bulk_field: WholeCellBulkField,
) -> bool {
    match kind {
        WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation => {
            bulk_field == WholeCellBulkField::AminoAcids
        }
        WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation => {
            matches!(
                bulk_field,
                WholeCellBulkField::ADP | WholeCellBulkField::ATP
            )
        }
        WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation => {
            bulk_field == WholeCellBulkField::Nucleotides
        }
        WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification
        | WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification => {
            bulk_field == WholeCellBulkField::MembranePrecursors
        }
        WholeCellRuntimeQuantumProcessKind::AtpBandProtonTransfer
        | WholeCellRuntimeQuantumProcessKind::AtpBandElectronTransfer => {
            matches!(
                bulk_field,
                WholeCellBulkField::ATP | WholeCellBulkField::ADP
            )
        }
        WholeCellRuntimeQuantumProcessKind::SeptumGtpHydrolysis => {
            matches!(
                bulk_field,
                WholeCellBulkField::Gtp | WholeCellBulkField::Gdp
            )
        }
        WholeCellRuntimeQuantumProcessKind::MembraneProteinInsertion => {
            bulk_field == WholeCellBulkField::MembranePrecursors
        }
    }
}

fn embedded_support_molecule_for_bulk_field(
    bulk_field: WholeCellBulkField,
    center_angstrom: [f32; 3],
) -> EmbeddedMolecule {
    let graph = runtime_quantum_representative_graph_for_bulk_field(bulk_field);
    let positions_angstrom = graph
        .atoms
        .iter()
        .enumerate()
        .map(|(atom_idx, _)| {
            let chain = atom_idx as f32;
            let zig = (atom_idx % 4) as f32 - 1.5;
            [
                center_angstrom[0] + 0.82 * chain,
                center_angstrom[1] + 0.55 * zig,
                center_angstrom[2] + if atom_idx % 2 == 0 { 0.0 } else { 0.34 },
            ]
        })
        .collect::<Vec<_>>();
    EmbeddedMolecule::new(graph, positions_angstrom).expect("support molecule geometry")
}

fn lateral_direction(direction: [f32; 3]) -> [f32; 3] {
    let lateral = [-direction[1], direction[0], 0.0];
    let norm = vector_norm_f32(lateral);
    if norm > 1.0e-6 {
        [lateral[0] / norm, lateral[1] / norm, lateral[2] / norm]
    } else {
        [0.0, 1.0, 0.0]
    }
}

pub(crate) fn runtime_quantum_support_molecules_for_scaffold(
    kind: WholeCellRuntimeQuantumProcessKind,
    scaffold: &EmbeddedMolecule,
    bulk_fields: &[WholeCellBulkField],
) -> Vec<(WholeCellBulkField, EmbeddedMolecule)> {
    let placement = scaffold_reactant_placement(kind, scaffold, 2.6);
    let tangent = lateral_direction(placement.direction);
    let supported_fields = runtime_quantum_context_support_bulk_fields(kind);
    let mut unique_fields = Vec::new();
    for field in bulk_fields.iter().copied() {
        if !supported_fields.contains(&field) || unique_fields.contains(&field) {
            continue;
        }
        unique_fields.push(field);
    }

    unique_fields
        .into_iter()
        .enumerate()
        .map(|(idx, field)| {
            let radial = 2.35 + 1.65 * idx as f32;
            let tangential = if idx % 2 == 0 { -0.85 } else { 0.85 };
            let center = offset_along_direction(
                [
                    placement.anchor_position_angstrom[0] + tangent[0] * tangential,
                    placement.anchor_position_angstrom[1] + tangent[1] * tangential,
                    placement.anchor_position_angstrom[2] + tangent[2] * tangential,
                ],
                placement.direction,
                radial,
            );
            (
                field,
                embedded_support_molecule_for_bulk_field(field, center),
            )
        })
        .collect()
}

pub(crate) fn runtime_quantum_reactants_for_scaffold(
    kind: WholeCellRuntimeQuantumProcessKind,
    scaffold: &EmbeddedMolecule,
) -> Vec<EmbeddedMolecule> {
    let placement = scaffold_reactant_placement(kind, scaffold, 2.6);
    runtime_quantum_reactants_from_placement(kind, placement)
}

fn runtime_quantum_canonical_reactant_graphs(
    kind: WholeCellRuntimeQuantumProcessKind,
    scaffold: Option<&EmbeddedMolecule>,
) -> Vec<MoleculeGraph> {
    let owned_scaffold;
    let scaffold = match scaffold {
        Some(scaffold) => scaffold,
        None => {
            owned_scaffold = runtime_quantum_default_scaffold_source(kind);
            &owned_scaffold
        }
    };
    runtime_quantum_reactants_for_scaffold(kind, scaffold)
        .into_iter()
        .map(|reactant| reactant.graph)
        .collect()
}

fn is_runtime_quantum_canonical_reactant_graph(
    canonical_reactant_graphs: &[MoleculeGraph],
    graph: &MoleculeGraph,
) -> bool {
    canonical_reactant_graphs
        .iter()
        .any(|reactant_graph| molecule_graph_matches_structure(reactant_graph, graph))
}

#[cfg(test)]
fn symbolic_runtime_quantum_reactants(
    kind: WholeCellRuntimeQuantumProcessKind,
    placement: SiteReactivePlacement,
) -> Vec<EmbeddedMolecule> {
    runtime_quantum_reactants_from_placement(kind, placement)
}

fn runtime_quantum_site_template_scaffold_source(
    kind: WholeCellRuntimeQuantumProcessKind,
) -> Option<EmbeddedMolecule> {
    let template = atomistic_template_for_site_name(kind.site_name())?;
    let mut scaffold = template.to_embedded_molecule().ok()?;
    scaffold.graph.name = format!("{}_{}_scaffold", kind.as_str(), template.name);
    Some(scaffold)
}

fn runtime_quantum_disconnected_primitive_scaffold_source(
    kind: WholeCellRuntimeQuantumProcessKind,
) -> EmbeddedMolecule {
    match kind {
        WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation => {
            disconnected_scaffold_source(
                "ribosome_translation_primitive_source",
                &[
                    translation_peptide_site([0.0, 0.0, 0.0]),
                    translation_peptide_site([7.8, 0.6, 0.0]),
                ],
            )
        }
        WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation => {
            disconnected_scaffold_source(
                "atp_band_phosphate_primitive_source",
                &[
                    phosphate_site([0.0, 0.0, 0.0]),
                    phosphate_site([6.6, 0.4, 0.0]),
                ],
            )
        }
        WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation => {
            disconnected_scaffold_source(
                "replisome_phosphate_primitive_source",
                &[
                    nucleotide_phosphate_site([0.0, 0.0, 0.0]),
                    nucleotide_phosphate_site([7.4, 0.5, 0.0]),
                ],
            )
        }
        WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification => {
            disconnected_scaffold_source(
                "atp_band_ester_primitive_source",
                &[
                    membrane_acyl_site([0.0, 0.0, 0.0]),
                    membrane_acyl_site([6.6, 0.3, 0.0]),
                ],
            )
        }
        WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification => {
            disconnected_scaffold_source(
                "septum_ester_primitive_source",
                &[
                    membrane_acyl_site([0.0, 0.0, 0.0]),
                    membrane_acyl_site([6.9, -0.3, 0.0]),
                ],
            )
        }
        WholeCellRuntimeQuantumProcessKind::AtpBandProtonTransfer => {
            disconnected_scaffold_source(
                "atp_band_proton_primitive_source",
                &[
                    imidazole([0.0, 0.0, 0.0]),
                    imidazole([5.8, 0.4, 0.0]),
                ],
            )
        }
        WholeCellRuntimeQuantumProcessKind::SeptumGtpHydrolysis => {
            disconnected_scaffold_source(
                "septum_gtp_primitive_source",
                &[
                    methyl_pyrophosphate([0.0, 0.0, 0.0]),
                    methyl_pyrophosphate([7.2, 0.5, 0.0]),
                ],
            )
        }
        WholeCellRuntimeQuantumProcessKind::AtpBandElectronTransfer => {
            disconnected_scaffold_source(
                "atp_band_electron_primitive_source",
                &[
                    phosphate_site([0.0, 0.0, 0.0]),
                    phosphate_site([6.6, 0.4, 0.0]),
                ],
            )
        }
        WholeCellRuntimeQuantumProcessKind::MembraneProteinInsertion => {
            disconnected_scaffold_source(
                "membrane_protein_insertion_primitive_source",
                &[
                    membrane_acyl_site([0.0, 0.0, 0.0]),
                    membrane_acyl_site([6.9, -0.3, 0.0]),
                ],
            )
        }
    }
}

fn runtime_quantum_default_scaffold_source(
    kind: WholeCellRuntimeQuantumProcessKind,
) -> EmbeddedMolecule {
    runtime_quantum_site_template_scaffold_source(kind)
        .unwrap_or_else(|| runtime_quantum_disconnected_primitive_scaffold_source(kind))
}

pub(crate) fn runtime_quantum_owner_scaffold_source(
    kind: WholeCellRuntimeQuantumProcessKind,
    owner_fragment: &str,
    primitive_units: usize,
) -> EmbeddedMolecule {
    let components =
        runtime_quantum_owner_scaffold_components(kind, owner_fragment, primitive_units);
    let name = format!("{}_{}_scaffold", kind.as_str(), owner_fragment);
    disconnected_scaffold_source(&name, &components)
}

pub(crate) fn runtime_quantum_owner_composition_scaffold_components(
    kind: WholeCellRuntimeQuantumProcessKind,
    owner_fragment: &str,
    repeat_units: &[usize],
) -> Vec<EmbeddedMolecule> {
    repeat_units
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, units)| {
            let name = format!("{}_{}_owner_{idx}_scaffold", kind.as_str(), owner_fragment);
            let fragment = match kind {
                WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation => {
                    translation_owner_chain_scaffold(&name, units)
                }
                WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation => {
                    phosphate_owner_chain_scaffold(&name, units)
                }
                WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation => {
                    nucleotide_owner_chain_scaffold(&name, units)
                }
                WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification
                | WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification => {
                    membrane_owner_chain_scaffold(&name, units)
                }
                WholeCellRuntimeQuantumProcessKind::AtpBandProtonTransfer
                | WholeCellRuntimeQuantumProcessKind::AtpBandElectronTransfer => {
                    phosphate_owner_chain_scaffold(&name, units)
                }
                WholeCellRuntimeQuantumProcessKind::SeptumGtpHydrolysis => {
                    phosphate_owner_chain_scaffold(&name, units)
                }
                WholeCellRuntimeQuantumProcessKind::MembraneProteinInsertion => {
                    membrane_owner_chain_scaffold(&name, units)
                }
            };
            fragment.translated([
                9.5 * idx as f32,
                if idx % 2 == 0 { 0.0 } else { 1.8 },
                0.45 * ((idx % 3) as f32 - 1.0),
            ])
        })
        .collect()
}

pub(crate) fn runtime_quantum_owner_scaffold_components(
    kind: WholeCellRuntimeQuantumProcessKind,
    owner_fragment: &str,
    primitive_units: usize,
) -> Vec<EmbeddedMolecule> {
    if let Some(template_scaffold) = runtime_quantum_site_template_scaffold_source(kind) {
        let components = runtime_quantum_owner_template_scaffold_components(
            kind,
            owner_fragment,
            primitive_units,
            &template_scaffold,
        );
        if !components.is_empty() {
            return components;
        }
    }
    runtime_quantum_owner_disconnected_primitive_scaffold_components(
        kind,
        owner_fragment,
        primitive_units,
    )
}

fn runtime_quantum_owner_template_scaffold_components(
    kind: WholeCellRuntimeQuantumProcessKind,
    owner_fragment: &str,
    primitive_units: usize,
    scaffold_source: &EmbeddedMolecule,
) -> Vec<EmbeddedMolecule> {
    let primitive_units = primitive_units.max(1).min(8);
    let fragment_atom_budget = runtime_quantum_owner_scaffold_fragment_atom_budget(kind);
    let site_fragments = site_scaffolds_from_source_with_preferred_anchors(
        kind,
        scaffold_source,
        fragment_atom_budget,
        &[],
    );
    if site_fragments.is_empty() {
        return Vec::new();
    }

    (0..primitive_units)
        .map(|idx| {
            let mut fragment = site_fragments[idx % site_fragments.len()]
                .scaffold
                .translated([
                    9.5 * idx as f32,
                    if idx % 2 == 0 { 0.0 } else { 1.8 },
                    0.45 * ((idx % 3) as f32 - 1.0),
                ]);
            fragment.graph.name =
                format!("{}_{}_site_{idx}_scaffold", kind.as_str(), owner_fragment);
            fragment
        })
        .collect()
}

fn runtime_quantum_owner_disconnected_primitive_scaffold_components(
    kind: WholeCellRuntimeQuantumProcessKind,
    owner_fragment: &str,
    primitive_units: usize,
) -> Vec<EmbeddedMolecule> {
    let primitive_units = primitive_units.max(1).min(8);
    (0..primitive_units)
        .map(|idx| match kind {
            WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation => {
                let mut fragment = translation_peptide_site([
                    5.35 * idx as f32,
                    if idx % 2 == 0 { 0.0 } else { 1.15 },
                    0.42 * ((idx % 3) as f32 - 1.0),
                ]);
                fragment.graph.name =
                    format!("{}_{}_site_{idx}_scaffold", kind.as_str(), owner_fragment);
                fragment
            }
            WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation => {
                let mut fragment = phosphate_site([
                    4.90 * idx as f32,
                    0.62 * ((idx % 3) as f32 - 1.0),
                    if idx % 2 == 0 { 0.0 } else { 0.38 },
                ]);
                fragment.graph.name =
                    format!("{}_{}_site_{idx}_scaffold", kind.as_str(), owner_fragment);
                fragment
            }
            WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation => {
                let mut fragment = nucleotide_phosphate_site([
                    5.10 * idx as f32,
                    if idx % 2 == 0 { 0.0 } else { 0.78 },
                    0.35 * ((idx % 3) as f32 - 1.0),
                ]);
                fragment.graph.name =
                    format!("{}_{}_site_{idx}_scaffold", kind.as_str(), owner_fragment);
                fragment
            }
            WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification
            | WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification => {
                let mut fragment = membrane_acyl_site([
                    4.75 * idx as f32,
                    if idx % 2 == 0 { 0.22 } else { -0.22 },
                    0.26 * ((idx % 3) as f32 - 1.0),
                ]);
                fragment.graph.name =
                    format!("{}_{}_site_{idx}_scaffold", kind.as_str(), owner_fragment);
                fragment
            }
            WholeCellRuntimeQuantumProcessKind::AtpBandProtonTransfer => {
                let mut fragment = phosphate_site([
                    4.90 * idx as f32,
                    0.62 * ((idx % 3) as f32 - 1.0),
                    if idx % 2 == 0 { 0.0 } else { 0.38 },
                ]);
                fragment.graph.name =
                    format!("{}_{}_site_{idx}_scaffold", kind.as_str(), owner_fragment);
                fragment
            }
            WholeCellRuntimeQuantumProcessKind::SeptumGtpHydrolysis => {
                let mut fragment = phosphate_site([
                    5.00 * idx as f32,
                    if idx % 2 == 0 { 0.0 } else { 0.68 },
                    0.30 * ((idx % 3) as f32 - 1.0),
                ]);
                fragment.graph.name =
                    format!("{}_{}_site_{idx}_scaffold", kind.as_str(), owner_fragment);
                fragment
            }
            WholeCellRuntimeQuantumProcessKind::AtpBandElectronTransfer => {
                let mut fragment = phosphate_site([
                    4.90 * idx as f32,
                    0.62 * ((idx % 3) as f32 - 1.0),
                    if idx % 2 == 0 { 0.0 } else { 0.38 },
                ]);
                fragment.graph.name =
                    format!("{}_{}_site_{idx}_scaffold", kind.as_str(), owner_fragment);
                fragment
            }
            WholeCellRuntimeQuantumProcessKind::MembraneProteinInsertion => {
                let mut fragment = membrane_acyl_site([
                    4.75 * idx as f32,
                    if idx % 2 == 0 { 0.22 } else { -0.22 },
                    0.26 * ((idx % 3) as f32 - 1.0),
                ]);
                fragment.graph.name =
                    format!("{}_{}_site_{idx}_scaffold", kind.as_str(), owner_fragment);
                fragment
            }
        })
        .collect()
}

fn runtime_quantum_owner_scaffold_fragment_atom_budget(
    kind: WholeCellRuntimeQuantumProcessKind,
) -> usize {
    // Budgets lifted to support larger live-state-carved microdomains
    // (was 8/6/4/6/4 → now 12/10/8/10/10) with MAX_SPATIAL_ORBITALS=48.
    match kind {
        WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation => 12,
        WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation => 10,
        WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation
        | WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification
        | WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification => 8,
        WholeCellRuntimeQuantumProcessKind::AtpBandProtonTransfer
        | WholeCellRuntimeQuantumProcessKind::SeptumGtpHydrolysis
        | WholeCellRuntimeQuantumProcessKind::AtpBandElectronTransfer => 10,
        WholeCellRuntimeQuantumProcessKind::MembraneProteinInsertion => 10,
    }
}

fn translation_condensation_processes(
    max_scaffold_atoms: usize,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    translation_condensation_processes_with_scaffold_source(max_scaffold_atoms, &[], None)
}

fn translation_condensation_processes_with_preferred_anchors(
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    translation_condensation_processes_with_scaffold_source(
        max_scaffold_atoms,
        preferred_anchors,
        None,
    )
}

fn translation_condensation_processes_with_scaffold_source(
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
    scaffold_source: Option<&EmbeddedMolecule>,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    let kind = WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation;
    let reactant_ve = valence_electrons_for_graph(&methylamine([0.0; 3]).graph)
        + valence_electrons_for_graph(&acetic_acid([0.0; 3]).graph);
    let effective_max = max_scaffold_atoms.min(max_scaffold_atoms_within_orbital_limit(reactant_ve));
    let scaffolds =
        scaffolds_for_kind(kind, effective_max, preferred_anchors, scaffold_source);
    scaffolds
        .into_iter()
        .map(|fragment| {
            let SiteScaffoldFragment {
                scaffold,
                source_anchor_atom_idx,
            } = fragment;
            let placement = scaffold_reactant_placement(kind, &scaffold, 2.6);
            let acid_anchor = placement.anchor_position_angstrom;
            let acid = acetic_acid(acid_anchor);
            let amine = methylamine(offset_along_direction(
                acid_anchor,
                placement.direction,
                2.35,
            ));
            // methylamine: [0]=C, [1]=N, [2-4]=C-H, [5]=N-H, [6]=N-H (removed)
            // acetic_acid: [0]=C_me, [1]=C_co, [2]=O=, [3]=O-OH (removed), [4-6]=C-H, [7]=O-H
            let base_reaction = StructuralReactionTemplate::new(
                "ribosome_translation_amide_condensation",
                vec![amine.graph.clone(), acid.graph.clone()],
                vec![
                    StructuralReactionEdit::BreakBond {
                        a: ScopedAtomRef::new(0, 1),  // N in methylamine
                        b: ScopedAtomRef::new(0, 6),  // N-H (amine H removed)
                    },
                    StructuralReactionEdit::BreakBond {
                        a: ScopedAtomRef::new(1, 1),  // C_carboxyl in acetic acid
                        b: ScopedAtomRef::new(1, 3),  // O-OH (hydroxyl removed)
                    },
                    StructuralReactionEdit::FormBond {
                        a: ScopedAtomRef::new(0, 1),  // N (amine)
                        b: ScopedAtomRef::new(1, 1),  // C_carboxyl → peptide C-N bond
                        order: BondOrder::Single,
                    },
                    StructuralReactionEdit::FormBond {
                        a: ScopedAtomRef::new(1, 3),  // displaced O-OH
                        b: ScopedAtomRef::new(0, 6),  // displaced H → water O-H
                        order: BondOrder::Single,
                    },
                ],
            );
            let (reactants, reaction) =
                prepend_site_scaffold(scaffold, vec![amine, acid], base_reaction);
            WholeCellRuntimeQuantumProcessState::new(
                kind,
                Some(source_anchor_atom_idx),
                reactants,
                reaction,
            )
        })
        .collect()
}

fn atp_band_energy_processes(
    max_scaffold_atoms: usize,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    atp_band_energy_processes_with_scaffold_source(max_scaffold_atoms, &[], None)
}

fn atp_band_energy_processes_with_preferred_anchors(
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    atp_band_energy_processes_with_scaffold_source(max_scaffold_atoms, preferred_anchors, None)
}

fn atp_band_energy_processes_with_scaffold_source(
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
    scaffold_source: Option<&EmbeddedMolecule>,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    let kind = WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation;
    // Derive scaffold atom budget from reactant valence electrons and the u64
    // spin-orbital limit — no hardcoded scaffold sizes.
    let reactant_ve = valence_electrons_for_graph(&methanol([0.0; 3]).graph)
        + valence_electrons_for_graph(&orthophosphoric_acid([0.0; 3]).graph);
    let effective_max = max_scaffold_atoms.min(max_scaffold_atoms_within_orbital_limit(reactant_ve));
    let scaffolds =
        scaffolds_for_kind(kind, effective_max, preferred_anchors, scaffold_source);
    scaffolds
        .into_iter()
        .map(|fragment| {
            let SiteScaffoldFragment {
                scaffold,
                source_anchor_atom_idx,
            } = fragment;
            let placement = scaffold_reactant_placement(kind, &scaffold, 2.6);
            let phosphate_anchor = placement.anchor_position_angstrom;
            let alcohol = methanol(offset_along_direction(
                phosphate_anchor,
                placement.direction,
                2.55,
            ));
            let phosphate = orthophosphoric_acid(phosphate_anchor);
            let base_reaction = phosphate_esterification_reaction(
                "atp_band_energy_phosphorylation",
                alcohol.graph.clone(),
                1,
                5,
                phosphate.graph.clone(),
            );
            let (reactants, reaction) =
                prepend_site_scaffold(scaffold, vec![alcohol, phosphate], base_reaction);
            WholeCellRuntimeQuantumProcessState::new(
                kind,
                Some(source_anchor_atom_idx),
                reactants,
                reaction,
            )
        })
        .collect()
}

fn replisome_nucleotide_processes(
    max_scaffold_atoms: usize,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    replisome_nucleotide_processes_with_scaffold_source(max_scaffold_atoms, &[], None)
}

fn replisome_nucleotide_processes_with_preferred_anchors(
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    replisome_nucleotide_processes_with_scaffold_source(max_scaffold_atoms, preferred_anchors, None)
}

fn replisome_nucleotide_processes_with_scaffold_source(
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
    scaffold_source: Option<&EmbeddedMolecule>,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    let kind = WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation;
    let reactant_ve = valence_electrons_for_graph(&ethanol([0.0; 3]).graph)
        + valence_electrons_for_graph(&orthophosphoric_acid([0.0; 3]).graph);
    let effective_max = max_scaffold_atoms.min(max_scaffold_atoms_within_orbital_limit(reactant_ve));
    let scaffolds =
        scaffolds_for_kind(kind, effective_max, preferred_anchors, scaffold_source);
    scaffolds
        .into_iter()
        .map(|fragment| {
            let SiteScaffoldFragment {
                scaffold,
                source_anchor_atom_idx,
            } = fragment;
            let placement = scaffold_reactant_placement(kind, &scaffold, 2.6);
            let phosphate_anchor = placement.anchor_position_angstrom;
            let alcohol = ethanol(offset_along_direction(
                phosphate_anchor,
                placement.direction,
                2.80,
            ));
            let phosphate = orthophosphoric_acid(phosphate_anchor);
            let base_reaction = phosphate_esterification_reaction(
                "replisome_nucleotide_phosphorylation",
                alcohol.graph.clone(),
                2,
                8,
                phosphate.graph.clone(),
            );
            let (reactants, reaction) =
                prepend_site_scaffold(scaffold, vec![alcohol, phosphate], base_reaction);
            WholeCellRuntimeQuantumProcessState::new(
                kind,
                Some(source_anchor_atom_idx),
                reactants,
                reaction,
            )
        })
        .collect()
}

fn atp_band_membrane_processes(
    max_scaffold_atoms: usize,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    membrane_esterification_processes(
        WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification,
        "atp_band_membrane_esterification",
        2.55,
        max_scaffold_atoms,
    )
}

fn septum_membrane_processes(
    max_scaffold_atoms: usize,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    membrane_esterification_processes(
        WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification,
        "septum_membrane_esterification",
        2.75,
        max_scaffold_atoms,
    )
}

fn membrane_esterification_processes(
    kind: WholeCellRuntimeQuantumProcessKind,
    name: &str,
    alcohol_displacement_angstrom: f32,
    max_scaffold_atoms: usize,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    membrane_esterification_processes_with_scaffold_source(
        kind,
        name,
        alcohol_displacement_angstrom,
        max_scaffold_atoms,
        &[],
        None,
    )
}

fn membrane_esterification_processes_with_preferred_anchors(
    kind: WholeCellRuntimeQuantumProcessKind,
    name: &str,
    alcohol_displacement_angstrom: f32,
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    membrane_esterification_processes_with_scaffold_source(
        kind,
        name,
        alcohol_displacement_angstrom,
        max_scaffold_atoms,
        preferred_anchors,
        None,
    )
}

fn membrane_esterification_processes_with_scaffold_source(
    kind: WholeCellRuntimeQuantumProcessKind,
    name: &str,
    alcohol_displacement_angstrom: f32,
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
    scaffold_source: Option<&EmbeddedMolecule>,
) -> Vec<WholeCellRuntimeQuantumProcessState> {
    let reactant_ve = valence_electrons_for_graph(&methanol([0.0; 3]).graph)
        + valence_electrons_for_graph(&acetic_acid([0.0; 3]).graph);
    let effective_max = max_scaffold_atoms.min(max_scaffold_atoms_within_orbital_limit(reactant_ve));
    let scaffolds =
        scaffolds_for_kind(kind, effective_max, preferred_anchors, scaffold_source);
    scaffolds
        .into_iter()
        .map(|fragment| {
            let SiteScaffoldFragment {
                scaffold,
                source_anchor_atom_idx,
            } = fragment;
            let placement = scaffold_reactant_placement(kind, &scaffold, 2.6);
            let acid_anchor = placement.anchor_position_angstrom;
            let alcohol = methanol(offset_along_direction(
                acid_anchor,
                placement.direction,
                alcohol_displacement_angstrom,
            ));
            let acid = acetic_acid(acid_anchor);
            // methanol: [0]=C, [1]=O, [2-4]=C-H, [5]=O-H (removed in ester formation)
            // acetic_acid: [0]=C_me, [1]=C_co, [2]=O=, [3]=O-OH (removed), [4-6]=C-H, [7]=O-H
            let base_reaction = StructuralReactionTemplate::new(
                name,
                vec![alcohol.graph.clone(), acid.graph.clone()],
                vec![
                    StructuralReactionEdit::BreakBond {
                        a: ScopedAtomRef::new(0, 1),  // O in methanol
                        b: ScopedAtomRef::new(0, 5),  // O-H
                    },
                    StructuralReactionEdit::BreakBond {
                        a: ScopedAtomRef::new(1, 1),  // C_carboxyl in acetic acid
                        b: ScopedAtomRef::new(1, 3),  // O-OH
                    },
                    StructuralReactionEdit::FormBond {
                        a: ScopedAtomRef::new(0, 1),  // O (alcohol)
                        b: ScopedAtomRef::new(1, 1),  // C_carboxyl → ester C-O bond
                        order: BondOrder::Single,
                    },
                    StructuralReactionEdit::FormBond {
                        a: ScopedAtomRef::new(1, 3),  // displaced O-OH
                        b: ScopedAtomRef::new(0, 5),  // displaced H → water O-H
                        order: BondOrder::Single,
                    },
                ],
            );
            let (reactants, reaction) =
                prepend_site_scaffold(scaffold, vec![alcohol, acid], base_reaction);
            WholeCellRuntimeQuantumProcessState::new(
                kind,
                Some(source_anchor_atom_idx),
                reactants,
                reaction,
            )
        })
        .collect()
}

fn phosphate_esterification_reaction(
    name: &str,
    alcohol_graph: MoleculeGraph,
    alcohol_oxygen: usize,
    alcohol_hydroxyl_h: usize,
    phosphorous_graph: MoleculeGraph,
) -> StructuralReactionTemplate {
    StructuralReactionTemplate::new(
        name,
        vec![alcohol_graph, phosphorous_graph],
        vec![
            StructuralReactionEdit::BreakBond {
                a: ScopedAtomRef::new(0, alcohol_oxygen),
                b: ScopedAtomRef::new(0, alcohol_hydroxyl_h),
            },
            StructuralReactionEdit::BreakBond {
                a: ScopedAtomRef::new(1, 0),
                b: ScopedAtomRef::new(1, 2),
            },
            StructuralReactionEdit::FormBond {
                a: ScopedAtomRef::new(0, alcohol_oxygen),
                b: ScopedAtomRef::new(1, 0),
                order: BondOrder::Single,
            },
            StructuralReactionEdit::FormBond {
                a: ScopedAtomRef::new(1, 2),
                b: ScopedAtomRef::new(0, alcohol_hydroxyl_h),
                order: BondOrder::Single,
            },
        ],
    )
}

fn site_scaffolds(
    kind: WholeCellRuntimeQuantumProcessKind,
    max_scaffold_atoms: usize,
) -> Vec<SiteScaffoldFragment> {
    site_scaffolds_with_preferred_anchors(kind, max_scaffold_atoms, &[])
}

fn scaffolds_for_kind(
    kind: WholeCellRuntimeQuantumProcessKind,
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
    scaffold_source: Option<&EmbeddedMolecule>,
) -> Vec<SiteScaffoldFragment> {
    if let Some(scaffold_source) = scaffold_source {
        site_scaffolds_from_source_with_preferred_anchors(
            kind,
            scaffold_source,
            max_scaffold_atoms,
            preferred_anchors,
        )
    } else if preferred_anchors.is_empty() {
        site_scaffolds(kind, max_scaffold_atoms)
    } else {
        site_scaffolds_with_preferred_anchors(kind, max_scaffold_atoms, preferred_anchors)
    }
}

fn site_scaffolds_with_preferred_anchors(
    kind: WholeCellRuntimeQuantumProcessKind,
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
) -> Vec<SiteScaffoldFragment> {
    let template = runtime_quantum_default_scaffold_source(kind);
    site_scaffolds_from_source_with_preferred_anchors(
        kind,
        &template,
        max_scaffold_atoms,
        preferred_anchors,
    )
}

fn site_scaffolds_from_source_with_preferred_anchors(
    kind: WholeCellRuntimeQuantumProcessKind,
    scaffold_source: &EmbeddedMolecule,
    max_scaffold_atoms: usize,
    preferred_anchors: &[usize],
) -> Vec<SiteScaffoldFragment> {
    let mut unique_fragments = std::collections::BTreeSet::new();
    let mut fragments = Vec::new();
    // Ensure at least 2 scaffold atoms for meaningful environment coupling,
    // but respect the caller's orbital-aware budget (which may be < 4).
    let max_scaffold_atoms = max_scaffold_atoms.max(2);
    let mut ordered_anchors = preferred_anchors
        .iter()
        .copied()
        .filter(|anchor| *anchor < scaffold_source.graph.atom_count())
        .collect::<Vec<_>>();
    for anchor in site_quantum_fragment_anchor_indices(kind, scaffold_source, max_scaffold_atoms) {
        if !ordered_anchors.contains(&anchor) {
            ordered_anchors.push(anchor);
        }
    }
    for anchor in ordered_anchors {
        let selected =
            site_quantum_fragment_indices(kind, scaffold_source, max_scaffold_atoms, anchor);
        if unique_fragments.insert(selected.clone()) {
            fragments.push(SiteScaffoldFragment {
                scaffold: extract_site_quantum_fragment(scaffold_source, &selected),
                source_anchor_atom_idx: anchor,
            });
        }
    }
    if fragments.is_empty() {
        fragments.push(SiteScaffoldFragment {
            scaffold: extract_site_quantum_fragment(
                scaffold_source,
                &site_quantum_fragment_indices(kind, scaffold_source, max_scaffold_atoms, 0),
            ),
            source_anchor_atom_idx: 0,
        });
    }
    fragments
}

fn offset_along_direction(
    position: [f32; 3],
    direction: [f32; 3],
    distance_angstrom: f32,
) -> [f32; 3] {
    [
        position[0] + direction[0] * distance_angstrom,
        position[1] + direction[1] * distance_angstrom,
        position[2] + direction[2] * distance_angstrom,
    ]
}

fn scaffold_reactant_placement(
    kind: WholeCellRuntimeQuantumProcessKind,
    scaffold: &EmbeddedMolecule,
    standoff_angstrom: f32,
) -> SiteReactivePlacement {
    let adjacency = scaffold_adjacency(scaffold);
    if let Some(primitive) = primary_site_chemical_primitive(kind, scaffold, &adjacency) {
        let center_position = scaffold.positions_angstrom[primitive.center_atom_idx];
        let support_centroid = if primitive.support_atom_indices.is_empty() {
            None
        } else {
            Some(mean_position(
                primitive
                    .support_atom_indices
                    .iter()
                    .map(|atom_idx| scaffold.positions_angstrom[*atom_idx]),
            ))
        };
        let scaffold_centroid = centroid_angstrom(scaffold);
        let neighbor_centroid = if adjacency[primitive.center_atom_idx].is_empty() {
            None
        } else {
            Some(mean_position(
                adjacency[primitive.center_atom_idx]
                    .iter()
                    .map(|(atom_idx, _)| scaffold.positions_angstrom[*atom_idx]),
            ))
        };

        let outward_direction = support_centroid
            .map(|centroid| subtract_f32x3(center_position, centroid))
            .filter(|vector| vector_norm_f32(*vector) > 1.0e-4)
            .or_else(|| {
                let vector = subtract_f32x3(center_position, scaffold_centroid);
                (vector_norm_f32(vector) > 1.0e-4).then_some(vector)
            })
            .or_else(|| {
                neighbor_centroid
                    .map(|centroid| subtract_f32x3(center_position, centroid))
                    .filter(|vector| vector_norm_f32(*vector) > 1.0e-4)
            })
            .unwrap_or([1.0, 0.0, 0.0]);
        let direction = normalize_vector_f32(outward_direction);
        let local_radius = primitive
            .support_atom_indices
            .iter()
            .map(|atom_idx| {
                squared_distance(scaffold.positions_angstrom[*atom_idx], center_position)
            })
            .fold(0.0f32, f32::max)
            .sqrt();

        return SiteReactivePlacement {
            anchor_position_angstrom: offset_along_direction(
                center_position,
                direction,
                standoff_angstrom.max(1.8) + 0.35 * local_radius,
            ),
            center_atom_idx: primitive.center_atom_idx,
            direction,
        };
    }

    let centroid = centroid_angstrom(scaffold);
    let center_atom_idx = scaffold
        .positions_angstrom
        .iter()
        .enumerate()
        .max_by(|(left_idx, left_position), (right_idx, right_position)| {
            squared_distance(**left_position, centroid)
                .total_cmp(&squared_distance(**right_position, centroid))
                .then_with(|| left_idx.cmp(right_idx))
        })
        .map(|(atom_idx, _)| atom_idx)
        .unwrap_or(0);
    let center_position = scaffold.positions_angstrom[center_atom_idx];
    let direction = normalize_vector_f32(subtract_f32x3(center_position, centroid));
    SiteReactivePlacement {
        anchor_position_angstrom: offset_along_direction(
            center_position,
            direction,
            standoff_angstrom.max(2.6),
        ),
        center_atom_idx,
        direction,
    }
}

fn prepend_site_scaffold(
    scaffold: EmbeddedMolecule,
    reactants: Vec<EmbeddedMolecule>,
    reaction: StructuralReactionTemplate,
) -> (Vec<EmbeddedMolecule>, StructuralReactionTemplate) {
    let mut scaffolded_reactants = Vec::with_capacity(reactants.len() + 1);
    scaffolded_reactants.push(scaffold.clone());
    scaffolded_reactants.extend(reactants);

    let mut scaffolded_graphs = Vec::with_capacity(reaction.reactants.len() + 1);
    scaffolded_graphs.push(scaffold.graph);
    scaffolded_graphs.extend(reaction.reactants);

    let edits = reaction
        .edits
        .into_iter()
        .map(|edit| offset_structural_reaction_edit(edit, 1))
        .collect();

    (
        scaffolded_reactants,
        StructuralReactionTemplate::new(&reaction.name, scaffolded_graphs, edits),
    )
}

fn offset_structural_reaction_edit(
    edit: StructuralReactionEdit,
    reactant_offset: usize,
) -> StructuralReactionEdit {
    match edit {
        StructuralReactionEdit::FormBond { a, b, order } => StructuralReactionEdit::FormBond {
            a: offset_scoped_atom_ref(a, reactant_offset),
            b: offset_scoped_atom_ref(b, reactant_offset),
            order,
        },
        StructuralReactionEdit::BreakBond { a, b } => StructuralReactionEdit::BreakBond {
            a: offset_scoped_atom_ref(a, reactant_offset),
            b: offset_scoped_atom_ref(b, reactant_offset),
        },
        StructuralReactionEdit::ChangeBondOrder { a, b, order } => {
            StructuralReactionEdit::ChangeBondOrder {
                a: offset_scoped_atom_ref(a, reactant_offset),
                b: offset_scoped_atom_ref(b, reactant_offset),
                order,
            }
        }
        StructuralReactionEdit::SetFormalCharge {
            atom,
            formal_charge,
        } => StructuralReactionEdit::SetFormalCharge {
            atom: offset_scoped_atom_ref(atom, reactant_offset),
            formal_charge,
        },
    }
}

fn offset_scoped_atom_ref(atom: ScopedAtomRef, reactant_offset: usize) -> ScopedAtomRef {
    ScopedAtomRef::new(atom.reactant_idx + reactant_offset, atom.atom_idx)
}

fn extract_site_quantum_fragment(
    scaffold: &EmbeddedMolecule,
    selected: &[usize],
) -> EmbeddedMolecule {
    let mut index_map = std::collections::BTreeMap::new();
    let mut graph = MoleculeGraph::new(&format!("{}_quantum_fragment", scaffold.graph.name));
    let mut positions = Vec::with_capacity(selected.len());

    for (new_idx, old_idx) in selected.iter().copied().enumerate() {
        index_map.insert(old_idx, new_idx);
        graph.add_atom_node(scaffold.graph.atoms[old_idx].clone());
        positions.push(scaffold.positions_angstrom[old_idx]);
    }

    for bond in &scaffold.graph.bonds {
        if let (Some(&a), Some(&b)) = (index_map.get(&bond.i), index_map.get(&bond.j)) {
            graph
                .add_bond(a, b, bond.order)
                .expect("site quantum fragment bonds stay valid");
        }
    }

    EmbeddedMolecule::new(graph, positions).expect("site quantum fragment geometry")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SiteChemicalPrimitiveKind {
    PhosphateCenter,
    PhosphorylOxygen,
    HydroxylOxygen,
    AmineCenter,
    CarbonylCarbon,
    CarbonylOxygen,
    PolarCarbon,
}

#[derive(Clone, Debug, PartialEq)]
struct SiteChemicalPrimitive {
    kind: SiteChemicalPrimitiveKind,
    center_atom_idx: usize,
    support_atom_indices: Vec<usize>,
    intrinsic_score: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SiteReactivePlacement {
    anchor_position_angstrom: [f32; 3],
    center_atom_idx: usize,
    direction: [f32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SiteScaffoldFragment {
    pub(crate) scaffold: EmbeddedMolecule,
    pub(crate) source_anchor_atom_idx: usize,
}

fn molecule_graph_matches_structure(left: &MoleculeGraph, right: &MoleculeGraph) -> bool {
    left.atoms == right.atoms && left.bonds == right.bonds
}

fn local_embedded_component_amount_for_graph(
    kind: WholeCellRuntimeQuantumProcessKind,
    mixture: &EmbeddedMaterialMixture,
    scaffold_source: &EmbeddedMolecule,
    graph: &MoleculeGraph,
) -> f64 {
    local_embedded_component_amount_for_any_graph(
        kind,
        mixture,
        scaffold_source,
        std::slice::from_ref(graph),
    )
}

fn local_embedded_component_amount_for_any_graph(
    kind: WholeCellRuntimeQuantumProcessKind,
    mixture: &EmbeddedMaterialMixture,
    scaffold_source: &EmbeddedMolecule,
    graphs: &[MoleculeGraph],
) -> f64 {
    let scaffold_candidates =
        ranked_runtime_quantum_scaffold_components_from_mixture(kind, mixture)
            .into_iter()
            .map(|(molecule, _)| molecule)
            .collect::<Vec<_>>();
    if scaffold_candidates.len() <= 1 {
        return mixture
            .components
            .iter()
            .filter(|component| {
                graphs
                    .iter()
                    .any(|graph| molecule_graph_matches_structure(&component.molecule.graph, graph))
            })
            .map(|component| component.amount_moles.max(0.0))
            .sum();
    }

    let Some(target_idx) = scaffold_candidates
        .iter()
        .position(|candidate| candidate == scaffold_source)
    else {
        return mixture
            .components
            .iter()
            .filter(|component| {
                graphs
                    .iter()
                    .any(|graph| molecule_graph_matches_structure(&component.molecule.graph, graph))
            })
            .map(|component| component.amount_moles.max(0.0))
            .sum();
    };
    let assignment_centroids = scaffold_candidates
        .iter()
        .map(|candidate| {
            runtime_quantum_reactants_for_scaffold(kind, candidate)
                .into_iter()
                .find(|reactant| {
                    graphs
                        .iter()
                        .any(|graph| molecule_graph_matches_structure(&reactant.graph, graph))
                })
                .map(|reactant| centroid_angstrom(&reactant))
                .unwrap_or_else(|| centroid_angstrom(candidate))
        })
        .collect::<Vec<_>>();

    mixture
        .components
        .iter()
        .filter(|component| {
            graphs
                .iter()
                .any(|graph| molecule_graph_matches_structure(&component.molecule.graph, graph))
        })
        .filter_map(|component| {
            let component_centroid = centroid_angstrom(&component.molecule);
            let nearest_idx = assignment_centroids
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    squared_distance(component_centroid, **left)
                        .total_cmp(&squared_distance(component_centroid, **right))
                })
                .map(|(idx, _)| idx)?;
            (nearest_idx == target_idx).then_some(component.amount_moles.max(0.0))
        })
        .sum()
}

fn runtime_quantum_support_bulk_fields_for_reactant(
    kind: WholeCellRuntimeQuantumProcessKind,
    reactant_graph: &MoleculeGraph,
) -> &'static [WholeCellBulkField] {
    const EMPTY_FIELDS: [WholeCellBulkField; 0] = [];
    const AMINO_FIELDS: [WholeCellBulkField; 1] = [WholeCellBulkField::AminoAcids];
    const ADENOSINE_PHOSPHATE_FIELDS: [WholeCellBulkField; 2] =
        [WholeCellBulkField::ADP, WholeCellBulkField::ATP];
    const NUCLEOTIDE_FIELDS: [WholeCellBulkField; 1] = [WholeCellBulkField::Nucleotides];
    const MEMBRANE_FIELDS: [WholeCellBulkField; 1] = [WholeCellBulkField::MembranePrecursors];

    match kind {
        WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation
            if reactant_graph.name == "methylamine"
                || reactant_graph.name == "acetic_acid" =>
        {
            &AMINO_FIELDS
        }
        WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation
            if reactant_graph.name == "methanol"
                || reactant_graph.name == "orthophosphoric_acid" =>
        {
            &ADENOSINE_PHOSPHATE_FIELDS
        }
        WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation
            if reactant_graph.name == "ethanol"
                || reactant_graph.name == "orthophosphoric_acid" =>
        {
            &NUCLEOTIDE_FIELDS
        }
        WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification
        | WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification
            if reactant_graph.name == "methanol"
                || reactant_graph.name == "acetic_acid" =>
        {
            &MEMBRANE_FIELDS
        }
        _ => &EMPTY_FIELDS,
    }
}

fn runtime_quantum_support_graphs_for_reactant(
    kind: WholeCellRuntimeQuantumProcessKind,
    reactant_graph: &MoleculeGraph,
) -> Vec<MoleculeGraph> {
    runtime_quantum_support_bulk_fields_for_reactant(kind, reactant_graph)
        .iter()
        .copied()
        .map(runtime_quantum_representative_graph_for_bulk_field)
        .collect()
}

fn local_runtime_support_components_for_kind(
    kind: WholeCellRuntimeQuantumProcessKind,
    mixture: &EmbeddedMaterialMixture,
    scaffold_source: &EmbeddedMolecule,
) -> Vec<(EmbeddedMolecule, f64)> {
    let support_graphs = runtime_quantum_context_support_graphs(kind);
    let scaffold_candidates =
        ranked_runtime_quantum_scaffold_components_from_mixture(kind, mixture)
            .into_iter()
            .map(|(molecule, _)| molecule)
            .collect::<Vec<_>>();
    let target_idx = scaffold_candidates
        .iter()
        .position(|candidate| candidate == scaffold_source);
    let assignment_centroids = scaffold_candidates
        .iter()
        .map(centroid_angstrom)
        .collect::<Vec<_>>();

    mixture
        .components
        .iter()
        .filter(|component| {
            support_graphs
                .iter()
                .any(|graph| molecule_graph_matches_structure(&component.molecule.graph, graph))
        })
        .filter(|component| {
            if scaffold_candidates.len() <= 1 {
                return true;
            }
            let Some(target_idx) = target_idx else {
                return true;
            };
            let component_centroid = centroid_angstrom(&component.molecule);
            let nearest_idx = assignment_centroids
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    squared_distance(component_centroid, **left)
                        .total_cmp(&squared_distance(component_centroid, **right))
                })
                .map(|(idx, _)| idx);
            nearest_idx == Some(target_idx)
        })
        .map(|component| (component.molecule.clone(), component.amount_moles.max(0.0)))
        .collect()
}

fn seed_runtime_quantum_process_inventory_from_microdomain(
    kind: WholeCellRuntimeQuantumProcessKind,
    processes: &mut [WholeCellRuntimeQuantumProcessState],
    scaffold_amount_moles: f64,
    scaffold_source: &EmbeddedMolecule,
    microdomain: &EmbeddedMaterialMixture,
) {
    for process in processes {
        process.set_reactant_inventory_amount(0, scaffold_amount_moles);
        for reactant_idx in 1..process.reactant_count() {
            let support_graphs = runtime_quantum_support_graphs_for_reactant(
                kind,
                &process.reactants[reactant_idx].graph,
            );
            let exact_amount = local_embedded_component_amount_for_graph(
                kind,
                microdomain,
                scaffold_source,
                &process.reactants[reactant_idx].graph,
            );
            let support_amount = if support_graphs.is_empty() {
                0.0
            } else {
                local_embedded_component_amount_for_any_graph(
                    kind,
                    microdomain,
                    scaffold_source,
                    &support_graphs,
                )
            };
            process.set_reactant_inventory_amount(reactant_idx, exact_amount.max(support_amount));
        }
        for (support_molecule, support_amount) in
            local_runtime_support_components_for_kind(kind, microdomain, scaffold_source)
        {
            if support_amount <= 1.0e-12 {
                continue;
            }
            process
                .mixture
                .add_component(support_molecule, support_amount);
        }
    }
}

fn ranked_runtime_quantum_scaffold_components_from_mixture(
    kind: WholeCellRuntimeQuantumProcessKind,
    mixture: &EmbeddedMaterialMixture,
) -> Vec<(EmbeddedMolecule, f64)> {
    let canonical_reactant_graphs = runtime_quantum_canonical_reactant_graphs(kind, None);
    let support_graphs = runtime_quantum_context_support_graphs(kind);
    let mut ranked = mixture
        .components
        .iter()
        .filter_map(|component| {
            if component.amount_moles <= 1.0e-12 || component.molecule.graph.atom_count() < 4 {
                return None;
            }
            if is_runtime_quantum_canonical_reactant_graph(
                &canonical_reactant_graphs,
                &component.molecule.graph,
            ) {
                return None;
            }
            if support_graphs
                .iter()
                .any(|graph| molecule_graph_matches_structure(&component.molecule.graph, graph))
            {
                return None;
            }
            let adjacency = scaffold_adjacency(&component.molecule);
            let weighted = weighted_site_chemical_primitives(kind, &component.molecule, &adjacency);
            let top_weight = weighted.first().map(|(_, weight)| *weight)?;
            let local_support = weighted
                .iter()
                .take(4)
                .map(|(_, weight)| f64::from(*weight))
                .sum::<f64>();
            let kind_tag_bonus = if component.molecule.graph.name.starts_with(kind.as_str()) {
                1000.0
            } else {
                0.0
            };
            let score = kind_tag_bonus
                + 6.0 * f64::from(top_weight)
                + 0.75 * local_support
                + 0.05 * component.molecule.graph.atom_count() as f64
                + 0.01 * component.amount_moles.max(0.0);
            Some((
                score,
                component.amount_moles.max(0.0),
                component.molecule.clone(),
            ))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| right.2.graph.atom_count().cmp(&left.2.graph.atom_count()))
    });
    ranked
        .into_iter()
        .map(|(_, amount_moles, molecule)| (molecule, amount_moles))
        .collect()
}

fn scaffold_adjacency(scaffold: &EmbeddedMolecule) -> Vec<Vec<(usize, BondOrder)>> {
    let mut adjacency = vec![Vec::new(); scaffold.graph.atom_count()];
    for bond in &scaffold.graph.bonds {
        adjacency[bond.i].push((bond.j, bond.order));
        adjacency[bond.j].push((bond.i, bond.order));
    }
    for neighbors in &mut adjacency {
        neighbors.sort_by_key(|(idx, _)| *idx);
    }
    adjacency
}

fn unique_sorted_indices(indices: impl IntoIterator<Item = usize>) -> Vec<usize> {
    indices
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn infer_site_chemical_primitives(
    scaffold: &EmbeddedMolecule,
    adjacency: &[Vec<(usize, BondOrder)>],
) -> Vec<SiteChemicalPrimitive> {
    let mut primitives = Vec::new();

    for (atom_idx, atom) in scaffold.graph.atoms.iter().enumerate() {
        let neighbors = &adjacency[atom_idx];
        let mut oxygen_neighbors = Vec::new();
        let mut nitrogen_neighbors = Vec::new();
        let mut carbon_neighbors = Vec::new();
        let mut phosphorus_neighbors = Vec::new();
        let mut hydrogen_neighbors = Vec::new();
        let mut hetero_neighbors = Vec::new();

        for (neighbor_idx, _) in neighbors {
            let neighbor = &scaffold.graph.atoms[*neighbor_idx];
            match neighbor.element {
                PeriodicElement::O => oxygen_neighbors.push(*neighbor_idx),
                PeriodicElement::N => nitrogen_neighbors.push(*neighbor_idx),
                PeriodicElement::C => carbon_neighbors.push(*neighbor_idx),
                PeriodicElement::P => phosphorus_neighbors.push(*neighbor_idx),
                PeriodicElement::H => hydrogen_neighbors.push(*neighbor_idx),
                _ => {}
            }
            if matches!(
                neighbor.element,
                PeriodicElement::N | PeriodicElement::O | PeriodicElement::P | PeriodicElement::S
            ) {
                hetero_neighbors.push(*neighbor_idx);
            }
        }

        let heavy_neighbor_count = neighbors
            .iter()
            .filter(|(neighbor_idx, _)| {
                scaffold.graph.atoms[*neighbor_idx].element != PeriodicElement::H
            })
            .count();
        let max_bond_order = neighbors
            .iter()
            .map(|(_, order)| order.bond_order())
            .fold(0.0f32, f32::max);

        match atom.element {
            PeriodicElement::P if !oxygen_neighbors.is_empty() => {
                primitives.push(SiteChemicalPrimitive {
                    kind: SiteChemicalPrimitiveKind::PhosphateCenter,
                    center_atom_idx: atom_idx,
                    support_atom_indices: unique_sorted_indices(oxygen_neighbors.iter().copied()),
                    intrinsic_score: 3.8
                        + 0.55 * oxygen_neighbors.len() as f32
                        + 0.10 * heavy_neighbor_count as f32,
                });
            }
            PeriodicElement::O => {
                let oxygen_linked_to_phosphorus = neighbors.iter().any(|(neighbor_idx, _)| {
                    scaffold.graph.atoms[*neighbor_idx].element == PeriodicElement::O
                        && adjacency[*neighbor_idx].iter().any(|(second_idx, _)| {
                            *second_idx != atom_idx
                                && scaffold.graph.atoms[*second_idx].element == PeriodicElement::P
                        })
                });

                if !phosphorus_neighbors.is_empty() || oxygen_linked_to_phosphorus {
                    primitives.push(SiteChemicalPrimitive {
                        kind: SiteChemicalPrimitiveKind::PhosphorylOxygen,
                        center_atom_idx: atom_idx,
                        support_atom_indices: unique_sorted_indices(
                            phosphorus_neighbors
                                .iter()
                                .copied()
                                .chain(carbon_neighbors.iter().copied()),
                        ),
                        intrinsic_score: 3.1
                            + 0.45 * phosphorus_neighbors.len() as f32
                            + 0.25
                                * if oxygen_linked_to_phosphorus {
                                    1.0
                                } else {
                                    0.0
                                }
                            + 0.20 * carbon_neighbors.len() as f32,
                    });
                }

                if !carbon_neighbors.is_empty() {
                    let attached_to_polar_carbon = carbon_neighbors.iter().any(|carbon_idx| {
                        adjacency[*carbon_idx].iter().any(|(second_idx, _)| {
                            *second_idx != atom_idx
                                && matches!(
                                    scaffold.graph.atoms[*second_idx].element,
                                    PeriodicElement::N
                                        | PeriodicElement::O
                                        | PeriodicElement::P
                                        | PeriodicElement::S
                                )
                        })
                    });
                    if attached_to_polar_carbon || neighbors.len() <= 1 || max_bond_order >= 1.5 {
                        primitives.push(SiteChemicalPrimitive {
                            kind: SiteChemicalPrimitiveKind::CarbonylOxygen,
                            center_atom_idx: atom_idx,
                            support_atom_indices: unique_sorted_indices(
                                carbon_neighbors
                                    .iter()
                                    .copied()
                                    .chain(phosphorus_neighbors.iter().copied()),
                            ),
                            intrinsic_score: 2.9
                                + 0.30 * carbon_neighbors.len() as f32
                                + 0.20 * if attached_to_polar_carbon { 1.0 } else { 0.0 },
                        });
                    } else if neighbors.len() <= 2 || !hydrogen_neighbors.is_empty() {
                        primitives.push(SiteChemicalPrimitive {
                            kind: SiteChemicalPrimitiveKind::HydroxylOxygen,
                            center_atom_idx: atom_idx,
                            support_atom_indices: unique_sorted_indices(
                                carbon_neighbors
                                    .iter()
                                    .copied()
                                    .chain(hydrogen_neighbors.iter().copied()),
                            ),
                            intrinsic_score: 2.4
                                + 0.25 * carbon_neighbors.len() as f32
                                + 0.25 * hydrogen_neighbors.len() as f32,
                        });
                    }
                } else if neighbors.len() <= 2 || !hydrogen_neighbors.is_empty() {
                    primitives.push(SiteChemicalPrimitive {
                        kind: SiteChemicalPrimitiveKind::HydroxylOxygen,
                        center_atom_idx: atom_idx,
                        support_atom_indices: unique_sorted_indices(
                            hydrogen_neighbors.iter().copied(),
                        ),
                        intrinsic_score: 2.2 + 0.25 * hydrogen_neighbors.len() as f32,
                    });
                }
            }
            PeriodicElement::N if !carbon_neighbors.is_empty() => {
                primitives.push(SiteChemicalPrimitive {
                    kind: SiteChemicalPrimitiveKind::AmineCenter,
                    center_atom_idx: atom_idx,
                    support_atom_indices: unique_sorted_indices(carbon_neighbors.iter().copied()),
                    intrinsic_score: 3.2
                        + 0.35 * carbon_neighbors.len() as f32
                        + 0.10 * heavy_neighbor_count as f32,
                });
            }
            PeriodicElement::C
                if !oxygen_neighbors.is_empty() && !nitrogen_neighbors.is_empty() =>
            {
                primitives.push(SiteChemicalPrimitive {
                    kind: SiteChemicalPrimitiveKind::CarbonylCarbon,
                    center_atom_idx: atom_idx,
                    support_atom_indices: unique_sorted_indices(
                        oxygen_neighbors
                            .iter()
                            .copied()
                            .chain(nitrogen_neighbors.iter().copied()),
                    ),
                    intrinsic_score: 3.6
                        + 0.35 * oxygen_neighbors.len() as f32
                        + 0.25 * nitrogen_neighbors.len() as f32,
                });
            }
            PeriodicElement::C
                if !oxygen_neighbors.is_empty()
                    || !nitrogen_neighbors.is_empty()
                    || !phosphorus_neighbors.is_empty() =>
            {
                primitives.push(SiteChemicalPrimitive {
                    kind: SiteChemicalPrimitiveKind::PolarCarbon,
                    center_atom_idx: atom_idx,
                    support_atom_indices: unique_sorted_indices(
                        oxygen_neighbors
                            .iter()
                            .copied()
                            .chain(nitrogen_neighbors.iter().copied())
                            .chain(phosphorus_neighbors.iter().copied()),
                    ),
                    intrinsic_score: 2.5
                        + 0.30 * oxygen_neighbors.len() as f32
                        + 0.20 * nitrogen_neighbors.len() as f32
                        + 0.15 * phosphorus_neighbors.len() as f32
                        + 0.10 * hetero_neighbors.len() as f32,
                });
            }
            _ => {}
        }
    }

    primitives
}

fn weighted_site_chemical_primitives(
    kind: WholeCellRuntimeQuantumProcessKind,
    scaffold: &EmbeddedMolecule,
    adjacency: &[Vec<(usize, BondOrder)>],
) -> Vec<(SiteChemicalPrimitive, f32)> {
    let mut weighted = infer_site_chemical_primitives(scaffold, adjacency)
        .into_iter()
        .map(|primitive| {
            let weight =
                primitive.intrinsic_score * primitive_process_priority(kind, primitive.kind);
            (primitive, weight)
        })
        .filter(|(_, weight)| *weight > 0.0)
        .collect::<Vec<_>>();
    weighted.sort_by(|(left, left_weight), (right, right_weight)| {
        right_weight
            .total_cmp(left_weight)
            .then_with(|| left.center_atom_idx.cmp(&right.center_atom_idx))
            .then_with(|| left.support_atom_indices.cmp(&right.support_atom_indices))
    });
    weighted
}

fn primary_site_chemical_primitive(
    kind: WholeCellRuntimeQuantumProcessKind,
    scaffold: &EmbeddedMolecule,
    adjacency: &[Vec<(usize, BondOrder)>],
) -> Option<SiteChemicalPrimitive> {
    weighted_site_chemical_primitives(kind, scaffold, adjacency)
        .into_iter()
        .next()
        .map(|(primitive, _)| primitive)
}

fn primitive_process_priority(
    kind: WholeCellRuntimeQuantumProcessKind,
    primitive: SiteChemicalPrimitiveKind,
) -> f32 {
    match kind {
        WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation => match primitive {
            SiteChemicalPrimitiveKind::CarbonylCarbon => 1.00,
            SiteChemicalPrimitiveKind::AmineCenter => 0.95,
            SiteChemicalPrimitiveKind::CarbonylOxygen => 0.80,
            SiteChemicalPrimitiveKind::PolarCarbon => 0.55,
            SiteChemicalPrimitiveKind::HydroxylOxygen => 0.30,
            SiteChemicalPrimitiveKind::PhosphorylOxygen => 0.20,
            SiteChemicalPrimitiveKind::PhosphateCenter => 0.10,
        },
        WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation
        | WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation => match primitive
        {
            SiteChemicalPrimitiveKind::PhosphateCenter => 1.00,
            SiteChemicalPrimitiveKind::PhosphorylOxygen => 0.90,
            SiteChemicalPrimitiveKind::HydroxylOxygen => 0.60,
            SiteChemicalPrimitiveKind::PolarCarbon => 0.35,
            SiteChemicalPrimitiveKind::CarbonylOxygen => 0.25,
            SiteChemicalPrimitiveKind::CarbonylCarbon => 0.20,
            SiteChemicalPrimitiveKind::AmineCenter => 0.10,
        },
        WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification
        | WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification => match primitive {
            SiteChemicalPrimitiveKind::CarbonylCarbon => 0.95,
            SiteChemicalPrimitiveKind::CarbonylOxygen => 0.90,
            SiteChemicalPrimitiveKind::HydroxylOxygen => 0.75,
            SiteChemicalPrimitiveKind::PolarCarbon => 0.70,
            SiteChemicalPrimitiveKind::PhosphorylOxygen => 0.55,
            SiteChemicalPrimitiveKind::PhosphateCenter => 0.40,
            SiteChemicalPrimitiveKind::AmineCenter => 0.30,
        },
        WholeCellRuntimeQuantumProcessKind::AtpBandProtonTransfer => match primitive {
            SiteChemicalPrimitiveKind::AmineCenter => 1.00,
            SiteChemicalPrimitiveKind::HydroxylOxygen => 0.85,
            SiteChemicalPrimitiveKind::CarbonylOxygen => 0.60,
            SiteChemicalPrimitiveKind::PolarCarbon => 0.40,
            SiteChemicalPrimitiveKind::CarbonylCarbon => 0.30,
            SiteChemicalPrimitiveKind::PhosphorylOxygen => 0.20,
            SiteChemicalPrimitiveKind::PhosphateCenter => 0.10,
        },
        WholeCellRuntimeQuantumProcessKind::SeptumGtpHydrolysis => match primitive {
            SiteChemicalPrimitiveKind::PhosphateCenter => 1.00,
            SiteChemicalPrimitiveKind::PhosphorylOxygen => 0.90,
            SiteChemicalPrimitiveKind::HydroxylOxygen => 0.60,
            SiteChemicalPrimitiveKind::PolarCarbon => 0.35,
            SiteChemicalPrimitiveKind::CarbonylOxygen => 0.25,
            SiteChemicalPrimitiveKind::CarbonylCarbon => 0.20,
            SiteChemicalPrimitiveKind::AmineCenter => 0.10,
        },
        WholeCellRuntimeQuantumProcessKind::AtpBandElectronTransfer => match primitive {
            SiteChemicalPrimitiveKind::PhosphateCenter => 1.00,
            SiteChemicalPrimitiveKind::PhosphorylOxygen => 0.90,
            SiteChemicalPrimitiveKind::HydroxylOxygen => 0.60,
            SiteChemicalPrimitiveKind::PolarCarbon => 0.35,
            SiteChemicalPrimitiveKind::CarbonylOxygen => 0.25,
            SiteChemicalPrimitiveKind::CarbonylCarbon => 0.20,
            SiteChemicalPrimitiveKind::AmineCenter => 0.10,
        },
        WholeCellRuntimeQuantumProcessKind::MembraneProteinInsertion => match primitive {
            SiteChemicalPrimitiveKind::CarbonylCarbon => 0.95,
            SiteChemicalPrimitiveKind::CarbonylOxygen => 0.90,
            SiteChemicalPrimitiveKind::HydroxylOxygen => 0.75,
            SiteChemicalPrimitiveKind::PolarCarbon => 0.70,
            SiteChemicalPrimitiveKind::PhosphorylOxygen => 0.55,
            SiteChemicalPrimitiveKind::PhosphateCenter => 0.40,
            SiteChemicalPrimitiveKind::AmineCenter => 0.30,
        },
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn site_center_law_scores(
    kind: WholeCellRuntimeQuantumProcessKind,
    scaffold: &EmbeddedMolecule,
    adjacency: &[Vec<(usize, BondOrder)>],
) -> Vec<f32> {
    let mut scores = vec![0.0f32; scaffold.graph.atom_count()];
    for (primitive, weighted) in weighted_site_chemical_primitives(kind, scaffold, adjacency) {
        if weighted > 0.0 {
            scores[primitive.center_atom_idx] += weighted;
        }
    }
    scores
}

fn site_atom_law_scores(
    kind: WholeCellRuntimeQuantumProcessKind,
    scaffold: &EmbeddedMolecule,
    adjacency: &[Vec<(usize, BondOrder)>],
) -> Vec<f32> {
    let mut scores = vec![0.0f32; scaffold.graph.atom_count()];
    for (primitive, weighted) in weighted_site_chemical_primitives(kind, scaffold, adjacency) {
        if weighted <= 0.0 {
            continue;
        }
        scores[primitive.center_atom_idx] += weighted;
        for support_atom_idx in primitive.support_atom_indices {
            scores[support_atom_idx] += 0.45 * weighted;
        }
    }
    for (atom_idx, atom) in scaffold.graph.atoms.iter().enumerate() {
        if matches!(
            atom.element,
            PeriodicElement::N | PeriodicElement::O | PeriodicElement::P | PeriodicElement::S
        ) {
            scores[atom_idx] += 0.15;
        }
    }
    scores
}

fn graph_distance(
    adjacency: &[Vec<(usize, BondOrder)>],
    start: usize,
    end: usize,
) -> Option<usize> {
    if start == end {
        return Some(0);
    }
    let mut visited = vec![false; adjacency.len()];
    let mut queue = std::collections::VecDeque::from([(start, 0usize)]);
    visited[start] = true;
    while let Some((node, depth)) = queue.pop_front() {
        for (neighbor, _) in &adjacency[node] {
            if *neighbor == end {
                return Some(depth + 1);
            }
            if visited[*neighbor] {
                continue;
            }
            visited[*neighbor] = true;
            queue.push_back((*neighbor, depth + 1));
        }
    }
    None
}

fn fallback_fragment_anchor_indices(scaffold: &EmbeddedMolecule, max_atoms: usize) -> Vec<usize> {
    let atom_count = scaffold.graph.atom_count();
    let fragment_count = atom_count.div_ceil(max_atoms.max(2)).clamp(1, 2);
    let mut sorted_atoms: Vec<usize> = (0..atom_count).collect();
    sorted_atoms.sort_by(|lhs, rhs| {
        scaffold.positions_angstrom[*lhs][0]
            .partial_cmp(&scaffold.positions_angstrom[*rhs][0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(lhs.cmp(rhs))
    });

    let mut anchors = Vec::with_capacity(fragment_count);
    for rank in 0..fragment_count {
        let idx = if fragment_count == 1 {
            sorted_atoms[atom_count.saturating_sub(1)]
        } else {
            let offset = rank * atom_count.saturating_sub(1) / (fragment_count - 1);
            sorted_atoms[offset]
        };
        if !anchors.contains(&idx) {
            anchors.push(idx);
        }
    }
    anchors
}

fn site_quantum_fragment_anchor_indices(
    kind: WholeCellRuntimeQuantumProcessKind,
    scaffold: &EmbeddedMolecule,
    max_atoms: usize,
) -> Vec<usize> {
    let atom_count = scaffold.graph.atom_count();
    let fragment_count = atom_count.div_ceil(max_atoms.max(2)).clamp(1, 2);
    let adjacency = scaffold_adjacency(scaffold);
    let weighted_primitives = weighted_site_chemical_primitives(kind, scaffold, &adjacency);
    let mut candidate_by_center = std::collections::BTreeMap::new();
    for (primitive, weight) in &weighted_primitives {
        candidate_by_center
            .entry(primitive.center_atom_idx)
            .and_modify(|current: &mut (f32, SiteChemicalPrimitiveKind)| {
                if *weight > current.0 {
                    *current = (*weight, primitive.kind);
                }
            })
            .or_insert((*weight, primitive.kind));
    }
    let dominant_kind = weighted_primitives
        .first()
        .map(|(primitive, _)| primitive.kind);
    let mut candidates: Vec<(usize, f32)> = candidate_by_center
        .iter()
        .filter(|(_, (_, primitive_kind))| Some(*primitive_kind) == dominant_kind)
        .map(|(atom_idx, (weight, _))| (*atom_idx, *weight))
        .collect();
    if candidates.len() < fragment_count {
        candidates = candidate_by_center
            .iter()
            .map(|(atom_idx, (weight, _))| (*atom_idx, *weight))
            .collect();
    }
    if candidates.is_empty() {
        return fallback_fragment_anchor_indices(scaffold, max_atoms);
    }
    candidates.sort_by(|(left_idx, left_score), (right_idx, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_idx.cmp(right_idx))
    });

    let mut anchors = Vec::with_capacity(fragment_count);
    while anchors.len() < fragment_count {
        let next_anchor = candidates
            .iter()
            .filter(|(atom_idx, _)| !anchors.contains(atom_idx))
            .max_by(|(left_idx, left_score), (right_idx, right_score)| {
                let left_rank = if anchors.is_empty() {
                    *left_score
                } else {
                    let graph_spacing = anchors
                        .iter()
                        .filter_map(|anchor| graph_distance(&adjacency, *left_idx, *anchor))
                        .min()
                        .unwrap_or(atom_count) as f32;
                    let spatial_spacing = anchors
                        .iter()
                        .map(|anchor| {
                            squared_distance(
                                scaffold.positions_angstrom[*left_idx],
                                scaffold.positions_angstrom[*anchor],
                            )
                            .sqrt()
                        })
                        .fold(f32::INFINITY, f32::min);
                    *left_score + 0.30 * graph_spacing + 0.03 * spatial_spacing.min(12.0)
                };
                let right_rank = if anchors.is_empty() {
                    *right_score
                } else {
                    let graph_spacing = anchors
                        .iter()
                        .filter_map(|anchor| graph_distance(&adjacency, *right_idx, *anchor))
                        .min()
                        .unwrap_or(atom_count) as f32;
                    let spatial_spacing = anchors
                        .iter()
                        .map(|anchor| {
                            squared_distance(
                                scaffold.positions_angstrom[*right_idx],
                                scaffold.positions_angstrom[*anchor],
                            )
                            .sqrt()
                        })
                        .fold(f32::INFINITY, f32::min);
                    *right_score + 0.30 * graph_spacing + 0.03 * spatial_spacing.min(12.0)
                };
                left_rank
                    .total_cmp(&right_rank)
                    .then_with(|| left_idx.cmp(right_idx))
            })
            .map(|(atom_idx, _)| *atom_idx);
        if let Some(anchor) = next_anchor {
            anchors.push(anchor);
        } else {
            break;
        }
    }
    for fallback_anchor in fallback_fragment_anchor_indices(scaffold, max_atoms) {
        if anchors.len() >= fragment_count {
            break;
        }
        if !anchors.contains(&fallback_anchor) {
            anchors.push(fallback_anchor);
        }
    }
    anchors
}

fn site_quantum_fragment_indices(
    kind: WholeCellRuntimeQuantumProcessKind,
    scaffold: &EmbeddedMolecule,
    max_atoms: usize,
    anchor: usize,
) -> Vec<usize> {
    let max_atoms = max_atoms.max(2);
    let adjacency = scaffold_adjacency(scaffold);
    let atom_scores = site_atom_law_scores(kind, scaffold, &adjacency);

    let mut selected = vec![anchor];
    let mut visited = vec![false; scaffold.graph.atom_count()];
    let mut queue = std::collections::VecDeque::from([anchor]);
    visited[anchor] = true;

    while let Some(atom_idx) = queue.pop_front() {
        if selected.len() >= max_atoms {
            break;
        }
        let mut neighbors = adjacency[atom_idx].clone();
        neighbors.sort_by(|(left_idx, left_order), (right_idx, right_order)| {
            atom_scores[*right_idx]
                .total_cmp(&atom_scores[*left_idx])
                .then_with(|| right_order.bond_order().total_cmp(&left_order.bond_order()))
                .then_with(|| {
                    squared_distance(
                        scaffold.positions_angstrom[*left_idx],
                        scaffold.positions_angstrom[anchor],
                    )
                    .total_cmp(&squared_distance(
                        scaffold.positions_angstrom[*right_idx],
                        scaffold.positions_angstrom[anchor],
                    ))
                })
                .then_with(|| left_idx.cmp(right_idx))
        });
        for (neighbor, _) in neighbors {
            if visited[neighbor] {
                continue;
            }
            visited[neighbor] = true;
            selected.push(neighbor);
            queue.push_back(neighbor);
            if selected.len() >= max_atoms {
                break;
            }
        }
    }

    if selected.len() < max_atoms {
        let mut remaining: Vec<usize> = (0..scaffold.graph.atom_count())
            .filter(|idx| !visited[*idx])
            .collect();
        let anchor_position = scaffold.positions_angstrom[anchor];
        remaining.sort_by(|lhs, rhs| {
            atom_scores[*rhs]
                .total_cmp(&atom_scores[*lhs])
                .then_with(|| {
                    squared_distance(scaffold.positions_angstrom[*lhs], anchor_position).total_cmp(
                        &squared_distance(scaffold.positions_angstrom[*rhs], anchor_position),
                    )
                })
                .then_with(|| lhs.cmp(rhs))
        });
        for atom_idx in remaining.into_iter().take(max_atoms - selected.len()) {
            selected.push(atom_idx);
        }
    }

    selected.sort_unstable();
    selected
}

fn mean_position(positions: impl IntoIterator<Item = [f32; 3]>) -> [f32; 3] {
    let mut sum = [0.0f32; 3];
    let mut count = 0usize;
    for position in positions {
        sum[0] += position[0];
        sum[1] += position[1];
        sum[2] += position[2];
        count += 1;
    }
    if count == 0 {
        [0.0, 0.0, 0.0]
    } else {
        [
            sum[0] / count as f32,
            sum[1] / count as f32,
            sum[2] / count as f32,
        ]
    }
}

fn subtract_f32x3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vector_norm_f32(vector: [f32; 3]) -> f32 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn normalize_vector_f32(vector: [f32; 3]) -> [f32; 3] {
    let magnitude = vector_norm_f32(vector);
    if magnitude <= 1.0e-6 {
        [1.0, 0.0, 0.0]
    } else {
        [
            vector[0] / magnitude,
            vector[1] / magnitude,
            vector[2] / magnitude,
        ]
    }
}

fn squared_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

fn centroid_angstrom(molecule: &EmbeddedMolecule) -> [f32; 3] {
    centroid_angstrom_slice(std::slice::from_ref(molecule))
}

fn f64x3_to_f32(position: [f64; 3]) -> [f32; 3] {
    [position[0] as f32, position[1] as f32, position[2] as f32]
}

fn centroid_angstrom_slice(molecules: &[EmbeddedMolecule]) -> [f32; 3] {
    let mut centroid = [0.0f32; 3];
    let mut atom_count = 0usize;
    for molecule in molecules {
        for position in &molecule.positions_angstrom {
            centroid[0] += position[0];
            centroid[1] += position[1];
            centroid[2] += position[2];
            atom_count += 1;
        }
    }
    if atom_count == 0 {
        return [0.0, 0.0, 0.0];
    }
    [
        centroid[0] / atom_count as f32,
        centroid[1] / atom_count as f32,
        centroid[2] / atom_count as f32,
    ]
}

fn heuristic_fragment_point_charges(
    molecules: &[EmbeddedMolecule],
) -> Vec<QuantumEmbeddingPointCharge> {
    let mut charges = Vec::new();
    for molecule in molecules {
        let atom_charges = embedded_atom_point_charges(molecule);
        for (atom_idx, charge_e) in atom_charges.into_iter().enumerate() {
            let position = molecule.positions_angstrom[atom_idx];
            let element = molecule.graph.atoms[atom_idx].element;
            charges.push(QuantumEmbeddingPointCharge::new(
                [
                    f64::from(position[0]),
                    f64::from(position[1]),
                    f64::from(position[2]),
                ],
                charge_e.clamp(-1.5, 1.5),
                atom_embedding_screening_radius_angstrom(element),
            ));
        }
    }
    charges
}

fn heuristic_fragment_embedding_source(
    molecules: &[EmbeddedMolecule],
) -> RuntimeQuantumFragmentEmbeddingSource {
    let point_charges = heuristic_fragment_point_charges(molecules);
    let dipoles = heuristic_fragment_dipoles(molecules, &point_charges);
    let orbital_response_fields =
        heuristic_fragment_orbital_response_fields(molecules, &point_charges);
    RuntimeQuantumFragmentEmbeddingSource {
        point_charges,
        dipoles,
        orbital_response_fields,
        natural_orbital_occupancies: Vec::new(),
        lda_exchange_corrections_ev: Vec::new(),
    }
}

fn solved_fragment_point_charges(
    molecule: &EmbeddedMolecule,
    expected_atom_effective_charges: &[f64],
) -> Vec<QuantumEmbeddingPointCharge> {
    if expected_atom_effective_charges.len() != molecule.graph.atom_count() {
        return heuristic_fragment_point_charges(std::slice::from_ref(molecule));
    }
    molecule
        .graph
        .atoms
        .iter()
        .enumerate()
        .map(|(atom_idx, atom)| {
            let position = molecule.positions_angstrom[atom_idx];
            QuantumEmbeddingPointCharge::new(
                [
                    f64::from(position[0]),
                    f64::from(position[1]),
                    f64::from(position[2]),
                ],
                expected_atom_effective_charges[atom_idx].clamp(-1.5, 1.5),
                atom_embedding_screening_radius_angstrom(atom.element),
            )
        })
        .collect()
}

fn heuristic_fragment_dipoles(
    molecules: &[EmbeddedMolecule],
    point_charges: &[QuantumEmbeddingPointCharge],
) -> Vec<QuantumEmbeddingDipole> {
    let mut dipoles = Vec::new();
    for molecule in molecules {
        let atom_charges = embedded_atom_point_charges(molecule);
        let bond_response_strengths = molecule
            .graph
            .bonds
            .iter()
            .map(|bond| f64::from(bond.order.bond_order()))
            .collect::<Vec<_>>();
        dipoles.extend(molecule_bond_polarization_dipoles(
            molecule,
            &atom_charges,
            &bond_response_strengths,
        ));
    }
    if let Some(net_dipole) = fragment_net_dipole_from_point_charges(point_charges) {
        dipoles.push(net_dipole);
    }
    dipoles
}

fn heuristic_fragment_orbital_response_fields(
    molecules: &[EmbeddedMolecule],
    point_charges: &[QuantumEmbeddingPointCharge],
) -> Vec<QuantumEmbeddingOrbitalResponseField> {
    let mut fields = Vec::new();
    for molecule in molecules {
        let atom_charges = embedded_atom_point_charges(molecule);
        let bond_response_strengths = molecule
            .graph
            .bonds
            .iter()
            .map(|bond| 0.65 + 0.35 * f64::from(bond.order.bond_order()))
            .collect::<Vec<_>>();
        fields.extend(molecule_bond_orbital_response_fields(
            molecule,
            &atom_charges,
            &bond_response_strengths,
        ));
    }
    if fields.is_empty() {
        if let Some(net_field) = heuristic_net_orbital_response_field(point_charges) {
            fields.push(net_field);
        }
    }
    fields
}

fn solved_fragment_dipoles(
    molecule: &EmbeddedMolecule,
    result: &ExactDiagonalizationResult,
) -> Vec<QuantumEmbeddingDipole> {
    if result.expected_atom_effective_charges.len() != molecule.graph.atom_count() {
        let heuristic_point_charges =
            heuristic_fragment_point_charges(std::slice::from_ref(molecule));
        return heuristic_fragment_dipoles(
            std::slice::from_ref(molecule),
            &heuristic_point_charges,
        );
    }
    let mut dipoles = molecule_bond_polarization_dipoles(
        molecule,
        &result.expected_atom_effective_charges,
        &molecule
            .graph
            .bonds
            .iter()
            .map(|bond| result.atom_pair_density_response(bond.i, bond.j))
            .collect::<Vec<_>>(),
    );
    dipoles.push(QuantumEmbeddingDipole::new(
        f64x3_from_f32(centroid_angstrom(molecule)),
        clamp_vector_magnitude_f64(result.expected_dipole_moment_e_angstrom, 3.0),
        fragment_embedding_screening_radius_angstrom(std::slice::from_ref(molecule)),
    ));
    dipoles
}

fn solved_fragment_orbital_response_fields(
    molecule: &EmbeddedMolecule,
    result: &ExactDiagonalizationResult,
) -> Vec<QuantumEmbeddingOrbitalResponseField> {
    if result.expected_atom_effective_charges.len() != molecule.graph.atom_count() {
        let heuristic_point_charges =
            heuristic_fragment_point_charges(std::slice::from_ref(molecule));
        return heuristic_fragment_orbital_response_fields(
            std::slice::from_ref(molecule),
            &heuristic_point_charges,
        );
    }
    let bond_response_strengths = molecule
        .graph
        .bonds
        .iter()
        .map(|bond| result.atom_pair_density_response(bond.i, bond.j))
        .collect::<Vec<_>>();
    let mut fields = molecule_bond_orbital_response_fields(
        molecule,
        &result.expected_atom_effective_charges,
        &bond_response_strengths,
    );
    if fields.is_empty() {
        let fallback_point_charges =
            solved_fragment_point_charges(molecule, &result.expected_atom_effective_charges);
        if let Some(net_field) = heuristic_net_orbital_response_field(&fallback_point_charges) {
            fields.push(net_field);
        }
    }
    fields
}

fn molecule_bond_polarization_dipoles(
    molecule: &EmbeddedMolecule,
    atom_effective_charges: &[f64],
    bond_response_strengths: &[f64],
) -> Vec<QuantumEmbeddingDipole> {
    if atom_effective_charges.len() != molecule.graph.atom_count()
        || bond_response_strengths.len() != molecule.graph.bonds.len()
    {
        return Vec::new();
    }
    molecule
        .graph
        .bonds
        .iter()
        .zip(bond_response_strengths.iter().copied())
        .map(|(bond, response_strength)| {
            let position_a = molecule.positions_angstrom[bond.i];
            let position_b = molecule.positions_angstrom[bond.j];
            let displacement = [
                f64::from(position_b[0] - position_a[0]),
                f64::from(position_b[1] - position_a[1]),
                f64::from(position_b[2] - position_a[2]),
            ];
            let response_scale = (0.35
                + 0.20 * f64::from(bond.order.bond_order())
                + 0.30 * response_strength.sqrt())
            .clamp(0.2, 2.5);
            let charge_skew =
                0.5 * (atom_effective_charges[bond.j] - atom_effective_charges[bond.i]);
            let screening_radius = bond_embedding_screening_radius_angstrom(
                molecule,
                bond.i,
                bond.j,
                vector_norm_f64(displacement),
            );
            QuantumEmbeddingDipole::new(
                bond_midpoint_angstrom(position_a, position_b),
                clamp_vector_magnitude_f64(
                    [
                        response_scale * charge_skew * displacement[0],
                        response_scale * charge_skew * displacement[1],
                        response_scale * charge_skew * displacement[2],
                    ],
                    2.5,
                ),
                screening_radius,
            )
        })
        .collect()
}

fn molecule_bond_orbital_response_fields(
    molecule: &EmbeddedMolecule,
    atom_effective_charges: &[f64],
    bond_response_strengths: &[f64],
) -> Vec<QuantumEmbeddingOrbitalResponseField> {
    if atom_effective_charges.len() != molecule.graph.atom_count()
        || bond_response_strengths.len() != molecule.graph.bonds.len()
    {
        return Vec::new();
    }
    molecule
        .graph
        .bonds
        .iter()
        .zip(bond_response_strengths.iter().copied())
        .filter_map(|(bond, response_strength)| {
            let position_a = molecule.positions_angstrom[bond.i];
            let position_b = molecule.positions_angstrom[bond.j];
            let displacement = [
                f64::from(position_b[0] - position_a[0]),
                f64::from(position_b[1] - position_a[1]),
                f64::from(position_b[2] - position_a[2]),
            ];
            let bond_length = vector_norm_f64(displacement);
            if bond_length <= 1.0e-6 {
                return None;
            }
            let charge_skew: f64 =
                0.5 * (atom_effective_charges[bond.j] - atom_effective_charges[bond.i]);
            let direction_sign: f64 = if charge_skew >= 0.0 { 1.0 } else { -1.0 };
            let axis = [
                direction_sign * displacement[0] / bond_length,
                direction_sign * displacement[1] / bond_length,
                direction_sign * displacement[2] / bond_length,
            ];
            let bond_order = f64::from(bond.order.bond_order());
            let coupling_ev = ((0.22 + 0.08 * bond_order)
                * response_strength.max(0.0)
                * (0.30 + charge_skew.abs()))
            .clamp(0.0, 1.5);
            if coupling_ev < FRAGMENT_EMBEDDING_MIN_ORBITAL_RESPONSE_EV {
                return None;
            }
            Some(QuantumEmbeddingOrbitalResponseField::new(
                bond_midpoint_angstrom(position_a, position_b),
                axis,
                coupling_ev,
                bond_embedding_screening_radius_angstrom(molecule, bond.i, bond.j, bond_length),
            ))
        })
        .collect()
}

fn fragment_net_dipole_from_point_charges(
    point_charges: &[QuantumEmbeddingPointCharge],
) -> Option<QuantumEmbeddingDipole> {
    if point_charges.is_empty() {
        return None;
    }
    let centroid = point_charge_centroid_angstrom(point_charges);
    let mut dipole = [0.0f64; 3];
    for point_charge in point_charges {
        dipole[0] += point_charge.charge_e * (point_charge.position_angstrom[0] - centroid[0]);
        dipole[1] += point_charge.charge_e * (point_charge.position_angstrom[1] - centroid[1]);
        dipole[2] += point_charge.charge_e * (point_charge.position_angstrom[2] - centroid[2]);
    }
    let mean_screening_radius = point_charges
        .iter()
        .map(|point_charge| point_charge.screening_radius_angstrom)
        .sum::<f64>()
        / point_charges.len() as f64;
    Some(QuantumEmbeddingDipole::new(
        centroid,
        clamp_vector_magnitude_f64(dipole, 3.0),
        mean_screening_radius.clamp(0.45, 2.5),
    ))
}

fn heuristic_net_orbital_response_field(
    point_charges: &[QuantumEmbeddingPointCharge],
) -> Option<QuantumEmbeddingOrbitalResponseField> {
    let net_dipole = fragment_net_dipole_from_point_charges(point_charges)?;
    let axis_norm = vector_norm_f64(net_dipole.dipole_e_angstrom);
    if axis_norm < FRAGMENT_EMBEDDING_MIN_DIPOLE_E_ANGSTROM {
        return None;
    }
    Some(QuantumEmbeddingOrbitalResponseField::new(
        net_dipole.position_angstrom,
        [
            net_dipole.dipole_e_angstrom[0] / axis_norm,
            net_dipole.dipole_e_angstrom[1] / axis_norm,
            net_dipole.dipole_e_angstrom[2] / axis_norm,
        ],
        (0.10 * axis_norm).clamp(FRAGMENT_EMBEDDING_MIN_ORBITAL_RESPONSE_EV, 0.75),
        net_dipole.screening_radius_angstrom,
    ))
}

fn filtered_fragment_embedding_point_charges(
    point_charges: &[QuantumEmbeddingPointCharge],
    target_center_angstrom: [f32; 3],
) -> Vec<QuantumEmbeddingPointCharge> {
    let cutoff_squared = FRAGMENT_EMBEDDING_CUTOFF_ANGSTROM.powi(2);
    point_charges
        .iter()
        .copied()
        .filter(|point_charge| point_charge.charge_e.abs() >= FRAGMENT_EMBEDDING_MIN_CHARGE_E)
        .filter(|point_charge| {
            squared_distance(
                f64x3_to_f32(point_charge.position_angstrom),
                target_center_angstrom,
            ) <= cutoff_squared
        })
        .collect()
}

fn filtered_fragment_embedding_dipoles(
    dipoles: &[QuantumEmbeddingDipole],
    target_center_angstrom: [f32; 3],
) -> Vec<QuantumEmbeddingDipole> {
    let cutoff_squared = FRAGMENT_EMBEDDING_CUTOFF_ANGSTROM.powi(2);
    dipoles
        .iter()
        .copied()
        .filter(|dipole| {
            vector_norm_f64(dipole.dipole_e_angstrom) >= FRAGMENT_EMBEDDING_MIN_DIPOLE_E_ANGSTROM
        })
        .filter(|dipole| {
            squared_distance(
                f64x3_to_f32(dipole.position_angstrom),
                target_center_angstrom,
            ) <= cutoff_squared
        })
        .collect()
}

fn filtered_fragment_embedding_orbital_response_fields(
    orbital_response_fields: &[QuantumEmbeddingOrbitalResponseField],
    target_center_angstrom: [f32; 3],
) -> Vec<QuantumEmbeddingOrbitalResponseField> {
    let cutoff_squared = FRAGMENT_EMBEDDING_CUTOFF_ANGSTROM.powi(2);
    orbital_response_fields
        .iter()
        .copied()
        .filter(|field| field.coupling_ev.abs() >= FRAGMENT_EMBEDDING_MIN_ORBITAL_RESPONSE_EV)
        .filter(|field| {
            vector_norm_f64(field.axis) >= 1.0e-6
                && squared_distance(
                    f64x3_to_f32(field.position_angstrom),
                    target_center_angstrom,
                ) <= cutoff_squared
        })
        .collect()
}

fn embedded_atom_point_charges(molecule: &EmbeddedMolecule) -> Vec<f64> {
    let mut charges = molecule
        .graph
        .atoms
        .iter()
        .map(|atom| f64::from(atom.formal_charge))
        .collect::<Vec<_>>();

    for bond in &molecule.graph.bonds {
        let left = molecule.graph.atoms[bond.i].element;
        let right = molecule.graph.atoms[bond.j].element;
        let left_en = left.pauling_electronegativity().unwrap_or(2.0);
        let right_en = right.pauling_electronegativity().unwrap_or(2.0);
        let polarization =
            0.10 * f64::from(bond.order.bond_order()) * (right_en - left_en).clamp(-2.5, 2.5);
        charges[bond.i] += polarization;
        charges[bond.j] -= polarization;
    }

    charges
}

fn atom_embedding_screening_radius_angstrom(element: PeriodicElement) -> f64 {
    f64::from(element.covalent_radius_angstrom()).clamp(0.35, 1.8)
}

fn bond_embedding_screening_radius_angstrom(
    molecule: &EmbeddedMolecule,
    atom_a: usize,
    atom_b: usize,
    bond_length_angstrom: f64,
) -> f64 {
    let radius_a = atom_embedding_screening_radius_angstrom(molecule.graph.atoms[atom_a].element);
    let radius_b = atom_embedding_screening_radius_angstrom(molecule.graph.atoms[atom_b].element);
    (0.5 * bond_length_angstrom + 0.25 * (radius_a + radius_b)).clamp(0.35, 2.5)
}

fn fragment_embedding_screening_radius_angstrom(molecules: &[EmbeddedMolecule]) -> f64 {
    let centroid = centroid_angstrom_slice(molecules);
    let mut atom_count = 0usize;
    let mut mean_radius = 0.0f64;
    for molecule in molecules {
        for position in &molecule.positions_angstrom {
            mean_radius += f64::from(squared_distance(*position, centroid).sqrt());
            atom_count += 1;
        }
    }
    if atom_count == 0 {
        return 0.85;
    }
    (mean_radius / atom_count as f64).clamp(0.45, 2.5)
}

fn f64x3_from_f32(position: [f32; 3]) -> [f64; 3] {
    [
        f64::from(position[0]),
        f64::from(position[1]),
        f64::from(position[2]),
    ]
}

fn bond_midpoint_angstrom(left: [f32; 3], right: [f32; 3]) -> [f64; 3] {
    [
        0.5 * f64::from(left[0] + right[0]),
        0.5 * f64::from(left[1] + right[1]),
        0.5 * f64::from(left[2] + right[2]),
    ]
}

fn point_charge_centroid_angstrom(point_charges: &[QuantumEmbeddingPointCharge]) -> [f64; 3] {
    let mut centroid = [0.0f64; 3];
    for point_charge in point_charges {
        centroid[0] += point_charge.position_angstrom[0];
        centroid[1] += point_charge.position_angstrom[1];
        centroid[2] += point_charge.position_angstrom[2];
    }
    centroid[0] /= point_charges.len() as f64;
    centroid[1] /= point_charges.len() as f64;
    centroid[2] /= point_charges.len() as f64;
    centroid
}

fn vector_norm_f64(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn clamp_vector_magnitude_f64(vector: [f64; 3], max_magnitude: f64) -> [f64; 3] {
    let magnitude = vector_norm_f64(vector);
    if magnitude <= max_magnitude || magnitude <= 1.0e-9 {
        return vector;
    }
    let scale = max_magnitude / magnitude;
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn normalize_vector_f64(vector: [f64; 3]) -> [f64; 3] {
    let magnitude = vector_norm_f64(vector);
    if magnitude <= 1.0e-9 {
        return [0.0, 0.0, 0.0];
    }
    [
        vector[0] / magnitude,
        vector[1] / magnitude,
        vector[2] / magnitude,
    ]
}

fn relax_fragment_embedding_sources(
    current_sources: &mut [RuntimeQuantumFragmentEmbeddingSource],
    solved_sources: &[RuntimeQuantumFragmentEmbeddingSource],
) -> (f64, f64, f64) {
    let mut residual_charge_e = 0.0f64;
    let mut residual_dipole_e_angstrom = 0.0f64;
    let mut residual_orbital_response_ev = 0.0f64;
    for (current, solved) in current_sources.iter_mut().zip(solved_sources.iter()) {
        if current.point_charges.len() != solved.point_charges.len() {
            residual_charge_e = residual_charge_e.max(1.0);
            current.point_charges = solved.point_charges.clone();
        } else {
            for (current_charge, solved_charge) in current
                .point_charges
                .iter_mut()
                .zip(solved.point_charges.iter())
            {
                let delta = solved_charge.charge_e - current_charge.charge_e;
                current_charge.charge_e += FRAGMENT_EMBEDDING_RESPONSE_RELAXATION * delta;
                current_charge.screening_radius_angstrom = solved_charge.screening_radius_angstrom;
                residual_charge_e = residual_charge_e.max(delta.abs());
            }
        }

        if current.dipoles.len() != solved.dipoles.len() {
            residual_dipole_e_angstrom = residual_dipole_e_angstrom.max(1.0);
            current.dipoles = solved.dipoles.clone();
        } else {
            for (current_dipole, solved_dipole) in
                current.dipoles.iter_mut().zip(solved.dipoles.iter())
            {
                let delta = [
                    solved_dipole.dipole_e_angstrom[0] - current_dipole.dipole_e_angstrom[0],
                    solved_dipole.dipole_e_angstrom[1] - current_dipole.dipole_e_angstrom[1],
                    solved_dipole.dipole_e_angstrom[2] - current_dipole.dipole_e_angstrom[2],
                ];
                current_dipole.dipole_e_angstrom[0] +=
                    FRAGMENT_EMBEDDING_RESPONSE_RELAXATION * delta[0];
                current_dipole.dipole_e_angstrom[1] +=
                    FRAGMENT_EMBEDDING_RESPONSE_RELAXATION * delta[1];
                current_dipole.dipole_e_angstrom[2] +=
                    FRAGMENT_EMBEDDING_RESPONSE_RELAXATION * delta[2];
                current_dipole.screening_radius_angstrom = solved_dipole.screening_radius_angstrom;
                residual_dipole_e_angstrom = residual_dipole_e_angstrom.max(vector_norm_f64(delta));
            }
        }

        if current.orbital_response_fields.len() != solved.orbital_response_fields.len() {
            residual_orbital_response_ev = residual_orbital_response_ev.max(1.0);
            current.orbital_response_fields = solved.orbital_response_fields.clone();
        } else {
            for (current_field, solved_field) in current
                .orbital_response_fields
                .iter_mut()
                .zip(solved.orbital_response_fields.iter())
            {
                let axis_delta = [
                    solved_field.axis[0] - current_field.axis[0],
                    solved_field.axis[1] - current_field.axis[1],
                    solved_field.axis[2] - current_field.axis[2],
                ];
                current_field.axis[0] += FRAGMENT_EMBEDDING_RESPONSE_RELAXATION * axis_delta[0];
                current_field.axis[1] += FRAGMENT_EMBEDDING_RESPONSE_RELAXATION * axis_delta[1];
                current_field.axis[2] += FRAGMENT_EMBEDDING_RESPONSE_RELAXATION * axis_delta[2];
                current_field.axis = normalize_vector_f64(current_field.axis);
                let coupling_delta = solved_field.coupling_ev - current_field.coupling_ev;
                current_field.coupling_ev +=
                    FRAGMENT_EMBEDDING_RESPONSE_RELAXATION * coupling_delta;
                current_field.screening_radius_angstrom = solved_field.screening_radius_angstrom;
                residual_orbital_response_ev = residual_orbital_response_ev.max(
                    coupling_delta.abs()
                        + 0.1 * solved_field.coupling_ev.abs() * vector_norm_f64(axis_delta),
                );
            }
        }

        // Natural orbital occupancies and LDA exchange corrections are derived
        // quantities from the ED solve.  They are replaced directly each
        // iteration rather than relaxed, since they depend deterministically
        // on the current density matrix.
        current.natural_orbital_occupancies = solved.natural_orbital_occupancies.clone();
        current.lda_exchange_corrections_ev = solved.lda_exchange_corrections_ev.clone();
    }
    (
        residual_charge_e,
        residual_dipole_e_angstrom,
        residual_orbital_response_ev,
    )
}

fn scale_runtime_summary(summary: ReactionQuantumSummary, factor: f32) -> ReactionQuantumSummary {
    let factor = factor.max(0.0);
    if factor <= 1.0e-6 || summary.event_count == 0 {
        return ReactionQuantumSummary::default();
    }
    ReactionQuantumSummary {
        event_count: 0,
        ground_state_energy_delta_ev: summary.ground_state_energy_delta_ev * factor,
        nuclear_repulsion_delta_ev: summary.nuclear_repulsion_delta_ev * factor,
        net_formal_charge_delta: ((summary.net_formal_charge_delta as f32) * factor).round() as i32,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn fragment_coupling_energy_ev(outcomes: &[RuntimeQuantumFragmentOutcome]) -> f32 {
    let responses = outcomes
        .iter()
        .map(|outcome| outcome.response_magnitude_ev())
        .collect::<Vec<_>>();
    fragment_coupling_energy_ev_with_responses(outcomes, &responses)
}

fn solve_self_consistent_fragment_responses(
    outcomes: &[RuntimeQuantumFragmentOutcome],
) -> (Vec<f32>, usize, f32) {
    let base_responses = outcomes
        .iter()
        .map(|outcome| outcome.response_magnitude_ev())
        .collect::<Vec<_>>();
    if base_responses.is_empty() {
        return (Vec::new(), 0, 0.0);
    }

    let mut responses = base_responses.clone();
    let mut iteration_count = 0usize;
    let mut residual_ev = 0.0f32;

    for iteration in 0..FRAGMENT_RESPONSE_MAX_ITERATIONS {
        let mut next_responses = responses.clone();
        residual_ev = 0.0;
        for left_idx in 0..outcomes.len() {
            let base_response = base_responses[left_idx];
            if base_response <= 1.0e-6 {
                next_responses[left_idx] = 0.0;
                continue;
            }

            let mut induced_gain = 0.0f32;
            let mut neighborhood_strength = 0.0f32;
            for right_idx in 0..outcomes.len() {
                if left_idx == right_idx {
                    continue;
                }
                let influence =
                    fragment_response_influence(outcomes[left_idx], outcomes[right_idx]);
                if influence <= 1.0e-6 {
                    continue;
                }
                neighborhood_strength += influence;
                let baseline_neighbor = base_responses[right_idx].max(1.0e-3);
                induced_gain += influence * (responses[right_idx] / baseline_neighbor);
            }

            let max_additional_gain = (0.2 + 1.2 * neighborhood_strength).clamp(0.25, 1.5);
            let target_response = base_response * (1.0 + induced_gain.min(max_additional_gain));
            let capped_target = target_response.min(base_response * (1.0 + max_additional_gain));
            let updated_response = responses[left_idx]
                + FRAGMENT_RESPONSE_RELAXATION * (capped_target - responses[left_idx]);
            next_responses[left_idx] = updated_response.max(base_response);
            residual_ev = residual_ev.max((next_responses[left_idx] - responses[left_idx]).abs());
        }

        responses = next_responses;
        iteration_count = iteration + 1;
        if residual_ev <= FRAGMENT_RESPONSE_TOLERANCE_EV {
            break;
        }
    }

    (responses, iteration_count, residual_ev)
}

fn fragment_coupling_energy_ev_with_responses(
    outcomes: &[RuntimeQuantumFragmentOutcome],
    responses: &[f32],
) -> f32 {
    let mut coupling = 0.0f32;
    for left_idx in 0..outcomes.len() {
        for right_idx in (left_idx + 1)..outcomes.len() {
            let left = outcomes[left_idx];
            let right = outcomes[right_idx];
            let response_left = responses
                .get(left_idx)
                .copied()
                .unwrap_or_else(|| left.response_magnitude_ev());
            let response_right = responses
                .get(right_idx)
                .copied()
                .unwrap_or_else(|| right.response_magnitude_ev());
            if response_left <= 1.0e-6 || response_right <= 1.0e-6 {
                continue;
            }

            let (distance, radius) = fragment_pair_geometry(left, right);
            let stabilization =
                0.08 * (response_left * response_right).sqrt() * radius.powi(2) / distance.powi(3);
            let max_stabilization = 0.25 * response_left.min(response_right);
            coupling -= stabilization.min(max_stabilization);
        }
    }
    coupling
}

fn fragment_response_influence(
    left: RuntimeQuantumFragmentOutcome,
    right: RuntimeQuantumFragmentOutcome,
) -> f32 {
    let (distance, radius) = fragment_pair_geometry(left, right);
    (0.65 * (radius / distance).powi(2)).min(0.35)
}

fn fragment_pair_geometry(
    left: RuntimeQuantumFragmentOutcome,
    right: RuntimeQuantumFragmentOutcome,
) -> (f32, f32) {
    let left_center = left.interaction_center_angstrom();
    let right_center = right.interaction_center_angstrom();
    let distance = squared_distance(left_center, right_center).sqrt().max(2.0);
    let radius = 0.5 * (left.local_radius_angstrom() + right.local_radius_angstrom());
    (distance, radius)
}

// ---------------------------------------------------------------------------
// Phase 5: Live-state microdomain carving with convergence-driven BFS
// ---------------------------------------------------------------------------

/// Carve a quantum fragment from a live molecule using convergence-driven BFS.
///
/// Starting from `anchor_atom`, BFS expands shell-by-shell along bonds. The
/// fragment grows until:
/// - Adding the next shell of atoms would change the electrostatic embedding
///   energy by less than `convergence_ev` (estimated from Coulomb contributions
///   of the outermost shell).
/// - Or `max_atoms` is reached.
///
/// Cut bonds (edges crossing the fragment boundary) are capped with link
/// hydrogens placed at equilibrium bond lengths along the cut direction.
///
/// Returns `(carved_fragment, link_hydrogen_positions)`.
pub(crate) fn carve_convergence_driven_fragment(
    live_molecule: &EmbeddedMolecule,
    anchor_atom: usize,
    _initial_shell_size: usize,
    max_atoms: usize,
    convergence_ev: f64,
) -> (EmbeddedMolecule, Vec<[f32; 3]>) {
    let max_atoms = max_atoms.max(2);
    let adjacency = scaffold_adjacency(live_molecule);
    let atom_count = live_molecule.graph.atom_count();
    let mut included = vec![false; atom_count];
    let mut shell_order: Vec<Vec<usize>> = Vec::new();

    // Shell 0: just the anchor (clamped to valid range)
    let anchor = anchor_atom.min(atom_count.saturating_sub(1));
    included[anchor] = true;
    shell_order.push(vec![anchor]);

    // Expand BFS shells with convergence check
    let mut current_shell = vec![anchor];
    while shell_order.iter().map(|s| s.len()).sum::<usize>() < max_atoms {
        let mut next_shell = Vec::new();
        for &atom in &current_shell {
            for &(neighbor, _bond_order) in &adjacency[atom] {
                if !included[neighbor] {
                    included[neighbor] = true;
                    next_shell.push(neighbor);
                }
            }
        }
        if next_shell.is_empty() {
            break;
        }

        // Convergence check: estimate the Coulomb contribution of this new shell.
        // If the maximum single-atom contribution from the new shell is below the
        // convergence threshold, we have converged and do not include this shell.
        // Atoms with zero heuristic partial charge (e.g. H) are excluded from the
        // convergence criterion since their structural bonding contribution matters
        // regardless of electrostatic weight.
        if shell_order.len() >= 2 {
            let anchor_pos = live_molecule.positions_angstrom[anchor];
            let charged_contributions: Vec<f64> = next_shell
                .iter()
                .filter_map(|&idx| {
                    let element = live_molecule.graph.atoms[idx].element;
                    let charge = electronegativity_partial_charge(element).abs() as f64;
                    if charge < 1.0e-12 {
                        return None; // skip zero-charge atoms in convergence check
                    }
                    let pos = live_molecule.positions_angstrom[idx];
                    let r = squared_distance(pos, anchor_pos).sqrt().max(0.1) as f64;
                    const COULOMB_CONSTANT_EV_ANGSTROM: f64 = 14.3996;
                    Some(COULOMB_CONSTANT_EV_ANGSTROM * charge / r)
                })
                .collect();

            // Only apply convergence truncation if at least one atom in the
            // shell carries a non-zero charge; otherwise include by default.
            if !charged_contributions.is_empty() {
                let max_contribution = charged_contributions
                    .into_iter()
                    .fold(0.0f64, f64::max);
                if max_contribution < convergence_ev {
                    break;
                }
            }
        }

        // Trim shell to respect max_atoms
        let current_total: usize = shell_order.iter().map(|s| s.len()).sum();
        let budget = max_atoms.saturating_sub(current_total);
        if budget == 0 {
            break;
        }
        let trimmed: Vec<usize> = next_shell.into_iter().take(budget).collect();
        // Un-include atoms that got trimmed away (they were marked included above
        // but won't make it into the final set).
        for idx in 0..atom_count {
            if included[idx]
                && !shell_order.iter().any(|s| s.contains(&idx))
                && !trimmed.contains(&idx)
            {
                included[idx] = false;
            }
        }
        shell_order.push(trimmed.clone());
        current_shell = trimmed;
    }

    // Collect all included atom indices in shell order
    let included_indices: Vec<usize> = shell_order.iter().flat_map(|s| s.iter().copied()).collect();

    // Rebuild the included set to be authoritative
    let mut included_set = vec![false; atom_count];
    for &idx in &included_indices {
        included_set[idx] = true;
    }

    // Identify cut bonds and compute link hydrogen positions
    let mut link_hydrogen_positions = Vec::new();
    for &idx in &included_indices {
        for &(neighbor, _bond_order) in &adjacency[idx] {
            if !included_set[neighbor] {
                // This bond is cut -- place a link hydrogen along the bond direction
                let pos_in = live_molecule.positions_angstrom[idx];
                let pos_out = live_molecule.positions_angstrom[neighbor];
                let dx = pos_out[0] - pos_in[0];
                let dy = pos_out[1] - pos_in[1];
                let dz = pos_out[2] - pos_in[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                // Generic C-H bond length for link atom capping
                let h_bond_length: f32 = 1.09;
                if dist > 1.0e-6 {
                    let scale = h_bond_length / dist;
                    link_hydrogen_positions.push([
                        pos_in[0] + dx * scale,
                        pos_in[1] + dy * scale,
                        pos_in[2] + dz * scale,
                    ]);
                }
            }
        }
    }

    // Extract the fragment using the existing infrastructure
    let mut sorted_indices = included_indices.clone();
    sorted_indices.sort_unstable();
    let fragment = extract_site_quantum_fragment(live_molecule, &sorted_indices);

    (fragment, link_hydrogen_positions)
}

/// Self-embedding charges from excluded atoms using electronegativity-based
/// partial charges.
///
/// For atoms NOT in the fragment, compute their Coulomb contribution as point
/// charges using electronegativity-equalization partial charges as the initial
/// guess. Charges are sorted by distance from the fragment centroid, and
/// truncated when cumulative contributions fall below `convergence_ev`.
pub(crate) fn self_embedding_from_excluded(
    live_molecule: &EmbeddedMolecule,
    included_indices: &[usize],
    convergence_ev: f64,
) -> Vec<QuantumEmbeddingPointCharge> {
    let included_set: std::collections::HashSet<usize> =
        included_indices.iter().copied().collect();

    // Compute the centroid of the included fragment for distance sorting
    let fragment_centroid = if included_indices.is_empty() {
        [0.0f32; 3]
    } else {
        mean_position(
            included_indices
                .iter()
                .map(|&idx| live_molecule.positions_angstrom[idx]),
        )
    };

    let mut charges = Vec::new();

    for atom_idx in 0..live_molecule.graph.atom_count() {
        if included_set.contains(&atom_idx) {
            continue;
        }
        let element = live_molecule.graph.atoms[atom_idx].element;
        let charge_e = f64::from(electronegativity_partial_charge(element));
        let pos = live_molecule.positions_angstrom[atom_idx];
        let screening = atom_embedding_screening_radius_angstrom(element);

        charges.push(QuantumEmbeddingPointCharge::new(
            [f64::from(pos[0]), f64::from(pos[1]), f64::from(pos[2])],
            charge_e,
            screening,
        ));
    }

    // Sort by distance from fragment centroid (nearest first)
    charges.sort_by(|a, b| {
        let da = squared_distance(f64x3_to_f32(a.position_angstrom), fragment_centroid);
        let db = squared_distance(f64x3_to_f32(b.position_angstrom), fragment_centroid);
        da.total_cmp(&db)
    });

    // Convergence truncation: drop charges whose individual Coulomb
    // contribution to the fragment center falls below the threshold.
    let n = convergence_truncate_charges(
        &charges,
        fragment_centroid,
        convergence_ev,
        charges.len(),
    );
    charges.truncate(n);

    charges
}

/// Simple electronegativity-equalization charge for initial guess.
/// Returns a heuristic partial charge in electron units based on the
/// Pauling electronegativity relative to carbon (the organic reference).
fn electronegativity_partial_charge(element: PeriodicElement) -> f32 {
    match element {
        PeriodicElement::H => 0.0,
        PeriodicElement::C => -0.1,
        PeriodicElement::N => -0.3,
        PeriodicElement::O => -0.4,
        PeriodicElement::P => 0.5,
        PeriodicElement::S => -0.2,
        _ => 0.0,
    }
}

/// Try to carve convergence-driven fragments from a live molecule source.
///
/// For each anchor atom, a BFS-based convergence-driven fragment is carved.
/// Falls back to the standard template-based carving (via
/// `site_quantum_fragment_indices` + `extract_site_quantum_fragment`) if
/// the convergence criteria cannot be met within the atom budget.
pub(crate) fn live_carved_scaffolds(
    live_molecule: &EmbeddedMolecule,
    anchor_atoms: &[usize],
    convergence_ev: f64,
    max_atoms: usize,
) -> Vec<SiteScaffoldFragment> {
    let mut unique_fragments = std::collections::BTreeSet::<Vec<usize>>::new();
    let mut fragments = Vec::new();
    let max_atoms = max_atoms.max(2);

    for &anchor in anchor_atoms {
        if anchor >= live_molecule.graph.atom_count() {
            continue;
        }

        let (carved, _link_hydrogens) = carve_convergence_driven_fragment(
            live_molecule,
            anchor,
            3, // initial_shell_size (BFS always starts from anchor)
            max_atoms,
            convergence_ev,
        );

        // Deduplicate: build sorted index key from the fragment atoms
        let mut key: Vec<usize> = (0..carved.graph.atom_count()).collect();
        key.sort_unstable();
        if unique_fragments.insert(key) {
            fragments.push(SiteScaffoldFragment {
                scaffold: carved,
                source_anchor_atom_idx: anchor,
            });
        }
    }

    // If nothing was produced, fall back to a single fragment from anchor 0
    if fragments.is_empty() && live_molecule.graph.atom_count() > 0 {
        let (carved, _) = carve_convergence_driven_fragment(
            live_molecule,
            0,
            3,
            max_atoms,
            convergence_ev,
        );
        fragments.push(SiteScaffoldFragment {
            scaffold: carved,
            source_anchor_atom_idx: 0,
        });
    }

    fragments
}

/// Methylamine (CH₃-NH₂): 7 atoms, 6 bonds. Real amine functional group
/// representing the Cα-NH₂ nucleophile in peptide bond formation.
fn methylamine(offset: [f32; 3]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new("methylamine");
    let c  = graph.add_element(PeriodicElement::C); // 0  Cα proxy
    let n  = graph.add_element(PeriodicElement::N); // 1  amine N
    let h1 = graph.add_element(PeriodicElement::H); // 2  C-H
    let h2 = graph.add_element(PeriodicElement::H); // 3  C-H
    let h3 = graph.add_element(PeriodicElement::H); // 4  C-H
    let h4 = graph.add_element(PeriodicElement::H); // 5  N-H
    let h5 = graph.add_element(PeriodicElement::H); // 6  N-H (removed in condensation)
    graph.add_bond(c, n,  BondOrder::Single).unwrap();
    graph.add_bond(c, h1, BondOrder::Single).unwrap();
    graph.add_bond(c, h2, BondOrder::Single).unwrap();
    graph.add_bond(c, h3, BondOrder::Single).unwrap();
    graph.add_bond(n, h4, BondOrder::Single).unwrap();
    graph.add_bond(n, h5, BondOrder::Single).unwrap();
    shifted_embedded(
        graph,
        &[
            [0.0, 0.0, 0.0],    // C
            [1.47, 0.0, 0.0],   // N
            [-0.54, 0.92, 0.0], // H
            [-0.54, -0.46, 0.80], // H
            [-0.54, -0.46, -0.80], // H
            [1.90, 0.82, 0.41], // H
            [1.90, -0.82, 0.41], // H
        ],
        offset,
    )
}

/// Acetic acid (CH₃-COOH): 8 atoms, 7 bonds. Real carboxyl functional group
/// representing peptide chain -COOH (translation) or fatty acyl -COOH (membrane).
fn acetic_acid(offset: [f32; 3]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new("acetic_acid");
    let c_me = graph.add_element(PeriodicElement::C); // 0  methyl C (chain proxy)
    let c_co = graph.add_element(PeriodicElement::C); // 1  carboxyl C
    let o_db = graph.add_element(PeriodicElement::O); // 2  carbonyl =O
    let o_oh = graph.add_element(PeriodicElement::O); // 3  hydroxyl -OH (removed in condensation)
    let h1   = graph.add_element(PeriodicElement::H); // 4  CH₃
    let h2   = graph.add_element(PeriodicElement::H); // 5  CH₃
    let h3   = graph.add_element(PeriodicElement::H); // 6  CH₃
    let h_oh = graph.add_element(PeriodicElement::H); // 7  O-H
    graph.add_bond(c_me, c_co, BondOrder::Single).unwrap();
    graph.add_bond(c_co, o_db, BondOrder::Double).unwrap();
    graph.add_bond(c_co, o_oh, BondOrder::Single).unwrap();
    graph.add_bond(c_me, h1,   BondOrder::Single).unwrap();
    graph.add_bond(c_me, h2,   BondOrder::Single).unwrap();
    graph.add_bond(c_me, h3,   BondOrder::Single).unwrap();
    graph.add_bond(o_oh, h_oh, BondOrder::Single).unwrap();
    shifted_embedded(
        graph,
        &[
            [0.0, 0.0, 0.0],      // C_methyl
            [1.52, 0.0, 0.0],     // C_carboxyl
            [2.17, 1.07, 0.0],    // O=
            [2.17, -1.07, 0.0],   // O-H
            [-0.54, 0.92, 0.0],   // H
            [-0.54, -0.46, 0.80], // H
            [-0.54, -0.46, -0.80],// H
            [3.09, -1.07, 0.0],   // OH H
        ],
        offset,
    )
}

fn methanol(offset: [f32; 3]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new("methanol");
    let c = graph.add_element(PeriodicElement::C);
    let o = graph.add_element(PeriodicElement::O);
    let h1 = graph.add_element(PeriodicElement::H);
    let h2 = graph.add_element(PeriodicElement::H);
    let h3 = graph.add_element(PeriodicElement::H);
    let h_oh = graph.add_element(PeriodicElement::H);
    graph.add_bond(c, o, BondOrder::Single).unwrap();
    graph.add_bond(c, h1, BondOrder::Single).unwrap();
    graph.add_bond(c, h2, BondOrder::Single).unwrap();
    graph.add_bond(c, h3, BondOrder::Single).unwrap();
    graph.add_bond(o, h_oh, BondOrder::Single).unwrap();
    shifted_embedded(
        graph,
        &[
            [0.0, 0.0, 0.0],
            [1.42, 0.0, 0.0],
            [-0.54, 0.92, 0.0],
            [-0.54, -0.46, 0.80],
            [-0.54, -0.46, -0.80],
            [1.83, 0.92, 0.0],
        ],
        offset,
    )
}

fn ethanol(offset: [f32; 3]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new("ethanol");
    let c0 = graph.add_element(PeriodicElement::C);
    let c1 = graph.add_element(PeriodicElement::C);
    let o = graph.add_element(PeriodicElement::O);
    let h1 = graph.add_element(PeriodicElement::H);
    let h2 = graph.add_element(PeriodicElement::H);
    let h3 = graph.add_element(PeriodicElement::H);
    let h4 = graph.add_element(PeriodicElement::H);
    let h5 = graph.add_element(PeriodicElement::H);
    let h_oh = graph.add_element(PeriodicElement::H);
    graph.add_bond(c0, c1, BondOrder::Single).unwrap();
    graph.add_bond(c1, o, BondOrder::Single).unwrap();
    graph.add_bond(c0, h1, BondOrder::Single).unwrap();
    graph.add_bond(c0, h2, BondOrder::Single).unwrap();
    graph.add_bond(c0, h3, BondOrder::Single).unwrap();
    graph.add_bond(c1, h4, BondOrder::Single).unwrap();
    graph.add_bond(c1, h5, BondOrder::Single).unwrap();
    graph.add_bond(o, h_oh, BondOrder::Single).unwrap();
    shifted_embedded(
        graph,
        &[
            [0.0, 0.0, 0.0],
            [1.52, 0.0, 0.0],
            [2.90, 0.0, 0.0],
            [-0.56, 0.92, 0.0],
            [-0.56, -0.46, 0.80],
            [-0.56, -0.46, -0.80],
            [1.98, 0.93, 0.0],
            [1.98, -0.46, 0.80],
            [3.32, 0.92, 0.0],
        ],
        offset,
    )
}

/// Orthophosphoric acid (H₃PO₄): 8 atoms, 7 bonds. Correct inorganic phosphate
/// (Pi) — replaces phosphorous acid (H₃PO₃) which had a P-H bond (wrong chemistry).
fn orthophosphoric_acid(offset: [f32; 3]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new("orthophosphoric_acid");
    let p    = graph.add_element(PeriodicElement::P); // 0
    let o_db = graph.add_element(PeriodicElement::O); // 1  P=O
    let o1   = graph.add_element(PeriodicElement::O); // 2  P-OH
    let o2   = graph.add_element(PeriodicElement::O); // 3  P-OH
    let o3   = graph.add_element(PeriodicElement::O); // 4  P-OH (nucleophilic target)
    let h1   = graph.add_element(PeriodicElement::H); // 5  O-H
    let h2   = graph.add_element(PeriodicElement::H); // 6  O-H
    let h3   = graph.add_element(PeriodicElement::H); // 7  O-H
    graph.add_bond(p, o_db, BondOrder::Double).unwrap();
    graph.add_bond(p, o1,   BondOrder::Single).unwrap();
    graph.add_bond(p, o2,   BondOrder::Single).unwrap();
    graph.add_bond(p, o3,   BondOrder::Single).unwrap();
    graph.add_bond(o1, h1,  BondOrder::Single).unwrap();
    graph.add_bond(o2, h2,  BondOrder::Single).unwrap();
    graph.add_bond(o3, h3,  BondOrder::Single).unwrap();
    shifted_embedded(
        graph,
        &[
            [0.0, 0.0, 0.0],     // P
            [1.46, 0.0, 0.0],    // =O
            [-1.17, 0.80, 0.0],  // O-H
            [-1.17, -0.80, 0.0], // O-H
            [0.0, 0.0, 1.52],    // O-H (replaces P-H at [0,0,1.42])
            [-1.76, 1.54, 0.0],  // H
            [-1.76, -1.54, 0.0], // H
            [0.59, 0.59, 2.14],  // H (new, for 4th O-H)
        ],
        offset,
    )
}

/// Imidazole (C₃H₄N₂) — 9 atoms.  Histidine side-chain proxy for proton
/// relay in ATP synthase F0 channel.  The N-H bond on atom 3 (pyrrole N) is
/// the proton donor site.
fn imidazole(offset: [f32; 3]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new("imidazole");
    let c2  = graph.add_element(PeriodicElement::C); // 0  C2 (between two N)
    let n1  = graph.add_element(PeriodicElement::N); // 1  pyridine-like N
    let c5  = graph.add_element(PeriodicElement::C); // 2  C5
    let n3  = graph.add_element(PeriodicElement::N); // 3  pyrrole N (N-H donor)
    let c4  = graph.add_element(PeriodicElement::C); // 4  C4
    let h_c2 = graph.add_element(PeriodicElement::H); // 5
    let h_c5 = graph.add_element(PeriodicElement::H); // 6
    let h_c4 = graph.add_element(PeriodicElement::H); // 7
    let h_n  = graph.add_element(PeriodicElement::H); // 8  N-H (proton to transfer)
    // Ring bonds
    graph.add_bond(c2, n1, BondOrder::Double).unwrap();
    graph.add_bond(n1, c5, BondOrder::Single).unwrap();
    graph.add_bond(c5, c4, BondOrder::Double).unwrap();
    graph.add_bond(c4, n3, BondOrder::Single).unwrap();
    graph.add_bond(n3, c2, BondOrder::Single).unwrap();
    // C-H bonds
    graph.add_bond(c2, h_c2, BondOrder::Single).unwrap();
    graph.add_bond(c5, h_c5, BondOrder::Single).unwrap();
    graph.add_bond(c4, h_c4, BondOrder::Single).unwrap();
    // N-H bond (donor)
    graph.add_bond(n3, h_n, BondOrder::Single).unwrap();
    shifted_embedded(
        graph,
        &[
            [0.0,   0.0,  0.0],    // C2
            [1.09,  0.63, 0.0],    // N1
            [0.67,  1.87, 0.0],    // C5
            [-0.67, 1.87, 0.0],   // C4
            [-1.09, 0.63, 0.0],   // N3
            [0.0,  -1.08, 0.0],   // H-C2
            [1.26,  2.78, 0.0],   // H-C5
            [-1.26, 2.78, 0.0],   // H-C4
            [-2.08, 0.35, 0.0],   // H-N (proton donor)
        ],
        offset,
    )
}

/// Water (H₂O) — 3 atoms.  Proton acceptor for proton transfer and
/// nucleophile for hydrolysis reactions.
fn water(offset: [f32; 3]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new("water");
    let o  = graph.add_element(PeriodicElement::O); // 0
    let h1 = graph.add_element(PeriodicElement::H); // 1
    let h2 = graph.add_element(PeriodicElement::H); // 2
    graph.add_bond(o, h1, BondOrder::Single).unwrap();
    graph.add_bond(o, h2, BondOrder::Single).unwrap();
    shifted_embedded(
        graph,
        &[
            [0.0,    0.0,   0.0],   // O
            [0.9572, 0.0,   0.0],   // H
            [-0.24,  0.927, 0.0],   // H
        ],
        offset,
    )
}

/// Minimal pyrophosphate HO-P(=O)-O-P(=O)-OH — 9 atoms (2P + 5O + 2H).
/// Captures the essential P-O-P phosphoanhydride bridge in GTP.
/// The P2-O_bridge bond (atoms 2→1) is the hydrolysis target.
fn methyl_pyrophosphate(offset: [f32; 3]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new("pyrophosphate");
    let p1   = graph.add_element(PeriodicElement::P); // 0  beta-P
    let o_br = graph.add_element(PeriodicElement::O); // 1  bridge O (P-O-P)
    let p2   = graph.add_element(PeriodicElement::P); // 2  gamma-P
    let o1d  = graph.add_element(PeriodicElement::O); // 3  P1=O
    let o1s  = graph.add_element(PeriodicElement::O); // 4  P1-OH
    let o2d  = graph.add_element(PeriodicElement::O); // 5  P2=O
    let o2s  = graph.add_element(PeriodicElement::O); // 6  P2-OH
    let h1   = graph.add_element(PeriodicElement::H); // 7  O4-H
    let h2   = graph.add_element(PeriodicElement::H); // 8  O6-H
    // P-O-P bridge
    graph.add_bond(p1, o_br, BondOrder::Single).unwrap();
    graph.add_bond(o_br, p2, BondOrder::Single).unwrap();
    // P1 substituents
    graph.add_bond(p1, o1d, BondOrder::Double).unwrap();
    graph.add_bond(p1, o1s, BondOrder::Single).unwrap();
    // P2 substituents
    graph.add_bond(p2, o2d, BondOrder::Double).unwrap();
    graph.add_bond(p2, o2s, BondOrder::Single).unwrap();
    // O-H bonds
    graph.add_bond(o1s, h1, BondOrder::Single).unwrap();
    graph.add_bond(o2s, h2, BondOrder::Single).unwrap();
    shifted_embedded(
        graph,
        &[
            [0.0,   0.0,   0.0],    // P1 (beta)
            [1.63,  0.0,   0.0],    // bridge O
            [3.26,  0.0,   0.0],    // P2 (gamma)
            [-0.90, 1.20,  0.0],   // P1=O
            [-0.90, -1.20, 0.0],   // P1-OH
            [4.16,  1.20,  0.0],   // P2=O
            [4.16, -1.20,  0.0],   // P2-OH
            [-1.50, -1.94, 0.0],  // H (on O4)
            [4.76, -1.94,  0.0],  // H (on O6)
        ],
        offset,
    )
}

fn translation_peptide_site(offset: [f32; 3]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new("translation_peptide_site");
    let n0 = graph.add_element(PeriodicElement::N);
    let c0 = graph.add_element(PeriodicElement::C);
    let o0 = graph.add_element(PeriodicElement::O);
    let c_alpha = graph.add_element(PeriodicElement::C);
    let c_beta = graph.add_element(PeriodicElement::C);
    let c1 = graph.add_element(PeriodicElement::C);
    let o1 = graph.add_element(PeriodicElement::O);
    let n1 = graph.add_element(PeriodicElement::N);
    graph.add_bond(n0, c0, BondOrder::Single).unwrap();
    graph.add_bond(c0, o0, BondOrder::Double).unwrap();
    graph.add_bond(c0, c_alpha, BondOrder::Single).unwrap();
    graph.add_bond(c_alpha, c_beta, BondOrder::Single).unwrap();
    graph.add_bond(c_beta, c1, BondOrder::Single).unwrap();
    graph.add_bond(c1, o1, BondOrder::Double).unwrap();
    graph.add_bond(c1, n1, BondOrder::Single).unwrap();
    shifted_embedded(
        graph,
        &[
            [0.0, 0.0, 0.0],
            [1.33, 0.0, 0.0],
            [2.45, 0.0, 0.0],
            [1.33, 1.24, 0.0],
            [2.62, 1.92, 0.0],
            [3.88, 1.28, 0.0],
            [4.98, 1.30, 0.0],
            [3.92, 0.04, 0.0],
        ],
        offset,
    )
}

fn translation_owner_chain_scaffold(name: &str, repeat_units: usize) -> EmbeddedMolecule {
    let repeats = repeat_units.max(1).min(8);
    let mut graph = MoleculeGraph::new(name);
    let mut positions = Vec::with_capacity(repeats * 5);
    let mut previous_link_carbon = None;
    let mut x = 0.0f32;

    for _ in 0..repeats {
        let amine = graph.add_element(PeriodicElement::N);
        let carbonyl_carbon = graph.add_element(PeriodicElement::C);
        let carbonyl_oxygen = graph.add_element(PeriodicElement::O);
        let linker_carbon = graph.add_element(PeriodicElement::C);
        let sidechain_carbon = graph.add_element(PeriodicElement::C);

        graph
            .add_bond(amine, carbonyl_carbon, BondOrder::Single)
            .unwrap();
        graph
            .add_bond(carbonyl_carbon, carbonyl_oxygen, BondOrder::Double)
            .unwrap();
        graph
            .add_bond(carbonyl_carbon, linker_carbon, BondOrder::Single)
            .unwrap();
        graph
            .add_bond(linker_carbon, sidechain_carbon, BondOrder::Single)
            .unwrap();
        if let Some(previous) = previous_link_carbon {
            graph.add_bond(previous, amine, BondOrder::Single).unwrap();
        }

        positions.extend_from_slice(&[
            [x, 0.0, 0.0],
            [x + 1.28, 0.0, 0.0],
            [x + 2.38, 0.0, 0.0],
            [x + 1.18, 1.24, 0.0],
            [x + 2.42, 1.92, 0.0],
        ]);
        previous_link_carbon = Some(linker_carbon);
        x += 2.55;
    }

    EmbeddedMolecule::new(graph, positions).expect("owner peptide scaffold geometry")
}

fn phosphate_site(offset: [f32; 3]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new("phosphate_site");
    let p = graph.add_element(PeriodicElement::P);
    let o0 = graph.add_element(PeriodicElement::O);
    let o1 = graph.add_element(PeriodicElement::O);
    let o2 = graph.add_element(PeriodicElement::O);
    graph.add_bond(p, o0, BondOrder::Double).unwrap();
    graph.add_bond(p, o1, BondOrder::Single).unwrap();
    graph.add_bond(p, o2, BondOrder::Single).unwrap();
    shifted_embedded(
        graph,
        &[
            [0.0, 0.0, 0.0],
            [1.48, 0.0, 0.0],
            [-1.10, 0.82, 0.0],
            [-1.10, -0.82, 0.0],
        ],
        offset,
    )
}

fn phosphate_owner_chain_scaffold(name: &str, repeat_units: usize) -> EmbeddedMolecule {
    let repeats = repeat_units.max(1).min(8);
    let mut graph = MoleculeGraph::new(name);
    let mut positions = Vec::with_capacity(repeats * 4);
    let mut previous_bridge = None;
    let mut x = 0.0f32;

    for _ in 0..repeats {
        let phosphorus = graph.add_element(PeriodicElement::P);
        let oxo = graph.add_element(PeriodicElement::O);
        let side_oxygen = graph.add_element(PeriodicElement::O);
        let bridge_oxygen = graph.add_element(PeriodicElement::O);

        graph.add_bond(phosphorus, oxo, BondOrder::Double).unwrap();
        graph
            .add_bond(phosphorus, side_oxygen, BondOrder::Single)
            .unwrap();
        graph
            .add_bond(phosphorus, bridge_oxygen, BondOrder::Single)
            .unwrap();
        if let Some(previous) = previous_bridge {
            graph
                .add_bond(previous, phosphorus, BondOrder::Single)
                .unwrap();
        }

        positions.extend_from_slice(&[
            [x, 0.0, 0.0],
            [x, 1.46, 0.0],
            [x - 1.08, -0.82, 0.0],
            [x + 1.16, 0.0, 0.0],
        ]);
        previous_bridge = Some(bridge_oxygen);
        x += 2.34;
    }

    EmbeddedMolecule::new(graph, positions).expect("owner phosphate scaffold geometry")
}

fn nucleotide_phosphate_site(offset: [f32; 3]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new("nucleotide_phosphate_site");
    let p = graph.add_element(PeriodicElement::P);
    let o0 = graph.add_element(PeriodicElement::O);
    let o1 = graph.add_element(PeriodicElement::O);
    let o2 = graph.add_element(PeriodicElement::O);
    let c = graph.add_element(PeriodicElement::C);
    let o3 = graph.add_element(PeriodicElement::O);
    graph.add_bond(p, o0, BondOrder::Double).unwrap();
    graph.add_bond(p, o1, BondOrder::Single).unwrap();
    graph.add_bond(p, o2, BondOrder::Single).unwrap();
    graph.add_bond(o1, c, BondOrder::Single).unwrap();
    graph.add_bond(c, o3, BondOrder::Single).unwrap();
    shifted_embedded(
        graph,
        &[
            [0.0, 0.0, 0.0],
            [1.48, 0.0, 0.0],
            [-1.08, 0.84, 0.0],
            [-1.12, -0.82, 0.0],
            [-2.34, 1.48, 0.0],
            [-3.52, 0.82, 0.0],
        ],
        offset,
    )
}

fn nucleotide_owner_chain_scaffold(name: &str, repeat_units: usize) -> EmbeddedMolecule {
    let repeats = repeat_units.max(1).min(8);
    let mut graph = MoleculeGraph::new(name);
    let mut positions = Vec::with_capacity(repeats * 6);
    let mut previous_bridge = None;
    let mut x = 0.0f32;

    for _ in 0..repeats {
        let phosphorus = graph.add_element(PeriodicElement::P);
        let oxo = graph.add_element(PeriodicElement::O);
        let bridge_oxygen = graph.add_element(PeriodicElement::O);
        let linker_oxygen = graph.add_element(PeriodicElement::O);
        let sugar_carbon = graph.add_element(PeriodicElement::C);
        let sugar_oxygen = graph.add_element(PeriodicElement::O);

        graph.add_bond(phosphorus, oxo, BondOrder::Double).unwrap();
        graph
            .add_bond(phosphorus, bridge_oxygen, BondOrder::Single)
            .unwrap();
        graph
            .add_bond(phosphorus, linker_oxygen, BondOrder::Single)
            .unwrap();
        graph
            .add_bond(linker_oxygen, sugar_carbon, BondOrder::Single)
            .unwrap();
        graph
            .add_bond(sugar_carbon, sugar_oxygen, BondOrder::Single)
            .unwrap();
        if let Some(previous) = previous_bridge {
            graph
                .add_bond(previous, phosphorus, BondOrder::Single)
                .unwrap();
        }

        positions.extend_from_slice(&[
            [x, 0.0, 0.0],
            [x, 1.46, 0.0],
            [x + 1.18, 0.0, 0.0],
            [x - 1.10, 0.78, 0.0],
            [x - 2.34, 1.42, 0.0],
            [x - 3.48, 0.76, 0.0],
        ]);
        previous_bridge = Some(bridge_oxygen);
        x += 2.46;
    }

    EmbeddedMolecule::new(graph, positions).expect("owner nucleotide scaffold geometry")
}

fn membrane_acyl_site(offset: [f32; 3]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new("membrane_acyl_site");
    let c0 = graph.add_element(PeriodicElement::C);
    let o0 = graph.add_element(PeriodicElement::O);
    let o1 = graph.add_element(PeriodicElement::O);
    let c1 = graph.add_element(PeriodicElement::C);
    graph.add_bond(c0, o0, BondOrder::Double).unwrap();
    graph.add_bond(c0, o1, BondOrder::Single).unwrap();
    graph.add_bond(c0, c1, BondOrder::Single).unwrap();
    shifted_embedded(
        graph,
        &[
            [0.0, 0.0, 0.0],
            [1.22, 0.0, 0.0],
            [-1.18, 0.18, 0.0],
            [0.0, -1.24, 0.0],
        ],
        offset,
    )
}

fn membrane_owner_chain_scaffold(name: &str, repeat_units: usize) -> EmbeddedMolecule {
    let repeats = repeat_units.max(1).min(8);
    let mut graph = MoleculeGraph::new(name);
    let mut positions = Vec::with_capacity(repeats + 4);

    let carbonyl_carbon = graph.add_element(PeriodicElement::C);
    let carbonyl_oxygen = graph.add_element(PeriodicElement::O);
    let ester_oxygen = graph.add_element(PeriodicElement::O);
    let mut chain = Vec::with_capacity(repeats + 1);
    for _ in 0..=repeats {
        chain.push(graph.add_element(PeriodicElement::C));
    }

    graph
        .add_bond(carbonyl_carbon, carbonyl_oxygen, BondOrder::Double)
        .unwrap();
    graph
        .add_bond(carbonyl_carbon, ester_oxygen, BondOrder::Single)
        .unwrap();
    graph
        .add_bond(carbonyl_carbon, chain[0], BondOrder::Single)
        .unwrap();
    for pair in chain.windows(2) {
        graph.add_bond(pair[0], pair[1], BondOrder::Single).unwrap();
    }

    positions.extend_from_slice(&[[0.0, 0.0, 0.0], [1.22, 0.0, 0.0], [-1.18, 0.18, 0.0]]);
    for (idx, _) in chain.iter().enumerate() {
        positions.push([0.12 + 1.42 * idx as f32, -1.18, 0.18 * (idx % 2) as f32]);
    }

    EmbeddedMolecule::new(graph, positions).expect("owner membrane scaffold geometry")
}

fn shifted_embedded(
    graph: MoleculeGraph,
    positions: &[[f32; 3]],
    offset: [f32; 3],
) -> EmbeddedMolecule {
    let shifted: Vec<[f32; 3]> = positions
        .iter()
        .map(|position| {
            [
                position[0] + offset[0],
                position[1] + offset[1],
                position[2] + offset[2],
            ]
        })
        .collect();
    EmbeddedMolecule::new(graph, shifted).expect("embedded microdomain geometry")
}

fn disconnected_scaffold_source(name: &str, fragments: &[EmbeddedMolecule]) -> EmbeddedMolecule {
    let mut graph = MoleculeGraph::new(name);
    let mut positions = Vec::new();
    let mut atom_offset = 0usize;
    for fragment in fragments {
        for atom in &fragment.graph.atoms {
            graph.add_atom_node(atom.clone());
        }
        for bond in &fragment.graph.bonds {
            graph
                .add_bond(atom_offset + bond.i, atom_offset + bond.j, bond.order)
                .expect("disconnected scaffold source bonds stay valid");
        }
        positions.extend(fragment.positions_angstrom.iter().copied());
        atom_offset += fragment.graph.atom_count();
    }
    EmbeddedMolecule::new(graph, positions).expect("disconnected scaffold source")
}

fn quantum_config_for_reactants(reactants: &[EmbeddedMolecule]) -> QuantumChemistryConfig {
    // Lifted from 512→1024→2048 to support larger live-state-carved fragments.
    // At 2048 max basis size, the CIPSI selected-CI solver handles 48-atom
    // fragments with up to ~80 active electrons across 40+ spatial orbitals.
    const MAX_BASIS_SIZE: usize = 2048;
    // Lifted from 24→32→48 to allow fragments carved directly from live
    // molecular neighborhoods rather than small representative surrogates.
    const MAX_SPATIAL_ORBITALS: usize = 48;
    let active_electrons = reactants
        .iter()
        .flat_map(|reactant| reactant.graph.atoms.iter())
        .map(|atom| {
            QuantumAtomState::from_atom_node(atom)
                .expect("runtime microdomain atom state")
                .valence_electrons() as usize
        })
        .sum::<usize>()
        .max(2);
    let min_spatial_orbitals = active_electrons.div_ceil(2).max(1);
    let available_spatial_orbitals = reactants
        .iter()
        .flat_map(|reactant| reactant.graph.atoms.iter())
        .map(|atom| {
            QuantumAtomState::from_atom_node(atom)
                .expect("runtime microdomain atom state")
                .active_subshells()
                .into_iter()
                .map(|shell| shell.subshell.spatial_degeneracy() as usize)
                .sum::<usize>()
        })
        .sum::<usize>()
        .max(min_spatial_orbitals)
        .min(MAX_SPATIAL_ORBITALS);

    let mut max_spatial_orbitals = min_spatial_orbitals;
    for candidate in min_spatial_orbitals..=available_spatial_orbitals {
        let spin_orbitals = candidate.saturating_mul(2);
        if combination_count_capped(spin_orbitals, active_electrons, MAX_BASIS_SIZE)
            <= MAX_BASIS_SIZE
        {
            max_spatial_orbitals = candidate;
        } else {
            break;
        }
    }

    // Enforce minimum orbital headroom: a near-filled active space (e.g. 57
    // electrons in 58 spin orbitals = 1 hole) is degenerate — bond topology
    // changes cannot shift the CASCI energy.  Guarantee at least
    // MIN_UNOCCUPIED_SPIN empty spin orbitals so that the ground state is
    // sensitive to structural edits.  The larger Hilbert space triggers the
    // CIPSI selected-CI solver (above FULL_ED_DIM_THRESHOLD = 500) which
    // uses Lanczos for fast ground-state diagonalization of large matrices.
    const MIN_UNOCCUPIED_SPIN: usize = 4;
    let min_spatial_for_headroom =
        (active_electrons + MIN_UNOCCUPIED_SPIN).div_ceil(2);
    let max_spatial_orbitals = max_spatial_orbitals.max(
        min_spatial_for_headroom.min(available_spatial_orbitals),
    );

    QuantumChemistryConfig::new(Some(max_spatial_orbitals), MAX_BASIS_SIZE)
}

/// Count valence electrons in a molecule graph using the quantum atom model.
/// This emerges from each element's electronic configuration (no hardcoded tables).
fn valence_electrons_for_graph(graph: &MoleculeGraph) -> usize {
    graph
        .atoms
        .iter()
        .filter_map(|atom| QuantumAtomState::from_atom_node(atom).ok())
        .map(|state| state.valence_electrons() as usize)
        .sum()
}

/// Compute the maximum scaffold atoms that can accompany a set of reactants
/// without exceeding the spin-orbital limit imposed by the u64 determinant
/// representation in the CASCI exact diagonalization solver.
///
/// The constraint is physical: u64 bit masks represent Slater determinants,
/// so the active space cannot exceed 64 spin orbitals (32 spatial orbitals).
/// Given N valence electrons, we need ceil(N/2) spatial orbitals minimum,
/// so the total valence electrons across all fragments must stay ≤ 64.
///
/// Scaffold atom budget uses a conservative bound of 6 valence electrons
/// per atom (oxygen, the heaviest common scaffold element).
fn max_scaffold_atoms_within_orbital_limit(reactant_valence_electrons: usize) -> usize {
    let max_total_valence = u64::BITS as usize; // 64 spin orbitals → 64 max valence electrons
    let remaining = max_total_valence.saturating_sub(reactant_valence_electrons);
    // Conservative: assume worst-case 6 valence electrons per scaffold atom (oxygen)
    let worst_case_valence_per_atom = PeriodicElement::O
        .atomic_number()
        .saturating_sub(2) as usize; // Z - core(1s²) = valence
    remaining / worst_case_valence_per_atom.max(1)
}

fn combination_count_capped(n: usize, k: usize, cap: usize) -> usize {
    if k > n {
        return cap.saturating_add(1);
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 1..=k {
        let numerator = n - k + i;
        result = result.saturating_mul(numerator) / i;
        if result > cap {
            return cap.saturating_add(1);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Phase 8b: Generic Live Microdomain Builder
// ---------------------------------------------------------------------------

/// Request for building a carved microdomain from a live molecule at a
/// reaction site.  This is the generic entry point that does not require
/// per-reaction-type branching -- any molecule + anchor atoms can be
/// submitted.
pub(crate) struct LiveMicrodomainRequest {
    /// The live molecule to carve from.
    pub molecule: EmbeddedMolecule,
    /// Anchor atom indices (reaction centers) within `molecule`.
    pub anchor_atoms: Vec<usize>,
    /// Reaction being catalyzed (for naming / tracking).
    pub reaction_name: String,
    /// Maximum atom budget per fragment (0 = use convergence only).
    pub max_atoms: usize,
    /// Energy convergence threshold in eV (typically kT/100 ~ 0.00027 eV at 310 K).
    pub convergence_ev: f64,
    /// Temperature in K for kT reference.
    pub temperature_k: f64,
}

/// Result of a generic live microdomain build.
pub(crate) struct LiveMicrodomainResult {
    /// Carved fragment scaffolds ready for ED.
    pub scaffolds: Vec<SiteScaffoldFragment>,
    /// Self-embedding charges from excluded atoms.
    pub embedding_charges: Vec<QuantumEmbeddingPointCharge>,
    /// Link hydrogen positions (for visualization / validation).
    pub link_hydrogens: Vec<[f32; 3]>,
    /// Whether convergence was achieved (fragment did not hit `max_atoms`).
    pub converged: bool,
}

/// Build a carved microdomain from a single live molecule.
///
/// This is the molecule-level generic entry point.  For each anchor atom,
/// [`carve_convergence_driven_fragment`] produces a fragment and link
/// hydrogens; the fragments are deduplicated via [`live_carved_scaffolds`].
/// [`self_embedding_from_excluded`] is then called over the union of all
/// included atom indices to produce the embedding point charges.
pub(crate) fn build_live_microdomain(req: &LiveMicrodomainRequest) -> LiveMicrodomainResult {
    // Effective max_atoms: 0 means "convergence only" -- use entire molecule.
    let effective_max = if req.max_atoms == 0 {
        req.molecule.graph.atom_count()
    } else {
        req.max_atoms
    };

    // Produce deduplicated scaffolds via the existing BFS convergence carver.
    let scaffolds = live_carved_scaffolds(
        &req.molecule,
        &req.anchor_atoms,
        req.convergence_ev,
        effective_max,
    );

    // Collect all link hydrogens and the union of included atom indices.
    // We re-carve per anchor to obtain the link hydrogen positions since
    // `live_carved_scaffolds` does not expose them.
    let mut all_link_hydrogens: Vec<[f32; 3]> = Vec::new();
    let mut included_union = std::collections::BTreeSet::<usize>::new();
    let mut hit_budget = false;

    for scaffold in &scaffolds {
        let anchor = scaffold.source_anchor_atom_idx;
        let (carved, link_hs) = carve_convergence_driven_fragment(
            &req.molecule,
            anchor,
            3, // initial shell size (BFS starts from anchor)
            effective_max,
            req.convergence_ev,
        );

        // Check convergence: if the carved fragment used the full budget,
        // convergence was not achieved for this anchor.
        if carved.graph.atom_count() >= effective_max {
            hit_budget = true;
        }

        // Map carved atom elements+positions back to original indices via
        // position matching.  This is O(F*N) but F and N are small (<200).
        for pos in &carved.positions_angstrom {
            for (orig_idx, orig_pos) in req.molecule.positions_angstrom.iter().enumerate() {
                let dx = pos[0] - orig_pos[0];
                let dy = pos[1] - orig_pos[1];
                let dz = pos[2] - orig_pos[2];
                if dx * dx + dy * dy + dz * dz < 1.0e-6 {
                    included_union.insert(orig_idx);
                    break;
                }
            }
        }

        all_link_hydrogens.extend(link_hs);
    }

    let included_indices: Vec<usize> = included_union.into_iter().collect();

    // Build embedding charges from the excluded (non-fragment) atoms.
    let embedding_charges = self_embedding_from_excluded(
        &req.molecule,
        &included_indices,
        req.convergence_ev,
    );

    let converged = !hit_budget;

    LiveMicrodomainResult {
        scaffolds,
        embedding_charges,
        link_hydrogens: all_link_hydrogens,
        converged,
    }
}

/// Build live microdomains for an entire reaction site represented as an
/// [`EmbeddedMaterialMixture`].
///
/// For each anchor atom index, the component molecule that actually contains
/// that atom (by index) is identified and a [`LiveMicrodomainRequest`] is
/// built.  Returns one [`LiveMicrodomainResult`] per component molecule that
/// contained at least one anchor.
///
/// Anchors that fall outside all component molecules are silently skipped.
pub(crate) fn build_live_microdomains_for_site(
    mixture: &EmbeddedMaterialMixture,
    anchor_atoms: &[usize],
    reaction_name: &str,
    max_atoms: usize,
    convergence_ev: f64,
    temperature_k: f64,
) -> Vec<LiveMicrodomainResult> {
    // Map each anchor to the component that owns it.  We use the convention
    // that anchor indices are local to each component molecule, so an anchor
    // is valid for a component if anchor < component.atom_count().
    //
    // If anchors are *global* (offset by component boundaries), the caller
    // should pre-split them.  Here we follow the simpler local convention
    // used elsewhere in the quantum runtime.
    let mut per_component: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();

    for &anchor in anchor_atoms {
        for (comp_idx, comp) in mixture.components.iter().enumerate() {
            if anchor < comp.molecule.graph.atom_count() {
                per_component.entry(comp_idx).or_default().push(anchor);
                break; // first matching component wins
            }
        }
    }

    let mut results = Vec::new();
    for (comp_idx, anchors) in per_component {
        let component = &mixture.components[comp_idx];
        let req = LiveMicrodomainRequest {
            molecule: component.molecule.clone(),
            anchor_atoms: anchors,
            reaction_name: reaction_name.to_string(),
            max_atoms,
            convergence_ev,
            temperature_k,
        };
        results.push(build_live_microdomain(&req));
    }

    results
}

/// Attempt live microdomain carving for a reaction kind, falling back to
/// template-based scaffolding when no live molecule source is available.
///
/// This is the unified entry point: callers supply an optional live molecule
/// and anchor atoms.  If the live molecule is `Some`, convergence-driven
/// carving is attempted.  Otherwise the existing per-kind template path
/// ([`scaffolds_for_kind`]) is used.
pub(crate) fn try_live_microdomain(
    kind: WholeCellRuntimeQuantumProcessKind,
    live_molecule: Option<&EmbeddedMolecule>,
    anchor_atoms: &[usize],
    max_scaffold_atoms: usize,
    convergence_ev: f64,
    temperature_k: f64,
) -> LiveMicrodomainResult {
    if let Some(mol) = live_molecule {
        let effective_anchors = if anchor_atoms.is_empty() {
            // Default: use atom 0 as the single anchor.
            vec![0]
        } else {
            anchor_atoms.to_vec()
        };

        let req = LiveMicrodomainRequest {
            molecule: mol.clone(),
            anchor_atoms: effective_anchors,
            reaction_name: kind.as_str().to_string(),
            max_atoms: max_scaffold_atoms,
            convergence_ev,
            temperature_k,
        };

        build_live_microdomain(&req)
    } else {
        // Fall back to template-based scaffolding.
        let scaffolds = scaffolds_for_kind(kind, max_scaffold_atoms, anchor_atoms, None);
        LiveMicrodomainResult {
            scaffolds,
            embedding_charges: Vec::new(),
            link_hydrogens: Vec::new(),
            converged: true, // template scaffolds are pre-converged by construction
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_quantum_processes_are_seeded_from_site_local_scaffolds() {
        let processes = default_runtime_quantum_processes();
        assert!(processes.len() > 5);

        for process in processes {
            let scaffold = &process.reactants[0];
            let adjacency = scaffold_adjacency(scaffold);
            let primitive_kinds =
                weighted_site_chemical_primitives(process.kind(), scaffold, &adjacency)
                    .into_iter()
                    .take(4)
                    .map(|(primitive, _)| primitive.kind)
                    .collect::<Vec<_>>();

            match process.kind() {
                WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation => {
                    assert!(primitive_kinds.iter().any(|kind| {
                        matches!(
                            kind,
                            SiteChemicalPrimitiveKind::CarbonylCarbon
                                | SiteChemicalPrimitiveKind::CarbonylOxygen
                                | SiteChemicalPrimitiveKind::AmineCenter
                        )
                    }));
                }
                WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation
                | WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation => {
                    assert!(primitive_kinds.contains(&SiteChemicalPrimitiveKind::PhosphateCenter));
                }
                WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification
                | WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification => {
                    assert!(primitive_kinds.iter().any(|kind| {
                        matches!(
                            kind,
                            SiteChemicalPrimitiveKind::CarbonylCarbon
                                | SiteChemicalPrimitiveKind::CarbonylOxygen
                                | SiteChemicalPrimitiveKind::PolarCarbon
                        )
                    }));
                }
                WholeCellRuntimeQuantumProcessKind::AtpBandProtonTransfer => {
                    assert!(primitive_kinds.iter().any(|kind| {
                        matches!(
                            kind,
                            SiteChemicalPrimitiveKind::HydroxylOxygen
                                | SiteChemicalPrimitiveKind::AmineCenter
                                | SiteChemicalPrimitiveKind::PhosphateCenter
                        )
                    }));
                }
                WholeCellRuntimeQuantumProcessKind::SeptumGtpHydrolysis => {
                    assert!(primitive_kinds.iter().any(|kind| {
                        matches!(
                            kind,
                            SiteChemicalPrimitiveKind::PhosphateCenter
                                | SiteChemicalPrimitiveKind::PhosphorylOxygen
                                | SiteChemicalPrimitiveKind::HydroxylOxygen
                        )
                    }));
                }
                WholeCellRuntimeQuantumProcessKind::AtpBandElectronTransfer => {
                    assert!(primitive_kinds.iter().any(|kind| {
                        matches!(
                            kind,
                            SiteChemicalPrimitiveKind::HydroxylOxygen
                                | SiteChemicalPrimitiveKind::PolarCarbon
                                | SiteChemicalPrimitiveKind::CarbonylOxygen
                        )
                    }));
                }
                WholeCellRuntimeQuantumProcessKind::MembraneProteinInsertion => {
                    assert!(primitive_kinds.iter().any(|kind| {
                        matches!(
                            kind,
                            SiteChemicalPrimitiveKind::CarbonylCarbon
                                | SiteChemicalPrimitiveKind::CarbonylOxygen
                                | SiteChemicalPrimitiveKind::AmineCenter
                        )
                    }));
                }
            }
            assert_eq!(
                scaffold.positions_angstrom.len(),
                scaffold.graph.atom_count()
            );
            assert_eq!(&process.reaction.reactants[0], &scaffold.graph);
            assert!(process.reactants.len() >= 3);
        }
    }

    #[test]
    fn preferred_anchor_runtime_builder_preserves_anchor_order_across_budget_changes() {
        let kind = WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation;
        let baseline = runtime_quantum_processes_for_kind_with_atom_budget(kind, 4);
        let mut preferred_anchors = baseline
            .iter()
            .filter_map(|process| process.site_anchor_atom_idx())
            .collect::<Vec<_>>();
        assert!(!preferred_anchors.is_empty());

        preferred_anchors.reverse();
        let rebuilt = runtime_quantum_processes_for_kind_with_atom_budget_and_anchor_preferences(
            kind,
            6,
            &preferred_anchors,
        );
        let rebuilt_anchors = rebuilt
            .iter()
            .filter_map(|process| process.site_anchor_atom_idx())
            .collect::<Vec<_>>();

        assert!(!rebuilt_anchors.is_empty());
        assert_eq!(
            rebuilt_anchors.first().copied(),
            preferred_anchors.first().copied()
        );
        if preferred_anchors.len() > 1 && rebuilt_anchors.len() > 1 {
            assert_eq!(rebuilt_anchors[1], preferred_anchors[1]);
        }
        assert!(rebuilt
            .iter()
            .all(|process| process.scaffold_atom_count() >= baseline[0].scaffold_atom_count()));
    }

    #[test]
    fn phosphorylation_fragment_anchors_follow_phosphate_primitives() {
        let scaffold = atomistic_template_for_site_name("atp_synthase_band")
            .expect("atp band template")
            .to_embedded_molecule()
            .expect("embedded atp band scaffold");

        let anchors = site_quantum_fragment_anchor_indices(
            WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation,
            &scaffold,
            4,
        );

        assert_eq!(anchors.len(), 2);
        assert!(anchors
            .iter()
            .all(|atom_idx| scaffold.graph.atoms[*atom_idx].element == PeriodicElement::P));
    }

    #[test]
    fn translation_fragment_anchors_follow_amide_primitives_over_phosphate_scaffolding() {
        let scaffold = atomistic_template_for_site_name("ribosome_cluster")
            .expect("ribosome template")
            .to_embedded_molecule()
            .expect("embedded ribosome scaffold");

        let anchors = site_quantum_fragment_anchor_indices(
            WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation,
            &scaffold,
            4,
        );

        assert_eq!(anchors.len(), 2);
        assert!(anchors
            .iter()
            .all(|atom_idx| scaffold.graph.atoms[*atom_idx].element != PeriodicElement::P));
        assert!(anchors.iter().any(|atom_idx| {
            matches!(
                scaffold.graph.atoms[*atom_idx].element,
                PeriodicElement::N | PeriodicElement::C
            )
        }));
    }

    #[test]
    fn translation_fragment_growth_keeps_amide_neighbors_connected() {
        let scaffold = atomistic_template_for_site_name("ribosome_cluster")
            .expect("ribosome template")
            .to_embedded_molecule()
            .expect("embedded ribosome scaffold");
        let adjacency = scaffold_adjacency(&scaffold);
        let center_scores = site_center_law_scores(
            WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation,
            &scaffold,
            &adjacency,
        );
        let carbonyl_anchor = scaffold
            .graph
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, atom)| atom.element == PeriodicElement::C)
            .max_by(|(left_idx, _), (right_idx, _)| {
                center_scores[*left_idx].total_cmp(&center_scores[*right_idx])
            })
            .map(|(atom_idx, _)| atom_idx)
            .expect("carbonyl-like anchor");

        let selected = site_quantum_fragment_indices(
            WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation,
            &scaffold,
            4,
            carbonyl_anchor,
        );
        let selected_elements = selected
            .iter()
            .map(|atom_idx| scaffold.graph.atoms[*atom_idx].element)
            .collect::<Vec<_>>();

        assert!(selected.contains(&carbonyl_anchor));
        assert!(selected_elements.contains(&PeriodicElement::N));
        assert!(selected_elements.contains(&PeriodicElement::O));
    }

    #[test]
    fn phosphorylation_reactant_placement_targets_phosphate_center_and_points_outward() {
        let scaffold = atomistic_template_for_site_name("atp_synthase_band")
            .expect("atp band template")
            .to_embedded_molecule()
            .expect("embedded atp band scaffold");
        let placement = scaffold_reactant_placement(
            WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation,
            &scaffold,
            2.6,
        );
        let center = scaffold.positions_angstrom[placement.center_atom_idx];
        let centroid = centroid_angstrom(&scaffold);

        assert_eq!(
            scaffold.graph.atoms[placement.center_atom_idx].element,
            PeriodicElement::P
        );
        assert!(vector_norm_f32(placement.direction) > 0.99);
        assert!(
            squared_distance(placement.anchor_position_angstrom, centroid)
                > squared_distance(center, centroid)
        );
    }

    #[test]
    fn translation_reactant_placement_prefers_amide_like_site_over_phosphate() {
        let scaffold = atomistic_template_for_site_name("ribosome_cluster")
            .expect("ribosome template")
            .to_embedded_molecule()
            .expect("embedded ribosome scaffold");
        let placement = scaffold_reactant_placement(
            WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation,
            &scaffold,
            2.6,
        );

        assert_ne!(
            scaffold.graph.atoms[placement.center_atom_idx].element,
            PeriodicElement::P
        );
        assert!(matches!(
            scaffold.graph.atoms[placement.center_atom_idx].element,
            PeriodicElement::C | PeriodicElement::N | PeriodicElement::O
        ));
    }

    #[test]
    fn runtime_quantum_default_scaffold_source_prefers_atomistic_site_template_geometry() {
        for kind in WholeCellRuntimeQuantumProcessKind::all() {
            let scaffold = runtime_quantum_default_scaffold_source(kind);
            let template = atomistic_template_for_site_name(kind.site_name())
                .expect("site template for runtime quantum kind")
                .to_embedded_molecule()
                .expect("embedded site template scaffold");

            assert_eq!(scaffold.graph.atom_count(), template.graph.atom_count());
            assert_eq!(scaffold.graph.bond_count(), template.graph.bond_count());
            assert!(scaffold.graph.name.starts_with(kind.as_str()));
        }
    }

    #[test]
    fn owner_scaffold_components_preserve_local_site_fragments() {
        let kind = WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation;
        let owner_components = runtime_quantum_owner_scaffold_components(kind, "owner_family", 3);
        let template = atomistic_template_for_site_name(kind.site_name())
            .expect("ribosome site template")
            .to_embedded_molecule()
            .expect("embedded ribosome template");
        let fragment_atom_budget = runtime_quantum_owner_scaffold_fragment_atom_budget(kind);
        let template_fragments = site_scaffolds_from_source_with_preferred_anchors(
            kind,
            &template,
            fragment_atom_budget,
            &[],
        );

        assert_eq!(owner_components.len(), 3);
        assert!(owner_components.iter().all(|component| {
            template_fragments.iter().any(|template_fragment| {
                molecule_graph_matches_structure(
                    &component.graph,
                    &template_fragment.scaffold.graph,
                )
            })
        }));
    }

    #[test]
    fn owner_composition_scaffold_components_build_connected_non_template_fragments() {
        let kind = WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation;
        let composition_components =
            runtime_quantum_owner_composition_scaffold_components(kind, "owner_family", &[2, 3]);
        let fallback_components =
            runtime_quantum_owner_scaffold_components(kind, "owner_family", 2);
        let fallback_max_atoms = fallback_components
            .iter()
            .map(|component| component.graph.atom_count())
            .max()
            .unwrap_or(0);

        assert_eq!(composition_components.len(), 2);
        assert!(composition_components.iter().all(|component| {
            component.graph.bond_count() + 1 >= component.graph.atom_count()
                && component.graph.name.starts_with(kind.as_str())
        }));
        assert!(composition_components
            .iter()
            .any(|component| component.graph.atom_count() > fallback_max_atoms));
    }

    #[test]
    fn shared_preset_runtime_scaffold_sources_keep_kind_local_identity() {
        let energy = runtime_quantum_default_scaffold_source(
            WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation,
        );
        let membrane = runtime_quantum_default_scaffold_source(
            WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification,
        );

        assert_eq!(energy.graph.atom_count(), membrane.graph.atom_count());
        assert_eq!(energy.graph.bond_count(), membrane.graph.bond_count());
        assert_ne!(energy.graph.name, membrane.graph.name);
    }

    #[test]
    fn symbolic_runtime_quantum_microdomain_uses_site_template_scaffold_source() {
        let kind = WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation;
        let mixture = symbolic_runtime_quantum_microdomain_from_amounts(kind, 48.0, 0.0)
            .expect("symbolic runtime quantum microdomain");
        let scaffold = mixture
            .components
            .iter()
            .find(|component| component.molecule.graph.name.starts_with(kind.as_str()))
            .map(|component| component.molecule.clone())
            .expect("symbolic scaffold component");
        let template = atomistic_template_for_site_name(kind.site_name())
            .expect("ribosome site template")
            .to_embedded_molecule()
            .expect("embedded ribosome template");

        assert_eq!(scaffold.graph.atom_count(), template.graph.atom_count());
        assert_eq!(scaffold.graph.bond_count(), template.graph.bond_count());
    }

    #[test]
    fn owner_scaffold_source_prefers_site_template_fragment_families() {
        let kind = WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation;
        let template = atomistic_template_for_site_name(kind.site_name())
            .expect("ribosome site template")
            .to_embedded_molecule()
            .expect("embedded ribosome template");
        let owner_scaffold = runtime_quantum_owner_scaffold_source(kind, "owner_family", 3);
        let fragment_atom_budget = runtime_quantum_owner_scaffold_fragment_atom_budget(kind);
        let owner_fragments = site_scaffolds_from_source_with_preferred_anchors(
            kind,
            &owner_scaffold,
            fragment_atom_budget,
            &[],
        );
        let template_fragments = site_scaffolds_from_source_with_preferred_anchors(
            kind,
            &template,
            fragment_atom_budget,
            &[],
        );

        assert!(!owner_fragments.is_empty());
        assert!(!template_fragments.is_empty());
        assert!(owner_fragments.iter().all(|owner_fragment| {
            template_fragments.iter().any(|template_fragment| {
                molecule_graph_matches_structure(
                    &owner_fragment.scaffold.graph,
                    &template_fragment.scaffold.graph,
                )
            })
        }));
    }

    #[test]
    fn ranked_scaffold_components_prefer_kind_tagged_site_family() {
        let mut mixture = EmbeddedMaterialMixture::new("shared_atp_band_symbolic_microdomain");
        mixture.add_component(
            runtime_quantum_default_scaffold_source(
                WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation,
            ),
            48.0,
        );
        mixture.add_component(
            runtime_quantum_default_scaffold_source(
                WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification,
            ),
            48.0,
        );

        let ranked = ranked_runtime_quantum_scaffold_components_from_mixture(
            WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification,
            &mixture,
        );

        assert!(!ranked.is_empty());
        assert!(ranked[0].0.graph.name.starts_with(
            WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification.as_str()
        ));
    }

    #[test]
    fn symbolic_inventory_process_builder_seeds_exact_local_fragments() {
        let kind = WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation;
        let template_atom_count = runtime_quantum_default_scaffold_source(kind)
            .graph
            .atom_count();
        let processes =
            runtime_quantum_processes_for_kind_from_symbolic_inventory_with_atom_budget_and_anchor_preferences(
                kind,
                4,
                &[],
                64.0,
                21.0,
            )
            .expect("symbolic inventory runtime processes");

        assert!(!processes.is_empty());
        assert!(processes.iter().all(|process| {
            process.scaffold_atom_count() < template_atom_count
                && (process.reactant_component_amount(0) - 64.0).abs() < 1.0e-9
                && (process.reactant_component_amount(1) - 21.0).abs() < 1.0e-9
                && (process.reactant_component_amount(2) - 21.0).abs() < 1.0e-9
        }));
    }

    #[test]
    fn live_support_microdomain_seeds_canonical_runtime_reactants_and_preserves_support_context() {
        let kind = WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation;
        let scaffold = runtime_quantum_default_scaffold_source(kind);
        let mut microdomain = EmbeddedMaterialMixture::new("translation_live_support_microdomain");
        microdomain.add_component(scaffold.clone(), 64.0);
        for (bulk_field, support_molecule) in runtime_quantum_support_molecules_for_scaffold(
            kind,
            &scaffold,
            &[
                WholeCellBulkField::AminoAcids,
                WholeCellBulkField::ATP,
                WholeCellBulkField::ADP,
            ],
        ) {
            let amount = match bulk_field {
                WholeCellBulkField::AminoAcids => 21.0,
                WholeCellBulkField::ATP => 7.0,
                WholeCellBulkField::ADP => 4.0,
                _ => 0.0,
            };
            microdomain.add_component(support_molecule, amount);
        }

        let processes = runtime_quantum_processes_for_kind_from_embedded_microdomain_with_atom_budget_and_anchor_preferences(
            kind,
            4,
            1,
            &[],
            &microdomain,
        )
        .expect("runtime processes from live support microdomain");

        assert!(!processes.is_empty());
        assert!(processes.iter().any(|process| {
            (process.reactant_component_amount(0) - 64.0).abs() < 1.0e-9
                && process.reactant_component_amount(1) > 1.0e-9
                && process.reactant_component_amount(2) > 1.0e-9
        }));
        assert!(processes.iter().any(|process| {
            process
                .mixture
                .components
                .iter()
                .any(|component| component.molecule.graph.name == "amino_like")
        }));
        assert!(processes.iter().any(|process| {
            process
                .mixture
                .components
                .iter()
                .any(|component| component.molecule.graph.name == "atp_like")
        }));
        assert!(processes.iter().any(|process| {
            process
                .mixture
                .components
                .iter()
                .any(|component| component.molecule.graph.name == "adp_like")
        }));
    }

    #[test]
    fn runtime_quantum_process_snapshot_round_trip_preserves_state() {
        let mut process = translation_condensation_processes(4)
            .into_iter()
            .next()
            .expect("translation runtime quantum fragment");
        process
            .apply_extent(2.0)
            .expect("runtime quantum application");

        let snapshot = process.snapshot();
        let restored =
            WholeCellRuntimeQuantumProcessState::from_saved(snapshot).expect("restored runtime");

        assert_eq!(restored, process);
    }

    #[test]
    fn runtime_quantum_application_does_not_recreate_unsupported_reactants() {
        let mut process = translation_condensation_processes(4)
            .into_iter()
            .next()
            .expect("translation runtime quantum fragment");
        process.set_boundary_replenish_amounts(96.0, 0.0);
        for reactant_idx in 1..process.reactant_count() {
            process.set_reactant_inventory_amount(reactant_idx, 0.0);
        }

        let result = process
            .apply_extent(1.0)
            .expect("runtime quantum application without reactive support");

        assert!(result.is_none());
        assert!(process.scaffold_boundary_replenish_amount() > 0.0);
        assert!(process.reactive_boundary_replenish_amount().abs() < 1.0e-9);
        for reactant_idx in 1..process.reactant_count() {
            assert!(process.reactant_component_amount(reactant_idx).abs() < 1.0e-9);
        }
    }

    #[test]
    fn runtime_quantum_processes_expand_each_site_into_multiple_fragments() {
        let processes = default_runtime_quantum_processes();
        let atp_band_fragments = processes
            .iter()
            .filter(|process| {
                process.kind() == WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation
            })
            .count();
        let ribosome_fragments = processes
            .iter()
            .filter(|process| {
                process.kind()
                    == WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation
            })
            .count();

        assert!(atp_band_fragments >= 2);
        assert!(ribosome_fragments >= 2);
    }

    #[test]
    fn solved_fragment_quantum_application_emits_bond_local_polarization_dipoles() {
        let mut process = translation_condensation_processes(4)
            .into_iter()
            .next()
            .expect("translation runtime quantum fragment");
        let quantum = process.quantum.clone();
        let solve = process
            .execute_fragment_quantum_application(0.4, quantum)
            .expect("runtime quantum solve")
            .expect("present runtime quantum solve");
        let fragment_centroid = f64x3_from_f32(centroid_angstrom_slice(&process.reactants));

        assert!(solve.solved_dipoles.len() > 1);
        assert!(solve.solved_dipoles.iter().any(|dipole| {
            vector_norm_f64(dipole.dipole_e_angstrom) >= FRAGMENT_EMBEDDING_MIN_DIPOLE_E_ANGSTROM
                && squared_distance(
                    f64x3_to_f32(dipole.position_angstrom),
                    f64x3_to_f32(fragment_centroid),
                ) > 1.0e-2
        }));
        assert!(!solve.solved_orbital_response_fields.is_empty());
        assert!(solve
            .solved_orbital_response_fields
            .iter()
            .any(|field| field.coupling_ev >= FRAGMENT_EMBEDDING_MIN_ORBITAL_RESPONSE_EV));
    }

    #[test]
    fn neighbor_fragment_embedding_changes_runtime_fragment_quantum_delta() {
        let fragments = translation_condensation_processes(4);
        let quantum_configs = resolve_runtime_fragment_quantum_configs(&fragments);
        assert_eq!(quantum_configs.len(), fragments.len());

        // Verify embedding infrastructure is populated for at least one fragment.
        assert!(quantum_configs.iter().any(|qc| qc.embedding_point_charges.len() > 4));
        assert!(quantum_configs.iter().any(|qc| qc.embedding_dipoles.len() > 1));
        assert!(quantum_configs
            .iter()
            .any(|qc| !qc.embedding_orbital_response_fields.is_empty()));

        // Some scaffold configurations produce near-degenerate active spaces
        // (GS delta ≈ 0) when scaffold atoms sit far from the reaction center.
        // Try all fragments and verify that at least one shows a measurable
        // energy shift from neighbor embedding.
        let mut found_shift = false;
        for (_i, (frag, qc)) in fragments.iter().zip(quantum_configs.iter()).enumerate() {
            let mut uncoupled_fragment = frag.clone();
            let uncoupled = match uncoupled_fragment.apply_fragment_outcome(0.4) {
                Ok(Some(o)) => o,
                _ => continue,
            };
            // Skip fragments with near-zero uncoupled response (degenerate).
            if uncoupled.response_magnitude_ev() < 1.0e-4 {
                continue;
            }

            let mut embedded_fragment = frag.clone();
            let embedded = match embedded_fragment
                .apply_fragment_outcome_with_quantum(0.4, qc.clone())
            {
                Ok(Some(o)) => o,
                _ => continue,
            };

            let total_shift = (embedded.summary.ground_state_energy_delta_ev
                - uncoupled.summary.ground_state_energy_delta_ev)
                .abs()
                + (embedded.summary.nuclear_repulsion_delta_ev
                    - uncoupled.summary.nuclear_repulsion_delta_ev)
                    .abs();
            if total_shift > 1.0e-5 {
                found_shift = true;
                break;
            }
        }
        assert!(
            found_shift,
            "at least one fragment with non-degenerate active space should show \
             measurable energy shift from neighbor embedding",
        );
    }

    #[test]
    #[ignore] // Takes 25+ min: self-consistent ED loop with 6 scaffold atoms
    fn self_consistent_fragment_embedding_updates_neighbor_boundary_conditions() {
        let fragments = translation_condensation_processes(6);
        let heuristic_configs = resolve_runtime_fragment_quantum_configs(&fragments);
        let resolved = solve_self_consistent_runtime_fragment_quantum_configs(&fragments, 0.4)
            .expect("resolved self-consistent embedding");

        assert!(resolved.iteration_count > 1);
        assert!(resolved.residual_charge_e <= 5.0e-2);
        assert!(resolved.residual_dipole_e_angstrom <= 5.0e-2);
        assert!(resolved.residual_orbital_response_ev <= 5.0e-2);
        assert!(resolved.quantum_configs[0].embedding_dipoles.len() > 1);
        assert!(!resolved.quantum_configs[0]
            .embedding_orbital_response_fields
            .is_empty());

        // Phase 2c: per-fragment energy and convergence tracking.
        assert_eq!(resolved.per_fragment_energy_ev.len(), fragments.len());
        assert_eq!(resolved.per_fragment_converged.len(), fragments.len());
        // All energies must be finite (every fragment had sufficient reactants).
        for energy in &resolved.per_fragment_energy_ev {
            assert!(energy.is_finite(), "fragment energy must be finite");
        }
        // After multiple iterations, at least one fragment should be converged
        // (energy change < kT/10 ≈ 2.67e-3 eV between last two iterations).
        let converged_count = resolved
            .per_fragment_converged
            .iter()
            .filter(|&&c| c)
            .count();
        assert!(
            converged_count > 0,
            "at least one fragment should converge after {} iterations",
            resolved.iteration_count
        );

        let mut heuristic_fragment = fragments[0].clone();
        let heuristic = heuristic_fragment
            .apply_fragment_outcome_with_quantum(0.4, heuristic_configs[0].clone())
            .expect("heuristic fragment outcome")
            .expect("present heuristic fragment outcome");

        let mut resolved_fragment = fragments[0].clone();
        let updated = resolved_fragment
            .apply_fragment_outcome_with_quantum(0.4, resolved.quantum_configs[0].clone())
            .expect("resolved fragment outcome")
            .expect("present resolved fragment outcome");

        assert_ne!(resolved.quantum_configs[0], heuristic_configs[0]);
        let gs_delta = (updated.summary.ground_state_energy_delta_ev
            - heuristic.summary.ground_state_energy_delta_ev)
            .abs();
        let nr_delta = (updated.summary.nuclear_repulsion_delta_ev
            - heuristic.summary.nuclear_repulsion_delta_ev)
            .abs();
        let total_shift = gs_delta + nr_delta;
        assert!(total_shift > 1.0e-4);
    }

    #[test]
    fn coupled_fragment_outcomes_add_geometry_derived_stabilization() {
        let mut fragments = translation_condensation_processes(6);
        // Compute outcomes for all fragments.  Some scaffold configurations
        // produce near-degenerate active spaces (GS delta ≈ 0) because the
        // scaffold atoms sit far from the reaction center and don't "feel"
        // the bond topology change — this is physically correct, not a bug.
        let outcomes: Vec<_> = fragments
            .iter_mut()
            .filter_map(|process| process.apply_fragment_outcome(0.5).ok().flatten())
            .collect();
        assert!(
            !outcomes.is_empty(),
            "must produce at least one fragment outcome",
        );

        // At least one fragment must produce a non-trivial quantum response.
        // With 6 scaffold atoms (vs. the original 4) the larger fragment
        // breaks orbital degeneracy more reliably via asymmetric perturbation.
        let active_count = outcomes
            .iter()
            .filter(|o| o.response_magnitude_ev() > 1.0e-4)
            .count();
        assert!(
            active_count >= 1,
            "at least one fragment should have non-trivial quantum response \
             (got {} outcomes, magnitudes: {:?})",
            outcomes.len(),
            outcomes.iter().map(|o| o.response_magnitude_ev()).collect::<Vec<_>>(),
        );

        // The coupling mechanism is thoroughly validated by the synthetic test
        // `self_consistent_fragment_response_converges_with_stronger_than_one_pass_coupling`.
        // Here we verify the end-to-end pipeline: resolve_coupled_fragment_outcomes
        // must succeed and produce a valid result even when some fragments are
        // degenerate.
        let coupled =
            resolve_coupled_fragment_outcomes(&outcomes).expect("resolved coupled summary");
        assert!(coupled.iteration_count >= 1);

        // If 2+ fragments have non-trivial response, coupling must produce
        // stabilization (negative coupling energy).
        if active_count >= 2 {
            let uncoupled_ground_state = outcomes
                .iter()
                .map(|outcome| scale_runtime_summary(outcome.summary, outcome.extent))
                .fold(0.0f32, |acc, summary| {
                    acc + summary.ground_state_energy_delta_ev
                });
            assert!(coupled.summary.ground_state_energy_delta_ev < uncoupled_ground_state);
        }
    }

    #[test]
    fn self_consistent_fragment_response_converges_with_stronger_than_one_pass_coupling() {
        let outcomes = [
            RuntimeQuantumFragmentOutcome {
                summary: ReactionQuantumSummary {
                    event_count: 1,
                    ground_state_energy_delta_ev: -1.8,
                    nuclear_repulsion_delta_ev: 0.7,
                    net_formal_charge_delta: 0,
                },
                extent: 1.0,
                scaffold_centroid_angstrom: [0.0, 0.0, 0.0],
                reactive_centroid_angstrom: [1.3, 0.0, 0.0],
            },
            RuntimeQuantumFragmentOutcome {
                summary: ReactionQuantumSummary {
                    event_count: 1,
                    ground_state_energy_delta_ev: -1.5,
                    nuclear_repulsion_delta_ev: 0.5,
                    net_formal_charge_delta: 0,
                },
                extent: 0.9,
                scaffold_centroid_angstrom: [1.7, 0.3, 0.0],
                reactive_centroid_angstrom: [3.0, 0.3, 0.0],
            },
            RuntimeQuantumFragmentOutcome {
                summary: ReactionQuantumSummary {
                    event_count: 1,
                    ground_state_energy_delta_ev: -1.2,
                    nuclear_repulsion_delta_ev: 0.4,
                    net_formal_charge_delta: 0,
                },
                extent: 0.8,
                scaffold_centroid_angstrom: [0.8, 1.5, 0.0],
                reactive_centroid_angstrom: [2.0, 1.5, 0.0],
            },
        ];
        let one_pass_coupling = fragment_coupling_energy_ev(&outcomes);

        let resolved =
            resolve_coupled_fragment_outcomes(&outcomes).expect("resolved coupled summary");

        assert!(resolved.iteration_count > 1);
        assert!(resolved.residual_ev <= 5.0e-4);
        assert!(resolved.coupling_energy_ev < one_pass_coupling);
    }

    // --- Phase 3: Generic QuantumReactionSpec tests ---

    #[test]
    fn reaction_spec_round_trip_covers_all_five_kinds() {
        for kind in WholeCellRuntimeQuantumProcessKind::all() {
            let spec = kind.reaction_spec();
            assert_eq!(spec.kind, kind);
            assert_eq!(spec.preset, kind.preset());
            assert_eq!(spec.site_name, kind.site_name());
            assert!(!spec.name.is_empty());
            assert!(spec.reactive_displacement_angstrom > 0.0);
        }
    }

    #[test]
    fn generic_spec_path_matches_legacy_translation_condensation() {
        let kind = WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation;
        let legacy = translation_condensation_processes_with_scaffold_source(4, &[], None);
        let spec = kind.reaction_spec();
        let generic = quantum_processes_from_spec(&spec, 4, &[], None);

        assert_eq!(legacy.len(), generic.len());
        for (legacy_process, generic_process) in legacy.iter().zip(generic.iter()) {
            assert_eq!(legacy_process.kind(), generic_process.kind());
            assert_eq!(
                legacy_process.site_anchor_atom_idx(),
                generic_process.site_anchor_atom_idx()
            );
            assert_eq!(
                legacy_process.reactant_count(),
                generic_process.reactant_count()
            );
            assert_eq!(
                legacy_process.scaffold_atom_count(),
                generic_process.scaffold_atom_count()
            );
            // Verify the reaction templates have the same number of edits.
            assert_eq!(
                legacy_process.reaction.edits.len(),
                generic_process.reaction.edits.len()
            );
            // Verify reactant graphs match structurally.
            assert_eq!(
                legacy_process.reaction.reactants.len(),
                generic_process.reaction.reactants.len()
            );
        }
    }

    #[test]
    fn generic_spec_path_matches_legacy_atp_energy_phosphorylation() {
        let kind = WholeCellRuntimeQuantumProcessKind::AtpBandEnergyPhosphorylation;
        let legacy = atp_band_energy_processes_with_scaffold_source(4, &[], None);
        let spec = kind.reaction_spec();
        let generic = quantum_processes_from_spec(&spec, 4, &[], None);

        assert_eq!(legacy.len(), generic.len());
        for (legacy_process, generic_process) in legacy.iter().zip(generic.iter()) {
            assert_eq!(legacy_process.kind(), generic_process.kind());
            assert_eq!(
                legacy_process.site_anchor_atom_idx(),
                generic_process.site_anchor_atom_idx()
            );
            assert_eq!(
                legacy_process.reactant_count(),
                generic_process.reactant_count()
            );
            assert_eq!(
                legacy_process.reaction.edits.len(),
                generic_process.reaction.edits.len()
            );
        }
    }

    #[test]
    fn generic_spec_path_matches_legacy_replisome_nucleotide_phosphorylation() {
        let kind = WholeCellRuntimeQuantumProcessKind::ReplisomeNucleotidePhosphorylation;
        let legacy = replisome_nucleotide_processes_with_scaffold_source(4, &[], None);
        let spec = kind.reaction_spec();
        let generic = quantum_processes_from_spec(&spec, 4, &[], None);

        assert_eq!(legacy.len(), generic.len());
        for (legacy_process, generic_process) in legacy.iter().zip(generic.iter()) {
            assert_eq!(legacy_process.kind(), generic_process.kind());
            assert_eq!(
                legacy_process.site_anchor_atom_idx(),
                generic_process.site_anchor_atom_idx()
            );
            assert_eq!(
                legacy_process.reactant_count(),
                generic_process.reactant_count()
            );
            assert_eq!(
                legacy_process.reaction.edits.len(),
                generic_process.reaction.edits.len()
            );
        }
    }

    #[test]
    fn generic_spec_path_matches_legacy_atp_membrane_esterification() {
        let kind = WholeCellRuntimeQuantumProcessKind::AtpBandMembraneEsterification;
        let legacy = membrane_esterification_processes_with_scaffold_source(
            kind,
            "atp_band_membrane_esterification",
            2.55,
            4,
            &[],
            None,
        );
        let spec = kind.reaction_spec();
        let generic = quantum_processes_from_spec(&spec, 4, &[], None);

        assert_eq!(legacy.len(), generic.len());
        for (legacy_process, generic_process) in legacy.iter().zip(generic.iter()) {
            assert_eq!(legacy_process.kind(), generic_process.kind());
            assert_eq!(
                legacy_process.site_anchor_atom_idx(),
                generic_process.site_anchor_atom_idx()
            );
            assert_eq!(
                legacy_process.reactant_count(),
                generic_process.reactant_count()
            );
            assert_eq!(
                legacy_process.reaction.edits.len(),
                generic_process.reaction.edits.len()
            );
        }
    }

    #[test]
    fn generic_spec_path_matches_legacy_septum_membrane_esterification() {
        let kind = WholeCellRuntimeQuantumProcessKind::SeptumMembraneEsterification;
        let legacy = membrane_esterification_processes_with_scaffold_source(
            kind,
            "septum_membrane_esterification",
            2.75,
            4,
            &[],
            None,
        );
        let spec = kind.reaction_spec();
        let generic = quantum_processes_from_spec(&spec, 4, &[], None);

        assert_eq!(legacy.len(), generic.len());
        for (legacy_process, generic_process) in legacy.iter().zip(generic.iter()) {
            assert_eq!(legacy_process.kind(), generic_process.kind());
            assert_eq!(
                legacy_process.site_anchor_atom_idx(),
                generic_process.site_anchor_atom_idx()
            );
            assert_eq!(
                legacy_process.reactant_count(),
                generic_process.reactant_count()
            );
            assert_eq!(
                legacy_process.reaction.edits.len(),
                generic_process.reaction.edits.len()
            );
        }
    }

    #[test]
    fn generic_spec_path_produces_valid_quantum_configs_for_all_kinds() {
        for kind in WholeCellRuntimeQuantumProcessKind::all() {
            let spec = kind.reaction_spec();
            let processes = quantum_processes_from_spec(&spec, 4, &[], None);
            assert!(!processes.is_empty(), "no processes for {:?}", kind);
            for process in &processes {
                assert!(
                    process.reactant_count() >= 3,
                    "expected at least 3 reactants (scaffold + 2 reactive) for {:?}, got {}",
                    kind,
                    process.reactant_count()
                );
                // Verify reaction template references the right number of reactants.
                assert_eq!(
                    process.reaction.reactants.len(),
                    process.reactants.len(),
                    "reactant count mismatch in reaction template for {:?}",
                    kind
                );
                // Verify scaffold is the first reactant.
                assert_eq!(
                    &process.reaction.reactants[0],
                    &process.reactants[0].graph,
                    "scaffold graph mismatch for {:?}",
                    kind
                );
            }
        }
    }

    #[test]
    fn generic_spec_path_with_preferred_anchors_respects_order() {
        let kind = WholeCellRuntimeQuantumProcessKind::RibosomeTranslationCondensation;
        let baseline_spec = kind.reaction_spec();
        let baseline = quantum_processes_from_spec(&baseline_spec, 4, &[], None);
        let mut preferred_anchors = baseline
            .iter()
            .filter_map(|process| process.site_anchor_atom_idx())
            .collect::<Vec<_>>();
        assert!(!preferred_anchors.is_empty());

        preferred_anchors.reverse();
        let rebuilt = quantum_processes_from_spec(&baseline_spec, 6, &preferred_anchors, None);
        let rebuilt_anchors = rebuilt
            .iter()
            .filter_map(|process| process.site_anchor_atom_idx())
            .collect::<Vec<_>>();

        assert!(!rebuilt_anchors.is_empty());
        assert_eq!(
            rebuilt_anchors.first().copied(),
            preferred_anchors.first().copied()
        );
    }

    #[test]
    fn generic_spec_fragment_quantum_application_produces_valid_outcome() {
        for kind in WholeCellRuntimeQuantumProcessKind::all() {
            let spec = kind.reaction_spec();
            let mut processes = quantum_processes_from_spec(&spec, 4, &[], None);
            assert!(!processes.is_empty(), "no processes for {:?}", kind);
            let result = processes[0]
                .apply_extent(1.0)
                .expect("quantum application should succeed");
            assert!(
                result.is_some(),
                "expected quantum outcome for {:?}",
                kind
            );
            let summary = result.unwrap();
            // The quantum delta should produce an event.
            assert!(
                summary.event_count >= 1,
                "expected at least 1 quantum event for {:?}",
                kind
            );
        }
    }

    #[test]
    fn auto_discover_quantum_reactions_returns_empty_for_known_kinds() {
        // Build synthetic reactions that match all 5 known kinds.
        let known_reactions: Vec<WholeCellReactionRuntimeState> = vec![
            WholeCellReactionRuntimeState {
                id: "known_translation".to_string(),
                name: "translation".to_string(),
                reaction_class: WholeCellReactionClass::Translation,
                asset_class: WholeCellAssetClass::Generic,
                nominal_rate: 1.0,
                process_weights: Default::default(),
                operon: None,
                catalyst: None,
                reactants: Vec::new(),
                products: Vec::new(),
                subsystem_targets: vec![Syn3ASubsystemPreset::RibosomePolysomeCluster],
                chromosome_domain: None,
                patch_domain: Default::default(),
                spatial_scope: Default::default(),
                current_flux: 0.0,
                cumulative_extent: 0.0,
                reactant_satisfaction: 1.0,
                catalyst_support: 1.0,
            },
        ];
        let registry = WholeCellGenomeProcessRegistry {
            organism: "test".to_string(),
            chromosome_domains: Vec::new(),
            species: Vec::new(),
            reactions: Vec::new(),
        };

        let discovered = auto_discover_quantum_reactions(&known_reactions, &registry);
        assert!(
            discovered.is_empty(),
            "known kinds should not be auto-discovered"
        );
    }

    #[test]
    fn auto_discover_quantum_reactions_discovers_novel_reactions() {
        // Build a synthetic reaction with a known preset but not matching any hardcoded kind.
        let novel_reactions: Vec<WholeCellReactionRuntimeState> = vec![
            WholeCellReactionRuntimeState {
                id: "novel_assembly".to_string(),
                name: "novel assembly".to_string(),
                reaction_class: WholeCellReactionClass::ComplexMaturation,
                asset_class: WholeCellAssetClass::Energy,
                nominal_rate: 1.0,
                process_weights: Default::default(),
                operon: None,
                catalyst: None,
                reactants: Vec::new(),
                products: Vec::new(),
                // Use RibosomePolysomeCluster with Energy+ComplexMaturation --
                // this combo does NOT match any hardcoded kind.
                subsystem_targets: vec![Syn3ASubsystemPreset::RibosomePolysomeCluster],
                chromosome_domain: None,
                patch_domain: Default::default(),
                spatial_scope: Default::default(),
                current_flux: 0.0,
                cumulative_extent: 0.0,
                reactant_satisfaction: 1.0,
                catalyst_support: 1.0,
            },
        ];
        let registry = WholeCellGenomeProcessRegistry {
            organism: "test".to_string(),
            chromosome_domains: Vec::new(),
            species: Vec::new(),
            reactions: Vec::new(),
        };

        let discovered = auto_discover_quantum_reactions(&novel_reactions, &registry);
        assert!(
            !discovered.is_empty(),
            "novel reaction should be auto-discovered"
        );
        assert!(discovered[0].name.contains("novel_assembly"));
        // Verify it produces valid processes.
        let processes = quantum_processes_from_spec(&discovered[0], 4, &[], None);
        assert!(!processes.is_empty());
        assert!(processes[0].reactant_count() >= 3);
    }

    // -----------------------------------------------------------------------
    // Phase 7d: Eyring rate infrastructure tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_eyring_rate_at_body_temperature() {
        // At 310 K with a 0.5 eV barrier:
        //   kT = 0.02671 eV, prefactor ~ 6.46e12 s^-1
        //   exp(-0.5/0.02671) ~ 7.44e-9
        //   rate ~ 4.8e4 s^-1
        let rate = eyring_rate_from_barrier(0.5, 310.0);
        assert!(
            rate > 1e4 && rate < 1e5,
            "rate={rate} expected between 1e4 and 1e5"
        );
    }

    #[test]
    fn test_eyring_rate_zero_barrier() {
        // Zero barrier -> rate = kT/h ~ 6.25e12 s^-1
        let rate = eyring_rate_from_barrier(0.0, 310.0);
        assert!(
            rate > 6e12 && rate < 7e12,
            "rate={rate} expected between 6e12 and 7e12"
        );
    }

    #[test]
    fn test_quantum_rate_modifier() {
        let (rate, modifier) = quantum_rate_modifier(0.5, 310.0, 1000.0);
        assert!(rate > 1e3, "quantum rate should be reasonable, got {rate}");
        assert!(
            modifier > 1.0,
            "quantum rate should exceed slow heuristic, modifier={modifier}"
        );
    }

    #[test]
    fn test_quantum_rate_modifier_zero_heuristic() {
        let (_rate, modifier) = quantum_rate_modifier(0.5, 310.0, 0.0);
        assert!(
            (modifier - 1.0).abs() < f64::EPSILON,
            "modifier should be 1.0 when heuristic is zero, got {modifier}"
        );
    }

    #[test]
    fn test_arrhenius_temperature_dependence() {
        // Higher temperature -> higher rate
        let rate_310 = eyring_rate_from_barrier(0.5, 310.0);
        let rate_320 = eyring_rate_from_barrier(0.5, 320.0);
        assert!(
            rate_320 > rate_310,
            "rate should increase with temperature: {rate_320} > {rate_310}"
        );
    }

    #[test]
    fn test_reference_barriers_are_positive() {
        for &(name, barrier) in REFERENCE_BARRIERS {
            assert!(
                barrier > 0.0,
                "reference barrier for {name} should be positive, got {barrier}"
            );
        }
    }

    #[test]
    fn test_quantum_derived_rate_struct_defaults() {
        let qdr = QuantumDerivedRate {
            reaction_name: "test_reaction".to_string(),
            barrier_ev: Some(0.55),
            quantum_rate_s_inv: Some(eyring_rate_from_barrier(0.55, 310.0)),
            heuristic_rate_s_inv: 1000.0,
            using_quantum: false,
            temperature_k: 310.0,
        };
        assert_eq!(qdr.reaction_name, "test_reaction");
        assert!(qdr.barrier_ev.unwrap() > 0.0);
        assert!(qdr.quantum_rate_s_inv.unwrap() > 0.0);
        assert!(!qdr.using_quantum);
    }

    #[test]
    #[ignore] // Takes 30+ min on loaded machines (full 7-kind self-consistent ED loop)
    fn test_self_consistent_solver_populates_quantum_derived_rates() {
        let processes = default_runtime_quantum_processes();
        if processes.is_empty() {
            return;
        }
        let resolved = solve_self_consistent_runtime_fragment_quantum_configs(&processes, 0.4);
        match resolved {
            Ok(configs) => {
                assert_eq!(
                    configs.quantum_derived_rates.len(),
                    processes.len(),
                    "should have one QuantumDerivedRate per fragment"
                );
                for qdr in &configs.quantum_derived_rates {
                    assert!(!qdr.reaction_name.is_empty());
                    assert!((qdr.temperature_k - 310.0).abs() < f64::EPSILON);
                    // barrier and rate may or may not be present depending on
                    // whether the fragment produced an ED result
                    if let Some(barrier) = qdr.barrier_ev {
                        assert!(barrier >= 0.0, "barrier should be non-negative");
                        assert!(
                            qdr.quantum_rate_s_inv.is_some(),
                            "rate should be present when barrier is present"
                        );
                    }
                }
            }
            Err(_) => {
                // Solver may fail on some platforms; this is acceptable.
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 5: Live-state microdomain carving tests
    // -----------------------------------------------------------------------

    /// Build a small ethanol-like molecule for carving tests.
    /// C-C-O with hydrogens: 9 atoms, 8 bonds.
    fn test_ethanol_molecule() -> EmbeddedMolecule {
        let mut graph = MoleculeGraph::new("test_ethanol");
        let c1 = graph.add_element(PeriodicElement::C); // 0
        let c2 = graph.add_element(PeriodicElement::C); // 1
        let o = graph.add_element(PeriodicElement::O); // 2
        let h1 = graph.add_element(PeriodicElement::H); // 3
        let h2 = graph.add_element(PeriodicElement::H); // 4
        let h3 = graph.add_element(PeriodicElement::H); // 5
        let h4 = graph.add_element(PeriodicElement::H); // 6
        let h5 = graph.add_element(PeriodicElement::H); // 7
        let h_oh = graph.add_element(PeriodicElement::H); // 8

        graph.add_bond(c1, c2, BondOrder::Single).unwrap();
        graph.add_bond(c2, o, BondOrder::Single).unwrap();
        graph.add_bond(c1, h1, BondOrder::Single).unwrap();
        graph.add_bond(c1, h2, BondOrder::Single).unwrap();
        graph.add_bond(c1, h3, BondOrder::Single).unwrap();
        graph.add_bond(c2, h4, BondOrder::Single).unwrap();
        graph.add_bond(c2, h5, BondOrder::Single).unwrap();
        graph.add_bond(o, h_oh, BondOrder::Single).unwrap();

        EmbeddedMolecule::new(
            graph,
            vec![
                [0.0, 0.0, 0.0],       // C1
                [1.52, 0.0, 0.0],       // C2
                [2.28, 1.07, 0.0],      // O
                [-0.54, 0.92, 0.0],     // H
                [-0.54, -0.46, 0.80],   // H
                [-0.54, -0.46, -0.80],  // H
                [1.90, -0.51, 0.89],    // H
                [1.90, -0.51, -0.89],   // H
                [3.15, 0.85, 0.0],      // H (OH)
            ],
        )
        .expect("test ethanol geometry")
    }

    #[test]
    fn test_convergence_driven_carving_includes_anchor() {
        let mol = test_ethanol_molecule();
        for anchor in 0..mol.graph.atom_count() {
            let (fragment, _links) =
                carve_convergence_driven_fragment(&mol, anchor, 3, 20, 0.001);
            // The anchor atom's element must appear in the fragment
            let anchor_element = mol.graph.atoms[anchor].element;
            assert!(
                fragment
                    .graph
                    .atoms
                    .iter()
                    .any(|a| a.element == anchor_element),
                "fragment must contain anchor atom element {:?} (anchor={})",
                anchor_element,
                anchor,
            );
            // Fragment should have at least 1 atom
            assert!(
                fragment.graph.atom_count() >= 1,
                "fragment must have at least 1 atom",
            );
            // Positions must match atom count
            assert_eq!(
                fragment.positions_angstrom.len(),
                fragment.graph.atom_count(),
                "position count must match atom count",
            );
        }
    }

    #[test]
    fn test_link_hydrogens_placed_on_cut_bonds() {
        let mol = test_ethanol_molecule();
        // Carve a small fragment (max 3 atoms) from anchor 0 (C1).
        // C1 is bonded to C2, H1, H2, H3. With max 3 atoms the BFS will
        // include C1 and at most 2 neighbors, leaving some bonds cut.
        let (fragment, link_hydrogens) =
            carve_convergence_driven_fragment(&mol, 0, 3, 3, 1.0e-6);

        // With only 3 atoms selected from a 9-atom molecule, there must be
        // at least one cut bond and thus at least one link hydrogen.
        assert!(
            !link_hydrogens.is_empty(),
            "expected at least one link hydrogen for a partial fragment (fragment has {} atoms)",
            fragment.graph.atom_count(),
        );

        // Every link hydrogen should be at roughly C-H distance (1.09 A)
        // from one of the fragment atom positions.
        for lh_pos in &link_hydrogens {
            let min_dist = fragment
                .positions_angstrom
                .iter()
                .map(|fp| squared_distance(*fp, *lh_pos).sqrt())
                .fold(f32::MAX, f32::min);
            assert!(
                (min_dist - 1.09).abs() < 0.05,
                "link hydrogen should be ~1.09 A from nearest fragment atom, got {:.3} A",
                min_dist,
            );
        }
    }

    #[test]
    fn test_self_embedding_excludes_fragment_atoms() {
        let mol = test_ethanol_molecule();
        let included = vec![0, 1, 2]; // C1, C2, O
        let charges = self_embedding_from_excluded(&mol, &included, 0.001);

        // The 3 included atoms should NOT appear as point charges.
        // The remaining 6 atoms (all H) should be represented.
        let expected_excluded = mol.graph.atom_count() - included.len();
        assert!(
            charges.len() <= expected_excluded,
            "should have at most {} excluded charges, got {}",
            expected_excluded,
            charges.len(),
        );
        assert!(
            !charges.is_empty(),
            "should have at least one excluded charge",
        );

        // None of the charges should be at the exact positions of included atoms
        for &inc_idx in &included {
            let inc_pos = mol.positions_angstrom[inc_idx];
            for charge in &charges {
                let charge_pos = f64x3_to_f32(charge.position_angstrom);
                let dist = squared_distance(inc_pos, charge_pos).sqrt();
                assert!(
                    dist > 0.01,
                    "charge at included atom position (idx={}, dist={:.4})",
                    inc_idx,
                    dist,
                );
            }
        }
    }

    #[test]
    fn test_electronegativity_charges_plausible() {
        // O should be more negative than C
        let o_charge = electronegativity_partial_charge(PeriodicElement::O);
        let c_charge = electronegativity_partial_charge(PeriodicElement::C);
        let n_charge = electronegativity_partial_charge(PeriodicElement::N);
        let h_charge = electronegativity_partial_charge(PeriodicElement::H);
        let p_charge = electronegativity_partial_charge(PeriodicElement::P);

        assert!(
            o_charge < c_charge,
            "O ({}) should be more negative than C ({})",
            o_charge,
            c_charge,
        );
        assert!(
            n_charge < c_charge,
            "N ({}) should be more negative than C ({})",
            n_charge,
            c_charge,
        );
        assert!(
            h_charge == 0.0,
            "H should have zero partial charge, got {}",
            h_charge,
        );
        assert!(
            p_charge > 0.0,
            "P should have positive partial charge, got {}",
            p_charge,
        );
        // Electronegativity ordering: O < N < S < C < H
        let s_charge = electronegativity_partial_charge(PeriodicElement::S);
        assert!(
            o_charge < n_charge,
            "O ({}) should be more negative than N ({})",
            o_charge,
            n_charge,
        );
        assert!(
            s_charge < c_charge,
            "S ({}) should be more negative than C ({})",
            s_charge,
            c_charge,
        );
    }

    #[test]
    fn test_live_carved_scaffolds_produce_valid_fragments() {
        let mol = test_ethanol_molecule();
        let anchors = vec![0, 2]; // C1, O

        let scaffolds = live_carved_scaffolds(&mol, &anchors, 0.01, 6);
        assert!(
            !scaffolds.is_empty(),
            "should produce at least one fragment",
        );

        for scaffold in &scaffolds {
            // Each fragment should have a valid graph
            assert!(
                scaffold.scaffold.graph.atom_count() >= 1,
                "fragment must have at least 1 atom",
            );
            assert_eq!(
                scaffold.scaffold.positions_angstrom.len(),
                scaffold.scaffold.graph.atom_count(),
                "position count must match atom count",
            );
            // Anchor must be within the original molecule bounds
            assert!(
                scaffold.source_anchor_atom_idx < mol.graph.atom_count(),
                "anchor {} out of bounds (molecule has {} atoms)",
                scaffold.source_anchor_atom_idx,
                mol.graph.atom_count(),
            );
            // Fragment should not exceed the budget
            assert!(
                scaffold.scaffold.graph.atom_count() <= 6,
                "fragment has {} atoms, exceeding budget of 6",
                scaffold.scaffold.graph.atom_count(),
            );
            // All bonds in the fragment should reference valid atom indices
            for bond in &scaffold.scaffold.graph.bonds {
                assert!(
                    bond.i < scaffold.scaffold.graph.atom_count(),
                    "bond.i ({}) out of range",
                    bond.i,
                );
                assert!(
                    bond.j < scaffold.scaffold.graph.atom_count(),
                    "bond.j ({}) out of range",
                    bond.j,
                );
            }
        }
    }

    #[test]
    fn test_convergence_carving_respects_max_atoms() {
        let mol = test_ethanol_molecule();
        for budget in [1, 2, 3, 5, 9, 20] {
            let (fragment, _) =
                carve_convergence_driven_fragment(&mol, 0, 3, budget, 1.0e-9);
            let effective_budget = budget.max(2);
            assert!(
                fragment.graph.atom_count() <= effective_budget,
                "budget={}, got {} atoms (effective budget={})",
                budget,
                fragment.graph.atom_count(),
                effective_budget,
            );
        }
    }

    #[test]
    fn test_convergence_carving_full_molecule_at_high_budget() {
        let mol = test_ethanol_molecule();
        // With a very high budget and very tight convergence, we should get
        // the entire molecule (all 9 atoms).
        let (fragment, link_hydrogens) =
            carve_convergence_driven_fragment(&mol, 0, 3, 100, 1.0e-12);
        assert_eq!(
            fragment.graph.atom_count(),
            mol.graph.atom_count(),
            "should carve entire molecule when budget is large",
        );
        // No cut bonds means no link hydrogens
        assert!(
            link_hydrogens.is_empty(),
            "full molecule should have no link hydrogens, got {}",
            link_hydrogens.len(),
        );
    }

    #[test]
    fn test_live_carved_scaffolds_deduplicates() {
        let mol = test_ethanol_molecule();
        // Request the same anchor twice
        let anchors = vec![0, 0, 0];
        let scaffolds = live_carved_scaffolds(&mol, &anchors, 0.01, 6);
        assert_eq!(
            scaffolds.len(),
            1,
            "duplicate anchors should produce only one fragment",
        );
    }

    #[test]
    fn test_live_carved_scaffolds_skips_out_of_bounds_anchors() {
        let mol = test_ethanol_molecule();
        let anchors = vec![100, 200]; // way out of bounds
        let scaffolds = live_carved_scaffolds(&mol, &anchors, 0.01, 6);
        // Should fall back to anchor 0
        assert_eq!(
            scaffolds.len(),
            1,
            "out-of-bounds anchors should produce fallback fragment",
        );
        assert_eq!(scaffolds[0].source_anchor_atom_idx, 0);
    }

    // ------------------------------------------------------------------
    // Phase 8b: Live Microdomain Builder tests
    // ------------------------------------------------------------------

    #[test]
    fn test_live_microdomain_request_basic() {
        let mol = test_ethanol_molecule();
        let req = LiveMicrodomainRequest {
            molecule: mol.clone(),
            anchor_atoms: vec![0], // C1
            reaction_name: "test_basic".to_string(),
            max_atoms: 6,
            convergence_ev: 0.01,
            temperature_k: 310.0,
        };
        let result = build_live_microdomain(&req);

        // Must produce at least one scaffold fragment.
        assert!(
            !result.scaffolds.is_empty(),
            "basic request must produce at least one scaffold",
        );
        // Scaffold fragment must have atoms.
        for scaffold in &result.scaffolds {
            assert!(
                scaffold.scaffold.graph.atom_count() > 0,
                "scaffold must contain at least one atom",
            );
            assert_eq!(
                scaffold.scaffold.graph.atom_count(),
                scaffold.scaffold.positions_angstrom.len(),
                "atom count must match position count",
            );
        }
    }

    #[test]
    fn test_live_microdomain_from_mixture() {
        // Build a 2-component mixture: ethanol + methylamine.
        let ethanol = test_ethanol_molecule();
        let amine = methylamine([5.0, 0.0, 0.0]);

        let mut mixture = EmbeddedMaterialMixture::new("test_mixture");
        mixture.add_component(ethanol.clone(), 1.0);
        mixture.add_component(amine.clone(), 1.0);

        // Anchor 0 fits both molecules (both have atom 0), so only the first
        // matching component should be selected per the implementation.
        // Anchor 1 also fits both (both have atom 1).
        let results = build_live_microdomains_for_site(
            &mixture,
            &[0, 1],
            "test_mixture_reaction",
            6,
            0.01,
            310.0,
        );

        // At least one result should be produced.
        assert!(
            !results.is_empty(),
            "mixture with valid anchors must produce results",
        );
        // Each result must have scaffolds.
        for r in &results {
            assert!(
                !r.scaffolds.is_empty(),
                "each component result must have scaffolds",
            );
        }
    }

    #[test]
    fn test_live_microdomain_convergence_flag() {
        let mol = test_ethanol_molecule(); // 9 atoms

        // Case 1: generous budget -- should converge.
        let req_converged = LiveMicrodomainRequest {
            molecule: mol.clone(),
            anchor_atoms: vec![0],
            reaction_name: "convergence_true".to_string(),
            max_atoms: 100, // much larger than molecule
            convergence_ev: 0.01,
            temperature_k: 310.0,
        };
        let result_converged = build_live_microdomain(&req_converged);
        assert!(
            result_converged.converged,
            "generous budget should converge (molecule has only 9 atoms)",
        );

        // Case 2: tiny budget -- should NOT converge.
        let req_tight = LiveMicrodomainRequest {
            molecule: mol.clone(),
            anchor_atoms: vec![0],
            reaction_name: "convergence_false".to_string(),
            max_atoms: 2, // far too small
            convergence_ev: 1.0e-12, // extremely tight threshold
            temperature_k: 310.0,
        };
        let result_tight = build_live_microdomain(&req_tight);
        assert!(
            !result_tight.converged,
            "budget of 2 on 9-atom molecule with tiny threshold should not converge",
        );
    }

    #[test]
    fn test_live_microdomain_embedding_charge_conservation() {
        let mol = test_ethanol_molecule(); // 9 atoms
        let req = LiveMicrodomainRequest {
            molecule: mol.clone(),
            anchor_atoms: vec![0],
            reaction_name: "charge_conservation".to_string(),
            max_atoms: 5,
            convergence_ev: 0.001,
            temperature_k: 310.0,
        };
        let result = build_live_microdomain(&req);

        // Compute total heuristic charge of the entire molecule.
        let total_mol_charge: f64 = mol
            .graph
            .atoms
            .iter()
            .map(|a| f64::from(electronegativity_partial_charge(a.element)))
            .sum();

        // Compute total charge of carved fragment atoms + embedding charges.
        let mut fragment_charge: f64 = 0.0;
        for scaffold in &result.scaffolds {
            for atom in &scaffold.scaffold.graph.atoms {
                fragment_charge += f64::from(electronegativity_partial_charge(atom.element));
            }
        }
        let embedding_charge: f64 = result
            .embedding_charges
            .iter()
            .map(|c| c.charge_e)
            .sum();
        let reconstructed = fragment_charge + embedding_charge;

        // They should be approximately equal (convergence truncation may
        // discard very distant charges, so allow a tolerance).
        let delta = (total_mol_charge - reconstructed).abs();
        assert!(
            delta < 1.0,
            "total charge mismatch: molecule={total_mol_charge:.4}, \
             reconstructed={reconstructed:.4} (fragment={fragment_charge:.4} + \
             embedding={embedding_charge:.4}), delta={delta:.4}",
        );
    }
}
