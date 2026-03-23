//! Built-in molecule library for the pharma lab.
//!
//! ~50 common molecules across categories: drugs, amino acids, solvents, metabolites.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct LibraryEntry {
    pub name: &'static str,
    pub smiles: &'static str,
    pub category: &'static str,
    pub description: &'static str,
}

/// Look up a molecule by name (case-insensitive).
pub fn get_library_molecule(name: &str) -> Option<&'static LibraryEntry> {
    LIBRARY.iter().find(|e| e.name.eq_ignore_ascii_case(name))
}

/// Return the full library.
pub fn all_library_molecules() -> &'static [LibraryEntry] {
    &LIBRARY
}

static LIBRARY: [LibraryEntry; 50] = [
    // --- Common Drugs ---
    LibraryEntry { name: "aspirin", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O", category: "drug", description: "Acetylsalicylic acid — NSAID" },
    LibraryEntry { name: "ibuprofen", smiles: "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", category: "drug", description: "Propionic acid NSAID" },
    LibraryEntry { name: "caffeine", smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", category: "drug", description: "Xanthine alkaloid stimulant" },
    LibraryEntry { name: "acetaminophen", smiles: "CC(=O)NC1=CC=C(O)C=C1", category: "drug", description: "Paracetamol — analgesic/antipyretic" },
    LibraryEntry { name: "penicillin_g", smiles: "CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O", category: "drug", description: "Beta-lactam antibiotic" },
    LibraryEntry { name: "metformin", smiles: "CN(C)C(=N)NC(=N)N", category: "drug", description: "Biguanide antidiabetic" },
    LibraryEntry { name: "diazepam", smiles: "CN1C(=O)CN=C(C2=CC=CC=C2)C2=CC=CC(Cl)=C21", category: "drug", description: "Benzodiazepine anxiolytic" },
    LibraryEntry { name: "fluoxetine", smiles: "CNCCC(OC1=CC=C(C(F)(F)F)C=C1)C1=CC=CC=C1", category: "drug", description: "SSRI antidepressant (Prozac)" },
    LibraryEntry { name: "atorvastatin", smiles: "CC(C)C1=C(C(=O)NC2=CC=CC=C2)C(=C(N1CCC(CC(CC(=O)O)O)O)C1=CC=C(F)C=C1)C1=CC=CC=C1", category: "drug", description: "HMG-CoA reductase inhibitor (statin)" },
    LibraryEntry { name: "morphine", smiles: "CN1CCC23C4OC5=C(O)C=CC(=C25)C(O)C=CC3C1C4", category: "drug", description: "Opioid analgesic" },
    // --- Amino Acids ---
    LibraryEntry { name: "glycine", smiles: "NCC(=O)O", category: "amino_acid", description: "Simplest amino acid" },
    LibraryEntry { name: "alanine", smiles: "CC(N)C(=O)O", category: "amino_acid", description: "Nonpolar amino acid" },
    LibraryEntry { name: "valine", smiles: "CC(C)C(N)C(=O)O", category: "amino_acid", description: "Branched-chain amino acid" },
    LibraryEntry { name: "leucine", smiles: "CC(C)CC(N)C(=O)O", category: "amino_acid", description: "Essential branched-chain" },
    LibraryEntry { name: "isoleucine", smiles: "CCC(C)C(N)C(=O)O", category: "amino_acid", description: "Essential branched-chain" },
    LibraryEntry { name: "proline", smiles: "OC(=O)C1CCCN1", category: "amino_acid", description: "Cyclic amino acid" },
    LibraryEntry { name: "phenylalanine", smiles: "NC(CC1=CC=CC=C1)C(=O)O", category: "amino_acid", description: "Aromatic amino acid" },
    LibraryEntry { name: "tryptophan", smiles: "NC(CC1=CNC2=CC=CC=C12)C(=O)O", category: "amino_acid", description: "Aromatic amino acid" },
    LibraryEntry { name: "serine", smiles: "NC(CO)C(=O)O", category: "amino_acid", description: "Polar hydroxyl amino acid" },
    LibraryEntry { name: "cysteine", smiles: "NC(CS)C(=O)O", category: "amino_acid", description: "Thiol-containing amino acid" },
    // --- Nucleotides ---
    LibraryEntry { name: "adenine", smiles: "C1=NC2=NC=NC(=C2N1)N", category: "nucleotide", description: "Purine nucleobase" },
    LibraryEntry { name: "guanine", smiles: "C1=NC2=C(N1)C(=O)NC(=N2)N", category: "nucleotide", description: "Purine nucleobase" },
    LibraryEntry { name: "cytosine", smiles: "C1=C(NC(=O)N=C1)N", category: "nucleotide", description: "Pyrimidine nucleobase" },
    LibraryEntry { name: "thymine", smiles: "CC1=CNC(=O)NC1=O", category: "nucleotide", description: "Pyrimidine nucleobase (DNA)" },
    LibraryEntry { name: "uracil", smiles: "C1=CNC(=O)NC1=O", category: "nucleotide", description: "Pyrimidine nucleobase (RNA)" },
    // --- Metabolites ---
    LibraryEntry { name: "glucose", smiles: "OCC(O)C(O)C(O)C(O)C=O", category: "metabolite", description: "Primary cellular fuel" },
    LibraryEntry { name: "pyruvate", smiles: "CC(=O)C(=O)O", category: "metabolite", description: "Glycolysis end product" },
    LibraryEntry { name: "lactate", smiles: "CC(O)C(=O)O", category: "metabolite", description: "Anaerobic metabolite" },
    LibraryEntry { name: "citrate", smiles: "OC(CC(=O)O)(CC(=O)O)C(=O)O", category: "metabolite", description: "TCA cycle intermediate" },
    LibraryEntry { name: "succinate", smiles: "OC(=O)CCC(=O)O", category: "metabolite", description: "TCA cycle intermediate" },
    LibraryEntry { name: "acetyl_coa", smiles: "CC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)O", category: "metabolite", description: "Central metabolic carrier" },
    LibraryEntry { name: "ethanol", smiles: "CCO", category: "metabolite", description: "Fermentation product" },
    LibraryEntry { name: "urea", smiles: "NC(=O)N", category: "metabolite", description: "Nitrogen waste product" },
    // --- Solvents & Common ---
    LibraryEntry { name: "water", smiles: "O", category: "solvent", description: "Universal solvent" },
    LibraryEntry { name: "methanol", smiles: "CO", category: "solvent", description: "Simplest alcohol" },
    LibraryEntry { name: "acetone", smiles: "CC(=O)C", category: "solvent", description: "Simple ketone" },
    LibraryEntry { name: "dmso", smiles: "CS(=O)C", category: "solvent", description: "Dimethyl sulfoxide" },
    LibraryEntry { name: "benzene", smiles: "C1=CC=CC=C1", category: "solvent", description: "Aromatic hydrocarbon" },
    LibraryEntry { name: "toluene", smiles: "CC1=CC=CC=C1", category: "solvent", description: "Methylbenzene" },
    LibraryEntry { name: "chloroform", smiles: "ClC(Cl)Cl", category: "solvent", description: "Trichloromethane" },
    // --- Signaling & Vitamins ---
    LibraryEntry { name: "dopamine", smiles: "NCCC1=CC(O)=C(O)C=C1", category: "signaling", description: "Neurotransmitter" },
    LibraryEntry { name: "serotonin", smiles: "NCCC1=CNC2=CC(O)=CC=C12", category: "signaling", description: "5-HT neurotransmitter" },
    LibraryEntry { name: "adrenaline", smiles: "CNCC(O)C1=CC(O)=C(O)C=C1", category: "signaling", description: "Epinephrine — fight or flight" },
    LibraryEntry { name: "gaba", smiles: "NCCCC(=O)O", category: "signaling", description: "Inhibitory neurotransmitter" },
    LibraryEntry { name: "glutamate", smiles: "NC(CCC(=O)O)C(=O)O", category: "signaling", description: "Excitatory neurotransmitter" },
    LibraryEntry { name: "ascorbic_acid", smiles: "OCC(O)C1OC(=O)C(O)=C1O", category: "vitamin", description: "Vitamin C — antioxidant" },
    LibraryEntry { name: "cholesterol", smiles: "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C", category: "lipid", description: "Membrane sterol" },
    // --- Simple Gases ---
    LibraryEntry { name: "carbon_dioxide", smiles: "O=C=O", category: "gas", description: "CO2" },
    LibraryEntry { name: "methane", smiles: "C", category: "gas", description: "Simplest hydrocarbon" },
    LibraryEntry { name: "ammonia", smiles: "N", category: "gas", description: "NH3" },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_has_50_entries() {
        assert_eq!(all_library_molecules().len(), 50);
    }

    #[test]
    fn lookup_by_name() {
        assert!(get_library_molecule("water").is_some());
        assert!(get_library_molecule("ASPIRIN").is_some());
        assert!(get_library_molecule("nonexistent").is_none());
    }

    #[test]
    fn all_smiles_parseable() {
        for entry in all_library_molecules() {
            let result = super::super::smiles::parse_smiles(entry.smiles);
            assert!(result.is_ok(), "failed to parse '{}' SMILES: {} — {:?}",
                entry.name, entry.smiles, result.err());
        }
    }
}
