//! Biomolecular-level chemistry types: elements, bond orders, molecular graphs.
//!
//! This module provides the structural types used by [`crate::structure_ingest`]
//! to represent parsed PDB/mmCIF structures with full biomolecular topology
//! (residues, chains, secondary structure).  The types here sit between raw
//! coordinate files and the MD engine's low-level atom arrays.
//!
//! # Design Notes
//!
//! - [`PeriodicElement`] covers all 118 elements of the periodic table.
//!   Property data is stored in a compact static table indexed by enum ordinal.
//! - [`MoleculeGraph`] stores atoms + bonds + optional residue topology.
//! - [`EmbeddedMolecule`] pairs a graph with 3-D coordinates.
//! - [`BiomolecularTopology`] adds chain/residue/secondary-structure annotation.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Element data record
// ---------------------------------------------------------------------------

/// Static property record for a single element.
pub struct ElementRecord {
    pub atomic_number: u8,
    pub symbol: &'static str,
    pub name: &'static str,
    pub mass_daltons: f32,
    /// Cordero et al., Dalton Trans. 2008; Pyykko 2009 for Z>96.
    pub covalent_radius_angstrom: f32,
    /// Bondi 1964 / Mantina 2009; estimated for heavy/superheavy elements.
    pub van_der_waals_radius_angstrom: f32,
    /// CPK / Jmol standard atom color.
    pub cpk_color_rgb: [u8; 3],
    /// Pauling electronegativity (None for noble gases and some superheavy).
    pub pauling_electronegativity: Option<f64>,
    /// Abbreviated electron configuration.
    pub electron_configuration_short: &'static str,
}

const fn er(
    z: u8, sym: &'static str, name: &'static str, mass: f32,
    cov: f32, vdw: f32, cpk: [u8; 3], en: Option<f64>, conf: &'static str,
) -> ElementRecord {
    ElementRecord {
        atomic_number: z, symbol: sym, name: name, mass_daltons: mass,
        covalent_radius_angstrom: cov, van_der_waals_radius_angstrom: vdw,
        cpk_color_rgb: cpk, pauling_electronegativity: en,
        electron_configuration_short: conf,
    }
}

// ---------------------------------------------------------------------------
// Periodic element enum — all 118 elements
// ---------------------------------------------------------------------------

/// All 118 elements of the periodic table.
///
/// Ordered by atomic number. Property accessors delegate to a static
/// [`ElementRecord`] table for O(1) lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PeriodicElement {
    // Period 1
    H = 0, He,
    // Period 2
    Li, Be, B, C, N, O, F, Ne,
    // Period 3
    Na, Mg, Al, Si, P, S, Cl, Ar,
    // Period 4
    K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn,
    Ga, Ge, As, Se, Br, Kr,
    // Period 5
    Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd,
    In, Sn, Sb, Te, I, Xe,
    // Period 6
    Cs, Ba,
    La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu,
    Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn,
    // Period 7
    Fr, Ra,
    Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr,
    Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg, Cn, Nh, Fl, Mc, Lv, Ts, Og,
}

// ---------------------------------------------------------------------------
// Static element data table (indexed by enum ordinal 0..117)
// ---------------------------------------------------------------------------
// Sources: IUPAC 2021 (masses), Cordero 2008 / Pyykko 2009 (cov radii),
//          Bondi 1964 / Mantina 2009 (VDW radii), Jmol (CPK colors),
//          Pauling scale (EN), NIST (electron configurations).
// ---------------------------------------------------------------------------

static ELEMENTS: [ElementRecord; 118] = [
    // Period 1
    er(  1, "H",  "Hydrogen",      1.008,  0.31, 1.20, [255,255,255], Some(2.20), "1s1"),
    er(  2, "He", "Helium",        4.003,  0.28, 1.40, [217,255,255], None,        "1s2"),
    // Period 2
    er(  3, "Li", "Lithium",       6.941,  1.28, 1.82, [204,128,255], Some(0.98), "[He]2s1"),
    er(  4, "Be", "Beryllium",     9.012,  0.96, 1.53, [194,255,  0], Some(1.57), "[He]2s2"),
    er(  5, "B",  "Boron",        10.81,   0.84, 1.92, [255,181,181], Some(2.04), "[He]2s2 2p1"),
    er(  6, "C",  "Carbon",       12.011,  0.76, 1.70, [ 80, 80, 80], Some(2.55), "[He]2s2 2p2"),
    er(  7, "N",  "Nitrogen",     14.007,  0.71, 1.55, [ 48, 80,248], Some(3.04), "[He]2s2 2p3"),
    er(  8, "O",  "Oxygen",       15.999,  0.66, 1.52, [255, 13, 13], Some(3.44), "[He]2s2 2p4"),
    er(  9, "F",  "Fluorine",     18.998,  0.57, 1.47, [144,224, 80], Some(3.98), "[He]2s2 2p5"),
    er( 10, "Ne", "Neon",         20.180,  0.58, 1.54, [179,227,245], None,        "[He]2s2 2p6"),
    // Period 3
    er( 11, "Na", "Sodium",       22.990,  1.66, 2.27, [171, 92,242], Some(0.93), "[Ne]3s1"),
    er( 12, "Mg", "Magnesium",    24.305,  1.41, 1.73, [138,255,  0], Some(1.31), "[Ne]3s2"),
    er( 13, "Al", "Aluminium",    26.982,  1.21, 1.84, [191,166,166], Some(1.61), "[Ne]3s2 3p1"),
    er( 14, "Si", "Silicon",      28.086,  1.11, 2.10, [240,200,160], Some(1.90), "[Ne]3s2 3p2"),
    er( 15, "P",  "Phosphorus",   30.974,  1.07, 1.80, [255,165,  0], Some(2.19), "[Ne]3s2 3p3"),
    er( 16, "S",  "Sulfur",       32.065,  1.05, 1.80, [255,255, 48], Some(2.58), "[Ne]3s2 3p4"),
    er( 17, "Cl", "Chlorine",     35.453,  1.02, 1.75, [ 31,240, 31], Some(3.16), "[Ne]3s2 3p5"),
    er( 18, "Ar", "Argon",        39.948,  1.06, 1.88, [128,209,227], None,        "[Ne]3s2 3p6"),
    // Period 4
    er( 19, "K",  "Potassium",    39.098,  2.03, 2.75, [143, 64,212], Some(0.82), "[Ar]4s1"),
    er( 20, "Ca", "Calcium",      40.078,  1.76, 2.31, [ 61,255,  0], Some(1.00), "[Ar]4s2"),
    er( 21, "Sc", "Scandium",     44.956,  1.70, 2.15, [230,230,230], Some(1.36), "[Ar]3d1 4s2"),
    er( 22, "Ti", "Titanium",     47.867,  1.60, 2.11, [191,194,199], Some(1.54), "[Ar]3d2 4s2"),
    er( 23, "V",  "Vanadium",     50.942,  1.53, 2.07, [166,166,171], Some(1.63), "[Ar]3d3 4s2"),
    er( 24, "Cr", "Chromium",     51.996,  1.39, 2.06, [138,153,199], Some(1.66), "[Ar]3d5 4s1"),
    er( 25, "Mn", "Manganese",    54.938,  1.39, 2.05, [156,122,199], Some(1.55), "[Ar]3d5 4s2"),
    er( 26, "Fe", "Iron",         55.845,  1.32, 2.05, [224,102, 51], Some(1.83), "[Ar]3d6 4s2"),
    er( 27, "Co", "Cobalt",       58.933,  1.26, 2.00, [240,144,160], Some(1.88), "[Ar]3d7 4s2"),
    er( 28, "Ni", "Nickel",       58.693,  1.24, 1.97, [ 80,208, 80], Some(1.91), "[Ar]3d8 4s2"),
    er( 29, "Cu", "Copper",       63.546,  1.32, 1.96, [200,128, 51], Some(1.90), "[Ar]3d10 4s1"),
    er( 30, "Zn", "Zinc",         65.38,   1.22, 2.01, [125,128,176], Some(1.65), "[Ar]3d10 4s2"),
    er( 31, "Ga", "Gallium",      69.723,  1.22, 1.87, [194,143,143], Some(1.81), "[Ar]3d10 4s2 4p1"),
    er( 32, "Ge", "Germanium",    72.630,  1.20, 2.11, [102,143,143], Some(2.01), "[Ar]3d10 4s2 4p2"),
    er( 33, "As", "Arsenic",      74.922,  1.19, 1.85, [189,128,227], Some(2.18), "[Ar]3d10 4s2 4p3"),
    er( 34, "Se", "Selenium",     78.971,  1.20, 1.90, [255,161,  0], Some(2.55), "[Ar]3d10 4s2 4p4"),
    er( 35, "Br", "Bromine",      79.904,  1.20, 1.85, [166, 41, 41], Some(2.96), "[Ar]3d10 4s2 4p5"),
    er( 36, "Kr", "Krypton",      83.798,  1.16, 2.02, [ 92,184,209], Some(3.00), "[Ar]3d10 4s2 4p6"),
    // Period 5
    er( 37, "Rb", "Rubidium",     85.468,  2.20, 3.03, [112, 46,176], Some(0.82), "[Kr]5s1"),
    er( 38, "Sr", "Strontium",    87.62,   1.95, 2.49, [  0,255,  0], Some(0.95), "[Kr]5s2"),
    er( 39, "Y",  "Yttrium",      88.906,  1.90, 2.32, [148,255,255], Some(1.22), "[Kr]4d1 5s2"),
    er( 40, "Zr", "Zirconium",    91.224,  1.75, 2.23, [148,224,224], Some(1.33), "[Kr]4d2 5s2"),
    er( 41, "Nb", "Niobium",      92.906,  1.64, 2.18, [115,194,201], Some(1.60), "[Kr]4d4 5s1"),
    er( 42, "Mo", "Molybdenum",   95.95,   1.54, 2.17, [ 84,181,181], Some(2.16), "[Kr]4d5 5s1"),
    er( 43, "Tc", "Technetium",   98.0,    1.47, 2.16, [ 59,158,158], Some(1.90), "[Kr]4d5 5s2"),
    er( 44, "Ru", "Ruthenium",   101.07,   1.46, 2.13, [ 36,143,143], Some(2.20), "[Kr]4d7 5s1"),
    er( 45, "Rh", "Rhodium",     102.906,  1.42, 2.10, [ 10,125,140], Some(2.28), "[Kr]4d8 5s1"),
    er( 46, "Pd", "Palladium",   106.42,   1.39, 2.10, [  0,105,133], Some(2.20), "[Kr]4d10"),
    er( 47, "Ag", "Silver",      107.868,  1.45, 2.11, [192,192,192], Some(1.93), "[Kr]4d10 5s1"),
    er( 48, "Cd", "Cadmium",     112.414,  1.44, 2.18, [255,217,143], Some(1.69), "[Kr]4d10 5s2"),
    er( 49, "In", "Indium",      114.818,  1.42, 1.93, [166,117,115], Some(1.78), "[Kr]4d10 5s2 5p1"),
    er( 50, "Sn", "Tin",         118.710,  1.39, 2.17, [102,128,128], Some(1.96), "[Kr]4d10 5s2 5p2"),
    er( 51, "Sb", "Antimony",    121.760,  1.39, 2.06, [158, 99,181], Some(2.05), "[Kr]4d10 5s2 5p3"),
    er( 52, "Te", "Tellurium",   127.60,   1.38, 2.06, [212,122,  0], Some(2.10), "[Kr]4d10 5s2 5p4"),
    er( 53, "I",  "Iodine",      126.904,  1.39, 1.98, [148,  0,148], Some(2.66), "[Kr]4d10 5s2 5p5"),
    er( 54, "Xe", "Xenon",       131.293,  1.40, 2.16, [ 66,158,176], Some(2.60), "[Kr]4d10 5s2 5p6"),
    // Period 6
    er( 55, "Cs", "Caesium",     132.905,  2.44, 3.43, [ 87, 23,143], Some(0.79), "[Xe]6s1"),
    er( 56, "Ba", "Barium",      137.327,  2.15, 2.68, [  0,201,  0], Some(0.89), "[Xe]6s2"),
    // Lanthanides
    er( 57, "La", "Lanthanum",   138.905,  2.07, 2.43, [112,212,255], Some(1.10), "[Xe]5d1 6s2"),
    er( 58, "Ce", "Cerium",      140.116,  2.04, 2.42, [255,255,199], Some(1.12), "[Xe]4f1 5d1 6s2"),
    er( 59, "Pr", "Praseodymium",140.908,  2.03, 2.40, [217,255,199], Some(1.13), "[Xe]4f3 6s2"),
    er( 60, "Nd", "Neodymium",   144.242,  2.01, 2.39, [199,255,199], Some(1.14), "[Xe]4f4 6s2"),
    er( 61, "Pm", "Promethium",  145.0,    1.99, 2.38, [163,255,199], Some(1.13), "[Xe]4f5 6s2"),
    er( 62, "Sm", "Samarium",    150.36,   1.98, 2.36, [143,255,199], Some(1.17), "[Xe]4f6 6s2"),
    er( 63, "Eu", "Europium",    151.964,  1.98, 2.35, [ 97,255,199], Some(1.20), "[Xe]4f7 6s2"),
    er( 64, "Gd", "Gadolinium",  157.25,   1.96, 2.34, [ 69,255,199], Some(1.20), "[Xe]4f7 5d1 6s2"),
    er( 65, "Tb", "Terbium",     158.925,  1.94, 2.33, [ 48,255,199], Some(1.10), "[Xe]4f9 6s2"),
    er( 66, "Dy", "Dysprosium",  162.500,  1.92, 2.31, [ 31,255,199], Some(1.22), "[Xe]4f10 6s2"),
    er( 67, "Ho", "Holmium",     164.930,  1.92, 2.30, [  0,255,156], Some(1.23), "[Xe]4f11 6s2"),
    er( 68, "Er", "Erbium",      167.259,  1.89, 2.29, [  0,230,117], Some(1.24), "[Xe]4f12 6s2"),
    er( 69, "Tm", "Thulium",     168.934,  1.90, 2.27, [  0,212, 82], Some(1.25), "[Xe]4f13 6s2"),
    er( 70, "Yb", "Ytterbium",   173.045,  1.87, 2.26, [  0,191, 56], Some(1.10), "[Xe]4f14 6s2"),
    er( 71, "Lu", "Lutetium",    174.967,  1.87, 2.24, [  0,171, 36], Some(1.27), "[Xe]4f14 5d1 6s2"),
    // Period 6 d-block (continued)
    er( 72, "Hf", "Hafnium",     178.49,   1.75, 2.23, [ 77,194,255], Some(1.30), "[Xe]4f14 5d2 6s2"),
    er( 73, "Ta", "Tantalum",    180.948,  1.70, 2.22, [ 77,166,255], Some(1.50), "[Xe]4f14 5d3 6s2"),
    er( 74, "W",  "Tungsten",    183.84,   1.62, 2.17, [ 38,102,150], Some(2.36), "[Xe]4f14 5d4 6s2"),
    er( 75, "Re", "Rhenium",     186.207,  1.51, 2.16, [ 38,125,171], Some(1.90), "[Xe]4f14 5d5 6s2"),
    er( 76, "Os", "Osmium",      190.23,   1.44, 2.16, [ 38,102,150], Some(2.20), "[Xe]4f14 5d6 6s2"),
    er( 77, "Ir", "Iridium",     192.217,  1.41, 2.13, [ 23, 84,135], Some(2.20), "[Xe]4f14 5d7 6s2"),
    er( 78, "Pt", "Platinum",    195.084,  1.36, 2.13, [208,208,224], Some(2.28), "[Xe]4f14 5d9 6s1"),
    er( 79, "Au", "Gold",        196.967,  1.36, 2.14, [255,209, 35], Some(2.54), "[Xe]4f14 5d10 6s1"),
    er( 80, "Hg", "Mercury",     200.592,  1.32, 2.23, [184,184,208], Some(2.00), "[Xe]4f14 5d10 6s2"),
    er( 81, "Tl", "Thallium",    204.383,  1.45, 1.96, [166, 84, 77], Some(1.62), "[Xe]4f14 5d10 6s2 6p1"),
    er( 82, "Pb", "Lead",        207.2,    1.46, 2.02, [ 87, 89, 97], Some(1.87), "[Xe]4f14 5d10 6s2 6p2"),
    er( 83, "Bi", "Bismuth",     208.980,  1.48, 2.07, [158, 79,181], Some(2.02), "[Xe]4f14 5d10 6s2 6p3"),
    er( 84, "Po", "Polonium",    209.0,    1.40, 1.97, [171, 92,  0], Some(2.00), "[Xe]4f14 5d10 6s2 6p4"),
    er( 85, "At", "Astatine",    210.0,    1.50, 2.02, [117, 79, 69], Some(2.20), "[Xe]4f14 5d10 6s2 6p5"),
    er( 86, "Rn", "Radon",       222.0,    1.50, 2.20, [ 66,130,150], None,        "[Xe]4f14 5d10 6s2 6p6"),
    // Period 7
    er( 87, "Fr", "Francium",    223.0,    2.60, 3.48, [ 66,  0,102], Some(0.70), "[Rn]7s1"),
    er( 88, "Ra", "Radium",      226.0,    2.21, 2.83, [  0,125,  0], Some(0.90), "[Rn]7s2"),
    // Actinides
    er( 89, "Ac", "Actinium",    227.0,    2.15, 2.47, [112,171,250], Some(1.10), "[Rn]6d1 7s2"),
    er( 90, "Th", "Thorium",     232.038,  2.06, 2.45, [  0,186,255], Some(1.30), "[Rn]6d2 7s2"),
    er( 91, "Pa", "Protactinium",231.036,  2.00, 2.43, [  0,161,255], Some(1.50), "[Rn]5f2 6d1 7s2"),
    er( 92, "U",  "Uranium",     238.029,  1.96, 2.41, [  0,143,255], Some(1.38), "[Rn]5f3 6d1 7s2"),
    er( 93, "Np", "Neptunium",   237.0,    1.90, 2.39, [  0,128,255], Some(1.36), "[Rn]5f4 6d1 7s2"),
    er( 94, "Pu", "Plutonium",   244.0,    1.87, 2.43, [  0,107,255], Some(1.28), "[Rn]5f6 7s2"),
    er( 95, "Am", "Americium",   243.0,    1.80, 2.44, [ 84, 92,242], Some(1.13), "[Rn]5f7 7s2"),
    er( 96, "Cm", "Curium",      247.0,    1.69, 2.45, [120, 92,227], Some(1.28), "[Rn]5f7 6d1 7s2"),
    er( 97, "Bk", "Berkelium",   247.0,    1.68, 2.44, [138, 79,227], Some(1.30), "[Rn]5f9 7s2"),
    er( 98, "Cf", "Californium", 251.0,    1.68, 2.45, [161, 54,212], Some(1.30), "[Rn]5f10 7s2"),
    er( 99, "Es", "Einsteinium", 252.0,    1.65, 2.45, [179, 31,212], Some(1.30), "[Rn]5f11 7s2"),
    er(100, "Fm", "Fermium",     257.0,    1.67, 2.45, [179, 31,186], Some(1.30), "[Rn]5f12 7s2"),
    er(101, "Md", "Mendelevium", 258.0,    1.73, 2.46, [179, 13,166], Some(1.30), "[Rn]5f13 7s2"),
    er(102, "No", "Nobelium",    259.0,    1.76, 2.46, [189, 13,135], Some(1.30), "[Rn]5f14 7s2"),
    er(103, "Lr", "Lawrencium",  266.0,    1.61, 2.46, [199,  0,102], Some(1.30), "[Rn]5f14 7s2 7p1"),
    // Period 7 d-block (superheavy)
    er(104, "Rf", "Rutherfordium",267.0,   1.57, 2.30, [204,  0, 89], None,        "[Rn]5f14 6d2 7s2"),
    er(105, "Db", "Dubnium",     268.0,    1.49, 2.30, [209,  0, 79], None,        "[Rn]5f14 6d3 7s2"),
    er(106, "Sg", "Seaborgium",  269.0,    1.43, 2.30, [217,  0, 69], None,        "[Rn]5f14 6d4 7s2"),
    er(107, "Bh", "Bohrium",     270.0,    1.41, 2.30, [224,  0, 56], None,        "[Rn]5f14 6d5 7s2"),
    er(108, "Hs", "Hassium",     277.0,    1.34, 2.30, [230,  0, 46], None,        "[Rn]5f14 6d6 7s2"),
    er(109, "Mt", "Meitnerium",  278.0,    1.29, 2.30, [235,  0, 38], None,        "[Rn]5f14 6d7 7s2"),
    er(110, "Ds", "Darmstadtium",281.0,    1.28, 2.30, [240,  0, 33], None,        "[Rn]5f14 6d8 7s2"),
    er(111, "Rg", "Roentgenium", 282.0,    1.21, 2.30, [241,  0, 30], None,        "[Rn]5f14 6d9 7s2"),
    er(112, "Cn", "Copernicium", 285.0,    1.22, 2.30, [242,  0, 26], None,        "[Rn]5f14 6d10 7s2"),
    er(113, "Nh", "Nihonium",    286.0,    1.36, 2.30, [243,  0, 23], None,        "[Rn]5f14 6d10 7s2 7p1"),
    er(114, "Fl", "Flerovium",   289.0,    1.43, 2.30, [244,  0, 21], None,        "[Rn]5f14 6d10 7s2 7p2"),
    er(115, "Mc", "Moscovium",   290.0,    1.62, 2.30, [245,  0, 18], None,        "[Rn]5f14 6d10 7s2 7p3"),
    er(116, "Lv", "Livermorium", 293.0,    1.75, 2.30, [246,  0, 16], None,        "[Rn]5f14 6d10 7s2 7p4"),
    er(117, "Ts", "Tennessine",  294.0,    1.65, 2.30, [247,  0, 14], None,        "[Rn]5f14 6d10 7s2 7p5"),
    er(118, "Og", "Oganesson",   294.0,    1.57, 2.30, [248,  0, 12], None,        "[Rn]5f14 6d10 7s2 7p6"),
];

// ---------------------------------------------------------------------------
// PeriodicElement methods — all delegate to ELEMENTS table
// ---------------------------------------------------------------------------

impl PeriodicElement {
    /// Access the full data record for this element.
    #[inline]
    fn data(self) -> &'static ElementRecord {
        &ELEMENTS[self as usize]
    }

    /// Total number of elements.
    pub const COUNT: usize = 118;

    /// Look up element from a 1- or 2-character symbol or full name (case-insensitive).
    pub fn from_symbol_or_name(s: &str) -> Option<Self> {
        let trimmed = s.trim();
        for (i, rec) in ELEMENTS.iter().enumerate() {
            if trimmed.eq_ignore_ascii_case(rec.symbol) || trimmed.eq_ignore_ascii_case(rec.name) {
                // SAFETY: PeriodicElement is #[repr(u8)] with 118 variants 0..117.
                return Some(unsafe { std::mem::transmute(i as u8) });
            }
        }
        None
    }

    /// Look up element by atomic number (1–118).
    pub fn from_atomic_number(z: u8) -> Option<Self> {
        if z == 0 || z > 118 { return None; }
        // SAFETY: PeriodicElement is #[repr(u8)] with 118 variants 0..117.
        // Atomic number z maps to index z-1.
        Some(unsafe { std::mem::transmute((z - 1) as u8) })
    }

    /// Atomic mass in daltons (IUPAC 2021; most stable isotope for radioactive).
    pub fn mass_daltons(self) -> f32 { self.data().mass_daltons }

    /// Standard covalent radius in angstroms (Cordero et al. 2008; Pyykko 2009 for Z>96).
    pub fn covalent_radius_angstrom(self) -> f32 { self.data().covalent_radius_angstrom }

    /// Van der Waals radius in angstroms (Bondi 1964 / Mantina 2009).
    pub fn van_der_waals_radius_angstrom(self) -> f32 { self.data().van_der_waals_radius_angstrom }

    /// Standard CPK (Corey-Pauling-Koltun) / Jmol atom color as [R, G, B].
    pub fn cpk_color_rgb(self) -> [u8; 3] { self.data().cpk_color_rgb }

    /// CPK color as normalized [0.0-1.0] float triple.
    pub fn cpk_color_f32(self) -> [f32; 3] {
        let [r, g, b] = self.cpk_color_rgb();
        [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
    }

    /// Pauling electronegativity (None for noble gases and some superheavy).
    pub fn pauling_electronegativity(self) -> Option<f64> { self.data().pauling_electronegativity }

    /// Short element symbol (1-2 chars, title case).
    pub fn symbol(self) -> &'static str { self.data().symbol }

    /// Full element name.
    pub fn name(self) -> &'static str { self.data().name }

    /// Atomic number (number of protons).
    pub fn atomic_number(self) -> u8 { self.data().atomic_number }

    /// Abbreviated electron configuration string.
    pub fn electron_configuration_short(self) -> &'static str { self.data().electron_configuration_short }

    /// Return an iterator over all 118 elements in atomic-number order.
    pub fn all() -> impl Iterator<Item = PeriodicElement> {
        (0u8..118).map(|i| {
            // SAFETY: PeriodicElement is #[repr(u8)] with 118 variants 0..117.
            unsafe { std::mem::transmute(i) }
        })
    }

    /// Universal Force Field Lennard-Jones parameters (Rappe et al. 1992).
    /// Returns (sigma_angstrom, epsilon_kcal_mol).
    pub fn uff_lj_params(self) -> (f32, f32) {
        // UFF: sigma = 2^(1/6) * r_vdw, epsilon from UFF paper.
        // For elements not in UFF, estimate from VDW radius with generic epsilon.
        let r = self.van_der_waals_radius_angstrom();
        let sigma = r; // UFF convention: sigma ≈ VDW radius
        let epsilon = match self.atomic_number() {
            1 => 0.044,   // H
            2 => 0.056,   // He
            6 => 0.105,   // C
            7 => 0.069,   // N
            8 => 0.060,   // O
            9 => 0.050,   // F
            15 => 0.305,  // P
            16 => 0.274,  // S
            17 => 0.227,  // Cl
            35 => 0.251,  // Br
            53 => 0.339,  // I
            26 => 0.013,  // Fe
            29 => 0.005,  // Cu
            30 => 0.124,  // Zn
            _ => {
                // Generic estimate: scale by polarizability proxy (VDW^3)
                let r3 = r * r * r;
                (0.01 * r3).min(0.5)
            }
        };
        (sigma, epsilon)
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
    pub fn add_atom(
        &mut self,
        element: PeriodicElement,
        atom_name: &str,
        residue: Option<ResidueInfo>,
    ) -> usize {
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
            .filter_map(|a| a.residue.as_ref().map(|r| (r.chain_id.clone(), r.seq_num)))
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
    pub fn representative_atp() -> Self {
        Self::new("ATP")
    }
    /// Stub ADP graph (adenosine diphosphate).
    pub fn representative_adp() -> Self {
        Self::new("ADP")
    }
    /// Stub orthophosphoric acid graph (Pi / H3PO4).
    pub fn representative_orthophosphoric_acid() -> Self {
        Self::new("Pi")
    }
    /// Stub glucose graph.
    pub fn representative_glucose() -> Self {
        Self::new("glucose")
    }
    /// Stub O2 graph.
    pub fn representative_oxygen_gas() -> Self {
        Self::new("O2")
    }
    /// Stub amino-acid pool graph.
    pub fn representative_amino_acid_pool() -> Self {
        Self::new("amino_acid_pool")
    }
    /// Stub nucleotide pool graph.
    pub fn representative_nucleotide_pool() -> Self {
        Self::new("nucleotide_pool")
    }
    /// Stub membrane precursor pool graph.
    pub fn representative_membrane_precursor_pool() -> Self {
        Self::new("membrane_precursor_pool")
    }
    /// Stub NAD+ (oxidized) graph.
    pub fn representative_nad_oxidized() -> Self {
        Self::new("NAD_oxidized")
    }
    /// Stub NADH (reduced) graph.
    pub fn representative_nad_reduced() -> Self {
        Self::new("NAD_reduced")
    }
    /// Stub coenzyme A graph.
    pub fn representative_coenzyme_a() -> Self {
        Self::new("CoA")
    }

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
    pub fn secondary_structure_at(
        &self,
        chain_id: &str,
        seq_num: i32,
    ) -> Option<SecondaryStructureKind> {
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
    pub amount_moles: f64,
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

    pub fn add_component(&mut self, molecule: EmbeddedMolecule, amount_moles: f64) {
        self.components.push(EmbeddedMaterialMixtureComponent {
            molecule,
            amount_moles,
        });
    }

    /// Apply a structural reaction (stub for quantum runtime).
    pub fn apply_structural_reaction(
        &mut self,
        _reactants: &[EmbeddedMolecule],
        _reaction: &StructuralReactionTemplate,
        _extent: f64,
        _quantum: crate::subatomic_quantum::QuantumChemistryConfig,
    ) -> Result<StructuralReactionResult, EmbeddedMaterialStructuralReactionError> {
        let combined = if let Some(first) = self.components.first() {
            first.molecule.clone()
        } else {
            let mut g = MoleculeGraph::new("empty");
            g.add_element(PeriodicElement::H);
            EmbeddedMolecule {
                graph: g,
                positions_angstrom: vec![[0.0, 0.0, 0.0]],
            }
        };
        Ok(StructuralReactionResult {
            event_result: StructuralReactionEventResult {
                quantum_delta: StructuralReactionQuantumDelta::zeroed(),
                combined_molecule: combined,
            },
        })
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
// Structural reaction result types (for quantum runtime)
// ---------------------------------------------------------------------------

/// Result of applying a structural reaction.
#[derive(Debug, Clone)]
pub struct StructuralReactionResult {
    pub event_result: StructuralReactionEventResult,
}

/// Event-level result with quantum delta and product molecule.
#[derive(Debug, Clone)]
pub struct StructuralReactionEventResult {
    pub quantum_delta: StructuralReactionQuantumDelta,
    pub combined_molecule: EmbeddedMolecule,
}

/// Quantum chemistry before/after pair for a structural reaction.
#[derive(Debug, Clone)]
pub struct StructuralReactionQuantumDelta {
    pub before: crate::subatomic_quantum::ExactDiagonalizationResult,
    pub after: crate::subatomic_quantum::ExactDiagonalizationResult,
}

impl StructuralReactionQuantumDelta {
    pub fn zeroed() -> Self {
        let empty = crate::subatomic_quantum::ExactDiagonalizationResult {
            energies_ev: vec![0.0],
            ground_state_vector: vec![1.0],
            basis_states: vec![0],
            expected_spatial_occupancies: Vec::new(),
            spatial_orbital_atom_indices: Vec::new(),
            spatial_one_particle_density_matrix: Vec::new(),
            expected_atom_effective_charges: Vec::new(),
            expected_dipole_moment_e_angstrom: [0.0; 3],
            expected_electron_count: 0.0,
            nuclear_repulsion_ev: 0.0,
            solver_tier: Default::default(),
        };
        Self {
            before: empty.clone(),
            after: empty,
        }
    }

    pub fn summary(&self) -> crate::substrate_ir::ReactionQuantumSummary {
        let before_e = self.before.energies_ev.first().copied().unwrap_or(0.0);
        let after_e = self.after.energies_ev.first().copied().unwrap_or(0.0);
        crate::substrate_ir::ReactionQuantumSummary {
            event_count: 1,
            ground_state_energy_delta_ev: (after_e - before_e) as f32,
            nuclear_repulsion_delta_ev: (self.after.nuclear_repulsion_ev
                - self.before.nuclear_repulsion_ev) as f32,
            net_formal_charge_delta: 0,
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
    fn element_from_symbol() {
        assert_eq!(
            PeriodicElement::from_symbol_or_name("C"),
            Some(PeriodicElement::C)
        );
        assert_eq!(
            PeriodicElement::from_symbol_or_name("fe"),
            Some(PeriodicElement::Fe)
        );
        assert_eq!(
            PeriodicElement::from_symbol_or_name("ZN"),
            Some(PeriodicElement::Zn)
        );
        assert_eq!(PeriodicElement::from_symbol_or_name("XX"), None);
    }

    #[test]
    fn element_from_name() {
        assert_eq!(
            PeriodicElement::from_symbol_or_name("Carbon"),
            Some(PeriodicElement::C)
        );
        assert_eq!(
            PeriodicElement::from_symbol_or_name("GOLD"),
            Some(PeriodicElement::Au)
        );
        assert_eq!(
            PeriodicElement::from_symbol_or_name("oganesson"),
            Some(PeriodicElement::Og)
        );
    }

    #[test]
    fn element_from_atomic_number() {
        assert_eq!(PeriodicElement::from_atomic_number(1), Some(PeriodicElement::H));
        assert_eq!(PeriodicElement::from_atomic_number(6), Some(PeriodicElement::C));
        assert_eq!(PeriodicElement::from_atomic_number(79), Some(PeriodicElement::Au));
        assert_eq!(PeriodicElement::from_atomic_number(118), Some(PeriodicElement::Og));
        assert_eq!(PeriodicElement::from_atomic_number(0), None);
        assert_eq!(PeriodicElement::from_atomic_number(119), None);
    }

    #[test]
    fn element_mass_reasonable() {
        assert!((PeriodicElement::C.mass_daltons() - 12.011).abs() < 0.01);
        assert!((PeriodicElement::Fe.mass_daltons() - 55.845).abs() < 0.01);
    }

    #[test]
    fn all_118_elements_present() {
        let count = PeriodicElement::all().count();
        assert_eq!(count, 118);
        // First and last
        let first = PeriodicElement::all().next().unwrap();
        assert_eq!(first, PeriodicElement::H);
        let last = PeriodicElement::all().last().unwrap();
        assert_eq!(last, PeriodicElement::Og);
    }

    #[test]
    fn all_elements_have_correct_atomic_numbers() {
        for (i, elem) in PeriodicElement::all().enumerate() {
            assert_eq!(
                elem.atomic_number() as usize,
                i + 1,
                "{}: expected Z={}, got Z={}",
                elem.symbol(),
                i + 1,
                elem.atomic_number()
            );
        }
    }

    #[test]
    fn all_elements_roundtrip_atomic_number() {
        for elem in PeriodicElement::all() {
            let z = elem.atomic_number();
            assert_eq!(
                PeriodicElement::from_atomic_number(z),
                Some(elem),
                "roundtrip failed for Z={z} ({})",
                elem.symbol()
            );
        }
    }

    #[test]
    fn all_elements_roundtrip_symbol() {
        for elem in PeriodicElement::all() {
            let sym = elem.symbol();
            assert_eq!(
                PeriodicElement::from_symbol_or_name(sym),
                Some(elem),
                "roundtrip failed for symbol '{sym}'"
            );
        }
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
        g.add_atom(
            PeriodicElement::N,
            "N",
            Some(ResidueInfo {
                name: "ALA".to_string(),
                chain_id: "A".to_string(),
                seq_num: 1,
                ins_code: String::new(),
            }),
        );
        g.add_atom(
            PeriodicElement::C,
            "CA",
            Some(ResidueInfo {
                name: "ALA".to_string(),
                chain_id: "A".to_string(),
                seq_num: 1,
                ins_code: String::new(),
            }),
        );
        g.add_atom(
            PeriodicElement::N,
            "N",
            Some(ResidueInfo {
                name: "GLY".to_string(),
                chain_id: "B".to_string(),
                seq_num: 1,
                ins_code: String::new(),
            }),
        );
        g.add_bond(0, 1, BondOrder::Single).unwrap();
        let mol = EmbeddedMolecule::new(g, vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0]])
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
        let mol =
            EmbeddedMolecule::new(g, vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
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
        g.add_atom(
            PeriodicElement::N,
            "N",
            Some(ResidueInfo {
                name: "ALA".to_string(),
                chain_id: "A".to_string(),
                seq_num: 5,
                ins_code: String::new(),
            }),
        );
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

    #[test]
    fn test_vdw_radii_physical_range() {
        for elem in PeriodicElement::all() {
            let r = elem.van_der_waals_radius_angstrom();
            assert!(
                (1.0..=3.5).contains(&r),
                "{}: VDW radius {r} outside [1.0, 3.5] A",
                elem.symbol()
            );
        }
    }

    #[test]
    fn test_cpk_colors_nonzero() {
        for elem in PeriodicElement::all() {
            let [r, g, b] = elem.cpk_color_rgb();
            assert!(
                r > 0 || g > 0 || b > 0,
                "{}: CPK color is pure black [0,0,0]",
                elem.symbol()
            );
        }
    }

    #[test]
    fn test_electron_config_format() {
        for elem in PeriodicElement::all() {
            let config = elem.electron_configuration_short();
            assert!(
                !config.is_empty(),
                "{}: empty electron configuration",
                elem.symbol()
            );
            assert!(
                config.contains('s') || config.contains('p')
                    || config.contains('d') || config.contains('f'),
                "{}: config '{config}' has no orbital label",
                elem.symbol()
            );
        }
    }

    #[test]
    fn test_atomic_detail_carbon() {
        let c = PeriodicElement::C;
        assert_eq!(c.atomic_number(), 6);
        assert!((c.van_der_waals_radius_angstrom() - 1.70).abs() < 0.01);
        assert_eq!(c.cpk_color_rgb(), [80, 80, 80]);
        assert_eq!(c.electron_configuration_short(), "[He]2s2 2p2");
    }

    #[test]
    fn test_new_elements_gold_and_uranium() {
        let au = PeriodicElement::Au;
        assert_eq!(au.atomic_number(), 79);
        assert_eq!(au.symbol(), "Au");
        assert_eq!(au.name(), "Gold");
        assert!((au.mass_daltons() - 196.967).abs() < 0.01);
        assert_eq!(au.cpk_color_rgb(), [255, 209, 35]);

        let u = PeriodicElement::U;
        assert_eq!(u.atomic_number(), 92);
        assert_eq!(u.symbol(), "U");
        assert_eq!(u.name(), "Uranium");
        assert!((u.mass_daltons() - 238.029).abs() < 0.01);
    }

    #[test]
    fn test_uff_lj_params_reasonable() {
        for elem in PeriodicElement::all() {
            let (sigma, epsilon) = elem.uff_lj_params();
            assert!(sigma > 0.5 && sigma < 4.0,
                "{}: sigma={sigma} outside [0.5, 4.0]", elem.symbol());
            assert!(epsilon > 0.0 && epsilon < 1.0,
                "{}: epsilon={epsilon} outside (0, 1.0)", elem.symbol());
        }
    }
}
