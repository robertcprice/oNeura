//! Multi-scale zoom detail rendering.
//!
//! At each zoom level, generates appropriate detail geometry for the selected entity:
//! - Ecosystem: standard terrain + entity meshes (handled by main loop)
//! - Organism: detailed body view — plant tissues, fly body segments
//! - Cellular: metabolite pools as color-coded spheres, organelle layout
//! - Molecular: atom positions as spheres, bonds as cylinders from atomistic probes

use oneuro_metal::TerrariumWorld;
use super::math::*;
use super::mesh::*;
use super::camera::ZoomLevel;
use super::selection::Selection;

/// Element color table (CPK coloring convention).
fn atom_color(element_idx: usize) -> V3 {
    match element_idx {
        1  => [1.0, 1.0, 1.0],       // H — white
        6  => [0.30, 0.30, 0.30],     // C — dark gray
        7  => [0.20, 0.20, 0.90],     // N — blue
        8  => [0.90, 0.15, 0.15],     // O — red
        15 => [1.0, 0.50, 0.0],       // P — orange
        16 => [1.0, 1.0, 0.20],       // S — yellow
        26 => [0.80, 0.40, 0.10],     // Fe — brown
        12 => [0.0, 0.80, 0.0],       // Mg — green
        _  => [0.60, 0.60, 0.60],     // default — silver
    }
}

/// Van der Waals radius approximations (Angstroms, scaled for rendering).
fn atom_radius(element_idx: usize) -> f32 {
    match element_idx {
        1  => 0.015,  // H — small
        6  => 0.025,  // C
        7  => 0.024,  // N
        8  => 0.023,  // O
        15 => 0.030,  // P — larger
        16 => 0.028,  // S
        26 => 0.035,  // Fe — large
        _  => 0.025,  // default
    }
}

/// Metabolite pool color mapping.
fn metabolite_color(name: &str) -> V3 {
    match name {
        "ATP" | "atp"       => [1.0, 0.85, 0.0],     // gold
        "ADP" | "adp"       => [0.85, 0.65, 0.0],     // darker gold
        "Glucose" | "glucose" => [0.2, 0.7, 1.0],      // blue
        "O2" | "oxygen"      => [0.9, 0.2, 0.2],      // red
        "CO2" | "co2"        => [0.6, 0.6, 0.6],      // gray
        "Trehalose"          => [0.3, 0.9, 0.6],      // teal
        "Glycogen"           => [0.9, 0.6, 0.2],      // orange
        "Lipid"              => [0.95, 0.95, 0.3],    // yellow
        _                     => [0.7, 0.7, 0.7],
    }
}

/// Build detail meshes for Organism zoom level based on selected entity.
pub fn build_organism_detail(world: &TerrariumWorld, sel: &Selection, focus: V3) -> Vec<Triangle> {
    let mut tris = Vec::new();
    match sel.tag {
        EntityTag::Plant(idx) => {
            if idx < world.plants.len() {
                let plant = &world.plants[idx];
                // Show tissue clusters as 4 colored spheres around the plant
                let tissues = [
                    ("Leaf",     [0.20, 0.75, 0.20], v3(0.0, 0.15, 0.0)),
                    ("Stem",     [0.47, 0.31, 0.16], v3(0.0, 0.0, 0.0)),
                    ("Root",     [0.55, 0.40, 0.25], v3(0.0, -0.12, 0.0)),
                    ("Meristem", [0.85, 0.92, 0.35], v3(0.0, 0.22, 0.0)),
                ];
                let snap = plant.cellular.snapshot();
                for (i, (name, base_color, offset)) in tissues.iter().enumerate() {
                    let cluster = &snap[i];
                    let vitality = cluster.vitality;
                    let size = (cluster.cell_count * 0.002).clamp(0.03, 0.2);
                    let color = lerp3(*base_color, [0.3, 0.15, 0.05], 1.0 - vitality);
                    let pos = add3(focus, *offset);
                    let mut sphere = make_sphere(pos, size, color, 16.0);
                    tag_all(&mut sphere, EntityTag::Metabolite(i));
                    tris.extend(sphere);
                }
            }
        }
        EntityTag::Fly(idx) => {
            if idx < world.flies.len() && idx < world.fly_metabolisms.len() {
                let meta = &world.fly_metabolisms[idx];
                // Show 7 metabolism pools as spheres arranged in a body layout
                let pools: [(&str, f32, f32, V3); 7] = [
                    ("Crop",       meta.crop_sugar_mg,           0.5,  v3(-0.08, 0.0, 0.06)),
                    ("Trehalose",  meta.hemolymph_trehalose_mm,  40.0, v3(0.0, 0.02, 0.03)),
                    ("HemoGluc",   meta.hemolymph_glucose_mm,    10.0, v3(0.0, 0.02, -0.03)),
                    ("Glycogen",   meta.fat_body_glycogen_mg,    0.03, v3(0.05, 0.0, 0.0)),
                    ("Lipid",      meta.fat_body_lipid_mg,       0.09, v3(-0.05, 0.0, 0.0)),
                    ("ATP",        meta.muscle_atp_mm,           8.0,  v3(0.0, -0.02, 0.06)),
                    ("ADP",        meta.muscle_adp_mm,           8.0,  v3(0.0, -0.02, -0.06)),
                ];
                for (i, (name, value, max_val, offset)) in pools.iter().enumerate() {
                    let fill = (*value / max_val).clamp(0.0, 1.0);
                    let size = 0.02 + fill * 0.03;
                    let color = metabolite_color(name);
                    let pos = add3(focus, *offset);
                    let mut sphere = make_sphere(pos, size, color, 24.0);
                    tag_all(&mut sphere, EntityTag::Metabolite(i));
                    tris.extend(sphere);
                }
            }
        }
        _ => {}
    }
    tris
}

/// Build detail meshes for Cellular zoom level — metabolite concentrations as spheres.
pub fn build_cellular_detail(world: &TerrariumWorld, sel: &Selection, focus: V3) -> Vec<Triangle> {
    let mut tris = Vec::new();
    match sel.tag {
        EntityTag::Plant(idx) => {
            if idx < world.plants.len() {
                let plant = &world.plants[idx];
                let snap = plant.cellular.snapshot();
                // Per-tissue: show individual metabolite pools
                for (ti, cluster) in snap.iter().enumerate() {
                    let tissue_offset = v3(0.0, (ti as f32 - 1.5) * 0.05, 0.0);
                    let metabolites = [
                        ("ATP",     cluster.state_atp,     10.0),
                        ("Glucose", cluster.state_glucose,  5.0),
                        ("O2",      cluster.state_water,    5.0),
                    ];
                    for (mi, (name, val, max_val)) in metabolites.iter().enumerate() {
                        let fill = (val / max_val).clamp(0.0, 1.0);
                        if fill < 0.01 { continue; }
                        let color = metabolite_color(name);
                        let angle = (mi as f32 / metabolites.len() as f32) * std::f32::consts::TAU;
                        let r = 0.04;
                        let offset = add3(tissue_offset, v3(angle.cos() * r, 0.0, angle.sin() * r));
                        let pos = add3(focus, offset);
                        let size = 0.005 + fill * 0.015;
                        let mut sphere = make_sphere(pos, size, color, 32.0);
                        tag_all(&mut sphere, EntityTag::Metabolite(ti * 10 + mi));
                        tris.extend(sphere);
                    }
                }
            }
        }
        EntityTag::Fly(idx) => {
            if idx < world.fly_metabolisms.len() {
                let meta = &world.fly_metabolisms[idx];
                // Detailed metabolite view with sub-pools
                let pools: [(&str, f32, f32); 7] = [
                    ("Crop",       meta.crop_sugar_mg,           0.5),
                    ("Trehalose",  meta.hemolymph_trehalose_mm,  40.0),
                    ("Glucose",    meta.hemolymph_glucose_mm,    10.0),
                    ("Glycogen",   meta.fat_body_glycogen_mg,    0.03),
                    ("Lipid",      meta.fat_body_lipid_mg,       0.09),
                    ("ATP",        meta.muscle_atp_mm,           8.0),
                    ("ADP",        meta.muscle_adp_mm,           8.0),
                ];
                for (i, (name, value, max_val)) in pools.iter().enumerate() {
                    let fill = (*value / max_val).clamp(0.0, 1.0);
                    if fill < 0.01 { continue; }
                    let angle = (i as f32 / pools.len() as f32) * std::f32::consts::TAU;
                    let r = 0.03;
                    let offset = v3(angle.cos() * r, (i as f32 - 3.0) * 0.01, angle.sin() * r);
                    let pos = add3(focus, offset);
                    let size = 0.003 + fill * 0.01;
                    let color = metabolite_color(name);
                    let mut sphere = make_sphere(pos, size, color, 32.0);
                    tag_all(&mut sphere, EntityTag::Metabolite(i));
                    tris.extend(sphere);
                }
            }
        }
        _ => {}
    }
    tris
}

/// Build detail meshes for Molecular zoom level — atomistic probe atoms + bonds.
pub fn build_molecular_detail(world: &TerrariumWorld, sel: &Selection, focus: V3) -> Vec<Triangle> {
    let mut tris = Vec::new();

    // If we have atomistic probes, render them
    if !world.atomistic_probes.is_empty() {
        let probe = &world.atomistic_probes[0]; // Closest probe
        let positions = probe.md.positions();
        let n_atoms = probe.md.n_atoms;
        let scale = 0.001; // Angstroms → world units

        // Render atoms as spheres
        for atom_idx in 0..n_atoms {
            let px = positions.get(atom_idx * 3).copied().unwrap_or(0.0);
            let py = positions.get(atom_idx * 3 + 1).copied().unwrap_or(0.0);
            let pz = positions.get(atom_idx * 3 + 2).copied().unwrap_or(0.0);

            // Determine element from mass (approximate)
            let mass = probe.md.masses.get(atom_idx).copied().unwrap_or(12.0);
            let element = if mass < 2.0 { 1 }       // H
                         else if mass < 14.0 { 6 }   // C
                         else if mass < 15.0 { 7 }   // N
                         else if mass < 18.0 { 8 }   // O
                         else if mass < 33.0 { 15 }   // P
                         else if mass < 34.0 { 16 }   // S
                         else { 6 };                   // default C

            let world_pos = add3(focus, v3(px * scale, py * scale, pz * scale));
            let color = atom_color(element);
            let radius = atom_radius(element);
            let mut atom_mesh = make_sphere(world_pos, radius, color, 48.0);
            tag_all(&mut atom_mesh, EntityTag::Atom(atom_idx));
            tris.extend(atom_mesh);
        }

        // Render bonds as cylinders
        for (bi, bond) in probe.md.bonds.iter().enumerate() {
            let ai = bond.0;
            let aj = bond.1;
            if ai >= n_atoms || aj >= n_atoms { continue; }
            let p0 = v3(
                positions.get(ai * 3).copied().unwrap_or(0.0) * scale,
                positions.get(ai * 3 + 1).copied().unwrap_or(0.0) * scale,
                positions.get(ai * 3 + 2).copied().unwrap_or(0.0) * scale,
            );
            let p1 = v3(
                positions.get(aj * 3).copied().unwrap_or(0.0) * scale,
                positions.get(aj * 3 + 1).copied().unwrap_or(0.0) * scale,
                positions.get(aj * 3 + 2).copied().unwrap_or(0.0) * scale,
            );
            let world_p0 = add3(focus, p0);
            let world_p1 = add3(focus, p1);
            let mut bond_mesh = make_cylinder(world_p0, world_p1, 0.004, [0.5, 0.5, 0.5], 16.0);
            tag_all(&mut bond_mesh, EntityTag::Bond(bi));
            tris.extend(bond_mesh);
        }
    } else {
        // No atomistic probes — render substrate chemistry as a molecular cloud
        // Show the 14 substrate species as colored spheres in a grid pattern
        let substrate_species = [
            ("Carbon",  [0.30, 0.30, 0.30]),
            ("Hydrogen", [0.90, 0.90, 0.90]),
            ("Oxygen",  [0.90, 0.15, 0.15]),
            ("Nitrogen", [0.20, 0.20, 0.90]),
            ("Water",   [0.30, 0.60, 0.95]),
            ("Glucose", [0.20, 0.70, 1.0]),
            ("O2",      [0.95, 0.30, 0.30]),
            ("NH4+",    [0.20, 0.50, 0.90]),
            ("NO3-",    [0.40, 0.40, 0.90]),
            ("CO2",     [0.60, 0.60, 0.60]),
            ("H+",      [1.0, 0.40, 0.80]),
            ("ATP",     [1.0, 0.85, 0.0]),
        ];
        for (i, (name, color)) in substrate_species.iter().enumerate() {
            let angle = (i as f32 / substrate_species.len() as f32) * std::f32::consts::TAU;
            let r = 0.015;
            let pos = add3(focus, v3(angle.cos() * r, (i as f32 % 3.0) * 0.005 - 0.005, angle.sin() * r));
            let mut sphere = make_sphere(pos, 0.003, *color, 32.0);
            tag_all(&mut sphere, EntityTag::Metabolite(i));
            tris.extend(sphere);
        }
    }
    tris
}
