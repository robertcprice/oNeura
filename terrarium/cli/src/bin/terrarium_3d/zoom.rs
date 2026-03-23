//! Multi-scale zoom detail rendering.
//!
//! At each zoom level, generates appropriate detail geometry for the selected entity:
//! - Ecosystem: standard terrain + entity meshes (handled by main loop)
//! - Organism: detailed body view — plant tissues, fly body segments
//! - Cellular: metabolite pools as color-coded spheres
//! - Molecular: substrate chemistry as a color-coded sphere cloud

use super::math::*;
use super::mesh::*;
use super::selection::Selection;
use oneura_core::plant_cellular::PlantTissue;
use oneura_core::terrarium::TerrariumWorld;

/// Metabolite pool color mapping.
fn metabolite_color(name: &str) -> V3 {
    match name {
        "ATP" => [1.0, 0.85, 0.0],
        "ADP" => [0.85, 0.65, 0.0],
        "Glucose" => [0.2, 0.7, 1.0],
        "O2" => [0.9, 0.2, 0.2],
        "CO2" => [0.6, 0.6, 0.6],
        "Trehalose" => [0.3, 0.9, 0.6],
        "Glycogen" => [0.9, 0.6, 0.2],
        "Lipid" => [0.95, 0.95, 0.3],
        "Water" => [0.30, 0.60, 0.95],
        "Energy" => [1.0, 0.7, 0.0],
        "Speed" => [0.3, 0.8, 1.0],
        "Altitude" => [0.6, 0.9, 0.5],
        _ => [0.7, 0.7, 0.7],
    }
}

/// Build detail meshes for Organism zoom level based on selected entity.
pub fn build_organism_detail(world: &TerrariumWorld, sel: &Selection, focus: V3) -> Vec<Triangle> {
    let mut tris = Vec::new();
    match sel.tag {
        EntityTag::Plant(idx) => {
            if idx < world.plants.len() {
                let plant = &world.plants[idx];
                let tissues = [
                    (PlantTissue::Leaf, [0.20, 0.75, 0.20], v3(0.0, 0.15, 0.0)),
                    (PlantTissue::Stem, [0.47, 0.31, 0.16], v3(0.0, 0.0, 0.0)),
                    (PlantTissue::Root, [0.55, 0.40, 0.25], v3(0.0, -0.12, 0.0)),
                    (
                        PlantTissue::Meristem,
                        [0.85, 0.92, 0.35],
                        v3(0.0, 0.22, 0.0),
                    ),
                ];
                for (i, (tissue, base_color, offset)) in tissues.iter().enumerate() {
                    let cluster = plant.cellular.cluster_snapshot(*tissue);
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
            if idx < world.flies.len() {
                let b = world.flies[idx].body_state();
                // Show fly body segments scaled by energy
                let segments: [(&str, V3, V3, f32); 4] = [
                    ("Head", [0.85, 0.75, 0.20], v3(0.0, 0.02, 0.04), 0.02),
                    ("Thorax", [0.80, 0.70, 0.15], v3(0.0, 0.02, 0.0), 0.03),
                    ("Abdomen", [0.75, 0.65, 0.10], v3(0.0, 0.0, -0.04), 0.025),
                    ("Wings", [0.90, 0.92, 0.85], v3(0.0, 0.04, 0.0), 0.02),
                ];
                for (i, (_name, color, offset, base_size)) in segments.iter().enumerate() {
                    let pos = add3(focus, *offset);
                    let energy_scale = b.energy.clamp(0.1, 1.0);
                    let s = base_size * (0.5 + energy_scale * 0.5);
                    let mut sphere = make_sphere(pos, s, *color, 24.0);
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
                let tissues = [
                    PlantTissue::Leaf,
                    PlantTissue::Stem,
                    PlantTissue::Root,
                    PlantTissue::Meristem,
                ];
                for (ti, tissue) in tissues.iter().enumerate() {
                    let cluster = plant.cellular.cluster_snapshot(*tissue);
                    let tissue_offset = v3(0.0, (ti as f32 - 1.5) * 0.05, 0.0);
                    let metabolites: [(&str, f32, f32); 3] = [
                        ("ATP", cluster.state_atp, 10.0),
                        ("Glucose", cluster.state_glucose, 5.0),
                        ("Water", cluster.state_water, 5.0),
                    ];
                    for (mi, (name, val, max_val)) in metabolites.iter().enumerate() {
                        let fill = (*val / *max_val).clamp(0.0, 1.0);
                        if fill < 0.01 {
                            continue;
                        }
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
            let fly_metas = &world.fly_metabolisms;
            if idx < fly_metas.len() {
                let meta = &fly_metas[idx];
                // 7-pool Michaelis-Menten metabolite spheres
                let pools: [(&str, f32, f32); 7] = [
                    ("Glucose", meta.hemolymph_glucose_mm, 10.0),
                    ("Trehalose", meta.hemolymph_trehalose_mm, 40.0),
                    ("Glycogen", meta.fat_body_glycogen_mg, 0.03),
                    ("Lipid", meta.fat_body_lipid_mg, 0.09),
                    ("ATP", meta.muscle_atp_mm, 8.0),
                    ("ADP", meta.muscle_adp_mm, 8.0),
                    ("Energy", meta.crop_sugar_mg, 0.5),
                ];
                for (i, (name, value, max_val)) in pools.iter().enumerate() {
                    let fill = (*value / max_val).clamp(0.0, 1.0);
                    if fill < 0.01 {
                        continue;
                    }
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
            } else if idx < world.flies.len() {
                // Fallback: use body_state if no metabolism data
                let b = world.flies[idx].body_state();
                let pools: [(&str, f32, f32); 3] = [
                    ("Energy", b.energy, 1.0),
                    ("Speed", b.speed, 5.0),
                    ("Altitude", b.z.max(0.0), 4.0),
                ];
                for (i, (name, value, max_val)) in pools.iter().enumerate() {
                    let fill = (*value / max_val).clamp(0.0, 1.0);
                    if fill < 0.01 {
                        continue;
                    }
                    let angle = (i as f32 / pools.len() as f32) * std::f32::consts::TAU;
                    let r = 0.03;
                    let offset = v3(angle.cos() * r, (i as f32 - 1.0) * 0.01, angle.sin() * r);
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

/// CPK element color table.
fn atom_color(element_idx: usize) -> V3 {
    match element_idx {
        1 => [1.0, 1.0, 1.0],     // H - white
        6 => [0.30, 0.30, 0.30],  // C - dark gray
        7 => [0.20, 0.20, 0.90],  // N - blue
        8 => [0.90, 0.15, 0.15],  // O - red
        15 => [1.0, 0.50, 0.0],   // P - orange
        16 => [1.0, 1.0, 0.20],   // S - yellow
        26 => [0.80, 0.40, 0.10], // Fe - brown
        12 => [0.0, 0.80, 0.0],   // Mg - green
        _ => [0.60, 0.60, 0.60],  // default - silver
    }
}

/// Van der Waals radius approximations (scaled for rendering).
fn atom_radius(element_idx: usize) -> f32 {
    match element_idx {
        1 => 0.015,  // H
        6 => 0.025,  // C
        7 => 0.024,  // N
        8 => 0.023,  // O
        15 => 0.030, // P
        16 => 0.028, // S
        26 => 0.035, // Fe
        _ => 0.025,
    }
}

/// Infer element from atomic mass.
fn element_from_mass(mass: f32) -> usize {
    if mass < 2.0 {
        1
    }
    // H
    else if mass < 14.0 {
        6
    }
    // C
    else if mass < 15.0 {
        7
    }
    // N
    else if mass < 18.0 {
        8
    }
    // O
    else if mass < 33.0 {
        15
    }
    // P
    else if mass < 34.0 {
        16
    }
    // S
    else if mass < 60.0 {
        26
    }
    // Fe
    else {
        6
    }
}

/// Build detail meshes for Molecular zoom level — atomistic probe atoms + bonds,
/// or substrate species as a color-coded ring if no probes exist.
pub fn build_molecular_detail(
    world: &TerrariumWorld,
    _sel: &Selection,
    focus: V3,
) -> Vec<Triangle> {
    let mut tris = Vec::new();
    let probes = world.probes();

    if !probes.is_empty() {
        let probe = &probes[0];
        let positions = probe.md.positions();
        let n_atoms = probe.md.n_atoms();
        let masses = probe.md.masses();
        let bonds = probe.md.bonds();
        let scale = 0.001; // Angstroms → world units

        // Render atoms with CPK coloring
        for atom_idx in 0..n_atoms {
            let px = positions.get(atom_idx * 3).copied().unwrap_or(0.0);
            let py = positions.get(atom_idx * 3 + 1).copied().unwrap_or(0.0);
            let pz = positions.get(atom_idx * 3 + 2).copied().unwrap_or(0.0);
            let mass = masses.get(atom_idx).copied().unwrap_or(12.0);
            let element = element_from_mass(mass);
            let world_pos = add3(focus, v3(px * scale, py * scale, pz * scale));
            let color = atom_color(element);
            let radius = atom_radius(element);
            let mut atom_mesh = make_sphere(world_pos, radius, color, 48.0);
            tag_all(&mut atom_mesh, EntityTag::Atom(atom_idx));
            tris.extend(atom_mesh);
        }

        // Render bonds as gray cylinders
        for (bi, bond) in bonds.iter().enumerate() {
            let ai = bond.i;
            let aj = bond.j;
            if ai >= n_atoms || aj >= n_atoms {
                continue;
            }
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
        let substrate_species: [(&str, V3); 12] = [
            ("Carbon", [0.30, 0.30, 0.30]),
            ("Hydrogen", [0.90, 0.90, 0.90]),
            ("Oxygen", [0.90, 0.15, 0.15]),
            ("Nitrogen", [0.20, 0.20, 0.90]),
            ("Water", [0.30, 0.60, 0.95]),
            ("Glucose", [0.20, 0.70, 1.0]),
            ("O2", [0.95, 0.30, 0.30]),
            ("NH4+", [0.20, 0.50, 0.90]),
            ("NO3-", [0.40, 0.40, 0.90]),
            ("CO2", [0.60, 0.60, 0.60]),
            ("H+", [1.0, 0.40, 0.80]),
            ("ATP", [1.0, 0.85, 0.0]),
        ];
        for (i, (_name, color)) in substrate_species.iter().enumerate() {
            let angle = (i as f32 / substrate_species.len() as f32) * std::f32::consts::TAU;
            let r = 0.015;
            let pos = add3(
                focus,
                v3(
                    angle.cos() * r,
                    (i as f32 % 3.0) * 0.005 - 0.005,
                    angle.sin() * r,
                ),
            );
            let mut sphere = make_sphere(pos, 0.003, *color, 32.0);
            tag_all(&mut sphere, EntityTag::Metabolite(i));
            tris.extend(sphere);
        }
    }
    tris
}

/// Build translucent chemistry overlay spheres showing local substrate concentrations.
/// Samples a 5x5 grid of voxels around the focus position and renders them as
/// color-coded semi-transparent spheres sized by concentration.
pub fn build_chemistry_overlay(world: &TerrariumWorld, focus: V3, cell_size: f32) -> Vec<Triangle> {
    let mut tris = Vec::new();
    let gw = world.config.width;
    let gh = world.config.height;

    // Center grid position
    let cx = (focus[0] / cell_size).round() as i32;
    let cz = (focus[2] / cell_size).round() as i32;

    // Species to visualize with their colors
    use oneura_core::terrarium::TerrariumSpecies;
    let species_vis: [(TerrariumSpecies, V3, f32); 4] = [
        (TerrariumSpecies::Glucose, [0.2, 0.7, 1.0], 2.0),
        (TerrariumSpecies::OxygenGas, [0.9, 0.3, 0.3], 1.0),
        (TerrariumSpecies::CarbonDioxide, [0.6, 0.6, 0.6], 0.5),
        (TerrariumSpecies::AtpFlux, [1.0, 0.85, 0.0], 0.3),
    ];

    for dx in -2..=2 {
        for dz in -2..=2 {
            let gx = (cx + dx).clamp(0, gw as i32 - 1) as usize;
            let gz = (cz + dz).clamp(0, gh as i32 - 1) as usize;

            for (si, (species, color, max_val)) in species_vis.iter().enumerate() {
                // Read species value using species_field and index
                let field = world.substrate.species_field(*species);
                let idx = world.substrate.index(*species, gx, 0, gz);
                let val = field.get(idx).copied().unwrap_or(0.0);

                let fill = (val / max_val).clamp(0.0, 1.0);
                if fill < 0.05 {
                    continue;
                }

                let wx = gx as f32 * cell_size;
                let wz = gz as f32 * cell_size;
                let layer_y = si as f32 * 0.03;
                let pos = v3(wx, focus[1] + 0.02 + layer_y, wz);
                let size = 0.01 + fill * 0.03;
                let mut sphere = make_sphere(pos, size, *color, 16.0);
                tag_all(
                    &mut sphere,
                    EntityTag::Metabolite(100 + si * 25 + gx % 5 * 5 + gz % 5),
                );
                tris.extend(sphere);
            }
        }
    }
    tris
}
