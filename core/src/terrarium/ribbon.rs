//! Ribbon/tube geometry generation for tree branches and protein backbones.
//!
//! Ported from ProteinView (github.com/001TMF/ProteinView) and adapted for
//! tree branch rendering with tapering cross-sections.
//!
//! Pipeline:
//! 1. Extract branch chains from morphology skeleton
//! 2. Catmull-Rom spline interpolation through control points
//! 3. Parallel-transport frame propagation (twist-free)
//! 4. Circular cross-section extrusion with radius tapering
//! 5. Triangle strip emission between consecutive cross-sections
//! 6. End caps
//!
//! The output is a flat triangle list suitable for both WebGL serialization
//! and terminal ASCII 3D rendering.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls the quality/detail of ribbon geometry.
pub struct RibbonConfig {
    /// Catmull-Rom subdivisions between each pair of control points.
    pub subdivisions: usize,
    /// Number of vertices around the circular cross-section.
    pub radial_segments: usize,
    /// Minimum taper ratio at branch tips (0.1 = 10% of base radius).
    pub min_taper_ratio: f32,
    /// Whether to generate end caps.
    pub caps: bool,
}

impl Default for RibbonConfig {
    fn default() -> Self {
        Self {
            subdivisions: 6,
            radial_segments: 8,
            min_taper_ratio: 0.15,
            caps: true,
        }
    }
}

impl RibbonConfig {
    /// Low-quality config for terminal rendering.
    pub fn terminal() -> Self {
        Self {
            subdivisions: 3,
            radial_segments: 5,
            min_taper_ratio: 0.15,
            caps: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// A single triangle in the ribbon mesh.
#[derive(Debug, Clone, Serialize)]
pub struct RibbonTriangle {
    /// Three vertices in 3D world space, each `[x, y, z]`.
    pub verts: [[f32; 3]; 3],
    /// Per-vertex normals for smooth shading.
    pub normals: [[f32; 3]; 3],
    /// RGB color [0-255].
    pub color: [u8; 3],
    /// Branch depth [0.0 = trunk base, 1.0 = tip] for wind animation.
    pub branch_depth: f32,
}

/// Pre-built mesh data for WebGL buffer upload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RibbonMeshData {
    /// Flat array of vertex positions [x,y,z, x,y,z, ...].
    pub positions: Vec<f32>,
    /// Flat array of vertex normals [nx,ny,nz, ...].
    pub normals: Vec<f32>,
    /// Triangle indices.
    pub indices: Vec<u32>,
    /// Per-vertex branch depth for wind shader animation.
    pub branch_depth: Vec<f32>,
    /// Per-vertex color [r,g,b, r,g,b, ...] normalized 0-1.
    pub colors: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Vec3 helpers (f32)
// ---------------------------------------------------------------------------

type V3 = [f32; 3];

#[inline]
fn v3_add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn v3_sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn v3_scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn v3_dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn v3_cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn v3_len(a: V3) -> f32 {
    v3_dot(a, a).sqrt()
}

#[inline]
fn v3_normalize(a: V3) -> V3 {
    let l = v3_len(a);
    if l < 1e-7 {
        [0.0, 1.0, 0.0]
    } else {
        v3_scale(a, 1.0 / l)
    }
}

// ---------------------------------------------------------------------------
// Catmull-Rom spline (ported from ProteinView ribbon.rs:225)
// ---------------------------------------------------------------------------

/// Evaluate Catmull-Rom spline between p1 and p2 at parameter t in [0,1].
fn catmull_rom(p0: V3, p1: V3, p2: V3, p3: V3, t: f32) -> V3 {
    let t2 = t * t;
    let t3 = t2 * t;
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        out[i] = 0.5
            * ((2.0 * p1[i])
                + (-p0[i] + p2[i]) * t
                + (2.0 * p0[i] - 5.0 * p1[i] + 4.0 * p2[i] - p3[i]) * t2
                + (-p0[i] + 3.0 * p1[i] - 3.0 * p2[i] + p3[i]) * t3);
    }
    out
}

/// Evaluate Catmull-Rom on a scalar (for radius interpolation).
fn catmull_rom_scalar(r0: f32, r1: f32, r2: f32, r3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * r1)
        + (-r0 + r2) * t
        + (2.0 * r0 - 5.0 * r1 + 4.0 * r2 - r3) * t2
        + (-r0 + 3.0 * r1 - 3.0 * r2 + r3) * t3)
}

// ---------------------------------------------------------------------------
// Spline point with local frame
// ---------------------------------------------------------------------------

struct SplinePoint {
    pos: V3,
    tangent: V3,
    normal: V3,
    binormal: V3,
    radius: f32,
    branch_depth: f32, // 0.0 at base, 1.0 at tip
    color: [u8; 3],
}

// ---------------------------------------------------------------------------
// Branch control point
// ---------------------------------------------------------------------------

/// A control point for a branch chain.
pub struct BranchControlPoint {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [u8; 3],
    /// 0.0 at trunk base, 1.0 at branch tips.
    pub branch_depth: f32,
}

// ---------------------------------------------------------------------------
// Core ribbon builder
// ---------------------------------------------------------------------------

/// Build a smooth tube mesh through a sequence of branch control points.
///
/// Uses Catmull-Rom spline interpolation + parallel-transport frames
/// (ported from ProteinView build_spline_tube).
pub fn build_branch_tube(
    control_points: &[BranchControlPoint],
    config: &RibbonConfig,
) -> Vec<RibbonTriangle> {
    let n = control_points.len();
    if n < 2 {
        return Vec::new();
    }

    let mut out = Vec::new();

    // --- Step 1: Catmull-Rom spline interpolation ---
    let mut spline_points: Vec<SplinePoint> = Vec::new();

    for seg in 0..n - 1 {
        let i0 = if seg == 0 { 0 } else { seg - 1 };
        let i1 = seg;
        let i2 = seg + 1;
        let i3 = if seg + 2 >= n { n - 1 } else { seg + 2 };

        let p0 = control_points[i0].position;
        let p1 = control_points[i1].position;
        let p2 = control_points[i2].position;
        let p3 = control_points[i3].position;

        let r0 = control_points[i0].radius;
        let r1 = control_points[i1].radius;
        let r2 = control_points[i2].radius;
        let r3 = control_points[i3].radius;

        let subdivs = if seg == n - 2 {
            config.subdivisions + 1
        } else {
            config.subdivisions
        };

        for sub in 0..subdivs {
            let t = sub as f32 / config.subdivisions as f32;
            let pos = catmull_rom(p0, p1, p2, p3, t);
            let radius = catmull_rom_scalar(r0, r1, r2, r3, t)
                .max(config.min_taper_ratio * r1);

            // Interpolate branch_depth and color
            let d1 = control_points[i1].branch_depth;
            let d2 = control_points[i2].branch_depth;
            let branch_depth = d1 + (d2 - d1) * t;

            let color = if t < 0.5 {
                control_points[i1].color
            } else {
                control_points[i2].color
            };

            spline_points.push(SplinePoint {
                pos,
                tangent: [0.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                binormal: [0.0, 0.0, 1.0],
                radius,
                branch_depth,
                color,
            });
        }
    }

    if spline_points.len() < 2 {
        return out;
    }

    // --- Step 2: Finite-difference tangents ---
    let sp_len = spline_points.len();
    for i in 0..sp_len {
        let prev = if i == 0 { 0 } else { i - 1 };
        let next = if i == sp_len - 1 { sp_len - 1 } else { i + 1 };
        spline_points[i].tangent =
            v3_normalize(v3_sub(spline_points[next].pos, spline_points[prev].pos));
    }

    // --- Step 3: Parallel-transport frames (ProteinView lines 563-593) ---
    {
        let t0 = spline_points[0].tangent;
        let arbitrary = if t0[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let mut prev_normal = v3_normalize(v3_cross(t0, arbitrary));

        for sp in spline_points.iter_mut() {
            let t = sp.tangent;
            let proj = v3_scale(t, v3_dot(prev_normal, t));
            let mut nr = v3_sub(prev_normal, proj);
            let nl = v3_len(nr);
            if nl < 1e-7 {
                let arb = if t[0].abs() < 0.9 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                nr = v3_normalize(v3_cross(t, arb));
            } else {
                nr = v3_scale(nr, 1.0 / nl);
            }
            let b = v3_normalize(v3_cross(t, nr));
            sp.normal = nr;
            sp.binormal = b;
            prev_normal = nr;
        }
    }

    // --- Step 4: Cross-section extrusion + triangle strips ---
    let radial = config.radial_segments;
    let mut prev_ring = circular_cross_section(&spline_points[0], radial);

    for sp in spline_points.iter().skip(1) {
        let curr_ring = circular_cross_section(sp, radial);
        emit_strip(&prev_ring, &curr_ring, sp.color, sp.branch_depth, &mut out);
        prev_ring = curr_ring;
    }

    // --- Step 5: End caps ---
    if config.caps && spline_points.len() >= 2 {
        let first = &spline_points[0];
        let first_ring = circular_cross_section(first, radial);
        emit_cap(&first_ring, first.pos, first.color, first.branch_depth, false, &mut out);

        let last = spline_points.last().unwrap();
        let last_ring = circular_cross_section(last, radial);
        emit_cap(&last_ring, last.pos, last.color, last.branch_depth, true, &mut out);
    }

    out
}

// ---------------------------------------------------------------------------
// Cross-section generation
// ---------------------------------------------------------------------------

/// Generate a circular cross-section ring at a spline point.
fn circular_cross_section(sp: &SplinePoint, segments: usize) -> Vec<V3> {
    let n = sp.normal;
    let b = sp.binormal;
    let r = sp.radius;
    let mut pts = Vec::with_capacity(segments);
    for i in 0..segments {
        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
        let (sin_a, cos_a) = angle.sin_cos();
        let offset = v3_add(v3_scale(n, cos_a * r), v3_scale(b, sin_a * r));
        pts.push(v3_add(sp.pos, offset));
    }
    pts
}

// ---------------------------------------------------------------------------
// Triangle strip emission (ProteinView lines 336-373)
// ---------------------------------------------------------------------------

fn emit_strip(
    ring_a: &[V3],
    ring_b: &[V3],
    color: [u8; 3],
    branch_depth: f32,
    out: &mut Vec<RibbonTriangle>,
) {
    let n = ring_a.len();
    if n == 0 || ring_b.len() != n {
        return;
    }
    for i in 0..n {
        let j = (i + 1) % n;
        let a0 = ring_a[i];
        let a1 = ring_a[j];
        let b0 = ring_b[i];
        let b1 = ring_b[j];

        let n1 = v3_normalize(v3_cross(v3_sub(a1, a0), v3_sub(b0, a0)));
        out.push(RibbonTriangle {
            verts: [a0, a1, b0],
            normals: [n1, n1, n1],
            color,
            branch_depth,
        });

        let n2 = v3_normalize(v3_cross(v3_sub(b1, a1), v3_sub(b0, a1)));
        out.push(RibbonTriangle {
            verts: [a1, b1, b0],
            normals: [n2, n2, n2],
            color,
            branch_depth,
        });
    }
}

// ---------------------------------------------------------------------------
// End cap (ProteinView lines 378-403)
// ---------------------------------------------------------------------------

fn emit_cap(
    ring: &[V3],
    center: V3,
    color: [u8; 3],
    branch_depth: f32,
    facing_forward: bool,
    out: &mut Vec<RibbonTriangle>,
) {
    let n = ring.len();
    if n < 3 {
        return;
    }
    for i in 0..n {
        let j = (i + 1) % n;
        let (v0, v1) = if facing_forward {
            (ring[j], ring[i])
        } else {
            (ring[i], ring[j])
        };
        let norm = v3_normalize(v3_cross(v3_sub(v0, center), v3_sub(v1, center)));
        out.push(RibbonTriangle {
            verts: [center, v0, v1],
            normals: [norm, norm, norm],
            color,
            branch_depth,
        });
    }
}

// ---------------------------------------------------------------------------
// Convert triangle list to indexed mesh (for WebGL)
// ---------------------------------------------------------------------------

/// Convert flat triangle list to indexed mesh for efficient WebGL upload.
pub fn triangles_to_mesh(triangles: &[RibbonTriangle]) -> RibbonMeshData {
    let vert_count = triangles.len() * 3;
    let mut positions = Vec::with_capacity(vert_count * 3);
    let mut normals = Vec::with_capacity(vert_count * 3);
    let mut indices = Vec::with_capacity(vert_count);
    let mut branch_depth = Vec::with_capacity(vert_count);
    let mut colors = Vec::with_capacity(vert_count * 3);

    for (tri_idx, tri) in triangles.iter().enumerate() {
        for v in 0..3 {
            positions.push(tri.verts[v][0]);
            positions.push(tri.verts[v][1]);
            positions.push(tri.verts[v][2]);
            normals.push(tri.normals[v][0]);
            normals.push(tri.normals[v][1]);
            normals.push(tri.normals[v][2]);
            branch_depth.push(tri.branch_depth);
            colors.push(tri.color[0] as f32 / 255.0);
            colors.push(tri.color[1] as f32 / 255.0);
            colors.push(tri.color[2] as f32 / 255.0);
            indices.push((tri_idx * 3 + v) as u32);
        }
    }

    RibbonMeshData {
        positions,
        normals,
        indices,
        branch_depth,
        colors,
    }
}

// ---------------------------------------------------------------------------
// Build branch mesh from morphology nodes
// ---------------------------------------------------------------------------

use crate::botany::morphology::{MorphNode, NodeType};

/// Build smooth ribbon mesh for all trunk/branch nodes of a plant.
///
/// Extracts branch chains from the morphology node array by tracing
/// connected Trunk→Branch paths, then builds a parallel-transport tube
/// through each chain.
pub fn build_plant_branch_mesh(
    nodes: &[MorphNode],
    stem_rgb: [u8; 3],
    config: &RibbonConfig,
) -> RibbonMeshData {
    let chains = extract_branch_chains(nodes);
    let mut all_triangles = Vec::new();

    for chain in &chains {
        let control_points: Vec<BranchControlPoint> = chain
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let depth = if chain.len() > 1 {
                    i as f32 / (chain.len() - 1) as f32
                } else {
                    0.0
                };
                BranchControlPoint {
                    position: node.position,
                    radius: node.radius.max(0.02),
                    color: stem_rgb,
                    branch_depth: depth,
                }
            })
            .collect();

        let triangles = build_branch_tube(&control_points, config);
        all_triangles.extend(triangles);
    }

    triangles_to_mesh(&all_triangles)
}

/// Extract connected branch chains from flat morphology node array.
///
/// Groups Trunk and Branch nodes into chains by spatial proximity.
/// Each chain is a sequence of connected nodes for spline fitting.
fn extract_branch_chains(nodes: &[MorphNode]) -> Vec<Vec<&MorphNode>> {
    let woody: Vec<&MorphNode> = nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Trunk | NodeType::Branch))
        .collect();

    if woody.is_empty() {
        return Vec::new();
    }

    // Sort by height (Y position) to establish ordering
    let mut sorted = woody.clone();
    sorted.sort_by(|a, b| {
        a.position[1]
            .partial_cmp(&b.position[1])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Main trunk: all Trunk-type nodes sorted by height
    let trunk_nodes: Vec<&MorphNode> = sorted
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Trunk))
        .copied()
        .collect();

    let branch_nodes: Vec<&MorphNode> = sorted
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Branch))
        .copied()
        .collect();

    let mut chains = Vec::new();

    // Main trunk chain (needs at least 2 nodes for a spline)
    if trunk_nodes.len() >= 2 {
        chains.push(trunk_nodes);
    } else if !trunk_nodes.is_empty() {
        // Single trunk node — still include it but add a ground point conceptually
        chains.push(trunk_nodes);
    }

    // Branch chains: cluster by spatial proximity
    let mut used = vec![false; branch_nodes.len()];
    let threshold = 0.5f32; // max distance between consecutive branch nodes

    for i in 0..branch_nodes.len() {
        if used[i] {
            continue;
        }
        let mut chain = vec![branch_nodes[i]];
        used[i] = true;

        // Extend chain by finding nearest unused neighbor
        loop {
            let last = chain.last().unwrap();
            let last_pos = last.position;
            let mut best_idx = None;
            let mut best_dist = threshold;

            for j in 0..branch_nodes.len() {
                if used[j] {
                    continue;
                }
                let dx = branch_nodes[j].position[0] - last_pos[0];
                let dy = branch_nodes[j].position[1] - last_pos[1];
                let dz = branch_nodes[j].position[2] - last_pos[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = Some(j);
                }
            }

            if let Some(idx) = best_idx {
                chain.push(branch_nodes[idx]);
                used[idx] = true;
            } else {
                break;
            }
        }

        if chain.len() >= 2 {
            chains.push(chain);
        }
    }

    // Limit to reasonable number of chains
    chains.truncate(30);
    chains
}

// ---------------------------------------------------------------------------
// Protein backbone ribbon
// ---------------------------------------------------------------------------

use crate::terrarium::scale_level::AtomVisual;

/// Build a protein backbone ribbon from atom positions.
///
/// Extracts backbone atoms (C-alpha or all backbone atoms), fits a smooth
/// tube through them using the same parallel-transport algorithm as tree
/// branches. Colors from CPK. Suitable for molecular zoom visualization.
pub fn build_protein_backbone_ribbon(
    atoms: &[AtomVisual],
    config: &RibbonConfig,
) -> RibbonMeshData {
    // Extract backbone atoms (C-alpha positions for proteins,
    // or all heavy atoms for small molecules)
    let backbone: Vec<BranchControlPoint> = atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| {
            let depth = if atoms.len() > 1 {
                i as f32 / (atoms.len() - 1) as f32
            } else {
                0.0
            };
            BranchControlPoint {
                position: atom.position,
                radius: atom.vdw_radius * 0.3, // thinner than VDW for backbone trace
                color: atom.cpk_color,
                branch_depth: depth,
            }
        })
        .collect();

    if backbone.len() < 2 {
        return RibbonMeshData {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            branch_depth: Vec::new(),
            colors: Vec::new(),
        };
    }

    let triangles = build_branch_tube(&backbone, config);
    triangles_to_mesh(&triangles)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_tube_produces_triangles() {
        let points = vec![
            BranchControlPoint {
                position: [0.0, 0.0, 0.0],
                radius: 0.1,
                color: [128, 64, 32],
                branch_depth: 0.0,
            },
            BranchControlPoint {
                position: [0.0, 1.0, 0.0],
                radius: 0.1,
                color: [128, 64, 32],
                branch_depth: 0.5,
            },
            BranchControlPoint {
                position: [0.0, 2.0, 0.0],
                radius: 0.05,
                color: [128, 64, 32],
                branch_depth: 1.0,
            },
        ];
        let config = RibbonConfig::default();
        let triangles = build_branch_tube(&points, &config);
        assert!(!triangles.is_empty(), "should produce triangles");
        // With 6 subdivisions, 2 segments = ~13 spline points
        // 12 rings x 8 segments x 2 tris = ~192 + 2 caps x 8 = 208
        assert!(
            triangles.len() > 100,
            "expected >100 triangles, got {}",
            triangles.len()
        );
    }

    #[test]
    fn tapered_tube_has_shrinking_radius() {
        let points = vec![
            BranchControlPoint {
                position: [0.0, 0.0, 0.0],
                radius: 0.2,
                color: [128, 64, 32],
                branch_depth: 0.0,
            },
            BranchControlPoint {
                position: [0.0, 1.0, 0.0],
                radius: 0.2,
                color: [128, 64, 32],
                branch_depth: 0.5,
            },
            BranchControlPoint {
                position: [0.0, 2.0, 0.0],
                radius: 0.04,
                color: [128, 64, 32],
                branch_depth: 1.0,
            },
        ];
        let config = RibbonConfig::default();
        let mesh = triangles_to_mesh(&build_branch_tube(&points, &config));
        // Check that vertices near the top (y > 1.5) have smaller radius
        // than vertices near the bottom (y < 0.5)
        let mut max_r_bottom = 0.0f32;
        let mut max_r_top = 0.0f32;
        for i in (0..mesh.positions.len()).step_by(3) {
            let x = mesh.positions[i];
            let y = mesh.positions[i + 1];
            let z = mesh.positions[i + 2];
            let r = (x * x + z * z).sqrt();
            if y < 0.5 {
                max_r_bottom = max_r_bottom.max(r);
            } else if y > 1.5 {
                max_r_top = max_r_top.max(r);
            }
        }
        assert!(
            max_r_top < max_r_bottom,
            "top radius ({}) should be less than bottom ({})",
            max_r_top,
            max_r_bottom
        );
    }

    #[test]
    fn bent_tube_has_stable_frames() {
        // 90-degree bend should not produce degenerate normals
        let points = vec![
            BranchControlPoint {
                position: [0.0, 0.0, 0.0],
                radius: 0.1,
                color: [128, 64, 32],
                branch_depth: 0.0,
            },
            BranchControlPoint {
                position: [0.0, 1.0, 0.0],
                radius: 0.1,
                color: [128, 64, 32],
                branch_depth: 0.33,
            },
            BranchControlPoint {
                position: [1.0, 1.0, 0.0],
                radius: 0.08,
                color: [128, 64, 32],
                branch_depth: 0.66,
            },
            BranchControlPoint {
                position: [1.0, 1.0, 1.0],
                radius: 0.05,
                color: [128, 64, 32],
                branch_depth: 1.0,
            },
        ];
        let config = RibbonConfig::default();
        let triangles = build_branch_tube(&points, &config);
        assert!(!triangles.is_empty());
        // All normals should be finite
        for tri in &triangles {
            for n in &tri.normals {
                assert!(n[0].is_finite() && n[1].is_finite() && n[2].is_finite());
            }
        }
    }

    #[test]
    fn mesh_conversion_preserves_data() {
        let points = vec![
            BranchControlPoint {
                position: [0.0, 0.0, 0.0],
                radius: 0.1,
                color: [255, 0, 0],
                branch_depth: 0.0,
            },
            BranchControlPoint {
                position: [0.0, 1.0, 0.0],
                radius: 0.05,
                color: [0, 255, 0],
                branch_depth: 1.0,
            },
        ];
        let config = RibbonConfig {
            subdivisions: 2,
            radial_segments: 4,
            ..Default::default()
        };
        let triangles = build_branch_tube(&points, &config);
        let mesh = triangles_to_mesh(&triangles);
        assert_eq!(mesh.positions.len(), mesh.normals.len());
        assert_eq!(mesh.positions.len() / 3, mesh.branch_depth.len());
        assert_eq!(mesh.positions.len(), mesh.colors.len());
        assert_eq!(mesh.indices.len(), triangles.len() * 3);
    }
}
