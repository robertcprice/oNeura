//! Molecular Botany: Epigenetic Morphology and L-Systems.

use super::species::{species_profile_by_taxonomy, BotanicalGrowthForm};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A node in the physical plant structure graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphNode {
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub radius: f32,
    pub node_type: NodeType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    Trunk,
    Branch,
    Leaf,
    Fruit,
    Bud,
}

/// Represents the physical branching structure of a plant.
/// Governed by L-System rules and modulated by epigenetic signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantMorphology {
    pub taxonomy_id: u32,
    pub growth_form: BotanicalGrowthForm,
    /// The current L-System symbolic string (e.g., "FF[+F]F").
    pub l_string: String,
    /// Biological growth rules: Symbol -> Replacement.
    pub rules: HashMap<char, String>,

    /// Geometric parameters modulated by environment (Epigenetics).
    pub internode_length: f32,
    pub branch_angle_rad: f32,
    pub thickness_base: f32,
    pub fruit_radius_scale: f32,
    pub leaf_radius_scale: f32,
    pub lateral_bias: f32,
    pub droop_factor: f32,
    pub branch_twist_rad: f32,
    pub branch_depth_attenuation: f32,
    pub canopy_depth_scale: f32,
    pub leaf_cluster_density: f32,

    /// Current iteration/depth of growth.
    pub iterations: u32,

    // -- Phototropism (Phase D) --
    /// Direction of light asymmetry (normalized), used for phototropic leaning.
    pub phototropic_direction: [f32; 3],
    /// Strength of phototropic response: PIN1_expression × light_asymmetry.
    pub phototropic_strength: f32,

    // -- Turgor-driven visual droop (Phase E) --
    /// Molecular-driven droop override. When set > 0, overrides the epigenetic droop_factor
    /// with turgor-pressure-derived drooping from the metabolome.
    pub molecular_droop: f32,

    // -- Wind biomechanics --
    /// Current wind deflection angle (radians) from vertical.
    pub wind_deflection: f32,
    /// Accumulated mechanical damage from wind stress [0.0, 1.0].
    pub mechanical_damage: f32,
}

impl Default for PlantMorphology {
    fn default() -> Self {
        Self::for_taxonomy(3702)
    }
}

impl PlantMorphology {
    pub fn new(rules: HashMap<char, String>) -> Self {
        Self {
            rules,
            ..Default::default()
        }
    }

    pub fn for_taxonomy(taxonomy_id: u32) -> Self {
        let profile = species_profile_by_taxonomy(taxonomy_id);
        let growth_form = profile
            .map(|profile| profile.growth_form)
            .unwrap_or(BotanicalGrowthForm::RosetteHerb);
        let mut rules = HashMap::new();
        let (
            l_string,
            internode_length,
            branch_angle_rad,
            thickness_base,
            fruit_radius_scale,
            leaf_radius_scale,
            lateral_bias,
            droop_factor,
            branch_twist_rad,
            branch_depth_attenuation,
            canopy_depth_scale,
            leaf_cluster_density,
        ) = match growth_form {
            BotanicalGrowthForm::RosetteHerb => {
                rules.insert('A', "T[+L][/L][-L][\\L]A".to_string());
                rules.insert('T', "TT".to_string());
                (
                    "A".to_string(),
                    0.42,
                    0.82,
                    0.10,
                    0.72,
                    1.18,
                    0.96,
                    0.10,
                    0.18,
                    0.62,
                    0.44,
                    0.54,
                )
            }
            BotanicalGrowthForm::GrassClump => {
                rules.insert('A', "T[+L][-L][/L][\\L]A".to_string());
                rules.insert('T', "TT".to_string());
                (
                    "A".to_string(),
                    0.36,
                    0.26,
                    0.05,
                    0.38,
                    0.88,
                    0.64,
                    0.48,
                    0.08,
                    0.82,
                    0.24,
                    0.28,
                )
            }
            BotanicalGrowthForm::FloatingAquatic => {
                rules.insert('A', "T[/L][\\L][+L][-L]A".to_string());
                rules.insert('T', "TT".to_string());
                (
                    "A".to_string(),
                    0.16,
                    1.12,
                    0.04,
                    0.30,
                    1.24,
                    1.10,
                    0.06,
                    0.24,
                    0.96,
                    0.12,
                    0.92,
                )
            }
            BotanicalGrowthForm::SubmergedAquatic => {
                rules.insert('A', "TT[+B/L][/B\\L][-B/L]A".to_string());
                rules.insert('B', "B[/L][\\L][+L]L".to_string());
                (
                    "A".to_string(),
                    0.54,
                    0.74,
                    0.06,
                    0.34,
                    0.72,
                    0.88,
                    0.34,
                    0.42,
                    0.94,
                    0.88,
                    0.94,
                )
            }
            BotanicalGrowthForm::OrchardTree => {
                rules.insert('A', "TT[+B/L][-B\\L]TB".to_string());
                rules.insert('B', "B[+B/L][-B\\L]L".to_string());
                (
                    "A".to_string(),
                    0.72,
                    0.52,
                    0.16,
                    1.18,
                    1.04,
                    1.02,
                    0.22,
                    0.28,
                    0.88,
                    0.92,
                    0.82,
                )
            }
            BotanicalGrowthForm::StoneFruitTree => {
                rules.insert('A', "TT[+B/L][-B\\L][/BL]TB".to_string());
                rules.insert('B', "B[+BL][-B\\L]L".to_string());
                (
                    "A".to_string(),
                    0.68,
                    0.60,
                    0.14,
                    1.10,
                    0.94,
                    1.00,
                    0.30,
                    0.36,
                    0.82,
                    0.84,
                    0.76,
                )
            }
            BotanicalGrowthForm::CitrusTree => {
                rules.insert('A', "TT[+B/L][-B\\L][/B\\L]TB".to_string());
                rules.insert('B', "B[+BL][/BL][-B\\L]L".to_string());
                (
                    "A".to_string(),
                    0.60,
                    0.44,
                    0.15,
                    0.96,
                    0.90,
                    0.82,
                    0.16,
                    0.24,
                    0.94,
                    0.70,
                    0.68,
                )
            }
        };

        Self {
            taxonomy_id,
            growth_form,
            l_string,
            rules,
            internode_length,
            branch_angle_rad,
            thickness_base,
            fruit_radius_scale,
            leaf_radius_scale,
            lateral_bias,
            droop_factor,
            branch_twist_rad,
            branch_depth_attenuation,
            canopy_depth_scale,
            leaf_cluster_density,
            iterations: 0,
            phototropic_direction: [0.0, 0.0, 0.0],
            phototropic_strength: 0.0,
            molecular_droop: 0.0,
            wind_deflection: 0.0,
            mechanical_damage: 0.0,
        }
    }

    /// Advance the L-System by one iteration.
    pub fn grow(&mut self) {
        let mut next_string = String::with_capacity(self.l_string.len() * 2);
        for c in self.l_string.chars() {
            if let Some(replacement) = self.rules.get(&c) {
                next_string.push_str(replacement);
            } else {
                next_string.push(c);
            }
        }
        self.l_string = next_string;
        self.iterations += 1;
    }

    /// Modulate geometric parameters based on environmental stress.
    /// This is the 'Epigenetic' layer where the same DNA produces different shapes.
    pub fn modulate_epigenetics(&mut self, moisture: f32, light: f32) {
        match self.growth_form {
            BotanicalGrowthForm::RosetteHerb => {
                self.internode_length = 0.24 + moisture * 0.48;
                self.branch_angle_rad = 0.68 + (1.0 - light) * 0.30;
                self.thickness_base = 0.06 + moisture * 0.08;
                self.leaf_radius_scale = 1.0 + moisture * 0.42;
                self.lateral_bias = 0.84 + (1.0 - light) * 0.22;
                self.droop_factor = 0.06 + (1.0 - moisture) * 0.18;
                self.branch_twist_rad = 0.12 + (1.0 - light) * 0.10;
                self.branch_depth_attenuation = 0.56 + moisture * 0.16;
                self.canopy_depth_scale = 0.34 + moisture * 0.22;
                self.leaf_cluster_density = 0.46 + moisture * 0.28;
            }
            BotanicalGrowthForm::GrassClump => {
                self.internode_length = 0.28 + moisture * 0.70;
                self.branch_angle_rad = 0.14 + (1.0 - light) * 0.12;
                self.thickness_base = 0.028 + moisture * 0.045;
                self.leaf_radius_scale = 0.72 + moisture * 0.24;
                self.lateral_bias = 0.46 + (1.0 - light) * 0.18;
                self.droop_factor = 0.22 + (1.0 - moisture) * 0.38;
                self.branch_twist_rad = 0.06 + (1.0 - light) * 0.04;
                self.branch_depth_attenuation = 0.78 + moisture * 0.10;
                self.canopy_depth_scale = 0.18 + moisture * 0.12;
                self.leaf_cluster_density = 0.24 + moisture * 0.18;
            }
            BotanicalGrowthForm::FloatingAquatic => {
                self.internode_length = 0.10 + moisture * 0.14 + light * 0.06;
                self.branch_angle_rad = 0.92 + (1.0 - light) * 0.18;
                self.thickness_base = 0.026 + moisture * 0.022;
                self.leaf_radius_scale = 1.04 + moisture * 0.34;
                self.lateral_bias = 1.02 + (1.0 - light) * 0.12;
                self.droop_factor = 0.04 + (1.0 - moisture) * 0.08;
                self.branch_twist_rad = 0.18 + (1.0 - light) * 0.10;
                self.branch_depth_attenuation = 0.90 + moisture * 0.06;
                self.canopy_depth_scale = 0.08 + moisture * 0.08;
                self.leaf_cluster_density = 0.86 + moisture * 0.10;
            }
            BotanicalGrowthForm::SubmergedAquatic => {
                self.internode_length = 0.42 + moisture * 0.42;
                self.branch_angle_rad = 0.58 + (1.0 - light) * 0.22;
                self.thickness_base = 0.032 + moisture * 0.040;
                self.leaf_radius_scale = 0.62 + moisture * 0.18;
                self.lateral_bias = 0.82 + (1.0 - light) * 0.14;
                self.droop_factor = 0.20 + (1.0 - moisture) * 0.12;
                self.branch_twist_rad = 0.30 + (1.0 - light) * 0.14;
                self.branch_depth_attenuation = 0.88 + moisture * 0.08;
                self.canopy_depth_scale = 0.72 + moisture * 0.12;
                self.leaf_cluster_density = 0.82 + moisture * 0.12;
            }
            BotanicalGrowthForm::OrchardTree => {
                self.internode_length = 0.42 + moisture * 0.72;
                self.branch_angle_rad = 0.34 + (1.0 - light) * 0.28;
                self.thickness_base = 0.11 + moisture * 0.11;
                self.leaf_radius_scale = 0.86 + moisture * 0.20;
                self.lateral_bias = 0.92 + (1.0 - light) * 0.16;
                self.droop_factor = 0.10 + (1.0 - moisture) * 0.18;
                self.branch_twist_rad = 0.20 + (1.0 - light) * 0.12;
                self.branch_depth_attenuation = 0.78 + moisture * 0.14;
                self.canopy_depth_scale = 0.66 + moisture * 0.20;
                self.leaf_cluster_density = 0.62 + moisture * 0.18;
            }
            BotanicalGrowthForm::StoneFruitTree => {
                self.internode_length = 0.40 + moisture * 0.66;
                self.branch_angle_rad = 0.42 + (1.0 - light) * 0.24;
                self.thickness_base = 0.10 + moisture * 0.10;
                self.leaf_radius_scale = 0.82 + moisture * 0.18;
                self.lateral_bias = 0.94 + (1.0 - light) * 0.14;
                self.droop_factor = 0.12 + (1.0 - moisture) * 0.24;
                self.branch_twist_rad = 0.28 + (1.0 - light) * 0.14;
                self.branch_depth_attenuation = 0.72 + moisture * 0.16;
                self.canopy_depth_scale = 0.60 + moisture * 0.18;
                self.leaf_cluster_density = 0.56 + moisture * 0.20;
            }
            BotanicalGrowthForm::CitrusTree => {
                self.internode_length = 0.36 + moisture * 0.58;
                self.branch_angle_rad = 0.30 + (1.0 - light) * 0.22;
                self.thickness_base = 0.10 + moisture * 0.10;
                self.leaf_radius_scale = 0.76 + moisture * 0.16;
                self.lateral_bias = 0.72 + (1.0 - light) * 0.16;
                self.droop_factor = 0.08 + (1.0 - moisture) * 0.14;
                self.branch_twist_rad = 0.18 + (1.0 - light) * 0.10;
                self.branch_depth_attenuation = 0.84 + moisture * 0.12;
                self.canopy_depth_scale = 0.54 + moisture * 0.14;
                self.leaf_cluster_density = 0.50 + moisture * 0.16;
            }
        }
    }

    /// Convert the symbolic L-string into a graph of nodes for rendering.
    pub fn generate_nodes(&self) -> Vec<MorphNode> {
        self.generate_nodes_with_context(0, 1.0)
    }

    pub fn generate_nodes_with_context(&self, fruit_count: u32, vigor: f32) -> Vec<MorphNode> {
        let mut nodes = Vec::new();
        let mut stack = Vec::new();

        let mut pos = [0.0, 0.0, 0.0];
        let mut dir = [0.0, 1.0, 0.0]; // Moving up Y-axis
        // Phototropic bias: initial angle offset toward light source
        let photo_bias_x = self.phototropic_direction[0] * self.phototropic_strength * 0.3;
        let photo_bias_y = self.phototropic_direction[2] * self.phototropic_strength * 0.3;
        let mut angle_y = photo_bias_y;
        let mut angle_x = photo_bias_x;
        let mut branch_depth = 0u32;
        let mut segment_index = 0usize;

        let phase_at = |idx: usize, depth: u32, salt: f32| {
            ((idx as f32 * 0.971
                + depth as f32 * 1.713
                + self.taxonomy_id as f32 * 0.00091
                + self.branch_twist_rad * 7.0
                + self.leaf_cluster_density * 3.0
                + salt)
                .sin()
                * 0.5)
                + 0.5
        };

        for c in self.l_string.chars() {
            match c {
                'F' | 'T' | 'B' => {
                    let phase = phase_at(
                        segment_index,
                        branch_depth,
                        if c == 'B' { 0.41 } else { 0.17 },
                    );
                    let depth_scale = (1.0
                        - branch_depth as f32 * 0.08 * self.branch_depth_attenuation)
                        .max(0.58);
                    angle_y += (phase - 0.5)
                        * self.branch_twist_rad
                        * (if c == 'B' { 1.0 } else { 0.35 })
                        * (1.0 + branch_depth as f32 * 0.18);
                    angle_x -= (phase - 0.5) * self.branch_angle_rad * 0.12 * (1.0 - depth_scale);
                    dir = self.calculate_direction(angle_x, angle_y);

                    let segment_len = match c {
                        'B' => self.internode_length * (0.72 + depth_scale * 0.18),
                        _ => self.internode_length * (0.88 + depth_scale * 0.12),
                    };
                    // Move forward
                    pos[0] += dir[0] * segment_len;
                    pos[1] += dir[1] * segment_len;
                    pos[2] += dir[2] * segment_len;
                    let effective_droop = if self.molecular_droop > 0.0 {
                        self.molecular_droop
                    } else {
                        self.droop_factor
                    };
                    let droop = match c {
                        'B' => effective_droop * 0.10,
                        _ => effective_droop * 0.05,
                    };
                    pos[1] -= droop * segment_len;

                    nodes.push(MorphNode {
                        position: pos,
                        rotation: [angle_x, angle_y, 0.0],
                        radius: if c == 'B' {
                            self.thickness_base * 0.72
                        } else {
                            self.thickness_base
                        },
                        node_type: if c == 'B' {
                            NodeType::Branch
                        } else {
                            NodeType::Trunk
                        },
                    });
                    segment_index += 1;
                }
                '+' => {
                    let depth_scale = (1.0
                        - branch_depth as f32 * 0.08 * self.branch_depth_attenuation)
                        .max(0.58);
                    angle_x += self.branch_angle_rad * self.lateral_bias * depth_scale;
                    dir = self.calculate_direction(angle_x, angle_y);
                }
                '-' => {
                    let depth_scale = (1.0
                        - branch_depth as f32 * 0.08 * self.branch_depth_attenuation)
                        .max(0.58);
                    angle_x -= self.branch_angle_rad * self.lateral_bias * depth_scale;
                    dir = self.calculate_direction(angle_x, angle_y);
                }
                '/' => {
                    let yaw_gain = 0.62
                        + self.lateral_bias * 0.28
                        + phase_at(segment_index, branch_depth, 0.73) * self.branch_twist_rad * 0.4;
                    angle_y += self.branch_angle_rad * yaw_gain;
                    dir = self.calculate_direction(angle_x, angle_y);
                }
                '\\' => {
                    let yaw_gain = 0.62
                        + self.lateral_bias * 0.28
                        + phase_at(segment_index, branch_depth, 1.19) * self.branch_twist_rad * 0.4;
                    angle_y -= self.branch_angle_rad * yaw_gain;
                    dir = self.calculate_direction(angle_x, angle_y);
                }
                '[' => {
                    stack.push((pos, dir, angle_x, angle_y, branch_depth));
                    branch_depth += 1;
                    let branch_phase = phase_at(segment_index, branch_depth, 1.57) - 0.5;
                    angle_y +=
                        branch_phase * self.branch_twist_rad * (1.1 + branch_depth as f32 * 0.12);
                    angle_x += branch_phase * self.branch_angle_rad * 0.16;
                    dir = self.calculate_direction(angle_x, angle_y);
                }
                ']' => {
                    if let Some(state) = stack.pop() {
                        pos = state.0;
                        dir = state.1;
                        angle_x = state.2;
                        angle_y = state.3;
                        branch_depth = state.4;
                    }
                }
                'L' => {
                    let cluster_count = match self.growth_form {
                        BotanicalGrowthForm::GrassClump => 1,
                        BotanicalGrowthForm::RosetteHerb => (1.0 + self.leaf_cluster_density * 1.6)
                            .round()
                            .clamp(1.0, 3.0)
                            as usize,
                        BotanicalGrowthForm::FloatingAquatic => {
                            (3.0 + self.leaf_cluster_density * 2.0)
                                .round()
                                .clamp(3.0, 5.0) as usize
                        }
                        BotanicalGrowthForm::SubmergedAquatic => {
                            (3.0 + self.leaf_cluster_density * 2.8 + self.canopy_depth_scale * 0.8)
                                .round()
                                .clamp(4.0, 7.0) as usize
                        }
                        BotanicalGrowthForm::OrchardTree
                        | BotanicalGrowthForm::StoneFruitTree
                        | BotanicalGrowthForm::CitrusTree => {
                            (2.0 + self.leaf_cluster_density * 2.4 + self.canopy_depth_scale * 0.8)
                                .round()
                                .clamp(3.0, 6.0) as usize
                        }
                    };
                    let radial_base = self.thickness_base
                        * (match self.growth_form {
                            BotanicalGrowthForm::GrassClump => 0.38,
                            BotanicalGrowthForm::RosetteHerb => 0.74,
                            BotanicalGrowthForm::FloatingAquatic => 1.34,
                            BotanicalGrowthForm::SubmergedAquatic => 0.58,
                            _ => 1.16,
                        })
                        * (1.0 + self.leaf_cluster_density * 0.55);
                    let vertical_span = self.internode_length
                        * self.canopy_depth_scale
                        * match self.growth_form {
                            BotanicalGrowthForm::GrassClump => 0.06,
                            BotanicalGrowthForm::RosetteHerb => 0.10,
                            BotanicalGrowthForm::FloatingAquatic => 0.03,
                            BotanicalGrowthForm::SubmergedAquatic => 0.30,
                            _ => 0.22,
                        };
                    for cluster_idx in 0..cluster_count {
                        let cluster_t = if cluster_count <= 1 {
                            0.5
                        } else {
                            cluster_idx as f32 / (cluster_count - 1) as f32
                        };
                        let phase = phase_at(
                            segment_index + cluster_idx,
                            branch_depth,
                            cluster_idx as f32 * 0.53,
                        );
                        let azimuth = angle_y
                            + cluster_t * std::f32::consts::TAU
                            + (phase - 0.5) * self.branch_twist_rad * 1.8;
                        let radial = radial_base * (0.72 + phase * 0.58);
                        let leaf_pos = [
                            pos[0] + azimuth.cos() * radial,
                            pos[1] + (cluster_t - 0.5) * vertical_span,
                            pos[2] + azimuth.sin() * radial,
                        ];
                        nodes.push(MorphNode {
                            position: leaf_pos,
                            rotation: [angle_x * 0.55, azimuth, 0.0],
                            radius: self.thickness_base
                                * self.leaf_radius_scale
                                * match self.growth_form {
                                    BotanicalGrowthForm::GrassClump => 1.05,
                                    BotanicalGrowthForm::RosetteHerb => 1.40,
                                    BotanicalGrowthForm::FloatingAquatic => 1.56,
                                    BotanicalGrowthForm::SubmergedAquatic => 0.90,
                                    _ => 1.12,
                                },
                            node_type: NodeType::Leaf,
                        });
                    }
                }
                'A' => {
                    nodes.push(MorphNode {
                        position: pos,
                        rotation: [angle_x, angle_y, 0.0],
                        radius: self.thickness_base * 0.85,
                        node_type: NodeType::Bud,
                    });
                }
                _ => {}
            }
        }

        if fruit_count > 0 {
            let mut anchors: Vec<[f32; 3]> = nodes
                .iter()
                .filter(|node| {
                    matches!(
                        node.node_type,
                        NodeType::Bud | NodeType::Leaf | NodeType::Branch
                    )
                })
                .map(|node| node.position)
                .collect();
            anchors.sort_by(|a, b| b[1].partial_cmp(&a[1]).unwrap_or(std::cmp::Ordering::Equal));
            let fruit_sites = usize::min(
                fruit_count.max(1) as usize,
                anchors.len().clamp(
                    1,
                    match self.growth_form {
                        BotanicalGrowthForm::GrassClump => 3,
                        BotanicalGrowthForm::RosetteHerb => 4,
                        BotanicalGrowthForm::FloatingAquatic => 2,
                        BotanicalGrowthForm::SubmergedAquatic => 3,
                        _ => 8,
                    },
                ),
            );
            let vigor_scale = vigor.clamp(0.4, 1.4);
            for (i, anchor) in anchors.into_iter().enumerate() {
                if i >= fruit_sites {
                    break;
                }
                let offset = [
                    ((i as f32 * 0.73) + self.taxonomy_id as f32 * 0.0007).sin()
                        * self.internode_length
                        * 0.16,
                    -self.internode_length * 0.10,
                    ((i as f32 * 0.41) + self.taxonomy_id as f32 * 0.0003).cos()
                        * self.internode_length
                        * 0.12,
                ];
                nodes.push(MorphNode {
                    position: [
                        anchor[0] + offset[0],
                        anchor[1] + offset[1],
                        anchor[2] + offset[2],
                    ],
                    rotation: [0.0, 0.0, 0.0],
                    radius: (self.thickness_base * self.fruit_radius_scale * vigor_scale)
                        .clamp(0.05, 0.22),
                    node_type: NodeType::Fruit,
                });
            }
        }
        nodes
    }

    fn calculate_direction(&self, ax: f32, ay: f32) -> [f32; 3] {
        [ax.sin() * ay.cos(), ax.cos(), ax.sin() * ay.sin()]
    }
}
