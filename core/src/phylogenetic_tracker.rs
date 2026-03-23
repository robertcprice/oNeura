//! Phylogenetic lineage tracker for terrarium ecosystem evolution.
//!
//! Maintains a full evolutionary tree of organisms across generations,
//! supporting ancestry queries (MRCA, lineage tracing, coalescent distance),
//! speciation event detection via genome-hash divergence, phylogenetic diversity
//! metrics, and Newick-format export for downstream visualization tools
//! (e.g. FigTree, iTOL, ggtree).
//!
//! # Design
//!
//! The tree is stored as a flat `HashMap<u64, PhyloNode>` keyed by unique node
//! id.  Each node records its parent, generation, fitness, genome hash, birth/
//! death times, trait vector, and child list.  All queries are O(depth) or
//! O(n) in the number of nodes -- acceptable for populations up to ~10^5 per
//! evolutionary run.
//!
//! # Self-Containedness
//!
//! This module imports **nothing** from the parent crate.  It depends only on
//! `std::collections::HashMap` and is safe to compile as a standalone unit.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Quantitative trait vector attached to every phylogenetic node.
///
/// These values are snapshot copies taken at the moment the organism is
/// registered in the tree -- they do **not** mutate over the organism's
/// lifetime.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PhyloTraits {
    /// Total biomass (arbitrary units, typically grams dry weight).
    pub biomass: f32,
    /// Survival fraction under drought stress (0.0 = dies immediately, 1.0 = unaffected).
    pub drought_tolerance: f32,
    /// Catalytic efficiency of primary metabolic enzymes (dimensionless, higher is better).
    pub enzyme_efficacy: f32,
    /// Expected offspring per unit time under ideal conditions.
    pub reproductive_rate: f32,
    /// Breadth of environmental conditions tolerated (higher = generalist).
    pub niche_width: f32,
}

/// A single node in the phylogenetic tree.
///
/// Represents one organism (or one representative genotype) at a specific
/// generation.  Leaf nodes whose `death_time_s` is `None` are considered
/// alive.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PhyloNode {
    /// Unique identifier (monotonically increasing).
    pub id: u64,
    /// Parent node id.  `None` for the root(s) of the tree.
    pub parent_id: Option<u64>,
    /// Discrete generation counter (0-indexed).
    pub generation: u32,
    /// Fitness score at birth.
    pub fitness: f32,
    /// Hash of the genome bit-string -- used for divergence calculations.
    pub genome_hash: u64,
    /// Simulation wall-clock time at birth (seconds).
    pub birth_time_s: f32,
    /// Simulation wall-clock time at death (seconds).  `None` while alive.
    pub death_time_s: Option<f32>,
    /// Direct child node ids.
    pub children: Vec<u64>,
    /// Phenotypic trait snapshot at birth.
    pub traits: PhyloTraits,
}

/// Record of a detected speciation event.
///
/// Two daughter lineages are considered to have speciated when the Hamming-
/// style divergence of their genome hashes (measured as XOR bit-count
/// normalised to `[0, 1]`) exceeds a caller-supplied threshold.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpeciationEvent {
    /// Simulation time at which the divergence was first detected.
    pub time_s: f32,
    /// Node id of the last common ancestor before the split.
    pub parent_id: u64,
    /// The two daughter node ids that diverged.
    pub daughter_ids: (u64, u64),
    /// Normalised divergence score in `[0.0, 1.0]`.
    pub divergence: f32,
}

// ---------------------------------------------------------------------------
// Tree implementation
// ---------------------------------------------------------------------------

/// Complete phylogenetic tree with O(1) node lookup and ancestry queries.
///
/// # Usage
///
/// ```ignore
/// let mut tree = PhyloTree::new();
/// let root = tree.add_node(None, 0, 1.0, 0xDEAD, 0.0, PhyloTraits::default());
/// let child = tree.add_node(Some(root), 1, 1.2, 0xBEEF, 1.0, PhyloTraits::default());
/// assert_eq!(tree.lineage(child), vec![root, child]);
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PhyloTree {
    /// All nodes keyed by unique id.
    nodes: HashMap<u64, PhyloNode>,
    /// Monotonically increasing id counter.
    next_id: u64,
}

impl PhyloTree {
    // -- construction -------------------------------------------------------

    /// Create an empty phylogenetic tree with no nodes.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }

    /// Register a new organism in the tree.
    ///
    /// If `parent_id` is `Some(p)`, the new node is appended to parent `p`'s
    /// child list.  Returns the freshly-minted node id.
    ///
    /// # Panics
    ///
    /// Panics (debug only) if `parent_id` refers to a node that does not
    /// exist in the tree.
    pub fn add_node(
        &mut self,
        parent_id: Option<u64>,
        generation: u32,
        fitness: f32,
        genome_hash: u64,
        birth_time_s: f32,
        traits: PhyloTraits,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        if let Some(pid) = parent_id {
            debug_assert!(
                self.nodes.contains_key(&pid),
                "parent id {pid} not found in phylogenetic tree"
            );
            if let Some(parent) = self.nodes.get_mut(&pid) {
                parent.children.push(id);
            }
        }

        self.nodes.insert(
            id,
            PhyloNode {
                id,
                parent_id,
                generation,
                fitness,
                genome_hash,
                birth_time_s,
                death_time_s: None,
                children: Vec::new(),
                traits,
            },
        );

        id
    }

    // -- mortality -----------------------------------------------------------

    /// Record the death of a node.  Has no effect if the node is already dead
    /// or does not exist.
    pub fn mark_dead(&mut self, id: u64, death_time_s: f32) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.death_time_s = Some(death_time_s);
        }
    }

    // -- queries ------------------------------------------------------------

    /// Return ids of all nodes that are still alive (no recorded death time).
    pub fn living_nodes(&self) -> Vec<u64> {
        self.nodes
            .values()
            .filter(|n| n.death_time_s.is_none())
            .map(|n| n.id)
            .collect()
    }

    /// Compute the most recent common ancestor (MRCA) of two nodes.
    ///
    /// Walks both lineages to the root and returns the deepest shared
    /// ancestor.  Returns `None` if either id is missing or if the nodes
    /// belong to disconnected trees.
    pub fn mrca(&self, id_a: u64, id_b: u64) -> Option<u64> {
        let ancestors_a = self.ancestor_set(id_a);
        let mut current = Some(id_b);
        while let Some(cid) = current {
            if ancestors_a.contains(&cid) {
                return Some(cid);
            }
            current = self.nodes.get(&cid).and_then(|n| n.parent_id);
        }
        None
    }

    /// Return the full lineage from the root to the given node (inclusive).
    ///
    /// The first element is the root ancestor; the last is `id` itself.
    /// Returns an empty `Vec` if the id is not in the tree.
    pub fn lineage(&self, id: u64) -> Vec<u64> {
        if !self.nodes.contains_key(&id) {
            return Vec::new();
        }
        let mut path = Vec::new();
        let mut current = Some(id);
        while let Some(cid) = current {
            if !self.nodes.contains_key(&cid) {
                break;
            }
            path.push(cid);
            current = self.nodes.get(&cid).and_then(|n| n.parent_id);
        }
        path.reverse();
        path
    }

    /// Maximum depth of the tree (longest root-to-leaf path in generations).
    ///
    /// An empty tree has depth 0.  A tree with only a root has depth 1.
    pub fn tree_depth(&self) -> u32 {
        self.nodes
            .values()
            .map(|n| n.generation + 1)
            .max()
            .unwrap_or(0)
    }

    /// Mean branch length across all parent-child edges.
    ///
    /// Branch length is defined as the absolute difference in `birth_time_s`
    /// between parent and child.  Returns 0.0 for trees with zero edges.
    pub fn mean_branch_length(&self) -> f32 {
        let mut total = 0.0_f32;
        let mut count = 0_u32;
        for node in self.nodes.values() {
            if let Some(pid) = node.parent_id {
                if let Some(parent) = self.nodes.get(&pid) {
                    total += (node.birth_time_s - parent.birth_time_s).abs();
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }

    /// Detect speciation events by genome-hash divergence.
    ///
    /// For every node with two or more children, the pairwise normalised
    /// Hamming distance (XOR popcount / 64) of children's genome hashes is
    /// computed.  Any pair exceeding `divergence_threshold` is reported as a
    /// speciation event.
    pub fn speciation_events(&self, divergence_threshold: f32) -> Vec<SpeciationEvent> {
        let mut events = Vec::new();
        for node in self.nodes.values() {
            if node.children.len() < 2 {
                continue;
            }
            // Check all pairs of children.
            for i in 0..node.children.len() {
                for j in (i + 1)..node.children.len() {
                    let cid_a = node.children[i];
                    let cid_b = node.children[j];
                    if let (Some(a), Some(b)) = (self.nodes.get(&cid_a), self.nodes.get(&cid_b)) {
                        let divergence = normalised_hamming(a.genome_hash, b.genome_hash);
                        if divergence >= divergence_threshold {
                            let time = a.birth_time_s.max(b.birth_time_s);
                            events.push(SpeciationEvent {
                                time_s: time,
                                parent_id: node.id,
                                daughter_ids: (cid_a, cid_b),
                                divergence,
                            });
                        }
                    }
                }
            }
        }
        // Sort by time for deterministic output.
        events.sort_by(|a, b| {
            a.time_s
                .partial_cmp(&b.time_s)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        events
    }

    /// Count living organisms in each time bin.
    ///
    /// For each value `t` in `time_bins`, counts nodes whose
    /// `birth_time_s <= t` and (`death_time_s` is `None` or `> t`).
    /// Returns `(t, count)` pairs in the same order as the input bins.
    pub fn diversity_over_time(&self, time_bins: &[f32]) -> Vec<(f32, usize)> {
        time_bins
            .iter()
            .map(|&t| {
                let count = self
                    .nodes
                    .values()
                    .filter(|n| {
                        n.birth_time_s <= t
                            && match n.death_time_s {
                                Some(d) => d > t,
                                None => true,
                            }
                    })
                    .count();
                (t, count)
            })
            .collect()
    }

    /// Number of generations separating two nodes through their MRCA.
    ///
    /// Defined as `gen(a) + gen(b) - 2 * gen(mrca)`.  Returns 0 if the
    /// MRCA cannot be found.
    pub fn coalescent_distance(&self, id_a: u64, id_b: u64) -> u32 {
        if let Some(mrca_id) = self.mrca(id_a, id_b) {
            let gen_a = self.nodes.get(&id_a).map_or(0, |n| n.generation);
            let gen_b = self.nodes.get(&id_b).map_or(0, |n| n.generation);
            let gen_m = self.nodes.get(&mrca_id).map_or(0, |n| n.generation);
            (gen_a + gen_b).saturating_sub(2 * gen_m)
        } else {
            0
        }
    }

    /// Faith's phylogenetic diversity: sum of all branch lengths in the tree.
    ///
    /// Uses the same branch-length metric as [`mean_branch_length`] (absolute
    /// difference in `birth_time_s`).  An empty or single-node tree returns
    /// 0.0.
    pub fn phylogenetic_diversity(&self) -> f32 {
        let mut total = 0.0_f32;
        for node in self.nodes.values() {
            if let Some(pid) = node.parent_id {
                if let Some(parent) = self.nodes.get(&pid) {
                    total += (node.birth_time_s - parent.birth_time_s).abs();
                }
            }
        }
        total
    }

    /// Serialize the tree to Newick format.
    ///
    /// Newick is the de-facto standard for phylogenetic tree interchange
    /// (<https://en.wikipedia.org/wiki/Newick_format>).  Branch lengths are
    /// encoded as `:length` suffixes (absolute birth-time difference to
    /// parent).  The root is terminated with a semicolon.
    ///
    /// If the tree has multiple roots (disconnected sub-trees), they are
    /// concatenated as siblings under a synthetic root.
    pub fn to_newick(&self) -> String {
        let roots: Vec<u64> = self
            .nodes
            .values()
            .filter(|n| n.parent_id.is_none())
            .map(|n| n.id)
            .collect();

        if roots.is_empty() {
            return String::from(";");
        }

        if roots.len() == 1 {
            let mut buf = String::new();
            self.newick_subtree(roots[0], &mut buf);
            buf.push(';');
            return buf;
        }

        // Multiple roots -- wrap in synthetic parent.
        let mut buf = String::from("(");
        for (i, &rid) in roots.iter().enumerate() {
            if i > 0 {
                buf.push(',');
            }
            self.newick_subtree(rid, &mut buf);
        }
        buf.push_str(");");
        buf
    }

    // -- read access --------------------------------------------------------

    /// Get an immutable reference to a node by id.
    pub fn get_node(&self, id: u64) -> Option<&PhyloNode> {
        self.nodes.get(&id)
    }

    /// Total number of nodes in the tree (living and dead).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True if the tree contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // -- private helpers ----------------------------------------------------

    /// Collect the set of all ancestors of `id` (including `id` itself).
    fn ancestor_set(&self, id: u64) -> std::collections::HashSet<u64> {
        let mut set = std::collections::HashSet::new();
        let mut current = Some(id);
        while let Some(cid) = current {
            set.insert(cid);
            current = self.nodes.get(&cid).and_then(|n| n.parent_id);
        }
        set
    }

    /// Recursively build the Newick sub-tree rooted at `id`.
    fn newick_subtree(&self, id: u64, buf: &mut String) {
        let node = match self.nodes.get(&id) {
            Some(n) => n,
            None => return,
        };

        if !node.children.is_empty() {
            buf.push('(');
            for (i, &cid) in node.children.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                self.newick_subtree(cid, buf);
            }
            buf.push(')');
        }

        // Node label = "n<id>"
        buf.push_str(&format!("n{}", node.id));

        // Branch length = time difference to parent.
        if let Some(pid) = node.parent_id {
            if let Some(parent) = self.nodes.get(&pid) {
                let bl = (node.birth_time_s - parent.birth_time_s).abs();
                buf.push_str(&format!(":{:.4}", bl));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Normalised Hamming distance between two 64-bit genome hashes.
///
/// Returns a value in `[0.0, 1.0]` where 0.0 means identical and 1.0 means
/// all 64 bits differ.
fn normalised_hamming(a: u64, b: u64) -> f32 {
    (a ^ b).count_ones() as f32 / 64.0
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: default traits with overridable biomass.
    fn traits_with_biomass(biomass: f32) -> PhyloTraits {
        PhyloTraits {
            biomass,
            ..Default::default()
        }
    }

    #[test]
    fn tree_starts_empty() {
        let tree = PhyloTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.tree_depth(), 0);
        assert!(tree.living_nodes().is_empty());
    }

    #[test]
    fn add_root_node() {
        let mut tree = PhyloTree::new();
        let root = tree.add_node(None, 0, 1.0, 0xAAAA, 0.0, PhyloTraits::default());
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.tree_depth(), 1);

        let node = tree.get_node(root).unwrap();
        assert_eq!(node.id, root);
        assert!(node.parent_id.is_none());
        assert_eq!(node.generation, 0);
        assert!(node.death_time_s.is_none());
    }

    #[test]
    fn add_child_nodes() {
        let mut tree = PhyloTree::new();
        let root = tree.add_node(None, 0, 1.0, 0x1111, 0.0, PhyloTraits::default());
        let c1 = tree.add_node(Some(root), 1, 1.1, 0x2222, 1.0, traits_with_biomass(10.0));
        let c2 = tree.add_node(Some(root), 1, 0.9, 0x3333, 1.0, traits_with_biomass(8.0));

        assert_eq!(tree.len(), 3);
        assert_eq!(tree.tree_depth(), 2);

        let root_node = tree.get_node(root).unwrap();
        assert_eq!(root_node.children.len(), 2);
        assert!(root_node.children.contains(&c1));
        assert!(root_node.children.contains(&c2));

        let child = tree.get_node(c1).unwrap();
        assert_eq!(child.parent_id, Some(root));
        assert!((child.traits.biomass - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn mrca_of_siblings() {
        let mut tree = PhyloTree::new();
        let root = tree.add_node(None, 0, 1.0, 0x0, 0.0, PhyloTraits::default());
        let c1 = tree.add_node(Some(root), 1, 1.0, 0x1, 1.0, PhyloTraits::default());
        let c2 = tree.add_node(Some(root), 1, 1.0, 0x2, 1.0, PhyloTraits::default());

        assert_eq!(tree.mrca(c1, c2), Some(root));
    }

    #[test]
    fn mrca_of_cousins() {
        //       root (gen 0)
        //      /    \
        //    a (1)   b (1)
        //   /         \
        //  aa (2)     bb (2)
        let mut tree = PhyloTree::new();
        let root = tree.add_node(None, 0, 1.0, 0x0, 0.0, PhyloTraits::default());
        let a = tree.add_node(Some(root), 1, 1.0, 0x10, 1.0, PhyloTraits::default());
        let b = tree.add_node(Some(root), 1, 1.0, 0x20, 1.0, PhyloTraits::default());
        let aa = tree.add_node(Some(a), 2, 1.0, 0x11, 2.0, PhyloTraits::default());
        let bb = tree.add_node(Some(b), 2, 1.0, 0x21, 2.0, PhyloTraits::default());

        // Cousins share root as MRCA.
        assert_eq!(tree.mrca(aa, bb), Some(root));
        // Parent-child: MRCA is the parent.
        assert_eq!(tree.mrca(a, aa), Some(a));
        // Self: MRCA is itself.
        assert_eq!(tree.mrca(aa, aa), Some(aa));

        // Coalescent distance for cousins: gen(aa) + gen(bb) - 2*gen(root) = 2+2-0 = 4
        assert_eq!(tree.coalescent_distance(aa, bb), 4);
        // Coalescent distance for siblings: 1+1-0 = 2
        assert_eq!(tree.coalescent_distance(a, b), 2);
    }

    #[test]
    fn lineage_traces_to_root() {
        let mut tree = PhyloTree::new();
        let root = tree.add_node(None, 0, 1.0, 0x0, 0.0, PhyloTraits::default());
        let mid = tree.add_node(Some(root), 1, 1.0, 0x1, 1.0, PhyloTraits::default());
        let leaf = tree.add_node(Some(mid), 2, 1.0, 0x2, 2.0, PhyloTraits::default());

        let lin = tree.lineage(leaf);
        assert_eq!(lin, vec![root, mid, leaf]);

        // Root lineage is just itself.
        assert_eq!(tree.lineage(root), vec![root]);

        // Non-existent node returns empty.
        assert!(tree.lineage(999).is_empty());
    }

    #[test]
    fn speciation_detection() {
        let mut tree = PhyloTree::new();
        let root = tree.add_node(None, 0, 1.0, 0x0, 0.0, PhyloTraits::default());

        // Two children with maximally different genome hashes.
        let _c1 = tree.add_node(
            Some(root),
            1,
            1.0,
            0x0000_0000_0000_0000,
            1.0,
            PhyloTraits::default(),
        );
        let _c2 = tree.add_node(
            Some(root),
            1,
            1.0,
            0xFFFF_FFFF_FFFF_FFFF,
            1.0,
            PhyloTraits::default(),
        );

        // Divergence = 64/64 = 1.0 -- should detect with threshold 0.5.
        let events = tree.speciation_events(0.5);
        assert_eq!(events.len(), 1);
        assert!((events[0].divergence - 1.0).abs() < f32::EPSILON);
        assert_eq!(events[0].parent_id, root);

        // With threshold 1.1 -- nothing should match.
        let events_strict = tree.speciation_events(1.1);
        assert!(events_strict.is_empty());

        // Two children with identical hashes -- no speciation at any threshold.
        let mut tree2 = PhyloTree::new();
        let r2 = tree2.add_node(None, 0, 1.0, 0x0, 0.0, PhyloTraits::default());
        let _d1 = tree2.add_node(Some(r2), 1, 1.0, 0xABCD, 1.0, PhyloTraits::default());
        let _d2 = tree2.add_node(Some(r2), 1, 1.0, 0xABCD, 1.0, PhyloTraits::default());
        assert!(tree2.speciation_events(0.01).is_empty());
    }

    #[test]
    fn mark_dead_filters_living() {
        let mut tree = PhyloTree::new();
        let root = tree.add_node(None, 0, 1.0, 0x0, 0.0, PhyloTraits::default());
        let c1 = tree.add_node(Some(root), 1, 1.0, 0x1, 1.0, PhyloTraits::default());
        let c2 = tree.add_node(Some(root), 1, 1.0, 0x2, 1.0, PhyloTraits::default());

        assert_eq!(tree.living_nodes().len(), 3);

        tree.mark_dead(root, 5.0);
        tree.mark_dead(c1, 6.0);

        let living = tree.living_nodes();
        assert_eq!(living.len(), 1);
        assert_eq!(living[0], c2);

        // Verify death time recorded correctly.
        let dead_node = tree.get_node(root).unwrap();
        assert!((dead_node.death_time_s.unwrap() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn newick_format_correct() {
        // Build: root -> (a, b)
        let mut tree = PhyloTree::new();
        let root = tree.add_node(None, 0, 1.0, 0x0, 0.0, PhyloTraits::default());
        let _a = tree.add_node(Some(root), 1, 1.0, 0x1, 1.0, PhyloTraits::default());
        let _b = tree.add_node(Some(root), 1, 1.0, 0x2, 2.0, PhyloTraits::default());

        let nwk = tree.to_newick();

        // Must end with semicolon.
        assert!(nwk.ends_with(';'), "Newick must end with ';': {nwk}");
        // Must contain both children and the root label.
        assert!(nwk.contains("n0"), "should contain root label n0: {nwk}");
        assert!(nwk.contains("n1"), "should contain child label n1: {nwk}");
        assert!(nwk.contains("n2"), "should contain child label n2: {nwk}");
        // Must contain parentheses for the clade.
        assert!(
            nwk.contains('(') && nwk.contains(')'),
            "should have clade parens: {nwk}"
        );
        // Must contain branch lengths (colon-delimited).
        assert!(nwk.contains(':'), "should contain branch lengths: {nwk}");

        // Empty tree should produce just ";".
        let empty = PhyloTree::new();
        assert_eq!(empty.to_newick(), ";");
    }

    #[test]
    fn phylogenetic_diversity_computation() {
        // root (t=0) -> a (t=1) -> aa (t=3)
        //            -> b (t=2)
        // Branch lengths: root->a = 1, root->b = 2, a->aa = 2
        // Total PD = 1 + 2 + 2 = 5
        let mut tree = PhyloTree::new();
        let root = tree.add_node(None, 0, 1.0, 0x0, 0.0, PhyloTraits::default());
        let a = tree.add_node(Some(root), 1, 1.0, 0x1, 1.0, PhyloTraits::default());
        let _b = tree.add_node(Some(root), 1, 1.0, 0x2, 2.0, PhyloTraits::default());
        let _aa = tree.add_node(Some(a), 2, 1.0, 0x3, 3.0, PhyloTraits::default());

        let pd = tree.phylogenetic_diversity();
        assert!(
            (pd - 5.0).abs() < f32::EPSILON,
            "PD should be 5.0, got {pd}"
        );

        // Mean branch length = 5.0 / 3 edges = 1.6667
        let mbl = tree.mean_branch_length();
        assert!(
            (mbl - 5.0 / 3.0).abs() < 0.001,
            "mean BL should be ~1.667, got {mbl}"
        );

        // Empty tree PD = 0.
        let empty = PhyloTree::new();
        assert!((empty.phylogenetic_diversity() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn diversity_over_time_bins() {
        let mut tree = PhyloTree::new();
        // root born at t=0, dies at t=3
        let root = tree.add_node(None, 0, 1.0, 0x0, 0.0, PhyloTraits::default());
        // a born at t=1, lives forever
        let _a = tree.add_node(Some(root), 1, 1.0, 0x1, 1.0, PhyloTraits::default());
        // b born at t=2, dies at t=4
        let b = tree.add_node(Some(root), 1, 1.0, 0x2, 2.0, PhyloTraits::default());

        tree.mark_dead(root, 3.0);
        tree.mark_dead(b, 4.0);

        let bins = vec![0.5, 1.5, 2.5, 3.5, 5.0];
        let div = tree.diversity_over_time(&bins);

        // t=0.5: root alive                  -> 1
        // t=1.5: root alive, a alive          -> 2
        // t=2.5: root alive, a alive, b alive -> 3
        // t=3.5: a alive, b alive             -> 2  (root died at 3.0)
        // t=5.0: a alive                      -> 1  (b died at 4.0)
        assert_eq!(div[0], (0.5, 1));
        assert_eq!(div[1], (1.5, 2));
        assert_eq!(div[2], (2.5, 3));
        assert_eq!(div[3], (3.5, 2));
        assert_eq!(div[4], (5.0, 1));
    }

    #[test]
    fn normalised_hamming_distance() {
        // Identical hashes -> 0.0
        assert!((normalised_hamming(0xABCD, 0xABCD) - 0.0).abs() < f32::EPSILON);
        // All bits different -> 1.0
        assert!((normalised_hamming(0x0, 0xFFFF_FFFF_FFFF_FFFF) - 1.0).abs() < f32::EPSILON);
        // One bit different -> 1/64
        assert!((normalised_hamming(0x0, 0x1) - 1.0 / 64.0).abs() < f32::EPSILON);
    }
}
