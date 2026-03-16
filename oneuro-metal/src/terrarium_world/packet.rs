// ── Phase 4: Genotype-ID Packet Populations ──────────────────────────
//
// Lightweight bottom-up microbial state for owned cells. Each packet
// references a reusable genotype catalog entry instead of duplicating
// a full gene vector. Packets are cheaper than WholeCellSimulator
// instances and can exist in larger numbers per owned cell.

use crate::constants::clamp;
use crate::soil_broad::SecondaryCatalogBankEntry;

// Phase 4: Genotype-ID packet population constants
pub(super) const GENOTYPE_PACKET_MAX_PER_CELL: usize = 12;
pub(super) const GENOTYPE_PACKET_POPULATION_MAX_CELLS: usize = 128;
const GENOTYPE_PACKET_MIN_ACTIVITY: f32 = 0.01;
const GENOTYPE_PACKET_GROWTH_RATE: f32 = 5.0e-4;
const GENOTYPE_PACKET_DECAY_RATE: f32 = 3.0e-4;
const GENOTYPE_PACKET_DAMAGE_RATE: f32 = 1.2e-4;
const GENOTYPE_PACKET_REPAIR_RATE: f32 = 6.0e-5;
const GENOTYPE_PACKET_DORMANCY_ENTRY_STRESS: f32 = 0.65;
const GENOTYPE_PACKET_DORMANCY_EXIT_ENERGY: f32 = 0.45;
const GENOTYPE_PACKET_PROMOTION_ACTIVITY: f32 = 0.85;
const GENOTYPE_PACKET_PROMOTION_ENERGY: f32 = 0.7;

/// A single genotype-ID packet: the lightweight unit of bottom-up
/// microbial state in an owned cell.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Scaffolding for genotype-level microbial tracking
pub(super) struct GenotypePacket {
    /// Index into the per-guild secondary catalog bank.
    pub(super) catalog_slot: u32,
    /// Genotype identity for lineage tracking.
    pub(super) genotype_id: u32,
    /// Lineage identity for phylogenetic tracking.
    pub(super) lineage_id: u32,
    /// Number of real cells this packet represents.
    pub(super) represented_cells: f32,
    /// Current metabolic activity [0..1]. High activity packets are
    /// candidates for promotion to full WholeCellSimulator.
    pub(super) activity: f32,
    /// Dormancy level [0..1]. Dormant packets consume less but also
    /// produce less flux.
    pub(super) dormancy: f32,
    /// Energy reserve [0..1]. Drives growth when high, triggers
    /// dormancy when low.
    pub(super) reserve: f32,
    /// Accumulated damage [0..1]. Irreversible above a threshold;
    /// kills the packet at 1.0.
    pub(super) damage: f32,
    /// Cumulative glucose uptake since packet creation.
    pub(super) cumulative_glucose_draw: f32,
    /// Cumulative oxygen uptake since packet creation.
    pub(super) cumulative_oxygen_draw: f32,
    /// Cumulative CO2 released since packet creation.
    pub(super) cumulative_co2_release: f32,
    /// Cumulative ammonium uptake for nitrogen metabolism.
    pub(super) cumulative_ammonium_draw: f32,
    /// Cumulative proton release from metabolic acidification.
    pub(super) cumulative_proton_release: f32,
}

impl GenotypePacket {
    pub(super) fn new(catalog_slot: u32, genotype_id: u32, lineage_id: u32, represented_cells: f32) -> Self {
        Self {
            catalog_slot,
            genotype_id,
            lineage_id,
            represented_cells,
            activity: 0.5,
            dormancy: 0.0,
            reserve: 0.5,
            damage: 0.0,
            cumulative_glucose_draw: 0.0,
            cumulative_oxygen_draw: 0.0,
            cumulative_co2_release: 0.0,
            cumulative_ammonium_draw: 0.0,
            cumulative_proton_release: 0.0,
        }
    }

    pub(super) fn is_alive(&self) -> bool {
        self.damage < 1.0 && self.represented_cells > 0.1
    }

    /// Whether this packet qualifies for promotion to a full
    /// WholeCellSimulator instance (Phase 5 hook).
    pub(super) fn qualifies_for_promotion(&self) -> bool {
        self.activity >= GENOTYPE_PACKET_PROMOTION_ACTIVITY
            && self.reserve >= GENOTYPE_PACKET_PROMOTION_ENERGY
            && self.dormancy < 0.1
            && self.damage < 0.2
    }

    /// Lightweight per-step metabolism. Reads local chemistry signals
    /// and updates packet state without a full WholeCellSimulator.
    pub(super) fn step(&mut self, dt: f32, local_glucose: f32, local_oxygen: f32, local_stress: f32) {
        let active_fraction = (1.0 - self.dormancy).max(0.0);

        // Energy intake: proportional to local resources and activity
        let glucose_available = local_glucose.clamp(0.0, 0.5);
        let oxygen_available = local_oxygen.clamp(0.0, 0.5);
        let intake_potential = (glucose_available * 0.6 + oxygen_available * 0.4)
            * active_fraction
            * self.represented_cells;
        let glucose_draw = intake_potential * 0.55 * dt;
        let oxygen_draw = intake_potential * 0.35 * dt;
        self.cumulative_glucose_draw += glucose_draw;
        self.cumulative_oxygen_draw += oxygen_draw;

        // Energy balance
        let energy_gain = intake_potential * 0.4 * dt;
        let maintenance_cost =
            self.represented_cells * (0.02 + self.activity * 0.03) * active_fraction * dt;
        self.reserve = clamp(self.reserve + energy_gain - maintenance_cost, 0.0, 1.0);

        // CO2 release from metabolism
        let co2_out = (glucose_draw * 0.72 + oxygen_draw * 0.28) * 0.8;
        self.cumulative_co2_release += co2_out;

        // Ammonium draw for nitrogen metabolism (~8% of carbon flux, Redfield-like)
        let nh4_draw = glucose_draw * 0.08;
        self.cumulative_ammonium_draw += nh4_draw;

        // Proton release from aerobic respiration (metabolic acidification)
        let proton_out = glucose_draw * 0.12;
        self.cumulative_proton_release += proton_out;

        // Activity update: rises when well-fed, falls under stress
        let activity_target = if self.reserve > 0.5 && local_stress < 0.4 {
            (self.reserve * 1.2).min(1.0)
        } else {
            (self.reserve * 0.6).max(GENOTYPE_PACKET_MIN_ACTIVITY)
        };
        self.activity = clamp(
            self.activity + (activity_target - self.activity) * dt * 2.0,
            GENOTYPE_PACKET_MIN_ACTIVITY,
            1.0,
        );

        // Dormancy: enter when stressed/low energy, exit when conditions improve
        if self.reserve < 0.2 || local_stress > GENOTYPE_PACKET_DORMANCY_ENTRY_STRESS {
            self.dormancy = clamp(self.dormancy + dt * 0.8 * (1.0 - self.dormancy), 0.0, 0.95);
        } else if self.reserve > GENOTYPE_PACKET_DORMANCY_EXIT_ENERGY && local_stress < 0.3 {
            self.dormancy = clamp(self.dormancy - dt * 0.4, 0.0, 0.95);
        }

        // Damage: accumulates under stress/starvation, slowly repaired
        let stress_damage = if self.reserve < 0.1 {
            GENOTYPE_PACKET_DAMAGE_RATE * (1.0 - self.reserve * 5.0)
        } else {
            0.0
        } + local_stress * GENOTYPE_PACKET_DAMAGE_RATE * 0.3;
        let repair = if self.reserve > 0.3 {
            GENOTYPE_PACKET_REPAIR_RATE * self.reserve
        } else {
            0.0
        };
        self.damage = clamp(self.damage + (stress_damage - repair) * dt, 0.0, 1.0);

        // Growth/decay of represented cells
        let growth_signal = if self.reserve > 0.6 && self.damage < 0.3 {
            GENOTYPE_PACKET_GROWTH_RATE * (self.reserve - 0.4) * active_fraction
        } else if self.reserve < 0.15 || self.damage > 0.6 {
            -GENOTYPE_PACKET_DECAY_RATE * (1.0 + self.damage)
        } else {
            0.0
        };
        self.represented_cells = (self.represented_cells * (1.0 + growth_signal * dt)).max(0.0);
    }
}

/// A population of genotype-ID packets at a single owned cell position.
/// This replaces the coarse microbial aggregate state for that cell.
#[derive(Debug, Clone)]
pub(super) struct GenotypePacketPopulation {
    /// Grid position.
    pub(super) x: usize,
    pub(super) y: usize,
    pub(super) z: usize,
    /// Owned packets, sorted by activity (descending) after each step.
    pub(super) packets: Vec<GenotypePacket>,
    /// Total represented cells across all packets (cached).
    pub(super) total_cells: f32,
    /// Age of this population in simulation seconds.
    pub(super) age_s: f32,
}

impl GenotypePacketPopulation {
    pub(super) fn new(x: usize, y: usize, z: usize) -> Self {
        Self {
            x,
            y,
            z,
            packets: Vec::new(),
            total_cells: 0.0,
            age_s: 0.0,
        }
    }

    pub(super) fn is_alive(&self) -> bool {
        !self.packets.is_empty() && self.total_cells > 0.5
    }

    #[allow(dead_code)]
    pub(super) fn add_packet(&mut self, packet: GenotypePacket) {
        if self.packets.len() < GENOTYPE_PACKET_MAX_PER_CELL {
            self.packets.push(packet);
            self.recompute_total();
        }
    }

    /// Seed population from the coarse secondary bank state at a cell.
    pub(super) fn seed_from_secondary_bank(
        &mut self,
        bank_entries: &[SecondaryCatalogBankEntry],
        total_coarse_cells: f32,
    ) {
        self.packets.clear();
        for (slot, entry) in bank_entries.iter().enumerate() {
            if entry.occupancy == 0 || entry.packet_mass < 0.01 {
                continue;
            }
            let cells = (total_coarse_cells * entry.packet_mass).max(0.5);
            self.packets.push(GenotypePacket::new(
                slot as u32,
                entry.record.genotype_id,
                entry.record.lineage_id,
                cells,
            ));
            if self.packets.len() >= GENOTYPE_PACKET_MAX_PER_CELL {
                break;
            }
        }
        self.recompute_total();
    }

    /// Step all packets with shared local chemistry signals.
    pub(super) fn step(&mut self, dt: f32, local_glucose: f32, local_oxygen: f32, local_stress: f32) {
        self.age_s += dt;

        // Per-packet share of local resources (competition)
        let n = self.packets.len().max(1) as f32;
        let per_packet_glucose = local_glucose / n;
        let per_packet_oxygen = local_oxygen / n;

        for packet in &mut self.packets {
            packet.step(dt, per_packet_glucose, per_packet_oxygen, local_stress);
        }

        // Remove dead packets
        self.packets.retain(|p| p.is_alive());

        // Sort by activity (highest first) for promotion scanning
        self.packets
            .sort_by(|a, b| b.activity.total_cmp(&a.activity));

        self.recompute_total();
    }

    /// Total glucose draw this step across all packets.
    pub(super) fn total_glucose_draw(&self) -> f32 {
        self.packets.iter().map(|p| p.cumulative_glucose_draw).sum()
    }

    /// Total CO2 release this step across all packets.
    pub(super) fn total_co2_release(&self) -> f32 {
        self.packets.iter().map(|p| p.cumulative_co2_release).sum()
    }

    /// Total oxygen draw this step across all packets.
    pub(super) fn total_oxygen_draw(&self) -> f32 {
        self.packets.iter().map(|p| p.cumulative_oxygen_draw).sum()
    }

    /// Total ammonium draw this step across all packets.
    pub(super) fn total_ammonium_draw(&self) -> f32 {
        self.packets
            .iter()
            .map(|p| p.cumulative_ammonium_draw)
            .sum()
    }

    /// Total proton release this step across all packets.
    pub(super) fn total_proton_release(&self) -> f32 {
        self.packets
            .iter()
            .map(|p| p.cumulative_proton_release)
            .sum()
    }

    /// Mean activity across packets.
    pub(super) fn mean_activity(&self) -> f32 {
        if self.packets.is_empty() {
            return 0.0;
        }
        self.packets.iter().map(|p| p.activity).sum::<f32>() / self.packets.len() as f32
    }

    /// Mean dormancy across packets.
    pub(super) fn mean_dormancy(&self) -> f32 {
        if self.packets.is_empty() {
            return 0.0;
        }
        self.packets.iter().map(|p| p.dormancy).sum::<f32>() / self.packets.len() as f32
    }

    /// Count of packets that qualify for promotion to WholeCellSimulator.
    pub(super) fn promotion_candidates(&self) -> usize {
        self.packets
            .iter()
            .filter(|p| p.qualifies_for_promotion())
            .count()
    }

    pub(crate) fn recompute_total(&mut self) {
        self.total_cells = self.packets.iter().map(|p| p.represented_cells).sum();
    }

    /// Try to bud a new packet into a neighboring cell.
    /// Returns `Some((target_x, target_y, new_packet))` if budding occurs.
    pub(super) fn try_bud(&mut self, width: usize, height: usize) -> Option<(usize, usize, GenotypePacket)> {
        // Find the first packet with high activity, high reserve, sufficient
        // cells, and low dormancy.
        let bud_idx = self.packets.iter().position(|p| {
            p.activity > 0.7
                && p.reserve > 0.6
                && p.represented_cells > 5.0
                && p.dormancy < 0.1
        })?;

        // Pick a deterministic-but-varying adjacent cell (4-connected).
        let directions: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        let dir_idx = (self.age_s * 7.3) as usize % 4;
        let (dx, dy) = directions[dir_idx];
        let nx = (self.x as i32 + dx).clamp(0, width as i32 - 1) as usize;
        let ny = (self.y as i32 + dy).clamp(0, height as i32 - 1) as usize;
        if nx == self.x && ny == self.y {
            return None;
        }

        // Split: parent keeps 60%, daughter gets 40%.
        let parent = &mut self.packets[bud_idx];
        let daughter_cells = parent.represented_cells * 0.4;
        parent.represented_cells *= 0.6;
        parent.reserve *= 0.8; // budding costs energy

        let daughter = GenotypePacket::new(
            parent.catalog_slot,
            parent.genotype_id,
            parent.lineage_id,
            daughter_cells,
        );

        self.recompute_total();
        Some((nx, ny, daughter))
    }
}
