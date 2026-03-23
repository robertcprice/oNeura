//! Pure chromosome geometry and seeding helpers extracted from `whole_cell.rs`.
//!
//! This module keeps circular-genome math and chromosome seed/normalize logic
//! out of the main simulator implementation.

use crate::whole_cell_data::{
    WholeCellChromosomeForkDirection, WholeCellChromosomeForkState, WholeCellChromosomeLocusState,
    WholeCellChromosomeState, WholeCellOrganismSpec,
};

fn finite_scale(value: f32, fallback: f32, min_value: f32, max_value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min_value, max_value)
    } else {
        fallback.clamp(min_value, max_value)
    }
}

pub(crate) fn directional_distance_bp(
    origin_bp: u32,
    position_bp: u32,
    direction: i8,
    genome_bp: u32,
) -> f32 {
    let genome_bp = genome_bp.max(1) as i64;
    let origin_bp = origin_bp.min(genome_bp as u32 - 1) as i64;
    let position_bp = position_bp.min(genome_bp as u32 - 1) as i64;
    if direction >= 0 {
        ((position_bp - origin_bp).rem_euclid(genome_bp)) as f32
    } else {
        ((origin_bp - position_bp).rem_euclid(genome_bp)) as f32
    }
}

pub(crate) fn advance_circular_position(
    origin_bp: u32,
    progress_bp: f32,
    direction: i8,
    genome_bp: u32,
) -> u32 {
    let genome_bp = genome_bp.max(1);
    let origin_bp = origin_bp.min(genome_bp - 1) as i64;
    let offset = progress_bp.max(0.0).round() as i64;
    let signed = if direction >= 0 {
        origin_bp + offset
    } else {
        origin_bp - offset
    };
    signed.rem_euclid(genome_bp as i64) as u32
}

pub(crate) fn default_chromosome_loci(
    genome_bp: u32,
    origin_bp: u32,
    terminus_bp: u32,
) -> Vec<WholeCellChromosomeLocusState> {
    let genome_bp = genome_bp.max(1);
    let arm_quarter = (genome_bp / 4).max(1);
    vec![
        WholeCellChromosomeLocusState {
            id: "origin".to_string(),
            midpoint_bp: origin_bp.min(genome_bp - 1),
            strand: 0,
            copy_number: 1.0,
            accessibility: 1.0,
            torsional_stress: 0.0,
            replicated: false,
            segregating: false,
            domain_index: 0,
        },
        WholeCellChromosomeLocusState {
            id: "clockwise_arm".to_string(),
            midpoint_bp: ((origin_bp as u64 + arm_quarter as u64) % genome_bp as u64) as u32,
            strand: 0,
            copy_number: 1.0,
            accessibility: 1.0,
            torsional_stress: 0.0,
            replicated: false,
            segregating: false,
            domain_index: 0,
        },
        WholeCellChromosomeLocusState {
            id: "counterclockwise_arm".to_string(),
            midpoint_bp: ((origin_bp as i64 - arm_quarter as i64).rem_euclid(genome_bp as i64))
                as u32,
            strand: 0,
            copy_number: 1.0,
            accessibility: 1.0,
            torsional_stress: 0.0,
            replicated: false,
            segregating: false,
            domain_index: 0,
        },
        WholeCellChromosomeLocusState {
            id: "terminus".to_string(),
            midpoint_bp: terminus_bp.min(genome_bp - 1),
            strand: 0,
            copy_number: 1.0,
            accessibility: 1.0,
            torsional_stress: 0.0,
            replicated: false,
            segregating: false,
            domain_index: 0,
        },
    ]
}

pub(crate) fn seeded_chromosome_state(
    genome_bp: u32,
    origin_bp: u32,
    terminus_bp: u32,
    chromosome_separation_nm: f32,
) -> WholeCellChromosomeState {
    let genome_bp = genome_bp.max(1);
    let origin_bp = origin_bp.min(genome_bp - 1);
    let terminus_bp = terminus_bp.min(genome_bp - 1);
    WholeCellChromosomeState {
        chromosome_length_bp: genome_bp,
        origin_bp,
        terminus_bp,
        replicated_bp: 0,
        replicated_fraction: 0.0,
        segregation_progress: chromosome_separation_nm.max(0.0),
        initiation_potential: 0.0,
        initiation_events: 0,
        completion_events: 0,
        torsional_stress: 0.0,
        compaction_fraction: 1.0,
        mean_locus_accessibility: 1.0,
        forks: vec![
            WholeCellChromosomeForkState {
                id: "clockwise".to_string(),
                direction: WholeCellChromosomeForkDirection::Clockwise,
                active: false,
                traveled_bp: 0,
                position_bp: origin_bp,
                paused: false,
                pause_pressure: 0.0,
                collision_pressure: 0.0,
                pause_events: 0,
                completion_fraction: 0.0,
                completed: false,
            },
            WholeCellChromosomeForkState {
                id: "counterclockwise".to_string(),
                direction: WholeCellChromosomeForkDirection::CounterClockwise,
                active: false,
                traveled_bp: 0,
                position_bp: origin_bp,
                paused: false,
                pause_pressure: 0.0,
                collision_pressure: 0.0,
                pause_events: 0,
                completion_fraction: 0.0,
                completed: false,
            },
        ],
        loci: default_chromosome_loci(genome_bp, origin_bp, terminus_bp),
    }
}

pub(crate) fn normalize_chromosome_state(
    chromosome: &WholeCellChromosomeState,
    fallback_genome_bp: u32,
    fallback_origin_bp: u32,
    fallback_terminus_bp: u32,
    fallback_separation_nm: f32,
) -> WholeCellChromosomeState {
    let genome_bp = chromosome
        .chromosome_length_bp
        .max(fallback_genome_bp.max(1));
    let origin_bp = chromosome
        .origin_bp
        .min(genome_bp - 1)
        .max(fallback_origin_bp.min(genome_bp - 1));
    let terminus_bp = chromosome
        .terminus_bp
        .min(genome_bp - 1)
        .max(fallback_terminus_bp.min(genome_bp - 1));
    let mut normalized = if chromosome.forks.is_empty() && chromosome.loci.is_empty() {
        seeded_chromosome_state(
            genome_bp,
            origin_bp,
            terminus_bp,
            chromosome.segregation_progress.max(fallback_separation_nm),
        )
    } else {
        chromosome.clone()
    };
    normalized.chromosome_length_bp = genome_bp;
    normalized.origin_bp = origin_bp;
    normalized.terminus_bp = terminus_bp;
    normalized.replicated_bp = normalized.replicated_bp.min(genome_bp);
    normalized.segregation_progress = normalized
        .segregation_progress
        .max(fallback_separation_nm)
        .max(0.0);
    normalized.initiation_potential = normalized.initiation_potential.clamp(0.0, 2.0);
    normalized.torsional_stress = normalized.torsional_stress.clamp(0.0, 2.5);
    normalized.compaction_fraction = finite_scale(normalized.compaction_fraction, 1.0, 0.65, 1.75);
    for fork in &mut normalized.forks {
        fork.traveled_bp = (fork.traveled_bp as f32).clamp(0.0, 0.5 * genome_bp as f32) as u32;
        fork.position_bp = fork.position_bp.min(genome_bp - 1);
        // direction is an enum — no normalization needed
    }
    if normalized.loci.is_empty() {
        normalized.loci = default_chromosome_loci(genome_bp, origin_bp, terminus_bp);
    }
    for locus in &mut normalized.loci {
        locus.midpoint_bp = locus.midpoint_bp.min(genome_bp - 1);
        locus.copy_number = locus.copy_number.clamp(1.0, 2.0);
        locus.torsional_stress = locus.torsional_stress.max(0.0);
    }
    normalized
}

pub(crate) fn organism_origin_bp(organism: &WholeCellOrganismSpec) -> u32 {
    organism
        .origin_bp
        .min(organism.chromosome_length_bp.max(1) - 1)
}

pub(crate) fn organism_terminus_bp(organism: &WholeCellOrganismSpec) -> u32 {
    organism
        .terminus_bp
        .min(organism.chromosome_length_bp.max(1) - 1)
}

pub(crate) fn circular_feature_length_bp(start_bp: u32, end_bp: u32, genome_bp: u32) -> u32 {
    let genome_bp = genome_bp.max(1);
    let start_bp = start_bp.min(genome_bp - 1);
    let end_bp = end_bp.min(genome_bp - 1);
    if end_bp >= start_bp {
        end_bp - start_bp + 1
    } else {
        genome_bp - start_bp + end_bp + 1
    }
}

pub(crate) fn circular_feature_midpoint_bp(start_bp: u32, end_bp: u32, genome_bp: u32) -> u32 {
    let genome_bp = genome_bp.max(1);
    let start_bp = start_bp.min(genome_bp - 1);
    let span_bp = circular_feature_length_bp(start_bp, end_bp, genome_bp).max(1);
    advance_circular_position(
        start_bp,
        0.5 * span_bp.saturating_sub(1) as f32,
        1,
        genome_bp,
    )
}

#[cfg(all(test, feature = "satellite_tests"))]
mod tests {
    use super::*;

    #[test]
    fn directional_distance_respects_direction() {
        assert!((directional_distance_bp(10, 30, 1, 100) - 20.0).abs() < 1.0e-6);
        assert!((directional_distance_bp(10, 30, -1, 100) - 80.0).abs() < 1.0e-6);
    }

    #[test]
    fn circular_feature_midpoint_wraps_across_origin() {
        assert_eq!(circular_feature_midpoint_bp(90, 10, 100), 0);
    }

    #[test]
    fn normalize_chromosome_state_seeds_missing_geometry() {
        let chromosome = WholeCellChromosomeState {
            chromosome_length_bp: 100,
            origin_bp: 0,
            terminus_bp: 50,
            replicated_bp: 150,
            replicated_fraction: 0.0,
            segregation_progress: -5.0,
            initiation_potential: 4.0,
            initiation_events: 0,
            completion_events: 0,
            torsional_stress: 9.0,
            compaction_fraction: f32::NAN,
            mean_locus_accessibility: 1.0,
            forks: Vec::new(),
            loci: Vec::new(),
        };

        let normalized = normalize_chromosome_state(&chromosome, 120, 10, 70, 25.0);

        assert_eq!(normalized.chromosome_length_bp, 120);
        assert_eq!(normalized.origin_bp, 10);
        assert_eq!(normalized.terminus_bp, 70);
        assert_eq!(normalized.replicated_bp, 0);
        assert_eq!(normalized.forks.len(), 2);
        assert_eq!(normalized.loci.len(), 4);
        assert!((normalized.segregation_progress - 25.0).abs() < 1.0e-6);
        assert!((normalized.compaction_fraction - 1.0).abs() < 1.0e-6);
    }
}
