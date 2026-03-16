//! Pure chromosome geometry and seeding helpers extracted from `whole_cell.rs`.
//!
//! This module keeps circular-genome math and chromosome seed/normalize logic
//! out of the main simulator implementation.

use crate::whole_cell_data::{
    WholeCellChromosomeForkState, WholeCellChromosomeLocusState, WholeCellChromosomeState,
    WholeCellOrganismSpec,
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
            position_bp: origin_bp.min(genome_bp - 1),
            copy_number: 1.0,
            separation_nm: 0.0,
        },
        WholeCellChromosomeLocusState {
            id: "clockwise_arm".to_string(),
            position_bp: ((origin_bp as u64 + arm_quarter as u64) % genome_bp as u64) as u32,
            copy_number: 1.0,
            separation_nm: 0.0,
        },
        WholeCellChromosomeLocusState {
            id: "counterclockwise_arm".to_string(),
            position_bp: ((origin_bp as i64 - arm_quarter as i64).rem_euclid(genome_bp as i64))
                as u32,
            copy_number: 1.0,
            separation_nm: 0.0,
        },
        WholeCellChromosomeLocusState {
            id: "terminus".to_string(),
            position_bp: terminus_bp.min(genome_bp - 1),
            copy_number: 1.0,
            separation_nm: 0.0,
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
        genome_bp,
        origin_bp,
        terminus_bp,
        replicated_bp: 0,
        chromosome_separation_nm: chromosome_separation_nm.max(0.0),
        initiation_potential: 0.0,
        torsional_stress: 0.0,
        compaction: 1.0,
        forks: vec![
            WholeCellChromosomeForkState {
                id: "clockwise".to_string(),
                direction: 1,
                active: false,
                progress_bp: 0.0,
                position_bp: origin_bp,
                speed_bp_per_ms: 0.0,
            },
            WholeCellChromosomeForkState {
                id: "counterclockwise".to_string(),
                direction: -1,
                active: false,
                progress_bp: 0.0,
                position_bp: origin_bp,
                speed_bp_per_ms: 0.0,
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
    let genome_bp = chromosome.genome_bp.max(fallback_genome_bp.max(1));
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
            chromosome
                .chromosome_separation_nm
                .max(fallback_separation_nm),
        )
    } else {
        chromosome.clone()
    };
    normalized.genome_bp = genome_bp;
    normalized.origin_bp = origin_bp;
    normalized.terminus_bp = terminus_bp;
    normalized.replicated_bp = normalized.replicated_bp.min(genome_bp);
    normalized.chromosome_separation_nm = normalized
        .chromosome_separation_nm
        .max(fallback_separation_nm)
        .max(0.0);
    normalized.initiation_potential = normalized.initiation_potential.clamp(0.0, 2.0);
    normalized.torsional_stress = normalized.torsional_stress.clamp(0.0, 2.5);
    normalized.compaction = finite_scale(normalized.compaction, 1.0, 0.65, 1.75);
    for fork in &mut normalized.forks {
        fork.progress_bp = fork.progress_bp.clamp(0.0, 0.5 * genome_bp as f32);
        fork.position_bp = fork.position_bp.min(genome_bp - 1);
        fork.direction = if fork.direction >= 0 { 1 } else { -1 };
        fork.speed_bp_per_ms = fork.speed_bp_per_ms.max(0.0);
    }
    if normalized.loci.is_empty() {
        normalized.loci = default_chromosome_loci(genome_bp, origin_bp, terminus_bp);
    }
    for locus in &mut normalized.loci {
        locus.position_bp = locus.position_bp.min(genome_bp - 1);
        locus.copy_number = locus.copy_number.clamp(1.0, 2.0);
        locus.separation_nm = locus.separation_nm.max(0.0);
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

#[cfg(test)]
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
            genome_bp: 100,
            origin_bp: 0,
            terminus_bp: 50,
            replicated_bp: 150,
            chromosome_separation_nm: -5.0,
            initiation_potential: 4.0,
            torsional_stress: 9.0,
            compaction: f32::NAN,
            forks: Vec::new(),
            loci: Vec::new(),
        };

        let normalized = normalize_chromosome_state(&chromosome, 120, 10, 70, 25.0);

        assert_eq!(normalized.genome_bp, 120);
        assert_eq!(normalized.origin_bp, 10);
        assert_eq!(normalized.terminus_bp, 70);
        assert_eq!(normalized.replicated_bp, 0);
        assert_eq!(normalized.forks.len(), 2);
        assert_eq!(normalized.loci.len(), 4);
        assert!((normalized.chromosome_separation_nm - 25.0).abs() < 1.0e-6);
        assert!((normalized.compaction - 1.0).abs() < 1.0e-6);
    }
}
