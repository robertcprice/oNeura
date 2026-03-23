use std::collections::BTreeMap;

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrganismNameAssignment {
    pub organism_id: u64,
    pub display_name: String,
}

fn kind_label(kind: organism_identity::TerrariumOrganismKind) -> &'static str {
    match kind {
        organism_identity::TerrariumOrganismKind::Plant => "plant",
        organism_identity::TerrariumOrganismKind::Seed => "seed",
        organism_identity::TerrariumOrganismKind::Fruit => "fruit",
        organism_identity::TerrariumOrganismKind::Fly => "fly",
    }
}

fn format_seed_provenance(seed: &TerrariumSeedProvenance) -> String {
    let source = match seed.source {
        TerrariumSeedSource::Explicit => "explicit",
        TerrariumSeedSource::OsRandom => "os-random",
        TerrariumSeedSource::OpenEntropyMixed => "openentropy-mixed",
        TerrariumSeedSource::CheckpointRestore => "checkpoint-restore",
    };
    let mut detail = format!(
        "seed {} | source {} ({})",
        seed.seed, source, seed.source_label
    );
    if let (Some(total), Some(healthy)) = (
        seed.openentropy_total_sources,
        seed.openentropy_healthy_sources,
    ) {
        detail.push_str(&format!(" | openentropy healthy {healthy}/{total}"));
    }
    if seed.os_entropy_bytes > 0 {
        detail.push_str(&format!(" | os bytes {}", seed.os_entropy_bytes));
    }
    detail
}

fn format_climate(snapshot: &TerrariumWorldSnapshot) -> Option<String> {
    snapshot.climate.as_ref().map(|climate| {
        format!(
            "climate: year {:.3} | T {:.2} C | precip {:.1} mm | RH {:.2} | CO2 {:.1} ppm | wind {:.2} m/s | drought {:.2} | flood {:.2}",
            climate.year,
            climate.temperature_c,
            climate.precipitation_mm,
            climate.humidity,
            climate.co2_ppm,
            climate.wind_speed_m_s,
            climate.drought_severity,
            climate.flood_risk,
        )
    })
}

fn format_summary(
    label: &str,
    config: &TerrariumWorldConfig,
    snapshot: &TerrariumWorldSnapshot,
    seed: &TerrariumSeedProvenance,
) -> String {
    let preset_hint = TerrariumDemoPreset::infer_from_config(config)
        .map(|preset| preset.label().to_string())
        .unwrap_or_else(|| "Custom World".to_string());
    let explicit = &snapshot.conservation.explicit_domain_total;
    let mut lines = vec![
        format!("{label}"),
        format!("preset hint: {preset_hint}"),
        format_seed_provenance(seed),
        format!(
            "grid: {} x {} x {} | cell {:.3} mm | substrate {} | sim time {:.2} s",
            config.width,
            config.height,
            config.depth,
            config.cell_size_mm,
            snapshot.substrate_backend,
            snapshot.time_s,
        ),
        format!(
            "organisms: tracked {} | named {} | plants {} | fruits {} | seeds {} | adult flies {} | fly population {} | explicit microbes {}",
            snapshot.tracked_organisms,
            snapshot.named_organisms,
            snapshot.plants,
            snapshot.fruits,
            snapshot.seeds,
            snapshot.flies,
            snapshot.fly_population_total,
            snapshot.full_explicit_microbes.len(),
        ),
        format!(
            "conditions: T {:.2} C | RH {:.2} | soil {:.3} | deep soil {:.3} | pressure {:.2} kPa | lunar {:.3} | moonlight {:.3}",
            snapshot.temperature,
            snapshot.humidity,
            snapshot.mean_soil_moisture,
            snapshot.mean_deep_moisture,
            snapshot.mean_air_pressure_kpa,
            snapshot.lunar_phase,
            snapshot.moonlight,
        ),
        format!(
            "conservation explicit: H {:.4} | C {:.4} | N {:.4} | P {:.4} | O {:.4} | water {:.4} | energy {:.4}",
            explicit.hydrogen,
            explicit.carbon,
            explicit.nitrogen,
            explicit.phosphorus,
            explicit.oxygen,
            explicit.water,
            explicit.energy_equivalent,
        ),
        format!(
            "geochemistry explicit: Si {:.4} | Al {:.4} | Ca {:.4} | Mg {:.4} | K {:.4} | Na {:.4} | Fe {:.4}",
            explicit.silicon,
            explicit.aluminum,
            explicit.calcium,
            explicit.magnesium,
            explicit.potassium,
            explicit.sodium,
            explicit.iron,
        ),
    ];
    if let Some(climate_line) = format_climate(snapshot) {
        lines.push(climate_line);
    }
    lines.join("\n")
}

fn registry_counts(
    registry: &organism_identity::OrganismRegistry,
) -> BTreeMap<organism_identity::TerrariumOrganismKind, usize> {
    let mut counts = BTreeMap::new();
    for entry in registry.values() {
        *counts.entry(entry.identity.kind).or_default() += 1;
    }
    counts
}

fn lineage_from_registry(
    registry: &organism_identity::OrganismRegistry,
    organism_id: u64,
) -> Result<Vec<organism_identity::OrganismRegistryEntry>, String> {
    let mut lineage = Vec::new();
    let mut current = Some(organism_id);
    while let Some(id) = current {
        let Some(entry) = registry.get(&id) else {
            return Err(format!(
                "organism {organism_id} is not present in the registry"
            ));
        };
        lineage.push(entry.clone());
        current = entry.identity.parent_organism_id;
    }
    lineage.reverse();
    Ok(lineage)
}

fn format_registry_entry(entry: &organism_identity::OrganismRegistryEntry) -> String {
    let status = entry
        .death_time_s
        .map(|time_s| format!("dead @ {:.2}s", time_s))
        .unwrap_or_else(|| "alive".to_string());
    let mut detail = format!(
        "#{} | {} | {} | {} ({}) | gen {} | born {:.2}s | {}",
        entry.identity.organism_id,
        kind_label(entry.identity.kind),
        status,
        entry.preferred_name(),
        entry.scientific_name,
        entry.identity.generation,
        entry.identity.birth_time_s,
        entry.common_name,
    );
    if let Some(parent) = entry.identity.parent_organism_id {
        detail.push_str(&format!(" | parent #{parent}"));
    }
    if let Some(co_parent) = entry.identity.co_parent_organism_id {
        detail.push_str(&format!(" | co-parent #{co_parent}"));
    }
    if let Some(phylo_id) = entry.identity.phylo_id {
        detail.push_str(&format!(" | phylo #{phylo_id}"));
    }
    detail
}

fn format_registry_listing(registry: &organism_identity::OrganismRegistry) -> String {
    let counts = registry_counts(registry);
    let mut header = format!(
        "tracked organisms: {} | named {}",
        registry.len(),
        registry
            .values()
            .filter(|entry| entry.identity.display_name.is_some())
            .count()
    );
    for kind in [
        organism_identity::TerrariumOrganismKind::Plant,
        organism_identity::TerrariumOrganismKind::Fruit,
        organism_identity::TerrariumOrganismKind::Seed,
        organism_identity::TerrariumOrganismKind::Fly,
    ] {
        let count = counts.get(&kind).copied().unwrap_or(0);
        header.push_str(&format!(" | {} {}", kind_label(kind), count));
    }
    let mut lines = vec![header];
    lines.extend(registry.values().map(format_registry_entry));
    lines.join("\n")
}

fn set_archive_display_name(
    archive: &mut archive::TerrariumWorldArchive,
    assignment: &OrganismNameAssignment,
) -> bool {
    let trimmed = assignment.display_name.trim();
    if trimmed.is_empty() {
        return false;
    }
    let Some(entry) = archive.organism_registry.get_mut(&assignment.organism_id) else {
        return false;
    };
    entry.identity.display_name = Some(trimmed.to_string());
    for plant in &mut archive.snapshot.full_plants {
        if plant.organism_id == assignment.organism_id {
            plant.display_name = Some(trimmed.to_string());
        }
    }
    for fruit in &mut archive.snapshot.full_fruits {
        if fruit.organism_id == assignment.organism_id {
            fruit.display_name = Some(trimmed.to_string());
        }
    }
    for seed in &mut archive.snapshot.full_seeds {
        if seed.organism_id == assignment.organism_id {
            seed.display_name = Some(trimmed.to_string());
        }
    }
    for fly in &mut archive.snapshot.full_flies {
        if fly.organism_id == assignment.organism_id {
            fly.display_name = Some(trimmed.to_string());
        }
    }
    for fly in &mut archive.snapshot.full_fly_population {
        if fly.organism_id == Some(assignment.organism_id) {
            fly.display_name = Some(trimmed.to_string());
        }
    }
    archive.snapshot.named_organisms = archive
        .organism_registry
        .values()
        .filter(|entry| entry.identity.display_name.is_some())
        .count();
    archive.snapshot.tracked_organisms = archive.organism_registry.len();
    true
}

pub fn parse_name_assignment(spec: &str) -> Result<OrganismNameAssignment, String> {
    let (id, name) = spec
        .split_once('=')
        .ok_or_else(|| format!("invalid --name-organism value '{spec}'; expected <ID>=<NAME>"))?;
    let organism_id = id
        .trim()
        .parse::<u64>()
        .map_err(|_| format!("invalid organism id in --name-organism value '{spec}'"))?;
    let display_name = name.trim();
    if display_name.is_empty() {
        return Err(format!(
            "invalid --name-organism value '{spec}'; organism name cannot be empty"
        ));
    }
    Ok(OrganismNameAssignment {
        organism_id,
        display_name: display_name.to_string(),
    })
}

pub fn apply_world_name_assignments(
    world: &mut TerrariumWorld,
    assignments: &[OrganismNameAssignment],
) -> Result<(), String> {
    for assignment in assignments {
        if !world.set_organism_name(assignment.organism_id, assignment.display_name.clone()) {
            return Err(format!(
                "organism {} is not present in the live world",
                assignment.organism_id
            ));
        }
    }
    Ok(())
}

pub fn apply_archive_name_assignments(
    archive: &mut archive::TerrariumWorldArchive,
    assignments: &[OrganismNameAssignment],
) -> Result<(), String> {
    for assignment in assignments {
        if !set_archive_display_name(archive, assignment) {
            return Err(format!(
                "organism {} is not present in the archive registry",
                assignment.organism_id
            ));
        }
    }
    Ok(())
}

pub fn format_world_summary(world: &TerrariumWorld) -> String {
    format_summary(
        "Live Terrarium World",
        &world.config,
        &world.snapshot(),
        world.seed_provenance(),
    )
}

pub fn format_archive_summary(archive: &archive::TerrariumWorldArchive) -> String {
    format_summary(
        "Terrarium Archive",
        &archive.config,
        &archive.snapshot,
        &archive.seed_provenance,
    )
}

pub fn format_world_organism_listing(world: &TerrariumWorld) -> String {
    format_registry_listing(&world.organism_registry)
}

pub fn format_archive_organism_listing(archive: &archive::TerrariumWorldArchive) -> String {
    format_registry_listing(&archive.organism_registry)
}

pub fn format_world_lineage(world: &TerrariumWorld, organism_id: u64) -> Result<String, String> {
    let lineage = lineage_from_registry(&world.organism_registry, organism_id)?;
    let mut lines = vec![format!("lineage for organism #{organism_id}")];
    lines.extend(lineage.iter().map(format_registry_entry));
    Ok(lines.join("\n"))
}

pub fn format_archive_lineage(
    archive: &archive::TerrariumWorldArchive,
    organism_id: u64,
) -> Result<String, String> {
    let lineage = lineage_from_registry(&archive.organism_registry, organism_id)?;
    let mut lines = vec![format!("lineage for organism #{organism_id}")];
    lines.extend(lineage.iter().map(format_registry_entry));
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_name_assignment_preserves_inner_equals() {
        let parsed = parse_name_assignment("42=Fig = Alpha").unwrap();
        assert_eq!(parsed.organism_id, 42);
        assert_eq!(parsed.display_name, "Fig = Alpha");
    }

    #[test]
    fn archive_name_assignment_updates_registry_and_snapshot() {
        let world = TerrariumWorld::demo_preset(7, false, TerrariumDemoPreset::MicroTerrarium)
            .expect("demo world should build");
        let plant_id = world
            .snapshot()
            .full_plants
            .first()
            .expect("demo world should have a plant")
            .organism_id;
        let mut archive = world.archive();

        apply_archive_name_assignments(
            &mut archive,
            &[OrganismNameAssignment {
                organism_id: plant_id,
                display_name: "Named Plant".into(),
            }],
        )
        .expect("archive naming should succeed");

        assert_eq!(
            archive
                .organism_registry
                .get(&plant_id)
                .and_then(|entry| entry.identity.display_name.as_deref()),
            Some("Named Plant")
        );
        assert_eq!(
            archive
                .snapshot
                .full_plants
                .iter()
                .find(|plant| plant.organism_id == plant_id)
                .and_then(|plant| plant.display_name.as_deref()),
            Some("Named Plant")
        );
    }
}
