use oneuro_metal::TerrariumWorld;

fn main() -> Result<(), String> {
    let mut world = TerrariumWorld::demo(7, true)?;
    world.run_frames(160)?;
    let snapshot = world.snapshot();

    println!("native_terrarium_stack_summary");
    println!("plants={}", snapshot.plants);
    println!("fruits={}", snapshot.fruits);
    println!("seeds={}", snapshot.seeds);
    println!("flies={}", snapshot.flies);
    println!("food_remaining={:.6}", snapshot.food_remaining);
    println!("fly_food_total={:.6}", snapshot.fly_food_total);
    println!("avg_fly_energy={:.6}", snapshot.avg_fly_energy);
    println!("avg_altitude={:.6}", snapshot.avg_altitude);
    println!("light={:.6}", snapshot.light);
    println!("temperature={:.6}", snapshot.temperature);
    println!("humidity={:.6}", snapshot.humidity);
    println!("soil_moisture={:.6}", snapshot.mean_soil_moisture);
    println!("deep_moisture={:.6}", snapshot.mean_deep_moisture);
    println!("soil_microbes={:.6}", snapshot.mean_microbes);
    println!("soil_symbionts={:.6}", snapshot.mean_symbionts);
    println!("canopy_cover={:.6}", snapshot.mean_canopy);
    println!("root_density={:.6}", snapshot.mean_root_density);
    println!("plant_cells={:.6}", snapshot.total_plant_cells);
    println!("cell_vitality={:.6}", snapshot.mean_cell_vitality);
    println!("cell_energy={:.6}", snapshot.mean_cell_energy);
    println!("division_pressure={:.6}", snapshot.mean_division_pressure);
    println!("soil_glucose={:.6}", snapshot.mean_soil_glucose);
    println!("soil_oxygen={:.6}", snapshot.mean_soil_oxygen);
    println!("soil_ammonium={:.6}", snapshot.mean_soil_ammonium);
    println!("soil_nitrate={:.6}", snapshot.mean_soil_nitrate);
    println!("soil_redox={:.6}", snapshot.mean_soil_redox);
    println!("soil_atp_flux={:.6}", snapshot.mean_soil_atp_flux);
    println!("substrate_backend={}", snapshot.substrate_backend);
    println!("substrate_steps={}", snapshot.substrate_steps);
    println!("substrate_time_ms={:.6}", snapshot.substrate_time_ms);
    println!("time={}", world.time_label());
    Ok(())
}
