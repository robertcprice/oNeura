use oneuro_metal::{BatchedAtomTerrarium, TerrariumSpecies};

fn main() {
    let mut terrarium = BatchedAtomTerrarium::new(24, 24, 12, 0.5, true);
    terrarium.add_hotspot(TerrariumSpecies::Glucose, 12, 12, 4, 1.5);
    terrarium.add_hotspot(TerrariumSpecies::Ammonium, 10, 13, 5, 0.8);

    let before = terrarium.snapshot();
    terrarium.run(600, 0.25);
    let after = terrarium.snapshot();

    println!("batched_atom_terrarium_summary");
    println!("backend={}", after.backend.as_str());
    println!("time_ms={:.3}", after.time_ms);
    println!("step_count={}", after.step_count);
    println!("mean_hydration={:.6}", after.mean_hydration);
    println!("mean_microbes={:.6}", after.mean_microbes);
    println!("mean_plant_drive={:.6}", after.mean_plant_drive);
    println!("mean_glucose_before={:.6}", before.mean_glucose);
    println!("mean_glucose_after={:.6}", after.mean_glucose);
    println!("mean_oxygen_gas_after={:.6}", after.mean_oxygen_gas);
    println!("mean_ammonium_after={:.6}", after.mean_ammonium);
    println!("mean_nitrate_after={:.6}", after.mean_nitrate);
    println!("mean_carbon_dioxide_after={:.6}", after.mean_carbon_dioxide);
    println!("mean_atp_flux_after={:.6}", after.mean_atp_flux);
    println!("elemental_carbon_after={:.6}", after.elemental_carbon);
    println!("elemental_nitrogen_after={:.6}", after.elemental_nitrogen);
}
