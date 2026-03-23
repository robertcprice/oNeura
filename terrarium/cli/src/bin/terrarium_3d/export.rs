//! CSV data export (E key), JSON scene export (J key), time-lapse BMP recording (V key).

use oneura_core::terrarium::{TerrariumWorld, TerrariumWorldSnapshot};
use std::fs::{self, File};
use std::io::Write;

/// Export a full simulation snapshot to CSV files in a timestamped directory.
pub fn export_snapshot(
    world: &TerrariumWorld,
    snapshot: &TerrariumWorldSnapshot,
    frame: usize,
) -> Result<String, String> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let dir = format!("terrarium_export_{}", ts);
    fs::create_dir_all(&dir).map_err(|e| format!("{}", e))?;

    // Summary CSV
    {
        let mut f = File::create(format!("{}/summary.csv", dir)).map_err(|e| format!("{}", e))?;
        writeln!(f, "field,value").map_err(|e| format!("{}", e))?;
        writeln!(f, "frame,{}", frame).map_err(|e| format!("{}", e))?;
        writeln!(f, "plants,{}", snapshot.plants).map_err(|e| format!("{}", e))?;
        writeln!(f, "flies,{}", snapshot.flies).map_err(|e| format!("{}", e))?;
        writeln!(f, "seeds,{}", snapshot.seeds).map_err(|e| format!("{}", e))?;
        writeln!(f, "fruits,{}", snapshot.fruits).map_err(|e| format!("{}", e))?;
        writeln!(f, "food_remaining,{:.4}", snapshot.food_remaining)
            .map_err(|e| format!("{}", e))?;
        writeln!(f, "fly_food_total,{:.4}", snapshot.fly_food_total)
            .map_err(|e| format!("{}", e))?;
        writeln!(f, "light,{:.4}", snapshot.light).map_err(|e| format!("{}", e))?;
        writeln!(f, "humidity,{:.4}", snapshot.humidity).map_err(|e| format!("{}", e))?;
        writeln!(f, "temperature,{:.4}", snapshot.temperature).map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_cell_vitality,{:.4}", snapshot.mean_cell_vitality)
            .map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_cell_energy,{:.4}", snapshot.mean_cell_energy)
            .map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_soil_moisture,{:.4}", snapshot.mean_soil_moisture)
            .map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_deep_moisture,{:.4}", snapshot.mean_deep_moisture)
            .map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_microbes,{:.6}", snapshot.mean_microbes).map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_symbionts,{:.6}", snapshot.mean_symbionts)
            .map_err(|e| format!("{}", e))?;
        writeln!(f, "total_plant_cells,{:.0}", snapshot.total_plant_cells)
            .map_err(|e| format!("{}", e))?;
        writeln!(
            f,
            "mean_atmospheric_co2,{:.6}",
            snapshot.mean_atmospheric_co2
        )
        .map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_atmospheric_o2,{:.4}", snapshot.mean_atmospheric_o2)
            .map_err(|e| format!("{}", e))?;
    }

    // Plants CSV
    {
        let mut f = File::create(format!("{}/plants.csv", dir)).map_err(|e| format!("{}", e))?;
        writeln!(f, "id,x,y,height_mm,vitality,energy_charge,total_cells")
            .map_err(|e| format!("{}", e))?;
        for (i, plant) in world.plants.iter().enumerate() {
            writeln!(
                f,
                "{},{},{},{:.2},{:.4},{:.4},{:.0}",
                i,
                plant.x,
                plant.y,
                plant.physiology.height_mm(),
                plant.cellular.vitality(),
                plant.cellular.energy_charge(),
                plant.cellular.total_cells(),
            )
            .map_err(|e| format!("{}", e))?;
        }
    }

    // Flies CSV
    {
        let mut f = File::create(format!("{}/flies.csv", dir)).map_err(|e| format!("{}", e))?;
        writeln!(f, "id,x,y,z,speed,heading,energy,is_flying").map_err(|e| format!("{}", e))?;
        for (i, fly) in world.flies.iter().enumerate() {
            let b = fly.body_state();
            writeln!(
                f,
                "{},{:.3},{:.3},{:.3},{:.4},{:.3},{:.4},{}",
                i, b.x, b.y, b.z, b.speed, b.heading, b.energy, b.is_flying,
            )
            .map_err(|e| format!("{}", e))?;
        }
    }

    // Soil moisture grid CSV
    {
        let moisture = world.moisture_field();
        let gw = world.config.width;
        let gh = world.config.height;
        let mut f =
            File::create(format!("{}/soil_moisture.csv", dir)).map_err(|e| format!("{}", e))?;
        writeln!(f, "x,y,moisture").map_err(|e| format!("{}", e))?;
        for gy in 0..gh {
            for gx in 0..gw {
                let mi = gy * gw + gx;
                let m = if mi < moisture.len() {
                    moisture[mi]
                } else {
                    0.0
                };
                writeln!(f, "{},{},{:.4}", gx, gy, m).map_err(|e| format!("{}", e))?;
            }
        }
    }

    // Water entities CSV
    {
        let mut f = File::create(format!("{}/water.csv", dir)).map_err(|e| format!("{}", e))?;
        writeln!(f, "id,x,y,volume,alive").map_err(|e| format!("{}", e))?;
        for (i, water) in world.waters.iter().enumerate() {
            writeln!(
                f,
                "{},{},{},{:.4},{}",
                i, water.x, water.y, water.volume, water.alive
            )
            .map_err(|e| format!("{}", e))?;
        }
    }

    // Fruits CSV
    {
        let mut f = File::create(format!("{}/fruits.csv", dir)).map_err(|e| format!("{}", e))?;
        writeln!(f, "id,x,y,sugar,ripeness,alive").map_err(|e| format!("{}", e))?;
        for (i, fruit) in world.fruits.iter().enumerate() {
            writeln!(
                f,
                "{},{},{},{:.4},{:.4},{}",
                i,
                fruit.source.x,
                fruit.source.y,
                fruit.source.sugar_content,
                fruit.source.ripeness,
                fruit.source.alive
            )
            .map_err(|e| format!("{}", e))?;
        }
    }

    Ok(dir)
}

/// Export full simulation state as a JSON file for external analysis.
pub fn export_json(
    world: &TerrariumWorld,
    snapshot: &TerrariumWorldSnapshot,
    frame: usize,
) -> Result<String, String> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let filename = format!("terrarium_scene_{}.json", ts);

    let plants_json: Vec<String> = world.plants.iter().enumerate().map(|(i, p)| {
        format!(
            r#"{{"id":{},"x":{},"y":{},"height_mm":{:.2},"vitality":{:.4},"energy_charge":{:.4},"total_cells":{:.0}}}"#,
            i, p.x, p.y, p.physiology.height_mm(), p.cellular.vitality(), p.cellular.energy_charge(), p.cellular.total_cells()
        )
    }).collect();

    let flies_json: Vec<String> = world.flies.iter().enumerate().map(|(i, f)| {
        let b = f.body_state();
        format!(
            r#"{{"id":{},"x":{:.3},"y":{:.3},"z":{:.3},"speed":{:.4},"heading":{:.3},"energy":{:.4},"is_flying":{}}}"#,
            i, b.x, b.y, b.z, b.speed, b.heading, b.energy, b.is_flying
        )
    }).collect();

    let waters_json: Vec<String> = world
        .waters
        .iter()
        .enumerate()
        .map(|(i, w)| {
            format!(
                r#"{{"id":{},"x":{},"y":{},"volume":{:.4},"alive":{}}}"#,
                i, w.x, w.y, w.volume, w.alive
            )
        })
        .collect();

    let fruits_json: Vec<String> = world
        .fruits
        .iter()
        .enumerate()
        .map(|(i, f)| {
            format!(
                r#"{{"id":{},"x":{},"y":{},"sugar":{:.4},"ripeness":{:.4},"alive":{}}}"#,
                i,
                f.source.x,
                f.source.y,
                f.source.sugar_content,
                f.source.ripeness,
                f.source.alive
            )
        })
        .collect();

    // Substrate grid summary (first z-layer, every 4th cell for compactness)
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();
    let mut soil_json = Vec::new();
    for gy in (0..gh).step_by(4) {
        for gx in (0..gw).step_by(4) {
            let mi = gy * gw + gx;
            let m = if mi < moisture.len() {
                moisture[mi]
            } else {
                0.0
            };
            soil_json.push(format!(r#"{{"x":{},"y":{},"moisture":{:.3}}}"#, gx, gy, m));
        }
    }

    let json = format!(
        r#"{{"frame":{},"timestamp":{},"environment":{{"light":{:.4},"humidity":{:.4},"temperature":{:.2},"mean_soil_moisture":{:.4},"mean_atmospheric_co2":{:.6},"mean_atmospheric_o2":{:.4},"moonlight":{:.4},"tidal_factor":{:.4}}},"counts":{{"plants":{},"flies":{},"seeds":{},"fruits":{}}},"plants":[{}],"flies":[{}],"waters":[{}],"fruits":[{}],"soil_grid":[{}]}}"#,
        frame,
        ts,
        snapshot.light,
        snapshot.humidity,
        snapshot.temperature,
        snapshot.mean_soil_moisture,
        snapshot.mean_atmospheric_co2,
        snapshot.mean_atmospheric_o2,
        snapshot.moonlight,
        snapshot.tidal_moisture_factor,
        snapshot.plants,
        snapshot.flies,
        snapshot.seeds,
        snapshot.fruits,
        plants_json.join(","),
        flies_json.join(","),
        waters_json.join(","),
        fruits_json.join(","),
        soil_json.join(","),
    );

    let mut f = File::create(&filename).map_err(|e| format!("{}", e))?;
    f.write_all(json.as_bytes()).map_err(|e| format!("{}", e))?;
    Ok(filename)
}

/// Export Prometheus-compatible metrics file for monitoring dashboards.
pub fn export_metrics(snapshot: &TerrariumWorldSnapshot, frame: usize) -> Result<String, String> {
    let filename = "terrarium_metrics.prom";
    let mut f = File::create(filename).map_err(|e| format!("{}", e))?;
    let w = |f: &mut File, name: &str, help: &str, val: f64| -> Result<(), String> {
        writeln!(f, "# HELP terrarium_{} {}", name, help).map_err(|e| format!("{}", e))?;
        writeln!(f, "# TYPE terrarium_{} gauge", name).map_err(|e| format!("{}", e))?;
        writeln!(f, "terrarium_{} {:.6}", name, val).map_err(|e| format!("{}", e))
    };
    w(&mut f, "frame", "Current simulation frame", frame as f64)?;
    w(
        &mut f,
        "plants_total",
        "Number of living plants",
        snapshot.plants as f64,
    )?;
    w(
        &mut f,
        "flies_total",
        "Number of living flies",
        snapshot.flies as f64,
    )?;
    w(
        &mut f,
        "seeds_total",
        "Number of seeds",
        snapshot.seeds as f64,
    )?;
    w(
        &mut f,
        "fruits_total",
        "Number of fruits",
        snapshot.fruits as f64,
    )?;
    w(
        &mut f,
        "light",
        "Day/night light level",
        snapshot.light as f64,
    )?;
    w(
        &mut f,
        "humidity",
        "Environmental humidity",
        snapshot.humidity as f64,
    )?;
    w(
        &mut f,
        "temperature_celsius",
        "Temperature in Celsius",
        snapshot.temperature as f64,
    )?;
    w(
        &mut f,
        "soil_moisture_mean",
        "Mean soil moisture",
        snapshot.mean_soil_moisture as f64,
    )?;
    w(
        &mut f,
        "deep_moisture_mean",
        "Mean deep moisture",
        snapshot.mean_deep_moisture as f64,
    )?;
    w(
        &mut f,
        "cell_vitality_mean",
        "Mean cell vitality",
        snapshot.mean_cell_vitality as f64,
    )?;
    w(
        &mut f,
        "cell_energy_mean",
        "Mean cell energy",
        snapshot.mean_cell_energy as f64,
    )?;
    w(
        &mut f,
        "microbes_mean",
        "Mean microbial density",
        snapshot.mean_microbes as f64,
    )?;
    w(
        &mut f,
        "symbionts_mean",
        "Mean symbiont density",
        snapshot.mean_symbionts as f64,
    )?;
    w(
        &mut f,
        "atmospheric_co2",
        "Mean atmospheric CO2",
        snapshot.mean_atmospheric_co2 as f64,
    )?;
    w(
        &mut f,
        "atmospheric_o2",
        "Mean atmospheric O2",
        snapshot.mean_atmospheric_o2 as f64,
    )?;
    w(
        &mut f,
        "moonlight",
        "Moonlight intensity",
        snapshot.moonlight as f64,
    )?;
    w(
        &mut f,
        "tidal_factor",
        "Tidal moisture factor",
        snapshot.tidal_moisture_factor as f64,
    )?;
    w(
        &mut f,
        "plant_cells_total",
        "Total plant cells",
        snapshot.total_plant_cells as f64,
    )?;
    w(
        &mut f,
        "food_remaining",
        "Total food remaining",
        snapshot.food_remaining as f64,
    )?;
    Ok(filename.to_string())
}

/// Generate a Python analysis notebook that loads exported JSON data.
pub fn generate_notebook(json_file: &str) -> Result<String, String> {
    let filename = "terrarium_analysis.py";
    let content = format!(
        r#"#!/usr/bin/env python3
"""oNeura Terrarium Analysis — auto-generated from 3D viewer export.

Usage: python terrarium_analysis.py
Reads: {json_file}
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load exported scene
with open("{json_file}") as f:
    data = json.load(f)

env = data["environment"]
counts = data["counts"]
plants = data["plants"]
flies = data["flies"]

print(f"Frame {{data['frame']}} | Plants: {{counts['plants']}} | Flies: {{counts['flies']}}")
print(f"Light: {{env['light']:.2f}} | Temp: {{env['temperature']:.1f}}C | Humidity: {{env['humidity']:.2f}}")
print(f"CO2: {{env['mean_atmospheric_co2']:.4f}} | O2: {{env['mean_atmospheric_o2']:.2f}}")
print(f"Moonlight: {{env['moonlight']:.2f}} | Tidal: {{env['tidal_factor']:.3f}}")

# --- Figure 1: Spatial distribution ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plants
if plants:
    px = [p["x"] for p in plants]
    py = [p["y"] for p in plants]
    ph = [p["height_mm"] for p in plants]
    axes[0].scatter(px, py, c=ph, cmap="Greens", s=20, edgecolors="k", linewidths=0.3)
    axes[0].set_title(f"Plants (n={{len(plants)}})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

# Flies
if flies:
    fx = [f["x"] for f in flies]
    fy = [f["y"] for f in flies]
    fe = [f["energy"] for f in flies]
    axes[1].scatter(fx, fy, c=fe, cmap="YlOrRd", s=15, edgecolors="k", linewidths=0.3)
    axes[1].set_title(f"Flies (n={{len(flies)}})")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

# Soil moisture heatmap
soil = data.get("soil_grid", [])
if soil:
    xs = sorted(set(s["x"] for s in soil))
    ys = sorted(set(s["y"] for s in soil))
    grid = np.zeros((len(ys), len(xs)))
    x_map = {{x: i for i, x in enumerate(xs)}}
    y_map = {{y: i for i, y in enumerate(ys)}}
    for s in soil:
        grid[y_map[s["y"]], x_map[s["x"]]] = s["moisture"]
    axes[2].imshow(grid, cmap="Blues", origin="lower", aspect="auto")
    axes[2].set_title("Soil Moisture")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")

plt.tight_layout()
plt.savefig("terrarium_spatial.png", dpi=150)
print("Saved: terrarium_spatial.png")

# --- Figure 2: Plant vitality distribution ---
if plants:
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
    vit = [p["vitality"] for p in plants]
    ec = [p["energy_charge"] for p in plants]
    ax2[0].hist(vit, bins=20, color="green", alpha=0.7, edgecolor="k")
    ax2[0].set_title("Plant Vitality Distribution")
    ax2[0].set_xlabel("Vitality"); ax2[0].set_ylabel("Count")
    ax2[1].hist(ec, bins=20, color="orange", alpha=0.7, edgecolor="k")
    ax2[1].set_title("Plant Energy Charge Distribution")
    ax2[1].set_xlabel("Energy Charge"); ax2[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("terrarium_plant_stats.png", dpi=150)
    print("Saved: terrarium_plant_stats.png")

print("Analysis complete.")
"#,
        json_file = json_file
    );

    let mut f = File::create(filename).map_err(|e| format!("{}", e))?;
    f.write_all(content.as_bytes())
        .map_err(|e| format!("{}", e))?;
    Ok(filename.to_string())
}

/// State for time-lapse BMP sequence recording.
pub struct TimeLapse {
    pub recording: bool,
    pub frame_count: usize,
    pub dir: String,
}

impl TimeLapse {
    pub fn new() -> Self {
        Self {
            recording: false,
            frame_count: 0,
            dir: String::new(),
        }
    }

    /// Toggle recording on/off. Returns status message.
    pub fn toggle(&mut self) -> String {
        if self.recording {
            self.recording = false;
            format!(
                "Recording stopped ({} frames in {})",
                self.frame_count, self.dir
            )
        } else {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            self.dir = format!("terrarium_timelapse_{}", ts);
            if let Err(e) = fs::create_dir_all(&self.dir) {
                return format!("Record error: {}", e);
            }
            self.frame_count = 0;
            self.recording = true;
            format!("Recording to {}/", self.dir)
        }
    }

    /// Save current viewport frame as a numbered BMP. Returns Ok on success.
    pub fn save_frame(
        &mut self,
        buffer: &[u32],
        width: usize,
        height: usize,
    ) -> Result<(), String> {
        if !self.recording {
            return Ok(());
        }
        let filename = format!("{}/frame_{:06}.bmp", self.dir, self.frame_count);
        save_bmp(&filename, buffer, width, height)?;
        self.frame_count += 1;
        Ok(())
    }
}

fn save_bmp(path: &str, buffer: &[u32], width: usize, height: usize) -> Result<(), String> {
    let row_size = ((width * 3 + 3) / 4) * 4;
    let data_size = row_size * height;
    let file_size = 54 + data_size;

    let mut file = File::create(path).map_err(|e| format!("{}", e))?;
    let mut header = vec![0u8; 54];
    header[0] = b'B';
    header[1] = b'M';
    header[2..6].copy_from_slice(&(file_size as u32).to_le_bytes());
    header[10..14].copy_from_slice(&54u32.to_le_bytes());
    header[14..18].copy_from_slice(&40u32.to_le_bytes());
    header[18..22].copy_from_slice(&(width as u32).to_le_bytes());
    header[22..26].copy_from_slice(&(height as u32).to_le_bytes());
    header[26..28].copy_from_slice(&1u16.to_le_bytes());
    header[28..30].copy_from_slice(&24u16.to_le_bytes());
    header[34..38].copy_from_slice(&(data_size as u32).to_le_bytes());
    file.write_all(&header).map_err(|e| format!("{}", e))?;

    let mut row_buf = vec![0u8; row_size];
    for y in (0..height).rev() {
        for x in 0..width {
            let c = buffer[y * width + x];
            let offset = x * 3;
            row_buf[offset] = (c & 0xff) as u8;
            row_buf[offset + 1] = ((c >> 8) & 0xff) as u8;
            row_buf[offset + 2] = ((c >> 16) & 0xff) as u8;
        }
        for i in (width * 3)..row_size {
            row_buf[i] = 0;
        }
        file.write_all(&row_buf).map_err(|e| format!("{}", e))?;
    }
    Ok(())
}
