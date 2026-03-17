//! CSV data export (E key) + time-lapse BMP recording (V key).

use std::fs::{self, File};
use std::io::Write;
use oneuro_metal::{TerrariumWorld, TerrariumWorldSnapshot};

/// Export a full simulation snapshot to CSV files in a timestamped directory.
pub fn export_snapshot(world: &TerrariumWorld, snapshot: &TerrariumWorldSnapshot, frame: usize) -> Result<String, String> {
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
        writeln!(f, "food_remaining,{:.4}", snapshot.food_remaining).map_err(|e| format!("{}", e))?;
        writeln!(f, "fly_food_total,{:.4}", snapshot.fly_food_total).map_err(|e| format!("{}", e))?;
        writeln!(f, "light,{:.4}", snapshot.light).map_err(|e| format!("{}", e))?;
        writeln!(f, "humidity,{:.4}", snapshot.humidity).map_err(|e| format!("{}", e))?;
        writeln!(f, "temperature,{:.4}", snapshot.temperature).map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_cell_vitality,{:.4}", snapshot.mean_cell_vitality).map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_cell_energy,{:.4}", snapshot.mean_cell_energy).map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_soil_moisture,{:.4}", snapshot.mean_soil_moisture).map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_deep_moisture,{:.4}", snapshot.mean_deep_moisture).map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_microbes,{:.6}", snapshot.mean_microbes).map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_symbionts,{:.6}", snapshot.mean_symbionts).map_err(|e| format!("{}", e))?;
        writeln!(f, "total_plant_cells,{:.0}", snapshot.total_plant_cells).map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_atmospheric_co2,{:.6}", snapshot.mean_atmospheric_co2).map_err(|e| format!("{}", e))?;
        writeln!(f, "mean_atmospheric_o2,{:.4}", snapshot.mean_atmospheric_o2).map_err(|e| format!("{}", e))?;
    }

    // Plants CSV
    {
        let mut f = File::create(format!("{}/plants.csv", dir)).map_err(|e| format!("{}", e))?;
        writeln!(f, "id,x,y,height_mm,vitality,energy_charge,total_cells").map_err(|e| format!("{}", e))?;
        for (i, plant) in world.plants.iter().enumerate() {
            writeln!(f, "{},{},{},{:.2},{:.4},{:.4},{:.0}",
                i, plant.x, plant.y,
                plant.physiology.height_mm(),
                plant.cellular.vitality(),
                plant.cellular.energy_charge(),
                plant.cellular.total_cells(),
            ).map_err(|e| format!("{}", e))?;
        }
    }

    // Flies CSV
    {
        let mut f = File::create(format!("{}/flies.csv", dir)).map_err(|e| format!("{}", e))?;
        writeln!(f, "id,x,y,z,speed,heading,energy,is_flying").map_err(|e| format!("{}", e))?;
        for (i, fly) in world.flies.iter().enumerate() {
            let b = fly.body_state();
            writeln!(f, "{},{:.3},{:.3},{:.3},{:.4},{:.3},{:.4},{}",
                i, b.x, b.y, b.z, b.speed, b.heading, b.energy, b.is_flying,
            ).map_err(|e| format!("{}", e))?;
        }
    }

    // Soil moisture grid CSV
    {
        let moisture = world.moisture_field();
        let gw = world.config.width;
        let gh = world.config.height;
        let mut f = File::create(format!("{}/soil_moisture.csv", dir)).map_err(|e| format!("{}", e))?;
        writeln!(f, "x,y,moisture").map_err(|e| format!("{}", e))?;
        for gy in 0..gh {
            for gx in 0..gw {
                let mi = gy * gw + gx;
                let m = if mi < moisture.len() { moisture[mi] } else { 0.0 };
                writeln!(f, "{},{},{:.4}", gx, gy, m).map_err(|e| format!("{}", e))?;
            }
        }
    }

    // Water entities CSV
    {
        let mut f = File::create(format!("{}/water.csv", dir)).map_err(|e| format!("{}", e))?;
        writeln!(f, "id,x,y,volume,alive").map_err(|e| format!("{}", e))?;
        for (i, water) in world.waters.iter().enumerate() {
            writeln!(f, "{},{},{},{:.4},{}", i, water.x, water.y, water.volume, water.alive).map_err(|e| format!("{}", e))?;
        }
    }

    // Fruits CSV
    {
        let mut f = File::create(format!("{}/fruits.csv", dir)).map_err(|e| format!("{}", e))?;
        writeln!(f, "id,x,y,sugar,ripeness,alive").map_err(|e| format!("{}", e))?;
        for (i, fruit) in world.fruits.iter().enumerate() {
            writeln!(f, "{},{},{},{:.4},{:.4},{}", i, fruit.source.x, fruit.source.y,
                fruit.source.sugar_content, fruit.source.ripeness, fruit.source.alive).map_err(|e| format!("{}", e))?;
        }
    }

    Ok(dir)
}

/// State for time-lapse BMP sequence recording.
pub struct TimeLapse {
    pub recording: bool,
    pub frame_count: usize,
    pub dir: String,
}

impl TimeLapse {
    pub fn new() -> Self {
        Self { recording: false, frame_count: 0, dir: String::new() }
    }

    /// Toggle recording on/off. Returns status message.
    pub fn toggle(&mut self) -> String {
        if self.recording {
            self.recording = false;
            format!("Recording stopped ({} frames in {})", self.frame_count, self.dir)
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
    pub fn save_frame(&mut self, buffer: &[u32], width: usize, height: usize) -> Result<(), String> {
        if !self.recording { return Ok(()); }
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
    header[0] = b'B'; header[1] = b'M';
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
            row_buf[offset]     = (c & 0xff) as u8;
            row_buf[offset + 1] = ((c >> 8) & 0xff) as u8;
            row_buf[offset + 2] = ((c >> 16) & 0xff) as u8;
        }
        for i in (width * 3)..row_size { row_buf[i] = 0; }
        file.write_all(&row_buf).map_err(|e| format!("{}", e))?;
    }
    Ok(())
}
