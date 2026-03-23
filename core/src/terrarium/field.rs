//! Native terrarium sensory lattice for fly/world integration.
//!
//! Python still owns the authoritative `MolecularWorld`, but this type
//! stores a compact snapshot of the local sensory fields so fly sampling
//! no longer has to happen through repeated Python callbacks.

#[derive(Clone, Copy, Debug, PartialEq)]
struct FoodPatch {
    x: f32,
    y: f32,
    radius: f32,
    remaining: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlySensorySample {
    pub odorant: f32,
    pub left_odorant: f32,
    pub right_odorant: f32,
    pub left_light: f32,
    pub right_light: f32,
    pub temperature: f32,
    pub wind_x: f32,
    pub wind_y: f32,
    pub wind_z: f32,
    pub sugar_taste: f32,
    pub bitter_taste: f32,
    pub amino_taste: f32,
    pub food_available: f32,
}

pub struct TerrariumSensoryField {
    width: usize,
    height: usize,
    depth: usize,
    odorant: Vec<f32>,
    temperature: Vec<f32>,
    wind_x: Vec<f32>,
    wind_y: Vec<f32>,
    wind_z: Vec<f32>,
    ambient_light: f32,
    food_patches: Vec<FoodPatch>,
}

impl TerrariumSensoryField {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let depth = depth.max(1);
        let len = width.max(1) * height.max(1) * depth;
        Self {
            width: width.max(1),
            height: height.max(1),
            depth,
            odorant: vec![0.0; len],
            temperature: vec![22.0; len],
            wind_x: vec![0.0; len],
            wind_y: vec![0.0; len],
            wind_z: vec![0.0; len],
            ambient_light: 0.5,
            food_patches: Vec::new(),
        }
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.depth)
    }

    fn expected_len(&self) -> usize {
        self.width * self.height * self.depth
    }

    fn validate_len(&self, name: &str, values: &[f32]) -> Result<(), String> {
        let expected = self.expected_len();
        if values.len() != expected {
            return Err(format!(
                "{} length mismatch: expected {}, got {}",
                name,
                expected,
                values.len()
            ));
        }
        Ok(())
    }

    pub fn load_state(
        &mut self,
        odorant: &[f32],
        temperature: &[f32],
        wind_x: &[f32],
        wind_y: &[f32],
        wind_z: &[f32],
        ambient_light: f32,
        food_patches_flat: &[f32],
    ) -> Result<(), String> {
        self.validate_len("odorant", odorant)?;
        self.validate_len("temperature", temperature)?;
        self.validate_len("wind_x", wind_x)?;
        self.validate_len("wind_y", wind_y)?;
        self.validate_len("wind_z", wind_z)?;
        if food_patches_flat.len() % 4 != 0 {
            return Err(format!(
                "food_patches length must be a multiple of 4, got {}",
                food_patches_flat.len()
            ));
        }

        self.odorant.copy_from_slice(odorant);
        self.temperature.copy_from_slice(temperature);
        self.wind_x.copy_from_slice(wind_x);
        self.wind_y.copy_from_slice(wind_y);
        self.wind_z.copy_from_slice(wind_z);
        self.ambient_light = ambient_light.max(0.0);
        self.food_patches.clear();
        for chunk in food_patches_flat.chunks_exact(4) {
            self.food_patches.push(FoodPatch {
                x: chunk[0],
                y: chunk[1],
                radius: chunk[2].max(0.1),
                remaining: chunk[3].max(0.0),
            });
        }
        Ok(())
    }

    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.height + y) * self.width + x
    }

    fn sample_trilinear(&self, field: &[f32], x: f32, y: f32, z: f32) -> f32 {
        let x = x.clamp(0.0, (self.width - 1) as f32);
        let y = y.clamp(0.0, (self.height - 1) as f32);
        let z = z.clamp(0.0, (self.depth - 1) as f32);

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let z0 = z.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let z1 = (z0 + 1).min(self.depth - 1);
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;
        let tz = z - z0 as f32;

        let c000 = field[self.idx(x0, y0, z0)];
        let c100 = field[self.idx(x1, y0, z0)];
        let c010 = field[self.idx(x0, y1, z0)];
        let c110 = field[self.idx(x1, y1, z0)];
        let c001 = field[self.idx(x0, y0, z1)];
        let c101 = field[self.idx(x1, y0, z1)];
        let c011 = field[self.idx(x0, y1, z1)];
        let c111 = field[self.idx(x1, y1, z1)];

        let c00 = c000 + (c100 - c000) * tx;
        let c10 = c010 + (c110 - c010) * tx;
        let c01 = c001 + (c101 - c001) * tx;
        let c11 = c011 + (c111 - c011) * tx;
        let c0 = c00 + (c10 - c00) * ty;
        let c1 = c01 + (c11 - c01) * ty;
        c0 + (c1 - c0) * tz
    }

    fn sample_food_signal(&self, x: f32, y: f32) -> f32 {
        let mut total = 0.0f32;
        for patch in &self.food_patches {
            let dx = x - patch.x;
            let dy = y - patch.y;
            let dist_sq = dx * dx + dy * dy;
            let radius = patch.radius.max(0.1);
            if dist_sq.sqrt() < radius * 2.0 {
                let sigma = radius * 0.5;
                total += patch.remaining * (-dist_sq / (2.0 * sigma * sigma)).exp();
            }
        }
        total.clamp(0.0, 1.0)
    }

    pub fn consume_food_near(&mut self, x: f32, y: f32, eat_radius: f32, amount: f32) -> bool {
        let mut eaten = false;
        for patch in &mut self.food_patches {
            let dx = x - patch.x;
            let dy = y - patch.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < eat_radius + patch.radius && patch.remaining > 0.0 {
                patch.remaining = (patch.remaining - amount.max(0.0)).max(0.0);
                eaten = true;
            }
        }
        eaten
    }

    pub fn food_patches_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.food_patches.len() * 4);
        for patch in &self.food_patches {
            flat.push(patch.x);
            flat.push(patch.y);
            flat.push(patch.radius);
            flat.push(patch.remaining);
        }
        flat
    }

    pub fn sample_fly(
        &self,
        x: f32,
        y: f32,
        z: f32,
        heading: f32,
        is_flying: bool,
    ) -> FlySensorySample {
        let ax = x + heading.cos() * 0.5;
        let ay = y + heading.sin() * 0.5;
        let az = z + 0.3;
        let antenna_forward = if is_flying { 0.7 } else { 0.45 };
        let antenna_lateral = 0.42;
        let sensor_x = x + heading.cos() * antenna_forward;
        let sensor_y = y + heading.sin() * antenna_forward;
        let left_x = sensor_x + heading.sin() * antenna_lateral;
        let left_y = sensor_y - heading.cos() * antenna_lateral;
        let right_x = sensor_x - heading.sin() * antenna_lateral;
        let right_y = sensor_y + heading.cos() * antenna_lateral;
        let left_odorant = self
            .sample_trilinear(&self.odorant, left_x, left_y, az)
            .clamp(0.0, 1.0);
        let right_odorant = self
            .sample_trilinear(&self.odorant, right_x, right_y, az)
            .clamp(0.0, 1.0);
        let odorant = self
            .sample_trilinear(&self.odorant, ax, ay, az)
            .max((left_odorant + right_odorant) * 0.5)
            .clamp(0.0, 1.0);
        let temperature = self.sample_trilinear(&self.temperature, x, y, z);
        let wind_x = self.sample_trilinear(&self.wind_x, x, y, z);
        let wind_y = self.sample_trilinear(&self.wind_y, x, y, z);
        let wind_z = self.sample_trilinear(&self.wind_z, x, y, z);
        let food_available = if is_flying {
            0.0
        } else {
            self.sample_food_signal(x, y)
        };

        FlySensorySample {
            odorant,
            left_odorant,
            right_odorant,
            left_light: self.ambient_light,
            right_light: self.ambient_light,
            temperature,
            wind_x,
            wind_y,
            wind_z,
            sugar_taste: food_available,
            bitter_taste: 0.0,
            amino_taste: 0.0,
            food_available,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TerrariumSensoryField;

    #[test]
    fn samples_food_patch_and_fields() {
        let mut field = TerrariumSensoryField::new(8, 6, 1);
        let len = 8 * 6;
        let mut odor = vec![0.0; len];
        odor[3 * 8 + 4] = 0.8;
        let temp = vec![21.5; len];
        let wind_x = vec![0.2; len];
        let wind_y = vec![0.1; len];
        let wind_z = vec![0.0; len];
        field
            .load_state(
                &odor,
                &temp,
                &wind_x,
                &wind_y,
                &wind_z,
                0.7,
                &[4.0, 3.0, 2.0, 1.0],
            )
            .unwrap();
        let sample = field.sample_fly(4.0, 3.0, 0.0, 0.0, false);
        assert!(sample.odorant >= 0.0);
        assert!(sample.food_available > 0.2);
        assert_eq!(sample.left_light, 0.7);
        assert_eq!(sample.right_light, 0.7);
    }

    #[test]
    fn food_patch_does_not_fake_odorant_signal() {
        let mut field = TerrariumSensoryField::new(8, 6, 1);
        let len = 8 * 6;
        let zero = vec![0.0; len];
        field
            .load_state(
                &zero,
                &zero,
                &zero,
                &zero,
                &zero,
                0.5,
                &[4.0, 3.0, 2.0, 1.0],
            )
            .unwrap();
        let sample = field.sample_fly(4.0, 3.0, 0.0, 0.0, false);
        assert_eq!(sample.odorant, 0.0);
        assert!(sample.food_available > 0.2);
    }

    #[test]
    fn consumes_food_patch_in_place() {
        let mut field = TerrariumSensoryField::new(8, 6, 1);
        let len = 8 * 6;
        let zero = vec![0.0; len];
        field
            .load_state(
                &zero,
                &zero,
                &zero,
                &zero,
                &zero,
                0.5,
                &[4.0, 3.0, 2.0, 1.0],
            )
            .unwrap();
        assert!(field.consume_food_near(4.0, 3.0, 2.0, 0.25));
        let patches = field.food_patches_flat();
        assert_eq!(patches.len(), 4);
        assert!(patches[3] < 1.0);
    }
}
