fn idx2(width: usize, x: usize, y: usize) -> usize {
    y * width + x
}

fn idx3(width: usize, height: usize, x: usize, y: usize, z: usize) -> usize {
    (z * height + y) * width + x
}

pub fn deposit_2d(
    field: &mut [f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    radius: usize,
    amount: f32,
) {
    if amount.abs() <= 1.0e-12 {
        return;
    }
    let radius = radius.max(1);
    let sigma = (radius as f32 * 0.72).max(0.75);
    let denom = 2.0 * sigma * sigma;
    let y0 = y.saturating_sub(radius);
    let y1 = (y + radius + 1).min(height);
    let x0 = x.saturating_sub(radius);
    let x1 = (x + radius + 1).min(width);
    let mut kernel_total = 0.0f32;

    for yy in y0..y1 {
        for xx in x0..x1 {
            let dx = xx as f32 - x as f32;
            let dy = yy as f32 - y as f32;
            kernel_total += (-(dx * dx + dy * dy) / denom).exp();
        }
    }
    if kernel_total <= 1.0e-9 {
        return;
    }
    let scale = amount / kernel_total;
    for yy in y0..y1 {
        for xx in x0..x1 {
            let dx = xx as f32 - x as f32;
            let dy = yy as f32 - y as f32;
            let kernel = (-(dx * dx + dy * dy) / denom).exp();
            field[idx2(width, xx, yy)] += kernel * scale;
        }
    }
}

pub fn deposit_2d_background_only(
    field: &mut [f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    radius: usize,
    amount: f32,
    owned_mask: &[f32],
    ownership_threshold: f32,
) {
    if amount.abs() <= 1.0e-12 || field.len() != owned_mask.len() {
        return;
    }
    let radius = radius.max(1);
    let sigma = (radius as f32 * 0.72).max(0.75);
    let denom = 2.0 * sigma * sigma;
    let y0 = y.saturating_sub(radius);
    let y1 = (y + radius + 1).min(height);
    let x0 = x.saturating_sub(radius);
    let x1 = (x + radius + 1).min(width);
    let mut kernel_total = 0.0f32;

    for yy in y0..y1 {
        for xx in x0..x1 {
            let flat = idx2(width, xx, yy);
            if owned_mask[flat] >= ownership_threshold {
                continue;
            }
            let dx = xx as f32 - x as f32;
            let dy = yy as f32 - y as f32;
            kernel_total += (-(dx * dx + dy * dy) / denom).exp();
        }
    }
    if kernel_total <= 1.0e-9 {
        return;
    }
    let scale = amount / kernel_total;
    for yy in y0..y1 {
        for xx in x0..x1 {
            let flat = idx2(width, xx, yy);
            if owned_mask[flat] >= ownership_threshold {
                continue;
            }
            let dx = xx as f32 - x as f32;
            let dy = yy as f32 - y as f32;
            let kernel = (-(dx * dx + dy * dy) / denom).exp();
            field[flat] += kernel * scale;
        }
    }
}

pub fn exchange_layer_patch(
    field: &mut [f32],
    width: usize,
    height: usize,
    depth: usize,
    x: usize,
    y: usize,
    z: usize,
    radius: usize,
    amount: f32,
    min_value: f32,
    max_value: f32,
) {
    let _ = exchange_layer_patch_actual(
        field, width, height, depth, x, y, z, radius, amount, min_value, max_value,
    );
}

pub fn exchange_layer_patch_actual(
    field: &mut [f32],
    width: usize,
    height: usize,
    depth: usize,
    x: usize,
    y: usize,
    z: usize,
    radius: usize,
    amount: f32,
    min_value: f32,
    max_value: f32,
) -> f32 {
    if amount.abs() <= 1.0e-12 {
        return 0.0;
    }
    let radius = radius.max(1);
    let z = z.min(depth.saturating_sub(1));
    let sigma = (radius as f32 * 0.72).max(0.75);
    let denom = 2.0 * sigma * sigma;
    let y0 = y.saturating_sub(radius);
    let y1 = (y + radius + 1).min(height);
    let x0 = x.saturating_sub(radius);
    let x1 = (x + radius + 1).min(width);
    let mut kernel_total = 0.0f32;

    for yy in y0..y1 {
        for xx in x0..x1 {
            let dx = xx as f32 - x as f32;
            let dy = yy as f32 - y as f32;
            kernel_total += (-(dx * dx + dy * dy) / denom).exp();
        }
    }
    if kernel_total <= 1.0e-9 {
        return 0.0;
    }
    let scale = amount / kernel_total;
    let mut actual = 0.0f32;
    for yy in y0..y1 {
        for xx in x0..x1 {
            let dx = xx as f32 - x as f32;
            let dy = yy as f32 - y as f32;
            let kernel = (-(dx * dx + dy * dy) / denom).exp();
            let cell = &mut field[idx3(width, height, xx, yy, z)];
            let before = *cell;
            *cell = (*cell + kernel * scale).clamp(min_value, max_value);
            actual += *cell - before;
        }
    }
    actual
}

pub fn layer_mean_map(
    width: usize,
    height: usize,
    depth: usize,
    field: &[f32],
    z0: usize,
    z1: usize,
) -> Vec<f32> {
    let plane = width * height;
    let total_z = depth.max(1);
    let start = z0.min(total_z - 1);
    let end = z1.max(start + 1).min(total_z);
    let count = (end - start) as f32;
    let mut out = vec![0.0f32; plane];
    for z in start..end {
        let slice = &field[z * plane..(z + 1) * plane];
        for (dst, src) in out.iter_mut().zip(slice.iter().copied()) {
            *dst += src;
        }
    }
    for value in &mut out {
        *value /= count;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        deposit_2d, deposit_2d_background_only, exchange_layer_patch, exchange_layer_patch_actual,
        layer_mean_map,
    };

    #[test]
    fn deposit_2d_distributes_requested_amount() {
        let mut field = vec![0.0f32; 25];
        deposit_2d(&mut field, 5, 5, 2, 2, 1, 1.25);
        let total = field.iter().sum::<f32>();
        assert!(
            (total - 1.25).abs() < 1.0e-5,
            "total deposition was {total}"
        );
    }

    #[test]
    fn background_only_deposition_skips_owned_cells() {
        let mut field = vec![0.0f32; 9];
        let mut owned_mask = vec![0.0f32; 9];
        owned_mask[4] = 0.8;
        deposit_2d_background_only(&mut field, 3, 3, 1, 1, 1, 1.0, &owned_mask, 0.5);
        assert_eq!(field[4], 0.0);
        assert!(field.iter().sum::<f32>() > 0.0);
    }

    #[test]
    fn exchange_layer_patch_clamps_values() {
        let mut field = vec![0.0f32; 3 * 3 * 2];
        exchange_layer_patch(&mut field, 3, 3, 2, 1, 1, 1, 1, 5.0, 0.0, 1.0);
        assert!(field.iter().all(|value| *value <= 1.0));
        assert!(field.iter().any(|value| *value > 0.0));
    }

    #[test]
    fn exchange_layer_patch_actual_returns_net_delta() {
        let mut field = vec![0.0f32; 3 * 3 * 2];
        let actual = exchange_layer_patch_actual(&mut field, 3, 3, 2, 1, 1, 1, 1, 0.5, 0.0, 1.0);
        let total = field.iter().sum::<f32>();
        assert!((actual - total).abs() <= 1.0e-6);
        assert!((total - 0.5).abs() <= 1.0e-5);
    }

    #[test]
    fn layer_mean_map_averages_requested_span() {
        let field = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ];
        let reduced = layer_mean_map(2, 2, 3, &field, 1, 3);
        assert_eq!(reduced, vec![7.0, 8.0, 9.0, 10.0]);
    }
}
