//! Pure complex-channel assembly helpers extracted from `whole_cell.rs`.

pub(crate) fn complex_channel_step(
    current: f32,
    target: f32,
    assembly_support: f32,
    degradation_pressure: f32,
    dt_scale: f32,
    max_value: f32,
) -> (f32, f32, f32) {
    let current = current.max(0.0);
    let target = target.max(0.0);
    let assembly_rate =
        (target - current).max(0.0) * (0.06 + 0.10 * assembly_support.clamp(0.55, 1.65));
    let degradation_rate = current * (0.005 + 0.018 * degradation_pressure.clamp(0.65, 1.80));
    let next = (current + dt_scale * (assembly_rate - degradation_rate)).clamp(0.0, max_value);
    (next, assembly_rate.max(0.0), degradation_rate.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_channel_step_grows_toward_target() {
        let (next, assembly_rate, degradation_rate) =
            complex_channel_step(10.0, 20.0, 1.0, 1.0, 1.0, 100.0);
        assert!(next > 10.0);
        assert!(assembly_rate > degradation_rate);
    }

    #[test]
    fn complex_channel_step_respects_maximum() {
        let (next, _, _) = complex_channel_step(90.0, 200.0, 1.65, 0.65, 10.0, 95.0);
        assert!((next - 95.0).abs() < 1.0e-6);
    }
}
