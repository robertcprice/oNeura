//! Convergence visualization dashboard for Terrarium Evolution telemetry.
//!
//! Reads GenerationTelemetry JSON and produces SVG plots:
//!   1. Convergence curve (best/mean/worst fitness)
//!   2. Diversity over generations
//!   3. Parameter heatmap (normalized genome params)
//!   4. Multi-objective radar (spider chart, Pareto modes)
//!   5. Stress resilience bar chart (stress modes)
//!
//! Usage:
//!   evolve_dashboard --telemetry telemetry.json --output figures/
//!
//! All output is SVG -- zero external dependencies beyond serde_json.

use oneuro_metal::GenerationTelemetry;
use std::env;
use std::fs;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct DashArgs {
    telemetry: PathBuf,
    output: PathBuf,
    title: String,
}

fn parse_dash_args() -> DashArgs {
    let mut telemetry = PathBuf::from("telemetry.json");
    let mut output = PathBuf::from("figures");
    let mut title = String::from("Terrarium Evolution");
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--telemetry" => {
                if i + 1 < args.len() {
                    telemetry = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--title" => {
                if i + 1 < args.len() {
                    title = args[i + 1].clone();
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("Terrarium Evolution Dashboard (SVG output)");
                println!();
                println!("Usage: evolve_dashboard [options]");
                println!();
                println!("Options:");
                println!("  --telemetry <PATH>  Input telemetry JSON file");
                println!("  --output <DIR>      Output directory for SVG files (default: figures/)");
                println!("  --title <STR>       Title prefix for plots");
                println!("  --help, -h          Show this help");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    DashArgs {
        telemetry,
        output,
        title,
    }
}

// ---------------------------------------------------------------------------
// SVG helpers
// ---------------------------------------------------------------------------

const W: f32 = 800.0;
const H: f32 = 500.0;
const MARGIN_L: f32 = 70.0;
const MARGIN_R: f32 = 30.0;
const MARGIN_T: f32 = 50.0;
const MARGIN_B: f32 = 60.0;

fn plot_w() -> f32 {
    W - MARGIN_L - MARGIN_R
}
fn plot_h() -> f32 {
    H - MARGIN_T - MARGIN_B
}

fn svg_header(width: f32, height: f32) -> String {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        width, height, width, height
    )
}

fn svg_footer() -> &'static str {
    "</svg>"
}

fn svg_text(x: f32, y: f32, text: &str, anchor: &str, font_size: f32, bold: bool) -> String {
    let weight = if bold { "bold" } else { "normal" };
    format!(
        r#"<text x="{:.1}" y="{:.1}" text-anchor="{}" font-family="monospace" font-size="{:.0}" font-weight="{}">{}</text>"#,
        x, y, anchor, font_size, weight, text
    )
}

fn svg_line(x1: f32, y1: f32, x2: f32, y2: f32, color: &str, width: f32) -> String {
    format!(
        r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="{:.1}"/>"#,
        x1, y1, x2, y2, color, width
    )
}

fn svg_rect(x: f32, y: f32, w: f32, h: f32, fill: &str, opacity: f32) -> String {
    format!(
        r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" opacity="{:.2}"/>"#,
        x, y, w, h, fill, opacity
    )
}

fn svg_polyline(points: &[(f32, f32)], color: &str, width: f32) -> String {
    let pts: String = points
        .iter()
        .map(|(x, y)| format!("{:.1},{:.1}", x, y))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        r#"<polyline points="{}" fill="none" stroke="{}" stroke-width="{:.1}" stroke-linejoin="round"/>"#,
        pts, color, width
    )
}

fn svg_circle(cx: f32, cy: f32, r: f32, fill: &str) -> String {
    format!(
        r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="{}"/>"#,
        cx, cy, r, fill
    )
}

fn svg_polygon(points: &[(f32, f32)], fill: &str, stroke: &str, opacity: f32) -> String {
    let pts: String = points
        .iter()
        .map(|(x, y)| format!("{:.1},{:.1}", x, y))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        r#"<polygon points="{}" fill="{}" fill-opacity="{:.2}" stroke="{}" stroke-width="2"/>"#,
        pts, fill, opacity, stroke
    )
}

/// Map a value from [lo, hi] to pixel X in the plot area.
fn map_x(val: f32, lo: f32, hi: f32) -> f32 {
    if (hi - lo).abs() < 1e-10 {
        return MARGIN_L + plot_w() / 2.0;
    }
    MARGIN_L + (val - lo) / (hi - lo) * plot_w()
}

/// Map a value from [lo, hi] to pixel Y (inverted) in the plot area.
fn map_y(val: f32, lo: f32, hi: f32) -> f32 {
    if (hi - lo).abs() < 1e-10 {
        return MARGIN_T + plot_h() / 2.0;
    }
    MARGIN_T + plot_h() - (val - lo) / (hi - lo) * plot_h()
}

fn axes(title: &str, x_label: &str, y_label: &str) -> String {
    let mut s = String::new();
    // Background
    s.push_str(&svg_rect(0.0, 0.0, W, H, "#fafafa", 1.0));
    // Plot area
    s.push_str(&svg_rect(
        MARGIN_L,
        MARGIN_T,
        plot_w(),
        plot_h(),
        "white",
        1.0,
    ));
    // Border
    s.push_str(&svg_line(
        MARGIN_L,
        MARGIN_T,
        MARGIN_L + plot_w(),
        MARGIN_T,
        "#ccc",
        1.0,
    ));
    s.push_str(&svg_line(
        MARGIN_L,
        MARGIN_T + plot_h(),
        MARGIN_L + plot_w(),
        MARGIN_T + plot_h(),
        "#333",
        1.5,
    ));
    s.push_str(&svg_line(
        MARGIN_L,
        MARGIN_T,
        MARGIN_L,
        MARGIN_T + plot_h(),
        "#333",
        1.5,
    ));
    // Title
    s.push_str(&svg_text(W / 2.0, 30.0, title, "middle", 16.0, true));
    // X label
    s.push_str(&svg_text(
        MARGIN_L + plot_w() / 2.0,
        H - 10.0,
        x_label,
        "middle",
        13.0,
        false,
    ));
    // Y label (rotated)
    s.push_str(&format!(
        r#"<text x="15" y="{:.1}" text-anchor="middle" font-family="monospace" font-size="13" transform="rotate(-90, 15, {:.1})">{}</text>"#,
        MARGIN_T + plot_h() / 2.0,
        MARGIN_T + plot_h() / 2.0,
        y_label
    ));
    s
}

fn tick_labels_x(lo: f32, hi: f32, count: usize) -> String {
    let mut s = String::new();
    for i in 0..=count {
        let val = lo + (hi - lo) * i as f32 / count as f32;
        let x = map_x(val, lo, hi);
        let y = MARGIN_T + plot_h() + 18.0;
        s.push_str(&svg_text(x, y, &format!("{:.0}", val), "middle", 11.0, false));
        s.push_str(&svg_line(x, MARGIN_T + plot_h(), x, MARGIN_T + plot_h() + 4.0, "#333", 1.0));
    }
    s
}

fn tick_labels_y(lo: f32, hi: f32, count: usize) -> String {
    let mut s = String::new();
    for i in 0..=count {
        let val = lo + (hi - lo) * i as f32 / count as f32;
        let y = map_y(val, lo, hi);
        s.push_str(&svg_text(
            MARGIN_L - 8.0,
            y + 4.0,
            &format!("{:.1}", val),
            "end",
            11.0,
            false,
        ));
        // Grid line
        s.push_str(&svg_line(MARGIN_L, y, MARGIN_L + plot_w(), y, "#eee", 0.5));
    }
    s
}

fn legend(entries: &[(&str, &str)], x: f32, y: f32) -> String {
    let mut s = String::new();
    for (i, (label, color)) in entries.iter().enumerate() {
        let yy = y + i as f32 * 20.0;
        s.push_str(&svg_line(x, yy, x + 20.0, yy, color, 2.5));
        s.push_str(&svg_text(x + 25.0, yy + 4.0, label, "start", 11.0, false));
    }
    s
}

// ---------------------------------------------------------------------------
// Plot generators
// ---------------------------------------------------------------------------

fn plot_convergence(records: &[GenerationTelemetry], title_prefix: &str) -> String {
    if records.is_empty() {
        return String::new();
    }

    let gens: Vec<f32> = records.iter().map(|r| r.generation as f32).collect();
    let best: Vec<f32> = records.iter().map(|r| r.best_fitness).collect();
    let mean: Vec<f32> = records.iter().map(|r| r.mean_fitness).collect();
    let worst: Vec<f32> = records.iter().map(|r| r.worst_fitness).collect();

    let x_lo = *gens.first().unwrap();
    let x_hi = *gens.last().unwrap();
    let all_vals: Vec<f32> = best.iter().chain(mean.iter()).chain(worst.iter()).copied().collect();
    let y_lo = all_vals.iter().cloned().fold(f32::INFINITY, f32::min) * 0.9;
    let y_hi = all_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 1.1;

    let mut svg = svg_header(W, H);
    svg.push_str(&axes(
        &format!("{} - Convergence", title_prefix),
        "Generation",
        "Fitness",
    ));
    svg.push_str(&tick_labels_x(x_lo, x_hi, 5.min(records.len() - 1)));
    svg.push_str(&tick_labels_y(y_lo, y_hi, 5));

    // Best line
    let best_pts: Vec<(f32, f32)> = gens
        .iter()
        .zip(best.iter())
        .map(|(&g, &v)| (map_x(g, x_lo, x_hi), map_y(v, y_lo, y_hi)))
        .collect();
    svg.push_str(&svg_polyline(&best_pts, "#2196F3", 2.5));

    // Mean line
    let mean_pts: Vec<(f32, f32)> = gens
        .iter()
        .zip(mean.iter())
        .map(|(&g, &v)| (map_x(g, x_lo, x_hi), map_y(v, y_lo, y_hi)))
        .collect();
    svg.push_str(&svg_polyline(&mean_pts, "#4CAF50", 2.0));

    // Worst line
    let worst_pts: Vec<(f32, f32)> = gens
        .iter()
        .zip(worst.iter())
        .map(|(&g, &v)| (map_x(g, x_lo, x_hi), map_y(v, y_lo, y_hi)))
        .collect();
    svg.push_str(&svg_polyline(&worst_pts, "#FF5722", 1.5));

    // Data points on best
    for &(px, py) in &best_pts {
        svg.push_str(&svg_circle(px, py, 3.0, "#2196F3"));
    }

    svg.push_str(&legend(
        &[("Best", "#2196F3"), ("Mean", "#4CAF50"), ("Worst", "#FF5722")],
        MARGIN_L + plot_w() - 120.0,
        MARGIN_T + 20.0,
    ));

    svg.push_str(svg_footer());
    svg
}

fn plot_diversity(records: &[GenerationTelemetry], title_prefix: &str) -> String {
    if records.is_empty() {
        return String::new();
    }

    let gens: Vec<f32> = records.iter().map(|r| r.generation as f32).collect();
    let diversity: Vec<f32> = records.iter().map(|r| r.population_diversity).collect();

    let x_lo = *gens.first().unwrap();
    let x_hi = *gens.last().unwrap();
    let y_lo = 0.0_f32;
    let y_hi = diversity.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 1.2;
    let y_hi = if y_hi <= 0.0 { 1.0 } else { y_hi };

    let mut svg = svg_header(W, H);
    svg.push_str(&axes(
        &format!("{} - Population Diversity", title_prefix),
        "Generation",
        "Diversity",
    ));
    svg.push_str(&tick_labels_x(x_lo, x_hi, 5.min(records.len() - 1)));
    svg.push_str(&tick_labels_y(y_lo, y_hi, 5));

    let pts: Vec<(f32, f32)> = gens
        .iter()
        .zip(diversity.iter())
        .map(|(&g, &v)| (map_x(g, x_lo, x_hi), map_y(v, y_lo, y_hi)))
        .collect();
    svg.push_str(&svg_polyline(&pts, "#9C27B0", 2.5));

    // Fill area under curve
    if pts.len() >= 2 {
        let mut fill_pts = pts.clone();
        fill_pts.push((pts.last().unwrap().0, map_y(0.0, y_lo, y_hi)));
        fill_pts.push((pts.first().unwrap().0, map_y(0.0, y_lo, y_hi)));
        svg.push_str(&svg_polygon(&fill_pts, "#9C27B0", "#9C27B0", 0.15));
    }

    for &(px, py) in &pts {
        svg.push_str(&svg_circle(px, py, 3.0, "#9C27B0"));
    }

    svg.push_str(svg_footer());
    svg
}

fn plot_param_heatmap(records: &[GenerationTelemetry], title_prefix: &str) -> String {
    if records.is_empty() {
        return String::new();
    }

    let param_names = [
        "proton_scale",
        "temperature",
        "water_count",
        "water_vol",
        "moisture",
        "plants",
        "fruits",
        "flies",
        "microbes",
        "resp_vmax",
        "nitr_vmax",
        "photo_vmax",
        "miner_vmax",
        "seed",
        "time_warp",
    ];

    let n_gens = records.len();
    let n_params = param_names.len();
    let cell_w = plot_w() / n_gens as f32;
    let cell_h = plot_h() / n_params as f32;

    let mut svg = svg_header(W, H + 20.0);
    svg.push_str(&svg_rect(0.0, 0.0, W, H + 20.0, "#fafafa", 1.0));
    svg.push_str(&svg_text(
        W / 2.0,
        30.0,
        &format!("{} - Parameter Heatmap", title_prefix),
        "middle",
        16.0,
        true,
    ));

    for (gi, rec) in records.iter().enumerate() {
        let params = &rec.best_genome_params;
        for pi in 0..n_params.min(params.len()) {
            let val = params[pi].clamp(0.0, 1.0);
            let x = MARGIN_L + gi as f32 * cell_w;
            let y = MARGIN_T + pi as f32 * cell_h;

            // Blue (0) -> Red (1) colormap
            let r = (val * 255.0) as u8;
            let b = ((1.0 - val) * 255.0) as u8;
            let g_color = ((1.0 - (2.0 * val - 1.0).abs()) * 128.0) as u8;
            let color = format!("rgb({},{},{})", r, g_color, b);
            svg.push_str(&svg_rect(x, y, cell_w + 0.5, cell_h + 0.5, &color, 1.0));
        }
    }

    // Y labels (param names)
    for (pi, name) in param_names.iter().enumerate() {
        let y = MARGIN_T + pi as f32 * cell_h + cell_h / 2.0 + 4.0;
        svg.push_str(&svg_text(MARGIN_L - 5.0, y, name, "end", 9.0, false));
    }

    // X labels (generation numbers)
    let step = (n_gens / 5).max(1);
    for gi in (0..n_gens).step_by(step) {
        let x = MARGIN_L + gi as f32 * cell_w + cell_w / 2.0;
        svg.push_str(&svg_text(
            x,
            MARGIN_T + plot_h() + 15.0,
            &format!("{}", records[gi].generation),
            "middle",
            10.0,
            false,
        ));
    }

    svg.push_str(svg_footer());
    svg
}

fn plot_multi_objective_radar(records: &[GenerationTelemetry], title_prefix: &str) -> String {
    // Find records with multi_objective_fitness
    let mo_records: Vec<&GenerationTelemetry> = records
        .iter()
        .filter(|r| r.multi_objective_fitness.is_some())
        .collect();

    if mo_records.is_empty() {
        return String::new();
    }

    // Use the last record (final generation)
    let rec = mo_records.last().unwrap();
    let mo = rec.multi_objective_fitness.as_ref().unwrap();

    let objectives = [
        ("Biomass", mo.biomass),
        ("Biodiversity", mo.biodiversity),
        ("Stability", mo.stability),
        ("Carbon", mo.carbon),
        ("Fruit", mo.fruit),
        ("Microbial", mo.microbial),
    ];

    let n = objectives.len();
    let cx = W / 2.0;
    let cy = H / 2.0 + 10.0;
    let radius = 180.0;

    // Find max value for normalization
    let max_val = objectives
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1.0);

    let mut svg = svg_header(W, H);
    svg.push_str(&svg_rect(0.0, 0.0, W, H, "#fafafa", 1.0));
    svg.push_str(&svg_text(
        cx,
        30.0,
        &format!("{} - Multi-Objective Radar", title_prefix),
        "middle",
        16.0,
        true,
    ));

    // Draw grid rings
    for ring in 1..=4 {
        let r = radius * ring as f32 / 4.0;
        let mut ring_pts = Vec::new();
        for i in 0..n {
            let angle = std::f32::consts::PI * 2.0 * i as f32 / n as f32 - std::f32::consts::FRAC_PI_2;
            ring_pts.push((cx + r * angle.cos(), cy + r * angle.sin()));
        }
        ring_pts.push(ring_pts[0]);
        svg.push_str(&svg_polyline(&ring_pts, "#ddd", 0.5));
    }

    // Draw axis lines and labels
    for (i, (label, _)) in objectives.iter().enumerate() {
        let angle = std::f32::consts::PI * 2.0 * i as f32 / n as f32 - std::f32::consts::FRAC_PI_2;
        let x_end = cx + radius * angle.cos();
        let y_end = cy + radius * angle.sin();
        svg.push_str(&svg_line(cx, cy, x_end, y_end, "#bbb", 1.0));

        let label_r = radius + 20.0;
        let lx = cx + label_r * angle.cos();
        let ly = cy + label_r * angle.sin() + 4.0;
        svg.push_str(&svg_text(lx, ly, label, "middle", 11.0, false));
    }

    // Draw data polygon
    let mut data_pts = Vec::new();
    for (i, (_, val)) in objectives.iter().enumerate() {
        let angle = std::f32::consts::PI * 2.0 * i as f32 / n as f32 - std::f32::consts::FRAC_PI_2;
        let r = radius * (*val / max_val).clamp(0.0, 1.0);
        data_pts.push((cx + r * angle.cos(), cy + r * angle.sin()));
    }
    svg.push_str(&svg_polygon(&data_pts, "#2196F3", "#1565C0", 0.3));

    // Data points
    for &(px, py) in &data_pts {
        svg.push_str(&svg_circle(px, py, 4.0, "#1565C0"));
    }

    // Value labels
    for (i, (_, val)) in objectives.iter().enumerate() {
        let angle = std::f32::consts::PI * 2.0 * i as f32 / n as f32 - std::f32::consts::FRAC_PI_2;
        let r = radius * (*val / max_val).clamp(0.0, 1.0) + 12.0;
        let vx = cx + r * angle.cos();
        let vy = cy + r * angle.sin() + 4.0;
        svg.push_str(&svg_text(vx, vy, &format!("{:.1}", val), "middle", 9.0, true));
    }

    svg.push_str(svg_footer());
    svg
}

fn plot_stress_resilience(records: &[GenerationTelemetry], title_prefix: &str) -> String {
    let stress_records: Vec<&GenerationTelemetry> = records
        .iter()
        .filter(|r| r.stress_metrics.is_some())
        .collect();

    if stress_records.is_empty() {
        return String::new();
    }

    // Use last record
    let rec = stress_records.last().unwrap();
    let sm = rec.stress_metrics.as_ref().unwrap();

    let bars = [
        ("Pre-stress", sm.pre_stress_biomass, "#4CAF50"),
        ("Min-stress", sm.min_stress_biomass, "#FF5722"),
        ("Post-recovery", sm.post_recovery_biomass, "#2196F3"),
    ];

    let max_val = bars
        .iter()
        .map(|(_, v, _)| *v)
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1.0);

    let mut svg = svg_header(W, H);
    svg.push_str(&axes(
        &format!("{} - Stress Resilience", title_prefix),
        "",
        "Biomass Fitness",
    ));
    svg.push_str(&tick_labels_y(0.0, max_val * 1.1, 5));

    let bar_count = bars.len();
    let total_bar_width = plot_w() * 0.6;
    let bar_w = total_bar_width / bar_count as f32;
    let bar_gap = (plot_w() - total_bar_width) / 2.0;

    for (i, (label, val, color)) in bars.iter().enumerate() {
        let x = MARGIN_L + bar_gap + i as f32 * bar_w + bar_w * 0.1;
        let bar_height = (*val / (max_val * 1.1)) * plot_h();
        let y = MARGIN_T + plot_h() - bar_height;
        svg.push_str(&svg_rect(x, y, bar_w * 0.8, bar_height, color, 0.85));

        // Value label
        svg.push_str(&svg_text(
            x + bar_w * 0.4,
            y - 8.0,
            &format!("{:.1}", val),
            "middle",
            11.0,
            true,
        ));

        // Bar label
        svg.push_str(&svg_text(
            x + bar_w * 0.4,
            MARGIN_T + plot_h() + 18.0,
            label,
            "middle",
            11.0,
            false,
        ));
    }

    // Recovery ratio annotation
    if sm.pre_stress_biomass > 0.0 {
        let ratio = sm.post_recovery_biomass / sm.pre_stress_biomass;
        svg.push_str(&svg_text(
            MARGIN_L + plot_w() / 2.0,
            MARGIN_T + plot_h() + 45.0,
            &format!("Recovery ratio: {:.1}%", ratio * 100.0),
            "middle",
            12.0,
            true,
        ));
    }

    svg.push_str(svg_footer());
    svg
}

// ---------------------------------------------------------------------------
// Summary report
// ---------------------------------------------------------------------------

fn generate_summary(records: &[GenerationTelemetry], title_prefix: &str) -> String {
    let mut report = String::new();
    report.push_str(&format!("# {} - Evolution Summary Report\n\n", title_prefix));

    if records.is_empty() {
        report.push_str("No telemetry records found.\n");
        return report;
    }

    let mode = records[0]
        .mode
        .as_deref()
        .unwrap_or("unknown");
    report.push_str(&format!("**Mode**: {}\n\n", mode));
    report.push_str(&format!("**Generations**: {}\n\n", records.len()));

    if let Some(first) = records.first() {
        report.push_str(&format!(
            "**Initial fitness**: best={:.4}, mean={:.4}\n\n",
            first.best_fitness, first.mean_fitness
        ));
    }
    if let Some(last) = records.last() {
        report.push_str(&format!(
            "**Final fitness**: best={:.4}, mean={:.4}\n\n",
            last.best_fitness, last.mean_fitness
        ));
    }

    let total_time: f32 = records.iter().map(|r| r.elapsed_ms).sum();
    report.push_str(&format!(
        "**Total wall time**: {:.2}s\n\n",
        total_time / 1000.0
    ));

    // Improvement
    if records.len() >= 2 {
        let first_best = records.first().unwrap().best_fitness;
        let last_best = records.last().unwrap().best_fitness;
        let improvement = if first_best.abs() > 1e-10 {
            (last_best - first_best) / first_best.abs() * 100.0
        } else {
            0.0
        };
        report.push_str(&format!(
            "**Fitness improvement**: {:.1}%\n\n",
            improvement
        ));
    }

    // Multi-objective breakdown
    if let Some(last) = records.last() {
        if let Some(ref mo) = last.multi_objective_fitness {
            report.push_str("## Multi-Objective Breakdown\n\n");
            report.push_str(&format!("| Objective | Value |\n"));
            report.push_str(&format!("|-----------|-------|\n"));
            report.push_str(&format!("| Biomass | {:.4} |\n", mo.biomass));
            report.push_str(&format!("| Biodiversity | {:.4} |\n", mo.biodiversity));
            report.push_str(&format!("| Stability | {:.4} |\n", mo.stability));
            report.push_str(&format!("| Carbon | {:.4} |\n", mo.carbon));
            report.push_str(&format!("| Fruit | {:.4} |\n", mo.fruit));
            report.push_str(&format!("| Microbial | {:.4} |\n", mo.microbial));
            report.push('\n');
        }
    }

    // Stress metrics
    if let Some(last) = records.last() {
        if let Some(ref sm) = last.stress_metrics {
            report.push_str("## Stress Resilience\n\n");
            report.push_str(&format!(
                "| Metric | Value |\n|--------|-------|\n"
            ));
            report.push_str(&format!(
                "| Pre-stress biomass | {:.4} |\n",
                sm.pre_stress_biomass
            ));
            report.push_str(&format!(
                "| Min-stress biomass | {:.4} |\n",
                sm.min_stress_biomass
            ));
            report.push_str(&format!(
                "| Post-recovery biomass | {:.4} |\n",
                sm.post_recovery_biomass
            ));
            if sm.pre_stress_biomass > 0.0 {
                report.push_str(&format!(
                    "| Recovery ratio | {:.1}% |\n",
                    sm.post_recovery_biomass / sm.pre_stress_biomass * 100.0
                ));
            }
            report.push('\n');
        }
    }

    report.push_str("## Generated Files\n\n");
    report.push_str("- `convergence.svg` - Fitness convergence over generations\n");
    report.push_str("- `diversity.svg` - Population diversity over generations\n");
    report.push_str("- `param_heatmap.svg` - Best genome parameters per generation\n");
    report.push_str("- `multi_objective_radar.svg` - Multi-objective radar (if Pareto mode)\n");
    report.push_str("- `stress_resilience.svg` - Stress resilience bar chart (if stress mode)\n");

    report
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_dash_args();

    // Read telemetry
    let json_str = match fs::read_to_string(&args.telemetry) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading telemetry file '{}': {}", args.telemetry.display(), e);
            std::process::exit(1);
        }
    };

    let records: Vec<GenerationTelemetry> = match serde_json::from_str(&json_str) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error parsing telemetry JSON: {}", e);
            std::process::exit(1);
        }
    };

    eprintln!("Loaded {} telemetry records", records.len());

    // Create output directory
    if let Err(e) = fs::create_dir_all(&args.output) {
        eprintln!("Error creating output directory: {}", e);
        std::process::exit(1);
    }

    let mut files_written = 0;

    // 1. Convergence curve
    let svg = plot_convergence(&records, &args.title);
    if !svg.is_empty() {
        let path = args.output.join("convergence.svg");
        fs::write(&path, &svg).expect("write convergence.svg");
        eprintln!("  Wrote {}", path.display());
        files_written += 1;
    }

    // 2. Diversity
    let svg = plot_diversity(&records, &args.title);
    if !svg.is_empty() {
        let path = args.output.join("diversity.svg");
        fs::write(&path, &svg).expect("write diversity.svg");
        eprintln!("  Wrote {}", path.display());
        files_written += 1;
    }

    // 3. Parameter heatmap
    let svg = plot_param_heatmap(&records, &args.title);
    if !svg.is_empty() {
        let path = args.output.join("param_heatmap.svg");
        fs::write(&path, &svg).expect("write param_heatmap.svg");
        eprintln!("  Wrote {}", path.display());
        files_written += 1;
    }

    // 4. Multi-objective radar (Pareto modes)
    let svg = plot_multi_objective_radar(&records, &args.title);
    if !svg.is_empty() {
        let path = args.output.join("multi_objective_radar.svg");
        fs::write(&path, &svg).expect("write multi_objective_radar.svg");
        eprintln!("  Wrote {}", path.display());
        files_written += 1;
    }

    // 5. Stress resilience (stress modes)
    let svg = plot_stress_resilience(&records, &args.title);
    if !svg.is_empty() {
        let path = args.output.join("stress_resilience.svg");
        fs::write(&path, &svg).expect("write stress_resilience.svg");
        eprintln!("  Wrote {}", path.display());
        files_written += 1;
    }

    // 6. Summary report
    let report = generate_summary(&records, &args.title);
    let path = args.output.join("summary_report.md");
    fs::write(&path, &report).expect("write summary_report.md");
    eprintln!("  Wrote {}", path.display());
    files_written += 1;

    eprintln!("\nDashboard complete: {} files written to {}", files_written, args.output.display());
}
