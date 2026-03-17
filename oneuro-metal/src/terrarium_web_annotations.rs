//! Educational annotations for the terrarium web app.
//!
//! Static embedded content describing the biology/ecology models used in the
//! terrarium simulation. Each annotation includes interactive parameter sliders
//! and mathematical equations.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Annotation {
    pub id: &'static str,
    pub title: &'static str,
    pub category: &'static str,
    pub short_desc: &'static str,
    pub explanation: &'static str,
    pub equation: Option<&'static str>,
    pub params: Vec<AnnotationParam>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnnotationParam {
    pub name: &'static str,
    pub label: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pub step: f32,
}

/// Returns all 10 bundled educational annotations.
pub fn all_annotations() -> Vec<Annotation> {
    vec![
        Annotation {
            id: "beer_lambert",
            title: "Beer-Lambert Law",
            category: "Ecology",
            short_desc: "Light attenuates exponentially through plant canopy layers.",
            explanation: "The Beer-Lambert law models how light intensity decreases as it passes through the canopy. Each leaf layer absorbs a fraction of incoming light, creating vertical light gradients that drive plant competition. Plants with higher Leaf Area Index (LAI) shade out competitors below. The extinction coefficient k depends on leaf angle distribution — horizontal leaves (k≈1.0) shade more effectively than vertical leaves (k≈0.5). This is the foundation of the plant competition module.",
            equation: Some("I = I₀ × e^(-k × LAI)"),
            params: vec![
                AnnotationParam { name: "lai", label: "LAI", min: 0.0, max: 8.0, default: 3.0, step: 0.1 },
                AnnotationParam { name: "k", label: "Extinction (k)", min: 0.1, max: 1.0, default: 0.65, step: 0.05 },
            ],
        },
        Annotation {
            id: "michaelis_menten",
            title: "Michaelis-Menten Kinetics",
            category: "Chemistry",
            short_desc: "Enzyme reaction rate saturates at high substrate concentration.",
            explanation: "Michaelis-Menten kinetics describe how enzyme-catalyzed reactions speed up with substrate concentration but eventually saturate when all enzyme active sites are occupied. Vmax is the maximum rate when enzyme is fully saturated. Km (Michaelis constant) is the substrate concentration at half-maximum rate — lower Km means higher enzyme affinity. This model governs glucose metabolism, ATP synthesis, and nutrient uptake throughout the terrarium.",
            equation: Some("v = Vmax × [S] / (Km + [S])"),
            params: vec![
                AnnotationParam { name: "vmax", label: "Vmax", min: 0.1, max: 10.0, default: 5.0, step: 0.1 },
                AnnotationParam { name: "km", label: "Km", min: 0.1, max: 10.0, default: 2.0, step: 0.1 },
                AnnotationParam { name: "s", label: "[S]", min: 0.0, max: 20.0, default: 5.0, step: 0.5 },
            ],
        },
        Annotation {
            id: "sharpe_schoolfield",
            title: "Sharpe-Schoolfield Temperature Response",
            category: "Ecology",
            short_desc: "Insect development rate peaks at an optimal temperature then drops sharply.",
            explanation: "The Sharpe-Schoolfield model captures how insect metabolic and developmental rates depend on temperature. Unlike simple Q10 models, it correctly predicts the sharp decline at high temperatures due to enzyme denaturation. The model uses thermodynamic parameters: activation enthalpy (ΔH_A) controls the rising phase, while high-temperature deactivation enthalpy (ΔH_H) and half-life temperature (T_H) control the decline. This drives the fly population lifecycle in the terrarium — egg, larva, pupa, and adult stages each have temperature-dependent durations.",
            equation: Some("r(T) = r₂₅ × (T/298) × exp(ΔH_A/R × (1/298 - 1/T)) / (1 + exp(ΔH_H/R × (1/T_H - 1/T)))"),
            params: vec![
                AnnotationParam { name: "temp", label: "Temperature (°C)", min: 5.0, max: 45.0, default: 25.0, step: 0.5 },
            ],
        },
        Annotation {
            id: "monod",
            title: "Monod Growth Kinetics",
            category: "Chemistry",
            short_desc: "Microbial growth rate depends on limiting substrate availability.",
            explanation: "The Monod equation describes microbial population growth as a function of substrate (nutrient) concentration. It has the same mathematical form as Michaelis-Menten but applies at the population level rather than the enzyme level. µmax is the maximum specific growth rate, and Ks is the half-saturation constant. When substrate is scarce ([S] << Ks), growth is approximately linear with substrate. When abundant ([S] >> Ks), growth approaches µmax. This governs soil microbe dynamics in the terrarium.",
            equation: Some("µ = µmax × [S] / (Ks + [S])"),
            params: vec![
                AnnotationParam { name: "umax", label: "µmax (h⁻¹)", min: 0.01, max: 2.0, default: 0.5, step: 0.01 },
                AnnotationParam { name: "ks", label: "Ks (mM)", min: 0.01, max: 5.0, default: 0.5, step: 0.01 },
                AnnotationParam { name: "s", label: "[S] (mM)", min: 0.0, max: 10.0, default: 2.0, step: 0.1 },
            ],
        },
        Annotation {
            id: "fly_metabolism",
            title: "Fly Molecular Metabolism",
            category: "Metabolism",
            short_desc: "7-pool Michaelis-Menten model of Drosophila energy metabolism.",
            explanation: "Each fly maintains 7 metabolic pools: hemolymph trehalose (blood sugar), hemolymph glucose, fat body glycogen, fat body lipids, flight muscle ATP, ADP, and AMP. Trehalose is the primary energy currency — it's converted to glucose by trehalase, then phosphorylated and oxidized to produce ATP. The energy charge EC = (ATP + 0.5×ADP) / (ATP + ADP + AMP) indicates cellular energy status. When EC drops below 0.3, ATP crash occurs. When trehalose drops below 5.0 mM, starvation onset is triggered. Feeding replenishes trehalose from ingested sugars.",
            equation: Some("EC = (ATP + 0.5×ADP) / (ATP + ADP + AMP)"),
            params: vec![
                AnnotationParam { name: "trehalose", label: "Trehalose (mM)", min: 0.0, max: 30.0, default: 15.0, step: 0.5 },
                AnnotationParam { name: "activity", label: "Activity", min: 0.0, max: 1.0, default: 0.3, step: 0.05 },
            ],
        },
        Annotation {
            id: "nsga2",
            title: "NSGA-II Multi-Objective Evolution",
            category: "Evolution",
            short_desc: "Non-dominated sorting finds Pareto-optimal genome trade-offs.",
            explanation: "NSGA-II (Non-dominated Sorting Genetic Algorithm II) optimizes multiple objectives simultaneously. Instead of finding a single 'best' solution, it discovers the Pareto front — the set of solutions where improving one objective necessarily worsens another. Solutions are ranked by Pareto dominance: rank 0 entries are not dominated by any other solution. Crowding distance preserves diversity along the front. The terrarium uses 6 objectives: biomass, biodiversity, stability, carbon sequestration, fruit production, and microbial health.",
            equation: None,
            params: vec![],
        },
        Annotation {
            id: "lotka_volterra",
            title: "Lotka-Volterra Predation",
            category: "Ecology",
            short_desc: "Predator-prey cycles between nematodes and soil microbes.",
            explanation: "The Lotka-Volterra equations model predator-prey dynamics. Prey (microbes) grow exponentially when predators are absent, but are consumed at a rate proportional to the product of both populations. Predators (nematodes) decline without prey but increase when food is available. This creates characteristic oscillatory dynamics — predator population peaks lag behind prey peaks. In the terrarium's soil fauna module, bacterivorous nematodes control microbial populations, creating nutrient cycling feedback loops.",
            equation: Some("dN/dt = αN - βNP ; dP/dt = δNP - γP"),
            params: vec![
                AnnotationParam { name: "alpha", label: "Prey growth (α)", min: 0.01, max: 2.0, default: 0.5, step: 0.01 },
                AnnotationParam { name: "beta", label: "Predation (β)", min: 0.001, max: 0.1, default: 0.02, step: 0.001 },
            ],
        },
        Annotation {
            id: "hodgkin_huxley",
            title: "Hodgkin-Huxley Ion Channel Dynamics",
            category: "Physics",
            short_desc: "Voltage-gated ion channels generate neural action potentials.",
            explanation: "The Hodgkin-Huxley model describes how neurons generate electrical spikes (action potentials) through voltage-gated sodium and potassium channels. Each channel type has activation (m, n) and inactivation (h) gates with voltage-dependent opening/closing rates. When membrane potential reaches threshold, Na+ channels open rapidly (depolarization), then K+ channels open (repolarization), and Na+ channels inactivate (refractory period). This is the foundation of the neural simulation that drives fly behavior in the terrarium — each DrosophilaSim instance runs a molecular HH network.",
            equation: Some("C × dV/dt = -g_Na × m³h × (V-E_Na) - g_K × n⁴ × (V-E_K) - g_L × (V-E_L) + I"),
            params: vec![
                AnnotationParam { name: "current", label: "Stimulus (nA)", min: 0.0, max: 20.0, default: 7.0, step: 0.5 },
            ],
        },
        Annotation {
            id: "nitrogen_cycle",
            title: "Nitrogen Cycle",
            category: "Chemistry",
            short_desc: "Nitrification converts ammonium to nitrate via nitrite intermediary.",
            explanation: "The nitrogen cycle in soil converts ammonium (NH4+) to nitrate (NO3-) through a two-step bacterial process called nitrification. Ammonia-oxidizing bacteria (AOB like Nitrosomonas) oxidize NH4+ to nitrite (NO2-), then nitrite-oxidizing bacteria (NOB like Nitrobacter) oxidize NO2- to NO3-. Plants can absorb both NH4+ and NO3-, but most prefer nitrate. Under low-oxygen conditions, denitrifying bacteria convert NO3- back to N2 gas. The terrarium tracks NH4+, NO3-, and redox potential to model these transformations in soil chemistry.",
            equation: Some("NH4+ → NO2- → NO3- (nitrification) ; NO3- → N2 (denitrification)"),
            params: vec![
                AnnotationParam { name: "nh4", label: "NH4+ (µM)", min: 0.0, max: 100.0, default: 20.0, step: 1.0 },
                AnnotationParam { name: "o2", label: "O2 (%)", min: 0.0, max: 21.0, default: 15.0, step: 0.5 },
            ],
        },
        Annotation {
            id: "farquhar",
            title: "Farquhar Photosynthesis Model",
            category: "Metabolism",
            short_desc: "C3 carbon fixation limited by RuBisCO or electron transport.",
            explanation: "The Farquhar-von Caemmerer-Berry (FvCB) model describes C3 photosynthesis as the minimum of two rates: RuBisCO-limited (Wc, depends on CO2 and enzyme kinetics) and electron-transport-limited (Wj, depends on light). At low light, photosynthesis is light-limited. At high light and low CO2, it's RuBisCO-limited. The model predicts the CO2 compensation point and the light saturation curve. In the terrarium, plant cells use a simplified FvCB to compute carbon fixation rate, coupling light availability (Beer-Lambert) with atmospheric CO2.",
            equation: Some("A = min(Wc, Wj) - Rd"),
            params: vec![
                AnnotationParam { name: "light", label: "PAR (µmol/m²/s)", min: 0.0, max: 2000.0, default: 500.0, step: 50.0 },
                AnnotationParam { name: "co2", label: "CO2 (ppm)", min: 100.0, max: 1000.0, default: 400.0, step: 10.0 },
            ],
        },
    ]
}
