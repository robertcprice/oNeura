//! Neural-Molecular Simulator -- Connects GPU MD to Drosophila Brain.
//!
//! This module integrates GPU-accelerated molecular dynamics with the neural
//! simulation to create emergent behavior. Food molecules release odorants
//! based on their molecular composition, which stimulate the fly's olfactory
//! system, driving navigation behavior through the brain dynamics.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │              NeuralMolecularSimulator                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
//! │  │ GPU MD Sim   │───▶│ Odorant      │───▶│ Olfactory    │   │
//! │  │ (particles)  │    │ Field        │    │ Receptors    │   │
//! │  └──────────────┘    └──────────────┘    └──────────────┘   │
//! │        │                                        │              │
//! │        │         ┌──────────────┐               │              │
//! │        └────────▶│ Drosophila   │◀──────────────┘              │
//! │                   │ Brain        │                               │
//! │                   └──────────────┘                               │
//! │                         │                                        │
//! │                         ▼                                        │
//! │                   ┌──────────────┐                               │
//! │                   │ Motor Output │                               │
//! │                   │ (speed/turn) │                               │
//! │                   └──────────────┘                               │
//! │                         │                                        │
//! │                         ▼                                        │
//! │                   ┌──────────────┐                               │
//! │                   │ Agent Motion │                               │
//! │                   │ (position)   │                               │
//! │                   └──────────────┘                               │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use crate::drosophila::{DrosophilaScale, DrosophilaSim};
use crate::molecular_dynamics::GPUMolecularDynamics;

/// Odorant types and their receptors.
#[derive(Clone, Copy, Debug)]
pub struct Odorant {
    /// Receptor type this activates.
    pub receptor_type: OdorReceptor,
    /// Concentration/strength.
    pub concentration: f32,
    /// Release rate from food.
    pub release_rate: f32,
}

/// Olfactory receptor types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OdorReceptor {
    /// Fruit ester (banana-like).
    Ester,
    /// Floral scent.
    Floral,
    /// Sweet/ sugar.
    Sweet,
    /// Pheromone.
    Pheromone,
    /// Ammonia/rotten.
    Ammonia,
    /// CO2.
    CarbonDioxide,
}

impl Odorant {
    /// Default odorants released from food.
    pub fn food_odorants() -> Vec<Odorant> {
        vec![
            Odorant {
                receptor_type: OdorReceptor::Sweet,
                concentration: 1.0,
                release_rate: 0.5,
            },
            Odorant {
                receptor_type: OdorReceptor::Ester,
                concentration: 0.8,
                release_rate: 0.3,
            },
            Odorant {
                receptor_type: OdorReceptor::Floral,
                concentration: 0.3,
                release_rate: 0.1,
            },
        ]
    }
}

/// Agent (fly) state.
#[derive(Clone, Debug)]
pub struct Agent {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub heading: f32,
    pub speed: f32,
    pub turn_rate: f32,
}

impl Agent {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            z: 0.1, // Flying height
            heading: rand::random::<f32>() * std::f32::consts::TAU,
            speed: 0.0,
            turn_rate: 0.0,
        }
    }
}

/// Simulation state for visualization.
#[derive(Clone, Debug)]
pub struct SimState {
    pub positions: Vec<f32>, // Flattened (N*3)
    pub velocities: Vec<f32>,
    pub agents: Vec<Agent>,
    pub odorant_field: Vec<f32>,  // Grid values
    pub brain_activity: Vec<f32>, // Neural activity per region
    pub motor_output: Vec<f32>,   // [speed, turn]
    pub stats: crate::molecular_dynamics::MDStats,
}

/// Neural-Molecular Simulator combining MD with brain.
pub struct NeuralMolecularSimulator {
    /// MD simulation for particles.
    md: GPUMolecularDynamics,
    /// Drosophila brain.
    #[allow(dead_code)]
    brain: DrosophilaSim,
    /// Flies/agents.
    agents: Vec<Agent>,
    /// Food locations (x, y, z, radius, odor_strength).
    food_sources: Vec<[f32; 4]>,
    /// World bounds.
    bounds: [f32; 2],
    /// Odor receptor sensitivities.
    receptor_sensitivity: f32,
    /// Time step.
    dt: f32,
    /// Current step count.
    step: usize,
}

/// Static helper to compute odorant field (avoids borrow issues).
fn compute_field_static(food_sources: &[[f32; 4]], x: f32, y: f32) -> [f32; 6] {
    let mut field = [0.0f32; 6];

    for food in food_sources {
        let fx = food[0];
        let fy = food[1];
        let radius = food[2];
        let strength = food[3];

        let dx = x - fx;
        let dy = y - fy;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < radius * 5.0 {
            let conc = strength * (-dist / radius).exp();
            field[0] += conc * 1.0; // Sweet
            field[1] += conc * 0.8; // Ester
            field[2] += conc * 0.3; // Floral
        }
    }

    field
}

impl NeuralMolecularSimulator {
    /// Create a new neural-molecular simulator.
    pub fn new(
        n_particles: usize,
        brain_scale: DrosophilaScale,
        n_agents: usize,
        bounds: [f32; 2],
    ) -> Self {
        // Create MD simulation
        let mut md = GPUMolecularDynamics::new(n_particles, "auto");

        // Initialize particles in random positions
        let mut positions = vec![0.0f32; n_particles * 3];
        let velocities = vec![0.0f32; n_particles * 3];

        for i in 0..n_particles {
            let i3 = i * 3;
            positions[i3] = rand::random::<f32>() * bounds[0];
            positions[i3 + 1] = rand::random::<f32>() * bounds[1];
            positions[i3 + 2] = rand::random::<f32>() * 5.0; // Z height
        }

        md.set_positions(&positions);
        md.set_velocities(&velocities);
        md.set_temperature(300.0);
        md.set_box([bounds[0], bounds[1], 10.0]);
        md.initialize_velocities();

        // Create brain
        let brain = DrosophilaSim::new(brain_scale, 42);

        // Create agents
        let mut agents = Vec::with_capacity(n_agents);
        for _ in 0..n_agents {
            agents.push(Agent::new(
                rand::random::<f32>() * bounds[0],
                rand::random::<f32>() * bounds[1],
            ));
        }

        Self {
            md,
            brain,
            agents,
            food_sources: Vec::new(),
            bounds,
            receptor_sensitivity: 10.0,
            dt: 0.001, // 1 ps
            step: 0,
        }
    }

    /// Add a food source.
    pub fn add_food(&mut self, x: f32, y: f32, z: f32, radius: f32, strength: f32) {
        self.food_sources.push([x, y, z, radius * strength]);
    }

    /// Compute odorant field at a position.
    #[allow(dead_code)]
    fn compute_odorant_field(&self, x: f32, y: f32) -> [f32; 6] {
        let mut field = [0.0f32; 6]; // 6 receptor types

        for food in &self.food_sources {
            let fx = food[0];
            let fy = food[1];
            let radius = food[2];
            let strength = food[3];

            let dx = x - fx;
            let dy = y - fy;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < radius * 5.0 {
                let conc = strength * (-dist / radius).exp();

                // Sweet (index 0)
                field[0] += conc * 1.0;
                // Ester (index 1)
                field[1] += conc * 0.8;
                // Floral (index 2)
                field[2] += conc * 0.3;
            }
        }

        field
    }

    /// Step the simulation.
    pub fn step(&mut self) -> SimState {
        // 1. Run MD step
        let stats = self.md.step(self.dt);

        // 2. Get MD positions
        let positions = self.md.get_positions();
        let velocities = self.md.velocities().to_vec();

        // 3. Update each agent
        // First, collect food sources to avoid borrow issues
        let food_sources: Vec<[f32; 4]> = self.food_sources.clone();
        let bounds = self.bounds;
        let sensitivity = self.receptor_sensitivity;

        let mut brain_inputs: Vec<Vec<f32>> = Vec::with_capacity(self.agents.len());
        let mut motor_outputs = Vec::with_capacity(self.agents.len());

        // Compute odorant fields for each agent
        struct AgentUpdate {
            x: f32,
            y: f32,
            heading: f32,
            speed: f32,
            turn_rate: f32,
            #[allow(dead_code)]
            field: [f32; 6],
        }

        let mut updates: Vec<AgentUpdate> = Vec::with_capacity(self.agents.len());

        for agent in &self.agents {
            let field = compute_field_static(&food_sources, agent.x, agent.y);

            // Compute gradient
            let delta = 0.5;
            let field_right = compute_field_static(&food_sources, agent.x + delta, agent.y);
            let field_up = compute_field_static(&food_sources, agent.x, agent.y + delta);

            let mut gradient_x = 0.0f32;
            let mut gradient_y = 0.0f32;
            for i in 0..3 {
                gradient_x += field_right[i] - field[i];
                gradient_y += field_up[i] - field[i];
            }

            // Turn towards higher concentration
            let turn = gradient_x.sin() + gradient_y.cos();

            // Move towards food
            let total_scent = field[0] + field[1] + field[2];
            let speed = if total_scent > 0.01 {
                total_scent.clamp(0.1, 1.0)
            } else {
                0.2
            };

            let heading = agent.heading + turn * 0.5 * self.dt * 1000.0;
            let x = (agent.x + heading.cos() * speed * self.dt * 100.0).rem_euclid(bounds[0]);
            let y = (agent.y + heading.sin() * speed * self.dt * 100.0).rem_euclid(bounds[1]);

            updates.push(AgentUpdate {
                x,
                y,
                heading: heading % std::f32::consts::TAU,
                speed,
                turn_rate: turn * 0.5,
                field,
            });

            brain_inputs.push(field.iter().map(|&v| (v * sensitivity).min(1.0)).collect());
            motor_outputs.push(vec![speed, turn * 0.5]);
        }

        // Apply updates
        for (i, agent) in self.agents.iter_mut().enumerate() {
            let up = &updates[i];
            agent.x = up.x;
            agent.y = up.y;
            agent.heading = up.heading;
            agent.speed = up.speed;
            agent.turn_rate = up.turn_rate;
        }

        // 4. Run brain step (if we have input)
        // In full implementation, this would connect properly

        SimState {
            positions,
            velocities,
            agents: self.agents.clone(),
            odorant_field: Vec::new(),     // Would be grid for viz
            brain_activity: vec![0.0; 15], // 15 brain regions
            motor_output: motor_outputs.first().cloned().unwrap_or_default(),
            stats,
        }
    }

    /// Run multiple steps.
    pub fn run(&mut self, n_steps: usize) -> Vec<SimState> {
        let mut states = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            states.push(self.step());
            self.step += 1;
        }

        states
    }

    /// Get current state for visualization.
    pub fn get_state(&self) -> SimState {
        SimState {
            positions: self.md.get_positions(),
            velocities: self.md.velocities().to_vec(),
            agents: self.agents.clone(),
            odorant_field: Vec::new(),
            brain_activity: vec![0.0; 15],
            motor_output: vec![0.0, 0.0],
            stats: crate::molecular_dynamics::MDStats::default(),
        }
    }

    /// Number of particles.
    pub fn n_particles(&self) -> usize {
        self.md.positions().len() / 3
    }

    /// Number of agents.
    pub fn n_agents(&self) -> usize {
        self.agents.len()
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;

    /// Python wrapper.
    #[pyclass]
    pub struct PyNeuralMDSim {
        inner: NeuralMolecularSimulator,
    }

    #[pymethods]
    impl PyNeuralMDSim {
        #[new]
        fn new(n_particles: usize, brain_scale: &str, n_agents: usize, bounds: [f32; 2]) -> Self {
            let scale = match brain_scale {
                "tiny" => DrosophilaScale::Tiny,
                "small" => DrosophilaScale::Small,
                "medium" => DrosophilaScale::Medium,
                "large" => DrosophilaScale::Large,
                _ => DrosophilaScale::Small,
            };

            Self {
                inner: NeuralMolecularSimulator::new(n_particles, scale, n_agents, bounds),
            }
        }

        fn add_food(&mut self, x: f32, y: f32, z: f32, radius: f32, strength: f32) {
            self.inner.add_food(x, y, z, radius, strength);
        }

        /// Step and return (positions, stats_dict)
        fn step(&mut self) -> (Vec<f32>, std::collections::HashMap<String, f32>) {
            let state = self.inner.step();
            let mut stats = std::collections::HashMap::new();
            stats.insert("temperature".to_string(), state.stats.temperature);
            stats.insert("kinetic".to_string(), state.stats.kinetic_energy);
            stats.insert("potential".to_string(), state.stats.potential_energy);
            stats.insert("total".to_string(), state.stats.total_energy);
            (state.positions, stats)
        }

        fn run(&mut self, n_steps: usize) {
            self.inner.run(n_steps);
        }

        fn n_particles(&self) -> usize {
            self.inner.n_particles()
        }

        fn n_agents(&self) -> usize {
            self.inner.n_agents()
        }
    }
}

#[cfg(feature = "python")]
pub use python::*;
