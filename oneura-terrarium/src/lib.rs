//! oneura-terrarium: Multi-scale biological simulation framework
//!
//! This crate will contain the terrarium simulation extracted from oneuro-metal.
//! Currently a scaffold — see docs/CRATE_EXTRACTION_PLAN.md for the migration plan.
//!
//! # Architecture
//!
//! The terrarium operates at 7 biological scales:
//! - **Quantum**: Exact diagonalization, electron transfer (Marcus theory)
//! - **Atomistic**: Molecular dynamics with TIP3P water
//! - **Metabolic**: 7-pool Michaelis-Menten kinetics (Drosophila)
//! - **Cellular**: Stochastic gene expression (Gillespie tau-leaping)
//! - **Organismal**: Temperature-dependent development (Sharpe-Schoolfield)
//! - **Ecological**: Beer-Lambert competition, Lotka-Volterra soil fauna
//! - **Evolutionary**: NSGA-II multi-objective optimization

// Placeholder — modules will be migrated from oneuro-metal/src/
