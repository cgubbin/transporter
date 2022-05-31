// Copyright 2022 Chris Gubbin
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Transporter is an incoherent electron transport solver written in Rust
//!
//! [Documentation of most recent release](https://docs.rs/transporter/latest/transporter)
//!
//! # Overview
//! Transporter calculates the electron transport in semiconductor structures utilising the non-equilibrium
//! Green's function approach ([Lake 1997](https://doi.org/10.1063/1.365394)). In this formalism the electronic
//! density is calculated from the Green's functions of the system Hamiltonian. As the electronic density is dependent
//! on the electrostatic potential in the system, arriving at a self-consistent solution entails iteratively solving the Poisson
//! and Schrodinger equations until convergence is reached.
//!
//! The aim of Transporter is to study the interaction between electrons and surface phonon polaritons in polar dielectric
//! systems. Electrons emit longitudinal phonons in a polar crystal, this can be described in an NEGF formalism by
//! modelling the interaction through a coupling self energy. Emission of LO phonons shuffles electrons between states, this
//! means describing the process entails a second self-consistent calculation.
//!
//! # Usage
//! Transporter is distributed as a binary crate, and is intended to be run from the command line. To run the software first define
//! a structure in a `.toml` file:
//!
//! ```toml
//! temperature = 300.0
//! voltage_offsets = [0.0, 0.00]
//!
//! [[layers]]
//! thickness = [10.0]
//! material = "GaAs"
//! acceptor_density = 0.0
//! donor_density = 1e23
//! ```
//!
//! where additional layers can be appended with subsequent `layers` fields.

#![warn(missing_docs)]
#![allow(dead_code)]
#![allow(clippy::type_complexity)]
#![feature(path_file_prefix)]

extern crate openblas_src;

/// The command line global application, tracing and display primitives
pub mod app;

/// Physical constants
mod constants;

/// Device and geometry
pub mod device;

/// Error handling
mod error;

/// Fermi integrals and their inverses
pub mod fermi;

/// Greens function methods
pub mod greens_functions;

/// System Hamiltonian
pub mod hamiltonian;

/// The inner loop, which computes the electronic density
mod inner_loop;

/// The outer loop, which iteratively solves an inner loop and the Poisson equation for the electric potential
mod outer_loop;

/// Computes quantities of interest from Greens functions, such as the electron density and current
mod postprocessor;

/// Self energies for coherent and incoherent transport
pub mod self_energy;

/// Discrete energy and wavevector spaces
pub mod spectral;

/// Helper functions and traits
mod utilities;

// pub use constants::*;
// pub use hamiltonian::*;
