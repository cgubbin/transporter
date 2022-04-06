#![allow(dead_code)]
/// This module solves the Poisson equation for electron transport on a cell-centered
/// grid.
///
/// In 1-dimension the Poisson equation is
/// $ \mathrm{d}^2 / \mathrm{d} x^2 \phi = F
mod allocators;
mod jacobian;
mod operator;
mod poisson1dsource;
mod solve;
mod source;

pub use poisson1dsource::*;
