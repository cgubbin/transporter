// Copyright 2022 Chris Gubbin
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Green's Function Methods
//!
//! This module contains algorithms to calculate Greens functions for various degrees of
//! numerical approximation

/// Aggregate methods. Defines composite structs holding Greens functions for all energies and wavevctors in the spectral grid
pub mod aggregate;

/// Dense methods: Calculate Green's functions as full matrices at all mesh points in the spatial grid
pub mod dense;

/// Mixed methods: Calculate dense greens functions in a limited core region where incoherent scattering is considered
/// and sparse Green's functions in the coherent contacts
pub mod mixed;

/// Recursive methods to allow for rapid calculation of sparse Green's functions without matrix inversion
pub mod recursive;
/// Sparse methods: For coherent calculations: calculate the Greens functions in sparse CsrMat form, only the diagonal
/// and left and right columns are calculated
pub mod sparse;
