// Copyright 2022 Chris Gubbin
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Operator
//!
//! Constructors to build the differential operator and source term for a Poisson problem

/// Constructors for the global (whole mesh) operator
mod global;

/// Constructors for single finite elements
mod local;

use global::CsrAssembler;

use super::{BuildError, PoissonProblemBuilder};
use crate::{
    device::info_desk::DeviceInfoDesk, outer_loop::Potential, postprocessor::Charge,
    utilities::assemblers::VertexAssemblerBuilder,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use ndarray::Array1;
use sprs::CsMat;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

/// Common behaviours for a PoissonOperator -> allows for an agnostic backend
pub(crate) trait PoissonOperator {
    /// The type of the differential operator
    type Operator;
    /// The type of the source term
    type Source;
    /// Builds the differential operator
    fn build_operator(&self) -> Result<Self::Operator, BuildError>;
    /// Builds the source term
    fn build_source(&self) -> Result<Self::Source, BuildError>;
}

impl<T: Copy + RealField, GeometryDim, Conn, BandDim> PoissonOperator
    for PoissonProblemBuilder<
        T,
        &'_ Charge<T, BandDim>,
        &'_ DeviceInfoDesk<T, GeometryDim, BandDim>,
        &'_ Potential<T>,
        &'_ Mesh<T, GeometryDim, Conn>,
    >
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<T, GeometryDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<Array1<T>, BandDim>,
{
    type Operator = CsMat<T>;
    type Source = Array1<T>;
    fn build_operator(&self) -> Result<CsMat<T>, BuildError> {
        // Build out the constructors
        let vertex_assembler = VertexAssemblerBuilder::new()
            .with_info_desk(self.info_desk)
            .with_mesh(self.mesh)
            .build();
        let poisson_operator_constructor: CsrAssembler<T> =
            CsrAssembler::from_vertex_assembler(&vertex_assembler)?;
        poisson_operator_constructor.assemble_operator(&vertex_assembler)
    }
    fn build_source(&self) -> Result<Array1<T>, BuildError> {
        // Build out the constructors
        let vertex_assembler = VertexAssemblerBuilder::new()
            .with_info_desk(self.info_desk)
            .with_mesh(self.mesh)
            .build();
        let poisson_operator_constructor: CsrAssembler<T> =
            CsrAssembler::from_vertex_assembler(&vertex_assembler)?;
        poisson_operator_constructor.assemble_static_source(&vertex_assembler)
    }
}

/// An `InfoDesk` trait defining the necessary methods to construct the Poisson problem
pub trait PoissonInfoDesk<T: Copy + RealField> {
    /// The spatial dimension
    type GeometryDim: SmallDim;
    /// The number of bands in the problem
    type BandDim: SmallDim;
    /// The zero-frequency dielectric constant along the three Cartesian axis
    fn get_static_dielectric_constant(&self, region_index: usize) -> &[T; 3];
    /// The acceptor dopant density in 1 / m^3
    fn get_acceptor_density(&self, region_index: usize) -> T;
    /// The donor dopant density in 1 / m^3
    fn get_donor_density(&self, region_index: usize) -> T;
}

/// Implement the PoissonInfoDesk for the general DeviceInfoDesk struct
impl<T, GeometryDim, BandDim> PoissonInfoDesk<T> for DeviceInfoDesk<T, GeometryDim, BandDim>
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    type GeometryDim = GeometryDim;
    type BandDim = BandDim;
    fn get_static_dielectric_constant(&self, region_index: usize) -> &[T; 3] {
        &self.dielectric_constants[region_index]
    }
    fn get_acceptor_density(&self, region_index: usize) -> T {
        self.acceptor_densities[region_index]
    }
    fn get_donor_density(&self, region_index: usize) -> T {
        self.donor_densities[region_index]
    }
}
