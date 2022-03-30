#![allow(dead_code)]

mod connectivity;
mod generate;
mod mesh;
mod primitives;

pub use connectivity::*;
pub use generate::*;
pub use mesh::*;
pub use primitives::*;

use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimMin, DimName, OPoint, RealField, U1, U2};

/// A composite trait for the mesh allocator
pub trait MeshAllocator<T: RealField, GeometryDim: SmallDim>: Allocator<T, GeometryDim> {}

/// We only want to allow dimensions that we can use to be called, so this
/// custom trait is ONLY implemented for the dimensions of mesh we are using
/// which simplifies the generic code needed in the NEGF module
pub trait SmallDim: DimName + DimMin<Self, Output = Self> {}
//impl<D> SmallDim for U2 where D: DimName + DimMin<Self, Output = Self> {}
impl SmallDim for U1 {}
impl SmallDim for U2 {}

pub trait FiniteDifferenceMesh<T>
where
    T: RealField,
    DefaultAllocator: Allocator<T, Self::GeometryDim>,
{
    type GeometryDim: SmallDim;

    fn number_of_nodes(&self) -> usize;
    fn get_vertices(&self) -> &[(OPoint<T, Self::GeometryDim>, Assignment)];
    fn get_connectivity(&self) -> Vec<&[usize]>;
}

impl<T, D, C> FiniteDifferenceMesh<T> for Mesh<T, D, C>
where
    T: RealField,
    C: Connectivity<T, D>,
    D: SmallDim,
    DefaultAllocator: Allocator<T, D>,
{
    type GeometryDim = D;
    fn number_of_nodes(&self) -> usize {
        self.num_nodes()
    }
    fn get_vertices(&self) -> &[(OPoint<T, Self::GeometryDim>, Assignment)] {
        self.vertices()
    }
    fn get_connectivity(&self) -> Vec<&[usize]> {
        self.connectivity()
    }
}
