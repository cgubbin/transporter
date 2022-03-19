mod connectivity;
mod generate;
mod mesh;

pub use connectivity::*;
pub use generate::*;
pub use mesh::*;

use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimMin, DimName, OPoint, RealField};

pub trait SmallDim: DimName + DimMin<Self, Output = Self> {}
impl<D> SmallDim for D where D: DimName + DimMin<Self, Output = Self> {}

pub trait FiniteDifferenceMesh<T>
where
    T: RealField,
    DefaultAllocator: Allocator<T, Self::GeometryDim>,
{
    type GeometryDim: SmallDim;

    fn number_of_nodes(&self) -> usize;
    fn get_vertices(&self) -> &[OPoint<T, Self::GeometryDim>];
    fn get_connectivity(&self) -> Vec<&[usize]>;
}

impl<T, D, C> FiniteDifferenceMesh<T> for Mesh<T, D, C>
where
    T: RealField,
    C: Connectivity,
    D: SmallDim,
    DefaultAllocator: Allocator<T, D>,
{
    type GeometryDim = D;
    fn number_of_nodes(&self) -> usize {
        self.num_nodes()
    }
    fn get_vertices(&self) -> &[OPoint<T, Self::GeometryDim>] {
        self.vertices()
    }
    fn get_connectivity(&self) -> Vec<&[usize]> {
        self.connectivity()
    }
}
