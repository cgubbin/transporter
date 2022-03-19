use crate::{Connectivity, Segment1dConnectivity};
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint, RealField, U1};

/// A generic `Mesh` object valid in all dimensions
pub struct Mesh<T: RealField, D, Connectivity>
where
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    vertices: Vec<OPoint<T, D>>,
    connectivity: Vec<Connectivity>,
}

impl<T, D, Connectivity> Mesh<T, D, Connectivity>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn num_nodes(&self) -> usize {
        self.vertices.len()
    }
}

/// Type aliases for implemented mesh dimensionalitys and discretisations
pub type Mesh1d<T> = Mesh<T, U1, Segment1dConnectivity>;

impl<T, D, C> Mesh<T, D, C>
where
    T: RealField,
    C: Connectivity,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn vertices_owned(self) -> Vec<OPoint<T, D>> {
        self.vertices
    }

    pub fn vertices_mut(&mut self) -> &mut [OPoint<T, D>] {
        &mut self.vertices
    }

    pub fn vertices(&self) -> &[OPoint<T, D>] {
        &self.vertices
    }

    pub fn connectivity(&self) -> Vec<&[usize]> {
        self.connectivity.iter().map(|x| x.as_inner()).collect()
    }

    pub fn from_vertices_and_connectivity(
        vertices: Vec<OPoint<T, D>>,
        connectivity: Vec<C>,
    ) -> Self {
        Self {
            vertices,
            connectivity,
        }
    }
}
