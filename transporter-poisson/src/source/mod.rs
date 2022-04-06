use crate::{allocators::BiDimAllocator, operator::PoissonOperator};
use nalgebra::{
    DVector, DVectorSliceMut, DefaultAllocator, DimName, OPoint, OVector, RealField, Scalar,
};
use std::ops::AddAssign;
use transporter_mesher::{Assignment, FiniteDifferenceMesh, SmallDim};

pub trait SourceFunction<T, GeometryDim>: PoissonOperator<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, Self::SolutionDim>,
{
    /// Useful when the source is defined as a function
    fn from_point(&self, _point: &OPoint<T, GeometryDim>) -> Option<OVector<T, Self::SolutionDim>> {
        None
    }
    /// Useful when the source is precomputed on the mesh
    fn from_mesh(&self, node_index: usize) -> Option<OVector<T, Self::SolutionDim>>;
}

pub struct SourceFunctionAssemblerBuilder<T, RefMesh, RefSource> {
    mesh: RefMesh,
    source: RefSource,
    at_central_points: bool,
    marker: std::marker::PhantomData<T>,
}

impl SourceFunctionAssemblerBuilder<(), (), ()> {
    pub fn new() -> Self {
        Self {
            mesh: (),
            source: (),
            at_central_points: false,
            marker: std::marker::PhantomData,
        }
    }
}

impl<RefMesh, RefSource> SourceFunctionAssemblerBuilder<(), RefMesh, RefSource> {
    pub fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> SourceFunctionAssemblerBuilder<(), &Mesh, RefSource> {
        SourceFunctionAssemblerBuilder {
            mesh,
            source: self.source,
            at_central_points: self.at_central_points,
            marker: std::marker::PhantomData,
        }
    }

    pub fn with_source<Source>(
        self,
        source: &Source,
    ) -> SourceFunctionAssemblerBuilder<(), RefMesh, &Source> {
        SourceFunctionAssemblerBuilder {
            mesh: self.mesh,
            source,
            at_central_points: self.at_central_points,
            marker: std::marker::PhantomData,
        }
    }

    pub fn evaluate_at_central_points(
        self,
    ) -> SourceFunctionAssemblerBuilder<(), RefMesh, RefSource> {
        SourceFunctionAssemblerBuilder {
            mesh: self.mesh,
            source: self.source,
            at_central_points: true,
            marker: std::marker::PhantomData,
        }
    }
}

pub(crate) struct SourceFunctionAssembler<'a, T, Mesh, Source> {
    mesh: &'a Mesh,
    source: &'a Source,
    at_central_points: bool,
    marker: std::marker::PhantomData<T>,
}

impl<'a, Mesh, Source> SourceFunctionAssemblerBuilder<(), &'a Mesh, &'a Source> {
    pub(crate) fn build<T>(self) -> SourceFunctionAssembler<'a, T, Mesh, Source> {
        SourceFunctionAssembler {
            mesh: self.mesh,
            source: self.source,
            at_central_points: self.at_central_points,
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T, Mesh, Source> SourceFunctionAssembler<'a, T, Mesh, Source>
where
    T: RealField,
    Mesh: FiniteDifferenceMesh<T>,
    Source: SourceFunction<T, Mesh::GeometryDim>,
    DefaultAllocator: BiDimAllocator<T, Source::SolutionDim, Mesh::GeometryDim>,
{
    fn solution_dim(&self) -> usize {
        Source::SolutionDim::dim()
    }

    fn geometry_dim(&self) -> usize {
        Mesh::GeometryDim::dim()
    }

    fn num_nodes(&self) -> usize {
        self.mesh.number_of_nodes()
    }

    fn get_vertices(&self) -> &[(OPoint<T, Mesh::GeometryDim>, Assignment)] {
        self.mesh.get_vertices()
    }

    fn assemble_vector_into(&self, mut output: DVectorSliceMut<T>) -> color_eyre::Result<()> {
        assert_eq!(output.len(), self.num_nodes() * self.geometry_dim());
        let m = self.geometry_dim();

        if let Some(_source) = self.source.from_mesh(0) {
            for i in 0..self.num_nodes() {
                output
                    .index_mut((m * i..m * i + m, 0))
                    .add_assign(self.source.from_mesh(i).unwrap());
            }
        } else {
            for (i, point) in self.get_vertices().iter().enumerate() {
                output
                    .index_mut((m * i..m * i + m, 0))
                    .add_assign(self.source.from_point(&point.0).unwrap());
            }
        }
        Ok(())
    }

    pub fn assemble_vector(&self, n: usize) -> color_eyre::Result<DVector<T>> {
        let ndof = n * self.geometry_dim() * self.solution_dim();
        let mut output = DVector::zeros(ndof);
        self.assemble_vector_into(DVectorSliceMut::from(&mut output))?;
        Ok(output)
    }
}
