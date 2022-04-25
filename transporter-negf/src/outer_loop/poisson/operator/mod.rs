mod global;
mod local;

use global::CsrAssembler;

use super::{BuildError, PoissonProblemBuilder};
use crate::device::info_desk::DeviceInfoDesk;
use crate::outer_loop::Potential;
use crate::postprocessor::Charge;
use crate::utilities::assemblers::VertexAssemblerBuilder;
use nalgebra::{allocator::Allocator, DVector, DefaultAllocator, RealField};
use nalgebra::{Const, Dynamic, Matrix, VecStorage};
use nalgebra_sparse::CsrMatrix;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

pub(crate) trait PoissonOperator {
    type Operator;
    type Source;
    fn build_operator(&self) -> Result<Self::Operator, BuildError>;
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
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    type Operator = CsrMatrix<T>;
    type Source = DVector<T>;
    fn build_operator(&self) -> Result<CsrMatrix<T>, BuildError> {
        // Build out the constructors
        let vertex_assembler = VertexAssemblerBuilder::new()
            .with_info_desk(self.info_desk)
            .with_mesh(self.mesh)
            .build();
        let poisson_operator_constructor: CsrAssembler<T> =
            CsrAssembler::from_vertex_assembler(&vertex_assembler)?;
        poisson_operator_constructor.assemble_operator(&vertex_assembler)
    }
    fn build_source(&self) -> Result<DVector<T>, BuildError> {
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

pub trait PoissonInfoDesk<T: Copy + RealField> {
    type GeometryDim: SmallDim;
    type BandDim: SmallDim;
    fn get_static_dielectric_constant(&self, region_index: usize) -> &[T; 3];
    fn get_acceptor_density(&self, region_index: usize) -> T;
    fn get_donor_density(&self, region_index: usize) -> T;
}

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
