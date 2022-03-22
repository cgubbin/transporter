mod global;
mod local;

pub use global::*;
pub use local::*;

use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use nalgebra_sparse::CsrMatrix;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

pub struct Hamiltonian<T: RealField>(CsrMatrix<T>);

impl<T> AsRef<CsrMatrix<T>> for Hamiltonian<T>
where
    T: RealField,
{
    fn as_ref(&self) -> &CsrMatrix<T> {
        &self.0
    }
}

impl<T> Hamiltonian<T>
where
    T: RealField,
{
    fn new<GeometryDim, C>(
        mesh: &Mesh<T, GeometryDim, C>,
        num_bands: usize,
    ) -> color_eyre::Result<Self>
    where
        GeometryDim: SmallDim,
        C: Connectivity,
        DefaultAllocator: Allocator<T, GeometryDim>,
    {
        let hamiltonian_constructor: CsrAssembler<T> = CsrAssembler::default();
        let element_assembler = ElementHamiltonianAssembler::new()
            .with_solution_dim(mesh.solution_dim())
            .with_num_bands(num_bands)
            .with_space(mesh);

        let matrix = hamiltonian_constructor.assemble(&element_assembler)?;
        Ok(Self(matrix))
    }

    pub(crate) fn num_rows(&self) -> usize {
        self.0.nrows()
    }
}
