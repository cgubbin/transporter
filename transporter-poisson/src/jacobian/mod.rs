use nalgebra::{DVector, RealField};
use nalgebra_sparse::CsrMatrix;
use transporter_mesher::{Mesh1d, SmallDim};

pub trait Jacobian<T, D>
where
    T: Copy + RealField,
    D: SmallDim,
{
    fn update_jacobian(&mut self, solution: DVector<T>) -> color_eyre::Result<CsrMatrix<T>>;
}

impl<T, D> Jacobian<T, D>
    for crate::poisson1dsource::PoissonSourceb<
        '_,
        Mesh1d<T>,
        CsrMatrix<T>,
        DVector<T>,
        CsrMatrix<T>,
    >
where
    T: Copy + RealField,
    D: SmallDim,
{
    fn update_jacobian(&mut self, _solution: DVector<T>) -> color_eyre::Result<CsrMatrix<T>> {
        todo!()
    }
}
