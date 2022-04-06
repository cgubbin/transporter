use crate::PoissonMethods;
use nalgebra::{DVector, RealField, U1};
use nalgebra_sparse::CsrMatrix;
use transporter_mesher::{Mesh1d, Segment1dConnectivity, SmallDim};

pub trait Jacobian<T, D>
where
    T: Copy + RealField,
    D: SmallDim,
{
    fn update_jacobian(&mut self, solution: DVector<T>) -> color_eyre::Result<CsrMatrix<T>>;
}

impl<T, D, InfoDesk> Jacobian<T, D>
    for crate::poisson1dsource::PoissonSourceb<'_, InfoDesk, Mesh1d<T>, CsrMatrix<T>, DVector<T>>
where
    T: Copy + RealField,
    D: SmallDim,
    InfoDesk: PoissonMethods<T, U1, Segment1dConnectivity>,
{
    fn update_jacobian(&mut self, _solution: DVector<T>) -> color_eyre::Result<CsrMatrix<T>> {
        todo!()
    }
}
