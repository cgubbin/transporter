use crate::SmallDim;
use nalgebra::{DVector, RealField};
use transporter_mesher::Mesh1d;

pub trait NewtonSolver<T, GeometryDim>
where
    T: RealField,
    GeometryDim: SmallDim,
{
    fn solve_into(&self, solution: DVector<T>) -> color_eyre::Result<DVector<T>>;
}

use nalgebra_sparse::CscMatrix;

impl<T> NewtonSolver<T, nalgebra::U1>
    for crate::poisson1dsource::PoissonSourceb<
        '_,
        Mesh1d<T>,
        CscMatrix<T>,
        DVector<T>,
        CscMatrix<T>,
    >
where
    T: RealField,
{
    fn solve_into(&self, mut solution: DVector<T>) -> color_eyre::Result<DVector<T>> {
        let mut residual: DVector<T> = self.source() - self.operator() * &solution;
        let mut iter = 0;
        while residual.norm() > T::from_f64(std::f64::EPSILON).expect("cannot fit into T")
            && iter < 100
        {
            let update = self.factorised_jacobian().solve(&residual);
            println!("{iter}: {:?}", residual.norm());
            solution += &update;
            residual -= self.operator() * &update;
            self.update_jacobian(&solution).unwrap();
            iter += 1;
        }

        Ok(solution)
    }
}
