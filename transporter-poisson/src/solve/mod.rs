use crate::PoissonMethods;
use nalgebra::{DVector, RealField};
use transporter_mesher::{Mesh1d, SmallDim};

pub trait NewtonSolver<'a, T, GeometryDim>
where
    T: RealField,
    GeometryDim: SmallDim,
{
    fn solve_into(
        &'a self,
        solution: DVector<T>,
        charge_density: DVector<T>,
    ) -> color_eyre::Result<DVector<T>>;
}

use nalgebra_sparse::CscMatrix;

impl<'a, T, InfoDesk> NewtonSolver<'a, T, nalgebra::U1>
    for crate::poisson1dsource::PoissonSourceb<'a, InfoDesk, Mesh1d<T>, CscMatrix<T>, DVector<T>>
where
    T: Copy + RealField,
    InfoDesk: PoissonMethods<T>,
{
    fn solve_into(
        &'a self,
        mut solution: DVector<T>,
        charge_density: DVector<T>,
    ) -> color_eyre::Result<DVector<T>> {
        let mut iter = 0;
        let mut jacobian_diagonal: DVector<T> = self.source().clone() * T::zero();
        let mut charge_density = charge_density;
        let mut residual: DVector<T> =
            &charge_density + self.source() - self.operator() * &solution;
        while residual.norm() > T::from_f64(std::f64::EPSILON).expect("cannot fit into T")
            && iter < 100
        {
            // Update the diagonal of the Jacobian
            self.update_jacobian_diagonal(&solution, &mut jacobian_diagonal)?;
            // Update the charge density vector
            let old_charge_density = charge_density.clone();
            self.update_charge_density(&solution, &mut charge_density)?;

            let update = self
                .factorised_jacobian(jacobian_diagonal.clone())?
                .solve(&residual);

            println!("{iter}: {:?}", residual.norm());
            solution += &update;
            residual -= self.operator() * &update + old_charge_density - &charge_density;
            iter += 1;
        }

        Ok(solution)
    }
}
