use crate::PoissonMethods;
use nalgebra::{allocator::Allocator, DVector, DefaultAllocator, RealField};
use std::ops::AddAssign;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

pub trait NewtonSolver<'a, T>
where
    T: RealField,
{
    fn solve_into(
        &'a self,
        solution: DVector<T>,
        charge_density: &DVector<T>,
        fermi_level: &DVector<T>,
    ) -> color_eyre::Result<DVector<T>>;
}

use nalgebra_sparse::CscMatrix;

impl<'a, T, GeometryDim, Conn, InfoDesk> NewtonSolver<'a, T>
    for crate::poisson1dsource::PoissonSourceb<
        'a,
        InfoDesk,
        Mesh<T, GeometryDim, Conn>,
        CscMatrix<T>,
        DVector<T>,
    >
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    InfoDesk: PoissonMethods<T, GeometryDim, Conn>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn solve_into(
        &'a self,
        mut solution: DVector<T>,
        charge_density: &DVector<T>,
        fermi_level: &DVector<T>,
    ) -> color_eyre::Result<DVector<T>> {
        let mut iter = 0;
        let mut jacobian_diagonal: DVector<T> = self.source().clone() * T::zero();
        let mut charge_density = charge_density.clone();
        // F =  n - N_D  + A (\phi_0 + \phi_m)
        // A = - d/ dz (\epsilon d/ dz)
        let mut residual: DVector<T> = self.operator() * &solution + &charge_density;
        dbg!(&charge_density / T::from_f64(1.6e-19).unwrap());
        // while residual.norm() > T::from_f64(std::f64::EPSILON).expect("cannot fit into T")
        //     && iter < 60
        loop {
            // Update the diagonal of the Jacobian
            self.update_jacobian_diagonal(
                self.mesh,
                fermi_level,
                &solution,
                &mut jacobian_diagonal,
            )?;
            // Update the charge density vector
            self.update_charge_density(self.mesh, fermi_level, &solution, &mut charge_density)?;

            let dense_jac = self.factorised_jacobian(jacobian_diagonal.clone())?;
            let factor = dense_jac.svd(true, true);
            let mut update: DVector<T> = -factor.solve(&residual, T::zero()).unwrap();
            // Enforce a Dirichlet condition on the source contact
            let offset = update[0];
            for ele in update.iter_mut() {
                *ele -= offset;
            }

            // let update = self
            //     .factorised_jacobian(jacobian_diagonal.clone())?
            //     .solve(&residual);

            //dbg!(&old_charge_density, &charge_density, &residual, &update);
            solution += &update / T::from_f64(10.).unwrap();

            if update.norm() < T::from_f64(1e-13).expect("cannot fit into T") {
                break;
            }
            //residual -= self.operator() * &update + old_charge_density - &charge_density;
            residual = self.operator() * &solution + &charge_density;
            iter += 1;
        }
        dbg!(charge_density / T::from_f64(1.6e-19).unwrap());

        dbg!(residual.norm(), &solution);
        std::thread::sleep(std::time::Duration::from_secs(5));

        Ok(solution)
    }
}
