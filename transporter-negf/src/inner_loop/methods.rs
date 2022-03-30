use super::InnerLoop;
use crate::greens_functions::GreensFunctionMethods;
use crate::postprocessor::{ChargeAndCurrent, PostProcess, PostProcessor, PostProcessorBuilder};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, RealField};
use transporter_mesher::{Connectivity, SmallDim};

pub(crate) trait Inner<T>
where
    T: RealField,
{
    /// Compute the updated charge and current densities, and confirm
    /// whether the change is within tolerance of the values on the
    /// previous loop iteration
    fn is_loop_converged(
        &self,
        charge_and_current: &mut ChargeAndCurrent<T>,
    ) -> color_eyre::Result<bool>;
    /// Carry out a single iteration of the self-consistent inner loop
    fn single_iteration(&mut self) -> color_eyre::Result<()>;
    /// Run the self-consistent inner loop to convergence
    fn run_loop(&mut self, charge_and_current: &mut ChargeAndCurrent<T>) -> color_eyre::Result<()>;
}

impl<'a, T, GeometryDim, Conn, Matrix> Inner<T::RealField>
    for InnerLoop<'a, T, GeometryDim, Conn, Matrix>
where
    T: Copy + ComplexField,
    GeometryDim: SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    <T as ComplexField>::RealField: Copy,
    Matrix: GreensFunctionMethods<T>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    /// Check convergence and re-assign the new charge density to the old one
    fn is_loop_converged(
        &self,
        previous_charge_and_current: &mut ChargeAndCurrent<T::RealField>,
    ) -> color_eyre::Result<bool> {
        let postprocessor: PostProcessor<T, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge_and_current: ChargeAndCurrent<T::RealField> =
            postprocessor.recompute_currents_and_densities(self.greens_functions, self.spectral)?;
        let result = charge_and_current.is_change_within_tolerance(
            previous_charge_and_current,
            self.convergence_settings.inner_tolerance(),
        );
        let _ = std::mem::replace(previous_charge_and_current, charge_and_current);
        result
    }

    fn single_iteration(&mut self) -> color_eyre::Result<()> {
        // TODO Recompute se, check it's ok, recompute green's functions
        //self.greens_functions
        //    .update_greens_functions(self.hamiltonian)?;
        Ok(())
    }

    fn run_loop(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<T::RealField>,
    ) -> color_eyre::Result<()> {
        let mut iteration = 0;
        while !self.is_loop_converged(previous_charge_and_current)? {
            self.single_iteration()?;
            iteration += 1;
            if iteration >= self.convergence_settings.maximum_inner_iterations() {
                return Err(color_eyre::eyre::eyre!(
                    "Reached maximum iteration count in the inner loop"
                ));
            }
        }
        Ok(())
    }
}
