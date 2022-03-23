use super::InnerLoop;
use crate::greens_functions::{AggregateGreensFunctions, GreensFunctionMethods};
use crate::postprocessor::{
    Charge, ChargeAndCurrent, PostProcess, PostProcessor, PostProcessorBuilder,
};
use crate::spectral::{BallisticSpectral, ScatteringSpectral};
use crate::Hamiltonian;
use nalgebra::{ComplexField, DVector, RealField};

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
    fn run_loop(&mut self, charge_and_current: ChargeAndCurrent<T>) -> color_eyre::Result<()>;
}

impl<'a, T, Mesh, SelfEnergies, Matrix> Inner<T::RealField>
    for InnerLoop<
        'a,
        T,
        Mesh,
        ScatteringSpectral<T::RealField>,
        Hamiltonian<T::RealField>,
        AggregateGreensFunctions<'a, Matrix, T>,
        SelfEnergies,
    >
where
    T: Copy + ComplexField,
    <T as ComplexField>::RealField: Copy,
    Matrix: GreensFunctionMethods<T>,
{
    /// Check convergence and re-assign the new charge density to the old one
    fn is_loop_converged(
        &self,
        previous_charge_and_current: &mut ChargeAndCurrent<T::RealField>,
    ) -> color_eyre::Result<bool> {
        let postprocessor: PostProcessor<T::RealField, Mesh> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge_and_current: ChargeAndCurrent<T::RealField> =
            postprocessor.recompute_currents_and_densities()?;
        let result = charge_and_current
            .is_change_within_tolerance(previous_charge_and_current, self.tolerance);
        let _ = std::mem::replace(previous_charge_and_current, charge_and_current);
        result
    }

    fn single_iteration(&mut self) -> color_eyre::Result<()> {
        // TODO Recompute se, check it's ok, recompute green's functions
        self.greens_functions
            .update_greens_functions(self.hamiltonian)?;
        Ok(())
    }

    fn run_loop(
        &mut self,
        mut previous_charge_and_current: ChargeAndCurrent<T::RealField>,
    ) -> color_eyre::Result<()> {
        let mut iteration = 0;
        while !self.is_loop_converged(&mut previous_charge_and_current)? {
            self.single_iteration()?;
            iteration += 1;
            if iteration >= self.maximum_iterations {
                return Err(color_eyre::eyre::eyre!(
                    "Reached maximum iteration count in the inner loop"
                ));
            }
        }
        Ok(())
    }
}

//impl<'a, T, Mesh> Inner<T> for InnerLoop<'a, T, Mesh, BallisticSpectral<T>>
//where
//    T: Copy + RealField,
//{
//    /// There is no inner loop for the ballistic transport case, this function just recomputes the currents and charge
//    fn is_loop_converged(&self, _: &mut ChargeAndCurrent<T>) -> color_eyre::Result<bool> {
//        unreachable!();
//    }
//
//    /// The single iteration is exactly the same as for the Incoherent case
//    fn single_iteration(&self) -> color_eyre::Result<()> {
//        // TODO Recompute se, check it's ok, recompute green's functions
//        Ok(())
//    }
//
//    /// For ballistic transport there is no inner loop, so we just return the result of a single iteration
//    fn run_loop(&self) -> color_eyre::Result<Charge<T>> {
//        self.single_iteration()?;
//        // Compute the charge here, in the incoherent case this is done automatically in teh convergence calculation
//        let postprocessor = PostProcessorBuilder::new().with_mesh(self.mesh).build();
//        Ok(postprocessor
//            .recompute_currents_and_densities()?
//            .deref_charge())
//    }
//}
//
