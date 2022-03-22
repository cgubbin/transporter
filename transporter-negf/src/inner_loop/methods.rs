use super::InnerLoop;
use crate::postprocessor::{ChargeAndCurrent, PostProcess, PostProcessorBuilder};
use nalgebra::RealField;

pub(crate) trait Inner<T> {
    /// Compute the updated charge and current densities, and confirm
    /// whether the change is within tolerance of the values on the
    /// previous loop iteration
    fn is_loop_converged(
        &self,
        charge_and_current: &mut ChargeAndCurrent<T>,
    ) -> color_eyre::Result<bool>;
    /// Carry out a single iteration of the self-consistent inner loop
    fn single_iteration(&self) -> color_eyre::Result<()>;
    /// Run the self-consistent inner loop to convergence
    fn run_loop(&self) -> color_eyre::Result<()>;
}

impl<'a, T, Mesh> Inner<T> for InnerLoop<'a, T, Mesh>
where
    T: Copy + RealField,
{
    fn is_loop_converged(
        &self,
        previous_charge_and_current: &mut ChargeAndCurrent<T>,
    ) -> color_eyre::Result<bool> {
        let postprocessor = PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge_and_current = postprocessor.recompute_currents_and_densities()?;
        charge_and_current.is_change_within_tolerance(previous_charge_and_current, self.tolerance)
    }

    fn single_iteration(&self) -> color_eyre::Result<()> {
        // TODO Recompute se, check it's ok, recompute green's functions
        Ok(())
    }

    fn run_loop(&self) -> color_eyre::Result<()> {
        todo!()
    }
}
