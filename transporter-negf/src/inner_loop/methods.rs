use super::InnerLoop;
use crate::greens_functions::GreensFunctionMethods;
use crate::postprocessor::{ChargeAndCurrent, PostProcess, PostProcessor, PostProcessorBuilder};
use crate::spectral::{SpectralSpace, WavevectorSpace};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, RealField};
use nalgebra::{Const, Dynamic, Matrix, VecStorage};
use nalgebra_sparse::CsrMatrix;
use transporter_mesher::{Connectivity, SmallDim};

pub(crate) trait Inner<T, BandDim>
where
    T: RealField,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<
        Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
        BandDim,
    >,
{
    /// Compute the updated charge and current densities, and confirm
    /// whether the change is within tolerance of the values on the
    /// previous loop iteration
    fn is_loop_converged(
        &self,
        charge_and_current: &mut ChargeAndCurrent<T, BandDim>,
    ) -> color_eyre::Result<bool>;
    /// Carry out a single iteration of the self-consistent inner loop
    fn single_iteration(&mut self) -> color_eyre::Result<()>;
    /// Run the self-consistent inner loop to convergence
    fn run_loop(
        &mut self,
        charge_and_current: &mut ChargeAndCurrent<T, BandDim>,
    ) -> color_eyre::Result<()>;
}

impl<'a, T, GeometryDim, Conn, BandDim> Inner<T::RealField, BandDim>
    for InnerLoop<'a, T, GeometryDim, Conn, CsrMatrix<T>, SpectralSpace<T::RealField, ()>>
where
    T: Copy + ComplexField,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    <T as ComplexField>::RealField: Copy,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>
        + Allocator<
            Matrix<
                T::RealField,
                Dynamic,
                Const<1_usize>,
                VecStorage<T::RealField, Dynamic, Const<1_usize>>,
            >,
            BandDim,
        >,
{
    /// Check convergence and re-assign the new charge density to the old one
    fn is_loop_converged(
        &self,
        previous_charge_and_current: &mut ChargeAndCurrent<T::RealField, BandDim>,
    ) -> color_eyre::Result<bool> {
        let postprocessor: PostProcessor<T, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge_and_current: ChargeAndCurrent<T::RealField, BandDim> =
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
        self.self_energies.recalculate()?;
        self.greens_functions.update_greens_functions(
            self.hamiltonian,
            self.self_energies,
            self.spectral,
        )?;
        Ok(())
    }

    fn run_loop(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<T::RealField, BandDim>,
    ) -> color_eyre::Result<()> {
        // In a coherent calculation there is no inner loop
        self.single_iteration()
    }
}

impl<'a, T, GeometryDim, Conn, BandDim, MatrixType> Inner<T::RealField, BandDim>
    for InnerLoop<
        'a,
        T,
        GeometryDim,
        Conn,
        MatrixType,
        SpectralSpace<T::RealField, WavevectorSpace<T::RealField, GeometryDim, Conn>>,
    >
where
    T: Copy + ComplexField,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    <T as ComplexField>::RealField: Copy,
    MatrixType: GreensFunctionMethods<T>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>
        + Allocator<
            Matrix<
                T::RealField,
                Dynamic,
                Const<1_usize>,
                VecStorage<T::RealField, Dynamic, Const<1_usize>>,
            >,
            BandDim,
        >,
{
    /// Check convergence and re-assign the new charge density to the old one
    fn is_loop_converged(
        &self,
        previous_charge_and_current: &mut ChargeAndCurrent<T::RealField, BandDim>,
    ) -> color_eyre::Result<bool> {
        //let postprocessor: PostProcessor<T, GeometryDim, Conn> =
        //    PostProcessorBuilder::new().with_mesh(self.mesh).build();
        //let charge_and_current: ChargeAndCurrent<T::RealField, BandDim> =
        //    postprocessor.recompute_currents_and_densities(self.greens_functions, self.spectral)?;
        //let result = charge_and_current.is_change_within_tolerance(
        //    previous_charge_and_current,
        //    self.convergence_settings.inner_tolerance(),
        //);
        //let _ = std::mem::replace(previous_charge_and_current, charge_and_current);
        //result
        Ok(false)
    }

    fn single_iteration(&mut self) -> color_eyre::Result<()> {
        // TODO Recompute se, check it's ok, recompute green's functions
        //self.greens_functions
        //    .update_greens_functions(self.hamiltonian)?;
        Ok(())
    }

    fn run_loop(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<T::RealField, BandDim>,
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
