use super::InnerLoop;
use crate::{
    greens_functions::GreensFunctionMethods,
    postprocessor::{ChargeAndCurrent, PostProcess, PostProcessor, PostProcessorBuilder},
    spectral::{SpectralSpace, WavevectorSpace},
};
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dynamic, Matrix, RealField, VecStorage,
};
use nalgebra_sparse::CsrMatrix;
use num_complex::Complex;
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
    for InnerLoop<'a, T, GeometryDim, Conn, CsrMatrix<Complex<T>>, SpectralSpace<T, ()>, BandDim>
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim> + Send + Sync,
    <Conn as Connectivity<T, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
{
    /// Check convergence and re-assign the new charge density to the old one
    fn is_loop_converged(
        &self,
        previous_charge_and_current: &mut ChargeAndCurrent<T, BandDim>,
    ) -> color_eyre::Result<bool> {
        let postprocessor: PostProcessor<T, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge_and_current: ChargeAndCurrent<T::RealField, BandDim> = postprocessor
            .recompute_currents_and_densities(
                self.greens_functions,
                self.self_energies,
                self.spectral,
            )?;
        let result = charge_and_current.is_change_within_tolerance(
            previous_charge_and_current,
            self.convergence_settings.inner_tolerance(),
        );
        let _ = std::mem::replace(previous_charge_and_current, charge_and_current);
        result
    }

    fn single_iteration(&mut self) -> color_eyre::Result<()> {
        // TODO Recompute se, check it's ok, recompute green's functions
        self.self_energies.recalculate_contact_self_energy(
            self.mesh,
            self.hamiltonian,
            self.spectral,
        )?;
        self.greens_functions.update_greens_functions(
            self.hamiltonian,
            self.self_energies,
            self.spectral,
        )?;
        Ok(())
    }

    #[tracing::instrument("Inner loop", skip_all)]
    fn run_loop(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<T, BandDim>,
    ) -> color_eyre::Result<()> {
        // In a coherent calculation there is no inner loop
        tracing::info!("Recalculating electron density");
        self.single_iteration()?;
        // Run the convergence check, this is solely to update the charge and current in the tracker
        // as we do not track convergence in a coherent calculation
        let _ = self.is_loop_converged(previous_charge_and_current)?;
        Ok(())
    }
}

/// Coherent scattering impl but with a full wavevector space: the impl for when the effective mass varies
impl<'a, T, GeometryDim, Conn, BandDim> Inner<T, BandDim>
    for InnerLoop<
        'a,
        T,
        GeometryDim,
        Conn,
        CsrMatrix<Complex<T>>,
        SpectralSpace<T, WavevectorSpace<T, GeometryDim, Conn>>,
        BandDim,
    >
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim> + Send + Sync,
    <Conn as Connectivity<T, GeometryDim>>::Element: Send + Sync,
    // MatrixType: GreensFunctionMethods<T>,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
{
    /// Check convergence and re-assign the new charge density to the old one
    fn is_loop_converged(
        &self,
        previous_charge_and_current: &mut ChargeAndCurrent<T, BandDim>,
    ) -> color_eyre::Result<bool> {
        let postprocessor: PostProcessor<T, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge_and_current: ChargeAndCurrent<T::RealField, BandDim> = postprocessor
            .recompute_currents_and_densities(
                self.greens_functions,
                self.self_energies,
                self.spectral,
            )?;
        let result = charge_and_current.is_change_within_tolerance(
            previous_charge_and_current,
            self.convergence_settings.inner_tolerance(),
        );
        let _ = std::mem::replace(previous_charge_and_current, charge_and_current);
        result
    }

    fn single_iteration(&mut self) -> color_eyre::Result<()> {
        dbg!("The inner loop");
        // TODO Recompute se, check it's ok, recompute green's functions
        self.self_energies.recalculate_contact_self_energy(
            self.mesh,
            self.hamiltonian,
            self.spectral,
        )?;
        self.greens_functions.update_greens_functions(
            self.hamiltonian,
            self.self_energies,
            self.spectral,
        )?;
        Ok(())
    }

    fn run_loop(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<T, BandDim>,
    ) -> color_eyre::Result<()> {
        //let mut iteration = 0;
        //while !self.is_loop_converged(previous_charge_and_current)? {
        //    self.single_iteration()?;
        //    iteration += 1;
        //    if iteration >= self.convergence_settings.maximum_inner_iterations() {
        //        return Err(color_eyre::eyre::eyre!(
        //            "Reached maximum iteration count in the inner loop"
        //        ));
        //    }
        //}
        tracing::info!("Recalculating electron density");
        self.single_iteration()?;
        // Run the convergence check, this is solely to update the charge and current in the tracker
        // as we do not track convergence in a coherent calculation
        let _ = self.is_loop_converged(previous_charge_and_current)?;
        Ok(())
    }
}
