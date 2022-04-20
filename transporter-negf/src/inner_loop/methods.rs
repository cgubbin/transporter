use super::InnerLoop;
use crate::{
    postprocessor::{ChargeAndCurrent, PostProcess, PostProcessor, PostProcessorBuilder},
    spectral::{SpectralSpace, WavevectorSpace},
};
use nalgebra::{
    allocator::Allocator, Const, DMatrix, DefaultAllocator, Dynamic, Matrix, RealField, VecStorage,
};
use nalgebra_sparse::CsrMatrix;
use num_complex::Complex;
use std::io::Write;
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

/// Incoherent scattering impl with a full wavevector space
impl<'a, T, GeometryDim, Conn, BandDim> Inner<T, BandDim>
    for InnerLoop<
        'a,
        T,
        GeometryDim,
        Conn,
        DMatrix<Complex<T>>,
        SpectralSpace<T, WavevectorSpace<T, GeometryDim, Conn>>,
        BandDim,
    >
where
    T: Copy + RealField + argmin::core::ArgminFloat,
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
        // TODO Recompute se, check it's ok, recompute green's functions
        tracing::trace!("Inner loop with scaling {}", self.scattering_scaling);
        self.self_energies.recalculate_contact_self_energy(
            self.mesh,
            self.hamiltonian,
            self.spectral,
        )?;
        self.self_energies
            .recalculate_localised_lo_lesser_self_energy(
                self.scattering_scaling,
                self.mesh,
                self.spectral,
                self.greens_functions,
            )?;
        self.self_energies
            .recalculate_localised_lo_retarded_self_energy(
                self.scattering_scaling,
                self.mesh,
                self.spectral,
                self.greens_functions,
            )?;

        self.greens_functions.update_greens_functions(
            self.hamiltonian,
            self.self_energies,
            self.spectral,
        )?;

        let postprocessor: PostProcessor<T, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge = postprocessor
            .recompute_currents_and_densities(
                self.greens_functions,
                self.self_energies,
                self.spectral,
            )?
            .charge_as_ref()
            .net_charge();

        let system_time = std::time::SystemTime::now();
        let datetime: chrono::DateTime<chrono::Utc> = system_time.into();
        let mut file = std::fs::File::create(format!(
            "../results/inner_charge_{}_{}.txt",
            self.scattering_scaling, datetime
        ))?;
        for value in charge.row_iter() {
            let value = value[0].to_f64().unwrap().to_string();
            writeln!(file, "{}", value)?;
        }
        Ok(())
    }

    fn run_loop(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<T, BandDim>,
    ) -> color_eyre::Result<()> {
        //}
        // TODO Only on the first full iteration
        // self.coherent_step()?;
        // self.ramp_scattering()?;
        tracing::info!("Beginning loop");
        self.single_iteration()?;

        let mut iteration = 0;
        while !self.is_loop_converged(previous_charge_and_current)? {
            tracing::info!("The inner loop at iteration {iteration}");
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

/// Incoherent scattering impl with a full wavevector space
impl<'a, T, GeometryDim, Conn, BandDim>
    InnerLoop<
        'a,
        T,
        GeometryDim,
        Conn,
        DMatrix<Complex<T>>,
        SpectralSpace<T, WavevectorSpace<T, GeometryDim, Conn>>,
        BandDim,
    >
where
    T: Copy + RealField + argmin::core::ArgminFloat,
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
    fn coherent_step(&mut self) -> color_eyre::Result<()> {
        dbg!("Initial coherent step");
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

        let postprocessor: PostProcessor<T, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge = postprocessor
            .recompute_currents_and_densities(
                self.greens_functions,
                self.self_energies,
                self.spectral,
            )?
            .charge_as_ref()
            .net_charge();

        let mut file = std::fs::File::create(&"../results/ramp_charge_0.txt")?;
        for value in charge.row_iter() {
            let value = value[0].to_f64().unwrap().to_string();
            writeln!(file, "{}", value)?;
        }
        Ok(())
    }

    fn ramp_scattering(&mut self) -> color_eyre::Result<()> {
        tracing::info!("Ramping scattering to physical value");
        let n_steps = 100;
        // Testing
        let postprocessor: PostProcessor<T, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();

        for index in 1..=n_steps {
            let scaling = T::from_usize(index).unwrap() / T::from_usize(n_steps).unwrap();
            tracing::info!("Scattering Ramp: {}", scaling);
            self.self_energies
                .recalculate_localised_lo_lesser_self_energy(
                    scaling,
                    self.mesh,
                    self.spectral,
                    self.greens_functions,
                )?;
            self.self_energies
                .recalculate_localised_lo_retarded_self_energy(
                    scaling,
                    self.mesh,
                    self.spectral,
                    self.greens_functions,
                )?;
            self.greens_functions.update_greens_functions(
                self.hamiltonian,
                self.self_energies,
                self.spectral,
            )?;

            let charge = postprocessor
                .recompute_currents_and_densities(
                    self.greens_functions,
                    self.self_energies,
                    self.spectral,
                )?
                .charge_as_ref()
                .net_charge();

            let mut file = std::fs::File::create(format!("../results/ramp_charge_{}.txt", index))?;
            for value in charge.row_iter() {
                let value = value[0].to_f64().unwrap().to_string();
                writeln!(file, "{}", value)?;
            }
        }
        Ok(())
    }
}
