use super::{InnerLoop, InnerLoopError};
use crate::{
    greens_functions::MMatrix,
    postprocessor::{ChargeAndCurrent, PostProcess, PostProcessor, PostProcessorBuilder},
    spectral::{SpectralSpace, WavevectorSpace},
};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use ndarray::{Array1, Array2};
use num_complex::Complex;
use num_traits::ToPrimitive;
use sprs::CsMat;
use std::io::Write;
use std::time::Instant;

use transporter_mesher::{Connectivity, SmallDim};

pub(crate) trait Inner<T, BandDim>
where
    T: RealField,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<Array1<T>, BandDim>,
{
    /// Compute the updated charge and current densities, and confirm
    /// whether the change is within tolerance of the values on the
    /// previous loop iteration
    /// This takes a mutable reference to self to enable update of the progress struct
    fn is_loop_converged(
        &mut self,
        charge_and_current: &mut ChargeAndCurrent<T, BandDim>,
    ) -> Result<bool, InnerLoopError>;
    /// Carry out a single iteration of the self-consistent inner loop
    fn single_iteration(&mut self) -> Result<(), InnerLoopError>;
    /// Run the self-consistent inner loop to convergence
    fn run_loop(
        &mut self,
        charge_and_current: &mut ChargeAndCurrent<T, BandDim>,
    ) -> Result<(), InnerLoopError>;
}

impl<'a, GeometryDim, Conn, BandDim> Inner<f64, BandDim>
    for InnerLoop<'a, f64, GeometryDim, Conn, CsMat<Complex<f64>>, SpectralSpace<f64, ()>, BandDim>
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<Array1<f64>, BandDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    /// Check convergence and re-assign the new charge density to the old one
    fn is_loop_converged(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<f64, BandDim>,
    ) -> Result<bool, InnerLoopError> {
        let postprocessor: PostProcessor<f64, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge_and_current: ChargeAndCurrent<f64, BandDim> = postprocessor
            .recompute_currents_and_densities(
                self.voltage,
                self.greens_functions,
                self.self_energies,
                self.spectral,
            )
            .unwrap();
        let (result, _) = charge_and_current.is_change_within_tolerance(
            previous_charge_and_current,
            self.convergence_settings.inner_tolerance(),
        )?;
        let _ = std::mem::replace(previous_charge_and_current, charge_and_current);
        Ok(result)
    }

    fn single_iteration(&mut self) -> Result<(), InnerLoopError> {
        // TODO Recompute se, check it's ok, recompute green's functions
        self.self_energies.recalculate_contact_self_energy(
            self.mesh,
            self.hamiltonian,
            self.spectral,
        )?;
        self.greens_functions.update_greens_functions(
            self.voltage,
            self.hamiltonian,
            self.self_energies,
            self.spectral,
        )?;
        Ok(())
    }

    #[tracing::instrument("Inner loop", skip_all)]
    fn run_loop(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<f64, BandDim>,
    ) -> Result<(), InnerLoopError> {
        // In a coherent calculation there is no inner loop
        // self.term.move_cursor_to(0, 5)?;
        // self.term.clear_to_end_of_screen()?;
        tracing::info!("Recalculating electron density");
        self.single_iteration()?;
        // Run the convergence check, this is solely to update the charge and current in the tracker
        // as we do not track convergence in a coherent calculation
        let _ = self.is_loop_converged(previous_charge_and_current)?;
        Ok(())
    }
}

impl<'a, GeometryDim, Conn, BandDim> Inner<f64, BandDim>
    for InnerLoop<
        'a,
        f64,
        GeometryDim,
        Conn,
        CsMat<Complex<f64>>,
        SpectralSpace<f64, WavevectorSpace<f64, GeometryDim, Conn>>,
        BandDim,
    >
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    // MatrixType: GreensFunctionMethods<T>,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<Array1<f64>, BandDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    /// Check convergence and re-assign the new charge density to the old one
    fn is_loop_converged(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<f64, BandDim>,
    ) -> Result<bool, InnerLoopError> {
        let postprocessor: PostProcessor<f64, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge_and_current: ChargeAndCurrent<f64, BandDim> = postprocessor
            .recompute_currents_and_densities(
                self.voltage,
                self.greens_functions,
                self.self_energies,
                self.spectral,
            )?;

        let (result, _) = charge_and_current.is_change_within_tolerance(
            previous_charge_and_current,
            self.convergence_settings.inner_tolerance(),
        )?;
        let _ = std::mem::replace(previous_charge_and_current, charge_and_current);
        Ok(result)
    }

    fn single_iteration(&mut self) -> Result<(), InnerLoopError> {
        // TODO Recompute se, check it's ok, recompute green's functions
        self.self_energies.recalculate_contact_self_energy(
            self.mesh,
            self.hamiltonian,
            self.spectral,
        )?;
        self.greens_functions.update_greens_functions(
            self.voltage,
            self.hamiltonian,
            self.self_energies,
            self.spectral,
        )?;
        Ok(())
    }

    fn run_loop(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<f64, BandDim>,
    ) -> Result<(), InnerLoopError> {
        tracing::info!("Recalculating electron density");
        self.single_iteration()?;
        // Run the convergence check, this is solely to update the charge and current in the tracker
        // as we do not track convergence in a coherent calculation
        let _ = self.is_loop_converged(previous_charge_and_current)?;
        Ok(())
    }
}

/// Incoherent scattering impl with a full wavevector space
impl<'a, GeometryDim, Conn, BandDim> Inner<f64, BandDim>
    for InnerLoop<
        'a,
        f64,
        GeometryDim,
        Conn,
        Array2<Complex<f64>>,
        SpectralSpace<f64, WavevectorSpace<f64, GeometryDim, Conn>>,
        BandDim,
    >
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<Array1<f64>, BandDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    /// Check convergence and re-assign the new charge density to the old one
    fn is_loop_converged(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<f64, BandDim>,
    ) -> Result<bool, InnerLoopError> {
        let postprocessor: PostProcessor<f64, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge_and_current: ChargeAndCurrent<f64, BandDim> = postprocessor
            .recompute_currents_and_densities(
                self.voltage,
                self.greens_functions,
                self.self_energies,
                self.spectral,
            )?;
        let (result, current_value) = charge_and_current.is_change_within_tolerance(
            previous_charge_and_current,
            self.convergence_settings.inner_tolerance(),
        )?;
        self.progress.set_inner_residual(current_value);

        let _ = std::mem::replace(previous_charge_and_current, charge_and_current);
        Ok(result)
    }

    fn single_iteration(&mut self) -> Result<(), InnerLoopError> {
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
            self.voltage,
            self.hamiltonian,
            self.self_energies,
            self.spectral,
        )?;

        let postprocessor: PostProcessor<f64, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let _charge = postprocessor
            .recompute_currents_and_densities(
                self.voltage,
                self.greens_functions,
                self.self_energies,
                self.spectral,
            )?
            .charge_as_ref()
            .net_charge();
        Ok(())
    }

    fn run_loop(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<f64, BandDim>,
    ) -> Result<(), InnerLoopError> {
        //}
        // TODO Only on the first full iteration
        tracing::info!("Beginning loop");
        let now = Instant::now();
        self.single_iteration()?;

        self.progress.set_inner_iteration(1);
        self.progress
            .set_target_inner_residual(self.convergence_settings.inner_tolerance());
        self.progress.set_time_for_inner_iteration(now.elapsed());

        let mut iteration = 0;
        while !self.is_loop_converged(previous_charge_and_current)? {
            // Communicate progress to the main thread
            if let Err(e) = self.mpsc_sender.blocking_send(self.progress.clone()) {
                tracing::warn!("Failed to emit progress from the Inner Loop: {:?}", e);
            }

            tracing::info!("Inner loop at iteration {}", iteration + 2);

            let now = Instant::now();

            self.single_iteration()?;

            // Set the iteration and the inner iteration time
            self.progress.set_inner_iteration(iteration + 2);
            self.progress.set_time_for_inner_iteration(now.elapsed());

            iteration += 1;
            if iteration >= self.convergence_settings.maximum_inner_iterations() {
                return Err(InnerLoopError::OutOfIterations);
            }
        }
        Ok(())
    }
}

impl<'a, GeometryDim, Conn, BandDim>
    InnerLoop<
        'a,
        f64,
        GeometryDim,
        Conn,
        Array2<Complex<f64>>,
        SpectralSpace<f64, WavevectorSpace<f64, GeometryDim, Conn>>,
        BandDim,
    >
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<Array1<f64>, BandDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    fn coherent_step(&mut self) -> Result<(), InnerLoopError> {
        dbg!("Initial coherent step");
        // TODO Recompute se, check it's ok, recompute green's functions
        self.self_energies.recalculate_contact_self_energy(
            self.mesh,
            self.hamiltonian,
            self.spectral,
        )?;
        self.greens_functions.update_greens_functions(
            self.voltage,
            self.hamiltonian,
            self.self_energies,
            self.spectral,
        )?;

        let postprocessor: PostProcessor<f64, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge = postprocessor
            .recompute_currents_and_densities(
                self.voltage,
                self.greens_functions,
                self.self_energies,
                self.spectral,
            )?
            .charge_as_ref()
            .net_charge();

        let mut file = std::fs::File::create(&"../results/ramp_charge_0.txt")?;
        for value in charge.iter() {
            let value = value.to_f64().unwrap().to_string();
            writeln!(file, "{}", value)?;
        }
        Ok(())
    }

    fn ramp_scattering(&mut self) -> Result<(), InnerLoopError> {
        tracing::info!("Ramping scattering to physical value");
        let n_steps = 100;
        // Testing
        let postprocessor: PostProcessor<f64, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();

        for index in 1..=n_steps {
            let scaling = index as f64 / n_steps as f64;
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
                self.voltage,
                self.hamiltonian,
                self.self_energies,
                self.spectral,
            )?;

            let charge = postprocessor
                .recompute_currents_and_densities(
                    self.voltage,
                    self.greens_functions,
                    self.self_energies,
                    self.spectral,
                )?
                .charge_as_ref()
                .net_charge();

            let mut file = std::fs::File::create(format!("../results/ramp_charge_{}.txt", index))?;
            for value in charge.iter() {
                let value = value.to_f64().unwrap().to_string();
                writeln!(file, "{}", value)?;
            }
        }
        Ok(())
    }
}

// Mixed impl
/// Incoherent scattering impl with a full wavevector space
impl<'a, GeometryDim, Conn, BandDim> Inner<f64, BandDim>
    for InnerLoop<
        'a,
        f64,
        GeometryDim,
        Conn,
        MMatrix<Complex<f64>>,
        SpectralSpace<f64, WavevectorSpace<f64, GeometryDim, Conn>>,
        BandDim,
    >
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<Array1<f64>, BandDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    /// Check convergence and re-assign the new charge density to the old one
    fn is_loop_converged(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<f64, BandDim>,
    ) -> Result<bool, InnerLoopError> {
        let postprocessor: PostProcessor<f64, GeometryDim, Conn> =
            PostProcessorBuilder::new().with_mesh(self.mesh).build();
        let charge_and_current: ChargeAndCurrent<f64, BandDim> = postprocessor
            .recompute_currents_and_densities(
                self.voltage,
                self.greens_functions,
                self.self_energies,
                self.spectral,
            )?;

        let (result, value) = charge_and_current.is_change_within_tolerance(
            previous_charge_and_current,
            self.convergence_settings.inner_tolerance(),
        )?;
        self.progress.set_inner_residual(value);
        let _ = std::mem::replace(previous_charge_and_current, charge_and_current);
        Ok(result)
    }

    fn single_iteration(&mut self) -> Result<(), InnerLoopError> {
        // TODO Recompute se, check it's ok, recompute green's functions
        self.self_energies.recalculate_contact_self_energy(
            self.mesh,
            self.hamiltonian,
            self.spectral,
        )?;
        self.self_energies
            .recalculate_localised_lo_lesser_self_energy_mixed(
                self.scattering_scaling,
                self.mesh,
                self.spectral,
                self.greens_functions,
            )?;
        self.self_energies
            .recalculate_localised_lo_retarded_self_energy_mixed(
                self.scattering_scaling,
                self.mesh,
                self.spectral,
                self.greens_functions,
            )?;

        self.greens_functions.update_greens_functions(
            self.voltage,
            self.hamiltonian,
            self.self_energies,
            self.spectral,
        )?;

        Ok(())
    }

    fn run_loop(
        &mut self,
        previous_charge_and_current: &mut ChargeAndCurrent<f64, BandDim>,
    ) -> Result<(), InnerLoopError> {
        // self.term.move_cursor_to(0, 6)?;
        // self.term.clear_to_end_of_screen()?;
        tracing::info!("Inner loop at iteration 1");
        let now = Instant::now();
        self.single_iteration()?;

        self.progress.set_inner_iteration(1);
        self.progress
            .set_target_inner_residual(self.convergence_settings.inner_tolerance());
        self.progress.set_time_for_inner_iteration(now.elapsed());

        let mut iteration = 0;
        // Run to iteration == 2 because on the first iteration incoherent
        // self energies will be trivially zero as the Greens functions are uninitialised
        while !self.is_loop_converged(previous_charge_and_current)? | (iteration < 2) {
            // Communicate progress to the main thread
            if let Err(e) = self.mpsc_sender.blocking_send(self.progress.clone()) {
                tracing::warn!("Failed to emit progress from the Inner Loop: {:?}", e);
            }

            tracing::info!("Inner loop at iteration {}", iteration + 2);

            let now = Instant::now();

            // Do the numerical calculations
            self.single_iteration()?;

            // Communicate the progress of the simulation to the `Master` thread
            self.progress.set_inner_iteration(iteration + 2);
            self.progress.set_time_for_inner_iteration(now.elapsed());

            iteration += 1;
            if iteration >= self.convergence_settings.maximum_inner_iterations() {
                return Err(InnerLoopError::OutOfIterations);
            }
        }

        self.rate = Some(self.self_energies.calculate_localised_lo_scattering_rate(
            self.spectral,
            self.mesh,
            self.greens_functions,
        )?);

        Ok(())
    }
}

impl<'a, GeometryDim, Conn, BandDim>
    InnerLoop<
        'a,
        f64,
        GeometryDim,
        Conn,
        MMatrix<Complex<f64>>,
        SpectralSpace<f64, WavevectorSpace<f64, GeometryDim, Conn>>,
        BandDim,
    >
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<Array1<f64>, BandDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    pub(crate) fn rate(&self) -> Result<Array1<f64>, InnerLoopError> {
        let rate = self
            .self_energies
            .calculate_resolved_localised_lo_emission_rate(
                self.spectral,
                self.mesh,
                self.greens_functions,
            )?;
        Ok(rate)
    }
}
