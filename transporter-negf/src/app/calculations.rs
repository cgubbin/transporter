//! # Calculations
//!
//! Delegated functions from `App` to run coherent and incoherent calculations with fixed applied voltage
//!

use super::Progress;
use super::{Calculation, Configuration, Tracker};
use crate::{
    app::tui::NEGFResult,
    outer_loop::{Outer, OuterLoopError, Potential},
    spectral::WavevectorSpace,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, U1};
use ndarray::Array1;
use tokio::sync::mpsc::Sender;
use transporter_mesher::{Mesh, Segment1dConnectivity, SmallDim};

pub(crate) fn coherent_calculation_at_fixed_voltage<BandDim: SmallDim>(
    voltage: f64,
    initial_potential: Potential<f64>,
    config: &Configuration<f64>,
    mesh: &Mesh<f64, U1, Segment1dConnectivity>,
    tracker: &Tracker<'_, f64, U1, BandDim>,
    progress: Progress<f64>,
    progress_sender: Sender<Progress<f64>>,
    result_sender: Sender<NEGFResult<f64>>,
) -> Result<Potential<f64>, OuterLoopError<f64>>
where
    DefaultAllocator: Allocator<f64, U1>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>
        + Allocator<Array1<f64>, BandDim>,
    <DefaultAllocator as Allocator<f64, U1>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    // if all the masses are equal we do not need to discretise wavevectors for a coherent calculation
    tracing::info!("Coherent calculation");
    let first = tracker.info_desk.effective_masses[0].clone();
    match tracker
        .info_desk
        .effective_masses
        .iter()
        .all(|item| item.clone() == first)
    {
        true => coherent_calculation_at_fixed_voltage_with_constant_mass(
            voltage,
            initial_potential,
            config,
            mesh,
            tracker,
            progress,
            progress_sender,
            result_sender,
        ),
        false => coherent_calculation_at_fixed_voltage_with_changing_mass(
            voltage,
            initial_potential,
            config,
            mesh,
            tracker,
            progress,
            progress_sender,
            result_sender,
        ),
    }
}

fn coherent_calculation_at_fixed_voltage_with_constant_mass<BandDim: SmallDim>(
    voltage: f64,
    initial_potential: Potential<f64>,
    config: &Configuration<f64>,
    mesh: &Mesh<f64, U1, Segment1dConnectivity>,
    tracker: &Tracker<'_, f64, U1, BandDim>,
    progress: Progress<f64>,
    progress_sender: Sender<Progress<f64>>,
    result_sender: Sender<NEGFResult<f64>>,
) -> Result<Potential<f64>, OuterLoopError<f64>>
where
    DefaultAllocator: Allocator<f64, U1>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>
        + Allocator<Array1<f64>, BandDim>,
    <DefaultAllocator as Allocator<f64, U1>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    // Build calculation independent structures
    let mut hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
        .with_mesh(mesh)
        .with_info_desk(tracker)
        .build()?;

    // Get the eigenvalues which sit in the range of energies studied
    let eigenvalues: Array1<f64> = hamiltonian
        .eigenvalues(0_f64, initial_potential.as_ref())?
        .into_iter()
        .map(|x| x.re)
        .filter(|&x| (x > config.spectral.minimum_energy) & (x < config.spectral.maximum_energy))
        .collect::<Array1<_>>();

    // Best to maintain a uniform density of energies in the grid to get smooth convergence over a sweep
    let number_of_energy_points = config.spectral.number_of_energy_points
        * (((config.spectral.maximum_energy + voltage - config.spectral.minimum_energy)
            / (config.spectral.maximum_energy - config.spectral.minimum_energy))
            as usize);

    let spectral_space_builder = crate::spectral::SpectralSpaceBuilder::default()
        .with_number_of_energy_points(number_of_energy_points)
        .with_energy_range(std::ops::Range {
            start: config.spectral.minimum_energy,
            end: config.spectral.maximum_energy + voltage,
        })
        .with_energy_integration_method(config.spectral.energy_integration_rule)
        .with_zone_centre_eigenvalues(&eigenvalues);

    let spectral_space = spectral_space_builder.build_coherent();
    let outer_config = crate::outer_loop::Convergence {
        outer_tolerance: config.outer_loop.tolerance,
        maximum_outer_iterations: config.outer_loop.maximum_iterations,
        inner_tolerance: config.inner_loop.tolerance,
        maximum_inner_iterations: config.inner_loop.maximum_iterations,
        security_checks: config.global.security_checks,
        calculation_type: Calculation::Coherent {
            voltage_target: 0_f64,
        },
    };
    let mut outer_loop: crate::outer_loop::OuterLoop<
        f64,
        U1,
        Segment1dConnectivity,
        BandDim,
        crate::spectral::SpectralSpace<f64, ()>,
    > = crate::outer_loop::OuterLoopBuilder::new()
        .with_mesh(mesh)
        .with_hamiltonian(&mut hamiltonian)
        .with_spectral_space(&spectral_space)
        .with_convergence_settings(&outer_config)
        .with_tracker(tracker)
        .with_info_desk(tracker.info_desk)
        .with_progress(&progress)
        .with_sender(&progress_sender)
        .build(voltage)
        .unwrap();

    outer_loop.run_loop(initial_potential)?;

    let result = crate::app::tui::NEGFResult {
        calculation: Calculation::Coherent {
            voltage_target: voltage,
        },
        current: outer_loop.tracker.current_as_ref().net_current()[0],
        potential: outer_loop.tracker.potential_as_ref().clone(),
        electron_density: outer_loop.tracker.charge_as_ref().clone().net_charge(),
        scattering_rates: None,
    };
    if let Err(err) = result_sender.blocking_send(result) {
        tracing::warn!("Failed to communicate result at {voltage}V: {:?}", err);
    }
    Ok(outer_loop.potential_owned())
}

fn coherent_calculation_at_fixed_voltage_with_changing_mass<BandDim: SmallDim>(
    voltage: f64,
    initial_potential: Potential<f64>,
    config: &Configuration<f64>,
    mesh: &Mesh<f64, U1, Segment1dConnectivity>,
    tracker: &Tracker<'_, f64, U1, BandDim>,
    progress: Progress<f64>,
    progress_sender: Sender<Progress<f64>>,
    result_sender: Sender<NEGFResult<f64>>,
) -> Result<Potential<f64>, OuterLoopError<f64>>
where
    DefaultAllocator: Allocator<f64, U1>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>
        + Allocator<Array1<f64>, BandDim>,
    <DefaultAllocator as Allocator<f64, U1>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    // Build calculation independent structures
    let mut hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
        .with_mesh(mesh)
        .with_info_desk(tracker)
        .build()?;

    let spectral_space_builder = crate::spectral::SpectralSpaceBuilder::default()
        .with_mesh(mesh)
        .with_number_of_energy_points(config.spectral.number_of_energy_points)
        .with_energy_range(std::ops::Range {
            start: config.spectral.minimum_energy,
            end: config.spectral.maximum_energy + voltage,
        })
        .with_energy_integration_method(config.spectral.energy_integration_rule)
        .with_maximum_wavevector(config.spectral.maximum_wavevector)
        .with_number_of_wavevector_points(config.spectral.number_of_wavevector_points)
        .with_wavevector_integration_method(config.spectral.wavevector_integration_rule);

    let spectral_space = spectral_space_builder.build_incoherent();

    let outer_config = crate::outer_loop::Convergence {
        outer_tolerance: config.outer_loop.tolerance,
        maximum_outer_iterations: config.outer_loop.maximum_iterations,
        inner_tolerance: config.inner_loop.tolerance,
        maximum_inner_iterations: config.inner_loop.maximum_iterations,
        security_checks: config.global.security_checks,
        calculation_type: Calculation::Coherent {
            voltage_target: 0_f64,
        },
    };
    let mut outer_loop: crate::outer_loop::OuterLoop<
        f64,
        U1,
        Segment1dConnectivity,
        BandDim,
        crate::spectral::SpectralSpace<f64, WavevectorSpace<f64, U1, Segment1dConnectivity>>,
    > = crate::outer_loop::OuterLoopBuilder::new()
        .with_mesh(mesh)
        .with_hamiltonian(&mut hamiltonian)
        .with_spectral_space(&spectral_space)
        .with_convergence_settings(&outer_config)
        .with_tracker(tracker)
        .with_info_desk(tracker.info_desk)
        .with_progress(&progress)
        .with_sender(&progress_sender)
        .build(voltage)
        .unwrap();

    outer_loop.run_loop(initial_potential)?;

    let result = crate::app::tui::NEGFResult {
        calculation: Calculation::Coherent {
            voltage_target: voltage,
        },
        current: outer_loop.tracker.current_as_ref().net_current()[0],
        potential: outer_loop.tracker.potential_as_ref().clone(),
        electron_density: outer_loop.tracker.charge_as_ref().clone().net_charge(),
        scattering_rates: None,
    };
    if let Err(err) = result_sender.blocking_send(result) {
        tracing::warn!("Failed to communicate result at {voltage}V: {:?}", err);
    }
    Ok(outer_loop.potential_owned())
}

pub(crate) fn incoherent_calculation_at_fixed_voltage<BandDim: SmallDim>(
    voltage: f64,
    initial_potential: Potential<f64>,
    config: &Configuration<f64>,
    mesh: &Mesh<f64, U1, Segment1dConnectivity>,
    tracker: &Tracker<'_, f64, U1, BandDim>,
    mut progress: Progress<f64>,
    progress_sender: Sender<Progress<f64>>,
    result_sender: Sender<NEGFResult<f64>>,
) -> Result<Potential<f64>, OuterLoopError<f64>>
where
    DefaultAllocator: Allocator<f64, U1>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>
        + Allocator<Array1<f64>, BandDim>,
    <DefaultAllocator as Allocator<f64, U1>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    // if all the masses are equal we do not need to discretise wavevectors for a coherent calculation
    let first = tracker.info_desk.effective_masses[0].clone();
    // do an initial coherent calculation
    tracing::info!("Initial coherent calculation");
    let mut potential = match tracker
        .info_desk
        .effective_masses
        .iter()
        .all(|item| item.clone() == first)
    {
        true => coherent_calculation_at_fixed_voltage_with_constant_mass(
            voltage,
            initial_potential,
            config,
            mesh,
            tracker,
            progress.clone(),
            progress_sender.clone(),
            result_sender.clone(),
        ),
        false => coherent_calculation_at_fixed_voltage_with_changing_mass(
            voltage,
            initial_potential,
            config,
            mesh,
            tracker,
            progress.clone(),
            progress_sender.clone(),
            result_sender.clone(),
        ),
    }?;
    // do an incoherent calculation ramping the scattering from 0 to 1

    tracing::info!("Incoherent calculation");
    // Build calculation independent structures
    let mut hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
        .with_mesh(mesh)
        .with_info_desk(tracker)
        .build()?;

    let spectral_space_builder = crate::spectral::SpectralSpaceBuilder::default()
        .with_mesh(mesh)
        .with_number_of_energy_points(config.spectral.number_of_energy_points)
        .with_energy_range(std::ops::Range {
            start: config.spectral.minimum_energy,
            end: config.spectral.maximum_energy,
        })
        .with_energy_integration_method(config.spectral.energy_integration_rule)
        .with_maximum_wavevector(config.spectral.maximum_wavevector)
        .with_number_of_wavevector_points(config.spectral.number_of_wavevector_points)
        .with_wavevector_integration_method(config.spectral.wavevector_integration_rule);

    let spectral_space: crate::spectral::SpectralSpace<
        f64,
        WavevectorSpace<f64, U1, Segment1dConnectivity>,
    > = spectral_space_builder.build_incoherent();

    let outer_config = crate::outer_loop::Convergence {
        outer_tolerance: config.outer_loop.tolerance / 100_f64, // Lowering because the shift may be small -> if the scattering is weak
        maximum_outer_iterations: config.outer_loop.maximum_iterations,
        inner_tolerance: config.inner_loop.tolerance,
        maximum_inner_iterations: config.inner_loop.maximum_iterations,
        security_checks: config.global.security_checks,
        calculation_type: Calculation::Incoherent {
            voltage_target: 0_f64,
        },
    };
    // Clone to avoid lifetime issues
    let outer_progress = progress.clone();
    let mut outer_loop: crate::outer_loop::OuterLoop<
        f64,
        U1,
        Segment1dConnectivity,
        BandDim,
        crate::spectral::SpectralSpace<
            f64,
            crate::spectral::WavevectorSpace<f64, U1, Segment1dConnectivity>,
        >,
    > = crate::outer_loop::OuterLoopBuilder::new()
        .with_mesh(mesh)
        .with_hamiltonian(&mut hamiltonian)
        .with_spectral_space(&spectral_space)
        .with_convergence_settings(&outer_config)
        .with_tracker(tracker)
        .with_info_desk(tracker.info_desk)
        .with_progress(&outer_progress)
        .with_sender(&progress_sender)
        .build(voltage)
        .unwrap();
    while outer_loop.scattering_scaling() <= 1_f64 {
        // let mut file = std::fs::File::create(format!(
        //     "../results/converged_potential_{}.txt",
        //     outer_loop.scattering_scaling()
        // ))?;

        // for value in potential.as_ref().row_iter() {
        //     let value = value[0].to_f64().unwrap().to_string();
        //     writeln!(file, "{}", value)?;
        // }
        progress.set_scattering_scale_factor(outer_loop.scattering_scaling());
        if let Err(err) = progress_sender.blocking_send(progress.clone()) {
            tracing::warn!("Failed to update the progress status: {:?}", err);
        }
        tracing::info!("Scattering scaled at {}", outer_loop.scattering_scaling());
        outer_loop.run_loop(potential.clone())?;
        potential = outer_loop.potential_owned();
        outer_loop.increment_scattering_scaling();
    }

    // Arrange the results of the calculation and communicate them to the master process
    let result = crate::app::tui::NEGFResult {
        calculation: Calculation::Incoherent {
            voltage_target: voltage,
        },
        current: outer_loop.tracker.current_as_ref().net_current()[0],
        potential: outer_loop.tracker.potential_as_ref().clone(),
        electron_density: outer_loop.tracker.charge_as_ref().clone().net_charge(),
        scattering_rates: None,
    };
    if let Err(err) = result_sender.blocking_send(result) {
        tracing::warn!("Failed to communicate result at {voltage}V: {:?}", err);
    }

    Ok(outer_loop.potential_owned())
}
