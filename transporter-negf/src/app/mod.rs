// Copyright 2022 Chris Gubbin
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # App
//! The command line interface.
//!
//! The App module drives the simulation. It hands command-line argument parsing,
//! parses the configuration files and delegates to the numerical methods in other
//! sub-modules.

#![warn(missing_docs)]

mod calculations;
mod configuration;
mod error;
pub(crate) mod styles;
mod telemetry;
pub(crate) mod tracker;

use calculations::coherent_calculation_at_fixed_voltage;
use calculations::incoherent_calculation_at_fixed_voltage;
pub(crate) use configuration::Configuration;
pub(crate) use error::TransporterError;
use telemetry::{get_subscriber, init_subscriber};
pub(crate) use tracker::Tracker;

use crate::{
    device::{info_desk::BuildInfoDesk, reader::Device},
    outer_loop::Potential,
};
use argmin::core::ArgminFloat;
use clap::{ArgEnum, Args, Parser};
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dynamic, Matrix, RealField, VecStorage, U1,
};
use num_traits::{NumCast, ToPrimitive};
use serde::{de::DeserializeOwned, Deserialize};
use std::path::PathBuf;
use transporter_mesher::{Mesh, Mesh1d, Segment1dConnectivity, SmallDim};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct App {
    #[clap(flatten)]
    global_opts: GlobalOpts,
}

#[derive(Debug, Args)]
struct GlobalOpts {
    /// The path to the input structure
    input: Option<PathBuf>,
    /// The output directory
    output: Option<PathBuf>,
    /// The level of logging to display
    #[clap(arg_enum, short, long)]
    log_level: LogLevel,
    /// The calculation type
    #[clap(arg_enum, short, long)]
    calculation: Calculation,
    /// The dimension of the calculation
    #[clap(arg_enum, short, long)]
    dimension: Dimension,
    /// Whether to display in color
    #[clap(long, arg_enum, global = true, default_value_t = Color::Auto)]
    color: Color,
}

#[derive(Clone, Debug, ArgEnum)]
enum Color {
    Always,
    Auto,
    Never,
}

impl Color {
    fn supports_color_on(self, stream: owo_colors::Stream) -> bool {
        match self {
            Color::Always => true,
            Color::Auto => supports_color::on_cached(stream).is_some(),
            Color::Never => false,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
pub(crate) enum LogLevel {
    Trace,
    Info,
    Debug,
    Error,
}

impl std::string::ToString for LogLevel {
    fn to_string(&self) -> String {
        match self {
            LogLevel::Trace => "TRACE".into(),
            LogLevel::Info => "INFO".into(),
            LogLevel::Debug => "DEBUG".into(),
            LogLevel::Error => "ERROR".into(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
pub(crate) enum Calculation {
    Coherent,
    Incoherent,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum Dimension {
    D1,
    D2,
}

use miette::IntoDiagnostic;

/// Top level function to run the application
///
/// This parses the configuration, and the device, identifies the calculation type and
/// attempts to run it to completion.
pub fn run<T>() -> miette::Result<()>
where
    T: ArgminFloat
        + Copy
        + DeserializeOwned
        + NumCast
        + RealField
        + ToPrimitive
        + ndarray::ScalarOperand,
{
    // Prepare terminal environment
    let term = console::Term::stdout();
    term.set_title("Transporter NEGF Solver");
    term.hide_cursor().into_diagnostic()?;
    // .map_err(|e| TransporterError::IoError(e))?;

    // Parse the global app options
    let cli = App::parse();

    // Initiate the tracing subscriber
    let subscriber = get_subscriber(cli.global_opts.log_level);
    init_subscriber(subscriber);

    let __marker: std::marker::PhantomData<T> = std::marker::PhantomData;

    let config: Configuration<T::RealField> = Configuration::build()?;

    let path = cli
        .global_opts
        .input
        .ok_or_else(|| miette::miette!("a valid input path is necessary"))?;

    match cli.global_opts.dimension {
        Dimension::D1 => {
            // Initialise and pretty print device
            tracing::trace!("Initialising device");
            let device: Device<T::RealField, U1> = Device::build(path)?;
            let mut device_display = device.display();
            if cli
                .global_opts
                .color
                .supports_color_on(owo_colors::Stream::Stdout)
            {
                device_display.colorize();
            }
            term.write_line(&format!("{device_display}"))
                .into_diagnostic()?;

            // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
            let info_desk = device.build_device_info_desk()?;
            tracing::trace!("Initialising mesh");
            let voltage_target = device.voltage_offsets[1];
            let mesh: Mesh1d<T::RealField> =
                build_mesh_with_config(&config, device).map_err(|e| miette::miette!("{:?}", e))?;
            tracing::info!("Mesh initialised with {} elements", mesh.elements().len());

            let tracker = tracker::TrackerBuilder::new(cli.global_opts.calculation)
                .with_mesh(&mesh)
                .with_info_desk(&info_desk)
                .build()
                .map_err(|e| miette::miette!("{:?}", e))?;

            let calculation = match cli.global_opts.calculation {
                Calculation::Coherent => CalculationB::Coherent { voltage_target },
                Calculation::Incoherent => CalculationB::Incoherent { voltage_target },
            };

            build_and_run(config, &mesh, &tracker, calculation)
                .map_err(|e| miette::miette!("{:?}", e))?;
        }
        Dimension::D2 => {
            unimplemented!()
        }
    }

    Ok(())
}

pub(crate) fn build_mesh_with_config<T: Copy + DeserializeOwned + RealField + ToPrimitive>(
    config: &Configuration<T>,
    device: Device<T, U1>,
) -> miette::Result<Mesh1d<T>>
where
    DefaultAllocator: Allocator<T, U1>,
    <DefaultAllocator as Allocator<T, U1>>::Buffer: Deserialize<'static>,
{
    // Get configuration stuff from the config, what do we want? Minimum element size, growth rate etc

    let widths: Vec<T> = device
        .iter()
        .map(|layer| layer.thickness.coords[0])
        .collect();

    // TODO refine the mesh

    Ok(
        transporter_mesher::create_line_segment_mesh_1d_from_regions(
            config.mesh.unit_size,
            &widths,
            config.mesh.elements_per_unit,
            &nalgebra::Vector1::new(T::zero()),
        ),
    )
}

enum CalculationB<T: RealField> {
    Coherent { voltage_target: T },
    Incoherent { voltage_target: T },
}

use crate::outer_loop::OuterLoopError;

fn build_and_run<T, BandDim: SmallDim>(
    config: Configuration<T>,
    mesh: &Mesh<T, U1, Segment1dConnectivity>,
    tracker: &Tracker<'_, T, U1, BandDim>,
    calculation_type: CalculationB<T>,
) -> Result<(), TransporterError<T>>
where
    T: ArgminFloat + Copy + num_traits::NumCast + RealField + ndarray::ScalarOperand,
    //Tracker: crate::HamiltonianInfoDesk<T::RealField>,
    DefaultAllocator: Allocator<T, U1>
        + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
    <DefaultAllocator as Allocator<T, U1>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
{
    // Todo allow an initial potential to be read from a file
    let mut initial_potential: Potential<T> = Potential::from_vector(
        nalgebra::DVector::from_element(mesh.num_nodes(), T::zero().real()),
    );
    match calculation_type {
        CalculationB::Coherent { voltage_target } => {
            let mut current_voltage = T::zero();
            let mut voltage_step = config.global.voltage_step;
            while current_voltage <= voltage_target {
                // Do a single calculation
                tracing::info!("Solving for current voltage {current_voltage}V");
                match coherent_calculation_at_fixed_voltage(
                    current_voltage,
                    initial_potential.clone(),
                    &config,
                    mesh,
                    tracker,
                ) {
                    // If it converged proceed
                    Ok(converged_potential) => {
                        let _ = std::mem::replace(&mut initial_potential, converged_potential);
                    }
                    // If there is an error, either return if unrecoverable or reduce the voltage step
                    Err(OuterLoopError::FixedPoint(fixed_point_error)) => match fixed_point_error {
                        conflux::core::FixedPointError::TooManyIterations(_cost) => {
                            current_voltage -= voltage_step;
                            voltage_step /= T::one() + T::one();
                        }
                        _ => {
                            return Err(OuterLoopError::FixedPoint(fixed_point_error).into());
                        }
                    },
                    Err(e) => {
                        return Err(e.into());
                    }
                }
                // increment
                current_voltage += voltage_step;
            }
        }
        CalculationB::Incoherent { voltage_target } => {
            let mut current_voltage = T::zero();
            let mut voltage_step = config.global.voltage_step;
            while current_voltage <= voltage_target {
                // Do a single calculation
                tracing::info!("Solving for current voltage {current_voltage}V");
                match incoherent_calculation_at_fixed_voltage(
                    current_voltage,
                    initial_potential.clone(),
                    &config,
                    mesh,
                    tracker,
                ) {
                    // If it converged proceed
                    Ok(converged_potential) => {
                        let _ = std::mem::replace(&mut initial_potential, converged_potential);
                    }
                    // If there is an error, either return if unrecoverable or reduce the voltage step
                    Err(OuterLoopError::FixedPoint(fixed_point_error)) => match fixed_point_error {
                        conflux::core::FixedPointError::TooManyIterations(_cost) => {
                            current_voltage -= voltage_step;
                            voltage_step /= T::one() + T::one();
                        }
                        _ => {
                            return Err(OuterLoopError::FixedPoint(fixed_point_error).into());
                        }
                    },
                    Err(e) => {
                        return Err(e.into());
                    }
                }
                // increment
                current_voltage += voltage_step;
            }
        }
    }

    Ok(())
}

//fn build_and_run<T, BandDim: SmallDim>(
//    config: Configuration<T>,
//    mesh: &Mesh<T, U1, Segment1dConnectivity>,
//    tracker: &Tracker<'_, T, U1, BandDim>,
//    calculation_type: Calculation,
//    _marker: std::marker::PhantomData<T>,
//) -> Result<(), TransporterError<T>>
//where
//    T: ArgminFloat + Copy + num_traits::NumCast + RealField + ndarray::ScalarOperand,
//    //Tracker: crate::HamiltonianInfoDesk<T::RealField>,
//    DefaultAllocator: Allocator<T, U1>
//        + Allocator<T, BandDim>
//        + Allocator<[T; 3], BandDim>
//        + Allocator<
//            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
//            BandDim,
//        >,
//    <DefaultAllocator as Allocator<T, U1>>::Buffer: Send + Sync,
//    <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
//    <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
//{
//    // Build calculation independent structures
//    let mut hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
//        .with_mesh(mesh)
//        .with_info_desk(tracker)
//        .build()?;
//    let initial_potential = Potential::from_vector(nalgebra::DVector::from_element(
//        mesh.num_nodes(),
//        T::zero().real(),
//    ));
//
//    match calculation_type {
//        Calculation::Coherent => {
//            // If we asked for a coherent calculation ALL the masses need to be equal (probably only along z, check)
//            let first = tracker.info_desk.effective_masses[0].clone();
//            assert!(tracker
//                .info_desk
//                .effective_masses
//                .iter()
//                .all(|item| item.clone() == first));
//
//            let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
//                .with_number_of_energy_points(config.spectral.number_of_energy_points)
//                .with_energy_range(std::ops::Range {
//                    start: config.spectral.minimum_energy,
//                    end: config.spectral.maximum_energy,
//                })
//                .with_energy_integration_method(config.spectral.energy_integration_rule);
//
//            let spectral_space = spectral_space_builder.build_coherent();
//            let outer_config = crate::outer_loop::Convergence {
//                outer_tolerance: config.outer_loop.tolerance,
//                maximum_outer_iterations: config.outer_loop.maximum_iterations,
//                inner_tolerance: config.inner_loop.tolerance,
//                maximum_inner_iterations: config.inner_loop.maximum_iterations,
//                calculation_type: Calculation::Coherent,
//            };
//            let mut outer_loop: crate::outer_loop::OuterLoop<
//                T,
//                U1,
//                Segment1dConnectivity,
//                BandDim,
//                crate::spectral::SpectralSpace<T::RealField, ()>,
//            > = crate::outer_loop::OuterLoopBuilder::new()
//                .with_mesh(mesh)
//                .with_hamiltonian(&mut hamiltonian)
//                .with_spectral_space(&spectral_space)
//                .with_convergence_settings(&outer_config)
//                .with_tracker(tracker)
//                .with_info_desk(tracker.info_desk)
//                .build()
//                .unwrap();
//
//            Ok(outer_loop.run_loop(initial_potential)?)
//        }
//        Calculation::Incoherent => {
//            let first = tracker.info_desk.effective_masses[0].clone();
//            let all_masses_equal = tracker
//                .info_desk
//                .effective_masses
//                .iter()
//                .all(|item| item.clone() == first);
//
//            let coherent_result = if !all_masses_equal {
//                let spectral_space_builder =
//                    crate::spectral::constructors::SpectralSpaceBuilder::new()
//                        .with_number_of_energy_points(config.spectral.number_of_energy_points)
//                        .with_energy_range(std::ops::Range {
//                            start: config.spectral.minimum_energy,
//                            end: config.spectral.maximum_energy,
//                        })
//                        .with_energy_integration_method(config.spectral.energy_integration_rule);
//                let spectral_space = spectral_space_builder.build_coherent();
//                let outer_config = crate::outer_loop::Convergence {
//                    outer_tolerance: config.outer_loop.tolerance,
//                    maximum_outer_iterations: config.outer_loop.maximum_iterations,
//                    inner_tolerance: config.inner_loop.tolerance,
//                    maximum_inner_iterations: config.inner_loop.maximum_iterations,
//                    calculation_type: Calculation::Coherent,
//                };
//                let mut outer_loop: crate::outer_loop::OuterLoop<
//                    T,
//                    U1,
//                    Segment1dConnectivity,
//                    BandDim,
//                    crate::spectral::SpectralSpace<T::RealField, ()>,
//                > = crate::outer_loop::OuterLoopBuilder::new()
//                    .with_mesh(mesh)
//                    .with_hamiltonian(&mut hamiltonian)
//                    .with_spectral_space(&spectral_space)
//                    .with_convergence_settings(&outer_config)
//                    .with_tracker(tracker)
//                    .with_info_desk(tracker.info_desk)
//                    .build_coherent()
//                    .unwrap();
//
//                outer_loop.run_loop(initial_potential)?;
//                outer_loop.potential_owned()
//            // Else run a coherent calculation with a wavevector discretisation
//            } else {
//                dbg!("Worked");
//                let spectral_space_builder =
//                    crate::spectral::constructors::SpectralSpaceBuilder::new()
//                        .with_number_of_energy_points(config.spectral.number_of_energy_points)
//                        .with_energy_range(std::ops::Range {
//                            start: config.spectral.minimum_energy,
//                            end: config.spectral.maximum_energy,
//                        })
//                        .with_energy_integration_method(config.spectral.energy_integration_rule)
//                        .with_maximum_wavevector(config.spectral.maximum_wavevector)
//                        .with_number_of_wavevector_points(
//                            config.spectral.number_of_wavevector_points,
//                        )
//                        .with_wavevector_integration_method(
//                            config.spectral.wavevector_integration_rule,
//                        )
//                        .with_mesh(mesh);
//                let spectral_space = spectral_space_builder.build_incoherent();
//
//                let outer_config = crate::outer_loop::Convergence {
//                    outer_tolerance: config.outer_loop.tolerance,
//                    maximum_outer_iterations: config.outer_loop.maximum_iterations,
//                    inner_tolerance: config.inner_loop.tolerance,
//                    maximum_inner_iterations: config.inner_loop.maximum_iterations,
//                    calculation_type: Calculation::Coherent,
//                };
//                let mut outer_loop: crate::outer_loop::OuterLoop<
//                    T,
//                    U1,
//                    Segment1dConnectivity,
//                    BandDim,
//                    crate::spectral::SpectralSpace<
//                        T::RealField,
//                        crate::spectral::WavevectorSpace<T, U1, Segment1dConnectivity>,
//                    >,
//                > = crate::outer_loop::OuterLoopBuilder::new()
//                    .with_mesh(mesh)
//                    .with_hamiltonian(&mut hamiltonian)
//                    .with_spectral_space(&spectral_space)
//                    .with_convergence_settings(&outer_config)
//                    .with_tracker(tracker)
//                    .with_info_desk(tracker.info_desk)
//                    .build_coherent()
//                    .unwrap();
//                outer_loop.run_loop(initial_potential)?;
//                outer_loop.potential_owned()
//            };
//
//            // Do the incoherent calculation
//
//            let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
//                .with_number_of_energy_points(config.spectral.number_of_energy_points)
//                .with_energy_range(std::ops::Range {
//                    start: config.spectral.minimum_energy,
//                    end: config.spectral.maximum_energy,
//                })
//                .with_energy_integration_method(config.spectral.energy_integration_rule)
//                .with_maximum_wavevector(config.spectral.maximum_wavevector)
//                .with_number_of_wavevector_points(config.spectral.number_of_wavevector_points)
//                .with_wavevector_integration_method(config.spectral.wavevector_integration_rule)
//                .with_mesh(mesh);
//            let spectral_space = spectral_space_builder.build_incoherent();
//
//            let outer_config = crate::outer_loop::Convergence {
//                outer_tolerance: config.outer_loop.tolerance / T::from_f64(100.).unwrap(), // Lowering because the shift may be small -> if the scattering is weak
//                maximum_outer_iterations: config.outer_loop.maximum_iterations,
//                inner_tolerance: config.inner_loop.tolerance,
//                maximum_inner_iterations: config.inner_loop.maximum_iterations,
//                calculation_type: Calculation::Incoherent,
//            };
//            let mut outer_loop: crate::outer_loop::OuterLoop<
//                T,
//                U1,
//                Segment1dConnectivity,
//                BandDim,
//                crate::spectral::SpectralSpace<
//                    T::RealField,
//                    crate::spectral::WavevectorSpace<T, U1, Segment1dConnectivity>,
//                >,
//            > = crate::outer_loop::OuterLoopBuilder::new()
//                .with_mesh(mesh)
//                .with_hamiltonian(&mut hamiltonian)
//                .with_spectral_space(&spectral_space)
//                .with_convergence_settings(&outer_config)
//                .with_tracker(tracker)
//                .with_info_desk(tracker.info_desk)
//                .build()
//                .unwrap();
//            let mut potential = coherent_result;
//            while outer_loop.scattering_scaling() <= T::one() {
//                let mut file = std::fs::File::create(format!(
//                    "../results/converged_potential_{}.txt",
//                    outer_loop.scattering_scaling()
//                ))?;
//
//                for value in potential.as_ref().row_iter() {
//                    let value = value[0].to_f64().unwrap().to_string();
//                    writeln!(file, "{}", value)?;
//                }
//                tracing::info!("Scattering scaled at {}", outer_loop.scattering_scaling());
//                outer_loop.run_loop(potential.clone())?;
//                potential = outer_loop.potential_owned();
//                outer_loop.increment_scattering_scaling();
//            }
//            panic!()
//        }
//    }
//}
//
