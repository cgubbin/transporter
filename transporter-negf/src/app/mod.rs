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
//! parses the configuration files and delegates to the solvers andnumerical methods in other
//! sub-modules.

#![warn(missing_docs)]

/// The terminal user interface, which allows for interactive sessions
pub mod tui;

mod calculations;
mod configuration;
mod error;
pub(crate) mod styles;
mod telemetry;
pub(crate) mod tracker;
pub use configuration::Configuration;
pub(crate) use error::TransporterError;
pub use tracker::{Tracker, TrackerBuilder};

use self::tui::NEGFResult;
use self::tui::Progress;
use crate::{
    device::{info_desk::BuildInfoDesk, reader::Device},
    outer_loop::OuterLoopError,
    outer_loop::Potential,
};
use calculations::{
    coherent_calculation_at_fixed_voltage, incoherent_calculation_at_fixed_voltage,
};
use clap::ArgEnum;
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField, U1};
use ndarray::Array1;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Instant;
use tokio::sync::mpsc::Sender;
use transporter_mesher::{Mesh, Mesh1d, Segment1dConnectivity, SmallDim};

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

// #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
/// The flavour of calculation to run
///
/// A `Coherent` variation runs a calculation with no scattering -> an electron entering from a given contact
/// can leave the system via any contact at the same energy.
/// An `Incoherent` variation includes any scattering defined in the configuration file.
pub enum Calculation<T: RealField> {
    /// A calculation with no scattering.
    Coherent {
        /// The target voltage to ramp to
        voltage_target: T,
    },
    /// A calculation with scattering
    Incoherent {
        /// The target voltage to ramp to
        voltage_target: T,
    },
}

impl<T: Copy + RealField> std::fmt::Display for Calculation<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Calculation::Coherent { voltage_target: x } => {
                write!(f, "coherent calculation to max {:.2}V", x)
            }
            Calculation::Incoherent { voltage_target: x } => {
                write!(f, "incoherent calculation to max {:.2}V", x)
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum Dimension {
    D1,
    D2,
}

/// Top level function to run the application
///
/// This parses the configuration, and the device, identifies the calculation type and
/// attempts to run it to completion.
pub fn run_simulation(
    file_path: PathBuf,
    calculation: Calculation<f64>,
    progress_sender: Sender<Progress<f64>>,
    result_sender: Sender<NEGFResult<f64>>,
) -> miette::Result<()> {
    // Initiate the tracing subscriber

    // let (subscriber, _guard) = get_subscriber(LogLevel::Info);
    // init_subscriber(subscriber);

    let config: Configuration<f64> = Configuration::build()?;

    tracing::trace!("Initialising device");
    let device: Device<f64, U1> = Device::build(file_path)?;
    let info_desk = device.build_device_info_desk()?;
    tracing::trace!("Initialising mesh");
    let mesh: Mesh1d<f64> = build_mesh_with_config(&config, device)
        .map_err(|e| miette::miette!("{:?}", e))
        .unwrap();
    tracing::info!("Mesh initialised with {} elements", mesh.elements().len());

    let tracker = tracker::TrackerBuilder::new(calculation)
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .map_err(|e| miette::miette!("{:?}", e))?;

    let mut progress = Progress::default();
    progress.set_calculation(calculation);

    build_and_run(
        config,
        &mesh,
        &tracker,
        calculation,
        progress,
        progress_sender,
        result_sender,
    )
    .map_err(|e| miette::miette!("{:?}", e))?;
    Ok(())
}

/// Builds the mesh in the device from the parsed configuration file
pub fn build_mesh_with_config(
    config: &Configuration<f64>,
    device: Device<f64, U1>,
) -> miette::Result<Mesh1d<f64>>
where
    DefaultAllocator: Allocator<f64, U1>,
    <DefaultAllocator as Allocator<f64, U1>>::Buffer: Deserialize<'static>,
{
    // Get configuration stuff from the config, what do we want? Minimum element size, growth rate etc

    let widths: Vec<f64> = device
        .iter()
        .map(|layer| layer.thickness.coords[0])
        .collect();

    // TODO refine the mesh

    Ok(
        transporter_mesher::create_line_segment_mesh_1d_from_regions(
            config.mesh.unit_size,
            &widths,
            config.mesh.elements_per_unit,
            &nalgebra::Vector1::new(0_f64),
        ),
    )
}

fn build_and_run<BandDim: SmallDim>(
    config: Configuration<f64>,
    mesh: &Mesh<f64, U1, Segment1dConnectivity>,
    tracker: &Tracker<'_, f64, U1, BandDim>,
    calculation_type: Calculation<f64>,
    mut progress: Progress<f64>,
    progress_sender: Sender<Progress<f64>>,
    result_sender: Sender<NEGFResult<f64>>,
) -> Result<(), TransporterError<f64>>
where
    DefaultAllocator: Allocator<f64, U1>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>
        + Allocator<Array1<f64>, BandDim>,
    <DefaultAllocator as Allocator<f64, U1>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    // Todo allow an initial potential to be read from a file
    let mut initial_potential: Potential<f64> =
        Potential::from_vector(ndarray::Array1::from(vec![0_f64; mesh.num_nodes()]));
    match calculation_type {
        Calculation::Coherent { voltage_target } => {
            let mut current_voltage = 0_f64;
            let mut voltage_step = config.global.voltage_step;
            while current_voltage <= voltage_target {
                progress.set_voltage(current_voltage);
                if let Err(e) = progress_sender.blocking_send(progress.clone()) {
                    tracing::warn!("Failed to send the progress report {:?}", e);
                }
                std::thread::sleep(std::time::Duration::from_secs(2));
                // Do a single calculation
                tracing::info!("Solving for current voltage {current_voltage}V");

                let start = Instant::now();
                match coherent_calculation_at_fixed_voltage(
                    current_voltage,
                    initial_potential.clone(),
                    &config,
                    mesh,
                    tracker,
                    progress.clone(),
                    progress_sender.clone(),
                    result_sender.clone(),
                ) {
                    // If it converged proceed
                    Ok(converged_potential) => {
                        let _ = std::mem::replace(&mut initial_potential, converged_potential);
                    }
                    // If there is an error, either return if unrecoverable or reduce the voltage step
                    Err(OuterLoopError::FixedPoint(fixed_point_error)) => match fixed_point_error {
                        conflux::core::FixedPointError::TooManyIterations(_cost) => {
                            tracing::info!("Too many outer iterations: decreasing voltage step");
                            current_voltage -= voltage_step;
                            voltage_step /= 2_f64;
                        }
                        _ => {
                            return Err(OuterLoopError::FixedPoint(fixed_point_error).into());
                        }
                    },
                    Err(e) => {
                        return Err(e.into());
                    }
                }
                let elapsed = start.elapsed();
                progress.set_time_for_voltage_point(elapsed);
                // increment
                current_voltage += voltage_step;
            }
        }
        Calculation::Incoherent { voltage_target } => {
            let mut current_voltage = 0_f64;
            let mut voltage_step = config.global.voltage_step;
            while current_voltage <= voltage_target {
                progress.set_voltage(current_voltage);
                if let Err(e) = progress_sender.blocking_send(progress.clone()) {
                    tracing::warn!("Failed to send the progress report {:?}", e);
                }
                tracing::info!("Solving for current voltage {current_voltage}V");
                match incoherent_calculation_at_fixed_voltage(
                    current_voltage,
                    initial_potential.clone(),
                    &config,
                    mesh,
                    tracker,
                    progress.clone(),
                    progress_sender.clone(),
                    result_sender.clone(),
                ) {
                    // If it converged proceed
                    Ok(converged_potential) => {
                        let _ = std::mem::replace(&mut initial_potential, converged_potential);
                    }
                    // If there is an error, either return if unrecoverable or reduce the voltage step
                    Err(OuterLoopError::FixedPoint(fixed_point_error)) => match fixed_point_error {
                        conflux::core::FixedPointError::TooManyIterations(_cost) => {
                            tracing::info!("Too many outer iterations: decreasing voltage step");
                            current_voltage -= voltage_step;
                            voltage_step /= 2_f64;
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
