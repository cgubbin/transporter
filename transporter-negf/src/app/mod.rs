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

mod calculations;
mod configuration;
mod error;
pub(crate) mod styles;
mod telemetry;
pub(crate) mod tracker;
pub use configuration::Configuration;
pub(crate) use error::TransporterError;
pub use tracker::{Tracker, TrackerBuilder};

use crate::{
    device::{info_desk::BuildInfoDesk, reader::Device},
    outer_loop::OuterLoopError,
    outer_loop::Potential,
};
use calculations::{
    coherent_calculation_at_fixed_voltage, incoherent_calculation_at_fixed_voltage,
};
use clap::{ArgEnum, Args, Parser};
use miette::IntoDiagnostic;
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField, U1};
use ndarray::Array1;
use serde::Deserialize;
use std::path::PathBuf;
use telemetry::{get_subscriber, init_subscriber};
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
    calculation: CalculationDeserialize,
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
/// The flavour of calculation to run, local variant with no voltage attached for deserialization
enum CalculationDeserialize {
    Coherent,
    Incoherent,
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum Dimension {
    D1,
    D2,
}

/// Top level function to run the application
///
/// This parses the configuration, and the device, identifies the calculation type and
/// attempts to run it to completion.
pub fn run() -> miette::Result<()>
where
{
    // Prepare terminal environment
    let term = console::Term::stdout();
    term.set_title("Transporter NEGF Solver");
    term.hide_cursor().into_diagnostic()?;
    // .map_err(|e| TransporterError::IoError(e))?;

    // Parse the global app options
    let cli = App::parse();

    // Initiate the tracing subscriber
    let (subscriber, _guard) = get_subscriber(cli.global_opts.log_level);
    init_subscriber(subscriber);

    let __marker: std::marker::PhantomData<f64> = std::marker::PhantomData;

    let config: Configuration<f64> = Configuration::build()?;

    let path = cli
        .global_opts
        .input
        .ok_or_else(|| miette::miette!("a valid input path is necessary"))?;

    match cli.global_opts.dimension {
        Dimension::D1 => {
            // Initialise and pretty print device
            term.move_cursor_to(0, 0).into_diagnostic()?;
            tracing::trace!("Initialising device");
            let device: Device<f64, U1> = Device::build(path)?;
            let mut device_display = device.display();
            if cli
                .global_opts
                .color
                .supports_color_on(owo_colors::Stream::Stdout)
            {
                device_display.colorize();
            }
            term.move_cursor_to(0, 0).into_diagnostic()?;
            term.write_line(&format!("{device_display}"))
                .into_diagnostic()?;

            // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
            let info_desk = device.build_device_info_desk()?;

            term.move_cursor_to(0, 0).into_diagnostic()?;
            term.clear_screen().into_diagnostic()?;
            tracing::trace!("Initialising mesh");

            let voltage_target = device.voltage_offsets[1];
            let mesh: Mesh1d<f64> =
                build_mesh_with_config(&config, device).map_err(|e| miette::miette!("{:?}", e))?;

            term.move_cursor_to(0, 0).into_diagnostic()?;
            term.clear_to_end_of_screen().into_diagnostic()?;

            tracing::info!("Mesh initialised with {} elements", mesh.elements().len());

            let calculation = match cli.global_opts.calculation {
                CalculationDeserialize::Coherent => Calculation::Coherent { voltage_target },
                CalculationDeserialize::Incoherent => Calculation::Incoherent { voltage_target },
            };

            let tracker = tracker::TrackerBuilder::new(calculation)
                .with_mesh(&mesh)
                .with_info_desk(&info_desk)
                .build()
                .map_err(|e| miette::miette!("{:?}", e))?;

            build_and_run(config, &mesh, &tracker, calculation, term)
                .map_err(|e| miette::miette!("{:?}", e))?;
        }
        Dimension::D2 => {
            unimplemented!()
        }
    }

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
    term: console::Term,
) -> Result<(), TransporterError<f64>>
where
    //Tracker: crate::HamiltonianInfoDesk<T::RealField>,
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
                // Do a single calculation
                term.move_cursor_to(0, 0)?;
                term.clear_to_end_of_screen()?;
                tracing::info!("Solving for current voltage {current_voltage}V");
                match coherent_calculation_at_fixed_voltage(
                    current_voltage,
                    initial_potential.clone(),
                    &config,
                    mesh,
                    tracker,
                    &term,
                ) {
                    // If it converged proceed
                    Ok(converged_potential) => {
                        let _ = std::mem::replace(&mut initial_potential, converged_potential);
                    }
                    // If there is an error, either return if unrecoverable or reduce the voltage step
                    Err(OuterLoopError::FixedPoint(fixed_point_error)) => match fixed_point_error {
                        conflux::core::FixedPointError::TooManyIterations(_cost) => {
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
        Calculation::Incoherent { voltage_target } => {
            let mut current_voltage = 0_f64;
            let mut voltage_step = config.global.voltage_step;
            while current_voltage <= voltage_target {
                // Do a single calculation
                term.move_cursor_to(0, 0)?;
                term.clear_to_end_of_screen()?;
                tracing::info!("Solving for current voltage {current_voltage}V");
                match incoherent_calculation_at_fixed_voltage(
                    current_voltage,
                    initial_potential.clone(),
                    &config,
                    mesh,
                    tracker,
                    &term,
                ) {
                    // If it converged proceed
                    Ok(converged_potential) => {
                        let _ = std::mem::replace(&mut initial_potential, converged_potential);
                    }
                    // If there is an error, either return if unrecoverable or reduce the voltage step
                    Err(OuterLoopError::FixedPoint(fixed_point_error)) => match fixed_point_error {
                        conflux::core::FixedPointError::TooManyIterations(_cost) => {
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
