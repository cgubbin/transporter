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
pub use configuration::Configuration;
pub(crate) use error::TransporterError;
use telemetry::{get_subscriber, init_subscriber};
pub use tracker::{Tracker, TrackerBuilder};

use crate::{
    device::{info_desk::BuildInfoDesk, reader::Device},
    outer_loop::Potential,
};
use argmin::core::ArgminFloat;
use clap::{ArgEnum, Args, Parser};
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dynamic, Matrix, RealField, VecStorage, U1,
};
use num_complex::Complex;
use num_traits::{NumCast, ToPrimitive};
use serde::{de::DeserializeOwned, Deserialize};
use std::path::PathBuf;
use transporter_mesher::{Mesh, Mesh1d, Segment1dConnectivity, SmallDim};

#[cfg(feature = "ndarray")]
pub trait NEGFFloat: ArgminFloat + Copy + NumCast + RealField + ndarray::ScalarOperand {}

#[cfg(not(feature = "ndarray"))]
pub trait NEGFFloat: ArgminFloat + Copy + NumCast + RealField {}

impl NEGFFloat for f32 {}
impl NEGFFloat for f64 {}

#[cfg(feature = "ndarray")]
pub trait NEGFComplex: Copy + nalgebra::ComplexField + ndarray::ScalarOperand {}

#[cfg(not(feature = "ndarray"))]
pub trait NEGFComplex: Copy + nalgebra::ComplexField {}

#[cfg(feature = "ndarray")]
impl<T> NEGFComplex for Complex<T>
where
    T: NEGFFloat,
    Complex<T>: ndarray::ScalarOperand,
{
}

#[cfg(not(feature = "ndarray"))]
impl<T> NEGFComplex for Complex<T> where T: NEGFFloat {}

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
pub enum Calculation {
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
    T: NEGFFloat + DeserializeOwned + ToPrimitive, // + ndarray::ScalarOperand,
    Complex<T>: NEGFComplex,
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

    let __marker: std::marker::PhantomData<T> = std::marker::PhantomData;

    let config: Configuration<T::RealField> = Configuration::build()?;

    let path = cli
        .global_opts
        .input
        .ok_or_else(|| miette::miette!("a valid input path is necessary"))?;

    match cli.global_opts.dimension {
        Dimension::D1 => {
            // Initialise and pretty print device
            term.move_cursor_to(0, 0).into_diagnostic()?;
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
            term.move_cursor_to(0, 0).into_diagnostic()?;
            term.write_line(&format!("{device_display}"))
                .into_diagnostic()?;

            // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
            let info_desk = device.build_device_info_desk()?;

            term.move_cursor_to(0, 0).into_diagnostic()?;
            term.clear_screen().into_diagnostic()?;
            tracing::trace!("Initialising mesh");

            let voltage_target = device.voltage_offsets[1];
            let mesh: Mesh1d<T::RealField> =
                build_mesh_with_config(&config, device).map_err(|e| miette::miette!("{:?}", e))?;

            term.move_cursor_to(0, 0).into_diagnostic()?;
            term.clear_to_end_of_screen().into_diagnostic()?;
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

            build_and_run(config, &mesh, &tracker, calculation, term)
                .map_err(|e| miette::miette!("{:?}", e))?;
        }
        Dimension::D2 => {
            unimplemented!()
        }
    }

    Ok(())
}

pub fn build_mesh_with_config<T: Copy + DeserializeOwned + RealField + ToPrimitive>(
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
    term: console::Term,
) -> Result<(), TransporterError<T>>
where
    T: NEGFFloat, // + ndarray::ScalarOperand,
    Complex<T>: NEGFComplex,
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
