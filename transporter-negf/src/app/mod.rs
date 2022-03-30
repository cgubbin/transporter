/// This module governs the high-level implementation of the simulation
mod configuration;
mod tracker;
use crate::device::{
    info_desk::{BuildInfoDesk, DeviceInfoDesk},
    reader::Device,
};
//use crate::hamiltonian::HamiltonianConstructor;
use clap::{ArgEnum, Parser};
use color_eyre::eyre::eyre;
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField, U1, U2};
use num_traits::ToPrimitive;
use serde::{de::DeserializeOwned, Deserialize};
use std::path::PathBuf;
use transporter_mesher::{Connectivity, Mesh, Mesh1d, SmallDim};

use configuration::Configuration;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct App {
    file_path: Option<PathBuf>,
    #[clap(arg_enum, short, long)]
    log_level: LogLevel,
    #[clap(arg_enum, short, long)]
    calculation: Calculation,
    #[clap(arg_enum, short, long)]
    dimension: Dimension,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum LogLevel {
    Trace,
    Info,
    Debug,
    Error,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum Calculation {
    Coherent,
    Incoherent,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum Dimension {
    D1,
    D2,
}

pub fn run<T: Copy + DeserializeOwned + RealField + ToPrimitive>() -> color_eyre::Result<()> {
    let cli = App::parse();

    println!("calculation: {:?}", cli.calculation);
    println!("log_level: {:?}", cli.log_level);
    println!("path: {:?}", cli.file_path);
    println!("dimension: {:?}", cli.dimension);

    let config: Configuration<T> = Configuration::build()?;

    let path = cli
        .file_path
        .ok_or(eyre!("A file path needs to be passed."))?;

    match cli.dimension {
        Dimension::D1 => {
            let device: Device<T, U1> = Device::build(path)?;
            // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
            let info_desk = device.build_device_info_desk()?;
            let mesh: Mesh1d<T> = build_mesh_with_config(&config, device)?;
            let tracker = tracker::TrackerBuilder::new()
                .with_mesh(&mesh)
                .with_info_desk(&info_desk)
                .build();
            run_internal(config, &mesh, &tracker, cli.calculation)?;
        }
        Dimension::D2 => {
            unimplemented!()
        }
    }

    Ok(())
}

fn build_mesh_with_config<T: Copy + DeserializeOwned + RealField + ToPrimitive>(
    config: &Configuration<T>,
    device: Device<T, U1>,
) -> color_eyre::Result<Mesh1d<T>>
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

use tracker::Tracker;
fn run_internal<T: Copy + RealField, Conn, Tracker>(
    config: Configuration<T>,
    mesh: &Mesh<T, Tracker::GeometryDim, Conn>,
    tracker: &Tracker,
    calculation_type: Calculation,
) -> color_eyre::Result<()>
where
    Conn: Connectivity<T, Tracker::GeometryDim>,
    Tracker: crate::HamiltonianInfoDesk<T>,
    DefaultAllocator: Allocator<T, Tracker::GeometryDim>
        + Allocator<T, Tracker::BandDim>
        + Allocator<[T; 3], Tracker::BandDim>,
{
    let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
        .with_mesh(mesh)
        .with_info_desk(tracker)
        .build()?;

    let tmp = hamiltonian.calculate_total(T::one());

    dbg!(tmp);

    //let spectral_discretisation = crate::spectral::constructors::SpectralSpaceBuilder::new()
    //    .with_number_of_energy_points(todo!()) //config.global.number_of_energy_points)
    //    .with_energy_range(todo!()) //config.global.energy_range)
    //    .with_energy_integration_method(todo!()); //config.global.energy_integration_rule);

    //let spectral: impl crate::spectral::SpectralDiscretisation<T> = match calculation_type {
    //    Calculation::Coherent => spectral_discretisation.build(),
    //    Calculation::Incoherent => {
    //        let spectral_discretisation = spectral_discretisation
    //            .with_number_of_wavevector_points(todo!()) //config.global.number_of_energy_points)
    //            .with_wavevector_range(todo!()) //config.global.energy_range)
    //            .with_wavevector_integration_method(todo!()); //config.global.energy_integration_rule);todo!()
    //        spectral_discretisation.build();
    //    }
    //};
    Ok(())
}
