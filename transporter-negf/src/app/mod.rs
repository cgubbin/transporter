/// This module governs the high-level implementation of the simulation
mod configuration;
mod tracker;
pub(crate) use configuration::Configuration;
pub(crate) use tracker::{Tracker, TrackerBuilder};

use crate::{
    device::{info_desk::BuildInfoDesk, reader::Device},
    outer_loop::{Outer, Potential},
};
use clap::{ArgEnum, Parser};
use color_eyre::eyre::eyre;
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dynamic, Matrix, RealField, VecStorage, U1,
};
use num_traits::{NumCast, ToPrimitive};
use serde::{de::DeserializeOwned, Deserialize};
use std::path::PathBuf;
use transporter_mesher::{Connectivity, Mesh, Mesh1d, SmallDim};

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
pub(crate) enum Calculation {
    Coherent,
    Incoherent,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum Dimension {
    D1,
    D2,
}

pub fn run<T>() -> color_eyre::Result<()>
where
    T: Copy + DeserializeOwned + NumCast + RealField + ToPrimitive,
{
    let cli = App::parse();

    let __marker: std::marker::PhantomData<T> = std::marker::PhantomData;

    println!("calculation: {:?}", cli.calculation);
    println!("log_level: {:?}", cli.log_level);
    println!("path: {:?}", cli.file_path);
    println!("dimension: {:?}", cli.dimension);

    let config: Configuration<T::RealField> = Configuration::build()?;

    let path = cli
        .file_path
        .ok_or(eyre!("A file path needs to be passed."))?;

    match cli.dimension {
        Dimension::D1 => {
            let device: Device<T::RealField, U1> = Device::build(path)?;
            // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
            let info_desk = device.build_device_info_desk()?;
            let mesh: Mesh1d<T::RealField> = build_mesh_with_config(&config, device)?;
            let tracker = tracker::TrackerBuilder::new()
                .with_mesh(&mesh)
                .with_info_desk(&info_desk)
                .build()?;

            build_and_run(config, &mesh, &tracker, cli.calculation, __marker)?;
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

fn build_and_run<T, GeometryDim: SmallDim, Conn, BandDim: SmallDim>(
    config: Configuration<T>,
    mesh: &Mesh<T, GeometryDim, Conn>,
    tracker: &Tracker<'_, T, GeometryDim, BandDim, Conn>,
    _calculation_type: Calculation,
    _marker: std::marker::PhantomData<T>,
) -> color_eyre::Result<()>
where
    T: Copy + num_traits::NumCast + RealField,
    Conn: Connectivity<T, GeometryDim>,
    //Tracker: crate::HamiltonianInfoDesk<T::RealField>,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
        .with_mesh(mesh)
        .with_info_desk(tracker)
        .build()?;

    // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
    let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
        .with_number_of_energy_points(config.spectral.number_of_energy_points)
        .with_energy_range(std::ops::Range {
            start: config.spectral.minimum_energy,
            end: config.spectral.maximum_energy,
        })
        .with_energy_integration_method(config.spectral.energy_integration_rule);

    let spectral_space = spectral_space_builder.build_coherent();

    let outer_config = crate::outer_loop::Convergence {
        outer_tolerance: config.outer_loop.tolerance,
        maximum_outer_iterations: config.outer_loop.maximum_iterations,
        inner_tolerance: config.inner_loop.tolerance,
        maximum_inner_iterations: config.inner_loop.maximum_iterations,
        calculation_type: Calculation::Coherent,
    };
    let mut outer_loop: crate::outer_loop::OuterLoop<
        T,
        GeometryDim,
        Conn,
        BandDim,
        crate::spectral::SpectralSpace<T::RealField, ()>,
    > = crate::outer_loop::OuterLoopBuilder::new()
        .with_mesh(mesh)
        .with_hamiltonian(&hamiltonian)
        .with_spectral_space(&spectral_space)
        .with_convergence_settings(&outer_config)
        .with_tracker(tracker)
        .with_info_desk(tracker.info_desk)
        .build()?;

    let initial_potential = Potential::from_vector(nalgebra::DVector::from_element(
        mesh.num_nodes(),
        T::zero().real(),
    ));
    outer_loop.run_loop(initial_potential)
}
