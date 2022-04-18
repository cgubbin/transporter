#![allow(dead_code)]

pub mod app;
mod constants;
mod device;
mod fermi;
pub mod greens_functions;
pub mod hamiltonian;
mod inner_loop;
mod outer_loop;
mod postprocessor;
mod self_energy;
mod spectral;
mod utilities;

pub use constants::*;
pub use hamiltonian::*;

use nalgebra::RealField;
use std::marker::PhantomData;
use transporter_mesher::Mesh1d;

pub struct App<Device, T> {
    device: Device,
    marker: PhantomData<T>,
}

impl<Device, T> App<Device, T>
where
    T: Copy + RealField,
{
    pub fn run(_mesh_producer: impl Fn(usize) -> Mesh1d<T>) -> color_eyre::Result<()> {
        todo!()
    }
}
