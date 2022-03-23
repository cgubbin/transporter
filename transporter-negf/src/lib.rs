mod greens_functions;
mod hamiltonian;
mod inner_loop;
mod postprocessor;
mod self_energy;
mod spectral;

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
    T: RealField,
{
    pub fn run(mesh_producer: impl Fn(usize) -> Mesh1d<T>) -> color_eyre::Result<()> {
        todo!()
    }
}
