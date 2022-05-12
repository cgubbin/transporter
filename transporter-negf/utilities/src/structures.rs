use nalgebra::{RealField, U1};
use nalgebra_sparse::CsrMatrix;
use num_complex::Complex;
use std::path::PathBuf;
use transporter_mesher::Mesh1d;
use transporter_negf::{
    app::Configuration,
    device::{
        info_desk::{BuildInfoDesk, DeviceInfoDesk},
        Device,
    },
    greens_functions::{AggregateGreensFunctions, GreensFunctionBuilder},
    hamiltonian::Hamiltonian,
    spectral::SpectralSpace,
};

pub fn construct_device<
    T: RealField + Copy + serde::de::DeserializeOwned + num_traits::ToPrimitive + num_traits::NumCast,
>() -> (DeviceInfoDesk<T, U1, U1>, Mesh1d<T>, SpectralSpace<T, ()>) {
    let path: PathBuf = "/Users/cgubbin/rust/work/transporterb/transporter-negf/utilities/test_structures/single.toml".into();
    let device: Device<T, U1> = Device::build(path).unwrap();
    let info_desk = device.build_device_info_desk().unwrap();

    let config: Configuration<T> = Configuration::build().unwrap();
    let mesh: Mesh1d<T> = transporter_negf::app::build_mesh_with_config(&config, device).unwrap();

    let spectral_space_builder = transporter_negf::spectral::SpectralSpaceBuilder::new()
        .with_number_of_energy_points(config.spectral.number_of_energy_points)
        .with_energy_range(std::ops::Range {
            start: config.spectral.minimum_energy,
            end: config.spectral.maximum_energy,
        })
        .with_energy_integration_method(config.spectral.energy_integration_rule);

    let spectral_space = spectral_space_builder.build_coherent();

    (info_desk, mesh, spectral_space)
}

pub fn construct_sparse_greens_function<'a, T>(
    mesh: &'a Mesh1d<T>,
    info_desk: &'a DeviceInfoDesk<T, U1, U1>,
    spectral_space: &'a SpectralSpace<T, ()>,
) -> AggregateGreensFunctions<'a, T, CsrMatrix<Complex<T>>, U1, U1>
where
    T: RealField
        + Copy
        + serde::de::DeserializeOwned
        + num_traits::ToPrimitive
        + num_traits::NumCast,
{
    GreensFunctionBuilder::new()
        .with_info_desk(info_desk)
        .with_mesh(mesh)
        .with_spectral_discretisation(spectral_space)
        .build()
        .unwrap()
}

pub fn construct_sparse_self_energy<'a, T>(
    mesh: &'a Mesh1d<T>,
    spectral_space: &'a SpectralSpace<T, ()>,
) -> transporter_negf::self_energy::SelfEnergy<T, U1, transporter_mesher::Segment1dConnectivity>
where
    T: RealField
        + Copy
        + serde::de::DeserializeOwned
        + num_traits::ToPrimitive
        + num_traits::NumCast,
{
    transporter_negf::self_energy::SelfEnergyBuilder::new()
        .with_mesh(mesh)
        .with_spectral_discretisation(spectral_space)
        .build_coherent()
        .unwrap()
}

pub fn construct_device_hamiltonian<'a, T: Copy + RealField>(
    mesh: &'a Mesh1d<T>,
    info_desk: &'a DeviceInfoDesk<T, U1, U1>,
) -> Hamiltonian<T> {
    let tracker =
        transporter_negf::app::TrackerBuilder::new(transporter_negf::app::Calculation::Coherent)
            .with_mesh(mesh)
            .with_info_desk(info_desk)
            .build()
            .unwrap();
    transporter_negf::hamiltonian::HamiltonianBuilder::new()
        .with_mesh(mesh)
        .with_info_desk(&tracker)
        .build()
        .unwrap()
}
