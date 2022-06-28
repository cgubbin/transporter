// Copyright 2022 Chris Gubbin
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Green's Functions
//!
//! This module calculates the Green's functions utilised to find the
//! quantum electronic density in a device at fixed electrostatic potential.

/// Methods for computation of the Green's functions
pub mod methods;

pub use methods::aggregate::{
    AggregateGreensFunctionMethods, AggregateGreensFunctions, GreensFunctionBuilder,
};
pub(crate) use methods::mixed::MMatrix;

use crate::{device::info_desk::DeviceInfoDesk, hamiltonian::Hamiltonian};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use transporter_mesher::SmallDim;

#[derive(thiserror::Error, Debug, miette::Diagnostic)]
/// Error type for creation and calculation of the `GreensFunctions`
pub enum GreensFunctionError {
    // We wrap Csr errors in an anyhow error here so they are Send + Sync, something not
    // implemented by the `SparseFormatError` natively
    /// Errors related to construction of, or access to Csr matrices
    #[error(transparent)]
    Csr(#[from] anyhow::Error),
    /// Failues to invert in calculation of the retarded Green's function
    #[error("Failed to invert for the retarded Green's Function")]
    Inversion,
    /// Failures in the recursive Green's function algorithms
    #[error(transparent)]
    Recursion(#[from] RecursionError),
    /// A failed security check -> a Hermitian matrix is not computed as hermitian
    #[error(transparent)]
    SecurityCheck(#[from] SecurityCheck),
}

/// Error for the recursive Green's function methods. Currently unimplemented
#[derive(thiserror::Error, Debug, miette::Diagnostic)]
pub enum RecursionError {}

/// Error for failed security checks
#[derive(thiserror::Error, Debug, miette::Diagnostic)]
#[error("Security error in {calculation:?} at global index {index:?}")]
pub struct SecurityCheck {
    /// The calculation that has failed
    pub(crate) calculation: String,
    /// The global index in the spectral space that has failed
    pub(crate) index: usize,
}

/// A Greens function for a single spectral point
#[derive(Clone, Debug)]
pub(crate) struct GreensFunction<Matrix, T>
where
    Matrix: GreensFunctionMethods<T> + Send + Sync,
    T: RealField + Copy + Send + Sync,
{
    /// The matrix storing the Green's function
    matrix: Matrix,
    /// A marker type for the numerical field (f64, f32 etc)
    marker: std::marker::PhantomData<T>,
}

impl<Matrix, T> GreensFunction<Matrix, T>
where
    Matrix: GreensFunctionMethods<T> + Send + Sync,
    T: RealField + Copy + Send + Sync,
{
    fn as_mut(&mut self) -> &mut Matrix {
        &mut self.matrix
    }

    pub(crate) fn as_ref(&self) -> &Matrix {
        &self.matrix
    }
}

/// Methods for computation of `GreensFunction`
pub trait GreensFunctionMethods<T>
where
    T: RealField + Copy,
{
    /// The backend used in the self-energy computation
    type SelfEnergy;
    /// Compute the advanced greens function into `self`
    fn generate_advanced_into(&mut self, retarded: &Self) -> Result<(), GreensFunctionError>;
    /// Compute the greater greens function into `self`
    fn generate_greater_into(
        &mut self,
        lesser: &Self,
        retarded: &Self,
        advanced: &Self,
    ) -> Result<(), GreensFunctionError>;
    /// Compute the lesser greens function into `self`
    fn generate_lesser_into(
        &mut self,
        energy: T,
        wavevector: T,
        hamiltonian: &Hamiltonian<T>,
        retarded_greens_function: &Self,
        retarded_self_energy: &Self,
        fermi_levels: &[T],
    ) -> Result<(), GreensFunctionError>;
    /// Compute the retarded greens function into `self`
    fn generate_retarded_into(
        &mut self,
        energy: T,
        wavevector: T,
        hamiltonian: &Hamiltonian<T>,
        self_energy: &Self,
    ) -> Result<(), GreensFunctionError>;
}

/// InfoDesk trait to provide the Fermi levels and 0th order Fermi integrals to Greens Function methods
/// These methods currently only really make sense in a one-dimensional structure. In a higher order system
/// we need an iterator over the contact regions
pub(crate) trait GreensFunctionInfoDesk<T: Copy + RealField> {
    /// The fermi level in the source contact in eV
    fn get_fermi_level_at_source(&self) -> T;
    /// The Fermi level in the drain contact in eV
    fn get_fermi_level_at_drain(&self, voltage: T) -> T;
    /// The zero order fermi integral in the source contact
    fn get_fermi_integral_at_source(&self, energy: T) -> T;
    /// The zero order fermi integral in the drain contact
    fn get_fermi_integral_at_drain(&self, energy: T, voltage: T) -> T;
    /// The fermi function in the source contact
    fn get_fermi_function_at_source(&self, energy: T) -> T;
    /// The fermi function in the drain contact
    fn get_fermi_function_at_drain(&self, energy: T, voltage: T) -> T;
}

impl<'a, T: Copy + RealField, BandDim: SmallDim, GeometryDim: SmallDim> GreensFunctionInfoDesk<T>
    for &'a DeviceInfoDesk<T, GeometryDim, BandDim>
where
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    fn get_fermi_level_at_source(&self) -> T {
        // The net doping density in the source contact (region 0)
        let doping_density = self.donor_densities[0] - self.acceptor_densities[0];

        //TODO : A single band impl here.
        let n_band = 0;
        let band_offset = self.band_offsets[0][n_band];
        let n3d = (T::one() + T::one())
            * (self.effective_masses[0][0][n_band]
                * T::from_f64(crate::constants::ELECTRON_MASS).unwrap()
                * T::from_f64(crate::constants::BOLTZMANN).unwrap()
                * self.temperature
                / T::from_f64(crate::constants::HBAR).unwrap().powi(2)
                / (T::one() + T::one())
                / T::from_f64(std::f64::consts::PI).unwrap())
            .powf(T::from_f64(1.5).unwrap());

        let (factor, gamma) = (
            T::from_f64(crate::constants::ELECTRON_CHARGE / crate::constants::BOLTZMANN).unwrap()
                / self.temperature,
            T::from_f64(std::f64::consts::PI.sqrt() / 2.).unwrap(),
        );
        let eta_f = crate::fermi::inverse_fermi_integral_05(gamma * doping_density / n3d);

        let ef_minus_ec = eta_f / factor;
        ef_minus_ec + band_offset
    }

    fn get_fermi_level_at_drain(&self, voltage: T) -> T {
        self.get_fermi_level_at_source() + voltage //self.voltage_offsets[1]
    }

    fn get_fermi_integral_at_source(&self, energy: T) -> T {
        let fermi_level = self.get_fermi_level_at_source(); // - T::one() / (T::one() + T::one());
        let argument = T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
            * (fermi_level - energy)
            / (T::from_f64(crate::constants::BOLTZMANN).unwrap() * self.temperature);
        (T::one() + argument.exp()).ln()
    }

    fn get_fermi_integral_at_drain(&self, energy: T, voltage: T) -> T {
        let fermi_level = self.get_fermi_level_at_drain(voltage); // - T::one() / (T::one() + T::one());
        let argument = T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
            * (fermi_level - energy)
            / (T::from_f64(crate::constants::BOLTZMANN).unwrap() * self.temperature);
        (T::one() + argument.exp()).ln()
    }

    fn get_fermi_function_at_source(&self, energy: T) -> T {
        let fermi_level = self.get_fermi_level_at_source(); // - T::one() / (T::one() + T::one());
        let argument = -T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
            * (fermi_level - energy)
            / (T::from_f64(crate::constants::BOLTZMANN).unwrap() * self.temperature);
        T::one() / (T::one() + argument.exp())
    }

    fn get_fermi_function_at_drain(&self, energy: T, voltage: T) -> T {
        let fermi_level = self.get_fermi_level_at_drain(voltage); // - T::one() / (T::one() + T::one());
        let argument = -T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
            * (fermi_level - energy)
            / (T::from_f64(crate::constants::BOLTZMANN).unwrap() * self.temperature);
        T::one() / (T::one() + argument.exp())
    }
}

#[cfg(test)]
mod test {
    use super::{
        AggregateGreensFunctionMethods, AggregateGreensFunctions, DeviceInfoDesk, GreensFunction,
        GreensFunctionBuilder,
    };
    use crate::{
        self_energy::SelfEnergy,
        spectral::{SpectralDiscretisation, SpectralSpace},
    };
    use nalgebra::{allocator::Allocator, DefaultAllocator};
    use nalgebra_sparse::CsrMatrix;
    use num_complex::Complex;
    use transporter_mesher::{Connectivity, Mesh, SmallDim};

    impl<'a, GeometryDim, Conn, BandDim>
        GreensFunctionBuilder<
            f64,
            &'a DeviceInfoDesk<f64, GeometryDim, BandDim>,
            &'a Mesh<f64, GeometryDim, Conn>,
            &'a SpectralSpace<f64, ()>,
            (),
        >
    where
        GeometryDim: SmallDim,
        BandDim: SmallDim,
        Conn: Connectivity<f64, GeometryDim>,
        DefaultAllocator:
            Allocator<f64, GeometryDim> + Allocator<f64, BandDim> + Allocator<[f64; 3], BandDim>,
    {
        /// A test helper method to build dense representations for coherent calculations. This is to aid testing, as we can
        /// then directly compare the results of the sparse and dense implementations of the update trait.
        pub(crate) fn build_dense(
            self,
        ) -> color_eyre::Result<
            AggregateGreensFunctions<'a, f64, ndarray::Array2<Complex<f64>>, GeometryDim, BandDim>,
        > {
            // A 1D implementation. All 2D should redirect to the dense method
            let dmatrix =
                ndarray::Array2::zeros((self.mesh.vertices().len(), self.mesh.vertices().len()));

            let spectrum_of_dmatrix = (0..self.spectral.total_number_of_points())
                .map(|_| GreensFunction {
                    matrix: dmatrix.clone(),
                    marker: std::marker::PhantomData,
                })
                .collect::<Vec<_>>();

            // In the coherent calculation we do not use the advanced or greater Greens function, other than transiently
            Ok(AggregateGreensFunctions {
                info_desk: self.info_desk,
                security_checks: self.security_checks,
                retarded: spectrum_of_dmatrix.clone(),
                advanced: spectrum_of_dmatrix.clone(),
                lesser: spectrum_of_dmatrix.clone(),
                greater: spectrum_of_dmatrix,
            })
        }
    }

    impl<'a, GeometryDim, Conn>
        crate::self_energy::SelfEnergyBuilder<
            f64,
            &'a SpectralSpace<f64, ()>,
            &'a Mesh<f64, GeometryDim, Conn>,
        >
    where
        GeometryDim: SmallDim,
        Conn: Connectivity<f64, GeometryDim> + Send + Sync,
        <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
        DefaultAllocator: Allocator<f64, GeometryDim>,
        <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
    {
        pub(crate) fn build_incoherent_b(
            self,
        ) -> color_eyre::Result<SelfEnergy<f64, GeometryDim, Conn>> {
            // Collect the indices of all elements at the boundaries
            let vertices_at_boundary: Vec<usize> = self
                .mesh
                .connectivity()
                .iter()
                .enumerate()
                .filter_map(|(vertex_index, vertex_connectivity)| {
                    if vertex_connectivity.len() == 1 {
                        Some(vertex_index)
                    } else {
                        None
                    }
                })
                .collect();
            let pattern = crate::self_energy::construct_csr_pattern_from_vertices(
                &vertices_at_boundary,
                self.mesh.vertices().len(),
            )?;
            let initial_values = vertices_at_boundary
                .iter()
                .map(|_| Complex::from(0_f64))
                .collect::<Vec<_>>();
            let csrmatrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_values.clone())
                .map_err(|e| {
                    color_eyre::eyre::eyre!("Failed to initialise sparse self energy matrix {}", e)
                })?;

            let csrmatrix = sprs::CsMat::new(
                (csrmatrix.nrows(), csrmatrix.ncols()),
                csrmatrix.row_offsets().to_vec(),
                csrmatrix.col_indices().to_vec(),
                initial_values,
            );

            let dmatrix =
                ndarray::Array2::zeros((self.mesh.vertices().len(), self.mesh.vertices().len()));
            let num_spectral_points = self.spectral.number_of_wavevector_points()
                * self.spectral.number_of_energy_points();

            Ok(SelfEnergy {
                ma: std::marker::PhantomData,
                mc: std::marker::PhantomData,
                marker: std::marker::PhantomData,
                security_checks: false,
                contact_retarded: vec![csrmatrix.clone(); num_spectral_points],
                contact_lesser: Some(vec![csrmatrix; num_spectral_points]),
                incoherent_retarded: Some(vec![dmatrix.clone(); num_spectral_points]),
                incoherent_lesser: Some(vec![dmatrix; num_spectral_points]),
            })
        }
    }

    use crate::app::Calculation;
    use crate::app::{tracker::TrackerBuilder, Configuration};
    use crate::device::{info_desk::BuildInfoDesk, Device};
    use nalgebra::U1;

    #[test]
    fn diagonal_elements_of_retarded_csr_match_dense() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::SpectralSpaceBuilder::default()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_coherent()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();
        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_incoherent_b()
            .unwrap();

        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        for ((sparse, dense), _energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let sparse_diagonal = sparse.matrix.diag().to_dense();
            let dense_diagonal = dense.matrix.diag();
            for (sparse_value, dense_value) in
                sparse_diagonal.iter().zip(dense_diagonal.into_iter())
            {
                approx::assert_relative_eq!(
                    sparse_value.re,
                    dense_value.re,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
                approx::assert_relative_eq!(
                    sparse_value.im,
                    dense_value.im,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
            }
        }
    }

    #[test]
    fn diagonal_elements_of_retarded_csr_match_dense_with_internal_leads() {
        let path = std::path::PathBuf::try_from("../.config/structureinternal.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::SpectralSpaceBuilder::default()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_coherent()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();
        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_incoherent_b()
            .unwrap();

        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        for ((sparse, dense), _energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let sparse_diagonal = sparse.matrix.diag().to_dense();
            let dense_diagonal = dense.matrix.diag();
            for (sparse_value, dense_value) in
                sparse_diagonal.iter().zip(dense_diagonal.into_iter())
            {
                approx::assert_relative_eq!(
                    sparse_value.re,
                    dense_value.re,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
                approx::assert_relative_eq!(
                    sparse_value.im,
                    dense_value.im,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
            }
        }
    }

    #[test]
    fn left_column_elements_of_retarded_csr_match_dense() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::SpectralSpaceBuilder::default()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_coherent()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_incoherent_b()
            .unwrap();

        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        for ((sparse, dense), _energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let dense_diagonal = dense.matrix.column(0);
            for (sparse_row, dense_value) in sparse
                .matrix
                .outer_iterator()
                .zip(dense_diagonal.into_iter())
            {
                let sparse_value = sparse_row.data()[0];
                approx::assert_relative_eq!(
                    sparse_value.re,
                    dense_value.re,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
                approx::assert_relative_eq!(
                    sparse_value.im,
                    dense_value.im,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
            }
        }
    }

    #[test]
    fn left_column_elements_of_retarded_csr_match_dense_with_internal_leads() {
        let path = std::path::PathBuf::try_from("../.config/structureinternal.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::SpectralSpaceBuilder::default()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_coherent()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_incoherent_b()
            .unwrap();

        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let number_of_vertices_in_internal_lead =
            (3 * gf.retarded[0].matrix.shape().0 - gf.retarded[0].matrix.nnz() - 2) / 4;
        let number_of_filled_rows =
            gf.retarded[0].matrix.shape().0 - 2 * number_of_vertices_in_internal_lead;

        for ((sparse, dense), _energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let dense_column = dense.matrix.column(number_of_vertices_in_internal_lead);
            for (sparse_row, dense_value) in sparse
                .matrix
                .outer_iterator()
                .zip(dense_column.into_iter())
                .skip(number_of_vertices_in_internal_lead)
                .take(number_of_filled_rows)
            {
                let sparse_value = sparse_row.data()[0]; // left column
                approx::assert_relative_eq!(
                    sparse_value.re,
                    dense_value.re,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
                approx::assert_relative_eq!(
                    sparse_value.im,
                    dense_value.im,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
            }
        }
    }

    #[test]
    fn right_column_elements_of_retarded_csr_match_dense() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::SpectralSpaceBuilder::default()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_coherent()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_incoherent_b()
            .unwrap();

        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        for ((sparse, dense), _energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let dense_diagonal = dense.matrix.column(dense.matrix.ncols() - 1);
            for (sparse_row, dense_value) in sparse
                .matrix
                .outer_iterator()
                .zip(dense_diagonal.into_iter())
            {
                let nvals = sparse_row.data().len();
                let sparse_value = sparse_row.data()[nvals - 1];
                approx::assert_relative_eq!(
                    sparse_value.re,
                    dense_value.re,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
                approx::assert_relative_eq!(
                    sparse_value.im,
                    dense_value.im,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
            }
        }
    }

    #[test]
    fn right_column_elements_of_retarded_csr_match_dense_with_internal_leads() {
        let path = std::path::PathBuf::try_from("../.config/structureinternal.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::SpectralSpaceBuilder::default()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_coherent()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_incoherent_b()
            .unwrap();

        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let number_of_vertices_in_internal_lead =
            (3 * gf.retarded[0].matrix.shape().0 - gf.retarded[0].matrix.nnz() - 2) / 4;
        let number_of_filled_rows =
            gf.retarded[0].matrix.shape().0 - 2 * number_of_vertices_in_internal_lead;

        for ((sparse, dense), _energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let dense_column = dense
                .matrix
                .column(dense.matrix.ncols() - 1 - number_of_vertices_in_internal_lead);
            for (sparse_row, dense_value) in sparse
                .matrix
                .outer_iterator()
                .zip(dense_column.into_iter())
                .skip(number_of_vertices_in_internal_lead)
                .take(number_of_filled_rows)
            {
                let nvals = sparse_row.data().len();
                let sparse_value = sparse_row.data()[nvals - 1];
                approx::assert_relative_eq!(
                    sparse_value.re,
                    dense_value.re,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
                approx::assert_relative_eq!(
                    sparse_value.im,
                    dense_value.im,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
            }
        }
    }

    #[test]
    fn diagonal_elements_of_lesser_csr_match_dense() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::SpectralSpaceBuilder::default()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_coherent()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        // Act
        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();
        gf.update_aggregate_lesser_greens_function(0., &hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_incoherent_b()
            .unwrap();

        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        dense_gf
            .update_aggregate_lesser_greens_function(
                0.,
                &hamiltonian,
                &self_energy,
                &spectral_space,
            )
            .unwrap();

        for ((sparse, dense), _energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let sparse_diagonal = sparse.matrix.diag().to_dense();
            let dense_diagonal = dense.matrix.diag();
            for (sparse_value, dense_value) in
                sparse_diagonal.iter().zip(dense_diagonal.into_iter())
            {
                approx::assert_relative_eq!(
                    sparse_value.re,
                    dense_value.re,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
                approx::assert_relative_eq!(
                    sparse_value.im,
                    dense_value.im,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
            }
        }
    }

    use rand::{thread_rng, Rng};
    #[test]
    fn diagonal_elements_of_lesser_csr_match_dense_with_internal_leads() {
        let path = std::path::PathBuf::try_from("../.config/structureinternal.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let mut tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let mut hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();
        let mut rng = thread_rng();
        let potential = crate::outer_loop::Potential::from(ndarray::Array1::from(
            (0..mesh.vertices().len())
                .map(|_| rng.gen())
                .collect::<Vec<_>>(),
        ));
        tracker.update_potential(potential);
        hamiltonian.update_potential(&tracker, &mesh).unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::SpectralSpaceBuilder::default()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_coherent()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        // Act
        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();
        gf.update_aggregate_lesser_greens_function(0., &hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_incoherent_b()
            .unwrap();

        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        dense_gf
            .update_aggregate_lesser_greens_function(
                0.,
                &hamiltonian,
                &self_energy,
                &spectral_space,
            )
            .unwrap();

        for ((sparse, dense), _energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let sparse_diagonal = sparse.matrix.diag().to_dense();
            let dense_diagonal = dense.matrix.diag();
            for (sparse_value, dense_value) in
                sparse_diagonal.iter().zip(dense_diagonal.into_iter())
            {
                approx::assert_relative_eq!(
                    sparse_value.re,
                    dense_value.re,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
                approx::assert_relative_eq!(
                    sparse_value.im,
                    dense_value.im,
                    epsilon = std::f64::EPSILON * 10000000_f64
                );
            }
        }
    }

    #[test]
    fn accumulated_charge_matches_the_edge_doping_with_zero_potential() {
        let path =
            std::path::PathBuf::try_from("../test_structures/homogeneous_1e24_im3.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let n_layer = info_desk.donor_densities.len();
        let _edge_doping = (
            info_desk.donor_densities[0],
            info_desk.donor_densities[n_layer - 1],
        );

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::SpectralSpaceBuilder::default()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::default()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new(false)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_coherent()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();
        gf.update_aggregate_lesser_greens_function(0., &hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let charge = gf
            .accumulate_into_charge_density_vector(&mesh, &spectral_space)
            .unwrap();

        let x = &charge.as_ref()[0]; // We have one band so take the charge density at index 0

        let _n_elements = x.len();
        // approx::assert_relative_eq!(x[0], edge_doping.0);
        // approx::assert_relative_eq!(x[n_elements - 1], edge_doping.1);
        println!("{:?}", x);
    }
}
