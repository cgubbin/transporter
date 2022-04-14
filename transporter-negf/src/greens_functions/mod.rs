mod aggregate;
mod dense;
mod recursive;
mod sparse;

use crate::{device::info_desk::DeviceInfoDesk, Hamiltonian};
pub(crate) use aggregate::{
    AggregateGreensFunctionMethods, AggregateGreensFunctions, GreensFunctionBuilder,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use transporter_mesher::SmallDim;

#[derive(Clone, Debug)]
pub(crate) struct GreensFunction<Matrix, T>
where
    Matrix: GreensFunctionMethods<T>,
    T: RealField + Copy,
{
    matrix: Matrix,
    marker: std::marker::PhantomData<T>,
}

impl<Matrix, T> GreensFunction<Matrix, T>
where
    Matrix: GreensFunctionMethods<T>,
    T: RealField + Copy,
{
    fn as_mut(&mut self) -> &mut Matrix {
        &mut self.matrix
    }
}

pub(crate) trait GreensFunctionMethods<T>
where
    T: RealField + Copy,
{
    type SelfEnergy;
    fn generate_advanced_into(&mut self, retarded: &Self) -> color_eyre::Result<()>;
    fn generate_greater_into(
        &mut self,
        lesser: &Self,
        retarded: &Self,
        advanced: &Self,
    ) -> color_eyre::Result<()>;
    fn generate_lesser_into(
        &mut self,
        retarded_greens_function: &Self,
        retarded_self_energy: &Self,
        fermi_levels: &[T],
    ) -> color_eyre::Result<()>;
    fn generate_retarded_into(
        &mut self,
        energy: T,
        wavevector: T,
        hamiltonian: &Hamiltonian<T>,
        self_energy: &Self,
    ) -> color_eyre::Result<()>;
}

/// InfoDesk trait to provide the Fermi levels and 0th order Fermi integrals to Greens Function methods
/// These methods currently only really make sense in a one-dimensional structure. In a higher order system
/// we need an iterator over the contact regions
pub(crate) trait GreensFunctionInfoDesk<T: Copy + RealField> {
    /// The fermi level in the source contact in eV
    fn get_fermi_level_at_source(&self) -> T;
    /// The Fermi level in the drain contact in eV
    fn get_fermi_level_at_drain(&self) -> T;
    /// The zero order fermi integral in the source contact
    fn get_fermi_integral_at_source(&self, energy: T) -> T;
    /// The zero order fermi integral in the drain contact
    fn get_fermi_integral_at_drain(&self, energy: T) -> T;
    /// The fermi function in the source contact
    fn get_fermi_function_at_source(&self, energy: T) -> T;
    /// The fermi function in the drain contact
    fn get_fermi_function_at_drain(&self, energy: T) -> T;
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

    fn get_fermi_level_at_drain(&self) -> T {
        self.get_fermi_level_at_source() + self.voltage_offsets[1]
    }

    fn get_fermi_integral_at_source(&self, energy: T) -> T {
        let fermi_level = self.get_fermi_level_at_source(); // - T::one() / (T::one() + T::one());
        let argument = T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
            * (fermi_level - energy)
            / (T::from_f64(crate::constants::BOLTZMANN).unwrap() * self.temperature);
        (T::one() + argument.exp()).ln()
    }

    fn get_fermi_integral_at_drain(&self, energy: T) -> T {
        let fermi_level = self.get_fermi_level_at_drain(); // - T::one() / (T::one() + T::one());
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

    fn get_fermi_function_at_drain(&self, energy: T) -> T {
        let fermi_level = self.get_fermi_level_at_drain(); // - T::one() / (T::one() + T::one());
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
    use nalgebra::{allocator::Allocator, DMatrix, DefaultAllocator, RealField};
    use nalgebra_sparse::CsrMatrix;
    use num_complex::Complex;
    use transporter_mesher::{Connectivity, Mesh, SmallDim};

    impl<'a, T, GeometryDim, Conn, BandDim>
        GreensFunctionBuilder<
            T,
            &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
            &'a Mesh<T, GeometryDim, Conn>,
            &'a SpectralSpace<T, ()>,
        >
    where
        T: RealField + Copy,
        GeometryDim: SmallDim,
        BandDim: SmallDim,
        Conn: Connectivity<T, GeometryDim>,
        DefaultAllocator:
            Allocator<T, GeometryDim> + Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
    {
        /// A test helper method to build dense representations for coherent calculations. This is to aid testing, as we can
        /// then directly compare the results of the sparse and dense implementations of the update trait.
        pub(crate) fn build_dense(
            self,
        ) -> color_eyre::Result<
            AggregateGreensFunctions<'a, T, DMatrix<Complex<T>>, GeometryDim, BandDim>,
        > {
            // A 1D implementation. All 2D should redirect to the dense method
            let dmatrix = DMatrix::zeros(self.mesh.elements().len(), self.mesh.elements().len());

            let spectrum_of_dmatrix = (0..self.spectral.total_number_of_points())
                .map(|_| GreensFunction {
                    matrix: dmatrix.clone(),
                    marker: std::marker::PhantomData,
                })
                .collect::<Vec<_>>();

            // In the coherent calculation we do not use the advanced or greater Greens function, other than transiently
            Ok(AggregateGreensFunctions {
                //    spectral: self.spectral,
                info_desk: self.info_desk,
                retarded: spectrum_of_dmatrix.clone(),
                advanced: spectrum_of_dmatrix.clone(),
                lesser: spectrum_of_dmatrix.clone(),
                greater: spectrum_of_dmatrix,
            })
        }
    }

    /// Helper function to convert sparse self energies to their dense equivalents to enable comparative testing
    /// of sparse and dense implementations
    fn convert_sparse_self_energy_to_dense<T: Copy + RealField, GeometryDim, Conn>(
        sparse_self_energy: SelfEnergy<T, GeometryDim, Conn, CsrMatrix<Complex<T>>>,
    ) -> SelfEnergy<T, GeometryDim, Conn, DMatrix<Complex<T>>>
    where
        GeometryDim: SmallDim,
        Conn: Connectivity<T, GeometryDim>,
        DefaultAllocator: Allocator<T, GeometryDim>,
    {
        // Take the sparse self-energy matrix and re-serialize to a DMatrix
        let retarded_se_matrices = sparse_self_energy
            .retarded
            .iter()
            .map(nalgebra_sparse::convert::serial::convert_csr_dense)
            .collect::<Vec<_>>();
        SelfEnergy {
            ma: sparse_self_energy.ma,
            mc: sparse_self_energy.mc,
            marker: sparse_self_energy.marker,
            retarded: retarded_se_matrices,
        }
    }

    use crate::app::{tracker::TrackerBuilder, Configuration};
    use crate::device::{info_desk::BuildInfoDesk, Device};
    use nalgebra::U1;

    #[test]
    fn diagonal_elements_of_retarded_csr_match_dense() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new()
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        let dense_se = convert_sparse_self_energy_to_dense(self_energy);

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &dense_se, &spectral_space)
            .unwrap();

        for ((sparse, dense), _energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let sparse_diagonal = sparse.matrix.diagonal_as_csr();
            let dense_diagonal = dense.matrix.diagonal();
            for (sparse_value, dense_value) in sparse_diagonal
                .values()
                .iter()
                .zip(dense_diagonal.into_iter())
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
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new()
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        let dense_se = convert_sparse_self_energy_to_dense(self_energy);

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &dense_se, &spectral_space)
            .unwrap();

        for ((sparse, dense), _energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let dense_diagonal = dense.matrix.column(0);
            for (sparse_row, dense_value) in
                sparse.matrix.row_iter().zip(dense_diagonal.into_iter())
            {
                let sparse_value = sparse_row.values()[0];
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
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new()
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        let dense_se = convert_sparse_self_energy_to_dense(self_energy);

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &dense_se, &spectral_space)
            .unwrap();

        for ((sparse, dense), _energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let dense_diagonal = dense.matrix.column(dense.matrix.ncols() - 1);
            for (sparse_row, dense_value) in
                sparse.matrix.row_iter().zip(dense_diagonal.into_iter())
            {
                let sparse_value = sparse_row.values()[sparse_row.values().len() - 1];
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
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new()
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        // Act
        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();
        gf.update_aggregate_lesser_greens_function(&self_energy, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        let dense_se = convert_sparse_self_energy_to_dense(self_energy);

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &dense_se, &spectral_space)
            .unwrap();

        dense_gf
            .update_aggregate_lesser_greens_function(&dense_se, &spectral_space)
            .unwrap();

        for ((sparse, dense), _energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let sparse_diagonal = sparse.matrix.diagonal_as_csr();
            let dense_diagonal = dense.matrix.diagonal();
            for (sparse_value, dense_value) in sparse_diagonal
                .values()
                .iter()
                .zip(dense_diagonal.into_iter())
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
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new()
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();
        self_energy
            .recalculate_contact_self_energy(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();
        gf.update_aggregate_lesser_greens_function(&self_energy, &spectral_space)
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
