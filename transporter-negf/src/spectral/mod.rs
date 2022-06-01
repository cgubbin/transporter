//! # Spectral
//! This module provides the discrete energy and wavevector spaces necessary to
//! scaffold the Green's functions and their relation to the electron density
//!
//! A spectral space consists of an `EnergySpace` and an optional `WavevectorSpace`
//! the latter only necessary to run an incoherent calculation

mod constructors;
mod energy;
mod wavevector;

pub use constructors::SpectralSpaceBuilder;
pub(crate) use wavevector::WavevectorSpace;

use energy::EnergySpace;
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use ndarray::{Array1, ArrayView1, Axis};
use transporter_mesher::{Connectivity, ElementMethods, Mesh, SmallDim};

/// Trait for methods associated with the spectral-space
pub trait SpectralDiscretisation<T: RealField + Send>: Send + Sync {
    /// A mixed iterator over both the wavevector and energy grids, in the order they are
    /// stored in the GreensFunction and SelfEnergies
    type Iter: Iterator<Item = T> + Clone;
    /// The total number of unique points in the energy and wavevector grids
    fn total_number_of_points(&self) -> usize {
        self.number_of_energy_points() * self.number_of_wavevector_points()
    }
    /// The total number of points in the energy grid
    fn number_of_energy_points(&self) -> usize;
    /// The total number of points in the wavevector grid
    fn number_of_wavevector_points(&self) -> usize;
    /// Integrate a quantity defined discretely on the full grid over both energy and wavevector
    /// Panics if the `ArrayView1` has a different amount of entries to `self.total_number_of_points`
    fn integrate(&self, integrand: ArrayView1<T>) -> T;
    /// Integrate a quantity at fixed energy over the wavevector grid
    /// Panics if the `ArrayView1` has a different amount of entries to `self.number_of_wavevector_points`
    fn integrate_over_wavevector(&self, integrand: ArrayView1<T>) -> T;
    /// Integrate a quantity at fixed wavevector over the energy grid
    /// Panics if the `ArrayView1` has a different amount of entries to `self.number_of_energy_points`
    fn integrate_over_energy(&self, integrand: ArrayView1<T>) -> T;
    /// Return the energy at `index` in the energy grid
    fn energy_at(&self, index: usize) -> T;
    /// Return the wavevector at `index` in the wavevector grid
    fn wavevector_at(&self, index: usize) -> T;
    /// Returns an iterator over the energies in the energy grid
    fn iter_energies(&self) -> Self::Iter;
    /// Returns an iterator over the wavevectors in the wavevetor grid
    fn iter_wavevectors(&self) -> Self::Iter;
    /// Returns an iterator over the weights of the energies in the energy grid
    fn iter_energy_weights(&self) -> Self::Iter;
    /// Returns an iterator over the weights of wavevectors in the wavevector grid
    fn iter_wavevector_weights(&self) -> Self::Iter;
    /// Returns an iterator over the widths (dE) of energies in the energy grid
    fn iter_energy_widths(&self) -> Self::Iter;
    /// Returns an iterator over the widths (dk) of wavevectors in the wavevector grid
    fn iter_wavevector_widths(&self) -> Self::Iter;
    /// Identifies the two weights of the points bracketing energy `target_energy` in the energy grid
    fn identify_bracketing_weights(&self, target_energy: T) -> color_eyre::Result<[T; 2]>;
    /// Identifies the two indices of the points bracketing energy `target_energy` in the energy grid
    fn identify_bracketing_indices(&self, target_energy: T) -> color_eyre::Result<[usize; 2]>;
}

/// A general `SpectralSpace` which contains the wavevector and energy discretisation and associated integration rules
pub struct SpectralSpace<T: Copy + RealField, WavevectorSpace> {
    /// A `SpectralSpace` always has an associated energy space, so this is a concrete type
    pub(crate) energy: EnergySpace<T>,
    /// A `SpectralSpace` may have a wavevector space, either if the calculation is incoherent or the
    /// carrier effective mass varies across the structure
    wavevector: WavevectorSpace,
}

impl<T: Copy + RealField> SpectralSpace<T, ()> {
    pub(crate) fn iter_energy(&self) -> impl std::iter::Iterator<Item = &T> {
        self.energy.points()
    }
    pub(crate) fn number_of_energies(&self) -> usize {
        self.energy.points().count()
    }
}

impl<T: RealField + Copy + Clone, GeometryDim: SmallDim, Conn> SpectralDiscretisation<T>
    for SpectralSpace<T, WavevectorSpace<T, GeometryDim, Conn>>
where
    Conn: Connectivity<T, GeometryDim> + Send + Sync,
    <Conn as Connectivity<T, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Send + Sync,
{
    type Iter = std::vec::IntoIter<T>;
    fn number_of_energy_points(&self) -> usize {
        self.energy.num_points()
    }
    fn number_of_wavevector_points(&self) -> usize {
        // TODO implement in a blanket mesh method
        self.wavevector.num_points()
    }

    fn integrate(&self, integrand: ArrayView1<T>) -> T {
        assert_eq!(
            integrand.len(),
            self.total_number_of_points(),
            "The integrand must be evaluated on the e-k grid"
        );
        let chunked_iterator =
            integrand.axis_chunks_iter(Axis(0), self.number_of_wavevector_points());
        let energy_integrand: Array1<T> = Array1::from(
            chunked_iterator
                .map(|wavevector_integrand_at_fixed_energy| {
                    self.integrate_over_wavevector(wavevector_integrand_at_fixed_energy)
                })
                .collect::<Vec<_>>(),
        );

        self.integrate_over_energy(energy_integrand.view())
    }

    fn integrate_over_wavevector(&self, integrand: ArrayView1<T>) -> T {
        assert_eq!(
            integrand.len(),
            self.number_of_wavevector_points(),
            "The integrand must be evaluated on the k grid"
        );

        let point = self.wavevector.points().take(1).collect::<Vec<_>>()[0];
        let dim = point.coords.shape().1;

        match dim {
            1_usize => integrand
                .iter()
                .zip(self.iter_wavevector_weights())
                .zip(self.iter_wavevector_widths())
                .zip(self.wavevector.points())
                .fold(
                    T::zero(),
                    |sum, (((&integrand, weight), width), wavevector)| {
                        sum + integrand
                            * weight
                            * width
                            * wavevector.coords[0]
                            * T::from_f64(std::f64::consts::PI).unwrap()
                    },
                ),
            // If we are not in 1D we panic
            _ => unimplemented!(),
        }
    }

    fn integrate_over_energy(&self, integrand: ArrayView1<T>) -> T {
        assert!(
            integrand.len() == self.energy.num_points(),
            "We can only integrate if the Greens functions are evaluated on-grid"
        );

        integrand
            .iter()
            .zip(self.iter_energy_weights())
            .zip(self.iter_energy_widths())
            .fold(T::zero(), |sum, ((&integrand, weight), width)| {
                sum + integrand * weight * width
            })
    }

    fn energy_at(&self, index: usize) -> T {
        self.energy.grid.vertices()[index].0[0]
    }

    fn wavevector_at(&self, index: usize) -> T {
        self.wavevector.grid.vertices()[index].0[0]
    }

    #[allow(clippy::needless_collect)]
    fn iter_energies(&self) -> Self::Iter {
        let x = self.energy.points().copied().collect::<Vec<_>>();
        x.into_iter()
    }

    #[allow(clippy::needless_collect)]
    fn iter_wavevectors(&self) -> Self::Iter {
        let x = self.wavevector.points().map(|x| x[0]).collect::<Vec<_>>();
        x.into_iter()
    }

    #[allow(clippy::needless_collect)]
    fn iter_energy_widths(&self) -> Self::Iter {
        let x = self.energy.widths().copied().collect::<Vec<_>>();
        x.into_iter()
    }

    #[allow(clippy::needless_collect)]
    fn iter_wavevector_widths(&self) -> Self::Iter {
        let x = self
            .wavevector
            .grid
            .elements()
            .iter()
            .map(|x| x.0.diameter())
            .collect::<Vec<_>>();
        x.into_iter()
    }

    #[allow(clippy::needless_collect)]
    fn iter_energy_weights(&self) -> Self::Iter {
        let x = self.energy.weights().copied().collect::<Vec<_>>();
        x.into_iter()
    }

    #[allow(clippy::needless_collect)]
    fn iter_wavevector_weights(&self) -> Self::Iter {
        let x = self.wavevector.weights().copied().collect::<Vec<_>>();
        x.into_iter()
    }

    fn identify_bracketing_weights(&self, target_energy: T) -> color_eyre::Result<[T; 2]> {
        let idx_upper = self
            .iter_energies()
            .position(|energy| energy > target_energy);
        if let Some(idx_upper) = idx_upper {
            let idx_lower = idx_upper - 1;
            let delta_upper = (self.energy_at(idx_upper) - target_energy).abs();
            let delta_lower = (self.energy_at(idx_lower) - target_energy).abs();
            let delta = delta_upper + delta_lower;
            return Ok([delta - delta_lower, delta - delta_upper]);
        }
        Err(color_eyre::eyre::eyre!(
            "Failed to find a bracketed value, the passed energy cannot be in the mesh"
        ))
    }

    fn identify_bracketing_indices(&self, target_energy: T) -> color_eyre::Result<[usize; 2]> {
        let idx_upper = self
            .iter_energies()
            .position(|energy| energy > target_energy);
        if let Some(idx_upper) = idx_upper {
            let idx_lower = idx_upper - 1;
            return Ok([idx_lower, idx_upper]);
        }
        Err(color_eyre::eyre::eyre!(
            "Failed to find a bracketed value, the passed energy cannot be in the mesh"
        ))
    }
}

impl<T: RealField + Copy> SpectralDiscretisation<T> for SpectralSpace<T, ()> {
    type Iter = std::vec::IntoIter<T>;
    fn number_of_energy_points(&self) -> usize {
        self.energy.num_points()
    }

    fn number_of_wavevector_points(&self) -> usize {
        1
    }

    fn integrate(&self, integrand: ArrayView1<T>) -> T {
        self.integrate_over_energy(integrand)
    }

    fn integrate_over_energy(&self, integrand: ArrayView1<T>) -> T {
        assert!(
            integrand.len() == self.energy.num_points(),
            "We can only integrate if the Greens functions are evaluated on-grid"
        );

        integrand
            .iter()
            .zip(self.energy.weights())
            .fold(T::zero(), |sum, (&point, &weight)| sum + point * weight)
    }

    fn integrate_over_wavevector(&self, _: ArrayView1<T>) -> T {
        unreachable!()
    }

    fn energy_at(&self, index: usize) -> T {
        self.energy.grid.vertices()[index].0[0]
    }

    fn wavevector_at(&self, _: usize) -> T {
        T::zero()
    }

    #[allow(clippy::needless_collect)]
    fn iter_energies(&self) -> Self::Iter {
        let x = self.energy.points().copied().collect::<Vec<_>>();
        x.into_iter()
    }
    fn iter_wavevectors(&self) -> Self::Iter {
        vec![T::zero(); 1].into_iter()
    }

    #[allow(clippy::needless_collect)]
    fn iter_energy_widths(&self) -> Self::Iter {
        let x = self.energy.widths().copied().collect::<Vec<_>>();
        x.into_iter()
    }

    fn iter_wavevector_widths(&self) -> Self::Iter {
        vec![T::one(); 1].into_iter()
    }

    #[allow(clippy::needless_collect)]
    fn iter_energy_weights(&self) -> Self::Iter {
        let x = self.energy.weights().copied().collect::<Vec<_>>();
        x.into_iter()
    }

    fn iter_wavevector_weights(&self) -> Self::Iter {
        vec![T::one(); 1].into_iter()
    }

    fn identify_bracketing_weights(&self, target_energy: T) -> color_eyre::Result<[T; 2]> {
        let idx_upper = self
            .iter_energies()
            .position(|energy| energy > target_energy);
        if let Some(idx_upper) = idx_upper {
            let idx_lower = idx_upper - 1;
            let delta_upper = (self.energy_at(idx_upper) - target_energy).abs();
            let delta_lower = (self.energy_at(idx_lower) - target_energy).abs();
            let delta = delta_upper + delta_lower;
            return Ok([delta - delta_lower, delta - delta_upper]);
        }
        Err(color_eyre::eyre::eyre!(
            "Failed to find a bracketed value, the passed energy cannot be in the mesh"
        ))
    }

    fn identify_bracketing_indices(&self, target_energy: T) -> color_eyre::Result<[usize; 2]> {
        let idx_upper = self
            .iter_energies()
            .position(|energy| energy > target_energy);
        if let Some(idx_upper) = idx_upper {
            let idx_lower = idx_upper - 1;
            return Ok([idx_lower, idx_upper]);
        }
        Err(color_eyre::eyre::eyre!(
            "Failed to find a bracketed value, the passed energy cannot be in the mesh"
        ))
    }
}

/// Enum for discrete integration methods
#[derive(Clone, Copy, Debug, serde::Deserialize)]
pub enum IntegrationRule {
    /// Trapezium rule
    Trapezium,
    /// Romberg integration
    Romberg,
    /// Three point integration
    ThreePoint,
    /// Gauss-Kronrod
    GaussKronrod,
}

/// trait to generate weights for the spectral space
pub trait GenerateWeights<T, GeometryDim, Conn>
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    /// Return a reference to the `SpectralSpace` integration rule
    fn query_integration_rule(&self) -> IntegrationRule;
    /// Returns the weights as an owned vector
    fn generate_weights_from_grid(&self, grid: &Mesh<T, GeometryDim, Conn>) -> Array1<T>;
}

impl<T, GeometryDim, Conn> GenerateWeights<T, GeometryDim, Conn> for IntegrationRule
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn query_integration_rule(&self) -> IntegrationRule {
        *self
    }
    fn generate_weights_from_grid(&self, grid: &Mesh<T, GeometryDim, Conn>) -> Array1<T> {
        let num_points = grid.vertices().len();
        // A closure generating the weight for a given point index
        let weight = |idx: usize| -> T {
            match self {
                IntegrationRule::Trapezium => {
                    if (idx == 0) | (idx == num_points - 1) {
                        T::one() / (T::one() + T::one())
                    } else {
                        T::one()
                    }
                }
                IntegrationRule::Romberg => {
                    if (idx == 0) | (idx == num_points - 1) {
                        T::from_f64(1. / 3.).unwrap()
                    } else if idx % 2 == 0 {
                        T::from_f64(2. / 3.).unwrap()
                    } else {
                        T::from_f64(4. / 3.).unwrap()
                    }
                }
                IntegrationRule::ThreePoint => {
                    if (idx == 0) | (idx == num_points - 1) {
                        T::from_f64(17. / 48.).unwrap()
                    } else if (idx == 1) | (idx == num_points - 2) {
                        T::from_f64(59. / 48.).unwrap()
                    } else if (idx == 2) | (idx == num_points - 3) {
                        T::from_f64(43. / 48.).unwrap()
                    } else if (idx == 3) | (idx == num_points - 4) {
                        T::from_f64(49. / 48.).unwrap()
                    } else {
                        T::one()
                    }
                }
                IntegrationRule::GaussKronrod => unreachable!(),
            }
        };
        Array1::from_iter((0..num_points).map(weight))
    }
}
