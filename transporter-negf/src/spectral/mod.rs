/// This module provides the discrete energy and wavevector spaces necessary to
/// scaffold the Green's functions and their relation to the electron density
pub(crate) mod constructors;
mod energy;
mod wavevector;

pub(crate) use wavevector::WavevectorSpace;

use energy::EnergySpace;
use nalgebra::{allocator::Allocator, DVector, DefaultAllocator, RealField};
use transporter_mesher::{Connectivity, ElementMethods, Mesh, SmallDim};

pub(crate) trait SpectralDiscretisation<T: RealField + Send>: Send + Sync {
    type Iter: Iterator<Item = T> + Clone;
    fn total_number_of_points(&self) -> usize {
        self.number_of_energy_points() * self.number_of_wavevector_points()
    }
    fn number_of_energy_points(&self) -> usize;
    fn number_of_wavevector_points(&self) -> usize;
    fn integrate(&self, integrand: &[T]) -> T;
    fn integrate_over_wavevector(&self, integrand: &[T]) -> T;
    fn integrate_over_energy(&self, integrand: &[T]) -> T;
    fn energy_at(&self, index: usize) -> T;
    fn wavevector_at(&self, index: usize) -> T;
    fn iter_energies(&self) -> Self::Iter;
    fn iter_wavevectors(&self) -> Self::Iter;
    fn iter_energy_weights(&self) -> Self::Iter;
    fn iter_wavevector_weights(&self) -> Self::Iter;
    fn iter_energy_widths(&self) -> Self::Iter;
    fn iter_wavevector_widths(&self) -> Self::Iter;
    fn identify_bracketing_weights(&self, target_energy: T) -> color_eyre::Result<[T; 2]>;
    fn identify_bracketing_indices(&self, target_energy: T) -> color_eyre::Result<[usize; 2]>;
}

/// A general `SpectralSpace` which contains the wavevector and energy discretisation and associated integration rules
pub(crate) struct SpectralSpace<T: Copy + RealField, WavevectorSpace> {
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

    fn integrate(&self, integrand: &[T]) -> T {
        assert_eq!(
            integrand.len(),
            self.total_number_of_points(),
            "The integrand must be evaluated on the e-k grid"
        );
        let chunked_iterator = integrand.chunks(self.number_of_wavevector_points());
        let energy_integrand: Vec<T> = chunked_iterator
            .map(|wavevector_integrand_at_fixed_energy| {
                self.integrate_over_wavevector(wavevector_integrand_at_fixed_energy)
            })
            .collect();

        self.integrate_over_energy(&energy_integrand)
    }

    fn integrate_over_wavevector(&self, integrand: &[T]) -> T {
        assert_eq!(
            integrand.len(),
            self.number_of_wavevector_points(),
            "The integrand must be evaluated on the k grid"
        );

        integrand
            .iter()
            .zip(self.energy.weights())
            .fold(T::zero(), |sum, (&point, &weight)| sum + point * weight)
    }

    fn integrate_over_energy(&self, integrand: &[T]) -> T {
        assert!(
            integrand.len() == self.energy.num_points(),
            "We can only integrate if the Greens functions are evaluated on-grid"
        );

        integrand
            .iter()
            .zip(self.energy.weights())
            .fold(T::zero(), |sum, (&point, &weight)| sum + point * weight)
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
        let x = self
            .energy
            .grid
            .elements()
            .iter()
            .map(|x| x.0.diameterb())
            .collect::<Vec<_>>();
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

    fn integrate(&self, integrand: &[T]) -> T {
        self.integrate_over_energy(integrand)
    }

    fn integrate_over_energy(&self, integrand: &[T]) -> T {
        assert!(
            integrand.len() == self.energy.num_points(),
            "We can only integrate if the Greens functions are evaluated on-grid"
        );

        integrand
            .iter()
            .zip(self.energy.weights())
            .fold(T::zero(), |sum, (&point, &weight)| sum + point * weight)
    }

    fn integrate_over_wavevector(&self, _: &[T]) -> T {
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
        let x = self
            .energy
            .grid
            .elements()
            .iter()
            .map(|x| x.0.diameterb())
            .collect::<Vec<_>>();
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
}

pub(crate) trait GenerateWeights<T, GeometryDim, Conn>
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn query_integration_rule(&self) -> IntegrationRule;
    fn generate_weights_from_grid(&self, grid: &Mesh<T, GeometryDim, Conn>) -> DVector<T>;
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
    fn generate_weights_from_grid(&self, grid: &Mesh<T, GeometryDim, Conn>) -> DVector<T> {
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
            }
        };
        DVector::from_iterator(num_points, (0..num_points).map(weight))
    }
}
