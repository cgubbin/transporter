/// This module provides the discrete energy and wavevector spaces necessary to
/// scaffold the Green's functions and their relation to the electron density
pub(crate) mod constructors;
mod energy;
mod wavevector;

use energy::EnergySpace;
use nalgebra::allocator::Allocator;
use nalgebra::{DVector, DefaultAllocator, RealField, U1};
use transporter_mesher::{Connectivity, Mesh, Segment1dConnectivity, SmallDim};
use wavevector::WavevectorSpace;

pub(crate) trait SpectralDiscretisation<T: RealField> {
    fn total_number_of_points(&self) -> usize {
        self.number_of_energy_points() * self.number_of_wavevector_points()
    }
    fn number_of_energy_points(&self) -> usize;
    fn number_of_wavevector_points(&self) -> usize;
    fn integrate(&self, integrand: &[T]) -> T;
    fn integrate_over_wavevector(&self, integrand: &[T]) -> T;
    fn integrate_over_energy(&self, integrand: &[T]) -> T;
    // fn iter_all(&self) -> std::slice::Iter<'_, (T, T)>;
}

pub(crate) struct BallisticSpectral<T: Copy + RealField> {
    energy: EnergySpace<T>,
}

impl<T: Copy + RealField> BallisticSpectral<T> {
    fn new(energy: EnergySpace<T>) -> Self {
        Self { energy }
    }
}

pub(crate) struct ScatteringSpectral<T: Copy + RealField, GeometryDim: SmallDim, Conn>
where
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    energy: EnergySpace<T>,
    wavevector: WavevectorSpace<T, GeometryDim, Conn>,
}

impl<T: Copy + RealField, GeometryDim: SmallDim, Conn> ScatteringSpectral<T, GeometryDim, Conn>
where
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn new(energy: EnergySpace<T>, wavevector: WavevectorSpace<T, GeometryDim, Conn>) -> Self {
        Self { energy, wavevector }
    }
}

impl<T: RealField + Copy, GeometryDim: SmallDim, Conn> SpectralDiscretisation<T>
    for ScatteringSpectral<T, GeometryDim, Conn>
where
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn number_of_energy_points(&self) -> usize {
        self.energy.num_points()
    }
    fn number_of_wavevector_points(&self) -> usize {
        // TODO implement in a blanket mesh method
        self.energy.num_points()
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
}

impl<T: RealField + Copy> SpectralDiscretisation<T> for BallisticSpectral<T> {
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

    // fn iter_all(&self) -> std::slice::Iter<'_, (T, T)> {
    //     self.energy.points().map(|x| (T::zero(), x))
    // }
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

impl<T> GenerateWeights<T, U1, Segment1dConnectivity> for IntegrationRule
where
    T: Copy + RealField,
    DefaultAllocator: Allocator<T, U1>,
{
    fn query_integration_rule(&self) -> IntegrationRule {
        *self
    }
    fn generate_weights_from_grid(&self, grid: &Mesh<T, U1, Segment1dConnectivity>) -> DVector<T> {
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
