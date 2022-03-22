/// This module provides the discrete energy and wavevector spaces necessary to
/// scaffold the Green's functions and their relation to the electron density
mod energy;
mod wavevector;

use energy::EnergySpace;
use nalgebra::allocator::Allocator;
use nalgebra::{DVector, DefaultAllocator, DimName, RealField, U1};
use transporter_mesher::{Mesh, Segment1dConnectivity};

pub(crate) struct SpectralDiscretisation<T: RealField> {
    energy: EnergySpace<T, IntegrationRule>,
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

pub(crate) trait GenerateWeights<T, GeometryDim, C>
where
    T: RealField,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn generate_weights_from_grid(&self, grid: &Mesh<T, GeometryDim, C>) -> DVector<T>;
}

impl<T> GenerateWeights<T, U1, Segment1dConnectivity> for IntegrationRule
where
    T: RealField,
    DefaultAllocator: Allocator<T, U1>,
{
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
