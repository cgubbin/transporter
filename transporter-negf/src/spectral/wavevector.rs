//! # Wavevector
//!
//! The wavevector space utilised to run a simulation. This is created based on the configuration
//! with a fixed number `number_of_points` nodes linearly distributed from zero wavevector to the provided
//! cut-off `maximum_wavevector`. The wavevector grid inherits the `GeometryDim` from the simulation

use super::GenerateWeights;
use nalgebra::{allocator::Allocator, DefaultAllocator, OPoint, RealField, U1};
use ndarray::Array1;
use num_traits::NumCast;
use transporter_mesher::{
    create_line_segment_from_endpoints_and_number_of_points, Connectivity, Mesh,
    Segment1dConnectivity, SmallDim,
};

pub(crate) struct WavevectorSpaceBuilder<T, IntegrationMethod, GeometryDim> {
    number_of_points: Option<usize>,
    maximum_wavevector: Option<T>,
    integration_rule: IntegrationMethod,
    marker: std::marker::PhantomData<GeometryDim>,
}

impl<T, GeometryDim> WavevectorSpaceBuilder<T, (), GeometryDim> {
    pub(crate) fn new() -> Self {
        Self {
            number_of_points: None,
            maximum_wavevector: None,
            integration_rule: (),
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, IntegrationMethod, GeometryDim> WavevectorSpaceBuilder<T, IntegrationMethod, GeometryDim>
where
    GeometryDim: SmallDim,
{
    pub(crate) fn with_integration_rule<IntegrationRule>(
        self,
        integration_rule: IntegrationRule,
    ) -> WavevectorSpaceBuilder<T, IntegrationRule, GeometryDim> {
        WavevectorSpaceBuilder {
            number_of_points: self.number_of_points,
            maximum_wavevector: self.maximum_wavevector,
            integration_rule,
            marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_number_of_points(self, number_of_points: usize) -> Self {
        WavevectorSpaceBuilder {
            number_of_points: Some(number_of_points),
            maximum_wavevector: self.maximum_wavevector,
            integration_rule: self.integration_rule,
            marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_maximum_wavevector(
        self,
        maximum_wavevector: T,
    ) -> WavevectorSpaceBuilder<T, IntegrationMethod, GeometryDim> {
        WavevectorSpaceBuilder {
            number_of_points: self.number_of_points,
            maximum_wavevector: Some(maximum_wavevector),
            integration_rule: self.integration_rule,
            marker: std::marker::PhantomData,
        }
    }
}

pub(crate) struct WavevectorSpace<T: RealField, GeometryDim, Conn>
where
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub(crate) grid: Mesh<T, GeometryDim, Conn>,
    weights: Array1<T>,
    integration_rule: super::IntegrationRule,
}

pub(crate) trait BuildWavevectorSpace<T, IntegrationRule, GeometryDim: SmallDim, Conn>
where
    T: Copy + RealField + NumCast,
    IntegrationRule: GenerateWeights<T, GeometryDim, Conn>,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn build(self) -> WavevectorSpace<T, GeometryDim, Conn>;
}

impl<T, IntegrationRule> BuildWavevectorSpace<T, IntegrationRule, U1, Segment1dConnectivity>
    for WavevectorSpaceBuilder<T, IntegrationRule, U1>
where
    T: Copy + RealField + NumCast,
    IntegrationRule: GenerateWeights<T, U1, Segment1dConnectivity>,
    DefaultAllocator: Allocator<T, U1>,
{
    fn build(self) -> WavevectorSpace<T, U1, Segment1dConnectivity> {
        // Build the wavevector mesh in meV
        let wavevector_range = std::ops::Range {
            start: T::zero(),
            end: self.maximum_wavevector.unwrap(),
        };
        let grid = create_line_segment_from_endpoints_and_number_of_points(
            wavevector_range,
            self.number_of_points.unwrap(),
            0,
        );
        let weights = self.integration_rule.generate_weights_from_grid(&grid);
        WavevectorSpace {
            grid,
            weights,
            integration_rule: self.integration_rule.query_integration_rule(),
        }
    }
}

impl<T: Copy + RealField, GeometryDim: SmallDim, Conn: Connectivity<T, GeometryDim>>
    WavevectorSpace<T, GeometryDim, Conn>
where
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub(crate) fn num_points(&self) -> usize {
        self.grid.vertices().len()
    }

    pub(crate) fn points(&self) -> impl Iterator<Item = &OPoint<T, GeometryDim>> + '_ {
        self.grid.vertices().iter().map(|x| &x.0)
    }

    pub(crate) fn weights(&self) -> impl Iterator<Item = &T> {
        self.weights.iter()
    }
}
