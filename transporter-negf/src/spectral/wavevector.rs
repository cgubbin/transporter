use super::GenerateWeights;
use nalgebra::{allocator::Allocator, DefaultAllocator};
use nalgebra::{DVector, OPoint, RealField, U1};
use num_traits::NumCast;
use transporter_mesher::{Connectivity, Mesh, Segment1dConnectivity, SmallDim};

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
    grid: Mesh<T, GeometryDim, Conn>,
    weights: DVector<T>,
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

impl<T, IntegrationRule, GeometryDim: SmallDim, Conn>
    BuildWavevectorSpace<T, IntegrationRule, GeometryDim, Conn>
    for WavevectorSpaceBuilder<T, IntegrationRule, GeometryDim>
where
    T: Copy + RealField + NumCast,
    Conn: Connectivity<T, GeometryDim>,
    IntegrationRule: GenerateWeights<T, GeometryDim, Conn>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn build(self) -> WavevectorSpace<T, GeometryDim, Conn> {
        todo!()
    }
}

//impl<T, IntegrationRule> BuildWavevectorSpace<T, IntegrationRule, U1, Segment1dConnectivity>
//    for WavevectorSpaceBuilder<T, IntegrationRule, U1>
//where
//    T: Copy + RealField + NumCast,
//    IntegrationRule: GenerateWeights<T, U1, Segment1dConnectivity>,
//{
//    fn build(self) -> WavevectorSpace<T, IntegrationRule, U1, Segment1dConnectivity> {
//        // Build the energy mesh in meV
//        let grid = create_line_segment_from_endpoints_and_number_of_points(
//            std::ops::Range {
//                start: T::zero(),
//                end: self.maximum_wavevector.unwrap(),
//            },
//            self.number_of_points.unwrap(),
//        );
//        let weights = self.integration_rule.generate_weights_from_grid(&grid);
//        WavevectorSpace {
//            grid,
//            weights,
//            integration_rule: self.integration_rule,
//        }
//    }
//}

//impl<T, IntegrationRule> WavevectorSpaceBuilder<T, IntegrationRule, U1>
//where
//    T: Copy + RealField + NumCast,
//    IntegrationRule: GenerateWeights<T, U1, Segment1dConnectivity>,
//{
//    pub(crate) fn build(self) -> WavevectorSpace<T, IntegrationRule, U1, Segment1dConnectivity> {
//        // Build the energy mesh in meV
//        let grid = create_line_segment_from_endpoints_and_number_of_points(
//            std::ops::Range {
//                start: T::zero(),
//                end: self.maximum_wavevector.unwrap(),
//            },
//            self.number_of_points.unwrap(),
//        );
//        let weights = self.integration_rule.generate_weights_from_grid(&grid);
//        WavevectorSpace {
//            grid,
//            weights,
//            integration_rule: self.integration_rule,
//        }
//    }
//}

impl<T: Copy + RealField> WavevectorSpace<T, U1, Segment1dConnectivity> {
    pub(crate) fn num_points(&self) -> usize {
        self.grid.vertices().len()
    }

    pub(crate) fn points(&self) -> impl Iterator<Item = &OPoint<T, U1>> + '_ {
        self.grid.vertices().iter().map(|x| &x.0)
    }

    pub(crate) fn weights(&self) -> impl Iterator<Item = &T> {
        self.weights.iter()
    }
}
