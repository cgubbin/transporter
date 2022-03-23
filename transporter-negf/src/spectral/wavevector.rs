use super::GenerateWeights;
use nalgebra::{DVector, RealField, U1};
use num_traits::NumCast;
use transporter_mesher::{
    create_line_segment_from_endpoints_and_number_of_points, Mesh1d, Segment1dConnectivity,
};

pub(crate) struct WavevectorSpaceBuilder<T, IntegrationMethod> {
    number_of_points: usize,
    maximum_wavevector: T,
    integration_rule: IntegrationMethod,
}

impl WavevectorSpaceBuilder<(), ()> {
    fn new() -> Self {
        Self {
            number_of_points: 0,
            maximum_wavevector: (),
            integration_rule: (),
        }
    }
}

impl<IntegrationMethod> WavevectorSpaceBuilder<(), IntegrationMethod> {
    fn with_integration_rule<IntegrationRule>(
        self,
        integration_rule: IntegrationRule,
    ) -> WavevectorSpaceBuilder<(), IntegrationRule> {
        WavevectorSpaceBuilder {
            number_of_points: self.number_of_points,
            maximum_wavevector: self.maximum_wavevector,
            integration_rule,
        }
    }

    fn with_number_of_points(self, number_of_points: usize) -> Self {
        WavevectorSpaceBuilder {
            number_of_points,
            maximum_wavevector: self.maximum_wavevector,
            integration_rule: self.integration_rule,
        }
    }

    fn with_maximum_wavevector<T>(
        self,
        maximum_wavevector: T,
    ) -> WavevectorSpaceBuilder<T, IntegrationMethod> {
        WavevectorSpaceBuilder {
            number_of_points: self.number_of_points,
            maximum_wavevector,
            integration_rule: self.integration_rule,
        }
    }
}

pub(crate) struct WavevectorSpace<T: RealField, IntegrationRule> {
    grid: Mesh1d<T>,
    weights: DVector<T>,
    integration_rule: IntegrationRule,
}

impl<T, IntegrationRule> WavevectorSpaceBuilder<T, IntegrationRule>
where
    T: Copy + RealField + NumCast,
    IntegrationRule: GenerateWeights<T, U1, Segment1dConnectivity>,
{
    fn build(self) -> WavevectorSpace<T, IntegrationRule> {
        // Build the energy mesh in meV
        let grid = create_line_segment_from_endpoints_and_number_of_points(
            std::ops::Range {
                start: T::zero(),
                end: self.maximum_wavevector,
            },
            self.number_of_points,
        );
        let weights = self.integration_rule.generate_weights_from_grid(&grid);
        WavevectorSpace {
            grid,
            weights,
            integration_rule: self.integration_rule,
        }
    }
}
