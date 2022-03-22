use super::GenerateWeights;
use nalgebra::{DVector, RealField, U1};
use num_traits::NumCast;
use std::ops::Range;
use transporter_mesher::{
    create_line_segment_from_endpoints_and_number_of_points, Mesh1d, Segment1dConnectivity,
};

pub(crate) struct EnergySpaceBuilder<T, EnergyRange, IntegrationMethod> {
    number_of_points: usize,
    energy_range: EnergyRange,
    integration_rule: IntegrationMethod,
    marker: std::marker::PhantomData<T>,
}

impl EnergySpaceBuilder<(), (), ()> {
    fn new() -> Self {
        Self {
            number_of_points: 0,
            energy_range: (),
            integration_rule: (),
            marker: std::marker::PhantomData,
        }
    }
}

impl<EnergyRange, IntegrationMethod> EnergySpaceBuilder<(), EnergyRange, IntegrationMethod> {
    fn with_integration_rule<IntegrationRule>(
        self,
        integration_rule: IntegrationRule,
    ) -> EnergySpaceBuilder<(), EnergyRange, IntegrationRule> {
        EnergySpaceBuilder {
            number_of_points: self.number_of_points,
            energy_range: self.energy_range,
            integration_rule,
            marker: std::marker::PhantomData,
        }
    }

    fn with_number_of_points(self, number_of_points: usize) -> Self {
        EnergySpaceBuilder {
            number_of_points,
            energy_range: self.energy_range,
            integration_rule: self.integration_rule,
            marker: std::marker::PhantomData,
        }
    }

    fn with_energy_range<T>(
        self,
        energy_range: Range<T>,
    ) -> EnergySpaceBuilder<(), Range<T>, IntegrationMethod> {
        EnergySpaceBuilder {
            number_of_points: self.number_of_points,
            energy_range,
            integration_rule: self.integration_rule,
            marker: std::marker::PhantomData,
        }
    }
}

pub(crate) struct EnergySpace<T: RealField, IntegrationRule> {
    grid: Mesh1d<T>,
    weights: DVector<T>,
    integration_rule: IntegrationRule,
}

impl<T, IntegrationRule> EnergySpaceBuilder<T, Range<T>, IntegrationRule>
where
    T: Copy + RealField + NumCast,
    IntegrationRule: GenerateWeights<T, U1, Segment1dConnectivity>,
{
    fn build(self) -> EnergySpace<T, IntegrationRule> {
        // Build the energy mesh in meV
        assert!(self.energy_range.end > self.energy_range.start); // Need an order range, or something upsteam went wrong
        let grid = create_line_segment_from_endpoints_and_number_of_points(
            self.energy_range,
            self.number_of_points,
        );
        let weights = self.integration_rule.generate_weights_from_grid(&grid);
        EnergySpace {
            grid,
            weights,
            integration_rule: self.integration_rule,
        }
    }
}
