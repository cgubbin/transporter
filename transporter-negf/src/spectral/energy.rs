use super::GenerateWeights;
use nalgebra::{allocator::Allocator, DVector, DefaultAllocator, RealField, U1};
use num_traits::NumCast;
use std::ops::Range;
use transporter_mesher::{
    create_line_segment_from_endpoints_and_number_of_points, Connectivity, Mesh1d,
    Segment1dConnectivity, SmallDim,
};

pub(crate) struct EnergySpaceBuilder<T, EnergyRange, IntegrationMethod> {
    number_of_points: usize,
    energy_range: EnergyRange,
    integration_rule: IntegrationMethod,
    marker: std::marker::PhantomData<T>,
}

impl<T> EnergySpaceBuilder<T, (), ()> {
    pub(crate) fn new() -> Self {
        Self {
            number_of_points: 0,
            energy_range: (),
            integration_rule: (),
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, EnergyRange, IntegrationMethod> EnergySpaceBuilder<T, EnergyRange, IntegrationMethod> {
    pub(crate) fn with_integration_rule<IntegrationRule>(
        self,
        integration_rule: IntegrationRule,
    ) -> EnergySpaceBuilder<T, EnergyRange, IntegrationRule> {
        EnergySpaceBuilder {
            number_of_points: self.number_of_points,
            energy_range: self.energy_range,
            integration_rule,
            marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_number_of_points(self, number_of_points: usize) -> Self {
        EnergySpaceBuilder {
            number_of_points,
            energy_range: self.energy_range,
            integration_rule: self.integration_rule,
            marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_energy_range(
        self,
        energy_range: Range<T>,
    ) -> EnergySpaceBuilder<T, Range<T>, IntegrationMethod> {
        EnergySpaceBuilder {
            number_of_points: self.number_of_points,
            energy_range,
            integration_rule: self.integration_rule,
            marker: std::marker::PhantomData,
        }
    }
}

#[derive(Debug)]
pub(crate) struct EnergySpace<T: Copy + RealField> {
    pub(crate) grid: Mesh1d<T>,
    weights: DVector<T>,
    integration_rule: super::IntegrationRule,
}

pub(crate) trait BuildEnergySpace<T, IntegrationRule, GeometryDim: SmallDim, Conn>
where
    T: Copy + RealField + NumCast,
    Conn: Connectivity<T, GeometryDim>,
    IntegrationRule: GenerateWeights<T, U1, Segment1dConnectivity>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn build(self) -> EnergySpace<T>;
}

impl<T, IntegrationRule> EnergySpaceBuilder<T, Range<T>, IntegrationRule>
where
    T: Copy + RealField + NumCast,
    IntegrationRule: GenerateWeights<T, U1, Segment1dConnectivity>,
{
    pub(crate) fn build(self) -> EnergySpace<T> {
        // Build the energy mesh in meV
        assert!(self.energy_range.end > self.energy_range.start); // Need an order range, or something upsteam went wrong
                                                                  // Need the first cell to have zero, and the last cell end energy
        let cell_width = T::zero() * (self.energy_range.end - self.energy_range.start)
            / T::from_usize(self.number_of_points).unwrap();
        let energy_range = std::ops::Range {
            start: self.energy_range.start - cell_width / (T::one() + T::one()),
            end: self.energy_range.end - cell_width / (T::one() + T::one()),
        };
        let grid = create_line_segment_from_endpoints_and_number_of_points(
            energy_range,
            self.number_of_points,
            0,
        );
        let weights = self.integration_rule.generate_weights_from_grid(&grid);
        EnergySpace {
            grid,
            weights,
            integration_rule: self.integration_rule.query_integration_rule(),
        }
    }
}

impl<T: Copy + RealField> EnergySpace<T> {
    pub(crate) fn num_points(&self) -> usize {
        self.grid.vertices().len()
    }

    pub(crate) fn points(&self) -> impl Iterator<Item = &T> + '_ {
        self.grid.vertices().iter().map(|x| &x.0[0])
    }

    pub(crate) fn weights(&self) -> impl Iterator<Item = &T> {
        self.weights.iter()
    }
}
