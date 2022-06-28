//! # Energy
//!
//! The energy space utilised to run a simulation. This is created based on the configuration
//! with a fixed number `number_of_points` nodes linearly distributed in the provided `energy_range`

use super::{GenerateWeights, IntegrationRule};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField, U1};
use ndarray::Array1;
use num_traits::NumCast;
use std::ops::Range;
use transporter_mesher::{
    create_line_segment_from_endpoints_and_number_of_points, create_line_segment_from_vertices,
    Connectivity, Mesh1d, Segment1dConnectivity, SmallDim,
};

pub(crate) struct EnergySpaceBuilder<T, EnergyRange, IntegrationMethod, RefEigenvalues> {
    number_of_points: usize,
    energy_range: EnergyRange,
    integration_rule: IntegrationMethod,
    eigenvalues: RefEigenvalues,
    marker: std::marker::PhantomData<T>,
}

impl<T> EnergySpaceBuilder<T, (), (), ()> {
    pub(crate) fn new() -> Self {
        Self {
            number_of_points: 0,
            energy_range: (),
            integration_rule: (),
            eigenvalues: (),
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, EnergyRange, IntegrationMethod, RefEigenvalues>
    EnergySpaceBuilder<T, EnergyRange, IntegrationMethod, RefEigenvalues>
{
    pub(crate) fn with_integration_rule<IntegrationRule>(
        self,
        integration_rule: IntegrationRule,
    ) -> EnergySpaceBuilder<T, EnergyRange, IntegrationRule, RefEigenvalues> {
        EnergySpaceBuilder {
            number_of_points: self.number_of_points,
            energy_range: self.energy_range,
            integration_rule,
            eigenvalues: self.eigenvalues,
            marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_number_of_points(self, number_of_points: usize) -> Self {
        EnergySpaceBuilder {
            number_of_points,
            energy_range: self.energy_range,
            integration_rule: self.integration_rule,
            eigenvalues: self.eigenvalues,
            marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_energy_range(
        self,
        energy_range: Range<T>,
    ) -> EnergySpaceBuilder<T, Range<T>, IntegrationMethod, RefEigenvalues> {
        EnergySpaceBuilder {
            number_of_points: self.number_of_points,
            energy_range,
            integration_rule: self.integration_rule,
            eigenvalues: self.eigenvalues,
            marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_eigenvalues<Eigenvalues>(
        self,
        eigenvalues: &Eigenvalues,
    ) -> EnergySpaceBuilder<T, EnergyRange, IntegrationMethod, &Eigenvalues> {
        EnergySpaceBuilder {
            number_of_points: self.number_of_points,
            energy_range: self.energy_range,
            integration_rule: self.integration_rule,
            eigenvalues,
            marker: std::marker::PhantomData,
        }
    }
}

#[derive(Debug)]
pub(crate) struct EnergySpace<T: Copy + RealField> {
    pub(crate) grid: Mesh1d<T>,
    weights: Array1<T>,
    widths: Array1<T>,
    pub(crate) integration_rule: super::IntegrationRule,
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

impl<T> EnergySpaceBuilder<T, Range<T>, IntegrationRule, ()>
where
    T: Copy + RealField + NumCast,
{
    pub(crate) fn build(self) -> EnergySpace<T> {
        // Build the energy mesh in meV
        assert!(self.energy_range.end > self.energy_range.start); // Need an order range, or something upsteam went wrong
                                                                  // Need the first cell to have zero, and the last cell end energy

        let (grid, weights, widths) = match self.integration_rule {
            IntegrationRule::GaussKronrod => {
                let integrator = quad_rs::GaussKronrod::default();
                let (vertices, weights) =
                    integrator.generate(self.energy_range, None, self.number_of_points);

                let vertices = vertices.iter().map(|x| *x).collect::<Vec<_>>();
                let widths = weights.iter().map(|x| *x).collect::<Array1<_>>();
                let grid = create_line_segment_from_vertices(vertices);
                let weights = Array1::ones(grid.elements().len());
                (grid, weights, widths)
            }
            _ => {
                let grid = create_line_segment_from_endpoints_and_number_of_points(
                    self.energy_range,
                    self.number_of_points,
                    0,
                );
                let weights = self.integration_rule.generate_weights_from_grid(&grid);
                let widths = grid
                    .elements()
                    .iter()
                    .map(|element| element.0.diameterb())
                    .collect::<Array1<T>>();
                (grid, weights, widths)
            }
        };
        EnergySpace {
            grid,
            weights,
            widths,
            integration_rule: self.integration_rule,
        }
    }
}

use quad_rs::Generate;

impl<T> EnergySpaceBuilder<T, Range<T>, IntegrationRule, &Array1<T>>
where
    T: Copy + RealField + NumCast,
{
    pub(crate) fn build(self) -> EnergySpace<T> {
        // Build the energy mesh in meV
        assert!(self.energy_range.end > self.energy_range.start); // Need an ordered range, or something upsteam went wrong
                                                                  // Need the first cell to have zero, and the last cell end energy

        let (grid, weights, widths) = match self.integration_rule {
            IntegrationRule::GaussKronrod => {
                let integrator = quad_rs::GaussKronrod::default();
                let (vertices, weights) =
                    integrator.generate(self.energy_range, None, self.number_of_points);

                let vertices = vertices.iter().map(|x| *x).collect::<Vec<_>>();
                let widths = weights.iter().map(|x| *x).collect::<Array1<_>>();
                let grid = create_line_segment_from_vertices(vertices);
                let weights = Array1::ones(grid.elements().len());
                (grid, weights, widths)
            }
            _ => {
                let grid = create_line_segment_from_endpoints_and_number_of_points(
                    self.energy_range,
                    self.number_of_points,
                    0,
                );
                let weights = self.integration_rule.generate_weights_from_grid(&grid);
                let widths = grid
                    .elements()
                    .iter()
                    .map(|element| element.0.diameterb())
                    .collect::<Array1<T>>();
                (grid, weights, widths)
            }
        };

        EnergySpace {
            grid,
            weights,
            widths,
            integration_rule: self.integration_rule,
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

    pub(crate) fn widths(&self) -> impl Iterator<Item = &T> {
        self.widths.iter()
    }
}
