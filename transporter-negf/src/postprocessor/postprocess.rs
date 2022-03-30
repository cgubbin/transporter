use super::{ChargeAndCurrent, PostProcessor};
use crate::greens_functions::AggregateGreensFunctionMethods;
use crate::spectral::SpectralDiscretisation;
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator};
use transporter_mesher::{Connectivity, SmallDim};

pub(crate) trait PostProcess<T>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    fn recompute_currents_and_densities<AggregateGreensFunctions, Spectral>(
        &self,
        greens_functions: &AggregateGreensFunctions,
        spectral_discretisation: &Spectral,
    ) -> color_eyre::Result<ChargeAndCurrent<T::RealField>>
    where
        AggregateGreensFunctions: AggregateGreensFunctionMethods<T, Spectral>,
        Spectral: SpectralDiscretisation<T::RealField>;
}

impl<T, GeometryDim, Conn> PostProcess<T> for PostProcessor<'_, T, GeometryDim, Conn>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    Conn: Connectivity<T::RealField, GeometryDim>,
    GeometryDim: SmallDim,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    fn recompute_currents_and_densities<AggregateGreensFunctions, Spectral>(
        &self,
        greens_functions: &AggregateGreensFunctions,
        spectral_discretisation: &Spectral,
    ) -> color_eyre::Result<ChargeAndCurrent<T::RealField>>
    where
        AggregateGreensFunctions: AggregateGreensFunctionMethods<T, Spectral>,
        Spectral: SpectralDiscretisation<T::RealField>,
    {
        //todo Do we want to get the LDOS or are we ok with doing this inside the Greens funciton itself
        let charge =
            greens_functions.accumulate_into_charge_density_vector(spectral_discretisation);
        let current =
            greens_functions.accumulate_into_current_density_vector(spectral_discretisation);
        Ok(ChargeAndCurrent::from_charge_and_current(charge, current))
    }
}
