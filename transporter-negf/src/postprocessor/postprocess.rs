use super::{ChargeAndCurrent, PostProcessor};
use crate::greens_functions::AggregateGreensFunctionMethods;
use crate::spectral::{SpectralDiscretisation, SpectralSpace};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator};
use nalgebra::{Const, Dynamic, Matrix, VecStorage};
use transporter_mesher::{Connectivity, SmallDim};

pub(crate) trait PostProcess<T, BandDim: SmallDim, Spectral>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    Spectral: SpectralDiscretisation<T::RealField>,
    DefaultAllocator: Allocator<
        Matrix<
            T::RealField,
            Dynamic,
            Const<1_usize>,
            VecStorage<T::RealField, Dynamic, Const<1_usize>>,
        >,
        BandDim,
    >,
{
    fn recompute_currents_and_densities<AggregateGreensFunctions>(
        &self,
        greens_functions: &AggregateGreensFunctions,
        spectral_discretisation: &Spectral,
    ) -> color_eyre::Result<ChargeAndCurrent<T::RealField, BandDim>>
    where
        AggregateGreensFunctions: AggregateGreensFunctionMethods<T, BandDim, Spectral>;
}

impl<T, GeometryDim, Conn, BandDim> PostProcess<T, BandDim, SpectralSpace<T::RealField, ()>>
    for PostProcessor<'_, T, GeometryDim, Conn>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    Conn: Connectivity<T::RealField, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>
        + Allocator<
            Matrix<
                T::RealField,
                Dynamic,
                Const<1_usize>,
                VecStorage<T::RealField, Dynamic, Const<1_usize>>,
            >,
            BandDim,
        >,
{
    fn recompute_currents_and_densities<AggregateGreensFunctions>(
        &self,
        greens_functions: &AggregateGreensFunctions,
        spectral_discretisation: &SpectralSpace<T::RealField, ()>,
    ) -> color_eyre::Result<ChargeAndCurrent<T::RealField, BandDim>>
    where
        AggregateGreensFunctions:
            AggregateGreensFunctionMethods<T, BandDim, SpectralSpace<T::RealField, ()>>,
    {
        //todo Do we want to get the LDOS or are we ok with doing this inside the Greens funciton itself
        let charge =
            greens_functions.accumulate_into_charge_density_vector(spectral_discretisation)?;
        let current =
            greens_functions.accumulate_into_current_density_vector(spectral_discretisation)?;
        Ok(ChargeAndCurrent::from_charge_and_current(charge, current))
    }
}
