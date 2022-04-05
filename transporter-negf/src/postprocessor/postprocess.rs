use super::{ChargeAndCurrent, PostProcessor};
use crate::{
    greens_functions::AggregateGreensFunctionMethods,
    spectral::{SpectralDiscretisation, SpectralSpace},
};
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dynamic, Matrix, RealField, VecStorage,
};
use transporter_mesher::{Connectivity, SmallDim};

pub(crate) trait PostProcess<T, BandDim: SmallDim, Spectral>
where
    T: RealField + Copy,
    Spectral: SpectralDiscretisation<T>,
    DefaultAllocator: Allocator<
        Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
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

impl<T, GeometryDim, Conn, BandDim> PostProcess<T, BandDim, SpectralSpace<T, ()>>
    for PostProcessor<'_, T, GeometryDim, Conn>
where
    T: RealField + Copy,
    Conn: Connectivity<T, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    fn recompute_currents_and_densities<AggregateGreensFunctions>(
        &self,
        greens_functions: &AggregateGreensFunctions,
        spectral_discretisation: &SpectralSpace<T, ()>,
    ) -> color_eyre::Result<ChargeAndCurrent<T, BandDim>>
    where
        AggregateGreensFunctions: AggregateGreensFunctionMethods<T, BandDim, SpectralSpace<T, ()>>,
    {
        //todo Do we want to get the LDOS or are we ok with doing this inside the Greens funciton itself
        let charge = greens_functions
            .accumulate_into_charge_density_vector(self.mesh, spectral_discretisation)?;
        let current = greens_functions
            .accumulate_into_current_density_vector(self.mesh, spectral_discretisation)?;
        Ok(ChargeAndCurrent::from_charge_and_current(charge, current))
    }
}
