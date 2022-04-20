use super::{ChargeAndCurrent, PostProcessor};
use crate::{
    greens_functions::AggregateGreensFunctionMethods, self_energy::SelfEnergy,
    spectral::SpectralDiscretisation,
};
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dynamic, Matrix, RealField, VecStorage,
};
use nalgebra_sparse::CsrMatrix;
use num_complex::Complex;
use transporter_mesher::{Connectivity, SmallDim};

pub(crate) trait PostProcess<T, BandDim: SmallDim, GeometryDim, Conn, Spectral, SelfEnergy>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    Spectral: SpectralDiscretisation<T>,
    DefaultAllocator: Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, GeometryDim>,
{
    fn recompute_currents_and_densities<AggregateGreensFunctions>(
        &self,
        greens_functions: &AggregateGreensFunctions,
        self_energy: &SelfEnergy,
        spectral_discretisation: &Spectral,
    ) -> color_eyre::Result<ChargeAndCurrent<T::RealField, BandDim>>
    where
        AggregateGreensFunctions:
            AggregateGreensFunctionMethods<T, BandDim, GeometryDim, Conn, Spectral, SelfEnergy>;
}

impl<T, GeometryDim, Conn, BandDim, Spectral>
    PostProcess<T, BandDim, GeometryDim, Conn, Spectral, SelfEnergy<T, GeometryDim, Conn>>
    for PostProcessor<'_, T, GeometryDim, Conn>
where
    T: RealField + Copy,
    Conn: Connectivity<T, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Spectral: SpectralDiscretisation<T>,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    fn recompute_currents_and_densities<AggregateGreensFunctions>(
        &self,
        greens_functions: &AggregateGreensFunctions,
        self_energy: &SelfEnergy<T, GeometryDim, Conn>,
        spectral_discretisation: &Spectral,
    ) -> color_eyre::Result<ChargeAndCurrent<T, BandDim>>
    where
        AggregateGreensFunctions: AggregateGreensFunctionMethods<
            T,
            BandDim,
            GeometryDim,
            Conn,
            Spectral,
            SelfEnergy<T, GeometryDim, Conn>,
        >,
    {
        //todo Do we want to get the LDOS or are we ok with doing this inside the Greens funciton itself
        let charge = greens_functions
            .accumulate_into_charge_density_vector(self.mesh, spectral_discretisation)?;
        let current = greens_functions.accumulate_into_current_density_vector(
            self.mesh,
            self_energy,
            spectral_discretisation,
        )?;
        Ok(ChargeAndCurrent::from_charge_and_current(charge, current))
    }
}
