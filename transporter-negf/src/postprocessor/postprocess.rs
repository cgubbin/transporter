use super::{ChargeAndCurrent, PostProcessor, PostProcessorError};
use crate::{
    greens_functions::AggregateGreensFunctionMethods, self_energy::SelfEnergy,
    spectral::SpectralDiscretisation,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use ndarray::Array1;
use transporter_mesher::{Connectivity, SmallDim};

pub(crate) trait PostProcess<T, BandDim: SmallDim, GeometryDim, Conn, Spectral, SelfEnergy>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    Spectral: SpectralDiscretisation<T>,
    DefaultAllocator: Allocator<Array1<T>, BandDim> + Allocator<T, GeometryDim>,
{
    fn recompute_currents_and_densities<AggregateGreensFunctions>(
        &self,
        voltage: T,
        greens_functions: &AggregateGreensFunctions,
        self_energy: &SelfEnergy,
        spectral_discretisation: &Spectral,
    ) -> Result<ChargeAndCurrent<T::RealField, BandDim>, PostProcessorError>
    where
        AggregateGreensFunctions:
            AggregateGreensFunctionMethods<T, BandDim, GeometryDim, Conn, Spectral, SelfEnergy>;
}

pub(crate) trait PostProcessLOGenerationRate<
    T: RealField + Copy,
    BandDim: SmallDim,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    Spectral,
    SelfEnergy,
> where
    Spectral: SpectralDiscretisation<T>,
    DefaultAllocator: Allocator<Array1<T>, BandDim> + Allocator<T, GeometryDim>,
{
    fn compute_momentum_resolved_lo_generation_rate<AggregateGreensFunctions>(
        &self,
        greens_functions: &AggregateGreensFunctions,
        self_energy: &SelfEnergy,
        spectral_discretisation: &Spectral,
    ) -> Result<nalgebra::DVector<T::RealField>, PostProcessorError>
    where
        AggregateGreensFunctions:
            AggregateGreensFunctionMethods<T, BandDim, GeometryDim, Conn, Spectral, SelfEnergy>;

    fn compute_total_lo_generation_rate<AggregateGreensFunctions>(
        &self,
        greens_functions: &AggregateGreensFunctions,
        self_energy: &SelfEnergy,
        spectral_discretisation: &Spectral,
    ) -> Result<T::RealField, PostProcessorError>
    where
        AggregateGreensFunctions:
            AggregateGreensFunctionMethods<T, BandDim, GeometryDim, Conn, Spectral, SelfEnergy>,
    {
        // TODO Add weightings for integration
        Ok(self
            .compute_momentum_resolved_lo_generation_rate(
                greens_functions,
                self_energy,
                spectral_discretisation,
            )?
            .iter()
            .fold(T::zero(), |acc, &x| acc + x))
    }
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
    DefaultAllocator: Allocator<T, GeometryDim> + Allocator<Array1<T>, BandDim>,
{
    fn recompute_currents_and_densities<AggregateGreensFunctions>(
        &self,
        voltage: T,
        greens_functions: &AggregateGreensFunctions,
        self_energy: &SelfEnergy<T, GeometryDim, Conn>,
        spectral_discretisation: &Spectral,
    ) -> Result<ChargeAndCurrent<T, BandDim>, PostProcessorError>
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
            voltage,
            self.mesh,
            self_energy,
            spectral_discretisation,
        )?;
        Ok(ChargeAndCurrent::from_charge_and_current(charge, current))
    }
}
