use super::{
    energy::EnergySpaceBuilder, wavevector::BuildWavevectorSpace, GenerateWeights, SpectralSpace,
    WavevectorSpace,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField, U1};
use num_traits::NumCast;
use std::ops::Range;
use transporter_mesher::{Connectivity, Segment1dConnectivity, SmallDim};

pub(crate) struct SpectralSpaceBuilder<
    T,
    RefEnergyRange,
    RefEnergyIntegrationMethod,
    RefWavevectorIntegrationMethod,
    GeometryDim,
    Conn,
> {
    number_of_energy_points: Option<usize>,
    energy_range: RefEnergyRange,
    number_of_wavevector_points: Option<usize>,
    maximum_wavevector: Option<T>,
    energy_integration_rule: RefEnergyIntegrationMethod,
    wavevector_integration_rule: RefWavevectorIntegrationMethod,
    _marker: std::marker::PhantomData<GeometryDim>,
    __marker: std::marker::PhantomData<Conn>,
}

impl<T> SpectralSpaceBuilder<T, (), (), (), (), ()> {
    pub(crate) fn new() -> Self {
        Self {
            number_of_energy_points: None,
            energy_range: (),
            number_of_wavevector_points: None,
            maximum_wavevector: None,
            energy_integration_rule: (),
            wavevector_integration_rule: (),
            _marker: std::marker::PhantomData,
            __marker: std::marker::PhantomData,
        }
    }
}

impl<
        T,
        RefEnergyRange,
        RefEnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
        GeometryDim,
        Conn,
    >
    SpectralSpaceBuilder<
        T,
        RefEnergyRange,
        RefEnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
        GeometryDim,
        Conn,
    >
{
    pub(crate) fn with_number_of_energy_points(self, number_of_energy_points: usize) -> Self {
        Self {
            number_of_energy_points: Some(number_of_energy_points),
            energy_range: self.energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            _marker: std::marker::PhantomData,
            __marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_number_of_wavevector_points(
        self,
        number_of_wavevector_points: usize,
    ) -> Self {
        Self {
            number_of_energy_points: self.number_of_energy_points,
            energy_range: self.energy_range,
            number_of_wavevector_points: Some(number_of_wavevector_points),
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            _marker: std::marker::PhantomData,
            __marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_maximum_wavevector(
        self,
        maximum_wavevector: T,
    ) -> SpectralSpaceBuilder<
        T,
        RefEnergyRange,
        RefEnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
        GeometryDim,
        Conn,
    > {
        SpectralSpaceBuilder {
            number_of_energy_points: self.number_of_energy_points,
            energy_range: self.energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: Some(maximum_wavevector),
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            _marker: std::marker::PhantomData,
            __marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_energy_range(
        self,
        energy_range: Range<T>,
    ) -> SpectralSpaceBuilder<
        T,
        Range<T>,
        RefEnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
        GeometryDim,
        Conn,
    > {
        SpectralSpaceBuilder {
            number_of_energy_points: self.number_of_energy_points,
            energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            _marker: std::marker::PhantomData,
            __marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_energy_integration_method<EnergyIntegrationMethod>(
        self,
        energy_integration_rule: EnergyIntegrationMethod,
    ) -> SpectralSpaceBuilder<
        T,
        RefEnergyRange,
        EnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
        GeometryDim,
        Conn,
    > {
        SpectralSpaceBuilder {
            number_of_energy_points: self.number_of_energy_points,
            energy_range: self.energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            _marker: std::marker::PhantomData,
            __marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_wavevector_integration_method<WavevectorIntegrationMethod>(
        self,
        wavevector_integration_rule: WavevectorIntegrationMethod,
    ) -> SpectralSpaceBuilder<
        T,
        RefEnergyRange,
        RefEnergyIntegrationMethod,
        WavevectorIntegrationMethod,
        GeometryDim,
        Conn,
    > {
        SpectralSpaceBuilder {
            number_of_energy_points: self.number_of_energy_points,
            energy_range: self.energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule,
            _marker: std::marker::PhantomData,
            __marker: std::marker::PhantomData,
        }
    }
}

impl<T, EnergyIntegrationRule> SpectralSpaceBuilder<T, Range<T>, EnergyIntegrationRule, (), (), ()>
where
    T: Copy + RealField + NumCast,
    EnergyIntegrationRule: super::GenerateWeights<T, U1, Segment1dConnectivity>,
{
    pub(crate) fn build_coherent(self) -> SpectralSpace<T, ()> {
        let energy = super::energy::EnergySpaceBuilder::new()
            .with_integration_rule(self.energy_integration_rule)
            .with_number_of_points(self.number_of_energy_points.unwrap())
            .with_energy_range(self.energy_range)
            .build();
        SpectralSpace {
            energy,
            wavevector: (),
        }
    }
}

impl<T, Conn, GeometryDim, WavevectorIntegrationRule, EnergyIntegrationRule>
    SpectralSpaceBuilder<
        T,
        Range<T>,
        EnergyIntegrationRule,
        WavevectorIntegrationRule,
        GeometryDim,
        Conn,
    >
where
    T: Copy + RealField + NumCast,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    EnergyIntegrationRule: GenerateWeights<T, U1, Segment1dConnectivity>,
    WavevectorIntegrationRule: GenerateWeights<T, GeometryDim, Conn>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub(crate) fn build_incoherent(
        self,
    ) -> SpectralSpace<T, WavevectorSpace<T, GeometryDim, Conn>> {
        let energy = EnergySpaceBuilder::new()
            .with_integration_rule(self.energy_integration_rule)
            .with_number_of_points(self.number_of_energy_points.unwrap())
            .with_energy_range(self.energy_range)
            .build();
        let wavevector = super::wavevector::WavevectorSpaceBuilder::new()
            .with_integration_rule(self.wavevector_integration_rule)
            .with_number_of_points(self.number_of_wavevector_points.unwrap())
            .with_maximum_wavevector(self.maximum_wavevector.unwrap())
            .build();
        SpectralSpace { energy, wavevector }
    }
}
