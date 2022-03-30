use super::wavevector::BuildWavevectorSpace;
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField, U1};
use num_traits::NumCast;
use std::ops::Range;
use transporter_mesher::{Connectivity, Segment1dConnectivity, SmallDim};

pub(crate) struct SpectralSpaceBuilder<
    T,
    RefEnergyRange,
    GeometryDim,
    RefEnergyIntegrationMethod,
    RefWavevectorIntegrationMethod,
> {
    number_of_energy_points: Option<usize>,
    energy_range: RefEnergyRange,
    number_of_wavevector_points: Option<usize>,
    maximum_wavevector: Option<T>,
    energy_integration_rule: RefEnergyIntegrationMethod,
    wavevector_integration_rule: RefWavevectorIntegrationMethod,
    marker: std::marker::PhantomData<GeometryDim>,
}

impl<T> SpectralSpaceBuilder<T, (), (), (), ()> {
    pub(crate) fn new() -> Self {
        Self {
            number_of_energy_points: None,
            energy_range: (),
            number_of_wavevector_points: None,
            maximum_wavevector: None,
            energy_integration_rule: (),
            wavevector_integration_rule: (),
            marker: std::marker::PhantomData,
        }
    }
}

impl<
        T,
        RefEnergyRange,
        GeometryDim,
        RefEnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
    >
    SpectralSpaceBuilder<
        T,
        RefEnergyRange,
        GeometryDim,
        RefEnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
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
            marker: std::marker::PhantomData,
        }
    }

    fn with_number_of_wavevector_points(self, number_of_wavevector_points: usize) -> Self {
        Self {
            number_of_energy_points: self.number_of_energy_points,
            energy_range: self.energy_range,
            number_of_wavevector_points: Some(number_of_wavevector_points),
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            marker: std::marker::PhantomData,
        }
    }

    fn with_maximum_wavevector(
        self,
        maximum_wavevector: T,
    ) -> SpectralSpaceBuilder<
        T,
        RefEnergyRange,
        GeometryDim,
        RefEnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
    > {
        SpectralSpaceBuilder {
            number_of_energy_points: self.number_of_energy_points,
            energy_range: self.energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: Some(maximum_wavevector),
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_energy_range(
        self,
        energy_range: Range<T>,
    ) -> SpectralSpaceBuilder<
        T,
        Range<T>,
        GeometryDim,
        RefEnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
    > {
        SpectralSpaceBuilder {
            number_of_energy_points: self.number_of_energy_points,
            energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_energy_integration_method<EnergyIntegrationMethod>(
        self,
        energy_integration_rule: EnergyIntegrationMethod,
    ) -> SpectralSpaceBuilder<
        T,
        RefEnergyRange,
        GeometryDim,
        EnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
    > {
        SpectralSpaceBuilder {
            number_of_energy_points: self.number_of_energy_points,
            energy_range: self.energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            marker: std::marker::PhantomData,
        }
    }
}

pub(crate) trait SpectralSpaceConstructor<T, GeometryDim, C>
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    C: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn build_coherent(self) -> super::BallisticSpectral<T>;
    fn build_incoherent(self) -> super::ScatteringSpectral<T, GeometryDim, C>;
}

impl<T, C, GeometryDim, WavevectorIntegrationRule, EnergyIntegrationRule>
    SpectralSpaceConstructor<T, GeometryDim, C>
    for SpectralSpaceBuilder<
        T,
        Range<T>,
        GeometryDim,
        EnergyIntegrationRule,
        WavevectorIntegrationRule,
    >
where
    T: Copy + RealField + NumCast,
    GeometryDim: SmallDim,
    C: Connectivity<T, GeometryDim>,
    EnergyIntegrationRule: super::GenerateWeights<T, U1, Segment1dConnectivity>,
    WavevectorIntegrationRule: super::GenerateWeights<T, GeometryDim, C>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn build_coherent(self) -> super::BallisticSpectral<T> {
        let energy_space = super::energy::EnergySpaceBuilder::new()
            .with_integration_rule(self.energy_integration_rule)
            .with_number_of_points(self.number_of_energy_points.unwrap())
            .with_energy_range(self.energy_range)
            .build();
        super::BallisticSpectral::new(energy_space)
    }

    fn build_incoherent(self) -> super::ScatteringSpectral<T, GeometryDim, C> {
        let energy_space = super::energy::EnergySpaceBuilder::new()
            .with_integration_rule(self.energy_integration_rule)
            .with_number_of_points(self.number_of_energy_points.unwrap())
            .with_energy_range(self.energy_range)
            .build();
        let wavevector_space = super::wavevector::WavevectorSpaceBuilder::new()
            .with_integration_rule(self.wavevector_integration_rule)
            .with_number_of_points(self.number_of_wavevector_points.unwrap())
            .with_maximum_wavevector(self.maximum_wavevector.unwrap())
            .build();
        super::ScatteringSpectral::new(energy_space, wavevector_space)
    }
}
