use super::{
    energy::EnergySpaceBuilder, wavevector::BuildWavevectorSpace, GenerateWeights, SpectralSpace,
    WavevectorSpace,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField, U1};
use num_traits::NumCast;
use std::ops::Range;
use transporter_mesher::{Connectivity, Mesh, Segment1dConnectivity, SmallDim};

pub struct SpectralSpaceBuilder<
    T,
    RefEnergyRange,
    RefEnergyIntegrationMethod,
    RefWavevectorIntegrationMethod,
    RefMesh,
> {
    number_of_energy_points: Option<usize>,
    energy_range: RefEnergyRange,
    number_of_wavevector_points: Option<usize>,
    maximum_wavevector: Option<T>,
    energy_integration_rule: RefEnergyIntegrationMethod,
    wavevector_integration_rule: RefWavevectorIntegrationMethod,
    mesh: RefMesh,
}

impl<T> SpectralSpaceBuilder<T, (), (), (), ()> {
    pub fn new() -> Self {
        Self {
            number_of_energy_points: None,
            energy_range: (),
            number_of_wavevector_points: None,
            maximum_wavevector: None,
            energy_integration_rule: (),
            wavevector_integration_rule: (),
            mesh: (),
        }
    }
}

impl<
        T: RealField,
        RefEnergyRange,
        RefEnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
        RefMesh,
    >
    SpectralSpaceBuilder<
        T,
        RefEnergyRange,
        RefEnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
        RefMesh,
    >
{
    pub fn with_number_of_energy_points(self, number_of_energy_points: usize) -> Self {
        Self {
            number_of_energy_points: Some(number_of_energy_points),
            energy_range: self.energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            mesh: self.mesh,
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
            mesh: self.mesh,
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
        RefMesh,
    > {
        SpectralSpaceBuilder {
            number_of_energy_points: self.number_of_energy_points,
            energy_range: self.energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: Some(maximum_wavevector),
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            mesh: self.mesh,
        }
    }

    pub fn with_energy_range(
        self,
        energy_range: Range<T>,
    ) -> SpectralSpaceBuilder<
        T,
        Range<T>,
        RefEnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
        RefMesh,
    > {
        SpectralSpaceBuilder {
            number_of_energy_points: self.number_of_energy_points,
            energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            mesh: self.mesh,
        }
    }

    pub fn with_energy_integration_method<EnergyIntegrationMethod>(
        self,
        energy_integration_rule: EnergyIntegrationMethod,
    ) -> SpectralSpaceBuilder<
        T,
        RefEnergyRange,
        EnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
        RefMesh,
    > {
        SpectralSpaceBuilder {
            number_of_energy_points: self.number_of_energy_points,
            energy_range: self.energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            mesh: self.mesh,
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
        RefMesh,
    > {
        SpectralSpaceBuilder {
            number_of_energy_points: self.number_of_energy_points,
            energy_range: self.energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule,
            mesh: self.mesh,
        }
    }

    pub(crate) fn with_mesh<GeometryDim: SmallDim, Conn: Connectivity<T, GeometryDim>>(
        self,
        mesh: &Mesh<T, GeometryDim, Conn>,
    ) -> SpectralSpaceBuilder<
        T,
        RefEnergyRange,
        RefEnergyIntegrationMethod,
        RefWavevectorIntegrationMethod,
        &Mesh<T, GeometryDim, Conn>,
    >
    where
        DefaultAllocator: Allocator<T, GeometryDim>,
    {
        SpectralSpaceBuilder {
            number_of_energy_points: self.number_of_energy_points,
            energy_range: self.energy_range,
            number_of_wavevector_points: self.number_of_wavevector_points,
            maximum_wavevector: self.maximum_wavevector,
            energy_integration_rule: self.energy_integration_rule,
            wavevector_integration_rule: self.wavevector_integration_rule,
            mesh,
        }
    }
}

impl<T, EnergyIntegrationRule> SpectralSpaceBuilder<T, Range<T>, EnergyIntegrationRule, (), ()>
where
    T: Copy + RealField + NumCast,
    EnergyIntegrationRule: super::GenerateWeights<T, U1, Segment1dConnectivity>,
{
    pub fn build_coherent(self) -> SpectralSpace<T, ()> {
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

impl<T, WavevectorIntegrationRule, EnergyIntegrationRule>
    SpectralSpaceBuilder<
        T,
        Range<T>,
        EnergyIntegrationRule,
        WavevectorIntegrationRule,
        &Mesh<T, U1, Segment1dConnectivity>,
    >
where
    T: Copy + RealField + NumCast,
    EnergyIntegrationRule: GenerateWeights<T, U1, Segment1dConnectivity>,
    WavevectorIntegrationRule: GenerateWeights<T, U1, Segment1dConnectivity>,
    DefaultAllocator: Allocator<T, U1>,
{
    pub(crate) fn build_incoherent(
        self,
    ) -> SpectralSpace<T, WavevectorSpace<T, U1, Segment1dConnectivity>> {
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
