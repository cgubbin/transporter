mod materials;

use super::Device;
pub(crate) use materials::Material;

use color_eyre::eyre::eyre;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, OMatrix, OPoint, OVector, Point1, RealField, Vector3,
    U1, U3,
};
use std::marker::PhantomData;
use transporter_mesher::SmallDim;

/// Struct holding all the material information necessary to solve the problem
pub(crate) struct DeviceInfoDesk<T: RealField, GeometryDim: SmallDim, BandDim: SmallDim>
where
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    /// Each layer in the stack has 'BandDim' effective masses, each of which have three Cartesian components
    pub(crate) effective_masses: Vec<OVector<[T; 3], BandDim>>,
    /// Each layer in the stack has `BandDim` band offsets, which are the band levels at zero momentum
    pub(crate) band_offsets: Vec<OPoint<T, BandDim>>,
    /// Each layer in the stack has a static dielectric constant with a component for each Cartesian axis
    dielectric_constants: Vec<[T; 3]>,
    marker: PhantomData<GeometryDim>,
}

impl<T: RealField, GeometryDim: SmallDim, BandDim: SmallDim> Default
    for DeviceInfoDesk<T, GeometryDim, BandDim>
where
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    fn default() -> Self {
        Self {
            effective_masses: Vec::new(),
            band_offsets: Vec::new(),
            dielectric_constants: Vec::new(),
            marker: PhantomData,
        }
    }
}

/// Struct holding all the material information necessary to solve the problem
pub(crate) struct LayerInfoDesk<T: RealField, BandDim: SmallDim>
where
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    /// Each layer in the stack has 'BandDim' effective masses, each of which have three Cartesian components
    effective_mass: OVector<[T; 3], BandDim>,
    /// Each layer in the stack has `BandDim` band offsets, which are the band levels at zero momentum
    band_offset: OPoint<T, BandDim>,
    /// Each layer in the stack has a static dielectric constant with a component for each Cartesian axis
    dielectric_constant: [T; 3],
}

pub(crate) trait BuildInfoDesk<T: RealField, GeometryDim: SmallDim, BandDim: SmallDim>
where
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    fn build_device_info_desk(&self)
        -> color_eyre::Result<DeviceInfoDesk<T, GeometryDim, BandDim>>;
    fn assemble_from_layers(
        layers: Vec<LayerInfoDesk<T, BandDim>>,
    ) -> DeviceInfoDesk<T, GeometryDim, BandDim>;
}

/// SINGLE BAND
///
///

impl<T: serde::de::DeserializeOwned + RealField, GeometryDim: SmallDim>
    BuildInfoDesk<T, GeometryDim, U1> for Device<T, GeometryDim>
where
    DefaultAllocator: Allocator<T, U1> + Allocator<T, U1, U3> + Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: serde::Deserialize<'static>,
{
    fn build_device_info_desk(&self) -> color_eyre::Result<DeviceInfoDesk<T, GeometryDim, U1>> {
        let mut layers = Vec::new();
        for layer in self.layers.iter() {
            layers.push(layer.material.get_info()?);
        }
        Ok(Self::assemble_from_layers(layers))
    }

    fn assemble_from_layers(
        layers: Vec<LayerInfoDesk<T, U1>>,
    ) -> DeviceInfoDesk<T, GeometryDim, U1> {
        // Naive implementation: must be a better way to do this.
        let mut info_desk = DeviceInfoDesk::default();
        for layer in layers {
            info_desk.band_offsets.push(layer.band_offset);
            info_desk.effective_masses.push(layer.effective_mass);
            info_desk
                .dielectric_constants
                .push(layer.dielectric_constant);
        }
        info_desk
    }
}

// A single band implementation
impl Material {
    fn get_info<T: RealField>(&self) -> color_eyre::Result<LayerInfoDesk<T, U1>>
    where
        DefaultAllocator: Allocator<T, U1> + Allocator<T, U1, U3>,
    {
        match self {
            Material::GaAs => Ok(LayerInfoDesk::gaas()),
            Material::SiC => Ok(LayerInfoDesk::sic()),
            _ => Err(eyre!(
                "The material {} does not have a `get_info` implementation",
                self
            )),
        }
    }
}

impl<T: RealField> LayerInfoDesk<T, U1> {
    #[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
    fn gaas() -> Self {
        Self {
            effective_mass: nalgebra::Vector1::new([0.25, 0.25, 0.25]),
            band_offset: Point1::new(0.5),
            dielectric_constant: [8.5, 8.5, 8.5],
        }
    }

    #[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
    fn sic() -> Self {
        Self {
            effective_mass: nalgebra::Vector1::new([0.25, 0.25, 0.25]),
            band_offset: Point1::new(0.5),
            dielectric_constant: [8.5, 8.5, 8.5],
        }
    }
}
