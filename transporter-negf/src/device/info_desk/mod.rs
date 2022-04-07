mod materials;

pub(crate) use materials::Material;

use super::Device;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, OPoint, OVector, Point1, RealField, U1, U3,
};
use std::marker::PhantomData;
use transporter_mesher::SmallDim;

/// Struct holding all the material information necessary to solve the problem
#[derive(Debug)]
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
    /// Each layer in the stack has an effective doping density -> This should probably be a Point2<> for both doping types..
    pub(crate) donor_densities: Vec<T>,
    pub(crate) acceptor_densities: Vec<T>,
    pub(crate) temperature: T,
    pub(crate) voltage_offsets: Vec<T>,
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
            donor_densities: Vec::new(),
            acceptor_densities: Vec::new(),
            temperature: T::zero(),
            voltage_offsets: Vec::new(),
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
    fn assemble_from_layers_and_doping_densities(
        layers: Vec<LayerInfoDesk<T, BandDim>>,
        acceptor_density: Vec<T>,
        donor_density: Vec<T>,
        temperature: T,
        voltage_offsets: Vec<T>,
    ) -> DeviceInfoDesk<T, GeometryDim, BandDim>;
}

/// SINGLE BAND
///
///

impl<T: Copy + serde::de::DeserializeOwned + RealField, GeometryDim: SmallDim>
    BuildInfoDesk<T, GeometryDim, U1> for Device<T, GeometryDim>
where
    DefaultAllocator: Allocator<T, U1> + Allocator<T, U1, U3> + Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: serde::Deserialize<'static>,
{
    fn build_device_info_desk(&self) -> color_eyre::Result<DeviceInfoDesk<T, GeometryDim, U1>> {
        let mut layers = Vec::new();
        let mut acceptor_densities = Vec::new();
        let mut donor_densities = Vec::new();
        for layer in self.layers.iter() {
            layers.push(layer.material.get_info()?);
            acceptor_densities.push(layer.acceptor_density);
            donor_densities.push(layer.donor_density);
        }
        Ok(Self::assemble_from_layers_and_doping_densities(
            layers,
            acceptor_densities,
            donor_densities,
            self.temperature,
            self.voltage_offsets.clone(),
        ))
    }

    fn assemble_from_layers_and_doping_densities(
        layers: Vec<LayerInfoDesk<T, U1>>,
        acceptor_densities: Vec<T>,
        donor_densities: Vec<T>,
        temperature: T,
        voltage_offsets: Vec<T>,
    ) -> DeviceInfoDesk<T, GeometryDim, U1> {
        // Naive implementation: must be a better way to do this.
        let mut info_desk = DeviceInfoDesk::default();
        for (layer, (acceptor_density, donor_density)) in layers.into_iter().zip(
            acceptor_densities
                .into_iter()
                .zip(donor_densities.into_iter()),
        ) {
            info_desk.band_offsets.push(layer.band_offset);
            info_desk.effective_masses.push(layer.effective_mass);
            info_desk
                .dielectric_constants
                .push(layer.dielectric_constant);
            info_desk.acceptor_densities.push(acceptor_density);
            info_desk.donor_densities.push(donor_density);
        }
        info_desk.temperature = temperature;
        info_desk.voltage_offsets = voltage_offsets;
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
            Material::AlGaAs => Ok(LayerInfoDesk::algaas()),
            Material::SiC => Ok(LayerInfoDesk::sic()),
            // _ => Err(eyre!(
            //     "The material {} does not have a `get_info` implementation",
            //     self
            // )),
        }
    }
}

impl<T: RealField> LayerInfoDesk<T, U1> {
    #[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
    fn gaas() -> Self {
        Self {
            effective_mass: nalgebra::Vector1::new([0.067, 0.067, 0.067]),
            band_offset: Point1::new(0.0),
            dielectric_constant: [11.5, 11.5, 11.5],
        }
    }

    #[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
    fn algaas() -> Self {
        Self {
            effective_mass: nalgebra::Vector1::new([0.067, 0.067, 0.067]),
            band_offset: Point1::new(0.3),
            dielectric_constant: [11.5, 11.5, 11.5],
        }
    }

    #[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
    fn sic() -> Self {
        Self {
            effective_mass: nalgebra::Vector1::new([0.25, 0.25, 0.25]),
            band_offset: Point1::new(0.2),
            dielectric_constant: [8.5, 8.5, 8.5],
        }
    }
}
