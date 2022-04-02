use super::Material;
use color_eyre::eyre::eyre;
use config::{Config, File};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use serde::{de::DeserializeOwned, Deserialize};
use std::{ops::Deref, path::PathBuf};
use transporter_mesher::SmallDim;

#[derive(Debug, Deserialize)]
#[serde(bound(deserialize = "'de: 'static"))]
pub(crate) struct Device<T: DeserializeOwned + RealField, GeometryDim: SmallDim>
where
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Deserialize<'static>,
{
    pub(crate) voltage_offsets: Vec<T>,
    pub(crate) temperature: T,
    pub(crate) layers: Vec<Layer<T, GeometryDim>>,
}

impl<T: DeserializeOwned + RealField, GeometryDim: SmallDim> Deref for Device<T, GeometryDim>
where
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Deserialize<'static>,
{
    type Target = Vec<Layer<T, GeometryDim>>;

    fn deref(&self) -> &Self::Target {
        &self.layers
    }
}

#[derive(Debug, Deserialize)]
#[serde(bound(deserialize = "'de: 'static"))]
pub(crate) struct Layer<T: DeserializeOwned + RealField, GeometryDim: SmallDim>
where
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Deserialize<'static>,
{
    pub(crate) thickness: nalgebra::OPoint<T, GeometryDim>,
    pub(crate) material: Material,
    pub(crate) acceptor_density: T,
    pub(crate) donor_density: T,
}

impl<T: DeserializeOwned + RealField, GeometryDim: SmallDim> Device<T, GeometryDim>
where
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Deserialize<'static>,
{
    pub(crate) fn build(path: PathBuf) -> color_eyre::Result<Self> {
        let s = Config::builder().add_source(File::from(path)).build()?;
        s.try_deserialize()
            .map_err(|e| eyre!("Failed to deserialize device: {:?}", e))
    }
}
