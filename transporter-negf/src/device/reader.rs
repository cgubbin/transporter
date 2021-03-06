use super::Material;
use crate::app::styles::Styles;
use config::{Config, File};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField, U1};
use owo_colors::OwoColorize;
use serde::{de::DeserializeOwned, Deserialize};
use std::{ops::Deref, path::PathBuf};
use transporter_mesher::SmallDim;

#[derive(Debug, Deserialize)]
#[serde(bound(deserialize = "'de: 'static"))]
/// A top-level device struct holding the geometrical information, and experimental constraints
pub struct Device<T: DeserializeOwned + RealField, GeometryDim: SmallDim>
where
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Deserialize<'static>,
{
    pub(crate) voltage_offsets: Vec<T>,
    pub(crate) temperature: T,
    pub(crate) lead_length: Option<T>,
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
/// Struct to hold the user-defined information for a single device layer.
pub struct Layer<T: DeserializeOwned + RealField, GeometryDim: SmallDim>
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
    /// Builds an instance of `Device` from a given path to a configuration file
    pub fn build(path: PathBuf) -> miette::Result<Self> {
        let s = Config::builder()
            .add_source(File::from(path))
            .build()
            .map_err(|e| miette::miette!("Device file path not read {:?}", e))?;
        s.try_deserialize()
            .map_err(|e| miette::miette!("Failed to deserialize device: {:?}", e))
    }
}

impl<T: DeserializeOwned + RealField, GeometryDim: SmallDim> Device<T, GeometryDim>
where
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Deserialize<'static>,
{
    /// Returns a type that can display `MyValue`.
    pub fn display(&self) -> DeviceDisplay<'_, T, GeometryDim> {
        DeviceDisplay {
            device: self,
            styles: Box::new(Styles::default()),
        }
    }
}

pub struct DeviceDisplay<'a, T: DeserializeOwned + RealField, GeometryDim: SmallDim>
where
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Deserialize<'static>,
{
    device: &'a Device<T, GeometryDim>,
    styles: Box<Styles>,
}

impl<'a, T: DeserializeOwned + RealField, GeometryDim: SmallDim> DeviceDisplay<'a, T, GeometryDim>
where
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Deserialize<'static>,
{
    /// Colorizes the output.
    pub fn colorize(&mut self) {
        self.styles.colorize();
    }
}

impl<'a, T: DeserializeOwned + RealField> std::fmt::Display for DeviceDisplay<'a, T, U1>
where
    DefaultAllocator: Allocator<T, U1>,
    <DefaultAllocator as Allocator<T, U1>>::Buffer: Deserialize<'static>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "     {:-^30}", "-".style(self.styles.device_style))?;
        for layer in self.device.layers.iter() {
            let info = format!("{}: {}nm", layer.material, layer.thickness[0]);
            writeln!(
                f,
                "     {}{: ^28}{}",
                "|".style(self.styles.device_style),
                info,
                "|".style(self.styles.device_style),
            )?;
            writeln!(f, "     {:-^30}", "-".style(self.styles.device_style))?;
        }
        writeln!(
            f,
            "     {: <30}",
            format!("System of {} unique layers", self.device.layers.len())
        )?;
        Ok(())
    }
}
