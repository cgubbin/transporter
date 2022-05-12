use crate::spectral::IntegrationRule;
use config::{Config, File};
use miette::IntoDiagnostic;
use serde::{de::DeserializeOwned, Deserialize};
use std::env;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct Configuration<T> {
    pub global: GlobalConfiguration<T>,
    pub mesh: MeshConfiguration<T>,
    pub inner_loop: InnerConfiguration<T>,
    pub outer_loop: OuterConfiguration<T>,
    pub spectral: SpectralConfiguration<T>,
}

#[derive(Debug, Deserialize)]
pub struct SpectralConfiguration<T> {
    pub number_of_energy_points: usize,
    // TODO : These should be OPoint of size `BandDim`
    pub minimum_energy: T,
    pub maximum_energy: T,
    pub energy_integration_rule: IntegrationRule,
    pub number_of_wavevector_points: usize,
    pub maximum_wavevector: T,
    pub wavevector_integration_rule: IntegrationRule,
}

#[derive(Debug, Deserialize)]
pub struct GlobalConfiguration<T> {
    pub number_of_bands: usize,
    pub voltage_step: T,
}

#[derive(Debug, Deserialize)]
pub struct MeshConfiguration<T> {
    pub unit_size: T,
    pub elements_per_unit: usize,
    pub maximum_growth_rate: T,
}

#[derive(Debug, Deserialize)]
pub struct InnerConfiguration<T> {
    pub maximum_iterations: usize,
    pub tolerance: T,
}

#[derive(Debug, Deserialize)]
pub struct OuterConfiguration<T> {
    pub maximum_iterations: usize,
    pub tolerance: T,
}

impl<T: DeserializeOwned> Configuration<T> {
    pub fn build() -> miette::Result<Self> {
        // If I am running it here we should automatically be more debuggy
        let run_mode = env::var("RUN_MODE").unwrap_or_else(|_| "development".into());

        let s = Config::builder()
            // The default settings for the simulation which we use in the general case
            .add_source(File::with_name("../.config/default"))
            // The override settings which may be set by the user, optional
            .add_source(File::with_name(&format!("../.config/{}", run_mode)).required(false))
            .build()
            .into_diagnostic()?;

        s.try_deserialize()
            .map_err(|e| miette::miette!(format!("Failed to deserialize the config file: {:?}", e)))
    }
}
