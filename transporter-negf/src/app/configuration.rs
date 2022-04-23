use crate::spectral::IntegrationRule;
use color_eyre::eyre::eyre;
use config::{Config, File};
use miette::IntoDiagnostic;
use serde::{de::DeserializeOwned, Deserialize};
use std::env;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub(crate) struct Configuration<T> {
    pub(crate) global: GlobalConfiguration<T>,
    pub(crate) mesh: MeshConfiguration<T>,
    pub(crate) inner_loop: InnerConfiguration<T>,
    pub(crate) outer_loop: OuterConfiguration<T>,
    pub(crate) spectral: SpectralConfiguration<T>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SpectralConfiguration<T> {
    pub(crate) number_of_energy_points: usize,
    // TODO : These should be OPoint of size `BandDim`
    pub(crate) minimum_energy: T,
    pub(crate) maximum_energy: T,
    pub(crate) energy_integration_rule: IntegrationRule,
    pub(crate) number_of_wavevector_points: usize,
    pub(crate) maximum_wavevector: T,
    pub(crate) wavevector_integration_rule: IntegrationRule,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GlobalConfiguration<T> {
    pub(crate) number_of_bands: usize,
    #[serde(skip)]
    pub(crate) marker: std::marker::PhantomData<T>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MeshConfiguration<T> {
    pub(crate) unit_size: T,
    pub(crate) elements_per_unit: usize,
    pub(crate) maximum_growth_rate: T,
}

#[derive(Debug, Deserialize)]
pub(crate) struct InnerConfiguration<T> {
    pub(crate) maximum_iterations: usize,
    pub(crate) tolerance: T,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OuterConfiguration<T> {
    pub(crate) maximum_iterations: usize,
    pub(crate) tolerance: T,
}

impl<T: DeserializeOwned> Configuration<T> {
    pub(crate) fn build() -> miette::Result<Self> {
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
