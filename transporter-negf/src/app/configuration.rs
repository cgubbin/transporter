use color_eyre::eyre::eyre;
use config::{Config, File};
use serde::{de::DeserializeOwned, Deserialize};
use std::env;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub(crate) struct Configuration<T> {
    pub(crate) global: GlobalConfiguration<T>,
    pub(crate) mesh: MeshConfiguration<T>,
    pub(crate) inner_loop: InnerConfiguration<T>,
    pub(crate) outer_loop: OuterConfiguration<T>,
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
    maximum_iterations: usize,
    tolerance: T,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OuterConfiguration<T> {
    maximum_iterations: usize,
    tolerance: T,
}

impl<T: DeserializeOwned> Configuration<T> {
    pub(crate) fn build() -> color_eyre::Result<Self> {
        // If I am running it here we should automatically be more debuggy
        let run_mode = env::var("RUN_MODE").unwrap_or_else(|_| "development".into());

        let s = Config::builder()
            // The default settings for the simulation which we use in the general case
            .add_source(File::with_name("../.config/default"))
            // The override settings which may be set by the user, optional
            .add_source(File::with_name(&format!("../.config/{}", run_mode)).required(false))
            .build()?;

        s.try_deserialize()
            .map_err(|e| eyre!(format!("Failed to deserialize the config file: {:?}", e)))
    }
}
