//! Controls the deserialization and storage of the top-level device structure,
//! and the `InfoDesk` traits which yield all the material information necessary
//! to run the simulation

/// The info-desk traits which describe the material parameters used in the simulation
pub mod info_desk;
/// The deserialization and storage of the `Device`
pub(crate) mod reader;

pub(crate) use info_desk::Material;
pub use reader::Device;
