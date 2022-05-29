use std::path::PathBuf;

pub mod handler;

// We have an initial dummy impl
#[derive(Clone, Debug)]
pub enum IoEvent {
    // The application is being initialized
    Initialize,
    // Run the simulation
    Run(PathBuf),
}
