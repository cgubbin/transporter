use log::{error, info};
use nalgebra::RealField;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::super::App;
use super::IoEvent;

// We handle IO in the IO thread without blocking the UI thread
pub struct IoAsyncHandler<T: RealField + Copy> {
    app: Arc<Mutex<App<T>>>,
}

impl<T: RealField + Copy> IoAsyncHandler<T> {
    pub fn new(app: Arc<Mutex<App<T>>>) -> Self {
        Self { app }
    }

    pub async fn handle_io_event(&mut self, io_event: IoEvent) {
        let result = match io_event {
            IoEvent::Initialize => self.do_initialize().await,
            IoEvent::Run(file_to_run) => self.run_simulation(file_to_run).await,
          
        };
        if let Err(err) = result {
            error!("Oops, bad has happened: {:?}", err);
        }

        let mut app = self.app.lock().await;
        app.loaded();
    }

    // A dummy impl
    async fn do_initialize(&mut self) -> miette::Result<()> {
        info!("🚀 Initialize the application");
        let mut app = self.app.lock().await;
        app.initialized(); // we could update the app state
        info!("👍 Application initialized");

        Ok(())
    }

    // Run the simulation
    async fn run_simulation(&mut self, file_to_run: PathBuf) -> miette::Result<()> {
        info!(
            "😴 Running transport calculation sleeping for {:?}...",
            file_to_run
        );
        // Notify the app for having slept
        let mut app = self.app.lock().await;
        app.initialized();

        Ok(())
    }
}
