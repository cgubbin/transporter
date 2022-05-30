use self::actions::{Action, Actions};
use self::state::{AppState, Progress};
use super::{inputs::keys::Key, io::IoEvent};
use crate::app::Calculation;
use glob::glob;
use log::{debug, error, warn};
use std::path::PathBuf;
use tokio::sync::mpsc::Sender;

pub mod actions;
pub mod state;
pub mod ui;

#[derive(Debug, PartialEq, Eq)]
pub enum AppReturn {
    Exit,
    Continue,
    Run,
}

const VOLTAGE_INCREMENT: f64 = 0.01;

pub struct App {
    /// The files in the current directory -> read at startup to save overhead
    files_in_directory: Vec<PathBuf>,
    // TODO Re-institute the parsed files when we integrate -> allows for us to display the info in the browser
    // /// Parsed files in current directory -> parsed at startup to save overhead
    // parsed_files_in_directory: Vec<Structure>,
    /// Dispatch an io event
    io_tx: Sender<IoEvent>,
    /// Contextual actions
    actions: Actions,
    /// State
    is_loading: bool,
    state: AppState,
    calculation_type: Calculation<f64>,
}

impl<'a> App {
    pub fn new(io_tx: Sender<IoEvent>) -> Self {
        let actions = vec![
            Action::Quit,
            Action::Up,
            Action::Down,
            Action::Execute,
            Action::ToggleCalculation,
            Action::IncrementVoltageTarget,
            Action::DecrementVoltageTarget,
        ]
        .into();
        let is_loading = false;
        let state = AppState::default();

        let files_in_directory = glob("structures/*.toml")
            .unwrap()
            .map(|file| file.unwrap())
            .collect::<Vec<_>>();

        if files_in_directory.is_empty() {
            panic!()
        }

        Self {
            files_in_directory,
            io_tx,
            actions,
            is_loading,
            state,
            calculation_type: Calculation::Coherent {
                voltage_target: 0_f64,
            },
        }
    }

    pub(crate) async fn do_action(
        &mut self,
        key: Key,
        files_list_state: &mut tui::widgets::ListState,
        sender: &tokio::sync::mpsc::Sender<AppReturn>,
    ) {
        if let Some(action) = self.actions.find(key) {
            debug!("Run Action [{:?}]", action);
            match action {
                Action::Quit => sender.send(AppReturn::Exit).await.unwrap(),
                Action::Up => match self.state {
                    AppState::Running { .. } => sender.send(AppReturn::Continue).await.unwrap(),
                    _ => {
                        if let Some(selected) = files_list_state.selected() {
                            let number_of_files = self.files_in_directory.len();
                            if selected > 0 {
                                files_list_state.select(Some(selected - 1));
                            } else {
                                files_list_state.select(Some(number_of_files - 1));
                            }
                        }
                        sender.send(AppReturn::Continue).await.unwrap()
                    }
                },
                Action::Down => match self.state {
                    AppState::Running { .. } => sender.send(AppReturn::Continue).await.unwrap(),
                    _ => {
                        if let Some(selected) = files_list_state.selected() {
                            let number_of_files = self.files_in_directory.len();
                            if selected >= number_of_files - 1 {
                                files_list_state.select(Some(0));
                            } else {
                                files_list_state.select(Some(selected + 1));
                            }
                        }
                        sender.send(AppReturn::Continue).await.unwrap()
                    }
                },
                Action::ToggleCalculation => match self.state {
                    AppState::Running { .. } => sender.send(AppReturn::Continue).await.unwrap(),
                    _ => match self.calculation_type {
                        Calculation::Coherent { voltage_target: x } => {
                            self.calculation_type = Calculation::Incoherent { voltage_target: x };
                            sender.send(AppReturn::Continue).await.unwrap()
                        }
                        Calculation::Incoherent { voltage_target: x } => {
                            self.calculation_type = Calculation::Coherent { voltage_target: x };
                            sender.send(AppReturn::Continue).await.unwrap()
                        }
                    },
                },
                Action::DecrementVoltageTarget => {
                    match self.state {
                        // If we are running we currently do nothing, but this could change in the future
                        AppState::Running { .. } => sender.send(AppReturn::Continue).await.unwrap(),
                        _ => match self.calculation_type {
                            Calculation::Coherent {
                                ref mut voltage_target,
                            } => {
                                if *voltage_target > VOLTAGE_INCREMENT {
                                    *voltage_target -= VOLTAGE_INCREMENT;
                                }
                            }
                            Calculation::Incoherent {
                                ref mut voltage_target,
                            } => {
                                if *voltage_target > VOLTAGE_INCREMENT {
                                    *voltage_target -= VOLTAGE_INCREMENT;
                                }
                            }
                        },
                    }
                    sender.send(AppReturn::Continue).await.unwrap()
                }
                Action::IncrementVoltageTarget => {
                    match self.state {
                        // If we are running we currently do nothing, but this could change in the future
                        AppState::Running { .. } => sender.send(AppReturn::Continue).await.unwrap(),
                        _ => match self.calculation_type {
                            Calculation::Coherent {
                                ref mut voltage_target,
                            } => {
                                *voltage_target += VOLTAGE_INCREMENT;
                            }
                            Calculation::Incoherent {
                                ref mut voltage_target,
                            } => {
                                *voltage_target += VOLTAGE_INCREMENT;
                            }
                        },
                    }
                    sender.send(AppReturn::Continue).await.unwrap()
                }
                Action::Execute => {
                    match self.state {
                        AppState::Running { .. } => {}
                        _ => {
                            if let Some(file_to_run) = files_list_state.selected() {
                                // TODO Mutate the `app` into the `Running` state
                                // arrange the tracker and set the calculation to running
                                let file_to_run = self.file_in_directory(file_to_run).clone();
                                let tracker = std::sync::Arc::new(Progress::new());
                                self.running(tracker);
                                sender.send(AppReturn::Run).await.unwrap();
                                // let file_to_run = self.files_in_directory[selected].clone();
                                // self.run_simulation(file_to_run).await;
                            } else {
                                warn!("No file selected!");
                                sender.send(AppReturn::Continue).await.unwrap()
                            }
                        }
                    };
                }
            }
        } else {
            warn!("No action associated with key {}", key);
            sender.send(AppReturn::Continue).await.unwrap()
        }
    }

    pub(crate) async fn dispatch(&mut self, action: IoEvent) {
        // `is_loading` will set to false again after the async action has finished in io/handler.rs
        self.is_loading = true;
        if let Err(e) = self.io_tx.send(action).await {
            self.is_loading = false;
            error!("Error from dispatch {}", e);
        };
    }

    pub(crate) fn actions(&self) -> &Actions {
        &self.actions
    }

    pub(crate) fn state(&self) -> &AppState {
        &self.state
    }

    pub(crate) fn state_mut(&mut self) -> &mut AppState {
        &mut self.state
    }

    pub(crate) fn is_running(&self) -> bool {
        matches!(self.state(), AppState::Running { .. })
    }

    pub(crate) fn is_loading(&self) -> bool {
        !matches!(self.state(), AppState::Init)
    }

    pub(crate) fn initialized(&mut self) {
        // Update the contextual actions
        self.actions = vec![
            Action::Quit,
            Action::Up,
            Action::Down,
            Action::Execute,
            Action::ToggleCalculation,
            Action::IncrementVoltageTarget,
            Action::DecrementVoltageTarget,
        ]
        .into();
        self.state = AppState::initialized();
    }

    pub(crate) fn loaded(&mut self) {
        self.is_loading = false;
    }

    pub(crate) fn calculation_type(&self) -> &Calculation<f64> {
        &self.calculation_type
    }

    pub(crate) fn running(&mut self, tracker: std::sync::Arc<Progress>) {
        // Update the contextual variables for the running state
        self.actions = vec![Action::Quit].into();
        self.state = AppState::running(tracker);
    }

    pub(crate) fn file_in_directory(&self, index: usize) -> &std::path::PathBuf {
        &self.files_in_directory[index]
    }

    /// Just send a tick
    pub async fn update_on_tick(&self, sender: &tokio::sync::mpsc::Sender<AppReturn>) {
        // here we just increment a counter
        sender.send(AppReturn::Continue).await;
    }
}

use crate::app::build_mesh_with_config;
use crate::app::configuration::Configuration;
use crate::app::tracker::TrackerBuilder;
use crate::device::{info_desk::BuildInfoDesk, Device};
use nalgebra::U1;
use transporter_mesher::Mesh1d;

pub(crate) async fn run_simulation(
    app: std::sync::Arc<tokio::sync::Mutex<App>>,
    file_to_run: PathBuf,
    calculation: Calculation<f64>,
    sender: tokio::sync::mpsc::Sender<Progress>,
) -> miette::Result<()> {
    let config: Configuration<f64> = Configuration::build()?;
    let device: Device<f64, U1> = Device::build(file_to_run)?;
    let progress = Progress::new();
    let info_desk = device.build_device_info_desk()?;
    let mesh: Mesh1d<f64> =
        build_mesh_with_config(&config, device).map_err(|e| miette::miette!("{:?}", e))?;

    let tracker = TrackerBuilder::new(calculation)
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .map_err(|e| miette::miette!("{:?}", e))?;

    let term = console::Term::stdout();
    build_and_run(config, &mesh, &tracker, calculation, progress, &sender)
        .await
        .unwrap();

    // // will be available for Subscribers as a tracing Event
    std::thread::sleep(std::time::Duration::from_secs(1));
    let mut app = app.lock().await;
    log::info!("The simulation finished in the background");
    app.initialized();

    Ok(())
}

use crate::app::tracker::Tracker;
use crate::app::TransporterError;
use crate::outer_loop::{OuterLoopError, Potential};
use nalgebra::{allocator::Allocator, DefaultAllocator};
use ndarray::Array1;
use transporter_mesher::{Mesh, Segment1dConnectivity, SmallDim};

async fn build_and_run<BandDim: SmallDim>(
    config: Configuration<f64>,
    mesh: &Mesh<f64, U1, Segment1dConnectivity>,
    tracker: &Tracker<'_, f64, U1, BandDim>,
    calculation_type: Calculation<f64>,
    progress: Progress,
    sender: &tokio::sync::mpsc::Sender<Progress>,
) -> Result<(), TransporterError<f64>>
where
    DefaultAllocator: Allocator<f64, U1>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>
        + Allocator<Array1<f64>, BandDim>,
    <DefaultAllocator as Allocator<f64, U1>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    // Set the initial progress state
    progress.set_calculation(calculation_type);

    // Todo allow an initial potential to be read from a file
    let mut initial_potential: Potential<f64> =
        Potential::from_vector(ndarray::Array1::from(vec![0_f64; mesh.num_nodes()]));
    match calculation_type {
        Calculation::Coherent { voltage_target } => {
            let mut current_voltage = 0_f64;
            let mut voltage_step = config.global.voltage_step;
            while current_voltage <= voltage_target {
                // Set the current voltage and communicate to the master thread
                progress.set_voltage(current_voltage);
                sender.send(progress.clone()).await;
                // Do a single calculation
                tracing::info!("Solving for current voltage {current_voltage}V");
                match coherent_calculation_at_fixed_voltage(
                    current_voltage,
                    initial_potential.clone(),
                    &config,
                    mesh,
                    tracker,
                    &sender,
                ) {
                    // If it converged proceed
                    Ok(converged_potential) => {
                        let _ = std::mem::replace(&mut initial_potential, converged_potential);
                    }
                    // If there is an error, either return if unrecoverable or reduce the voltage step
                    Err(OuterLoopError::FixedPoint(fixed_point_error)) => match fixed_point_error {
                        conflux::core::FixedPointError::TooManyIterations(_cost) => {
                            current_voltage -= voltage_step;
                            voltage_step /= 2_f64;
                        }
                        _ => {
                            return Err(OuterLoopError::FixedPoint(fixed_point_error).into());
                        }
                    },
                    Err(e) => {
                        return Err(e.into());
                    }
                }
                // increment
                current_voltage += voltage_step;
            }
        }
        Calculation::Incoherent { voltage_target } => {
            let mut current_voltage = 0_f64;
            let mut voltage_step = config.global.voltage_step;
            while current_voltage <= voltage_target {
                // Set the current voltage and communicate to the master thread
                progress.set_voltage(current_voltage);
                sender.send(progress.clone()).await;
                tracing::info!("Solving for current voltage {current_voltage}V");
                match incoherent_calculation_at_fixed_voltage(
                    current_voltage,
                    initial_potential.clone(),
                    &config,
                    mesh,
                    tracker,
                    &sender,
                ) {
                    // If it converged proceed
                    Ok(converged_potential) => {
                        let _ = std::mem::replace(&mut initial_potential, converged_potential);
                    }
                    // If there is an error, either return if unrecoverable or reduce the voltage step
                    Err(OuterLoopError::FixedPoint(fixed_point_error)) => match fixed_point_error {
                        conflux::core::FixedPointError::TooManyIterations(_cost) => {
                            current_voltage -= voltage_step;
                            voltage_step /= 2_f64;
                        }
                        _ => {
                            return Err(OuterLoopError::FixedPoint(fixed_point_error).into());
                        }
                    },
                    Err(e) => {
                        return Err(e.into());
                    }
                }
                // increment
                current_voltage += voltage_step;
            }
        }
    }

    Ok(())
}
