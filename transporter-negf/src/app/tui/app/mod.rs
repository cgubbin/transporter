use self::actions::{Action, Actions};
use self::state::{AppState, Progress};
use super::{inputs::keys::Key, io::IoEvent};
use crate::app::Calculation;
use glob::glob;
use log::{debug, error, warn};
use nalgebra::RealField;
use ndarray::Array1;
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

pub struct App<T: RealField + Copy> {
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
    state: AppState<T>,
    calculation_type: Calculation<T>,
    pub(crate) potential: Option<Array1<T>>,
}

impl<'a, T: RealField + Copy> App<T> {
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
                voltage_target: T::zero(),
            },
            potential: None,
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
                    let voltage_increment: T = T::from_f64(0.01).unwrap();
                    match self.state {
                        // If we are running we currently do nothing, but this could change in the future
                        AppState::Running { .. } => sender.send(AppReturn::Continue).await.unwrap(),
                        _ => match self.calculation_type {
                            Calculation::Coherent {
                                ref mut voltage_target,
                            } => {
                                *voltage_target -= voltage_increment;
                                if *voltage_target < T::zero() {
                                    *voltage_target = T::zero();
                                }
                            }
                            Calculation::Incoherent {
                                ref mut voltage_target,
                            } => {
                                *voltage_target -= voltage_increment;
                                if *voltage_target < T::zero() {
                                    *voltage_target = T::zero();
                                }
                            }
                        },
                    }
                    sender.send(AppReturn::Continue).await.unwrap()
                }
                Action::IncrementVoltageTarget => {
                    let voltage_increment: T = T::from_f64(0.01).unwrap();
                    match self.state {
                        // If we are running we currently do nothing, but this could change in the future
                        AppState::Running { .. } => sender.send(AppReturn::Continue).await.unwrap(),
                        _ => match self.calculation_type {
                            Calculation::Coherent {
                                ref mut voltage_target,
                            } => {
                                *voltage_target += voltage_increment;
                            }
                            Calculation::Incoherent {
                                ref mut voltage_target,
                            } => {
                                *voltage_target += voltage_increment;
                            }
                        },
                    }
                    sender.send(AppReturn::Continue).await.unwrap()
                }
                Action::Execute => {
                    match self.state {
                        AppState::Running { .. } => {}
                        _ => {
                            if files_list_state.selected().is_some() {
                                // TODO Mutate the `app` into the `Running` state
                                // arrange the tracker and set the calculation to running
                                let tracker = std::sync::Arc::new(Progress::default());
                                self.running(tracker);
                                sender.send(AppReturn::Run).await.unwrap();
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

    pub(crate) fn state(&self) -> &AppState<T> {
        &self.state
    }

    pub(crate) fn state_mut(&mut self) -> &mut AppState<T> {
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

    pub(crate) fn calculation_type(&self) -> &Calculation<T> {
        &self.calculation_type
    }

    pub(crate) fn running(&mut self, tracker: std::sync::Arc<Progress<T>>) {
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
        sender.send(AppReturn::Continue).await.unwrap();
    }
}
