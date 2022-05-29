use crate::app::Calculation;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Tracker {
    pub file_name: PathBuf,
    pub current_voltage: f64,
    pub calculation_type: Calculation<f64>,
    pub outer_iteration: u32,
    pub inner_iteration: Option<u32>,
    pub current_outer_residual: f64,
    pub target_outer_residual: f64,
    pub current_inner_residual: Option<f64>,
    pub target_inner_residual: Option<f64>,
}

impl Tracker {
    pub fn new(file_name: PathBuf) -> Self {
        Self {
            file_name,
            current_voltage: 0_f64,
            calculation_type: Calculation::Coherent {
                voltage_target: 0_f64,
            },
            outer_iteration: 0,
            inner_iteration: None,
            current_outer_residual: 0_f64,
            target_outer_residual: 0_f64,
            current_inner_residual: None,
            target_inner_residual: None,
        }
    }

    pub fn new_random() -> Self {
        Self {
            file_name: PathBuf::new(),
            current_voltage: 3_f64,
            calculation_type: Calculation::Coherent {
                voltage_target: 2_f64,
            },
            outer_iteration: 0,
            inner_iteration: None,
            current_outer_residual: 1_f64,
            target_outer_residual: 2_f64,
            current_inner_residual: None,
            target_inner_residual: None,
        }
    }
}

#[derive(Clone)]
pub enum AppState {
    Init,
    Initialized,
    Running(std::sync::Arc<Tracker>),
}

impl Default for AppState {
    fn default() -> Self {
        Self::Init
    }
}

impl AppState {
    pub(crate) fn initialized() -> Self {
        Self::Initialized
    }

    pub(crate) fn running(tracker: std::sync::Arc<Tracker>) -> Self {
        Self::Running(tracker)
    }

    pub(crate) fn is_initialized(&self) -> bool {
        matches!(self, &Self::Initialized { .. })
    }
}
