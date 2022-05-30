use crate::app::Calculation;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Progress {
    pub current_voltage: f64,
    pub calculation_type: Calculation<f64>,
    pub outer_iteration: u32,
    pub inner_iteration: Option<u32>,
    pub current_outer_residual: f64,
    pub target_outer_residual: f64,
    pub current_inner_residual: Option<f64>,
    pub target_inner_residual: Option<f64>,
}

impl Progress {
    pub fn new() -> Self {
        Self {
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

    pub(crate) fn set_calculation(&mut self, calculation: Calculation<f64>) {
        self.calculation_type = calculation;
    }

    pub(crate) fn set_voltage(&mut self, voltage: f64) {
        self.current_voltage = voltage;
    }
}

#[derive(Clone)]
pub enum AppState {
    Init,
    Initialized,
    Running(std::sync::Arc<Progress>),
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

    pub(crate) fn running(tracker: std::sync::Arc<Progress>) -> Self {
        Self::Running(tracker)
    }

    pub(crate) fn is_initialized(&self) -> bool {
        matches!(self, &Self::Initialized { .. })
    }
}
