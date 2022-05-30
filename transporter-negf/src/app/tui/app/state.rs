use crate::app::Calculation;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Progress {
    pub current_voltage: f64,
    pub calculation_type: Calculation<f64>,
    pub scattering_scale_factor: f64,
    pub outer_iteration: usize,
    pub inner_iteration: Option<usize>,
    pub current_outer_residual: f64,
    pub target_outer_residual: f64,
    pub current_inner_residual: Option<f64>,
    pub target_inner_residual: Option<f64>,
    pub time_for_voltage_point: Duration,
    pub time_for_outer_iteration: Duration,
    pub time_for_inner_iteration: Option<Duration>,
}

impl Default for Progress {
    fn default() -> Self {
        Self {
            current_voltage: 0_f64,
            calculation_type: Calculation::Coherent {
                voltage_target: 0_f64,
            },
            scattering_scale_factor: 1_f64,
            outer_iteration: 0,
            inner_iteration: None,
            current_outer_residual: 0_f64,
            target_outer_residual: 0_f64,
            current_inner_residual: None,
            target_inner_residual: None,
            time_for_voltage_point: Duration::default(),
            time_for_outer_iteration: Duration::default(),
            time_for_inner_iteration: None,
        }
    }
}

impl Progress {
    pub(crate) fn set_calculation(&mut self, calculation: Calculation<f64>) {
        self.calculation_type = calculation;
    }

    pub(crate) fn set_voltage(&mut self, voltage: f64) {
        self.current_voltage = voltage;
    }

    pub(crate) fn set_scattering_scale_factor(&mut self, scattering_scale_factor: f64) {
        self.scattering_scale_factor = scattering_scale_factor;
    }

    pub(crate) fn set_outer_iteration(&mut self, outer_iteration: usize) {
        self.outer_iteration = outer_iteration;
    }

    pub(crate) fn set_outer_residual(&mut self, outer_residual: f64) {
        self.current_outer_residual = outer_residual;
    }

    pub(crate) fn set_target_outer_residual(&mut self, target_outer_residual: f64) {
        self.target_outer_residual = target_outer_residual;
    }

    pub(crate) fn set_time_for_outer_iteration(&mut self, time_for_outer_iteration: Duration) {
        self.time_for_outer_iteration = time_for_outer_iteration;
    }

    pub(crate) fn set_time_for_voltage_point(&mut self, time_for_voltage_point: Duration) {
        self.time_for_voltage_point = time_for_voltage_point;
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
