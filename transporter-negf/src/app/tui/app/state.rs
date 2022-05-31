use crate::app::Calculation;
use nalgebra::RealField;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Progress<T>
where
    T: RealField + Copy,
{
    pub current_voltage: T,
    pub calculation_type: Calculation<T>,
    pub scattering_scale_factor: T,
    pub outer_iteration: usize,
    pub inner_iteration: Option<usize>,
    pub current_outer_residual: T,
    pub target_outer_residual: T,
    pub current_inner_residual: Option<T>,
    pub target_inner_residual: Option<T>,
    pub time_for_voltage_point: Duration,
    pub time_for_outer_iteration: Duration,
    pub time_for_inner_iteration: Option<Duration>,
}

impl<T: RealField + Copy> Default for Progress<T> {
    fn default() -> Self {
        Self {
            current_voltage: T::zero(),
            calculation_type: Calculation::Coherent {
                voltage_target: T::zero(),
            },
            scattering_scale_factor: T::one(),
            outer_iteration: 0,
            inner_iteration: None,
            current_outer_residual: T::zero(),
            target_outer_residual: T::zero(),
            current_inner_residual: None,
            target_inner_residual: None,
            time_for_voltage_point: Duration::default(),
            time_for_outer_iteration: Duration::default(),
            time_for_inner_iteration: None,
        }
    }
}

impl<T: RealField + Copy> Progress<T> {
    pub(crate) fn set_calculation(&mut self, calculation: Calculation<T>) {
        self.calculation_type = calculation;
    }

    pub(crate) fn set_voltage(&mut self, voltage: T) {
        self.current_voltage = voltage;
    }

    pub(crate) fn set_scattering_scale_factor(&mut self, scattering_scale_factor: T) {
        self.scattering_scale_factor = scattering_scale_factor;
    }

    pub(crate) fn set_outer_iteration(&mut self, outer_iteration: usize) {
        self.outer_iteration = outer_iteration;
    }

    pub(crate) fn set_inner_iteration(&mut self, inner_iteration: usize) {
        self.inner_iteration = Some(inner_iteration);
    }

    pub(crate) fn set_outer_residual(&mut self, outer_residual: T) {
        self.current_outer_residual = outer_residual;
    }

    pub(crate) fn set_target_outer_residual(&mut self, target_outer_residual: T) {
        self.target_outer_residual = target_outer_residual;
    }

    pub(crate) fn set_time_for_outer_iteration(&mut self, time_for_outer_iteration: Duration) {
        self.time_for_outer_iteration = time_for_outer_iteration;
    }

    pub(crate) fn set_inner_residual(&mut self, inner_residual: T) {
        self.current_inner_residual = Some(inner_residual);
    }

    pub(crate) fn set_target_inner_residual(&mut self, target_inner_residual: T) {
        self.target_inner_residual = Some(target_inner_residual);
    }

    pub(crate) fn set_time_for_inner_iteration(&mut self, time_for_inner_iteration: Duration) {
        self.time_for_inner_iteration = Some(time_for_inner_iteration);
    }

    pub(crate) fn set_time_for_voltage_point(&mut self, time_for_voltage_point: Duration) {
        self.time_for_voltage_point = time_for_voltage_point;
    }
}

#[derive(Clone)]
pub enum AppState<T: RealField + Copy> {
    Init,
    Initialized,
    Running(std::sync::Arc<Progress<T>>),
}

impl<T: RealField + Copy> Default for AppState<T> {
    fn default() -> Self {
        Self::Init
    }
}

impl<T: RealField + Copy> AppState<T> {
    pub(crate) fn initialized() -> Self {
        Self::Initialized
    }

    pub(crate) fn running(tracker: std::sync::Arc<Progress<T>>) -> Self {
        Self::Running(tracker)
    }

    pub(crate) fn is_initialized(&self) -> bool {
        matches!(self, &Self::Initialized { .. })
    }
}
