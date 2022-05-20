use crate::app::Calculation;
use nalgebra::RealField;

pub(crate) struct Convergence<T>
where
    T: RealField,
{
    pub(crate) outer_tolerance: T,
    pub(crate) inner_tolerance: T,
    pub(crate) maximum_outer_iterations: usize,
    pub(crate) maximum_inner_iterations: usize,
    pub(crate) calculation_type: Calculation<T>,
}

impl<T: Copy + RealField> Convergence<T> {
    pub(crate) fn maximum_outer_iterations(&self) -> usize {
        self.maximum_outer_iterations
    }

    pub(crate) fn maximum_inner_iterations(&self) -> usize {
        self.maximum_inner_iterations
    }

    pub(crate) fn outer_tolerance(&self) -> T {
        self.outer_tolerance
    }

    pub(crate) fn inner_tolerance(&self) -> T {
        self.inner_tolerance
    }

    pub(crate) fn calculation_type(&self) -> &Calculation<T> {
        &self.calculation_type
    }
}
