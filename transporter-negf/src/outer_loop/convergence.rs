use nalgebra::RealField;

pub(crate) enum CalculationType {
    Coherent,
    Incoherent,
}

pub(crate) struct Convergence<T>
where
    T: RealField,
{
    outer_tolerance: T,
    inner_tolerance: T,
    maximum_outer_iterations: usize,
    maximum_inner_iterations: usize,
    calculation_type: CalculationType,
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

    pub(crate) fn calculation_type(&self) -> &CalculationType {
        &self.calculation_type
    }
}
