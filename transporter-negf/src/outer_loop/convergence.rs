use crate::app::Calculation;
use nalgebra::RealField;

/// Holds the convergence criterion for the `OuterLoop` and `InnerLoop` as defined in the configuration file
pub(crate) struct Convergence<T>
where
    T: RealField,
{
    /// The absolute tolerance necessary to terminate the outer iteration
    ///
    /// This is used to check the update of
    // TODO
    pub(crate) outer_tolerance: T,
    /// The absolute tolerance necessary to terminate the inner iteration
    ///
    /// This is used to check the update of
    // TODO
    pub(crate) inner_tolerance: T,
    /// The maximum number of iterations the outer loop is allowed to enact
    pub(crate) maximum_outer_iterations: usize,
    /// The maximum number of iterations an instance of the inner loop is allowed to enact
    pub(crate) maximum_inner_iterations: usize,
    /// Whether or not to do numerical security checks as the calculation progresses
    ///
    /// If this is enabled then the programme will check the calculated lesser Green's functions
    /// are anti-hermitian by confirming
    /// G^< = - [G^<] ^ \dag
    /// In addition it will confirm that all dense lesser self-energies are Hermitian
    /// \Sigma^< = [\Sigma^<] ^ \dag
    /// This adds a (small) overhead to the calculations as each check requires a new matrix allocation
    /// and as each check is carried out for all combinations of energy and wavevector this can balloon.
    pub(crate) security_checks: bool,
    /// Whether the calculation is incoherent or coherent, and the current value of the source-drain potential difference
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

    pub(crate) fn security_checks(&self) -> bool {
        self.security_checks
    }
}
