use nalgebra::{DVector, RealField};

#[derive(Clone)]
pub(crate) struct ChargeAndCurrent<T> {
    charge: Charge<T>,
    current: Current<T>,
}

impl<T> ChargeAndCurrent<T>
where
    T: RealField,
{
    pub(crate) fn new(
        charge: Vec<DVector<T>>,
        current: Vec<DVector<T>>,
    ) -> color_eyre::Result<Self> {
        Ok(Self {
            charge: Charge::new(charge)?,
            current: Current::new(current)?,
        })
    }

    pub(crate) fn from_charge_and_current(charge: Charge<T>, current: Current<T>) -> Self {
        Self { charge, current }
    }

    pub(crate) fn deref_charge(self) -> Charge<T> {
        self.charge
    }

    /// Returns a chained iterator over both charge and current densities
    fn charge_and_current_iter(
        &self,
    ) -> std::iter::Chain<std::slice::Iter<'_, DVector<T>>, std::slice::Iter<'_, DVector<T>>> {
        self.charge_iter().chain(self.current_iter())
    }

    /// Returns an iterator over the held charge densities
    fn charge_iter(&self) -> std::slice::Iter<'_, DVector<T>> {
        self.charge.charge.iter()
    }

    /// Returns an iterator over the held current densities
    fn current_iter(&self) -> std::slice::Iter<'_, DVector<T>> {
        self.current.current.iter()
    }

    /// Given a tolerance, and a previous value for the charge and current
    /// calculates whether the change in the norms is less than the tolerance.
    pub(crate) fn is_change_within_tolerance(
        &self,
        previous: &ChargeAndCurrent<T>,
        tolerance: T,
    ) -> color_eyre::Result<bool> {
        Ok(self
            .charge_and_current_iter()
            .zip(previous.charge_and_current_iter())
            .map(|(new, previous)| (new - previous, new.norm()))
            .map(|(difference, norm)| difference.norm() / norm)
            .all(|difference| difference < tolerance))
    }
}

#[derive(Clone)]
pub(crate) struct Charge<T> {
    charge: Vec<DVector<T>>,
}

#[derive(Clone)]
pub(crate) struct Current<T> {
    current: Vec<DVector<T>>,
}

impl<T> Charge<T> {
    pub(crate) fn new(charge: Vec<DVector<T>>) -> color_eyre::Result<Self> {
        let length = charge[0].shape();
        if charge.iter().all(|x| x.shape() == length) {
            Ok(Self { charge })
        } else {
            Err(color_eyre::eyre::eyre!(
                "All charges must have the same number of points"
            ))
        }
    }
}

impl<T> Current<T> {
    pub(crate) fn new(current: Vec<DVector<T>>) -> color_eyre::Result<Self> {
        let length = current[0].shape();
        if current.iter().all(|x| x.shape() == length) {
            Ok(Self { current })
        } else {
            Err(color_eyre::eyre::eyre!(
                "All currents must have the same number of points"
            ))
        }
    }
}
