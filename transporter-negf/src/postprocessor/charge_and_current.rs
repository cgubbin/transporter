use nalgebra::{
    allocator::Allocator, Const, DVector, DefaultAllocator, Dynamic, Matrix, OVector, RealField,
    VecStorage,
};
use transporter_mesher::SmallDim;

#[derive(Clone)]
pub(crate) struct ChargeAndCurrent<T, BandDim: SmallDim>
where
    DefaultAllocator: Allocator<
        Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
        BandDim,
    >,
{
    charge: Charge<T, BandDim>,
    current: Current<T, BandDim>,
}

impl<T, BandDim> ChargeAndCurrent<T, BandDim>
where
    T: Copy + RealField,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<
        Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
        BandDim,
    >,
{
    pub(crate) fn new(
        charge: OVector<DVector<T>, BandDim>,
        current: OVector<DVector<T>, BandDim>,
    ) -> color_eyre::Result<Self> {
        Ok(Self {
            charge: Charge::new(charge)?,
            current: Current::new(current)?,
        })
    }

    pub(crate) fn from_charge_and_current(
        charge: Charge<T, BandDim>,
        current: Current<T, BandDim>,
    ) -> Self {
        Self { charge, current }
    }

    pub(crate) fn deref_charge(self) -> Charge<T, BandDim> {
        self.charge
    }

    /// Returns a chained iterator over both charge and current densities
    fn charge_and_current_iter(
        &self,
        // ) -> std::iter::Chain<impl Iterator<Item = DVector<T>>, impl Iterator<Item = DVector<T>>> {
    ) -> impl Iterator<Item = DVector<T>> {
        self.charge_iter() //.chain(self.current_iter())
    }

    /// Returns an iterator over the held charge densities
    #[allow(clippy::needless_collect)]
    fn charge_iter(&self) -> impl Iterator<Item = DVector<T>> {
        let x: Vec<DVector<T>> = (0..BandDim::dim())
            .map(|i| self.charge.charge[i].clone())
            .collect::<Vec<_>>();
        x.into_iter()
    }

    #[allow(clippy::needless_collect)] // Allowed so we can return a DVector over generic soup
    fn current_iter(&self) -> impl Iterator<Item = DVector<T>> {
        let x: Vec<DVector<T>> = (0..BandDim::dim())
            .map(|i| self.current.current[i].clone())
            .collect::<Vec<_>>();
        x.into_iter()
    }

    pub(crate) fn charge_as_ref(&self) -> &Charge<T, BandDim> {
        &self.charge
    }

    /// Given a tolerance, and a previous value for the charge and current
    /// calculates whether the change in the norms is less than the tolerance.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub(crate) fn is_change_within_tolerance(
        &self,
        previous: &ChargeAndCurrent<T, BandDim>,
        tolerance: T,
    ) -> color_eyre::Result<bool> {
        Ok(self
            .charge_and_current_iter()
            .zip(previous.charge_and_current_iter())
            .map(|(new, previous)| {
                let norm = new.norm();
                (new - previous, norm)
            })
            .map(|(difference, norm)| {
                let difference = difference.norm() / (norm);
                tracing::info!("Difference {difference}, tolerance {tolerance}");
                difference
            }) // This breaks when norm is zero, need to compute currents for a result
            .filter(|&x| {
                !((x != T::zero()) & !(x > T::zero())) // Hacky trick to filter NaN from the result
            })
            .all(|difference| difference < tolerance))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Charge<T, BandDim: SmallDim>
where
    DefaultAllocator: Allocator<
        Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
        BandDim,
    >,
{
    charge: OVector<DVector<T>, BandDim>,
}

impl<T: RealField, BandDim: SmallDim> AsRef<OVector<DVector<T>, BandDim>> for Charge<T, BandDim>
where
    DefaultAllocator: Allocator<
        Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
        BandDim,
    >,
{
    fn as_ref(&self) -> &OVector<DVector<T>, BandDim> {
        &self.charge
    }
}

impl<T: RealField, BandDim: SmallDim> Charge<T, BandDim>
where
    DefaultAllocator: Allocator<
        Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
        BandDim,
    >,
{
    pub(crate) fn net_charge(&self) -> DVector<T> {
        self.charge.iter().sum()
    }

    pub(crate) fn as_ref_mut(&mut self) -> &mut OVector<DVector<T>, BandDim> {
        &mut self.charge
    }
}
#[derive(Clone)]
pub(crate) struct Current<T, BandDim: SmallDim>
where
    DefaultAllocator: Allocator<
        Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
        BandDim,
    >,
{
    current: OVector<DVector<T>, BandDim>,
}

impl<T, BandDim: SmallDim> Charge<T, BandDim>
where
    DefaultAllocator: Allocator<
        Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
        BandDim,
    >,
{
    pub(crate) fn new(charge: OVector<DVector<T>, BandDim>) -> color_eyre::Result<Self> {
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

impl<T, BandDim: SmallDim> Current<T, BandDim>
where
    DefaultAllocator: Allocator<
        Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
        BandDim,
    >,
{
    pub(crate) fn new(current: OVector<DVector<T>, BandDim>) -> color_eyre::Result<Self> {
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
