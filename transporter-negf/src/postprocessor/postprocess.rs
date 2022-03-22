use super::{ChargeAndCurrent, PostProcessor};

pub(crate) trait PostProcess<T> {
    fn recompute_currents_and_densities(&self) -> color_eyre::Result<ChargeAndCurrent<T>>;
}

impl<T, Mesh> PostProcess<T> for PostProcessor<'_, T, Mesh> {
    fn recompute_currents_and_densities(&self) -> color_eyre::Result<ChargeAndCurrent<T>> {
        todo!()
    }
}
