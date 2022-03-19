/// This module solves the Poisson equation for electron transport on a cell-centered
/// grid.
///
/// In 1-dimension the Poisson equation is
/// $ \mathrm{d}^2 / \mathrm{d} x^2 \phi = F
use nalgebra::{DimMin, DimName};

mod allocators;
mod jacobian;
mod operator;
mod poisson1dsource;
mod solve;
mod source;

pub trait SmallDim: DimName + DimMin<Self, Output = Self> {}
impl<D> SmallDim for D where D: DimName + DimMin<Self, Output = Self> {}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
