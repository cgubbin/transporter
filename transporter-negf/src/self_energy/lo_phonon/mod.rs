// TODO The logic is replicated across both sub-crates, only the iterators over the 2D matrix rows is different:
// this could be improved by moving teh logic into here, and only placing the iteration in the sub-crates

#[cfg(not(feature = "ndarray"))]
pub(crate) mod nalgebra;

#[cfg(feature = "ndarray")]
pub(crate) mod ndarray;
