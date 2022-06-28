mod charge_and_current;
mod postprocess;

pub(crate) use charge_and_current::{Charge, ChargeAndCurrent, Current};
pub(crate) use postprocess::PostProcess;

use miette::Diagnostic;
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator};
use ndarray::Array1;
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

#[derive(Clone, Debug)]
pub(crate) struct ScatteringRate<T> {
    pub(crate) rate: Array1<T>,
}

impl<T> ScatteringRate<T> {
    pub(crate) fn new(rate: Array1<T>) -> Self {
        Self { rate }
    }
}

#[derive(thiserror::Error, Debug, Diagnostic)]
pub enum PostProcessorError {
    #[error(transparent)]
    InconsistentDimensions(#[from] anyhow::Error),
}

pub(crate) struct PostProcessorBuilder<T, RefMesh> {
    mesh: RefMesh,
    marker: PhantomData<T>,
}

pub(crate) struct PostProcessor<'a, T, GeometryDim, Conn>
where
    T: ComplexField,
    Conn: Connectivity<T::RealField, GeometryDim>,
    GeometryDim: SmallDim,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    mesh: &'a Mesh<T::RealField, GeometryDim, Conn>,
    marker: PhantomData<T>,
}

impl<T: ComplexField> PostProcessorBuilder<T, ()> {
    pub(crate) fn new() -> Self {
        PostProcessorBuilder {
            mesh: (),
            marker: PhantomData,
        }
    }
}

impl<T: ComplexField, RefMesh> PostProcessorBuilder<T, RefMesh> {
    pub(crate) fn with_mesh<Mesh>(self, mesh: &Mesh) -> PostProcessorBuilder<T, &Mesh> {
        PostProcessorBuilder {
            mesh,
            marker: PhantomData,
        }
    }
}

impl<'a, T, GeometryDim, Conn> PostProcessorBuilder<T, &'a Mesh<T::RealField, GeometryDim, Conn>>
where
    T: ComplexField,
    Conn: Connectivity<T::RealField, GeometryDim>,
    GeometryDim: SmallDim,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    pub(crate) fn build(self) -> PostProcessor<'a, T, GeometryDim, Conn> {
        PostProcessor {
            mesh: self.mesh,
            marker: PhantomData,
        }
    }
}
