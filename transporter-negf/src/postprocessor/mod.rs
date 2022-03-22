mod charge_and_current;
mod postprocess;

pub(crate) use charge_and_current::{Charge, ChargeAndCurrent, Current};
pub(crate) use postprocess::PostProcess;

pub(crate) struct PostProcessorBuilder<T, RefMesh> {
    mesh: RefMesh,
    marker: std::marker::PhantomData<T>,
}

pub(crate) struct PostProcessor<'a, T, Mesh> {
    mesh: &'a Mesh,
    marker: std::marker::PhantomData<T>,
}

impl PostProcessorBuilder<(), ()> {
    pub(crate) fn new() -> Self {
        PostProcessorBuilder {
            mesh: (),
            marker: std::marker::PhantomData,
        }
    }
}

impl<RefMesh> PostProcessorBuilder<(), RefMesh> {
    pub(crate) fn with_mesh<Mesh>(self, mesh: &Mesh) -> PostProcessorBuilder<(), &Mesh> {
        PostProcessorBuilder {
            mesh,
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a, Mesh> PostProcessorBuilder<(), &'a Mesh> {
    pub(crate) fn build<T>(self) -> PostProcessor<'a, T, Mesh> {
        PostProcessor {
            mesh: self.mesh,
            marker: std::marker::PhantomData,
        }
    }
}
