mod methods;

pub(crate) struct InnerLoopBuilder<T, RefMesh> {
    mesh: RefMesh,
    tolerance: T,
}

impl InnerLoopBuilder<(), ()> {
    fn new() -> Self {
        Self {
            mesh: (),
            tolerance: (),
        }
    }
}

impl<RefMesh> InnerLoopBuilder<(), RefMesh> {
    fn with_mesh<Mesh>(self, mesh: &Mesh) -> InnerLoopBuilder<(), &Mesh> {
        InnerLoopBuilder {
            mesh,
            tolerance: self.tolerance,
        }
    }

    fn with_tolerance<T>(self, tolerance: T) -> InnerLoopBuilder<T, RefMesh> {
        InnerLoopBuilder {
            mesh: self.mesh,
            tolerance,
        }
    }
}

impl<'a, Mesh, T> InnerLoopBuilder<T, &'a Mesh> {
    fn build(self) -> InnerLoop<'a, T, Mesh> {
        InnerLoop {
            mesh: self.mesh,
            tolerance: self.tolerance,
        }
    }
}

pub(crate) struct InnerLoop<'a, T, Mesh> {
    mesh: &'a Mesh,
    tolerance: T,
}
