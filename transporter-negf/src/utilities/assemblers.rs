use nalgebra::{allocator::Allocator, DVector, DVectorSliceMut, DefaultAllocator, RealField};
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

/// Trait giving the information necessary to construct the row of differential operator for a single vertex in a mesh
pub trait VertexConnectivityAssembler {
    /// Returns the dimension of the solution as a `usize`: for our problems this is always 1 so we can probably delete the method
    fn solution_dim(&self) -> usize;
    /// The number of elements contained in the entire mesh
    fn num_elements(&self) -> usize;
    /// The number of vertices, or nodes contained in the entire mesh
    fn num_vertices(&self) -> usize;
    /// The number of vertices connected to the vertex at `vertex_index`
    fn vertex_connection_count(&self, vertex_index: usize) -> usize;
    /// Populates the indices of vertices connected to the vertex at `vertex_index` into the slice `output`. The passed slice
    /// must have length `self.vertex_connection_count(vertex_index)`
    fn populate_vertex_connections(&self, output: &mut [usize], element_index: usize);
}

/// Implement `ElementConnectivityAssembler` for the generic `Mesh`
impl<T, GeometryDim, C> VertexConnectivityAssembler for Mesh<T, GeometryDim, C>
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    C: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn solution_dim(&self) -> usize {
        1
    }
    fn num_elements(&self) -> usize {
        self.elements().len()
    }
    fn num_vertices(&self) -> usize {
        self.vertices().len()
    }

    fn vertex_connection_count(&self, vertex_index: usize) -> usize {
        self.connectivity()[vertex_index].len()
    }

    fn populate_vertex_connections(&self, output: &mut [usize], vertex_index: usize) {
        output.copy_from_slice(self.connectivity()[vertex_index])
    }
}

/// Helper trait to construct the diagonal elements of a differential operator
pub(crate) trait AssembleVertexDiagonal<T: RealField>: VertexConnectivityAssembler {
    /// Assembles the wavevector component into `output` for the element at `element_index`. Takes an output vector of length
    /// `num_bands` which is enforced by an assertion
    fn assemble_vertex_diagonal(&self, vertex_index: usize) -> color_eyre::Result<T>;
}

/// Helper trait to construct the fixed component of the operator (ie: the differential bit)
pub(crate) trait AssembleVertexMatrix<T: RealField>: VertexConnectivityAssembler {
    /// Takes an output matrix of dimension `num_bands` * `num_connections + 1` which is enforced by an assertion, and fills with the fixed
    ///  component of the Hamiltonian
    fn assemble_vertex_matrix_into(
        &self,
        vertex_index: usize,
        output: DVectorSliceMut<T>,
    ) -> color_eyre::Result<()>;

    fn assemble_vertex_matrix(
        &self,
        vertex_index: usize,
        num_connections: usize,
    ) -> color_eyre::Result<DVector<T>> {
        let mut output = DVector::zeros(num_connections + 1);
        self.assemble_vertex_matrix_into(vertex_index, DVectorSliceMut::from(&mut output))?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
/// An assembler for a single vertex in the mesh
pub struct VertexAssembler<'a, T, InfoDesk, Mesh> {
    /// The `InfoDesk` provides all the external information necessary to construct the operator
    pub(crate) info_desk: &'a InfoDesk,
    /// The `Mesh` tells us which elements our element is connected to, and how far away they are
    pub(crate) mesh: &'a Mesh,
    /// A marker so we can be generic over the field `T`. This can probably go away with some forethought
    __marker: PhantomData<T>,
}

/// Factory builder for an `ElementPoissonAssembler`
pub struct VertexAssemblerBuilder<T, RefInfoDesk, RefMesh> {
    /// Reference to an `InfoDesk` which must impl `PoissonInfoDesk<T>`
    info_desk: RefInfoDesk,
    /// Reference to the structure mesh
    mesh: RefMesh,
    /// Marker allowing the structure to be generic over the field `T`
    marker: PhantomData<T>,
}

impl<T> VertexAssemblerBuilder<T, (), ()> {
    /// Initialise the builder
    pub(crate) fn new() -> Self {
        Self {
            info_desk: (),
            mesh: (),
            marker: PhantomData,
        }
    }
}

impl<T, RefInfoDesk, RefMesh> VertexAssemblerBuilder<T, RefInfoDesk, RefMesh> {
    /// Attach the info desk
    pub(crate) fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> VertexAssemblerBuilder<T, &InfoDesk, RefMesh> {
        VertexAssemblerBuilder {
            info_desk,
            mesh: self.mesh,
            marker: PhantomData,
        }
    }
    /// Attach the mesh
    pub(crate) fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> VertexAssemblerBuilder<T, RefInfoDesk, &Mesh> {
        VertexAssemblerBuilder {
            info_desk: self.info_desk,
            mesh,
            marker: PhantomData,
        }
    }
}

impl<'a, T, InfoDesk, Mesh> VertexAssemblerBuilder<T, &'a InfoDesk, &'a Mesh> {
    /// Build out the VertexAssembler from the builder
    pub(crate) fn build(self) -> VertexAssembler<'a, T, InfoDesk, Mesh> {
        VertexAssembler {
            info_desk: self.info_desk,
            mesh: self.mesh,
            __marker: PhantomData,
        }
    }
}

/// Inplement `ElementConnectivityAssembler` for the element assembler to reduce verbiosity.
impl<'a, T: Copy + RealField, InfoDesk, Mesh> VertexConnectivityAssembler
    for VertexAssembler<'a, T, InfoDesk, Mesh>
where
    Mesh: VertexConnectivityAssembler,
{
    fn solution_dim(&self) -> usize {
        1
    }

    fn num_elements(&self) -> usize {
        self.mesh.num_elements()
    }
    fn num_vertices(&self) -> usize {
        self.mesh.num_vertices()
    }

    fn vertex_connection_count(&self, vertex_index: usize) -> usize {
        self.mesh.vertex_connection_count(vertex_index)
    }

    fn populate_vertex_connections(&self, output: &mut [usize], element_index: usize) {
        self.mesh.populate_vertex_connections(output, element_index)
    }
}

#[derive(Debug)]
/// Struct to
pub struct AggregateVertexAssembler<'a, VertexAssembler> {
    assemblers: &'a [VertexAssembler],
    solution_dim: usize,
    num_elements: usize,
    num_vertices: usize,
    vertex_offsets: Vec<usize>,
}

impl<'a, VertexAssembler> VertexConnectivityAssembler
    for AggregateVertexAssembler<'a, VertexAssembler>
where
    VertexAssembler: VertexConnectivityAssembler,
{
    fn solution_dim(&self) -> usize {
        self.solution_dim
    }

    fn num_elements(&self) -> usize {
        self.num_elements
    }

    fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    fn vertex_connection_count(&self, aggregate_vertex_index: usize) -> usize {
        let (assembler, vertex_offset) =
            self.find_assembler_and_offset_for_vertex_index(aggregate_vertex_index);
        assembler.vertex_connection_count(aggregate_vertex_index - vertex_offset)
    }

    fn populate_vertex_connections(&self, output: &mut [usize], aggregate_vertex_index: usize) {
        let (assembler, vertex_offset) =
            self.find_assembler_and_offset_for_vertex_index(aggregate_vertex_index);
        assembler.populate_vertex_connections(output, aggregate_vertex_index - vertex_offset)
    }
}

impl<'a, VertexAssembler> AggregateVertexAssembler<'a, VertexAssembler>
where
    VertexAssembler: VertexConnectivityAssembler,
{
    pub fn from_assemblers(assemblers: &'a [VertexAssembler]) -> Self {
        assert!(
            !assemblers.is_empty(),
            "The aggregate Hamiltonian must have at least one (1) assembler."
        );
        let solution_dim = assemblers[0].solution_dim();
        let num_vertices = assemblers[0].num_vertices();
        assert!(
            assemblers
                .iter()
                .all(|assembler| assembler.solution_dim() == solution_dim),
            "All assemblers must have the same solution dimension"
        );
        assert!(
            assemblers
                .iter()
                .all(|assembler| assembler.num_vertices() == num_vertices),
            "All assemblers must have the same node index space (same num_nodes)"
        );
        let mut num_total_vertices = 0;
        let mut vertex_offsets = Vec::with_capacity(assemblers.len());
        for assembler in assemblers {
            vertex_offsets.push(num_total_vertices);
            num_total_vertices += assembler.num_vertices();
        }
        Self {
            assemblers,
            solution_dim,
            num_elements: assemblers[0].num_elements(),
            num_vertices,
            vertex_offsets,
        }
    }

    fn find_assembler_and_offset_for_vertex_index(
        &self,
        vertex_index: usize,
    ) -> (&VertexAssembler, usize) {
        assert!(vertex_index <= self.num_vertices);
        let assembler_idx = match self.vertex_offsets.binary_search(&vertex_index) {
            Ok(idx) => idx,
            Err(idx) => idx - 1,
        };
        (
            &self.assemblers[assembler_idx],
            self.vertex_offsets[assembler_idx],
        )
    }
}

impl<'a, T, VertexAssembler> AssembleVertexMatrix<T>
    for AggregateVertexAssembler<'a, VertexAssembler>
where
    T: RealField,
    VertexAssembler: AssembleVertexMatrix<T>,
{
    fn assemble_vertex_matrix_into(
        &self,
        aggregate_vertex_index: usize,
        output: DVectorSliceMut<T>,
    ) -> color_eyre::Result<()> {
        let (assembler, vertex_offset) =
            self.find_assembler_and_offset_for_vertex_index(aggregate_vertex_index);
        assembler.assemble_vertex_matrix_into(aggregate_vertex_index - vertex_offset, output)
    }
}

impl<'a, T, VertexAssembler> AssembleVertexDiagonal<T>
    for AggregateVertexAssembler<'a, VertexAssembler>
where
    T: RealField,
    VertexAssembler: AssembleVertexDiagonal<T>,
{
    fn assemble_vertex_diagonal(&self, aggregate_vertex_index: usize) -> color_eyre::Result<T> {
        let (assembler, vertex_offset) =
            self.find_assembler_and_offset_for_vertex_index(aggregate_vertex_index);
        assembler.assemble_vertex_diagonal(aggregate_vertex_index - vertex_offset)
    }
}
