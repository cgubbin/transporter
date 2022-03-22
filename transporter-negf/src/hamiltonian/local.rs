/// Submodule to construct local elements of the system Hamiltonian, ie: those associated with a
/// single block for the given number of bands and degree of nearest neighbour coupling
use nalgebra::{allocator::Allocator, DMatrix, DMatrixSliceMut, DefaultAllocator, RealField};
use transporter_mesher::{Connectivity, FiniteDifferenceMesh, Mesh, SmallDim};

pub trait ElementConnectivityAssembler {
    fn solution_dim(&self) -> usize;
    fn num_cells(&self) -> usize;
    fn num_nodes(&self) -> usize;
    fn cell_connection_count(&self, cell_index: usize) -> usize;
    fn populate_cell_connections(&self, output: &mut [usize], cell_index: usize);
}

impl<T, GeometryDim, C> ElementConnectivityAssembler for Mesh<T, GeometryDim, C>
where
    T: RealField,
    GeometryDim: SmallDim,
    C: Connectivity,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn solution_dim(&self) -> usize {
        1
    }
    fn num_cells(&self) -> usize {
        self.vertices().len() - 1
    }
    fn num_nodes(&self) -> usize {
        self.vertices().len()
    }

    fn cell_connection_count(&self, cell_index: usize) -> usize {
        self.connectivity()[cell_index].len()
    }

    fn populate_cell_connections(&self, output: &mut [usize], cell_index: usize) {
        output.copy_from_slice(self.connectivity()[cell_index])
    }
}

pub struct AggregateCellAssembler<'a, CellAssembler> {
    assemblers: &'a [CellAssembler],
    solution_dim: usize,
    num_cells: usize,
    num_nodes: usize,
    num_bands: usize,
    cell_offsets: Vec<usize>,
}

impl<'a, CellAssembler> ElementConnectivityAssembler for AggregateCellAssembler<'a, CellAssembler>
where
    CellAssembler: ElementConnectivityAssembler,
{
    fn solution_dim(&self) -> usize {
        self.solution_dim
    }

    fn num_cells(&self) -> usize {
        self.num_cells
    }

    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    fn cell_connection_count(&self, aggregate_cell_index: usize) -> usize {
        let (assembler, cell_offset) =
            self.find_assembler_and_offset_for_element_index(aggregate_cell_index);
        assembler.cell_connection_count(aggregate_cell_index - cell_offset)
    }

    fn populate_cell_connections(&self, output: &mut [usize], aggregate_cell_index: usize) {
        let (assembler, cell_offset) =
            self.find_assembler_and_offset_for_element_index(aggregate_cell_index);
        assembler.populate_cell_connections(output, aggregate_cell_index - cell_offset)
    }
}

impl<'a, CellAssembler> AggregateCellAssembler<'a, CellAssembler>
where
    CellAssembler: ElementConnectivityAssembler,
{
    pub fn from_assemblers(assemblers: &'a [CellAssembler]) -> Self {
        assert!(
            !assemblers.is_empty(),
            "The aggregate Hamiltonian must have at least one (1) assembler."
        );
        let solution_dim = assemblers[0].solution_dim();
        let num_nodes = assemblers[0].num_nodes();
        assert!(
            assemblers
                .iter()
                .all(|assembler| assembler.solution_dim() == solution_dim),
            "All assemblers must have the same solution dimension"
        );
        assert!(
            assemblers
                .iter()
                .all(|assembler| assembler.num_nodes() == num_nodes),
            "All assemblers must have the same node index space (same num_nodes)"
        );
        let mut num_total_cells = 0;
        let mut cell_offsets = Vec::with_capacity(assemblers.len());
        for assembler in assemblers {
            cell_offsets.push(num_total_cells);
            num_total_cells += assembler.num_cells();
        }
        Self {
            assemblers,
            solution_dim,
            num_cells: assemblers[0].num_cells(),
            num_nodes,
            num_bands: 1,
            cell_offsets,
        }
    }

    fn find_assembler_and_offset_for_element_index(
        &self,
        element_index: usize,
    ) -> (&CellAssembler, usize) {
        assert!(element_index <= self.num_cells);
        let assembler_idx = match self.cell_offsets.binary_search(&element_index) {
            Ok(idx) => idx,
            Err(idx) => idx - 1,
        };
        (
            &self.assemblers[assembler_idx],
            self.cell_offsets[assembler_idx],
        )
    }
}

pub trait AssembleCellMatrix<T: RealField>: ElementConnectivityAssembler {
    fn assemble_cell_matrix_into(
        &self,
        element_index: usize,
        output: DMatrixSliceMut<T>,
    ) -> color_eyre::Result<()>;

    fn assemble_cell_matrix(&self, element_index: usize) -> color_eyre::Result<DMatrix<T>> {
        let ndof = self.solution_dim(); // Todo propagate the number of bands into this functoin
        let mut output = DMatrix::zeros(ndof, ndof);
        self.assemble_cell_matrix_into(element_index, DMatrixSliceMut::from(&mut output))?;
        Ok(output)
    }
}

impl<'a, T, CellAssembler> AssembleCellMatrix<T> for AggregateCellAssembler<'a, CellAssembler>
where
    T: RealField,
    CellAssembler: AssembleCellMatrix<T>,
{
    fn assemble_cell_matrix_into(
        &self,
        aggregate_element_index: usize,
        output: DMatrixSliceMut<T>,
    ) -> color_eyre::Result<()> {
        let (assembler, element_offset) =
            self.find_assembler_and_offset_for_element_index(aggregate_element_index);
        assembler.assemble_cell_matrix_into(aggregate_element_index - element_offset, output)
    }
}

#[derive(Debug, Clone)]
pub struct ElementHamiltonianAssembler<'a, Space> {
    space: &'a Space,
    num_bands: usize,
    solution_dim: usize,
}

impl<'a> ElementHamiltonianAssembler<'a, ()> {
    pub fn new() -> Self {
        Self {
            space: &(),
            num_bands: 0,
            solution_dim: 0,
        }
    }

    pub fn with_solution_dim(self, solution_dim: usize) -> Self {
        Self {
            space: self.space,
            num_bands: self.num_bands,
            solution_dim,
        }
    }

    pub fn with_num_bands(self, num_bands: usize) -> Self {
        Self {
            space: self.space,
            num_bands,
            solution_dim: self.solution_dim,
        }
    }
}

impl<'a> ElementHamiltonianAssembler<'a, ()> {
    pub fn with_space<Space>(self, space: &'a Space) -> ElementHamiltonianAssembler<'a, Space> {
        ElementHamiltonianAssembler {
            space,
            num_bands: self.num_bands,
            solution_dim: self.solution_dim,
        }
    }
}

impl<'a, Space> ElementConnectivityAssembler for ElementHamiltonianAssembler<'a, Space>
where
    Space: ElementConnectivityAssembler,
{
    fn solution_dim(&self) -> usize {
        1
    }
    fn num_cells(&self) -> usize {
        self.space.num_cells()
    }
    fn num_nodes(&self) -> usize {
        self.space.num_nodes()
    }

    fn cell_connection_count(&self, cell_index: usize) -> usize {
        self.space.cell_connection_count(cell_index)
    }

    fn populate_cell_connections(&self, output: &mut [usize], cell_index: usize) {
        self.space.populate_cell_connections(output, cell_index)
    }
}

pub struct ElementInSpace<'a, Space> {
    space: &'a Space,
    element_index: usize,
}

impl<'a, Space> ElementInSpace<'a, Space> {
    fn from_space_and_element_index(space: &'a Space, element_index: usize) -> Self {
        Self {
            space,
            element_index,
        }
    }
}

impl<'a, Space, T> AssembleCellMatrix<T> for ElementHamiltonianAssembler<'a, Space>
where
    T: RealField,
    Space: FiniteDifferenceMesh<T> + ElementConnectivityAssembler,
    DefaultAllocator: Allocator<T, Space::GeometryDim>,
{
    /// Assembles the cell matrix, forming an `num_bands` row array with
    /// `num_connections * num_nearest_neighbours + 1` columns in each row
    fn assemble_cell_matrix_into(
        &self,
        element_index: usize,
        output: DMatrixSliceMut<T>,
    ) -> color_eyre::Result<()> {
        let element = ElementInSpace::from_space_and_element_index(self.space, element_index);
        assemble_element_hamiltonian(output, &element, self.num_bands, self.num_cells())
    }
}

fn assemble_element_hamiltonian<T, Element>(
    output: DMatrixSliceMut<T>,
    element: &Element,
    num_bands: usize,
    num_cells: usize,
) -> color_eyre::Result<()> {
    let shape = output.shape();
    assert_eq!(
        shape.1, 3,
        "Output matrix should have `n_conns * n_neighbour + 1` columns"
    );
    assert_eq!(
        shape.0, num_bands,
        "Output matrix should have `n_bands` rows"
    );
    for band in 0..num_bands {
        todo!()
    }
    Ok(())
}
