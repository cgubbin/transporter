//! Global assemblers for the `Hamiltonian`
//!
//! This module provides global assemblers for the `Hamiltonian` matrix, taking the elements produced in the
//! local subcrate and throwing them into a global CsrMatrix

use super::local::{AssembleVertexHamiltonianDiagonal, AssembleVertexHamiltonianMatrix};
use super::{BuildError, CsrError, HamiltonianInfoDesk, PotentialInfoDesk};
use crate::utilities::assemblers::VertexConnectivityAssembler;
use nalgebra::{
    allocator::Allocator, DMatrix, DVector, DefaultAllocator, Dynamic, Matrix, RealField, Storage,
    U1,
};
use nalgebra_sparse::{
    pattern::{SparsityPattern, SparsityPatternFormatError},
    CsrMatrix,
};
use std::cell::RefCell;

/// An assembler for CSR matrices.
#[derive(Debug, Clone)]
pub struct CsrAssembler<T: RealField> {
    // All members are buffers that help prevent unnecessary allocations
    // when assembling multiple matrices with the same assembler
    workspace: RefCell<CsrAssemblerWorkspace<T>>,
}

#[derive(Debug, Clone)]
struct CsrAssemblerWorkspace<T: RealField> {
    // The sparsity pattern on the diagonal of the CsrMatrix
    diagonal_sparsity_pattern: SparsityPattern,
    /// The complete SparsityPattern
    full_sparsity_pattern: SparsityPattern,
    /// Scratch space for the element_matrix constructor
    vertex_matrix: DMatrix<T>,
    /// Scratch space for the element vector constructor
    vertex_vector: DVector<T>,
}

impl<T: RealField> CsrAssemblerWorkspace<T> {
    fn vertex_matrix(&self) -> &DMatrix<T> {
        &self.vertex_matrix
    }
    fn vertex_matrix_mut(&mut self) -> &mut DMatrix<T> {
        &mut self.vertex_matrix
    }
    fn vertex_vector(&self) -> &DVector<T> {
        &self.vertex_vector
    }
    fn vertex_vector_mut(&mut self) -> &mut DVector<T> {
        &mut self.vertex_vector
    }
}

/// Initialisation methods for the CsrAssembler
///
/// Constructs a CsrAssembler from the element assembler, initialising the scratch space and
/// sparsity patterns
impl<T: Copy + RealField> CsrAssembler<T> {
    pub(crate) fn from_vertex_assembler<Assembler>(
        vertex_assembler: &Assembler,
    ) -> Result<Self, CsrError>
    where
        Assembler: VertexConnectivityAssembler + HamiltonianInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let full_sparsity_pattern = CsrAssembler::assemble_full_sparsity_pattern(vertex_assembler)?;
        let diagonal_sparsity_pattern =
            CsrAssembler::assemble_diagonal_sparsity_pattern(vertex_assembler)?;

        Ok(Self {
            workspace: RefCell::new(CsrAssemblerWorkspace {
                diagonal_sparsity_pattern,
                full_sparsity_pattern,
                vertex_matrix: DMatrix::from_element(
                    vertex_assembler.number_of_bands(),
                    vertex_assembler.vertex_connection_count(1) + 1,
                    T::zero(),
                ),
                vertex_vector: DVector::from_element(vertex_assembler.number_of_bands(), T::zero()),
            }),
        })
    }

    /// Construct the full CsrMatrix sparsity pattern from the element assembler -
    fn assemble_full_sparsity_pattern<Assembler>(
        vertex_assembler: &Assembler,
    ) -> Result<SparsityPattern, SparsityPatternFormatError>
    where
        Assembler: VertexConnectivityAssembler + HamiltonianInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let sdim = vertex_assembler.solution_dim();
        let n_cells = vertex_assembler.num_elements() + 1;
        let num_rows = sdim * vertex_assembler.num_vertices() * vertex_assembler.number_of_bands();

        let mut vertex_connections = Vec::new();
        let mut matrix_entries = std::collections::BTreeSet::new();
        for i in 0..vertex_assembler.num_vertices() {
            let vertex_connection_count = vertex_assembler.vertex_connection_count(i);
            vertex_connections.resize(vertex_connection_count, usize::MAX);
            vertex_assembler.populate_vertex_connections(&mut vertex_connections, i);
            for n_band in 0..vertex_assembler.number_of_bands() {
                matrix_entries.insert((n_band * n_cells + i, n_band * n_cells + i)); // The diagonal element
                for j in &vertex_connections {
                    matrix_entries.insert((n_band * n_cells + i, n_band * n_cells + j));
                    // The hopping elements
                }
            }
        }

        let mut offsets = Vec::with_capacity(num_rows + 1);
        let mut column_indices = Vec::with_capacity(matrix_entries.len());
        offsets.push(0);
        for (i, j) in matrix_entries {
            while i + 1 > offsets.len() {
                offsets.push(column_indices.len());
            }
            column_indices.push(j);
        }

        while offsets.len() < (num_rows + 1) {
            offsets.push(column_indices.len())
        }

        SparsityPattern::try_from_offsets_and_indices(num_rows, num_rows, offsets, column_indices)
    }

    /// Assemble the sparsity pattern for the diagonal of the CsrMatrix
    fn assemble_diagonal_sparsity_pattern<Assembler>(
        vertex_assembler: &Assembler,
    ) -> Result<SparsityPattern, SparsityPatternFormatError>
    where
        Assembler: VertexConnectivityAssembler + HamiltonianInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let sdim = vertex_assembler.solution_dim();
        let num_rows = sdim * vertex_assembler.num_vertices() * vertex_assembler.number_of_bands();

        let offsets = (0..num_rows + 1).collect::<Vec<_>>();
        let column_indices = (0..num_rows).collect::<Vec<_>>();

        SparsityPattern::try_from_offsets_and_indices(num_rows, num_rows, offsets, column_indices)
    }
}

/// High level constructors used to initialise the Hamiltonian
impl<T: Copy + RealField> CsrAssembler<T> {
    /// Assembles the fixed component of the Hamiltonian: ie that which is independent of potential and wavevector
    pub(crate) fn assemble_fixed<Assembler>(
        &self,
        vertex_assembler: &Assembler,
    ) -> Result<CsrMatrix<T>, BuildError>
    where
        Assembler: AssembleVertexHamiltonianMatrix<T> + HamiltonianInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let pattern = self.workspace.borrow().full_sparsity_pattern.clone();
        let initial_matrix_values = vec![T::zero(); pattern.nnz()];
        let mut matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_matrix_values)
            .expect("CSR data must be valid by definition");
        self.assemble_into_csr(&mut matrix, vertex_assembler)?;
        Ok(matrix)
    }

    /// Assembles the component of the Hamiltonian proportional to the transverse wavevector
    pub(crate) fn assemble_wavevector<Assembler>(
        &self,
        vertex_assembler: &Assembler,
    ) -> Result<CsrMatrix<T>, BuildError>
    where
        Assembler: AssembleVertexHamiltonianDiagonal<T> + HamiltonianInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let pattern = self.workspace.borrow().diagonal_sparsity_pattern.clone();
        let initial_matrix_values = vec![T::zero(); pattern.nnz()];
        let mut matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_matrix_values)
            .expect("CSR data must be valid by definition");
        self.assemble_into_csr_diagonal(&mut matrix, vertex_assembler)?;
        Ok(matrix)
    }

    /// Assembles the potential into a diagonal CsrMatrix: This method is only called in the constructor
    pub(crate) fn assemble_potential<Assembler>(
        &self,
        vertex_assembler: &Assembler,
    ) -> Result<CsrMatrix<T>, BuildError>
    where
        Assembler: VertexConnectivityAssembler + PotentialInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::BandDim>,
    {
        let pattern = self.workspace.borrow().diagonal_sparsity_pattern.clone();
        let initial_matrix_values = vec![T::zero(); pattern.nnz()];
        let mut matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_matrix_values)
            .expect("CSR data must be valid by definition");
        CsrAssembler::assemble_potential_into_csr_diagonal(&mut matrix, vertex_assembler)?;
        Ok(matrix)
    }

    /// Assemble the potential into a pre-initialised CsrMatrix -> This method is called after each update of the potential
    pub(crate) fn assemble_potential_into<Assembler>(
        vertex_assembler: &Assembler,
        potential: &mut CsrMatrix<T>,
    ) -> Result<(), BuildError>
    where
        Assembler: VertexConnectivityAssembler + PotentialInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::BandDim>,
    {
        CsrAssembler::assemble_potential_into_csr_diagonal(potential, vertex_assembler)?;
        Ok(())
    }
}

/// Lower level constructors to compute the local assemblers  for each element in the mesh
impl<T: Copy + RealField> CsrAssembler<T> {
    /// Assembles the fixed component of the Hamiltonian into the CsrMatrix `csr`
    fn assemble_into_csr<Assembler>(
        &self,
        csr: &mut CsrMatrix<T>,
        vertex_assembler: &Assembler,
    ) -> Result<(), BuildError>
    where
        Assembler: AssembleVertexHamiltonianMatrix<T> + HamiltonianInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let sdim = vertex_assembler.solution_dim();
        let num_single_band_rows = sdim * vertex_assembler.num_vertices(); // We have an issue with cells and nodes, this needs to be pinned down

        let mut workspace = self.workspace.borrow_mut();

        // Assemble the differential operator for the Hamiltonian
        for n_row in 0..num_single_band_rows {
            // This is still annoying because we have less connections at the edges so have to
            // re-initialise this matrix on every loop. Can we refactor the mesh to avoid this problem
            let num_connections = vertex_assembler.vertex_connection_count(n_row);
            // The element matrix has `num_connections + 1` elements for each band in a nearest-neighbour model
            // This is mainly just a pull from the `workspace`, the size only changes for the edge elements
            // let mut element_matrix = self.workspace.into_inner().element_matrix.resize(
            let vertex_matrix = workspace.vertex_matrix_mut();
            let matrix_slice = vertex_matrix.columns_mut(0, num_connections + 1);
            vertex_assembler.assemble_vertex_matrix_into(n_row, matrix_slice)?;
            for n_band in 0..vertex_assembler.number_of_bands() {
                let band_row = workspace.vertex_matrix().row(n_band);
                let mut csr_row = csr.row_mut(n_row + n_row * n_band);
                let values = csr_row.values_mut();
                add_row_to_csr_row(values, band_row);
            }
        }

        Ok(())
    }

    fn assemble_into_csr_diagonal<Assembler>(
        &self,
        csr: &mut CsrMatrix<T>,
        vertex_assembler: &Assembler,
    ) -> Result<(), BuildError>
    where
        Assembler: AssembleVertexHamiltonianDiagonal<T> + HamiltonianInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let sdim = vertex_assembler.solution_dim();
        let num_single_band_rows = sdim * vertex_assembler.num_vertices(); // We have an issue with cells and nodes, this needs to be pinned down

        let mut workspace = self.workspace.borrow_mut();

        // Assemble the differential operator for the Hamiltonian
        for n_row in 0..num_single_band_rows {
            let vertex_vector = workspace.vertex_vector_mut();
            let vector_slice = nalgebra::DVectorSliceMut::from(vertex_vector);
            vertex_assembler.assemble_vertex_diagonal_into(n_row, vector_slice)?;
            for n_band in 0..vertex_assembler.number_of_bands() {
                let band_row = workspace.vertex_vector().row(n_band)[0];
                let mut csr_row = csr.row_mut(n_row + n_row * n_band);
                let values = csr_row.values_mut(); //t
                add_element_to_csr_row_diagonal(values, band_row);
            }
        }
        Ok(())
    }

    /// Assemble the potential from the `element_assembler` into the diagonal CsrMatrix `csr`
    /// TODO -> This will panic when the diagonal contains zeros, which it may well do
    fn assemble_potential_into_csr_diagonal<Assembler>(
        csr: &mut CsrMatrix<T>,
        vertex_assembler: &Assembler,
    ) -> Result<(), CsrError>
    where
        Assembler: VertexConnectivityAssembler + PotentialInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::BandDim>,
    {
        let sdim = vertex_assembler.solution_dim();
        let num_single_band_rows = sdim * vertex_assembler.num_vertices(); // We have an issue with cells and nodes, this needs to be pinned down
        let mut _scratch = vec![0_usize; 1]; // Hacky, add a method;

        // Assemble the potential into a diagonal
        for n_row in 0..num_single_band_rows {
            let potential = vertex_assembler.potential(n_row);
            for n_band in 0..vertex_assembler.number_of_bands() {
                let mut csr_row = csr.row_mut(n_row + n_row * n_band);
                if let Some(diagonal_entry) = csr_row.get_entry_mut(n_row) {
                    match diagonal_entry {
                        nalgebra_sparse::SparseEntryMut::NonZero(x) => *x = potential,
                        _ => {
                            return Err(CsrError::Access(
                                "The diagonal should always be nonzero if filled".into(),
                            ));
                        }
                    }
                } else {
                    return Err(CsrError::Access(
                        "The diagonal should always be filled by the pattern builder".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// A helper method to add an element to the diagonal of a CsrMatrix -> in this method
/// `row_values` is always a slice of length 1
fn add_element_to_csr_row_diagonal<T>(row_values: &mut [T], local_row: T)
where
    T: Copy + RealField,
{
    assert_eq!(
        row_values.len(),
        1,
        "There can only be a single element on a matrix diagonal"
    );
    row_values[0] += local_row;
}

/// Adds a whole row to the CsrMatrix -> This assumes the new row `local_row` has the same ordering as
/// the Csr column indices of `row_values`, or that the elements of `local_row` are ordered in terms of
/// increasing column index
fn add_row_to_csr_row<T, S>(row_values: &mut [T], local_row: Matrix<T, U1, Dynamic, S>)
where
    T: Copy + RealField,
    S: Storage<T, U1, Dynamic>,
{
    for (row_value, value) in row_values.iter_mut().zip(local_row.into_iter()) {
        *row_value += *value;
    }
}

//#[cfg(test)]
//mod test {
//    use super::CsrAssembler;
//    use matrixcompare::assert_matrix_eq;
//    use transporter_mesher::{create_unit_line_segment_mesh_1d, Mesh1d};
//    #[test]
//    fn csr_assembly_pattern_is_correct_for_1d_mesh_and_single_band() {
//        let cells_per_dim = 100;
//        let mesh: Mesh1d<f64> = create_unit_line_segment_mesh_1d(cells_per_dim);
//
//        let assembler: CsrAssembler<f64> = CsrAssembler::default();
//        let pattern = assembler.assemble_pattern(&mesh).unwrap();
//
//        let mut dense = nalgebra::DMatrix::from_element(cells_per_dim + 1, cells_per_dim + 1, 0i8);
//        dense[(0, 0)] = 1;
//        dense[(0, 1)] = 1;
//        for i in 1..cells_per_dim {
//            dense[(i, i)] = 1;
//            dense[(i, i - 1)] = 1;
//            dense[(i, i + 1)] = 1;
//        }
//        dense[(cells_per_dim, cells_per_dim)] = 1;
//        dense[(cells_per_dim, cells_per_dim - 1)] = 1;
//
//        let (offsets, indices) = pattern.disassemble();
//        let initial_matrix_values = vec![1i8; indices.len()];
//        let csr = nalgebra_sparse::CsrMatrix::try_from_csr_data(
//            cells_per_dim + 1,
//            cells_per_dim + 1,
//            offsets,
//            indices,
//            initial_matrix_values,
//        )
//        .unwrap();
//
//        assert_matrix_eq!(csr, dense);
//    }
//
//    #[test]
//    fn csr_assembly_pattern_is_correct_for_1d_mesh_and_two_bands() {
//        let cells_per_dim = 4;
//        let nv = cells_per_dim + 1;
//        let mesh: Mesh1d<f64> = create_unit_line_segment_mesh_1d(cells_per_dim);
//        let num_bands = 2;
//
//        let assembler: CsrAssembler<f64> = CsrAssembler::default();
//        let pattern = assembler.assemble_pattern(&mesh).unwrap();
//
//        let mut dense = nalgebra::DMatrix::from_element(
//            num_bands * (cells_per_dim + 1),
//            num_bands * (cells_per_dim + 1),
//            0i8,
//        );
//        for n_band in 0..num_bands {
//            let off = n_band * nv;
//            dense[(off, off)] = 1;
//            dense[(off, off + 1)] = 1;
//            for i in 1..cells_per_dim {
//                dense[(off + i, off + i)] = 1;
//                dense[(off + i, off + i - 1)] = 1;
//                dense[(off + i, off + i + 1)] = 1;
//            }
//            dense[(off + cells_per_dim, off + cells_per_dim)] = 1;
//            dense[(off + cells_per_dim, off + cells_per_dim - 1)] = 1;
//        }
//
//        let (offsets, indices) = pattern.disassemble();
//        let initial_matrix_values = vec![1i8; indices.len()];
//        let csr = nalgebra_sparse::CsrMatrix::try_from_csr_data(
//            num_bands * (cells_per_dim + 1),
//            num_bands * (cells_per_dim + 1),
//            offsets,
//            indices,
//            initial_matrix_values,
//        )
//        .unwrap();
//
//        assert_matrix_eq!(csr, dense);
//    }
//
//    //::Matrix5x5::new(1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0);
//
//    //let initial_matrix_values = vec![0f64; pattern.nnz()];
//    //let mut csr =
//    //    nalgebra_sparse::CsrMatrix::try_from_pattern_and_values(pattern, initial_matrix_values)
//    //        .expect("CSR data must be valid by definition");
//
//    //let local_row = vec![1f64, 2f64];
//
//    //let local_row = DMatrix::from_row_slice(1, 2, &local_row);
//    //let local_row = local_row.row(0);
//
//    //let n_row = 0;
//    //let n_band = 1;
//    //let mut csr_row = csr.row_mut(n_row + n_row * n_band);
//    //let (cols, values) = csr_row.cols_and_values_mut();
//    //super::add_row_to_csr_row(values, cols, &local_row);
//
//    //dbg!(csr);
//}
//
