//! Global assemblers for the `Poisson` system
//!
//! This module provides global assemblers for the `Hamiltonian` matrix, taking the elements produced in the
//! local subcrate and throwing them into a global CsrMatrix

use super::local::{AssembleVertexPoissonDiagonal, AssembleVertexPoissonMatrix};
use super::{super::super::CsrError, BuildError, PoissonInfoDesk};
use crate::utilities::assemblers::VertexConnectivityAssembler;
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use nalgebra_sparse::{
    pattern::{SparsityPattern, SparsityPatternFormatError},
    CsrMatrix,
};
use ndarray::{s, Array1};
use sprs::CsMat;
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
    vertex_matrix: Array1<T>,
}

impl<T: RealField> CsrAssemblerWorkspace<T> {
    fn vertex_matrix(&self) -> &Array1<T> {
        &self.vertex_matrix
    }
    fn vertex_matrix_mut(&mut self) -> &mut Array1<T> {
        &mut self.vertex_matrix
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
        Assembler: VertexConnectivityAssembler + PoissonInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let full_sparsity_pattern = CsrAssembler::assemble_full_sparsity_pattern(vertex_assembler)?;
        let diagonal_sparsity_pattern =
            CsrAssembler::assemble_diagonal_sparsity_pattern(vertex_assembler)?;

        Ok(Self {
            workspace: RefCell::new(CsrAssemblerWorkspace {
                diagonal_sparsity_pattern,
                full_sparsity_pattern,
                vertex_matrix: Array1::from_elem(
                    vertex_assembler.vertex_connection_count(1) + 1,
                    T::zero(),
                ),
            }),
        })
    }

    /// Construct the full CsrMatrix sparsity pattern from the element assembler -
    fn assemble_full_sparsity_pattern<Assembler>(
        vertex_assembler: &Assembler,
    ) -> Result<SparsityPattern, SparsityPatternFormatError>
    where
        Assembler: VertexConnectivityAssembler + PoissonInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let sdim = vertex_assembler.solution_dim();
        let num_rows = sdim * vertex_assembler.num_vertices();

        let mut vertex_connections = Vec::new();
        let mut matrix_entries = std::collections::BTreeSet::new();
        for i in 0..vertex_assembler.num_vertices() {
            let vertex_connection_count = vertex_assembler.vertex_connection_count(i);
            vertex_connections.resize(vertex_connection_count, usize::MAX);
            vertex_assembler.populate_vertex_connections(&mut vertex_connections, i);
            matrix_entries.insert((i, i)); // The diagonal element
            for j in vertex_connections.iter() {
                matrix_entries.insert((i, *j));
                // The hopping elements
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
        Assembler: VertexConnectivityAssembler + PoissonInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let sdim = vertex_assembler.solution_dim();
        let num_rows = sdim * vertex_assembler.num_vertices();

        let offsets = (0..num_rows + 1).collect::<Vec<_>>();
        let column_indices = (0..num_rows).collect::<Vec<_>>();

        SparsityPattern::try_from_offsets_and_indices(num_rows, num_rows, offsets, column_indices)
    }
}

/// High level constructors used to initialise the Hamiltonian
impl<T: Copy + RealField> CsrAssembler<T> {
    /// Assembles the fixed component of the Hamiltonian: ie that which is independent of potential and wavevector
    pub(crate) fn assemble_operator<Assembler>(
        &self,
        vertex_assembler: &Assembler,
    ) -> Result<CsMat<T>, BuildError>
    where
        Assembler: AssembleVertexPoissonMatrix<T> + PoissonInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let pattern = self.workspace.borrow().full_sparsity_pattern.clone();
        let initial_matrix_values = vec![T::zero(); pattern.nnz()];
        let matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_matrix_values.clone())
            .map_err(CsrError::from)?;

        let mut matrix = CsMat::new(
            (matrix.nrows(), matrix.ncols()),
            matrix.row_offsets().to_vec(),
            matrix.col_indices().to_vec(),
            initial_matrix_values,
        );

        self.assemble_into_csr(&mut matrix, vertex_assembler)?;
        Ok(matrix)
    }

    /// Assembles the source vector from the static charge densities
    pub(crate) fn assemble_static_source<Assembler>(
        &self,
        vertex_assembler: &Assembler,
    ) -> Result<Array1<T>, BuildError>
    where
        Assembler: AssembleVertexPoissonDiagonal<T> + PoissonInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let pattern = self.workspace.borrow().diagonal_sparsity_pattern.clone();
        let initial_matrix_values = vec![T::zero(); pattern.nnz()];
        let mut diagonal = Array1::from(initial_matrix_values);
        self.assemble_static_charges(&mut diagonal, vertex_assembler)?;
        Ok(diagonal)
    }

    /// Assembles the calculated quantity into a diagonal CsrMatrix:
    pub(crate) fn assemble_diagonal_quantity<Assembler>(
        &self,
        vertex_assembler: &Assembler,
    ) -> Result<CsMat<T>, BuildError>
    where
        Assembler: AssembleVertexPoissonMatrix<T> + PoissonInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let pattern = self.workspace.borrow().diagonal_sparsity_pattern.clone();
        let initial_matrix_values = vec![T::zero(); pattern.nnz()];

        let mut matrix = CsMat::new(
            (
                pattern.major_offsets().len() - 1,
                pattern.major_offsets().len() - 1,
            ),
            pattern.major_offsets().to_vec(),
            pattern.minor_indices().to_vec(),
            initial_matrix_values,
        );
        CsrAssembler::assemble_into_csr_diagonal(&mut matrix, vertex_assembler)?;
        Ok(matrix)
    }

    /// Assemble the diagonal into a pre-initialised CsrMatrix -> This method is called after each iteration to update the changing charge and jacobian diagonal
    pub(crate) fn assemble_diagonal_into<Assembler>(
        element_assembler: &Assembler,
        quantity: &mut CsMat<T>,
    ) -> Result<(), BuildError>
    where
        Assembler: AssembleVertexPoissonMatrix<T> + PoissonInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        CsrAssembler::assemble_into_csr_diagonal(quantity, element_assembler)?;
        Ok(())
    }
}

/// Lower level constructors to compute the local assemblers  for each element in the mesh
impl<T: Copy + RealField> CsrAssembler<T> {
    /// Assembles the fixed component of the Hamiltonian into the CsrMatrix `csr`
    fn assemble_into_csr<Assembler>(
        &self,
        csr: &mut CsMat<T>,
        vertex_assembler: &Assembler,
    ) -> Result<(), BuildError>
    where
        Assembler: AssembleVertexPoissonMatrix<T> + PoissonInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        let sdim = vertex_assembler.solution_dim();
        let num_single_band_rows = sdim * vertex_assembler.num_vertices();

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
            let matrix_slice = vertex_matrix.slice_mut(s![0..num_connections + 1]);
            // let matrix_slice = nalgebra::DVectorSliceMut::from(vertex_matrix);
            vertex_assembler.assemble_vertex_matrix_into(n_row, matrix_slice)?;
            let row = workspace.vertex_matrix();
            let csr_row = csr.outer_view_mut(n_row).unwrap();
            add_row_to_csr_row(csr_row, row);
        }

        Ok(())
    }

    fn assemble_static_charges<Assembler>(
        &self,
        csr: &mut Array1<T>,
        vertex_assembler: &Assembler,
    ) -> Result<(), BuildError>
    where
        Assembler: AssembleVertexPoissonDiagonal<T> + PoissonInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        // Assemble the differential operator for the Hamiltonian
        for (n_row, val) in csr.iter_mut().enumerate() {
            let value = vertex_assembler.assemble_vertex_diagonal(n_row)?;
            *val = value;
        }
        Ok(())
    }

    /// Assemble the potential from the `element_assembler` into the diagonal CsrMatrix `csr`
    /// TODO -> This will panic when the diagonal contains zeros, which it may well do
    fn assemble_into_csr_diagonal<Assembler>(
        csr: &mut CsMat<T>,
        _vertex_assembler: &Assembler,
    ) -> Result<(), BuildError>
    where
        Assembler: AssembleVertexPoissonMatrix<T> + PoissonInfoDesk<T>,
        DefaultAllocator: Allocator<T, Assembler::GeometryDim> + Allocator<T, Assembler::BandDim>,
    {
        // Assemble the potential into a diagonal
        for diagonal_element in csr.diag_iter_mut() {
            let value = T::zero(); // TODO get the value
            if let Some(diagonal_element) = diagonal_element {
                *diagonal_element = value;
            } else {
                return Err(BuildError::Csr(CsrError::Access(
                    "Necessary element not present in CSR Matrix".into(),
                )));
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
// TODO Is this allocation the right way around?
fn add_row_to_csr_row<T>(
    mut local_row: sprs::CsVecBase<&[usize], &mut [T], T>,
    row_values: &Array1<T>,
) where
    T: Copy + RealField,
{
    for (row_value, (_, value)) in row_values.iter().zip(local_row.iter_mut()) {
        *value += *row_value;
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
