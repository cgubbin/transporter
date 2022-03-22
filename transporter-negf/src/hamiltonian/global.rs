// TODO Actually use the workspace, make a global workspace. Assemble the final Hamiltonian

use super::local::{AssembleCellMatrix, ElementConnectivityAssembler};
use nalgebra::{DMatrix, DMatrixSliceMut, RealField};
use nalgebra_sparse::pattern::SparsityPattern;
use nalgebra_sparse::CsrMatrix;
use std::cell::RefCell;

/// An assembler for CSR matrices.
#[derive(Debug, Clone)]
pub struct CsrAssembler<T: RealField> {
    // All members are buffers that help prevent unnecessary allocations
    // when assembling multiple matrices with the same assembler
    workspace: RefCell<CsrAssemblerWorkspace<T>>,
}

impl<T: RealField> Default for CsrAssembler<T> {
    fn default() -> Self {
        Self {
            workspace: RefCell::new(CsrAssemblerWorkspace::default()),
        }
    }
}

#[derive(Debug, Clone)]
struct CsrAssemblerWorkspace<T: RealField> {
    // All members are buffers that help prevent unnecessary allocations
    // when assembling multiple matrices with the same assembler
    connectivity_permutation: Vec<usize>,
    element_global_nodes: Vec<usize>,
    element_matrix: DMatrix<T>,
}

impl<T: RealField> Default for CsrAssemblerWorkspace<T> {
    fn default() -> Self {
        Self {
            connectivity_permutation: Vec::new(),
            element_global_nodes: Vec::new(),
            element_matrix: DMatrix::from_row_slice(0, 0, &[]),
        }
    }
}

impl<T: RealField> CsrAssembler<T> {
    pub fn assemble(
        &self,
        element_assembler: &impl AssembleCellMatrix<T>,
    ) -> color_eyre::Result<CsrMatrix<T>> {
        let num_bands = 1;
        let pattern = self.assemble_pattern(element_assembler, num_bands);
        let initial_matrix_values = vec![T::zero(); pattern.nnz()];
        let mut matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_matrix_values)
            .expect("CSR data must be valid by definition");
        self.assemble_into_csr(&mut matrix, element_assembler, num_bands)?;
        Ok(matrix)
    }

    pub fn assemble_pattern(
        &self,
        element_assembler: &impl ElementConnectivityAssembler,
        num_bands: usize,
    ) -> SparsityPattern {
        let sdim = element_assembler.solution_dim();
        let n_cells = element_assembler.num_cells() + 1;
        let num_rows = sdim * element_assembler.num_nodes() * num_bands;

        let mut element_connections = Vec::new();
        let mut matrix_entries = std::collections::BTreeSet::new();
        for i in 0..element_assembler.num_cells() + 1 {
            let element_connection_count = element_assembler.cell_connection_count(i);
            element_connections.resize(element_connection_count, usize::MAX);
            element_assembler.populate_cell_connections(&mut element_connections, i);
            for n_band in 0..num_bands {
                matrix_entries.insert((n_band * n_cells + i, n_band * n_cells + i)); // The diagonal element
                for j in &element_connections {
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
            .expect("Pattern data must be valid")
    }

    fn assemble_into_csr(
        &self,
        csr: &mut CsrMatrix<T>,
        element_assembler: &impl AssembleCellMatrix<T>,
        num_bands: usize,
    ) -> color_eyre::Result<()> {
        let sdim = element_assembler.solution_dim();
        let num_single_band_rows = sdim * element_assembler.num_nodes() + 1; // We have an issue with cells and nodes, this needs to be pinned down

        for n_row in 0..num_single_band_rows {
            // This is still annoying because we have less connections at the edges so have to
            // re-initialise this matrix on every loop. Can we refactor the mesh to avoid this problem
            let num_connections = element_assembler.cell_connection_count(n_row);
            let mut element_matrix = DMatrix::zeros(num_connections + 1, num_bands);
            let matrix_slice = DMatrixSliceMut::from(&mut element_matrix);
            element_assembler.assemble_cell_matrix_into(n_row, matrix_slice)?;
            for n_band in 0..num_bands {
                let band_row = element_matrix.row(n_band);
                let mut csr_row = csr.row_mut(n_row + n_row * n_band);
                let (cols, values) = csr_row.cols_and_values_mut();
                add_row_to_csr_row(values, cols, &band_row);
            }
        }
        todo!()
    }
}
use nalgebra::{Dynamic, Matrix, Storage, U1};

fn add_row_to_csr_row<T, S>(
    row_values: &mut [T],
    row_col_indices: &[usize],
    local_row: &Matrix<T, U1, Dynamic, S>,
) where
    T: RealField,
    S: Storage<T, U1, Dynamic>,
{
    for (idx, row_value) in row_values.iter_mut().enumerate() {
        *row_value += local_row[idx].clone(); //TODO Avoid this clone...
    }
}

#[cfg(test)]
mod test {
    use super::CsrAssembler;
    use matrixcompare::assert_matrix_eq;
    use nalgebra::DMatrix;
    use transporter_mesher::{create_unit_line_segment_mesh_1d, Mesh1d};
    #[test]
    fn csr_assembly_pattern_is_correct_for_1d_mesh_and_single_band() {
        let cells_per_dim = 100;
        let mesh: Mesh1d<f64> = create_unit_line_segment_mesh_1d(cells_per_dim);

        let assembler: CsrAssembler<f64> = CsrAssembler::default();
        let pattern = assembler.assemble_pattern(&mesh, 1);

        let mut dense = nalgebra::DMatrix::from_element(cells_per_dim + 1, cells_per_dim + 1, 0i8);
        dense[(0, 0)] = 1;
        dense[(0, 1)] = 1;
        for i in 1..cells_per_dim {
            dense[(i, i)] = 1;
            dense[(i, i - 1)] = 1;
            dense[(i, i + 1)] = 1;
        }
        dense[(cells_per_dim, cells_per_dim)] = 1;
        dense[(cells_per_dim, cells_per_dim - 1)] = 1;

        let (offsets, indices) = pattern.disassemble();
        let initial_matrix_values = vec![1i8; indices.len()];
        let csr = nalgebra_sparse::CsrMatrix::try_from_csr_data(
            cells_per_dim + 1,
            cells_per_dim + 1,
            offsets,
            indices,
            initial_matrix_values,
        )
        .unwrap();

        assert_matrix_eq!(csr, dense);
    }

    #[test]
    fn csr_assembly_pattern_is_correct_for_1d_mesh_and_two_bands() {
        let cells_per_dim = 4;
        let nv = cells_per_dim + 1;
        let mesh: Mesh1d<f64> = create_unit_line_segment_mesh_1d(cells_per_dim);
        let num_bands = 2;

        let assembler: CsrAssembler<f64> = CsrAssembler::default();
        let pattern = assembler.assemble_pattern(&mesh, num_bands);

        let mut dense = nalgebra::DMatrix::from_element(
            num_bands * (cells_per_dim + 1),
            num_bands * (cells_per_dim + 1),
            0i8,
        );
        for n_band in 0..num_bands {
            let off = n_band * nv;
            dense[(off, off)] = 1;
            dense[(off, off + 1)] = 1;
            for i in 1..cells_per_dim {
                dense[(off + i, off + i)] = 1;
                dense[(off + i, off + i - 1)] = 1;
                dense[(off + i, off + i + 1)] = 1;
            }
            dense[(off + cells_per_dim, off + cells_per_dim)] = 1;
            dense[(off + cells_per_dim, off + cells_per_dim - 1)] = 1;
        }

        let (offsets, indices) = pattern.disassemble();
        let initial_matrix_values = vec![1i8; indices.len()];
        let csr = nalgebra_sparse::CsrMatrix::try_from_csr_data(
            num_bands * (cells_per_dim + 1),
            num_bands * (cells_per_dim + 1),
            offsets,
            indices,
            initial_matrix_values,
        )
        .unwrap();

        assert_matrix_eq!(csr, dense);
    }

    //::Matrix5x5::new(1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0);

    //let initial_matrix_values = vec![0f64; pattern.nnz()];
    //let mut csr =
    //    nalgebra_sparse::CsrMatrix::try_from_pattern_and_values(pattern, initial_matrix_values)
    //        .expect("CSR data must be valid by definition");

    //let local_row = vec![1f64, 2f64];

    //let local_row = DMatrix::from_row_slice(1, 2, &local_row);
    //let local_row = local_row.row(0);

    //let n_row = 0;
    //let n_band = 1;
    //let mut csr_row = csr.row_mut(n_row + n_row * n_band);
    //let (cols, values) = csr_row.cols_and_values_mut();
    //super::add_row_to_csr_row(values, cols, &local_row);

    //dbg!(csr);
}
