pub mod structures;

use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};
use rand::{thread_rng, Rng};

fn generate_diagonal_sparsity_pattern(num_rows: usize) -> SparsityPattern {
    let mut row_offsets = Vec::with_capacity(num_rows + 1);
    let mut column_indices = Vec::with_capacity(num_rows);

    row_offsets.push(0);
    column_indices.push(0);
    column_indices.push(1);
    row_offsets.push(2);
    for i in 1..num_rows - 1 {
        column_indices.push(i - 1);
        column_indices.push(i);
        column_indices.push(i + 1);
        row_offsets.push(row_offsets.last().unwrap() + 3);
    }
    column_indices.push(num_rows - 2);
    column_indices.push(num_rows - 1);
    row_offsets.push(row_offsets.last().unwrap() + 2);

    SparsityPattern::try_from_offsets_and_indices(num_rows, num_rows, row_offsets, column_indices)
        .expect("Pattern data must be valid: {:?}")
}

pub fn construct_test_hamiltonian(num_rows: usize) -> CsrMatrix<f64> {
    let mut rng = thread_rng();
    let sparsity_pattern = generate_diagonal_sparsity_pattern(num_rows);
    let number_of_entries = num_rows * 3 - 2; // Tridiagonal
    let values = (0..number_of_entries)
        .map(|_| rng.gen())
        .collect::<Vec<_>>();
    CsrMatrix::try_from_pattern_and_values(sparsity_pattern, values).expect("Csr data is invalid")
}
