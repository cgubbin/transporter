use transporter_negf::app::run;
fn main() {
    run::<f64>().unwrap();
}
//    let cells_per_dim = 4;
//    let nv = cells_per_dim + 1;
//    let mesh: Mesh1d<f64> = create_unit_line_segment_mesh_1d(cells_per_dim);
//    let num_bands = 2;
//
//    let assembler: CsrAssembler<f64> = CsrAssembler::default();
//    let pattern = assembler.assemble_pattern(&mesh, num_bands);
//
//    let mut dense = nalgebra::DMatrix::from_element(
//        num_bands * (cells_per_dim + 1),
//        num_bands * (cells_per_dim + 1),
//        0i8,
//    );
//    for n_band in 0..num_bands {
//        let off = n_band * nv;
//        dense[(off, off)] = 1;
//        dense[(off, off + 1)] = 1;
//        for i in 1..cells_per_dim {
//            dense[(off + i, off + i)] = 1;
//            dense[(off + i, off + i - 1)] = 1;
//            dense[(off + i, off + i + 1)] = 1;
//        }
//        dense[(off + cells_per_dim, off + cells_per_dim)] = 1;
//        dense[(off + cells_per_dim, off + cells_per_dim - 1)] = 1;
//    }
//
//    let (offsets, indices) = pattern.disassemble();
//    let initial_matrix_values = vec![1i8; indices.len()];
//    let csr = nalgebra_sparse::CsrMatrix::try_from_csr_data(
//        num_bands * (cells_per_dim + 1),
//        num_bands * (cells_per_dim + 1),
//        offsets,
//        indices,
//        initial_matrix_values,
//    )
//    .unwrap();
//
//    dbg!(csr);
//}
//
