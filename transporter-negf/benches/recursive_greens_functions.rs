use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use num_complex::Complex;
use rand::{thread_rng, Rng};
use transporter_negf::greens_functions::methods::recursive::{
    diagonal, right_connected_diagonal, right_connected_diagonal_no_alloc,
};
use utilities::construct_test_hamiltonian;

//pub fn bench_left_connected_diagonal(c: &mut Criterion) {
//    let mut rng = thread_rng();
//    let self_energies = (
//        Complex::new(rng.gen(), rng.gen()),
//        Complex::new(rng.gen(), rng.gen()),
//    );
//    let energy = rng.gen();
//
//    let mut group = c.benchmark_group("left_connected_diagonal");
//
//    for num_rows in [32, 64, 128, 256, 512, 1024, 2048].iter() {
//        let hamiltonian = construct_test_hamiltonian(*num_rows);
//        // let mut diagonal = DVector::zeros(*num_rows);
//        let mut diagonal = ndarray::Array1::zeros(*num_rows);
//        group.bench_with_input(
//            BenchmarkId::from_parameter(*num_rows),
//            num_rows,
//            |b, &num_rows| {
//                b.iter(|| {
//                    left_connected_diagonal_no_alloc(
//                        black_box(energy),
//                        black_box(&hamiltonian),
//                        black_box(&self_energies),
//                        black_box(&mut diagonal),
//                    )
//                })
//            },
//        );
//    }
//}

pub fn bench_right_connected_diagonal(c: &mut Criterion) {
    let mut rng = thread_rng();
    let self_energies = (
        Complex::new(rng.gen(), rng.gen()),
        Complex::new(rng.gen(), rng.gen()),
    );
    let energy = rng.gen();

    let mut group = c.benchmark_group("right_connected_diagonal");

    for num_rows in [32, 64, 128, 256, 512, 1024, 2048].into_iter() {
        let hamiltonian = construct_test_hamiltonian(num_rows);
        // let mut diagonal = Array1::zeros(*num_rows);
        group.bench_with_input(BenchmarkId::new("alloc", num_rows), &num_rows, |b, _| {
            b.iter(|| {
                right_connected_diagonal(
                    black_box(energy),
                    black_box(&hamiltonian),
                    black_box(&self_energies),
                    black_box(num_rows),
                    black_box(0),
                )
            })
        });

        let mut diagonal = Array1::zeros(num_rows);
        let mut view = diagonal.view_mut();
        group.bench_with_input(BenchmarkId::new("no_alloc", num_rows), &num_rows, |b, _| {
            b.iter(|| {
                right_connected_diagonal_no_alloc(
                    black_box(energy),
                    black_box(&hamiltonian),
                    black_box(&self_energies),
                    black_box(&mut view),
                )
            })
        });
    }
}

pub fn bench_fully_connected_diagonal(c: &mut Criterion) {
    let mut rng = thread_rng();
    let self_energies = (
        Complex::new(rng.gen(), rng.gen()),
        Complex::new(rng.gen(), rng.gen()),
    );
    let energy = rng.gen();

    let mut group = c.benchmark_group("fully_connected_diagonal");

    for num_rows in [32, 64, 128, 256, 512, 1024, 2048].iter() {
        let hamiltonian = construct_test_hamiltonian(*num_rows);
        group.bench_with_input(
            BenchmarkId::from_parameter(*num_rows),
            num_rows,
            |b, &_num_rows| {
                b.iter(|| {
                    diagonal(
                        black_box(energy),
                        black_box(&hamiltonian),
                        black_box(&self_energies),
                        black_box(0),
                    )
                })
            },
        );
    }
}

criterion_group!(
    recursive_greens_functions,
    // bench_left_connected_diagonal,
    bench_right_connected_diagonal,
    // bench_fully_connected_diagonal
);
criterion_main!(recursive_greens_functions);
