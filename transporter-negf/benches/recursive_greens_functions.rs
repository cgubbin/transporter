use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::DVector;
use num_complex::Complex;
use rand::{thread_rng, Rng};
use transporter_negf::greens_functions::recursive::{
    diagonal, left_connected_diagonal, left_connected_diagonal_no_alloc, right_connected_diagonal,
    right_connected_diagonal_no_alloc,
};
use utilities::construct_test_hamiltonian;

pub fn bench_left_connected_diagonal(c: &mut Criterion) {
    let mut rng = thread_rng();
    let self_energies = (
        Complex::new(rng.gen(), rng.gen()),
        Complex::new(rng.gen(), rng.gen()),
    );
    let energy = rng.gen();

    let mut group = c.benchmark_group("left_connected_diagonal");

    for num_rows in [32, 64, 128, 256, 512, 1024, 2048].iter() {
        let hamiltonian = construct_test_hamiltonian(*num_rows);
        // let mut diagonal = DVector::zeros(*num_rows);
        let mut diagonal = ndarray::Array1::zeros(*num_rows);
        group.bench_with_input(
            BenchmarkId::from_parameter(*num_rows),
            num_rows,
            |b, &num_rows| {
                b.iter(|| {
                    left_connected_diagonal_no_alloc(
                        black_box(energy),
                        black_box(&hamiltonian),
                        black_box(&self_energies),
                        black_box(&mut diagonal),
                    )
                })
            },
        );
    }
}

pub fn bench_right_connected_diagonal(c: &mut Criterion) {
    let mut rng = thread_rng();
    let self_energies = (
        Complex::new(rng.gen(), rng.gen()),
        Complex::new(rng.gen(), rng.gen()),
    );
    let energy = rng.gen();

    let mut group = c.benchmark_group("right_connected_diagonal");

    for num_rows in [32, 64, 128, 256, 512, 1024, 2048].iter() {
        let hamiltonian = construct_test_hamiltonian(*num_rows);
        let mut diagonal = DVector::zeros(*num_rows);
        group.bench_with_input(
            BenchmarkId::from_parameter(*num_rows),
            num_rows,
            |b, &num_rows| {
                b.iter(|| {
                    right_connected_diagonal_no_alloc(
                        black_box(energy),
                        black_box(&hamiltonian),
                        black_box(&self_energies),
                        black_box(&mut diagonal),
                    )
                })
            },
        );
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
            |b, &num_rows| {
                b.iter(|| {
                    diagonal(
                        black_box(energy),
                        black_box(&hamiltonian),
                        black_box(&self_energies),
                    )
                })
            },
        );
    }
}

criterion_group!(
    recursive_greens_functions,
    bench_left_connected_diagonal,
    bench_right_connected_diagonal,
    // bench_fully_connected_diagonal
);
criterion_main!(recursive_greens_functions);
