use ndarray::ArrayView2;
use num_complex::Complex;

// pub(crate) fn is_hermitian(matrix: ArrayView2<f64>) -> bool {
//     let matrix_transpose = matrix.t();
//     matrix.abs_diff_eq(&matrix_transpose, std::f64::EPSILON)
// }

/// Tests for hermiticity of a matrix
pub(crate) fn is_hermitian(matrix: ArrayView2<Complex<f64>>) -> bool {
    let matrix_transpose = matrix.t();
    matrix
        .iter()
        .zip(matrix_transpose.iter())
        .all(|(element, adjoint_element)| {
            (element - adjoint_element.conj()).norm() < std::f64::EPSILON * 100_f64
        })
}

/// Tests for anti-hermiticity of a matrix
pub(crate) fn is_anti_hermitian(matrix: ArrayView2<Complex<f64>>) -> bool {
    let mut mean = matrix.mean().unwrap_or_else(|| Complex::from(1_f64)).norm();
    if mean == 0_f64 {
        mean = 1_f64;
    }
    let matrix_transpose = matrix.t();
    matrix
        .iter()
        .zip(matrix_transpose.iter())
        .all(|(element, adjoint_element)| {
            (element + adjoint_element.conj()).norm() / mean < std::f64::EPSILON * 10000_f64
        })
}

#[cfg(test)]
mod test {
    use super::{is_anti_hermitian, is_hermitian};
    use ndarray::array;
    use num_complex::Complex;

    #[test]
    fn real_non_hermitian_matrix_returns_false() {
        let matrix = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let matrix = matrix.mapv(Complex::from);
        assert!(!is_hermitian(matrix.view()));
    }

    #[test]
    fn real_hermitian_matrix_returns_true() {
        let matrix = array![[1., 2., 3.], [2., 5., 6.], [3., 6., 9.]];
        let matrix = matrix.mapv(Complex::from);
        assert!(is_hermitian(matrix.view()));
    }

    #[test]
    fn complex_non_hermitian_matrix_returns_false() {
        let matrix = array![
            [
                Complex::new(1., 1.),
                Complex::new(2., 2.),
                Complex::new(3., 3.)
            ],
            [
                Complex::new(4., 4.),
                Complex::new(5., 5.),
                Complex::new(6., 6.)
            ],
            [
                Complex::new(7., 7.),
                Complex::new(8., 8.),
                Complex::new(9., 9.)
            ]
        ];
        assert!(!is_hermitian(matrix.view()));
    }

    #[test]
    fn complex_hermitian_matrices_return_true() {
        let matrix = array![
            [
                Complex::new(1., 0.),
                Complex::new(1., -2.),
                Complex::new(0., 0.)
            ],
            [
                Complex::new(1., 2.),
                Complex::new(0., 0.),
                Complex::new(0., -1.)
            ],
            [
                Complex::new(0., 0.),
                Complex::new(0., 1.),
                Complex::new(1., 0.)
            ]
        ];
        assert!(is_hermitian(matrix.view()));
        let matrix = array![
            [
                Complex::new(1., 0.),
                Complex::new(1., 1.),
                Complex::new(0., 2.)
            ],
            [
                Complex::new(1., -1.),
                Complex::new(5., 0.),
                Complex::new(-3., 0.)
            ],
            [
                Complex::new(0., -2.),
                Complex::new(-3., 0.),
                Complex::new(0., 0.)
            ]
        ];
        assert!(is_hermitian(matrix.view()));
    }

    #[test]
    fn anti_hermitian_matrix_returns_true() {
        let matrix = array![
            [Complex::new(0., -1.), Complex::new(2., 1.)],
            [Complex::new(-2., 1.), Complex::new(0., 0.)]
        ];

        assert!(is_anti_hermitian(matrix.view()));
    }
}
