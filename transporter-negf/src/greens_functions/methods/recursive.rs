//! This module provides recursive methods to calculate elements of the retarded greens function.
//!
//! Rather than fully inverting the Hamiltonian and self-energies to find the greens function, an operation which scales as O(N^3) it is often preferable to
//! simply calculate the elements necessary to calculate the observables in which we are interested. This sub-module
//! provides methods to calculate the following:
//! - A single element of the Green's function diagonal -> This is useful for calculating the current in a device
//! - The full diagonal of the Green's function -> This is useful for calculating the charge density in a device
//! - Any off-diagonal row of the Green's function -> This is useful for bracketing regions in which incoherent modelling is carried out
//!

use super::super::RecursionError;
use nalgebra::{ComplexField, RealField};
use ndarray::{Array1, ArrayViewMut1};
use num_complex::Complex;
use sprs::CsMat;

/// Calculates the left connected diagonal from the Hamiltonian of the system, and the self energy in the left lead.
///
/// This process is initialised using the self-energy in the left lead, and solving
/// g_{00}^{RL} = D_{0}^{-1}
/// where D_{0} is the diagonal component of the inverse Greens function in row 0.
/// We then propagate the left connected Green's function through the device by solving the equation
/// g_{ii}^{RL} = (D_{i} - t_{i, i-1} g_{i-1i-1}^{RL} t_{i-1, i})^{-1}
/// at each layer. Here t_{i, j} are the hopping elements in the Hamiltonian.
pub fn left_connected_diagonal<T>(
    energy: T,
    hamiltonian: &CsMat<T>,
    self_energies: &(Complex<T>, Complex<T>),
    // An optional early termination argument, which is used when we wish to
    // calculate the extended contact GF. To go down the whole matrix pass rows
    terminate_after: usize,
    number_of_vertices_in_reservoir: usize,
) -> Result<Array1<Complex<T>>, RecursionError>
where
    T: RealField + Copy,
    // <T as ComplexField>::RealField: Copy,
{
    let optical_potential = Complex::new(T::zero(), T::zero());
    let mut diagonal = Array1::zeros(terminate_after);
    // at the left contact g_{00}^{LR} is just the inverse of the diagonal matrix element D_{0}
    diagonal[0] = Complex::from(T::one())
        / (Complex::from(energy - hamiltonian.data()[0])
            - self_energies.0
            - if number_of_vertices_in_reservoir != 0 {
                optical_potential
            } else {
                Complex::from(T::zero())
            });
    let mut previous = diagonal[0]; // g_{00}^{LR}
    let mut previous_hopping_element = T::from_real(hamiltonian.data()[1]); // t_{0 1}

    for (idx, (element, row)) in diagonal
        .iter_mut()
        .zip(hamiltonian.outer_iterator())
        .skip(1)
        .enumerate()
    {
        let hopping_element = T::from_real(row.data()[0]); //  t_{i-1, i}
        let diagonal = Complex::from(energy - row.data()[1])
            - if idx == hamiltonian.rows() - 2 {
                self_energies.1
            } else {
                Complex::from(T::zero())
            }
            - if number_of_vertices_in_reservoir != 0 {
                if (idx < number_of_vertices_in_reservoir - 1)
                    | (idx > hamiltonian.rows() - 2 - number_of_vertices_in_reservoir)
                {
                    optical_potential
                } else {
                    Complex::from(T::zero())
                }
            } else {
                Complex::from(T::zero())
            };
        *element = Complex::from(T::one())
            / (diagonal - previous * hopping_element * previous_hopping_element); // g_{ii}^{LR}
        previous_hopping_element = hopping_element;
        previous = *element;
    }
    Ok(diagonal)
}

//#[inline]
///// Calculate the left connected diagonal by solving into the Array1 diagonal, this avoids the allocation
//pub fn left_connected_diagonal_no_alloc<T>(
//    energy: T::RealField,
//    hamiltonian: &CsrMatrix<T::RealField>,
//    self_energies: &(T, T),
//    diagonal: &mut ndarray::Array1<T>,
//    // An optional early termination argument, to go down the whole matrix pass rows
//) -> Result<(), RecursionError>
//where
//    T: ComplexField + Copy,
//    <T as ComplexField>::RealField: Copy,
//{
//    // at the left contact g_{00}^{LR} is just the inverse of the diagonal matrix element D_{0}
//    diagonal[0] =
//        T::one() / (T::from_real(energy - hamiltonian.row(0).values()[0]) - self_energies.0);
//    let mut previous = diagonal[0]; // g_{00}^{LR}
//    let mut previous_hopping_element = T::from_real(hamiltonian.row(0).values()[1]); // t_{0 1}
//
//    for (idx, (element, row)) in diagonal
//        .iter_mut()
//        .zip(hamiltonian.row_iter())
//        .skip(1)
//        .enumerate()
//    {
//        let hopping_element = T::from_real(row.values()[0]); //  t_{i-1, i}
//        let diagonal = T::from_real(energy - row.values()[1])
//            - if idx == hamiltonian.rows() - 2 {
//                self_energies.1
//            } else {
//                T::zero()
//            };
//        *element = T::one() / (diagonal - previous * hopping_element * previous_hopping_element); // g_{ii}^{LR}
//        previous_hopping_element = hopping_element;
//        previous = *element;
//    }
//    Ok(())
//}

/// Calculates the right connected diagonal from the Hamiltonian of the system, and the self energy in the semi-infinite right lead.
///
/// This process is initialised using the self-energy in the right lead, and solving
/// g_{N-1N-1}^{RR} = D_{N-1}^{-1}
/// where D_{N-1} is the diagonal component of the inverse Greens function in row N-1 (the last row).
/// We then propagate the right connected Green's function through the device by solving the equation
/// g_{ii}^{RR} = (D_{i} - t_{i, i+1} g_{i+1i+1}^{RR} t_{i+1, i})^{-1}
/// at each layer. Here t_{i, j} are the hopping elements in the Hamiltonian.
pub fn right_connected_diagonal<T>(
    energy: T,
    hamiltonian: &CsMat<T>,
    self_energies: &(Complex<T>, Complex<T>),
    terminate_after: usize,
    number_of_vertices_in_reservoir: usize,
) -> Result<Array1<Complex<T>>, RecursionError>
where
    T: RealField + Copy,
{
    let optical_potential = Complex::new(T::zero(), T::zero()); // 15mev optical potential
    let nnz = hamiltonian.nnz();
    let mut diagonal: Array1<Complex<T>> = Array1::zeros(terminate_after);
    // g_{N-1N-1}^{RR} = D_{N-1}^{-1}
    diagonal[terminate_after - 1] = Complex::from(T::one())
        / (Complex::from(energy - hamiltonian.data()[nnz - 1])
            - self_energies.1
            - if number_of_vertices_in_reservoir != 0 {
                optical_potential
            } else {
                Complex::from(T::zero())
            });
    let mut previous = diagonal[terminate_after - 1];
    let mut previous_hopping_element = hamiltonian.data()[nnz - 2]; // t_{i, i+1}
    for (idx, (element, row)) in diagonal
        .iter_mut()
        .rev()
        .zip(hamiltonian.outer_iterator().rev())
        .skip(1)
        .enumerate()
    {
        let num_in_row = row.data().len();
        let hopping_element = row.data()[num_in_row - 1];
        let diagonal = Complex::from(energy - row.data()[num_in_row - 2])
            - if idx == hamiltonian.rows() - 2 {
                self_energies.0
            } else {
                Complex::from(T::zero())
            }
            - if number_of_vertices_in_reservoir != 0 {
                if (idx < number_of_vertices_in_reservoir - 1)
                    | (idx > hamiltonian.rows() - 2 - number_of_vertices_in_reservoir)
                {
                    optical_potential
                } else {
                    Complex::from(T::zero())
                }
            } else {
                Complex::from(T::zero())
            };
        *element = Complex::from(T::one())
            / (diagonal - previous * hopping_element * previous_hopping_element);

        previous_hopping_element = hopping_element;
        previous = *element;
    }
    Ok(diagonal)
}

/// Calculate the right connected diagonal by solving into the Array1 diagonal, this avoids the allocation
pub fn right_connected_diagonal_no_alloc<T>(
    energy: T,
    hamiltonian: &CsMat<T>,
    self_energies: &(Complex<T>, Complex<T>),
    diagonal: &mut ArrayViewMut1<Complex<T>>,
) -> Result<(), RecursionError>
where
    T: RealField + Copy,
{
    let rows = hamiltonian.rows();
    let nnz = hamiltonian.nnz();
    // assert_eq!(rows, diagonal.len());
    // g_{N-1N-1}^{RR} = D_{N-1}^{-1}
    diagonal[rows - 1] = Complex::from(T::one())
        / (Complex::from(energy - hamiltonian.data()[nnz - 1]) - self_energies.1);
    let mut previous = diagonal[rows - 1];
    let mut previous_hopping_element = Complex::from(hamiltonian.data()[nnz - 2]); // t_{i, i+1}
    for (idx, (element, row)) in diagonal
        .iter_mut()
        .rev()
        .zip(hamiltonian.outer_iterator().rev())
        .skip(1)
        .enumerate()
    {
        let num_in_row = row.data().len();
        let hopping_element = Complex::from(row.data()[num_in_row - 1]);
        let diagonal = Complex::from(energy - row.data()[num_in_row - 2])
            - if idx == hamiltonian.rows() - 2 {
                self_energies.0
            } else {
                Complex::from(T::zero())
            };
        *element = Complex::from(T::one())
            / (diagonal - previous * hopping_element * previous_hopping_element);

        previous_hopping_element = hopping_element;
        previous = *element;
    }
    Ok(())
}

/// Calculate the full, fully connected diagonal of the retarded Green's function
///
/// The full retarded Greens function diagonal can be found from the left-connected diagonal g_{ii}^{LR}
/// and the self-energy in the semi-infinite right lead. The value in the right element N-1 is exact
/// G_{N-1N-1}^{R} = g_{N-1N-1}^{LR}
/// and we can find the previous elements using the recursion
/// G_{i i}^{R} = g_{i i}^{LR} (1 + t_{i i+1} G_{i+1 i+1}^{R} t_{i+1 i} g_{i i}^{LR})
pub fn diagonal<T>(
    energy: T,
    hamiltonian: &CsMat<T>,
    self_energies: &(Complex<T>, Complex<T>),
    number_of_vertices_in_reservoir: usize,
) -> Result<Array1<Complex<T>>, RecursionError>
where
    T: RealField + Copy,
{
    let rows = hamiltonian.rows();
    let nnz = hamiltonian.nnz();
    let left_diagonal = left_connected_diagonal(
        energy,
        hamiltonian,
        self_energies,
        rows,
        number_of_vertices_in_reservoir,
    )?;

    let mut diagonal = Array1::zeros(rows);
    diagonal[rows - 1] = left_diagonal[rows - 1];

    let mut previous = diagonal[rows - 1];
    let mut previous_hopping_element = T::from_real(hamiltonian.data()[nnz - 2]);
    for ((element, &left_diagonal_element), row) in diagonal
        .iter_mut()
        .zip(left_diagonal.iter())
        .rev()
        .zip(hamiltonian.outer_iterator().rev())
        .skip(1)
    {
        let hopping_element = if row.data().len() == 3 {
            T::from_real(row.data()[0])
        } else {
            T::from_real(row.data()[1])
        };
        *element = left_diagonal_element
            * (Complex::from(T::one())
                + left_diagonal_element * previous * hopping_element * previous_hopping_element);
        previous_hopping_element = hopping_element;
        previous = *element;
    }
    Ok(diagonal)
}

/// Calculate a single element of the fully connected diagonal of the retarded Green's function
///
/// We calculate the left connected and right connected Greens functions in the elements adjacent to the element. Then
/// the value of the fully connected Greens function is
/// G_{ii}^{R} = (D_i - t_{i, i-1} g_{i-1 i-1}^{RL} t_{i-1, i} - t_{i, i+1} g_{i+1i+1}^{RR} t_{i+1, i})^{-1}
pub(crate) fn diagonal_element<T>(
    energy: T,
    hamiltonian: &CsMat<T>,
    self_energies: &(Complex<T>, Complex<T>),
    element_index: usize,
) -> Result<Complex<T>, RecursionError>
where
    T: RealField + Copy,
{
    let rows = hamiltonian.rows();
    if element_index == 0 {
        let right_diagonal =
            right_connected_diagonal(energy, hamiltonian, self_energies, rows - 1, 0)?;
        return Ok(Complex::from(T::one())
            / (Complex::from(energy - hamiltonian.outer_view(0).unwrap().data()[0])
                - self_energies.0
                - right_diagonal[0]
                    * Complex::from(hamiltonian.data()[1] * hamiltonian.data()[2])));
    } else if element_index == rows - 1 {
        let left_diagonal =
            left_connected_diagonal(energy, hamiltonian, self_energies, rows - 1, 0)?;
        let nnz = hamiltonian.nnz();
        return Ok(Complex::from(T::one())
            / (Complex::from(energy - hamiltonian.outer_view(rows - 1).unwrap().data()[1])
                - self_energies.1
                - left_diagonal[rows - 2]
                    * Complex::from(hamiltonian.data()[nnz - 2] * hamiltonian.data()[nnz - 3])));
    } else {
        let _right_diagonal = right_connected_diagonal(
            energy,
            hamiltonian,
            self_energies,
            rows - element_index - 1,
            0,
        )?;
        let _left_diagonal =
            left_connected_diagonal(energy, hamiltonian, self_energies, element_index, 0)?;
        todo!()
        // return Ok(Complex::from(T::one())
        //     / (Complex::from(energy - hamiltonian.row(element_index).values()[1])
        //         - left_diagonal[element_index - 1]
        //             * Complex::from(
        //                 hamiltonian.row(element_index).values()[0]
        //                     * hamiltonian.row(element_index - 1).values()
        //                         [hamiltonian.row(element_index - 1).values().len() - 1],
        //             )
        //         - right_diagonal[0]
        //             * Complex::from(
        //                 hamiltonian.row(element_index).values()[2]
        //                     * hamiltonian.row(element_index + 1).values()[0],
        //             )));
    }
}

/// Builds out a row in the retarded Green's function
///
/// This calculates a row in the retarded Greens function from the left connected diagonal
/// and fully connected diagonal, which can be calcualted using `diagonals`
/// The values of rows and columns are related
/// G_{i j} | i < j  = G_{j i} | i < j = - g_{ii}^{LR} t_{i i+1} G_{ii}^{R}
///
/// The row is build from element `row_index` to the device end for elements in the first half of the stack,
/// for those in the second half it is build from `row_index` to the device start
pub(crate) fn build_out_row<T>(
    energy: T,
    hamiltonian: &CsMat<T>,
    full_diagonal: &Array1<Complex<T>>,
    self_energies: &(Complex<T>, Complex<T>),
    row_index: usize,
    number_of_vertices_in_reservoir: usize,
) -> Result<Array1<Complex<T>>, RecursionError>
where
    T: RealField + Copy,
{
    let rows = hamiltonian.rows();
    if row_index > rows / 2 {
        // Build a row to the left of the diagonal
        let mut row: Array1<Complex<T>> = Array1::zeros(row_index + 1);
        let left_diagonal = left_connected_diagonal(
            energy,
            hamiltonian,
            self_energies,
            row_index + 1,
            number_of_vertices_in_reservoir,
        )?;
        row[row_index] = full_diagonal[row_index];
        let mut previous = row[row_index];
        for (idx, (element, &g_lr)) in row
            .iter_mut()
            .rev()
            .zip(left_diagonal.iter().rev())
            .skip(1)
            .enumerate()
        {
            let row = hamiltonian.outer_view(row_index - idx - 1).unwrap();
            let hopping_element = -T::from_real(row.data()[row.data().len() - 1]);
            *element = -g_lr * hopping_element * previous;
            previous = *element;
        }
        Ok(row)
    } else {
        let mut row: Array1<Complex<T>> = Array1::zeros(rows - row_index);
        // Build a row to the right of the diagonal
        let right_diagonal = right_connected_diagonal(
            energy,
            hamiltonian,
            self_energies,
            rows - row_index,
            number_of_vertices_in_reservoir,
        )?;
        row[0] = full_diagonal[row_index];
        let mut previous = row[0];
        for (idx, (element, &g_rr)) in row
            .iter_mut()
            .zip(right_diagonal.iter())
            .enumerate()
            .skip(1)
        {
            let row = hamiltonian.outer_view(row_index + idx - 1).unwrap();
            let hopping_element = if row_index + idx - 1 == 0 {
                -T::from_real(row.data()[1])
            } else {
                -T::from_real(row.data()[0])
            };
            *element = -g_rr * hopping_element * previous;
            previous = *element;
        }
        Ok(row)
    }
}

pub(crate) fn build_out_column<T>(
    energy: T,
    hamiltonian: &CsMat<T>,
    full_diagonal: &Array1<Complex<T>>,
    self_energies: &(Complex<T>, Complex<T>),
    row_index: usize,
    number_of_vertices_in_reservoir: usize,
) -> Result<Array1<Complex<T>>, RecursionError>
where
    T: RealField + Copy,
{
    build_out_row(
        energy,
        hamiltonian,
        full_diagonal,
        self_energies,
        row_index,
        number_of_vertices_in_reservoir,
    )
}

pub(crate) fn left_column<T>(
    _energy: T::RealField,
    _hamiltonian: &CsMat<T::RealField>,
    _diagonal: &Array1<T>,
    _right_self_energy: T,
) -> Result<Array1<T>, RecursionError>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    todo!()
    //let rows = hamiltonian.rows();
    //let right_connected_diagonal =
    //    right_connected_diagonal(energy, hamiltonian, right_self_energy, rows)?;
    //let mut left_column = DVector::zeros(rows);
    //left_column[0] = diagonal[0];
    //let mut previous = left_column[0];
    //for ((element, row), &right_diagonal_element) in left_column
    //    .iter_mut()
    //    .zip(hamiltonian.row_iter())
    //    .zip(right_connected_diagonal.iter())
    //    .skip(1)
    //    .take(rows - 1)
    //{
    //    let hopping_element = T::from_real(row.values()[2]);

    //    *element = -right_diagonal_element * hopping_element * previous;
    //    previous = *element;
    //}
    //Ok(left_column)
}

pub(crate) fn right_column<T>(
    fully_connected_diagonal: &Array1<T>,
    left_connected_diagonal: &Array1<T>,
    hamiltonian: &CsMat<T::RealField>,
) -> Result<Array1<T>, RecursionError>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let rows = hamiltonian.rows();
    let mut right_column: Array1<T> = Array1::zeros(rows);
    right_column[rows - 1] = fully_connected_diagonal[rows - 1];
    let mut previous = right_column[rows - 1];

    //TODO No double ended iterator available for the CsrMatrix
    for (idx, (element, &left_diagonal_element)) in right_column
        .iter_mut()
        .zip(left_connected_diagonal.iter())
        .rev()
        .skip(1)
        .take(rows - 1)
        .enumerate()
    {
        let row = hamiltonian.outer_view(rows - 2 - idx).unwrap();
        let hopping = T::from_real(row.data()[2]);
        *element = -left_diagonal_element * previous * hopping;
        previous = *element;
    }
    Ok(right_column)
}

#[cfg(test)]
mod test {
    use crate::app::{tracker::TrackerBuilder, Configuration};
    use crate::device::{info_desk::BuildInfoDesk, Device};
    use nalgebra::U1;
    use num_complex::Complex;
    use sprs::CsMat;

    #[test]
    fn recursive_diagonal_coincides_with_dense_inverse() {
        // let path = std::path::PathBuf::try_from("../.config/single.toml").unwrap();
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        let hamiltonian = hamiltonian.calculate_total(0f64);
        let right_self_energy = Complex::new(0.5f64, 0.1f64);
        let left_self_energy = Complex::new(0.25f64, 0.05f64);
        let energy = 0.9;

        let my_diagonal = super::diagonal(
            energy,
            &hamiltonian,
            &(left_self_energy, right_self_energy),
            0,
        )
        .unwrap();

        let complex_values = hamiltonian
            .data()
            .iter()
            .map(Complex::from)
            .collect::<Vec<_>>();
        let mut hamiltonian = CsMat::new(
            hamiltonian.shape(),
            hamiltonian.indptr().raw_storage().to_vec(),
            hamiltonian.indices().to_vec(),
            complex_values,
        );
        let nrows = hamiltonian.shape().0;

        *hamiltonian.get_mut(0, 0).unwrap() += left_self_energy;
        *hamiltonian.get_mut(nrows - 1, nrows - 1).unwrap() += right_self_energy;
        let energy_matrix =
            ndarray::Array2::from_diag_elem(mesh.vertices().len(), Complex::from(energy));

        let dense_matrix = energy_matrix - hamiltonian.to_dense();
        let inverse = ndarray_linalg::Inverse::inv(&dense_matrix).unwrap();

        for (inv_val, my_val) in inverse.diag().iter().zip(my_diagonal.iter()) {
            approx::assert_relative_eq!(inv_val.re, my_val.re, epsilon = 1e-5);
            approx::assert_relative_eq!(inv_val.im, my_val.im, epsilon = 1e-9);
        }
    }

    use crate::app::Calculation;

    // #[test]
    // fn elements_on_the_recursive_diagonal_coincide_with_dense_inverse() {
    //     let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
    //     let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
    //     // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
    //     let info_desk = device.build_device_info_desk().unwrap();

    //     let config: Configuration<f64> = Configuration::build().unwrap();
    //     let mesh: transporter_mesher::Mesh1d<f64> =
    //         crate::app::build_mesh_with_config(&config, device).unwrap();
    //     let tracker = TrackerBuilder::new(Calculation::Coherent{voltage_target: 0_f64}))
    //         .with_mesh(&mesh)
    //         .with_info_desk(&info_desk)
    //         .build()
    //         .unwrap();

    //     let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
    //         .with_mesh(&mesh)
    //         .with_info_desk(&tracker)
    //         .build()
    //         .unwrap();

    //     let hamiltonian_csr = hamiltonian.calculate_total(0f64);
    //     let right_self_energy = Complex::from(0.5f64);
    //     let energy = 0.9;

    //     let complex_values = hamiltonian_csr
    //         .data()
    //         .iter()
    //         .map(Complex::from)
    //         .collect::<Vec<_>>();

    //     let mut hamiltonian_csr = CsMat::new(
    //         hamiltonian_csr.shape(),
    //         hamiltonian_csr.indptr().raw_storage().to_vec(),
    //         hamiltonian_csr.indices().to_vec(),
    //         complex_values,
    //     );
    //     let nrows = hamiltonian_csr.shape().0;

    //     *hamiltonian_csr.get_mut(0, 0).unwrap() += right_self_energy;
    //     *hamiltonian_csr.get_mut(nrows - 1, nrows - 2).unwrap() += right_self_energy;
    //     let energy_matrix =
    //         ndarray::Array2::from_diag_elem(mesh.vertices().len(), Complex::from(energy));

    //     let dense_matrix = energy_matrix - hamiltonian_csr.to_dense();

    //     let inverse = ndarray_linalg::Inverse::inv(&dense_matrix).unwrap();

    //     for (element_index, inv_val) in inverse.diag().iter().enumerate() {
    //         let value = super::diagonal_element(
    //             energy,
    //             &hamiltonian.calculate_total(0_f64),
    //             &(right_self_energy, right_self_energy),
    //             element_index,
    //         )
    //         .unwrap();

    //         println!("{}, {}", inv_val.re, value.re);
    //         println!("{}, {}", inv_val.im, value.im);

    //         // approx::assert_relative_eq!(inv_val.re, value.re, epsilon = 1e-5);
    //         // approx::assert_relative_eq!(inv_val.im, value.im);
    //     }
    // }

    #[test]
    fn rows_constructed_recursively_match_those_from_a_dense_inversion() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        let hamiltonian_csr = hamiltonian.calculate_total(0f64);

        let right_self_energy = Complex::new(0.5f64, 0.1f64);
        let left_self_energy = Complex::new(0.25f64, 0.05f64);
        let energy = 0.9;

        let diagonal = super::diagonal(
            energy,
            &hamiltonian_csr,
            &(left_self_energy, right_self_energy),
            0,
        )
        .unwrap();

        let complex_values = hamiltonian_csr
            .data()
            .iter()
            .map(Complex::from)
            .collect::<Vec<_>>();

        let mut hamiltonian_csr = CsMat::new(
            hamiltonian_csr.shape(),
            hamiltonian_csr.indptr().raw_storage().to_vec(),
            hamiltonian_csr.indices().to_vec(),
            complex_values,
        );
        let nrows = hamiltonian_csr.shape().0;

        *hamiltonian_csr.get_mut(0, 0).unwrap() += left_self_energy;
        *hamiltonian_csr.get_mut(nrows - 1, nrows - 1).unwrap() += right_self_energy;
        let energy_matrix =
            ndarray::Array2::from_diag_elem(mesh.vertices().len(), Complex::from(energy));

        let dense_matrix = energy_matrix - hamiltonian_csr.to_dense();

        let inverse = ndarray_linalg::Inverse::inv(&dense_matrix).unwrap();

        let rows = hamiltonian.num_rows();

        for idx in 0..nrows / 2 {
            //diagonal().iter().enumerate() {
            let recursive_row = super::build_out_row(
                energy,
                &hamiltonian.calculate_total(0_f64),
                &diagonal,
                &(left_self_energy, right_self_energy),
                idx,
                0,
            )
            .unwrap();

            let row = inverse.row(idx);

            for (element, recursive_element) in row.iter().skip(idx).zip(recursive_row.iter()) {
                approx::assert_relative_eq!(element.re, recursive_element.re, epsilon = 1e-5);
                approx::assert_relative_eq!(element.im, recursive_element.im, epsilon = 1e-5);
            }
        }

        for (idx, row) in inverse.outer_iter().enumerate().skip(rows / 2 + 1) {
            //diagonal().iter().enumerate() {
            let recursive_row = super::build_out_row(
                energy,
                &hamiltonian.calculate_total(0_f64),
                &inverse.diag().to_owned(),
                &(left_self_energy, right_self_energy),
                idx,
                0,
            )
            .unwrap();

            for (element, recursive_element) in row.iter().take(idx).zip(recursive_row.iter()) {
                approx::assert_relative_eq!(element.re, recursive_element.re, epsilon = 1e-5);
                approx::assert_relative_eq!(element.im, recursive_element.im, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn columns_constructed_recursively_match_those_from_a_dense_inversion() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        let hamiltonian_csr = hamiltonian.calculate_total(0f64);
        let right_self_energy = Complex::new(0.5f64, 0.1f64);
        let left_self_energy = Complex::new(0.25f64, 0.05f64);
        let energy = 0.9;

        let complex_values = hamiltonian_csr
            .data()
            .iter()
            .map(Complex::from)
            .collect::<Vec<_>>();

        let mut hamiltonian_csr = CsMat::new(
            hamiltonian_csr.shape(),
            hamiltonian_csr.indptr().raw_storage().to_vec(),
            hamiltonian_csr.indices().to_vec(),
            complex_values,
        );
        let nrows = hamiltonian_csr.shape().0;

        *hamiltonian_csr.get_mut(0, 0).unwrap() += left_self_energy;
        *hamiltonian_csr.get_mut(nrows - 1, nrows - 1).unwrap() += right_self_energy;
        let energy_matrix =
            ndarray::Array2::from_diag_elem(mesh.vertices().len(), Complex::from(energy));

        let dense_matrix = energy_matrix - hamiltonian_csr.to_dense();

        let inverse = ndarray_linalg::Inverse::inv(&dense_matrix).unwrap();

        for (idx, column) in inverse
            .axis_iter(ndarray::Axis(1))
            .enumerate()
            .take(nrows / 2)
        {
            //diagonal().iter().enumerate() {
            let recursive_column = super::build_out_column(
                energy,
                &hamiltonian.calculate_total(0_f64),
                &inverse.diag().to_owned(),
                &(left_self_energy, right_self_energy),
                idx,
                0,
            )
            .unwrap();

            for (element, recursive_element) in column.iter().skip(idx).zip(recursive_column.iter())
            {
                approx::assert_relative_eq!(element.re, recursive_element.re, epsilon = 1e-5);
                approx::assert_relative_eq!(element.im, recursive_element.im, epsilon = 1e-5);
            }
        }

        for (idx, column) in inverse
            .axis_iter(ndarray::Axis(1))
            .enumerate()
            .skip(nrows / 2 + 1)
        {
            //diagonal().iter().enumerate() {
            let recursive_column = super::build_out_column(
                energy,
                &hamiltonian.calculate_total(0_f64),
                &inverse.diag().to_owned(),
                &(left_self_energy, right_self_energy),
                idx,
                0,
            )
            .unwrap();

            for (element, recursive_element) in column.iter().take(idx).zip(recursive_column.iter())
            {
                approx::assert_relative_eq!(element.re, recursive_element.re, epsilon = 1e-5);
                approx::assert_relative_eq!(element.im, recursive_element.im, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn left_and_right_connected_are_equal_in_a_symmetric_structure() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::default()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        let hamiltonian = hamiltonian.calculate_total(0f64);
        let self_energy = Complex::from(0.5f64);
        let energy = 0.9;

        let lefts = super::left_connected_diagonal(
            energy,
            &hamiltonian,
            &(self_energy, self_energy),
            mesh.elements().len(),
            0,
        )
        .unwrap();

        let rights = super::right_connected_diagonal(
            energy,
            &hamiltonian,
            &(self_energy, self_energy),
            mesh.elements().len(),
            0,
        )
        .unwrap();

        for (left, right) in lefts.iter().zip(rights.iter().rev()) {
            approx::assert_relative_eq!(
                left.re,
                right.re,
                epsilon = std::f64::EPSILON * 1000000_f64
            ); // Why cant we get to machine precision here? The calculations should be the same
            approx::assert_relative_eq!(left.im, right.im);
        }
    }
}
