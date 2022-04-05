//! This module provides recursive methods to calculate elements of the retarded greens function.
//!
//! Rather than fully inverting the Hamiltonian and self-energies to find the greens function, an operation which scales as O(N^3) it is often preferable to
//! simply calculate the elements necessary to calculate the observables in which we are interested. This sub-module
//! provides methods to calculate the following:
//! - A single element of the Green's function diagonal -> This is useful for calculating the current in a device
//! - The full diagonal of the Green's function -> This is useful for calculating the charge density in a device
//! - Any off-diagonal row of the Green's function -> This is useful for bracketing regions in which incoherent modelling is carried out
//!
use nalgebra::{ComplexField, DVector};
use nalgebra_sparse::CsrMatrix;

/// Calculates the left connected diagonal from the Hamiltonian of the system, and the self energy in the left lead.
///
/// This process is initialised using the self-energy in the left lead, and solving
/// g_{00}^{RL} = D_{0}^{-1}
/// where D_{0} is the diagonal component of the inverse Greens function in row 0.
/// We then propagate the left connected Green's function through the device by solving the equation
/// g_{ii}^{RL} = (D_{i} - t_{i, i-1} g_{i-1i-1}^{RL} t_{i-1, i})^{-1}
/// at each layer. Here t_{i, j} are the hopping elements in the Hamiltonian.
fn left_connected_diagonal<T>(
    energy: T::RealField,
    hamiltonian: &CsrMatrix<T::RealField>,
    self_energies: &(T, T),
    // An optional early termination argument, to go down the whole matrix pass nrows
    terminate_after: usize,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let mut diagonal = DVector::zeros(terminate_after);
    // at the left contact g_{00}^{LR} is just the inverse of the diagonal matrix element D_{0}
    diagonal[0] =
        T::one() / (T::from_real(energy - hamiltonian.row(0).values()[0]) - self_energies.0);
    let mut previous = diagonal[0]; // g_{00}^{LR}
    let mut previous_hopping_element = T::from_real(hamiltonian.row(0).values()[1]); // t_{0 1}

    for (idx, (element, row)) in diagonal
        .iter_mut()
        .zip(hamiltonian.row_iter())
        .skip(1)
        .enumerate()
    {
        let hopping_element = T::from_real(row.values()[0]); //  t_{i-1, i}
        let diagonal = T::from_real(energy - row.values()[1])
            - if idx == hamiltonian.nrows() - 2 {
                self_energies.1
            } else {
                T::zero()
            };
        *element = T::one() / (diagonal - previous * hopping_element * previous_hopping_element); // g_{ii}^{LR}
        previous_hopping_element = hopping_element;
        previous = *element;
    }
    Ok(diagonal)
}

/// Calculates the right connected diagonal from the Hamiltonian of the system, and the self energy in the semi-infinite right lead.
///
/// This process is initialised using the self-energy in the right lead, and solving
/// g_{N-1N-1}^{RR} = D_{N-1}^{-1}
/// where D_{N-1} is the diagonal component of the inverse Greens function in row N-1 (the last row).
/// We then propagate the right connected Green's function through the device by solving the equation
/// g_{ii}^{RR} = (D_{i} - t_{i, i+1} g_{i+1i+1}^{RR} t_{i+1, i})^{-1}
/// at each layer. Here t_{i, j} are the hopping elements in the Hamiltonian.
fn right_connected_diagonal<T>(
    energy: T::RealField,
    hamiltonian: &CsrMatrix<T::RealField>,
    self_energies: &(T, T),
    terminate_after: usize,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let nrows = hamiltonian.nrows();
    let mut diagonal = DVector::zeros(terminate_after);
    // g_{N-1N-1}^{RR} = D_{N-1}^{-1}
    diagonal[terminate_after - 1] = T::one()
        / (T::from_real(energy - hamiltonian.row(nrows - 1).values()[1]) - self_energies.1);
    let mut previous = diagonal[terminate_after - 1];
    let mut previous_hopping_element = T::from_real(hamiltonian.row(nrows - 1).values()[0]); // t_{i, i+1}
    for (idx, element) in diagonal.iter_mut().rev().skip(1).enumerate() {
        let row = hamiltonian.row(nrows - 2 - idx);
        let hopping_element = T::from_real(row.values()[row.values().len() - 1]);
        let diagonal = T::from_real(energy - row.values()[row.values().len() - 2])
            - if idx == hamiltonian.nrows() - 2 {
                self_energies.1
            } else {
                T::zero()
            };
        *element = T::one() / (diagonal - previous * hopping_element * previous_hopping_element);

        previous_hopping_element = hopping_element;
        previous = *element;
    }
    Ok(diagonal)
}

/// Calculate the full, fully connected diagonal of the retarded Green's function
///
/// The full retarded Greens function diagonal can be found from the left-connected diagonal g_{ii}^{LR}
/// and the self-energy in the semi-infinite right lead. The value in the right element N-1 is exact
/// G_{N-1N-1}^{R} = g_{N-1N-1}^{LR}
/// and we can find the previous elements using the recursion
/// G_{i i}^{R} = g_{i i}^{LR} (1 + t_{i i+1} G_{i+1 i+1}^{R} t_{i+1 i} g_{i i}^{LR})
pub(crate) fn diagonal<T>(
    energy: T::RealField,
    hamiltonian: &CsrMatrix<T::RealField>,
    self_energies: &(T, T),
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let nrows = hamiltonian.nrows();
    let left_diagonal = left_connected_diagonal(energy, hamiltonian, self_energies, nrows)?;

    let mut diagonal = DVector::zeros(nrows);
    diagonal[nrows - 1] = left_diagonal[nrows - 1];

    let mut previous = diagonal[nrows - 1];
    let mut previous_hopping_element = T::from_real(hamiltonian.row(nrows - 1).values()[0]);
    for (idx, (element, &left_diagonal_element)) in diagonal
        .iter_mut()
        .zip(left_diagonal.iter())
        .rev()
        .skip(1)
        .enumerate()
    {
        let row = hamiltonian.row(nrows - 2 - idx);
        let hopping_element = if row.values().len() == 3 {
            T::from_real(row.values()[0])
        } else {
            T::from_real(row.values()[1])
        };
        *element = left_diagonal_element
            * (T::one()
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
    energy: T::RealField,
    hamiltonian: &CsrMatrix<T::RealField>,
    self_energies: &(T, T),
    element_index: usize,
) -> color_eyre::Result<T>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let nrows = hamiltonian.nrows();
    if element_index == 0 {
        let right_diagonal =
            right_connected_diagonal(energy, hamiltonian, self_energies, nrows - 1)?;
        return Ok(T::one()
            / (T::from_real(energy - hamiltonian.row(0).values()[0])
                - self_energies.0
                - right_diagonal[0]
                    * T::from_real(
                        hamiltonian.row(0).values()[1] * hamiltonian.row(1).values()[0],
                    )));
    } else if element_index == nrows - 1 {
        let left_diagonal = left_connected_diagonal(energy, hamiltonian, self_energies, nrows - 1)?;
        return Ok(T::one()
            / (T::from_real(energy - hamiltonian.row(nrows - 1).values()[1])
                - self_energies.1
                - left_diagonal[nrows - 2]
                    * T::from_real(
                        hamiltonian.row(nrows - 1).values()[0]
                            * hamiltonian.row(nrows - 2).values()[2],
                    )));
    } else {
        let right_diagonal = right_connected_diagonal(
            energy,
            hamiltonian,
            self_energies,
            nrows - element_index - 1,
        )?;
        let left_diagonal =
            left_connected_diagonal(energy, hamiltonian, self_energies, element_index)?;
        return Ok(T::one()
            / (T::from_real(energy - hamiltonian.row(element_index).values()[1])
                - left_diagonal[element_index - 1]
                    * T::from_real(
                        hamiltonian.row(element_index).values()[0]
                            * hamiltonian.row(element_index - 1).values()
                                [hamiltonian.row(element_index - 1).values().len() - 1],
                    )
                - right_diagonal[0]
                    * T::from_real(
                        hamiltonian.row(element_index).values()[2]
                            * hamiltonian.row(element_index + 1).values()[0],
                    )));
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
    energy: T::RealField,
    hamiltonian: &CsrMatrix<T::RealField>,
    full_diagonal: &DVector<T>,
    self_energies: &(T, T),
    row_index: usize,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let nrows = hamiltonian.nrows();
    if row_index > nrows / 2 {
        // Build a row to the left of the diagonal
        let mut row: DVector<T> = DVector::zeros(row_index + 1);
        let left_diagonal =
            left_connected_diagonal(energy, hamiltonian, self_energies, row_index + 1)?;
        row[row_index] = full_diagonal[row_index];
        let mut previous = row[row_index];
        for (idx, (element, &g_lr)) in row
            .iter_mut()
            .rev()
            .zip(left_diagonal.iter().rev())
            .skip(1)
            .enumerate()
        {
            let hopping_element = -T::from_real(
                hamiltonian.row(row_index - idx - 1).values()
                    [hamiltonian.row(row_index - idx - 1).values().len() - 1],
            );
            *element = -g_lr * hopping_element * previous;
            previous = *element;
        }
        Ok(row)
    } else {
        let mut row: DVector<T> = DVector::zeros(nrows - row_index);
        // Build a row to the right of the diagonal
        let right_diagonal =
            right_connected_diagonal(energy, hamiltonian, self_energies, nrows - row_index)?;
        row[0] = full_diagonal[row_index];
        let mut previous = row[0];
        for (idx, (element, &g_rr)) in row
            .iter_mut()
            .zip(right_diagonal.iter())
            .skip(1)
            .enumerate()
        {
            let hopping_element = -T::from_real(hamiltonian.row(row_index + idx + 1).values()[0]);
            *element = -g_rr * hopping_element * previous;
            previous = *element;
        }
        Ok(row)
    }
}

pub(crate) fn build_out_column<T>(
    energy: T::RealField,
    hamiltonian: &CsrMatrix<T::RealField>,
    full_diagonal: &DVector<T>,
    self_energies: &(T, T),
    row_index: usize,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    build_out_row(energy, hamiltonian, full_diagonal, self_energies, row_index)
}

pub(crate) fn left_column<T>(
    _energy: T::RealField,
    _hamiltonian: &CsrMatrix<T::RealField>,
    _diagonal: &DVector<T>,
    _right_self_energy: T,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    todo!()
    //let nrows = hamiltonian.nrows();
    //let right_connected_diagonal =
    //    right_connected_diagonal(energy, hamiltonian, right_self_energy, nrows)?;
    //let mut left_column = DVector::zeros(nrows);
    //left_column[0] = diagonal[0];
    //let mut previous = left_column[0];
    //for ((element, row), &right_diagonal_element) in left_column
    //    .iter_mut()
    //    .zip(hamiltonian.row_iter())
    //    .zip(right_connected_diagonal.iter())
    //    .skip(1)
    //    .take(nrows - 1)
    //{
    //    let hopping_element = T::from_real(row.values()[2]);

    //    *element = -right_diagonal_element * hopping_element * previous;
    //    previous = *element;
    //}
    //Ok(left_column)
}

pub(crate) fn right_column<T>(
    fully_connected_diagonal: &DVector<T>,
    left_connected_diagonal: &DVector<T>,
    hamiltonian: &CsrMatrix<T::RealField>,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let nrows = hamiltonian.nrows();
    let mut right_column: DVector<T> = DVector::zeros(nrows);
    right_column[nrows - 1] = fully_connected_diagonal[nrows - 1];
    let mut previous = right_column[nrows - 1];

    //TODO No double ended iterator available for the CsrMatrix
    for (idx, (element, &left_diagonal_element)) in right_column
        .iter_mut()
        .zip(left_connected_diagonal.iter())
        .rev()
        .skip(1)
        .take(nrows - 1)
        .enumerate()
    {
        let hopping = T::from_real(hamiltonian.row(nrows - 2 - idx).values()[2]);
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
    use nalgebra_sparse::CsrMatrix;
    use num_complex::Complex;

    #[test]
    fn recursive_diagonal_coincides_with_dense_inverse() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        let hamiltonian = hamiltonian.calculate_total(0f64);
        let right_self_energy = Complex::from(0.5f64);
        let energy = 0.9;

        let my_diagonal = super::diagonal(
            energy,
            &hamiltonian,
            &(right_self_energy, right_self_energy),
        )
        .unwrap();

        let complex_values = hamiltonian
            .values()
            .iter()
            .map(Complex::from)
            .collect::<Vec<_>>();
        let mut hamiltonian =
            CsrMatrix::try_from_pattern_and_values(hamiltonian.pattern().clone(), complex_values)
                .unwrap();
        let n_vals = hamiltonian.values().len();

        hamiltonian.values_mut()[0] += right_self_energy;
        hamiltonian.values_mut()[n_vals - 1] += right_self_energy;
        let energy_matrix =
            nalgebra::DMatrix::identity(mesh.elements().len(), mesh.elements().len())
                * Complex::from(energy);

        let dense_matrix =
            energy_matrix - nalgebra_sparse::convert::serial::convert_csr_dense(&hamiltonian);

        let inverse = dense_matrix.try_inverse().unwrap();

        for (inv_val, my_val) in inverse.diagonal().iter().zip(my_diagonal.iter()) {
            approx::assert_relative_eq!(inv_val.re, my_val.re, epsilon = 1e-5);
            approx::assert_relative_eq!(inv_val.im, my_val.im);
        }
    }

    #[test]
    fn elements_on_the_recursive_diagonal_coincide_with_dense_inverse() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        let hamiltonian_csr = hamiltonian.calculate_total(0f64);
        let right_self_energy = Complex::from(0.5f64);
        let energy = 0.9;

        let complex_values = hamiltonian_csr
            .values()
            .iter()
            .map(Complex::from)
            .collect::<Vec<_>>();
        let mut hamiltonian_cplx = CsrMatrix::try_from_pattern_and_values(
            hamiltonian_csr.pattern().clone(),
            complex_values,
        )
        .unwrap();
        let n_vals = hamiltonian_cplx.values().len();

        hamiltonian_cplx.values_mut()[0] += right_self_energy;
        hamiltonian_cplx.values_mut()[n_vals - 1] += right_self_energy;
        let energy_matrix =
            nalgebra::DMatrix::identity(mesh.elements().len(), mesh.elements().len())
                * Complex::from(energy);

        let dense_matrix =
            energy_matrix - nalgebra_sparse::convert::serial::convert_csr_dense(&hamiltonian_cplx);

        let inverse = dense_matrix.try_inverse().unwrap();

        for (element_index, inv_val) in inverse.diagonal().iter().enumerate() {
            let value = super::diagonal_element(
                energy,
                &hamiltonian.calculate_total(0_f64),
                &(right_self_energy, right_self_energy),
                element_index,
            )
            .unwrap();

            approx::assert_relative_eq!(inv_val.re, value.re, epsilon = 1e-5);
            approx::assert_relative_eq!(inv_val.im, value.im);
        }
    }

    #[test]
    fn rows_constructed_recursively_match_those_from_a_dense_inversion() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        let hamiltonian_csr = hamiltonian.calculate_total(0f64);
        let right_self_energy = Complex::from(0.5f64);
        let energy = 0.9;

        let complex_values = hamiltonian_csr
            .values()
            .iter()
            .map(Complex::from)
            .collect::<Vec<_>>();
        let mut hamiltonian_cplx = CsrMatrix::try_from_pattern_and_values(
            hamiltonian_csr.pattern().clone(),
            complex_values,
        )
        .unwrap();
        let n_vals = hamiltonian_cplx.values().len();

        hamiltonian_cplx.values_mut()[0] += right_self_energy;
        hamiltonian_cplx.values_mut()[n_vals - 1] += right_self_energy;
        let energy_matrix =
            nalgebra::DMatrix::identity(mesh.elements().len(), mesh.elements().len())
                * Complex::from(energy);

        let dense_matrix =
            energy_matrix - nalgebra_sparse::convert::serial::convert_csr_dense(&hamiltonian_cplx);

        let inverse = dense_matrix.try_inverse().unwrap();

        let _fully_connected_diagonal = super::diagonal(
            energy,
            &hamiltonian.calculate_total(0.),
            &(right_self_energy, right_self_energy),
        )
        .unwrap();

        let nrows = hamiltonian.num_rows();
        for (idx, row) in inverse.row_iter().enumerate().take(nrows / 2) {
            //diagonal().iter().enumerate() {
            let recursive_row = super::build_out_row(
                energy,
                &hamiltonian.calculate_total(0_f64),
                &inverse.diagonal(),
                &(right_self_energy, right_self_energy),
                idx,
            )
            .unwrap();

            for (element, recursive_element) in row.iter().skip(idx).zip(recursive_row.iter()) {
                approx::assert_relative_eq!(element.re, recursive_element.re, epsilon = 1e-5);
                approx::assert_relative_eq!(element.im, recursive_element.im);
            }
        }

        for (idx, row) in inverse.row_iter().enumerate().skip(nrows / 2 + 1) {
            //diagonal().iter().enumerate() {
            let recursive_row = super::build_out_row(
                energy,
                &hamiltonian.calculate_total(0_f64),
                &inverse.diagonal(),
                &(right_self_energy, right_self_energy),
                idx,
            )
            .unwrap();

            for (element, recursive_element) in row.iter().take(idx).zip(recursive_row.iter()) {
                approx::assert_relative_eq!(element.re, recursive_element.re, epsilon = 1e-5);
                approx::assert_relative_eq!(element.im, recursive_element.im);
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
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&tracker)
            .build()
            .unwrap();

        let hamiltonian_csr = hamiltonian.calculate_total(0f64);
        let right_self_energy = Complex::from(0.5f64);
        let energy = 0.9;

        let complex_values = hamiltonian_csr
            .values()
            .iter()
            .map(Complex::from)
            .collect::<Vec<_>>();
        let mut hamiltonian_cplx = CsrMatrix::try_from_pattern_and_values(
            hamiltonian_csr.pattern().clone(),
            complex_values,
        )
        .unwrap();
        let n_vals = hamiltonian_cplx.values().len();

        hamiltonian_cplx.values_mut()[0] += right_self_energy;
        hamiltonian_cplx.values_mut()[n_vals - 1] += right_self_energy;
        let energy_matrix =
            nalgebra::DMatrix::identity(mesh.elements().len(), mesh.elements().len())
                * Complex::from(energy);

        let dense_matrix =
            energy_matrix - nalgebra_sparse::convert::serial::convert_csr_dense(&hamiltonian_cplx);

        let inverse = dense_matrix.try_inverse().unwrap();

        let _fully_connected_diagonal = super::diagonal(
            energy,
            &hamiltonian.calculate_total(0.),
            &(right_self_energy, right_self_energy),
        )
        .unwrap();

        let nrows = hamiltonian.num_rows();
        for (idx, column) in inverse.column_iter().enumerate().take(nrows / 2) {
            //diagonal().iter().enumerate() {
            let recursive_column = super::build_out_column(
                energy,
                &hamiltonian.calculate_total(0_f64),
                &inverse.diagonal(),
                &(right_self_energy, right_self_energy),
                idx,
            )
            .unwrap();

            for (element, recursive_element) in column.iter().skip(idx).zip(recursive_column.iter())
            {
                approx::assert_relative_eq!(element.re, recursive_element.re, epsilon = 1e-5);
                approx::assert_relative_eq!(element.im, recursive_element.im);
            }
        }

        for (idx, column) in inverse.column_iter().enumerate().skip(nrows / 2 + 1) {
            //diagonal().iter().enumerate() {
            let recursive_column = super::build_out_column(
                energy,
                &hamiltonian.calculate_total(0_f64),
                &inverse.diagonal(),
                &(right_self_energy, right_self_energy),
                idx,
            )
            .unwrap();

            for (element, recursive_element) in column.iter().take(idx).zip(recursive_column.iter())
            {
                approx::assert_relative_eq!(element.re, recursive_element.re, epsilon = 1e-5);
                approx::assert_relative_eq!(element.im, recursive_element.im);
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
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
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
        )
        .unwrap();

        let rights = super::right_connected_diagonal(
            energy,
            &hamiltonian,
            &(self_energy, self_energy),
            mesh.elements().len(),
        )
        .unwrap();

        dbg!(&lefts);
        dbg!(&rights);

        for (left, right) in lefts.iter().zip(rights.iter().rev()) {
            approx::assert_relative_eq!(left.re, right.re, epsilon = std::f64::EPSILON * 10000_f64); // Why cant we get to machine precision here? The calculations should be the same
            approx::assert_relative_eq!(left.im, right.im);
        }
    }
}
