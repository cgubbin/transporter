use nalgebra::{ComplexField, DVector};
use nalgebra_sparse::CsrMatrix;

pub(crate) fn diagonals<T>(
    energy: T::RealField,
    hamiltonian: &CsrMatrix<T::RealField>,
    self_energies: &(T, T),
) -> color_eyre::Result<(DVector<T>, DVector<T>)>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let nrows = hamiltonian.nrows();
    let left_diagonal = left_connected_diagonal(energy, hamiltonian, self_energies.0, nrows)?;

    let mut diagonal = DVector::zeros(nrows);
    diagonal[nrows - 1] = T::one()
        / (T::from_real(energy - hamiltonian.row(nrows - 1).values()[1])
            - self_energies.1
            - T::from_real(hamiltonian.row(nrows - 1).values()[0].powi(2))
                * left_diagonal[nrows - 1]);

    let mut previous = diagonal[nrows - 1];
    let mut previous_hopping_element = T::from_real(hamiltonian.row(nrows - 1).values()[0]);
    for (idx, (element, &left_diagonal_element)) in diagonal
        .iter_mut()
        .rev()
        .zip(left_diagonal.iter())
        .skip(1)
        .take(nrows - 2)
        .enumerate()
    {
        let row = hamiltonian.row(nrows - 2 - idx);
        let hopping_element = T::from_real(row.values()[2]);

        *element = left_diagonal_element
            * (T::one()
                + left_diagonal_element * previous * hopping_element * previous_hopping_element);
        previous_hopping_element = hopping_element;
        previous = *element;
    }
    Ok((diagonal, left_diagonal))
}

pub(crate) fn top_row<T>(
    energy: T::RealField,
    hamiltonian: &CsrMatrix<T::RealField>,
    diagonal: &DVector<T>,
    right_self_energy: T,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let nrows = hamiltonian.nrows();
    let right_connected_diagonal =
        right_connected_diagonal(energy, hamiltonian, right_self_energy, nrows)?;
    let mut top_row = DVector::zeros(nrows);
    top_row[0] = diagonal[0];
    let mut previous = top_row[0];
    for ((element, row), &right_diagonal_element) in top_row
        .iter_mut()
        .zip(hamiltonian.row_iter())
        .zip(right_connected_diagonal.iter())
        .skip(1)
        .take(nrows - 2)
    {
        let hopping_element = T::from_real(row.values()[2]);

        *element = -right_diagonal_element * hopping_element * previous;
        previous = *element;
    }
    Ok(top_row)
}

pub(crate) fn bottom_row<T>(
    fully_connected_diagonal: &DVector<T>,
    left_connected_diagonal: &DVector<T>,
    hamiltonian: &CsrMatrix<T::RealField>,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let nrows = hamiltonian.nrows();
    let mut bottom_row: DVector<T> = DVector::zeros(nrows);
    bottom_row[nrows - 1] = fully_connected_diagonal[nrows - 1];
    let mut previous = bottom_row[nrows - 1];

    //TODO No double ended iterator available for the CsrMatrix
    for (idx, (element, &left_diagonal_element)) in bottom_row
        .iter_mut()
        .zip(left_connected_diagonal.iter())
        .rev()
        .skip(1)
        .take(nrows - 2)
        .enumerate()
    {
        let hopping = T::from_real(hamiltonian.row(nrows - 2 - idx).values()[2]);
        *element = -left_diagonal_element * previous * hopping;
        previous = *element;
    }
    Ok(bottom_row)
}

fn left_connected_diagonal<T>(
    energy: T::RealField,
    hamiltonian: &CsrMatrix<T::RealField>,
    left_self_energy: T,
    terminate_after: usize,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let nrows = hamiltonian.nrows();
    let mut diagonal = DVector::zeros(terminate_after);
    diagonal[0] =
        T::one() / (T::from_real(energy - hamiltonian.row(0).values()[0]) - left_self_energy);
    let mut previous = diagonal[0];
    let mut previous_hopping_element = T::from_real(hamiltonian.row(0).values()[1]);

    for (element, row) in diagonal
        .iter_mut()
        .skip(1)
        .zip(hamiltonian.row_iter())
        .take(nrows - 2)
    {
        let hopping_element = T::from_real(row.values()[0]);
        *element = T::one()
            / (T::from_real(energy)
                - T::from_real(row.values()[1])
                - previous * hopping_element * previous_hopping_element);
        previous_hopping_element = hopping_element;
        previous = *element;
    }
    Ok(diagonal)
}

fn right_connected_diagonal<T>(
    energy: T::RealField,
    hamiltonian: &CsrMatrix<T::RealField>,
    right_self_energy: T,
    terminate_after: usize,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let nrows = hamiltonian.nrows();
    let mut diagonal = DVector::zeros(terminate_after);
    diagonal[terminate_after - 1] = T::one()
        / (T::from_real(energy - hamiltonian.row(nrows - 1).values()[1]) - right_self_energy);
    let mut previous = diagonal[0];
    let mut previous_hopping_element = T::from_real(hamiltonian.row(nrows - 1).values()[0]);
    // TODO CsrMatrix does not implement double ended iter, so we can't zip with the diagonal. When this changes change.
    for (idx, element) in diagonal
        .iter_mut()
        .rev()
        .skip(1)
        .enumerate()
        .take(nrows - 2)
    {
        let row = hamiltonian.row(nrows - 2 - idx);
        let hopping_element = T::from_real(row.values()[2]);
        *element = T::one()
            / (T::from_real(energy)
                - T::from_real(row.values()[1])
                - previous * hopping_element * previous_hopping_element);
        previous_hopping_element = hopping_element;
        previous = *element;
    }
    Ok(diagonal)
}
