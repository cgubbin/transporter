use crate::Hamiltonian;
use nalgebra::{ComplexField, DVector};

pub(crate) fn diagonal<T>(
    energy: T::RealField,
    hamiltonian: &Hamiltonian<T::RealField>,
    self_energies: &(T, T),
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let num_rows = hamiltonian.num_rows();
    let left_diagonal = left_connected_diagonal(energy, hamiltonian, self_energies.0, num_rows)?;

    let mut diagonal = DVector::zeros(num_rows);
    diagonal[num_rows - 1] = T::one()
        / (T::from_real(energy - hamiltonian.as_ref().row(num_rows - 1).values()[1])
            - self_energies.1
            - T::from_real(hamiltonian.as_ref().row(num_rows - 1).values()[0].powi(2))
                * left_diagonal[num_rows - 1]);

    let mut previous = diagonal[num_rows - 1];
    let mut previous_hopping_element =
        T::from_real(hamiltonian.as_ref().row(num_rows - 1).values()[0]);
    for (idx, (element, &left_diagonal_element)) in diagonal
        .iter_mut()
        .rev()
        .zip(left_diagonal.iter())
        .skip(1)
        .enumerate()
    {
        let row = hamiltonian.as_ref().row(num_rows - 2 - idx);
        let hopping_element = T::from_real(row.values()[2]);
        *element = left_diagonal_element
            * (T::one()
                + left_diagonal_element * previous * hopping_element * previous_hopping_element);
        previous_hopping_element = T::from_real(row.values()[0]);
        previous = *element;
    }
    Ok(diagonal)
}

pub(crate) fn top_row<T>(
    energy: T::RealField,
    hamiltonian: &Hamiltonian<T::RealField>,
    diagonal: &DVector<T>,
    right_self_energy: T,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let num_rows = hamiltonian.num_rows();
    let right_connected_diagonal =
        right_connected_diagonal(energy, hamiltonian, right_self_energy, num_rows)?;
    let mut top_row = DVector::zeros(num_rows);
    top_row[0] = diagonal[0];
    let mut previous = top_row[0];
    for ((element, row), &right_diagonal_element) in top_row
        .iter_mut()
        .zip(hamiltonian.as_ref().row_iter())
        .zip(right_connected_diagonal.iter())
        .skip(1)
    {
        let hopping_element = T::from_real(row.values()[2]);
        *element = -right_diagonal_element * hopping_element * previous;
        previous = *element;
    }
    Ok(top_row)
}

pub(crate) fn bottom_row<T>(
    _energy: T::RealField,
    fully_connected_diagonal: &DVector<T>,
    left_connected_diagonal: &DVector<T>,
    hamiltonian: &Hamiltonian<T::RealField>,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let num_rows = hamiltonian.num_rows();
    let mut bottom_row: DVector<T> = DVector::zeros(num_rows);
    bottom_row[num_rows - 1] = fully_connected_diagonal[num_rows - 1];
    let mut previous = bottom_row[num_rows - 1];

    //TODO No double ended iterator available for the CsrMatrix
    for (idx, (element, &left_diagonal_element)) in bottom_row
        .iter_mut()
        .zip(left_connected_diagonal.iter())
        .rev()
        .skip(1)
        .enumerate()
    {
        let hopping = T::from_real(hamiltonian.as_ref().row(num_rows - 2 - idx).values()[2]);
        *element = -left_diagonal_element * previous * hopping;
        previous = *element;
    }
    Ok(bottom_row)
}

fn left_connected_diagonal<T>(
    energy: T::RealField,
    hamiltonian: &Hamiltonian<T::RealField>,
    left_self_energy: T,
    terminate_after: usize,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let mut diagonal = DVector::zeros(terminate_after);
    diagonal[0] = T::one()
        / (T::from_real(energy - hamiltonian.as_ref().row(0).values()[0]) - left_self_energy);
    let mut previous = diagonal[0];
    let mut previous_hopping_element = T::from_real(hamiltonian.as_ref().row(0).values()[1]);

    for (element, row) in diagonal
        .iter_mut()
        .skip(1)
        .zip(hamiltonian.as_ref().row_iter())
    {
        let hopping_element = T::from_real(row.values()[0]);
        *element = T::one()
            / (T::from_real(energy)
                - T::from_real(row.values()[1])
                - previous * hopping_element * previous_hopping_element);
        previous_hopping_element = T::from_real(row.values()[2]);
        previous = *element;
    }
    Ok(diagonal)
}

fn right_connected_diagonal<T>(
    energy: T::RealField,
    hamiltonian: &Hamiltonian<T::RealField>,
    right_self_energy: T,
    terminate_after: usize,
) -> color_eyre::Result<DVector<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    let num_rows = hamiltonian.num_rows();
    let mut diagonal = DVector::zeros(terminate_after);
    diagonal[terminate_after - 1] = T::one()
        / (T::from_real(energy - hamiltonian.as_ref().row(num_rows - 1).values()[1])
            - right_self_energy);
    let mut previous = diagonal[0];
    let mut previous_hopping_element =
        T::from_real(hamiltonian.as_ref().row(num_rows - 1).values()[0]);
    // TODO CsrMatrix does not implement double ended iter, so we can't zip with the diagonal. When this changes change.
    for (idx, element) in diagonal.iter_mut().rev().skip(1).enumerate() {
        let row = hamiltonian.as_ref().row(num_rows - 2 - idx);
        let hopping_element = T::from_real(row.values()[2]);
        *element = T::one()
            / (T::from_real(energy)
                - T::from_real(row.values()[1])
                - previous * hopping_element * previous_hopping_element);
        previous_hopping_element = T::from_real(row.values()[0]);
        previous = *element;
    }
    Ok(diagonal)
}
