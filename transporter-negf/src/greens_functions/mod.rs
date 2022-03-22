mod tridiagonal;

use tridiagonal::*;

use crate::{spectral::SpectralDiscretisation, Hamiltonian};
use nalgebra::{ComplexField, DMatrix};
use nalgebra_sparse::CsrMatrix;
use std::boxed::Box;

pub struct GreensFunctionsBuilder<T, RefSpectralDiscretisation> {
    spectral_discretisation: RefSpectralDiscretisation,
    marker: std::marker::PhantomData<T>,
}

impl GreensFunctionsBuilder<(), ()> {
    fn new() -> Self {
        Self {
            spectral_discretisation: (),
            marker: std::marker::PhantomData,
        }
    }
}

impl<RefSpectralDiscretisation> GreensFunctionsBuilder<(), RefSpectralDiscretisation> {
    fn with_spectral_discretisation<SpectralDiscretisation>(
        self,
        spectral_discretisation: &SpectralDiscretisation,
    ) -> GreensFunctionsBuilder<(), &SpectralDiscretisation> {
        GreensFunctionsBuilder {
            spectral_discretisation,
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T> GreensFunctionsBuilder<T, &'a SpectralDiscretisation<T::RealField>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    fn build<Matrix>(self) -> GreensFunctions<'a, Matrix, T>
    where
        Matrix: GreensFunctionMethods<T>,
    {
        GreensFunctions {
            spectral_discretisation: self.spectral_discretisation,
            retarded: Vec::new(),
            advanced: Vec::new(),
            lesser: Vec::new(),
            greater: Vec::new(),
        }
    }
}

pub struct GreensFunctions<'a, Matrix, T>
where
    Matrix: GreensFunctionMethods<T>,
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    spectral_discretisation: &'a SpectralDiscretisation<T::RealField>,
    retarded: Vec<GreensFunction<Matrix, T>>,
    advanced: Vec<GreensFunction<Matrix, T>>,
    lesser: Vec<GreensFunction<Matrix, T>>,
    greater: Vec<GreensFunction<Matrix, T>>,
}

pub struct GreensFunction<Matrix, T>
where
    Matrix: GreensFunctionMethods<T>,
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    matrix: Matrix,
    marker: std::marker::PhantomData<T>,
}

pub trait GreensFunctionMethods<T>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    type Output;
    type SelfEnergy;
    fn generate_advanced(retarded: Self::Output) -> color_eyre::Result<Box<Self::Output>>;
    //fn generate_greater() -> Self;
    fn generate_lesser(
        retarded: Self::Output,
        advanced: Self::Output,
    ) -> color_eyre::Result<Box<Self::Output>>;
    fn generate_retarded(
        energy: T::RealField,
        hamiltonian: &Hamiltonian<T::RealField>,
        self_energy: &Self::SelfEnergy,
    ) -> color_eyre::Result<Box<Self>>;
}

impl<T> GreensFunctionMethods<T> for CsrMatrix<T>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    type Output = CsrMatrix<T>;
    type SelfEnergy = (T, T);
    fn generate_retarded(
        energy: T::RealField,
        hamiltonian: &Hamiltonian<T::RealField>,
        self_energies: &Self::SelfEnergy,
    ) -> color_eyre::Result<Box<Self>> {
        // Use the fast inversion algorithms to find the inverse, we only need the diagonal in this case
        let num_rows = hamiltonian.num_rows();
        let diagonal = tridiagonal::diagonal(energy, hamiltonian, self_energies)?;
        let top_row = tridiagonal::top_row(energy, hamiltonian, &diagonal, self_energies.1);
        let row_offsets: Vec<usize> = (0..=num_rows).collect(); // One entry per row
        let col_indices: Vec<usize> = (0..num_rows).collect();
        let csr = CsrMatrix::try_from_csr_data(
            num_rows,
            num_rows,
            row_offsets,
            col_indices,
            diagonal.data.as_vec().to_owned(),
        )
        .expect("Failed to initialise CSR matrix");
        Ok(Box::new(csr))
    }

    fn generate_advanced(retarded: CsrMatrix<T>) -> color_eyre::Result<Box<Self>> {
        // TODO Currently we are lifting out into a completely new array, by clone and then conjugating
        // This is less than ideal. When new complex methods are pushed into nalgebra_sparse this can be refactored.
        let values = retarded.values();
        let mut y = Vec::with_capacity(values.len());
        for value in values {
            y.push(value.clone().conjugate());
        }
        let conjugated_self =
            CsrMatrix::try_from_pattern_and_values(retarded.pattern().clone(), y).unwrap();
        Ok(Box::new(conjugated_self.transpose()))
    }

    fn generate_lesser(
        retarded: CsrMatrix<T>,
        lesser: CsrMatrix<T>,
    ) -> color_eyre::Result<Box<Self>> {
        todo!()
    }
}

impl<T> GreensFunctionMethods<T> for DMatrix<T>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    type Output = DMatrix<T>;
    type SelfEnergy = DMatrix<T>;
    fn generate_retarded(
        energy: T::RealField,
        hamiltonian: &Hamiltonian<T::RealField>,
        self_energy: &Self::SelfEnergy,
    ) -> color_eyre::Result<Box<Self>> {
        // do a slow matrix inversion
        let num_rows = hamiltonian.num_rows();

        let x: &CsrMatrix<T::RealField> = hamiltonian.as_ref();
        // TODO Casting to Complex here is verbose and wasteful, can we try not to do this?
        // Maybe the Hamiltonian needs to be made in terms of `ComplexField`?
        let values = x.values();
        let mut y = Vec::with_capacity(values.len());
        for value in values {
            y.push(T::from_real(value.clone()));
        }
        let ham = CsrMatrix::try_from_pattern_and_values(x.pattern().clone(), y).unwrap();
        let matrix = DMatrix::identity(num_rows, num_rows) * T::from_real(energy)
            - nalgebra_sparse::convert::serial::convert_csr_dense(&ham); //TODO Do we have to convert? Seems dumb. Should we store H in dense form too?&ham;
                                                                         //     - nalgebra::Complex::new(T::one(), T::zero())
                                                                         //         * nalgebra_sparse::convert::serial::convert_csr_dense(hamiltonian.as_ref()); //TODO Do we have to convert? Seems dumb. Should we store H in dense form too?
        match matrix.try_inverse() {
            Some(matrix) => Ok(Box::new(matrix)),
            None => Err(color_eyre::eyre::eyre!(
                "Failed to invert for the retarded Green's function",
            )),
        }
    }

    fn generate_advanced(retarded: DMatrix<T>) -> color_eyre::Result<Box<Self>> {
        Ok(Box::new(retarded.conjugate().transpose()))
    }

    fn generate_lesser(retarded: DMatrix<T>, lesser: DMatrix<T>) -> color_eyre::Result<Box<Self>> {
        todo!()
    }
}
