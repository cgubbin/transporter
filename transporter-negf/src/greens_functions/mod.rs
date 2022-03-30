mod tridiagonal;

use crate::postprocessor::{Charge, Current};
use crate::{spectral::SpectralDiscretisation, Hamiltonian};
use nalgebra::{ComplexField, DMatrix};
use nalgebra_sparse::CsrMatrix;
use std::boxed::Box;

pub(crate) struct GreensFunctionBuilder<T, RefSpectral> {
    spectral: RefSpectral,
    marker: std::marker::PhantomData<T>,
}

impl<T> GreensFunctionBuilder<T, ()>
where
    T: ComplexField,
{
    pub(crate) fn new() -> Self {
        Self {
            spectral: (),
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, RefSpectral> GreensFunctionBuilder<T, RefSpectral> {
    pub(crate) fn with_spectral_discretisation<Spectral>(
        self,
        spectral: &Spectral,
    ) -> GreensFunctionBuilder<T, &Spectral> {
        GreensFunctionBuilder {
            spectral,
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T, Spectral> GreensFunctionBuilder<T, &'a Spectral>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    Spectral: SpectralDiscretisation<T::RealField>,
{
    pub(crate) fn build<Matrix>(self) -> AggregateGreensFunctions<T, Matrix>
    where
        Matrix: GreensFunctionMethods<T>,
    {
        AggregateGreensFunctions {
            //    spectral: self.spectral,
            retarded: Vec::with_capacity(self.spectral.total_number_of_points()),
            advanced: Vec::with_capacity(self.spectral.total_number_of_points()),
            lesser: Vec::with_capacity(self.spectral.total_number_of_points()),
            greater: Vec::with_capacity(self.spectral.total_number_of_points()),
        }
    }
}

pub(crate) struct AggregateGreensFunctions<T, Matrix>
where
    Matrix: GreensFunctionMethods<T>,
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    // spectral: &'a dyn SpectralDiscretisation<T::RealField>,
    retarded: Vec<GreensFunction<Matrix, T>>,
    advanced: Vec<GreensFunction<Matrix, T>>,
    lesser: Vec<GreensFunction<Matrix, T>>,
    greater: Vec<GreensFunction<Matrix, T>>,
}

pub(crate) trait AggregateGreensFunctionMethods<T, Integrator>
where
    T: ComplexField,
    Integrator: SpectralDiscretisation<T::RealField>,
{
    fn accumulate_into_charge_density_vector(
        &self,
        integrator: &Integrator,
    ) -> Charge<T::RealField>;
    fn accumulate_into_current_density_vector(
        &self,
        integrator: &Integrator,
    ) -> Current<T::RealField>;
    fn update_greens_functions(&self) -> color_eyre::Result<()>;
}

impl<Integrator, Matrix, T> AggregateGreensFunctionMethods<T, Integrator>
    for AggregateGreensFunctions<T, Matrix>
where
    Matrix: GreensFunctionMethods<T>,
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    Integrator: SpectralDiscretisation<T::RealField>,
{
    fn accumulate_into_charge_density_vector(
        &self,
        _integrator: &Integrator,
    ) -> Charge<T::RealField> {
        todo!()
    }

    fn accumulate_into_current_density_vector(
        &self,
        _integrator: &Integrator,
    ) -> Current<T::RealField> {
        todo!()
    }

    fn update_greens_functions(&self) -> color_eyre::Result<()> {
        todo!()
    }
}

impl<'a, Matrix, T> AggregateGreensFunctions<T, Matrix>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    Matrix: GreensFunctionMethods<T>,
{
    pub(crate) fn update_greens_functions(
        &mut self,
        hamiltonian: &Hamiltonian<T::RealField>,
    ) -> color_eyre::Result<()> {
        self.update_retarded_greens_function(hamiltonian)?;
        todo!()
    }

    fn update_retarded_greens_function(
        &mut self,
        _hamiltonian: &Hamiltonian<T::RealField>,
    ) -> color_eyre::Result<()> {
        //for (retarded, (_, energy)) in self.retarded.iter_mut().zip(self.spectral.iter_all()) {
        //    retarded
        //        .as_mut()
        //        .generate_retarded_into(*energy, hamiltonian, todo!())?;
        //}
        Ok(())
    }
}

pub(crate) struct GreensFunction<Matrix, T>
where
    Matrix: GreensFunctionMethods<T>,
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    matrix: Matrix,
    marker: std::marker::PhantomData<T>,
}

impl<Matrix, T> GreensFunction<Matrix, T>
where
    Matrix: GreensFunctionMethods<T>,
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    fn as_mut(&mut self) -> &mut Matrix {
        &mut self.matrix
    }
}

pub(crate) trait GreensFunctionMethods<T>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    type SelfEnergy;
    fn generate_advanced(retarded: &Self) -> color_eyre::Result<Box<Self>>;
    fn generate_advanced_into(&mut self, retarded: &Self) -> color_eyre::Result<()>;
    fn generate_greater() -> color_eyre::Result<Box<Self>>;
    fn generate_greater_into(&mut self) -> color_eyre::Result<()>;
    fn generate_lesser(retarded: &Self, advanced: &Self) -> color_eyre::Result<Box<Self>>;
    fn generate_lesser_into(&mut self, retarded: &Self, advanced: &Self) -> color_eyre::Result<()>;
    fn generate_retarded(
        energy: T::RealField,
        hamiltonian: &Hamiltonian<T::RealField>,
        self_energy: &Self::SelfEnergy,
    ) -> color_eyre::Result<Box<Self>>;
    fn generate_retarded_into(
        &mut self,
        energy: T::RealField,
        hamiltonian: &Hamiltonian<T::RealField>,
        self_energy: &Self::SelfEnergy,
    ) -> color_eyre::Result<()>;
}

//impl<T> GreensFunctionMethods<T> for CsrMatrix<T>
//where
//    T: ComplexField + Copy,
//    <T as ComplexField>::RealField: Copy,
//{
//    type Output = CsrMatrix<T>;
//    type SelfEnergy = (T, T);
//    fn generate_retarded(
//        energy: T::RealField,
//        hamiltonian: &Hamiltonian<T::RealField>,
//        self_energies: &Self::SelfEnergy,
//    ) -> color_eyre::Result<Box<Self>> {
//        // Use the fast inversion algorithms to find the inverse, we only need the diagonal in this case
//        let num_rows = hamiltonian.num_rows();
//        let diagonal = tridiagonal::diagonal(energy, hamiltonian, self_energies)?;
//        let top_row = tridiagonal::top_row(energy, hamiltonian, &diagonal, self_energies.1);
//        let row_offsets: Vec<usize> = (0..=num_rows).collect(); // One entry per row
//        let col_indices: Vec<usize> = (0..num_rows).collect();
//        let csr = CsrMatrix::try_from_csr_data(
//            num_rows,
//            num_rows,
//            row_offsets,
//            col_indices,
//            diagonal.data.as_vec().to_owned(),
//        )
//        .expect("Failed to initialise CSR matrix");
//        Ok(Box::new(csr))
//    }
//
//    fn generate_advanced(retarded: CsrMatrix<T>) -> color_eyre::Result<Box<Self>> {
//        // TODO Currently we are lifting out into a completely new array, by clone and then conjugating
//        // This is less than ideal. When new complex methods are pushed into nalgebra_sparse this can be refactored.
//        let values = retarded.values();
//        let mut y = Vec::with_capacity(values.len());
//        for value in values {
//            y.push(value.clone().conjugate());
//        }
//        let conjugated_self =
//            CsrMatrix::try_from_pattern_and_values(retarded.pattern().clone(), y).unwrap();
//        Ok(Box::new(conjugated_self.transpose()))
//    }
//
//    fn generate_lesser(
//        retarded: CsrMatrix<T>,
//        lesser: CsrMatrix<T>,
//    ) -> color_eyre::Result<Box<Self>> {
//        todo!()
//    }
//}

impl<T> GreensFunctionMethods<T> for CsrMatrix<T>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    type SelfEnergy = CsrMatrix<T>;

    fn generate_retarded_into(
        &mut self,
        _energy: T::RealField,
        _hamiltonian: &Hamiltonian<T::RealField>,
        _self_energy: &Self::SelfEnergy,
    ) -> color_eyre::Result<()> {
        todo!()
    }

    fn generate_retarded(
        _energy: T::RealField,
        _hamiltonian: &Hamiltonian<T::RealField>,
        _self_energy: &Self::SelfEnergy,
    ) -> color_eyre::Result<Box<Self>> {
        todo!()
    }

    fn generate_greater() -> color_eyre::Result<Box<Self>> {
        todo!()
    }

    fn generate_greater_into(&mut self) -> color_eyre::Result<()> {
        todo!()
    }

    fn generate_advanced(_retarded: &CsrMatrix<T>) -> color_eyre::Result<Box<Self>> {
        todo!()
    }

    fn generate_advanced_into(&mut self, _retarded: &CsrMatrix<T>) -> color_eyre::Result<()> {
        todo!()
    }

    fn generate_lesser(
        _retarded: &CsrMatrix<T>,
        _lesser: &CsrMatrix<T>,
    ) -> color_eyre::Result<Box<Self>> {
        todo!()
    }

    fn generate_lesser_into(
        &mut self,
        _retarded: &CsrMatrix<T>,
        _lesser: &CsrMatrix<T>,
    ) -> color_eyre::Result<()> {
        todo!()
    }
}

impl<T> GreensFunctionMethods<T> for DMatrix<T>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    type SelfEnergy = DMatrix<T>;

    fn generate_retarded_into(
        &mut self,
        energy: T::RealField,
        hamiltonian: &Hamiltonian<T::RealField>,
        _self_energy: &Self::SelfEnergy,
    ) -> color_eyre::Result<()> {
        let mut output: nalgebra::DMatrixSliceMut<T> = self.into();

        // do a slow matrix inversion
        let num_rows = hamiltonian.num_rows();

        let x: &CsrMatrix<T::RealField> = hamiltonian.as_ref();
        // TODO Casting to Complex here is verbose and wasteful, can we try not to do this?
        // Maybe the Hamiltonian needs to be made in terms of `ComplexField`?
        let values = x.values();
        let mut y = Vec::with_capacity(values.len());
        for value in values {
            y.push(T::from_real(*value));
        }
        let ham = CsrMatrix::try_from_pattern_and_values(x.pattern().clone(), y).unwrap();
        // Avoid allocation: https://github.com/InteractiveComputerGraphics/fenris/blob/e4161887669acb366cad312cfa68d106e6cf576c/src/assembly/operators.rs
        // Look at lines 164-172
        let matrix = DMatrix::identity(num_rows, num_rows) * T::from_real(energy)
            - nalgebra_sparse::convert::serial::convert_csr_dense(&ham); //TODO Do we have to convert? Seems dumb. Should we store H in dense form too?&ham;

        if let Some(matrix) = matrix.try_inverse() {
            output.copy_from(&matrix);
        }

        Err(color_eyre::eyre::eyre!(
            "Failed to invert for the retarded Green's function",
        ))
    }

    fn generate_retarded(
        energy: T::RealField,
        hamiltonian: &Hamiltonian<T::RealField>,
        self_energy: &Self::SelfEnergy,
    ) -> color_eyre::Result<Box<Self>> {
        let num_rows = hamiltonian.num_rows();
        let mut matrix = DMatrix::zeros(num_rows, num_rows);
        matrix.generate_retarded_into(energy, hamiltonian, self_energy)?;
        Ok(Box::new(matrix))
    }

    fn generate_greater() -> color_eyre::Result<Box<Self>> {
        todo!()
    }

    fn generate_greater_into(&mut self) -> color_eyre::Result<()> {
        todo!()
    }

    fn generate_advanced(retarded: &DMatrix<T>) -> color_eyre::Result<Box<Self>> {
        let num_rows = retarded.shape().0;
        let mut matrix = DMatrix::zeros(num_rows, num_rows);
        matrix.generate_advanced_into(retarded)?;
        Ok(Box::new(matrix))
    }

    fn generate_advanced_into(&mut self, retarded: &DMatrix<T>) -> color_eyre::Result<()> {
        let mut output: nalgebra::DMatrixSliceMut<T> = self.into();
        output.copy_from(&retarded.conjugate().transpose());
        Ok(())
    }

    fn generate_lesser(
        retarded: &DMatrix<T>,
        lesser: &DMatrix<T>,
    ) -> color_eyre::Result<Box<Self>> {
        let num_rows = retarded.shape().0;
        let mut matrix = DMatrix::zeros(num_rows, num_rows);
        matrix.generate_lesser_into(retarded, lesser)?;
        Ok(Box::new(matrix))
    }

    fn generate_lesser_into(
        &mut self,
        _retarded: &DMatrix<T>,
        _lesser: &DMatrix<T>,
    ) -> color_eyre::Result<()> {
        todo!()
    }
}
