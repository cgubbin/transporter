//! Dense implementations for single and aggregated Greens functions
use super::GreensFunctionMethods;
use crate::hamiltonian::Hamiltonian;
use nalgebra::{ComplexField, DMatrix, RealField};
use nalgebra_sparse::CsrMatrix;
use num_complex::Complex;

impl<T> GreensFunctionMethods<T> for DMatrix<Complex<T>>
where
    T: RealField + Copy,
{
    type SelfEnergy = DMatrix<Complex<T>>;

    fn generate_retarded_into(
        &mut self,
        energy: T,
        _wavevector: T,
        hamiltonian: &Hamiltonian<T>,
        self_energy: &Self::SelfEnergy,
    ) -> color_eyre::Result<()> {
        let mut output: nalgebra::DMatrixSliceMut<Complex<T>> = self.into();

        // do a slow matrix inversion
        let num_rows = hamiltonian.num_rows();

        let x: &CsrMatrix<T> = hamiltonian.as_ref();
        // TODO Casting to Complex here is verbose and wasteful, can we try not to do this?
        // Maybe the Hamiltonian needs to be made in terms of `ComplexField`?
        let values = x.values();
        let mut y = Vec::with_capacity(values.len());
        for value in values {
            y.push(Complex::from(*value));
        }
        let ham = CsrMatrix::try_from_pattern_and_values(x.pattern().clone(), y).unwrap();
        // Avoid allocation: https://github.com/InteractiveComputerGraphics/fenris/blob/e4161887669acb366cad312cfa68d106e6cf576c/src/assembly/operators.rs
        // Look at lines 164-172
        let mut matrix = DMatrix::identity(num_rows, num_rows) * Complex::from(energy)
            - nalgebra_sparse::convert::serial::convert_csr_dense(&ham)
            - self_energy; //TODO Do we have to convert? Seems dumb. Should we store H in dense form too?&ham;

        if matrix.try_inverse_mut() {
            output.copy_from(&matrix);
            return Ok(());
        }

        Err(color_eyre::eyre::eyre!(
            "Failed to invert for the retarded Green's function",
        ))
    }

    fn generate_greater_into(
        &mut self,
        lesser: &DMatrix<Complex<T>>,
        retarded: &DMatrix<Complex<T>>,
        advanced: &DMatrix<Complex<T>>,
    ) -> color_eyre::Result<()> {
        let mut output: nalgebra::DMatrixSliceMut<Complex<T>> = self.into();
        output.copy_from(&(retarded - advanced + lesser));
        Ok(())
    }

    fn generate_advanced_into(&mut self, retarded: &DMatrix<Complex<T>>) -> color_eyre::Result<()> {
        let mut output: nalgebra::DMatrixSliceMut<Complex<T>> = self.into();
        output.copy_from(&retarded.conjugate().transpose());
        Ok(())
    }

    fn generate_lesser_into(
        &mut self,
        retarded_greens_function: &DMatrix<Complex<T>>,
        retarded_self_energy: &Self::SelfEnergy,
        fermi_functions: &[T],
    ) -> color_eyre::Result<()> {
        let mut advanced_gf = retarded_greens_function.transpose();
        advanced_gf.iter_mut().for_each(|x| *x = x.conjugate());

        let mut gamma_source = retarded_self_energy.clone();
        gamma_source[(advanced_gf.nrows() - 1, advanced_gf.nrows() - 1)] = Complex::from(T::zero());
        gamma_source[(0, 0)] = Complex::new(T::zero(), T::one())
            * (gamma_source[(0, 0)] - gamma_source[(0, 0)].conjugate());
        let mut gamma_drain = retarded_self_energy.clone();
        gamma_drain[(0, 0)] = Complex::from(T::zero());
        gamma_drain[(advanced_gf.nrows() - 1, advanced_gf.nrows() - 1)] =
            Complex::new(T::zero(), T::one())
                * (gamma_source[(advanced_gf.nrows() - 1, advanced_gf.nrows() - 1)]
                    - gamma_source[(advanced_gf.nrows() - 1, advanced_gf.nrows() - 1)].conjugate());
        let sigma_lesser = gamma_source * Complex::new(T::zero(), fermi_functions[0])
            + gamma_drain * Complex::new(T::zero(), fermi_functions[1]);

        self.iter_mut()
            .zip((retarded_greens_function * sigma_lesser * advanced_gf).iter())
            .for_each(|(element, &value)| {
                *element = value;
            });
        Ok(())
    }
}
