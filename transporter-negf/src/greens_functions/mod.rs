mod tridiagonal;

use tridiagonal::{bottom_row, diagonals, top_row};

use crate::postprocessor::{Charge, Current};
use crate::self_energy::SelfEnergy;
use crate::{
    spectral::{SpectralDiscretisation, SpectralSpace},
    Hamiltonian,
};
use nalgebra::{
    allocator::Allocator, ComplexField, Const, DMatrix, DVector, DefaultAllocator, Dynamic, Matrix,
    OVector, SimdComplexField, VecStorage,
};
use nalgebra_sparse::CsrMatrix;
use transporter_mesher::{Connectivity, SmallDim};

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

pub(crate) trait AggregateGreensFunctionMethods<T, BandDim, Integrator>
where
    T: ComplexField,
    BandDim: SmallDim,
    Integrator: SpectralDiscretisation<T::RealField>,
    DefaultAllocator: Allocator<
        Matrix<
            T::RealField,
            Dynamic,
            Const<1_usize>,
            VecStorage<T::RealField, Dynamic, Const<1_usize>>,
        >,
        BandDim,
    >,
{
    fn accumulate_into_charge_density_vector(
        &self,
        integrator: &Integrator,
    ) -> color_eyre::Result<Charge<T::RealField, BandDim>>;
    fn accumulate_into_current_density_vector(
        &self,
        integrator: &Integrator,
    ) -> color_eyre::Result<Current<T::RealField, BandDim>>;
}

impl<T, BandDim> AggregateGreensFunctionMethods<T, BandDim, SpectralSpace<T::RealField, ()>>
    for AggregateGreensFunctions<T, CsrMatrix<T>>
where
    T: ComplexField + Copy,
    BandDim: SmallDim,
    <T as ComplexField>::RealField: Copy,
    DefaultAllocator: Allocator<
        Matrix<
            T::RealField,
            Dynamic,
            Const<1_usize>,
            VecStorage<T::RealField, Dynamic, Const<1_usize>>,
        >,
        BandDim,
    >,
{
    /// Only for one band -> need to skip the iter
    fn accumulate_into_charge_density_vector(
        &self,
        spectral_space: &SpectralSpace<T::RealField, ()>,
    ) -> color_eyre::Result<Charge<T::RealField, BandDim>> {
        let mut charges: Vec<DVector<T::RealField>> = Vec::with_capacity(BandDim::dim());
        let summed_diagonal = self
            .retarded
            .iter()
            .zip(spectral_space.energy.weights())
            .fold(
                &self.retarded[0].matrix * T::zero(),
                |sum, (value, &weight)| sum + &value.matrix * T::from_real(weight),
            )
            .diagonal_as_csr();

        for band_number in 0..BandDim::dim() {
            charges.push(DVector::from(
                summed_diagonal
                    .values()
                    .iter()
                    .skip(band_number)
                    .step_by(BandDim::dim())
                    .map(|&x| x.real())
                    .collect::<Vec<_>>(),
            ));
        }

        dbg!(&charges);

        Charge::new(OVector::<DVector<T::RealField>, BandDim>::from_iterator(
            charges.into_iter(),
        ))
    }

    fn accumulate_into_current_density_vector(
        &self,
        _spectral_space: &SpectralSpace<T::RealField, ()>,
    ) -> color_eyre::Result<Current<T::RealField, BandDim>> {
        todo!()
    }
}

// TODO This is a single band implementation
impl<'a, T> AggregateGreensFunctions<T, CsrMatrix<T>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
{
    pub(crate) fn update_greens_functions<GeometryDim: SmallDim, Conn>(
        &mut self,
        hamiltonian: &Hamiltonian<T::RealField>,
        self_energy: &SelfEnergy<T, GeometryDim, Conn, CsrMatrix<T>>,
        spectral_space: &SpectralSpace<T::RealField, ()>,
    ) -> color_eyre::Result<()>
    where
        Conn: Connectivity<T::RealField, GeometryDim>,
        DefaultAllocator: Allocator<T::RealField, GeometryDim>,
    {
        // In the coherent transport case we only need the retarded and lesser Greens functions (see Lake 1997)
        self.update_aggregate_retarded_greens_function(hamiltonian, self_energy, spectral_space)?;
        self.update_aggregate_lesser_greens_function(self_energy, spectral_space)?;
        Ok(())
    }

    fn update_aggregate_retarded_greens_function<GeometryDim: SmallDim, Conn>(
        &mut self,
        hamiltonian: &Hamiltonian<T::RealField>,
        self_energy: &SelfEnergy<T, GeometryDim, Conn, CsrMatrix<T>>,
        spectral_space: &SpectralSpace<T::RealField, ()>,
    ) -> color_eyre::Result<()>
    where
        Conn: Connectivity<T::RealField, GeometryDim>,
        DefaultAllocator: Allocator<T::RealField, GeometryDim>,
    {
        for ((retarded_gf, retarded_self_energy), energy) in self
            .retarded
            .iter_mut()
            .zip(self_energy.retarded.iter())
            .zip(spectral_space.iter_energy())
        {
            retarded_gf.as_mut().generate_retarded_into(
                *energy,
                T::zero().real(),
                hamiltonian,
                retarded_self_energy,
            )?;
        }
        Ok(())
    }

    fn update_aggregate_lesser_greens_function<GeometryDim: SmallDim, Conn>(
        &mut self,
        self_energy: &SelfEnergy<T, GeometryDim, Conn, CsrMatrix<T>>,
        spectral_space: &SpectralSpace<T::RealField, ()>,
    ) -> color_eyre::Result<()>
    where
        Conn: Connectivity<T::RealField, GeometryDim>,
        DefaultAllocator: Allocator<T::RealField, GeometryDim>,
    {
        for (((lesser_gf, retarded_gf), retarded_self_energy), _energy) in self
            .lesser
            .iter_mut()
            .zip(self.retarded.iter())
            .zip(self_energy.retarded.iter())
            .zip(spectral_space.iter_energy())
        {
            lesser_gf
                .as_mut()
                .generate_lesser_into(&retarded_gf.matrix, retarded_self_energy)?;
        }
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
    fn generate_advanced_into(&mut self, retarded: &Self) -> color_eyre::Result<()>;
    fn generate_greater_into(&mut self) -> color_eyre::Result<()>;
    fn generate_lesser_into(
        &mut self,
        retarded_greens_function: &Self,
        retarded_self_energy: &Self,
    ) -> color_eyre::Result<()>;
    fn generate_retarded_into(
        &mut self,
        energy: T::RealField,
        wavevector: T::RealField,
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
        energy: T::RealField,
        wavevector: T::RealField,
        hamiltonian: &Hamiltonian<T::RealField>,
        self_energy: &Self::SelfEnergy,
    ) -> color_eyre::Result<()> {
        let self_energy_values = self_energy.values();
        assert_eq!(
            self_energy_values.len(),
            2,
            "In a 1D problem there should be two non-zero elements in the self-energy matrix"
        );
        let self_energy_values = (self_energy_values[0], self_energy_values[1]);
        // Get the Hamiltonian at this wavevector
        let hamiltonian = hamiltonian.calculate_total(wavevector);
        // Generate the diagonal component of the CSR matrix
        let (retarded_diagonal, left_diagonal) =
            diagonals(energy, &hamiltonian, &self_energy_values)?;
        // Generate the top row
        let retarded_top_row = top_row(
            energy,
            &hamiltonian,
            &retarded_diagonal,
            self_energy_values.1,
        )?;

        // Generate the bottom row
        let retarded_bottom_row = bottom_row(&retarded_diagonal, &left_diagonal, &hamiltonian)?;

        self.assemble_retarded_rows_and_columns_into_csr(
            retarded_diagonal,
            retarded_top_row,
            retarded_bottom_row,
        )
    }

    fn generate_greater_into(&mut self) -> color_eyre::Result<()> {
        unreachable!()
    }

    fn generate_advanced_into(&mut self, _retarded: &CsrMatrix<T>) -> color_eyre::Result<()> {
        unreachable!()
    }

    fn generate_lesser_into(
        &mut self,
        retarded_greens_function: &CsrMatrix<T>,
        retarded_self_energy: &CsrMatrix<T>,
    ) -> color_eyre::Result<()> {
        // In 1D and for 1 band:
        let values = retarded_self_energy.values();
        let se = (values[0], values[1]);
        let pattern = retarded_self_energy.pattern();
        let fermi_source = T::one(); // TODO get from an info desk
        let fermi_drain = T::one();
        let i = T::one(); //TODO Should be the imaginary unit
        let source_se = CsrMatrix::try_from_pattern_and_values(
            pattern.clone(),
            vec![i * fermi_source * se.0, T::zero()],
        )
        .unwrap();
        let drain_se = CsrMatrix::try_from_pattern_and_values(
            pattern.clone(),
            vec![T::zero(), i * fermi_drain * se.1],
        )
        .unwrap();

        let advanced_gf_values = retarded_greens_function
            .values()
            .iter()
            .map(|&x| x.simd_conjugate())
            .collect::<Vec<_>>();
        let retarded_gf_pattern = retarded_greens_function.pattern().clone();
        let advanced_greens_function =
            CsrMatrix::try_from_pattern_and_values(retarded_gf_pattern, advanced_gf_values)
                .unwrap()
                .transpose();

        let gamma_values = retarded_self_energy
            .values()
            .iter()
            .map(|&x| -(T::one() + T::one()) * T::from_real(x.imaginary()))
            .collect::<Vec<_>>();
        let _gamma = CsrMatrix::try_from_pattern_and_values(
            retarded_self_energy.pattern().clone(),
            gamma_values,
        )
        .unwrap();

        let _spectral_density =
            retarded_greens_function * (source_se + drain_se) * advanced_greens_function;
        todo!()
    }
}

trait CsrAssembly<T: ComplexField> {
    fn assemble_retarded_rows_and_columns_into_csr(
        &mut self,
        diagonal: DVector<T>,
        top_row: DVector<T>,
        bottom_row: DVector<T>,
    ) -> color_eyre::Result<()>;
}

impl<T: Copy + ComplexField> CsrAssembly<T> for CsrMatrix<T> {
    fn assemble_retarded_rows_and_columns_into_csr(
        &mut self,
        diagonal: DVector<T>,
        top_row: DVector<T>,
        bottom_row: DVector<T>,
    ) -> color_eyre::Result<()> {
        assert_eq!(
            self.values().len(),
            diagonal.len() + top_row.len() + bottom_row.len() - 2
        );
        for (output, calculated) in self.values_mut().iter_mut().zip(
            top_row
                .into_iter()
                .chain(diagonal.into_iter().skip(1).take(diagonal.len() - 2))
                .chain(bottom_row.into_iter()),
        ) {
            *output = *calculated;
        }
        Ok(())
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
        _wavevector: T::RealField,
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

    fn generate_greater_into(&mut self) -> color_eyre::Result<()> {
        todo!()
    }

    fn generate_advanced_into(&mut self, retarded: &DMatrix<T>) -> color_eyre::Result<()> {
        let mut output: nalgebra::DMatrixSliceMut<T> = self.into();
        output.copy_from(&retarded.conjugate().transpose());
        Ok(())
    }

    fn generate_lesser_into(
        &mut self,
        _retarded: &DMatrix<T>,
        _lesser: &DMatrix<T>,
    ) -> color_eyre::Result<()> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::CsrAssembly;
    use nalgebra::{DMatrix, DVector};
    use nalgebra_sparse::CsrMatrix;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_csr_assemble_of_diagonal_and_top_and_bottom_rows() {
        let nrows = 50;
        let mut rng = thread_rng();

        let top_row: DVector<f64> =
            DVector::from((0..nrows).map(|_| rng.gen()).collect::<Vec<_>>());
        let bottom_row: DVector<f64> =
            DVector::from((0..nrows).map(|_| rng.gen()).collect::<Vec<_>>());
        let mut diagonal: DVector<f64> =
            DVector::from((0..nrows).map(|_| rng.gen()).collect::<Vec<_>>());
        diagonal[0] = top_row[0];
        diagonal[nrows - 1] = bottom_row[nrows - 1];

        let mut dense_matrix: DMatrix<f64> = DMatrix::from_element(nrows, nrows, 0f64);
        for idx in 0..nrows {
            dense_matrix[(0, idx)] = top_row[idx];
            dense_matrix[(nrows - 1, idx)] = bottom_row[idx];
            dense_matrix[(idx, idx)] = diagonal[idx];
        }

        // Construct the sparsity pattern
        let col_indices = (0..top_row.len())
            .chain(1..diagonal.len() - 1)
            .chain(0..bottom_row.len())
            .collect::<Vec<_>>();
        let mut row_offsets = vec![0, top_row.len()];
        let mut diagonal_row_offsets =
            (top_row.len() + 1..top_row.len() + diagonal.len() - 1).collect();
        row_offsets.append(&mut diagonal_row_offsets);
        row_offsets.push(row_offsets.last().unwrap() + bottom_row.len());
        let values = vec![1f64; col_indices.len()];
        let mut csr =
            CsrMatrix::try_from_csr_data(nrows, nrows, row_offsets, col_indices, values).unwrap();

        csr.assemble_retarded_rows_and_columns_into_csr(diagonal, top_row, bottom_row)
            .unwrap();

        let csr_to_dense = nalgebra_sparse::convert::serial::convert_csr_dense(&csr);

        for (element, other) in dense_matrix.into_iter().zip(csr_to_dense.into_iter()) {
            assert_eq!(element, other);
        }
    }
}
