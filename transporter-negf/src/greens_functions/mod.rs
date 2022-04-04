mod recursive;

use recursive::{build_out_column, diagonal};

use crate::postprocessor::{Charge, Current};
use crate::self_energy::SelfEnergy;
use crate::{
    device::info_desk::DeviceInfoDesk,
    spectral::{SpectralDiscretisation, SpectralSpace, WavevectorSpace},
    Hamiltonian,
};
use color_eyre::eyre::eyre;
use nalgebra::{
    allocator::Allocator, ComplexField, Const, DMatrix, DVector, DefaultAllocator, Dynamic, Matrix,
    OVector, RealField, VecStorage,
};
use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};
use num_complex::Complex;
use transporter_mesher::{Connectivity, ElementMethods, Mesh, SmallDim};

#[derive(Clone)]
pub(crate) struct GreensFunctionBuilder<T, RefInfoDesk, RefMesh, RefSpectral> {
    info_desk: RefInfoDesk,
    mesh: RefMesh,
    spectral: RefSpectral,
    marker: std::marker::PhantomData<T>,
}

impl<T> GreensFunctionBuilder<T, (), (), ()>
where
    T: RealField,
{
    pub(crate) fn new() -> Self {
        Self {
            info_desk: (),
            mesh: (),
            spectral: (),
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, RefInfoDesk, RefMesh, RefSpectral>
    GreensFunctionBuilder<T, RefInfoDesk, RefMesh, RefSpectral>
{
    pub(crate) fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> GreensFunctionBuilder<T, &InfoDesk, RefMesh, RefSpectral> {
        GreensFunctionBuilder {
            info_desk,
            mesh: self.mesh,
            spectral: self.spectral,
            marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> GreensFunctionBuilder<T, RefInfoDesk, &Mesh, RefSpectral> {
        GreensFunctionBuilder {
            info_desk: self.info_desk,
            mesh,
            spectral: self.spectral,
            marker: std::marker::PhantomData,
        }
    }
    pub(crate) fn with_spectral_discretisation<Spectral>(
        self,
        spectral: &Spectral,
    ) -> GreensFunctionBuilder<T, RefInfoDesk, RefMesh, &Spectral> {
        GreensFunctionBuilder {
            info_desk: self.info_desk,
            mesh: self.mesh,
            spectral,
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T, GeometryDim, Conn, BandDim>
    GreensFunctionBuilder<
        T,
        &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
        &'a Mesh<T, GeometryDim, Conn>,
        &'a SpectralSpace<T, ()>,
    >
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator:
        Allocator<T, GeometryDim> + Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    pub(crate) fn build(
        self,
    ) -> color_eyre::Result<
        AggregateGreensFunctions<'a, T, CsrMatrix<Complex<T>>, GeometryDim, BandDim>,
    > {
        // A 1D implementation. All 2D should redirect to the dense method
        let pattern = assemble_csr_sparsity_for_retarded_gf(self.mesh.elements().len())?;
        let values = vec![Complex::from(T::zero()); pattern.nnz()];
        let csr = CsrMatrix::try_from_pattern_and_values(pattern, values)
            .map_err(|e| eyre!("Failed to write values to Csr GF Matrix {:?}", e))?;
        let spectrum_of_csr = (0..self.spectral.total_number_of_points())
            .map(|_| GreensFunction {
                matrix: csr.clone(),
                marker: std::marker::PhantomData,
            })
            .collect::<Vec<_>>();

        let diagonal_pattern = csr.diagonal_as_csr().pattern().clone();
        let values = vec![Complex::from(T::zero()); diagonal_pattern.nnz()];
        let diagonal_csr = CsrMatrix::try_from_pattern_and_values(diagonal_pattern, values)
            .map_err(|e| eyre!("Failed to write values to Csr GF Matrix {:?}", e))?;
        let spectrum_of_diagonal_csr = (0..self.spectral.total_number_of_points())
            .map(|_| GreensFunction {
                matrix: diagonal_csr.clone(),
                marker: std::marker::PhantomData,
            })
            .collect::<Vec<_>>();

        // In the coherent calculation we do not use the advanced or greater Greens function, other than transiently
        Ok(AggregateGreensFunctions {
            //    spectral: self.spectral,
            info_desk: self.info_desk,
            retarded: spectrum_of_csr,
            advanced: Vec::new(),
            lesser: spectrum_of_diagonal_csr,
            greater: Vec::new(),
        })
    }
}

fn assemble_csr_sparsity_for_retarded_gf(
    number_of_elements_in_mesh: usize,
) -> color_eyre::Result<SparsityPattern> {
    let mut col_indices = vec![0, number_of_elements_in_mesh - 1];
    let mut row_offsets = vec![0, 2]; // 2 elements in the first row
    for idx in 1..number_of_elements_in_mesh - 1 {
        col_indices.push(0);
        col_indices.push(idx);
        col_indices.push(number_of_elements_in_mesh - 1);
        row_offsets.push(row_offsets.last().unwrap() + 3)
    }
    col_indices.push(0);
    col_indices.push(number_of_elements_in_mesh - 1);
    row_offsets.push(row_offsets.last().unwrap() + 2);
    SparsityPattern::try_from_offsets_and_indices(
        number_of_elements_in_mesh,
        number_of_elements_in_mesh,
        row_offsets,
        col_indices,
    )
    .map_err(|e| eyre!("Failed to construct Csr Sparsity pattern {:?}", e))
}

impl<'a, T, GeometryDim, Conn, BandDim>
    GreensFunctionBuilder<
        T,
        &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
        &'a Mesh<T, GeometryDim, Conn>,
        &'a SpectralSpace<T, WavevectorSpace<T, GeometryDim, Conn>>,
    >
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator:
        Allocator<T, GeometryDim> + Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    pub(crate) fn build(
        self,
    ) -> AggregateGreensFunctions<'a, T, DMatrix<Complex<T>>, GeometryDim, BandDim> {
        AggregateGreensFunctions {
            //    spectral: self.spectral,
            info_desk: self.info_desk,
            retarded: Vec::with_capacity(self.spectral.total_number_of_points()),
            advanced: Vec::with_capacity(self.spectral.total_number_of_points()),
            lesser: Vec::with_capacity(self.spectral.total_number_of_points()),
            greater: Vec::with_capacity(self.spectral.total_number_of_points()),
        }
    }
}
#[derive(Debug)]
pub(crate) struct AggregateGreensFunctions<'a, T, Matrix, GeometryDim, BandDim>
where
    Matrix: GreensFunctionMethods<T>,
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    // spectral: &'a dyn SpectralDiscretisation<T>,
    info_desk: &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
    retarded: Vec<GreensFunction<Matrix, T>>,
    advanced: Vec<GreensFunction<Matrix, T>>,
    lesser: Vec<GreensFunction<Matrix, T>>,
    greater: Vec<GreensFunction<Matrix, T>>,
}

pub(crate) trait AggregateGreensFunctionMethods<T, BandDim, Integrator>
where
    T: RealField,
    BandDim: SmallDim,
    Integrator: SpectralDiscretisation<T>,
    DefaultAllocator: Allocator<
        Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
        BandDim,
    >,
{
    fn accumulate_into_charge_density_vector<GeometryDim, Conn>(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        integrator: &Integrator,
    ) -> color_eyre::Result<Charge<T, BandDim>>
    where
        GeometryDim: SmallDim,
        Conn: Connectivity<T, GeometryDim>,
        DefaultAllocator: Allocator<T, GeometryDim>;
    fn accumulate_into_current_density_vector(
        &self,
        integrator: &Integrator,
    ) -> color_eyre::Result<Current<T, BandDim>>;
}

impl<T, BandDim, GeometryDim> AggregateGreensFunctionMethods<T, BandDim, SpectralSpace<T, ()>>
    for AggregateGreensFunctions<'_, T, CsrMatrix<Complex<T>>, GeometryDim, BandDim>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>,
{
    /// Only for one band -> need to skip the iter
    fn accumulate_into_charge_density_vector<GeometryDimB: SmallDim, Conn>(
        &self,
        mesh: &Mesh<T, GeometryDimB, Conn>,
        spectral_space: &SpectralSpace<T, ()>,
    ) -> color_eyre::Result<Charge<T, BandDim>>
    where
        Conn: Connectivity<T, GeometryDimB>,
        DefaultAllocator: Allocator<T, GeometryDimB>,
    {
        let mut charges: Vec<DVector<T>> = Vec::with_capacity(BandDim::dim());
        let summed_diagonal = self
            .lesser
            .iter()
            .zip(spectral_space.energy.weights())
            .zip(spectral_space.energy.grid.elements())
            .fold(
                &self.lesser[0].matrix * Complex::from(T::zero()),
                |sum, ((value, &weight), element)| {
                    sum + &value.matrix
                        * Complex::from(
                            weight * element.0.diameter()
                                / T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap(), // The Green's function is an inverse energy stored in eV
                        )
                },
            )
            .values()
            .iter()
            .map(|x| x.real())
            .collect::<Vec<_>>();

        for band_number in 0..BandDim::dim() {
            charges.push(DVector::from(
                summed_diagonal
                    .iter()
                    .skip(band_number)
                    .step_by(BandDim::dim())
                    .map(|&x| x.real())
                    .collect::<Vec<_>>(),
            ));
        }

        for (n_band, charge) in charges.iter_mut().enumerate() {
            for (charge_at_element, element) in charge.iter_mut().zip(mesh.elements()) {
                let region = element.1;
                let prefactor = T::from_f64(
                    crate::constants::BOLTZMANN
                        * crate::constants::ELECTRON_CHARGE
                        * crate::constants::ELECTRON_MASS
                        / 2.
                        / std::f64::consts::PI.powi(2)
                        / crate::constants::HBAR.powi(2),
                )
                .unwrap()
                    * self.info_desk.temperature
                    * self.info_desk.effective_masses[region][n_band][1]
                    / element.0.diameter();
                *charge_at_element *= -prefactor;
            }
        }

        dbg!(&charges);

        Charge::new(OVector::<DVector<T>, BandDim>::from_iterator(
            charges.into_iter(),
        ))
    }

    fn accumulate_into_current_density_vector(
        &self,
        _spectral_space: &SpectralSpace<T, ()>,
    ) -> color_eyre::Result<Current<T, BandDim>> {
        todo!()
    }
}

// TODO This is a single band implementation
impl<'a, T, GeometryDim, BandDim, Matrix>
    AggregateGreensFunctions<'a, T, Matrix, GeometryDim, BandDim>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Matrix: GreensFunctionMethods<T>,
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    pub(crate) fn update_greens_functions<Conn>(
        &mut self,
        hamiltonian: &Hamiltonian<T>,
        self_energy: &SelfEnergy<T, GeometryDim, Conn, Matrix>,
        spectral_space: &SpectralSpace<T, ()>,
    ) -> color_eyre::Result<()>
    where
        Conn: Connectivity<T, GeometryDim>,
        DefaultAllocator: Allocator<T, GeometryDim>,
    {
        // In the coherent transport case we only need the retarded and lesser Greens functions (see Lake 1997)
        self.update_aggregate_retarded_greens_function(hamiltonian, self_energy, spectral_space)?;
        self.update_aggregate_lesser_greens_function(self_energy, spectral_space)?;
        Ok(())
    }

    fn update_aggregate_retarded_greens_function<Conn>(
        &mut self,
        hamiltonian: &Hamiltonian<T>,
        self_energy: &SelfEnergy<T, GeometryDim, Conn, Matrix>,
        spectral_space: &SpectralSpace<T, ()>,
    ) -> color_eyre::Result<()>
    where
        Conn: Connectivity<T, GeometryDim>,
        DefaultAllocator: Allocator<T, GeometryDim>,
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

    fn update_aggregate_lesser_greens_function<Conn>(
        &mut self,
        self_energy: &SelfEnergy<T, GeometryDim, Conn, Matrix>,
        spectral_space: &SpectralSpace<T, ()>,
    ) -> color_eyre::Result<()>
    where
        Conn: Connectivity<T, GeometryDim>,
        DefaultAllocator: Allocator<T, GeometryDim>,
    {
        for (((lesser_gf, retarded_gf), retarded_self_energy), &energy) in self
            .lesser
            .iter_mut()
            .zip(self.retarded.iter())
            .zip(self_energy.retarded.iter())
            .zip(spectral_space.iter_energy())
        {
            let source_fermi_integral = self.info_desk.get_fermi_integral_at_source(energy);
            let drain_fermi_integral = self.info_desk.get_fermi_integral_at_drain(energy);
            lesser_gf.as_mut().generate_lesser_into(
                &retarded_gf.matrix,
                retarded_self_energy,
                &[source_fermi_integral, drain_fermi_integral],
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct GreensFunction<Matrix, T>
where
    Matrix: GreensFunctionMethods<T>,
    T: RealField + Copy,
{
    matrix: Matrix,
    marker: std::marker::PhantomData<T>,
}

impl<Matrix, T> GreensFunction<Matrix, T>
where
    Matrix: GreensFunctionMethods<T>,
    T: RealField + Copy,
{
    fn as_mut(&mut self) -> &mut Matrix {
        &mut self.matrix
    }
}

pub(crate) trait GreensFunctionMethods<T>
where
    T: RealField + Copy,
{
    type SelfEnergy;
    fn generate_advanced_into(&mut self, retarded: &Self) -> color_eyre::Result<()>;
    fn generate_greater_into(
        &mut self,
        lesser: &Self,
        retarded: &Self,
        advanced: &Self,
    ) -> color_eyre::Result<()>;
    fn generate_lesser_into(
        &mut self,
        retarded_greens_function: &Self,
        retarded_self_energy: &Self,
        fermi_levels: &[T],
    ) -> color_eyre::Result<()>;
    fn generate_retarded_into(
        &mut self,
        energy: T,
        wavevector: T,
        hamiltonian: &Hamiltonian<T>,
        self_energy: &Self,
    ) -> color_eyre::Result<()>;
}

//impl<T> GreensFunctionMethods<T> for CsrMatrix<Complex<T>>
//where
//    T: ComplexField + Copy,
//    <T as ComplexField>::RealField: Copy,
//{
//    type Output = CsrMatrix<Complex<T>>;
//    type SelfEnergy = (T, T);
//    fn generate_retarded(
//        energy: T,
//        hamiltonian: &Hamiltonian<T>,
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
//    fn generate_advanced(retarded: CsrMatrix<Complex<T>>) -> color_eyre::Result<Box<Self>> {
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
//        retarded: CsrMatrix<Complex<T>>,
//        lesser: CsrMatrix<Complex<T>>,
//    ) -> color_eyre::Result<Box<Self>> {
//        todo!()
//    }
//}

impl<T> GreensFunctionMethods<T> for CsrMatrix<Complex<T>>
where
    T: RealField + Copy,
{
    type SelfEnergy = CsrMatrix<Complex<T>>;

    fn generate_retarded_into(
        &mut self,
        energy: T,
        wavevector: T,
        hamiltonian: &Hamiltonian<T>,
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
        let hamiltonian = hamiltonian.calculate_total(wavevector); // The hamiltonian is minus itself because we are stupid
                                                                   // Generate the diagonal component of the CSR matrix
        let g_ii = diagonal(energy, &hamiltonian, &self_energy_values)?;
        // Generate the top row
        let g_i0 = build_out_column(energy, &hamiltonian, &g_ii, &self_energy_values, 0)?;

        // Generate the bottom row
        let g_in = build_out_column(
            energy,
            &hamiltonian,
            &g_ii,
            &self_energy_values,
            hamiltonian.nrows() - 1,
        )?;

        self.assemble_retarded_diagonal_and_columns_into_csr(g_ii, g_i0, g_in)
    }

    fn generate_greater_into(&mut self, _: &Self, _: &Self, _: &Self) -> color_eyre::Result<()> {
        unreachable!()
    }

    fn generate_advanced_into(
        &mut self,
        _retarded: &CsrMatrix<Complex<T>>,
    ) -> color_eyre::Result<()> {
        unreachable!()
    }

    fn generate_lesser_into(
        &mut self,
        retarded_greens_function: &CsrMatrix<Complex<T>>,
        retarded_self_energy: &CsrMatrix<Complex<T>>,
        fermi_functions: &[T],
    ) -> color_eyre::Result<()> {
        // In 1D and for 1 band:
        let advanced_gf_values = retarded_greens_function
            .values()
            .iter()
            .map(|&x| x.conjugate())
            .collect::<Vec<_>>();
        let retarded_gf_pattern = retarded_greens_function.pattern().clone();
        let advanced_greens_function =
            CsrMatrix::try_from_pattern_and_values(retarded_gf_pattern, advanced_gf_values)
                .unwrap()
                .transpose();

        let gamma_values = retarded_self_energy
            .values()
            .iter()
            .zip(fermi_functions)
            .map(|(&x, &fermi)| Complex::new(T::zero(), fermi * T::one()) * (x - x.conjugate()))
            .collect::<Vec<_>>();
        let gamma = CsrMatrix::try_from_pattern_and_values(
            retarded_self_energy.pattern().clone(),
            gamma_values,
        )
        .unwrap();

        let spectral_density_diagonal =
            (retarded_greens_function * (gamma) * advanced_greens_function).diagonal_as_csr();

        for (element, &value) in self
            .values_mut()
            .iter_mut()
            .zip(spectral_density_diagonal.values().iter())
        {
            *element = value;
        }
        Ok(())
    }
}

trait CsrAssembly<T: RealField> {
    fn assemble_retarded_diagonal_and_columns_into_csr(
        &mut self,
        diagonal: DVector<Complex<T>>,
        left_column: DVector<Complex<T>>,
        right_column: DVector<Complex<T>>,
    ) -> color_eyre::Result<()>;
}

impl<T: Copy + RealField> CsrAssembly<T> for CsrMatrix<Complex<T>> {
    fn assemble_retarded_diagonal_and_columns_into_csr(
        &mut self,
        diagonal: DVector<Complex<T>>,
        left_column: DVector<Complex<T>>,
        right_column: DVector<Complex<T>>,
    ) -> color_eyre::Result<()> {
        let n_values = self.values().len();
        assert_eq!(
            n_values,
            diagonal.len() + left_column.len() + right_column.len() - 2
        );
        self.values_mut()[0] = diagonal[0];
        self.values_mut()[1] = right_column[0];
        for row in 1..diagonal.len() - 1 {
            self.values_mut()[2 + (row - 1) * 3] = left_column[row];
            self.values_mut()[3 + (row - 1) * 3] = diagonal[row];
            self.values_mut()[4 + (row - 1) * 3] = right_column[row];
        }
        self.values_mut()[n_values - 2] = left_column[left_column.len() - 1];
        self.values_mut()[n_values - 1] = diagonal[diagonal.len() - 1];
        Ok(())
    }
}

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

pub(crate) trait GreensFunctionInfoDesk<T: Copy + RealField> {
    fn get_fermi_level_at_source(&self) -> T;
    fn get_fermi_level_at_drain(&self) -> T;
    fn get_fermi_integral_at_source(&self, energy: T) -> T;
    fn get_fermi_integral_at_drain(&self, energy: T) -> T;
}

impl<'a, T: Copy + RealField, BandDim: SmallDim, GeometryDim: SmallDim> GreensFunctionInfoDesk<T>
    for &'a DeviceInfoDesk<T, GeometryDim, BandDim>
where
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    fn get_fermi_level_at_source(&self) -> T {
        // The net doping density in the source contact (region 0)
        let doping_density = self.donor_densities[0] - self.acceptor_densities[0];

        //TODO : A single band impl here.
        let n_band = 0;
        let n = (T::one() + T::one())
            * (self.effective_masses[0][0][n_band]
                * T::from_f64(crate::constants::ELECTRON_MASS).unwrap()
                * T::from_f64(crate::constants::BOLTZMANN).unwrap()
                * self.temperature
                / T::from_f64(crate::constants::HBAR).unwrap().powi(2)
                / (T::one() + T::one())
                / T::from_f64(std::f64::consts::PI).unwrap())
            .powf(T::from_f64(1.5).unwrap());

        let band_offset = self.band_offsets[0][n_band]; // + T::from_f64(0.02).unwrap();

        let (factor, gamma) = (
            T::from_f64(crate::constants::ELECTRON_CHARGE / crate::constants::BOLTZMANN).unwrap()
                / self.temperature,
            T::from_f64(std::f64::consts::PI.sqrt() / 2.).unwrap(),
        );
        let eta_f = crate::fermi::inverse_fermi_integral_p(gamma * doping_density / n);
        let ef_minus_ec = eta_f / factor;
        ef_minus_ec + band_offset
    }

    fn get_fermi_level_at_drain(&self) -> T {
        self.get_fermi_level_at_source() + self.voltage_offsets[1]
    }

    fn get_fermi_integral_at_source(&self, energy: T) -> T {
        let fermi_level = self.get_fermi_level_at_source(); // - T::one() / (T::one() + T::one());
        let argument = T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
            * (fermi_level - energy)
            / (T::from_f64(crate::constants::BOLTZMANN).unwrap() * self.temperature);
        (T::one() + argument.exp()).ln()
    }

    fn get_fermi_integral_at_drain(&self, energy: T) -> T {
        let fermi_level = self.get_fermi_level_at_drain(); // - T::one() / (T::one() + T::one());
        let argument = T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
            * (fermi_level - energy)
            / (T::from_f64(crate::constants::BOLTZMANN).unwrap() * self.temperature);
        (T::one() + argument.exp()).ln()
    }
}

#[cfg(test)]
mod test {
    use super::{AggregateGreensFunctionMethods, CsrAssembly};
    use nalgebra::{DMatrix, DVector};
    use nalgebra_sparse::CsrMatrix;
    use num_complex::Complex;
    use rand::{thread_rng, Rng};

    use super::{
        AggregateGreensFunctions, DeviceInfoDesk, GreensFunction, GreensFunctionBuilder,
        SpectralDiscretisation, SpectralSpace,
    };
    use crate::self_energy::SelfEnergy;
    use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
    use transporter_mesher::{Connectivity, Mesh, SmallDim};

    impl<'a, T, GeometryDim, Conn, BandDim>
        GreensFunctionBuilder<
            T,
            &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
            &'a Mesh<T, GeometryDim, Conn>,
            &'a SpectralSpace<T, ()>,
        >
    where
        T: RealField + Copy,
        GeometryDim: SmallDim,
        BandDim: SmallDim,
        Conn: Connectivity<T, GeometryDim>,
        DefaultAllocator:
            Allocator<T, GeometryDim> + Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
    {
        /// A test helper method to build dense representations for coherent calculations. This is to aid testing, as we can
        /// then directly compare the results of the sparse and dense implementations of the update trait.
        pub(crate) fn build_dense(
            self,
        ) -> color_eyre::Result<
            AggregateGreensFunctions<'a, T, DMatrix<Complex<T>>, GeometryDim, BandDim>,
        > {
            // A 1D implementation. All 2D should redirect to the dense method
            let dmatrix = DMatrix::zeros(self.mesh.elements().len(), self.mesh.elements().len());

            let spectrum_of_dmatrix = (0..self.spectral.total_number_of_points())
                .map(|_| GreensFunction {
                    matrix: dmatrix.clone(),
                    marker: std::marker::PhantomData,
                })
                .collect::<Vec<_>>();

            // In the coherent calculation we do not use the advanced or greater Greens function, other than transiently
            Ok(AggregateGreensFunctions {
                //    spectral: self.spectral,
                info_desk: self.info_desk,
                retarded: spectrum_of_dmatrix.clone(),
                advanced: spectrum_of_dmatrix.clone(),
                lesser: spectrum_of_dmatrix.clone(),
                greater: spectrum_of_dmatrix,
            })
        }
    }

    /// Helper function to convert sparse self energies to their dense equivalents to enable comparative testing
    /// of sparse and dense implementations
    fn convert_sparse_self_energy_to_dense<T: Copy + RealField, GeometryDim, Conn>(
        sparse_self_energy: SelfEnergy<T, GeometryDim, Conn, CsrMatrix<Complex<T>>>,
    ) -> SelfEnergy<T, GeometryDim, Conn, DMatrix<Complex<T>>>
    where
        GeometryDim: SmallDim,
        Conn: Connectivity<T, GeometryDim>,
        DefaultAllocator: Allocator<T, GeometryDim>,
    {
        // Take the sparse self-energy matrix and re-serialize to a DMatrix
        let retarded_se_matrices = sparse_self_energy
            .retarded
            .iter()
            .map(nalgebra_sparse::convert::serial::convert_csr_dense)
            .collect::<Vec<_>>();
        SelfEnergy {
            ma: sparse_self_energy.ma,
            mc: sparse_self_energy.mc,
            marker: sparse_self_energy.marker,
            retarded: retarded_se_matrices,
        }
    }

    #[test]
    fn test_csr_assemble_of_diagonal_and_left_and_right_columns() {
        let nrows = 5;
        let mut rng = thread_rng();

        let left_column: DVector<Complex<f64>> = DVector::from(
            (0..nrows)
                .map(|_| rng.gen::<f64>())
                .map(Complex::from)
                .collect::<Vec<_>>(),
        );
        let right_column: DVector<Complex<f64>> = DVector::from(
            (0..nrows)
                .map(|_| rng.gen::<f64>())
                .map(Complex::from)
                .collect::<Vec<_>>(),
        );
        let mut diagonal: DVector<Complex<f64>> = DVector::from(
            (0..nrows)
                .map(|_| rng.gen::<f64>())
                .map(Complex::from)
                .collect::<Vec<_>>(),
        );
        diagonal[0] = left_column[0];
        diagonal[nrows - 1] = right_column[nrows - 1];

        let mut dense_matrix: DMatrix<Complex<f64>> =
            DMatrix::from_element(nrows, nrows, Complex::from(0f64));
        for idx in 0..nrows {
            dense_matrix[(idx, 0)] = left_column[idx];
            dense_matrix[(idx, nrows - 1)] = right_column[idx];
            dense_matrix[(idx, idx)] = diagonal[idx];
        }

        // Construct the sparsity pattern
        let number_of_elements_in_mesh = diagonal.len();
        let pattern =
            super::assemble_csr_sparsity_for_retarded_gf(number_of_elements_in_mesh).unwrap();
        let values = vec![Complex::from(0_f64); pattern.nnz()];
        let mut csr = CsrMatrix::try_from_pattern_and_values(pattern, values).unwrap();

        csr.assemble_retarded_diagonal_and_columns_into_csr(diagonal, left_column, right_column)
            .unwrap();

        let csr_to_dense = nalgebra_sparse::convert::serial::convert_csr_dense(&csr);

        println!("{csr_to_dense}");
        println!("{dense_matrix}");

        for (element, other) in dense_matrix.into_iter().zip(csr_to_dense.into_iter()) {
            assert_eq!(element, other);
        }
    }

    use crate::app::{Configuration, TrackerBuilder};
    use crate::device::{info_desk::BuildInfoDesk, Device};
    use nalgebra::U1;

    #[test]
    fn diagonal_elements_of_retarded_csr_match_dense() {
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

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new()
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();
        self_energy
            .recalculate(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        let dense_se = convert_sparse_self_energy_to_dense(self_energy);

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &dense_se, &spectral_space)
            .unwrap();

        for ((sparse, dense), energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let sparse_diagonal = sparse.matrix.diagonal_as_csr();
            let dense_diagonal = dense.matrix.diagonal();
            for (sparse_value, dense_value) in sparse_diagonal
                .values()
                .iter()
                .zip(dense_diagonal.into_iter())
            {
                approx::assert_relative_eq!(sparse_value.re, dense_value.re, epsilon = 1e-10);
                approx::assert_relative_eq!(sparse_value.im, dense_value.im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn left_column_elements_of_retarded_csr_match_dense() {
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

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new()
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();
        self_energy
            .recalculate(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        let dense_se = convert_sparse_self_energy_to_dense(self_energy);

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &dense_se, &spectral_space)
            .unwrap();

        for ((sparse, dense), energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let dense_diagonal = dense.matrix.column(0);
            for (sparse_row, dense_value) in
                sparse.matrix.row_iter().zip(dense_diagonal.into_iter())
            {
                let sparse_value = sparse_row.values()[0];
                approx::assert_relative_eq!(sparse_value.re, dense_value.re, epsilon = 1e-10);
                approx::assert_relative_eq!(sparse_value.im, dense_value.im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn right_column_elements_of_retarded_csr_match_dense() {
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

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new()
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();
        self_energy
            .recalculate(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        let dense_se = convert_sparse_self_energy_to_dense(self_energy);

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &dense_se, &spectral_space)
            .unwrap();

        for ((sparse, dense), energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let dense_diagonal = dense.matrix.column(dense.matrix.ncols() - 1);
            for (sparse_row, dense_value) in
                sparse.matrix.row_iter().zip(dense_diagonal.into_iter())
            {
                let sparse_value = sparse_row.values()[sparse_row.values().len() - 1];
                approx::assert_relative_eq!(sparse_value.re, dense_value.re, epsilon = 1e-10);
                approx::assert_relative_eq!(sparse_value.im, dense_value.im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn diagonal_elements_of_lesser_csr_match_dense() {
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

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new()
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();
        self_energy
            .recalculate(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        // Act
        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();
        gf.update_aggregate_lesser_greens_function(&self_energy, &spectral_space)
            .unwrap();

        let mut dense_gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build_dense()
            .unwrap();

        let dense_se = convert_sparse_self_energy_to_dense(self_energy);

        dense_gf
            .update_aggregate_retarded_greens_function(&hamiltonian, &dense_se, &spectral_space)
            .unwrap();

        dense_gf
            .update_aggregate_lesser_greens_function(&dense_se, &spectral_space)
            .unwrap();

        for ((sparse, dense), energy) in gf
            .retarded
            .iter()
            .zip(dense_gf.retarded.iter())
            .zip(spectral_space.energy.points())
        {
            let sparse_diagonal = sparse.matrix.diagonal_as_csr();
            let dense_diagonal = dense.matrix.diagonal();
            for (sparse_value, dense_value) in sparse_diagonal
                .values()
                .iter()
                .zip(dense_diagonal.into_iter())
            {
                println!("energy: {energy}, sparse: {sparse_value}, dense: {dense_value}");
                approx::assert_relative_eq!(sparse_value.re, dense_value.re, epsilon = 1e-10);
                approx::assert_relative_eq!(sparse_value.im, dense_value.im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn accumulated_charge_matches_the_edge_doping_with_zero_potential() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let n_layer = info_desk.donor_densities.len();
        let edge_doping = (
            info_desk.donor_densities[0],
            info_desk.donor_densities[n_layer - 1],
        );

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

        // Begin by building a coherent spectral space, regardless of calculation we begin with a coherent loop
        let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
            .with_number_of_energy_points(config.spectral.number_of_energy_points)
            .with_energy_range(std::ops::Range {
                start: config.spectral.minimum_energy,
                end: config.spectral.maximum_energy,
            })
            .with_energy_integration_method(config.spectral.energy_integration_rule);

        let spectral_space = spectral_space_builder.build_coherent();

        let mut gf = super::GreensFunctionBuilder::new()
            .with_info_desk(&info_desk)
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();

        let mut self_energy = crate::self_energy::SelfEnergyBuilder::new()
            .with_mesh(&mesh)
            .with_spectral_discretisation(&spectral_space)
            .build()
            .unwrap();
        self_energy
            .recalculate(&mesh, &hamiltonian, &spectral_space)
            .unwrap();

        gf.update_aggregate_retarded_greens_function(&hamiltonian, &self_energy, &spectral_space)
            .unwrap();
        gf.update_aggregate_lesser_greens_function(&self_energy, &spectral_space)
            .unwrap();

        let charge = gf
            .accumulate_into_charge_density_vector(&mesh, &spectral_space)
            .unwrap();

        let x = &charge.as_ref()[0]; // We have one band so take the charge density at index 0

        let n_elements = x.len();
        approx::assert_relative_eq!(x[0], edge_doping.0);
        approx::assert_relative_eq!(x[n_elements - 1], edge_doping.1);
        println!("{:?}", x);
    }
}
