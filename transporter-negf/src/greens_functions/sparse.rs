//! Sparse methods for coherent transport
//!
//! In the coherent transport case it is only necessary to calculate the leading diagonal and two
//! columns of the retarded Greens function, and the diagonal of the lesser Green's function, to
//! progress the self-coherent iteration. This module provides methods for this optimal scenario,
//! where the Green's functions are stored as sparse CsrMatrix.
use super::{
    aggregate::{AggregateGreensFunctionMethods, AggregateGreensFunctions},
    recursive::{build_out_column, diagonal},
    {GreensFunctionInfoDesk, GreensFunctionMethods},
};
use crate::{
    hamiltonian::Hamiltonian,
    postprocessor::{Charge, Current},
    self_energy::SelfEnergy,
    spectral::SpectralSpace,
};
use nalgebra::{
    allocator::Allocator, ComplexField, Const, DVector, DefaultAllocator, Dynamic, Matrix, OVector,
    RealField, VecStorage,
};
use nalgebra_sparse::CsrMatrix;
use num_complex::Complex;
use transporter_mesher::{Connectivity, ElementMethods, Mesh, SmallDim};

/// A trait for commonly used csr assembly patterns
pub(crate) trait CsrAssembly<T: RealField> {
    /// Assembles the given diagonal, and first and last columns into the CsrMatrix `self`
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

/// Implementation of the accumulator methods for a sparse AggregateGreensFunction
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
        // Sum over the diagonal of the calculated spectral density
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
                            weight * element.0.diameter() // Weighted by the integration weight from the `SpectralSpace` and the diameter of the element in the grid
                                / T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap(), // The Green's function is an inverse energy stored in eV
                        )
                },
            )
            .values()
            .iter()
            .map(|&x| x.real()) // The charge in the device is a real quantity
            .collect::<Vec<_>>();

        // Separate out the diagonals for each `BandDim` into their own charge vector
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

        // Multiply by the scalar prefactor to arrive at a physical quantity
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
                *charge_at_element *= prefactor;
            }
        }

        Charge::new(OVector::<DVector<T>, BandDim>::from_iterator(
            charges.into_iter(),
        ))
    }

    fn accumulate_into_current_density_vector<GeometryDimB: SmallDim, Conn>(
        &self,
        mesh: &Mesh<T, GeometryDimB, Conn>,
        _spectral_space: &SpectralSpace<T, ()>,
    ) -> color_eyre::Result<Current<T, BandDim>>
    where
        Conn: Connectivity<T, GeometryDimB>,
        DefaultAllocator: Allocator<T, GeometryDimB>,
    {
        let mut currents: Vec<DVector<T>> = Vec::with_capacity(BandDim::dim());
        // Multiply by the scalar prefactor to arrive at a physical quantity
        for _band_number in 0..BandDim::dim() {
            currents.push(DVector::from(vec![T::zero(); mesh.elements().len()]));
        }
        for (_n_band, current) in currents.iter_mut().enumerate() {
            for (current_at_element, _element) in current.iter_mut().zip(mesh.elements()) {
                *current_at_element = T::zero();
            }
        }
        Current::new(OVector::<DVector<T>, BandDim>::from_iterator(
            currents.into_iter(),
        ))
    }
}

// TODO This is a single band implementation
/// Implementation of the aggregate update methods for coherent transport
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

    pub(crate) fn update_aggregate_retarded_greens_function<Conn>(
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

    pub(crate) fn update_aggregate_lesser_greens_function<Conn>(
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

/// Implemetation of the single point Greens Function update methods for sparse backend
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

    // We never generate a greater greens function in the self-consistent loop. Maybe we will in the future when moving to incoherent transport
    fn generate_greater_into(&mut self, _: &Self, _: &Self, _: &Self) -> color_eyre::Result<()> {
        unreachable!()
    }

    // We also never generate the advanced greens function, instead transiently calculating it in `generate_lesser_into`. Maybe we will in future
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
