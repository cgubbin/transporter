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
    spectral::SpectralDiscretisation,
};
use console::Term;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, ComplexField, Const, DVector, DefaultAllocator, Dynamic, Matrix, OVector,
    RealField, VecStorage,
};
use nalgebra_sparse::CsrMatrix;
use num_complex::Complex;
use rayon::prelude::*;
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
impl<T, BandDim, GeometryDim, Conn, Spectral>
    AggregateGreensFunctionMethods<
        T,
        BandDim,
        GeometryDim,
        Conn,
        Spectral,
        SelfEnergy<T, GeometryDim, Conn, CsrMatrix<Complex<T>>>,
    > for AggregateGreensFunctions<'_, T, CsrMatrix<Complex<T>>, GeometryDim, BandDim>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    Spectral: SpectralDiscretisation<T>,
    DefaultAllocator: Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<T, GeometryDim>,
{
    fn accumulate_into_charge_density_vector(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        spectral_space: &Spectral,
    ) -> color_eyre::Result<Charge<T, BandDim>> {
        let mut charges: Vec<DVector<T>> = Vec::with_capacity(BandDim::dim());
        // Sum over the diagonal of the calculated spectral density
        let mut summed_diagonal = vec![T::zero(); self.lesser[0].matrix.values().len()];

        for (idx, ((wavevector, weight), width)) in spectral_space
            .iter_wavevectors()
            .zip(spectral_space.iter_wavevector_weights())
            .zip(spectral_space.iter_wavevector_widths())
            .enumerate()
        {
            let new_diagonal = self
                .lesser
                .iter()
                .skip(idx * spectral_space.number_of_energy_points())
                .zip(spectral_space.iter_energy_weights())
                .zip(spectral_space.iter_energy_widths())
                .fold(
                    &self.lesser[0].matrix * Complex::from(T::zero()),
                    |sum, ((value, weight), width)| {
                        sum + &value.matrix
                            * Complex::from(
                                weight * width // Weighted by the integration weight from the `SpectralSpace` and the diameter of the element in the grid
                                / T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap(), // The Green's function is an inverse energy stored in eV
                            )
                    },
                )
                .values()
                .iter()
                .map(|&x| x.real())
                .collect::<Vec<_>>(); // The charge in the device is a real quantity
            summed_diagonal
                .iter_mut()
                .zip(new_diagonal.into_iter())
                .for_each(|(ele, new)| *ele += wavevector * weight * width * new);
        }
        // let summed_diagonal = self
        //     .lesser
        //     .iter()
        //     .zip(spectral_space.iter_energy_weights())
        //     .zip(spectral_space.iter_energy_widths())
        //     .fold(
        //         &self.lesser[0].matrix * Complex::from(T::zero()),
        //         |sum, ((value, weight), width)| {
        //             sum + &value.matrix
        //                 * Complex::from(
        //                     weight * width // Weighted by the integration weight from the `SpectralSpace` and the diameter of the element in the grid
        //                         / T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap(), // The Green's function is an inverse energy stored in eV
        //                 )
        //         },
        //     )
        //     .values()
        //     .iter()
        //     .map(|&x| x.real()) // The charge in the device is a real quantity
        //     .collect::<Vec<_>>();

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

        // Multiply by the scalar prefactor in to arrive at a physical quantity
        for (n_band, charge) in charges.iter_mut().enumerate() {
            for (charge_at_element, element) in charge.iter_mut().zip(mesh.elements()) {
                let region = element.1;
                let prefactor = match spectral_space.number_of_wavevector_points() {
                    1 => {
                        T::from_f64(
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
                            / element.0.diameter()
                    }
                    _ => {
                        T::from_f64(
                            crate::constants::ELECTRON_CHARGE
                                / 2_f64
                                / std::f64::consts::PI.powi(2),
                        )
                        .unwrap()
                            / element.0.diameter()
                    }
                };
                *charge_at_element *= prefactor;
            }
        }

        Charge::new(OVector::<DVector<T>, BandDim>::from_iterator(
            charges.into_iter(),
        ))
    }

    fn accumulate_into_current_density_vector(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        self_energy: &SelfEnergy<T, GeometryDim, Conn, CsrMatrix<Complex<T>>>,
        spectral_space: &Spectral,
    ) -> color_eyre::Result<Current<T, BandDim>> {
        let mut currents: Vec<DVector<T>> = Vec::with_capacity(BandDim::dim());

        let summed_current = self
            .retarded
            .iter()
            .zip(self_energy.retarded.iter())
            .zip(spectral_space.iter_energy_weights())
            .zip(spectral_space.iter_energy_widths())
            .zip(spectral_space.iter_energies())
            .fold(
                vec![T::zero(); mesh.elements().len() * BandDim::dim()],
                |sum, ((((gf_r, se_r), weight), width), energy)| {
                    // Gamma = i (\Sigma_r - \Sigma_a) -> in coherent transport \Sigma has only two non-zero elements
                    let gamma_source = -(T::one() + T::one()) * se_r.values()[0].im;
                    let gamma_drain = -(T::one() + T::one()) * se_r.values()[1].im;
                    // Zero order Fermi integrals
                    let fermi_source = self.info_desk.get_fermi_integral_at_source(energy);
                    let fermi_drain = self.info_desk.get_fermi_integral_at_drain(energy);

                    let values = gf_r
                        .matrix
                        .values()
                        .iter()
                        .take(mesh.elements().len() * BandDim::dim());
                    values
                        .map(|value| (value * value.conj()).re)
                        .map(|grga| {
                            grga * weight
                                * width
                                * weight
                                * gamma_source
                                * gamma_drain
                                * (fermi_source - fermi_drain)
                                / T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
                        })
                        .zip(sum.into_iter())
                        .map(|(grga, sum)| sum + grga)
                        .collect()
                },
            );
        // Separate out the diagonals for each `BandDim` into their own charge vector
        for band_number in 0..BandDim::dim() {
            currents.push(DVector::from(
                summed_current
                    .iter()
                    .skip(band_number)
                    .step_by(BandDim::dim())
                    .map(|&x| x.real())
                    .collect::<Vec<_>>(),
            ));
        }
        // Multiply by the scalar prefactor to arrive at a physical quantity
        for (n_band, current) in currents.iter_mut().enumerate() {
            for (current_at_element, element) in current.iter_mut().zip(mesh.elements()) {
                let region = element.1;
                let prefactor = T::from_f64(
                    crate::constants::BOLTZMANN
                        * crate::constants::ELECTRON_CHARGE.powi(2)
                        * crate::constants::ELECTRON_MASS
                        / 2.
                        / std::f64::consts::PI.powi(2)
                        / crate::constants::HBAR.powi(3),
                )
                .unwrap()
                    * self.info_desk.temperature
                    * self.info_desk.effective_masses[region][n_band][1];
                *current_at_element *= prefactor;
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
    T: RealField + Copy + Clone + Send + Sync,
    GeometryDim: SmallDim + Send + Sync,
    BandDim: SmallDim,
    Matrix: GreensFunctionMethods<T> + Send + Sync,
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    #[tracing::instrument(name = "Updating Greens Functions", skip_all)]
    pub(crate) fn update_greens_functions<Conn, Spectral>(
        &mut self,
        hamiltonian: &Hamiltonian<T>,
        self_energy: &SelfEnergy<T, GeometryDim, Conn, Matrix>,
        spectral_space: &Spectral,
    ) -> color_eyre::Result<()>
    where
        Conn: Connectivity<T, GeometryDim> + Send + Sync,
        Spectral: SpectralDiscretisation<T>,
        DefaultAllocator: Allocator<T, GeometryDim>,
        <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
    {
        // In the coherent transport case we only need the retarded and lesser Greens functions (see Lake 1997)
        self.update_aggregate_retarded_greens_function(hamiltonian, self_energy, spectral_space)?;
        self.update_aggregate_lesser_greens_function(self_energy, spectral_space)?;
        Ok(())
    }

    pub(crate) fn update_aggregate_retarded_greens_function<Conn, Spectral>(
        &mut self,
        hamiltonian: &Hamiltonian<T>,
        self_energy: &SelfEnergy<T, GeometryDim, Conn, Matrix>,
        spectral_space: &Spectral,
    ) -> color_eyre::Result<()>
    where
        Conn: Connectivity<T, GeometryDim> + Send + Sync,
        Spectral: SpectralDiscretisation<T>,
        DefaultAllocator: Allocator<T, GeometryDim>,
    {
        tracing::info!("retarded ");

        let term = Term::stdout();

        // Display
        let spinner_style = ProgressStyle::default_spinner()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
            .template(
                "{prefix:.bold.dim} {spinner} {msg} [{wide_bar:.cyan/blue}] {percent}% ({eta})",
            );
        let pb = ProgressBar::with_draw_target(
            (spectral_space.number_of_energy_points()
                * spectral_space.number_of_wavevector_points()) as u64,
            ProgressDrawTarget::term(term, 60),
        );
        pb.set_style(spinner_style);

        let n_energies = spectral_space.number_of_energy_points();
        self.retarded
            .par_iter_mut()
            .enumerate()
            .progress_with(pb)
            // .try_for_each(|(index, (wavevector, energy))| {
            .try_for_each(|(index, gf)| {
                let energy = spectral_space.energy_at(index % n_energies);
                let wavevector = spectral_space.wavevector_at(index / n_energies);
                gf.as_mut().generate_retarded_into(
                    energy,
                    wavevector,
                    hamiltonian,
                    &self_energy.retarded[index],
                )
            })?;
        Ok(())
    }

    pub(crate) fn update_aggregate_lesser_greens_function<Conn, Spectral>(
        &mut self,
        self_energy: &SelfEnergy<T, GeometryDim, Conn, Matrix>,
        spectral_space: &Spectral,
    ) -> color_eyre::Result<()>
    where
        Conn: Connectivity<T, GeometryDim> + Send + Sync,
        Spectral: SpectralDiscretisation<T>,
        DefaultAllocator: Allocator<T, GeometryDim>,
        <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
    {
        tracing::info!("lesser");

        let term = Term::stdout();

        // Display
        let spinner_style = ProgressStyle::default_spinner()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
            .template(
                "{prefix:.bold.dim} {spinner} {msg} [{wide_bar:.cyan/blue}] {percent}% ({eta})",
            );
        let pb = ProgressBar::with_draw_target(
            (spectral_space.number_of_energy_points()
                * spectral_space.number_of_wavevector_points()) as u64,
            ProgressDrawTarget::term(term, 60),
        );
        pb.set_style(spinner_style);

        let n_wavevectors = spectral_space.number_of_wavevector_points();
        let n_energies = spectral_space.number_of_energy_points();

        self.lesser
            .par_iter_mut()
            .enumerate()
            .progress_with(pb)
            // .try_for_each(|(index, (wavevector, energy))| {
            .try_for_each(|(index, gf)| {
                let energy = spectral_space.energy_at(index % n_energies);
                let (source, drain) = match n_wavevectors {
                    1 => (
                        self.info_desk.get_fermi_integral_at_source(energy),
                        self.info_desk.get_fermi_integral_at_drain(energy),
                    ),
                    _ => (
                        self.info_desk.get_fermi_function_at_source(energy),
                        self.info_desk.get_fermi_function_at_drain(energy),
                    ),
                };
                gf.as_mut().generate_lesser_into(
                    &self.retarded[index].matrix,
                    &self_energy.retarded[index],
                    &[source, drain],
                )
            })?;
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
