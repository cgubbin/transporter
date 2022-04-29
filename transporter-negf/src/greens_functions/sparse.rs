//! Sparse methods for coherent transport
//!
//! In the coherent transport case it is only necessary to calculate the leading diagonal and two
//! columns of the retarded Greens function, and the diagonal of the lesser Green's function, to
//! progress the self-coherent iteration. This module provides methods for this optimal scenario,
//! where the Green's functions are stored as sparse CsrMatrix.
use super::{
    aggregate::{AggregateGreensFunctionMethods, AggregateGreensFunctions},
    recursive::{build_out_column, diagonal, left_connected_diagonal, right_connected_diagonal},
    {GreensFunctionError, GreensFunctionInfoDesk, GreensFunctionMethods},
};
use crate::{
    hamiltonian::Hamiltonian,
    postprocessor::{Charge, Current},
    self_energy::SelfEnergy,
    spectral::SpectralDiscretisation,
};
use console::Term;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressDrawTarget, ProgressStyle};
use nalgebra::{
    allocator::Allocator, ComplexField, Const, DVector, DefaultAllocator, Dynamic, Matrix, OVector,
    RealField, SimdComplexField, VecStorage,
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
    ) -> Result<(), GreensFunctionError>;
}

impl<T: Copy + RealField> CsrAssembly<T> for CsrMatrix<Complex<T>> {
    fn assemble_retarded_diagonal_and_columns_into_csr(
        &mut self,
        diagonal: DVector<Complex<T>>,
        left_column: DVector<Complex<T>>,
        right_column: DVector<Complex<T>>,
    ) -> Result<(), GreensFunctionError> {
        let n_values = self.values().len();
        assert_eq!(
            n_values,
            diagonal.len() + left_column.len() + right_column.len() - 2
        );
        assert_eq!(left_column.len(), right_column.len());
        if diagonal.len() == left_column.len() {
            self.values_mut()[0] = diagonal[0];
            self.values_mut()[1] = right_column[0];
            for row in 1..diagonal.len() - 1 {
                self.values_mut()[2 + (row - 1) * 3] = left_column[row];
                self.values_mut()[3 + (row - 1) * 3] = diagonal[row];
                self.values_mut()[4 + (row - 1) * 3] = right_column[row];
            }
            self.values_mut()[n_values - 2] = left_column[left_column.len() - 1];
            self.values_mut()[n_values - 1] = diagonal[diagonal.len() - 1];
        } else {
            assert_eq!(
                (diagonal.len() - left_column.len()) % 2,
                0,
                "There must be equal numbers of elements in each extended contact"
            );
            let num_elements_in_contact = (diagonal.len() - left_column.len()) / 2;
            // Diagonal entries only
            for row in 0..num_elements_in_contact {
                self.values_mut()[row] = diagonal[row];
                self.values_mut()[n_values - num_elements_in_contact + row] =
                    diagonal[diagonal.len() - num_elements_in_contact + row]
            }
            // Rows with two elements
            self.values_mut()[num_elements_in_contact] = left_column[0];
            self.values_mut()[num_elements_in_contact + 1] = right_column[0];

            self.values_mut()[n_values - 2 - num_elements_in_contact] =
                left_column[left_column.len() - 1];
            self.values_mut()[n_values - 1 - num_elements_in_contact] =
                right_column[right_column.len() - 1];

            for index in 1..(diagonal.len() - 2 * num_elements_in_contact - 1) {
                self.values_mut()[num_elements_in_contact + 2 + (index - 1) * 3] =
                    left_column[index];
                self.values_mut()[num_elements_in_contact + 3 + (index - 1) * 3] =
                    diagonal[num_elements_in_contact + index];
                self.values_mut()[num_elements_in_contact + 4 + (index - 1) * 3] =
                    right_column[index];
            }
        }
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
        SelfEnergy<T, GeometryDim, Conn>,
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
    ) -> Result<Charge<T, BandDim>, crate::postprocessor::PostProcessorError> {
        let mut charges: Vec<DVector<T>> = Vec::with_capacity(BandDim::dim());
        // Sum over the diagonal of the calculated spectral density
        let mut summed_diagonal = vec![T::zero(); self.lesser[0].matrix.values().len()];

        if spectral_space.number_of_wavevector_points() == 1 {}

        for (idx, ((wavevector, weight), width)) in spectral_space
            .iter_wavevectors()
            .zip(spectral_space.iter_wavevector_weights())
            .zip(spectral_space.iter_wavevector_widths())
            .enumerate()
        {
            let wavevector = if spectral_space.number_of_wavevector_points() == 1 {
                T::one()
            } else {
                wavevector
            };

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
                );
            summed_diagonal
                .iter_mut()
                .zip(
                    new_diagonal
                        .values()
                        .iter()
                        .map(|&x| -(Complex::new(T::zero(), T::one()) * x).real()),
                )
                .for_each(|(ele, new)| *ele += wavevector * weight * width * new);
        }

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
            for (idx, (charge_at_element, vertex)) in
                charge.iter_mut().zip(mesh.vertices()).enumerate()
            {
                let assignment = &vertex.1;

                // STRICTLY 1D ONLY - REDO
                let diameter = if idx == 0 {
                    [0, 0]
                } else if idx == mesh.vertices().len() - 1 {
                    [idx - 1, idx - 1]
                } else {
                    [idx, idx - 1]
                }
                .into_iter()
                .fold(T::zero(), |acc, idx| {
                    acc + mesh.elements()[idx].0.diameter() / (T::one() + T::one())
                });

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
                            * self
                                .info_desk
                                .effective_mass_from_assignment(assignment, n_band, 1)
                            / diameter
                    }
                    _ => {
                        T::from_f64(
                            crate::constants::ELECTRON_CHARGE
                                / 2_f64
                                / std::f64::consts::PI.powi(2),
                        )
                        .unwrap()
                            / diameter
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
        voltage: T,
        mesh: &Mesh<T, GeometryDim, Conn>,
        self_energy: &SelfEnergy<T, GeometryDim, Conn>,
        spectral_space: &Spectral,
    ) -> Result<Current<T, BandDim>, crate::postprocessor::PostProcessorError> {
        let mut currents: Vec<DVector<T>> = Vec::with_capacity(BandDim::dim());
        let number_of_vertices_in_internal_lead =
            (3 * self.retarded[0].matrix.nrows() - self.retarded[0].matrix.nnz() - 2) / 4;

        let summed_current = self
            .retarded
            .iter()
            .zip(self_energy.contact_retarded.iter())
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
                    let fermi_drain = self.info_desk.get_fermi_integral_at_drain(energy, voltage);

                    // TODO Handle finite internal leads. This assumes the device continues to the contacts
                    let gf_r_1n = gf_r.matrix.values()[number_of_vertices_in_internal_lead + 1];

                    let abs_gf_r_1n_with_factor = (gf_r_1n * gf_r_1n.conj()).re
                        * width
                        * weight
                        * gamma_source
                        * gamma_drain
                        * (fermi_source - fermi_drain)
                        * T::from_f64(0.01.powi(2) / 1e5).unwrap(); // Convert to x 10^5 A / cm^2

                    sum.into_iter()
                        .map(|sum| sum + abs_gf_r_1n_with_factor)
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
impl<'a, T, GeometryDim, BandDim>
    AggregateGreensFunctions<'a, T, CsrMatrix<Complex<T>>, GeometryDim, BandDim>
where
    T: RealField + Copy + Clone + Send + Sync,
    GeometryDim: SmallDim + Send + Sync,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    #[tracing::instrument(name = "Updating Greens Functions", skip_all)]
    pub(crate) fn update_greens_functions<Conn, Spectral>(
        &mut self,
        voltage: T,
        hamiltonian: &Hamiltonian<T>,
        self_energy: &SelfEnergy<T, GeometryDim, Conn>,
        spectral_space: &Spectral,
    ) -> Result<(), GreensFunctionError>
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
        self.update_aggregate_lesser_greens_function(
            voltage,
            hamiltonian,
            self_energy,
            spectral_space,
        )?;
        Ok(())
    }

    pub(crate) fn update_aggregate_retarded_greens_function<Conn, Spectral>(
        &mut self,
        hamiltonian: &Hamiltonian<T>,
        self_energy: &SelfEnergy<T, GeometryDim, Conn>,
        spectral_space: &Spectral,
    ) -> Result<(), GreensFunctionError>
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
                    &self_energy.contact_retarded[index],
                )
            })?;
        Ok(())
    }

    pub(crate) fn update_aggregate_lesser_greens_function<Conn, Spectral>(
        &mut self,
        voltage: T,
        hamiltonian: &Hamiltonian<T>,
        self_energy: &SelfEnergy<T, GeometryDim, Conn>,
        spectral_space: &Spectral,
    ) -> Result<(), GreensFunctionError>
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
                        self.info_desk.get_fermi_integral_at_drain(energy, voltage),
                    ),
                    _ => (
                        self.info_desk.get_fermi_function_at_source(energy),
                        self.info_desk.get_fermi_function_at_drain(energy, voltage),
                    ),
                };

                let energy = spectral_space.energy_at(index % n_energies);
                let wavevector = spectral_space.wavevector_at(index / n_energies);
                gf.as_mut().generate_lesser_into(
                    energy,
                    wavevector,
                    hamiltonian,
                    &self.retarded[index].matrix,
                    &self_energy.contact_retarded[index],
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
    ) -> Result<(), GreensFunctionError> {
        // The number of entries is n_rows + 2 * (n_rows - 2 * n_lead - 1)
        assert_eq!((3 * self.nrows() - self.nnz() - 2) % 4, 0);
        let number_of_vertices_in_internal_lead = (3 * self.nrows() - self.nnz() - 2) / 4;

        let self_energy_values_at_contact = self_energy.values();
        assert_eq!(
            self_energy_values_at_contact.len(),
            2,
            "In a 1D problem there should be two non-zero elements in the self-energy matrix"
        );
        let self_energy_values_at_contact = (
            self_energy_values_at_contact[0],
            self_energy_values_at_contact[1],
        );
        // Get the Hamiltonian at this wavevector
        let hamiltonian = hamiltonian.calculate_total(wavevector); // The hamiltonian is minus itself because we are stupid
                                                                   // Generate the diagonal component of the CSR matrix
        let g_ii = diagonal(
            energy,
            &hamiltonian,
            &self_energy_values_at_contact,
            number_of_vertices_in_internal_lead,
        )?;
        // Generate the top row
        let g_i0 = build_out_column(
            energy,
            &hamiltonian,
            &g_ii,
            &self_energy_values_at_contact,
            number_of_vertices_in_internal_lead,
            number_of_vertices_in_internal_lead,
        )?;
        let g_i0 = g_i0
            .rows(0, self.nrows() - 2 * number_of_vertices_in_internal_lead)
            .clone_owned();

        // Generate the bottom row
        let g_in = build_out_column(
            energy,
            &hamiltonian,
            &g_ii,
            &self_energy_values_at_contact,
            hamiltonian.nrows() - 1 - number_of_vertices_in_internal_lead,
            number_of_vertices_in_internal_lead,
        )?;
        let g_in = g_in
            .rows(
                number_of_vertices_in_internal_lead,
                self.nrows() - 2 * number_of_vertices_in_internal_lead,
            )
            .clone_owned();

        self.assemble_retarded_diagonal_and_columns_into_csr(g_ii, g_i0, g_in)
    }

    // We never generate a greater greens function in the self-consistent loop. Maybe we will in the future when moving to incoherent transport
    fn generate_greater_into(
        &mut self,
        _: &Self,
        _: &Self,
        _: &Self,
    ) -> Result<(), GreensFunctionError> {
        unimplemented!()
    }

    // We also never generate the advanced greens function, instead transiently calculating it in `generate_lesser_into`. Maybe we will in future
    fn generate_advanced_into(
        &mut self,
        _retarded: &CsrMatrix<Complex<T>>,
    ) -> Result<(), GreensFunctionError> {
        unimplemented!()
    }

    fn generate_lesser_into(
        &mut self,
        energy: T,
        wavevector: T,
        hamiltonian: &Hamiltonian<T>,
        retarded_greens_function: &CsrMatrix<Complex<T>>,
        retarded_self_energy: &CsrMatrix<Complex<T>>,
        fermi_functions: &[T],
    ) -> Result<(), GreensFunctionError> {
        // The number of entries is n_rows + 2 * (n_rows - 2 * n_lead - 1)
        let number_of_vertices_in_internal_lead =
            (3 * retarded_greens_function.nrows() - retarded_greens_function.nnz() - 2) / 4;
        let n_ele = self.values().len();

        if number_of_vertices_in_internal_lead == 0 {
            // In 1D and for 1 band:
            let gamma_values = retarded_self_energy
                .values()
                .iter()
                .zip(fermi_functions)
                .map(|(&x, &fermi)| Complex::new(T::zero(), fermi * T::one()) * (x - x.conjugate()))
                .collect::<Vec<_>>();

            for (element, value) in self
                .values_mut()
                .iter_mut()
                .zip(retarded_greens_function.row_iter())
            {
                let left = Complex::from(value.get_entry(0).unwrap().into_value().simd_abs());
                let right =
                    Complex::from(value.get_entry(n_ele - 1).unwrap().into_value().simd_abs());
                *element = Complex::new(T::zero(), T::one())
                    * (left.powi(2) * gamma_values[0] + right.powi(2) * gamma_values[1]);
            }
        } else {
            // G^{<} = i f_{e} A
            for (row, (element, value)) in self
                .values_mut()
                .iter_mut()
                .zip(retarded_greens_function.row_iter())
                .enumerate()
                .take(number_of_vertices_in_internal_lead)
            {
                let g_r = value.get_entry(row).unwrap().into_value();
                let spectral_density = Complex::new(T::zero(), T::one()) * (g_r - g_r.conj());
                *element = Complex::new(T::zero(), fermi_functions[0]) * spectral_density;
            }
            let self_energies_at_external_contacts = (
                retarded_self_energy.values()[0],
                retarded_self_energy.values()[1],
            );

            let hamiltonian = hamiltonian.calculate_total(wavevector);
            let left_internal_self_energy = left_connected_diagonal(
                energy,
                &hamiltonian,
                &self_energies_at_external_contacts,
                number_of_vertices_in_internal_lead,
                number_of_vertices_in_internal_lead,
            )?;
            let left_internal_self_energy = left_internal_self_energy
                [(left_internal_self_energy.shape().0 - 1, 0)]
                * hamiltonian
                    .row(number_of_vertices_in_internal_lead)
                    .values()[2]
                    .powi(2);

            let right_internal_self_energy = right_connected_diagonal(
                energy,
                &hamiltonian,
                &self_energies_at_external_contacts,
                number_of_vertices_in_internal_lead,
                number_of_vertices_in_internal_lead,
            )?;
            let right_internal_self_energy = right_internal_self_energy[(0, 0)]
                * hamiltonian
                    .row(hamiltonian.nrows() - 1 - number_of_vertices_in_internal_lead)
                    .values()[2]
                    .powi(2);
            // let gamma_values = vec![
            //     Complex::new(T::zero(), fermi_functions[0])
            //         * (self_energies_at_external_contacts.0
            //             - self_energies_at_external_contacts.0.conj()),
            //     Complex::new(T::zero(), fermi_functions[1])
            //         * (self_energies_at_external_contacts.1
            //             - self_energies_at_external_contacts.1.conj()),
            // ];

            let gamma_values = vec![
                Complex::new(T::zero(), fermi_functions[0])
                    * (left_internal_self_energy - left_internal_self_energy.conj()),
                Complex::new(T::zero(), fermi_functions[1])
                    * (right_internal_self_energy - right_internal_self_energy.conj()),
            ];

            for (element, value) in self
                .values_mut()
                .iter_mut()
                .zip(retarded_greens_function.row_iter())
                .skip(number_of_vertices_in_internal_lead)
                .take(n_ele - 2 * number_of_vertices_in_internal_lead)
            {
                let left = Complex::from(
                    value
                        .get_entry(number_of_vertices_in_internal_lead)
                        .unwrap()
                        .into_value()
                        .simd_abs(),
                );
                let right = Complex::from(
                    value
                        .get_entry(n_ele - 1 - number_of_vertices_in_internal_lead)
                        .unwrap()
                        .into_value()
                        .simd_abs(),
                );
                *element = Complex::new(T::zero(), T::one())
                    * (left.powi(2) * gamma_values[0] + right.powi(2) * gamma_values[1]);
            }

            for (row, (element, value)) in self
                .values_mut()
                .iter_mut()
                .zip(retarded_greens_function.row_iter())
                .enumerate()
                .skip(n_ele - number_of_vertices_in_internal_lead)
            {
                let g_r = value.get_entry(row).unwrap().into_value();
                let spectral_density = Complex::new(T::zero(), T::one()) * (g_r - g_r.conj());
                *element = Complex::new(T::zero(), fermi_functions[1]) * spectral_density;
            }
        }
        Ok(())
    }
}
