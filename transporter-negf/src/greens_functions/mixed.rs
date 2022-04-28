//! Methods for mixed-sparse-dense Green;s functions
//!
//! These methods are for the common situation where we want to consider incoherent transport in the central
//! region of the device, but do not want the numerical overhead of doing so. By partitioning the device
//! into an incoherent region and two reservoirs we can only solve the full transport equations in the core,
//! vastly reducing the numerical overhead

use super::{
    aggregate::{AggregateGreensFunctionMethods, AggregateGreensFunctions},
    recursive::{left_connected_diagonal, right_connected_diagonal},
    GreensFunctionInfoDesk, GreensFunctionMethods,
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
    allocator::Allocator, ComplexField, Const, DMatrix, DVector, DefaultAllocator, Dynamic, Matrix,
    OVector, RealField, VecStorage,
};
use nalgebra_sparse::CsrMatrix;
use num_complex::Complex;
use rayon::prelude::*;
use transporter_mesher::{Connectivity, ElementMethods, Mesh, SmallDim};

/// A mixed matrix, holding the quantity in the reservoirs in `Vec` and in a dense
/// `DMatrix` in the incoherent core.
#[derive(Clone, Debug)]
pub(crate) struct MMatrix<T> {
    source_diagonal: Vec<T>,
    drain_diagonal: Vec<T>,
    core_matrix: DMatrix<T>,
}

impl<T: ComplexField> MMatrix<T> {
    pub(crate) fn zeros(
        number_of_vertices_in_reservoir: usize,
        number_of_vertices_in_core: usize,
    ) -> Self {
        MMatrix {
            source_diagonal: vec![T::zero(); number_of_vertices_in_reservoir],
            drain_diagonal: vec![T::zero(); number_of_vertices_in_reservoir],
            core_matrix: DMatrix::zeros(number_of_vertices_in_core, number_of_vertices_in_core),
        }
    }

    pub(crate) fn core_as_ref(&self) -> &DMatrix<T> {
        &self.core_matrix
    }
}

impl<'a, T, GeometryDim, BandDim>
    AggregateGreensFunctions<'a, T, MMatrix<Complex<T>>, GeometryDim, BandDim>
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
                    &MMatrix {
                        source_diagonal: vec![self_energy.contact_retarded[index].values()[0]],
                        drain_diagonal: vec![self_energy.contact_retarded[index].values()[1]],
                        core_matrix: self_energy.incoherent_retarded.as_deref().unwrap()[index]
                            .clone(),
                    },
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

        let n_energies = spectral_space.number_of_energy_points();
        let n_ele = self_energy.contact_retarded[0].nrows();

        self.lesser
            .par_iter_mut()
            .enumerate()
            .progress_with(pb)
            // .try_for_each(|(index, (wavevector, energy))| {
            .try_for_each(|(index, gf)| {
                let energy = spectral_space.energy_at(index % n_energies);
                let wavevector = spectral_space.wavevector_at(index / n_energies);
                let (source, drain) = (
                    self.info_desk.get_fermi_function_at_source(energy),
                    self.info_desk.get_fermi_function_at_drain(energy, voltage),
                );
                // let contact_lesser = nalgebra_sparse::convert::serial::convert_csr_dense(
                //     &self_energy.contact_retarded[index],
                // );
                let mut contact_lesser = self_energy.contact_retarded[index].clone();
                contact_lesser
                    .values_mut()
                    .iter_mut()
                    .for_each(|val| *val = Complex::new(T::zero(), T::one()) * (*val - val.conj()));
                contact_lesser.values_mut()[0] *= Complex::new(T::zero(), source);
                contact_lesser.values_mut()[1] *= Complex::new(T::zero(), drain);

                let internal_lesser = compute_internal_lesser_self_energies(
                    energy,
                    wavevector,
                    hamiltonian,
                    [
                        self_energy.contact_retarded[index].values()[0],
                        self_energy.contact_retarded[index].values()[1],
                    ],
                    self.retarded[0].matrix.drain_diagonal.len(),
                    [source, drain],
                )?;

                let nrows = self_energy.incoherent_lesser.as_deref().unwrap()[index].nrows();
                let mut se_lesser_core =
                    self_energy.incoherent_lesser.as_deref().unwrap()[index].clone();
                se_lesser_core[(0, 0)] += internal_lesser[0];
                se_lesser_core[(nrows - 1, nrows - 1)] += internal_lesser[1];

                gf.as_mut().generate_lesser_into(
                    energy,
                    wavevector,
                    hamiltonian,
                    &self.retarded[index].matrix,
                    &MMatrix {
                        source_diagonal: vec![contact_lesser.values()[0]],
                        drain_diagonal: vec![contact_lesser.values()[1]],
                        core_matrix: se_lesser_core,
                    },
                    &[source, drain],
                )
            })?;
        Ok(())
    }
}

impl<T> GreensFunctionMethods<T> for MMatrix<Complex<T>>
where
    T: RealField + Copy,
{
    type SelfEnergy = MMatrix<Complex<T>>;

    fn generate_retarded_into(
        &mut self,
        energy: T,
        wavevector: T,
        hamiltonian: &Hamiltonian<T>,
        self_energy: &Self::SelfEnergy,
    ) -> color_eyre::Result<()> {
        // The number of entries is n_rows + 2 * (n_rows - 2 * n_lead - 1)
        let number_of_vertices_in_reservoir = self.source_diagonal.len();
        let number_of_vertices_in_core = self.core_matrix.nrows();

        assert!(
            (self_energy.source_diagonal.len() == 1) & (self_energy.drain_diagonal.len() == 1),
            "both self energies in the contact must have exactly one entry"
        );
        let self_energies_at_external_contacts = (
            self_energy.source_diagonal[0],
            self_energy.drain_diagonal[0],
        );

        // Get the Hamiltonian at this wavevector
        let hamiltonian = hamiltonian.calculate_total(wavevector); // The hamiltonian is minus itself because we are stupid
                                                                   // Generate the diagonal component of the CSR matrix

        // Get the self-energies at the edge of the core region
        let g_00 = left_connected_diagonal(
            energy,
            &hamiltonian,
            &self_energies_at_external_contacts,
            number_of_vertices_in_reservoir,
            number_of_vertices_in_reservoir,
        )?;
        let left_internal_self_energy = g_00[(g_00.shape().0 - 1, 0)]
            * hamiltonian.row(number_of_vertices_in_reservoir).values()[2].powi(2);
        let g_ll = right_connected_diagonal(
            energy,
            &hamiltonian,
            &self_energies_at_external_contacts,
            number_of_vertices_in_reservoir,
            number_of_vertices_in_reservoir,
        )?;
        let right_internal_self_energy = g_ll[(0, 0)]
            * hamiltonian
                .row(hamiltonian.nrows() - 1 - number_of_vertices_in_reservoir)
                .values()[2]
                .powi(2);

        // TODO Casting to Complex here is verbose and wasteful, can we try not to do this?
        // Maybe the Hamiltonian needs to be made in terms of `ComplexField`?
        let values = hamiltonian.values();
        let mut y = Vec::with_capacity(values.len());
        for value in values {
            y.push(Complex::from(*value));
        }
        let mut dense_hamiltonian = nalgebra_sparse::convert::serial::convert_csr_dense(
            &CsrMatrix::try_from_pattern_and_values(hamiltonian.pattern().clone(), y).unwrap(),
        );
        dense_hamiltonian[(
            number_of_vertices_in_reservoir,
            number_of_vertices_in_reservoir,
        )] += left_internal_self_energy;
        dense_hamiltonian[(
            number_of_vertices_in_reservoir + number_of_vertices_in_core - 1,
            number_of_vertices_in_reservoir + number_of_vertices_in_core - 1,
        )] += right_internal_self_energy;

        let mut matrix = DMatrix::identity(number_of_vertices_in_core, number_of_vertices_in_core)
            * Complex::from(energy)
            - dense_hamiltonian.slice(
                (
                    number_of_vertices_in_reservoir,
                    number_of_vertices_in_reservoir,
                ),
                (number_of_vertices_in_core, number_of_vertices_in_core),
            ); //TODO Do we have to convert? Seems dumb. Should we store H in dense form too?&ham;
        if matrix.try_inverse_mut() {
            self.core_matrix.copy_from(&matrix);
        } else {
            return Err(color_eyre::eyre::eyre!("Failed to invert for rgf"));
        }

        // Use the dyson equation to fill G^R in the drain
        let right_diagonal = right_connected_diagonal(
            energy,
            &hamiltonian,
            &self_energies_at_external_contacts,
            2 * number_of_vertices_in_reservoir + number_of_vertices_in_core - 1,
            number_of_vertices_in_reservoir,
        )?;
        let mut previous = self.core_matrix[(
            number_of_vertices_in_core - 1,
            number_of_vertices_in_core - 1,
        )];
        let mut previous_hopping_element = hamiltonian
            .row(number_of_vertices_in_reservoir + number_of_vertices_in_core - 1)
            .values()[2];
        self.drain_diagonal
            .iter_mut()
            .zip(
                hamiltonian
                    .row_iter()
                    .zip(right_diagonal.into_iter())
                    .skip(number_of_vertices_in_reservoir + number_of_vertices_in_core - 1),
            )
            .for_each(|(element, (hamiltonian_row, right_diagonal_element))| {
                let hopping_element = hamiltonian_row.values()[2];
                *element = right_diagonal_element
                    * (Complex::from(T::one())
                        + right_diagonal_element
                            * previous
                            * hopping_element
                            * previous_hopping_element);
                previous_hopping_element = hopping_element;
                previous = *element;
            });

        // Use the Dyson equation to fill G^R in the source
        let left_diagonal = left_connected_diagonal(
            energy,
            &hamiltonian,
            &self_energies_at_external_contacts,
            hamiltonian.nrows(),
            number_of_vertices_in_reservoir,
        )?;
        previous = self.core_matrix[(0, 0)];
        previous_hopping_element = hamiltonian.row(number_of_vertices_in_reservoir).values()[2];
        self.source_diagonal
            .iter_mut()
            .zip(
                left_diagonal
                    .into_iter()
                    .take(number_of_vertices_in_reservoir),
            )
            .rev()
            .enumerate()
            .for_each(|(idx, (element, left_diagonal_element))| {
                let row = hamiltonian.row(number_of_vertices_in_reservoir - 1 - idx);
                let hopping_element = if row.values().len() == 3 {
                    T::from_real(row.values()[0])
                } else {
                    T::from_real(row.values()[1])
                };
                *element = left_diagonal_element
                    * (Complex::from(T::one())
                        + left_diagonal_element
                            * previous
                            * hopping_element
                            * previous_hopping_element);
                previous_hopping_element = hopping_element;
                previous = *element;
            });

        Ok(())
    }

    // We never generate a greater greens function in the self-consistent loop. Maybe we will in the future when moving to incoherent transport
    fn generate_greater_into(&mut self, _: &Self, _: &Self, _: &Self) -> color_eyre::Result<()> {
        unreachable!()
    }

    // We also never generate the advanced greens function, instead transiently calculating it in `generate_lesser_into`. Maybe we will in future
    fn generate_advanced_into(
        &mut self,
        _retarded: &MMatrix<Complex<T>>,
    ) -> color_eyre::Result<()> {
        unreachable!()
    }

    fn generate_lesser_into(
        &mut self,
        energy: T,
        wavevector: T,
        hamiltonian: &Hamiltonian<T>,
        retarded_greens_function: &MMatrix<Complex<T>>,
        lesser_self_energy: &MMatrix<Complex<T>>,
        fermi_functions: &[T],
    ) -> color_eyre::Result<()> {
        // Expensive matrix inversion
        self.core_matrix
            .iter_mut()
            .zip(
                (&retarded_greens_function.core_matrix
                    * &lesser_self_energy.core_matrix
                    * retarded_greens_function.core_matrix.transpose().conjugate())
                .iter(),
            )
            .for_each(|(element, &value)| {
                *element = value;
            });

        // Fill the simple diagonal elements G^{<} = i f A
        self.source_diagonal
            .iter_mut()
            .zip(retarded_greens_function.source_diagonal.iter())
            .for_each(|(element, g_r)| {
                let spectral_density = Complex::new(T::zero(), T::one()) * (g_r - g_r.conj());
                *element = Complex::new(T::zero(), fermi_functions[0]) * spectral_density;
            });
        self.drain_diagonal
            .iter_mut()
            .zip(retarded_greens_function.drain_diagonal.iter())
            .for_each(|(element, g_r)| {
                let spectral_density = Complex::new(T::zero(), T::one()) * (g_r - g_r.conj());
                *element = Complex::new(T::zero(), fermi_functions[1]) * spectral_density;
            });

        Ok(())
    }
}

fn compute_internal_lesser_self_energies<T: RealField + Copy>(
    energy: T,
    wavevector: T,
    hamiltonian: &Hamiltonian<T>,
    edge_retarded_self_energies: [Complex<T>; 2],
    number_of_vertices_in_reservoir: usize,
    fermi_functions: [T; 2],
) -> color_eyre::Result<[Complex<T>; 2]> {
    let retarded_self_energies_internal = compute_internal_retarded_self_energies(
        energy,
        wavevector,
        hamiltonian,
        edge_retarded_self_energies,
        number_of_vertices_in_reservoir,
    )?;

    let lesser_se = retarded_self_energies_internal
        .iter()
        .zip(fermi_functions.into_iter())
        .map(|(s_re, f)| {
            Complex::new(T::zero(), f) * Complex::new(T::zero(), T::one()) * (s_re - s_re.conj())
        })
        .collect::<Vec<_>>();
    Ok([lesser_se[0], lesser_se[1]])
}

fn compute_internal_retarded_self_energies<T: RealField + Copy>(
    energy: T,
    wavevector: T,
    hamiltonian: &Hamiltonian<T>,
    edge_retarded_self_energies: [Complex<T>; 2],
    number_of_vertices_in_reservoir: usize,
) -> color_eyre::Result<[Complex<T>; 2]> {
    let self_energies_at_external_contacts = (
        edge_retarded_self_energies[0],
        edge_retarded_self_energies[1],
    );
    // Get the Hamiltonian at this wavevector
    let hamiltonian = hamiltonian.calculate_total(wavevector); // The hamiltonian is minus itself because we are stupid
                                                               // Generate the diagonal component of the CSR matrix

    // Get the self-energies at the edge of the core region
    let g_00 = left_connected_diagonal(
        energy,
        &hamiltonian,
        &self_energies_at_external_contacts,
        number_of_vertices_in_reservoir,
        number_of_vertices_in_reservoir,
    )?;
    let left_internal_self_energy = g_00[(g_00.shape().0 - 1, 0)]
        * hamiltonian.row(number_of_vertices_in_reservoir).values()[2].powi(2);
    let g_ll = right_connected_diagonal(
        energy,
        &hamiltonian,
        &self_energies_at_external_contacts,
        number_of_vertices_in_reservoir,
        number_of_vertices_in_reservoir,
    )?;
    let right_internal_self_energy = g_ll[(0, 0)]
        * hamiltonian
            .row(hamiltonian.nrows() - 1 - number_of_vertices_in_reservoir)
            .values()[2]
            .powi(2);
    Ok([left_internal_self_energy, right_internal_self_energy])
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
    > for AggregateGreensFunctions<'_, T, MMatrix<Complex<T>>, GeometryDim, BandDim>
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
        // TODO Only one band
        let mut summed_diagonal = vec![
            T::zero();
            self.retarded[0].as_ref().drain_diagonal.len() * 2
                + self.retarded[0].as_ref().core_matrix.nrows()
        ];

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
                    DVector::zeros(summed_diagonal.len()),
                    |sum, ((value, weight), width)| {
                        let matrix = &value.matrix;
                        let diagonal = DVector::from(
                            matrix
                                .source_diagonal
                                .iter()
                                .chain(matrix.core_matrix.diagonal().iter())
                                .chain(matrix.drain_diagonal.iter())
                                .map(|x| *x)
                                .collect::<Vec<_>>(),
                        );
                        sum + diagonal
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
                        .iter()
                        .map(|&x| -(Complex::new(T::zero(), T::one()) * x).re),
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
    ) -> color_eyre::Result<Current<T, BandDim>> {
        let mut currents: Vec<DVector<T>> = Vec::with_capacity(BandDim::dim());
        let number_of_vertices_in_internal_lead = self.retarded[0].as_ref().source_diagonal.len();

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
                    let gf_r_1n = gf_r.as_ref().core_matrix[(
                        gf_r.as_ref().core_matrix.nrows() - 1,
                        gf_r.as_ref().core_matrix.nrows() - 1,
                    )];

                    let abs_gf_r_1n_with_factor = (gf_r_1n * gf_r_1n.conj()).re
                        * width
                        * weight
                        * gamma_source
                        * gamma_drain
                        * (fermi_source - fermi_drain)
                        * T::from_f64(0.01_f64.powi(2) / 1e5).unwrap(); // Convert to x 10^5 A / cm^2

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
