//! Methods for mixed-sparse-dense Green;s functions
//!
//! These methods are for the common situation where we want to consider incoherent transport in the central
//! region of the device, but do not want the numerical overhead of doing so. By partitioning the device
//! into an incoherent region and two reservoirs we can only solve the full transport equations in the core,
//! vastly reducing the numerical overhead

use super::{
    super::{GreensFunctionError, GreensFunctionInfoDesk, GreensFunctionMethods},
    aggregate::{AggregateGreensFunctionMethods, AggregateGreensFunctions},
    recursive::{left_connected_diagonal, right_connected_diagonal},
};
use crate::{
    hamiltonian::Hamiltonian,
    postprocessor::{Charge, Current},
    self_energy::SelfEnergy,
    spectral::SpectralDiscretisation,
};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, OVector};
use ndarray::{s, Array1, Array2};
use num_complex::Complex;
use rayon::prelude::*;
use transporter_mesher::{Connectivity, ElementMethods, Mesh, SmallDim};

/// A mixed matrix, holding the quantity in the reservoirs in `Vec` and in a dense
/// `DMatrix` in the incoherent core.
#[derive(Clone, Debug)]
pub(crate) struct MMatrix<T> {
    source_diagonal: Vec<T>,
    pub(crate) drain_diagonal: Vec<T>,
    pub(crate) core_matrix: Array2<T>,
}

impl<T: ComplexField> MMatrix<T> {
    pub(crate) fn zeros(
        number_of_vertices_in_reservoir: usize,
        number_of_vertices_in_core: usize,
    ) -> Self {
        MMatrix {
            source_diagonal: vec![T::zero(); number_of_vertices_in_reservoir],
            drain_diagonal: vec![T::zero(); number_of_vertices_in_reservoir],
            core_matrix: Array2::zeros((number_of_vertices_in_core, number_of_vertices_in_core)),
        }
    }

    pub(crate) fn core_as_ref(&self) -> &Array2<T> {
        &self.core_matrix
    }
}

impl<'a, GeometryDim, BandDim>
    AggregateGreensFunctions<'a, f64, MMatrix<Complex<f64>>, GeometryDim, BandDim>
where
    GeometryDim: SmallDim + Send + Sync,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<f64, BandDim> + Allocator<[f64; 3], BandDim>,
{
    #[tracing::instrument(name = "Updating Greens Functions", skip_all)]
    pub(crate) fn update_greens_functions<Conn, Spectral>(
        &mut self,
        voltage: f64,
        hamiltonian: &Hamiltonian<f64>,
        self_energy: &SelfEnergy<f64, GeometryDim, Conn>,
        spectral_space: &Spectral,
    ) -> Result<(), GreensFunctionError>
    where
        Conn: Connectivity<f64, GeometryDim> + Send + Sync,
        Spectral: SpectralDiscretisation<f64>,
        DefaultAllocator: Allocator<f64, GeometryDim>,
        <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
    {
        // In the coherent transport case we only need the retarded and lesser Greens functions (see Lake 1997)
        self.update_aggregate_retarded_greens_function(hamiltonian, self_energy, spectral_space)?;
        self.update_aggregate_lesser_greens_function(
            voltage,
            hamiltonian,
            self_energy,
            spectral_space,
        )?;

        if self.security_checks {
            self.check_carrier_conservation(voltage, hamiltonian, self_energy, spectral_space)?;
        }
        Ok(())
    }

    pub(crate) fn update_aggregate_retarded_greens_function<Conn, Spectral>(
        &mut self,
        hamiltonian: &Hamiltonian<f64>,
        self_energy: &SelfEnergy<f64, GeometryDim, Conn>,
        spectral_space: &Spectral,
    ) -> Result<(), GreensFunctionError>
    where
        Conn: Connectivity<f64, GeometryDim> + Send + Sync,
        Spectral: SpectralDiscretisation<f64>,
        DefaultAllocator: Allocator<f64, GeometryDim>,
    {
        // let term = console::Term::stdout();
        // term.move_cursor_to(0, 7).unwrap();
        // term.clear_to_end_of_screen().unwrap();
        tracing::info!("Calculating retarded Green's functions");

        // Display
        // let spinner_style = ProgressStyle::default_spinner()
        //     .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
        //     .template(
        //         "{prefix:.bold.dim} {spinner} {msg} [{wide_bar:.cyan/blue}] {percent}% ({eta})",
        //     );
        // let pb = ProgressBar::with_draw_target(
        //     (spectral_space.number_of_energy_points()
        //         * spectral_space.number_of_wavevector_points()) as u64,
        //     ProgressDrawTarget::term(term, 60),
        // );
        // pb.set_style(spinner_style);

        let n_energies = spectral_space.number_of_energy_points();
        self.retarded
            .par_iter_mut()
            .enumerate()
            // .progress_with(pb)
            // .try_for_each(|(index, (wavevector, energy))| {
            .try_for_each(|(index, gf)| {
                let energy = spectral_space.energy_at(index % n_energies);
                let wavevector = spectral_space.wavevector_at(index / n_energies);
                gf.as_mut().generate_retarded_into(
                    energy,
                    wavevector,
                    hamiltonian,
                    &MMatrix {
                        source_diagonal: vec![self_energy.contact_retarded[index].data()[0]],
                        drain_diagonal: vec![self_energy.contact_retarded[index].data()[1]],
                        core_matrix: self_energy.incoherent_retarded.as_deref().unwrap()[index]
                            .clone(),
                    },
                )
            })?;
        Ok(())
    }

    pub(crate) fn update_aggregate_lesser_greens_function<Conn, Spectral>(
        &mut self,
        voltage: f64,
        hamiltonian: &Hamiltonian<f64>,
        self_energy: &SelfEnergy<f64, GeometryDim, Conn>,
        spectral_space: &Spectral,
    ) -> Result<(), GreensFunctionError>
    where
        Conn: Connectivity<f64, GeometryDim> + Send + Sync,
        Spectral: SpectralDiscretisation<f64>,
        DefaultAllocator: Allocator<f64, GeometryDim>,
        <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
    {
        // let term = console::Term::stdout();
        // term.move_cursor_to(0, 7).unwrap();
        // term.clear_to_end_of_screen().unwrap();
        tracing::info!("Calculating lesser Green's functions");

        // Display
        // let spinner_style = ProgressStyle::default_spinner()
        //     .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
        //     .template(
        //         "{prefix:.bold.dim} {spinner} {msg} [{wide_bar:.cyan/blue}] {percent}% ({eta})",
        //     );
        // let pb = ProgressBar::with_draw_target(
        //     (spectral_space.number_of_energy_points()
        //         * spectral_space.number_of_wavevector_points()) as u64,
        //     ProgressDrawTarget::term(term, 60),
        // );
        // pb.set_style(spinner_style);

        let n_energies = spectral_space.number_of_energy_points();

        self.lesser
            .par_iter_mut()
            .enumerate()
            // .progress_with(pb)
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
                    .data_mut()
                    .iter_mut()
                    .for_each(|val| *val = Complex::new(0_f64, 1_f64) * (*val - val.conj()));
                contact_lesser.data_mut()[0] *= Complex::new(0_f64, source);
                contact_lesser.data_mut()[1] *= Complex::new(0_f64, drain);

                // Security check (anti-hermitian self energy)
                for contact_lesser_se in contact_lesser.data().iter() {
                    let sum = contact_lesser_se + contact_lesser_se.conj();
                    approx::assert_relative_eq!(sum.re, 0_f64, epsilon = std::f64::EPSILON);
                    approx::assert_relative_eq!(sum.im, 0_f64, epsilon = std::f64::EPSILON);
                }

                let internal_lesser = compute_internal_lesser_self_energies(
                    energy,
                    wavevector,
                    hamiltonian,
                    [
                        self_energy.contact_retarded[index].data()[0],
                        self_energy.contact_retarded[index].data()[1],
                    ],
                    self.retarded[0].matrix.drain_diagonal.len(),
                    [source, drain],
                )?;

                // Security check (anti-hermitian self energy)
                for internal_lesser_se in internal_lesser.iter() {
                    let sum = internal_lesser_se + internal_lesser_se.conj();
                    approx::assert_relative_eq!(sum.re, 0_f64, epsilon = std::f64::EPSILON);
                    approx::assert_relative_eq!(sum.im, 0_f64, epsilon = std::f64::EPSILON);
                }

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
                        source_diagonal: vec![contact_lesser.data()[0]],
                        drain_diagonal: vec![contact_lesser.data()[1]],
                        core_matrix: se_lesser_core.clone(),
                    },
                    &[source, drain],
                )
            })?;

        Ok(())
    }

    pub(crate) fn check_carrier_conservation<Conn, Spectral>(
        &self,
        voltage: f64,
        hamiltonian: &Hamiltonian<f64>,
        self_energy: &SelfEnergy<f64, GeometryDim, Conn>,
        spectral_space: &Spectral,
    ) -> Result<(), GreensFunctionError>
    where
        Conn: Connectivity<f64, GeometryDim> + Send + Sync,
        Spectral: SpectralDiscretisation<f64>,
        DefaultAllocator: Allocator<f64, GeometryDim>,
        <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
    {
        // let term = console::Term::stdout();
        // term.move_cursor_to(0, 7).unwrap();
        // term.clear_to_end_of_screen().unwrap();
        tracing::info!("Calculating lesser Green's functions");

        // Display
        // let spinner_style = ProgressStyle::default_spinner()
        //     .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
        //     .template(
        //         "{prefix:.bold.dim} {spinner} {msg} [{wide_bar:.cyan/blue}] {percent}% ({eta})",
        //     );
        // let pb = ProgressBar::with_draw_target(
        //     (spectral_space.number_of_energy_points()
        //         * spectral_space.number_of_wavevector_points()) as u64,
        //     ProgressDrawTarget::term(term, 60),
        // );
        // pb.set_style(spinner_style);

        let n_energies = spectral_space.number_of_energy_points();

        let nrows = self.lesser[0].as_ref().core_as_ref().nrows();
        let acc: Array1<Complex<f64>> = Array1::zeros(nrows);

        let energy_weights = spectral_space.iter_energy_weights().collect::<Vec<_>>();
        let wavevector_weights = spectral_space.iter_wavevector_weights().collect::<Vec<_>>();
        let mut energy_widths = spectral_space.iter_energy_widths().collect::<Vec<_>>();
        energy_widths.push(energy_widths[energy_widths.len() - 1]);
        let mut wavevector_widths = spectral_space.iter_wavevector_widths().collect::<Vec<_>>();
        wavevector_widths.push(wavevector_widths[wavevector_widths.len() - 1]);

        let res = self
            .lesser
            .par_iter()
            .enumerate()
            .try_fold(
                || acc.clone(),
                |acc, (index, gf)| {
                    let e_idx = index % n_energies;
                    let k_idx = index / n_energies;
                    let energy = spectral_space.energy_at(e_idx);
                    let wavevector = spectral_space.wavevector_at(k_idx);

                    let energy_weight = energy_weights[e_idx];
                    let wavevector_weight = wavevector_weights[k_idx];
                    let energy_width = energy_widths[e_idx];
                    let wavevector_width = wavevector_widths[k_idx];

                    let (source, drain) = (
                        self.info_desk.get_fermi_function_at_source(energy),
                        self.info_desk.get_fermi_function_at_drain(energy, voltage),
                    );
                    let lesser = gf.as_ref().core_as_ref().clone();
                    let retarded = self.retarded[index].as_ref().core_as_ref();
                    // Find the greater Greens function
                    let advanced = retarded.t().mapv(|x| x.conj());
                    let greater = retarded - advanced + &lesser;

                    // Find the lesser self energy
                    let internal_lesser = compute_internal_lesser_self_energies(
                        energy,
                        wavevector,
                        hamiltonian,
                        [
                            self_energy.contact_retarded[index].data()[0],
                            self_energy.contact_retarded[index].data()[1],
                        ],
                        self.retarded[0].matrix.drain_diagonal.len(),
                        [source, drain],
                    )?;

                    let nrows = self_energy.incoherent_lesser.as_deref().unwrap()[index].nrows();
                    let mut se_lesser_core =
                        self_energy.incoherent_lesser.as_deref().unwrap()[index].clone();
                    se_lesser_core[(0, 0)] += internal_lesser[0];
                    se_lesser_core[(nrows - 1, nrows - 1)] += internal_lesser[1];

                    // Find the greater self energy
                    let _contact_retarded = self_energy.contact_retarded[index].clone();
                    let internal_retarded = compute_internal_retarded_self_energies(
                        energy,
                        wavevector,
                        hamiltonian,
                        [
                            self_energy.contact_retarded[index].data()[0],
                            self_energy.contact_retarded[index].data()[1],
                        ],
                        self.retarded[0].matrix.drain_diagonal.len(),
                    )?;
                    let mut se_retarded_core =
                        self_energy.incoherent_retarded.as_deref().unwrap()[index].clone();
                    se_retarded_core[(0, 0)] += internal_retarded[0];
                    se_retarded_core[(nrows - 1, nrows - 1)] += internal_retarded[1];

                    let se_advanced_core = se_retarded_core.t().mapv(|x| x.conj());
                    let se_greater_core = se_retarded_core - se_advanced_core + &se_lesser_core;

                    // Sigma^< G^> * dE * k * dk
                    let outflow = se_lesser_core.dot(&greater).diag().to_owned()
                        * energy_width
                        * energy_weight
                        * wavevector_width
                        * wavevector_weight
                        * wavevector;
                    // Sigma^> G^< * dE * k * dk
                    let inflow = se_greater_core.dot(&lesser).diag().to_owned()
                        * energy_width
                        * energy_weight
                        * wavevector_width
                        * wavevector_weight
                        * wavevector;

                    let res = acc + (outflow - inflow);

                    if true {
                        Ok(res)
                    } else {
                        Err(anyhow::anyhow!("unreachable!()"))
                    }
                },
            )
            .try_reduce(|| acc.clone(), |acc, x| Ok(acc + x))?;

        let norm = res
            .iter()
            .fold(0_f64, |acc, x| acc + x.norm().powi(2))
            .sqrt();
        let sum_norm = res.sum().norm();

        if sum_norm / norm < 1e-3 {
            Ok(())
        } else {
            Err(crate::greens_functions::SecurityCheck {
                calculation: "greens function update, conservation violated".into(),
                index: 0,
            })
        }?;

        Ok(())
    }
}

impl GreensFunctionMethods<f64> for MMatrix<Complex<f64>> {
    type SelfEnergy = MMatrix<Complex<f64>>;

    fn generate_retarded_into(
        &mut self,
        energy: f64,
        wavevector: f64,
        hamiltonian: &Hamiltonian<f64>,
        self_energy: &Self::SelfEnergy,
    ) -> Result<(), GreensFunctionError> {
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
        let left_internal_self_energy = g_00[g_00.shape()[0] - 1]
            * hamiltonian
                .outer_view(number_of_vertices_in_reservoir)
                .unwrap()
                .data()[2]
                .powi(2);

        let g_ll = right_connected_diagonal(
            energy,
            &hamiltonian,
            &self_energies_at_external_contacts,
            number_of_vertices_in_reservoir,
            number_of_vertices_in_reservoir,
        )?;
        let right_internal_self_energy = g_ll[0]
            * hamiltonian
                .outer_view(hamiltonian.rows() - 1 - number_of_vertices_in_reservoir)
                .unwrap()
                .data()[2]
                .powi(2);

        // TODO Casting to Complex here is verbose and wasteful, can we try not to do this?
        // Maybe the Hamiltonian needs to be made in terms of `ComplexField`?
        let data = hamiltonian.data();
        let mut y = Vec::with_capacity(data.len());
        for x in data {
            y.push(Complex::from(*x));
        }
        let mut dense_hamiltonian = sprs::CsMat::new(
            hamiltonian.shape(),
            hamiltonian.indptr().raw_storage().to_vec(),
            hamiltonian.indices().to_vec(),
            y,
        )
        .to_dense();

        dense_hamiltonian[(
            number_of_vertices_in_reservoir,
            number_of_vertices_in_reservoir,
        )] += left_internal_self_energy;
        dense_hamiltonian[(
            number_of_vertices_in_reservoir + number_of_vertices_in_core - 1,
            number_of_vertices_in_reservoir + number_of_vertices_in_core - 1,
        )] += right_internal_self_energy;
        let dense_hamiltonian = dense_hamiltonian.slice(s![
            number_of_vertices_in_reservoir
                ..number_of_vertices_in_reservoir + number_of_vertices_in_core,
            number_of_vertices_in_reservoir
                ..number_of_vertices_in_reservoir + number_of_vertices_in_core
        ]);

        let matrix = Array2::from_diag_elem(number_of_vertices_in_core, Complex::from(energy))
            - dense_hamiltonian
            - &self_energy.core_matrix;

        let matrix = ndarray_linalg::solve::Inverse::inv(&matrix);
        if let Ok(matrix) = matrix {
            self.core_matrix = matrix;
        } else {
            return Err(GreensFunctionError::Inversion);
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
        let mut previous_hopping_element = Complex::from(
            hamiltonian
                .outer_view(number_of_vertices_in_reservoir + number_of_vertices_in_core - 1)
                .unwrap()
                .data()[2],
        );
        self.drain_diagonal
            .iter_mut()
            .zip(
                hamiltonian
                    .outer_iterator()
                    .zip(right_diagonal.into_iter())
                    .skip(number_of_vertices_in_reservoir + number_of_vertices_in_core - 1),
            )
            .for_each(|(element, (hamiltonian_row, right_diagonal_element))| {
                let hopping_element = if hamiltonian_row.data().len() == 3 {
                    Complex::from(hamiltonian_row.data()[2])
                } else {
                    Complex::from(hamiltonian_row.data()[0])
                };
                *element = right_diagonal_element
                    * (Complex::from(1_f64)
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
            hamiltonian.rows(),
            number_of_vertices_in_reservoir,
        )?;
        previous = self.core_matrix[(0, 0)];
        previous_hopping_element = Complex::from(
            hamiltonian
                .outer_view(number_of_vertices_in_reservoir)
                .unwrap()
                .data()[2],
        );
        self.source_diagonal
            .iter_mut()
            .zip(left_diagonal.iter().take(number_of_vertices_in_reservoir))
            .rev()
            .enumerate()
            .for_each(|(idx, (element, left_diagonal_element))| {
                let row = hamiltonian
                    .outer_view(number_of_vertices_in_reservoir - 1 - idx)
                    .unwrap();
                let hopping_element = if row.data().len() == 3 {
                    Complex::from(row.data()[0])
                } else {
                    Complex::from(row.data()[1])
                };
                *element = left_diagonal_element
                    * (Complex::from(1_f64)
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
        _retarded: &MMatrix<Complex<f64>>,
    ) -> Result<(), GreensFunctionError> {
        unimplemented!()
    }

    fn generate_lesser_into(
        &mut self,
        _energy: f64,
        _wavevector: f64,
        _hamiltonian: &Hamiltonian<f64>,
        retarded_greens_function: &MMatrix<Complex<f64>>,
        lesser_self_energy: &MMatrix<Complex<f64>>,
        fermi_functions: &[f64],
    ) -> Result<(), GreensFunctionError> {
        // Expensive matrix inversion
        let advanced = retarded_greens_function
            .core_matrix
            .view()
            .t()
            .mapv(|x| x.conj());
        self.core_matrix
            .iter_mut()
            .zip(
                (&retarded_greens_function
                    .core_matrix
                    .dot(&lesser_self_energy.core_matrix.dot(&advanced)))
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
                let spectral_density = Complex::new(0_f64, 1_f64) * (g_r - g_r.conj());
                *element = Complex::new(0_f64, fermi_functions[0]) * spectral_density;
            });
        self.drain_diagonal
            .iter_mut()
            .zip(retarded_greens_function.drain_diagonal.iter())
            .for_each(|(element, g_r)| {
                let spectral_density = Complex::new(0_f64, 1_f64) * (g_r - g_r.conj());
                *element = Complex::new(0_f64, fermi_functions[1]) * spectral_density;
            });

        // Security check, it should be the case that G^< = - [G^<]^{\dag}
        let norm = self
            .source_diagonal
            .iter()
            .chain(self.drain_diagonal.iter())
            .fold(Complex::from(0_f64), |acc, x| acc + (x + x.conj()).abs());

        approx::assert_relative_eq!(norm.re, 0_f64, epsilon = std::f64::EPSILON * 100_f64);
        approx::assert_relative_eq!(norm.im, 0_f64, epsilon = std::f64::EPSILON * 100_f64);
        assert!(crate::utilities::matrices::is_anti_hermitian(
            self.core_as_ref().view()
        ));

        Ok(())
    }
}

fn compute_internal_lesser_self_energies(
    energy: f64,
    wavevector: f64,
    hamiltonian: &Hamiltonian<f64>,
    edge_retarded_self_energies: [Complex<f64>; 2],
    number_of_vertices_in_reservoir: usize,
    fermi_functions: [f64; 2],
) -> Result<[Complex<f64>; 2], GreensFunctionError> {
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
        .map(|(s_re, f)| Complex::new(0_f64, f) * Complex::new(0_f64, 1_f64) * (s_re - s_re.conj()))
        .collect::<Vec<_>>();
    Ok([lesser_se[0], lesser_se[1]])
}

fn compute_internal_retarded_self_energies(
    energy: f64,
    wavevector: f64,
    hamiltonian: &Hamiltonian<f64>,
    edge_retarded_self_energies: [Complex<f64>; 2],
    number_of_vertices_in_reservoir: usize,
) -> Result<[Complex<f64>; 2], GreensFunctionError> {
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
    let left_internal_self_energy = g_00[g_00.shape()[0] - 1]
        * hamiltonian
            .outer_view(number_of_vertices_in_reservoir)
            .unwrap()
            .data()[2]
            .powi(2);
    let g_ll = right_connected_diagonal(
        energy,
        &hamiltonian,
        &self_energies_at_external_contacts,
        number_of_vertices_in_reservoir,
        number_of_vertices_in_reservoir,
    )?;
    let right_internal_self_energy = g_ll[0]
        * hamiltonian
            .outer_view(hamiltonian.rows() - 1 - number_of_vertices_in_reservoir)
            .unwrap()
            .data()[2]
            .powi(2);
    Ok([left_internal_self_energy, right_internal_self_energy])
}

/// Implementation of the accumulator methods for a sparse AggregateGreensFunction
impl<BandDim, GeometryDim, Conn, Spectral>
    AggregateGreensFunctionMethods<
        f64,
        BandDim,
        GeometryDim,
        Conn,
        Spectral,
        SelfEnergy<f64, GeometryDim, Conn>,
    > for AggregateGreensFunctions<'_, f64, MMatrix<Complex<f64>>, GeometryDim, BandDim>
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim>,
    Spectral: SpectralDiscretisation<f64>,
    DefaultAllocator: Allocator<Array1<f64>, BandDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>
        + Allocator<f64, GeometryDim>,
{
    fn accumulate_into_charge_density_vector(
        &self,
        mesh: &Mesh<f64, GeometryDim, Conn>,
        spectral_space: &Spectral,
    ) -> Result<Charge<f64, BandDim>, crate::postprocessor::PostProcessorError> {
        let mut charges: Vec<Array1<f64>> = Vec::with_capacity(BandDim::dim());
        // Sum over the diagonal of the calculated spectral density
        // TODO Only one band
        let mut summed_diagonal = vec![
            0_f64;
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
                1_f64
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
                    Array1::zeros(summed_diagonal.len()),
                    |sum: Array1<Complex<f64>>, ((value, weight), width)| {
                        let matrix = &value.matrix;
                        let diagonal = Array1::from(
                            matrix
                                .source_diagonal
                                .iter()
                                .chain(matrix.core_matrix.diag().iter())
                                .chain(matrix.drain_diagonal.iter())
                                .copied()
                                .collect::<Vec<_>>(),
                        )
                        .mapv(|x| {
                            x * Complex::from(
                                weight * width // Weighted by the integration weight from the `SpectralSpace` and the diameter of the element in the grid
                            / crate::constants::ELECTRON_CHARGE, // The Green's function is an inverse energy stored in eV
                            )
                        });
                        sum + diagonal
                    },
                );
            summed_diagonal
                .iter_mut()
                .zip(
                    new_diagonal
                        .iter()
                        .map(|&x| -(Complex::new(0_f64, 1_f64) * x).re),
                )
                .for_each(|(ele, new)| *ele += wavevector * weight * width * new);
        }

        // Separate out the diagonals for each `BandDim` into their own charge vector
        for band_number in 0..BandDim::dim() {
            charges.push(Array1::from(
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
                .fold(0_f64, |acc, idx| {
                    acc + mesh.elements()[idx].0.diameter() / 2_f64
                });

                let prefactor = match spectral_space.number_of_wavevector_points() {
                    1 => {
                        crate::constants::BOLTZMANN
                            * crate::constants::ELECTRON_CHARGE
                            * crate::constants::ELECTRON_MASS
                            / 2.
                            / std::f64::consts::PI.powi(2)
                            / crate::constants::HBAR.powi(2)
                            * self.info_desk.temperature
                            * self
                                .info_desk
                                .effective_mass_from_assignment(assignment, n_band, 1)
                            / diameter
                    }
                    _ => {
                        crate::constants::ELECTRON_CHARGE
                            / 2_f64
                            / std::f64::consts::PI.powi(2)
                            / diameter
                    }
                };
                *charge_at_element *= prefactor;
            }
        }

        Charge::new(OVector::<Array1<f64>, BandDim>::from_iterator(
            charges.into_iter(),
        ))
    }

    fn accumulate_into_current_density_vector(
        &self,
        voltage: f64,
        mesh: &Mesh<f64, GeometryDim, Conn>,
        self_energy: &SelfEnergy<f64, GeometryDim, Conn>,
        spectral_space: &Spectral,
    ) -> Result<Current<f64, BandDim>, crate::postprocessor::PostProcessorError> {
        let mut currents: Vec<Array1<f64>> = Vec::with_capacity(BandDim::dim());
        let _number_of_vertices_in_internal_lead = self.retarded[0].as_ref().source_diagonal.len();

        let summed_current = self
            .retarded
            .iter()
            .zip(self_energy.contact_retarded.iter())
            .zip(spectral_space.iter_energy_weights())
            .zip(spectral_space.iter_energy_widths())
            .zip(spectral_space.iter_energies())
            .fold(
                vec![0_f64; mesh.elements().len() * BandDim::dim()],
                |sum, ((((gf_r, se_r), weight), width), energy)| {
                    // Gamma = i (\Sigma_r - \Sigma_a) -> in coherent transport \Sigma has only two non-zero elements
                    let gamma_source = -2_f64 * se_r.data()[0].im;
                    let gamma_drain = -2_f64 * se_r.data()[1].im;
                    // Zero order Fermi integrals
                    let fermi_source = self.info_desk.get_fermi_integral_at_source(energy);
                    let fermi_drain = self.info_desk.get_fermi_integral_at_drain(energy, voltage);

                    // Get the element in row 1, column N of the dense Green's function matrix
                    let gf_r_1n =
                        gf_r.as_ref().core_matrix[(0, gf_r.as_ref().core_matrix.nrows() - 1)];

                    let abs_gf_r_1n_with_factor = (gf_r_1n * gf_r_1n.conj()).re
                        * width
                        * weight
                        * gamma_source
                        * gamma_drain
                        * (fermi_source - fermi_drain)
                        * 0.01_f64.powi(2)
                        / 1e5; // Convert to x 10^5 A / cm^2

                    sum.into_iter()
                        .map(|sum| sum + abs_gf_r_1n_with_factor)
                        .collect()
                },
            );
        // Separate out the diagonals for each `BandDim` into their own charge vector
        for band_number in 0..BandDim::dim() {
            currents.push(Array1::from(
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
                let prefactor = crate::constants::BOLTZMANN
                    * crate::constants::ELECTRON_CHARGE.powi(2)
                    * crate::constants::ELECTRON_MASS
                    / 2.
                    / std::f64::consts::PI.powi(2)
                    / crate::constants::HBAR.powi(3)
                    * self.info_desk.temperature
                    * self.info_desk.effective_masses[region][n_band][1];
                *current_at_element *= prefactor;
            }
        }
        Current::new(OVector::<Array1<f64>, BandDim>::from_iterator(
            currents.into_iter(),
        ))
    }
}
