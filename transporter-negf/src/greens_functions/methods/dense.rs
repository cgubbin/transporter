//! Dense implementations for single and aggregated Greens functions
use super::super::{GreensFunctionError, GreensFunctionMethods};
use crate::hamiltonian::Hamiltonian;
use nalgebra::ComplexField;
use ndarray::Array2;
use num_complex::Complex;
use sprs::CsMat;

use super::{
    super::GreensFunctionInfoDesk,
    aggregate::{AggregateGreensFunctionMethods, AggregateGreensFunctions},
};
use crate::{
    postprocessor::{Charge, Current},
    self_energy::SelfEnergy,
    spectral::SpectralDiscretisation,
};
use console::Term;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressDrawTarget, ProgressStyle};
use nalgebra::{allocator::Allocator, DefaultAllocator, OVector};
use ndarray::Array1;
use rayon::prelude::*;
use transporter_mesher::{Connectivity, ElementMethods, Mesh, SmallDim};

impl GreensFunctionMethods<f64> for Array2<Complex<f64>> {
    type SelfEnergy = Array2<Complex<f64>>;

    fn generate_retarded_into(
        &mut self,
        energy: f64,
        wavevector: f64,
        hamiltonian: &Hamiltonian<f64>,
        retarded_self_energy: &Self::SelfEnergy,
    ) -> Result<(), GreensFunctionError> {
        let output = self.as_slice_mut().unwrap();

        // do a slow matrix inversion
        let num_rows = hamiltonian.num_rows();

        let hamiltonian = hamiltonian.calculate_total(wavevector);
        // TODO Casting to Complex here is verbose and wasteful, can we try not to do this?
        // Maybe the Hamiltonian needs to be made in terms of `ComplexField`?
        let values = hamiltonian.data();
        let mut y = Vec::with_capacity(values.len());
        for value in values {
            y.push(Complex::from(*value));
        }
        let dense_hamiltonian = CsMat::new(
            hamiltonian.shape(),
            hamiltonian.indptr().into_raw_storage().to_vec(),
            hamiltonian.indices().to_vec(),
            y,
        )
        .to_dense();
        // Avoid allocation: https://github.com/InteractiveComputerGraphics/fenris/blob/e4161887669acb366cad312cfa68d106e6cf576c/src/assembly/operators.rs
        // Look at lines 164-172
        let matrix = Array2::from_diag_elem(num_rows, Complex::from(energy))
            - &dense_hamiltonian
            - retarded_self_energy; //TODO Do we have to convert? Seems dumb. Should we store H in dense form too?&ham;

        let matrix = ndarray_linalg::solve::Inverse::inv(&matrix);
        if let Ok(matrix) = matrix {
            output.copy_from_slice(matrix.as_slice().unwrap());
            return Ok(());
        }
        Err(GreensFunctionError::Inversion)
    }

    fn generate_greater_into(
        &mut self,
        lesser: &Array2<Complex<f64>>,
        retarded: &Array2<Complex<f64>>,
        advanced: &Array2<Complex<f64>>,
    ) -> Result<(), GreensFunctionError> {
        let output = self.as_slice_mut().unwrap();
        output.copy_from_slice((retarded - advanced + lesser).as_slice().unwrap());
        Ok(())
    }

    fn generate_advanced_into(
        &mut self,
        retarded: &Array2<Complex<f64>>,
    ) -> Result<(), GreensFunctionError> {
        let output = self.as_slice_mut().unwrap();
        let advanced = retarded.t().mapv(|x| x.conj());
        output.copy_from_slice(advanced.as_slice().unwrap());
        Ok(())
    }

    fn generate_lesser_into(
        &mut self,
        _energy: f64,
        _wavevector: f64,
        _hamiltonian: &Hamiltonian<f64>,
        retarded_greens_function: &Array2<Complex<f64>>,
        lesser_self_energy: &Self::SelfEnergy,
        _fermi_functions: &[f64],
    ) -> Result<(), GreensFunctionError> {
        let advanced = retarded_greens_function.t().mapv(|x| x.conj());
        self.iter_mut()
            .zip((retarded_greens_function * lesser_self_energy * advanced).iter())
            .for_each(|(element, &value)| {
                *element = value;
            });

        // Security check, it should be the case that G^< = - [G^<]^{\dag}
        let norm = self
            .iter()
            .zip(self.t().iter())
            .fold(Complex::from(0_f64), |acc, (x, y)| acc + x + y.conj());
        // Handle the error
        approx::assert_relative_eq!(norm.re, 0_f64);
        approx::assert_relative_eq!(norm.im, 0_f64);

        Ok(())
    }
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
    > for AggregateGreensFunctions<'_, f64, Array2<Complex<f64>>, GeometryDim, BandDim>
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
        let n_ele = self.lesser[0].matrix.shape()[0];
        let mut summed_diagonal = vec![0_f64; n_ele];

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
                    Array1::zeros(n_ele),
                    |sum: Array1<Complex<f64>>, ((value, weight), width)| {
                        let diag = value.matrix.diag().mapv(|x| {
                            x * Complex::from(
                                weight * width // Weighted by the integration weight from the `SpectralSpace` and the diameter of the element in the grid
                            / crate::constants::ELECTRON_CHARGE, // The Green's function is an inverse energy stored in eV
                            )
                        });
                        sum + diag
                    },
                );
            summed_diagonal
                .iter_mut()
                .zip(new_diagonal.into_iter().map(|x| x.real()))
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

                    let values = gf_r.matrix.row(0);
                    values
                        .map(|value| (value * value.conj()).re)
                        .map(|grga| {
                            grga * weight
                                * width
                                * weight
                                * gamma_source
                                * gamma_drain
                                * (fermi_source - fermi_drain)
                                / crate::constants::ELECTRON_CHARGE
                        })
                        .into_iter()
                        .zip(sum.into_iter())
                        .map(|(grga, sum)| sum + grga)
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

impl<'a, GeometryDim, BandDim>
    AggregateGreensFunctions<'a, f64, Array2<Complex<f64>>, GeometryDim, BandDim>
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
                    &(&self_energy.contact_retarded[index].to_dense()
                        + &self_energy.incoherent_retarded.as_deref().unwrap()[index]),
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
        let n_ele = self.lesser[0].as_ref().shape()[0];

        self.lesser
            .par_iter_mut()
            .enumerate()
            .progress_with(pb)
            // .try_for_each(|(index, (wavevector, energy))| {
            .try_for_each(|(index, gf)| {
                let energy = spectral_space.energy_at(index % n_energies);
                let (source, drain) = (
                    self.info_desk.get_fermi_function_at_source(energy),
                    self.info_desk.get_fermi_function_at_drain(energy, voltage),
                );
                let contact_retarded = self_energy.contact_retarded[index].to_dense();
                let mut contact_lesser =
                    &contact_retarded - &contact_retarded.mapv(|x| x.conj()).t();
                contact_lesser[(0, 0)] *= Complex::new(0_f64, source);
                contact_lesser[(n_ele - 1, n_ele - 1)] *= Complex::new(0_f64, drain);

                let energy = spectral_space.energy_at(index % n_energies);
                let wavevector = spectral_space.wavevector_at(index / n_energies);

                gf.as_mut().generate_lesser_into(
                    energy,
                    wavevector,
                    hamiltonian,
                    &self.retarded[index].matrix,
                    &(&contact_lesser + &self_energy.incoherent_lesser.as_deref().unwrap()[index]),
                    &[source, drain],
                )
            })?;
        Ok(())
    }
}
