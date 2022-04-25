//! Dense implementations for single and aggregated Greens functions
use super::GreensFunctionMethods;
use crate::hamiltonian::Hamiltonian;
use nalgebra::{ComplexField, DMatrix, RealField};
use nalgebra_sparse::CsrMatrix;
use num_complex::Complex;

use super::{
    aggregate::{AggregateGreensFunctionMethods, AggregateGreensFunctions},
    GreensFunctionInfoDesk,
};
use crate::{
    postprocessor::{Charge, Current},
    self_energy::SelfEnergy,
    spectral::SpectralDiscretisation,
};
use console::Term;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressDrawTarget, ProgressStyle};
use nalgebra::{
    allocator::Allocator, Const, DVector, DefaultAllocator, Dynamic, Matrix, OVector, VecStorage,
};
use rayon::prelude::*;
use transporter_mesher::{Connectivity, ElementMethods, Mesh, SmallDim};

impl<T> GreensFunctionMethods<T> for DMatrix<Complex<T>>
where
    T: RealField + Copy,
{
    type SelfEnergy = DMatrix<Complex<T>>;

    fn generate_retarded_into(
        &mut self,
        energy: T,
        wavevector: T,
        hamiltonian: &Hamiltonian<T>,
        retarded_self_energy: &Self::SelfEnergy,
    ) -> color_eyre::Result<()> {
        let mut output: nalgebra::DMatrixSliceMut<Complex<T>> = self.into();

        // do a slow matrix inversion
        let num_rows = hamiltonian.num_rows();

        let hamiltonian = hamiltonian.calculate_total(wavevector);
        // TODO Casting to Complex here is verbose and wasteful, can we try not to do this?
        // Maybe the Hamiltonian needs to be made in terms of `ComplexField`?
        let values = hamiltonian.values();
        let mut y = Vec::with_capacity(values.len());
        for value in values {
            y.push(Complex::from(*value));
        }
        let hamiltonian =
            CsrMatrix::try_from_pattern_and_values(hamiltonian.pattern().clone(), y).unwrap();
        // Avoid allocation: https://github.com/InteractiveComputerGraphics/fenris/blob/e4161887669acb366cad312cfa68d106e6cf576c/src/assembly/operators.rs
        // Look at lines 164-172
        let mut matrix = DMatrix::identity(num_rows, num_rows) * Complex::from(energy)
            - nalgebra_sparse::convert::serial::convert_csr_dense(&hamiltonian)
            - retarded_self_energy; //TODO Do we have to convert? Seems dumb. Should we store H in dense form too?&ham;

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
        _energy: T,
        _wavevector: T,
        _hamiltonian: &Hamiltonian<T>,
        retarded_greens_function: &DMatrix<Complex<T>>,
        lesser_self_energy: &Self::SelfEnergy,
        _fermi_functions: &[T],
    ) -> color_eyre::Result<()> {
        self.iter_mut()
            .zip(
                (retarded_greens_function
                    * lesser_self_energy
                    * retarded_greens_function.transpose().conjugate())
                .iter(),
            )
            .for_each(|(element, &value)| {
                *element = value;
            });
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
    > for AggregateGreensFunctions<'_, T, DMatrix<Complex<T>>, GeometryDim, BandDim>
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
        let n_ele = self.lesser[0].matrix.shape().0;
        let mut summed_diagonal = vec![T::zero(); n_ele];

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
                    nalgebra::DVector::zeros(n_ele),
                    |sum, ((value, weight), width)| {
                        sum + &value.matrix.diagonal()
                            * Complex::from(
                                weight * width // Weighted by the integration weight from the `SpectralSpace` and the diameter of the element in the grid
                                / T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap(), // The Green's function is an inverse energy stored in eV
                            )
                    },
                );
            summed_diagonal
                .iter_mut()
                .zip(new_diagonal.into_iter().map(|&x| x.real()))
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
                                / T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
                        })
                        .into_iter()
                        .zip(sum.into_iter())
                        .map(|(&grga, sum)| sum + grga)
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

impl<'a, T, GeometryDim, BandDim>
    AggregateGreensFunctions<'a, T, DMatrix<Complex<T>>, GeometryDim, BandDim>
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
                    &(&nalgebra_sparse::convert::serial::convert_csr_dense(
                        &self_energy.contact_retarded[index],
                    ) + &self_energy.incoherent_retarded.as_deref().unwrap()[index]),
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
        let n_ele = self.lesser[0].as_ref().shape().0;

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
                let contact_lesser = nalgebra_sparse::convert::serial::convert_csr_dense(
                    &self_energy.contact_retarded[index],
                );
                let mut contact_lesser = &contact_lesser - contact_lesser.transpose().conjugate();
                contact_lesser[(0, 0)] *= Complex::new(T::zero(), source);
                contact_lesser[(n_ele - 1, n_ele - 1)] *= Complex::new(T::zero(), drain);

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
