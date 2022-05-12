use super::{SelfEnergy, SelfEnergyError};
use crate::constants::{ELECTRON_CHARGE, EPSILON_0};
use crate::greens_functions::{mixed::MMatrix, AggregateGreensFunctions};
use crate::spectral::SpectralDiscretisation;
use console::Term;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressDrawTarget, ProgressStyle};
use nalgebra::{allocator::Allocator, DMatrix, DefaultAllocator, RealField};
use num_complex::Complex;
use rayon::prelude::*;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

impl<T, GeometryDim, Conn> SelfEnergy<T, GeometryDim, Conn>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim> + Send + Sync,
    <Conn as Connectivity<T, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Send + Sync,
{
    /// Updates the lesser LO Phonon scattering Self Energy at the contacts into the scratch matrix held in `self`
    ///
    ///
    pub(crate) fn recalculate_localised_lo_lesser_self_energy<BandDim: SmallDim, Spectral>(
        &mut self,
        scaling: T,
        mesh: &Mesh<T, GeometryDim, Conn>,
        spectral_space: &Spectral,
        greens_functions: &AggregateGreensFunctions<
            '_,
            T,
            DMatrix<Complex<T>>,
            GeometryDim,
            BandDim,
        >,
    ) -> Result<(), SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<T>,
        DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
        <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
    {
        tracing::info!("Calculating LO Lesser SE {}", scaling);
        let n_0 = T::from_f64(0.3481475088177923).unwrap(); // The LO phonon population, to be updated as we pass through the loop
        let e_0 = T::from_f64(0.035).unwrap(); // The phonon energy in electron volts
        let eps_fr = T::from_f64(20_f64).unwrap();
        let gamma_frohlich =
            scaling * T::from_f64(ELECTRON_CHARGE / 2_f64 / EPSILON_0).unwrap() * e_0 * eps_fr; // In electron volts

        assert!(self.incoherent_lesser.is_some());

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

        // For each LO phonon wavevector integrate over all the connected electronic wavevectors
        self.incoherent_lesser
            .as_deref_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .progress_with(pb)
            .for_each(|(index, lesser_self_energy_matrix)| {
                let energy_index = index % spectral_space.number_of_energy_points();
                let wavevector_index_k = index / spectral_space.number_of_energy_points();
                let wavevector_k = spectral_space.wavevector_at(wavevector_index_k);

                // Reset the matrix
                lesser_self_energy_matrix.fill(Complex::from(T::zero()));
                for (_wavevector_index_l, ((weight, width), wavevector_l)) in spectral_space
                    .iter_wavevector_weights()
                    .zip(spectral_space.iter_wavevector_widths())
                    .zip(spectral_space.iter_wavevectors())
                    .enumerate()
                {
                    if spectral_space.energy_at(energy_index)
                        < spectral_space.energy_at(spectral_space.number_of_energy_points() - 1)
                            - e_0
                    {
                        // Construct the outscattering term
                        let energy_scattered_from = spectral_space.energy_at(energy_index) + e_0;
                        // Find the global indices of the two points in the energy mesh which bracket `energy_scattered_from`
                        let energy_indices = spectral_space
                            .identify_bracketing_indices(energy_scattered_from)
                            .unwrap();
                        let global_indices = [
                            energy_indices[0]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                            energy_indices[1]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                        ];
                        // Get the weights linearly interpolating around `energy_scattered_from`
                        let weights = spectral_space
                            .identify_bracketing_weights(energy_scattered_from)
                            .unwrap();
                        // Form the best guess for the lesser GF at `energy_scattered_from`
                        let mut lesser_gf_scattered_from =
                            greens_functions.lesser[global_indices[0]].as_ref()
                                * Complex::from(weights[0])
                                + greens_functions.lesser[global_indices[1]].as_ref()
                                    * Complex::from(weights[1]);
                        lesser_gf_scattered_from *= Complex::from(
                            gamma_frohlich * T::from_f64(1. / 4. / std::f64::consts::PI).unwrap(),
                        );
                        for (idx, mut row) in lesser_self_energy_matrix.row_iter_mut().enumerate() {
                            for (jdx, entry) in row.iter_mut().enumerate() {
                                let prefactor = Self::bulk_lo_phonon_potential(
                                    mesh,
                                    idx,
                                    jdx,
                                    (wavevector_k - wavevector_l).abs(),
                                );
                                *entry += Complex::from(
                                    prefactor * (T::one() + n_0) * weight * width * wavevector_l,
                                ) * lesser_gf_scattered_from[(idx, jdx)];
                            }
                        }
                    }

                    if spectral_space.energy_at(energy_index) > e_0 {
                        // Construct the outscattering term
                        let energy_scattered_to = spectral_space.energy_at(energy_index) - e_0;
                        // Find the global indices of the two points in the energy mesh which bracket `energy_scattered_from`
                        let energy_indices = spectral_space
                            .identify_bracketing_indices(energy_scattered_to)
                            .unwrap();
                        let global_indices = [
                            energy_indices[0]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                            energy_indices[1]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                        ];
                        // Get the weights linearly interpolating around `energy_scattered_from`
                        let weights = spectral_space
                            .identify_bracketing_weights(energy_scattered_to)
                            .unwrap();
                        // Form the best guess for the lesser GF at `energy_scattered_from`
                        let mut lesser_gf_scattered_to = greens_functions.lesser[global_indices[0]]
                            .as_ref()
                            * Complex::from(weights[0])
                            + greens_functions.lesser[global_indices[1]].as_ref()
                                * Complex::from(weights[1]);
                        lesser_gf_scattered_to *= Complex::from(
                            gamma_frohlich * T::from_f64(1. / 4. / std::f64::consts::PI).unwrap(),
                        );

                        for (idx, mut row) in lesser_self_energy_matrix.row_iter_mut().enumerate() {
                            for (jdx, entry) in row.iter_mut().enumerate() {
                                let prefactor = Self::bulk_lo_phonon_potential(
                                    mesh,
                                    idx,
                                    jdx,
                                    (wavevector_k - wavevector_l).abs(),
                                );
                                *entry += Complex::from(prefactor * (n_0) * weight * wavevector_l)
                                    * lesser_gf_scattered_to[(idx, jdx)];
                            }
                        }
                    }
                }
            });
        Ok(())
    }

    pub(crate) fn recalculate_localised_lo_retarded_self_energy<BandDim: SmallDim, Spectral>(
        &mut self,
        scaling: T,
        mesh: &Mesh<T, GeometryDim, Conn>,
        spectral_space: &Spectral,
        greens_functions: &AggregateGreensFunctions<
            '_,
            T,
            DMatrix<Complex<T>>,
            GeometryDim,
            BandDim,
        >,
    ) -> Result<(), SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<T>,
        DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
        <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
    {
        tracing::info!("Calculating LO retarded SE {}", scaling);
        let n_0 = T::from_f64(0.3481475088177923).unwrap(); // The LO phonon population, to be updated as we pass through the loop
        let e_0 = T::from_f64(0.035).unwrap(); // The phonon energy in electron volts
        let eps_fr = T::from_f64(20_f64).unwrap();
        let gamma_frohlich =
            scaling * T::from_f64(ELECTRON_CHARGE / 2_f64 / EPSILON_0).unwrap() * e_0 * eps_fr; // In electron volts!!

        assert!(self.incoherent_retarded.is_some());

        // Display
        let term = Term::stdout();
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

        self.incoherent_retarded
            .as_deref_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .progress_with(pb)
            .for_each(|(index, retarded_self_energy_matrix)| {
                let energy_index = index % spectral_space.number_of_energy_points();
                let wavevector_index_k = index / spectral_space.number_of_energy_points();
                let wavevector_k = spectral_space.wavevector_at(wavevector_index_k);

                // Reset the matrix
                retarded_self_energy_matrix.fill(Complex::from(T::zero()));
                for (_wavevector_index_l, ((weight, width), wavevector_l)) in spectral_space
                    .iter_wavevector_weights()
                    .zip(spectral_space.iter_wavevector_widths())
                    .zip(spectral_space.iter_wavevectors())
                    .enumerate()
                {
                    if spectral_space.energy_at(energy_index)
                        < spectral_space.energy_at(spectral_space.number_of_energy_points() - 1)
                            - e_0
                    {
                        // Construct the outscattering term
                        let energy_scattered_from = spectral_space.energy_at(energy_index) + e_0;
                        // Find the global indices of the two points in the energy mesh which bracket `energy_scattered_from`
                        let energy_indices = spectral_space
                            .identify_bracketing_indices(energy_scattered_from)
                            .unwrap();
                        let global_indices = [
                            energy_indices[0]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                            energy_indices[1]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                        ];
                        // Get the weights linearly interpolating around `energy_scattered_from`
                        let weights = spectral_space
                            .identify_bracketing_weights(energy_scattered_from)
                            .unwrap();
                        // Form the best guess for the lesser GF at `energy_scattered_from` (Eq 6, no PV)
                        let mut gf_scattered_from = greens_functions.retarded[global_indices[0]]
                            .as_ref()
                            * Complex::from(weights[0] * n_0)
                            + greens_functions.retarded[global_indices[1]].as_ref()
                                * Complex::from(weights[1] * n_0)
                            - greens_functions.lesser[global_indices[0]].as_ref()
                                * Complex::from(weights[0] / (T::one() + T::one()))
                            - greens_functions.lesser[global_indices[1]].as_ref()
                                * Complex::from(weights[1] / (T::one() + T::one()));
                        gf_scattered_from *= Complex::from(
                            gamma_frohlich * T::from_f64(1. / 4. / std::f64::consts::PI).unwrap(),
                        );
                        for (idx, mut row) in retarded_self_energy_matrix.row_iter_mut().enumerate()
                        {
                            for (jdx, entry) in row.iter_mut().enumerate() {
                                let prefactor = Self::bulk_lo_phonon_potential(
                                    mesh,
                                    idx,
                                    jdx,
                                    (wavevector_k - wavevector_l).abs(),
                                );
                                *entry += Complex::from(prefactor * weight * width * wavevector_l)
                                    * gf_scattered_from[(idx, jdx)];
                            }
                        }
                    }

                    if spectral_space.energy_at(energy_index) > e_0 {
                        // Construct the outscattering term
                        let energy_scattered_to = spectral_space.energy_at(energy_index) - e_0;
                        // Find the global indices of the two points in the energy mesh which bracket `energy_scattered_from`
                        let energy_indices = spectral_space
                            .identify_bracketing_indices(energy_scattered_to)
                            .unwrap();
                        let global_indices = [
                            energy_indices[0]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                            energy_indices[1]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                        ];
                        // Get the weights linearly interpolating around `energy_scattered_from`
                        let weights = spectral_space
                            .identify_bracketing_weights(energy_scattered_to)
                            .unwrap();
                        // Form the best guess for the lesser GF at `energy_scattered_from` (Eq 6, no PV)
                        let mut gf_scattered_to = greens_functions.retarded[global_indices[0]]
                            .as_ref()
                            * Complex::from(weights[0] * (T::one() + n_0))
                            + greens_functions.retarded[global_indices[1]].as_ref()
                                * Complex::from(weights[1] * (T::one() + n_0))
                            + greens_functions.lesser[global_indices[0]].as_ref()
                                * Complex::from(weights[0] / (T::one() + T::one()))
                            + greens_functions.lesser[global_indices[1]].as_ref()
                                * Complex::from(weights[1] / (T::one() + T::one()));
                        gf_scattered_to *= Complex::from(
                            gamma_frohlich * T::from_f64(1. / 4. / std::f64::consts::PI).unwrap(),
                        );
                        for (idx, mut row) in retarded_self_energy_matrix.row_iter_mut().enumerate()
                        {
                            for (jdx, entry) in row.iter_mut().enumerate() {
                                let prefactor = Self::bulk_lo_phonon_potential(
                                    mesh,
                                    idx,
                                    jdx,
                                    (wavevector_k - wavevector_l).abs(),
                                );
                                *entry += Complex::from(prefactor * weight * wavevector_l)
                                    * gf_scattered_to[(idx, jdx)];
                            }
                        }
                    }
                }
            });
        Ok(())
    }

    fn bulk_lo_phonon_potential(
        mesh: &Mesh<T, GeometryDim, Conn>,
        vertex_a: usize,
        vertex_b: usize,
        wavevector: T,
    ) -> T {
        let debye_wavevector = T::from_f64(1_f64 / 10e-9).unwrap();
        let _common_wavevector = (debye_wavevector.powi(2) + wavevector.powi(2)).sqrt();
        let region_a = &mesh.vertices()[vertex_a].1;
        if *region_a != transporter_mesher::Assignment::Core(1) {
            return T::zero();
        }
        let d = T::from_f64(5e-9).unwrap();
        let center = T::from_f64(12.5e-9).unwrap();
        let xi = T::from_f64(std::f64::consts::PI).unwrap() / d;
        let z_a = &mesh.vertices()[vertex_a].0;
        let z_b = &mesh.vertices()[vertex_b].0;
        let _abs_offset = (z_a - z_b).norm();

        // (-common_wavevector * abs_offset).simd_exp() / common_wavevector
        //     * (T::one()
        //         - debye_wavevector.powi(2) * abs_offset / (T::one() + T::one()) / common_wavevector
        //         - debye_wavevector.powi(2) / (T::one() + T::one()) / common_wavevector.powi(2))
        ((z_a[0] - center) * xi).cos() * ((z_b[0] - center) * xi).cos()
            / (wavevector.powi(2) + xi.powi(2))
            / d
    }
}

impl<T, GeometryDim, Conn> SelfEnergy<T, GeometryDim, Conn>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim> + Send + Sync,
    <Conn as Connectivity<T, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Send + Sync,
{
    /// Updates the lesser LO Phonon scattering Self Energy at the contacts into the scratch matrix held in `self`
    ///
    ///
    pub(crate) fn recalculate_localised_lo_lesser_self_energy_mixed<BandDim: SmallDim, Spectral>(
        &mut self,
        scaling: T,
        mesh: &Mesh<T, GeometryDim, Conn>,
        spectral_space: &Spectral,
        greens_functions: &AggregateGreensFunctions<
            '_,
            T,
            MMatrix<Complex<T>>,
            GeometryDim,
            BandDim,
        >,
    ) -> Result<(), SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<T>,
        DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
        <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
    {
        let term = console::Term::stdout();
        term.move_cursor_to(0, 7).unwrap();
        term.clear_to_end_of_screen().unwrap();
        tracing::info!("Calculating LO Lesser self energy");
        let n_0 = T::from_f64(0.3481475088177923).unwrap(); // The LO phonon population, to be updated as we pass through the loop
        let e_0 = T::from_f64(0.035).unwrap(); // The phonon energy in electron volts
        let eps_fr = T::from_f64(20_f64).unwrap();
        let gamma_frohlich =
            scaling * T::from_f64(ELECTRON_CHARGE / 2_f64 / EPSILON_0).unwrap() * e_0 * eps_fr; // In electron volts

        let prefactor =
            Complex::from(gamma_frohlich * T::from_f64(1. / 4. / std::f64::consts::PI).unwrap());

        let num_vertices_in_reservoir = greens_functions.retarded[0].as_ref().drain_diagonal.len();
        let num_vertices_in_core = greens_functions.retarded[0].as_ref().core_matrix.shape().0;

        assert!(self.incoherent_lesser.is_some());

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

        // For each LO phonon wavevector integrate over all the connected electronic wavevectors
        self.incoherent_lesser
            .as_deref_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .progress_with(pb)
            .for_each(|(index, lesser_self_energy_matrix)| {
                let mut phonon_workspace: DMatrix<Complex<T>> =
                    DMatrix::zeros(num_vertices_in_core, num_vertices_in_core);
                let energy_index = index % spectral_space.number_of_energy_points();
                let wavevector_index_k = index / spectral_space.number_of_energy_points();
                let wavevector_k = spectral_space.wavevector_at(wavevector_index_k);

                // Reset the matrix
                lesser_self_energy_matrix.fill(Complex::from(T::zero()));

                for (_wavevector_index_l, ((weight, width), wavevector_l)) in spectral_space
                    .iter_wavevector_weights()
                    .zip(spectral_space.iter_wavevector_widths())
                    .zip(spectral_space.iter_wavevectors())
                    .enumerate()
                {
                    Self::assemble_phonon_potential(
                        &mut phonon_workspace,
                        num_vertices_in_reservoir,
                        mesh,
                        (wavevector_k - wavevector_l).abs(),
                    );
                    if spectral_space.energy_at(energy_index)
                        < spectral_space.energy_at(spectral_space.number_of_energy_points() - 1)
                            - e_0
                    {
                        // Construct the outscattering term
                        let energy_scattered_from = spectral_space.energy_at(energy_index) + e_0;
                        // Find the global indices of the two points in the energy mesh which bracket `energy_scattered_from`
                        let energy_indices = spectral_space
                            .identify_bracketing_indices(energy_scattered_from)
                            .unwrap();
                        let global_indices = [
                            energy_indices[0]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                            energy_indices[1]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                        ];
                        // Get the weights linearly interpolating around `energy_scattered_from`
                        let weights = spectral_space
                            .identify_bracketing_weights(energy_scattered_from)
                            .unwrap();

                        *lesser_self_energy_matrix = phonon_workspace
                            .scale((T::one() + n_0) * weight * width * wavevector_l)
                            * prefactor
                            * (greens_functions.lesser[global_indices[0]]
                                .as_ref()
                                .core_as_ref()
                                * Complex::from(weights[0])
                                + greens_functions.lesser[global_indices[1]]
                                    .as_ref()
                                    .core_as_ref()
                                    * Complex::from(weights[1]));
                    }

                    if spectral_space.energy_at(energy_index) > e_0 {
                        // Construct the outscattering term
                        let energy_scattered_to = spectral_space.energy_at(energy_index) - e_0;
                        // Find the global indices of the two points in the energy mesh which bracket `energy_scattered_from`
                        let energy_indices = spectral_space
                            .identify_bracketing_indices(energy_scattered_to)
                            .unwrap();
                        let global_indices = [
                            energy_indices[0]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                            energy_indices[1]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                        ];
                        // Get the weights linearly interpolating around `energy_scattered_from`
                        let weights = spectral_space
                            .identify_bracketing_weights(energy_scattered_to)
                            .unwrap();

                        *lesser_self_energy_matrix += phonon_workspace
                            .scale(n_0 * weight * width * wavevector_l)
                            * prefactor
                            * (greens_functions.lesser[global_indices[0]]
                                .as_ref()
                                .core_as_ref()
                                * Complex::from(weights[0])
                                + greens_functions.lesser[global_indices[1]]
                                    .as_ref()
                                    .core_as_ref()
                                    * Complex::from(weights[1]));
                    };
                }
            });

        Ok(())
    }

    pub(crate) fn recalculate_localised_lo_retarded_self_energy_mixed<BandDim: SmallDim, Spectral>(
        &mut self,
        scaling: T,
        mesh: &Mesh<T, GeometryDim, Conn>,
        spectral_space: &Spectral,
        greens_functions: &AggregateGreensFunctions<
            '_,
            T,
            MMatrix<Complex<T>>,
            GeometryDim,
            BandDim,
        >,
    ) -> Result<(), SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<T>,
        DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
        <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
    {
        let term = console::Term::stdout();
        term.move_cursor_to(0, 7).unwrap();
        term.clear_to_end_of_screen().unwrap();
        tracing::info!("Calculating LO retarded self energy");
        let n_0 = T::from_f64(0.3481475088177923).unwrap(); // The LO phonon population, to be updated as we pass through the loop
        let e_0 = T::from_f64(0.035).unwrap(); // The phonon energy in electron volts
        let eps_fr = T::from_f64(20_f64).unwrap();
        let gamma_frohlich =
            scaling * T::from_f64(ELECTRON_CHARGE / 2_f64 / EPSILON_0).unwrap() * e_0 * eps_fr; // In electron volts!!
        let prefactor =
            Complex::from(gamma_frohlich * T::from_f64(1. / 4. / std::f64::consts::PI).unwrap());

        let num_vertices_in_reservoir = greens_functions.retarded[0].as_ref().drain_diagonal.len();
        let num_vertices_in_core = greens_functions.retarded[0].as_ref().core_matrix.shape().0;

        assert!(self.incoherent_retarded.is_some());

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

        self.incoherent_retarded
            .as_deref_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .progress_with(pb)
            .for_each(|(index, retarded_self_energy_matrix)| {
                let energy_index = index % spectral_space.number_of_energy_points();
                let wavevector_index_k = index / spectral_space.number_of_energy_points();
                let wavevector_k = spectral_space.wavevector_at(wavevector_index_k);
                let mut phonon_workspace: DMatrix<Complex<T>> =
                    DMatrix::zeros(num_vertices_in_core, num_vertices_in_core);

                // Reset the matrix
                retarded_self_energy_matrix.fill(Complex::from(T::zero()));
                for (_wavevector_index_l, ((weight, width), wavevector_l)) in spectral_space
                    .iter_wavevector_weights()
                    .zip(spectral_space.iter_wavevector_widths())
                    .zip(spectral_space.iter_wavevectors())
                    .enumerate()
                {
                    Self::assemble_phonon_potential(
                        &mut phonon_workspace,
                        num_vertices_in_reservoir,
                        mesh,
                        (wavevector_k - wavevector_l).abs(),
                    );
                    if spectral_space.energy_at(energy_index)
                        < spectral_space.energy_at(spectral_space.number_of_energy_points() - 1)
                            - e_0
                    {
                        // Construct the outscattering term
                        let energy_scattered_from = spectral_space.energy_at(energy_index) + e_0;
                        // Find the global indices of the two points in the energy mesh which bracket `energy_scattered_from`
                        let energy_indices = spectral_space
                            .identify_bracketing_indices(energy_scattered_from)
                            .unwrap();
                        let global_indices = [
                            energy_indices[0]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                            energy_indices[1]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                        ];
                        // Get the weights linearly interpolating around `energy_scattered_from`
                        let weights = spectral_space
                            .identify_bracketing_weights(energy_scattered_from)
                            .unwrap();
                        // Form the best guess for the lesser GF at `energy_scattered_from` (Eq 6, no PV)

                        *retarded_self_energy_matrix = phonon_workspace
                            .scale(weight * width * wavevector_l)
                            * prefactor
                            * (greens_functions.retarded[global_indices[0]]
                                .as_ref()
                                .core_as_ref()
                                * Complex::from(weights[0] * (T::one() + n_0))
                                + greens_functions.retarded[global_indices[1]]
                                    .as_ref()
                                    .core_as_ref()
                                    * Complex::from(weights[1] * (T::one() + n_0))
                                - greens_functions.lesser[global_indices[0]]
                                    .as_ref()
                                    .core_as_ref()
                                    * Complex::from(weights[0] / (T::one() + T::one()))
                                - greens_functions.lesser[global_indices[1]]
                                    .as_ref()
                                    .core_as_ref()
                                    * Complex::from(weights[1] / (T::one() + T::one())));
                    }

                    if spectral_space.energy_at(energy_index) > e_0 {
                        // Construct the outscattering term
                        let energy_scattered_to = spectral_space.energy_at(energy_index) - e_0;
                        // Find the global indices of the two points in the energy mesh which bracket `energy_scattered_from`
                        let energy_indices = spectral_space
                            .identify_bracketing_indices(energy_scattered_to)
                            .unwrap();
                        let global_indices = [
                            energy_indices[0]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                            energy_indices[1]
                                + wavevector_index_k * spectral_space.number_of_energy_points(),
                        ];
                        // Get the weights linearly interpolating around `energy_scattered_from`
                        let weights = spectral_space
                            .identify_bracketing_weights(energy_scattered_to)
                            .unwrap();
                        // Form the best guess for the lesser GF at `energy_scattered_from` (Eq 6, no PV)
                        *retarded_self_energy_matrix += phonon_workspace
                            .scale(weight * width * wavevector_l)
                            * prefactor
                            * (greens_functions.retarded[global_indices[0]]
                                .as_ref()
                                .core_as_ref()
                                * Complex::from(weights[0] * n_0)
                                + greens_functions.retarded[global_indices[1]]
                                    .as_ref()
                                    .core_as_ref()
                                    * Complex::from(weights[1] * n_0)
                                - greens_functions.lesser[global_indices[0]]
                                    .as_ref()
                                    .core_as_ref()
                                    * Complex::from(weights[0] / (T::one() + T::one()))
                                - greens_functions.lesser[global_indices[1]]
                                    .as_ref()
                                    .core_as_ref()
                                    * Complex::from(weights[1] / (T::one() + T::one())));
                    }
                }
            });

        Ok(())
    }

    pub(crate) fn calculate_localised_lo_scattering_rate<BandDim: SmallDim, Spectral>(
        &self,
        spectral_space: &Spectral,
        greens_functions: &AggregateGreensFunctions<
            '_,
            T,
            MMatrix<Complex<T>>,
            GeometryDim,
            BandDim,
        >,
    ) -> Result<Complex<T>, SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<T>,
        DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
        <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
    {
        let term = console::Term::stdout();
        term.move_cursor_to(0, 7).unwrap();
        term.clear_to_end_of_screen().unwrap();
        tracing::info!("Calculating LO scattering rate");

        assert!(self.incoherent_retarded.is_some());

        let mut result = Complex::from(T::zero());

        // \Sigma^{<} =

        let wavevector_weights = spectral_space.iter_wavevector_weights().collect::<Vec<_>>();
        let energy_weights = spectral_space.iter_energy_weights().collect::<Vec<_>>();

        for index in 0..self.incoherent_retarded.as_deref().unwrap().len() {
            let energy_index = index % spectral_space.number_of_energy_points();
            let wavevector_index_k = index / spectral_space.number_of_energy_points();
            let wavevector_k = spectral_space.wavevector_at(wavevector_index_k);

            // Sigma^> = \Sigma^R - \Sigma^A + \Sigma^<
            let incoherent_greater = &self.incoherent_retarded.as_deref().unwrap()[index]
                - self.incoherent_retarded.as_deref().unwrap()[index].adjoint()
                + &self.incoherent_lesser.as_deref().unwrap()[index];

            // G^> = G^R - G^A + G^<
            let g_greater = &greens_functions.retarded[index].as_ref().core_matrix
                - &greens_functions.retarded[index]
                    .as_ref()
                    .core_matrix
                    .adjoint()
                + &greens_functions.lesser[index].as_ref().core_matrix;

            let integrand = (&self.incoherent_lesser.as_deref().unwrap()[index] * g_greater
                - incoherent_greater * &greens_functions.lesser[index].as_ref().core_matrix)
                .diagonal();
            if integrand.sum() == Complex::from(T::zero()) {
                dbg!(
                    integrand,
                    &greens_functions.retarded[index].as_ref().core_matrix,
                    &self.incoherent_lesser.as_deref().unwrap()[index]
                );
                panic!();
            }
            let integrand = integrand.sum();
            let prefactor = wavevector_k
                * T::from_f64(
                    2_f64 * ELECTRON_CHARGE
                        / crate::constants::HBAR
                        / (2_f64 * std::f64::consts::PI).powi(2),
                )
                .unwrap()
                * wavevector_weights[wavevector_index_k]
                * energy_weights[energy_index];
            result += Complex::from(prefactor) * integrand;
        }
        Ok(result)
    }

    fn assemble_phonon_potential(
        output: &mut nalgebra::DMatrix<Complex<T>>,
        number_of_vertices_in_reservoir: usize,
        mesh: &Mesh<T, GeometryDim, Conn>,
        wavevector: T,
    ) {
        let number_of_vertices_in_core = output.shape().0;
        for (index, element) in output.iter_mut().enumerate() {
            *element = Complex::from(Self::bulk_lo_phonon_potential_mixed(
                mesh,
                index / number_of_vertices_in_core + number_of_vertices_in_reservoir,
                index % number_of_vertices_in_core + number_of_vertices_in_reservoir,
                wavevector,
            ));
        }
    }

    fn bulk_lo_phonon_potential_mixed(
        mesh: &Mesh<T, GeometryDim, Conn>,
        vertex_a: usize,
        vertex_b: usize,
        wavevector: T,
    ) -> T {
        let debye_wavevector = T::from_f64(1_f64 / 10e-9).unwrap();
        let _common_wavevector = (debye_wavevector.powi(2) + wavevector.powi(2)).sqrt();
        let region_a = &mesh.vertices()[vertex_a].1;
        if *region_a != transporter_mesher::Assignment::Core(2) {
            return T::zero();
        }
        let d = T::from_f64(5e-9).unwrap();
        let center = T::from_f64(37.5e-9).unwrap();
        let xi = T::from_f64(std::f64::consts::PI).unwrap() / d;
        let z_a = &mesh.vertices()[vertex_a].0;
        let z_b = &mesh.vertices()[vertex_b].0;
        let _abs_offset = (z_a - z_b).norm();

        ((z_a[0] - center) * xi).cos() * ((z_b[0] - center) * xi).cos()
            / (wavevector.powi(2) + xi.powi(2))
            / d
    }
}
