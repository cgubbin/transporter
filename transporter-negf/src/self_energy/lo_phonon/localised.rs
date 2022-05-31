//! # Localised longitudinal phonon scattering
//!
//! This module calculates the self-energies for scattering between electronic
//! states via either emission or absorption of a quantised, localised longitudinal
//! optic phonon

use super::super::{SelfEnergy, SelfEnergyError};
use crate::{
    constants::{ELECTRON_CHARGE, EPSILON_0},
    greens_functions::{methods::mixed::MMatrix, AggregateGreensFunctions},
    spectral::SpectralDiscretisation,
};
use nalgebra::{allocator::Allocator, DefaultAllocator};
use ndarray::{Array1, Array2};
use num_complex::Complex;
use rayon::prelude::*;
use transporter_mesher::{Connectivity, ElementMethods, Mesh, SmallDim};

impl<GeometryDim, Conn> SelfEnergy<f64, GeometryDim, Conn>
where
    GeometryDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<f64, GeometryDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
{
    /// Updates the lesser LO Phonon scattering Self Energy at the contacts into the scratch matrix held in `self`
    ///
    ///
    pub(crate) fn recalculate_localised_lo_lesser_self_energy<BandDim: SmallDim, Spectral>(
        &mut self,
        scaling: f64,
        mesh: &Mesh<f64, GeometryDim, Conn>,
        spectral_space: &Spectral,
        greens_functions: &AggregateGreensFunctions<
            '_,
            f64,
            Array2<Complex<f64>>,
            GeometryDim,
            BandDim,
        >,
    ) -> Result<(), SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<f64>,
        DefaultAllocator: Allocator<f64, BandDim> + Allocator<[f64; 3], BandDim>,
        <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
    {
        tracing::info!("Calculating LO Lesser SE {}", scaling);
        let n_0 = 0.3481475088177923_f64; // The LO phonon population, to be updated as we pass through the loop
        let e_0 = 0.035_f64; // The phonon energy in electron volts
        let eps_fr = 20_f64;
        let gamma_frohlich = scaling * ELECTRON_CHARGE / 2_f64 / EPSILON_0 * e_0 * eps_fr; // In electron volts

        assert!(self.incoherent_lesser.is_some());

        // For each LO phonon wavevector integrate over all the connected electronic wavevectors
        self.incoherent_lesser
            .as_deref_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, lesser_self_energy_matrix)| {
                let energy_index = index % spectral_space.number_of_energy_points();
                let wavevector_index_k = index / spectral_space.number_of_energy_points();
                let wavevector_k = spectral_space.wavevector_at(wavevector_index_k);

                // Reset the matrix
                lesser_self_energy_matrix.fill(Complex::from(0_f64));
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
                        lesser_gf_scattered_from *=
                            Complex::from(gamma_frohlich * 1. / 4. / std::f64::consts::PI);
                        for (idx, mut row) in lesser_self_energy_matrix.outer_iter_mut().enumerate()
                        {
                            for (jdx, entry) in row.iter_mut().enumerate() {
                                let prefactor = Self::bulk_lo_phonon_potential(
                                    mesh,
                                    idx,
                                    jdx,
                                    (wavevector_k - wavevector_l).abs(),
                                );
                                *entry += Complex::from(
                                    prefactor * (1_f64 + n_0) * weight * width * wavevector_l,
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
                        lesser_gf_scattered_to *=
                            Complex::from(gamma_frohlich * 1. / 4. / std::f64::consts::PI);

                        for (idx, mut row) in lesser_self_energy_matrix.outer_iter_mut().enumerate()
                        {
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
        scaling: f64,
        mesh: &Mesh<f64, GeometryDim, Conn>,
        spectral_space: &Spectral,
        greens_functions: &AggregateGreensFunctions<
            '_,
            f64,
            Array2<Complex<f64>>,
            GeometryDim,
            BandDim,
        >,
    ) -> Result<(), SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<f64>,
        DefaultAllocator: Allocator<f64, BandDim> + Allocator<[f64; 3], BandDim>,
        <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
    {
        tracing::info!("Calculating LO retarded SE {}", scaling);
        let n_0 = 0.3481475088177923; // The LO phonon population, to be updated as we pass through the loop
        let e_0 = 0.035; // The phonon energy in electron volts
        let eps_fr = 20_f64;
        let gamma_frohlich = scaling * ELECTRON_CHARGE / 2_f64 / EPSILON_0 * e_0 * eps_fr; // In electron volts!!

        assert!(self.incoherent_retarded.is_some());

        self.incoherent_retarded
            .as_deref_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, retarded_self_energy_matrix)| {
                let energy_index = index % spectral_space.number_of_energy_points();
                let wavevector_index_k = index / spectral_space.number_of_energy_points();
                let wavevector_k = spectral_space.wavevector_at(wavevector_index_k);

                // Reset the matrix
                retarded_self_energy_matrix.fill(Complex::from(0_f64));
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
                                * Complex::from(weights[0] / 2_f64)
                            - greens_functions.lesser[global_indices[1]].as_ref()
                                * Complex::from(weights[1] / 2_f64);
                        gf_scattered_from *=
                            Complex::from(gamma_frohlich * 1. / 4. / std::f64::consts::PI);
                        for (idx, mut row) in
                            retarded_self_energy_matrix.outer_iter_mut().enumerate()
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
                            * Complex::from(weights[0] * (1_f64 + n_0))
                            + greens_functions.retarded[global_indices[1]].as_ref()
                                * Complex::from(weights[1] * (1_f64 + n_0))
                            + greens_functions.lesser[global_indices[0]].as_ref()
                                * Complex::from(weights[0] / 2_f64)
                            + greens_functions.lesser[global_indices[1]].as_ref()
                                * Complex::from(weights[1] / 2_f64);
                        gf_scattered_to *=
                            Complex::from(gamma_frohlich * 1. / 4. / std::f64::consts::PI);
                        for (idx, mut row) in
                            retarded_self_energy_matrix.outer_iter_mut().enumerate()
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
        mesh: &Mesh<f64, GeometryDim, Conn>,
        vertex_a: usize,
        vertex_b: usize,
        wavevector: f64,
    ) -> f64 {
        let debye_wavevector = 1_f64 / 10e-9;
        let _common_wavevector = (debye_wavevector.powi(2) + wavevector.powi(2)).sqrt();
        let region_a = &mesh.vertices()[vertex_a].1;
        if *region_a != transporter_mesher::Assignment::Core(1) {
            return 0_f64;
        }
        let d = 5e-9;
        let center = 12.5e-9;
        let xi = std::f64::consts::PI / d;
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

impl<GeometryDim, Conn> SelfEnergy<f64, GeometryDim, Conn>
where
    GeometryDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<f64, GeometryDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
{
    /// Updates the lesser LO Phonon scattering Self Energy at the contacts into the scratch matrix held in `self`
    ///
    ///
    pub(crate) fn recalculate_localised_lo_lesser_self_energy_mixed<BandDim: SmallDim, Spectral>(
        &mut self,
        scaling: f64,
        mesh: &Mesh<f64, GeometryDim, Conn>,
        spectral_space: &Spectral,
        greens_functions: &AggregateGreensFunctions<
            '_,
            f64,
            MMatrix<Complex<f64>>,
            GeometryDim,
            BandDim,
        >,
    ) -> Result<(), SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<f64>,
        DefaultAllocator: Allocator<f64, BandDim> + Allocator<[f64; 3], BandDim>,
        <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
    {
        tracing::info!("Calculating LO Lesser self energy");
        let n_0 = 0.3481475088177923; // The LO phonon population, to be updated as we pass through the loop
        let e_0 = 0.035; // The phonon energy in electron volts
        let eps_fr = 20_f64;
        let gamma_frohlich = scaling * ELECTRON_CHARGE / 2_f64 / EPSILON_0 * e_0 * eps_fr; // In electron volts

        let prefactor = Complex::from(gamma_frohlich * 1. / 4. / std::f64::consts::PI);

        let num_vertices_in_reservoir = greens_functions.retarded[0].as_ref().drain_diagonal.len();
        let num_vertices_in_core = greens_functions.retarded[0].as_ref().core_matrix.shape()[0];

        assert!(self.incoherent_lesser.is_some());

        // For each LO phonon wavevector integrate over all the connected electronic wavevectors
        self.incoherent_lesser
            .as_deref_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, lesser_self_energy_matrix)| {
                let mut phonon_workspace: Array2<Complex<f64>> =
                    Array2::zeros((num_vertices_in_core, num_vertices_in_core));
                // let phonon_workspace =
                //     Array2::from_diag_elem(num_vertices_in_core, Complex::from(1_f64));
                let energy_index = index % spectral_space.number_of_energy_points();
                let wavevector_index_k = index / spectral_space.number_of_energy_points();
                let wavevector_k = spectral_space.wavevector_at(wavevector_index_k);

                // Reset the matrix
                lesser_self_energy_matrix.fill(Complex::from(0_f64));

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

                        // // TODO remove these checks when satisfied with security checks elsewhere
                        // assert!(crate::utilities::matrices::is_anti_hermitian(
                        //     greens_functions.lesser[global_indices[0]]
                        //         .as_ref()
                        //         .core_as_ref()
                        //         .view()
                        // ));
                        // assert!(crate::utilities::matrices::is_anti_hermitian(
                        //     greens_functions.lesser[global_indices[1]]
                        //         .as_ref()
                        //         .core_as_ref()
                        //         .view()
                        // ));

                        *lesser_self_energy_matrix = lesser_self_energy_matrix.clone()
                            + (&phonon_workspace
                                * Complex::from((1_f64 + n_0) * weight * width * wavevector_l)
                                * prefactor)
                                .dot(
                                    &(greens_functions.lesser[global_indices[0]]
                                        .as_ref()
                                        .core_as_ref()
                                        * Complex::from(weights[0])
                                        + greens_functions.lesser[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1]))
                                    .dot(&phonon_workspace.t()),
                                );
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

                        // // TODO remove these checks when satisfied with security checks elsewhere
                        // assert!(crate::utilities::matrices::is_anti_hermitian(
                        //     greens_functions.lesser[global_indices[0]]
                        //         .as_ref()
                        //         .core_as_ref()
                        //         .view()
                        // ));
                        // assert!(crate::utilities::matrices::is_anti_hermitian(
                        //     greens_functions.lesser[global_indices[1]]
                        //         .as_ref()
                        //         .core_as_ref()
                        //         .view()
                        // ));

                        *lesser_self_energy_matrix = lesser_self_energy_matrix.clone()
                            + &(&phonon_workspace
                                * Complex::from(n_0 * weight * width * wavevector_l)
                                * prefactor)
                                .dot(
                                    &(greens_functions.lesser[global_indices[0]]
                                        .as_ref()
                                        .core_as_ref()
                                        * Complex::from(weights[0])
                                        + greens_functions.lesser[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1]))
                                    .dot(&phonon_workspace.t()),
                                );
                    };
                }
            });

        // Security check for the Lesser Self Energy
        //
        // It should be the case that, as the lesser Green's function is anti-hermitian and we are left multiplying
        // by M, and right multiplying by the adjoint M^\dag that the product M G^< M^{\dag} persists the anti
        // hermiticity of the Green's function. This is necessary for conservation of particle number.
        if self.security_checks {
            self.incoherent_lesser
                .as_deref()
                .unwrap()
                .par_iter()
                .enumerate()
                .try_for_each(|(index, lesser_self_energy_matrix)| {
                    // Check the self energy is anti-hermitian
                    if crate::utilities::matrices::is_anti_hermitian(
                        lesser_self_energy_matrix.view(),
                    ) {
                        Ok(())
                    } else {
                        Err(crate::greens_functions::SecurityCheck {
                            calculation: "lesser localised phonon self-energy".into(),
                            index,
                        })
                    }
                })?;
        }

        Ok(())
    }

    pub(crate) fn recalculate_localised_lo_retarded_self_energy_mixed<BandDim: SmallDim, Spectral>(
        &mut self,
        scaling: f64,
        mesh: &Mesh<f64, GeometryDim, Conn>,
        spectral_space: &Spectral,
        greens_functions: &AggregateGreensFunctions<
            '_,
            f64,
            MMatrix<Complex<f64>>,
            GeometryDim,
            BandDim,
        >,
    ) -> Result<(), SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<f64>,
        DefaultAllocator: Allocator<f64, BandDim> + Allocator<[f64; 3], BandDim>,
        <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
    {
        tracing::info!("Calculating LO retarded self energy");
        let n_0 = 0.3481475088177923; // The LO phonon population, to be updated as we pass through the loop
        let e_0 = 0.035; // The phonon energy in electron volts
        let eps_fr = 20_f64;
        let gamma_frohlich = scaling * ELECTRON_CHARGE / 2_f64 / EPSILON_0 * e_0 * eps_fr; // In electron volts!!
        let prefactor = Complex::from(gamma_frohlich * 1. / 4. / std::f64::consts::PI);

        let num_vertices_in_reservoir = greens_functions.retarded[0].as_ref().drain_diagonal.len();
        let num_vertices_in_core = greens_functions.retarded[0].as_ref().core_matrix.shape()[0];

        assert!(self.incoherent_retarded.is_some());

        self.incoherent_retarded
            .as_deref_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, retarded_self_energy_matrix)| {
                let energy_index = index % spectral_space.number_of_energy_points();
                let wavevector_index_k = index / spectral_space.number_of_energy_points();
                let wavevector_k = spectral_space.wavevector_at(wavevector_index_k);
                let mut phonon_workspace: Array2<Complex<f64>> =
                    Array2::zeros((num_vertices_in_core, num_vertices_in_core));

                // let phonon_workspace =
                //     Array2::from_diag_elem(num_vertices_in_core, Complex::from(1_f64));

                // Reset the matrix
                retarded_self_energy_matrix.fill(Complex::from(0_f64));
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

                        *retarded_self_energy_matrix = retarded_self_energy_matrix.clone()
                            + (&phonon_workspace
                                * Complex::from(weight * width * wavevector_l)
                                * prefactor)
                                .dot(
                                    &(greens_functions.retarded[global_indices[0]]
                                        .as_ref()
                                        .core_as_ref()
                                        * Complex::from(weights[0] * (n_0))
                                        + greens_functions.retarded[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1] * (n_0))
                                        - greens_functions.lesser[global_indices[0]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[0] / (2_f64))
                                        - greens_functions.lesser[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1] / (2_f64)))
                                    .dot(&phonon_workspace.t()),
                                );
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
                        *retarded_self_energy_matrix = retarded_self_energy_matrix.clone()
                            + &(&phonon_workspace
                                * Complex::from(weight * width * wavevector_l)
                                * prefactor)
                                .dot(
                                    &(greens_functions.retarded[global_indices[0]]
                                        .as_ref()
                                        .core_as_ref()
                                        * Complex::from(weights[0] * (1_f64 + n_0))
                                        + greens_functions.retarded[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1] * (1_f64 + n_0))
                                        + greens_functions.lesser[global_indices[0]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[0] / (2_f64))
                                        + greens_functions.lesser[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1] / (2_f64)))
                                    .dot(&phonon_workspace.t()),
                                );
                    }
                }
            });

        Ok(())
    }

    /// Find the net scattering rate of LO phonons integrated over all energy and wavevector states
    ///
    /// This should approach zero if the solution is converged because LO scattering conserves particle
    /// number: ie we should be able to track the progress of an incoherent calculation by observation
    /// of this observable. At the moment this is fundamentally not happening so we need to work out why
    /// this is.
    // todo currently assumes the mesh is uniform: need to zip the sum with the element widths
    pub(crate) fn calculate_localised_lo_scattering_rate<BandDim: SmallDim, Spectral>(
        &self,
        spectral_space: &Spectral,
        mesh: &Mesh<f64, GeometryDim, Conn>,
        greens_functions: &AggregateGreensFunctions<
            '_,
            f64,
            MMatrix<Complex<f64>>,
            GeometryDim,
            BandDim,
        >,
    ) -> Result<Complex<f64>, SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<f64>,
        DefaultAllocator: Allocator<f64, BandDim> + Allocator<[f64; 3], BandDim>,
        <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
    {
        tracing::info!("Calculating LO scattering rate");

        assert!(self.incoherent_retarded.is_some());

        let mut result = Complex::from(0_f64);

        // \Sigma^{<} =

        let wavevector_weights = spectral_space.iter_wavevector_weights().collect::<Vec<_>>();
        let energy_weights = spectral_space.iter_energy_weights().collect::<Vec<_>>();

        let mut wavevector_widths = spectral_space.iter_wavevector_widths().collect::<Vec<_>>();
        let mut energy_widths = spectral_space.iter_energy_widths().collect::<Vec<_>>();
        // Widths are 1 shorter than weights so we do a dumb-extend here. Do this better later
        // TODO
        wavevector_widths.push(wavevector_widths[0]);
        energy_widths.push(energy_widths[0]);

        for index in 0..self.incoherent_retarded.as_deref().unwrap().len() {
            let energy_index = index % spectral_space.number_of_energy_points();
            let wavevector_index_k = index / spectral_space.number_of_energy_points();
            let wavevector_k = spectral_space.wavevector_at(wavevector_index_k);

            // Sigma^> = \Sigma^R - \Sigma^A + \Sigma^<
            // let incoherent_advanced = self.incoherent_retarded.as_deref().unwrap()[index]
            //     .clone()
            //     .t()
            //     .mapv(|x| x.conj());
            // let _incoherent_greater = &self.incoherent_retarded.as_deref().unwrap()[index]
            //     - &incoherent_advanced
            //     + &self.incoherent_lesser.as_deref().unwrap()[index];

            // G^> = G^R - G^A + G^<
            let g_advanced = greens_functions.retarded[index]
                .as_ref()
                .core_matrix
                .clone()
                .t()
                .mapv(|x| x.conj());
            let g_greater = &greens_functions.retarded[index].as_ref().core_matrix - &g_advanced
                + &greens_functions.lesser[index].as_ref().core_matrix;

            if self.security_checks {
                // Check the greater self energy is anti-hermitian
                if crate::utilities::matrices::is_anti_hermitian(g_greater.view()) {
                    Ok(())
                } else {
                    Err(crate::greens_functions::SecurityCheck {
                        calculation: "retarded localised phonon self-energy".into(),
                        index,
                    })
                }?
            }

            // R = \Sigma^< G^> - \Sigma^> G^<
            let integrand = (
                &self.incoherent_lesser.as_deref().unwrap()[index].dot(&g_greater)
                // - incoherent_greater.dot(&greens_functions.lesser[index].as_ref().core_matrix))
            )
                .diag()
                .to_owned();

            // If the integrand is zero we have called the method without setting the self-energies or Greens functions. This should not
            // happen so we panic.
            if integrand.sum() == Complex::from(0_f64) {
                panic!();
            }
            // Take the trace of the resulting matrix
            let integrand = integrand.sum();

            // Get the prefactor
            // k dk dE * e * 2 / (2 \pi)^2 / hbar
            // with weighting according to the integration rule chosen at runtime
            let prefactor = wavevector_k * 2_f64 * ELECTRON_CHARGE
                / crate::constants::HBAR
                / (2_f64 * std::f64::consts::PI).powi(2)
                * wavevector_weights[wavevector_index_k]
                * wavevector_widths[wavevector_index_k]
                * energy_weights[energy_index]
                * energy_widths[energy_index]
                * mesh.elements()[0].0.diameter();
            result += Complex::from(prefactor) * integrand;
        }
        Ok(result)
    }

    /// \Sigma^< * G^>
    pub(crate) fn calculate_resolved_localised_lo_emission_rate<BandDim: SmallDim, Spectral>(
        &self,
        spectral_space: &Spectral,
        mesh: &Mesh<f64, GeometryDim, Conn>,
        greens_functions: &AggregateGreensFunctions<
            '_,
            f64,
            MMatrix<Complex<f64>>,
            GeometryDim,
            BandDim,
        >,
    ) -> Result<Array1<Complex<f64>>, SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<f64>,
        DefaultAllocator: Allocator<f64, BandDim> + Allocator<[f64; 3], BandDim>,
        <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
    {
        tracing::info!("Calculating LO generation rate");

        let mut result = Array1::zeros(spectral_space.number_of_wavevector_points());

        let n_0 = 0.3481475088177923; // The LO phonon population, to be updated as we pass through the loop
        let e_0 = 0.035; // The phonon energy in electron volts
        let eps_fr = 20_f64;
        let gamma_frohlich = ELECTRON_CHARGE / 2_f64 / EPSILON_0 * e_0 * eps_fr; // In electron volts

        let prefactor = Complex::from(
            gamma_frohlich * 1. / 4. / std::f64::consts::PI * 2_f64
                / crate::constants::HBAR
                / (2_f64 * std::f64::consts::PI).powi(2),
        );

        let num_vertices_in_reservoir = greens_functions.retarded[0].as_ref().drain_diagonal.len();
        let num_vertices_in_core = greens_functions.retarded[0].as_ref().core_matrix.shape()[0];

        assert!(self.incoherent_lesser.is_some());

        result
            .as_slice_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(wavevector_index_l, element)| {
                let mut phonon_workspace: Array2<Complex<f64>> =
                    Array2::zeros((num_vertices_in_core, num_vertices_in_core));
                let wavevector_l = spectral_space.wavevector_at(wavevector_index_l);

                // Reset the matrix
                for (wavevector_index_k, ((weight, width), wavevector_k)) in spectral_space
                    .iter_wavevector_weights()
                    .zip(spectral_space.iter_wavevector_widths())
                    .zip(spectral_space.iter_wavevectors())
                    .enumerate()
                {
                    for (energy_index, (energy_weight, energy_width)) in spectral_space
                        .iter_energy_weights()
                        .zip(spectral_space.iter_energy_widths())
                        .enumerate()
                    {
                        // Index of the greens function at energy_index
                        let base_index = energy_index
                            + wavevector_index_k * spectral_space.number_of_energy_points();
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
                            let energy_scattered_from =
                                spectral_space.energy_at(energy_index) + e_0;
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

                            let g_advanced = greens_functions.retarded[base_index]
                                .as_ref()
                                .core_matrix
                                .clone()
                                .t()
                                .mapv(|x| x.conj());
                            let g_greater =
                                &greens_functions.retarded[base_index].as_ref().core_matrix
                                    - &g_advanced
                                    + &greens_functions.lesser[base_index].as_ref().core_matrix;

                            *element += ((&phonon_workspace
                                * Complex::from(
                                    (1_f64 + n_0)
                                        * weight
                                        * width
                                        * wavevector_k
                                        * energy_weight
                                        * energy_width
                                        * crate::constants::ELECTRON_CHARGE,
                                )
                                * prefactor)
                                .dot(
                                    &(greens_functions.lesser[global_indices[0]]
                                        .as_ref()
                                        .core_as_ref()
                                        * Complex::from(weights[0])
                                        + greens_functions.lesser[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1]))
                                    .dot(&phonon_workspace.t()),
                                ))
                            .dot(&g_greater)
                            .diag()
                            .sum()
                                * mesh.elements()[0].0.diameter();
                            //TODO Currently assuming a uniform meshing
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

                            let g_advanced = greens_functions.retarded[base_index]
                                .as_ref()
                                .core_matrix
                                .clone()
                                .t()
                                .mapv(|x| x.conj());

                            let g_greater =
                                &greens_functions.retarded[base_index].as_ref().core_matrix
                                    - &g_advanced
                                    + &greens_functions.lesser[base_index].as_ref().core_matrix;

                            *element += ((&phonon_workspace
                                * Complex::from(
                                    n_0 * weight
                                        * width
                                        * wavevector_k
                                        * energy_width
                                        * energy_weight
                                        * crate::constants::ELECTRON_CHARGE,
                                )
                                * prefactor)
                                .dot(
                                    &(greens_functions.lesser[global_indices[0]]
                                        .as_ref()
                                        .core_as_ref()
                                        * Complex::from(weights[0])
                                        + greens_functions.lesser[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1]))
                                    .dot(&phonon_workspace.t()),
                                ))
                            .dot(&g_greater)
                            .sum()
                                * mesh.elements()[0].0.diameter();
                        };
                    }
                }
            });
        Ok(result)
    }

    /// /Sigma^> * G^<
    /// or - \Im \Sigma^{R}
    pub(crate) fn calculate_resolved_localised_lo_absorption_rate<BandDim: SmallDim, Spectral>(
        &self,
        spectral_space: &Spectral,
        mesh: &Mesh<f64, GeometryDim, Conn>,
        greens_functions: &AggregateGreensFunctions<
            '_,
            f64,
            MMatrix<Complex<f64>>,
            GeometryDim,
            BandDim,
        >,
    ) -> Result<Array1<Complex<f64>>, SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<f64>,
        DefaultAllocator: Allocator<f64, BandDim> + Allocator<[f64; 3], BandDim>,
        <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
        <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
    {
        tracing::info!("Calculating LO generation rate");

        let mut result = Array1::zeros(spectral_space.number_of_wavevector_points());

        let n_0 = 0.3481475088177923; // The LO phonon population, to be updated as we pass through the loop
        let e_0 = 0.035; // The phonon energy in electron volts
        let eps_fr = 20_f64;
        let gamma_frohlich = ELECTRON_CHARGE / 2_f64 / EPSILON_0 * e_0 * eps_fr; // In electron volts

        let prefactor = Complex::from(
            gamma_frohlich * 1. / 4. / std::f64::consts::PI * 2_f64
                / crate::constants::HBAR
                / (2_f64 * std::f64::consts::PI).powi(2),
        );

        let num_vertices_in_reservoir = greens_functions.retarded[0].as_ref().drain_diagonal.len();
        let num_vertices_in_core = greens_functions.retarded[0].as_ref().core_matrix.shape()[0];

        assert!(self.incoherent_lesser.is_some());

        result
            .as_slice_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(wavevector_index_l, element)| {
                let mut phonon_workspace: Array2<Complex<f64>> =
                    Array2::zeros((num_vertices_in_core, num_vertices_in_core));
                let wavevector_l = spectral_space.wavevector_at(wavevector_index_l);

                // Reset the matrix
                for (wavevector_index_k, ((weight, width), wavevector_k)) in spectral_space
                    .iter_wavevector_weights()
                    .zip(spectral_space.iter_wavevector_widths())
                    .zip(spectral_space.iter_wavevectors())
                    .enumerate()
                {
                    for (energy_index, (energy_weight, energy_width)) in spectral_space
                        .iter_energy_weights()
                        .zip(spectral_space.iter_energy_widths())
                        .enumerate()
                    {
                        // Index of the greens function at energy_index
                        let base_index = energy_index
                            + wavevector_index_k * spectral_space.number_of_energy_points();
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
                            let energy_scattered_from =
                                spectral_space.energy_at(energy_index) + e_0;
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

                            let lesser_se = (&phonon_workspace
                                * Complex::from(
                                    (1_f64 + n_0)
                                        * weight
                                        * width
                                        * wavevector_k
                                        * energy_width
                                        * energy_weight
                                        * crate::constants::ELECTRON_CHARGE,
                                )
                                * prefactor)
                                .dot(
                                    &(greens_functions.lesser[global_indices[0]]
                                        .as_ref()
                                        .core_as_ref()
                                        * Complex::from(weights[0])
                                        + greens_functions.lesser[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1]))
                                    .dot(&phonon_workspace.t()),
                                );

                            let retarded_se = (&phonon_workspace
                                * Complex::from(
                                    weight
                                        * width
                                        * wavevector_k
                                        * energy_width
                                        * energy_weight
                                        * crate::constants::ELECTRON_CHARGE,
                                )
                                * prefactor)
                                .dot(
                                    &(greens_functions.retarded[global_indices[0]]
                                        .as_ref()
                                        .core_as_ref()
                                        * Complex::from(weights[0] * (n_0))
                                        + greens_functions.retarded[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1] * (n_0))
                                        - greens_functions.lesser[global_indices[0]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[0] / (2_f64))
                                        - greens_functions.lesser[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1] / (2_f64)))
                                    .dot(&phonon_workspace.t()),
                                );

                            let advanced_se = retarded_se.clone().t().mapv(|x| x.conj());
                            let greater_se = &retarded_se - &advanced_se + &lesser_se;

                            *element += greater_se
                                .dot(greens_functions.lesser[base_index].as_ref().core_as_ref())
                                .diag()
                                .sum()
                                * mesh.elements()[0].0.diameter();

                            //TODO Currently assuming uniform meshing
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

                            let lesser_se = (&phonon_workspace
                                * Complex::from(
                                    n_0 * weight
                                        * width
                                        * wavevector_k
                                        * energy_width
                                        * energy_weight
                                        * crate::constants::ELECTRON_CHARGE,
                                )
                                * prefactor)
                                .dot(
                                    &(greens_functions.lesser[global_indices[0]]
                                        .as_ref()
                                        .core_as_ref()
                                        * Complex::from(weights[0])
                                        + greens_functions.lesser[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1]))
                                    .dot(&phonon_workspace.t()),
                                );

                            let retarded_se = (&phonon_workspace
                                * Complex::from(
                                    weight
                                        * width
                                        * wavevector_k
                                        * energy_width
                                        * energy_weight
                                        * crate::constants::ELECTRON_CHARGE,
                                )
                                * prefactor)
                                .dot(
                                    &(greens_functions.retarded[global_indices[0]]
                                        .as_ref()
                                        .core_as_ref()
                                        * Complex::from(weights[0] * (1_f64 + n_0))
                                        + greens_functions.retarded[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1] * (1_f64 + n_0))
                                        + greens_functions.lesser[global_indices[0]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[0] / (2_f64))
                                        + greens_functions.lesser[global_indices[1]]
                                            .as_ref()
                                            .core_as_ref()
                                            * Complex::from(weights[1] / (2_f64)))
                                    .dot(&phonon_workspace.t()),
                                );

                            let advanced_se = retarded_se.clone().t().mapv(|x| x.conj());
                            let greater_se = &retarded_se - &advanced_se + &lesser_se;

                            *element += greater_se
                                .dot(greens_functions.lesser[base_index].as_ref().core_as_ref())
                                .sum()
                                * mesh.elements()[0].0.diameter();
                        };
                    }
                }
            });
        Ok(result)
    }

    fn assemble_phonon_potential(
        output: &mut Array2<Complex<f64>>,
        number_of_vertices_in_reservoir: usize,
        mesh: &Mesh<f64, GeometryDim, Conn>,
        wavevector: f64,
    ) {
        let number_of_vertices_in_core = output.shape()[0];
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
        mesh: &Mesh<f64, GeometryDim, Conn>,
        vertex_a: usize,
        vertex_b: usize,
        wavevector: f64,
    ) -> f64 {
        let debye_wavevector = 1_f64 / 10e-9;
        let _common_wavevector = (debye_wavevector.powi(2) + wavevector.powi(2)).sqrt();
        let region_a = &mesh.vertices()[vertex_a].1;
        if *region_a != transporter_mesher::Assignment::Core(2) {
            return 0_f64;
        }

        let region_b = &mesh.vertices()[vertex_b].1;
        if *region_b != transporter_mesher::Assignment::Core(2) {
            return 0_f64;
        }
        let d = 5e-9;
        let center = 37.5e-9;
        let xi = std::f64::consts::PI / d;
        let z_a = &mesh.vertices()[vertex_a].0;
        let z_b = &mesh.vertices()[vertex_b].0;
        let abs_offset = (z_a - z_b).norm();

        // if vertex_a == vertex_b {
        ((abs_offset - center) * xi).cos() / ((wavevector.powi(2) + xi.powi(2)) * d).sqrt()
        // } else {
        //     0_f64
        // }
        // if vertex_a == vertex_b {
        //     1_f64 / (wavevector.powi(2) + xi.powi(2)) / d
        // } else {
        //     0_f64
        // }
    }
}
