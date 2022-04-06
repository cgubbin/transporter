use super::OuterLoop;
use crate::{
    inner_loop::{Inner, InnerLoop},
    self_energy::{SelfEnergy, SelfEnergyBuilder},
};
use nalgebra::{allocator::Allocator, DMatrix, DVector, DefaultAllocator, RealField};
use nalgebra_sparse::CsrMatrix;
use num_complex::Complex;
use transporter_mesher::{Assignment, Connectivity, SmallDim};

use nalgebra::{Const, Dynamic, Matrix, VecStorage};

/// A wrapper for the calculated electrostatic potential
#[derive(Clone, Debug)]
pub(crate) struct Potential<T: RealField>(DVector<T>);

impl<T: RealField> Potential<T> {
    pub(crate) fn from_vector(vector: DVector<T>) -> Self {
        Self(vector)
    }
    /// Check whether the change in the normalised potential is within the requested tolerance
    fn is_change_within_tolerance(&self, other: &Potential<T>, tolerance: T) -> bool {
        let norm = self.0.norm();
        let difference = (&self.0 - &other.0).norm() / norm;
        difference < tolerance
    }
}

impl<T: RealField> AsRef<DVector<T>> for Potential<T> {
    fn as_ref(&self) -> &DVector<T> {
        &self.0
    }
}

impl<T: Copy + RealField> Potential<T> {
    pub(crate) fn get(&self, vertex_index: usize) -> T {
        self.0[vertex_index]
    }
}

pub(crate) trait Outer<T>
where
    T: RealField,
{
    /// Compute the updated electric potential and confirm
    /// whether the change is within tolerance of the values on the
    /// previous loop iteration
    fn is_loop_converged(&self, previous_potential: &mut Potential<T>) -> color_eyre::Result<bool>;
    /// Carry out a single iteration of the self-consistent inner loop
    fn single_iteration(&mut self) -> color_eyre::Result<()>;
    /// Run the self-consistent inner loop to convergence
    fn run_loop(&mut self, potential: Potential<T>) -> color_eyre::Result<()>;
}

impl<T, GeometryDim, Conn, BandDim> Outer<T>
    for OuterLoop<'_, T, GeometryDim, Conn, BandDim, SpectralSpace<T, ()>>
where
    T: RealField + Copy,
    Conn: Connectivity<T, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>,
{
    fn is_loop_converged(&self, previous_potential: &mut Potential<T>) -> color_eyre::Result<bool> {
        let potential = self.update_potential(previous_potential)?;
        let result = potential.is_change_within_tolerance(
            previous_potential,
            self.convergence_settings.outer_tolerance(),
        );
        let _ = std::mem::replace(previous_potential, potential);
        Ok(result)
    }
    /// Carry out a single iteration of the self-consistent outer loop
    fn single_iteration(&mut self) -> color_eyre::Result<()> {
        // Build the inner loop, if we are running a ballistic calculation or have not arrived
        // at an initial converged ballistic solution then we create a
        // coherent inner loop, with sparse matrices, else we create a dense one.
        // TODO Building the gfs and SE here is a bad idea, we should do this else where so it is not redone on every iteration
        let mut greens_functions = GreensFunctionBuilder::new()
            .with_info_desk(self.info_desk)
            .with_mesh(self.mesh)
            .with_spectral_discretisation(self.spectral)
            .build()?;
        let mut self_energies = SelfEnergyBuilder::new()
            .with_mesh(self.mesh)
            .with_spectral_discretisation(self.spectral)
            .build()?;
        let mut inner_loop =
            self.build_coherent_inner_loop(&mut greens_functions, &mut self_energies);
        let mut charge_and_currents = self.tracker.charge_and_currents.clone();
        inner_loop.run_loop(&mut charge_and_currents)?;
        let _ = std::mem::replace(self.tracker.charge_and_currents_mut(), charge_and_currents);
        Ok(())
    }

    fn run_loop(&mut self, mut potential: Potential<T>) -> color_eyre::Result<()> {
        let mut iteration = 0;
        while !self.is_loop_converged(&mut potential)? {
            self.single_iteration()?;
            iteration += 1;
            if iteration >= self.convergence_settings.maximum_outer_iterations() {
                return Err(color_eyre::eyre::eyre!(
                    "Reached maximum iteration count in the outer loop"
                ));
            }
        }
        Ok(())
    }
}

use crate::spectral::{SpectralSpace, WavevectorSpace};

impl<T, GeometryDim, Conn, BandDim> Outer<T>
    for OuterLoop<
        '_,
        T,
        GeometryDim,
        Conn,
        BandDim,
        SpectralSpace<T, WavevectorSpace<T, GeometryDim, Conn>>,
    >
where
    T: RealField + Copy,
    Conn: Connectivity<T, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>,
{
    fn is_loop_converged(
        &self,
        _previous_potential: &mut Potential<T>,
    ) -> color_eyre::Result<bool> {
        //let potential = self.update_potential()?;
        //let result = potential.is_change_within_tolerance(
        //    previous_potential,
        //    self.convergence_settings.outer_tolerance(),
        //);
        //let _ = std::mem::replace(previous_potential, potential);
        //Ok(result)
        Ok(false)
    }
    /// Carry out a single iteration of the self-consistent outer loop
    fn single_iteration(&mut self) -> color_eyre::Result<()> {
        // Build the inner loop, if we are running a ballistic calculation or have not arrived
        // at an initial converged ballistic solution then we create a
        // coherent inner loop, with sparse matrices, else we create a dense one.
        // TODO Builder the gfs and SE here is a bad idea, we should do this else where so it is not redone on every iteration
        let mut greens_functions = GreensFunctionBuilder::new()
            .with_info_desk(self.info_desk)
            .with_mesh(self.mesh)
            .with_spectral_discretisation(self.spectral)
            .build();
        let mut self_energies = SelfEnergyBuilder::new()
            .with_mesh(self.mesh)
            .with_spectral_discretisation(self.spectral)
            .build();
        let mut inner_loop =
            self.build_incoherent_inner_loop(&mut greens_functions, &mut self_energies);
        let mut charge_and_currents = self.tracker.charge_and_currents.clone();
        inner_loop.run_loop(&mut charge_and_currents)?;
        let _ = std::mem::replace(self.tracker.charge_and_currents_mut(), charge_and_currents);
        Ok(())
    }

    fn run_loop(&mut self, mut potential: Potential<T>) -> color_eyre::Result<()> {
        let mut iteration = 0;
        while !self.is_loop_converged(&mut potential)? {
            self.single_iteration()?;
            iteration += 1;
            if iteration >= self.convergence_settings.maximum_outer_iterations() {
                return Err(color_eyre::eyre::eyre!(
                    "Reached maximum iteration count in the inner loop"
                ));
            }
        }
        Ok(())
    }
}

use crate::greens_functions::{AggregateGreensFunctions, GreensFunctionBuilder};
use crate::inner_loop::InnerLoopBuilder;
use transporter_poisson::PoissonSourceBuilder;

impl<T, GeometryDim, Conn, BandDim, SpectralSpace>
    OuterLoop<'_, T, GeometryDim, Conn, BandDim, SpectralSpace>
where
    T: RealField + Copy,
    Conn: Connectivity<T, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>,
{
    fn update_potential(
        &self,
        previous_potential: &Potential<T>,
    ) -> color_eyre::Result<Potential<T::RealField>> {
        // Calculate the Fermi level
        let _fermi_level = self.info_desk.determine_fermi_level(
            self.mesh,
            previous_potential,
            self.tracker.charge_as_ref(),
        );

        let source_vector: DVector<T> = self
            .info_desk
            .calculate_source_vector(self.mesh, self.tracker.charge_as_ref());

        //let _poisson_problem = PoissonSourceBuilder::new()
        //    .with_mesh(self.mesh)
        //    .with_source(&source_vector)
        //    .build();

        todo!()
    }
}

impl<T, GeometryDim, Conn, BandDim>
    OuterLoop<'_, T, GeometryDim, Conn, BandDim, SpectralSpace<T, ()>>
where
    T: RealField + Copy,
    Conn: Connectivity<T, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>,
{
    fn build_coherent_inner_loop<'a>(
        &'a self,
        greens_functions: &'a mut AggregateGreensFunctions<
            'a,
            T,
            CsrMatrix<Complex<T>>,
            GeometryDim,
            BandDim,
        >,
        self_energies: &'a mut SelfEnergy<T, GeometryDim, Conn, CsrMatrix<Complex<T>>>,
    ) -> InnerLoop<'a, T, GeometryDim, Conn, CsrMatrix<Complex<T>>, SpectralSpace<T, ()>, BandDim>
    {
        InnerLoopBuilder::new()
            .with_convergence_settings(self.convergence_settings)
            .with_mesh(self.mesh)
            .with_spectral_discretisation(self.spectral)
            .with_hamiltonian(self.hamiltonian)
            .with_greens_functions(greens_functions)
            .with_self_energies(self_energies)
            .build()
    }
}

impl<T, GeometryDim, Conn, BandDim>
    OuterLoop<
        '_,
        T,
        GeometryDim,
        Conn,
        BandDim,
        SpectralSpace<T, WavevectorSpace<T, GeometryDim, Conn>>,
    >
where
    T: RealField + Copy,
    Conn: Connectivity<T, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>,
{
    fn build_incoherent_inner_loop<'a>(
        &'a self,
        greens_functions: &'a mut AggregateGreensFunctions<
            'a,
            T,
            DMatrix<Complex<T>>,
            GeometryDim,
            BandDim,
        >,
        self_energies: &'a mut SelfEnergy<T, GeometryDim, Conn, DMatrix<Complex<T>>>,
    ) -> InnerLoop<
        'a,
        T,
        GeometryDim,
        Conn,
        DMatrix<Complex<T>>,
        SpectralSpace<T, WavevectorSpace<T, GeometryDim, Conn>>,
        BandDim,
    > {
        InnerLoopBuilder::new()
            .with_convergence_settings(self.convergence_settings)
            .with_mesh(self.mesh)
            .with_spectral_discretisation(self.spectral)
            .with_hamiltonian(self.hamiltonian)
            .with_greens_functions(greens_functions)
            .with_self_energies(self_energies)
            .build()
    }
}

use crate::device::info_desk::DeviceInfoDesk;
use crate::postprocessor::Charge;
use transporter_mesher::Mesh;

trait OuterLoopInfoDesk<T: Copy + RealField, GeometryDim: SmallDim, Conn, BandDim: SmallDim>
where
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, GeometryDim>,
{
    fn determine_fermi_level(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        potential: &Potential<T>,
        charge: &Charge<T, BandDim>,
    ) -> Vec<T>;

    fn calculate_source_vector(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        charge: &Charge<T, BandDim>,
    ) -> DVector<T>;

    fn compute_jacobian_diagonal(
        &self,
        fermi_level: &DVector<T>,
        potential: &Potential<T>,
        mesh: &Mesh<T, GeometryDim, Conn>,
    ) -> DVector<T>;
}

impl<T: Copy + RealField, GeometryDim: SmallDim, Conn, BandDim: SmallDim>
    OuterLoopInfoDesk<T, GeometryDim, Conn, BandDim> for DeviceInfoDesk<T, GeometryDim, BandDim>
where
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<T, GeometryDim>,
{
    // Rebasing from elements to vertices
    fn determine_fermi_level(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        potential: &Potential<T>,
        charge: &Charge<T, BandDim>,
    ) -> Vec<T> {
        // Find the net charge in the multiband system
        let fermi_at_elements = charge
            .net_charge()
            .into_iter()
            .zip(potential.as_ref().iter())
            .zip(mesh.elements().iter())
            .map(|((&n, &phi), element)| {
                let region = element.1;
                let n3d = (T::one() + T::one()) // Currently always getting the x-component, is this dumb?
                    * (self.effective_masses[region][0][0] // The conduction band is always supposed to be in position 0
                        * T::from_f64(crate::constants::ELECTRON_MASS).unwrap()
                        * T::from_f64(crate::constants::BOLTZMANN).unwrap()
                        * self.temperature
                        / T::from_f64(crate::constants::HBAR).unwrap().powi(2)
                        / (T::one() + T::one())
                        / T::from_f64(std::f64::consts::PI).unwrap())
                    .powf(T::from_f64(1.5).unwrap());

                let (factor, gamma) = (
                    T::from_f64(crate::constants::ELECTRON_CHARGE / crate::constants::BOLTZMANN)
                        .unwrap()
                        / self.temperature,
                    T::from_f64(std::f64::consts::PI.sqrt() / 2.).unwrap(),
                );
                let eta_f = crate::fermi::inverse_fermi_integral_05(gamma * n / n3d);

                let ef_minus_ec = eta_f / factor;

                let band_offset = self.band_offsets[region][0]; // Again we assume the band offset for the c-band is in position 0
                ef_minus_ec + band_offset - phi // TODO should this be a plus phi or a minus phi??
            })
            .collect::<Vec<_>>();
        // Move to a result evaluated at the vertices by averaging
        let mut result = vec![fermi_at_elements[0]];
        let mut core = fermi_at_elements
            .windows(2)
            .map(|x| (x[0] + x[1]) / (T::one() + T::one()))
            .collect::<Vec<_>>();
        result.append(&mut core);
        result.push(fermi_at_elements[fermi_at_elements.len() - 1]);
        result
    }

    fn calculate_source_vector(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        charge: &Charge<T, BandDim>,
    ) -> DVector<T> {
        let net_charge = charge.net_charge();

        let net_charge = net_charge
            .iter()
            .zip(mesh.elements())
            .map(|(&n, element)| {
                let region = element.1;
                let acceptor_density = self.acceptor_densities[region];
                let donor_density = self.donor_densities[region];
                n + acceptor_density - donor_density
            })
            .collect::<Vec<_>>();

        // Move to a result evaluated at the vertices by averaging
        let mut result = vec![net_charge[0]];
        let mut core = net_charge
            .windows(2)
            .map(|x| (x[0] + x[1]) / (T::one() + T::one()))
            .collect::<Vec<_>>();
        result.append(&mut core);
        result.push(net_charge[net_charge.len() - 1]);
        DVector::from(result)
    }

    // A naive implementation of Eq C4 from Lake et al (1997)
    // TODO can improve using Eq. B3 of guo (2014), constructing from the GFs
    fn compute_jacobian_diagonal(
        &self,
        fermi_level: &DVector<T>,
        potential: &Potential<T>,
        mesh: &Mesh<T, GeometryDim, Conn>,
    ) -> DVector<T>
    where
        DefaultAllocator: Allocator<[T; 3], BandDim>,
    {
        DVector::from(
            mesh.vertices()
                .iter()
                .zip(fermi_level.iter())
                .zip(potential.as_ref().iter())
                .map(|((vertex, &fermi_level), &potential)| {
                    let region = &vertex.1;
                    let (band_offset, effective_mass) = match region {
                        Assignment::Boundary(x) => (
                            x.iter()
                                .fold(T::zero(), |acc, &i| acc + self.band_offsets[i][0])
                                / T::from_usize(x.len()).unwrap(),
                            x.iter()
                                .fold(T::zero(), |acc, &i| acc + self.effective_masses[i][0][0])
                                / T::from_usize(x.len()).unwrap(),
                        ),
                        Assignment::Core(x) => {
                            (self.band_offsets[*x][0], self.effective_masses[*x][0][0])
                        }
                    };

                    let n3d = (T::one() + T::one())
                        * (effective_mass
                            * T::from_f64(crate::constants::ELECTRON_MASS).unwrap()
                            * T::from_f64(crate::constants::BOLTZMANN).unwrap()
                            * self.temperature
                            / T::from_f64(crate::constants::HBAR).unwrap().powi(2)
                            / (T::one() + T::one())
                            / T::from_f64(std::f64::consts::PI).unwrap())
                        .powf(T::from_f64(1.5).unwrap());

                    T::from_f64(crate::constants::ELECTRON_CHARGE / crate::constants::BOLTZMANN)
                        .unwrap()
                        / self.temperature
                        * n3d
                        * crate::fermi::fermi_integral_m05(
                            T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
                                * (fermi_level - band_offset + potential)
                                / T::from_f64(crate::constants::BOLTZMANN).unwrap()
                                / self.temperature,
                        )
                })
                .collect::<Vec<_>>(),
        )
    }
}
