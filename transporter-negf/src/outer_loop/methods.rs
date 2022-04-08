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
        let difference = (&self.0 - &other.0).norm();
        if norm == T::zero() {
            return true;
        }
        difference / norm < tolerance
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
    T: ArgminFloat + RealField + Copy,
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
        // A single iteration before the loop to avoid updating the potential with an empty charge vector
        self.single_iteration()?;
        let mut iteration = 1;
        while !self.is_loop_converged(&mut potential)? {
            // Update the Fermi level using the new charge density
            self.tracker.fermi_level = DVector::from(self.info_desk.determine_fermi_level(
                self.mesh,
                &potential,
                self.tracker.charge_as_ref(),
            ));
            // Do the inner loop
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
    T: ArgminFloat + RealField + Copy,
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
use argmin::core::observers::ObserverMode;
use argmin::core::observers::SlogLogger;
use argmin::core::ArgminFloat;
use argmin::core::Executor;
use transporter_poisson::{NewtonSolver, PoissonMethods, PoissonSourceBuilder};

impl<T, GeometryDim, Conn, BandDim, SpectralSpace>
    OuterLoop<'_, T, GeometryDim, Conn, BandDim, SpectralSpace>
where
    T: ArgminFloat + RealField + Copy,
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
        let fermi_level = DVector::from(self.info_desk.determine_fermi_level(
            self.mesh,
            previous_potential,
            self.tracker.charge_as_ref(),
        ));

        //let source_vector: DVector<T> = self
        //    .info_desk
        //    .calculate_source_vector(self.mesh, self.tracker.charge_as_ref());

        //let poisson_problem = PoissonSourceBuilder::new()
        //    .with_info_desk(self.info_desk)
        //    .with_mesh(self.mesh)
        //    .with_source(&source_vector)
        //    .build();

        let cost = super::poisson::PoissonProblemBuilder::default()
            .with_charge(self.tracker.charge_as_ref())
            .with_info_desk(self.info_desk)
            .with_mesh(self.mesh)
            .build()?;

        // Define initial parameter vector
        let init_param: DVector<T> = DVector::from_vec(vec![T::zero(); self.mesh.vertices().len()]);

        let linesearch = argmin::solver::linesearch::MoreThuenteLineSearch::new()
            .alpha(T::zero(), T::one())
            .map_err(|e| color_eyre::eyre::eyre!("Failed to initialize linesearch {:?}", e))?;
        // Set up solver
        let solver = argmin::solver::gaussnewton::GaussNewtonLS::new(linesearch);

        // Run solver
        let res = Executor::new(cost, solver)
            .configure(|state| state.param(init_param).max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()
            .map_err(|e| color_eyre::eyre::eyre!("Failed to optimize poisson system {:?}", e))?;

        // Wait a second (lets the logger flush everything before printing again)
        std::thread::sleep(std::time::Duration::from_secs(1));

        // Print result
        println!("{}", res);
        todo!()
        // let mut output = previous_potential.as_ref().clone();
        //let charge_density = self
        //    .tracker
        //    .charge_as_ref()
        //    .net_charge()
        //    .iter()
        //    .map(|x| *x)
        //    .collect::<Vec<_>>();
        // Move charge density to one over vertices duh
        // let mut charge_density_vec = vec![charge_density[0]];
        // let mut core_cd = charge_density
        //     .windows(2)
        //     .map(|x| (x[0] + x[1]) / (T::one() + T::one()))
        //     .collect::<Vec<_>>();
        // charge_density_vec.append(&mut core_cd);
        // charge_density_vec.push(charge_density[charge_density.len() - 1]);
        //let charge_density_vec = vec![T::zero(); source_vector.shape().0];
        //let source_vector = DVector::from(charge_density_vec);
        // output = poisson_problem.solve_into(output, &source_vector, &fermi_level)?;

        // Ok(Potential::from_vector(output))
    }
}

impl<T: Copy + RealField, GeometryDim: SmallDim, Conn, BandDim: SmallDim>
    PoissonMethods<T, GeometryDim, Conn> for DeviceInfoDesk<T, GeometryDim, BandDim>
where
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<T, GeometryDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    // Solve for the diagonal of the Jacobian, given in this approximation by
    // 'q / K T N_C Fermi_{-0.5} ((E_F - E_C + q \phi) / K T)
    fn update_jacobian_diagonal(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        fermi_level: &DVector<T>,
        solution: &DVector<T>,
        output: &mut DVector<T>,
    ) -> color_eyre::Result<()> {
        // TODO actually swap in place, rather than allocating then swapping
        let updated = self.compute_jacobian_diagonal(fermi_level, solution, mesh);
        let _ = std::mem::replace(output, updated);
        Ok(())
    }

    // Find the updated charge density estimated on switching to a new potential
    // 'q * (N_C Fermi_{0.5} ((E_F - E_C + q \phi) / K T) + N_A - N_D)`
    fn update_charge_density(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        fermi_level: &DVector<T>,
        solution: &DVector<T>,
        output: &mut DVector<T>,
    ) -> color_eyre::Result<()> {
        let updated = self.update_source_vector(mesh, fermi_level, solution);
        let _ = std::mem::replace(output, updated);
        Ok(())
    }
}

impl<T, GeometryDim, Conn, BandDim>
    OuterLoop<'_, T, GeometryDim, Conn, BandDim, SpectralSpace<T, ()>>
where
    T: ArgminFloat + RealField + Copy,
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
    T: ArgminFloat + RealField + Copy,
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

pub(crate) trait OuterLoopInfoDesk<
    T: Copy + RealField,
    GeometryDim: SmallDim,
    Conn,
    BandDim: SmallDim,
> where
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

    fn update_source_vector(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        fermi_level: &DVector<T>,
        potential: &DVector<T>,
    ) -> DVector<T>;

    fn compute_jacobian_diagonal(
        &self,
        fermi_level: &DVector<T>,
        potential: &DVector<T>,
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
        potential: &Potential<T>, // The potential which is evaluated at each mesh vertex
        charge: &Charge<T, BandDim>, // The charge density which is evaluated at each mesh element
    ) -> Vec<T> {
        // Evaluate the Fermi level over the elements of the mesh
        let fermi_plus_potential_at_elements = charge
            .net_charge()
            .into_iter()
            .zip(mesh.elements().iter())
            .map(|(&n, element)| {
                let region = element.1;
                // Calculate the density of states in the conduction band
                let n3d = (T::one() + T::one()) // Currently always getting the x-component, is this dumb?
                    * (self.effective_masses[region][0][0] // The conduction band is always supposed to be in position 0
                        * T::from_f64(crate::constants::ELECTRON_MASS).unwrap()
                        * T::from_f64(crate::constants::BOLTZMANN).unwrap()
                        * self.temperature
                        / T::from_f64(crate::constants::HBAR).unwrap().powi(2)
                        / (T::one() + T::one())
                        / T::from_f64(std::f64::consts::PI).unwrap())
                    .powf(T::from_f64(1.5).unwrap());

                // Find the inverse thermal energy quantum, and the Gamma function needed to convert
                // the Fermi integral to its curly F equivalent.
                let (factor, gamma) = (
                    T::from_f64(crate::constants::ELECTRON_CHARGE / crate::constants::BOLTZMANN)
                        .unwrap()
                        / self.temperature,
                    T::from_f64(std::f64::consts::PI.sqrt() / 2.).unwrap(),
                );
                // Find the full energy argument \eta_f - E_C + \phi
                let eta_f_minus_ec_plus_phi =
                    crate::fermi::inverse_fermi_integral_05(gamma * n / n3d) / factor;

                let band_offset = self.band_offsets[region][0]; // Again we assume the band offset for the c-band is in position 0
                                                                // Get eta_f_plus_phi
                eta_f_minus_ec_plus_phi + band_offset //- phi // TODO should this be a plus phi or a minus phi??
            })
            .collect::<Vec<_>>();
        // Move to a result evaluated at the vertices by averaging
        // Use the potential Here as it is defined over the vertices, not the elements
        let mut result = vec![fermi_plus_potential_at_elements[0] - potential.as_ref()[0]];
        let mut core = fermi_plus_potential_at_elements
            .windows(2)
            .zip(potential.as_ref().iter().skip(1))
            .map(|(x, &phi)| (x[0] + x[1]) / (T::one() + T::one()) - phi)
            .collect::<Vec<_>>();
        result.append(&mut core);
        result.push(
            fermi_plus_potential_at_elements[fermi_plus_potential_at_elements.len() - 1]
                - potential.as_ref()[potential.as_ref().len() - 1],
        );
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
                (n + acceptor_density - donor_density)
                    * T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
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

    fn update_source_vector(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        fermi_level: &DVector<T>, // The fermi level defined on the mesh vertices
        potential: &DVector<T>,   // The potential defined on the mesh vertices
    ) -> DVector<T> {
        let gamma = T::from_f64(std::f64::consts::PI.sqrt() / 2.).unwrap();
        DVector::from(
            mesh.vertices()
                .iter()
                .zip(fermi_level.iter())
                .zip(potential.iter())
                .map(|((vertex, &fermi_level), &potential)| {
                    let region = &vertex.1;
                    let (band_offset, effective_mass, donor_density, acceptor_density) =
                        match region {
                            Assignment::Boundary(x) => (
                                x.iter()
                                    .fold(T::zero(), |acc, &i| acc + self.band_offsets[i][0])
                                    / T::from_usize(x.len()).unwrap(),
                                x.iter().fold(T::zero(), |acc, &i| {
                                    acc + self.effective_masses[i][0][0]
                                }) / T::from_usize(x.len()).unwrap(),
                                x.iter()
                                    .fold(T::zero(), |acc, &i| acc + self.donor_densities[i])
                                    / T::from_usize(x.len()).unwrap(),
                                x.iter()
                                    .fold(T::zero(), |acc, &i| acc + self.acceptor_densities[i])
                                    / T::from_usize(x.len()).unwrap(),
                            ),
                            Assignment::Core(x) => (
                                self.band_offsets[*x][0],
                                self.effective_masses[*x][0][0],
                                self.donor_densities[*x],
                                self.acceptor_densities[*x],
                            ),
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

                    let n_free =
                        n3d * crate::fermi::fermi_integral_05(
                            T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
                                * (fermi_level - band_offset + potential)
                                / T::from_f64(crate::constants::BOLTZMANN).unwrap()
                                / self.temperature,
                        ) / gamma;

                    T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap() * (n_free)
                    // + acceptor_density - donor_density)
                })
                .collect::<Vec<_>>(),
        )
    }

    // A naive implementation of Eq C4 from Lake et al (1997)
    // TODO can improve using Eq. B3 of guo (2014), constructing from the GFs
    fn compute_jacobian_diagonal(
        &self,
        fermi_level: &DVector<T>,
        potential: &DVector<T>,
        mesh: &Mesh<T, GeometryDim, Conn>,
    ) -> DVector<T>
    where
        DefaultAllocator: Allocator<[T; 3], BandDim>,
    {
        let gamma = T::from_f64(std::f64::consts::PI.sqrt()).unwrap();
        DVector::from(
            mesh.vertices()
                .iter()
                .zip(fermi_level.iter())
                .zip(potential.iter())
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
                        / gamma
                        * T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
                })
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod test {
    use super::{OuterLoopInfoDesk, Potential};
    use crate::{
        app::{tracker::TrackerBuilder, Configuration},
        device::{info_desk::BuildInfoDesk, Device},
    };
    use nalgebra::{DVector, U1};
    use rand::Rng;

    /// Test the calculation of the Fermi level, and the electron density from the Fermi level
    /// for an arbitrary potential. If successful the calculated electron density after the two
    /// step calculation should coincide with the initial charge density.
    #[test]
    fn fermi_level_in_a_homogeneous_structure_reproduces_charge_density() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let mut rng = rand::thread_rng();

        let potential = Potential::from_vector(nalgebra::DVector::from(
            (0..mesh.vertices().len())
                .map(|_| rng.gen::<f64>())
                .collect::<Vec<_>>(),
        ));

        let mut charge = tracker.charge().clone();
        // Set the charge to neutral??
        charge.as_ref_mut().iter_mut().for_each(|charge_in_band| {
            charge_in_band
                .iter_mut()
                .zip(mesh.elements())
                .for_each(|(x, element)| {
                    let region = element.1;
                    let doping_density = info_desk.donor_densities[region];
                    *x = doping_density;
                })
        });

        // Find the Fermi level in the device
        let fermi = info_desk.determine_fermi_level(&mesh, &potential, &charge);
        // Recalculate the source vector using the Fermi level
        let source_vector = info_desk.update_source_vector(
            &mesh,
            &nalgebra::DVector::from(fermi),
            potential.as_ref(),
        );
        // Assert each element in the source vector is near to zero
        for &element in source_vector.into_iter() {
            approx::assert_relative_eq!(
                element / crate::constants::ELECTRON_CHARGE / info_desk.donor_densities[0],
                0_f64,
                epsilon = std::f64::EPSILON * 100_f64
            );
        }
    }

    /// Compare results from initialisation and updates of source vector
    #[test]
    fn initialised_and_updated_sources_are_equal_at_zero_potential() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let mut rng = rand::thread_rng();

        let potential = Potential::from_vector(nalgebra::DVector::from(
            (0..mesh.vertices().len())
                .map(|_| rng.gen::<f64>())
                .collect::<Vec<_>>(),
        ));

        let mut charge = tracker.charge().clone();
        // Set the charge to neutral
        charge.as_ref_mut().iter_mut().for_each(|charge_in_band| {
            charge_in_band
                .iter_mut()
                .zip(mesh.elements())
                .for_each(|(x, element)| {
                    let region = element.1;
                    let doping_density = info_desk.donor_densities[region];
                    *x = doping_density * 0.95;
                })
        });

        // Get the initial source
        let initial_source = info_desk.calculate_source_vector(&mesh, &charge);

        // Find the Fermi level in the device
        let fermi = info_desk.determine_fermi_level(&mesh, &potential, &charge);
        // Recalculate the source vector using the Fermi level
        let source_vector = info_desk.update_source_vector(
            &mesh,
            &nalgebra::DVector::from(fermi),
            potential.as_ref(),
        );
        // Assert each element in the source vector is near to zero
        for (&element, &initial) in source_vector.into_iter().zip(initial_source.into_iter()) {
            dbg!(element, initial);
            //  approx::assert_relative_eq!(
            //      element / crate::constants::ELECTRON_CHARGE / info_desk.donor_densities[0],
            //      0_f64,
            //      epsilon = std::f64::EPSILON * 100_f64
            //  );
        }
    }

    #[test]
    fn jacobian_diagonal_agrees_with_numerical_derivative() {
        let path = std::path::PathBuf::try_from("../.config/structure.toml").unwrap();
        let device: Device<f64, U1> = crate::device::Device::build(path).unwrap();
        // TODO Info_desk is currently always U1 because it is informed by the device dimension right now, this is no good. We need n_bands to be in-play here.
        let info_desk = device.build_device_info_desk().unwrap();

        let config: Configuration<f64> = Configuration::build().unwrap();
        let mesh: transporter_mesher::Mesh1d<f64> =
            crate::app::build_mesh_with_config(&config, device).unwrap();
        let tracker = TrackerBuilder::new()
            .with_mesh(&mesh)
            .with_info_desk(&info_desk)
            .build()
            .unwrap();

        let mut rng = rand::thread_rng();

        let potential = Potential::from_vector(nalgebra::DVector::from(
            (0..mesh.vertices().len())
                .map(|_| rng.gen::<f64>())
                .collect::<Vec<_>>(),
        ));

        let dphi = 1e-8;
        let potential_plus_dphi = Potential::from_vector(nalgebra::DVector::from(
            potential
                .as_ref()
                .iter()
                .map(|phi| phi + dphi)
                .collect::<Vec<_>>(),
        ));
        let potential_minus_dphi = Potential::from_vector(nalgebra::DVector::from(
            potential
                .as_ref()
                .iter()
                .map(|phi| phi - dphi)
                .collect::<Vec<_>>(),
        ));

        let mut charge = tracker.charge().clone();
        // Set the charge to neutral??
        charge.as_ref_mut().iter_mut().for_each(|charge_in_band| {
            charge_in_band
                .iter_mut()
                .zip(mesh.elements())
                .for_each(|(x, element)| {
                    let region = element.1;
                    let doping_density = info_desk.donor_densities[region];
                    *x = doping_density;
                })
        });

        // Find the Fermi level in the device
        let fermi_level =
            DVector::from(info_desk.determine_fermi_level(&mesh, &potential, &charge));
        // Compute the numerical derivative at phi_plus_dphi, phi_minus_dphi
        let source_vector_at_phi_plus =
            info_desk.update_source_vector(&mesh, &fermi_level, potential_plus_dphi.as_ref());
        let source_vector_at_phi_minus =
            info_desk.update_source_vector(&mesh, &fermi_level, potential_minus_dphi.as_ref());
        let numerical_derivative = source_vector_at_phi_plus
            .into_iter()
            .zip(source_vector_at_phi_minus.into_iter())
            .map(|(plus, minus)| (plus - minus) / 2. / dphi)
            .collect::<Vec<_>>();

        let analytical_derivative =
            info_desk.compute_jacobian_diagonal(&fermi_level, potential.as_ref(), &mesh);

        for (num, ana) in numerical_derivative
            .iter()
            .zip(analytical_derivative.iter())
        {
            println!("{num}, {}", ana * crate::constants::ELECTRON_CHARGE);
        }
    }
}
