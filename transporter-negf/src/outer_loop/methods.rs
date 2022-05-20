use super::{OuterLoop, OuterLoopError};
use crate::{
    app::Calculation,
    greens_functions::{AggregateGreensFunctions, GreensFunctionBuilder, MMatrix},
    inner_loop::{Inner, InnerLoop, InnerLoopBuilder},
    self_energy::{SelfEnergy, SelfEnergyBuilder},
};
use argmin::core::{ArgminOp, Executor};
use conflux::{core::FixedPointSolver, solvers::anderson::Type1AndersonMixer};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use ndarray::{Array1, Array2};
use ndarray_stats::DeviationExt;
use num_complex::Complex;
use num_traits::ToPrimitive;
use std::fs::OpenOptions;
use std::io::Write;
use transporter_mesher::{Connectivity, SmallDim};

/// A wrapper for the calculated electrostatic potential
#[derive(Clone, Debug)]
pub(crate) struct Potential<T: RealField>(Array1<T>);

impl<T: Copy + RealField + ToPrimitive> Potential<T> {
    pub(crate) fn from_vector(vector: Array1<T>) -> Self {
        Self(vector)
    }
    /// Check whether the change in the normalised potential is within the requested tolerance
    fn is_change_within_tolerance(&self, other: &Potential<T>, tolerance: T) -> bool {
        let norm = self
            .0
            .iter()
            .fold(T::zero(), |acc, &x| acc + x.powi(2))
            .sqrt();
        let difference = T::from_f64(self.0.l2_dist(&other.0).unwrap()).unwrap();
        if norm == T::zero() {
            return true;
        }
        difference / norm < tolerance
    }
}

impl<T: RealField> AsRef<Array1<T>> for Potential<T> {
    fn as_ref(&self) -> &Array1<T> {
        &self.0
    }
}

impl<T: Copy + RealField> Potential<T> {
    pub(crate) fn get(&self, vertex_index: usize) -> T {
        self.0[vertex_index]
    }
}

pub(crate) trait Outer {
    /// Compute the updated electric potential and confirm
    /// whether the change is within tolerance of the values on the
    /// previous loop iteration
    fn is_loop_converged(
        &self,
        previous_potential: &mut Potential<f64>,
    ) -> Result<bool, OuterLoopError<f64>>;
    /// Carry out a single iteration of the self-consistent inner loop
    fn single_iteration(
        &mut self,
        potential: &Array1<f64>,
    ) -> Result<Array1<f64>, OuterLoopError<f64>>;
    /// Run the self-consistent inner loop to convergence
    fn run_loop(&mut self, potential: Potential<f64>) -> Result<(), OuterLoopError<f64>>;
    fn potential_owned(&self) -> Potential<f64>;
}

impl<GeometryDim, Conn, BandDim> Outer
    for OuterLoop<'_, f64, GeometryDim, Conn, BandDim, SpectralSpace<f64, ()>>
where
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<Array1<f64>, BandDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    fn is_loop_converged(
        &self,
        previous_potential: &mut Potential<f64>,
    ) -> Result<bool, OuterLoopError<f64>> {
        let potential = self.update_potential(previous_potential)?.0;
        let result = potential.is_change_within_tolerance(
            previous_potential,
            self.convergence_settings.outer_tolerance(),
        );
        let _ = std::mem::replace(previous_potential, potential);
        Ok(result)
    }
    /// Carry out a single iteration of the self-consistent outer loop
    fn single_iteration(
        &mut self,
        previous_potential: &Array1<f64>,
    ) -> Result<Array1<f64>, OuterLoopError<f64>> {
        // Build the inner loop, if we are running a ballistic calculation or have not arrived
        // at an initial converged ballistic solution then we create a
        // coherent inner loop, with sparse matrices, else we create a dense one.
        // Update the Fermi level in the device
        // self.tracker.fermi_level = Array1::from(self.info_desk.determine_fermi_level(
        //     self.mesh,
        //     &Potential::from_vector(previous_potential.clone()),
        //     self.tracker.charge_as_ref(),
        // ));
        // Put the new potential into the tracker so the GF can see it.
        self.tracker
            .update_potential(Potential::from_vector(previous_potential.clone()));

        // TODO Building the gfs and SE here is a bad idea, we should do this else where so it is not redone on every iteration
        self.term.move_cursor_to(0, 5)?;
        self.term.clear_to_end_of_screen()?;
        tracing::trace!("Initialising Greens Functions");
        let mut greens_functions = GreensFunctionBuilder::default()
            .with_info_desk(self.info_desk)
            .with_mesh(self.mesh)
            .with_spectral_discretisation(self.spectral)
            .build()?;
        self.term.move_cursor_to(0, 5)?;
        self.term.clear_to_end_of_screen()?;
        tracing::trace!("Initialising Self Energies");
        let mut self_energies = SelfEnergyBuilder::default()
            .with_mesh(self.mesh)
            .with_spectral_discretisation(self.spectral)
            .build_coherent()?;

        // Todo Get the new potential into the new hamiltonian...
        self.hamiltonian
            .update_potential(&self.tracker, self.mesh)?;
        let mut inner_loop =
            self.build_coherent_inner_loop(&mut greens_functions, &mut self_energies);
        let mut charge_and_currents = self.tracker.charge_and_currents.clone();

        inner_loop
            .run_loop(&mut charge_and_currents)
            .expect("Inner loop failed");
        let _ = std::mem::replace(self.tracker.charge_and_currents_mut(), charge_and_currents);

        // Update the Fermi level in the device
        self.tracker.fermi_level = Array1::from(self.info_desk.determine_fermi_level(
            self.mesh,
            &Potential::from_vector(previous_potential.clone()),
            self.tracker.charge_as_ref(),
        ));

        let (potential, residual) = self
            .update_potential(&Potential::from_vector(previous_potential.clone()))
            .expect("Potential update failed");
        self.tracker.current_residual = residual;

        Ok(potential.as_ref().clone())
    }

    #[tracing::instrument(name = "Outer loop", skip_all)]
    fn run_loop(&mut self, mut potential: Potential<f64>) -> Result<(), OuterLoopError<f64>> {
        let mixer = Type1AndersonMixer::new(
            potential.as_ref().len(),
            self.convergence_settings.outer_tolerance(),
            self.convergence_settings.maximum_outer_iterations() as u64,
        )
        .beta(1_f64)
        .memory(2);
        // let vec_para = potential.as_ref().iter().copied().collect::<Vec<_>>();
        // let initial_parameter = ndarray::Array1::from(vec_para);
        let initial_parameter = potential.as_ref().clone();
        // let mut solver = FixedPointSolver::new(mixer, potential.as_ref().clone());
        let mut solver = FixedPointSolver::new(mixer, initial_parameter);

        self.term.move_cursor_to(0, 2)?;
        self.term.clear_to_end_of_screen()?;
        tracing::info!("Outer self-consistent loop with Anderson mixing");
        let solution = solver.run(self)?;

        let solution = Array1::from(solution.get_param().iter().copied().collect::<Vec<_>>());
        potential = Potential::from_vector(solution);
        self.tracker.update_potential(potential);
        //// A single iteration before the loop to avoid updating the potential with an empty charge vector
        // Postprocessing steps
        self.tracker.write_to_file("coherent")?;
        Ok(())
    }

    fn potential_owned(&self) -> Potential<f64> {
        self.tracker.potential.clone()
    }
}

use crate::spectral::{SpectralSpace, WavevectorSpace};

impl<GeometryDim, Conn, BandDim> Outer
    for OuterLoop<
        '_,
        f64,
        GeometryDim,
        Conn,
        BandDim,
        SpectralSpace<f64, WavevectorSpace<f64, GeometryDim, Conn>>,
    >
where
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<Array1<f64>, BandDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    fn is_loop_converged(
        &self,
        previous_potential: &mut Potential<f64>,
    ) -> Result<bool, OuterLoopError<f64>> {
        let potential = self.update_potential(previous_potential)?.0;
        let result = potential.is_change_within_tolerance(
            previous_potential,
            self.convergence_settings.outer_tolerance(),
        );
        let _ = std::mem::replace(previous_potential, potential);
        Ok(result)
    }
    /// Carry out a single iteration of the self-consistent outer loop
    fn single_iteration(
        &mut self,
        previous_potential: &Array1<f64>,
    ) -> Result<Array1<f64>, OuterLoopError<f64>> {
        self.tracker
            .update_potential(Potential::from_vector(previous_potential.clone()));

        // TODO Building the gfs and SE here is a bad idea, we should do this else where so it is not redone on every iteration
        match self.tracker.calculation {
            Calculation::Coherent { voltage_target: _ } => {
                dbg!("Coherent Path");
                self.term.move_cursor_to(0, 5)?;
                self.term.clear_to_end_of_screen()?;
                tracing::trace!("Initialising Greens Functions");
                let mut greens_functions = GreensFunctionBuilder::default()
                    .with_info_desk(self.info_desk)
                    .with_mesh(self.mesh)
                    .with_spectral_discretisation(self.spectral)
                    .build()?;
                self.term.move_cursor_to(0, 5)?;
                self.term.clear_to_end_of_screen()?;
                tracing::trace!("Initialising Self Energies");
                let mut self_energies = SelfEnergyBuilder::default()
                    .with_mesh(self.mesh)
                    .with_spectral_discretisation(self.spectral)
                    .build_coherent()?;

                // Todo Get the new potential into the new hamiltonian...
                self.hamiltonian
                    .update_potential(&self.tracker, self.mesh)?;

                let mut inner_loop =
                    self.build_coherent_inner_loop(&mut greens_functions, &mut self_energies);
                let mut charge_and_currents = self.tracker.charge_and_currents.clone();
                inner_loop
                    .run_loop(&mut charge_and_currents)
                    .expect("Inner loop failed");
                let _ =
                    std::mem::replace(self.tracker.charge_and_currents_mut(), charge_and_currents);
            }
            Calculation::Incoherent { voltage_target: _ } => {
                self.term.move_cursor_to(0, 6)?;
                self.term.clear_to_end_of_screen()?;
                tracing::trace!("Initialising Greens Functions");
                let mut greens_functions = GreensFunctionBuilder::default()
                    .with_info_desk(self.info_desk)
                    .with_mesh(self.mesh)
                    .with_spectral_discretisation(self.spectral)
                    .incoherent_calculation(&Calculation::Incoherent {
                        voltage_target: 0_f64,
                    })
                    // .build()?;
                    .build_mixed()?;
                self.term.move_cursor_to(0, 6)?;
                self.term.clear_to_end_of_screen()?;
                tracing::trace!("Initialising Self Energies");
                let mut self_energies = SelfEnergyBuilder::default()
                    .with_mesh(self.mesh)
                    .with_spectral_discretisation(self.spectral)
                    .build_incoherent(self.info_desk.lead_length)?; // We can only do an incoherent calculation with leads at the moment
                                                                    // Todo Get the new potential into the new hamiltonian...
                self.hamiltonian
                    .update_potential(&self.tracker, self.mesh)?;

                let mut charge_and_currents = self.tracker.charge_and_currents.clone();
                let mut inner_loop = self
                    .build_incoherent_inner_loop_mixed(&mut greens_functions, &mut self_energies);
                inner_loop
                    .run_loop(&mut charge_and_currents)
                    .expect("Inner loop failed");

                let mut file = OpenOptions::new()
                    .write(true)
                    .append(true)
                    .create(true)
                    .open("../results/scattering_rate.txt")
                    .unwrap();
                writeln!(
                    file,
                    "{}, {}, {}, {}",
                    self.tracker.voltage,
                    self.tracker.scattering_scaling,
                    inner_loop.rate.unwrap().re,
                    inner_loop.rate.unwrap().im,
                )?;

                let _ =
                    std::mem::replace(self.tracker.charge_and_currents_mut(), charge_and_currents);
            }
        }

        // Update the Fermi level in the device
        self.tracker.fermi_level = Array1::from(self.info_desk.determine_fermi_level(
            self.mesh,
            &Potential::from_vector(previous_potential.clone()),
            self.tracker.charge_as_ref(),
        ));
        let (potential, residual) = self
            .update_potential(&Potential::from_vector(previous_potential.clone()))
            .expect("Potential update failed");
        self.tracker.current_residual = residual;

        Ok(potential.as_ref().clone())
    }

    fn run_loop(&mut self, mut potential: Potential<f64>) -> Result<(), OuterLoopError<f64>> {
        let mixer = Type1AndersonMixer::new(
            potential.as_ref().len(),
            self.convergence_settings.outer_tolerance(),
            self.convergence_settings.maximum_outer_iterations() as u64,
        );
        // let vec_para = potential.as_ref().iter().copied().collect::<Vec<_>>();
        // let initial_parameter = ndarray::Array1::from(vec_para);
        let initial_parameter = potential.as_ref().clone();
        let mut solver = FixedPointSolver::new(mixer, initial_parameter);

        // We print 1 line further down in an incoherent loop
        match self.tracker.calculation {
            Calculation::Coherent { voltage_target: _ } => {
                self.term.move_cursor_to(0, 2)?;
                self.term.clear_to_end_of_screen()?;
            }
            Calculation::Incoherent { voltage_target: _ } => {
                self.term.move_cursor_to(0, 3)?;
                self.term.clear_to_end_of_screen()?;
            }
        }
        tracing::info!("Beginning outer self-consistent loop with Anderson Mixing");
        let solution = solver.run(self)?;

        let solution = Array1::from(solution.get_param().iter().copied().collect::<Vec<_>>());
        potential = Potential::from_vector(solution);
        self.tracker.update_potential(potential);
        // potential = Potential::from_vector(solution.get_param());
        if self.tracker.scattering_scaling > 0.95_f64 {
            self.tracker.write_to_file("incoherent")?;
        }
        //// A single iteration before the loop to avoid updating the potential with an empty charge vector
        Ok(())
    }

    fn potential_owned(&self) -> Potential<f64> {
        self.tracker.potential.clone()
    }
}

impl<GeometryDim, Conn, BandDim, SpectralSpace>
    OuterLoop<'_, f64, GeometryDim, Conn, BandDim, SpectralSpace>
where
    Conn: Connectivity<f64, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<Array1<f64>, BandDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>,
{
    #[tracing::instrument("Potential update", skip_all)]
    fn update_potential(
        &self,
        previous_potential: &Potential<f64>,
    ) -> Result<(Potential<f64>, f64), OuterLoopError<f64>> {
        let cost = super::poisson::PoissonProblemBuilder::default()
            .with_charge(self.tracker.charge_as_ref())
            .with_info_desk(self.info_desk)
            .with_mesh(self.mesh)
            .with_initial_potential(previous_potential)
            .build()?;

        // Define initial parameter vector
        let init_param: Array1<f64> = Array1::from_vec(vec![0_f64; self.mesh.vertices().len()]);

        let residual = cost
            .apply(&init_param)?
            .l2_dist(&Array1::from(vec![0_f64; init_param.len()]))
            .unwrap()
            / previous_potential.as_ref().len() as f64;
        let target = self.convergence_settings.outer_tolerance()
            * self.info_desk.donor_densities[0]
            * crate::constants::ELECTRON_CHARGE;

        self.term.move_cursor_to(0, 5)?;
        self.term.clear_to_end_of_screen()?;
        tracing::info!("Solving Poisson Equation",);

        if residual < target {
            return Ok((previous_potential.clone(), residual));
        }

        let linesearch =
            argmin::solver::linesearch::MoreThuenteLineSearch::new().alpha(0_f64, 0.1_f64)?;
        // Set up solver
        let solver = argmin::solver::gaussnewton::GaussNewtonLS::new(linesearch);

        // Run solver
        let res = Executor::new(cost, solver, init_param)
            .max_iters(25)
            // .configure(|state| state.param(init_param).max_iters(25))
            //.add_observer(SlogLogger::term(), ObserverMode::Never)
            .run()?;

        self.term.move_cursor_to(0, 5)?;
        self.term.clear_to_end_of_screen()?;
        tracing::info!(
            "Poisson calculation converged in {} iterations",
            res.state.iter
        );

        let output = res.state.best_param;
        // We found the change in potential, so add the full solution back on to find the net result...
        let output =
            previous_potential.as_ref() + &output - Array1::from(vec![output[0]; output.len()]);

        // Writing to file
        // let system_time = std::time::SystemTime::now();
        // let datetime: chrono::DateTime<chrono::Utc> = system_time.into();
        // let mut file = std::fs::File::create(format!("../results/potential_{}.txt", datetime))?;
        // for value in previous_potential.as_ref().iter() {
        //     let value = value.to_string();
        //     writeln!(file, "{}", value)?;
        // }
        // let mut file = std::fs::File::create(format!("../results/charge_{}.txt", datetime))?;
        // for value in self.tracker.charge_as_ref().net_charge().iter() {
        //     let value = value.to_string();
        //     writeln!(file, "{}", value)?;
        // }
        // panic!();

        Ok((Potential::from_vector(output), residual))
    }

    pub(crate) fn scattering_scaling(&self) -> f64 {
        self.tracker.scattering_scaling
    }

    pub(crate) fn increment_scattering_scaling(&mut self) {
        self.tracker.scattering_scaling += 0.3_f64;
    }
}

trait PoissonMethods<T: Copy + RealField, GeometryDim: SmallDim, Conn: Connectivity<T, GeometryDim>>
where
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn update_jacobian_diagonal(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        fermi_level: &Array1<T>,
        solution: &Array1<T>,
        output: &mut Array1<T>,
    ) -> color_eyre::Result<()>;

    // Find the updated charge density estimated on switching to a new potential
    // 'q * (N_C Fermi_{0.5} ((E_F - E_C + q \phi) / K T) + N_A - N_D)`
    fn update_charge_density(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        fermi_level: &Array1<T>,
        solution: &Array1<T>,
        output: &mut Array1<T>,
    ) -> color_eyre::Result<()>;
}

impl<T: Copy + RealField, GeometryDim: SmallDim, Conn, BandDim: SmallDim>
    PoissonMethods<T, GeometryDim, Conn> for DeviceInfoDesk<T, GeometryDim, BandDim>
where
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<T, GeometryDim>
        + Allocator<Array1<T>, BandDim>,
{
    // Solve for the diagonal of the Jacobian, given in this approximation by
    // 'q / K T N_C Fermi_{-0.5} ((E_F - E_C + q \phi) / K T)
    fn update_jacobian_diagonal(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        fermi_level: &Array1<T>,
        solution: &Array1<T>,
        output: &mut Array1<T>,
    ) -> color_eyre::Result<()> {
        // TODO actually swap in place, rather than allocating then swapping
        let updated = self.compute_jacobian_diagonal(fermi_level, solution, mesh);
        // Neumann
        let _ = std::mem::replace(output, updated);
        Ok(())
    }

    // Find the updated charge density estimated on switching to a new potential
    // 'q * (N_C Fermi_{0.5} ((E_F - E_C + q \phi) / K T) + N_A - N_D)`
    fn update_charge_density(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        fermi_level: &Array1<T>,
        solution: &Array1<T>,
        output: &mut Array1<T>,
    ) -> color_eyre::Result<()> {
        let updated = self.update_source_vector(mesh, fermi_level, solution);
        let _ = std::mem::replace(output, updated);
        Ok(())
    }
}

use sprs::CsMat;

impl<GeometryDim, Conn, BandDim>
    OuterLoop<'_, f64, GeometryDim, Conn, BandDim, SpectralSpace<f64, ()>>
where
    Conn: Connectivity<f64, GeometryDim>,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<Array1<f64>, BandDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>,
{
    fn build_coherent_inner_loop<'a>(
        &'a self,
        greens_functions: &'a mut AggregateGreensFunctions<
            'a,
            f64,
            CsMat<Complex<f64>>,
            GeometryDim,
            BandDim,
        >,
        self_energies: &'a mut SelfEnergy<f64, GeometryDim, Conn>,
    ) -> InnerLoop<'a, f64, GeometryDim, Conn, CsMat<Complex<f64>>, SpectralSpace<f64, ()>, BandDim>
    {
        InnerLoopBuilder::new()
            .with_convergence_settings(self.convergence_settings)
            .with_mesh(self.mesh)
            .with_spectral_discretisation(self.spectral)
            .with_hamiltonian(self.hamiltonian)
            .with_greens_functions(greens_functions)
            .with_self_energies(self_energies)
            .build(self.tracker.voltage)
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
    <Conn as Connectivity<T, GeometryDim>>::Element: Send + Sync,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<Array1<T>, BandDim>
        + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>,
{
    fn build_coherent_inner_loop<'a>(
        &'a self,
        greens_functions: &'a mut AggregateGreensFunctions<
            'a,
            T,
            CsMat<Complex<T>>,
            GeometryDim,
            BandDim,
        >,
        self_energies: &'a mut SelfEnergy<T, GeometryDim, Conn>,
    ) -> InnerLoop<
        'a,
        T,
        GeometryDim,
        Conn,
        CsMat<Complex<T>>,
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
            .build(self.tracker.voltage)
    }
}

impl<GeometryDim, Conn, BandDim>
    OuterLoop<
        '_,
        f64,
        GeometryDim,
        Conn,
        BandDim,
        SpectralSpace<f64, WavevectorSpace<f64, GeometryDim, Conn>>,
    >
where
    Conn: Connectivity<f64, GeometryDim>,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<Array1<f64>, BandDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>,
{
    fn build_incoherent_inner_loop<'a>(
        &'a self,
        greens_functions: &'a mut AggregateGreensFunctions<
            'a,
            f64,
            ndarray::Array2<Complex<f64>>,
            GeometryDim,
            BandDim,
        >,
        self_energies: &'a mut SelfEnergy<f64, GeometryDim, Conn>,
    ) -> InnerLoop<
        'a,
        f64,
        GeometryDim,
        Conn,
        ndarray::Array2<Complex<f64>>,
        SpectralSpace<f64, WavevectorSpace<f64, GeometryDim, Conn>>,
        BandDim,
    > {
        InnerLoopBuilder::new()
            .with_convergence_settings(self.convergence_settings)
            .with_mesh(self.mesh)
            .with_spectral_discretisation(self.spectral)
            .with_hamiltonian(self.hamiltonian)
            .with_greens_functions(greens_functions)
            .with_self_energies(self_energies)
            .with_scattering_scaling(self.tracker.scattering_scaling)
            .build(self.tracker.voltage)
    }

    fn build_incoherent_inner_loop_mixed<'a>(
        &'a self,
        greens_functions: &'a mut AggregateGreensFunctions<
            'a,
            f64,
            MMatrix<Complex<f64>>,
            GeometryDim,
            BandDim,
        >,
        self_energies: &'a mut SelfEnergy<f64, GeometryDim, Conn>,
    ) -> InnerLoop<
        'a,
        f64,
        GeometryDim,
        Conn,
        MMatrix<Complex<f64>>,
        SpectralSpace<f64, WavevectorSpace<f64, GeometryDim, Conn>>,
        BandDim,
    > {
        InnerLoopBuilder::new()
            .with_convergence_settings(self.convergence_settings)
            .with_mesh(self.mesh)
            .with_spectral_discretisation(self.spectral)
            .with_hamiltonian(self.hamiltonian)
            .with_greens_functions(greens_functions)
            .with_self_energies(self_energies)
            .with_scattering_scaling(self.tracker.scattering_scaling)
            .build(self.tracker.voltage)
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
    DefaultAllocator: Allocator<Array1<T>, BandDim> + Allocator<T, GeometryDim>,
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
    ) -> Array1<T>;

    fn update_source_vector(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        fermi_level: &Array1<T>,
        potential: &Array1<T>,
    ) -> Array1<T>;

    fn compute_jacobian_diagonal(
        &self,
        fermi_level: &Array1<T>,
        potential: &Array1<T>,
        mesh: &Mesh<T, GeometryDim, Conn>,
    ) -> Array1<T>;
}

impl<T: Copy + RealField, GeometryDim: SmallDim, Conn, BandDim: SmallDim>
    OuterLoopInfoDesk<T, GeometryDim, Conn, BandDim> for DeviceInfoDesk<T, GeometryDim, BandDim>
where
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<Array1<T>, BandDim>
        + Allocator<T, BandDim>
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
        let result = charge
            .net_charge()
            .into_iter()
            .zip(mesh.vertices().iter())
            .zip(potential.as_ref().iter())
            .map(|((n, vertex), &phi)| {
                let assignment = &vertex.1;
                // Calculate the density of states in the conduction band
                let n3d = (T::one() + T::one()) // Currently always getting the x-component, is this dumb?
                    * (self.effective_mass_from_assignment(assignment, 0, 0) // The conduction band is always supposed to be in position 0
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

                let band_offset = self.band_offset_from_assignment(assignment, 0); // Again we assume the band offset for the c-band is in position 0
                                                                                   // Get eta_f_plus_phi
                eta_f_minus_ec_plus_phi + band_offset - phi // TODO should this be a plus phi or a minus phi??
            })
            .collect::<Vec<_>>();

        result
    }

    fn calculate_source_vector(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        charge: &Charge<T, BandDim>,
    ) -> Array1<T> {
        let net_charge = charge.net_charge();

        let result = net_charge
            .iter()
            .zip(mesh.vertices())
            .map(|(&n, vertex)| {
                let assignment = &vertex.1;
                let acceptor_density = self.acceptor_density_from_assignment(assignment);
                let donor_density = self.donor_density_from_assignment(assignment);
                (n + acceptor_density - donor_density)
                    * T::from_f64(crate::constants::ELECTRON_CHARGE).unwrap()
            })
            .collect::<Vec<_>>();

        Array1::from(result)
    }

    fn update_source_vector(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        fermi_level: &Array1<T>, // The fermi level defined on the mesh vertices
        potential: &Array1<T>,   // The potential defined on the mesh vertices
    ) -> Array1<T> {
        let gamma = T::from_f64(std::f64::consts::PI.sqrt() / 2.).unwrap();
        assert_eq!(
            potential.len(),
            mesh.vertices().len(),
            "potential must be evaluated on vertices"
        );
        assert_eq!(
            fermi_level.len(),
            mesh.vertices().len(),
            "Fermi level must be evaluated on vertices"
        );
        Array1::from(
            mesh.vertices()
                .iter()
                .zip(fermi_level.iter())
                .zip(potential.iter())
                .map(|((vertex, &fermi_level), &potential)| {
                    let assignment = &vertex.1;
                    let band_offset = self.band_offset_from_assignment(assignment, 0);
                    let effective_mass = self.effective_mass_from_assignment(assignment, 0, 0);

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
        fermi_level: &Array1<T>,
        potential: &Array1<T>,
        mesh: &Mesh<T, GeometryDim, Conn>,
    ) -> Array1<T>
    where
        DefaultAllocator: Allocator<[T; 3], BandDim>,
    {
        let gamma = T::from_f64(std::f64::consts::PI.sqrt()).unwrap();
        Array1::from(
            mesh.vertices()
                .iter()
                .zip(fermi_level.iter())
                .zip(potential.iter())
                .map(|((vertex, &fermi_level), &potential)| {
                    let assignment = &vertex.1;
                    let band_offset = self.band_offset_from_assignment(assignment, 0);
                    let effective_mass = self.effective_mass_from_assignment(assignment, 0, 0);

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

impl<GeometryDim, Conn, BandDim> conflux::core::FixedPointProblem
    for OuterLoop<'_, f64, GeometryDim, Conn, BandDim, SpectralSpace<f64, ()>>
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>
        + Allocator<Array1<f64>, BandDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    type Output = Array1<f64>;
    type Param = Array1<f64>;
    type Float = f64;
    type Square = Array2<f64>;

    #[tracing::instrument(name = "Single iteration", fields(iteration = self.tracker.iteration + 1), skip_all)]
    fn update(
        &mut self,
        potential: &Self::Param,
    ) -> Result<Self::Param, conflux::core::FixedPointError<f64>> {
        let target = self.convergence_settings.outer_tolerance()
            * self.info_desk.donor_densities[0]
            * crate::constants::ELECTRON_CHARGE;

        self.term.move_cursor_to(0, 3).unwrap();
        self.term.clear_to_end_of_screen().unwrap();
        if self.tracker.iteration > 0 {
            tracing::info!(
                "Current residual: {}, target residual: {}",
                self.tracker.current_residual,
                target
            );
        } else {
            tracing::info!("First iteration with target residual: {}", target);
        }

        let potential = Array1::from(potential.into_iter().copied().collect::<Vec<_>>());
        let new_potential = self.single_iteration(&potential).expect("It should work");
        let change = new_potential.l2_dist(&potential).unwrap() / potential.len() as f64;

        self.term.move_cursor_to(0, 5).unwrap();
        self.term.clear_to_end_of_screen().unwrap();
        tracing::info!("Change in potential per element: {change}");
        self.tracker.iteration += 1;

        // let vec_para = new_potential.iter().copied().collect::<Vec<_>>();
        // let new_potential = ndarray::Array1::from(vec_para);
        Ok(new_potential)
    }
}

impl<GeometryDim, Conn, BandDim> conflux::core::FixedPointProblem
    for OuterLoop<
        '_,
        f64,
        GeometryDim,
        Conn,
        BandDim,
        SpectralSpace<f64, WavevectorSpace<f64, GeometryDim, Conn>>,
    >
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<f64, GeometryDim>
        + Allocator<f64, BandDim>
        + Allocator<[f64; 3], BandDim>
        + Allocator<Array1<f64>, BandDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<f64, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[f64; 3], BandDim>>::Buffer: Send + Sync,
{
    type Output = Array1<f64>;
    type Param = Array1<f64>;
    type Float = f64;
    type Square = Array2<f64>;

    #[tracing::instrument(name = "Single iteration", fields(iteration = self.tracker.iteration + 1), skip_all)]
    fn update(
        &mut self,
        potential: &Self::Param,
    ) -> Result<Self::Param, conflux::core::FixedPointError<f64>> {
        let target = self.convergence_settings.outer_tolerance()
            * self.info_desk.donor_densities[0]
            * crate::constants::ELECTRON_CHARGE;

        match self.tracker.calculation {
            Calculation::Coherent { voltage_target: _ } => self.term.move_cursor_to(0, 3).unwrap(),
            Calculation::Incoherent { voltage_target: _ } => {
                self.term.move_cursor_to(0, 4).unwrap()
            }
        };
        self.term.clear_to_end_of_screen().unwrap();
        if self.tracker.iteration > 0 {
            tracing::info!(
                "Current residual: {}, target residual: {}",
                self.tracker.current_residual,
                target
            );
        } else {
            tracing::info!("First iteration with target residual: {}", target);
        }
        let potential = Array1::from(potential.into_iter().copied().collect::<Vec<_>>());
        let new_potential = self.single_iteration(&potential).expect("It should work");
        let change = new_potential.l2_dist(&potential).unwrap() / potential.len() as f64;

        match self.tracker.calculation {
            Calculation::Coherent { voltage_target: _ } => self.term.move_cursor_to(0, 5).unwrap(),
            Calculation::Incoherent { voltage_target: _ } => {
                self.term.move_cursor_to(0, 6).unwrap()
            }
        };
        self.term.clear_to_end_of_screen().unwrap();
        tracing::info!("Change in potential per element: {change}");
        self.tracker.iteration += 1;
        // let vec_para = new_potential.iter().copied().collect::<Vec<_>>();
        // let new_potential = ndarray::Array1::from(vec_para);
        Ok(new_potential)
    }
}

#[cfg(test)]
mod test {
    use super::{OuterLoopInfoDesk, Potential};
    use crate::{
        app::{tracker::TrackerBuilder, Calculation, Configuration},
        device::{info_desk::BuildInfoDesk, Device},
    };
    use nalgebra::U1;
    use ndarray::Array1;
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
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let mut rng = rand::thread_rng();

        let potential = Potential::from_vector(Array1::from(
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
        let source_vector =
            info_desk.update_source_vector(&mesh, &Array1::from(fermi), potential.as_ref());
        // Assert each element in the source vector is near to zero
        for element in source_vector.into_iter() {
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
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let mut rng = rand::thread_rng();

        let potential = Potential::from_vector(Array1::from(
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
        let source_vector =
            info_desk.update_source_vector(&mesh, &Array1::from(fermi), potential.as_ref());
        // Assert each element in the source vector is near to zero
        for (element, initial) in source_vector.into_iter().zip(initial_source.into_iter()) {
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
        let tracker = TrackerBuilder::new(Calculation::Coherent {
            voltage_target: 0_f64,
        })
        .with_mesh(&mesh)
        .with_info_desk(&info_desk)
        .build()
        .unwrap();

        let mut rng = rand::thread_rng();

        let potential = Potential::from_vector(Array1::from(
            (0..mesh.vertices().len())
                .map(|_| rng.gen::<f64>())
                .collect::<Vec<_>>(),
        ));

        let dphi = 1e-8;
        let potential_plus_dphi = Potential::from_vector(Array1::from(
            potential
                .as_ref()
                .iter()
                .map(|phi| phi + dphi)
                .collect::<Vec<_>>(),
        ));
        let potential_minus_dphi = Potential::from_vector(Array1::from(
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
        let fermi_level = Array1::from(info_desk.determine_fermi_level(&mesh, &potential, &charge));
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
