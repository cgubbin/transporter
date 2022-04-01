use super::OuterLoop;
use crate::{
    inner_loop::{Inner, InnerLoop},
    self_energy::{SelfEnergy, SelfEnergyBuilder},
};
use nalgebra::{allocator::Allocator, ComplexField, DMatrix, DVector, DefaultAllocator, RealField};
use nalgebra_sparse::CsrMatrix;
use transporter_mesher::{Connectivity, SmallDim};

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

impl<T, GeometryDim, Conn, BandDim> Outer<T::RealField>
    for OuterLoop<'_, T, GeometryDim, Conn, BandDim, SpectralSpace<T::RealField, ()>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    Conn: Connectivity<T::RealField, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>
        + Allocator<
            Matrix<
                T::RealField,
                Dynamic,
                Const<1_usize>,
                VecStorage<T::RealField, Dynamic, Const<1_usize>>,
            >,
            BandDim,
        >,
{
    fn is_loop_converged(
        &self,
        previous_potential: &mut Potential<T::RealField>,
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
        // TODO Building the gfs and SE here is a bad idea, we should do this else where so it is not redone on every iteration
        let mut greens_functions = GreensFunctionBuilder::new()
            .with_spectral_discretisation(self.spectral)
            .build();
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

    fn run_loop(&mut self, mut potential: Potential<T::RealField>) -> color_eyre::Result<()> {
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

impl<T, GeometryDim, Conn, BandDim> Outer<T::RealField>
    for OuterLoop<
        '_,
        T,
        GeometryDim,
        Conn,
        BandDim,
        SpectralSpace<T::RealField, WavevectorSpace<T::RealField, GeometryDim, Conn>>,
    >
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    Conn: Connectivity<T::RealField, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>
        + Allocator<
            Matrix<
                T::RealField,
                Dynamic,
                Const<1_usize>,
                VecStorage<T::RealField, Dynamic, Const<1_usize>>,
            >,
            BandDim,
        >,
{
    fn is_loop_converged(
        &self,
        previous_potential: &mut Potential<T::RealField>,
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

    fn run_loop(&mut self, mut potential: Potential<T::RealField>) -> color_eyre::Result<()> {
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

impl<T, GeometryDim, Conn, BandDim, SpectralSpace>
    OuterLoop<'_, T, GeometryDim, Conn, BandDim, SpectralSpace>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    Conn: Connectivity<T::RealField, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>
        + Allocator<
            Matrix<
                T::RealField,
                Dynamic,
                Const<1_usize>,
                VecStorage<T::RealField, Dynamic, Const<1_usize>>,
            >,
            BandDim,
        >,
{
    fn update_potential(&self) -> color_eyre::Result<Potential<T::RealField>> {
        todo!()
    }
}

impl<T, GeometryDim, Conn, BandDim>
    OuterLoop<'_, T, GeometryDim, Conn, BandDim, SpectralSpace<T::RealField, ()>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    Conn: Connectivity<T::RealField, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>
        + Allocator<
            Matrix<
                T::RealField,
                Dynamic,
                Const<1_usize>,
                VecStorage<T::RealField, Dynamic, Const<1_usize>>,
            >,
            BandDim,
        >,
{
    fn build_coherent_inner_loop<'a>(
        &'a self,
        greens_functions: &'a mut AggregateGreensFunctions<T, CsrMatrix<T>>,
        self_energies: &'a mut SelfEnergy<T, GeometryDim, Conn, CsrMatrix<T>>,
    ) -> InnerLoop<'a, T, GeometryDim, Conn, CsrMatrix<T>, SpectralSpace<T::RealField, ()>> {
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
        SpectralSpace<T::RealField, WavevectorSpace<T::RealField, GeometryDim, Conn>>,
    >
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    Conn: Connectivity<T::RealField, GeometryDim>,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>
        + Allocator<
            Matrix<
                T::RealField,
                Dynamic,
                Const<1_usize>,
                VecStorage<T::RealField, Dynamic, Const<1_usize>>,
            >,
            BandDim,
        >,
{
    fn build_incoherent_inner_loop<'a>(
        &'a self,
        greens_functions: &'a mut AggregateGreensFunctions<T, DMatrix<T>>,
        self_energies: &'a mut SelfEnergy<T, GeometryDim, Conn, DMatrix<T>>,
    ) -> InnerLoop<
        'a,
        T,
        GeometryDim,
        Conn,
        DMatrix<T>,
        SpectralSpace<T::RealField, WavevectorSpace<T::RealField, GeometryDim, Conn>>,
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
