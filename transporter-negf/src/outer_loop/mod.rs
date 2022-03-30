mod convergence;
mod methods;

pub(crate) use convergence::{CalculationType, Convergence};

use crate::{hamiltonian::Hamiltonian, spectral::ScatteringSpectral};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator};
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

pub(crate) struct OuterLoopBuilder<T, RefConvergenceSettings, RefMesh, RefSpectral, RefHamiltonian>
{
    mesh: RefMesh,
    spectral: RefSpectral,
    hamiltonian: RefHamiltonian,
    convergence_settings: RefConvergenceSettings,
    marker: PhantomData<T>,
}

impl<T> OuterLoopBuilder<T, (), (), (), ()> {
    fn new() -> Self {
        Self {
            mesh: (),
            spectral: (),
            hamiltonian: (),
            convergence_settings: (),
            marker: PhantomData,
        }
    }
}

impl<T, RefConvergenceSettings, RefMesh, RefSpectral, RefHamiltonian>
    OuterLoopBuilder<T, RefConvergenceSettings, RefMesh, RefSpectral, RefHamiltonian>
{
    fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> OuterLoopBuilder<T, RefConvergenceSettings, &Mesh, RefSpectral, RefHamiltonian> {
        OuterLoopBuilder {
            mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            marker: PhantomData,
        }
    }

    fn with_spectral_space<Spectral>(
        self,
        spectral: &Spectral,
    ) -> OuterLoopBuilder<T, RefConvergenceSettings, RefMesh, &Spectral, RefHamiltonian> {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            marker: PhantomData,
        }
    }

    fn with_hamiltonian<Hamiltonian>(
        self,
        hamiltonian: &Hamiltonian,
    ) -> OuterLoopBuilder<T, RefConvergenceSettings, RefMesh, RefSpectral, &Hamiltonian> {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian,
            convergence_settings: self.convergence_settings,
            marker: PhantomData,
        }
    }

    fn with_convergence_settings<ConvergenceSettings>(
        self,
        convergence_settings: &ConvergenceSettings,
    ) -> OuterLoopBuilder<T, &ConvergenceSettings, RefMesh, RefSpectral, RefHamiltonian> {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings,
            marker: PhantomData,
        }
    }
}

struct OuterLoop<'a, T, GeometryDim, Conn>
where
    T: ComplexField,
    <T as ComplexField>::RealField: Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    convergence_settings: &'a Convergence<T::RealField>,
    mesh: &'a Mesh<T::RealField, GeometryDim, Conn>,
    spectral: &'a ScatteringSpectral<T::RealField, GeometryDim, Conn>,
    hamiltonian: &'a Hamiltonian<T::RealField>,
    tracker: Tracker<T::RealField>,
}

impl<'a, T, GeometryDim, Conn>
    OuterLoopBuilder<
        T,
        &'a Convergence<T::RealField>,
        &'a Mesh<T::RealField, GeometryDim, Conn>,
        &'a ScatteringSpectral<T::RealField, GeometryDim, Conn>,
        &'a Hamiltonian<T::RealField>,
    >
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    GeometryDim: transporter_mesher::SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    fn build(self) -> color_eyre::Result<OuterLoop<'a, T, GeometryDim, Conn>> {
        Ok(OuterLoop {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            hamiltonian: self.hamiltonian,
            spectral: self.spectral,
            tracker: Tracker::new(),
        })
    }
}

use crate::postprocessor::ChargeAndCurrent;
use methods::Potential;

pub(crate) struct Tracker<T: nalgebra::RealField> {
    converged_coherent_calculation: bool,
    charge_and_currents: ChargeAndCurrent<T>,
    potential: Potential<T>,
    marker: PhantomData<T>,
}

impl<T: nalgebra::RealField> Tracker<T> {
    pub(crate) fn new() -> Self {
        todo!()
        //Self {
        //    converged_coherent_calculation: false,
        //    marker: PhantomData,
        //}
    }

    pub(crate) fn coherent_is_converged(&self) -> bool {
        self.converged_coherent_calculation
    }

    pub(crate) fn charge_and_currents_mut(&mut self) -> &mut ChargeAndCurrent<T> {
        &mut self.charge_and_currents
    }

    pub(crate) fn potential_mut(&mut self) -> &mut Potential<T> {
        &mut self.potential
    }
}
