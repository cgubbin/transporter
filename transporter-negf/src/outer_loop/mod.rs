mod convergence;
mod methods;

pub(crate) use convergence::Convergence;
pub(crate) use methods::{Outer, Potential};

use crate::{hamiltonian::Hamiltonian, postprocessor::ChargeAndCurrent};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator};
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

/// Builder struct for the outer loop allows for polymorphism over the `SpectralSpace`
pub(crate) struct OuterLoopBuilder<T, RefConvergenceSettings, RefMesh, RefSpectral, RefHamiltonian>
{
    mesh: RefMesh,
    spectral: RefSpectral,
    hamiltonian: RefHamiltonian,
    convergence_settings: RefConvergenceSettings,
    marker: PhantomData<T>,
}

impl<T> OuterLoopBuilder<T, (), (), (), ()> {
    /// Initialise an empty OuterLoopBuilder
    pub(crate) fn new() -> Self {
        Self {
            mesh: (),
            spectral: (),
            hamiltonian: (),
            convergence_settings: (),
            marker: PhantomData,
        }
    }
}

impl<T: ComplexField, RefConvergenceSettings, RefMesh, RefSpectral, RefHamiltonian>
    OuterLoopBuilder<T, RefConvergenceSettings, RefMesh, RefSpectral, RefHamiltonian>
{
    /// Attach the problem's `Mesh`
    pub(crate) fn with_mesh<Mesh>(
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

    /// Attach the `SpectralSpace` associated with the problem
    pub(crate) fn with_spectral_space<Spectral>(
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

    /// Attach the constructed `Hamiltonian` associated with the problem
    pub(crate) fn with_hamiltonian<Hamiltonian>(
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

    /// Attach convergence information for the inner and outer loop
    pub(crate) fn with_convergence_settings<ConvergenceSettings>(
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

/// A structure holding the information to carry out the outer iteration
pub(crate) struct OuterLoop<'a, T, GeometryDim, Conn, SpectralSpace>
where
    T: ComplexField,
    <T as ComplexField>::RealField: Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    /// The convergence information for the outerloop and the spawned innerloop
    convergence_settings: &'a Convergence<T::RealField>,
    /// The mesh associated with the problem
    mesh: &'a Mesh<T::RealField, GeometryDim, Conn>,
    /// The spectral mesh and integration weights associated with the problem
    spectral: &'a SpectralSpace,
    /// The Hamiltonian associated with the problem
    hamiltonian: &'a Hamiltonian<T::RealField>,
    // TODO A solution tracker, think about this IMPL. We already have a top-level tracker
    tracker: Tracker<T::RealField>,
}

impl<'a, T, GeometryDim, Conn, SpectralSpace>
    OuterLoopBuilder<
        T,
        &'a Convergence<T::RealField>,
        &'a Mesh<T::RealField, GeometryDim, Conn>,
        &'a SpectralSpace,
        &'a Hamiltonian<T::RealField>,
    >
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    GeometryDim: transporter_mesher::SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    /// Build out the OuterLoop -> Generic over the SpectralSpace so the OuterLoop can do both coherent and incoherent transport
    pub(crate) fn build(
        self,
    ) -> color_eyre::Result<OuterLoop<'a, T, GeometryDim, Conn, SpectralSpace>> {
        Ok(OuterLoop {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            hamiltonian: self.hamiltonian,
            spectral: self.spectral,
            tracker: Tracker::new(),
        })
    }
}

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
