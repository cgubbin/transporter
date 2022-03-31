mod methods;

use crate::{
    greens_functions::{AggregateGreensFunctions, GreensFunctionMethods},
    hamiltonian::Hamiltonian,
    outer_loop::Convergence,
    self_energy::SelfEnergy,
};
pub(crate) use methods::Inner;
use nalgebra::ComplexField;
use nalgebra::{allocator::Allocator, DefaultAllocator};
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

pub(crate) struct InnerLoopBuilder<
    T,
    RefConvergenceSettings,
    RefMesh,
    RefSpectral,
    RefHamiltonian,
    RefGreensFunctions,
    RefSelfEnergies,
> where
    T: ComplexField,
{
    convergence_settings: RefConvergenceSettings,
    mesh: RefMesh,
    spectral: RefSpectral,
    hamiltonian: RefHamiltonian,
    greens_functions: RefGreensFunctions,
    self_energies: RefSelfEnergies,
    marker: PhantomData<T>,
}

impl<T> InnerLoopBuilder<T, (), (), (), (), (), ()>
where
    T: ComplexField,
{
    pub(crate) fn new() -> Self {
        Self {
            convergence_settings: (),
            mesh: (),
            spectral: (),
            hamiltonian: (),
            greens_functions: (),
            self_energies: (),
            marker: PhantomData,
        }
    }
}

impl<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefGreensFunctions,
        RefSelfEnergies,
    >
    InnerLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefGreensFunctions,
        RefSelfEnergies,
    >
where
    T: ComplexField,
{
    pub(crate) fn with_convergence_settings<ConvergenceSettings>(
        self,
        convergence_settings: &ConvergenceSettings,
    ) -> InnerLoopBuilder<
        T,
        &ConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefGreensFunctions,
        RefSelfEnergies,
    > {
        InnerLoopBuilder {
            convergence_settings,
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> InnerLoopBuilder<
        T,
        RefConvergenceSettings,
        &Mesh,
        RefSpectral,
        RefHamiltonian,
        RefGreensFunctions,
        RefSelfEnergies,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_spectral_discretisation<Spectral>(
        self,
        spectral: &Spectral,
    ) -> InnerLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        &Spectral,
        RefHamiltonian,
        RefGreensFunctions,
        RefSelfEnergies,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_hamiltonian<Hamiltonian>(
        self,
        hamiltonian: &Hamiltonian,
    ) -> InnerLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        &Hamiltonian,
        RefGreensFunctions,
        RefSelfEnergies,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_greens_functions<GreensFunctions>(
        self,
        greens_functions: &mut GreensFunctions,
    ) -> InnerLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        &mut GreensFunctions,
        RefSelfEnergies,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions,
            self_energies: self.self_energies,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_self_energies<SelfEnergies>(
        self,
        self_energies: &mut SelfEnergies,
    ) -> InnerLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefGreensFunctions,
        &mut SelfEnergies,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies,
            marker: PhantomData,
        }
    }
}

pub(crate) struct InnerLoop<'a, T, GeometryDim, Conn, Matrix, SpectralSpace>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    Matrix: GreensFunctionMethods<T>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    mesh: &'a Mesh<T::RealField, GeometryDim, Conn>,
    spectral: &'a SpectralSpace,
    hamiltonian: &'a Hamiltonian<T::RealField>,
    greens_functions: &'a AggregateGreensFunctions<T, Matrix>,
    self_energies: &'a SelfEnergy<T, GeometryDim, Conn, Matrix>,
    convergence_settings: &'a Convergence<T::RealField>,
}

impl<'a, T, GeometryDim, Conn, Matrix, SpectralSpace>
    InnerLoopBuilder<
        T,
        &'a Convergence<T::RealField>,
        &'a Mesh<T::RealField, GeometryDim, Conn>,
        &'a SpectralSpace,
        &'a Hamiltonian<T::RealField>,
        &'a mut AggregateGreensFunctions<T, Matrix>,
        &'a mut SelfEnergy<T, GeometryDim, Conn, Matrix>,
    >
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    Matrix: GreensFunctionMethods<T>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    pub(crate) fn build(self) -> InnerLoop<'a, T, GeometryDim, Conn, Matrix, SpectralSpace> {
        InnerLoop {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            convergence_settings: self.convergence_settings,
        }
    }
}
