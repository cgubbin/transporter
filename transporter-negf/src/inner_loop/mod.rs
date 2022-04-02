mod methods;

use crate::{
    greens_functions::{AggregateGreensFunctions, GreensFunctionMethods},
    hamiltonian::Hamiltonian,
    outer_loop::Convergence,
    self_energy::SelfEnergy,
};
pub(crate) use methods::Inner;
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
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
    T: RealField,
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
    T: RealField,
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
    T: RealField,
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

pub(crate) struct InnerLoop<'a, T, GeometryDim, Conn, Matrix, SpectralSpace, BandDim>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    Matrix: GreensFunctionMethods<T>,
    DefaultAllocator:
        Allocator<T, GeometryDim> + Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    mesh: &'a Mesh<T, GeometryDim, Conn>,
    spectral: &'a SpectralSpace,
    hamiltonian: &'a Hamiltonian<T>,
    greens_functions: &'a mut AggregateGreensFunctions<'a, T, Matrix, GeometryDim, BandDim>,
    self_energies: &'a mut SelfEnergy<T, GeometryDim, Conn, Matrix>,
    convergence_settings: &'a Convergence<T>,
}

impl<'a, T, GeometryDim, Conn, Matrix, SpectralSpace, BandDim>
    InnerLoopBuilder<
        T,
        &'a Convergence<T>,
        &'a Mesh<T, GeometryDim, Conn>,
        &'a SpectralSpace,
        &'a Hamiltonian<T>,
        &'a mut AggregateGreensFunctions<'a, T, Matrix, GeometryDim, BandDim>,
        &'a mut SelfEnergy<T, GeometryDim, Conn, Matrix>,
    >
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    Matrix: GreensFunctionMethods<T>,
    DefaultAllocator:
        Allocator<T, GeometryDim> + Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    pub(crate) fn build(
        self,
    ) -> InnerLoop<'a, T, GeometryDim, Conn, Matrix, SpectralSpace, BandDim> {
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
