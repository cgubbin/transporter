mod methods;

use crate::{
    app::NEGFFloat,
    greens_functions::{AggregateGreensFunctions, GreensFunctionMethods},
    hamiltonian::Hamiltonian,
    outer_loop::Convergence,
    self_energy::SelfEnergy,
};
pub(crate) use methods::Inner;
use miette::Diagnostic;
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

#[derive(thiserror::Error, Debug, Diagnostic)]
pub(crate) enum InnerLoopError {
    #[error(transparent)]
    GreensFunction(#[from] crate::greens_functions::GreensFunctionError),
    #[error(transparent)]
    SelfEnergy(#[from] crate::self_energy::SelfEnergyError),
    #[error(transparent)]
    PostProcessor(#[from] crate::postprocessor::PostProcessorError),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error("Exceeded max inner iterations")]
    OutOfIterations,
}

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
    scattering_scaling: T,
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
            scattering_scaling: T::one(),
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
            scattering_scaling: self.scattering_scaling,
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
            scattering_scaling: self.scattering_scaling,
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
            scattering_scaling: self.scattering_scaling,
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
            scattering_scaling: self.scattering_scaling,
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
            scattering_scaling: self.scattering_scaling,
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
            scattering_scaling: self.scattering_scaling,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_scattering_scaling(
        self,
        scattering_scaling: T,
    ) -> InnerLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefGreensFunctions,
        RefSelfEnergies,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            scattering_scaling,
            marker: PhantomData,
        }
    }
}

pub(crate) struct InnerLoop<'a, T, GeometryDim, Conn, Matrix, SpectralSpace, BandDim>
where
    T: NEGFFloat,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    <Conn as Connectivity<T, GeometryDim>>::Element: Send + Sync,
    Matrix: GreensFunctionMethods<T> + Send + Sync,
    DefaultAllocator:
        Allocator<T, GeometryDim> + Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    mesh: &'a Mesh<T, GeometryDim, Conn>,
    spectral: &'a SpectralSpace,
    hamiltonian: &'a Hamiltonian<T>,
    greens_functions: &'a mut AggregateGreensFunctions<'a, T, Matrix, GeometryDim, BandDim>,
    self_energies: &'a mut SelfEnergy<T, GeometryDim, Conn>,
    convergence_settings: &'a Convergence<T>,
    scattering_scaling: T,
    voltage: T,
    pub(crate) rate: Option<num_complex::Complex<T>>,
    term: console::Term,
}

impl<'a, T, GeometryDim, Conn, Matrix, SpectralSpace, BandDim>
    InnerLoopBuilder<
        T,
        &'a Convergence<T>,
        &'a Mesh<T, GeometryDim, Conn>,
        &'a SpectralSpace,
        &'a Hamiltonian<T>,
        &'a mut AggregateGreensFunctions<'a, T, Matrix, GeometryDim, BandDim>,
        &'a mut SelfEnergy<T, GeometryDim, Conn>,
    >
where
    T: NEGFFloat,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    <Conn as Connectivity<T, GeometryDim>>::Element: Send + Sync,
    Matrix: GreensFunctionMethods<T> + Send + Sync,
    DefaultAllocator:
        Allocator<T, GeometryDim> + Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    pub(crate) fn build(
        self,
        voltage: T,
    ) -> InnerLoop<'a, T, GeometryDim, Conn, Matrix, SpectralSpace, BandDim> {
        InnerLoop {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            convergence_settings: self.convergence_settings,
            scattering_scaling: self.scattering_scaling,
            voltage,
            rate: None,
            term: console::Term::stdout(),
        }
    }
}
