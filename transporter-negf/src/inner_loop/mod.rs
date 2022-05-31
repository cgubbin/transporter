mod methods;

use crate::{
    app::tui::Progress,
    greens_functions::{AggregateGreensFunctions, GreensFunctionMethods},
    hamiltonian::Hamiltonian,
    outer_loop::Convergence,
    self_energy::SelfEnergy,
};
pub(crate) use methods::Inner;
use miette::Diagnostic;
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use std::marker::PhantomData;
use tokio::sync::mpsc::Sender;
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
    RefProgress,
    RefProgressSender,
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
    progress: RefProgress,
    mpsc_sender: RefProgressSender,
    marker: PhantomData<T>,
}

impl<T> InnerLoopBuilder<T, (), (), (), (), (), (), (), ()>
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
            progress: (),
            mpsc_sender: (),
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
        RefProgress,
        RefProgressSender,
    >
    InnerLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefGreensFunctions,
        RefSelfEnergies,
        RefProgress,
        RefProgressSender,
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
        RefProgress,
        RefProgressSender,
    > {
        InnerLoopBuilder {
            convergence_settings,
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            scattering_scaling: self.scattering_scaling,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
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
        RefProgress,
        RefProgressSender,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            scattering_scaling: self.scattering_scaling,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
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
        RefProgress,
        RefProgressSender,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            scattering_scaling: self.scattering_scaling,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
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
        RefProgress,
        RefProgressSender,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            scattering_scaling: self.scattering_scaling,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
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
        RefProgress,
        RefProgressSender,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions,
            self_energies: self.self_energies,
            scattering_scaling: self.scattering_scaling,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
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
        RefProgress,
        RefProgressSender,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies,
            scattering_scaling: self.scattering_scaling,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
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
        RefProgress,
        RefProgressSender,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            scattering_scaling,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_progress<Progress>(
        self,
        progress: &Progress,
    ) -> InnerLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefGreensFunctions,
        RefSelfEnergies,
        &Progress,
        RefProgressSender,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            scattering_scaling: self.scattering_scaling,
            progress,
            mpsc_sender: self.mpsc_sender,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_sender<ProgressSender>(
        self,
        mpsc_sender: &ProgressSender,
    ) -> InnerLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefGreensFunctions,
        RefSelfEnergies,
        RefProgress,
        &ProgressSender,
    > {
        InnerLoopBuilder {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            scattering_scaling: self.scattering_scaling,
            progress: self.progress,
            mpsc_sender,
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
    progress: Progress<T>,
    mpsc_sender: Sender<Progress<T>>,
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
        &'a Progress<T>,
        &'a Sender<Progress<T>>,
    >
where
    T: RealField + Copy,
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
            progress: self.progress.clone(),
            mpsc_sender: self.mpsc_sender.clone(),
        }
    }
}
