mod methods;
use crate::greens_functions::GreensFunctionMethods;
use crate::spectral::{BallisticSpectral, ScatteringSpectral};
use nalgebra::{ComplexField, RealField};

pub(crate) struct InnerLoopBuilder<
    T,
    RefMesh,
    RefSpectral,
    RefHamiltonian,
    RefGreensFunctions,
    RefSelfEnergies,
> where
    T: ComplexField,
{
    mesh: RefMesh,
    spectral: RefSpectral,
    hamiltonian: RefHamiltonian,
    greens_functions: RefGreensFunctions,
    self_energies: RefSelfEnergies,
    tolerance: T::RealField,
    maximum_iterations: usize,
    marker: std::marker::PhantomData<T>,
}

impl<T> InnerLoopBuilder<T, (), (), (), (), ()>
where
    T: ComplexField,
{
    fn new() -> Self {
        Self {
            mesh: (),
            spectral: (),
            hamiltonian: (),
            greens_functions: (),
            self_energies: (),
            tolerance: T::zero().real(),
            maximum_iterations: 0,
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, RefMesh, RefSpectral, RefHamiltonian, RefGreensFunctions, RefSelfEnergies>
    InnerLoopBuilder<T, RefMesh, RefSpectral, RefHamiltonian, RefGreensFunctions, RefSelfEnergies>
where
    T: ComplexField,
{
    fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> InnerLoopBuilder<T, &Mesh, RefSpectral, RefHamiltonian, RefGreensFunctions, RefSelfEnergies>
    {
        InnerLoopBuilder {
            mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            tolerance: self.tolerance,
            maximum_iterations: self.maximum_iterations,
            marker: std::marker::PhantomData,
        }
    }

    fn with_spectral_discretisation<Spectral>(
        self,
        spectral: &Spectral,
    ) -> InnerLoopBuilder<T, RefMesh, &Spectral, RefHamiltonian, RefGreensFunctions, RefSelfEnergies>
    {
        InnerLoopBuilder {
            mesh: self.mesh,
            spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            tolerance: self.tolerance,
            maximum_iterations: self.maximum_iterations,
            marker: std::marker::PhantomData,
        }
    }

    fn with_hamiltonian<Hamiltonian>(
        self,
        hamiltonian: &Hamiltonian,
    ) -> InnerLoopBuilder<T, RefMesh, RefSpectral, &Hamiltonian, RefGreensFunctions, RefSelfEnergies>
    {
        InnerLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            tolerance: self.tolerance,
            maximum_iterations: self.maximum_iterations,
            marker: std::marker::PhantomData,
        }
    }

    fn with_greens_functions<GreensFunctions>(
        self,
        greens_functions: &mut GreensFunctions,
    ) -> InnerLoopBuilder<
        T,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        &mut GreensFunctions,
        RefSelfEnergies,
    > {
        InnerLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions,
            self_energies: self.self_energies,
            tolerance: self.tolerance,
            maximum_iterations: self.maximum_iterations,
            marker: std::marker::PhantomData,
        }
    }

    fn with_self_energies<SelfEnergies>(
        self,
        self_energies: &mut SelfEnergies,
    ) -> InnerLoopBuilder<
        T,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefGreensFunctions,
        &mut SelfEnergies,
    > {
        InnerLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies,
            tolerance: self.tolerance,
            maximum_iterations: self.maximum_iterations,
            marker: std::marker::PhantomData,
        }
    }

    fn with_tolerance(
        self,
        tolerance: T::RealField,
    ) -> InnerLoopBuilder<
        T,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefGreensFunctions,
        RefSelfEnergies,
    > {
        InnerLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            tolerance,
            maximum_iterations: self.maximum_iterations,
            marker: std::marker::PhantomData,
        }
    }

    fn with_maximum_iterations(
        self,
        maximum_iterations: usize,
    ) -> InnerLoopBuilder<
        T,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefGreensFunctions,
        RefSelfEnergies,
    > {
        InnerLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            tolerance: self.tolerance,
            maximum_iterations,
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a, Mesh, Spectral, Hamiltonian, GreensFunctions, SelfEnergies, T>
    InnerLoopBuilder<
        T,
        &'a Mesh,
        &'a Spectral,
        &'a Hamiltonian,
        &'a mut GreensFunctions,
        &'a mut SelfEnergies,
    >
where
    T: ComplexField,
{
    fn build(self) -> InnerLoop<'a, T, Mesh, Spectral, Hamiltonian, GreensFunctions, SelfEnergies> {
        InnerLoop {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            greens_functions: self.greens_functions,
            self_energies: self.self_energies,
            tolerance: self.tolerance,
            maximum_iterations: self.maximum_iterations,
        }
    }
}
pub(crate) struct InnerLoop<'a, T, Mesh, Spectral, Hamiltonian, GreensFunctions, SelfEnergies>
where
    T: ComplexField,
{
    mesh: &'a Mesh,
    spectral: &'a Spectral,
    hamiltonian: &'a Hamiltonian,
    greens_functions: &'a mut GreensFunctions,
    self_energies: &'a mut SelfEnergies,
    tolerance: T::RealField,
    maximum_iterations: usize,
}
