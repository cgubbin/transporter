mod convergence;
mod methods;
mod poisson;

pub(crate) use convergence::Convergence;
pub(crate) use methods::{Outer, Potential};

use crate::{
    app::Tracker,
    device::info_desk::DeviceInfoDesk,
    hamiltonian::Hamiltonian,
    postprocessor::{Charge, ChargeAndCurrent},
};
use nalgebra::{allocator::Allocator, ComplexField, DVector, DefaultAllocator};
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

/// Builder struct for the outer loop allows for polymorphism over the `SpectralSpace`
pub(crate) struct OuterLoopBuilder<
    T,
    RefConvergenceSettings,
    RefMesh,
    RefSpectral,
    RefHamiltonian,
    RefTracker,
    RefInfoDesk,
> {
    mesh: RefMesh,
    spectral: RefSpectral,
    hamiltonian: RefHamiltonian,
    convergence_settings: RefConvergenceSettings,
    tracker: RefTracker,
    info_desk: RefInfoDesk,
    marker: PhantomData<T>,
}

impl<T> OuterLoopBuilder<T, (), (), (), (), (), ()> {
    /// Initialise an empty OuterLoopBuilder
    pub(crate) fn new() -> Self {
        Self {
            mesh: (),
            spectral: (),
            hamiltonian: (),
            convergence_settings: (),
            tracker: (),
            info_desk: (),
            marker: PhantomData,
        }
    }
}

impl<
        T: ComplexField,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
    >
    OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
    >
{
    /// Attach the problem's `Mesh`
    pub(crate) fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        &Mesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
    > {
        OuterLoopBuilder {
            mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            marker: PhantomData,
        }
    }

    /// Attach the `SpectralSpace` associated with the problem
    pub(crate) fn with_spectral_space<Spectral>(
        self,
        spectral: &Spectral,
    ) -> OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        &Spectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            marker: PhantomData,
        }
    }

    /// Attach the constructed `Hamiltonian` associated with the problem
    pub(crate) fn with_hamiltonian<Hamiltonian>(
        self,
        hamiltonian: &Hamiltonian,
    ) -> OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        &Hamiltonian,
        RefTracker,
        RefInfoDesk,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            marker: PhantomData,
        }
    }

    /// Attach convergence information for the inner and outer loop
    pub(crate) fn with_convergence_settings<ConvergenceSettings>(
        self,
        convergence_settings: &ConvergenceSettings,
    ) -> OuterLoopBuilder<
        T,
        &ConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            marker: PhantomData,
        }
    }

    /// Attach the global tracker
    pub(crate) fn with_tracker<Tracker>(
        self,
        tracker: &Tracker,
    ) -> OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        &Tracker,
        RefInfoDesk,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker,
            info_desk: self.info_desk,
            marker: PhantomData,
        }
    }

    /// Attach the info desk
    pub(crate) fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        &InfoDesk,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk,
            marker: PhantomData,
        }
    }
}

/// A structure holding the information to carry out the outer iteration
pub(crate) struct OuterLoop<'a, T, GeometryDim, Conn, BandDim, SpectralSpace>
where
    T: ComplexField,
    <T as ComplexField>::RealField: Copy,
    BandDim: SmallDim,
    GeometryDim: SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>
        + Allocator<
            Matrix<
                T::RealField,
                Dynamic,
                Const<1_usize>,
                VecStorage<T::RealField, Dynamic, Const<1_usize>>,
            >,
            BandDim,
        > + Allocator<T::RealField, BandDim>
        + Allocator<[T::RealField; 3], BandDim>,
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
    tracker: LoopTracker<T::RealField, BandDim>,
    info_desk: &'a DeviceInfoDesk<T::RealField, GeometryDim, BandDim>,
}

impl<'a, T, GeometryDim, Conn, BandDim, SpectralSpace>
    OuterLoopBuilder<
        T,
        &'a Convergence<T::RealField>,
        &'a Mesh<T::RealField, GeometryDim, Conn>,
        &'a SpectralSpace,
        &'a Hamiltonian<T::RealField>,
        &'a Tracker<'a, T::RealField, GeometryDim, BandDim, Conn>,
        &'a DeviceInfoDesk<T::RealField, GeometryDim, BandDim>,
    >
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>
        + Allocator<T::RealField, BandDim>
        + Allocator<[T::RealField; 3], BandDim>
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
    /// Build out the OuterLoop -> Generic over the SpectralSpace so the OuterLoop can do both coherent and incoherent transport
    pub(crate) fn build(
        self,
    ) -> color_eyre::Result<OuterLoop<'a, T, GeometryDim, Conn, BandDim, SpectralSpace>> {
        let tracker = LoopTracker::from_global_tracker(self.tracker);
        Ok(OuterLoop {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            hamiltonian: self.hamiltonian,
            spectral: self.spectral,
            tracker,
            info_desk: self.info_desk,
        })
    }
}

use nalgebra::RealField;
use nalgebra::{Const, Dynamic, Matrix, VecStorage};
pub(crate) struct LoopTracker<T: nalgebra::RealField, BandDim: SmallDim>
where
    DefaultAllocator: Allocator<
        Matrix<
            T::RealField,
            Dynamic,
            Const<1_usize>,
            VecStorage<T::RealField, Dynamic, Const<1_usize>>,
        >,
        BandDim,
    >,
{
    charge_and_currents: ChargeAndCurrent<T, BandDim>,
    potential: Potential<T>,
    fermi_level: DVector<T>,
}

impl<T: Copy + RealField, BandDim: SmallDim> LoopTracker<T, BandDim>
where
    DefaultAllocator: Allocator<
        Matrix<
            T::RealField,
            Dynamic,
            Const<1_usize>,
            VecStorage<T::RealField, Dynamic, Const<1_usize>>,
        >,
        BandDim,
    >,
{
    pub(crate) fn from_global_tracker<GeometryDim: SmallDim, Conn: Connectivity<T, GeometryDim>>(
        global_tracker: &Tracker<'_, T, GeometryDim, BandDim, Conn>,
    ) -> Self
    where
        DefaultAllocator: Allocator<T::RealField, GeometryDim>
            + Allocator<T::RealField, BandDim>
            + Allocator<[T::RealField; 3], BandDim>,
    {
        // This is a dirty clone, it might be best to just mutably update the global tracker
        Self {
            potential: global_tracker.potential().clone(),
            charge_and_currents: ChargeAndCurrent::from_charge_and_current(
                global_tracker.charge().clone(),
                global_tracker.current().clone(),
            ),
            fermi_level: DVector::from(
                (0..global_tracker.num_vertices())
                    .map(|_| T::zero())
                    .collect::<Vec<_>>(),
            ),
        }
    }

    pub(crate) fn charge_and_currents_mut(&mut self) -> &mut ChargeAndCurrent<T, BandDim> {
        &mut self.charge_and_currents
    }

    pub(crate) fn charge_as_ref(&self) -> &Charge<T, BandDim> {
        self.charge_and_currents.charge_as_ref()
    }

    pub(crate) fn potential_mut(&mut self) -> &mut Potential<T> {
        &mut self.potential
    }

    pub(crate) fn fermi_level(&self) -> &DVector<T> {
        &self.fermi_level
    }

    pub(crate) fn fermi_level_mut(&mut self) -> &mut DVector<T> {
        &mut self.fermi_level
    }
}
