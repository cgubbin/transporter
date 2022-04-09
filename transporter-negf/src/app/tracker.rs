use crate::{
    device::info_desk::DeviceInfoDesk,
    hamiltonian::HamiltonianInfoDesk,
    outer_loop::Potential,
    postprocessor::{Charge, Current},
};
use nalgebra::{
    allocator::Allocator, Const, DVector, DefaultAllocator, Dynamic, Matrix, OPoint, OVector,
    RealField, VecStorage,
};
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

/// A tracker struct, holds the state of the solution. We `impl` all InfoDesk methods on the tracker
pub(crate) struct Tracker<'a, T: RealField, GeometryDim: SmallDim, BandDim: SmallDim>
where
    // C: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        // + Allocator<T, GeometryDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    // mesh: &'a Mesh<T, GeometryDim, C>,
    pub(crate) info_desk: &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
    charge_densities: Charge<T, BandDim>,
    current_densities: Current<T, BandDim>,
    potential: Potential<T>,
    __marker: PhantomData<GeometryDim>,
}

impl<'a, T: Copy + RealField, GeometryDim: SmallDim, BandDim: SmallDim>
    Tracker<'a, T, GeometryDim, BandDim>
where
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<T, GeometryDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    pub(crate) fn potential(&self) -> &Potential<T> {
        &self.potential
    }
    pub(crate) fn charge(&self) -> &Charge<T, BandDim> {
        &self.charge_densities
    }

    pub(crate) fn current(&self) -> &Current<T, BandDim> {
        &self.current_densities
    }

    pub(crate) fn num_vertices(&self) -> usize {
        self.potential.as_ref().len()
    }

    pub(crate) fn update_potential(&mut self, potential: Potential<T>) {
        self.potential = potential;
    }
}

impl<T: Copy + RealField, BandDim: SmallDim, GeometryDim: SmallDim> HamiltonianInfoDesk<T>
    for Tracker<'_, T, GeometryDim, BandDim>
where
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    // type BandDim = BandDim;
    type GeometryDim = GeometryDim;

    fn get_band_levels(&self, region_index: usize) -> &OPoint<T, Self::BandDim> {
        &self.info_desk.band_offsets[region_index]
    }

    fn get_effective_mass(&self, region_index: usize, band_index: usize) -> &[T; 3] {
        &self.info_desk.effective_masses[region_index][(band_index, 0)]
    }
}

impl<T: Copy + RealField, BandDim: SmallDim, GeometryDim: SmallDim>
    crate::hamiltonian::PotentialInfoDesk<T> for Tracker<'_, T, GeometryDim, BandDim>
where
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    type BandDim = BandDim;
    fn potential(&self, vertex_indices: &[usize]) -> T {
        vertex_indices.iter().fold(T::zero(), |acc, &vertex_index| {
            acc + self.potential.get(vertex_index)
        }) / T::from_usize(vertex_indices.len()).unwrap()
    }
}

pub(crate) struct TrackerBuilder<RefInfoDesk, RefMesh> {
    info_desk: RefInfoDesk,
    mesh: RefMesh,
}

impl TrackerBuilder<(), ()> {
    pub(crate) fn new() -> Self {
        Self {
            info_desk: (),
            mesh: (),
        }
    }
}

impl<RefInfoDesk, RefMesh> TrackerBuilder<RefInfoDesk, RefMesh> {
    pub(crate) fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> TrackerBuilder<&InfoDesk, RefMesh> {
        TrackerBuilder {
            info_desk,
            mesh: self.mesh,
        }
    }
    pub(crate) fn with_mesh<Mesh>(self, mesh: &Mesh) -> TrackerBuilder<RefInfoDesk, &Mesh> {
        TrackerBuilder {
            info_desk: self.info_desk,
            mesh,
        }
    }
}

impl<'a, T: Copy + RealField, GeometryDim: SmallDim, BandDim: SmallDim, Conn>
    TrackerBuilder<&'a DeviceInfoDesk<T, GeometryDim, BandDim>, &'a Mesh<T, GeometryDim, Conn>>
where
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<T, GeometryDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    pub(crate) fn build(self) -> color_eyre::Result<Tracker<'a, T, GeometryDim, BandDim>> {
        let potential = Potential::from_vector(DVector::zeros(self.mesh.vertices().len()));
        let empty_vector: DVector<T> = DVector::zeros(self.mesh.elements().len());
        let charge_densities: Charge<T, BandDim> = Charge::new(
            OVector::<DVector<T>, BandDim>::from_element(empty_vector.clone()),
        )?;
        let current_densities: Current<T, BandDim> =
            Current::new(OVector::<DVector<T>, BandDim>::from_element(empty_vector))?;
        Ok(Tracker {
            info_desk: self.info_desk,
            // mesh: self.mesh,
            potential,
            charge_densities,
            current_densities,
            __marker: PhantomData,
        })
    }
}
