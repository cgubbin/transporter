use crate::device::info_desk::DeviceInfoDesk;
use nalgebra::{allocator::Allocator, DVector, DefaultAllocator, OPoint, OVector, RealField};
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

/// A tracker struct, holds the state of the solution. We `impl` all InfoDesk methods on the tracker
pub(crate) struct Tracker<'a, T: RealField, GeometryDim: SmallDim, BandDim: SmallDim, C>
where
    C: Connectivity<T, GeometryDim>,
    DefaultAllocator:
        Allocator<T, BandDim> + Allocator<T, GeometryDim> + Allocator<[T; 3], BandDim>,
{
    mesh: &'a Mesh<T, GeometryDim, C>,
    info_desk: &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
    charge_densities: nalgebra::DVector<nalgebra::OVector<T, BandDim>>,
    potential: DVector<T>,
    __marker: PhantomData<GeometryDim>,
}

impl<T: Copy + RealField, BandDim: SmallDim, GeometryDim: SmallDim, C>
    crate::hamiltonian::HamiltonianInfoDesk<T> for Tracker<'_, T, GeometryDim, BandDim, C>
where
    C: Connectivity<T, GeometryDim>,
    DefaultAllocator:
        Allocator<T, BandDim> + Allocator<T, GeometryDim> + Allocator<[T; 3], BandDim>,
{
    type BandDim = BandDim;
    type GeometryDim = GeometryDim;

    fn get_band_levels(&self, region_index: usize) -> &OPoint<T, Self::BandDim> {
        &self.info_desk.band_offsets[region_index]
    }

    fn get_effective_mass(&self, region_index: usize, band_index: usize) -> &[T; 3] {
        &self.info_desk.effective_masses[region_index][(band_index, 0)]
    }

    fn potential(&self, element_index: usize) -> T {
        let vertex_indices = self.mesh.get_vertex_indices_in_element(element_index);
        (self.potential[vertex_indices[0]] + self.potential[vertex_indices[1]])
            / (T::one() + T::one())
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

impl<'a, T: RealField, GeometryDim: SmallDim, Conn, BandDim: SmallDim>
    TrackerBuilder<&'a DeviceInfoDesk<T, GeometryDim, BandDim>, &'a Mesh<T, GeometryDim, Conn>>
where
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator:
        Allocator<T, BandDim> + Allocator<T, GeometryDim> + Allocator<[T; 3], BandDim>,
{
    pub(crate) fn build(self) -> Tracker<'a, T, GeometryDim, BandDim, Conn> {
        let potential = DVector::zeros(self.mesh.vertices().len());
        let element: OVector<T, BandDim> = OVector::<T, BandDim>::zeros();
        let charge_densities: DVector<OVector<T, BandDim>> =
            DVector::<OVector<T, BandDim>>::from_element(self.mesh.vertices().len(), element);
        Tracker {
            info_desk: self.info_desk,
            mesh: self.mesh,
            potential,
            charge_densities,
            __marker: PhantomData,
        }
    }
}
