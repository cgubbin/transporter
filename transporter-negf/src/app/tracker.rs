use super::Calculation;
use crate::{
    device::info_desk::DeviceInfoDesk,
    hamiltonian::HamiltonianInfoDesk,
    outer_loop::Potential,
    postprocessor::{Charge, Current},
};
use nalgebra::{allocator::Allocator, DefaultAllocator, OPoint, OVector, RealField};
use ndarray::Array1;
use num_traits::ToPrimitive;
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

/// A tracker struct, holds the state of the solution. We `impl` all InfoDesk methods on the tracker
pub struct Tracker<'a, T: RealField, GeometryDim: SmallDim, BandDim: SmallDim>
where
    // C: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        // + Allocator<T, GeometryDim>
        + Allocator<Array1<T>, BandDim>,
{
    // mesh: &'a Mesh<T, GeometryDim, C>,
    pub(crate) info_desk: &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
    charge_densities: Charge<T, BandDim>,
    current_densities: Current<T, BandDim>,
    potential: Potential<T>,
    calculation: Calculation<T>,
    __marker: PhantomData<GeometryDim>,
}

impl<'a, T: Copy + RealField, GeometryDim: SmallDim, BandDim: SmallDim>
    Tracker<'a, T, GeometryDim, BandDim>
where
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<T, GeometryDim>
        + Allocator<Array1<T>, BandDim>,
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

    pub(crate) fn calculation(&self) -> Calculation<T> {
        self.calculation
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
    DefaultAllocator:
        Allocator<T, BandDim> + Allocator<[T; 3], BandDim> + Allocator<Array1<T>, BandDim>,
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
    DefaultAllocator:
        Allocator<T, BandDim> + Allocator<[T; 3], BandDim> + Allocator<Array1<T>, BandDim>,
{
    type BandDim = BandDim;
    fn potential(&self, vertex_index: usize) -> T {
        self.potential.get(vertex_index)
    }
}

/// A builder struct to aid the construction of the simulation `Tracker`
pub struct TrackerBuilder<T: RealField, RefInfoDesk, RefMesh> {
    info_desk: RefInfoDesk,
    mesh: RefMesh,
    calculation: Calculation<T>,
}

impl<T: RealField> TrackerBuilder<T, (), ()> {
    /// Generate a new `Tracker` for a given `Calculation`
    pub fn new(calculation: Calculation<T>) -> Self {
        Self {
            info_desk: (),
            mesh: (),
            calculation,
        }
    }
}

impl<T: RealField, RefInfoDesk, RefMesh> TrackerBuilder<T, RefInfoDesk, RefMesh> {
    /// attach an impl of the info desk traits
    pub fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> TrackerBuilder<T, &InfoDesk, RefMesh> {
        TrackerBuilder {
            info_desk,
            mesh: self.mesh,
            calculation: self.calculation,
        }
    }

    /// Attach a reference to a `Mesh`
    pub fn with_mesh<Mesh>(self, mesh: &Mesh) -> TrackerBuilder<T, RefInfoDesk, &Mesh> {
        TrackerBuilder {
            info_desk: self.info_desk,
            mesh,
            calculation: self.calculation,
        }
    }
}

impl<'a, T: Copy + RealField + ToPrimitive, GeometryDim: SmallDim, BandDim: SmallDim, Conn>
    TrackerBuilder<T, &'a DeviceInfoDesk<T, GeometryDim, BandDim>, &'a Mesh<T, GeometryDim, Conn>>
where
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<T, GeometryDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<Array1<T>, BandDim>,
{
    /// Build an instance of `Tracker`
    pub fn build(self) -> color_eyre::Result<Tracker<'a, T, GeometryDim, BandDim>> {
        let potential = Potential::from(Array1::zeros(self.mesh.vertices().len()));
        let empty_vector: Array1<T> = Array1::zeros(self.mesh.vertices().len());
        let charge_densities: Charge<T, BandDim> = Charge::new(
            OVector::<Array1<T>, BandDim>::from_element(empty_vector.clone()),
        )?;
        let current_densities: Current<T, BandDim> =
            Current::new(OVector::<Array1<T>, BandDim>::from_element(empty_vector))?;
        Ok(Tracker {
            info_desk: self.info_desk,
            potential,
            charge_densities,
            current_densities,
            calculation: self.calculation,
            __marker: PhantomData,
        })
    }
}
