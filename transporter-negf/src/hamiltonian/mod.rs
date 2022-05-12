//! Hamiltonian module
//!
//! Creates a Hamiltonian structure for use in the NEGF calculation:
//!
//! The Hamiltonian has three components:
//! - `fixed`: Holds the component which is unchanging through the calculation
//! - `potential`: Holds the contribution from the electrostatic potential at the current step
//! - `wavevector`: Holds the dispersive part, proportional to the transverse electronic wavevector
//!
//! A Hamiltonian is constructed through the `HamiltonianBuilder` class from `mesh: Mesh` and `tracker: Tracker` as
//!
//! ```ignore
//! HamiltonianBuilder::new()
//!     .with_mesh(&mesh)
//!     .with_tracker(&tracker)
//!     .build();
//! ```
//!
//! The Hamiltonian matrix is of dimension `num_elements * num_bands` where `num_elements` is the number of elements in the mesh
//! and `num_bands` is the number of conduction bands considered in the problem. The independent Hamiltonians for each band are stacked
//! in block-diagonal form. The Hamiltonian is evaluated in the elements of the mesh, contrasting the potential which is evaluated at the
//! nodes of the mesh.
//!

pub mod global;
mod local;

pub use global::*;

use crate::{
    error::{BuildError, CsrError},
    utilities::assemblers::VertexAssemblerBuilder,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint, RealField};
use nalgebra_sparse::CsrMatrix;
#[cfg(feature = "ndarray")]
use sprs::CsMat;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

#[derive(Debug)]
/// The Hamiltonian wrapper. Data is stored in `CsrMatrix` as the differential operator is sparse, although the constructor
/// should be generic it is intended to work with nearest neighbour coupling schemes
pub struct Hamiltonian<T: Copy + RealField> {
    /// Contains the fixed component of the Hamiltonian, comprising the differential operator and conduction offsets
    #[cfg(not(feature = "ndarray"))]
    fixed: CsrMatrix<T>,
    /// The CsrMatrix containing the potential at the current stage of the calculation, this is diagonal
    #[cfg(not(feature = "ndarray"))]
    potential: CsrMatrix<T>,
    /// The wavevector dependent component. This component is multiplied by the wavevector during the calculation
    #[cfg(not(feature = "ndarray"))]
    wavevector: CsrMatrix<T>,
    /// Contains the fixed component of the Hamiltonian, comprising the differential operator and conduction offsets
    #[cfg(feature = "ndarray")]
    fixed: CsMat<T>,
    /// The CsrMatrix containing the potential at the current stage of the calculation, this is diagonal
    #[cfg(feature = "ndarray")]
    potential: CsMat<T>,
    /// The wavevector dependent component. This component is multiplied by the wavevector during the calculation
    #[cfg(feature = "ndarray")]
    wavevector: CsMat<T>,
}

/// An InfoDesk trait providing all the necessary external information required to construct the Hamiltonian
pub trait HamiltonianInfoDesk<T: RealField>: PotentialInfoDesk<T>
where
    DefaultAllocator: Allocator<T, Self::BandDim>,
{
    /// Type alias for the number of carrier bands in the problem
    // type BandDim: SmallDim;
    /// Type alias for the geometry of the problem (1D, 2D,...)
    type GeometryDim: SmallDim;
    /// Find the level of all the bands considered in the system in region `region_index`
    fn get_band_levels(&self, region_index: usize) -> &OPoint<T, Self::BandDim>;
    /// Find the effective mass along the three Cartesian axis
    fn get_effective_mass(&self, region_index: usize, band_index: usize) -> &[T; 3];
    /// Return the potential at mesh element `element` index by averaging over the values at each vertex providec
    // fn potential(&self, vertex_indices: &[usize]) -> T;
    /// Return the geometrical dimension as a `usize`
    fn geometry_dim(&self) -> usize {
        Self::GeometryDim::dim()
    }
    // Return the number of bands as a `usize`
    // fn number_of_bands(&self) -> usize {
    // Self::BandDim::dim()
    // }
}

pub trait PotentialInfoDesk<T: RealField> {
    type BandDim: SmallDim;
    /// Return the potential at mesh element `element` index by averaging over the values at each vertex providec
    fn potential(&self, vertex_index: usize) -> T;
    fn number_of_bands(&self) -> usize {
        Self::BandDim::dim()
    }
}

impl<T: Copy + RealField> Hamiltonian<T> {
    /// Updates the `potential` component from an impl of the trait `HamiltonianInfoDesk`
    pub(crate) fn update_potential<GeometryDim: SmallDim, C, InfoDesk>(
        &mut self,
        info_desk: &InfoDesk,
        mesh: &Mesh<T, GeometryDim, C>,
    ) -> Result<(), BuildError>
    where
        InfoDesk: PotentialInfoDesk<T>,
        C: Connectivity<T, GeometryDim>,
        DefaultAllocator: Allocator<T, InfoDesk::BandDim> + Allocator<T, GeometryDim>,
    {
        let vertex_assembler = VertexAssemblerBuilder::new()
            .with_info_desk(info_desk)
            .with_mesh(mesh)
            .build();
        CsrAssembler::assemble_potential_into(&vertex_assembler, &mut self.potential)
    }

    /// Finds the total Hamiltonian at fixed `wavevector`
    #[cfg(not(feature = "ndarray"))]
    pub(crate) fn calculate_total(&self, wavevector: T) -> CsrMatrix<T> {
        &self.fixed - &self.potential + &self.wavevector * wavevector.powi(2)
    }

    #[cfg(feature = "ndarray")]
    pub(crate) fn calculate_total(&self, wavevector: T) -> CsMat<T> {
        sprs::binop::csmat_binop(
            sprs::binop::csmat_binop(self.fixed.view(), self.potential.view(), |x, y| x.sub(*y))
                .view(),
            self.wavevector.view(),
            |x, y| x.add(*y * wavevector.powi(2)),
        )
        // &self.fixed - &self.potential + &self.wavevector * wavevector.powi(2)
    }
}

/// Builder for a Hamiltonian from the reference to a Mesh and an object implementing HamiltonianInfoDesk
pub struct HamiltonianBuilder<RefInfoDesk, RefMesh> {
    info_desk: RefInfoDesk,
    mesh: RefMesh,
}

impl HamiltonianBuilder<(), ()> {
    /// Initialize an empty instand of HamiltonianBuilder
    pub fn new() -> Self {
        Self {
            info_desk: (),
            mesh: (),
        }
    }
}

impl<RefInfoDesk, RefMesh> HamiltonianBuilder<RefInfoDesk, RefMesh> {
    /// Attach a mesh
    pub fn with_mesh<Mesh>(self, mesh: &Mesh) -> HamiltonianBuilder<RefInfoDesk, &Mesh> {
        HamiltonianBuilder {
            info_desk: self.info_desk,
            mesh,
        }
    }

    /// Attach an implementation of `HamiltonianInfoDesk`
    pub fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> HamiltonianBuilder<&InfoDesk, RefMesh> {
        HamiltonianBuilder {
            info_desk,
            mesh: self.mesh,
        }
    }
}

impl<T, C, InfoDesk> HamiltonianBuilder<&InfoDesk, &Mesh<T, InfoDesk::GeometryDim, C>>
where
    T: Copy + RealField,
    C: Connectivity<T, InfoDesk::GeometryDim>,
    InfoDesk: HamiltonianInfoDesk<T>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    /// Builds an instance of `Hamiltonian` from a `HamiltonianBuilder`
    #[tracing::instrument(name = "Hamiltonian Builder", level = "info", skip(self))]
    pub fn build(self) -> Result<Hamiltonian<T>, BuildError> {
        Hamiltonian::build_operator(self.info_desk, self.mesh)
    }
}

// TODO Delete this impl when the Greens Functions have been updated
#[cfg(not(feature = "ndarray"))]
impl<T> AsRef<CsrMatrix<T>> for Hamiltonian<T>
where
    T: Copy + RealField,
{
    fn as_ref(&self) -> &CsrMatrix<T> {
        &self.fixed
    }
}

// TODO Delete this impl when the Greens Functions have been updated
#[cfg(feature = "ndarray")]
impl<T> AsRef<CsMat<T>> for Hamiltonian<T>
where
    T: Copy + RealField,
{
    fn as_ref(&self) -> &CsMat<T> {
        &self.fixed
    }
}

impl<T> Hamiltonian<T>
where
    T: Copy + RealField,
{
    /// Build the operator components for the Hamiltonian
    ///
    /// This function takes the components provided to `HamiltonianBuilder` and constructs the three
    /// `CsrMatrix` comprising a `Hamiltonian`
    fn build_operator<C, InfoDesk>(
        info_desk: &InfoDesk,
        mesh: &Mesh<T, InfoDesk::GeometryDim, C>,
    ) -> Result<Self, BuildError>
    where
        InfoDesk: HamiltonianInfoDesk<T> + PotentialInfoDesk<T>,
        C: Connectivity<T, InfoDesk::GeometryDim>,
        DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
    {
        let term = console::Term::stdout();
        // Build out the constructors
        let vertex_assembler = VertexAssemblerBuilder::new()
            .with_info_desk(info_desk)
            .with_mesh(mesh)
            .build();
        let hamiltonian_constructor: CsrAssembler<T> =
            CsrAssembler::from_vertex_assembler(&vertex_assembler)?;

        // Build the fixed component: the differential operator and band offsets
        term.move_cursor_to(0, 2).unwrap();
        term.clear_to_end_of_screen().unwrap();
        tracing::trace!("Assembling hamiltonian differential operator");
        let fixed = hamiltonian_constructor.assemble_fixed(&vertex_assembler)?;
        // Assemble the potential into a diagonal CsrMatrix
        term.move_cursor_to(0, 2).unwrap();
        term.clear_to_end_of_screen().unwrap();
        tracing::trace!("Initialising the potential diagonal");
        let potential = hamiltonian_constructor.assemble_potential(&vertex_assembler)?;
        // Assemble the dispersive component
        term.move_cursor_to(0, 2).unwrap();
        term.clear_to_end_of_screen().unwrap();
        tracing::trace!("Assembling the dispersive diagonal");
        let wavevector = hamiltonian_constructor.assemble_wavevector(&vertex_assembler)?;

        Ok(Self {
            fixed,
            potential,
            wavevector,
        })
    }

    #[cfg(not(feature = "ndarray"))]
    pub(crate) fn num_rows(&self) -> usize {
        self.fixed.nrows()
    }

    #[cfg(feature = "ndarray")]
    pub(crate) fn num_rows(&self) -> usize {
        self.fixed.outer_dims()
    }
}

pub(crate) trait AccessMethods<T> {
    fn get_elements_at_source(&self) -> [T; 2];
    fn get_elements_at_drain(&self) -> [T; 2];
}

#[cfg(not(feature = "ndarray"))]
impl<T: Copy + RealField> AccessMethods<T> for CsrMatrix<T> {
    fn get_elements_at_source(&self) -> [T; 2] {
        let x = self.values();
        todo!()
    }

    fn get_elements_at_drain(&self) -> [T; 2] {
        self.values();
        todo!()
    }
}

#[cfg(feature = "ndarray")]
impl<T: Copy + RealField> AccessMethods<T> for CsMat<T> {
    fn get_elements_at_source(&self) -> [T; 2] {
        [self.data()[0], self.data()[1]]
    }

    fn get_elements_at_drain(&self) -> [T; 2] {
        let num_points = self.data().len();
        [self.data()[num_points - 2], self.data()[num_points - 1]]
    }
}
