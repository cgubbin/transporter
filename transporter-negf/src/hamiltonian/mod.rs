//! Hamiltonian module
//!
//! Creates a Hamiltonian structure for use in the NEGF calculation:
//!
//! The Hamiltonian has three components:
//! - `fixed`: Holds the component which is unchanging through the calculation -> this is defined by the geometry and band structure
//! - `potential`: Holds the diagonal contribution from the electrostatic potential at the current step
//! - `wavevector`: Holds the dispersive part, proportional to the transverse electronic wavevector
//!
//! A Hamiltonian is constructed through the `HamiltonianBuilder` class from a `mesh: Mesh` and a `tracker: Tracker` as
//!
//! ```ignore
//! HamiltonianBuilder::new()
//!     .with_mesh(&mesh)
//!     .with_tracker(&tracker)
//!     .build();
//! ```
//!
//! The Hamiltonian matrix is of dimension `num_elements * num_bands` where `num_elements` is the number of elements in the mesh
//! and `num_bands` is the number of transport bands considered in the problem. The independent Hamiltonians for each band are stacked
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
use ndarray::Array1;
use ndarray_linalg::eig::EigVals;
use num_complex::Complex;
use sprs::CsMat;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

#[derive(Clone, Debug)]
/// The Hamiltonian wrapper. Data is stored in `CsMat` as the differential operator is sparse, although the constructor
/// should be generic it is intended to work with nearest neighbour coupling schemes
pub struct Hamiltonian<T: Copy + RealField> {
    /// Contains the fixed component of the Hamiltonian, comprising the differential operator and conduction offsets
    fixed: CsMat<T>,
    /// The CsrMatrix containing the potential at the current stage of the calculation, this is diagonal
    potential: CsMat<T>,
    /// The wavevector dependent component. This component is multiplied by the wavevector during the calculation
    wavevector: CsMat<T>,
}

/// An InfoDesk trait providing all the necessary external information required to construct the Hamiltonian
pub trait HamiltonianInfoDesk<T: RealField>: PotentialInfoDesk<T>
where
    DefaultAllocator: Allocator<T, Self::BandDim>,
{
    /// Type alias for the spatial dimension of the problem (1D, 2D,...)
    type GeometryDim: SmallDim;
    /// Returns the level of all the bands considered in the system in region `region_index`
    fn get_band_levels(&self, region_index: usize) -> &OPoint<T, Self::BandDim>;
    /// Returns the effective mass along the three Cartesian axis [x, y, z]
    fn get_effective_mass(&self, region_index: usize, band_index: usize) -> &[T; 3];
    /// Return the geometrical dimension as a `usize`, 1D->1, 2D->2, ...
    fn geometry_dim(&self) -> usize {
        Self::GeometryDim::dim()
    }
}

/// An InfoDesk trait providing all the information to construct the potential contribution to a `Hamiltonian`
pub trait PotentialInfoDesk<T: RealField> {
    /// A type alias for the number of bands considered in the transport problem
    type BandDim: SmallDim;
    /// Return the potential at mesh element `element` index by averaging over the values at each vertex providec
    fn potential(&self, vertex_index: usize) -> T;
    /// Returns the number of bands in the problem as a `usize
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

    pub(crate) fn calculate_total(&self, wavevector: T) -> CsMat<T> {
        sprs::binop::csmat_binop(
            sprs::binop::csmat_binop(self.fixed.view(), self.potential.view(), |x, y| x.sub(*y))
                .view(),
            self.wavevector.view(),
            |x, y| x.add(*y * wavevector.powi(2)),
        )
    }
}

impl Hamiltonian<f64> {
    // Find the eigenvalues of the closed system
    pub(crate) fn eigenvalues(
        &self,
        wavevector: f64,
        initial_potential: &Array1<f64>,
    ) -> Result<Array1<Complex<f64>>, ndarray_linalg::error::LinalgError> {
        // Create the dense Hamiltonian
        let total = self.calculate_total(wavevector).to_dense()
            - ndarray::Array2::from_diag(initial_potential);
        // Find the eigenvalues
        total.eigvals()
    }
}

/// Builder for a Hamiltonian from the reference to a Mesh and an object implementing HamiltonianInfoDesk
pub struct HamiltonianBuilder<RefInfoDesk, RefMesh> {
    info_desk: RefInfoDesk,
    mesh: RefMesh,
}

impl Default for HamiltonianBuilder<(), ()> {
    /// Initialize an empty instand of HamiltonianBuilder
    fn default() -> Self {
        Self {
            info_desk: (),
            mesh: (),
        }
    }
}

impl<RefInfoDesk, RefMesh> HamiltonianBuilder<RefInfoDesk, RefMesh> {
    /// Attach a mesh instance
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

impl<T> Hamiltonian<T>
where
    T: Copy + RealField,
{
    /// Build the operator components for the Hamiltonian
    ///
    /// This function takes the components provided to `HamiltonianBuilder` and constructs the three
    /// `CsMat` comprising a `Hamiltonian`
    fn build_operator<C, InfoDesk>(
        info_desk: &InfoDesk,
        mesh: &Mesh<T, InfoDesk::GeometryDim, C>,
    ) -> Result<Self, BuildError>
    where
        InfoDesk: HamiltonianInfoDesk<T> + PotentialInfoDesk<T>,
        C: Connectivity<T, InfoDesk::GeometryDim>,
        DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
    {
        // let term = console::Term::stdout();
        // Build out the constructors
        let vertex_assembler = VertexAssemblerBuilder::new()
            .with_info_desk(info_desk)
            .with_mesh(mesh)
            .build();
        let hamiltonian_constructor: CsrAssembler<T> =
            CsrAssembler::from_vertex_assembler(&vertex_assembler)?;

        // Build the fixed component: the differential operator and band offsets
        // term.move_cursor_to(0, 2).unwrap();
        // term.clear_to_end_of_screen().unwrap();
        tracing::trace!("Assembling hamiltonian differential operator");
        let fixed = hamiltonian_constructor.assemble_fixed(&vertex_assembler)?;
        // Assemble the potential into a diagonal CsrMatrix
        // term.move_cursor_to(0, 2).unwrap();
        // term.clear_to_end_of_screen().unwrap();
        tracing::trace!("Initialising the potential diagonal");
        let potential = hamiltonian_constructor.assemble_potential(&vertex_assembler)?;
        // Assemble the dispersive component
        // term.move_cursor_to(0, 2).unwrap();
        // term.clear_to_end_of_screen().unwrap();
        tracing::trace!("Assembling the dispersive diagonal");
        let wavevector = hamiltonian_constructor.assemble_wavevector(&vertex_assembler)?;

        Ok(Self {
            fixed,
            potential,
            wavevector,
        })
    }

    /// Return the number of rows in the full `Hamiltonian` matrix
    pub(crate) fn num_rows(&self) -> usize {
        self.fixed.outer_dims()
    }
}

/// A helper trait to get the elements at the source contact and drain contact
/// This is only really valid in 1D but this is NOT reflected in the current impl
pub(crate) trait AccessMethods<T> {
    fn get_elements_at_source(&self) -> [T; 2];
    fn get_elements_at_drain(&self) -> [T; 2];
}

impl<T: Copy + RealField> AccessMethods<T> for CsMat<T> {
    fn get_elements_at_source(&self) -> [T; 2] {
        [self.data()[1], self.data()[0]]
    }

    fn get_elements_at_drain(&self) -> [T; 2] {
        let num_points = self.data().len();
        [self.data()[num_points - 2], self.data()[num_points - 1]]
    }
}
