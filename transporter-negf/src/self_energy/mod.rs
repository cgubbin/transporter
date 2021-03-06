//! # Self Energy
//!
//! Self energies define how the closed electronic system described by a `Hamiltonian` interacts
//! with it's external environment. They define the coherent interaction rates between the closed device
//! and the extended source and drain contacts, and incoherent interactions between electrons
//! within the system and phonons, photons or impurities.
//!
//! We only store the lesser and retarded self-energies for efficiency. The advanced self energy can be
//! computed trivially by taking the conjugate transpose of the retarded self-energy and the greater self
//! energy can then be computed by linear addition of the three

/// The self-energies of the source and drain contacts which add and remove electrons
mod contact;
/// Self energies arising from interaction with longitudinal optic phonons
mod lo_phonon;

use crate::{
    error::{BuildError, CsrError},
    greens_functions::SecurityCheck,
    spectral::{SpectralDiscretisation, SpectralSpace, WavevectorSpace},
};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, RealField};
use nalgebra_sparse::{
    pattern::{SparsityPattern, SparsityPatternFormatError},
    CsrMatrix,
};
use ndarray::Array2;
use num_complex::Complex;
use sprs::CsMat;
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, ElementMethods, Mesh, SmallDim};

#[derive(thiserror::Error, Debug, miette::Diagnostic)]
/// Error type for the self-energy computation
pub(crate) enum SelfEnergyError {
    #[error(transparent)]
    Computation(#[from] anyhow::Error),
    #[error(transparent)]
    SecurityCheck(#[from] SecurityCheck),
}

#[derive(Clone)]
/// Self energy struct which contains the sparse contact components
/// and dense incoherent components
///
/// All self-energies are stored in `Vec` with length equal to the total
/// number of points in the simulation `SpectralSpace`, with ordering as
/// defined in that sub-module
pub struct SelfEnergy<T, GeometryDim, Conn>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub(crate) ma: PhantomData<GeometryDim>,
    pub(crate) mc: PhantomData<Conn>,
    pub(crate) marker: PhantomData<T>,
    pub(crate) security_checks: bool,
    /// The retarded self energy at the device contacts
    pub(crate) contact_retarded: Vec<CsMat<Complex<T>>>,
    /// The lesser self-energy at the device contacts
    pub(crate) contact_lesser: Option<Vec<CsMat<Complex<T>>>>,
    /// The optional retarded self energy within the device
    pub(crate) incoherent_retarded: Option<Vec<Array2<Complex<T>>>>,
    /// The optional lesser self energy within the device
    pub(crate) incoherent_lesser: Option<Vec<Array2<Complex<T>>>>,
}

/// Factory struct to build out self-energy
pub struct SelfEnergyBuilder<T, RefSpectral, RefMesh> {
    pub(crate) spectral: RefSpectral,
    pub(crate) mesh: RefMesh,
    pub(crate) security_checks: bool,
    marker: PhantomData<T>,
}

impl<T: ComplexField> SelfEnergyBuilder<T, (), ()> {
    /// Initialise an empty `SelfEnergyBuilder`
    pub fn new(security_checks: bool) -> Self {
        Self {
            spectral: (),
            mesh: (),
            security_checks,
            marker: PhantomData,
        }
    }
}

impl<T, RefSpectral, RefMesh> SelfEnergyBuilder<T, RefSpectral, RefMesh> {
    /// Attach a spectral space
    pub fn with_spectral_discretisation<Spectral>(
        self,
        spectral: &Spectral,
    ) -> SelfEnergyBuilder<T, &Spectral, RefMesh> {
        SelfEnergyBuilder {
            spectral,
            mesh: self.mesh,
            security_checks: self.security_checks,
            marker: PhantomData,
        }
    }

    /// Attach a mesh
    pub fn with_mesh<Mesh>(self, mesh: &Mesh) -> SelfEnergyBuilder<T, RefSpectral, &Mesh> {
        SelfEnergyBuilder {
            spectral: self.spectral,
            mesh,
            security_checks: self.security_checks,
            marker: PhantomData,
        }
    }
}

/// Builder for coherent self-energies, in which only the self-energies at the contacts
/// are utilised
impl<'a, T, GeometryDim, Conn>
    SelfEnergyBuilder<T, &'a SpectralSpace<T, ()>, &'a Mesh<T, GeometryDim, Conn>>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    /// Build a coherent (k=0) spectral space
    pub fn build_coherent(self) -> Result<SelfEnergy<T, GeometryDim, Conn>, BuildError> {
        Ok(self.build_coherent_inner()?)
    }

    fn build_coherent_inner(self) -> Result<SelfEnergy<T, GeometryDim, Conn>, CsrError> {
        // Collect the indices of all elements at the boundaries
        let vertices_at_boundary: Vec<usize> = self
            .mesh
            .connectivity()
            .iter()
            .enumerate()
            .filter_map(|(vertex_index, vertex_connectivity)| {
                if vertex_connectivity.len() == 1 {
                    Some(vertex_index)
                } else {
                    None
                }
            })
            .collect();
        let pattern =
            construct_csr_pattern_from_vertices(&vertices_at_boundary, self.mesh.vertices().len())?;
        let initial_values = vertices_at_boundary
            .iter()
            .map(|_| Complex::from(T::zero()))
            .collect::<Vec<_>>();
        // We serialize into a `CsrMatrix` because this allows us to handle errors in the event the pattern is invalid
        let matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_values)?;
        // Re-serialize into a CsMat
        let matrix = CsMat::new(
            (matrix.nrows(), matrix.ncols()),
            matrix.row_offsets().to_vec(),
            matrix.col_indices().to_vec(),
            matrix.values().to_vec(),
        );

        Ok(SelfEnergy {
            ma: PhantomData,
            mc: PhantomData,
            marker: PhantomData,
            security_checks: self.security_checks,
            contact_retarded: vec![matrix; self.spectral.number_of_energies()],
            contact_lesser: None,
            incoherent_retarded: None,
            incoherent_lesser: None,
        })
    }
}

/// Builder for incoherent self-energies, where the dense matrices are necessary
impl<'a, T, GeometryDim, Conn>
    SelfEnergyBuilder<
        T,
        &'a SpectralSpace<T, WavevectorSpace<T, GeometryDim, Conn>>,
        &'a Mesh<T, GeometryDim, Conn>,
    >
where
    T: RealField + Copy + num_traits::ToPrimitive,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim> + Send + Sync,
    <Conn as Connectivity<T, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Send + Sync,
{
    /// Build an incoherent spectral space
    pub(crate) fn build_incoherent(
        self,
        lead_length: Option<T>,
    ) -> Result<SelfEnergy<T, GeometryDim, Conn>, BuildError> {
        Ok(self.build_incoherent_inner(lead_length)?)
    }

    fn build_incoherent_inner(
        self,
        lead_length: Option<T>,
    ) -> Result<SelfEnergy<T, GeometryDim, Conn>, CsrError> {
        let number_of_vertices_in_reservoir = if let Some(lead_length) = lead_length {
            (lead_length * T::from_f64(1e-9).unwrap() / self.mesh.elements()[0].0.diameter())
                .to_usize()
                .unwrap()
        } else {
            0
        };
        let number_of_vertices_in_core =
            self.mesh.vertices().len() - 2 * number_of_vertices_in_reservoir;

        // Collect the indices of all elements at the boundaries
        let vertices_at_boundary: Vec<usize> = self
            .mesh
            .connectivity()
            .iter()
            .enumerate()
            .filter_map(|(vertex_index, vertex_connectivity)| {
                if vertex_connectivity.len() == 1 {
                    Some(vertex_index)
                } else {
                    None
                }
            })
            .collect();
        // Use nalgebra_sparse to construct initially because this allows us to error handle before moving to sprs
        let pattern =
            construct_csr_pattern_from_vertices(&vertices_at_boundary, self.mesh.vertices().len())?;
        let initial_values = vertices_at_boundary
            .iter()
            .map(|_| Complex::from(T::zero()))
            .collect::<Vec<_>>();
        let csrmatrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_values)?;
        // Re-serialize into a CsMat
        let csrmatrix = CsMat::new(
            (csrmatrix.nrows(), csrmatrix.ncols()),
            csrmatrix.row_offsets().to_vec(),
            csrmatrix.col_indices().to_vec(),
            csrmatrix.values().to_vec(),
        );

        let dmatrix = Array2::zeros((number_of_vertices_in_core, number_of_vertices_in_core));
        let num_spectral_points =
            self.spectral.number_of_wavevector_points() * self.spectral.number_of_energy_points();

        Ok(SelfEnergy {
            ma: PhantomData,
            mc: PhantomData,
            marker: PhantomData,
            security_checks: self.security_checks,
            contact_retarded: vec![csrmatrix.clone(); num_spectral_points],
            contact_lesser: Some(vec![csrmatrix; num_spectral_points]),
            incoherent_retarded: Some(vec![dmatrix.clone(); num_spectral_points]),
            incoherent_lesser: Some(vec![dmatrix; num_spectral_points]),
        })
    }
}

pub(crate) fn construct_csr_pattern_from_vertices(
    boundary_vertices: &[usize],
    total_number_of_vertices: usize,
) -> Result<SparsityPattern, SparsityPatternFormatError> {
    let col_indices = boundary_vertices.to_vec(); // The self energies are on the diagonal elements
    let mut count = 0_usize;
    let mut row_offsets = vec![0];
    row_offsets.extend((0..total_number_of_vertices).map(|row| {
        if col_indices.contains(&(row)) {
            count += 1;
        }
        count
    }));

    SparsityPattern::try_from_offsets_and_indices(
        total_number_of_vertices,
        total_number_of_vertices,
        row_offsets,
        col_indices,
    )
}

/// Coherent impl with wavevector dispersion
impl<'a, T, GeometryDim, Conn>
    SelfEnergyBuilder<
        T,
        &'a SpectralSpace<T, WavevectorSpace<T, GeometryDim, Conn>>,
        &'a Mesh<T, GeometryDim, Conn>,
    >
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim> + Send + Sync,
    <Conn as Connectivity<T, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator: Allocator<T, GeometryDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Send + Sync,
{
    pub(crate) fn build_coherent(self) -> Result<SelfEnergy<T, GeometryDim, Conn>, BuildError> {
        Ok(self.build_coherent_inner()?)
    }

    fn build_coherent_inner(self) -> Result<SelfEnergy<T, GeometryDim, Conn>, CsrError> {
        // TODO Should take a Vec<DMatrix<T>> -> which corresponds to the spectral space
        // Collect the indices of all elements at the boundaries
        let vertices_at_boundary: Vec<usize> = self
            .mesh
            .connectivity()
            .iter()
            .enumerate()
            .filter_map(|(vertex_index, vertex_connectivity)| {
                if vertex_connectivity.len() == 1 {
                    Some(vertex_index)
                } else {
                    None
                }
            })
            .collect();
        let pattern =
            construct_csr_pattern_from_vertices(&vertices_at_boundary, self.mesh.vertices().len())?;
        let initial_values = vertices_at_boundary
            .iter()
            .map(|_| Complex::from(T::zero()))
            .collect::<Vec<_>>();
        let matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_values)?;

        // Re-serialize into a CsMat
        // TODO do this in one step with a custom error handling rather than passing through the sparse function
        let matrix = CsMat::new(
            (matrix.nrows(), matrix.ncols()),
            matrix.row_offsets().to_vec(),
            matrix.col_indices().to_vec(),
            matrix.values().to_vec(),
        );

        Ok(SelfEnergy {
            ma: PhantomData,
            mc: PhantomData,
            marker: PhantomData,
            security_checks: self.security_checks,
            contact_retarded: vec![
                matrix;
                self.spectral.number_of_wavevector_points()
                    * self.spectral.number_of_energy_points()
            ],
            contact_lesser: None,
            incoherent_retarded: None,
            incoherent_lesser: None,
        })
    }
}

pub(crate) trait SelfEnergyInfoDesk<T: RealField> {
    fn get_fermi_level_at_source(&self) -> T;
}

#[cfg(test)]
mod test {
    #[test]
    fn test_contact_self_energy_sparsity_pattern() {
        let nrows = 10;
        let boundary_elements = [0, 4, 6, nrows - 1];
        let values = vec![1_f64; boundary_elements.len()];
        let pattern =
            super::construct_csr_pattern_from_vertices(&boundary_elements, nrows).unwrap();
        let matrix =
            nalgebra_sparse::CsrMatrix::try_from_pattern_and_values(pattern, values).unwrap();
        let dense = nalgebra_sparse::convert::serial::convert_csr_dense(&matrix);
        println!("{dense}");
    }
}
