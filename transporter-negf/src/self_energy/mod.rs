mod contact;
mod lo_phonon;
use crate::{
    error::{BuildError, CsrError},
    spectral::{SpectralDiscretisation, SpectralSpace, WavevectorSpace},
};
use nalgebra::{allocator::Allocator, ComplexField, DMatrix, DefaultAllocator, RealField};
use nalgebra_sparse::{
    pattern::{SparsityPattern, SparsityPatternFormatError},
    CsrMatrix,
};
use num_complex::Complex;
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, ElementMethods, Mesh, SmallDim};

#[derive(thiserror::Error, Debug, miette::Diagnostic)]
pub(crate) enum SelfEnergyError {
    #[error(transparent)]
    Computation(#[from] anyhow::Error),
}

#[derive(Clone)]
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
    pub(crate) contact_retarded: Vec<CsrMatrix<Complex<T>>>,
    pub(crate) contact_lesser: Option<Vec<CsrMatrix<Complex<T>>>>,
    pub(crate) incoherent_retarded: Option<Vec<DMatrix<Complex<T>>>>,
    pub(crate) incoherent_lesser: Option<Vec<DMatrix<Complex<T>>>>,
}

pub struct SelfEnergyBuilder<T, RefSpectral, RefMesh> {
    pub(crate) spectral: RefSpectral,
    pub(crate) mesh: RefMesh,
    marker: PhantomData<T>,
}

impl<T: ComplexField> SelfEnergyBuilder<T, (), ()> {
    pub fn new() -> Self {
        Self {
            spectral: (),
            mesh: (),
            marker: PhantomData,
        }
    }
}

impl<T, RefSpectral, RefMesh> SelfEnergyBuilder<T, RefSpectral, RefMesh> {
    pub fn with_spectral_discretisation<Spectral>(
        self,
        spectral: &Spectral,
    ) -> SelfEnergyBuilder<T, &Spectral, RefMesh> {
        SelfEnergyBuilder {
            spectral,
            mesh: self.mesh,
            marker: PhantomData,
        }
    }

    pub fn with_mesh<Mesh>(self, mesh: &Mesh) -> SelfEnergyBuilder<T, RefSpectral, &Mesh> {
        SelfEnergyBuilder {
            spectral: self.spectral,
            mesh,
            marker: PhantomData,
        }
    }
}

/// Coherent builder
impl<'a, T, GeometryDim, Conn>
    SelfEnergyBuilder<T, &'a SpectralSpace<T, ()>, &'a Mesh<T, GeometryDim, Conn>>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
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
        let matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_values)?;

        Ok(SelfEnergy {
            ma: PhantomData,
            mc: PhantomData,
            marker: PhantomData,
            contact_retarded: vec![matrix; self.spectral.number_of_energies()],
            contact_lesser: None,
            incoherent_retarded: None,
            incoherent_lesser: None,
        })
    }
}

/// InCoherent builder
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
        let pattern =
            construct_csr_pattern_from_vertices(&vertices_at_boundary, self.mesh.vertices().len())?;
        let initial_values = vertices_at_boundary
            .iter()
            .map(|_| Complex::from(T::zero()))
            .collect::<Vec<_>>();
        let csrmatrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_values)?;

        let dmatrix = DMatrix::zeros(number_of_vertices_in_core, number_of_vertices_in_core);
        let num_spectral_points =
            self.spectral.number_of_wavevector_points() * self.spectral.number_of_energy_points();

        Ok(SelfEnergy {
            ma: PhantomData,
            mc: PhantomData,
            marker: PhantomData,
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

        Ok(SelfEnergy {
            ma: PhantomData,
            mc: PhantomData,
            marker: PhantomData,
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
