mod contact;
use crate::{
    greens_functions::GreensFunctionMethods,
    spectral::{SpectralDiscretisation, SpectralSpace, WavevectorSpace},
};
use color_eyre::eyre::eyre;
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, RealField};
use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};
use num_complex::Complex;
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

#[derive(Clone)]
pub(crate) struct SelfEnergy<T, GeometryDim, Conn, Matrix>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    Matrix: GreensFunctionMethods<T>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub(crate) ma: PhantomData<GeometryDim>,
    pub(crate) mc: PhantomData<Conn>,
    pub(crate) marker: PhantomData<T>,
    pub(crate) retarded: Vec<Matrix>,
    //pub(crate) lesser: Vec<Matrix>,
    //pub(crate) greater: Vec<Matrix>,
}

pub(crate) struct SelfEnergyBuilder<T, RefSpectral, RefMesh> {
    spectral: RefSpectral,
    mesh: RefMesh,
    marker: PhantomData<T>,
}

impl<T: ComplexField> SelfEnergyBuilder<T, (), ()> {
    pub(crate) fn new() -> Self {
        Self {
            spectral: (),
            mesh: (),
            marker: PhantomData,
        }
    }
}

impl<T, RefSpectral, RefMesh> SelfEnergyBuilder<T, RefSpectral, RefMesh> {
    pub(crate) fn with_spectral_discretisation<Spectral>(
        self,
        spectral: &Spectral,
    ) -> SelfEnergyBuilder<T, &Spectral, RefMesh> {
        SelfEnergyBuilder {
            spectral,
            mesh: self.mesh,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_mesh<Mesh>(self, mesh: &Mesh) -> SelfEnergyBuilder<T, RefSpectral, &Mesh> {
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
    pub(crate) fn build(
        self,
    ) -> color_eyre::Result<SelfEnergy<T, GeometryDim, Conn, CsrMatrix<Complex<T>>>> {
        // In the coherent case there is no inner loop -> there is no need to retain the calculated self-energies

        // Collect the indices of all elements at the boundaries
        let elements_at_boundary: Vec<usize> = self
            .mesh
            .iter_element_connectivity()
            .enumerate()
            .filter_map(|(element_index, element_connectivity)| {
                if element_connectivity.is_at_boundary() {
                    Some(element_index)
                } else {
                    None
                }
            })
            .collect();
        let pattern =
            construct_csr_pattern_from_elements(&elements_at_boundary, self.mesh.elements().len())?;
        let initial_values = elements_at_boundary
            .iter()
            .map(|_| Complex::from(T::zero()))
            .collect::<Vec<_>>();
        let matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_values)
            .map_err(|e| eyre!("Failed to initialise sparse self energy matrix {}", e))?;

        Ok(SelfEnergy {
            ma: PhantomData,
            mc: PhantomData,
            marker: PhantomData,
            retarded: vec![matrix; self.spectral.number_of_energies()],
            //lesser: vec![matrix.clone(); self.spectral.number_of_energies()],
            //greater: vec![matrix; self.spectral.number_of_energies()],
        })
    }
}

fn construct_csr_pattern_from_elements(
    boundary_elements: &[usize],
    total_number_of_elements: usize,
) -> color_eyre::Result<SparsityPattern> {
    let col_indices = boundary_elements.to_vec(); // The self energies are on the diagonal elements
    let mut count = 0_usize;
    let mut row_offsets = vec![0];
    row_offsets.extend((0..total_number_of_elements).map(|row| {
        if col_indices.contains(&(row)) {
            count += 1;
        }
        count
    }));

    SparsityPattern::try_from_offsets_and_indices(
        total_number_of_elements,
        total_number_of_elements,
        row_offsets,
        col_indices,
    )
    .map_err(|e| eyre!("Failed to create sparsity pattern {:?}", e))
}

/// Coherent impl with wavevector dispersion
impl<'a, T, GeometryDim, Conn>
    SelfEnergyBuilder<
        T,
        &'a SpectralSpace<T, WavevectorSpace<T::RealField, GeometryDim, Conn>>,
        &'a Mesh<T, GeometryDim, Conn>,
    >
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    pub(crate) fn build(
        self,
    ) -> color_eyre::Result<SelfEnergy<T, GeometryDim, Conn, CsrMatrix<Complex<T>>>> {
        // TODO Should take a Vec<DMatrix<T>> -> which corresponds to the spectral space
        // Collect the indices of all elements at the boundaries
        let elements_at_boundary: Vec<usize> = self
            .mesh
            .iter_element_connectivity()
            .enumerate()
            .filter_map(|(element_index, element_connectivity)| {
                if element_connectivity.is_at_boundary() {
                    Some(element_index)
                } else {
                    None
                }
            })
            .collect();
        let pattern =
            construct_csr_pattern_from_elements(&elements_at_boundary, self.mesh.elements().len())?;
        let initial_values = elements_at_boundary
            .iter()
            .map(|_| Complex::from(T::zero()))
            .collect::<Vec<_>>();
        let matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_values)
            .map_err(|e| eyre!("Failed to initialise sparse self energy matrix {}", e))?;

        Ok(SelfEnergy {
            ma: PhantomData,
            mc: PhantomData,
            marker: PhantomData,
            retarded: vec![
                matrix;
                self.spectral.number_of_wavevector_points()
                    * self.spectral.number_of_energy_points()
            ],
            //lesser: vec![matrix.clone(); self.spectral.number_of_energies()],
            //greater: vec![matrix; self.spectral.number_of_energies()],
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
            super::construct_csr_pattern_from_elements(&boundary_elements, nrows).unwrap();
        let matrix =
            nalgebra_sparse::CsrMatrix::try_from_pattern_and_values(pattern, values).unwrap();
        let dense = nalgebra_sparse::convert::serial::convert_csr_dense(&matrix);
        println!("{dense}");
    }
}
