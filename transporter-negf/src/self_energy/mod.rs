mod contact;

use crate::{greens_functions::GreensFunctionMethods, hamiltonian::Hamiltonian};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, RealField};
use num_complex::Complex;
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

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

use crate::spectral::{SpectralSpace, WavevectorSpace};
use nalgebra::DMatrix;
use nalgebra_sparse::CsrMatrix;

/// Coherent impl
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
            retarded: vec![matrix.clone(); self.spectral.number_of_energies()],
            //lesser: vec![matrix.clone(); self.spectral.number_of_energies()],
            //greater: vec![matrix; self.spectral.number_of_energies()],
        })
    }
}

impl<T, GeometryDim, Conn> SelfEnergy<T, GeometryDim, Conn, CsrMatrix<Complex<T>>>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    // Updates the coherent Self Energy at the contacts into the scratch matrix held in `self`
    pub(crate) fn recalculate(
        &mut self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        hamiltonian: &Hamiltonian<T>,
        spectral_space: &SpectralSpace<T, ()>,
    ) -> color_eyre::Result<()> {
        match GeometryDim::dim() {
            1 => {
                // We have 2 elements in 1D, with positions first and last in the mesh
                let n_elements = mesh.elements().len();
                let hamiltonian = hamiltonian.calculate_total(T::zero().real()); // We are at 0 wavevector for this spectral_space
                for (boundary_element, diagonal_element, idx) in [
                    (hamiltonian.values()[1], hamiltonian.values()[0], 0),
                    (
                        hamiltonian.values()[hamiltonian.values().len() - 2],
                        hamiltonian.values()[hamiltonian.values().len() - 1],
                        n_elements - 1,
                    ),
                ] {
                    let ec_plus_u_plus_ek = diagonal_element - boundary_element - boundary_element;
                    let connected_idx = mesh.element_connectivity()[idx];
                    assert_eq!(connected_idx.len(), 1);
                    let a = (mesh.get_element_midpoint(idx)
                        - mesh.get_element_midpoint(connected_idx[0]))
                    .norm();
                    let imaginary_unit = Complex::new(T::zero(), T::one());
                    for (jdx, &energy) in spectral_space.iter_energy().enumerate() {
                        let z = Complex::from(
                            T::one()
                                - (energy - ec_plus_u_plus_ek)
                                    / (boundary_element + boundary_element),
                        );
                        let k1 = Complex::from(T::one() / a) * z.acos();
                        if idx == 0 {
                            self.retarded[jdx].values_mut()[0] = -Complex::from(boundary_element)
                                * (imaginary_unit * k1 * T::from_real(a)).exp();
                        } else {
                            self.retarded[jdx].values_mut()[1] = -Complex::from(boundary_element)
                                * (imaginary_unit * k1 * T::from_real(a)).exp();
                        }
                    }
                }
                Ok(())
            }
            _ => unimplemented!("No self-energy implementation for 2D geometries"),
        }
    }
}

use color_eyre::eyre::eyre;
use nalgebra_sparse::pattern::SparsityPattern;
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

/// Incoherent impl
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
    pub(crate) fn build(self) -> SelfEnergy<T, GeometryDim, Conn, DMatrix<Complex<T>>> {
        // TODO Should take a Vec<DMatrix<T>> -> which corresponds to the spectral space
        // In the incoherent case the inner loop iterates between updating the Green's functions and self-energies
        // so we need to retain the full stack
        //SelfEnergy {
        //ma: PhantomData,
        //mc: PhantomData,
        //marker: PhantomData,
        //matrix: todo!(),
        //}
        todo!()
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
