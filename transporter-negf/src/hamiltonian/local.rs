//! Element level constructors for the Hamiltonian matrix
//!
//! This submodule constructs the components of the Hamiltonian differential operator, and diagonal
//! for a single element of the mesh, over `NumBands` (ie: the number of carrier bands in the problem).

use super::{BuildError, HamiltonianInfoDesk, PotentialInfoDesk};
use crate::{
    constants::{ELECTRON_CHARGE, ELECTRON_MASS, HBAR},
    utilities::assemblers::{VertexAssembler, VertexConnectivityAssembler},
};
use nalgebra::{allocator::Allocator, DefaultAllocator, OPoint, OVector, RealField};
use ndarray::{Array2, ArrayViewMut1, ArrayViewMut2};
use transporter_mesher::{Assignment, Connectivity, Mesh, SmallDim};

/// Helper trait to construct the diagonal elements of a differential operator
pub(crate) trait AssembleVertexHamiltonianDiagonal<T: RealField>:
    VertexConnectivityAssembler
{
    /// Assembles the wavevector component into `output` for the element at `element_index`. Takes an output vector of length
    /// `num_bands` which is enforced by an assertion
    fn assemble_vertex_diagonal_into(
        &self,
        vertex_index: usize,
        output: ArrayViewMut1<T>,
    ) -> Result<(), BuildError>;
}

/// Helper trait to construct the fixed component of the operator (ie: the differential bit)
pub(crate) trait AssembleVertexHamiltonianMatrix<T: RealField>:
    VertexConnectivityAssembler
{
    /// Takes an output matrix of dimension `num_bands` * `num_connections + 1` which is enforced by an assertion, and fills with the fixed
    ///  component of the Hamiltonian
    fn assemble_vertex_matrix_into(
        &self,
        vertex_index: usize,
        output: ArrayViewMut2<T>,
    ) -> Result<(), BuildError>;

    fn assemble_vertex_matrix(
        &self,
        vertex_index: usize,
        num_connections: usize,
        num_bands: usize,
    ) -> Result<Array2<T>, BuildError> {
        let mut output = Array2::zeros((num_bands, num_connections + 1));
        self.assemble_vertex_matrix_into(vertex_index, output.view_mut())?;
        Ok(output)
    }
}

/// Implement the `HamiltonianInfoDesk` trait for the element assembler to reduce verbiosity
impl<T, Conn, InfoDesk> HamiltonianInfoDesk<T>
    for VertexAssembler<'_, T, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
where
    T: RealField,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    InfoDesk: HamiltonianInfoDesk<T>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    // type BandDim = InfoDesk::BandDim;
    type GeometryDim = InfoDesk::GeometryDim;
    fn get_band_levels(&self, region_index: usize) -> &OPoint<T, Self::BandDim> {
        self.info_desk.get_band_levels(region_index)
    }
    fn get_effective_mass(&self, region_index: usize, band_index: usize) -> &[T; 3] {
        self.info_desk.get_effective_mass(region_index, band_index)
    }
}

/// Re-implement `PotentialInfoDesk` for the element assembler as a pass-through
impl<T, GeometryDim: SmallDim, Conn, InfoDesk> PotentialInfoDesk<T>
    for VertexAssembler<'_, T, InfoDesk, Mesh<T, GeometryDim, Conn>>
where
    T: RealField,
    Conn: Connectivity<T, GeometryDim>,
    InfoDesk: PotentialInfoDesk<T>,
    DefaultAllocator: Allocator<T, GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    type BandDim = InfoDesk::BandDim;
    fn potential(&self, vertex_index: usize) -> T {
        self.info_desk.potential(vertex_index)
    }
}

#[derive(Debug)]
/// An abstraction describing a single element in the mesh
///
/// This allows us to define element specific methods, which contrast the generic methods
/// in the assembly traits above
pub(crate) struct VertexInMesh<'a, InfoDesk, Mesh> {
    /// A reference to an impl of `HamiltonianInfoDesk`
    info_desk: &'a InfoDesk,
    /// A reference to the mesh used in the calculation
    mesh: &'a Mesh,
    /// The index of the vertex in the mesh
    vertex_index: usize,
}

impl<'a, InfoDesk, Mesh> VertexInMesh<'a, InfoDesk, Mesh> {
    /// Construct the ElementInMesh at `element_index` from the mesh and info_desk
    fn from_mesh_vertex_index_and_info_desk(
        mesh: &'a Mesh,
        info_desk: &'a InfoDesk,
        vertex_index: usize,
    ) -> Self {
        Self {
            info_desk,
            mesh,
            vertex_index,
        }
    }
}

impl<'a, T, InfoDesk, Conn> VertexInMesh<'a, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
where
    T: Copy + RealField,
    InfoDesk: HamiltonianInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    /// The index of the `ElementInMesh`
    fn index(&self) -> usize {
        self.vertex_index
    }
    /// A slice, returning the indices of the elements connected to `ElementInMesh`
    fn connections(&self) -> &[usize] {
        self.mesh.connectivity()[self.vertex_index]
    }
    /// The number of elements connected to `ElementInMesh`
    fn connection_count(&self) -> usize {
        self.connections().len()
    }
    /// The central coordinate of `ElementInMesh`
    fn coordinate(&self) -> OPoint<T, InfoDesk::GeometryDim> {
        self.mesh.vertices()[self.index()].0.clone()
    }
    /// The central coordinate of an element at `other_index`
    fn other_coordinate(&self, other_index: usize) -> OPoint<T, InfoDesk::GeometryDim> {
        self.mesh.get_element_midpoint(other_index)
    }
    /// The distance between `ElementInMesh` and the element at `other_index`
    fn get_distance(&self, other_index: usize) -> OVector<T, InfoDesk::GeometryDim> {
        let coordinate = self.coordinate();
        let other_coordinate = self.other_coordinate(other_index);
        coordinate - other_coordinate
    }
    /// Walk over the connectivity in the square mesh by Cartesian dimension
    fn connectivity_by_dimension(&self) -> Vec<&[usize]> {
        self.mesh
            .vertex_connectivity_by_dimension(self.vertex_index)
    }
    /// Walk over the distances to connected elements in the square mesh by Cartesian dimension
    fn deltas_by_dimension(&self) -> Vec<Vec<T>> {
        self.mesh.vertex_deltas_by_dimension(self.vertex_index)
    }

    /// The region in the simulation `Device` to which the element at `other_element_index` is assigned
    fn get_region_of_other(&self, other_vertex_index: usize) -> &Assignment {
        &self.mesh.vertices()[other_vertex_index].1
    }

    /// The band offset for `ElementInMesh` in carrier band `band_index`
    fn conduction_offset(&self, band_index: usize) -> T {
        match self.get_region_of_other(self.vertex_index) {
            Assignment::Core(x) => self.info_desk.get_band_levels(*x)[band_index],
            Assignment::Boundary(x) => {
                let n_points = T::from_usize(x.len()).unwrap();
                x.iter().fold(T::zero(), |acc, &region| {
                    let value = self.info_desk.get_band_levels(region);
                    acc + value[band_index] / n_points
                })
            }
        }
    }

    /// The band offset for `ElementInMesh` in carrier band `band_index`
    fn effective_mass(&self, vertex_index: usize, band_index: usize) -> [T; 3] {
        match self.get_region_of_other(vertex_index) {
            Assignment::Core(x) => *self.info_desk.get_effective_mass(*x, band_index),
            Assignment::Boundary(x) => {
                let n_points = T::from_usize(x.len()).unwrap();
                x.iter().fold([T::zero(); 3], |acc, &region| {
                    let values = self.info_desk.get_effective_mass(region, band_index);
                    [
                        acc[0] + values[0] / n_points,
                        acc[1] + values[1] / n_points,
                        acc[2] + values[2] / n_points,
                    ]
                })
            }
        }
    }
}

impl<'a, T, Conn, InfoDesk> AssembleVertexHamiltonianMatrix<T>
    for VertexAssembler<'a, T, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
where
    T: Copy + RealField,
    InfoDesk: HamiltonianInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    /// Assembles the cell matrix, forming an `num_bands` row array with
    /// `num_connections * num_nearest_neighbours + 1` columns in each row
    fn assemble_vertex_matrix_into(
        &self,
        vertex_index: usize,
        output: ArrayViewMut2<T>,
    ) -> Result<(), BuildError> {
        // Construct the element at `element_index`
        let vertex = VertexInMesh::from_mesh_vertex_index_and_info_desk(
            self.mesh,
            self.info_desk,
            vertex_index,
        );
        // Assemble the differential operator into `output`
        assemble_vertex_differential_operator(output, &vertex, self.info_desk.number_of_bands())
    }
}

/// Fills the differential operator in the Hamiltonian for a single vertex. The vertices in `output` are sorted in the order of
/// their column indices in the final hamiltonian matrix. In one spatial dimension the differential operator is given by
/// ' - \hbar^2 \left[d / dz 1/m_{\parallel} d/dz + 1/m_{\parallel} d^2/dz^2\right]'
fn assemble_vertex_differential_operator<T, InfoDesk, Conn>(
    mut output: ArrayViewMut2<T>,
    vertex: &VertexInMesh<InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>,
    num_bands: usize,
) -> Result<(), BuildError>
where
    T: Copy + RealField,
    InfoDesk: super::HamiltonianInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    let shape = output.shape();

    if shape[1] != vertex.connection_count() + 1 {
        return Err(BuildError::MissizedAllocator(
            "Output matrix should have `n_conns * n_neighbour + 1` columns".into(),
        ));
    }

    if shape[0] != num_bands {
        return Err(BuildError::MissizedAllocator(
            "Output matrix should have `n_bands` rows".into(),
        ));
    }

    // The position and band independent prefactor
    let prefactor = -T::from_f64(HBAR * HBAR / ELECTRON_CHARGE / ELECTRON_MASS / 2.)
        .expect("Prefactor must fit in T");
    for (band_index, mut row) in output.outer_iter_mut().enumerate() {
        // Get the indices of the elements connected to `element` and their displacements from `element`
        let deltas = vertex.deltas_by_dimension();
        let connections = vertex.connectivity_by_dimension();

        // Initialize the diagonal component for `band_index`
        let mut diagonal = T::zero();
        // Holding vector for the values in the band `band_index`
        let mut single_band_values = Vec::with_capacity(vertex.connection_count() + 1);

        // let region_index = element.get_region_of_element();
        // let effective_masses = element
        //     .info_desk
        //     .get_effective_mass(region_index, band_index);

        // Walk over the Cartesian axis in the mesh -> For each axis we add their components to `single_band_values`
        for (spatial_idx, (indices, delta_row)) in
            connections.into_iter().zip(deltas.into_iter()).enumerate()
        {
            assert!(delta_row.len() <= 2, "The mesh should be square");

            let delta_m = delta_row[0];
            // If there is only one connected element we are at the edge of the mesh, so we reuse `delta_m` to prevent panics
            let delta_p = if delta_row.len() == 1 {
                delta_m
            } else {
                delta_row[1]
            };

            // masses is an element containing the masses in the connected elements, followed by that in the current element
            let mut masses = indices
                .iter()
                .map(|&i| vertex.effective_mass(i, band_index)[spatial_idx])
                .collect::<Vec<_>>();
            masses.push(vertex.effective_mass(vertex.index(), band_index)[spatial_idx]);

            // The epsilon on the elements adjoining in our staggered grid
            let inverse_masses = if masses.len() == 3 {
                [
                    (T::one() / masses[0] + T::one() / masses[2]) / (T::one() + T::one()),
                    (T::one() / masses[1] + T::one() / masses[2]) / (T::one() + T::one()),
                ]
            } else {
                [
                    (T::one() / masses[0] + T::one() / masses[1]) / (T::one() + T::one()),
                    (T::one() / masses[0] + T::one() / masses[1]) / (T::one() + T::one()),
                ]
            };

            // Construct the components of the Hamiltonian at the elements considered
            let elements = construct_internal(delta_m, delta_p, &inverse_masses, prefactor);
            single_band_values.push((elements[0], indices[0]));
            // If the length of `delta_row != 2` we are at the edge of the mesh and there is only a single connected element
            if delta_row.len() == 2 {
                single_band_values.push((elements[2], indices[1]));
            }
            // Build the diagonal component
            diagonal += elements[1];
        }
        single_band_values.push((
            diagonal + vertex.conduction_offset(band_index),
            vertex.index(),
        ));
        // Sort `single_band_values` by the index of the element so it can be quickly added to the `CsrMatrix`
        single_band_values.sort_unstable_by(|&a, &b| a.1.cmp(&b.1));
        for (ele, val) in row.iter_mut().zip(single_band_values.into_iter()) {
            *ele = val.0;
        }
    }
    Ok(())
}

/// Helper method to construct the differential operator in the Hamiltonian given the distances to adjacent mesh elements
/// `delta_m`, `delta_p`, the effective mass at the three elements [m_{j-1}, m_{j+1}, m_{j}] and the scalar prefactor multiplying
/// all components of the operator `prefactor`
fn construct_internal<T: Copy + RealField>(
    delta_m: T,
    delta_p: T,
    effective_masses: &[T],
    prefactor: T,
) -> [T; 3] {
    // Get the first derivative differential operator
    let first_derivatives = first_derivative(delta_m, delta_p, prefactor);
    let second_derivatives = second_derivative(
        delta_m,
        delta_p,
        (effective_masses[0] + effective_masses[1]) / (T::one() + T::one()),
        prefactor,
    );
    // Get the first derivative of the mass
    let mass_first_derivative =
        (effective_masses[1] - effective_masses[0]) / (T::one() + T::one()) / (delta_m + delta_p);

    [
        second_derivatives[0] + first_derivatives[0] * mass_first_derivative,
        second_derivatives[1] + first_derivatives[1] * mass_first_derivative,
        second_derivatives[2] + first_derivatives[2] * mass_first_derivative,
    ]
}

fn is_all_same<T: PartialEq>(arr: &[T]) -> bool {
    arr.windows(2).all(|w| w[0] == w[1])
}

/// Computes the second derivatve component of the differential for an inhomogeneous mesh assuming a three point stencil
fn second_derivative<T: Copy + RealField>(
    delta_m: T,
    delta_p: T,
    effective_mass: T,
    prefactor: T,
) -> [T; 3] {
    let prefactor = prefactor * (T::one() + T::one()) / (delta_m * delta_p * (delta_m + delta_p))
        * effective_mass;

    let minus_term = prefactor * delta_m.powi(2) / delta_p;
    let plus_term = prefactor * delta_p.powi(2) / delta_m;
    let central_term = prefactor * (delta_p.powi(2) - delta_m.powi(2)) - minus_term - plus_term;

    [minus_term, central_term, plus_term]
}

/// Computes the first derivatve component of the differential for an inhomogeneous mesh assuming a three point stencil
fn first_derivative<T: Copy + RealField>(delta_m: T, delta_p: T, prefactor: T) -> [T; 3] {
    let prefactor = prefactor / (delta_m * delta_p * (delta_m + delta_p));

    let minus_term = -prefactor * delta_p.powi(2);
    let plus_term = prefactor * delta_m.powi(2);
    let central_term = -plus_term - minus_term;

    [minus_term, central_term, plus_term]
}

impl<'a, T, Conn, InfoDesk> AssembleVertexHamiltonianDiagonal<T>
    for VertexAssembler<'a, T, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
where
    T: Copy + RealField,
    InfoDesk: HamiltonianInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    /// Assembles the cell matrix, forming an `num_bands` row array with
    /// `num_connections * num_nearest_neighbours + 1` columns in each row
    fn assemble_vertex_diagonal_into(
        &self,
        vertex_index: usize,
        output: ArrayViewMut1<T>,
    ) -> Result<(), BuildError> {
        let vertex = VertexInMesh::from_mesh_vertex_index_and_info_desk(
            self.mesh,
            self.info_desk,
            vertex_index,
        );
        assemble_vertex_diagonal(output, &vertex, self.info_desk.number_of_bands())
    }
}

/// Assembles the wavevector component along the element diagonal
fn assemble_vertex_diagonal<T, InfoDesk, Conn>(
    mut output: ArrayViewMut1<T>,
    vertex: &VertexInMesh<InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>,
    num_bands: usize,
) -> Result<(), BuildError>
where
    T: Copy + RealField,
    InfoDesk: super::HamiltonianInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    let shape = output.shape();

    if shape[0] != num_bands {
        return Err(BuildError::MissizedAllocator(
            "Output vector should have `num_bands` rows".into(),
        ));
    }

    let prefactor =
        T::from_f64(HBAR * HBAR / ELECTRON_CHARGE / 2.).expect("Prefactor must fit in T");
    for (band_index, mut row) in output.outer_iter_mut().enumerate() {
        let effective_masses = vertex.effective_mass(vertex.index(), band_index);

        //TODO For brevity this is a one-dimensional implementation. We can improve this at a later date if we want to
        let parallel_mass =
            effective_masses[0] * T::from_f64(ELECTRON_MASS).expect("Electron mass must fit in T");
        row.fill(prefactor / parallel_mass);
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use rand::Rng;

    #[test]
    fn first_derivative_sum_is_zero_when_deltas_are_equal() {
        let mut rng = rand::thread_rng();
        let delta_m: f64 = rng.gen();
        let result: f64 = super::first_derivative(delta_m, delta_m, 1f64).iter().sum();
        assert_relative_eq!(result, 0f64);
    }

    #[test]
    fn first_derivative_sum_is_zero_when_deltas_are_not_equal() {
        let mut rng = rand::thread_rng();
        let delta_m: f64 = rng.gen();
        let delta_p = rng.gen();
        let result: f64 = super::first_derivative(delta_m, delta_p, 1f64).iter().sum();
        assert_relative_eq!(result, 0f64);
    }

    #[test]
    fn full_derivative_is_equal_to_second_derivative_when_masses_are_equal() {
        let mut rng = rand::thread_rng();
        let mass: f64 = rng.gen();
        let delta_m = rng.gen();
        let delta_p = rng.gen();
        let prefactor = rng.gen();

        let masses = [mass, mass, mass];
        let second_derivative = super::second_derivative(delta_m, delta_p, masses[2], prefactor);
        let full_result = super::construct_internal(delta_m, delta_p, &masses, prefactor);

        for (full, second) in full_result.into_iter().zip(second_derivative.into_iter()) {
            assert_relative_eq!(full, second);
        }
    }
}
