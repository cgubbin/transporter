// Copyright 2022 Chris Gubbin
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Local
//!
//! Element level constructors for the Poisson equation operator
//!
//! This submodule constructs the components of the Hamiltonian differential operator, and diagonal
//! for a single element of the mesh, over `NumBands` (ie: the number of carrier bands in the problem).
use super::{BuildError, PoissonInfoDesk};
use crate::{
    constants::{ELECTRON_CHARGE, EPSILON_0},

utilities::assemblers::{VertexAssembler, VertexConnectivityAssembler}
};
use nalgebra::{allocator::Allocator, DefaultAllocator, OPoint, OVector, RealField};
use ndarray::{Array1, ArrayViewMut1};
use transporter_mesher::{Assignment, Connectivity, Mesh};


/// Helper trait to construct the diagonal elements of a differential operator
pub(crate) trait AssembleVertexPoissonDiagonal<T: RealField>:
    VertexConnectivityAssembler
{
    /// Assembles the wavevector component into `output` for the element at `element_index`. Takes an output vector of length
    /// `num_bands` which is enforced by an assertion
    fn assemble_vertex_diagonal(&self, vertex_index: usize) -> Result<T, BuildError>;
}

/// Helper trait to construct the fixed component of the operator (ie: the differential bit)
pub(crate) trait AssembleVertexPoissonMatrix<T: RealField>:
    VertexConnectivityAssembler
{
    /// Takes an output matrix of dimension `num_bands` * `num_connections + 1` which is enforced by an assertion, and fills with the fixed
    ///  component of the Hamiltonian
    fn assemble_vertex_matrix_into(
        &self,
        vertex_index: usize,
        output: ArrayViewMut1<T>,
    ) -> Result<(), BuildError>;

    fn assemble_vertex_matrix(
        &self,
        vertex_index: usize,
        num_connections: usize,
    ) -> Result<Array1<T>, BuildError> {
        let mut output = Array1::zeros(num_connections + 1);
        self.assemble_vertex_matrix_into(vertex_index, ArrayViewMut1::from(&mut output))?;
        Ok(output)
    }
}

/// Implement the `HamiltonianInfoDesk` trait for the element assembler to reduce verbiosity
impl<T, Conn, InfoDesk> PoissonInfoDesk<T>
    for VertexAssembler<'_, T, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
where
    T: Copy + RealField,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    InfoDesk: PoissonInfoDesk<T>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    type BandDim = InfoDesk::BandDim;
    type GeometryDim = InfoDesk::GeometryDim;
    fn get_static_dielectric_constant(&self, region_index: usize) -> &[T; 3] {
        self.info_desk.get_static_dielectric_constant(region_index)
    }
    fn get_acceptor_density(&self, region_index: usize) -> T {
        self.info_desk.get_acceptor_density(region_index)
    }
    fn get_donor_density(&self, region_index: usize) -> T {
        self.info_desk.get_donor_density(region_index)
    }
}

#[derive(Debug)]
/// An abstraction describing a single element in the mesh
///
/// This allows us to define element specific methods, which contrast the generic methods
/// in the assembly traits above
/// TODO is this true? What does this really gain us?
pub(crate) struct VertexInMesh<'a, InfoDesk, Mesh> {
    /// A reference to an impl of `HamiltonianInfoDesk`
    info_desk: &'a InfoDesk,
    /// A reference to the mesh used in the calculation
    mesh: &'a Mesh,
    /// The index of the element in the mesh
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
    InfoDesk: PoissonInfoDesk<T>,
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
    fn coordinate(&self) -> &OPoint<T, InfoDesk::GeometryDim> {
        &self.mesh.vertices()[self.vertex_index].0
    }
    /// The central coordinate of an element at `other_index`
    fn other_coordinate(&self, other_index: usize) -> &OPoint<T, InfoDesk::GeometryDim> {
        &self.mesh.vertices()[other_index].0
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
    /// The region in the simulation `Device` to which `ElementInMesh` is assigned
    fn get_region_of_vertex(&self) -> &Assignment {
        &self.mesh.vertices()[self.vertex_index].1
    }
    /// The region in the simulation `Device` to which the element at `other_element_index` is assigned
    fn get_region_of_other(&self, other_vertex_index: usize) -> &Assignment {
        &self.mesh.vertices()[other_vertex_index].1
    }
    /// The band offset for `ElementInMesh` in carrier band `band_index`
    fn static_dielectric_constant(&self, vertex_index: usize) -> [T; 3] {
        match self.get_region_of_other(vertex_index) {
            Assignment::Core(x) => *self.info_desk.get_static_dielectric_constant(*x),
            Assignment::Boundary(x) => {
                let n_points = T::from_usize(x.len()).unwrap();
                x.iter().fold([T::zero(); 3], |acc, &region| {
                    let values = self.info_desk.get_static_dielectric_constant(region);
                    [
                        acc[0] + values[0] / n_points,
                        acc[1] + values[1] / n_points,
                        acc[2] + values[2] / n_points,
                    ]
                })
            }
        }
    }

    /// The band offset for `ElementInMesh` in carrier band `band_index`
    fn acceptor_density(&self) -> T {
        match self.get_region_of_other(self.index()) {
            Assignment::Core(x) => self.info_desk.get_acceptor_density(*x),
            Assignment::Boundary(x) => {
                let n_points = T::from_usize(x.len()).unwrap();
                x.iter().fold(T::zero(), |acc, &region| {
                    acc + self.info_desk.get_acceptor_density(region)
                }) / n_points
            }
        }
    }

    /// The band offset for `ElementInMesh` in carrier band `band_index`
    fn donor_density(&self) -> T {
        match self.get_region_of_other(self.index()) {
            Assignment::Core(x) => self.info_desk.get_donor_density(*x),
            Assignment::Boundary(x) => {
                let n_points = T::from_usize(x.len()).unwrap();
                x.iter().fold(T::zero(), |acc, &region| {
                    acc + self.info_desk.get_donor_density(region)
                }) / n_points
            }
        }
    }
}

impl<'a, T, Conn, InfoDesk> AssembleVertexPoissonMatrix<T>
    for VertexAssembler<'a, T, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
where
    T: Copy + RealField,
    InfoDesk: PoissonInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    /// Assembles the cell matrix, forming an `num_bands` row array with
    /// `num_connections * num_nearest_neighbours + 1` columns in each row
    fn assemble_vertex_matrix_into(
        &self,
        vertex_index: usize,
        output: ArrayViewMut1<T>,
    ) -> Result<(), BuildError> {
        // Construct the element at `element_index`
        let vertex = VertexInMesh::from_mesh_vertex_index_and_info_desk(
            self.mesh,
            self.info_desk,
            vertex_index,
        );
        // Assemble the differential operator into `output`
        assemble_vertex_differential_operator(output, &vertex)
    }
}

/// Fills the differential operator in the Poisson equation for a single element. The elements in `output` are sorted in the order of
/// their column indices in the final hamiltonian matrix. In one spatial dimension the differential operator is given by
/// ' -  \left[d / dz \epsilon_{\perp} d/dz + \epsilon_{\perp} d^2/dz^2\right]'
fn assemble_vertex_differential_operator<T, InfoDesk, Conn>(
    mut output: ArrayViewMut1<T>,
    vertex: &VertexInMesh<InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>,
) -> Result<(), BuildError>
where
    T: Copy + RealField,
    InfoDesk: PoissonInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    let shape = output.shape();

    if shape[0] != vertex.connection_count() + 1 {
        return Err(BuildError::MissizedAllocator(
            "Output matrix should have `n_conns * n_neighbour + 1` columns".into(),
        ));
    }

    // The position and band independent prefactor
    let prefactor = -T::from_f64(EPSILON_0).expect("Prefactor must fit in T");
    // Get the indices of the elements connected to `element` and their displacements from `element`
    let deltas = vertex.deltas_by_dimension();
    let connections = vertex.connectivity_by_dimension();

    // Initialize the diagonal component for `band_index`
    let mut diagonal = T::zero();

    let mut values = Vec::with_capacity(vertex.connection_count() + 1);

    // Walk over the Cartesian axis in the mesh -> For each axis we add their components to `single_band_values`
    for (spatial_idx, (indices, delta_row)) in
        connections.into_iter().zip(deltas.into_iter()).enumerate()
    {
        assert!(delta_row.len() <= 2, "The mesh should be square");
        if delta_row.len() > 2 {
            return Err(BuildError::Mesh("The provided mesh is not square".into()));
        }

        let delta_m = delta_row[0];
        // If there is only one connected element we are at the edge of the mesh, so we reuse `delta_m` to prevent panics
        let delta_p = if delta_row.len() == 1 {
            delta_m
        } else {
            delta_row[1]
        };

        // masses is an element containing the static dielectric constant for the connected elements, followed by that at the current vertex
        let mut static_epsilon = indices
            .iter()
            .map(|&vertex_index| vertex.static_dielectric_constant(vertex_index)[spatial_idx])
            .collect::<Vec<_>>();
        static_epsilon.push(vertex.static_dielectric_constant(vertex.index())[spatial_idx]);

        // The epsilon on the elements adjoining in our staggered grid
        let static_epsilon = if static_epsilon.len() == 3 {
            [
                (static_epsilon[0] + static_epsilon[2]) / (T::one() + T::one()),
                (static_epsilon[1] + static_epsilon[2]) / (T::one() + T::one()),
            ]
        } else {
            [
                (static_epsilon[0] + static_epsilon[1]) / (T::one() + T::one()),
                (static_epsilon[0] + static_epsilon[1]) / (T::one() + T::one()),
            ]
        };

        // Construct the components of the Hamiltonian at the elements considered
        let elements = construct_internal(delta_m, delta_p, &static_epsilon, prefactor);

        values.push((elements[0], indices[0]));
        // If the length of `delta_row != 2` we are at the edge of the mesh and there is only a single connected element
        if delta_row.len() == 2 {
            values.push((elements[2], indices[1]));
        }
        // Build the diagonal component
        if delta_row.len() == 2 {
            diagonal += elements[1];
        } else {
            diagonal += elements[1] / (T::one() + T::one()); // Neumann condition at the edges
        }
    }
    values.push((diagonal, vertex.index()));
    // Sort `single_band_values` by the index of the element so it can be quickly added to the `CsrMatrix`
    values.sort_unstable_by(|&a, &b| a.1.cmp(&b.1));
    for (ele, val) in output.iter_mut().zip(values.into_iter()) {
        *ele = val.0;
    }
    Ok(())
}

/// Helper method to construct the differential operator in the Hamiltonian given the distances to adjacent mesh elements
/// `delta_m`, `delta_p`, the effective mass at the three elements [m_{j-1}, m_{j+1}, m_{j}] and the scalar prefactor multiplying
/// all components of the operator `prefactor`
fn construct_internal<T: Copy + RealField>(
    delta_m: T,
    delta_p: T,
    epsilons: &[T],
    prefactor: T,
) -> [T; 3] {
    // Get the first derivative differential operator
    let first_derivatives = first_derivative(delta_m, delta_p, prefactor);
    // Get the second derivative differential operator
    let second_derivatives = second_derivative(
        delta_m,
        delta_p,
        (epsilons[0] + epsilons[1]) / (T::one() + T::one()),
        prefactor,
    );
    let epsilon_first_derivatives =
        (epsilons[1] - epsilons[0]) / (T::one() + T::one()) / (delta_m + delta_p);

    [
        second_derivatives[0] + first_derivatives[0] * epsilon_first_derivatives,
        second_derivatives[1] + first_derivatives[1] * epsilon_first_derivatives,
        second_derivatives[2] + first_derivatives[2] * epsilon_first_derivatives,
    ]
}

/// Computes the second derivatve component of the differential for an inhomogeneous mesh assuming a three point stencil
fn second_derivative<T: Copy + RealField>(
    delta_m: T,
    delta_p: T,
    epsilon_static: T,
    prefactor: T,
) -> [T; 3] {
    let prefactor = prefactor * (T::one() + T::one()) / (delta_m * delta_p * (delta_m + delta_p))
        * epsilon_static;

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

impl<'a, T, Conn, InfoDesk> AssembleVertexPoissonDiagonal<T>
    for VertexAssembler<'a, T, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
where
    T: Copy + RealField,
    InfoDesk: PoissonInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    /// Assembles the cell matrix, forming an `num_bands` row array with
    /// `num_connections * num_nearest_neighbours + 1` columns in each row
    fn assemble_vertex_diagonal(&self, vertex_index: usize) -> Result<T, BuildError> {
        let vertex = VertexInMesh::from_mesh_vertex_index_and_info_desk(
            self.mesh,
            self.info_desk,
            vertex_index,
        );
        assemble_vertex_diagonal(&vertex)
    }
}

/// Assembles the wavevector component along the element diagonal
fn assemble_vertex_diagonal<T, InfoDesk, Conn>(
    vertex: &VertexInMesh<InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>,
) -> Result<T, BuildError>
where
    T: Copy + RealField,
    InfoDesk: PoissonInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    let prefactor = T::from_f64(ELECTRON_CHARGE).expect("Prefactor must fit in T");
    Ok(prefactor * (vertex.acceptor_density() - vertex.donor_density()))
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use rand::Rng;

    #[test]
    fn poisson_first_derivative_sum_is_zero_when_deltas_are_equal() {
        let mut rng = rand::thread_rng();
        let delta_m: f64 = rng.gen();
        let result: f64 = super::first_derivative(delta_m, delta_m, 1f64).iter().sum();
        assert_relative_eq!(result, 0f64);
    }

    #[test]
    fn poisson_first_derivative_sum_is_zero_when_deltas_are_not_equal() {
        let mut rng = rand::thread_rng();
        let delta_m: f64 = rng.gen();
        let delta_p = rng.gen();
        let result: f64 = super::first_derivative(delta_m, delta_p, 1f64).iter().sum();
        assert_relative_eq!(result, 0f64);
    }

    #[test]
    fn poisson_full_derivative_is_equal_to_second_derivative_when_epsilons_are_equal() {
        let mut rng = rand::thread_rng();
        let epsilon: f64 = rng.gen();
        let delta_m = rng.gen();
        let delta_p = rng.gen();
        let prefactor = rng.gen();

        let epsilons = [epsilon, epsilon, epsilon];
        let second_derivative = super::second_derivative(delta_m, delta_p, epsilons[2], prefactor);
        let full_result = super::construct_internal(delta_m, delta_p, &epsilons, prefactor);

        for (full, second) in full_result.into_iter().zip(second_derivative.into_iter()) {
            assert_relative_eq!(full, second);
        }
    }
}
