//! Element level constructors for the Poisson equation operator
//!
//! This submodule constructs the components of the Hamiltonian differential operator, and diagonal
//! for a single element of the mesh, over `NumBands` (ie: the number of carrier bands in the problem).
use super::PoissonInfoDesk;
use crate::constants::{ELECTRON_CHARGE, ELECTRON_MASS, EPSILON_0};
use crate::hamiltonian::{
    AssembleElementDiagonal, AssembleElementMatrix, ElementConnectivityAssembler,
};
use nalgebra::{
    allocator::Allocator, DMatrix, DMatrixSliceMut, DVector, DVectorSliceMut, DefaultAllocator,
    OPoint, OVector, RealField,
};
use std::marker::PhantomData;
use transporter_mesher::{Assignment, Connectivity, Mesh, SmallDim};

/// Trait giving the information necessary to construct the Hamiltonian differential operator for a given local element
pub trait VertexConnectivityAssembler {
    /// Returns the dimension of the solution as a `usize`: for our problems this is always 1 so we can probably delete the method
    fn solution_dim(&self) -> usize;
    /// The number of elements contained in the entire mesh
    fn num_elements(&self) -> usize;
    /// The number of vertices, or nodes contained in the entire mesh
    fn num_vertices(&self) -> usize;
    /// The number of vertices connected to the vertex at `vertex_index`
    fn vertex_connection_count(&self, vertex_index: usize) -> usize;
    /// Populates the indices of vertices connected to the vertex at `vertex_index` into the slice `output`. The passed slice
    /// must have length `self.vertex_connection_count(vertex_index)`
    fn populate_vertex_connections(&self, output: &mut [usize], element_index: usize);
}

/// Implement `ElementConnectivityAssembler` for the generic `Mesh`
impl<T, GeometryDim, C> VertexConnectivityAssembler for Mesh<T, GeometryDim, C>
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    C: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn solution_dim(&self) -> usize {
        1
    }
    fn num_elements(&self) -> usize {
        self.elements().len()
    }
    fn num_vertices(&self) -> usize {
        self.vertices().len()
    }

    fn vertex_connection_count(&self, vertex_index: usize) -> usize {
        self.connectivity()[vertex_index].len()
    }

    fn populate_vertex_connections(&self, output: &mut [usize], vertex_index: usize) {
        output.copy_from_slice(self.connectivity()[vertex_index])
    }
}

/// Helper trait to construct the diagonal elements of the Hamiltonian (the wavevector component).
pub(crate) trait AssembleVertexDiagonal<T: RealField>: VertexConnectivityAssembler {
    /// Assembles the wavevector component into `output` for the element at `element_index`. Takes an output vector of length
    /// `num_bands` which is enforced by an assertion
    fn assemble_vertex_diagonal(&self, vertex_index: usize) -> color_eyre::Result<T>;
}

/// Helper trait to construct the fixed component of the Hamiltonian, which holds the differential operator and the conduction offsets
pub(crate) trait AssembleVertexMatrix<T: RealField>: VertexConnectivityAssembler {
    /// Takes an output matrix of dimension `num_bands` * `num_connections + 1` which is enforced by an assertion, and fills with the fixed
    ///  component of the Hamiltonian
    fn assemble_vertex_matrix_into(
        &self,
        vertex_index: usize,
        output: DVectorSliceMut<T>,
    ) -> color_eyre::Result<()>;

    fn assemble_vertex_matrix(
        &self,
        vertex_index: usize,
        num_connections: usize,
    ) -> color_eyre::Result<DVector<T>> {
        let mut output = DVector::zeros(num_connections + 1);
        self.assemble_vertex_matrix_into(vertex_index, DVectorSliceMut::from(&mut output))?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
/// An assembler for the Poisson operator at a single mesh element,
pub struct VertexPoissonAssembler<'a, T, InfoDesk, Mesh> {
    /// The `InfoDesk` provides all the external information necessary to construct the Poisson operator
    info_desk: &'a InfoDesk,
    /// The `Mesh` tells us which elements our element is connected to, and how far away they are
    mesh: &'a Mesh,
    /// A marker so we can be generic over the field `T`. This can probably go away with some forethought
    __marker: PhantomData<T>,
}

/// Factory builder for an `ElementPoissonAssembler`
pub struct VertexPoissonAssemblerBuilder<T, RefInfoDesk, RefMesh> {
    /// Reference to an `InfoDesk` which must impl `PoissonInfoDesk<T>`
    info_desk: RefInfoDesk,
    /// Reference to the structure mesh
    mesh: RefMesh,
    /// Marker allowing the structure to be generic over the field `T`
    marker: PhantomData<T>,
}

impl<T> VertexPoissonAssemblerBuilder<T, (), ()> {
    /// Initialise the builder
    pub(crate) fn new() -> Self {
        Self {
            info_desk: (),
            mesh: (),
            marker: PhantomData,
        }
    }
}

impl<T, RefInfoDesk, RefMesh> VertexPoissonAssemblerBuilder<T, RefInfoDesk, RefMesh> {
    /// Attach the info desk
    pub(crate) fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> VertexPoissonAssemblerBuilder<T, &InfoDesk, RefMesh> {
        VertexPoissonAssemblerBuilder {
            info_desk,
            mesh: self.mesh,
            marker: PhantomData,
        }
    }
    /// Attach the mesh
    pub(crate) fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> VertexPoissonAssemblerBuilder<T, RefInfoDesk, &Mesh> {
        VertexPoissonAssemblerBuilder {
            info_desk: self.info_desk,
            mesh,
            marker: PhantomData,
        }
    }
}

impl<'a, T, InfoDesk, Mesh> VertexPoissonAssemblerBuilder<T, &'a InfoDesk, &'a Mesh> {
    /// Build out the ElementPoissonAssembler from the builder
    pub(crate) fn build(self) -> VertexPoissonAssembler<'a, T, InfoDesk, Mesh> {
        VertexPoissonAssembler {
            info_desk: self.info_desk,
            mesh: self.mesh,
            __marker: PhantomData,
        }
    }
}

/// Implement the `HamiltonianInfoDesk` trait for the element assembler to reduce verbiosity
impl<T, Conn, InfoDesk> PoissonInfoDesk<T>
    for VertexPoissonAssembler<'_, T, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
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

/// Inplement `ElementConnectivityAssembler` for the element assembler to reduce verbiosity.
impl<'a, T: Copy + RealField, InfoDesk, Mesh> VertexConnectivityAssembler
    for VertexPoissonAssembler<'a, T, InfoDesk, Mesh>
where
    InfoDesk: PoissonInfoDesk<T>,
    Mesh: VertexConnectivityAssembler,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    fn solution_dim(&self) -> usize {
        1
    }

    fn num_elements(&self) -> usize {
        self.mesh.num_elements()
    }
    fn num_vertices(&self) -> usize {
        self.mesh.num_vertices()
    }

    fn vertex_connection_count(&self, vertex_index: usize) -> usize {
        self.mesh.vertex_connection_count(vertex_index)
    }

    fn populate_vertex_connections(&self, output: &mut [usize], element_index: usize) {
        self.mesh.populate_vertex_connections(output, element_index)
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

impl<'a, T, Conn, InfoDesk> AssembleVertexMatrix<T>
    for VertexPoissonAssembler<'a, T, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
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
        output: DVectorSliceMut<T>,
    ) -> color_eyre::Result<()> {
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
    mut output: DVectorSliceMut<T>,
    vertex: &VertexInMesh<InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>,
) -> color_eyre::Result<()>
where
    T: Copy + RealField,
    InfoDesk: PoissonInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    let shape = output.shape();

    assert_eq!(
        shape.0,
        vertex.connection_count() + 1,
        "Output matrix should have `n_conns * n_neighbour + 1` columns"
    );

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

impl<'a, T, Conn, InfoDesk> AssembleVertexDiagonal<T>
    for VertexPoissonAssembler<'a, T, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
where
    T: Copy + RealField,
    InfoDesk: PoissonInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    /// Assembles the cell matrix, forming an `num_bands` row array with
    /// `num_connections * num_nearest_neighbours + 1` columns in each row
    fn assemble_vertex_diagonal(&self, vertex_index: usize) -> color_eyre::Result<T> {
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
) -> color_eyre::Result<T>
where
    T: Copy + RealField,
    InfoDesk: PoissonInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    let prefactor = T::from_f64(ELECTRON_CHARGE).expect("Prefactor must fit in T");
    Ok(prefactor * (vertex.acceptor_density() - vertex.donor_density()))
}

#[derive(Debug)]
/// Struct to
pub struct AggregateVertexAssembler<'a, VertexAssembler> {
    assemblers: &'a [VertexAssembler],
    solution_dim: usize,
    num_elements: usize,
    num_vertices: usize,
    vertex_offsets: Vec<usize>,
}

impl<'a, VertexAssembler> VertexConnectivityAssembler
    for AggregateVertexAssembler<'a, VertexAssembler>
where
    VertexAssembler: VertexConnectivityAssembler,
{
    fn solution_dim(&self) -> usize {
        self.solution_dim
    }

    fn num_elements(&self) -> usize {
        self.num_elements
    }

    fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    fn vertex_connection_count(&self, aggregate_vertex_index: usize) -> usize {
        let (assembler, vertex_offset) =
            self.find_assembler_and_offset_for_vertex_index(aggregate_vertex_index);
        assembler.vertex_connection_count(aggregate_vertex_index - vertex_offset)
    }

    fn populate_vertex_connections(&self, output: &mut [usize], aggregate_vertex_index: usize) {
        let (assembler, vertex_offset) =
            self.find_assembler_and_offset_for_vertex_index(aggregate_vertex_index);
        assembler.populate_vertex_connections(output, aggregate_vertex_index - vertex_offset)
    }
}

impl<'a, VertexAssembler> AggregateVertexAssembler<'a, VertexAssembler>
where
    VertexAssembler: VertexConnectivityAssembler,
{
    pub fn from_assemblers(assemblers: &'a [VertexAssembler]) -> Self {
        assert!(
            !assemblers.is_empty(),
            "The aggregate Hamiltonian must have at least one (1) assembler."
        );
        let solution_dim = assemblers[0].solution_dim();
        let num_vertices = assemblers[0].num_vertices();
        assert!(
            assemblers
                .iter()
                .all(|assembler| assembler.solution_dim() == solution_dim),
            "All assemblers must have the same solution dimension"
        );
        assert!(
            assemblers
                .iter()
                .all(|assembler| assembler.num_vertices() == num_vertices),
            "All assemblers must have the same node index space (same num_nodes)"
        );
        let mut num_total_vertices = 0;
        let mut vertex_offsets = Vec::with_capacity(assemblers.len());
        for assembler in assemblers {
            vertex_offsets.push(num_total_vertices);
            num_total_vertices += assembler.num_vertices();
        }
        Self {
            assemblers,
            solution_dim,
            num_elements: assemblers[0].num_elements(),
            num_vertices,
            vertex_offsets,
        }
    }

    fn find_assembler_and_offset_for_vertex_index(
        &self,
        vertex_index: usize,
    ) -> (&VertexAssembler, usize) {
        assert!(vertex_index <= self.num_vertices);
        let assembler_idx = match self.vertex_offsets.binary_search(&vertex_index) {
            Ok(idx) => idx,
            Err(idx) => idx - 1,
        };
        (
            &self.assemblers[assembler_idx],
            self.vertex_offsets[assembler_idx],
        )
    }
}

impl<'a, T, VertexAssembler> AssembleVertexMatrix<T>
    for AggregateVertexAssembler<'a, VertexAssembler>
where
    T: RealField,
    VertexAssembler: AssembleVertexMatrix<T>,
{
    fn assemble_vertex_matrix_into(
        &self,
        aggregate_vertex_index: usize,
        output: DVectorSliceMut<T>,
    ) -> color_eyre::Result<()> {
        let (assembler, vertex_offset) =
            self.find_assembler_and_offset_for_vertex_index(aggregate_vertex_index);
        assembler.assemble_vertex_matrix_into(aggregate_vertex_index - vertex_offset, output)
    }
}

impl<'a, T, VertexAssembler> AssembleVertexDiagonal<T>
    for AggregateVertexAssembler<'a, VertexAssembler>
where
    T: RealField,
    VertexAssembler: AssembleVertexDiagonal<T>,
{
    fn assemble_vertex_diagonal(&self, aggregate_vertex_index: usize) -> color_eyre::Result<T> {
        let (assembler, vertex_offset) =
            self.find_assembler_and_offset_for_vertex_index(aggregate_vertex_index);
        assembler.assemble_vertex_diagonal(aggregate_vertex_index - vertex_offset)
    }
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
