//! Element level constructors for the Hamiltonian matrix
//!
//! This submodule constructs the components of the Hamiltonian differential operator, and diagonal
//! for a single element of the mesh, over `NumBands` (ie: the number of carrier bands in the problem).
use super::HamiltonianInfoDesk;
use crate::constants::{ELECTRON_CHARGE, ELECTRON_MASS, HBAR};
use nalgebra::{
    allocator::Allocator, DMatrix, DMatrixSliceMut, DVectorSliceMut, DefaultAllocator, OPoint,
    OVector, RealField,
};
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

/// Trait giving the information necessary to construct the Hamiltonian differential operator for a given local element
pub trait ElementConnectivityAssembler {
    /// Returns the dimension of the solution as a `usize`: for our problems this is always 1 so we can probably delete the method
    fn solution_dim(&self) -> usize;
    /// The number of elements contained in the entire mesh
    fn num_elements(&self) -> usize;
    /// The number of vertices, or nodes contained in the entire mesh
    fn num_nodes(&self) -> usize;
    /// The number of elements connected to the element at `element_index`
    fn element_connection_count(&self, elelment_index: usize) -> usize;
    /// Populates the indices of elements connected to the element at `element_index` into the slice `output`. The passed slice
    /// must have length `self.element_connection_count(element_index)`
    fn populate_element_connections(&self, output: &mut [usize], element_index: usize);
}

/// Implement `ElementConnectivityAssembler` for the generic `Mesh`
impl<T, GeometryDim, C> ElementConnectivityAssembler for Mesh<T, GeometryDim, C>
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
    fn num_nodes(&self) -> usize {
        self.vertices().len()
    }

    fn element_connection_count(&self, element_index: usize) -> usize {
        self.get_element_connectivity(element_index).len()
    }

    fn populate_element_connections(&self, output: &mut [usize], element_index: usize) {
        output.copy_from_slice(self.get_element_connectivity(element_index))
    }
}

/// Helper trait to construct the diagonal elements of the Hamiltonian (the wavevector component).
pub(crate) trait AssembleElementDiagonal<T: RealField>:
    ElementConnectivityAssembler
{
    /// Assembles the wavevector component into `output` for the element at `element_index`. Takes an output vector of length
    /// `num_bands` which is enforced by an assertion
    fn assemble_element_diagonal_into(
        &self,
        element_index: usize,
        output: DVectorSliceMut<T>,
    ) -> color_eyre::Result<()>;
}

/// Helper trait to construct the fixed component of the Hamiltonian, which holds the differential operator and the conduction offsets
pub(crate) trait AssembleElementMatrix<T: RealField>: ElementConnectivityAssembler {
    /// Takes an output matrix of dimension `num_bands` * `num_connections + 1` which is enforced by an assertion, and fills with the fixed
    ///  component of the Hamiltonian
    fn assemble_element_matrix_into(
        &self,
        element_index: usize,
        output: DMatrixSliceMut<T>,
    ) -> color_eyre::Result<()>;

    fn assemble_element_matrix(
        &self,
        element_index: usize,
        num_bands: usize,
        num_connections: usize,
    ) -> color_eyre::Result<DMatrix<T>> {
        let mut output = DMatrix::zeros(num_bands, num_connections + 1);
        self.assemble_element_matrix_into(element_index, DMatrixSliceMut::from(&mut output))?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
/// An assembler for the Hamiltonian of a single mesh element,
pub struct ElementHamiltonianAssembler<'a, T, InfoDesk, Mesh> {
    /// The `InfoDesk` provides all the external information necessary to construct the Hamiltonian
    info_desk: &'a InfoDesk,
    /// The `Mesh` tells us which elements our element is connected to, and how far away they are
    mesh: &'a Mesh,
    /// A marker so we can be generic over the field `T`. This can probably go away with some forethought
    __marker: PhantomData<T>,
}

/// Factory builder for an `ElementHamiltonianAssembler`
pub struct ElementHamiltonianAssemblerBuilder<T, RefInfoDesk, RefMesh> {
    /// Reference to an `InfoDesk` which must impl `HamiltonianInfoDesk<T>`
    info_desk: RefInfoDesk,
    /// Reference to the structure mesh
    mesh: RefMesh,
    /// Marker allowing the structure to be generic over the field `T`
    marker: PhantomData<T>,
}

impl<T> ElementHamiltonianAssemblerBuilder<T, (), ()> {
    /// Initialise the builder
    pub(crate) fn new() -> Self {
        Self {
            info_desk: (),
            mesh: (),
            marker: PhantomData,
        }
    }
}

impl<T, RefInfoDesk, RefMesh> ElementHamiltonianAssemblerBuilder<T, RefInfoDesk, RefMesh> {
    /// Attach the info desk
    pub(crate) fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> ElementHamiltonianAssemblerBuilder<T, &InfoDesk, RefMesh> {
        ElementHamiltonianAssemblerBuilder {
            info_desk,
            mesh: self.mesh,
            marker: PhantomData,
        }
    }
    /// Attach the mesh
    pub(crate) fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> ElementHamiltonianAssemblerBuilder<T, RefInfoDesk, &Mesh> {
        ElementHamiltonianAssemblerBuilder {
            info_desk: self.info_desk,
            mesh,
            marker: PhantomData,
        }
    }
}

impl<'a, T, InfoDesk, Mesh> ElementHamiltonianAssemblerBuilder<T, &'a InfoDesk, &'a Mesh> {
    /// Build out the ElementHamiltonianAssembler from the builder
    pub(crate) fn build(self) -> ElementHamiltonianAssembler<'a, T, InfoDesk, Mesh> {
        ElementHamiltonianAssembler {
            info_desk: self.info_desk,
            mesh: self.mesh,
            __marker: PhantomData,
        }
    }
}

/// Implement the `HamiltonianInfoDesk` trait for the element assembler to reduce verbiosity
impl<T, Conn, InfoDesk> HamiltonianInfoDesk<T>
    for ElementHamiltonianAssembler<'_, T, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
where
    T: RealField,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    InfoDesk: HamiltonianInfoDesk<T>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    type BandDim = InfoDesk::BandDim;
    type GeometryDim = InfoDesk::GeometryDim;
    fn get_band_levels(&self, region_index: usize) -> &OPoint<T, Self::BandDim> {
        self.info_desk.get_band_levels(region_index)
    }
    fn get_effective_mass(&self, region_index: usize, band_index: usize) -> &[T; 3] {
        self.info_desk.get_effective_mass(region_index, band_index)
    }
    fn potential(&self, element_index: usize) -> T {
        self.info_desk.potential(element_index)
    }
}

/// Inplement `ElementConnectivityAssembler` for the element assembler to reduce verbiosity.
impl<'a, T: RealField, InfoDesk, Mesh> ElementConnectivityAssembler
    for ElementHamiltonianAssembler<'a, T, InfoDesk, Mesh>
where
    InfoDesk: HamiltonianInfoDesk<T>,
    Mesh: ElementConnectivityAssembler,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    fn solution_dim(&self) -> usize {
        1
    }

    fn num_elements(&self) -> usize {
        self.mesh.num_elements()
    }
    fn num_nodes(&self) -> usize {
        self.mesh.num_nodes()
    }

    fn element_connection_count(&self, element_index: usize) -> usize {
        self.mesh.element_connection_count(element_index)
    }

    fn populate_element_connections(&self, output: &mut [usize], element_index: usize) {
        self.mesh
            .populate_element_connections(output, element_index)
    }
}

#[derive(Debug)]
/// An abstraction describing a single element in the mesh
///
/// This allows us to define element specific methods, which contrast the generic methods
/// in the assembly traits above
/// TODO is this true? What does this really gain us?
pub(crate) struct ElementInMesh<'a, InfoDesk, Mesh> {
    /// A reference to an impl of `HamiltonianInfoDesk`
    info_desk: &'a InfoDesk,
    /// A reference to the mesh used in the calculation
    mesh: &'a Mesh,
    /// The index of the element in the mesh
    element_index: usize,
}

impl<'a, InfoDesk, Mesh> ElementInMesh<'a, InfoDesk, Mesh> {
    /// Construct the ElementInMesh at `element_index` from the mesh and info_desk
    fn from_mesh_element_index_and_info_desk(
        mesh: &'a Mesh,
        info_desk: &'a InfoDesk,
        element_index: usize,
    ) -> Self {
        Self {
            info_desk,
            mesh,
            element_index,
        }
    }
}

impl<'a, T, InfoDesk, Conn> ElementInMesh<'a, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
where
    T: Copy + RealField,
    InfoDesk: HamiltonianInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    /// The index of the `ElementInMesh`
    fn index(&self) -> usize {
        self.element_index
    }
    /// A slice, returning the indices of the elements connected to `ElementInMesh`
    fn connections(&self) -> &[usize] {
        self.mesh.get_element_connectivity(self.element_index)
    }
    /// The number of elements connected to `ElementInMesh`
    fn connection_count(&self) -> usize {
        self.connections().len()
    }
    /// The central coordinate of `ElementInMesh`
    fn coordinate(&self) -> OPoint<T, InfoDesk::GeometryDim> {
        self.mesh.get_element_midpoint(self.element_index)
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
            .element_connectivity_by_dimension(self.element_index)
    }
    /// Walk over the distances to connected elements in the square mesh by Cartesian dimension
    fn deltas_by_dimension(&self) -> Vec<Vec<T>> {
        self.mesh.deltas_by_dimension(self.element_index)
    }
    /// The region in the simulation `Device` to which `ElementInMesh` is assigned
    fn get_region_of_element(&self) -> usize {
        self.mesh.get_region_of_element(self.element_index)
    }
    /// The region in the simulation `Device` to which the element at `other_element_index` is assigned
    fn get_region_of_other(&self, other_element_index: usize) -> usize {
        self.mesh.get_region_of_element(other_element_index)
    }
    /// The band offset for `ElementInMesh` in carrier band `band_index`
    fn conduction_offset(&self, band_index: usize) -> T {
        self.info_desk.get_band_levels(self.get_region_of_element())[band_index]
    }
}

impl<'a, T, Conn, InfoDesk> AssembleElementMatrix<T>
    for ElementHamiltonianAssembler<'a, T, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
where
    T: Copy + RealField,
    InfoDesk: HamiltonianInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    /// Assembles the cell matrix, forming an `num_bands` row array with
    /// `num_connections * num_nearest_neighbours + 1` columns in each row
    fn assemble_element_matrix_into(
        &self,
        element_index: usize,
        output: DMatrixSliceMut<T>,
    ) -> color_eyre::Result<()> {
        // Construct the element at `element_index`
        let element = ElementInMesh::from_mesh_element_index_and_info_desk(
            self.mesh,
            self.info_desk,
            element_index,
        );
        // Assemble the differential operator into `output`
        assemble_element_differential_operator(output, &element, self.info_desk.number_of_bands())
    }
}

/// Fills the differential operator in the Hamiltonian for a single element. The elements in `output` are sorted in the order of
/// their column indices in the final hamiltonian matrix. In one spatial dimension the differential operator is given by
/// ' - \hbar^2 \left[d / dz 1/m_{\parallel} d/dz + 1/m_{\parallel} d^2/dz^2\right]'
fn assemble_element_differential_operator<T, InfoDesk, Conn>(
    mut output: DMatrixSliceMut<T>,
    element: &ElementInMesh<InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>,
    num_bands: usize,
) -> color_eyre::Result<()>
where
    T: Copy + RealField,
    InfoDesk: super::HamiltonianInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    let shape = output.shape();

    assert_eq!(
        shape.1,
        element.connection_count() + 1,
        "Output matrix should have `n_conns * n_neighbour + 1` columns"
    );
    assert_eq!(
        shape.0, num_bands,
        "Output matrix should have `n_bands` rows"
    );

    // The position and band independent prefactor
    let prefactor =
        -T::from_f64(HBAR * HBAR / ELECTRON_CHARGE / 2.).expect("Prefactor must fit in T");
    for (band_index, mut row) in output.row_iter_mut().enumerate() {
        // Get the indices of the elements connected to `element` and their displacements from `element`
        let deltas = element.deltas_by_dimension();
        let connections = element.connectivity_by_dimension();

        // Initialize the diagonal component for `band_index`
        let mut diagonal = T::zero();
        // Holding vector for the values in the band `band_index`
        let mut single_band_values = Vec::with_capacity(element.connection_count() + 1);

        let region_index = element.get_region_of_element();
        let effective_masses = element
            .info_desk
            .get_effective_mass(region_index, band_index);

        // Walk over the Cartesian axis in the mesh -> For each axis we add their components to `single_band_values`
        for (spatial_idx, ((indices, delta_row), &mass)) in connections
            .into_iter()
            .zip(deltas.into_iter())
            .zip(effective_masses.iter())
            .enumerate()
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
                .map(|&i| element.get_region_of_other(i))
                .map(|region_i| {
                    element.info_desk.get_effective_mass(region_i, band_index)[spatial_idx]
                })
                .collect::<Vec<_>>();
            masses.push(mass);

            // Construct the components of the Hamiltonian at the elements considered
            let elements = construct_internal(delta_m, delta_p, &masses, prefactor);
            single_band_values.push((elements[0], indices[0]));
            // If the length of `delta_row != 2` we are at the edge of the mesh and there is only a single connected element
            if delta_row.len() == 2 {
                single_band_values.push((elements[2], indices[1]));
            }
            // Build the diagonal component
            diagonal += elements[1];
        }
        single_band_values.push((
            diagonal + element.conduction_offset(band_index),
            element.index(),
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
    // Get the second derivative differential operator
    let l = effective_masses.len();
    let second_derivatives =
        second_derivative(delta_m, delta_p, effective_masses[l - 1], prefactor);
    // Get the first derivative of the mass
    let mass_first_derivatives = mass_first_derivative(delta_m, delta_p, effective_masses);

    [
        second_derivatives[0]
            + first_derivatives[0] * mass_first_derivatives
                / T::from_f64(ELECTRON_MASS).expect("Electron mass must fit in T"),
        second_derivatives[1]
            + first_derivatives[1] * mass_first_derivatives
                / T::from_f64(ELECTRON_MASS).expect("Electron mass must fit in T"),
        if effective_masses.len() == 3 {
            second_derivatives[2]
                + first_derivatives[2] * mass_first_derivatives
                    / T::from_f64(ELECTRON_MASS).expect("Electron mass must fit in T")
        } else {
            T::zero()
        },
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
    let prefactor = prefactor * (T::one() + T::one())
        / (delta_m * delta_p * (delta_m + delta_p))
        / effective_mass
        / T::from_f64(ELECTRON_MASS).expect("Electron mass must fit in T");

    let minus_term = prefactor * delta_m.powi(2) / delta_p;
    let plus_term = prefactor * delta_p.powi(2) / delta_m;
    let central_term = prefactor * (delta_p.powi(2) - delta_m.powi(2)) - minus_term - plus_term;

    [minus_term, central_term, plus_term]
}

/// Computes the first derivatve masses for an inhomogeneous mesh assuming a three point stencil
fn mass_first_derivative<T: Copy + RealField>(delta_m: T, delta_p: T, masses: &[T]) -> T {
    if is_all_same(masses) {
        return T::zero();
    }
    let first_derivative_operator_components = first_derivative(delta_m, delta_p, T::one());

    let mut result = first_derivative_operator_components[0] / masses[0];

    // If we are at the mesh edge assume the mass and mesh spacing is constant out into the next element
    if masses.len() == 2 {
        result += first_derivative_operator_components[2] / masses[0]
            + first_derivative_operator_components[1] / masses[1];
    } else {
        result += first_derivative_operator_components[2] / masses[1]
            + first_derivative_operator_components[1] / masses[2];
    }

    result
}

/// Computes the first derivatve component of the differential for an inhomogeneous mesh assuming a three point stencil
fn first_derivative<T: Copy + RealField>(delta_m: T, delta_p: T, prefactor: T) -> [T; 3] {
    let prefactor = prefactor / (delta_m * delta_p * (delta_m + delta_p));

    let minus_term = -prefactor * delta_p.powi(2);
    let plus_term = prefactor * delta_m.powi(2);
    let central_term = -plus_term - minus_term;

    [minus_term, central_term, plus_term]
}

impl<'a, T, Conn, InfoDesk> AssembleElementDiagonal<T>
    for ElementHamiltonianAssembler<'a, T, InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>
where
    T: Copy + RealField,
    InfoDesk: HamiltonianInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    /// Assembles the cell matrix, forming an `num_bands` row array with
    /// `num_connections * num_nearest_neighbours + 1` columns in each row
    fn assemble_element_diagonal_into(
        &self,
        element_index: usize,
        output: DVectorSliceMut<T>,
    ) -> color_eyre::Result<()> {
        let element = ElementInMesh::from_mesh_element_index_and_info_desk(
            self.mesh,
            self.info_desk,
            element_index,
        );
        assemble_element_diagonal(output, &element, self.info_desk.number_of_bands())
    }
}

/// Assembles the wavevector component along the element diagonal
fn assemble_element_diagonal<T, InfoDesk, Conn>(
    mut output: DVectorSliceMut<T>,
    element: &ElementInMesh<InfoDesk, Mesh<T, InfoDesk::GeometryDim, Conn>>,
    num_bands: usize,
) -> color_eyre::Result<()>
where
    T: Copy + RealField,
    InfoDesk: super::HamiltonianInfoDesk<T>,
    Conn: Connectivity<T, InfoDesk::GeometryDim>,
    DefaultAllocator: Allocator<T, InfoDesk::GeometryDim> + Allocator<T, InfoDesk::BandDim>,
{
    let shape = output.shape();

    assert_eq!(
        shape.1, 1,
        "Output matrix should have `n_conns * n_neighbour + 1` columns"
    );
    assert_eq!(
        shape.0, num_bands,
        "Output matrix should have `num_bands` rows"
    );

    let prefactor =
        T::from_f64(HBAR * HBAR / ELECTRON_CHARGE / 2.).expect("Prefactor must fit in T");
    for (band_index, mut row) in output.row_iter_mut().enumerate() {
        let region_index = element.get_region_of_element();
        let effective_masses = element
            .info_desk
            .get_effective_mass(region_index, band_index);

        //TODO For brevity this is a one-dimensional implementation. We can improve this at a later date if we want to
        let parallel_mass =
            effective_masses[0] * T::from_f64(ELECTRON_MASS).expect("Electron mass must fit in T");
        let ele = row.get_mut(0).unwrap();
        *ele = prefactor / parallel_mass;
    }
    Ok(())
}

#[derive(Debug)]
/// Struct to
pub struct AggregateElementAssembler<'a, ElementAssembler> {
    assemblers: &'a [ElementAssembler],
    solution_dim: usize,
    num_elements: usize,
    num_nodes: usize,
    element_offsets: Vec<usize>,
}

impl<'a, ElementAssembler> ElementConnectivityAssembler
    for AggregateElementAssembler<'a, ElementAssembler>
where
    ElementAssembler: ElementConnectivityAssembler,
{
    fn solution_dim(&self) -> usize {
        self.solution_dim
    }

    fn num_elements(&self) -> usize {
        self.num_elements
    }

    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    fn element_connection_count(&self, aggregate_element_index: usize) -> usize {
        let (assembler, cell_offset) =
            self.find_assembler_and_offset_for_element_index(aggregate_element_index);
        assembler.element_connection_count(aggregate_element_index - cell_offset)
    }

    fn populate_element_connections(&self, output: &mut [usize], aggregate_element_index: usize) {
        let (assembler, element_offset) =
            self.find_assembler_and_offset_for_element_index(aggregate_element_index);
        assembler.populate_element_connections(output, aggregate_element_index - element_offset)
    }
}

impl<'a, CellAssembler> AggregateElementAssembler<'a, CellAssembler>
where
    CellAssembler: ElementConnectivityAssembler,
{
    pub fn from_assemblers(assemblers: &'a [CellAssembler]) -> Self {
        assert!(
            !assemblers.is_empty(),
            "The aggregate Hamiltonian must have at least one (1) assembler."
        );
        let solution_dim = assemblers[0].solution_dim();
        let num_nodes = assemblers[0].num_nodes();
        assert!(
            assemblers
                .iter()
                .all(|assembler| assembler.solution_dim() == solution_dim),
            "All assemblers must have the same solution dimension"
        );
        assert!(
            assemblers
                .iter()
                .all(|assembler| assembler.num_nodes() == num_nodes),
            "All assemblers must have the same node index space (same num_nodes)"
        );
        let mut num_total_cells = 0;
        let mut element_offsets = Vec::with_capacity(assemblers.len());
        for assembler in assemblers {
            element_offsets.push(num_total_cells);
            num_total_cells += assembler.num_elements();
        }
        Self {
            assemblers,
            solution_dim,
            num_elements: assemblers[0].num_elements(),
            num_nodes,
            element_offsets,
        }
    }

    fn find_assembler_and_offset_for_element_index(
        &self,
        element_index: usize,
    ) -> (&CellAssembler, usize) {
        assert!(element_index <= self.num_elements);
        let assembler_idx = match self.element_offsets.binary_search(&element_index) {
            Ok(idx) => idx,
            Err(idx) => idx - 1,
        };
        (
            &self.assemblers[assembler_idx],
            self.element_offsets[assembler_idx],
        )
    }
}

impl<'a, T, ElementAssembler> AssembleElementMatrix<T>
    for AggregateElementAssembler<'a, ElementAssembler>
where
    T: RealField,
    ElementAssembler: AssembleElementMatrix<T>,
{
    fn assemble_element_matrix_into(
        &self,
        aggregate_element_index: usize,
        output: DMatrixSliceMut<T>,
    ) -> color_eyre::Result<()> {
        let (assembler, element_offset) =
            self.find_assembler_and_offset_for_element_index(aggregate_element_index);
        assembler.assemble_element_matrix_into(aggregate_element_index - element_offset, output)
    }
}

impl<'a, T, ElementAssembler> AssembleElementDiagonal<T>
    for AggregateElementAssembler<'a, ElementAssembler>
where
    T: RealField,
    ElementAssembler: AssembleElementDiagonal<T>,
{
    fn assemble_element_diagonal_into(
        &self,
        aggregate_element_index: usize,
        output: DVectorSliceMut<T>,
    ) -> color_eyre::Result<()> {
        let (assembler, element_offset) =
            self.find_assembler_and_offset_for_element_index(aggregate_element_index);
        assembler.assemble_element_diagonal_into(aggregate_element_index - element_offset, output)
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use rand::Rng;

    #[test]
    fn mass_first_derivative_is_zero_when_masses_are_equal() {
        let mut rng = rand::thread_rng();
        let mass: f64 = rng.gen();
        let delta_m = rng.gen();
        let delta_p = rng.gen();

        let masses = [mass, mass, mass];
        let result = super::mass_first_derivative(delta_m, delta_p, &masses);
        assert_relative_eq!(result, 0f64);
    }

    #[test]
    fn mass_first_derivative_is_zero_when_masses_are_equal_at_mesh_edge() {
        let mut rng = rand::thread_rng();
        let mass: f64 = rng.gen();
        let delta_m = rng.gen();
        let delta_p = rng.gen();

        let masses = [mass, mass];
        let result = super::mass_first_derivative(delta_m, delta_p, &masses);
        assert_relative_eq!(result, 0f64);
    }

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
