use crate::{primitives::ElementMethods, Connectivity, Segment1dConnectivity, SmallDim};
use nalgebra::{allocator::Allocator, DefaultAllocator, OPoint, RealField, U1};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Assignment {
    Boundary(Vec<usize>),
    Core(usize),
}

/// A generic `Mesh` object valid in all dimensions. A `Mesh` for our purposes is made up of vertices,
/// and elements. Each element contains a fixed number of vertices, determined by the connectivity of the mesh
/// and the spatial dimension of the problem.
#[derive(Debug)]
pub struct Mesh<T: RealField, D, Conn>
where
    D: SmallDim,
    Conn: Connectivity<T, D>,
    DefaultAllocator: Allocator<T, D>,
{
    /// The vertices comprising the mesh and their assigned regions
    vertices: Vec<Vertex<T, D>>, //Vec<(OPoint<T, D>, Assignment)>,
    /// Vector of length `number_of_vertices` whose ith element containts indices of the vertices connected to the ith vertex.
    connectivity: Vec<Conn>,
    /// The elements comprising the mesh and their assigned regions (usize as elements are entirely within a region)
    elements: Vec<Element<Conn::Element>>,
    /// Vector of length `number_of_elements` whose ith element contains indices of the elements connected to the ith vertex
    element_connectivity: Vec<Conn>,
}

/// Type alias for a mesh vertex, wrapping the point position and the region `Assignment`
pub type Vertex<T, D> = (OPoint<T, D>, Assignment);
/// Type alias for a mesh element, wrapping the point position and region index as a `usize`
pub type Element<Ele> = (Ele, usize);

impl<T, D, Conn> Mesh<T, D, Conn>
where
    T: RealField,
    D: SmallDim,
    Conn: Connectivity<T, D>,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn num_nodes(&self) -> usize {
        self.vertices.len()
    }
}

/// Type aliases for implemented mesh dimensionalitys and discretisations
pub type Mesh1d<T> = Mesh<T, U1, Segment1dConnectivity>;

impl<T, D, C> Mesh<T, D, C>
where
    T: Copy + RealField,
    C: Connectivity<T, D>,
    D: SmallDim,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn drop_last(&mut self) {
        let _ = self.vertices.pop();
    }

    pub fn vertices_owned(self) -> Vec<(OPoint<T, D>, Assignment)> {
        self.vertices
    }

    pub fn vertex_at_mut(&mut self, idx: usize) -> &mut (OPoint<T, D>, Assignment) {
        &mut self.vertices[idx]
    }

    pub fn vertices_mut(&mut self) -> &mut [(OPoint<T, D>, Assignment)] {
        &mut self.vertices
    }

    pub fn vertices(&self) -> &[(OPoint<T, D>, Assignment)] {
        &self.vertices
    }

    pub fn elements(&self) -> &Vec<Element<C::Element>> {
        &self.elements
    }

    pub fn inspect_connectivity(&self) -> &[C] {
        &self.connectivity
    }

    pub fn connectivity(&self) -> Vec<&[usize]> {
        self.connectivity.iter().map(|x| x.as_inner()).collect()
    }

    pub fn iter_element_connectivity(&self) -> impl std::iter::Iterator<Item = &C> {
        self.element_connectivity.iter()
    }

    pub fn element_connectivity(&self) -> Vec<&[usize]> {
        self.element_connectivity
            .iter()
            .map(|x| x.as_inner())
            .collect()
    }

    pub fn get_element_connectivity(&self, element_index: usize) -> &[usize] {
        self.element_connectivity[element_index].as_inner()
    }

    pub fn element_connectivity_by_dimension(&self, element_index: usize) -> Vec<&[usize]> {
        self.element_connectivity[element_index].over_dimensions()
    }

    pub fn vertex_connectivity_by_dimension(&self, vertex_index: usize) -> Vec<&[usize]> {
        self.connectivity[vertex_index].over_dimensions()
    }

    pub fn deltas_by_dimension(&self, element_index: usize) -> Vec<Vec<T>> {
        let connections = self.get_element_connectivity(element_index);
        let this_element = &self.elements[element_index].0;
        let connected_elements = connections
            .iter()
            .map(|&idx| &self.elements[idx].0)
            .collect::<Vec<_>>();
        self.element_connectivity[element_index]
            .deltas_by_dimension(this_element, &connected_elements)
    }

    pub fn vertex_deltas_by_dimension(&self, vertex_index: usize) -> Vec<Vec<T>> {
        let connections = self.connectivity()[vertex_index];
        let this_vertex = &self.vertices[vertex_index].0;
        let connected_vertices = connections
            .iter()
            .map(|&idx| &self.vertices[idx].0)
            .collect::<Vec<_>>();
        this_vertex
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                connected_vertices
                    .iter()
                    .map(|other| (other[i] - x).abs())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    pub fn get_element_midpoint(&self, element_index: usize) -> OPoint<T, D> {
        let element = &self.elements[element_index].0;
        element.midpoint()
    }

    pub fn get_region_of_element(&self, element_index: usize) -> usize {
        self.elements[element_index].1
    }

    pub fn get_vertex_indices_in_element(&self, element_index: usize) -> &[usize] {
        self.elements[element_index].0.vertex_indices()
    }
}

impl<T: Copy + RealField> Mesh1d<T>
where
    DefaultAllocator: Allocator<T, U1>,
{
    pub fn single_region_from_vertices_and_connectivity(
        vertices: Vec<(OPoint<T, U1>, Assignment)>,
        connectivity: Vec<Segment1dConnectivity>,
        region_index: usize,
    ) -> Self {
        let elements: Vec<(crate::primitives::LineSegment1d<T>, usize)> = vertices
            .windows(2)
            .enumerate()
            .map(|(idx, vertices)| {
                let vertices: Vec<OPoint<T, U1>> =
                    vertices.iter().map(|vertex| vertex.0.clone()).collect();
                (idx, vertices)
            })
            .map(|(idx, x)| Segment1dConnectivity::generate_element(&x, &[idx, idx + 1]))
            .map(|element| (element, region_index))
            .collect();

        let mut element_connectivity = vec![];

        let to_global_vertex_index = |i| i;
        element_connectivity.push(Segment1dConnectivity::Boundary([to_global_vertex_index(1)]));
        for i in 1..elements.len() - 1 {
            element_connectivity.push(Segment1dConnectivity::Core([
                to_global_vertex_index(i - 1),
                to_global_vertex_index(i + 1),
            ]));
        }
        element_connectivity.push(Segment1dConnectivity::Boundary([to_global_vertex_index(
            elements.len() - 2,
        )]));

        Self {
            elements,
            vertices,
            connectivity,
            element_connectivity,
        }
    }

    pub fn from_vertices_and_connectivity(
        vertices: Vec<(OPoint<T, U1>, Assignment)>,
        connectivity: Vec<Segment1dConnectivity>,
    ) -> Self {
        let elements: Vec<(crate::primitives::LineSegment1d<T>, usize)> = vertices
            .windows(2)
            .enumerate()
            .map(|(idx, vertices)| {
                let index = match &vertices[1].1 {
                    Assignment::Core(idx) => idx,
                    Assignment::Boundary(x) => &x[0],
                };
                let index = *index;
                let vertices: Vec<OPoint<T, U1>> =
                    vertices.iter().map(|vertex| vertex.0.clone()).collect();
                (idx, vertices, index)
            })
            .map(|(idx, x, index)| {
                (
                    Segment1dConnectivity::generate_element(&x, &[idx, idx + 1]),
                    index,
                )
            })
            .collect();

        let mut element_connectivity = vec![];

        let to_global_vertex_index = |i| i;
        element_connectivity.push(Segment1dConnectivity::Boundary([to_global_vertex_index(1)]));
        for i in 1..elements.len() - 1 {
            element_connectivity.push(Segment1dConnectivity::Core([
                to_global_vertex_index(i - 1),
                to_global_vertex_index(i + 1),
            ]));
        }
        element_connectivity.push(Segment1dConnectivity::Boundary([to_global_vertex_index(
            elements.len() - 2,
        )]));

        Self {
            elements,
            vertices,
            connectivity,
            element_connectivity,
        }
    }
}
