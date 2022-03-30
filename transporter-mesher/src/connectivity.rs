use crate::{primitives::ElementMethods, SmallDim};
use nalgebra::{allocator::Allocator, DefaultAllocator, OPoint, Point1, RealField, Vector2, U1};

#[derive(Debug)]
pub enum Segment1dConnectivity {
    Core([usize; 2]),
    Boundary([usize; 1]),
}

pub trait Connectivity<T: RealField, GeometryDim: SmallDim>: std::fmt::Debug
where
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    type Element: std::fmt::Debug + ElementMethods<T, GeometryDim>;
    fn as_inner(&self) -> &[usize];
    fn generate_element(vertices: &[OPoint<T, GeometryDim>], indices: &[usize]) -> Self::Element;
    fn over_dimensions(&self) -> Vec<&[usize]>;
    fn deltas_by_dimension(
        &self,
        this_element: &Self::Element,
        connected_elements: &[&Self::Element],
    ) -> Vec<Vec<T>>;
}

impl<T: Copy + RealField> Connectivity<T, U1> for Segment1dConnectivity
where
    DefaultAllocator: Allocator<T, U1>,
{
    type Element = crate::primitives::LineSegment1d<T>;
    fn as_inner(&self) -> &[usize] {
        match self {
            Segment1dConnectivity::Core(x) => x,
            Segment1dConnectivity::Boundary(x) => x,
        }
    }

    fn generate_element(vertices: &[Point1<T>], indices: &[usize]) -> Self::Element {
        assert_eq!(vertices.len(), 2);
        let vertices = [vertices[0].clone(), vertices[1].clone()];
        let indices = [indices[0], indices[1]];
        crate::primitives::LineSegment1d::from_vertices(&vertices, &indices)
    }

    fn over_dimensions(&self) -> Vec<&[usize]> {
        let ind: &[usize] = match self {
            Segment1dConnectivity::Core(x) => x,
            Segment1dConnectivity::Boundary(x) => x,
        };
        vec![ind]
    }

    fn deltas_by_dimension(
        &self,
        this_element: &Self::Element,
        connected_elements: &[&Self::Element],
    ) -> Vec<Vec<T>> {
        assert_eq!(connected_elements.len(), self.as_inner().len());
        let midpoint = this_element.midpoint();
        let other_midpoints = connected_elements
            .iter()
            .map(|connected_element| connected_element.midpoint())
            .map(|other_midpoint| (&midpoint - other_midpoint).norm())
            .collect::<Vec<_>>();
        vec![other_midpoints]
    }
}
