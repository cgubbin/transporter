use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint, Point1, RealField, U1};

pub trait Distance<T: RealField, Point> {
    fn distance(&self, point: &Point) -> T;
}

pub trait ElementMethods<T: RealField, GeometryDim: DimName>
where
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn vertex_indices(&self) -> &[usize];
    fn midpoint(&self) -> OPoint<T, GeometryDim>;
    fn diameter(&self) -> T;
}

#[derive(Debug)]
pub struct LineSegment1d<T>
where
    T: RealField,
{
    vertices: [Point1<T>; 2],
    vertex_indices: [usize; 2],
}

impl<T> LineSegment1d<T>
where
    T: Copy + RealField,
{
    pub fn from_vertices(vertices: &[Point1<T>; 2], vertex_indices: &[usize; 2]) -> Self {
        Self {
            vertices: vertices.to_owned(),
            vertex_indices: vertex_indices.to_owned(),
        }
    }
    pub fn reference() -> Self {
        Self::from_vertices(&[Point1::new(-T::one()), Point1::new(T::one())], &[0, 1])
    }
    pub fn midpoint(&self) -> Point1<T> {
        Point1::new((self.vertices[0].x + self.vertices[1].x) / (T::one() + T::one()))
    }
    // Temporary while we deal with visibility of elementmethods with multiple grids
    pub fn diameterb(&self) -> T {
        (self.vertices[0].x - self.vertices[1].x).abs()
    }
}

impl<T> Distance<T, Point1<T>> for LineSegment1d<T>
where
    T: Copy + RealField,
{
    fn distance(&self, point: &Point1<T>) -> T {
        let signed_dist = self.midpoint().x - point.x;
        T::max(signed_dist, T::zero())
    }
}

impl<T: Copy + RealField> ElementMethods<T, U1> for LineSegment1d<T> {
    fn midpoint(&self) -> Point1<T> {
        Point1::new((self.vertices[0].x + self.vertices[1].x) / (T::one() + T::one()))
    }
    fn vertex_indices(&self) -> &[usize] {
        &self.vertex_indices
    }
    fn diameter(&self) -> T {
        (self.vertices[0].x - self.vertices[1].x).abs()
    }
}
