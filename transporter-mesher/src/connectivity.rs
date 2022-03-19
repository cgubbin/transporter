#[derive(Debug)]
pub enum Segment1dConnectivity {
    Core([usize; 2]),
    Boundary([usize; 1]),
}

pub trait Connectivity {
    fn as_inner(&self) -> &[usize];
}

impl Connectivity for Segment1dConnectivity {
    fn as_inner(&self) -> &[usize] {
        match self {
            Segment1dConnectivity::Core(x) => x,
            Segment1dConnectivity::Boundary(x) => x,
        }
    }
}
