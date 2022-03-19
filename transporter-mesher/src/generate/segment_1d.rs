use crate::connectivity::Segment1dConnectivity;
use crate::mesh::Mesh1d;
use nalgebra::{OPoint, Point1, RealField, Vector1, U1};

pub fn create_unit_line_segment_mesh_1d<T>(cells_per_dim: usize) -> Mesh1d<T>
where
    T: Copy + RealField,
{
    create_line_segment_mesh_1d(T::one(), 1, cells_per_dim, &Vector1::new(T::zero()))
}

pub fn create_line_segment_mesh_1d<T>(
    unit_length: T,
    units_x: usize,
    cells_per_unit: usize,
    left: &Vector1<T>,
) -> Mesh1d<T>
where
    T: Copy + RealField,
{
    if cells_per_unit == 0 || units_x == 0 {
        Mesh1d::from_vertices_and_connectivity(Vec::new(), Vec::new())
    } else {
        let cell_size =
            T::from_f64(unit_length.to_subset().unwrap() / cells_per_unit as f64).unwrap();
        let num_cells_x = units_x * cells_per_unit;
        let num_vertices_x = num_cells_x + 1;
        let mut vertices = Vec::with_capacity(num_vertices_x);
        let mut cells = Vec::with_capacity(num_vertices_x);

        let to_global_vertex_index = |i| i;
        for i in 0..num_vertices_x {
            let i_as_t = T::from_usize(i).expect("Must be able to fit usize in T");
            let v = left + Vector1::new(i_as_t) * cell_size;
            vertices.push(Point1::from(v));
        }

        cells.push(Segment1dConnectivity::Boundary([to_global_vertex_index(1)]));
        for i in 1..num_vertices_x - 1 {
            cells.push(Segment1dConnectivity::Core([
                to_global_vertex_index(i - 1),
                to_global_vertex_index(i + 1),
            ]));
        }
        cells.push(Segment1dConnectivity::Boundary([to_global_vertex_index(
            num_vertices_x - 2,
        )]));

        Mesh1d::from_vertices_and_connectivity(vertices, cells)
    }
}

pub(crate) fn create_line_segment_mesh_1d_from_regions<T>(
    unit_length: T,
    units_x: &[usize],
    cells_per_unit: &[usize],
    left: &Vector1<T>,
) -> Mesh1d<T>
where
    T: Copy + RealField,
{
    assert_eq!(units_x.len(), cells_per_unit.len());

    let mut left = *left;

    let mut meshes = vec![];
    for (&units_x, &cells_per_unit) in units_x.iter().zip(cells_per_unit.iter()) {
        meshes.push(create_line_segment_mesh_1d(
            unit_length,
            units_x,
            cells_per_unit,
            &left,
        ));
        let units_x_as_t = T::from_usize(units_x).expect("Must be able to fit usize in T");
        left += Vector1::new(units_x_as_t) * unit_length;
    }

    Mesh1d::dedup(meshes)
}

impl<T> Mesh1d<T>
where
    T: RealField,
{
    fn dedup(meshes: Vec<Mesh1d<T>>) -> Mesh1d<T> {
        let mut vertices: Vec<OPoint<T, U1>> = meshes
            .into_iter()
            .flat_map(|x| x.vertices_owned())
            .collect();
        vertices.dedup();

        let num_cells_x = vertices.len() - 1;
        let to_global_vertex_index = |i| i;

        let mut cells = Vec::with_capacity(vertices.len());
        cells.push(Segment1dConnectivity::Boundary([to_global_vertex_index(1)]));
        for i in 1..vertices.len() - 1 {
            cells.push(Segment1dConnectivity::Core([
                to_global_vertex_index(i - 1),
                to_global_vertex_index(i + 1),
            ]));
        }
        cells.push(Segment1dConnectivity::Boundary([to_global_vertex_index(
            vertices.len() - 2,
        )]));

        Mesh1d::from_vertices_and_connectivity(vertices, cells)
    }
}

#[cfg(test)]
mod test {
    use super::create_line_segment_mesh_1d_from_regions;
    use super::create_unit_line_segment_mesh_1d;

    #[test]
    fn mesh_from_regions_eliminates_repeated_vertices() {
        let unit_length = 1f64;
        let left = nalgebra::Vector1::new(0f64);
        let units_x = [2, 5, 10, 20];
        let cells_per_unit = [10, 10, 10, 10];
        let mesh =
            create_line_segment_mesh_1d_from_regions(unit_length, &units_x, &cells_per_unit, &left);

        let delta: Vec<f64> = mesh
            .vertices()
            .windows(2)
            .map(|vertices| (vertices[0].x - vertices[1].x).abs())
            .collect();
        assert!(delta.iter().all(|&x| x > 0.001));
    }

    #[test]
    fn vertices_and_connectivity_file_correctly() {
        let mesh: super::Mesh1d<f64> = create_unit_line_segment_mesh_1d(20);
        dbg!(mesh.vertices());
        dbg!(mesh.connectivity());
    }
}
