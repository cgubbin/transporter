use itertools::izip;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint, RealField, Scalar};
use nalgebra_sparse::CsrMatrix;
use transporter_mesher::{Assignment, FiniteDifferenceMesh, SmallDim};

const EPSILON_0: f64 = 8.85418782e-12;
const ELECTRON_CHARGE: f64 = 1.60217662e-19;

pub trait PoissonOperator<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
{
    type SolutionDim: SmallDim;
}

pub struct OperatorAssemblerBuilder<T, RefMesh> {
    //, RefSource> {
    mesh: RefMesh,
    //source: RefSource,
    at_central_points: bool,
    marker: std::marker::PhantomData<T>,
}

impl OperatorAssemblerBuilder<(), ()> {
    pub fn new() -> Self {
        Self {
            mesh: (),
            // source: (),
            at_central_points: false,
            marker: std::marker::PhantomData,
        }
    }
}

impl<RefMesh> OperatorAssemblerBuilder<(), RefMesh> {
    //, RefSource> {
    pub fn with_mesh<Mesh>(self, mesh: &Mesh) -> OperatorAssemblerBuilder<(), &Mesh> {
        //, RefSource> {
        OperatorAssemblerBuilder {
            mesh,
            //  source: self.source,
            at_central_points: self.at_central_points,
            marker: std::marker::PhantomData,
        }
    }

    //  pub fn with_source<Source>(
    //      self,
    //      source: &Source,
    //  ) -> OperatorAssemblerBuilder<(), RefMesh, &Source> {
    //      OperatorAssemblerBuilder {
    //          mesh: self.mesh,
    //          source,
    //          at_central_points: self.at_central_points,
    //          marker: std::marker::PhantomData,
    //      }
    //  }

    //  pub fn evaluate_at_central_points(self) -> OperatorAssemblerBuilder<(), RefMesh, RefSource> {
    //      OperatorAssemblerBuilder {
    //          mesh: self.mesh,
    //          source: self.source,
    //          at_central_points: true,
    //          marker: std::marker::PhantomData,
    //      }
    //  }
}

pub(crate) struct OperatorAssembler<'a, T, Mesh> {
    //, Source> {
    mesh: &'a Mesh,
    //source: &'a Source,
    at_central_points: bool,
    marker: std::marker::PhantomData<T>,
}

impl<'a, Mesh> OperatorAssemblerBuilder<(), &'a Mesh> {
    //, &'a Source> {
    pub(crate) fn build<T>(self) -> OperatorAssembler<'a, T, Mesh> {
        //, Source> {
        OperatorAssembler {
            mesh: self.mesh,
            //    source: self.source,
            at_central_points: self.at_central_points,
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T, Mesh> OperatorAssembler<'a, T, Mesh>
//, Operator>
where
    T: Copy + RealField,
    Mesh: FiniteDifferenceMesh<T>,
    //  Operator: PoissonOperator<T, Mesh::GeometryDim>,
    DefaultAllocator: Allocator<T, Mesh::GeometryDim>,
{
    fn solution_dim(&self) -> usize {
        Mesh::GeometryDim::dim()
    }

    fn geometry_dim(&self) -> usize {
        Mesh::GeometryDim::dim()
    }

    fn num_nodes(&self) -> usize {
        self.mesh.number_of_nodes()
    }

    fn get_vertices(&self) -> &[(OPoint<T, Mesh::GeometryDim>, Assignment)] {
        self.mesh.get_vertices()
    }

    fn get_connectivity(&self) -> Vec<&[usize]> {
        self.mesh.get_connectivity()
    }

    #[allow(clippy::if_same_then_else)]
    pub fn assemble_matrix(&self, n: usize) -> color_eyre::Result<nalgebra_sparse::CscMatrix<T>> {
        let scaling = T::from_f64(EPSILON_0).unwrap();
        let ndof = n * self.geometry_dim() * self.solution_dim();
        let mut row_offsets = Vec::with_capacity(ndof);
        let mut col_indices = Vec::with_capacity(ndof * 3 - 2);
        let mut values = Vec::with_capacity(ndof * 3 - 2);

        let mesh_data = izip![self.get_vertices(), self.get_connectivity()];
        let mut row_tick = 0;
        row_offsets.push(row_tick);

        for (idx, (vertex, connections)) in mesh_data.enumerate() {
            let mut new_cols = vec![];
            let mut new_vals = vec![];
            let dx = connections
                .iter()
                .map(|&x| (vertex.0.coords[0] - self.get_vertices()[x].0.coords[0]).abs())
                .collect::<Vec<T>>(); // The first element is delta_- and the second is delta_+ for a 1D mesh
                                      // Ignore variation in epsilon
                                      // At the moment the matrix is minus what we want so that it can undergo cholesky decomp. This is fine, just remember later
            if idx == 0 {
                let delta = dx[0]; // Both are equal at the mesh edge
                new_cols.push(idx);
                new_vals.push(scaling * (T::one() + T::one()) / delta.powi(2));
                // Change for the bc, this is a Neumann one
                new_cols.push(idx + 1);
                new_vals.push(-scaling * (T::one() + T::one()) / delta.powi(2));
                row_tick += 2;
                // row_tick += 1;
            } else if idx == n - 1 {
                let delta = dx[0]; // Both are equal at the mesh edge
                                   // new_cols.push(idx - 1);
                                   // new_vals.push(T::one() / delta.powi(2));
                                   // Change for the bc
                new_cols.push(idx - 1);
                new_vals.push(-scaling * (T::one() + T::one()) / delta.powi(2));
                new_cols.push(idx);
                new_vals.push(scaling * (T::one() + T::one()) / delta.powi(2));
                row_tick += 2;
                //row_tick += 1;
            } else {
                let delta_minus = dx[0];
                let delta_plus = dx[1];
                let first_derivative_h =
                    -delta_plus.powi(2) / (delta_plus + delta_minus) / delta_plus / delta_minus;
                let second_derivative_h =
                    (T::one() + T::one()) / (delta_plus + delta_minus) / delta_plus / delta_minus
                        * ((delta_plus.powi(2) - delta_minus.powi(2)) * first_derivative_h
                            + delta_plus.powi(2) / delta_minus);
                new_cols.push(idx - 1);
                new_vals.push(-scaling * second_derivative_h);
                let first_derivative_i = (delta_plus.powi(2) - delta_minus.powi(2))
                    / (delta_plus + delta_minus)
                    / delta_plus
                    / delta_minus;
                let second_derivative_i =
                    (T::one() + T::one()) / (delta_plus + delta_minus) / delta_plus / delta_minus
                        * ((delta_plus.powi(2) - delta_minus.powi(2)) * first_derivative_i
                            - delta_plus.powi(2) / delta_minus
                            - delta_minus.powi(2) / delta_plus);
                new_cols.push(idx);
                new_vals.push(-scaling * second_derivative_i);
                let first_derivative_j =
                    delta_minus.powi(2) / (delta_plus + delta_minus) / delta_plus / delta_minus;
                let second_derivative_j =
                    (T::one() + T::one()) / (delta_plus + delta_minus) / delta_plus / delta_minus
                        * ((delta_plus.powi(2) - delta_minus.powi(2)) * first_derivative_j
                            + delta_minus.powi(2) / delta_plus);
                new_cols.push(idx + 1);
                new_vals.push(-scaling * second_derivative_j);

                row_tick += 3;
            }
            col_indices.append(&mut new_cols);
            values.append(&mut new_vals);

            row_offsets.push(row_tick);
        }
        let csr = CsrMatrix::try_from_csr_data(ndof, ndof, row_offsets, col_indices, values)
            .expect("CSR data must conform to format specifications");
        let csc = csr.transpose().transpose_as_csc();
        Ok(csc)
    }
}
