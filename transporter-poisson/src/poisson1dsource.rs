use crate::allocators::DimAllocator;
use crate::operator::OperatorAssemblerBuilder;
use nalgebra::DVector;
use nalgebra::{DefaultAllocator, RealField};

pub struct PoissonSourceBuilder<RefMesh, RefSource> {
    mesh: RefMesh,
    source: RefSource,
}

impl PoissonSourceBuilder<(), ()> {
    fn new() -> Self {
        Self {
            mesh: (),
            source: (),
        }
    }
}

impl<RefMesh, RefSource> PoissonSourceBuilder<RefMesh, RefSource> {
    pub fn with_mesh<Mesh>(self, mesh: &Mesh) -> PoissonSourceBuilder<&Mesh, RefSource> {
        PoissonSourceBuilder {
            mesh,
            source: self.source,
        }
    }

    pub fn with_source<Source>(self, source: &Source) -> PoissonSourceBuilder<RefMesh, &Source> {
        PoissonSourceBuilder {
            mesh: self.mesh,
            source,
        }
    }
}

pub struct PoissonSourceb<'a, Mesh, Operator, Source, Jacobian> {
    mesh: &'a Mesh,
    source: &'a Source,
    pub operator: Operator,
    inverse_jacobian: Jacobian,
}

impl<'a, T> PoissonSourceBuilder<&'a transporter_mesher::Mesh1d<T>, &'a DVector<T>>
where
    T: Copy + RealField,
    DefaultAllocator: DimAllocator<T, nalgebra::U1>,
{
    pub fn build(
        self,
    ) -> PoissonSourceb<
        'a,
        transporter_mesher::Mesh1d<T>,
        nalgebra_sparse::CscMatrix<T>,
        DVector<T>,
        nalgebra_sparse::CscMatrix<T>,
    > {
        let builder = OperatorAssemblerBuilder::new().with_mesh(self.mesh).build();
        let operator = builder.assemble_matrix(self.source.len()).unwrap();

        PoissonSourceb {
            mesh: self.mesh,
            source: self.source,
            operator: operator.clone(),
            inverse_jacobian: operator,
        }
    }
}

use nalgebra_sparse::factorization::CscCholesky;
use nalgebra_sparse::CscMatrix;
use transporter_mesher::Mesh1d;

impl<'a, T> PoissonSourceb<'a, Mesh1d<T>, CscMatrix<T>, DVector<T>, CscMatrix<T>>
where
    T: Copy + RealField,
{
    pub fn operator(&'a self) -> &'a CscMatrix<T> {
        &self.operator
    }

    pub fn source(&'a self) -> &'a DVector<T> {
        self.source
    }

    pub fn jacobian(&'a self) -> &'a CscMatrix<T> {
        self.operator()
    }

    pub fn factorised_jacobian(&'a self) -> CscCholesky<T> {
        nalgebra_sparse::factorization::CscCholesky::factor(self.jacobian()).unwrap()
    }

    /// Update the Jacobian based on the current potential
    pub fn update_jacobian(&'a self, _solution: &DVector<T>) -> color_eyre::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::solve::NewtonSolver;

    use transporter_mesher::{create_unit_line_segment_mesh_1d, Mesh1d};

    #[test]
    fn test_poisson_1d() {
        let mesh: Mesh1d<f64> = create_unit_line_segment_mesh_1d(100);
        let source = |x: f64| 2. * x.powi(2);

        let source_vector = mesh
            .vertices()
            .iter()
            .map(|x| source(x.0.x))
            .collect::<Vec<f64>>();
        let n = source_vector.len();
        let mut source_ovector: nalgebra::DVector<f64> = nalgebra::DVector::<f64>::zeros(n);
        for (s, sb) in source_vector.iter().zip(source_ovector.iter_mut()) {
            *sb = *s;
        }

        let problem = super::PoissonSourceBuilder::new()
            .with_mesh(&mesh)
            .with_source(&source_ovector)
            .build();

        let mut test_solution: nalgebra::DVector<f64> = nalgebra::DVector::<f64>::zeros(n);
        for (s, sb) in source_vector.iter().zip(test_solution.iter_mut()) {
            *sb = *s;
        }

        test_solution = problem.solve_into(test_solution).unwrap();

        let dense_operator =
            nalgebra_sparse::convert::serial::convert_csc_dense(problem.operator());
        let factorized_operator = dense_operator.lu();
        let solb = factorized_operator.solve(problem.source()).unwrap();

        for (x, y) in test_solution.iter().zip(solb.iter()) {
            approx::assert_relative_eq!(x, y, epsilon = std::f64::EPSILON * 100.);
        }
    }
}
