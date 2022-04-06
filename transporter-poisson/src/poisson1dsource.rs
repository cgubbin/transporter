use crate::operator::OperatorAssemblerBuilder;
use nalgebra::{allocator::Allocator, DVector, DefaultAllocator, RealField};
use nalgebra_sparse::{factorization::CscCholesky, CscMatrix};
use transporter_mesher::{Connectivity, Mesh, Mesh1d, SmallDim};

pub struct PoissonSourceBuilder<RefInfoDesk, RefMesh, RefSource> {
    info_desk: RefInfoDesk,
    mesh: RefMesh,
    source: RefSource,
}

impl PoissonSourceBuilder<(), (), ()> {
    pub fn new() -> Self {
        Self {
            info_desk: (),
            mesh: (),
            source: (),
        }
    }
}

impl<RefInfoDesk, RefMesh, RefSource> PoissonSourceBuilder<RefInfoDesk, RefMesh, RefSource> {
    pub fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> PoissonSourceBuilder<&InfoDesk, RefMesh, RefSource> {
        PoissonSourceBuilder {
            info_desk,
            mesh: self.mesh,
            source: self.source,
        }
    }

    pub fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> PoissonSourceBuilder<RefInfoDesk, &Mesh, RefSource> {
        PoissonSourceBuilder {
            info_desk: self.info_desk,
            mesh,
            source: self.source,
        }
    }

    pub fn with_source<Source>(
        self,
        source: &Source,
    ) -> PoissonSourceBuilder<RefInfoDesk, RefMesh, &Source> {
        PoissonSourceBuilder {
            info_desk: self.info_desk,
            mesh: self.mesh,
            source,
        }
    }
}

pub struct PoissonSourceb<'a, InfoDesk, Mesh, Operator, Source> {
    info_desk: &'a InfoDesk,
    mesh: &'a Mesh,
    source: &'a Source,
    pub operator: Operator,
}

impl<'a, T, GeometryDim: SmallDim, Conn, InfoDesk>
    PoissonSourceBuilder<&'a InfoDesk, &'a Mesh<T, GeometryDim, Conn>, &'a DVector<T>>
where
    T: Copy + RealField,
    Conn: Connectivity<T, GeometryDim>,
    InfoDesk: PoissonMethods<T>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub fn build(
        self,
    ) -> PoissonSourceb<
        'a,
        InfoDesk,
        Mesh<T, GeometryDim, Conn>,
        nalgebra_sparse::CscMatrix<T>,
        DVector<T>,
    > {
        let builder = OperatorAssemblerBuilder::new().with_mesh(self.mesh).build();
        let operator = builder.assemble_matrix(self.source.len()).unwrap();

        PoissonSourceb {
            info_desk: self.info_desk,
            mesh: self.mesh,
            source: self.source,
            operator,
        }
    }
}

impl<'a, T, InfoDesk> PoissonSourceb<'a, InfoDesk, Mesh1d<T>, CscMatrix<T>, DVector<T>>
where
    T: Copy + RealField,
    InfoDesk: PoissonMethods<T>,
{
    pub fn operator(&'a self) -> &'a CscMatrix<T> {
        &self.operator
    }

    pub fn source(&'a self) -> &'a DVector<T> {
        self.source
    }

    /// Construct the Jacobian from the stored operator, which describes the hopping, and the diagonal, which
    /// describes the charge and potential
    pub fn jacobian(&'a self, jacobian_diagonal: DVector<T>) -> color_eyre::Result<CscMatrix<T>> {
        let col_indices = (0..self.source.len()).collect::<Vec<_>>();
        let row_offsets = (0..=self.source.len()).collect::<Vec<_>>();
        let pattern = nalgebra_sparse::pattern::SparsityPattern::try_from_offsets_and_indices(
            self.source.len(),
            self.source.len(),
            row_offsets,
            col_indices,
        )
        .map_err(|e| {
            color_eyre::eyre::eyre!("Failed to generate sparsity pattern for jacobian {:?}", e)
        })?;
        let vals = jacobian_diagonal
            .into_iter()
            .map(|x| *x)
            .collect::<Vec<_>>();
        Ok(self.operator()
            + CscMatrix::try_from_pattern_and_values(pattern, vals).map_err(|e| {
                color_eyre::eyre::eyre!("Failed to generate CscMatrix for jacobian {:?}", e)
            })?)
    }

    pub fn factorised_jacobian(
        &'a self,
        jacobian_diagonal: DVector<T>,
    ) -> color_eyre::Result<CscCholesky<T>> {
        Ok(
            nalgebra_sparse::factorization::CscCholesky::factor(&self.jacobian(jacobian_diagonal)?)
                .unwrap(),
        )
    }

    /// Update the Jacobian based on the current potential
    pub fn update_jacobian_diagonal(
        &'a self,
        solution: &'a DVector<T>,
        jacobian_diagonal: &'a mut DVector<T>,
    ) -> color_eyre::Result<()> {
        self.info_desk
            .update_jacobian_diagonal(solution, jacobian_diagonal)
    }

    /// Update the Jacobian based on the current potential
    pub fn update_charge_density(
        &'a self,
        solution: &'a DVector<T>,
        charge_density: &'a mut DVector<T>,
    ) -> color_eyre::Result<()> {
        self.info_desk
            .update_charge_density(solution, charge_density)
    }
}

pub trait PoissonMethods<T: Copy + RealField> {
    fn update_jacobian_diagonal(
        &self,
        solution: &DVector<T>,
        output: &mut DVector<T>,
    ) -> color_eyre::Result<()>;
    fn update_charge_density(
        &self,
        solution: &DVector<T>,
        output: &mut DVector<T>,
    ) -> color_eyre::Result<()>;
}

//#[cfg(test)]
//mod test {
//    use crate::solve::NewtonSolver;
//
//    use transporter_mesher::{create_unit_line_segment_mesh_1d, Mesh1d};
//
//    #[test]
//    fn test_poisson_1d() {
//        let mesh: Mesh1d<f64> = create_unit_line_segment_mesh_1d(100);
//        let source = |x: f64| 2. * x.powi(2);
//
//        let source_vector = mesh
//            .vertices()
//            .iter()
//            .map(|x| source(x.0.x))
//            .collect::<Vec<f64>>();
//        let n = source_vector.len();
//        let mut source_ovector: nalgebra::DVector<f64> = nalgebra::DVector::<f64>::zeros(n);
//        for (s, sb) in source_vector.iter().zip(source_ovector.iter_mut()) {
//            *sb = *s;
//        }
//
//        let problem = super::PoissonSourceBuilder::new()
//            .with_mesh(&mesh)
//            .with_source(&source_ovector)
//            .build();
//
//        let mut test_solution: nalgebra::DVector<f64> = nalgebra::DVector::<f64>::zeros(n);
//        for (s, sb) in source_vector.iter().zip(test_solution.iter_mut()) {
//            *sb = *s;
//        }
//
//        test_solution = problem.solve_into(test_solution).unwrap();
//
//        let dense_operator =
//            nalgebra_sparse::convert::serial::convert_csc_dense(problem.operator());
//        let factorized_operator = dense_operator.lu();
//        let solb = factorized_operator.solve(problem.source()).unwrap();
//
//        for (x, y) in test_solution.iter().zip(solb.iter()) {
//            approx::assert_relative_eq!(x, y, epsilon = std::f64::EPSILON * 100.);
//        }
//    }
//}
//
