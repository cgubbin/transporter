//! Methods for aggregated Green's functions
//!
//! Aggregated Green's functions are super-structures containing the Green's function matrices themselves
//! evaluated at all energy and wavevector points in the spectral grid over which the problem is defined.
//! Aggregated structures and methods are designed to integrate with the inner loop, whereas the individual
//! Green's functions should not be called directly.
use super::{GreensFunction, GreensFunctionMethods};
use crate::postprocessor::{Charge, Current};
use crate::{
    device::info_desk::DeviceInfoDesk,
    spectral::{SpectralDiscretisation, SpectralSpace, WavevectorSpace},
};
use color_eyre::eyre::eyre;
use nalgebra::{
    allocator::Allocator, Const, DMatrix, DefaultAllocator, Dynamic, Matrix, RealField, VecStorage,
};
use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};
use num_complex::Complex;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

/// Builder struct for aggregated Green's functions
#[derive(Clone)]
pub(crate) struct GreensFunctionBuilder<T, RefInfoDesk, RefMesh, RefSpectral> {
    /// Placeholder for a reference to the problem's information desk
    pub(crate) info_desk: RefInfoDesk,
    /// Placeholder for a reference to the problem's Mesh
    pub(crate) mesh: RefMesh,
    /// Placeholder for a reference to the problems spectral discretisation
    pub(crate) spectral: RefSpectral,
    /// Marker to set the numeric type `T` on instantiation
    pub(crate) marker: std::marker::PhantomData<T>,
}

impl<T> GreensFunctionBuilder<T, (), (), ()>
where
    T: RealField,
{
    /// Initialise an empty instance of `GreensFunctionBuilder`
    pub(crate) fn new() -> Self {
        Self {
            info_desk: (),
            mesh: (),
            spectral: (),
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, RefInfoDesk, RefMesh, RefSpectral>
    GreensFunctionBuilder<T, RefInfoDesk, RefMesh, RefSpectral>
{
    /// Attach an information desk to the builder
    pub(crate) fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> GreensFunctionBuilder<T, &InfoDesk, RefMesh, RefSpectral> {
        GreensFunctionBuilder {
            info_desk,
            mesh: self.mesh,
            spectral: self.spectral,
            marker: std::marker::PhantomData,
        }
    }

    /// Attach a mesh to the builder
    pub(crate) fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> GreensFunctionBuilder<T, RefInfoDesk, &Mesh, RefSpectral> {
        GreensFunctionBuilder {
            info_desk: self.info_desk,
            mesh,
            spectral: self.spectral,
            marker: std::marker::PhantomData,
        }
    }

    /// Attach the spectral discretisation to the builder
    pub(crate) fn with_spectral_discretisation<Spectral>(
        self,
        spectral: &Spectral,
    ) -> GreensFunctionBuilder<T, RefInfoDesk, RefMesh, &Spectral> {
        GreensFunctionBuilder {
            info_desk: self.info_desk,
            mesh: self.mesh,
            spectral,
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T, GeometryDim, Conn, BandDim>
    GreensFunctionBuilder<
        T,
        &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
        &'a Mesh<T, GeometryDim, Conn>,
        &'a SpectralSpace<T, ()>,
    >
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator:
        Allocator<T, GeometryDim> + Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    /// When the spectral discretisation attached is `SpectralSpace<T, ()>` the calculation proposed does not require
    /// a numerical integration over wavevector. In this case this function constructs an instance of `AggregateGreensFunctions`
    /// where the underlying data structure is a sparse `CsrMatrix`. This is possible because transport is coherent. The coherence of
    /// the transport should be determined before calling this method. This method is only currently suitable for 1d structures
    /// with contacts in the edge elements.
    // TODO Gatekeep for U1
    pub(crate) fn build(
        self,
    ) -> color_eyre::Result<
        AggregateGreensFunctions<'a, T, CsrMatrix<Complex<T>>, GeometryDim, BandDim>,
    > {
        // Assemble the sparsity pattern for the retarded green's function
        let sparsity_pattern = assemble_csr_sparsity_for_retarded_gf(self.mesh.elements().len())?;
        // Fill with dummy values
        let initial_values = vec![Complex::from(T::zero()); sparsity_pattern.nnz()];
        let csr = CsrMatrix::try_from_pattern_and_values(sparsity_pattern, initial_values)
            .map_err(|e| eyre!("Failed to write values to Csr GF Matrix {:?}", e))?;
        // Map the csr matrix over the number of points in the spectral space to assemble the initial Green's function vector
        let spectrum_of_csr = (0..self.spectral.total_number_of_points())
            .map(|_| GreensFunction {
                matrix: csr.clone(),
                marker: std::marker::PhantomData,
            })
            .collect::<Vec<_>>();

        // Get the sparsity pattern corresponding to the diagonal of `csr`
        let diagonal_sparsity_pattern = csr.diagonal_as_csr().pattern().clone();
        let initial_values = vec![Complex::from(T::zero()); diagonal_sparsity_pattern.nnz()];
        let diagonal_csr =
            CsrMatrix::try_from_pattern_and_values(diagonal_sparsity_pattern, initial_values)
                .map_err(|e| eyre!("Failed to write values to Csr GF Matrix {:?}", e))?;
        let spectrum_of_diagonal_csr = (0..self.spectral.total_number_of_points())
            .map(|_| GreensFunction {
                matrix: diagonal_csr.clone(),
                marker: std::marker::PhantomData,
            })
            .collect::<Vec<_>>();

        // In the coherent calculation we do not use the advanced or greater Greens function. We therefore do not fill these elements
        // in the resulting `AggregateGreensFunctions` struct
        Ok(AggregateGreensFunctions {
            //    spectral: self.spectral,
            info_desk: self.info_desk,
            retarded: spectrum_of_csr,
            advanced: Vec::new(),
            lesser: spectrum_of_diagonal_csr,
            greater: Vec::new(),
        })
    }
}

impl<'a, T, GeometryDim, Conn, BandDim>
    GreensFunctionBuilder<
        T,
        &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
        &'a Mesh<T, GeometryDim, Conn>,
        &'a SpectralSpace<T, WavevectorSpace<T, GeometryDim, Conn>>,
    >
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator:
        Allocator<T, GeometryDim> + Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    /// When the attached spectral space has wavevector discretisation we build out a vector of dense
    /// `DMatrix` as the scattering process under study is incoherent.
    pub(crate) fn build(
        self,
    ) -> AggregateGreensFunctions<'a, T, DMatrix<Complex<T>>, GeometryDim, BandDim> {
        AggregateGreensFunctions {
            //    spectral: self.spectral,
            info_desk: self.info_desk,
            retarded: Vec::with_capacity(self.spectral.total_number_of_points()),
            advanced: Vec::with_capacity(self.spectral.total_number_of_points()),
            lesser: Vec::with_capacity(self.spectral.total_number_of_points()),
            greater: Vec::with_capacity(self.spectral.total_number_of_points()),
        }
    }
}
#[derive(Debug)]
pub(crate) struct AggregateGreensFunctions<'a, T, Matrix, GeometryDim, BandDim>
where
    Matrix: GreensFunctionMethods<T>,
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    /// A reference to the problem info desk, allowing material parameters to be evaluated in called methods
    pub(crate) info_desk: &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
    /// The retarded Greens function. The length of the Vec is n_energy * n_wavevector. All wavevectors are stored sequentially before the energy is incremented
    pub(crate) retarded: Vec<GreensFunction<Matrix, T>>,
    pub(crate) advanced: Vec<GreensFunction<Matrix, T>>,
    pub(crate) lesser: Vec<GreensFunction<Matrix, T>>,
    pub(crate) greater: Vec<GreensFunction<Matrix, T>>,
}

pub(crate) trait AggregateGreensFunctionMethods<
    T,
    BandDim,
    GeometryDim,
    Conn,
    Integrator,
    SelfEnergy,
> where
    T: RealField,
    BandDim: SmallDim,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    Integrator: SpectralDiscretisation<T>,
    DefaultAllocator: Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        > + Allocator<T, GeometryDim>,
{
    /// Operates on the collected Green's functions to calculate the total charge in each of the bands
    fn accumulate_into_charge_density_vector(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        integrator: &Integrator,
    ) -> color_eyre::Result<Charge<T, BandDim>>;
    /// Operates on the collected Green's functions to calculate the total current in each band
    fn accumulate_into_current_density_vector(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        self_energy: &SelfEnergy,
        integrator: &Integrator,
    ) -> color_eyre::Result<Current<T, BandDim>>;
}

/// A helper function to assemble the csr pattern for the retarded Green's function
///
/// The desired Green's function has non-zero values on the leading diagonal, and in it's first
/// and last columns
pub(crate) fn assemble_csr_sparsity_for_retarded_gf(
    number_of_elements_in_mesh: usize,
) -> color_eyre::Result<SparsityPattern> {
    let mut col_indices = vec![0, number_of_elements_in_mesh - 1]; // In the first row the first and last columns are occupied
    let mut row_offsets = vec![0, 2]; // 2 elements in the first row
                                      // In the core rows of the matrix there are three entries, in the first, last and diagonal columns
    for idx in 1..number_of_elements_in_mesh - 1 {
        col_indices.push(0);
        col_indices.push(idx);
        col_indices.push(number_of_elements_in_mesh - 1);
        row_offsets.push(row_offsets.last().unwrap() + 3)
    }
    // The last row has entries in the first and last column
    col_indices.push(0);
    col_indices.push(number_of_elements_in_mesh - 1);
    row_offsets.push(row_offsets.last().unwrap() + 2);
    // Construct the pattern
    SparsityPattern::try_from_offsets_and_indices(
        number_of_elements_in_mesh,
        number_of_elements_in_mesh,
        row_offsets,
        col_indices,
    )
    .map_err(|e| eyre!("Failed to construct Csr Sparsity pattern {:?}", e))
}

#[cfg(test)]
mod test {
    use super::super::sparse::CsrAssembly;
    use nalgebra::{DMatrix, DVector};
    use nalgebra_sparse::CsrMatrix;
    use num_complex::Complex;

    #[test]
    fn test_csr_assemble_of_diagonal_and_left_and_right_columns() {
        use rand::Rng;
        let nrows = 5;
        let mut rng = rand::thread_rng();

        let left_column: DVector<Complex<f64>> = DVector::from(
            (0..nrows)
                .map(|_| rng.gen::<f64>())
                .map(Complex::from)
                .collect::<Vec<_>>(),
        );
        let right_column: DVector<Complex<f64>> = DVector::from(
            (0..nrows)
                .map(|_| rng.gen::<f64>())
                .map(Complex::from)
                .collect::<Vec<_>>(),
        );
        let mut diagonal: DVector<Complex<f64>> = DVector::from(
            (0..nrows)
                .map(|_| rng.gen::<f64>())
                .map(Complex::from)
                .collect::<Vec<_>>(),
        );
        diagonal[0] = left_column[0];
        diagonal[nrows - 1] = right_column[nrows - 1];

        let mut dense_matrix: DMatrix<Complex<f64>> =
            DMatrix::from_element(nrows, nrows, Complex::from(0f64));
        for idx in 0..nrows {
            dense_matrix[(idx, 0)] = left_column[idx];
            dense_matrix[(idx, nrows - 1)] = right_column[idx];
            dense_matrix[(idx, idx)] = diagonal[idx];
        }

        // Construct the sparsity pattern
        let number_of_elements_in_mesh = diagonal.len();
        let pattern =
            super::assemble_csr_sparsity_for_retarded_gf(number_of_elements_in_mesh).unwrap();
        let values = vec![Complex::from(0_f64); pattern.nnz()];
        let mut csr = CsrMatrix::try_from_pattern_and_values(pattern, values).unwrap();

        csr.assemble_retarded_diagonal_and_columns_into_csr(diagonal, left_column, right_column)
            .unwrap();

        let csr_to_dense = nalgebra_sparse::convert::serial::convert_csr_dense(&csr);

        println!("{csr_to_dense}");
        println!("{dense_matrix}");

        for (element, other) in dense_matrix.into_iter().zip(csr_to_dense.into_iter()) {
            assert_eq!(element, other);
        }
    }
}
