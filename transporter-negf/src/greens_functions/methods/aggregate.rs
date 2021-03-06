//! Methods for aggregated Green's functions
//!
//! Aggregated Green's functions are super-structures containing the Green's function matrices themselves
//! evaluated at all energy and wavevector points in the spectral grid over which the problem is defined.
//! Aggregated structures and methods are designed to integrate with the inner loop, whereas the individual
//! Green's functions should not be called directly.

use super::{
    super::{GreensFunction, GreensFunctionMethods},
    mixed::MMatrix,
};
use crate::{
    app::Calculation,
    device::info_desk::DeviceInfoDesk,
    error::{BuildError, CsrError},
    postprocessor::{Charge, Current},
    spectral::{SpectralDiscretisation, SpectralSpace, WavevectorSpace},
};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};
use ndarray::{Array1, Array2};
use num_complex::Complex;
use num_traits::ToPrimitive;
use sprs::CsMat;
use transporter_mesher::{Connectivity, ElementMethods, Mesh, SmallDim};

/// Builder struct for aggregated Green's functions
#[derive(Clone)]
pub struct GreensFunctionBuilder<T, RefInfoDesk, RefMesh, RefSpectral, RefCalculationType> {
    /// Placeholder for a reference to the problem's information desk
    pub(crate) info_desk: RefInfoDesk,
    /// Placeholder for a reference to the problem's Mesh
    pub(crate) mesh: RefMesh,
    /// Placeholder for a reference to the problems spectral discretisation
    pub(crate) spectral: RefSpectral,
    /// Placeholder for the flavour of calculation (coherent or incoherent)
    pub(crate) calculation: RefCalculationType,
    /// Whether or not to do numerical security checks
    pub(crate) security_checks: bool,
    /// Marker to set the numeric type `T` on instantiation
    pub(crate) marker: std::marker::PhantomData<T>,
}

impl<T> Default for GreensFunctionBuilder<T, (), (), (), ()>
where
    T: RealField,
{
    /// Initialise an empty instance of `GreensFunctionBuilder`
    fn default() -> Self {
        Self {
            info_desk: (),
            mesh: (),
            spectral: (),
            calculation: (),
            security_checks: false,
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, RefInfoDesk, RefMesh, RefSpectral, RefCalculationType>
    GreensFunctionBuilder<T, RefInfoDesk, RefMesh, RefSpectral, RefCalculationType>
{
    /// Attach an information desk to the builder
    pub fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> GreensFunctionBuilder<T, &InfoDesk, RefMesh, RefSpectral, RefCalculationType> {
        GreensFunctionBuilder {
            info_desk,
            mesh: self.mesh,
            spectral: self.spectral,
            calculation: self.calculation,
            security_checks: self.security_checks,
            marker: std::marker::PhantomData,
        }
    }

    /// Attach a mesh to the builder
    pub fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> GreensFunctionBuilder<T, RefInfoDesk, &Mesh, RefSpectral, RefCalculationType> {
        GreensFunctionBuilder {
            info_desk: self.info_desk,
            mesh,
            spectral: self.spectral,
            calculation: self.calculation,
            security_checks: self.security_checks,
            marker: std::marker::PhantomData,
        }
    }

    /// Attach the spectral discretisation to the builder
    pub fn with_spectral_discretisation<Spectral>(
        self,
        spectral: &Spectral,
    ) -> GreensFunctionBuilder<T, RefInfoDesk, RefMesh, &Spectral, RefCalculationType> {
        GreensFunctionBuilder {
            info_desk: self.info_desk,
            mesh: self.mesh,
            spectral,
            calculation: self.calculation,
            security_checks: self.security_checks,
            marker: std::marker::PhantomData,
        }
    }

    /// Select the calculation flavour
    pub(crate) fn incoherent_calculation<CalculationType>(
        self,
        calculation: &CalculationType,
    ) -> GreensFunctionBuilder<T, RefInfoDesk, RefMesh, RefSpectral, &CalculationType> {
        GreensFunctionBuilder {
            info_desk: self.info_desk,
            mesh: self.mesh,
            spectral: self.spectral,
            calculation,
            security_checks: self.security_checks,
            marker: std::marker::PhantomData,
        }
    }

    /// Change the default security setting
    pub(crate) fn with_security_checks(
        self,
        security_checks: bool,
    ) -> GreensFunctionBuilder<T, RefInfoDesk, RefMesh, RefSpectral, RefCalculationType> {
        GreensFunctionBuilder {
            info_desk: self.info_desk,
            mesh: self.mesh,
            spectral: self.spectral,
            calculation: self.calculation,
            security_checks,
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a, GeometryDim, Conn, BandDim>
    GreensFunctionBuilder<
        f64,
        &'a DeviceInfoDesk<f64, GeometryDim, BandDim>,
        &'a Mesh<f64, GeometryDim, Conn>,
        &'a SpectralSpace<f64, ()>,
        (),
    >
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim>,
    DefaultAllocator:
        Allocator<f64, GeometryDim> + Allocator<f64, BandDim> + Allocator<[f64; 3], BandDim>,
{
    /// When the spectral discretisation attached is `SpectralSpace<T, ()>` the calculation proposed does not require
    /// a numerical integration over wavevector. In this case this function constructs an instance of `AggregateGreensFunctions`
    /// where the underlying data structure is a sparse `CsrMatrix`. This is possible because transport is coherent. The coherence of
    /// the transport should be determined before calling this method. This method is only currently suitable for 1d structures
    /// with contacts in the edge elements.
    // TODO Gatekeep for U1
    pub fn build(
        self,
    ) -> Result<
        AggregateGreensFunctions<'a, f64, CsMat<Complex<f64>>, GeometryDim, BandDim>,
        BuildError,
    > {
        Ok(self.build_inner()?)
    }

    fn build_inner(
        self,
    ) -> Result<
        AggregateGreensFunctions<'a, f64, CsMat<Complex<f64>>, GeometryDim, BandDim>,
        CsrError,
    > {
        // Assemble the sparsity pattern for the retarded green's function
        let number_of_vertices_in_internal_lead =
            if let Some(lead_length) = self.info_desk.lead_length {
                (lead_length * 1e-9 / self.mesh.elements()[0].0.diameter())
                    .to_usize()
                    .unwrap()
            } else {
                0
            };
        let sparsity_pattern = assemble_csr_sparsity_for_retarded_gf(
            self.mesh.vertices().len(),
            number_of_vertices_in_internal_lead,
        )?;
        // Fill with dummy values
        let initial_values = vec![Complex::from(0_f64); sparsity_pattern.nnz()];
        let csr = CsrMatrix::try_from_pattern_and_values(sparsity_pattern, initial_values.clone())?;
        let diagonal_sparsity_pattern = csr.diagonal_as_csr().pattern().clone();

        let csr = CsMat::new(
            (csr.nrows(), csr.ncols()),
            csr.row_offsets().to_vec(),
            csr.col_indices().to_vec(),
            initial_values,
        );
        // Map the csr matrix over the number of points in the spectral space to assemble the initial Green's function vector
        let spectrum_of_csr = (0..self.spectral.total_number_of_points())
            .map(|_| GreensFunction {
                matrix: csr.clone(),
                marker: std::marker::PhantomData,
            })
            .collect::<Vec<_>>();

        // Get the sparsity pattern corresponding to the diagonal of `csr`
        let initial_values = vec![Complex::from(0_f64); diagonal_sparsity_pattern.nnz()];
        let diagonal_csr = CsrMatrix::try_from_pattern_and_values(
            diagonal_sparsity_pattern,
            initial_values.clone(),
        )?;
        let diagonal_csr = CsMat::new(
            (diagonal_csr.nrows(), diagonal_csr.ncols()),
            diagonal_csr.row_offsets().to_vec(),
            diagonal_csr.col_indices().to_vec(),
            initial_values,
        );
        // Map the csr matrix over the number of points in the spectral space to assemble the initial Green's function vector
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
            security_checks: self.security_checks,
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
        (),
    >
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim> + Send + Sync,
    <Conn as Connectivity<T, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator:
        Allocator<T, GeometryDim> + Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
    <DefaultAllocator as Allocator<T, GeometryDim>>::Buffer: Send + Sync,
{
    /// When the attached spectral space has wavevector discretisation we build out a vector of dense
    /// `DMatrix` as the scattering process under study is incoherent.
    /// The nested method lets me upcast the error to `BuildError`.
    pub fn build(
        self,
    ) -> Result<AggregateGreensFunctions<'a, T, CsMat<Complex<T>>, GeometryDim, BandDim>, BuildError>
    {
        Ok(self.build_inner()?)
    }

    fn build_inner(
        self,
    ) -> Result<AggregateGreensFunctions<'a, T, CsMat<Complex<T>>, GeometryDim, BandDim>, CsrError>
    {
        // Assemble the sparsity pattern for the retarded green's function
        let sparsity_pattern =
            assemble_csr_sparsity_for_retarded_gf(self.mesh.vertices().len(), 0)?;
        // Fill with dummy values
        let initial_values = vec![Complex::from(T::zero()); sparsity_pattern.nnz()];
        let csr = CsrMatrix::try_from_pattern_and_values(sparsity_pattern, initial_values.clone())?;
        let diagonal_sparsity_pattern = csr.diagonal_as_csr().pattern().clone();

        let csr = CsMat::new(
            (csr.nrows(), csr.ncols()),
            csr.row_offsets().to_vec(),
            csr.col_indices().to_vec(),
            initial_values,
        );
        // Map the csr matrix over the number of points in the spectral space to assemble the initial Green's function vector
        let spectrum_of_csr = (0..self.spectral.total_number_of_points())
            .map(|_| GreensFunction {
                matrix: csr.clone(),
                marker: std::marker::PhantomData,
            })
            .collect::<Vec<_>>();

        // Get the sparsity pattern corresponding to the diagonal of `csr`
        let initial_values = vec![Complex::from(T::zero()); diagonal_sparsity_pattern.nnz()];
        let diagonal_csr = CsrMatrix::try_from_pattern_and_values(
            diagonal_sparsity_pattern,
            initial_values.clone(),
        )?;

        let diagonal_csr = CsMat::new(
            (diagonal_csr.nrows(), diagonal_csr.ncols()),
            diagonal_csr.row_offsets().to_vec(),
            diagonal_csr.col_indices().to_vec(),
            initial_values,
        );
        let spectrum_of_diagonal_csr = (0..self.spectral.total_number_of_points())
            .map(|_| GreensFunction {
                matrix: diagonal_csr.clone(),
                marker: std::marker::PhantomData,
            })
            .collect::<Vec<_>>();

        // In the coherent calculation we do not use the advanced or greater Greens function. We therefore do not fill these elements
        // in the resulting `AggregateGreensFunctions` struct
        Ok(AggregateGreensFunctions {
            info_desk: self.info_desk,
            security_checks: self.security_checks,
            retarded: spectrum_of_csr,
            advanced: Vec::new(),
            lesser: spectrum_of_diagonal_csr,
            greater: Vec::new(),
        })
    }
}

impl<'a, GeometryDim, Conn, BandDim>
    GreensFunctionBuilder<
        f64,
        &'a DeviceInfoDesk<f64, GeometryDim, BandDim>,
        &'a Mesh<f64, GeometryDim, Conn>,
        &'a SpectralSpace<f64, WavevectorSpace<f64, GeometryDim, Conn>>,
        &'a Calculation<f64>,
    >
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<f64, GeometryDim> + Send + Sync,
    <Conn as Connectivity<f64, GeometryDim>>::Element: Send + Sync,
    DefaultAllocator:
        Allocator<f64, GeometryDim> + Allocator<f64, BandDim> + Allocator<[f64; 3], BandDim>,
    <DefaultAllocator as Allocator<f64, GeometryDim>>::Buffer: Send + Sync,
{
    /// When the attached spectral space has wavevector discretisation we build out a vector of dense
    /// `DMatrix` as the scattering process under study is incoherent.
    pub fn build(
        self,
    ) -> Result<
        AggregateGreensFunctions<'a, f64, Array2<Complex<f64>>, GeometryDim, BandDim>,
        BuildError,
    > {
        let matrix = GreensFunction {
            matrix: Array2::zeros((self.mesh.vertices().len(), self.mesh.vertices().len())),
            marker: std::marker::PhantomData,
        };

        let num_spectral_points = self.spectral.total_number_of_points();
        Ok(AggregateGreensFunctions {
            info_desk: self.info_desk,
            security_checks: self.security_checks,
            retarded: vec![matrix.clone(); num_spectral_points],
            advanced: Vec::new(),
            lesser: vec![matrix; num_spectral_points],
            greater: Vec::new(),
        })
    }

    /// Build an `AggregateGreensFunction` with the mixed `MMatrix` backend
    pub fn build_mixed(
        self,
    ) -> Result<
        AggregateGreensFunctions<'a, f64, MMatrix<Complex<f64>>, GeometryDim, BandDim>,
        BuildError,
    > {
        // Assemble the sparsity pattern for the retarded green's function
        let number_of_vertices_in_reservoir = if let Some(lead_length) = self.info_desk.lead_length
        {
            (lead_length * 1e-9 / self.mesh.elements()[0].0.diameter())
                .to_usize()
                .unwrap()
        } else {
            0
        };
        let number_of_vertices_in_core =
            self.mesh.vertices().len() - 2 * number_of_vertices_in_reservoir;

        let matrix = GreensFunction {
            matrix: MMatrix::zeros(number_of_vertices_in_reservoir, number_of_vertices_in_core),
            marker: std::marker::PhantomData,
        };

        let num_spectral_points = self.spectral.total_number_of_points();
        Ok(AggregateGreensFunctions {
            info_desk: self.info_desk,
            security_checks: self.security_checks,
            retarded: vec![matrix.clone(); num_spectral_points],
            advanced: Vec::new(),
            lesser: vec![matrix; num_spectral_points],
            greater: Vec::new(),
        })
    }
}

#[derive(Debug)]
/// A struct to hold `GreensFunction` for each point in the simulation `SpectralSpace`
/// These are stored in `Vec` with ordering as explained in the `spectral` module
pub struct AggregateGreensFunctions<'a, T, Matrix, GeometryDim, BandDim>
where
    Matrix: GreensFunctionMethods<T> + Send + Sync,
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    DefaultAllocator: Allocator<T, BandDim> + Allocator<[T; 3], BandDim>,
{
    /// A reference to the problem info desk, allowing material parameters to be evaluated in called methods
    pub(crate) info_desk: &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
    /// The security check setting
    pub(crate) security_checks: bool,
    /// The retarded Greens function. The length of the Vec is n_energy * n_wavevector. All wavevectors are stored sequentially before the energy is incremented
    pub(crate) retarded: Vec<GreensFunction<Matrix, T>>,
    pub(crate) advanced: Vec<GreensFunction<Matrix, T>>,
    pub(crate) lesser: Vec<GreensFunction<Matrix, T>>,
    pub(crate) greater: Vec<GreensFunction<Matrix, T>>,
}

/// Methods for `AggregateGreensFunctions`
///
/// These calculate observable deriveable from the full distribution of Green's functions
pub trait AggregateGreensFunctionMethods<T, BandDim, GeometryDim, Conn, Integrator, SelfEnergy>
where
    T: RealField,
    BandDim: SmallDim,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    Integrator: SpectralDiscretisation<T>,
    DefaultAllocator: Allocator<Array1<T>, BandDim> + Allocator<T, GeometryDim>,
{
    /// Operates on the collected Green's functions to calculate the total charge in each of the bands
    fn accumulate_into_charge_density_vector(
        &self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        integrator: &Integrator,
    ) -> Result<Charge<T, BandDim>, crate::postprocessor::PostProcessorError>;
    /// Operates on the collected Green's functions to calculate the total current in each band
    fn accumulate_into_current_density_vector(
        &self,
        voltage: T,
        mesh: &Mesh<T, GeometryDim, Conn>,
        self_energy: &SelfEnergy,
        integrator: &Integrator,
    ) -> Result<Current<T, BandDim>, crate::postprocessor::PostProcessorError>;
}

/// A helper function to assemble the csr pattern for the retarded Green's function
///
/// The desired Green's function has non-zero values on the leading diagonal, and in it's first
/// and last columns
pub(crate) fn assemble_csr_sparsity_for_retarded_gf(
    number_of_elements_in_mesh: usize,
    number_of_elements_in_contacts: usize,
) -> Result<SparsityPattern, nalgebra_sparse::pattern::SparsityPatternFormatError> {
    if number_of_elements_in_contacts == 0 {
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
        SparsityPattern::try_from_offsets_and_indices(
            number_of_elements_in_mesh,
            number_of_elements_in_mesh,
            row_offsets,
            col_indices,
        )
    } else {
        let mut col_indices = vec![0]; // In the first row only the diagonal is occupied
        let mut row_offsets = vec![0, 1]; // with a single element
        for idx in 1..number_of_elements_in_contacts {
            col_indices.push(idx);
            row_offsets.push(row_offsets.last().unwrap() + 1);
        }
        col_indices.push(number_of_elements_in_contacts);
        col_indices.push(number_of_elements_in_mesh - 1 - number_of_elements_in_contacts);
        row_offsets.push(row_offsets.last().unwrap() + 2);
        for idx in (number_of_elements_in_contacts + 1)
            ..(number_of_elements_in_mesh - number_of_elements_in_contacts - 1)
        {
            col_indices.push(number_of_elements_in_contacts);
            col_indices.push(idx);
            col_indices.push(number_of_elements_in_mesh - 1 - number_of_elements_in_contacts);
            row_offsets.push(row_offsets.last().unwrap() + 3)
        }
        col_indices.push(number_of_elements_in_contacts);
        col_indices.push(number_of_elements_in_mesh - 1 - number_of_elements_in_contacts);
        row_offsets.push(row_offsets.last().unwrap() + 2);
        for idx in (number_of_elements_in_mesh - number_of_elements_in_contacts)
            ..number_of_elements_in_mesh
        {
            col_indices.push(idx);
            row_offsets.push(row_offsets.last().unwrap() + 1);
        }
        SparsityPattern::try_from_offsets_and_indices(
            number_of_elements_in_mesh,
            number_of_elements_in_mesh,
            row_offsets,
            col_indices,
        )
    }
}

#[cfg(test)]
mod test {
    // use super::super::sparse::CsrAssembly;
    // use nalgebra::{DMatrix, DVector};
    // use nalgebra_sparse::CsrMatrix;
    use num_complex::Complex;

    // #[test]
    // fn test_csr_assemble_of_diagonal_and_left_and_right_columns() {
    //     use rand::Rng;
    //     let mut rng = rand::thread_rng();

    //     for nrows in 5..20 {
    //         let left_column: DVector<Complex<f64>> = DVector::from(
    //             (0..nrows)
    //                 .map(|_| rng.gen::<f64>())
    //                 .map(Complex::from)
    //                 .collect::<Vec<_>>(),
    //         );
    //         let right_column: DVector<Complex<f64>> = DVector::from(
    //             (0..nrows)
    //                 .map(|_| rng.gen::<f64>())
    //                 .map(Complex::from)
    //                 .collect::<Vec<_>>(),
    //         );
    //         let mut diagonal: DVector<Complex<f64>> = DVector::from(
    //             (0..nrows)
    //                 .map(|_| rng.gen::<f64>())
    //                 .map(Complex::from)
    //                 .collect::<Vec<_>>(),
    //         );
    //         diagonal[0] = left_column[0];
    //         diagonal[nrows - 1] = right_column[nrows - 1];

    //         let mut dense_matrix: DMatrix<Complex<f64>> =
    //             DMatrix::from_element(nrows, nrows, Complex::from(0f64));
    //         for idx in 0..nrows {
    //             dense_matrix[(idx, 0)] = left_column[idx];
    //             dense_matrix[(idx, nrows - 1)] = right_column[idx];
    //             dense_matrix[(idx, idx)] = diagonal[idx];
    //         }

    //         // Construct the sparsity pattern
    //         let number_of_elements_in_mesh = diagonal.len();
    //         let pattern =
    //             super::assemble_csr_sparsity_for_retarded_gf(number_of_elements_in_mesh, 0)
    //                 .unwrap();
    //         let values = vec![Complex::from(0_f64); pattern.nnz()];
    //         let mut csr = CsrMatrix::try_from_pattern_and_values(pattern, values).unwrap();

    //         csr.assemble_retarded_diagonal_and_columns_into_csr(
    //             diagonal,
    //             left_column,
    //             right_column,
    //         )
    //         .unwrap();

    //         let csr_to_dense = nalgebra_sparse::convert::serial::convert_csr_dense(&csr);

    //         for (element, other) in dense_matrix.into_iter().zip(csr_to_dense.into_iter()) {
    //             assert_eq!(element, other);
    //         }
    //     }
    // }

    use ndarray::Array1;

    #[test]
    fn test_csr_sparsity_of_diagonal_and_left_and_right_columns_with_finite_leads() {
        for nrows in 5..30 {
            for n_elements_in_contact in 1..(nrows / 2 - 1) {
                let left_column: Array1<Complex<f64>> = Array1::from(
                    (0..nrows - 2 * n_elements_in_contact)
                        .map(|_| 1_f64)
                        .map(Complex::from)
                        .collect::<Vec<_>>(),
                );
                let right_column: Array1<Complex<f64>> = Array1::from(
                    (0..nrows - 2 * n_elements_in_contact)
                        .map(|_| 1_f64)
                        .map(Complex::from)
                        .collect::<Vec<_>>(),
                );
                let mut diagonal: Array1<Complex<f64>> = Array1::from(
                    (0..nrows)
                        .map(|_| 1_f64)
                        .map(Complex::from)
                        .collect::<Vec<_>>(),
                );
                // Fill the edge elements
                diagonal[n_elements_in_contact] = left_column[0];
                diagonal[nrows - 1 - n_elements_in_contact] = right_column[right_column.len() - 1];

                let mut dense_matrix: ndarray::Array2<Complex<f64>> =
                    ndarray::Array2::zeros((nrows, nrows));
                for idx in 0..nrows {
                    dense_matrix[(idx, idx)] = diagonal[idx];
                }
                for idx in 0..(nrows - 2 * n_elements_in_contact) {
                    dense_matrix[(idx + n_elements_in_contact, n_elements_in_contact)] =
                        left_column[idx];
                    dense_matrix[(
                        idx + n_elements_in_contact,
                        nrows - 1 - n_elements_in_contact,
                    )] = right_column[idx];
                }

                // Construct the sparsity pattern
                // let pattern =
                //     super::assemble_csr_sparsity_for_retarded_gf(nrows, n_elements_in_contact)
                //         .unwrap();
                // let values = vec![Complex::from(1_f64); pattern.nnz()];
                // let csr = CsrMatrix::try_from_pattern_and_values(pattern, values).unwrap();

                // let csr_to_dense = nalgebra_sparse::convert::serial::convert_csr_dense(&csr);

                // for (element, other) in dense_matrix.into_iter().zip(csr_to_dense.into_iter()) {
                //     assert_eq!(element, other);
                // }
            }
        }
    }

    // #[test]
    // fn test_csr_assemble_of_diagonal_and_left_and_right_columns_with_finite_leads() {
    //     use rand::Rng;
    //     let mut rng = rand::thread_rng();

    //     for nrows in 5..30 {
    //         for n_elements_in_contact in 1..(nrows / 2 - 1) {
    //             let left_column: DVector<Complex<f64>> = DVector::from(
    //                 (0..nrows - 2 * n_elements_in_contact)
    //                     .map(|_| rng.gen::<f64>())
    //                     .map(Complex::from)
    //                     .collect::<Vec<_>>(),
    //             );
    //             let right_column: DVector<Complex<f64>> = DVector::from(
    //                 (0..nrows - 2 * n_elements_in_contact)
    //                     .map(|_| rng.gen::<f64>())
    //                     .map(Complex::from)
    //                     .collect::<Vec<_>>(),
    //             );
    //             let mut diagonal: DVector<Complex<f64>> = DVector::from(
    //                 (0..nrows)
    //                     .map(|_| rng.gen::<f64>())
    //                     .map(Complex::from)
    //                     .collect::<Vec<_>>(),
    //             );
    //             // Fill the edge elements
    //             diagonal[n_elements_in_contact] = left_column[0];
    //             diagonal[nrows - 1 - n_elements_in_contact] = right_column[right_column.len() - 1];

    //             let mut dense_matrix: DMatrix<Complex<f64>> =
    //                 DMatrix::from_element(nrows, nrows, Complex::from(0f64));
    //             for idx in 0..nrows {
    //                 dense_matrix[(idx, idx)] = diagonal[idx];
    //             }
    //             for idx in 0..(nrows - 2 * n_elements_in_contact) {
    //                 dense_matrix[(idx + n_elements_in_contact, n_elements_in_contact)] =
    //                     left_column[idx];
    //                 dense_matrix[(
    //                     idx + n_elements_in_contact,
    //                     nrows - 1 - n_elements_in_contact,
    //                 )] = right_column[idx];
    //             }

    //             // Construct the sparsity pattern
    //             let pattern =
    //                 super::assemble_csr_sparsity_for_retarded_gf(nrows, n_elements_in_contact)
    //                     .unwrap();
    //             let values = vec![Complex::from(0_f64); pattern.nnz()];
    //             let mut csr = CsrMatrix::try_from_pattern_and_values(pattern, values).unwrap();

    //             csr.assemble_retarded_diagonal_and_columns_into_csr(
    //                 diagonal,
    //                 left_column,
    //                 right_column,
    //             )
    //             .unwrap();

    //             let csr_to_dense = nalgebra_sparse::convert::serial::convert_csr_dense(&csr);

    //             for (element, other) in dense_matrix.into_iter().zip(csr_to_dense.into_iter()) {
    //                 assert_eq!(element, other);
    //             }
    //         }
    //     }
    // }
}
