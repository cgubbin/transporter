// Copyright 2022 Chris Gubbin
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Poisson
//!
//! Arranges and solves the Poisson equation using a Gauss-Newton scheme with linesearch.
//! A predictor-correct scheme is utilised to accelerate the convergence of the simulation
//! Outer Loop. This estimates the change in the electron density arising from a small change
//! in the electrostatic potential using a semi-analytical formalism based on the Fermi distribution
//! of electrons in the gas (described in ...). Near convergence this reduces the number of outer
//! iterations necessary to beat the tolerance.
//!

/// Arranges the Poisson problem by constructing the differential operator and source term
mod operator;

use operator::PoissonOperator;

use super::{methods::OuterLoopInfoDesk, BuildError};
use crate::{device::info_desk::DeviceInfoDesk, postprocessor::Charge};
use argmin::core::{ArgminOp, Error};
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use ndarray::{Array1, Array2};
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

/// A builder struct, allowing for creating of a Poisson problem from common constructs in the crate
pub(crate) struct PoissonProblemBuilder<
    T: Copy + RealField,
    RefCharge,
    RefInfoDesk,
    RefInitial,
    RefMesh,
> {
    charge: RefCharge,
    info_desk: RefInfoDesk,
    initial_potential: RefInitial,
    mesh: RefMesh,
    marker: PhantomData<T>,
}

impl<T: Copy + RealField> Default for PoissonProblemBuilder<T, (), (), (), ()> {
    fn default() -> Self {
        Self {
            charge: (),
            info_desk: (),
            initial_potential: (),
            mesh: (),
            marker: PhantomData,
        }
    }
}

impl<T, RefCharge, RefInfoDesk, RefInitial, RefMesh>
    PoissonProblemBuilder<T, RefCharge, RefInfoDesk, RefInitial, RefMesh>
where
    T: Copy + RealField,
{
    pub(crate) fn with_charge<Charge>(
        self,
        charge: &Charge,
    ) -> PoissonProblemBuilder<T, &Charge, RefInfoDesk, RefInitial, RefMesh> {
        PoissonProblemBuilder {
            charge,
            info_desk: self.info_desk,
            initial_potential: self.initial_potential,
            mesh: self.mesh,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> PoissonProblemBuilder<T, RefCharge, &InfoDesk, RefInitial, RefMesh> {
        PoissonProblemBuilder {
            charge: self.charge,
            info_desk,
            initial_potential: self.initial_potential,
            mesh: self.mesh,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> PoissonProblemBuilder<T, RefCharge, RefInfoDesk, RefInitial, &Mesh> {
        PoissonProblemBuilder {
            charge: self.charge,
            info_desk: self.info_desk,
            initial_potential: self.initial_potential,
            mesh,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_initial_potential<Initial>(
        self,
        initial_potential: &Initial,
    ) -> PoissonProblemBuilder<T, RefCharge, RefInfoDesk, &Initial, RefMesh> {
        PoissonProblemBuilder {
            charge: self.charge,
            info_desk: self.info_desk,
            initial_potential,
            mesh: self.mesh,
            marker: PhantomData,
        }
    }
}

impl<'a, T, GeometryDim, Conn, BandDim>
    PoissonProblemBuilder<
        T,
        &'a Charge<T, BandDim>,
        &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
        &'a Potential<T>,
        &'a Mesh<T, GeometryDim, Conn>,
    >
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<Array1<T>, BandDim>,
{
    /// Build the Poisson problem from the Builder struct
    pub(crate) fn build(
        self,
    ) -> Result<PoissonProblem<'a, T, GeometryDim, Conn, BandDim>, BuildError> {
        Ok(PoissonProblem {
            charge: self.charge,
            info_desk: self.info_desk,
            mesh: self.mesh,
            operator: self.build_operator()?,
            source: self.build_source()?,
            fermi_level: Array1::from(self.info_desk.determine_fermi_level(
                self.mesh,
                self.initial_potential,
                self.charge,
            )),
            initial_values: self.initial_potential.clone(),
        })
    }
}

/// A Poisson problem
///
///
pub(crate) struct PoissonProblem<'a, T, GeometryDim, Conn, BandDim>
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<T, GeometryDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<Array1<T>, BandDim>,
{
    /// A reference to the free charge distribution at the current outer iteration
    charge: &'a Charge<T, BandDim>,
    /// A reference to the DeviceInfoDesk, which gives access to all fixed parameters necessary to solve the problem
    info_desk: &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
    /// A reference to the device mesh, to be used in constructing the differential operator
    mesh: &'a Mesh<T, GeometryDim, Conn>,
    /// The differential operator \nabla \epsilon \nabla \phi constructed on build of the Poisson problem
    operator: sprs::CsMat<T>,
    /// The static component of the system: the source term q * (N_D - N_A)
    source: ndarray::Array1<T>,
    /// Fermi level in the device: this changes through the calculation as a consequence of the predictor-corrector scheme
    fermi_level: ndarray::Array1<T>,
    /// An initial guess for the electrostatic potential, usually the result of the mixing procedure in the previous iteration
    initial_values: Potential<T>,
}

use super::Potential;

/// Implement the [ArgminOp](argmin::core::ArgminOp) trait from the `argmin` crate.
///
/// This trait defines the methods necessary to run the Gauss-Newton algorithm.
impl<T, GeometryDim, Conn, BandDim> ArgminOp for PoissonProblem<'_, T, GeometryDim, Conn, BandDim>
where
    T: Copy + RealField + argmin::core::ArgminFloat,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<T, GeometryDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<Array1<T>, BandDim>,
{
    type Param = Array1<T>;
    type Output = Array1<T>;
    type Hessian = ();
    type Jacobian = Array2<T>;
    type Float = T;

    /// Calculate the residual vector of the Poisson equation
    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        // As we are solving a Neumann problem, we substract the mean of the potential from the solution.
        // This makes the problem well posed (...???)
        let mean = p.mean().unwrap();
        let p = Array1::from(p.iter().map(|x| *x - mean).collect::<Vec<_>>());
        // Recalculate the free charge using the predictor corrector scheme based on the updated potential
        let free_charge = self.info_desk.update_source_vector(
            self.mesh,
            &self.fermi_level,
            &(&p + self.initial_values.as_ref()), // Currently updating with the current potential. Is this spart
        );

        // The source is the sum of the fixed charges and the free charge
        let mut source = &self.source + &free_charge;
        // To apply a Neumann condition we divide the source term by a factor of 2 at the system boundary
        source[0] /= T::one() + T::one();
        source[self.source.len() - 1] /= T::one() + T::one();
        let mean_source = source.mean().unwrap();
        for element in source.iter_mut() {
            *element -= mean_source;
        }

        let operator = self.operator.clone().to_dense();

        Ok(operator.dot(&(p + self.initial_values.as_ref())) + &source)
    }

    /// Calculate the Jacobian of the Poisson equation
    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
        // As we are solving a Neumann problem, we substract the mean of the potential from the solution.
        // This makes the problem well posed (...???)
        let mean = p.mean().unwrap();
        let p = Array1::from(p.iter().map(|x| *x - mean).collect::<Vec<_>>());
        // Update the diagonal component of the Jacobian based on the predictor-corrector scheme
        let jacobian_diagonal = self.info_desk.compute_jacobian_diagonal(
            &self.fermi_level,
            &(&p + self.initial_values.as_ref()),
            self.mesh,
        );
        // Set the `jacobian_csr` to the diagonal of the Poisson operator
        // this allows us to use the sparsity pattern of the diagonal to construct the jacobian diagonal
        let mut jacobian_csr = self.operator.diag();
        // re-assign all the values of `jacobian_csr` based on the calculated values.
        for ((_, val), &value) in jacobian_csr.iter_mut().zip(jacobian_diagonal.iter()) {
            *val = value;
        }

        // Apply the Neumann condition to the Jacobian
        *jacobian_csr.get_mut(0).unwrap() /= T::one() + T::one();
        *jacobian_csr.get_mut(jacobian_diagonal.len() - 1).unwrap() /= T::one() + T::one();

        let operator = self.operator.clone();

        let jacobian =
            &operator.to_dense() + &Array2::from_diag(&Array1::from(jacobian_csr.data().to_vec()));
        Ok(jacobian)
    }
}
