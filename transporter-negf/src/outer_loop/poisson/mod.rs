mod operator;

use operator::PoissonOperator;

use super::methods::OuterLoopInfoDesk;
use crate::device::info_desk::DeviceInfoDesk;
use crate::postprocessor::Charge;
use argmin::core::{Error, Jacobian, Operator};
use nalgebra::{allocator::Allocator, DMatrix, DVector, DefaultAllocator, RealField};
use nalgebra_sparse::CsrMatrix;
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

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

impl<T: Copy + RealField, RefCharge, RefInfoDesk, RefInitial, RefMesh>
    PoissonProblemBuilder<T, RefCharge, RefInfoDesk, RefInitial, RefMesh>
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

impl<'a, T: Copy + RealField, GeometryDim, Conn, BandDim>
    PoissonProblemBuilder<
        T,
        &'a Charge<T, BandDim>,
        &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
        &'a Potential<T>,
        &'a Mesh<T, GeometryDim, Conn>,
    >
where
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    pub(crate) fn build(
        self,
    ) -> color_eyre::Result<PoissonProblem<'a, T, GeometryDim, Conn, BandDim>> {
        Ok(PoissonProblem {
            charge: self.charge,
            info_desk: self.info_desk,
            mesh: self.mesh,
            operator: self.build_operator()?,
            source: self.build_source()?,
            fermi_level: DVector::from(self.info_desk.determine_fermi_level(
                self.mesh,
                self.initial_potential,
                self.charge,
            )),
            initial_values: self.initial_potential.clone(),
        })
    }
}

pub(crate) struct PoissonProblem<'a, T, GeometryDim, Conn, BandDim>
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<T, GeometryDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    charge: &'a Charge<T, BandDim>,
    info_desk: &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
    mesh: &'a Mesh<T, GeometryDim, Conn>,
    // The differential operator \nabla \epsilon \nabla \phi
    operator: CsrMatrix<T>,
    // The static component of the system: the source term q * (N_D - N_A)
    source: DVector<T>,
    // Fermi level -> changes through the calculation
    fermi_level: DVector<T>,
    //
    initial_values: Potential<T>,
}

use super::Potential;
use nalgebra::{Const, Dynamic, Matrix, VecStorage};

impl<T, GeometryDim, Conn, BandDim> Operator for PoissonProblem<'_, T, GeometryDim, Conn, BandDim>
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<T, GeometryDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    type Param = DVector<T>;
    type Output = DVector<T>;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let p = p - DVector::from(vec![p[0]; p.len()]);
        // n * e
        let free_charge = self.info_desk.update_source_vector(
            self.mesh,
            &self.fermi_level,
            self.initial_values.as_ref(),
        );
        Ok(&self.operator * (p + self.initial_values.as_ref()) + &self.source + &free_charge)
    }
}

impl<T, GeometryDim, Conn, BandDim> Jacobian for PoissonProblem<'_, T, GeometryDim, Conn, BandDim>
where
    T: Copy + RealField,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, BandDim>
        + Allocator<T, GeometryDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
{
    type Param = DVector<T>;
    type Jacobian = DMatrix<T>;

    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
        let p = p - DVector::from(vec![p[0]; p.len()]);
        let jacobian_diagonal =
            self.info_desk
                .compute_jacobian_diagonal(&self.fermi_level, &p, self.mesh);
        let mut jacobian_csr = self.operator.diagonal_as_csr();
        for (val, &value) in jacobian_csr
            .values_mut()
            .iter_mut()
            .zip(jacobian_diagonal.iter())
        {
            *val = value;
        }
        let jacobian = &self.operator + jacobian_csr;
        Ok(nalgebra_sparse::convert::serial::convert_csr_dense(
            &jacobian,
        ))
    }
}
