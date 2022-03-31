use crate::greens_functions::GreensFunctionMethods;
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator};
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

pub(crate) struct SelfEnergy<T, GeometryDim, Conn, Matrix>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T::RealField, GeometryDim>,
    Matrix: GreensFunctionMethods<T>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    ma: PhantomData<GeometryDim>,
    mc: PhantomData<Conn>,
    marker: PhantomData<T>,
    mat: PhantomData<Matrix>,
}

pub(crate) struct SelfEnergyBuilder<T, RefSpectral, RefMesh> {
    spectral: RefSpectral,
    mesh: RefMesh,
    marker: PhantomData<T>,
}

impl<T: ComplexField> SelfEnergyBuilder<T, (), ()> {
    pub(crate) fn new() -> Self {
        Self {
            spectral: (),
            mesh: (),
            marker: PhantomData,
        }
    }
}

impl<T, RefSpectral, RefMesh> SelfEnergyBuilder<T, RefSpectral, RefMesh> {
    pub(crate) fn with_spectral_discretisation<Spectral>(
        self,
        spectral: &Spectral,
    ) -> SelfEnergyBuilder<T, &Spectral, RefMesh> {
        SelfEnergyBuilder {
            spectral,
            mesh: self.mesh,
            marker: PhantomData,
        }
    }

    pub(crate) fn with_mesh<Mesh>(self, mesh: &Mesh) -> SelfEnergyBuilder<T, RefSpectral, &Mesh> {
        SelfEnergyBuilder {
            spectral: self.spectral,
            mesh,
            marker: PhantomData,
        }
    }
}

impl<'a, T, GeometryDim, Conn, SpectralSpace>
    SelfEnergyBuilder<T, &'a SpectralSpace, &'a Mesh<T::RealField, GeometryDim, Conn>>
where
    T: ComplexField + Copy,
    <T as ComplexField>::RealField: Copy,
    GeometryDim: SmallDim,
    SpectralSpace: crate::spectral::SpectralDiscretisation<T::RealField>,
    Conn: Connectivity<T::RealField, GeometryDim>,
    DefaultAllocator: Allocator<T::RealField, GeometryDim>,
{
    pub(crate) fn build<Matrix>(self) -> SelfEnergy<T, GeometryDim, Conn, Matrix>
    where
        Matrix: GreensFunctionMethods<T>,
    {
        SelfEnergy {
            ma: PhantomData,
            mc: PhantomData,
            marker: PhantomData,
            mat: PhantomData,
        }
    }
}
