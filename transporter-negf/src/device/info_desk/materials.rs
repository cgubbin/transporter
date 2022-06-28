use super::LayerInfoDesk;
use nalgebra::{allocator::Allocator, DefaultAllocator, Point1, RealField, U1, U3};

#[derive(Debug, serde::Deserialize)]
#[non_exhaustive]
/// Enum with all implemented material types
///
/// As materials may be added in future this is labelled as `non_exhaustive`
pub(crate) enum Material {
    SiC,
    GaAs,
    AlGaAs,
}

impl std::fmt::Display for Material {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Material::SiC => {
                write!(f, "SiC")
            }
            Material::GaAs => {
                write!(f, "GaAs")
            }
            Material::AlGaAs => {
                write!(f, "AlGaAs")
            }
        }
    }
}

impl Material {
    /// Builds an instance of `LayerInfoDesk` for the given `Material` variant
    pub(crate) fn get_info<T: RealField>(&self) -> miette::Result<LayerInfoDesk<T, U1>>
    where
        DefaultAllocator: Allocator<T, U1> + Allocator<T, U1, U3>,
    {
        match self {
            Material::GaAs => Ok(LayerInfoDesk::gaas()),
            Material::AlGaAs => Ok(LayerInfoDesk::algaas()),
            Material::SiC => Ok(LayerInfoDesk::sic()),
        }
    }
}

impl<T: RealField> LayerInfoDesk<T, U1> {
    #[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
    fn gaas() -> Self {
        Self {
            effective_mass: nalgebra::Vector1::new([0.067, 0.067, 0.067]),
            band_offset: Point1::new(0.0),
            dielectric_constant: [11.5, 11.5, 11.5],
        }
    }

    #[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
    fn algaas() -> Self {
        Self {
            effective_mass: nalgebra::Vector1::new([0.066669, 0.066669, 0.066669]),
            band_offset: Point1::new(0.3),
            dielectric_constant: [11.5, 11.5, 11.5],
        }
    }

    #[numeric_literals::replace_float_literals(T::from_f64(literal).unwrap())]
    fn sic() -> Self {
        Self {
            effective_mass: nalgebra::Vector1::new([0.25, 0.25, 0.25]),
            band_offset: Point1::new(0.0),
            dielectric_constant: [8.5, 8.5, 8.5],
        }
    }
}
