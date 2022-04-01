#[derive(Debug, serde::Deserialize)]
#[non_exhaustive]
/// Enum with all implemented material types
pub(crate) enum Material {
    SiC,
    GaAs,
}

impl std::fmt::Display for Material {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Material::SiC => {
                write!(f, "SiC")
            }
            Material::GaAs => {
                write!(f, "GaAs")
            } // _ => {
              //     write!(f, "Display unimplemented")
              // }
        }
    }
}
