use miette::Diagnostic;

#[derive(thiserror::Error, Debug, Diagnostic)]
pub enum BuildError {
    #[error(transparent)]
    Csr(#[from] CsrError),
    #[error("{0}")]
    Mesh(String),
    #[error("{0}")]
    MissizedAllocator(String),
}

#[derive(thiserror::Error, Debug, Diagnostic)]
/// General error for Csr construction, patterns and element access problems
pub enum CsrError {
    #[error("{0}")]
    Access(String),
    #[error(transparent)]
    Pattern(#[from] nalgebra_sparse::pattern::SparsityPatternFormatError),
    #[error(transparent)]
    Construction(#[from] nalgebra_sparse::SparseFormatError),
}

#[derive(thiserror::Error, Debug, Diagnostic)]
/// Error for IO events
pub enum IOError {
    #[error("IO Failue: {0}")]
    IO(#[from] std::io::Error),
}
