// Copyright 2022 Chris Gubbin
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Error
//! The error type for the binary

use miette::Diagnostic;
use nalgebra::RealField;

#[derive(thiserror::Error, Debug, Diagnostic)]
pub(crate) enum TransporterError<T: RealField + Send + Sync> {
    #[error(transparent)]
    #[diagnostic(code(my_lib::io_error))]
    IoError(#[from] std::io::Error),
    #[error(transparent)]
    #[diagnostic(code(my_lib::io_error))]
    ConfigError(#[from] anyhow::Error),
    #[error(transparent)]
    OuterLoop(#[from] crate::outer_loop::OuterLoopError<T>),
    #[error(transparent)]
    Build(#[from] crate::error::BuildError),
}
