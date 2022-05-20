// Copyright 2022 Chris Gubbin
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Outer Loop
//!
//! The outer loop solves the Poisson equation in the heterostructure
//! $$ \frac{\mathrm{d}}{\mathrm{d} z} \left[ \epsilon\left(z\right) \frac{\mathrm{d} \phi}{\mathrm{d} z} + q \left[N_D^+ - N_A^- - n \right]$$
//! which gives the electrostatic potential, and the Schrödinger equation which yields the
//! carrier density.

mod convergence;
mod methods;
mod poisson;

pub(crate) use convergence::Convergence;
pub(crate) use methods::{Outer, Potential};

use crate::{
    app::{Calculation, Tracker},
    device::info_desk::DeviceInfoDesk,
    error::{BuildError, CsrError},
    hamiltonian::{Hamiltonian, PotentialInfoDesk},
    postprocessor::{Charge, ChargeAndCurrent},
};
use miette::Diagnostic;
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use ndarray::Array1;
use std::fs::OpenOptions;
use std::io::Write;
use std::marker::PhantomData;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

///Error for the outer loop

#[derive(thiserror::Error, Debug, Diagnostic)]
pub(crate) enum OuterLoopError<T: RealField> {
    // An error in building an operator, or a self energy. These errors are non-recoverable
    #[error(transparent)]
    BuilderError(#[from] BuildError),
    // #[error(transparent)]
    // InnerLoopError,
    // An error in the fixed-point iteration convergence: this can be recoverable if it is an
    // out-of-iteration variant. Sometimes convergence is close and we should just advance
    #[error(transparent)]
    FixedPoint(#[from] conflux::core::FixedPointError<T>),
    // Errors from the Poisson equation convergence. These errors are probably non-recoverable
    #[error(transparent)]
    PoissonError(#[from] argmin::core::Error),
    // #[error(transparent)]
    // Stagnation,
    // Write errors in post-processing. Non-recoverable
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

/// Builder struct for the outer loop
pub(crate) struct OuterLoopBuilder<
    T,
    RefConvergenceSettings,
    RefMesh,
    RefSpectral,
    RefHamiltonian,
    RefTracker,
    RefInfoDesk,
> {
    mesh: RefMesh,
    spectral: RefSpectral,
    hamiltonian: RefHamiltonian,
    convergence_settings: RefConvergenceSettings,
    tracker: RefTracker,
    info_desk: RefInfoDesk,
    marker: PhantomData<T>,
}

impl<T> OuterLoopBuilder<T, (), (), (), (), (), ()> {
    /// Initialise an empty OuterLoopBuilder
    pub(crate) fn new() -> Self {
        Self {
            mesh: (),
            spectral: (),
            hamiltonian: (),
            convergence_settings: (),
            tracker: (),
            info_desk: (),
            marker: PhantomData,
        }
    }
}

impl<
        T: RealField,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
    >
    OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
    >
{
    /// Attach the problem's `Mesh`
    pub(crate) fn with_mesh<Mesh>(
        self,
        mesh: &Mesh,
    ) -> OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        &Mesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
    > {
        OuterLoopBuilder {
            mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            marker: PhantomData,
        }
    }

    /// Attach the `SpectralSpace` associated with the problem
    pub(crate) fn with_spectral_space<Spectral>(
        self,
        spectral: &Spectral,
    ) -> OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        &Spectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            marker: PhantomData,
        }
    }

    /// Attach the constructed `Hamiltonian` associated with the problem
    pub(crate) fn with_hamiltonian<Hamiltonian>(
        self,
        hamiltonian: &mut Hamiltonian,
    ) -> OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        &mut Hamiltonian,
        RefTracker,
        RefInfoDesk,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            marker: PhantomData,
        }
    }

    /// Attach convergence information for the inner and outer loop
    pub(crate) fn with_convergence_settings<ConvergenceSettings>(
        self,
        convergence_settings: &ConvergenceSettings,
    ) -> OuterLoopBuilder<
        T,
        &ConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            marker: PhantomData,
        }
    }

    /// Attach the global tracker
    pub(crate) fn with_tracker<Tracker>(
        self,
        tracker: &Tracker,
    ) -> OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        &Tracker,
        RefInfoDesk,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker,
            info_desk: self.info_desk,
            marker: PhantomData,
        }
    }

    /// Attach the info desk
    pub(crate) fn with_info_desk<InfoDesk>(
        self,
        info_desk: &InfoDesk,
    ) -> OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        &InfoDesk,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk,
            marker: PhantomData,
        }
    }
}

/// A structure holding the information to carry out the outer iteration
pub(crate) struct OuterLoop<'a, T, GeometryDim, Conn, BandDim, SpectralSpace>
where
    T: RealField + Copy,
    BandDim: SmallDim,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<Array1<T::RealField>, BandDim>
        + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>,
{
    /// The convergence information for the outerloop and the spawned innerloop
    convergence_settings: &'a Convergence<T>,
    /// The mesh associated with the problem
    mesh: &'a Mesh<T, GeometryDim, Conn>,
    /// The spectral mesh and integration weights associated with the problem
    spectral: &'a SpectralSpace,
    /// The Hamiltonian associated with the problem
    hamiltonian: &'a mut Hamiltonian<T>,
    // TODO A solution tracker, think about this IMPL. We already have a top-level tracker
    tracker: LoopTracker<T, BandDim>,
    info_desk: &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
    term: console::Term,
}

impl<'a, T, GeometryDim, Conn, BandDim, SpectralSpace>
    OuterLoopBuilder<
        T,
        &'a Convergence<T>,
        &'a Mesh<T, GeometryDim, Conn>,
        &'a SpectralSpace,
        &'a mut Hamiltonian<T>,
        &'a Tracker<'a, T, GeometryDim, BandDim>,
        &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
    >
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    BandDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<Array1<T>, BandDim>,
{
    /// Build out the OuterLoop -> Generic over the SpectralSpace so the OuterLoop can do both coherent and incoherent transport
    pub(crate) fn build(
        self,
        voltage: T,
    ) -> color_eyre::Result<OuterLoop<'a, T, GeometryDim, Conn, BandDim, SpectralSpace>> {
        let tracker = LoopTracker::from_global_tracker(self.tracker, voltage);
        Ok(OuterLoop {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            hamiltonian: self.hamiltonian,
            spectral: self.spectral,
            tracker,
            info_desk: self.info_desk,
            term: console::Term::stdout(),
        })
    }

    pub(crate) fn build_coherent(
        self,
        voltage: T,
    ) -> color_eyre::Result<OuterLoop<'a, T, GeometryDim, Conn, BandDim, SpectralSpace>> {
        let mut tracker = LoopTracker::from_global_tracker(self.tracker, voltage);
        tracker.calculation = Calculation::Coherent {
            voltage_target: voltage,
        };
        Ok(OuterLoop {
            convergence_settings: self.convergence_settings,
            mesh: self.mesh,
            hamiltonian: self.hamiltonian,
            spectral: self.spectral,
            tracker,
            info_desk: self.info_desk,
            term: console::Term::stdout(),
        })
    }
}

pub(crate) struct LoopTracker<T: RealField, BandDim: SmallDim>
where
    DefaultAllocator: Allocator<Array1<T>, BandDim>,
{
    charge_and_currents: ChargeAndCurrent<T, BandDim>,
    potential: Potential<T>,
    fermi_level: Array1<T>,
    iteration: usize,
    calculation: Calculation<T>,
    pub(crate) scattering_scaling: T,
    pub(crate) current_residual: T,
    voltage: T,
}

impl<T: Copy + RealField, BandDim: SmallDim> LoopTracker<T, BandDim>
where
    DefaultAllocator: Allocator<Array1<T>, BandDim>,
{
    pub(crate) fn from_global_tracker<GeometryDim: SmallDim>(
        global_tracker: &Tracker<'_, T, GeometryDim, BandDim>,
        voltage: T,
    ) -> Self
    where
        DefaultAllocator: Allocator<T, GeometryDim>
            + Allocator<T::RealField, BandDim>
            + Allocator<[T::RealField; 3], BandDim>,
    {
        // This is a dirty clone, it might be best to just mutably update the global tracker
        Self {
            potential: global_tracker.potential().clone(),
            charge_and_currents: ChargeAndCurrent::from_charge_and_current(
                global_tracker.charge().clone(),
                global_tracker.current().clone(),
            ),
            fermi_level: Array1::from(
                (0..global_tracker.num_vertices())
                    .map(|_| T::zero())
                    .collect::<Vec<_>>(),
            ),
            calculation: global_tracker.calculation(),
            iteration: 0,
            scattering_scaling: T::from_f64(1.0).unwrap(),
            current_residual: T::max_value().unwrap(),
            voltage,
        }
    }

    pub(crate) fn charge_and_currents_mut(&mut self) -> &mut ChargeAndCurrent<T, BandDim> {
        &mut self.charge_and_currents
    }

    pub(crate) fn charge_as_ref(&self) -> &Charge<T, BandDim> {
        self.charge_and_currents.charge_as_ref()
    }

    pub(crate) fn potential_mut(&mut self) -> &mut Potential<T> {
        &mut self.potential
    }

    pub(crate) fn fermi_level(&self) -> &Array1<T> {
        &self.fermi_level
    }

    pub(crate) fn update_potential(&mut self, potential: Potential<T>) {
        self.potential = potential;
    }

    pub(crate) fn fermi_level_mut(&mut self) -> &mut Array1<T> {
        &mut self.fermi_level
    }

    pub(crate) fn scattering_scaling(&self) -> T {
        self.scattering_scaling
    }

    pub(crate) fn write_to_file(&self, calculation: &str) -> Result<(), std::io::Error>
    where
        T: argmin::core::ArgminFloat,
    {
        // Write potential
        let mut file = std::fs::File::create(format!(
            "../results/{calculation}_potential_{}V_{}_scaling.txt",
            self.voltage, self.scattering_scaling
        ))?;
        for value in self.potential.as_ref().iter() {
            let value = value.to_f64().unwrap().to_string();
            writeln!(file, "{}", value)?;
        }

        // Write net charge
        let mut file = std::fs::File::create(format!(
            "../results/{calculation}_charge_{}V_{}_scaling.txt",
            self.voltage, self.scattering_scaling
        ))?;
        for value in self.charge_and_currents.charge_as_ref().net_charge().iter() {
            let value = value.to_f64().unwrap().to_string();
            writeln!(file, "{}", value)?;
        }

        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open(format!(
                "../results/current_{}_scaling.txt",
                self.scattering_scaling
            ))
            .unwrap();
        writeln!(
            file,
            "{}, {}",
            self.voltage,
            self.charge_and_currents.current_as_ref().net_current()[0]
        )
    }
}

impl<T: Copy + RealField, BandDim: SmallDim> PotentialInfoDesk<T> for LoopTracker<T, BandDim>
where
    DefaultAllocator: Allocator<Array1<T>, BandDim>,
{
    type BandDim = BandDim;
    fn potential(&self, vertex_index: usize) -> T {
        self.potential.get(vertex_index)
    }
}
