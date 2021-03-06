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
//!
//! Utilising an initial guess for the electrostatic potential, an [Inner Loop](crate::inner_loop) is spawned. The inner loop calculates the
//! charge density in the device under the constraints laid out in the configuration file. The outer loop then calculates the electrostatic potential
//! using the [Poisson](crate::outer_loop::poisson) module. When the Poisson residual falls below a preset value the outer loop terminates.
//!
//! To accelerate convergence the outer loop mixes the potential update at each iteration using an Anderson acceleration scheme as described in the
//! [Conflux](conflux) crate. This increases the robustness of the outer iteration procedure, reducing the likelihood the calculation will diverge from
//! the true solution.

/// Convergence information for the inner and outer loops
mod convergence;

/// The methods which run the outer loop to convergence
mod methods;

/// Solvers and constructors for the Poisson equation
mod poisson;

pub(crate) use convergence::Convergence;
pub(crate) use methods::{Outer, Potential};

use crate::{
    app::{tui::Progress, Calculation, Tracker},
    device::info_desk::DeviceInfoDesk,
    error::{BuildError, CsrError},
    hamiltonian::{Hamiltonian, PotentialInfoDesk},
    postprocessor::{Charge, ChargeAndCurrent, Current},
};
use miette::Diagnostic;
use nalgebra::{allocator::Allocator, DefaultAllocator, RealField};
use ndarray::Array1;
use std::fs::OpenOptions;
use std::io::Write;
use std::marker::PhantomData;
use tokio::sync::mpsc::Sender;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

/// Error alias for the outer loop

#[derive(thiserror::Error, Debug, Diagnostic)]
pub(crate) enum OuterLoopError<T: RealField> {
    /// An error in building an operator, or a self energy. These errors are non-recoverable and lead to termination
    #[error(transparent)]
    BuilderError(#[from] BuildError),
    /// An error in the fixed-point iteration convergence: this can be recoverable if it is an
    /// out-of-iteration variant and convergence is close, or the voltage step can be reduced
    #[error(transparent)]
    FixedPoint(#[from] conflux::core::FixedPointError<T>),
    /// Errors from the Poisson equation convergence. These errors are probably non-recoverable and indicate that the
    /// calculation has diverged from the true solution. This indicates a problem with the structure provided
    #[error(transparent)]
    PoissonError(#[from] argmin::core::Error),
    /// An additional error variant for fixed point iterations. Before `max_iterations` is reached the calculation
    /// stagnates if it cannot advance beyond the current residual after `n` iterations. This indicates the voltage step
    /// is too large
    #[error("The fixed point iteration has stagnated at residual {0}")]
    Stagnation(T),
    /// Errors from linear algebra failure. These indicate something has gone wrong with the core logic: perhaps matrices
    /// have incompatible dimensions for linear algebra. It could also indicate failure in matrix inversion when calculating
    /// the retarded Green's function. These are non-recoverable errors.
    #[error(transparent)]
    LinAlg(#[from] ndarray_linalg::error::LinalgError),
    // Errors bubbled up from the inner loop
    #[error(transparent)]
    Inner(#[from] crate::inner_loop::InnerLoopError),
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
    RefProgress,
    RefProgressSender,
> {
    mesh: RefMesh,
    spectral: RefSpectral,
    hamiltonian: RefHamiltonian,
    convergence_settings: RefConvergenceSettings,
    tracker: RefTracker,
    info_desk: RefInfoDesk,
    progress: RefProgress,
    mpsc_sender: RefProgressSender,
    marker: PhantomData<T>,
}

impl<T> OuterLoopBuilder<T, (), (), (), (), (), (), (), ()> {
    /// Initialise an empty OuterLoopBuilder
    pub(crate) fn new() -> Self {
        Self {
            mesh: (),
            spectral: (),
            hamiltonian: (),
            convergence_settings: (),
            tracker: (),
            info_desk: (),
            progress: (),
            mpsc_sender: (),
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
        RefProgress,
        RefProgressSender,
    >
    OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
        RefProgress,
        RefProgressSender,
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
        RefProgress,
        RefProgressSender,
    > {
        OuterLoopBuilder {
            mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
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
        RefProgress,
        RefProgressSender,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
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
        RefProgress,
        RefProgressSender,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
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
        RefProgress,
        RefProgressSender,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
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
        RefProgress,
        RefProgressSender,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker,
            info_desk: self.info_desk,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
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
        RefProgress,
        RefProgressSender,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk,
            progress: self.progress,
            mpsc_sender: self.mpsc_sender,
            marker: PhantomData,
        }
    }

    /// Attach the progress report
    pub(crate) fn with_progress<Progress>(
        self,
        progress: &Progress,
    ) -> OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
        &Progress,
        RefProgressSender,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            progress,
            mpsc_sender: self.mpsc_sender,
            marker: PhantomData,
        }
    }

    /// Attach the sender
    pub(crate) fn with_sender<Sender>(
        self,
        mpsc_sender: &Sender,
    ) -> OuterLoopBuilder<
        T,
        RefConvergenceSettings,
        RefMesh,
        RefSpectral,
        RefHamiltonian,
        RefTracker,
        RefInfoDesk,
        RefProgress,
        &Sender,
    > {
        OuterLoopBuilder {
            mesh: self.mesh,
            spectral: self.spectral,
            hamiltonian: self.hamiltonian,
            convergence_settings: self.convergence_settings,
            tracker: self.tracker,
            info_desk: self.info_desk,
            progress: self.progress,
            mpsc_sender,
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
    pub(crate) tracker: LoopTracker<T, BandDim>,
    info_desk: &'a DeviceInfoDesk<T, GeometryDim, BandDim>,
    progress: Progress<T>,
    mpsc_sender: Sender<Progress<T>>,
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
        &'a Progress<T>,
        &'a Sender<Progress<T>>,
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
            progress: self.progress.clone(),
            mpsc_sender: self.mpsc_sender.clone(),
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
            progress: self.progress.clone(),
            mpsc_sender: self.mpsc_sender.clone(),
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
    rate: Option<crate::postprocessor::ScatteringRate<T>>,
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
            rate: None,
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

    pub(crate) fn current_as_ref(&self) -> &Current<T, BandDim> {
        self.charge_and_currents.current_as_ref()
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

    pub(crate) fn update_rate(&mut self, rate: crate::postprocessor::ScatteringRate<T>) {
        self.rate = Some(rate);
    }

    pub(crate) fn fermi_level_mut(&mut self) -> &mut Array1<T> {
        &mut self.fermi_level
    }

    pub(crate) fn scattering_scaling(&self) -> T {
        self.scattering_scaling
    }

    pub(crate) fn potential_as_ref(&self) -> &Array1<T> {
        self.potential.as_ref()
    }

    pub(crate) fn rate(&self) -> Option<crate::postprocessor::ScatteringRate<T>> {
        self.rate.clone()
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
