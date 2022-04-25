use super::{Calculation, Configuration, Tracker};
use crate::outer_loop::{Outer, OuterLoopError, Potential};
use crate::spectral::WavevectorSpace;
use argmin::core::ArgminFloat;
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dynamic, Matrix, RealField, VecStorage, U1,
};
use transporter_mesher::{Mesh, Segment1dConnectivity, SmallDim};

pub(crate) fn coherent_calculation_at_fixed_voltage<T, BandDim: SmallDim>(
    voltage: T,
    initial_potential: Potential<T>,
    config: &Configuration<T>,
    mesh: &Mesh<T, U1, Segment1dConnectivity>,
    tracker: &Tracker<'_, T, U1, BandDim>,
) -> Result<Potential<T>, OuterLoopError<T>>
where
    T: ArgminFloat + Copy + num_traits::NumCast + RealField + ndarray::ScalarOperand,
    DefaultAllocator: Allocator<T, U1>
        + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
    <DefaultAllocator as Allocator<T, U1>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
{
    // if all the masses are equal we do not need to discretise wavevectors for a coherent calculation
    let first = tracker.info_desk.effective_masses[0].clone();
    match tracker
        .info_desk
        .effective_masses
        .iter()
        .all(|item| item.clone() == first)
    {
        true => coherent_calculation_at_fixed_voltage_with_constant_mass(
            voltage,
            initial_potential,
            config,
            mesh,
            tracker,
        ),
        false => coherent_calculation_at_fixed_voltage_with_changing_mass(
            voltage,
            initial_potential,
            config,
            mesh,
            tracker,
        ),
    }
}

fn coherent_calculation_at_fixed_voltage_with_constant_mass<T, BandDim: SmallDim>(
    voltage: T,
    initial_potential: Potential<T>,
    config: &Configuration<T>,
    mesh: &Mesh<T, U1, Segment1dConnectivity>,
    tracker: &Tracker<'_, T, U1, BandDim>,
) -> Result<Potential<T>, OuterLoopError<T>>
where
    T: ArgminFloat + Copy + num_traits::NumCast + RealField + ndarray::ScalarOperand,
    DefaultAllocator: Allocator<T, U1>
        + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
    <DefaultAllocator as Allocator<T, U1>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
{
    // Build calculation independent structures
    let mut hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
        .with_mesh(mesh)
        .with_info_desk(tracker)
        .build()?;

    let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
        .with_number_of_energy_points(config.spectral.number_of_energy_points)
        .with_energy_range(std::ops::Range {
            start: config.spectral.minimum_energy,
            end: config.spectral.maximum_energy,
        })
        .with_energy_integration_method(config.spectral.energy_integration_rule);

    let spectral_space = spectral_space_builder.build_coherent();
    let outer_config = crate::outer_loop::Convergence {
        outer_tolerance: config.outer_loop.tolerance,
        maximum_outer_iterations: config.outer_loop.maximum_iterations,
        inner_tolerance: config.inner_loop.tolerance,
        maximum_inner_iterations: config.inner_loop.maximum_iterations,
        calculation_type: Calculation::Coherent,
    };
    let mut outer_loop: crate::outer_loop::OuterLoop<
        T,
        U1,
        Segment1dConnectivity,
        BandDim,
        crate::spectral::SpectralSpace<T::RealField, ()>,
    > = crate::outer_loop::OuterLoopBuilder::new()
        .with_mesh(mesh)
        .with_hamiltonian(&mut hamiltonian)
        .with_spectral_space(&spectral_space)
        .with_convergence_settings(&outer_config)
        .with_tracker(tracker)
        .with_info_desk(tracker.info_desk)
        .build(voltage)
        .unwrap();

    outer_loop.run_loop(initial_potential)?;
    Ok(outer_loop.potential_owned())
}

fn coherent_calculation_at_fixed_voltage_with_changing_mass<T, BandDim: SmallDim>(
    voltage: T,
    initial_potential: Potential<T>,
    config: &Configuration<T>,
    mesh: &Mesh<T, U1, Segment1dConnectivity>,
    tracker: &Tracker<'_, T, U1, BandDim>,
) -> Result<Potential<T>, OuterLoopError<T>>
where
    T: ArgminFloat + Copy + num_traits::NumCast + RealField + ndarray::ScalarOperand,
    DefaultAllocator: Allocator<T, U1>
        + Allocator<T, BandDim>
        + Allocator<[T; 3], BandDim>
        + Allocator<
            Matrix<T, Dynamic, Const<1_usize>, VecStorage<T, Dynamic, Const<1_usize>>>,
            BandDim,
        >,
    <DefaultAllocator as Allocator<T, U1>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<T, BandDim>>::Buffer: Send + Sync,
    <DefaultAllocator as Allocator<[T; 3], BandDim>>::Buffer: Send + Sync,
{
    // Build calculation independent structures
    let mut hamiltonian = crate::hamiltonian::HamiltonianBuilder::new()
        .with_mesh(mesh)
        .with_info_desk(tracker)
        .build()?;

    let spectral_space_builder = crate::spectral::constructors::SpectralSpaceBuilder::new()
        .with_number_of_energy_points(config.spectral.number_of_energy_points)
        .with_energy_range(std::ops::Range {
            start: config.spectral.minimum_energy,
            end: config.spectral.maximum_energy,
        })
        .with_energy_integration_method(config.spectral.energy_integration_rule)
        .with_maximum_wavevector(config.spectral.maximum_wavevector)
        .with_number_of_wavevector_points(config.spectral.number_of_wavevector_points)
        .with_wavevector_integration_method(config.spectral.wavevector_integration_rule)
        .with_mesh(mesh);
    let spectral_space = spectral_space_builder.build_incoherent();

    let outer_config = crate::outer_loop::Convergence {
        outer_tolerance: config.outer_loop.tolerance,
        maximum_outer_iterations: config.outer_loop.maximum_iterations,
        inner_tolerance: config.inner_loop.tolerance,
        maximum_inner_iterations: config.inner_loop.maximum_iterations,
        calculation_type: Calculation::Coherent,
    };
    let mut outer_loop: crate::outer_loop::OuterLoop<
        T,
        U1,
        Segment1dConnectivity,
        BandDim,
        crate::spectral::SpectralSpace<T::RealField, WavevectorSpace<T, U1, Segment1dConnectivity>>,
    > = crate::outer_loop::OuterLoopBuilder::new()
        .with_mesh(mesh)
        .with_hamiltonian(&mut hamiltonian)
        .with_spectral_space(&spectral_space)
        .with_convergence_settings(&outer_config)
        .with_tracker(tracker)
        .with_info_desk(tracker.info_desk)
        .build(voltage)
        .unwrap();

    outer_loop.run_loop(initial_potential)?;
    Ok(outer_loop.potential_owned())
}
