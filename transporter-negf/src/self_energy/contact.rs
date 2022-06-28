use super::{SelfEnergy, SelfEnergyError};
use crate::{
    hamiltonian::{AccessMethods, Hamiltonian},
    spectral::SpectralDiscretisation,
};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, RealField};
use ndarray::Array2;
use num_complex::Complex;
use transporter_mesher::{Connectivity, Mesh, SmallDim};

impl<T, GeometryDim, Conn> SelfEnergy<T, GeometryDim, Conn>
where
    T: RealField + Copy,
    GeometryDim: SmallDim,
    Conn: Connectivity<T, GeometryDim>,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    // Updates the coherent Self Energy at the contacts into the scratch matrix held in `self`
    pub(crate) fn recalculate_contact_self_energy<Spectral>(
        &mut self,
        mesh: &Mesh<T, GeometryDim, Conn>,
        hamiltonian: &Hamiltonian<T>,
        spectral_space: &Spectral,
    ) -> Result<(), SelfEnergyError>
    where
        Spectral: SpectralDiscretisation<T>,
    {
        // let term = console::Term::stdout();
        // if self.incoherent_lesser.is_none() {
        //     term.move_cursor_to(0, 5).unwrap();
        // } else {
        //     term.move_cursor_to(0, 7).unwrap();
        // }
        // term.clear_to_end_of_screen().unwrap();
        tracing::info!("Updating self-energies");
        match GeometryDim::dim() {
            1 => {
                // let term = Term::stdout();
                let n_vertices = mesh.vertices().len();

                let imaginary_unit = Complex::new(T::zero(), T::one());

                // // Display
                // let spinner_style = ProgressStyle::default_spinner()
                //     .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
                //     .template(
                //     "{prefix:.bold.dim} {spinner} {msg} [{wide_bar:.cyan/blue}] {percent}% ({eta})",
                // );
                // let pb = ProgressBar::with_draw_target(
                //     (spectral_space.number_of_energy_points()
                //         * spectral_space.number_of_wavevector_points()) as u64,
                //     ProgressDrawTarget::term(term, 60),
                // );
                // pb.set_style(spinner_style);

                for (idx, wavevector) in spectral_space.iter_wavevectors().enumerate() {
                    let hamiltonian_matrix = hamiltonian.calculate_total(wavevector);
                    for (jdx, energy) in spectral_space.iter_energies().enumerate() {
                        // pb.set_message(format!(
                        //     "Wavevector: {:.1}, Energy {:.5}eV",
                        //     wavevector, energy
                        // ));
                        // pb.set_position(
                        //     (idx * spectral_space.number_of_energy_points() + jdx) as u64,
                        // );
                        for ([boundary_element, diagonal_element], ind) in [
                            (hamiltonian_matrix.get_elements_at_source(), 0),
                            (hamiltonian_matrix.get_elements_at_drain(), n_vertices - 1),
                        ]
                        .into_iter()
                        {
                            let d = diagonal_element; // The hamiltonian is minus itself. Dumbo
                            let t = -boundary_element;
                            let z = Complex::from((d - energy) / (t + t));
                            if ind == 0 {
                                self.contact_retarded
                                    [idx * spectral_space.number_of_energy_points() + jdx]
                                    .data_mut()[0] =
                                    -Complex::from(t) * (imaginary_unit * z.acos()).exp();
                            } else {
                                self.contact_retarded
                                    [idx * spectral_space.number_of_energy_points() + jdx]
                                    .data_mut()[1] =
                                    -Complex::from(t) * (imaginary_unit * z.acos()).exp();
                            }
                        }
                    }
                }
                Ok(())
            }
            _ => unimplemented!("No self-energy implementation for 2D geometries"),
        }
    }

    pub(crate) fn contact_dense_retarded_self_energy(
        mesh: &Mesh<T, GeometryDim, Conn>,
        hamiltonian: &Hamiltonian<T>,
        energy: T,
        wavevector: T,
    ) -> Array2<Complex<T>> {
        let hamiltonian_matrix = hamiltonian.calculate_total(wavevector);
        let n_vertices = mesh.vertices().len();
        let imaginary_unit = Complex::new(T::zero(), T::one());
        let mut se = Array2::zeros((n_vertices, n_vertices));
        for ([boundary_element, diagonal_element], ind) in [
            (hamiltonian_matrix.get_elements_at_source(), 0),
            (hamiltonian_matrix.get_elements_at_drain(), n_vertices - 1),
        ]
        .into_iter()
        {
            let d = diagonal_element; // The hamiltonian is minus itself. Dumbo
            let t = -boundary_element;
            let z = Complex::from((d - energy) / (t + t));
            if ind == 0 {
                se[(0, 0)] = -Complex::from(t) * (imaginary_unit * z.acos()).exp();
            } else {
                se[(n_vertices - 1, n_vertices - 1)] =
                    -Complex::from(t) * (imaginary_unit * z.acos()).exp();
            }
        }
        se
    }
}
