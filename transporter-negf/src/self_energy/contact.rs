use super::SelfEnergy;
use crate::{hamiltonian::Hamiltonian, spectral::SpectralDiscretisation};
use console::Term;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, RealField};
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
    ) -> color_eyre::Result<()>
    where
        Spectral: SpectralDiscretisation<T>,
    {
        tracing::info!("Updating self-energies");
        match GeometryDim::dim() {
            1 => {
                let term = Term::stdout();
                let n_elements = mesh.elements().len();

                let imaginary_unit = Complex::new(T::zero(), T::one());

                // Display
                let spinner_style = ProgressStyle::default_spinner()
                    .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
                    .template(
                    "{prefix:.bold.dim} {spinner} {msg} [{wide_bar:.cyan/blue}] {percent}% ({eta})",
                );
                let pb = ProgressBar::with_draw_target(
                    (spectral_space.number_of_energy_points()
                        * spectral_space.number_of_wavevector_points()) as u64,
                    ProgressDrawTarget::term(term, 60),
                );
                pb.set_style(spinner_style);

                for (idx, wavevector) in spectral_space.iter_wavevectors().enumerate() {
                    let hamiltonian_matrix = hamiltonian.calculate_total(wavevector);
                    for (jdx, energy) in spectral_space.iter_energies().enumerate() {
                        pb.set_message(format!(
                            "Wavevector: {:.1}, Energy {:.5}eV",
                            wavevector, energy
                        ));
                        pb.set_position(
                            (idx * spectral_space.number_of_energy_points() + jdx) as u64,
                        );
                        for (boundary_element, diagonal_element, ind) in [
                            (
                                hamiltonian_matrix.values()[1],
                                hamiltonian_matrix.values()[0],
                                0,
                            ),
                            (
                                hamiltonian_matrix.values()[hamiltonian_matrix.values().len() - 2],
                                hamiltonian_matrix.values()[hamiltonian_matrix.values().len() - 1],
                                n_elements - 1,
                            ),
                        ]
                        .into_iter()
                        {
                            let d = diagonal_element; // The hamiltonian is minus itself. Dumbo
                            let t = -boundary_element;
                            let z = Complex::from((d - energy) / (t + t));
                            if ind == 0 {
                                self.contact_retarded
                                    [idx * spectral_space.number_of_energy_points() + jdx]
                                    .values_mut()[0] =
                                    -Complex::from(t) * (imaginary_unit * z.acos()).exp();
                            } else {
                                self.contact_retarded
                                    [idx * spectral_space.number_of_energy_points() + jdx]
                                    .values_mut()[1] =
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
}
