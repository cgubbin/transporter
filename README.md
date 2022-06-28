# Transporter

Transporter finds the self-consistent electron distribution in 1D heterostructures using the non-equilibrium Green's function algorithm. 

## Installation of Prerequisites

To use transporter you need to install the [Rust toolchain](https://www.rust-lang.org/tools/install). To do this run

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

and follow the onscreen instructions.

It is also currently necessary to install the intel-MKL oneAPI backend to compile the codebase: this is currently necessary to facilitate inversion of dense matrices. The API can be installed by visiting [https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) and following the onscreen instructions.

## Compilation

The application can be compiled using the nightly channel of Rust. Set this as a default after installation of the Rust toolchain by running

```bash
rustup default nightly
```

After this the binary can be compiled in release mode by running

```bash
cargo build --release
```

Note that compilation will fail if the Intel oneAPI backend is not available in the system path.

The (incomplete) documentation can be built by running

```bash
cargo doc --document-private-items
```

and opened by visiting `file:///path_to_repo/target/doc/transporter_negf/index.html` where `path_to_repo` is the on-disk location of the top level folder in the repository.

## Running the Software

The application can be run from the top level directory of the repository by running

```bash
./target/release/transporter-negf
```

Note that at the moment the application is somewhat inflexible and will fail to launch under the following conditions:

* If the Intel oneAPI bindings are not available in the path.
* If the terminal window launching the App is too small (the window must contain at least 28 vertical lines in order to render).
* If run from a different directory (the App looks for structures in the `structures` subdirectory, running it from somewhere this subdirectory is unavailable results in a panic).

When the Application has been launched a structure can be selected from the list using the arrow keys, and the self-consistent electronic distribution calculated by pressing `Enter`.  The target voltage can be altered by using the `k` (increase) or `j` (decrease) keys.

## Defining Structures

Structures are defined declaratively in `.toml` files in the `structures` subdirectory

```toml
lead_length = 5.0
temperature = 300.0
voltage_offsets = [0.0, 0.00]

[[layers]]
thickness = [10.0]
material = "GaAs"
acceptor_density = 0.0
donor_density = 1e23

[[layers]]
thickness = [10.0]
material = "AlGaAs"
acceptor_density = 0.0
donor_density = 1e20

...
```

The `lead_length` is an optional parameter which allows the device to be partioned into a coherent component of length `lead_length` on each end, and an incoherent part formed from the device core region. This reduces the computational load at the expense of accuracy. Layers can appended to the `.toml` file until the whole device is parameterised.

## Configuration

If convergence is poor it is probably necessary to tune the numerical parameters, particularly those used to define the energy and wavevector grids. The calculation configuration is defined in the `./config` sub-directory in the file `default.toml`:

```toml
[global]
number_of_bands = 1
security_checks = true
voltage_step = 0.01

[inner_loop]
maximum_iterations = 20
tolerance = 1e-3

[outer_loop]
maximum_iterations = 50
tolerance = 1e-3

[mesh]
unit_size = 1e-9
elements_per_unit = 4
maximum_growth_rate = 1.2

[spectral]
number_of_energy_points = 500
minimum_energy = 0.0
maximum_energy = 0.3
energy_integration_rule = "ThreePoint"
number_of_wavevector_points = 100
maximum_wavevector = 1e9
wavevector_integration_rule = "ThreePoint"
```

### Global

* `number_of_bands`: At the moment we can only solve for 1 band.
* `security_checks`: Whether to run numerical security checks. This involves checking for particle conservation using the lesser and greater self-energies and Green's functions at each stage of the calculation. It increases the calculation load significantly. Generally it is good to enable this for debugging purposes but not for running real calculations.
* `voltage_step`: The resolution of voltage points in a finite voltage calculation (a sweep is performed with spacing `voltage_step`).

### Inner / Outer Loop

* `maximum_iterations`: The maximum allowed iteration count.
* `tolerance`: The tolerance to be achieved for the loop to be considered converged.

### Mesh

* `unit_size`: The unit of measurement in the `structure` file.
* `elements_per_unit`: The number of mesh elements per `unit_size`.
* `maximum_growth_rate`: The maximum allowable factor by which element size can change between adjacent elements.

### Spectral

* `number_of_energy_points`: The number of points in the energy grid
* `minimum_energy`: The minimum energy in the grid
* `maximum_energy`: The maximum energy in the grid
* `energy_integration_rule`: The integration method to use for energy ("ThreePoint", "Trapezium", "Romberg")
* `number_of_wavevector_points`: The number of wavevector points to be used in the grid
* `maximum_wavevector`: The maximum wavevector in the grid
* `wavevector_integration_rule`: The integration method to use for wavevector ("ThreePoint", "Trapezium", "Romberg")