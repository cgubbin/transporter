#[cfg(feature = "cli")]
use transporter_negf::app::run;

#[cfg(feature = "tui")]
use transporter_negf::app::tui::run;

fn main() {
    #[cfg(feature = "tui")]
    run().unwrap();
}
