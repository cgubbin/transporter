mod app;
mod inputs;
mod io;
mod rayon_async;

use crate::{
    app::{run_simulation, Calculation},
    error::IOError,
};
pub(crate) use app::state::Progress;
use app::ui;
pub(crate) use app::App;
use app::{state::AppState, AppReturn};
use inputs::events::Events;
use inputs::Event;
use io::{handler::IoAsyncHandler, IoEvent};
use nalgebra::RealField;
use ndarray::{Array1, Array2};
use rayon_async::async_threadpool::AsyncThreadPool;
use std::io::Write;
use std::{sync::Arc, time::Duration};
use tokio::sync::mpsc::channel;
use tokio::sync::Mutex;
use tui::{backend::CrosstermBackend, widgets::ListState, Terminal};

/// A structure to hold the result of a full computation.
///
/// Using this high-level abstraction the running calculation is able to communicate
/// the converged results of the simulation back to the master thread, which is able
/// to handle file writing IO
#[derive(Debug)]
pub struct NEGFResult<T: Copy + RealField> {
    /// The calculation type
    pub(crate) calculation: Calculation<T>,
    /// The current calculated between the contacts
    pub(crate) current: T,
    /// The converged electronic potential
    pub(crate) potential: Array1<T>,
    /// The converged charge density
    pub(crate) electron_density: Array1<T>,
    /// The converged momentum-resolved LO phonon scattering rate
    pub(crate) scattering_rates: Option<Array2<T>>,
}

#[tokio::main]
/// Run the TUI app and start the GUI
pub async fn run() -> Result<(), IOError> {
    let (sync_io_tx, mut sync_io_rx) = channel::<IoEvent>(100);

    let app: Arc<Mutex<App<f64>>> = Arc::new(Mutex::new(App::new(sync_io_tx.clone())));
    let app_ui = Arc::clone(&app);

    // Configure logger
    tui_logger::init_logger(log::LevelFilter::Trace).unwrap();
    tui_logger::set_default_level(log::LevelFilter::Trace);

    // Handle IO in a specific thread
    tokio::spawn(async move {
        let mut handler = IoAsyncHandler::new(app);
        while let Some(io_event) = sync_io_rx.recv().await {
            handler.handle_io_event(io_event).await;
        }
    });

    start_ui(&app_ui).await?;

    Ok(())
}

/// Start the gui and run the internal feedback loop
pub async fn start_ui(app: &Arc<Mutex<App<f64>>>) -> Result<(), IOError> {
    // Configure Crossterm for tui
    let stdout = std::io::stdout();
    crossterm::terminal::enable_raw_mode()?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    terminal.hide_cursor()?;

    // The tick rate -> this defines the rate at which the inner rendering and input loop updates
    let tick_rate = Duration::from_millis(200);
    let mut events = Events::new(tick_rate)?;

    // trigger a state change from Init to Initialized
    {
        let mut app = app.lock().await;
        // Assume the first load is a long task
        app.dispatch(IoEvent::Initialize).await;
    }

    // An interactive widget to choose an input file in the directory
    let mut files_list_state = ListState::default();
    files_list_state.select(Some(0));

    // A multiple producer, single consumer channel to handle keyboard input
    let (action_sender, mut action_receiver) = tokio::sync::mpsc::channel::<AppReturn>(20);
    // A multiple producer, single consumer channel to monitor the status of the running calculation
    let (progress_sender, mut progress_receiver) = tokio::sync::mpsc::channel::<Progress<f64>>(20);

    // A multiple producer, single consumer channel to record converged results
    let (result_sender, mut result_receiver) = tokio::sync::mpsc::channel::<NEGFResult<f64>>(20);

    // A one-shot channel to track progress from the rayon pool
    let (rayon_sender, mut rayon_receiver) = tokio::sync::mpsc::channel::<miette::Result<()>>(1);

    // A threadpool for CPU-intensive simulations
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();

    loop {
        let mut local_app = app.lock().await;
        // Render the app at the current state
        terminal.draw(|rect| ui::draw(rect, &local_app, &mut files_list_state).unwrap())?;

        // Handle keyboard input
        match events.next().await {
            Event::Tick => local_app.update_on_tick(&action_sender).await,
            Event::Input(key) => {
                local_app
                    .do_action(key, &mut files_list_state, &action_sender)
                    .await
            }
        }

        // Listen on the `action_receiver` channel, to see if a new `AppReturn` has been recorded
        if let Some(result) = action_receiver.recv().await {
            // If the `AppReturn` is `Exit` break the loop and close the program
            if result == AppReturn::Exit {
                events.close();
                break;
            }
            // If the `AppReturn` is `Run` begin the simulation from the selected file
            if result == AppReturn::Run {
                if let Some(selected) = files_list_state.selected() {
                    let ui_progress_sender = progress_sender.clone();
                    let ui_result_sender = result_sender.clone();
                    let file_in_directory = local_app.file_in_directory(selected).clone();
                    let calculation_to_run = *local_app.calculation_type();
                    let rayon_tx = rayon_sender.clone();

                    pool.spawn_async(move || {
                        let res = rayon_tx.blocking_send(run_simulation(
                            file_in_directory,
                            calculation_to_run,
                            ui_progress_sender,
                            ui_result_sender,
                        ));
                        if let Err(e) = res {
                            log::warn!("Failed to send...: {:?}", e);
                        }
                    });
                }
            }
        }

        // Update the `AppState` if the simulation is running
        if let AppState::Running(tracker) = local_app.state_mut() {
            // If there is an updated value in the `tracker_receiver` channel then update `AppState`
            if let Ok(result) = progress_receiver.try_recv() {
                *tracker = Arc::new(result);
            }
        }

        // If we get a result then write it to file
        // This is spawned on a seperate thread, to run in the background and prevent
        // io from affecting the application loop
        if let Ok(result) = result_receiver.try_recv() {
            local_app.potential = Some(result.potential.clone());
            if let Some(selected) = files_list_state.selected() {
                let file_name = local_app.file_in_directory(selected).clone();
                tokio::spawn(write_result_to_file(result, file_name));
            }
        }

        if rayon_receiver.try_recv().is_ok() {
            tracing::info!("Finished");
            local_app.initialized();
        }
    }

    terminal.clear()?;
    terminal.show_cursor()?;
    crossterm::terminal::disable_raw_mode()?;

    Ok(())
}

/// Writes an `NEGFResult` to file
///
/// By default this creates a new subdirectory with the name of the file
/// (stripped of it's extension) and then places results into different
/// subfolders for each voltage point
async fn write_result_to_file<T: Copy + RealField>(
    result: NEGFResult<T>,
    file_name: std::path::PathBuf,
) -> Result<(), std::io::Error> {
    let mut write_path = std::path::PathBuf::default();
    let folder_name = file_name.file_prefix().unwrap();
    write_path.push(folder_name);

    // If the write directory does not exist then create it
    if !write_path.exists() {
        std::fs::create_dir(write_path.clone())?;
    }

    // Write the current, appending to the collected file
    let (calculation_type, voltage) = match result.calculation {
        Calculation::Coherent {
            voltage_target: voltage,
        } => ("coherent", voltage),
        Calculation::Incoherent {
            voltage_target: voltage,
        } => ("incoherent", voltage),
    };
    let current_path: std::path::PathBuf = [
        write_path.clone(),
        std::path::PathBuf::from(format!("{}_current.csv", calculation_type)),
    ]
    .iter()
    .collect();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(current_path)?;
    writeln!(file, "{}, {}", voltage, result.current,)?;

    write_path.push(format!("{:.2}V", voltage));

    // If the write directory for this voltage point does not exist then create it
    if !write_path.exists() {
        std::fs::create_dir(write_path.clone())?;
    }

    let mut potential_path = write_path.clone();
    potential_path.push(format!("{}_potential.csv", calculation_type));

    let mut file = std::fs::File::create(potential_path)?;
    // Write the potential
    for value in result.potential.iter() {
        let value = value.to_string();
        writeln!(file, "{}", value)?;
    }

    let mut charge_path = write_path.clone();
    charge_path.push(format!("{}_charge.csv", calculation_type));

    let mut file = std::fs::File::create(charge_path)?;
    // Write the charge
    for value in result.electron_density.iter() {
        let value = value.to_string();
        writeln!(file, "{}", value)?;
    }

    Ok(())
}
