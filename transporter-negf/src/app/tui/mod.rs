mod app;
mod inputs;
mod io;
mod rayon_async;

pub(crate) use app::App;
use rayon_async::async_threadpool::AsyncThreadPool;

use std::{sync::Arc, time::Duration};
use tokio::sync::Mutex;
use tui::{backend::CrosstermBackend, widgets::ListState, Terminal};

use crate::{app::run_simulation, error::IOError};
pub(crate) use app::state::Progress;
use app::ui;
use app::{state::AppState, AppReturn};
use inputs::events::Events;
use inputs::Event;
use io::{handler::IoAsyncHandler, IoEvent};
use tokio::sync::mpsc::channel;

#[tokio::main]
/// Run the TUI app and start the GUI
pub async fn run() -> Result<(), IOError> {
    let (sync_io_tx, mut sync_io_rx) = channel::<IoEvent>(100);

    let app = Arc::new(Mutex::new(App::new(sync_io_tx.clone())));
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
pub async fn start_ui(app: &Arc<Mutex<App>>) -> Result<(), IOError> {
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
    let (progress_sender, mut progress_receiver) = tokio::sync::mpsc::channel::<Progress>(20);

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
                    let file_in_directory = local_app.file_in_directory(selected).clone();
                    let calculation_to_run = *local_app.calculation_type();
                    let rayon_tx = rayon_sender.clone();

                    pool.spawn_async(move || {
                        let res = rayon_tx.blocking_send(run_simulation(
                            file_in_directory,
                            calculation_to_run,
                            ui_progress_sender,
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
