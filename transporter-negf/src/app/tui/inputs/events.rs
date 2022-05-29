use super::{keys::Key, Event};
use crate::error::IOError;
use crossterm::event::{self};
use log::error;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::sync::mpsc::{self, Receiver, Sender};

/// An event handler wrapping the crossterm input and tick events.
/// It handles each `E` in it's own thread and returns to a common `Receiver`
pub(crate) struct Events<E> {
    rx: Receiver<E>,
    /// We need to retain the sender to prevent drops
    _tx: Sender<E>,
    /// Whether the loop should terminate
    stop_capture: Arc<AtomicBool>,
}

impl Events<Event<Key>> {
    /// Create a new default instance of `Events`
    pub(crate) fn new(tick_rate: Duration) -> Result<Events<Event<Key>>, IOError> {
        // Initialise a multi-producer, single receiver channel
        let (tx, rx) = mpsc::channel(100);
        let stop_capture = Arc::new(AtomicBool::new(false));

        // Clone the sender and the stop condition
        let event_tx = tx.clone();
        let event_stop_capture = stop_capture.clone();

        tokio::spawn(async move {
            loop {
                // Check whether an event is available
                match event::poll(tick_rate) {
                    // If there is an event -> check whether it is a key-input and send it
                    Ok(true) => {
                        if let Ok(event::Event::Key(key_event)) = event::read() {
                            let key = Key::from(key_event);
                            if let Err(err) = event_tx.send(Event::Input(key)).await {
                                error!("Oops!, {}", err);
                            };
                        } else {
                            {}
                        }
                    }
                    // If there was no key-input we send a tick
                    Ok(false) => {}
                    _ => {}
                }

                if let Err(err) = event_tx.send(Event::Tick).await {
                    error!("Oops!, {}", err);
                }
                if event_stop_capture.load(Ordering::Relaxed) {
                    break;
                }
            }
        });
        Ok(Events {
            rx,
            _tx: tx,
            stop_capture,
        })
    }

    /// Read an event
    pub(crate) async fn next(&mut self) -> Event<Key> {
        self.rx.recv().await.unwrap_or(Event::Tick)
    }

    /// Close the handler
    pub(crate) fn close(&mut self) {
        self.stop_capture.store(true, Ordering::Relaxed)
    }
}
