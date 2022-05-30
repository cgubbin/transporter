use super::async_handle::AsyncRayonHandle;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tokio::sync::oneshot;

/// Create an `async` wrapper around Rayon's [`spawn`](rayon::spawn)
///
/// Runs a function on Rayon's global threadpool with LIFO priority which
/// returns a future resolving to the function's return value
///
/// # Panics
/// If the task function panics the panic propagates through the future returned
/// without triggering the Rayon threadpool panic handler
///
/// If the handle returned is dropped and then the return value panics it triggers
/// the Rayon threadpool panic handler
pub(crate) fn spawn<F, R>(function: F) -> AsyncRayonHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = oneshot::channel();

    rayon::spawn(move || {
        let _ = tx.send(catch_unwind(AssertUnwindSafe(function)));
    });

    AsyncRayonHandle { rx }
}

/// An async wrapper around Rayon's [`spawn_fifo`](rayon::spawn_fifo)
///
/// Run a `function` on the global rayon threadpool with FIFO priority
/// and produce a future resolving to `function`s return value
///
/// # Panics
/// If the task function panics the panic propagates through the future returned
/// without triggering the Rayon threadpool panic handler
///
/// If the handle returned is dropped and then the return value panics it triggers
/// the Rayon threadpool panic handler
pub(crate) fn spawn_fifo<F, R>(function: F) -> AsyncRayonHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = oneshot::channel();

    rayon::spawn_fifo(move || {
        let _ = tx.send(catch_unwind(AssertUnwindSafe(function)));
    });

    AsyncRayonHandle { rx }
}
