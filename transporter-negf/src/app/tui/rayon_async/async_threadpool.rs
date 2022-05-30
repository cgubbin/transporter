use super::async_handle::AsyncRayonHandle;
use rayon::ThreadPool;
use std::panic::{catch_unwind, AssertUnwindSafe};
use tokio::sync::oneshot;

/// An extension trait to integrate Rayon's [`ThreadPool`](rayon::ThreadPool)
/// and tokio
pub(crate) trait AsyncThreadPool {
    /// Asynchronous wrapper around Rayon's
    /// [`ThreadPool::spawn`](rayon::ThreadPool::spawn).
    ///
    /// Runs a function on the global Rayon thread pool with LIFO priority,
    /// produciing a future that resolves with the function's return value.
    ///
    /// # Panics
    /// If the task function panics, the panic will be propagated through the
    /// returned future. This will NOT trigger the Rayon thread pool's panic
    /// handler.
    ///
    /// If the returned handle is dropped, and the return value of `func` panics
    /// when dropped, that panic WILL trigger the thread pool's panic
    /// handler.
    fn spawn_async<F, R>(&self, func: F) -> AsyncRayonHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;

    /// Asynchronous wrapper around Rayon's
    /// [`ThreadPool::spawn_fifo`](rayon::ThreadPool::spawn_fifo).
    ///
    /// Runs a function on the global Rayon thread pool with FIFO priority,
    /// produciing a future that resolves with the function's return value.
    ///
    /// # Panics
    /// If the task function panics, the panic will be propagated through the
    /// returned future. This will NOT trigger the Rayon thread pool's panic
    /// handler.
    ///
    /// If the returned handle is dropped, and the return value of `func` panics
    /// when dropped, that panic WILL trigger the thread pool's panic
    /// handler.
    fn spawn_fifo_async<F, R>(&self, f: F) -> AsyncRayonHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;
}

impl AsyncThreadPool for ThreadPool {
    fn spawn_async<F, R>(&self, function: F) -> AsyncRayonHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();

        self.spawn(move || {
            let _ = tx.send(catch_unwind(AssertUnwindSafe(function)));
        });

        AsyncRayonHandle { rx }
    }

    fn spawn_fifo_async<F, R>(&self, function: F) -> AsyncRayonHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();

        self.spawn_fifo(move || {
            let _ = tx.send(catch_unwind(AssertUnwindSafe(function)));
        });

        AsyncRayonHandle { rx }
    }
}
