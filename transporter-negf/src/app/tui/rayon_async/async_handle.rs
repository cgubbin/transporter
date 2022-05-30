use std::{
    future::Future,
    panic::resume_unwind,
    pin::Pin,
    task::{Context, Poll},
    thread,
};
use tokio::sync::oneshot::Receiver;

/// An async handle for a blocked task running in a Rayon threadpool
///
/// If the spawned task panics `poll()` will propagate the panic
#[derive(Debug)]
pub(crate) struct AsyncRayonHandle<T> {
    pub(crate) rx: Receiver<thread::Result<T>>,
}

impl<T> Future for AsyncRayonHandle<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let rx = Pin::new(&mut self.rx);
        rx.poll(cx).map(|result| {
            result
                .expect("Error unreachable: Tokio closed channel")
                .unwrap_or_else(|err| resume_unwind(err))
        })
    }
}
