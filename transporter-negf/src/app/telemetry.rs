use tracing::{subscriber::set_global_default, Subscriber};
use tracing_log::LogTracer;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

/// Creates a subscriber which write to `console::Term::stdout` and to a log file `log.log`
/// located in the results directory.
pub(crate) fn get_subscriber(
    env_filter: super::LogLevel,
) -> (
    impl Subscriber + Send + Sync,
    tracing_appender::non_blocking::WorkerGuard,
) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(env_filter.to_string()));

    let fmt_layer = tracing_subscriber::fmt::Layer::new()
        .with_writer(console::Term::stdout)
        .without_time();

    let appender = tracing_appender::rolling::never("../results", "log.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(appender);

    (
        Registry::default().with(env_filter).with(fmt_layer).with(
            tracing_subscriber::fmt::Layer::new()
                .with_writer(non_blocking)
                .json(),
        ),
        guard,
    )
}

pub(crate) fn init_subscriber(subscriber: impl Subscriber + Send + Sync) {
    LogTracer::init().expect("Failed to initialise logger.");
    set_global_default(subscriber).expect("Failed to set a subscriber.");
}
