use tracing::{subscriber::set_global_default, Subscriber};
use tracing_log::LogTracer;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

pub(crate) fn get_subscriber(env_filter: super::LogLevel) -> impl Subscriber + Send + Sync {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(env_filter.to_string()));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .without_time();
    //.with_writer(sink);

    Registry::default().with(env_filter).with(fmt_layer)
    //     .with(JsonStorageLayer)
    //     .with(formatting_layer)
}

pub(crate) fn init_subscriber(subscriber: impl Subscriber + Send + Sync) {
    LogTracer::init().expect("Failed to initialise logger.");
    set_global_default(subscriber).expect("Failed to set a subscriber.");
}
