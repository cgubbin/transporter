pub mod events;
pub mod keys;

/// An enum which either stores a keystroke `I`, or returns a `Tick`
#[derive(Debug)]
pub enum Event<I>
where
    I: std::fmt::Debug,
{
    /// A user keystroke `I`
    Input(I),
    /// A tick at the pre-defined rate
    Tick,
}
