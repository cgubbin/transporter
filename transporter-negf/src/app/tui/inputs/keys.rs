use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// Represents the key-presses recognised by the program
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) enum Key {
    /// Enter and numpad-Enter
    Enter,
    /// Escape
    Esc,
    /// Left arrow
    Left,
    /// Right arrow
    Right,
    /// Up arrow
    Up,
    /// Down arrow
    Down,
    Char(char),
    Ctrl(char),
    /// All other key-presses which have no specific use in the program
    Unknown,
}

impl Key {
    /// Has the user asked to exit the program?
    pub(crate) fn is_exit(&self) -> bool {
        matches!(self, Key::Ctrl('c') | Key::Char('q') | Key::Esc)
    }
}

impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Key::Char(' ') => write!(f, "<Space>"),
            Key::Ctrl(' ') => write!(f, "<Ctrl+Space>"),
            Key::Char(c) => write!(f, "<{c}>"),
            Key::Ctrl(c) => write!(f, "<Ctrl+{c}>"),
            _ => write!(f, "<{:?}>", self),
        }
    }
}

impl From<KeyEvent> for Key {
    fn from(key_event: KeyEvent) -> Self {
        match key_event {
            KeyEvent {
                code: KeyCode::Esc, ..
            } => Key::Esc,
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => Key::Enter,
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => Key::Left,
            KeyEvent {
                code: KeyCode::Right,
                ..
            } => Key::Right,
            KeyEvent {
                code: KeyCode::Up, ..
            } => Key::Up,
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => Key::Down,
            KeyEvent {
                code: KeyCode::Char(c),
                modifiers: KeyModifiers::CONTROL,
            } => Key::Ctrl(c),
            KeyEvent {
                code: KeyCode::Char(c),
                ..
            } => Key::Char(c),
            _ => Key::Unknown,
        }
    }
}
