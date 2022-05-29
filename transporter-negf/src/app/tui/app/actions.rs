use super::super::inputs::keys::Key;
use std::{collections::HashMap, slice::Iter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Quit,
    Up,
    Down,
    Execute,
    ToggleCalculation,
    IncrementVoltageTarget,
    DecrementVoltageTarget,
}

impl Action {
    pub(crate) fn keys(&self) -> &[Key] {
        match self {
            Action::Quit => &[Key::Ctrl('c'), Key::Char('q')],
            Action::Up => &[Key::Up],
            Action::Down => &[Key::Down],
            Action::Execute => &[Key::Enter],
            Action::ToggleCalculation => &[Key::Char('x')],
            Action::IncrementVoltageTarget => &[Key::Char('k')],
            Action::DecrementVoltageTarget => &[Key::Char('l')],
        }
    }

    fn iterator() -> Iter<'static, Action> {
        static ACTIONS: [Action; 7] = [
            Action::Quit,
            // Actions we are actually using
            Action::Up,
            Action::Down,
            Action::Execute,
            Action::ToggleCalculation,
            Action::IncrementVoltageTarget,
            Action::DecrementVoltageTarget,
        ];
        ACTIONS.iter()
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let str = match self {
            Action::Quit => "Quit",
            Action::Up => "Move up",
            Action::Down => "Move down",
            Action::Execute => "Run file",
            Action::ToggleCalculation => "Toggle calculation",
            Action::IncrementVoltageTarget => "Increase voltage target",
            Action::DecrementVoltageTarget => "Decrease voltage target",
        };
        write!(f, "{}", str)
    }
}

pub struct Actions(Vec<Action>);

impl From<Vec<Action>> for Actions {
    /// Build contextual action
    ///
    /// # Panics
    ///
    /// If two actions have same key
    fn from(actions: Vec<Action>) -> Self {
        // Check unicity of keys
        let mut map: HashMap<Key, Vec<Action>> = HashMap::new();
        for &action in actions.iter() {
            for &key in action.keys().iter() {
                match map.get_mut(&key) {
                    Some(vec) => vec.push(action),
                    None => {
                        map.insert(key, vec![action]);
                    }
                }
            }
        }

        let errors = map
            .iter()
            .filter(|(_, actions)| actions.len() > 1) // Two actions have the same shortcut
            .map(|(key, actions)| {
                let actions = actions
                    .iter()
                    .map(Action::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("Conflict key {} with actions {}", key, actions)
            })
            .collect::<Vec<_>>();

        if !errors.is_empty() {
            panic!("{}", errors.join("; "))
        }

        Self(actions)
    }
}

impl Actions {
    pub(crate) fn find(&self, key: Key) -> Option<&Action> {
        Action::iterator()
            .filter(|action| self.0.contains(action))
            .find(|action| action.keys().contains(&key))
    }

    pub(crate) fn actions(&self) -> &[Action] {
        self.0.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_find_action_by_key() {
        let actions: Actions = vec![Action::Quit, Action::Up].into();
        let result = actions.find(Key::Ctrl('c'));
        assert_eq!(result, Some(&Action::Quit));
    }

    #[test]
    fn should_find_action_by_key_not_found() {
        let actions: Actions = vec![Action::Quit, Action::Up].into();
        let result = actions.find(Key::Ctrl('q'));
        assert_eq!(result, None);
    }

    #[test]
    fn should_create_actions_from_vec() {
        let _actions: Actions =
            vec![Action::Quit, Action::Up, Action::Down, Action::Execute].into();
    }

    #[test]
    #[should_panic]
    fn should_panic_when_create_actions_conflict_key() {
        let _actions: Actions = vec![
            Action::Quit,
            Action::Up,
            Action::Execute,
            Action::Down,
            Action::Down,
            Action::Quit,
            Action::Up,
        ]
        .into();
    }
}
