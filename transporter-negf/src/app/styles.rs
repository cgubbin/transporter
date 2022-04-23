use owo_colors::Style;

// Stylesheet used to colorize prints.
#[derive(Debug, Default)]
pub(crate) struct Styles {
    pub device_style: Style,
    pub layer_style: Style,
    // ... other styles
}

impl Styles {
    pub(crate) fn colorize(&mut self) {
        self.device_style = Style::new().bright_blue();
        self.layer_style = Style::new().bright_green();
        // ... other styles
    }
}
