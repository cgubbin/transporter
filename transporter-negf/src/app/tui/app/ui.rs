use tui::{
    backend::Backend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Span, Spans},
    widgets::{Block, BorderType, Borders, Cell, List, ListItem, ListState, Paragraph, Row, Table},
    Frame,
};
use tui_logger::TuiLoggerWidget;

use super::{actions::Actions, state::AppState, App};
use crate::app::Calculation;

pub(crate) fn draw<B>(
    rect: &mut Frame<B>,
    app: &App,
    files_list_state: &mut ListState,
) -> miette::Result<()>
where
    B: Backend,
{
    let size = rect.size();
    check_size(&size);

    // The vertical layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(12),
            ]
            .as_ref(),
        )
        .split(size);

    // The title
    let title = draw_title();
    rect.render_widget(title, chunks[0]);

    // Body and help
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(20), Constraint::Length(32)].as_ref())
        .split(chunks[1]);

    match app.state() {
        AppState::Running(tracker) => {
            let running_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(40), Constraint::Percentage(60)].as_ref())
                .split(body_chunks[0]);

            match tracker.calculation_type {
                // For a coherent calculation we split the left panel into two segments
                Calculation::Coherent { .. } => {
                    let tracker_chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(running_chunks[0]);
                    let tracker_info = render_coherent_tracker(
                        tracker.current_voltage,
                        tracker.outer_iteration,
                        tracker.current_outer_residual,
                        tracker.target_outer_residual,
                    );
                    rect.render_widget(tracker_info.0, tracker_chunks[0]);
                    rect.render_widget(tracker_info.1, tracker_chunks[1]);
                }
                // For an incoherent calculation we split the left panel into three segments
                Calculation::Incoherent { .. } => {
                    let tracker_chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([
                            Constraint::Percentage(30),
                            Constraint::Percentage(30),
                            Constraint::Percentage(40),
                        ])
                        .split(running_chunks[0]);

                    let tracker_info = render_incoherent_tracker(
                        tracker.current_voltage,
                        tracker.outer_iteration,
                        tracker.inner_iteration.unwrap_or(0_u32),
                        tracker.current_outer_residual,
                        tracker.target_outer_residual,
                    );
                    rect.render_widget(tracker_info.0, tracker_chunks[0]);
                    rect.render_widget(tracker_info.1, tracker_chunks[1]);
                    rect.render_widget(tracker_info.2, tracker_chunks[2]);
                }
            }

            let graph = render_electron_density();
            rect.render_widget(graph, running_chunks[1]);
            // let left = render_simulation_tracker()
            // let (left, right) = render_simulation_tracker();
        }
        _ => {
            let files_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(40), Constraint::Percentage(60)].as_ref())
                .split(body_chunks[0]);
            let (left, right) = render_files(files_list_state, &app.files_in_directory)?;
            rect.render_stateful_widget(left, files_chunks[0], files_list_state);
            rect.render_widget(right, files_chunks[1]);
        }
    }

    // Render the help tool-tips

    let help_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(75), Constraint::Percentage(25)].as_ref())
        .split(body_chunks[1]);
    let help = draw_help(app.actions());
    let calc = render_calc(app.calculation_type);
    rect.render_widget(help, help_chunks[0]);
    rect.render_widget(calc, help_chunks[1]);

    // Logs
    let logs = draw_logs();
    rect.render_widget(logs, chunks[2]);

    Ok(())
}

fn check_size(rect: &Rect) {
    if rect.width < 52 {
        panic!("Require width >= 52, (got {})", rect.width);
    }
    if rect.height < 28 {
        panic!("Require height >= 28, (got {})", rect.height);
    }
}

fn draw_title<'a>() -> Paragraph<'a> {
    Paragraph::new("Transporter NEGF Solver")
        .style(Style::default().fg(Color::LightCyan))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::White))
                .border_type(BorderType::Plain),
        )
}

fn draw_help(actions: &Actions) -> Table {
    let key_style = Style::default().fg(Color::LightCyan);
    let help_style = Style::default().fg(Color::Gray);

    let mut rows = vec![];
    for action in actions.actions().iter() {
        let mut first = true;
        for key in action.keys() {
            let help = if first {
                first = false;
                action.to_string()
            } else {
                String::from("")
            };
            let row = Row::new(vec![
                Cell::from(Span::styled(key.to_string(), key_style)),
                Cell::from(Span::styled(help, help_style)),
            ]);
            rows.push(row);
        }
    }

    Table::new(rows)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Plain)
                .title("Help"),
        )
        .widths(&[Constraint::Length(11), Constraint::Min(20)])
        .column_spacing(1)
}

fn draw_logs<'a>() -> TuiLoggerWidget<'a> {
    TuiLoggerWidget::default()
        .style_error(Style::default().fg(Color::Red))
        .style_debug(Style::default().fg(Color::Green))
        .style_warn(Style::default().fg(Color::Yellow))
        .style_trace(Style::default().fg(Color::Gray))
        .style_info(Style::default().fg(Color::Blue))
        .block(
            Block::default()
                .title("Logs")
                .border_style(Style::default().fg(Color::White).bg(Color::Black))
                .borders(Borders::ALL),
        )
        .style(Style::default().fg(Color::White).bg(Color::Black))
}

fn render_files<'a>(
    files_list_state: &ListState,
    files_in_directory: &[std::path::PathBuf],
) -> miette::Result<(List<'a>, Table<'a>)> {
    let files = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::White))
        .title(".toml Files")
        .border_type(BorderType::Plain);

    let items = files_in_directory
        .iter()
        .map(|file| {
            let file = file.clone().into_os_string().into_string().unwrap();
            ListItem::new(Spans::from(vec![Span::styled(file, Style::default())]))
        })
        .collect::<Vec<_>>();

    let selected_file = files_in_directory
        .get(
            files_list_state
                .selected()
                .expect("A file is always selected"),
        )
        .expect("The file must exist")
        .clone();

    let list = List::new(items).block(files).highlight_style(
        Style::default()
            .bg(Color::Yellow)
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    );

    // Read in the structure to allow `std::fmt::Display` formatting
    let device: crate::device::Device<f64, nalgebra::U1> =
        crate::device::Device::build(selected_file)?;

    let data = device
        .layers
        .iter()
        .map(|layer| {
            vec![
                layer.material.to_string(),
                format!("{:.2}", layer.thickness[0]),
                format!("{:+e}", layer.acceptor_density),
                format!("{:+e}", layer.donor_density),
            ]
        })
        .collect::<Vec<_>>();
    let selected_style = Style::default().add_modifier(Modifier::REVERSED);
    let normal_style = Style::default().bg(Color::Blue);
    let header_cells = [
        "Material",
        "Thickness (nm)",
        "Acceptor Density (1/cm^3)",
        "Donor Density (1/cm^3)",
    ]
    .iter()
    .map(|h| Cell::from(*h).style(Style::default().fg(Color::Red)));

    let header = tui::widgets::Row::new(header_cells)
        .style(normal_style)
        .height(1)
        .bottom_margin(1);

    let rows = data.iter().map(|item| {
        let height = item
            .iter()
            .map(|content| content.chars().filter(|c| *c == '\n').count())
            .max()
            .unwrap_or(0)
            + 1;
        let cells = item.iter().map(|c| Cell::from(c.clone()));
        Row::new(cells).height(height as u16).bottom_margin(1)
    });

    let table = Table::new(rows)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Structure Information"),
        )
        .highlight_style(selected_style)
        .highlight_symbol(">> ")
        .column_spacing(1)
        .widths(&[
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ]);

    Ok((list, table))
}
use std::path::PathBuf;

fn render_coherent_tracker<'a>(
    voltage: f64,
    outer_iteration: u32,
    current_residual: f64,
    target_residual: f64,
) -> (Paragraph<'a>, Paragraph<'a>) {
    let upper_title = format!("Running file");
    let time_in_seconds = 10.5;
    let upper_box = Paragraph::new(vec![
        Spans::from(Span::raw(format!("Solving for voltage {}V", voltage))),
        Spans::from(Span::raw(format!(
            "Current simulation time {} seconds",
            time_in_seconds
        ))),
    ])
    .style(Style::default().fg(Color::LightCyan))
    .alignment(Alignment::Left)
    .block(
        Block::default()
            .title(upper_title)
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::White))
            .border_type(BorderType::Plain),
    );

    let outer_loop_title = "Outer Loop";
    let outer_box = Paragraph::new(vec![
        Spans::from(Span::raw(format!("Outer iteration {}", outer_iteration))),
        Spans::from(Span::raw(format!(
            "Current time to run one outer loop {}",
            time_in_seconds
        ))),
    ])
    .style(Style::default().fg(Color::LightCyan))
    .alignment(Alignment::Left)
    .block(
        Block::default()
            .title(outer_loop_title)
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::White))
            .border_type(BorderType::Plain),
    );
    (upper_box, outer_box)
}

fn render_incoherent_tracker<'a>(
    voltage: f64,
    outer_iteration: u32,
    inner_iteration: u32,
    current_residual: f64,
    target_residual: f64,
) -> (Paragraph<'a>, Paragraph<'a>, Paragraph<'a>) {
    let upper_title = format!("Running file");
    let time_in_seconds = 10.5;
    let upper_box = Paragraph::new(vec![
        Spans::from(Span::raw(format!("Solving for voltage {}V", voltage))),
        Spans::from(Span::raw(format!(
            "Current simulation time {} seconds",
            time_in_seconds
        ))),
    ])
    .style(Style::default().fg(Color::LightCyan))
    .alignment(Alignment::Left)
    .block(
        Block::default()
            .title(upper_title)
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::White))
            .border_type(BorderType::Plain),
    );

    let outer_loop_title = "Outer Loop";
    let outer_box = Paragraph::new(vec![
        Spans::from(Span::raw(format!("Outer iteration {}", outer_iteration))),
        Spans::from(Span::raw(format!(
            "Current time to run one outer loop {}",
            time_in_seconds
        ))),
    ])
    .style(Style::default().fg(Color::LightCyan))
    .alignment(Alignment::Left)
    .block(
        Block::default()
            .title(outer_loop_title)
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::White))
            .border_type(BorderType::Plain),
    );

    let inner_loop_title = "Inner Loop";
    let inner_box = Paragraph::new(vec![
        Spans::from(Span::raw(format!("Inner iteration {}", inner_iteration))),
        Spans::from(Span::raw(format!(
            "Current time to run one inner loop {}",
            time_in_seconds
        ))),
    ])
    .style(Style::default().fg(Color::LightCyan))
    .alignment(Alignment::Left)
    .block(
        Block::default()
            .title(inner_loop_title)
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::White))
            .border_type(BorderType::Plain),
    );
    (upper_box, outer_box, inner_box)
}

use tui::widgets::Axis;
use tui::widgets::Chart;
use tui::widgets::Dataset;
use tui::widgets::GraphType;

fn render_electron_density<'a>() -> Chart<'a> {
    const DATA2: [(f64, f64); 7] = [
        (0.0, 0.0),
        (10.0, 1.0),
        (20.0, 0.5),
        (30.0, 1.5),
        (40.0, 1.0),
        (50.0, 2.5),
        (60.0, 3.0),
    ];
    let datasets = vec![Dataset::default()
        .name("data")
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Yellow))
        .graph_type(GraphType::Line)
        .data(&DATA2)];
    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(Span::styled(
                    "Electronic Density",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("x-coordinate (nm)")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, 50.0])
                .labels(vec![
                    Span::styled("0", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw("25"),
                    Span::styled("50", Style::default().add_modifier(Modifier::BOLD)),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("Electron density (x 10^{24})")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, 50.0])
                .labels(vec![
                    Span::styled("0", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw("25"),
                    Span::styled("50", Style::default().add_modifier(Modifier::BOLD)),
                ]),
        );
    chart
}

use tui::widgets::Wrap;

fn render_calc<'a>(calculation: crate::app::Calculation<f64>) -> Paragraph<'a> {
    let title = "Calculation Type";
    let para = Paragraph::new(vec![Spans::from(Span::raw(calculation.to_string()))])
        .wrap(Wrap { trim: true })
        .style(Style::default().fg(Color::LightCyan))
        .alignment(Alignment::Left)
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::White))
                .border_type(BorderType::Plain),
        );
    para
}
