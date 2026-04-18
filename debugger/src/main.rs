//! modgrad debugger: live neural computer inspector.
//!
//! Connects to any running modgrad NC via TCP and shows:
//!   - 3D brain: neurons as particles, colored by region
//!   - Token stream: live feed, color-coded by modality
//!   - Region activations: per-region heatmaps
//!   - NLM traces: memory trace per region
//!   - Global sync: pair values
//!   - Controls: pause/resume/step, inject text/actions
//!
//! Usage:
//!   modgrad-debugger [host:port]
//!   modgrad-debugger 127.0.0.1:4747

mod connection;

use eframe::egui;
use egui::{Color32, Pos2, Stroke, Vec2};
use connection::DebugClient;
use isis_runtime::nc_socket::{DebugResponse, DebugRequest};

const REGION_COLORS: &[Color32] = &[
    Color32::from_rgb(0x44, 0x88, 0xff), // input: blue
    Color32::from_rgb(0x44, 0xff, 0x88), // attention: green
    Color32::from_rgb(0xff, 0x88, 0x44), // output: orange
    Color32::from_rgb(0xff, 0x44, 0x88), // motor: pink
    Color32::from_rgb(0x88, 0x44, 0xff), // cerebellum: purple
    Color32::from_rgb(0x88, 0xff, 0x44), // basal_ganglia: lime
    Color32::from_rgb(0xff, 0x88, 0xff), // insula: magenta
    Color32::from_rgb(0xff, 0xff, 0x44), // hippocampus: yellow
    Color32::from_rgb(0x44, 0xff, 0xff), // extra: cyan
    Color32::from_rgb(0xff, 0xaa, 0x88), // extra: salmon
];

fn main() -> eframe::Result<()> {
    let addr = std::env::args().nth(1)
        .unwrap_or_else(|| "127.0.0.1:4747".to_string());

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 1000.0])
            .with_title("modgrad debugger"),
        ..Default::default()
    };

    eframe::run_native(
        "modgrad debugger",
        options,
        Box::new(move |_cc| Ok(Box::new(App::new(&addr)))),
    )
}

/// Per-neuron 3D position.
struct NeuronPos {
    x: f32,
    y: f32,
    z: f32,
    region: usize,
    index: usize,
}

struct App {
    client: Option<DebugClient>,
    addr: String,

    // Metadata (from GetMeta)
    region_names: Vec<String>,
    region_d_model: Vec<usize>,
    region_memory: Vec<usize>,
    region_params: Vec<usize>,
    n_connections: usize,
    vocab_size: usize,
    total_params: usize,
    n_sync: usize,

    // Live state (from GetState)
    region_activations: Vec<Vec<f32>>,
    region_outputs: Vec<Vec<f32>>,
    global_sync: Vec<f32>,
    history_len: usize,
    exit_lambdas: Vec<f32>,
    ticks_used: usize,

    // Token history (from GetHistory)
    recent_tokens: Vec<usize>,

    // NLM traces (from GetTrace)
    trace_region: usize,
    trace_data: Vec<f32>,
    trace_d_model: usize,
    trace_memory: usize,

    // 3D vis
    neuron_positions: Vec<NeuronPos>,
    visible: Vec<bool>,
    rot_x: f32,
    rot_y: f32,
    zoom: f32,
    size_scale: f32,
    show_sync: bool,

    // Controls
    paused: bool,
    inject_text: String,
    poll_interval: f32,
    last_poll: std::time::Instant,

    // Active panel
    right_panel: RightPanel,

    // Command center
    cmd_input: String,
    cmd_history: Vec<String>,
    cmd_output: Vec<(String, Color32)>,
    cmd_history_pos: usize,
}

#[derive(PartialEq)]
enum RightPanel {
    State,
    Tokens,
    Traces,
}

impl App {
    fn new(addr: &str) -> Self {
        let mut app = Self {
            client: None,
            addr: addr.to_string(),
            region_names: Vec::new(),
            region_d_model: Vec::new(),
            region_memory: Vec::new(),
            region_params: Vec::new(),
            n_connections: 0,
            vocab_size: 0,
            total_params: 0,
            n_sync: 0,
            region_activations: Vec::new(),
            region_outputs: Vec::new(),
            global_sync: Vec::new(),
            exit_lambdas: Vec::new(),
            ticks_used: 0,
            history_len: 0,
            recent_tokens: Vec::new(),
            trace_region: 0,
            trace_data: Vec::new(),
            trace_d_model: 0,
            trace_memory: 0,
            neuron_positions: Vec::new(),
            visible: Vec::new(),
            rot_x: -0.3,
            rot_y: 0.5,
            zoom: 1.0,
            size_scale: 1.5,
            show_sync: true,
            paused: false,
            inject_text: String::new(),
            poll_interval: 0.1,
            last_poll: std::time::Instant::now(),
            right_panel: RightPanel::State,
            cmd_input: String::new(),
            cmd_history: Vec::new(),
            cmd_output: Vec::new(),
            cmd_history_pos: 0,
        };
        app.try_connect();
        app
    }

    fn try_connect(&mut self) {
        match DebugClient::connect(&self.addr) {
            Ok(c) => {
                eprintln!("Connected to {}", self.addr);
                self.client = Some(c);
                self.fetch_meta();
            }
            Err(e) => {
                eprintln!("Connection failed: {e}");
                self.client = None;
            }
        }
    }

    fn fetch_meta(&mut self) {
        let Some(client) = &mut self.client else { return };
        if let Ok(resp) = client.request(&DebugRequest::GetMeta) {
            if let DebugResponse::Meta {
                region_names, region_params, region_d_model, region_memory,
                n_connections, vocab_size, total_params, n_global_sync,
            } = resp {
                self.region_names = region_names;
                self.region_params = region_params;
                self.region_d_model = region_d_model.clone();
                self.region_memory = region_memory.clone();
                self.n_connections = n_connections;
                self.vocab_size = vocab_size;
                self.total_params = total_params;
                self.n_sync = n_global_sync;
                self.visible = vec![true; self.region_names.len()];
                self.build_neuron_positions(&region_d_model);
            }
        }
    }

    fn fetch_state(&mut self) -> bool {
        let Some(client) = &mut self.client else { return false };
        match client.request(&DebugRequest::GetState) {
            Ok(DebugResponse::State {
                region_activations, region_outputs, global_sync, history_len,
                exit_lambdas, ticks_used,
            }) => {
                self.region_activations = region_activations;
                self.region_outputs = region_outputs;
                self.global_sync = global_sync;
                self.history_len = history_len;
                self.exit_lambdas = exit_lambdas;
                self.ticks_used = ticks_used;
                true
            }
            Err(_) => false,
            _ => true,
        }
    }

    fn fetch_history(&mut self) {
        let Some(client) = &mut self.client else { return };
        if let Ok(resp) = client.request(&DebugRequest::GetHistory { last_n: 100 }) {
            if let DebugResponse::History { tokens } = resp {
                self.recent_tokens = tokens;
            }
        }
    }

    fn fetch_trace(&mut self) {
        let Some(client) = &mut self.client else { return };
        if let Ok(resp) = client.request(&DebugRequest::GetTrace { region: self.trace_region }) {
            if let DebugResponse::Trace { d_model, memory_length, trace, .. } = resp {
                self.trace_d_model = d_model;
                self.trace_memory = memory_length;
                self.trace_data = trace;
            }
        }
    }

    fn build_neuron_positions(&mut self, d_models: &[usize]) {
        self.neuron_positions.clear();
        let n = d_models.len();
        let mut rng = 42u64;
        let mut rand_f = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };

        for (ri, &neurons) in d_models.iter().enumerate() {
            // Layout: cortical (top row), subcortical (bottom row)
            let col = ri % 4;
            let row = ri / 4;
            let cx = col as f32 * 3.5 - 5.0;
            let cy = -(row as f32) * 4.0;
            let cz = (ri as f32 * 0.3) - 0.5;
            let spread = (neurons as f32).sqrt() * 0.25;

            for ni in 0..neurons {
                let theta = rand_f() * std::f32::consts::PI;
                let phi = rand_f() * std::f32::consts::TAU;
                let r = spread * (0.3 + 0.7 * rand_f().abs());
                self.neuron_positions.push(NeuronPos {
                    x: cx + r * theta.sin() * phi.cos(),
                    y: cy + r * theta.cos(),
                    z: cz + r * theta.sin() * phi.sin(),
                    region: ri,
                    index: ni,
                });
            }
        }
    }

    fn project(&self, x: f32, y: f32, z: f32, center: Pos2) -> Option<(Pos2, f32)> {
        let (sy, cy) = self.rot_y.sin_cos();
        let (sx, cx) = self.rot_x.sin_cos();
        let rx = x * cy + z * sy;
        let ry = y * cx - (-x * sy + z * cy) * sx;
        let rz = y * sx + (-x * sy + z * cy) * cx;
        let depth = rz + 10.0;
        if depth < 0.5 { return None; }
        let fov = 300.0 * self.zoom;
        Some((Pos2::new(center.x + rx * fov / depth, center.y - ry * fov / depth), depth))
    }

    fn region_color(&self, idx: usize) -> Color32 {
        REGION_COLORS[idx % REGION_COLORS.len()]
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll NC state periodically
        if self.last_poll.elapsed().as_secs_f32() >= self.poll_interval {
            if self.client.is_none() { self.try_connect(); }
            if self.client.is_some() && !self.paused {
                // If any request fails, disconnect (server probably died)
                let ok = self.fetch_state();
                if ok {
                    if self.right_panel == RightPanel::Tokens { self.fetch_history(); }
                    if self.right_panel == RightPanel::Traces { self.fetch_trace(); }
                } else {
                    eprintln!("  Lost connection");
                    self.client = None;
                }
            }
            self.last_poll = std::time::Instant::now();
        }

        // ── Top panel: controls ──
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let status = if self.client.is_some() { "🟢 Connected" } else { "🔴 Disconnected" };
                ui.label(status);
                ui.label(format!("{}:{}", self.addr, ""));
                if ui.button("Reconnect").clicked() { self.try_connect(); }
                ui.separator();

                if ui.button(if self.paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                    self.paused = !self.paused;
                    if let Some(client) = &mut self.client {
                        let req = if self.paused { DebugRequest::Pause } else { DebugRequest::Resume };
                        client.request(&req).ok();
                    }
                }

                ui.separator();
                ui.add(egui::Slider::new(&mut self.size_scale, 0.1..=5.0).text("Size"));
                ui.add(egui::Slider::new(&mut self.poll_interval, 0.05..=2.0).text("Poll (s)"));
                ui.checkbox(&mut self.show_sync, "Sync lines");
            });

            // Region toggles
            ui.horizontal(|ui| {
                for (i, name) in self.region_names.iter().enumerate() {
                    let color = self.region_color(i);
                    let label = egui::RichText::new(name)
                        .color(if self.visible[i] { color } else { Color32::GRAY })
                        .strong();
                    if ui.selectable_label(self.visible[i], label).clicked() {
                        self.visible[i] = !self.visible[i];
                    }
                }
            });
        });

        // ── Bottom panel: command center ──
        egui::TopBottomPanel::bottom("command_center").min_height(160.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Command Center");
                ui.separator();
                ui.label(format!("{} params | {} vocab | {} tokens",
                    self.total_params, self.vocab_size, self.history_len));
            });

            // Quick action buttons
            ui.horizontal(|ui| {
                if ui.button("⏸ Pause").clicked() { self.exec_cmd("/pause"); }
                if ui.button("▶ Resume").clicked() { self.exec_cmd("/resume"); }
                if ui.button("🔄 Reset").clicked() { self.exec_cmd("/reset"); }
                if ui.button("📋 History").clicked() { self.exec_cmd("/history"); }
                if ui.button("🧠 Meta").clicked() { self.exec_cmd("/meta"); }
                ui.separator();
                if ui.button("🎤 Say hello").clicked() { self.exec_cmd("hello"); }
                if ui.button("🖱 Click center").clicked() { self.exec_cmd("/click 0.5 0.5"); }
                if ui.button("⌨ Enter").clicked() { self.exec_cmd("/key enter"); }
            });

            // Output log (scrollable)
            let out_height = 80.0;
            egui::ScrollArea::vertical().max_height(out_height).stick_to_bottom(true).show(ui, |ui| {
                for (msg, color) in &self.cmd_output {
                    ui.label(egui::RichText::new(msg).color(*color).monospace().size(11.0));
                }
            });

            // Command input
            ui.horizontal(|ui| {
                ui.label(">");
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut self.cmd_input)
                        .desired_width(ui.available_width() - 60.0)
                        .font(egui::FontId::monospace(13.0))
                        .hint_text("type text or /command (try /help)")
                );

                if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    let cmd = self.cmd_input.clone();
                    if !cmd.is_empty() {
                        self.exec_cmd(&cmd);
                        self.cmd_input.clear();
                    }
                    resp.request_focus();
                }
            });
        });

        // ── Right panel: state / tokens / traces ──
        egui::SidePanel::right("info").default_width(300.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.right_panel, RightPanel::State, "State");
                ui.selectable_value(&mut self.right_panel, RightPanel::Tokens, "Tokens");
                ui.selectable_value(&mut self.right_panel, RightPanel::Traces, "Traces");
            });
            ui.separator();

            match self.right_panel {
                RightPanel::State => self.draw_state_panel(ui),
                RightPanel::Tokens => self.draw_token_panel(ui),
                RightPanel::Traces => self.draw_trace_panel(ui),
            }
        });

        // ── Central panel: 3D brain ──
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_brain(ui);
        });

        ctx.request_repaint_after(std::time::Duration::from_secs_f32(self.poll_interval));
    }
}

impl App {
    fn log(&mut self, msg: &str, color: Color32) {
        self.cmd_output.push((msg.to_string(), color));
        // Keep last 200 lines
        if self.cmd_output.len() > 200 { self.cmd_output.drain(..self.cmd_output.len() - 200); }
    }

    fn exec_cmd(&mut self, input: &str) {
        let input = input.trim();
        if input.is_empty() { return; }

        // Log the command
        self.cmd_history.push(input.to_string());
        self.cmd_history_pos = self.cmd_history.len();
        self.log(&format!("> {input}"), Color32::from_rgb(0x88, 0x88, 0xaa));

        if !input.starts_with('/') {
            // Plain text — inject into NC
            if let Some(client) = &mut self.client {
                match client.request(&DebugRequest::InjectText { text: input.to_string() }) {
                    Ok(DebugResponse::Ok) => self.log("  injected", Color32::GREEN),
                    Ok(DebugResponse::Error { msg }) => self.log(&format!("  error: {msg}"), Color32::RED),
                    Err(e) => self.log(&format!("  send failed: {e}"), Color32::RED),
                    _ => {}
                }
            } else {
                self.log("  not connected", Color32::RED);
            }
            return;
        }

        let parts: Vec<&str> = input.splitn(4, ' ').collect();
        match parts[0] {
            "/help" => {
                self.log("Commands:", Color32::WHITE);
                self.log("  /pause          — pause NC", Color32::GRAY);
                self.log("  /resume         — resume NC", Color32::GRAY);
                self.log("  /step <token>   — single step (token id)", Color32::GRAY);
                self.log("  /meta           — show model metadata", Color32::GRAY);
                self.log("  /history [n]    — show last n tokens", Color32::GRAY);
                self.log("  /trace <region> — fetch NLM trace", Color32::GRAY);
                self.log("  /click <x> <y>  — inject mouse click", Color32::GRAY);
                self.log("  /move <x> <y>   — inject mouse move", Color32::GRAY);
                self.log("  /key <name>     — inject key (enter/tab/esc/up/down)", Color32::GRAY);
                self.log("  /ctrl <char>    — inject ctrl+key", Color32::GRAY);
                self.log("  /type <text>    — inject key_type action", Color32::GRAY);
                self.log("  /reset          — reset NC state", Color32::GRAY);
                self.log("  /connect [addr] — reconnect", Color32::GRAY);
                self.log("  <text>          — inject text directly", Color32::GRAY);
            }

            "/pause" => {
                if let Some(client) = &mut self.client {
                    client.request(&DebugRequest::Pause).ok();
                    self.paused = true;
                    self.log("  paused", Color32::YELLOW);
                }
            }

            "/resume" => {
                if let Some(client) = &mut self.client {
                    client.request(&DebugRequest::Resume).ok();
                    self.paused = false;
                    self.log("  resumed", Color32::GREEN);
                }
            }

            "/step" => {
                let token = parts.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(32); // space
                if let Some(client) = &mut self.client {
                    client.request(&DebugRequest::Step { token }).ok();
                    self.log(&format!("  stepped token {token}"), Color32::YELLOW);
                }
            }

            "/meta" => {
                self.fetch_meta();
                let summary = format!("  {} regions, {} params, vocab {}",
                    self.region_names.len(), self.total_params, self.vocab_size);
                self.log(&summary, Color32::WHITE);
                let lines: Vec<(String, Color32)> = (0..self.region_names.len()).map(|i| {
                    let name = &self.region_names[i];
                    let d = self.region_d_model.get(i).copied().unwrap_or(0);
                    let m = self.region_memory.get(i).copied().unwrap_or(0);
                    let p = self.region_params.get(i).copied().unwrap_or(0);
                    (format!("  {name}: d_model={d} memory={m} params={p}"), self.region_color(i))
                }).collect();
                for (msg, color) in lines { self.log(&msg, color); }
            }

            "/history" => {
                let n = parts.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(50);
                if let Some(client) = &mut self.client {
                    if let Ok(DebugResponse::History { tokens }) = client.request(&DebugRequest::GetHistory { last_n: n }) {
                        let display: String = tokens.iter().map(|&t| {
                            let (ch, _) = token_display(t);
                            ch
                        }).collect();
                        self.log(&format!("  [{} tokens] {display}", tokens.len()), Color32::WHITE);
                    }
                }
            }

            "/trace" => {
                let r = parts.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                self.trace_region = r;
                self.fetch_trace();
                self.right_panel = RightPanel::Traces;
                self.log(&format!("  showing trace for region {r}"), Color32::WHITE);
            }

            "/click" => {
                let x: f32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.5);
                let y: f32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.5);
                if let Some(client) = &mut self.client {
                    use isis_runtime::regional::*;
                    let tokens = action_click(x, y);
                    client.request(&DebugRequest::Inject { tokens }).ok();
                    self.log(&format!("  click ({x:.2}, {y:.2})"), Color32::from_rgb(0xff, 0x44, 0x44));
                }
            }

            "/move" => {
                let x: f32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.5);
                let y: f32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.5);
                if let Some(client) = &mut self.client {
                    use isis_runtime::regional::*;
                    let tokens = action_mouse_move(x, y);
                    client.request(&DebugRequest::Inject { tokens }).ok();
                    self.log(&format!("  move ({x:.2}, {y:.2})"), Color32::from_rgb(0xff, 0x88, 0x44));
                }
            }

            "/key" => {
                if let Some(name) = parts.get(1) {
                    use isis_runtime::regional::*;
                    let key = match *name {
                        "enter" => Some(ACT_KEY_ENTER),
                        "tab" => Some(ACT_KEY_TAB),
                        "esc" | "escape" => Some(ACT_KEY_ESCAPE),
                        "up" => Some(ACT_KEY_UP),
                        "down" => Some(ACT_KEY_DOWN),
                        "left" => Some(ACT_KEY_LEFT),
                        "right" => Some(ACT_KEY_RIGHT),
                        "backspace" | "bs" => Some(ACT_KEY_BACKSPACE),
                        _ => None,
                    };
                    if let Some(k) = key {
                        if let Some(client) = &mut self.client {
                            let tokens = action_key(k);
                            client.request(&DebugRequest::Inject { tokens }).ok();
                            self.log(&format!("  key {name}"), Color32::from_rgb(0xff, 0x44, 0x44));
                        }
                    } else {
                        self.log(&format!("  unknown key: {name}"), Color32::RED);
                    }
                }
            }

            "/ctrl" => {
                if let Some(ch) = parts.get(1).and_then(|s| s.bytes().next()) {
                    use isis_runtime::regional::*;
                    if let Some(client) = &mut self.client {
                        let tokens = action_modified_key(ACT_KEY_CTRL, ch);
                        client.request(&DebugRequest::Inject { tokens }).ok();
                        self.log(&format!("  ctrl+{}", ch as char), Color32::from_rgb(0xff, 0x44, 0x44));
                    }
                }
            }

            "/type" => {
                if let Some(text) = parts.get(1) {
                    use isis_runtime::regional::*;
                    if let Some(client) = &mut self.client {
                        let tokens = action_type_text(text);
                        client.request(&DebugRequest::Inject { tokens }).ok();
                        self.log(&format!("  typed: {text}"), Color32::from_rgb(0x44, 0xff, 0x88));
                    }
                }
            }

            "/reset" => {
                self.log("  reset not yet implemented server-side", Color32::YELLOW);
            }

            "/connect" => {
                if let Some(addr) = parts.get(1) {
                    self.addr = addr.to_string();
                }
                self.try_connect();
                if self.client.is_some() {
                    self.log(&format!("  connected to {}", self.addr), Color32::GREEN);
                } else {
                    self.log(&format!("  failed to connect to {}", self.addr), Color32::RED);
                }
            }

            _ => {
                self.log(&format!("  unknown command: {}", parts[0]), Color32::RED);
                self.log("  try /help", Color32::GRAY);
            }
        }
    }

    fn draw_state_panel(&self, ui: &mut egui::Ui) {
        ui.heading("Region Activations");
        for (i, name) in self.region_names.iter().enumerate() {
            let color = self.region_color(i);
            let mag = self.region_activations.get(i)
                .map(|a| a.iter().map(|x| x * x).sum::<f32>().sqrt())
                .unwrap_or(0.0);
            let d = self.region_d_model.get(i).copied().unwrap_or(0);
            let m = self.region_memory.get(i).copied().unwrap_or(0);
            let p = self.region_params.get(i).copied().unwrap_or(0);
            ui.horizontal(|ui| {
                ui.colored_label(color, format!("{name:12} mag={mag:.2} d={d} mem={m} p={p}"));
            });

            // Mini activation bar
            if let Some(acts) = self.region_activations.get(i) {
                let (rect, _) = ui.allocate_exact_size(Vec2::new(280.0, 8.0), egui::Sense::hover());
                let n = acts.len().max(1);
                let w = rect.width() / n as f32;
                for (j, &v) in acts.iter().enumerate() {
                    let intensity = v.abs().clamp(0.0, 1.0);
                    let c = Color32::from_rgba_unmultiplied(
                        color.r(), color.g(), color.b(), (intensity * 255.0) as u8);
                    let r = egui::Rect::from_min_size(
                        Pos2::new(rect.min.x + j as f32 * w, rect.min.y),
                        Vec2::new(w, rect.height()));
                    ui.painter().rect_filled(r, 0.0, c);
                }
            }
        }

        ui.separator();
        ui.heading("Global Sync");
        if !self.global_sync.is_empty() {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(280.0, 40.0), egui::Sense::hover());
            let n = self.global_sync.len().max(1);
            let w = rect.width() / n as f32;
            let max_v = self.global_sync.iter().map(|v| v.abs()).fold(0.01f32, f32::max);
            for (j, &v) in self.global_sync.iter().enumerate() {
                let frac = (v / max_v).clamp(-1.0, 1.0);
                let mid_y = rect.center().y;
                let bar_h = frac.abs() * rect.height() / 2.0;
                let (top, bot) = if frac >= 0.0 {
                    (mid_y - bar_h, mid_y)
                } else {
                    (mid_y, mid_y + bar_h)
                };
                let c = if frac >= 0.0 {
                    Color32::from_rgb(0x44, 0xbb, 0xff)
                } else {
                    Color32::from_rgb(0xff, 0x66, 0x44)
                };
                let r = egui::Rect::from_min_max(
                    Pos2::new(rect.min.x + j as f32 * w, top),
                    Pos2::new(rect.min.x + (j + 1) as f32 * w, bot));
                ui.painter().rect_filled(r, 0.0, c);
            }
        }

        // Exit gate telemetry
        if !self.exit_lambdas.is_empty() {
            ui.separator();
            ui.heading("Exit Gate");
            ui.label(format!("Ticks used: {}/{}", self.ticks_used,
                self.exit_lambdas.len().max(self.ticks_used)));

            // Per-tick lambda bars
            let (rect, _) = ui.allocate_exact_size(Vec2::new(280.0, 30.0), egui::Sense::hover());
            let n = self.exit_lambdas.len().max(1);
            let w = rect.width() / n as f32;
            for (j, &lam) in self.exit_lambdas.iter().enumerate() {
                let bar_h = lam * rect.height();
                let c = if j < self.ticks_used {
                    Color32::from_rgb(0x44, 0xdd, 0x66) // green = tick ran
                } else {
                    Color32::from_rgb(0x88, 0x88, 0x88) // grey = skipped
                };
                let r = egui::Rect::from_min_max(
                    Pos2::new(rect.min.x + j as f32 * w, rect.max.y - bar_h),
                    Pos2::new(rect.min.x + (j + 1) as f32 * w - 1.0, rect.max.y));
                ui.painter().rect_filled(r, 1.0, c);
            }
            ui.label("λ per tick (height = exit probability)");
        }
    }

    fn draw_token_panel(&self, ui: &mut egui::Ui) {
        ui.heading("Token Stream");
        ui.label(format!("{} total, showing last {}", self.history_len, self.recent_tokens.len()));
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Build colored spans for the token stream
            let mut job = egui::text::LayoutJob::default();
            for &t in &self.recent_tokens {
                let (ch, color) = token_display(t);
                let s = if ch == '\n' { "\n".to_string() } else { ch.to_string() };
                job.append(&s, 0.0, egui::TextFormat {
                    color,
                    font_id: egui::FontId::monospace(12.0),
                    ..Default::default()
                });
            }
            ui.label(job);
        });
    }

    fn draw_trace_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("NLM Traces");
        let mut new_trace_region = None;
        ui.horizontal(|ui| {
            ui.label("Region:");
            let names: Vec<String> = self.region_names.clone();
            for (i, name) in names.iter().enumerate() {
                if ui.selectable_label(self.trace_region == i, name).clicked() {
                    new_trace_region = Some(i);
                }
            }
        });
        if let Some(r) = new_trace_region {
            self.trace_region = r;
            self.fetch_trace();
        }
        ui.separator();

        if self.trace_data.is_empty() {
            ui.label("No trace data");
            return;
        }

        ui.label(format!("d_model={} memory={}", self.trace_d_model, self.trace_memory));

        // Heatmap: rows = neurons, cols = memory positions
        let d = self.trace_d_model.max(1);
        let m = self.trace_memory.max(1);
        let cell_w = (280.0 / m as f32).min(12.0);
        let cell_h = (400.0 / d as f32).min(6.0);
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(cell_w * m as f32, cell_h * d as f32),
            egui::Sense::hover());

        let max_v = self.trace_data.iter().map(|v| v.abs()).fold(0.01f32, f32::max);
        for n in 0..d {
            for t in 0..m {
                let idx = n * m + t;
                if idx >= self.trace_data.len() { break; }
                let v = self.trace_data[idx] / max_v;
                let c = if v >= 0.0 {
                    let i = (v.clamp(0.0, 1.0) * 255.0) as u8;
                    Color32::from_rgb(i / 3, i / 2, i)
                } else {
                    let i = (v.abs().clamp(0.0, 1.0) * 255.0) as u8;
                    Color32::from_rgb(i, i / 3, i / 4)
                };
                let r = egui::Rect::from_min_size(
                    Pos2::new(rect.min.x + t as f32 * cell_w, rect.min.y + n as f32 * cell_h),
                    Vec2::new(cell_w, cell_h));
                ui.painter().rect_filled(r, 0.0, c);
            }
        }
    }

    fn draw_brain(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let center = rect.center();
        let painter = ui.painter_at(rect);

        painter.rect_filled(rect, 0.0, Color32::from_rgb(8, 8, 16));

        // Mouse drag for rotation
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        if response.dragged() {
            let d = response.drag_delta();
            self.rot_y += d.x * 0.005;
            self.rot_x += d.y * 0.005;
        }
        if let Some(scroll) = ui.input(|i| {
            if i.raw_scroll_delta.y != 0.0 { Some(i.raw_scroll_delta.y) } else { None }
        }) {
            self.zoom = (self.zoom * (1.0 + scroll * 0.001)).clamp(0.2, 5.0);
        }

        // Build activation lookup
        let get_activation = |region: usize, index: usize| -> f32 {
            self.region_activations.get(region)
                .and_then(|a| a.get(index))
                .copied()
                .unwrap_or(0.0)
        };

        // Sort neurons back-to-front
        let mut draw_list: Vec<(Pos2, f32, Color32, f32)> = Vec::new();
        for np in &self.neuron_positions {
            if !self.visible.get(np.region).copied().unwrap_or(true) { continue; }
            let act = get_activation(np.region, np.index).abs();
            if act < 0.02 { continue; }

            if let Some((screen, depth)) = self.project(np.x, np.y, np.z, center) {
                if !rect.contains(screen) { continue; }
                let size = ((0.8 + act * 1.5 * self.size_scale) * 80.0 / depth).clamp(0.3, 10.0);
                let color = self.region_color(np.region);
                let alpha = (act * 0.8).clamp(0.05, 0.9);
                let c = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), (alpha * 255.0) as u8);
                draw_list.push((screen, depth, c, size));
            }
        }
        draw_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (pos, _, color, size) in &draw_list {
            if *size > 4.0 {
                let glow = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), color.a() / 5);
                painter.circle_filled(*pos, size * 1.3, glow);
            }
            painter.circle_filled(*pos, *size, *color);
        }

        // Sync lines
        if self.show_sync && !self.global_sync.is_empty() {
            let mut sync_sorted: Vec<(usize, f32)> = self.global_sync.iter()
                .enumerate().map(|(i, &v)| (i, v.abs())).collect();
            sync_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let max_s = sync_sorted.first().map(|s| s.1).unwrap_or(1.0).max(0.01);
            let total = self.neuron_positions.len();
            if total >= 2 {
                for &(pi, strength) in sync_sorted.iter().take(30) {
                    if strength < max_s * 0.2 { break; }
                    // Map pair index to neuron positions (deterministic hash)
                    let n1 = (pi.wrapping_mul(7).wrapping_add(3)) % total;
                    let n2 = (pi.wrapping_mul(13).wrapping_add(11)) % total;
                    let p1 = &self.neuron_positions[n1];
                    let p2 = &self.neuron_positions[n2];
                    if p1.region == p2.region { continue; }
                    if !self.visible.get(p1.region).copied().unwrap_or(true) { continue; }
                    if !self.visible.get(p2.region).copied().unwrap_or(true) { continue; }

                    if let (Some((s1, _)), Some((s2, _))) = (
                        self.project(p1.x, p1.y, p1.z, center),
                        self.project(p2.x, p2.y, p2.z, center),
                    ) {
                        let i = (strength / max_s).clamp(0.0, 1.0);
                        let c1 = self.region_color(p1.region);
                        let c2 = self.region_color(p2.region);
                        let r = ((c1.r() as f32 + c2.r() as f32) / 2.0) as u8;
                        let g = ((c1.g() as f32 + c2.g() as f32) / 2.0) as u8;
                        let b = ((c1.b() as f32 + c2.b() as f32) / 2.0) as u8;
                        let lc = Color32::from_rgba_unmultiplied(r, g, b, (i * 100.0) as u8);
                        painter.line_segment([s1, s2], Stroke::new(0.5 + i * 1.5, lc));
                    }
                }
            }
        }

        painter.text(
            Pos2::new(rect.min.x + 10.0, rect.max.y - 20.0),
            egui::Align2::LEFT_BOTTOM,
            format!("{} neurons | {} regions | {} sync pairs",
                draw_list.len(), self.region_names.len(), self.n_sync),
            egui::FontId::default(),
            Color32::GRAY,
        );
    }
}

/// Map token to display character and color.
fn token_display(t: usize) -> (char, Color32) {
    use isis_runtime::regional::*;
    match t {
        0..=255 => {
            let b = t as u8;
            let ch = if b.is_ascii_graphic() || b == b' ' { b as char } else { '.' };
            (ch, Color32::WHITE)
        }
        TOKEN_IMG_START => ('[', Color32::from_rgb(0x44, 0x88, 0xff)),
        TOKEN_IMG_END => (']', Color32::from_rgb(0x44, 0x88, 0xff)),
        TOKEN_AUD_START => ('[', Color32::from_rgb(0x44, 0xff, 0x88)),
        TOKEN_AUD_END => (']', Color32::from_rgb(0x44, 0xff, 0x88)),
        TOKEN_VID_START => ('[', Color32::from_rgb(0xff, 0xff, 0x44)),
        TOKEN_VID_END => (']', Color32::from_rgb(0xff, 0xff, 0x44)),
        ACT_START => ('{', Color32::from_rgb(0xff, 0x44, 0x44)),
        ACT_END => ('}', Color32::from_rgb(0xff, 0x44, 0x44)),
        _ if t >= TOKEN_TS_OFFSET && t < TOKEN_TS_OFFSET + TOKEN_TS_COUNT => {
            ('⏱', Color32::from_rgb(0xaa, 0xaa, 0x44))
        }
        _ if t >= TOKEN_IMG_OFFSET && t < TOKEN_IMG_OFFSET + TOKEN_IMG_CODES => {
            ('▪', Color32::from_rgb(0x44, 0x88, 0xff))
        }
        _ if t >= TOKEN_AUD_OFFSET && t < TOKEN_AUD_OFFSET + TOKEN_AUD_CODES => {
            ('♪', Color32::from_rgb(0x44, 0xff, 0x88))
        }
        _ if t >= TOKEN_COORD_OFFSET && t < TOKEN_COORD_OFFSET + TOKEN_COORD_COUNT => {
            ('·', Color32::from_rgb(0xff, 0x88, 0x44))
        }
        _ => ('?', Color32::GRAY),
    }
}
