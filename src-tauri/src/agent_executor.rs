use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentCommand {
    pub action: String,
    pub target: Option<String>,
    pub args: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub message: String,
    pub output: Option<String>,
}

/// Parse LLM output to extract agent commands
pub fn parse_agent_response(response: &str) -> Vec<AgentCommand> {
    let mut commands = Vec::new();
    let lower_response = response.to_lowercase();
    
    // Launch application commands
    if lower_response.contains("launch") || lower_response.contains("open") || lower_response.contains("start") {
        if lower_response.contains("calculator") || lower_response.contains("calc") {
            commands.push(AgentCommand {
                action: "launch".to_string(),
                target: Some("calculator".to_string()),
                args: None,
            });
        }
        if lower_response.contains("firefox") || lower_response.contains("browser") {
            commands.push(AgentCommand {
                action: "launch".to_string(),
                target: Some("firefox".to_string()),
                args: None,
            });
        }
        if lower_response.contains("terminal") {
            commands.push(AgentCommand {
                action: "launch".to_string(),
                target: Some("terminal".to_string()),
                args: None,
            });
        }
        if lower_response.contains("text editor") || lower_response.contains("editor") || lower_response.contains("gedit") {
            commands.push(AgentCommand {
                action: "launch".to_string(),
                target: Some("gedit".to_string()),
                args: None,
            });
        }
        if lower_response.contains("file manager") || lower_response.contains("files") {
            commands.push(AgentCommand {
                action: "launch".to_string(),
                target: Some("files".to_string()),
                args: None,
            });
        }
    }
    
    // Media control commands - play/pause/next/previous
    if (lower_response.contains("play") || lower_response.contains("resume")) && !lower_response.contains("pause") {
        commands.push(AgentCommand {
            action: "media".to_string(),
            target: Some("play".to_string()),
            args: None,
        });
    }
    if lower_response.contains("pause") || lower_response.contains("stop music") || lower_response.contains("stop song") {
        commands.push(AgentCommand {
            action: "media".to_string(),
            target: Some("pause".to_string()),
            args: None,
        });
    }
    if (lower_response.contains("next") || lower_response.contains("skip")) && (lower_response.contains("song") || lower_response.contains("track") || lower_response.contains("music")) {
        commands.push(AgentCommand {
            action: "media".to_string(),
            target: Some("next".to_string()),
            args: None,
        });
    }
    if (lower_response.contains("previous") || lower_response.contains("back")) && (lower_response.contains("song") || lower_response.contains("track") || lower_response.contains("music")) {
        commands.push(AgentCommand {
            action: "media".to_string(),
            target: Some("previous".to_string()),
            args: None,
        });
    }
    
    // Volume control
    if lower_response.contains("volume up") || lower_response.contains("louder") {
        commands.push(AgentCommand {
            action: "media".to_string(),
            target: Some("volume-up".to_string()),
            args: None,
        });
    }
    if lower_response.contains("volume down") || lower_response.contains("quieter") {
        commands.push(AgentCommand {
            action: "media".to_string(),
            target: Some("volume-down".to_string()),
            args: None,
        });
    }
    
    // Switch to application
    if lower_response.contains("switch to") || lower_response.contains("focus on") || lower_response.contains("go to") {
        if lower_response.contains("spotify") {
            commands.push(AgentCommand {
                action: "switch".to_string(),
                target: Some("spotify".to_string()),
                args: None,
            });
        }
        if lower_response.contains("firefox") {
            commands.push(AgentCommand {
                action: "switch".to_string(),
                target: Some("firefox".to_string()),
                args: None,
            });
        }
        if lower_response.contains("chrome") || lower_response.contains("chromium") {
            commands.push(AgentCommand {
                action: "switch".to_string(),
                target: Some("chrome".to_string()),
                args: None,
            });
        }
        if lower_response.contains("terminal") {
            commands.push(AgentCommand {
                action: "switch".to_string(),
                target: Some("terminal".to_string()),
                args: None,
            });
        }
    }
    
    commands
}

/// Execute a single agent command
pub fn execute_command(command: &AgentCommand) -> ExecutionResult {
    match command.action.as_str() {
        "launch" => execute_launch(command),
        "media" => execute_media_control(command),
        "switch" => execute_switch_app(command),
        _ => ExecutionResult {
            success: false,
            message: format!("Unknown action: {}", command.action),
            output: None,
        },
    }
}

/// Launch an application
fn execute_launch(command: &AgentCommand) -> ExecutionResult {
    let app_name = command.target.as_ref().map(|s| s.as_str()).unwrap_or("");
    
    let launch_command = match app_name {
        "calculator" => {
            // Try common calculator apps
            vec!["gnome-calculator", "kcalc", "galculator", "qalculate-gtk"]
        },
        "firefox" => vec!["firefox"],
        "terminal" => vec!["gnome-terminal", "konsole", "xterm", "alacritty", "kitty"],
        "gedit" => vec!["gedit", "kate", "kwrite"],
        "files" => vec!["nautilus", "dolphin", "thunar", "pcmanfm"],
        _ => return ExecutionResult {
            success: false,
            message: format!("Unknown application: {}", app_name),
            output: None,
        },
    };
    
    // Try each command until one succeeds
    for cmd in launch_command {
        let result = Command::new(cmd)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        
        if result.is_ok() {
            return ExecutionResult {
                success: true,
                message: format!("✓ Launched {}", app_name),
                output: Some(format!("Started {} using {}", app_name, cmd)),
            };
        }
    }
    
    ExecutionResult {
        success: false,
        message: format!("✗ Failed to launch {}", app_name),
        output: Some("Application not found on system".to_string()),
    }
}

/// Control media playback
fn execute_media_control(command: &AgentCommand) -> ExecutionResult {
    let action = command.target.as_ref().map(|s| s.as_str()).unwrap_or("");
    
    // Try playerctl first (supports most media players)
    let playerctl_cmd = match action {
        "play" => "play",
        "pause" => "pause",
        "next" => "next",
        "previous" => "previous",
        "volume-up" => "volume",
        "volume-down" => "volume",
        _ => return ExecutionResult {
            success: false,
            message: format!("Unknown media action: {}", action),
            output: None,
        },
    };
    
    let result = if action == "volume-up" {
        Command::new("playerctl")
            .args(&["volume", "0.1+"])
            .output()
    } else if action == "volume-down" {
        Command::new("playerctl")
            .args(&["volume", "0.1-"])
            .output()
    } else {
        Command::new("playerctl")
            .arg(playerctl_cmd)
            .output()
    };
    
    if let Ok(output) = result {
        if output.status.success() {
            return ExecutionResult {
                success: true,
                message: format!("✓ Media {}", action),
                output: Some(format!("Executed media control: {}", action)),
            };
        }
    }
    
    // Fallback: try dbus for specific players
    let dbus_action = match action {
        "play" => "Play",
        "pause" => "Pause",
        "next" => "Next",
        "previous" => "Previous",
        _ => "Pause",
    };
    
    let dbus_result = Command::new("dbus-send")
        .args(&[
            "--print-reply",
            "--dest=org.mpris.MediaPlayer2.spotify",
            "/org/mpris/MediaPlayer2",
            &format!("org.mpris.MediaPlayer2.Player.{}", dbus_action),
        ])
        .output();
    
    if let Ok(output) = dbus_result {
        if output.status.success() {
            return ExecutionResult {
                success: true,
                message: format!("✓ Media {} (via dbus)", action),
                output: Some(format!("Controlled via dbus: {}", action)),
            };
        }
    }
    
    ExecutionResult {
        success: false,
        message: format!("✗ Failed to control media: {}", action),
        output: Some("playerctl or dbus not available".to_string()),
    }
}

/// Switch to an application window
fn execute_switch_app(command: &AgentCommand) -> ExecutionResult {
    let app_name = command.target.as_ref().map(|s| s.as_str()).unwrap_or("");
    
    // Use wmctrl to switch to application window
    let result = Command::new("wmctrl")
        .args(&["-a", app_name])
        .output();
    
    if let Ok(output) = result {
        if output.status.success() {
            return ExecutionResult {
                success: true,
                message: format!("✓ Switched to {}", app_name),
                output: Some(format!("Focused window: {}", app_name)),
            };
        }
    }
    
    // Fallback: try xdotool
    let xdotool_result = Command::new("bash")
        .arg("-c")
        .arg(format!("xdotool search --name '{}' windowactivate", app_name))
        .output();
    
    if let Ok(output) = xdotool_result {
        if output.status.success() {
            return ExecutionResult {
                success: true,
                message: format!("✓ Switched to {} (via xdotool)", app_name),
                output: Some(format!("Activated window: {}", app_name)),
            };
        }
    }
    
    ExecutionResult {
        success: false,
        message: format!("✗ Failed to switch to {}", app_name),
        output: Some(format!("Window not found: {}", app_name)),
    }
}

/// Execute multiple commands in sequence and return combined results
pub fn execute_agent_task(llm_response: &str) -> Vec<ExecutionResult> {
    let commands = parse_agent_response(llm_response);
    
    if commands.is_empty() {
        return vec![ExecutionResult {
            success: false,
            message: "No executable commands found in response".to_string(),
            output: Some(llm_response.to_string()),
        }];
    }
    
    commands.iter().map(|cmd| execute_command(cmd)).collect()
}
