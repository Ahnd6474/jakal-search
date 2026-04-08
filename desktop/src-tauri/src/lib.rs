use serde::Deserialize;
use serde_json::{json, Value};
use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonPayload {
    query: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FeedbackPayload {
    run_id: String,
    choice: String,
}

#[tauri::command]
fn run_comparison(payload: ComparisonPayload) -> Result<Value, String> {
    run_python_bridge("compare", json!({ "query": payload.query }))
}

#[tauri::command]
fn submit_feedback(payload: FeedbackPayload) -> Result<Value, String> {
    run_python_bridge(
        "feedback",
        json!({
            "run_id": payload.run_id,
            "choice": payload.choice
        }),
    )
}

#[tauri::command]
fn get_dashboard() -> Result<Value, String> {
    run_python_bridge("dashboard", json!({}))
}

fn run_python_bridge(command_name: &str, payload: Value) -> Result<Value, String> {
    let repo_root = repo_root()?;
    let (program, prefix_args) = locate_python(&repo_root)?;
    let mut command_args = prefix_args;
    command_args.push("-m".to_string());
    command_args.push("jakal_search.gui_api".to_string());
    command_args.push(command_name.to_string());

    let mut child = Command::new(&program)
        .args(&command_args)
        .current_dir(&repo_root)
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start python bridge: {error}"))?;

    let payload_bytes =
        serde_json::to_vec(&payload).map_err(|error| format!("failed to encode payload: {error}"))?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(&payload_bytes)
            .map_err(|error| format!("failed to write python payload: {error}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|error| format!("failed waiting for python bridge: {error}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !output.status.success() {
        let reason = if stderr.is_empty() {
            stdout
        } else {
            stderr
        };
        return Err(format!("python bridge failed: {reason}"));
    }

    serde_json::from_str(&stdout).map_err(|error| format!("failed to parse python response: {error}"))
}

fn locate_python(repo_root: &Path) -> Result<(String, Vec<String>), String> {
    let mut candidates: Vec<(String, Vec<String>)> = Vec::new();
    if let Ok(explicit) = env::var("JAKAL_SEARCH_PYTHON") {
        if !explicit.trim().is_empty() {
            candidates.push((explicit, Vec::new()));
        }
    }
    candidates.push(("python".to_string(), Vec::new()));
    #[cfg(target_os = "windows")]
    candidates.push(("py".to_string(), vec!["-3".to_string()]));

    for (program, prefix_args) in candidates {
        let mut probe_args = prefix_args.clone();
        probe_args.push("--version".to_string());
        let status = Command::new(&program)
            .args(&probe_args)
            .current_dir(repo_root)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if let Ok(status) = status {
            if status.success() {
                return Ok((program, prefix_args));
            }
        }
    }

    Err("Python runtime not found. Set JAKAL_SEARCH_PYTHON to a valid interpreter.".to_string())
}

fn repo_root() -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Ok(explicit_root) = env::var("JAKAL_SEARCH_ROOT") {
        if !explicit_root.trim().is_empty() {
            candidates.push(PathBuf::from(explicit_root));
        }
    }
    if let Ok(current_dir) = env::current_dir() {
        candidates.push(current_dir);
    }
    if let Ok(current_exe) = env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            candidates.push(parent.to_path_buf());
        }
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));

    for candidate in candidates {
        if let Some(root) = find_repo_root(&candidate) {
            return Ok(root);
        }
    }

    Err("failed to locate repository root. Set JAKAL_SEARCH_ROOT to the repo path.".to_string())
}

fn find_repo_root(start: &Path) -> Option<PathBuf> {
    for directory in start.ancestors() {
        if directory.join("pyproject.toml").exists() && directory.join("jakal_search").is_dir() {
            return Some(directory.to_path_buf());
        }
    }
    None
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            run_comparison,
            submit_feedback,
            get_dashboard
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
