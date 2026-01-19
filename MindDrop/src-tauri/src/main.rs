// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ai_runner;
mod system_info;
mod commands;
mod execution_config;
mod system_profiler;
mod memory_estimator;
mod auto_mode;
mod inference_runner;
mod agent_executor;

use tauri::Manager;

fn main() {
    // WebKitGTK can fail to allocate GBM buffers on some Linux/Wayland setups.
    // Disabling the DMABuf renderer is a common workaround.
    #[cfg(target_os = "linux")]
    {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            commands::get_system_info,
            commands::check_python,
            commands::save_temp_file,
            commands::list_downloaded_models,
            commands::download_model,
            commands::delete_model,
            commands::set_model_quantization,
            commands::run_inference,
            commands::run_streaming_inference,
            commands::set_gpu_settings,
            commands::generate_image,
            commands::generate_video,
            commands::search_huggingface,
            commands::search_models,
            commands::get_download_dir,
            commands::set_download_dir,
            commands::web_search,
            commands::get_system_resources,
            commands::refresh_system_resources,
            commands::get_gpu_info,
            commands::get_execution_config,
            commands::set_execution_config,
            commands::estimate_vram_usage,
            commands::get_model_info,
            commands::estimate_max_context,
            commands::decide_execution_mode,
            commands::recommend_execution_mode,
            commands::calculate_max_model_capacity,
            commands::run_inference_with_fallback,
            commands::get_running_applications,
            commands::get_agent_permissions,
            commands::save_agent_permissions,
            commands::execute_agent_commands,
            commands::execute_agent_action,
            commands::text_to_speech,
            commands::cleanup_model_memory,
        ])
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let _window = app.get_webview_window("main").unwrap();
                // Don't open devtools by default - use F12 to open manually
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
