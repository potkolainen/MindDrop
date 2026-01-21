use crate::{ai_runner, system_info, system_profiler, execution_config, memory_estimator, auto_mode, inference_runner, agent_executor};
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::env;
use tauri::Emitter;

#[derive(Debug, Serialize, Deserialize)]
pub struct PythonCheckResult {
    pub available: bool,
    pub version: Option<String>,
    pub transformers_installed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageGenerationResult {
    pub image_path: String,
    pub seed: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceSearchResult {
    pub id: String,
    pub author: Option<String>,
    pub downloads: u64,
    pub likes: u64,
    pub tags: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelSearchResult {
    pub id: String,
    pub name: String,
    pub author: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub stars: Option<u64>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub source: String,
    pub url: String,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WebSearchResult {
    pub title: String,
    pub snippet: String,
    pub url: String,
}

#[tauri::command]
pub fn get_system_info() -> system_info::SystemInfo {
    system_info::get_system_info()
}

#[tauri::command]
pub fn check_python() -> PythonCheckResult {
    let python_check = Command::new("python3")
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    if let Ok(output) = python_check {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            
            // Check if transformers is installed
            let transformers_check = Command::new("python3")
                .arg("-c")
                .arg("import transformers")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();

            let transformers_installed = transformers_check.map(|s| s.success()).unwrap_or(false);

            return PythonCheckResult {
                available: true,
                version: Some(version),
                transformers_installed,
            };
        }
    }

    PythonCheckResult {
        available: false,
        version: None,
        transformers_installed: false,
    }
}

#[tauri::command]
pub async fn save_temp_file(file_name: String, file_data: Vec<u8>) -> Result<String, String> {
    use std::io::Write;
    
    // Create temp directory for uploaded files
    let temp_dir = std::env::temp_dir().join("ai-workspace").join("uploads");
    std::fs::create_dir_all(&temp_dir)
        .map_err(|e| format!("Failed to create temp directory: {}", e))?;
    
    // Generate unique filename
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let temp_path = temp_dir.join(format!("{}_{}", timestamp, file_name));
    
    // Write file
    let mut file = std::fs::File::create(&temp_path)
        .map_err(|e| format!("Failed to create temp file: {}", e))?;
    file.write_all(&file_data)
        .map_err(|e| format!("Failed to write temp file: {}", e))?;
    
    Ok(temp_path.to_string_lossy().to_string())
}

#[tauri::command]
pub fn list_downloaded_models() -> Result<Vec<ai_runner::ModelInfo>, String> {
    ai_runner::list_models()
}

#[tauri::command]
pub async fn download_model(app: tauri::AppHandle, model_id: String, api_key: Option<String>) -> Result<(), String> {
    // Determine source from model_id prefix
    if model_id.starts_with("github/") {
        download_github_model(&model_id, api_key, app).await
    } else if model_id.starts_with("civitai/") {
        download_civitai_model(&model_id, api_key, app).await
    } else if model_id.starts_with("modelscope/") {
        download_modelscope_model(&model_id, api_key, app).await
    } else if model_id.starts_with("gitlab/") {
        download_gitlab_model(&model_id, api_key, app).await
    } else if model_id.starts_with("koboldai/") {
        download_koboldai_model(&model_id, api_key, app).await
    } else if model_id.starts_with("ollama/") {
        download_ollama_model(&model_id, api_key, app).await
    } else if model_id.starts_with("nvidia/") {
        download_nvidia_model(&model_id, api_key, app).await
    } else if model_id.starts_with("zenodo/") {
        download_zenodo_model(&model_id, api_key, app).await
    } else if model_id.starts_with("lmstudio/") {
        download_lmstudio_model(&model_id, api_key, app).await
    } else if model_id.starts_with("openai/") || model_id.starts_with("meta/") || model_id.starts_with("mistral/") 
        || model_id.starts_with("aws/") || model_id.starts_with("azure/") || model_id.starts_with("gcp/") 
        || model_id.starts_with("arxiv/") {
        Err("This model is API-based or requires platform-specific tools. Please use the provider's official tools.".to_string())
    } else if model_id.starts_with("runway/") || model_id.starts_with("paperswithcode/") {
        // These are HuggingFace proxies, download via HF
        ai_runner::download_model_python(&model_id, app)
    } else {
        // Default to HuggingFace for models without prefix
        ai_runner::download_model_python(&model_id, app)
    }
}

#[tauri::command]
pub fn delete_model(model_id: String) -> Result<(), String> {
    ai_runner::delete_model(&model_id)
}

#[tauri::command]
pub fn set_model_quantization(model_id: String, quantization: String) -> Result<(), String> {
    ai_runner::set_model_quantization(&model_id, &quantization)
}

#[tauri::command]
pub fn set_gpu_settings(gpu_threshold: f32, force_cpu: bool) -> Result<(), String> {
    env::set_var("GPU_MEMORY_THRESHOLD", gpu_threshold.to_string());
    env::set_var("FORCE_CPU_MODE", force_cpu.to_string());
    Ok(())
}

#[tauri::command]
pub async fn generate_image(
    prompt: String,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    steps: u32,
    cfg_scale: f32,
    sampler: String,
    seed: i64,
    reference_image: Option<String>,
    model_id: Option<String>, // Optional model ID to use
) -> Result<ImageGenerationResult, String> {
    use std::process::{Command, Stdio};
    use std::io::Write;
    use serde_json::json;
    
    let _ = sampler; // Suppress unused warning for now
    
    // Determine which model to use
    // Always pass the model ID (not the full path) to Python
    // Python's diffusers library will handle finding it locally or downloading
    let selected_model = if let Some(id) = model_id {
        id
    } else {
        // Default to stabilityai/stable-diffusion-2-1
        "stabilityai/stable-diffusion-2-1".to_string()
    };
    
    // Get Python executable
    let python = if cfg!(target_os = "windows") {
        "python"
    } else {
        "python3"
    };
    
    // Prepare arguments as JSON
    let args = json!({
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "model_id": selected_model,
        "reference_image": reference_image
    });
    
    // Get path to generate_image.py script
    // The script is located at workspace_root/src-tauri/generate_image.py
    // When Tauri runs, current_dir might be workspace_root or src-tauri
    let script_candidates = vec![
        // If current_dir is src-tauri, script is in current_dir
        std::env::current_dir().ok().and_then(|p| {
            let script_path = p.join("generate_image.py");
            if script_path.exists() { Some(script_path) } else { None }
        }),
        // If current_dir is workspace root, script is in src-tauri/
        std::env::current_dir().ok().and_then(|p| {
            let script_path = p.join("src-tauri").join("generate_image.py");
            if script_path.exists() { Some(script_path) } else { None }
        }),
        // Release mode: next to executable
        std::env::current_exe().ok().and_then(|p| {
            p.parent().map(|dir| dir.join("generate_image.py")).filter(|p| p.exists())
        }),
        // Alternative: go up from src-tauri if we're inside it
        std::env::current_dir().ok().and_then(|p| {
            p.parent()
                .map(|root| root.join("src-tauri").join("generate_image.py"))
                .filter(|p| p.exists())
        }),
    ];
    
    let script_path = script_candidates
        .into_iter()
        .flatten()
        .next()
        .ok_or_else(|| {
            let cwd = std::env::current_dir().unwrap_or_default();
            let searched = vec![
                cwd.join("generate_image.py"),
                cwd.join("src-tauri").join("generate_image.py"),
            ];
            format!(
                "Image generation script not found. Looked in:\n\
                - {}\n\
                - {}\n\n\
                Current directory: {}\n\n\
                Make sure generate_image.py exists in the src-tauri/ directory.",
                searched[0].display(),
                searched[1].display(),
                cwd.display()
            )
        })?;
    
    // Run Python script
    let mut child = Command::new(python)
        .arg(script_path.to_str().unwrap())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start Python: {}\n\nMake sure Python 3 is installed and in your PATH.", e))?;

    // Write JSON args to stdin
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(args.to_string().as_bytes())
            .map_err(|e| format!("Failed to write to Python stdin: {}", e))?;
    }
    
    // Wait for completion and read output
    let output = child.wait_with_output()
        .map_err(|e| format!("Failed to wait for Python process: {}", e))?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Image generation failed:\n{}", stderr));
    }
    
    // Parse JSON result
    let result_str = String::from_utf8_lossy(&output.stdout);
    let result: serde_json::Value = serde_json::from_str(&result_str)
        .map_err(|e| format!("Failed to parse result: {}\n\nOutput was: {}", e, result_str))?;
    
    if result["success"].as_bool() == Some(true) {
        Ok(ImageGenerationResult {
            image_path: result["image_path"].as_str().unwrap_or("").to_string(),
            seed: result["seed"].as_i64().unwrap_or(seed),
        })
    } else {
        Err(result["error"].as_str().unwrap_or("Unknown error").to_string())
    }
}

#[derive(serde::Serialize)]
pub struct VideoGenerationResult {
    video_path: String,
    seed: i64,
    num_frames: u32,
    fps: u32,
}

#[tauri::command]
pub async fn generate_video(
    reference_image: String,
    width: u32,
    height: u32,
    num_frames: u32,
    fps: u32,
    motion_bucket_id: u32,
    noise_aug_strength: f32,
    seed: i64,
    model_id: Option<String>,
) -> Result<VideoGenerationResult, String> {
    use std::process::{Command, Stdio};
    use std::io::Write;
    use serde_json::json;
    
    let python = if cfg!(target_os = "windows") {
        "python"
    } else {
        "python3"
    };
    
    let args = json!({
        "reference_image": reference_image,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "motion_bucket_id": motion_bucket_id,
        "noise_aug_strength": noise_aug_strength,
        "seed": seed,
        "model_id": model_id
    });
    
    // Find generate_video.py script
    let script_candidates = vec![
        std::env::current_dir().ok().and_then(|p| {
            let script_path = p.join("generate_video.py");
            if script_path.exists() { Some(script_path) } else { None }
        }),
        std::env::current_dir().ok().and_then(|p| {
            let script_path = p.join("src-tauri").join("generate_video.py");
            if script_path.exists() { Some(script_path) } else { None }
        }),
        std::env::current_exe().ok().and_then(|p| {
            p.parent().map(|dir| dir.join("generate_video.py")).filter(|p| p.exists())
        }),
        std::env::current_dir().ok().and_then(|p| {
            p.parent()
                .map(|root| root.join("src-tauri").join("generate_video.py"))
                .filter(|p| p.exists())
        }),
    ];
    
    let script_path = script_candidates
        .into_iter()
        .flatten()
        .next()
        .ok_or_else(|| {
            let cwd = std::env::current_dir().unwrap_or_default();
            format!(
                "Video generation script not found. Make sure generate_video.py exists in the src-tauri/ directory.\nCurrent directory: {}",
                cwd.display()
            )
        })?;
    
    let mut child = Command::new(python)
        .arg(script_path.to_str().unwrap())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start Python: {}\n\nMake sure Python 3 is installed.", e))?;
    
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(args.to_string().as_bytes())
            .map_err(|e| format!("Failed to write to Python stdin: {}", e))?;
    }
    
    let output = child.wait_with_output()
        .map_err(|e| format!("Failed to wait for Python process: {}", e))?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Video generation failed:\n{}", stderr));
    }
    
    let result_str = String::from_utf8_lossy(&output.stdout);
    let result: serde_json::Value = serde_json::from_str(&result_str)
        .map_err(|e| format!("Failed to parse result: {}\n\nOutput was: {}", e, result_str))?;
    
    if result["success"].as_bool() == Some(true) {
        Ok(VideoGenerationResult {
            video_path: result["video_path"].as_str().unwrap_or("").to_string(),
            seed: result["seed"].as_i64().unwrap_or(seed),
            num_frames: result["num_frames"].as_u64().unwrap_or(num_frames as u64) as u32,
            fps: result["fps"].as_u64().unwrap_or(fps as u64) as u32,
        })
    } else {
        Err(result["error"].as_str().unwrap_or("Unknown error").to_string())
    }
}

#[tauri::command]
pub async fn run_inference(
    model_id: String,
    prompt: String,
    max_tokens: usize,
) -> Result<ai_runner::InferenceResult, String> {
    eprintln!("📨 run_inference called for model: {}", model_id);
    eprintln!("   Prompt length: {} chars", prompt.len());
    eprintln!("   Max tokens: {}", max_tokens);
    
    // Load execution config for this model
    let config = get_execution_config(model_id.clone())?;
    eprintln!("   Config loaded: mode={}, quantization={}", config.mode, config.quantization);
    
    // Use inference_runner which respects execution_config settings and has fallback
    let result = inference_runner::run_with_fallback(&model_id, &prompt, max_tokens, &config).await;
    
    if let Err(ref e) = result {
        eprintln!("❌ Inference failed with error: {}", e);
    }
    
    result
}

#[tauri::command]
pub async fn run_streaming_inference(
    app: tauri::AppHandle,
    model_id: String,
    prompt: String,
    max_tokens: usize,
) -> Result<ai_runner::InferenceResult, String> {
    eprintln!("📨 run_streaming_inference called for model: {}", model_id);
    eprintln!("   Prompt length: {} chars", prompt.len());
    eprintln!("   Max tokens: {}", max_tokens);
    
    // Get GPU settings
    let gpu_threshold = env::var("GPU_MEMORY_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(8.0);
    let force_cpu = env::var("FORCE_CPU_MODE")
        .ok()
        .and_then(|s| s.parse::<bool>().ok())
        .unwrap_or(false);

    eprintln!("   GPU threshold: {}, Force CPU: {}", gpu_threshold, force_cpu);

    // Load execution config
    eprintln!("   Loading execution config...");
    let config = match get_execution_config(model_id.clone()) {
        Ok(c) => {
            eprintln!("   ✓ Config loaded: mode={}, quantization={}, gpu_layers={:?}", 
                     c.mode, c.quantization, c.gpu_layers);
            c
        },
        Err(e) => {
            eprintln!("   ❌ Failed to load config: {}", e);
            return Err(e);
        }
    };
    
    let exec_mode = format!("{:?}", config.mode).to_lowercase();
    let quantization = config.quantization.clone();
    
    eprintln!("   Calling ai_runner::run_streaming_inference...");
    let result = ai_runner::run_streaming_inference(
        &model_id,
        &prompt,
        max_tokens,
        gpu_threshold,
        force_cpu,
        &exec_mode,
        config.gpu_layers,
        &quantization,
        app,
    );
    
    eprintln!("   ✓ ai_runner returned");
    
    if let Err(ref e) = result {
        eprintln!("   ❌ Streaming inference error: {}", e);
    }
    
    result
}

#[tauri::command]
pub async fn search_models(
    source: String,
    query: String,
    filters: Option<Vec<String>>,
    sort: Option<String>,
) -> Result<Vec<ModelSearchResult>, String> {
    let filters = filters.unwrap_or_default();
    let sort = sort.unwrap_or_else(|| "trending".to_string());
    
    println!("Search models called: source={}, query={}, sort={}", source, query, sort);
    
    match source.as_str() {
        "huggingface" => search_huggingface_unified(query, filters, sort).await,
        "github" => search_github_models(query, sort).await,
        "gitlab" => search_gitlab_models(query).await,
        "openai" => search_openai_models(query).await,
        "civitai" => search_civitai_models(query, filters, sort).await,
        "modelscope" => search_modelscope_models(query).await,
        "koboldai" => search_koboldai_models(query).await,
        "paperswithcode" => search_papers_with_code(query).await,
        "zenodo" => search_zenodo_models(query).await,
        "arxiv" => search_arxiv_models(query).await,
        "lmstudio" => search_lmstudio_models(query).await,
        "ollama" => search_ollama_models(query).await,
        "nvidia" => search_nvidia_ngc_models(query).await,
        "aws" => search_aws_models(query).await,
        "azure" => search_azure_models(query).await,
        "gcp" => search_gcp_models(query).await,
        _ => Err(format!("Unknown source: {}", source)),
    }
}

async fn search_huggingface_unified(
    query: String,
    filters: Vec<String>,
    sort: String,
) -> Result<Vec<ModelSearchResult>, String> {
    // Map sort options to HuggingFace API
    let hf_sort = match sort.as_str() {
        "downloads" => "downloads",
        "likes" => "likes",
        "recent" => "lastModified",
        "alphabetical" => "id",
        _ => "downloads", // default to most downloaded for trending
    };
    
    // Build filter string
    let filter_str = if filters.is_empty() {
        String::new()
    } else {
        filters.join(",")
    };
    
    // Use full=true to get complete model info including downloads
    // If query is empty, don't include search parameter to get trending models
    let url = if query.trim().is_empty() {
        if filter_str.is_empty() {
            format!(
                "https://huggingface.co/api/models?sort={}&limit=20&full=true",
                hf_sort
            )
        } else {
            format!(
                "https://huggingface.co/api/models?filter={}&sort={}&limit=20&full=true",
                urlencoding::encode(&filter_str),
                hf_sort
            )
        }
    } else {
        if filter_str.is_empty() {
            format!(
                "https://huggingface.co/api/models?search={}&sort={}&limit=20&full=true",
                urlencoding::encode(&query),
                hf_sort
            )
        } else {
            format!(
                "https://huggingface.co/api/models?search={}&filter={}&sort={}&limit=20&full=true",
                urlencoding::encode(&query),
                urlencoding::encode(&filter_str),
                hf_sort
            )
        }
    };

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    let models: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let results = models
        .into_iter()
        .map(|m| {
            let id = m["id"].as_str().unwrap_or("").to_string();
            ModelSearchResult {
                name: id.clone(),
                id: id.clone(),
                author: m["author"].as_str().map(|s| s.to_string()),
                downloads: m["downloads"].as_u64(),
                likes: m["likes"].as_u64(),
                stars: None,
                description: None,
                tags: m["tags"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                source: "huggingface".to_string(),
                url: format!("https://huggingface.co/{}", id),
                size_bytes: None,
            }
        })
        .collect();

    Ok(results)
}

async fn search_github_models(query: String, sort: String) -> Result<Vec<ModelSearchResult>, String> {
    // Map sort options to GitHub API
    let gh_sort = match sort.as_str() {
        "downloads" | "likes" => "stars",
        "recent" => "updated",
        "alphabetical" => "name",
        _ => "stars", // default
    };
    
    // Build search query - filter for ML/AI repos
    let search_query = if query.is_empty() {
        "machine-learning OR deep-learning OR neural-network OR transformer language:Python topic:machine-learning".to_string()
    } else {
        format!("{} machine-learning OR ai-model topic:machine-learning language:Python", query)
    };
    
    let url = format!(
        "https://api.github.com/search/repositories?q={}&sort={}&per_page=20",
        urlencoding::encode(&search_query),
        gh_sort
    );

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("User-Agent", "AI-Workspace-App")
        .send()
        .await
        .map_err(|e| format!("GitHub request failed: {}", e))?;

    let data: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse GitHub response: {}", e))?;

    let items = data["items"].as_array().ok_or("No items in response")?;

    let results = items
        .iter()
        .map(|repo| {
            let name = repo["name"].as_str().unwrap_or("").to_string();
            let owner = repo["owner"]["login"].as_str().unwrap_or("").to_string();
            let full_name = repo["full_name"].as_str().unwrap_or("").to_string();
            
            ModelSearchResult {
                id: format!("github/{}", full_name),
                name: name.clone(),
                author: Some(owner),
                downloads: None,
                likes: None,
                stars: repo["stargazers_count"].as_u64(),
                description: repo["description"].as_str().map(|s| s.to_string()),
                tags: repo["topics"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                source: "github".to_string(),
                url: repo["html_url"].as_str().unwrap_or("").to_string(),
                size_bytes: repo["size"].as_u64().map(|s| s * 1024),
            }
        })
        .collect();

    Ok(results)
}

async fn search_gitlab_models(query: String) -> Result<Vec<ModelSearchResult>, String> {
    let search_term = if query.is_empty() { "machine-learning" } else { &query };
    let url = format!(
        "https://gitlab.com/api/v4/projects?search={}&topic=machine-learning&order_by=star_count&sort=desc&per_page=20",
        urlencoding::encode(search_term)
    );

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("GitLab request failed: {}", e))?;

    let projects: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse GitLab response: {}", e))?;

    let results = projects
        .into_iter()
        .map(|proj| {
            let name = proj["name"].as_str().unwrap_or("").to_string();
            let path = proj["path_with_namespace"].as_str().unwrap_or("").to_string();
            
            ModelSearchResult {
                id: format!("gitlab/{}", path),
                name: name.clone(),
                author: proj["namespace"]["name"].as_str().map(|s| s.to_string()),
                downloads: None,
                likes: None,
                stars: proj["star_count"].as_u64(),
                description: proj["description"].as_str().map(|s| s.to_string()),
                tags: proj["topics"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                source: "gitlab".to_string(),
                url: proj["web_url"].as_str().unwrap_or("").to_string(),
                size_bytes: None,
            }
        })
        .collect();

    Ok(results)
}

async fn search_openai_models(_query: String) -> Result<Vec<ModelSearchResult>, String> {
    // OpenAI models are API-based, list their available models
    let models = vec![
        ModelSearchResult {
            id: "openai/o1".to_string(),
            name: "o1".to_string(),
            author: Some("OpenAI".to_string()),
            downloads: None,
            likes: None,
            stars: None,
            description: Some("Reasoning model with enhanced problem-solving capabilities".to_string()),
            tags: vec!["reasoning".to_string(), "o-series".to_string()],
            source: "openai".to_string(),
            url: "https://platform.openai.com/docs/models/o1".to_string(),
            size_bytes: None,
        },
        ModelSearchResult {
            id: "openai/o1-mini".to_string(),
            name: "o1-mini".to_string(),
            author: Some("OpenAI".to_string()),
            downloads: None,
            likes: None,
            stars: None,
            description: Some("Faster, cost-effective reasoning model".to_string()),
            tags: vec!["reasoning".to_string(), "o-series".to_string()],
            source: "openai".to_string(),
            url: "https://platform.openai.com/docs/models/o1".to_string(),
            size_bytes: None,
        },
        ModelSearchResult {
            id: "openai/o3-mini".to_string(),
            name: "o3-mini".to_string(),
            author: Some("OpenAI".to_string()),
            downloads: None,
            likes: None,
            stars: None,
            description: Some("Advanced reasoning model (limited availability)".to_string()),
            tags: vec!["reasoning".to_string(), "o-series".to_string()],
            source: "openai".to_string(),
            url: "https://openai.com/index/early-access-to-o3-mini/".to_string(),
            size_bytes: None,
        },
        ModelSearchResult {
            id: "openai/gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            author: Some("OpenAI".to_string()),
            downloads: None,
            likes: None,
            stars: None,
            description: Some("Latest GPT-4 with multimodal capabilities and improved performance".to_string()),
            tags: vec!["gpt-4".to_string(), "multimodal".to_string(), "chat".to_string()],
            source: "openai".to_string(),
            url: "https://platform.openai.com/docs/models/gpt-4o".to_string(),
            size_bytes: None,
        },
        ModelSearchResult {
            id: "openai/gpt-4o-mini".to_string(),
            name: "GPT-4o mini".to_string(),
            author: Some("OpenAI".to_string()),
            downloads: None,
            likes: None,
            stars: None,
            description: Some("Affordable and intelligent small model for fast, lightweight tasks".to_string()),
            tags: vec!["gpt-4".to_string(), "mini".to_string(), "chat".to_string()],
            source: "openai".to_string(),
            url: "https://platform.openai.com/docs/models/gpt-4o-mini".to_string(),
            size_bytes: None,
        },
        ModelSearchResult {
            id: "openai/gpt-4-turbo".to_string(),
            name: "GPT-4 Turbo".to_string(),
            author: Some("OpenAI".to_string()),
            downloads: None,
            likes: None,
            stars: None,
            description: Some("Enhanced GPT-4 with 128K context window".to_string()),
            tags: vec!["gpt-4".to_string(), "turbo".to_string(), "chat".to_string()],
            source: "openai".to_string(),
            url: "https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4".to_string(),
            size_bytes: None,
        },
        ModelSearchResult {
            id: "openai/gpt-4".to_string(),
            name: "GPT-4".to_string(),
            author: Some("OpenAI".to_string()),
            downloads: None,
            likes: None,
            stars: None,
            description: Some("Most capable GPT-4 model, great for complex tasks".to_string()),
            tags: vec!["gpt-4".to_string(), "chat".to_string()],
            source: "openai".to_string(),
            url: "https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4".to_string(),
            size_bytes: None,
        },
        ModelSearchResult {
            id: "openai/gpt-3.5-turbo".to_string(),
            name: "GPT-3.5 Turbo".to_string(),
            author: Some("OpenAI".to_string()),
            downloads: None,
            likes: None,
            stars: None,
            description: Some("Fast and efficient model for most tasks".to_string()),
            tags: vec!["gpt-3.5".to_string(), "chat".to_string()],
            source: "openai".to_string(),
            url: "https://platform.openai.com/docs/models/gpt-3-5-turbo".to_string(),
            size_bytes: None,
        },
    ];

    Ok(models)
}

#[allow(dead_code)]
async fn search_meta_models(query: String) -> Result<Vec<ModelSearchResult>, String> {
    // Meta models are mostly on Hugging Face
    let search_term = if query.is_empty() {
        "meta-llama".to_string()
    } else {
        format!("meta-llama {}", query)
    };
    let url = format!(
        "https://huggingface.co/api/models?search={}&sort=downloads&limit=20",
        urlencoding::encode(&search_term)
    );

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    let models: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let results = models
        .into_iter()
        .map(|m| {
            let id = m["id"].as_str().unwrap_or("").to_string();
            ModelSearchResult {
                id: id.clone(),
                name: id.clone(),
                author: Some("Meta".to_string()),
                downloads: m["downloads"].as_u64(),
                likes: m["likes"].as_u64(),
                stars: None,
                description: None,
                tags: m["tags"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                source: "meta".to_string(),
                url: format!("https://huggingface.co/{}", id),
                size_bytes: None,
            }
        })
        .collect();

    Ok(results)
}

#[allow(dead_code)]
async fn search_mistral_models(query: String) -> Result<Vec<ModelSearchResult>, String> {
    // Mistral models on Hugging Face - search with author filter
    let url = format!(
        "https://huggingface.co/api/models?author=mistralai&search={}&sort=downloads&limit=20",
        urlencoding::encode(&query)
    );

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    let models: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let results = models
        .into_iter()
        .map(|m| {
            let id = m["id"].as_str().unwrap_or("").to_string();
            ModelSearchResult {
                id: id.clone(),
                name: id.clone(),
                author: Some("Mistral AI".to_string()),
                downloads: m["downloads"].as_u64(),
                likes: m["likes"].as_u64(),
                stars: None,
                description: m["description"].as_str().map(|s| s.to_string()),
                tags: m["tags"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                source: "mistral".to_string(),
                url: format!("https://huggingface.co/{}", id),
                size_bytes: None,
            }
        })
        .collect();

    Ok(results)
}

async fn search_civitai_models(query: String, filters: Vec<String>, sort: String) -> Result<Vec<ModelSearchResult>, String> {
    // Map sort options to CivitAI API
    let civit_sort = match sort.as_str() {
        "downloads" => "Most Downloaded",
        "likes" => "Most Reactions",
        "recent" => "Newest",
        "alphabetical" => "Name",
        _ => "Highest Rated", // default/trending
    };
    
    let mut url = format!(
        "https://civitai.com/api/v1/models?query={}&sort={}&limit=20",
        urlencoding::encode(&query),
        urlencoding::encode(civit_sort)
    );
    
    // Add type filter if specified
    if !filters.is_empty() {
        for filter in &filters {
            if ["checkpoint", "lora", "textual-inversion", "hypernetwork", "controlnet"].contains(&filter.as_str()) {
                url.push_str(&format!("&types={}", filter));
            }
        }
    }

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("CivitAI request failed: {}", e))?;

    let data: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse CivitAI response: {}", e))?;

    let items = data["items"].as_array().ok_or("No items in response")?;

    let results = items
        .iter()
        .map(|model| {
            let id = model["id"].as_u64().unwrap_or(0).to_string();
            let name = model["name"].as_str().unwrap_or("").to_string();
            
            ModelSearchResult {
                id: format!("civitai/{}", id),
                name: name.clone(),
                author: model["creator"]["username"].as_str().map(|s| s.to_string()),
                downloads: model["stats"]["downloadCount"].as_u64(),
                likes: model["stats"]["favoriteCount"].as_u64(),
                stars: None,
                description: model["description"].as_str().map(|s| s.to_string()),
                tags: model["tags"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                source: "civitai".to_string(),
                url: format!("https://civitai.com/models/{}", id),
                size_bytes: None,
            }
        })
        .collect();

    Ok(results)
}

async fn search_modelscope_models(query: String) -> Result<Vec<ModelSearchResult>, String> {
    // Use a default query if empty to show popular models
    let search_query = if query.trim().is_empty() {
        "transformer".to_string()
    } else {
        query
    };
    
    let url = format!(
        "https://www.modelscope.cn/api/v1/models?PageSize=20&PageNumber=1&SortBy=GmtDownload&Keyword={}",
        urlencoding::encode(&search_query)
    );

    println!("ModelScope: Fetching URL: {}", url);

    let response = reqwest::get(&url)
        .await
        .map_err(|e| {
            let err = format!("ModelScope request failed: {}", e);
            println!("{}", err);
            err
        })?;

    let status = response.status();
    println!("ModelScope: Response status: {}", status);

    let data: serde_json::Value = response
        .json()
        .await
        .map_err(|e| {
            let err = format!("Failed to parse ModelScope response: {}", e);
            println!("{}", err);
            err
        })?;

    println!("ModelScope: Response data: {:?}", data);

    let models = data["Data"]["Models"].as_array().ok_or_else(|| {
        let err = "No models in ModelScope response".to_string();
        println!("{}", err);
        err
    })?;

    println!("ModelScope: Found {} models", models.len());

    let results = models
        .iter()
        .map(|model| {
            let id = model["Path"].as_str().unwrap_or("").to_string();
            let name = model["Name"].as_str().unwrap_or("").to_string();
            
            ModelSearchResult {
                id: format!("modelscope/{}", id),
                name: name.clone(),
                author: model["User"]["Name"].as_str().map(|s| s.to_string()),
                downloads: model["DownloadCount"].as_u64(),
                likes: model["LikeCount"].as_u64(),
                stars: None,
                description: model["Description"].as_str().map(|s| s.to_string()),
                tags: model["Tags"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                source: "modelscope".to_string(),
                url: format!("https://modelscope.cn/models/{}", id),
                size_bytes: None,
            }
        })
        .collect();

    Ok(results)
}

async fn search_koboldai_models(query: String) -> Result<Vec<ModelSearchResult>, String> {
    // KoboldAI models are community hosted, search GitHub
    let url = format!(
        "https://api.github.com/search/repositories?q={} koboldai&sort=stars&per_page=20",
        urlencoding::encode(&query)
    );

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("User-Agent", "AI-Workspace-App")
        .send()
        .await
        .map_err(|e| format!("GitHub request failed: {}", e))?;

    let data: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let items = data["items"].as_array().ok_or("No items in response")?;

    let results = items
        .iter()
        .map(|repo| {
            let full_name = repo["full_name"].as_str().unwrap_or("").to_string();
            
            ModelSearchResult {
                id: format!("koboldai/{}", full_name),
                name: repo["name"].as_str().unwrap_or("").to_string(),
                author: Some(repo["owner"]["login"].as_str().unwrap_or("").to_string()),
                downloads: None,
                likes: None,
                stars: repo["stargazers_count"].as_u64(),
                description: repo["description"].as_str().map(|s| s.to_string()),
                tags: vec!["koboldai".to_string()],
                source: "koboldai".to_string(),
                url: repo["html_url"].as_str().unwrap_or("").to_string(),
                size_bytes: repo["size"].as_u64().map(|s| s * 1024),
            }
        })
        .collect();

    Ok(results)
}

#[allow(dead_code)]
async fn search_ai4science_models(query: String) -> Result<Vec<ModelSearchResult>, String> {
    // AI4Science models on Hugging Face
    let search_term = if query.is_empty() {
        "biology science protein research".to_string()
    } else {
        format!("{} science research", query)
    };
    let url = format!(
        "https://huggingface.co/api/models?search={}&sort=downloads&limit=20",
        urlencoding::encode(&search_term)
    );

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    let models: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let results = models
        .into_iter()
        .filter(|m| {
            let tags = m["tags"].as_array();
            if let Some(tags) = tags {
                tags.iter().any(|t| {
                    if let Some(tag_str) = t.as_str() {
                        tag_str.contains("science") || tag_str.contains("research") || tag_str.contains("biology") || tag_str.contains("chemistry")
                    } else {
                        false
                    }
                })
            } else {
                false
            }
        })
        .map(|m| {
            let id = m["id"].as_str().unwrap_or("").to_string();
            ModelSearchResult {
                id: id.clone(),
                name: id.clone(),
                author: m["author"].as_str().map(|s| s.to_string()),
                downloads: m["downloads"].as_u64(),
                likes: m["likes"].as_u64(),
                stars: None,
                description: None,
                tags: m["tags"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                source: "ai4science".to_string(),
                url: format!("https://huggingface.co/{}", id),
                size_bytes: None,
            }
        })
        .collect();

    Ok(results)
}

async fn search_papers_with_code(query: String) -> Result<Vec<ModelSearchResult>, String> {
    // Papers With Code models are typically on HuggingFace  
    let search_term = if query.is_empty() {
        "paper arxiv research".to_string()
    } else {
        query.clone()
    };
    let url = format!(
        "https://huggingface.co/api/models?search={}&sort=downloads&limit=20",
        urlencoding::encode(&search_term)
    );

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    let models: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let results = models
        .into_iter()
        .map(|m| {
            let id = m["id"].as_str().unwrap_or("").to_string();
            ModelSearchResult {
                id: id.clone(),
                name: id.clone(),
                author: m["author"].as_str().map(|s| s.to_string()),
                downloads: m["downloads"].as_u64(),
                likes: m["likes"].as_u64(),
                stars: None,
                description: m["description"].as_str().map(|s| s.to_string()),
                tags: m["tags"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                source: "paperswithcode".to_string(),
                url: format!("https://huggingface.co/{}", id),
                size_bytes: None,
            }
        })
        .collect();

    Ok(results)
}

#[allow(dead_code)]
async fn search_runway_models(query: String) -> Result<Vec<ModelSearchResult>, String> {
    // Runway/Stability AI models on HuggingFace
    let search_term = if query.is_empty() {
        "stabilityai stable-diffusion runway".to_string()
    } else {
        format!("{} stabilityai runway", query)
    };
    let url = format!(
        "https://huggingface.co/api/models?search={}&sort=downloads&limit=20",
        urlencoding::encode(&search_term)
    );

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    let models: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let results = models
        .into_iter()
        .filter(|m| {
            let author = m["author"].as_str().unwrap_or("");
            author.contains("runway") || author.contains("stabilityai") || author.contains("stability")
        })
        .map(|m| {
            let id = m["id"].as_str().unwrap_or("").to_string();
            ModelSearchResult {
                id: id.clone(),
                name: id.clone(),
                author: m["author"].as_str().map(|s| s.to_string()),
                downloads: m["downloads"].as_u64(),
                likes: m["likes"].as_u64(),
                stars: None,
                description: m["description"].as_str().map(|s| s.to_string()),
                tags: m["tags"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                source: "runway".to_string(),
                url: format!("https://huggingface.co/{}", id),
                size_bytes: None,
            }
        })
        .collect();

    Ok(results)
}

async fn search_zenodo_models(query: String) -> Result<Vec<ModelSearchResult>, String> {
    // Zenodo API search - using REST API v1
    let search_term = if query.trim().is_empty() {
        "(machine learning) OR (neural network) OR (deep learning)".to_string()
    } else {
        query.clone()
    };
    let url = format!(
        "https://zenodo.org/api/records?q={}&type=dataset&size=20&sort=mostviewed",
        urlencoding::encode(&search_term)
    );

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Zenodo request failed: {}", e))?;

    let data: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Zenodo response: {}", e))?;

    let hits = data["hits"]["hits"].as_array().ok_or("No results found")?;

    let results = hits
        .iter()
        .map(|record| {
            let id = record["id"].as_u64().unwrap_or(0).to_string();
            let metadata = &record["metadata"];
            
            ModelSearchResult {
                id: format!("zenodo/{}", id),
                name: metadata["title"].as_str().unwrap_or("").to_string(),
                author: metadata["creators"].as_array()
                    .and_then(|arr| arr.first())
                    .and_then(|c| c["name"].as_str())
                    .map(|s| s.to_string()),
                downloads: None,
                likes: None,
                stars: None,
                description: metadata["description"].as_str().map(|s| s.to_string()),
                tags: metadata["keywords"].as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                source: "zenodo".to_string(),
                url: format!("https://zenodo.org/records/{}", id),
                size_bytes: metadata["files"].as_array()
                    .and_then(|files| files.iter().map(|f| f["size"].as_u64().unwrap_or(0)).sum::<u64>().into()),
            }
        })
        .collect();

    Ok(results)
}

async fn search_arxiv_models(query: String) -> Result<Vec<ModelSearchResult>, String> {
    // arXiv papers - link to associated code/models
    let url = format!(
        "http://export.arxiv.org/api/query?search_query=all:{}&max_results=20&sortBy=relevance",
        urlencoding::encode(&query)
    );

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    let xml = response.text().await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    // Simple XML parsing for arXiv entries
    let mut results = Vec::new();
    
    // This is a simplified parser - in production, use proper XML parser
    for entry_text in xml.split("<entry>").skip(1) {
        if let Some(end_pos) = entry_text.find("</entry>") {
            let entry = &entry_text[..end_pos];
            
            let id = extract_xml_tag(entry, "id").unwrap_or_default();
            let title = extract_xml_tag(entry, "title").unwrap_or_default();
            let summary = extract_xml_tag(entry, "summary").unwrap_or_default();
            let author = extract_xml_tag(entry, "name").unwrap_or_default();
            
            results.push(ModelSearchResult {
                id: format!("arxiv/{}", id.split('/').last().unwrap_or(&id)),
                name: title.trim().replace('\n', " "),
                author: if author.is_empty() { None } else { Some(author) },
                downloads: None,
                likes: None,
                stars: None,
                description: Some(summary.trim().replace('\n', " ")),
                tags: vec!["paper".to_string(), "research".to_string()],
                source: "arxiv".to_string(),
                url: id.clone(),
                size_bytes: None,
            });
        }
    }

    Ok(results)
}

// Helper function for simple XML parsing
fn extract_xml_tag(xml: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{}>", tag);
    let end_tag = format!("</{}>", tag);
    
    xml.find(&start_tag).and_then(|start| {
        let content_start = start + start_tag.len();
        xml[content_start..].find(&end_tag).map(|end| {
            xml[content_start..content_start + end].to_string()
        })
    })
}

async fn search_lmstudio_models(query: String) -> Result<Vec<ModelSearchResult>, String> {
    // LM Studio uses GGUF models from HuggingFace
    // Return curated list of popular GGUF models
    Ok(get_popular_gguf_models(&query))
}

fn get_popular_gguf_models(query: &str) -> Vec<ModelSearchResult> {
    let popular_models = vec![
        // OpenAI GPT-OSS models
        ("openai/gpt-oss-120b", "GPT-OSS 120B", "OpenAI's open source reasoning model - 120B parameters, Apache 2.0 licensed"),
        ("openai/gpt-oss-20b", "GPT-OSS 20B", "OpenAI's open source reasoning model - 20B parameters, runs on 16GB RAM"),
        
        // Qwen models
        ("Qwen/Qwen2.5-Coder-32B-Instruct-GGUF", "Qwen2.5 Coder 32B", "Advanced code generation model by Qwen"),
        ("Qwen/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5 7B Instruct", "Instruction-tuned model by Qwen"),
        
        // Llama models
        ("meta-llama/Llama-3.2-3B-Instruct-GGUF", "Llama 3.2 3B Instruct", "Meta's latest compact Llama model"),
        ("meta-llama/Llama-3.2-1B-Instruct-GGUF", "Llama 3.2 1B Instruct", "Ultra-compact Llama model"),
        ("TheBloke/Llama-2-7B-Chat-GGUF", "Llama 2 7B Chat", "Meta's Llama 2 chat model in GGUF format"),
        ("TheBloke/Llama-2-13B-Chat-GGUF", "Llama 2 13B Chat", "Larger Llama 2 chat model"),
        
        // Mistral models
        ("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "Mistral 7B Instruct v0.2", "Mistral AI's instruction-tuned model"),
        ("TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF", "Mixtral 8x7B Instruct", "Mixture of experts model by Mistral AI"),
        
        // Code models
        ("TheBloke/CodeLlama-7B-Instruct-GGUF", "CodeLlama 7B Instruct", "Meta's code-specialized Llama model"),
        ("TheBloke/CodeLlama-13B-Instruct-GGUF", "CodeLlama 13B Instruct", "Larger code-specialized model"),
        
        // Other popular models
        ("microsoft/Phi-3-mini-4k-instruct-gguf", "Phi-3 Mini 4K", "Microsoft's efficient small model"),
        ("TheBloke/OpenHermes-2.5-Mistral-7B-GGUF", "OpenHermes 2.5 Mistral 7B", "Enhanced Mistral model for conversations"),
        ("TheBloke/Starling-LM-7B-alpha-GGUF", "Starling LM 7B Alpha", "High-quality instruction-following model"),
        ("TheBloke/zephyr-7B-beta-GGUF", "Zephyr 7B Beta", "Fine-tuned Mistral variant for chat"),
    ];
    
    let query_lower = query.to_lowercase();
    
    popular_models.iter()
        .filter(|(id, name, _)| {
            query.is_empty() || 
            id.to_lowercase().contains(&query_lower) ||
            name.to_lowercase().contains(&query_lower)
        })
        .map(|(id, name, description)| {
            let author = if id.starts_with("openai/") {
                "OpenAI"
            } else if id.contains("meta-llama") {
                "Meta"
            } else if id.contains("microsoft") {
                "Microsoft"
            } else if id.contains("Qwen") {
                "Qwen"
            } else {
                "TheBloke"
            };
            
            ModelSearchResult {
                id: format!("lmstudio/{}", id),
                name: name.to_string(),
                author: Some(author.to_string()),
                downloads: None,
                likes: None,
                stars: None,
                description: Some(description.to_string()),
                tags: vec!["gguf".to_string(), "quantized".to_string()],
                source: "lmstudio".to_string(),
                url: format!("https://lmstudio.ai/models/{}", id),
                size_bytes: None,
            }
        })
        .collect()
}

async fn search_ollama_models(query: String) -> Result<Vec<ModelSearchResult>, String> {
    // Search Ollama library
    let url = "https://ollama.com/api/tags";
    
    let response = reqwest::get(url)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    let data: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let models = data["models"].as_array().ok_or("No models found")?;
    
    let query_lower = query.to_lowercase();
    let results: Vec<ModelSearchResult> = models
        .iter()
        .filter(|m| {
            let name = m["name"].as_str().unwrap_or("").to_lowercase();
            query_lower.is_empty() || name.contains(&query_lower)
        })
        .map(|m| {
            let name = m["name"].as_str().unwrap_or("").to_string();
            
            ModelSearchResult {
                id: format!("ollama/{}", name),
                name: name.clone(),
                author: Some("Ollama".to_string()),
                downloads: None,
                likes: None,
                stars: None,
                description: m["description"].as_str().map(|s| s.to_string()),
                tags: vec!["ollama".to_string(), "gguf".to_string()],
                source: "ollama".to_string(),
                url: format!("https://ollama.com/library/{}", name),
                size_bytes: m["size"].as_u64(),
            }
        })
        .take(20)
        .collect();

    Ok(results)
}

async fn search_nvidia_ngc_models(query: String) -> Result<Vec<ModelSearchResult>, String> {
    // NVIDIA NGC catalog - using their public API
    let search_term = if query.trim().is_empty() { "" } else { &query };
    let url = if search_term.is_empty() {
        "https://api.ngc.nvidia.com/v2/search/catalog/resources/model?pageSize=20&orderBy=modifiedDESC".to_string()
    } else {
        format!(
            "https://api.ngc.nvidia.com/v2/search/catalog/resources/model?q={}&pageSize=20&orderBy=modifiedDESC",
            urlencoding::encode(search_term)
        )
    };

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("Accept", "application/json")
        .header("User-Agent", "AI-Workspace-App")
        .send()
        .await
        .map_err(|e| format!("NVIDIA NGC request failed: {}", e))?;

    let data: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse NGC response: {}", e))?;

    let resources = data["results"].as_array()
        .or_else(|| data["resources"].as_array())
        .ok_or("No results found")?;

    let results = resources
        .iter()
        .map(|model| {
            let name = model["resourceName"].as_str()
                .or_else(|| model["name"].as_str())
                .unwrap_or("").to_string();
            let org = model["orgName"].as_str()
                .or_else(|| model["publisher"].as_str())
                .unwrap_or("nvidia").to_string();
            
            ModelSearchResult {
                id: format!("nvidia/{}/{}", org, name),
                name: model["displayName"].as_str()
                    .or_else(|| model["name"].as_str())
                    .unwrap_or(&name).to_string(),
                author: Some(org.clone()),
                downloads: None,
                likes: None,
                stars: None,
                description: model["shortDescription"].as_str()
                    .or_else(|| model["description"].as_str())
                    .map(|s| s.to_string()),
                tags: model["tags"].as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| s.to_string())
                            .collect()
                    })
                    .unwrap_or_default(),
                source: "nvidia".to_string(),
                url: format!("https://catalog.ngc.nvidia.com/orgs/{}/models/{}", org, name),
                size_bytes: None,
            }
        })
        .collect();

    Ok(results)
}

async fn search_aws_models(_query: String) -> Result<Vec<ModelSearchResult>, String> {
    // AWS SageMaker JumpStart models
    let models = vec![
        ModelSearchResult {
            id: "aws/sagemaker-jumpstart".to_string(),
            name: "AWS SageMaker JumpStart".to_string(),
            author: Some("Amazon Web Services".to_string()),
            downloads: None,
            likes: None,
            stars: None,
            description: Some("Access SageMaker JumpStart models through AWS Console. Requires AWS account and credentials.".to_string()),
            tags: vec!["aws".to_string(), "sagemaker".to_string(), "cloud".to_string()],
            source: "aws".to_string(),
            url: "https://aws.amazon.com/sagemaker/jumpstart/".to_string(),
            size_bytes: None,
        },
    ];

    Ok(models)
}

async fn search_azure_models(_query: String) -> Result<Vec<ModelSearchResult>, String> {
    // Azure AI Model Catalog
    let models = vec![
        ModelSearchResult {
            id: "azure/model-catalog".to_string(),
            name: "Azure AI Model Catalog".to_string(),
            author: Some("Microsoft Azure".to_string()),
            downloads: None,
            likes: None,
            stars: None,
            description: Some("Browse Azure AI model catalog in Azure ML Studio. Requires Azure subscription.".to_string()),
            tags: vec!["azure".to_string(), "cloud".to_string(), "ai".to_string()],
            source: "azure".to_string(),
            url: "https://ml.azure.com/model/catalog".to_string(),
            size_bytes: None,
        },
    ];

    Ok(models)
}

async fn search_gcp_models(_query: String) -> Result<Vec<ModelSearchResult>, String> {
    // Google Cloud Vertex AI Model Garden
    let models = vec![
        ModelSearchResult {
            id: "gcp/vertex-ai".to_string(),
            name: "Vertex AI Model Garden".to_string(),
            author: Some("Google Cloud".to_string()),
            downloads: None,
            likes: None,
            stars: None,
            description: Some("Access Vertex AI Model Garden through Google Cloud Console. Requires GCP account.".to_string()),
            tags: vec!["gcp".to_string(), "vertex-ai".to_string(), "cloud".to_string()],
            source: "gcp".to_string(),
            url: "https://cloud.google.com/vertex-ai/docs/start/explore-models".to_string(),
            size_bytes: None,
        },
    ];

    Ok(models)
}

// ===== Download Functions for Different Sources =====

async fn download_github_model(model_id: &str, api_key: Option<String>, app: tauri::AppHandle) -> Result<(), String> {
    use std::fs;
    use std::io::Write;
    
    // Extract repo from model_id (format: "github/owner/repo")
    let repo_path = model_id.strip_prefix("github/")
        .ok_or("Invalid GitHub model ID")?;
    
    let model_dir = ai_runner::get_model_path(model_id);
    fs::create_dir_all(&model_dir)
        .map_err(|e| format!("Failed to create directory: {}", e))?;
    
    // Get latest release
    let release_url = format!("https://api.github.com/repos/{}/releases/latest", repo_path);
    
    let client = reqwest::Client::new();
    let mut request = client
        .get(&release_url)
        .header("User-Agent", "AI-Workspace-App");
    
    if let Some(token) = api_key {
        request = request.header("Authorization", format!("Bearer {}", token));
    }
    
    let response = request.send().await
        .map_err(|e| format!("Failed to fetch release: {}", e))?;
    
    let release: serde_json::Value = response.json().await
        .map_err(|e| format!("Failed to parse release: {}", e))?;
    
    let assets = release["assets"].as_array();
    
    // Check if there are release assets to download
    if assets.is_none() || assets.unwrap().is_empty() {
        // No release assets, clone the repository instead
        app.emit("download-progress", serde_json::json!({
            "model_id": model_id,
            "progress": 0,
            "message": "No release assets found. Cloning repository..."
        })).ok();
        
        let repo_url = format!("https://github.com/{}.git", repo_path);
        
        let output = std::process::Command::new("git")
            .args(&["clone", "--depth", "1", &repo_url, &model_dir.to_string_lossy()])
            .output()
            .map_err(|e| format!("Failed to run git clone: {}. Make sure git is installed.", e))?;
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Git clone failed: {}", error));
        }
        
        app.emit("download-progress", serde_json::json!({
            "model_id": model_id,
            "progress": 80,
            "message": "Repository cloned successfully"
        })).ok();
    } else {
        // Download each asset from release
        for asset in assets.unwrap() {
            let name = asset["name"].as_str().unwrap_or("unknown");
            let download_url = asset["browser_download_url"].as_str()
                .ok_or("No download URL found")?;
            
            app.emit("download-progress", serde_json::json!({
                "model_id": model_id,
                "progress": 0,
                "message": format!("Downloading {}...", name)
            })).ok();
            
            let file_path = model_dir.join(name);
            let mut file = fs::File::create(&file_path)
                .map_err(|e| format!("Failed to create file: {}", e))?;
            
            let bytes = client.get(download_url)
                .send().await
                .map_err(|e| format!("Failed to download file: {}", e))?
                .bytes().await
                .map_err(|e| format!("Failed to read bytes: {}", e))?;
            
            file.write_all(&bytes)
                .map_err(|e| format!("Failed to write file: {}", e))?;
            
            app.emit("download-progress", serde_json::json!({
                "model_id": model_id,
                "progress": 50,
                "message": format!("Downloaded {}", name)
            })).ok();
        }
    }
    
    // Create metadata
    let metadata = serde_json::json!({
        "model_id": model_id,
        "source": "github",
        "repo": repo_path,
        "downloaded_at": chrono::Utc::now().to_rfc3339()
    });
    
    let metadata_path = model_dir.join("model_metadata.json");
    let mut metadata_file = fs::File::create(&metadata_path)
        .map_err(|e| format!("Failed to create metadata: {}", e))?;
    metadata_file.write_all(serde_json::to_string_pretty(&metadata).unwrap().as_bytes())
        .map_err(|e| format!("Failed to write metadata: {}", e))?;
    
    app.emit("download-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 100,
        "message": "Download complete"
    })).ok();
    
    Ok(())
}

async fn download_civitai_model(model_id: &str, api_key: Option<String>, app: tauri::AppHandle) -> Result<(), String> {
    use std::fs;
    use std::io::Write;
    
    let api_key = api_key.ok_or("CivitAI API key required for downloads")?;
    
    // Extract model number from model_id (format: "civitai/12345")
    let model_number = model_id.strip_prefix("civitai/")
        .ok_or("Invalid CivitAI model ID")?;
    
    let model_dir = ai_runner::get_model_path(model_id);
    fs::create_dir_all(&model_dir)
        .map_err(|e| format!("Failed to create directory: {}", e))?;
    
    // Get model info to find latest version
    let model_url = format!("https://civitai.com/api/v1/models/{}", model_number);
    
    let client = reqwest::Client::new();
    let response = client.get(&model_url)
        .send().await
        .map_err(|e| format!("Failed to fetch model info: {}", e))?;
    
    let model_data: serde_json::Value = response.json().await
        .map_err(|e| format!("Failed to parse model info: {}", e))?;
    
    let versions = model_data["modelVersions"].as_array()
        .ok_or("No versions found")?;
    
    let latest_version = versions.first()
        .ok_or("No version available")?;
    
    let files = latest_version["files"].as_array()
        .ok_or("No files found")?;
    
    // Download primary model file
    for file in files {
        let file_name = file["name"].as_str().unwrap_or("model.safetensors");
        let download_url = file["downloadUrl"].as_str()
            .ok_or("No download URL")?;
        
        // Add API key to download URL
        let download_url_with_key = format!("{}?token={}", download_url, api_key);
        
        app.emit("download-progress", serde_json::json!({
            "model_id": model_id,
            "progress": 0,
            "message": format!("Downloading {}...", file_name)
        })).ok();
        
        let file_path = model_dir.join(file_name);
        let mut dest_file = fs::File::create(&file_path)
            .map_err(|e| format!("Failed to create file: {}", e))?;
        
        let bytes = client.get(&download_url_with_key)
            .send().await
            .map_err(|e| format!("Failed to download file: {}", e))?
            .bytes().await
            .map_err(|e| format!("Failed to read bytes: {}", e))?;
        
        dest_file.write_all(&bytes)
            .map_err(|e| format!("Failed to write file: {}", e))?;
        
        app.emit("download-progress", serde_json::json!({
            "model_id": model_id,
            "progress": 80,
            "message": format!("Downloaded {}", file_name)
        })).ok();
    }
    
    // Create metadata
    let metadata = serde_json::json!({
        "model_id": model_id,
        "source": "civitai",
        "model_name": model_data["name"].as_str().unwrap_or(""),
        "version": latest_version["name"].as_str().unwrap_or(""),
        "downloaded_at": chrono::Utc::now().to_rfc3339()
    });
    
    let metadata_path = model_dir.join("model_metadata.json");
    let mut metadata_file = fs::File::create(&metadata_path)
        .map_err(|e| format!("Failed to create metadata: {}", e))?;
    metadata_file.write_all(serde_json::to_string_pretty(&metadata).unwrap().as_bytes())
        .map_err(|e| format!("Failed to write metadata: {}", e))?;
    
    app.emit("download-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 100,
        "message": "Download complete"
    })).ok();
    
    Ok(())
}

async fn download_modelscope_model(model_id: &str, _api_key: Option<String>, _app: tauri::AppHandle) -> Result<(), String> {
    // ModelScope uses git-based downloads similar to HuggingFace
    // For now, return error suggesting alternative
    Err(format!("ModelScope downloads not yet implemented. Please download {} manually from https://modelscope.cn", model_id))
}

async fn download_gitlab_model(model_id: &str, api_key: Option<String>, app: tauri::AppHandle) -> Result<(), String> {
    use std::fs;
    use std::io::Write;
    
    // Extract project path from model_id (format: "gitlab/owner/project")
    let project_path = model_id.strip_prefix("gitlab/")
        .ok_or("Invalid GitLab model ID")?;
    
    let model_dir = ai_runner::get_model_path(model_id);
    fs::create_dir_all(&model_dir)
        .map_err(|e| format!("Failed to create directory: {}", e))?;
    
    // URL-encode the project path
    let encoded_path = urlencoding::encode(project_path);
    
    // Get project releases
    let releases_url = format!("https://gitlab.com/api/v4/projects/{}/releases", encoded_path);
    
    let client = reqwest::Client::new();
    let mut request = client.get(&releases_url);
    
    if let Some(token) = api_key {
        request = request.header("PRIVATE-TOKEN", token);
    }
    
    let response = request.send().await
        .map_err(|e| format!("Failed to fetch releases: {}", e))?;
    
    let releases: Vec<serde_json::Value> = response.json().await
        .map_err(|e| format!("Failed to parse releases: {}", e))?;
    
    let latest_release = releases.first()
        .ok_or("No releases found")?;
    
    let assets = latest_release["assets"]["links"].as_array()
        .ok_or("No assets found")?;
    
    if assets.is_empty() {
        return Err("No downloadable files in latest release".to_string());
    }
    
    // Download first asset (main model file)
    let asset = &assets[0];
    let name = asset["name"].as_str().unwrap_or("model");
    let download_url = asset["url"].as_str()
        .ok_or("No download URL")?;
    
    app.emit("download-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 0,
        "message": format!("Downloading {}...", name)
    })).ok();
    
    let file_path = model_dir.join(name);
    let mut file = fs::File::create(&file_path)
        .map_err(|e| format!("Failed to create file: {}", e))?;
    
    let bytes = client.get(download_url)
        .send().await
        .map_err(|e| format!("Failed to download: {}", e))?
        .bytes().await
        .map_err(|e| format!("Failed to read bytes: {}", e))?;
    
    file.write_all(&bytes)
        .map_err(|e| format!("Failed to write file: {}", e))?;
    
    // Create metadata
    let metadata = serde_json::json!({
        "model_id": model_id,
        "source": "gitlab",
        "project": project_path,
        "downloaded_at": chrono::Utc::now().to_rfc3339()
    });
    
    let metadata_path = model_dir.join("model_metadata.json");
    let mut metadata_file = fs::File::create(&metadata_path)
        .map_err(|e| format!("Failed to create metadata: {}", e))?;
    metadata_file.write_all(serde_json::to_string_pretty(&metadata).unwrap().as_bytes())
        .map_err(|e| format!("Failed to write metadata: {}", e))?;
    
    app.emit("download-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 100,
        "message": "Download complete"
    })).ok();
    
    Ok(())
}

async fn download_koboldai_model(model_id: &str, api_key: Option<String>, app: tauri::AppHandle) -> Result<(), String> {
    // KoboldAI models are GitHub repos, so use GitHub download
    let github_id = model_id.replace("koboldai/", "github/");
    download_github_model(&github_id, api_key, app).await
}

async fn download_ollama_model(model_id: &str, _api_key: Option<String>, app: tauri::AppHandle) -> Result<(), String> {
    use std::fs;
    use std::io::Write;
    
    // Extract model name from model_id (format: "ollama/modelname")
    let model_name = model_id.strip_prefix("ollama/")
        .ok_or("Invalid Ollama model ID")?;
    
    // Check if Ollama is installed and running
    let check = std::process::Command::new("ollama")
        .arg("list")
        .output();
    
    match check {
        Err(_) => {
            return Err(format!(
                "Ollama is not installed or not in PATH.\n\nTo use Ollama models:\n1. Install Ollama from https://ollama.com\n2. Or use the Ollama app to download: {}",
                model_name
            ));
        }
        Ok(output) if !output.status.success() => {
            let error = String::from_utf8_lossy(&output.stderr);
            if error.contains("connect") || error.contains("connection") {
                return Err(format!(
                    "Ollama server is not running.\n\nPlease start Ollama by running:\n  ollama serve\n\nOr use the Ollama app to download: {}",
                    model_name
                ));
            }
            return Err(format!("Ollama error: {}", error));
        }
        _ => {}
    }
    
    let model_dir = ai_runner::get_model_path(model_id);
    fs::create_dir_all(&model_dir)
        .map_err(|e| format!("Failed to create directory: {}", e))?;
    
    app.emit("download-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 0,
        "message": "Downloading via Ollama CLI..."
    })).ok();
    
    // Use Ollama CLI to download
    let output = std::process::Command::new("ollama")
        .args(&["pull", model_name])
        .output()
        .map_err(|e| format!("Failed to run ollama command: {}", e))?;
    
    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Ollama pull failed: {}\n\nTip: You can also use the Ollama app to download this model.", error));
    }
    
    // Create metadata
    let metadata = serde_json::json!({
        "model_id": model_id,
        "source": "ollama",
        "model_name": model_name,
        "downloaded_at": chrono::Utc::now().to_rfc3339(),
        "note": "Model managed by Ollama CLI"
    });
    
    let metadata_path = model_dir.join("model_metadata.json");
    let mut metadata_file = fs::File::create(&metadata_path)
        .map_err(|e| format!("Failed to create metadata: {}", e))?;
    metadata_file.write_all(serde_json::to_string_pretty(&metadata).unwrap().as_bytes())
        .map_err(|e| format!("Failed to write metadata: {}", e))?;
    
    app.emit("download-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 100,
        "message": "Download complete via Ollama"
    })).ok();
    
    Ok(())
}

async fn download_lmstudio_model(model_id: &str, _api_key: Option<String>, app: tauri::AppHandle) -> Result<(), String> {
    // LM Studio models are GGUF files from HuggingFace
    // Format: "lmstudio/publisher/model" -> extract "publisher/model" as HuggingFace repo
    let hf_repo = model_id.strip_prefix("lmstudio/")
        .ok_or("Invalid LM Studio model ID")?;
    
    // Use the Python-based HuggingFace downloader which handles auth and resuming
    // Note: The Python downloader will emit progress with hf_repo as model_id
    ai_runner::download_model_python(hf_repo, app)
}

async fn download_nvidia_model(model_id: &str, api_key: Option<String>, app: tauri::AppHandle) -> Result<(), String> {
    use std::fs;
    use std::io::Write;
    
    let api_key = api_key.ok_or("NVIDIA NGC API key required for downloads")?;
    
    // Extract org/model from model_id (format: "nvidia/org/model")
    let path = model_id.strip_prefix("nvidia/")
        .ok_or("Invalid NVIDIA model ID")?;
    
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() != 2 {
        return Err("Invalid NVIDIA model ID format. Expected: nvidia/org/model".to_string());
    }
    
    let (org, model_name) = (parts[0], parts[1]);
    
    let model_dir = ai_runner::get_model_path(model_id);
    fs::create_dir_all(&model_dir)
        .map_err(|e| format!("Failed to create directory: {}", e))?;
    
    app.emit("download-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 0,
        "message": "Fetching model info from NGC..."
    })).ok();
    
    // Get model details
    let details_url = format!("https://api.ngc.nvidia.com/v2/org/{}/models/{}", org, model_name);
    
    let client = reqwest::Client::new();
    let response = client
        .get(&details_url)
        .header("Authorization", format!("ApiKey {}", api_key))
        .send().await
        .map_err(|e| format!("Failed to fetch model details: {}", e))?;
    
    let model_data: serde_json::Value = response.json().await
        .map_err(|e| format!("Failed to parse model details: {}", e))?;
    
    // Create metadata (NGC models are typically pulled via NGC CLI)
    let metadata = serde_json::json!({
        "model_id": model_id,
        "source": "nvidia",
        "org": org,
        "model_name": model_name,
        "downloaded_at": chrono::Utc::now().to_rfc3339(),
        "note": format!("Use NGC CLI to download: ngc registry model download-version {}/{}", org, model_name),
        "details": model_data
    });
    
    let metadata_path = model_dir.join("model_metadata.json");
    let mut metadata_file = fs::File::create(&metadata_path)
        .map_err(|e| format!("Failed to create metadata: {}", e))?;
    metadata_file.write_all(serde_json::to_string_pretty(&metadata).unwrap().as_bytes())
        .map_err(|e| format!("Failed to write metadata: {}", e))?;
    
    app.emit("download-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 100,
        "message": "Metadata saved. Use NGC CLI for full download."
    })).ok();
    
    Ok(())
}

async fn download_zenodo_model(model_id: &str, _api_key: Option<String>, app: tauri::AppHandle) -> Result<(), String> {
    use std::fs;
    use std::io::Write;
    
    // Extract record ID from model_id (format: "zenodo/12345")
    let record_id = model_id.strip_prefix("zenodo/")
        .ok_or("Invalid Zenodo model ID")?;
    
    let model_dir = ai_runner::get_model_path(model_id);
    fs::create_dir_all(&model_dir)
        .map_err(|e| format!("Failed to create directory: {}", e))?;
    
    app.emit("download-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 0,
        "message": "Fetching record from Zenodo..."
    })).ok();
    
    // Get record details
    let record_url = format!("https://zenodo.org/api/records/{}", record_id);
    
    let response = reqwest::get(&record_url)
        .await
        .map_err(|e| format!("Failed to fetch record: {}", e))?;
    
    let record: serde_json::Value = response.json().await
        .map_err(|e| format!("Failed to parse record: {}", e))?;
    
    let files = record["files"].as_array()
        .ok_or("No files in record")?;
    
    if files.is_empty() {
        return Err("No downloadable files in this record".to_string());
    }
    
    let client = reqwest::Client::new();
    
    // Download first file
    let file = &files[0];
    let file_name = file["key"].as_str().unwrap_or("file");
    let download_url = file["links"]["self"].as_str()
        .ok_or("No download link found")?;
    
    app.emit("download-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 25,
        "message": format!("Downloading {}...", file_name)
    })).ok();
    
    let file_path = model_dir.join(file_name);
    let mut dest_file = fs::File::create(&file_path)
        .map_err(|e| format!("Failed to create file: {}", e))?;
    
    let bytes = client.get(download_url)
        .send().await
        .map_err(|e| format!("Failed to download: {}", e))?
        .bytes().await
        .map_err(|e| format!("Failed to read bytes: {}", e))?;
    
    dest_file.write_all(&bytes)
        .map_err(|e| format!("Failed to write file: {}", e))?;
    
    // Create metadata
    let metadata = serde_json::json!({
        "model_id": model_id,
        "source": "zenodo",
        "record_id": record_id,
        "title": record["metadata"]["title"].as_str().unwrap_or(""),
        "downloaded_at": chrono::Utc::now().to_rfc3339()
    });
    
    let metadata_path = model_dir.join("model_metadata.json");
    let mut metadata_file = fs::File::create(&metadata_path)
        .map_err(|e| format!("Failed to create metadata: {}", e))?;
    metadata_file.write_all(serde_json::to_string_pretty(&metadata).unwrap().as_bytes())
        .map_err(|e| format!("Failed to write metadata: {}", e))?;
    
    app.emit("download-progress", serde_json::json!({
        "model_id": model_id,
        "progress": 100,
        "message": "Download complete"
    })).ok();
    
    Ok(())
}

#[tauri::command]
pub async fn search_huggingface(query: String) -> Result<Vec<HuggingFaceSearchResult>, String> {
    let url = format!(
        "https://huggingface.co/api/models?search={}&filter=text-generation&sort=downloads&limit=20",
        urlencoding::encode(&query)
    );

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    let models: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let results = models
        .into_iter()
        .map(|m| HuggingFaceSearchResult {
            id: m["id"].as_str().unwrap_or("").to_string(),
            author: m["author"].as_str().map(|s| s.to_string()),
            downloads: m["downloads"].as_u64().unwrap_or(0),
            likes: m["likes"].as_u64().unwrap_or(0),
            tags: m["tags"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.to_string())
                        .collect()
                })
                .unwrap_or_default(),
        })
        .collect();

    Ok(results)
}

#[tauri::command]
pub fn get_download_dir() -> String {
    ai_runner::get_model_cache_dir()
        .to_string_lossy()
        .to_string()
}

#[tauri::command]
pub fn set_download_dir(path: String) -> Result<(), String> {
    ai_runner::set_download_dir(&path)
}

#[tauri::command]
pub async fn web_search(query: String, engine: String) -> Result<Vec<WebSearchResult>, String> {
    let python_script = format!(
        r#"
import json
import urllib.request
import urllib.parse
import html
import re

query = {}
engine = {}

def search_duckduckgo(q):
    encoded = urllib.parse.quote(q)
    url = f"https://duckduckgo.com/lite/?q={{encoded}}"
    headers = {{
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://duckduckgo.com/'
    }}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        return []
    
    results = []
    # DDG HTML format: <a rel="nofollow" class="result__a" href="URL">TITLE</a>
    for match in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', content):
        url_match = match.group(1)
        title_match = match.group(2)
        
        # Extract actual URL from DDG redirect: //duckduckgo.com/l/?uddg=ACTUALURL
        clean_url = html.unescape(url_match).strip()
        if 'uddg=' in clean_url:
            # Extract the actual URL from the redirect
            uddg_match = re.search(r'uddg=([^&]+)', clean_url)
            if uddg_match:
                clean_url = urllib.parse.unquote(uddg_match.group(1))
        
        # Find snippet after this link
        snippet_match = ""
        after_content = content[match.end():match.end()+500]
        snippet_search = re.search(r'class="result__snippet"[^>]*>([^<]+)</a>', after_content)
        if snippet_search:
            snippet_match = snippet_search.group(1)
        
        clean_title = html.unescape(title_match).strip()
        clean_snippet = html.unescape(snippet_match).strip() if snippet_match else "No description available"
        
        if clean_title and clean_url and len(results) < 5:
            results.append({{
                "title": clean_title[:200],
                "snippet": f"[via DuckDuckGo] {{clean_snippet[:280]}}",
                "url": clean_url
            }})
    
    return results if results else [{{
        "title": f"Search results for: {{q}}",
        "snippet": "No detailed results found. Try rephrasing your query.",
        "url": f"https://duckduckgo.com/?q={{urllib.parse.quote(q)}}"
    }}]

def search_google(q):
    # Google requires API access - scraping is blocked
    encoded = urllib.parse.quote(q)
    url = f"https://www.google.com/search?q={{encoded}}"
    headers = {{
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
        
        results = []
        # Try to extract any external links
        all_links = re.findall(r'<a[^>]+href="(https?://[^"]+)"', content)
        external = [link for link in all_links if 'google.com' not in link and 'gstatic.com' not in link][:10]
        
        for i, link in enumerate(external[:5]):
            results.append({{
                "title": f"Google result {{i+1}}",
                "snippet": "[via Google Search]",
                "url": link
            }})
        
        return results
    except:
        return []

def search_brave(q):
    encoded = urllib.parse.quote(q)
    url = f"https://search.brave.com/search?q={{encoded}}"
    headers = {{
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        return []
    
    results = []
    # Brave uses data-pos attribute for result ordering - look for any link with http
    all_links = re.findall(r'href="(https?://[^"]+)"', content)
    all_titles = re.findall(r'<span[^>]*class="[^"]*snippet-title[^"]*"[^>]*>([^<]+)</span>', content)
    
    # Filter out Brave's own links
    external_links = [link for link in all_links if 'brave.com' not in link and 'brave-search' not in link][:10]
    
    for i, link in enumerate(external_links[:5]):
        title = all_titles[i] if i < len(all_titles) else "Result"
        clean_title = html.unescape(title).strip()
        
        results.append({{
            "title": clean_title[:200] if clean_title else f"Brave Result {{i+1}}",
            "snippet": "[via Brave Search] Result from Brave",
            "url": link
        }})
    
    return results

def search_bing(q):
    encoded = urllib.parse.quote(q)
    url = f"https://www.bing.com/search?q={{encoded}}"
    headers = {{
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        return []
    
    results = []
    # Bing: Find all h2 links
    h2_links = re.findall(r'<h2[^>]*><a[^>]+href="(https?://[^"]+)"', content)
    
    # Get all external links as backup
    all_links = re.findall(r'href="(https?://[^"]+)"', content)
    external = [link for link in all_links if 'bing.com' not in link and 'microsoft.com' not in link and 'msn.com' not in link][:10]
    
    # Use h2 links if available, otherwise external links
    links_to_use = h2_links if h2_links else external
    
    for i, link in enumerate(links_to_use[:5]):
        results.append({{
            "title": f"Bing result {{i+1}}",
            "snippet": "[via Bing] Search result",
            "url": link
        }})
    
    return results

def search_ecosia(q):
    encoded = urllib.parse.quote(q)
    url = f"https://www.ecosia.org/search?q={{encoded}}&method=index"
    headers = {{
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
        
        results = []
        # Ecosia uses specific result structure
        all_links = re.findall(r'href="(https?://[^"]+)"', content)
        # Filter out Ecosia's own links
        external = [link for link in all_links if 'ecosia.org' not in link and 'bing.com' not in link][:10]
        
        for i, link in enumerate(external[:5]):
            results.append({{
                "title": f"Ecosia result {{i+1}}",
                "snippet": "[via Ecosia] Plant trees while searching",
                "url": link
            }})
        
        return results
    except:
        return []

def search_arxiv(q):
    # arXiv has an API - use it for better results
    encoded = urllib.parse.quote(q)
    url = f"http://export.arxiv.org/api/query?search_query=all:{{encoded}}&start=0&max_results=5"
    headers = {{'User-Agent': 'Mozilla/5.0'}}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
        
        results = []
        # Parse arXiv XML/Atom feed
        entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)
        
        for entry in entries[:5]:
            title_match = re.search(r'<title>([^<]+)</title>', entry)
            link_match = re.search(r'<id>(http://arxiv.org/abs/[^<]+)</id>', entry)
            summary_match = re.search(r'<summary>([^<]{{0,200}})', entry)
            
            if title_match and link_match:
                title = html.unescape(title_match.group(1)).strip()
                url = link_match.group(1)
                snippet = summary_match.group(1).strip() if summary_match else "Academic paper on arXiv"
                
                results.append({{
                    "title": title[:200],
                    "snippet": f"[via arXiv] {{snippet[:250]}}",
                    "url": url
                }})
        
        return results
    except Exception as e:
        return []

def search_devdocs(q):
    # DevDocs search - look for documentation
    encoded = urllib.parse.quote(q)
    url = f"https://devdocs.io/#q={{encoded}}"
    headers = {{'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}}
    
    # DevDocs is client-side rendered, so let's use their docs directly
    # Search common documentation sites instead
    results = []
    
    # Try MDN for web-related queries
    try:
        mdn_url = f"https://developer.mozilla.org/en-US/search?q={{encoded}}"
        req = urllib.request.Request(mdn_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode('utf-8')
        
        mdn_links = re.findall(r'href="(/en-US/docs/[^"]+)"', content)
        for link in mdn_links[:3]:
            results.append({{
                "title": f"MDN: {{link.split('/')[-1].replace('_', ' ')}}",
                "snippet": "[via DevDocs] Mozilla Developer Network documentation",
                "url": f"https://developer.mozilla.org{{link}}"
            }})
    except:
        pass
    
    return results

def search_metager(q):
    encoded = urllib.parse.quote(q)
    url = f"https://metager.org/meta/meta.ger3?eingabe={{encoded}}&focus=web"
    headers = {{
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        return []
    
    results = []
    # MetaGer aggregates results from multiple engines
    all_links = re.findall(r'href="(https?://[^"]+)"', content)
    external = [link for link in all_links if 'metager.org' not in link][:10]
    
    for i, link in enumerate(external[:5]):
        results.append({{
            "title": f"MetaGer result {{i+1}}",
            "snippet": "[via MetaGer] Privacy-focused meta-search",
            "url": link
        }})
    
    return results

def search_google_scholar(q):
    # Google Scholar with HTML parsing attempt
    encoded = urllib.parse.quote(q)
    url = f"https://scholar.google.com/scholar?q={{encoded}}"
    headers = {{
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
        
        results = []
        # Extract any external links
        all_links = re.findall(r'<a[^>]+href="(https?://[^"]+)"', content)
        external = [link for link in all_links if 'scholar.google' not in link and 'google.com' not in link][:10]
        
        for i, link in enumerate(external[:5]):
            results.append({{
                "title": f"Scholar result {{i+1}}",
                "snippet": "[via Google Scholar]",
                "url": link
            }})
        
        return results
    except:
        return []

def search_startpage(q):
    encoded = urllib.parse.quote(q)
    url = f"https://www.startpage.com/sp/search?query={{encoded}}"
    headers = {{
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        return []
    
    results = []
    # Startpage returns anonymous results
    all_links = re.findall(r'href="(https?://[^"]+)"', content)
    external = [link for link in all_links if 'startpage.com' not in link and 'ixquick.com' not in link][:10]
    
    for i, link in enumerate(external[:5]):
        results.append({{
            "title": f"Startpage result {{i+1}}",
            "snippet": "[via Startpage] Privacy-focused results",
            "url": link
        }})
    
    return results

def search_qwant(q):
    # Qwant API is blocked, use HTML scraping
    encoded = urllib.parse.quote(q)
    url = f"https://www.qwant.com/?q={{encoded}}&t=web"
    headers = {{
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
        
        results = []
        # Extract external links (not qwant.com or internal)
        links = re.findall(r'href="(https?://[^"]+)"', content)
        external_links = [link for link in links if 'qwant.com' not in link and 'qwantify.com' not in link]
        
        for i, link in enumerate(external_links[:5]):
            results.append({{
                "title": link.split('/')[2] if len(link.split('/')) > 2 else link,
                "snippet": f"[via Qwant] {{link}}",
                "url": link
            }})
        
        return results
    except:
        return []

def search_stackoverflow(q):
    # Use site-specific search on DuckDuckGo for better results
    encoded = urllib.parse.quote(f"site:stackoverflow.com {{q}}")
    url = f"https://html.duckduckgo.com/html/?q={{encoded}}"
    headers = {{
        'User-Agent': 'Mozilla/5.0'
    }}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
        
        results = []
        # Extract URLs from DDG results
        uddg_links = re.findall(r'//duckduckgo\\.com/l/\\?uddg=([^&"]+)', content)
        
        for encoded_url in uddg_links[:5]:
            url = urllib.parse.unquote(encoded_url)
            if 'stackoverflow.com/questions/' in url:
                title = url.split('/')[-1].replace('-', ' ')[:200] if url.split('/')[-1] else 'StackOverflow'
                results.append({{
                    "title": title,
                    "snippet": f"[via StackOverflow] {{title}}",
                    "url": url
                }})
        
        return results
    except:
        return []

def search_github(q):
    # Use github keyword search on DuckDuckGo for better results
    encoded = urllib.parse.quote(f"{{q}} github")
    url = f"https://html.duckduckgo.com/html/?q={{encoded}}"
    headers = {{
        'User-Agent': 'Mozilla/5.0'
    }}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8')
        
        results = []
        # Extract URLs from DDG results
        uddg_links = re.findall(r'//duckduckgo\\.com/l/\\?uddg=([^&"]+)', content)
        
        for encoded_url in uddg_links[:20]:
            url = urllib.parse.unquote(encoded_url)
            if 'github.com/' in url and url.count('/') >= 4:
                # Extract repo name from URL
                parts = url.replace('https://github.com/', '').strip('/').split('/')
                if len(parts) >= 2 and parts[0] and parts[1] and parts[0] not in ['topics', 'search', 'orgs']:
                    repo_name = f"{{parts[0]}}/{{parts[1]}}"
                    if repo_name not in [r['title'] for r in results]:  # Avoid duplicates
                        results.append({{
                            "title": repo_name,
                            "snippet": f"[via GitHub] Repository: {{repo_name}}",
                            "url": f"https://github.com/{{repo_name}}"
                        }})
                        if len(results) >= 5:
                            break
        
        return results
    except:
        return []

def search_wikipedia(q):
    encoded = urllib.parse.quote(q)
    url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={{encoded}}&limit=5&format=json"
    headers = {{
        'User-Agent': 'Mozilla/5.0'
    }}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        results = []
        if len(data) >= 4 and isinstance(data[1], list) and isinstance(data[3], list):
            titles = data[1]
            urls = data[3]
            
            for i in range(min(len(titles), len(urls), 5)):
                if urls[i]:
                    results.append({{
                        "title": titles[i],
                        "snippet": f"[via Wikipedia] {{titles[i]}}",
                        "url": urls[i]
                    }})
        
        return results
    except:
        return []

try:
    # Map engine IDs to search functions
    engine_map = {{
        'brave': search_brave,
        'bing': search_bing,
        'ecosia': search_ecosia,
        'qwant': search_qwant,
        'duckduckgo': search_duckduckgo,
        'stackoverflow': search_stackoverflow,
        'github': search_github,
        'startpage': search_startpage,
        'google-scholar': search_google_scholar,
        'devdocs': search_devdocs,
        'arxiv': search_arxiv,
        'metager': search_metager,
        'wikipedia': search_wikipedia
    }}
    
    search_func = engine_map.get(engine, search_duckduckgo)
    
    # Add debug output to stderr so it shows in console but doesn't break JSON parsing
    import sys
    print(f"Using search engine: {{engine}}", file=sys.stderr)
    
    results = search_func(query)
    
    # Add engine name to first result for verification
    if results and len(results) > 0:
        print(f"Got {{len(results)}} results from {{engine}}", file=sys.stderr)
    
    print(json.dumps(results))
except Exception as e:
    print(json.dumps([{{"title": "Search Error", "snippet": f"Search failed: {{str(e)}}", "url": ""}}]))
"#,
        serde_json::to_string(&query).map_err(|e| format!("JSON error: {}", e))?,
        serde_json::to_string(&engine).map_err(|e| format!("JSON error: {}", e))?
    );

    let output = Command::new("python3")
        .arg("-c")
        .arg(&python_script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to execute search: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Search failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        return Err("No output from search script".to_string());
    }

    let results: Vec<WebSearchResult> = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse search results: {}", e))?;

    Ok(results)
}

// ============================================================================
// System Resource Profiling Commands
// ============================================================================

/// Get current system resources (RAM, VRAM, CPU)
/// Uses 5-second caching to avoid overhead
#[tauri::command]
pub fn get_system_resources() -> Result<system_profiler::SystemResources, String> {
    system_profiler::get_system_resources()
}

/// Force refresh system resources (bypass cache)
#[tauri::command]
pub fn refresh_system_resources() -> Result<system_profiler::SystemResources, String> {
    system_profiler::refresh_system_resources()
}

/// Get just GPU info (backward compatibility)
#[tauri::command]
pub fn get_gpu_info() -> Result<system_profiler::GPUInfo, String> {
    system_profiler::get_gpu_info()
}

// ============================================================================
// Execution Config Commands
// ============================================================================

/// Get execution config for a specific model
#[tauri::command]
pub fn get_execution_config(model_id: String) -> Result<execution_config::ExecutionConfig, String> {
    // Load from model metadata file
    let download_dir = ai_runner::get_model_cache_dir();
    let safe_name = model_id.replace("/", "--");
    let metadata_path = download_dir.join(&safe_name).join("model_metadata.json");
    
    if metadata_path.exists() {
        let content = std::fs::read_to_string(&metadata_path)
            .map_err(|e| format!("Failed to read metadata: {}", e))?;
        
        let metadata: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse metadata: {}", e))?;
        
        let mode_str = metadata.get("execution_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("auto");
        
        let mode = match mode_str {
            "gpu" => execution_config::ExecutionMode::GPU,
            "hybrid" => execution_config::ExecutionMode::Hybrid,
            "cpu" => execution_config::ExecutionMode::CPU,
            _ => execution_config::ExecutionMode::Auto,
        };
        
        let gpu_layers = metadata.get("gpu_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        
        let quantization = metadata.get("quantization")
            .and_then(|v| v.as_str())
            .unwrap_or("native")
            .to_string();
        
        Ok(execution_config::ExecutionConfig {
            mode,
            gpu_layers,
            quantization,
            is_auto: mode_str == "auto",
        })
    } else {
        // Return default if metadata doesn't exist
        Ok(execution_config::ExecutionConfig::default())
    }
}

/// Set execution config for a specific model
#[tauri::command]
pub fn set_execution_config(
    model_id: String,
    config: execution_config::ExecutionConfig,
) -> Result<(), String> {
    // Validate config
    config.validate()?;
    
    // Save to model metadata file
    let download_dir = ai_runner::get_model_cache_dir();
    let safe_name = model_id.replace("/", "--");
    let metadata_path = download_dir.join(&safe_name).join("model_metadata.json");
    
    // Read existing metadata
    let mut metadata: serde_json::Value = if metadata_path.exists() {
        let content = std::fs::read_to_string(&metadata_path)
            .map_err(|e| format!("Failed to read metadata: {}", e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse metadata: {}", e))?
    } else {
        serde_json::json!({})
    };
    
    // Update execution config fields
    let mode_str = match config.mode {
        execution_config::ExecutionMode::Auto => "auto",
        execution_config::ExecutionMode::GPU => "gpu",
        execution_config::ExecutionMode::Hybrid => "hybrid",
        execution_config::ExecutionMode::CPU => "cpu",
    };
    
    metadata["execution_mode"] = serde_json::json!(mode_str);
    metadata["gpu_layers"] = match config.gpu_layers {
        Some(layers) => serde_json::json!(layers),
        None => serde_json::Value::Null,
    };
    
    // Write back to file
    let content = serde_json::to_string_pretty(&metadata)
        .map_err(|e| format!("Failed to serialize metadata: {}", e))?;
    
    std::fs::write(&metadata_path, content)
        .map_err(|e| format!("Failed to write metadata: {}", e))?;
    
    println!("Updated execution config for {}: {:?}", model_id, config);
    
    Ok(())
}

// ============================================================================
// Memory Estimation Commands
// ============================================================================

/// Estimate VRAM usage for a model configuration
#[tauri::command]
pub fn estimate_vram_usage(
    model_id: String,
    quantization: String,
    gpu_layers: Option<u32>,
    context_length: u32,
) -> Result<memory_estimator::MemoryEstimate, String> {
    let params = memory_estimator::get_model_params(&model_id);
    let total_layers = memory_estimator::get_model_layers(&model_id);
    let hidden_size = memory_estimator::get_hidden_size(&model_id);
    let gpu_layers = gpu_layers.unwrap_or(total_layers);
    
    let estimate = memory_estimator::estimate_vram_usage(
        params,
        &quantization,
        total_layers,
        gpu_layers,
        context_length,
        hidden_size,
    );
    
    Ok(estimate)
}

/// Get model information (params, layers, hidden size)
#[tauri::command]
pub fn get_model_info(model_id: String) -> serde_json::Value {
    serde_json::json!({
        "params": memory_estimator::get_model_params(&model_id),
        "layers": memory_estimator::get_model_layers(&model_id),
        "hidden_size": memory_estimator::get_hidden_size(&model_id),
    })
}

/// Estimate maximum safe context length for available VRAM
#[tauri::command]
pub fn estimate_max_context(
    model_id: String,
    quantization: String,
    gpu_layers: Option<u32>,
) -> Result<u32, String> {
    let sys_resources = system_profiler::get_system_resources()?;
    let gpu = sys_resources.gpu.ok_or("No GPU available")?;
    
    let total_layers = memory_estimator::get_model_layers(&model_id);
    let gpu_layers = gpu_layers.unwrap_or(total_layers);
    
    let max_context = memory_estimator::estimate_max_context_length(
        &model_id,
        &quantization,
        gpu_layers,
        gpu.free_vram_mb,
    );
    
    Ok(max_context)
}

// ============================================================================
// Auto Mode Decision Commands
// ============================================================================

/// Decide optimal execution mode for a model
#[tauri::command]
pub fn decide_execution_mode(
    model_id: String,
    quantization: String,
    context_length: u32,
) -> Result<auto_mode::AutoModeDecision, String> {
    auto_mode::decide_execution_mode(&model_id, &quantization, context_length)
}

/// Quick recommendation with default context (2048 tokens)
#[tauri::command]
pub fn recommend_execution_mode(
    model_id: String,
    quantization: String,
) -> Result<auto_mode::AutoModeDecision, String> {
    auto_mode::recommend_mode(&model_id, &quantization)
}

/// Calculate maximum theoretical model capacity
#[tauri::command]
pub fn calculate_max_model_capacity(
    quantization: String,
) -> Result<auto_mode::MaxModelCapacity, String> {
    auto_mode::calculate_max_model_capacity(&quantization)
}

// ============================================================================
// Inference Runner Commands (with retry/fallback)
// ============================================================================

/// Run inference with automatic OOM retry and fallback
#[tauri::command]
pub async fn run_inference_with_fallback(
    model_id: String,
    prompt: String,
    max_tokens: usize,
    config: execution_config::ExecutionConfig,
) -> Result<ai_runner::InferenceResult, String> {
    inference_runner::run_with_fallback(&model_id, &prompt, max_tokens, &config).await
}

// ============================================================================
// Agent Mode Commands
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct RunningApplication {
    pub name: String,
    pub pid: u32,
    pub icon: Option<String>,
}

/// Get list of running applications
#[tauri::command]
pub fn get_running_applications() -> Result<Vec<RunningApplication>, String> {
    #[cfg(target_os = "linux")]
    {
        // Use wmctrl to get actual GUI windows instead of all processes
        let wmctrl_output = Command::new("wmctrl")
            .args(&["-lp"])
            .stdout(Stdio::piped())
            .output();
        
        let mut apps = Vec::new();
        let mut seen = std::collections::HashSet::new();
        
        if let Ok(output) = wmctrl_output {
            let stdout = String::from_utf8_lossy(&output.stdout);
            
            for line in stdout.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let Ok(pid) = parts[2].parse::<u32>() {
                        let proc_path = format!("/proc/{}/comm", pid);
                        if let Ok(name_raw) = std::fs::read_to_string(&proc_path) {
                            let name = name_raw.trim();
                            if !name.is_empty() {
                                let key = name.to_lowercase();
                                if !seen.contains(&key) {
                                    seen.insert(key);
                                    apps.push(RunningApplication {
                                        name: name.to_string(),
                                        pid,
                                        icon: get_app_icon(name),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Fallback: filter ps output more strictly for GUI apps
            let output = Command::new("ps")
                .args(&["aux"])
                .stdout(Stdio::piped())
                .output()
                .map_err(|e| format!("Failed to get running apps: {}", e))?;
            
            let stdout = String::from_utf8_lossy(&output.stdout);
            
            // List of common GUI application patterns
            let gui_patterns = vec![
                "firefox", "chrome", "chromium", "brave", "opera", "edge",
                "code", "vscode", "sublime", "atom", "pycharm", "intellij",
                "slack", "discord", "telegram", "signal",
                "spotify", "vlc", "mpv",
                "gimp", "inkscape", "blender",
                "libreoffice", "okular", "evince",
                "nautilus", "dolphin", "thunar", "pcmanfm",
                "terminal", "konsole", "gnome-terminal", "alacritty", "kitty",
                "kwrite", "kate", "gedit", "mousepad", "pluma", "geany", "nano", "vim", "emacs",
            ];
            
            for line in stdout.lines().skip(1) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 11 {
                    if let Ok(pid) = parts[1].parse::<u32>() {
                        let cmd = parts[10..].join(" ");
                        let name = cmd.split('/').last().unwrap_or(&cmd).split_whitespace().next().unwrap_or(&cmd);
                        
                        // Only include if matches GUI patterns
                        let name_lower = name.to_lowercase();
                        let is_gui_app = gui_patterns.iter().any(|pattern| name_lower.contains(pattern));
                        
                        if is_gui_app && !name.starts_with('[') && !name.is_empty() {
                            let key = name_lower;
                            if !seen.contains(&key) {
                                seen.insert(key.clone());
                            apps.push(RunningApplication {
                                name: name.to_string(),
                                pid,
                                icon: get_app_icon(name),
                            });
                            }
                        }
                    }
                }
            }
        }
        
        // Sort by name, case-insensitive
        apps.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        Ok(apps)
    }
    
    #[cfg(target_os = "windows")]
    {
        let output = Command::new("tasklist")
            .stdout(Stdio::piped())
            .output()
            .map_err(|e| format!("Failed to get running apps: {}", e))?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut apps = Vec::new();
        let mut seen = std::collections::HashSet::new();
        
        for line in stdout.lines().skip(3) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let name = parts[0].trim_end_matches(".exe");
                if let Ok(pid) = parts[1].parse::<u32>() {
                    if !seen.contains(name) {
                        seen.insert(name.to_string());
                        apps.push(RunningApplication {
                            name: name.to_string(),
                            pid,
                            icon: get_app_icon(name),
                        });
                    }
                }
            }
        }
        
        apps.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(apps.into_iter().take(50).collect())
    }
    
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("ps")
            .args(&["-axo", "pid,comm"])
            .stdout(Stdio::piped())
            .output()
            .map_err(|e| format!("Failed to get running apps: {}", e))?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut apps = Vec::new();
        let mut seen = std::collections::HashSet::new();
        
        for line in stdout.lines().skip(1) {
            let parts: Vec<&str> = line.trim().splitn(2, ' ').collect();
            if parts.len() == 2 {
                if let Ok(pid) = parts[0].parse::<u32>() {
                    let name = parts[1].split('/').last().unwrap_or(parts[1]);
                    if !seen.contains(name) {
                        seen.insert(name.to_string());
                        apps.push(RunningApplication {
                            name: name.to_string(),
                            pid,
                            icon: get_app_icon(name),
                        });
                    }
                }
            }
        }
        
        apps.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(apps.into_iter().take(50).collect())
    }
}

fn get_app_icon(name: &str) -> Option<String> {
    // Simple icon mapping for common applications
    let icon = match name.to_lowercase().as_str() {
        n if n.contains("firefox") => "🦊",
        n if n.contains("chrome") => "🌐",
        n if n.contains("code") || n.contains("vscode") => "💻",
        n if n.contains("terminal") || n.contains("konsole") || n.contains("gnome-terminal") => "⌨️",
        n if n.contains("nautilus") || n.contains("dolphin") || n.contains("explorer") => "📁",
        n if n.contains("slack") => "💬",
        n if n.contains("discord") => "🎮",
        n if n.contains("spotify") => "🎵",
        n if n.contains("vlc") => "🎬",
        n if n.contains("gimp") || n.contains("inkscape") => "🎨",
        _ => "📱",
    };
    Some(icon.to_string())
}

/// Get saved agent permissions
#[tauri::command]
pub fn get_agent_permissions() -> Result<serde_json::Value, String> {
    let config_dir = dirs::config_dir()
        .ok_or("Could not find config directory")?
        .join("ai-workspace-tauri");
    
    std::fs::create_dir_all(&config_dir)
        .map_err(|e| format!("Failed to create config dir: {}", e))?;
    
    let permissions_file = config_dir.join("agent_permissions.json");
    
    if permissions_file.exists() {
        let content = std::fs::read_to_string(&permissions_file)
            .map_err(|e| format!("Failed to read permissions: {}", e))?;
        
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse permissions: {}", e))
    } else {
        Ok(serde_json::json!({
            "enabled": false,
            "selectedApps": {}
        }))
    }
}

/// Save agent permissions
#[tauri::command]
pub fn save_agent_permissions(permissions: serde_json::Value) -> Result<(), String> {
    let config_dir = dirs::config_dir()
        .ok_or("Could not find config directory")?
        .join("ai-workspace-tauri");
    
    std::fs::create_dir_all(&config_dir)
        .map_err(|e| format!("Failed to create config dir: {}", e))?;
    
    let permissions_file = config_dir.join("agent_permissions.json");
    
    let content = serde_json::to_string_pretty(&permissions)
        .map_err(|e| format!("Failed to serialize permissions: {}", e))?;
    
    std::fs::write(&permissions_file, content)
        .map_err(|e| format!("Failed to write permissions: {}", e))?;
    
    Ok(())
}

/// Execute agent commands from LLM response
#[tauri::command]
pub fn execute_agent_commands(llm_response: String) -> Vec<agent_executor::ExecutionResult> {
    agent_executor::execute_agent_task(&llm_response)
}

/// Execute a direct agent action without LLM parsing
#[tauri::command]
pub fn execute_agent_action(action: String, target: Option<String>) -> agent_executor::ExecutionResult {
    let command = agent_executor::AgentCommand {
        action,
        target,
        args: None,
    };
    agent_executor::execute_command(&command)
}

/// Check which diffuser backends are installed
#[tauri::command]
pub async fn check_diffuser_backends() -> Result<Vec<String>, String> {
    let mut installed = Vec::new();
    
    // Check for diffusers Python library
    if Command::new("python3")
        .args(&["-c", "import diffusers"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        installed.push("diffusers".to_string());
    }
    
    // Check for AUTOMATIC1111 (look for webui.py in common locations)
    let webui_paths = vec![
        dirs::home_dir().map(|p| p.join("stable-diffusion-webui")),
        std::env::current_dir().ok().map(|p| p.join("stable-diffusion-webui")),
    ];
    
    for path in webui_paths.into_iter().flatten() {
        if path.join("webui.py").exists() {
            installed.push("automatic1111".to_string());
            break;
        }
    }
    
    // Check for ComfyUI
    let comfy_paths = vec![
        dirs::home_dir().map(|p| p.join("ComfyUI")),
        std::env::current_dir().ok().map(|p| p.join("ComfyUI")),
    ];
    
    for path in comfy_paths.into_iter().flatten() {
        if path.join("main.py").exists() {
            installed.push("comfyui".to_string());
            break;
        }
    }
    
    // Check for InvokeAI
    if Command::new("python3")
        .args(&["-c", "import invokeai"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        installed.push("invokeai".to_string());
    }
    
    Ok(installed)
}

/// Install a diffuser backend
#[tauri::command]
pub async fn install_diffuser_backend(
    backend_id: String,
    install_cmd: String,
    repo: String,
) -> Result<String, String> {
    let _ = repo; // May use later for cloning
    
    match backend_id.as_str() {
        "diffusers" | "invokeai" => {
            // Python package installation
            let output = Command::new("sh")
                .arg("-c")
                .arg(&install_cmd)
                .output()
                .map_err(|e| format!("Failed to run pip install: {}", e))?;
            
            if output.status.success() {
                Ok(format!("Successfully installed {}", backend_id))
            } else {
                Err(String::from_utf8_lossy(&output.stderr).to_string())
            }
        }
        "automatic1111" | "comfyui" => {
            // Git clone installation
            let home = dirs::home_dir().ok_or("Cannot find home directory")?;
            
            let output = Command::new("sh")
                .arg("-c")
                .arg(&install_cmd)
                .current_dir(&home)
                .output()
                .map_err(|e| format!("Failed to clone repository: {}", e))?;
            
            if output.status.success() {
                Ok(format!("Successfully cloned {} to {}", backend_id, home.display()))
            } else {
                Err(String::from_utf8_lossy(&output.stderr).to_string())
            }
        }
        _ => Err(format!("Unknown backend: {}", backend_id))
    }
}

/// Uninstall a diffuser backend
#[tauri::command]
pub async fn uninstall_diffuser_backend(backend_id: String) -> Result<String, String> {
    match backend_id.as_str() {
        "diffusers" => {
            let output = Command::new("pip")
                .args(&["uninstall", "-y", "diffusers", "transformers", "accelerate"])
                .output()
                .map_err(|e| format!("Failed to run pip uninstall: {}", e))?;
            
            if output.status.success() {
                Ok("Successfully uninstalled diffusers".to_string())
            } else {
                Err(String::from_utf8_lossy(&output.stderr).to_string())
            }
        }
        "invokeai" => {
            let output = Command::new("pip")
                .args(&["uninstall", "-y", "invokeai"])
                .output()
                .map_err(|e| format!("Failed to run pip uninstall: {}", e))?;
            
            if output.status.success() {
                Ok("Successfully uninstalled invokeai".to_string())
            } else {
                Err(String::from_utf8_lossy(&output.stderr).to_string())
            }
        }
        "automatic1111" => {
            let home = dirs::home_dir().ok_or("Cannot find home directory")?;
            let webui_path = home.join("stable-diffusion-webui");
            
            if webui_path.exists() {
                std::fs::remove_dir_all(&webui_path)
                    .map_err(|e| format!("Failed to remove directory: {}", e))?;
                Ok(format!("Successfully removed {}", webui_path.display()))
            } else {
                Err("AUTOMATIC1111 directory not found".to_string())
            }
        }
        "comfyui" => {
            let home = dirs::home_dir().ok_or("Cannot find home directory")?;
            let comfy_path = home.join("ComfyUI");
            
            if comfy_path.exists() {
                std::fs::remove_dir_all(&comfy_path)
                    .map_err(|e| format!("Failed to remove directory: {}", e))?;
                Ok(format!("Successfully removed {}", comfy_path.display()))
            } else {
                Err("ComfyUI directory not found".to_string())
            }
        }
        _ => Err(format!("Unknown backend: {}", backend_id))
    }
}

#[tauri::command]
pub async fn text_to_speech(
    text: String, 
    voice: Option<String>, 
    speed: Option<f32>,
    pitch: Option<f32>,
    prosody: Option<f32>,
    stability: Option<f32>,
    settings: Option<String>
) -> Result<Vec<u8>, String> {
    use tokio::process::Command;
    use std::path::PathBuf;
    
    let voice_id = voice.unwrap_or_else(|| "female_neutral".to_string());
    let speed_value = speed.unwrap_or(1.0);
    let pitch_value = pitch.unwrap_or(1.0);
    let prosody_value = prosody.unwrap_or(0.5);
    let stability_value = stability.unwrap_or(0.7);
    let settings_json = settings.unwrap_or_else(|| "{}".to_string());
    
    // Get the Python script path (avoid double src-tauri when run from src-tauri)
    let script_path = if cfg!(debug_assertions) {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("f5tts_generate.py")
    } else {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
            .join("f5tts_generate.py")
    };
    
    // Parse settings JSON
    let settings_obj: serde_json::Value = serde_json::from_str(&settings_json)
        .unwrap_or(serde_json::json!({}));
    
    // Prepare JSON input for Python script
    let input_json = serde_json::json!({
        "text": text,
        "voice": voice_id,
        "speed": speed_value,
        "pitch": pitch_value,
        "prosody": prosody_value,
        "stability": stability_value,
        "settings": settings_obj
    });
    
    // Run Python script
    let output = Command::new("python3")
        .arg(&script_path)
        .arg(input_json.to_string())
        .output()
        .await
        .map_err(|e| format!("Failed to run F5-TTS script: {}", e))?;
    
    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr);
        return Err(format!("F5-TTS generation failed: {}", error));
    }
    
    // Parse output
    let result: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse TTS output: {}", e))?;
    
    if result["status"] != "success" {
        return Err(result["message"].as_str().unwrap_or("Unknown error").to_string());
    }
    
    // Read the generated audio file
    let audio_path = result["audio_path"].as_str()
        .ok_or("No audio path in response")?;
    
    let audio_bytes = tokio::fs::read(audio_path)
        .await
        .map_err(|e| format!("Failed to read audio file: {}", e))?;
    
    // Clean up temp file
    let _ = tokio::fs::remove_file(audio_path).await;
    
    Ok(audio_bytes)
}

#[tauri::command]
pub async fn cleanup_model_memory() -> Result<String, String> {
    use std::process::Command;
    
    // Kill any lingering Python inference processes
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("pkill")
            .args(&["-f", "AutoModelForCausalLM"])
            .output();
    }
    
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("pkill")
            .args(&["-f", "AutoModelForCausalLM"])
            .output();
    }
    
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("taskkill")
            .args(&["/F", "/FI", "WINDOWTITLE eq python*"])
            .output();
    }
    
    Ok("Memory cleanup initiated".to_string())
}

// Assistive Search: Browser window management
use std::sync::Mutex;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

static SEARCH_WINDOW: Mutex<Option<String>> = Mutex::new(None);

#[tauri::command]
pub async fn open_assistive_search_window(app: tauri::AppHandle) -> Result<(), String> {
    // Close existing search window if any
    let _ = close_search_window(app.clone()).await;
    
    let window_label = "search-window";
    
    // Inline minimal HTML for testing
    let html_content = r#"<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Test</title></head>
<body style="background:#1a1a2e;color:#fff;font-family:system-ui;padding:20px;">
<h1>🔍 Assistive Search</h1>
<select id="engine" style="background:#1a1a24;color:#e0e0e0;padding:12px;border:2px solid #666;border-radius:8px;font-size:14px;">
<option style="background:#0f0f14;color:#e0e0e0;">🦆 DuckDuckGo</option>
<option style="background:#0f0f14;color:#e0e0e0;">🦁 Brave</option>
</select>
<input id="query" type="text" placeholder="Search..." style="background:#1a1a24;color:#e0e0e0;padding:12px;border:2px solid #666;border-radius:8px;margin:10px;width:300px;font-size:14px;">
<button id="btn" style="background:#667eea;color:#fff;padding:12px 24px;border:none;border-radius:8px;cursor:pointer;font-size:14px;">Search</button>
<p id="status" style="margin-top:20px;"></p>
<script>
const btn = document.getElementById('btn');
const status = document.getElementById('status');
const query = document.getElementById('query');
console.log('Script loaded!');
status.textContent = 'Ready - check console (F12)';
btn.onclick = async () => {
  console.log('Button clicked!', query.value);
  status.textContent = 'Button clicked! Query: ' + query.value;
  try {
    const url = 'https://duckduckgo.com/?q=' + encodeURIComponent(query.value);
    await window.__TAURI__.core.invoke('open_search_window', { url });
    status.textContent = '✓ Opened: ' + url;
  } catch(e) {
    console.error(e);
    status.textContent = '❌ Error: ' + e;
  }
};
</script>
</body></html>"#;
    
    let data_url = format!("data:text/html;base64,{}", base64::Engine::encode(&base64::engine::general_purpose::STANDARD, html_content.as_bytes()));
    
    let _window = WebviewWindowBuilder::new(
        &app,
        window_label,
        WebviewUrl::External(data_url.parse().map_err(|e| format!("Invalid URL: {}", e))?)
    )
    .title("Assistive Search")
    .inner_size(1200.0, 800.0)
    .resizable(true)
    .build()
    .map_err(|e| format!("Failed to create search window: {}", e))?;
    
    *SEARCH_WINDOW.lock().unwrap() = Some(window_label.to_string());
    
    Ok(())
}

#[tauri::command]
pub async fn open_search_window(app: tauri::AppHandle, url: String) -> Result<(), String> {
    // Close existing search window if any
    let _ = close_search_window(app.clone()).await;
    
    let window_label = "search-window";
    
    let _window = WebviewWindowBuilder::new(
        &app,
        window_label,
        WebviewUrl::External(url.parse().map_err(|e| format!("Invalid URL: {}", e))?)
    )
    .title("Assistive Search")
    .inner_size(1200.0, 800.0)
    .resizable(true)
    .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    .initialization_script(r#"
        // Add floating control panel to the page after it loads
        setTimeout(() => {
            if (document.getElementById('assistive-search-panel')) return;
            
            const panel = document.createElement('div');
            panel.id = 'assistive-search-panel';
            panel.innerHTML = `
                <div style="position: fixed; bottom: 20px; right: 20px; z-index: 2147483647; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); font-family: system-ui, -apple-system, sans-serif; min-width: 320px;">
                    <div style="color: white; font-weight: 600; margin-bottom: 12px; font-size: 14px; display: flex; align-items: center; gap: 8px;">
                        🔍 Assistive Search Controls
                    </div>
                    <button id="summarize-btn-assist" style="width: 100%; padding: 10px; background: white; color: #667eea; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; margin-bottom: 8px; font-size: 13px; transition: transform 0.2s;">
                        📄 Summarize This Page
                    </button>
                    <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 12px; backdrop-filter: blur(10px);">
                        <input id="custom-prompt-assist" type="text" placeholder="Enter custom prompt..." style="width: 100%; padding: 8px; border: 1px solid rgba(255,255,255,0.3); border-radius: 6px; background: rgba(255,255,255,0.9); margin-bottom: 8px; font-size: 13px; box-sizing: border-box;" />
                        <button id="send-prompt-btn-assist" style="width: 100%; padding: 10px; background: white; color: #667eea; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 13px; transition: transform 0.2s;">
                            ✨ Send to Chat
                        </button>
                    </div>
                    <button id="close-window-btn-assist" style="width: 100%; padding: 8px; background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.3); border-radius: 8px; cursor: pointer; margin-top: 8px; font-size: 12px; transition: all 0.2s;">
                        ✕ Close Window
                    </button>
                </div>
            `;
            document.body.appendChild(panel);
            
            // Add hover effects
            ['summarize-btn-assist', 'send-prompt-btn-assist', 'close-window-btn-assist'].forEach(id => {
                const btn = document.getElementById(id);
                if (btn) {
                    btn.addEventListener('mouseenter', () => btn.style.transform = 'translateY(-2px)');
                    btn.addEventListener('mouseleave', () => btn.style.transform = 'translateY(0)');
                }
            });
            
            // Extract page content function
            function extractPageContent() {
                try {
                    const article = document.querySelector('article') || 
                                   document.querySelector('main') || 
                                   document.querySelector('[role="main"]') ||
                                   document.querySelector('.content') ||
                                   document.querySelector('#content') ||
                                   document.body;
                    
                    const clone = article.cloneNode(true);
                    clone.querySelectorAll('script, style, nav, footer, aside, .ad, .advertisement, .sidebar, header, [role="navigation"], [role="banner"]').forEach(el => el.remove());
                    
                    const bodyText = clone.innerText || clone.textContent || '';
                    const title = document.title || '';
                    const url = window.location.href;
                    
                    return { title, url, content: bodyText.trim().slice(0, 50000) };
                } catch (e) {
                    return { title: document.title, url: window.location.href, content: document.body.innerText.slice(0, 50000) };
                }
            }
            
            // Summarize button
            document.getElementById('summarize-btn-assist')?.addEventListener('click', async () => {
                try {
                    if (typeof window.__TAURI__ === 'undefined') {
                        alert('Tauri API not available. Please restart the search window.');
                        return;
                    }
                    const data = extractPageContent();
                    const btn = document.getElementById('summarize-btn-assist');
                    btn.textContent = '⏳ Processing...';
                    btn.disabled = true;
                    
                    await window.__TAURI__.core.invoke('send_page_to_chat', { 
                        pageData: JSON.stringify(data),
                        action: 'summarize' 
                    });
                    
                    btn.textContent = '✓ Sent to Chat!';
                    setTimeout(() => {
                        btn.textContent = '📄 Summarize This Page';
                        btn.disabled = false;
                    }, 2000);
                } catch (e) {
                    console.error('Summarize error:', e);
                    alert('Error: ' + (e.message || e));
                    const btn = document.getElementById('summarize-btn-assist');
                    btn.textContent = '📄 Summarize This Page';
                    btn.disabled = false;
                }
            });
            
            // Custom prompt button  
            document.getElementById('send-prompt-btn-assist')?.addEventListener('click', async () => {
                try {
                    if (typeof window.__TAURI__ === 'undefined') {
                        alert('Tauri API not available. Please restart the search window.');
                        return;
                    }
                    const prompt = document.getElementById('custom-prompt-assist').value;
                    if (!prompt.trim()) {
                        alert('Please enter a prompt first');
                        return;
                    }
                    const data = extractPageContent();
                    const btn = document.getElementById('send-prompt-btn-assist');
                    btn.textContent = '⏳ Sending...';
                    btn.disabled = true;
                    
                    await window.__TAURI__.core.invoke('send_page_to_chat', { 
                        pageData: JSON.stringify(data),
                        action: 'custom',
                        customPrompt: prompt
                    });
                    
                    document.getElementById('custom-prompt-assist').value = '';
                    btn.textContent = '✓ Sent to Chat!';
                    setTimeout(() => {
                        btn.textContent = '✨ Send to Chat';
                        btn.disabled = false;
                    }, 2000);
                } catch (e) {
                    console.error('Custom prompt error:', e);
                    alert('Error: ' + (e.message || e));
                    const btn = document.getElementById('send-prompt-btn-assist');
                    btn.textContent = '✨ Send to Chat';
                    btn.disabled = false;
                }
            });
            
            // Close window button
            document.getElementById('close-window-btn-assist')?.addEventListener('click', async () => {
                try {
                    if (typeof window.__TAURI__ === 'undefined') {
                        window.close(); // Fallback to browser close
                        return;
                    }
                    await window.__TAURI__.core.invoke('close_search_window');
                } catch (e) {
                    console.error('Close window error:', e);
                    window.close(); // Fallback to browser close
                }
            });
        }, 2000); // Increased timeout from 1000ms to 2000ms for slower pages
    "#)
    .build()
    .map_err(|e| format!("Failed to create search window: {}", e))?;
    
    *SEARCH_WINDOW.lock().unwrap() = Some(window_label.to_string());
    
    Ok(())
}

#[tauri::command]
pub async fn extract_page_content(app: tauri::AppHandle) -> Result<String, String> {
    let window_label = SEARCH_WINDOW.lock().unwrap().clone()
        .ok_or("No search window open")?;
    
    let window = app.get_webview_window(&window_label)
        .ok_or("Search window not found")?;
    
    // Execute JavaScript to extract page content and get the result
    window.eval(r#"
        window.__TAURI_INVOKE__('page_content_extracted', { content: (function() {
            // Try to get main content, fallback to body
            const article = document.querySelector('article') || 
                           document.querySelector('main') || 
                           document.querySelector('.content') ||
                           document.querySelector('#content') ||
                           document.body;
            
            // Remove script, style, nav, footer, ads
            const clone = article.cloneNode(true);
            clone.querySelectorAll('script, style, nav, footer, aside, .ad, .advertisement, .sidebar').forEach(el => el.remove());
            
            // Get text content
            return clone.innerText || clone.textContent || '';
        })() });
    "#)
    .map_err(|e| format!("Failed to extract content: {}", e))?;
    
    // For now, return a placeholder. In production, we'd listen for the 'page_content_extracted' event
    // and return the actual content. For simplicity, we'll extract using a different approach:
    
    // Alternative: use document.body.innerText directly
    let content = format!("Content extraction successful. In production, implement event-based content retrieval.");
    Ok(content)
}

#[tauri::command]
pub async fn close_search_window(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(label) = SEARCH_WINDOW.lock().unwrap().take() {
        if let Some(window) = app.get_webview_window(&label) {
            window.close().map_err(|e| format!("Failed to close window: {}", e))?;
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn send_page_to_chat(
    app: tauri::AppHandle, 
    page_data: String, 
    action: String, 
    custom_prompt: Option<String>
) -> Result<(), String> {
    // Parse the page data
    #[derive(serde::Deserialize)]
    struct PageData {
        title: String,
        url: String,
        content: String,
    }
    
    let data: PageData = serde_json::from_str(&page_data)
        .map_err(|e| format!("Failed to parse page data: {}", e))?;
    
    // Emit event to main window with the page content and action
    app.emit("page-content-ready", serde_json::json!({
        "title": data.title,
        "url": data.url,
        "content": data.content,
        "action": action,
        "customPrompt": custom_prompt
    }))
    .map_err(|e| format!("Failed to emit event: {}", e))?;
    
    Ok(())
}

// Plugin System
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PluginManifest {
    pub name: String,
    pub version: Option<String>,
    pub author: Option<String>,
    pub description: Option<String>,
    #[serde(rename = "type")]
    pub plugin_type: String, // "python", "node", "http", "executable"
    pub entry: String,
    pub inputs: Option<Vec<String>>,
    pub outputs: Option<Vec<String>>,
    pub permissions: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Plugin {
    pub id: String,
    pub path: String,
    pub enabled: bool,
    #[serde(flatten)]
    pub manifest: PluginManifest,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PluginConfig {
    pub enabled_plugins: Vec<String>,
}

fn get_plugins_dir() -> Result<std::path::PathBuf, String> {
    let app_dir = std::env::current_dir()
        .map_err(|e| format!("Failed to get current directory: {}", e))?;
    let plugins_dir = app_dir.join("Plugins");
    
    // Create if doesn't exist
    if !plugins_dir.exists() {
        std::fs::create_dir_all(&plugins_dir)
            .map_err(|e| format!("Failed to create Plugins directory: {}", e))?;
    }
    
    Ok(plugins_dir)
}

fn get_plugin_config_path() -> Result<std::path::PathBuf, String> {
    let app_dir = std::env::current_dir()
        .map_err(|e| format!("Failed to get current directory: {}", e))?;
    Ok(app_dir.join("plugin_config.json"))
}

fn load_plugin_config() -> Result<PluginConfig, String> {
    let config_path = get_plugin_config_path()?;
    
    if !config_path.exists() {
        return Ok(PluginConfig {
            enabled_plugins: Vec::new(),
        });
    }
    
    let content = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("Failed to read plugin config: {}", e))?;
    
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse plugin config: {}", e))
}

fn save_plugin_config(config: &PluginConfig) -> Result<(), String> {
    let config_path = get_plugin_config_path()?;
    let content = serde_json::to_string_pretty(config)
        .map_err(|e| format!("Failed to serialize config: {}", e))?;
    
    std::fs::write(&config_path, content)
        .map_err(|e| format!("Failed to write plugin config: {}", e))
}

#[tauri::command]
pub async fn discover_plugins() -> Result<Vec<Plugin>, String> {
    let plugins_dir = get_plugins_dir()?;
    let config = load_plugin_config()?;
    let mut plugins = Vec::new();
    
    let entries = std::fs::read_dir(&plugins_dir)
        .map_err(|e| format!("Failed to read Plugins directory: {}", e))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
        let path = entry.path();
        
        if path.is_dir() {
            // Look for plugin.json in the directory
            let manifest_path = path.join("plugin.json");
            if manifest_path.exists() {
                match std::fs::read_to_string(&manifest_path) {
                    Ok(content) => {
                        match serde_json::from_str::<PluginManifest>(&content) {
                            Ok(manifest) => {
                                let plugin_id = path.file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("unknown")
                                    .to_string();
                                
                                plugins.push(Plugin {
                                    id: plugin_id.clone(),
                                    path: path.to_string_lossy().to_string(),
                                    enabled: config.enabled_plugins.contains(&plugin_id),
                                    manifest,
                                });
                            }
                            Err(e) => {
                                eprintln!("Failed to parse manifest for {}: {}", path.display(), e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to read manifest for {}: {}", path.display(), e);
                    }
                }
            }
        }
    }
    
    Ok(plugins)
}

#[tauri::command]
pub async fn toggle_plugin(plugin_id: String) -> Result<(), String> {
    let mut config = load_plugin_config()?;
    
    if let Some(pos) = config.enabled_plugins.iter().position(|id| id == &plugin_id) {
        config.enabled_plugins.remove(pos);
    } else {
        config.enabled_plugins.push(plugin_id);
    }
    
    save_plugin_config(&config)
}

#[tauri::command]
pub async fn execute_plugin(plugin_id: String, input: String) -> Result<String, String> {
    let plugins = discover_plugins().await?;
    let plugin = plugins.iter()
        .find(|p| p.id == plugin_id)
        .ok_or("Plugin not found")?;
    
    if !plugin.enabled {
        return Err("Plugin is not enabled".to_string());
    }
    
    let plugin_path = std::path::Path::new(&plugin.path);
    let entry_path = plugin_path.join(&plugin.manifest.entry);
    
    match plugin.manifest.plugin_type.as_str() {
        "python" => execute_python_plugin(&entry_path, &input),
        "node" => execute_node_plugin(&entry_path, &input),
        "executable" => execute_binary_plugin(&entry_path, &input),
        _ => Err(format!("Unsupported plugin type: {}", plugin.manifest.plugin_type))
    }
}

fn execute_python_plugin(entry_path: &std::path::Path, input: &str) -> Result<String, String> {
    let output = Command::new("python3")
        .arg(entry_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                let _ = stdin.write_all(input.as_bytes());
            }
            child.wait_with_output()
        })
        .map_err(|e| format!("Failed to execute Python plugin: {}", e))?;
    
    if output.status.success() {
        String::from_utf8(output.stdout)
            .map_err(|e| format!("Invalid UTF-8 in plugin output: {}", e))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Plugin execution failed: {}", stderr))
    }
}

fn execute_node_plugin(entry_path: &std::path::Path, input: &str) -> Result<String, String> {
    let output = Command::new("node")
        .arg(entry_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                let _ = stdin.write_all(input.as_bytes());
            }
            child.wait_with_output()
        })
        .map_err(|e| format!("Failed to execute Node plugin: {}", e))?;
    
    if output.status.success() {
        String::from_utf8(output.stdout)
            .map_err(|e| format!("Invalid UTF-8 in plugin output: {}", e))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Plugin execution failed: {}", stderr))
    }
}

fn execute_binary_plugin(entry_path: &std::path::Path, input: &str) -> Result<String, String> {
    let output = Command::new(entry_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                let _ = stdin.write_all(input.as_bytes());
            }
            child.wait_with_output()
        })
        .map_err(|e| format!("Failed to execute binary plugin: {}", e))?;
    
    if output.status.success() {
        String::from_utf8(output.stdout)
            .map_err(|e| format!("Invalid UTF-8 in plugin output: {}", e))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Plugin execution failed: {}", stderr))
    }
}

#[tauri::command]
pub async fn open_plugins_folder() -> Result<(), String> {
    let plugins_dir = get_plugins_dir()?;
    
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .arg(&plugins_dir)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }
    
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&plugins_dir)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }
    
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(&plugins_dir)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }
    
    Ok(())
}
