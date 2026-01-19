use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::path::PathBuf;
use std::fs;
use std::env;
use tauri::Emitter;
use std::io::{BufRead, BufReader};

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub task: String,
    pub engine: String,
    pub quantization: String,  // Quantization type: "fp16", "8bit", or "4bit"
    
    // Execution mode configuration
    #[serde(default)]
    pub execution_mode: Option<String>,  // "auto", "gpu", "hybrid", "cpu"
    
    #[serde(default)]
    pub gpu_layers: Option<u32>,  // For hybrid mode
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResult {
    pub success: bool,
    pub text: Option<String>,
    pub error: Option<String>,
    pub stats: Option<GenerationStats>,
    #[serde(default)]
    pub peak_vram_mb: Option<u32>,
    #[serde(default)]
    pub execution_settings: Option<ExecutionSettings>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecutionSettings {
    pub mode: String,
    pub quantization: String,
    pub gpu_layers: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationStats {
    pub total_tokens: usize,
    pub new_tokens: usize,
    pub time_seconds: f64,
    pub tokens_per_second: f64,
}

const CUSTOM_DOWNLOAD_DIR_ENV: &str = "AI_WORKSPACE_DOWNLOAD_DIR";

pub fn get_default_download_dir() -> PathBuf {
    // Get the current executable's directory
    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            // In dev mode, go up from target/debug/ to workspace root
            let workspace_root = if exe_dir.ends_with("target/debug") || exe_dir.ends_with("target/release") {
                exe_dir.parent().and_then(|p| p.parent()).unwrap_or(exe_dir)
            } else {
                exe_dir
            };
            return workspace_root.join("Downloads");
        }
    }
    
    // Fallback to home directory if we can't determine exe path
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ai-workspace-downloads")
}

pub fn get_model_cache_dir() -> PathBuf {
    // Check if custom download directory is set
    if let Ok(custom_dir) = env::var(CUSTOM_DOWNLOAD_DIR_ENV) {
        PathBuf::from(custom_dir)
    } else {
        get_default_download_dir()
    }
}

pub fn set_download_dir(path: &str) -> Result<(), String> {
    let path_buf = PathBuf::from(path);
    
    // Create the directory if it doesn't exist
    if !path_buf.exists() {
        fs::create_dir_all(&path_buf)
            .map_err(|e| format!("Failed to create directory: {}", e))?;
    }
    
    // Verify it's a directory
    if !path_buf.is_dir() {
        return Err("Path is not a directory".to_string());
    }
    
    env::set_var(CUSTOM_DOWNLOAD_DIR_ENV, path);
    Ok(())
}

pub fn get_model_path(model_id: &str) -> PathBuf {
    let cache_dir = get_model_cache_dir();
    // Use model ID directly as folder name (replace / with --)
    let safe_name = model_id.replace('/', "--");
    cache_dir.join(&safe_name)
}

pub fn list_models() -> Result<Vec<ModelInfo>, String> {
    let mut models = Vec::new();
    
    // 1. Scan Downloads directory (default cache dir)
    let cache_dir = get_model_cache_dir();
    if cache_dir.exists() {
        if let Ok(entries) = fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                if let Ok(file_name) = entry.file_name().into_string() {
                    // Skip hidden files and directories like .locks
                    if file_name.starts_with('.') {
                        continue;
                    }
                    
                    // Skip nested model cache directories
                    if file_name.starts_with("models--") {
                        continue;
                    }
                    
                    // Convert folder name back to model ID (replace -- with /)
                    let model_id = file_name.replace("--", "/");
                    
                    // Check if this folder contains model files or metadata
                    let model_path = cache_dir.join(&file_name);
                    if model_path.is_dir() {
                        let metadata_path = model_path.join("model_metadata.json");
                        
                        // Only include if metadata exists or if it contains model cache
                        if metadata_path.exists() || model_path.join(format!("models--{}", file_name)).exists() {
                            // Load metadata if exists, otherwise default to fp16
                            let quantization = load_model_quantization(&model_id);
                            
                            // Load execution config from metadata
                            let (execution_mode, gpu_layers) = load_execution_config(&model_id);
                            
                            models.push(ModelInfo {
                                id: model_id,
                                task: "text-generation".to_string(),
                                engine: "python".to_string(),
                                quantization,
                                execution_mode,
                                gpu_layers,
                            });
                        }
                    }
                }
            }
        }
    }
    
    // 2. Scan LM Studio models directory
    let home_dir = env::var("HOME").or_else(|_| env::var("USERPROFILE"))
        .unwrap_or_else(|_| "/home/user".to_string());
    let lmstudio_dir = PathBuf::from(&home_dir).join(".cache").join("lm-studio").join("models");
    
    if lmstudio_dir.exists() {
        scan_lmstudio_models(&lmstudio_dir, &mut models);
    }
    
    // 3. Scan Ollama models directory
    let ollama_dir = PathBuf::from(&home_dir).join(".ollama").join("models").join("manifests").join("registry.ollama.ai");
    
    if ollama_dir.exists() {
        scan_ollama_models(&ollama_dir, &mut models);
    }

    Ok(models)
}

fn scan_lmstudio_models(lmstudio_dir: &PathBuf, models: &mut Vec<ModelInfo>) {
    // LM Studio stores models in subdirectories like: publisher/model-name/
    if let Ok(entries) = fs::read_dir(lmstudio_dir) {
        for entry in entries.flatten() {
            if let Ok(publisher) = entry.file_name().into_string() {
                if publisher.starts_with('.') {
                    continue;
                }
                
                let publisher_path = lmstudio_dir.join(&publisher);
                if publisher_path.is_dir() {
                    // Scan models in this publisher directory
                    if let Ok(model_entries) = fs::read_dir(&publisher_path) {
                        for model_entry in model_entries.flatten() {
                            if let Ok(model_name) = model_entry.file_name().into_string() {
                                if model_name.starts_with('.') {
                                    continue;
                                }
                                
                                let model_path = publisher_path.join(&model_name);
                                if model_path.is_dir() {
                                    // Check if there are GGUF files
                                    if has_gguf_files(&model_path) {
                                        let model_id = format!("lmstudio/{}/{}", publisher, model_name);
                                        models.push(ModelInfo {
                                            id: model_id,
                                            task: "text-generation".to_string(),
                                            engine: "lmstudio".to_string(),
                                            quantization: "gguf".to_string(),
                                            execution_mode: Some("auto".to_string()),
                                            gpu_layers: None,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn scan_ollama_models(ollama_dir: &PathBuf, models: &mut Vec<ModelInfo>) {
    // Ollama stores manifests in: ~/.ollama/models/manifests/registry.ollama.ai/library/
    let library_path = ollama_dir.join("library");
    
    if !library_path.exists() {
        return;
    }
    
    if let Ok(entries) = fs::read_dir(&library_path) {
        for entry in entries.flatten() {
            if let Ok(model_name) = entry.file_name().into_string() {
                if model_name.starts_with('.') {
                    continue;
                }
                
                let model_path = library_path.join(&model_name);
                if model_path.is_dir() || model_path.is_file() {
                    let model_id = format!("ollama/{}", model_name);
                    models.push(ModelInfo {
                        id: model_id,
                        task: "text-generation".to_string(),
                        engine: "ollama".to_string(),
                        quantization: "gguf".to_string(),
                        execution_mode: Some("auto".to_string()),
                        gpu_layers: None,
                    });
                }
            }
        }
    }
}

fn has_gguf_files(dir: &PathBuf) -> bool {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Ok(file_name) = entry.file_name().into_string() {
                if file_name.ends_with(".gguf") {
                    return true;
                }
            }
        }
    }
    false
}

fn get_metadata_path(model_id: &str) -> PathBuf {
    let model_path = get_model_path(model_id);
    model_path.join("model_metadata.json")
}

pub fn load_model_quantization(model_id: &str) -> String {
    let metadata_path = get_metadata_path(model_id);
    if let Ok(content) = fs::read_to_string(&metadata_path) {
        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(quant) = data.get("quantization").and_then(|v| v.as_str()) {
                return quant.to_string();
            }
        }
    }
    "native".to_string()  // Default to native (no quantization)
}

fn load_execution_config(model_id: &str) -> (Option<String>, Option<u32>) {
    let metadata_path = get_metadata_path(model_id);
    if let Ok(content) = fs::read_to_string(&metadata_path) {
        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&content) {
            let execution_mode = data.get("execution_mode")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            
            let gpu_layers = data.get("gpu_layers")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            
            return (execution_mode, gpu_layers);
        }
    }
    (Some("auto".to_string()), None)  // Default to auto mode
}

pub fn set_model_quantization(model_id: &str, quantization: &str) -> Result<(), String> {
    let metadata_path = get_metadata_path(model_id);
    let metadata = serde_json::json!({
        "quantization": quantization
    });
    fs::write(&metadata_path, serde_json::to_string_pretty(&metadata).unwrap())
        .map_err(|e| format!("Failed to save metadata: {}", e))?;
    Ok(())
}

pub fn delete_model(model_id: &str) -> Result<(), String> {
    let model_path = get_model_path(model_id);

    if model_path.exists() {
        fs::remove_dir_all(&model_path)
            .map_err(|e| format!("Failed to delete model: {}", e))?;
        Ok(())
    } else {
        Err("Model not found".to_string())
    }
}

pub fn run_python_inference(
    model_id: &str, 
    prompt: &str, 
    max_tokens: usize, 
    quantization: &str,
    execution_mode: Option<&str>,
    gpu_layers: Option<u32>,
) -> Result<InferenceResult, String> {
    // Force kill any Python3 processes to free GPU memory
    let _ = Command::new("pkill")
        .args(&["-9", "-f", "python3"])
        .output();
    
    // Wait longer for processes to die and GPU to fully clear
    std::thread::sleep(std::time::Duration::from_secs(2));
    
    let model_path = get_model_path(model_id);
    let cache_dir_str = model_path.to_string_lossy().to_string();
    
    // Read GPU settings from environment or use defaults
    let gpu_threshold = env::var("GPU_MEMORY_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(10.0);  // Increased default to 10GB for better safety margin
    
    let force_cpu = env::var("FORCE_CPU_MODE")
        .ok()
        .and_then(|v| v.parse::<bool>().ok())
        .unwrap_or(false);
    
    let quantization_type = quantization;
    let exec_mode = execution_mode.unwrap_or("auto");
    let gpu_layers_val = gpu_layers.map(|l| l.to_string()).unwrap_or_else(|| "None".to_string());
    
    let python_script = format!(
        r#"
import sys
import json
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set environment variable to reduce GPU memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

model_id = "{}"
cache_dir = "{}"
prompt = {}
max_tokens = {}
gpu_threshold = {}
force_cpu = {}
execution_mode = "{}"
gpu_layers = {}

try:
    print("Loading model...", file=sys.stderr)
    
    # Clear GPU cache aggressively before checking memory
    if torch.cuda.is_available() and not force_cpu:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        gc.collect()
        torch.cuda.empty_cache()
    
    # Determine device_map and max_memory based on execution mode
    device_map = None
    max_memory = None
    use_gpu = False
    
    if execution_mode == "cpu" or force_cpu:
        # CPU-only mode
        device_map = "cpu"
        max_memory = None
        use_gpu = False
        device = "cpu"
        print("Execution mode: CPU-only", file=sys.stderr)
    elif execution_mode == "hybrid" and gpu_layers is not None:
        # Hybrid mode: use device_map with max_memory to limit GPU usage
        # This forces overflow layers to CPU
        device_map = "auto"
        # Estimate ~500MB per layer on GPU, rest goes to CPU
        max_memory = {{0: f"{{gpu_layers * 500}}MiB", "cpu": "80GiB"}}
        use_gpu = True
        device = "cuda"
        print(f"Execution mode: Hybrid ({{gpu_layers}} layers on GPU)", file=sys.stderr)
    elif not force_cpu and torch.cuda.is_available():
        # GPU mode or auto mode with GPU available
        try:
            # Get free GPU memory in GB
            free_memory = torch.cuda.mem_get_info()[0] / (1024**3)
            total_memory = torch.cuda.mem_get_info()[1] / (1024**3)
            print(f"GPU memory: {{free_memory:.2f}} GB free / {{total_memory:.2f}} GB total", file=sys.stderr)
            
            # Only use GPU if we have enough free memory
            if free_memory > gpu_threshold:
                use_gpu = True
                device = "cuda"
                device_map = "cuda:0"  # Full GPU mode
                max_memory = None
                print(f"Execution mode: GPU (threshold: {{gpu_threshold}}GB)", file=sys.stderr)
            else:
                print(f"Not enough GPU memory (need {{gpu_threshold}}GB, have {{free_memory:.2f}}GB), using CPU", file=sys.stderr)
                device = "cpu"
                device_map = "cpu"
                use_gpu = False
        except Exception as e:
            print(f"GPU check failed, using CPU: {{e}}", file=sys.stderr)
            device = "cpu"
            device_map = "cpu"
            use_gpu = False
    else:
        # No GPU available or force CPU
        device = "cpu"
        device_map = "cpu"
        use_gpu = False
        print("No GPU available, using CPU", file=sys.stderr)
    
    # Load model with automatic dtype handling, quantization, and fallback on OOM
    try:
        quant_type = "{quantization_type}"
        
        if use_gpu:
            if quant_type == "4bit":
                # 4-bit quantization - most memory efficient
                try:
                    print("Loading with 4-bit quantization (most memory efficient)...", file=sys.stderr)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        cache_dir=cache_dir,
                        load_in_4bit=True,
                        device_map=device_map,
                        max_memory=max_memory,
                        low_cpu_mem_usage=True
                    )
                    print("Model loaded with 4-bit quantization", file=sys.stderr)
                except Exception as quant_error:
                    print(f"4-bit quantization failed: {{quant_error}}", file=sys.stderr)
                    print("Falling back to native...", file=sys.stderr)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        cache_dir=cache_dir,
                        torch_dtype="auto",
                        device_map=device_map,
                        max_memory=max_memory,
                        low_cpu_mem_usage=True
                    )
            elif quant_type == "8bit":
                # 8-bit quantization - balanced
                try:
                    print("Loading with 8-bit quantization...", file=sys.stderr)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        cache_dir=cache_dir,
                        load_in_8bit=True,
                        device_map=device_map,
                        max_memory=max_memory,
                        low_cpu_mem_usage=True
                    )
                    print("Model loaded with 8-bit quantization", file=sys.stderr)
                except Exception as quant_error:
                    print(f"8-bit quantization failed: {{quant_error}}", file=sys.stderr)
                    print("Falling back to native...", file=sys.stderr)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        cache_dir=cache_dir,
                        torch_dtype="auto",
                        device_map=device_map,
                        max_memory=max_memory,
                        low_cpu_mem_usage=True
                    )
            elif quant_type == "fp16":
                # Force FP16 quantization
                import torch
                print("Loading with FP16 quantization...", file=sys.stderr)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    max_memory=max_memory,
                    low_cpu_mem_usage=True
                )
                print("Model loaded with FP16", file=sys.stderr)
            else:
                # native - no quantization, use model's native dtype
                print("Loading with native dtype (no quantization)...", file=sys.stderr)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    torch_dtype="auto",
                    device_map=device_map,
                    max_memory=max_memory,
                    low_cpu_mem_usage=True
                )
                print("Model loaded with native dtype", file=sys.stderr)
        else:
            # For CPU, use auto dtype and let the model handle it
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
    except torch.cuda.OutOfMemoryError as oom:
        # If GPU runs out of memory, fall back to CPU
        print(f"GPU OOM error during model load, falling back to CPU: {{oom}}", file=sys.stderr)
        use_gpu = False
        device = "cpu"
        # Clear GPU and try CPU
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Generating on {{device}}...", file=sys.stderr)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    end_time = time.time()
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed_sec = end_time - start_time
    new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    total_tokens = outputs.shape[1]
    
    # Clean up memory aggressively
    del model
    del tokenizer
    del inputs
    del outputs
    import gc
    gc.collect()
    
    # Get peak VRAM usage before cleanup
    peak_vram_mb = 0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.empty_cache()
    
    result = {{
        "success": True,
        "text": generated_text,
        "stats": {{
            "total_tokens": total_tokens,
            "new_tokens": new_tokens,
            "time_seconds": round(elapsed_sec, 2),
            "tokens_per_second": round(new_tokens / elapsed_sec, 1) if elapsed_sec > 0 else 0
        }},
        "peak_vram_mb": int(peak_vram_mb),
        "execution_settings": {{
            "mode": "{}",
            "quantization": "{}",
            "gpu_layers": {}
        }}
    }}
    
    print(json.dumps(result))
    
except Exception as e:
    # Clean up on error
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    error_result = {{
        "success": False,
        "error": str(e),
        "execution_settings": {{
            "mode": "{}",
            "quantization": "{}",
            "gpu_layers": {}
        }}
    }}
    print(json.dumps(error_result))
    sys.exit(1)
"#,
        model_id,
        cache_dir_str,
        serde_json::to_string(prompt).unwrap(),
        max_tokens,
        gpu_threshold,
        if force_cpu { "True" } else { "False" },
        exec_mode,
        gpu_layers_val,
        exec_mode,
        quantization_type,
        gpu_layers_val,
        exec_mode,
        quantization_type,
        gpu_layers_val
    );

    let output = Command::new("python3")
        .arg("-c")
        .arg(&python_script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to execute Python: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Log stderr for debugging
    if !stderr.is_empty() {
        eprintln!("Python stderr: {}", stderr);
    }
    
    // Find the last line that looks like JSON
    let last_line = stdout.lines()
        .filter(|line| line.trim().starts_with('{'))
        .last()
        .unwrap_or("{}");
    
    if last_line == "{}" {
        return Err(format!("No JSON output from Python script. Stdout: {}, Stderr: {}", 
            stdout.lines().take(5).collect::<Vec<_>>().join(" | "),
            stderr.lines().take(5).collect::<Vec<_>>().join(" | ")));
    }
    
    // Parse the result - if it's a valid JSON with success=false, return that error
    let result: InferenceResult = serde_json::from_str(last_line)
        .map_err(|e| format!("Failed to parse result from line '{}': {}. Full stdout: {}", 
            last_line, e, stdout))?;
    
    Ok(result)
}

pub fn run_streaming_inference(
    model_id: &str,
    prompt: &str,
    max_tokens: usize,
    gpu_threshold: f64,
    force_cpu: bool,
    exec_mode: &str,
    gpu_layers_val: Option<u32>,
    quantization_type: &str,
    app: tauri::AppHandle,
) -> Result<InferenceResult, String> {
    let cache_dir = get_model_cache_dir();
    let cache_dir_str = cache_dir.to_string_lossy().to_string();

    // Build the Python script with streaming support
    let python_script = format!(
        r#"
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

model_id = "{}"
cache_dir = "{}"
prompt = {}
max_tokens = {}
gpu_threshold = {}
force_cpu = {}
exec_mode = "{}"
gpu_layers = {}
quant_type = "{}"

try:
    # GPU availability check
    if torch.cuda.is_available() and not force_cpu:
        free_memory = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        total_memory = torch.cuda.mem_get_info()[1] / (1024 ** 3)
        print(f"GPU memory: {{free_memory:.2f}} GB free / {{total_memory:.2f}} GB total", file=sys.stderr)
        
        if free_memory > gpu_threshold:
            use_gpu = True
            device = "cuda"
            device_map = "cuda:0"
            max_memory = None
        else:
            device = "cpu"
            device_map = "cpu"
            use_gpu = False
    else:
        device = "cpu"
        device_map = "cpu"
        use_gpu = False
    
    # Load model with quantization
    if use_gpu:
        if quant_type == "4bit":
            model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir=cache_dir, load_in_4bit=True,
                device_map=device_map, low_cpu_mem_usage=True
            )
        elif quant_type == "8bit":
            model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir=cache_dir, load_in_8bit=True,
                device_map=device_map, low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir=cache_dir, torch_dtype="auto",
                device_map=device_map, low_cpu_mem_usage=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, torch_dtype="auto",
            low_cpu_mem_usage=True, device_map="cpu"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    
    # Create streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generation arguments
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )
    
    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream tokens as they are generated
    full_text = ""
    start_time = time.time()
    token_count = 0
    
    for new_text in streamer:
        full_text += new_text
        token_count += 1
        # Output each token for real-time streaming
        print(f"TOKEN:{{new_text}}", flush=True)
    
    thread.join()
    end_time = time.time()
    elapsed_sec = end_time - start_time
    
    # Clean up
    del model
    del tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    result = {{
        "success": True,
        "text": full_text,
        "stats": {{
            "total_tokens": token_count + inputs['input_ids'].shape[1],
            "new_tokens": token_count,
            "time_seconds": round(elapsed_sec, 2),
            "tokens_per_second": round(token_count / elapsed_sec, 1) if elapsed_sec > 0 else 0
        }}
    }}
    
    print(f"DONE:{{json.dumps(result)}}", flush=True)
    
except Exception as e:
    error_result = {{
        "success": False,
        "error": str(e)
    }}
    print(f"ERROR:{{json.dumps(error_result)}}", flush=True)
    sys.exit(1)
"#,
        model_id,
        cache_dir_str,
        serde_json::to_string(prompt).unwrap(),
        max_tokens,
        gpu_threshold,
        if force_cpu { "True" } else { "False" },
        exec_mode,
        gpu_layers_val.map(|n| n.to_string()).unwrap_or_else(|| "None".to_string()),
        quantization_type
    );

    let mut child = Command::new("python3")
        .arg("-u")  // Unbuffered output
        .arg("-c")
        .arg(&python_script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to execute Python: {}", e))?;

    // Read stdout in real-time
    let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
    let reader = BufReader::new(stdout);
    
    let mut full_result: Option<InferenceResult> = None;
    
    for line in reader.lines() {
        if let Ok(line) = line {
            if line.starts_with("TOKEN:") {
                // Extract token and emit event
                let token = line.strip_prefix("TOKEN:").unwrap_or("");
                let _ = app.emit("inference-token", serde_json::json!({"token": token}));
            } else if line.starts_with("DONE:") {
                // Parse final result
                let json_str = line.strip_prefix("DONE:").unwrap_or("{}");
                full_result = serde_json::from_str(json_str).ok();
            } else if line.starts_with("ERROR:") {
                let json_str = line.strip_prefix("ERROR:").unwrap_or("{}");
                full_result = serde_json::from_str(json_str).ok();
            }
        }
    }

    child.wait().map_err(|e| format!("Failed to wait for process: {}", e))?;
    
    full_result.ok_or_else(|| "No result received from streaming inference".to_string())
}

pub fn download_model_python(model_id: &str, app: tauri::AppHandle) -> Result<(), String> {
    use std::io::{BufRead, BufReader};
    use serde_json::json;
    use std::thread;
    
    eprintln!("Starting download for model: {}", model_id);
    
    // Create the model-specific directory
    let model_path = get_model_path(model_id);
    eprintln!("Model path: {:?}", model_path);
    
    fs::create_dir_all(&model_path)
        .map_err(|e| format!("Failed to create model directory: {}", e))?;

    let cache_dir_str = model_path.to_string_lossy().to_string();
    let model_id_clone = model_id.to_string();
    
    // Use Python script with file-by-file download tracking
    let python_script = format!(
        r#"
import sys
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path

model_id = "{}"
cache_dir = "{}"

try:
    print(f"PROGRESS:0:Starting download...", flush=True)
    print(f"PROGRESS:2:Fetching file list...", flush=True)
    
    # Get list of all files in the repo
    files = list(list_repo_files(repo_id=model_id))
    total_files = len(files)
    
    print(f"PROGRESS:5:Found {{total_files}} files", flush=True)
    
    # Download each file and track progress
    downloaded_count = 0
    total_bytes = 0
    
    for i, file_path in enumerate(files, 1):
        # Map file progress to 10-95% range
        # 10% at start, 95% at end = 85% range for files
        progress = 10 + int((i - 1) / total_files * 85)
        
        print(f"PROGRESS:{{progress}}:[{{i}}/{{total_files}}] Downloading {{file_path}}...", flush=True)
        
        try:
            # Download the file
            local_path = hf_hub_download(
                repo_id=model_id,
                filename=file_path,
                cache_dir=cache_dir,
                resume_download=True
            )
            
            # Get file size
            file_size = Path(local_path).stat().st_size
            total_bytes += file_size
            downloaded_count += 1
            
        except Exception as e:
            print(f"PROGRESS:{{progress}}:[{{i}}/{{total_files}}] Warning: {{file_path}} - {{str(e)}}", flush=True)
    
    print(f"PROGRESS:95:Finalizing...", flush=True)
    
    # Final stats
    total_mb = total_bytes / (1024 * 1024)
    print(f"PROGRESS:100:Complete! {{downloaded_count}} files ({{total_mb:.1f}}MB)", flush=True)
    
except Exception as e:
    print(f"ERROR:{{e}}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"#,
        model_id,
        cache_dir_str
    );

    let mut child = Command::new("python3")
        .arg("-u")  // Unbuffered output
        .arg("-c")
        .arg(&python_script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to execute Python: {}", e))?;

    // Read stdout in real-time in a separate thread
    let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
    let app_clone = app.clone();
    let model_id_for_thread = model_id_clone.clone();
    
    let stdout_thread = thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(line) = line {
                eprintln!("Python stdout: {}", line);
                if line.starts_with("PROGRESS:") {
                    let parts: Vec<&str> = line.splitn(3, ':').collect();
                    if parts.len() == 3 {
                        if let Ok(progress) = parts[1].parse::<u32>() {
                            let message = parts[2];
                            eprintln!("Emitting progress: {}% - {}", progress, message);
                            let _ = app_clone.emit("download-progress", json!({
                                "model_id": &model_id_for_thread,
                                "progress": progress,
                                "message": message
                            }));
                        }
                    }
                } else if line.starts_with("ERROR:") {
                    let error_msg = line.strip_prefix("ERROR:").unwrap_or(&line);
                    eprintln!("Download error: {}", error_msg);
                }
            }
        }
        eprintln!("Stdout reading thread finished");
    });

    eprintln!("Waiting for download process to complete...");
    let status = child.wait()
        .map_err(|e| format!("Failed to wait for Python process: {}", e))?;

    stdout_thread.join().ok();
    eprintln!("Download process finished with status: {:?}", status);

    if !status.success() {
        return Err("Download failed".to_string());
    }

    eprintln!("Download completed successfully for {}", model_id_clone);
    Ok(())
}
