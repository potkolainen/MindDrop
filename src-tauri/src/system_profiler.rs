use serde::{Deserialize, Serialize};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUInfo {
    pub cuda_available: bool,
    pub total_vram_mb: u32,
    pub free_vram_mb: u32,
    pub used_vram_mb: u32,
}

/// Complete system resource snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResources {
    pub gpu: Option<GPUInfo>,
    pub total_ram_mb: u32,
    pub available_ram_mb: u32,  // Key: accounts for OS cache, not just "free"
    pub used_ram_mb: u32,
    pub cpu_cores: u32,
    pub cpu_usage_percent: f32,
    pub cpu_available_percent: f32,
}

/// Detect current system resources using Python + psutil + torch
fn detect_system_resources() -> Result<SystemResources, String> {
    let python_script = r#"
import json
import sys

try:
    import torch
    import psutil
except ImportError as e:
    print(json.dumps({"error": f"Missing dependency: {e}"}))
    sys.exit(1)

try:
    # GPU detection
    gpu_info = None
    if torch.cuda.is_available():
        free_vram = torch.cuda.mem_get_info()[0] / 1_048_576
        total_vram = torch.cuda.mem_get_info()[1] / 1_048_576
        gpu_info = {
            "cuda_available": True,
            "free_vram_mb": int(free_vram),
            "total_vram_mb": int(total_vram),
            "used_vram_mb": int(total_vram - free_vram)
        }
    
    # RAM detection (use 'available' not 'free' - accounts for OS cache)
    mem = psutil.virtual_memory()
    total_ram = mem.total / 1_048_576
    available_ram = mem.available / 1_048_576  # This is the key metric
    used_ram = mem.used / 1_048_576
    
    # CPU detection
    cpu_cores = psutil.cpu_count(logical=True)
    cpu_percent = psutil.cpu_percent(interval=0.5)  # 500ms sample
    cpu_available = 100.0 - cpu_percent
    
    result = {
        "gpu": gpu_info,
        "total_ram_mb": int(total_ram),
        "available_ram_mb": int(available_ram),
        "used_ram_mb": int(used_ram),
        "cpu_cores": cpu_cores,
        "cpu_usage_percent": cpu_percent,
        "cpu_available_percent": cpu_available
    }
    
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
"#;

    let output = Command::new("python3")
        .arg("-c")
        .arg(python_script)
        .output()
        .map_err(|e| format!("Failed to run system detection: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python script failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let sys_info: serde_json::Value = serde_json::from_str(stdout.trim())
        .map_err(|e| format!("Failed to parse system info: {} (output: {})", e, stdout))?;

    // Check for error in JSON
    if let Some(error) = sys_info.get("error") {
        return Err(format!("System detection error: {}", error.as_str().unwrap_or("unknown")));
    }

    let gpu = if let Some(gpu_data) = sys_info["gpu"].as_object() {
        Some(GPUInfo {
            cuda_available: gpu_data["cuda_available"].as_bool().unwrap_or(false),
            free_vram_mb: gpu_data["free_vram_mb"].as_u64().unwrap_or(0) as u32,
            total_vram_mb: gpu_data["total_vram_mb"].as_u64().unwrap_or(0) as u32,
            used_vram_mb: gpu_data["used_vram_mb"].as_u64().unwrap_or(0) as u32,
        })
    } else {
        None
    };

    Ok(SystemResources {
        gpu,
        total_ram_mb: sys_info["total_ram_mb"].as_u64().unwrap_or(0) as u32,
        available_ram_mb: sys_info["available_ram_mb"].as_u64().unwrap_or(0) as u32,
        used_ram_mb: sys_info["used_ram_mb"].as_u64().unwrap_or(0) as u32,
        cpu_cores: sys_info["cpu_cores"].as_u64().unwrap_or(1) as u32,
        cpu_usage_percent: sys_info["cpu_usage_percent"].as_f64().unwrap_or(0.0) as f32,
        cpu_available_percent: sys_info["cpu_available_percent"].as_f64().unwrap_or(100.0) as f32,
    })
}

// Cache system info for 5 seconds to avoid overhead from repeated calls
static SYSTEM_CACHE: Mutex<Option<(SystemResources, Instant)>> = Mutex::new(None);

/// Get system resources with 1-second caching
pub fn get_system_resources() -> Result<SystemResources, String> {
    let mut cache = SYSTEM_CACHE.lock().unwrap();
    
    if let Some((ref info, ref timestamp)) = *cache {
        if timestamp.elapsed() < Duration::from_secs(1) {
            // Return cloned cached data
            return Ok(info.clone());
        }
    }
    
    // Cache expired or doesn't exist, fetch new data
    let info = detect_system_resources()?;
    *cache = Some((info.clone(), Instant::now()));
    Ok(info)
}

/// Force refresh system resources (bypass cache)
pub fn refresh_system_resources() -> Result<SystemResources, String> {
    let info = detect_system_resources()?;
    let mut cache = SYSTEM_CACHE.lock().unwrap();
    *cache = Some((info.clone(), Instant::now()));
    Ok(info)
}

/// Get just GPU info (for backward compatibility)
pub fn get_gpu_info() -> Result<GPUInfo, String> {
    let sys = get_system_resources()?;
    sys.gpu.ok_or_else(|| "No GPU available".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_system_resources() {
        // This test requires Python with torch and psutil installed
        match detect_system_resources() {
            Ok(resources) => {
                println!("System resources: {:?}", resources);
                assert!(resources.total_ram_mb > 0);
                assert!(resources.available_ram_mb > 0);
                assert!(resources.available_ram_mb <= resources.total_ram_mb);
                assert!(resources.cpu_cores > 0);
                assert!(resources.cpu_usage_percent >= 0.0);
                assert!(resources.cpu_usage_percent <= 100.0);
                
                if let Some(gpu) = resources.gpu {
                    assert!(gpu.cuda_available);
                    assert!(gpu.total_vram_mb > 0);
                    assert!(gpu.free_vram_mb <= gpu.total_vram_mb);
                }
            }
            Err(e) => {
                println!("Warning: System detection failed (expected if psutil not installed): {}", e);
            }
        }
    }

    #[test]
    fn test_caching() {
        // Two calls within 5 seconds should use cache
        let start = Instant::now();
        let _ = get_system_resources();
        let first_call = start.elapsed();
        
        let start = Instant::now();
        let _ = get_system_resources();
        let second_call = start.elapsed();
        
        // Second call should be much faster (cached)
        println!("First call: {:?}, Second call: {:?}", first_call, second_call);
        assert!(second_call < first_call / 2);
    }
}
