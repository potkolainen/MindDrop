use serde::{Deserialize, Serialize};
use sysinfo::System;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub gpu: Option<GpuInfo>,
    pub platform: PlatformInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total: u64,
    pub vram_used: u64,
    pub vram_free: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model: String,
    pub cores: usize,
    pub usage: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total: u64,
    pub used: u64,
    pub free: u64,
    pub available: u64,
    pub usage_percent: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub os: String,
    pub arch: String,
    pub hostname: String,
}

pub fn get_system_info() -> SystemInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    // sysinfo memory units have changed across versions/platforms.
    // We want to send *bytes* to the frontend. Detect if values look like KiB (old behavior)
    // and convert only in that case.
    let raw_total = sys.total_memory();
    let raw_used = sys.used_memory();
    let raw_available = sys.available_memory();
    let raw_free = sys.free_memory();

    // If the value is in KiB, a typical desktop total (8-256 GiB) will be in the millions.
    // If it's already bytes, it will be in the billions.
    let seems_bytes = raw_total > 4_u64 * 1024 * 1024 * 1024;
    let to_bytes = |v: u64| if seems_bytes { v } else { v.saturating_mul(1024) };

    let total_mem = to_bytes(raw_total);
    let used_mem = to_bytes(raw_used);
    let available_mem = to_bytes(raw_available);
    let free_mem = to_bytes(raw_free);

    let usage_percent = if total_mem > 0 {
        (used_mem as f32 / total_mem as f32) * 100.0
    } else {
        0.0
    };

    let cpu_model = sys.cpus().first()
        .map(|cpu| cpu.brand().to_string())
        .unwrap_or_else(|| "Unknown CPU".to_string());
    let cpu_cores = sys.cpus().len();
    let cpu_usage = sys.global_cpu_usage();

    let gpu = get_nvidia_gpu_info();

    SystemInfo {
        cpu: CpuInfo {
            model: cpu_model,
            cores: cpu_cores,
            usage: cpu_usage,
        },
        memory: MemoryInfo {
            total: total_mem,
            used: used_mem,
            free: free_mem,
            available: available_mem,
            usage_percent,
        },
        gpu,
        platform: PlatformInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            hostname: System::host_name().unwrap_or_else(|| "unknown".to_string()),
        },
    }
}

fn get_nvidia_gpu_info() -> Option<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next()?.trim();
    if line.is_empty() {
        return None;
    }

    // Expected: "GPU Name, 24564, 1234, 23330" (MiB)
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
    if parts.len() < 4 {
        return None;
    }

    let name = parts[0].to_string();
    let total_mib: u64 = parts[1].parse().ok()?;
    let used_mib: u64 = parts[2].parse().ok()?;
    let free_mib: u64 = parts[3].parse().ok()?;

    let mib_to_bytes = |mib: u64| mib.saturating_mul(1024).saturating_mul(1024);

    Some(GpuInfo {
        name,
        vram_total: mib_to_bytes(total_mib),
        vram_used: mib_to_bytes(used_mib),
        vram_free: mib_to_bytes(free_mib),
    })
}
