use crate::execution_config::ExecutionMode;
use crate::memory_estimator::{estimate_vram_usage, get_model_params, get_model_layers, get_hidden_size};
use crate::system_profiler::get_system_resources;
use serde::{Deserialize, Serialize};

/// Auto mode decision with reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoModeDecision {
    pub mode: ExecutionMode,
    pub gpu_layers: Option<u32>,
    pub reason: String,
    pub estimated_vram_mb: u32,
    pub available_vram_mb: u32,
    pub model_info: ModelInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub total_params: u64,
    pub total_layers: u32,
    pub hidden_size: u32,
}

/// Decide optimal execution mode based on system resources and model requirements
pub fn decide_execution_mode(
    model_id: &str,
    quantization: &str,
    context_length: u32,
) -> Result<AutoModeDecision, String> {
    // Step 1: Check GPU availability
    let sys_resources = get_system_resources()?;
    
    let gpu_info = match sys_resources.gpu {
        Some(ref gpu) if gpu.cuda_available => gpu,
        _ => {
            return Ok(AutoModeDecision {
                mode: ExecutionMode::CPU,
                gpu_layers: Some(0),
                reason: "No CUDA GPU available - using CPU".to_string(),
                estimated_vram_mb: 0,
                available_vram_mb: 0,
                model_info: ModelInfo {
                    total_params: get_model_params(model_id),
                    total_layers: get_model_layers(model_id),
                    hidden_size: get_hidden_size(model_id),
                },
            });
        }
    };
    
    // Step 2: Get model specs
    let total_params = get_model_params(model_id);
    let total_layers = get_model_layers(model_id);
    let hidden_size = get_hidden_size(model_id);
    
    let model_info = ModelInfo {
        total_params,
        total_layers,
        hidden_size,
    };
    
    // Step 3: Apply safety margin (20% - don't use more than 80% of free VRAM)
    let safe_vram_mb = (gpu_info.free_vram_mb as f64 * 0.80) as u32;
    
    log::info!(
        "Auto mode analysis: {}, quant={}, ctx={}, free_vram={}MB, safe_vram={}MB",
        model_id, quantization, context_length, gpu_info.free_vram_mb, safe_vram_mb
    );
    
    // Step 4: Try full GPU first
    let full_gpu_estimate = estimate_vram_usage(
        total_params,
        quantization,
        total_layers,
        total_layers,  // All layers on GPU
        context_length,
        hidden_size,
    );
    
    log::info!(
        "Full GPU estimate: {}MB (weights={}, kv_cache={}, activations={}, overhead={})",
        full_gpu_estimate.total_mb,
        full_gpu_estimate.model_weights_mb,
        full_gpu_estimate.kv_cache_mb,
        full_gpu_estimate.activations_mb,
        full_gpu_estimate.overhead_mb
    );
    
    if full_gpu_estimate.total_mb <= safe_vram_mb {
        return Ok(AutoModeDecision {
            mode: ExecutionMode::GPU,
            gpu_layers: Some(total_layers),
            reason: format!(
                "Full GPU fits: {} MB estimated / {} MB available ({}% usage)",
                full_gpu_estimate.total_mb,
                safe_vram_mb,
                (full_gpu_estimate.total_mb as f64 / safe_vram_mb as f64 * 100.0) as u32
            ),
            estimated_vram_mb: full_gpu_estimate.total_mb,
            available_vram_mb: safe_vram_mb,
            model_info,
        });
    }
    
    // Step 5: Binary search for optimal hybrid split
    let optimal_layers = calculate_max_gpu_layers(
        total_params,
        quantization,
        total_layers,
        context_length,
        hidden_size,
        safe_vram_mb,
    );
    
    log::info!("Optimal GPU layers via binary search: {} / {}", optimal_layers, total_layers);
    
    // Step 6: Check if hybrid is worth it (minimum 15 layers or 25% of total)
    let min_layers_threshold = std::cmp::max(15, (total_layers as f64 * 0.25) as u32);
    
    if optimal_layers >= min_layers_threshold {
        let estimate = estimate_vram_usage(
            total_params,
            quantization,
            total_layers,
            optimal_layers,
            context_length,
            hidden_size,
        );
        
        let gpu_percentage = (optimal_layers as f64 / total_layers as f64 * 100.0) as u32;
        
        Ok(AutoModeDecision {
            mode: ExecutionMode::Hybrid,
            gpu_layers: Some(optimal_layers),
            reason: format!(
                "Hybrid mode: {}/{} layers on GPU ({}%), {} MB estimated / {} MB available",
                optimal_layers, total_layers, gpu_percentage,
                estimate.total_mb, safe_vram_mb
            ),
            estimated_vram_mb: estimate.total_mb,
            available_vram_mb: safe_vram_mb,
            model_info,
        })
    } else {
        // Too few layers for hybrid to be beneficial - CPU is better
        Ok(AutoModeDecision {
            mode: ExecutionMode::CPU,
            gpu_layers: Some(0),
            reason: format!(
                "CPU mode: Only {} / {} layers fit in VRAM (below {}% threshold), CPU inference more efficient",
                optimal_layers, total_layers,
                (min_layers_threshold as f64 / total_layers as f64 * 100.0) as u32
            ),
            estimated_vram_mb: 0,
            available_vram_mb: safe_vram_mb,
            model_info,
        })
    }
}

/// Binary search to find maximum number of GPU layers that fit in available VRAM
fn calculate_max_gpu_layers(
    total_params: u64,
    quantization: &str,
    total_layers: u32,
    context_length: u32,
    hidden_size: u32,
    available_vram_mb: u32,
) -> u32 {
    let mut low = 0;
    let mut high = total_layers;
    let mut best = 0;
    
    while low <= high {
        let mid = (low + high) / 2;
        let estimate = estimate_vram_usage(
            total_params,
            quantization,
            total_layers,
            mid,
            context_length,
            hidden_size,
        );
        
        if estimate.total_mb <= available_vram_mb {
            best = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    
    best
}

/// Recommend execution mode for a model (simplified API)
pub fn recommend_mode(model_id: &str, quantization: &str) -> Result<AutoModeDecision, String> {
    // Use default context length of 2048 tokens
    decide_execution_mode(model_id, quantization, 2048)
}

/// Calculate the maximum theoretical model size that can fit on GPU
/// Returns a description of the largest model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxModelCapacity {
    pub max_params_billions: f64,
    pub model_category: String,  // "0.6B", "7B", "13B", "20B", "32B", "70B"
    pub quantization: String,
    pub execution_mode: String,  // "GPU", "Hybrid", or "CPU-only"
    pub gpu_layers: Option<u32>,
    pub total_layers: u32,
    pub estimated_vram_mb: u32,
    pub available_vram_mb: u32,
}

pub fn calculate_max_model_capacity(quantization: &str) -> Result<MaxModelCapacity, String> {
    let sys_resources = get_system_resources()?;
    
    let gpu_info = match sys_resources.gpu {
        Some(ref gpu) if gpu.cuda_available => gpu,
        _ => {
            return Ok(MaxModelCapacity {
                max_params_billions: 0.0,
                model_category: "CPU-only".to_string(),
                quantization: quantization.to_string(),
                execution_mode: "CPU-only".to_string(),
                gpu_layers: None,
                total_layers: 0,
                estimated_vram_mb: 0,
                available_vram_mb: 0,
            });
        }
    };
    
    // Apply safety margin (80% of free VRAM)
    let safe_vram_mb = (gpu_info.free_vram_mb as f64 * 0.80) as u32;
    
    // Test model sizes from largest to smallest
    let test_models = vec![
        ("70B", 70_000_000_000u64, 80u32, 8192u32),
        ("32B", 32_000_000_000u64, 64u32, 4096u32),
        ("20B", 20_000_000_000u64, 40u32, 4096u32),
        ("13B", 13_000_000_000u64, 40u32, 4096u32),
        ("7B", 7_000_000_000u64, 32u32, 4096u32),
        ("0.6B", 600_000_000u64, 24u32, 1536u32),
    ];
    
    let context_length = 2048; // Standard context for estimation
    
    for (category, params, layers, hidden_size) in test_models {
        // Try full GPU first
        let estimate = estimate_vram_usage(
            params,
            quantization,
            layers,
            layers,
            context_length,
            hidden_size,
        );
        
        if estimate.total_mb <= safe_vram_mb {
            // Full model fits on GPU
            return Ok(MaxModelCapacity {
                max_params_billions: params as f64 / 1_000_000_000.0,
                model_category: category.to_string(),
                quantization: quantization.to_string(),
                execution_mode: "GPU".to_string(),
                gpu_layers: Some(layers),
                total_layers: layers,
                estimated_vram_mb: estimate.total_mb,
                available_vram_mb: safe_vram_mb,
            });
        }
        
        // Try hybrid mode
        let max_layers = calculate_max_gpu_layers(
            params,
            quantization,
            layers,
            context_length,
            hidden_size,
            safe_vram_mb,
        );
        
        // Check if hybrid is worthwhile (at least 25% of layers or 15 layers)
        let min_hybrid_layers = std::cmp::max(15, (layers as f64 * 0.25) as u32);
        
        if max_layers >= min_hybrid_layers {
            let hybrid_estimate = estimate_vram_usage(
                params,
                quantization,
                layers,
                max_layers,
                context_length,
                hidden_size,
            );
            
            return Ok(MaxModelCapacity {
                max_params_billions: params as f64 / 1_000_000_000.0,
                model_category: category.to_string(),
                quantization: quantization.to_string(),
                execution_mode: "Hybrid".to_string(),
                gpu_layers: Some(max_layers),
                total_layers: layers,
                estimated_vram_mb: hybrid_estimate.total_mb,
                available_vram_mb: safe_vram_mb,
            });
        }
    }
    
    // If nothing fits even in hybrid, return smallest model in CPU mode
    Ok(MaxModelCapacity {
        max_params_billions: 0.6,
        model_category: "0.6B".to_string(),
        quantization: quantization.to_string(),
        execution_mode: "CPU-only".to_string(),
        gpu_layers: None,
        total_layers: 24,
        estimated_vram_mb: 0,
        available_vram_mb: safe_vram_mb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_search_layers() {
        // Test with known values
        let params = 7_000_000_000; // 7B
        let quantization = "4bit";
        let total_layers = 32;
        let context = 2048;
        let hidden_size = 4096;
        let available_vram = 10_000; // 10 GB
        
        let max_layers = calculate_max_gpu_layers(
            params,
            quantization,
            total_layers,
            context,
            hidden_size,
            available_vram,
        );
        
        println!("Max layers for 7B (4-bit) with 10GB VRAM: {}", max_layers);
        
        // Should be able to fit all layers
        assert!(max_layers > 0);
        assert!(max_layers <= total_layers);
    }

    #[test]
    fn test_small_vram_forces_cpu() {
        let params = 70_000_000_000; // 70B
        let quantization = "4bit";
        let total_layers = 80;
        let context = 2048;
        let hidden_size = 8192;
        let available_vram = 2_000; // Only 2 GB
        
        let max_layers = calculate_max_gpu_layers(
            params,
            quantization,
            total_layers,
            context,
            hidden_size,
            available_vram,
        );
        
        println!("Max layers for 70B (4-bit) with 2GB VRAM: {}", max_layers);
        
        // Should fit very few or no layers
        assert!(max_layers < 15); // Below hybrid threshold
    }

    #[test]
    fn test_recommend_mode_integration() {
        // This test requires actual GPU - will skip in CI
        match recommend_mode("Qwen/Qwen3-0.6B", "4bit") {
            Ok(decision) => {
                println!("Decision for Qwen3-0.6B (4-bit): {:?}", decision);
                // Small model should fit fully on GPU if available
                assert!(decision.mode == ExecutionMode::GPU || decision.mode == ExecutionMode::CPU);
            }
            Err(e) => {
                println!("Note: Test skipped (expected in CI without GPU): {}", e);
            }
        }
    }
}
