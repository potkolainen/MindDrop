use serde::{Deserialize, Serialize};

/// Memory usage breakdown for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub model_weights_mb: u32,
    pub kv_cache_mb: u32,
    pub activations_mb: u32,
    pub overhead_mb: u32,
    pub total_mb: u32,
}

/// Estimate VRAM usage for a model configuration
pub fn estimate_vram_usage(
    model_params: u64,      // e.g., 7_000_000_000 for 7B
    quantization: &str,     // "4bit", "8bit", "fp16", "native"
    num_layers: u32,        // Total layers in model
    gpu_layers: u32,        // Layers on GPU
    context_length: u32,    // Max context tokens
    hidden_size: u32,       // e.g., 4096 for most models
) -> MemoryEstimate {
    // Bytes per parameter based on quantization
    let bytes_per_param = match quantization {
        "4bit" => 0.5,
        "8bit" => 1.0,
        "fp16" => 2.0,
        "native" => 2.0,
        _ => 2.0,  // Default to fp16
    };
    
    // Model weights (only GPU portion)
    let gpu_fraction = gpu_layers as f64 / num_layers as f64;
    let params_on_gpu = model_params as f64 * gpu_fraction;
    let model_weights_mb = (params_on_gpu * bytes_per_param / 1_048_576.0) as u32;
    
    // KV cache: 2 × layers × hidden_size × context × bytes
    // (key and value tensors, only for GPU layers)
    // Using fp16 (2 bytes) for KV cache storage
    let kv_cache_mb = (2.0 * gpu_layers as f64 * hidden_size as f64 * 
                       context_length as f64 * 2.0 / 1_048_576.0) as u32;
    
    // Activations (~20% of model weights)
    // Temporary buffers during forward pass
    let activations_mb = (model_weights_mb as f64 * 0.20) as u32;
    
    // Overhead (fragmentation + PyTorch reserved: 15%)
    let subtotal = model_weights_mb + kv_cache_mb + activations_mb;
    let overhead_mb = (subtotal as f64 * 0.15) as u32;
    
    let total_mb = subtotal + overhead_mb;
    
    MemoryEstimate {
        model_weights_mb,
        kv_cache_mb,
        activations_mb,
        overhead_mb,
        total_mb,
    }
}

/// Quick lookup for common model sizes
/// Returns approximate parameter count
pub fn get_model_params(model_id: &str) -> u64 {
    let model_lower = model_id.to_lowercase();
    
    // Match common patterns
    if model_lower.contains("0.6b") || model_lower.contains("600m") {
        600_000_000
    } else if model_lower.contains("1.5b") || model_lower.contains("1b") {
        1_500_000_000
    } else if model_lower.contains("3b") {
        3_000_000_000
    } else if model_lower.contains("7b") {
        7_000_000_000
    } else if model_lower.contains("8b") {
        8_000_000_000
    } else if model_lower.contains("13b") {
        13_000_000_000
    } else if model_lower.contains("14b") {
        14_000_000_000
    } else if model_lower.contains("20b") {
        20_000_000_000
    } else if model_lower.contains("32b") {
        32_000_000_000
    } else if model_lower.contains("34b") {
        34_000_000_000
    } else if model_lower.contains("70b") {
        70_000_000_000
    } else if model_lower.contains("72b") {
        72_000_000_000
    } else {
        // Default: assume 7B if unknown
        7_000_000_000
    }
}

/// Estimate number of layers for a model
/// These are rough approximations based on common architectures
pub fn get_model_layers(model_id: &str) -> u32 {
    let model_lower = model_id.to_lowercase();
    
    if model_lower.contains("0.6b") { 24 }
    else if model_lower.contains("1b") || model_lower.contains("1.5b") { 24 }
    else if model_lower.contains("3b") { 28 }
    else if model_lower.contains("7b") || model_lower.contains("8b") { 32 }
    else if model_lower.contains("13b") || model_lower.contains("14b") { 40 }
    else if model_lower.contains("20b") { 40 }
    else if model_lower.contains("32b") || model_lower.contains("34b") { 64 }
    else if model_lower.contains("70b") || model_lower.contains("72b") { 80 }
    else { 32 }  // Default
}

/// Get hidden size for common model architectures
pub fn get_hidden_size(model_id: &str) -> u32 {
    let model_lower = model_id.to_lowercase();
    
    // Qwen models
    if model_lower.contains("qwen") {
        if model_lower.contains("0.6b") { 896 }
        else if model_lower.contains("1.5b") { 1536 }
        else if model_lower.contains("7b") || model_lower.contains("8b") { 3584 }
        else if model_lower.contains("14b") { 5120 }
        else if model_lower.contains("32b") { 5120 }
        else if model_lower.contains("72b") { 8192 }
        else { 4096 }
    }
    // Llama models
    else if model_lower.contains("llama") {
        if model_lower.contains("7b") || model_lower.contains("8b") { 4096 }
        else if model_lower.contains("13b") { 5120 }
        else if model_lower.contains("70b") { 8192 }
        else { 4096 }
    }
    // Default for unknown models
    else {
        4096
    }
}

/// Calculate maximum context length based on available VRAM
/// Returns conservative estimate
pub fn estimate_max_context_length(
    model_id: &str,
    quantization: &str,
    gpu_layers: u32,
    available_vram_mb: u32,
) -> u32 {
    let params = get_model_params(model_id);
    let layers = get_model_layers(model_id);
    let hidden_size = get_hidden_size(model_id);
    
    // Binary search for max context length
    let mut low = 512;
    let mut high = 128_000;  // Maximum reasonable context
    let mut best = 512;
    
    while low <= high {
        let mid = (low + high) / 2;
        let estimate = estimate_vram_usage(
            params,
            quantization,
            layers,
            gpu_layers,
            mid,
            hidden_size,
        );
        
        // Use 80% of available VRAM (20% safety margin)
        let safe_vram = (available_vram_mb as f64 * 0.80) as u32;
        
        if estimate.total_mb <= safe_vram {
            best = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_params_lookup() {
        assert_eq!(get_model_params("Qwen/Qwen3-0.6B"), 600_000_000);
        assert_eq!(get_model_params("Qwen/Qwen3-8B"), 8_000_000_000);
        assert_eq!(get_model_params("meta-llama/Llama-3-7B"), 7_000_000_000);
        assert_eq!(get_model_params("TeichAI/gpt-oss-20b"), 20_000_000_000);
        assert_eq!(get_model_params("Qwen/Qwen3-32B"), 32_000_000_000);
    }

    #[test]
    fn test_model_layers_lookup() {
        assert_eq!(get_model_layers("Qwen/Qwen3-0.6B"), 24);
        assert_eq!(get_model_layers("Qwen/Qwen3-8B"), 32);
        assert_eq!(get_model_layers("TeichAI/gpt-oss-20b"), 40);
        assert_eq!(get_model_layers("Qwen/Qwen3-32B"), 64);
    }

    #[test]
    fn test_hidden_size_lookup() {
        assert_eq!(get_hidden_size("Qwen/Qwen3-0.6B"), 896);
        assert_eq!(get_hidden_size("Qwen/Qwen3-8B"), 3584);
        assert_eq!(get_hidden_size("Qwen/Qwen3-7B"), 3584);
        assert_eq!(get_hidden_size("meta-llama/Llama-3-7B"), 4096);
    }

    #[test]
    fn test_vram_estimation_qwen_0_6b() {
        // Qwen3-0.6B, 4-bit, full GPU, 2k context
        let estimate = estimate_vram_usage(
            600_000_000,  // 0.6B params
            "4bit",
            24,           // total layers
            24,           // all on GPU
            2048,         // context
            896,          // hidden size
        );
        
        // Model weights: 600M × 0.5 bytes = 300MB / 1.048576 ≈ 286 MB
        // KV cache: 2 × 24 × 896 × 2048 × 2 bytes ≈ 168 MB
        // Activations: ~57 MB (20% of weights)
        // Overhead: 15% of subtotal
        // Total should be around 590 MB
        
        println!("Qwen3-0.6B estimate: {:?}", estimate);
        assert!(estimate.total_mb > 500 && estimate.total_mb < 700);
    }

    #[test]
    fn test_vram_estimation_qwen_8b_4bit() {
        // Qwen3-8B, 4-bit, full GPU, 2k context
        let estimate = estimate_vram_usage(
            8_000_000_000,  // 8B params
            "4bit",
            32,             // total layers
            32,             // all on GPU
            2048,           // context
            3584,           // hidden size
        );
        
        // Model weights: 8B × 0.5 bytes ≈ 3815 MB
        // KV cache: 2 × 32 × 3584 × 2048 × 2 bytes ≈ 896 MB
        // Should be around 5-6 GB total
        
        println!("Qwen3-8B (4-bit) estimate: {:?}", estimate);
        assert!(estimate.total_mb > 5000 && estimate.total_mb < 7000);
    }

    #[test]
    fn test_vram_estimation_20b_4bit_hybrid() {
        // 20B model, 4-bit, hybrid (20/40 layers), 2k context
        let estimate = estimate_vram_usage(
            20_000_000_000,  // 20B params
            "4bit",
            40,              // total layers
            20,              // half on GPU
            2048,            // context
            4096,            // hidden size
        );
        
        // Model weights: 20B × 0.5 (4-bit) × 0.5 (half layers) ≈ 4768 MB
        // KV cache: 2 × 20 × 4096 × 2048 × 2 bytes ≈ 656 MB
        // Should be around 6-8 GB
        
        println!("20B (4-bit, hybrid) estimate: {:?}", estimate);
        assert!(estimate.total_mb > 6000 && estimate.total_mb < 9000);
    }

    #[test]
    fn test_hybrid_reduces_vram() {
        let params = 20_000_000_000;
        let quantization = "4bit";
        let layers = 40;
        let context = 2048;
        let hidden_size = 4096;
        
        let full_gpu = estimate_vram_usage(params, quantization, layers, layers, context, hidden_size);
        let half_gpu = estimate_vram_usage(params, quantization, layers, layers / 2, context, hidden_size);
        
        // Half layers should use roughly half the VRAM
        assert!(half_gpu.total_mb < full_gpu.total_mb);
        assert!(half_gpu.total_mb > full_gpu.total_mb / 3);  // More than 1/3 due to overhead
    }

    #[test]
    fn test_quantization_reduces_vram() {
        let params = 7_000_000_000;
        let layers = 32;
        let context = 2048;
        let hidden_size = 4096;
        
        let native = estimate_vram_usage(params, "native", layers, layers, context, hidden_size);
        let fp16 = estimate_vram_usage(params, "fp16", layers, layers, context, hidden_size);
        let bit8 = estimate_vram_usage(params, "8bit", layers, layers, context, hidden_size);
        let bit4 = estimate_vram_usage(params, "4bit", layers, layers, context, hidden_size);
        
        println!("7B model VRAM usage:");
        println!("  Native: {} MB", native.total_mb);
        println!("  FP16:   {} MB", fp16.total_mb);
        println!("  8-bit:  {} MB", bit8.total_mb);
        println!("  4-bit:  {} MB", bit4.total_mb);
        
        // More aggressive quantization = less VRAM
        assert!(native.total_mb == fp16.total_mb);  // Same
        assert!(bit8.total_mb < fp16.total_mb);
        assert!(bit4.total_mb < bit8.total_mb);
    }

    #[test]
    fn test_max_context_estimation() {
        // With 14GB free VRAM, 4-bit Qwen3-8B should support long context
        let max_ctx = estimate_max_context_length(
            "Qwen/Qwen3-8B",
            "4bit",
            32,      // all layers
            14_000,  // 14GB VRAM
        );
        
        println!("Max context for Qwen3-8B (4-bit, 14GB VRAM): {}", max_ctx);
        // Should support at least 8k context, possibly up to 32k
        assert!(max_ctx >= 8_000);
    }
}
