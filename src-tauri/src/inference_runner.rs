use crate::auto_mode::decide_execution_mode;
use crate::execution_config::{ExecutionConfig, ExecutionMode};
use crate::ai_runner::{run_python_inference, InferenceResult};
use crate::memory_estimator::get_model_layers;

/// Maximum number of retry attempts when OOM errors occur
const MAX_RETRIES: u32 = 2;

/// Run inference with automatic fallback on OOM errors
/// 
/// This function implements intelligent retry logic:
/// 1. Executes inference with the provided configuration
/// 2. Detects OOM errors in stderr/error messages
/// 3. Downgrades execution config and retries
/// 4. Falls through GPU → Hybrid(75%) → Hybrid(50%) → CPU
pub async fn run_with_fallback(
    model_id: &str,
    prompt: &str,
    max_tokens: usize,
    config: &ExecutionConfig,
) -> Result<InferenceResult, String> {
    let mut current_config = config.clone();
    let mut retry_count = 0;
    
    loop {
        // Determine actual execution strategy
        let (mode, gpu_layers) = match current_config.mode {
            ExecutionMode::Auto => {
                match decide_execution_mode(
                    model_id,
                    &current_config.quantization,
                    max_tokens as u32,
                ) {
                    Ok(decision) => {
                        log::info!("Auto mode decision: {}", decision.reason);
                        (decision.mode, decision.gpu_layers)
                    }
                    Err(e) => {
                        log::error!("Auto mode decision failed: {}", e);
                        // Fall back to CPU mode on error
                        (ExecutionMode::CPU, None)
                    }
                }
            }
            ExecutionMode::GPU => (ExecutionMode::GPU, None),
            ExecutionMode::Hybrid => (ExecutionMode::Hybrid, current_config.gpu_layers),
            ExecutionMode::CPU => (ExecutionMode::CPU, None),
        };
        
        // Log execution attempt
        log::info!(
            "Executing inference with {:?}, gpu_layers: {:?}, attempt {}/{}",
            mode,
            gpu_layers,
            retry_count + 1,
            MAX_RETRIES + 1
        );
        
        // Execute inference
        match execute_inference(
            model_id,
            prompt,
            max_tokens,
            &mode,
            gpu_layers,
            &current_config.quantization
        ).await {
            Ok(result) => {
                log::info!("Inference succeeded with {:?}", mode);
                return Ok(result);
            }
            Err(error) if is_oom_error(&error) && retry_count < MAX_RETRIES => {
                retry_count += 1;
                log::warn!(
                    "OOM error detected (retry {}/{}): {}",
                    retry_count,
                    MAX_RETRIES,
                    error
                );
                
                // Downgrade configuration
                current_config = downgrade_config(current_config, mode, gpu_layers, model_id);
                log::info!("Downgraded to {:?}", current_config.mode);
            }
            Err(error) => {
                log::error!("Inference failed: {}", error);
                return Err(error);
            }
        }
    }
}

/// Execute inference with specific execution mode parameters
async fn execute_inference(
    model_id: &str,
    prompt: &str,
    max_tokens: usize,
    mode: &ExecutionMode,
    gpu_layers: Option<u32>,
    quantization: &str,
) -> Result<InferenceResult, String> {
    // Log the execution mode being used
    let mode_str = match mode {
        ExecutionMode::Auto => "auto",
        ExecutionMode::GPU => "gpu",
        ExecutionMode::Hybrid => "hybrid",
        ExecutionMode::CPU => "cpu",
    };
    
    log::info!(
        "Calling Python inference: mode={}, gpu_layers={:?}, quantization={}",
        mode_str,
        gpu_layers,
        quantization
    );
    
    // Call Python inference with execution mode parameters
    run_python_inference(
        model_id, 
        prompt, 
        max_tokens, 
        quantization,
        Some(mode_str),
        gpu_layers
    )
}

/// Detect if an error message indicates an Out-Of-Memory condition
fn is_oom_error(error: &str) -> bool {
    let error_lower = error.to_lowercase();
    
    // Common OOM error patterns
    error_lower.contains("out of memory") ||
    error_lower.contains("oom") ||
    error_lower.contains("cuda out of memory") ||
    error_lower.contains("cudaerror: out of memory") ||
    error_lower.contains("cuda error") && error_lower.contains("memory") ||
    error_lower.contains("torch.cuda.outofmemoryerror") ||
    error_lower.contains("allocation failed")
}

/// Downgrade execution configuration when OOM occurs
/// 
/// Fallback sequence:
/// - GPU/Auto → Hybrid with 75% of layers
/// - Hybrid(75%) → Hybrid with 50% of layers
/// - Hybrid(50%) → CPU only
/// - CPU → Cannot downgrade further
fn downgrade_config(
    config: ExecutionConfig,
    mode: ExecutionMode,
    gpu_layers: Option<u32>,
    model_id: &str,
) -> ExecutionConfig {
    let total_layers = get_model_layers(model_id);
    
    match mode {
        ExecutionMode::GPU | ExecutionMode::Auto => {
            // First downgrade: Try hybrid with 75% of layers
            let new_layers = (total_layers as f64 * 0.75) as u32;
            log::info!(
                "Downgrading from {:?} to Hybrid mode with {} / {} layers",
                mode,
                new_layers,
                total_layers
            );
            
            ExecutionConfig {
                mode: ExecutionMode::Hybrid,
                gpu_layers: Some(new_layers),
                ..config
            }
        }
        ExecutionMode::Hybrid => {
            // Check current layer count
            let current_layers = gpu_layers.unwrap_or(total_layers);
            
            // If we're using more than 50%, try 50%
            if current_layers > (total_layers / 2) {
                let new_layers = total_layers / 2;
                log::info!(
                    "Downgrading from Hybrid({}/{}) to Hybrid({}/{})",
                    current_layers,
                    total_layers,
                    new_layers,
                    total_layers
                );
                
                ExecutionConfig {
                    mode: ExecutionMode::Hybrid,
                    gpu_layers: Some(new_layers),
                    ..config
                }
            } else {
                // Already at 50% or less, fall back to CPU
                log::info!(
                    "Downgrading from Hybrid({}/{}) to CPU mode",
                    current_layers,
                    total_layers
                );
                
                ExecutionConfig {
                    mode: ExecutionMode::CPU,
                    gpu_layers: None,
                    ..config
                }
            }
        }
        ExecutionMode::CPU => {
            // Cannot downgrade further
            log::warn!("Already in CPU mode, cannot downgrade further");
            config
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oom_detection() {
        assert!(is_oom_error("CUDA out of memory"));
        assert!(is_oom_error("torch.cuda.OutOfMemoryError"));
        assert!(is_oom_error("OOM when allocating tensor"));
        assert!(is_oom_error("CUDA Error: out of memory"));
        assert!(is_oom_error("Allocation failed"));
        
        assert!(!is_oom_error("Model not found"));
        assert!(!is_oom_error("Invalid input"));
        assert!(!is_oom_error("Connection timeout"));
    }

    #[test]
    fn test_downgrade_from_gpu() {
        let config = ExecutionConfig {
            mode: ExecutionMode::GPU,
            gpu_layers: None,
            quantization: "4bit".to_string(),
            is_auto: false,
        };
        
        let downgraded = downgrade_config(
            config,
            ExecutionMode::GPU,
            None,
            "Qwen/Qwen3-8B"
        );
        
        assert!(matches!(downgraded.mode, ExecutionMode::Hybrid));
        assert_eq!(downgraded.gpu_layers, Some(24)); // 75% of 32 layers
    }

    #[test]
    fn test_downgrade_from_hybrid() {
        let config = ExecutionConfig {
            mode: ExecutionMode::Hybrid,
            gpu_layers: Some(24), // 75% of 32
            quantization: "4bit".to_string(),
            is_auto: false,
        };
        
        let downgraded = downgrade_config(
            config,
            ExecutionMode::Hybrid,
            Some(24),
            "Qwen/Qwen3-8B"
        );
        
        // Should go to 50% (16 layers)
        assert!(matches!(downgraded.mode, ExecutionMode::Hybrid));
        assert_eq!(downgraded.gpu_layers, Some(16)); // 50% of 32 layers
    }

    #[test]
    fn test_downgrade_hybrid_to_cpu() {
        let config = ExecutionConfig {
            mode: ExecutionMode::Hybrid,
            gpu_layers: Some(16), // 50% of 32
            quantization: "4bit".to_string(),
            is_auto: false,
        };
        
        let downgraded = downgrade_config(
            config,
            ExecutionMode::Hybrid,
            Some(16),
            "Qwen/Qwen3-8B"
        );
        
        // Should go to CPU mode
        assert!(matches!(downgraded.mode, ExecutionMode::CPU));
        assert_eq!(downgraded.gpu_layers, None);
    }

    #[test]
    fn test_cpu_no_downgrade() {
        let config = ExecutionConfig {
            mode: ExecutionMode::CPU,
            gpu_layers: None,
            quantization: "4bit".to_string(),
            is_auto: false,
        };
        
        let downgraded = downgrade_config(
            config.clone(),
            ExecutionMode::CPU,
            None,
            "Qwen/Qwen3-8B"
        );
        
        // Should remain unchanged
        assert!(matches!(downgraded.mode, ExecutionMode::CPU));
        assert_eq!(downgraded.gpu_layers, None);
    }
}
