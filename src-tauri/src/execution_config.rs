use serde::{Deserialize, Serialize};

/// Execution mode for running AI models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionMode {
    /// Automatically choose best execution strategy based on system resources
    Auto,
    /// Run fully on GPU (requires sufficient VRAM)
    #[serde(rename = "gpu")]
    GPU,
    /// Split model between GPU and CPU (hybrid offloading)
    Hybrid,
    /// Run fully on CPU (slowest but most stable)
    #[serde(rename = "cpu")]
    CPU,
}

impl Default for ExecutionMode {
    fn default() -> Self {
        ExecutionMode::Auto
    }
}

impl std::fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionMode::Auto => write!(f, "Auto"),
            ExecutionMode::GPU => write!(f, "GPU"),
            ExecutionMode::Hybrid => write!(f, "Hybrid"),
            ExecutionMode::CPU => write!(f, "CPU"),
        }
    }
}

/// Configuration for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Execution mode
    pub mode: ExecutionMode,
    
    /// Number of layers to offload to GPU (for Hybrid mode)
    /// None means use auto-calculated value
    pub gpu_layers: Option<u32>,
    
    /// Quantization level: "native", "fp16", "8bit", "4bit"
    pub quantization: String,
    
    /// Whether this config was auto-generated or manually set
    #[serde(default)]
    pub is_auto: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        ExecutionConfig {
            mode: ExecutionMode::Auto,
            gpu_layers: None,
            quantization: "native".to_string(),
            is_auto: true,
        }
    }
}

#[allow(dead_code)]
impl ExecutionConfig {
    /// Create a new config with Auto mode
    pub fn new_auto(quantization: String) -> Self {
        ExecutionConfig {
            mode: ExecutionMode::Auto,
            gpu_layers: None,
            quantization,
            is_auto: true,
        }
    }
    
    /// Create a new config with GPU mode
    pub fn new_gpu(quantization: String) -> Self {
        ExecutionConfig {
            mode: ExecutionMode::GPU,
            gpu_layers: None,
            quantization,
            is_auto: false,
        }
    }
    
    /// Create a new config with Hybrid mode
    pub fn new_hybrid(quantization: String, gpu_layers: u32) -> Self {
        ExecutionConfig {
            mode: ExecutionMode::Hybrid,
            gpu_layers: Some(gpu_layers),
            quantization,
            is_auto: false,
        }
    }
    
    /// Create a new config with CPU mode
    pub fn new_cpu(quantization: String) -> Self {
        ExecutionConfig {
            mode: ExecutionMode::CPU,
            gpu_layers: Some(0), // Explicitly 0 layers on GPU
            quantization,
            is_auto: false,
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Check quantization is valid
        match self.quantization.as_str() {
            "native" | "fp16" | "8bit" | "4bit" => {},
            _ => return Err(format!("Invalid quantization: {}", self.quantization)),
        }
        
        // Check gpu_layers makes sense for mode
        match self.mode {
            ExecutionMode::GPU => {
                if self.gpu_layers.is_some() && self.gpu_layers != Some(0) {
                    // GPU mode shouldn't have explicit layer count (use all layers)
                    // But we allow None or 0
                }
            }
            ExecutionMode::Hybrid => {
                if let Some(layers) = self.gpu_layers {
                    if layers == 0 {
                        return Err("Hybrid mode requires gpu_layers > 0".to_string());
                    }
                }
            }
            ExecutionMode::CPU => {
                if self.gpu_layers.is_some() && self.gpu_layers != Some(0) {
                    return Err("CPU mode should have gpu_layers = 0 or None".to_string());
                }
            }
            ExecutionMode::Auto => {
                // Auto mode can have any gpu_layers (it's a hint)
            }
        }
        
        Ok(())
    }
    
    /// Get a human-readable description of this config
    pub fn describe(&self) -> String {
        match self.mode {
            ExecutionMode::Auto => {
                format!("Auto mode ({})", self.quantization)
            }
            ExecutionMode::GPU => {
                format!("GPU mode ({})", self.quantization)
            }
            ExecutionMode::Hybrid => {
                let layers = self.gpu_layers.unwrap_or(0);
                format!("Hybrid mode ({}, {} GPU layers)", self.quantization, layers)
            }
            ExecutionMode::CPU => {
                format!("CPU mode ({})", self.quantization)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_execution_mode() {
        let mode = ExecutionMode::default();
        assert_eq!(mode, ExecutionMode::Auto);
    }

    #[test]
    fn test_default_execution_config() {
        let config = ExecutionConfig::default();
        assert_eq!(config.mode, ExecutionMode::Auto);
        assert_eq!(config.quantization, "native");
        assert!(config.is_auto);
        assert!(config.gpu_layers.is_none());
    }

    #[test]
    fn test_new_configs() {
        let auto = ExecutionConfig::new_auto("fp16".to_string());
        assert_eq!(auto.mode, ExecutionMode::Auto);
        assert!(auto.is_auto);

        let gpu = ExecutionConfig::new_gpu("4bit".to_string());
        assert_eq!(gpu.mode, ExecutionMode::GPU);
        assert!(!gpu.is_auto);

        let hybrid = ExecutionConfig::new_hybrid("8bit".to_string(), 20);
        assert_eq!(hybrid.mode, ExecutionMode::Hybrid);
        assert_eq!(hybrid.gpu_layers, Some(20));

        let cpu = ExecutionConfig::new_cpu("native".to_string());
        assert_eq!(cpu.mode, ExecutionMode::CPU);
        assert_eq!(cpu.gpu_layers, Some(0));
    }

    #[test]
    fn test_validation() {
        // Valid configs
        assert!(ExecutionConfig::new_auto("fp16".to_string()).validate().is_ok());
        assert!(ExecutionConfig::new_gpu("4bit".to_string()).validate().is_ok());
        assert!(ExecutionConfig::new_hybrid("8bit".to_string(), 20).validate().is_ok());
        assert!(ExecutionConfig::new_cpu("native".to_string()).validate().is_ok());

        // Invalid quantization
        let mut bad_quant = ExecutionConfig::default();
        bad_quant.quantization = "invalid".to_string();
        assert!(bad_quant.validate().is_err());

        // Invalid hybrid with 0 layers
        let mut bad_hybrid = ExecutionConfig::new_hybrid("fp16".to_string(), 0);
        assert!(bad_hybrid.validate().is_err());
    }

    #[test]
    fn test_describe() {
        let auto = ExecutionConfig::new_auto("fp16".to_string());
        assert_eq!(auto.describe(), "Auto mode (fp16)");

        let gpu = ExecutionConfig::new_gpu("4bit".to_string());
        assert_eq!(gpu.describe(), "GPU mode (4bit)");

        let hybrid = ExecutionConfig::new_hybrid("8bit".to_string(), 25);
        assert_eq!(hybrid.describe(), "Hybrid mode (8bit, 25 GPU layers)");

        let cpu = ExecutionConfig::new_cpu("native".to_string());
        assert_eq!(cpu.describe(), "CPU mode (native)");
    }

    #[test]
    fn test_serialization() {
        let config = ExecutionConfig::new_hybrid("fp16".to_string(), 30);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ExecutionConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.mode, deserialized.mode);
        assert_eq!(config.gpu_layers, deserialized.gpu_layers);
        assert_eq!(config.quantization, deserialized.quantization);
    }
}
