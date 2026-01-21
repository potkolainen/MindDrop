import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './Models.css';

function Models({ onModelSelect, selectedModel, downloadQueue, downloadProgress }) {
  const [downloadedModels, setDownloadedModels] = useState([]);
  const [autoDecisions, setAutoDecisions] = useState({});
  const [loadingDecisions, setLoadingDecisions] = useState({});

  useEffect(() => {
    loadDownloadedModels();
  }, []);
  
  // Reload when downloads complete
  useEffect(() => {
    const completedIds = downloadQueue
      .filter(item => item.status === 'complete')
      .map(item => item.id);
    
    if (completedIds.length > 0) {
      loadDownloadedModels();
    }
  }, [downloadQueue]);

  const loadDownloadedModels = async () => {
    try {
      const models = await invoke('list_downloaded_models');
      setDownloadedModels(models);
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };

  const handleDelete = async (modelId) => {
    if (!confirm(`Delete model ${modelId}?`)) return;
    
    try {
      await invoke('delete_model', { modelId });
      await loadDownloadedModels();
      if (selectedModel?.id === modelId) {
        onModelSelect?.(null);
      }
    } catch (error) {
      console.error('Delete failed:', error);
      alert(`Delete failed: ${error}`);
    }
  };

  const handleQuantizationToggle = async (modelId, currentValue) => {
    try {
      await invoke('set_model_quantization', { 
        modelId, 
        quantization: !currentValue 
      });
      await loadDownloadedModels();
    } catch (error) {
      console.error('Failed to toggle quantization:', error);
      alert(`Failed to update quantization: ${error}`);
    }
  };

  const handleQuantizationChange = async (modelId, newValue) => {
    try {
      await invoke('set_model_quantization', { 
        modelId, 
        quantization: newValue 
      });
      await loadDownloadedModels();
    } catch (error) {
      console.error('Failed to update quantization:', error);
      alert(`Failed to update quantization: ${error}`);
    }
  };

  const handleExecutionModeChange = async (modelId, newMode) => {
    try {
      const model = downloadedModels.find(m => m.id === modelId);
      const config = {
        mode: newMode,
        gpu_layers: model?.gpu_layers || null,
        quantization: model?.quantization || 'native',
        is_auto: newMode === 'auto'
      };
      
      await invoke('set_execution_config', { modelId, config });
      await loadDownloadedModels();
      
      // If auto mode, fetch decision
      if (newMode === 'auto') {
        fetchAutoDecision(modelId, model.quantization);
      }
    } catch (error) {
      console.error('Failed to update execution mode:', error);
      alert(`Failed to update execution mode: ${error}`);
    }
  };

  const handleGPULayersChange = async (modelId, layers) => {
    try {
      const model = downloadedModels.find(m => m.id === modelId);
      const config = {
        mode: 'hybrid',
        gpu_layers: layers,
        quantization: model?.quantization || 'native',
        is_auto: false
      };
      
      await invoke('set_execution_config', { modelId, config });
      await loadDownloadedModels();
    } catch (error) {
      console.error('Failed to update GPU layers:', error);
    }
  };

  const fetchAutoDecision = async (modelId, quantization) => {
    setLoadingDecisions(prev => ({ ...prev, [modelId]: true }));
    try {
      const decision = await invoke('recommend_execution_mode', {
        modelId,
        quantization: quantization || 'native'
      });
      setAutoDecisions(prev => ({ ...prev, [modelId]: decision }));
    } catch (error) {
      console.error('Failed to fetch auto decision:', error);
    } finally {
      setLoadingDecisions(prev => ({ ...prev, [modelId]: false }));
    }
  };

  const getModelLayers = (modelId) => {
    if (modelId.includes('0.6B') || modelId.includes('600M')) return 24;
    if (modelId.includes('7B') || modelId.includes('8B')) return 32;
    if (modelId.includes('13B')) return 40;
    if (modelId.includes('20B')) return 40;
    if (modelId.includes('32B')) return 64;
    if (modelId.includes('70B')) return 80;
    return 32; // Default
  };

  const estimateParamsB = (modelId) => {
    if (!modelId) return null;

    // MoE patterns like "8x7B" or "8x7b"
    const moe = modelId.match(/(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*[bB]/);
    if (moe) {
      const experts = Number(moe[1]);
      const perExpert = Number(moe[2]);
      if (Number.isFinite(experts) && Number.isFinite(perExpert)) {
        return Math.round(experts * perExpert);
      }
    }

    // Plain patterns like "7B", "13b", "27B", "28B"
    const plain = modelId.match(/(\d+(?:\.\d+)?)\s*[bB](?!yte)/);
    if (plain) {
      const b = Number(plain[1]);
      if (Number.isFinite(b)) return Math.round(b);
    }

    return null;
  };

  const getQueuePosition = (modelId) => {
    const queuedItems = downloadQueue.filter(item => item.status === 'queued');
    const index = queuedItems.findIndex(item => item.id === modelId);
    return index >= 0 ? index + 1 : null;
  };

  return (
    <div className="models">
      <div className="header">
        <h2>🤖 My Models</h2>
        <p className="subtitle">Manage your downloaded models ({downloadedModels.length})</p>
      </div>
      
      {downloadQueue.length > 0 && (
        <div className="download-status-section">
          <h3>📥 Downloads in Progress</h3>
          {downloadQueue.map((item) => {
            const progress = downloadProgress[item.id];
            const queuePos = getQueuePosition(item.id);
            return (
              <div key={item.id} className="download-status-item">
                <div className="status-header">
                  <span className="status-model-name">{item.id}</span>
                  {item.status === 'downloading' && progress && (
                    <span className="status-progress">{progress.progress}%</span>
                  )}
                  {item.status === 'queued' && queuePos && (
                    <span className="status-badge queued">#{queuePos} in queue</span>
                  )}
                  {item.status === 'downloading' && (
                    <span className="status-badge downloading">⬇️ Downloading</span>
                  )}
                  {item.status === 'paused' && (
                    <span className="status-badge paused">⏸ Paused</span>
                  )}
                  {item.status === 'complete' && (
                    <span className="status-badge complete">✓ Complete</span>
                  )}
                  {item.status === 'error' && (
                    <span className="status-badge error">✗ Error</span>
                  )}
                </div>
                {item.status === 'downloading' && progress && (
                  <>
                    <div className="mini-progress-bar">
                      <div 
                        className="mini-progress-fill" 
                        style={{ width: `${progress.progress || 0}%` }}
                      />
                    </div>
                    <span className="status-message">{progress.message}</span>
                  </>
                )}
              </div>
            );
          })}
        </div>
      )}

      {downloadedModels.length === 0 && downloadQueue.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📦</div>
          <p>No models downloaded yet</p>
          <p className="hint">Go to Library to download models</p>
        </div>
      ) : (
        <div className="model-list">
          {downloadedModels.map((model) => {
            const paramsB = estimateParamsB(model.id);
            const totalLayers = getModelLayers(model.id);
            const executionMode = model.execution_mode || 'auto';
            const gpuLayers = model.gpu_layers || Math.floor(totalLayers / 2);
            const autoDecision = autoDecisions[model.id];
            const loadingDecision = loadingDecisions[model.id];
            
            return (
              <div 
                key={model.id} 
                className={`model-card ${selectedModel?.id === model.id ? 'selected' : ''}`}
              >
                <div className="model-info">
                  <h4>{model.id}</h4>
                  <div className="model-meta">
                    <span className="badge">{model.engine}</span>
                    <span className="badge">{model.task}</span>
                    {paramsB != null && <span className="badge">~{paramsB}B</span>}
                    {model.quantization === '4bit' && <span className="badge badge-quant-4">⚡ 4-bit</span>}
                    {model.quantization === '8bit' && <span className="badge badge-quant-8">⚡ 8-bit</span>}
                    {model.quantization === 'fp16' && <span className="badge badge-quant-fp16">FP16</span>}
                    {model.quantization === 'native' && <span className="badge badge-quant-native">Native</span>}
                  </div>
                </div>
                
                <div className="model-execution-config">
                  <div className="config-row">
                    <label>Execution Mode:</label>
                    <select 
                      value={executionMode}
                      onChange={(e) => handleExecutionModeChange(model.id, e.target.value)}
                      className="execution-mode-select"
                    >
                      <option value="auto">🤖 Auto (Recommended)</option>
                      <option value="gpu">⚡ GPU Only</option>
                      <option value="hybrid">⚙️ Hybrid (CPU+GPU)</option>
                      <option value="cpu">🖥️ CPU Only</option>
                    </select>
                  </div>
                  
                  {executionMode === 'hybrid' && (
                    <div className="config-row hybrid-slider">
                      <label>GPU Layers: {gpuLayers} / {totalLayers}</label>
                      <input 
                        type="range"
                        min="0"
                        max={totalLayers}
                        value={gpuLayers}
                        onChange={(e) => handleGPULayersChange(model.id, parseInt(e.target.value))}
                        className="gpu-layers-slider"
                      />
                      <div className="slider-labels">
                        <span>CPU ←</span>
                        <span>→ GPU</span>
                      </div>
                    </div>
                  )}
                  
                  {executionMode === 'auto' && (
                    <div className="auto-decision-info">
                      {loadingDecision ? (
                        <small className="calculating">Calculating optimal mode...</small>
                      ) : autoDecision ? (
                        <small>
                          Auto: {autoDecision.mode.toLowerCase()} mode
                          {autoDecision.gpu_layers && ` (${autoDecision.gpu_layers}/${totalLayers} layers)`}
                        </small>
                      ) : (
                        <small 
                          className="fetch-decision"
                          onClick={() => fetchAutoDecision(model.id, model.quantization)}
                        >
                          Click to see auto mode decision
                        </small>
                      )}
                    </div>
                  )}
                </div>
                
                <div className="model-actions">
                  <div className="quantization-selector">
                    <label className="quant-label">Precision:</label>
                    <select 
                      value={model.quantization || 'native'} 
                      onChange={(e) => handleQuantizationChange(model.id, e.target.value)}
                      className="quant-select"
                      title="Select quantization level: Native (model default), FP16 (force 16-bit float), 8-bit (50% memory), or 4-bit (75% memory)"
                    >
                      <option value="native">Native (Auto)</option>
                      <option value="fp16">FP16</option>
                      <option value="8bit">8-bit</option>
                      <option value="4bit">4-bit</option>
                    </select>
                  </div>
                  <button
                    onClick={() => onModelSelect?.(model)}
                    className={selectedModel?.id === model.id ? 'btn-selected' : 'btn-select'}
                  >
                    {selectedModel?.id === model.id ? '✓ Selected' : 'Select'}
                  </button>
                  <button
                    onClick={() => handleDelete(model.id)}
                    className="btn-delete"
                  >
                    🗑️
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default Models;
