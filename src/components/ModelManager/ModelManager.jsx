import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './ModelManager.css';

function ModelManager({ systemInfo, onModelSelect, selectedModel }) {
  const [downloadedModels, setDownloadedModels] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadingModel, setDownloadingModel] = useState(null);

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

  useEffect(() => {
    loadDownloadedModels();
  }, []);

  const loadDownloadedModels = async () => {
    try {
      const models = await invoke('list_downloaded_models');
      setDownloadedModels(models);
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    try {
      const results = await invoke('search_huggingface', {
        query: searchQuery
      });
      setSearchResults(results);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleDownload = async (modelId) => {
    setIsDownloading(true);
    setDownloadingModel(modelId);
    
    try {
      await invoke('download_model', { modelId });
      await loadDownloadedModels();
    } catch (error) {
      console.error('Download failed:', error);
      alert(`Download failed: ${error}`);
    } finally {
      setIsDownloading(false);
      setDownloadingModel(null);
    }
  };

  const handleDelete = async (modelId) => {
    if (!confirm(`Delete model ${modelId}?`)) return;
    
    try {
      await invoke('delete_model', { modelId });
      await loadDownloadedModels();
      if (selectedModel?.id === modelId) {
        onModelSelect(null);
      }
    } catch (error) {
      console.error('Delete failed:', error);
      alert(`Delete failed: ${error}`);
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="model-manager">
      <div className="header">
        <h2>📚 Model Library</h2>
        <p className="subtitle">Download and manage AI models</p>
      </div>

      <div className="search-section">
        <h3>Search HuggingFace</h3>
        <div className="search-bar">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Search for models (e.g., gpt2, llama, mistral)..."
          />
          <button 
            onClick={handleSearch}
            disabled={isSearching}
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>

        {searchResults.length > 0 && (
          <div className="search-results">
            {searchResults.map((model) => (
              (() => {
                const paramsB = estimateParamsB(model.id);
                return (
              <div key={model.id} className="model-card search-result">
                <div className="model-info">
                  <h4>{model.id}</h4>
                  <div className="model-meta">
                    <span>⬇️ {model.downloads.toLocaleString()}</span>
                    <span>❤️ {model.likes.toLocaleString()}</span>
                    {paramsB != null && <span className="badge">~{paramsB}B</span>}
                  </div>
                  {model.tags.length > 0 && (
                    <div className="tags">
                      {model.tags.slice(0, 3).map((tag, idx) => (
                        <span key={idx} className="tag">{tag}</span>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => handleDownload(model.id)}
                  disabled={isDownloading}
                  className="btn-download"
                >
                  {downloadingModel === model.id
                    ? 'Downloading...'
                    : 'Download'}
                </button>
              </div>
                );
              })()
            ))}
          </div>
        )}
      </div>

      <div className="downloaded-section">
        <h3>Downloaded Models ({downloadedModels.length})</h3>
        
        {downloadedModels.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📦</div>
            <p>No models downloaded yet</p>
            <p className="hint">Search and download models above to get started</p>
          </div>
        ) : (
          <div className="model-list">
            {downloadedModels.map((model) => (
              (() => {
                const paramsB = estimateParamsB(model.id);
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
                  </div>
                </div>
                <div className="model-actions">
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
              })()
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelManager;
