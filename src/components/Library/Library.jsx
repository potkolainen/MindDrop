import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-shell';
import './Library.css';

function Library({ systemInfo, downloadQueue, setDownloadQueue, currentDownload, setCurrentDownload, downloadProgress, setDownloadProgress }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [downloadedModels, setDownloadedModels] = useState([]);
  const [modelSource, setModelSource] = useState('huggingface');
  const [showSourceDropdown, setShowSourceDropdown] = useState(false);
  const [selectedFilters, setSelectedFilters] = useState([]);
  const [sortBy, setSortBy] = useState('trending');
  const [showFilters, setShowFilters] = useState(false);
  const [showSortDropdown, setShowSortDropdown] = useState(false);
  const unlistenRef = useRef(null);
  const timeoutRefs = useRef([]);

  const MODEL_SOURCES = [
    { id: 'huggingface', name: 'Hugging Face', icon: '🤗', available: true },
    { id: 'github', name: 'GitHub', icon: '💻', available: true },
    { id: 'gitlab', name: 'GitLab', icon: '🦊', available: true },
    { id: 'openai', name: 'OpenAI', icon: '🔮', available: true },
    { id: 'civitai', name: 'CivitAI', icon: '🎨', available: true },
    { id: 'modelscope', name: 'ModelScope', icon: '🔭', available: true },
    { id: 'koboldai', name: 'KoboldAI', icon: '🐉', available: true },
    { id: 'paperswithcode', name: 'Papers With Code', icon: '📄', available: true },
    { id: 'zenodo', name: 'Zenodo', icon: '📦', available: true },
    { id: 'arxiv', name: 'arXiv', icon: '📚', available: true },
    { id: 'lmstudio', name: 'LM Studio', icon: '🏢', available: true },
    { id: 'ollama', name: 'Ollama', icon: '🦙', available: true },
    { id: 'nvidia', name: 'NVIDIA NGC', icon: '🟢', available: true },
    { id: 'aws', name: 'AWS SageMaker', icon: '🟠', available: true },
    { id: 'azure', name: 'Azure AI', icon: '☁️', available: true },
    { id: 'gcp', name: 'Google Cloud AI', icon: '🔵', available: true },
  ];

  const SORT_OPTIONS = [
    { id: 'trending', name: 'Trending', icon: '🔥' },
    { id: 'downloads', name: 'Most Downloads', icon: '⬇️' },
    { id: 'likes', name: 'Most Liked', icon: '❤️' },
    { id: 'recent', name: 'Recently Updated', icon: '🕐' },
    { id: 'alphabetical', name: 'Alphabetical', icon: '🔤' },
  ];

  // Filter categories by source
  const FILTER_CATEGORIES = {
    huggingface: [
      { category: 'Multimodal', tasks: ['audio-text-to-text', 'image-text-to-text', 'image-text-to-image', 'image-text-to-video', 'visual-question-answering', 'document-question-answering', 'video-text-to-text', 'any-to-any'] },
      { category: 'Computer Vision', tasks: ['depth-estimation', 'image-classification', 'object-detection', 'image-segmentation', 'text-to-image', 'image-to-text', 'image-to-image', 'image-to-video', 'unconditional-image-generation', 'video-classification', 'text-to-video', 'zero-shot-image-classification', 'mask-generation', 'zero-shot-object-detection', 'text-to-3d', 'image-to-3d', 'image-feature-extraction', 'keypoint-detection', 'video-to-video'] },
      { category: 'NLP', tasks: ['text-classification', 'token-classification', 'table-question-answering', 'question-answering', 'zero-shot-classification', 'translation', 'summarization', 'feature-extraction', 'text-generation', 'fill-mask', 'sentence-similarity', 'text-ranking'] },
      { category: 'Audio', tasks: ['text-to-speech', 'text-to-audio', 'automatic-speech-recognition', 'audio-to-audio', 'audio-classification', 'voice-activity-detection'] },
      { category: 'Tabular', tasks: ['tabular-classification', 'tabular-regression', 'time-series-forecasting'] },
      { category: 'Reinforcement Learning', tasks: ['reinforcement-learning', 'robotics'] },
      { category: 'Other', tasks: ['graph-machine-learning'] },
    ],
    civitai: [
      { category: 'Type', tasks: ['checkpoint', 'lora', 'textual-inversion', 'hypernetwork', 'aesthetic-gradient', 'controlnet', 'vae'] },
      { category: 'Base Model', tasks: ['sd-1.5', 'sd-2.0', 'sd-2.1', 'sdxl-1.0', 'sdxl-turbo', 'pony'] },
    ],
    github: [
      { category: 'Language', tasks: ['python', 'javascript', 'typescript', 'rust', 'go', 'c++', 'java'] },
      { category: 'Topic', tasks: ['machine-learning', 'deep-learning', 'nlp', 'computer-vision', 'pytorch', 'tensorflow'] },
    ],
  };

  const toggleFilter = (task) => {
    setSelectedFilters(prev =>
      prev.includes(task) ? prev.filter(t => t !== task) : [...prev, task]
    );
  };

  const clearFilters = () => {
    setSelectedFilters([]);
  };

  useEffect(() => {
    // Clear old results and reload when source changes
    console.log(`[Library] Model source changed to: ${modelSource}, filters:`, selectedFilters, 'sort:', sortBy);
    setSearchResults([]);
    setSearchQuery('');
    setIsSearching(true);
    
    // Load trending for the new source
    const loadData = async () => {
      try {
        console.log(`[Library] Loading trending for ${modelSource}...`);
        const results = await invoke('search_models', {
          source: modelSource,
          query: '',
          filters: selectedFilters,
          sort: sortBy
        });
        console.log(`[Library] Got ${results.length} results for ${modelSource}`);
        setSearchResults(results);
      } catch (error) {
        console.error(`[Library] Failed to load trending for ${modelSource}:`, error);
        setSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    };
    
    loadData();
  }, [modelSource]);

  useEffect(() => {
    console.log('[Library] Component mounted, setting up...');
    loadDownloadedModels();
    
    // Listen for download progress events
    let unlistenFn = null;
    
    const setupListener = async () => {
      try {
        console.log('[Library] Setting up download-progress event listener...');
        unlistenFn = await listen('download-progress', (event) => {
          console.log('[Library] ✅ Download progress event received:', event);
          console.log('[Library] Event payload:', event.payload);
          
          const { model_id, progress, message } = event.payload;
          
          console.log(`[Library] Updating progress for ${model_id}: ${progress}% - ${message}`);
          
          // Find the queue item - handle both full ID and partial ID (for lmstudio/ prefix)
          const queueItem = downloadQueue.find(item => 
            item.id === model_id || item.id.endsWith(`/${model_id}`)
          );
          const actualId = queueItem ? queueItem.id : model_id;
          
          setDownloadProgress(prev => {
            const updated = {
              ...prev,
              [actualId]: { progress, message }
            };
            console.log('[Library] Download progress state updated:', updated);
            return updated;
          });
          
          // If download completed, mark as complete and start next in queue
          if (Number(progress) >= 100) {
            setTimeout(() => {
              setDownloadQueue(prev => {
                const updated = prev.map(item =>
                  (item.id === actualId || item.id === model_id) ? { ...item, status: 'complete' } : item
                );
                // Remove completed item after delay
                setTimeout(() => {
                  setDownloadQueue(p => p.filter(item => item.id !== actualId && item.id !== model_id));
                  setDownloadProgress(p => {
                    const newP = { ...p };
                    delete newP[actualId];
                    delete newP[model_id];
                    return newP;
                  });
                }, 2000);
                return updated;
              });
              
              // Start next queued download
              setDownloadQueue(prev => {
                const next = prev.find(item => item.status === 'queued');
                if (next) {
                  startDownload(next.id);
                }
                return prev;
              });
            }, 500);
          }
        });
        
        console.log('[Library] ✅ Download progress listener set up successfully');
        unlistenRef.current = unlistenFn;
      } catch (error) {
        console.error('[Library] ❌ Failed to set up event listener:', error);
      }
    };
    
    setupListener();
    
    return () => {
      console.log('[Library] Component unmounting, cleaning up...');
      timeoutRefs.current.forEach(clearTimeout);
      if (unlistenFn) {
        console.log('[Library] Removing download progress listener');
        unlistenFn();
      }
    };
  }, []);

  const loadDownloadedModels = async () => {
    try {
      const models = await invoke('list_downloaded_models');
      setDownloadedModels(models.map(m => m.id));
    } catch (error) {
      console.error('Failed to load downloaded models:', error);
    }
  };

  const loadTrending = async () => {
    setIsSearching(true);
    try {
      const results = await invoke('search_models', {
        source: modelSource,
        query: '',  // Empty query to get trending/popular models from all sources
        filters: selectedFilters,
        sort: sortBy
      });
      setSearchResults(results);
    } catch (error) {
      console.error('Failed to load trending:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const estimateParamsB = (modelId) => {
    if (!modelId) return null;
    const moe = modelId.match(/(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*[bB]/);
    if (moe) {
      const experts = Number(moe[1]);
      const perExpert = Number(moe[2]);
      if (Number.isFinite(experts) && Number.isFinite(perExpert)) {
        return Math.round(experts * perExpert);
      }
    }
    const plain = modelId.match(/(\d+(?:\.\d+)?)\s*[bB](?!yte)/);
    if (plain) {
      const b = Number(plain[1]);
      if (Number.isFinite(b)) return Math.round(b);
    }
    return null;
  };

  const formatBytes = (bytes) => {
    if (!bytes) return null;
    const kb = bytes / 1024;
    const mb = kb / 1024;
    const gb = mb / 1024;
    
    if (gb >= 1) {
      return `${gb.toFixed(2)} GB`;
    } else if (mb >= 1) {
      return `${mb.toFixed(2)} MB`;
    } else if (kb >= 1) {
      return `${kb.toFixed(2)} KB`;
    }
    return `${bytes} bytes`;
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    try {
      const results = await invoke('search_models', {
        source: modelSource,
        query: searchQuery,
        filters: selectedFilters,
        sort: sortBy
      });
      setSearchResults(results);
      if (results.length === 0) {
        console.warn(`No results found for "${searchQuery}" on ${modelSource}`);
      }
    } catch (error) {
      console.error('Search failed:', error);
      alert(`Search failed: ${error}`);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handlePauseDownload = (modelId) => {
    setDownloadQueue(prev =>
      prev.map(item =>
        item.id === modelId ? { ...item, status: 'paused' } : item
      )
    );
    setCurrentDownload(null);
  };

  const handleResumeDownload = (modelId) => {
    setDownloadQueue(prev =>
      prev.map(item =>
        item.id === modelId ? { ...item, status: 'downloading' } : item
      )
    );
    startDownload(modelId);
  };

  const handleRemoveFromQueue = (modelId) => {
    setDownloadQueue(prev => prev.filter(item => item.id !== modelId));
    setDownloadProgress(prev => {
      const newProgress = { ...prev };
      delete newProgress[modelId];
      return newProgress;
    });
  };

  const startDownload = async (modelId) => {
    setCurrentDownload(modelId);
    setDownloadQueue(prev =>
      prev.map(item =>
        item.id === modelId ? { ...item, status: 'downloading' } : item
      )
    );
    
    try {
      console.log(`Starting download for ${modelId}`);
      
      // Get API key based on source
      let apiKey = null;
      if (modelId.startsWith('civitai/')) {
        apiKey = localStorage.getItem('civitai-api-key');
      } else if (modelId.startsWith('github/')) {
        apiKey = localStorage.getItem('github-token');
      } else if (modelId.startsWith('gitlab/')) {
        apiKey = localStorage.getItem('gitlab-token');
      } else if (modelId.startsWith('modelscope/')) {
        apiKey = localStorage.getItem('modelscope-api-key');
      } else if (modelId.startsWith('koboldai/')) {
        apiKey = localStorage.getItem('koboldai-api-key');
      } else if (modelId.startsWith('nvidia/')) {
        apiKey = localStorage.getItem('nvidia-api-key');
      } else if (modelId.startsWith('aws/')) {
        apiKey = localStorage.getItem('aws-api-key');
      } else if (modelId.startsWith('azure/')) {
        apiKey = localStorage.getItem('azure-api-key');
      } else if (modelId.startsWith('gcp/')) {
        apiKey = localStorage.getItem('gcp-api-key');
      }
      
      await invoke('download_model', { modelId, apiKey });
      console.log(`Download completed for ${modelId}`);
      await loadDownloadedModels();
    } catch (error) {
      console.error('Download failed:', error);
      setDownloadQueue(prev =>
        prev.map(item =>
          item.id === modelId ? { ...item, status: 'error', error: error.toString() } : item
        )
      );
    } finally {
      setCurrentDownload(null);
    }
  };

  const handleDownload = async (modelId) => {
    // Check if already downloaded
    if (downloadedModels.includes(modelId)) {
      alert(`${modelId} is already downloaded!`);
      return;
    }

    // Check if already in queue
    if (downloadQueue.some(item => item.id === modelId)) {
      alert(`${modelId} is already in the download queue!`);
      return;
    }

    console.log(`Adding ${modelId} to download queue`);
    
    // Add to queue
    const isFirstInQueue = downloadQueue.length === 0;
    const newItem = {
      id: modelId,
      status: isFirstInQueue ? 'downloading' : 'queued',
      addedAt: Date.now()
    };
    
    setDownloadQueue(prev => [...prev, newItem]);
    setDownloadProgress(prev => ({ ...prev, [modelId]: { progress: 0, message: 'Queued...' } }));
    
    // Start download if it's the first in queue
    if (isFirstInQueue) {
      startDownload(modelId);
    }
  };

  const getQueuePosition = (modelId) => {
    const queuedItems = downloadQueue.filter(item => item.status === 'queued');
    const index = queuedItems.findIndex(item => item.id === modelId);
    return index >= 0 ? index + 1 : null;
  };

  // Removed old error handling code
  const handleDownloadError = async (modelId) => {
    try {
      console.log(`Handling error for ${modelId}`);
      await loadDownloadedModels();
      const models = await invoke('list_downloaded_models');
      const isDownloaded = models.some(m => m.id === modelId);
      
      if (isDownloaded) {
        setDownloadQueue(prev => prev.map(d => 
          d.id === modelId ? { ...d, status: 'complete' } : d
        ));
        
        const timeoutId = setTimeout(() => {
          setDownloadQueue(prev => prev.filter(d => d.id !== modelId));
          setDownloadProgress(prev => {
            const newProgress = { ...prev };
            delete newProgress[modelId];
            return newProgress;
          });
        }, 3000);
        
        return () => clearTimeout(timeoutId);
      } else {
        throw new Error('Download completed but model not found');
      }
    } catch (error) {
      console.error('Error handling failed:', error);
    }
  };

  return (
    <div className="library">
      <div className="header">
        <h2>📚 Model Library</h2>
        <p className="subtitle">Browse and search AI models from multiple sources</p>
      </div>

      <div className="search-section">
        <div className="source-selector-wrapper">
          <button
            className="source-selector"
            onClick={() => setShowSourceDropdown(!showSourceDropdown)}
          >
            <span className="source-icon">
              {MODEL_SOURCES.find(s => s.id === modelSource)?.icon}
            </span>
            <span className="source-name">
              {MODEL_SOURCES.find(s => s.id === modelSource)?.name}
            </span>
            <span className="chevron">▼</span>
          </button>
          
          {showSourceDropdown && (
            <div className="source-dropdown">
              {MODEL_SOURCES.map(source => (
                <div
                  key={source.id}
                  className={`source-item ${source.id === modelSource ? 'active' : ''} ${!source.available ? 'disabled' : ''}`}
                  onClick={() => {
                    if (source.available) {
                      setModelSource(source.id);
                      setShowSourceDropdown(false);
                      setSearchResults([]);
                      if (source.id === 'huggingface') {
                        loadTrending();
                      }
                    }
                  }}
                >
                  <span className="source-icon">{source.icon}</span>
                  <span className="source-name">{source.name}</span>
                  {source.id === modelSource && <span className="check-icon">✓</span>}
                </div>
              ))}
            </div>
          )}
        </div>
        
        <div className="search-bar">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Search for models (e.g., gpt2, llama, mistral)..."
          />
          <button onClick={handleSearch} disabled={isSearching}>
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>

        <div className="filter-sort-bar">
          <button
            className="filter-toggle-btn"
            onClick={() => setShowFilters(!showFilters)}
          >
            🔍 Filters {selectedFilters.length > 0 && `(${selectedFilters.length})`}
          </button>

          <div className="sort-selector-wrapper">
            <button
              className="sort-selector"
              onClick={() => setShowSortDropdown(!showSortDropdown)}
            >
              {SORT_OPTIONS.find(s => s.id === sortBy)?.icon} {SORT_OPTIONS.find(s => s.id === sortBy)?.name}
              <span className="chevron">▼</span>
            </button>
            
            {showSortDropdown && (
              <div className="sort-dropdown">
                {SORT_OPTIONS.map(option => (
                  <div
                    key={option.id}
                    className={`sort-item ${option.id === sortBy ? 'active' : ''}`}
                    onClick={() => {
                      setSortBy(option.id);
                      setShowSortDropdown(false);
                      setSearchResults([]);
                      setIsSearching(true);
                      if (searchQuery.trim()) {
                        handleSearch();
                      } else {
                        loadTrending();
                      }
                    }}
                  >
                    <span>{option.icon} {option.name}</span>
                    {option.id === sortBy && <span className="check-icon">✓</span>}
                  </div>
                ))}
              </div>
            )}
          </div>

          {selectedFilters.length > 0 && (
            <button className="clear-filters-btn" onClick={clearFilters}>
              Clear Filters
            </button>
          )}
        </div>

        {showFilters && FILTER_CATEGORIES[modelSource] && (
          <div className="filters-panel">
            {FILTER_CATEGORIES[modelSource].map(category => (
              <div key={category.category} className="filter-category">
                <h4>{category.category}</h4>
                <div className="filter-tags">
                  {category.tasks.map(task => (
                    <button
                      key={task}
                      className={`filter-tag ${selectedFilters.includes(task) ? 'active' : ''}`}
                      onClick={() => toggleFilter(task)}
                    >
                      {task.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            <button className="apply-filters-btn" onClick={() => {
              if (searchQuery.trim()) {
                handleSearch();
              } else {
                loadTrending();
              }
            }}>
              Apply Filters
            </button>
          </div>
        )}

        {downloadQueue.length > 0 && (
          <div className="active-downloads">
            <h3>Download Queue ({downloadQueue.length})</h3>
            {downloadQueue.map((item) => {
              const progress = downloadProgress[item.id];
              const queuePos = getQueuePosition(item.id);
              return (
                <div key={item.id} className="download-item">
                  <div className="download-header">
                    <span className="download-name">{item.id}</span>
                    <div className="download-actions">
                      {item.status === 'queued' && queuePos && (
                        <span className="queue-position">#{queuePos} in queue</span>
                      )}
                      {item.status === 'downloading' && (
                        <span className="download-percent">{progress?.progress || 0}%</span>
                      )}
                      {item.status === 'downloading' && (
                        <button
                          className="btn-pause"
                          onClick={() => handlePauseDownload(item.id)}
                          title="Pause download"
                        >
                          ⏸ Pause
                        </button>
                      )}
                      {item.status === 'paused' && (
                        <button
                          className="btn-resume"
                          onClick={() => handleResumeDownload(item.id)}
                          title="Resume download"
                        >
                          ▶ Resume
                        </button>
                      )}
                      {(item.status === 'queued' || item.status === 'paused' || item.status === 'error') && (
                        <button
                          className="btn-remove"
                          onClick={() => handleRemoveFromQueue(item.id)}
                          title="Remove from queue"
                        >
                          ✕ Remove
                        </button>
                      )}
                      {item.status === 'complete' && <span className="status-complete">✓ Complete</span>}
                      {item.status === 'error' && <span className="status-error">✗ Error</span>}
                    </div>
                  </div>
                  {(item.status === 'downloading' || item.status === 'complete') && (
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{ width: `${progress?.progress || 0}%` }}
                      />
                    </div>
                  )}
                  <span className="download-status">
                    {item.status === 'queued' && 'Waiting to download...'}
                    {item.status === 'downloading' && (progress?.message || 'Downloading...')}
                    {item.status === 'paused' && 'Paused'}
                    {item.status === 'complete' && 'Download complete!'}
                    {item.status === 'error' && `Error: ${item.error || 'Unknown error'}`}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {isSearching && (
          <div className="loading-state">
            <p>Loading models...</p>
          </div>
        )}

        {!isSearching && searchResults.length > 0 && (
          <div className="search-results">
            {searchResults.map((model) => {
              const paramsB = estimateParamsB(model.id);
              const isDownloaded = downloadedModels.includes(model.id);
              const sourceIcon = MODEL_SOURCES.find(s => s.id === model.source)?.icon || '📦';
              const sizeStr = formatBytes(model.size_bytes);
              
              return (
                <div key={model.id} className="model-card">
                  <div className="model-info">
                    <div className="model-header">
                      <h4>{model.name || model.id}</h4>
                      <span className="source-badge" title={`From ${model.source}`}>
                        {sourceIcon}
                      </span>
                    </div>
                    {model.author && (
                      <div className="model-author">by {model.author}</div>
                    )}
                    {model.description && (
                      <div className="model-description">{model.description.slice(0, 120)}...</div>
                    )}
                    <div className="model-meta">
                      {model.downloads && <span>⬇️ {model.downloads.toLocaleString()}</span>}
                      {model.likes && <span>❤️ {model.likes.toLocaleString()}</span>}
                      {sizeStr && <span>💾 {sizeStr}</span>}
                      {model.stars && <span>⭐ {model.stars.toLocaleString()}</span>}
                      {paramsB != null && <span className="badge">~{paramsB}B</span>}
                      {isDownloaded && <span className="badge downloaded">Downloaded ✓</span>}
                    </div>
                    {model.tags.length > 0 && (
                      <div className="tags">
                        {model.tags.slice(0, 3).map((tag, idx) => (
                          <span key={idx} className="tag">{tag}</span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="model-actions">
                    <button
                      onClick={async () => {
                        try {
                          await open(model.url);
                        } catch (error) {
                          console.error('Failed to open URL:', error);
                        }
                      }}
                      className="btn-view"
                      title="View on source website"
                    >
                      🔗 View
                    </button>
                    <button
                      onClick={() => handleDownload(model.id)}
                      disabled={isDownloaded || downloadQueue.some(item => item.id === model.id)}
                      className={`btn-download ${isDownloaded ? 'downloaded' : ''}`}
                      title={isDownloaded ? 'Already downloaded' : 'Download this model'}
                    >
                      {isDownloaded ? 'Downloaded ✓' : downloadQueue.some(item => item.id === model.id) ? 'In Queue' : 'Download'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default Library;
