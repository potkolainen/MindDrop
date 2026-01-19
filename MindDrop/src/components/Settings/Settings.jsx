import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import TTSAdvancedSettings from '../TTSAdvancedSettings/TTSAdvancedSettings';
import './Settings.css';

const SEARCH_ENGINES = [
  { id: 'duckduckgo', name: 'DuckDuckGo', icon: '🦆' },
  { id: 'google', name: 'Google', icon: '🔍' },
  { id: 'brave', name: 'Brave Search', icon: '🦁' },
  { id: 'bing', name: 'Bing', icon: '🅱️' },
  { id: 'qwant', name: 'Qwant', icon: '🇫🇷' },
  { id: 'google-scholar', name: 'Google Scholar', icon: '🎓' },
  { id: 'stackoverflow', name: 'Stack Overflow', icon: '📚' },
  { id: 'ecosia', name: 'Ecosia', icon: '🌱' },
  { id: 'github', name: 'GitHub Search', icon: '💻' },
  { id: 'devdocs', name: 'DevDocs', icon: '📖' },
  { id: 'arxiv', name: 'arXiv', icon: '📄' },
  { id: 'startpage', name: 'Startpage', icon: '🔐' },
  { id: 'metager', name: 'MetaGer', icon: '🔎' },
];

function Settings({ searchEngine, setSearchEngine }) {
  const [pythonInfo, setPythonInfo] = useState(null);
  const [downloadDir, setDownloadDir] = useState('');
  const [isChangingDir, setIsChangingDir] = useState(false);
  
  // Text-to-Speech settings
  const [availableVoices, setAvailableVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(() => {
    return localStorage.getItem('tts-voice') || '';
  });
  const [ttsAutoplay, setTtsAutoplay] = useState(() => {
    return localStorage.getItem('tts-autoplay') === 'true';
  });
  const [ttsSpeed, setTtsSpeed] = useState(() => {
    return parseFloat(localStorage.getItem('tts-speed')) || 1.0;
  });
  const [ttsVolume, setTtsVolume] = useState(() => {
    return parseFloat(localStorage.getItem('tts-volume')) || 1.0;
  });
  const [ttsPitch, setTtsPitch] = useState(() => {
    return parseFloat(localStorage.getItem('tts-pitch')) || 1.0;
  });
  const [previewText, setPreviewText] = useState('Hello! This is a preview of the selected voice.');
  const [showAdvancedTTS, setShowAdvancedTTS] = useState(false);
  const [advancedTTSSettings, setAdvancedTTSSettings] = useState(() => {
    const saved = localStorage.getItem('tts-advanced-settings');
    return saved ? JSON.parse(saved) : {
      prosody: 0.5,
      stability: 0.7,
      pauseMultiplier: 1.0,
      breathInsertion: 'light',
      warmth: 0.0,
      presence: 0.0,
      air: 0.0,
      reverbSize: 0.0,
      breathVolume: 0.3,
      deEsser: 0.5,
      normalize: true,
      limiter: true
    };
  });
  
  // API Credentials
  const [civitaiApiKey, setCivitaiApiKey] = useState(() => {
    return localStorage.getItem('civitai-api-key') || '';
  });
  const [githubToken, setGithubToken] = useState(() => {
    return localStorage.getItem('github-token') || '';
  });
  const [gitlabToken, setGitlabToken] = useState(() => {
    return localStorage.getItem('gitlab-token') || '';
  });
  const [openaiApiKey, setOpenaiApiKey] = useState(() => {
    return localStorage.getItem('openai-api-key') || '';
  });
  const [modelscopeApiKey, setModelscopeApiKey] = useState(() => {
    return localStorage.getItem('modelscope-api-key') || '';
  });
  const [koboldaiApiKey, setKoboldaiApiKey] = useState(() => {
    return localStorage.getItem('koboldai-api-key') || '';
  });
  const [nvidiaApiKey, setNvidiaApiKey] = useState(() => {
    return localStorage.getItem('nvidia-api-key') || '';
  });
  const [awsApiKey, setAwsApiKey] = useState(() => {
    return localStorage.getItem('aws-api-key') || '';
  });
  const [azureApiKey, setAzureApiKey] = useState(() => {
    return localStorage.getItem('azure-api-key') || '';
  });
  const [gcpApiKey, setGcpApiKey] = useState(() => {
    return localStorage.getItem('gcp-api-key') || '';
  });

  useEffect(() => {
    localStorage.setItem('civitai-api-key', civitaiApiKey);
  }, [civitaiApiKey]);

  useEffect(() => {
    localStorage.setItem('github-token', githubToken);
  }, [githubToken]);

  useEffect(() => {
    localStorage.setItem('gitlab-token', gitlabToken);
  }, [gitlabToken]);

  useEffect(() => {
    localStorage.setItem('openai-api-key', openaiApiKey);
  }, [openaiApiKey]);

  useEffect(() => {
    localStorage.setItem('modelscope-api-key', modelscopeApiKey);
  }, [modelscopeApiKey]);

  useEffect(() => {
    localStorage.setItem('koboldai-api-key', koboldaiApiKey);
  }, [koboldaiApiKey]);

  useEffect(() => {
    localStorage.setItem('nvidia-api-key', nvidiaApiKey);
  }, [nvidiaApiKey]);

  useEffect(() => {
    localStorage.setItem('aws-api-key', awsApiKey);
  }, [awsApiKey]);

  useEffect(() => {
    localStorage.setItem('azure-api-key', azureApiKey);
  }, [azureApiKey]);

  useEffect(() => {
    localStorage.setItem('gcp-api-key', gcpApiKey);
  }, [gcpApiKey]);

  useEffect(() => {
    localStorage.setItem('tts-voice', selectedVoice);
  }, [selectedVoice]);

  useEffect(() => {
    localStorage.setItem('tts-autoplay', ttsAutoplay);
  }, [ttsAutoplay]);

  useEffect(() => {
    localStorage.setItem('tts-speed', ttsSpeed.toString());
  }, [ttsSpeed]);

  useEffect(() => {
    localStorage.setItem('tts-volume', ttsVolume.toString());
  }, [ttsVolume]);

  useEffect(() => {
    localStorage.setItem('tts-pitch', ttsPitch.toString());
  }, [ttsPitch]);


  useEffect(() => {
    // Load F5-TTS voice variants with speaker embeddings
    const loadVoices = async () => {
      const f5Voices = [
        { id: 'male_neutral', name: 'Male - Neutral', gender: 'Male' },
        { id: 'male_deep', name: 'Male - Deep', gender: 'Male' },
        { id: 'male_soft', name: 'Male - Soft', gender: 'Male' },
        { id: 'female_neutral', name: 'Female - Neutral', gender: 'Female' },
        { id: 'female_warm', name: 'Female - Warm', gender: 'Female' },
        { id: 'female_bright', name: 'Female - Bright', gender: 'Female' },
        { id: 'androgynous', name: 'Androgynous', gender: 'Neutral' },
        { id: 'narrator', name: 'Narrator', gender: 'Female' },
      ];
      
      setAvailableVoices(f5Voices);
      
      if (!selectedVoice) {
        setSelectedVoice('female_neutral');
      }
    };

    loadVoices();
  }, []);

  useEffect(() => {
    checkPython();
    loadDownloadDir();
  }, []);

  const checkPython = async () => {
    try {
      const info = await invoke('check_python');
      setPythonInfo(info);
    } catch (error) {
      console.error('Failed to check Python:', error);
    }
  };

  const loadDownloadDir = async () => {
    try {
      const dir = await invoke('get_download_dir');
      setDownloadDir(dir);
    } catch (error) {
      console.error('Failed to get download directory:', error);
    }
  };

  const changeDownloadDir = async () => {
    try {
      setIsChangingDir(true);
      const newPath = prompt('Enter the full path to the download directory:', downloadDir);

      if (newPath && newPath.trim()) {
        const trimmed = newPath.trim();
        await invoke('set_download_dir', { path: trimmed });
        setDownloadDir(trimmed);
      }
    } catch (error) {
      console.error('Failed to change download directory:', error);
      alert('Failed to change directory: ' + error);
    } finally {
      setIsChangingDir(false);
    }
  };

  const previewVoice = async () => {
    const voiceToUse = selectedVoice || 'female_neutral';
    const text = previewText && previewText.trim().length > 0
      ? previewText.trim()
      : 'Hello! This is a preview of the selected voice.';
    
    try {
      // Build settings JSON
      const settings = JSON.stringify({
        warmth: advancedTTSSettings.warmth,
        presence: advancedTTSSettings.presence,
        air: advancedTTSSettings.air,
        reverb_size: advancedTTSSettings.reverbSize,
        breath_volume: advancedTTSSettings.breathVolume,
        de_esser: advancedTTSSettings.deEsser,
        normalize: advancedTTSSettings.normalize,
        limiter: advancedTTSSettings.limiter
      });
      
      const audioBytes = await invoke('text_to_speech', {
        text,
        voice: voiceToUse,
        speed: ttsSpeed,
        pitch: ttsPitch,
        prosody: advancedTTSSettings.prosody,
        stability: advancedTTSSettings.stability,
        settings: settings
      });
      // Tauri may return Uint8Array, Array, Buffer-like, or { data: [...] }
      const toBytes = (data) => {
        if (!data) return null;
        if (data instanceof Uint8Array) return data;
        if (Array.isArray(data)) return new Uint8Array(data);
        if (data.buffer instanceof ArrayBuffer) return new Uint8Array(data.buffer);
        if (data.data) return new Uint8Array(data.data);
        return null;
      };
      const byteArray = toBytes(audioBytes);
      if (!byteArray || byteArray.length === 0) {
        console.error('F5-TTS preview returned empty audio', { audioBytes });
        throw new Error('No audio returned from TTS engine');
      }
      
      // Convert bytes to blob and play
      const blob = new Blob([new Uint8Array(byteArray)], { type: 'audio/mpeg' });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      
      // Apply volume
      audio.volume = ttsVolume;
      
      await audio.play();
      audio.onended = () => URL.revokeObjectURL(url);
    } catch (error) {
      console.error('F5-TTS preview failed:', error);
    }
  };

  const handleAdvancedSave = (newSettings) => {
    setAdvancedTTSSettings(newSettings);
    // Also update main settings if they changed in advanced panel
    if (newSettings.voice) setSelectedVoice(newSettings.voice);
    if (newSettings.speed) setTtsSpeed(newSettings.speed);
    if (newSettings.pitch) setTtsPitch(newSettings.pitch);
  };

  return (
    <div className="settings">
      <div className="header">
        <h2>⚙️ Settings</h2>
      </div>

      <div className="settings-section">
        <h3>Text-to-Speech</h3>
        <div className="info-box">
          <p className="info-text hint" style={{ marginBottom: '12px' }}>
            🎙️ Uses F5-TTS with edge-tts voices and advanced post-processing for natural-sounding speech.
          </p>

          <div className="setting-item">
            <div className="setting-header">
              <label>Autoplay Messages</label>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={ttsAutoplay}
                  onChange={(e) => setTtsAutoplay(e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
            <p className="setting-hint">Automatically speak new assistant messages</p>
          </div>

          <div className="setting-item">
            <label htmlFor="tts-voice">Voice Variant</label>
            <select
              id="tts-voice"
              value={selectedVoice}
              onChange={(e) => setSelectedVoice(e.target.value)}
              className="select-input"
            >
              {availableVoices.map((voice, idx) => (
                <option key={voice.id || idx} value={voice.id}>
                  {voice.name} {voice.gender ? voice.gender === 'Male' ? '♂️' : '♀️' : ''}
                </option>
              ))}
            </select>
          </div>

          <div className="setting-item">
            <label htmlFor="tts-speed">
              Speed: {ttsSpeed.toFixed(1)}x
            </label>
            <div className="slider-container">
              <span className="slider-label">0.5x</span>
              <input
                id="tts-speed"
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={ttsSpeed}
                onChange={(e) => setTtsSpeed(parseFloat(e.target.value))}
                className="range-slider"
              />
              <span className="slider-label">2.0x</span>
            </div>
          </div>

          <div className="setting-item">
            <label htmlFor="tts-pitch">
              Pitch: {ttsPitch.toFixed(1)}
            </label>
            <div className="slider-container">
              <span className="slider-label">0.5</span>
              <input
                id="tts-pitch"
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={ttsPitch}
                onChange={(e) => setTtsPitch(parseFloat(e.target.value))}
                className="range-slider"
              />
              <span className="slider-label">2.0</span>
            </div>
            <p className="setting-hint">Higher = higher voice, Lower = deeper voice</p>
          </div>

          <div className="setting-item">
            <label htmlFor="tts-volume">
              Volume: {Math.round(ttsVolume * 100)}%
            </label>
            <div className="slider-container">
              <span className="slider-label">0%</span>
              <input
                id="tts-volume"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={ttsVolume}
                onChange={(e) => setTtsVolume(parseFloat(e.target.value))}
                className="range-slider"
              />
              <span className="slider-label">100%</span>
            </div>
          </div>
          
          <div className="setting-item">
            <label htmlFor="tts-preview">Preview Text</label>
            <input
              id="tts-preview"
              type="text"
              value={previewText}
              onChange={(e) => setPreviewText(e.target.value)}
              placeholder="Enter text to preview"
              className="text-input"
            />
          </div>
          
          <div className="tts-button-group">
            <button onClick={previewVoice} className="preview-button">
              🔊 Preview Voice
            </button>
            <button onClick={() => setShowAdvancedTTS(true)} className="advanced-button">
              ⚙️ Advanced Settings
            </button>
          </div>
        </div>
      </div>

      {/* Advanced TTS Settings Modal */}
      <TTSAdvancedSettings
        isOpen={showAdvancedTTS}
        onClose={() => setShowAdvancedTTS(false)}
        currentSettings={{
          voice: selectedVoice,
          speed: ttsSpeed,
          pitch: ttsPitch,
          ...advancedTTSSettings
        }}
        onSave={handleAdvancedSave}
        onPreview={previewVoice}
      />

      <div className="settings-section">
        <h3>Python Environment</h3>
        
        {pythonInfo ? (
          <div className="info-box">
            <div className="info-row">
              <span className="info-label">Status:</span>
              <span className={`status ${pythonInfo.available ? 'success' : 'error'}`}>
                {pythonInfo.available ? '✓ Available' : '✗ Not Found'}
              </span>
            </div>
            
            {pythonInfo.version && (
              <div className="info-row">
                <span className="info-label">Version:</span>
                <span className="info-value">{pythonInfo.version}</span>
              </div>
            )}
            
            <div className="info-row">
              <span className="info-label">Transformers:</span>
              <span className={`status ${pythonInfo.transformers_installed ? 'success' : 'error'}`}>
                {pythonInfo.transformers_installed ? '✓ Installed' : '✗ Not Installed'}
              </span>
            </div>
            
            {!pythonInfo.transformers_installed && pythonInfo.available && (
              <div className="install-hint">
                <p>Install transformers with:</p>
                <code>pip install transformers torch</code>
              </div>
            )}
          </div>
        ) : (
          <div className="info-box">
            <p className="info-text">Loading Python information...</p>
          </div>
        )}
      </div>

      <div className="settings-section">
        <h3>API Authentication</h3>
        <div className="info-box">
          <p className="info-text hint" style={{ marginBottom: '12px' }}>
            Configure API keys for downloading models from various sources. Most sources work without authentication for public models.
          </p>
          
          <div className="setting-item">
            <label htmlFor="openai-key">OpenAI API Key</label>
            <input
              id="openai-key"
              type="password"
              value={openaiApiKey}
              onChange={(e) => setOpenaiApiKey(e.target.value)}
              placeholder="sk-..."
              className="input-field"
            />
            <p className="info-text hint" style={{ marginTop: '6px' }}>
              For accessing OpenAI models. Get your key from{' '}
              <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer">
                OpenAI Platform
              </a>
            </p>
          </div>

          <div className="setting-item" style={{ marginTop: '20px' }}>
            <label htmlFor="civitai-key">CivitAI API Key</label>
            <input
              id="civitai-key"
              type="password"
              value={civitaiApiKey}
              onChange={(e) => setCivitaiApiKey(e.target.value)}
              placeholder="Enter your CivitAI API key"
              className="input-field"
            />
            <p className="info-text hint" style={{ marginTop: '6px' }}>
              Required for downloading from CivitAI. Get your key from{' '}
              <a href="https://civitai.com/user/account" target="_blank" rel="noopener noreferrer">
                CivitAI Account Settings
              </a>
            </p>
          </div>

          <div className="setting-item" style={{ marginTop: '20px' }}>
            <label htmlFor="github-token">GitHub Personal Access Token</label>
            <input
              id="github-token"
              type="password"
              value={githubToken}
              onChange={(e) => setGithubToken(e.target.value)}
              placeholder="ghp_..."
              className="input-field"
            />
            <p className="info-text hint" style={{ marginTop: '6px' }}>
              Optional. For private repos and higher rate limits. Generate at{' '}
              <a href="https://github.com/settings/tokens" target="_blank" rel="noopener noreferrer">
                GitHub Tokens
              </a>
            </p>
          </div>

          <div className="setting-item" style={{ marginTop: '20px' }}>
            <label htmlFor="modelscope-key">ModelScope API Key</label>
            <input
              id="modelscope-key"
              type="password"
              value={modelscopeApiKey}
              onChange={(e) => setModelscopeApiKey(e.target.value)}
              placeholder="Enter your ModelScope API key"
              className="input-field"
            />
            <p className="info-text hint" style={{ marginTop: '6px' }}>
              For accessing ModelScope models. Get your key from{' '}
              <a href="https://www.modelscope.cn/" target="_blank" rel="noopener noreferrer">
                ModelScope
              </a>
            </p>
          </div>

          <div className="setting-item" style={{ marginTop: '20px' }}>
            <label htmlFor="koboldai-key">KoboldAI API Key</label>
            <input
              id="koboldai-key"
              type="password"
              value={koboldaiApiKey}
              onChange={(e) => setKoboldaiApiKey(e.target.value)}
              placeholder="Enter your KoboldAI API key"
              className="input-field"
            />
            <p className="info-text hint" style={{ marginTop: '6px' }}>
              For hosted KoboldAI instances or community APIs
            </p>
          </div>

          <div className="setting-item" style={{ marginTop: '20px' }}>
            <label htmlFor="nvidia-api-key">NVIDIA NGC API Key</label>
            <input
              id="nvidia-api-key"
              type="password"
              value={nvidiaApiKey}
              onChange={(e) => setNvidiaApiKey(e.target.value)}
              placeholder="Enter your NGC API key"
              className="input-field"
            />
            <p className="info-text hint" style={{ marginTop: '6px' }}>
              Required for NVIDIA NGC downloads. Get your API key from{' '}
              <a href="https://ngc.nvidia.com/setup/api-key" target="_blank" rel="noopener noreferrer">
                NGC Setup
              </a>
            </p>
          </div>

          <div className="setting-item" style={{ marginTop: '20px' }}>
            <label htmlFor="aws-api-key">AWS Access Key</label>
            <input
              id="aws-api-key"
              type="password"
              value={awsApiKey}
              onChange={(e) => setAwsApiKey(e.target.value)}
              placeholder="Enter your AWS access key (optional)"
              className="input-field"
            />
            <p className="info-text hint" style={{ marginTop: '6px' }}>
              Optional. For AWS SageMaker model access.
            </p>
          </div>

          <div className="setting-item" style={{ marginTop: '20px' }}>
            <label htmlFor="azure-api-key">Azure API Key</label>
            <input
              id="azure-api-key"
              type="password"
              value={azureApiKey}
              onChange={(e) => setAzureApiKey(e.target.value)}
              placeholder="Enter your Azure API key (optional)"
              className="input-field"
            />
            <p className="info-text hint" style={{ marginTop: '6px' }}>
              Optional. For Azure AI model access.
            </p>
          </div>

          <div className="setting-item" style={{ marginTop: '20px' }}>
            <label htmlFor="gcp-api-key">Google Cloud API Key</label>
            <input
              id="gcp-api-key"
              type="password"
              value={gcpApiKey}
              onChange={(e) => setGcpApiKey(e.target.value)}
              placeholder="Enter your GCP API key (optional)"
              className="input-field"
            />
            <p className="info-text hint" style={{ marginTop: '6px' }}>
              Optional. For Google Cloud Vertex AI access.
            </p>
          </div>
        </div>
      </div>

      <div className="settings-section">
        <h3>Search Engine</h3>
        <div className="info-box">
          <p className="info-text hint" style={{ marginBottom: '12px' }}>
            Choose the default search engine for online search in chats
          </p>
          <div className="search-engine-grid">
            {SEARCH_ENGINES.map(engine => (
              <button
                key={engine.id}
                className={`search-engine-option ${searchEngine === engine.id ? 'active' : ''}`}
                onClick={() => setSearchEngine(engine.id)}
              >
                <span className="engine-icon">{engine.icon}</span>
                <span className="engine-name">{engine.name}</span>
                {searchEngine === engine.id && <span className="check-mark">✓</span>}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="settings-section">
        <h3>About</h3>
        <div className="info-box">
          <p className="info-text">
            <strong>MindDrop</strong> - Run AI models locally
          </p>
          <p className="info-text hint">
            Built with Tauri + React
          </p>
          <p className="info-text hint">
            No API keys required - everything runs on your machine
          </p>
        </div>
      </div>

      <div className="settings-section">
        <h3>Memory Management</h3>
        <div className="info-box">
          <p className="info-text hint" style={{ marginBottom: '12px' }}>
            Free up RAM and VRAM by stopping background model processes
          </p>
          <button 
            onClick={async () => {
              try {
                await invoke('cleanup_model_memory');
                alert('Memory cleanup completed. Running model processes have been stopped.');
              } catch (err) {
                console.error('Cleanup failed:', err);
                alert('Memory cleanup failed. Check console for details.');
              }
            }}
            className="preview-button"
            style={{ width: '100%' }}
          >
            🧹 Free Memory (Stop Models)
          </button>
          <p className="info-text hint" style={{ marginTop: '8px', fontSize: '0.85em' }}>
            Note: Models are automatically cleaned when switching chats
          </p>
        </div>
      </div>
    </div>
  );
}

export default Settings;
