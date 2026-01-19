import { useState, useEffect } from 'react';
import './TTSAdvancedSettings.css';

function TTSAdvancedSettings({ isOpen, onClose, currentSettings, onSave, onPreview }) {
  const [settings, setSettings] = useState({
    // Core TTS
    voice: 'female_neutral',
    speed: 1.0,
    pitch: 1.0,
    prosody: 0.5,
    stability: 0.7,
    
    // Timing
    pauseMultiplier: 1.0,
    breathInsertion: 'light',
    
    // Post-processing
    warmth: 0.0,
    presence: 0.0,
    air: 0.0,
    reverbSize: 0.0,
    reverbDecay: 0.5,
    breathVolume: 0.3,
    deEsser: 0.5,
    normalize: true,
    limiter: true,
    
    // Preset
    presetName: ''
  });

  const [presets, setPresets] = useState([]);
  const [selectedPreset, setSelectedPreset] = useState('');

  useEffect(() => {
    if (currentSettings) {
      setSettings({ ...settings, ...currentSettings });
    }
    
    // Load presets from localStorage
    const savedPresets = localStorage.getItem('tts-presets');
    if (savedPresets) {
      setPresets(JSON.parse(savedPresets));
    }
  }, [currentSettings]);

  const voiceProfiles = [
    { id: 'male_neutral', name: 'Male - Neutral', gender: 'Male' },
    { id: 'male_deep', name: 'Male - Deep', gender: 'Male' },
    { id: 'male_soft', name: 'Male - Soft', gender: 'Male' },
    { id: 'female_neutral', name: 'Female - Neutral', gender: 'Female' },
    { id: 'female_warm', name: 'Female - Warm', gender: 'Female' },
    { id: 'female_bright', name: 'Female - Bright', gender: 'Female' },
    { id: 'androgynous', name: 'Androgynous', gender: 'Neutral' },
    { id: 'narrator', name: 'Narrator', gender: 'Female' },
  ];

  const defaultPresets = [
    {
      name: 'Natural Conversation',
      settings: { speed: 1.0, prosody: 0.6, stability: 0.6, warmth: 0.2, breathVolume: 0.4 }
    },
    {
      name: 'Podcast Narrator',
      settings: { speed: 0.95, prosody: 0.7, stability: 0.8, warmth: 0.3, presence: 0.5 }
    },
    {
      name: 'Warm Audiobook',
      settings: { speed: 0.9, prosody: 0.8, stability: 0.9, warmth: 0.5, reverbSize: 0.2 }
    },
    {
      name: 'Cold AI Voice',
      settings: { speed: 1.1, prosody: 0.2, stability: 0.9, warmth: -0.3, presence: 0.3 }
    },
    {
      name: 'Fast Utility',
      settings: { speed: 1.3, prosody: 0.3, stability: 0.8, warmth: 0.0 }
    }
  ];

  const handleSave = () => {
    onSave(settings);
    onClose();
  };

  const saveAsPreset = () => {
    if (!settings.presetName) {
      alert('Please enter a preset name');
      return;
    }

    const newPreset = {
      name: settings.presetName,
      settings: { ...settings }
    };

    const updatedPresets = [...presets, newPreset];
    setPresets(updatedPresets);
    localStorage.setItem('tts-presets', JSON.stringify(updatedPresets));
    setSettings({ ...settings, presetName: '' });
  };

  const loadPreset = (presetName) => {
    const preset = [...defaultPresets, ...presets].find(p => p.name === presetName);
    if (preset) {
      setSettings({ ...settings, ...preset.settings });
      setSelectedPreset(presetName);
    }
  };

  const deletePreset = (presetName) => {
    const updatedPresets = presets.filter(p => p.name !== presetName);
    setPresets(updatedPresets);
    localStorage.setItem('tts-presets', JSON.stringify(updatedPresets));
  };

  if (!isOpen) return null;

  return (
    <div className="tts-modal-overlay" onClick={onClose}>
      <div className="tts-modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="tts-modal-header">
          <h2>🎙️ Advanced TTS Settings</h2>
          <button className="close-button" onClick={onClose}>✕</button>
        </div>

        <div className="tts-modal-body">
          {/* Presets Section */}
          <section className="tts-section">
            <h3>📋 Presets</h3>
            <div className="preset-controls">
              <select 
                value={selectedPreset}
                onChange={(e) => loadPreset(e.target.value)}
                className="preset-select"
              >
                <option value="">-- Select Preset --</option>
                <optgroup label="Default Presets">
                  {defaultPresets.map(p => (
                    <option key={p.name} value={p.name}>{p.name}</option>
                  ))}
                </optgroup>
                {presets.length > 0 && (
                  <optgroup label="Custom Presets">
                    {presets.map(p => (
                      <option key={p.name} value={p.name}>{p.name}</option>
                    ))}
                  </optgroup>
                )}
              </select>

              <div className="save-preset">
                <input
                  type="text"
                  placeholder="Preset name..."
                  value={settings.presetName}
                  onChange={(e) => setSettings({ ...settings, presetName: e.target.value })}
                  className="preset-name-input"
                />
                <button onClick={saveAsPreset} className="save-preset-btn">💾 Save</button>
              </div>
            </div>

            {selectedPreset && presets.find(p => p.name === selectedPreset) && (
              <button 
                onClick={() => deletePreset(selectedPreset)}
                className="delete-preset-btn"
              >
                🗑️ Delete Preset
              </button>
            )}
          </section>

          {/* Voice Profile */}
          <section className="tts-section">
            <h3>🎙️ Voice Profile</h3>
            <select
              value={settings.voice}
              onChange={(e) => setSettings({ ...settings, voice: e.target.value })}
              className="voice-select"
            >
              {voiceProfiles.map(v => (
                <option key={v.id} value={v.id}>
                  {v.name} {v.gender === 'Male' ? '♂️' : v.gender === 'Female' ? '♀️' : '⚧️'}
                </option>
              ))}
            </select>
          </section>

          {/* Core Settings */}
          <section className="tts-section">
            <h3>🎛️ Core Settings</h3>
            
            <div className="setting-row">
              <label>Speed: {settings.speed.toFixed(2)}x</label>
              <input
                type="range"
                min="0.7"
                max="1.3"
                step="0.05"
                value={settings.speed}
                onChange={(e) => setSettings({ ...settings, speed: parseFloat(e.target.value) })}
              />
              <span className="range-labels">
                <span>0.7x</span>
                <span>1.3x</span>
              </span>
            </div>

            <div className="setting-row">
              <label>Pitch: {settings.pitch.toFixed(2)}</label>
              <input
                type="range"
                min="0.85"
                max="1.15"
                step="0.05"
                value={settings.pitch}
                onChange={(e) => setSettings({ ...settings, pitch: parseFloat(e.target.value) })}
              />
              <span className="range-labels">
                <span>0.85</span>
                <span>1.15</span>
              </span>
            </div>

            <div className="setting-row">
              <label>Prosody (Expressiveness): {settings.prosody.toFixed(2)}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.prosody}
                onChange={(e) => setSettings({ ...settings, prosody: parseFloat(e.target.value) })}
              />
              <span className="range-labels">
                <span>Flat</span>
                <span>Expressive</span>
              </span>
            </div>

            <div className="setting-row">
              <label>Stability: {settings.stability.toFixed(2)}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.stability}
                onChange={(e) => setSettings({ ...settings, stability: parseFloat(e.target.value) })}
              />
              <span className="range-labels">
                <span>Variable</span>
                <span>Stable</span>
              </span>
            </div>
          </section>

          {/* Timing */}
          <section className="tts-section">
            <h3>⏱️ Timing</h3>
            
            <div className="setting-row">
              <label>Pause Multiplier: {settings.pauseMultiplier.toFixed(2)}</label>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={settings.pauseMultiplier}
                onChange={(e) => setSettings({ ...settings, pauseMultiplier: parseFloat(e.target.value) })}
              />
              <span className="range-labels">
                <span>0.5x</span>
                <span>2.0x</span>
              </span>
            </div>

            <div className="setting-row">
              <label>Breath Insertion</label>
              <select
                value={settings.breathInsertion}
                onChange={(e) => setSettings({ ...settings, breathInsertion: e.target.value })}
                className="breath-select"
              >
                <option value="off">Off</option>
                <option value="light">Light</option>
                <option value="normal">Normal</option>
              </select>
            </div>
          </section>

          {/* Post-Processing */}
          <section className="tts-section">
            <h3>🎚️ Post-Processing (Studio)</h3>
            
            <div className="setting-row">
              <label>Warmth: {settings.warmth.toFixed(2)}</label>
              <input
                type="range"
                min="-1"
                max="1"
                step="0.1"
                value={settings.warmth}
                onChange={(e) => setSettings({ ...settings, warmth: parseFloat(e.target.value) })}
              />
              <span className="range-labels">
                <span>Cold</span>
                <span>Warm</span>
              </span>
            </div>

            <div className="setting-row">
              <label>Presence (2-4kHz): {settings.presence.toFixed(2)}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.presence}
                onChange={(e) => setSettings({ ...settings, presence: parseFloat(e.target.value) })}
              />
            </div>

            <div className="setting-row">
              <label>Air (8-12kHz): {settings.air.toFixed(2)}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.air}
                onChange={(e) => setSettings({ ...settings, air: parseFloat(e.target.value) })}
              />
            </div>

            <div className="setting-row">
              <label>Reverb Room Size: {settings.reverbSize.toFixed(2)}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.reverbSize}
                onChange={(e) => setSettings({ ...settings, reverbSize: parseFloat(e.target.value) })}
              />
              <span className="range-labels">
                <span>Dry</span>
                <span>Large Room</span>
              </span>
            </div>

            <div className="setting-row">
              <label>Breath Volume: {settings.breathVolume.toFixed(2)}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.breathVolume}
                onChange={(e) => setSettings({ ...settings, breathVolume: parseFloat(e.target.value) })}
              />
            </div>

            <div className="setting-row">
              <label>De-Esser: {settings.deEsser.toFixed(2)}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.deEsser}
                onChange={(e) => setSettings({ ...settings, deEsser: parseFloat(e.target.value) })}
              />
            </div>

            <div className="setting-row checkbox-row">
              <label>
                <input
                  type="checkbox"
                  checked={settings.normalize}
                  onChange={(e) => setSettings({ ...settings, normalize: e.target.checked })}
                />
                LUFS Normalization
              </label>
            </div>

            <div className="setting-row checkbox-row">
              <label>
                <input
                  type="checkbox"
                  checked={settings.limiter}
                  onChange={(e) => setSettings({ ...settings, limiter: e.target.checked })}
                />
                Limiter
              </label>
            </div>
          </section>
        </div>

        <div className="tts-modal-footer">
          <button onClick={onClose} className="cancel-btn">Cancel</button>
          <button
            onClick={() => {
              if (onPreview) {
                onSave(settings);
                onPreview();
              }
            }}
            className="secondary-btn"
          >
            Preview
          </button>
          <button onClick={handleSave} className="save-btn">Save & Apply</button>
        </div>
      </div>
    </div>
  );
}

export default TTSAdvancedSettings;
