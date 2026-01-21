import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './Plugins.css';

function Plugins() {
  const [plugins, setPlugins] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadPlugins();
  }, []);

  const loadPlugins = async () => {
    setLoading(true);
    setError(null);
    try {
      const discovered = await invoke('discover_plugins');
      setPlugins(discovered);
    } catch (err) {
      console.error('Failed to load plugins:', err);
      setError(err.toString());
    } finally {
      setLoading(false);
    }
  };

  const togglePlugin = async (pluginId) => {
    try {
      await invoke('toggle_plugin', { pluginId });
      await loadPlugins();
    } catch (err) {
      console.error('Failed to toggle plugin:', err);
      setError(err.toString());
    }
  };

  const executePlugin = async (pluginId, input) => {
    try {
      const result = await invoke('execute_plugin', { pluginId, input });
      console.log('Plugin result:', result);
      return result;
    } catch (err) {
      console.error('Failed to execute plugin:', err);
      setError(err.toString());
    }
  };

  const getPluginTypeIcon = (type) => {
    switch (type) {
      case 'python': return '🐍';
      case 'node': return '📦';
      case 'http': return '🌐';
      case 'executable': return '⚙️';
      default: return '🔌';
    }
  };

  return (
    <div className="plugins-container">
      <div className="plugins-header">
        <h2>Plugins</h2>
        <button onClick={loadPlugins} className="refresh-button">
          🔄 Refresh
        </button>
      </div>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {loading ? (
        <div className="loading">Loading plugins...</div>
      ) : (
        <>
          <div className="plugins-info">
            <p>
              Plugins extend the functionality of this application. They run out-of-process
              and communicate via JSON. Place plugins in the <code>Plugins/</code> folder.
            </p>
            <p>
              <strong>Found {plugins.length} plugin(s)</strong>
            </p>
          </div>

          {plugins.length === 0 ? (
            <div className="no-plugins">
              <p>No plugins found.</p>
              <p>Create a plugin folder in <code>Plugins/</code> with a <code>plugin.json</code> manifest.</p>
              <p>
                <a href="#" onClick={(e) => {
                  e.preventDefault();
                  invoke('open_plugins_folder');
                }}>
                  📁 Open Plugins Folder
                </a>
              </p>
            </div>
          ) : (
            <div className="plugins-list">
              {plugins.map((plugin) => (
                <div
                  key={plugin.id}
                  className={`plugin-card ${plugin.enabled ? 'enabled' : 'disabled'}`}
                >
                  <div className="plugin-header">
                    <div className="plugin-title">
                      <span className="plugin-icon">{getPluginTypeIcon(plugin.type)}</span>
                      <h3>{plugin.name}</h3>
                      <span className={`plugin-status ${plugin.enabled ? 'active' : 'inactive'}`}>
                        {plugin.enabled ? '✓ Enabled' : '○ Disabled'}
                      </span>
                    </div>
                    <label className="toggle-switch">
                      <input
                        type="checkbox"
                        checked={plugin.enabled}
                        onChange={() => togglePlugin(plugin.id)}
                      />
                      <span className="toggle-slider"></span>
                    </label>
                  </div>

                  <div className="plugin-info">
                    <p className="plugin-description">{plugin.description || 'No description'}</p>
                    
                    <div className="plugin-meta">
                      <div className="meta-item">
                        <span className="meta-label">Type:</span>
                        <span className="meta-value">{plugin.type}</span>
                      </div>
                      {plugin.version && (
                        <div className="meta-item">
                          <span className="meta-label">Version:</span>
                          <span className="meta-value">{plugin.version}</span>
                        </div>
                      )}
                      {plugin.author && (
                        <div className="meta-item">
                          <span className="meta-label">Author:</span>
                          <span className="meta-value">{plugin.author}</span>
                        </div>
                      )}
                    </div>

                    {plugin.entry && (
                      <div className="plugin-details">
                        <span className="detail-label">Entry:</span>
                        <code>{plugin.entry}</code>
                      </div>
                    )}

                    {plugin.inputs && plugin.inputs.length > 0 && (
                      <div className="plugin-details">
                        <span className="detail-label">Inputs:</span>
                        <code>{plugin.inputs.join(', ')}</code>
                      </div>
                    )}

                    {plugin.outputs && plugin.outputs.length > 0 && (
                      <div className="plugin-details">
                        <span className="detail-label">Outputs:</span>
                        <code>{plugin.outputs.join(', ')}</code>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default Plugins;
