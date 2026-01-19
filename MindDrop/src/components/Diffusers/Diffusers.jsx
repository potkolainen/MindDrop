import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './Diffusers.css';

const DIFFUSER_BACKENDS = [
  {
    id: 'diffusers',
    name: 'Diffusers (Python Library)',
    description: 'Official Hugging Face diffusers library - lightweight, fast, supports all major SD models',
    repo: 'https://github.com/huggingface/diffusers',
    installCmd: 'pip install diffusers transformers accelerate torch',
    features: ['Stable Diffusion', 'SDXL', 'ControlNet', 'LoRA', 'Img2Img', 'Inpainting'],
    icon: '🤗'
  },
  {
    id: 'automatic1111',
    name: 'Stable Diffusion WebUI (AUTOMATIC1111)',
    description: 'Most popular SD interface with extensive features and community extensions',
    repo: 'https://github.com/AUTOMATIC1111/stable-diffusion-webui',
    installCmd: 'git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git',
    features: ['Web UI', 'Extensions', 'Custom Scripts', 'API Server', 'Batch Processing'],
    icon: '🖼️'
  },
  {
    id: 'comfyui',
    name: 'ComfyUI',
    description: 'Node-based interface for advanced workflows and custom pipelines',
    repo: 'https://github.com/comfyanonymous/ComfyUI',
    installCmd: 'git clone https://github.com/comfyanonymous/ComfyUI.git',
    features: ['Node Workflow', 'Custom Nodes', 'Advanced Control', 'Efficient Memory', 'Animation'],
    icon: '🎨'
  },
  {
    id: 'invokeai',
    name: 'InvokeAI',
    description: 'Professional-grade SD with unified canvas and powerful workflows',
    repo: 'https://github.com/invoke-ai/InvokeAI',
    installCmd: 'pip install invokeai --use-pep517',
    features: ['Unified Canvas', 'Outpainting', 'Professional UI', 'Model Manager', 'Workflows'],
    icon: '⚡'
  }
];

function Diffusers() {
  const [installedBackends, setInstalledBackends] = useState([]);
  const [installing, setInstalling] = useState(null);
  const [uninstalling, setUninstalling] = useState(null);
  const [installLogs, setInstallLogs] = useState({});

  useEffect(() => {
    checkInstalled();
  }, []);

  const checkInstalled = async () => {
    // Check which backends are already installed
    try {
      const result = await invoke('check_diffuser_backends');
      setInstalledBackends(result);
    } catch (error) {
      console.error('Failed to check backends:', error);
    }
  };

  const installBackend = async (backend) => {
    if (installedBackends.includes(backend.id)) {
      setInstallLogs(prev => ({ 
        ...prev, 
        [backend.id]: '⚠️ Already installed. Uninstall first if you want to reinstall.\n' 
      }));
      return;
    }

    setInstalling(backend.id);
    setInstallLogs(prev => ({ ...prev, [backend.id]: 'Starting installation...\n' }));

    try {
      await invoke('install_diffuser_backend', {
        backendId: backend.id,
        installCmd: backend.installCmd,
        repo: backend.repo
      });
      
      setInstallLogs(prev => ({ 
        ...prev, 
        [backend.id]: prev[backend.id] + '\n✅ Installation complete!' 
      }));
      
      await checkInstalled();
    } catch (error) {
      setInstallLogs(prev => ({ 
        ...prev, 
        [backend.id]: prev[backend.id] + `\n❌ Error: ${error}` 
      }));
    } finally {
      setInstalling(null);
    }
  };

  const uninstallBackend = async (backend) => {
    if (!installedBackends.includes(backend.id)) {
      return;
    }

    if (!confirm(`Are you sure you want to uninstall ${backend.name}?`)) {
      return;
    }

    setUninstalling(backend.id);
    setInstallLogs(prev => ({ ...prev, [backend.id]: 'Uninstalling...\n' }));

    try {
      const result = await invoke('uninstall_diffuser_backend', {
        backendId: backend.id
      });
      
      setInstallLogs(prev => ({ 
        ...prev, 
        [backend.id]: prev[backend.id] + `\n✅ ${result}` 
      }));
      
      await checkInstalled();
    } catch (error) {
      setInstallLogs(prev => ({ 
        ...prev, 
        [backend.id]: prev[backend.id] + `\n❌ Error: ${error}` 
      }));
    } finally {
      setUninstalling(null);
    }
  };

  return (
    <div className="diffusers-container">
      <div className="diffusers-header">
        <h2>Image Generation Backends</h2>
        <p>Install and manage different Stable Diffusion interfaces</p>
      </div>

      <div className="backends-grid">
        {DIFFUSER_BACKENDS.map(backend => (
          <div key={backend.id} className="backend-card">
            <div className="backend-header">
              <span className="backend-icon">{backend.icon}</span>
              <div>
                <h3>{backend.name}</h3>
                <a href={backend.repo} target="_blank" rel="noopener noreferrer" className="repo-link">
                  {backend.repo}
                </a>
              </div>
            </div>

            <p className="backend-description">{backend.description}</p>

            <div className="backend-features">
              {backend.features.map(feature => (
                <span key={feature} className="feature-tag">{feature}</span>
              ))}
            </div>

            <div className="backend-install">
              <code className="install-cmd">{backend.installCmd}</code>
              <div className="backend-actions">
                <button
                  className={`install-btn ${installedBackends.includes(backend.id) ? 'installed' : ''}`}
                  onClick={() => installBackend(backend)}
                  disabled={installing === backend.id || installedBackends.includes(backend.id)}
                >
                  {installing === backend.id ? '⏳ Installing...' : 
                   installedBackends.includes(backend.id) ? '✅ Installed' : 
                   '📥 Install'}
                </button>
                {installedBackends.includes(backend.id) && (
                  <button
                    className="uninstall-btn"
                    onClick={() => uninstallBackend(backend)}
                    disabled={uninstalling === backend.id}
                  >
                    {uninstalling === backend.id ? '⏳ Uninstalling...' : '🗑️ Uninstall'}
                  </button>
                )}
              </div>
            </div>

            {installLogs[backend.id] && (
              <pre className="install-log">{installLogs[backend.id]}</pre>
            )}
          </div>
        ))}
      </div>

      <div className="diffusers-info">
        <h3>ℹ️ Quick Start</h3>
        <ol>
          <li><strong>Diffusers</strong> - Best for integration with this app, install first</li>
          <li><strong>AUTOMATIC1111</strong> - Best for casual users, feature-rich UI</li>
          <li><strong>ComfyUI</strong> - Best for advanced users, node-based workflows</li>
          <li><strong>InvokeAI</strong> - Best for professionals, unified canvas</li>
        </ol>
        <p>After installation, download models from the <strong>Models</strong> tab and use them in the <strong>Image</strong> chat category.</p>
      </div>
    </div>
  );
}

export default Diffusers;
