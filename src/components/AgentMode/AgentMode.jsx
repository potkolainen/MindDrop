import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './AgentMode.css';

function AgentMode() {
  const [runningApps, setRunningApps] = useState([]);
  const [selectedApps, setSelectedApps] = useState({});
  const [permissions, setPermissions] = useState({});
  const [loading, setLoading] = useState(true);
  const [agentEnabled, setAgentEnabled] = useState(false);
  const [agentScope, setAgentScope] = useState('user-apps'); // 'user-apps' or 'whole-computer'

  useEffect(() => {
    loadRunningApps();
    loadSavedPermissions();
  }, []);

  const loadRunningApps = async () => {
    try {
      setLoading(true);
      const apps = await invoke('get_running_applications');
      setRunningApps(apps);
    } catch (error) {
      console.error('Failed to load running applications:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadSavedPermissions = async () => {
    try {
      const saved = await invoke('get_agent_permissions');
      setPermissions(saved);
      setSelectedApps(saved.selectedApps || {});
      setAgentEnabled(saved.enabled || false);
    } catch (error) {
      console.error('Failed to load permissions:', error);
    }
  };

  const toggleAppSelection = (appName) => {
    const newSelected = {
      ...selectedApps,
      [appName]: !selectedApps[appName]
    };
    setSelectedApps(newSelected);
    savePermissions({ ...permissions, selectedApps: newSelected });
  };

  const togglePermission = (appName, permissionType) => {
    const appPerms = permissions[appName] || {};
    const newAppPerms = {
      ...appPerms,
      [permissionType]: !appPerms[permissionType]
    };
    const newPermissions = {
      ...permissions,
      [appName]: newAppPerms
    };
    setPermissions(newPermissions);
    savePermissions(newPermissions);
  };

  const savePermissions = async (perms) => {
    try {
      await invoke('save_agent_permissions', { permissions: perms });
    } catch (error) {
      console.error('Failed to save permissions:', error);
    }
  };

  const toggleAgentMode = async () => {
    const newEnabled = !agentEnabled;
    setAgentEnabled(newEnabled);
    const newPermissions = { ...permissions, enabled: newEnabled };
    setPermissions(newPermissions);
    await savePermissions(newPermissions);
  };

  const permissionTypes = [
    { id: 'read', label: 'Read', description: 'View window content and state' },
    { id: 'write', label: 'Write', description: 'Input text and data' },
    { id: 'execute', label: 'Execute', description: 'Trigger actions and commands' },
    { id: 'navigate', label: 'Navigate', description: 'Switch windows and tabs' }
  ];

  return (
    <div className="agent-mode">
      <div className="header">
        <h2>Agent Mode</h2>
        <p className="subtitle">Control which applications the AI agent can interact with</p>
      </div>

      <div className="agent-toggle-section">
        <div className="toggle-header">
          <div>
            <h3>Agent Mode</h3>
            <p>Enable AI agent to interact with selected applications</p>
          </div>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={agentEnabled}
              onChange={toggleAgentMode}
            />
            <span className="toggle-slider"></span>
          </label>
        </div>
        {agentEnabled && (
          <div className="warning-box">
            <span className="warning-icon">⚠️</span>
            <span>Agent mode is active. The AI can interact with selected applications.</span>
          </div>
        )}
      </div>

      {agentEnabled && (
        <div className="scope-selector-section">
          <h3>Agent Scope</h3>
          <div className="scope-options">
            <label className={`scope-option ${agentScope === 'user-apps' ? 'active' : ''}`}>
              <input
                type="radio"
                name="scope"
                value="user-apps"
                checked={agentScope === 'user-apps'}
                onChange={(e) => setAgentScope(e.target.value)}
              />
              <div className="scope-content">
                <div className="scope-header">
                  <span className="scope-icon">🪟</span>
                  <span className="scope-name">User Applications</span>
                </div>
                <p className="scope-description">Agent can only interact with the applications you select below</p>
              </div>
            </label>
            
            <label className={`scope-option danger ${agentScope === 'whole-computer' ? 'active' : ''}`}>
              <input
                type="radio"
                name="scope"
                value="whole-computer"
                checked={agentScope === 'whole-computer'}
                onChange={(e) => setAgentScope(e.target.value)}
              />
              <div className="scope-content">
                <div className="scope-header">
                  <span className="scope-icon nuclear">☢️</span>
                  <span className="scope-name">Whole Computer</span>
                </div>
                <p className="scope-description danger-text">⚠️ WARNING: Agent has full system access. Can modify files, settings, and execute commands. USE WITH EXTREME CAUTION!</p>
              </div>
            </label>
          </div>
          
          {agentScope === 'whole-computer' && (
            <div className="nuclear-warning">
              <div className="nuclear-header">
                <span className="nuclear-icon">☢️</span>
                <span className="nuclear-title">DANGER ZONE</span>
                <span className="nuclear-icon">☢️</span>
              </div>
              <ul className="nuclear-warnings">
                <li>⚠️ Agent can DELETE or MODIFY any file on your computer</li>
                <li>⚠️ Agent can INSTALL or UNINSTALL software</li>
                <li>⚠️ Agent can ACCESS sensitive data including passwords</li>
                <li>⚠️ Agent can EXECUTE arbitrary system commands</li>
                <li>⚠️ Actions may be IRREVERSIBLE and cause DATA LOSS</li>
              </ul>
              <p className="nuclear-advice">Only use this mode if you fully understand the risks and trust the AI model completely.</p>
            </div>
          )}
        </div>
      )}

      {agentEnabled && agentScope === 'user-apps' && (
        <div className="apps-section">
          <div className="section-header">
            <h3>Currently Open Applications</h3>
            <button onClick={loadRunningApps} className="refresh-btn">
              🔄 Refresh
            </button>
          </div>

        {loading ? (
          <div className="loading">Loading applications...</div>
        ) : runningApps.length === 0 ? (
          <div className="empty-state">No applications detected</div>
        ) : (
          <div className="apps-list">
            {runningApps.map((app) => (
              <div key={app.name} className="app-item">
                <div className="app-header">
                  <div className="app-info">
                    <input
                      type="checkbox"
                      checked={selectedApps[app.name] || false}
                      onChange={() => toggleAppSelection(app.name)}
                      disabled={!agentEnabled}
                    />
                    <div className="app-details">
                      <span className="app-name">{app.name}</span>
                      <span className="app-pid">PID: {app.pid}</span>
                    </div>
                  </div>
                  {app.icon && <span className="app-icon">{app.icon}</span>}
                </div>

                {selectedApps[app.name] && (
                  <div className="app-permissions">
                    <h4>Permissions</h4>
                    <div className="permissions-grid">
                      {permissionTypes.map((perm) => (
                        <label key={perm.id} className="permission-item">
                          <input
                            type="checkbox"
                            checked={permissions[app.name]?.[perm.id] || false}
                            onChange={() => togglePermission(app.name, perm.id)}
                            disabled={!agentEnabled}
                          />
                          <div>
                            <div className="perm-label">{perm.label}</div>
                            <div className="perm-description">{perm.description}</div>
                          </div>
                        </label>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        </div>
      )}

      {agentEnabled && agentScope === 'user-apps' && (
        <div className="safety-section">
          <h3>Safety Limits (User Applications Mode)</h3>
          <p className="safety-description">
            In User Applications mode, the agent is restricted and prevented from:
          </p>
          <ul className="safety-list">
            <li>Deleting files or data without confirmation</li>
            <li>Accessing system-critical applications</li>
            <li>Executing shell commands directly</li>
            <li>Accessing sensitive information (passwords, credentials)</li>
            <li>Performing actions outside selected applications</li>
          </ul>
        </div>
      )}
    </div>
  );
}

export default AgentMode;
