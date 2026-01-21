import './SystemStats.css';

function SystemStats({ systemInfo, maxModelCapacity, calculatingMaxModel }) {
  if (!systemInfo) {
    return (
      <div className="system-stats loading">
        <div className="stats-header">
          <h3>System Stats</h3>
        </div>
        <p>Loading...</p>
      </div>
    );
  }

  const formatBytes = (bytes) => {
    const gb = bytes / (1024 ** 3);
    return `${gb.toFixed(1)} GB`;
  };

  return (
    <div className="system-stats">
      <div className="stats-header">
        <h3>💻 System Stats</h3>
      </div>
      
      <div className="stat-group">
        <div className="stat-label">CPU</div>
        <div className="stat-value">{systemInfo.cpu.model}</div>
        <div className="stat-detail">{systemInfo.cpu.cores} cores</div>
        <div className="stat-bar">
          <div 
            className="stat-bar-fill"
            style={{ width: `${Math.min(systemInfo.cpu.usage, 100)}%` }}
          />
        </div>
        <div className="stat-detail">{systemInfo.cpu.usage.toFixed(1)}%</div>
      </div>

      <div className="stat-group">
        <div className="stat-label">RAM</div>
        <div className="stat-value">
          {formatBytes(systemInfo.memory.used)} / {formatBytes(systemInfo.memory.total)}
        </div>
        <div className="stat-bar">
          <div 
            className="stat-bar-fill"
            style={{ width: `${systemInfo.memory.usage_percent}%` }}
          />
        </div>
        <div className="stat-detail">{systemInfo.memory.usage_percent.toFixed(1)}%</div>
      </div>

      {systemInfo.gpu && (
        <div className="stat-group">
          <div className="stat-label">VRAM</div>
          <div className="stat-value">
            {formatBytes(systemInfo.gpu.vram_used)} / {formatBytes(systemInfo.gpu.vram_total)}
          </div>
          <div className="stat-bar">
            <div
              className="stat-bar-fill"
              style={{
                width: `${Math.min((systemInfo.gpu.vram_used / systemInfo.gpu.vram_total) * 100, 100)}%`,
              }}
            />
          </div>
          <div className="stat-detail">{systemInfo.gpu.name}</div>
        </div>
      )}

      <div className="stat-group">
        <div className="stat-label">Max Theoretical Model</div>
        {calculatingMaxModel ? (
          <div className="stat-value calculating">Calculating...</div>
        ) : maxModelCapacity ? (
          <div className="stat-value">{maxModelCapacity.max_params_billions.toFixed(2)}B</div>
        ) : (
          <div className="stat-value">Unknown</div>
        )}
      </div>

      <div className="system-info">
        <div className="info-item">
          <span className="info-label">OS:</span>
          <span className="info-value">{systemInfo.platform.os}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Arch:</span>
          <span className="info-value">{systemInfo.platform.arch}</span>
        </div>
      </div>
    </div>
  );
}

export default SystemStats;
