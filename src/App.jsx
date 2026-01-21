import { useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { getCurrentWindow } from '@tauri-apps/api/window';
import './App.css';
import AgentMode from './components/AgentMode/AgentMode';
import Chat from './components/Chat/Chat';
import Diffusers from './components/Diffusers/Diffusers';
import Library from './components/Library/Library';
import Models from './components/Models/Models';
import Plugins from './components/Plugins/Plugins';
import Settings from './components/Settings/Settings';
import SystemStats from './components/SystemStats/SystemStats';

function App() {
  const [activeTab, setActiveTab] = useState('models');
  const [sidebarMode, setSidebarMode] = useState('workspace');
  const [sidebarVisible, setSidebarVisible] = useState(true);
  const sidebarRef = useRef(null);
  const [systemInfo, setSystemInfo] = useState(null);
  const [maxModelCapacity, setMaxModelCapacity] = useState(null);
  const [calculatingMaxModel, setCalculatingMaxModel] = useState(true);
  const [selectedModel, setSelectedModel] = useState(null);
  const [collapsedCategories, setCollapsedCategories] = useState(() => {
    const saved = localStorage.getItem('collapsed-categories');
    return saved ? JSON.parse(saved) : [];
  });

  // Window state management
  useEffect(() => {
    const appWindow = getCurrentWindow();
    
    // Restore window state on mount
    const restoreWindowState = async () => {
      try {
        const savedState = localStorage.getItem('window-state');
        if (savedState) {
          const state = JSON.parse(savedState);
          
          // Restore maximized/fullscreen state first
          if (state.isMaximized) {
            await appWindow.maximize();
          } else if (state.isFullscreen) {
            await appWindow.setFullscreen(true);
          } else if (state.position && state.size) {
            // Restore position and size
            await appWindow.setPosition(state.position);
            await appWindow.setSize(state.size);
          }
        }
      } catch (error) {
        console.error('Failed to restore window state:', error);
      }
    };

    restoreWindowState();

    // Save window state on changes
    let saveTimeout;
    const saveWindowState = async () => {
      try {
        const isMaximized = await appWindow.isMaximized();
        const isFullscreen = await appWindow.isFullscreen();
        
        const state = {
          isMaximized,
          isFullscreen,
        };

        // Only save position/size if not maximized or fullscreen
        if (!isMaximized && !isFullscreen) {
          const position = await appWindow.outerPosition();
          const size = await appWindow.outerSize();
          state.position = { x: position.x, y: position.y };
          state.size = { width: size.width, height: size.height };
        }

        localStorage.setItem('window-state', JSON.stringify(state));
      } catch (error) {
        console.error('Failed to save window state:', error);
      }
    };

    // Debounced save function
    const debouncedSave = () => {
      clearTimeout(saveTimeout);
      saveTimeout = setTimeout(saveWindowState, 500);
    };

    // Listen for window events
    const unlistenResize = appWindow.onResized(debouncedSave);
    const unlistenMove = appWindow.onMoved(debouncedSave);

    return () => {
      clearTimeout(saveTimeout);
      unlistenResize.then(fn => fn());
      unlistenMove.then(fn => fn());
    };
  }, []);

  const [downloadQueue, setDownloadQueue] = useState(() => {
    try {
      const saved = localStorage.getItem('download-queue');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [currentDownload, setCurrentDownload] = useState(null);
  const [downloadProgress, setDownloadProgress] = useState({});

  const [chats, setChats] = useState(() => {
    const saved = localStorage.getItem('ai-workspace-chats');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        // Add type field if missing (for backward compatibility)
        return parsed.map(chat => ({
          ...chat,
          type: chat.type || 'text'
        }));
      } catch (e) {
        console.error('Failed to parse saved chats:', e);
      }
    }
    return [
      {
        id: 1,
        name: 'Chat 1',
        type: 'text',
        messages: [
          {
            role: 'assistant',
            content:
              'Hello! 👋 Select a model to get started.\n\nGo to Library tab to download models first!',
          },
        ],
        model: null,
      },
    ];
  });
  const [activeTabId, setActiveTabId] = useState(() => {
    const saved = localStorage.getItem('ai-workspace-active-chat');
    return saved ? parseInt(saved, 10) : 1;
  });
  const [nextChatId, setNextChatId] = useState(() => {
    const saved = localStorage.getItem('ai-workspace-next-chat-id');
    return saved ? parseInt(saved, 10) : 2;
  });
  const [searchEngine, setSearchEngine] = useState(() => {
    const saved = localStorage.getItem('ai-workspace-search-engine');
    return saved || 'duckduckgo';
  });

  useEffect(() => {
    localStorage.setItem('ai-workspace-chats', JSON.stringify(chats));
  }, [chats]);

  useEffect(() => {
    localStorage.setItem('ai-workspace-active-chat', activeTabId.toString());
    // Cleanup models when switching chats to free VRAM
    invoke('cleanup_model_memory').catch(err => console.warn('Memory cleanup failed:', err));
  }, [activeTabId]);

  useEffect(() => {
    localStorage.setItem('ai-workspace-next-chat-id', nextChatId.toString());
  }, [nextChatId]);

  useEffect(() => {
    localStorage.setItem('ai-workspace-search-engine', searchEngine);
  }, [searchEngine]);

  useEffect(() => {
    localStorage.setItem('download-queue', JSON.stringify(downloadQueue));
  }, [downloadQueue]);

  useEffect(() => {
    localStorage.setItem('collapsed-categories', JSON.stringify(collapsedCategories));
  }, [collapsedCategories]);

  useEffect(() => {
    loadSystemInfo();
    const interval = setInterval(loadSystemInfo, 1000);

    setTimeout(async () => {
      try {
        const capacity = await invoke('calculate_max_model_capacity', {
          quantization: '4bit',
        });
        setMaxModelCapacity(capacity);
        setCalculatingMaxModel(false);
      } catch (error) {
        console.error('Failed to calculate max model capacity:', error);
        setCalculatingMaxModel(false);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const loadSystemInfo = async () => {
    try {
      const info = await invoke('get_system_info');
      setSystemInfo(info);
    } catch (error) {
      console.error('Failed to load system info:', error);
    }
  };

  const toggleCategoryCollapse = (categoryId) => {
    setCollapsedCategories(prev => 
      prev.includes(categoryId)
        ? prev.filter(id => id !== categoryId)
        : [...prev, categoryId]
    );
  };

  return (
    <div className="app">
      {!sidebarVisible && (
        <div className="sidebar-icon-trigger" onClick={() => setSidebarVisible(true)}>
          {sidebarMode === 'workspace' ? '🤖' : '💬'}
        </div>
      )}

      <div ref={sidebarRef} className={`sidebar ${sidebarVisible ? 'visible' : 'hidden'}`}>
        <div className="sidebar-content">
          <div className="sidebar-mode-toggle">
            <button
              className={`mode-toggle-btn ${sidebarMode === 'workspace' ? 'active' : 'inactive'}`}
              onClick={() => setSidebarMode('workspace')}
            >
              {sidebarMode === 'workspace' ? '🤖 MindDrop' : '🤖 →'}
            </button>
            <button
              className={`mode-toggle-btn ${sidebarMode === 'chat' ? 'active' : 'inactive'}`}
              onClick={() => setSidebarMode('chat')}
            >
              {sidebarMode === 'chat' ? '💬 Chat' : '← 💬'}
            </button>
            <button className="sidebar-hide-btn" onClick={() => setSidebarVisible(false)} title="Hide sidebar">
              ‹
            </button>
          </div>

          {sidebarMode === 'workspace' && (
            <>
              <nav className="nav">
                <button
                  className={`nav-item ${activeTab === 'library' ? 'active' : ''}`}
                  onClick={() => setActiveTab('library')}
                >
                  📚 Library
                </button>
                <button
                  className={`nav-item ${activeTab === 'models' ? 'active' : ''}`}
                  onClick={() => setActiveTab('models')}
                >
                  🤖 Models
                </button>
                <button
                  className={`nav-item ${activeTab === 'diffusers' ? 'active' : ''}`}
                  onClick={() => setActiveTab('diffusers')}
                >
                  🎨 Diffusers
                </button>
                <button
                  className={`nav-item ${activeTab === 'agent' ? 'active' : ''}`}
                  onClick={() => setActiveTab('agent')}
                >
                  🤝 Agent Mode
                </button>
                <button
                  className={`nav-item ${activeTab === 'plugins' ? 'active' : ''}`}
                  onClick={() => setActiveTab('plugins')}
                >
                  🔌 Plugins
                </button>
                <button
                  className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`}
                  onClick={() => setActiveTab('settings')}
                >
                  ⚙️ Settings
                </button>
              </nav>

              <SystemStats
                systemInfo={systemInfo}
                maxModelCapacity={maxModelCapacity}
                calculatingMaxModel={calculatingMaxModel}
              />
            </>
          )}

          {sidebarMode === 'chat' && (
            <div className="chat-conversations-list">
              <div className="conversations-header">
                <h3>Conversations</h3>
              </div>
              <div className="conversations-scroll">
                {[
                  { id: 'text', name: 'Text', icon: '💬' },
                  { id: 'image', name: 'Image', icon: '🎨' },
                  { id: 'video', name: 'Video', icon: '🎬' },
                  { id: 'audio', name: 'Audio', icon: '🎵' },
                  { id: '3d', name: '3D', icon: '🎲' },
                  { id: 'code', name: 'Code', icon: '💻' },
                  { id: 'multimodal', name: 'Multimodal', icon: '🔮' },
                ].map((category) => {
                  const categoryChats = chats.filter((c) => c.type === category.id);
                  const isCollapsed = collapsedCategories.includes(category.id);
                  const hasChats = categoryChats.length > 0;
                  
                  return (
                    <div key={category.id} className="conversation-category">
                      <div 
                        className="category-header"
                        onClick={() => toggleCategoryCollapse(category.id)}
                        style={{ cursor: 'pointer' }}
                      >
                        <span className="category-title">
                          {category.icon} {category.name}
                          {hasChats && (
                            <span className="category-arrow" style={{ marginLeft: '6px' }}>
                              {isCollapsed ? '▶' : '▼'}
                            </span>
                          )}
                          {hasChats && (
                            <span className="category-count" style={{ marginLeft: '6px', fontSize: '11px', opacity: 0.6 }}>
                              ({categoryChats.length})
                            </span>
                          )}
                        </span>
                        <button
                          className="category-new-btn"
                          onClick={(e) => {
                            e.stopPropagation(); // Prevent category collapse toggle
                            const existingNumbers = chats
                              .filter((c) => c.type === category.id)
                              .map((c) => {
                                const match = c.name.match(/^Chat (\d+)$/);
                                return match ? parseInt(match[1], 10) : 0;
                              })
                              .filter((n) => n > 0);

                            const nextNumber = existingNumbers.length > 0 ? Math.max(...existingNumbers) + 1 : 1;

                            const newChat = {
                              id: nextChatId,
                              name: `Chat ${nextNumber}`,
                              type: category.id,
                              messages: [{ role: 'assistant', content: 'Hello! 👋 Select a model to get started!' }],
                              model: null,
                            };
                            setChats([...chats, newChat]);
                            setActiveTabId(nextChatId);
                            setNextChatId(nextChatId + 1);
                            setActiveTab('chat');
                          }}
                          title={`New ${category.name} Chat`}
                        >
                          +
                        </button>
                      </div>
                      {!isCollapsed && categoryChats.map((chat) => (
                        <div
                          key={chat.id}
                          className={`conversation-item ${chat.id === activeTabId ? 'active' : ''}`}
                          onClick={() => {
                            setActiveTabId(chat.id);
                            setActiveTab('chat');
                          }}
                        >
                          <div className="conversation-name">{chat.name}</div>
                          <div className="conversation-actions">
                            <button
                              className="conversation-rename-btn"
                              onClick={(e) => {
                                e.stopPropagation();
                                const newName = prompt('Rename conversation:', chat.name);
                                if (newName && newName.trim()) {
                                  setChats((prevChats) =>
                                    prevChats.map((c) => (c.id === chat.id ? { ...c, name: newName.trim() } : c)),
                                  );
                                }
                              }}
                              title="Rename"
                            >
                              ✏️
                            </button>
                            {chats.length > 1 && (
                              <button
                                className="conversation-delete-btn"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  if (confirm(`Delete "${chat.name}"?`)) {
                                    const newChats = chats.filter((c) => c.id !== chat.id);
                                    setChats(newChats);
                                    if (activeTabId === chat.id) {
                                      setActiveTabId(newChats[0].id);
                                    }
                                  }
                                }}
                                title="Delete"
                              >
                                🗑️
                              </button>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>
              
              <SystemStats
                systemInfo={systemInfo}
                maxModelCapacity={maxModelCapacity}
                calculatingMaxModel={calculatingMaxModel}
              />
            </div>
          )}
        </div>
      </div>

      <div className={`main-content ${!sidebarVisible ? 'sidebar-hidden' : ''}`}>
        {activeTab === 'chat' && (
          <Chat
            chats={chats}
            setChats={setChats}
            activeTabId={activeTabId}
            setActiveTabId={setActiveTabId}
            nextChatId={nextChatId}
            setNextChatId={setNextChatId}
            searchEngine={searchEngine}
            setSearchEngine={setSearchEngine}
          />
        )}
        {activeTab === 'library' && (
          <Library
            systemInfo={systemInfo}
            downloadQueue={downloadQueue}
            setDownloadQueue={setDownloadQueue}
            currentDownload={currentDownload}
            setCurrentDownload={setCurrentDownload}
            downloadProgress={downloadProgress}
            setDownloadProgress={setDownloadProgress}
          />
        )}
        {activeTab === 'models' && (
          <Models
            onModelSelect={setSelectedModel}
            selectedModel={selectedModel}
            downloadQueue={downloadQueue}
            downloadProgress={downloadProgress}
          />
        )}
        {activeTab === 'agent' && <AgentMode />}
        {activeTab === 'diffusers' && <Diffusers />}
        {activeTab === 'plugins' && <Plugins />}
        {activeTab === 'settings' && <Settings searchEngine={searchEngine} setSearchEngine={setSearchEngine} />}
      </div>
    </div>
  );
}

export default App;
