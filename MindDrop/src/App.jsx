import { useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './App.css';
import AgentMode from './components/AgentMode/AgentMode';
import Chat from './components/Chat/Chat';
import Diffusers from './components/Diffusers/Diffusers';
import Library from './components/Library/Library';
import Models from './components/Models/Models';
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
                  return (
                    <div key={category.id} className="conversation-category">
                      <div className="category-header">
                        <span className="category-title">
                          {category.icon} {category.name}
                        </span>
                        <button
                          className="category-new-btn"
                          onClick={() => {
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
                      {categoryChats.map((chat) => (
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
        {activeTab === 'settings' && <Settings searchEngine={searchEngine} setSearchEngine={setSearchEngine} />}
      </div>
    </div>
  );
}

export default App;
