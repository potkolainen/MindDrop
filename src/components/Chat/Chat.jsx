import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { convertFileSrc } from '@tauri-apps/api/core';
import ReactMarkdown from 'react-markdown';
import './Chat.css';

// Custom code block component with language label and copy button
const CodeBlock = ({ node, inline, className, children, ...props }) => {
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';
  const code = String(children).replace(/\n$/, '');

  const copyCode = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (inline) {
    return <code className={className} {...props}>{children}</code>;
  }

  return (
    <div className="code-block-container">
      <div className="code-block-header">
        <span className="code-language">{language || 'plaintext'}</span>
        <button className="code-copy-btn" onClick={copyCode}>
          {copied ? '✓ Copied' : '📋 Copy'}
        </button>
      </div>
      <pre className={className}>
        <code {...props}>{children}</code>
      </pre>
    </div>
  );
};

const SEARCH_ENGINES = [
  { id: 'multi-engine', name: 'Multi-Engine', icon: '🔀', url: null },
  { id: 'duckduckgo', name: 'DuckDuckGo', icon: '🦆', url: 'https://duckduckgo.com/?q=' },
  { id: 'brave', name: 'Brave', icon: '🦁', url: 'https://search.brave.com/search?q=' },
  { id: 'bing', name: 'Bing', icon: '🅱️', url: 'https://www.bing.com/search?q=' },
  { id: 'qwant', name: 'Qwant', icon: '🇫🇷', url: 'https://www.qwant.com/?q=' },
  { id: 'google-scholar', name: 'Scholar', icon: '🎓', url: 'https://scholar.google.com/scholar?q=' },
  { id: 'stackoverflow', name: 'StackOverflow', icon: '📚', url: 'https://stackoverflow.com/search?q=' },
  { id: 'ecosia', name: 'Ecosia', icon: '🌱', url: 'https://www.ecosia.org/search?q=' },
  { id: 'github', name: 'GitHub', icon: '💻', url: 'https://github.com/search?q=' },
  { id: 'devdocs', name: 'DevDocs', icon: '📖', url: 'https://devdocs.io/#q=' },
  { id: 'arxiv', name: 'arXiv', icon: '📄', url: 'https://arxiv.org/search/?query=' },
  { id: 'startpage', name: 'Startpage', icon: '🔐', url: 'https://www.startpage.com/do/search?q=' },
  { id: 'metager', name: 'MetaGer', icon: '🔎', url: 'https://metager.org/meta/meta.ger3?eingabe=' },
  { id: 'wikipedia', name: 'Wikipedia', icon: '📚', url: 'https://en.wikipedia.org/wiki/Special:Search?search=' },
];

const CHAT_CATEGORIES = [
  { id: 'text', name: 'Text', icon: '💬', description: 'LLMs for chat & completion' },
  { id: 'image', name: 'Image', icon: '🎨', description: 'Stable Diffusion & image generation' },
  { id: 'video', name: 'Video', icon: '🎬', description: 'Video generation & editing' },
  { id: 'audio', name: 'Audio', icon: '🎵', description: 'TTS, STT, music generation' },
  { id: '3d', name: '3D', icon: '🎲', description: '3D model generation' },
  { id: 'code', name: 'Code', icon: '💻', description: 'Code-specific models' },
  { id: 'multimodal', name: 'Multimodal', icon: '🔮', description: 'Vision-language models' },
];

function Chat({ chats, setChats, activeTabId, setActiveTabId, nextChatId, setNextChatId, searchEngine, setSearchEngine }) {
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [showSearchDropdown, setShowSearchDropdown] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [onlineSearch, setOnlineSearch] = useState(false);
  const [agentModeEnabled, setAgentModeEnabled] = useState(false);
  const [showAssistiveSearch, setShowAssistiveSearch] = useState(false);
  const [assistiveQuery, setAssistiveQuery] = useState('');
  const [assistiveEngine, setAssistiveEngine] = useState('duckduckgo');
  const [imageGenerationEnabled, setImageGenerationEnabled] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [speakingMessageId, setSpeakingMessageId] = useState(null);
  const speechSynthesisRef = useRef(null);
  const lastAutoplayMessageRef = useRef(null);
  const chatJustOpenedRef = useRef(true); // Track if chat was just opened
  const [ttsAutoplay, setTtsAutoplay] = useState(() => {
    return localStorage.getItem('tts-autoplay') === 'true';
  });
  const [imageModel, setImageModel] = useState(() => {
    const saved = localStorage.getItem('image-model');
    // If saved value looks like a path (contains /mnt/ or /home/), clear it
    if (saved && (saved.includes('/mnt/') || saved.includes('/home/') || saved.includes('Downloads'))) {
      localStorage.removeItem('image-model');
      return null;
    }
    return saved || null;
  });
  const [imageBackend, setImageBackend] = useState(() => {
    const saved = localStorage.getItem('image-backend');
    return saved || 'diffusers'; // diffusers, webui, comfyui, invokeai
  });
  const [installedBackends, setInstalledBackends] = useState([]);
  const [maxTokens, setMaxTokens] = useState(() => {
    const saved = localStorage.getItem('max-tokens');
    return saved ? parseInt(saved, 10) : 512;
  });
  const [copiedMessageIndex, setCopiedMessageIndex] = useState(null);
  
  // Image generation settings
  const [imgWidth, setImgWidth] = useState(() => {
    const saved = localStorage.getItem('img-width');
    return saved ? parseInt(saved, 10) : 512;
  });
  const [imgHeight, setImgHeight] = useState(() => {
    const saved = localStorage.getItem('img-height');
    return saved ? parseInt(saved, 10) : 512;
  });
  const [imgSteps, setImgSteps] = useState(() => {
    const saved = localStorage.getItem('img-steps');
    return saved ? parseInt(saved, 10) : 20;
  });
  const [imgCfgScale, setImgCfgScale] = useState(() => {
    const saved = localStorage.getItem('img-cfg-scale');
    return saved ? parseFloat(saved) : 7.5;
  });
  const [imgSampler, setImgSampler] = useState(() => {
    const saved = localStorage.getItem('img-sampler');
    return saved || 'euler_a';
  });
  const [negativePrompt, setNegativePrompt] = useState(() => {
    const saved = localStorage.getItem('img-negative-prompt');
    return saved || '';
  });
  const [imgSeed, setImgSeed] = useState(() => {
    const saved = localStorage.getItem('img-seed');
    return saved ? parseInt(saved, 10) : -1;
  });
  
  // Video generation settings
  const [videoGenerationEnabled, setVideoGenerationEnabled] = useState(false);
  const [videoModel, setVideoModel] = useState(() => {
    const saved = localStorage.getItem('video-model');
    if (saved && (saved.includes('/mnt/') || saved.includes('/home/') || saved.includes('Downloads'))) {
      localStorage.removeItem('video-model');
      return null;
    }
    return saved || null;
  });
  const [videoWidth, setVideoWidth] = useState(() => {
    const saved = localStorage.getItem('video-width');
    return saved ? parseInt(saved, 10) : 1024;
  });
  const [videoHeight, setVideoHeight] = useState(() => {
    const saved = localStorage.getItem('video-height');
    return saved ? parseInt(saved, 10) : 576;
  });
  const [videoNumFrames, setVideoNumFrames] = useState(() => {
    const saved = localStorage.getItem('video-num-frames');
    return saved ? parseInt(saved, 10) : 25;
  });
  const [videoFps, setVideoFps] = useState(() => {
    const saved = localStorage.getItem('video-fps');
    return saved ? parseInt(saved, 10) : 7;
  });
  const [videoMotionBucket, setVideoMotionBucket] = useState(() => {
    const saved = localStorage.getItem('video-motion-bucket');
    return saved ? parseInt(saved, 10) : 127;
  });
  const [videoNoiseStrength, setVideoNoiseStrength] = useState(() => {
    const saved = localStorage.getItem('video-noise-strength');
    return saved ? parseFloat(saved) : 0.02;
  });
  const [videoSeed, setVideoSeed] = useState(() => {
    const saved = localStorage.getItem('video-seed');
    return saved ? parseInt(saved, 10) : -1;
  });
  
  // File attachments
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [referenceImage, setReferenceImage] = useState(null);
  const fileInputRef = useRef(null);
  const refImageInputRef = useRef(null);
  
  const messagesEndRef = useRef(null);

  const activeChat = chats.find(chat => chat.id === activeTabId);

  // Auto-enable features based on chat type
  useEffect(() => {
    // Reset the flag when switching chats to prevent auto-TTS
    chatJustOpenedRef.current = true;
    
    if (activeChat?.type === 'image') {
      setImageGenerationEnabled(true);
      setVideoGenerationEnabled(false);
    } else if (activeChat?.type === 'video') {
      setImageGenerationEnabled(false);
      setVideoGenerationEnabled(true);
    } else {
      setImageGenerationEnabled(false);
      setVideoGenerationEnabled(false);
    }
  }, [activeChat?.type, activeTabId]);

  useEffect(() => {
    const handleStorageChange = () => {
      setTtsAutoplay(localStorage.getItem('tts-autoplay') === 'true');
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // Save maxTokens to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('max-tokens', maxTokens.toString());
  }, [maxTokens]);

  // Save image generation settings to localStorage
  useEffect(() => {
    localStorage.setItem('img-width', imgWidth.toString());
  }, [imgWidth]);
  
  useEffect(() => {
    localStorage.setItem('img-height', imgHeight.toString());
  }, [imgHeight]);
  
  useEffect(() => {
    localStorage.setItem('img-steps', imgSteps.toString());
  }, [imgSteps]);
  
  useEffect(() => {
    localStorage.setItem('img-cfg-scale', imgCfgScale.toString());
  }, [imgCfgScale]);
  
  useEffect(() => {
    localStorage.setItem('img-sampler', imgSampler);
  }, [imgSampler]);
  
  useEffect(() => {
    localStorage.setItem('img-negative-prompt', negativePrompt);
  }, [negativePrompt]);
  
  useEffect(() => {
    localStorage.setItem('img-seed', imgSeed.toString());
  }, [imgSeed]);

  // Save video generation settings to localStorage
  useEffect(() => {
    localStorage.setItem('video-width', videoWidth.toString());
  }, [videoWidth]);
  
  useEffect(() => {
    localStorage.setItem('video-height', videoHeight.toString());
  }, [videoHeight]);
  
  useEffect(() => {
    localStorage.setItem('video-num-frames', videoNumFrames.toString());
  }, [videoNumFrames]);
  
  useEffect(() => {
    localStorage.setItem('video-fps', videoFps.toString());
  }, [videoFps]);
  
  useEffect(() => {
    localStorage.setItem('video-motion-bucket', videoMotionBucket.toString());
  }, [videoMotionBucket]);
  
  useEffect(() => {
    localStorage.setItem('video-noise-strength', videoNoiseStrength.toString());
  }, [videoNoiseStrength]);
  
  useEffect(() => {
    localStorage.setItem('video-seed', videoSeed.toString());
  }, [videoSeed]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Stop TTS when switching chats or unmounting
  useEffect(() => {
    return () => {
      if (speechSynthesisRef.current) {
        speechSynthesisRef.current.pause();
        speechSynthesisRef.current = null;
        setSpeakingMessageId(null);
      }
    };
  }, [activeTabId]);

  // Listen for page content from assistive search window
  useEffect(() => {
    const unlisten = listen('page-content-ready', async (event) => {
      const { title, url, content, action, customPrompt } = event.payload;
      
      // Add user message showing what was captured
      const userMessage = action === 'summarize' 
        ? `📄 Summarize this page:\n\nTitle: ${title}\nURL: ${url}`
        : `${customPrompt}\n\nPage: ${title}\nURL: ${url}`;
      
      addMessage(activeTabId, {
        role: 'user',
        content: userMessage
      });
      
      // Check if model is selected
      if (!activeChat?.model) {
        addMessage(activeTabId, {
          role: 'assistant',
          content: 'No model selected. Please select a model to process the page.'
        });
        return;
      }
      
      // Build the prompt
      const prompt = action === 'summarize'
        ? `Please provide a concise summary of the following webpage:\n\nTitle: ${title}\nURL: ${url}\n\nContent:\n${content.substring(0, 8000)}\n\nSummary:`
        : `${customPrompt}\n\nWebpage Title: ${title}\nURL: ${url}\n\nContent:\n${content.substring(0, 8000)}\n\nResponse:`;
      
      // Generate response
      setIsGenerating(true);
      try {
        const result = await invoke('run_inference', {
          modelId: activeChat.model,
          prompt: prompt,
          maxTokens: 1000
        });
        
        if (result.success) {
          addMessage(activeTabId, {
            role: 'assistant',
            content: result.text
          });
        } else {
          addMessage(activeTabId, {
            role: 'assistant',
            content: `❌ Error: ${result.error || 'Failed to process page'}`
          });
        }
      } catch (error) {
        console.error('Failed to process page:', error);
        addMessage(activeTabId, {
          role: 'assistant',
          content: `❌ Failed to process page: ${error}`
        });
      } finally {
        setIsGenerating(false);
      }
    });
    
    return () => {
      unlisten.then(fn => fn());
    };
  }, [activeTabId, activeChat]); // Stop when active tab changes

  useEffect(() => {
    const msgs = activeChat?.messages || [];
    scrollToBottom();
    
    // Skip autoplay if chat was just opened
    if (chatJustOpenedRef.current) {
      chatJustOpenedRef.current = false;
      return;
    }
    
    // Autoplay TTS for new assistant messages
    if (ttsAutoplay && msgs.length > 0) {
      const lastMessage = msgs[msgs.length - 1];
      const messageKey = `${msgs.length - 1}-${lastMessage.content?.substring(0, 50)}`;
      
      // Only trigger TTS if this is a complete assistant message and we haven't played it yet
      if (lastMessage.role === 'assistant' && 
          !lastMessage.image && 
          !lastMessage.video && 
          !isGenerating && // Wait until generation is complete
          lastAutoplayMessageRef.current !== messageKey &&
          speakingMessageId === null) { // Don't start if already speaking
        lastAutoplayMessageRef.current = messageKey;
        // Start TTS immediately
        speakMessage(lastMessage.content, msgs.length - 1);
      }
    }
  }, [activeChat?.messages, isGenerating, ttsAutoplay, speakingMessageId]);

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (activeChat?.type === 'image') {
      checkInstalledBackends();
    }
  }, [activeChat?.type]);

  const checkInstalledBackends = async () => {
    try {
      // For now, just assume diffusers is available since we removed backend commands
      setInstalledBackends(['diffusers']);
      setImageBackend('diffusers');
      localStorage.setItem('image-backend', 'diffusers');
    } catch (error) {
      console.error('Failed to check backends:', error);
    }
  };

  // Validate models when available models list changes
  useEffect(() => {
    if (availableModels.length > 0) {
      validateChatModels();
    }
  }, [availableModels]);

  const loadModels = async () => {
    try {
      const models = await invoke('list_downloaded_models');
      setAvailableModels(models || []);
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const validateChatModels = () => {
    const availableModelIds = availableModels.map(m => m.id);
    setChats(prevChats => prevChats.map(chat => {
      // If chat has a model set but it's no longer available
      if (chat.model && !availableModelIds.includes(chat.model)) {
        // Check if the last message is already a "model deleted" warning
        const lastMessage = chat.messages[chat.messages.length - 1];
        const isAlreadyWarned = lastMessage?.content?.includes('has been deleted');
        
        if (!isAlreadyWarned) {
          return {
            ...chat,
            model: null,
            messages: [
              ...chat.messages,
              {
                role: 'assistant',
                content: `⚠️ The model "${chat.model}" has been deleted. Please choose another model from the dropdown above.`
              }
            ]
          };
        }
        return { ...chat, model: null };
      }
      return chat;
    }));
  };

  const updateChatProperty = (chatId, property, value) => {
    setChats(prevChats => prevChats.map(chat =>
      chat.id === chatId ? { ...chat, [property]: value } : chat
    ));
  };

  const handleSelectModel = (modelId) => {
    updateChatProperty(activeTabId, 'model', modelId);
    setShowModelDropdown(false);
    addMessage(activeTabId, {
      role: 'assistant',
      content: `💻 Model ${modelId} selected! You can now chat with me.`
    });
  };

  const addMessage = (chatId, message) => {
    setChats(prevChats => prevChats.map(chat =>
      chat.id === chatId
        ? { ...chat, messages: [...chat.messages, message] }
        : chat
    ));
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    
    // Stop any currently playing TTS
    if (speechSynthesisRef.current) {
      stopSpeaking();
    }

    const userMessage = { role: 'user', content: input.trim() };
    const userInput = input.trim();
    addMessage(activeTabId, userMessage);
    setInput('');
    setIsGenerating(true);

    try {
      // Handle image generation mode
      if (activeChat?.type === 'image') {
        addMessage(activeTabId, {
          role: 'assistant',
          content: '🎨 Generating image...',
          id: Date.now(),
          isGenerating: true
        });
        
        try {
          // Check if backend is available
          if (imageBackend !== 'diffusers') {
            throw new Error(`${imageBackend.toUpperCase()} backend requires:\n\n1. Install the backend from the Diffusers tab\n2. Start the server with API enabled:\n   - WebUI: --api flag\n   - ComfyUI: default API on port 8188\n   - InvokeAI: default API on port 9090\n\nCurrently only Diffusers (local script) is supported.`);
          }
          
          const imageParams = {
            prompt: userInput,
            negativePrompt: negativePrompt || null,
            width: imgWidth,
            height: imgHeight,
            steps: imgSteps,
            cfgScale: imgCfgScale,
            sampler: imgSampler,
            seed: imgSeed,
            referenceImage: referenceImage ? referenceImage.path : null,
            modelId: imageModel, // Pass selected model or null for default
            backend: imageBackend // Pass backend type
          };
          
          const result = await invoke('generate_image', imageParams);
          
          // Remove generating message and add image result
          setChats(prevChats => prevChats.map(chat => {
            if (chat.id === activeTabId) {
              const messages = chat.messages.filter(m => !m.isGenerating);
              return {
                ...chat,
                messages: [...messages, {
                  role: 'assistant',
                  content: `Generated image for: "${userInput}"`,
                  image: result.image_path,
                  metadata: {
                    width: imgWidth,
                    height: imgHeight,
                    steps: imgSteps,
                    cfg_scale: imgCfgScale,
                    sampler: imgSampler,
                    seed: result.seed
                  }
                }]
              };
            }
            return chat;
          }));
        } catch (error) {
          // Remove generating message and show error with detailed settings
          const settingsInfo = `
📊 Settings Used:
  • Backend: ${imageBackend}
  • Model: ${imageModel || 'auto-detect from Downloads folder'}
  • Dimensions: ${imgWidth}x${imgHeight}
  • Steps: ${imgSteps}
  • CFG Scale: ${imgCfgScale}
  • Sampler: ${imgSampler}
  • Seed: ${imgSeed === -1 ? 'random' : imgSeed}
  • Negative Prompt: ${negativePrompt || 'none'}

❌ Error: ${error}`;
          
          setChats(prevChats => prevChats.map(chat => {
            if (chat.id === activeTabId) {
              const messages = chat.messages.filter(m => !m.isGenerating);
              return {
                ...chat,
                messages: [...messages, {
                  role: 'assistant',
                  content: settingsInfo
                }]
              };
            }
            return chat;
          }));
        }
        setIsGenerating(false);
        return;
      }

      // Handle video generation mode
      if (activeChat?.type === 'video') {
        if (!referenceImage) {
          addMessage(activeTabId, {
            role: 'assistant',
            content: '⚠️ Video generation requires a source image. Please upload an image using the 🖼️ button.'
          });
          setIsGenerating(false);
          return;
        }

        addMessage(activeTabId, {
          role: 'assistant',
          content: '🎬 Generating video...',
          id: Date.now(),
          isGenerating: true
        });
        
        try {
          const videoParams = {
            referenceImage: referenceImage.path, // Use the file path
            width: videoWidth,
            height: videoHeight,
            numFrames: videoNumFrames,
            fps: videoFps,
            motionBucketId: videoMotionBucket,
            noiseAugStrength: videoNoiseStrength,
            seed: videoSeed,
            modelId: videoModel
          };
          
          const result = await invoke('generate_video', videoParams);
          
          // Remove generating message and add video result
          setChats(prevChats => prevChats.map(chat => {
            if (chat.id === activeTabId) {
              const messages = chat.messages.filter(m => !m.isGenerating);
              return {
                ...chat,
                messages: [...messages, {
                  role: 'assistant',
                  content: userInput ? `Generated video: "${userInput}"` : 'Generated video from image',
                  video: result.video_path,
                  metadata: {
                    num_frames: result.num_frames,
                    fps: result.fps,
                    seed: result.seed
                  }
                }]
              };
            }
            return chat;
          }));
          
          setInput('');
          setReferenceImage(null);
        } catch (error) {
          const settingsInfo = `📊 Settings Used:
  • Model: ${videoModel || 'auto-detect from Downloads folder'}
  • Resolution: ${videoWidth}x${videoHeight}
  • Frames: ${videoNumFrames}
  • FPS: ${videoFps}
  • Motion Bucket: ${videoMotionBucket}
  • Noise Strength: ${videoNoiseStrength}
  • Seed: ${videoSeed === -1 ? 'random' : videoSeed}

❌ Error: ${error}`;
          
          setChats(prevChats => prevChats.map(chat => {
            if (chat.id === activeTabId) {
              const messages = chat.messages.filter(m => !m.isGenerating);
              return {
                ...chat,
                messages: [...messages, {
                  role: 'assistant',
                  content: settingsInfo
                }]
              };
            }
            return chat;
          }));
        }
        setIsGenerating(false);
        return;
      }

      // Check if model is needed for this chat type
      if (!activeChat.model && activeChat.type !== 'image' && activeChat.type !== 'video') {
        addMessage(activeTabId, {
          role: 'assistant',
          content: 'No model is currently loaded. Please select a model from the dropdown above.'
        });
        setIsGenerating(false);
        return;
      }

      let finalPrompt = userInput;
      let searchSources = null;

      // Perform web search if enabled
      if (onlineSearch) {
        setIsSearching(true);
        try {
          if (searchEngine === 'multi-engine') {
            // Get enabled engines from localStorage
            const enabledEngines = JSON.parse(localStorage.getItem('enabled-search-engines') || '["duckduckgo", "brave", "bing"]');
            console.log('Performing multi-engine search with:', enabledEngines);
            
            // Search all enabled engines in parallel
            const allResults = await Promise.all(
              enabledEngines.map(async (engine) => {
                try {
                  const results = await invoke('web_search', { 
                    query: userInput,
                    engine: engine 
                  });
                  return { engine, results: results || [] };
                } catch (error) {
                  console.error(`Search failed for ${engine}:`, error);
                  return { engine, results: [] };
                }
              })
            );
            
            // Combine and format results from all engines
            let combinedResults = [];
            let contextParts = [];
            let sourceCounter = 1;
            
            allResults.forEach(({ engine, results }) => {
              if (results && results.length > 0) {
                const engineInfo = SEARCH_ENGINES.find(e => e.id === engine);
                const engineIcon = engineInfo?.icon || '🔍';
                const engineName = engineInfo?.name || engine;
                
                // Take top 2 results from each engine
                const topResults = results.slice(0, 2);
                contextParts.push(`\n--- ${engineIcon} ${engineName} ---`);
                
                topResults.forEach((r) => {
                  combinedResults.push(r);
                  contextParts.push(`[${sourceCounter}] ${r.title}\n${r.snippet}\nSource: ${r.url}`);
                  sourceCounter++;
                });
              }
            });
            
            if (combinedResults.length > 0) {
              searchSources = combinedResults.slice(0, 10);
              const context = contextParts.join('\n\n');
              finalPrompt = `Context from multi-engine web search:\n${context}\n\nUser question: ${userInput}\n\nPlease answer based on the context above if relevant, and reference sources using [1], [2], [3], etc. notation when citing information.`;
            }
          } else {
            // Single engine search
            console.log('Performing web search with engine:', searchEngine);
            const searchResults = await invoke('web_search', { 
              query: userInput,
              engine: searchEngine 
            });
            console.log('Search results:', searchResults);
            if (searchResults && searchResults.length > 0) {
              // Save sources for citation display
              searchSources = searchResults.slice(0, 3);
              
              // Find the selected engine's icon
              const engineIcon = SEARCH_ENGINES.find(e => e.id === searchEngine)?.icon || '🔍';
              
              const context = searchSources
                .map((r, i) => `[${i + 1}] ${r.title}\n${r.snippet}\nSource: ${r.url}`)
                .join('\n\n');
              finalPrompt = `Context from web search (${engineIcon} ${searchEngine}):\n${context}\n\nUser question: ${userInput}\n\nPlease answer based on the context above if relevant, and reference sources using [1], [2], [3] notation when citing information.`;
            }
          }
        } catch (error) {
          console.error('Web search failed:', error);
          addMessage(activeTabId, {
            role: 'assistant',
            content: `⚠️ Web search failed: ${error}. Answering without search results...`
          });
        } finally {
          setIsSearching(false);
        }
      }

      // Apply GPU settings before running inference
      const gpuThreshold = parseFloat(localStorage.getItem('gpu-memory-threshold') || '8.0');
      const forceCpu = localStorage.getItem('force-cpu-mode') === 'true';
      await invoke('set_gpu_settings', { gpuThreshold, forceCpu });

      // Create a placeholder message for streaming
      const streamingMessageId = Date.now();
      addMessage(activeTabId, {
        role: 'assistant',
        content: '',
        id: streamingMessageId,
        isStreaming: true
      });

      let streamedText = '';
      
      // Listen for streaming tokens
      const unlisten = await listen('inference-token', (event) => {
        streamedText += event.payload.token;
        // Update the streaming message
        setChats(prevChats => prevChats.map(chat => {
          if (chat.id === activeTabId) {
            return {
              ...chat,
              messages: chat.messages.map(msg => 
                msg.id === streamingMessageId 
                  ? { ...msg, content: streamedText }
                  : msg
              )
            };
          }
          return chat;
        }));
        scrollToBottom();
      });

      const result = await invoke('run_streaming_inference', {
        modelId: activeChat.model,
        prompt: finalPrompt,
        maxTokens: maxTokens
      });

      // Stop listening
      unlisten();

      if (result.success) {
        // Update final message with complete text and stats
        setChats(prevChats => prevChats.map(chat => {
          if (chat.id === activeTabId) {
            return {
              ...chat,
              messages: chat.messages.map(msg => 
                msg.id === streamingMessageId 
                  ? { 
                      ...msg, 
                      content: result.text,
                      stats: result.stats,
                      sources: searchSources,
                      isStreaming: false
                    }
                  : msg
              )
            };
          }
          return chat;
        }));

        // Execute agent commands if agent mode is enabled
        if (agentModeEnabled) {
          try {
            const executionResults = await invoke('execute_agent_commands', {
              llmResponse: result.text
            });

            if (executionResults && executionResults.length > 0) {
              // Format execution results as a message
              const resultsText = executionResults
                .map(r => r.message)
                .join('\n');
              
              addMessage(activeTabId, {
                role: 'assistant',
                content: `**Agent Execution:**\n${resultsText}`
              });
            }
          } catch (error) {
            console.error('Agent execution failed:', error);
            addMessage(activeTabId, {
              role: 'assistant',
              content: `⚠️ Agent execution error: ${error}`
            });
          }
        }
      } else {
        // Check if it's an OOM error and provide helpful suggestion
        const isOOMError = result.error?.toLowerCase().includes('out of memory') || 
                           result.error?.toLowerCase().includes('oom');
        
        let errorMessage = `Error: ${result.error || 'Generation failed'}`;
        
        // Add execution settings info for debugging
        if (result.execution_settings) {
          const settings = result.execution_settings;
          errorMessage += `\n\n**Current Settings:**\n- Mode: ${settings.mode}\n- Quantization: ${settings.quantization}\n- GPU Layers: ${settings.gpu_layers || 'N/A'}`;
        }
        
        if (isOOMError) {
          errorMessage += `\n\n💡 **Suggestion:** This model ran out of GPU memory. Go to the **Models** tab and try:\n- **4-bit quantization** (reduces memory by ~75%, best for large models)\n- **8-bit quantization** (reduces memory by ~50%, balanced)\n- Or enable **Force CPU Mode** in Settings if quantization doesn't help`;
        }
    
        addMessage(activeTabId, {
          role: 'assistant',
          content: errorMessage
        });
      }
    } catch (error) {
      console.error('Chat error:', error);
      addMessage(activeTabId, {
        role: 'assistant',
        content: `Error: ${error}`
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  };
  
  const handleFileAttachment = (e) => {
    const files = Array.from(e.target.files);
    const newFiles = files.map(file => ({
      name: file.name,
      size: file.size,
      type: file.type,
      file: file
    }));
    setAttachedFiles(prev => [...prev, ...newFiles]);
  };
  
  const handleReferenceImage = async (e) => {
    const file = e.target.files[0];
    if (file) {
      try {
        // Read file as array buffer
        const arrayBuffer = await file.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        
        // Save to temp location and get path
        const tempPath = await invoke('save_temp_file', {
          fileName: file.name,
          fileData: Array.from(uint8Array)
        });
        
        setReferenceImage({
          name: file.name,
          size: file.size,
          type: file.type,
          file: file,
          path: tempPath,
          preview: URL.createObjectURL(file)
        });
      } catch (error) {
        console.error('Failed to save reference image:', error);
        alert('Failed to upload reference image: ' + error);
      }
    }
  };
  
  const removeAttachment = (index) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };
  
  const removeReferenceImage = () => {
    if (referenceImage?.preview) {
      URL.revokeObjectURL(referenceImage.preview);
    }
    setReferenceImage(null);
  };
  
  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };
  
  const getFileIcon = (type) => {
    if (type.startsWith('image/')) return '🖼️';
    if (type.startsWith('video/')) return '🎬';
    if (type.includes('pdf')) return '📄';
    if (type.includes('sheet') || type.includes('excel')) return '📊';
    if (type.includes('document') || type.includes('word')) return '📝';
    if (type.startsWith('text/')) return '📃';
    return '📎';
  };
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const copyMessage = async (content, index) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageIndex(index);
      setTimeout(() => setCopiedMessageIndex(null), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const speakMessage = async (content, messageId) => {
    // If already speaking this message, stop it
    if (speakingMessageId === messageId) {
      stopSpeaking();
      return;
    }
    
    // Stop any other currently playing speech
    if (speechSynthesisRef.current) {
      stopSpeaking();
    }
    
    // Clean URLs from content before TTS
    let cleanContent = content;
    // Replace full URLs with "link"
    cleanContent = cleanContent.replace(/https?:\/\/[^\s]+/g, 'https link');
    // Replace remaining URL-like patterns
    cleanContent = cleanContent.replace(/www\.[^\s]+/g, 'www blaa blaa blaa');
    cleanContent = cleanContent.replace(/\/\/[^\s]+/g, 'link to source');
    
    // Load basic settings
    const selectedVoice = localStorage.getItem('tts-voice') || 'female_neutral';
    const speed = parseFloat(localStorage.getItem('tts-speed')) || 1.0;
    const volume = parseFloat(localStorage.getItem('tts-volume')) || 1.0;
    const pitch = parseFloat(localStorage.getItem('tts-pitch')) || 1.0;
    
    // Load advanced settings
    const advancedSettings = JSON.parse(localStorage.getItem('advanced-tts-settings') || '{}');
    const prosody = advancedSettings.prosody || 0.5;
    const stability = advancedSettings.stability || 0.7;
    
    // Build post-processing settings JSON
    const settings = {
      warmth: advancedSettings.warmth || 0,
      presence: advancedSettings.presence || 0,
      air: advancedSettings.air || 0,
      reverb_size: advancedSettings.reverbSize || 0,
      breath_volume: advancedSettings.breathVolume || 0,
      de_esser: advancedSettings.deEsser || 0,
      normalize: advancedSettings.normalize !== false,
      limiter: advancedSettings.limiter !== false
    };
    
    setSpeakingMessageId(messageId);
    
    try {
      // Use F5-TTS with full parameters
      const audioBytes = await invoke('text_to_speech', {
        text: cleanContent,
        voice: selectedVoice,
        speed: speed,
        pitch: pitch,
        prosody: prosody,
        stability: stability,
        settings: JSON.stringify(settings)
      });
      
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
        throw new Error('No audio returned from TTS engine');
      }
      
      // Convert bytes to blob and play
      const blob = new Blob([new Uint8Array(byteArray)], { type: 'audio/mpeg' });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      
      // Apply volume
      audio.volume = volume;
      
      audio.onended = () => {
        URL.revokeObjectURL(url);
        setSpeakingMessageId(null);
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        setSpeakingMessageId(null);
      };
      
      speechSynthesisRef.current = audio;
      await audio.play();
    } catch (error) {
      console.error('F5-TTS failed:', error);
      setSpeakingMessageId(null);
    }
  };

  const stopSpeaking = () => {
    if (speechSynthesisRef.current) {
      if (speechSynthesisRef.current instanceof Audio) {
        speechSynthesisRef.current.pause();
        speechSynthesisRef.current.currentTime = 0;
      } else {
        window.speechSynthesis.cancel();
      }
      speechSynthesisRef.current = null;
    }
    setSpeakingMessageId(null);
  };

  // Stop any ongoing TTS playback when switching chats/tabs or unmounting
  useEffect(() => {
    stopSpeaking();
    return () => stopSpeaking();
  }, [activeTabId]);

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div className="chat-title">
          <div className="title-icon">{CHAT_CATEGORIES.find(c => c.id === activeChat?.type)?.icon || '🤖'}</div>
          <h2>{activeChat?.name || 'Chat'}</h2>
          {activeChat?.type !== 'image' && activeChat?.type !== 'video' && (
            <div className="model-selector-wrapper">
              <button
                className={`model-selector ${activeChat?.model ? 'loaded' : 'not-loaded'}`}
                onClick={() => setShowModelDropdown(!showModelDropdown)}
                disabled={isGenerating}
              >
                {activeChat?.model ? (
                  <>
                    <span className="check-icon">✓</span>
                    <span className="model-name">{activeChat.model}</span>
                    {(() => {
                      const modelInfo = availableModels.find(m => m.id === activeChat.model);
                      if (modelInfo) {
                        const quantBadges = {
                          '4bit': <span className="quant-badge quant-4bit">4-bit</span>,
                          '8bit': <span className="quant-badge quant-8bit">8-bit</span>,
                          'fp16': <span className="quant-badge quant-fp16">FP16</span>,
                          'native': <span className="quant-badge quant-native">Native</span>
                        };
                        return quantBadges[modelInfo.quantization] || null;
                      }
                      return null;
                    })()}
                  </>
                ) : (
                  <>
                    <span className="alert-icon">⚠</span>
                    <span className="model-name">Select a model</span>
                  </>
                )}
                <span className="chevron">▼</span>
              </button>

              {showModelDropdown && (
                <div className="model-dropdown">
                  <div className="dropdown-header">
                    💻 Downloaded Models
                  </div>
                  {availableModels.length === 0 ? (
                    <div className="dropdown-empty">
                      No models available. Download from Model Manager tab.
                    </div>
                  ) : (
                    availableModels.map((modelInfo, index) => {
                      const quantBadges = {
                        '4bit': '⚡4-bit',
                        '8bit': '⚡8-bit',
                        'fp16': 'FP16',
                        'native': 'Native'
                      };
                      return (
                        <button
                          key={index}
                          className={`dropdown-item ${activeChat?.model === modelInfo.id ? 'active' : ''}`}
                          onClick={() => handleSelectModel(modelInfo.id)}
                        >
                          {activeChat?.model === modelInfo.id && <span className="check-icon">✓</span>}
                          <span>{modelInfo.id}</span>
                          <span className="dropdown-quant-badge">{quantBadges[modelInfo.quantization]}</span>
                        </button>
                      );
                    })
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="chat-main-content">
        <div className="messages-container">
        {activeChat?.messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <div className="message-avatar">
              {message.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="message-content">
              <div className="message-text">
                <ReactMarkdown
                  components={{
                    code: CodeBlock
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
              {message.image && (
                <div className="message-image">
                  <img src={convertFileSrc(message.image)} alt="Generated" />
                  {message.metadata && (
                    <div className="image-metadata">
                      <span>Size: {message.metadata.width}x{message.metadata.height}</span>
                      <span>Steps: {message.metadata.steps}</span>
                      <span>CFG: {message.metadata.cfg_scale}</span>
                      <span>Sampler: {message.metadata.sampler}</span>
                      <span>Seed: {message.metadata.seed}</span>
                    </div>
                  )}
                </div>
              )}
              {message.video && (
                <div className="message-video">
                  <video controls autoPlay muted loop>
                    <source src={convertFileSrc(message.video)} type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                  {message.metadata && (
                    <div className="image-metadata">
                      <span>Frames: {message.metadata.num_frames}</span>
                      <span>FPS: {message.metadata.fps}</span>
                      <span>Seed: {message.metadata.seed}</span>
                    </div>
                  )}
                </div>
              )}
              {message.sources && message.sources.length > 0 && (
                <div className="message-sources">
                  <div className="sources-header">📚 Sources:</div>
                  {message.sources.map((source, idx) => (
                    <div key={idx} className="source-item">
                      <span className="source-number">[{idx + 1}]</span>
                      <a href={source.url} target="_blank" rel="noopener noreferrer" className="source-link">
                        {source.title}
                      </a>
                    </div>
                  ))}
                </div>
              )}
              {message.stats && (
                <div className="message-stats">
                  <span title="Tokens used">🧠 {message.stats.total_tokens} tokens ({message.stats.new_tokens} new)</span>
                  <span title="Generation time">⏱️ {message.stats.time_seconds}s</span>
                  <span title="Speed">⚡ {message.stats.tokens_per_second} tok/s</span>
                </div>
              )}
              <div className="message-actions">
                <button 
                  className="copy-button"
                  onClick={() => copyMessage(message.content, index)}
                  title="Copy message"
                >
                  {copiedMessageIndex === index ? '✓ Copied' : '📋 Copy'}
                </button>
                <button 
                  className={`speak-button ${speakingMessageId === index ? 'speaking' : ''}`}
                  onClick={() => speakingMessageId === index ? stopSpeaking() : speakMessage(message.content, index)}
                  title={speakingMessageId === index ? "Stop speaking" : "Play message"}
                >
                  {speakingMessageId === index ? '⏸️ Stop' : '▶️ Play'}
                </button>
              </div>
            </div>
          </div>
        ))}
        {(isGenerating || isSearching) && (
          <div className="message assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
              {isSearching && <div style={{ marginTop: '4px', fontSize: '12px', color: '#999' }}>🔍 Searching the web...</div>}
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <div className="input-options">
          {activeChat?.type === 'text' && (
            <>
              <label className="search-toggle">
                <input
                  type="checkbox"
                  checked={onlineSearch}
                  onChange={(e) => setOnlineSearch(e.target.checked)}
                />
                <span>🔍 Enable online search</span>
              </label>
              
              <label className="agent-toggle">
                <input
                  type="checkbox"
                  checked={agentModeEnabled}
                  onChange={(e) => setAgentModeEnabled(e.target.checked)}
                />
                <span>🤖 Enable agent mode</span>
              </label>
              
              <button 
                className="assistive-search-btn"
                onClick={() => setShowAssistiveSearch(!showAssistiveSearch)}
              >
                🔍 Assistive Search {showAssistiveSearch ? '▲' : '▼'}
              </button>
            </>
          )}
          
          {onlineSearch && activeChat?.type === 'text' && (
            <div className="engine-selector-wrapper">
              <button
                className="engine-selector"
                onClick={() => setShowSearchDropdown(!showSearchDropdown)}
              >
                <span>
                  {SEARCH_ENGINES.find(e => e.id === searchEngine)?.icon || '🔍'}
                  {' '}
                  {SEARCH_ENGINES.find(e => e.id === searchEngine)?.name || 'Select Engine'}
                </span>
                <span className="chevron">{showSearchDropdown ? '▲' : '▼'}</span>
              </button>
              {showSearchDropdown && (
                <div className="search-dropdown">
                  {SEARCH_ENGINES.map(engine => (
                    <button
                      key={engine.id}
                      className={`dropdown-item ${searchEngine === engine.id ? 'active' : ''}`}
                      onClick={() => {
                        setSearchEngine(engine.id);
                        setShowSearchDropdown(false);
                      }}
                    >
                      {searchEngine === engine.id && <span className="check-icon">✓</span>}
                      <span>{engine.icon} {engine.name}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          
          {showAssistiveSearch && activeChat?.type === 'text' && (
            <div className="assistive-search-panel">
              <div className="assistive-search-controls">
                <select 
                  value={assistiveEngine} 
                  onChange={(e) => setAssistiveEngine(e.target.value)}
                  className="engine-selector-small"
                >
                  {SEARCH_ENGINES.map(engine => (
                    <option key={engine.id} value={engine.id}>
                      {engine.icon} {engine.name}
                    </option>
                  ))}
                </select>

                <input
                  type="text"
                  value={assistiveQuery}
                  onChange={(e) => setAssistiveQuery(e.target.value)}
                  onKeyPress={async (e) => {
                    if (e.key === 'Enter' && assistiveQuery.trim()) {
                      const engine = SEARCH_ENGINES.find(eng => eng.id === assistiveEngine);
                      const searchUrl = engine.url + encodeURIComponent(assistiveQuery);
                      try {
                        await invoke('open_search_window', { url: searchUrl });
                      } catch (error) {
                        console.error('Failed to open search:', error);
                      }
                    }
                  }}
                  placeholder="Enter search query..."
                  className="assistive-search-input"
                />

                <button
                  onClick={async () => {
                    if (!assistiveQuery.trim()) return;
                    const engine = SEARCH_ENGINES.find(eng => eng.id === assistiveEngine);
                    const searchUrl = engine.url + encodeURIComponent(assistiveQuery);
                    try {
                      await invoke('open_search_window', { url: searchUrl });
                    } catch (error) {
                      console.error('Failed to open search:', error);
                    }
                  }}
                  className="open-search-btn-small"
                  disabled={!assistiveQuery.trim()}
                >
                  Open Search
                </button>
              </div>
              
              <p className="assistive-search-hint">
                💡 Use the floating panel in the browser window to summarize pages or send custom prompts
              </p>
            </div>
          )}
          
          {activeChat?.type === 'text' && (
            <div className="max-tokens-control">
              <label htmlFor="max-tokens">Max tokens:</label>
              <input
                id="max-tokens"
                type="number"
                min="64"
                max="4096"
                step="64"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value) || 512)}
                className="max-tokens-input"
              />
            </div>
          )}
        </div>
        
        {activeChat?.type === 'image' && (
          <div className="image-generation-options">
            <div className="img-backend-selector">
              <label>Backend:</label>
              {installedBackends.length > 0 ? (
                <>
                  <select
                    value={imageBackend}
                    onChange={(e) => {
                      setImageBackend(e.target.value);
                      localStorage.setItem('image-backend', e.target.value);
                    }}
                  >
                    {installedBackends.includes('diffusers') && (
                      <option value="diffusers">🤗 Diffusers (Local Script)</option>
                    )}
                    {installedBackends.includes('automatic1111') && (
                      <option value="webui">🖼️ AUTOMATIC1111 WebUI (API)</option>
                    )}
                    {installedBackends.includes('comfyui') && (
                      <option value="comfyui">🎨 ComfyUI (API)</option>
                    )}
                    {installedBackends.includes('invokeai') && (
                      <option value="invokeai">⚡ InvokeAI (API)</option>
                    )}
                  </select>
                  {imageBackend !== 'diffusers' && (
                    <span className="backend-note">⚠️ Requires server running with API enabled</span>
                  )}
                </>
              ) : (
                <span className="backend-note">⚠️ No backends installed. Go to Diffusers tab to install one.</span>
              )}
            </div>
            
            <div className="img-backend-selector">
              <label>Model:</label>
              <select
                value={imageModel || ''}
                onChange={(e) => {
                  const value = e.target.value || null;
                  setImageModel(value);
                  localStorage.setItem('image-model', value || '');
                }}
              >
                <option value="">Auto-detect (first SD model in Downloads)</option>
                {availableModels
                  .filter(m => {
                    // Filter for image generation models
                    const id = m.id.toLowerCase();
                    return id.includes('stable-diffusion') || 
                           id.includes('sd-') || 
                           id.includes('sdxl') ||
                           id.includes('diffusion') ||
                           id.includes('text-to-image');
                  })
                  .map(model => {
                    // model.id is already in the correct format from Rust (e.g., "stabilityai/stable-diffusion-xl-base-1.0")
                    // No need to extract or normalize - just use it directly
                    return (
                      <option key={model.id} value={model.id}>
                        {model.id}
                      </option>
                    );
                  })
                }
              </select>
              <span className="backend-note">💡 Select your downloaded SD model or use auto-detect</span>
            </div>
            
            <div className="img-option-row">
              <div className="img-option">
                <label>Width:</label>
                <input
                  type="number"
                  min="128"
                  max="2048"
                  step="1"
                  value={imgWidth}
                  onChange={(e) => setImgWidth(parseInt(e.target.value) || 512)}
                />
              </div>
              <div className="img-option">
                <label>Height:</label>
                <input
                  type="number"
                  min="128"
                  max="2048"
                  step="1"
                  value={imgHeight}
                  onChange={(e) => setImgHeight(parseInt(e.target.value) || 512)}
                />
              </div>
              <div className="img-option">
                <label>Steps:</label>
                <input
                  type="number"
                  min="1"
                  max="150"
                  value={imgSteps}
                  onChange={(e) => setImgSteps(parseInt(e.target.value) || 20)}
                />
              </div>
              <div className="img-option">
                <label>CFG Scale:</label>
                <input
                  type="number"
                  min="1"
                  max="30"
                  step="0.5"
                  value={imgCfgScale}
                  onChange={(e) => setImgCfgScale(parseFloat(e.target.value) || 7.5)}
                />
              </div>
              <div className="img-option">
                <label>Sampler:</label>
                <select
                  value={imgSampler}
                  onChange={(e) => setImgSampler(e.target.value)}
                >
                  <option value="euler_a">Euler a</option>
                  <option value="euler">Euler</option>
                  <option value="ddim">DDIM</option>
                  <option value="dpm2">DPM2</option>
                  <option value="dpm_adaptive">DPM adaptive</option>
                  <option value="lms">LMS</option>
                  <option value="heun">Heun</option>
                  <option value="ddpm">DDPM</option>
                </select>
              </div>
              <div className="img-option">
                <label>Seed:</label>
                <input
                  type="number"
                  value={imgSeed}
                  onChange={(e) => setImgSeed(parseInt(e.target.value) || -1)}
                  placeholder="-1 (random)"
                />
              </div>
            </div>
            <div className="img-negative-prompt">
              <label htmlFor="negative-prompt">Negative Prompt:</label>
              <textarea
                id="negative-prompt"
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                placeholder="What to avoid in the image..."
                rows="2"
              />
            </div>
            {referenceImage && (
              <div className="reference-image-preview">
                <div className="ref-img-header">
                  <span>🖼️ Reference Image:</span>
                  <button onClick={removeReferenceImage} className="remove-ref-btn">✕</button>
                </div>
                <div className="ref-img-content">
                  <img src={referenceImage.preview} alt="Reference" />
                  <div className="ref-img-info">
                    <span className="ref-img-name">{referenceImage.name}</span>
                    <span className="ref-img-size">{formatFileSize(referenceImage.size)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        
        {videoGenerationEnabled && (
          <div className="img-generation-settings">
            <div className="img-backend-selector">
              <label>Video Model:</label>
              <select
                value={videoModel || ''}
                onChange={(e) => {
                  const value = e.target.value || null;
                  setVideoModel(value);
                  localStorage.setItem('video-model', value || '');
                }}
              >
                <option value="">Auto-detect (first video model in Downloads)</option>
                {availableModels
                  .filter(m => {
                    const id = m.id.toLowerCase();
                    return id.includes('video') || id.includes('svd') || id.includes('img2vid');
                  })
                  .map(model => (
                    <option key={model.id} value={model.id}>
                      {model.id}
                    </option>
                  ))
                }
              </select>
              <span className="backend-note">💡 SVD generates short videos from images</span>
            </div>
            
            <div className="img-option-row">
              <div className="img-option">
                <label>Width:</label>
                <input
                  type="number"
                  min="256"
                  max="1024"
                  step="64"
                  value={videoWidth}
                  onChange={(e) => setVideoWidth(parseInt(e.target.value) || 1024)}
                />
              </div>
              <div className="img-option">
                <label>Height:</label>
                <input
                  type="number"
                  min="256"
                  max="1024"
                  step="64"
                  value={videoHeight}
                  onChange={(e) => setVideoHeight(parseInt(e.target.value) || 576)}
                />
              </div>
              <div className="img-option">
                <label>Frames:</label>
                <input
                  type="number"
                  min="14"
                  max="25"
                  step="1"
                  value={videoNumFrames}
                  onChange={(e) => {
                    const val = parseInt(e.target.value) || 25;
                    // Snap to 14 or 25
                    setVideoNumFrames(val < 20 ? 14 : 25);
                  }}
                />
                <span className="help-text">14 or 25</span>
              </div>
              <div className="img-option">
                <label>FPS:</label>
                <input
                  type="number"
                  min="1"
                  max="30"
                  value={videoFps}
                  onChange={(e) => setVideoFps(parseInt(e.target.value) || 7)}
                />
              </div>
              <div className="img-option">
                <label>Motion:</label>
                <input
                  type="number"
                  min="1"
                  max="255"
                  value={videoMotionBucket}
                  onChange={(e) => setVideoMotionBucket(parseInt(e.target.value) || 127)}
                />
                <span className="help-text">1-255 intensity</span>
              </div>
              <div className="img-option">
                <label>Noise:</label>
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={videoNoiseStrength}
                  onChange={(e) => setVideoNoiseStrength(parseFloat(e.target.value) || 0.02)}
                />
              </div>
              <div className="img-option">
                <label>Seed:</label>
                <input
                  type="number"
                  value={videoSeed}
                  onChange={(e) => setVideoSeed(parseInt(e.target.value) || -1)}
                  placeholder="-1 (random)"
                />
              </div>
            </div>
            {referenceImage && (
              <div className="reference-image-preview">
                <div className="ref-img-header">
                  <span>🖼️ Source Image for Video:</span>
                  <button onClick={removeReferenceImage} className="remove-ref-btn">✕</button>
                </div>
                <div className="ref-img-content">
                  <img src={referenceImage.preview} alt="Reference" />
                  <div className="ref-img-info">
                    <span className="ref-img-name">{referenceImage.name}</span>
                    <span className="ref-img-size">{formatFileSize(referenceImage.size)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        
        {attachedFiles.length > 0 && (
          <div className="attached-files">
            <div className="attached-files-header">📎 Attachments:</div>
            <div className="attached-files-list">
              {attachedFiles.map((file, index) => (
                <div key={index} className="attached-file-item">
                  <span className="file-icon">{getFileIcon(file.type)}</span>
                  <div className="file-info">
                    <span className="file-name">{file.name}</span>
                    <span className="file-size">{formatFileSize(file.size)}</span>
                  </div>
                  <button onClick={() => removeAttachment(index)} className="remove-file-btn">✕</button>
                </div>
              ))}
            </div>
          </div>
        )}
        
        <div className="input-row">
          <div className="input-actions">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileAttachment}
              multiple
              accept=".pdf,.doc,.docx,.xls,.xlsx,.txt,.jpg,.jpeg,.png,.tiff,.mp4,.avi,.mov,.mkv,.webm"
              style={{ display: 'none' }}
            />
            <button
              className="attach-btn"
              onClick={() => fileInputRef.current?.click()}
              title="Attach files"
            >
              📎
            </button>
            
            {(imageGenerationEnabled || videoGenerationEnabled) && (
              <>
                <input
                  type="file"
                  ref={refImageInputRef}
                  onChange={handleReferenceImage}
                  accept="image/jpeg,image/jpg,image/png,image/tiff"
                  style={{ display: 'none' }}
                />
                <button
                  className="attach-btn ref-image-btn"
                  onClick={() => refImageInputRef.current?.click()}
                  title={videoGenerationEnabled ? "Add source image for video" : "Add reference image"}
                >
                  🖼️
                </button>
              </>
            )}
          </div>
          
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={
              imageGenerationEnabled ? "Enter your image prompt..." :
              videoGenerationEnabled ? "Optional description (image required for video)..." :
              "Type your message... (Shift+Enter for new line)"
            }
            rows="3"
          />
          <button 
            onClick={handleSend} 
            disabled={
              isGenerating || 
              isSearching || 
              (!videoGenerationEnabled && !input.trim()) || 
              (videoGenerationEnabled && !referenceImage)
            }
          >
            {imageGenerationEnabled ? '🎨' : videoGenerationEnabled ? '🎬' : '➤'}
          </button>
        </div>
      </div>
      </div>
    </div>
  );
}

export default Chat;
