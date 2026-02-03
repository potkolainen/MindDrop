# MindDrop Alpha 0.5.0

<div align="center">
**A flexible local AI workspace built with Tauri, React, and Rust**
[![License: Unlicense](https://img.shields.io/badge/License-Unlicense-blue.svg)](LICENSE)
[![Tauri](https://img.shields.io/badge/Tauri-2.2-blue)](https://tauri.app/)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://reactjs.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange)](https://www.rust-lang.org/)

MindDrop is a local-first, experimental AI hub designed for users who want flexibility, privacy, and control. It brings multiple AI capabilities (chat, search, image/video generation, TTS) into a single desktop app, with a strong focus on extensibility via plugins.
Status: Alpha 0.5.0
This release focuses on fixing core systems and stabilizing existing features. Expect rough edges.

</div>

---

### ✨ What MindDrop Is
- **Local-first AI hub** (no mandatory cloud APIs)
- **Supports multiple AI workflows in one app**
- **Plugin-based and hackable by design**
- **Built for experimentation, not mass-market polish**
---
### **Fixes and changes to 0.5.0 from before**

### Core & UI
* Working quantization (future versions will expose true native precision)
* System stats visible in both menu and chat panels
* Improved chat panel:
  * Collapsible chat-type buttons (text, image, video, etc.)
  * Active chat indicators
  * Scroll support

### Text-to-Speech (TTS)
* Fixed overlapping voices
* TTS no longer auto-plays when opening chats
* Speech triggers only on new incoming messages.

 ## Quantize
* **fp16, 8-bit and 4-bit** - they should now actually work

## Hybrid-mode
* **GPU + CPU** - should work better now and has less issues when dividing load
* 
### Search
* Online search fixed
* Default search engine selectable
* Multi-engine search mode (query multiple engines in parallel)
* Results grouped and labeled by engine

### Diffusers
* Installed diffusers state is now correctly remembered

### Plugins (Early / Alpha)
* Adapter-based plugin system
* Plugins run out-of-process
* JSON-based communication
* Plugins are discovered from:

  ```
  src-tauri/Plugins/
  ```
* Enable / disable plugins via UI
* Example plugin included

---

## ⚠️ Known Issues (Alpha)

* Assistive search is partially broken:
  * CAPTCHA handling (Qwant)
  * Summarize page / send-to-chat buttons
  * Popup close button
* Window size / position memory not working reliably
* Some filters and effects may have little or no effect

---

## 🧭 Roadmap / Future Plans (0.6.0+)

* Fix assistive search completely
* Window state persistence
* Expanded model controls (temperature, top-p, presets, LoRA)
* Improved plugin adapters (HTTP / WebSocket)
* Image & video generation improvements
* Conversation memory & file export/import

---

> MindDrop prioritizes flexibility and local control over polish. Stability will improve over time.

---

## ✨ Features

### 🔌 Plugin System (NEW!)
- **Language-agnostic**: Write plugins in Python, Node.js, or any executable
- **Out-of-process**: Plugins run in separate processes for security
- **JSON protocol**: Simple stdin/stdout communication
- **Auto-discovery**: Drop plugins in `Plugins/` folder
- **Easy creation**: See [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) for guide

### 🎙️ Advanced Text-to-Speech
- **Multiple TTS Engines**: F5-TTS for high-quality synthesis, automatic fallback to edge-tts
- **8 Voice Profiles**: Male (neutral, deep, soft), Female (neutral, warm, bright), Androgynous, Narrator
- **Advanced Controls**: Speed, pitch, prosody, stability, volume, pause timing, breath insertion
- **Audio Post-Processing**: Warmth, presence, air, reverb, de-esser, normalization, limiter
- **Custom Presets**: Create and save your own voice configurations
- **Auto-play**: Optional automatic playback of AI responses (only on new messages)

### 💬 Intelligent Chat Interface
- **Multi-chat Support**: Create and manage multiple chat sessions by type (Text, Image, Video)
- **Collapsible Categories**: Organize chats with expandable sections showing counts
- **Model Selection**: Choose from downloaded AI models
- **Multi-Engine Search**: Query DuckDuckGo, Brave, and Bing simultaneously
- **13 Search Engines**: DuckDuckGo, Google, Brave, Bing, GitHub, Stack Overflow, and more
- **Message Playback**: Click-to-play TTS for any message
- **Auto-stop**: TTS stops automatically when switching chats
- **System Stats**: Real-time resource monitoring in chat sidebar

### 🤖 AI Model Management
- **16+ Model Sources**: Hugging Face, Civitai, GitHub, GitLab, OpenAI, ModelScope, KoboldAI, Papers with Code, Zenodo, arXiv, LM Studio, Ollama, NVIDIA NGC, AWS, Azure, GCP (i noticed Ollama does not actually work. will fix into next update)
- **Easy Downloads**: One-click model downloads with progress tracking (files are big and sometimes it looks like download is stuck. it is not)
- **Smart Execution**: Automatic GPU/CPU detection and optimization
- **Memory Management**: Built-in cleanup to free RAM and VRAM

### 🎨 Image & Video Generation
- **Multiple Backends**: Diffusers, Stable Diffusion WebUI, ComfyUI, InvokeAI
- **Backend Persistence**: System remembers installed backends
- **SDXL Support**: High-quality image generation
- **Video Generation**: Stable Video Diffusion for animations
- **Custom Parameters**: Control steps, guidance, seed, size

### 🖥️ System Optimization
- **Window Memory**: Position, size, and state persistence (BORKED)
- **Resource Monitoring**: Real-time CPU, RAM, and VRAM usage tracking
- **Memory Cleanup**: Automatic model unloading when switching chats
- **Manual Controls**: Force memory cleanup from Settings
- **Execution Modes**: Auto, GPU, Hybrid, CPU modes with smart fallback

---

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ and npm
- **Rust** 1.70+ ([Install Rust](https://rustup.rs/))
- **Python** 3.10+ ([Install Python](https://www.python.org/downloads/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/minddrop.git
   cd minddrop
   ```

2. **Install JavaScript dependencies**
   ```bash
   npm install
   ```

3. **Install Python dependencies**
   ```bash
   # Core dependencies
   pip install edge-tts pydub
   
   # Optional: For higher quality TTS
   pip install f5-tts
   
   # Optional: For image generation
   pip install torch torchvision diffusers transformers accelerate
   
   # Optional: For video generation
   pip install opencv-python
   ```

4. **Run the development server**
   ```bash
   npm run tauri:dev
   ```

The app will launch automatically!

---

## 🔌 Creating Plugins

See [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) for a comprehensive guide on creating plugins.

### Quick Example

1. Create a folder in `Plugins/my-plugin/`
2. Add `plugin.json`:
   ```json
   {
     "name": "My Plugin",
     "version": "1.0.0",
     "type": "python",
     "entry": "main.py",
     "description": "Does something cool"
   }
   ```
3. Add `main.py`:
   ```python
   import sys
   import json
   
   input_data = json.load(sys.stdin)
   result = {"output": "Processed: " + input_data["text"]}
   print(json.dumps(result))
   ```
4. Refresh plugins in the app!

---

## 📁 Project Structure

```
minddrop/
├── src/                          # React frontend
│   ├── components/
│   │   ├── Chat/                # Chat interface
│   │   ├── Models/              # Model management
│   │   ├── Settings/            # App settings
│   │   ├── Plugins/             # Plugin manager (NEW!)
│   │   ├── Diffusers/           # Image/video generation
│   │   ├── SystemStats/         # Resource monitoring
│   │   └── TTSAdvancedSettings/ # TTS configuration
│   ├── App.jsx                  # Main app component
│   └── main.jsx                 # Entry point
├── src-tauri/                   # Rust backend
│   ├── src/
│   │   ├── main.rs             # Entry point
│   │   ├── commands.rs         # Tauri commands
│   │   ├── ai_runner.rs        # Model inference
│   │   ├── system_info.rs      # System monitoring
│   │   └── ...
│   ├── f5tts_generate.py       # TTS generation
│   ├── generate_image.py       # Image generation
│   └── generate_video.py       # Video generation
├── Plugins/                     # Plugin directory (NEW!)
│   └── example-text-transform/ # Example plugin
├── PLUGIN_DEVELOPMENT.md        # Plugin creation guide (NEW!)
└── package.json                # NPM dependencies
```

---

## 🛠️ Configuration

### Python Environment
- Set your Python path in Settings → Python Environment
- The app will use this for all Python-based features (TTS, image generation, plugins)

### Memory Management
- **Auto Cleanup**: Enabled by default, frees memory when switching chats
- **Manual Cleanup**: Click "Clear Memory" in Settings to force cleanup

### Search Engines
- **Default Engine**: Choose from 13 search engines
- **Multi-Engine**: Enable to query multiple engines simultaneously
- **Enabled Engines**: Select which engines to include in multi-engine mode

### Plugins
- **Auto-scan**: Plugins are discovered automatically from `Plugins/` folder
- **Enable/Disable**: Toggle plugins individually
- **Execution**: Only enabled plugins can execute

---

## 🔧 Development

### Build for Production
```bash
npm run tauri:build
```

### Run Tests
```bash
# Frontend
npm test

# Backend
cd src-tauri && cargo test
```

### Code Structure
- **Frontend**: React with hooks, Tauri API integration
- **Backend**: Rust with async/await, process spawning for Python scripts
- **IPC**: Tauri commands for frontend-backend communication

---

## 📋 Requirements

### Minimum
- **OS**: Linux, Windows 10+, macOS 10.15+
- **CPU**: 4 cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space

### Recommended for AI Models
- **GPU**: NVIDIA GPU with 8GB+ VRAM (CUDA support)
- **RAM**: 16GB+ for large language models
- **Storage**: 50GB+ for model storage

---

## 🐛 Known Issues

### General
- First-time model downloads may be slow depending on internet speed
- Large models (>10GB) require significant RAM

### TTS
- F5-TTS requires GPU for real-time generation
- Edge-TTS requires internet connection (fallback mode)

### Image Generation
- SDXL models require 6GB+ VRAM
- First generation is slower due to model loading

### Plugins
- Python and Node.js must be in PATH for respective plugin types
- Plugin execution errors appear in app console

---

## 📄 License

This project is released into the **public domain** under the [Unlicense](LICENSE).

You are free to use, modify, distribute this software without any restrictions.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

### Areas for Contribution
- Plugin creation and examples
- Model integration and testing
- UI/UX improvements
- Documentation
- Bug fixes and optimizations
- Reddit https://www.reddit.com/r/MindDrop/s/ApG5GhabIO

---

## 🙏 Acknowledgments

- **Tauri** - Lightweight desktop framework
- **F5-TTS** - High-quality text-to-speech
- **Edge-TTS** - Microsoft's TTS service
- **Hugging Face** - Model hosting and transformers library
- **Stable Diffusion** - Image generation models
- **Rust Community** - Amazing language and ecosystem
- **React** - Frontend framework
- **vibe code involved**

---

<div align="center">


[⭐ Star on GitHub] • [🐛 Report Bug] • [💡 Request Feature]

</div>
