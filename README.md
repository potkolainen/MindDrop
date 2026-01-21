# MindDrop v0.5.0

<div align="center">

**A powerful local AI workspace built with Tauri, React, and Rust**

[![License: Unlicense](https://img.shields.io/badge/License-Unlicense-blue.svg)](LICENSE)
[![Tauri](https://img.shields.io/badge/Tauri-2.2-blue)](https://tauri.app/)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://reactjs.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange)](https://www.rust-lang.org/)

Run AI models, generate speech, images, and chat with AI—all locally on your machine. No cloud, no API keys required for core features.

**Version 0.5.0 "Alpha Release"**

</div>

---

## 🆕 What's New in v0.5.0

### 🔌 Plugin System
- **Adapter-based architecture** - Use existing CLI tools without modification
- **Multi-language support** - Python, Node.js, and binary executables
- **Auto-discovery** - Automatically scans `Plugins/` folder
- **Enable/disable toggles** - Manage plugins individually
- **JSON communication** - Simple stdin/stdout protocol
- **Example plugin included** - Text transformer demonstrating the system
- **Comprehensive documentation** - See [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md)

### 🪟 Window Memory
- **Position persistence** - Window remembers its location on screen
- **Size persistence** - Window dimensions saved between sessions
- **State restoration** - Maximized/fullscreen state preserved
- **Smooth UX** - Opens exactly where you left it

### 🎨 UI Improvements
- **Collapsible chat categories** - Text, Image, Video chats organized by type
- **Collapsible settings** - Clean, organized settings interface
- **Multi-engine search** - Query 3+ search engines simultaneously
- **System stats in chat** - Real-time resource monitoring in sidebar
- **Enhanced navigation** - Cleaner, more intuitive interface

### 🎙️ TTS Enhancements
- **Auto-play fix** - TTS only starts on new messages, not when opening chat
- **All settings verified** - Every advanced setting is functional
- **Better control** - Improved responsiveness and reliability

### 🖼️ Diffuser Backend Persistence
- **Installation tracking** - System remembers installed backends
- **No re-installs** - Backends stay marked as installed

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
- **10+ Built-in Presets**: Natural Conversation, Podcast Narrator, Warm Audiobook, Cold AI Voice, Fast Utility, and more
- **Custom Presets**: Create and save your own voice configurations
- **Auto-play**: Optional automatic playback of AI responses (only on new messages)

### 💬 Intelligent Chat Interface
- **Multi-chat Support**: Create and manage multiple chat sessions by type (Text, Image, Video)
- **Collapsible Categories**: Organize chats with expandable sections showing counts
- **Model Selection**: Choose from downloaded AI models
- **Multi-Engine Search**: Query DuckDuckGo, Brave, and Bing simultaneously
- **13 Search Engines**: DuckDuckGo, Google, Brave, Bing, GitHub, Stack Overflow, Reddit, and more
- **Message Playback**: Click-to-play TTS for any message
- **Auto-stop**: TTS stops automatically when switching chats
- **System Stats**: Real-time resource monitoring in chat sidebar

### 🤖 AI Model Management
- **16+ Model Sources**: Hugging Face, Civitai, GitHub, GitLab, OpenAI, ModelScope, KoboldAI, Papers with Code, Zenodo, arXiv, LM Studio, Ollama, NVIDIA NGC, AWS, Azure, GCP
- **Easy Downloads**: One-click model downloads with progress tracking
- **Smart Execution**: Automatic GPU/CPU detection and optimization
- **Memory Management**: Built-in cleanup to free RAM and VRAM

### 🎨 Image & Video Generation
- **Multiple Backends**: Diffusers, Stable Diffusion WebUI, ComfyUI, InvokeAI
- **Backend Persistence**: System remembers installed backends
- **SDXL Support**: High-quality image generation
- **Video Generation**: Stable Video Diffusion for animations
- **Custom Parameters**: Control steps, guidance, seed, size

### 🖥️ System Optimization
- **Window Memory**: Position, size, and state persistence
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
- **GPU**: NVIDIA GPU with 6GB+ VRAM (CUDA support)
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

## 🗺️ Roadmap

### v0.6.0 (Planned)
- [ ] Remote control with password protection
- [ ] Model parameter controls (temperature, top-p, top-k, etc.)
- [ ] LoRA support for models
- [ ] Batch image generation
- [ ] Chat performance optimization

### v0.7.0 (Planned)
- [ ] HTTP/REST plugin adapter
- [ ] Plugin marketplace
- [ ] Enhanced security sandboxing
- [ ] Preset system for model configurations

### Future
- [ ] Model quantization UI (FP16, 8-bit, 4-bit)
- [ ] Context length slider
- [ ] Auto-trim old messages
- [ ] WebSocket adapter for real-time plugins
- [ ] Plugin composition (chaining)

---

## 📄 License

This project is released into the **public domain** under the [Unlicense](LICENSE).

You are free to use, modify, distribute, and sell this software without any restrictions.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

### Areas for Contribution
- Plugin creation and examples
- Model integration and testing
- UI/UX improvements
- Documentation
- Bug fixes and optimizations

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/minddrop/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/minddrop/discussions)
- **Email**: your.email@example.com

---

## 🙏 Acknowledgments

- **Tauri** - Lightweight desktop framework
- **F5-TTS** - High-quality text-to-speech
- **Edge-TTS** - Microsoft's TTS service
- **Hugging Face** - Model hosting and transformers library
- **Stable Diffusion** - Image generation models
- **Rust Community** - Amazing language and ecosystem
- **React** - Frontend framework

---

<div align="center">

**Made with ❤️ by the MindDrop Team**

[⭐ Star on GitHub](https://github.com/yourusername/minddrop) • [🐛 Report Bug](https://github.com/yourusername/minddrop/issues) • [💡 Request Feature](https://github.com/yourusername/minddrop/issues)

</div>
