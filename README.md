# MindDrop

<div align="center">

**A powerful local AI workspace built with Tauri, React, and Rust**

[![License: Unlicense](https://img.shields.io/badge/License-Unlicense-blue.svg)](LICENSE)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-blue)](https://tauri.app/)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://reactjs.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange)](https://www.rust-lang.org/)

Run AI models, generate speech, and chat with AI—all locally on your machine. No cloud, no API keys required for core features.

</div>

---

## ✨ Features

### 🎙️ Advanced Text-to-Speech
- **Multiple TTS Engines**: F5-TTS for high-quality synthesis, with automatic fallback to edge-tts
- **8 Voice Profiles**: Male (neutral, deep, soft), Female (neutral, warm, bright), Androgynous, Narrator
- **Advanced Controls**: Speed, pitch, prosody, stability, volume, pause timing, breath insertion
- **Audio Post-Processing**: Warmth, presence, air, reverb, de-esser, normalization, limiter
- **5 Built-in Presets**: Natural Conversation, Podcast Narrator, Warm Audiobook, Cold AI Voice, Fast Utility
- **Custom Presets**: Create and save your own voice configurations
- **Auto-play**: Optional automatic playback of AI responses

### 💬 Intelligent Chat Interface
- **Multi-chat Support**: Create and manage multiple chat sessions
- **Model Selection**: Choose from downloaded AI models
- **Online Search Integration**: 13 search engines including DuckDuckGo, Google, Brave, GitHub, Stack Overflow
- **Message Playback**: Click-to-play TTS for any message
- **Auto-stop**: TTS stops automatically when switching chats

### 🤖 AI Model Management
- **16+ Model Sources**: Browse and download from:
  - **Hugging Face** - Largest open-source ML model hub
  - **Civitai** - Community-driven AI art models
  - **GitHub** - Code repositories with model releases
  - **GitLab** - Alternative Git hosting with models
  - **OpenAI** - Official OpenAI model access
  - **ModelScope** - Chinese AI model platform
  - **KoboldAI** - Text generation model repository
  - **Papers with Code** - Research paper implementations
  - **Zenodo** - Scientific research repository
  - **arXiv** - Academic preprint models
  - **LM Studio** - Local language model hub
  - **Ollama** - Optimized local LLM runner
  - **NVIDIA NGC** - NVIDIA GPU-optimized models
  - **AWS** - Amazon cloud AI models
  - **Azure** - Microsoft cloud AI models
  - **GCP** - Google Cloud AI models
- **Easy Downloads**: One-click model downloads with progress tracking
- **Smart Execution**: Automatic GPU/CPU detection and optimization
- **Memory Management**: Built-in cleanup to free RAM and VRAM

### 🔧 System Optimization
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
   pip install edge-tts pydub
   # Optional: For higher quality TTS
   pip install f5-tts
   ```

4. **Run the development server**
   ```bash
   npm run tauri:dev
   ```

The app will launch automatically!

---

## 🎯 Usage

### Basic Chat
1. Open the **Library** tab to download AI models
2. Switch to **Chat** tab and select a model
3. Start chatting! Enable TTS autoplay in Settings for voice responses

### Text-to-Speech
- **Quick Test**: Go to Settings → TTS section → Preview Voice
- **Advanced Settings**: Click "Advanced Settings" for presets and fine-tuning
- **In Chat**: Click the speaker icon (🔊) on any message to hear it

### Memory Management
- Models are automatically cleaned up when switching chats
- Manual cleanup: Settings → Memory Management → "Free Memory"

---

## 🏗️ Architecture

```
MindDrop
├── Frontend (React + Vite)
│   ├── Chat interface with TTS controls
│   ├── Model manager UI
│   └── Settings and system stats
│
├── Backend (Rust + Tauri)
│   ├── AI inference runner
│   ├── System profiling
│   └── Model management
│
└── Python Bridges
    ├── F5-TTS synthesis
    ├── edge-tts fallback
    └── Audio post-processing (pydub)
```

### Tech Stack
- **Frontend**: React, Vite, CSS Modules
- **Backend**: Rust, Tauri 2.0
- **Audio**: Python (F5-TTS, edge-tts, pydub)
- **AI**: Transformers, PyTorch

---

## 🎨 Voice Profiles

| Profile | Description | Use Case |
|---------|-------------|----------|
| male_neutral | Balanced male voice | General purpose |
| male_deep | Lower, authoritative | Narration, announcements |
| male_soft | Gentle, friendly | Conversations |
| female_neutral | Balanced female voice | General purpose |
| female_warm | Warmer, expressive | Audiobooks, stories |
| female_bright | Clear, energetic | Podcasts, tutorials |
| androgynous | Gender-neutral | Accessibility, preference |
| narrator | Professional narration | Long-form content |

---

## ⚙️ Configuration

### TTS Settings
- **Autoplay**: Enable/disable automatic message playback
- **Voice**: Select from 8 voice profiles
- **Speed**: 0.5x - 2.0x playback speed
- **Pitch**: 0.5 - 2.0 pitch adjustment
- **Volume**: 0% - 100%
- **Advanced**: Prosody, stability, warmth, presence, reverb, and more

### Search Engines
DuckDuckGo • Google • Brave Search • Bing • Qwant • Google Scholar • Stack Overflow • Ecosia • GitHub • DevDocs • arXiv • Startpage • MetaGer

### Environment Variables
- `TTS_FORCE_EDGE=1`: Force edge-tts instead of F5-TTS (lower memory usage)
- `CUDA_VISIBLE_DEVICES=`: Force CPU-only TTS

---

## 📊 System Requirements

### Minimum
- **OS**: Linux, macOS, Windows
- **RAM**: 4 GB
- **Storage**: 2 GB for app + models

### Recommended (for local AI models)
- **RAM**: 16 GB+
- **VRAM**: 8 GB+ (NVIDIA/AMD GPU)
- **Storage**: 50 GB+ for multiple models

---

## 🛠️ Development

### Project Structure
```
├── src/                    # React frontend
│   ├── components/         # UI components
│   └── App.jsx            # Main app
├── src-tauri/             # Rust backend
│   ├── src/
│   │   ├── commands.rs    # Tauri commands
│   │   ├── ai_runner.rs   # AI inference
│   │   └── main.rs        # Entry point
│   └── f5tts_generate.py  # TTS Python bridge
└── package.json           # npm config
```

### Build for Production
```bash
npm run tauri:build
```

---

## 🗺️ Roadmap

- [x] Advanced TTS with F5-TTS and edge-tts
- [x] Multi-chat interface
- [x] Model management and downloads
- [x] Memory optimization and cleanup
- [x] Search engine integration
- [X] Image generation (Stable Diffusion)
- [X] Video generation
- [ ] 3D model generation
- [ ] Code-specific AI models
- [ ] Multimodal vision-language models
- [/] Agent mode expansion
- [ ] Plugin system

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is released into the public domain under [The Unlicense](LICENSE). You are free to use, modify, and distribute this software without any restrictions.

---

## 🙏 Acknowledgments

- [Tauri](https://tauri.app/) - Desktop app framework
- [F5-TTS](https://github.com/SWivid/F5-TTS) - High-quality TTS
- [edge-tts](https://github.com/rany2/edge-tts) - Microsoft Edge TTS
- [Transformers](https://huggingface.co/docs/transformers/) - AI model library

---

<div align="center">

**Made with ❤️ for privacy-focused AI enthusiasts**

[Report Bug](https://github.com/yourusername/minddrop/issues) • [Request Feature](https://github.com/yourusername/minddrop/issues)

</div>

