# MindDrop v0.5.0 Installation Guide

## 📋 Prerequisites

Before installing MindDrop, ensure you have the following installed:

### Required
1. **Node.js 18+** and npm
   - Download: https://nodejs.org/
   - Verify: `node --version` and `npm --version`

2. **Rust 1.70+**
   - Install: https://rustup.rs/
   - Verify: `rustc --version`

3. **Python 3.10+**
   - Download: https://www.python.org/downloads/
   - Verify: `python3 --version`

### Recommended
- **NVIDIA GPU** with CUDA support (for AI models)
- **16GB+ RAM** (for large language models)
- **50GB+ free disk space** (for model storage)

---

## 🚀 Installation Steps

### 1. Extract the Release
```bash
# Extract the MindDrop 050 folder
cd MindDrop\ 050
```

### 2. Install JavaScript Dependencies
```bash
npm install
```

This will install all required frontend and build dependencies.

### 3. Install Python Dependencies

#### Core Dependencies (Required)
```bash
# For text-to-speech
pip install edge-tts pydub
```

#### Optional Dependencies

**High-Quality TTS:**
```bash
pip install f5-tts
```

**Image Generation:**
```bash
pip install torch torchvision diffusers transformers accelerate
```

**Video Generation:**
```bash
pip install opencv-python
```

**All Optional (for full features):**
```bash
pip install f5-tts torch torchvision diffusers transformers accelerate opencv-python
```

### 4. Run MindDrop

**Development Mode (with hot reload):**
```bash
npm run tauri:dev
```

**Build for Production:**
```bash
npm run tauri:build
```

The production build will be in `src-tauri/target/release/`.

---

## 🔧 Configuration

### First Launch
1. **Set Python Path**: Go to Settings → Python Environment and select your Python interpreter
2. **Download Models**: Visit the Library tab to browse and download AI models
3. **Configure TTS**: Settings → TTS to adjust voice and audio settings
4. **Set Download Directory**: Settings → Download Directory to choose where models are stored

### System Requirements Check
- Click on the system stats in the chat sidebar to view:
  - CPU usage
  - RAM usage
  - VRAM usage (if GPU available)
  - Maximum model capacity

---

## 🔌 Setting Up Plugins

### Quick Start
1. The `Plugins/` folder is created automatically
2. Example plugin is included: `Plugins/example-text-transform/`
3. Go to the Plugins tab to see discovered plugins
4. Toggle plugins on/off as needed

### Creating Your Own Plugin
See `PLUGIN_DEVELOPMENT.md` for a complete guide.

**Quick Example:**
```bash
# Create plugin folder
mkdir -p Plugins/my-plugin

# Create manifest
cat > Plugins/my-plugin/plugin.json << 'EOF'
{
  "name": "My Plugin",
  "version": "1.0.0",
  "type": "python",
  "entry": "main.py",
  "description": "My custom plugin"
}
EOF

# Create plugin script
cat > Plugins/my-plugin/main.py << 'EOF'
#!/usr/bin/env python3
import sys
import json

input_data = json.load(sys.stdin)
result = {"output": "Processed: " + input_data.get("text", "")}
print(json.dumps(result))
EOF

# Make executable
chmod +x Plugins/my-plugin/main.py
```

Refresh the Plugins tab to see your new plugin!

---

## 🐛 Troubleshooting

### Build Errors

**"Rust not found"**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Restart terminal
```

**"Python not found"**
```bash
# Ensure Python is in PATH
which python3
# Or set Python path in Settings
```

**"Cannot find module"**
```bash
# Reinstall npm dependencies
rm -rf node_modules package-lock.json
npm install
```

### Runtime Errors

**"TTS not working"**
```bash
# Install TTS dependencies
pip install edge-tts pydub
# Check Python path in Settings
```

**"Models won't load"**
- Check you have enough RAM/VRAM
- Try a smaller model first
- Check download directory permissions

**"Plugins not discovered"**
- Ensure `Plugins/` folder exists
- Check `plugin.json` syntax (valid JSON)
- Click "Refresh" in Plugins tab
- Check console for errors

### Performance Issues

**Slow generation:**
- Use smaller models
- Enable GPU mode (Settings → Execution Mode → GPU)
- Close other applications
- Increase system resources

**High memory usage:**
- Enable auto cleanup (Settings → Memory Management)
- Click "Clear Memory" manually
- Use quantized models (8-bit or 4-bit)

---

## 📦 Optional: FFmpeg for Advanced TTS

For best audio quality and post-processing:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

---

## 🔒 Security Notes

### Plugins
- Only install plugins from trusted sources
- Review plugin code before enabling
- Plugins run with same permissions as the app
- Consider using virtual environments

### API Keys
- API keys are stored locally (not sent anywhere)
- Keys are stored in localStorage (browser storage)
- Clear browser data to remove stored keys

---

## 📚 Next Steps

1. **Read the README.md** for feature overview
2. **Check PLUGIN_DEVELOPMENT.md** to create plugins
3. **Explore the UI** - all tabs have different features
4. **Download models** - start with smaller models first
5. **Configure TTS** - try different voices and presets

---

## 💡 Tips

### For Best Performance
- Use GPU mode for large models
- Enable auto memory cleanup
- Download models to SSD (not HDD)
- Close unnecessary applications

### For Plugin Development
- Test plugins manually first (`echo '{}' | python3 plugin.py`)
- Use `console.log` to debug in app
- Check terminal output for errors
- Start with the example plugin

### For TTS Quality
- Use F5-TTS for best quality (requires GPU)
- Adjust post-processing settings
- Create custom presets for different uses
- Edge-TTS works offline after first download

---

## ❓ Getting Help

- **Issues**: Check GitHub Issues
- **Docs**: Read README.md and PLUGIN_DEVELOPMENT.md
- **Logs**: Check browser console (F12) for errors
- **Community**: Join discussions on GitHub

---

**Happy AI generating!** 🚀
