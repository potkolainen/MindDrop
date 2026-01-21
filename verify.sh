#!/bin/bash

# MindDrop v0.5.0 - Verification Script
# Checks if all prerequisites are installed

echo "🔍 MindDrop v0.5.0 - Checking Prerequisites..."
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

all_good=true

# Check Node.js
echo -n "Checking Node.js... "
if command -v node &> /dev/null; then
    version=$(node --version)
    echo -e "${GREEN}✓${NC} $version"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "  Install from: https://nodejs.org/"
    all_good=false
fi

# Check npm
echo -n "Checking npm... "
if command -v npm &> /dev/null; then
    version=$(npm --version)
    echo -e "${GREEN}✓${NC} v$version"
else
    echo -e "${RED}✗ Not found${NC}"
    all_good=false
fi

# Check Rust
echo -n "Checking Rust... "
if command -v rustc &> /dev/null; then
    version=$(rustc --version | awk '{print $2}')
    echo -e "${GREEN}✓${NC} v$version"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "  Install from: https://rustup.rs/"
    all_good=false
fi

# Check Cargo
echo -n "Checking Cargo... "
if command -v cargo &> /dev/null; then
    version=$(cargo --version | awk '{print $2}')
    echo -e "${GREEN}✓${NC} v$version"
else
    echo -e "${RED}✗ Not found${NC}"
    all_good=false
fi

# Check Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    version=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}✓${NC} v$version"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "  Install from: https://www.python.org/downloads/"
    all_good=false
fi

# Check pip
echo -n "Checking pip... "
if command -v pip3 &> /dev/null; then
    version=$(pip3 --version | awk '{print $2}')
    echo -e "${GREEN}✓${NC} v$version"
else
    echo -e "${RED}✗ Not found${NC}"
    all_good=false
fi

echo ""
echo "📦 Checking Optional Dependencies..."
echo ""

# Check edge-tts
echo -n "Checking edge-tts (TTS)... "
if python3 -c "import edge_tts" &> /dev/null; then
    echo -e "${GREEN}✓${NC} Installed"
else
    echo -e "${YELLOW}○ Not installed${NC}"
    echo "  Install: pip install edge-tts"
fi

# Check pydub
echo -n "Checking pydub (Audio)... "
if python3 -c "import pydub" &> /dev/null; then
    echo -e "${GREEN}✓${NC} Installed"
else
    echo -e "${YELLOW}○ Not installed${NC}"
    echo "  Install: pip install pydub"
fi

# Check f5-tts
echo -n "Checking f5-tts (High-quality TTS)... "
if python3 -c "import f5_tts" &> /dev/null; then
    echo -e "${GREEN}✓${NC} Installed"
else
    echo -e "${YELLOW}○ Not installed (optional)${NC}"
    echo "  Install: pip install f5-tts"
fi

# Check torch
echo -n "Checking PyTorch (Image gen)... "
if python3 -c "import torch" &> /dev/null; then
    echo -e "${GREEN}✓${NC} Installed"
else
    echo -e "${YELLOW}○ Not installed (optional)${NC}"
    echo "  Install: pip install torch"
fi

# Check diffusers
echo -n "Checking diffusers (Image gen)... "
if python3 -c "import diffusers" &> /dev/null; then
    echo -e "${GREEN}✓${NC} Installed"
else
    echo -e "${YELLOW}○ Not installed (optional)${NC}"
    echo "  Install: pip install diffusers"
fi

echo ""
echo "📁 Checking Project Structure..."
echo ""

# Check key files
files=("package.json" "src/App.jsx" "src-tauri/Cargo.toml" "src-tauri/src/main.rs")
for file in "${files[@]}"; do
    echo -n "Checking $file... "
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ Missing${NC}"
        all_good=false
    fi
done

echo ""
if [ "$all_good" = true ]; then
    echo -e "${GREEN}✓ All required prerequisites are installed!${NC}"
    echo ""
    echo "🚀 You're ready to run MindDrop!"
    echo "   Run: npm run tauri:dev"
else
    echo -e "${RED}✗ Some required prerequisites are missing.${NC}"
    echo "   Please install the missing items and try again."
    exit 1
fi
