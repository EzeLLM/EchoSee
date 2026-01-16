# EchoSee Installation Guide

Complete installation instructions for the EchoSee voice assistant system.

## System Requirements

- Python 3.8 or higher
- Node.js 18 or higher
- FFmpeg (for audio conversion)
- Git

## Step 1: System Dependencies

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    nodejs \
    npm
```

### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python3 ffmpeg portaudio node
```

### Windows
1. Install Python 3.8+ from https://www.python.org/downloads/
2. Install Node.js from https://nodejs.org/
3. Install FFmpeg:
   - Download from https://ffmpeg.org/download.html
   - Extract and add to PATH
4. Install Visual C++ Build Tools for PyAudio

## Step 2: Clone Repository

```bash
git clone <repository-url>
cd EchoSee
```

## Step 3: Python Environment Setup

### Option A: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Option B: Using Conda
```bash
# Create conda environment
conda create -n echosee python=3.10
conda activate echosee

# Install Python dependencies
pip install -r requirements.txt
pip install -e .
```

## Step 4: Node.js Setup

```bash
cd echonoti
npm install
cd ..
```

## Step 5: Environment Configuration

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional (if using DeepSeek)
DEEPSEEK_API_KEY=your_deepseek_key_here

# Optional (for development/debugging)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=echosee
```

### Getting API Keys

1. **OpenAI API Key**: https://platform.openai.com/api-keys
2. **Tavily API Key**: https://tavily.com/
3. **DeepSeek API Key** (optional): https://platform.deepseek.com/
4. **LangSmith** (optional, for debugging): https://smith.langchain.com/

## Step 6: Configuration

Edit `config.yml` to customize:

```yaml
LLM:
  model: gpt-4o-mini  # or your preferred model
  provider: openai    # or deepseek

TTS:
  method: openai      # or kokoro for lightweight option
  voice: shimmer

STT:
  model: gpt-4o-mini-transcribe
```

## Step 7: Verify Installation

```bash
# Test Python installation
python -c "from agent_management.agent_manager import AgentManager; print('✓ Python setup OK')"

# Test FFmpeg
ffmpeg -version | head -1

# Test Node.js setup
cd echonoti && npm run typecheck && cd ..
```

## Step 8: Run Health Check

```bash
python -m app_manager.manager health
```

Should output:
```json
{
  "status": "healthy",
  "components": { ... }
}
```

## Troubleshooting

### PyAudio Installation Issues

**Linux:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

### FFmpeg Not Found

Ensure FFmpeg is in your PATH:
```bash
# Test
ffmpeg -version

# If not found, add to PATH or install properly
```

### ModuleNotFoundError

Make sure you installed in editable mode:
```bash
pip install -e .
```

### Permission Errors

On Linux, you may need to add your user to the `audio` group:
```bash
sudo usermod -a -G audio $USER
# Then log out and back in
```

## Quick Start After Installation

### Run Command-Line Voice Assistant
```bash
python -m app_manager.manager
```

### Run Voice UI (Web Interface)
```bash
./scripts/launch_voice_ui.sh
```
Then open http://localhost:9003/voice

### Run Echonoti PWA
```bash
cd echonoti
npm run dev
```
Then open http://localhost:9003

## Updating

```bash
# Pull latest changes
git pull

# Update Python dependencies
pip install -r requirements.txt --upgrade

# Update Node.js dependencies
cd echonoti
npm install
cd ..
```

## Uninstall

```bash
# Remove virtual environment
rm -rf venv

# Remove node_modules
rm -rf echonoti/node_modules

# Remove data files (optional)
rm -rf data/
```

## Development Setup (Optional)

For development with LangSmith tracing:

```bash
# Setup development environment
source setup_dev.sh  # or . setup_dev.sh

# This exports:
# - LANGSMITH_TRACING=true
# - LANGSMITH_API_KEY
# - LANGSMITH_PROJECT
```

## Platform-Specific Notes

### Raspberry Pi
- Use Kokoro TTS instead of OpenAI TTS for better performance
- Adjust `config.yml`:
  ```yaml
  TTS:
    method: kokoro
    lang_code: a
    voice: af_heart
  ```

### Docker (Advanced)
A Dockerfile is not provided, but you can create one based on:
- Base image: `python:3.10-slim`
- Install system dependencies
- Copy files and install Python/Node packages
- Expose ports 9003, 9004

## Getting Help

- Check logs in `logs/` directory
- Run health check: `python -m app_manager.manager health`
- Review `CLAUDE.md` for architecture details
- Check `VOICE_UI_FEATURES.md` for UI documentation

## Next Steps

1. Configure your preferences in `config.yml`
2. Test the CLI: `python -m app_manager.manager`
3. Try the Voice UI: `./scripts/launch_voice_ui.sh`
4. Read the documentation in project root
