# Voice UI Quick Start

Get the EchoSee Voice UI running in minutes!

## Prerequisites

- Python 3.8+ with pip
- Node.js 18+ with npm
- FFmpeg (for audio conversion)
- Microphone access
- OpenAI API key

### Installing FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

## Setup (First Time Only)

1. **Install Python package in editable mode:**
```bash
pip install -e .
```

2. **Install Node.js dependencies:**
```bash
cd echonoti
npm install
cd ..
```

3. **Configure environment variables:**
Make sure `.env` file exists in project root with:
```
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

## Running the Voice UI

Simply run the launch script:

```bash
./scripts/launch_voice_ui.sh
```

This will start:
- Voice API server on http://localhost:9004
- Web interface on http://localhost:9003

**Open your browser to: http://localhost:9003/voice**

## Using the Interface

1. Click the purple microphone button
2. Speak your query (button turns red while recording)
3. Click again to stop recording
4. Watch as your speech is transcribed and processed
5. Listen to the AI response with synchronized text highlighting

## Stopping

Press `Ctrl+C` in the terminal where the launch script is running.

## Troubleshooting

**Import errors when starting server:**
```bash
# Make sure you installed in editable mode
pip install -e .
```

**"No module named 'langgraph'":**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**Microphone not working:**
- Grant browser microphone permissions when prompted
- Check that no other app is using the microphone

**Port already in use:**
- Stop any other instances of the servers
- Change ports in scripts/launch_voice_ui.sh if needed

## Features

- Real-time sound wave visualization
- Word-by-word synchronized text display
- Full agent capabilities (search, alarms, notifications, etc.)
- Minimal, modern UI consistent with echonoti

Enjoy talking to EchoSee!
