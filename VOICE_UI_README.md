# Voice UI for EchoSee

A web-based voice interface for the EchoSee voice assistant, built with Next.js and integrated with the Python backend.

## Features

- **Voice Recording**: Click-to-record interface with real-time audio level visualization
- **Sound Wave Visualization**: Animated sound waves during recording
- **Speech-to-Text**: Transcribes your voice using OpenAI Whisper
- **AI Processing**: Processes requests through the LangGraph agent with full tool access
- **Text-to-Speech**: Generates spoken responses using OpenAI TTS
- **Synchronized Text Display**: Shows response text synchronized with audio playback
- **Minimal Design**: Consistent with echonoti's design system

## Architecture

### Backend (Flask API Server)

Located in `api/voice_server.py`

**Endpoints:**
- `GET /health` - Health check
- `POST /api/voice/process` - Main voice processing endpoint
  - Accepts: multipart/form-data with audio file
  - Returns: JSON with transcription, response text, audio (base64), and timestamps

**Flow:**
1. Receives WebM audio file from frontend
2. Converts to MP3 using FFmpeg (for OpenAI compatibility)
3. Transcribes using STT service (OpenAI Whisper)
4. Processes through AgentManager (LangGraph React agent)
5. Generates TTS audio response (OpenAI TTS)
6. Creates approximate timestamps for text synchronization
7. Returns JSON with transcription, response, audio (base64), and timestamps

### Frontend (Next.js Page)

Located in `echonoti/src/app/voice/page.tsx`

**Components:**
- Recording button with state management
- Real-time audio level monitoring using Web Audio API
- MediaRecorder API for audio capture
- Synchronized text display with word-level highlighting
- Loading states and error handling

## Installation

### Prerequisites

- Python 3.8+
- Node.js 18+
- FFmpeg (for audio conversion)
- Microphone access

**Install FFmpeg:**
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from https://ffmpeg.org/download.html

### Setup

1. Install Python dependencies:
```bash
pip install -e .
```

2. Install Node.js dependencies:
```bash
cd echonoti
npm install
```

## Running

### Option 1: Using the Launch Script

```bash
./scripts/launch_voice_ui.sh
```

This starts both servers:
- Voice API server on http://localhost:9004
- Next.js frontend on http://localhost:9003

Then navigate to http://localhost:9003/voice

### Option 2: Manual Launch

**Terminal 1 - Start Voice API Server:**
```bash
python -m api.voice_server
```

**Terminal 2 - Start Next.js Frontend:**
```bash
cd echonoti
npm run dev
```

Then navigate to http://localhost:9003/voice

## Usage

1. Click the microphone button to start recording
2. Speak your query
3. Click the button again to stop recording
4. Wait while the system processes your request
5. Listen to the response and watch the text highlight in sync with the audio

## Configuration

The voice UI uses the same configuration as the main EchoSee application:
- `config.yml` - LLM, TTS, STT, and agent settings
- `.env` - API keys (OPENAI_API_KEY, etc.)

## Browser Requirements

- Modern browser with support for:
  - MediaRecorder API
  - Web Audio API
  - getUserMedia for microphone access
- Microphone permissions must be granted

## Troubleshooting

**"Could not access microphone"**
- Check browser microphone permissions
- Ensure no other application is using the microphone

**"Audio conversion failed" or "Audio file might be corrupted"**
- Ensure FFmpeg is installed: `ffmpeg -version`
- FFmpeg must be in your system PATH
- On Linux: `sudo apt-get install ffmpeg`

**"Failed to process audio"**
- Ensure the voice API server is running on port 9004
- Check server logs for errors
- Verify all Python dependencies are installed: `pip install -e .`
- Check that app_context initializes properly

**No audio playback**
- Check browser audio permissions
- Ensure system audio is not muted
- Check browser console for errors

**Import errors (ModuleNotFoundError)**
- Install the package in editable mode: `pip install -e .`
- Make sure you're in the project root directory
- Verify all dependencies: `pip install -r requirements.txt`

## Future Enhancements

- Real word-level timestamps from OpenAI (when available)
- Streaming text generation
- Voice activity detection for auto-start/stop
- Multiple language support
- Custom wake word detection
