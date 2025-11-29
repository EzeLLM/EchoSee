# EchoSee: Intelligent Voice Assistant

![Welcome Image](assets/welcome.png)

EchoSee is a cutting-edge voice assistant pipeline designed to replace traditional rule-based assistants (such as Amazon's Alexa or Google Home) with an engaging, flexible, and intelligent conversational AI. Built with modern architecture patterns and streaming capabilities, EchoSee delivers natural, low-latency voice interactions.

## ✨ Latest Features

- **🎵 TTS Streaming**: Audio starts playing within ~500ms as the LLM generates text (sentence-by-sentence buffering)
- **💾 Conversation Persistence**: Full conversation history stored in SQLite with search capabilities
- **🏥 Health Monitoring**: Built-in health checks for all components and CLI diagnostics
- **🔧 Modular Architecture**: Clean separation of concerns with plugin-based audio providers
- **📱 Echonoti Integration**: Companion PWA for cross-device notifications

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Key Components](#key-components)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [Configuration](#configuration)
7. [Development Guidelines](#development-guidelines)
8. [Recent Changes](#recent-changes)
9. [Contributing](#contributing)
10. [License](#license)

## Overview

EchoSee is a production-ready voice assistant that combines advanced AI with modern software engineering practices. Key differentiators:

- **🎵 Real-time Streaming**: TTS audio plays as text is generated, reducing perceived latency by ~70%
- **💾 Persistent Memory**: Conversations survive restarts with full-text search capabilities
- **🏗️ Clean Architecture**: Modular design with dependency injection and plugin systems
- **🔍 Health Monitoring**: Built-in diagnostics for troubleshooting and deployment
- **📱 Multi-device**: Web PWA companion for notifications across devices
- **⚡ Low Latency**: Optimized for real-time voice interactions

---

## Project Structure

```
EchoSee/
├── core/                          # 🆕 Core infrastructure
│   ├── __init__.py
│   ├── app_context.py            # Application lifecycle management
│   ├── audio_providers.py        # TTS/STT provider abstraction
│   ├── clients.py                # API client factory
│   ├── config_manager.py         # Centralized configuration
│   ├── conversation_storage.py   # SQLite conversation persistence
│   ├── exceptions.py             # Custom exceptions
│   ├── health_check.py           # Component health monitoring
│   └── llm_factory.py            # LLM provider abstraction
├── agent_management/
│   ├── __init__.py
│   ├── agent_manager.py          # Main agent orchestrator
│   ├── agents.py                 # Legacy agent definitions
│   └── tools/                    # 🆕 Organized tool registry
│       ├── __init__.py
│       ├── code_tools.py         # LeetCode integration
│       ├── event_tools.py        # Alarm/scheduling tools
│       ├── notification_tools.py # Echonoti notifications
│       ├── search_tools.py       # Web search capabilities
│       └── time_tools.py         # Time/date utilities
├── app_manager/
│   ├── __init__.py
│   └── manager.py                # Main application loop
├── event_manager/
│   ├── __init__.py
│   ├── callbacks.py              # Event callbacks
│   ├── event_manager.py          # Event scheduling system
│   ├── scheduler.md              # Scheduler documentation
│   └── tools.py                  # Event management utilities
├── logger/
│   ├── __init__.py
│   └── logger.py                 # Centralized logging
├── tts/
│   ├── __init__.py
│   └── tts.py                    # 🆕 Streaming TTS implementation
├── stt/
│   ├── __init__.py
│   └── stt.py                    # Speech-to-text processing
├── utils/
│   ├── __init__.py
│   ├── cot.py                    # Chain-of-thought utilities
│   ├── env_validation.py         # Environment validation
│   └── utils.py                  # General utilities
├── echonoti/                     # 🆕 Companion PWA
│   ├── next.config.ts
│   ├── package.json
│   ├── src/
│   │   ├── app/
│   │   │   ├── api/notifications/
│   │   │   ├── chat/
│   │   │   └── bookmarked/
│   │   └── components/
│   └── tailwind.config.ts
├── assets/
│   ├── alarm1.wav
│   └── welcome.png
├── logs/                         # Application logs
├── mcp_servers/                  # 🆕 MCP server integrations
├── config.yml                    # Configuration file
├── requirements.txt
├── setup_dev.sh                  # Development setup script
├── README.md
└── LICENSE
```

---

## Key Components

### Core Infrastructure (`core/`)

#### Configuration Management (`config_manager.py`)
- **Purpose:** Centralized configuration with validation and singleton pattern
- **Features:** YAML loading, dot-notation access, automatic validation, environment variable checking
- **Highlights:** Prevents configuration errors at startup with clear error messages

#### Client Factory (`clients.py`)
- **Purpose:** Lazy-initialized API clients with singleton pattern
- **Supports:** OpenAI, Tavily, DeepSeek clients
- **Highlights:** Automatic API key validation and connection reuse

#### LLM Factory (`llm_factory.py`)
- **Purpose:** Unified interface for different LLM providers
- **Providers:** OpenAI, DeepSeek with plugin architecture
- **Highlights:** Supports both chat models and agent models (LiteLLM)

#### Audio Providers (`audio_providers.py`) 🆕
- **Purpose:** Plugin-based TTS/STT abstraction layer
- **Providers:** OpenAI TTS (with streaming), Kokoro TTS, OpenAI STT
- **Highlights:** Extensible design for adding new audio providers

#### Application Context (`app_context.py`)
- **Purpose:** Manages application lifecycle and shared resources
- **Features:** EventManager initialization, cleanup handling, singleton pattern
- **Highlights:** Eliminates module-level side effects

#### Conversation Storage (`conversation_storage.py`) 🆕
- **Purpose:** Persistent conversation history with SQLite backend
- **Features:** Full-text search, conversation retrieval, metadata storage
- **Highlights:** Survives application restarts, searchable past conversations

#### Health Check (`health_check.py`) 🆕
- **Purpose:** System health monitoring and diagnostics
- **Checks:** All components (LLM, TTS, STT, storage, config)
- **CLI:** `python -m app_manager.manager health`

### Agent Management (`agent_management/`)

#### Agent Manager (`agent_manager.py`) 🆕
- **Purpose:** Main orchestrator for conversational AI
- **Features:** Streaming responses, conversation persistence, tool integration
- **Tools:** Time, search, events, notifications, code analysis

#### Tool Registry (`tools/`) 🆕
- **Purpose:** Organized tool system for agent capabilities
- **Tools:**
  - `time_tools.py`: Current time/date functions
  - `search_tools.py`: Web search with Tavily
  - `event_tools.py`: Alarm and scheduling
  - `notification_tools.py`: Echonoti notifications
  - `code_tools.py`: LeetCode integration

### Audio Processing

#### Text-to-Speech (`tts/tts.py`) 🆕
- **Purpose:** Streaming-capable text-to-speech conversion
- **Features:** Sentence buffering, concurrent audio playback, interruption support
- **Providers:** OpenAI (streaming), Kokoro (offline)
- **Highlights:** Audio starts playing within ~500ms of LLM generation

#### Speech-to-Text (`stt/stt.py`)
- **Purpose:** Voice input transcription
- **Provider:** OpenAI Whisper API
- **Features:** Real-time recording, key-based activation

### Event Management (`event_manager/`)

#### Event Manager (`event_manager.py`)
- **Purpose:** High-performance event scheduling system
- **Features:** Heap-based priority queue, recurring events, callback system
- **Highlights:** Efficient for large numbers of scheduled events

#### Callbacks (`callbacks.py`)
- **Purpose:** Event execution handlers
- **Supports:** Audio playback, notifications, custom callbacks

### Echonoti PWA (`echonoti/`) 🆕

#### Companion App
- **Purpose:** Cross-device notification system
- **Tech:** Next.js PWA with API endpoints
- **Features:** Real-time notifications, bookmark management, chat interface

### Application Manager (`app_manager/manager.py`)

#### Main Loop
- **Purpose:** Integrates all components into cohesive voice assistant
- **Features:** Streaming mode selection, error handling, graceful shutdown
- **CLI Commands:** `health`, `help`, default (start assistant)

### Logger (`logger/logger.py`)

#### Centralized Logging
- **Purpose:** Structured logging with file output
- **Features:** Configurable levels, organized log files
- **Highlights:** Prevents log pollution of root logger

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- FFmpeg (for audio processing)
- Git

### Quick Start

1. **Clone the Repository:**
```bash
git clone https://github.com/EzeLLM/EchoSee.git
cd EchoSee
```

2. **Install Dependencies:**
```bash
pip install -e .
```

3. **Setup Environment Variables:**
Create a `.env` file or export environment variables:
```bash
# Required
OPENAI_API_KEY=sk-your-openai-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here

# Optional (for DeepSeek provider)
DEEPSEEK_API_KEY=sk-your-deepseek-key-here

# Optional (for LangSmith tracing)
LANGCHAIN_API_KEY=ls-your-langsmith-key-here
```

4. **Run Development Setup (Optional):**
```bash
./setup_dev.sh  # Sets up LangSmith tracing
```

5. **Start the Assistant:**
```bash
python -m app_manager.manager
```

### Health Check
Verify your installation:
```bash
python -m app_manager.manager health
```

### Echonoti PWA Setup (Optional)
```bash
cd echonoti
npm install
npm run dev
```

---

## Configuration

EchoSee uses a centralized `config.yml` file. Key sections:

### LLM Configuration
```yaml
LLM:
  provider: openai  # openai or deepseek
  model: gpt-4o-mini
  high_performance_provider: openai
  high_performance_model: gpt-4o-mini
  system_prompt: |
    You are a helpful assistant...
```

### TTS Configuration (Streaming)
```yaml
TTS:
  method: openai  # openai or kokoro
  voice: shimmer
  device: 2  # Audio device ID
  streaming_enabled: true  # Enable streaming for low latency
  sentence_buffer_chars: 50  # Minimum chars before TTS processes
```

### Agent Manager
```yaml
AgentManager:
  clear_time: 30  # Minutes before clearing memory
  persistence_enabled: true  # Save conversations to database
  max_context_turns: 5  # Past turns to load on startup
```

### Audio Configuration
```yaml
STT:
  sample_rate: 16000
  channels: 1
```

### Event Callbacks
```yaml
callbacks:
  alarm:
    sound: assets/alarm1.wav
```



## Usage Guide

### For End-Users

#### Basic Interaction
1. **Start the Assistant:**
```bash
python -m app_manager.manager
```

2. **Voice Commands:**
   - Press and hold **SPACEBAR** to record
   - Release to send your message
   - Say "quit" to exit

3. **Available Commands:**
   - **Questions:** "What's the weather like?"
   - **Time/Alarms:** "Set a timer for 5 minutes"
   - **Search:** "Search for Python tutorials"
   - **Code Help:** "Help me with LeetCode problem 104"
   - **Notifications:** "Send a notification to my phone"

#### Streaming Mode
- Audio starts playing as soon as the assistant has enough text (sentence-by-sentence)
- Much faster perceived response time (~500ms vs waiting for full response)

#### Conversation Persistence
- All conversations are saved automatically
- Context is maintained across restarts
- Search past conversations with the assistant

### CLI Commands

```bash
# Start assistant (default)
python -m app_manager.manager

# Health check all components
python -m app_manager.manager health

# Show help
python -m app_manager.manager help
```

### For Developers

#### Extending Tools
Add new tools in `agent_management/tools/`:
```python
# Example: weather_tools.py
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    # Implementation here
    pass
```

#### Adding Audio Providers
Extend `core/audio_providers.py`:
```python
class CustomTTSProvider(TTSProvider):
    def generate(self, text: str, **kwargs) -> List[np.ndarray]:
        # Your implementation
        pass
```

#### Configuration Validation
The system validates all configuration on startup. Invalid configs show clear error messages.

#### Testing
- Run health checks: `python -m app_manager.manager health`
- Check logs in `logs/` directory
- Test components individually with their `__main__` blocks

---

## Recent Changes

### v2.0.0 - Major Refactoring & Streaming TTS
- **🎵 TTS Streaming**: Audio plays as LLM generates text (sentence buffering)
- **💾 Conversation Persistence**: SQLite-based conversation history with search
- **🏥 Health Monitoring**: Built-in diagnostics (`python -m app_manager.manager health`)
- **🏗️ Core Architecture**: Modular design with dependency injection
- **📱 Echonoti PWA**: Companion web app for cross-device notifications

### Recent Commits
- `0c82f6aa`: Add conversation persistence and health check features
- `12ad1f6a`: Refactor agent tools and add audio provider abstraction
- `1347731d`: Refactor config and client management; add core modules

---

## Development Guidelines

### Code Style
- Follow **PEP 8** guidelines
- Use type hints for all function parameters and return values
- Add docstrings to all public functions and classes

### Architecture Principles
- **Dependency Injection**: Use factories and providers for external dependencies
- **Single Responsibility**: Each module has one clear purpose
- **Plugin Architecture**: Easy to extend with new providers/tools
- **Error Handling**: Fail fast with clear error messages

### Testing
- Run health checks before committing: `python -m app_manager.manager health`
- Test components individually with `__main__` blocks
- Validate configuration changes don't break existing functionality

---

## Future Roadmap

### Short Term (Next Release)
- **Voice Interruption**: Allow users to interrupt responses mid-speech
- **Wake Word Detection**: "Hey EchoSee" activation (low-latency)
- **Multi-language Support**: I18n for international users

### Medium Term
- **Docker Deployment**: Containerized deployment with docker-compose
- **Web Dashboard**: Real-time monitoring and configuration UI
- **Plugin Marketplace**: Community-contributed tools and providers

### Long Term
- **Offline Mode**: Local LLM support for privacy-focused deployments
- **Multi-modal**: Image/video understanding capabilities
- **Smart Home Integration**: IOT device control and automation

---

## Contributing

We welcome contributions! Please:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Setup
```bash
# Clone and setup
git clone https://github.com/EzeLLM/EchoSee.git
cd EchoSee
pip install -e .
./setup_dev.sh  # Optional: LangSmith setup

# Run tests
python -m app_manager.manager health
```

### Areas for Contribution
- **New Tools**: Add capabilities in `agent_management/tools/`
- **Audio Providers**: Extend TTS/STT options in `core/audio_providers.py`
- **UI Improvements**: Enhance the Echonoti PWA
- **Documentation**: Improve guides and API docs

---

## License

This project is licensed under the terms detailed in the `LICENSE` file.

**EchoSee** is continuously evolving. Your feedback and contributions are invaluable in making this assistant smarter, more efficient, and user-friendly. Enjoy using and developing **EchoSee**!
