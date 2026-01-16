# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EchoSee is an intelligent voice assistant pipeline that replaces traditional rule-based assistants with a flexible, LLM-powered conversational system. It integrates Speech-to-Text (STT), Text-to-Speech (TTS), LangGraph agents, and event management to provide a natural voice-driven interaction experience.

The project includes:
- **Python voice assistant** (main application in root directory)
- **Echonoti PWA** (Progressive Web App for notifications in `echonoti/` directory)
- **Voice UI** (Web-based voice interface at `echonoti/src/app/voice/` with Flask API in `api/`)

## Development Commands

### Python Assistant

**Setup:**
```bash
# Install in editable mode
pip install -e .

# Optional: Setup LangSmith for development/debugging
source setup_dev.sh  # or . setup_dev.sh
```

**Running:**
```bash
# Run the main voice assistant
python -m app_manager.manager

# Test individual components
python -m agent_management.agent_manager
python -m agent_management.agents
python -m event_manager.tools
```

**Testing:**
```bash
# Most modules have __main__ blocks for testing
# Run them directly to test functionality
python -m <module_path>
```

### Echonoti PWA

**Setup & Development:**
```bash
cd echonoti
npm install
npm run dev  # Runs on http://localhost:9003
```

**Building:**
```bash
cd echonoti
npm run build
```

### Voice UI

**Quick Start:**
```bash
./scripts/launch_voice_ui.sh  # Starts both API server and Next.js
```

**Manual Start:**
```bash
# Terminal 1: Start voice API server
python -m api.voice_server  # Runs on http://localhost:9004

# Terminal 2: Start Next.js (in echonoti directory)
cd echonoti && npm run dev  # Runs on http://localhost:9003
```

Then navigate to http://localhost:9003/voice

## Architecture

### Agent Architecture (LangGraph React Pattern)

The system uses LangGraph's `create_react_agent` pattern for LLM-based reasoning:

1. **AgentManager** (`agent_management/agent_manager.py`):
   - Maintains conversation history with auto-clear timer (configurable in `config.yml`)
   - Manages running message list for the agent
   - Processes messages through the React agent loop
   - The agent has access to tools and can call them reactively

2. **Agent Tools** (`agent_management/agents.py`):
   - `search()` - Basic web search via Tavily API
   - `robust_search()` - Advanced multi-query search with diversification (see SearchClient architecture below)
   - `get_current_time()` / `get_current_date()` - Time utilities
   - `leetcode_agent()` - LeetCode problem solver
   - Event management tools (from `event_manager/tools.py`)
   - Echonoti notification tool

3. **LLM Configuration**:
   - **Standard LLM**: Used for agent reasoning and basic tasks (configured in `config.yml` under `LLM.model` and `LLM.provider`)
   - **High-Performance LLM**: Used for complex synthesis tasks like robust search summarization (configured under `LLM.high_performance_model` and `LLM.high_performance_provider`)
   - Supports OpenAI and DeepSeek providers

### SearchClient Architecture (Diversified Search)

The `robust_search` agent (`agent_management/helpers/search/SearchClient.py`) implements a sophisticated multi-stage search pipeline:

1. **Query Diversification**:
   - Generates N×oversample_factor candidate queries from user query
   - Uses high-performance LLM to create diverse perspectives and phrasings
   - Embeds all candidates using OpenAI embeddings (text-embedding-ada-002)

2. **Query Clustering**:
   - Clusters candidate queries into max_queries groups using KMeans
   - Selects the query closest to each cluster centroid for best representation
   - Ensures diverse search coverage

3. **Parallel Search Execution**:
   - Executes all selected queries against Tavily API concurrently using asyncio
   - Caches results to avoid duplicate searches

4. **Snippet Filtering (MMR)**:
   - Uses Maximum Marginal Relevance (MMR) algorithm to balance:
     - **Relevance**: Similarity to user query (controlled by `mmr_lambda`)
     - **Novelty**: Dissimilarity to already selected snippets
   - Default `mmr_lambda=0.6` favors relevance; lower values increase novelty weight

5. **Synthesis**:
   - High-performance LLM synthesizes final answer from filtered snippets
   - Includes source citations and progressive report format

**Key Parameters** (in `SearchConfig`):
- `max_queries`: Number of diverse search queries to execute (default: 5)
- `oversample_factor`: Multiplier for candidate generation (default: 20)
- `mmr_lambda`: Relevance vs novelty trade-off (default: 0.6)
- `citation_limit`: Max evidence snippets for synthesis (default: 6)

### Event Management

The `EventManager` (`event_manager/event_manager.py`) uses a heap-based priority queue for efficient event scheduling:

- **Heap Structure**: Events stored in min-heap ordered by trigger time for O(log n) insertion and O(1) next-event lookup
- **Threading**: Runs in daemon thread with lock-protected event queue
- **Event Types**:
  - One-off events via `add_event(event_time, callback, *args, **kwargs)`
  - Recurring events via `add_recurring_event(interval, callback, *args, **kwargs)`
- **Deduplication**: Checks for duplicate events before insertion
- **Tools**:
  - `set_alarm_at_specific_time(time)` - Format: "YYYY:MM:DD:HH:MM:SS"
  - `set_alarm_with_time_delta(delta)` - Format: "DD:HH:MM:SS"
  - `stop_alarm()` - Stops active alarm

### Application Flow

1. **Initialization** (`app_manager/manager.py`):
   - Load config from `config.yml`
   - Initialize TTS, STT, AgentManager
   - Start EventManager (happens in `utils/utils.py` module import)

2. **Conversation Loop**:
   - STT captures and transcribes user speech
   - AgentManager processes message through React agent
   - Agent may invoke tools (search, alarms, notifications, etc.)
   - Response converted to speech via TTS and played

3. **Cleanup**:
   - `atexit` handler in `utils/utils.py` stops EventManager on exit

### Module Dependencies

```
app_manager.manager
  ├── agent_management.agent_manager (AgentManager)
  ├── tts.tts (TTS)
  ├── stt.stt (STT)
  └── utils.utils (config, LLM instances, EventManager)
      └── event_manager.event_manager (EventManager - auto-started)

agent_management.agent_manager
  ├── agent_management.agents (tools)
  ├── agent_management.helpers.Echonoti.agent (send_notification)
  ├── agent_management.helpers.search.SearchClient (robust_search)
  ├── event_manager.tools (alarm tools)
  └── utils.utils (llm, llm_config)

agent_management.agents
  └── agent_management.helpers.search.SearchClient (robust_search implementation)
```

### Echonoti Integration

**Backend**: PWA built with Next.js, SQLite database for notifications
**API Endpoints**:
- `GET /api/notifications` - List all notifications
- `POST /api/notifications` - Create notification (requires: headline, summary, content, type)
- `GET /api/notifications/{id}` - Get single notification
- `PUT /api/notifications/{id}` - Update notification
- `DELETE /api/notifications/{id}` - Delete notification

**Python Integration**: `agent_management/helpers/Echonoti/agent.py` provides `send_notification` tool that the agent can use to push notifications to the PWA.

## Configuration (`config.yml`)

**LLM Configuration**:
- `model`: Model for agent reasoning (e.g., "gpt-4o-mini")
- `provider`: LLM provider ("openai", "deepseek")
- `high_performance_model`: Model for complex synthesis tasks (e.g., "o4-mini", "deepseek-chat")
- `high_performance_provider`: Provider for high-performance model
- `system_prompt`: Agent personality and behavior instructions

**TTS Configuration**:
- `method`: "openai" or "kokoro" (lightweight for Raspberry Pi)
- `device`: Audio output device index
- For OpenAI: voice, model, response_format, speed
- For Kokoro: lang_code, voice

**STT Configuration**:
- `model`: Transcription model (e.g., "gpt-4o-mini-transcribe")
- `sample_rate`: Audio sample rate (16000)
- `channels`: Audio channels (1 for mono)
- `prompt`: Transcription guidance

**AgentManager**:
- `clear_time`: Conversation history auto-clear timeout in minutes (default: 30)

**Callbacks**:
- `alarm.sound`: Path to alarm sound file

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
DEEPSEEK_API_KEY=your_deepseek_key  # Optional, if using DeepSeek provider

# Optional LangSmith (for debugging/tracing)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=your_project_name
```

## Key Files

- `config.yml` - Central configuration for all components
- `CONSTANTS.py` - Project-wide constants
- `setup.py` - Package installation configuration
- `requirements.txt` - Python dependencies
- `utils/utils.py` - Shared utilities, LLM initialization, EventManager auto-start, exit handlers
- `logger/logger.py` - Centralized logging system

## Important Patterns

**Agent Tool Development**:
- Use `@tool` decorator from `langchain_core.tools`
- Provide detailed docstrings (used by LLM for tool selection)
- Register tools in `AgentManager.__init__()` tools list

**Event Scheduling**:
- Always use `utils.utils.event_manager_instance` (imported as `em`)
- Store returned event IDs for later cancellation
- Callbacks execute in EventManager thread - keep them short

**Search Strategy**:
- Use `search()` for simple factual queries
- Use `robust_search()` for complex research requiring multiple perspectives
- Adjust `max_queries` parameter based on query complexity (3-5 typical range)

**Configuration Changes**:
- Edit `config.yml` for runtime behavior changes
- No code changes needed for model switching, TTS/STT provider changes, or timing adjustments
- Restart application to apply configuration changes
