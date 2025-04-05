# EchoSee: Intelligent Voice Assistant
![Welcome Image](assets/welcome.png)
EchoSee is a cutting-edge assistant pipeline designed to replace traditional rule-based assistants (such as Amazon's Alexa or Google Home) with an engaging, flexible, and intelligent voice assistant. Whether you're a developer looking to extend and contribute to the project or a user eager to harness its conversational capabilities, this README will guide you through everything you need to know.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Key Components](#key-components)
    - [Agent Management](#agent-management)
    - [Logger](#logger)
    - [Utilities](#utilities)
    - [Text-to-Speech (TTS)](#text-to-speech-tts)
    - [Speech-to-Text (STT)](#speech-to-text-stt)
    - [Event Manager](#event-manager)
    - [Application Manager](#application-manager)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [Development Guidelines](#development-guidelines)
7. [Potential Improvements & Future Roadmap](#potential-improvements--future-roadmap)
8. [Contributing](#contributing)
9. [License](#license)
---
## ✨ Latest Feature: LeetCode Integration ✨

EchoSee now includes a powerful LeetCode assistant that helps you tackle coding challenges! This intelligent feature leverages AI to provide customized support for your programming practice sessions.

### LeetCode Features:
- **Problem Retrieval:** Get detailed problem descriptions directly from LeetCode by specifying the problem number.
- **Hint Generation:** Request targeted hints that guide your thinking without revealing the complete solution.
- **Step-by-Step Guides:** Receive conversational explanations of solution approaches with time and space complexity analysis.
- **Adaptive Assistance:** The system automatically classifies your request to provide the appropriate level of help.

### Using the LeetCode Feature:
Simply ask EchoSee about a LeetCode problem by referencing its number:
- "Give me a hint for LeetCode problem 141"
- "I need guidance on solving LeetCode number 23"
- "Explain the approach for problem 104 on LeetCode"

The LeetCode agent will detect the problem number, retrieve the problem details, and provide the appropriate level of assistance based on your request.
---

## Overview

EchoSee is built to provide an interactive, conversational assistant that overcomes the limitations of rule-based systems. By integrating advanced language models and leveraging a modular design, it aims to deliver a natural user experience with voice-driven interaction and powerful event handling.

---

## Project Structure


```
EchoSee
├── agent_management
│   ├── agents.py
│   ├── agent_manager.py
│   └── __inti__.py
├── logger
│   ├── __init__.py
│   └── logger.py
├── LICENSE
├── requirements.txt
├── CONSTANTS.py
├── utils
│   └── utils.py
├── README.md
├── setup.py
├── logs
├── tts
│   ├── tts.py
│   └── __init__.py
├── event_manager
│   ├── tools.py
│   ├── __init__.py
│   ├── scheduler.md
│   ├── callbacks.py
│   └── event_manager.py
├── config.yml
├── __inti__.py
├── app_manager
│   ├── __init__.py
│   └── manager.py
├── assets
│   └── alarm1.wav
└── stt
    ├── __init__.py
    └── stt.py

```

---

## Key Components

### Agent Management
- **Purpose:** Manages user interactions, processes messages, maintains conversation history, and interacts with various tools (like event scheduling and searches).
- **Files:** `agent_management/agents.py` and `agent_management/agent_manager.py`
- **Highlights:** Uses a React-based agent pattern to generate responses and supports conversation history management with an auto-clear timer.

### Logger
- **Purpose:** Centralized logging to capture debug, info, warning, and error messages.
- **File:** `logger/logger.py`
- **Highlights:** Configurable logging that writes to a file in the `logs` directory.

### Utilities
- **Purpose:** Provides helper functions such as YAML configuration loading, LLM initialization, and environment setup.
- **File:** `utils/utils.py`
- **Highlights:** Automatically loads environment variables and registers cleanup routines for graceful shutdown.

### Text-to-Speech (TTS)
- **Purpose:** Converts text responses into speech.
- **Files:** `tts/tts.py`
- **Options:**
  - **Kokoro:** A lightweight model suited for low-resource devices (like Raspberry Pi 5).
  - **OpenAI TTS:** Uses the GPT-4o-mini-tts model, ideal for scenarios where low latency is critical.
- **Configuration:** Set in `config.yml` under the `TTS` section.

### Speech-to-Text (STT)
- **Purpose:** Transcribes user speech to text.
- **File:** `stt/stt.py`
- **Highlights:** Uses OpenAI's STT API with plans to integrate Whisper for private server deployments.

### Event Manager
- **Purpose:** Schedules and manages events (such as alarms) using an efficient heap data structure.
- **Files:** `event_manager/event_manager.py`, `event_manager/callbacks.py`, and `event_manager/tools.py`
- **Highlights:** Supports both one-off and recurring events with callback functions to handle tasks.

### Application Manager
- **Purpose:** Integrates all components to create the full assistant experience.
- **File:** `app_manager/manager.py`
- **Highlights:** Manages the main loop for capturing user input (via STT), processing it with the agent, and converting responses back to speech using TTS.

---

## Installation & Setup

1. **Clone the Repository:**
```bash
git clone https://github.com/EzeLLM/EchoSee.git
cd echosee
```
2. **Install Dependencies: Install the project in editable mode to ease development:**
```bash
pip install -e .
```

3. **Configure the Application:**

- Open config.yml and adjust settings for TTS, STT, LLM, and event callbacks.

4. **Environment Variables:**
Set ```OPENAI_API_KEY``` , ```TAVILY_API_KEY``` , and (optionally) ```DeepSeek``` API keys in ```.env``` or as environment variable.
4. **LangSmith (Optional):**
Run ```setup_dev``` to setup LangSmith. Make sure api is set in ```.env``` or as environment variable.
```
. setup_dev.sh
```
5. **Run the Application:** 
Start the assistant by running:
python -m app_manager.manager



# Usage Guide

## For End-Users:
### Launching the Assistant:
- Run the application manager.
- You will be prompted to speak your query. Use the specified key (the spacebar) to record your voice.
- The assistant processes your query and responds with a spoken answer.

### Interacting:
- Ask questions, set alarms, or request information.
- The assistant maintains a short conversation history which auto-clears after a period of inactivity (configurable in config.yml).

## For Developers:
### Developing New Features:
- **Extend Tools:** Add new functionalities in the `agent_management` and `event_manager` modules.
- **Improve TTS/STT:** Experiment with different models or integrate new libraries.
- **Enhance Event Management:** Contribute optimizations or move to asynchronous handling for better performance.

### Debugging:
- Use the `logger` module for detailed logs.
- Adjust the logging level in `logger/logger.py` for more verbose output if needed.

### Testing:
- Write tests for new features.
- Validate configuration changes in `config.yml` to ensure compatibility.

### Code Contribution Guidelines:
- Follow **PEP 8** style guidelines.
- Ensure new features are well-documented.
- Write clear commit messages and include updates to this README if necessary.

## Potential Improvements & Future Roadmap
### Efficient Wake Word Detection:
- Integrate low-latency wake word detection algorithms suitable for resource-constrained devices.

### Enhanced Streaming:
- Implement streaming for both text generation and audio playback to reduce latency.

### Advanced Deep Search:
- Explore integration with more robust search libraries to improve the voice user experience during search queries.

### Private Server Deployments:
- Develop single-command solutions for self-hosted models (e.g., Whisper via `whisper.cpp`).

### UI/UX Enhancements:
- Consider adding a companion mobile or web interface for visual feedback and configuration.

### Modular Plugin System:
- Enable developers to add or remove functionalities easily without affecting the core codebase.



## Contributing
We welcome contributions from the community! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Ensure your code follows the project's style guidelines.
4. Open a pull request with a detailed description of your changes.

## License
This project is licensed under the terms detailed in the `LICENSE` file.

**EchoSee** is continuously evolving. Your feedback and contributions are invaluable in making this assistant smarter, more efficient, and user-friendly. Enjoy using and developing **EchoSee**!
