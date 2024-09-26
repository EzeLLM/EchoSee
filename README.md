# EchoSee README

Welcome to the EchoSee Project! This README will guide you through understanding, setting up, and contributing to EchoSee—a cutting-edge project leveraging the latest advancements in Language Learning Models (LLMs) to create an advanced Alexa and Google Home alternative, complete with LLM agents and action capabilities. Vision support to be added in the future.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Future Plans](#future-plans)
- [Contributing](#contributing)
- [License](#license)

## Introduction

EchoSee isn't just another voice assistant—it's a leap forward in AI-powered home automation. Utilizing state-of-the-art LLMs, EchoSee aims to surpass the capabilities of traditional assistants like Alexa and Google Home. With EchoSee, you'll not only experience voice-based interactions but also complex actions performed by LLM agents. In the future, EchoSee will even be able to "see" through a camera, adding another layer of interactive capability.

## Features

- **LLM Integration**: Incorporates advanced LLMs for more natural, nuanced conversations.
- **Multi-Model Text-to-Speech (TTS)**: Supports light, mid, heavy, and API-based TTS operations.
- **Speech-to-Text (STT)**: High-accuracy STT using local or Groq-based models.
- **Modular Design**: Easily extendable with new features like camera integration.
- **Logging**: Comprehensive logging for debugging and performance tracking.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10
- pip (Python package installer)
- Access to Groq API (With future ollama and vllm support this will be optional)
- Access to amazon aws free tier (Optional, required in api mode. In case of lack of gpu, api mode is recommended.)

### Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/EchoSee.git
    cd EchoSee
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Environment Variables**:
   Set up your environment variables in a `.env` file for sensitive data (e.g., Groq API keys):
    ```plaintext
    GROQAPI=your_groq_api_key
    AWSKEYID=your_aws_access_key_id
    AWSSECRETKEY=your_aws_secret_access_key
    ```

## Project Structure

Here's a high-level view of the project structure to help you find what you need quickly:

```
EchoSee
├── cb.t
└── dev
    ├── code
    │   ├── tinker
    │   ├── config
    │   ├── __pycache__
    │   ├── audio
    │   ├── log
    │   ├── EchoSee.py
    │   ├── LLMInference.py
    │   ├── TextToSpeech.py
    │   ├── SpeechToText.py
    │   └── logger.py
    ├── audio
    │   ├── embeddings
    │   ├── tts_output.wav
    │   ├── sound_effects
    │   ├── mid_speaker_wav
    │   ├── output_reduced.wav
    │   ├── output.wav
```

## Configuration

**echosee.yaml File**

This file configures components such as Speech-to-Text (STT), Text-to-Speech (TTS), and more.

```yaml
SpeechToText:
 ChunkSize: 4096
 Rate: 44100
 Channels: 1
 Threshold: 750
 WaitTime: 50
 Modes:
  local:
  groq:
 mode: 'groq' # local, groq

TTS:
 Modes:
  light:
   speaker_index: 7306
  mid:
   speaker_wav_path: 'dev/audio/mid_speaker_wav/female.wav'
   language: 'en'
  heavy:
  api:
   voice: 'Matthew'
 mode: 'api' # light, heavy, mid , api
 save_path: 'dev/audio/tts_output.wav'
```

## Usage

1. **Running EchoSee**:

Execute the main script:

```bash
python dev/code/EchoSee.py
```

2. **Logging**:

Check the logs in `dev/log/app.log` for detailed debugging information.

## Future Plans

EchoSee aims to continually advance with new features, including:

- **Camera Integration**: Enhancing the assistant with visual input capabilities.
- **Complex Action Handling**: More sophisticated LLM-driven actions.
- **Enhanced Security**: Incorporating stronger authentication mechanisms.

## Contributing

We welcome contributions to EchoSee! To get started, fork the repository, make your changes, and submit a pull request. Here's how you can help:

1. **Report Bugs**: File issues for any bugs you find.
2. **Suggest Features**: Use the issue tracker to suggest improvements.
3. **Submit Pull Requests**: Follow our [contributing guidelines](CONTRIBUTING.md).

## License

EchoSee is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Dive into the future with EchoSee and become a part of the evolution of home automation! Happy coding! 🚀