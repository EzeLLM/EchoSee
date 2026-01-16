"""Voice Assistant API Server.

Provides REST API endpoints for voice interaction with the EchoSee assistant.
"""

import os
import tempfile
import logging
import subprocess
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
import base64

# Import EchoSee components
from agent_management.agent_manager import AgentManager
from stt.stt import STT
from tts.tts import TTS
from core.config_manager import config
from core.app_context import app_context

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Initialize EchoSee components
app_context.initialize()
stt = STT()
tts = TTS()
agent_manager = AgentManager()

# Get TTS config
tts_config = config.get_section('TTS')


def convert_audio_to_mp3(input_path: str, output_path: str) -> bool:
    """Convert audio file to MP3 format using ffmpeg.

    Args:
        input_path: Path to input audio file
        output_path: Path for output MP3 file

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vn',  # No video
            '-ar', '16000',  # Sample rate 16kHz
            '-ac', '1',  # Mono
            '-b:a', '128k',  # Bitrate
            '-y',  # Overwrite output
            output_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return False


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


@app.route('/api/conversation/history', methods=['GET'])
def get_conversation_history():
    """Get current conversation history."""
    try:
        # Get from storage (persistent)
        history = []
        if agent_manager.storage:
            stored = agent_manager.storage.get_recent_history(limit=50)
            history = [{
                "id": item["id"],
                "timestamp": item["timestamp"],
                "user": item["user_message"],
                "assistant": item["assistant_message"],
                "metadata": item.get("metadata", {})
            } for item in stored]

        return jsonify({"history": history}), 200
    except Exception as e:
        logger.error(f"Error getting history: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversation/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history."""
    try:
        # Clear in-memory history
        agent_manager.clear_history()

        # Clear persistent storage
        if agent_manager.storage:
            agent_manager.storage.clear_history()

        logger.info("Conversation history cleared")
        return jsonify({"status": "cleared"}), 200
    except Exception as e:
        logger.error(f"Error clearing history: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversation/stats', methods=['GET'])
def get_conversation_stats():
    """Get conversation statistics."""
    try:
        stats = agent_manager.get_conversation_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/voice/process', methods=['POST'])
def process_voice():
    """Process voice input and return response.

    Expected request:
    - multipart/form-data with 'audio' file

    Returns:
    - JSON with transcription, response text, audio (base64), and timestamps
    """
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        # Create temporary file for converted audio
        temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_mp3_path = temp_mp3.name
        temp_mp3.close()

        try:
            # Step 1: Convert audio to MP3
            logger.info("Converting audio to MP3...")
            if not convert_audio_to_mp3(temp_audio_path, temp_mp3_path):
                return jsonify({"error": "Audio conversion failed"}), 500

            # Step 2: Transcribe audio
            logger.info("Transcribing audio...")
            transcription = stt.transcribe(temp_mp3_path)
            logger.info(f"Transcription: {transcription}")

            # Step 3: Process through agent
            logger.info("Processing through agent...")
            response_messages = agent_manager.process_message(transcription)
            response_text = response_messages[-1].content if response_messages else "No response"
            logger.info(f"Agent response: {response_text[:100]}...")

            # Step 4: Generate TTS audio
            logger.info("Generating TTS audio...")
            audio_segments = tts.generate(response_text)

            # Combine audio segments
            if audio_segments:
                import numpy as np
                combined_audio = np.concatenate(audio_segments)

                # Convert to bytes for transmission
                import soundfile as sf
                audio_buffer = io.BytesIO()
                sf.write(audio_buffer, combined_audio, tts.sample_rate, format='WAV')
                audio_buffer.seek(0)
                audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
            else:
                audio_base64 = None

            # Step 5: Get timestamps from OpenAI
            # Note: OpenAI TTS doesn't provide word-level timestamps by default
            # We'll use a simple approach: split by sentences and estimate timing
            timestamps = generate_timestamps(response_text, audio_segments, tts.sample_rate)

            return jsonify({
                "transcription": transcription,
                "response": response_text,
                "audio": audio_base64,
                "timestamps": timestamps
            }), 200

        finally:
            # Clean up temporary files
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            if os.path.exists(temp_mp3_path):
                os.unlink(temp_mp3_path)

    except Exception as e:
        logger.error(f"Error processing voice: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def generate_timestamps(text: str, audio_segments: list, sample_rate: int):
    """Generate approximate timestamps for text based on audio segments.

    Args:
        text: The text that was spoken
        audio_segments: List of audio arrays
        sample_rate: Audio sample rate

    Returns:
        List of timestamp objects with word/phrase and timing
    """
    import numpy as np
    import re

    if not audio_segments:
        return []

    # Calculate total audio duration
    total_samples = sum(len(seg) for seg in audio_segments)
    total_duration = total_samples / sample_rate

    # Split text into words
    words = re.findall(r'\S+', text)
    if not words:
        return []

    # Calculate approximate time per word
    time_per_word = total_duration / len(words)

    # Generate timestamps
    timestamps = []
    current_time = 0.0

    for word in words:
        timestamps.append({
            "word": word,
            "start": round(current_time, 2),
            "end": round(current_time + time_per_word, 2)
        })
        current_time += time_per_word

    return timestamps


@app.route('/api/voice/stream', methods=['POST'])
def process_voice_streaming():
    """Process voice with streaming response.

    Expected request:
    - multipart/form-data with 'audio' file

    Returns:
    - Streaming response with text chunks
    """
    # This endpoint could be implemented for streaming text responses
    # For now, using the standard process endpoint is simpler
    return jsonify({"error": "Streaming not implemented yet"}), 501


if __name__ == '__main__':
    # Run server
    port = int(os.environ.get('VOICE_API_PORT', 9004))
    app.run(host='0.0.0.0', port=port, debug=True)
