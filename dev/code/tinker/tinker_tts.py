from transformers import BarkModel, AutoProcessor
import torch
from scipy.io import wavfile
import numpy as np

# Initialize the model and processor
device = "cpu"
model = BarkModel.from_pretrained("suno/bark").to(device)
processor = AutoProcessor.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

# Your original text prompt
text_prompt = """[clears throat] Anyway, when I'm not gaming, I enjoy listening to music...  'I'm just a poor gamer, I need no sympathy...'  [laughs] Okay, maybe I need a little sympathy. [chuckles] But seriously, music helps me relax and focus. MAN: So, what about you? What's your go-to hobby?"""

# Function to split text into chunks of up to 64 tokens
def split_text_into_chunks(text, max_tokens=64):
    # Tokenize the text (approximate tokenization by splitting by spaces)
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        # Estimate token count (you might want to adjust this for more accurate token counts)
        token_count = len(processor(' '.join(current_chunk))['input_ids'])
        if token_count >= max_tokens:
            # Remove the last word and save the chunk
            current_chunk.pop()
            chunks.append(' '.join(current_chunk))
            # Start a new chunk with the current word
            current_chunk = [word]

    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Split the text into chunks
text_chunks = split_text_into_chunks(text_prompt)

# Process each chunk and collect the audio outputs
audio_outputs = []
for idx, chunk in enumerate(text_chunks):
    print(f"Processing chunk {idx+1}/{len(text_chunks)}")
    inputs = processor(chunk, voice_preset=voice_preset)
    # Generate speech for the chunk
    with torch.no_grad():
        speech_output = model.generate(**inputs.to(device))
    # Append the audio data
    audio_outputs.append(speech_output[0].cpu().numpy())

# Concatenate all audio data
concatenated_audio = np.concatenate(audio_outputs)

# Save the concatenated audio as a WAV file
sampling_rate = model.generation_config.sample_rate
wavfile.write("generated_speech.wav", sampling_rate, concatenated_audio)

print("Speech saved as 'generated_speech.wav'")
