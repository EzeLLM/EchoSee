from transformers import BarkModel, AutoProcessor
import torch
from scipy.io import wavfile
import numpy as np
import re

# Initialize the model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkModel.from_pretrained("suno/bark-small").to(device)
processor = AutoProcessor.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

# Your original text prompt
text_prompt = """ You dare to ask me about that fat cat Elon Musk taking your car? Listen, pal, if you're so worried about Elon, then maybe you should've thought twice before buying one of his electric cars. Fact is, you paid for that thing, and it's not my problem. So, go talk to Elon, not me. He's the one who sold you the car, and he's the one you're supposed to be mad at. Not me. Now stop bothering me.

However, if you're asking me how to get your car back from a place as remote as possible then go to hell."""

# Function to split text into chunks based on sentences and max characters
def split_text_into_chunks(text, max_chunk_length=200):
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text)

    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(sentence) > max_chunk_length:
            for i in range(0, len(sentence), max_chunk_length):
                chunks.append(sentence[i:i+max_chunk_length])
        elif len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
            current_chunk = current_chunk + ' ' + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Split the text into chunks
text_chunks = split_text_into_chunks(text_prompt)

# Process all chunks in a single batch
inputs = processor(text_chunks, voice_preset=voice_preset, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate speech for all chunks at once
with torch.no_grad():
    speech_outputs = model.generate(**inputs)

# Collect and concatenate audio outputs
audio_outputs = [output.cpu().numpy() for output in speech_outputs]
concatenated_audio = np.concatenate(audio_outputs)

# Save the concatenated audio as a WAV file
sampling_rate = model.generation_config.sample_rate
wavfile.write("generated_speech.wav", sampling_rate, concatenated_audio)

print("Speech saved as 'generated_speech.wav'")
from IPython.display import Audio

sampling_rate = model.generation_config.sample_rate
Audio(concatenated_audio, rate=sampling_rate)
