import FuncHub
from nltk.tokenize import sent_tokenize
import nltk
from transformers import BarkModel, AutoProcessor
import torch
from scipy.io import wavfile
import numpy as np
import re
nltk.data.path.append("punkt")
class TTS():
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BarkModel.from_pretrained("suno/bark-small").to(self.device)
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.voice_preset = "v2/en_speaker_6"

    def split_text_into_chunks(self, text, max_chunk_length=200):
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
    
    def normalizer(self,text):
        sentences = sent_tokenize(text)
        complete_sentences = [s for s in sentences if s.strip() in '.!?']
        result = ' '.join(complete_sentences)
        return result
    
    def echofy(self,text):
        normalized_text = text
        text_chunks = self.split_text_into_chunks(normalized_text)
        inputs = self.processor(text_chunks, voice_preset=self.voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}   
        with torch.no_grad():
            speech_outputs = self.model.generate(**inputs)
        audio_outputs = [output.cpu().numpy() for output in speech_outputs]
        concatenated_audio = np.concatenate(audio_outputs)
        sampling_rate = self.model.generation_config.sample_rate
        wavfile.write("dev/audio/generated_speech_bark.wav", sampling_rate, concatenated_audio)
        return concatenated_audio, sampling_rate

if __name__ == "__main__":
    tts = TTS()
    text = "You dare to ask me about that fat cat Elon Musk taking your car? Listen, pal, if you're so worried about Elon, then maybe you should've thought twice before buying one of his electric cars. Fact is, you paid for that thing, and it's not my problem. So, go talk to Elon, not me. He's the one who sold you the car, and he's the one you're supposed to be mad at. Not me. Now stop bothering me. However, if you're asking me how to get your car back from a place as remote as possible then go to hell."
    audio, sampling_rate = tts.echofy(text)
    print("Speech saved as 'generated_speech_bark.wav'")
    