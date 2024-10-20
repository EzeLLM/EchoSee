## TODO: normalizer is having a problem because of nltk.data.path.append("punkt")
## either remove it or fix it
import FuncHub
from nltk.tokenize import sent_tokenize
import nltk
from transformers import BarkModel, AutoProcessor
import torch
from scipy.io import wavfile
import numpy as np
import re
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from TTS.api import TTS as TTS_model
import os
import wave
import io
from contextlib import closing
import logger
import boto3
logger = logger.Logger()
logger.log('TextToSpeech initialized a fresh instance')
nltk.data.path.append("punkt")
class TTS():
    def __init__(self,config) -> None:
        self.config = FuncHub.open_yaml(config,'TTS')
        self.mode = self.config['mode'].lower()
        self.mode_config = self.config['Modes'][self.mode]
        self.SAVE_PATH = self.config['save_path']
        
        # the heavy model is not recommended to run on mps due to attention mask complexities.
        # TODO: fix heavy mps
        if self.mode in ['mid','light']:
            self.device = FuncHub.get_device()
            if self.mode == 'light':
                self.speaker_embedding = self.dynamic_embeddings(self.mode_config['speaker_index'])
                self.synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts",device=self.device)
            elif self.mode == 'mid':
                self.language = self.mode_config['language']
                self.speaker_wav = self.mode_config['speaker_wav_path']
        elif self.mode == 'heavy':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = BarkModel.from_pretrained("suno/bark-small").to(self.device)
            self.processor = AutoProcessor.from_pretrained("suno/bark")
            self.voice_preset = "v2/en_speaker_6"
        elif self.mode == 'api':
            self.region_name = "us-west-2" 
            self.polly = boto3.client("polly", region_name=self.region_name,
                                  aws_access_key_id=os.getenv('AWSKEYID'),
                                    aws_secret_access_key=os.getenv('AWSSECRETKEY'))
            self.voice = self.mode_config['voice']
        else:
            raise ValueError("Invalid mode for TTS. Check yaml config and make sure to follow github repo instructions.")

        if self.mode in ['light','mid','heavy']:
            logger.log(f'tts device: {self.device}')
        logger.log(f'tts mode: {self.mode}')
    def dynamic_embeddings(self, embedding_id):
        embeddings_path = "dev/audio/embeddings"
        saved_embed_list = os.listdir(embeddings_path)
        saved_embed_indices = [int(embed.split("_")[0]) for embed in saved_embed_list]
        if embedding_id in saved_embed_indices:
            speaker_embedding = torch.load(f"{embeddings_path}/{embedding_id}_embedding.pt")
            return speaker_embedding
        else:
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embedding = torch.tensor(embeddings_dataset[embedding_id]["xvector"]).unsqueeze(0)
            torch.save(speaker_embedding, f"{embeddings_path}/{embedding_id}_embedding.pt")
            return speaker_embedding

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
        # TODO fix
        sentences = sent_tokenize(text)
        complete_sentences = [s for s in sentences if s.strip() in '.!?']
        result = ' '.join(complete_sentences)
        return result
    
    def echofy_heavy(self,text):
        normalized_text = text
        text_chunks = self.split_text_into_chunks(normalized_text)
        inputs = self.processor(text_chunks, voice_preset=self.voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}   
        with torch.no_grad():
            speech_outputs = self.model.generate(**inputs)
        audio_outputs = [output.cpu().numpy() for output in speech_outputs]
        concatenated_audio = np.concatenate(audio_outputs)
        sampling_rate = self.model.generation_config.sample_rate
        wavfile.write(self.SAVE_PATH, sampling_rate, concatenated_audio)
    

    def echofy_mid(self,text):
        tts = TTS_model("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        tts.tts_to_file(text=text, speaker_wav=self.speaker_wav, language=self.language, file_path=self.SAVE_PATH)

    def echofy_light(self,text):
        speech = self.synthesiser(text, forward_params={"speaker_embeddings": self.speaker_embedding})
        sf.write(self.SAVE_PATH, speech["audio"], samplerate=speech["sampling_rate"])

    def echofy_api(self, text):
        response = self.polly.synthesize_speech(Text=text, OutputFormat="pcm", VoiceId=self.voice)
        if 'AudioStream' in response:
            with closing(response['AudioStream']) as stream:
                audio_data = stream.read()
            
            wav_file = wave.open(self.SAVE_PATH, 'wb')
            wav_file.setnchannels(1)  
            wav_file.setsampwidth(2) 
            wav_file.setframerate(16000) 
            wav_file.writeframes(audio_data)
            wav_file.close()
        
    def echofy(self,text):
        if self.mode == 'light':
            self.echofy_light(text)
        elif self.mode == 'heavy':
            self.echofy_heavy(text)
        elif self.mode == 'mid':
            self.echofy_mid(text)
        elif self.mode == 'api':
            self.echofy_api(text)

if __name__ == "__main__":
    tts = TTS()
    text = "You dare to ask me about that fat cat Elon Musk taking your car? Listen, pal, if you're so worried about Elon, then maybe you should've thought twice before buying one of his electric cars. Fact is, you paid for that thing, and it's not my problem. So, go talk to Elon, not me. He's the one who sold you the car, and he's the one you're supposed to be mad at. Not me. Now stop bothering me. However, if you're asking me how to get your car back from a place as remote as possible then go to hell."
    tts.echofy(text)
    print("Speech saved as 'generated_speech_bark.wav'")
    