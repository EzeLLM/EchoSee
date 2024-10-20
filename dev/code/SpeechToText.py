import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyaudio
import wave
import numpy as np
import FuncHub
import logger
import time
import os
import ffmpeg
from groq import Groq
logger = logger.Logger()
logger.log('SpeechToText initialized a fresh instance')
class SpeechToText():
    def __init__(self,config) -> None:
        logger.log('SpeechToText initialized a fresh instance')
        self.output_path = os.path.join('dev/audio','output.wav')
        self.output_reduced_path = os.path.join('dev/audio','output_reduced.wav')
        self.config = FuncHub.open_yaml(config,'SpeechToText')
        self.mode = self.config['mode'].lower()
        self.mode_config = self.config['Modes'][self.mode]
        self.ChunkSize = self.config['ChunkSize']
        self.Rate = self.config['Rate']
        self.Threshold = self.config['Threshold']
        self.WaitTime = self.config['WaitTime']
        self.Channels = self.config['Channels']
        self.device = FuncHub.get_device() # auto select device
        print(self.device)
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if self.mode == 'local':

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                'openai/whisper-large-v3', torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(self.model_name)
        elif self.mode  == 'groq':
            self.client = Groq(
                api_key=os.getenv('GROQAPI'),
            )


        self.output_path = os.path.join('dev/audio','output.wav')

    def transcribe_groq(self,audio_file):
        ffmpeg.input(audio_file).output(self.output_reduced_path, ar=16000, ac=1).overwrite_output().run()
        with open(self.output_reduced_path, "rb") as file:
            transcription = self.client.audio.transcriptions.create(
            file=(self.output_reduced_path, file.read()), 
            model='whisper-large-v3-turbo',
            )
            return transcription.text

    def transcribe_local(self,audio_file):
        device = FuncHub.get_device() # auto select device
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(self.model_name)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )


        result = pipe(audio_file)

        return result["text"]
    
    def transcribe(self,audio_file):
        if self.mode == 'local':
            return self.transcribe_local(audio_file)
        elif self.mode == 'groq':
            return self.transcribe_groq(audio_file)

    # function that records audio from microphone and returns it as a file
    def record_audio(self,write=False):
        CHUNK = self.ChunkSize
        FORMAT = pyaudio.paInt16
        CHANNELS = self.Channels
        RATE = self.Rate
        THRESHOLD = self.Threshold  # Adjust this value to set sensitivity
        WAIT_TIME = self.WaitTime  # Adjust this value to set how long to wait before stopping
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("Listening...")

        frames = []
        silent_chunks = 0
        audio_started = False

        while True:
            data = stream.read(CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            if np.abs(audio_data).mean() > THRESHOLD:
                if not audio_started:
                    print("Recording started...")
                    audio_started = True
                silent_chunks = 0
                frames.append(data)
            elif audio_started:
                silent_chunks += 1
                frames.append(data)

            if audio_started and silent_chunks > WAIT_TIME:
                break

        print("Recording finished.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Convert frames to a single byte string
        recorded_data = b''.join(frames)
        
        if write:
            start_write = time.time()
            # Save the recorded data as a WAV file
            wf = wave.open(self.output_path, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(recorded_data)
            wf.close()
            logger.log(f"File save time in milliseconds: {time.time()-start_write}")


        #recorded_data = b''.join(frames)

        return recorded_data, CHANNELS, p.get_sample_size(FORMAT), RATE

    def stt(self):
        _, _, _, _ = self.record_audio(write=True)
        time_start = time.time()
        result = self.transcribe(self.output_path)
        logger.log(f"Time taken to transcribe in milliseconds: {time.time()-time_start}")
        logger.log(f'Transcription: {result}')
        return result

    
if __name__ == '__main__':
    STT = SpeechToText()
    STT.main()
            