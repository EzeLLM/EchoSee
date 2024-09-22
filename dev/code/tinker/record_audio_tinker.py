import pyaudio
import wave
import numpy as np

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    THRESHOLD = 500  # Adjust this value to set sensitivity
    WAIT_TIME = 150  # Adjust this value to set how long to wait before stopping
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

        if audio_started and silent_chunks > WAIT_TIME:  # Adjust this value to set how long to wait before stopping
            break

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open("output.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

record_audio()
