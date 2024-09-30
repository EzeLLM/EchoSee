from time import sleep
import FuncHub
from SpeechToText import SpeechToText
from LLMInference import LLMInference
from TextToSpeech import TTS
from pydub import AudioSegment
from pydub.playback import play
class EchoSee:
    def __init__(self, config):
        self.config = FuncHub.open_yaml(config,'EchoSee')
        self.tts = TTS(config=config)
        self.llm = LLMInference(config=config)
        self.stt = SpeechToText(config=config)
        self.output_path = FuncHub.open_yaml(config,'TTS')['save_path']
    
    def echosee(self):
        while True:
            sleep(1)
            beep_sound = AudioSegment.from_file("dev/audio/sound_effects/sound_effect_1.mp3", format="mp3")
            play(beep_sound)
            user_input = self.stt.stt()
            print("\nYou: ",user_input)
            # ## for dev , delete later
            # user_input = "Wanna go to the park?"
            result = self.llm.llm(user_input)
            print("\nModel: ",result)
            self.tts.echofy(result)
            sound = AudioSegment.from_file(self.output_path, format="wav")
            play(sound)
            print("\nEchoSee: ",result)



            
if __name__ == "__main__":
    echosee = EchoSee('dev/code/config/echosee.yaml')
    echosee.echosee()
