from melo.api import TTS

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = 'auto' # Will automatically use GPU if available

# English 
text = "Did you ever hear a folk tale about a giant turtle?"
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

# # American accent
# output_path = 'en-us.wav'
# model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed)

# # British accent
# output_path = 'en-br.wav'
# model.tts_to_file(text, speaker_ids['EN-BR'], output_path, speed=speed)

# # Default accent
# output_path = 'en-default.wav'
# model.tts_to_file(text, speaker_ids['EN-Default'], output_path, speed=speed)
