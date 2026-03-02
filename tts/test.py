import scipy.io.wavfile
from pocket_tts import TTSModel

tts_model = TTSModel.load_model()
voice_state = tts_model.get_state_for_audio_prompt("./voice.safetensors")
audio = tts_model.generate_audio(voice_state, "And he opens the third file now!")
# Audio is a 1D torch tensor containing PCM data.
scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())
