import sounddevice as sd
from scipy.io.wavfile import write 


duration = 5
sample_rate = 44100

print("recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels = 1, dtype='int16')
sd.wait()
print("recording complete")

write("output.wav", sample_rate, audio)
print("saved as output.wav")


