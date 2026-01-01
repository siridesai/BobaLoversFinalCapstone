import sounddevice as sd
from scipy.io.wavfile import write
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import torchaudio

# Record voice
duration = 5  # seconds
sample_rate = 44100
print("Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()
print("Recording complete")
write("output.wav", sample_rate, audio)
print("Saved as output.wav")

# Predict emotion
def predict_emotion(test_file):
    model_path = "./final folder"
    model = AutoModelForAudioClassification.from_pretrained(model_path)
    extractor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
    model.eval()

    # Manual label fix
    model.config.id2label = {
        0: "angry",
        1: "disgust",
        2: "fearful",
        3: "happy",
        4: "sad",
        5: "surprised"
    }

    waveform, sr = torchaudio.load(test_file)
    waveform = waveform.mean(dim=0)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=-1).item()
    return model.config.id2label[pred_id]

# Run prediction
emotion = predict_emotion("output.wav")
print("Predicted Emotion:", emotion)
