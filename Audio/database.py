import numpy as np
import pickle
import librosa 
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, generate_binary_structure, binary_erosion

def save_recording_to_file(recording_dict, filename="recording_data.pkl"):
    """
    Saves a dictionary of recording objects to a file using pickle.
    """
    try:
        with open(filename, "wb") as f:
            pickle.dump(recording_dict, f)
        print(f"Recording data saved to {filename}")
    except Exception as e:
        print(f"Error saving recording data: {e}")


def load_recording_from_file(filename="recording_data.pkl"):
    """
    Loads a dictionary of recording objects from a file using pickle.
    """
    try:
        with open(filename, "rb") as f:
            recording_dict = pickle.load(f)
        print(f"Recording data loaded from {filename}")
        return recording_dict
    except FileNotFoundError:
        print(f"File '{filename}' not found. Returning an empty dictionary.")
        return {}
    except Exception as e:
        print(f"Error loading recording data: {e}")
        return {}


def load_audio_file(filepath: str, sr: int = 22050):
    samples, sample_rate = librosa.load(filepath, sr=sr)
    return samples, sample_rate


def save_audio(samples: np.ndarray, sr: int, filename: str):
    sf.write(filename, samples, sr)


def compute_log_spectrogram(samples, sr):
    n_fft = 2048
    hop_length = 512
    S = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop_length))**2
    log_S = librosa.power_to_db(S)
    return log_S

def extract_peaks(spectrogram, neighborhood_size = 20, threshold_db = -40):
    structure = generate_binary_structure(2,1)
    neighborhood = maximum_filter(spectrogram, footprint=structure, size=neighborhood_size)
    local_max = (spectrogram == neighborhood)
    background = (spectrogram < threshold_db)
    eroded = binary_erosion(local_max, structure=structure, border_value=1)
    peaks = eroded & ~background
    return np.argwhere(peaks)


#sample data for metadata structure 
#anger, disgust, surprise, happiness, sadness, fear
recording_data = {
    'recording_001': {
        #'features': np.random.rand(74, 100),  # 74 COVAREP features extracted from recording for 100 frames (based on CMU MOSEI)
        'emotion_labels': {
            'happiness': 2.0,  # Intensity from 0 to 3 (based on CMU MOSEI)
            'sadness': 0.5,
            'anger': 0.1,
            'surprise': 1.0,
            'disgust': 0.0,
            'fear': 0.0,
        },
        'speaker_id': 'speaker_001',
        'transcription': "I'm so happy with this result!",
        # Other metadata if needed
    },
    'recording_002': {
        #'features': np.random.rand(74, 120),
        'emotion_labels': {
            'happiness': 0.0,
            'sadness': 2.5,
            'anger': 0.8,
            'surprise': 0.0,
            'disgust': 0.0,
            'fear': 0.0,
        },
        'speaker_id': 'speaker_002',
        'transcription': "This news makes me incredibly sad.",
    },
}
