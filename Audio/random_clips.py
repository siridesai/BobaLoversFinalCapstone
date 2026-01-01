import numpy as np 
from typing import List
import librosa
import matplotlib.pyplot as plt
from typing import List
from IPython.display import Audio

def generate_random_clips(samples: np.ndarray, clip_length: int, sample_rate: int, num_clips: int) -> List[np.ndarray]:
    """
    Returns a list of random audio clips from a longer input signal.

    Parameters:
        samples (np.ndarray): 1D array of audio samples.
        clip_length (int): Length of each clip in seconds.
        sample_rate (int): Sampling rate in Hz.
        num_clips (int): Number of clips to generate.

    Returns:
        List[np.ndarray]: List of audio clips as NumPy arrays.

    Raises:
        ValueError: If clip length is longer than the input audio.
    """
    total_samples = len(samples) 
    clip_samples = int(clip_length * sample_rate) # 
    clips = []

    if total_samples < clip_samples:
        raise ValueError("Clip Length is longer than input audio")
    
    for i in range(num_clips):
        start = np.random.randint(0,total_samples - clip_samples)
        clip = samples[start:start + clip_samples]
        clips.append(clip)

    return clips

    