import torch
import torchaudio
import matplotlib.pyplot as plt

class AudioProcessor:
    def __init__(self, sample_rate=16000, normalize=True):
        self.sample_rate = sample_rate
        self.normalize = normalize

    def load(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        print(sr)
        if sr != self.sample_rate:
            print(f"Resampling from {audio_path}...")
            self.orig_freq = sr
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform

    def plot_waveform(self, waveform):
        plt.figure(figsize=(10, 4))
        plt.plot(waveform.numpy())  
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title('Audio Waveform')
        plt.grid(True)
        plt.show()