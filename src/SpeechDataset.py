import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import librosa
import config
from time import sleep

class SpeechDataset(Dataset):
    def __init__(self, sample_rate=16000):
        self.tsv_file = os.path.join(config.OUTPUT_PATH, f"{config.LANGUAGE}.tsv")
        self.sample_rate = sample_rate
        self.audio_paths = []
        self.labels = []


        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,            
            win_length=400,
            hop_length=160,         
            n_mels=80,
            normalized=True,
        )
        
        self.amplitude = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        self._load_data_from_tsv()
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(config.WAVS_PATH, self.audio_paths[idx] + ".wav")
        # audio_path = os.path.join(config.WAVS_PATH, "common_voice_en_41435787.wav")
        waveform = self._load_audio(audio_path).contiguous()
        trimmed_spectogram = F.vad(waveform, self.sample_rate)
        if trimmed_spectogram.numel() == 0:
             trimmed_spectogram = waveform
        orig_spectrogram = self.mel_spectrogram(waveform).contiguous()
        orig_spectrogram = self.amplitude(orig_spectrogram)


        spectrogram = orig_spectrogram.contiguous()
        label = self.labels[idx]
        features_len = spectrogram.size(-1)
        label_len = len(label)
        
        if spectrogram.min() < -80 or spectrogram.max() > 80:
            spectrogram = torch.clamp(spectrogram, min=-80, max=80)
            sleep(0.5)
            # self.plot_original_vs_normalized(spectrogram, clipped_spectrogram)
            # raise ValueError(f"Spectrogram values out of range: {spectrogram.min()} to {spectrogram.max()}")
        
        spectrogram = spectrogram.squeeze(0).transpose(0, 1).contiguous()
        return spectrogram, label, features_len, label_len

    def plot_original_vs_normalized(self, original_spectrogram, normalized_spectrogram):
        """Plots the original and normalized spectrograms side-by-side."""

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

        # Original Spectrogram
        im1 = axes[0].imshow(original_spectrogram.squeeze(0), cmap='inferno', origin='lower', aspect='auto', interpolation='none')
        axes[0].set_title("Original Mel Spectrogram")
        axes[0].set_ylabel("Frequency (Mel Bands)")
        axes[0].set_xlabel("Time (Frames)")
        fig.colorbar(im1, ax=axes[0])  # Use default colorbar for original

        # Normalized Spectrogram
        im2 = axes[1].imshow(normalized_spectrogram.squeeze(0), cmap='inferno', origin='lower', aspect='auto', interpolation='none')
        axes[1].set_title("Normalized Mel Spectrogram")
        axes[1].set_ylabel("Frequency (Mel Bands)")
        axes[1].set_xlabel("Time (Frames)")
        fig.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()
    
    def _load_data_from_tsv(self):
        if not os.path.exists(self.tsv_file):
            raise FileNotFoundError(f"TSV file not found at: {self.tsv_file}")
        df = pd.read_csv(self.tsv_file, sep='\t')

        for index, row in df.iterrows():
            tokenized_transcript = list(map(int, row['tokenized_transcription'].split()))
            self.audio_paths.append(row['file_name'])
            self.labels.append(tokenized_transcript)
            
    def _load_audio(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            print(f"Resampling {audio_path} to {self.sample_rate} Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        
        return waveform
    


def collate_fn(batch):
    features_batch, labels_batch, features_lens_batch, labels_lens_batch = zip(*batch)

    padded_features_batch = nn.utils.rnn.pad_sequence(features_batch, batch_first=True, padding_value=-80)
    padded_labels_batch = nn.utils.rnn.pad_sequence([torch.tensor(label) for label in labels_batch], batch_first=True, padding_value=0)
    features_lens_batch = torch.tensor(features_lens_batch)
    labels_lens_batch = torch.tensor(labels_lens_batch)
    
    return padded_features_batch, padded_labels_batch, features_lens_batch, labels_lens_batch


def load_data():
    dataset = SpeechDataset()
    
    total_size = len(dataset)
    quarter_size = 1000
    dataset_subset = torch.utils.data.Subset(dataset, range(quarter_size))

    train_size = int(0.8 * quarter_size)
    val_size = quarter_size - train_size

    train_dataset, val_dataset = random_split(dataset_subset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

     
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = load_data()
