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
from collections import defaultdict

class SpeechDataset(Dataset):
    def __init__(self, sample_rate=16000):
        self.tsv_file = os.path.join(config.OUTPUT_PATH, f"{config.LANGUAGE}.tsv")
        self.sample_rate = sample_rate
        self.spectrograms = []
        self.labels = []
        
        self.audio_paths = []
        self.string_labels = []
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,            
            win_length=400,
            hop_length=160,         
            n_mels=80,
            normalized=True,
        )
        
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)
        
        self.amplitude = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=100)
        self._load_data_from_tsv()
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        # print(f"Spectrogram Shape: {self.spectrograms[idx].shape}")
        label = self.labels[idx]
        features_len = self.spectrograms[idx].size(0)
        label_len = len(label)
        
        # print(f"Audio Name: {self.audio_paths[idx]} / Label: {label} / Label Length: {label_len} / Features Shape: {self.spectrograms[idx].shape}")
        # print(f"{self.string_labels[idx]}")
        # print(f"Min: {self.spectrograms[idx].min()} / Max: {self.spectrograms[idx].max()} / Mean: {self.spectrograms[idx].mean()} / Std: {self.spectrograms[idx].std()}")
        
              
        return self.spectrograms[idx], torch.tensor(label), features_len, label_len, self.string_labels[idx], self.audio_paths[idx]

    def _load_data_from_tsv(self):
        if not os.path.exists(self.tsv_file):
            raise FileNotFoundError(f"TSV file not found at: {self.tsv_file}")
        
        df = pd.read_csv(self.tsv_file, sep='\t')
        duration_buckets = defaultdict(int)
        for index, row in df.iterrows():
            tokenized_transcript = list(map(int, row['tokenized_transcription'].split()))
            audio_path = os.path.join(config.WAVS_PATH, row['file_name'] + ".wav")
            
            if not os.path.exists(audio_path):
                print(f"Audio file not found at: {audio_path}")
                continue
            
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            
            short = duration > 6 and duration < 7
            medium = duration > 7 and duration < 8
            long = duration > 8 and duration < 10
            target_frames = None
            
            if duration > 4 and duration < 6.25:
                spectrogram, save = self._load_audio(audio_path)
                
                if save:
                    print(f"Appended: Audio path: {audio_path} / Spectrogram Shape: {spectrogram.shape}")
                    self.spectrograms.append(spectrogram)
                    self.audio_paths.append(audio_path)
                    self.labels.append(tokenized_transcript)
                    self.string_labels.append(row['transcription'])
                    
        print("Loaded")
        
    def _load_audio(self, audio_path, apply_augmentation=False):
        save = False
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        
        info = torchaudio.info(audio_path)

        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            print(f"Resampling {audio_path} to {self.sample_rate} Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        waveform = waveform.contiguous()
        print(f"Waveform Stats: Min: {waveform.min()} / Max: {waveform.max()} / Mean: {waveform.mean()} / Std: {waveform.std()}")
            
        trimmed_waveform = F.vad(waveform, self.sample_rate)
        trimmed_duration = trimmed_waveform.size(1) / self.sample_rate

        if trimmed_duration >= 4.5 and trimmed_duration <= 5:
            save = True
        else:
            return None, False

        current_length = trimmed_waveform.size(1)
        padding = (5 * 16000) - current_length
        trimmed_waveform = torch.nn.functional.pad(trimmed_waveform, (0, padding))

        # print(f"Audio path: {audio_path} / Duration: {info.num_frames / info.sample_rate} / Waveform: {waveform.size(1) / info.sample_rate} / Trimmed: {trimmed_waveform.size(1) / info.sample_rate} / Sample rate: {self.sample_rate}")
        # print(f"Audio: {audio_path} / Duration: {trimmed_waveform.size(1) / info.sample_rate}")
        initial_spectrogram = self.mel_spectrogram(trimmed_waveform).contiguous()
        # print(f"Initial Spectrogram Stats: Min: {initial_spectrogram.min()} / Max: {initial_spectrogram.max()} / Mean: {initial_spectrogram.mean()} / Std: {initial_spectrogram.std()}")
        if apply_augmentation:
            initial_spectrogram = self.freq_mask(initial_spectrogram).contiguous()
            initial_spectrogram = self.time_mask(initial_spectrogram).contiguous()
        
        initial_spectrogram = self.amplitude(initial_spectrogram)
        
        spectrogram = initial_spectrogram.contiguous()
        # print(f"Spectrogram Stats: Min: {spectrogram.min()} / Max: {spectrogram.max()} / Mean: {spectrogram.mean()} / Std: {spectrogram.std()}")
        spectrogram_min = spectrogram.min()
        spectrogram_max = spectrogram.max()

        # Normalize the spectrogram
        spectrogram = (spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min)
        # print(f"Normalize Stats: Min: {spectrogram.min()} / Max: {spectrogram.max()} / Mean: {spectrogram.mean()} / Std: {spectrogram.std()}")
        # print(f"Min: {spectrogram.min()} / Max: {spectrogram.max()} / Mean: {spectrogram.mean()} / Std: {spectrogram.std()}")
        
        # plot_spectrogram(initial_spectrogram, spectrogram, sample_rate=self.sample_rate)
        spectrogram = spectrogram.squeeze(0).transpose(0, 1).contiguous()
        return spectrogram, save
    
import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(orig, normalized, sample_rate=16000):
    # Plot the raw (unnormalized) spectrogram
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convert tensors to numpy arrays
    orig = orig.squeeze(0).cpu().numpy()
    normalized = normalized.squeeze(0).cpu().numpy()
    
    # Plot the raw (unnormalized) spectrogram
    im0 = axs[0, 0].imshow(orig, aspect='auto', origin='lower', cmap='viridis')
    axs[0, 0].set_title('Raw Mel Spectrogram')
    axs[0, 0].set_xlabel('Time Frames')
    axs[0, 0].set_ylabel('Mel Bands')
    fig.colorbar(im0, ax=axs[0, 0], format="%+2.0f dB")

    # Plot the normalized spectrogram
    im1 = axs[0, 1].imshow(normalized, aspect='auto', origin='lower', cmap='viridis')
    axs[0, 1].set_title('Normalized Mel Spectrogram')
    axs[0, 1].set_xlabel('Time Frames')
    axs[0, 1].set_ylabel('Mel Bands')
    fig.colorbar(im1, ax=axs[0, 1], format="%+2.0f dB")

    # Plot the distribution (histogram) of the raw spectrogram values
    axs[1, 0].hist(orig.flatten(), bins=80, color='blue', alpha=0.7)
    axs[1, 0].set_title('Distribution of Raw Mel Spectrogram')
    axs[1, 0].set_xlabel('Amplitude')
    axs[1, 0].set_ylabel('Frequency')

    # Plot the distribution (histogram) of the normalized spectrogram values
    axs[1, 1].hist(normalized.flatten(), bins=80, color='green', alpha=0.7)
    axs[1, 1].set_title('Distribution of Normalized Mel Spectrogram')
    axs[1, 1].set_xlabel('Amplitude')
    axs[1, 1].set_ylabel('Frequency')

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

        
def plot_waveforms (waveform, trimmed):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(waveform.squeeze().numpy())
    plt.title(f"Waveform 1 - {len(waveform.squeeze())/16000:.2f}s")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    plt.subplot(1, 2, 2)
    plt.plot(trimmed.squeeze().numpy())
    plt.title(f"Waveform 2 - {len(trimmed.squeeze())/16000:.2f}s")
    plt.xlabel("Samples")

    plt.tight_layout()
    plt.show()
    
def plot_comparison(padded, unpadded, sample_idx=0):
    padded_sample = padded[sample_idx].squeeze(0).transpose(0,1).cpu().numpy()  # Padded (same sample as unpadded, we will manually pad it)
    max_length = padded_sample.shape[0]
    unpadded_sample = unpadded[sample_idx].squeeze(0).transpose(0,1).cpu().numpy()  # Padded (same sample as unpadded, we will manually pad it)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot unpadded spectrogram
    im1 = axes[0].imshow(unpadded_sample, aspect='auto', origin='lower', cmap='inferno')
    axes[0].set_title(f"Unpadded Mel Spectrogram - Sample {sample_idx}")
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Mel Frequency Bins')
    plt.colorbar(im1, ax=axes[0], format="%+2.0f dB")  # Add colorbar to the unpadded plot

    # Plot padded spectrogram
    im2 = axes[1].imshow(padded_sample, aspect='auto', origin='lower', cmap='inferno')
    axes[1].set_title(f"Padded Mel Spectrogram - Sample {sample_idx}")
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Mel Frequency Bins')
    plt.colorbar(im2, ax=axes[1], format="%+2.0f dB")  # Add colorbar to the padded plot
    
    plt.tight_layout()
    plt.show()

def collate_fn(batch):
    features_batch, labels_batch, features_lens_batch, labels_lens_batch, string_labels, audio_paths = zip(*batch)

    for idx in range(len(features_batch)):
        print(f"Audio Path: {audio_paths[idx]} / Label: {string_labels[idx]} / Features Shape: {features_batch[idx].shape}")
        break

    # padded_features_batch = nn.utils.rnn.pad_sequence(features_batch, batch_first=True , padding_value=0)
    # padded_labels_batch = nn.utils.rnn.pad_sequence([torch.tensor(label) for label in labels_batch], batch_first=True, padding_value=0)
    features_batch = torch.stack(features_batch, dim=0)  # [Batch, Time, Mel]
    flattened_labels = torch.tensor([label for seq in labels_batch for label in seq])
    features_lens_batch = torch.tensor(features_lens_batch)
    labels_lens_batch = torch.tensor(labels_lens_batch)
    # plot_comparison(padded_features_batch, features_batch ,sample_idx=0)
    return features_batch, flattened_labels, features_lens_batch, labels_lens_batch

def load_data():
    dataset = SpeechDataset()
    
    total_size = len(dataset)
    quarter_size = total_size
    dataset_subset = torch.utils.data.Subset(dataset, range(quarter_size))

    train_size = int(0.9 * quarter_size)
    val_size = quarter_size - train_size

    train_dataset, val_dataset = random_split(dataset_subset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    for batch_idx, (features, labels, features_len, labels_len) in enumerate(train_loader):
        print(f"Features Shape: {features.shape}")
        print(f"Labels: {labels[:25]} / {labels.shape}")
        print(f"Features Length: {features_len} / {features_len.shape}")
        print(f"Labels Length: {labels_len} / {labels_len.shape}")
        break 
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = load_data()
