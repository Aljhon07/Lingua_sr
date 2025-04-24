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
from collections import Counter
from tools import audio as audio_tools

class SpeechDataset(Dataset):
    def __init__(self, data, sample_rate=16000):
        self.tsv_file = os.path.join(config.OUTPUT_PATH, f"{config.LANGUAGE}.tsv")
        self.sample_rate = sample_rate
        self.data = data
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=25)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        spec_path, tokenized_transcript, metadata = self.data[idx]
        audio_path = os.path.join(config.WAVS_PATH, metadata['file_name'] + ".wav")
        
        # test, _, _ = audio_tools.load_audio(audio_path, torchaudio.info(audio_path), 16000)

        spectrogram = torch.load(spec_path) # [Channels, Mel, Time]
        label = tokenized_transcript
        features_len = spectrogram.shape[2]  # Time dimension
        label_len = len(label)
        
        transcription = metadata['transcription']

        # print(f"Loading spectrogram from {spec_path}")
        # print(f"Spectrogram Stats: {spectrogram.shape} | Min: {spectrogram.min()} | Max: {spectrogram.max()} | Mean: {spectrogram.mean()} | Std: {spectrogram.std()} | Contiguous: {spectrogram.is_contiguous()}")
        # print(f"Audio: {audio_path} | Transcription: {transcription} | Tokenized: {tokenized_transcript}")
        return spectrogram, torch.tensor(label, dtype=torch.long), features_len, label_len, metadata['transcription'], audio_path

    def get_total_durations(self):
        total_duration = 0
        for _, _, metadata in self.data:
            total_duration += metadata['duration']
        return total_duration
    
def collate_fn(batch):
    features_batch, labels_batch, features_lens_batch, labels_lens_batch, string_labels, file_name = zip(*batch)
    
    padded_labels_batch = nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=0)
    features_batch = torch.stack(features_batch, dim=0)  # [Batch, Time, Mel]
    features_lens_batch = torch.tensor(features_lens_batch, dtype=torch.long)
    labels_lens_batch = torch.tensor(labels_lens_batch, dtype=torch.long)
    return features_batch, padded_labels_batch, features_lens_batch, labels_lens_batch, string_labels, file_name

def load_data():
    categorized_data = load_data_from_tsv(os.path.join(config.OUTPUT_PATH, f"{config.LANGUAGE}.tsv"))
    sorted_keys = sorted(categorized_data.keys())
    datasets = []
# Iterate over the sorted keys and print the number of samples for each key
    for key in sorted_keys:
        print(f"Key: {key} | Number of samples: {len(categorized_data[key])}")
        datasets.append(SpeechDataset(categorized_data[key]))
        
        
    total_duration = 0
    for dataset in datasets:
        print(f"Dataset Size: {len(dataset)} samples | Duration: {dataset.get_total_durations() / 60 / 60:.2f} hours")
        total_duration = dataset.get_total_durations()
    print(f"Total Duration: {total_duration / 60 / 60:.2f} hours")
    
    # short_dataset = SpeechDataset()
    # medium_dataset = SpeechDataset()
    # long_dataset = SpeechDataset()
    # extended_dataset = SpeechDataset()
    
    
    # print(f"Short Dataset: {len(short_dataset)} samples")
    # print(f"Medium Dataset: {len(medium_dataset)} samples")
    # print(f"Long Dataset: {len(long_dataset)} samples")
    # print(f"Extended Dataset: {len(extended_dataset)} samples")
    # print(f"{(very_short_dataset.total_durations + short_dataset.total_durations + medium_dataset.total_durations + long_dataset.total_durations) / 60:.2f} hours of audio loaded")
    # datasets = [very_short_dataset]
    train_loaders, val_loaders = [], []
    
    for dataset in datasets:
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loaders.append(DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn))
        val_loaders.append(DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn))

    loaders = {'train': train_loaders, 'val': val_loaders}

    for i, (feature, label, feature_len, label_len, string_labels, file_name) in enumerate(train_loaders[0]):
        print(f"Features batch shape: {feature.shape}")
        print(f"Labels batch shape: {label.shape}")
        print(f"Feature lengths: {feature_len}")
        print(f"Label lengths: {label_len}")
        print(f"Sample: {i+1} | String labels: {string_labels}")
        break
        
    return loaders
    return train_loader, val_loader

def load_data_from_tsv(tsv_file):
    if not os.path.exists(tsv_file):
        raise FileNotFoundError(f"TSV file not found at: {tsv_file}")
    
    df = pd.read_csv(tsv_file, sep='\t')
    
    categorized_data = {}
    tokens = []
    for index, row in df.iterrows():
        file_name = row['file_name']
        spec_path = os.path.join(config.OUTPUT_PATH, 'spectrograms', file_name + ".pt")
        
        if not os.path.exists(spec_path):
            continue
        
        padded_duration = row['padded_duration']
        transcription = row['transcription']
        tokenized_transcription = row['tokenized_transcription']
        tokenized_transcription = [int(token) for token in tokenized_transcription.split()]
        
        metadata = {
            'transcription': transcription,
            'tokenized_transcription': tokenized_transcription,
            'tokenzed_transcription_str': row['tokenized_transcription_str'],
            'duration': row['duration'],
            'padded_duration': padded_duration,
            'num_frames': row['num_frames'],
            'sample_rate': row['sample_rate'],
            'file_name': file_name
        }
        categorized_data.setdefault(padded_duration, []).append((spec_path, tokenized_transcription, metadata))
        
    return categorized_data

            
if __name__ == "__main__":
    loaders = load_data()
