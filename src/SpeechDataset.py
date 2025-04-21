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
from tools import audio as audio_tools

class SpeechDataset(Dataset):
    def __init__(self, sample_rate=16000, duration_type='very_short'):
        self.tsv_file = os.path.join(config.OUTPUT_PATH, f"{config.LANGUAGE}.tsv")
        self.sample_rate = sample_rate
        self.data = []
        
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)
        
        self._load_data_from_tsv(duration_type)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        spec_path, tokenized_transcription, metadata = self.data[idx]
        audio_path = os.path.join(config.WAVS_PATH, metadata['file_name'] + ".wav")
        
        # test, _, _ = audio_tools.load_audio(audio_path, torchaudio.info(audio_path), 16000)

        spectrogram = torch.load(spec_path) # [Channels, Mel, Time]
        label = [int(token) for token in tokenized_transcription.split()]
        features_len = spectrogram.shape[2]  # Time dimension
        label_len = len(label)
        
        transcription = metadata['transcription']
    
        # print(f"Loading spectrogram from {spec_path}")
        # print(f"Spectrogram Stats: {spectrogram.shape} | Min: {spectrogram.min()} | Max: {spectrogram.max()} | Mean: {spectrogram.mean()} | Std: {spectrogram.std()} | Contiguous: {spectrogram.is_contiguous()}")
        # print(f"Audio: {audio_path} | Transcription: {transcription} | Tokenized: {tokenized_transcription}")
              
        return spectrogram, torch.tensor(label, dtype=torch.long), features_len, label_len, metadata['transcription'], audio_path

    def _load_data_from_tsv(self, duration_type):
        if not os.path.exists(self.tsv_file):
            raise FileNotFoundError(f"TSV file not found at: {self.tsv_file}")
        
        df = pd.read_csv(self.tsv_file, sep='\t')
        
        for index, row in df.iterrows():
            tokenized_transcript = list(map(int, row['tokenized_transcription'].split()))
            file_name = row['file_name']
            spec_path = os.path.join(config.OUTPUT_PATH, 'spectrograms', file_name + ".pt")
            
            
            if not os.path.exists(spec_path):
                print(f"Spectrogram file {spec_path} does not exist. Skipping.")
                continue
            
            category = row['duration_type']
            
            if category != duration_type:
                continue

            duration = row['duration']
            num_frames = row['num_frames']
            sr = row['sample_rate']
            transcription = row['transcription']
            tokenized_transcription = row['tokenized_transcription']
            tokenized_transcription_str = row['tokenized_transcription_str']
            
            metadata = {
                'transcription': transcription,
                'tokenized_transcription': tokenized_transcription,
                'tokenzed_transcription_str': tokenized_transcription_str,
                'duration': duration,
                'duration_type': category,
                'num_frames': num_frames,
                'sample_rate': sr,
                'file_name': file_name
            }
            self.data.append((spec_path, tokenized_transcription, metadata))
            
def collate_fn(batch):
    features_batch, labels_batch, features_lens_batch, labels_lens_batch, string_labels, file_name = zip(*batch)
    
    padded_labels_batch = nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=0)
    features_batch = torch.stack(features_batch, dim=0)  # [Batch, Time, Mel]
    features_lens_batch = torch.tensor(features_lens_batch, dtype=torch.long)
    labels_lens_batch = torch.tensor(labels_lens_batch, dtype=torch.long)
    return features_batch, padded_labels_batch, features_lens_batch, labels_lens_batch, string_labels, file_name

def load_data():
    very_short_dataset = SpeechDataset(duration_type='very_short')
    short_dataset = SpeechDataset(duration_type='short')
    print(f"Very short dataset size: {len(very_short_dataset)}")

    datasets = [very_short_dataset, short_dataset]
    train_loaders, val_loaders = [], []
    
    for dataset in datasets:
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loaders.append(DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn))
        val_loaders.append(DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn))

    loaders = {'train': train_loaders, 'val': val_loaders}

    return loaders
    # return train_loader, val_loader

if __name__ == "__main__":
    loaders = load_data()
