from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch
import torchaudio
import torchaudio.functional as F
import config
import os
import matplotlib.pyplot as plt

class SpeechDataset(Dataset):
    def __init__(self, sample_rate=16000):
        self.tsv_file = os.path.join(config.OUTPUT_PATH, f"{config.LANGUAGE}.tsv")
        self.sample_rate = sample_rate
        self.audio_paths = []
        self.labels = []


        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,            
            win_length=None,
            hop_length=160,         
            n_mels=80
        )
        self._load_data_from_tsv()
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(config.WAVS_PATH, self.audio_paths[idx] + ".wav")
        
        waveform = self._load_audio(audio_path).contiguous()
        trimmed_spectogram = F.vad(waveform, self.sample_rate)
        if trimmed_spectogram.numel() == 0:
             trimmed_spectogram = waveform
        orig_spectogram = self.mel_spectrogram(trimmed_spectogram).contiguous()
        orig_spectogram = F.amplitude_to_DB(
            orig_spectogram,
            multiplier=10.0,      
            amin=1e-10,         
            db_multiplier=1.0,  
            top_db=80.0           
        )
        spectogram = orig_spectogram.contiguous()
        label = self.labels[idx]
        features_len = spectogram.size(-1)
        label_len = len(label)
        
        spectogram = spectogram.squeeze(0)
        spectogram = (spectogram - spectogram.mean(dim=1, keepdim=True)) / (spectogram.std(dim=1, keepdim=True)+ 1e-8)
        spectogram = spectogram.transpose(0, 1).contiguous()
        return spectogram, label, features_len, label_len
    
    def normalize(self, waveform):
        mean = waveform.mean()
        std = waveform.std()
        waveform = (waveform - mean) / (std + 1e-8)
        return waveform
    
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

from torch.utils.data import random_split

def load_data():
    dataset = SpeechDataset()
    
    total_size = len(dataset)
    quarter_size = 100
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
