from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch
import torchaudio
import torchaudio.functional as F
import config
import os

class SpeechDataset(Dataset):
    def __init__(self, sample_rate=16000):
        self.tsv_file = os.path.join(config.OUTPUT_PATH, f"{config.LANGUAGE}.tsv")
        self.sample_rate = sample_rate
        self.audio_paths = []
        self.labels = []


        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,              # window size
            win_length=None,
            hop_length=160,         # how much it shifts per frame (10ms)
            n_mels=80
        )
        self._load_data_from_tsv()
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(config.WAVS_PATH, self.audio_paths[idx] + ".wav")
        waveform = self._load_audio(audio_path)
        
        features = self.mel_spectrogram(waveform)

        features = F.amplitude_to_DB(
            features,
            multiplier=10.0,      
            amin=1e-10,         
            db_multiplier=1.0,  
            top_db=80.0           
        )
    
        label = self.labels[idx]
        features_len = features.size(-1)
        label_len = len(label)
        return features, label, features_len, label_len
    
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
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    for i, (features, labels, features_len, labels_len) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print("Features shape:", features.shape)
        print("Labels shape:", labels.shape)
        print("Features length:", features_len)
        print("Labels length:", labels_len)      
        break 

if __name__ == "__main__":
    load_data()
    