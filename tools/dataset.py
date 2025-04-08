import torch
import os
import torchaudio
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

class SpeechDataset(Dataset):
    def __init__(self, audio_dir, tsv_file, sample_rate=16000, n_mels=64, win_length=400, hop_length=160):
        self.audio_dir = audio_dir
        self.data = pd.read_csv(tsv_file, sep='\t')
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length

        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=80.0)
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_name = row['file_name']
        audio_path = os.path.join(self.audio_dir, f"{audio_name}.wav")
        
       

        try:
            audio_tensor, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                audio_tensor = resampler(audio_tensor)
        except Exception as e:
            print(f"Error loading audio: {audio_path} - {e}")
            return None
   
        mel_tensor = self.mel_transform(audio_tensor)
        mel_tensor = self.amplitude_to_db(mel_tensor)
        transcription = list(map(int, row['tokenized_transcript'].split()))

        return mel_tensor, torch.tensor(transcription, dtype=torch.long), mel_tensor.size(-1),  len(transcription),
   


def collate_fn(batch):

    features = []
    transcripts = []
    input_lengths = []
    transcript_lengths = []
    
    for (feature, transcript, input_len, transcript_len) in batch:
        features.append(feature.squeeze(0).transpose(0, 1).contiguous())
        transcripts.append(transcript)
        input_lengths.append(input_len)
        transcript_lengths.append(transcript_len)

    
    normalized_features = []
    for feat in features:
        # Compute mean/std excluding padding (if needed, but here features are pre-padded)
        # Since features are from MelSpectrogram, they don't have padding yet
        mean = feat.mean()
        std = feat.std()
        normalized_feat = (feat - mean) / std
        normalized_features.append(normalized_feat)

    features = nn.utils.rnn.pad_sequence(normalized_features, batch_first=True, padding_value=-80)
    
    transcripts = nn.utils.rnn.pad_sequence(transcripts, batch_first=True)
    features = features.unsqueeze(1).transpose(2, 3).contiguous()
    return features, transcripts, torch.tensor(input_lengths, dtype=torch.long), torch.tensor(transcript_lengths, dtype=torch.long)

    
def load_data(audio_dir, tsv_file, batch_size=32):

    dataset = SpeechDataset(audio_dir, tsv_file)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    return data_loader
