from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
import config
from tools import language_corpus as ld

class SpeechDataset(Dataset):
    def __init__(self, data_dir, tsv_file):
        self.data_dir = data_dir
        self.data = pd.read_csv(tsv_file, sep='\t')
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Load and validate features
        features = np.load(f"{self.data_dir}/{row['file_name']}.npy")
        
        # Convert to tensor (time, n_mels)
        features = torch.from_numpy(features).float().T
        
        # Convert transcript
        transcription = list(map(int, row['tokenized_transcript'].split()))
        
        return features, torch.tensor(transcription, dtype=torch.long), len(transcription)


def collate_fn(batch):
    # Sort by feature length (descending)
    batch.sort(key=lambda x: x[0].shape[1], reverse=True)
    
    features, transcripts, transcript_lens = zip(*batch)
    input_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long).clone().detach()


    # Pad features (batch, time, freq) -> (batch, 1, freq, time)
    features_padded = torch.nn.utils.rnn.pad_sequence(
        features,
        batch_first=True,
        padding_value=-80.0
    ).unsqueeze(1) # Add channel dim
    
    # Pad transcripts
    transcripts_padded = torch.nn.utils.rnn.pad_sequence(
        transcripts,
        batch_first=True,
        padding_value=0
    )
    
    return (
        features_padded,  # (B,1,n_mels,T)
        input_lengths,  # Input lengths
        transcripts_padded,  # (B,T)
        torch.tensor(transcript_lens)  # Transcript lengths
    )


def load_data(data_dir, tsv_file, batch_size=10):
    dataset = SpeechDataset(data_dir, tsv_file)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Verify first batch
    features, input_length, transcripts, transcript_lens = next(iter(loader))
    # print("\n=== Batch Verification ===")
    # print(f"Features shape: {features[0].shape}")
    # print(f"Input lengths: {input_length}")
    # print(f"Transcripts shape: {transcripts.shape}")
    # print(f"Transcript lengths: {transcript_lens}")
    
    return loader