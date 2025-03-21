from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
import config

class SpeechDataset(Dataset):
    def __init__(self, data_dir, tsv_file):
        self.data_dir = data_dir
        self.data = pd.read_csv(tsv_file, sep='\t')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx]['file_name']
        transcription = self.data.iloc[idx]['tokenized_transcript']
        transcription = [int(token) for token in transcription.split()]  # Convert tokens to integers
        transcription_length = len(transcription)

        np_features = np.load(f"{self.data_dir}/{file_name}.npy")
        features = torch.from_numpy(np_features).float().transpose(0, 1)

        return file_name, features.clone().detach(), torch.tensor(transcription, dtype=torch.long).clone().detach(), torch.tensor(transcription_length).clone().detach()


def collate_fn(batch):
    file_names, features, transcriptions, transcription_lengths = zip(*batch)
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    
    transcriptions = torch.nn.utils.rnn.pad_sequence(transcriptions, batch_first=True)
    transcription_lengths = torch.tensor(transcription_lengths)

    return file_names, features, transcriptions, transcription_lengths

def load_data(data_dir, tsv_file, batch_size=32):
    dataset = SpeechDataset(config.PATHS["output"], f"{config.PATHS["base_output"]}/{config.STAGE}.tsv")
    train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_data_loader


# train_loader = DataLoader(SpeechDataset(config.PATHS["output"], f"{config.PATHS["base_output"]}/{config.STAGE}.tsv"), batch_size=32, shuffle=False, collate_fn=collate_fn)
# for batch_idx, (file_names, features, transcriptions, transcription_lengths) in enumerate(train_loader):
#     print(f"Batch {batch_idx + 1}:")
#     print(f"  Features batch shape: {features.shape}") 
#     print(f"  Transcriptions batch shape: {transcriptions.shape}") 
#     print(f"  Transcription lengths: {transcription_lengths}")  # Should match actual lengths
#     print("-" * 50)

#     # Print first example in batch
#     print(f"Sample feature shape(first item in batch): {features[0].shape}")
#     print(f"Sample transcription shape(first item in batch): {transcriptions[0].shape}")
#     print(f"Sample transcription (first item in batch): {transcriptions[0]}")
#     print(f"Sample feature (first item in batch): {features[0]}")
#     print(f"Length of transcription: {transcription_lengths[0]}")
#     # Stop after checking the first few batches
#     if batch_idx == 2:
#         break
