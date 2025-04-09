from torch.utils.data import Dataset, DataLoader
import torch

class SpeechDataset(Dataset):
    def __init__(self, data, labels, input_dim):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def load_data(output_dir):
    
    data = []  # Load your audio features here
    labels = []  # Load your labels here

    dataset = SpeechDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    return dataloader