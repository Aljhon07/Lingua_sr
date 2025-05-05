import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import config
from tools import utils
class SpeechDataset(Dataset):
    def __init__(self, data, sample_rate=16000, apply_mask = False):
        self.tsv_file = os.path.join(config.OUTPUT_PATH, f"{config.LANGUAGE}.tsv")
        self.sample_rate = sample_rate
        self.data = data
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=25)
        self.apply_mask = apply_mask
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
        
        if self.apply_mask:
            spectrogram = self.freq_mask(spectrogram)
            spectrogram = self.time_mask(spectrogram)
            
        transcription = metadata['transcription']

        # utils.plot_spectrogram(spectrogram, spectrogram)
        # print(f"Loading spectrogram from {spec_path}")
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
    train_loaders, val_loaders = [], []
    total_duration = 0

    for idx, (key) in enumerate(sorted_keys):
        if len(categorized_data[key]) < 200:
            continue
        data_samples = categorized_data[key]
        split_idx = int(0.9 * len(data_samples))
        
        # 2. Create separate train/val data lists
        train_samples = data_samples[:split_idx]
        val_samples = data_samples[split_idx:]
        # 3. Create datasets with proper masking settings
        train_dataset = SpeechDataset(train_samples, apply_mask=True)  # Only train gets masked
        val_dataset = SpeechDataset(val_samples, apply_mask=False)    # Val stays clean
        # 4. Create DataLoaders
        train_loaders.append(DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn))
        val_loaders.append(DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn))

        ds_duration = train_dataset.get_total_durations() + val_dataset.get_total_durations()
        total_duration += ds_duration
        print(f"Key: {key} | Dataset Size: {len(train_dataset) + len(val_dataset)} samples | Duration: {ds_duration / 60 / 60:.2f} hours")
        
    print(f"Total Duration: {total_duration / 60 / 60:.2f} hours")

    full_overfit_samples = categorized_data[3.0][:500]

    overfit_train_samples = full_overfit_samples[:450]  # First 32 for training
    overfit_val_samples = full_overfit_samples[450:]    # Last 10 for validation

    # Create datasets
    overfit_dataset = SpeechDataset(overfit_train_samples, apply_mask=True)
    overfit_val_dataset = SpeechDataset(overfit_val_samples)

    # Create loaders
    overfit_loader = DataLoader(overfit_dataset, batch_size=25, shuffle=True, collate_fn=collate_fn)
    overfit_val_loader = DataLoader(overfit_val_dataset, batch_size=25, shuffle=False, collate_fn=collate_fn)
    
    # loaders = {'train': train_loaders[:5], 'val': val_loaders[:5]}

    # loaders = {'train': [train_loaders[3]], 'val': [val_loaders[3]]}


    loaders = {"train" : [overfit_loader], "val": [overfit_val_loader]}

    # for i, (feature, label, feature_len, label_len, string_labels, audio_paths) in enumerate(train_loaders[3]):
    #     print(f"Features batch shape: {feature.shape}")
    #     print(f"Labels batch shape: {label.shape}")
    #     print(f"Feature lengths: {feature_len}")
    #     print(f"Label lengths: {label_len}")
    #     break
        
    return loaders

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
