import torch
import torchaudio.transforms as T
import torchaudio.functional as F
import matplotlib.pyplot as plt
import numpy as np


def min_max_normalize(spectrogram):
    spectrogram_min = spectrogram.min()
    spectrogram_max = spectrogram.max()
    normalized_spec = (spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min + 1e-6)
    return normalized_spec

def mean_norm(spectrogram):
    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)
    return spectrogram
def plot_waveforms(waveform, trimmed):
    plt.figure(figsize=(12, 4))

    waveform = waveform[0, :1000]
    trimmed = trimmed[0, :1000]
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

def plot_spectrogram(orig, normalized, sample_rate=16000):
    
    orig = orig.detach()
    normalized = normalized.detach()
    
    # Plot the raw (unnormalized) spectrogram
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convert tensors to numpy arrays
    orig = orig.squeeze(0).cpu().numpy()
    normalized = normalized.squeeze(0).cpu().numpy()
    
    # Plot the raw (unnormalized) spectrogram
    im0 = axs[0, 0].imshow(orig, aspect='auto', origin='lower', cmap='inferno')
    axs[0, 0].set_title('Raw Mel Spectrogram')
    axs[0, 0].set_xlabel('Time Frames')
    axs[0, 0].set_ylabel('Mel Bands')
    fig.colorbar(im0, ax=axs[0, 0], format="%+2.0f dB")

    # Plot the normalized spectrogram
    im1 = axs[0, 1].imshow(normalized, aspect='auto', origin='lower', cmap='inferno')
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

def double_vad(audio, sample_rate):
    # Apply VAD from the front
    trimmed_front = F.vad(audio, sample_rate=sample_rate, trigger_level=5.0)
    
    if trimmed_front.size(1) < 2:
        print("Less than 2.5")
        return audio
    # Reverse the waveform along the time dimension
    reversed_audio = torch.flip(trimmed_front, dims=[-1])
    
    # Apply VAD on reversed audio (trims end of original)
    trimmed_back = F.vad(reversed_audio, sample_rate=sample_rate, trigger_level=5.0)
    
    # Flip back to original orientation
    final_audio = torch.flip(trimmed_back, dims=[-1])
    
    return final_audio

def extract_spectrogram(waveform):
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=16000,
        n_mels=128,
        n_fft=400,
        win_length=400,
        hop_length=160,
        power=1.0
    )
    amplitude = T.AmplitudeToDB(stype='power', top_db=80)

    spec = mel_spectrogram(waveform).contiguous()
    # print(f"[Initial Spectrogram Stats] Shape: {spec.shape} / Min: {spec.min()} / Max: {spec.max()} / Mean: {spec.mean()} / Std: {spec.std()}")
    spec = amplitude(spec)
    # print(f"[Ampltude to DB] Min: {spec.min()} / Max: {spec.max()} / Mean: {spec.mean()} / Std: {spec.std()}")
    
    return  spec

def get_padded_duration(trimmed_duration, current_length):
    # Define non-overlapping buckets: (min (inclusive), max (exclusive), target_duration)
    buckets = [
        (0.0, 1.0, 1.0),      # 0.0 ≤ x < 0.5 → 0.5
        (1.0, 1.5, 1.5),      # 0.5 ≤ x < 1.0 → 1.0
        (1.5, 2.0, 2),      # 1.0 ≤ x < 1.5 → 1.5
        (2.0, 2.5, 2.5),      # 2.0 ≤ x < 2.5 → 2.5
        (2.5, 3.0, 3),      # 2.5 ≤ x < 3.0 → 3.0
        (3.0, 3.5, 3.5),      # 3.0 ≤ x < 3.5 → 3.5
        (3.5, 4.0, 4),      # 3.0 ≤ x < 4.5 → 4.5
        (4.0, 4.5, 4.5),      # 4.0 ≤ x < 4.5 → 4.5
        (4.5, 5.0, 5),      # 4.5 ≤ x < 5.0 → 5.0
        (5.0, 5.5, 5.5),      # 5.0 ≤ x < 5.5 → 5.5
        (5.5, 6.0, 6),      # 5.5 ≤ x < 6.0 → 6.0
        (6.0, 6.5, 6.5),      # 6.0 ≤ x < 6.5 → 6.5
        (6.5, 7.0, 7),      # 6.5 ≤ x < 7.0 → 7.0
        (7.0, 7.5, 7.5),      # 7.0 ≤ x < 7.5 → 7.5
        (7.5, 8.0, 8),      # 7.5 ≤ x < 8.0 → 8.0
        (8.0, 8.5, 8.5),      # 8.0 ≤ x < 8.5 → 8.5
        (8.5, 9.0, 9),      # 8.5 ≤ x < 9.0 → 9.0
    ]
    
    # Check buckets first
    for min_dur, max_dur, target in buckets:
        if min_dur <= trimmed_duration < max_dur:
            padding = int(target * 16000 - current_length)
            return target, padding, True

    return trimmed_duration, 0 , False


def pad_waveform(waveform, sr = 16000):
    current_length = waveform.size(1)
    trimmed_duration = current_length / sr
    padded_duration, padding, save = get_padded_duration(trimmed_duration, current_length)
            
    waveform = torch.nn.functional.pad(waveform, (0, padding), value=1e-6)
    return waveform, ( padded_duration,  save,  current_length, trimmed_duration)

def ctc_decoder(preds):
    decoded = []
    prev_char = None
    for char_idx in preds:
        if char_idx != 0 and char_idx != prev_char:
            decoded.append(char_idx)
        prev_char = char_idx
    return decoded  