import os
import subprocess
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import config
import matplotlib.pyplot as plt
import numpy as np
import time

def to_wav(input_file, output_file):
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        command = [
            'ffmpeg',
            '-i', input_file,
            '-acodec', 'pcm_s16le',  # WAV format, 16-bit PCM
            '-ac', '1',              # Mono audio
            '-ar', '16000',           # 16kHz sample rate
            output_file
        ]
        subprocess.run(command, check=True, stderr=subprocess.PIPE) #added stderr to help with debugging.

        print(f"Successfully converted {input_file} to {output_file}")

    except FileNotFoundError:
        print("Error: FFmpeg not found. Make sure it's installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr.decode()}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def to_wav_batch(file_names, input_dir, output_dir):
    """
    Convert a batch of audio files to WAV format.
    """
    for file_name in file_names:
        input_file = os.path.join(input_dir, f"{file_name}.mp3")
        output_file = os.path.join(output_dir, f"{file_name}.wav")
        
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping conversion.")
            continue
        
        print(f"Processing {input_file} to {output_file}")
        if os.path.exists(input_file): 
            to_wav(input_file, output_file)
        else:
            print(f"File {input_file} does not exist. Skipping.")
            
def classify_batch(tsv_file, wavs_path):
    print(f"Classifying audio files using {tsv_file}...")
    
    if not os.path.exists(tsv_file):
        raise(f"TSV file {tsv_file} does not exist.")
    
    output_dir = os.path.join(config.OUTPUT_PATH, "spectrograms")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory {output_dir} created.")
        
    df = pd.read_csv(tsv_file, sep='\t')

    durations = []
    padded_durations = []
    num_frames = []
    sr = []
    for index, row in df.iterrows():
        file_name = row['file_name']
        audio_path = os.path.join(wavs_path, f"{file_name}.wav")
                
        if not os.path.exists(audio_path):
            time.sleep(1)
            print(f"File {audio_path} does not exist. Skipping.")
            padded_durations.append("excluded")
            durations.append(0)
            num_frames.append(0)
            sr.append(0)
            continue
        
        else:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate

            durations.append(duration)
            num_frames.append(info.num_frames)
            sr.append(info.sample_rate)
              
            spectrogram, padded_duration, save = load_audio(audio_path, info)
            padded_durations.append(padded_duration)
            if save:
                torch.save(spectrogram, f"{output_dir}/{file_name}.pt")
            
            # if os.path.exists(f"{output_dir}/{file_name}.pt"):
                # spectrogram = torch.load(f"{output_dir}/{file_name}.pt")
                # print(f"Loaded Spectrogram Stats: Shape: {spectrogram.shape} / Min: {spectrogram.min()} / Max: {spectrogram.max()} / Mean: {spectrogram.mean()} / Std: {spectrogram.std()}")
                # plot_spectrogram(spectrogram, spectrogram)
    
    df['duration'] = durations
    df['padded_duration'] = padded_durations
    df['num_frames'] = num_frames
    df['sample_rate'] = sr
    df.to_csv(tsv_file, sep='\t', index=False)
    print(f"Classified audio files and saved results to {tsv_file}")

def load_audio(audio_path, info, sample_rate = 16000):
    mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_mels=80,
            n_fft=400,
            win_length=400,
            hop_length=160,
        )
        
    amplitude = T.AmplitudeToDB(stype='power', top_db=100)
    padded_duration = 3
    initial_waveform, sample_rate = torchaudio.load(audio_path)
    # print(f"Loaded {audio_path}. Shape: {initial_waveform.shape} / Sample Rate: {sample_rate} / Duration: {info.num_frames / info.sample_rate} seconds")

    if info.sample_rate != sample_rate:
        print(f"Resampling {audio_path} to {sample_rate} Hz...")
        resampler = T.Resample(orig_freq=info.sample_rate, new_freq=sample_rate)
        waveform = resampler(waveform)
    
    initial_waveform = initial_waveform.contiguous()
    trimmed_waveform = double_vad(initial_waveform, sample_rate=sample_rate)
    current_length = trimmed_waveform.size(1)
    trimmed_duration = current_length / sample_rate
    
    padded_duration, padding, save = get_padded_duration(trimmed_duration, current_length)
            
    waveform = torch.nn.functional.pad(trimmed_waveform, (0, padding))
    
    print(f"Audio Path: {audio_path} | Padded Duration: {padded_duration} | Original Length: {current_length} | Padded Length: {waveform.size(1)} | Orig (s): {info.num_frames / info.sample_rate} Trimmed (s): {trimmed_duration} Curr: {waveform.size(1) / sample_rate:.2f} seconds")
    initial_spectrogram = mel_spectrogram(waveform).contiguous()
    spectrogram = amplitude(initial_spectrogram)
    spectrogram_min = spectrogram.min(dim=2, keepdim=True)[0]
    spectrogram_max = spectrogram.max(dim=2, keepdim=True)[0]

    # Normalize the spectrogram
    normalized_spec = (spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min + 1e-6)
    print(f"Normalized Spectrogram: {normalized_spec.shape} / Min: {normalized_spec.min()} / Max: {normalized_spec.max()} / Mean: {normalized_spec.mean()} / Std: {normalized_spec.std()}")
    # print(normalized_spec)
    # print(f"Spectrogram Stats: Shape: {initial_spectrogram.shape} / Min: {initial_spectrogram.min()} / Max: {initial_spectrogram.max()} / Mean: {initial_spectrogram.mean()} / Std: {initial_spectrogram.std()}")
    # print(f"Amplitude Stats: Shape: {spectrogram.shape} / Min: {spectrogram.min()} / Max: {spectrogram.max()} / Mean: {spectrogram.mean()} / Std: {spectrogram.std()}")
    # print(f"Normalized Spectrogram Stats: Shape: {normalized_spec.shape} / Min: {normalized_spec.min()} / Max: {normalized_spec.max()} / Mean: {normalized_spec.mean()} / Std: {normalized_spec.std()}")
    
    # plot_waveforms(initial_waveform, waveform)
    # plot_spectrogram(initial_spectrogram, normalized_spec, sample_rate=16000)
    return normalized_spec, padded_duration, save

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

def plot_waveforms(waveform, trimmed):
    plt.figure(figsize=(12, 4))

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
    # Plot the raw (unnormalized) spectrogram
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convert tensors to numpy arrays
    orig = orig.squeeze(0).cpu().numpy()
    normalized = normalized.squeeze(0).cpu().numpy()
    
    # Plot the raw (unnormalized) spectrogram
    im0 = axs[0, 0].imshow(orig, aspect='auto', origin='lower', cmap='viridis')
    axs[0, 0].set_title('Raw Mel Spectrogram')
    axs[0, 0].set_xlabel('Time Frames')
    axs[0, 0].set_ylabel('Mel Bands')
    fig.colorbar(im0, ax=axs[0, 0], format="%+2.0f dB")

    # Plot the normalized spectrogram
    im1 = axs[0, 1].imshow(normalized, aspect='auto', origin='lower', cmap='viridis')
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