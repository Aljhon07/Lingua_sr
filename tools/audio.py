import os
import subprocess
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import config
import time
from tools.utils import min_max_normalize, double_vad, pad_waveform, plot_waveforms, plot_spectrogram, extract_spectrogram 
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
              
            spectrogram, padded_duration, save = load_audio(index, audio_path, info)
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

def load_audio(idx = 0, audio_path = None, info = None, sample_rate = 16000):

    if info is None:
        info = torchaudio.info(audio_path)

    initial_waveform, sample_rate = torchaudio.load(audio_path)
    # print(f"Loaded {audio_path}. Shape: {initial_waveform.shape} / Sample Rate: {sample_rate} / Duration: {info.num_frames / info.sample_rate} seconds")
    if info.sample_rate != sample_rate:
        print(f"Resampling {audio_path} to {sample_rate} Hz...")
        resampler = T.Resample(orig_freq=info.sample_rate, new_freq=sample_rate)
        initial_waveform = resampler(initial_waveform)
    # print(f"[Waveform] Min: {initial_waveform.min()} / Max: {initial_waveform.max()} / Mean: {initial_waveform.mean()} / Std: {initial_waveform.std()}")    

    initial_waveform = initial_waveform.contiguous()
    trimmed_waveform = double_vad(initial_waveform, sample_rate=sample_rate)

    rms = trimmed_waveform.pow(2).mean().sqrt()
    gain = 0.1 / rms
    if trimmed_waveform.numel() == 0 or rms < 1e-5:
        return trimmed_waveform, 0, False
    
    rms_normalized = trimmed_waveform * gain
    # print(f"[RMS Nrm]: {rms_normalized.pow(2).mean().sqrt()} | Min: {rms_normalized.min()} | Max: {rms_normalized.max()}")
    # torchaudio.save("RMS_" + str(idx) + '.wav', rms_normalized, 16000)

    rms_waveform, rms_metadata = pad_waveform(rms_normalized)
    padded_duration, save, orig_length, trimmed_duration = rms_metadata
    # print(f"Audio Path: {audio_path} | Padded Duration: {padded_duration} | Original Length: {orig_length} | Padded Length: {rms_waveform.size(1)} | Orig (s): {info.num_frames / info.sample_rate} Trimmed (s): {trimmed_duration} Curr: {rms_waveform.size(1) / sample_rate:.2f} seconds")

    rms_spec = extract_spectrogram(rms_waveform)
    # print(f"[RMS Spec]: {rms_normalized.pow(2).mean().sqrt()} | Min: {rms_normalized.min()} | Max: {rms_normalized.max()}")
    rms_normalized = min_max_normalize(rms_spec)
    # print(f"[RMS Norm]: {rms_normalized.pow(2).mean().sqrt()} | Min: {rms_normalized.min()} | Max: {rms_normalized.max()}")
    # plot_waveforms( trimmed_waveform, rms_waveform)
    # plot_spectrogram(rms_spec, rms_normalized)
    spec = rms_normalized
    print(f"{idx}", end="\r")
    return spec, padded_duration, save

