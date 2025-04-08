import librosa
import numpy as np

def extract_mel_features(audio_path, sr=16000, n_mels=80, hop_length=160, n_fft=512, 
                        win_length=400, fmin=20, fmax=8000, normalize=True):

    # Load audio with librosa's built-in peak normalization
    y, _ = librosa.load(audio_path, sr=sr)
    
    # Validate audio length
    if len(y) < 0.1 * sr:  # Minimum 100ms
        raise ValueError(f"Audio too short: {len(y)/sr:.2f}s")
    
    # Mel spectrogram extraction
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
    
    if normalize:
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    return mel_spec_db

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def time_stretch(audio, rate=1.1):
    return librosa.effects.time_stretch(y=audio, rate=rate)

def pitch_shift(y, sr=16000, n_steps=2):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
