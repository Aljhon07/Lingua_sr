import librosa
import numpy as np

def extract_mel_features(audio, sr=16000, n_mels=80, hop_length=160, target_length=500):
    y, _ = librosa.load(audio, sr=sr)
    # ascii_art = plot_waveform(y, sr)
    # print(ascii_art)
    
    def pad_or_trim(mel_spec_db):
        if mel_spec_db.shape[1] > target_length:
            mel_spec_db = mel_spec_db[:, :target_length]
        elif mel_spec_db.shape[1] < target_length:
            padding = target_length - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, padding)), mode='constant')
        return mel_spec_db

    # Original features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec)
    original_features = pad_or_trim(mel_spec_db)

    # Augmented features
    noisy_audio = add_noise(y)
    mel_spec_noisy = librosa.feature.melspectrogram(y=noisy_audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db_noisy = librosa.power_to_db(mel_spec_noisy)
    noisy_features = pad_or_trim(mel_spec_db_noisy)
   
    pitched_audio = pitch_shift(y, sr=sr, n_steps=2)
    mel_spec_pitched = librosa.feature.melspectrogram(y=pitched_audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db_pitched = librosa.power_to_db(mel_spec_pitched)
    pitched_features = pad_or_trim(mel_spec_db_pitched)
  
    stretched_audio = time_stretch(y, rate=1.1)
    mel_spec_stretched = librosa.feature.melspectrogram(y=stretched_audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db_stretched = librosa.power_to_db(mel_spec_stretched)
    stretched_features = pad_or_trim(mel_spec_db_stretched)

    return original_features, noisy_features, stretched_features, pitched_features, stretched_audio, pitched_audio, noisy_audio 


def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def time_stretch(audio, rate=1.1):
    return librosa.effects.time_stretch(y=audio, rate=rate)

def pitch_shift(y, sr=16000, n_steps=2):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
