import os
import librosa
import numpy as np
import pandas as pd
import config
import soundfile as sf
from tools.extract_mel_features import extract_mel_features

def preprocess_audio_files(input_tsv, audio_dir, output_dir, base_output_dir, target_length=500):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = []
    transcriptions = []
    df = pd.read_csv(input_tsv, sep='\t')  
    
    for index, row in df.iterrows():
        file_name = row['path'].split('/')[-1].split('.')[0]  
        audio_path = os.path.join(audio_dir, file_name + ".wav")  
        transcription = row['sentence']
        
        if os.path.exists(audio_path):
            original_features, noisy_features, stretched_features, pitched_features, pitched_audio, stretched_audio, noisy_audio = extract_mel_features(audio_path, target_length=target_length)
            
            np.save(os.path.join(output_dir, f"{file_name}.npy"), original_features)
            data.append((file_name, transcription))
            
            np.save(os.path.join(output_dir, f"{file_name}_noisy.npy"), noisy_features)
            data.append((f"{file_name}_noisy", transcription))
            
            np.save(os.path.join(output_dir, f"{file_name}_stretched.npy"), stretched_features)
            data.append((f"{file_name}_stretched", transcription))
            
            np.save(os.path.join(output_dir, f"{file_name}_pitched.npy"), pitched_features)
            data.append((f"{file_name}_pitched", transcription))
            
            data.append((file_name, transcription))
            transcriptions.append(transcription)
            print(f"Processed {file_name}: {transcription}")
            # sf.write(os.path.join(output_dir, f"{file_name}_noisy.wav"), noisy_audio, 16000)
            # sf.write(os.path.join(output_dir, f"{file_name}_stretched.wav"), stretched_audio, 16000)
            # sf.write(os.path.join(output_dir, f"{file_name}_pitched.wav"), pitched_audio, 16000)

    df_clean = pd.DataFrame(data, columns=['file_name', 'transcript'])
    a = df_clean.to_csv(os.path.join(base_output_dir, f'{config.STAGE}.tsv'), sep='\t', index=False)
