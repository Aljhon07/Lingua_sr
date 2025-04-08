import os
import librosa
import numpy as np
import pandas as pd
import config
import soundfile as sf
from tools.extract_mel_features import extract_mel_features
from pydub import AudioSegment

def preprocess_audio_files(input_tsv, audio_dir, output_dir, base_output_dir, target_length=500):
    print(f"Preprocessing audio files from {input_tsv} to {output_dir}")
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
            original_features = extract_mel_features(audio_path)
            
            np.save(os.path.join(output_dir, f"{file_name}.npy"), original_features)
            data.append((file_name, transcription))
            print(original_features)
         
            transcriptions.append(transcription)
            print(f"Processed {index} {file_name}: {transcription}")

    df_clean = pd.DataFrame(data, columns=['file_name', 'transcript'])
    df_clean.to_csv(os.path.join(base_output_dir, f'{config.STAGE}.tsv'), sep='\t', index=False)


def extract_data_from_tsv(input_tsv, output_dir, base_output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = []
    transcriptions = []
    df = pd.read_csv(input_tsv, sep='\t')  
    
    for index, row in df.iterrows():
        file_name = row['path'].split('/')[-1].split('.')[0]  
        transcription = row['sentence']
           
        data.append((file_name, transcription))
        transcriptions.append(transcription)
        print(f"Processed {index} {file_name}: {transcription}")
        
    df_clean = pd.DataFrame(data, columns=['file_name', 'transcript'])
    df_clean.to_csv(os.path.join(base_output_dir, f'{config.STAGE}.tsv'), sep='\t', index=False)

def convert_audio_to_wav(input_tsv, audio_dir, output_dir):
    print(f"Converting audio files from {input_tsv} to {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(input_tsv, sep='\t')
    for index, row in df.iterrows():
        audio_filename = row['path']
        audio_path = f"{audio_dir}/{audio_filename}"
        print(f"Converting {audio_path} to WAV format")
        sound = AudioSegment.from_mp3(audio_path)
        sound.export(os.path.join(output_dir, f"{audio_filename.replace("mp3", "wav")}"), format="wav")
    