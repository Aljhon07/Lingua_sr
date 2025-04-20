import os
from tools import tsv_extractor as te, audio, language_corpus as lc
from src import SpeechDataset as sd
from src import SpeechRecognitionModel as model
import pandas as pd
import config

if __name__ == '__main__':
    language = config.LANGUAGE
    common_voice_path = config.COMMON_VOICE_PATH
    output_path = config.OUTPUT_PATH
    wavs_path = config.WAVS_PATH
    
    # tsv_files = ["validated.tsv", "train.tsv", "dev.tsv", "clean.tsv", "test.tsv"]
    # if os.path.exists(common_voice_path):
    #     te.process_and_encode_common_voice(common_voice_path, tsv_files, output_path)

    # if not os.path.exists(wavs_path):
    #     print("Converting audio files to WAV format...")
    #     os.makedirs(wavs_path, exist_ok=True)
    #     print(f"Directory {wavs_path} created.")
    #     df = pd.read_csv(f"{output_path}/{language}.tsv", sep='\t')
    #     file_names = df['file_name'].tolist()
    #     audio.to_wav_batch(file_names, f"{common_voice_path}/clips" , wavs_path)
    #     print(f"Converted audio files to WAV format and saved in {wavs_path}.")
    # else:
    #     print(f"Directory {wavs_path} already exists. Skipping.")

    model.train()
    
  



    

