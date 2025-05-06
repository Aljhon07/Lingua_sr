import os
import csv
from tools import language_corpus as lc
import pandas as pd
import config
import re

def process_and_encode_common_voice(common_voice_path, tsv_files, output_path):
    remove_chars = r"[?!’–—‘\-\.:;()“”\"]"
    output_path = os.path.join(output_path, config.LANGUAGE)
    file_names = []
    transcriptions = []

    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping.")
        return
    
    for file in tsv_files:
        file_path = os.path.join(common_voice_path, file)
        if os.path.exists(file_path):
            print(f"Processing {file} in {common_voice_path}")
            df = pd.read_csv(file_path, sep='\t')
            for index, row in df.iterrows():
                if 'path' in row and 'sentence' in row:
                    file_names.append(row['path'].replace('.mp3', ''))
                    transcription = row['sentence'].lower()
                    transcription = transcription.replace('"', ' ').replace("'", '')

                    transcription = re.sub(remove_chars, '', transcription)
                    transcriptions.append(transcription)

                if index > 5000:
                     break    
                
    sentences_filename = output_path + "_sentences.txt"
    os.makedirs(os.path.dirname(sentences_filename), exist_ok=True)
    with open(sentences_filename, 'w', encoding='utf-8') as sentences_file:
        for transcript in transcriptions:
            sentences_file.write(transcript + '\n')
    print(f"Transcriptions written to {sentences_filename}")


    df_output = pd.DataFrame({
        'file_name': file_names,
        'duration': [None] * len(file_names),
        'padded_duration': [None] * len(file_names),
        'num_frames': [None] * len(file_names),
        'sample_rate': [None] * len(file_names),
        'transcription': transcriptions,
    })
    tsv_filename = output_path + ".tsv"

    os.makedirs(os.path.dirname(tsv_filename), exist_ok=True)

    df_output.to_csv(tsv_filename, sep='\t', index=False)
    print(f"TSV file written to {tsv_filename}")

    