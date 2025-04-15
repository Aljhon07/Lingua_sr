import os
import csv
from tools import language_corpus as lc
import pandas as pd
import config
def process_and_encode_common_voice(common_voice_path, tsv_files, output_path):

    output_path = os.path.join(output_path, config.LANGUAGE)
    file_names = []
    transcriptions = []
    tokenized_transcriptions = []
    tokenized_transcriptions_str = []


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
                    transcriptions.append(row['sentence'])
    # Training the SentencePiece model
    sentences_filename = output_path + "_sentences.txt"
    os.makedirs(os.path.dirname(sentences_filename), exist_ok=True)
    with open(sentences_filename, 'w', encoding='utf-8') as sentences_file:
        for transcript in transcriptions:
            sentences_file.write(transcript + '\n')
    lc.train()
    print(f"Transcriptions written to {sentences_filename}")

    for transcription in transcriptions:
            encoded_ids, encoded_pieces = lc.encode(transcription)
            tokenized_transcriptions.append(encoded_ids)
            tokenized_transcriptions_str.append(encoded_pieces)   

    df_output = pd.DataFrame({
        'file_name': file_names,
        'transcription': transcriptions,
        'tokenized_transcription': tokenized_transcriptions,
        'tokenized_transcription_str': tokenized_transcriptions_str
    })
    tsv_filename = output_path + ".tsv"

    os.makedirs(os.path.dirname(tsv_filename), exist_ok=True)

    df_output.to_csv(tsv_filename, sep='\t', index=False)
    print(f"TSV file written to {tsv_filename}")

    