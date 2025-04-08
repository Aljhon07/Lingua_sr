import os
import sentencepiece as spm
import pandas as pd
from collections import Counter
from time import sleep
import config

def train_tokenizer(dataset_src, output_dir, language):
    if os.path.exists(f"{output_dir}/{language}.model"):
        print("Model already exists")
        sleep(2.5)
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Preprocessing Dataset")
    transcripts = []
    tsv_file = os.path.join(output_dir, f'{config.STAGE}.tsv')
    if os.path.exists(tsv_file):
        df = pd.read_csv(tsv_file, sep='\t')
        transcripts.extend(df['transcript'].tolist())

    transcript_file = os.path.join(output_dir, 'sentences.txt')
    with open(transcript_file, 'w', encoding='utf-8') as f:
        for transcript in transcripts:
            f.write(transcript + '\n')

    sentences_file = os.path.join(output_dir, 'sentences.txt')
    model_prefix = os.path.join(output_dir, f'{language}')
    spm.SentencePieceTrainer.train(
    input=sentences_file,
    model_prefix=model_prefix,
    model_type='bpe',  # Better for morphologically rich languages
    vocab_size=5000,  # Adjust based on your dataset size
    character_coverage=1,
    split_digits=True,  # Crucial for numbers/dates
    max_sentence_length=8192,
    split_by_unicode_script=True,
)

    print(f"SentencePiece model trained and saved as {model_prefix}.model and {model_prefix}.vocab")
    
    
def tokenize_tsv(input_tsv, model_file):
    print(f"Tokenizing TSV: {model_file}")
    sp = spm.SentencePieceProcessor(model_file=model_file)
    
    df = pd.read_csv(input_tsv, sep='\t')
    
    tokenized_transcripts = []
    tokenized_transcripts_str = []
    for transcript in df['transcript']:
        tokens = sp.encode_as_ids(transcript)
        tokens_str = sp.encode_as_pieces(transcript)
        tokenized_transcript = ' '.join(map(str, tokens))
        tokenized_transcript_str = ' '.join(tokens_str)
        print(tokens)
        tokenized_transcripts.append(tokenized_transcript)
        tokenized_transcripts_str.append(tokenized_transcript_str)
        # print(f"Original: {transcript}")
        # print(f"Tokenized: {tokenized_transcript}")
        # print(f"Detokenized: {sp.decode(tokens)}")
    
    df['tokenized_transcript'] = tokenized_transcripts
    df['tokenized_transcript_str'] = tokenized_transcripts_str
    
    df['tokenized_transcript'] = tokenized_transcripts
    
    df.to_csv(input_tsv, sep='\t', index=False)
    print(f"Tokenized TSV saved to {input_tsv}")


def encode_text(text):
    model_file = f"{config.PATHS['base_output']}/{config.LANG}.model"

    print(f"Encoding {text} using model {model_file}")
    
    sp = spm.SentencePieceProcessor(model_file=model_file)
    
    token_ids = sp.encode_as_ids(text)
    
    return token_ids

def decoder(token_ids):
    
    model_file = f"{config.PATHS['base_output']}/{config.LANG}.model"

    print(f"Decoding {token_ids} using model {model_file}")
    
    sp = spm.SentencePieceProcessor(model_file=model_file)
    
    decoded_transcript = sp.decode(token_ids)
    
    return decoded_transcript
    print(f"Decoded Transcript: {decoded_transcript}")
        

