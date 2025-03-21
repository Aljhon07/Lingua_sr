import os
import config
from tools import preprocess, language_corpus as ls

def main():
    print("Preprocessing Dataset")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(audio_dir):
        # Means no wav, select /clips instead then convert to wav
        pass
    
    preprocess.preprocess_audio_files(input_tsv, audio_dir, output_dir, base_output_dir)
    ls.train_tokenizer(config.PATHS['common_voice'], base_output_dir, config.LANG)
    ls.tokenize_tsv(f"{base_output_dir}/{config.STAGE}.tsv", f"{base_output_dir}/{config.LANG}.model")
    # Initialize model here
    pass
        

input_tsv = f'{config.PATHS['common_voice']}/{config.STAGE}.tsv'
audio_dir = f'{config.PATHS['common_voice']}/wavs'
base_output_dir = f'{config.PATHS["base_output"]}'
output_dir = f'{config.PATHS["output"]}'

if __name__ == '__main__':
    main()
    pass