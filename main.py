import os
import config
from tools import preprocess, language_corpus as ls, load_data, dataset as ds
from models import SimpleModel as model
def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(audio_dir):
        # Means no wav, select /clips instead then convert to wav
        preprocess.convert_audio_to_wav(input_tsv, f"{config.PATHS["common_voice"]}/clips", audio_dir)
        pass
    
    # preprocess.extract_data_from_tsv(input_tsv, output_dir, base_output_dir)
    # ls.train_tokenizer(config.PATHS['common_voice'], base_output_dir, config.LANG)
    # ls.tokenize_tsv(f"{base_output_dir}/{config.STAGE}.tsv", f"{base_output_dir}/{config.LANG}.model")
    # load_data.load_data(config.PATHS['output'], f"{base_output_dir}/{config.STAGE}.tsv")
    # data = ds.load_data(audio_dir, f"{base_output_dir}/{config.STAGE}.tsv")
    model.train()
    pass
        

input_tsv = f'{config.PATHS['common_voice']}/{config.STAGE}.tsv'
audio_dir = f'{config.PATHS['common_voice']}/wavs'
base_output_dir = f'{config.PATHS["base_output"]}'
output_dir = f'{config.PATHS["output"]}'

if __name__ == '__main__':
    main()
    pass