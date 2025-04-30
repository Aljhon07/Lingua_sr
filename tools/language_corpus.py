import os
import sentencepiece as spm
import config

def train(model_type='bpe', vocab_size=750,model_prefix = config.LANGUAGE,):
    input_file = os.path.join(config.OUTPUT_PATH, config.LANGUAGE+"_sentences.txt")
    print(f"Training SentencePiece model with input file: {input_file}")
    model_path = os.path.join(config.OUTPUT_PATH, model_prefix)
    
    if os.path.exists(model_path + ".model"):
        print(f"Model {model_path}.model already exists. Skipping training.")
        return
    
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_path,
        model_type=model_type,
        character_coverage=1.0,
        vocab_size=vocab_size,
    )
    print(f"SentencePiece model trained successfully: {model_prefix}.model")

def encode(input_text):
    try:
        sp = spm.SentencePieceProcessor(model_file=model_file)
        encoded = sp.encode_as_ids(input_text) # or out_type=int for IDs
        encoded_str = sp.encode_as_pieces(input_text) # or out_type=str for pieces
        return ' '.join(map(str, encoded)), ' '.join(encoded_str)
    except Exception as e:
        print(f"Error encoding with SentencePiece: {e}")
        return None

def decode(encoded_tokens):
    """Decodes tokens using a SentencePiece model."""
    try:
        sp = spm.SentencePieceProcessor(model_file=model_file)
        decoded = sp.decode(encoded_tokens)
        return decoded
    except Exception as e:
        print(f"Error decoding with SentencePiece: {e}")
        return None
    
model_file = config.OUTPUT_PATH + "/" + config.LANGUAGE + ".model"

if __name__ == "__main__":
    encoded = encode("Hello world")
    print(f"Encoded: {encoded}")
    decoded = decode(encoded)
    print(f"Decoded: {decoded}")
