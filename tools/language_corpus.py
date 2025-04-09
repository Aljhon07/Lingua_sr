import os
import sentencepiece as spm
import config

def train(model_type='bpe', vocab_size=5000,model_prefix = config.LANGUAGE,):
    input_file = os.path.join(config.OUTPUT_PATH, config.LANGUAGE+"_sentences.txt")
    print(input_file)
    model_path = os.path.join(config.OUTPUT_PATH, model_prefix)
    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_path,
            vocab_size=8000,
            model_type=model_type,
            character_coverage=1.0,
        )
        print(f"SentencePiece model trained successfully: {model_prefix}.model")
    except Exception as e:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_path,
            vocab_size=5000,
            model_type=model_type,
            character_coverage=1.0,
        )
        print(f"Error training SentencePiece model: {e}")

def encode(input_text):
    
    try:
        sp = spm.SentencePieceProcessor(model_file=model_file)
        encoded = sp.encode(input_text, out_type=int) # or out_type=int for IDs
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
