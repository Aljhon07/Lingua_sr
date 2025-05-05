import torch
from src.SpeechRecognitionModel import SpeechRecognitionModel
from tools import audio, language_corpus as lc

model = SpeechRecognitionModel(1000)
dict = torch.load("./5.0 DS Model/target_reached_model.pth")

try:
    model.load_state_dict(dict)
    print('Model Loaded')
except RuntimeError as e:
    print(e)


def ctc_decoder(preds):
    decoded = []
    prev_char = None
    for char_idx in preds:
        if char_idx != 0 and char_idx != prev_char:
            decoded.append(char_idx)
        prev_char = char_idx
    return decoded 
 
model.eval()
with torch.no_grad():
    spec, _, _ = audio.load_audio(audio_path="./commonvoice/en/wavs/common_voice_en_40988119.wav")
    print(spec.shape)
    spec = spec.unsqueeze(0)
    output = model(spec).contiguous()
    pred_raw = torch.argmax(output, dim=2).transpose(0, 1).contiguous()
    pred = ctc_decoder(pred_raw[0].tolist())
    decoded = lc.decode(pred)
    print(decoded)

if __name__ == "__main__":
    pass