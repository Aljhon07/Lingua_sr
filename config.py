LANGUAGE = "en"
COMMON_VOICE_PATH = f"./commonvoice/{LANGUAGE}"
OUTPUT_PATH = f"./output/{LANGUAGE}"
WAVS_PATH = COMMON_VOICE_PATH + "/wavs"
LOG_DIR = "./logs"

H_PARAMS = {
    "BASE_LR": 0.0001,
    "TOTAL_EPOCH": 100,
    "VOCAB_SIZE": 31,
    "N_FEATS": 80,
    "VERBOSE": False
}

