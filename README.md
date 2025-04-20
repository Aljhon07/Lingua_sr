# Lingua_ASR_Process

## Dataset Gathering

## Data Preprocessing

1.  **Divide Datasets**: Divide datasets into 80/10/10 for training/dev/testing respectively.
2.  **Convert to Appropriate Format**: 16000 sample rate, and wav format.
3.  **Adding Noise**: Random noise is added to the audio.
4.  **Time Stretching**: The audio is stretched by a factor of
5.  **Pitch Shifting**: The pitch of the audio is shifted by 2 steps.
6.  **Extract Feature**: extract Mel-spectogram features from the audio files and saved as `.npy` files for model training.
7.  **Tokenization of Transcriptions**: Tokenize text....

## Model Training

Includes callbacks for checkpointing and early stopping.

Model Arhictecture and Training Configuration:

1.  CNN
2.  RNN (BiGRU)
3.  Set up an optimizer (AdamW)
4.  CTC Loss
5.  Implement scheduler

## Evaluation

## Inference

## Finetune

## Export

## Requirements

- Python 3.x
- librosa
- numpy
- pandas
- PyTorch
- sentencepiece

<<<<<<< HEAD

## Usage

1. Create commonvoice folder within the same directory of the project folder.
2. # Download dataset from commonvoice and
   .model and .vocab
   https://github.com/bheinzerling/bpemb?tab=readme-ov-file#downloads-for-each-language
   > > > > > > > 87a552b (Reinitialized project)
