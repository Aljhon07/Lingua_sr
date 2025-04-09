import torch
import torch.nn as nn
import torch.optim as optim
import config
from tools import dataset as ds
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


class SimpleSpeechModel(nn.Module):
    def __init__(self, input_dim, cnn_output_dim, rnn_output_dim, vocab_size, dropout_prob=0.2):
        super(SimpleSpeechModel, self).__init__()

        self.features = {}
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, cnn_output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(cnn_output_dim, cnn_output_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_output_dim * 2),
            nn.ReLU(),
        )

        # Adjust RNN input size based on CNN output
        self.gru = nn.GRU(
            input_size=(cnn_output_dim * 2) * (input_dim // 2),  # Adjust for maxpool
            hidden_size=rnn_output_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout = dropout_prob
        )

        self.rnn_norm = nn.LayerNorm(rnn_output_dim * 2) # layer norm after rnn.
        self.rnn_gelu = nn.GELU() 
        self.rnn_dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(rnn_output_dim * 2, vocab_size)  # *2 for bidirectional

    def forward(self, x, input_lengths=None):
        print(f"Input shape: {x.shape}")
        x = self.cnn1(x) #(batch_size, channels, features, time_steps)
        print(f"After CNN - 1: {x.shape}") 
        self.features["cnn1"] = x
        x = self.cnn2(x)
        print(f"After CNN - 2: {x.shape}")
        self.features["cnn2"] = x
        x = x.transpose(2, 3).contiguous()  # (batch_size,features , time_steps, channels)
        print(f"After Transpose: {x.shape}")
        x = x.view(x.size(0), x.size(2), -1)
        print(f"After View: {x.shape}")
        x = nn.utils.rnn.pack_padded_sequence(x, torch.floor(input_lengths / 2).int(), batch_first=True, enforce_sorted=False)
        print(f"After Pack: {x.data.shape}")
        x, _= self.gru(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        print(f"After GRU: {x.shape}")
        x = self.rnn_norm(x)
        print(f"After LayerNorm: {x.shape}")
        x = self.rnn_gelu(x)
        print(f"After GELU: {x.shape}")
        x = self.rnn_dropout(x)
        print(f"After Dropout: {x.shape}")
        x = self.fc(x)
        self.features["rnn"] = x
        print(f"After FC: {x.shape}")

        return x

def train():
    input_dim = 64
    cnn_output_dim = 32
    cnn_output_dim2 = 64
    rnn_output_dim = 512
    vocab_size = 5000 
    dropout_prob = 0.2
    blank_id = 0

    # Model and optimizer
    model = SimpleSpeechModel(input_dim, cnn_output_dim, rnn_output_dim, vocab_size, dropout_prob)
    loader = ds.load_data( f'{config.PATHS['common_voice']}/wavs', f"{config.PATHS["base_output"]}/{config.STAGE}.tsv")
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.CTCLoss(blank=blank_id)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (features, transcriptions, input_length, transcription_lengths) in enumerate(loader):
            print("=" * 60)
            print(f"🌀 Epoch {epoch + 1} - Batch {batch_idx+1}/{len(loader)}")
            print("=" * 60)
            optimizer.zero_grad() 

            output = model(features, input_length)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)
            
            
            loss = criterion(output, transcriptions, input_length // 2, transcription_lengths)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print(f"Output shape: {output.shape}")
            # print(f"Output: {output}")
            # print(f"📊 Features shape: {features.shape}")
            # print(f"⏱️ Input lengths: {input_length}")
            # print(f"📝 Transcriptions shape: {transcriptions.shape}")
            # print(f"🔢 Transcription lengths: {transcription_lengths}")

            print('_' * 60)
            print("\n✨ ===== Sample Preview =====")
            print(f"🎧 Feature shape: {features[0].shape}")  
            print(f"⏳ Input length: {input_length[0]}")
            print(f"📏 Transcript length: {transcription_lengths[0]}")
            print(f"🔡 Transcript (token ids): {transcriptions[0]}")
            print(f"Predictions: {preds[0]}")
            print(f"Loss: {loss.item()}")
          
           

        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(loader)}")
        scheduler.step(epoch_loss)


def predict_ctc(model, features, input_length):
    model.eval()
    with torch.no_grad():
        output = model(features, input_length)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)
        preds = torch.argmax(output.transpose(0, 1), dim=2)
        return preds
        
def validate_feature_stats(feature_list, titles = ["Input", "CNN2"]):
    """
    Analyzes and visualizes a list of feature tensors in subplots.
    """
    num_features = len(feature_list)
    fig, axes = plt.subplots(1, num_features, figsize=(10 * num_features, 4)) #adjust figsize.
    if num_features == 1:
        axes = [axes] #make iterable.

    for i, features in enumerate(feature_list):
        try:
            feature_np = features.squeeze(0).cpu().numpy()
        except:
            feature_np = features[0].detach().numpy()

        print(f"\n📊 {titles[i]} Statistics:")
        print(f"    Mean: {np.mean(feature_np)}")
        print(f"    Variance: {np.var(feature_np)}")
        print(f"    Min: {np.min(feature_np)}")
        print(f"    Max: {np.max(feature_np)}")

        librosa.display.specshow(feature_np, sr=16000, hop_length=160, x_axis='time', y_axis='mel', ax=axes[i])
        axes[i].set_title(titles[i])
        fig.colorbar(axes[i].collections[0], ax=axes[i], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()