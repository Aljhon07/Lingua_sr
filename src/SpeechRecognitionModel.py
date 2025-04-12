import torch
import torch.nn as nn
import src.SpeechDataset as sd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import config
import os

class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_rate=0.1):
        super(SpeechRecognitionModel, self).__init__()
        
        # Layer normalization before conv layers
        self.layer_norm1 = nn.LayerNorm(input_dim)
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),  # BatchNorm added after conv1
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(dropout_rate)  # Dropout after pooling
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout_rate)  # Dropout after conv
        )

        # Layer normalization after conv layers
        self.layer_norm2 = nn.LayerNorm(64 * (input_dim // 2))
        
        # Bidirectional GRU and dropout added to GRU
        self.gru = nn.GRU(64 * (input_dim // 2), hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectional GRU

    def forward(self, x, lengths):
        # print("Input Shape: ", x.shape)
        # print("Input Stride: ", x.stride())

        # Apply layer normalization
        x = self.layer_norm1(x)
        # print("After LayerNorm Shape: ", x.shape)
        # print("After LayerNorm (contiguous): ", x.is_contiguous())
        # print("After LayerNorm Stride: ", x.stride())
        
        # Add channel dimension for conv layers
        x = x.unsqueeze(1)
        # print("After unsqueeze Shape: ", x.shape)
        # print("After unsqueeze (contiguous): ", x.is_contiguous())
        # print("After unsqueeze Stride: ", x.stride())
        
        # Apply first convolutional layer
        x = self.conv1(x)
        # print("After conv1 Shape: ", x.shape)
        # print("After conv1 Stride: ", x.stride())
        # print("After conv1 (contiguous): ", x.is_contiguous())

        # Apply second convolutional layer
        x = self.conv2(x)
        # print("After conv2 Shape: ", x.shape)
        # print("After conv2 Stride: ", x.stride())
        # print("After conv2 (contiguous): ", x.is_contiguous())

        # Transpose and flatten for GRU input
        x = x.transpose(1, 2).contiguous()
        # print("After transpose Shape: ", x.shape)
        # print("After transpose (contiguous): ", x.is_contiguous())
        # print("After transpose Stride: ", x.stride())
        
        x = x.flatten(start_dim=2)
        # print("After flatten Shape: ", x.shape)
        # print("After flatten (contiguous): ", x.is_contiguous())
        # print("After flatten Stride: ", x.stride())

        # Apply layer normalization after convs
        x = self.layer_norm2(x)
        # print("After LayerNorm2 Shape: ", x.shape)
        # print("After LayerNorm2 (contiguous): ", x.is_contiguous())
        # print("After LayerNorm2 Stride: ", x.stride())
        
        # Pack padded sequences for variable-length input
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # print("After pack_padded_sequence Shape: ", packed_x.data.shape)
        # print("After pack_padded_sequence Stride: ", packed_x.data.stride())
        # print("After pack_padded_sequence (contiguous): ", packed_x.data.is_contiguous())
        
        # Pass through the bidirectional GRU
        packed_out, _ = self.gru(packed_x)
        # print("After GRU Shape: ", packed_out.data.shape)
        # print("After GRU Stride: ", packed_out.data.stride())
        # print("After GRU (contiguous): ", packed_out.data.is_contiguous())

        # Unpack sequences after GRU
        x, _ = pad_packed_sequence(packed_out, batch_first=True)
        # print("After pad_packed_sequence Shape: ", x.shape)
        # print("After pad_packed_sequence Stride: ", x.stride())
        # print("After pad_packed_sequence (contiguous): ", x.is_contiguous())
        
        # Final output from the fully connected layer
        x = self.ff(x)
        # print("After FF Shape: ", x.shape)
        # print("After FF Stride: ", x.stride())
        # print("After FF (contiguous): ", x.is_contiguous())
        out = self.fc(x)
        return out

def train():
    input_dim = 80
    hidden_dim = 256
    output_dim = 73
    total_epoch = 25

    train_data, val_data = sd.load_data()
    model = SpeechRecognitionModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=0.1,
        steps_per_epoch=int(len(train_data)),
        epochs=total_epoch,
        anneal_strategy='linear')
    
    epoch_losses = []
    val_losses = []
    for epoch in range(total_epoch):
        epoch_loss = 0.0
        
        for i, (features, labels, features_len, labels_len) in enumerate(train_data):
           
            optimizer.zero_grad()
            
            output = model(features, features_len // 2)

            input = torch.nn.functional.log_softmax(output, dim=-1).transpose(0, 1).contiguous()  # (B, T, C)
            # print("Input Shape: ", input.shape)
            # print("Input Stride: ", input.stride())
            # print("Input (contiguous): ", input.is_contiguous())
            loss = criterion(input, labels, features_len // 2, labels_len)

            loss.backward()

            optimizer.step()
            
            pred = torch.argmax(output, dim=-1)
            # print(f"Predicted Shape: ", pred.shape)
            # print(f"Predicted Stride: ", pred.stride())
            # print(f"Predicted (contiguous): ", pred.is_contiguous())
            print(f"\n[Batch {i+1} / {len(train_data)}] Loss: {loss.item():.4f}")
            epoch_loss += loss.item()
            if i == 5:
                print("Features Shape: ", features.shape)
                print("Labels Shape: ", labels.shape)
                print("Features Length: ", features_len)
                print("Labels Length: ", labels_len)
                print("Labels (Sample 0): ", labels[0].tolist())
                print("Predicted (Sample 0): ", pred[0].tolist())
            
        scheduler.step(epoch_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, labels, features_len, labels_len in val_data:
                output = model(features, features_len // 2)

                input = torch.nn.functional.log_softmax(output, dim=-1).transpose(0, 1).contiguous()
                loss = criterion(input, labels, features_len // 2, labels_len)

                val_loss += loss.item()

        val_loss = val_loss / len(val_data)
        
        epoch_loss  = f"{epoch_loss:.2f}"
        val_loss = f"{val_loss:.2f}"
        epoch_losses.append(epoch_loss)
        val_losses.append(val_loss)
        print(f"\n[Epoch {epoch+1}] Loss: {epoch_loss} / Val Loss: {val_loss}")
        print("Epoch Losses: ", epoch_losses)
        print("Validation Losses: ", val_losses)        