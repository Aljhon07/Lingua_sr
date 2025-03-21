import torch
import torch.nn as nn
import torch.optim as optim
import config
import tools.load_data as ld
import os

class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x: (batch, channel, feature, time)
        print("CNN Layer (before): ", x)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)

class SimpleSpeechModel(nn.Module):
    def __init__(self, input_dim, cnn_output_dim, vocab_size, dropout_prob=0.3):
        super(SimpleSpeechModel, self).__init__()
        
        # 1-layer CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_output_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Fully connected layer for output
        self.fc = nn.Linear(32 * 50 * 80, vocab_size)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        
        # Add channel dimension for CNN
        x = x.unsqueeze(1)  # (batch, 1, height, width)
        print(f"After unsqueeze: {x.shape}")
        
        x = self.cnn(x)
        print(f"After CNN: {x.shape}")

        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, -1)  # (batch, features)
        print(f"After flatten: {x.shape}")

        x = self.fc(x)  # (batch, vocab_size)
        print(f"After FC: {x.shape}")

        return x.log_softmax(1)  # Ensure log-softmax is applied before loss


if __name__ == "__main__":
    input_dim = 80  
    cnn_output_dim = 32
    vocab_size = 10000  
    dropout_prob = 0.3

    model = SimpleSpeechModel(input_dim, cnn_output_dim, vocab_size, dropout_prob)
    print(model)
    train_loader = ld.load_data(config.PATHS["output"], f"{config.PATHS['base_output']}/{config.STAGE}.tsv")

    criterion = nn.CTCLoss(blank=0)  
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (file_names, features, transcriptions, transcription_lengths) in enumerate(train_loader):
            print(f"Processing Batch {batch_idx + 1}")
            print(f"File Names: {file_names}")
            print(f"Features Shape: {features.shape}")
            print(f"Transcriptions: {transcriptions}")
            print(f"Transcription Lengths: {transcription_lengths}")
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            
            # Prepare inputs for CTC loss
            input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long)
            target_lengths = transcription_lengths
            
            # Compute CTC loss
            loss = criterion(outputs, transcriptions, input_lengths, target_lengths)
            
            # Print model predictions and actual values
            print(f"Model Predictions: {outputs.argmax(1)}")
            print(f"Actual Values: {transcriptions}")
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 10 == 9:  # Print every 10 batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0
        
        # Validate model
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

    print("Training complete.")
