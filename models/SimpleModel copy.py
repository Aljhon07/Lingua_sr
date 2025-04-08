import torch
import torch.nn as nn
import torch.optim as optim
import config
import tools.load_data as ld
import os

class SimpleSpeechModel(nn.Module):
    def __init__(self, input_dim, cnn_output_dim, vocab_size, dropout_prob=0.3):
        super(SimpleSpeechModel, self).__init__()

        # 1-layer CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_output_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.AdaptiveAvgPool2d((50, 80))  # Output shape will be (batch_size, channels, 50, 80)
        )

        # Fully connected layer for output
        # Flatten the output from CNN, so input to FC is (batch_size, time_steps, cnn_output_dim * width)
        self.fc = nn.Linear(cnn_output_dim * 80, vocab_size)  # 32 * 80 = 2560

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        
        x = x.unsqueeze(1)  # Add channel dimension, shape becomes [batch_size, 1, height, width]
        print(f"After unsqueeze: {x.shape}")
        
        x = self.cnn(x)  # Pass through CNN
        print(f"After CNN: {x.shape}")  # Output shape will be [batch_size, channels, 50, 80]
        
        batch_size, channels, height, width = x.shape
        # Flatten CNN output to [batch_size, time_steps, features]
        x = x.view(batch_size, height, -1)  # Flatten, shape becomes [batch_size, time_steps, features]
        print(f"After flatten: {x.shape}")
        
        # Pass through fully connected layer
        x = self.fc(x)  # Output shape will be [batch_size, time_steps, vocab_size]
        print(f"After FC: {x.shape}")
        print(f"Softmax shape: {x.log_softmax(-1).shape}")
        return x.log_softmax(-1).transpose(0,1)  # Log softmax for CTC loss




def main():
    input_dim = 80  
    cnn_output_dim = 32
    vocab_size = 10000  
    dropout_prob = 0.3

    model = SimpleSpeechModel(input_dim, cnn_output_dim, vocab_size, dropout_prob)
    train_loader = ld.load_data(config.PATHS["output"], f"{config.PATHS['base_output']}/{config.STAGE}.tsv")

    criterion = nn.CTCLoss(blank=0)  
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n🌀 Epoch [{epoch + 1}/{num_epochs}]")
        model.train()
        running_loss = 0.0
        val_loss = 0.0
        for batch_idx, (file_names, features, transcriptions, transcription_lengths, transcription_strs) in enumerate(train_loader):
            print(f"\n🔁 Batch {batch_idx + 1}")
            print(f"Features Shape: {features.shape}")
            print(f"Transcriptions: {transcriptions.shape}")
            print(f"Transcription Lengths: {transcription_lengths}")
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            
            # Prepare inputs for CTC loss
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
            print(f"Output Shape: {outputs.shape}")
            print(f"Output Size: {outputs.size(1)}")
            print(f"Input Lengths: {input_lengths}")
            print("-" * 20)

            target_lengths = transcription_lengths
            
            # Compute CTC loss
            loss = criterion(outputs, transcriptions, input_lengths, target_lengths)
            running_loss += loss.item()
            val_loss += loss.item()

            preds = outputs.argmax(dim=-1)  # [T, B]
            

            print(f"🔠 Actual Transcription: {transcriptions[0]}")
            print(f"🔠 Actual Transcription (Shape): {transcriptions[0].shape}")
            print(f"\n🔮 Predicted Transcription : {preds[0]}")
            print(f"📉 Loss: {loss.item():.4f}")
            print("-" * 40)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Batch [{batch_idx + 1}], Loss: {loss.item():.4f}")
            print("-" * 20)

        
        # Validate model
        avg_loss = running_loss / len(train_loader)
        print(f"\n✅ Epoch [{epoch + 1}] Avg Loss: {avg_loss:.4f}")

    print("Training complete.")
