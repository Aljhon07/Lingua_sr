import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes=10000, rnn_hidden=256, num_layers=2):
        super(CRNN, self).__init__()

        # CNN Feature Extractor (Convolves over the spectrogram)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # Reducing time & frequency

        self.rnn = nn.LSTM(input_size=64 * 40, hidden_size=rnn_hidden, num_layers=num_layers, batch_first=True, bidirectional=True)

        # Fully Connected Layer for classification
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)  # *2 because bidirectional

    def forward(self, x):
        # Reshape: (batch, time, freq) -> (batch, channels, freq, time) for CNN
        x = x.unsqueeze(1)  # Add channel dim: (32, 1, 500, 80)
        
        # CNN feature extraction
        x = self.pool(self.relu((self.conv1(x))))  # (32, 32, 250, 40)
        x = self.pool(self.relu((self.conv2(x))))  # (32, 64, 125, 20)
        
        # Reshape for RNN: (batch, channels, time, freq) -> (batch, time, features)
        x = x.permute(0, 2, 1, 3).contiguous()  # (32, 125, 64, 20)
        x = x.view(x.size(0), x.size(1), -1)  # (32, 125, 64 * 20 = 1280)

        # Pass through RNN
        x, _ = self.rnn(x)  # Output shape: (batch, time, rnn_hidden * 2)
        
        # Fully connected layer to map to vocab size
        x = self.fc(x)  # (batch, time, num_classes)

        return x

# Example usage
batch_size = 32
sequence_length = 500
feature_dim = 80
num_classes = 100

# Input features: (batch, time, frequency)
features = torch.randn(batch_size, sequence_length, feature_dim)

# Initialize model and forward pass
model = CRNN(num_classes=num_classes)
output = model(features)

print(output.shape)  # Expected: (32, 125, 10000) -> (batch, reduced time, vocab size)
