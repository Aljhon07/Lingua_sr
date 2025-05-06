import torch.nn as nn
import config
from tools import utils

verbose = config.H_PARAMS["VERBOSE"]
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout2d(dropout)
            
        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        if verbose: 
            print(f"[After Conv2d] Shape: {x.shape} | Min: {x.min()} | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = self.norm1(x)
        if verbose:
            print(f"[After BN1 in ResBlock] Shape: {x.shape} | Min: {x.min()} | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = self.relu(x)
        if verbose:
            print(f"[After ReLU] Shape: {x.shape} | Min: {x.min()} | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        # if self.dropout is not None:
        #     x = self.dropout(x)
        #     if verbose:
        #         print("Dropout2d")
        x = self.conv2(x)
        x = self.norm2(x)

        if self.shortcut is not None:
            identity = self.shortcut(identity)
        x += identity 
        return self.relu(x)

# ======= Model =======
class SpeechRecognitionModel(nn.Module):
    def __init__(self, vocab_size=1000,):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=5//2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.layer1 = self._make_layer(32, 64, blocks=2, dropout=0.1)        
        self.layer2 = self._make_layer(64, 128, blocks=2, dropout=0.3)
        self.layer3 = self._make_layer(128, 256, blocks=2, dropout=0.3)
        
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AdaptiveAvgPool2d((8, None))
        )     
        
        self.gru = nn.GRU(
            input_size=128*8,  # From AdaptiveAvgPool output
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        # Remove Dropout, add LayerNorm
        self.fc = nn.Linear(128 * 8, vocab_size)
        
    def _make_layer(self, in_channels, out_channels, blocks, dropout=None):
        layers = []
        for _ in range(blocks):
            layers.append(ResidualBlock(in_channels, out_channels, dropout=dropout))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if verbose:
            print(f"Input Stats: {x.shape} | Min: {x.min()} |  Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = self.cnn(x) # [B, C, F, T]
        if verbose:
            print(f"[Initial Downsample] Shape: {x.shape}  | Min: {x.min()} | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = self.layer1(x)
        if verbose:
            print(f"[Layer 1] Shape: {x.shape} | Min: {x.min()} | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = self.layer2(x)
        if verbose:
            print(f"[Layer 2] Shape: {x.shape} | Min: {x.min()}  | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")

        # x = self.layer3(x)
        # if verbose:
        #     print(f"After layer3: {x.shape}")
        
        x = self.pool(x)
        if verbose:
            print(f"[After Pooling] Shape: {x.shape} | Min: {x.min()}  | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = x.view(x.size(0), x.size(3), -1 ).contiguous()  # [B, T, F]
        if verbose:
            print(f"[After View] Shape: {x.shape} | Min: {x.min()}  | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")

        # x, _ = self.gru(x)
        # if verbose:
        #     print(f"[After GRU] Shape: {x.shape} | Min: {x.min()}  | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
            
        x = self.fc(x)
        if verbose:
            print(f"[After FC] Shape: {x.shape} | Min: {x.min()}  | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")

        return x.transpose(0, 1).contiguous() #[T, B, C]
    