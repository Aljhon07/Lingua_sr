import torch.nn as nn
import torch.nn.functional as F

verbose = False

class CNNLayerNorm(nn.Module):
   def __init__(self, n_feats):
       super(CNNLayerNorm, self).__init__()
       self.layer_norm = nn.LayerNorm(n_feats)

   def forward(self, x):
       # x (batch, channel, feature, time)
       x = x.transpose(2, 3).contiguous()
       if verbose:
           print(f"[After Tranpose]: {x.shape}")
       x = self.layer_norm(x)
       return x.transpose(2, 3).contiguous()
   
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2, n_feats=80):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=True)

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        # self.norm1 = CNNLayerNorm(n_feats)
        # self.norm2 = CNNLayerNorm(n_feats)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
            
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x
        if verbose: 
            print(f"[After Conv2d] Shape: {x.shape} | Min: {x.min()} | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = self.norm1(x)
        if verbose:
            print(f"[After Normalization in ResBlock] Shape: {x.shape} | Min: {x.min()} | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = self.relu(x)
        if verbose:
            print(f"[After GELU] Shape: {x.shape} | Min: {x.min()} | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = self.dropout1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.relu(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.dropout2(x)
        x = self.conv2(x)
        x = x + identity 
        return x

class ResNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.cnn = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=5//2)
        
        self.layer1 = self._make_layer(32, 64, blocks=2)        
        self.layer2 = self._make_layer(64, 128, blocks=2)
        self.layer3 = self._make_layer(128, 256, blocks=2)
        
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

        self.fc = nn.Linear(128 * 8, vocab_size)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
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
    