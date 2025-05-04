import torch.nn as nn

verbose = True
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
            
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        if verbose: 
            print(f"[After Conv2d] Shape: {x.shape} | Min: {x.min()} | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = self.bn1(x)
        if verbose:
            print(f"[After BN1 in ResBlock] Shape: {x.shape} | Min: {x.min()} | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = self.relu(x)
        if verbose:
            print(f"[After ReLU] Shape: {x.shape} | Min: {x.min()} | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity 
        
        return self.relu(x)

# ======= Model =======
class SpeechRecognitionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.initial_downsample = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=5//2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.layer1 = self._make_layer(32, 64, blocks=2)        
        self.layer2 = self._make_layer(64, 128, blocks=2)
        self.layer3 = self._make_layer(128, 256, blocks=2)
        
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, None)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )     
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, vocab_size)
        )
 
        # nn.init.xavier_uniform_(self.fc[-1].weight, gain=0.01)  # Tiny initial weights
        # nn.init.constant_(self.fc[-1].bias, -3.0)  # Strong blank suppression
        # self.fc[-1].bias.data[0] = 0.0
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if verbose:
            print(f"Input Stats: {x.shape} | Min: {x.min()} |  Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")
        x = self.initial_downsample(x)
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
        x = x.view(x.size(0), x.size(3), -1 ).contiguous()  
        if verbose:
            print(f"[After View] Shape: {x.shape} | Min: {x.min()}  | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")

        x = self.fc(x)
        if verbose:
            print(f"[After FC] Shape: {x.shape} | Min: {x.min()}  | Max: {x.max()} | Std: {x.std()} | Mean: {x.mean()}")

        return x.transpose(0, 1).contiguous() 
    