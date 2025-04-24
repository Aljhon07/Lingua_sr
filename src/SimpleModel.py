import torch
import torch.nn as nn
import src.SpeechDataset as sd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import config
import os
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
verbose = False

class CNNLayerNorm(nn.Module):
   def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)
        nn.init.zeros_(self.layer_norm.bias)
   def forward(self, x):
       # x (batch, channel, feature, time)
       x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
       if verbose:
           print(f"Layer Norm Transposed Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")
       x = self.layer_norm(x)
       return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 
   
   
       
class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = CNNLayerNorm(n_feats)

        
    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm(x)
        if verbose:
            print(f"Layer Norm Output Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")
        x = F.gelu(x)   
        x = self.dropout(x)
        x = self.cnn1(x)
        if verbose:
            print(f"Conv1 Output Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")
        x = self.layer_norm(x)
        if verbose:
            print(f"Layer Norm2 Output Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")
        x = F.gelu(x)   
        x = self.dropout(x)

        x = self.cnn2(x)
        if verbose:
            print(f"Conv2 Output Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")
        x += residual
        return x # (batch, channel, feature, time)
   
class BidirectionalGRU(nn.Module):
   def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
       super(BidirectionalGRU, self).__init__()

       self.BiGRU = nn.GRU(
           input_size=rnn_dim, hidden_size=hidden_size,
           num_layers=1, batch_first=batch_first, bidirectional=True)
       self.layer_norm = nn.LayerNorm(rnn_dim)
       self.dropout = nn.Dropout(dropout)

   def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        if verbose:
                print(f"Layer Norm Output Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")
        x, _ = self.BiGRU(x)
        x = x.contiguous()
        if verbose:
                print(f"BiGRU Output Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")
        x = self.dropout(x)
        return x    
   
   
class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_class, n_feats, in_channels, out_channels, rnn_dim, rnn_num_layers=1, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats // 2

        self.rescnn_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=3//2),
            *[
           ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
           for _ in range(2)
       ])
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
        
        self.birnn_layers = nn.Sequential(*[
           BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                            hidden_size=rnn_dim, dropout=dropout, batch_first=True)
           for i in range(3)
       ])
        
        self.classifier = nn.Sequential(
           nn.Linear(rnn_dim*2, rnn_dim), 
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(rnn_dim, n_class)
       )
        
    def forward(self, x):
        if verbose:
            print(f"Input Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")

        x = self.rescnn_layers(x)
        if verbose:
            print(f"Residual CNN Output Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")

        sizes = x.size()
        if verbose:
            print(f"Sizes: {sizes}")
            
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]).contiguous()  # (batch, feature, time)
        if verbose:
            print(f"View Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")
            
        x = x.transpose(1, 2).contiguous() # (batch, time, feature)
        if verbose:
            print(f"Transpose Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")
            
        x = self.fully_connected(x)
        if verbose:
            print(f"Fully Connected Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")
            
        x = self.birnn_layers(x)
        
        x = self.classifier(x)
        if verbose:
            print(f"Classifier Shape: {x.shape} / Stride: {x.stride()} / Contiguous: {x.is_contiguous()}")

        return x

def train():
    log_file = os.path.join(config.LOG_DIR, f"{config.LANGUAGE}.log")
    n_feats = 80
    in_channels = 1
    out_channels = 32
    rnn_dim = 512
    vocab_size = 1000
    total_epochs = 75
    epoch_losses = []
    val_losses = []
    
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loaders = sd.load_data()
    model = SpeechRecognitionModel(vocab_size, n_feats, in_channels, out_channels, rnn_dim).to(device)
    
    criterion = nn.CTCLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.005)
    num_batches_per_epoch = sum(len(loader) for loader in data_loaders['train'])
    total_steps = num_batches_per_epoch * total_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=total_epochs, steps_per_epoch=num_batches_per_epoch, anneal_strategy='linear')
    
    print(f"Very Short Dataset: {len(data_loaders['train'][0].dataset)}")
    print(f"Short Dataset: {len(data_loaders['train'][1].dataset)}")
    print(f"Validation Dataset: {len(data_loaders['val'][0].dataset) + len(data_loaders['val'][1].dataset)}")
    
    batch_counter = 0
    with open(log_file, 'a') as f:
        for epoch in range(total_epochs):
            model.train()
            epoch_loss = 0.0
            batch_counter = 0
            for idx, (loader) in enumerate(data_loaders['train']):
                for batch_idx, (features, labels, features_len, labels_len, string_labels, audio_path) in enumerate(loader):
                    optimizer.zero_grad()
                    batch_counter += 1
                    if batch_idx == 0 and epoch == 0:
                        print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}\n")
                        print(f"Features Shape: {features.shape}\n")
                        print(f"Labels Shape: {labels.shape}\n")
                        print(f"Features Length: {features_len}\nShape: {features_len.shape}\n")
                        print(f"Labels Length: {labels_len}\nShape: {labels_len.shape}\n")
                        
                        f.write(f"Model Summary: {model}\n")
                        f.write(f"Model Parameters: {sum(p.numel() for p in model.parameters())}\n")
                        f.write(f"Features Shape: {features.shape}\n")
                        f.write(f"Labels Shape: {labels.shape}\n")
                        f.write(f"Features Length: {features_len}\nShape: {features_len.shape}\n")
                        f.write(f"Labels Length: {labels_len}\nShape: {labels_len.shape}\n")
                        
                        print(f"Audio Path: {audio_path[2]}\nLabels: {labels[2]}\nString Label: {string_labels[2]}\nLabels Length: {labels_len[2]}\nActual Length: {len(string_labels[2])}\n")
                        
                    print(f"{"-"*75}\n[Epoch {epoch + 1}] - [Batch {batch_counter} / {num_batches_per_epoch}]\n{"-"*75}")
                    f.write(f"[Epoch {epoch + 1}] - [Batch {batch_counter} / {num_batches_per_epoch}]\n")
                    
                    features, labels = features.to(device), labels.to(device)
                    
                    output = model(features)

                    print(f"Features Stats: Shape: {features.shape} / Min: {features.min()} / Max: {features.max()} / Mean: {features.mean()} / Std: {features.std()}")
                    print(f"Output Stats: Shape: {output.shape} / Min: {output.min()} / Max: {output.max()} / Mean: {output.mean()} / Std: {output.std()}")

                    if output.isnan().any():
                        raise ValueError("NaN detected in output")

                    probs = F.log_softmax(output, dim=2)
                    
                    print(f"Output: {output}")
                    print(f"Log Softmax: {probs}")
                    probs = probs.transpose(0, 1).contiguous()
                    print(f"Probs Stats: Shape: {probs.shape} / Min: {probs.min()} / Max: {probs.max()} / Mean: {probs.mean()} / Std: {probs.std()}")

                    if output.isnan().any():
                        print(f"Output: {output}")
                        print(f"Softmax: {probs}")
                        raise ValueError("NaN detected in output")

                    if probs.isnan().any():
                        print(f"Output: {output}")
                        print(f"Softmax: {probs}")
                        raise ValueError("NaN detected in input")

                    loss = criterion(probs, labels, features_len // 2, labels_len)
                    loss.backward()
                    if loss.item() < 0 :
                        raise ValueError("Negative loss detected")
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    
                    print(f"Scheduler LR: {scheduler.get_last_lr()}")
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm(2)
                            print(f"Layer: {name}, Gradient Norm: {grad_norm.item():.4f}")

                    optimizer.step()
                    scheduler.step()

                    preds = torch.argmax(probs, dim=2).transpose(0, 1).contiguous()
                    epoch_loss += loss.item()
                    print(f"\n[Batch {batch_counter} / {num_batches_per_epoch}] Loss: {loss.item():.4f}")
                    f.write(f"Batch Loss: {loss.item():.4f}\n")
                    f.write(f"Target: {labels[0].tolist()} \nPredicted: {preds[0].tolist()}")
                    f.write("\n"+"-"* 100 + "\n")
                    print(f"Target: {labels[0].tolist()} \nPredicted: {ctc_decoder(preds[0].tolist())}")
            
            for loader in data_loaders['val']:            
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for features, labels, features_len, labels_len, _, _ in loader:
                        features, labels = features.to(device), labels.to(device)
                        output = model(features)
                        input = output.log_softmax(2).transpose(0, 1).contiguous()
                        loss = criterion(input, labels, features_len // 2, labels_len)

                        print(f"Target: {labels[0].tolist()} \nPredicted: {ctc_decoder(preds[0].tolist())}")
                        val_loss += loss.item()

            epoch_loss /= num_batches_per_epoch
            val_loss /= sum(len(loader) for loader in data_loaders['val'])
            print(f"\n[Epoch {epoch+1}] Loss: {epoch_loss:.4f} / Val Loss: {val_loss:.4f}")
            f.write(f">>>[Epoch {epoch+1}] Loss: {epoch_loss:.4f} / Val Loss: {val_loss:.4f}<<<\n")
            f.write("\n"+"-"* 100 + "\n")
            
            val_losses.append(f"{val_loss:.3f}")
            epoch_losses.append(f"{epoch_loss:.3f}")
            print("Epoch Losses: ", epoch_losses)
            print("Validation Losses: ", val_losses)

def ctc_decoder(preds):
    decoded = []
    prev_char = None
    for char_idx in preds:
        if char_idx != 0 and char_idx != prev_char:
            decoded.append(char_idx)
        prev_char = char_idx

    return decoded           

if __name__ == "__main__":
    train()