import torch
import torch.nn as nn
import src.SpeechDataset as sd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import config
import os
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)
       
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.layer_norm = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, verbose=True):
        if verbose:
            print(f"Input Shape: {x.shape}")
            print(f"Input Stride: {x.stride()}")
            print(f"Input (contiguous): {x.is_contiguous()}")
        x = self.conv(x)
        if verbose:
            print(f"Conv Shape: {x.shape}")
            print(f"Conv Stride: {x.stride()}")
            print(f"Conv (contiguous): {x.is_contiguous()}")
        x = x.transpose(1, 3).contiguous()
        if verbose:
            print(f"Transpose Shape: {x.shape}")
            print(f"Transpose Stride: {x.stride()}")
            print(f"Transpose (contiguous): {x.is_contiguous()}")
            
        x = self.layer_norm(x)
        if verbose:
            print(f"Layer Norm Shape: {x.shape}")
            print(f"Layer Norm Stride: {x.stride()}")
            print(f"Layer Norm (contiguous): {x.is_contiguous()}")
        
        x = x.transpose(1, 3).contiguous()
        if verbose:
            print(f"Transpose Shape: {x.shape}")
            print(f"Transpose Stride: {x.stride()}")
            print(f"Transpose (contiguous): {x.is_contiguous()}")
        return x
    
class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_class, n_feats, in_channels, out_channels, rnn_dim, rnn_num_layers=1, dropout_rate=0.2):
        super(SpeechRecognitionModel, self).__init__()
        self.conv1 = Conv2dBlock(in_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), padding=5//2)
        self.conv2 = Conv2dBlock(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=3//2)
        self.conv_down = Conv2dBlock(out_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=3//2)
        self.conv3 = Conv2dBlock(out_channels, out_channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=3//2)
        n_feats = n_feats // 2
        dense_output_1 = (n_feats * (out_channels * 2)) // 2
        self.dense = nn.Sequential(
            nn.Linear(n_feats * (out_channels * 2), dense_output_1),
            nn.LayerNorm(dense_output_1),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dense_output_1, dense_output_1 // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dense_output_1 // 2, 512))
        
        self.gru = nn.GRU(input_size=rnn_dim * 2, hidden_size=rnn_dim, num_layers=rnn_num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_dim * 2, n_class)
        
    def forward(self, x, verbose=False):
        x = self.conv1(x, verbose=verbose)
        x = self.conv2(x, verbose=verbose)
        x = self.conv_down(x, verbose=verbose)
        x = self.conv3(x,   verbose=verbose)
    
        B, C, H, W = x.shape
        if verbose:
            print(f"B: {B} / C: {C} / H: {H} / W: {W}")
        x = x.view(B, W, C * H).contiguous()
        if verbose:
            print(f"View Shape: {x.shape}")
            print(f"View Stride: {x.stride()}")
            print(f"View (contiguous): {x.is_contiguous()}")
        x = self.dense(x)
        if verbose:
            print(f"Dense Shape: {x.shape}")
            print(f"Dense Stride: {x.stride()}")
            print(f"Dense (contiguous): {x.is_contiguous()}")
        x, _= self.gru(x)
        if verbose:
            print(f"GRU Shape: {x.shape}")
            print(f"GRU Stride: {x.stride()}")
        x = self.fc(x)
        if verbose:
            print(f"FC Shape: {x.shape}")
            print(f"FC Stride: {x.stride()}")
            print(f"FC (contiguous): {x.is_contiguous()}")
        return x

def train():
    log_file = os.path.join(config.LOG_DIR, f"{config.LANGUAGE}.log")
    n_feats = 80
    in_channels = 1
    out_channels = 32
    rnn_dim = 256
    vocab_size = 29
    total_epoch = 75
    epoch_losses = []
    val_losses = []
    
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = sd.load_data()
    model = SpeechRecognitionModel(vocab_size, n_feats, in_channels, out_channels, rnn_dim).to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    total_steps = len(train_data) * total_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps, epochs=total_epoch, steps_per_epoch=len(train_data), pct_start=0.1, anneal_strategy='linear')
    
    with open(log_file, 'a') as f:
        for epoch in range(total_epoch):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, (features, labels,features_len, labels_len) in enumerate(train_data):
                optimizer.zero_grad()

                print(f"[Epoch {epoch + 1}] - [Batch {batch_idx+1} / {len(train_data)}]\n")
                f.write(f"[Epoch {epoch + 1}] - [Batch {batch_idx+1} / {len(train_data)}]\n")
                
                features, labels = features.to(device), labels.to(device)
                features = features.unsqueeze(1)
                features = features.transpose(2, 3).contiguous()
                
                output = model(features, verbose=False)

                print(f"Features Stats: Shape: {features.shape} / Min: {features.min()} / Max: {features.max()} / Mean: {features.mean()} / Std: {features.std()}")
                print(f"Output Stats: Shape: {output.shape} / Min: {output.min()} / Max: {output.max()} / Mean: {output.mean()} / Std: {output.std()}")

                if output.isnan().any():
                    raise ValueError("NaN detected in output")

                probs = torch.nn.functional.log_softmax(output, dim=2).transpose(0, 1).contiguous()  
                print(f"Probs Stats: Shape: {probs.shape} / Min: {probs.min()} / Max: {probs.max()} / Mean: {probs.mean()} / Std: {probs.std()}")

                if probs.isnan().any():
                    raise ValueError("NaN detected in input")
                
                print(f"Output: {output}")
                print(f"Softmax: {probs}")
                loss = criterion(probs, labels, features_len//2, labels_len)
                # print(f"Probs Min: {probs.min()} / Max: {probs.max()} / Mean: {probs.mean()} / Std: {probs.std()}")
                # print(f"Input Shape: {probs.shape}")
                # print(f"Input Stride: {probs.stride()}")
                if loss.item() < 0:
                    raise ValueError("Negative loss detected")
                
                f.write(f"Batch Loss: {loss.item():.4f}\n")
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                
                print(f"Scheduler LR: {scheduler.get_last_lr()}")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"Gradient for {name}: {param.grad.mean()}")

                optimizer.step()
                
                preds = torch.argmax(probs, dim=2).transpose(0, 1).contiguous()

                first_sample_len = labels_len[0].item()
                first_sample_labels = labels[:first_sample_len]
                epoch_loss += loss.item()
                print(f"\n[Batch {batch_idx+1} / {len(train_data)}] Loss: {loss.item():.4f}")
                f.write(f"Batch Loss: {loss.item():.4f}\n")
                f.write(f"Target: {first_sample_labels.tolist()} \nPredicted: {preds[0].tolist()}")
                f.write("\n"+"-"* 100 + "\n")
                print(f"Target: {first_sample_labels.tolist()} \nPredicted: {ctc_decoder(preds[0].tolist())}")
            model.eval()
            val_loss = 0.0

            
            with torch.no_grad():
                for features, labels, features_len, labels_len in val_data:
                    features, labels = features.to(device), labels.to(device)
                    features = features.unsqueeze(1)
                    features = features.transpose(2, 3).contiguous()

                    output = model(features, features_len)
                    
                    input = torch.nn.functional.log_softmax(output, dim=-1).transpose(0, 1).contiguous()
                    loss = criterion(input, labels, features_len, labels_len)

                    val_loss += loss.item()

            epoch_loss /= len(train_data)
            scheduler.step()
            val_loss /= len(val_data)
            print(f"\n[Epoch {epoch+1}] Loss: {epoch_loss:.4f} / Val Loss: {val_loss:.4f}")
            f.write(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f} / Val Loss: {val_loss:.4f}\n")
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