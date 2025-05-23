import torch
import torch.nn as nn
import src.SpeechDataset as sd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import config
import os

class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x, verbose=True):
        x = x.permute(0, 2, 3, 1).contiguous() 
        if verbose:
            print(f"Permuted Shape: {x.shape}")
            print(f"Permuted Stride: {x.stride()}")
            print(f"Permuted (contiguous): {x.is_contiguous()}")
        
        x = self.layer_norm(x)
        if verbose:
            print(f"Layer Norm Shape: {x.shape}")
            print(f"Layer Norm Stride: {x.stride()}")
        
        x = x.permute(0, 3, 1 , 2).contiguous()
        if verbose:
            print(f"Permuted Shape: {x.shape}")
            print(f"Permuted Stride: {x.stride()}")
            
        return x
    
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=0.2):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.layer_norm = CNNLayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, verbose=True):
        x = self.conv(x)
        if verbose:
            print(f"Conv Shape: {x.shape}")
            print(f"Conv Stride: {x.stride()}")
        x = self.layer_norm(x, verbose)
        if verbose:
            print(f"Layer Norm Min: {x.min()} / Max: {x.max()}")
            # print(f"Layer Norm: {x}")
        x = self.relu(x)
        if verbose:
            dead_ratio = (x == 0).float().mean().item()
            print(f"{dead_ratio:.1%} dead neurons")
        x = self.dropout(x)
        return x
    

class BidirectionalGRU(nn.Module):
    def __init__(self,input_dim = 256,  rnn_dim = 256, num_layers=1, dropout=0.1):
        super(BidirectionalGRU, self).__init__()

        self.gru = nn.GRU(input_dim, rnn_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(rnn_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths, verbose=True):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        if verbose:
            print(f"Packed Shape: {packed_x.data.shape}")
            print(f"Packed Stride: {packed_x.data.stride()}")
            
        packed_out, _ = self.gru(packed_x)
        if verbose:
            print(f"GRU Shape: {packed_out.data.shape}")
            print(f"GRU Stride: {packed_out.data.stride()}")
            print(f"Lengths: {lengths}")
        x, _ = pad_packed_sequence(packed_out, batch_first=True)
        if verbose:
            print(f"Unpacked Shape: {x.shape}")
            print(f"Unpacked Stride: {x.stride()}")
            print(f"Unpacked (contiguous): {x.is_contiguous()}")
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, n_class, dropout=0.1):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(output_dim, n_class)
    def forward(self, x, verbose):
        x = self.linear1(x)
        if verbose:
            print(f"Linear 1 Shape: {x.shape}")
            print(f"Linear 1 Stride: {x.stride()}")
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        if verbose:
            print(f"Linear 2 Shape: {x.shape}")
            print(f"Linear 2 Stride: {x.stride()}")
        return x 
       
class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_class, n_feats, in_channels, out_channels, rnn_dim, rnn_num_layers=2, dropout_rate=0.2):
        super(SpeechRecognitionModel, self).__init__()
        
        out_channels_2 = out_channels * 2
        out_channels_3 = out_channels_2 * 2
        self.conv1 = Conv2dBlock(in_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), padding="same", dropout=dropout_rate)
        self.conv2 = Conv2dBlock(out_channels, out_channels_2, kernel_size=(3, 3), stride=(1, 1), padding="same", dropout=dropout_rate)
        self.conv3 = Conv2dBlock(out_channels_2, out_channels_3, kernel_size=(3, 3), stride=(1, 1), padding="same", dropout=dropout_rate)
        
        self.fully_connected = nn.Linear(out_channels_3 * 80, 512)
         
        self.gru = BidirectionalGRU(input_dim=512, rnn_dim=rnn_dim, num_layers=2, dropout=dropout_rate)
        
        self.classifier = Classifier(input_dim=rnn_dim * 2, output_dim=512, n_class=n_class, dropout=0.5)
        
    def forward(self, x, lengths, verbose=True):
        if verbose:
            print(f"Input Shape: {x.shape}")
            print(f"Input Stride: {x.stride()}")
        
        x = self.conv1(x, verbose)
        x = self.conv2(x, verbose)
        x = self.conv3(x, verbose)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        if verbose:
            print(f"After Permute Shape: {x.shape}")
            print(f"After Permute Stride: {x.stride()}")
        
        x = x.view(x.size(0), x.size(1), -1)
        if verbose:
            print(f"After View Shape: {x.shape}")
            print(f"After View Stride: {x.stride()}")
            
        x = self.fully_connected(x)  
        if verbose:
            print(f"After Fully Connected Shape: {x.shape}")
            print(f"After Fully Connected Stride: {x.stride()}")
        
        x = self.gru(x, lengths, verbose)
        x = self.classifier(x, verbose)            
        
        return x

def train():
    
    log_file = os.path.join(config.LOG_DIR, f"{config.LANGUAGE}.log")
    n_feats = 80
    in_channels = 1
    out_channels = 32
    rnn_dim = 256
    vocab_size = 29
    total_epoch = 100
    
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = sd.load_data()
    model = SpeechRecognitionModel(vocab_size, n_feats, in_channels, out_channels, rnn_dim).to(device)
    print(f"Using device: {device}")
    
    total_steps = len(train_data) * total_epoch
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=0.01,
        total_steps=total_steps,
        anneal_strategy='linear')

    epoch_losses = []
    val_losses = []

    with open(log_file, 'a') as f:
        f.write(f"Training {config.LANGUAGE}\n")
        f.write(f"Using device: {device}\n")
        for epoch in range(total_epoch):
            epoch_loss = 0.0
            model.train()
            f.write("="* 100)
            f.write(f"\n[Epoch {epoch+1}]\n")
            f.write("="* 100 + "\n")
            
            for batch_idx, (features, labels, features_len, labels_len) in enumerate(train_data):
                print(f"[Batch {batch_idx+1} / {len(train_data)}]\n")
                f.write(f"[Batch {batch_idx+1} / {len(train_data)}]\n")
                features, labels = features.to(device), labels.to(device)
                features = features.unsqueeze(1)
                features = features.transpose(2, 3).contiguous()

                if torch.isnan(features).any():
                    raise ValueError("NaN values in input features")
                if torch.isnan(labels).any():
                    raise ValueError("NaN values in labels")
                
                print(f"Input: {features}")
                output = model(features, features_len, verbose=False)
                if torch.isnan(output).any():
                    raise ValueError("NaN values in output")
                print(f"Output: {output}")
                input = torch.nn.functional.log_softmax(output, dim=-1).transpose(0, 1).contiguous()  
                loss = criterion(input, labels, features_len, labels_len)
                
                optimizer.zero_grad()
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm(2)
                        print(f"Gradient Norm for {name}: {grad_norm:.4f}")
                optimizer.step()
                
                print(input.shape)
                preds = torch.argmax(output, dim=-1)

                epoch_loss += loss.item()
                print(f"\n[Batch {batch_idx+1} / {len(train_data)}] Loss: {loss.item():.4f}")
                f.write(f"Batch Loss: {loss.item():.4f}\n")
                
                if (batch_idx == 0 and (epoch + 1) % 10 == 0) or (batch_idx == 0 and epoch == 0):
                    print(f"Output Shape: {output.shape}")
                    print("Features Length: ", features_len.shape)
                    print(f"Input Shape: {input.shape}")
                    print(f"Input Stride: {input.stride()}")
                    print(f"Predicted Shape: {preds.shape}")
                    print(f"Predicted Stride: {preds.stride()}")
                print(f"Target: {labels[0].tolist()} \nPredicted: {ctc_decoder(preds[0].tolist())}")
                f.write(f"Target: {labels[0].tolist()} \nPredicted: {preds[0].tolist()}")
                f.write("\n"+"-"* 100 + "\n")
            scheduler.step(epoch_loss / len(train_data))
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                print("Validating...")
                for features, labels, features_len, labels_len in val_data:
                    features, labels = features.to(device), labels.to(device)
                    features = features.unsqueeze(1)
                    features = features.transpose(2, 3).contiguous()

                    output = model(features, features_len)
                    
                    input = torch.nn.functional.log_softmax(output, dim=-1).transpose(0, 1).contiguous()
                    loss = criterion(input, labels, features_len, labels_len)

                    val_loss += loss.item()

            epoch_loss /= len(train_data)
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
# def train():
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     print(f"Using device: {device}")
#     n_mels = 80
#     in_channels = 32
#     out_channels = 64
#     rnn_dim = 256
#     vocab_size = 73
#     total_epoch = 25

#     train_data, val_data = sd.load_data()
#     model = SpeechRecognitionModel(vocab_size, in_channels, out_channels, rnn_dim)
#     criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
#     model.to(device)
    
#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
#         max_lr=0.1,
#         steps_per_epoch=int(len(train_data)),
#         epochs=total_epoch,
#         anneal_strategy='linear')
    
#     epoch_losses = []
#     val_losses = []
#     for epoch in range(total_epoch):
#         epoch_loss = 0.0
        
#         for i, (features, labels, features_len, labels_len) in enumerate(train_data):
#             features, labels = features.to(device), labels.to(device)
#             optimizer.zero_grad()
            
#             output = model(features, features_len // 2, i == 1)

#             input = torch.nn.functional.log_softmax(output, dim=-1).transpose(0, 1).contiguous()  # (B, T, C)
#             loss = criterion(input, labels, features_len // 2, labels_len)

#             loss.backward()

#             optimizer.step()
            
#             pred = torch.argmax(output, dim=-1)
#             # print(f"Predicted Shape: ", pred.shape)
#             # print(f"Predicted Stride: ", pred.stride())
#             # print(f"Predicted (contiguous): ", pred.is_contiguous())
#             print(f"\n[Batch {i+1} / {len(train_data)}] Loss: {loss.item():.4f}")
#             epoch_loss += loss.item()
#             if i % 10 == 0:
#                 print("Features Shape: ", features.shape)
#                 print("Labels Shape: ", labels.shape)
#                 print("Features Length: ", features_len)
#                 print("Labels Length: ", labels_len)
#                 print("Labels (Sample 0): ", labels[0].tolist())
#                 print("Predicted (Sample 0): ", pred[0].tolist())
            
#         scheduler.step(epoch_loss / len(train_data))
#         model.eval()
#         val_loss = 0.0

#         with torch.no_grad():
#             for features, labels, features_len, labels_len in val_data:
#                 features, labels = features.to(device), labels.to(device) 
#                 output = model(features, features_len // 2)

#                 input = torch.nn.functional.log_softmax(output, dim=-1).transpose(0, 1).contiguous()
#                 loss = criterion(input, labels, features_len // 2, labels_len)

#                 val_loss += loss.item()

#         val_loss = val_loss / len(val_data)
        
#         epoch_loss  = f"{epoch_loss / len(train_data):.2f}"
#         val_loss = f"{val_loss / len(val_data):.2f}"
#         epoch_losses.append(epoch_loss)
#         val_losses.append(val_loss)
#         print(f"\n[Epoch {epoch+1}] Loss: {epoch_loss} / Val Loss: {val_loss}")
#         print("Epoch Losses: ", epoch_losses)
#         print("Validation Losses: ", val_losses)        