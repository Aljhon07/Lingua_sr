import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import src.SpeechDataset as sd
from torch.utils.data import Dataset, DataLoader
import os
import config

verbose = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity 
        return self.relu(out)

# ======= Model =======
class SimpleCTCModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=5//2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(32, 64, blocks=2)        
        self.layer2 = self._make_layer(64, 128, blocks=2)
        self.layer3 = self._make_layer(128, 256, blocks=2)
        
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )     
        
        self.fc = nn.Linear(256, vocab_size)  # Adjusted to match the new input size
        # self.lstm = nn.LSTM(256, 256, num_layers=3, batch_first=True, bidirectional=True)
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(512, vocab_size)  # Adjusted to match the new vocabulary size
        # )
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if verbose:
            print(f"Input Shape: {x.shape}")
        x = F.gelu(self.conv1(x))
        x = self.bn1(x)
        x = self.layer1(x)
        if verbose:
            print(f"After layer1: {x.shape}")
        x = self.layer2(x)
        if verbose:
            print(f"After layer2: {x.shape}")
        x = self.layer3(x)
        if verbose:
            print(f"After layer3: {x.shape}")
        x = self.pool(x)
        if verbose:
            print(f"After pooling: {x.shape}")
        x = x.view(x.size(0), x.size(3), -1).contiguous()  
        if verbose:
            print(f"After view: {x.shape}")
        x = self.fc(x)
  
        return x.transpose(0, 1).contiguous()  # (T, B, C)

# ======= Training Pipeline =======
def train():
    model = SimpleCTCModel(40).to(device)
    total_epochs = 100
    log_file = os.path.join(config.LOG_DIR, f"test_{config.LANGUAGE}.log")
    criterion = nn.CTCLoss(blank=0, zero_infinity=True) # blank is the last index.
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    batch_counter = 0
    loaders = sd.load_data()
    # num_batches_per_epoch = sum(len(loader) for loader in loaders['train'])
    with open(log_file, 'a') as f:
        f.write(f"Using device: {device}\n")
        f.write(f"Model Summary: {model}\n")

        for epoch in range(total_epochs):
            epoch_batch_counter = 0
            epoch_loss = 0.0
            model.train()

            if epoch < 25:
                current_train_loaders = loaders['train'][:2]  # Use only first 2 datasets
            elif 25 <= epoch < 50:
                current_train_loaders = loaders['train'][:5]  # Use first 5 datasets
            else:
                current_train_loaders = loaders['train']
            
            num_batches_per_epoch = sum(len(loader) for loader in current_train_loaders)
            
            for idx, (loader) in enumerate(current_train_loaders):
                for batch_idx, (spec, targets, spec_len, target_len, _, _) in enumerate(loader):
                    batch_counter += 1
                    epoch_batch_counter += 1
                    
                    spec = spec.to(device)  
                    targets = targets.to(device) 
                    spec_len = spec_len.to(device) 
                    target_len = target_len.to(device)
                    
                    print(f"[Epoch {epoch + 1}] | Batch {batch_counter}/{num_batches_per_epoch}")
                    if (batch_idx + 1) & 10 == 0 
                        f.write(f"[Epoch {epoch + 1}] | Batch {batch_counter}/{num_batches_per_epoch}\n")
                        
                    optimizer.zero_grad()
                    outputs = model(spec).contiguous()
                    print(f"Output: ", outputs.shape)
                    output = torch.nn.functional.log_softmax(outputs, dim=-1)
                    
                    if torch.isnan(loss).any() or torch.isnan(outputs).any() or torch.isnan(outputs).any():
                        raise ValueError("NaN detected!!")
                    
                    loss = criterion(output, targets, spec_len // 2, target_len)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    epoch_loss += loss.item()
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            print(f"Gradient for {name}: {param.grad.norm():.4f}")
                    optimizer.step()
                    
                    # Raw predictions (no CTC decoding)
                    pred_raw = torch.argmax(outputs, dim=2).transpose(0, 1).contiguous()  # (B, T)
                    # Print predictions vs targets
                    print(f"Target: {targets[1]}\nRaw Prediction: {pred_raw[1].tolist()}")
                    print(f"\n[Epoch {epoch + 1}] - [Batch {batch_counter}/{num_batches_per_epoch * total_epochs}] Loss: {loss.item():.4f}")
                    if (batch_idx + 1) & 10 == 0 
                        f.write(f"[Epoch {epoch + 1}] - [Batch {batch_counter}/{num_batches_per_epoch * total_epochs}] Loss: {loss.item():.4f}\n")
                        f.write(f"Target: {targets[1]}\nRaw Prediction: {pred_raw[1].tolist()}\n")
                    if(loss.item() < 0.5):
                        torch.save(model.state_dict(), "model.pth")
                        print("Loss is too low, stopping training.")
                        break
                        
                    if batch_counter == 1 or batch_counter == 20 or (batch_counter % 50 == 0 and batch_counter < 350):
                        save_checkpoint(model, optimizer, epoch + 1, loss.item(), filename=f"checkpoint_batch_{batch_counter}.pth")
                        print(f"Checkpoint saved at epoch {epoch}")
                    
                    if (epoch + 1) % 50 == 0 and batch_idx == 0:
                        save_checkpoint(model, optimizer, epoch + 1, loss.item(), filename=f"checkpoint_epoch_{epoch}.pth")
                        print(f"Checkpoint saved at epoch {epoch}")

            scheduler.step(epoch_loss / num_batches_per_epoch)

            if epoch < 25:
                current_val_loaders = loaders['val'][:2]  # Use only first 2 datasets
            elif 25 <= epoch < 50:
                current_val_loaders = loaders['val'][:5]  # Use first 5 datasets
            else:
                current_val_loaders = loaders['val']
                
            val_loss = 0.0
            for idx, (loader) in enumerate(current_val_loaders):
                model.eval()

                with torch.no_grad():
                    for batch_idx, (spec, targets, spec_len, target_len, _, _) in enumerate(loader):
                        outputs = model(spec)
                        input = torch.nn.functional.log_softmax(outputs, dim=-1)
                        
                        loss = criterion(input, targets, spec_len // 2, target_len)
                        preds = torch.argmax(input, dim=2).transpose(0, 1).contiguous()
                        print(f"Loss: {loss.item()}")
                        print(f"Target: {targets[0].tolist()}\nPredicted: {ctc_decoder(preds[0].tolist())}")
                        val_loss += loss.item()
                                
            epoch_loss /= num_batches_per_epoch
            val_loss /= sum(len(loader) for loader in current_val_loaders)
            print(f"Epoch {epoch+1}/{total_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
            f.write(f"Epoch {epoch+1}/{total_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}\n")

        torch.save(model.state_dict(), "final_model.pth")
    
def ctc_decoder(preds):
    decoded = []
    prev_char = None
    for char_idx in preds:
        if char_idx != 0 and char_idx != prev_char:
            decoded.append(char_idx)
        prev_char = char_idx
    return decoded  

def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"✅ Saved checkpoint at epoch {epoch}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint.get('loss', None)
    print(f"📦 Loaded checkpoint from epoch {start_epoch}")
    return start_epoch, loss


BATCH_SIZE = 32
SEQ_LEN = 501  # Example sequence length
INPUT_DIM = 80  # Input feature dimension
TARGET_LEN = 20  # Example target sequence length
VOCAB_SIZE = 500  # Vocabulary size for targets

class FixedCTCDataset(Dataset):
    def __init__(self, total_samples):
        self.inputs = [torch.randn(INPUT_DIM, SEQ_LEN).unsqueeze(0) for _ in range(total_samples)]
        self.input_lengths = [torch.randint(SEQ_LEN - 50, SEQ_LEN + 1, (1,)).item() for _ in range(total_samples)]
        self.targets = [torch.randint(1, VOCAB_SIZE, (TARGET_LEN,)) for _ in range(total_samples)]
        self.target_lengths = [torch.randint(10, TARGET_LEN + 1, (1,)).item() for _ in range(total_samples)]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.input_lengths[idx], self.targets[idx], self.target_lengths[idx]
    
# Collate function to batch variable-length inputs
def collate_fn(batch):
    inputs, input_lengths, targets, target_lengths = zip(*batch)
    inputs = torch.stack(inputs)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    targets = torch.stack(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return inputs, input_lengths, targets, target_lengths

# Create DataLoader

def test():

    # # Generate random input data
    # inputs = torch.randn(BATCH_SIZE, INPUT_DIM, SEQ_LEN , dtype=torch.float32)
    # inputs = inputs.unsqueeze(1)  

    # # Input lengths (can vary per sample)
    # input_lengths = torch.randint(low=SEQ_LEN - 50, high=SEQ_LEN + 1, size=(BATCH_SIZE,), dtype=torch.long)

    # # Random target data within vocabulary
    # targets = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, TARGET_LEN), dtype=torch.long)

    # # Target lengths (can also vary per sample)
    # target_lengths = torch.randint(low=10, high=TARGET_LEN + 1, size=(BATCH_SIZE,), dtype=torch.long)

    dataset = FixedCTCDataset(600)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = SimpleCTCModel(VOCAB_SIZE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)  # blank is the last index.
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    for epoch in range(20):  # Loop through dataset 3 times (just an example)
        print(f"\n--- Epoch {epoch + 1} ---")
        for i, (inputs, input_lengths, targets, target_lengths) in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(inputs)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
            preds = torch.argmax(log_probs, dim=2).transpose(0, 1).contiguous()

          
                
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient for {name}: {param.grad.norm():.4f}")
            optimizer.step()

            print(f"Target: {targets[1]}\nRaw Prediction: {preds[1].tolist()}")
            print(f"Epoch {epoch + 1} | Batch {i + 1}/{len(dataloader)} | Loss: {loss.item():.4f}")
            print(f"="*50)
            
            
        
    
if __name__ == "__main__":
    # test()
    train()
    
    