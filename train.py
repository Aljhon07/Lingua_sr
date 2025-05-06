import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import src.SpeechDataset as sd
from torch.utils.data import Dataset, DataLoader
import os
import config
from tools import utils
from src.SpeechRecognitionModel import SpeechRecognitionModel
from src.ResNet import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")


def train():
    log_file = os.path.join(config.LOG_DIR, f"test_{config.LANGUAGE}.log")
    model = SpeechRecognitionModel(config.H_PARAMS["VOCAB_SIZE"]).to(device)

    input_params = [p for n,p in model.named_parameters()
                if 'cnn' in n and 'weight' in n]
    conv_params = [p for n,p in model.named_parameters() 
                if 'layer' in n and 'weight' in n]
    norm_params = [p for n,p in model.named_parameters() 
                if 'norm' in n]
    fc_params = [p for n,p in model.named_parameters() 
                if 'fc' in n]
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True) # blank is the last index.
    optimizer = optim.AdamW(model.parameters(), lr=config.H_PARAMS["BASE_LR"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=config.H_PARAMS["BASE_LR"],       # Lower bound
    max_lr=0.005,        # Upper bound
    step_size_up=200,  
    step_size_down=200,
    mode='triangular',  
    cycle_momentum=False 
)
    batch_counter = 0
    loaders = sd.load_data()
    epoch_losses, val_losses= [], []
  
    # num_batches_per_epoch = sum(len(loader) for loader in loaders['train'])
    with open(log_file, "w") as f:
        f.write("")

    with open(log_file, 'a') as f:
        f.write(f"Using device: {device}\n")
        f.write(f"Model Summary: {model}\n")

        for epoch in range(config.H_PARAMS["TOTAL_EPOCH"]):
            torch.cuda.empty_cache() 
            epoch_batch_counter = 0
            epoch_loss = 0.0
            
            # if epoch < 5:
            #     current_train_loaders = loaders['train'][:1]
            #     current_val_loaders = loaders['val'][:1]
            # elif epoch < 5:
            #     current_train_loaders = loaders["train"][:3]
            #     current_val_loaders = loaders["val"][:3]
            # elif epoch < 15:
            #     current_train_loaders = loaders["train"][:5]
            #     current_val_loaders = loaders["val"][:5]
            # else:
            #     current_train_loaders = loaders["train"]
            #     current_val_loaders = loaders["val"]

            current_train_loaders = loaders["train"]
            current_val_loaders = loaders["val"]
            
            num_batches_per_epoch = sum(len(loader) for loader in current_train_loaders)
            
            for idx, (loader) in enumerate(current_train_loaders):
                batch_losses = []
                for batch_idx, batch_data in enumerate(loader):
                    spec, targets, spec_len, target_len, string_labels, audio_path = batch_data
                    
                    if batch_counter == 0:
                        sanity_check(batch_data)

                    model.train()
                    batch_counter += 1
                    epoch_batch_counter += 1
                    
                    spec = spec.to(device)  
                    targets = targets.to(device) 

                    print(f"*"*75)
                    print(f"[Epoch {epoch + 1}] | Dataset: {idx + 1}/{len(current_train_loaders)} | Batch {batch_idx + 1}/{len(loader)}")
                    if (batch_idx + 1) % 10 == 0:
                        f.write(f"[Epoch {epoch + 1}] | Dataset: {idx + 1}/{len(current_train_loaders)} | Batch {batch_idx + 1}/{len(loader)}\n")
                    
                    optimizer.zero_grad()
                    output = model(spec).contiguous()
                    log_softmax = torch.nn.functional.log_softmax(output, dim=-1)
                    
                    print_output_value(output, log_softmax)
                    
                    loss = criterion(log_softmax, targets, spec_len // 2, target_len)
                    if torch.isnan(loss).any() or torch.isnan(output).any() or torch.isnan(output).any():
                        raise ValueError("NaN detected!!")
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(input_params, max_norm=0.3)  # Tighten downsample weights
                    torch.nn.utils.clip_grad_norm_(fc_params, max_norm=0.3) 
                    torch.nn.utils.clip_grad_norm_(norm_params,max_norm=0.3)
                    torch.nn.utils.clip_grad_norm_(conv_params, max_norm=0.5)  # Tighten conv weights

                    epoch_loss += loss.item()
                    batch_losses.append(f"{loss.item():.4f}")
                    print(f"LR: {scheduler.get_last_lr()}")
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            print(f"{name}: {param.grad.norm():.4f}")
                            
                    optimizer.step()
                    scheduler.step()
                    
                    pred_raw = torch.argmax(output, dim=2).transpose(0, 1).contiguous()  # (B, T)
                    print(f"="*75)
                    print(f"Target: {targets[1]}\nRaw Prediction: {pred_raw[1].tolist()}")
                    print(f"\n[Epoch {epoch + 1}] - [Batch {batch_counter}/{num_batches_per_epoch * config.H_PARAMS["TOTAL_EPOCH"]}] Loss: {loss.item():.4f}")
                    print(f"Batch Losses: {batch_losses}")
                    print(f"="*75)
  
                
            val_loss = 0.0
            for idx, (loader) in enumerate(current_val_loaders):
                model.eval()

                with torch.no_grad():
                    for batch_idx, (spec, targets, spec_len, target_len, _, _) in enumerate(loader):
                        spec = spec.to(device)
                        targets = targets.to(device)

                        outputs = model(spec).to(device)
                        input = torch.nn.functional.log_softmax(outputs, dim=-1)
                        
                        loss = criterion(input, targets, spec_len // 2, target_len)
                        preds = torch.argmax(input, dim=2).transpose(0, 1).contiguous()
                        print(f"Target [{target_len[0]}]:  {targets[0].tolist()}\nPredicted: {utils.ctc_decoder(preds[0].tolist())}")
                        print(f"Loss: {loss.item()}")
                        val_loss += loss.item()
                                
            epoch_loss /= num_batches_per_epoch
            val_loss /= sum(len(loader) for loader in current_val_loaders)
            epoch_losses.append(f"{epoch_loss:.2f}")
            val_losses.append(f"{val_loss:.2f}")
            print(f"Epoch {epoch+1}/{config.H_PARAMS["TOTAL_EPOCH"]} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
            print(f"Losses: {epoch_losses}\nVal Losses: {val_losses}")
            print(f"="*50)
            
            f.write(f"Epoch {epoch+1}/{config.H_PARAMS["TOTAL_EPOCH"]} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}\n")

        torch.save({
            "model_dict": model.state_dict(),
            "epoch_losses": epoch_losses,
            "val_losses": val_losses
        }, "final_model.pth")
    

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

def print_output_value(output, log_softmax):
    output_sample = output[:, 0, :]
    softmax_output = log_softmax[:, 0, :]
    print(f"Output: {output.shape} | Min: {output.min():.4f} | Max: {output.max():.4f} | {output.std().item():.4f} | Mean: {output.mean().item():.4f}")
    print(f"Outputs First 10: {output_sample[0][:10].tolist()}")
    print(f"Outputs Last 10: {output_sample[0][-10:].tolist()}")

    print(f"Log_Softmax: {log_softmax.shape} | Min: {log_softmax.min():.4f} | Max: {log_softmax.max():.4f} | {log_softmax.std().item():.4f} | Mean: {log_softmax.mean().item():.4f}")
    print(f"Log_Softmax First 10: {softmax_output[0][:10].tolist()}")
    print(f"Log_Softmax Last 10: {softmax_output[0][-10:].tolist()}")

def sanity_check(batch_data):
    spec, targets, spec_len, target_len, string_labels, audio_paths = batch_data
    print("🧠 Sanity Check: Audio-Label Alignment — Batch 0 -----------")
    print(f"Spectrogram shape: {spec.shape}")
    print(f"Spectrogram length (frames): {spec_len[2].item()}")
    print(f"Target label length: {target_len[2].item()}")
    print(f"Audio path: {audio_paths[2]}")
    print(f"📝 String label (transcription):\n{string_labels[2]}")
    print(f"Target: {targets[2]}")

    utils.plot_spectrogram(spec[2], spec[2])

if __name__ == "__main__":
    train()
    
    