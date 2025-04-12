import torch
import torch.nn as nn
import torch.optim as optim

# ======= Config =======
BATCH_SIZE = 2
SEQ_LEN = 200  # Increased sequence length
INPUT_DIM = 80  # Increased input dimension
TARGET_LEN = 10 # Increased target length
VOCAB_SIZE = 500  # Increased vocabulary size

# ======= Hardcoded Inputs =======
# Generate random inputs with the new shape
inputs = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM, dtype=torch.float32)

# Input lengths can vary
input_lengths = torch.tensor([SEQ_LEN, SEQ_LEN - 50], dtype=torch.long) # example length variation

# Generate random targets within the new vocabulary size
targets = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, TARGET_LEN), dtype=torch.long)
target_lengths = torch.tensor([10, 5], dtype=torch.long) # example of target length variation

# ======= Model =======
class SimpleCTCModel(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 4, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(8, vocab_size + 1)  # +1 for optional blank

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.contiguous()  # Ensure contiguous memory layout
        x = self.fc(x)
        return x.transpose(0, 1)  # (T, B, C)

# ======= Training Pipeline =======
model = SimpleCTCModel(INPUT_DIM, VOCAB_SIZE)
criterion = nn.CTCLoss(blank=0, zero_infinity=True) # blank is the last index.
# optimizer = optim.Adam(model.parameters(), lr=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.1)


for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs).contiguous()
    input = torch.nn.functional.log_softmax(outputs, dim=-1)  # (T, B, C)
    loss = criterion(input, targets, input_lengths, target_lengths)
    loss.backward()
    optimizer.step()

    print(f"\n[Epoch {epoch+1}] Loss: {loss.item():.4f}")

    # Raw predictions (no CTC decoding)
    pred_raw = torch.argmax(outputs, dim=2).transpose(0, 1).contiguous()  # (B, T)

    # Target reconstruction
    target_split = []
    offset = 0
    for length in target_lengths:
        target_split.append(targets[offset:offset + length].tolist())
        offset += length

    # Print predictions vs targets
    if (epoch + 1) % 20 == 0:
        print(f"Sample {1} - Target: {targets[1]}, Raw Prediction: {pred_raw[1].tolist()}")