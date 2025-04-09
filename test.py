import torch

import torch.nn as nn

import torch.optim as optim


# ======= Config =======

BATCH_SIZE = 2

SEQ_LEN = 4

INPUT_DIM = 3

TARGET_LEN = 2

VOCAB_SIZE = 4  # Tokens: 1–4, with 0 as optional blank


# ======= Hardcoded Inputs =======

inputs = torch.tensor([

    [  # Sample 1

        [-0.1, -0.2, 0.3],

        [0.4, 0.5, -0.6],

        [0.7, 0.8, 0.9],

        [1.0, 1.1, 1.2]

    ],

    [  # Sample 2

        [0.3, -0.2, 0.1],

        [-0.6, 0.5, 0.4],

        [0.9, -0.8, 0.7],

        [100, 100, 100]

    ]

], dtype=torch.float32)


input_lengths = torch.tensor([SEQ_LEN, SEQ_LEN - 1], dtype=torch.long)

targets = torch.tensor([[1, 2], [2, 4]], dtype=torch.long)

target_lengths = torch.tensor([2, 1], dtype=torch.long)


# ======= Model =======

class SimpleCTCModel(nn.Module):

    def __init__(self, input_dim, vocab_size):

        super().__init__()

        self.lstm = nn.LSTM(input_dim, 4, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(8, vocab_size + 1)  # +1 for optional blank


    def forward(self, x):

        x, _ = self.lstm(x)

        x = self.fc(x)

        return x.transpose(0, 1)  # (T, B, C)


# ======= Training Pipeline =======

model = SimpleCTCModel(INPUT_DIM, VOCAB_SIZE)

criterion = nn.CTCLoss(blank=4, zero_infinity=True)

optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(100):

    optimizer.zero_grad()


    outputs = model(inputs)  # (T, B, C)

    input = torch.nn.functional.log_softmax(outputs, dim=-1)  # (T, B, C)


    loss = criterion(input, targets, input_lengths, target_lengths)

    loss.backward()

    optimizer.step()


    print(f"\n[Epoch {epoch+1}] Loss: {loss.item():.4f}")

    print(outputs.shape)  # (T, B, C)

    # Raw predictions (no CTC decoding)

    pred_raw = torch.argmax(outputs, dim=2).transpose(0, 1)  # (B, T)


    # Target reconstruction

    target_split = []

    offset = 0

    for length in target_lengths:

        target_split.append(targets[offset:offset+length].tolist())

        offset += length


    # Print predictions vs targets

    for i in range(BATCH_SIZE):

        print(f"Sample {i+1} - Target: {targets[i]}, Raw Prediction: {pred_raw[i].tolist()}") 
        