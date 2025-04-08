import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


x = torch.randn(1, 4, 6)  # Example input tensor
print(x)
x = torch.argmax(x, dim=2)  # Apply argmax along the last dimension
print(x)