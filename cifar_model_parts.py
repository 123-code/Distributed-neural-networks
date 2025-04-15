
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512) 
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(-1, 64 * 8 * 8) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x) 
        return x
class ModelPart0(nn.Module):
    def __init__(self, original_model: NeuralNetwork):
        super().__init__()
        self.conv1 = original_model.conv1
        self.relu = original_model.relu
        self.pool = original_model.pool
    def forward(self, x):
        return self.pool(self.relu(self.conv1(x))) # Output shape: [batch, 32, 16, 16]

# --- Part 1 (Node 2) ---
class ModelPart1(nn.Module):
    def __init__(self, original_model: NeuralNetwork):
        super().__init__()
        self.conv2 = original_model.conv2
        self.relu = original_model.relu
        self.pool = original_model.pool
    def forward(self, x):
        x = self.pool(self.relu(self.conv2(x))) # Output shape: [batch, 64, 8, 8]
        x = x.view(-1, 64 * 8 * 8) # Flatten. Output shape: [batch, 4096]
        return x

# --- Part 2 (Node 3 - Final) ---
class ModelPart2(nn.Module):
    def __init__(self, original_model: NeuralNetwork):
        super().__init__()
        self.fc1 = original_model.fc1
        self.relu = original_model.relu
        self.fc2 = original_model.fc2
        self.softmax = original_model.softmax
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x) 
        return x

