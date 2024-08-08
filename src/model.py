import torch
import torch.nn as nn
import torch.nn.functional as F

class VoiceVerificationModel(nn.Module):
    def __init__(self):
        super(VoiceVerificationModel, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Linear layers
        self.fc1 = nn.Linear(2 * 64 * 1000, 256)  # Adjusted input size after pooling
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward_cnn(self, x):
        # x: (batch_size, 1, 8000)
        x = F.relu(self.conv1(x))
        # x: (batch_size, 16, 8000)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        # x: (batch_size, 16, 4000)
        
        x = F.relu(self.conv2(x))
        # x: (batch_size, 32, 4000)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        # x: (batch_size, 32, 2000)
        
        x = F.relu(self.conv3(x))
        # x: (batch_size, 64, 2000)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        # x: (batch_size, 64, 1000)
        
        x = x.view(x.size(0), -1)  # Flatten
        # x: (batch_size, 64 * 1000)
        return x

    def forward(self, x1, x2):
        # Process both inputs through CNN layers
        out1 = self.forward_cnn(x1)
        # out1: (batch_size, 64 * 1000)
        out2 = self.forward_cnn(x2)
        # out2: (batch_size, 64 * 1000)
        
        # Concatenate the outputs
        out = torch.cat((out1, out2), dim=1)
        # out: (batch_size, 2 * 64 * 1000)
        
        # Apply Linear layers
        out = F.relu(self.fc1(out))
        # out: (batch_size, 256)
        out = F.relu(self.fc2(out))
        # out: (batch_size, 64)
        out = torch.sigmoid(self.fc3(out))
        # out: (batch_size, 1)
        return out

# Example usage
# Initialize model
model = VoiceVerificationModel()

# Example input tensors (batch_size, channels, length)
x1 = torch.randn(8, 1, 8000)
x2 = torch.randn(8, 1, 8000)

# Forward pass
output = model(x1, x2)
print(output)
