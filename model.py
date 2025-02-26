# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GMRMemoryModel(nn.Module):
    """
    A simple CNN model that takes in a feature tensor of shape [6, 8]
    and outputs a prediction tensor of shape [3, 1].

    The architecture is as follows:
      - Two convolutional layers with ReLU activations and max pooling.
      - A fully connected layer that maps the flattened features to an intermediate vector.
      - A final linear layer to produce a 3-dimensional output,
        which is then reshaped to [3, 1].
    """
    def __init__(self):
        super(GMRMemoryModel, self).__init__()
        # First convolution: output size remains 6x8 with padding.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # output: [16, 3, 4]

        # Second convolution: output ramains 3x4 with padding.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # output: [32, 1, 2]

        # Flatten and pass through fully connect layers.
        self.fc1 = nn.Linear(32 * 1 * 2, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 3)  # final output: [3, 1]

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the GMRMemoryModel.
        
        Parameters:
            x (Tensor): Input tensor of shape [B, 6, 8].
        
        Returns:
            Tensor: Output tensor of shape [B, 3, 1].
        """
        # Add channel dimension
        x = x.unsqueeze(1)          # [B, 1, 6, 8]

        # convolutional block 1
        x = self.conv1(x)           # [B, 16, 6, 8]
        x = F.relu(x)
        x = self.pool1(x)           # [B, 16, 3, 4]

        # convolutional block 2
        x = self.conv2(x)           # [B, 32, 3, 4]
        x = F.relu(x)
        x = self.pool2(x)           # [B, 32, 1, 2]

        # flatten feature maps
        x = x.view(x.size(0), -1)   # [B, 32*1*2]

        # fc
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)             # [B, 3]

        # reshape to output vector
        x = x.unsqueeze(2)          # [B, 3, 1]
        return x

if __name__ == '__main__':
    # test with dummy input
    model = GMRMemoryModel()
    dummy_input = torch.randn(4, 6, 8)  # Example: batch of 4 samples.
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
