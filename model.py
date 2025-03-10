# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GMRMemoryModel(nn.Module):
    """
    A simple CNN model that takes in a feature tensor of shape [6, 8]
    and outputs a prediction tensor of shape [B, output_dim, 1].

    The architecture is as follows:
      - Two convolutional layers with ReLU activations and max pooling.
      - A fully connected layer that maps the flattened features to an intermediate vector.
      - A final linear layer to produce a [B, output_dim, 1] output.

    NOTE: For phase 1 training (CS-only), output_dim should be set to 2.
    """
    def __init__(self, output_dim=2):
        super(GMRMemoryModel, self).__init__()
        # First convolution: output size remains 6x8 with padding.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # output: [B, 16, 3, 4]

        # Second convolution: output ramains 3x4 with padding.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # output: [B, 32, 1, 2]

        # Flatten and pass through fully connect layers.
        self.fc1 = nn.Linear(32 * 1 * 2, 64)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(64, output_dim)  # final output: [2, 1] for phase 1

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
        x = self.fc2(x)             # [B, output_dim]

        # reshape to output vector
        x = x.unsqueeze(2)          # [B, output_dim, 1]
        return x

class GMRMemoryModelDualHead(nn.Module):
    """
    A dual-head CNN model that takes in a feature tensor of shape [6, 8]
    and outputs two prediction tensors:
      - A binary presence tensor of shape [B, num_shapes, 1] indicating whether each shape is currently present.
      - A regression tensor of shape [B, num_shapes, 1] providing the estimated accumulated time for each shape.

    This model reuses most of the network parameters from GMRMemoryModel:
      - Two convolutional layers with ReLU activations and max pooling.
      - A fully connected layer that maps the flattened features to an intermediate vector.
      - A shared dropout layer for regularization.
    It then splits into two separate output branches:
      - The classification head (using a sigmoid activation) predicts binary presence.
      - The regression head estimates the accumulated time.
    
    NOTE: The parameter num_shapes determines the output dimension for both heads.
    """
    def __init__(self, output_dim=3):
        super(GMRMemoryModelDualHead, self).__init__()
        # First convolution: output size remains 6x8 with padding.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # output: [B, 16, 3, 4]

        # Second convolution: output remains 3x4 with padding.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # output: [B, 32, 1, 2]

        # Flatten and pass through a shared fully connected layer.
        self.fc1 = nn.Linear(32 * 1 * 2, 64)
        self.dropout = nn.Dropout(p=0.1)

        # Dual output heads:
        # Classification head for binary presence detection.
        self.fc_class = nn.Linear(64, output_dim)
        # Regression head for accumulated time estimation.
        self.fc_time = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the GMRMemoryModelDualHead.

        Parameters:
            x (Tensor): Input tensor of shape [B, 6, 8].

        Returns:
            Tuple[Tensor, Tensor]:
                - Binary presence prediction of shape [B, num_shapes, 1].
                - Accumulated time prediction of shape [B, num_shapes, 1].
        """
        # Add channel dimension: [B, 1, 6, 8]
        x = x.unsqueeze(1)

        # Convolutional block 1
        x = self.conv1(x)      # [B, 16, 6, 8]
        x = F.relu(x)
        x = self.pool1(x)      # [B, 16, 3, 4]

        # Convolutional block 2
        x = self.conv2(x)      # [B, 32, 3, 4]
        x = F.relu(x)
        x = self.pool2(x)      # [B, 32, 1, 2]

        # Flatten feature maps
        x = x.view(x.size(0), -1)  # [B, 32*1*2]

        # Shared fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Regression head: estimates accumulated time.
        x_time = self.fc_time(x)            # [B, output_dim]
        x_time = x_time.unsqueeze(2)        # Reshape to [B, output_dim, 1]

        # Classification head: predicts binary presence.
        x_class = self.fc_class(x)          # [B, output_dim]
        x_class = F.sigmoid(x_class)        # Apply sigmoid activation
        x_class = x_class.unsqueeze(2)      # Reshape to [B, output_dim, 1]

        # Stack
        out = torch.stack((x_time, x_class), dim=1)

        return out

class GMRMemoryModelPTuning(nn.Module):
    """
    P-tuning adaptation model. Wraps a pretrained GMRMemoryModel and
    adds a trainable prompt to adapt the model to triangle data.

    The base model parameters are frozen so that only the prompt is updated.
    The prompt is added to the hidden representation after the first FC layer.
    """
    def __init__(self, base_model: GMRMemoryModel):
        super(GMRMemoryModelPTuning, self).__init__()
        self.base_model = base_model

        # Freeze all parameters in the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Create a prompt parameter; its size matches the output of fc1 (i.e. 64).
        self.prompt = nn.Parameter(torch.zeros(64))

        # New fc2 head: from 64 to 3 outputs
        self.fc2_pt = nn.Linear(64, 3)
        with torch.no_grad():
            # copy the pretrained weights from base_model.fc2 (2 x 64) to the last two rows
            # assume the base_model's outputs correspond to [square, circle] in order.
            self.fc2_pt.weight[1:] = self.base_model.fc2.weight.clone()
            self.fc2_pt.bias[1:] = self.base_model.fc2.bias.clone()

    def forward(self, x: torch.Tensor):
        """
        Forward pass with p-tuning. The operations mirror the base model until
        after the first fully connected layer.
        Parameters:
            x (Tensor): With shape [B, 6, 8].
        """
        # Add channel dimension
        x = x.unsqueeze(1)      # [B, 1, 6, 8]

        # Reuse base model layers
        x = self.base_model.conv1(x)
        x = F.relu(x)
        x = self.base_model.pool1(x)
        x = self.base_model.conv2(x)
        x = F.relu(x)
        x = self.base_model.pool2(x)
        x = x.view(x.size(0), -1)   # [B, 64]

        x = self.base_model.fc1(x)
        x = F.relu(x)

        # Add the trainable prompt
        x = x + self.prompt     # NOTE: prompt is the only trainable params
        x = self.base_model.dropout(x)
        x = self.fc2_pt(x)  # [B, 3]
        x = x.unsqueeze(2)  # [B, 3, 1]
        return x
    
class GMRMemoryModelAdapted(nn.Module):
    # TODO: add a Linear layer in front of the GMRMemoryModel to do domain adaptation
    ...

if __name__ == '__main__':
    # test base model (phase 1)
    # base_model = GMRMemoryModel(output_dim=2)
    # dummy_input = torch.randn(4, 6, 8)  # Example: batch of 4 samples.
    # output = base_model(dummy_input)
    # print("GMRMemoryModel (phase 1) input shape:", dummy_input.shape)
    # print("GMRMemoryModel (phase 1) output shape:", output.shape)  # Expect: [4, 2, 1]
    
    base_model = GMRMemoryModelDualHead(output_dim=2)
    dummy_input = torch.randn(16, 6, 8)  # Example: batch of 16 samples.
    output = base_model(dummy_input)
    print("GMRMemoryModelDualHead (phase 1) input shape:", dummy_input.shape)
    print("GMRMemoryModelDualHead (phase 1) output shape:", output.shape)  # Expect: [16, 2, 2, 1]

    # Test the p-tuning model (phase 2).
    # pt_model = GMRMemoryModelPTuning(base_model)
    # output_pt = pt_model(dummy_input)
    # print("GMRMemoryModelPTuning output shape:", output_pt.shape)  # Expect: [4, 3, 1]