import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from cmap import Colormap

from gmr.model import GMRMemoryModelDualHead

# Get current directory (assumes the report file is in the same relative location)
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def forward_with_intermediates(model, x):
    """
    Passes the input x through the model layer by layer while saving intermediate activations.
    
    Args:
        model: The GMRMemoryModelDualHead model.
        x (Tensor): Input tensor of shape [6, 8] (or with batch dimension [B,6,8]).
        
    Returns:
        dict: A dictionary mapping layer names to their corresponding activations.
    """
    intermediates = {}
    # Add channel dimension if needed: from [6, 8] to [B, 1, 6, 8]
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 6, 8]
    else:
        x = x.unsqueeze(1)  # assuming x is [B, 6, 8]
    intermediates['input'] = x

    # Convolutional block 1
    x = model.conv1(x)                # [B, 16, 6, 8]
    intermediates['conv1'] = x
    x = F.relu(x)
    intermediates['relu1'] = x
    x = model.pool1(x)                # [B, 16, 3, 4]
    intermediates['pool1'] = x

    # Convolutional block 2
    x = model.conv2(x)                # [B, 32, 3, 4]
    intermediates['conv2'] = x
    x = F.relu(x)
    intermediates['relu2'] = x
    x = model.pool2(x)                # [B, 32, 1, 2]
    intermediates['pool2'] = x

    # Flatten the output
    x = x.view(x.size(0), -1)         # [B, 32*1*2]
    # intermediates['flatten'] = x

    # Fully connected layer
    x = model.fc1(x)                  # [B, 64]
    # intermediates['fc1'] = x
    x = F.relu(x)
    # intermediates['relu_fc1'] = x
    x = model.dropout(x)
    # intermediates['dropout'] = x

    # Regression head
    x_time = model.fc_time(x)         # [B, output_dim]
    # intermediates['fc_time'] = x_time
    x_time = x_time.unsqueeze(2)      # [B, output_dim, 1]
    # intermediates['fc_time_unsqueezed'] = x_time

    # Classification head
    x_class = model.fc_class(x)       # [B, output_dim]
    # intermediates['fc_class'] = x_class
    x_class = F.tanh(x_class)
    # intermediates['tanh_fc_class'] = x_class
    x_class = x_class.unsqueeze(2)     # [B, output_dim, 1]
    # intermediates['fc_class_unsqueezed'] = x_class

    # Final output (stacking both heads)
    final_out = torch.stack((x_time, x_class), dim=1)
    # intermediates['final'] = final_out

    return intermediates

def visualize_activation(layer_name, activation, save_dir='results/layer_activations'):
    """
    Visualizes the activation of a layer.
    
    If the activation is 4D (e.g. [B, C, H, W]), it displays each channel as an image.
    If it is 2D (e.g. [B, features]), it plots a line graph.

    Args:
        layer_name (str): Name of the current layer.
        activation (Tensor): Activation tensor from the layer.
        save_dir (str): Directory where images are saved.
    """
    act_np = activation.cpu().detach().numpy()
    
    if act_np.ndim == 4:
        # For 4D tensors: shape [B, C, H, W].
        # We will stack each channel's activation as a flat image on the xz-plane.
        # For 4D tensors: shape [B, C, H, W]. We'll stack the channels along the y-axis.
        batch, channels, _, _ = act_np.shape
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_proj_type('ortho')

        layer_dir = os.path.join(save_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)

        for i in range(channels):
        #     # Get the i-th channel from the first sample.
        #     img = act_np[0, i, :, :]
        #     H, W = img.shape
        #     # If the grid is too small for a surface plot, replicate rows/columns.
        #     if H < 2:
        #         img = np.repeat(img, 2, axis=0)
        #         H = 2
        #     if W < 2:
        #         img = np.repeat(img, 2, axis=1)
        #         W = 2

        #     # Create meshgrid for width (X) and height (Z) based on the new dimensions.
        #     X, Z = np.meshgrid(np.arange(W + 1), np.arange(H + 1))
        #     Y = np.full((H + 1, W + 1), i)  # Each channel is placed at a unique y-level.

        #     norm = plt.Normalize(vmin=img.min(), vmax=img.max())  # Use a common normalization for the colormap over all channels
        #     color_map = Colormap('colorbrewer:blues')
        #     colors = color_map(norm(img))
        #     # Plot the surface for this channel.
        #     ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors, shade=False)

        # # Turn off all axes, panes, and gridlines
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # ax.xaxis.line.set_color((0, 0, 0, 0))
        # ax.yaxis.line.set_color((0, 0, 0, 0))
        # ax.zaxis.line.set_color((0, 0, 0, 0))
        # ax.xaxis.pane.set_visible(False)
        # ax.yaxis.pane.set_visible(False)
        # ax.zaxis.pane.set_visible(False)
        # ax.grid(False)
        # ax.set_axis_off()

        # plt.tight_layout()
        # plt.savefig(f'results/layer_activations//{layer_name}.png', bbox_inches='tight', dpi=600, transparent=True)
        # plt.show()
        # plt.close()
            img = act_np[0, i, :, :]

            plt.figure(figsize=(4, 4))
            color_map = Colormap('colorbrewer:blues').to_mpl()
            plt.imshow(img, interpolation='nearest', aspect='equal', cmap=color_map)

            plt.axis('off')

            filename = os.path.join(layer_dir, f'{layer_name}_channel_{i}.png')
            plt.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
        
    elif act_np.ndim == 3:
        # For 3D tensors such as [B, features, 1]
        act_np = np.squeeze(act_np, axis=-1)  # now [B, features]
        if act_np.ndim == 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(act_np[0])
            ax.set_title(layer_name)
            plt.show()
        else:
            print(f"Unexpected 3D shape for {layer_name}: {act_np.shape}")
            
    elif act_np.ndim == 2:
        # For 2D tensors: [B, features]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(act_np[0])
        ax.set_title(layer_name)
        plt.show()
    else:
        print(f"Cannot visualize activation of shape {act_np.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMRMemoryModel Visualizer')
    parser.add_argument('-i', '--idx', type=int, default=0,
                        help='Selected index from original dataset for evaluation.')
    parser.add_argument('-l', '--label', type=str,
                        help='Label to select original dataset for evaluation.')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=1,
                        help='Training phase: 1 for CS-only, 2 for p-tuning')
    parser.add_argument('--phase1-ckpt', type=str, default='phase1.pth',
                        help='Checkpoint path for phase 1')
    parser.add_argument('--phase2-ckpt', type=str, default='phase2.pth',
                        help='Checkpoint path for phase 2')
    args = parser.parse_args()

    idx = args.idx
    label = args.label

    # Load the report containing the original data.
    report = np.load(os.path.join(__location__, f'{label}_original_prediction.npz'))
    # Assume that the report contains 'original_data' of shape [num_samples, 6, 8]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_data = report['original_data']
    sample = torch.tensor(original_data[idx], dtype=torch.float32).to(device)
    print(sample.shape)

    # Load the model and checkpoint according to the specified phase.
    if args.phase == 1:
        model = GMRMemoryModelDualHead(output_dim=2)
        if not os.path.exists(args.phase1_ckpt):
            raise FileNotFoundError(f'Phase 1 checkpoint not found at {args.phase1_ckpt}')
        model.load_state_dict(torch.load(args.phase1_ckpt, map_location=device))
    elif args.phase == 2:
        model = GMRMemoryModelDualHead(output_dim=3)
        if not os.path.exists(args.phase2_ckpt):
            raise FileNotFoundError(f'Phase 2 checkpoint not found at {args.phase2_ckpt}')
        model.load_state_dict(torch.load(args.phase2_ckpt, map_location=device))
    model.to(device)
    model.eval()
    print(model)

    # Get intermediate activations for the sample
    intermediates = forward_with_intermediates(model, sample)

    # Visualize the activation from each layer.
    for layer_name, activation in intermediates.items():
        print(f"Visualizing {layer_name} with shape {activation.shape}")
        visualize_activation(layer_name, activation)
