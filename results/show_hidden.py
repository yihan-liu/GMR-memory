
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import matplotlib.pyplot as plt

from gmr.model import GMRMemoryModelDualHead

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMRMemoryModel Visulizer')
    parser.add_argument('-i', '--idx', type=int, default=0, help='Selected index from original dataset for evaluation.')
    parser.add_argument('-l', '--label', type=str, help='Label to select original dataset for evaluation.')
    parser.add_argument('-c', '--cumulation-rate', type=float, default=0.001, help='Cumulation rate for target values')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=1, help='Training phase: 1 for CS-only, 2 for p-tuning')
    parser.add_argument('--phase1-ckpt', type=str, default='phase1.pth', help='Checkpoint path for phase 1')
    parser.add_argument('--phase2-ckpt', type=str, default='phase2.pth', help='Checkpoint path for phase 2')

    args = parser.parse_args()

    idx = args.idx
    label = args.label
    cumulation_rate = args.cumulation_rate
    sample_rate = 20

    report = np.load(os.path.join(__location__, f'{label}_original_prediction.npz'))

    if args.phase == 1:
        model = GMRMemoryModelDualHead(output_dim=2)
        if not os.path.exists(args.phase1_ckpt):
            raise FileNotFoundError(f"Phase 1 checkpoint not found at {args.phase1_ckpt}")
        model.load_state_dict(torch.load(args.phase1_ckpt, map_location=device))
    elif args.phase == 2:
        model = GMRMemoryModelDualHead(output_dim=3)
        if not os.path.exists(args.phase2_ckpt):
            raise FileNotFoundError(f"Phase 2 checkpoint not found at {args.phase2_ckpt}")
        model.load_state_dict(torch.load(args.phase2_ckpt, map_location=device))

    model.to(device)
    model.eval()

    fig, axes = plt.subplots(figsize=(8, 8), nrows=4, ncols=4)
    for i, ax in enumerate(axes.flat):
        head = model.conv1.weight[i, ...].squeeze(0).cpu().detach().numpy()
        ax.imshow(head, vmin=-1, vmax=1)
        ax.set_axis_off()
    plt.show()