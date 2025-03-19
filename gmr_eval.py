# gmr_eval.py

import os
import argparse

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from gmr.gmr_memory_dataset import GMRMemoryDataset
from gmr.model import GMRMemoryModelDualHead
from gmr.utils import *

def gmr_predict(args):
    """
    Loads the original dataset for a given label, loads the saved phase 1 model,
    and outputs model predictions for all samples in the dataset.

    Parameters:
      label (str): The label of the dataset (e.g. "cscs", "sccs").
      phase1_ckpt (str): File path to the saved phase 1 model checkpoint.
      root (str): Directory where the dataset file is located.
      num_samples (int): Number of samples to generate from the dataset.
      downsample_factor (int): Factor to downsample the raw data.
      memory_length (int): Memory length used in generating target samples.
      batch_size (int): Batch size for prediction.

    Returns:
      np.array: Array of predictions with shape [num_samples, 2] (for phase 1).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    label = args.label
    dataset = GMRMemoryDataset(label=label, run_augment=False, num_samples=1)
    original_data = torch.tensor(dataset.signal_interp, dtype=torch.float32)
    original_targets = dataset.accumulation_time

    dataloader = DataLoader(original_data, batch_size=64, shuffle=False)

    # Instantiate the base model (phase 1 outputs 2 channels) and load the checkpoint.
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

    pred_times = []
    pred_presences = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            features = batch.to(device)             # Expected shape: [B, 6, 8]
            outputs = model(features)               # Output shape: [B, 2, num_shape, 1]

            pred_time = outputs[:, 0, :, :]
            pred_presence = outputs[:, 1, :, :]
            pred_presence = torch.where(pred_presence < 0, -1, 1)

            pred_times.append(pred_time.squeeze(2).cpu().numpy())
            pred_presences.append(pred_presence.squeeze(2).cpu().numpy())
            predictions.append((pred_time * pred_presence).squeeze(2).cpu().numpy())

    pred_times = np.concatenate(pred_times, axis=0)
    pred_presences = np.concatenate(pred_presences, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    np.savez(f'results/{label}_original_prediction',
             original_data=original_data,
             original_targets=original_targets)

    plt.plot(original_targets)
    # plt.plot(pred_times)
    # plt.plot(pred_presences)
    plt.plot(predictions)
    plt.hlines(0, 0, len(original_data), linestyles='--', colors='k')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMRMemoryModel Evaluation')
    parser.add_argument('-n', '--num-total-samples', type=int, default=20, help='Total number of samples to use across all labels')
    parser.add_argument('-c', '--cumulation-rate', type=float, default=0.001, help='Cumulation rate for target values')
    parser.add_argument('--root', type=str, default='./dataset/', help='Root directory for raw data files')
    parser.add_argument('--downsample-factor', type=int, default=1, help='Downsample factor')
    parser.add_argument('--memory-length', type=int, default=1, help='Number of timesteps in each target sample')

    parser.add_argument('-l', '--label', type=str, help='Label to select original dataset for evaluation.')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=1, help='Training phase: 1 for CS-only, 2 for p-tuning')
    parser.add_argument('--phase1-ckpt', type=str, default='phase1.pth', help='Checkpoint path for phase 1')
    parser.add_argument('--phase2-ckpt', type=str, default='phase2.pth', help='Checkpoint path for phase 2')
    
    args = parser.parse_args()
    gmr_predict(args)