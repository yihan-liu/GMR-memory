
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
from cmap import Colormap
import torch

from gmr.gmr_memory_dataset import GMRMemoryDataset
from gmr.utils import *

def gmr_show_sample(args):
    seed = args.seed
    gmr_set_seeds(seed)

    labels = list(KEY_FRAMES_DICT.keys())
    num_labels = len(labels)

    dataset = GMRMemoryDataset(
        label=labels[args.idx],
        root=args.root,
        seed=seed,
        num_samples=1,
        random_low=0.0,
        random_high=0.2,
        run_augment=True
    )
    gaussian_feature_sample = dataset.get_augment_tool_samples().get('gaussian').squeeze(0)
    offset_feature_sample = dataset.get_augment_tool_samples().get('offset').squeeze(0)

    original_feature_samples = torch.load('./results/augmentation_example_datasets/seed-0-original_features.pt', weights_only=False)
    original_feature_sample = original_feature_samples.numpy()[args.idx]

    augmented_feature_sample = original_feature_sample + gaussian_feature_sample + offset_feature_sample

    cmap = Colormap('colorbrewer:blues').to_mpl()
    _, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(original_feature_sample, vmin=-2, vmax=2, cmap=cmap)
    plt.axis('off')
    plt.savefig(f'results/samples/original_feature_samples_l{args.idx}.svg',
                bbox_inches='tight', dpi=600, pad_inches=0, transparent=True)
    # plt.show()

    _, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(augmented_feature_sample, vmin=-2, vmax=2, cmap=cmap)
    plt.axis('off')
    plt.savefig(f'results/samples/augmented_feature_samples_l{args.idx}s{seed}.svg',
                bbox_inches='tight', dpi=600, pad_inches=0, transparent=True)
    # plt.show()
    
    cmap = Colormap('colorbrewer:greys').to_mpl()
    _, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(gaussian_feature_sample, vmin=-2, vmax=2, cmap=cmap)
    plt.axis('off')
    plt.savefig(f'results/samples/gaussian_feature_samples_l{args.idx}s{seed}.svg',
                bbox_inches='tight', dpi=600, pad_inches=0, transparent=True)
    # plt.show()

    _, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(offset_feature_sample, vmin=0, vmax=0.2, cmap=cmap)
    plt.axis('off')
    plt.savefig(f'results/samples/offset_feature_samples_l{args.idx}s{seed}.svg',
                bbox_inches='tight', dpi=600, pad_inches=0, transparent=True)
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMRMemoryModel Evaluation')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Decides the seed to randomly select index from original dataset for evaluation.')
    parser.add_argument('-i', '--idx', type=int, default=0, help='Index of label to select original data from.')
    parser.add_argument('-n', '--num-total-samples', type=int, default=10, help='Total number of samples to use across all labels')
    parser.add_argument('--root', type=str, default='./dataset/', help='Root directory for raw data files')
    parser.add_argument('--auto', action='store_true', help='Automatically loop over a few labels and random seeds.')

    args = parser.parse_args()

    if not args.auto:
        gmr_show_sample(args)
    else:
        for seed in range(5):         # seeds 0 to 4
            for idx in range(3):      # indices 0 to 2
                args.seed = seed
                args.idx = idx
                print(f"Processing seed {seed} and label index {idx}")
                gmr_show_sample(args)