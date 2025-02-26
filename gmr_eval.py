# gmr_eval.py

import argparse

import matplotlib.pyplot as plt

from gmr_memory_dataset import GMRMemoryDataset
from model import GMRMemoryModel
from utils import *

def show_sample(args):
    labels = list(KEY_FRAMES_DICT.keys())
    print(f'Using labels: {labels}')

    num_labels = len(labels)
    samples_per_label = args.num_total_samples // num_labels
    print(f'Allocating {samples_per_label} samples per label (Total: {samples_per_label * num_labels})')

    datasets = []
    for label in labels:
        dataset = GMRMemoryDataset(
            label=label,
            num_samples=samples_per_label,
            run_augment=True,
            cumulation_rate=args.cumulation_rate,
            root=args.root
        )
        datasets.append(dataset)

    original_feature_samples = datasets[0].get_original_samples().get('features')
    gaussian_feature_samples = datasets[0].get_augment_tool_samples().get('gaussian')
    offset_feature_samples = datasets[0].get_augment_tool_samples().get('offset')

    augmented_feature_samples = datasets[0].feature_samples

    _, axes = plt.subplots(nrows=2, ncols=2)
    im0 = axes[0, 0].imshow(original_feature_samples[0, ...])
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(augmented_feature_samples[0, ...])
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(gaussian_feature_samples[0, ...])
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(offset_feature_samples[0, ...])
    plt.colorbar(im3, ax=axes[1, 1])
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN model with GMRMemoryDataset')
    parser.add_argument('-n', '--num-total-samples', type=int, default=10000, help='Total number of samples to use across all labels')
    parser.add_argument('-c', '--cumulation-rate', type=float, default=0.001, help='Cumulation rate for target values')
    parser.add_argument('--root', type=str, default='./dataset/', help='Root directory for raw data files')
    parser.add_argument('--downsample-factor', type=int, default=1, help='Downsample factor')
    parser.add_argument('--memory-length', type=int, default=1, help='Number of timesteps in each target sample')
    
    args = parser.parse_args()
    show_sample(args)