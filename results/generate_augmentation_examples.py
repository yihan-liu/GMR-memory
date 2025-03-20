
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse

from torch.utils.data import ConcatDataset

from gmr.gmr_memory_dataset import GMRMemoryDataset
from gmr.utils import *

def save_augmentation_example_data(args):
    seed = args.seed

    labels = list(KEY_FRAMES_DICT.keys())
    print(f'Using labels: {labels}')

    num_labels = len(labels)
    samples_per_label = args.num_total_samples // num_labels
    print(f'Allocating {samples_per_label} samples per label (Total: {samples_per_label * num_labels})')

    datasets = []
    original_sample_features = []
    for label in labels:
        dataset = GMRMemoryDataset(
            label=label,
            root=args.root,
            seed=seed,
            num_samples=samples_per_label,
            run_augment=True,
            random_low=0,
            random_high=0.5,
        )
        datasets.append(dataset)
        original_sample_features.append(torch.tensor(dataset.get_original_samples().get('features')))
    combined_dataset = ConcatDataset(datasets)
    original_sample_features = torch.cat(original_sample_features)

    torch.save(combined_dataset, f'./results/augmentation_example_datasets/seed-{seed}.pt')
    torch.save(original_sample_features, f'./results/augmentation_example_datasets/seed-{seed}-original_features.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMRMemoryModel Evaluation')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Decides the seed to randomly select index from original dataset for evaluation.')
    parser.add_argument('-n', '--num-total-samples', type=int, default=10, help='Total number of samples to use across all labels')
    parser.add_argument('--root', type=str, default='./dataset/', help='Root directory for raw data files')

    args = parser.parse_args()
    save_augmentation_example_data(args)