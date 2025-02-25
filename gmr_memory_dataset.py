# gmr_memory_dataset.py

import os
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import RegularGridInterpolator
import torch
from torch.utils.data import Dataset
from utils import *

class GMRMemoryDataset(Dataset):
    def __init__(self, 
                 label: str, 
                 downsample_factor: int, 
                 num_samples: int, 
                 memory_length: int,
                 cumulation_rate: float=0.001,
                 root: str="./dataset/raw/"):
        """
        Initializes the dataset by loading the raw data, processing it, and generating samples.

        The following attributes are created:
          - self.all_targets: The computed target array (after applying log1p).
          - self.data: Sensor data with the time column removed and normalized.
          - self.data_norm: Sensor data after a signed logarithmic transformation.
          - self.data_interp: Data interpolated onto an 8x6 grid.
          - self.feature_samples: Final feature samples for training.
          - self.target_samples: Final target sequences for training.

        Parameters:
            label (str): Used for both the filename and the keyframe order (e.g., "tsc").
            downsample_factor (int): Factor by which to reduce temporal resolution.
            num_samples (int): Number of random samples to generate.
            memory_length (int): Number of previous timesteps included in each target sequence.
            rate (float): Rate at which target values increase between keyframes.
            root (str): Directory where the raw TXT file is located.
        """
        self.feature_samples, self.target_samples = None, None
        
        self.load_data(label, cumulation_rate, root)
        self.feature_interpolate()
        self.generate_samples(downsample_factor, num_samples, memory_length)

    def __len__(self):
        """Returns the number of samples."""
        return len(self.feature_samples)

    def __getitem__(self, index):
        """
        Retrieves the feature sample and corresponding target sequence at the given index,
        converting them into PyTorch tensors.

        Parameters:
            index (int): Index of the desired sample.

        Returns:
            dictionary: {feature, target}
                - feature: Tensor of shape (6, 8)
                - target: Tensor of shape (memory_length, 3)
        """
        feature = torch.tensor(self.feature_samples[index], dtype=torch.float32)
        target = torch.tensor(self.target_samples[index], dtype=torch.float32)
        return {'feature': feature, 'target': target}

    def load_data(self, 
                  label: str, 
                  rate: float, 
                  root: str):
        """
        Loads and processes the raw data from a TXT file.

        The file is expected to have a time column (first column) followed by sensor readings.
        The method computes a target array based on keyframes (for triangle, square, and circle)
        and applies a logarithmic transformation to the sensor data.

        This method sets the following attributes:
          - self.target, self.data, self.data_norm

        Parameters:
            label (str): Used both as the file name (label + '.txt') and to determine the keyframe order.
            rate (float): Rate for increasing target values.
            root (str): Directory containing the raw data file.
        """
        file_path = os.path.join(root, label + '.txt')
        raw = np.loadtxt(file_path, skiprows=6)  # Assumes first column is time
        timestamps = raw[:, 0]
        total_time = len(timestamps)

        # Initialize target with zeros for three shapes (triangle, square, circle)
        self.all_targets = np.zeros((total_time, 3), dtype=float)
        key_frames = KEY_FRAMES_DICT.get(label)
        if key_frames is None:
            raise ValueError(f"Key frames for label '{label}' not found in KEY_FRAMES_DICT.")

        # Mapping for shape characters to target columns.
        shape_map = {'t': 0, 's': 1, 'c': 2}
        shape_indices = {'t': [], 's': [], 'c': []}

        # Determine order based on characters in label.
        for i, shape in enumerate(label):
            if shape in shape_map:
                shape_indices[shape].append(i)

        # Process start-end pairs for each shape.
        for shape, indices in shape_indices.items():
            shape_idx = shape_map[shape]
            for j in range(0, len(indices), 2):
                if j + 1 < len(indices):
                    start_time = key_frames[indices[j]]
                    end_time = key_frames[indices[j + 1]]
                    start_idx = np.argwhere(timestamps == start_time).flatten()
                    end_idx = np.argwhere(timestamps == end_time).flatten()
                    if start_idx.size and end_idx.size:
                        s = start_idx[0]
                        e = end_idx[0]
                        for k in range(s, e + 1):
                            self.all_targets[k, shape_idx] = (k - s) * rate
        self.all_targets = np.log1p(self.all_targets)

        # Remove the time column and normalize sensor data.
        self.data = raw[:, 1:]
        for i in range(self.data.shape[1]):
            self.data[:, i] -= self.data[0, i]
        # Apply a signed logarithmic transformation.
        self.data_norm = np.sign(self.data) * np.log1p(np.abs(self.data))

    def feature_interpolate(self):
        """
        Interpolates the transformed sensor data onto an 8x6 grid using the Clough-Tocher method.

        This method reads from self.data_norm (set in load_data) and creates an interpolated
        array stored in self.data_interp.
        """
        x = KNOWN_CHANNELS[:, 0]
        y = KNOWN_CHANNELS[:, 1]
        X = np.arange(8)
        Y = np.arange(6)
        X, Y = np.meshgrid(X, Y)
        interp_results = []
        for t in range(self.data_norm.shape[0]):
            interp_func = CloughTocher2DInterpolator(list(zip(x, y)), self.data_norm[t, :])
            Z = interp_func(X, Y)
            Z = np.nan_to_num(Z, nan=0)
            interp_results.append(Z)
        self.data_interp = np.stack(interp_results, axis=0)  # Shape: (total_time, 6, 8)

    @staticmethod
    def feature_augment(feature_sample,
                        mirror_prob=0.5,
                        noise_mean=0.0,
                        noise_std=0.8,
                        random_low=0.0,
                        random_high=1.0):
        """
        Apply data augmentation to a single feature sample via mirroring, noise, and
        a smoothly varying offset. Specifically:
        
        1. **Mirror**:
        - With probability `mirror_prob`, the input is flipped left-to-right.
        - If `feature_sample` is 2D (height x width), `np.fliplr` is used.
        - If `feature_sample` is 3D (e.g., time x height x width),
            it is flipped along `axis=2`.
        
        2. **Gaussian Noise**:
        - After the optional flip, Gaussian noise is added to each element.
        - Noise is drawn from a normal distribution with mean `noise_mean` and
            standard deviation `noise_std`.
        
        3. **Baseline shift**:
        - Two random values (p1 and p2) are sampled from a uniform distribution
            in the range [random_low, random_high].
        - Two additional values (p3, p4) are computed as the average of `p1` and `p2`.
        - These four corner values (p1, p3, p4, p2) define a small “height map”
        for the corners of the grid (shape: [0,6] x [0,8]).
        - The RegularGridInterpolator is then used to interpolate a per-pixel offset
        across the 2D grid.
        - This offset is added to the feature sample, creating a smoothly varying
        deformation.
        
        Parameters:
          - feature_sample (numpy.ndarray): The feature map or volume to be augmented.
          - mirror_prob (float, optional): Probability of applying the horizontal flip. Default is 0.5.
          - noise_mean (float, optional): Mean of the Gaussian noise. Default is 0.0.
          - noise_std (float, optional): Standard deviation of the Gaussian noise. Default is 0.8.
          - random_low (float, optional): Lower bound for the random values used in corner interpolation. Default is 0.0.
          - random_high (float, optional): Upper bound for the random values used in corner interpolation. Default is 1.0.
        
        Returns:
          - numpy.ndarray: The augmented feature sample, with the same shape as the input.
        """
        # Add random mirror
        if np.random.rand() < mirror_prob:
            if feature_sample.ndim == 2:
                return np.fliplr(feature_sample)
            elif feature_sample.ndim == 3:
                return np.flip(feature_sample, axis=2)
            else:
                raise ValueError("Unsupported data dimensions for mirror transformation")
        
        # Add noise
        gauss_noise = np.random.normal(loc=noise_mean, scale=noise_std, size=feature_sample.shape)
        feature_sample = feature_sample + gauss_noise

        # Add base plane offset
        p1 = np.random.uniform(random_low, random_high)
        p2 = np.random.uniform(random_low, random_high)
        p3 = p4 = (p1 + p2) / 2.0
        corner_data = np.array([
            [p1, p3],
            [p4, p2]
        ])
        interp = RegularGridInterpolator(([0, 6], [0, 8]), corner_data, bounds_error=False, fill_value=None)
        xx = np.arange(6)
        yy = np.arange(8)
        X, Y = np.meshgrid(xx, yy, indexing='ij')  # (6,8)
        points = np.stack((X, Y), axis=-1)
        offset = interp(points)
        feature_sample = feature_sample + offset
        return feature_sample

    def generate_samples(self, 
                         downsample_factor: int, 
                         num_samples: int, 
                         memory_length: int):
        """
        Generates random samples from the interpolated features and target array.

        This method downsamples the data along the time axis and then randomly selects timepoints,
        ensuring that each sample's historical target sequence (of length memory_length) contains
        at least one non-zero value.

        It sets the following attributes:
          - self.feature_samples: An array of shape (num_samples, 6, 8).
          - self.target_samples: An array of shape (num_samples, memory_length, 3).

        Parameters:
            downsample_factor (int): Factor by which to downsample the data.
            num_samples (int): Number of random samples to generate.
            memory_length (int): Number of previous timesteps to include in each target sample.
        """
        downsampled_features = self.data_interp[::downsample_factor, ...]
        downsampled_targets = self.all_targets[::downsample_factor, ...]
        total_downsampled = downsampled_features.shape[0]
        if total_downsampled < memory_length:
            raise ValueError("Downsampled dataset length is shorter than the selected memory length.")

        feature_sample_list = []
        target_sample_list = []
        for _ in range(num_samples):
            while True:
                t_random = np.random.randint(memory_length, total_downsampled)
                feature_sample = downsampled_features[t_random, ...]
                target_sample = downsampled_targets[t_random - memory_length:t_random, ...]
                if np.any(target_sample):
                    break

            feature_sample_list.append(feature_sample)
            target_sample_list.append(target_sample)
        self.feature_samples = np.stack(feature_sample_list, axis=0)  # Shape: (num_samples, 6, 8)
        self.target_samples = np.stack(target_sample_list, axis=0)      # Shape: (num_samples, memory_length, 3)

