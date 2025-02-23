# preprocessor.py

import copy

import numpy as np

from key_frames import *

def get_feature_and_target(file_path: str,
                           downsample_factor: int,
                           intervals: dict):
    """
    Loads the dataset from file_path and creates a target binary array indicating 
    when each shape is present. 
    
    Parameters:
      file_path: str
          Path to the .npy file containing the feature array.
      intervals: dict
          Mapping from shape index to a list of (start, end) tuples.
          We assume column 0 is triangle, 1 is square, and 2 is circle.
    
    Returns:
      dataset: dict
          Dictionary with keys:
            'all_features': numpy array of sensor readings.
            'all_targets': numpy binary array of shape (total_time, 3) indicating shape presence.
    """
    all_features = np.load(file_path)
    total_time = all_features.shape[-1]
    all_features = all_features.transpose(2, 0, 1)

    all_targets = np.zeros((total_time, 3), dtype=int)
    for shape_idx, pairs in intervals.items():
        for start, end in pairs:
            # Mark the interval [start, end] (inclusive) as 1.
            all_targets[start:end+1, shape_idx] = 1

    all_features = all_features[::downsample_factor, :]
    all_targets = all_targets[::downsample_factor, :]

    dataset = {
        'all_features': all_features,
        'all_targets': all_targets
    }
    return dataset

def get_samples(dataset: dict,
                num_samples: int,
                feature_length: int,
                target_length: int,
                noise_std: float = 0.5):
    """
    Randomly samples data from a dataset and adds Gaussian noise to each feature sample.

    For each sample, a random timestamp (t_random) is chosen such that there is
    enough history to extract both a feature sample and a target sample.
    
    - Feature sample: Taken from dataset['all_features'][t_random - feature_length : t_random, :],
      then transposed to shape [num_channel, feature_length].
    - Target sample: Taken from dataset['all_targets'][t_random - target_length : t_random, :],
      yielding shape [target_length, 3].

    Additionally, Gaussian noise (with standard deviation `noise_std`) is added
    to the extracted feature sample.
    
    Parameters:
      dataset: dict
          Dictionary with keys 'all_features' and 'all_targets' where:
            - features have shape [t, width, height]
            - targets have shape [t, 3]
      num_samples: int
          Number of random samples to extract.
      feature_length: int
          Number of timestamps to include in each feature sample.
      target_length: int
          Number of timestamps to include in each target sample.
      noise_std: float
          Standard deviation of the Gaussian noise to be added. (Default=0.5 -> 更强噪音)

    Returns:
      A dictionary with:
        'features': numpy array of shape [num_samples, num_channel, feature_length]
        'features_noisy': same shape as 'features', but with Gaussian noise
        'features_aug': numpy array of shape [num_samples, feature_length, height, width]
        'features_aug_flat': numpy array of shape [num_samples, num_channel, -1]
        'targets': numpy array of shape [num_samples, target_length, 3]
    """
    t = dataset['all_features'].shape[0]
    width = dataset['all_features'].shape[1]
    height = dataset['all_features'].shape[2]
    num_channel = width * height

    if t < target_length:
        raise ValueError('Dataset length is less than target_length.')
    
    feature_samples = []
    feature_samples_aug = []
    feature_samples_aug_flat = []
    target_samples = []
    target_samples_noisy = []

    for _ in range(num_samples):

        while True:
            t_random = np.random.randint(target_length, t + 1)

            # Extract feature sample and transpose:
            # Original slice shape: [feature_length, width, height]
            # After .T: shape is [height, width, feature_length], 
            # but we usually want [num_channel, feature_length].
            feature_sample = dataset['all_features'][t_random - feature_length : t_random, :].T

            # Extract target sample with shape [target_length, 3]
            target_sample = dataset['all_targets'][t_random - target_length : t_random, :]

            # Only break if the target sample is not all zeros
            if np.any(target_sample):
                break

        feature_sample = np.squeeze(feature_sample)

        # gaussian
        noise = np.random.normal(loc=0.0, scale=noise_std, size=target_sample.shape)
        target_sample_noisy = target_sample + noise

        feature_sample_aug = sample_augment(feature_sample)  # shape: (feature_length, height, width)
        feature_sample_aug_flat = feature_sample_aug.reshape(num_channel, -1)

        feature_samples.append(feature_sample)
        feature_samples_aug.append(feature_sample_aug)
        feature_samples_aug_flat.append(feature_sample_aug_flat)
        target_samples.append(target_sample)
        target_samples_noisy.append(target_sample_noisy)

    return {
        'features': np.stack(feature_samples, axis=0),         
        'features_aug': np.stack(feature_samples_aug, axis=0),
        'features_aug_flat': np.stack(feature_samples_aug_flat, axis=0),
        'targets': np.stack(target_samples, axis=0),
        'target_noisy': np.stack(target_samples_noisy, axis=0) 
    }


def sample_augment(frame: np.ndarray,
                   trans_max: int = 2):
    """
    Apply a random spatial transformation (translation, rotation, and mirroring)
    to the feature sample. The same transformation is applied to all frames in the sample.

    Parameters:
        frame: np.ndarray of shape (height, width)

    Returns:
        Augmented sample_feature of the same shape.
    """
    trans_x = np.random.randint(-trans_max, trans_max)
    trans_y = np.random.randint(-trans_max, trans_max)
    flip_h = np.random.rand() > 0.5  # horizontal flip with 50% probability
    flip_v = np.random.rand() > 0.5  # vertical flip with 50% probability

    augmented_frame = copy.deepcopy(frame)
    augmented_frame = integer_translate(augmented_frame, trans_x, trans_y)
    if flip_h:
        augmented_frame = np.fliplr(augmented_frame)
    if flip_v:
        augmented_frame = np.flipud(augmented_frame)

    return augmented_frame

def integer_translate(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Translates a 2D frame by integer amounts dx and dy without wrapping.
    Empty regions are filled with zeros.
    
    Parameters:
        frame: 2D numpy array of shape (height, width)
        dx: integer translation along the x-axis (columns). Positive shifts right.
        dy: integer translation along the y-axis (rows). Positive shifts down.
    
    Returns:
        Translated 2D frame of the same shape.
    """
    height, width = frame.shape
    # Create an empty frame.
    new_frame = np.zeros_like(frame)
    
    # Determine the source and destination coordinate ranges.
    if dx >= 0:
        src_x_start = 0
        src_x_end = width - dx
        dst_x_start = dx
        dst_x_end = width
    else:
        src_x_start = -dx
        src_x_end = width
        dst_x_start = 0
        dst_x_end = width + dx

    if dy >= 0:
        src_y_start = 0
        src_y_end = height - dy
        dst_y_start = dy
        dst_y_end = height
    else:
        src_y_start = -dy
        src_y_end = height
        dst_y_start = 0
        dst_y_end = height + dy

    new_frame[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = frame[src_y_start:src_y_end, src_x_start:src_x_end]
    return new_frame