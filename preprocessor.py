# preprocessor.py

import os
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from utils import *

def load_data(label: str, rate: float=0.001, root: str="./dataset/raw/"):
    """
    Reads a TXT file where the first column represents time in seconds.
    Converts timestamps into corresponding row numbers.
    Generates a dictionary that contains both data and a (t, 3) binary array for triangle, square, and circle.
    Ensures all rows between start and end timestamps are marked as 1.

    Parameters:
        label (str): A string label (e.g., "tsts" for triangle/square start/end). It is also the name for the file.
        rate: the rate of increase in the target, with the unit of (frequency)^-1
        root: the directory where the file is in.

    Returns:
        dictionary: 
            "x": A (t, 16) array which is the processed version of the raw data
            "y": A (t, 3) array where 1s indicate shape presence.
    """
    file_path = os.path.join(root, label + '.txt')
    key_frames = KEY_FRAMES_DICT.get(label)
    data = np.loadtxt(file_path, skiprows=6)
    timestamps = data[:, 0]
    total_time = len(timestamps)
    target = np.zeros((total_time, 3), dtype=float)  # (t, 3)

    # Define label mapping and start/end indexes(0=triangle, 1=square, 2=circle)
    shape_map = {'t': 0, 's': 1, 'c': 2}
    shape_indices = {'t': [], 's': [], 'c': []}

    # Step 1: Identify start and end indices for each shape
    for i, shape in enumerate(label):  
        if shape in shape_map:
            shape_indices[shape].append(i)

    # Step 2: Process each shape (triangle, square, circle)
    for shape, indices in shape_indices.items():
        shape_idx = shape_map[shape]  # Get corresponding column index

        # Ensure there are start-end pairs
        for j in range(0, len(indices), 2):  # Step by 2 (start-end pairs)
            if j + 1 < len(indices):  # Ensure end exists
                start_time = key_frames[indices[j]]
                end_time = key_frames[indices[j + 1]]

                start_idx = np.argwhere(timestamps == start_time).flatten()
                end_idx = np.argwhere(timestamps == end_time).flatten()

                if len(start_idx) > 0 and len(end_idx) > 0:
                    start_idx = start_idx[0]
                    end_idx = end_idx[0]

                    for k in range(start_idx, end_idx + 1):
                        target[k, shape_idx] = (k - start_idx) * rate             
    target = np.log1p(target)
    data = data[:, 1:]
    for i in range(data.shape[1]):
        data[:, i] -= data[0, i]
    transformed = np.sign(data) * np.log1p(np.abs(data))
    transformed = interpolate(transformed)

    return {'x': transformed, 'y': target}


def interpolate(data):
    """
    Performs 2D interpolation on the given data using Clough-Tocher interpolation.

    This function takes a dataset where known values are assigned to specific 
    spatial locations (`utils.py`), and interpolates the missing values 
    over a full 8x6 grid using the Clough-Tocher interpolation method. This 
    ensures that the spatial distribution of the data is smooth and completed.

    Parameters:
        data (numpy.ndarray): A 2D NumPy array representing sensor or channel 
                              data over time, where each row corresponds to a 
                              different timestamp.

    Returns:
        numpy.ndarray: An 8x6 interpolated data grid with missing values filled 
                       using the Clough-Tocher interpolation method. Any NaN 
                       values are replaced with 0.
    """
    x = KNOWN_CHANNELS[:,0]
    y = KNOWN_CHANNELS[:,1]
    X = np.arange(8)
    Y = np.arange(6)
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation

    interp_results = []
    for time in range(data.shape[0]):
        interp = CloughTocher2DInterpolator(list(zip(x, y)), data[time,:])
        Z = interp(X, Y)
        Z = np.nan_to_num(Z, nan=0)
        interp_results.append(Z)
    interp_results = np.stack(interp_results)

    return interp_results


def get_samples(dataset: list, downsample_factor: int, num_samples: int, memory_length: int):

    """
    Generates a specified number of randomized samples from the given dataset, where each sample includes a 
    feature snapshot at a randomly chosen timepoint, and the corresponding sequence of historical target labels.

    Parameters:
        dataset (list): 
            A dictionary containing:
                - 'x': numpy array of shape (T, 8, 6), the interpolated features.
                - 'y': numpy array of shape (T, 3), the corresponding target labels indicating the presence of triangle, square, or circle.
        downsample_factor (int): 
            Factor by which the dataset's temporal resolution is reduced. Higher factors reduce data frequency.
        num_samples (int): 
            Number of random samples to generate from the dataset.
        memory_length (int): 
            Number of previous timesteps included in each target sample.

    Returns:
        tuple: A tuple containing:
            - features (numpy.ndarray): Array of shape (num_samples, 8, 6), each entry representing a feature snapshot at a random timepoint.
            - targets (numpy.ndarray): Array of shape (num_samples, memory_length, 3), each entry representing historical target labels leading up to that timepoint.

    Note:
        The function ensures each sample's target history contains at least one positive (non-zero) label.
    """

    all_features = dataset['x']  
    all_targets = dataset['y']
    all_features = all_features[::downsample_factor, :]
    all_targets = all_targets[::downsample_factor, :]
    


    all_feature_length = all_features.shape[0]

    if all_feature_length < memory_length:
        raise ValueError("Downsampled dataset length is shorter than selected memory length.")

    feature_samples = []
    target_samples = []

    for _ in range(num_samples):
        
        while True:
            t_random = np.random.randint(memory_length, all_feature_length)  
            feature_sample = all_features[t_random, ...]
            target_sample = all_targets[t_random - memory_length:t_random, ...]
            if np.any(target_sample):
                break

        feature_samples.append(feature_sample)
        target_samples.append(target_sample)

    features = np.stack(feature_samples, axis=0)  
    targets = np.stack(target_samples, axis=0)

    return features, targets