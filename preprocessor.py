# preprocessor.py

import os
import numpy as np

def process_txt_data(label: str, key_frames: list, root: str = "./dataset/raw/part1"):
    """
    Reads a TXT file where the first column represents time in seconds.
    Converts timestamps into corresponding row numbers.
    Generates a dictionary that contains both data and a (t, 3) binary array for triangle, square, and circle.
    Ensures all rows between start and end timestamps are marked as 1.

    Parameters:
        label (str): A string label (e.g., "tsts" for triangle/square start/end). It is also the name for the file.
        key_frames (list): List of timestamps in seconds.
        root: the directory where the file is in.

    Returns:
        dictionary: 
            "x": A (t, 16) array which is the processed version of the raw data
            "y": A (t, 3) array where 1s indicate shape presence.
    """
    file_path = os.path.join(root, label + '.txt')
    data = np.loadtxt(file_path, skiprows=6)
    timestamps = data[:, 0]
    total_time = len(timestamps)
    target = np.zeros((total_time, 3), dtype=int)  # (t, 3)

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

                    # Mark the time range as 1 in the corresponding column
                    target[start_idx:end_idx+1, shape_idx] = 1

    data = data[:, 1:]
    for i in range(data.shape[1]):
        data[:, i] -= data[0, i]
    transformed = np.sign(data) * np.log1p(np.abs(data))

    return {'x': transformed, 'y': target}