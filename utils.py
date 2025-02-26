# utils.py

import numpy as np

KNOWN_CHANNELS = np.array([
    [0, 0], [0, 3],
    [1, 2], [1, 5],
    [2, 0], [2, 3],
    [3, 2], [3, 5],
    [4, 0], [4, 3],
    [5, 2], [5, 5],
    [6, 0], [6, 3],
    [7, 2], [7, 5]
])

KEY_FRAMES_DICT = {
    'cscs':     [3.5,  58.75, 96.25, 142.75],
    'csttcs':   [2.5,  39.75, 56.25, 77.5, 114.75, 150],
    'tsts':     [4,    40.5,  81,    121.25],
    'ctct':     [2.5,  41.75, 82,    113],
    'tsst':     [2.1,  29.9,  55.8,  97.6],
    'tcct':     [1,    43.5,  89,    119],
    'sccs':     [4,    40,    70.5,  121.5],
    'tcssct':   [1.15, 45.05, 79.45, 120,  160.3,  193.2]
}

def r2(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute the coefficient of determination R^2.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0