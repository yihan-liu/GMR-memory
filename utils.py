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
    'cscs':                     [3.5, 58.75, 96.25, 142.75],
    'csttcs':                   [2.5, 39.75, 56.25, 77.5, 114.75, 150],
    'tsts':                     [4,   40.5, 81,   121.25],
    'ctct':                     [2.5, 41.75, 82,   113],
    'tsst':                     [2.1, 29.9, 55.8, 97.6],
    'tcct':                     [1,   43.5, 89,   119],
    'sccs':                     [4,   40,   70.5, 121.5],
    'tcssct':                   [1.15, 45.05, 79.45, 120, 160.3, 193.2],
    # 'cccccccc':                 [53.95, 58.20, 68.20, 72.25, 79.35, 81.90, 93.75, 97.10],
    # 'ssssssss':                 [11.35, 13.70, 31.20, 34.30, 45.70, 49.25, 75.15, 78.90],
    # 'tttttttttttt':             [183.30, 186.50, 194.65, 198.20, 213.40, 217.50, 227.50, 231.80, 240.95, 246.25, 259.80, 264.65],
    # 'ssccssccssccssccssccss':   [6.70, 9.10, 24.50, 29.45, 36.90, 40.45, 51.35, 55.05, 59.95, 63.30, 74.40, 78.10, 83.50, 87.05, 112.55, 116.70, 119.80, 122.50, 135.30, 138.50, 145.95, 149.35],
    # 'ttssttssttssttssttss':     [14.15, 23.90, 50.70, 57.75, 80.25, 88.55, 131.30, 135.95, 213.80, 221.50, 261.20, 262.80, 301.60, 309.50, 395.75, 402.75, 464.05, 470.60, 501.25, 508.10],
}

def r2(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute the coefficient of determination R^2.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0