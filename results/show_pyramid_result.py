
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from cmap import Colormap

from gmr.model import GMRMemoryModelDualHead

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def smoothstep(x):
    """
    Cubic smoothstep function that eases in and out.
    
    Parameters:
        x (float): Input value.
        
    Returns:
        float: Output value. For x in [0,1] it returns 3*x^2 - 2*x^3,
               clamps to 0 if x < 0, and to 1 if x > 1.
    """
    if x < 0:
        return 0.0
    elif x > 1:
        return 1.0
    else:
        return 3 * x**2 - 2* x**3

def compute_cumulative_scores(prediction,
                              history_len=60,
                              interval=0.05,
                              initial_score=0,
                              inc_rate=0.005,
                              dec_rate=0.001,
                              transition_window=1.0):
    """
    Compute the cumulative score history for three shapes over a given traceback period.
    
    Parameters:
        prediction (np.array): A [2, 3] array where:
            - prediction[0, :] holds the cumulative time from the last placement/removal event.
            - prediction[1, :] holds the current status (1 for placed, -1 for absent).
        history_len (float): The length of the traceback period in seconds (default is 30).
        interval (float): Time step in seconds (default is 0.05).
        initial_score (float): Initial score for each shape at time -history_len.
        increase_rate (float): Score increment per dt when the shape is placed.
        decrease_rate (float): Score decrement per dt when the shape is absent.
        transition_window (float): Duration in seconds over which to smooth the transition at the event (default is 1.0).
    
    Returns:
        t_values (np.array): Array of time points (from -history_len to 0, not including 0).
        scores (np.array): Array of shape [num_samples, 3] with cumulative scores for triangle, square, and circle.
    """
    num_samples = int(history_len / interval)
    # Generate time stamps from -history_len up to but not including 0.
    timestamps = np.arange(-history_len, 0, interval)

    scores = np.zeros((num_samples, 3))
    scores[0, :] = initial_score  # Set the initial score for all shapes

    if prediction.shape[1] != 3:
        # phase 1, add a dummy column for triangle (first column)
        dummy_column = np.array([[0], [-1]])
        prediction = np.hstack((dummy_column, prediction))

    for i in range(3):
        pred_time = prediction[0, i]
        pred_class = prediction[1, i]

        # Check if the event (change in status) occurred within the traceback period.
        # If so, the event time (relative to t=0) is event_time = -pred_time.
        if pred_time < history_len:
            event_time = -pred_time
        else:
            event_time = None  # No event in the history, status has been constant

        for j in range(1, num_samples):
            t = timestamps[j]
            if event_time is None:
                effective_rate = inc_rate if pred_class == 1 else -dec_rate
            else:
                half_window = transition_window / 2.0
                # Before the transitional window: use the opposite state rate
                if t < event_time - half_window:
                    effective_rate = -dec_rate if pred_class == 1 else inc_rate
                # After the transitional window: use the current state rate
                elif t > event_time + half_window:
                    effective_rate = inc_rate if pred_class == 1 else -dec_rate
                # Within the transition window: smoothly blend the two rates
                else:
                    x = (t - (event_time - half_window)) / transition_window  # Normalize x to [0, 1]
                    weight = smoothstep(x)
                    if pred_class == 1:
                        # Transition from absent update (-dec_rate) to placed update (inc_rate)
                        effective_rate = (-dec_rate) * (1 - weight) + inc_rate * weight
                    else:
                        # Transition from placed update (inc_rate) to absent udpate (dec_rate)
                        effective_rate = inc_rate * (1 - weight) + (-dec_rate) * weight

            scores[j, i] = scores[j - 1, i] + effective_rate

    return timestamps, scores

def plot_3d_cumulative_scores_with_time(timestamps, scores,
                                        size_min=5, size_max=500,
                                        downsample=1):
    """
    Visualize the cumulative score history for three shapes in a 3D scatter plot.
    Marker size and transparency (alpha) increase as the timestamp approaches t=0.
    
    Parameters:
        t_values (np.array): Array of time points (e.g. from -30s to 0s).
        scores (np.array): Array of shape [num_points, 3] where each column corresponds
                           to the cumulative score of triangle, square, and circle.
        size_min (float): Minimum marker size for the oldest timestamp.
        size_max (float): Maximum marker size for the current timestamp (t = 0).
        downsample (int): Plot every nth point to reduce density (default is 1, i.e., no downsampling).
        cmap_name (str): Colormap name from the cmap library (e.g., "magma", "viridis").
        initial_score (float): The starting score for all shapes (used as the origin for the axis lines).
    """

    timestamps = timestamps[::downsample]
    scores = scores[::downsample]

    timestamps_norm = (timestamps + np.abs(timestamps[0])) / np.abs(timestamps[0])
    marker_sizes = size_min + (size_max - size_min) * timestamps_norm

    # cmap = Colormap(cmap_name)
    cmap = Colormap(['#FFFFFF', '#C87271'])
    colors = cmap(timestamps_norm)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=-50, azim=-31, roll=-117)
    # ax.view_init(elev=14, azim=-31, roll=58)
    
    # Plot the cumulative trajectory.
    ax.scatter(scores[:, 0], scores[:, 1], scores[:, 2], s=marker_sizes, c=colors)

    # Determine the origin and the endpoints for the three axis lines.
    # The origin is where all scores are equal to the initial_score.
    axis_len = 7
    origin = np.array([0, 0, 0])
    triangle_end = np.array([axis_len, 0, 0])
    square_end   = np.array([0, axis_len, 0])
    circle_end   = np.array([0, 0, axis_len])

    # Draw the three axis lines (from the origin to each endpoint)
    axis_color = '#81B5D5'
    ax.plot([origin[0], triangle_end[0]], [origin[1], triangle_end[1]], [origin[2], triangle_end[2]],
            color=axis_color, linewidth=5)
    ax.plot([origin[0], square_end[0]],   [origin[1], square_end[1]],   [origin[2], square_end[2]],
            color=axis_color, linewidth=5)
    ax.plot([origin[0], circle_end[0]],   [origin[1], circle_end[1]],   [origin[2], circle_end[2]],
            color=axis_color, linewidth=5)
    
    # Connect the endpoints to form a visual triangle
    # Triangle: triangle_end -> square_end -> circle_end -> triangle_end
    ax.plot([triangle_end[0], square_end[0]], [triangle_end[1], square_end[1]], [triangle_end[2], square_end[2]],
            color='black', linewidth=2, linestyle='--')
    ax.plot([square_end[0], circle_end[0]],     [square_end[1], circle_end[1]],     [square_end[2], circle_end[2]],
            color='black', linewidth=2, linestyle='--')
    ax.plot([circle_end[0], triangle_end[0]],     [circle_end[1], triangle_end[1]],     [circle_end[2], triangle_end[2]],
            color='black', linewidth=2, linestyle='--')

    max_range = axis_len / 2
    ax.set_xlim3d(-max_range, max_range)
    ax.set_ylim3d(-max_range, max_range)
    ax.set_zlim3d(-max_range, max_range)

    # Beautify the plot: remove grid, background panes, and axis lines
    ax.set_facecolor('white')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.line.set_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))

    plt.tight_layout() 

    plt.savefig('example_pyramid.svg', bbox_inches='tight', dpi=600, transparent=True)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMRMemoryModel Visulizer')
    parser.add_argument('-i', '--idx', type=int, default=0, help='Selected index from original dataset for evaluation.')
    parser.add_argument('-l', '--label', type=str, help='Label to select original dataset for evaluation.')
    parser.add_argument('-c', '--cumulation-rate', type=float, default=0.001, help='Cumulation rate for target values')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=1, help='Training phase: 1 for CS-only, 2 for p-tuning')
    parser.add_argument('--phase1-ckpt', type=str, default='phase1.pth', help='Checkpoint path for phase 1')
    parser.add_argument('--phase2-ckpt', type=str, default='phase2.pth', help='Checkpoint path for phase 2')

    args = parser.parse_args()

    idx = args.idx
    label = args.label
    cumulation_rate = args.cumulation_rate
    sample_rate = 20

    report = np.load(os.path.join(__location__, f'{label}_original_prediction.npz'))

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

    with torch.no_grad():
        data_in = report['original_data'][idx, ...]
        data_in = torch.tensor(data_in).to(device)
        data_in = data_in.unsqueeze(0)

        data_out = model(data_in)
        data_out = data_out.squeeze(0).squeeze(-1).cpu().numpy()

    pred_time = np.floor((np.expm1(data_out[0, :]) / cumulation_rate)) / sample_rate
    pred_class = np.sign(data_out[1, :])
    prediction = np.vstack((pred_time, pred_class))

    timestamps, scores = compute_cumulative_scores(prediction, transition_window=10.0)

    plot_3d_cumulative_scores_with_time(timestamps, scores, downsample=50)

    plt.show()